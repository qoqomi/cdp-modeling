"""
CDP Questionnaire Section Parser
=================================
CDP 질문지 PDF를 구조화된 JSON으로 변환하는 파서

대상 파일: Full_Corporate_Questionnaire_*.pdf
출력: CDPQuestion 리스트 (질문 스키마)

파이프라인:
1. PDF → 이미지 변환
2. Table Transformer로 표 영역 감지
3. 섹션별 텍스트 추출 (Rationale, Ambition, Requested Content, Response Format)
4. 구조화된 JSON 출력
"""

from __future__ import annotations

import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, TYPE_CHECKING
from enum import Enum

# PDF 처리
import fitz  # PyMuPDF

# Table Detection & Structure Recognition (Optional)
try:
    from transformers import pipeline, AutoModelForObjectDetection, AutoImageProcessor
    from PIL import Image
    import torch
    import io
    HAS_TABLE_DETECTION = True
except ImportError:
    HAS_TABLE_DETECTION = False

if TYPE_CHECKING:
    from PIL import Image

# 스키마 로더 (같은 폴더)
from .schema_loader import CDPSchemaLoader


class FieldType(str, Enum):
    SELECT = "select"
    MULTISELECT = "multiselect"
    TEXT = "text"
    TEXTAREA = "textarea"
    NUMBER = "number"
    PERCENTAGE = "percentage"
    TABLE = "table"
    GROUPED_SELECT = "grouped_select"
    ATTACHMENT = "attachment"


@dataclass
class ResponseColumn:
    """응답 테이블의 컬럼 정의"""

    id: str
    field: str
    field_ko: Optional[str] = None
    type: FieldType = FieldType.TEXT
    options: Optional[List[str]] = None
    grouped_options: Optional[Dict[str, List[str]]] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    required: bool = False
    condition: Optional[Dict[str, Any]] = None


@dataclass
class Tag:
    """질문 태그"""

    authority_type: str = "All requesters"
    environmental_issue: str = "All"
    sector: str = "All"
    question_level: str = "All"


@dataclass
class CDPQuestion:
    """CDP 질문 구조"""

    question_id: str
    title_en: str
    title_ko: Optional[str] = None

    # 메타데이터
    change_from_last_year: str = "No change"
    question_dependencies: Optional[str] = None
    tags: Tag = field(default_factory=Tag)

    # 핵심 콘텐츠
    rationale: Optional[str] = None
    ambition: Optional[List[str]] = None
    requested_content: Optional[List[str]] = None
    explanation_of_terms: Optional[Dict[str, str]] = None
    additional_information: Optional[str] = None

    # 응답 형식 (스키마 기반)
    response_format: Optional[Dict[str, Any]] = None

    # 하위 질문
    children: Optional[List["CDPQuestion"]] = None


class CDPQuestionnaireSectionParser:
    """CDP Questionnaire PDF 섹션 파서

    파싱 대상 섹션:
    - Rationale (질문 배경)
    - Ambition (모범 사례)
    - Requested Content (요청 내용)
    - Response Format (응답 형식)
    """

    # 섹션 패턴 정규식
    PATTERNS = {
        # 2.2 / 2.2.1 / 2.2.2.11 처럼 depth 제한 없이 잡기
        "question_id": r"\((\d+(?:\.\d+)+)\)",
        # 제목은 다음 섹션/메타가 나오기 전까지(줄바꿈 포함) 수집
        "question_title": r"\((\d+(?:\.\d+)+)\)\s*(.+?)(?=Question\s*details|Question\s*dependencies|Change from last\s*year|Rationale|\n{2,}|$)",
        "question_dependencies_section": r"Question\s*dependencies\s*[:\|]?\s*(.+?)(?=Change from last\s*year|Rationale|Ambition|Requested content|Response options|Tags|$)",
        "change_from_last_year": r"Change from last\s*year\s*[:\|]?\s*(.+?)(?=\n|Rationale)",
        "rationale": r"Rationale\s*[:\|]?\s*(.+?)(?=Ambition|Requested\s*content|Response\s*options?|Response\s*format|Tags|$)",
        "ambition": r"Ambition\s*[:\|]?\s*(.+?)(?=Requested\s*content|Response\s*options?|Response\s*format|Tags|$)",
        "requested_content": r"Requested\s*content\s*[:\|]?\s*(.+?)(?=Explanation\s*of\s*terms|Additional\s*information|Response\s*options?|Response\s*format|Tags|$)",
        "explanation_of_terms": r"Explanation\s*of\s*terms\s*[:\|]?\s*(.+?)(?=Additional\s*information|Response\s*options?|Response\s*format|Tags|$)",
        "additional_information": r"Additional\s*information\s*[:\|]?\s*(.+?)(?=Response\s*options?|Response\s*format|Tags|$)",
        "tags_authority": r"Authority Type\s*[:\|]?\s*(.+?)(?=\n|Environmental)",
        "tags_theme": r"Environmental Issue\s*\(Theme\)\s*[:\|]?\s*Question level\s*[:\|]?\s*(.+?)(?=\n|Sector)",
        "tags_sector": r"Sector\s*[:\|]?\s*Question level\s*[:\|]?\s*(.+?)(?=\n|$)",
        "max_characters": r"\[maximum\s*(\d+(?:,\d+)?)\s*characters?\]",
        "select_from": r"Select from:\s*(.+?)(?=\n\n|\[|$)",
        "select_all": r"Select all that apply:\s*(.+?)(?=\n\n|\[|$)",
        "condition": r"This column only appears if\s*(.+?)(?=\.|$)",
        "text_field": r"Text field\s*\[maximum\s*(\d+(?:,\d+)?)\s*characters?\]",
        "numerical_field": r"Numerical field\s*\[.*?(\d+)-(\d+).*?\]",
        "percentage_field": r"Percentage field\s*\[.*?(\d+)-(\d+).*?\]",
    }

    def __init__(
        self,
        pdf_path: str,
        schema_version: str = "2025",
        use_table_detection: bool = False,
        table_detection_threshold: float = 0.7
    ):
        requested_path = Path(pdf_path)
        if requested_path.exists():
            self.pdf_path = requested_path
        else:
            # 편의: 파일명을 주면 backend/data 하위에서 찾기
            backend_dir = Path(__file__).resolve().parents[2]
            data_candidate = backend_dir / "data" / pdf_path
            if data_candidate.exists():
                self.pdf_path = data_candidate
            else:
                raise FileNotFoundError(
                    f"PDF not found: {pdf_path} (also tried: {data_candidate})"
                )

        self.doc = fitz.open(str(self.pdf_path))
        self.full_text = ""
        self.questions: List[CDPQuestion] = []

        # 스키마 로더에서 옵션 로드 (선택적)
        self.schema_loader = None
        self.common_options = {}
        self.grouped_options = {}

        try:
            backend_dir = Path(__file__).resolve().parents[2]
            self.schema_loader = CDPSchemaLoader(
                schema_dir=str(backend_dir / "schemas"),
                version=schema_version,
            )
            self.common_options = self.schema_loader.common_options
            self.grouped_options = self.schema_loader.grouped_options
        except FileNotFoundError:
            print(f"Warning: Schema files not found for version {schema_version}. Using PDF extraction only.")

        # Table Detection & Structure Recognition 설정
        self.use_table_detection = use_table_detection and HAS_TABLE_DETECTION
        self.table_detection_threshold = table_detection_threshold
        self.table_detector = None
        self.structure_model = None
        self.structure_processor = None
        self.detected_tables: Dict[int, List[Dict]] = {}  # page_num -> tables
        self.table_structures: Dict[int, List[Dict]] = {}  # page_num -> structure elements

        if self.use_table_detection:
            self._init_table_detector()
            self._init_structure_recognizer()

    def _init_table_detector(self):
        """Table Transformer 모델 초기화"""
        if not HAS_TABLE_DETECTION:
            print("Warning: transformers/PIL not installed. Table detection disabled.")
            return

        print("Loading Table Transformer detection model...")
        self.table_detector = pipeline(
            "object-detection",
            model="microsoft/table-transformer-detection",
            device=-1  # CPU only
        )
        print("Table Transformer detection model loaded.")

    def _init_structure_recognizer(self):
        """Table Structure Recognition 모델 초기화"""
        if not HAS_TABLE_DETECTION:
            return

        print("Loading Table Structure Recognition model...")
        self.structure_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        self.structure_model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        print("Table Structure Recognition model loaded.")

    def _page_to_image(self, page_num: int, dpi: int = 150) -> Optional[Image.Image]:
        """PDF 페이지를 PIL Image로 변환"""
        if not HAS_TABLE_DETECTION:
            return None

        page = self.doc[page_num]
        # 고해상도 렌더링
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        # PIL Image로 변환
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))

    def detect_tables_in_page(self, page_num: int) -> List[Dict[str, Any]]:
        """페이지에서 표 영역 감지"""
        if not self.table_detector:
            return []

        image = self._page_to_image(page_num)
        if image is None:
            return []

        # Table Transformer로 표 감지
        results = self.table_detector(image)

        # threshold 이상인 결과만 필터링
        tables = []
        for result in results:
            if result["score"] >= self.table_detection_threshold:
                tables.append({
                    "label": result["label"],
                    "score": result["score"],
                    "box": result["box"],  # {"xmin", "ymin", "xmax", "ymax"}
                    "page_num": page_num
                })

        return tables

    def detect_all_tables(self) -> Dict[int, List[Dict]]:
        """모든 페이지에서 표 감지"""
        if not self.use_table_detection:
            return {}

        print(f"Detecting tables in {len(self.doc)} pages...")
        for page_num in range(len(self.doc)):
            tables = self.detect_tables_in_page(page_num)
            if tables:
                self.detected_tables[page_num] = tables
                print(f"  Page {page_num + 1}: Found {len(tables)} table(s)")

        total_tables = sum(len(t) for t in self.detected_tables.values())
        print(f"Total tables detected: {total_tables}")
        return self.detected_tables

    def extract_table_text(self, page_num: int, box: Dict[str, float]) -> str:
        """표 영역에서 텍스트 추출"""
        page = self.doc[page_num]

        # box 좌표를 PDF 좌표로 변환 (이미지 DPI 고려)
        # Note: box는 이미지 좌표이므로 PDF 좌표로 변환 필요
        dpi_scale = 150 / 72  # _page_to_image에서 사용한 DPI
        rect = fitz.Rect(
            box["xmin"] / dpi_scale,
            box["ymin"] / dpi_scale,
            box["xmax"] / dpi_scale,
            box["ymax"] / dpi_scale
        )

        # 해당 영역의 텍스트 추출
        text = page.get_text("text", clip=rect)
        return text.strip()

    def recognize_table_structure(self, page_num: int, table_box: Optional[Dict] = None) -> List[Dict]:
        """테이블 구조 인식 (행, 열, 헤더 감지)

        Args:
            page_num: 페이지 번호
            table_box: 테이블 영역 (None이면 전체 페이지)

        Returns:
            감지된 구조 요소 리스트 (columns, rows, headers)
        """
        if not self.structure_model or not self.structure_processor:
            return []

        image = self._page_to_image(page_num)
        if image is None:
            return []

        # 테이블 영역만 크롭 (선택적)
        if table_box:
            dpi_scale = 150 / 72
            crop_box = (
                int(table_box.get("xmin", 0)),
                int(table_box.get("ymin", 0)),
                int(table_box.get("xmax", image.width)),
                int(table_box.get("ymax", image.height))
            )
            image = image.crop(crop_box)

        # 모델 추론
        inputs = self.structure_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.structure_model(**inputs)

        # 후처리
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.structure_processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes
        )[0]

        # 결과 구조화
        elements = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.structure_model.config.id2label[label.item()]
            box_coords = box.tolist()

            elements.append({
                "label": label_name,
                "score": score.item(),
                "box": {
                    "xmin": box_coords[0],
                    "ymin": box_coords[1],
                    "xmax": box_coords[2],
                    "ymax": box_coords[3]
                },
                "page_num": page_num
            })

        # 라벨별로 정렬 (column headers 우선)
        elements.sort(key=lambda x: (
            0 if "header" in x["label"] else 1,
            x["box"]["ymin"],
            x["box"]["xmin"]
        ))

        return elements

    def extract_column_headers_from_structure(self, page_num: int) -> List[Dict[str, Any]]:
        """구조 인식을 통해 컬럼 헤더 추출

        Returns:
            컬럼 정보 리스트: [{"name": "Time horizon", "box": {...}, "index": 0}, ...]
        """
        elements = self.recognize_table_structure(page_num)
        if not elements:
            return []

        page = self.doc[page_num]
        dpi_scale = 150 / 72

        # 컬럼 헤더 요소 필터링
        headers = [e for e in elements if e["label"] == "table column header"]
        columns = [e for e in elements if e["label"] == "table column"]

        # 컬럼별로 헤더 텍스트 추출
        column_info = []

        # 헤더 영역에서 텍스트 추출
        for header in headers:
            box = header["box"]
            rect = fitz.Rect(
                box["xmin"] / dpi_scale,
                box["ymin"] / dpi_scale,
                box["xmax"] / dpi_scale,
                box["ymax"] / dpi_scale
            )
            header_text = page.get_text("text", clip=rect).strip()

            # 여러 줄로 된 헤더 처리
            lines = [l.strip() for l in header_text.split('\n') if l.strip()]

            # 컬럼별로 분리 (x 좌표 기준)
            for idx, col in enumerate(sorted(columns, key=lambda c: c["box"]["xmin"])):
                col_box = col["box"]

                # 헤더 영역과 컬럼이 겹치는 부분 찾기
                overlap_xmin = max(box["xmin"], col_box["xmin"])
                overlap_xmax = min(box["xmax"], col_box["xmax"])

                if overlap_xmax > overlap_xmin:  # 겹치는 영역이 있음
                    # 해당 컬럼의 헤더 텍스트 추출
                    col_header_rect = fitz.Rect(
                        overlap_xmin / dpi_scale,
                        box["ymin"] / dpi_scale,
                        overlap_xmax / dpi_scale,
                        box["ymax"] / dpi_scale
                    )
                    col_header_text = page.get_text("text", clip=col_header_rect).strip()

                    if col_header_text:
                        # 줄바꿈을 공백으로 대체
                        col_header_text = " ".join(col_header_text.split())

                        column_info.append({
                            "name": col_header_text,
                            "box": col_box,
                            "index": idx
                        })

        # 중복 제거 및 x 좌표로 정렬
        seen_names = set()
        unique_columns = []
        for col in sorted(column_info, key=lambda c: c["box"]["xmin"]):
            if col["name"] not in seen_names:
                seen_names.add(col["name"])
                unique_columns.append(col)

        return unique_columns

    def extract_table_with_structure(self, page_num: int) -> Dict[str, Any]:
        """구조 인식을 사용하여 테이블 전체 추출

        Returns:
            {
                "headers": ["Time horizon", "From (years)", ...],
                "rows": [
                    {"Time horizon": "Short-term", "From (years)": "1", ...},
                    ...
                ],
                "columns_info": [{"name": ..., "type": ..., "options": ...}, ...]
            }
        """
        elements = self.recognize_table_structure(page_num)
        if not elements:
            return {"headers": [], "rows": [], "columns_info": []}

        page = self.doc[page_num]
        dpi_scale = 150 / 72

        # 요소 분류
        header_elements = [e for e in elements if e["label"] == "table column header"]
        column_elements = sorted([e for e in elements if e["label"] == "table column"],
                                  key=lambda c: c["box"]["xmin"])
        row_elements = sorted([e for e in elements if e["label"] == "table row"],
                               key=lambda r: r["box"]["ymin"])

        # 헤더 행 찾기 (첫 번째 행 또는 헤더 요소)
        header_row = None
        if header_elements:
            header_row = header_elements[0]
        elif row_elements:
            header_row = row_elements[0]

        if not header_row:
            return {"headers": [], "rows": [], "columns_info": []}

        # 컬럼 헤더 추출
        headers = []
        columns_info = []

        for col_idx, col in enumerate(column_elements):
            col_box = col["box"]
            header_box = header_row["box"]

            # 헤더 행과 컬럼의 교차점에서 텍스트 추출
            cell_rect = fitz.Rect(
                col_box["xmin"] / dpi_scale,
                header_box["ymin"] / dpi_scale,
                col_box["xmax"] / dpi_scale,
                header_box["ymax"] / dpi_scale
            )
            header_text = page.get_text("text", clip=cell_rect).strip()
            header_text = " ".join(header_text.split())  # 줄바꿈 정리

            # 숫자 인덱스(0, 1, 2...) 건너뛰기
            if header_text.isdigit():
                continue

            headers.append(header_text if header_text else f"Column {col_idx}")
            columns_info.append({
                "name": header_text,
                "index": col_idx,
                "box": col_box
            })

        # 데이터 행 추출
        rows = []
        for row_idx, row in enumerate(row_elements):
            # 헤더 행 건너뛰기
            if header_row and abs(row["box"]["ymin"] - header_row["box"]["ymin"]) < 5:
                continue

            row_data = {}
            row_box = row["box"]

            for col_idx, col in enumerate(column_elements):
                if col_idx >= len(headers):
                    continue

                col_box = col["box"]

                # 행과 컬럼의 교차점에서 텍스트 추출
                cell_rect = fitz.Rect(
                    col_box["xmin"] / dpi_scale,
                    row_box["ymin"] / dpi_scale,
                    col_box["xmax"] / dpi_scale,
                    row_box["ymax"] / dpi_scale
                )
                cell_text = page.get_text("text", clip=cell_rect).strip()
                cell_text = " ".join(cell_text.split())  # 줄바꿈 정리

                row_data[headers[col_idx]] = cell_text

            if row_data:
                rows.append(row_data)

        return {
            "headers": headers,
            "rows": rows,
            "columns_info": columns_info
        }

    def extract_full_text(self) -> str:
        """PDF에서 전체 텍스트 추출"""
        text_parts = []
        for page_num, page in enumerate(self.doc):
            text = page.get_text("text")
            text_parts.append(f"\n--- Page {page_num + 1} ---\n{text}")
        self.full_text = "\n".join(text_parts)
        return self.full_text

    def find_all_questions(self) -> List[str]:
        """모든 질문 ID 찾기"""
        pattern = self.PATTERNS["question_id"]
        matches = re.findall(pattern, self.full_text)
        # 중복 제거 및 정렬
        unique_ids = sorted(
            set(matches), key=lambda x: [int(p) for p in x.split(".")]
        )
        return unique_ids

    def extract_question_block(self, question_id: str) -> str:
        """특정 질문 ID에 해당하는 텍스트 블록 추출"""
        escaped_id = re.escape(f"({question_id})")
        next_q_pattern = r"\(\d+(?:\.\d+)+\)"

        start_match = re.search(escaped_id, self.full_text)
        if not start_match:
            return ""

        start_pos = start_match.start()
        remaining_text = self.full_text[start_match.end():]
        next_match = re.search(next_q_pattern, remaining_text)

        if next_match:
            end_pos = start_match.end() + next_match.start()
        else:
            end_pos = len(self.full_text)

        return self.full_text[start_pos:end_pos]

    def parse_section(self, text: str, pattern_name: str) -> Optional[str]:
        """텍스트에서 특정 섹션 추출"""
        pattern = self.PATTERNS.get(pattern_name)
        if not pattern:
            return None

        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return self._clean_extracted_text(match.group(1))
        return None

    def _clean_extracted_text(self, text: str) -> str:
        """PDF 추출 텍스트 후처리

        - 페이지 마커 제거: --- Page N ---
        - 공백/줄바꿈 정리
        """
        if not text:
            return ""

        # --- Page N --- 같은 마커 제거
        text = re.sub(r"---\s*Page\s*\d+\s*---", "", text, flags=re.IGNORECASE)

        # 여러 줄 공백 정리
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    def parse_bullet_list(self, text: str) -> List[str]:
        """불릿 리스트 파싱 (개선된 버전)

        다양한 bullet 형식 지원:
        - • (bullet point)
        - - (dash)
        - ○ (circle)
        - 숫자 리스트 (1., 2., etc.)
        """
        items = []

        # 텍스트 정규화: 줄바꿈 정리
        text = self._clean_extracted_text(text)

        # 방법 1: 같은 줄에 bullet + 텍스트가 있는 경우
        bullet_pattern = r'(?:^|\n)\s*[•\-○]\s*(.+?)(?=(?:\n\s*[•\-○])|(?:\n\n)|$)'
        matches = re.findall(bullet_pattern, text, re.DOTALL)

        if matches:
            for m in matches:
                # 각 항목 내 줄바꿈을 공백으로 대체하고 정리
                cleaned = re.sub(r'\s+', ' ', m.strip())
                if cleaned and len(cleaned) > 2:
                    items.append(cleaned)

        # 방법 2: bullet 마커가 단독 줄이고, 다음 줄(들)에 본문이 오는 경우
        if not items:
            lines = [l.rstrip() for l in text.splitlines()]
            current: List[str] = []

            def flush():
                nonlocal current
                if not current:
                    return
                joined = " ".join([re.sub(r"\s+", " ", s).strip() for s in current if s.strip()])
                joined = re.sub(r"\s+", " ", joined).strip()
                if joined and len(joined) > 2:
                    items.append(joined)
                current = []

            for raw in lines:
                line = raw.strip()

                # 페이지 마커/빈 줄/섹션 헤더 노이즈 제거
                if not line or re.match(r"^---\s*Page\s*\d+\s*---$", line, re.IGNORECASE):
                    continue
                if re.fullmatch(r"Requested\s*content\s*–?\s*\[?sector\]?", line, re.IGNORECASE):
                    continue
                if re.fullmatch(r"\(if\s*applicable\)", line, re.IGNORECASE):
                    continue

                # 단독 bullet 라인: 새 항목 시작
                if line in {"•", "-", "○"}:
                    flush()
                    continue

                # 소문자 o 서브불릿 (PDF에서 종종 단독으로 찍힘)
                if line == "o":
                    current.append("-")
                    continue

                # bullet 단독이 없었더라도, "• Text" 형태가 들어오면 처리
                if re.match(r"^[•\-○]\s+\S", line):
                    flush()
                    current.append(re.sub(r"^[•\-○]\s*", "", line))
                    continue

                # 일반 텍스트 라인은 현재 항목에 누적
                current.append(line)

            flush()

        # 방법 3: bullet이 없으면 줄 단위로 분리 시도
        if not items:
            lines = text.strip().split('\n')
            for line in lines:
                cleaned = line.strip()
                if re.match(r"^---\s*Page\s*\d+\s*---$", cleaned, re.IGNORECASE):
                    continue
                # 빈 줄이나 너무 짧은 줄 제외
                if cleaned and len(cleaned) > 5:
                    # bullet 마커 제거
                    cleaned = re.sub(r'^[•\-○]\s*', '', cleaned)
                    items.append(cleaned)

        return items

    def parse_tags(self, text: str) -> Tag:
        """태그 정보 파싱"""
        tag = Tag()

        authority = self.parse_section(text, "tags_authority")
        if authority:
            tag.authority_type = authority

        theme = self.parse_section(text, "tags_theme")
        if theme:
            tag.environmental_issue = theme

        sector = self.parse_section(text, "tags_sector")
        if sector:
            tag.sector = sector

        return tag

    def parse_response_columns(
        self, text: str, question_id: str
    ) -> Dict[str, Any]:
        """응답 형식 파싱 (개선된 버전)

        Response Options 섹션에서 테이블 구조 추출:
        - 테이블 형식 (columns 포함)
        - 단일 필드 형식 (textarea, select 등)
        """
        # Response Options/Response format 섹션 찾기
        response_section_pattern = r"(?:Response\s*options?|Response\s*format)\s*[:\|]?\s*(.+?)(?=Rationale|Ambition|Tags|$)"
        response_match = re.search(response_section_pattern, text, re.DOTALL | re.IGNORECASE)

        if not response_match:
            # 섹션이 없으면 기본 형식 반환
            return self._parse_simple_response_format(text, question_id)

        response_text = response_match.group(1)

        # 테이블 형식 감지 (여러 방법으로)
        is_table = self._detect_table_format(response_text)

        if is_table:
            return self._parse_table_format(response_text, question_id)
        else:
            return self._parse_simple_response_format(response_text, question_id)

    def _detect_table_format(self, text: str) -> bool:
        """테이블 형식 감지

        다양한 패턴으로 테이블 구조 감지:
        1. "complete the following table" 문구
        2. 숫자 컬럼 인덱스 (0 1 2 3 4 또는 1 2 3 4 5)
        3. 여러 개의 "Select from:" 패턴
        4. 여러 컬럼 헤더 패턴
        """
        # 방법 1: 명시적 테이블 문구
        if re.search(r"(?:complete|fill)\s+(?:the\s+)?(?:following\s+)?table", text, re.IGNORECASE):
            return True

        # 방법 2: 숫자 컬럼 인덱스 패턴 (0으로 시작 또는 1로 시작)
        # "0 \n 1 \n 2" 또는 "1 \n 2 \n 3" 형태
        lines = text.split('\n')
        consecutive_numbers = 0
        for line in lines:
            stripped = line.strip()
            if stripped.isdigit() and int(stripped) <= 10:
                consecutive_numbers += 1
                if consecutive_numbers >= 3:  # 3개 이상 연속 숫자 = 테이블
                    return True
            elif stripped and not stripped.isdigit():
                consecutive_numbers = 0

        # 방법 3: 여러 개의 "Select from:" 패턴 (2개 이상이면 테이블)
        select_count = len(re.findall(r'Select\s*from:', text, re.IGNORECASE))
        if select_count >= 2:
            return True

        # 방법 4: 여러 개의 컬럼 헤더가 연속으로 나타남
        # "column 1" ... "column 2" 패턴
        column_refs = re.findall(r'\(column\s*\d+\)', text, re.IGNORECASE)
        if len(column_refs) >= 2:
            return True

        return False

    def _parse_table_format(self, text: str, question_id: str) -> Dict[str, Any]:
        """테이블 형식 응답 파싱 (CDP Questionnaire 구조)

        CDP PDF 테이블 구조:
        - 컬럼 번호 행: 0 1 2 3 4
        - 컬럼 헤더 행: Time horizon | From (years) | Is your... | To (years) | ...
        - 데이터 행: Short-term | Numerical field [...] | N/A | ...
        """
        columns = []

        # 방법 1: CDP 테이블 구조 파싱 (컬럼 번호 0, 1, 2...)
        cdp_columns = self._parse_cdp_table_structure(text, question_id)
        if cdp_columns:
            columns = cdp_columns

        # 방법 2: "Column N: name" 패턴
        if not columns:
            column_pattern = r"Column\s*(\d+)\s*[:\-]?\s*([^\n]+?)(?=\n|$)"
            column_matches = re.findall(column_pattern, text)

            for col_num, col_name in column_matches:
                col_name = col_name.strip()
                col_section = self._extract_column_section(text, int(col_num))
                column_info = self._parse_column_info(col_section, question_id, int(col_num), col_name)
                if column_info:
                    columns.append(column_info)

        # 방법 3: Select from 패턴으로 추론
        if not columns:
            columns = self._parse_columns_by_select(text, question_id)

        return {
            "type": "table",
            "columns": [self._response_column_to_dict(c) for c in columns]
        }

    def _parse_cdp_table_structure(self, text: str, question_id: str) -> List[ResponseColumn]:
        """CDP Questionnaire 테이블 구조 파싱

        1. 컬럼 헤더 추출 (알려진 패턴 또는 휴리스틱)
        2. 각 헤더에 대해 필드 타입 추론
        3. 텍스트에서 해당 컬럼의 옵션/제약조건 추출
        """
        columns = []

        # 컬럼 헤더 추출
        column_headers = self._extract_headers_from_text(text)

        if not column_headers:
            # 헤더가 없으면 필드 타입 기반으로 추론
            field_types = self._extract_field_types_from_table(text)
            for idx, field_info in enumerate(field_types):
                field_type = field_info.get('type', FieldType.TEXT)
                field_name = self._infer_field_name(field_type, idx, field_info)
                col = ResponseColumn(
                    id=f"{question_id}.col{idx}",
                    field=field_name,
                    type=field_type,
                    options=field_info.get('options'),
                    max_length=field_info.get('max_length'),
                    min_value=field_info.get('min_value'),
                    max_value=field_info.get('max_value')
                )
                columns.append(col)
            return columns

        # 각 헤더에 대해 필드 정보 추출
        for idx, header in enumerate(column_headers):
            field_info = self._infer_field_info_from_header(header, text)

            col = ResponseColumn(
                id=f"{question_id}.col{idx}",
                field=header,
                type=field_info.get('type', FieldType.TEXT),
                options=field_info.get('options'),
                max_length=field_info.get('max_length'),
                min_value=field_info.get('min_value'),
                max_value=field_info.get('max_value')
            )
            columns.append(col)

        return columns

    # 헤더명 → 스키마 옵션 키 매핑
    # common_options.json의 키를 사용하거나, ADDITIONAL_OPTIONS 키 사용
    HEADER_TO_SCHEMA_KEY = {
        # Yes/No/Plan 유형
        "process in place": "yes_no_plan",
        "identification of priority locations": "yes_no_plan",
        "are the interconnections": "yes_no_plan",
        "have you identified": "yes_no_plan",
        "will you be disclosing": "yes_no_plan",
        "location-specificity used": "yes_no",
        "location-specificity": "yes_no",

        # Dependencies/Impacts 유형
        "dependencies and/or impacts evaluated": "dependencies_impacts",
        "biodiversity impacts evaluated": "biodiversity_impacts_evaluated",

        # Risks/Opportunities 유형
        "risks and/or opportunities evaluated": "risks_opportunities",

        # Primary reason 유형
        "primary reason": "primary_reasons_no_process",

        # Environmental issue
        "environmental issue": "environmental_issues",

        # Coverage & Value chain
        "coverage": "coverage",
        "value chain stages covered": "value_chain_stages_covered",
        "value chain stages where priority": "value_chain_stages_covered",
        "value chain stages": "value_chain_stages",
        "supplier tiers covered": "supplier_tiers",
        "supplier tiers": "supplier_tiers",
        "mining projects covered": "mining_projects_covered",

        # Dependencies/Impacts/Risks/Opportunities covered
        "dependencies/impacts/risks/opportunities covered": "dependencies/impacts/risks/opportunities covered",

        # Assessment
        "type of assessment": "assessment_type",
        "frequency of assessment": "frequency",
        "frequency": "frequency",
        "time horizons covered": "time_horizons",
        "time horizons": "time_horizons",
        "integration of risk management": "integration_level",
        "integration": "integration",

        # Q2.4: Substantive effects
        "effect type": "effect_type",
        "type of definition": "definition_type",
        "change to indicator": "change_to_indicator",
        "% change to indicator": "percent_change",
        "% change": "percent_change",

        # Priority locations
        "types of priority locations": "types_of_priority_locations",

        # Tools
        "tools or methodologies": "tools_methodologies",
        "tools and methods": "tools_methodologies",

        # Likelihood/Magnitude
        "likelihood": "likelihood",
        "magnitude": "magnitude",
    }

    # 스키마 파일에 없는 추가 옵션 (하드코딩)
    ADDITIONAL_OPTIONS = {
        "biodiversity_impacts_evaluated": [
            "Yes, in all cases",
            "Yes, in some cases",
            "No"
        ],
        "dependencies/impacts/risks/opportunities covered": [
            "Dependencies only",
            "Impacts only",
            "Risks only",
            "Opportunities only",
            "Dependencies and impacts",
            "Risks and opportunities",
            "All (dependencies, impacts, risks, and opportunities)"
        ],
        "value_chain_stages_covered": [
            "Direct operations only",
            "Direct operations and upstream",
            "Direct operations and downstream",
            "All stages (direct operations, upstream and downstream)"
        ],
        "mining_projects_covered": [
            "Exploration",
            "Development",
            "Operation",
            "Closure",
            "Post-closure",
            "All stages"
        ],
        "integration_level": [
            "Fully integrated",
            "Partially integrated",
            "Not integrated"
        ],
        "effect_type": ["Risks", "Opportunities"],
        "definition_type": ["Qualitative", "Quantitative"],
        "change_to_indicator": [
            "Absolute decrease",
            "Absolute increase",
            "% decrease",
            "% increase"
        ],
        "percent_change": [
            "Less than 1%",
            "1-10",
            "11-20",
            "21-30",
            "31-40",
            "41-50",
            "51-60",
            "61-70",
            "71-80",
            "81-90",
            "91-99",
            "100%"
        ],
        "types_of_priority_locations": [
            "Sites located in or near protected areas",
            "Sites located in or near key biodiversity areas",
            "Sites with water stress",
            "Sites in proximity to local communities",
            "Other, please specify"
        ],
        "tools_methodologies": [
            "ENCORE",
            "IBAT",
            "SBTN",
            "TNFD LEAP approach",
            "WWF Biodiversity Risk Filter",
            "Other, please specify"
        ],
    }

    def _get_options_from_schema(self, key: str) -> Optional[List[str]]:
        """스키마에서 옵션 가져오기

        1. common_options에서 먼저 찾기
        2. ADDITIONAL_OPTIONS에서 찾기
        """
        # 1. 스키마 로더의 common_options 확인
        if self.common_options and key in self.common_options:
            return self.common_options[key]

        # 2. ADDITIONAL_OPTIONS 확인
        if key in self.ADDITIONAL_OPTIONS:
            return self.ADDITIONAL_OPTIONS[key]

        return None

    def _infer_field_info_from_header(self, header: str, text: str) -> Dict[str, Any]:
        """헤더 이름으로부터 필드 타입 추론

        Args:
            header: 컬럼 헤더 이름
            text: 전체 응답 섹션 텍스트 (옵션 추출용)

        Returns:
            필드 정보 딕셔너리 (type, options, max_length, etc.)
        """
        header_lower = header.lower()
        field_info = {}

        # 0. HEADER_TO_SCHEMA_KEY 매핑으로 스키마 옵션 확인 (최우선)
        # 긴 패턴부터 매칭하여 더 구체적인 패턴 우선 적용
        sorted_patterns = sorted(self.HEADER_TO_SCHEMA_KEY.items(),
                                  key=lambda x: len(x[0]), reverse=True)
        for header_pattern, schema_key in sorted_patterns:
            if header_pattern in header_lower:
                options = self._get_options_from_schema(schema_key)
                if options:
                    field_info['type'] = FieldType.SELECT
                    field_info['options'] = options
                    return field_info

        # 1. 헤더 패턴 기반 추론 (더 구체적인 패턴 먼저!)

        # Is ... open ended? - Yes/No 선택 (more specific, check first)
        if 'open ended' in header_lower or ('is your' in header_lower and '?' in header):
            field_info['type'] = FieldType.SELECT
            field_info['options'] = ['Yes', 'No']
            return field_info

        # From/To (years) - 숫자 입력
        if re.search(r'from\s*\(years?\)', header_lower) or re.search(r'to\s*\(years?\)', header_lower):
            field_info['type'] = FieldType.NUMBER
            # 텍스트에서 숫자 범위 추출
            range_match = re.search(r'Numerical\s*field\s*\[.*?(\d+)\s*[-–]\s*(\d+)', text, re.IGNORECASE)
            if range_match:
                field_info['min_value'] = float(range_match.group(1))
                field_info['max_value'] = float(range_match.group(2))
            else:
                field_info['min_value'] = 0.0
                field_info['max_value'] = 100.0
            return field_info

        # Time horizon - 고정 행 헤더 (row identifier) - check after more specific patterns
        if 'time horizon' in header_lower and 'linked' not in header_lower:
            field_info['type'] = FieldType.SELECT
            field_info['options'] = ['Short-term', 'Medium-term', 'Long-term']
            return field_info

        # How ... linked to ... / Description / Explain - 텍스트 영역
        if any(kw in header_lower for kw in ['how', 'linked', 'description', 'explain', 'planning', 'methodology']):
            field_info['type'] = FieldType.TEXTAREA
            # 텍스트에서 최대 길이 추출
            length_match = re.search(r'Text\s*field\s*\[maximum\s*(\d+(?:,\d+)?)\s*characters?\]', text, re.IGNORECASE)
            if length_match:
                field_info['max_length'] = int(length_match.group(1).replace(',', ''))
            else:
                field_info['max_length'] = 2500
            return field_info

        # Identifier - 텍스트 입력
        if 'identifier' in header_lower:
            field_info['type'] = FieldType.TEXT
            return field_info

        # Process in place, Have you identified, Will you be disclosing, Identification of - Yes/No 선택
        if any(kw in header_lower for kw in ['process in place', 'have you identified', 'will you be disclosing', 'identification of']):
            field_info['type'] = FieldType.SELECT
            # 텍스트에서 옵션 추출 시도
            options = self._extract_options_near_keyword(text, header)
            if options:
                field_info['options'] = options
            else:
                field_info['options'] = ['Yes', 'No']
            return field_info

        # Effect type - 고정 행 (Risks/Opportunities)
        if 'effect type' in header_lower:
            field_info['type'] = FieldType.SELECT
            field_info['options'] = ['Risks', 'Opportunities']
            return field_info

        # Type of definition - 선택
        if 'type of definition' in header_lower:
            field_info['type'] = FieldType.MULTISELECT
            field_info['options'] = ['Qualitative', 'Quantitative']
            return field_info

        # Change to indicator - 선택
        if 'change to indicator' in header_lower:
            field_info['type'] = FieldType.SELECT
            field_info['options'] = ['Absolute decrease', 'Absolute increase', '% decrease', '% increase']
            return field_info

        # % change to indicator - 선택
        if '% change' in header_lower:
            field_info['type'] = FieldType.SELECT
            field_info['options'] = ['Less than 1%', '1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-99', '100%']
            return field_info

        # Absolute increase/decrease figure - 숫자
        if 'absolute' in header_lower and ('increase' in header_lower or 'decrease' in header_lower or 'figure' in header_lower):
            field_info['type'] = FieldType.NUMBER
            return field_info

        # Primary reason - 선택
        if 'primary reason' in header_lower:
            field_info['type'] = FieldType.SELECT
            options = self._extract_options_near_keyword(text, header)
            if options:
                field_info['options'] = options
            return field_info

        # Dependencies/impacts/risks evaluated - 선택
        if any(kw in header_lower for kw in ['evaluated', 'assessment', 'coverage', 'scale', 'frequency']):
            field_info['type'] = FieldType.SELECT
            options = self._extract_options_near_keyword(text, header)
            if options:
                field_info['options'] = options
            return field_info

        # Risk type, Opportunity type, Category, Driver, Impact 등 - 선택
        if any(kw in header_lower for kw in ['type', 'category', 'driver', 'impact', 'issue']):
            field_info['type'] = FieldType.SELECT
            # 텍스트에서 옵션 추출 시도
            options = self._extract_options_near_keyword(text, header)
            if options:
                field_info['options'] = options
            return field_info

        # Where in the value chain - 다중 선택 가능
        if 'value chain' in header_lower or 'which parts' in header_lower:
            field_info['type'] = FieldType.MULTISELECT
            options = self._extract_options_near_keyword(text, header)
            if options:
                field_info['options'] = options
            else:
                field_info['options'] = ['Direct operations', 'Upstream value chain', 'Downstream value chain']
            return field_info

        # Percentage - 퍼센트
        if 'percentage' in header_lower or '%' in header:
            field_info['type'] = FieldType.PERCENTAGE
            field_info['min_value'] = 0.0
            field_info['max_value'] = 100.0
            return field_info

        # Magnitude, Likelihood, Threshold - 숫자 또는 선택
        if any(kw in header_lower for kw in ['magnitude', 'likelihood', 'threshold']):
            field_info['type'] = FieldType.SELECT
            options = self._extract_options_near_keyword(text, header)
            if options:
                field_info['options'] = options
            return field_info

        # 기본값: 텍스트 영역
        field_info['type'] = FieldType.TEXTAREA
        field_info['max_length'] = 2500
        return field_info

    def _extract_options_near_keyword(self, text: str, keyword: str) -> Optional[List[str]]:
        """키워드 근처에서 Select 옵션 추출"""
        # 키워드 위치 찾기
        keyword_match = re.search(re.escape(keyword), text, re.IGNORECASE)
        if not keyword_match:
            return None

        # 키워드 이후 텍스트에서 Select from 패턴 찾기
        after_text = text[keyword_match.end():keyword_match.end() + 500]
        select_match = re.search(r'Select\s*from:\s*\n?((?:•\s*\n?[^\n•]+\n?)+)', after_text, re.DOTALL | re.IGNORECASE)

        if select_match:
            return self._parse_select_options_multiline(select_match.group(1))

        return None

    def _extract_headers_from_text(self, text: str) -> List[str]:
        """텍스트에서 테이블 헤더 추출

        CDP PDF 구조:
        - 컬럼 인덱스: 0, 1, 2, 3, 4 또는 1, 2, 3, 4, 5 (개별 줄)
        - 헤더: 여러 줄에 걸쳐 추출됨
        - 데이터 행: Short-term, Select from 등
        """
        lines = text.strip().split('\n')

        # 컬럼 인덱스 시작/끝 위치 찾기 (0 또는 1로 시작)
        idx_start = -1
        idx_end = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            # 0 또는 1로 시작하는 숫자 인덱스 찾기
            if stripped in ('0', '1') and idx_start == -1:
                idx_start = i
            elif idx_start >= 0 and stripped.isdigit() and int(stripped) <= 10:
                idx_end = i
            elif idx_start >= 0 and not stripped.isdigit() and stripped:
                idx_end = i - 1
                break

        # 데이터 행/Select from 시작 위치 찾기
        data_patterns = [
            'Short-term', 'Medium-term', 'Long-term', 'Short-', 'Medium-', 'Long-',
            'Select from:', 'Select all', 'Numerical field', 'Text field', '[Fixed row]'
        ]
        data_start = -1
        for i, line in enumerate(lines):
            if any(p in line for p in data_patterns):
                data_start = i
                break

        if data_start == -1:
            data_start = len(lines)

        # 헤더 영역 결정
        if idx_start >= 0 and idx_end >= idx_start:
            header_start = idx_end + 1
        else:
            # 인덱스가 없으면 첫 줄부터
            header_start = 0

        header_text = '\n'.join(lines[header_start:data_start])

        # 알려진 CDP 헤더 패턴 매칭
        known_headers = self._match_known_headers(header_text)
        if known_headers:
            return known_headers

        # 전체 텍스트에서도 패턴 매칭 시도
        known_headers = self._match_known_headers(text)
        if known_headers:
            return known_headers

        # 패턴 매칭 실패 시 휴리스틱으로 추출
        return self._extract_headers_heuristic(header_text)

    def _match_known_headers(self, text: str) -> List[str]:
        """알려진 CDP 헤더 패턴 매칭"""
        # 정규화: 줄바꿈을 공백으로
        normalized = ' '.join(text.split())

        # CDP 질문별 알려진 헤더들
        patterns = [
            # ===== Q2.1: Time horizons =====
            (r'Time\s*horizon.*?From\s*\(years\).*?open\s*ended.*?To\s*\(years\).*?linked\s*to',
             ['Time horizon', 'From (years)', 'Is your long-term time horizon open ended?',
              'To (years)', 'How this time horizon is linked to strategic and/or financial planning']),
 
            # ===== Q2.2: Dependencies/impacts process =====
            (r'Process\s*in\s*place.*?Dependencies\s*and/or\s*impacts\s*evaluated.*?Biodiversity\s*impacts.*?Primary\s*reason.*?Explain\s*why',
             ['Process in place', 'Dependencies and/or impacts evaluated in this process',
              'Biodiversity impacts evaluated before the mining project development stage',
              'Primary reason for not evaluating dependencies and/or impacts',
              'Explain why you do not evaluate dependencies and/or impacts']),

            # ===== Q2.2.1: Risks/opportunities process =====
            (r'Process\s*in\s*place.*?Risks\s*and/or\s*opportunities\s*evaluated.*?informed\s*by.*?Primary\s*reason.*?Explain\s*why',
             ['Process in place', 'Risks and/or opportunities evaluated in this process',
              'Is this process informed by the dependencies and/or impacts process?',
              'Primary reason for not evaluating risks and/or opportunities',
              'Explain why you do not evaluate risks and/or opportunities']),

            # ===== Q2.2.2: Process details (complex 12-column table) =====
            (r'Environmental\s*issue.*?dependencies.*?impacts.*?risks.*?opportunities.*?Value\s*chain\s*stages.*?Coverage.*?Supplier\s*tiers',
             ['Environmental issue', 'Dependencies/impacts/risks/opportunities covered',
              'Value chain stages covered', 'Coverage', 'Supplier tiers covered', 'Mining projects covered',
              'Type of assessment', 'Frequency of assessment', 'Time horizons covered',
              'Integration of risk management process', 'Location-specificity used', 'Tools and methods used']),

            # ===== Q2.2.3: Mining-specific biodiversity impacts (8 columns) =====
            (r'Mining\s*project\s*ID.*?Extent\s*of\s*assessment.*?Impacts\s*considered.*?Scope\s*defined\s*by',
             ['Mining project ID', 'Extent of assessment', 'Impacts considered', 'Scope defined by',
              'Aspects considered', 'Baseline biodiversity data available',
              'Environmental Impact Statement publicly available', 'Please explain']),

            # ===== Q2.2.4: Risks process =====
            (r'Identifier.*?Where\s*in\s*the\s*value\s*chain.*?Risk\s*type.*?Primary.*?risk\s*driver',
             ['Identifier', 'Where in the value chain does the risk driver occur?',
              'Risk type', 'Primary environmental risk driver', 'Type of financial impact',
              'Company-specific description', 'Time horizon', 'Likelihood', 'Magnitude of impact',
              'Are you able to provide a potential financial impact figure?', 'Potential financial impact figure',
              'Potential financial impact figure - minimum', 'Potential financial impact figure - maximum',
              'Explanation of financial impact figure', 'Cost of response to risk',
              'Description of response and explanation of cost calculation', 'Comment']),

            # ===== Q2.2.5: Opportunities process =====
            (r'Identifier.*?Where\s*in\s*the\s*value\s*chain.*?Opportunity\s*type.*?Primary.*?opportunity\s*driver',
             ['Identifier', 'Where in the value chain does the opportunity occur?',
              'Opportunity type', 'Primary environmental opportunity driver', 'Type of financial impact',
              'Company-specific description', 'Time horizon', 'Likelihood', 'Magnitude of impact',
              'Are you able to provide a potential financial impact figure?', 'Potential financial impact figure',
              'Potential financial impact figure - minimum', 'Potential financial impact figure - maximum',
              'Explanation of financial impact figure', 'Cost to realize opportunity',
              'Strategy to realize opportunity and explanation of cost calculation', 'Comment']),

            # ===== Q2.2.6: Process details (simpler version) =====
            (r'Environmental\s*issue.*?Coverage.*?Frequency\s*of\s*assessment.*?Time\s*horizons',
             ['Environmental issue', 'Coverage', 'Frequency of assessment',
              'Time horizons covered', 'Tools or methodologies used']),

            # ===== Q2.2.7: Interconnections =====
            (r'Interconnections.*?considered.*?management\s*process',
             ['Are the interconnections between environmental dependencies, impacts, risks, and opportunities considered in your management process?']),

            # ===== Q2.2.9: Environmental information =====
            (r'Environmental\s*information.*?scenario\s*analysis',
             ['Environmental information used in scenario analysis',
              'How the information is used']),

            # ===== Q2.3: Priority locations =====
            (r'Identification\s*of\s*priority\s*locations.*?Value\s*chain\s*stages.*?Types\s*of\s*priority\s*locations.*?Description\s*of\s*process',
             ['Identification of priority locations',
              'Value chain stages where priority locations have been identified',
              'Types of priority locations identified',
              'Description of process to identify priority locations']),

            # ===== Q2.4: Substantive effects =====
            (r'Effect\s*type.*?Type\s*of\s*definition.*?Indicator\s*used.*?substantive.*?Change\s*to\s*indicator',
             ['Effect type', 'Type of definition', 'Indicator used to define substantive effect',
              'Change to indicator', '% change to indicator', 'Absolute increase/decrease figure',
              'Metrics considered in definition', 'Application of definition']),
        ]

        for pattern, headers in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                return headers

        return []

    def _extract_headers_heuristic(self, text: str) -> List[str]:
        """휴리스틱 방식으로 헤더 추출"""
        headers = []

        # 알려진 단어들로 헤더 구분점 찾기
        header_keywords = [
            # Time/Date related
            r'(Time\s*horizon)',
            r'(From\s*\(years?\))',
            r'(To\s*\(years?\))',
            r'(Time\s*frame)',
            # Yes/No questions
            r'(Is\s*your.*?open\s*ended\??)',
            r'(Process\s*in\s*place)',
            r'(Have\s*you\s*identified)',
            # Assessment related
            r'(Dependencies\s*and/or\s*impacts\s*evaluated)',
            r'(Risks\s*and/or\s*opportunities\s*evaluated)',
            r'(Biodiversity\s*impacts\s*evaluated)',
            # Reasons/Explanations
            r'(Primary\s*reason)',
            r'(Explain\s*why)',
            r'(How\s*this.*?planning)',
            r'(Description)',
            # Identifiers/Classifications
            r'(Identifier)',
            r'(Where\s*in.*?chain)',
            r'(Risk\s*type)',
            r'(Opportunity\s*type)',
            r'(Primary.*?driver)',
            r'(Primary.*?impact)',
            r'(Environmental\s*issue)',
            # Coverage/Scale
            r'(Coverage)',
            r'(Scale\s*of\s*assessment)',
            r'(Frequency\s*of\s*assessment)',
            # Metrics
            r'(Magnitude)',
            r'(Likelihood)',
            r'(Threshold)',
            r'(Metric)',
            # Value chain
            r'(Direct\s*operations)',
            r'(Upstream)',
            r'(Downstream)',
            # Disclosure
            r'(Will\s*you\s*be\s*disclosing)',
            r'(Methodology\s*used)',
        ]

        normalized = ' '.join(text.split())

        for pattern in header_keywords:
            match = re.search(pattern, normalized, re.IGNORECASE | re.DOTALL)
            if match:
                # 줄바꿈 정리
                header = ' '.join(match.group(1).split())
                if header not in headers:
                    headers.append(header)

        return headers

    def _infer_field_name(self, field_type: FieldType, idx: int, field_info: Dict) -> str:
        """필드 타입에서 이름 추론"""
        type_names = {
            FieldType.SELECT: "Selection",
            FieldType.MULTISELECT: "Multi-selection",
            FieldType.NUMBER: "Numeric value",
            FieldType.PERCENTAGE: "Percentage",
            FieldType.TEXTAREA: "Description",
            FieldType.TEXT: "Text input"
        }

        base_name = type_names.get(field_type, "Field")

        # 옵션이 있으면 첫 번째 옵션으로 힌트
        if field_info.get('options'):
            opts = field_info['options']
            if len(opts) <= 3:
                return f"{base_name} ({'/'.join(opts[:3])})"

        return f"{base_name} {idx + 1}"

    def _parse_table_headers(self, text: str) -> List[Dict[str, str]]:
        """테이블 헤더 추출"""
        headers = []

        # 첫 번째 줄에서 컬럼 번호 건너뛰고 다음 줄에서 헤더 추출
        lines = text.strip().split('\n')

        # 헤더 후보 줄 찾기 (Time horizon, From (years) 등)
        for line in lines[1:10]:
            line = line.strip()
            # 숫자만 있는 줄이나 너무 짧은 줄 건너뛰기
            if not line or re.match(r'^[\d\s]+$', line) or len(line) < 3:
                continue

            # 필드 타입 지시자가 아닌 줄에서 헤더 추출
            if not re.search(r'(Select from|Numerical field|Text field|N/A)', line, re.IGNORECASE):
                # 공백이나 탭으로 분리된 헤더들
                parts = re.split(r'\s{2,}|\t', line)
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 1:
                        headers.append({'name': part})

            if headers:
                break

        return headers if headers else [{'name': f'Column {i}'} for i in range(5)]

    def _extract_field_types_from_table(self, text: str) -> List[Dict[str, Any]]:
        """테이블에서 필드 타입 추출

        CDP PDF 구조:
        - Select from: \\n• \\nYes \\n• \\nNo  (bullet과 텍스트가 별도 줄)
        - Numerical field [enter a number from 0-100...]
        - Text field [maximum 1,500 characters]
        """
        positions = []

        # 1. Select from 패턴 (bullet과 텍스트가 별도 줄인 경우 처리)
        # "Select from: \n• \nYes \n• \nNo" 형태
        select_pattern = r'Select\s*from:\s*\n?((?:•\s*\n?[^\n•]+\n?)+)'
        for match in re.finditer(select_pattern, text, re.DOTALL | re.IGNORECASE):
            options = self._parse_select_options_multiline(match.group(1))
            if options:
                positions.append((match.start(), {
                    'type': FieldType.SELECT,
                    'options': options
                }))

        # 2. Select all that apply 패턴
        multiselect_pattern = r'Select\s*all\s*that\s*apply:\s*\n?((?:•\s*\n?[^\n•]+\n?)+)'
        for match in re.finditer(multiselect_pattern, text, re.DOTALL | re.IGNORECASE):
            options = self._parse_select_options_multiline(match.group(1))
            if options:
                positions.append((match.start(), {
                    'type': FieldType.MULTISELECT,
                    'options': options
                }))

        # 3. Numerical field 패턴 (다양한 형식)
        num_patterns = [
            r'Numerical\s*field\s*\[.*?(\d+)\s*[-–]\s*(\d+)',  # 0-100
            r'Numerical\s*field\s*\[enter\s*a\s*number\s*from\s*(\d+)\s*[-–]\s*(\d+)',  # enter a number from 0-100
        ]
        for num_pattern in num_patterns:
            for match in re.finditer(num_pattern, text, re.IGNORECASE):
                positions.append((match.start(), {
                    'type': FieldType.NUMBER,
                    'min_value': float(match.group(1)),
                    'max_value': float(match.group(2))
                }))

        # 4. Text field 패턴
        text_pattern = r'Text\s*field\s*\[maximum\s*(\d+(?:,\d+)?)\s*characters?\]'
        for match in re.finditer(text_pattern, text, re.IGNORECASE):
            positions.append((match.start(), {
                'type': FieldType.TEXTAREA,
                'max_length': int(match.group(1).replace(',', ''))
            }))

        # 5. Percentage field 패턴
        pct_patterns = [
            r'Percentage\s*field\s*\[.*?(\d+)\s*[-–]\s*(\d+)',
            r'Percentage\s*field'
        ]
        for pct_pattern in pct_patterns:
            for match in re.finditer(pct_pattern, text, re.IGNORECASE):
                if match.lastindex and match.lastindex >= 2:
                    positions.append((match.start(), {
                        'type': FieldType.PERCENTAGE,
                        'min_value': float(match.group(1)),
                        'max_value': float(match.group(2))
                    }))
                else:
                    positions.append((match.start(), {
                        'type': FieldType.PERCENTAGE
                    }))

        # 중복 제거 (같은 위치에 여러 패턴이 매칭될 수 있음)
        seen_positions = set()
        unique_positions = []
        for pos, info in sorted(positions, key=lambda x: x[0]):
            # 50자 이내에 이미 추출된 필드가 있으면 건너뛰기
            if any(abs(pos - p) < 50 for p in seen_positions):
                continue
            seen_positions.add(pos)
            unique_positions.append((pos, info))

        return [p[1] for p in unique_positions]

    def _parse_select_options_multiline(self, text: str) -> List[str]:
        """Select 옵션 파싱 (bullet과 텍스트가 별도 줄인 경우)

        입력 형식: "• \\nYes \\n• \\nNo" 또는 "• Yes\\n• No"
        """
        options = []

        # 줄바꿈으로 분리
        parts = text.strip().split('\n')

        current_option = ""
        for part in parts:
            part = part.strip()

            # bullet만 있는 줄
            if part in ['•', '-', '○']:
                if current_option:
                    options.append(current_option)
                    current_option = ""
                continue

            # bullet로 시작하는 줄
            if part.startswith(('• ', '- ', '○ ')):
                if current_option:
                    options.append(current_option)
                current_option = part[2:].strip()
            elif part:
                # bullet 없이 텍스트만 있는 줄 (이전 bullet의 내용)
                if current_option:
                    current_option += " " + part
                else:
                    current_option = part

        # 마지막 옵션 추가
        if current_option:
            options.append(current_option)

        return [opt for opt in options if opt and len(opt) > 0]

    def _extract_column_section(self, text: str, col_num: int) -> str:
        """특정 컬럼에 해당하는 텍스트 섹션 추출"""
        # 현재 컬럼 시작 위치 찾기
        current_col_pattern = rf"Column\s*{col_num}\s*[:\-]?"
        current_match = re.search(current_col_pattern, text, re.IGNORECASE)

        if not current_match:
            return ""

        start_pos = current_match.start()

        # 다음 컬럼 시작 위치 찾기
        next_col_pattern = rf"Column\s*{col_num + 1}\s*[:\-]?"
        next_match = re.search(next_col_pattern, text[current_match.end():], re.IGNORECASE)

        if next_match:
            end_pos = current_match.end() + next_match.start()
        else:
            # 다음 컬럼이 없으면 끝까지
            end_pos = len(text)

        return text[start_pos:end_pos]

    def _parse_column_info(
        self, col_text: str, question_id: str, col_num: int, col_name: str
    ) -> Optional[ResponseColumn]:
        """컬럼 정보 파싱"""
        field_type = FieldType.TEXT
        options = None
        grouped_opts = None
        max_length = None
        min_val = None
        max_val = None
        condition = None

        # 1. Select from: 옵션 추출
        select_match = re.search(
            r"Select\s*from:\s*\n?((?:[•\-]\s*.+?\n?)+)",
            col_text, re.DOTALL | re.IGNORECASE
        )
        if select_match:
            field_type = FieldType.SELECT
            options = self._parse_select_options(select_match.group(1))

        # 2. Select all that apply: 다중 선택
        multiselect_match = re.search(
            r"Select\s*all\s*that\s*apply:\s*\n?((?:[•\-]\s*.+?\n?)+)",
            col_text, re.DOTALL | re.IGNORECASE
        )
        if multiselect_match:
            field_type = FieldType.MULTISELECT
            options = self._parse_select_options(multiselect_match.group(1))

        # 3. Text field [maximum N characters]
        text_match = re.search(r"Text\s*field\s*\[maximum\s*(\d+(?:,\d+)?)\s*characters?\]", col_text, re.IGNORECASE)
        if text_match:
            field_type = FieldType.TEXTAREA
            max_length = int(text_match.group(1).replace(",", ""))

        # 4. Numerical field
        num_match = re.search(r"Numerical\s*field\s*\[.*?(\d+)\s*-\s*(\d+)", col_text, re.IGNORECASE)
        if num_match:
            field_type = FieldType.NUMBER
            min_val = float(num_match.group(1))
            max_val = float(num_match.group(2))

        # 5. Percentage field
        pct_match = re.search(r"Percentage\s*field\s*\[.*?(\d+)\s*-\s*(\d+)", col_text, re.IGNORECASE)
        if pct_match:
            field_type = FieldType.PERCENTAGE
            min_val = float(pct_match.group(1))
            max_val = float(pct_match.group(2))

        # 6. 조건부 표시
        cond_match = re.search(r"This\s*column\s*only\s*appears\s*if\s*(.+?)(?:\.|$)", col_text, re.IGNORECASE)
        if cond_match:
            condition = {"appears_if": cond_match.group(1).strip()}

        return ResponseColumn(
            id=f"{question_id}.col{col_num}",
            field=col_name,
            type=field_type,
            options=options,
            grouped_options=grouped_opts,
            max_length=max_length,
            min_value=min_val,
            max_value=max_val,
            condition=condition
        )

    def _parse_select_options(self, options_text: str) -> List[str]:
        """Select 옵션 파싱"""
        options = []
        lines = options_text.strip().split('\n')

        for line in lines:
            # bullet 마커 제거
            cleaned = re.sub(r'^\s*[•\-○]\s*', '', line).strip()
            if cleaned and len(cleaned) > 0:
                options.append(cleaned)

        return options

    def _parse_columns_by_select(self, text: str, question_id: str) -> List[ResponseColumn]:
        """Select from 패턴으로 컬럼 추론"""
        columns = []

        select_pattern = r"Select\s*(?:from|all\s*that\s*apply):\s*\n?((?:[•\-]\s*.+?\n?)+)"
        matches = list(re.finditer(select_pattern, text, re.DOTALL | re.IGNORECASE))

        for idx, match in enumerate(matches):
            is_multi = "all that apply" in match.group(0).lower()
            options = self._parse_select_options(match.group(1))

            col = ResponseColumn(
                id=f"{question_id}.col{idx}",
                field=f"Column {idx}",
                type=FieldType.MULTISELECT if is_multi else FieldType.SELECT,
                options=options
            )
            columns.append(col)

        return columns

    def _parse_simple_response_format(self, text: str, question_id: str) -> Dict[str, Any]:
        """단순 응답 형식 파싱 (테이블 아닌 경우)"""
        # Text field 감지
        text_match = re.search(r"Text\s*field\s*\[maximum\s*(\d+(?:,\d+)?)\s*characters?\]", text, re.IGNORECASE)
        if text_match:
            return {
                "type": "textarea",
                "max_length": int(text_match.group(1).replace(",", ""))
            }

        # Select 감지
        select_match = re.search(r"Select\s*from:\s*\n?((?:[•\-]\s*.+?\n?)+)", text, re.DOTALL | re.IGNORECASE)
        if select_match:
            options = self._parse_select_options(select_match.group(1))
            return {
                "type": "select",
                "options": options
            }

        # 기본값
        return {"type": "textarea", "max_length": 1500}

    def _response_column_to_dict(self, col: ResponseColumn) -> Dict[str, Any]:
        """ResponseColumn을 딕셔너리로 변환"""
        result = {
            "id": col.id,
            "field": col.field,
            "type": col.type.value if isinstance(col.type, Enum) else col.type
        }

        if col.options:
            result["options"] = col.options
        if col.grouped_options:
            result["grouped_options"] = col.grouped_options
        if col.max_length:
            result["max_length"] = col.max_length
        if col.min_value is not None:
            result["min_value"] = col.min_value
        if col.max_value is not None:
            result["max_value"] = col.max_value
        if col.condition:
            result["condition"] = col.condition

        return result

    def parse_conditions(self, text: str) -> Optional[Dict[str, Any]]:
        """조건부 표시 로직 파싱"""
        match = re.search(self.PATTERNS["condition"], text, re.IGNORECASE)
        if match:
            condition_text = match.group(1)
            if "'Yes'" in condition_text or '"Yes"' in condition_text:
                return {"value": "Yes"}
            elif "'No'" in condition_text or '"No"' in condition_text:
                return {"value": "No"}
        return None

    def parse_explanation_of_terms(self, text: str) -> Dict[str, str]:
        """용어 설명 파싱"""
        terms = {}
        pattern = r"[•\-]\s*([^:–]+?)[:–]\s*(.+?)(?=[•\-]|$)"
        matches = re.findall(pattern, text, re.DOTALL)

        for term, definition in matches:
            term = term.strip()
            definition = definition.strip()
            if term and definition:
                terms[term] = definition

        return terms

    def parse_question(self, question_id: str) -> Optional[CDPQuestion]:
        """단일 질문 파싱"""
        block = self.extract_question_block(question_id)
        if not block:
            return None

        title_match = re.search(
            self.PATTERNS["question_title"], block, re.DOTALL | re.IGNORECASE
        )
        title = title_match.group(2).strip() if title_match else ""
        title = " ".join(title.split())

        question = CDPQuestion(
            question_id=question_id,
            title_en=title,
            change_from_last_year=self.parse_section(block, "change_from_last_year")
            or "No change",
            rationale=self.parse_section(block, "rationale"),
            tags=self.parse_tags(block),
        )

        # Ambition 파싱
        ambition_text = self.parse_section(block, "ambition")
        if ambition_text:
            question.ambition = self.parse_bullet_list(ambition_text)
            if not question.ambition:
                question.ambition = [ambition_text]

        # Requested content 파싱
        requested_text = self.parse_section(block, "requested_content")
        if requested_text:
            question.requested_content = self.parse_bullet_list(requested_text)
            if not question.requested_content:
                question.requested_content = [requested_text]

        # 용어 설명 파싱
        terms_text = self.parse_section(block, "explanation_of_terms")
        if terms_text:
            question.explanation_of_terms = self.parse_explanation_of_terms(terms_text)

        # 추가 정보
        question.additional_information = self.parse_section(
            block, "additional_information"
        )

        # 응답 형식 파싱
        question.response_format = self.parse_response_columns(block, question_id)

        # 질문 의존성
        deps_section = self.parse_section(block, "question_dependencies_section")
        if deps_section:
            question.question_dependencies = deps_section

        dep_match = re.search(
            r"This question only appears if\s*(.+?)(?=\.|Change)", block, re.IGNORECASE
        )
        if dep_match:
            question.question_dependencies = dep_match.group(1).strip()

        # 스키마가 있으면 응답 형식은 스키마 우선(일관된 컬럼/조건/옵션)
        if self.schema_loader:
            try:
                resolved = self.schema_loader.get_resolved_schema(question_id, module="2")
            except Exception:
                resolved = None

            if resolved and resolved.get("response_type") == "table":
                cols = []
                for idx, col in enumerate(resolved.get("columns", [])):
                    cols.append(
                        {
                            # 기존 포맷(col0, col1...) 유지 + 스키마/표시 ID 추가
                            "id": f"{question_id}.col{idx}",
                            "schema_id": col.get("id"),
                            "display_id": f"{question_id}.{idx + 1}",
                            "field": col.get("header"),
                            "type": col.get("type"),
                            "options": col.get("options"),
                            "grouped_options": col.get("grouped_options"),
                            "max_length": col.get("max_length"),
                            "min_value": col.get("min_value"),
                            "max_value": col.get("max_value"),
                            "required": col.get("required", False),
                            "condition": col.get("condition"),
                        }
                    )
                question.response_format = {"type": "table", "columns": cols}

        return question

    def parse(self) -> List[CDPQuestion]:
        """모든 질문 파싱 (메인 메서드)"""
        self.extract_full_text()
        question_ids = self.find_all_questions()

        print(f"Found {len(question_ids)} questions: {question_ids}")

        for qid in question_ids:
            question = self.parse_question(qid)
            if question:
                self.questions.append(question)
                print(f"  Parsed: {qid} - {question.title_en[:50]}...")

        # 계층 구조 구축
        self._build_hierarchy()

        return self.questions

    def _build_hierarchy(self):
        """질문 계층 구조 구축"""
        # 재호출 시 중복 append 방지
        for q in self.questions:
            q.children = None

        question_map = {q.question_id: q for q in self.questions}
        root_questions = []

        for q in self.questions:
            parts = q.question_id.split(".")
            if len(parts) == 2:
                root_questions.append(q)
            elif len(parts) > 2:
                parent_id = ".".join(parts[:-1])
                parent = question_map.get(parent_id)
                if parent:
                    if parent.children is None:
                        parent.children = []
                    parent.children.append(q)

        # to_dict()/to_json()에서 상위 질문만 직렬화해 중복(상위+하위) 노출 방지
        self.questions = root_questions

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""

        def convert(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, "__dataclass_fields__"):
                return {k: convert(v) for k, v in asdict(obj).items() if v is not None}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items() if v is not None}
            return obj

        return {
            "version": "2025",
            "module": "2",
            "title": "Dependencies, Impacts, Risks, and Opportunities",
            "questions": [convert(q) for q in self.questions],
        }

    def to_json(self, output_path: Optional[str] = None, indent: int = 2):
        """JSON 파일로 저장"""
        data = self.to_dict()
        # 항상 backend/output 하위에 저장되도록 강제 (실행 위치/입력 경로에 영향받지 않게)
        output_dir = Path(__file__).resolve().parents[2] / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_path:
            requested = Path(output_path)
            filename = requested.name
        else:
            filename = f"{self.pdf_path.stem}_questions_parsed.json"

        filename = filename or "cdp_questions_parsed.json"
        if not filename.lower().endswith(".json"):
            filename = f"{filename}.json"

        final_path = output_dir / filename

        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        print(f"Saved to {final_path}")
        return data
