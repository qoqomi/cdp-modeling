"""
CDP Questionnaire Parser
========================
CDP 질문지 PDF를 구조화된 JSON으로 변환하는 파서

대상 파일: Full_Corporate_Questionnaire_*.pdf
출력: CDPQuestion 리스트 (질문 스키마)
"""

import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum

# PDF 처리
import fitz  # PyMuPDF

# 스키마 로더
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from cdp_schema_loader import CDPSchemaLoader


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


class QuestionnaireParser:
    """CDP Questionnaire PDF 파서"""

    # 섹션 패턴 정규식
    PATTERNS = {
        "question_id": r"\((\d+\.\d+(?:\.\d+)?)\)",
        "question_title": r"\((\d+\.\d+(?:\.\d+)?)\)\s*(.+?)(?=\n|$)",
        "change_from_last_year": r"Change from last\s*year\s*[:\|]?\s*(.+?)(?=\n|Rationale)",
        "rationale": r"Rationale\s*[:\|]?\s*(.+?)(?=Ambition|Response options|$)",
        "ambition": r"Ambition\s*[:\|]?\s*(.+?)(?=Response options|Requested content|$)",
        "requested_content": r"Requested\s*content\s*[:\|]?\s*(.+?)(?=Explanation of terms|Additional information|Tags|$)",
        "explanation_of_terms": r"Explanation of\s*terms\s*[:\|]?\s*(.+?)(?=Additional information|Tags|$)",
        "additional_information": r"Additional\s*information\s*[:\|]?\s*(.+?)(?=Tags|$)",
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

    def __init__(self, pdf_path: str, schema_version: str = "2025"):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.doc = fitz.open(pdf_path)
        self.full_text = ""
        self.questions: List[CDPQuestion] = []

        # 스키마 로더에서 옵션 로드
        self.schema_loader = CDPSchemaLoader(version=schema_version)
        self.common_options = self.schema_loader.common_options
        self.grouped_options = self.schema_loader.grouped_options

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
        unique_ids = sorted(set(matches), key=lambda x: [int(p) for p in x.split(".")])
        return unique_ids

    def extract_question_block(self, question_id: str) -> str:
        """특정 질문 ID에 해당하는 텍스트 블록 추출"""
        escaped_id = re.escape(f"({question_id})")
        next_q_pattern = r"\(\d+\.\d+(?:\.\d+)?\)"

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
            return match.group(1).strip()
        return None

    def parse_bullet_list(self, text: str) -> List[str]:
        """불릿 리스트 파싱"""
        items = []
        pattern = r"[•\-○]\s*(.+?)(?=[•\-○]|\n\n|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        for m in matches:
            cleaned = m.strip()
            if cleaned and len(cleaned) > 2:
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
    ) -> List[ResponseColumn]:
        """응답 컬럼 파싱"""
        columns = []

        select_matches = re.findall(
            r"Select from:\s*\n?((?:[•\-]\s*.+?\n?)+)", text, re.DOTALL
        )
        text_field_matches = re.findall(self.PATTERNS["text_field"], text)

        col_idx = 1
        for match in select_matches:
            options = self.parse_bullet_list(match)
            if options:
                col = ResponseColumn(
                    id=f"{question_id}.{col_idx}",
                    field=f"Column {col_idx}",
                    type=FieldType.SELECT,
                    options=options,
                )
                columns.append(col)
                col_idx += 1

        for max_chars in text_field_matches:
            max_len = int(max_chars.replace(",", ""))
            col = ResponseColumn(
                id=f"{question_id}.{col_idx}",
                field=f"Column {col_idx}",
                type=FieldType.TEXTAREA,
                max_length=max_len,
            )
            columns.append(col)
            col_idx += 1

        return columns

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

        title_match = re.search(self.PATTERNS["question_title"], block)
        title = title_match.group(2).strip() if title_match else ""

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
        dep_match = re.search(
            r"This question only appears if\s*(.+?)(?=\.|Change)", block, re.IGNORECASE
        )
        if dep_match:
            question.question_dependencies = dep_match.group(1).strip()

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
        question_map = {q.question_id: q for q in self.questions}
        root_questions = []

        for q in self.questions:
            parts = q.question_id.split(".")
            if len(parts) == 2:
                root_questions.append(q)
            elif len(parts) == 3:
                parent_id = ".".join(parts[:2])
                parent = question_map.get(parent_id)
                if parent:
                    if parent.children is None:
                        parent.children = []
                    parent.children.append(q)

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

    def to_json(self, output_path: str, indent: int = 2):
        """JSON 파일로 저장"""
        data = self.to_dict()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        print(f"Saved to {output_path}")
        return data
