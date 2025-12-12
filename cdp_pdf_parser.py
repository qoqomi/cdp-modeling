"""
CDP Questionnaire PDF to JSON Parser
=====================================
CDP 질문지 PDF를 구조화된 JSON으로 변환하는 파서

사용 방법:
    python cdp_pdf_parser.py --input data/Full_Corporate_Questionnaire.pdf --output data/cdp_questions.json
"""

import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum

# PDF 처리
import fitz  # PyMuPDF - 텍스트 추출용

# 이미지 처리 (표 감지용)
try:
    from pdf2image import convert_from_path
    from PIL import Image
    import torch
    from transformers import AutoModelForObjectDetection, AutoImageProcessor
    HAS_TABLE_DETECTION = True
except ImportError:
    HAS_TABLE_DETECTION = False
    print("Warning: Table detection libraries not installed. Using text-only parsing.")

# OCR (선택적)
try:
    from paddleocr import PaddleOCR
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


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
    condition: Optional[Dict[str, Any]] = None  # {"field": "2.2.1", "value": "Yes"}


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

    # 응답 형식
    response_format: Optional[List[ResponseColumn]] = None

    # 하위 질문
    children: Optional[List['CDPQuestion']] = None


class CDPPDFParser:
    """CDP PDF 파서 메인 클래스"""

    # 섹션 패턴 정규식
    PATTERNS = {
        'question_id': r'\((\d+\.\d+(?:\.\d+)?)\)',
        'question_title': r'\((\d+\.\d+(?:\.\d+)?)\)\s*(.+?)(?=\n|$)',
        'change_from_last_year': r'Change from last\s*year\s*[:\|]?\s*(.+?)(?=\n|Rationale)',
        'rationale': r'Rationale\s*[:\|]?\s*(.+?)(?=Ambition|Response options|$)',
        'ambition': r'Ambition\s*[:\|]?\s*(.+?)(?=Response options|Requested content|$)',
        'requested_content': r'Requested\s*content\s*[:\|]?\s*(.+?)(?=Explanation of terms|Additional information|Tags|$)',
        'explanation_of_terms': r'Explanation of\s*terms\s*[:\|]?\s*(.+?)(?=Additional information|Tags|$)',
        'additional_information': r'Additional\s*information\s*[:\|]?\s*(.+?)(?=Tags|$)',
        'tags_authority': r'Authority Type\s*[:\|]?\s*(.+?)(?=\n|Environmental)',
        'tags_theme': r'Environmental Issue\s*\(Theme\)\s*[:\|]?\s*Question level\s*[:\|]?\s*(.+?)(?=\n|Sector)',
        'tags_sector': r'Sector\s*[:\|]?\s*Question level\s*[:\|]?\s*(.+?)(?=\n|$)',
        'max_characters': r'\[maximum\s*(\d+(?:,\d+)?)\s*characters?\]',
        'select_from': r'Select from:\s*(.+?)(?=\n\n|\[|$)',
        'select_all': r'Select all that apply:\s*(.+?)(?=\n\n|\[|$)',
        'condition': r'This column only appears if\s*(.+?)(?=\.|$)',
        'text_field': r'Text field\s*\[maximum\s*(\d+(?:,\d+)?)\s*characters?\]',
        'numerical_field': r'Numerical field\s*\[.*?(\d+)-(\d+).*?\]',
        'percentage_field': r'Percentage field\s*\[.*?(\d+)-(\d+).*?\]',
    }

    # 드롭다운 옵션 목록 (자주 사용되는 것들)
    COMMON_OPTIONS = {
        'yes_no': ['Yes', 'No'],
        'yes_no_plan': [
            'Yes',
            'No, but we plan to within the next two years',
            'No, and we do not plan to within the next two years'
        ],
        'time_horizons': ['Short-term', 'Medium-term', 'Long-term'],
        'coverage': ['Full', 'Partial'],
        'assessment_type': ['Qualitative only', 'Quantitative only', 'Qualitative and quantitative'],
        'frequency': [
            'More than once a year',
            'Annually',
            'Every two years',
            'Every three years or more',
            'As important matters arise',
            'Not defined'
        ],
        'environmental_issues': [
            'Climate change',
            'Forests',
            'Water',
            'Plastics',
            'Biodiversity'
        ],
        'value_chain_stages': [
            'Direct operations',
            'Upstream value chain',
            'Downstream value chain',
            'End of life management'
        ],
        'supplier_tiers': [
            'Tier 1 suppliers',
            'Tier 2 suppliers',
            'Tier 3 suppliers',
            'Tier 4+ suppliers'
        ],
        'location_specificity': [
            'Site-specific',
            'Local',
            'Sub-national',
            'National',
            'Not location specific'
        ],
        'integration': [
            'Integrated into multi-disciplinary organization-wide risk management process',
            'A specific environmental risk management process'
        ],
        'primary_reasons': [
            'Lack of internal resources, capabilities, or expertise (e.g., due to organization size)',
            'No standardized procedure',
            'Not an immediate strategic priority',
            'Judged to be unimportant or not relevant',
            'Other, please specify'
        ],
        'industry_sectors': [
            'Apparel',
            'Biotech, health care & pharma',
            'Food, beverage & agriculture',
            'Fossil Fuels',
            'Hospitality',
            'Infrastructure',
            'International bodies',
            'Manufacturing',
            'Materials',
            'Power generation',
            'Retail',
            'Services',
            'Transportation services'
        ],
    }

    # 리스크 유형 (그룹 드롭다운)
    RISK_TYPES = {
        'Policy': [
            'Carbon pricing mechanisms',
            'Changes to international law and bilateral agreements',
            'Changes to national legislation',
            'Increased difficulty in obtaining operations permits',
            'Poor coordination between regulatory bodies',
            'Poor enforcement of environmental regulation',
            'Other policy, please specify'
        ],
        'Technology': [
            'Transition to lower emissions technology and products',
            'Unsuccessful investment in new technologies',
            'Data access/availability or monitoring systems',
            'Other technology, please specify'
        ],
        'Market': [
            'Availability and/or increased cost of certified sustainable material',
            'Availability and/or increased cost of raw materials',
            'Changing customer behavior',
            'Uncertainty in the market signals',
            'Other market, please specify'
        ],
        'Reputation': [
            'Impact on human health',
            'Increased partner and stakeholder concern and partner and stakeholder negative feedback',
            'Stigmatization of sector',
            'Other reputation, please specify'
        ],
        'Acute physical': [
            'Avalanche',
            'Cold wave/frost',
            'Cyclones, hurricanes, typhoons',
            'Drought',
            'Flood (coastal, fluvial, pluvial, ground water)',
            'Heat waves',
            'Heavy precipitation (rain, hail, snow/ice)',
            'Landslide',
            'Storm (including blizzards, dust, and sandstorms)',
            'Tornado',
            'Wildfires',
            'Other acute physical risk, please specify'
        ],
        'Chronic physical': [
            'Change in land-use',
            'Changing precipitation patterns and types (rain, hail, snow/ice)',
            'Changing temperature (air, freshwater, marine water)',
            'Coastal erosion',
            'Declining ecosystem services',
            'Increased severity of extreme weather events',
            'Sea level rise',
            'Soil degradation',
            'Soil erosion',
            'Temperature variability',
            'Water stress',
            'Other chronic physical driver, please specify'
        ],
        'Liability': [
            'Exposure to litigation',
            'Non-compliance with regulations',
            'Other liability, please specify'
        ]
    }

    # 도구 및 방법론 (그룹 드롭다운)
    TOOLS_AND_METHODS = {
        'Enterprise Risk Management': [
            'COSO Enterprise Risk Management Framework',
            'Enterprise Risk Management',
            'ISO 31000 Risk Management Standard',
            'Risk models',
            'Stress tests',
            'Other enterprise risk management, please specify'
        ],
        'International methodologies and standards': [
            'Environmental Impact Assessment',
            'Global Forest Watch',
            'IPCC Climate Change Projections',
            'ISO 14001 Environmental Management Standard',
            'Life Cycle Assessment',
            'Paris Agreement Capital Transition Assessment (PACTA) tool',
            'Other international methodologies and standards, please specify'
        ],
        'Commercially/publicly available tools': [
            'Circulytics',
            'ENCORE tool',
            'IBAT for Business',
            'LEAP (Locate, Evaluate, Assess and Prepare) approach, TNFD',
            'TNFD – Taskforce on Nature-related Financial Disclosures',
            'WRI Aqueduct',
            'WWF Biodiversity Risk Filter',
            'WWF Water Risk Filter',
            'Other commercially/publicly available tools, please specify'
        ],
        'Databases': [
            'Nation-specific databases, tools, or standards',
            'Regional government databases',
            'Other databases, please specify'
        ],
        'Other': [
            'Desk-based research',
            'External consultants',
            'Internal company methods',
            'Materiality assessment',
            'Partner and stakeholder consultation/analysis',
            'Scenario analysis',
            'Other, please specify'
        ]
    }

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.full_text = ""
        self.questions: List[CDPQuestion] = []

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
        pattern = self.PATTERNS['question_id']
        matches = re.findall(pattern, self.full_text)
        # 중복 제거 및 정렬
        unique_ids = sorted(set(matches), key=lambda x: [int(p) for p in x.split('.')])
        return unique_ids

    def extract_question_block(self, question_id: str) -> str:
        """특정 질문 ID에 해당하는 텍스트 블록 추출"""
        # 현재 질문부터 다음 질문 또는 페이지 끝까지
        escaped_id = re.escape(f"({question_id})")

        # 다음 질문 ID 패턴
        next_q_pattern = r'\(\d+\.\d+(?:\.\d+)?\)'

        # 현재 질문 시작 위치 찾기
        start_match = re.search(escaped_id, self.full_text)
        if not start_match:
            return ""

        start_pos = start_match.start()

        # 다음 질문 시작 위치 찾기
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
        # • 또는 - 또는 ○ 로 시작하는 항목들
        pattern = r'[•\-○]\s*(.+?)(?=[•\-○]|\n\n|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        for m in matches:
            cleaned = m.strip()
            if cleaned and len(cleaned) > 2:
                items.append(cleaned)
        return items

    def parse_tags(self, text: str) -> Tag:
        """태그 정보 파싱"""
        tag = Tag()

        authority = self.parse_section(text, 'tags_authority')
        if authority:
            tag.authority_type = authority

        theme = self.parse_section(text, 'tags_theme')
        if theme:
            tag.environmental_issue = theme

        sector = self.parse_section(text, 'tags_sector')
        if sector:
            tag.sector = sector

        return tag

    def parse_response_columns(self, text: str, question_id: str) -> List[ResponseColumn]:
        """응답 컬럼 파싱"""
        columns = []

        # 테이블 헤더 패턴 찾기 (숫자 컬럼)
        header_pattern = r'(\d+)\s+(\d+)\s+(\d+)'

        # Select from 패턴
        select_matches = re.findall(r'Select from:\s*\n?((?:[•\-]\s*.+?\n?)+)', text, re.DOTALL)

        # Text field 패턴
        text_field_matches = re.findall(self.PATTERNS['text_field'], text)

        # 기본 컬럼 구조 추출 시도
        col_idx = 1
        for match in select_matches:
            options = self.parse_bullet_list(match)
            if options:
                col = ResponseColumn(
                    id=f"{question_id}.{col_idx}",
                    field=f"Column {col_idx}",
                    type=FieldType.SELECT,
                    options=options
                )
                columns.append(col)
                col_idx += 1

        for max_chars in text_field_matches:
            max_len = int(max_chars.replace(',', ''))
            col = ResponseColumn(
                id=f"{question_id}.{col_idx}",
                field=f"Column {col_idx}",
                type=FieldType.TEXTAREA,
                max_length=max_len
            )
            columns.append(col)
            col_idx += 1

        return columns

    def parse_conditions(self, text: str) -> Optional[Dict[str, Any]]:
        """조건부 표시 로직 파싱"""
        match = re.search(self.PATTERNS['condition'], text, re.IGNORECASE)
        if match:
            condition_text = match.group(1)
            # 간단한 조건 파싱
            # 예: "you select 'Yes' in column 1"
            if "'Yes'" in condition_text or '"Yes"' in condition_text:
                return {"value": "Yes"}
            elif "'No'" in condition_text or '"No"' in condition_text:
                return {"value": "No"}
        return None

    def parse_explanation_of_terms(self, text: str) -> Dict[str, str]:
        """용어 설명 파싱"""
        terms = {}

        # 패턴: • Term: definition 또는 Term – definition
        pattern = r'[•\-]\s*([^:–]+?)[:–]\s*(.+?)(?=[•\-]|$)'
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

        # 질문 제목 추출
        title_match = re.search(self.PATTERNS['question_title'], block)
        title = title_match.group(2).strip() if title_match else ""

        # 각 섹션 파싱
        question = CDPQuestion(
            question_id=question_id,
            title_en=title,
            change_from_last_year=self.parse_section(block, 'change_from_last_year') or "No change",
            rationale=self.parse_section(block, 'rationale'),
            tags=self.parse_tags(block)
        )

        # Ambition 파싱 (불릿 리스트)
        ambition_text = self.parse_section(block, 'ambition')
        if ambition_text:
            question.ambition = self.parse_bullet_list(ambition_text)
            if not question.ambition:
                question.ambition = [ambition_text]

        # Requested content 파싱
        requested_text = self.parse_section(block, 'requested_content')
        if requested_text:
            question.requested_content = self.parse_bullet_list(requested_text)
            if not question.requested_content:
                question.requested_content = [requested_text]

        # 용어 설명 파싱
        terms_text = self.parse_section(block, 'explanation_of_terms')
        if terms_text:
            question.explanation_of_terms = self.parse_explanation_of_terms(terms_text)

        # 추가 정보
        question.additional_information = self.parse_section(block, 'additional_information')

        # 응답 형식 파싱
        question.response_format = self.parse_response_columns(block, question_id)

        # 질문 의존성
        dep_match = re.search(r'This question only appears if\s*(.+?)(?=\.|Change)', block, re.IGNORECASE)
        if dep_match:
            question.question_dependencies = dep_match.group(1).strip()

        return question

    def parse_all(self) -> List[CDPQuestion]:
        """모든 질문 파싱"""
        self.extract_full_text()
        question_ids = self.find_all_questions()

        print(f"Found {len(question_ids)} questions: {question_ids}")

        for qid in question_ids:
            question = self.parse_question(qid)
            if question:
                self.questions.append(question)
                print(f"  Parsed: {qid} - {question.title_en[:50]}...")

        # 계층 구조 구축 (하위 질문 연결)
        self._build_hierarchy()

        return self.questions

    def _build_hierarchy(self):
        """질문 계층 구조 구축"""
        question_map = {q.question_id: q for q in self.questions}
        root_questions = []

        for q in self.questions:
            parts = q.question_id.split('.')
            if len(parts) == 2:
                # 루트 질문 (예: 2.1, 2.2)
                root_questions.append(q)
            elif len(parts) == 3:
                # 하위 질문 (예: 2.2.1, 2.2.2)
                parent_id = '.'.join(parts[:2])
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
            elif hasattr(obj, '__dataclass_fields__'):
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
            "questions": [convert(q) for q in self.questions]
        }

    def to_json(self, output_path: str, indent: int = 2):
        """JSON 파일로 저장"""
        data = self.to_dict()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        print(f"Saved to {output_path}")
        return data


class TableExtractor:
    """표 추출기 (Table Transformer 사용)"""

    def __init__(self):
        if not HAS_TABLE_DETECTION:
            raise ImportError("Table detection libraries not installed")

        self.processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        self.model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        )

        if HAS_OCR:
            self.ocr = PaddleOCR(lang="en", use_gpu=False)

    def extract_tables_from_page(self, image: Image.Image) -> List[Dict]:
        """페이지에서 표 감지 및 추출"""
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 후처리
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes
        )[0]

        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.tolist()
            tables.append({
                "score": score.item(),
                "label": self.model.config.id2label[label.item()],
                "box": box  # [x1, y1, x2, y2]
            })

        return tables

    def ocr_table_region(self, image: Image.Image, box: List[float]) -> List[List[str]]:
        """표 영역 OCR"""
        if not HAS_OCR:
            return []

        # 표 영역 크롭
        x1, y1, x2, y2 = [int(v) for v in box]
        cropped = image.crop((x1, y1, x2, y2))

        # OCR 실행
        import numpy as np
        result = self.ocr.ocr(np.array(cropped), cls=True)

        # 결과 정리
        texts = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                texts.append(text)

        return texts


def create_rag_chunks(questions: List[CDPQuestion]) -> List[Dict]:
    """RAG용 청크 생성"""
    chunks = []

    for q in questions:
        # 메인 질문 청크
        main_content = f"""
Question ID: {q.question_id}
Title: {q.title_en}

Rationale: {q.rationale or 'N/A'}

Ambition: {' '.join(q.ambition) if q.ambition else 'N/A'}

Requested Content: {' '.join(q.requested_content) if q.requested_content else 'N/A'}
        """.strip()

        chunks.append({
            "id": q.question_id,
            "type": "question",
            "content": main_content,
            "metadata": {
                "question_id": q.question_id,
                "title": q.title_en,
                "tags": asdict(q.tags) if q.tags else {},
                "has_children": bool(q.children)
            }
        })

        # 용어 설명 청크
        if q.explanation_of_terms:
            for term, definition in q.explanation_of_terms.items():
                chunks.append({
                    "id": f"{q.question_id}_term_{term}",
                    "type": "term",
                    "content": f"Term: {term}\nDefinition: {definition}",
                    "metadata": {
                        "question_id": q.question_id,
                        "term": term
                    }
                })

        # 하위 질문 재귀 처리
        if q.children:
            chunks.extend(create_rag_chunks(q.children))

    return chunks


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CDP PDF to JSON Parser")
    parser.add_argument("--input", "-i", required=True, help="Input PDF path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path")
    parser.add_argument("--rag-chunks", "-r", help="Output RAG chunks JSON path")

    args = parser.parse_args()

    # PDF 파싱
    print(f"Parsing {args.input}...")
    parser = CDPPDFParser(args.input)
    questions = parser.parse_all()

    # JSON 저장
    parser.to_json(args.output)

    # RAG 청크 생성 (선택적)
    if args.rag_chunks:
        chunks = create_rag_chunks(questions)
        with open(args.rag_chunks, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(chunks)} RAG chunks to {args.rag_chunks}")

    print(f"\nTotal questions parsed: {len(questions)}")


if __name__ == "__main__":
    main()
