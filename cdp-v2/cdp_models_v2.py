"""
CDP PDF Parser - Data Models (v2)
테이블 구조 지원 추가
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class InputType(Enum):
    SINGLE_SELECT = "singleSelect"
    MULTI_SELECT = "multiSelect"
    GROUPED_MULTI_SELECT = "groupedMultiSelect"
    TEXT = "text"
    TEXTAREA = "textarea"
    TABLE = "table"


class RowType(Enum):
    FIXED = "fixed"
    ADDABLE = "addable"
    NONE = "none"


@dataclass
class ParsedOption:
    """PDF에서 추출한 개별 옵션"""
    label: str
    is_selected: bool
    group: Optional[str] = None


@dataclass
class TableColumn:
    """테이블 컬럼 정의"""
    column_id: str
    header: str
    input_type: str  # "singleSelect", "multiSelect", etc.
    options: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ParsedQuestion:
    """PDF에서 추출한 질문 구조 (v2)"""
    question_id: str
    title: str
    input_type: InputType
    options: List[ParsedOption] = field(default_factory=list)
    text_response: Optional[str] = None
    row_type: RowType = RowType.NONE
    parent_id: Optional[str] = None
    raw_text: str = ""
    
    # 테이블 관련 필드 (v2 추가)
    table_columns: List[Dict[str, Any]] = field(default_factory=list)
    table_responses: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedSection:
    """PDF에서 추출한 섹션"""
    section_id: str
    title: str
    questions: List[ParsedQuestion] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """PDF 전체 파싱 결과"""
    sections: List[ParsedSection] = field(default_factory=list)
    raw_text: str = ""
    page_count: int = 0


# CDP 질문 타입 감지를 위한 패턴 정의
CDP_PATTERNS = {
    "question_id": r"\((\d+(?:\.\d+)*)\)",
    "checkbox_checked": r"☑\s*(.+?)(?=☑|☐|$|\n)",
    "checkbox_unchecked": r"☐\s*(.+?)(?=☑|☐|$|\n)",
    "select_from": r"Select from:",
    "select_all": r"Select all that apply",
    "fixed_row": r"\[Fixed row\]",
    "add_row": r"\[Add row\]",
}

# 그룹화된 옵션의 그룹 헤더들
OPTION_GROUP_HEADERS = [
    "Acute physical",
    "Chronic physical",
    "Policy",
    "Market",
    "Reputation",
    "Technology",
    "Liability",
    "Enterprise Risk Management",
    "International methodologies and standards",
    "Other",
]

# ============================================================
# 서술식 질문 (Narrative Question) 설정
# - 확장성: 새로운 패턴 추가 시 이 설정만 수정하면 됨
# ============================================================

@dataclass
class NarrativePattern:
    """서술식 질문 패턴 정의"""
    title: str                          # 질문 제목 (분리 기준)
    separator: Optional[str] = None     # 구분자 (None이면 제목 바로 뒤에서 분리)
    min_response_length: int = 50       # 최소 응답 길이 (이보다 짧으면 응답 없음으로 처리)


# 서술식 질문 패턴 목록 (우선순위 순서대로 매칭)
NARRATIVE_PATTERNS: List[NarrativePattern] = [
    # 정의 적용 관련
    NarrativePattern(title="Application of definition"),

    # 프로세스 상세 설명
    NarrativePattern(title="Further details of process"),

    # 상호연결성 설명
    NarrativePattern(title="Description of how interconnections are assessed"),

    # 우선순위 위치 미선정 이유
    NarrativePattern(title="Explain why you do not identify priority locations"),

    # 일반적인 설명/기술 패턴
    NarrativePattern(title="Description of"),
    NarrativePattern(title="Explanation of"),
    NarrativePattern(title="Details of"),
    NarrativePattern(title="Describe"),
    NarrativePattern(title="Explain"),
    NarrativePattern(title="Provide details"),
]


# 서술식 응답 끝에 붙는 후행 마커 (제거 대상)
NARRATIVE_TRAILING_MARKERS = [
    "Opportunities",
    "Opportunity",
    "Risks",
    "Risk",
    "[Add row]",
    "[Fixed row]",
]


# 서술식 질문 감지를 위한 추가 힌트
NARRATIVE_HINTS = {
    # rowType이 이 값이면 서술식일 가능성 높음
    "row_types": ["fixed", "addable"],

    # 이 패턴이 블록에 없으면 서술식일 가능성 높음
    "exclude_patterns": ["Select from:", "Select all that apply"],

    # 응답 최소 길이 기본값
    "default_min_response_length": 50,
}


# 알려진 CDP 테이블 컬럼 정의
CDP_TABLE_COLUMNS = {
    "2.2": [
        {
            "columnId": "process_in_place",
            "header": "Process in place",
            "inputType": {
                "type": "singleSelect",
                "options": [
                    {"value": "yes", "label": "Yes"},
                    {"value": "no", "label": "No"}
                ]
            }
        },
        {
            "columnId": "dependencies_impacts",
            "header": "Dependencies and/or impacts evaluated in this process",
            "inputType": {
                "type": "singleSelect",
                "options": [
                    {"value": "dependencies", "label": "Dependencies"},
                    {"value": "impacts", "label": "Impacts"},
                    {"value": "both", "label": "Both dependencies and impacts"}
                ]
            }
        }
    ],
    "2.2.1": [
        {
            "columnId": "process_in_place",
            "header": "Process in place",
            "inputType": {
                "type": "singleSelect",
                "options": [
                    {"value": "yes", "label": "Yes"},
                    {"value": "no", "label": "No"}
                ]
            }
        },
        {
            "columnId": "risks_opportunities",
            "header": "Risks and/or opportunities evaluated in this process",
            "inputType": {
                "type": "singleSelect",
                "options": [
                    {"value": "risks", "label": "Risks"},
                    {"value": "opportunities", "label": "Opportunities"},
                    {"value": "both", "label": "Both risks and opportunities"}
                ]
            }
        },
        {
            "columnId": "informed_by_dependencies",
            "header": "Is this process informed by the dependencies and/or impacts process?",
            "inputType": {
                "type": "singleSelect",
                "options": [
                    {"value": "yes", "label": "Yes"},
                    {"value": "no", "label": "No"}
                ]
            }
        }
    ]
}
