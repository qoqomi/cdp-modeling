"""
RAG Document Schema - 메타데이터 필수 스키마

핵심 원칙:
- 모든 문서는 source_type, year 메타데이터 필수
- historical flag로 "참고용" 여부 표시
- CDP 답변과 지속가능경영보고서 구분
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum


class SourceType(Enum):
    """문서 유형"""
    CDP_ANSWER = "CDP_ANSWER"                        # 과거 CDP 답변
    SUSTAINABILITY_REPORT = "SUSTAINABILITY_REPORT"  # 지속가능경영보고서


@dataclass
class RAGDocument:
    """
    RAG 저장용 문서 스키마 - 메타데이터 필수

    금지사항:
    - 연도 메타데이터 없이 벡터화 금지
    - historical flag 없이 저장 금지
    """

    # === 필수 메타데이터 ===
    source_type: SourceType      # 문서 유형 (CDP_ANSWER / SUSTAINABILITY_REPORT)
    year: int                    # 연도 (2023, 2024 등)
    text: str                    # 본문 내용

    # === CDP 답변 전용 (source_type == CDP_ANSWER) ===
    question_code: Optional[str] = None    # C1.2, C2.1a 등
    module: Optional[str] = None           # Climate Governance, Risks and Opportunities 등

    # === 지속가능경영보고서 전용 ===
    section: Optional[str] = None          # 환경, 사회, 지배구조 등
    page_num: Optional[int] = None         # 페이지 번호

    # === 공통 ===
    historical: bool = True                # 과거 데이터 여부 (참고용 표시)
    chunk_id: Optional[str] = None         # 청크 고유 ID
    score: float = 0.0                     # 검색 점수 (검색 결과에서 설정)

    # === 추가 메타데이터 ===
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """유효성 검증"""
        if not self.text:
            raise ValueError("text는 필수입니다")
        if not isinstance(self.year, int) or self.year < 2000:
            raise ValueError(f"유효하지 않은 연도: {self.year}")
        if self.source_type == SourceType.CDP_ANSWER and not self.question_code:
            raise ValueError("CDP_ANSWER 타입은 question_code가 필수입니다")

    def to_payload(self) -> Dict[str, Any]:
        """Qdrant 저장용 payload 변환"""
        return {
            "source_type": self.source_type.value,
            "year": self.year,
            "text": self.text,
            "question_code": self.question_code,
            "module": self.module,
            "section": self.section,
            "page_num": self.page_num,
            "historical": self.historical,
            "chunk_id": self.chunk_id,
            **self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], score: float = 0.0) -> "RAGDocument":
        """Qdrant payload에서 복원"""
        return cls(
            source_type=SourceType(payload["source_type"]),
            year=payload["year"],
            text=payload["text"],
            question_code=payload.get("question_code"),
            module=payload.get("module"),
            section=payload.get("section"),
            page_num=payload.get("page_num"),
            historical=payload.get("historical", True),
            chunk_id=payload.get("chunk_id"),
            score=score,
        )

    @classmethod
    def create_cdp_answer(
        cls,
        year: int,
        question_code: str,
        text: str,
        module: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> "RAGDocument":
        """CDP 답변 문서 생성 헬퍼"""
        return cls(
            source_type=SourceType.CDP_ANSWER,
            year=year,
            text=text,
            question_code=question_code,
            module=module,
            historical=True,  # 과거 CDP 답변은 항상 historical
            chunk_id=chunk_id or f"cdp_{year}_{question_code}",
        )

    @classmethod
    def create_sustainability_report(
        cls,
        year: int,
        text: str,
        section: Optional[str] = None,
        page_num: Optional[int] = None,
        chunk_id: Optional[str] = None,
    ) -> "RAGDocument":
        """지속가능경영보고서 문서 생성 헬퍼"""
        return cls(
            source_type=SourceType.SUSTAINABILITY_REPORT,
            year=year,
            text=text,
            section=section,
            page_num=page_num,
            historical=False,  # 현재 연도 보고서는 historical=False
            chunk_id=chunk_id or f"sr_{year}_{page_num or 'unknown'}",
        )


@dataclass
class SearchFilter:
    """RAG 검색 필터"""
    source_types: Optional[List[SourceType]] = None
    years: Optional[List[int]] = None
    question_codes: Optional[List[str]] = None
    modules: Optional[List[str]] = None
    historical_only: Optional[bool] = None

    def to_qdrant_filter(self) -> Dict[str, Any]:
        """Qdrant 필터 형식으로 변환"""
        must_conditions = []

        if self.source_types:
            must_conditions.append({
                "key": "source_type",
                "match": {"any": [st.value for st in self.source_types]}
            })

        if self.years:
            must_conditions.append({
                "key": "year",
                "match": {"any": self.years}
            })

        if self.question_codes:
            must_conditions.append({
                "key": "question_code",
                "match": {"any": self.question_codes}
            })

        if self.modules:
            must_conditions.append({
                "key": "module",
                "match": {"any": self.modules}
            })

        if self.historical_only is not None:
            must_conditions.append({
                "key": "historical",
                "match": {"value": self.historical_only}
            })

        if not must_conditions:
            return None

        return {"must": must_conditions}
