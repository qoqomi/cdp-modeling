"""
Mapping Layer - 연도별 질문 매핑

핵심 원칙:
- RAG가 아닌 Rule 기반 매핑
- 유사도 검색만으로 문항 매칭 금지
- JSON 테이블로 명시적 매핑 관리
"""

from .question_mapper import QuestionMapper, QuestionMapping

__all__ = [
    "QuestionMapper",
    "QuestionMapping",
]
