"""
CDP Module
==========
CDP 질문지 및 응답서 PDF 파싱

Components:
- parser.py: CDPQuestionnaireSectionParser (질문지 PDF → 구조화 데이터)
- response_parser.py: CDPResponseParser (응답서 PDF → RAG 청크)
- merger.py: merge_questionnaire (Schema + PDF → 최종 JSON)
"""

from .parser import (
    CDPQuestionnaireSectionParser,
    CDPQuestion,
    ResponseColumn,
    FieldType,
    Tag,
)
from .response_parser import CDPResponseParser, CDPResponseChunk
from .merger import merge_questionnaire

__all__ = [
    # 질문지 파서
    "CDPQuestionnaireSectionParser",
    "CDPQuestion",
    "ResponseColumn",
    "FieldType",
    "Tag",
    "merge_questionnaire",
    # 응답서 파서
    "CDPResponseParser",
    "CDPResponseChunk",
]
