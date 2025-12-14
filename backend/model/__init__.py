"""
CDP Model Module
================
PDF 파서 모듈 - 문서 타입별 분리된 파이프라인

Structure:
    model/
    ├── base.py                     # 공통 베이스 클래스
    ├── cdp/                        # CDP Questionnaire 전용
    │   ├── parser.py               # CDPQuestionnaireSectionParser
    │   └── merger.py               # merge_questionnaire
    └── sustainability/             # 지속가능성 보고서 전용
        ├── parser.py               # SustainabilityReportParser
        └── chunker.py              # RAG 청킹

Usage:
    # CDP 질문지 파싱
    from model.cdp import CDPQuestionnaireSectionParser
    parser = CDPQuestionnaireSectionParser("questionnaire.pdf")
    questions = parser.parse()

    # 스키마 병합
    from model.cdp import merge_questionnaire
    merge_questionnaire(schema_dir, output_path, pdf_path=pdf)

    # 지속가능성 보고서 청킹
    from model.sustainability import SustainabilityReportChunker
    chunker = SustainabilityReportChunker("report.pdf")
    chunks = chunker.parse()
"""

from .base import BasePDFParser

# CDP Questionnaire
from .cdp import (
    CDPQuestionnaireSectionParser,
    CDPQuestion,
    ResponseColumn,
    FieldType,
    Tag,
    merge_questionnaire,
)

# Sustainability Report
from .sustainability import (
    SustainabilityReportParser,
    SustainabilityReportChunker,
    ReportChunk,
)

__all__ = [
    # Base
    "BasePDFParser",
    # CDP
    "CDPQuestionnaireSectionParser",
    "CDPQuestion",
    "ResponseColumn",
    "FieldType",
    "Tag",
    "merge_questionnaire",
    # Sustainability
    "SustainabilityReportParser",
    "SustainabilityReportChunker",
    "ReportChunk",
]
