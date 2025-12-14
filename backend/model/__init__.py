"""
CDP Model Module
================
PDF 파서 모듈 - 문서 타입별 분리된 파이프라인

- QuestionnaireParser: CDP Questionnaire PDF 파싱 (질문 스키마 추출)
- ReportParser: Sustainability Report PDF 파싱 (RAG용 청킹)
"""

from abc import ABC, abstractmethod
from typing import List, Any
from pathlib import Path


class BasePDFParser(ABC):
    """PDF 파서 공통 인터페이스"""

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    @abstractmethod
    def parse(self) -> List[Any]:
        """PDF를 파싱하여 결과 반환"""
        pass

    @abstractmethod
    def to_json(self, output_path: str) -> None:
        """결과를 JSON으로 저장"""
        pass


# Export parsers
from .questionnaire import QuestionnaireParser
from .report import ReportParser

__all__ = ["BasePDFParser", "QuestionnaireParser", "ReportParser"]
