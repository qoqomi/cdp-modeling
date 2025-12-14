"""Common PDF parser interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List


class BasePDFParser(ABC):
    """PDF 파서 공통 인터페이스"""

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    @abstractmethod
    def parse(self) -> List[Any]:
        """PDF를 파싱하여 결과 반환"""
        raise NotImplementedError

    @abstractmethod
    def to_json(self, output_path: str) -> None:
        """결과를 JSON으로 저장"""
        raise NotImplementedError
