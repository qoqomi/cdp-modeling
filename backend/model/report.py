"""
Sustainability Report Parser
=============================
지속가능성 보고서 PDF를 RAG용 청크로 변환하는 파서

대상 파일: *_Sustainability_Report_*.pdf
출력: ReportChunk 리스트 (RAG 인덱싱용)
"""

import json
import os
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

# PDF 처리
import fitz  # PyMuPDF


@dataclass
class ReportChunk:
    """지속가능성 보고서 청크"""

    id: str
    content: str
    page_num: int
    section: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportParser:
    """지속가능성 보고서 PDF 파서"""

    # 주요 섹션 키워드
    SECTION_KEYWORDS = {
        "governance": ["governance", "board", "committee", "management", "oversight"],
        "strategy": ["strategy", "strategic", "roadmap", "vision", "target"],
        "risk_management": [
            "risk",
            "opportunity",
            "assessment",
            "identification",
            "mitigation",
        ],
        "emissions": ["emissions", "ghg", "scope 1", "scope 2", "scope 3", "carbon"],
        "energy": ["energy", "renewable", "consumption", "efficiency"],
        "water": ["water", "withdrawal", "discharge", "stress"],
        "biodiversity": ["biodiversity", "ecosystem", "habitat", "species"],
        "waste": ["waste", "recycling", "circular", "disposal"],
        "supply_chain": ["supplier", "supply chain", "procurement", "vendor"],
        "targets": ["target", "goal", "commitment", "net zero", "reduction"],
    }

    def __init__(
        self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.doc = fitz.open(pdf_path)
        self.chunks: List[ReportChunk] = []

    def extract_text_with_structure(self) -> List[Dict[str, Any]]:
        """페이지별 텍스트 추출"""
        pages = []

        for page_num, page in enumerate(self.doc):
            text = page.get_text("text")

            # 섹션 감지
            section = self._detect_section(text)

            pages.append({"page_num": page_num + 1, "text": text, "section": section})

        return pages

    def _detect_section(self, text: str) -> Optional[str]:
        """텍스트에서 섹션 감지"""
        text_lower = text.lower()

        for section, keywords in self.SECTION_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches >= 2:  # 최소 2개 키워드 매칭
                return section

        return None

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """텍스트를 문단으로 분할"""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def parse(self) -> List[ReportChunk]:
        """청크 생성 (메인 메서드)"""
        pages = self.extract_text_with_structure()

        chunk_id = 0
        for page_data in pages:
            text = page_data["text"]
            page_num = page_data["page_num"]
            section = page_data["section"]

            # 문단 단위로 먼저 분할
            paragraphs = self._split_into_paragraphs(text)

            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) <= self.chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk.strip():
                        self.chunks.append(
                            ReportChunk(
                                id=f"report_chunk_{chunk_id}",
                                content=current_chunk.strip(),
                                page_num=page_num,
                                section=section,
                                metadata={
                                    "source": os.path.basename(self.pdf_path),
                                    "page": page_num,
                                },
                            )
                        )
                        chunk_id += 1

                    # 오버랩 처리
                    if self.chunk_overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-self.chunk_overlap :]
                        current_chunk = overlap_text + para + "\n\n"
                    else:
                        current_chunk = para + "\n\n"

            # 마지막 청크
            if current_chunk.strip():
                self.chunks.append(
                    ReportChunk(
                        id=f"report_chunk_{chunk_id}",
                        content=current_chunk.strip(),
                        page_num=page_num,
                        section=section,
                        metadata={
                            "source": os.path.basename(self.pdf_path),
                            "page": page_num,
                        },
                    )
                )
                chunk_id += 1

        print(f"Created {len(self.chunks)} chunks from report")
        return self.chunks

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "source": os.path.basename(self.pdf_path),
            "total_pages": len(self.doc),
            "total_chunks": len(self.chunks),
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }

    def to_json(self, output_path: str, indent: int = 2):
        """JSON으로 저장"""
        data = self.to_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

        print(f"Saved {len(self.chunks)} chunks to {output_path}")
        return data
