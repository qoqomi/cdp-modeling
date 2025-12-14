"""Sustainability Report Parser.

지속가능성 보고서 PDF를 구조화된 JSON으로 변환하는 파서.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF

from ..base import BasePDFParser


def _clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


@dataclass
class ReportPage:
    page_num: int
    text: str


@dataclass
class ReportSection:
    id: str
    title: str
    level: int
    page_start: int
    page_end: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportChunk:
    id: str
    content: str
    page_num: int
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SustainabilityReportParser(BasePDFParser):
    """
    Sustainability report PDF parser.

    - 섹션 감지는 PDF의 span font-size를 이용한 휴리스틱.
    - 텍스트 기반이므로 완벽한 "레이아웃 복원"은 목표가 아님.
    """

    def __init__(
        self,
        pdf_path: str,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        heading_min_chars: int = 3,
        heading_max_chars: int = 140,
    ):
        super().__init__(pdf_path)
        self.doc = fitz.open(str(self.pdf_path))
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.heading_min_chars = heading_min_chars
        self.heading_max_chars = heading_max_chars

        self.pages: List[ReportPage] = []
        self.sections: List[ReportSection] = []
        self.chunks: List[ReportChunk] = []

    def _iter_spans(self, page: fitz.Page) -> Iterable[Dict[str, Any]]:
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    yield span

    def _estimate_body_font_size(self, page: fitz.Page) -> Optional[float]:
        sizes: List[float] = []
        for span in self._iter_spans(page):
            text = _normalize_whitespace(span.get("text", ""))
            if not text:
                continue
            # 숫자/페이지 번호/짧은 토막은 제외
            if len(text) <= 2:
                continue
            if re.fullmatch(r"[\d\W_]+", text):
                continue
            size = span.get("size")
            if isinstance(size, (int, float)):
                sizes.append(float(size))
        if not sizes:
            return None
        return float(statistics.median(sizes))

    def _extract_heading_candidates(
        self, page: fitz.Page, body_size: float
    ) -> List[Tuple[float, str]]:
        """
        Returns a list of (y0, text) heading candidates ordered by y0 (top -> bottom).
        """
        data = page.get_text("dict")
        candidates: List[Tuple[float, str]] = []

        def is_heading_text(t: str) -> bool:
            t = _normalize_whitespace(t)
            if len(t) < self.heading_min_chars or len(t) > self.heading_max_chars:
                return False
            if re.fullmatch(r"[\d\W_]+", t):
                return False
            # 흔한 헤더/푸터 노이즈
            if re.fullmatch(r"page\s*\d+", t, re.IGNORECASE):
                return False
            return True

        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = _normalize_whitespace("".join(s.get("text", "") for s in spans))
                if not is_heading_text(line_text):
                    continue

                max_size = max(float(s.get("size", 0.0) or 0.0) for s in spans)

                # heading 판단: body 대비 충분히 큰 폰트
                if max_size >= body_size + 1.8:
                    y0 = float(line.get("bbox", [0, 0, 0, 0])[1])
                    candidates.append((y0, line_text))

        candidates.sort(key=lambda x: x[0])
        return candidates

    def _merge_nearby_headings(
        self, headings: List[Tuple[float, str]], y_threshold: float = 14.0
    ) -> List[str]:
        """
        같은 heading이 여러 라인으로 나뉜 경우 합치기.
        """
        if not headings:
            return []
        merged: List[str] = []
        buf = headings[0][1]
        prev_y = headings[0][0]
        for y, text in headings[1:]:
            if abs(y - prev_y) <= y_threshold:
                buf = _normalize_whitespace(f"{buf} {text}")
            else:
                merged.append(buf)
                buf = text
            prev_y = y
        merged.append(buf)
        # 중복 제거
        out: List[str] = []
        seen = set()
        for t in merged:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                out.append(t)
        return out

    def _split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", _clean_text(text))
        return [p.strip() for p in paragraphs if p.strip()]

    def _chunk_text(self, text: str) -> List[str]:
        paragraphs = self._split_into_paragraphs(text)
        chunks: List[str] = []
        current = ""
        for para in paragraphs:
            if not current:
                current = para
                continue
            if len(current) + 2 + len(para) <= self.chunk_size:
                current += "\n\n" + para
                continue

            chunks.append(current.strip())
            if self.chunk_overlap > 0:
                overlap = current[-self.chunk_overlap :]
                current = overlap + "\n\n" + para
            else:
                current = para
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def parse_pages(self, page_limit: Optional[int] = None) -> List[ReportPage]:
        self.pages = []
        total = len(self.doc) if page_limit is None else min(len(self.doc), page_limit)
        for i in range(total):
            text = _clean_text(self.doc[i].get_text("text"))
            self.pages.append(ReportPage(page_num=i + 1, text=text))
        return self.pages

    def parse_sections(self, page_limit: Optional[int] = None) -> List[ReportSection]:
        """
        섹션 추출(heading 기반).
        - level은 현재는 단일 레벨(1)로 두고, 향후 필요 시 확장.
        """
        self.sections = []
        pages = self.parse_pages(page_limit=page_limit)

        # 페이지별 heading 후보 수집
        page_headings: Dict[int, List[str]] = {}
        for page in self.doc[: len(pages)]:
            body_size = self._estimate_body_font_size(page) or 10.0
            headings = self._extract_heading_candidates(page, body_size)
            merged = self._merge_nearby_headings(headings)
            if merged:
                page_headings[page.number + 1] = merged

        # 단순 섹션 분할: "새 heading"을 만나면 새 섹션 시작
        # - 다만 같은 heading이 연속 페이지에 반복될 경우 하나로 합침
        current_title: Optional[str] = None
        current_start: int = 1
        current_content: List[str] = []
        section_idx = 0
        last_page_num: Optional[int] = None

        def flush(end_page: int):
            nonlocal section_idx, current_title, current_start, current_content
            if not current_title:
                return
            content = _clean_text("\n\n".join(current_content))
            if not content:
                return
            section_idx += 1
            self.sections.append(
                ReportSection(
                    id=f"section_{section_idx}",
                    title=current_title,
                    level=1,
                    page_start=current_start,
                    page_end=end_page,
                    content=content,
                    metadata={},
                )
            )

        for page in pages:
            headings = page_headings.get(page.page_num, [])
            # 한 페이지에 heading이 여러개인 경우: 첫 heading을 섹션 타이틀로 취급
            # (나머지는 후속 섹션 후보로 간주하는 고급 로직은 추후 확장)
            if headings:
                next_title = headings[0]
                if (
                    current_title
                    and next_title.strip().lower() == current_title.strip().lower()
                    and last_page_num is not None
                    and page.page_num == last_page_num + 1
                ):
                    # 같은 제목이 연속 페이지에서 반복되는 경우: 섹션 유지
                    pass
                else:
                    # 기존 섹션 flush
                    flush(page.page_num - 1)
                    current_title = next_title
                    current_start = page.page_num
                    current_content = []

            current_content.append(page.text)
            last_page_num = page.page_num

        flush(pages[-1].page_num if pages else 1)
        return self.sections

    def parse_chunks(self) -> List[ReportChunk]:
        """
        섹션 기반 청킹.
        parse_sections() 이후 호출 권장.
        """
        self.chunks = []
        if not self.sections:
            self.parse_sections()

        chunk_idx = 0
        for sec in self.sections:
            for c in self._chunk_text(sec.content):
                chunk_idx += 1
                self.chunks.append(
                    ReportChunk(
                        id=f"report_chunk_{chunk_idx}",
                        content=c,
                        page_num=sec.page_start,
                        section_id=sec.id,
                        section_title=sec.title,
                        metadata={"source": os.path.basename(self.pdf_path), "page": sec.page_start},
                    )
                )
        return self.chunks

    def parse(self) -> List[ReportSection]:
        self.parse_sections()
        self.parse_chunks()
        return self.sections

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": os.path.basename(self.pdf_path),
            "total_pages": len(self.doc),
            "sections": [asdict(s) for s in self.sections],
            "chunks": [asdict(c) for c in self.chunks],
        }

    def to_json(self, output_path: str, indent: int = 2) -> None:
        if not self.sections:
            self.parse()
        data = self.to_dict()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
