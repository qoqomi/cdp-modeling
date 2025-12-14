"""
Sustainability Report Chunker (Enhanced)
=========================================
지속가능성 보고서 PDF를 RAG 인덱싱용 청크로 변환.

개선 사항:
- 테이블 감지 및 추출
- 헤더/섹션 기반 청킹
- 시맨틱 청킹 (문장 단위 분리)
- 메타데이터 강화
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF


@dataclass
class ReportChunk:
    """지속가능성 보고서 청크"""

    id: str
    content: str
    page_num: int
    section: Optional[str] = None
    chunk_type: str = "text"  # text, table, header
    metadata: Dict[str, Any] = field(default_factory=dict)


class SustainabilityReportChunker:
    """지속가능성 보고서 PDF → RAG 청크 변환 (Enhanced)"""

    # 섹션 키워드 (CDP 카테고리 기반)
    SECTION_KEYWORDS = {
        "governance": [
            "governance", "board", "committee", "management", "oversight",
            "director", "executive", "leadership", "responsibility",
            "이사회", "위원회", "경영진", "거버넌스"
        ],
        "strategy": [
            "strategy", "strategic", "roadmap", "vision", "target", "goal",
            "pathway", "transition", "scenario", "planning",
            "전략", "비전", "로드맵", "목표"
        ],
        "risk_management": [
            "risk", "opportunity", "assessment", "identification", "mitigation",
            "climate risk", "physical risk", "transition risk",
            "리스크", "위험", "기회", "평가"
        ],
        "emissions": [
            "emissions", "ghg", "scope 1", "scope 2", "scope 3", "carbon",
            "co2", "greenhouse gas", "carbon footprint", "carbon intensity",
            "배출", "온실가스", "탄소"
        ],
        "energy": [
            "energy", "renewable", "consumption", "efficiency", "electricity",
            "solar", "wind", "power", "fuel",
            "에너지", "전력", "재생에너지"
        ],
        "water": [
            "water", "withdrawal", "discharge", "stress", "wastewater",
            "freshwater", "recycled water", "water intensity",
            "물", "용수", "폐수"
        ],
        "biodiversity": [
            "biodiversity", "ecosystem", "habitat", "species", "nature",
            "deforestation", "land use", "protected area",
            "생물다양성", "생태계"
        ],
        "waste": [
            "waste", "recycling", "circular", "disposal", "landfill",
            "hazardous", "non-hazardous", "waste reduction",
            "폐기물", "재활용", "순환"
        ],
        "supply_chain": [
            "supplier", "supply chain", "procurement", "vendor", "sourcing",
            "upstream", "downstream", "value chain",
            "공급망", "협력사", "공급업체"
        ],
        "targets": [
            "target", "goal", "commitment", "net zero", "reduction",
            "sbti", "science-based", "carbon neutral", "2030", "2050",
            "목표", "감축", "넷제로"
        ],
        "metrics": [
            "metric", "kpi", "indicator", "performance", "data",
            "measurement", "baseline", "progress",
            "지표", "성과", "데이터"
        ],
    }

    # 테이블 감지 패턴
    TABLE_PATTERNS = [
        r'\b\d{4}\b.*\b\d{4}\b',  # 연도 비교 (2022, 2023)
        r'\b\d+\.?\d*\s*%',  # 퍼센트
        r'\b\d+,\d{3}',  # 천 단위 숫자
        r'scope\s*[123]',  # Scope 1/2/3
        r'tCO2e?|ton|톤|MWh|GJ',  # 단위
    ]

    # 헤더 패턴
    HEADER_PATTERNS = [
        r'^[0-9]+\.(?:[0-9]+\.)*\s+.+',  # 1. 또는 1.1. 형식
        r'^[A-Z][A-Z\s]{3,}$',  # 대문자 제목
        r'^[가-힣]{2,}\s*$',  # 한글 제목
    ]

    def __init__(
        self,
        pdf_path: str,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        extract_tables: bool = True,
    ):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.extract_tables = extract_tables
        self.doc = fitz.open(pdf_path)
        self.chunks: List[ReportChunk] = []

    def _detect_section(self, text: str) -> Optional[str]:
        """텍스트에서 섹션 감지"""
        text_lower = (text or "").lower()
        section_scores = {}

        for section, keywords in self.SECTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score >= 2:
                section_scores[section] = score

        if section_scores:
            return max(section_scores, key=section_scores.get)
        return None

    def _is_table_content(self, text: str) -> bool:
        """테이블 형식 콘텐츠 감지"""
        if not text:
            return False

        # 여러 열이 있는 패턴 감지
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False

        # 탭이나 여러 공백으로 구분된 데이터
        tab_lines = sum(1 for line in lines if '\t' in line or '  ' in line)

        # 숫자가 많은 라인
        number_lines = sum(1 for line in lines if re.search(r'\d+\.?\d*', line))

        # 패턴 매칭
        pattern_matches = sum(
            1 for pattern in self.TABLE_PATTERNS
            if re.search(pattern, text, re.IGNORECASE)
        )

        return (tab_lines >= len(lines) * 0.5 or
                (number_lines >= len(lines) * 0.6 and pattern_matches >= 2))

    def _is_header(self, text: str) -> bool:
        """헤더/제목 감지"""
        text = text.strip()
        if not text or len(text) > 200:
            return False

        for pattern in self.HEADER_PATTERNS:
            if re.match(pattern, text):
                return True

        # 짧고 숫자가 없는 텍스트
        if len(text) < 50 and not re.search(r'\d', text):
            words = text.split()
            if len(words) <= 5:
                return True

        return False

    def _split_into_sentences(self, text: str) -> List[str]:
        """문장 단위로 분리"""
        # 문장 끝 패턴
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z가-힣])'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_tables_from_page(self, page) -> List[Dict[str, Any]]:
        """페이지에서 테이블 추출"""
        tables = []

        try:
            # PyMuPDF의 테이블 감지 시도
            tab = page.find_tables()
            if tab.tables:
                for table in tab.tables:
                    table_data = table.extract()
                    if table_data:
                        # 테이블을 텍스트로 변환
                        table_text = self._table_to_text(table_data)
                        if table_text and len(table_text) > 50:
                            tables.append({
                                "content": table_text,
                                "bbox": table.bbox,
                                "rows": len(table_data),
                                "cols": len(table_data[0]) if table_data else 0,
                            })
        except Exception:
            pass  # 테이블 추출 실패 시 무시

        return tables

    def _table_to_text(self, table_data: List[List]) -> str:
        """테이블 데이터를 구조화된 텍스트로 변환"""
        if not table_data:
            return ""

        lines = []
        headers = table_data[0] if table_data else []

        # 헤더 추가
        if headers:
            header_text = " | ".join(str(cell or "").strip() for cell in headers)
            lines.append(f"[Table Headers: {header_text}]")

        # 데이터 행 추가
        for row in table_data[1:]:
            row_parts = []
            for i, cell in enumerate(row):
                cell_text = str(cell or "").strip()
                if cell_text:
                    header_label = str(headers[i]).strip() if i < len(headers) else f"Col{i}"
                    row_parts.append(f"{header_label}: {cell_text}")
            if row_parts:
                lines.append(" | ".join(row_parts))

        return "\n".join(lines)

    def _create_semantic_chunks(
        self,
        text: str,
        page_num: int,
        section: Optional[str],
        chunk_type: str = "text"
    ) -> List[ReportChunk]:
        """시맨틱 청킹 (문장 경계 존중)"""
        chunks = []
        sentences = self._split_into_sentences(text)

        current_chunk = ""
        chunk_sentences = []

        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
                chunk_sentences.append(sentence)
            else:
                # 현재 청크 저장
                if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
                    chunk_id = f"chunk_{page_num}_{len(chunks)}"
                    chunks.append(ReportChunk(
                        id=chunk_id,
                        content=current_chunk.strip(),
                        page_num=page_num,
                        section=section,
                        chunk_type=chunk_type,
                        metadata={
                            "source": os.path.basename(self.pdf_path),
                            "page": page_num,
                            "sentence_count": len(chunk_sentences),
                        }
                    ))

                # 오버랩 적용
                if self.chunk_overlap > 0 and chunk_sentences:
                    overlap_text = ""
                    for s in reversed(chunk_sentences):
                        if len(overlap_text) + len(s) <= self.chunk_overlap:
                            overlap_text = s + " " + overlap_text
                        else:
                            break
                    current_chunk = overlap_text.strip() + " " + sentence
                else:
                    current_chunk = sentence
                chunk_sentences = [sentence]

        # 마지막 청크 저장
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk_id = f"chunk_{page_num}_{len(chunks)}"
            chunks.append(ReportChunk(
                id=chunk_id,
                content=current_chunk.strip(),
                page_num=page_num,
                section=section,
                chunk_type=chunk_type,
                metadata={
                    "source": os.path.basename(self.pdf_path),
                    "page": page_num,
                    "sentence_count": len(chunk_sentences),
                }
            ))

        return chunks

    def extract_page_content(self, page, page_num: int) -> Dict[str, Any]:
        """페이지에서 구조화된 콘텐츠 추출"""
        text = page.get_text("text")
        section = self._detect_section(text)

        # 테이블 추출
        tables = []
        if self.extract_tables:
            tables = self._extract_tables_from_page(page)

        # 블록 단위로 텍스트 추출 (레이아웃 유지)
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        header_blocks = []

        for block in blocks:
            if block.get("type") == 0:  # 텍스트 블록
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "") + " "
                    block_text += "\n"

                block_text = block_text.strip()
                if block_text:
                    if self._is_header(block_text):
                        header_blocks.append(block_text)
                    else:
                        text_blocks.append(block_text)

        return {
            "page_num": page_num,
            "text": text,
            "section": section,
            "tables": tables,
            "text_blocks": text_blocks,
            "headers": header_blocks,
        }

    def parse(self) -> List[ReportChunk]:
        """PDF를 청크로 변환"""
        self.chunks = []
        chunk_counter = 0

        for page_num, page in enumerate(self.doc):
            page_data = self.extract_page_content(page, page_num + 1)
            section = page_data["section"]

            # 1. 헤더 청크 (짧은 제목들은 다음 텍스트와 결합)
            current_header = ""
            for header in page_data["headers"]:
                current_header += header + "\n"

            # 2. 테이블 청크
            for table in page_data["tables"]:
                table_content = table["content"]
                if current_header:
                    table_content = f"{current_header.strip()}\n\n{table_content}"
                    current_header = ""

                chunk_id = f"table_{page_num + 1}_{chunk_counter}"
                self.chunks.append(ReportChunk(
                    id=chunk_id,
                    content=table_content,
                    page_num=page_num + 1,
                    section=section,
                    chunk_type="table",
                    metadata={
                        "source": os.path.basename(self.pdf_path),
                        "page": page_num + 1,
                        "rows": table.get("rows", 0),
                        "cols": table.get("cols", 0),
                    }
                ))
                chunk_counter += 1

            # 3. 텍스트 블록을 시맨틱 청킹
            combined_text = "\n\n".join(page_data["text_blocks"])
            if current_header:
                combined_text = f"{current_header.strip()}\n\n{combined_text}"

            if combined_text.strip():
                page_chunks = self._create_semantic_chunks(
                    combined_text,
                    page_num + 1,
                    section,
                    chunk_type="text"
                )

                # ID 재할당
                for chunk in page_chunks:
                    chunk.id = f"report_chunk_{chunk_counter}"
                    chunk_counter += 1
                    self.chunks.append(chunk)

        # 중복 제거 (유사한 청크)
        self.chunks = self._deduplicate_chunks(self.chunks)

        return self.chunks

    def _deduplicate_chunks(
        self,
        chunks: List[ReportChunk],
        similarity_threshold: float = 0.9
    ) -> List[ReportChunk]:
        """유사한 청크 중복 제거"""
        if not chunks:
            return chunks

        unique_chunks = []
        seen_content = set()

        for chunk in chunks:
            # 정규화된 콘텐츠
            normalized = re.sub(r'\s+', ' ', chunk.content.lower().strip())

            # 첫 100자를 기준으로 중복 체크 (완전 중복 방지)
            content_key = normalized[:100]

            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_chunks.append(chunk)

        return unique_chunks

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "source": os.path.basename(self.pdf_path),
            "total_pages": len(self.doc),
            "total_chunks": len(self.chunks),
            "chunk_types": {
                "text": sum(1 for c in self.chunks if c.chunk_type == "text"),
                "table": sum(1 for c in self.chunks if c.chunk_type == "table"),
            },
            "sections": list(set(c.section for c in self.chunks if c.section)),
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }

    def to_json(self, output_path: str, indent: int = 2):
        """JSON으로 저장"""
        data = self.to_dict()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        print(f"Saved {len(self.chunks)} chunks to {output_path}")
        return data

    def get_stats(self) -> Dict[str, Any]:
        """청킹 통계 반환"""
        if not self.chunks:
            return {"error": "No chunks available. Run parse() first."}

        content_lengths = [len(c.content) for c in self.chunks]
        sections = [c.section for c in self.chunks if c.section]

        return {
            "total_chunks": len(self.chunks),
            "avg_chunk_length": sum(content_lengths) / len(content_lengths),
            "min_chunk_length": min(content_lengths),
            "max_chunk_length": max(content_lengths),
            "chunk_types": {
                "text": sum(1 for c in self.chunks if c.chunk_type == "text"),
                "table": sum(1 for c in self.chunks if c.chunk_type == "table"),
            },
            "sections": dict((s, sections.count(s)) for s in set(sections)),
            "pages_covered": len(set(c.page_num for c in self.chunks)),
        }


def main():
    """CLI 테스트"""
    import argparse

    parser = argparse.ArgumentParser(description="Sustainability Report Chunker")
    parser.add_argument("pdf_path", help="PDF file path")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--stats", action="store_true", help="Print stats only")

    args = parser.parse_args()

    chunker = SustainabilityReportChunker(
        args.pdf_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = chunker.parse()

    if args.stats:
        import pprint
        pprint.pprint(chunker.get_stats())
    elif args.output:
        chunker.to_json(args.output)
    else:
        print(f"Created {len(chunks)} chunks")
        for chunk in chunks[:3]:
            print(f"\n--- {chunk.id} (page {chunk.page_num}, {chunk.chunk_type}) ---")
            print(chunk.content[:300] + "...")


if __name__ == "__main__":
    main()
