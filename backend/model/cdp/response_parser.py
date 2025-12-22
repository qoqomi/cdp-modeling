"""
CDP Response PDF Parser
=======================
과거 CDP 응답서 PDF를 파싱하여 RAG 인덱싱용 청크로 변환

대상 파일: SK_Inc.-16-09-2025-CDP_sample.pdf 등
출력: 질문별 답변 청크 (RAG 인덱싱용)

Note: 기존 parser.py (질문지 파서)와 별개의 모듈
"""

import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple

import fitz  # PyMuPDF


@dataclass
class CDPResponseChunk:
    """CDP 응답서 청크"""

    question_id: str           # 질문 ID (예: "2.2", "2.2.1")
    question_text: str         # 질문 내용
    answer_text: str           # 답변 내용
    page_num: int              # 페이지 번호
    module: Optional[str] = None

    def to_text(self) -> str:
        """RAG 인덱싱용 텍스트 변환"""
        parts = [
            f"Question {self.question_id}: {self.question_text}",
            f"Answer: {self.answer_text}",
        ]
        if self.module:
            parts.insert(0, f"Module: {self.module}")
        return "\n".join(parts)


class CDPResponseParser:
    """
    CDP 응답서 PDF 파서

    - PDF에서 질문 ID와 답변 추출
    - 질문별로 청크 생성
    - RAG 인덱싱에 적합한 형태로 변환
    """

    # 질문 ID 패턴: (2.2), (2.2.1), (C1.1) 등
    QUESTION_PATTERN = re.compile(r'\((\d+\.[\d\.]+[a-z]?|C\d+\.[\d\.]+[a-z]?)\)')

    # 선택 답변 패턴
    CHECKBOX_PATTERN = re.compile(r'☑\s*(.+?)(?=☑|☐|$|\n)', re.DOTALL)
    UNCHECKED_PATTERN = re.compile(r'☐\s*(.+?)(?=☑|☐|$|\n)', re.DOTALL)

    # 섹션 구분 패턴
    SELECT_FROM_PATTERN = re.compile(r'Select from:|Select all that apply', re.IGNORECASE)
    ROW_PATTERN = re.compile(r'^Row\s+\d+', re.IGNORECASE | re.MULTILINE)

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.doc = fitz.open(str(self.pdf_path))
        self.chunks: List[CDPResponseChunk] = []

    def _extract_all_text(self) -> List[Tuple[int, str]]:
        """모든 페이지 텍스트 추출"""
        pages = []
        for page_num, page in enumerate(self.doc):
            text = page.get_text("text")
            pages.append((page_num + 1, text))
        return pages

    def _find_questions(self, text: str) -> List[Tuple[int, str]]:
        """텍스트에서 질문 ID와 위치 찾기"""
        matches = []
        for match in self.QUESTION_PATTERN.finditer(text):
            question_id = match.group(1)
            start_pos = match.start()
            matches.append((start_pos, question_id))
        return matches

    def _extract_selected_options(self, text: str) -> List[str]:
        """체크된 옵션 추출"""
        selected = []
        for match in self.CHECKBOX_PATTERN.finditer(text):
            option = match.group(1).strip()
            # 다음 질문 ID가 포함되어 있으면 제외
            if not self.QUESTION_PATTERN.search(option):
                selected.append(option)
        return selected

    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        # 연속 공백/줄바꿈 제거
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _extract_question_content(
        self,
        text: str,
        start_pos: int,
        end_pos: int
    ) -> Tuple[str, str]:
        """질문 내용과 답변 추출"""
        content = text[start_pos:end_pos]

        # 질문 텍스트와 답변 분리
        lines = content.split('\n')
        question_lines = []
        answer_lines = []

        in_answer = False
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 답변 시작 감지
            if self.SELECT_FROM_PATTERN.search(line) or '☑' in line or '☐' in line:
                in_answer = True

            if in_answer:
                answer_lines.append(line)
            else:
                question_lines.append(line)

        question_text = ' '.join(question_lines)
        answer_text = '\n'.join(answer_lines)

        # 선택된 옵션 정리
        selected = self._extract_selected_options(answer_text)
        if selected:
            answer_text = "Selected: " + ", ".join(selected)

        return self._clean_text(question_text), self._clean_text(answer_text)

    def parse(self) -> List[CDPResponseChunk]:
        """PDF 파싱하여 청크 생성"""
        self.chunks = []
        pages = self._extract_all_text()

        # 전체 텍스트 결합 (페이지 경계 처리)
        full_text = ""
        page_offsets = []  # (offset, page_num)

        for page_num, text in pages:
            page_offsets.append((len(full_text), page_num))
            full_text += text + "\n\n"

        # 질문 위치 찾기
        questions = self._find_questions(full_text)

        # 질문별 청크 생성
        for i, (start_pos, question_id) in enumerate(questions):
            # 다음 질문까지의 범위
            if i + 1 < len(questions):
                end_pos = questions[i + 1][0]
            else:
                end_pos = len(full_text)

            # 질문 내용과 답변 추출
            question_text, answer_text = self._extract_question_content(
                full_text, start_pos, end_pos
            )

            # 페이지 번호 결정
            page_num = 1
            for offset, pnum in page_offsets:
                if offset <= start_pos:
                    page_num = pnum
                else:
                    break

            # 빈 답변 건너뛰기
            if not answer_text or len(answer_text) < 5:
                continue

            chunk = CDPResponseChunk(
                question_id=question_id,
                question_text=question_text[:500],  # 질문 텍스트 제한
                answer_text=answer_text,
                page_num=page_num,
            )
            self.chunks.append(chunk)

        print(f"Parsed {len(self.chunks)} CDP response chunks from {self.pdf_path.name}")
        return self.chunks

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "source": self.pdf_path.name,
            "total_pages": len(self.doc),
            "total_chunks": len(self.chunks),
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }

    def get_chunks_for_indexing(self, year: int) -> List[Dict[str, Any]]:
        """
        RAG 인덱싱용 청크 반환

        Args:
            year: CDP 응답 연도

        Returns:
            인덱싱용 청크 리스트 (question_code, text, module 포함)
        """
        if not self.chunks:
            self.parse()

        indexing_chunks = []
        for chunk in self.chunks:
            indexing_chunks.append({
                "question_code": chunk.question_id,
                "text": chunk.to_text(),
                "module": chunk.module,
                "page_num": chunk.page_num,
            })

        return indexing_chunks


def main():
    """테스트"""
    import argparse
    import json

    arg_parser = argparse.ArgumentParser(description="CDP Response PDF Parser")
    arg_parser.add_argument("pdf_path", help="CDP response PDF path")
    arg_parser.add_argument("--output", "-o", help="Output JSON path")
    arg_parser.add_argument("--year", "-y", type=int, default=2024, help="Response year")

    args = arg_parser.parse_args()

    parser = CDPResponseParser(args.pdf_path)
    chunks = parser.parse()

    print(f"\n=== Parsed {len(chunks)} chunks ===\n")

    # 샘플 출력
    for chunk in chunks[:5]:
        print(f"--- {chunk.question_id} (page {chunk.page_num}) ---")
        print(f"Q: {chunk.question_text[:100]}...")
        print(f"A: {chunk.answer_text[:200]}...")
        print()

    if args.output:
        data = parser.to_dict()
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
