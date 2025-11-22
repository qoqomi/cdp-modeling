"""
CDP 2025 업데이트 문서 파싱 (개선 버전)
- 테이블 구조 기반 파싱
- 질문별 변경사항 정확히 추출
"""

import pdfplumber
import json
import re
from typing import Dict, List, Any


class CDPUpdatesParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.updates = {}

    def parse(self) -> Dict[str, Any]:
        """CDP 업데이트 문서 파싱 - 테이블 기반"""

        with pdfplumber.open(self.pdf_path) as pdf:
            print(f"✅ PDF 로드: {len(pdf.pages)} 페이지\n")

            # Page 5부터 테이블 추출 (0-indexed: 4)
            for page_num in range(4, len(pdf.pages)):
                page = pdf.pages[page_num]
                tables = page.extract_tables()

                if not tables:
                    continue

                print(f"📄 Page {page_num + 1}: {len(tables)}개 테이블 발견")

                # 각 테이블 파싱
                for table in tables:
                    self._parse_table(table, page_num + 1)

        return self.updates

    def _parse_table(self, table: List[List], page_num: int):
        """테이블 파싱"""

        if not table or len(table) < 2:
            return

        # 각 행 처리
        for row_idx, row in enumerate(table):
            if row_idx == 0:  # 헤더 행 스킵
                continue

            if len(row) < 3:
                continue

            # 질문 ID 찾기 (어느 컬럼에 있는지 확인)
            question_id = ""
            desc_col_idx = -1
            changes_col_idx = -1

            # 첫 번째 컬럼에서 질문 ID 찾기
            for col_idx, cell in enumerate(row):
                if cell and re.match(r'^\d+\.\d+(?:\.\d+)?$', str(cell).strip()):
                    question_id = str(cell).strip()
                    # 다음 non-None 컬럼들이 description, actions(N/A), changes
                    non_none_indices = []
                    for next_idx in range(col_idx + 1, len(row)):
                        if row[next_idx]:
                            non_none_indices.append(next_idx)

                    # description은 첫 번째 non-None
                    if len(non_none_indices) >= 1:
                        desc_col_idx = non_none_indices[0]

                    # changes는 마지막 non-None (N/A 건너뛰기)
                    if len(non_none_indices) >= 2:
                        changes_col_idx = non_none_indices[-1]  # 마지막 컬럼

                    break

            if not question_id or desc_col_idx == -1:
                continue

            # 컬럼 추출
            description = str(row[desc_col_idx]).strip() if desc_col_idx != -1 and row[desc_col_idx] else ""
            changes = str(row[changes_col_idx]).strip() if changes_col_idx != -1 and changes_col_idx < len(row) and row[changes_col_idx] else ""

            # 질문 ID 추출 (예: "2.2.2")
            question_match = re.match(r'^(\d+\.\d+(?:\.\d+)?)', question_id)
            if not question_match:
                continue

            q_id = question_match.group(1)

            # 이미 파싱된 질문이면 스킵 (첫 번째 항목만 유지)
            if q_id in self.updates:
                continue

            print(f"  📝 질문 {q_id} 발견 (Page {page_num})")

            # 저장
            self._save_update(q_id, description, changes)

    def _save_update(self, question_id: str, description: str, changes: str):
        """업데이트 저장"""

        # "N/A" 제거
        description = description.replace('N/A', '').strip()

        # 변경 유형 감지
        change_type = self._detect_change_type(description + " " + changes)

        # "Updated from:" / "Updated to:" 추출
        updated_from = self._extract_section(changes, "Updated from:")
        updated_to = self._extract_section(changes, "Updated to:")

        self.updates[question_id] = {
            "description": description,
            "changes_to_scoring": changes,
            "change_type": change_type,
            "updated_from": updated_from,
            "updated_to": updated_to
        }

        print(f"    ✅ 변경 유형: {change_type}")

    def _detect_change_type(self, text: str) -> str:
        """변경 유형 감지"""

        text_lower = text.lower()

        if "no change" in text_lower:
            return "no_change"
        elif "addition" in text_lower or "added" in text_lower:
            return "addition"
        elif "correction" in text_lower or "corrected" in text_lower:
            return "correction"
        elif "clarified" in text_lower or "clarification" in text_lower:
            return "clarification"
        elif "updated from" in text_lower or "updated to" in text_lower:
            return "update"
        else:
            return "other"

    def _extract_section(self, text: str, marker: str) -> str:
        """특정 마커 이후 텍스트 추출"""

        if not text or marker not in text:
            return ""

        # marker 이후 텍스트 찾기
        parts = text.split(marker)
        if len(parts) < 2:
            return ""

        # 다음 마커 전까지 또는 끝까지
        content = parts[1]

        # "Updated to:" 또는 "Updated from:" 전까지
        for end_marker in ["Updated to:", "Updated from:", "\n\n"]:
            if end_marker in content:
                content = content.split(end_marker)[0]
                break

        return content.strip()

    def save_to_json(self, output_path: str):
        """JSON 저장"""

        result = {
            "document": "CDP Corporate Questionnaires and Scoring Methodologies Updates 2025",
            "version": "1.3",
            "release_date": "2025-09-16",
            "updates": self.updates
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"💾 저장 완료: {output_path}")
        print(f"📊 총 {len(self.updates)}개 질문 업데이트 추출")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = CDPUpdatesParser("data/Corporate_Questionnaires_and_Scoring_Methodologies_Updates_2025_V1.3__16_Sep_.pdf")
    updates = parser.parse()
    parser.save_to_json("config/cdp_2025_updates.json")

    # 결과 확인
    print("\n📋 주요 업데이트 요약:")
    for q_id, info in sorted(updates.items())[:5]:
        print(f"\n질문 {q_id}:")
        print(f"  변경 유형: {info['change_type']}")
        print(f"  설명: {info['description'][:100]}...")
