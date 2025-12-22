"""
Question Mapper - 연도별 CDP 질문 매핑

핵심 원칙:
- 유사도 검색만으로 문항 매칭 금지
- Rule 기반 명시적 매핑 사용
- 매핑 신뢰도(confidence) 관리
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class HistoricalQuestion:
    """과거 질문 정보"""
    year: int
    code: str


@dataclass
class QuestionMapping:
    """질문 매핑 정보"""
    current_code: str                          # 현재 질문 코드 (C1.2)
    current_year: int                          # 현재 연도 (2025)
    module: str                                # 모듈명 (Climate Governance)
    mapped_historical: List[HistoricalQuestion]  # 과거 질문 리스트
    mapping_confidence: str                    # high / medium / low
    notes: Optional[str] = None                # 매핑 노트


class QuestionMapper:
    """
    연도별 CDP 질문 매핑 관리자

    금지사항:
    - Mapping 없이 유사도만으로 문항 매칭 금지
    """

    def __init__(self, mappings_path: Optional[str] = None):
        """
        Args:
            mappings_path: 매핑 JSON 파일 경로
        """
        if mappings_path is None:
            mappings_path = Path(__file__).parent / "mappings.json"

        self.mappings_path = Path(mappings_path)
        self.mappings: Dict[str, QuestionMapping] = {}
        self.version: str = ""

        self._load_mappings()

    def _load_mappings(self) -> None:
        """매핑 테이블 로드"""
        if not self.mappings_path.exists():
            print(f"Warning: Mappings file not found at {self.mappings_path}")
            return

        with open(self.mappings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.version = data.get("version", "unknown")
        current_year = int(self.version) if self.version.isdigit() else 2025

        for code, mapping_data in data.get("mappings", {}).items():
            historical = [
                HistoricalQuestion(year=h["year"], code=h["code"])
                for h in mapping_data.get("mapped_historical", [])
            ]

            self.mappings[code] = QuestionMapping(
                current_code=code,
                current_year=current_year,
                module=mapping_data.get("module", ""),
                mapped_historical=historical,
                mapping_confidence=mapping_data.get("mapping_confidence", "medium"),
                notes=mapping_data.get("notes"),
            )

        print(f"Loaded {len(self.mappings)} question mappings (version: {self.version})")

    def get_mapping(self, question_code: str) -> Optional[QuestionMapping]:
        """질문 매핑 정보 조회"""
        return self.mappings.get(question_code)

    def get_historical_questions(self, question_code: str) -> List[HistoricalQuestion]:
        """현재 질문에 대한 과거 질문 리스트 반환"""
        mapping = self.get_mapping(question_code)
        if not mapping:
            return []
        return mapping.mapped_historical

    def get_historical_codes_for_rag(self, question_code: str) -> List[str]:
        """RAG 검색용 과거 질문 코드 리스트"""
        historical = self.get_historical_questions(question_code)
        return [h.code for h in historical]

    def get_historical_years_for_rag(self, question_code: str) -> List[int]:
        """RAG 검색용 과거 연도 리스트"""
        historical = self.get_historical_questions(question_code)
        return [h.year for h in historical]

    def get_module(self, question_code: str) -> Optional[str]:
        """질문의 모듈명 조회"""
        mapping = self.get_mapping(question_code)
        return mapping.module if mapping else None

    def get_mapping_confidence(self, question_code: str) -> str:
        """매핑 신뢰도 조회"""
        mapping = self.get_mapping(question_code)
        return mapping.mapping_confidence if mapping else "unknown"

    def has_historical_mapping(self, question_code: str) -> bool:
        """과거 매핑 존재 여부 확인"""
        mapping = self.get_mapping(question_code)
        return mapping is not None and len(mapping.mapped_historical) > 0

    def get_all_codes(self) -> List[str]:
        """모든 질문 코드 반환"""
        return list(self.mappings.keys())

    def add_mapping(
        self,
        question_code: str,
        module: str,
        historical: List[Dict],
        confidence: str = "medium",
        notes: Optional[str] = None
    ) -> None:
        """매핑 추가 (메모리)"""
        historical_questions = [
            HistoricalQuestion(year=h["year"], code=h["code"])
            for h in historical
        ]

        self.mappings[question_code] = QuestionMapping(
            current_code=question_code,
            current_year=int(self.version) if self.version.isdigit() else 2025,
            module=module,
            mapped_historical=historical_questions,
            mapping_confidence=confidence,
            notes=notes,
        )

    def save_mappings(self) -> None:
        """매핑 테이블 저장"""
        data = {
            "version": self.version,
            "mappings": {}
        }

        for code, mapping in self.mappings.items():
            data["mappings"][code] = {
                "current_question": f"{code} ({mapping.current_year})",
                "module": mapping.module,
                "mapped_historical": [
                    {"year": h.year, "code": h.code}
                    for h in mapping.mapped_historical
                ],
                "mapping_confidence": mapping.mapping_confidence,
                "notes": mapping.notes,
            }

        with open(self.mappings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(self.mappings)} mappings to {self.mappings_path}")
