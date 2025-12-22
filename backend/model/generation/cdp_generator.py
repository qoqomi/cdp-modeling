"""
CDP Answer Generator - 3-Layer 기반 실시간 답변 생성

핵심 원칙:
- RAG Layer: 메타데이터 필터 검색
- Mapping Layer: 연도별 질문 매핑
- Prompt Layer: 과거 답변은 참고용 명시

금지사항:
- 과거 답변을 정답처럼 그대로 사용 금지
- Mapping 없이 유사도만으로 문항 매칭 금지
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from openai import OpenAI
from dotenv import load_dotenv


def strip_markdown(text: str) -> str:
    """마크다운 특수문자 제거

    - `---`, `**text**`, `*text*`, `# heading` 등 제거
    """
    if not text:
        return ""
    text = re.sub(r'^---+\s*', '', text, flags=re.MULTILINE)  # --- 구분선 제거
    text = re.sub(r'---+\s*$', '', text, flags=re.MULTILINE)  # 끝의 --- 제거
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)            # **bold** -> bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)                # *italic* -> italic
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)    # # heading 제거
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)  # - item -> • item
    text = re.sub(r'\n{3,}', '\n\n', text)                    # 과도한 줄바꿈 정리
    return text.strip()


def remove_meta_text(text: str) -> str:
    """메타 텍스트 제거

    - 요약, 참조, 불필요한 설명 제거
    """
    if not text:
        return ""

    # 제거할 패턴들
    patterns_to_remove = [
        r'요약하자면[,:]?\s*',
        r'In summary[,:]?\s*',
        r'CDP 설문지의 \d+\.\d+(\.\d+)? 섹션에서 제공할 예정입니다\.?',
        r'For further details.*?section \d+\.\d+(\.\d+)?.*?questionnaire\.?',
        r'추가 (세부)?정보는.*?제공.*?예정입니다\.?',
    ]

    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text.strip()

import sys
sys.path.append(str(Path(__file__).parent.parent))

from rag.document_schema import RAGDocument, SourceType
from rag.retriever import RAGRetriever
from mapping.question_mapper import QuestionMapper
from .prompt_builder import PromptBuilder

load_dotenv()


@dataclass
class GeneratedAnswer:
    """생성된 답변"""
    question_id: str
    question_title: str
    rows: List[Dict[str, Any]]
    overall_confidence: float
    sources: List[Dict[str, Any]]
    rationale_en: str
    rationale_ko: str
    historical_reference_used: bool
    mapping_confidence: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CDPAnswerGenerator:
    """
    3-Layer 기반 CDP 답변 생성기

    금지사항:
    - 과거 답변을 정답처럼 그대로 사용 금지
    - 연도 메타데이터 없이 벡터화 금지
    - Prompt에서 연도 언급 통제 안 함 금지
    - Mapping 없이 유사도만으로 문항 매칭 금지
    """

    def __init__(
        self,
        rag_retriever: Optional[RAGRetriever] = None,
        question_mapper: Optional[QuestionMapper] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        questions_path: Optional[str] = None,
        current_year: int = 2025,
    ):
        """
        Args:
            rag_retriever: RAG 검색기 (None이면 새로 생성)
            question_mapper: 질문 매퍼 (None이면 새로 생성)
            prompt_builder: 프롬프트 빌더 (None이면 새로 생성)
            questions_path: 질문 JSON 파일 경로
            current_year: 현재 연도
        """
        self.rag = rag_retriever or RAGRetriever()
        self.mapper = question_mapper or QuestionMapper()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.current_year = current_year

        # 질문 데이터 로드 (schemas/2025/module2.json 사용)
        if questions_path is None:
            questions_path = Path(__file__).parent.parent.parent / "schemas" / "2025" / "module2.json"
        self.questions = self._load_questions(questions_path)

        # common_options 로드 (select 필드의 옵션 참조용)
        options_path = Path(__file__).parent.parent.parent / "schemas" / "2025" / "common_options.json"
        self.common_options = self._load_common_options(options_path)

        # OpenAI 클라이언트
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = OpenAI(api_key=api_key) if api_key else None

    def _load_questions(self, path: Path) -> Dict[str, Dict]:
        """질문 데이터 로드 (module2.json 형식 지원)"""
        if not path.exists():
            print(f"Warning: Questions file not found at {path}")
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = {}

        # module2.json 형식: {"questions": {"2.1": {...}, "2.2": {...}}}
        if "questions" in data and isinstance(data["questions"], dict):
            for qid, q_data in data["questions"].items():
                q_data["question_id"] = qid
                questions[qid] = q_data
        # 기존 형식: {"questions": [{...}, {...}]}
        elif "questions" in data and isinstance(data["questions"], list):
            for q in data["questions"]:
                qid = q.get("question_id")
                if qid:
                    questions[qid] = q

        print(f"Loaded {len(questions)} questions from {path}")
        return questions

    def _load_common_options(self, path: Path) -> Dict[str, list]:
        """공통 옵션 로드 (select 필드 옵션 참조용)"""
        if not path.exists():
            print(f"Warning: Common options file not found at {path}")
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded common options: {list(data.keys())}")
        return data

    def _get_question(self, question_id: str) -> Dict[str, Any]:
        """질문 정보 조회 (options_ref를 실제 옵션으로 변환)"""
        import copy
        question = self.questions.get(question_id, {"question_id": question_id, "title": question_id})

        # 깊은 복사하여 원본 수정 방지
        question = copy.deepcopy(question)

        # columns의 options_ref를 실제 옵션으로 변환
        columns = question.get("columns", [])
        for col in columns:
            if isinstance(col, dict) and "options_ref" in col:
                ref_key = col["options_ref"]
                if ref_key in self.common_options:
                    col["options"] = self.common_options[ref_key]

        return question

    def _get_guidance(self, question_id: str) -> Dict[str, Any]:
        """CDP 가이드라인 조회"""
        question = self._get_question(question_id)
        return {
            "rationale": question.get("rationale", ""),
            "requested_content": question.get("requested_content", []),
            "best_practices": question.get("best_practices", ""),
        }

    def _call_llm(self, prompt: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """LLM 호출"""
        if not self.llm:
            # LLM 미설정 시에도 RAG 결과는 반환하도록 소프트 폴백
            return {
                "rows": [],
                "rationale_en": "LLM not configured (OPENAI_API_KEY missing). Returned RAG context only.",
                "rationale_ko": "LLM이 설정되지 않아 RAG 컨텍스트만 반환합니다. OPENAI_API_KEY를 설정하세요.",
                "confidence": 0.0,
            }

        try:
            response = self.llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            print(f"LLM 호출 실패: {e}")
            return {
                "rows": [],
                "rationale_en": f"Error: {str(e)}",
                "rationale_ko": f"오류: {str(e)}",
                "confidence": 0.0,
            }

    def _calculate_confidence(
        self,
        has_historical: bool,
        rag_scores: List[float],
        mapping_confidence: str,
    ) -> float:
        """
        신뢰도 계산

        - 과거 답변은 참고용이므로 가중치 낮음 (10%)
        - 현재 RAG 점수가 핵심 (50%)
        - 매핑 신뢰도 반영 (10%)
        """
        base_score = 0.4

        # RAG 점수 반영 (50%) - 현재 데이터가 핵심
        if rag_scores:
            avg_rag_score = sum(rag_scores) / len(rag_scores)
            base_score += avg_rag_score * 0.5

        # 과거 답변 유무 반영 (10%) - 참고용이므로 가중치 낮음
        if has_historical:
            base_score += 0.1

        # 매핑 신뢰도 반영 (최대 10%)
        confidence_bonus = {"high": 0.1, "medium": 0.05, "low": 0.0}
        base_score += confidence_bonus.get(mapping_confidence, 0.0)

        return min(round(base_score, 2), 0.95)

    def _format_sources(
        self,
        historical: List[RAGDocument],
        current: List[RAGDocument],
    ) -> List[Dict[str, Any]]:
        """출처 정보 포맷팅 (chunk_id, page_num 등 메타 포함)"""
        sources = []

        for doc in current:
            sources.append({
                "type": "sustainability_report",
                "year": doc.year,
                "question_code": doc.question_code,
                "chunk_id": doc.chunk_id or "",
                "page_num": doc.page_num if doc.page_num and doc.page_num > 0 else None,
                "section": doc.section,
                "score": round(doc.score, 3),
                "is_primary": True,
                "preview": (doc.text[:200] + "...") if doc.text else None,
            })

        for doc in historical:
            sources.append({
                "type": "cdp_historical",
                "year": doc.year,
                "question_code": doc.question_code,
                "chunk_id": doc.chunk_id or "",
                "page_num": doc.page_num if doc.page_num and doc.page_num > 0 else None,
                "section": doc.section,
                "score": round(doc.score, 3),
                "is_primary": False,
                "note": "Reference only - not used as source of facts",
                "preview": (doc.text[:200] + "...") if doc.text else None,
            })

        return sources

    def _filter_conditional_fields(
        self,
        rows: List[Dict[str, Any]],
        columns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        조건부 필드 필터링

        LLM이 조건을 완벽히 따르지 않을 수 있으므로,
        조건에 맞지 않는 필드를 후처리로 제거

        Args:
            rows: LLM이 생성한 행 데이터
            columns: 질문 스키마의 컬럼 정보 (condition 포함)

        Returns:
            필터링된 행 데이터
        """
        if not columns:
            return rows

        # 컬럼별 조건 맵 생성
        condition_map = {}
        for col in columns:
            col_id = col.get("id")
            condition = col.get("condition")
            if col_id and condition:
                condition_map[col_id] = condition

        filtered_rows = []
        for row in rows:
            if not isinstance(row, dict):
                filtered_rows.append(row)
                continue

            row_columns = row.get("columns", row)
            if not isinstance(row_columns, dict):
                filtered_rows.append(row)
                continue

            # 조건에 맞지 않는 필드 제거
            filtered_columns = {}
            for col_id, value in row_columns.items():
                condition = condition_map.get(col_id)

                # _en, _ko 접미사 처리 (textarea 필드)
                base_col_id = col_id
                if col_id.endswith("_en") or col_id.endswith("_ko"):
                    base_col_id = col_id[:-3]
                    condition = condition_map.get(base_col_id)

                if condition:
                    # any 조건 처리
                    if "any" in condition:
                        any_conditions = condition["any"]
                        condition_met = False
                        for any_cond in any_conditions:
                            field = any_cond.get("field", "")
                            expected_value = any_cond.get("value")
                            not_expected_value = any_cond.get("value_not")

                            field_value = row_columns.get(field)

                            if expected_value is not None and field_value == expected_value:
                                condition_met = True
                                break
                            if not_expected_value is not None and field_value != not_expected_value:
                                condition_met = True
                                break

                        if not condition_met:
                            continue  # 조건 불충족 - 필드 제외
                    else:
                        # 단일 조건 처리
                        field = condition.get("field", "")
                        expected_value = condition.get("value")
                        not_expected_value = condition.get("value_not")

                        field_value = row_columns.get(field)

                        # value 조건: 특정 값이어야 함
                        if expected_value is not None:
                            if field_value != expected_value:
                                continue  # 조건 불충족 - 필드 제외

                        # value_not 조건: 특정 값이 아니어야 함
                        if not_expected_value is not None:
                            if field_value == not_expected_value:
                                continue  # 조건 불충족 - 필드 제외

                # 조건이 없거나 충족되면 포함
                filtered_columns[col_id] = value

            # row 구조 유지
            if "columns" in row:
                filtered_rows.append({
                    **row,
                    "columns": filtered_columns
                })
            else:
                filtered_rows.append(filtered_columns)

        return filtered_rows

    def _clean_text_fields(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        텍스트 필드 정제 (마크다운 제거, 메타 텍스트 제거)

        Args:
            rows: 행 데이터

        Returns:
            정제된 행 데이터
        """
        cleaned_rows = []

        for row in rows:
            if not isinstance(row, dict):
                cleaned_rows.append(row)
                continue

            row_columns = row.get("columns", row)
            if not isinstance(row_columns, dict):
                cleaned_rows.append(row)
                continue

            cleaned_columns = {}
            for col_id, value in row_columns.items():
                if isinstance(value, str):
                    # 마크다운 제거
                    value = strip_markdown(value)
                    # 메타 텍스트 제거
                    value = remove_meta_text(value)
                cleaned_columns[col_id] = value

            # row 구조 유지
            if "columns" in row:
                cleaned_rows.append({
                    **row,
                    "columns": cleaned_columns
                })
            else:
                cleaned_rows.append(cleaned_columns)

        return cleaned_rows

    def generate(self, question_id: str) -> GeneratedAnswer:
        """
        3-Layer 기반 실시간 답변 생성

        흐름:
        1. Mapping Layer: 과거 질문 코드 조회
        2. RAG Layer: 과거 CDP 답변 + 현재 보고서 검색
        3. Prompt Layer: 역할 명시 프롬프트 구성
        4. LLM 생성
        """
        # 질문 정보 로드
        question = self._get_question(question_id)
        guidance = self._get_guidance(question_id)

        # 1. Mapping Layer: 과거 질문 코드 조회
        historical_codes = self.mapper.get_historical_codes_for_rag(question_id)
        historical_years = self.mapper.get_historical_years_for_rag(question_id)
        mapping_confidence = self.mapper.get_mapping_confidence(question_id)

        # 2. RAG Layer: 과거 CDP 답변 검색 (메타데이터 필터)
        historical_answers = []
        if historical_codes and historical_years:
            try:
                historical_answers = self.rag.search_cdp_answers(
                    query=question.get("title", question_id),
                    years=historical_years,
                    question_codes=historical_codes,
                    top_k=3,
                )
            except Exception as e:
                print(f"과거 CDP 답변 검색 실패: {e}")

        # 3. RAG Layer: 현재 지속가능경영보고서 검색
        current_context = []
        try:
            current_context = self.rag.search_sustainability_report(
                query=question.get("title", question_id),
                year=self.current_year,
                top_k=5,
            )
        except Exception as e:
            print(f"지속가능경영보고서 검색 실패: {e}")

        # 4. Prompt Layer: 역할 명시 프롬프트 구성
        prompt = self.prompt_builder.build(
            question=question,
            guidance=guidance,
            historical_answers=historical_answers,
            current_context=current_context,
        )

        # 5. LLM 생성
        raw_answer = self._call_llm(prompt)

        # 6. 후처리: 조건부 필드 필터링 + 텍스트 정제
        raw_rows = raw_answer.get("rows", [])
        columns = question.get("columns", [])

        # 조건부 필드 필터링 (조건에 맞지 않는 필드 제거)
        filtered_rows = self._filter_conditional_fields(raw_rows, columns)

        # 텍스트 정제 (마크다운, 메타 텍스트 제거)
        cleaned_rows = self._clean_text_fields(filtered_rows)

        # rationale 텍스트도 정제
        rationale_en = strip_markdown(remove_meta_text(raw_answer.get("rationale_en", "")))
        rationale_ko = strip_markdown(remove_meta_text(raw_answer.get("rationale_ko", "")))

        # 7. 응답 구조화
        return GeneratedAnswer(
            question_id=question_id,
            question_title=question.get("title", question_id),
            rows=cleaned_rows,
            overall_confidence=self._calculate_confidence(
                has_historical=len(historical_answers) > 0,
                rag_scores=[doc.score for doc in current_context],
                mapping_confidence=mapping_confidence,
            ),
            sources=self._format_sources(historical_answers, current_context),
            rationale_en=rationale_en,
            rationale_ko=rationale_ko,
            historical_reference_used=len(historical_answers) > 0,
            mapping_confidence=mapping_confidence,
        )

    def generate_batch(self, question_ids: List[str]) -> List[GeneratedAnswer]:
        """일괄 답변 생성"""
        return [self.generate(qid) for qid in question_ids]
