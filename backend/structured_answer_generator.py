"""
Schema-Based Structured Answer Generator
=========================================
CDP 질문 스키마에 맞는 구조화된 답변 생성

컬럼 타입별 처리:
- select: 주어진 options 중 하나 선택
- multiselect: 주어진 options 중 여러 개 선택
- textarea: 서술형 텍스트 생성 (RAG 기반)
- number/percentage: 숫자 추출
- table: 여러 행의 구조화된 데이터

Usage:
    from structured_answer_generator import StructuredAnswerGenerator
    from model.sustainability import EnhancedRAGPipeline

    generator = StructuredAnswerGenerator(rag_pipeline)
    answers = generator.generate_all(questions)
    generator.save_results(answers, "output/cdp_structured_answers.json")
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from model.sustainability import EnhancedRAGPipeline, SearchResult
from utils.translator import Translator

load_dotenv()


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


def is_not_found_response(text: str) -> bool:
    """LLM 응답이 '정보를 찾을 수 없음' 메시지인지 확인

    이런 응답은 문서 뷰어에 적용하지 않아야 함
    """
    if not text:
        return False

    text_lower = text.lower()
    not_found_patterns = [
        "not found in the report",
        "was not found",
        "information was not found",
        "could not be found",
        "no information",
        "not available in",
        "not mentioned in",
        "보고서에서 발견되지 않",
        "찾을 수 없습니다",
        "정보가 없습니다",
        "언급되지 않",
    ]
    return any(pattern in text_lower for pattern in not_found_patterns)


@dataclass
class ColumnAnswer:
    """개별 컬럼 답변"""
    column_id: str
    column_type: str
    value: Any
    confidence: float = 0.0
    source_pages: List[int] = field(default_factory=list)


@dataclass
class RowAnswer:
    """테이블 행 답변"""
    row_index: int
    columns: Dict[str, Any]
    confidence: float = 0.0


@dataclass
class StructuredAnswer:
    """구조화된 질문 답변"""
    question_id: str
    question_title: str
    response_type: str
    rows: List[RowAnswer]
    overall_confidence: float = 0.0
    sources: List[Dict[str, Any]] = field(default_factory=list)
    rationale_en: str = ""  # 답변 근거 (영어)
    rationale_ko: str = ""  # 답변 근거 (한국어)
    notes: str = ""


class StructuredAnswerGenerator:
    """스키마 기반 구조화된 답변 생성기"""

    def __init__(
        self,
        rag_pipeline: EnhancedRAGPipeline,
        model: str = "gpt-4o-mini",
    ):
        self.rag = rag_pipeline
        self.model = model

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        self.openai_client = OpenAI(api_key=api_key)
        self.translator = Translator(model=model)

    def _get_column_prompt(self, column: Dict, context: str, row_label: Optional[str] = None) -> str:
        """컬럼 타입별 프롬프트 생성"""
        col_type = column.get("type", "text")
        col_id = column.get("id", "")
        col_header = column.get("header", col_id)
        options = column.get("options", [])

        # row_label 컨텍스트 추가
        row_context = f"\n현재 행: {row_label}" if row_label else ""

        if col_type == "select":
            options_str = "\n".join([f"  - {opt}" for opt in options])
            return f"""컬럼: {col_header}
타입: 단일 선택 (하나만 선택)
선택 가능한 옵션:
{options_str}

컨텍스트를 기반으로 가장 적절한 옵션 하나를 선택하세요.
옵션 중 하나를 정확히 그대로 반환하세요."""

        elif col_type == "multiselect":
            options_str = "\n".join([f"  - {opt}" for opt in options])
            return f"""컬럼: {col_header}
타입: 다중 선택 (여러 개 선택 가능)
선택 가능한 옵션:
{options_str}

컨텍스트를 기반으로 해당되는 옵션들을 모두 선택하세요.
JSON 배열 형식으로 반환하세요. 예: ["옵션1", "옵션2"]"""

        elif col_type == "textarea":
            max_length = column.get("max_length", 2500)
            return f"""컬럼: {col_header}
타입: 서술형 텍스트
최대 길이: {max_length}자

컨텍스트를 기반으로 상세하고 구체적인 답변을 작성하세요.
- 구체적인 데이터, 수치, 사례를 포함하세요
- 출처 페이지를 언급하세요 (예: Page 28)
- 정보가 없으면 "해당 정보를 보고서에서 찾을 수 없습니다"라고 명시하세요"""

        elif col_type in ("number", "percentage"):
            min_val = column.get("min_value", "")
            max_val = column.get("max_value", "")
            return f"""컬럼: {col_header}{row_context}
타입: {'퍼센트' if col_type == 'percentage' else '숫자'}
범위: {min_val} ~ {max_val}

컨텍스트에서 "{row_label if row_label else col_header}"에 해당하는 숫자를 찾아 반환하세요.
숫자만 반환하세요 (단위 제외). 예: 25.5
일반적인 정의를 참고하세요:
- Short-term: 보통 1~3년
- Medium-term: 보통 4~10년
- Long-term: 보통 11~25년
찾을 수 없으면 위 일반적 정의를 사용하세요."""

        else:
            return f"""컬럼: {col_header}
타입: {col_type}

컨텍스트를 기반으로 적절한 값을 반환하세요."""

    def _build_guidance_prompt(self, question_context: Dict) -> str:
        """CDP 가이드라인 정보를 프롬프트로 구성"""
        guidance = question_context.get("guidance_raw", {})
        if not guidance:
            return ""

        parts = []

        # 질문 배경 (rationale)
        if guidance.get("rationale"):
            rationale = guidance["rationale"]
            if isinstance(rationale, list):
                rationale = " ".join(rationale)
            parts.append(f"[질문 배경]\n{rationale[:500]}")

        # 요청 내용 (requested_content)
        if guidance.get("requested_content"):
            content = guidance["requested_content"]
            if isinstance(content, list):
                content = " ".join(content)
            parts.append(f"[요청 내용]\n{content[:500]}")

        # 모범 사례 (ambition)
        if guidance.get("ambition"):
            ambition = guidance["ambition"]
            if isinstance(ambition, list):
                ambition = " ".join(ambition)
            parts.append(f"[모범 사례]\n{ambition[:300]}")

        return "\n\n".join(parts)

    def _generate_column_value(
        self,
        column: Dict,
        context: str,
        question_context: Dict,
        row_label: Optional[str] = None,
    ) -> Union[Any, Dict[str, str]]:
        """개별 컬럼 값 생성

        Returns:
            - textarea: {"en": "...", "ko": "..."} 이중 언어 딕셔너리
            - 기타 타입: 단일 값
        """
        col_type = column.get("type", "text")
        col_header = column.get("header", column.get("id", ""))
        guidance_prompt = self._build_guidance_prompt(question_context)

        # textarea는 영어로 생성 후 한국어 번역
        if col_type == "textarea":
            max_length = column.get("max_length", 2500)

            system_prompt_en = """You are a sustainability expert answering CDP (Carbon Disclosure Project) questions.
Write ONLY the direct answer content. Do NOT include explanations of why the answer is appropriate - that will be provided separately.
- Include specific data, figures, and page references from the context
- Be concise and factual
- Do NOT use markdown formatting (no **, no ---, no # headers)
- If information is not found, state "This information was not found in the report" """

            user_prompt_en = f"""Question: {question_context.get('title', '')}

{guidance_prompt}

Column: {col_header}
Type: Text (max {max_length} characters)

Write ONLY the direct answer. Do NOT explain why this answer is appropriate.
- Include specific data and page references (e.g., Page 28)
- Do NOT use markdown formatting

---
Context (retrieved from sustainability report):
{context[:6000]}
---

Direct Answer (no rationale, no markdown):"""

            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt_en},
                        {"role": "user", "content": user_prompt_en}
                    ],
                    temperature=0.2,
                    max_tokens=1000
                )
                answer_en = response.choices[0].message.content.strip()
                answer_ko = self.translator.to_korean(answer_en)

                # 마크다운 제거
                return {"en": strip_markdown(answer_en), "ko": strip_markdown(answer_ko)}

            except Exception as e:
                print(f"Error generating textarea value: {e}")
                return {"en": "", "ko": ""}

        # 비-textarea 타입은 기존 로직 유지
        col_prompt = self._get_column_prompt(column, context, row_label)

        system_prompt = """당신은 CDP (Carbon Disclosure Project) 질문에 답변하는 지속가능성 전문가입니다.
주어진 컨텍스트와 CDP 가이드라인을 기반으로 정확한 답변을 제공하세요.
- 반드시 컨텍스트의 구체적인 데이터와 페이지 번호를 인용하세요
- 컨텍스트에 없는 정보는 추측하지 마세요"""

        user_prompt = f"""질문: {question_context.get('title', '')}

{guidance_prompt}

{col_prompt}

---
컨텍스트 (지속가능성 보고서에서 검색됨):
{context[:6000]}
---

답변:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )

            answer = response.choices[0].message.content.strip()

            # 타입별 후처리
            if col_type == "select":
                options = column.get("options", [])
                # 정확한 매칭 시도
                for opt in options:
                    if opt.lower() in answer.lower() or answer.lower() in opt.lower():
                        return opt
                # 첫 번째 옵션 반환 (fallback)
                return options[0] if options else answer

            elif col_type == "multiselect":
                # JSON 배열 파싱 시도
                try:
                    if "[" in answer:
                        json_match = re.search(r'\[.*\]', answer, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group())
                except:
                    pass
                # 옵션 매칭
                options = column.get("options", [])
                selected = [opt for opt in options if opt.lower() in answer.lower()]
                return selected if selected else []

            elif col_type in ("number", "percentage"):
                # 숫자 추출
                numbers = re.findall(r'[-+]?\d*\.?\d+', answer)
                if numbers:
                    return float(numbers[0])
                return None

            else:
                return answer

        except Exception as e:
            print(f"Error generating column value: {e}")
            return None

    def _generate_row(
        self,
        columns: List[Dict],
        context: str,
        question_context: Dict,
        row_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """테이블 행 생성

        textarea 타입은 {col_id}_en, {col_id}_ko로 분리하여 저장
        """
        row_data = {}

        for column in columns:
            col_id = column.get("id", "")
            col_type = column.get("type", "text")

            # 조건부 컬럼 처리
            condition = column.get("condition")
            if condition:
                field = condition.get("field", "")
                expected_value = condition.get("value")
                not_expected_value = condition.get("value_not")

                # value 조건: 특정 값이어야 함
                if expected_value is not None:
                    if field in row_data and row_data[field] != expected_value:
                        continue

                # value_not 조건: 특정 값이 아니어야 함
                if not_expected_value is not None:
                    if field in row_data and row_data[field] == not_expected_value:
                        continue

            # row_label이 있고 해당 컬럼이 시간 범위 등의 선택이면 row_label 사용
            if row_label and col_type == "select" and col_id in ("time_horizon", "row_type"):
                row_data[col_id] = row_label
            else:
                value = self._generate_column_value(column, context, question_context, row_label)

                # textarea는 _en, _ko로 분리하여 저장
                if col_type == "textarea" and isinstance(value, dict):
                    row_data[f"{col_id}_en"] = value.get("en", "")
                    row_data[f"{col_id}_ko"] = value.get("ko", "")
                else:
                    row_data[col_id] = value

        return row_data

    def _generate_rationale(
        self,
        question: Dict,
        rows: List[RowAnswer],
        search_results: list,
    ) -> Dict[str, str]:
        """답변에 대한 전체 근거 생성 (이중 언어)

        Returns:
            {"en": "English rationale", "ko": "한국어 근거"}
        """
        title = question.get("title", "")
        guidance = question.get("guidance_raw", {})

        # 출처 페이지 요약
        source_pages = list(set(r.page_num for r in search_results[:5]))
        source_summary_en = f"Reference pages: {', '.join(map(str, sorted(source_pages)))}"

        # 답변 요약
        answer_summary = json.dumps(
            {k: v for row in rows for k, v in row.columns.items()},
            ensure_ascii=False,
            default=str
        )[:500]

        # 영어로 먼저 생성
        prompt_en = f"""Write a rationale for the following CDP question answer.

Question: {title}

[CDP Guideline - Question Background]
{str(guidance.get('rationale', ''))[:300]}

[Generated Answer]
{answer_summary}

[Retrieved Sources]
{source_summary_en}

Based on the above information, explain in 2-3 sentences "why this answer is appropriate".
Be sure to mention the source page numbers."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at writing clear and concise rationales for CDP answers."},
                    {"role": "user", "content": prompt_en}
                ],
                temperature=0.3,
                max_tokens=300
            )
            rationale_en = response.choices[0].message.content.strip()

            # 한국어로 번역
            rationale_ko = self.translator.to_korean(rationale_en)

            # 마크다운 제거
            return {"en": strip_markdown(rationale_en), "ko": strip_markdown(rationale_ko)}

        except Exception as e:
            fallback_en = f"Rationale generation failed: {e}. {source_summary_en}"
            return {"en": fallback_en, "ko": f"근거 생성 실패: {e}. 참조 페이지: {', '.join(map(str, sorted(source_pages)))}"}

    def generate_answer(
        self,
        question: Dict,
        top_k: int = 10,
        min_score: float = 0.35,
        min_results: int = 3,
    ) -> StructuredAnswer:
        """단일 질문에 대한 구조화된 답변 생성

        Args:
            question: CDP 질문 딕셔너리
            top_k: 검색할 최대 청크 수
            min_score: 최소 rerank 점수 (이 이상만 사용)
            min_results: 최소 결과 수 (점수가 낮아도 이 개수는 보장)
        """
        question_id = question.get("question_id", "")
        title = question.get("title", "")
        columns = question.get("columns", [])
        row_labels = question.get("row_labels", [])
        response_type = question.get("response_type", "table")

        print(f"  Processing: {question_id} - {title[:40]}...")

        # RAG 검색 (더 많이 가져와서 필터링)
        search_query = f"{title} {question.get('guidance_raw', {}).get('rationale', '')}"
        all_results = self.rag.search(
            search_query,
            top_k=top_k,
            use_rerank=True,
            expand_query=True,
            question_context=question,
        )

        # 유사도 기반 필터링 (rerank_score 사용)
        filtered_results = [
            r for r in all_results
            if r.rerank_score and r.rerank_score >= min_score
        ]

        # 최소 결과 수 보장
        if len(filtered_results) < min_results:
            filtered_results = all_results[:min_results]

        search_results = filtered_results
        print(f"    → {len(all_results)}개 검색 → {len(search_results)}개 사용 (min_score={min_score})")

        # 컨텍스트 구성
        context = "\n\n---\n\n".join([
            f"[Page {r.page_num}] (score: {r.rerank_score:.2f})\n{r.content}"
            for r in search_results
        ])

        # 행 생성
        rows = []

        if row_labels:
            # row_labels가 있으면 각 레이블에 대해 행 생성
            for idx, label in enumerate(row_labels):
                row_data = self._generate_row(columns, context, question, row_label=label)
                rows.append(RowAnswer(
                    row_index=idx,
                    columns=row_data,
                    confidence=0.7,
                ))
        else:
            # 단일 행 생성
            row_data = self._generate_row(columns, context, question)
            rows.append(RowAnswer(
                row_index=0,
                columns=row_data,
                confidence=0.7,
            ))

        # 신뢰도 계산
        avg_score = sum(r.score for r in search_results) / len(search_results) if search_results else 0
        avg_rerank = sum(r.rerank_score for r in search_results if r.rerank_score) / len(search_results) if search_results else 0
        overall_confidence = min((avg_score * 0.4 + avg_rerank * 0.6) * 1.2, 1.0)

        # 전체 답변 근거 생성
        rationale = self._generate_rationale(question, rows, search_results)

        return StructuredAnswer(
            question_id=question_id,
            question_title=title,
            response_type=response_type,
            rows=rows,
            overall_confidence=round(overall_confidence, 2),
            sources=[{
                "chunk_id": r.chunk_id,
                "page_num": r.page_num,
                "score": round(r.score, 3),
                "rerank_score": round(r.rerank_score, 3) if r.rerank_score else None,
            } for r in search_results[:5]],
            rationale_en=rationale["en"],
            rationale_ko=rationale["ko"],
        )

    def generate_all(
        self,
        questions: List[Dict],
        top_k: int = 7,
    ) -> List[StructuredAnswer]:
        """모든 질문에 대한 구조화된 답변 생성"""
        answers = []

        # 평면화
        def flatten(qs):
            flat = []
            for q in qs:
                flat.append(q)
                if q.get('children'):
                    flat.extend(flatten(q['children']))
            return flat

        flat_questions = flatten(questions)
        print(f"\nGenerating structured answers for {len(flat_questions)} questions...")

        for question in tqdm(flat_questions, desc="Generating"):
            answer = self.generate_answer(question, top_k=top_k)
            answers.append(answer)

        return answers

    def save_results(
        self,
        answers: List[StructuredAnswer],
        output_path: str,
    ) -> Dict:
        """결과 저장"""
        result = {
            "total_questions": len(answers),
            "average_confidence": round(
                sum(a.overall_confidence for a in answers) / len(answers), 2
            ) if answers else 0,
            "answers": [
                {
                    "question_id": a.question_id,
                    "question_title": a.question_title,
                    "response_type": a.response_type,
                    "rows": [
                        {
                            "row_index": r.row_index,
                            "columns": r.columns,
                            "confidence": r.confidence,
                        }
                        for r in a.rows
                    ],
                    "overall_confidence": a.overall_confidence,
                    "rationale_en": a.rationale_en,
                    "rationale_ko": a.rationale_ko,
                    "sources": a.sources,
                }
                for a in answers
            ]
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(answers)} structured answers to {output_path}")
        return result


def run_structured_generation(
    report_pdf: str,
    questions_json: str,
    output_path: str = "output/cdp_structured_answers.json",
    sample: int = None,
    recreate_index: bool = False,
):
    """구조화된 답변 생성 파이프라인"""
    from model.sustainability import SustainabilityReportChunker, EnhancedRAGPipeline

    print("="*60)
    print("Structured Answer Generation Pipeline")
    print("="*60)

    # 1. 청킹
    print("\nStep 1: Chunking report...")
    chunker = SustainabilityReportChunker(report_pdf)
    chunks = chunker.parse()
    print(f"  Created {len(chunks)} chunks")

    # 2. RAG 초기화
    print("\nStep 2: Initializing RAG...")
    rag = EnhancedRAGPipeline(
        embedding_model="bge-m3",
        use_bm25=True,
        use_reranker=True,
    )
    rag.create_collection(recreate=recreate_index)
    rag.index_chunks(chunks)

    # 3. 질문 로드
    print("\nStep 3: Loading questions...")
    with open(questions_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data.get('questions', data)
    if isinstance(questions, dict):
        questions = [questions]

    if sample:
        questions = questions[:sample]

    print(f"  Loaded {len(questions)} questions")

    # 4. 구조화된 답변 생성
    print("\nStep 4: Generating structured answers...")
    generator = StructuredAnswerGenerator(rag)
    answers = generator.generate_all(questions)

    # 5. 저장
    print("\nStep 5: Saving results...")
    result = generator.save_results(answers, output_path)

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"  Output: {output_path}")
    print(f"  Total questions: {result['total_questions']}")
    print(f"  Average confidence: {result['average_confidence']:.2%}")

    return answers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Structured Answer Generator")
    parser.add_argument("--report", "-r", required=True, help="Sustainability report PDF")
    parser.add_argument("--questions", "-q", required=True, help="CDP questions JSON")
    parser.add_argument("--output", "-o", default="output/cdp_structured_answers.json")
    parser.add_argument("--sample", "-n", type=int, help="Sample questions count")
    parser.add_argument("--recreate-index", action="store_true")

    args = parser.parse_args()

    run_structured_generation(
        args.report,
        args.questions,
        args.output,
        sample=args.sample,
        recreate_index=args.recreate_index,
    )
