"""
CDP Answer Generator
====================
스키마 기반 CDP 응답 생성기

RAG 파이프라인에서 검색된 컨텍스트와 스키마를 결합하여
CDP 형식에 맞는 구조화된 응답을 생성합니다.

사용 방법:
    python cdp_answer_generator.py --questions data/cdp_questions_parsed.json --report data/2025_SK-Inc_Sustainability_Report_ENG.pdf
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from cdp_schema_loader import CDPSchemaLoader
from cdp_rag_pipeline import CDPRAGPipeline, SustainabilityReportParser, SearchResult, flatten_questions

load_dotenv()


class FieldType(str, Enum):
    SELECT = "select"
    MULTISELECT = "multiselect"
    TEXT = "text"
    TEXTAREA = "textarea"
    NUMBER = "number"
    PERCENTAGE = "percentage"
    TABLE = "table"
    GROUPED_SELECT = "grouped_select"
    ATTACHMENT = "attachment"


@dataclass
class StructuredAnswer:
    """구조화된 CDP 응답"""
    question_id: str
    question_title: str
    response: Dict[str, Any]  # 컬럼별 응답
    confidence: float
    sources: List[Dict[str, Any]]
    validation_errors: List[str] = field(default_factory=list)
    raw_llm_response: Optional[str] = None


class CDPAnswerGenerator:
    """스키마 기반 CDP 응답 생성기"""

    def __init__(
        self,
        schema_version: str = "2025",
        model: str = "gpt-4o-mini"
    ):
        self.schema_loader = CDPSchemaLoader(version=schema_version)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _build_schema_prompt(self, schema: Dict) -> str:
        """스키마를 프롬프트 형식으로 변환"""
        columns = schema.get("columns", [])
        if not columns:
            return "No specific response format required. Provide a comprehensive text response."

        prompt_parts = ["응답 형식 (각 컬럼에 맞게 JSON으로 응답하세요):\n"]

        for col in columns:
            col_id = col.get("id", "unknown")
            col_name = col.get("name", col.get("field", ""))
            col_name_ko = col.get("name_ko", "")
            col_type = col.get("type", "text")
            required = col.get("required", False)

            # 컬럼 설명 구성
            col_desc = f"\n- {col_id}: {col_name}"
            if col_name_ko:
                col_desc += f" ({col_name_ko})"
            col_desc += f"\n  타입: {col_type}"
            if required:
                col_desc += " [필수]"

            # 타입별 추가 정보
            if col_type == "select":
                options = col.get("options", [])
                if options:
                    col_desc += f"\n  선택 옵션: {options}"

            elif col_type == "multiselect":
                options = col.get("options", [])
                if options:
                    col_desc += f"\n  선택 옵션 (복수 선택 가능): {options}"

            elif col_type == "grouped_select":
                grouped = col.get("grouped_options", {})
                if grouped:
                    col_desc += f"\n  그룹별 옵션:"
                    for group, opts in grouped.items():
                        col_desc += f"\n    {group}: {opts[:3]}..." if len(opts) > 3 else f"\n    {group}: {opts}"

            elif col_type == "textarea":
                max_length = col.get("max_length")
                if max_length:
                    col_desc += f"\n  최대 글자 수: {max_length}"

            elif col_type in ("number", "percentage"):
                min_val = col.get("min_value")
                max_val = col.get("max_value")
                if min_val is not None or max_val is not None:
                    col_desc += f"\n  범위: {min_val or ''} ~ {max_val or ''}"

            # 조건부 표시
            condition = col.get("condition")
            if condition:
                col_desc += f"\n  조건: {condition}"

            prompt_parts.append(col_desc)

        return "\n".join(prompt_parts)

    def _build_response_template(self, schema: Dict) -> Dict[str, Any]:
        """응답 템플릿 생성"""
        template = {}
        columns = schema.get("columns", [])

        for col in columns:
            col_id = col.get("id", "unknown")
            col_type = col.get("type", "text")

            if col_type == "select":
                template[col_id] = "선택할 옵션"
            elif col_type == "multiselect":
                template[col_id] = ["선택 옵션1", "선택 옵션2"]
            elif col_type == "grouped_select":
                template[col_id] = {"group": "그룹명", "value": "선택값"}
            elif col_type == "textarea":
                template[col_id] = "텍스트 응답"
            elif col_type in ("number", "percentage"):
                template[col_id] = 0
            else:
                template[col_id] = "응답"

        return template

    def generate_structured_answer(
        self,
        question: Dict[str, Any],
        context_chunks: List[SearchResult]
    ) -> StructuredAnswer:
        """구조화된 응답 생성"""

        question_id = question.get("question_id", "N/A")

        # 스키마 로드
        schema = self.schema_loader.get_resolved_schema(question_id)

        # 컨텍스트 구성
        context_text = "\n\n---\n\n".join([
            f"[Source: Page {chunk.page_num}]\n{chunk.content}"
            for chunk in context_chunks
        ])

        # 스키마 프롬프트 구성
        if schema:
            schema_prompt = self._build_schema_prompt(schema)
            response_template = self._build_response_template(schema)
        else:
            schema_prompt = "자유 형식으로 응답하세요."
            response_template = {"answer": "응답 내용"}

        # 시스템 프롬프트
        system_prompt = """당신은 CDP(Carbon Disclosure Project) 설문 응답을 작성하는 지속가능성 전문가입니다.

지속가능성 보고서에서 제공된 정보를 바탕으로 CDP 질문에 답변합니다.

규칙:
1. 반드시 제공된 컨텍스트 정보만 사용하세요
2. 정보가 없으면 "정보 없음" 또는 적절한 기본값을 사용하세요
3. 응답은 반드시 지정된 JSON 형식으로 제공하세요
4. select/multiselect 타입은 반드시 주어진 옵션 중에서만 선택하세요
5. textarea는 최대 글자 수를 준수하세요
6. 숫자 필드는 적절한 숫자 값을 입력하세요"""

        # 사용자 프롬프트
        user_prompt = f"""CDP 질문 ID: {question_id}
질문: {question.get('title_en', question.get('title', 'N/A'))}

질문 의도 (Rationale):
{question.get('rationale', 'N/A')}

요청 내용 (Requested Content):
{json.dumps(question.get('requested_content', []), ensure_ascii=False, indent=2)}

---

{schema_prompt}

---

응답 템플릿 예시:
```json
{json.dumps(response_template, ensure_ascii=False, indent=2)}
```

---

참고 문서 (지속가능성 보고서):

{context_text}

---

위 정보를 바탕으로 JSON 형식의 응답을 생성하세요.
응답은 반드시 유효한 JSON 객체여야 합니다.
```json
"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )

            raw_response = response.choices[0].message.content

            # JSON 파싱
            parsed_response = self._parse_json_response(raw_response)

            # 검증
            validation_errors = []
            if schema:
                validation_errors = self.schema_loader.validate_response(
                    question_id, parsed_response
                )

            # 신뢰도 계산
            avg_score = sum(c.score for c in context_chunks) / len(context_chunks) if context_chunks else 0
            confidence = min(avg_score * 1.2, 1.0)

            # 검증 오류가 있으면 신뢰도 감소
            if validation_errors:
                confidence *= 0.7

            return StructuredAnswer(
                question_id=question_id,
                question_title=question.get('title_en', question.get('title', 'N/A')),
                response=parsed_response,
                confidence=round(confidence, 2),
                sources=[{
                    "chunk_id": c.chunk_id,
                    "page_num": c.page_num,
                    "score": round(c.score, 3),
                    "section": c.section,
                    "preview": c.content[:200] + "..." if len(c.content) > 200 else c.content
                } for c in context_chunks],
                validation_errors=validation_errors,
                raw_llm_response=raw_response
            )

        except Exception as e:
            print(f"Error generating answer for {question_id}: {e}")
            return StructuredAnswer(
                question_id=question_id,
                question_title=question.get('title_en', question.get('title', 'N/A')),
                response={"error": str(e)},
                confidence=0.0,
                sources=[],
                validation_errors=[f"Generation error: {str(e)}"]
            )

    def _parse_json_response(self, raw_response: str) -> Dict[str, Any]:
        """LLM 응답에서 JSON 파싱"""
        # JSON 블록 추출
        import re

        # ```json ... ``` 블록 추출
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # ``` ... ``` 블록 추출
            json_match = re.search(r'```\s*(.*?)\s*```', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # { ... } 직접 추출
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return {"raw_answer": raw_response}

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"raw_answer": raw_response}

    def process_all_questions(
        self,
        questions: List[Dict[str, Any]],
        rag_pipeline: CDPRAGPipeline,
        top_k: int = 5
    ) -> List[StructuredAnswer]:
        """모든 질문 처리"""
        answers = []

        for question in tqdm(questions, desc="Generating structured answers"):
            # 검색 쿼리 구성
            search_query = f"{question.get('title_en', '')} {question.get('rationale', '')}"

            # 유사 문서 검색
            relevant_chunks = rag_pipeline.search(search_query, top_k=top_k)

            # 구조화된 답변 생성
            answer = self.generate_structured_answer(question, relevant_chunks)
            answers.append(answer)

        return answers

    def generate_report(
        self,
        answers: List[StructuredAnswer],
        output_path: str
    ) -> Dict:
        """결과 리포트 생성"""
        # 통계 계산
        total = len(answers)
        valid_count = sum(1 for a in answers if not a.validation_errors)
        avg_confidence = sum(a.confidence for a in answers) / total if total else 0

        report = {
            "summary": {
                "total_questions": total,
                "valid_responses": valid_count,
                "invalid_responses": total - valid_count,
                "validation_rate": round(valid_count / total * 100, 1) if total else 0,
                "average_confidence": round(avg_confidence, 2)
            },
            "answers": [asdict(a) for a in answers]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n결과 저장: {output_path}")
        print(f"  총 질문 수: {total}")
        print(f"  유효 응답: {valid_count} ({report['summary']['validation_rate']}%)")
        print(f"  평균 신뢰도: {avg_confidence:.2%}")

        return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CDP Answer Generator (Schema-based)")
    parser.add_argument("--questions", "-q", required=True, help="CDP questions JSON path")
    parser.add_argument("--report", "-r", required=True, help="Sustainability report PDF path")
    parser.add_argument("--output", "-o", default="output/cdp_structured_answers.json", help="Output path")
    parser.add_argument("--schema-version", default="2025", help="Schema version")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--top-k", type=int, default=5, help="Number of similar chunks")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--recreate-index", action="store_true", help="Recreate vector index")

    args = parser.parse_args()

    # 1. 지속가능성 보고서 파싱 및 인덱싱
    print(f"\n{'='*50}")
    print("Step 1: 보고서 파싱 및 인덱싱")
    print(f"{'='*50}")

    report_parser = SustainabilityReportParser(
        args.report,
        chunk_size=args.chunk_size
    )
    chunks = report_parser.create_chunks()
    print(f"청크 {len(chunks)}개 생성")

    # RAG 파이프라인 초기화
    rag_pipeline = CDPRAGPipeline()
    rag_pipeline.create_collection(recreate=args.recreate_index)
    rag_pipeline.index_chunks(chunks)

    # 2. CDP 질문 로드
    print(f"\n{'='*50}")
    print("Step 2: CDP 질문 로드")
    print(f"{'='*50}")

    with open(args.questions, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    questions = questions_data.get('questions', questions_data)
    if isinstance(questions, dict):
        questions = [questions]

    flat_questions = flatten_questions(questions)
    print(f"질문 {len(flat_questions)}개 로드")

    # 3. 스키마 기반 응답 생성
    print(f"\n{'='*50}")
    print("Step 3: 스키마 기반 응답 생성")
    print(f"{'='*50}")

    generator = CDPAnswerGenerator(
        schema_version=args.schema_version,
        model=args.model
    )

    answers = generator.process_all_questions(
        flat_questions,
        rag_pipeline,
        top_k=args.top_k
    )

    # 4. 결과 저장
    print(f"\n{'='*50}")
    print("Step 4: 결과 저장")
    print(f"{'='*50}")

    generator.generate_report(answers, args.output)


if __name__ == "__main__":
    main()
