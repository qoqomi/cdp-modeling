"""
Prompt Builder - 역할 명시 프롬프트 구성

핵심 원칙:
- 과거 답변의 역할을 명확히 지시
- "참고자료일 뿐" 명시
- 연도 불일치 표현 제거 지시
- 현재 시점 기준 재구성 지시

금지사항:
- "관련 문서를 참고해서 답변해줘" 같은 모호한 프롬프트 금지
"""

from typing import Dict, List, Optional, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from rag.document_schema import RAGDocument


class PromptBuilder:
    """
    역할 명시 프롬프트 빌더

    금지사항:
    - Prompt에서 연도 언급 통제 안 함 금지
    """

    SYSTEM_PROMPT = """You are a CDP (Carbon Disclosure Project) disclosure expert.
Your task is to generate answers for the CURRENT YEAR's CDP questionnaire.

CRITICAL RULES - YOU MUST FOLLOW:
1. Historical CDP answers are REFERENCE ONLY - DO NOT copy them directly
2. Remove any year-specific expressions from historical answers (e.g., "In 2023" → "Currently")
3. Use ONLY current year data from sustainability reports for facts/numbers
4. Generalize past policies to current context
5. Exclude content that doesn't meet current scoring criteria
6. All numerical data must come from the current sustainability report

OUTPUT FORMAT:
- Respond in JSON format matching the provided schema
- Include rationale in both English and Korean
"""

    HISTORICAL_CONTEXT_TEMPLATE = """
## HISTORICAL REFERENCE (참고용 - 그대로 사용 금지!)

⚠️ WARNING: The following are PAST CDP answers for REFERENCE ONLY.
DO NOT copy these directly. Use them ONLY for:
- Understanding the expected format/structure
- Identifying relevant topics to cover
- Learning the appropriate tone/style

{historical_answers}

⚠️ CRITICAL REMINDERS:
- Remove year-specific dates (e.g., "In 2023" → "Currently")
- DO NOT use past numbers/metrics - use current sustainability report data
- Update any outdated policies to current context
- These are HISTORICAL - verify all facts against current data
"""

    CURRENT_CONTEXT_TEMPLATE = """
## CURRENT DATA SOURCE (현재 데이터 - 사실 기반)

✅ The following is from the {year} Sustainability Report.
Use this as the PRIMARY and ONLY source for facts, numbers, and current policies.

{sustainability_content}

✅ This is your authoritative source for:
- Current metrics and numbers
- Current policies and practices
- Current organizational structure
"""

    GUIDANCE_TEMPLATE = """
## CDP GUIDANCE (작성 가이드라인)

Question Rationale:
{rationale}

Requested Content:
{requested_content}

Best Practices:
{best_practices}
"""

    QUESTION_TEMPLATE = """
## CURRENT YEAR QUESTION (현재 연도 질문)

Question ID: {question_id}
Question: {question_title}

Table Columns (응답 테이블 컬럼):
{columns_description}

Row Labels (행 레이블):
{row_labels}
"""

    OUTPUT_INSTRUCTION_TEMPLATE = """
## OUTPUT INSTRUCTION

⚠️⚠️⚠️ CRITICAL RULES - READ CAREFULLY ⚠️⚠️⚠️

You must generate ACTUAL ANSWER VALUES for the SPECIFIC COLUMNS defined above!

RULE 1: Use column "id" as the KEY, put your ANSWER as the VALUE
RULE 2: For "select" type columns, pick ONE value from the "options" list
RULE 3: For "number" type columns, provide a NUMBER (not 0 for all rows!)
RULE 4: For "text/textarea" type columns, provide your written ANSWER
RULE 5: ONLY use the column IDs specified in the "Table Columns" section above

📅 TIME HORIZON VALUES (for question 2.1):
- Short-term: from_years=0, to_years=1 (or up to 3)
- Medium-term: from_years=1-3, to_years=3-10
- Long-term: from_years=5-10, to_years=10-30+
⚠️ IMPORTANT: Each row must have DIFFERENT from_years and to_years values!

{dynamic_example}

❌❌❌ ABSOLUTELY WRONG - NEVER DO THIS:
{{
  "rows": [
    {{
      "columns": {{
        "id": "column_name",         // ❌ WRONG! This is copying the schema!
        "header": "Column Header",   // ❌ WRONG! Don't include "header"!
        "type": "select",            // ❌ WRONG! Don't include "type"!
        "options": ["Option1"]       // ❌ WRONG! Don't include "options"!
      }}
    }}
  ]
}}

Requirements:
1. Column ID becomes the KEY, your answer becomes the VALUE
2. Create one row per row_label if row_labels are provided, otherwise create appropriate rows based on the question context
3. All facts/numbers MUST come from the sustainability report or reasonable estimates
4. Provide rationale in both English and Korean
5. ONLY include columns that are defined in the Table Columns section above - DO NOT add columns from other questions
"""

    def __init__(self):
        pass

    def build(
        self,
        question: Dict[str, Any],
        guidance: Dict[str, Any],
        historical_answers: List[RAGDocument],
        current_context: List[RAGDocument],
    ) -> str:
        """
        3-Layer 기반 프롬프트 구성

        Args:
            question: 현재 질문 정보
            guidance: CDP 가이드라인
            historical_answers: 과거 CDP 답변 (RAG 검색 결과)
            current_context: 현재 지속가능경영보고서 (RAG 검색 결과)

        Returns:
            구성된 프롬프트 문자열
        """
        prompt_parts = [self.SYSTEM_PROMPT]

        # CDP 가이드라인
        prompt_parts.append(self._build_guidance_section(guidance))

        # 과거 답변 (참고용 - 역할 명시)
        if historical_answers:
            historical_text = self._format_historical_answers(historical_answers)
            prompt_parts.append(
                self.HISTORICAL_CONTEXT_TEMPLATE.format(
                    historical_answers=historical_text
                )
            )

        # 현재 지속가능경영보고서 (사실 기반)
        if current_context:
            current_text = self._format_current_context(current_context)
            year = current_context[0].year if current_context else 2024
            prompt_parts.append(
                self.CURRENT_CONTEXT_TEMPLATE.format(
                    year=year,
                    sustainability_content=current_text
                )
            )

        # 현재 질문
        prompt_parts.append(self._build_question_section(question))

        # 출력 지시 (질문별 동적 예시 포함)
        dynamic_example = self._build_dynamic_example(question)
        prompt_parts.append(
            self.OUTPUT_INSTRUCTION_TEMPLATE.format(dynamic_example=dynamic_example)
        )

        return "\n\n".join(prompt_parts)

    def _build_dynamic_example(self, question: Dict[str, Any]) -> str:
        """
        질문별 동적 예시 생성

        각 질문의 컬럼 스키마에 맞는 예시를 생성하여
        LLM이 올바른 형식으로 답변하도록 유도
        """
        columns = question.get("columns", [])
        row_labels = question.get("row_labels", [])
        question_id = question.get("question_id", "")

        if not columns:
            return "CORRECT FORMAT: Generate rows with appropriate column values based on the question."

        # 컬럼 ID 목록 추출
        col_ids = []
        col_examples = {}
        col_example_by_row = {}  # row_label별 예시값 저장

        for col in columns:
            if isinstance(col, dict):
                col_id = col.get("id", "unknown")
                col_type = col.get("type", "text")
                options = col.get("options", [])
                example_by_row = col.get("example_by_row")  # JSON에서 예시값 읽기

                col_ids.append(col_id)

                # example_by_row가 있으면 저장
                if example_by_row:
                    col_example_by_row[col_id] = example_by_row

                # 타입별 예시 값 생성
                if col_type == "select" and options:
                    col_examples[col_id] = f'"{options[0]}"'
                elif col_type == "multiselect" and options:
                    col_examples[col_id] = f'["{options[0]}"]'
                elif col_type == "number":
                    # example_by_row가 있으면 첫 번째 row_label 값 사용
                    if example_by_row and row_labels:
                        first_label = row_labels[0]
                        col_examples[col_id] = str(example_by_row.get(first_label, 5))
                    else:
                        col_examples[col_id] = "5"
                elif col_type in ("text", "textarea"):
                    col_examples[col_id] = f'"Your answer for {col_id} here"'
                else:
                    col_examples[col_id] = f'"value_for_{col_id}"'

        # 행 개수 결정
        num_rows = len(row_labels) if row_labels else 1

        rows_example = []
        for i in range(min(num_rows, 3)):  # 최대 3개 행 예시로 표시
            row_label_comment = f" // Row for: {row_labels[i]}" if row_labels and i < len(row_labels) else ""
            row_label = row_labels[i] if row_labels and i < len(row_labels) else ""

            # example_by_row가 있는 컬럼들의 값을 row_label에 맞게 오버라이드
            row_col_examples = dict(col_examples)
            for col_id, example_map in col_example_by_row.items():
                if row_label in example_map:
                    row_col_examples[col_id] = str(example_map[row_label])

            # select 타입 컬럼 중 options에 row_label이 있으면 해당 값 사용
            for col in columns:
                if isinstance(col, dict):
                    col_id = col.get("id")
                    col_type = col.get("type")
                    options = col.get("options", [])
                    if col_type == "select" and row_label in options:
                        row_col_examples[col_id] = f'"{row_label}"'

            example_columns = ", ".join([f'"{cid}": {row_col_examples[cid]}' for cid in col_ids])

            rows_example.append(f"""    {{
      "row_index": {i},{row_label_comment}
      "columns": {{ {example_columns} }},
      "confidence": 0.8
    }}""")

        rows_str = ",\n".join(rows_example)
        if num_rows > 2:
            rows_str += f"\n    // ... (create {num_rows} rows total, one for each row_label)"

        return f"""CORRECT FORMAT for Question {question_id} - Each row must contain these columns: {col_ids}
{{
  "rows": [
{rows_str}
  ],
  "rationale_en": "Explanation of your answer in English...",
  "rationale_ko": "답변에 대한 한국어 설명...",
  "confidence": 0.8
}}

✅ CORRECT column keys for this question: {col_ids}
❌ DO NOT use column keys from other questions!"""

    def _build_guidance_section(self, guidance: Dict[str, Any]) -> str:
        """CDP 가이드라인 섹션 구성"""
        rationale = guidance.get("rationale", "N/A")
        requested_content = guidance.get("requested_content", [])
        best_practices = guidance.get("best_practices", "Follow CDP scoring methodology")

        if isinstance(requested_content, list):
            requested_content = "\n".join(f"- {item}" for item in requested_content)

        return self.GUIDANCE_TEMPLATE.format(
            rationale=rationale,
            requested_content=requested_content,
            best_practices=best_practices,
        )

    def _build_question_section(self, question: Dict[str, Any]) -> str:
        """질문 섹션 구성"""
        question_id = question.get("question_id", "N/A")
        question_title = question.get("title", question.get("title_en", "N/A"))
        columns = question.get("columns", [])
        row_labels = question.get("row_labels", [])

        # 컬럼 설명 포맷팅 - 각 컬럼의 id, header, type, options 포함
        columns_description = []
        for col in columns:
            if isinstance(col, dict):
                col_id = col.get("id", "unknown")
                col_header = col.get("header", col_id)
                col_type = col.get("type", "text")
                options = col.get("options", [])

                desc = f"- {col_id} ({col_header}): type={col_type}"
                if options:
                    desc += f", options={options}"
                columns_description.append(desc)
            else:
                columns_description.append(f"- {col}")

        columns_str = "\n".join(columns_description) if columns_description else "N/A"

        # 행 레이블 포맷팅
        if isinstance(row_labels, list) and row_labels:
            row_labels_str = "\n".join(f"- Row {i}: {label}" for i, label in enumerate(row_labels))
        else:
            row_labels_str = "Dynamic rows (no fixed labels)"

        return self.QUESTION_TEMPLATE.format(
            question_id=question_id,
            question_title=question_title,
            columns_description=columns_str,
            row_labels=row_labels_str,
        )

    def _format_historical_answers(self, docs: List[RAGDocument]) -> str:
        """
        과거 답변 포맷팅 - 연도 표시 필수

        주의: 과거 답변임을 명확히 표시하여 LLM이 그대로 복사하지 않도록
        """
        formatted = []
        for doc in docs:
            formatted.append(f"""
--- Historical Answer from {doc.year} (REFERENCE ONLY) ---
Question Code: {doc.question_code}
Module: {doc.module or 'N/A'}
Answer:
{doc.text}

[⚠️ Source: {doc.year} CDP Response - DO NOT COPY DIRECTLY]
""")
        return "\n".join(formatted)

    def _format_current_context(self, docs: List[RAGDocument]) -> str:
        """
        현재 지속가능경영보고서 포맷팅

        이 데이터가 사실 기반의 유일한 소스임을 강조
        """
        formatted = []
        for doc in docs:
            page_info = f"Page {doc.page_num}" if doc.page_num else "N/A"
            section_info = doc.section or "General"
            formatted.append(f"""
--- Current Report Data ---
Section: {section_info}
Page: {page_info}
Content:
{doc.text}

[✅ Source: {doc.year} Sustainability Report - USE THIS FOR FACTS]
""")
        return "\n".join(formatted)

    def build_without_historical(
        self,
        question: Dict[str, Any],
        guidance: Dict[str, Any],
        current_context: List[RAGDocument],
    ) -> str:
        """
        과거 답변 없이 프롬프트 구성 (Zero-Shot)

        과거 답변이 없는 경우 CDP 가이드라인을 더 강조
        """
        return self.build(
            question=question,
            guidance=guidance,
            historical_answers=[],  # 빈 리스트
            current_context=current_context,
        )
