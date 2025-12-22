"""
CDP Questionnaire Merger (Schema-First + PDF Enrichment)
=========================================================

스키마 파일을 SSOT(Single Source of Truth)로 사용하고,
PDF 파서를 직접 실행하여 텍스트 콘텐츠를 추출하여 병합

Architecture:
    schemas/2025/module2.json (구조/옵션/조건) ← SSOT
            +
    CDPQuestionnaireSectionParser(pdf) ← 텍스트 보강 (직접 실행)
            ↓
    merge_questionnaire.py
            ↓
    최종 JSON (프론트엔드용)

Usage:
    # PDF에서 직접 파싱 (권장)
    python backend/model/merge_questionnaire.py \
        --schema backend/schemas/2025 \
        --pdf backend/data/Full_Corporate_Questionnaire_Module2.pdf \
        --output backend/output/cdp_questions_merged.json

    # 또는 기존 파싱된 JSON 사용 (fallback)
    python backend/model/merge_questionnaire.py \
        --schema backend/schemas/2025 \
        --parsed backend/data/cdp_questions_parsed.json \
        --output backend/output/cdp_questions_merged.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# PDF Parser import - 같은 폴더의 parser 사용
import sys
_backend_dir = Path(__file__).resolve().parents[2]  # model/cdp -> model -> backend
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

# 같은 cdp 패키지 내의 parser import
from .parser import CDPQuestionnaireSectionParser

# Translator import (optional)
try:
    from utils.translator import Translator
    _translator_available = True
except ImportError:
    _translator_available = False
    Translator = None


# ============================================================
# Data Classes
# ============================================================

@dataclass
class QuestionDependency:
    """질문 표시 조건 (다른 질문 답변에 따라)"""
    depends_on_question: str
    depends_on_column: Optional[str] = None
    trigger_values: Optional[List[str]] = None
    raw_text: Optional[str] = None  # 원본 텍스트 보존


@dataclass
class ColumnSchema:
    """컬럼 스키마 (옵션 resolved)"""
    id: str
    header: str
    header_ko: Optional[str] = None
    type: str = "text"
    options: Optional[List[str]] = None
    grouped_options: Optional[Dict[str, List[str]]] = None
    required: bool = False
    condition: Optional[Dict[str, Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_length: Optional[int] = None


@dataclass
class GuidanceRaw:
    """PDF에서 추출한 텍스트 콘텐츠 (영어/한국어)"""
    # English (원본)
    rationale: Optional[str] = None
    ambition: Optional[List[str]] = None
    requested_content: Optional[List[str]] = None
    explanation_of_terms: Optional[Dict[str, str]] = None
    additional_information: Optional[str] = None
    # Korean (번역)
    rationale_ko: Optional[str] = None
    ambition_ko: Optional[List[str]] = None
    requested_content_ko: Optional[List[str]] = None
    additional_information_ko: Optional[str] = None
    # Summary fields (요약)
    rationale_summary: Optional[str] = None
    rationale_ko_summary: Optional[str] = None
    ambition_summary: Optional[List[str]] = None
    ambition_ko_summary: Optional[List[str]] = None
    requested_content_summary: Optional[List[str]] = None
    requested_content_ko_summary: Optional[List[str]] = None


@dataclass
class GuidanceSummary:
    """GPT 생성 가이드 요약"""
    summary_ko: str = ""
    checklist_ko: List[str] = field(default_factory=list)
    column_guides_ko: Optional[Dict[str, str]] = None  # column_id -> 가이드
    condition_note_ko: str = ""


@dataclass
class MergedQuestion:
    """최종 병합된 질문 구조"""
    question_id: str
    title: str
    title_ko: Optional[str] = None

    # 응답 스키마 (Schema-First)
    response_type: str = "table"
    allow_multiple_rows: bool = True
    row_labels: Optional[List[str]] = None
    columns: List[ColumnSchema] = field(default_factory=list)

    # 질문 레벨 의존성
    question_dependency: Optional[QuestionDependency] = None
    sector_specific: Optional[str] = None

    # 텍스트 콘텐츠 (PDF Enrichment)
    guidance_raw: Optional[GuidanceRaw] = None

    # GPT 요약 (Optional)
    guidance_summary: Optional[GuidanceSummary] = None


# ============================================================
# Schema Loader & Resolver
# ============================================================

class SchemaResolver:
    """스키마 옵션 참조 해결"""

    def __init__(self, schema_dir: Path):
        self.schema_dir = schema_dir
        self.common_options: Dict[str, List[str]] = {}
        self.grouped_options: Dict[str, Dict[str, List[str]]] = {}
        self.module_schema: Dict[str, Any] = {}

        self._load_all()

    def _load_json(self, filename: str) -> Dict:
        filepath = self.schema_dir / filename
        if not filepath.exists():
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_all(self):
        """모든 스키마 파일 로드"""
        self.common_options = self._load_json("common_options.json")
        self.grouped_options = self._load_json("grouped_options.json")
        self.module_schema = self._load_json("module2.json")

    def resolve_options(self, options_ref: str) -> Optional[List[str]]:
        """options_ref를 실제 옵션 리스트로 변환"""
        return self.common_options.get(options_ref)

    def resolve_grouped_options(self, ref: str) -> Optional[Dict[str, List[str]]]:
        """grouped_options_ref를 실제 그룹 옵션으로 변환"""
        return self.grouped_options.get(ref)

    def resolve_column(self, col_def: Dict[str, Any]) -> ColumnSchema:
        """컬럼 정의를 ColumnSchema로 변환 (옵션 resolve)"""
        options = None
        grouped_options = None

        # options_ref 해결
        if "options_ref" in col_def:
            options = self.resolve_options(col_def["options_ref"])
        elif "options" in col_def:
            options = col_def["options"]

        # grouped_options_ref 해결
        if "grouped_options_ref" in col_def:
            grouped_options = self.resolve_grouped_options(col_def["grouped_options_ref"])

        return ColumnSchema(
            id=col_def.get("id", ""),
            header=col_def.get("header", ""),
            header_ko=col_def.get("header_ko"),
            type=col_def.get("type", "text"),
            options=options,
            grouped_options=grouped_options,
            required=col_def.get("required", False),
            condition=col_def.get("condition"),
            min_value=col_def.get("min_value"),
            max_value=col_def.get("max_value"),
            max_length=col_def.get("max_length"),
        )

    def get_all_questions(self) -> Dict[str, Dict[str, Any]]:
        """모든 질문 스키마 반환"""
        return self.module_schema.get("questions", {})


# ============================================================
# PDF Parser / Loader
# ============================================================

def _questions_to_dict(questions: List[Any]) -> Dict[str, Dict[str, Any]]:
    """질문 리스트를 question_id 기준 딕셔너리로 변환"""
    by_id: Dict[str, Dict[str, Any]] = {}

    def walk(q):
        # CDPQuestion 객체인 경우 dict로 변환
        if hasattr(q, 'to_dict'):
            q_dict = q.to_dict()
        elif hasattr(q, '__dict__'):
            q_dict = q.__dict__
        else:
            q_dict = q

        by_id[q_dict["question_id"]] = q_dict

        children = q_dict.get("children") or []
        for child in children:
            walk(child)

    for q in questions:
        walk(q)

    return by_id


def parse_pdf_directly(
    pdf_path: Path,
    save_intermediate: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """PDF를 직접 파싱하여 question_id 기준 딕셔너리 반환

    Args:
        pdf_path: PDF 파일 경로
        save_intermediate: 중간 결과 저장 여부
        output_dir: 중간 결과 저장 디렉토리 (None이면 backend/output)
    """
    print(f"  Parsing PDF: {pdf_path}")
    parser = CDPQuestionnaireSectionParser(str(pdf_path))
    questions = parser.parse()
    print(f"  Parsed {len(questions)} top-level questions")

    # 중간 결과 저장 (PDF 파서 결과)
    if save_intermediate:
        if output_dir is None:
            output_dir = _backend_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        intermediate_path = output_dir / f"{pdf_path.stem}_parsed.json"
        parser.to_json(str(intermediate_path))
        print(f"  Saved intermediate: {intermediate_path}")

    return _questions_to_dict(questions)


def load_parsed_questions(parsed_path: Path) -> Dict[str, Dict[str, Any]]:
    """기존 파싱된 JSON 파일에서 로드 (fallback)"""
    if not parsed_path.exists():
        return {}

    with open(parsed_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    return _questions_to_dict(questions)


def clean_guidance_text(text: str, max_length: int = 500) -> str:
    """가이드라인 텍스트 정제

    - 마크다운 특수문자 제거
    - 과도한 줄바꿈 정리
    - 최대 길이 제한 (필요시 요약)
    """
    if not text:
        return ""

    import re

    # --- 구분자 완전 제거 (여러 패턴)
    text = re.sub(r'^\s*---+\s*$', '', text, flags=re.MULTILINE)  # 단독 줄의 ---
    text = re.sub(r'^---+\s*\n?', '', text)  # 문자열 시작의 ---
    text = re.sub(r'\n?---+\s*$', '', text)  # 문자열 끝의 ---
    text = re.sub(r'\n---+\n', '\n', text)   # 중간의 ---\n---

    # 마크다운 제거
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold** -> bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic* -> italic
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # # heading
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)  # - item -> • item
    text = re.sub(r'\n{3,}', '\n\n', text)  # 과도한 줄바꿈
    text = text.strip()

    # 최대 길이 제한 (문장 단위로 자르기)
    if len(text) > max_length:
        # 마지막 온점 위치 찾기
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.5:  # 50% 이상이면 거기서 자르기
            text = truncated[:last_period + 1]
        else:
            text = truncated.rstrip() + "..."

    return text


def summarize_text(text: Optional[str], max_length: int = 200) -> Optional[str]:
    """텍스트 요약 (문장 단위로 자르기)

    - 최대 길이까지 완전한 문장만 포함
    - 문장 중간에서 자르지 않음
    """
    if not text:
        return None

    text = text.strip()
    if len(text) <= max_length:
        return text

    # 문장 구분자로 분리
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) + 1 <= max_length:
            result.append(sentence)
            current_length += len(sentence) + 1
        else:
            break

    if result:
        return ' '.join(result)

    # 문장이 너무 길면 단어 단위로 자르기
    words = text.split()
    result = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_length - 3:  # ... 공간 확보
            result.append(word)
            current_length += len(word) + 1
        else:
            break

    return ' '.join(result) + '...' if result else text[:max_length-3] + '...'


def summarize_list(
    items: Optional[List[str]],
    max_items: int = 2,
    max_item_length: int = 100
) -> Optional[List[str]]:
    """리스트 요약 (상위 N개 항목만, 각 항목 길이 제한)"""
    if not items:
        return None

    result = []
    for item in items[:max_items]:
        if len(item) <= max_item_length:
            result.append(item)
        else:
            # 단어 단위로 자르기
            words = item.split()
            truncated = []
            current_length = 0
            for word in words:
                if current_length + len(word) + 1 <= max_item_length - 3:
                    truncated.append(word)
                    current_length += len(word) + 1
                else:
                    break
            result.append(' '.join(truncated) + '...' if truncated else item[:max_item_length-3] + '...')

    return result if result else None


def extract_guidance_raw(
    parsed_q: Dict[str, Any],
    translator: Optional[Any] = None,
) -> GuidanceRaw:
    """PDF 파서 출력에서 텍스트 콘텐츠 추출

    Args:
        parsed_q: 파싱된 질문 데이터
        translator: Translator 인스턴스 (None이면 번역 안 함)
    """
    # 원본 추출 및 정제
    rationale = clean_guidance_text(parsed_q.get("rationale", ""), max_length=500)
    ambition = parsed_q.get("ambition")
    requested_content = parsed_q.get("requested_content")
    additional_information = clean_guidance_text(parsed_q.get("additional_information", ""), max_length=300)

    # 리스트 항목도 정제 (빈 항목 제거)
    if ambition:
        ambition = [clean_guidance_text(item, max_length=200) for item in ambition]
        ambition = [item for item in ambition if item]  # 빈 문자열 제거
    if requested_content:
        requested_content = [clean_guidance_text(item, max_length=200) for item in requested_content]
        requested_content = [item for item in requested_content if item]  # 빈 문자열 제거

    # 한국어 번역 (translator가 있으면)
    rationale_ko = None
    ambition_ko = None
    requested_content_ko = None
    additional_information_ko = None

    if translator:
        if rationale:
            rationale_ko = clean_guidance_text(translator.to_korean(rationale), max_length=500)
        if ambition:
            ambition_ko = [clean_guidance_text(translator.to_korean(item), max_length=200) for item in ambition]
            ambition_ko = [item for item in ambition_ko if item]  # 빈 문자열 제거
        if requested_content:
            requested_content_ko = [clean_guidance_text(translator.to_korean(item), max_length=200) for item in requested_content]
            requested_content_ko = [item for item in requested_content_ko if item]  # 빈 문자열 제거
        if additional_information:
            additional_information_ko = clean_guidance_text(translator.to_korean(additional_information), max_length=300)

    # 요약 생성
    rationale_summary = summarize_text(rationale, max_length=200)
    rationale_ko_summary = summarize_text(rationale_ko, max_length=200) if rationale_ko else None
    ambition_summary = summarize_list(ambition, max_items=1, max_item_length=100)
    ambition_ko_summary = summarize_list(ambition_ko, max_items=1, max_item_length=100) if ambition_ko else None
    requested_content_summary = summarize_list(requested_content, max_items=2, max_item_length=80)
    requested_content_ko_summary = summarize_list(requested_content_ko, max_items=2, max_item_length=80) if requested_content_ko else None

    return GuidanceRaw(
        rationale=rationale,
        ambition=ambition,
        requested_content=requested_content,
        explanation_of_terms=parsed_q.get("explanation_of_terms"),
        additional_information=additional_information,
        rationale_ko=rationale_ko,
        ambition_ko=ambition_ko,
        requested_content_ko=requested_content_ko,
        additional_information_ko=additional_information_ko,
        # Summary fields
        rationale_summary=rationale_summary,
        rationale_ko_summary=rationale_ko_summary,
        ambition_summary=ambition_summary,
        ambition_ko_summary=ambition_ko_summary,
        requested_content_summary=requested_content_summary,
        requested_content_ko_summary=requested_content_ko_summary,
    )


def parse_question_dependency(parsed_q: Dict[str, Any]) -> Optional[QuestionDependency]:
    """질문 의존성 파싱

    예: "This question only appears if you select 'Yes' in column 3 of 2.2"
    """
    raw_text = parsed_q.get("question_dependencies")
    if not raw_text:
        return None

    # Unicode 따옴표를 ASCII로 변환 (PDF에서 curly quotes 사용)
    # U+201C (LEFT DOUBLE), U+201D (RIGHT DOUBLE), U+2018 (LEFT SINGLE), U+2019 (RIGHT SINGLE)
    normalized_text = raw_text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")

    # Pattern 1: 'select "X" in response to column N "..." of Q.Q.Q'
    # 예: you select "Yes" in response to column 1 "Process in place" of 2.2
    # Note: 질문 ID가 "2"로 끝나거나 "2.2"로 끝날 수 있음
    pattern1 = r'select\s+["\']?([^"\']+)["\']?\s+in\s+response\s+to\s+column\s+(\d+)\s+.*?of\s+(\d+(?:\.\d+)*)'
    match = re.search(pattern1, normalized_text, re.IGNORECASE)

    if match:
        return QuestionDependency(
            depends_on_question=match.group(3),
            depends_on_column=f"col{int(match.group(2)) - 1}",  # 1-indexed to 0-indexed
            trigger_values=[match.group(1)],
            raw_text=raw_text,
        )

    # Pattern 2: "select 'X' or 'Y' in column N of Q.Q.Q"
    pattern2 = r"select\s+['\"]?(.+?)['\"]?\s+(?:or\s+['\"]?(.+?)['\"]?\s+)?in\s+column\s+(\d+)\s+.*?of\s+(\d+(?:\.\d+)*)"
    match = re.search(pattern2, normalized_text, re.IGNORECASE)

    if match:
        values = [match.group(1)]
        if match.group(2):
            values.append(match.group(2))

        return QuestionDependency(
            depends_on_question=match.group(4),
            depends_on_column=f"col{int(match.group(3)) - 1}",  # 1-indexed to 0-indexed
            trigger_values=values,
            raw_text=raw_text,
        )

    # Pattern 3: "select 'X' in response to Q.Q.Q"
    pattern3 = r"select\s+['\"]?(.+?)['\"]?\s+in\s+response\s+to\s+(\d+\.\d+(?:\.\d+)?)"
    match = re.search(pattern3, normalized_text, re.IGNORECASE)

    if match:
        return QuestionDependency(
            depends_on_question=match.group(2),
            trigger_values=[match.group(1)],
            raw_text=raw_text,
        )

    # 패턴 매칭 실패 시 원본 텍스트만 보존
    return QuestionDependency(
        depends_on_question="",
        raw_text=raw_text,
    )


# ============================================================
# Merger
# ============================================================

def merge_question(
    qid: str,
    schema_q: Dict[str, Any],
    parsed_q: Dict[str, Any],
    resolver: SchemaResolver,
    translator: Optional[Any] = None,
) -> MergedQuestion:
    """단일 질문 병합

    Args:
        qid: 질문 ID
        schema_q: 스키마 질문 정의
        parsed_q: PDF 파싱 결과
        resolver: 스키마 resolver
        translator: Translator 인스턴스 (None이면 번역 안 함)
    """

    # 컬럼 스키마 resolve
    columns = [
        resolver.resolve_column(col_def)
        for col_def in schema_q.get("columns", [])
    ]

    # 텍스트 콘텐츠 추출 (+ 번역)
    guidance_raw = extract_guidance_raw(parsed_q, translator) if parsed_q else None

    # 질문 의존성 파싱
    question_dep = parse_question_dependency(parsed_q) if parsed_q else None

    return MergedQuestion(
        question_id=qid,
        title=schema_q.get("title", ""),
        title_ko=schema_q.get("title_ko"),
        response_type=schema_q.get("response_type", "table"),
        allow_multiple_rows=schema_q.get("allow_multiple_rows", True),
        row_labels=schema_q.get("row_labels"),
        columns=columns,
        question_dependency=question_dep,
        sector_specific=schema_q.get("sector_specific"),
        guidance_raw=guidance_raw,
    )


def merge_all(
    resolver: SchemaResolver,
    parsed_questions: Dict[str, Dict[str, Any]],
    translator: Optional[Any] = None,
) -> List[MergedQuestion]:
    """모든 질문 병합

    Args:
        resolver: 스키마 resolver
        parsed_questions: PDF 파싱 결과 (question_id -> 질문 데이터)
        translator: Translator 인스턴스 (None이면 번역 안 함)
    """
    merged = []

    for qid, schema_q in resolver.get_all_questions().items():
        parsed_q = parsed_questions.get(qid, {})
        merged.append(merge_question(qid, schema_q, parsed_q, resolver, translator))

    # 질문 ID로 정렬
    merged.sort(key=lambda q: [int(p) for p in q.question_id.split(".")])

    return merged


# ============================================================
# GPT Summary Generator
# ============================================================

def _maybe_import_openai():
    try:
        from openai import OpenAI
    except ImportError:
        return None
    return OpenAI


def build_gpt_prompt(question: MergedQuestion) -> List[Dict[str, str]]:
    """GPT 프롬프트 생성"""
    system = """You are a CDP (Carbon Disclosure Project) questionnaire expert.
Your task is to generate user-friendly guidance in Korean for each question.

Rules:
1. summary_ko: 2-4 sentences explaining what this question asks and why it matters
2. checklist_ko: 3-6 bullet points of key items to address
3. column_guides_ko: For each column, provide a brief tip on how to answer (if applicable)
4. condition_note_ko: Explain when this question/columns appear based on conditions

Be concise and practical. Focus on what the user needs to DO, not just understand."""

    # 질문 정보 구성
    columns_info = []
    for col in question.columns:
        col_info = {
            "id": col.id,
            "header": col.header,
            "type": col.type,
            "required": col.required,
        }
        if col.condition:
            col_info["condition"] = col.condition
        if col.options:
            col_info["options_count"] = len(col.options)
            col_info["options_sample"] = col.options[:3]
        columns_info.append(col_info)

    user_payload = {
        "question_id": question.question_id,
        "title": question.title,
        "title_ko": question.title_ko,
        "response_type": question.response_type,
        "columns": columns_info,
        "question_dependency": asdict(question.question_dependency) if question.question_dependency else None,
        "guidance_raw": asdict(question.guidance_raw) if question.guidance_raw else None,
    }

    user = f"""다음 CDP 질문에 대한 사용자 가이드를 생성하세요.

입력:
{json.dumps(user_payload, ensure_ascii=False, indent=2)}

출력 형식 (JSON):
{{
    "summary_ko": "이 질문이 무엇을 묻는지, 왜 중요한지 2-4문장으로 설명",
    "checklist_ko": ["체크포인트 1", "체크포인트 2", ...],
    "column_guides_ko": {{"column_id": "이 컬럼 작성 팁", ...}},
    "condition_note_ko": "이 질문/컬럼이 언제 표시되는지 설명 (조건이 없으면 빈 문자열)"
}}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_gpt(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
) -> Optional[GuidanceSummary]:
    """GPT 호출"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        if not content:
            return None

        data = json.loads(content)
        return GuidanceSummary(
            summary_ko=data.get("summary_ko", ""),
            checklist_ko=data.get("checklist_ko", []),
            column_guides_ko=data.get("column_guides_ko"),
            condition_note_ko=data.get("condition_note_ko", ""),
        )
    except Exception as e:
        print(f"[warn] GPT call failed: {e}")
        return None


def generate_summaries(
    questions: List[MergedQuestion],
    model: str = "gpt-4o-mini",
) -> None:
    """모든 질문에 대해 GPT 요약 생성"""
    OpenAI = _maybe_import_openai()
    api_key = os.getenv("OPENAI_API_KEY")

    if not OpenAI or not api_key:
        print("[warn] GPT 요약 생성 건너뜀 (openai 미설치 또는 API 키 없음)")
        return

    client = OpenAI(api_key=api_key)

    for q in questions:
        print(f"  Generating summary for {q.question_id}...")
        messages = build_gpt_prompt(q)
        summary = call_gpt(client, model, messages)
        if summary:
            q.guidance_summary = summary


# ============================================================
# Output
# ============================================================

def to_dict(questions: List[MergedQuestion]) -> Dict[str, Any]:
    """출력용 딕셔너리 변환"""

    def convert_column(col: ColumnSchema) -> Dict[str, Any]:
        result = {
            "id": col.id,
            "header": col.header,
            "type": col.type,
        }
        if col.header_ko:
            result["header_ko"] = col.header_ko
        if col.options:
            result["options"] = col.options
        if col.grouped_options:
            result["grouped_options"] = col.grouped_options
        if col.required:
            result["required"] = col.required
        if col.condition:
            result["condition"] = col.condition
        if col.min_value is not None:
            result["min_value"] = col.min_value
        if col.max_value is not None:
            result["max_value"] = col.max_value
        if col.max_length is not None:
            result["max_length"] = col.max_length
        return result

    def convert_question(q: MergedQuestion) -> Dict[str, Any]:
        result = {
            "question_id": q.question_id,
            "title": q.title,
            "response_type": q.response_type,
            "allow_multiple_rows": q.allow_multiple_rows,
            "columns": [convert_column(c) for c in q.columns],
        }

        if q.title_ko:
            result["title_ko"] = q.title_ko
        if q.row_labels:
            result["row_labels"] = q.row_labels
        if q.question_dependency:
            result["question_dependency"] = asdict(q.question_dependency)
        if q.sector_specific:
            result["sector_specific"] = q.sector_specific
        if q.guidance_raw:
            result["guidance_raw"] = asdict(q.guidance_raw)
        if q.guidance_summary:
            result["guidance_summary"] = asdict(q.guidance_summary)

        return result

    return {
        "version": "2025",
        "module": "2",
        "title": "Dependencies, Impacts, Risks, and Opportunities",
        "total_questions": len(questions),
        "questions": [convert_question(q) for q in questions],
    }


def save_output(questions: List[MergedQuestion], output_path: Path) -> None:
    """결과 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = to_dict(questions)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(questions)} questions to {output_path}")


# ============================================================
# Main
# ============================================================

def merge_questionnaire(
    schema_dir: Path,
    output_path: Path,
    pdf_path: Optional[Path] = None,
    parsed_path: Optional[Path] = None,
    with_gpt: bool = False,
    with_translation: bool = False,
    gpt_model: str = "gpt-4o-mini",
    save_intermediate: bool = True,
) -> Path:
    """메인 병합 함수

    Args:
        schema_dir: 스키마 디렉토리 (module2.json, common_options.json 등)
        output_path: 출력 JSON 경로
        pdf_path: CDP Questionnaire PDF 경로 (권장, 직접 파싱)
        parsed_path: 기존 파싱된 JSON 경로 (fallback)
        with_gpt: GPT 요약 생성 여부
        with_translation: 가이드라인 한국어 번역 여부
        gpt_model: OpenAI 모델명
        save_intermediate: PDF 파싱 중간 결과 저장 여부
    """

    if load_dotenv:
        load_dotenv()

    print(f"Loading schema from {schema_dir}...")
    resolver = SchemaResolver(schema_dir)

    # PDF 직접 파싱 (권장) 또는 JSON 로드 (fallback)
    if pdf_path and pdf_path.exists():
        print(f"Parsing PDF directly...")
        parsed_questions = parse_pdf_directly(
            pdf_path,
            save_intermediate=save_intermediate,
            output_dir=output_path.parent,
        )
    elif parsed_path and parsed_path.exists():
        print(f"Loading parsed questions from {parsed_path}...")
        parsed_questions = load_parsed_questions(parsed_path)
    else:
        raise ValueError("--pdf 또는 --parsed 중 하나는 필수입니다.")

    print(f"  Found {len(parsed_questions)} parsed questions")

    # 번역기 초기화 (옵션)
    translator = None
    if with_translation:
        if _translator_available:
            print("Initializing translator for Korean translation...")
            translator = Translator()
        else:
            print("[warn] Translator not available (utils.translator import failed)")

    print("Merging...")
    merged = merge_all(resolver, parsed_questions, translator)
    print(f"  Merged {len(merged)} questions")

    if with_gpt:
        print("Generating GPT summaries...")
        generate_summaries(merged, gpt_model)

    save_output(merged, output_path)

    return output_path


def main():
    backend_dir = Path(__file__).resolve().parents[2]  # model/cdp -> model -> backend

    default_schema = backend_dir / "schemas" / "2025"
    default_output = backend_dir / "output" / "cdp_questions_merged.json"

    parser = argparse.ArgumentParser(
        description="Merge CDP schema with parsed PDF content"
    )
    parser.add_argument(
        "--schema", type=Path, default=default_schema,
        help="Schema directory (contains module2.json, common_options.json, etc.)"
    )

    # PDF 또는 JSON 중 하나 선택
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--pdf", type=Path,
        help="CDP Questionnaire PDF path (권장, 직접 파싱)"
    )
    source_group.add_argument(
        "--parsed", type=Path,
        help="Parsed questionnaire JSON path (fallback)"
    )

    parser.add_argument(
        "--output", type=Path, default=default_output,
        help="Output merged JSON path"
    )
    parser.add_argument(
        "--no-intermediate", action="store_true",
        help="Skip saving intermediate parsed JSON"
    )
    parser.add_argument(
        "--with-gpt", action="store_true",
        help="Enable GPT summary generation"
    )
    parser.add_argument(
        "--with-translation", action="store_true",
        help="Enable Korean translation for guidance content"
    )
    parser.add_argument(
        "--gpt-model", type=str, default="gpt-4o-mini",
        help="OpenAI model name"
    )

    args = parser.parse_args()

    merge_questionnaire(
        schema_dir=args.schema,
        output_path=args.output,
        pdf_path=args.pdf,
        parsed_path=args.parsed,
        with_gpt=args.with_gpt,
        with_translation=args.with_translation,
        gpt_model=args.gpt_model,
        save_intermediate=not args.no_intermediate,
    )


if __name__ == "__main__":
    main()
