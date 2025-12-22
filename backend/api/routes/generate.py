"""
Generate Routes - 3-Layer 기반 실시간 CDP 답변 생성

Architecture:
- RAG Layer: 메타데이터 필터 검색
- Mapping Layer: 연도별 질문 매핑
- Prompt Layer: 과거 답변은 참고용 명시

Note: 텍스트 정제 및 요약 기능은 merger.py에서 JSON 생성 시 처리됨
      (cdp_questions_merged.json에 이미 summary 필드 포함)
"""

import json
import time
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException

from api.schemas import (
    GenerateAnswerRequest,
    GenerateAnswerResponse,
    GenerateBatchRequest,
    GenerateBatchResponse,
    QuestionsResponse,
    QuestionSchema,
    ColumnSchema,
    StructuredAnswer,
    RowData,
    SourceInfo,
)

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Router definitions
router = APIRouter(tags=["Generate"])
questions_router = APIRouter(tags=["Questions"])
questionnaire_router = APIRouter(tags=["Questionnaire"])
answers_router = APIRouter(tags=["Answers"])

# ============================================
# Generator Singleton (3-Layer Architecture)
# ============================================

_generator = None
_use_realtime_generation = True  # 실시간 생성 사용 여부


def get_generator():
    """
    CDP 답변 생성기 싱글톤

    3-Layer Architecture:
    - RAG Layer: 메타데이터 필터 검색
    - Mapping Layer: 연도별 질문 매핑
    - Prompt Layer: 과거 답변은 참고용 명시
    """
    global _generator

    if _generator is None and _use_realtime_generation:
        try:
            from model.generation.cdp_generator import CDPAnswerGenerator
            _generator = CDPAnswerGenerator()
            print("CDPAnswerGenerator initialized (3-Layer Architecture)")
        except Exception as e:
            print(f"Warning: Failed to initialize CDPAnswerGenerator: {e}")
            print("Falling back to JSON-based answers")

    return _generator


def load_json_file(file_path: Path) -> dict:
    """Load JSON file safely"""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_questions_data() -> dict:
    """Load CDP questions merged JSON"""
    return load_json_file(BASE_DIR / "output" / "cdp_questions_merged.json")


def get_answers_data() -> dict:
    """Load CDP structured answers JSON (fallback용)"""
    return load_json_file(BASE_DIR / "output" / "cdp_structured_answers.json")


def get_question_title(question_id: str) -> str:
    """질문 제목 조회"""
    questions_data = get_questions_data()
    for q in questions_data.get("questions", []):
        if q.get("question_id") == question_id:
            return q.get("title", question_id)
    return question_id


# ============================================
# Generate Router (/ai/v1/generate/*)
# ============================================

@router.post("/generate/answer", response_model=GenerateAnswerResponse)
async def generate_answer(request: GenerateAnswerRequest):
    """
    단일 질문에 대한 실시간 CDP 답변 생성

    3-Layer Architecture:
    1. Mapping Layer: 과거 질문 코드 조회
    2. RAG Layer: 과거 CDP 답변 + 현재 보고서 검색
    3. Prompt Layer: 역할 명시 프롬프트 구성
    4. LLM 생성
    """
    start_time = time.time()
    generator = get_generator()

    # 실시간 생성 시도
    if generator:
        try:
            answer = generator.generate(request.question_id)

            # RowData 변환
            rows = []
            for idx, row in enumerate(answer.rows):
                if isinstance(row, dict):
                    rows.append(RowData(
                        row_index=row.get("row_index", idx),
                        columns=row.get("columns", row),
                        confidence=row.get("confidence", answer.overall_confidence)
                    ))

            # SourceInfo 변환
            sources = []
            for src in answer.sources:
                sources.append(SourceInfo(
                    chunk_id=src.get("chunk_id", ""),
                    page_num=src.get("page_num", 0),
                    score=src.get("score", 0.0),
                    rerank_score=src.get("rerank_score"),
                    section=src.get("section"),
                    preview=src.get("preview"),
                ))

            processing_time = int((time.time() - start_time) * 1000)

            return GenerateAnswerResponse(
                success=True,
                answer=StructuredAnswer(
                    question_id=answer.question_id,
                    question_title=answer.question_title,
                    response_type="table",
                    rows=rows,
                    overall_confidence=answer.overall_confidence,
                    sources=sources,
                    rationale_en=answer.rationale_en,
                    rationale_ko=answer.rationale_ko,
                    validation_errors=[],
                ),
                processing_time_ms=processing_time,
            )

        except Exception as e:
            print(f"실시간 생성 실패, fallback 사용: {e}")

    # Fallback: 기존 JSON 기반 응답
    answers_data = get_answers_data()

    for answer in answers_data.get("answers", []):
        if answer.get("question_id") == request.question_id:
            rows = [
                RowData(
                    row_index=row.get("row_index", 0),
                    columns=row.get("columns", {}),
                    confidence=row.get("confidence", 0.0)
                )
                for row in answer.get("rows", [])
            ]

            sources = [
                SourceInfo(
                    chunk_id=src.get("chunk_id", ""),
                    page_num=src.get("page_num", 0),
                    score=src.get("score", 0.0),
                    rerank_score=src.get("rerank_score"),
                )
                for src in answer.get("sources", [])
            ]

            processing_time = int((time.time() - start_time) * 1000)

            return GenerateAnswerResponse(
                success=True,
                answer=StructuredAnswer(
                    question_id=answer.get("question_id", ""),
                    question_title=answer.get("question_title", ""),
                    response_type=answer.get("response_type", "table"),
                    rows=rows,
                    overall_confidence=answer.get("overall_confidence", 0.0),
                    sources=sources,
                    rationale_en=answer.get("rationale_en", ""),
                    rationale_ko=answer.get("rationale_ko", ""),
                    validation_errors=answer.get("validation_errors", []),
                ),
                processing_time_ms=processing_time,
            )

    return GenerateAnswerResponse(
        success=False,
        error=f"Question {request.question_id} not found",
    )


@router.post("/generate/batch", response_model=GenerateBatchResponse)
async def generate_batch(request: GenerateBatchRequest):
    """
    일괄 실시간 CDP 답변 생성

    3-Layer Architecture 기반
    """
    start_time = time.time()
    generator = get_generator()
    results = []
    errors = {}

    for question_id in request.question_ids:
        # 실시간 생성 시도
        if generator:
            try:
                answer = generator.generate(question_id)

                rows = []
                for idx, row in enumerate(answer.rows):
                    if isinstance(row, dict):
                        rows.append(RowData(
                            row_index=row.get("row_index", idx),
                            columns=row.get("columns", row),
                            confidence=row.get("confidence", answer.overall_confidence)
                        ))

                sources = []
                for src in answer.sources:
                    sources.append(SourceInfo(
                        chunk_id=src.get("chunk_id", ""),
                        page_num=src.get("page_num", 0),
                        score=src.get("score", 0.0),
                        rerank_score=src.get("rerank_score"),
                        section=src.get("section"),
                        preview=src.get("preview"),
                    ))

                results.append(StructuredAnswer(
                    question_id=answer.question_id,
                    question_title=answer.question_title,
                    response_type="table",
                    rows=rows,
                    overall_confidence=answer.overall_confidence,
                    sources=sources,
                    rationale_en=answer.rationale_en,
                    rationale_ko=answer.rationale_ko,
                    validation_errors=[],
                ))
                continue

            except Exception as e:
                print(f"실시간 생성 실패 ({question_id}): {e}")

        # Fallback: JSON 기반
        answers_data = get_answers_data()
        found = False

        for answer in answers_data.get("answers", []):
            if answer.get("question_id") == question_id:
                rows = [
                    RowData(
                        row_index=row.get("row_index", 0),
                        columns=row.get("columns", {}),
                        confidence=row.get("confidence", 0.0)
                    )
                    for row in answer.get("rows", [])
                ]

                sources = [
                    SourceInfo(
                        chunk_id=src.get("chunk_id", ""),
                        page_num=src.get("page_num", 0),
                        score=src.get("score", 0.0),
                        rerank_score=src.get("rerank_score"),
                    )
                    for src in answer.get("sources", [])
                ]

                results.append(StructuredAnswer(
                    question_id=answer.get("question_id", ""),
                    question_title=answer.get("question_title", ""),
                    response_type=answer.get("response_type", "table"),
                    rows=rows,
                    overall_confidence=answer.get("overall_confidence", 0.0),
                    sources=sources,
                    rationale_en=answer.get("rationale_en", ""),
                    rationale_ko=answer.get("rationale_ko", ""),
                    validation_errors=answer.get("validation_errors", []),
                ))
                found = True
                break

        if not found:
            errors[question_id] = "Question not found"

    processing_time = int((time.time() - start_time) * 1000)

    return GenerateBatchResponse(
        success=len(errors) == 0,
        total=len(request.question_ids),
        completed=len(results),
        failed=len(errors),
        answers=results,
        errors=errors,
        processing_time_ms=processing_time,
    )


# ============================================
# Questions Router (/ai/v1/questions)
# ============================================

@questions_router.get("/questions", response_model=QuestionsResponse)
async def get_all_questions():
    """전체 질문 스키마 조회"""
    questions_data = get_questions_data()
    questions = questions_data.get("questions", [])

    result = []
    for q in questions:
        columns = [
            ColumnSchema(
                id=col.get("id", ""),
                header=col.get("header", ""),
                type=col.get("type", "text"),
                required=col.get("required", False),
                options=col.get("options"),
                max_length=col.get("max_length"),
            )
            for col in q.get("columns", [])
        ]

        result.append(QuestionSchema(
            question_id=q.get("question_id", ""),
            title=q.get("title", ""),
            response_type=q.get("response_type", "table"),
            columns=columns,
            row_labels=q.get("row_labels"),
        ))

    return QuestionsResponse(
        success=True,
        total=len(result),
        questions=result,
    )


@questions_router.get("/questions/{question_id}", response_model=QuestionsResponse)
async def get_question(question_id: str):
    """특정 질문 스키마 조회"""
    questions_data = get_questions_data()

    for q in questions_data.get("questions", []):
        if q.get("question_id") == question_id:
            columns = [
                ColumnSchema(
                    id=col.get("id", ""),
                    header=col.get("header", ""),
                    type=col.get("type", "text"),
                    required=col.get("required", False),
                    options=col.get("options"),
                    max_length=col.get("max_length"),
                )
                for col in q.get("columns", [])
            ]

            return QuestionsResponse(
                success=True,
                total=1,
                questions=[QuestionSchema(
                    question_id=q.get("question_id", ""),
                    title=q.get("title", ""),
                    response_type=q.get("response_type", "table"),
                    columns=columns,
                    row_labels=q.get("row_labels"),
                )],
            )

    return QuestionsResponse(
        success=False,
        error=f"Question {question_id} not found",
    )


# ============================================
# Questionnaire Router (/ai/v1/questionnaire)
# ============================================

@questionnaire_router.get("/questionnaire")
async def get_questionnaire():
    """
    CDP 질문 + 가이드라인 전체 조회

    Returns:
        Full questionnaire data with questions and guidance
        (텍스트 정제 및 요약은 merger.py에서 JSON 생성 시 처리됨)
    """
    questions_data = get_questions_data()

    if not questions_data:
        raise HTTPException(status_code=404, detail="Questionnaire not found")

    return {
        "success": True,
        "data": questions_data,
    }


# ============================================
# Answers Router (/ai/v1/answers)
# ============================================

@answers_router.get("/answers")
async def get_all_answers():
    """
    사전 생성된 답변 전체 조회 (Fallback용)

    Note: 실시간 생성이 기본이므로 이 엔드포인트는 테스트/디버깅용
    """
    answers_data = get_answers_data()

    if not answers_data:
        raise HTTPException(status_code=404, detail="Answers not found")

    return {
        "success": True,
        "data": answers_data,
    }


@answers_router.get("/answers/{question_id}")
async def get_answer(question_id: str):
    """
    특정 질문에 대한 답변 조회

    Note: 실시간 생성을 원하면 POST /generate/answer 사용

    Args:
        question_id: Question ID (e.g., "C1.1")

    Returns:
        Answer for the specified question
    """
    answers_data = get_answers_data()
    questions_data = get_questions_data()

    # Find answer
    answer = None
    for a in answers_data.get("answers", []):
        if a.get("question_id") == question_id:
            answer = a
            break

    if not answer:
        raise HTTPException(status_code=404, detail=f"Answer for {question_id} not found")

    # Find schema
    schema = None
    for q in questions_data.get("questions", []):
        if q.get("question_id") == question_id:
            schema = q
            break

    return {
        "success": True,
        "answer": answer,
        "schema": schema,
    }
