"""
Upload Routes - 문서 업로드 및 인덱싱 API

엔드포인트:
- POST /ai/v1/upload/sustainability-report: 지속가능경영보고서 인덱싱 (고정 경로)
- POST /ai/v1/upload/cdp-response: 과거 CDP 응답서 PDF 업로드 및 인덱싱
- GET /ai/v1/upload/status: 인덱싱 상태 확인
- DELETE /ai/v1/upload/clear: 인덱스 삭제

3-Layer Architecture:
- 이 API는 RAG Layer의 데이터 저장을 담당
- 메타데이터 필수: source_type, year, question_code, historical
"""

import os
import shutil
from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

# 기존 모듈 사용
from model.sustainability.chunker import SustainabilityReportChunker
from model.cdp.response_parser import CDPResponseParser, CDPResponseChunk
from model.rag.indexer import RAGIndexer
from model.rag.document_schema import RAGDocument, SourceType

router = APIRouter(prefix="/upload", tags=["Upload"])

# 기본 경로
BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_SUSTAINABILITY_REPORT = BASE_DIR / "data" / "2025_SK-Inc_Sustainability_Report_ENG.pdf"
CDP_UPLOADS_DIR = BASE_DIR / "data" / "uploads" / "cdp_responses"

# 업로드 디렉토리 생성
CDP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# 인덱싱 상태 추적
_indexing_status = {
    "sustainability_report": {
        "indexed": False,
        "last_indexed": None,
        "chunks_count": 0,
        "file_path": None,
        "year": None,
    },
    "cdp_answers": {
        "indexed": False,
        "last_indexed": None,
        "answers_count": 0,
        "years": [],
    },
}


# ============================================
# Request/Response Schemas
# ============================================

class IndexSustainabilityReportRequest(BaseModel):
    """지속가능경영보고서 인덱싱 요청"""
    year: int = Field(default=2025, description="보고서 연도")
    chunk_size: int = Field(default=800, description="청크 크기")
    chunk_overlap: int = Field(default=200, description="청크 오버랩")




class IndexingResponse(BaseModel):
    """인덱싱 응답"""
    success: bool
    message: str
    indexed_count: int = 0
    details: Optional[Dict[str, Any]] = None


class IndexStatusResponse(BaseModel):
    """인덱싱 상태 응답"""
    sustainability_report: Dict[str, Any]
    cdp_answers: Dict[str, Any]
    collection_info: Optional[Dict[str, Any]] = None


# ============================================
# Helper Functions
# ============================================

# RAG 인덱서 싱글톤
_indexer_instance: Optional[RAGIndexer] = None


def get_indexer() -> RAGIndexer:
    """RAG 인덱서 싱글톤 인스턴스 반환"""
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = RAGIndexer(
            collection_name="cdp_rag_v2",
        )
    return _indexer_instance




# ============================================
# API Endpoints
# ============================================

@router.post("/sustainability-report", response_model=IndexingResponse)
async def index_sustainability_report(request: IndexSustainabilityReportRequest):
    """
    지속가능경영보고서 인덱싱

    - 고정 경로: data/2025_SK-Inc_Sustainability_Report_ENG.pdf
    - 청킹 후 벡터 DB에 저장
    - source_type: SUSTAINABILITY_REPORT
    - historical: False (현재 데이터)
    - 같은 연도 재업로드: 기존 데이터 자동 삭제 후 재인덱싱
    """
    if not DEFAULT_SUSTAINABILITY_REPORT.exists():
        raise HTTPException(
            status_code=404,
            detail=f"지속가능경영보고서를 찾을 수 없습니다: {DEFAULT_SUSTAINABILITY_REPORT}"
        )

    try:
        # 1. PDF 청킹
        print(f"Chunking sustainability report: {DEFAULT_SUSTAINABILITY_REPORT}")
        chunker = SustainabilityReportChunker(
            str(DEFAULT_SUSTAINABILITY_REPORT),
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        chunks = chunker.parse()
        print(f"Created {len(chunks)} chunks")

        # 2. RAGDocument로 변환 (historical=False: 현재 데이터)
        rag_docs = []
        for chunk in chunks:
            doc = RAGDocument.create_sustainability_report(
                year=request.year,
                text=chunk.content,
                section=chunk.section,
                page_num=chunk.page_num,
            )
            rag_docs.append(doc)

        # 3. 인덱싱
        indexer = get_indexer()
        indexer.create_collection(recreate=False)

        # 같은 연도 데이터가 존재하면 자동 삭제 (자동 중복 관리)
        try:
            indexer.delete_by_filter(
                source_type=SourceType.SUSTAINABILITY_REPORT,
                year=request.year
            )
            print(f"Deleted existing sustainability report data for year {request.year}")
        except Exception as e:
            print(f"No existing data to delete for year {request.year}: {e}")

        indexed_count = indexer.index_documents(rag_docs)

        # 4. 상태 업데이트
        _indexing_status["sustainability_report"] = {
            "indexed": True,
            "last_indexed": datetime.now().isoformat(),
            "chunks_count": indexed_count,
            "file_path": str(DEFAULT_SUSTAINABILITY_REPORT),
            "year": request.year,
        }

        return IndexingResponse(
            success=True,
            message=f"지속가능경영보고서 인덱싱 완료: {indexed_count}개 청크",
            indexed_count=indexed_count,
            details={
                "file": str(DEFAULT_SUSTAINABILITY_REPORT),
                "year": request.year,
                "stats": chunker.get_stats(),
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"인덱싱 실패: {str(e)}"
        )


@router.post("/cdp-response", response_model=IndexingResponse)
async def index_cdp_response(
    year: int = Form(..., description="CDP 응답 연도 (예: 2024)"),
    merge_duplicates: bool = Form(
        False,
        description="동일 question_id 청크를 하나로 병합할지 여부 (기본: False, 개별 Row 유지)"
    ),
    file: UploadFile = File(..., description="CDP 응답서 PDF 파일"),
):
    """
    과거 CDP 응답서 PDF 업로드 및 인덱싱

    - PDF 파일 업로드
    - CDPResponseParser로 질문별 답변 추출
    - source_type: CDP_ANSWER
    - historical: True (과거 데이터 = 참고용)
    - question_code 필터로 정확한 매핑 검색 가능
    - 같은 연도 재업로드: 기존 데이터 자동 삭제 후 재인덱싱

    3-Layer Note:
    - Mapping Layer에서 과거 질문 코드 조회
    - RAG Layer에서 question_code 필터로 검색
    - Prompt Layer에서 "참고용" 명시
    """
    # 파일 확장자 확인
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 업로드 가능합니다."
        )

    try:
        # 1. 파일 저장
        save_path = CDP_UPLOADS_DIR / f"{year}_{file.filename}"
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Saved CDP response PDF: {save_path}")

        # 2. PDF 파싱
        parser = CDPResponseParser(str(save_path))
        chunks = parser.parse()

        raw_chunks_count = len(chunks)
        if not raw_chunks_count:
            raise HTTPException(
                status_code=400,
                detail="PDF에서 CDP 답변을 추출할 수 없습니다."
            )

        # 질문 ID 통계 (중복 여부 파악)
        question_counts = Counter(c.question_id for c in chunks)
        unique_question_ids: List[str] = []
        seen_questions = set()
        for chunk in chunks:
            if chunk.question_id not in seen_questions:
                unique_question_ids.append(chunk.question_id)
                seen_questions.add(chunk.question_id)

        duplicate_question_ids = [
            {"question_id": qid, "count": count}
            for qid, count in question_counts.items()
            if count > 1
        ]

        # 중복 병합 옵션 처리
        if merge_duplicates:
            merged: Dict[str, CDPResponseChunk] = {}
            for chunk in chunks:
                existing = merged.get(chunk.question_id)
                if existing is None:
                    merged[chunk.question_id] = CDPResponseChunk(
                        question_id=chunk.question_id,
                        question_text=chunk.question_text,
                        answer_text=chunk.answer_text,
                        page_num=chunk.page_num,
                        module=chunk.module,
                    )
                else:
                    # 동일 질문에 대한 답변을 합친다 (줄바꿈으로 구분)
                    existing.answer_text = f"{existing.answer_text}\n\n{chunk.answer_text}"
                    existing.page_num = min(existing.page_num, chunk.page_num)
            chunks_for_indexing: List[CDPResponseParser.CDPResponseChunk] = list(merged.values())
        else:
            chunks_for_indexing = chunks

        # 3. RAGDocument로 변환 (historical=True: 참고용)
        rag_docs = []
        for chunk in chunks_for_indexing:
            doc = RAGDocument.create_cdp_answer(
                year=year,
                question_code=chunk.question_id,
                text=chunk.to_text(),
                module=chunk.module,
            )
            rag_docs.append(doc)

        # 4. 인덱싱
        indexer = get_indexer()
        indexer.create_collection(recreate=False)

        # 같은 연도 데이터가 존재하면 자동 삭제 (자동 중복 관리)
        try:
            indexer.delete_by_filter(
                source_type=SourceType.CDP_ANSWER,
                year=year
            )
            print(f"Deleted existing CDP answers for year {year}")
        except Exception as e:
            print(f"No existing data to delete for year {year}: {e}")

        indexed_count = indexer.index_documents(rag_docs)

        # 5. 상태 업데이트
        existing_years = _indexing_status["cdp_answers"].get("years", [])
        existing_files = _indexing_status["cdp_answers"].get("files", [])

        # 같은 연도 파일은 제거 (자동 중복 관리)
        existing_files = [f for f in existing_files if not f.startswith(str(CDP_UPLOADS_DIR / f"{year}_"))]

        if year not in existing_years:
            existing_years.append(year)

        _indexing_status["cdp_answers"] = {
            "indexed": True,
            "last_indexed": datetime.now().isoformat(),
            "answers_count": _indexing_status["cdp_answers"].get("answers_count", 0) + indexed_count,
            "years": sorted(existing_years),
            "files": existing_files + [str(save_path)],
        }

        return IndexingResponse(
            success=True,
            message=(
                f"{year}년 CDP 응답서 인덱싱 완료: "
                f"{indexed_count}개 청크 (고유 질문 {len(unique_question_ids)}개)"
            ),
            indexed_count=indexed_count,
            details={
                "year": year,
                "file": file.filename,
                "merge_duplicates": merge_duplicates,
                "total_chunks_indexed": indexed_count,
                "raw_chunks_found": raw_chunks_count,
                "questions_indexed": unique_question_ids,
                "duplicate_question_ids": duplicate_question_ids,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"CDP 응답서 인덱싱 실패: {str(e)}"
        )


@router.get("/status", response_model=IndexStatusResponse)
async def get_indexing_status():
    """인덱싱 상태 확인"""
    try:
        indexer = get_indexer()
        collection_info = indexer.get_collection_info()
    except Exception as e:
        collection_info = {"error": str(e)}

    return IndexStatusResponse(
        sustainability_report=_indexing_status["sustainability_report"],
        cdp_answers=_indexing_status["cdp_answers"],
        collection_info=collection_info,
    )


@router.delete("/clear")
async def clear_index(
    source_type: Optional[str] = None,
    year: Optional[int] = None,
):
    """
    인덱스 삭제

    Args:
        source_type: "CDP_ANSWER" 또는 "SUSTAINABILITY_REPORT"
        year: 특정 연도만 삭제
    """
    try:
        indexer = get_indexer()

        st = None
        if source_type:
            st = SourceType(source_type)

        indexer.delete_by_filter(source_type=st, year=year)

        # 상태 초기화
        if source_type == "SUSTAINABILITY_REPORT" or source_type is None:
            _indexing_status["sustainability_report"] = {
                "indexed": False,
                "last_indexed": None,
                "chunks_count": 0,
                "file_path": None,
                "year": None,
            }

        if source_type == "CDP_ANSWER" or source_type is None:
            if year:
                years = _indexing_status["cdp_answers"].get("years", [])
                if year in years:
                    years.remove(year)
                _indexing_status["cdp_answers"]["years"] = years
            else:
                _indexing_status["cdp_answers"] = {
                    "indexed": False,
                    "last_indexed": None,
                    "answers_count": 0,
                    "years": [],
                }

        return {
            "success": True,
            "message": f"인덱스 삭제 완료 (source_type={source_type}, year={year})"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"인덱스 삭제 실패: {str(e)}"
        )
