"""
CDP Modeling FastAPI Server
===========================
AI Worker for CDP answer generation (3-Layer Architecture)

Endpoints:
- POST /ai/v1/generate/answer   - 단일 질문 RAG 답변 생성
- POST /ai/v1/generate/batch    - 일괄 답변 생성
- GET  /ai/v1/questions         - 질문 스키마 조회
- GET  /ai/v1/questions/{id}    - 특정 질문 스키마 조회
- GET  /ai/v1/questionnaire     - CDP 질문 + 가이드라인 전체 조회
- GET  /ai/v1/answers           - 사전 생성된 답변 전체 조회
- GET  /ai/v1/answers/{id}      - 특정 답변 조회
- POST /ai/v1/upload/sustainability-report - 지속가능경영보고서 인덱싱
- POST /ai/v1/upload/cdp-response - 과거 CDP 응답서 PDF 업로드 및 인덱싱
- GET  /ai/v1/upload/status     - 인덱싱 상태 확인
- DELETE /ai/v1/upload/clear    - 인덱스 삭제
- GET  /ai/v1/health            - 헬스체크

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import httpx
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse
from api.routes.generate import (
    router as generate_router,
    questions_router,
    questionnaire_router,
    answers_router,
)
from api.routes.upload import router as upload_router

# 환경변수 로드
load_dotenv()

# 기본 경로
BASE_DIR = Path(__file__).parent

# Spring Boot API URL
SPRING_API_URL = os.getenv("SPRING_API_URL", "http://localhost:8080")


async def sync_questionnaire_to_spring():
    """CDP 질문지 데이터를 Spring Boot DB에 동기화"""
    questions_path = BASE_DIR / "output" / "cdp_questions_merged.json"

    if not questions_path.exists():
        print("  [WARN] cdp_questions_merged.json not found, skipping sync")
        return False

    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            questionnaire_data = json.load(f)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SPRING_API_URL}/api/v1/cdp/questionnaire/import",
                json=questionnaire_data,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                print(
                    f"  [OK] Questionnaire synced to Spring Boot: {result.get('total', 0)} questions"
                )
                return True
            else:
                print(f"  [WARN] Sync failed: {response.status_code} - {response.text}")
                return False

    except httpx.ConnectError:
        print(f"  [WARN] Spring Boot not available at {SPRING_API_URL}, skipping sync")
        return False
    except Exception as e:
        print(f"  [WARN] Sync error: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    # Startup
    print("=" * 60)
    print("CDP Modeling API Server Starting...")
    print("=" * 60)
    print(f"  Base directory: {BASE_DIR}")
    print(f"  Qdrant DB path: {BASE_DIR / 'data' / 'qdrant_cdp_db'}")
    print(f"  Questions path: {BASE_DIR / 'output' / 'cdp_questions_merged.json'}")
    print(f"  Spring API URL: {SPRING_API_URL}")
    print("=" * 60)

    # Spring Boot로 질문지 동기화
    print("Syncing questionnaire to Spring Boot...")
    await sync_questionnaire_to_spring()
    print("=" * 60)

    yield

    # Shutdown
    print("CDP Modeling API Server Shutting down...")


# FastAPI 앱 생성
app = FastAPI(
    title="CDP Modeling API",
    description="AI Worker for CDP answer generation (Stateless)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/ai/v1/docs",
    redoc_url="/ai/v1/redoc",
    openapi_url="/ai/v1/openapi.json",
)

# CORS 설정 (Spring Boot에서 호출 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Vue.js dev
        "http://localhost:5173",  # Vite dev
        "http://localhost:8080",  # Spring Boot
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록 (/ai/v1 prefix)
app.include_router(generate_router, prefix="/ai/v1")
app.include_router(questions_router, prefix="/ai/v1")
app.include_router(questionnaire_router, prefix="/ai/v1")
app.include_router(answers_router, prefix="/ai/v1")
app.include_router(upload_router, prefix="/ai/v1")


@app.get("/ai/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    헬스체크

    - RAG 파이프라인 상태
    - OpenAI API 연결 상태
    - Qdrant 연결 상태
    """
    components = {
        "rag_pipeline": False,
        "openai_api": False,
        "qdrant": False,
    }

    # OpenAI API 체크
    if os.getenv("OPENAI_API_KEY"):
        components["openai_api"] = True

    # Qdrant DB 체크
    qdrant_path = BASE_DIR / "data" / "qdrant_cdp_db"
    if qdrant_path.exists():
        components["qdrant"] = True

    # 질문 스키마 체크
    questions_path = BASE_DIR / "output" / "cdp_questions_merged.json"
    if questions_path.exists():
        components["rag_pipeline"] = True

    status = "healthy" if all(components.values()) else "degraded"

    return HealthResponse(
        status=status,
        version="1.0.0",
        components=components,
    )


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "name": "CDP Modeling API",
        "version": "1.0.0",
        "docs": "/ai/v1/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
