"""
Sustainability Report Module (Enhanced)
=======================================
지속가능성 보고서 파싱, RAG 청킹, 벡터 검색

Components:
- parser.py: SustainabilityReportParser (PDF → 구조화 데이터)
- chunker.py: SustainabilityReportChunker (RAG용 청킹, 테이블/헤더 인식)
- rag.py: EnhancedRAGPipeline (하이브리드 검색 + 리랭킹)

Features:
- BGE-M3 임베딩 모델 (다국어, 고정확도)
- 하이브리드 검색 (벡터 + BM25)
- Cross-encoder 리랭킹
- 쿼리 확장
- 시맨틱 청킹 (테이블/헤더 인식)

Usage:
    from model.sustainability import EnhancedRAGPipeline, SustainabilityReportChunker

    # 청킹
    chunker = SustainabilityReportChunker("report.pdf")
    chunks = chunker.parse()

    # RAG 파이프라인
    rag = EnhancedRAGPipeline()
    rag.index_chunks(chunks)
    results = rag.search("환경 영향", use_rerank=True)
"""

from .parser import SustainabilityReportParser
from .chunker import SustainabilityReportChunker, ReportChunk
from .rag import (
    EnhancedRAGPipeline,
    RAGPipeline,  # Backward compatibility alias
    BM25Index,
    SearchResult,
    GeneratedAnswer,
    run_rag_pipeline,
)
from .evaluator import (
    RAGEvaluator,
    EvaluationReport,
    quick_evaluate,
)

__all__ = [
    # Parser
    "SustainabilityReportParser",
    # Chunker
    "SustainabilityReportChunker",
    "ReportChunk",
    # RAG Pipeline
    "EnhancedRAGPipeline",
    "RAGPipeline",  # Backward compatibility
    "BM25Index",
    "SearchResult",
    "GeneratedAnswer",
    "run_rag_pipeline",
    # Evaluator
    "RAGEvaluator",
    "EvaluationReport",
    "quick_evaluate",
]
