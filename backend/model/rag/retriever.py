"""
RAG Retriever - 메타데이터 필터 기반 검색

핵심 원칙:
- 검색 시 반드시 source_type, year 필터 사용
- 과거 CDP 답변 검색 시 question_code 필터 필수
- 유사도 검색만으로 문항 매칭 금지 (Mapping Layer 사용)

외부 BGE Embedding 서비스 사용으로 경량화
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# Qdrant import
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None

from .document_schema import RAGDocument, SourceType, SearchFilter
from .embedding_client import get_embedding_client, EmbeddingClient


class RAGRetriever:
    """
    메타데이터 필터 기반 RAG 검색기

    외부 BGE Embedding 서비스를 사용하여 임베딩 및 리랭킹 수행

    금지사항:
    - Mapping 없이 유사도만으로 문항 매칭 금지
    - 연도 필터 없이 검색 금지
    """

    def __init__(
        self,
        collection_name: str = "cdp_rag_v2",
        qdrant_path: Optional[str] = None,
        use_reranker: bool = True,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        """
        Args:
            collection_name: Qdrant 컬렉션 이름
            qdrant_path: Qdrant DB 경로
            use_reranker: 리랭커 사용 여부
            embedding_client: 외부 임베딩 클라이언트 (기본: 싱글톤)
        """
        self.embedding_client = None
        self.client = None
        self.collection_name = collection_name
        self.use_reranker = use_reranker
        self._initialized = False

        # Qdrant 의존성 체크
        if not QDRANT_AVAILABLE:
            print("Warning: qdrant-client not available. RAG disabled.")
            return

        try:
            # 외부 임베딩 클라이언트 사용
            self.embedding_client = embedding_client or get_embedding_client()
            print(f"Using external embedding service: {self.embedding_client.base_url}")

            # Qdrant 클라이언트 초기화
            if qdrant_path is None:
                backend_dir = Path(__file__).resolve().parents[2]
                qdrant_path = str(backend_dir / "data" / "qdrant_cdp_db")

            self.client = QdrantClient(path=qdrant_path)

            # 컬렉션 존재 여부 확인
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            if collection_name not in collection_names:
                print(f"Warning: Collection '{collection_name}' not found. RAG search will return empty results.")
            else:
                self._initialized = True
                print(f"RAG Retriever initialized with collection '{collection_name}'")

        except Exception as e:
            print(f"Warning: Failed to initialize RAG Retriever: {e}")
            self._initialized = False

    def _embed(self, text: str) -> List[float]:
        """텍스트 임베딩 (외부 서비스 호출)"""
        if not self.embedding_client:
            raise ValueError("Embedding client not initialized")
        return self.embedding_client.embed(text)

    def _build_qdrant_filter(self, search_filter: SearchFilter) -> Optional[Filter]:
        """SearchFilter를 Qdrant Filter로 변환"""
        must_conditions = []

        if search_filter.source_types:
            must_conditions.append(
                FieldCondition(
                    key="source_type",
                    match=MatchAny(any=[st.value for st in search_filter.source_types])
                )
            )

        if search_filter.years:
            must_conditions.append(
                FieldCondition(
                    key="year",
                    match=MatchAny(any=search_filter.years)
                )
            )

        if search_filter.question_codes:
            must_conditions.append(
                FieldCondition(
                    key="question_code",
                    match=MatchAny(any=search_filter.question_codes)
                )
            )

        if search_filter.modules:
            must_conditions.append(
                FieldCondition(
                    key="module",
                    match=MatchAny(any=search_filter.modules)
                )
            )

        if search_filter.historical_only is not None:
            must_conditions.append(
                FieldCondition(
                    key="historical",
                    match=MatchValue(value=search_filter.historical_only)
                )
            )

        if not must_conditions:
            return None

        return Filter(must=must_conditions)

    def _rerank(
        self,
        query: str,
        documents: List[RAGDocument],
        top_k: int
    ) -> List[RAGDocument]:
        """외부 서비스로 리랭킹"""
        if not self.use_reranker or not self.embedding_client or not documents:
            return documents[:top_k]

        try:
            # 문서 텍스트 추출
            doc_texts = [doc.text for doc in documents]

            # 외부 서비스로 리랭킹
            rerank_results = self.embedding_client.rerank(query, doc_texts, top_k=top_k)

            # 결과 매핑 (원본 인덱스 → 점수)
            score_map = {idx: score for idx, score, _ in rerank_results}

            # 점수 업데이트
            for i, doc in enumerate(documents):
                if i in score_map:
                    doc.score = score_map[i]

            # 정렬 및 상위 k개 반환
            sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)
            return sorted_docs[:top_k]

        except Exception as e:
            print(f"Warning: Reranking failed, using original order: {e}")
            return documents[:top_k]

    def search(
        self,
        query: str,
        source_types: List[SourceType],
        years: List[int],
        question_codes: Optional[List[str]] = None,
        top_k: int = 5,
        use_rerank: bool = True,
    ) -> List[RAGDocument]:
        """
        메타데이터 필터를 적용한 RAG 검색

        Args:
            query: 검색 쿼리
            source_types: 문서 유형 필터 (필수)
            years: 연도 필터 (필수)
            question_codes: 질문 코드 필터 (CDP 답변 검색 시 권장)
            top_k: 반환할 결과 수
            use_rerank: 리랭킹 사용 여부

        Returns:
            검색된 RAGDocument 리스트
        """
        # RAG가 초기화되지 않은 경우 빈 리스트 반환
        if not self._initialized or not self.client or not self.embedding_client:
            print("Warning: RAG not initialized. Returning empty results.")
            return []

        try:
            # 필터 구성
            search_filter = SearchFilter(
                source_types=source_types,
                years=years,
                question_codes=question_codes,
            )

            qdrant_filter = self._build_qdrant_filter(search_filter)

            # 리랭킹 사용 시 더 많이 가져옴
            fetch_k = top_k * 3 if use_rerank and self.use_reranker else top_k

            # 벡터 검색 (외부 서비스로 임베딩)
            query_embedding = self._embed(query)

            # qdrant-client 1.7+ 에서는 query_points 사용
            if hasattr(self.client, 'query_points'):
                from qdrant_client.models import QueryRequest
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=qdrant_filter,
                    limit=fetch_k,
                ).points
            else:
                # 이전 버전에서는 search 사용
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    query_filter=qdrant_filter,
                    limit=fetch_k,
                )

            # RAGDocument로 변환
            documents = []
            for r in results:
                try:
                    doc = RAGDocument.from_payload(r.payload, score=r.score)
                    documents.append(doc)
                except Exception as e:
                    print(f"Warning: Failed to parse document: {e}")
                    continue

            # 리랭킹 (외부 서비스 사용)
            if use_rerank and self.use_reranker:
                documents = self._rerank(query, documents, top_k)

            return documents

        except Exception as e:
            print(f"RAG search failed: {e}")
            return []

    def search_cdp_answers(
        self,
        query: str,
        years: List[int],
        question_codes: List[str],
        top_k: int = 3,
    ) -> List[RAGDocument]:
        """
        과거 CDP 답변 검색 (Mapping Layer 결과 사용)

        Args:
            query: 검색 쿼리 (현재 질문 제목)
            years: Mapping Layer에서 가져온 과거 연도 리스트
            question_codes: Mapping Layer에서 가져온 과거 질문 코드 리스트
            top_k: 반환할 결과 수

        Returns:
            검색된 과거 CDP 답변 리스트
        """
        return self.search(
            query=query,
            source_types=[SourceType.CDP_ANSWER],
            years=years,
            question_codes=question_codes,
            top_k=top_k,
            use_rerank=True,
        )

    def search_sustainability_report(
        self,
        query: str,
        year: int,
        top_k: int = 5,
    ) -> List[RAGDocument]:
        """
        지속가능경영보고서 검색

        Args:
            query: 검색 쿼리
            year: 보고서 연도 (현재 연도)
            top_k: 반환할 결과 수

        Returns:
            검색된 보고서 청크 리스트
        """
        return self.search(
            query=query,
            source_types=[SourceType.SUSTAINABILITY_REPORT],
            years=[year],
            top_k=top_k,
            use_rerank=True,
        )