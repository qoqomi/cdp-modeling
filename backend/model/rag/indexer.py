"""
RAG Indexer - 메타데이터 포함 인덱싱

핵심 원칙:
- 모든 문서는 메타데이터와 함께 인덱싱
- source_type, year는 필수
- CDP 답변은 question_code 필수

외부 BGE Embedding 서비스 사용으로 경량화
"""

import os
import uuid
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)
from dotenv import load_dotenv

from .document_schema import RAGDocument, SourceType
from .embedding_client import get_embedding_client, EmbeddingClient

load_dotenv()


class RAGIndexer:
    """
    메타데이터 포함 RAG 인덱서

    외부 BGE Embedding 서비스를 사용하여 임베딩 생성

    금지사항:
    - 연도 메타데이터 없이 벡터화 금지
    """

    def __init__(
        self,
        collection_name: str = "cdp_rag_v2",
        qdrant_path: Optional[str] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        """
        Args:
            collection_name: Qdrant 컬렉션 이름
            qdrant_path: Qdrant DB 경로
            embedding_client: 외부 임베딩 클라이언트 (기본: 싱글톤)
        """
        # 외부 임베딩 클라이언트 사용
        self.embedding_client = embedding_client or get_embedding_client()
        print(f"Using external embedding service: {self.embedding_client.base_url}")
        self.embedding_dim = self.embedding_client.dimension

        # Qdrant 클라이언트 초기화
        if qdrant_path is None:
            backend_dir = Path(__file__).resolve().parents[2]
            qdrant_path = str(backend_dir / "data" / "qdrant_cdp_db")

        self.qdrant_path = qdrant_path
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name

    def create_collection(self, recreate: bool = False) -> None:
        """벡터 컬렉션 생성"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
            print(f"Deleted existing collection: {self.collection_name}")

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists")

    def _embed(self, text: str) -> List[float]:
        """텍스트 임베딩 (외부 서비스 호출)"""
        return self.embedding_client.embed(text)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩 (외부 서비스 호출)"""
        return self.embedding_client.embed_batch(texts)

    def index_document(self, doc: RAGDocument) -> str:
        """단일 문서 인덱싱"""
        point_id = str(uuid.uuid4())
        embedding = self._embed(doc.text)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=doc.to_payload()
                )
            ]
        )

        return point_id

    def index_documents(
        self,
        docs: List[RAGDocument],
        batch_size: int = 16
    ) -> int:
        """
        배치 문서 인덱싱

        Args:
            docs: 인덱싱할 문서 리스트
            batch_size: 배치 크기

        Returns:
            인덱싱된 문서 수
        """
        print(f"Indexing {len(docs)} documents...")

        indexed_count = 0
        for i in tqdm(range(0, len(docs), batch_size), desc="Indexing"):
            batch = docs[i:i + batch_size]

            # 배치 임베딩
            texts = [doc.text for doc in batch]
            embeddings = self._embed_batch(texts)

            # Qdrant에 저장
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=doc.to_payload()
                )
                for doc, embedding in zip(batch, embeddings)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            indexed_count += len(batch)

        print(f"Indexed {indexed_count} documents")
        return indexed_count

    def index_cdp_answers(
        self,
        answers: List[dict],
        year: int
    ) -> int:
        """
        CDP 답변 인덱싱

        Args:
            answers: CDP 답변 리스트 (question_code, text, module 포함)
            year: CDP 응답 연도

        Returns:
            인덱싱된 문서 수
        """
        docs = []
        for answer in answers:
            doc = RAGDocument.create_cdp_answer(
                year=year,
                question_code=answer["question_code"],
                text=answer["text"],
                module=answer.get("module"),
            )
            docs.append(doc)

        return self.index_documents(docs)

    def index_sustainability_report(
        self,
        chunks: List[dict],
        year: int
    ) -> int:
        """
        지속가능경영보고서 인덱싱

        Args:
            chunks: 청킹된 보고서 내용 리스트 (text, section, page_num 포함)
            year: 보고서 연도

        Returns:
            인덱싱된 문서 수
        """
        docs = []
        for chunk in chunks:
            doc = RAGDocument.create_sustainability_report(
                year=year,
                text=chunk["text"],
                section=chunk.get("section"),
                page_num=chunk.get("page_num"),
            )
            docs.append(doc)

        return self.index_documents(docs)

    def get_collection_info(self) -> dict:
        """컬렉션 정보 조회"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
            }
        except Exception as e:
            return {"error": str(e)}

    def delete_by_filter(
        self,
        source_type: Optional[SourceType] = None,
        year: Optional[int] = None
    ) -> int:
        """필터 조건으로 문서 삭제"""
        must_conditions = []

        if source_type:
            must_conditions.append({
                "key": "source_type",
                "match": {"value": source_type.value}
            })

        if year:
            must_conditions.append({
                "key": "year",
                "match": {"value": year}
            })

        if not must_conditions:
            raise ValueError("최소 하나의 필터 조건이 필요합니다")

        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector={
                "filter": {"must": must_conditions}
            }
        )

        return result
