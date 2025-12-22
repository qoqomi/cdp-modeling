"""
BGE Embedding Service Client

외부 BGE Embedding 서비스와 통신하는 클라이언트
- /embed: 임베딩 생성
- /rerank: 리랭킹
"""

import os
from typing import List, Optional, Tuple
import httpx
from dotenv import load_dotenv

load_dotenv()


class EmbeddingClient:
    """
    BGE Embedding Service 클라이언트

    외부 embedding 서비스를 호출하여 임베딩 및 리랭킹 수행
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Args:
            base_url: Embedding 서비스 URL (default: BGE_EMBEDDING_URL 환경변수)
            timeout: 요청 타임아웃 (초)
        """
        self.base_url = base_url or os.getenv(
            "BGE_EMBEDDING_URL",
            "http://climax-bge-embedding:5002"
        )
        self.timeout = timeout
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """임베딩 차원 (lazy load)"""
        if self._dimension is None:
            self._dimension = self._get_dimension()
        return self._dimension

    def _get_dimension(self) -> int:
        """서비스에서 임베딩 차원 조회"""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/health")
                response.raise_for_status()
                data = response.json()
                return data.get("dimension", 1024)
        except Exception as e:
            print(f"Warning: Failed to get dimension from service: {e}")
            return 1024  # BGE-M3 default

    def embed(self, text: str) -> List[float]:
        """
        단일 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (list of floats)
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/embed",
                json={"text": text}
            )
            response.raise_for_status()
            data = response.json()

            # 차원 정보 캐시
            if self._dimension is None:
                self._dimension = data.get("dimension", len(data["embedding"]))

            return data["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        배치 텍스트 임베딩

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        if not texts:
            return []

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/embed",
                json={"texts": texts}
            )
            response.raise_for_status()
            data = response.json()

            # 차원 정보 캐시
            if self._dimension is None and data.get("embeddings"):
                self._dimension = data.get("dimension", len(data["embeddings"][0]))

            return data["embeddings"]

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float, str]]:
        """
        문서 리랭킹

        Args:
            query: 쿼리
            documents: 리랭킹할 문서 리스트
            top_k: 상위 k개만 반환 (None이면 전체)

        Returns:
            [(원본_인덱스, 점수, 텍스트), ...] 리스트 (점수 내림차순)
        """
        if not documents:
            return []

        with httpx.Client(timeout=self.timeout) as client:
            payload = {
                "query": query,
                "documents": documents
            }
            if top_k:
                payload["top_k"] = top_k

            response = client.post(
                f"{self.base_url}/rerank",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            return [
                (r["index"], r["score"], r["text"])
                for r in data["results"]
            ]

    def health_check(self) -> dict:
        """서비스 헬스 체크"""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# 싱글톤 인스턴스
_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    """싱글톤 EmbeddingClient 반환"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client