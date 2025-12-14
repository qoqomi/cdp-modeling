"""
Sustainability Report RAG Pipeline (Enhanced)
==============================================
고정확도 RAG 파이프라인

개선 사항:
- BGE-M3 임베딩 모델 (다국어, 고정확도)
- 하이브리드 검색 (벡터 + BM25)
- Cross-encoder 리랭킹
- 쿼리 확장/재작성
- 메타데이터 필터링 강화

Usage:
    from model.sustainability import EnhancedRAGPipeline

    rag = EnhancedRAGPipeline()
    rag.index_chunks(chunks)
    results = rag.search("환경 영향 평가", top_k=10, rerank=True)
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import math

# 벡터 임베딩
from sentence_transformers import SentenceTransformer, CrossEncoder

# Qdrant 벡터 DB
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

# LLM API
from openai import OpenAI
from dotenv import load_dotenv

# 진행률 표시
from tqdm import tqdm

from .chunker import ReportChunk

load_dotenv()


@dataclass
class SearchResult:
    """검색 결과"""
    chunk_id: str
    content: str
    score: float
    page_num: int
    section: Optional[str] = None
    bm25_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class GeneratedAnswer:
    """생성된 답변"""
    question_id: str
    question_title: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]


class BM25Index:
    """BM25 키워드 검색 인덱스"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_freqs: Dict[str, int] = Counter()
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.vocab: set = set()

    def _tokenize(self, text: str) -> List[str]:
        """텍스트 토큰화"""
        text = text.lower()
        # 영어, 한글, 숫자만 추출
        tokens = re.findall(r'[a-z]+|[가-힣]+|\d+\.?\d*', text)
        return tokens

    def fit(self, documents: List[Dict[str, Any]]):
        """문서 인덱싱"""
        self.documents = documents
        self.doc_lengths = []
        self.doc_freqs = Counter()

        # 문서별 토큰화 및 통계
        for doc in documents:
            tokens = self._tokenize(doc.get("content", ""))
            self.doc_lengths.append(len(tokens))
            unique_tokens = set(tokens)
            self.vocab.update(unique_tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0

    def _compute_idf(self, term: str) -> float:
        """IDF 계산"""
        n = len(self.documents)
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25 검색"""
        query_tokens = self._tokenize(query)
        scores = []

        for doc_idx, doc in enumerate(self.documents):
            doc_tokens = self._tokenize(doc.get("content", ""))
            doc_len = self.doc_lengths[doc_idx]
            term_freqs = Counter(doc_tokens)

            score = 0.0
            for term in query_tokens:
                if term not in term_freqs:
                    continue
                tf = term_freqs[term]
                idf = self._compute_idf(term)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * (numerator / denominator)

            scores.append((doc_idx, score))

        # 상위 k개 반환
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class EnhancedRAGPipeline:
    """고정확도 RAG 파이프라인"""

    # 임베딩 모델 옵션
    EMBEDDING_MODELS = {
        "bge-m3": "BAAI/bge-m3",
        "bge-large": "BAAI/bge-large-en-v1.5",
        "e5-large": "intfloat/multilingual-e5-large",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    }

    # 리랭커 모델 옵션
    RERANKER_MODELS = {
        "bge-reranker": "BAAI/bge-reranker-v2-m3",
        "ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }

    def __init__(
        self,
        embedding_model: str = "bge-m3",
        reranker_model: str = "bge-reranker",
        collection_name: str = "sustainability_report_v2",
        qdrant_path: Optional[str] = None,
        use_bm25: bool = True,
        use_reranker: bool = True,
    ):
        """
        Args:
            embedding_model: 임베딩 모델 키 또는 전체 경로
            reranker_model: 리랭커 모델 키 또는 전체 경로
            collection_name: Qdrant 컬렉션 이름
            qdrant_path: Qdrant DB 경로
            use_bm25: BM25 하이브리드 검색 사용 여부
            use_reranker: 리랭커 사용 여부
        """
        # 임베딩 모델 초기화 (MPS 메모리 부족 방지를 위해 CPU 사용)
        model_name = self.EMBEDDING_MODELS.get(embedding_model, embedding_model)
        print(f"Loading embedding model: {model_name} (device: cpu)")
        self.embedding_model = SentenceTransformer(model_name, device="cpu")
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # 리랭커 초기화
        self.use_reranker = use_reranker
        self.reranker = None
        if use_reranker:
            reranker_name = self.RERANKER_MODELS.get(reranker_model, reranker_model)
            print(f"Loading reranker model: {reranker_name} (device: cpu)")
            self.reranker = CrossEncoder(reranker_name, device="cpu")

        # BM25 인덱스
        self.use_bm25 = use_bm25
        self.bm25_index = BM25Index() if use_bm25 else None
        self.indexed_documents: List[Dict[str, Any]] = []

        # Qdrant 클라이언트 초기화
        if qdrant_path is None:
            backend_dir = Path(__file__).resolve().parents[2]
            qdrant_path = str(backend_dir / "data" / "qdrant_cdp_db")

        self.qdrant_path = qdrant_path
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name

        # OpenAI 클라이언트
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=api_key) if api_key else None

    def create_collection(self, recreate: bool = False):
        """벡터 컬렉션 생성"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

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

    def index_chunks(self, chunks: List[ReportChunk], batch_size: int = 16):
        """청크 인덱싱 (벡터 + BM25)"""
        print(f"Indexing {len(chunks)} chunks...")

        # 문서 저장 (BM25용)
        self.indexed_documents = []
        for chunk in chunks:
            self.indexed_documents.append({
                "chunk_id": chunk.id,
                "content": chunk.content,
                "page_num": chunk.page_num,
                "section": chunk.section,
                "metadata": chunk.metadata,
            })

        # BM25 인덱스 구축
        if self.use_bm25:
            print("Building BM25 index...")
            self.bm25_index.fit(self.indexed_documents)

        # 벡터 인덱싱
        print("Building vector index...")
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i+batch_size]

            # 임베딩 생성
            texts = [chunk.content for chunk in batch]
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True  # BGE 모델은 정규화 필요
            )

            # Qdrant에 저장
            points = [
                PointStruct(
                    id=i + idx,
                    vector=embedding.tolist(),
                    payload={
                        "chunk_id": chunk.id,
                        "content": chunk.content,
                        "page_num": chunk.page_num,
                        "section": chunk.section,
                        "metadata": chunk.metadata
                    }
                )
                for idx, (chunk, embedding) in enumerate(zip(batch, embeddings))
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        print(f"Indexed {len(chunks)} chunks (Vector + BM25)")

    def _vector_search(
        self,
        query: str,
        top_k: int = 20,
        section_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """벡터 유사도 검색"""
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        filter_conditions = None
        if section_filter:
            filter_conditions = Filter(
                must=[
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=section_filter)
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=filter_conditions
        )

        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "content": r.payload["content"],
                "page_num": r.payload["page_num"],
                "section": r.payload.get("section"),
                "vector_score": r.score,
            }
            for r in results
        ]

    def _hybrid_search(
        self,
        query: str,
        top_k: int = 20,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        section_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """하이브리드 검색 (벡터 + BM25)"""
        # 벡터 검색
        vector_results = self._vector_search(query, top_k=top_k * 2, section_filter=section_filter)

        if not self.use_bm25:
            return vector_results[:top_k]

        # BM25 검색
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)

        # 점수 정규화 및 병합
        chunk_scores: Dict[str, Dict[str, Any]] = {}

        # 벡터 점수 추가
        max_vector_score = max((r["vector_score"] for r in vector_results), default=1.0)
        for r in vector_results:
            chunk_id = r["chunk_id"]
            normalized_score = r["vector_score"] / max_vector_score if max_vector_score > 0 else 0
            chunk_scores[chunk_id] = {
                **r,
                "vector_score": normalized_score,
                "bm25_score": 0.0,
            }

        # BM25 점수 추가
        max_bm25_score = max((score for _, score in bm25_results), default=1.0)
        for doc_idx, score in bm25_results:
            doc = self.indexed_documents[doc_idx]
            chunk_id = doc["chunk_id"]
            normalized_score = score / max_bm25_score if max_bm25_score > 0 else 0

            if chunk_id in chunk_scores:
                chunk_scores[chunk_id]["bm25_score"] = normalized_score
            else:
                chunk_scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": doc["content"],
                    "page_num": doc["page_num"],
                    "section": doc.get("section"),
                    "vector_score": 0.0,
                    "bm25_score": normalized_score,
                }

        # 최종 점수 계산
        for chunk_id in chunk_scores:
            item = chunk_scores[chunk_id]
            item["combined_score"] = (
                vector_weight * item["vector_score"] +
                bm25_weight * item["bm25_score"]
            )

        # 정렬 및 상위 k개 반환
        sorted_results = sorted(
            chunk_scores.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )

        return sorted_results[:top_k]

    def _rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Cross-encoder로 리랭킹"""
        if not self.reranker or not results:
            return results[:top_k]

        # 쿼리-문서 쌍 생성
        pairs = [(query, r["content"]) for r in results]

        # 리랭킹 점수 계산
        scores = self.reranker.predict(pairs)

        # 점수 추가 및 정렬
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        sorted_results = sorted(
            results,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return sorted_results[:top_k]

    def expand_query(self, query: str, question_context: Optional[Dict] = None) -> str:
        """LLM으로 쿼리 확장"""
        if not self.openai_client:
            return query

        context_info = ""
        if question_context:
            context_info = f"""
질문 배경 (Rationale): {question_context.get('rationale', '')}
요청 내용: {question_context.get('requested_content', [])}
"""

        prompt = f"""주어진 검색 쿼리를 지속가능성 보고서에서 관련 정보를 찾기 위해 확장하세요.
동의어, 관련 용어, 영문/한글 표현을 추가하세요.

원본 쿼리: {query}
{context_info}

확장된 쿼리 (한 줄로):"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            expanded = response.choices[0].message.content.strip()
            return f"{query} {expanded}"
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return query

    def search(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[str] = None,
        use_rerank: bool = True,
        expand_query: bool = False,
        question_context: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        통합 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            section_filter: 섹션 필터
            use_rerank: 리랭킹 사용 여부
            expand_query: 쿼리 확장 사용 여부
            question_context: 질문 컨텍스트 (쿼리 확장용)
        """
        # 쿼리 확장
        if expand_query:
            query = self.expand_query(query, question_context)

        # 하이브리드 검색 (더 많이 가져옴)
        fetch_k = top_k * 3 if use_rerank else top_k
        results = self._hybrid_search(
            query,
            top_k=fetch_k,
            section_filter=section_filter
        )

        # 리랭킹
        if use_rerank and self.use_reranker:
            results = self._rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        # SearchResult로 변환
        return [
            SearchResult(
                chunk_id=r["chunk_id"],
                content=r["content"],
                score=r.get("combined_score", r.get("vector_score", 0)),
                page_num=r["page_num"],
                section=r.get("section"),
                bm25_score=r.get("bm25_score", 0),
                rerank_score=r.get("rerank_score", 0),
            )
            for r in results
        ]

    def generate_answer(
        self,
        question: Dict[str, Any],
        context_chunks: List[SearchResult],
        model: str = "gpt-4o-mini"
    ) -> GeneratedAnswer:
        """LLM으로 답변 생성"""
        if not self.openai_client:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

        # 컨텍스트 구성 (리랭크 점수 순)
        sorted_chunks = sorted(context_chunks, key=lambda x: x.rerank_score or x.score, reverse=True)
        context_text = "\n\n---\n\n".join([
            f"[Source: Page {chunk.page_num}, Section: {chunk.section or 'N/A'}, Relevance: {chunk.rerank_score:.3f}]\n{chunk.content}"
            for chunk in sorted_chunks
        ])

        system_prompt = """You are an expert sustainability analyst for CDP (Carbon Disclosure Project) questionnaire responses.

Your task is to generate accurate, evidence-based answers using ONLY the provided context.

Guidelines:
1. Use ONLY information from the provided context - do not add external knowledge
2. If information is missing, explicitly state what data is not available
3. Include specific numbers, percentages, and metrics when available
4. Reference source page numbers for key claims
5. Structure answers according to CDP requirements
6. Be concise but comprehensive
7. Use professional sustainability reporting language"""

        user_prompt = f"""CDP Question ID: {question.get('question_id', 'N/A')}
Question: {question.get('title_en', question.get('title', 'N/A'))}

Rationale (why this question is important):
{question.get('rationale', 'N/A')}

Requested Content:
{json.dumps(question.get('requested_content', []), ensure_ascii=False, indent=2)}

---

CONTEXT FROM SUSTAINABILITY REPORT (sorted by relevance):

{context_text}

---

Generate a comprehensive CDP response. If certain information is not available, clearly indicate what additional data would be needed."""

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )

            answer_text = response.choices[0].message.content

            # 신뢰도 계산 (리랭크 점수 기반)
            if context_chunks:
                avg_rerank = sum(c.rerank_score for c in context_chunks if c.rerank_score) / len(context_chunks)
                avg_score = sum(c.score for c in context_chunks) / len(context_chunks)
                confidence = min((avg_rerank * 0.6 + avg_score * 0.4) * 1.1, 1.0)
            else:
                confidence = 0.0

            return GeneratedAnswer(
                question_id=question.get('question_id', 'N/A'),
                question_title=question.get('title_en', question.get('title', 'N/A')),
                answer=answer_text,
                confidence=round(confidence, 2),
                sources=[{
                    "chunk_id": c.chunk_id,
                    "page_num": c.page_num,
                    "section": c.section,
                    "score": round(c.score, 3),
                    "rerank_score": round(c.rerank_score, 3) if c.rerank_score else None,
                    "preview": c.content[:200] + "..." if len(c.content) > 200 else c.content
                } for c in context_chunks],
            )

        except Exception as e:
            print(f"Error generating answer for {question.get('question_id')}: {e}")
            return GeneratedAnswer(
                question_id=question.get('question_id', 'N/A'),
                question_title=question.get('title_en', question.get('title', 'N/A')),
                answer=f"Error generating answer: {str(e)}",
                confidence=0.0,
                sources=[]
            )

    def process_questions(
        self,
        questions: List[Dict[str, Any]],
        top_k: int = 7,
        model: str = "gpt-4o-mini",
        use_rerank: bool = True,
        expand_query: bool = True,
    ) -> List[GeneratedAnswer]:
        """모든 질문 처리"""
        answers = []

        for question in tqdm(questions, desc="Generating answers"):
            # 검색 쿼리 구성
            search_query = f"{question.get('title_en', '')} {question.get('rationale', '')}"

            # 관련 문서 검색
            relevant_chunks = self.search(
                search_query,
                top_k=top_k,
                use_rerank=use_rerank,
                expand_query=expand_query,
                question_context=question,
            )

            # 답변 생성
            answer = self.generate_answer(question, relevant_chunks, model=model)
            answers.append(answer)

        return answers

    def save_results(
        self,
        answers: List[GeneratedAnswer],
        output_path: str
    ) -> Dict:
        """결과 저장"""
        report = {
            "pipeline_config": {
                "embedding_model": str(self.embedding_model),
                "use_bm25": self.use_bm25,
                "use_reranker": self.use_reranker,
            },
            "total_questions": len(answers),
            "average_confidence": round(
                sum(a.confidence for a in answers) / len(answers), 2
            ) if answers else 0,
            "answers": [asdict(a) for a in answers]
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(answers)} answers to {output_path}")
        return report


# Backward compatibility alias
RAGPipeline = EnhancedRAGPipeline


def run_rag_pipeline(
    report_pdf: str,
    questions_json: str,
    output_path: str = "output/cdp_rag_answers.json",
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    top_k: int = 7,
    recreate_index: bool = False,
    use_rerank: bool = True,
    expand_query: bool = True,
):
    """향상된 RAG 파이프라인 실행"""
    from .chunker import SustainabilityReportChunker

    # 1. 청킹
    print("Step 1: Chunking report...")
    chunker = SustainabilityReportChunker(
        report_pdf,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = chunker.parse()
    print(f"  Created {len(chunks)} chunks")

    # 2. RAG 초기화 및 인덱싱
    print("Step 2: Initializing Enhanced RAG Pipeline...")
    rag = EnhancedRAGPipeline(
        embedding_model="bge-m3",
        reranker_model="bge-reranker",
        use_bm25=True,
        use_reranker=use_rerank,
    )
    rag.create_collection(recreate=recreate_index)
    rag.index_chunks(chunks)

    # 3. 질문 로드
    print("Step 3: Loading questions...")
    with open(questions_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data.get('questions', data)
    if isinstance(questions, dict):
        questions = [questions]

    def flatten(qs):
        flat = []
        for q in qs:
            flat.append(q)
            if q.get('children'):
                flat.extend(flatten(q['children']))
        return flat

    flat_questions = flatten(questions)
    print(f"  Loaded {len(flat_questions)} questions")

    # 4. 답변 생성
    print("Step 4: Generating answers with enhanced pipeline...")
    answers = rag.process_questions(
        flat_questions,
        top_k=top_k,
        use_rerank=use_rerank,
        expand_query=expand_query,
    )

    # 5. 저장
    print("Step 5: Saving results...")
    rag.save_results(answers, output_path)

    return answers


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Sustainability Report RAG Pipeline")
    parser.add_argument("--report", "-r", required=True, help="Sustainability report PDF")
    parser.add_argument("--questions", "-q", required=True, help="CDP questions JSON")
    parser.add_argument("--output", "-o", default="output/cdp_rag_answers.json")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--recreate-index", action="store_true")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--no-expand", action="store_true", help="Disable query expansion")

    args = parser.parse_args()

    run_rag_pipeline(
        report_pdf=args.report,
        questions_json=args.questions,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        recreate_index=args.recreate_index,
        use_rerank=not args.no_rerank,
        expand_query=not args.no_expand,
    )


if __name__ == "__main__":
    main()
