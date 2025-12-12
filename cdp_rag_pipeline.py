"""
CDP RAG Pipeline for Auto-generating Answers
=============================================
SK Inc 지속가능성 보고서를 활용한 CDP 질문 자동 답변 생성

사용 방법:
    python cdp_rag_pipeline.py --report data/2025_SK-Inc_Sustainability_Report_ENG.pdf --questions data/cdp_questions_parsed.json
"""

import json
import os
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

# PDF 처리
import fitz  # PyMuPDF

# 벡터 임베딩
from sentence_transformers import SentenceTransformer

# Qdrant 벡터 DB
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# LLM API
from openai import OpenAI
from dotenv import load_dotenv

# 진행률 표시
from tqdm import tqdm

load_dotenv()


@dataclass
class ReportChunk:
    """지속가능성 보고서 청크"""
    id: str
    content: str
    page_num: int
    section: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """검색 결과"""
    chunk_id: str
    content: str
    score: float
    page_num: int
    section: Optional[str] = None


@dataclass
class GeneratedAnswer:
    """생성된 답변"""
    question_id: str
    question_title: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    rationale_reference: Optional[str] = None
    ambition_reference: Optional[List[str]] = None


class SustainabilityReportParser:
    """지속가능성 보고서 파서"""

    # 주요 섹션 키워드
    SECTION_KEYWORDS = {
        'governance': ['governance', 'board', 'committee', 'management', 'oversight'],
        'strategy': ['strategy', 'strategic', 'roadmap', 'vision', 'target'],
        'risk_management': ['risk', 'opportunity', 'assessment', 'identification', 'mitigation'],
        'emissions': ['emissions', 'ghg', 'scope 1', 'scope 2', 'scope 3', 'carbon'],
        'energy': ['energy', 'renewable', 'consumption', 'efficiency'],
        'water': ['water', 'withdrawal', 'discharge', 'stress'],
        'biodiversity': ['biodiversity', 'ecosystem', 'habitat', 'species'],
        'waste': ['waste', 'recycling', 'circular', 'disposal'],
        'supply_chain': ['supplier', 'supply chain', 'procurement', 'vendor'],
        'targets': ['target', 'goal', 'commitment', 'net zero', 'reduction'],
    }

    def __init__(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.doc = fitz.open(pdf_path)
        self.chunks: List[ReportChunk] = []

    def extract_text_with_structure(self) -> List[Dict[str, Any]]:
        """페이지별 텍스트 추출"""
        pages = []

        for page_num, page in enumerate(self.doc):
            text = page.get_text("text")

            # 섹션 감지
            section = self._detect_section(text)

            pages.append({
                "page_num": page_num + 1,
                "text": text,
                "section": section
            })

        return pages

    def _detect_section(self, text: str) -> Optional[str]:
        """텍스트에서 섹션 감지"""
        text_lower = text.lower()

        for section, keywords in self.SECTION_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches >= 2:  # 최소 2개 키워드 매칭
                return section

        return None

    def create_chunks(self) -> List[ReportChunk]:
        """텍스트를 청크로 분할"""
        pages = self.extract_text_with_structure()

        chunk_id = 0
        for page_data in pages:
            text = page_data["text"]
            page_num = page_data["page_num"]
            section = page_data["section"]

            # 문단 단위로 먼저 분할
            paragraphs = self._split_into_paragraphs(text)

            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) <= self.chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk.strip():
                        self.chunks.append(ReportChunk(
                            id=f"report_chunk_{chunk_id}",
                            content=current_chunk.strip(),
                            page_num=page_num,
                            section=section,
                            metadata={
                                "source": os.path.basename(self.pdf_path),
                                "page": page_num
                            }
                        ))
                        chunk_id += 1

                    # 오버랩 처리
                    if self.chunk_overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + para + "\n\n"
                    else:
                        current_chunk = para + "\n\n"

            # 마지막 청크
            if current_chunk.strip():
                self.chunks.append(ReportChunk(
                    id=f"report_chunk_{chunk_id}",
                    content=current_chunk.strip(),
                    page_num=page_num,
                    section=section,
                    metadata={
                        "source": os.path.basename(self.pdf_path),
                        "page": page_num
                    }
                ))
                chunk_id += 1

        return self.chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """텍스트를 문단으로 분할"""
        # 연속된 줄바꿈으로 분할
        paragraphs = re.split(r'\n\s*\n', text)
        # 빈 문단 제거 및 정리
        return [p.strip() for p in paragraphs if p.strip()]

    def to_json(self, output_path: str):
        """JSON으로 저장"""
        data = {
            "source": os.path.basename(self.pdf_path),
            "total_pages": len(self.doc),
            "total_chunks": len(self.chunks),
            "chunks": [asdict(chunk) for chunk in self.chunks]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(self.chunks)} chunks to {output_path}")


class CDPRAGPipeline:
    """CDP RAG 파이프라인"""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "sustainability_report",
        qdrant_path: str = "./data/qdrant_cdp_db"
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Qdrant 클라이언트 초기화
        self.qdrant_path = qdrant_path
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name

        # OpenAI 클라이언트
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    def index_chunks(self, chunks: List[ReportChunk], batch_size: int = 32):
        """청크 인덱싱"""
        print(f"Indexing {len(chunks)} chunks...")

        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i+batch_size]

            # 임베딩 생성
            texts = [chunk.content for chunk in batch]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

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

        print(f"Indexed {len(chunks)} chunks")

    def search(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """유사도 검색"""
        # 쿼리 임베딩
        query_embedding = self.embedding_model.encode(query).tolist()

        # 필터 구성
        filter_conditions = None
        if section_filter:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            filter_conditions = Filter(
                must=[
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=section_filter)
                    )
                ]
            )

        # 검색
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=filter_conditions
        )

        return [
            SearchResult(
                chunk_id=r.payload["chunk_id"],
                content=r.payload["content"],
                score=r.score,
                page_num=r.payload["page_num"],
                section=r.payload.get("section")
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

        # 컨텍스트 구성
        context_text = "\n\n---\n\n".join([
            f"[Source: Page {chunk.page_num}]\n{chunk.content}"
            for chunk in context_chunks
        ])

        # 프롬프트 구성
        system_prompt = """You are an expert sustainability analyst helping to complete CDP (Carbon Disclosure Project) questionnaire responses.

Your task is to generate accurate and comprehensive answers based on the provided company sustainability report.

Guidelines:
1. Only use information from the provided context
2. If the information is not available, clearly state what is missing
3. Structure your answer according to the CDP question requirements
4. Include specific data, numbers, and examples where available
5. Reference the source page numbers
6. Be concise but comprehensive"""

        user_prompt = f"""CDP Question ID: {question.get('question_id', 'N/A')}
Question: {question.get('title_en', question.get('title', 'N/A'))}

Rationale (why this question is important):
{question.get('rationale', 'N/A')}

Ambition (best practices):
{json.dumps(question.get('ambition', []), ensure_ascii=False, indent=2)}

Requested Content (what should be included):
{json.dumps(question.get('requested_content', []), ensure_ascii=False, indent=2)}

---

CONTEXT FROM SUSTAINABILITY REPORT:

{context_text}

---

Please generate a comprehensive CDP response based on the above context. If certain information is not available in the context, indicate what additional data would be needed."""

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            answer_text = response.choices[0].message.content

            # 신뢰도 계산 (컨텍스트 점수 기반)
            avg_score = sum(c.score for c in context_chunks) / len(context_chunks) if context_chunks else 0
            confidence = min(avg_score * 1.2, 1.0)  # 스케일 조정

            return GeneratedAnswer(
                question_id=question.get('question_id', 'N/A'),
                question_title=question.get('title_en', question.get('title', 'N/A')),
                answer=answer_text,
                confidence=round(confidence, 2),
                sources=[{
                    "chunk_id": c.chunk_id,
                    "page_num": c.page_num,
                    "score": round(c.score, 3),
                    "section": c.section,
                    "preview": c.content[:200] + "..." if len(c.content) > 200 else c.content
                } for c in context_chunks],
                rationale_reference=question.get('rationale'),
                ambition_reference=question.get('ambition')
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
        top_k: int = 5
    ) -> List[GeneratedAnswer]:
        """모든 질문 처리"""
        answers = []

        for question in tqdm(questions, desc="Generating answers"):
            # 검색 쿼리 구성
            search_query = f"{question.get('title_en', '')} {question.get('rationale', '')}"

            # 유사 문서 검색
            relevant_chunks = self.search(search_query, top_k=top_k)

            # 답변 생성
            answer = self.generate_answer(question, relevant_chunks)
            answers.append(answer)

        return answers

    def generate_report(
        self,
        answers: List[GeneratedAnswer],
        output_path: str
    ):
        """결과 리포트 생성"""
        report = {
            "total_questions": len(answers),
            "average_confidence": round(
                sum(a.confidence for a in answers) / len(answers), 2
            ) if answers else 0,
            "answers": [asdict(a) for a in answers]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Saved report with {len(answers)} answers to {output_path}")
        return report


def flatten_questions(questions: List[Dict], parent_id: str = None) -> List[Dict]:
    """계층적 질문 구조를 평면화"""
    flat = []

    for q in questions:
        # 현재 질문 추가
        flat.append(q)

        # 하위 질문 재귀 처리
        if q.get('children'):
            flat.extend(flatten_questions(q['children'], q.get('question_id')))

    return flat


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CDP RAG Pipeline")
    parser.add_argument("--report", "-r", required=True, help="Sustainability report PDF path")
    parser.add_argument("--questions", "-q", required=True, help="CDP questions JSON path")
    parser.add_argument("--output", "-o", default="data/cdp_generated_answers.json", help="Output path")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--top-k", type=int, default=5, help="Number of similar chunks to retrieve")
    parser.add_argument("--recreate-index", action="store_true", help="Recreate vector index")

    args = parser.parse_args()

    # 1. 지속가능성 보고서 파싱
    print(f"\n{'='*50}")
    print("Step 1: Parsing sustainability report...")
    print(f"{'='*50}")

    report_parser = SustainabilityReportParser(
        args.report,
        chunk_size=args.chunk_size
    )
    chunks = report_parser.create_chunks()
    print(f"Created {len(chunks)} chunks from report")

    # 청크 저장
    report_chunks_path = Path(args.report).stem + "_chunks.json"
    report_parser.to_json(f"data/{report_chunks_path}")

    # 2. RAG 파이프라인 초기화
    print(f"\n{'='*50}")
    print("Step 2: Initializing RAG pipeline...")
    print(f"{'='*50}")

    pipeline = CDPRAGPipeline()
    pipeline.create_collection(recreate=args.recreate_index)

    # 3. 인덱싱
    print(f"\n{'='*50}")
    print("Step 3: Indexing chunks...")
    print(f"{'='*50}")

    pipeline.index_chunks(chunks)

    # 4. CDP 질문 로드
    print(f"\n{'='*50}")
    print("Step 4: Loading CDP questions...")
    print(f"{'='*50}")

    with open(args.questions, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    questions = questions_data.get('questions', questions_data)
    if isinstance(questions, dict):
        questions = [questions]

    # 계층 구조 평면화
    flat_questions = flatten_questions(questions)
    print(f"Loaded {len(flat_questions)} questions")

    # 5. 답변 생성
    print(f"\n{'='*50}")
    print("Step 5: Generating answers...")
    print(f"{'='*50}")

    answers = pipeline.process_questions(flat_questions, top_k=args.top_k)

    # 6. 결과 저장
    print(f"\n{'='*50}")
    print("Step 6: Saving results...")
    print(f"{'='*50}")

    pipeline.generate_report(answers, args.output)

    # 요약 출력
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total questions processed: {len(answers)}")
    print(f"Average confidence: {sum(a.confidence for a in answers) / len(answers):.2%}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
