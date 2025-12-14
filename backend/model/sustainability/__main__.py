"""
Sustainability RAG Pipeline - Main Entry Point
===============================================
python -m model.sustainability 로 실행

파이프라인:
1. PDF 보고서 로드 및 청킹
2. 벡터 + BM25 인덱싱
3. 검색 테스트 (선택)
4. 질문 답변 생성
5. 평가 (선택)

Usage:
    cd backend
    python -m model.sustainability                      # 전체 파이프라인
    python -m model.sustainability --sample 5           # 5개 질문만
    python -m model.sustainability --test-search        # 검색만 테스트
    python -m model.sustainability --evaluate           # 평가 포함
"""

import argparse
import json
import sys
from pathlib import Path

# 경로 설정
BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"
OUTPUT_DIR = BACKEND_DIR / "output"

# 기본 파일
DEFAULT_REPORT = DATA_DIR / "2025_SK-Inc_Sustainability_Report_ENG.pdf"
DEFAULT_QUESTIONS = OUTPUT_DIR / "cdp_questions_merged.json"


def step1_chunking(report_path: str, chunk_size: int = 800, chunk_overlap: int = 200):
    """Step 1: PDF 보고서 청킹"""
    print("\n" + "="*60)
    print("Step 1: PDF 보고서 청킹")
    print("="*60)

    from .chunker import SustainabilityReportChunker

    print(f"  입력: {report_path}")

    chunker = SustainabilityReportChunker(
        report_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = chunker.parse()

    stats = chunker.get_stats()
    print(f"  총 청크: {stats['total_chunks']}개")
    print(f"  청크 타입: text={stats['chunk_types']['text']}, table={stats['chunk_types']['table']}")
    print(f"  평균 길이: {stats['avg_chunk_length']:.0f}자")
    print(f"  섹션: {list(stats['sections'].keys())}")

    return chunks, chunker


def step2_indexing(chunks, recreate: bool = True, use_bm25: bool = True, use_reranker: bool = True):
    """Step 2: 벡터 + BM25 인덱싱"""
    print("\n" + "="*60)
    print("Step 2: 벡터 + BM25 인덱싱")
    print("="*60)

    from .rag import EnhancedRAGPipeline

    print(f"  임베딩 모델: BAAI/bge-m3")
    print(f"  BM25: {'ON' if use_bm25 else 'OFF'}")
    print(f"  리랭커: {'ON' if use_reranker else 'OFF'}")

    rag = EnhancedRAGPipeline(
        embedding_model="bge-m3",
        use_bm25=use_bm25,
        use_reranker=use_reranker,
    )

    rag.create_collection(recreate=recreate)
    rag.index_chunks(chunks)

    print(f"  인덱싱 완료: {len(chunks)}개 청크")

    return rag


def step3_test_search(rag, queries: list = None):
    """Step 3: 검색 테스트"""
    print("\n" + "="*60)
    print("Step 3: 검색 테스트")
    print("="*60)

    if queries is None:
        queries = [
            "greenhouse gas emissions scope 1 scope 2",
            "renewable energy targets",
            "water management",
            "biodiversity impact assessment",
        ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        results = rag.search(query, top_k=3, use_rerank=True)

        for i, r in enumerate(results):
            print(f"    [{i+1}] Page {r.page_num} | Score: {r.score:.3f} | Rerank: {r.rerank_score:.3f}")
            print(f"        {r.content[:100]}...")


def step4_generate_answers(rag, questions_path: str, sample: int = None, output_path: str = None):
    """Step 4: 질문 답변 생성"""
    print("\n" + "="*60)
    print("Step 4: 질문 답변 생성")
    print("="*60)

    # 질문 로드
    print(f"  질문 파일: {questions_path}")

    with open(questions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data.get('questions', data)
    if isinstance(questions, dict):
        questions = [questions]

    # 평면화
    def flatten(qs):
        flat = []
        for q in qs:
            flat.append(q)
            if q.get('children'):
                flat.extend(flatten(q['children']))
        return flat

    flat_questions = flatten(questions)

    if sample:
        flat_questions = flat_questions[:sample]

    print(f"  처리할 질문: {len(flat_questions)}개")

    # 답변 생성
    answers = rag.process_questions(
        flat_questions,
        top_k=7,
        use_rerank=True,
        expand_query=True,
    )

    # 저장
    if output_path is None:
        output_path = OUTPUT_DIR / "cdp_rag_answers.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = rag.save_results(answers, str(output_path))

    print(f"\n  결과 저장: {output_path}")
    print(f"  총 질문: {result['total_questions']}")
    print(f"  평균 신뢰도: {result['average_confidence']:.2%}")

    return answers, result


def step5_evaluate(rag, questions_path: str, sample: int = 5, output_path: str = None):
    """Step 5: RAG 평가"""
    print("\n" + "="*60)
    print("Step 5: RAG 평가")
    print("="*60)

    from .evaluator import RAGEvaluator

    # 질문 로드
    with open(questions_path, 'r', encoding='utf-8') as f:
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

    flat_questions = flatten(questions)[:sample]
    print(f"  평가 질문: {len(flat_questions)}개")

    # 평가 실행
    evaluator = RAGEvaluator()
    report = evaluator.evaluate_pipeline(rag, flat_questions)

    # 저장
    if output_path is None:
        output_path = OUTPUT_DIR / "rag_evaluation.json"

    evaluator.save_report(report, str(output_path))

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Sustainability Report RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 파일 경로
    parser.add_argument("--report", "-r", default=str(DEFAULT_REPORT),
                        help="지속가능성 보고서 PDF")
    parser.add_argument("--questions", "-q", default=str(DEFAULT_QUESTIONS),
                        help="CDP 질문 JSON")
    parser.add_argument("--output", "-o", help="출력 파일 경로")

    # 파이프라인 옵션
    parser.add_argument("--sample", "-n", type=int, help="샘플 질문 수 (테스트용)")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=200)

    # 기능 옵션
    parser.add_argument("--test-search", action="store_true", help="검색만 테스트")
    parser.add_argument("--evaluate", action="store_true", help="평가 포함")
    parser.add_argument("--recreate-index", action="store_true", help="인덱스 재생성")

    # RAG 옵션
    parser.add_argument("--no-rerank", action="store_true", help="리랭킹 비활성화")
    parser.add_argument("--no-bm25", action="store_true", help="BM25 비활성화")

    args = parser.parse_args()

    # 파일 확인
    if not Path(args.report).exists():
        print(f"ERROR: 보고서 파일이 없습니다: {args.report}")
        sys.exit(1)

    if not args.test_search and not Path(args.questions).exists():
        print(f"ERROR: 질문 파일이 없습니다: {args.questions}")
        print(f"  -> output/cdp_questions_merged.json 파일이 필요합니다.")
        sys.exit(1)

    print("\n" + "="*60)
    print("Sustainability Report RAG Pipeline")
    print("="*60)
    print(f"  보고서: {args.report}")
    print(f"  질문: {args.questions}")
    print(f"  샘플: {args.sample or 'ALL'}")

    # Step 1: 청킹
    chunks, chunker = step1_chunking(
        args.report,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Step 2: 인덱싱
    rag = step2_indexing(
        chunks,
        recreate=args.recreate_index,
        use_bm25=not args.no_bm25,
        use_reranker=not args.no_rerank,
    )

    # Step 3: 검색 테스트 (옵션)
    if args.test_search:
        step3_test_search(rag)
        print("\n검색 테스트 완료!")
        return

    # Step 4: 답변 생성
    answers, result = step4_generate_answers(
        rag,
        args.questions,
        sample=args.sample,
        output_path=args.output,
    )

    # Step 5: 평가 (옵션)
    if args.evaluate:
        step5_evaluate(
            rag,
            args.questions,
            sample=args.sample or 5,
        )

    print("\n" + "="*60)
    print("파이프라인 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
