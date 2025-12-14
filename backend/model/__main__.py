"""
CDP RAG Full Pipeline - Main Entry Point
=========================================
python -m model 로 전체 파이프라인 실행

파이프라인:
1. CDP 질문지 PDF 파싱 → cdp_questions_merged.json
2. 지속가능성 보고서 청킹 → 벡터 인덱싱
3. RAG 답변 생성:
   - 서술형: cdp_rag_answers.json (기본)
   - 구조화: cdp_structured_answers.json (--structured 옵션)
4. (선택) 평가 → rag_evaluation.json

Usage:
    cd backend

    # 전체 파이프라인 (서술형 답변)
    python -m model

    # 구조화된 답변 생성 (스키마 기반)
    python -m model --structured

    # 샘플 테스트
    python -m model --sample 5
    python -m model --structured --sample 5

    # 평가 포함
    python -m model --sample 5 --evaluate

    # 개별 모듈만
    python -m model.cdp              # CDP 질문 파싱만
    python -m model.sustainability   # RAG 답변 생성만
"""

import argparse
import sys
from pathlib import Path

# 경로 설정
BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
OUTPUT_DIR = BACKEND_DIR / "output"

# 기본 파일
DEFAULT_QUESTIONNAIRE_PDF = DATA_DIR / "Full_Corporate_Questionnaire_Modules_1-6_short.pdf"
DEFAULT_REPORT_PDF = DATA_DIR / "2025_SK-Inc_Sustainability_Report_ENG.pdf"
DEFAULT_QUESTIONS_JSON = OUTPUT_DIR / "cdp_questions_merged.json"
DEFAULT_ANSWERS_OUTPUT = OUTPUT_DIR / "cdp_rag_answers.json"
DEFAULT_STRUCTURED_OUTPUT = OUTPUT_DIR / "cdp_structured_answers.json"
DEFAULT_EVAL_OUTPUT = OUTPUT_DIR / "rag_evaluation.json"


def check_files():
    """파일 상태 확인"""
    print("\n=== 파일 상태 확인 ===")

    files = [
        ("CDP 질문지 PDF", DEFAULT_QUESTIONNAIRE_PDF),
        ("지속가능성 보고서 PDF", DEFAULT_REPORT_PDF),
        ("파싱된 질문 JSON", DEFAULT_QUESTIONS_JSON),
        ("RAG 답변 JSON", DEFAULT_ANSWERS_OUTPUT),
    ]

    for name, path in files:
        status = "OK" if path.exists() else "MISSING"
        print(f"  {name}: {status}")

    return all(p.exists() for _, p in files[:2])  # PDF 파일들만 필수


def run_cdp_pipeline(skip_if_exists: bool = True):
    """Step 1: CDP 질문지 파싱"""
    print("\n" + "="*60)
    print("PHASE 1: CDP 질문지 파싱")
    print("="*60)

    if skip_if_exists and DEFAULT_QUESTIONS_JSON.exists():
        print(f"  이미 존재: {DEFAULT_QUESTIONS_JSON}")
        print("  (재생성하려면 --force 옵션 사용)")
        return True

    from .cdp import CDPQuestionnaireSectionParser, merge_questionnaire

    # 파싱
    print(f"\n--- PDF 파싱 ---")
    print(f"  입력: {DEFAULT_QUESTIONNAIRE_PDF}")

    parser = CDPQuestionnaireSectionParser(str(DEFAULT_QUESTIONNAIRE_PDF))
    questions = parser.parse()
    print(f"  파싱 완료: {len(questions)}개 질문")

    # 저장
    parsed_path = OUTPUT_DIR / f"{DEFAULT_QUESTIONNAIRE_PDF.stem}_parsed.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parser.to_json(str(parsed_path))

    # 스키마 병합
    print(f"\n--- 스키마 병합 ---")
    schema_dir = BACKEND_DIR / "schemas" / "2025"

    merge_questionnaire(
        schema_dir=str(schema_dir),
        parsed_json_path=str(parsed_path),
        output_path=str(DEFAULT_QUESTIONS_JSON),
    )

    print(f"  출력: {DEFAULT_QUESTIONS_JSON}")
    return True


def run_sustainability_pipeline(sample: int = None, evaluate: bool = False, recreate: bool = True):
    """Step 2-3: RAG 답변 생성"""
    print("\n" + "="*60)
    print("PHASE 2: 지속가능성 보고서 RAG")
    print("="*60)

    from .sustainability import (
        SustainabilityReportChunker,
        EnhancedRAGPipeline,
        RAGEvaluator,
    )
    import json

    # 보고서 청킹
    print(f"\n--- 보고서 청킹 ---")
    print(f"  입력: {DEFAULT_REPORT_PDF}")

    chunker = SustainabilityReportChunker(str(DEFAULT_REPORT_PDF))
    chunks = chunker.parse()

    stats = chunker.get_stats()
    print(f"  청크 생성: {stats['total_chunks']}개")
    print(f"  타입: text={stats['chunk_types']['text']}, table={stats['chunk_types']['table']}")

    # RAG 인덱싱
    print(f"\n--- RAG 인덱싱 ---")

    rag = EnhancedRAGPipeline(
        embedding_model="bge-m3",
        use_bm25=True,
        use_reranker=True,
    )
    rag.create_collection(recreate=recreate)
    rag.index_chunks(chunks)

    # 질문 로드
    print(f"\n--- 질문 로드 ---")
    print(f"  입력: {DEFAULT_QUESTIONS_JSON}")

    with open(DEFAULT_QUESTIONS_JSON, 'r', encoding='utf-8') as f:
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

    print(f"  질문 수: {len(flat_questions)}")

    # 답변 생성
    print(f"\n--- RAG 답변 생성 ---")

    answers = rag.process_questions(
        flat_questions,
        top_k=7,
        use_rerank=True,
        expand_query=True,
    )

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = rag.save_results(answers, str(DEFAULT_ANSWERS_OUTPUT))

    print(f"\n  출력: {DEFAULT_ANSWERS_OUTPUT}")
    print(f"  총 질문: {result['total_questions']}")
    print(f"  평균 신뢰도: {result['average_confidence']:.2%}")

    # 평가 (선택)
    if evaluate:
        print(f"\n--- RAG 평가 ---")
        evaluator = RAGEvaluator()
        eval_questions = flat_questions[:min(5, len(flat_questions))]
        report = evaluator.evaluate_pipeline(rag, eval_questions)
        evaluator.save_report(report, str(DEFAULT_EVAL_OUTPUT))

    return result


def run_structured_pipeline(sample: int = None, recreate: bool = True):
    """Step 2-3: 구조화된 답변 생성 (스키마 기반)"""
    print("\n" + "="*60)
    print("PHASE 2: 구조화된 답변 생성 (스키마 기반)")
    print("="*60)

    from .sustainability import SustainabilityReportChunker, EnhancedRAGPipeline
    import sys
    sys.path.insert(0, str(BACKEND_DIR))
    from structured_answer_generator import StructuredAnswerGenerator
    import json

    # 보고서 청킹
    print(f"\n--- 보고서 청킹 ---")
    print(f"  입력: {DEFAULT_REPORT_PDF}")

    chunker = SustainabilityReportChunker(str(DEFAULT_REPORT_PDF))
    chunks = chunker.parse()

    stats = chunker.get_stats()
    print(f"  청크 생성: {stats['total_chunks']}개")
    print(f"  타입: text={stats['chunk_types']['text']}, table={stats['chunk_types']['table']}")

    # RAG 인덱싱
    print(f"\n--- RAG 인덱싱 ---")

    rag = EnhancedRAGPipeline(
        embedding_model="bge-m3",
        use_bm25=True,
        use_reranker=True,
    )
    rag.create_collection(recreate=recreate)
    rag.index_chunks(chunks)

    # 질문 로드
    print(f"\n--- 질문 로드 ---")
    print(f"  입력: {DEFAULT_QUESTIONS_JSON}")

    with open(DEFAULT_QUESTIONS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data.get('questions', data)
    if isinstance(questions, dict):
        questions = [questions]

    if sample:
        questions = questions[:sample]

    # 구조화된 답변 생성
    print(f"\n--- 구조화된 답변 생성 ---")
    generator = StructuredAnswerGenerator(rag)
    answers = generator.generate_all(questions)

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = generator.save_results(answers, str(DEFAULT_STRUCTURED_OUTPUT))

    print(f"\n  출력: {DEFAULT_STRUCTURED_OUTPUT}")
    print(f"  총 질문: {result['total_questions']}")
    print(f"  평균 신뢰도: {result['average_confidence']:.2%}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="CDP RAG Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m model                      # 전체 파이프라인 (서술형 답변)
  python -m model --structured         # 구조화된 답변 (스키마 기반)
  python -m model --sample 5           # 5개 질문만 테스트
  python -m model --evaluate           # 평가 포함
  python -m model --force              # 기존 파일 무시하고 재생성
  python -m model --check              # 파일 상태만 확인
        """
    )

    parser.add_argument("--sample", "-n", type=int, help="테스트할 질문 수")
    parser.add_argument("--structured", "-s", action="store_true", help="스키마 기반 구조화된 답변 생성")
    parser.add_argument("--evaluate", "-e", action="store_true", help="평가 포함")
    parser.add_argument("--force", "-f", action="store_true", help="기존 파일 무시하고 재생성")
    parser.add_argument("--check", action="store_true", help="파일 상태만 확인")
    parser.add_argument("--skip-cdp", action="store_true", help="CDP 파싱 스킵")
    parser.add_argument("--skip-rag", action="store_true", help="RAG 생성 스킵")

    args = parser.parse_args()

    # 파일 확인
    if args.check:
        check_files()
        return

    print("\n" + "="*60)
    print("CDP RAG Full Pipeline")
    print("="*60)

    # 필수 파일 확인
    if not DEFAULT_QUESTIONNAIRE_PDF.exists():
        print(f"ERROR: CDP 질문지 PDF 없음: {DEFAULT_QUESTIONNAIRE_PDF}")
        sys.exit(1)

    if not DEFAULT_REPORT_PDF.exists():
        print(f"ERROR: 지속가능성 보고서 PDF 없음: {DEFAULT_REPORT_PDF}")
        sys.exit(1)

    # Phase 1: CDP 질문 파싱
    if not args.skip_cdp:
        run_cdp_pipeline(skip_if_exists=not args.force)

    # Phase 2-3: RAG 답변 생성
    if not args.skip_rag:
        if not DEFAULT_QUESTIONS_JSON.exists():
            print(f"ERROR: 질문 JSON 없음: {DEFAULT_QUESTIONS_JSON}")
            print("  먼저 CDP 파싱을 실행하세요: python -m model.cdp")
            sys.exit(1)

        if args.structured:
            # 스키마 기반 구조화된 답변 생성
            run_structured_pipeline(
                sample=args.sample,
                recreate=True,
            )
        else:
            # 기존 서술형 RAG 답변 생성
            run_sustainability_pipeline(
                sample=args.sample,
                evaluate=args.evaluate,
                recreate=True,
            )

    # 완료
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n출력 파일:")
    print(f"  - 질문: {DEFAULT_QUESTIONS_JSON}")
    if args.structured:
        print(f"  - 답변: {DEFAULT_STRUCTURED_OUTPUT} (구조화)")
    else:
        print(f"  - 답변: {DEFAULT_ANSWERS_OUTPUT} (서술형)")
    if args.evaluate and not args.structured:
        print(f"  - 평가: {DEFAULT_EVAL_OUTPUT}")


if __name__ == "__main__":
    main()
