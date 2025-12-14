"""
CDP Questionnaire Pipeline - Main Entry Point
==============================================
python -m model.cdp 로 실행

파이프라인:
1. CDP 질문지 PDF 파싱
2. 스키마와 병합
3. 최종 질문 JSON 출력

Usage:
    cd backend
    python -m model.cdp                           # 전체 파이프라인
    python -m model.cdp --parse-only              # 파싱만
    python -m model.cdp --merge-only              # 병합만
"""

import argparse
import json
import sys
from pathlib import Path

# 경로 설정
BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"
OUTPUT_DIR = BACKEND_DIR / "output"
SCHEMA_DIR = BACKEND_DIR / "schemas" / "2025"

# 기본 파일
DEFAULT_QUESTIONNAIRE_PDF = DATA_DIR / "Full_Corporate_Questionnaire_Modules_1-6_short.pdf"
DEFAULT_PARSED_OUTPUT = OUTPUT_DIR / "cdp_questions_parsed.json"
DEFAULT_MERGED_OUTPUT = OUTPUT_DIR / "cdp_questions_merged.json"


def step1_parse_questionnaire(pdf_path: str, output_path: str = None):
    """Step 1: CDP 질문지 PDF 파싱"""
    print("\n" + "="*60)
    print("Step 1: CDP 질문지 PDF 파싱")
    print("="*60)

    from .parser import CDPQuestionnaireSectionParser

    print(f"  입력: {pdf_path}")

    parser = CDPQuestionnaireSectionParser(pdf_path)
    questions = parser.parse()

    print(f"  파싱 완료: {len(questions)}개 질문")

    # 통계 출력
    stats = {
        "total_questions": len(questions),
        "with_rationale": sum(1 for q in questions if q.rationale),
        "with_response_format": sum(1 for q in questions if q.response_columns),
    }
    print(f"  - Rationale 포함: {stats['with_rationale']}개")
    print(f"  - Response Format 포함: {stats['with_response_format']}개")

    # 저장
    if output_path is None:
        output_path = OUTPUT_DIR / f"{Path(pdf_path).stem}_parsed.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parser.to_json(str(output_path))
    print(f"  저장: {output_path}")

    return questions, output_path


def step2_merge_with_schema(parsed_path: str = None, pdf_path: str = None, output_path: str = None):
    """Step 2: 스키마와 병합"""
    print("\n" + "="*60)
    print("Step 2: 스키마와 병합")
    print("="*60)

    from .merger import merge_questionnaire

    print(f"  스키마: {SCHEMA_DIR}")

    if output_path is None:
        output_path = str(DEFAULT_MERGED_OUTPUT)

    # PDF 직접 파싱 또는 기존 파싱 파일 사용
    result = merge_questionnaire(
        schema_dir=str(SCHEMA_DIR),
        parsed_json_path=str(parsed_path) if parsed_path else None,
        pdf_path=str(pdf_path) if pdf_path else None,
        output_path=output_path,
    )

    print(f"  병합 완료")
    print(f"  저장: {output_path}")

    return result


def step3_validate_output(output_path: str):
    """Step 3: 출력 검증"""
    print("\n" + "="*60)
    print("Step 3: 출력 검증")
    print("="*60)

    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data.get('questions', [])

    # 통계
    def count_all(qs):
        count = len(qs)
        for q in qs:
            if q.get('children'):
                count += count_all(q['children'])
        return count

    total = count_all(questions)

    print(f"  총 질문 수: {total}")
    print(f"  최상위 질문: {len(questions)}개")

    # 샘플 출력
    if questions:
        sample = questions[0]
        print(f"\n  샘플 질문:")
        print(f"    ID: {sample.get('question_id')}")
        print(f"    제목: {sample.get('title_en', '')[:50]}...")
        if sample.get('response_columns'):
            print(f"    응답 컬럼: {len(sample['response_columns'])}개")

    return data


def main():
    parser = argparse.ArgumentParser(
        description="CDP Questionnaire Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 파일 경로
    parser.add_argument("--pdf", "-p", default=str(DEFAULT_QUESTIONNAIRE_PDF),
                        help="CDP 질문지 PDF")
    parser.add_argument("--parsed", help="기존 파싱된 JSON (스킵용)")
    parser.add_argument("--output", "-o", default=str(DEFAULT_MERGED_OUTPUT),
                        help="최종 출력 파일")

    # 모드 옵션
    parser.add_argument("--parse-only", action="store_true",
                        help="파싱만 수행")
    parser.add_argument("--merge-only", action="store_true",
                        help="병합만 수행 (기존 파싱 파일 필요)")

    args = parser.parse_args()

    # 파일 확인
    if not args.merge_only and not Path(args.pdf).exists():
        print(f"ERROR: PDF 파일이 없습니다: {args.pdf}")
        sys.exit(1)

    print("\n" + "="*60)
    print("CDP Questionnaire Pipeline")
    print("="*60)
    print(f"  PDF: {args.pdf}")
    print(f"  출력: {args.output}")

    parsed_path = None

    # Step 1: 파싱
    if not args.merge_only:
        questions, parsed_path = step1_parse_questionnaire(args.pdf)

        if args.parse_only:
            print("\n파싱 완료! (--parse-only)")
            return

    # Step 2: 병합
    if args.merge_only:
        if not args.parsed:
            # 기본 파싱 파일 확인
            default_parsed = OUTPUT_DIR / f"{Path(args.pdf).stem}_parsed.json"
            if default_parsed.exists():
                args.parsed = str(default_parsed)
            else:
                print(f"ERROR: 파싱된 파일이 없습니다. --parsed 옵션 필요")
                sys.exit(1)
        parsed_path = args.parsed

    step2_merge_with_schema(
        parsed_path=parsed_path,
        pdf_path=args.pdf if not args.merge_only else None,
        output_path=args.output,
    )

    # Step 3: 검증
    step3_validate_output(args.output)

    print("\n" + "="*60)
    print("파이프라인 완료!")
    print("="*60)
    print(f"\n최종 출력: {args.output}")
    print("\n다음 단계:")
    print("  python -m model.sustainability --sample 5")


if __name__ == "__main__":
    main()
