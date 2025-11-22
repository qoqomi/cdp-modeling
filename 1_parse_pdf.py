#!/usr/bin/env python3
"""
STEP 1: PDF 파싱
보고서 PDF에서 텍스트를 추출하여 JSON으로 저장
"""

import pdfplumber
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_text_from_pdf(pdf_path):
    """PDF에서 페이지별로 텍스트 추출 (pdfplumber 사용)"""
    print(f"\n📄 PDF 파싱 시작: {pdf_path}")

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        print(f"총 {total_pages}페이지")

        for page_num in tqdm(range(total_pages), desc="페이지 추출"):
            page = pdf.pages[page_num]
            text = page.extract_text()

            if text and text.strip():  # 빈 페이지 제외
                pages.append(
                    {"page_number": page_num + 1, "text": text, "char_count": len(text)}
                )

    return pages


def save_extracted_text(pages, output_path):
    """추출된 텍스트를 JSON으로 저장"""
    data = {
        "total_pages": len(pages),
        "total_chars": sum(p["char_count"] for p in pages),
        "pages": pages,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 저장 완료: {output_path}")
    print(f"   - 페이지 수: {data['total_pages']}")
    print(f"   - 총 글자 수: {data['total_chars']:,}")


def main():
    parser = argparse.ArgumentParser(description="PDF에서 텍스트 추출")
    parser.add_argument("--input", required=True, help="입력 PDF 파일 경로")
    parser.add_argument(
        "--output", default="data/extracted_text.json", help="출력 JSON 파일 경로"
    )

    args = parser.parse_args()

    # 경로 설정
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
        return

    # 텍스트 추출
    pages = extract_text_from_pdf(input_path)

    # 저장
    save_extracted_text(pages, output_path)


if __name__ == "__main__":
    main()
