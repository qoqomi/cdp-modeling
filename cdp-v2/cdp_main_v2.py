"""
CDP PDF Parser - Main Execution (v2)
테이블 구조 지원 버전
"""

import argparse
import sys
from pathlib import Path

from cdp_models_v2 import ParsedQuestion, InputType, RowType, CDP_TABLE_COLUMNS
from cdp_parser_v2 import CDPTextExtractor, CDPStructureParser
from cdp_converter_v3 import CDPJsonConverter


def parse_cdp_pdf(pdf_path: str, output_path: str, org_name: str = "") -> dict:
    """CDP PDF 파싱 메인 함수"""
    
    # 1. 텍스트 추출
    extractor = CDPTextExtractor(pdf_path)
    raw_text, page_count = extractor.extract()
    print(f"Extracted {page_count} pages")
    
    # 2. 구조 파싱
    parser = CDPStructureParser(raw_text)
    questions = parser.parse()
    print(f"Parsed {len(questions)} questions")
    
    # 3. JSON 변환
    converter = CDPJsonConverter(questions, org_name)
    result = converter.convert()
    
    # 4. 저장
    converter.save(output_path)
    print(f"Saved to {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="CDP PDF to JSON Parser")
    parser.add_argument("--input", "-i", required=True, help="Input PDF path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path")
    parser.add_argument("--org", default="", help="Organization name")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        sys.exit(1)
    
    parse_cdp_pdf(args.input, args.output, args.org)


if __name__ == "__main__":
    main()
