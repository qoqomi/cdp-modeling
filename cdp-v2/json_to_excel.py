"""
CDP JSON to Excel Converter
result.json을 엑셀로 변환
"""

import json
import pandas as pd
from pathlib import Path


def is_section(q):
    """섹션인지 질문인지 판별"""
    input_type_data = q.get("inputType", {})
    input_type = input_type_data.get("type", "") if isinstance(input_type_data, dict) else ""
    has_sub_questions = bool(q.get("subQuestions", []))
    has_table_columns = "tableColumns" in input_type_data if isinstance(input_type_data, dict) else False

    # 테이블이거나 하위 질문이 있으면 section
    if input_type == "table" or has_table_columns:
        return True
    if has_sub_questions:
        return True
    return False


def flatten_questions(questions, rows=None, section_id=None):
    """질문들을 평탄화하여 행 리스트로 변환"""
    if rows is None:
        rows = []

    for q in questions:
        q_id = q.get("questionId", "")
        title = q.get("title", "")
        parent_id = q.get("parentId", "")
        input_type_data = q.get("inputType", {})
        input_type = input_type_data.get("type", "") if isinstance(input_type_data, dict) else str(input_type_data)
        row_type = q.get("rowType", "")
        required = q.get("required", False)

        # 타입 판별
        item_type = "section" if is_section(q) else "question"

        # 옵션 추출
        options = []
        table_columns = []

        if isinstance(input_type_data, dict):
            if "options" in input_type_data:
                options = [opt.get("label", "") for opt in input_type_data.get("options", [])]
            elif "tableColumns" in input_type_data:
                table_columns = input_type_data.get("tableColumns", [])

        # 응답 추출
        response_data = q.get("response", {})
        response_value = ""
        response_status = response_data.get("status", "")
        table_responses = {}

        if "value" in response_data:
            val = response_data["value"]
            if isinstance(val, list):
                response_value = ", ".join(str(v) for v in val)
            else:
                response_value = str(val) if val else ""
        elif "rows" in response_data:
            table_rows_data = response_data.get("rows", [])
            if table_rows_data:
                fields = table_rows_data[0].get("fields", {})
                table_responses = {k: v.get("value", "") for k, v in fields.items()}

        # 계층 깊이 계산 (숫자)
        depth = len(q_id.split(".")) - 1

        # 메인 행
        row = {
            "type": item_type,
            "sectionId": section_id,
            "questionId": q_id,
            "depth": depth,
            "parentId": parent_id if parent_id else None,
            "title": title,
            "inputType": input_type if item_type == "question" else "",
            "rowType": row_type if item_type == "question" else "",
            "required": (1 if required else 0) if item_type == "question" else None,
            "options": " | ".join(options) if options else "",
            "responseValue": response_value if item_type == "question" else "",
            "responseStatus": response_status if item_type == "question" else "",
        }
        rows.append(row)

        # 테이블 컬럼들을 별도 행으로 추가 (question 타입)
        for col in table_columns:
            col_id = col.get("columnId", "")
            col_header = col.get("header", "")
            col_input_type = col.get("inputType", {})
            col_type = col_input_type.get("type", "") if isinstance(col_input_type, dict) else ""
            col_options = []
            if isinstance(col_input_type, dict) and "options" in col_input_type:
                col_options = [opt.get("label", "") for opt in col_input_type.get("options", [])]

            col_response = table_responses.get(col_id, "")

            col_row = {
                "type": "question",  # 컬럼은 항상 question
                "sectionId": section_id,
                "questionId": q_id,
                "depth": depth + 1,  # 컬럼은 한 단계 더 깊음
                "parentId": q_id,
                "title": col_header,
                "inputType": col_type,
                "rowType": "",
                "required": 0,
                "options": " | ".join(col_options) if col_options else "",
                "responseValue": col_response if col_response else "",
                "responseStatus": "complete" if col_response else "",
            }
            rows.append(col_row)

        # 하위 질문 재귀 처리
        sub_questions = q.get("subQuestions", [])
        if sub_questions:
            flatten_questions(sub_questions, rows, section_id)

    return rows


def json_to_excel(json_path: str, excel_path: str):
    """JSON 파일을 Excel로 변환"""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_rows = []

    # 섹션별로 처리
    for section in data.get("sections", []):
        section_id_str = section.get("sectionId", "")
        # sectionId를 숫자로 변환
        try:
            section_id = int(section_id_str)
        except (ValueError, TypeError):
            section_id = section_id_str

        questions = section.get("questions", [])
        rows = flatten_questions(questions, section_id=section_id)
        all_rows.extend(rows)

    # DataFrame 생성
    df = pd.DataFrame(all_rows)

    # 컬럼 순서 정리
    columns = [
        "type",
        "sectionId",
        "questionId",
        "depth",
        "parentId",
        "title",
        "inputType",
        "rowType",
        "required",
        "options",
        "responseValue",
        "responseStatus",
    ]
    df = df[columns]

    # 숫자 컬럼 타입 지정
    df["sectionId"] = pd.to_numeric(df["sectionId"], errors="coerce").astype("Int64")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce").astype("Int64")
    df["required"] = pd.to_numeric(df["required"], errors="coerce").astype("Int64")

    # Excel 저장
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="CDP Questions", index=False)

        # 컬럼 너비 자동 조정
        worksheet = writer.sheets["CDP Questions"]
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(col)
            )
            adjusted_width = min(max_length + 2, 50)
            # A-Z까지만 단일 문자, 그 이후는 AA, AB 등
            if idx < 26:
                col_letter = chr(65 + idx)
            else:
                col_letter = chr(64 + idx // 26) + chr(65 + idx % 26)
            worksheet.column_dimensions[col_letter].width = adjusted_width

    print(f"Excel 파일 생성 완료: {excel_path}")
    print(f"총 {len(df)}개 행 변환됨")
    print(f"  - section: {len(df[df['type'] == 'section'])}개")
    print(f"  - question: {len(df[df['type'] == 'question'])}개")

    return df


if __name__ == "__main__":
    import sys

    json_path = sys.argv[1] if len(sys.argv) > 1 else "result.json"
    excel_path = sys.argv[2] if len(sys.argv) > 2 else "cdp_questions.xlsx"

    json_to_excel(json_path, excel_path)
