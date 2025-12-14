"""
CDP Schema Loader
JSON 기반 스키마 로딩 및 관리

버전별로 분리된 스키마 파일을 로드하고 옵션 참조를 해결
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from functools import lru_cache


class CDPSchemaLoader:
    """CDP 스키마 로더"""

    def __init__(self, schema_dir: str = "schemas", version: str = "2025"):
        self.schema_dir = Path(schema_dir)
        self.version = version
        self.version_dir = self.schema_dir / version

        # 캐시
        self._common_options: Optional[Dict] = None
        self._grouped_options: Optional[Dict] = None
        self._module_schemas: Dict[str, Dict] = {}

    def _load_json(self, filename: str) -> Dict:
        """JSON 파일 로드"""
        filepath = self.version_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Schema file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def common_options(self) -> Dict:
        """공통 옵션 로드 (캐시됨)"""
        if self._common_options is None:
            self._common_options = self._load_json("common_options.json")
        return self._common_options

    @property
    def grouped_options(self) -> Dict:
        """그룹 옵션 로드 (캐시됨)"""
        if self._grouped_options is None:
            self._grouped_options = self._load_json("grouped_options.json")
        return self._grouped_options

    def load_module(self, module: str = "2") -> Dict:
        """모듈 스키마 로드"""
        if module not in self._module_schemas:
            self._module_schemas[module] = self._load_json(f"module{module}.json")
        return self._module_schemas[module]

    def get_question_schema(
        self, question_id: str, module: str = "2"
    ) -> Optional[Dict]:
        """질문 스키마 조회"""
        module_data = self.load_module(module)
        return module_data.get("questions", {}).get(question_id)

    def resolve_options(self, column: Dict) -> Dict:
        """
        옵션 참조를 실제 값으로 해결

        options_ref: "yes_no" -> options: ["Yes", "No"]
        grouped_options_ref: "risk_types" -> grouped_options: {...}
        """
        resolved = column.copy()

        # options_ref 해결
        if "options_ref" in resolved:
            ref_key = resolved.pop("options_ref")
            options = self.common_options.get(ref_key)
            if options:
                resolved["options"] = options
            else:
                print(f"Warning: options_ref '{ref_key}' not found")

        # grouped_options_ref 해결
        if "grouped_options_ref" in resolved:
            ref_key = resolved.pop("grouped_options_ref")
            grouped = self.grouped_options.get(ref_key)
            if grouped:
                resolved["grouped_options"] = grouped
            else:
                print(f"Warning: grouped_options_ref '{ref_key}' not found")

        return resolved

    def get_resolved_schema(
        self, question_id: str, module: str = "2"
    ) -> Optional[Dict]:
        """
        모든 참조가 해결된 질문 스키마 반환

        옵션 참조(options_ref, grouped_options_ref)를 실제 값으로 변환
        """
        schema = self.get_question_schema(question_id, module)
        if not schema:
            return None

        resolved = schema.copy()
        resolved["columns"] = [
            self.resolve_options(col) for col in schema.get("columns", [])
        ]

        return resolved

    def get_all_question_ids(self, module: str = "2") -> List[str]:
        """모듈의 모든 질문 ID 반환"""
        module_data = self.load_module(module)
        return list(module_data.get("questions", {}).keys())

    def validate_response(
        self, question_id: str, response: Dict[str, Any], module: str = "2"
    ) -> List[str]:
        """응답 유효성 검증"""
        errors = []
        schema = self.get_resolved_schema(question_id, module)

        if not schema:
            return [f"Unknown question ID: {question_id}"]

        columns = schema.get("columns", [])

        for col in columns:
            col_id = col["id"]
            col_type = col["type"]
            required = col.get("required", False)

            value = response.get(col_id)

            # Required 체크
            if required and (value is None or value == "" or value == []):
                errors.append(f"Required field '{col_id}' is missing")
                continue

            if value is None or value == "":
                continue

            # 타입별 검증
            if col_type == "select":
                options = col.get("options", [])
                if options and value not in options:
                    errors.append(f"Invalid option '{value}' for '{col_id}'")

            elif col_type == "multiselect":
                options = col.get("options", [])
                if isinstance(value, list):
                    for v in value:
                        if options and v not in options:
                            errors.append(f"Invalid option '{v}' for '{col_id}'")
                else:
                    errors.append(f"'{col_id}' should be a list")

            elif col_type in ("number", "percentage"):
                min_val = col.get("min_value")
                max_val = col.get("max_value")
                try:
                    num_val = float(value)
                    if min_val is not None and num_val < min_val:
                        errors.append(f"'{col_id}' must be >= {min_val}")
                    if max_val is not None and num_val > max_val:
                        errors.append(f"'{col_id}' must be <= {max_val}")
                except (ValueError, TypeError):
                    errors.append(f"'{col_id}' must be a number")

            elif col_type == "textarea":
                max_length = col.get("max_length")
                if max_length and len(str(value)) > max_length:
                    errors.append(f"'{col_id}' exceeds max length {max_length}")

        return errors

    def export_resolved_schema(self, output_path: str, module: str = "2"):
        """모든 참조가 해결된 스키마를 JSON으로 내보내기"""
        module_data = self.load_module(module)

        resolved = {
            "version": module_data.get("version"),
            "module": module_data.get("module"),
            "title": module_data.get("title"),
            "questions": {},
        }

        for qid in self.get_all_question_ids(module):
            resolved["questions"][qid] = self.get_resolved_schema(qid, module)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(resolved, f, ensure_ascii=False, indent=2)

        print(f"Resolved schema exported to {output_path}")

    def list_available_versions(self) -> List[str]:
        """사용 가능한 버전 목록"""
        if not self.schema_dir.exists():
            return []
        return [d.name for d in self.schema_dir.iterdir() if d.is_dir()]


# 편의 함수
_default_loader: Optional[CDPSchemaLoader] = None


def get_loader(version: str = "2025") -> CDPSchemaLoader:
    """기본 로더 인스턴스 반환"""
    global _default_loader
    if _default_loader is None or _default_loader.version != version:
        _default_loader = CDPSchemaLoader(version=version)
    return _default_loader


def get_schema(question_id: str, version: str = "2025") -> Optional[Dict]:
    """질문 스키마 조회 (해결됨)"""
    return get_loader(version).get_resolved_schema(question_id)


def get_all_question_ids(version: str = "2025") -> List[str]:
    """모든 질문 ID 반환"""
    return get_loader(version).get_all_question_ids()


def validate_response(
    question_id: str, response: Dict, version: str = "2025"
) -> List[str]:
    """응답 유효성 검증"""
    return get_loader(version).validate_response(question_id, response)


if __name__ == "__main__":
    # 테스트
    loader = CDPSchemaLoader()

    print("CDP Schema Loader Test")
    print("=" * 50)

    print(f"\nAvailable versions: {loader.list_available_versions()}")
    print(f"Question IDs: {loader.get_all_question_ids()}")

    # 2.2.2 스키마 테스트
    schema = loader.get_resolved_schema("2.2.2")
    if schema:
        print(f"\n2.2.2 Schema:")
        print(f"  Title: {schema['title']}")
        print(f"  Columns: {len(schema['columns'])}")
        for col in schema["columns"][:3]:
            print(f"    - {col['id']}: {col['type']}")
            if "options" in col:
                print(f"      Options: {col['options'][:3]}...")
        if len(schema["columns"]) > 3:
            print(f"    ... and {len(schema['columns']) - 3} more")

    # 해결된 스키마 내보내기
    loader.export_resolved_schema("output/cdp_resolved_schema.json")
