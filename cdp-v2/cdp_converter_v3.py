"""
CDP PDF Parser - JSON Schema Converter (v3)
테이블 컬럼 자동 매핑 + 응답값 추출
"""

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from cdp_models_v2 import (
    ParsedQuestion, InputType, RowType, 
    OPTION_GROUP_HEADERS, CDP_TABLE_COLUMNS
)


class CDPJsonConverter:
    """파싱된 CDP 데이터를 JSON 스키마로 변환 (v3)"""
    
    def __init__(self, questions: List[ParsedQuestion], org_name: str = ""):
        self.questions = questions
        self.org_name = org_name
        self.option_value_map: Dict[str, str] = {}
    
    def convert(self) -> Dict[str, Any]:
        """전체 변환 실행"""
        return {
            "metadata": self._create_metadata(),
            "sections": self._create_sections()
        }
    
    def _create_metadata(self) -> Dict[str, Any]:
        return {
            "documentId": str(uuid.uuid4()),
            "version": "1.0.0",
            "reportingYear": datetime.now().year,
            "organizationName": self.org_name,
            "submissionStatus": "draft",
            "lastModified": datetime.now().isoformat() + "Z",
            "createdAt": datetime.now().isoformat() + "Z",
            "createdBy": "pdf-parser"
        }
    
    def _create_sections(self) -> List[Dict[str, Any]]:
        section_map: Dict[str, List[ParsedQuestion]] = {}
        
        for q in self.questions:
            section_id = q.question_id.split('.')[0]
            if section_id not in section_map:
                section_map[section_id] = []
            section_map[section_id].append(q)
        
        sections = []
        for section_id, questions in sorted(section_map.items()):
            sections.append({
                "sectionId": section_id,
                "title": f"Section {section_id}",
                "description": "",
                "questions": self._create_questions(questions)
            })
        
        return sections
    
    def _create_questions(self, questions: List[ParsedQuestion]) -> List[Dict[str, Any]]:
        root_questions = [q for q in questions if q.parent_id is None]
        result = []
        
        for q in root_questions:
            question_json = self._create_question(q, questions)
            result.append(question_json)
        
        return result
    
    def _create_question(self, q: ParsedQuestion, all_questions: List[ParsedQuestion]) -> Dict[str, Any]:
        children = [child for child in all_questions if child.parent_id == q.question_id]
        
        # 테이블 타입 결정 (미리 정의된 테이블인지 확인)
        is_predefined_table = q.question_id in CDP_TABLE_COLUMNS
        
        question_json = {
            "questionId": q.question_id,
            "parentId": q.parent_id,
            "title": q.title,
            "inputType": self._create_input_type(q, is_predefined_table),
            "required": True,
            "rowType": q.row_type.value,
            "response": self._create_response(q, is_predefined_table)
        }
        
        if children:
            question_json["subQuestions"] = [
                self._create_question(child, all_questions) 
                for child in children
            ]
        
        return question_json
    
    def _create_input_type(self, q: ParsedQuestion, is_predefined_table: bool) -> Dict[str, Any]:
        """입력 타입 정의 생성"""
        
        # 미리 정의된 테이블 컬럼 사용
        if is_predefined_table:
            return {
                "type": "table",
                "tableColumns": CDP_TABLE_COLUMNS[q.question_id]
            }
        
        # 일반 입력 타입
        input_type: Dict[str, Any] = {"type": q.input_type.value}
        
        if q.input_type in [InputType.SINGLE_SELECT, InputType.MULTI_SELECT]:
            if q.options:
                input_type["options"] = self._create_options(q.options)
        elif q.input_type == InputType.GROUPED_MULTI_SELECT:
            if q.options:
                input_type["groupedOptions"] = self._create_grouped_options(q.options)
        elif q.input_type == InputType.TEXTAREA:
            input_type["maxLength"] = 5000
        
        return input_type
    
    def _create_options(self, options: List) -> List[Dict[str, str]]:
        result = []
        seen = set()
        for opt in options:
            value = self._label_to_value(opt.label)
            if value not in seen:
                result.append({"value": value, "label": opt.label})
                self.option_value_map[opt.label] = value
                seen.add(value)
        return result
    
    def _create_grouped_options(self, options: List) -> List[Dict[str, Any]]:
        groups: Dict[str, List] = {}
        
        for opt in options:
            group_name = opt.group or "Other"
            if group_name not in groups:
                groups[group_name] = []
            
            value = self._label_to_value(opt.label)
            groups[group_name].append({"value": value, "label": opt.label})
            self.option_value_map[opt.label] = value
        
        result = []
        for group_name in OPTION_GROUP_HEADERS:
            if group_name in groups:
                result.append({
                    "groupId": self._label_to_value(group_name),
                    "groupLabel": group_name,
                    "options": groups[group_name]
                })
        
        for group_name, opts in groups.items():
            if group_name not in OPTION_GROUP_HEADERS:
                result.append({
                    "groupId": self._label_to_value(group_name),
                    "groupLabel": group_name,
                    "options": opts
                })
        
        return result
    
    def _create_response(self, q: ParsedQuestion, is_predefined_table: bool) -> Dict[str, Any]:
        """응답 데이터 생성"""
        
        # 미리 정의된 테이블인 경우 - raw_text에서 응답 추출
        if is_predefined_table:
            return self._extract_table_response(q)
        
        # 일반 응답
        response: Dict[str, Any] = {"status": "empty"}
        
        selected = [opt for opt in q.options if opt.is_selected]
        
        if q.input_type == InputType.SINGLE_SELECT:
            if selected:
                response["value"] = self._label_to_value(selected[0].label)
                response["status"] = "complete"
        
        elif q.input_type in [InputType.MULTI_SELECT, InputType.GROUPED_MULTI_SELECT]:
            if selected:
                response["value"] = [self._label_to_value(opt.label) for opt in selected]
                response["status"] = "complete"
        
        elif q.input_type == InputType.TEXTAREA:
            if q.text_response:
                response["value"] = q.text_response
                response["status"] = "complete"
        
        response["lastModified"] = datetime.now().isoformat() + "Z"
        return response
    
    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화 - 줄바꿈, 다중 공백 등을 단일 공백으로"""
        # 줄바꿈을 공백으로
        text = text.replace('\n', ' ').replace('\r', ' ')
        # 다중 공백을 단일 공백으로
        text = re.sub(r'\s+', ' ', text)
        return text

    def _find_checked_option(self, normalized_text: str, options: List[Dict]) -> Optional[str]:
        """체크된 옵션 찾기 - 여러 전략 사용"""
        for opt in options:
            label = opt["label"]
            value = opt["value"]

            # 전략 1: 정확한 매칭 (정규화된 텍스트에서)
            pattern = r'☑\s*' + r'\s*'.join(re.escape(word) for word in label.split())
            if re.search(pattern, normalized_text, re.IGNORECASE):
                return value

            # 전략 2: 라벨의 핵심 키워드로 매칭 (3단어 이상인 경우)
            words = label.split()
            if len(words) >= 3:
                # 첫 2단어 + 마지막 단어로 매칭 시도
                key_pattern = r'☑\s*' + r'.*?'.join([
                    re.escape(words[0]),
                    re.escape(words[-1])
                ])
                if re.search(key_pattern, normalized_text, re.IGNORECASE):
                    return value

            # 전략 3: value 기반 매칭 (both, yes, no 등)
            if value in ['both', 'yes', 'no']:
                # "Both"가 체크되어 있고 컨텍스트가 맞는지 확인
                simple_pattern = r'☑\s*' + re.escape(value.capitalize())
                if re.search(simple_pattern, normalized_text):
                    return value

        return None

    def _extract_table_response(self, q: ParsedQuestion) -> Dict[str, Any]:
        """테이블 응답 추출 - raw_text에서 체크된 값 찾기"""
        columns = CDP_TABLE_COLUMNS.get(q.question_id, [])
        raw_text = q.raw_text
        normalized_text = self._normalize_text(raw_text)

        fields = {}

        for col in columns:
            col_id = col["columnId"]
            options = col.get("inputType", {}).get("options", [])

            # 체크된 옵션 찾기
            found_value = self._find_checked_option(normalized_text, options)

            if found_value:
                fields[col_id] = {
                    "value": found_value,
                    "status": "complete"
                }
            else:
                fields[col_id] = {"value": None, "status": "empty"}

        has_value = any(f["value"] is not None for f in fields.values())

        return {
            "rows": [{
                "rowId": "row-1",
                "rowIndex": 0,
                "fields": fields
            }] if has_value else [],
            "status": "complete" if has_value else "empty"
        }
    
    def _label_to_value(self, label: str) -> str:
        if label in self.option_value_map:
            return self.option_value_map[label]
        
        value = label.lower()
        value = value.replace(',', '').replace('/', '_').replace('-', '_')
        value = value.replace('(', '').replace(')', '')
        value = '_'.join(value.split())
        
        if len(value) > 50:
            words = value.split('_')[:5]
            value = '_'.join(words)
        
        return value
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.convert(), indent=indent, ensure_ascii=False)
    
    def save(self, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
