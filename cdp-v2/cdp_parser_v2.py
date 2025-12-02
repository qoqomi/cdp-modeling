"""
CDP PDF Parser - Text Extraction & Structure Parser (v2)
테이블 구조 감지 개선 버전
"""

import re
from typing import Optional, List, Tuple
from cdp_models_v2 import (
    ParsedQuestion,
    ParsedOption,
    InputType,
    RowType,
    CDP_PATTERNS,
    OPTION_GROUP_HEADERS,
    NARRATIVE_PATTERNS,
    NARRATIVE_TRAILING_MARKERS,
    NARRATIVE_HINTS,
    NarrativePattern,
)


class NarrativeExtractor:
    """
    서술식 질문의 제목과 응답을 분리하는 클래스

    확장성:
    - NARRATIVE_PATTERNS에 새 패턴 추가로 쉽게 확장
    - 다양한 분리 전략 지원 (패턴 매칭, 물음표 기반, 길이 기반)
    """

    def __init__(self, patterns: List[NarrativePattern] = None):
        self.patterns = patterns or NARRATIVE_PATTERNS
        self.trailing_markers = NARRATIVE_TRAILING_MARKERS
        self.hints = NARRATIVE_HINTS

    def extract(self, content: str) -> Tuple[str, Optional[str]]:
        """
        서술식 콘텐츠에서 제목과 응답을 분리

        Args:
            content: 질문 ID 이후의 텍스트 콘텐츠

        Returns:
            (title, response) 튜플. 응답이 없으면 (title, None)
        """
        # 전처리: row 마커 제거
        content_clean = self._clean_content(content)

        if not content_clean:
            return "", None

        # 1단계: 정의된 패턴으로 분리 시도
        result = self._extract_by_pattern(content_clean)
        if result:
            return result

        # 2단계: 물음표 기반 분리 시도
        result = self._extract_by_question_mark(content_clean)
        if result:
            return result

        # 3단계: 줄 단위 분리 시도
        result = self._extract_by_lines(content_clean)
        if result:
            return result

        # 4단계: 문장 단위 분리 시도
        result = self._extract_by_sentence(content_clean)
        if result:
            return result

        # Fallback: 전체를 제목으로
        return content_clean, None

    def _clean_content(self, content: str) -> str:
        """콘텐츠 전처리"""
        # [Fixed row], [Add row] 등 마커 제거
        cleaned = re.sub(r"\[.*?row\]", "", content, flags=re.IGNORECASE)
        # 다중 공백 정리
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _remove_trailing_markers(self, text: str) -> str:
        """응답 끝의 후행 마커 제거"""
        result = text.strip()
        for marker in self.trailing_markers:
            if result.endswith(marker):
                result = result[:-len(marker)].strip()
        return result

    def _is_valid_response(self, response: str, min_length: int = None) -> bool:
        """응답이 유효한지 검증"""
        if not response:
            return False
        min_len = min_length or self.hints.get("default_min_response_length", 50)
        return len(response.strip()) >= min_len

    def _extract_by_pattern(self, content: str) -> Optional[Tuple[str, Optional[str]]]:
        """정의된 패턴으로 제목-응답 분리"""
        for pattern in self.patterns:
            # 패턴이 콘텐츠에 포함되어 있는지 확인
            pattern_pos = content.find(pattern.title)
            if pattern_pos == -1:
                continue

            # 패턴 위치가 시작 부근(100자 이내)인 경우만 처리
            if pattern_pos > 100:
                continue

            # 구분자가 있으면 구분자로 분리, 없으면 패턴 제목 바로 뒤에서 분리
            if pattern.separator:
                sep_pos = content.find(pattern.separator, pattern_pos + len(pattern.title))
                if sep_pos > 0:
                    title = content[pattern_pos:sep_pos].strip()
                    response = content[sep_pos + len(pattern.separator):].strip()
                else:
                    title = pattern.title
                    response = content[pattern_pos + len(pattern.title):].strip()
            else:
                title = pattern.title
                response = content[pattern_pos + len(pattern.title):].strip()

            # 응답 후처리
            response = self._remove_trailing_markers(response)

            if self._is_valid_response(response, pattern.min_response_length):
                return title, response
            else:
                return title, None

        return None

    def _extract_by_question_mark(self, content: str) -> Optional[Tuple[str, Optional[str]]]:
        """물음표 기반 제목-응답 분리"""
        question_mark_pos = content.find("?")

        # 물음표가 300자 이내에 있어야 함
        if question_mark_pos <= 0 or question_mark_pos >= 300:
            return None

        title = content[:question_mark_pos + 1].strip()
        response = content[question_mark_pos + 1:].strip()
        response = self._remove_trailing_markers(response)

        if self._is_valid_response(response):
            return title, response
        return title, None

    def _extract_by_lines(self, content: str) -> Optional[Tuple[str, Optional[str]]]:
        """줄 단위 분리 (첫 줄이 짧으면 제목으로)"""
        lines = [line.strip() for line in content.split("\n") if line.strip()]

        if not lines:
            return None

        first_line = lines[0]

        # 첫 줄이 100자 미만이면 제목으로 간주
        if len(first_line) < 100:
            title = first_line
            if len(lines) > 1:
                response = " ".join(lines[1:])
                response = self._remove_trailing_markers(response)
                if self._is_valid_response(response):
                    return title, response
            return title, None

        return None

    def _extract_by_sentence(self, content: str) -> Optional[Tuple[str, Optional[str]]]:
        """문장 단위 분리 (첫 문장을 제목으로)"""
        # 마침표나 느낌표로 끝나는 첫 문장 찾기
        sentence_end = -1
        for i, char in enumerate(content):
            if char in ".!" and i < 200:
                sentence_end = i
                break

        if sentence_end <= 0:
            return None

        title = content[:sentence_end + 1].strip()
        response = content[sentence_end + 1:].strip()
        response = self._remove_trailing_markers(response)

        if self._is_valid_response(response):
            return title, response
        return title, None

    def has_narrative_content(self, block: str) -> bool:
        """블록에 서술식 응답이 있는지 감지"""
        # 제외 패턴이 있으면 서술식 아님
        for exclude in self.hints.get("exclude_patterns", []):
            if exclude in block:
                return False

        # 정의된 패턴 중 하나라도 포함되어 있으면 서술식
        for pattern in self.patterns:
            if pattern.title in block:
                return True

        return False


class CDPTextExtractor:
    """PDF에서 텍스트 추출"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract(self) -> Tuple[str, int]:
        """PDF 텍스트 추출"""
        try:
            import pdfplumber

            with pdfplumber.open(self.pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text, len(pdf.pages)
        except ImportError:
            raise ImportError("pdfplumber 설치 필요: pip install pdfplumber")


class CDPTableExtractor:
    """PDF에서 테이블 구조 추출"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_tables(self) -> dict:
        """페이지별 테이블 추출 → 질문ID와 매핑"""
        import pdfplumber

        tables_by_question = {}

        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                tables = page.extract_tables()

                if not tables:
                    continue

                # 이 페이지에서 질문 ID 찾기
                question_ids = re.findall(CDP_PATTERNS["question_id"], text)

                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    # 테이블 위치 기반으로 가장 가까운 질문 ID 찾기
                    # (간단히: 해당 페이지의 첫 번째 질문에 매핑)
                    if question_ids:
                        q_id = question_ids[0]
                        tables_by_question[q_id] = self._parse_table(table)

        return tables_by_question

    def _parse_table(self, table: List[List]) -> dict:
        """테이블 파싱 → 헤더와 데이터 분리"""
        if len(table) < 2:
            return {"headers": [], "rows": []}

        # 첫 번째 행 = 헤더
        headers = [cell.strip() if cell else "" for cell in table[0]]

        # 나머지 행 = 데이터
        rows = []
        for row in table[1:]:
            row_data = {}
            for i, cell in enumerate(row):
                if i < len(headers) and headers[i]:
                    row_data[headers[i]] = cell.strip() if cell else ""
            rows.append(row_data)

        return {"headers": headers, "rows": rows}


class CDPStructureParser:
    """추출된 텍스트를 구조화된 데이터로 파싱 (v2)"""

    # 테이블 헤더로 자주 쓰이는 패턴
    TABLE_HEADER_PATTERNS = [
        "Process in place",
        "Dependencies and/or impacts",
        "Risks and/or opportunities",
        "Select from",
        "Select all that apply",
        "evaluated in this process",
        "informed by",
    ]

    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.questions: List[ParsedQuestion] = []
        self.narrative_extractor = NarrativeExtractor()

    def parse(self) -> List[ParsedQuestion]:
        """전체 파싱 실행"""
        question_blocks = self._split_into_question_blocks()

        for block in question_blocks:
            question = self._parse_question_block(block)
            if question:
                self.questions.append(question)

        self._set_parent_relationships()
        return self.questions

    def _split_into_question_blocks(self) -> List[str]:
        """텍스트를 질문 ID 기준으로 분리"""
        pattern = CDP_PATTERNS["question_id"]
        matches = list(re.finditer(pattern, self.raw_text))

        if not matches:
            return []

        blocks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(self.raw_text)
            blocks.append(self.raw_text[start:end].strip())

        return blocks

    def _parse_question_block(self, block: str) -> Optional[ParsedQuestion]:
        """개별 질문 블록 파싱"""
        id_match = re.match(CDP_PATTERNS["question_id"], block)
        if not id_match:
            return None

        question_id = id_match.group(1)
        content_after_id = block[id_match.end():].strip()

        # 테이블 구조 감지
        table_info = self._detect_table_structure(block)

        # 입력 타입 및 옵션 파싱
        if table_info["is_table"]:
            input_type = InputType.TABLE
            options = []
            table_columns = table_info["columns"]
        else:
            input_type, options = self._parse_input_type_and_options(block)
            table_columns = []

        # Row 타입 (서술식 판단에 활용)
        row_type = self._detect_row_type(block)

        # ============================================================
        # 서술식 응답 분리 로직 (확장된 버전)
        # - 입력 타입과 관계없이 서술식 패턴이 있으면 분리 시도
        # ============================================================
        title = None
        text_response = None

        # 1. TEXTAREA 타입이거나 서술식 패턴이 감지되면 NarrativeExtractor 사용
        has_narrative = self.narrative_extractor.has_narrative_content(content_after_id)

        if input_type == InputType.TEXTAREA or has_narrative:
            title, text_response = self.narrative_extractor.extract(content_after_id)

            # 서술식 응답이 있으면 입력 타입을 TEXTAREA로 변경
            if text_response and input_type not in [InputType.TABLE]:
                input_type = InputType.TEXTAREA
        else:
            # 2. 서술식이 아닌 경우 기존 방식으로 제목 추출
            title = self._extract_title(block, id_match.end())
            text_response = self._extract_text_response(block, options)

        # 제목이 비어있으면 기존 방식으로 추출
        if not title:
            title = self._extract_title(block, id_match.end())

        question = ParsedQuestion(
            question_id=question_id,
            title=title,
            input_type=input_type,
            options=options,
            text_response=text_response,
            row_type=row_type,
            raw_text=block,
        )

        # 테이블 컬럼 정보 추가 (커스텀 속성)
        if table_columns:
            question.table_columns = table_columns
            question.table_responses = table_info.get("responses", {})

        return question

    def _extract_title(self, block: str, start_pos: int) -> str:
        """질문 제목만 추출 (테이블 헤더 제외)"""
        # 테이블 헤더 패턴이 시작되는 위치 찾기
        title_end = len(block)

        for pattern in self.TABLE_HEADER_PATTERNS:
            pos = block.find(pattern)
            if pos > start_pos:
                title_end = min(title_end, pos)

        # Select from, 체크박스 등 마커 위치도 확인
        for marker in ["Select from:", "Select all that apply", "☑", "☐"]:
            pos = block.find(marker)
            if pos > start_pos:
                title_end = min(title_end, pos)

        title = block[start_pos:title_end].strip()
        title = re.sub(r"\s+", " ", title)  # 공백 정리

        # 끝에 물음표가 있으면 거기서 자르기 (더 정확한 제목 추출)
        question_mark_pos = title.find("?")
        if question_mark_pos > 0:
            title = title[: question_mark_pos + 1]

        return title

    def _detect_table_structure(self, block: str) -> dict:
        """테이블 구조 감지"""
        result = {"is_table": False, "columns": [], "responses": {}}

        # 테이블 헤더 패턴 감지
        found_headers = []
        for pattern in self.TABLE_HEADER_PATTERNS:
            if pattern in block and pattern not in [
                "Select from",
                "Select all that apply",
            ]:
                found_headers.append(pattern)

        if len(found_headers) >= 1:
            result["is_table"] = True

            # 컬럼 정보 추출
            columns = self._extract_table_columns(block, found_headers)
            result["columns"] = columns

            # 각 컬럼별 응답 추출
            result["responses"] = self._extract_table_responses(block, columns)

        return result

    def _extract_table_columns(self, block: str, headers: List[str]) -> List[dict]:
        """테이블 컬럼 정보 추출"""
        columns = []

        # 알려진 CDP 테이블 컬럼 패턴
        known_columns = {
            "Process in place": {
                "columnId": "process_in_place",
                "header": "Process in place",
                "options": [
                    {"value": "yes", "label": "Yes"},
                    {"value": "no", "label": "No"},
                ],
            },
            "Dependencies and/or impacts": {
                "columnId": "dependencies_impacts",
                "header": "Dependencies and/or impacts evaluated in this process",
                "options": [
                    {"value": "dependencies", "label": "Dependencies"},
                    {"value": "impacts", "label": "Impacts"},
                    {"value": "both", "label": "Both dependencies and impacts"},
                ],
            },
            "Risks and/or opportunities": {
                "columnId": "risks_opportunities",
                "header": "Risks and/or opportunities evaluated in this process",
                "options": [
                    {"value": "risks", "label": "Risks"},
                    {"value": "opportunities", "label": "Opportunities"},
                    {"value": "both", "label": "Both risks and opportunities"},
                ],
            },
            "informed by": {
                "columnId": "informed_by_dependencies",
                "header": "Is this process informed by the dependencies and/or impacts process?",
                "options": [
                    {"value": "yes", "label": "Yes"},
                    {"value": "no", "label": "No"},
                ],
            },
        }

        for header in headers:
            for key, col_info in known_columns.items():
                if key.lower() in header.lower():
                    columns.append(col_info)
                    break

        return columns

    def _extract_table_responses(self, block: str, columns: List[dict]) -> dict:
        """테이블 각 컬럼의 응답값 추출"""
        responses = {}

        for col in columns:
            col_id = col["columnId"]
            options = col.get("options", [])

            # 해당 컬럼의 체크된 옵션 찾기
            for opt in options:
                label = opt["label"]
                # ☑ 바로 뒤에 이 라벨이 있는지 확인
                pattern = r"☑\s*" + re.escape(label)
                if re.search(pattern, block, re.IGNORECASE):
                    responses[col_id] = {"value": opt["value"], "status": "complete"}
                    break

            # 못 찾으면 empty
            if col_id not in responses:
                responses[col_id] = {"value": None, "status": "empty"}

        return responses

    def _parse_input_type_and_options(
        self, block: str
    ) -> Tuple[InputType, List[ParsedOption]]:
        """입력 타입과 옵션 추출 (비테이블 질문용)"""
        options = []

        # 체크된 항목 추출
        checked = re.findall(CDP_PATTERNS["checkbox_checked"], block)
        for item in checked:
            clean_item = item.strip()
            if clean_item and len(clean_item) > 1:
                group = self._detect_option_group(block, clean_item)
                options.append(
                    ParsedOption(label=clean_item, is_selected=True, group=group)
                )

        # 체크 안된 항목 추출
        unchecked = re.findall(CDP_PATTERNS["checkbox_unchecked"], block)
        for item in unchecked:
            clean_item = item.strip()
            if clean_item and len(clean_item) > 1:
                group = self._detect_option_group(block, clean_item)
                options.append(
                    ParsedOption(label=clean_item, is_selected=False, group=group)
                )

        # 입력 타입 결정
        if not options:
            input_type = InputType.TEXTAREA
        elif "Select from:" in block:
            input_type = InputType.SINGLE_SELECT
        elif "Select all that apply" in block:
            has_groups = any(opt.group for opt in options)
            input_type = (
                InputType.GROUPED_MULTI_SELECT if has_groups else InputType.MULTI_SELECT
            )
        else:
            input_type = InputType.MULTI_SELECT

        return input_type, options

    def _detect_option_group(self, block: str, option_label: str) -> Optional[str]:
        """옵션이 속한 그룹 감지"""
        for header in OPTION_GROUP_HEADERS:
            header_pos = block.find(header)
            option_pos = block.find(option_label)

            if header_pos >= 0 and option_pos > header_pos:
                next_header_pos = len(block)
                for other_header in OPTION_GROUP_HEADERS:
                    if other_header != header:
                        pos = block.find(other_header, header_pos + len(header))
                        if pos > 0:
                            next_header_pos = min(next_header_pos, pos)

                if option_pos < next_header_pos:
                    return header

        return None

    def _extract_text_response(
        self, block: str, options: List[ParsedOption]
    ) -> Optional[str]:
        """텍스트 응답 추출"""
        selected_options = [opt for opt in options if opt.is_selected]
        if not selected_options:
            return None

        last_checkbox_pos = 0
        for opt in selected_options:
            pos = block.rfind(opt.label)
            if pos > last_checkbox_pos:
                last_checkbox_pos = pos + len(opt.label)

        remaining_text = block[last_checkbox_pos:].strip()
        remaining_text = re.sub(r"\[.*?\]", "", remaining_text).strip()
        remaining_text = re.sub(CDP_PATTERNS["question_id"], "", remaining_text).strip()

        if len(remaining_text) > 100:
            return remaining_text

        return None

    def _extract_textarea_title_and_response(
        self, block: str, start_pos: int
    ) -> Tuple[str, Optional[str]]:
        """TEXTAREA 타입 질문의 제목과 응답 분리"""
        content = block[start_pos:].strip()

        # [Fixed row], [Add row] 등 마커 제거
        content_clean = re.sub(r"\[.*?row\]", "", content, flags=re.IGNORECASE).strip()

        # 1. 물음표(?)가 있으면 거기까지를 제목으로
        question_mark_pos = content_clean.find("?")
        if question_mark_pos > 0 and question_mark_pos < 300:
            title = content_clean[: question_mark_pos + 1].strip()
            title = re.sub(r"\s+", " ", title)  # 줄바꿈 → 공백
            response = content_clean[question_mark_pos + 1 :].strip()
            # Opportunities, Risks 같은 트레일러 헤더 제거
            for pattern in ["Opportunities", "Opportunity", "Risks", "Risk"]:
                if response.strip().endswith(pattern):
                    response = response.strip()[: -len(pattern)].strip()
            return title, response if len(response) > 50 else None

        # 2. 알려진 제목 패턴 매칭
        known_titles = [
            "Application of definition",
            "Further details of process",
            "Description of how interconnections are assessed",
            "Explain why you do not identify priority locations",
        ]

        for known_title in known_titles:
            if content_clean.startswith(known_title):
                title = known_title
                response = content_clean[len(known_title) :].strip()
                return title, response if len(response) > 50 else None

        # 3. 줄 단위로 분리
        lines = [line.strip() for line in content_clean.split("\n") if line.strip()]

        if not lines:
            return "", None

        # 첫 줄이 짧으면 (100자 미만) 제목으로 간주
        first_line = lines[0]

        if len(first_line) < 100:
            title = first_line
            if len(lines) > 1:
                response = " ".join(lines[1:])
                for pattern in ["Opportunities", "Opportunity", "Risks", "Risk"]:
                    if response.strip().endswith(pattern):
                        response = response.strip()[: -len(pattern)].strip()
                return title, response if len(response) > 50 else None
            return title, None

        # 4. 첫 줄이 길면 첫 문장까지를 제목으로
        sentence_end = -1
        for i, char in enumerate(first_line):
            if char in ".!" and i < 200:
                sentence_end = i
                break

        if sentence_end > 0:
            title = first_line[: sentence_end + 1].strip()
            response = first_line[sentence_end + 1 :].strip()
            if len(lines) > 1:
                response += " " + " ".join(lines[1:])
            return title, response if len(response) > 50 else None

        # 5. Fallback: 전체를 제목으로
        return content_clean, None

    def _detect_row_type(self, block: str) -> RowType:
        """Row 타입 감지"""
        if re.search(CDP_PATTERNS["fixed_row"], block):
            return RowType.FIXED
        elif re.search(CDP_PATTERNS["add_row"], block):
            return RowType.ADDABLE
        return RowType.NONE

    def _set_parent_relationships(self):
        """질문 간 부모-자식 관계 설정"""
        for question in self.questions:
            parts = question.question_id.split(".")
            if len(parts) > 1:
                parent_id = ".".join(parts[:-1])
                parent_exists = any(q.question_id == parent_id for q in self.questions)
                if parent_exists:
                    question.parent_id = parent_id
