"""
한/영 번역 유틸리티
===================
영어 ↔ 한국어 번역 및 이중 언어 텍스트 생성

Usage:
    from utils.translator import Translator

    translator = Translator()

    # 영어 → 한국어
    ko_text = translator.to_korean("Hello world")

    # 한국어 → 영어
    en_text = translator.to_english("안녕하세요")

    # 이중 언어 생성 (영어 먼저 생성 후 한국어 번역)
    bilingual = translator.generate_bilingual(
        prompt="CDP 답변 근거를 작성하세요",
        context="...",
    )
    # Returns: {"en": "...", "ko": "..."}
"""

import os
import re
from typing import Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class Translator:
    """한/영 번역 유틸리티"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        self.client = OpenAI(api_key=api_key)

    def _detect_language(self, text: str) -> str:
        """텍스트 언어 감지 (간단한 휴리스틱)"""
        # 한글 문자 비율로 판단
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.findall(r'\w', text))

        if total_chars == 0:
            return "unknown"

        korean_ratio = korean_chars / total_chars
        return "ko" if korean_ratio > 0.3 else "en"

    def to_korean(self, text: str, context: Optional[str] = None) -> str:
        """영어 텍스트를 한국어로 번역

        Args:
            text: 번역할 영어 텍스트
            context: 번역 맥락 (선택)

        Returns:
            한국어 번역 텍스트
        """
        if not text or not text.strip():
            return ""

        # 이미 한국어면 그대로 반환
        if self._detect_language(text) == "ko":
            return text

        system_prompt = """당신은 전문 번역가입니다.
영어 텍스트를 자연스러운 한국어로 번역하세요.

규칙:
- 전문 용어는 적절히 번역하되, 필요시 영어 원문을 괄호 안에 병기
- CDP, ESG, TCFD 등 약어는 그대로 유지
- 페이지 번호 참조 (예: Page 28)는 그대로 유지
- 마크다운 포맷팅 (**bold**, - 리스트 등)은 그대로 유지
- 번역문만 반환 (설명이나 주석 없이)"""

        context_info = f"\n\n맥락: {context[:500]}" if context else ""

        user_prompt = f"""다음 텍스트를 한국어로 번역하세요:{context_info}

---
{text}
---

번역:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Translator] 번역 실패: {e}")
            return text  # 실패 시 원문 반환

    def to_english(self, text: str, context: Optional[str] = None) -> str:
        """한국어 텍스트를 영어로 번역

        Args:
            text: 번역할 한국어 텍스트
            context: 번역 맥락 (선택)

        Returns:
            영어 번역 텍스트
        """
        if not text or not text.strip():
            return ""

        # 이미 영어면 그대로 반환
        if self._detect_language(text) == "en":
            return text

        system_prompt = """You are a professional translator.
Translate Korean text to natural English.

Rules:
- Maintain technical accuracy
- Keep abbreviations like CDP, ESG, TCFD as-is
- Keep page references (e.g., Page 28) as-is
- Preserve markdown formatting (**bold**, - lists, etc.)
- Return only the translation (no explanations or notes)"""

        context_info = f"\n\nContext: {context[:500]}" if context else ""

        user_prompt = f"""Translate the following text to English:{context_info}

---
{text}
---

Translation:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Translator] Translation failed: {e}")
            return text  # 실패 시 원문 반환

    def generate_bilingual(
        self,
        prompt: str,
        context: str = "",
        system_prompt: Optional[str] = None,
        generate_in: str = "en",
        max_tokens: int = 1000,
    ) -> Dict[str, str]:
        """이중 언어 텍스트 생성

        영어 또는 한국어로 먼저 생성한 후, 반대 언어로 번역하여
        양쪽 언어 버전을 모두 반환

        Args:
            prompt: 생성 프롬프트
            context: 컨텍스트 정보
            system_prompt: 시스템 프롬프트 (None이면 기본값 사용)
            generate_in: 먼저 생성할 언어 ("en" 또는 "ko")
            max_tokens: 최대 토큰 수

        Returns:
            {"en": "영어 텍스트", "ko": "한국어 텍스트"}
        """
        if system_prompt is None:
            if generate_in == "en":
                system_prompt = """You are a CDP (Carbon Disclosure Project) expert.
Provide clear, concise, and accurate responses based on the given context.
- Always cite source page numbers (e.g., Page 28)
- Be specific with data and examples
- Use professional but accessible language"""
            else:
                system_prompt = """당신은 CDP (Carbon Disclosure Project) 전문가입니다.
주어진 컨텍스트를 기반으로 명확하고 간결하며 정확한 답변을 제공하세요.
- 항상 출처 페이지 번호를 인용하세요 (예: Page 28)
- 구체적인 데이터와 사례를 포함하세요
- 전문적이면서도 이해하기 쉬운 언어를 사용하세요"""

        user_prompt = f"""{prompt}

---
Context:
{context[:6000]}
---"""

        try:
            # Step 1: 원본 언어로 생성
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
            original_text = response.choices[0].message.content.strip()

            # Step 2: 반대 언어로 번역
            if generate_in == "en":
                return {
                    "en": original_text,
                    "ko": self.to_korean(original_text, context[:500])
                }
            else:
                return {
                    "en": self.to_english(original_text, context[:500]),
                    "ko": original_text
                }

        except Exception as e:
            print(f"[Translator] 이중 언어 생성 실패: {e}")
            return {"en": "", "ko": ""}

    def ensure_bilingual(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> Dict[str, str]:
        """단일 언어 텍스트를 이중 언어로 변환

        텍스트의 언어를 감지하고, 반대 언어 버전을 생성하여 둘 다 반환

        Args:
            text: 입력 텍스트 (영어 또는 한국어)
            context: 번역 맥락 (선택)

        Returns:
            {"en": "영어 텍스트", "ko": "한국어 텍스트"}
        """
        if not text or not text.strip():
            return {"en": "", "ko": ""}

        detected_lang = self._detect_language(text)

        if detected_lang == "ko":
            return {
                "en": self.to_english(text, context),
                "ko": text
            }
        else:
            return {
                "en": text,
                "ko": self.to_korean(text, context)
            }
