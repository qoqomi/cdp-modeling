"""
Generation Layer - CDP 답변 생성

핵심 원칙:
- 과거 답변은 참고용 (정답처럼 사용 금지)
- Prompt에서 연도 통제 명시
- 현재 지속가능경영보고서가 사실 기반
"""

from .prompt_builder import PromptBuilder
from .cdp_generator import CDPAnswerGenerator, GeneratedAnswer

__all__ = [
    "PromptBuilder",
    "CDPAnswerGenerator",
    "GeneratedAnswer",
]
