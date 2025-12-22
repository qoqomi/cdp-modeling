"""
API Routes Package
"""

from .generate import router as generate_router
from .generate import questions_router, questionnaire_router, answers_router
from .upload import router as upload_router

__all__ = [
    "generate_router",
    "questions_router",
    "questionnaire_router",
    "answers_router",
    "upload_router",
]
