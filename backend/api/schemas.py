"""
API Schemas (Pydantic Models)
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, bool]


class SourceInfo(BaseModel):
    chunk_id: str = ""
    page_num: int = 0
    score: float = 0.0
    rerank_score: Optional[float] = None
    preview: Optional[str] = None
    section: Optional[str] = None


class RowData(BaseModel):
    row_index: int
    columns: Dict[str, Any]
    confidence: float = 0.0


class StructuredAnswer(BaseModel):
    question_id: str
    question_title: str = ""
    response_type: str = "table"
    rows: List[RowData] = []
    overall_confidence: float = 0.0
    sources: List[SourceInfo] = []
    rationale_en: str = ""
    rationale_ko: str = ""
    validation_errors: List[str] = []


class GenerateAnswerRequest(BaseModel):
    question_id: str
    feedback: Optional[str] = None


class GenerateAnswerResponse(BaseModel):
    success: bool
    answer: Optional[StructuredAnswer] = None
    error: Optional[str] = None
    processing_time_ms: Optional[int] = None


class GenerateBatchRequest(BaseModel):
    question_ids: List[str]


class GenerateBatchResponse(BaseModel):
    success: bool
    total: int
    completed: int
    failed: int
    answers: List[StructuredAnswer] = []
    errors: Dict[str, str] = {}
    processing_time_ms: Optional[int] = None


class ColumnSchema(BaseModel):
    id: str
    header: str = ""
    type: str = "text"
    required: bool = False
    options: Optional[List[str]] = None
    max_length: Optional[int] = None


class QuestionSchema(BaseModel):
    question_id: str
    title: str = ""
    response_type: str = "table"
    columns: List[ColumnSchema] = []
    row_labels: Optional[List[str]] = None


class QuestionsResponse(BaseModel):
    success: bool
    total: Optional[int] = None
    questions: Optional[List[QuestionSchema]] = None
    error: Optional[str] = None
