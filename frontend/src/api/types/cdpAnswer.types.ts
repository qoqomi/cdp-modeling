/**
 * CDP Answer Generator API Types
 * Backend cdp_structured_answers.json 형식과 Frontend 형식 정의
 */

// Backend에서 오는 원본 응답 형식
export interface CDPBackendSource {
  chunk_id: string;
  page_num: number;
  score: number;
  rerank_score?: number | null;
  section?: string | null;
  preview?: string;
}

// Backend row columns - textarea는 _en/_ko 분리
export interface CDPBackendRowColumns {
  [key: string]: string | number | boolean | null;
}

export interface CDPBackendRow {
  row_index: number;
  columns: CDPBackendRowColumns;
  confidence: number;
}

export interface CDPBackendAnswer {
  question_id: string;
  question_title: string;
  response_type: "table" | "single" | "text";
  rows: CDPBackendRow[];
  overall_confidence: number;
  rationale_en: string;
  rationale_ko: string;
  sources: CDPBackendSource[];
  // Legacy fields (for backward compatibility)
  response?: Record<string, any>;
  confidence?: number;
  validation_errors?: string[];
  raw_llm_response?: string | null;
}

export interface CDPBackendResponse {
  total_questions: number;
  average_confidence: number;
  answers: CDPBackendAnswer[];
  // Legacy summary format
  summary?: {
    total_questions: number;
    valid_responses: number;
    invalid_responses: number;
    validation_rate: number;
    average_confidence: number;
  };
}

// Frontend에서 사용하는 변환된 형식
export interface AnswerRow {
  number: string;
  detail: string;         // 필드명 (한글)
  detailEn?: string;      // 필드명 (영문)
  answer: string;         // 값
  answerKo?: string;      // 값 (한글)
  answerEn?: string;      // 값 (영문)
  type?: string;          // 필드 타입
  options?: string[];     // 선택 옵션
}

export interface AnswerContent {
  rows?: AnswerRow[];
  detailedText?: string;
  detailedTextKo?: string;
  detailedTextEn?: string;
  // 답변 레벨 rationale (한/영)
  rationaleEn?: string;
  rationaleKo?: string;
  // 인사이트 (신뢰도 기반)
  insight?: string;
  insightEn?: string;
  // 분석 텍스트
  analysis?: string;
  analysisKo?: string;
  analysisEn?: string;
  // 소스 근거
  evidence?: Array<{ source: string; excerpt: string; page?: number }>;
}

export interface GeneratedAnswer {
  id: string;
  questionId?: string;  // optional for backward compatibility
  content: AnswerContent;
  confidence?: number;
  sources?: CDPBackendSource[];
  validationErrors?: string[];
  // 테이블 타입 질문의 원본 행 데이터 (적용 시 사용)
  backendRows?: CDPBackendRow[];
  responseType?: "table" | "single" | "text";
}

// 스키마 정보 (원본 스키마 표시용)
export interface CDPSchemaColumn {
  id: string;
  name: string;
  name_ko?: string;
  type: "select" | "multiselect" | "textarea" | "number" | "percentage" | "grouped_select" | "text";
  options?: string[];
  grouped_options?: Record<string, string[]>;
  max_length?: number;
  required?: boolean;
  condition?: Record<string, any>;
}

export interface CDPQuestionSchema {
  question_id: string;
  title: string;
  title_ko?: string;
  columns: CDPSchemaColumn[];
  response_type?: "table" | "single" | "text" | "select" | "multiselect";
  allow_multiple_rows?: boolean;
  row_labels?: string[];
}

// API 요청/응답
export interface GenerateAnswerRequest {
  question_id: string;
  feedback?: string;
}

export interface GenerateAnswerResponse {
  answer: GeneratedAnswer;
  schema: CDPQuestionSchema;
}
