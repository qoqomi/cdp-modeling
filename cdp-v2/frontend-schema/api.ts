/**
 * CDP Frontend Schema - API Types
 * Backend API 요청/응답 타입 정의
 */

import type {
  CDPDocument,
  CDPSection,
  CDPQuestion,
  QuestionResponse,
  ResponseValue,
  SubmissionStatus,
  DocumentProgress,
} from './types';
import type { ValidationResult } from './schema';


// ============================================================
// 공통 API 응답 구조
// ============================================================

/** API 성공 응답 */
export interface ApiResponse<T> {
  success: true;
  data: T;
  message?: string;
  timestamp: string;
}

/** API 에러 응답 */
export interface ApiError {
  success: false;
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
  timestamp: string;
}

/** API 응답 통합 타입 */
export type ApiResult<T> = ApiResponse<T> | ApiError;

/** 페이지네이션 파라미터 */
export interface PaginationParams {
  page: number;
  limit: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

/** 페이지네이션 응답 */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  totalPages: number;
}


// ============================================================
// 문서 관련 API
// ============================================================

// --- GET /api/documents ---
export interface GetDocumentsParams extends PaginationParams {
  status?: SubmissionStatus;
  year?: number;
  search?: string;
}

export interface DocumentListItem {
  documentId: string;
  organizationName: string;
  reportingYear: number;
  submissionStatus: SubmissionStatus;
  lastModified: string;
  progress: number;  // 0-100
}

export type GetDocumentsResponse = PaginatedResponse<DocumentListItem>;


// --- GET /api/documents/:id ---
export type GetDocumentResponse = CDPDocument;


// --- POST /api/documents ---
export interface CreateDocumentRequest {
  organizationName: string;
  reportingYear: number;
  templateId?: string;  // 템플릿 기반 생성
}

export interface CreateDocumentResponse {
  documentId: string;
  createdAt: string;
}


// --- PUT /api/documents/:id ---
export interface UpdateDocumentRequest {
  metadata?: Partial<{
    organizationName: string;
    submissionStatus: SubmissionStatus;
  }>;
}

export interface UpdateDocumentResponse {
  documentId: string;
  lastModified: string;
}


// --- DELETE /api/documents/:id ---
export interface DeleteDocumentResponse {
  documentId: string;
  deletedAt: string;
}


// ============================================================
// 질문 응답 관련 API
// ============================================================

// --- GET /api/documents/:docId/questions/:questionId ---
export interface GetQuestionResponse {
  question: CDPQuestion;
  breadcrumb: BreadcrumbItem[];
}

export interface BreadcrumbItem {
  questionId: string;
  title: string;
  level: number;
}


// --- PUT /api/documents/:docId/questions/:questionId/response ---
export interface UpdateResponseRequest {
  value?: ResponseValue;
  rows?: Array<{
    rowId?: string;  // 기존 행 업데이트 시
    fields: Record<string, unknown>;
  }>;
}

export interface UpdateResponseResponse {
  questionId: string;
  response: QuestionResponse;
  progress: DocumentProgress;
  validation: ValidationResult;
}


// --- POST /api/documents/:docId/questions/:questionId/table/rows ---
export interface AddTableRowRequest {
  fields: Record<string, unknown>;
  insertAfter?: string;  // rowId
}

export interface AddTableRowResponse {
  rowId: string;
  rowIndex: number;
}


// --- DELETE /api/documents/:docId/questions/:questionId/table/rows/:rowId ---
export interface DeleteTableRowResponse {
  rowId: string;
  deletedAt: string;
}


// ============================================================
// 섹션 관련 API
// ============================================================

// --- GET /api/documents/:docId/sections/:sectionId ---
export interface GetSectionResponse {
  section: CDPSection;
  progress: {
    total: number;
    completed: number;
    percentage: number;
  };
}


// --- GET /api/documents/:docId/sections/:sectionId/validate ---
export interface ValidateSectionResponse {
  sectionId: string;
  validation: ValidationResult;
}


// ============================================================
// 파일 업로드 API
// ============================================================

// --- POST /api/upload ---
export interface UploadFileRequest {
  file: File;
  documentId?: string;
  questionId?: string;
}

export interface UploadFileResponse {
  fileId: string;
  fileName: string;
  fileSize: number;
  mimeType: string;
  url: string;
  uploadedAt: string;
}


// --- POST /api/documents/import ---
export interface ImportDocumentRequest {
  file: File;  // PDF or Excel
  organizationName?: string;
  reportingYear?: number;
}

export interface ImportDocumentResponse {
  documentId: string;
  parsedQuestions: number;
  warnings: string[];
}


// ============================================================
// AI 지원 API
// ============================================================

// --- POST /api/ai/suggest ---
export interface AISuggestRequest {
  documentId: string;
  questionId: string;
  context?: {
    previousAnswers?: Record<string, ResponseValue>;
    organizationInfo?: Record<string, unknown>;
  };
}

export interface AISuggestResponse {
  suggestion: string;
  confidence: number;  // 0-1
  reasoning?: string;
  sources?: string[];
}


// --- POST /api/ai/review ---
export interface AIReviewRequest {
  documentId: string;
  sectionId?: string;  // 특정 섹션만 리뷰
}

export interface AIReviewResponse {
  issues: AIReviewIssue[];
  suggestions: AIReviewSuggestion[];
  overallScore: number;  // 0-100
}

export interface AIReviewIssue {
  questionId: string;
  severity: 'low' | 'medium' | 'high';
  message: string;
  recommendation?: string;
}

export interface AIReviewSuggestion {
  questionId: string;
  currentValue: ResponseValue;
  suggestedValue: ResponseValue;
  reason: string;
}


// ============================================================
// 제출 관련 API
// ============================================================

// --- POST /api/documents/:id/submit ---
export interface SubmitDocumentRequest {
  confirmations: {
    dataAccuracy: boolean;
    termsAccepted: boolean;
  };
}

export interface SubmitDocumentResponse {
  documentId: string;
  submissionId: string;
  submittedAt: string;
  status: SubmissionStatus;
}


// --- GET /api/documents/:id/export ---
export interface ExportDocumentParams {
  format: 'pdf' | 'excel' | 'json';
}

export interface ExportDocumentResponse {
  downloadUrl: string;
  expiresAt: string;
}


// ============================================================
// 진행률 API
// ============================================================

// --- GET /api/documents/:id/progress ---
export type GetProgressResponse = DocumentProgress;


// ============================================================
// 인증 관련 API (참고용)
// ============================================================

export interface User {
  userId: string;
  email: string;
  name: string;
  organization: string;
  role: 'admin' | 'editor' | 'viewer';
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

// --- POST /api/auth/login ---
export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  user: User;
  tokens: AuthTokens;
}
