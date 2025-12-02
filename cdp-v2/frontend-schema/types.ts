/**
 * CDP Frontend Schema - Core Types
 * CDP 질문지 시스템의 기본 타입 정의
 */

// ============================================================
// 기본 열거형 (Enums)
// ============================================================

/** 입력 타입 */
export type InputType =
  | 'singleSelect'      // 단일 선택
  | 'multiSelect'       // 다중 선택
  | 'groupedMultiSelect' // 그룹화된 다중 선택
  | 'text'              // 단문 텍스트
  | 'textarea'          // 장문 텍스트 (서술식)
  | 'table'             // 테이블 형식
  | 'number'            // 숫자 입력
  | 'date'              // 날짜 입력
  | 'file';             // 파일 업로드

/** 응답 상태 */
export type ResponseStatus =
  | 'empty'             // 미응답
  | 'partial'           // 부분 응답
  | 'complete'          // 완료
  | 'invalid';          // 유효하지 않음

/** 행 타입 (테이블/서술식용) */
export type RowType =
  | 'none'              // 해당 없음
  | 'fixed'             // 고정 행
  | 'addable';          // 추가 가능한 행

/** 제출 상태 */
export type SubmissionStatus =
  | 'draft'             // 초안
  | 'in_progress'       // 작성 중
  | 'submitted'         // 제출됨
  | 'approved'          // 승인됨
  | 'rejected';         // 반려됨


// ============================================================
// 옵션 관련 타입
// ============================================================

/** 기본 옵션 */
export interface Option {
  value: string;
  label: string;
  disabled?: boolean;
  description?: string;
}

/** 그룹화된 옵션 */
export interface GroupedOption {
  groupId: string;
  groupLabel: string;
  options: Option[];
}

/** 테이블 컬럼 정의 */
export interface TableColumn {
  columnId: string;
  header: string;
  inputType: InputTypeDefinition;
  required?: boolean;
  width?: string;
  description?: string;
}


// ============================================================
// 입력 타입 정의
// ============================================================

/** 단일/다중 선택 입력 타입 */
export interface SelectInputType {
  type: 'singleSelect' | 'multiSelect';
  options: Option[];
  placeholder?: string;
}

/** 그룹화된 다중 선택 입력 타입 */
export interface GroupedSelectInputType {
  type: 'groupedMultiSelect';
  groupedOptions: GroupedOption[];
  placeholder?: string;
}

/** 텍스트 입력 타입 */
export interface TextInputType {
  type: 'text' | 'textarea';
  maxLength?: number;
  minLength?: number;
  placeholder?: string;
  rows?: number;  // textarea용
}

/** 숫자 입력 타입 */
export interface NumberInputType {
  type: 'number';
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  placeholder?: string;
}

/** 테이블 입력 타입 */
export interface TableInputType {
  type: 'table';
  tableColumns: TableColumn[];
  minRows?: number;
  maxRows?: number;
}

/** 날짜 입력 타입 */
export interface DateInputType {
  type: 'date';
  minDate?: string;
  maxDate?: string;
  format?: string;
}

/** 파일 입력 타입 */
export interface FileInputType {
  type: 'file';
  accept?: string[];
  maxSize?: number;  // bytes
  multiple?: boolean;
}

/** 입력 타입 통합 */
export type InputTypeDefinition =
  | SelectInputType
  | GroupedSelectInputType
  | TextInputType
  | NumberInputType
  | TableInputType
  | DateInputType
  | FileInputType;


// ============================================================
// 응답 관련 타입
// ============================================================

/** 기본 응답 값 */
export type ResponseValue =
  | string                    // 단일 선택, 텍스트
  | string[]                  // 다중 선택
  | number                    // 숫자
  | TableRowResponse[];       // 테이블

/** 테이블 행 응답 */
export interface TableRowResponse {
  rowId: string;
  rowIndex: number;
  fields: Record<string, FieldResponse>;
}

/** 필드 응답 */
export interface FieldResponse {
  value: string | string[] | number | null;
  status: ResponseStatus;
}

/** 질문 응답 */
export interface QuestionResponse {
  value?: ResponseValue;
  rows?: TableRowResponse[];  // 테이블용
  status: ResponseStatus;
  lastModified?: string;
  validationErrors?: string[];
}


// ============================================================
// 질문 관련 타입
// ============================================================

/** CDP 질문 */
export interface CDPQuestion {
  questionId: string;
  parentId: string | null;
  title: string;
  description?: string;
  inputType: InputTypeDefinition;
  required: boolean;
  rowType: RowType;
  response: QuestionResponse;
  subQuestions?: CDPQuestion[];

  // UI 관련 메타데이터
  helpText?: string;
  exampleAnswer?: string;
  guidance?: string;

  // 조건부 표시
  visibilityCondition?: VisibilityCondition;
}

/** 조건부 표시 규칙 */
export interface VisibilityCondition {
  dependsOn: string;          // 의존하는 질문 ID
  operator: 'equals' | 'notEquals' | 'contains' | 'isEmpty' | 'isNotEmpty';
  value?: string | string[];
}


// ============================================================
// 섹션 관련 타입
// ============================================================

/** CDP 섹션 */
export interface CDPSection {
  sectionId: string;
  title: string;
  description?: string;
  questions: CDPQuestion[];

  // 섹션 메타데이터
  order?: number;
  isRequired?: boolean;
  completionPercentage?: number;
}


// ============================================================
// 문서 관련 타입
// ============================================================

/** 문서 메타데이터 */
export interface DocumentMetadata {
  documentId: string;
  version: string;
  reportingYear: number;
  organizationName: string;
  submissionStatus: SubmissionStatus;
  lastModified: string;
  createdAt: string;
  createdBy: string;

  // 추가 메타데이터
  industry?: string;
  region?: string;
  templateVersion?: string;
}

/** CDP 문서 (전체 구조) */
export interface CDPDocument {
  metadata: DocumentMetadata;
  sections: CDPSection[];
}


// ============================================================
// 유틸리티 타입
// ============================================================

/** 질문 ID로 질문 찾기용 맵 */
export type QuestionMap = Map<string, CDPQuestion>;

/** 섹션별 완료율 */
export interface SectionProgress {
  sectionId: string;
  total: number;
  completed: number;
  percentage: number;
}

/** 전체 문서 진행률 */
export interface DocumentProgress {
  sections: SectionProgress[];
  totalQuestions: number;
  completedQuestions: number;
  overallPercentage: number;
}
