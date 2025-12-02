/**
 * CDP Frontend Schema - Form State Management
 * 폼 상태 관리를 위한 타입 및 유틸리티
 */

import type {
  CDPDocument,
  CDPQuestion,
  CDPSection,
  QuestionResponse,
  ResponseValue,
  ResponseStatus,
  QuestionMap,
  DocumentProgress,
} from './types';
import type { ValidationResult, ValidationError } from './schema';


// ============================================================
// 폼 상태 (Form State)
// ============================================================

/** 전체 폼 상태 */
export interface CDPFormState {
  // 문서 데이터
  document: CDPDocument | null;

  // UI 상태
  currentSectionId: string | null;
  currentQuestionId: string | null;
  expandedSections: Set<string>;

  // 편집 상태
  isDirty: boolean;
  dirtyQuestions: Set<string>;

  // 로딩/저장 상태
  isLoading: boolean;
  isSaving: boolean;
  lastSaved: string | null;

  // 검증 상태
  validationErrors: Map<string, ValidationError[]>;
  showValidationErrors: boolean;

  // 진행률
  progress: DocumentProgress | null;

  // 캐시
  questionMap: QuestionMap;
}

/** 폼 상태 초기값 */
export const initialFormState: CDPFormState = {
  document: null,
  currentSectionId: null,
  currentQuestionId: null,
  expandedSections: new Set(),
  isDirty: false,
  dirtyQuestions: new Set(),
  isLoading: false,
  isSaving: false,
  lastSaved: null,
  validationErrors: new Map(),
  showValidationErrors: false,
  progress: null,
  questionMap: new Map(),
};


// ============================================================
// 액션 타입 (Redux/Zustand 스타일)
// ============================================================

export type CDPFormAction =
  // 문서 로드
  | { type: 'LOAD_DOCUMENT_START' }
  | { type: 'LOAD_DOCUMENT_SUCCESS'; payload: CDPDocument }
  | { type: 'LOAD_DOCUMENT_ERROR'; payload: string }

  // 응답 업데이트
  | { type: 'UPDATE_RESPONSE'; payload: { questionId: string; value: ResponseValue } }
  | { type: 'UPDATE_TABLE_ROW'; payload: { questionId: string; rowId: string; fields: Record<string, unknown> } }
  | { type: 'ADD_TABLE_ROW'; payload: { questionId: string; rowId: string } }
  | { type: 'DELETE_TABLE_ROW'; payload: { questionId: string; rowId: string } }

  // 저장
  | { type: 'SAVE_START' }
  | { type: 'SAVE_SUCCESS'; payload: { timestamp: string } }
  | { type: 'SAVE_ERROR'; payload: string }

  // 검증
  | { type: 'SET_VALIDATION_ERRORS'; payload: Map<string, ValidationError[]> }
  | { type: 'CLEAR_VALIDATION_ERRORS'; payload?: string }  // questionId 또는 전체
  | { type: 'TOGGLE_VALIDATION_DISPLAY'; payload: boolean }

  // 네비게이션
  | { type: 'SET_CURRENT_SECTION'; payload: string }
  | { type: 'SET_CURRENT_QUESTION'; payload: string }
  | { type: 'TOGGLE_SECTION'; payload: string }

  // 진행률
  | { type: 'UPDATE_PROGRESS'; payload: DocumentProgress }

  // 리셋
  | { type: 'RESET_FORM' };


// ============================================================
// 개별 질문 상태 (단일 질문용)
// ============================================================

/** 개별 질문 폼 상태 */
export interface QuestionFormState {
  questionId: string;
  value: ResponseValue | undefined;
  status: ResponseStatus;
  isDirty: boolean;
  isTouched: boolean;
  errors: string[];
  isLoading: boolean;
}

/** 질문 폼 상태 초기값 생성 */
export function createQuestionFormState(question: CDPQuestion): QuestionFormState {
  return {
    questionId: question.questionId,
    value: question.response.value,
    status: question.response.status,
    isDirty: false,
    isTouched: false,
    errors: question.response.validationErrors || [],
    isLoading: false,
  };
}


// ============================================================
// 폼 컨텍스트 (React Context용)
// ============================================================

/** 폼 컨텍스트 값 */
export interface CDPFormContextValue {
  state: CDPFormState;
  dispatch: (action: CDPFormAction) => void;

  // 헬퍼 함수들
  getQuestion: (questionId: string) => CDPQuestion | undefined;
  getSection: (sectionId: string) => CDPSection | undefined;
  updateResponse: (questionId: string, value: ResponseValue) => void;
  saveDocument: () => Promise<void>;
  validateSection: (sectionId: string) => ValidationResult;
  validateAll: () => ValidationResult;
  navigateToQuestion: (questionId: string) => void;
}


// ============================================================
// 폼 이벤트 (컴포넌트 통신용)
// ============================================================

/** 폼 이벤트 타입 */
export type CDPFormEvent =
  | { type: 'value-change'; questionId: string; value: ResponseValue }
  | { type: 'blur'; questionId: string }
  | { type: 'focus'; questionId: string }
  | { type: 'submit'; sectionId?: string }
  | { type: 'validate'; questionId?: string }
  | { type: 'navigate'; target: 'next' | 'prev' | string };

/** 폼 이벤트 핸들러 */
export type CDPFormEventHandler = (event: CDPFormEvent) => void;


// ============================================================
// 유틸리티 함수
// ============================================================

/** 문서에서 질문 맵 생성 */
export function buildQuestionMap(document: CDPDocument): QuestionMap {
  const map: QuestionMap = new Map();

  const addQuestions = (questions: CDPQuestion[]) => {
    questions.forEach((question) => {
      map.set(question.questionId, question);
      if (question.subQuestions) {
        addQuestions(question.subQuestions);
      }
    });
  };

  document.sections.forEach((section) => {
    addQuestions(section.questions);
  });

  return map;
}

/** 질문 ID로 섹션 ID 추출 */
export function getSectionIdFromQuestionId(questionId: string): string {
  return questionId.split('.')[0];
}

/** 부모 질문 ID 추출 */
export function getParentQuestionId(questionId: string): string | null {
  const parts = questionId.split('.');
  if (parts.length <= 1) return null;
  return parts.slice(0, -1).join('.');
}

/** 질문 깊이 계산 */
export function getQuestionDepth(questionId: string): number {
  return questionId.split('.').length - 1;
}

/** 다음 질문 ID 찾기 */
export function getNextQuestionId(
  currentId: string,
  questionMap: QuestionMap
): string | null {
  const ids = Array.from(questionMap.keys()).sort((a, b) => {
    const aParts = a.split('.').map(Number);
    const bParts = b.split('.').map(Number);
    for (let i = 0; i < Math.max(aParts.length, bParts.length); i++) {
      if (aParts[i] !== bParts[i]) {
        return (aParts[i] || 0) - (bParts[i] || 0);
      }
    }
    return 0;
  });

  const currentIndex = ids.indexOf(currentId);
  if (currentIndex === -1 || currentIndex === ids.length - 1) return null;
  return ids[currentIndex + 1];
}

/** 이전 질문 ID 찾기 */
export function getPrevQuestionId(
  currentId: string,
  questionMap: QuestionMap
): string | null {
  const ids = Array.from(questionMap.keys()).sort((a, b) => {
    const aParts = a.split('.').map(Number);
    const bParts = b.split('.').map(Number);
    for (let i = 0; i < Math.max(aParts.length, bParts.length); i++) {
      if (aParts[i] !== bParts[i]) {
        return (aParts[i] || 0) - (bParts[i] || 0);
      }
    }
    return 0;
  });

  const currentIndex = ids.indexOf(currentId);
  if (currentIndex <= 0) return null;
  return ids[currentIndex - 1];
}

/** 응답값 비교 (dirty 체크용) */
export function isResponseEqual(a: ResponseValue | undefined, b: ResponseValue | undefined): boolean {
  if (a === b) return true;
  if (a === undefined || b === undefined) return false;

  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((val, idx) => {
      if (typeof val === 'object' && typeof b[idx] === 'object') {
        return JSON.stringify(val) === JSON.stringify(b[idx]);
      }
      return val === b[idx];
    });
  }

  return JSON.stringify(a) === JSON.stringify(b);
}


// ============================================================
// 자동 저장 설정
// ============================================================

export interface AutoSaveConfig {
  enabled: boolean;
  debounceMs: number;       // 디바운스 시간 (기본 2000ms)
  onSave: () => Promise<void>;
  onError?: (error: Error) => void;
}

export const defaultAutoSaveConfig: AutoSaveConfig = {
  enabled: true,
  debounceMs: 2000,
  onSave: async () => {},
  onError: console.error,
};


// ============================================================
// 키보드 네비게이션
// ============================================================

export interface KeyboardNavConfig {
  enabled: boolean;
  shortcuts: {
    nextQuestion: string;     // 기본: 'Tab'
    prevQuestion: string;     // 기본: 'Shift+Tab'
    nextSection: string;      // 기본: 'Ctrl+ArrowDown'
    prevSection: string;      // 기본: 'Ctrl+ArrowUp'
    save: string;             // 기본: 'Ctrl+S'
    submit: string;           // 기본: 'Ctrl+Enter'
  };
}

export const defaultKeyboardNavConfig: KeyboardNavConfig = {
  enabled: true,
  shortcuts: {
    nextQuestion: 'Tab',
    prevQuestion: 'Shift+Tab',
    nextSection: 'Ctrl+ArrowDown',
    prevSection: 'Ctrl+ArrowUp',
    save: 'Ctrl+S',
    submit: 'Ctrl+Enter',
  },
};
