/**
 * CDP Frontend Schema - Schema Definitions
 * 질문 유형별 검증 규칙, 렌더링 힌트, 변환 함수
 */

import type {
  CDPQuestion,
  CDPSection,
  CDPDocument,
  InputType,
  ResponseStatus,
  QuestionResponse,
  TableRowResponse,
  Option,
  SectionProgress,
  DocumentProgress,
} from './types';


// ============================================================
// 검증 규칙 (Validation Rules)
// ============================================================

/** 검증 결과 */
export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
}

/** 검증 오류 */
export interface ValidationError {
  questionId: string;
  field?: string;
  message: string;
  type: 'required' | 'format' | 'range' | 'length' | 'custom';
}

/** 입력 타입별 검증 함수 */
export const validators: Record<InputType, (question: CDPQuestion) => ValidationResult> = {
  singleSelect: (q) => validateSingleSelect(q),
  multiSelect: (q) => validateMultiSelect(q),
  groupedMultiSelect: (q) => validateMultiSelect(q),
  text: (q) => validateText(q),
  textarea: (q) => validateTextarea(q),
  table: (q) => validateTable(q),
  number: (q) => validateNumber(q),
  date: (q) => validateDate(q),
  file: (q) => validateFile(q),
};

function validateSingleSelect(question: CDPQuestion): ValidationResult {
  const errors: ValidationError[] = [];
  const { response, required, questionId } = question;

  if (required && (!response.value || response.status === 'empty')) {
    errors.push({
      questionId,
      message: '필수 항목입니다. 하나를 선택해주세요.',
      type: 'required',
    });
  }

  return { isValid: errors.length === 0, errors };
}

function validateMultiSelect(question: CDPQuestion): ValidationResult {
  const errors: ValidationError[] = [];
  const { response, required, questionId } = question;

  if (required) {
    const value = response.value as string[] | undefined;
    if (!value || value.length === 0) {
      errors.push({
        questionId,
        message: '필수 항목입니다. 최소 하나를 선택해주세요.',
        type: 'required',
      });
    }
  }

  return { isValid: errors.length === 0, errors };
}

function validateText(question: CDPQuestion): ValidationResult {
  const errors: ValidationError[] = [];
  const { response, required, questionId, inputType } = question;

  if (required && (!response.value || response.status === 'empty')) {
    errors.push({
      questionId,
      message: '필수 항목입니다.',
      type: 'required',
    });
  }

  if (inputType.type === 'text' || inputType.type === 'textarea') {
    const value = response.value as string;
    if (value) {
      if (inputType.maxLength && value.length > inputType.maxLength) {
        errors.push({
          questionId,
          message: `최대 ${inputType.maxLength}자까지 입력 가능합니다.`,
          type: 'length',
        });
      }
      if (inputType.minLength && value.length < inputType.minLength) {
        errors.push({
          questionId,
          message: `최소 ${inputType.minLength}자 이상 입력해주세요.`,
          type: 'length',
        });
      }
    }
  }

  return { isValid: errors.length === 0, errors };
}

function validateTextarea(question: CDPQuestion): ValidationResult {
  return validateText(question);
}

function validateTable(question: CDPQuestion): ValidationResult {
  const errors: ValidationError[] = [];
  const { response, required, questionId, inputType } = question;

  if (inputType.type !== 'table') {
    return { isValid: true, errors: [] };
  }

  const rows = response.rows || [];

  if (required && rows.length === 0) {
    errors.push({
      questionId,
      message: '필수 항목입니다. 최소 한 행을 입력해주세요.',
      type: 'required',
    });
  }

  // 각 행의 필수 필드 검증
  rows.forEach((row, index) => {
    inputType.tableColumns.forEach((col) => {
      if (col.required) {
        const field = row.fields[col.columnId];
        if (!field || field.value === null || field.value === '') {
          errors.push({
            questionId,
            field: `row${index + 1}.${col.columnId}`,
            message: `${col.header}는 필수 항목입니다.`,
            type: 'required',
          });
        }
      }
    });
  });

  return { isValid: errors.length === 0, errors };
}

function validateNumber(question: CDPQuestion): ValidationResult {
  const errors: ValidationError[] = [];
  const { response, required, questionId, inputType } = question;

  if (required && (response.value === undefined || response.value === null)) {
    errors.push({
      questionId,
      message: '필수 항목입니다.',
      type: 'required',
    });
  }

  if (inputType.type === 'number' && response.value !== undefined) {
    const value = response.value as number;
    if (inputType.min !== undefined && value < inputType.min) {
      errors.push({
        questionId,
        message: `최소값은 ${inputType.min}입니다.`,
        type: 'range',
      });
    }
    if (inputType.max !== undefined && value > inputType.max) {
      errors.push({
        questionId,
        message: `최대값은 ${inputType.max}입니다.`,
        type: 'range',
      });
    }
  }

  return { isValid: errors.length === 0, errors };
}

function validateDate(question: CDPQuestion): ValidationResult {
  const errors: ValidationError[] = [];
  const { response, required, questionId } = question;

  if (required && !response.value) {
    errors.push({
      questionId,
      message: '필수 항목입니다.',
      type: 'required',
    });
  }

  return { isValid: errors.length === 0, errors };
}

function validateFile(question: CDPQuestion): ValidationResult {
  const errors: ValidationError[] = [];
  const { response, required, questionId } = question;

  if (required && !response.value) {
    errors.push({
      questionId,
      message: '파일을 업로드해주세요.',
      type: 'required',
    });
  }

  return { isValid: errors.length === 0, errors };
}


// ============================================================
// 전체 문서 검증
// ============================================================

export function validateDocument(document: CDPDocument): ValidationResult {
  const allErrors: ValidationError[] = [];

  document.sections.forEach((section) => {
    validateSection(section).errors.forEach((error) => {
      allErrors.push(error);
    });
  });

  return {
    isValid: allErrors.length === 0,
    errors: allErrors,
  };
}

export function validateSection(section: CDPSection): ValidationResult {
  const allErrors: ValidationError[] = [];

  const validateQuestions = (questions: CDPQuestion[]) => {
    questions.forEach((question) => {
      const validator = validators[question.inputType.type];
      if (validator) {
        const result = validator(question);
        allErrors.push(...result.errors);
      }

      if (question.subQuestions) {
        validateQuestions(question.subQuestions);
      }
    });
  };

  validateQuestions(section.questions);

  return {
    isValid: allErrors.length === 0,
    errors: allErrors,
  };
}


// ============================================================
// 진행률 계산 (Progress Calculation)
// ============================================================

export function calculateDocumentProgress(document: CDPDocument): DocumentProgress {
  const sectionProgresses: SectionProgress[] = document.sections.map((section) =>
    calculateSectionProgress(section)
  );

  const totalQuestions = sectionProgresses.reduce((sum, s) => sum + s.total, 0);
  const completedQuestions = sectionProgresses.reduce((sum, s) => sum + s.completed, 0);

  return {
    sections: sectionProgresses,
    totalQuestions,
    completedQuestions,
    overallPercentage: totalQuestions > 0
      ? Math.round((completedQuestions / totalQuestions) * 100)
      : 0,
  };
}

export function calculateSectionProgress(section: CDPSection): SectionProgress {
  let total = 0;
  let completed = 0;

  const countQuestions = (questions: CDPQuestion[]) => {
    questions.forEach((question) => {
      // 필수 질문만 카운트 (선택 사항은 제외 가능)
      if (question.required) {
        total++;
        if (question.response.status === 'complete') {
          completed++;
        }
      }

      if (question.subQuestions) {
        countQuestions(question.subQuestions);
      }
    });
  };

  countQuestions(section.questions);

  return {
    sectionId: section.sectionId,
    total,
    completed,
    percentage: total > 0 ? Math.round((completed / total) * 100) : 0,
  };
}


// ============================================================
// 응답 상태 판단 (Response Status)
// ============================================================

export function determineResponseStatus(
  question: CDPQuestion,
  value: unknown
): ResponseStatus {
  const { inputType, required } = question;

  // 값이 없는 경우
  if (value === null || value === undefined || value === '') {
    return 'empty';
  }

  // 배열인 경우 (multiSelect, table)
  if (Array.isArray(value)) {
    if (value.length === 0) return 'empty';

    // 테이블인 경우 모든 필수 필드 확인
    if (inputType.type === 'table') {
      const rows = value as TableRowResponse[];
      const hasIncomplete = rows.some((row) =>
        inputType.tableColumns.some((col) => {
          if (!col.required) return false;
          const field = row.fields[col.columnId];
          return !field || field.value === null || field.value === '';
        })
      );
      return hasIncomplete ? 'partial' : 'complete';
    }

    return 'complete';
  }

  // 문자열/숫자인 경우
  return 'complete';
}


// ============================================================
// 렌더링 힌트 (Rendering Hints)
// ============================================================

/** 질문 유형별 UI 컴포넌트 매핑 */
export const componentMapping: Record<InputType, string> = {
  singleSelect: 'RadioGroup',       // 또는 'Select'
  multiSelect: 'CheckboxGroup',
  groupedMultiSelect: 'GroupedCheckboxGroup',
  text: 'TextField',
  textarea: 'TextArea',
  table: 'DataTable',
  number: 'NumberField',
  date: 'DatePicker',
  file: 'FileUpload',
};

/** 질문별 렌더링 설정 */
export interface RenderConfig {
  component: string;
  props: Record<string, unknown>;
  layout: 'full' | 'half' | 'third';
}

export function getRenderConfig(question: CDPQuestion): RenderConfig {
  const { inputType } = question;
  const component = componentMapping[inputType.type];

  const baseProps: Record<string, unknown> = {
    questionId: question.questionId,
    label: question.title,
    required: question.required,
    helpText: question.helpText,
    disabled: false,
  };

  switch (inputType.type) {
    case 'singleSelect':
    case 'multiSelect':
      return {
        component,
        props: {
          ...baseProps,
          options: inputType.options,
          placeholder: inputType.placeholder,
        },
        layout: 'full',
      };

    case 'groupedMultiSelect':
      return {
        component,
        props: {
          ...baseProps,
          groupedOptions: inputType.groupedOptions,
        },
        layout: 'full',
      };

    case 'textarea':
      return {
        component,
        props: {
          ...baseProps,
          maxLength: inputType.maxLength,
          rows: inputType.rows || 5,
          placeholder: inputType.placeholder,
        },
        layout: 'full',
      };

    case 'text':
      return {
        component,
        props: {
          ...baseProps,
          maxLength: inputType.maxLength,
          placeholder: inputType.placeholder,
        },
        layout: 'half',
      };

    case 'number':
      return {
        component,
        props: {
          ...baseProps,
          min: inputType.min,
          max: inputType.max,
          step: inputType.step,
          unit: inputType.unit,
        },
        layout: 'third',
      };

    case 'table':
      return {
        component,
        props: {
          ...baseProps,
          columns: inputType.tableColumns,
          minRows: inputType.minRows,
          maxRows: inputType.maxRows,
        },
        layout: 'full',
      };

    case 'date':
      return {
        component,
        props: {
          ...baseProps,
          minDate: inputType.minDate,
          maxDate: inputType.maxDate,
        },
        layout: 'third',
      };

    case 'file':
      return {
        component,
        props: {
          ...baseProps,
          accept: inputType.accept,
          maxSize: inputType.maxSize,
          multiple: inputType.multiple,
        },
        layout: 'full',
      };

    default:
      return {
        component: 'TextField',
        props: baseProps,
        layout: 'full',
      };
  }
}
