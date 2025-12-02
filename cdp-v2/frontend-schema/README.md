# CDP Frontend Schema

CDP 질문지 시스템의 Frontend TypeScript 스키마 정의

## 파일 구조

```
frontend-schema/
├── index.ts      # 모든 모듈 내보내기
├── types.ts      # 기본 타입 정의
├── schema.ts     # 검증 규칙, 렌더링 힌트
├── api.ts        # API 요청/응답 타입
├── form.ts       # 폼 상태 관리
└── README.md
```

## 사용법

### 1. 기본 타입 사용

```typescript
import type { CDPDocument, CDPQuestion, CDPSection } from './frontend-schema';

// 문서 로드
const document: CDPDocument = await fetchDocument(documentId);

// 질문 접근
document.sections.forEach((section: CDPSection) => {
  section.questions.forEach((question: CDPQuestion) => {
    console.log(question.questionId, question.title);
  });
});
```

### 2. 검증

```typescript
import { validateDocument, validateSection, validators } from './frontend-schema';

// 전체 문서 검증
const result = validateDocument(document);
if (!result.isValid) {
  console.log('검증 오류:', result.errors);
}

// 섹션 검증
const sectionResult = validateSection(section);

// 개별 질문 검증
const questionResult = validators[question.inputType.type](question);
```

### 3. 진행률 계산

```typescript
import { calculateDocumentProgress, calculateSectionProgress } from './frontend-schema';

const progress = calculateDocumentProgress(document);
console.log(`전체 진행률: ${progress.overallPercentage}%`);

progress.sections.forEach((section) => {
  console.log(`${section.sectionId}: ${section.percentage}%`);
});
```

### 4. 렌더링 설정

```typescript
import { getRenderConfig, componentMapping } from './frontend-schema';

const config = getRenderConfig(question);
console.log(`컴포넌트: ${config.component}`);
console.log(`레이아웃: ${config.layout}`);
console.log(`Props:`, config.props);
```

### 5. API 호출

```typescript
import type {
  GetDocumentResponse,
  UpdateResponseRequest,
  ApiResult,
} from './frontend-schema';

// 문서 조회
const response: ApiResult<GetDocumentResponse> = await api.get(`/documents/${id}`);

// 응답 업데이트
const updateReq: UpdateResponseRequest = {
  value: 'selected_option',
};
await api.put(`/documents/${docId}/questions/${questionId}/response`, updateReq);
```

### 6. 폼 상태 관리 (React 예시)

```typescript
import {
  CDPFormState,
  CDPFormAction,
  buildQuestionMap,
  initialFormState,
} from './frontend-schema';

// Reducer
function formReducer(state: CDPFormState, action: CDPFormAction): CDPFormState {
  switch (action.type) {
    case 'LOAD_DOCUMENT_SUCCESS':
      return {
        ...state,
        document: action.payload,
        questionMap: buildQuestionMap(action.payload),
        isLoading: false,
      };
    case 'UPDATE_RESPONSE':
      // 응답 업데이트 로직
      return state;
    default:
      return state;
  }
}

// Context Provider
const CDPFormContext = createContext<CDPFormContextValue | null>(null);

function CDPFormProvider({ children }) {
  const [state, dispatch] = useReducer(formReducer, initialFormState);

  const value: CDPFormContextValue = {
    state,
    dispatch,
    getQuestion: (id) => state.questionMap.get(id),
    // ... 기타 헬퍼 함수
  };

  return (
    <CDPFormContext.Provider value={value}>
      {children}
    </CDPFormContext.Provider>
  );
}
```

## 타입 다이어그램

```
CDPDocument
├── metadata: DocumentMetadata
│   ├── documentId
│   ├── organizationName
│   ├── reportingYear
│   └── submissionStatus
│
└── sections: CDPSection[]
    ├── sectionId
    ├── title
    └── questions: CDPQuestion[]
        ├── questionId
        ├── title
        ├── inputType: InputTypeDefinition
        │   ├── singleSelect → Option[]
        │   ├── multiSelect → Option[]
        │   ├── textarea → { maxLength }
        │   └── table → TableColumn[]
        ├── response: QuestionResponse
        │   ├── value
        │   ├── status
        │   └── rows (테이블용)
        └── subQuestions: CDPQuestion[]
```

## 입력 타입별 컴포넌트 매핑

| InputType | Component | Layout |
|-----------|-----------|--------|
| singleSelect | RadioGroup / Select | full |
| multiSelect | CheckboxGroup | full |
| groupedMultiSelect | GroupedCheckboxGroup | full |
| text | TextField | half |
| textarea | TextArea | full |
| table | DataTable | full |
| number | NumberField | third |
| date | DatePicker | third |
| file | FileUpload | full |
