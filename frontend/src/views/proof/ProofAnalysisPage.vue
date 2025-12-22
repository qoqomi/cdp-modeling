<template>
  <div
    class="flex flex-col gap-4 h-full w-full overflow-hidden"
    style="min-height: 0"
  >
    <!-- Back Button -->
    <div class="flex-shrink-0 flex items-center justify-between">
      <button
        @click="handleBackToList"
        class="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all hover:opacity-90"
        :style="{
          backgroundColor: themeStore.isDark
            ? 'rgba(16, 185, 129, 0.1)'
            : 'rgba(16, 185, 129, 0.1)',
          color: themeStore.theme.primary,
        }"
      >
        <ArrowLeft :size="16" />
        뒤로 가기
      </button>

      <!-- AI Panel Toggle Button (visible on screens smaller than 2xl) -->
      <button
        v-if="!isAIPanelOpen"
        @click="isAIPanelOpen = true"
        class="2xl:hidden flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all hover:opacity-90"
        :style="{
          backgroundColor: themeStore.isDark
            ? 'rgba(16, 185, 129, 0.2)'
            : 'rgba(16, 185, 129, 0.15)',
          color: themeStore.theme.primary,
        }"
      >
        <Sparkles :size="16" />
        AI 답변 생성
      </button>
    </div>

    <!-- Main Content -->
    <div
      class="flex-1 flex gap-4 overflow-hidden relative"
      style="min-height: 0"
    >
      <!-- Left: Question Sidebar -->
      <QuestionSidebar
        :selected-question-id="selectedQuestionId"
        :question-structure="questionStructure"
        @question-select="handleQuestionSelect"
      />

      <!-- Center: Editable Document Viewer -->
      <div
        class="flex-1 rounded-xl border overflow-hidden flex flex-col"
        :style="{
          backgroundColor: themeStore.isDark
            ? 'rgba(17, 24, 39, 0.95)'
            : 'rgba(255, 255, 255, 0.95)',
          borderColor: themeStore.theme.cardBorder,
        }"
      >
        <EditableDocumentViewer
          :document-url="null"
          :selected-question-id="selectedQuestionId"
          :show-translation="showTranslation"
          :document-sections="documentSections"
          :applied-answers="appliedAnswers"
          :active-section="activeSection"
          :is-full-view="isFullView"
          @question-select="handleQuestionSelect"
          @save="handleSave"
          @apply-answer="handleApplyAnswer"
          @toggle-translation="showTranslation = !showTranslation"
          @full-view="handleFullView"
        />
      </div>

      <!-- Right: Sunhat AI Panel - Desktop (always visible on 2xl/1536px+) -->
      <div
        class="hidden 2xl:flex rounded-xl border overflow-hidden flex-col"
        style="width: 420px; min-width: 380px; flex-shrink: 0; min-height: 0"
        :style="{
          borderColor: themeStore.theme.cardBorder,
        }"
      >
        <SunhatAIPanel
          :selected-question="selectedQuestion"
          :generated-answers="generatedAnswers"
          :applied-answer-id="
            selectedQuestionId
              ? appliedAnswers[selectedQuestionId] || null
              : null
          "
          @generate-answer="handleGenerateAnswer"
          @select-answer="handleSelectAnswer"
          @close="handleCloseAI"
          @toggle-expand="handleToggleExpand"
        />
      </div>

      <!-- Right: Sunhat AI Panel - Slide overlay (for screens smaller than 2xl) -->
      <Transition name="slide-panel">
        <div
          v-if="isAIPanelOpen && !isDesktop"
          class="2xl:hidden absolute inset-0 z-50 flex"
        >
          <!-- Backdrop -->
          <div
            class="absolute inset-0 bg-black/40 backdrop-blur-sm"
            @click="isAIPanelOpen = false"
          />

          <!-- Panel - Slide from right -->
          <div
            class="absolute right-0 top-0 bottom-0 w-full max-w-md rounded-l-xl border-l overflow-hidden flex flex-col shadow-2xl"
            :style="{
              backgroundColor: themeStore.isDark
                ? 'rgba(26, 26, 26, 1)'
                : 'rgba(45, 45, 45, 1)',
              borderColor: themeStore.theme.cardBorder,
            }"
          >
            <SunhatAIPanel
              :selected-question="selectedQuestion"
              :generated-answers="generatedAnswers"
              :applied-answer-id="
                selectedQuestionId
                  ? appliedAnswers[selectedQuestionId] || null
                  : null
              "
              @generate-answer="handleGenerateAnswer"
              @select-answer="handleSelectAnswer"
              @close="isAIPanelOpen = false"
              @toggle-expand="handleToggleExpand"
            />
          </div>
        </div>
      </Transition>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch } from "vue";
import { useRouter, useRoute } from "vue-router";
import { ArrowLeft, Sparkles } from "lucide-vue-next";
import { useThemeStore } from "../../stores/themeStore";
import QuestionSidebar from "../../components/proof/QuestionSidebar.vue";
import EditableDocumentViewer from "../../components/proof/EditableDocumentViewer.vue";
import SunhatAIPanel from "../../components/proof/SunhatAIPanel.vue";
import { loadLocalAnswers, loadLocalSchema } from "../../api/cdpAnswerApi";
import { loadLocalQuestionnaireModule2 } from "../../api/cdpQuestionnaireApi";
import type {
  CDPQuestionSchema,
  GeneratedAnswer,
} from "../../api/types/cdpAnswer.types";
import type {
  CDPQuestionnaireModule,
  CDPQuestionnaireQuestion,
} from "../../api/types/cdpQuestionnaire.types";

const themeStore = useThemeStore();
const router = useRouter();
const route = useRoute();

// AI Panel state for mobile/tablet
const isAIPanelOpen = ref(false);

// Responsive check for desktop (2xl breakpoint = 1536px)
const isDesktop = ref(window.innerWidth >= 1536);
const updateIsDesktop = () => {
  isDesktop.value = window.innerWidth >= 1536;
  // Auto-close panel when switching to desktop
  if (isDesktop.value) {
    isAIPanelOpen.value = false;
  }
};

onMounted(() => {
  window.addEventListener("resize", updateIsDesktop);
});

onUnmounted(() => {
  window.removeEventListener("resize", updateIsDesktop);
});

// Get reportId from route params
const reportId = computed(() => route.params.reportId as string);
const STORAGE_KEY = computed(() => `cdp_djsi_saved_state_${reportId.value}`);

// State
const selectedQuestionId = ref<string | null>(null);
const showTranslation = ref(false);
const generatedAnswersByQuestion = ref<
  Record<string, Array<GeneratedAnswer>>
>({});

// Backend에서 로드된 답변 캐시
const backendAnswersCache = ref<Map<string, GeneratedAnswer>>(new Map());
const generatedAnswers = computed(() => {
  if (!selectedQuestionId.value) return [];
  const list = generatedAnswersByQuestion.value[selectedQuestionId.value] || [];
  return [...list];
});
const documentSections = ref<any[]>([]);
const appliedAnswers = ref<Record<string, string>>({});
const activeSection = ref<string | null>(null);
const isFullView = ref(false);

type UIQuestion = {
  id: string; // e.g., q-2.2.4
  number: string; // e.g., 2.2.4
  title: string; // ko (if available)
  titleEn: string;
  shortTitle: string;
  type: string;
  rationale?: string | null;
  ambition?: string[] | null;
  requestedContent?: string[] | null;
  conditionalLogic?: string | null;
  // 한/영 가이드라인 (backend merger.py --with-translation)
  guidance_raw?: {
    rationale?: string | null;
    rationale_ko?: string | null;
    ambition?: string[] | null;
    ambition_ko?: string[] | null;
    requested_content?: string[] | null;
    requested_content_ko?: string[] | null;
    additional_information?: string | null;
    additional_information_ko?: string | null;
  } | null;
};

const questionnaireModule = ref<CDPQuestionnaireModule | null>(null);
const schemaMap = ref<Map<string, CDPQuestionSchema>>(new Map());
const questions = ref<UIQuestion[]>([]);
const baseDocumentSections = ref<any[]>([]);

const flattenQuestions = (
  q: CDPQuestionnaireQuestion
): CDPQuestionnaireQuestion[] => {
  const out: CDPQuestionnaireQuestion[] = [];
  const visit = (node: CDPQuestionnaireQuestion) => {
    out.push(node);
    if (node.children?.length) {
      for (const child of node.children) visit(child);
    }
  };
  visit(q);
  return out;
};

const buildCellFromColumn = (questionId: string, col: any, colIndex: number) => {
  const colType = String(col.type || "").toLowerCase();
  const base = {
    schemaId: col.id,
    subId: `${questionId}.${colIndex + 1}`,
    condition: col.condition || null,
  };

  if (colType === "multiselect") {
    return {
      ...base,
      type: "checkbox",
      value: [],
      valueEn: [],
      options: col.options || [],
      optionsEn: col.options || [],
    };
  }

  if (colType === "grouped_select") {
    const grouped = col.grouped_options || {};
    const flattened: string[] = [];
    for (const [group, opts] of Object.entries(grouped)) {
      for (const opt of opts as string[]) {
        flattened.push(`${group} - ${opt}`);
      }
    }
    return {
      ...base,
      type: "select",
      value: "",
      valueEn: "",
      options: flattened,
      optionsEn: flattened,
    };
  }

  if (colType === "select") {
    return {
      ...base,
      type: "select",
      value: "",
      valueEn: "",
      options: col.options || [],
      optionsEn: col.options || [],
    };
  }

  return {
    ...base,
    type: "text",
    value: "",
    valueEn: "",
  };
};

const buildQuestionFromSchema = (
  questionId: string,
  extracted?: CDPQuestionnaireQuestion | null
) => {
  const schema = schemaMap.value.get(questionId);
  // cdp_questions_merged.json: title(영어), title_ko(한국어)
  const extractedTitleEn = extracted?.title_en || (extracted as any)?.title || "";
  const schemaTitleEn = schema?.title || extractedTitleEn || questionId;
  const schemaTitleKo = schema?.title_ko || extracted?.title_ko || null;

  const responseType =
    schema?.response_type || extracted?.response_format?.type || "textarea";

  const base: any = {
    id: `q-${questionId}`,
    number: questionId,
    title: schemaTitleKo || schemaTitleEn,
    titleEn: schemaTitleEn,
    type: responseType,
  };

  if (responseType === "table") {
    const columns = schema?.columns || [];
    const headers = columns.map((c: any) => c.header);
    const headersKo = columns.map((c: any) => c.header_ko || c.header);

    // row_labels가 있으면 해당 개수만큼 초기 행 생성
    const rowLabels = schema?.row_labels || (extracted as any)?.row_labels || [];
    const rowCount = rowLabels.length > 0 ? rowLabels.length : 1;

    const tableRows = [];
    for (let i = 0; i < rowCount; i++) {
      tableRows.push([
        ...columns.map((c: any, idx: number) => buildCellFromColumn(questionId, c, idx)),
      ]);
    }

    return {
      ...base,
      type: "table",
      tableHeaders: headers,
      tableHeadersEn: headers,
      tableHeadersKo: headersKo,
      tableRows,
      rowLabels,
    };
  }

  if (responseType === "select") {
    const col = schema?.columns?.[0];
    const options = col?.options || extracted?.response_format?.options || [];
    return {
      ...base,
      type: "select",
      value: "",
      valueEn: "",
      options,
      optionsEn: options,
    };
  }

  if (responseType === "multiselect") {
    const col = schema?.columns?.[0];
    const options = col?.options || extracted?.response_format?.options || [];
    return {
      ...base,
      type: "checkbox",
      value: [],
      valueEn: [],
      options,
      optionsEn: options,
    };
  }

  if (
    responseType === "number" ||
    responseType === "percentage" ||
    responseType === "text"
  ) {
    return { ...base, type: responseType, value: "", valueEn: "" };
  }

  return { ...base, type: "textarea", value: "", valueEn: "" };
};

const buildFromQuestionnaire = () => {
  const module = questionnaireModule.value;
  if (!module) return;

  const nextSections: any[] = [];
  const nextQuestionsFlat: UIQuestion[] = [];

  for (const root of module.questions) {
    const sectionQuestions = flattenQuestions(root);
    // cdp_questions_merged.json: title(영어), title_ko(한국어)
    // cdp_questions_parsed_module2.json: title_en(영어), title_ko(한국어)
    const rootTitleEn = (root as any).title_en || (root as any).title || "";
    const rootTitleKo = root.title_ko || rootTitleEn;
    const section = {
      number: root.question_id,
      title: rootTitleKo,
      titleEn: rootTitleEn,
      description: "",
      questions: [] as any[],
    };

    for (const q of sectionQuestions) {
      const uiQuestion = buildQuestionFromSchema(q.question_id, q);
      const withGuideline = {
        ...uiQuestion,
        rationale: q.rationale || null,
        ambition: q.ambition || null,
        requestedContent: q.requested_content || null,
        conditionalLogic: q.question_dependencies || null,
      };

      section.questions.push(withGuideline);
      nextQuestionsFlat.push({
        id: uiQuestion.id,
        number: q.question_id,
        title: uiQuestion.title,
        titleEn: uiQuestion.titleEn,
        shortTitle:
          (schemaMap.value.get(q.question_id)?.title || (q as any).title_en || (q as any).title || "").slice(
            0,
            40
          ) || q.question_id,
        type: uiQuestion.type,
        rationale: q.rationale || null,
        ambition: q.ambition || null,
        requestedContent: q.requested_content || null,
        conditionalLogic: q.question_dependencies || null,
        // 한/영 가이드라인 전달
        guidance_raw: q.guidance_raw || null,
      });
    }

    nextSections.push(section);
  }

  baseDocumentSections.value = nextSections;
  questions.value = nextQuestionsFlat;
};

const initQuestionnaire = async () => {
  schemaMap.value = await loadLocalSchema();
  questionnaireModule.value = await loadLocalQuestionnaireModule2();
  buildFromQuestionnaire();
};

const questionStructure = computed(() => {
  const qById = new Map(questions.value.map((q) => [q.id, q]));
  return (documentSections.value || []).map((sec: any) => ({
    number: sec.number,
    questions: (sec.questions || [])
      .map((q: any) => {
        const ui = qById.get(q.id);
        return {
          id: q.id,
          number: q.number,
          shortTitle: ui?.shortTitle || ui?.titleEn?.slice(0, 40) || q.number,
        };
      })
      .filter(Boolean),
  }));
});

const selectedQuestion = computed(() => {
  if (!selectedQuestionId.value) return null;
  return questions.value.find((q) => q.id === selectedQuestionId.value) || null;
});

// Navigation
const handleBackToList = () => {
  router.push({ name: "proof" });
};

// Question selection
const handleQuestionSelect = (questionId: string | null) => {
  selectedQuestionId.value = questionId;
  isFullView.value = false;

  if (questionId) {
    const section = documentSections.value.find((sec: any) =>
      sec.questions?.some((q: any) => q.id === questionId)
    );
    if (section) {
      activeSection.value = section.number || null;
    }
  }
};

const handleFullView = () => {
  selectedQuestionId.value = null;
  isFullView.value = true;
};

// Save state
const persistState = (withAlert = false) => {
  try {
    const stateToSave = {
      documentSections: documentSections.value,
      generatedAnswersByQuestion: generatedAnswersByQuestion.value,
      appliedAnswers: appliedAnswers.value,
    };
    localStorage.setItem(STORAGE_KEY.value, JSON.stringify(stateToSave));
    console.log("Saving changes...", stateToSave);
    if (withAlert) {
      alert("변경사항이 저장되었습니다.");
    }
  } catch (e) {
    console.error("Failed to save state", e);
    if (withAlert) {
      alert("저장 중 오류가 발생했습니다.");
    }
  }
};

const handleSave = () => {
  persistState(true);
};

// Merge saved values into fresh sections
const mergeSavedValuesIntoSections = (
  freshSections: any[],
  savedSections: any[]
) => {
  // Create a map of saved question values by id
  const savedQuestionMap = new Map<string, any>();
  for (const section of savedSections) {
    for (const q of section.questions || []) {
      savedQuestionMap.set(q.id, q);
    }
  }

  // Apply saved values to fresh sections
  for (const section of freshSections) {
    for (const q of section.questions || []) {
      const savedQ = savedQuestionMap.get(q.id);
      if (savedQ) {
        // Preserve user-entered values
        if (q.type === "table" && savedQ.tableRows && q.tableRows) {
          // Merge table rows - preserve cell values
          for (let rIdx = 0; rIdx < Math.min(q.tableRows.length, savedQ.tableRows.length); rIdx++) {
            for (let cIdx = 0; cIdx < Math.min(q.tableRows[rIdx].length, savedQ.tableRows[rIdx]?.length || 0); cIdx++) {
              const savedCell = savedQ.tableRows[rIdx][cIdx];
              if (savedCell?.value || savedCell?.valueEn) {
                q.tableRows[rIdx][cIdx].value = savedCell.value || "";
                q.tableRows[rIdx][cIdx].valueEn = savedCell.valueEn || "";
              }
            }
          }
        } else if (savedQ.value || savedQ.valueEn) {
          q.value = savedQ.value || "";
          q.valueEn = savedQ.valueEn || "";
        }
      }
    }
  }

  return freshSections;
};

// Load saved state
const loadSavedState = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY.value);
    const freshSections = baseDocumentSections.value.length > 0
      ? baseDocumentSections.value
      : generateMockDocumentSections();

    if (raw) {
      const parsed = JSON.parse(raw);
      // Always use fresh structure, but merge saved values
      documentSections.value = mergeSavedValuesIntoSections(
        JSON.parse(JSON.stringify(freshSections)), // Deep clone
        parsed.documentSections || []
      );
      generatedAnswersByQuestion.value =
        parsed.generatedAnswersByQuestion || {};
      appliedAnswers.value = parsed.appliedAnswers || {};
    } else {
      documentSections.value = freshSections;
      generatedAnswersByQuestion.value = {};
      appliedAnswers.value = {};
    }

    if (documentSections.value.length > 0) {
      activeSection.value = documentSections.value[0]?.number || null;
    }
  } catch (e) {
    console.error("Failed to load saved state", e);
    documentSections.value = baseDocumentSections.value || generateMockDocumentSections();
    generatedAnswersByQuestion.value = {};
    appliedAnswers.value = {};
    activeSection.value = documentSections.value[0]?.number || null;
  }
};

// Find question in sections
const findQuestionInSections = (questionId: string) => {
  for (const section of documentSections.value) {
    const question = section.questions.find((q: any) => q.id === questionId);
    if (question) return question;
  }
  return null;
};

// Generate answer - Backend 데이터 사용
const handleGenerateAnswer = async (questionId: string, feedback?: string) => {
  console.log("Generating answer for:", questionId, "feedback:", feedback);

  const question = findQuestionInSections(questionId);
  if (!question) {
    console.error("Question not found:", questionId);
    return;
  }

  // question ID에서 실제 번호 추출 (q-2.2 -> 2.2)
  const actualQuestionId = questionId.replace("q-", "");

  // 캐시에서 먼저 확인
  if (backendAnswersCache.value.has(actualQuestionId)) {
    const cachedAnswer = backendAnswersCache.value.get(actualQuestionId)!;
    generatedAnswersByQuestion.value = {
      ...generatedAnswersByQuestion.value,
      [questionId]: [cachedAnswer],
    };
    console.log("Using cached answer for:", actualQuestionId);
    return;
  }

  // Backend에서 로드 시도
  try {
    const allAnswers = await loadLocalAnswers();
    backendAnswersCache.value = allAnswers;

    if (allAnswers.has(actualQuestionId)) {
      const backendAnswer = allAnswers.get(actualQuestionId)!;
      generatedAnswersByQuestion.value = {
        ...generatedAnswersByQuestion.value,
        [questionId]: [backendAnswer],
      };
      console.log("Loaded answer from backend:", actualQuestionId, backendAnswer);
      return;
    }
  } catch (error) {
    console.error("Failed to load backend answers:", error);
  }

  // Backend에 없으면 기존 Mock 데이터 사용
  setTimeout(() => {
    const baseAnswers = generateAnswersForQuestion(
      questionId,
      question,
      feedback
    );
    generatedAnswersByQuestion.value = {
      ...generatedAnswersByQuestion.value,
      [questionId]: baseAnswers,
    };
  }, 1000);
};

// "보고서에서 찾을 수 없음" 메시지 감지 - 문서 뷰어에 적용하지 않음
const isNotFoundResponse = (text: string): boolean => {
  if (!text) return false;
  const textLower = text.toLowerCase();
  const notFoundPatterns = [
    "not found in the report",
    "was not found",
    "information was not found",
    "could not be found",
    "no information",
    "not available in",
    "not mentioned in",
    "보고서에서 발견되지 않",
    "찾을 수 없습니다",
    "정보가 없습니다",
    "언급되지 않",
  ];
  return notFoundPatterns.some((pattern) => textLower.includes(pattern));
};

// Apply answer
const handleApplyAnswer = (fullAnswer: GeneratedAnswer, answerId?: string) => {
  if (!selectedQuestionId.value) return;

  const question = findQuestionInSections(selectedQuestionId.value);
  if (!question) return;

  const content = fullAnswer.content;

  // Apply based on question type
  if (question.type === "textarea") {
    const valueKo = content.detailedTextKo || content.detailedText || "";
    const valueEn = content.detailedTextEn || content.detailedText || "";
    // "찾을 수 없음" 메시지는 적용하지 않음
    if (!isNotFoundResponse(valueKo) && !isNotFoundResponse(valueEn)) {
      question.value = valueKo;
      question.valueEn = valueEn;
    }
  } else if (question.type === "select" && content.rows?.length > 0) {
    const answerRow = content.rows[0];
    const valueKo = answerRow.answerKo || answerRow.answer || "";
    const valueEn = answerRow.answerEn || answerRow.answer || "";
    // "찾을 수 없음" 메시지는 적용하지 않음
    if (!isNotFoundResponse(valueKo) && !isNotFoundResponse(valueEn)) {
      question.value = valueKo;
      question.valueEn = valueEn;
    }
  } else if (question.type === "table" && fullAnswer.backendRows?.length) {
    // 테이블 타입: 원본 backend rows를 사용하여 각 셀에 값 적용
    const backendRows = fullAnswer.backendRows;

    for (const backendRow of backendRows) {
      const rowIdx = backendRow.row_index;
      const cols = backendRow.columns;

      // 필요시 행 추가
      while (question.tableRows.length <= rowIdx) {
        // 첫 번째 행 구조 복제
        const newRow = question.tableRows[0].map((cell: any) => ({
          ...cell,
          value: "",
          valueEn: "",
        }));
        question.tableRows.push(newRow);
      }

      // 각 셀에 값 적용
      for (const [colId, value] of Object.entries(cols)) {
        // _en/_ko suffix는 별도 처리
        if (colId.endsWith("_en") || colId.endsWith("_ko")) continue;

        const strValue = String(value ?? "");
        // "찾을 수 없음" 메시지는 적용하지 않음
        if (isNotFoundResponse(strValue)) continue;

        const cell = question.tableRows[rowIdx]?.find(
          (c: any) => c.schemaId === colId
        );
        if (cell) {
          cell.value = strValue;
          cell.valueEn = strValue;
        }
      }

      // textarea 컬럼 (_en/_ko) 처리
      for (const [colId, value] of Object.entries(cols)) {
        const strValue = String(value ?? "");
        // "찾을 수 없음" 메시지는 적용하지 않음
        if (isNotFoundResponse(strValue)) continue;

        if (colId.endsWith("_en")) {
          const baseColId = colId.replace(/_en$/, "");
          const cell = question.tableRows[rowIdx]?.find(
            (c: any) => c.schemaId === baseColId
          );
          if (cell) {
            cell.valueEn = strValue;
          }
        } else if (colId.endsWith("_ko")) {
          const baseColId = colId.replace(/_ko$/, "");
          const cell = question.tableRows[rowIdx]?.find(
            (c: any) => c.schemaId === baseColId
          );
          if (cell) {
            cell.value = strValue;
          }
        }
      }
    }

    // Vue 반응성 트리거 - 테이블 행 재할당
    question.tableRows = [...question.tableRows.map((row: any[]) =>
      row.map((cell: any) => ({ ...cell }))
    )];

    // documentSections 전체 재할당으로 반응성 보장
    documentSections.value = [...documentSections.value];

    console.log("Applied table answer:", question.tableRows);
  }

  if (answerId) {
    appliedAnswers.value = {
      ...appliedAnswers.value,
      [selectedQuestionId.value]: answerId,
    };
    persistState(false);
  }
};

const handleSelectAnswer = (answerId: string) => {
  const answer = generatedAnswers.value.find((a) => a.id === answerId);
  if (answer && selectedQuestionId.value) {
    handleApplyAnswer(answer, answerId);
    persistState(false);
    alert("선택한 답변이 문항에 적용되었습니다.");
  }
};

const handleCloseAI = () => {
  selectedQuestionId.value = null;
};

const handleToggleExpand = () => {
  // Handle expand/collapse if needed
};

// Generate mock document sections
const generateMockDocumentSections = () => {
  return [
    {
      number: "2.2",
      title: "환경 의존성 및 영향 관리 프로세스",
      titleEn: "Environmental Dependencies and Impacts Management Process",
      description: "",
      questions: [
        {
          id: "q-2.2",
          number: "2.2",
          title:
            "귀 조직은 환경 의존성 및/또는 영향을 식별, 평가 및 관리하는 프로세스를 보유하고 있습니까?",
          titleEn:
            "Does your organization have a process for identifying, assessing, and managing environmental dependencies and/or impacts?",
          type: "table",
          tableHeaders: [
            "프로세스 구축 여부",
            "이 프로세스에서 평가되는 의존성 및/또는 영향",
          ],
          tableHeadersEn: [
            "Process in place",
            "Dependencies and/or impacts evaluated in this process",
          ],
          tableRows: [
            [
              {
                type: "select",
                value: "",
                valueEn: "",
                options: ["예", "아니오"],
                optionsEn: ["Yes", "No"],
              },
              {
                type: "select",
                value: "",
                valueEn: "",
                options: ["의존성만", "영향만", "의존성 및 영향 모두"],
                optionsEn: [
                  "Dependencies only",
                  "Impacts only",
                  "Both dependencies and impacts",
                ],
              },
            ],
          ],
        },
        {
          id: "q-2.2.1",
          number: "2.2.1",
          title:
            "귀 조직은 환경 리스크 및/또는 기회를 식별, 평가 및 관리하는 프로세스를 보유하고 있습니까?",
          titleEn:
            "Does your organization have a process for identifying, assessing, and managing environmental risks and/or opportunities?",
          type: "table",
          tableHeaders: [
            "프로세스 구축 여부",
            "이 프로세스에서 평가되는 리스크 및/또는 기회",
            "이 프로세스가 의존성 및/또는 영향 프로세스에 의해 정보를 제공받습니까?",
          ],
          tableHeadersEn: [
            "Process in place",
            "Risks and/or opportunities evaluated in this process",
            "Is this process informed by the dependencies and/or impacts process?",
          ],
          tableRows: [
            [
              {
                type: "select",
                value: "",
                valueEn: "",
                options: ["예", "아니오"],
                optionsEn: ["Yes", "No"],
              },
              {
                type: "select",
                value: "",
                valueEn: "",
                options: ["리스크만", "기회만", "리스크 및 기회 모두"],
                optionsEn: [
                  "Risks only",
                  "Opportunities only",
                  "Both risks and opportunities",
                ],
              },
              {
                type: "select",
                value: "",
                valueEn: "",
                options: ["예", "아니오"],
                optionsEn: ["Yes", "No"],
              },
            ],
          ],
        },
        {
          id: "q-2.2.2",
          number: "2.2.2",
          title:
            "귀 조직의 환경 의존성, 영향, 리스크 및/또는 기회를 식별, 평가 및 관리하는 프로세스의 세부사항을 제공하십시오.",
          titleEn:
            "Provide details of your organization's process for identifying, assessing, and managing environmental dependencies, impacts, risks, and/or opportunities.",
          type: "table",
          tableHeaders: [
            "환경 이슈",
            "이 환경 이슈에 대한 프로세스에서 다루는 의존성, 영향, 리스크 및 기회",
            "포함된 가치사슬 단계",
            "커버리지",
            "포함된 공급업체 계층",
            "평가 유형",
            "평가 빈도",
          ],
          tableHeadersEn: [
            "Environmental issue",
            "Indicate which of dependencies, impacts, risks, and opportunities are covered",
            "Value chain stages covered",
            "Coverage",
            "Supplier tiers covered",
            "Type of assessment",
            "Frequency of assessment",
          ],
          tableRows: [
            [
              { type: "text", value: "", valueEn: "" },
              {
                type: "checkbox",
                value: [],
                valueEn: [],
                options: ["의존성", "영향", "리스크", "기회"],
                optionsEn: [
                  "Dependencies",
                  "Impacts",
                  "Risks",
                  "Opportunities",
                ],
              },
              {
                type: "checkbox",
                value: [],
                valueEn: [],
                options: ["직접 운영", "상류 가치사슬", "하류 가치사슬"],
                optionsEn: [
                  "Direct operations",
                  "Upstream value chain",
                  "Downstream value chain",
                ],
              },
              {
                type: "select",
                value: "",
                valueEn: "",
                options: ["전체", "부분"],
                optionsEn: ["Full", "Partial"],
              },
              {
                type: "checkbox",
                value: [],
                valueEn: [],
                options: ["1차 공급업체", "2차 공급업체"],
                optionsEn: ["Tier 1 suppliers", "Tier 2 suppliers"],
              },
              {
                type: "select",
                value: "",
                valueEn: "",
                options: ["정성적만", "정량적만", "정성적 및 정량적"],
                optionsEn: [
                  "Qualitative only",
                  "Quantitative only",
                  "Qualitative and quantitative",
                ],
              },
              {
                type: "select",
                value: "",
                valueEn: "",
                options: ["연간", "분기별", "월별"],
                optionsEn: ["Annually", "Quarterly", "Monthly"],
              },
            ],
          ],
        },
        {
          id: "q-2.2.2.11",
          number: "2.2.2.11",
          title: "사용된 위치 특정성",
          titleEn: "Location-specificity used",
          type: "select",
          value: "",
          valueEn: "",
          options: ["사이트별", "지역별", "전역"],
          optionsEn: ["Site-specific", "Regional", "Global"],
        },
        {
          id: "q-2.2.2.16",
          number: "2.2.2.16",
          title: "프로세스의 추가 세부사항",
          titleEn: "Further details of process",
          type: "textarea",
          value: "",
          valueEn: "",
        },
      ],
    },
    {
      number: "3.1",
      title: "기후 관련 거버넌스 및 전략",
      titleEn: "Climate-related governance and strategy",
      description: "",
      questions: [
        {
          id: "q-3.1",
          number: "3.1",
          title:
            "귀 조직의 기후 관련 거버넌스 및 전략 수립 프로세스를 설명하십시오.",
          titleEn:
            "Describe your organization's climate-related governance and strategic planning process.",
          type: "textarea",
          value: "",
          valueEn: "",
        },
      ],
    },
  ];
};

// Generate answers for question (simplified version)
const generateAnswersForQuestion = (
  questionId: string,
  question: any,
  feedback?: string
) => {
  const timestamp = Date.now();

  if (question?.type === "textarea") {
    const baseTextKo = `우리 조직은 ${
      question.title || "해당 주제"
    }에 대해 체계적인 접근 방식을 취하고 있습니다.`;
    const baseTextEn = `Our organization takes a systematic approach to ${
      question.titleEn || question.title || "this topic"
    }.`;

    return [
      {
        id: `answer-${timestamp}-1`,
        content: {
          rows: [],
          detailedText: baseTextEn,
          detailedTextKo: baseTextKo,
          detailedTextEn: baseTextEn,
          insight: "이 답변은 조직의 체계적인 접근 방식을 강조합니다.",
          insightEn:
            "This answer emphasizes the organization's systematic approach.",
        },
      },
      {
        id: `answer-${timestamp}-2`,
        content: {
          rows: [],
          detailedText: baseTextEn + " We conduct regular reviews.",
          detailedTextKo: baseTextKo + " 우리는 정기적으로 검토합니다.",
          detailedTextEn: baseTextEn + " We conduct regular reviews.",
          insight: "이 답변은 정기적 검토를 강조합니다.",
          insightEn: "This answer emphasizes regular reviews.",
        },
      },
    ];
  }

  // Default for other types
  return [
    {
      id: `answer-${timestamp}-1`,
      content: {
        rows: [
          {
            number: question.number || "",
            detail: question.title || "",
            detailEn: question.titleEn || "",
            answer: "Yes",
            answerKo: "예",
            answerEn: "Yes",
          },
        ],
        insight: "이 답변은 CDP 평가 기준에 부합합니다.",
        insightEn: "This answer aligns with CDP assessment criteria.",
      },
    },
  ];
};

// Watch for state changes and auto-save
watch(
  () => ({
    documentSections: documentSections.value,
    generatedAnswersByQuestion: generatedAnswersByQuestion.value,
    appliedAnswers: appliedAnswers.value,
  }),
  () => {
    persistState(false);
  },
  { deep: true }
);

// Initialize on mount
onMounted(async () => {
  try {
    await initQuestionnaire();
  } catch (e) {
    console.warn(
      "Failed to load questionnaire/schema data; using saved/mock state",
      e
    );
  } finally {
    loadSavedState();
  }
});
</script>

<style scoped>
/* Slide panel transition */
.slide-panel-enter-active,
.slide-panel-leave-active {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.slide-panel-enter-active > div:first-child,
.slide-panel-leave-active > div:first-child {
  transition: opacity 0.3s ease;
}

.slide-panel-enter-active > div:last-child,
.slide-panel-leave-active > div:last-child {
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.slide-panel-enter-from,
.slide-panel-leave-to {
  opacity: 0;
}

.slide-panel-enter-from > div:last-child,
.slide-panel-leave-to > div:last-child {
  transform: translateX(100%);
}
</style>
