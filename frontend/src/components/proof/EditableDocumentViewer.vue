<template>
  <div class="flex flex-col h-full">
    <!-- Header with 3D Style -->
    <div
      class="flex-shrink-0 px-6 py-4 flex items-center justify-between"
      :style="{
        borderBottom: `1px solid ${themeStore.theme.cardBorder}`,
        backgroundColor: themeStore.isDark
          ? 'rgba(15, 23, 42, 0.8)'
          : 'rgba(249, 250, 251, 0.8)',
      }"
    >
      <div class="flex items-center gap-3">
        <!-- 3D Icon -->
        <div
          class="w-8 h-8 rounded-lg flex items-center justify-center"
          :style="{
            backgroundColor: themeStore.theme.primaryAlpha[15],
            boxShadow: themeStore.theme.elevation.icon,
          }"
        >
          <FileText :size="16" :style="{ color: themeStore.theme.primary }" />
        </div>
        <h3
          class="text-sm font-bold tracking-wide"
          :style="{ color: themeStore.theme.textPrimary }"
        >
          문서 뷰어
        </h3>
        <button
          @click="handleToggleTranslation"
          class="text-xs px-3 py-1.5 flex items-center gap-1.5 toggle-btn"
          :style="{
            backgroundColor: showTranslation
              ? themeStore.theme.primary
              : themeStore.theme.primaryAlpha[10],
            color: showTranslation ? 'white' : themeStore.theme.primary,
            borderRadius: '0.5rem',
            border: `1px solid ${
              showTranslation
                ? themeStore.theme.primary
                : themeStore.theme.cardBorder
            }`,
            boxShadow: showTranslation
              ? themeStore.theme.elevation.button
              : themeStore.theme.elevation.input,
          }"
        >
          <Languages :size="14" />
          {{ showTranslation ? "한국어" : "English" }}
        </button>
      </div>
      <div class="flex items-center gap-2">
        <button
          @click="handleSave"
          class="text-xs px-4 py-2 flex items-center gap-1.5 font-semibold save-btn"
          :style="{
            backgroundColor: themeStore.theme.primary,
            color: 'white',
            borderRadius: '0.5rem',
            boxShadow: themeStore.theme.elevation.button,
          }"
        >
          <Save :size="14" />
          Save
        </button>
        <button
          @click="handleFullView"
          class="text-xs px-3 py-2 view-btn"
          :style="{
            backgroundColor: themeStore.theme.primaryAlpha[10],
            color: themeStore.theme.primary,
            borderRadius: '0.5rem',
            border: `1px solid ${themeStore.theme.cardBorder}`,
            boxShadow: themeStore.theme.elevation.input,
          }"
        >
          전체 보기
        </button>
      </div>
    </div>

    <!-- Document Content -->
    <div
      ref="scrollContainer"
      class="flex-1 overflow-y-auto p-6 custom-scrollbar"
      style="min-height: 0"
      :style="{
        backgroundColor: themeStore.isDark
          ? 'rgba(15, 23, 42, 0.4)'
          : 'rgba(255, 255, 255, 0.5)',
      }"
    >
      <!-- PDF Viewer (if documentUrl is a PDF) -->
      <div v-if="props.documentUrl && isPdfFile" class="w-full h-full">
        <iframe
          :src="props.documentUrl"
          class="w-full h-full border-0 rounded-lg"
          style="min-height: 800px"
        ></iframe>
      </div>

      <!-- Editable Form Content -->
      <div
        v-else
        class="max-w-4xl mx-auto"
        :style="{
          color: themeStore.theme.textPrimary,
        }"
      >
        <!-- Mock CDP Form Structure -->
        <div
          v-for="(section, sectionIdx) in documentSections"
          :key="sectionIdx"
          class="mb-8"
        >
          <!-- Section Header with 3D Style -->
          <div
            class="mb-4 px-4 py-3 overflow-hidden"
            :style="{
              backgroundColor: themeStore.isDark
                ? 'rgba(30, 41, 59, 0.8)'
                : 'rgba(241, 245, 249, 0.9)',
              borderRadius: '0.75rem',
              border: `1px solid ${themeStore.theme.cardBorder}`,
              boxShadow: themeStore.theme.elevation.input,
            }"
          >
            <h4
              class="text-base font-bold mb-2"
              :style="{ color: themeStore.theme.textPrimary }"
            >
              {{ section.number }}
              {{
                showTranslation
                  ? section.title
                  : section.titleEn || section.title
              }}
            </h4>
            <p
              v-if="section.description"
              class="text-sm"
              :style="{ color: themeStore.theme.textSecondary }"
            >
              {{ section.description }}
            </p>
          </div>

          <!-- Questions in Section with 3D Style -->
          <div
            v-for="(question, qIdx) in section.questions"
            :key="qIdx"
            class="mb-6 p-4 transition-all cursor-pointer question-card"
            :ref="(el) => registerQuestionRef(question.id, el as HTMLElement | null)"
            :style="{
              backgroundColor: themeStore.isDark
                ? 'rgba(15, 23, 42, 0.6)'
                : themeStore.theme.whiteAlpha[95],
              border: `1px solid ${
                !props.isFullView && selectedQuestionId === question.id
                  ? themeStore.theme.primary
                  : themeStore.theme.cardBorder
              }`,
              borderRadius: '0.875rem',
              boxShadow:
                !props.isFullView && selectedQuestionId === question.id
                  ? `0 0 0 2px ${themeStore.theme.primaryAlpha[30]}, ${themeStore.theme.elevation.card}`
                  : themeStore.theme.elevation.input,
            }"
            @click="handleQuestionClick(question.id)"
          >
            <!-- Question Header with 3D Style -->
            <div
              class="mb-3 px-3 py-2.5 flex items-center justify-between overflow-hidden"
              :style="{
                backgroundColor: themeStore.isDark
                  ? 'rgba(30, 41, 59, 0.6)'
                  : 'rgba(241, 245, 249, 0.8)',
                borderRadius: '0.5rem',
                boxShadow: 'inset 0 1px 2px rgba(0, 0, 0, 0.04)',
              }"
            >
              <h5
                class="text-sm font-semibold"
                :style="{ color: themeStore.theme.textPrimary }"
              >
                {{ question.number }}
                {{
                  showTranslation
                    ? question.title
                    : question.titleEn || question.title
                }}
              </h5>
            </div>

            <!-- Question Content - Different Types -->
            <div class="space-y-4">
              <!-- Table Type Question -->
              <div v-if="question.type === 'table'" class="overflow-x-auto">
                <table class="w-full text-sm border-collapse">
                  <thead>
                    <tr
                      :style="{
                        backgroundColor: themeStore.isDark
                          ? 'rgba(30, 41, 59, 0.8)'
                          : 'rgba(241, 245, 249, 0.8)',
                      }"
                    >
                      <th
                        v-for="(header, hIdx) in showTranslation
                          ? question.tableHeaders
                          : question.tableHeadersEn || question.tableHeaders"
                        :key="hIdx"
                        class="px-4 py-2 text-left border"
                        :style="{
                          borderColor: themeStore.theme.cardBorder,
                          color: themeStore.theme.textPrimary,
                        }"
                      >
                        {{ header }}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="(row, rIdx) in question.tableRows"
                      :key="rIdx"
                      class="hover:bg-opacity-50"
                      :style="{
                        backgroundColor:
                          rIdx % 2 === 0
                            ? themeStore.isDark
                              ? 'rgba(15, 23, 42, 0.4)'
                              : 'rgba(255, 255, 255, 0.5)'
                            : 'transparent',
                      }"
                    >
                      <td
                        v-for="(cell, cIdx) in row"
                        :key="cIdx"
                        class="px-4 py-2 border"
                        :style="{
                          borderColor: themeStore.theme.cardBorder,
                          color: themeStore.theme.textPrimary,
                          backgroundColor: 'transparent',
                          boxShadow: shouldHighlightField(cell.value)
                            ? '0 0 0 1px rgba(250, 204, 21, 0.75), 0 0 10px rgba(250, 204, 21, 0.55)'
                            : 'none',
                        }"
                      >
                        <!-- Editable Cell -->
                        <template v-if="isCellVisible(row, cell)">
                        <input
                          v-if="cell.type === 'text'"
                          v-model="cell.value"
                          type="text"
                          class="w-full px-2 py-1 rounded border-0 bg-transparent focus:outline-none focus:ring-2 focus:ring-emerald-500/30"
                          :style="{
                            color: themeStore.theme.textPrimary,
                          }"
                          @input="
                            syncCellValue(
                              cell,
                              ($event.target as HTMLInputElement).value
                            )
                          "
                          @click="handleQuestionClick(question.id)"
                        />
                        <!-- Select Dropdown -->
                        <select
                          v-else-if="cell.type === 'select'"
                          v-model="cell.value"
                          class="w-full px-2 py-1 rounded border bg-transparent focus:outline-none focus:ring-2 focus:ring-emerald-500/30"
                          :style="{
                            backgroundColor: themeStore.isDark
                              ? 'rgba(15, 23, 42, 0.8)'
                              : 'rgba(255, 255, 255, 0.8)',
                            borderColor: themeStore.theme.cardBorder,
                            color: themeStore.theme.textPrimary,
                          }"
                          @change="
                            syncCellValue(
                              cell,
                              ($event.target as HTMLSelectElement).value
                            )
                          "
                          @click="handleQuestionClick(question.id)"
                        >
                          <option
                            v-for="(option, oIdx) in cell.optionsEn ||
                            cell.options"
                            :key="oIdx"
                            :value="
                              cell.optionsEn?.[oIdx] || cell.options[oIdx]
                            "
                          >
                            {{
                              showTranslation
                                ? cell.options?.[oIdx] || option
                                : cell.optionsEn?.[oIdx] || option
                            }}
                          </option>
                        </select>
                        <!-- Checkbox Group -->
                        <div
                          v-else-if="cell.type === 'checkbox'"
                          class="space-y-1"
                          @click="handleQuestionClick(question.id)"
                        >
                          <label
                            v-for="(option, oIdx) in cell.optionsEn ||
                            cell.options"
                            :key="oIdx"
                            class="flex items-center gap-2 cursor-pointer"
                          >
                            <input
                              type="checkbox"
                              :value="
                                cell.optionsEn?.[oIdx] || cell.options[oIdx]
                              "
                              :checked="
                                cell.value.includes(
                                  cell.optionsEn?.[oIdx] || cell.options[oIdx]
                                )
                              "
                              @change="
                                handleCellCheckboxToggle(
                                  cell,
                                  cell.optionsEn?.[oIdx] || cell.options[oIdx]
                                )
                              "
                              class="rounded"
                              :style="{
                                accentColor: themeStore.theme.primary,
                              }"
                            />
                            <span
                              class="text-xs"
                              :style="{ color: themeStore.theme.textPrimary }"
                            >
                              {{
                                showTranslation
                                  ? cell.options?.[oIdx] || option
                                  : cell.optionsEn?.[oIdx] || option
                              }}
                            </span>
                          </label>
                        </div>
                        </template>
                        <div v-else class="text-xs text-center italic select-none" :style="{ color: themeStore.theme.textSecondary, opacity: 0.5 }">—</div>
                      </td>
                    </tr>
                  </tbody>
                </table>
                <button
                  @click.stop="addTableRow(question)"
                  class="mt-3 px-3 py-1.5 text-xs font-medium add-row-btn"
                  :style="{
                    backgroundColor: themeStore.theme.primaryAlpha[10],
                    color: themeStore.theme.primary,
                    borderRadius: '0.5rem',
                    border: `1px solid ${themeStore.theme.cardBorder}`,
                    boxShadow: themeStore.theme.elevation.input,
                  }"
                >
                  + Add Row
                </button>
              </div>

              <!-- Textarea Type Question with 3D Style -->
              <div v-else-if="question.type === 'textarea'">
                <textarea
                  :value="
                    showTranslation
                      ? question.value || ''
                      : question.valueEn || ''
                  "
                  @input="
                    handleTextareaInput(
                      question,
                      ($event.target as HTMLTextAreaElement)?.value
                    )
                  "
                  @focus="handleQuestionClick(question.id)"
                  rows="6"
                  class="w-full px-4 py-3 resize-none focus:outline-none textarea-3d"
                  :style="{
                    backgroundColor: themeStore.isDark
                      ? 'rgba(15, 23, 42, 0.8)'
                      : themeStore.theme.whiteAlpha[95],
                    border: `1px solid ${
                      shouldHighlightField(
                        showTranslation ? question.value : question.valueEn
                      )
                        ? 'rgba(250, 204, 21, 0.9)'
                        : themeStore.theme.cardBorder
                    }`,
                    borderRadius: '0.75rem',
                    boxShadow: shouldHighlightField(
                      showTranslation ? question.value : question.valueEn
                    )
                      ? '0 0 0 2px rgba(250, 204, 21, 0.3), inset 0 1px 2px rgba(0, 0, 0, 0.04)'
                      : themeStore.theme.elevation.input,
                    color: themeStore.theme.textPrimary,
                  }"
                  :placeholder="
                    showTranslation
                      ? '답변을 입력하세요...'
                      : 'Enter your answer...'
                  "
                ></textarea>
              </div>

              <!-- Text/Number Type Question -->
              <div
                v-else-if="
                  question.type === 'text' ||
                  question.type === 'number' ||
                  question.type === 'percentage'
                "
              >
                <input
                  :value="
                    showTranslation ? question.value || '' : question.valueEn || ''
                  "
                  @input="
                    syncQuestionValue(
                      question,
                      ($event.target as HTMLInputElement)?.value
                    )
                  "
                  @focus="handleQuestionClick(question.id)"
                  class="w-full px-4 py-2.5 focus:outline-none textarea-3d"
                  :style="{
                    backgroundColor: themeStore.isDark
                      ? 'rgba(15, 23, 42, 0.8)'
                      : themeStore.theme.whiteAlpha[95],
                    border: `1px solid ${
                      shouldHighlightField(
                        showTranslation ? question.value : question.valueEn
                      )
                        ? 'rgba(250, 204, 21, 0.9)'
                        : themeStore.theme.cardBorder
                    }`,
                    borderRadius: '0.75rem',
                    boxShadow: shouldHighlightField(
                      showTranslation ? question.value : question.valueEn
                    )
                      ? '0 0 0 2px rgba(250, 204, 21, 0.3), inset 0 1px 2px rgba(0, 0, 0, 0.04)'
                      : themeStore.theme.elevation.input,
                    color: themeStore.theme.textPrimary,
                  }"
                  :placeholder="
                    showTranslation
                      ? '답변을 입력하세요...'
                      : 'Enter your answer...'
                  "
                />
              </div>

              <!-- Select Type Question with 3D Style -->
              <div v-else-if="question.type === 'select'">
                <select
                  v-model="question.value"
                  @change="
                    syncQuestionValue(
                      question,
                      ($event.target as HTMLSelectElement)?.value
                    )
                  "
                  class="w-full px-4 py-2.5 focus:outline-none select-3d"
                  :style="{
                    backgroundColor: themeStore.isDark
                      ? 'rgba(15, 23, 42, 0.8)'
                      : themeStore.theme.whiteAlpha[95],
                    border: `1px solid ${
                      shouldHighlightField(question.value)
                        ? 'rgba(250, 204, 21, 0.9)'
                        : themeStore.theme.cardBorder
                    }`,
                    borderRadius: '0.75rem',
                    boxShadow: shouldHighlightField(question.valueEn)
                      ? '0 0 0 2px rgba(250, 204, 21, 0.3), inset 0 1px 2px rgba(0, 0, 0, 0.04)'
                      : themeStore.theme.elevation.input,
                    color: themeStore.theme.textPrimary,
                  }"
                >
                  <option value="">
                    {{ showTranslation ? "선택하세요" : "Select" }}
                  </option>
                  <option
                    v-for="(optEn, oIdx) in question.optionsEn ||
                    question.options"
                    :key="oIdx"
                    :value="optEn"
                  >
                    {{
                      showTranslation
                        ? question.options?.[oIdx] || optEn
                        : optEn
                    }}
                  </option>
                </select>
              </div>

              <!-- Checkbox Type Question -->
              <div v-else-if="question.type === 'checkbox'">
                <div
                  class="space-y-2"
                  @click="handleQuestionClick(question.id)"
                >
                  <label
                    v-for="(optEn, oIdx) in question.optionsEn ||
                    question.options"
                    :key="oIdx"
                    class="flex items-center gap-2 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      :value="optEn"
                      :checked="question.value.includes(optEn)"
                      @change="handleQuestionCheckboxToggle(question, optEn)"
                      class="rounded"
                      :style="{
                        accentColor: themeStore.theme.primary,
                      }"
                    />
                    <span
                      class="text-sm"
                      :style="{ color: themeStore.theme.textPrimary }"
                    >
                      {{
                        showTranslation
                          ? question.options?.[oIdx] || optEn
                          : optEn
                      }}
                    </span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from "vue";
import { Languages, Save, FileText } from "lucide-vue-next";
import { useThemeStore } from "../../stores/themeStore";

const themeStore = useThemeStore();

const props = defineProps<{
  documentUrl?: string | null;
  selectedQuestionId?: string | null;
  showTranslation?: boolean;
  documentSections?: any[];
  appliedAnswers?: Record<string, string>;
  isFullView?: boolean;
  activeSection?: string | null;
}>();

const emit = defineEmits<{
  questionSelect: [questionId: string | null];
  save: [];
  applyAnswer: [answer: any];
  toggleTranslation: [];
  fullView: [];
}>();

const showTranslation = ref(props.showTranslation || false);
const scrollContainer = ref<HTMLElement | null>(null);
const questionRefs = new Map<string, HTMLElement>();

// Watch for prop changes
watch(
  () => props.showTranslation,
  (newVal) => {
    showTranslation.value = newVal || false;
  },
  { immediate: true }
);

// 적용된 문항에 대한 별도 네온 표시 제거 (이전 로직 비활성화)
const isApplied = (_questionId: string) => {
  return false;
};

// 개별 필드가 비어 있는지 체크 (전체 보기에서 노란색 하이라이트용)
const isEmptyValue = (v: any) => {
  if (Array.isArray(v)) return v.length === 0;
  return v === undefined || v === null || String(v).trim() === "";
};

const shouldHighlightField = (v: any) => {
  return !!(props.isFullView && isEmptyValue(v));
};

// 단일 문항 값 동기화 (value / valueEn 함께 유지)
const syncQuestionValue = (question: any, v: any) => {
  question.value = v;
  if ("valueEn" in question) {
    question.valueEn = v;
  }
};

// textarea 전용 입력 핸들러 (현재 보여주는 언어 기준으로 해당 필드만 수정)
const handleTextareaInput = (question: any, v: any) => {
  if (showTranslation.value) {
    // 한국어 보기 모드
    question.value = v;
  } else {
    // English 보기 모드
    if ("valueEn" in question) {
      question.valueEn = v;
    } else {
      question.value = v;
    }
  }
};

// 테이블 셀 값 동기화
const syncCellValue = (cell: any, v: any) => {
  cell.value = v;
  if ("valueEn" in cell) {
    cell.valueEn = v;
  }
};

// 체크박스 토글 유틸 (질문 단위)
const handleQuestionCheckboxToggle = (question: any, optEn: string) => {
  if (!Array.isArray(question.value)) {
    question.value = [];
  }
  const arr = [...question.value];
  const idx = arr.indexOf(optEn);
  if (idx >= 0) {
    arr.splice(idx, 1);
  } else {
    arr.push(optEn);
  }
  question.value = arr;
  if ("valueEn" in question) {
    question.valueEn = [...arr];
  }
};

// 체크박스 토글 유틸 (테이블 셀)
const handleCellCheckboxToggle = (cell: any, optEn: string) => {
  if (!Array.isArray(cell.value)) {
    cell.value = [];
  }
  const arr = [...cell.value];
  const idx = arr.indexOf(optEn);
  if (idx >= 0) {
    arr.splice(idx, 1);
  } else {
    arr.push(optEn);
  }
  cell.value = arr;
  if ("valueEn" in cell) {
    cell.valueEn = [...arr];
  }
};

const evaluateCondition = (
  condition: any,
  valuesByField: Record<string, any>
): boolean => {
  if (!condition) return true;

  if (Array.isArray(condition.any)) {
    return condition.any.some((c: any) => evaluateCondition(c, valuesByField));
  }
  if (Array.isArray(condition.all)) {
    return condition.all.every((c: any) => evaluateCondition(c, valuesByField));
  }

  const field = condition.field;
  if (!field) return true;

  const currentValue = valuesByField[field];
  if ("value" in condition) return currentValue === condition.value;
  if ("value_not" in condition) return currentValue !== condition.value_not;

  return true;
};

const isCellVisible = (row: any[], cell: any) => {
  const condition = cell?.condition;
  if (!condition) return true;

  const valuesByField: Record<string, any> = {};
  for (const c of row) {
    if (c?.schemaId) valuesByField[c.schemaId] = c.value;
  }
  return evaluateCondition(condition, valuesByField);
};

// 문항이 "아직 충분히 채워지지 않은" 상태인지 판단
// - textarea/select/checkbox: 값이 비어 있으면 미작성으로 간주
// - table: 하나라도 비어 있는 셀이 있으면 미작성으로 간주
const isQuestionEmpty = (question: any) => {
    switch (question.type) {
      case "textarea":
        return (
          (!question.value || !String(question.value).trim()) &&
          (!question.valueEn || !String(question.valueEn).trim())
        );
      case "text":
      case "number":
      case "percentage":
        return (
          (!question.value || !String(question.value).trim()) &&
          (!question.valueEn || !String(question.valueEn).trim())
        );
      case "select":
        return !question.value || question.value === "";
      case "checkbox":
        return (
          !question.value ||
        (Array.isArray(question.value) && question.value.length === 0)
      );
    case "table":
      if (!question.tableRows || !question.tableRows.length) return true;
      return question.tableRows.some((row: any[]) =>
        row.some((cell) => {
          if (!cell) return true;
          const v = cell.value;
          if (Array.isArray(v)) return v.length === 0;
          return v === undefined || v === null || String(v).trim() === "";
        })
      );
    default:
      return false;
  }
};

const registerQuestionRef = (id: string, el: HTMLElement | null) => {
  if (el) {
    questionRefs.set(id, el);
  } else {
    questionRefs.delete(id);
  }
};

const scrollToTop = () => {
  if (scrollContainer.value) {
    scrollContainer.value.scrollTo({ top: 0, behavior: "smooth" });
  }
};

const scrollToQuestion = (questionId: string) => {
  const el = questionRefs.get(questionId);
  const container = scrollContainer.value;
  if (el && container) {
    const offset = el.offsetTop - container.offsetTop - 16;
    container.scrollTo({ top: Math.max(offset, 0), behavior: "smooth" });
  }
};

const handleFullView = () => {
  scrollToTop();
  emit("fullView");
};

// 왼쪽 패널에서 문항 선택 시 자동 스크롤
watch(
  () => props.selectedQuestionId,
  (newId) => {
    if (newId) {
      scrollToQuestion(newId);
    } else {
      // 전체 보기일 때 맨 위로 스크롤
      scrollToTop();
    }
  }
);

// Check if document is PDF
const isPdfFile = computed(() => {
  if (!props.documentUrl) return false;
  return (
    props.documentUrl.toLowerCase().endsWith(".pdf") ||
    props.documentUrl.includes("application/pdf") ||
    (props.documentUrl.includes("blob:") && props.documentUrl.includes("pdf"))
  );
});

// 원본 섹션 목록 (props가 우선, 없으면 mock)
const allSections = computed(() => {
  if (props.documentSections && props.documentSections.length > 0) {
    return props.documentSections;
  }
  return mockDocumentSections.value;
});

// 전체보기 / 섹션별 보기 제어
const documentSections = computed(() => {
  const base = allSections.value;
  if (props.isFullView || !props.activeSection) {
    return base;
  }
  return base.filter((sec: any) => sec.number === props.activeSection);
});

const mockDocumentSections = ref([
  {
    number: "2.2",
    title:
      "Does your organization have a process for identifying, assessing, and managing environmental dependencies and/or impacts?",
    description: "",
    questions: [
      {
        id: "q-2.2",
        number: "2.2",
        title:
          "Does your organization have a process for identifying, assessing, and managing environmental dependencies and/or impacts?",
        type: "table",
        tableHeaders: [
          "Process in place",
          "Dependencies and/or impacts evaluated in this process",
        ],
        tableRows: [
          [
            {
              type: "select",
              value: "",
              options: ["Yes", "No"],
            },
            {
              type: "select",
              value: "",
              options: [
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
          "Does your organization have a process for identifying, assessing, and managing environmental risks and/or opportunities?",
        type: "table",
        tableHeaders: [
          "Process in place",
          "Risks and/or opportunities evaluated in this process",
          "Is this process informed by the dependencies and/or impacts process?",
        ],
        tableRows: [
          [
            {
              type: "select",
              value: "",
              options: ["Yes", "No"],
            },
            {
              type: "select",
              value: "",
              options: [
                "Risks only",
                "Opportunities only",
                "Both risks and opportunities",
              ],
            },
            {
              type: "select",
              value: "",
              options: ["Yes", "No"],
            },
          ],
        ],
      },
      {
        id: "q-2.2.2",
        number: "2.2.2",
        title:
          "Provide details of your organization's process for identifying, assessing, and managing environmental dependencies, impacts, risks, and/or opportunities.",
        type: "table",
        tableHeaders: [
          "Environmental issue",
          "Indicate which of dependencies, impacts, risks, and opportunities are covered by the process for this environmental issue",
          "Value chain stages covered",
          "Coverage",
          "Supplier tiers covered",
          "Type of assessment",
          "Frequency of assessment",
        ],
        tableRows: [
          [
            {
              type: "text",
              value: "",
            },
            {
              type: "checkbox",
              value: [],
              options: ["Climate change", "Water security", "Deforestation"],
            },
            {
              type: "checkbox",
              value: [],
              options: ["Dependencies", "Impacts", "Risks", "Opportunities"],
            },
            {
              type: "checkbox",
              value: [],
              options: [
                "Direct operations",
                "Upstream value chain",
                "Downstream value chain",
              ],
            },
            {
              type: "select",
              value: "",
              options: ["Full", "Partial"],
            },
            {
              type: "checkbox",
              value: [],
              options: ["Tier 1 suppliers", "Tier 2 suppliers"],
            },
            {
              type: "select",
              value: "",
              options: [
                "Qualitative only",
                "Quantitative only",
                "Qualitative and quantitative",
              ],
            },
            {
              type: "select",
              value: "",
              options: ["Annually", "Quarterly", "Monthly"],
            },
          ],
        ],
      },
      {
        id: "q-2.2.2.16",
        number: "2.2.2.16",
        title: "Further details of process",
        type: "textarea",
        value: "",
      },
    ],
  },
]);

const handleQuestionClick = (questionId: string) => {
  emit("questionSelect", questionId);
};

const handleSave = () => {
  emit("save");
};

const handleToggleTranslation = () => {
  showTranslation.value = !showTranslation.value;
  emit("toggleTranslation");
};

const addTableRow = (question: any) => {
  if (question.type === "table" && question.tableRows.length > 0) {
    const firstRow = question.tableRows[0];
    const newRow = firstRow.map((cell: any) => {
      if (cell.type === "checkbox") {
        return {
          ...cell,
          value: [],
        };
      } else if (cell.type === "select") {
        return {
          ...cell,
          value: cell.options[0] || "",
        };
      } else {
        return {
          ...cell,
          value: "",
        };
      }
    });
    question.tableRows.push(newRow);
  }
};
</script>

<style scoped>
/* 3D Button Effects */
.toggle-btn,
.save-btn,
.view-btn,
.add-row-btn {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  z-index: 1;
}

.toggle-btn:hover,
.view-btn:hover,
.add-row-btn:hover {
  transform: translateY(-1px);
  z-index: 2;
}

.save-btn:hover {
  transform: translateY(-1px) scale(1.02);
  filter: brightness(1.1);
  z-index: 2;
}

.toggle-btn:active,
.save-btn:active,
.view-btn:active,
.add-row-btn:active {
  transform: translateY(0) scale(0.98);
}

/* Question Card 3D Effect */
.question-card {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  z-index: 1;
  overflow: hidden;
}

.question-card:hover {
  transform: translateY(-2px);
  z-index: 2;
}

/* Textarea & Select 3D Focus Effect */
.textarea-3d,
.select-3d {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.textarea-3d:focus,
.select-3d:focus {
  transform: translateY(-1px);
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15),
    inset 0 1px 2px rgba(0, 0, 0, 0.04);
}
</style>
