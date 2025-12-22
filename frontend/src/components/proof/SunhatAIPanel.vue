<template>
  <div class="flex flex-col h-full overflow-hidden" :style="containerStyle">
    <!-- Header -->
    <div
      v-if="!props.hideHeader"
      class="flex-shrink-0 px-4 py-3 flex items-center justify-between"
      :style="headerStyle"
    >
      <div class="flex items-center gap-3">
        <div
          class="w-9 h-9 rounded-xl flex items-center justify-center font-bold text-lg"
          :style="logoStyle"
        >
          S
        </div>
        <div>
          <div class="text-xs font-semibold text-white">
            Sunny SE Shared Workspace
          </div>
          <div class="text-xs mt-0.5 text-white/60">
            > CDP 2025 (T... > {{ currentSection }}
          </div>
        </div>
      </div>
      <div class="flex items-center gap-1.5">
        <!-- 전역 언어 토글 -->
        <button
          @click="globalKorean = !globalKorean"
          class="px-2.5 py-1.5 rounded-lg btn-header flex items-center gap-1.5 text-xs font-medium"
          :class="globalKorean ? 'text-blue-400' : 'text-white/70'"
        >
          <Languages :size="14" />
          {{ globalKorean ? '한국어' : 'EN' }}
        </button>
        <button class="p-2 rounded-lg btn-header" @click="emit('toggleExpand')">
          <ChevronDown v-if="!isExpanded" :size="16" class="text-white/80" />
          <ChevronUp v-else :size="16" class="text-white/80" />
        </button>
        <button class="p-2 rounded-lg btn-header" @click="emit('close')">
          <X :size="16" class="text-white/80" />
        </button>
      </div>
    </div>

    <!-- Content -->
    <div class="flex-1 overflow-y-auto px-4 py-4 scrollbar-thin">
      <!-- CDP Guideline Tip -->
      <div
        v-if="selectedQuestion && guidelineContent"
        class="mb-4 p-3 rounded-lg border bg-blue-500/10 border-blue-500/30"
      >
        <div class="flex items-start gap-2">
          <Info :size="16" class="mt-0.5 flex-shrink-0 text-blue-400" />
          <div class="flex-1 min-w-0">
            <div
              class="text-xs font-semibold mb-1 flex items-center gap-2 text-blue-400"
            >
              <span>{{ globalKorean ? 'CDP 가이드라인' : 'CDP Guidelines' }}</span>
              <span class="px-1.5 py-0.5 rounded text-xs bg-blue-500/20">
                {{ selectedQuestion.number }}
              </span>
            </div>
            <div class="text-xs leading-relaxed text-white/80 space-y-3">
              <div v-if="guidelineContent.rationale">
                <div class="text-[11px] font-semibold text-white/70 mb-1">
                  {{ globalKorean ? '질문 배경' : 'Rationale' }}
                </div>
                <p>
                  {{ guidelineContent.rationale }}
                </p>
              </div>

              <div v-if="guidelineContent.ambition?.length">
                <div class="text-[11px] font-semibold text-white/70 mb-1">
                  {{ globalKorean ? '모범 사례' : 'Ambition' }}
                </div>
                <ul class="list-disc pl-4 space-y-1">
                  <li v-for="(item, idx) in guidelineContent.ambition" :key="idx">
                    {{ item }}
                  </li>
                </ul>
              </div>

              <div v-if="guidelineContent.requestedContent?.length">
                <div class="text-[11px] font-semibold text-white/70 mb-1">
                  {{ globalKorean ? '요청 내용' : 'Requested Content' }}
                </div>
                <ul class="list-disc pl-4 space-y-1">
                  <li
                    v-for="(item, idx) in guidelineContent.requestedContent"
                    :key="idx"
                  >
                    {{ item }}
                  </li>
                </ul>
              </div>

              <div v-if="guidelineContent.conditionalLogic">
                <div class="text-[11px] font-semibold text-white/70 mb-1">
                  {{ globalKorean ? '조건부 노출' : 'Conditional Logic' }}
                </div>
                <p>
                  {{ guidelineContent.conditionalLogic }}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Question Display -->
      <div v-if="selectedQuestion" class="mb-4">
        <h3 class="text-sm font-semibold mb-3 leading-relaxed text-white">
          {{ selectedQuestion.title }}
        </h3>

        <!-- Action Buttons -->
        <div class="flex flex-col gap-2 mb-4">
          <button
            v-if="isGenerating"
            disabled
            class="w-full px-4 py-2.5 rounded-lg text-sm font-semibold flex items-center justify-center gap-2 opacity-50 bg-white/10 text-white"
          >
            <Loader2 :size="16" class="animate-spin" />
            Generating your answer...
          </button>

          <button
            v-else-if="!hasValidAnswers"
            @click="handleGenerateAnswer()"
            class="w-full px-4 py-2.5 rounded-lg text-sm font-semibold transition-all hover:opacity-90 flex items-center justify-center gap-2"
            :style="{
              backgroundColor: themeStore.theme.primary,
              color: 'white',
            }"
          >
            <Sparkles :size="16" />
            Generate an answer
          </button>

          <button
            v-else
            @click="handleGenerateDifferent"
            class="w-full px-4 py-2.5 rounded-lg text-sm font-semibold transition-all hover:opacity-90 flex items-center justify-center gap-2 bg-white/10 text-white"
          >
            <RefreshCw :size="16" />
            Generate a different answer
          </button>
        </div>

        <!-- Confidence & Sources Section -->
        <div
          v-if="hasValidAnswers && currentGeneratedAnswer"
          class="mt-4 rounded-lg border border-white/10 bg-white/[0.03] p-4"
        >
          <!-- Header -->
          <div class="flex items-center gap-2 text-xs font-semibold text-white/90 mb-4">
            <TrendingUp :size="14" />
            {{ globalKorean ? "답변 분석" : "Answer Analysis" }}
          </div>

          <!-- Confidence Indicator -->
          <div class="mb-4">
            <div class="flex items-center justify-between mb-2">
              <span class="text-xs text-white/70">
                {{ globalKorean ? "신뢰도" : "Confidence" }}
              </span>
              <span
                class="text-xs font-semibold px-2 py-0.5 rounded"
                :style="{
                  backgroundColor: confidenceColor + '20',
                  color: confidenceColor,
                }"
              >
                {{ confidenceLabel }} ({{ confidenceScore }}%)
              </span>
            </div>
            <div class="h-2 rounded-full bg-white/10 overflow-hidden">
              <div
                class="h-full rounded-full transition-all duration-500"
                :style="{
                  width: confidenceScore + '%',
                  backgroundColor: confidenceColor,
                }"
              />
            </div>
          </div>

          <!-- Source Pages -->
          <div v-if="sourcePages.length" class="mb-4">
            <div class="flex items-center gap-2 mb-2">
              <BookOpen :size="12" class="text-white/70" />
              <span class="text-xs text-white/70">
                {{ globalKorean ? "참조 페이지" : "Source Pages" }}
              </span>
            </div>
            <div class="flex flex-wrap gap-1.5">
              <span
                v-for="page in sourcePages"
                :key="page"
                class="px-2 py-0.5 rounded text-xs bg-blue-500/20 text-blue-400"
              >
                Page {{ page }}
              </span>
            </div>
          </div>

          <!-- Rationale -->
          <div v-if="displayedRationale" class="mb-4">
            <div class="text-xs font-semibold mb-2 text-white/70">
              {{ globalKorean ? "답변 근거" : "Rationale" }}
            </div>
            <p
              class="text-xs leading-relaxed text-white/80"
              v-html="displayedRationale.replace(/\n/g, '<br>')"
            />
          </div>

          <!-- Insight -->
          <div
            v-if="selectedAnswerAnalysis?.reason"
            class="pt-3 border-t border-white/10"
          >
            <div class="flex items-start gap-1.5 p-2 rounded bg-white/5">
              <Info :size="12" class="mt-0.5 flex-shrink-0 text-white/60" />
              <p class="text-xs leading-relaxed text-white/70">
                {{ selectedAnswerAnalysis.reason }}
              </p>
            </div>
          </div>
        </div>

        <!-- Detailed Answer Text -->
        <div
          v-if="showDetailedText"
          class="mt-4 p-4 rounded-lg bg-white/5 text-white/90"
        >
          <p class="text-sm leading-relaxed">
            {{ displayedDetailedText }}
          </p>
        </div>

        <!-- Generated Answers List -->
        <div v-if="hasValidAnswers" class="mt-4 space-y-3">
          <div
            v-for="(answer, idx) in validAnswers"
            :key="answer.id"
            class="p-3 rounded-lg border transition-all hover:border-opacity-50"
            :style="answerCardStyle(answer.id)"
          >
            <div class="flex items-center justify-between mb-3">
              <!-- 답변이 여러 개일 때만 번호 표시 -->
              <span v-if="validAnswers.length > 1" class="text-xs font-semibold text-white/90"
                >{{ globalKorean ? `답변 ${idx + 1}` : `Answer ${idx + 1}` }}</span
              >
              <span v-else class="text-xs font-semibold text-white/90"
                >{{ globalKorean ? '생성된 답변' : 'Generated Answer' }}</span
              >
              <button
                class="px-3 py-1 rounded text-xs font-semibold transition-all flex items-center gap-1 flex-shrink-0"
                :style="applyBtnStyle(answer.id)"
                :disabled="props.appliedAnswerId === answer.id"
                @click.stop="handleSelectAnswer(answer.id)"
              >
                <Check :size="12" />
                <span>{{
                  props.appliedAnswerId === answer.id ? "적용됨" : "적용하기"
                }}</span>
              </button>
            </div>

            <!-- Answer Insight -->
            <div
              v-if="answer.content.insight"
              class="mb-3 p-2 rounded text-xs bg-white/5 text-white/70"
            >
              <div class="flex items-start gap-1.5">
                <Info :size="12" class="mt-0.5 flex-shrink-0" />
                <span>{{ getInsightText(answer.content) }}</span>
              </div>
            </div>

            <!-- Answer Rows (table type) -->
            <div v-if="answer.content.rows?.length" class="mb-3 space-y-2">
              <div
                v-for="(row, rowIdx) in answer.content.rows"
                :key="rowIdx"
                class="p-2 rounded bg-white/[0.02]"
              >
                <div class="flex items-start justify-between gap-2">
                  <div class="flex-1 min-w-0">
                    <div class="text-xs font-semibold mb-1 text-white/80">
                      {{ getRowDetail(row) }}
                    </div>
                    <!-- 짧은 답변 (textarea가 아닌 경우) -->
                    <p
                      v-if="row.type !== 'textarea'"
                      class="text-xs leading-relaxed text-white/70"
                    >
                      {{ getRowAnswer(row) }}
                    </p>
                    <!-- 긴 답변 (textarea) - 접기/펼치기 -->
                    <div v-else>
                      <p
                        class="text-xs leading-relaxed text-white/70"
                        :class="{ 'line-clamp-3': !expandedRows[`${answer.id}-${rowIdx}`] }"
                        v-html="getRowAnswer(row)?.replace(/\n/g, '<br>')"
                      />
                      <button
                        v-if="(getRowAnswer(row)?.length ?? 0) > 150"
                        @click.stop="toggleRowExpand(answer.id, rowIdx)"
                        class="text-xs mt-1 text-blue-400 hover:text-blue-300"
                      >
                        <span class="flex items-center gap-1">
                          <ChevronRight
                            :size="12"
                            :class="{ 'rotate-90': expandedRows[`${answer.id}-${rowIdx}`] }"
                            class="transition-transform"
                          />
                          {{ expandedRows[`${answer.id}-${rowIdx}`]
                            ? (globalKorean ? '접기' : 'Collapse')
                            : (globalKorean ? '더 보기' : 'Show more')
                          }}
                        </span>
                      </button>
                    </div>
                  </div>
                  <!-- 타입 뱃지 -->
                  <span
                    v-if="row.type && row.type !== 'text'"
                    class="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-white/50 flex-shrink-0"
                  >
                    {{ row.type }}
                  </span>
                </div>
              </div>
            </div>

            <!-- Detailed Text (textarea type fallback) -->
            <div v-else-if="answer.content.detailedText?.trim()" class="mb-2">
              <p
                class="text-xs leading-relaxed text-white/70"
                v-html="getDetailedText(answer.content)?.replace(/\n/g, '<br>')"
              />
            </div>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div
        v-else
        class="flex flex-col items-center justify-center h-full text-center py-12"
      >
        <FileText :size="48" class="text-white/30" />
        <p class="mt-4 text-sm text-white/60">
          문서에서 문항을 클릭하여 AI 답변을 생성하세요
        </p>
      </div>
    </div>

    <!-- Chat Input for Feedback -->
    <div
      v-if="selectedQuestion && currentAnswer"
      class="flex-shrink-0 px-4 py-3 border-t border-white/10"
      :style="{ backgroundColor: themeStore.isDark ? '#1a1a1a' : '#2d2d2d' }"
    >
      <div class="flex gap-4">
      
       
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from "vue";
import {
  Sparkles,
  FileText,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  X,
  RefreshCw,
  Loader2,
  Languages,
  Send,
  Info,
  Check,
  BookOpen,
  TrendingUp,
} from "lucide-vue-next";
import { useThemeStore } from "../../stores/themeStore";
import type {
  AnswerContent,
  AnswerRow,
  GeneratedAnswer,
} from "../../api/types/cdpAnswer.types";

// Guidance Raw 구조 (백엔드 형식)
interface GuidanceRaw {
  rationale?: string | null;
  rationale_ko?: string | null;
  ambition?: string[] | null;
  ambition_ko?: string[] | null;
  requested_content?: string[] | null;
  requested_content_ko?: string[] | null;
  additional_information?: string | null;
  additional_information_ko?: string | null;
}

// Types (Frontend specific)
interface Question {
  id: string;
  number: string;
  title: string;
  type: string;
  // 직접 필드 (레거시 호환)
  rationale?: string | null;
  ambition?: string[] | null;
  requestedContent?: string[] | null;
  conditionalLogic?: string | null;
  // 한국어 필드
  rationale_ko?: string | null;
  ambition_ko?: string[] | null;
  requestedContent_ko?: string[] | null;
  // 중첩 구조 (백엔드 형식)
  guidance_raw?: GuidanceRaw | null;
  guidanceRaw?: GuidanceRaw | null;
}

// Props & Emits
const props = defineProps<{
  selectedQuestion: Question | null;
  generatedAnswers?: GeneratedAnswer[];
  appliedAnswerId?: string | null;
  hideHeader?: boolean;
}>();

const emit = defineEmits<{
  generateAnswer: [questionId: string, feedback?: string];
  selectAnswer: [answerId: string];
  close: [];
  toggleExpand: [];
}>();

// Store
const themeStore = useThemeStore();

// State
const isExpanded = ref(true);
const isGenerating = ref(false);
const globalKorean = ref(true);  // 전역 언어 설정 (기본: 한국어)
const feedbackText = ref("");
const currentAnswer = ref<AnswerContent | null>(null);
const expandedRows = ref<Record<string, boolean>>({});

// 전역 언어 설정을 참조하는 computed (하위 호환성)
const showAnswerTranslation = computed(() => globalKorean.value);
const showGeneratedAnswerTranslation = computed(() => globalKorean.value);
const showGuidelineKorean = computed(() => globalKorean.value);

// Toggle row expansion for textarea content
const toggleRowExpand = (answerId: string, rowIdx: number) => {
  const key = `${answerId}-${rowIdx}`;
  expandedRows.value[key] = !expandedRows.value[key];
};

// Computed Styles
const containerStyle = computed(() => ({
  backgroundColor: themeStore.isDark ? "#1a1a1a" : "#2d2d2d",
  color: "#ffffff",
  borderRadius: "1rem",
  border: "1px solid rgba(255, 255, 255, 0.1)",
  boxShadow: themeStore.isDark
    ? "0 20px 60px rgba(0, 0, 0, 0.5), 0 8px 25px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05)"
    : "0 20px 60px rgba(0, 0, 0, 0.15), 0 8px 25px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.1)",
}));

const headerStyle = computed(() => ({
  borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
  backgroundColor: themeStore.isDark
    ? "rgba(26, 26, 26, 0.9)"
    : "rgba(45, 45, 45, 0.9)",
}));

const logoStyle = computed(() => ({
  backgroundColor: themeStore.theme.primary,
  color: "white",
  boxShadow:
    "0 4px 12px rgba(16, 185, 129, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2)",
}));

const sendBtnStyle = computed(() => ({
  backgroundColor:
    feedbackText.value.trim() && !isGenerating.value
      ? themeStore.theme.primary
      : "rgba(255, 255, 255, 0.1)",
  color: "white",
}));

// Helper Style Functions
const answerCardStyle = (answerId: string) => {
  const isApplied = props.appliedAnswerId === answerId;
  return {
    borderColor: isApplied
      ? themeStore.theme.primary
      : "rgba(255, 255, 255, 0.2)",
    backgroundColor: isApplied
      ? "rgba(16, 185, 129, 0.1)"
      : "rgba(255, 255, 255, 0.03)",
    boxShadow: isApplied ? `0 0 15px ${themeStore.theme.primary}40` : "none",
  };
};

const applyBtnStyle = (answerId: string) => {
  const isApplied = props.appliedAnswerId === answerId;
  return {
    backgroundColor: isApplied
      ? "rgba(16, 185, 129, 0.15)"
      : themeStore.theme.primary,
    color: isApplied ? themeStore.theme.primary : "white",
    opacity: isApplied ? 0.95 : 1,
    cursor: isApplied ? "default" : "pointer",
  };
};

// Computed Values
const currentSection = computed(() => {
  if (!props.selectedQuestion) return "2";
  const match = props.selectedQuestion.number.match(/^(\d+\.\d+)/);
  return match ? match[1] : "2";
});

const hasValidAnswers = computed(() => {
  if (!props.generatedAnswers?.length) return false;
  return props.generatedAnswers.some((answer) => {
    const hasRows = answer.content?.rows?.length;
    const hasDetailedText = answer.content?.detailedText?.trim();
    return hasRows || hasDetailedText;
  });
});

const validAnswers = computed(() => {
  if (!props.generatedAnswers) return [];
  return props.generatedAnswers.filter((answer) => {
    const hasRows = answer.content?.rows?.length;
    const hasDetailedText = answer.content?.detailedText?.trim();
    return hasRows || hasDetailedText;
  });
});

const showDetailedText = computed(() => {
  return (
    hasValidAnswers.value &&
    currentAnswer.value?.detailedText?.trim() &&
    !currentAnswer.value?.rows?.length
  );
});

const displayedDetailedText = computed(() => {
  if (!currentAnswer.value) return "";
  return showAnswerTranslation.value
    ? currentAnswer.value.detailedTextKo || currentAnswer.value.detailedText
    : currentAnswer.value.detailedTextEn || currentAnswer.value.detailedText;
});

const guidelineContent = computed(() => {
  const q = props.selectedQuestion;
  if (!q) return null;

  // guidance_raw 객체에서 가져오거나 직접 필드에서 가져옴
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const gr = (q.guidanceRaw || q.guidance_raw || q) as any;
  const useKorean = showGuidelineKorean.value;

  // 언어에 따라 적절한 필드 선택
  const rationale = useKorean
    ? gr.rationale_ko || gr.rationale
    : gr.rationale || gr.rationale_ko;

  const ambition = useKorean
    ? gr.ambition_ko || gr.ambition
    : gr.ambition || gr.ambition_ko;

  const requestedContent = useKorean
    ? gr.requested_content_ko || gr.requested_content || gr.requestedContent
    : gr.requested_content || gr.requestedContent || gr.requested_content_ko;

  const conditionalLogic = gr.conditionalLogic || gr.conditional_logic || q.conditionalLogic || null;

  if (
    !rationale &&
    (!ambition || ambition.length === 0) &&
    (!requestedContent || requestedContent.length === 0) &&
    !conditionalLogic
  ) {
    return null;
  }

  return { rationale, ambition, requestedContent, conditionalLogic };
});

// 현재 선택된/적용된 답변 가져오기
const currentGeneratedAnswer = computed(() => {
  if (!props.generatedAnswers?.length) return null;
  return props.appliedAnswerId
    ? props.generatedAnswers.find((a) => a.id === props.appliedAnswerId)
    : props.generatedAnswers[0];
});

// 신뢰도 표시 관련 computed
const confidenceScore = computed(() => {
  const answer = currentGeneratedAnswer.value;
  if (!answer) return 0;
  return Math.round((answer.confidence ?? 0) * 100);
});

const confidenceColor = computed(() => {
  const score = confidenceScore.value;
  if (score >= 70) return "#10b981"; // green
  if (score >= 50) return "#f59e0b"; // yellow
  return "#ef4444"; // red
});

const confidenceLabel = computed(() => {
  const score = confidenceScore.value;
  const useKorean = showAnswerTranslation.value;
  if (score >= 70) return useKorean ? "높음" : "High";
  if (score >= 50) return useKorean ? "중간" : "Medium";
  return useKorean ? "낮음" : "Low";
});

// 출처 페이지 번호 목록
const sourcePages = computed(() => {
  const answer = currentGeneratedAnswer.value;
  if (!answer?.sources?.length) return [];
  // 중복 제거하고 정렬
  const pages = [...new Set(answer.sources.map((s) => s.page_num))].sort(
    (a, b) => a - b
  );
  return pages.slice(0, 5);
});

// Rationale 텍스트 가져오기
const displayedRationale = computed(() => {
  const content = currentAnswer.value;
  if (!content) return "";
  return showAnswerTranslation.value
    ? content.rationaleKo || content.rationaleEn || ""
    : content.rationaleEn || content.rationaleKo || "";
});

const selectedAnswerAnalysis = computed(() => {
  const useKorean = showAnswerTranslation.value;

  const getAnalysis = (content: AnswerContent | null) => {
    if (!content) return null;

    const analysis = useKorean
      ? content.analysisKo || content.analysis || ""
      : content.analysisEn || content.analysis || "";
    const evidence = content.evidence || [];
    const reason = useKorean
      ? content.insight || ""
      : content.insightEn || content.insight || "";

    if (analysis || evidence.length || reason) {
      return { analysis, evidence, reason };
    }
    return null;
  };

  // Try from generatedAnswers first
  if (props.generatedAnswers?.length) {
    const targetAnswer = props.appliedAnswerId
      ? props.generatedAnswers.find((a) => a.id === props.appliedAnswerId)
      : props.generatedAnswers[0];

    if (targetAnswer?.content) {
      const result = getAnalysis(targetAnswer.content);
      if (result) return result;
    }
  }

  // Fallback to currentAnswer
  return getAnalysis(currentAnswer.value);
});

// Helper Functions for Display
const getInsightText = (content: AnswerContent) => {
  return showGeneratedAnswerTranslation.value
    ? content.insight || content.insightEn
    : content.insightEn || content.insight;
};

const getRowDetail = (row: AnswerRow) => {
  return showGeneratedAnswerTranslation.value
    ? row.detail || row.detailEn
    : row.detailEn || row.detail;
};

const getRowAnswer = (row: AnswerRow) => {
  return showGeneratedAnswerTranslation.value
    ? row.answerKo || row.answer
    : row.answerEn || row.answer;
};

const getDetailedText = (content: AnswerContent) => {
  return showGeneratedAnswerTranslation.value
    ? content.detailedTextKo || content.detailedText
    : content.detailedTextEn || content.detailedText;
};

// Event Handlers
const handleGenerateAnswer = async (feedback?: string) => {
  if (!props.selectedQuestion) return;

  isGenerating.value = true;
  currentAnswer.value = null;

  emit("generateAnswer", props.selectedQuestion.id, feedback);

  // Simulate API call
  setTimeout(() => {
    currentAnswer.value = feedback
      ? generateMockAnswerWithFeedback(props.selectedQuestion!, feedback)
      : generateMockAnswer(props.selectedQuestion!);
    isGenerating.value = false;
  }, 2000);
};

const handleGenerateDifferent = () => handleGenerateAnswer();

const handleSelectAnswer = (answerId: string) => {
  const answer = props.generatedAnswers?.find((a) => a.id === answerId);
  if (answer) {
    emit("selectAnswer", answerId);
    currentAnswer.value = answer.content;
  }
};

const handleSendFeedback = async () => {
  if (!feedbackText.value.trim() || !props.selectedQuestion) return;

  isGenerating.value = true;
  currentAnswer.value = null;

  emit("generateAnswer", props.selectedQuestion.id, feedbackText.value);

  setTimeout(() => {
    currentAnswer.value = generateMockAnswerWithFeedback(
      props.selectedQuestion!,
      feedbackText.value
    );
    isGenerating.value = false;
    feedbackText.value = "";
  }, 2000);
};

// Mock Data Generators
const generateMockAnswer = (question: Question): AnswerContent => {
  if (question.number.startsWith("2.2.2")) {
    return {
      rows: [
        {
          number: "",
          detail: "환경 이슈",
          detailEn: "Environmental issue",
          answer: "Climate change",
          answerKo: "기후변화",
          answerEn: "Climate change",
        },
        {
          number: "",
          detail: "범위",
          detailEn: "Coverage",
          answer: "Dependencies, Impacts, Risks, and Opportunities",
          answerKo: "의존성, 영향, 리스크 및 기회",
          answerEn: "Dependencies, Impacts, Risks, and Opportunities",
        },
        {
          number: "",
          detail: "가치사슬 단계",
          detailEn: "Value chain stages",
          answer:
            "Direct operations, Upstream value chain, Downstream value chain",
          answerKo: "직접 운영, 상류 가치사슬, 하류 가치사슬",
          answerEn:
            "Direct operations, Upstream value chain, Downstream value chain",
        },
        {
          number: "",
          detail: "범위",
          detailEn: "Coverage",
          answer: "Full",
          answerKo: "전체",
          answerEn: "Full",
        },
        {
          number: "",
          detail: "평가 유형",
          detailEn: "Type of assessment",
          answer: "Qualitative and quantitative",
          answerKo: "정성적 및 정량적",
          answerEn: "Qualitative and quantitative",
        },
        {
          number: "",
          detail: "빈도",
          detailEn: "Frequency",
          answer: "Annually",
          answerKo: "연간",
          answerEn: "Annually",
        },
      ],
      detailedText:
        "This process is company-wide, systematically identifies and assesses risks/opportunities based on TCFD recommendations.",
      detailedTextKo:
        "이 프로세스는 전사적으로 적용되며, TCFD 권고사항을 기반으로 리스크/기회를 체계적으로 식별하고 평가합니다.",
      detailedTextEn:
        "This process is company-wide, systematically identifies and assesses risks/opportunities based on TCFD recommendations.",
    };
  }

  if (question.number.startsWith("2.2")) {
    return {
      rows: [
        {
          number: "",
          detail: "프로세스 유무",
          detailEn: "Process in place",
          answer: "Yes - Our organization has a comprehensive process.",
          answerKo: "예 - 우리 조직은 포괄적인 프로세스를 보유하고 있습니다.",
          answerEn: "Yes - Our organization has a comprehensive process.",
        },
        {
          number: "",
          detail: "의존성/영향 평가",
          detailEn: "Dependencies and/or impacts evaluated",
          answer: "Both dependencies and impacts are evaluated.",
          answerKo: "의존성과 영향 모두를 평가합니다.",
          answerEn: "Both dependencies and impacts are evaluated.",
        },
      ],
      detailedText:
        "Our organization has established a systematic process that covers both environmental dependencies and impacts.",
      detailedTextKo:
        "우리 조직은 환경 의존성과 영향을 모두 다루는 체계적인 프로세스를 수립했습니다.",
      detailedTextEn:
        "Our organization has established a systematic process that covers both environmental dependencies and impacts.",
    };
  }

  return {
    rows: [
      {
        number: "",
        detail: "답변",
        detailEn: "Answer",
        answer: "A comprehensive response based on CDP guidelines.",
        answerKo: "CDP 가이드라인을 기반으로 한 포괄적인 답변입니다.",
        answerEn: "A comprehensive response based on CDP guidelines.",
      },
    ],
    detailedText:
      "This is a generated answer based on CDP assessment criteria.",
    detailedTextKo: "이는 CDP 평가 기준을 기반으로 생성된 답변입니다.",
    detailedTextEn:
      "This is a generated answer based on CDP assessment criteria.",
  };
};

const generateMockAnswerWithFeedback = (
  question: Question,
  feedback: string
): AnswerContent => {
  const baseAnswer = generateMockAnswer(question);
  if (baseAnswer.detailedText) {
    baseAnswer.detailedText += ` Additionally, based on your feedback: ${feedback}`;
    baseAnswer.detailedTextKo += ` 또한 귀하의 피드백을 바탕으로: ${feedback}`;
    baseAnswer.detailedTextEn += ` Additionally, based on your feedback: ${feedback}`;
  }
  return baseAnswer;
};

// Watchers
watch(
  () => props.selectedQuestion?.id,
  () => {
    // 전역 언어 설정은 유지, 피드백만 초기화
    feedbackText.value = "";

    const answers = props.generatedAnswers || [];
    if (answers.length > 0) {
      const appliedAnswer = props.appliedAnswerId
        ? answers.find((a) => a.id === props.appliedAnswerId)
        : null;
      currentAnswer.value =
        appliedAnswer?.content || answers[0]?.content || null;
    } else {
      currentAnswer.value = null;
    }
  },
  { immediate: true }
);

watch(
  () => props.generatedAnswers,
  (newAnswers) => {
    if (!props.selectedQuestion || !newAnswers?.length) return;

    if (props.appliedAnswerId) {
      const appliedAnswer = newAnswers.find(
        (a) => a.id === props.appliedAnswerId
      );
      if (appliedAnswer) {
        currentAnswer.value = appliedAnswer.content;
        return;
      }
    }
    currentAnswer.value = newAnswers[0]?.content || null;
  },
  { deep: true, immediate: true }
);

watch(
  () => props.selectedQuestion,
  (newQuestion) => {
    if (newQuestion) {
      currentAnswer.value = null;
      isGenerating.value = false;
    }
  }
);
</script>

<style scoped>
.scrollbar-thin {
  scrollbar-width: thin;
  scrollbar-color: rgba(255, 255, 255, 0.2) transparent;
}

.btn-header {
  background-color: rgba(255, 255, 255, 0.05);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.btn-header:hover {
  background-color: rgba(255, 255, 255, 0.1);
}

.line-clamp-3 {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>
