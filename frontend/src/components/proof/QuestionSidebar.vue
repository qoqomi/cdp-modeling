<template>
  <div
    class="flex flex-col h-full"
    :style="{
      backgroundColor: themeStore.isDark
        ? 'rgba(17, 24, 39, 0.95)'
        : themeStore.theme.whiteAlpha[95],
      border: `1px solid ${themeStore.theme.cardBorder}`,
      borderRadius: '1rem',
      boxShadow: themeStore.theme.elevation.card,
      width: '280px',
      minWidth: '280px',
      overflow: 'hidden',
    }"
  >
    <!-- Header with 3D Icon -->
    <div
      class="flex-shrink-0 px-4 py-3"
      :style="{
        borderBottom: `1px solid ${themeStore.theme.cardBorder}`,
        backgroundColor: themeStore.isDark
          ? 'rgba(15, 23, 42, 0.8)'
          : 'rgba(249, 250, 251, 0.8)',
      }"
    >
      <div class="flex items-center gap-3">
        <div
          class="w-8 h-8 rounded-lg flex items-center justify-center"
          :style="{
            backgroundColor: themeStore.theme.primaryAlpha[15],
            boxShadow: themeStore.theme.elevation.icon,
          }"
        >
          <List :size="16" :style="{ color: themeStore.theme.primary }" />
        </div>
        <h3
          class="text-sm font-bold tracking-wide"
          :style="{ color: themeStore.theme.textPrimary }"
        >
          문항 목록
        </h3>
      </div>
    </div>

    <!-- Question List -->
    <div
      class="flex-1 overflow-y-auto p-3 custom-scrollbar"
      style="
        min-height: 0;
        margin: -2px;
        padding-left: calc(0.75rem + 2px);
        padding-right: calc(0.75rem + 2px);
        padding-top: calc(0.75rem + 2px);
        padding-bottom: calc(0.75rem + 2px);
      "
    >
      <div
        v-for="(section, sIdx) in questionStructure"
        :key="sIdx"
        class="mb-3"
      >
        <!-- Section Header with 3D Style -->
        <button
          @click="toggleSection(section.number)"
          class="w-full px-3 py-2.5 text-left flex items-center justify-between mb-1.5 section-btn"
          :style="{
            backgroundColor: expandedSections.has(section.number)
              ? themeStore.theme.primaryAlpha[15]
              : themeStore.isDark
              ? 'rgba(255, 255, 255, 0.03)'
              : 'rgba(0, 0, 0, 0.02)',
            color: themeStore.theme.textPrimary,
            borderRadius: '0.75rem',
            border: `1px solid ${
              expandedSections.has(section.number)
                ? themeStore.theme.cardBorder
                : 'transparent'
            }`,
            boxShadow: expandedSections.has(section.number)
              ? themeStore.theme.elevation.input
              : 'none',
            transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          }"
        >
          <span class="text-sm font-semibold">{{ section.number }}</span>
          <ChevronDown
            :size="14"
            :class="{
              'rotate-180': expandedSections.has(section.number),
            }"
            class="transition-transform"
            :style="{ color: themeStore.theme.textSecondary }"
          />
        </button>

        <!-- Sub Questions with 3D Style -->
        <div
          v-if="expandedSections.has(section.number)"
          class="ml-3 space-y-1.5 mt-1"
        >
          <button
            v-for="(question, qIdx) in section.questions"
            :key="qIdx"
            @click="handleQuestionClick(question.id)"
            class="w-full px-3 py-2 text-left text-xs question-btn"
            :style="{
              backgroundColor:
                selectedQuestionId === question.id
                  ? themeStore.theme.primary
                  : themeStore.isDark
                  ? 'rgba(255, 255, 255, 0.03)'
                  : 'rgba(0, 0, 0, 0.02)',
              color:
                selectedQuestionId === question.id
                  ? 'white'
                  : themeStore.theme.textSecondary,
              borderRadius: '0.5rem',
              border: `1px solid ${
                selectedQuestionId === question.id
                  ? themeStore.theme.primary
                  : 'transparent'
              }`,
              boxShadow:
                selectedQuestionId === question.id
                  ? themeStore.theme.elevation.button
                  : 'none',
              transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
            }"
          >
            {{ question.number }} {{ question.shortTitle }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from "vue";
import { ChevronDown, List } from "lucide-vue-next";
import { useThemeStore } from "../../stores/themeStore";

const themeStore = useThemeStore();

const props = defineProps<{
  selectedQuestionId: string | null;
  questionStructure: Array<{
    number: string;
    questions: Array<{
      id: string;
      number: string;
      shortTitle: string;
    }>;
  }>;
}>();

const emit = defineEmits<{
  questionSelect: [questionId: string];
}>();

// 기본으로는 모든 섹션이 열린 상태가 되도록 props를 보고 초기화
const expandedSections = ref<Set<string>>(new Set());

const initExpandedSections = () => {
  const next = new Set<string>();
  for (const section of props.questionStructure) {
    next.add(section.number);
  }
  expandedSections.value = next;
};

// questionStructure가 바뀔 때마다(처음 포함) 전체 섹션을 열어 둔다
watch(
  () => props.questionStructure,
  () => {
    initExpandedSections();
  },
  { immediate: true, deep: true }
);

const toggleSection = (sectionNumber: string) => {
  if (expandedSections.value.has(sectionNumber)) {
    expandedSections.value.delete(sectionNumber);
  } else {
    expandedSections.value.add(sectionNumber);
  }
};

const handleQuestionClick = (questionId: string) => {
  emit("questionSelect", questionId);
};
</script>

<style scoped>
/* Section Button 3D Effect */
.section-btn {
  position: relative;
  z-index: 1;
}

.section-btn:hover {
  transform: translateY(-1px);
  z-index: 2;
}

.section-btn:active {
  transform: translateY(0);
}

/* Question Button 3D Effect */
.question-btn {
  position: relative;
  z-index: 1;
}

.question-btn:hover {
  transform: translateX(2px);
  z-index: 2;
}

.question-btn:active {
  transform: translateX(0);
}
</style>
