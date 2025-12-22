<template>
  <div
    class="flex gap-4 h-full w-full p-1"
    style="min-height: 0; box-shadow: none"
  >
    <!-- Left: Two Panels (Top: Analysis Target, Bottom: History) -->
    <div
      class="flex flex-col gap-4"
      style="
        width: 450px;
        min-width: 450px;
        max-width: 450px;
        flex-shrink: 0;
        min-height: 0;
      "
    >
      <!-- Top Panel: Analysis Target Report with 3D Style -->
      <div
        class="flex flex-col panel-3d"
        style="flex: 0 0 auto"
        :style="{
          backgroundColor: themeStore.isDark
            ? 'rgba(17, 24, 39, 0.95)'
            : themeStore.theme.whiteAlpha[95],
          border: `1px solid ${themeStore.theme.cardBorder}`,
          borderRadius: '1rem',
          boxShadow: themeStore.theme.elevation.card,
          padding: '1rem',
        }"
      >
        <!-- Header with 3D Icon -->
        <div class="flex items-center gap-3 mb-3">
          <div
            class="w-8 h-8 rounded-lg flex items-center justify-center"
            :style="{
              backgroundColor: themeStore.theme.primaryAlpha[15],
              boxShadow: themeStore.theme.elevation.icon,
            }"
          >
            <FileSearch
              :size="16"
              :style="{ color: themeStore.theme.primary }"
            />
          </div>
          <h3
            class="text-xs font-bold tracking-wide"
            :style="{ color: themeStore.theme.textPrimary }"
          >
            분석 대상 보고서
          </h3>
        </div>

        <!-- File Upload Card with 3D Style -->
        <div
          v-if="!currentYearDocument"
          @click="triggerFileUpload"
          @dragover.prevent
          @drop.prevent="handleDrop"
          class="flex flex-col items-center justify-center cursor-pointer transition-all mb-3 upload-card"
          :style="{
            border: `2px dashed ${
              themeStore.isDark
                ? 'rgba(16, 185, 129, 0.3)'
                : 'rgba(16, 185, 129, 0.25)'
            }`,
            borderRadius: '0.75rem',
            backgroundColor: themeStore.theme.primaryAlpha[5],
            boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.03)',
            minHeight: '100px',
            padding: '1rem 0.75rem',
          }"
        >
          <Upload :size="28" :style="{ color: themeStore.theme.primary }" />
          <p
            class="mt-1.5 text-xs font-semibold"
            :style="{ color: themeStore.theme.textPrimary }"
          >
            파일을 드래그하거나 클릭하여 업로드
          </p>
          <p
            class="mt-0.5 text-xs"
            :style="{ color: themeStore.theme.textSecondary }"
          >
            PDF, DOCX, XLSX 지원
          </p>
        </div>

        <!-- Current Year Document Display with 3D Style -->
        <div
          v-else
          class="p-3 flex items-center justify-between mb-3 doc-card"
          :style="{
            backgroundColor: themeStore.theme.primaryAlpha[10],
            border: `1px solid ${themeStore.theme.primary}`,
            borderRadius: '0.75rem',
            boxShadow: themeStore.theme.elevation.input,
          }"
        >
          <div class="flex items-center gap-2 flex-1 min-w-0">
            <FileText :size="18" :style="{ color: themeStore.theme.primary }" />
            <div class="flex-1 min-w-0">
              <p
                class="text-xs font-semibold truncate"
                :style="{ color: themeStore.theme.textPrimary }"
              >
                {{ currentYearDocument.name }}
              </p>
              <p
                class="text-xs mt-0.5"
                :style="{ color: themeStore.theme.textSecondary }"
              >
                {{ formatFileSize(currentYearDocument.size) }}
              </p>
            </div>
          </div>
          <button
            @click="removeCurrentYearDocument"
            class="p-1 rounded-lg hover:bg-red-500/20 transition-colors"
          >
            <X :size="14" :style="{ color: '#ef4444' }" />
          </button>
        </div>

        <!-- Report Type Selection for New Upload -->
        <div v-if="currentYearDocument" class="mb-3">
          <label
            class="text-xs font-semibold mb-1.5 block"
            :style="{ color: themeStore.theme.textSecondary }"
          >
            연도 선택
          </label>
          <div class="flex gap-2 mb-2">
            <button
              v-for="year in availableYears"
              :key="year"
              @click="handleYearSelect(year)"
              @mousedown.stop
              type="button"
              class="flex-1 px-2.5 py-2 text-xs font-semibold cursor-pointer year-btn"
              :style="{
                backgroundColor:
                  selectedYear === year
                    ? themeStore.theme.primary
                    : themeStore.isDark
                    ? 'rgba(255, 255, 255, 0.05)'
                    : 'rgba(0, 0, 0, 0.03)',
                color:
                  selectedYear === year
                    ? 'white'
                    : themeStore.theme.textPrimary,
                border: `1px solid ${
                  selectedYear === year
                    ? themeStore.theme.primary
                    : themeStore.theme.cardBorder
                }`,
                borderRadius: '0.5rem',
                boxShadow:
                  selectedYear === year
                    ? themeStore.theme.elevation.button
                    : themeStore.theme.elevation.input,
                pointerEvents: 'auto',
                zIndex: 10,
              }"
            >
              {{ year }}년
            </button>
          </div>

          <label
            class="text-xs font-semibold mb-1.5 block mt-2"
            :style="{ color: themeStore.theme.textSecondary }"
          >
            보고서 타입
          </label>
          <div class="flex gap-2">
            <button
              v-for="type in reportTypes"
              :key="type.value"
              @click="handleReportTypeSelect(type.value)"
              @mousedown.stop
              type="button"
              class="flex-1 px-2.5 py-2 text-xs font-semibold cursor-pointer type-btn"
              :style="{
                backgroundColor:
                  selectedReportType === type.value
                    ? themeStore.theme.primary
                    : themeStore.isDark
                    ? 'rgba(255, 255, 255, 0.05)'
                    : 'rgba(0, 0, 0, 0.03)',
                color:
                  selectedReportType === type.value
                    ? 'white'
                    : themeStore.theme.textPrimary,
                border: `1px solid ${
                  selectedReportType === type.value
                    ? themeStore.theme.primary
                    : themeStore.theme.cardBorder
                }`,
                borderRadius: '0.5rem',
                boxShadow:
                  selectedReportType === type.value
                    ? themeStore.theme.elevation.button
                    : themeStore.theme.elevation.input,
                pointerEvents: 'auto',
                zIndex: 10,
              }"
            >
              {{ type.label }}
            </button>
          </div>
        </div>

        <!-- Report Name Input (for new reports) -->
        <div
          v-if="currentYearDocument && selectedYear && selectedReportType"
          class="mb-3"
        >
          <label
            class="text-xs font-semibold mb-1.5 block"
            :style="{ color: themeStore.theme.textSecondary }"
          >
            리포트 이름
          </label>
          <input
            v-model="newReportName"
            type="text"
            :placeholder="`${selectedYear} ${selectedReportType} 평가대응 1차`"
            class="w-full px-3 py-2 text-xs focus:outline-none input-3d"
            :style="{
              backgroundColor: themeStore.isDark
                ? 'rgba(15, 23, 42, 0.6)'
                : themeStore.theme.whiteAlpha[90],
              border: `1px solid ${themeStore.theme.cardBorder}`,
              borderRadius: '0.5rem',
              boxShadow: themeStore.theme.elevation.input,
              color: themeStore.theme.textPrimary,
            }"
          />
        </div>

        <!-- Start Analysis Button with 3D Style -->
        <button
          @click.stop.prevent="handleStartAnalysis($event)"
          @mousedown.stop
          type="button"
          :disabled="
            !currentYearDocument || !selectedYear || !selectedReportType
          "
          class="mb-0 w-full py-2.5 font-semibold text-xs flex items-center justify-center gap-2 start-btn"
          :style="{
            background:
              currentYearDocument && selectedYear && selectedReportType
                ? `linear-gradient(135deg, ${themeStore.theme.primary}, ${themeStore.theme.primaryLight})`
                : 'rgba(128, 128, 128, 0.3)',
            color: 'white',
            borderRadius: '0.75rem',
            boxShadow:
              currentYearDocument && selectedYear && selectedReportType
                ? themeStore.theme.elevation.button
                : 'none',
            cursor:
              currentYearDocument && selectedYear && selectedReportType
                ? 'pointer'
                : 'not-allowed',
            opacity:
              currentYearDocument && selectedYear && selectedReportType
                ? 1
                : 0.5,
            zIndex: 10,
          }"
        >
          <Sparkles :size="14" />
          분석 시작
        </button>
      </div>

      <!-- Bottom Panel: History with 3D Style -->
      <div
        class="flex flex-col flex-1 panel-3d"
        style="min-height: 0"
        :style="{
          backgroundColor: themeStore.isDark
            ? 'rgba(17, 24, 39, 0.95)'
            : themeStore.theme.whiteAlpha[95],
          border: `1px solid ${themeStore.theme.cardBorder}`,
          borderRadius: '1rem',
          boxShadow: themeStore.theme.elevation.card,
          padding: '1.5rem',
        }"
      >
        <!-- Header with 3D Icon -->
        <div class="flex items-center gap-3 mb-3 flex-shrink-0">
          <div
            class="w-8 h-8 rounded-lg flex items-center justify-center"
            :style="{
              backgroundColor: themeStore.theme.primaryAlpha[15],
              boxShadow: themeStore.theme.elevation.icon,
            }"
          >
            <History :size="16" :style="{ color: themeStore.theme.primary }" />
          </div>
          <h4
            class="text-xs font-bold tracking-wide"
            :style="{ color: themeStore.theme.textPrimary }"
          >
            History
          </h4>
        </div>

        <!-- Year Accordion (Scrollable) -->
        <div
          class="flex-1 overflow-y-auto space-y-2 custom-scrollbar"
          style="min-height: 0; margin: -4px; padding: 4px"
        >
          <div
            v-for="year in availableYears"
            :key="year"
            class="accordion-item"
            :style="{
              border: `1px solid ${themeStore.theme.cardBorder}`,
              borderRadius: '0.75rem',
              backgroundColor: themeStore.isDark
                ? 'rgba(15, 23, 42, 0.6)'
                : themeStore.theme.whiteAlpha[80],
              boxShadow: themeStore.theme.elevation.input,
              overflow: 'hidden',
            }"
          >
            <!-- Accordion Header -->
            <button
              @click="toggleYear(year)"
              class="w-full px-3 py-2 flex items-center justify-between hover:opacity-90 transition-all"
              :style="{
                backgroundColor: expandedYears.includes(year)
                  ? themeStore.isDark
                    ? 'rgba(16, 185, 129, 0.1)'
                    : 'rgba(16, 185, 129, 0.05)'
                  : 'transparent',
              }"
            >
              <div class="flex items-center gap-2">
                <span
                  class="text-xs font-bold"
                  :style="{ color: themeStore.theme.textPrimary }"
                >
                  {{ year }}년
                </span>
                <span
                  v-if="getYearReportCount(year) > 0"
                  class="text-xs px-1.5 py-0.5 rounded-full"
                  :style="{
                    backgroundColor: themeStore.theme.primary + '20',
                    color: themeStore.theme.primary,
                  }"
                >
                  {{ getYearReportCount(year) }}개
                </span>
              </div>
              <ChevronDown
                :size="16"
                :style="{
                  color: themeStore.theme.textSecondary,
                  transform: expandedYears.includes(year)
                    ? 'rotate(180deg)'
                    : 'rotate(0deg)',
                  transition: 'transform 0.2s',
                }"
              />
            </button>

            <!-- Accordion Content - Report List -->
            <div
              v-if="expandedYears.includes(year)"
              class="px-3 pb-3 space-y-1.5"
            >
              <div
                v-for="report in getAllReportsForYear(year)"
                :key="report.id"
                @click.stop="selectReportFromHistory(report)"
                class="p-2.5 cursor-pointer report-card"
                :style="{
                  backgroundColor:
                    selectedReportFromHistory?.id === report.id
                      ? themeStore.theme.primaryAlpha[15]
                      : themeStore.isDark
                      ? 'rgba(15, 23, 42, 0.6)'
                      : themeStore.theme.whiteAlpha[90],
                  border: `1px solid ${
                    selectedReportFromHistory?.id === report.id
                      ? themeStore.theme.primary
                      : themeStore.theme.cardBorder
                  }`,
                  borderRadius: '0.5rem',
                  boxShadow:
                    selectedReportFromHistory?.id === report.id
                      ? `0 0 0 2px ${themeStore.theme.primaryAlpha[20]}`
                      : 'none',
                }"
              >
                <div class="flex items-center justify-between">
                  <p
                    class="text-xs font-semibold"
                    :style="{ color: themeStore.theme.textPrimary }"
                  >
                    {{ report.name }}
                  </p>
                  <span
                    class="text-xs px-1.5 py-0.5 rounded font-semibold"
                    :style="{
                      backgroundColor:
                        report.reportType === 'CDP'
                          ? 'rgba(59, 130, 246, 0.2)'
                          : 'rgba(168, 85, 247, 0.2)',
                      color:
                        report.reportType === 'CDP' ? '#3b82f6' : '#a855f7',
                    }"
                  >
                    {{ report.reportType }}
                  </span>
                </div>
                <p
                  class="text-xs mt-1"
                  :style="{ color: themeStore.theme.textSecondary }"
                >
                  {{ formatDate(report.savedAt) }}
                </p>
              </div>

              <!-- Empty State -->
              <div
                v-if="getAllReportsForYear(year).length === 0"
                class="text-center py-4 text-xs"
                :style="{ color: themeStore.theme.textSecondary }"
              >
                저장된 보고서가 없습니다
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Right: Reference Documents List with 3D Style -->
    <div
      class="flex-1 flex flex-col panel-3d"
      :style="{
        backgroundColor: themeStore.isDark
          ? 'rgba(17, 24, 39, 0.95)'
          : themeStore.theme.whiteAlpha[95],
        border: `1px solid ${themeStore.theme.cardBorder}`,
        borderRadius: '1rem',
        boxShadow: themeStore.theme.elevation.card,
        padding: '1.5rem',
      }"
    >
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center gap-3">
          <div
            class="w-8 h-8 rounded-lg flex items-center justify-center"
            :style="{
              backgroundColor: themeStore.theme.primaryAlpha[15],
              boxShadow: themeStore.theme.elevation.icon,
            }"
          >
            <FolderOpen
              :size="16"
              :style="{ color: themeStore.theme.primary }"
            />
          </div>
          <h3
            class="text-sm font-bold tracking-wide"
            :style="{ color: themeStore.theme.textPrimary }"
          >
            참고 문서
          </h3>
        </div>
        <button
          @click="triggerReferenceUpload"
          class="px-3 py-1.5 text-xs font-semibold flex items-center gap-1.5 add-btn"
          :style="{
            backgroundColor: themeStore.theme.primaryAlpha[10],
            color: themeStore.theme.primary,
            border: `1px solid ${themeStore.theme.cardBorder}`,
            borderRadius: '0.5rem',
            boxShadow: themeStore.theme.elevation.input,
          }"
        >
          <Plus :size="14" />
          추가
        </button>
      </div>

      <div
        class="flex-1 overflow-y-auto space-y-2 custom-scrollbar"
        style="margin: -4px; padding: 4px"
      >
        <div
          v-for="(doc, idx) in referenceDocuments"
          :key="idx"
          class="p-3 flex items-center gap-3 ref-doc-card"
          :style="{
            backgroundColor: themeStore.isDark
              ? 'rgba(15, 23, 42, 0.6)'
              : themeStore.theme.whiteAlpha[90],
            border: `1px solid ${themeStore.theme.cardBorder}`,
            borderRadius: '0.75rem',
            boxShadow: themeStore.theme.elevation.input,
          }"
        >
          <FileText
            :size="20"
            :style="{
              color: themeStore.isDark ? '#94a3b8' : '#64748b',
            }"
          />
          <div class="flex-1 min-w-0">
            <p
              class="text-sm font-semibold truncate"
              :style="{ color: themeStore.theme.textPrimary }"
            >
              {{ doc.name }}
            </p>
            <div class="flex items-center gap-2 mt-1">
              <span
                class="text-xs px-2 py-0.5 rounded-full"
                :style="{
                  backgroundColor: themeStore.isDark
                    ? 'rgba(59, 130, 246, 0.2)'
                    : 'rgba(59, 130, 246, 0.1)',
                  color: themeStore.isDark ? '#60a5fa' : '#2563eb',
                }"
              >
                {{ doc.category }}
              </span>
              <span
                class="text-xs"
                :style="{ color: themeStore.theme.textSecondary }"
              >
                {{ doc.date }}
              </span>
            </div>
          </div>
          <button
            @click.stop="removeReferenceDocument(idx)"
            type="button"
            class="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center transition-all hover:scale-110 hover:bg-red-500/20"
            :style="{
              backgroundColor: themeStore.isDark
                ? 'rgba(239, 68, 68, 0.1)'
                : 'rgba(239, 68, 68, 0.1)',
            }"
          >
            <X
              :size="14"
              :style="{
                color: '#ef4444',
              }"
            />
          </button>
        </div>

        <!-- Empty State -->
        <div
          v-if="referenceDocuments.length === 0"
          class="flex flex-col items-center justify-center py-12 text-center"
        >
          <FileText
            :size="48"
            :style="{
              color: themeStore.isDark ? '#475569' : '#cbd5e1',
            }"
          />
          <p
            class="mt-4 text-sm"
            :style="{ color: themeStore.theme.textSecondary }"
          >
            참고 문서가 없습니다
          </p>
        </div>
      </div>
    </div>

    <!-- Hidden File Input -->
    <input
      ref="fileInputRef"
      type="file"
      accept=".pdf,.docx,.xlsx"
      @change="handleFileSelect"
      style="display: none"
    />
    <input
      ref="referenceInputRef"
      type="file"
      accept=".pdf,.docx,.xlsx"
      multiple
      @change="handleReferenceSelect"
      style="display: none"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed, watch, nextTick } from "vue";
import {
  Upload,
  FileText,
  X,
  Sparkles,
  Plus,
  ChevronDown,
  FileSearch,
  History,
  FolderOpen,
} from "lucide-vue-next";
import { useThemeStore } from "../../stores/themeStore";

const themeStore = useThemeStore();

const emit = defineEmits<{
  filesUploaded: [
    files: {
      mainDocument: { name: string; url: string; size: number };
      referenceDocs: Array<{ name: string; category: string; date: string }>;
    }
  ];
  startAnalysis: [year: string, reportType: string, reportId: string];
}>();

const STORAGE_PREFIX = "cdp_djsi_report_";
const REPORTS_LIST_KEY = "cdp_djsi_reports_list";

// Current upload state
const currentYearDocument = ref<{
  name: string;
  url: string;
  size: number;
} | null>(null);
const selectedYear = ref<string | null>(null);
const selectedReportType = ref<string | null>(null);
const newReportName = ref<string>("");
const selectedReportFromHistory = ref<any | null>(null);

// Debug watch to check reactive updates
watch(
  [selectedYear, selectedReportType, currentYearDocument],
  ([year, type, doc]) => {
    console.log("State updated:", { year, type, hasDoc: !!doc });
  },
  { immediate: true }
);

// Accordion state
const expandedYears = ref<string[]>([]);
const selectedReports = ref<Record<string, string>>({}); // year -> reportType

// Available years (2022-2025)
const availableYears = ["2025", "2024", "2023", "2022"];

// Selected year/type for viewing previous reports

// Report types
const reportTypes = [
  { label: "CDP", value: "CDP" },
  { label: "DJSI", value: "DJSI" },
];

const fileInputRef = ref<HTMLInputElement | null>(null);
const referenceInputRef = ref<HTMLInputElement | null>(null);
const referenceDocuments = ref<
  Array<{ name: string; category: string; date: string }>
>([]);

// Mock reference documents
referenceDocuments.value = [
  {
    name: "2023 지속가능경영보고서.pdf",
    category: "ESG 보고서",
    date: "2023-12-31",
  },
  {
    name: "CDP 응답 가이드라인.docx",
    category: "가이드라인",
    date: "2024-01-15",
  },
  {
    name: "온실가스 배출량 인벤토리.xlsx",
    category: "데이터",
    date: "2024-02-20",
  },
];

// Storage helpers - Multiple reports per year/type
interface Report {
  id: string;
  name: string;
  year: string;
  reportType: string;
  document: { name: string; url: string; size: number };
  savedAt: string;
}

const getAllReports = (): Report[] => {
  const reportsJson = localStorage.getItem(REPORTS_LIST_KEY);
  if (reportsJson) {
    try {
      return JSON.parse(reportsJson);
    } catch (e) {
      return [];
    }
  }
  return [];
};

const saveReportToList = (report: Report) => {
  const reports = getAllReports();
  const existingIndex = reports.findIndex((r) => r.id === report.id);
  if (existingIndex >= 0) {
    reports[existingIndex] = report;
  } else {
    reports.push(report);
  }
  // Sort by savedAt descending
  reports.sort(
    (a, b) => new Date(b.savedAt).getTime() - new Date(a.savedAt).getTime()
  );
  localStorage.setItem(REPORTS_LIST_KEY, JSON.stringify(reports));
};

const getReportById = (id: string): Report | null => {
  const reports = getAllReports();
  return reports.find((r) => r.id === id) || null;
};

const saveReport = (
  year: string,
  reportType: string,
  document: { name: string; url: string; size: number },
  reportName?: string
) => {
  const id = `${year}_${reportType}_${Date.now()}`;
  const name = reportName || `${year} ${reportType} 평가대응`;

  const report: Report = {
    id,
    name,
    year,
    reportType,
    document,
    savedAt: new Date().toISOString(),
  };

  saveReportToList(report);

  // Also save individual report data
  const storageKey = `${STORAGE_PREFIX}${id}`;
  localStorage.setItem(storageKey, JSON.stringify(report));

  return id;
};

const getYearReportCount = (year: string) => {
  return getAllReports().filter((r) => r.year === year).length;
};

const getAllReportsForYear = (year: string): Report[] => {
  return getAllReports().filter((r) => r.year === year);
};

const formatDate = (dateString: string) => {
  try {
    const date = new Date(dateString);
    return date.toLocaleDateString("ko-KR", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
  } catch {
    return dateString;
  }
};

// Accordion functions
const toggleYear = (year: string) => {
  const index = expandedYears.value.indexOf(year);
  if (index > -1) {
    expandedYears.value.splice(index, 1);
  } else {
    expandedYears.value.push(year);
  }
};

const handleYearSelect = (year: string) => {
  console.log("handleYearSelect called with year:", year);
  selectedYear.value = year;
  console.log("Selected year is now:", selectedYear.value);
  // Force reactivity update
  nextTick(() => {
    console.log("After nextTick, selectedYear:", selectedYear.value);
  });
};

const handleReportTypeSelect = (type: string) => {
  console.log("handleReportTypeSelect called with type:", type);
  selectedReportType.value = type;
  console.log("Selected report type is now:", selectedReportType.value);
  // Force reactivity update
  nextTick(() => {
    console.log(
      "After nextTick, selectedReportType:",
      selectedReportType.value
    );
  });
};

const selectReportFromHistory = (report: Report) => {
  console.log("selectReportFromHistory called", report);

  selectedReportFromHistory.value = report;

  // Set current document and selections
  currentYearDocument.value = {
    ...report.document,
    // Ensure URL is valid (if it's "#", use empty string or data URL)
    url: report.document.url === "#" ? "" : report.document.url,
  };
  selectedYear.value = report.year;
  selectedReportType.value = report.reportType;
  newReportName.value = report.name;

  console.log(
    "Report selected, currentYearDocument:",
    currentYearDocument.value
  );
  console.log(
    "Selected year:",
    selectedYear.value,
    "Selected type:",
    selectedReportType.value
  );

  // Emit to parent to prepare for loading
  emit("filesUploaded", {
    mainDocument: currentYearDocument.value,
    referenceDocs: referenceDocuments.value,
  });
};

// File handling
const triggerFileUpload = () => {
  fileInputRef.value?.click();
};

const triggerReferenceUpload = () => {
  referenceInputRef.value?.click();
};

const handleFileSelect = (e: Event) => {
  const target = e.target as HTMLInputElement;
  if (target.files && target.files[0]) {
    const file = target.files[0];
    currentYearDocument.value = {
      name: file.name,
      url: URL.createObjectURL(file),
      size: file.size,
    };
    selectedReportFromHistory.value = null; // Reset history selection when new file uploaded
  }
  // Reset input
  if (target) {
    target.value = "";
  }
};

const handleReferenceSelect = (e: Event) => {
  const target = e.target as HTMLInputElement;
  if (target.files) {
    Array.from(target.files).forEach((file) => {
      referenceDocuments.value.push({
        name: file.name,
        category: "기타",
        date: new Date().toISOString().split("T")[0],
      });
    });
  }
};

const handleDrop = (e: DragEvent) => {
  e.preventDefault();
  if (e.dataTransfer?.files && e.dataTransfer.files[0]) {
    const file = e.dataTransfer.files[0];
    currentYearDocument.value = {
      name: file.name,
      url: URL.createObjectURL(file),
      size: file.size,
    };
    selectedReportFromHistory.value = null; // Reset history selection when new file uploaded
  }
};

const removeCurrentYearDocument = () => {
  if (currentYearDocument.value?.url) {
    URL.revokeObjectURL(currentYearDocument.value.url);
  }
  currentYearDocument.value = null;
};

const removeReferenceDocument = (index: number) => {
  if (index >= 0 && index < referenceDocuments.value.length) {
    referenceDocuments.value.splice(index, 1);
  }
};

const handleStartAnalysis = (e?: Event) => {
  if (e) {
    e.preventDefault();
    e.stopPropagation();
  }

  console.log("handleStartAnalysis called", {
    selectedYear: selectedYear.value,
    selectedReportType: selectedReportType.value,
    currentYearDocument: currentYearDocument.value,
    selectedReportFromHistory: selectedReportFromHistory.value,
  });

  // Check if disabled
  if (
    !selectedYear.value ||
    !selectedReportType.value ||
    !currentYearDocument.value
  ) {
    console.error("Button is disabled, missing required values:", {
      selectedYear: selectedYear.value,
      selectedReportType: selectedReportType.value,
      currentYearDocument: currentYearDocument.value,
    });
    return;
  }

  let reportId: string;

  if (selectedReportFromHistory.value) {
    // Using existing report
    reportId = selectedReportFromHistory.value.id;
  } else {
    // Save new report
    const reportName =
      newReportName.value ||
      `${selectedYear.value} ${selectedReportType.value} 평가대응`;
    reportId = saveReport(
      selectedYear.value,
      selectedReportType.value,
      currentYearDocument.value,
      reportName
    );
  }

  console.log("Emitting events with reportId:", reportId);

  emit("filesUploaded", {
    mainDocument: currentYearDocument.value,
    referenceDocs: referenceDocuments.value,
  });

  emit("startAnalysis", selectedYear.value, selectedReportType.value, reportId);
};

// Initialize example reports
const initializeExampleReports = () => {
  const existingReports = getAllReports();
  if (existingReports.length > 0) {
    return; // Don't initialize if reports already exist
  }

  const exampleReports: Report[] = [
    // 2025년
    {
      id: "2025_CDP_1",
      name: "2025 CDP 평가대응 1차",
      year: "2025",
      reportType: "CDP",
      document: {
        name: "2025_CDP_Response_Draft.pdf",
        url: "#",
        size: 2456789,
      },
      savedAt: new Date("2025-01-15").toISOString(),
    },
    {
      id: "2025_DJSI_1",
      name: "2025 DJSI 평가대응",
      year: "2025",
      reportType: "DJSI",
      document: {
        name: "2025_DJSI_Questionnaire.pdf",
        url: "#",
        size: 1890456,
      },
      savedAt: new Date("2025-02-01").toISOString(),
    },
    // 2024년
    {
      id: "2024_CDP_1",
      name: "2024 CDP 평가대응 1차",
      year: "2024",
      reportType: "CDP",
      document: {
        name: "2024_CDP_Climate_Change_Response.pdf",
        url: "#",
        size: 3124567,
      },
      savedAt: new Date("2024-07-15").toISOString(),
    },
    {
      id: "2024_CDP_2",
      name: "2024 CDP 평가대응 2차",
      year: "2024",
      reportType: "CDP",
      document: {
        name: "2024_CDP_Climate_Change_Response_Revised.pdf",
        url: "#",
        size: 3289123,
      },
      savedAt: new Date("2024-08-20").toISOString(),
    },
    {
      id: "2024_DJSI_1",
      name: "2024 DJSI 평가대응",
      year: "2024",
      reportType: "DJSI",
      document: {
        name: "2024_DJSI_Corporate_Sustainability_Assessment.pdf",
        url: "#",
        size: 2567890,
      },
      savedAt: new Date("2024-09-10").toISOString(),
    },
    // 2023년
    {
      id: "2023_CDP_1",
      name: "2023 CDP 평가대응",
      year: "2023",
      reportType: "CDP",
      document: {
        name: "2023_CDP_Disclosure_Response.pdf",
        url: "#",
        size: 2876543,
      },
      savedAt: new Date("2023-06-30").toISOString(),
    },
    {
      id: "2023_DJSI_1",
      name: "2023 DJSI 평가대응 1차",
      year: "2023",
      reportType: "DJSI",
      document: {
        name: "2023_DJSI_Questionnaire_Response.pdf",
        url: "#",
        size: 2234567,
      },
      savedAt: new Date("2023-08-15").toISOString(),
    },
    {
      id: "2023_DJSI_2",
      name: "2023 DJSI 평가대응 2차",
      year: "2023",
      reportType: "DJSI",
      document: {
        name: "2023_DJSI_Questionnaire_Response_Final.pdf",
        url: "#",
        size: 2345678,
      },
      savedAt: new Date("2023-09-25").toISOString(),
    },
    // 2022년
    {
      id: "2022_CDP_1",
      name: "2022 CDP 평가대응",
      year: "2022",
      reportType: "CDP",
      document: {
        name: "2022_CDP_Climate_Change_Questionnaire.pdf",
        url: "#",
        size: 2456789,
      },
      savedAt: new Date("2022-07-20").toISOString(),
    },
    {
      id: "2022_DJSI_1",
      name: "2022 DJSI 평가대응",
      year: "2022",
      reportType: "DJSI",
      document: {
        name: "2022_DJSI_Sustainability_Assessment.pdf",
        url: "#",
        size: 1987654,
      },
      savedAt: new Date("2022-08-30").toISOString(),
    },
  ];

  // Save all example reports
  localStorage.setItem(REPORTS_LIST_KEY, JSON.stringify(exampleReports));

  // Save individual report data for each
  exampleReports.forEach((report) => {
    const storageKey = `${STORAGE_PREFIX}${report.id}`;
    localStorage.setItem(storageKey, JSON.stringify(report));
  });
};

// Load saved reports on mount
onMounted(() => {
  // Initialize example reports if none exist
  initializeExampleReports();

  // Load saved selections
  const savedSelections = localStorage.getItem("cdp_djsi_selected_reports");
  if (savedSelections) {
    try {
      selectedReports.value = JSON.parse(savedSelections);
    } catch (e) {
      console.error("Failed to load saved selections", e);
    }
  }
});

// Save selections when changed
watch(
  selectedReports,
  () => {
    localStorage.setItem(
      "cdp_djsi_selected_reports",
      JSON.stringify(selectedReports.value)
    );
  },
  { deep: true }
);

// Auto-generate report name when year/type are selected
watch([selectedYear, selectedReportType], () => {
  if (
    selectedYear.value &&
    selectedReportType.value &&
    !selectedReportFromHistory.value &&
    !newReportName.value
  ) {
    const existingCount = getAllReportsForYear(selectedYear.value).filter(
      (r) => r.reportType === selectedReportType.value
    ).length;
    const suffix = existingCount > 0 ? ` ${existingCount + 1}차` : " 1차";
    newReportName.value = `${selectedYear.value} ${selectedReportType.value} 평가대응${suffix}`;
  }
});

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
};
</script>

<style scoped>
/* Panel 3D Effect */
.panel-3d {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  overflow: hidden;
}

/* Upload Card Hover */
.upload-card {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.upload-card:hover {
  border-style: solid !important;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
}

/* Document Card Hover */
.doc-card {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.doc-card:hover {
  transform: translateY(-1px);
}

/* Year/Type Button Hover */
.year-btn,
.type-btn {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.year-btn:hover,
.type-btn:hover {
  transform: translateY(-1px);
}

.year-btn:active,
.type-btn:active {
  transform: translateY(0) scale(0.98);
}

/* Input 3D Focus */
.input-3d {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.input-3d:focus {
  transform: translateY(-1px);
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15),
    inset 0 1px 2px rgba(0, 0, 0, 0.04);
}

/* Start Button Hover */
.start-btn {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.start-btn:not(:disabled):hover {
  transform: translateY(-2px) scale(1.02);
  filter: brightness(1.1);
}

.start-btn:not(:disabled):active {
  transform: translateY(0) scale(0.98);
}

/* Accordion Item Hover */
.accordion-item {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.accordion-item:hover {
  transform: translateY(-1px);
}

/* Report Card Hover */
.report-card {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.report-card:hover {
  transform: translateX(2px);
}

/* Add Button Hover */
.add-btn {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.add-btn:hover {
  transform: translateY(-1px) scale(1.02);
}

.add-btn:active {
  transform: translateY(0) scale(0.98);
}

/* Reference Document Card Hover */
.ref-doc-card {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.ref-doc-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
}
</style>
