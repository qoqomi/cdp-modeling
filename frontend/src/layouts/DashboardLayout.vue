<template>
  <div class="relative z-10 h-screen flex flex-col" style="box-shadow: none">
    <!-- Main Layout: Sidebar + Content -->
    <div class="flex-1 flex overflow-hidden p-4 gap-4" style="box-shadow: none">
      <!-- Left Sidebar Navigation - 3D Rounded Card Style -->
      <aside
        class="sidebar flex-shrink-0 flex flex-col"
        :class="{ collapsed: isCollapsed, dark: themeStore.isDark }"
        :style="{
          backgroundColor: themeStore.isDark
            ? 'rgba(17, 17, 17, 0.98)'
            : 'rgba(255, 255, 255, 0.98)',
          borderColor: themeStore.theme.border3d.dark,
          boxShadow: themeStore.theme.elevation.card,
        }"
      >
        <!-- Logo/Title Section -->
        <div
          class="sidebar-header"
          :style="{ borderColor: themeStore.theme.cardBorder }"
        >
          <div class="logo-wrapper" @click="handleBackToIntro">
            <div
              class="logo-icon"
              :style="{ backgroundColor: `${themeStore.theme.primary}15` }"
            >
              <Earth :size="22" :style="{ color: themeStore.theme.primary }" />
            </div>
            <span
              v-if="!isCollapsed"
              class="logo-text"
              :style="{ color: themeStore.theme.textPrimary, border: 'none' }"
            >
              CLIMA-X
            </span>
          </div>
          <!-- Toggle Button -->
          <button
            class="toggle-btn"
            :style="{
              backgroundColor: themeStore.theme.primary,
            }"
            @click="isCollapsed = !isCollapsed"
          >
            <ChevronLeft
              :size="16"
              :class="{ 'rotate-180': isCollapsed }"
              class="toggle-icon"
            />
          </button>
        </div>

        <!-- Navigation Menu -->
        <nav class="nav-menu">
          <router-link
            v-for="tab in tabs"
            :key="tab.id"
            :to="`/dashboard/${tab.id}`"
            class="nav-item"
            :class="{ active: route.name === tab.id }"
            :style="
              route.name === tab.id
                ? {
                    backgroundColor: `${themeStore.theme.primary}20`,
                    borderRadius: '14px',
                  }
                : {}
            "
          >
            <!-- Icon -->
            <div
              class="nav-icon-standalone"
              :class="{ 'active-icon': route.name === tab.id }"
              :style="{
                color:
                  route.name === tab.id
                    ? themeStore.theme.primary
                    : themeStore.theme.textSecondary,
              }"
            >
              <component :is="tab.icon" :size="20" />
            </div>
            <!-- Label (no pill for inactive, part of active pill) -->
            <span
              v-if="!isCollapsed"
              class="nav-label"
              :style="{
                color:
                  route.name === tab.id
                    ? themeStore.theme.primary
                    : themeStore.theme.textSecondary,
              }"
            >
              {{ tab.label }}
            </span>
          </router-link>
        </nav>

        <!-- Sidebar Footer - User Info & Theme Toggle -->
        <div
          class="sidebar-footer"
          :style="{ borderColor: themeStore.theme.cardBorder }"
        >
          <!-- 토큰 정보 -->
          <div
            class="token-info"
            :style="{
              backgroundColor: themeStore.isDark
                ? 'rgba(251, 191, 36, 0.1)'
                : 'rgba(251, 191, 36, 0.08)',
            }"
            :title="`남은 토큰: ${remainingTokens ?? '-'}`"
          >
            <Coins
              :size="isCollapsed ? 20 : 16"
              :style="{ color: '#fbbf24' }"
            />
            <span
              v-if="!isCollapsed"
              class="token-count"
              :style="{ color: themeStore.theme.textPrimary }"
            >
              {{ formattedTokens }}
            </span>
          </div>

          <!-- 사용자 정보 -->
          <div
            v-if="currentUser"
            class="user-info"
            :style="{
              backgroundColor: themeStore.isDark
                ? 'rgba(255, 255, 255, 0.05)'
                : 'rgba(0, 0, 0, 0.02)',
            }"
          >
            <div
              class="user-avatar"
              :style="{
                backgroundColor: `${themeStore.theme.primary}20`,
                color: themeStore.theme.primary,
              }"
            >
              {{ userInitial }}
            </div>
            <div v-if="!isCollapsed" class="user-details">
              <span
                class="user-name"
                :style="{ color: themeStore.theme.textPrimary }"
              >
                {{ currentUser.name }}
              </span>
              <span
                class="user-company"
                :style="{ color: themeStore.theme.textSecondary }"
              >
                {{ currentUser.company }}
              </span>
            </div>
            <button
              v-if="!isCollapsed"
              class="logout-btn"
              :style="{ color: themeStore.theme.textSecondary }"
              title="로그아웃"
              @click="handleLogout"
            >
              <LogOut :size="16" />
            </button>
          </div>

          <button
            class="theme-toggle-btn"
            :style="{
              backgroundColor: themeStore.isDark
                ? 'rgba(255, 255, 255, 0.08)'
                : 'rgba(0, 0, 0, 0.03)',
            }"
            :title="
              themeStore.isDark ? '밝은 테마로 전환' : '어두운 테마로 전환'
            "
            @click="themeStore.toggleTheme()"
          >
            <Sun
              v-if="themeStore.isDark"
              :size="18"
              :style="{ color: '#fbbf24' }"
            />
            <Moon
              v-else
              :size="18"
              :style="{ color: themeStore.theme.textSecondary }"
            />
            <span
              v-if="!isCollapsed"
              class="theme-label"
              :style="{ color: themeStore.theme.textSecondary }"
            >
              {{ themeStore.isDark ? "라이트 모드" : "다크 모드" }}
            </span>
          </button>
        </div>
      </aside>

      <!-- Main Content Area -->
      <main class="flex-1 flex overflow-visible" style="box-shadow: none">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import { useRoute, useRouter } from "vue-router";
import {
  MessagesSquare,
  BarChart3,
  FileCheck,
  FileSearch,
  ChevronLeft,
  Sun,
  Moon,
  Earth,
  LogOut,
  Coins,
} from "lucide-vue-next";
import { useThemeStore } from "../stores/themeStore";
import { apiClient } from "../api/client";

interface UserInfo {
  id: number;
  email: string;
  name: string;
  company: string;
  role: string;
}

const route = useRoute();
const router = useRouter();
const themeStore = useThemeStore();
const isCollapsed = ref(true);

// 사용자 정보
const currentUser = ref<UserInfo | null>(null);

// 토큰 정보
const remainingTokens = ref<number | null>(null);

// 사용자 이니셜 (이름의 첫 글자)
const userInitial = computed(() => {
  if (currentUser.value?.name) {
    return currentUser.value.name.charAt(0).toUpperCase();
  }
  return "U";
});

// 토큰 포맷팅 (1000 이상이면 K 단위로 표시)
const formattedTokens = computed(() => {
  if (remainingTokens.value === null) return "-";
  if (remainingTokens.value >= 1000) {
    return (remainingTokens.value / 1000).toFixed(1) + "K";
  }
  return remainingTokens.value.toString();
});

// localStorage에서 사용자 정보 로드
const loadUserInfo = () => {
  const userStr = localStorage.getItem("user");
  if (userStr) {
    try {
      currentUser.value = JSON.parse(userStr);
    } catch (e) {
      currentUser.value = null;
    }
  }
};

// API에서 토큰 정보 로드
const loadTokenInfo = async () => {
  try {
    const token = localStorage.getItem("accessToken");
    if (!token) return;

    const response = await apiClient.get("/v1/company/tokens", {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    remainingTokens.value = response.data.remainingTokens;
  } catch (e) {
    console.error("토큰 정보 로드 실패:", e);
    remainingTokens.value = null;
  }
};

// 로그아웃 처리
const handleLogout = () => {
  localStorage.removeItem("accessToken");
  localStorage.removeItem("refreshToken");
  localStorage.removeItem("user");
  currentUser.value = null;
  router.push("/");
};

// 컴포넌트 마운트 시 사용자 정보 및 토큰 정보 로드
onMounted(() => {
  loadUserInfo();
  loadTokenInfo();
});

const tabs = [
  { id: "risk" as const, label: "TCFD 분석 HUB", icon: BarChart3, badge: null },
  { id: "tnfd" as const, label: "TNFD 에이전트", icon: FileCheck, badge: null },
  {
    id: "proof" as const,
    label: "외부 ESG 평가 대응",
    icon: FileSearch,
    badge: null,
  },
  {
    id: "copilot" as const,
    label: "ESG AI 어시스턴트",
    icon: MessagesSquare,
    badge: null,
  },
];

const handleBackToIntro = () => {
  router.push("/");
};
</script>

<style scoped>
/* Sidebar Base - 3D Rounded Card Style */
.sidebar {
  width: 240px;
  border: 1px solid;
  border-radius: 1.5rem;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  flex-direction: column;
  backdrop-filter: blur(20px);
  overflow: hidden;
  transform: translateZ(0);
  will-change: transform, box-shadow;
}

.sidebar:hover {
  transform: translateY(-2px) translateZ(0);
}

.sidebar.collapsed {
  width: 72px;
}

/* Main Content Area - No shadow */

/* Sidebar Header */
.sidebar-header {
  padding: 1rem;
  border-bottom: 1px solid;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.logo-wrapper {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex: 1;
  min-width: 0;
  cursor: pointer;
  transition: opacity 0.2s;
}

.logo-icon {
  width: 40px;
  height: 40px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
  transition: all 0.2s;
}

.logo-wrapper:hover .logo-icon {
  transform: scale(1.05);
  box-shadow: 0 6px 16px rgba(16, 185, 129, 0.3),
    inset 0 1px 0 rgba(255, 255, 255, 0.15);
}

.logo-text {
  font-size: 1.25rem;
  font-weight: 700;
  font-style: italic;
  white-space: nowrap;
  letter-spacing: -0.02em;
  font-family: "Georgia", "Times New Roman", serif;
}

/* Toggle Button - 3D Style */
.toggle-btn {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
  color: white;
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4),
    inset 0 1px 0 rgba(255, 255, 255, 0.2);
}

.toggle-btn:hover {
  transform: scale(1.08) translateY(-1px);
  box-shadow: 0 6px 16px rgba(16, 185, 129, 0.5),
    inset 0 1px 0 rgba(255, 255, 255, 0.25);
}

.toggle-btn:active {
  transform: scale(0.98);
  box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
}

.toggle-icon {
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.toggle-icon.rotate-180 {
  transform: rotate(180deg);
}

/* Navigation Menu */
.nav-menu {
  flex: 1;
  padding: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  overflow-y: auto;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.625rem 0.75rem;
  text-decoration: none;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
}

.nav-item:hover {
  background-color: rgba(16, 185, 129, 0.08) !important;
  border-radius: 14px;
}

.nav-item:hover .nav-icon-standalone {
  color: #10b981 !important;
}

.nav-item:hover .nav-label {
  color: #10b981 !important;
}

/* Icon Style */
.nav-icon-standalone {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
  border-radius: 8px;
}

/* Active icon gets subtle background */
.nav-icon-standalone.active-icon {
  background: transparent;
}

/* Label Style - Simple text, no pill */
.nav-label {
  flex: 1;
  font-size: 0.9rem;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  transition: all 0.2s;
}

/* Collapsed State Adjustments */
.sidebar.collapsed .sidebar-header {
  padding: 0.75rem;
  flex-direction: column;
  gap: 0.5rem;
}

.sidebar.collapsed .logo-wrapper {
  justify-content: center;
}

.sidebar.collapsed .toggle-btn {
  width: 32px;
  height: 32px;
  border-radius: 50%;
}

.sidebar.collapsed .nav-item {
  justify-content: center;
  padding: 0.625rem;
}

.sidebar.collapsed .nav-item.active {
  background-color: rgba(16, 185, 129, 0.15) !important;
  border-radius: 12px;
}

.sidebar.collapsed .nav-icon-standalone {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: transparent;
  transition: all 0.2s;
}

.sidebar.collapsed .nav-item:hover {
  background-color: rgba(16, 185, 129, 0.1) !important;
  border-radius: 12px;
}

.sidebar.collapsed .nav-item:hover .nav-icon-standalone {
  transform: scale(1.05);
}

.sidebar.collapsed .nav-item.active .nav-icon-standalone {
  background: transparent;
}

/* Sidebar Footer */
.sidebar-footer {
  padding: 0.75rem;
  border-top: 1px solid;
}

.theme-toggle-btn {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.625rem;
  padding: 0.625rem 0.75rem;
  border: none;
  border-radius: 0.875rem;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05),
    0 1px 0 rgba(255, 255, 255, 0.5);
}

.sidebar.dark .theme-toggle-btn {
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.3),
    0 1px 0 rgba(255, 255, 255, 0.05);
}

.theme-toggle-btn:hover {
  background: rgba(16, 185, 129, 0.1) !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.theme-toggle-btn:active {
  transform: translateY(0);
}

.theme-label {
  font-size: 0.8125rem;
  font-weight: 500;
  white-space: nowrap;
}

.sidebar.collapsed .theme-toggle-btn {
  padding: 0.625rem;
}

/* User Info Styles */
.user-info {
  display: flex;
  align-items: center;
  gap: 0.625rem;
  padding: 0.625rem;
  border-radius: 0.875rem;
  margin-bottom: 0.5rem;
}

.user-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.875rem;
  font-weight: 600;
  flex-shrink: 0;
}

.user-details {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
}

.user-name {
  font-size: 0.8125rem;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.user-company {
  font-size: 0.6875rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.logout-btn {
  padding: 0.375rem;
  border: none;
  background: transparent;
  cursor: pointer;
  border-radius: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
}

.logout-btn:hover {
  background: rgba(239, 68, 68, 0.1);
  color: #ef4444 !important;
}

.sidebar.collapsed .user-info {
  justify-content: center;
  padding: 0.5rem;
}

.sidebar.collapsed .user-avatar {
  width: 36px;
  height: 36px;
}

/* Token Info Styles */
.token-info {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.5rem 0.625rem;
  border-radius: 0.75rem;
  margin-bottom: 0.5rem;
}

.token-count {
  font-size: 0.875rem;
  font-weight: 700;
}

.sidebar.collapsed .token-info {
  padding: 0.5rem;
}
</style>
