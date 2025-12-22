import { defineStore } from "pinia";
import { ref, computed } from "vue";

export type Theme = "dark" | "light";

export const useThemeStore = defineStore("theme", () => {
  const currentTheme = ref<Theme>("light");

  const isDark = computed(() => currentTheme.value === "dark");
  const isLight = computed(() => currentTheme.value === "light");

  const toggleTheme = () => {
    currentTheme.value = currentTheme.value === "dark" ? "light" : "dark";
  };

  const setTheme = (theme: Theme) => {
    currentTheme.value = theme;
  };

  // Primary 색상 Opacity 변형 (rgba(16, 185, 129, opacity))
  const primaryOpacity = {
    5: "rgba(16, 185, 129, 0.05)",
    10: "rgba(16, 185, 129, 0.1)",
    15: "rgba(16, 185, 129, 0.15)",
    20: "rgba(16, 185, 129, 0.2)",
    25: "rgba(16, 185, 129, 0.25)",
    30: "rgba(16, 185, 129, 0.3)",
    40: "rgba(16, 185, 129, 0.4)",
    50: "rgba(16, 185, 129, 0.5)",
    60: "rgba(16, 185, 129, 0.6)",
    70: "rgba(16, 185, 129, 0.7)",
    80: "rgba(16, 185, 129, 0.8)",
    90: "rgba(16, 185, 129, 0.9)",
    100: "rgba(16, 185, 129, 1)",
  };

  // White 색상 Opacity 변형 (rgba(255, 255, 255, opacity))
  const whiteOpacity = {
    5: "rgba(255, 255, 255, 0.05)",
    10: "rgba(255, 255, 255, 0.1)",
    15: "rgba(255, 255, 255, 0.15)",
    20: "rgba(255, 255, 255, 0.2)",
    25: "rgba(255, 255, 255, 0.25)",
    30: "rgba(255, 255, 255, 0.3)",
    40: "rgba(255, 255, 255, 0.4)",
    50: "rgba(255, 255, 255, 0.5)",
    60: "rgba(255, 255, 255, 0.6)",
    70: "rgba(255, 255, 255, 0.7)",
    80: "rgba(255, 255, 255, 0.8)",
    90: "rgba(255, 255, 255, 0.9)",
    95: "rgba(255, 255, 255, 0.95)",
    100: "rgba(255, 255, 255, 1)",
  };

  // 라이트 모드 테마 (흰 배경 + 초록색 포인트)
  const lightTheme = {
    background: "#fafbfc", // 3D 효과를 위한 아주 연한 회색 (거의 흰색)
    patternColor: "#10b981",
    patternOpacity: "0.03",
    textPrimary: "#1f2937", // 진한 그레이
    textSecondary: "#6b7280", // 중간 그레이
    primary: "#10b981", // Emerald-500 (메인 초록)
    primaryLight: "#34d399", // Emerald-400 (밝은 초록)
    primaryDark: "#059669", // Emerald-600 (진한 초록)
    // Primary Opacity 변형
    primaryAlpha: primaryOpacity,
    // White Opacity 변형
    whiteAlpha: whiteOpacity,
    cardBg: "rgba(255, 255, 255, 0.8)",
    cardBgLight: "rgba(255, 255, 255, 0.5)",
    cardBorder: "rgba(16, 185, 129, 0.15)",
    tabActive: "rgba(255, 255, 255, 0.95)",
    tabInactive: "rgba(255, 255, 255, 0.5)",
    tabBorder: "#10b981",
    tabBorderLight: "rgba(16, 185, 129, 0.2)",
    textColor: "text-slate-800",
    overlay: {
      backgroundImage: "none",
      backgroundSize: "0",
      opacity: 0,
    },
    header: {
      backgroundColor: "rgba(255, 255, 255, 1)",
      borderColor: "rgba(16, 185, 129, 0.1)",
      primaryColor: "#10b981",
      secondaryColor: "#059669",
      textColor: "#059669",
      subTextColor: "#6b7280",
    },
    // 3D 스타일
    elevation: {
      // 카드/패널 그림자
      card: "0 20px 60px rgba(0, 0, 0, 0.08), 0 8px 25px rgba(0, 0, 0, 0.04), inset 0 1px 0 rgba(255, 255, 255, 0.8)",
      cardHover:
        "0 25px 70px rgba(0, 0, 0, 0.12), 0 12px 30px rgba(0, 0, 0, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.9)",
      // 버튼 그림자
      button:
        "0 4px 12px rgba(16, 185, 129, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.2)",
      buttonHover:
        "0 6px 16px rgba(16, 185, 129, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.25)",
      buttonActive: "0 2px 8px rgba(16, 185, 129, 0.15)",
      // 입력 필드 그림자
      input:
        "inset 0 1px 2px rgba(0, 0, 0, 0.04), 0 1px 0 rgba(255, 255, 255, 0.5)",
      inputFocus:
        "0 0 0 3px rgba(16, 185, 129, 0.15), inset 0 1px 2px rgba(0, 0, 0, 0.04)",
      // 아이콘 그림자
      icon: "0 4px 12px rgba(16, 185, 129, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1)",
      iconHover:
        "0 6px 16px rgba(16, 185, 129, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.15)",
      // 드롭다운/팝업 그림자
      dropdown:
        "0 10px 40px rgba(0, 0, 0, 0.12), 0 4px 15px rgba(0, 0, 0, 0.06)",
      // 모달 그림자
      modal: "0 25px 80px rgba(0, 0, 0, 0.15), 0 10px 30px rgba(0, 0, 0, 0.08)",
    },
    // 3D 보더
    border3d: {
      light: "rgba(255, 255, 255, 0.8)",
      dark: "rgba(0, 0, 0, 0.06)",
    },
    // 3D 호버 효과
    hover: {
      card: "translateY(-2px)",
      cardShadow:
        "0 12px 24px rgba(0, 0, 0, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.1)",
      button: "translateY(-1px) scale(1.02)",
      buttonShadow:
        "0 6px 16px rgba(16, 185, 129, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2)",
      buttonActive: "translateY(0) scale(0.98)",
      input: "translateY(-1px)",
    },
    // 트랜지션
    transition: {
      default: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
      fast: "all 0.15s cubic-bezier(0.4, 0, 0.2, 1)",
      slow: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    },
  };

  // 다크 모드 테마 (검정 배경 + 네온 그린 포인트)
  const darkTheme = {
    background: "#000000",
    patternColor: "#10b981",
    patternOpacity: "0.05",
    textPrimary: "#e5e7eb", // 밝은 그레이
    textSecondary: "#9ca3af", // 중간 그레이
    primary: "#10b981", // Emerald-500 (네온 그린)
    primaryLight: "#34d399", // Emerald-400 (밝은 네온)
    primaryDark: "#059669", // Emerald-600 (진한 그린)
    // Primary Opacity 변형
    primaryAlpha: primaryOpacity,
    // White Opacity 변형
    whiteAlpha: whiteOpacity,
    cardBg: "rgba(17, 17, 17, 0.8)",
    cardBgLight: "rgba(26, 26, 26, 0.6)",
    cardBorder: "rgba(16, 185, 129, 0.2)",
    tabActive: "rgba(17, 17, 17, 0.95)",
    tabInactive: "rgba(17, 17, 17, 0.4)",
    tabBorder: "#10b981",
    tabBorderLight: "rgba(16, 185, 129, 0.3)",
    textColor: "text-slate-100",
    overlay: {
      background: "none",
      opacity: 0,
    },
    header: {
      backgroundColor: "rgba(0, 0, 0, 0.8)",
      borderColor: "rgba(16, 185, 129, 0.2)",
      primaryColor: "#10b981",
      secondaryColor: "#34d399",
      textColor: "#10b981",
      subTextColor: "#9ca3af",
    },
    // 3D 스타일
    elevation: {
      // 카드/패널 그림자
      card: "0 20px 60px rgba(0, 0, 0, 0.5), 0 8px 25px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05)",
      cardHover:
        "0 25px 70px rgba(0, 0, 0, 0.6), 0 12px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.08)",
      // 버튼 그림자
      button:
        "0 4px 12px rgba(16, 185, 129, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.15)",
      buttonHover:
        "0 6px 16px rgba(16, 185, 129, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.2)",
      buttonActive: "0 2px 8px rgba(16, 185, 129, 0.3)",
      // 입력 필드 그림자
      input:
        "inset 0 1px 2px rgba(0, 0, 0, 0.2), 0 1px 0 rgba(255, 255, 255, 0.05)",
      inputFocus:
        "0 0 0 3px rgba(16, 185, 129, 0.25), inset 0 1px 2px rgba(0, 0, 0, 0.2)",
      // 아이콘 그림자
      icon: "0 4px 12px rgba(16, 185, 129, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.08)",
      iconHover:
        "0 6px 16px rgba(16, 185, 129, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.12)",
      // 드롭다운/팝업 그림자
      dropdown: "0 10px 40px rgba(0, 0, 0, 0.5), 0 4px 15px rgba(0, 0, 0, 0.3)",
      // 모달 그림자
      modal: "0 25px 80px rgba(0, 0, 0, 0.6), 0 10px 30px rgba(0, 0, 0, 0.4)",
    },
    // 3D 보더
    border3d: {
      light: "rgba(255, 255, 255, 0.1)",
      dark: "rgba(0, 0, 0, 0.3)",
    },
    // 3D 호버 효과
    hover: {
      card: "translateY(-2px)",
      cardShadow:
        "0 12px 24px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.08)",
      button: "translateY(-1px) scale(1.02)",
      buttonShadow:
        "0 6px 16px rgba(16, 185, 129, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.15)",
      buttonActive: "translateY(0) scale(0.98)",
      input: "translateY(-1px)",
    },
    // 트랜지션
    transition: {
      default: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
      fast: "all 0.15s cubic-bezier(0.4, 0, 0.2, 1)",
      slow: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    },
  };

  const theme = computed(() => {
    return currentTheme.value === "dark" ? darkTheme : lightTheme;
  });

  // 하위 호환성을 위한 currentStyles
  const currentStyles = computed(() => {
    const t = theme.value;
    return {
      background: t.background,
      textColor: t.textColor,
      overlay: t.overlay,
      header: t.header,
    };
  });

  return {
    currentTheme,
    isDark,
    isLight,
    toggleTheme,
    setTheme,
    theme,
    currentStyles,
    lightTheme,
    darkTheme,
    primaryOpacity,
    whiteOpacity,
  };
});
