<template>
  <div
    class="relative min-h-screen"
    :class="themeStore.theme.textColor"
    :style="{
      overflow: route.name === 'landing' ? 'auto' : 'hidden',
      backgroundColor: themeStore.theme.background,
    }"
  >
    <!-- Pattern Overlay (hide on LandingPage) -->
    <div
      v-if="!themeStore.isDark && route.name !== 'landing'"
      class="absolute inset-0 pointer-events-none"
      :style="{
        backgroundImage: (themeStore.theme.overlay as any).backgroundImage,
        backgroundSize: (themeStore.theme.overlay as any).backgroundSize,
        opacity: themeStore.theme.overlay.opacity,
      }"
    />
    <div
      v-else-if="themeStore.isDark && route.name !== 'landing'"
      class="absolute inset-0 pointer-events-none"
      :style="{
        background: (themeStore.theme.overlay as any).background,
        opacity: themeStore.theme.overlay.opacity,
      }"
    />

    <!-- Router View -->
    <router-view />
  </div>
</template>

<script setup lang="ts">
import { useRoute } from "vue-router";
import { useThemeStore } from "./stores/themeStore";

const route = useRoute();
const themeStore = useThemeStore();
</script>
