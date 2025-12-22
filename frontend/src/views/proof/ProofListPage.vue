<template>
  <div
    class="flex gap-4 h-full w-full p-1"
    style="min-height: 0; height: 100%; overflow: visible"
  >
    <ProofUploadPanel
      @files-uploaded="handleFilesUploaded"
      @start-analysis="handleStartAnalysis"
    />
  </div>
</template>

<script setup lang="ts">
import { useRouter } from "vue-router";
import ProofUploadPanel from "../../components/proof/ProofUploadPanel.vue";

const router = useRouter();

const handleFilesUploaded = (files: {
  mainDocument: { name: string; url: string; size: number };
  referenceDocs: Array<{ name: string; category: string; date: string }>;
}) => {
  console.log("Files uploaded:", files);
};

const handleStartAnalysis = (
  year: string,
  reportType: string,
  reportId: string
) => {
  console.log("Starting analysis:", { year, reportType, reportId });

  // Navigate to analysis page with reportId
  router.push({
    name: "proof-analysis",
    params: { reportId },
  });
};
</script>
