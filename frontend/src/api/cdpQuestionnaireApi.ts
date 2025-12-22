import type { CDPQuestionnaireModule } from "./types/cdpQuestionnaire.types";

export async function loadLocalQuestionnaireModule2(): Promise<CDPQuestionnaireModule> {
  const response = await fetch("/data/cdp_questions_merged.json");
  if (!response.ok) {
    throw new Error(
      `Failed to load questionnaire: ${response.status} ${response.statusText}`
    );
  }
  return (await response.json()) as CDPQuestionnaireModule;
}

