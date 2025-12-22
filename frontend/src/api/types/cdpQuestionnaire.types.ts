export interface CDPQuestionnaireTag {
  authority_type?: string;
  environmental_issue?: string;
  sector?: string;
  question_level?: string;
}

export interface CDPQuestionnaireResponseColumn {
  id: string;
  field: string;
  type: string;
  options?: string[];
  grouped_options?: Record<string, string[]>;
  max_length?: number;
  min_value?: number;
  max_value?: number;
  required?: boolean;
  condition?: Record<string, unknown>;
}

export interface CDPQuestionnaireResponseFormat {
  type: string;
  columns?: CDPQuestionnaireResponseColumn[];
  options?: string[];
  max_length?: number;
}

// guidance_raw 구조 (백엔드 merger.py에서 생성)
export interface CDPGuidanceRaw {
  rationale?: string | null;
  rationale_ko?: string | null;
  ambition?: string[] | null;
  ambition_ko?: string[] | null;
  requested_content?: string[] | null;
  requested_content_ko?: string[] | null;
  explanation_of_terms?: Record<string, string> | null;
  additional_information?: string | null;
  additional_information_ko?: string | null;
}

export interface CDPQuestionnaireQuestion {
  question_id: string;
  // cdp_questions_parsed: title_en (영어)
  // cdp_questions_merged: title (영어)
  title?: string;
  title_en?: string;
  title_ko?: string;
  change_from_last_year?: string;
  question_dependencies?: string | null;
  tags?: CDPQuestionnaireTag;
  rationale?: string | null;
  ambition?: string[] | null;
  requested_content?: string[] | null;
  explanation_of_terms?: Record<string, string> | null;
  additional_information?: string | null;
  response_format?: CDPQuestionnaireResponseFormat | null;
  children?: CDPQuestionnaireQuestion[] | null;
  // 한/영 가이드라인 (merger.py --with-translation)
  guidance_raw?: CDPGuidanceRaw | null;
}

export interface CDPQuestionnaireModule {
  version: string;
  module: string;
  title: string;
  questions: CDPQuestionnaireQuestion[];
}

