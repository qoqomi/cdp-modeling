/**
 * CDP Answer Generator API
 * Backend의 cdp_answer_generator.py와 통신
 */

import { apiClient } from "./client";
import type {
  CDPBackendAnswer,
  CDPBackendResponse,
  CDPQuestionSchema,
  GeneratedAnswer,
  AnswerContent,
  AnswerRow,
} from "./types/cdpAnswer.types";

/**
 * Backend 응답을 Frontend 형식으로 변환
 * 새 형식: rows[].columns에 _en/_ko suffix, rationale_en/ko, overall_confidence
 */
export function transformBackendAnswer(
  backendAnswer: CDPBackendAnswer,
  schema?: CDPQuestionSchema
): GeneratedAnswer {
  const content: AnswerContent = {
    rows: [],
    evidence: [],
  };

  // 신뢰도 (새 형식: overall_confidence, 레거시: confidence)
  const confidence = backendAnswer.overall_confidence ?? backendAnswer.confidence ?? 0;

  // 새 형식: rows 배열 처리
  // 각 컬럼을 개별 AnswerRow로 변환 (컬럼별 타입 유지)
  if (backendAnswer.rows?.length) {
    const transformedRows: AnswerRow[] = [];

    for (const row of backendAnswer.rows) {
      const columns = row.columns;

      // 각 컬럼을 개별 AnswerRow로 변환
      const processedKeys = new Set<string>();

      // 컬럼 레이블 매핑 (영문 key -> 한글 표시명)
      const columnLabels: Record<string, { ko: string; en: string }> = {
        time_horizon: { ko: "시간 범위", en: "Time Horizon" },
        from_years: { ko: "시작 연도", en: "From Years" },
        to_years: { ko: "종료 연도", en: "To Years" },
        process_in_place: { ko: "프로세스 유무", en: "Process in Place" },
        deps_impacts_evaluated: { ko: "의존성/영향 평가", en: "Dependencies/Impacts Evaluated" },
        risks_opps_evaluated: { ko: "위험/기회 평가", en: "Risks/Opportunities Evaluated" },
        informed_by_deps_impacts: { ko: "의존성/영향 기반", en: "Informed by Dependencies/Impacts" },
        explain_gaps_or_not_informed: { ko: "답변 내용", en: "Answer Content" },
        rationale: { ko: "답변 내용", en: "Answer Content" },
        environmental_issue: { ko: "환경 이슈", en: "Environmental Issue" },
      };

      for (const [key, value] of Object.entries(columns)) {
        if (processedKeys.has(key)) continue;

        // _en/_ko suffix가 있는 textarea 필드 처리
        if (key.endsWith("_en")) {
          const baseKey = key.replace(/_en$/, "");
          const enValue = String(value ?? "");
          const koValue = String(columns[`${baseKey}_ko`] ?? "");
          processedKeys.add(key);
          processedKeys.add(`${baseKey}_ko`);

          // 컬럼명 가져오기
          const label = columnLabels[baseKey] || { ko: baseKey, en: baseKey };

          transformedRows.push({
            number: "",  // 번호 제거
            detail: label.ko,
            detailEn: label.en,
            answer: enValue || koValue,
            answerKo: koValue,
            answerEn: enValue,
            type: "textarea",
          });
        } else if (key.endsWith("_ko")) {
          // _en과 함께 처리되었을 것이므로 스킵
          continue;
        } else {
          // 일반 필드: 타입 추론
          processedKeys.add(key);
          const strValue = String(value ?? "");

          // 타입 결정
          let fieldType = "text";
          if (key === "time_horizon" || key === "row_type" || key === "process_in_place" || key === "deps_impacts_evaluated" || key === "risks_opps_evaluated" || key === "informed_by_deps_impacts" || key === "environmental_issue") {
            fieldType = "select";
          } else if (key === "from_years" || key === "to_years" || typeof value === "number") {
            fieldType = "number";
          }

          // 컬럼명 가져오기
          const label = columnLabels[key] || { ko: key, en: key };

          transformedRows.push({
            number: "",  // 번호 제거
            detail: label.ko,
            detailEn: label.en,
            answer: strValue,
            answerKo: strValue,
            answerEn: strValue,
            type: fieldType,
          });
        }
      }
    }

    content.rows = transformedRows;

    // textarea 타입의 긴 텍스트가 있으면 detailedText로도 설정
    const textareaRow = transformedRows.find(
      (r) => r.type === "textarea" && (r.answerEn?.length ?? 0) > 100
    );
    if (textareaRow) {
      content.detailedText = textareaRow.answer;
      content.detailedTextKo = textareaRow.answerKo;
      content.detailedTextEn = textareaRow.answerEn;
    }
  }
  // 레거시 형식: response 객체 처리
  else if (backendAnswer.response && typeof backendAnswer.response === "object") {
    const response = backendAnswer.response;

    if (response.error) {
      content.detailedText = `Error: ${response.error}`;
      content.detailedTextKo = `오류: ${response.error}`;
      content.detailedTextEn = `Error: ${response.error}`;
    } else if (response.raw_answer) {
      content.detailedText = response.raw_answer;
      content.detailedTextKo = response.raw_answer;
      content.detailedTextEn = response.raw_answer;
    } else {
      const rows: AnswerRow[] = [];
      for (const [key, value] of Object.entries(response)) {
        const schemaColumn = schema?.columns.find((col) => col.id === key);
        let displayValue = "";
        if (Array.isArray(value)) {
          displayValue = value.join(", ");
        } else if (typeof value === "object" && value !== null) {
          displayValue = JSON.stringify(value);
        } else if (value !== null && value !== undefined) {
          displayValue = String(value);
        }
        rows.push({
          number: `(${backendAnswer.question_id})`,
          detail: schemaColumn?.name_ko || schemaColumn?.name || key,
          detailEn: schemaColumn?.name || key,
          answer: displayValue,
          answerKo: displayValue,
          answerEn: displayValue,
          type: schemaColumn?.type || "text",
          options: schemaColumn?.options,
        });
      }
      content.rows = rows;
    }
  }

  // rationale (새 형식: rationale_en/ko)
  if (backendAnswer.rationale_en || backendAnswer.rationale_ko) {
    content.rationaleEn = backendAnswer.rationale_en;
    content.rationaleKo = backendAnswer.rationale_ko;
  }

  // sources를 evidence로 변환
  if (backendAnswer.sources?.length) {
    content.evidence = backendAnswer.sources.slice(0, 5).map((src) => ({
      source: `Page ${src.page_num}${src.section ? ` (${src.section})` : ""}`,
      excerpt: src.preview || "",
      page: src.page_num,
    }));
  }

  // 신뢰도 기반 insight 생성
  const confidencePercent = Math.round(confidence * 100);
  if (confidencePercent >= 70) {
    content.insight = "높은 신뢰도의 답변입니다. 지속가능성 보고서에서 관련 정보를 충분히 찾았습니다.";
    content.insightEn = "High confidence answer. Sufficient relevant information found in the sustainability report.";
  } else if (confidencePercent >= 50) {
    content.insight = "중간 수준의 신뢰도입니다. 일부 정보는 추가 검토가 필요할 수 있습니다.";
    content.insightEn = "Medium confidence. Some information may require additional review.";
  } else {
    content.insight = "신뢰도가 낮습니다. 보고서에서 관련 정보를 충분히 찾지 못했습니다. 수동 검토를 권장합니다.";
    content.insightEn = "Low confidence. Insufficient relevant information found. Manual review recommended.";
  }

  // 검증 오류가 있으면 analysis에 추가
  if (backendAnswer.validation_errors?.length) {
    content.analysisKo = `검증 주의사항:\n${backendAnswer.validation_errors.map((e) => `- ${e}`).join("\n")}`;
    content.analysisEn = `Validation notes:\n${backendAnswer.validation_errors.map((e) => `- ${e}`).join("\n")}`;
  }

  return {
    id: `answer-${backendAnswer.question_id}-${Date.now()}`,
    questionId: backendAnswer.question_id,
    content,
    confidence,
    sources: backendAnswer.sources,
    validationErrors: backendAnswer.validation_errors || [],
    // 테이블 타입의 원본 행 데이터 포함
    backendRows: backendAnswer.rows,
    responseType: backendAnswer.response_type,
  };
}

/**
 * 전체 CDP 답변 조회
 */
export async function fetchAllCDPAnswers(): Promise<CDPBackendResponse> {
  const response = await apiClient.get<CDPBackendResponse>("/cdp/answers");
  return response.data;
}

/**
 * 특정 질문에 대한 답변 조회
 */
export async function fetchCDPAnswer(
  questionId: string
): Promise<GeneratedAnswer | null> {
  try {
    const response = await apiClient.get<{
      answer: CDPBackendAnswer;
      schema: CDPQuestionSchema;
    }>(`/cdp/answers/${questionId}`);

    return transformBackendAnswer(response.data.answer, response.data.schema);
  } catch (error) {
    console.error(`Failed to fetch answer for ${questionId}:`, error);
    return null;
  }
}

/**
 * 답변 생성 요청
 */
export async function generateCDPAnswer(
  questionId: string,
  feedback?: string
): Promise<GeneratedAnswer | null> {
  try {
    const response = await apiClient.post<{
      answer: CDPBackendAnswer;
      schema: CDPQuestionSchema;
    }>("/cdp/generate", {
      question_id: questionId,
      feedback,
    });

    return transformBackendAnswer(response.data.answer, response.data.schema);
  } catch (error) {
    console.error(`Failed to generate answer for ${questionId}:`, error);
    return null;
  }
}

/**
 * 스키마 조회
 */
export async function fetchQuestionSchema(
  questionId: string
): Promise<CDPQuestionSchema | null> {
  try {
    const response = await apiClient.get<CDPQuestionSchema>(
      `/cdp/schema/${questionId}`
    );
    return response.data;
  } catch (error) {
    console.error(`Failed to fetch schema for ${questionId}:`, error);
    return null;
  }
}

/**
 * 로컬 JSON 파일에서 답변 로드 (개발/테스트용)
 */
export async function loadLocalAnswers(): Promise<Map<string, GeneratedAnswer>> {
  try {
    const response = await fetch("/data/cdp_structured_answers.json");
    const data: CDPBackendResponse = await response.json();

    const answersMap = new Map<string, GeneratedAnswer>();

    for (const backendAnswer of data.answers) {
      const transformed = transformBackendAnswer(backendAnswer);
      answersMap.set(backendAnswer.question_id, transformed);
    }

    return answersMap;
  } catch (error) {
    console.error("Failed to load local answers:", error);
    return new Map();
  }
}

/**
 * 로컬 스키마 파일에서 스키마 로드 (개발/테스트용)
 */
export async function loadLocalSchema(): Promise<Map<string, CDPQuestionSchema>> {
  try {
    const response = await fetch("/data/cdp_resolved_schema.json");
    const data = await response.json();

    const schemaMap = new Map<string, CDPQuestionSchema>();

    if (data.questions) {
      for (const [questionId, schema] of Object.entries(data.questions)) {
        schemaMap.set(questionId, schema as CDPQuestionSchema);
      }
    }

    return schemaMap;
  } catch (error) {
    console.error("Failed to load local schema:", error);
    return new Map();
  }
}
