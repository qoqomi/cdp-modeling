import { apiClient } from "./client";

/**
 * Exposure Summary API 타입 정의 (백엔드 DTO 기준)
 */
export interface CityExposureData {
  regionId: number;
  si: string; // 시도명
  city: string; // 시군구명
  scenario: string;
  time: number;
  exposure: number;
  additionalMetric: number;
}

export interface ExposureSummaryResponse {
  floodExposure: CityExposureData[];
  hotExposure: CityExposureData[];
  timestamp: string;
}

export interface ExposureSummaryParams {
  scenario?: string;
  time?: number; // 백엔드는 'time' 사용
}

/**
 * Exposure API 서비스
 */
export const exposureApi = {
  /**
   * 전체 지역 Exposure 요약 조회
   * GET /api/v1/exposure/summary?scenario={scenario}&time={time}
   *
   * @param params - 조회 파라미터 (scenario, time)
   * @returns 전체 지역별 exposure 데이터
   */
  getSummary: async (
    params: ExposureSummaryParams
  ): Promise<ExposureSummaryResponse> => {
    const response = await apiClient.get<ExposureSummaryResponse>(
      "/v1/exposure/summary",
      { params }
    );
    return response.data;
  },
};
