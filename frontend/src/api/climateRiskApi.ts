import { apiClient } from "./client";
import type {
  ClimateRiskResponse,
  ClimateRiskParams,
} from "./types/climateRisk.types";

/**
 * Climate Risk API 서비스
 */
export const climateRiskApi = {
  /**
   * 지역별 기후 위험 상세 조회
   * GET /api/v1/climate-risk?region_id={id}
   *
   * @param params - 조회 파라미터 (region_id)
   * @returns 기후 위험 상세 데이터 (홍수/폭염 Hazard, Exposure, MAAL)
   */
  getClimateRisk: async (
    params: ClimateRiskParams
  ): Promise<ClimateRiskResponse> => {
    const response = await apiClient.get<ClimateRiskResponse>(
      "/v1/climate-risk",
      { params }
    );
    return response.data;
  },
};
