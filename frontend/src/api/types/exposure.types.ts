// Exposure API 타입 정의

/**
 * 도시별 노출도 데이터
 */
export interface CityExposureData {
  regionId: number;
  si: string;
  city: string;
  scenario: string;
  time: number;
  exposure: number;
  additionalMetric: number;
}

/**
 * 시도별 노출도 데이터
 */
export interface ProvinceExposureData {
  regionId: number;
  province: string;
  scenario: string;
  time: number;
  avgExposure: number;
  maxExposure: number;
  minExposure: number;
  cityCount: number;
}

/**
 * 노출도 요약 응답
 */
export interface ExposureSummaryResponse {
  floodExposure: CityExposureData[];
  hotExposure: CityExposureData[];
  floodProvinceExposure: ProvinceExposureData[];
  hotProvinceExposure: ProvinceExposureData[];
}

/**
 * 노출도 조회 파라미터
 */
export interface ExposureSummaryParams {
  scenario?: string; // SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5
  time?: number; // 2030, 2040, 2050
}
