// Climate Risk API 타입 정의

/**
 * 시계열 데이터 (특정 시점의 Hazard + Exposure)
 */
export interface TimeSeriesData {
  time: number; // 년도 (2030, 2040, 2050)
  hazard: number; // 위험도
  exposure: number; // 노출도
  additionalMetric: number; // 추가 메트릭 (홍수: 평균 확률, 폭염: 평균 일수)
}

/**
 * 시나리오별 데이터
 */
export interface ScenarioData {
  scenario: string; // SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5
  timeSeries: TimeSeriesData[]; // 시간별 데이터
}

/**
 * Hazard + Exposure 통합 데이터
 */
export interface HazardExposureData {
  scenarios: ScenarioData[]; // 시나리오별 데이터
}

/**
 * MAAL 시계열 데이터
 */
export interface MaalTimeData {
  time: number; // 년도
  companyMaalPercent: number; // 기업 MAAL (백분율)
  companyMaalKrw: number; // 기업 MAAL (금액, KRW)
  weightedRiskIndex: number; // 가중 위험 지수
  finalRiskScore: number; // 최종 위험 점수
  finalRiskLevel: string; // 최종 위험 등급
}

/**
 * MAAL 시나리오별 데이터
 */
export interface MaalScenarioData {
  scenario: string; // 시나리오명
  timeSeries: MaalTimeData[]; // 시간별 MAAL 데이터
}

/**
 * MAAL (Maximum Annual Aggregate Loss) 데이터
 */
export interface MaalData {
  scenarios: MaalScenarioData[]; // 시나리오별 MAAL 데이터
}

/**
 * 기후 위험 응답
 */
export interface ClimateRiskResponse {
  regionId: number; // 지역 ID (sgg 코드)
  si: string; // 시도명
  city: string; // 시군구명
  floodData: HazardExposureData; // 홍수 위험 데이터
  hotData: HazardExposureData; // 폭염 위험 데이터
  maalData: MaalData; // MAAL 데이터
  timestamp: string; // 응답 생성 시간
}

/**
 * 기후 위험 조회 파라미터
 */
export interface ClimateRiskParams {
  region_id: number; // 지역 ID (sgg 코드 5자리, 예: 11110)
}
