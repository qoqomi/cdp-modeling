import type { GeoJsonData } from "@/composables/useGeoJson";

/**
 * GeoJSON Feature 속성
 */
export interface GeoJsonFeatureProperties {
  adm_nm: string; // 행정동 명칭 (예: "서울특별시 종로구 사직동")
  adm_cd2: string; // 행정동 코드 (10자리)
  sgg: string; // 시군구 코드 (5자리)
  sido: string; // 시도 코드 (2자리)
  sidonm: string; // 시도명 (예: "서울특별시")
  sggnm: string; // 시군구명 (예: "종로구")
  adm_cd: string; // 행정동 코드 (8자리)
}

/**
 * 주소 문자열로 GeoJSON에서 해당하는 sgg 코드 찾기
 *
 * @param geoJsonData - GeoJSON 데이터
 * @param address - 검색할 주소 (예: "서울특별시 종로구", "경기도 성남시")
 * @returns sgg 코드 (5자리 숫자) 또는 null
 */
export function findSggByAddress(
  geoJsonData: GeoJsonData | null,
  address: string
): number | null {
  if (!geoJsonData || !geoJsonData.features || !address) {
    return null;
  }

  const normalizedAddress = normalizeAddress(address);

  // 주소와 매칭되는 feature 찾기
  const feature = geoJsonData.features.find((f) => {
    const props = f.properties as GeoJsonFeatureProperties;
    if (!props) return false;

    // adm_nm, sidonm + sggnm으로 매칭 시도
    const admNm = normalizeAddress(props.adm_nm || "");
    const sidoSgg = normalizeAddress(`${props.sidonm}${props.sggnm}`);
    const sggOnly = normalizeAddress(props.sggnm || "");

    return (
      admNm.includes(normalizedAddress) ||
      normalizedAddress.includes(admNm) ||
      sidoSgg.includes(normalizedAddress) ||
      normalizedAddress.includes(sidoSgg) ||
      (sggOnly && normalizedAddress.includes(sggOnly))
    );
  });

  if (feature && feature.properties) {
    const sgg = (feature.properties as GeoJsonFeatureProperties).sgg;
    return sgg ? parseInt(sgg, 10) : null;
  }

  return null;
}

/**
 * 좌표로 해당하는 행정구역의 sgg 코드 찾기
 *
 * @param geoJsonData - GeoJSON 데이터
 * @param lng - 경도
 * @param lat - 위도
 * @returns sgg 코드 (5자리 숫자) 또는 null
 */
export function findSggByCoordinates(
  geoJsonData: GeoJsonData | null,
  lng: number,
  lat: number
): number | null {
  if (!geoJsonData || !geoJsonData.features) {
    return null;
  }

  // 좌표가 포함된 폴리곤 찾기
  for (const feature of geoJsonData.features) {
    if (isPointInFeature(feature, lng, lat)) {
      const props = feature.properties as GeoJsonFeatureProperties;
      if (props && props.sgg) {
        return parseInt(props.sgg, 10);
      }
    }
  }

  return null;
}

/**
 * 시도명과 시군구명으로 sgg 코드 찾기
 *
 * @param geoJsonData - GeoJSON 데이터
 * @param sidonm - 시도명 (예: "서울특별시")
 * @param sggnm - 시군구명 (예: "종로구")
 * @returns sgg 코드 (5자리 숫자) 또는 null
 */
export function findSggByRegion(
  geoJsonData: GeoJsonData | null,
  sidonm: string,
  sggnm: string
): number | null {
  if (!geoJsonData || !geoJsonData.features) {
    return null;
  }

  const normalizedSido = normalizeAddress(sidonm);
  const normalizedSgg = normalizeAddress(sggnm);

  const feature = geoJsonData.features.find((f) => {
    const props = f.properties as GeoJsonFeatureProperties;
    if (!props) return false;

    return (
      normalizeAddress(props.sidonm) === normalizedSido &&
      normalizeAddress(props.sggnm) === normalizedSgg
    );
  });

  if (feature && feature.properties) {
    const sgg = (feature.properties as GeoJsonFeatureProperties).sgg;
    return sgg ? parseInt(sgg, 10) : null;
  }

  return null;
}

/**
 * sgg 코드로 시도명과 시군구명 가져오기
 *
 * @param geoJsonData - GeoJSON 데이터
 * @param sggCode - 시군구 코드 (5자리 숫자)
 * @returns 시도명과 시군구명 또는 null
 */
export function getRegionNamesBySgg(
  geoJsonData: GeoJsonData | null,
  sggCode: number
): { sidonm: string; sggnm: string } | null {
  if (!geoJsonData || !geoJsonData.features) {
    return null;
  }

  const sggStr = sggCode.toString().padStart(5, "0");

  const feature = geoJsonData.features.find((f) => {
    const props = f.properties as GeoJsonFeatureProperties;
    return props && props.sgg === sggStr;
  });

  if (feature && feature.properties) {
    const props = feature.properties as GeoJsonFeatureProperties;
    return {
      sidonm: props.sidonm,
      sggnm: props.sggnm,
    };
  }

  return null;
}

// ========== Private Helper Functions ==========

/**
 * 주소 문자열 정규화 (공백, 특수문자 제거)
 */
function normalizeAddress(address: string): string {
  return address
    .replace(/\s+/g, "") // 공백 제거
    .replace(/[^\w가-힣]/g, "") // 한글, 영문, 숫자만 남기기
    .toLowerCase();
}

/**
 * 점이 feature(폴리곤) 내부에 있는지 확인
 * Ray Casting 알고리즘 사용
 */
function isPointInFeature(feature: any, lng: number, lat: number): boolean {
  const geometry = feature.geometry;
  if (!geometry) return false;

  if (geometry.type === "Polygon") {
    return isPointInPolygon([lng, lat], geometry.coordinates[0]);
  } else if (geometry.type === "MultiPolygon") {
    for (const polygon of geometry.coordinates) {
      if (isPointInPolygon([lng, lat], polygon[0])) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Ray Casting 알고리즘으로 점이 폴리곤 내부에 있는지 확인
 */
function isPointInPolygon(point: number[], polygon: number[][]): boolean {
  const [x, y] = point;
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];

    const intersect =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;

    if (intersect) {
      inside = !inside;
    }
  }

  return inside;
}
