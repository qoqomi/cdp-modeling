import { defineStore } from 'pinia';
import { ref } from 'vue';

export interface MapRegion {
  adm_nm?: string;
  sidonm?: string;
  sggnm?: string;
  adm_cd?: string;
}

export interface Location {
  lat: number;
  lng: number;
  address?: string;
}

export type SSPScenario = 'SSP1-2.6' | 'SSP2-4.5' | 'SSP3-7.0' | 'SSP5-8.5';
export type TargetYear = 2030 | 2040 | 2050;
export type ExposureType = 'flood' | 'hot';

export const useMapStore = defineStore('map', () => {
  const selectedRegion = ref<MapRegion | null>(null);
  const hoveredRegion = ref<MapRegion | null>(null);
  const currentLocation = ref<Location | null>(null);
  const mapInstance = ref<any>(null);
  const selectedFacility = ref<{ name: string; address: string; lat: number; lng: number } | null>(null);
  const selectedScenario = ref<SSPScenario>('SSP1-2.6');
  const selectedYear = ref<TargetYear>(2030);
  const exposureType = ref<ExposureType>('flood');
  const showRadiusCircle = ref(false);
  const radiusKm = ref(5);

  const setSelectedRegion = (region: MapRegion | null) => {
    selectedRegion.value = region;
  };

  const setHoveredRegion = (region: MapRegion | null) => {
    hoveredRegion.value = region;
  };

  const setCurrentLocation = (location: Location | null) => {
    currentLocation.value = location;
  };

  const setMapInstance = (map: any) => {
    mapInstance.value = map;
  };

  const flyToLocation = (lat: number, lng: number, zoom: number = 15, pitch?: number, bearing?: number) => {
    if (mapInstance.value) {
      // Mapbox GL JS인 경우 (3D 지원)
      if (mapInstance.value.flyTo && typeof mapInstance.value.flyTo === 'function') {
        // flyTo 메서드가 객체를 받는지 확인 (Mapbox GL JS 방식)
        try {
          mapInstance.value.flyTo({
            center: [lng, lat],
            zoom: zoom,
            pitch: pitch !== undefined ? pitch : 0, // 3D 기울기 (기본값: 0 = 평면)
            bearing: bearing !== undefined ? bearing : 0, // 3D 회전 (기본값: 0)
            duration: pitch !== undefined || bearing !== undefined ? 2000 : 1000, // 3D 전환 시 더 긴 애니메이션
            essential: true,
          });
          return;
        } catch (e) {
          // Mapbox GL JS가 아닌 경우 (Leaflet)
        }
      }
      
      // Leaflet인 경우 (하위 호환성)
      if (mapInstance.value.flyTo && typeof mapInstance.value.flyTo === 'function') {
        (mapInstance.value as any).flyTo([lat, lng], zoom, {
          duration: 1.0,
          easeLinearity: 0.25,
        });
      }
    }
  };

  const setSelectedFacility = (facility: { name: string; address: string; lat: number; lng: number } | null) => {
    selectedFacility.value = facility;
  };

  const setSelectedScenario = (scenario: SSPScenario) => {
    selectedScenario.value = scenario;
  };

  const setSelectedYear = (year: TargetYear) => {
    selectedYear.value = year;
  };

  const setExposureType = (type: ExposureType) => {
    exposureType.value = type;
  };

  const setShowRadiusCircle = (show: boolean, km: number = 5) => {
    showRadiusCircle.value = show;
    radiusKm.value = km;
  };

  return {
    selectedRegion,
    hoveredRegion,
    currentLocation,
    mapInstance,
    selectedFacility,
    selectedScenario,
    selectedYear,
    exposureType,
    showRadiusCircle,
    radiusKm,
    setSelectedRegion,
    setHoveredRegion,
    setCurrentLocation,
    setMapInstance,
    flyToLocation,
    setSelectedFacility,
    setSelectedScenario,
    setSelectedYear,
    setExposureType,
    setShowRadiusCircle,
  };
});
