export interface Facility {
  name: string;
  address: string;
  lat: number;
  lng: number;
  iconType?: "building" | "tower" | "office" | "data-center" | "headquarters";
}

// SK 시설 위치 데이터 (주소 기반 추정 좌표)
export const facilities: Facility[] = [
  {
    name: "SK AX 판교 데이터센터",
    address: "경기도 성남시 분당구 판교로 255번길 46 (13486)",
    lat: 37.4014,
    lng: 127.1083,
    iconType: "data-center",
  },
  {
    name: "SK AX 대덕 데이터센터",
    address: "대전광역시 유성구 엑스포로 325 (34124)",
    lat: 36.3852,
    lng: 127.4049,
    iconType: "data-center",
  },
  {
    name: "삼성 SDS 구미 데이터센터",
    address: "경상북도 구미시 3공단3로 302 (임수동) (39388)",
    lat: 36.127,
    lng: 128.344,
    iconType: "data-center",
  },
  {
    name: "네이버 각(閣) 춘천 데이터센터",
    address:
      "강원특별자치도 춘천시 동면 만천리 (위치: 북위 37.8891052, 동경 127.7741472) :contentReference[oaicite:0]{index=0}",
    lat: 37.8891052,
    lng: 127.7741472,
    iconType: "data-center",
  },
  {
    name: "카카오 데이터센터 안산",
    address:
      "경기도 안산시 상록구 해안로 689 (15588) :contentReference[oaicite:1]{index=1}",
    lat: 37.3075,
    lng: 126.822,
    iconType: "data-center",
  },
];
