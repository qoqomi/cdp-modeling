import axios from "axios";

// API 베이스 URL 설정
// Vite proxy를 사용하는 경우 상대 경로로 설정
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

// Axios 인스턴스 생성
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000, // 10분 (AI 처리 시간 고려)
  headers: {
    "Content-Type": "application/json",
  },
});

// 요청 인터셉터 (필요시 토큰 추가 등)
apiClient.interceptors.request.use(
  (config) => {
    // 예: 토큰이 있으면 헤더에 추가
    // const token = localStorage.getItem('token');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 응답 인터셉터 (에러 처리)
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // 에러 처리
    if (error.response) {
      // 서버가 응답을 반환한 경우
      console.error("API Error:", error.response.status, error.response.data);
    } else if (error.request) {
      // 요청은 보냈지만 응답을 받지 못한 경우
      console.error("Network Error:", error.request);
    } else {
      // 요청 설정 중 에러 발생
      console.error("Error:", error.message);
    }
    return Promise.reject(error);
  }
);
