# Climate Change Dashboard

Vue 3 기반 기후 위험 평가 대시보드입니다.

## 기술 스택

- **Vue 3** - Composition API 사용
- **TypeScript** - 타입 안정성
- **Vite** - 빠른 개발 서버 및 빌드
- **Tailwind CSS** - 유틸리티 기반 CSS
- **ECharts (vue-echarts)** - 차트 라이브러리
- **Lucide Vue Next** - 아이콘 라이브러리

## 설치

```bash
npm install
```

## 개발 서버 실행

```bash
npm run dev
```

개발 서버는 `http://localhost:3000`에서 실행됩니다.

## 빌드

```bash
npm run build
```

빌드된 파일은 `dist` 디렉토리에 생성됩니다.

## 프로젝트 구조

```
src/
├── components/
│   ├── Header.vue          # 헤더 컴포넌트
│   ├── FilterPanel.vue      # 필터 패널
│   ├── CenterPanel.vue     # 중앙 맵 패널
│   ├── RightPanel.vue      # 결과 패널 (차트 포함)
│   └── MapView.vue         # 지도 뷰 컴포넌트
├── App.vue                 # 메인 앱 컴포넌트
├── main.ts                 # 앱 진입점
└── index.css               # 글로벌 스타일
```

## 주요 기능

- 기후 위험 분석 대시보드
- 실시간 데이터 시각화
- 인터랙티브 지도 뷰
- 위험 평가 및 재무 영향 분석
- 반응형 UI 디자인
