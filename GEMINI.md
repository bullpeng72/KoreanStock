# Korean Stocks AI/ML Analysis System

이 프로젝트는 단순한 데이터 시각화를 넘어, AI와 머신러닝을 결합하여 실질적인 투자 의사결정을 지원하는 한국 주식 시장 분석 플랫폼입니다.

## 🎯 프로젝트 비전
"데이터에 기반한 객관적 분석과 AI의 통찰력을 결합하여, 수익률을 증명할 수 있는 투자 도구 구축"

## 🏗 아키텍처 원칙
1. **Decoupling (관심사 분리):** 비즈니스 로직(Core)과 사용자 인터페이스(Streamlit)를 엄격히 분리합니다. UI 없이도 독립적인 분석 엔진이 동작해야 합니다.
2. **Validation First (검증 우선):** 수익률이 검증되지 않은 모델은 노이즈일 뿐입니다. 모든 전략과 모델은 백테스팅 결과를 동반해야 합니다.
3. **Efficiency (효율성):** LLM(GPT-4o-mini) 호출 전 전처리를 통해 비용을 최적화하고 응답 속도를 높입니다.
4. **Automation (자동화):** 데이터 수집, 분석, 알림은 사용자 개입 없이 스케줄러(GitHub Actions 등)에 의해 수행됩니다.

## 🛠 기술 스택
- **UI:** Streamlit
- **Analytics:** Python (Pandas, Numpy)
- **ML:** Scikit-learn, XGBoost (Ensemble Model)
- **AI:** OpenAI GPT-4o-mini
- **Database:** SQLite
- **Automation:** GitHub Actions, Telegram Bot API

## 🚀 개선 로드맵

### Phase 1: 백테스팅 및 신뢰성 확보 (Completed) ✅
- [x] `utils/backtester.py` 구현: 수수료 및 슬리피지가 고려된 벡터화 백테스팅 엔진.
- [x] 모델 성능 리포트: 앙상블 모델 기반 예측 및 기술적 지표 점수화 완료.
- [x] 대시보드 업데이트: 종합 점수 및 AI 목표가 표시 기능 구현.

### Phase 2: 데이터 파이프라인 및 알림 자동화 (Completed) ✅
- [x] `core/engine/scheduler.py`: 매일 장 마감 후 자동 데이터 업데이트 및 분석 수행.
- [x] 텔레그램 알림 봇 연동: `notifier.py`를 통한 추천 리포트 자동 전송.
- [x] GitHub Actions 워크플로우 설정: `.github/workflows/daily_analysis.yml`을 통한 자동화 엔진 구동.

### Phase 3: 뉴스 감성 분석 고도화 (Completed) ✅
- [x] `core/engine/news_agent.py`: 네이버 뉴스 API를 통한 최신 뉴스 수집.
- [x] 스마트 필터링: GPT-4o-mini를 통한 호재/악재 점수화 및 분석 근거 산출.
- [x] 심리 지표 통합: 주가 데이터와 감성 점수를 결합한 종합 점수 체계 구축.

### Phase 4: 실전형 UX/UI 최적화 (Completed) ✅
- [x] 대시보드 및 워치리스트: Streamlit 기반 통합 인터페이스 구현.
- [x] 관심 종목 타임라인: 각 종목별 과거 분석 이력 조회 기능.
- [x] 테마 필터링: 반도체, AI 등 주요 테마별 종목 발굴 기능 추가.

### Phase 5: 시스템 확장성 및 고도화 (Upcoming)
- [ ] 실시간 데이터 스트리밍: WebSocket 기반 실시간 호가 및 체결 데이터 반영.
- [ ] 딥러닝 모델 도입: LSTM/Transformer 기반의 시계열 예측 모델 추가 및 앙상블 강화.
- [ ] 포트폴리오 최적화: 마코위츠 모델 등을 활용한 개인화된 자산 배분 제언.
- [ ] 사용자 맞춤형 알림: 특정 가격 도달 시 텔레그램 즉시 알림 기능.

## 📋 코딩 규칙 및 가이드라인
- **Error Handling:** 모든 데이터 크롤링 및 API 호출 시 예외 처리를 철저히 하고 로그를 남깁니다.
- **Type Hinting:** Python의 Type Hinting을 적극적으로 사용하여 코드 가독성과 안정성을 확보합니다.
- **Documentation:** 새로운 유틸리티나 에이전트 추가 시 함수 단위의 docstring을 작성합니다.
- **Cost Control:** LLM 사용 시 `max_tokens`를 제한하고, 필요한 정보만 추출하는 프롬프트를 유지합니다.
