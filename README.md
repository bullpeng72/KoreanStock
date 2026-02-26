# 📈 Korean Stocks AI/ML Analysis System

![version](https://img.shields.io/badge/version-0.2.2-blue)

> **KOSPI · KOSDAQ 종목을 AI와 머신러닝으로 분석하는 자동화 투자 보조 플랫폼**

---

## 목차
1. [프로젝트 소개](#-프로젝트-소개)
2. [주요 기능](#-주요-기능)
3. [기술 스택](#-기술-스택)
4. [시스템 아키텍처](#-시스템-아키텍처)
5. [분석 파이프라인](#-분석-파이프라인)
6. [점수 체계 해석](#-점수-체계-해석)
7. [실전 투자 활용 가이드](#-실전-투자-활용-가이드)
8. [설치 및 실행](#-설치-및-실행)
9. [자동화 설정](#-자동화-설정-github-actions)
10. [면책 조항](#-면책-조항)

---

## 🚀 프로젝트 소개

`Korean Stocks AI/ML Analysis System`은 기술적 지표 분석, 머신러닝 예측, 뉴스 감성 분석을 통합하여 한국 주식 시장의 유망 종목을 자동으로 발굴하고 리포트를 생성하는 플랫폼입니다.

매일 장 마감 후 자동으로 실행되어 KOSPI·KOSDAQ 전 종목 중 당일 **거래량 상위 + 등락률 상위** 종목을 스크리닝하고, 심층 분석 후 텔레그램으로 결과를 전송합니다.

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| **AI 종목 추천** | 기술적 지표·ML·뉴스를 종합한 복합 점수로 유망 종목 선정 |
| **날짜별 히스토리** | 과거 30일 분석 결과를 날짜 선택으로 조회 |
| **추천 지속성 히트맵** | 종목별 연속 추천 일수를 히트맵으로 시각화 (연속 2일+ 시 🔥 배지) |
| **DB 우선 조회 & 세션 캐시** | '새로 분석 실행' 클릭 시 당일 저장된 DB 결과 우선 표시 (불필요한 재분석 방지), 메뉴 이탈 후 재진입해도 결과 유지 |
| **DB 자동 동기화** | GitHub Actions 완료 후 분석 DB를 자동으로 저장소에 커밋·푸시 → 로컬에서 `git pull` 한 번으로 최신 추천 결과 반영 |
| **텔레그램 알림** | 종합점수 바·당일 등락률·RSI·뉴스 헤드라인·AI 강점 포함 구조화 리포트 발송 |
| **전략 백테스팅** | RSI · MACD · COMPOSITE 전략 시뮬레이션 (단순보유 비교, 원금선 차트, 초보자 해석 가이드 포함) |
| **관심 종목 관리** | Watchlist 등록 및 분석 이력 타임라인 제공 |
| **테마 필터링** | AI · 반도체 · 이차전지 · 바이오 등 테마별 종목 발굴 |
| **뉴스 기사 링크** | 감성 분석에 활용된 뉴스 기사 원문 링크 제공 |

---

## 🛠 기술 스택

```
UI          Streamlit
AI/LLM      OpenAI GPT-4o-mini
ML          Scikit-learn (Random Forest, Gradient Boosting), XGBoost
기술 지표    ta (RSI, MACD, Bollinger Bands, SMA, OBV)
데이터       FinanceDataReader, PyKrx (펀더멘털·수급), Naver News API
DB          SQLite
자동화       GitHub Actions (평일 16:30 KST), Telegram Bot API
시각화       Plotly, Matplotlib
언어         Python 3.11
```

---

## 🏗 시스템 아키텍처

```
Korean_Stocks/
├── main.py                          # Streamlit 웹 대시보드 (통합 UI)
├── core/
│   ├── config.py                    # 환경변수 및 설정 관리 (VERSION 포함)
│   ├── data/
│   │   ├── provider.py              # 주가 데이터 수집 (KOSPI/KOSDAQ 시장별 필터)
│   │   └── database.py              # SQLite 관리 (분석 결과, 워치리스트, 추천 이력)
│   ├── engine/
│   │   ├── indicators.py            # 기술적 지표 계산 (RSI, MACD, BB, SMA, OBV)
│   │   ├── strategy.py              # 전략별 시그널 생성
│   │   ├── prediction_model.py      # ML 앙상블 예측 (RF, GBR, XGB)
│   │   ├── news_agent.py            # 뉴스 수집 + 감성 분석 (GPT-4o-mini)
│   │   ├── analysis_agent.py        # 종목 심층 분석 오케스트레이터
│   │   ├── recommendation_agent.py  # 유망 종목 선정 + 추천 생성
│   │   └── scheduler.py             # 자동화 워크플로우 (Daily Task)
│   └── utils/
│       ├── backtester.py            # 전략 성과 검증 엔진
│       └── notifier.py              # 텔레그램 리포트 발송
├── models/saved/                    # 학습된 ML 모델 및 파라미터
├── data/storage/                    # SQLite 데이터베이스 파일
├── train_models.py                  # ML 모델 재학습 스크립트
├── tests/
│   └── test_backtester.py           # 백테스터 단위 테스트 (pytest)
└── .github/workflows/
    └── daily_analysis.yml           # GitHub Actions 자동화 스케줄러
```

---

## 🔬 분석 파이프라인

### 자동화 대상 종목 선정
```
KRX 전체 상장 종목
    ├── 당일 거래량 상위 100종목
    └── 당일 등락률 상위 100종목
            ↓ 합집합 (선택한 시장 KOSPI / KOSDAQ / ALL 필터 적용)
        최대 약 100개 후보
            ↓ 거래량 순 정렬 후 상위 30개 심층 분석
        최종 5종목 추천
```

### 종목별 심층 분석 (4단계)

```
1단계  기술적 지표 계산            → tech_score (0–100)
       SMA 5/20/60/120, MACD, RSI(14), Bollinger Bands, OBV, Stochastic, CCI, ATR

2단계  ML 앙상블 예측              → ml_score (0–100)
       Random Forest + Gradient Boosting + XGBoost (RMSE 역수 가중 앙상블)
       31개 피처 (BASE 22 + PyKrx 9, train_models.py FEATURE_COLS 기준):
         · 기본 지표 (4): RSI, MACD diff, 가격/SMA20 비율, 거래량 변화율
         · 추세 변화 (3): 가격/SMA5 비율, RSI 변화율, MACD diff 변화율
         · 볼린저 밴드 (3): BB 위치, BB 너비, 20일 평균 대비 거래량 비율
         · 모멘텀 오실레이터 (3): Stochastic K, Stochastic D, CCI
         · 변동성·거래량 (3): ATR 비율, 당일 캔들 바디 비율, OBV 변화율
         · 모멘텀 팩터 (4): 1m·3m 수익률, 52주 고점 비율, 모멘텀 가속도
         · 시장 상대강도 (2): KOSPI↔KS11 / KOSDAQ↔KQ11 기준 1m·3m 초과수익
         · 펀더멘털 (5): PBR, PER, DIV, PBR 크로스섹셔널 순위, PER 크로스섹셔널 순위
         · 수급 (4): 외국인/기관 5일 누적 순매수 비율, 외국인/기관 크로스섹셔널 순위
       모델 없을 경우 tech_score 폴백
       예측 의미: 향후 5거래일 크로스섹셔널 순위 (0=최하위, 50=평균, 100=최상위)

3단계  뉴스 감성 분석              → sentiment_score (-100–100)
       Naver News API → 최신 뉴스 15건 → GPT-4o-mini 감성 점수
       최신 뉴스(당일·1일 전)에 높은 가중치 부여

4단계  AI 종합 의견                → action (BUY/HOLD/SELL)
       GPT-4o-mini에 전 단계 데이터 + 점수 기준표 전달
       → 요약, 강점, 약점, 추천 사유, 목표가 생성
       → action과 목표가 일관성 자동 보정

종합 점수 (ML 모델 활성 시) = tech×0.40 + ml×0.35 + sentiment_norm×0.25
종합 점수 (ML 모델 없을 시) = tech×0.65 + sentiment_norm×0.35
  ※ sentiment_norm = (sentiment_score + 100) / 2  → 0–100 정규화
```

---

## 📊 점수 체계 해석

### Tech Score (기술적 지표 종합)

| 점수 | 해석 |
|------|------|
| 80–100 | 매우 강세 |
| 60–79 | 강세 |
| 40–59 | 중립 |
| 0–39 | 약세 |

**세부 구성 (합계 100점)**

**① 추세 (40점)**

| 조건 | SMA60 계산 가능 시 | SMA60 미계산 시 |
|------|-------------------|----------------|
| 종가 > SMA20 | +10 | +10 |
| SMA5 > SMA20 | +10 | +10 |
| MACD > Signal (골든크로스) | +15 | +20 |
| 종가 > SMA60 (중기 추세 확인) | +5 | — |

> SMA60 미계산 시 MACD에 가중치를 흡수하여 합계 40점 유지

**② 모멘텀 (30점) — RSI 구간별**

| RSI 구간 | 점수 | 해석 |
|----------|------|------|
| 45–65 | +30 | 상승 추세 내 최적 구간 |
| 35–45 | +22 | 하락 완화, 반등 준비 |
| 65–75 | +18 | 강한 상승 모멘텀 (과매수 경계) |
| 30–35 | +12 | 과매도 근접, 주의 필요 |
| 75 초과 | +8 | 강한 과매수 (오버히팅) |
| 30 미만 | +4 | 깊은 과매도 (고위험) |

**③ BB 위치 + 거래량 (30점)**

BB 위치 (25점) — MACD 방향에 따라 최적 구간 이동:

| BB 위치 | 상승추세 (MACD↑) | 하락/중립 (MACD↓) |
|---------|-----------------|------------------|
| 0.4–0.75 | +25 (추세 추종 최적) | +14 |
| 0.2–0.5 | +14 | +25 (반등 매수 최적) |
| 0.75–0.9 | +18 | +8 |
| 0.5–0.7 | — | +18 |
| 0.1–0.2 | — | +12 |
| 0.9 초과 | +8 | +3 (밴드 이탈) |
| 0.2 미만 | +3 (하단 이탈) | — |
| 0.1 미만 | — | +3 (밴드 이탈) |

거래량 확인 (5점): 당일 거래량 ≥ 20일 평균의 1.5배 → +5

### ML Score (머신러닝 예측)

향후 **5거래일 크로스섹셔널 순위** 예측값 (절대 수익률이 아닌 상대강도 순위).

| 점수 | 해석 |
|------|------|
| 60–100 | 상위권 상대강도 예상 |
| 40–59 | 중간 수준 |
| 0–39 | 하위권 상대강도 예상 |

### News Sentiment Score (뉴스 감성)

| 점수 | 해석 |
|------|------|
| 51–100 | Very Bullish (매우 긍정) |
| 1–50 | Bullish (긍정) |
| 0 | Neutral |
| -49 – -1 | Bearish (부정) |
| -100 – -50 | Very Bearish (매우 부정) |

---

## 💡 실전 투자 활용 가이드

> ⚠️ 본 시스템은 **투자 보조 도구**입니다. 최종 투자 결정은 반드시 본인이 직접 판단하세요.

### 1. 신호 해석 기준

```
강력 매수 후보 (모든 조건 충족 시)
  ✅ Tech Score >= 65
  ✅ ML Score >= 60
  ✅ News Score > 20
  ✅ AI action = BUY
  ✅ RSI: 35–50 구간 (과매도 탈출 또는 중립 하단)
  ✅ MACD: 골든크로스 발생 또는 유지

관망 권고
  - Tech < 50 이고 MACD 데드크로스 상태
  - News Score < -30 (강한 악재 뉴스)
  - RSI > 75 (과열 구간)

매도 검토
  - AI action = SELL + Tech Score < 40
  - RSI > 75 + MACD 데드크로스 동시 발생
```

### 2. 단계별 활용 방법

**Step 1 — 스크리닝 (매일 자동)**
- 텔레그램 알림으로 오늘의 추천 5종목 확인
- 종합 점수 상위 2–3종목을 후보로 선정

**Step 2 — 지속성 확인**
- AI Recommendations → 📅 추천 지속성 히트맵에서 연속 추천 일수 확인
- 2일+ 연속 추천 종목(🔥 배지)은 신호 신뢰도가 높음

**Step 3 — 심층 검증 (수동)**
- Dashboard 또는 AI Recommendations에서 상세 리포트 확인
- 강점/약점, 뉴스 근거 및 원문 링크, 목표가 근거 직접 검토
- Backtest Viewer에서 해당 종목의 전략별 과거 성과 확인

**Step 4 — 최종 판단 기준**
```
AI 추천만으로 매수 ❌
AI 추천 + 아래 조건 중 2개 이상 충족 시 매수 검토 ✅
  - 최근 5일 거래량이 20일 평균 대비 150% 이상
  - 52주 저점 대비 -20% 이내 (저점 매수 구간)
  - 섹터 전반적 상승 분위기
  - 뉴스 감성 Bullish 이상
  - 추천 지속성 히트맵에서 2일+ 연속 추천
```

### 3. 리스크 관리

| 원칙 | 설명 |
|------|------|
| **분산 투자** | 추천 5종목 중 동일 섹터에 몰리지 않도록 1–2종목만 선택 |
| **손절 기준** | 매수가 대비 7–8% 하락 시 손절 고려 (시스템은 손절선 미제공) |
| **목표가 활용** | 목표가는 단기 참고값이며 보장 수치가 아님. 실현 후 일부 익절 전략 권장 |
| **비중 관리** | 단일 종목에 총 자산의 10% 이상 집중 투자 지양 |
| **재검증** | 매수 후 3–5일 내 시스템 재분석으로 의견 변화 모니터링 |

### 4. 점수별 권장 포지션 크기

```
종합 점수 75 이상 + BUY → 일반 비중 (예: 5–7%)
종합 점수 65–74 + BUY   → 소규모 진입 (예: 3–5%)
종합 점수 55–64 + BUY   → 관망 또는 최소 비중 (예: 1–2%)
종합 점수 55 미만        → 매수 보류
```

### 5. 활용 시 주의사항

- **뉴스 감성은 당일 헤드라인 기반** — 호재성 기사 뒤 실적 부진 가능
- **ML 모델 미탑재 시 tech_score 대체** — 로그 메시지로 확인 가능
- **ML 점수는 절대 수익률이 아닌 상대강도 순위** — 시장 전체가 하락장이면 점수 높아도 손실 가능
- **자동화는 평일 16:30 KST 실행** — 당일 주가 반영, 다음날 매매 판단에 활용
- **빠른 시장 변동 반영 불가** — 급등락 당일은 직접 현재 가격 확인 필수

---

## ⚙️ 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/bullpeng72/KoreanStock.git
cd KoreanStock
```

### 2. Python 환경 설정 (Python 3.11 권장)
```bash
conda create -n stocks_env python=3.11
conda activate stocks_env
```

### 3. 의존성 설치
```bash
# XGBoost 구동에 필요한 시스템 라이브러리 (Ubuntu/Debian)
sudo apt-get install -y libomp-dev

pip install -r requirements.txt
```

### 4. 환경 변수 설정 (`.env` 파일)
```ini
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_TELEGRAM_CHAT_ID
NAVER_CLIENT_ID=YOUR_NAVER_CLIENT_ID
NAVER_CLIENT_SECRET=YOUR_NAVER_CLIENT_SECRET
DB_PATH=data/storage/stock_analysis.db
```

| 변수 | 발급처 |
|------|--------|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| `TELEGRAM_BOT_TOKEN` | 텔레그램 [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | 텔레그램 [@userinfobot](https://t.me/userinfobot) |
| `NAVER_CLIENT_ID/SECRET` | [developers.naver.com](https://developers.naver.com) — 검색 API 신청 |

### 5. ML 모델 학습 (최초 1회)
```bash
python train_models.py
```

### 6. 앱 실행
```bash
streamlit run main.py
```
브라우저에서 `http://localhost:8501` 접속

---

## 🤖 자동화 설정 (GitHub Actions)

**실행 시점:** 평일 오후 16:30 KST (UTC 07:30) — 장 마감 후 자동 실행

**설정 방법:**
1. 저장소를 GitHub에 푸시
2. `Settings > Secrets and variables > Actions`에서 아래 5개 Secret 등록

```
OPENAI_API_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
NAVER_CLIENT_ID      (없으면 뉴스 감성 분석 스킵)
NAVER_CLIENT_SECRET
```

3. `Actions` 탭에서 워크플로우 활성화 확인
4. 수동 실행: `Actions > Daily Stock Analysis > Run workflow`

**자동화 흐름:**
```
16:30 KST
  → KOSPI + KOSDAQ 종목 리스트 갱신
  → 거래량/등락률 상위 100종목 스크리닝
  → 상위 30종목 심층 분석 (기술 + ML + 뉴스 + GPT)
  → 종합 점수 상위 5종목 선정
  → SQLite DB 날짜별 저장
  → GitHub Artifact에 DB 백업 (90일 보존)
  → DB를 저장소에 자동 커밋·푸시 → 로컬 git pull로 즉시 반영
  → 텔레그램 리포트 발송
```

---

## 📱 메뉴 구성

| 메뉴 | 주요 기능 |
|------|----------|
| **Dashboard** | 시장 지수, 관심 종목 요약, 날짜별 AI 추천 리포트 조회 |
| **My Watchlist** | 관심 종목 등록/삭제, 실시간 심층 분석, 분석 이력 타임라인 |
| **AI Recommendations** | 테마·시장별 추천 생성, 당일 DB 우선 조회 (강제 재분석 옵션 별도 제공), 날짜 선택으로 히스토리 조회, 추천 지속성 히트맵 |
| **Backtest Viewer** | RSI/MACD/COMPOSITE 전략 시뮬레이션, 단순보유(Buy&Hold) 비교 차트, 초보자 해석 가이드 |
| **Automation & Settings** | 수동 자동화 실행, 텔레그램 설정 상태 확인 |

---

## ⚠️ 면책 조항

본 소프트웨어는 **교육 및 정보 제공 목적**으로만 제작되었습니다.

- 본 시스템의 분석 결과는 투자 권유 또는 금융 조언이 아닙니다.
- AI 및 ML 모델의 예측은 미래 수익을 보장하지 않습니다.
- 주식 투자에는 원금 손실의 위험이 있습니다.
- 최종 투자 결정과 그에 따른 손익은 전적으로 투자자 본인에게 있습니다.

---

## 📄 라이선스

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.

---

*(C) 2026. All rights reserved.*
