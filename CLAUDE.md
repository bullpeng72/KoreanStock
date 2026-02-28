# Korean Stocks AI/ML Analysis System `v0.2.6`

KOSPI·KOSDAQ 종목을 기술적 지표, 머신러닝, 뉴스 감성 분석으로 자동 스크리닝하고 텔레그램 리포트를 발송하는 투자 보조 플랫폼.

## 아키텍처 원칙

1. **Decoupling:** 비즈니스 로직(`koreanstocks.core/`)과 API/UI를 엄격히 분리. API 서버 없이도 CLI로 분석 엔진이 독립 동작해야 함.
2. **Validation First:** 모든 전략과 ML 모델은 백테스팅 결과를 동반해야 함.
3. **Cost Control:** LLM(GPT-4o-mini) 호출 전 전처리로 비용 최적화. `max_tokens` 제한 필수.
4. **Automation:** 데이터 수집·분석·알림은 GitHub Actions 스케줄러가 담당 (평일 16:30 KST). SQLite DB는 저장소에 자동 커밋·푸시되며, GitHub Artifact로도 병행 백업 (90일 보존).

## 기술 스택

- **UI:** FastAPI + Reveal.js (일일 브리핑 슬라이드) + Vanilla JS (인터랙티브 대시보드)
- **CLI:** Typer (`koreanstocks serve / recommend / analyze / train / init / sync / home`)
- **AI/LLM:** OpenAI GPT-4o-mini
- **ML:** Scikit-learn (Random Forest, Gradient Boosting), XGBoost
- **기술 지표:** `ta` 라이브러리 (RSI, MACD, Bollinger Bands, SMA, OBV)
- **데이터:** FinanceDataReader, PyKrx (펀더멘털·수급), Naver News API
- **DB:** SQLite (`data/storage/stock_analysis.db`)
- **자동화:** GitHub Actions, Telegram Bot API
- **시각화:** Plotly, Matplotlib, Chart.js (백테스트 차트)
- **언어:** Python 3.11 ~ 3.13

## 프로젝트 구조

```
pyproject.toml                       # pip 빌드 설정 (koreanstocks CLI 진입점)
requirements.txt                     # 개발/테스트 전용 (pytest 등)
src/
└── koreanstocks/
    ├── __init__.py                  # VERSION = "0.2.6"
    ├── cli.py                       # Typer CLI (serve/recommend/analyze/train/init/sync/home)
    ├── api/
    │   ├── app.py                   # FastAPI 앱 팩토리, StaticFiles 마운트
    │   ├── dependencies.py          # 공통 의존성 (db_manager, analysis_agent 등)
    │   └── routers/
    │       ├── recommendations.py   # GET/POST /api/recommendations
    │       ├── analysis.py          # GET/POST /api/analysis/{code}
    │       ├── watchlist.py         # CRUD /api/watchlist
    │       ├── backtest.py          # GET /api/backtest
    │       └── market.py            # GET /api/market
    ├── static/
    │   ├── index.html               # Reveal.js 일일 브리핑 슬라이드
    │   ├── dashboard.html           # 인터랙티브 대시보드 (5탭)
    │   ├── js/
    │   │   ├── slides.js            # 슬라이드 동적 생성 (API fetch)
    │   │   └── dashboard.js         # 대시보드 인터랙션
    │   └── css/
    │       └── theme.css            # 공통 스타일
    └── core/
        ├── config.py                # 환경변수 및 설정 (dotenv), VERSION 상수
        ├── data/
        │   ├── provider.py          # 주가·뉴스 데이터 수집
        │   └── database.py          # SQLite CRUD
        ├── engine/
        │   ├── indicators.py        # 기술적 지표 계산 (RSI, MACD, BB 등)
        │   ├── strategy.py          # 전략별 시그널 생성 (TechnicalStrategy)
        │   ├── prediction_model.py  # ML 앙상블 예측 (RF + GBR + XGB)
        │   ├── news_agent.py        # 뉴스 수집 + GPT 감성 분석
        │   ├── analysis_agent.py    # 종목 심층 분석 오케스트레이터
        │   ├── recommendation_agent.py  # 유망 종목 선정 + 추천 생성
        │   └── scheduler.py         # 자동화 워크플로우
        └── utils/
            ├── backtester.py        # 전략 성과 검증 엔진
            └── notifier.py          # 텔레그램 리포트 발송
models/saved/                        # 학습된 ML 모델 (.pkl) 및 파라미터 (.json)
data/storage/                        # SQLite DB 파일
train_models.py                      # ML 모델 재학습 스크립트
tests/
├── test_backtester.py               # 백테스터 단위 테스트 (pytest)
└── compat_check.py                  # Python 3.11~3.13 호환성 검증 스크립트
.github/workflows/
└── daily_analysis.yml               # GitHub Actions 자동화 스케줄러
```

## 분석 파이프라인

```
1단계  기술적 지표 → tech_score (0~100)
2단계  ML 앙상블   → ml_score (0~100)  [모델 없으면 tech_score 폴백]
3단계  뉴스 감성   → sentiment_score (-100~100)
4단계  GPT 종합    → action (BUY/HOLD/SELL), 요약, 목표가

종합 점수 (ML 모델 활성 시) = tech×0.40 + ml×0.35 + sentiment_norm×0.25
종합 점수 (ML 모델 없을 시) = tech×0.65 + sentiment_norm×0.35
  ※ sentiment_norm = (sentiment_score + 100) / 2  → 0~100 정규화
```

## 주요 명령어

```bash
# 패키지 설치
pip install -e .              # 개발 / git clone 환경 (editable)
pip install koreanstocks      # PyPI 전역 설치 (DB는 ~/.koreanstocks/ 에 생성)

# 초기 설정 (.env 대화형 생성 — API 키를 프롬프트로 입력)
koreanstocks init                   # 대화형 입력
koreanstocks init --non-interactive  # 빈 템플릿만 생성 (CI용)

# 데이터 홈 디렉토리 (.env, DB, 모델 저장 위치)
koreanstocks home                   # 경로 출력 (cd $(koreanstocks home) 로 이동)
koreanstocks home --open            # 파일 탐색기로 열기
koreanstocks home --setup           # 셸 alias 스니펫 출력 (~/.bashrc / ~/.zshrc)

# GitHub Actions 생성 DB 다운로드 (PyPI 설치 후 추천 데이터 즉시 사용 가능)
koreanstocks sync              # 최초 수신 또는 날짜 갱신
koreanstocks sync --force      # 로컬 DB가 있어도 강제 덮어쓰기

# 웹 대시보드 실행 (브라우저 자동 실행)
koreanstocks serve                     # http://localhost:8000/dashboard
koreanstocks serve --port 8080         # 포트 변경
koreanstocks serve --no-browser        # 브라우저 자동 실행 비활성화

# 오늘의 추천 종목 분석 (GitHub Actions용)
koreanstocks recommend
koreanstocks recommend --market KOSPI --limit 10

# 단일 종목 심층 분석
koreanstocks analyze 005930

# ML 모델 재학습
koreanstocks train
python train_models.py                 # 직접 실행도 가능

# 단위 테스트 실행
pytest tests/

# Python 3.11~3.13 호환성 검증
python tests/compat_check.py
```

## 환경 변수 (`.env`)

```ini
OPENAI_API_KEY=...          # 필수: GPT-4o-mini (감성 분석, AI 의견 생성)
TELEGRAM_BOT_TOKEN=...      # 필수: 추천 리포트 발송
TELEGRAM_CHAT_ID=...        # 필수: 수신 채팅방 ID
NAVER_CLIENT_ID=...         # 필수: 뉴스 검색 API
NAVER_CLIENT_SECRET=...
DART_API_KEY=...            # 선택: 금융감독원 공시 수집 (미설정 시 뉴스만 사용)
DB_PATH=data/storage/stock_analysis.db

# 경로 재정의 (기본값 그대로 사용 권장)
# KOREANSTOCKS_BASE_DIR=...           # 데이터 루트 강제 지정 (미설정 시 자동 탐지)
#   - editable install (pip install -e .): pyproject.toml 기준 프로젝트 루트
#   - PyPI 전역 설치: ~/.koreanstocks/ 자동 생성·사용
# KOREANSTOCKS_GITHUB_DB_URL=...      # sync 다운로드 URL (저장소 fork 시에만 변경)
```

## 코딩 규칙

- **Error Handling:** 데이터 크롤링 및 API 호출마다 try/except + 로그 필수.
- **Type Hinting:** 함수 시그니처에 타입 힌트 적극 사용.
- **Docstring:** 새 에이전트·유틸리티 함수에 docstring 작성.
- **LLM 비용:** GPT 호출 시 `max_tokens` 제한, 필요 정보만 포함한 프롬프트 유지.
- **ML 모델 경로:** 절대 경로 사용 (`pathlib.Path(__file__).parent` 기준).
- **모델-스케일러 무결성:** 모델 로드 시 반드시 대응하는 스케일러도 함께 로드.

## /techdebt 전용 아키텍처 규칙

글로벌 `/techdebt` skill이 이 섹션을 읽어 KoreanStocks 전용 검사를 추가로 수행합니다.

### 아키텍처 경계
- `src/koreanstocks/core/` 파일에 `import streamlit` 또는 `st.` 호출이 있으면 🔴 High (UI/Core 커플링 위반)
- `src/koreanstocks/core/` 파일이 `src/koreanstocks/api/`를 직접 import하면 🔴 High (역방향 의존성 위반)

### ML 모델 무결성
- 모델 파일(`.pkl`) 로드 시 대응 스케일러를 함께 로드하지 않으면 🔴 High
- 모델 경로가 하드코딩(`"models/saved/..."`)이면 🟡 Medium — `pathlib.Path(__file__).parent` 사용 필수
- `train_models.py` 실행 결과와 `models/saved/model_params/*.json` 불일치 시 🟡 Medium

### LLM 비용 리스크
- `news_agent.py`, `analysis_agent.py`, `recommendation_agent.py`의 GPT 호출에 `max_tokens` 없으면 🔴 High
- 종목 루프 안에서 GPT를 개별 호출하면 🟡 Medium (배치 처리 검토)

### 자동 수정 금지 대상 (Manual Only)
- 종합 점수 가중치 (`tech×0.40 + ml×0.35 + sentiment_norm×0.25`) 변경
- ML 피처 목록 변경 (모델 재학습 필요)
- GitHub Actions 스케줄 변경
