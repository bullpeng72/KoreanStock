import os
from dotenv import load_dotenv

load_dotenv()


def _resolve_base_dir() -> str:
    """저장소 루트 결정.

    우선순위:
    1) KOREANSTOCKS_BASE_DIR 환경변수 (전역 설치 또는 임의 경로 사용 시)
    2) __file__ 기준 3단계 상위 (editable install: src/koreanstocks/core/ → 저장소 루트)
    """
    from_env = os.getenv("KOREANSTOCKS_BASE_DIR")
    if from_env:
        return os.path.abspath(from_env)
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


class Config:
    # Version
    VERSION = "0.2.3"

    # Project Root
    # - editable install (pip install -e .): __file__ 기준 자동 탐지
    # - 전역 설치 또는 경로 오류 시: .env에 KOREANSTOCKS_BASE_DIR=/path/to/project 설정
    BASE_DIR = _resolve_base_dir()

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
    NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
    DART_API_KEY = os.getenv("DART_API_KEY", "")

    # Database — 상대 경로는 BASE_DIR 기준 절대 경로로 변환 (CWD 의존 방지)
    _db_raw = os.getenv(
        "DB_PATH",
        os.path.join(BASE_DIR, "data", "storage", "stock_analysis.db"),
    )
    DB_PATH = _db_raw if os.path.isabs(_db_raw) else os.path.join(BASE_DIR, _db_raw)
    
    # Model Settings
    DEFAULT_MODEL = "gpt-4o-mini"
    
    # Trading Settings
    TRANSACTION_FEE = 0.00015  # 0.015%
    TAX_RATE = 0.0018         # 0.18%
    
    # Cache Settings
    CACHE_EXPIRE_STOCKS = 1800  # 30 mins
    CACHE_EXPIRE_MARKET = 300   # 5 mins

    # Market Constants
    TRADING_DAYS_PER_YEAR = 252

config = Config()
