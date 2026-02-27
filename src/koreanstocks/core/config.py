import os
from dotenv import load_dotenv

load_dotenv()


def _resolve_base_dir() -> str:
    """저장소 루트 결정.

    우선순위:
    1) KOREANSTOCKS_BASE_DIR 환경변수 (임의 경로 지정 시)
    2) __file__ 기준 4단계 상위에 pyproject.toml이 있으면 프로젝트 루트
       (editable install: src/koreanstocks/core/ → src/koreanstocks/ → src/ → 루트/)
    3) ~/.koreanstocks/ — PyPI 전역 설치 시 사용자 홈 디렉토리
    """
    from_env = os.getenv("KOREANSTOCKS_BASE_DIR")
    if from_env:
        return os.path.abspath(from_env)

    candidate = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    if os.path.isfile(os.path.join(candidate, "pyproject.toml")):
        return candidate

    # PyPI 전역 설치: site-packages 구조이므로 사용자 홈 디렉토리로 fallback
    home_base = os.path.join(os.path.expanduser("~"), ".koreanstocks")
    os.makedirs(home_base, exist_ok=True)
    return home_base


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
    
    # GitHub DB 동기화 URL (koreanstocks sync 명령용)
    # 저장소를 포크했거나 private인 경우 KOREANSTOCKS_GITHUB_DB_URL 환경변수로 재정의
    GITHUB_RAW_DB_URL: str = os.getenv(
        "KOREANSTOCKS_GITHUB_DB_URL",
        "https://raw.githubusercontent.com/bullpeng72/KoreanStock/main/data/storage/stock_analysis.db",
    )

    # Cache Settings
    CACHE_EXPIRE_STOCKS = 1800  # 30 mins
    CACHE_EXPIRE_MARKET = 300   # 5 mins

    # Market Constants
    TRADING_DAYS_PER_YEAR = 252

config = Config()
