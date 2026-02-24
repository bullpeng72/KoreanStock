import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Project Root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
    NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
    
    # Database
    DB_PATH = os.getenv("DB_PATH", "data/storage/stock_analysis.db")
    
    # Model Settings
    DEFAULT_MODEL = "gpt-4o-mini"
    
    # Trading Settings
    TRANSACTION_FEE = 0.00015  # 0.015%
    TAX_RATE = 0.0018         # 0.18%
    
    # Cache Settings
    CACHE_EXPIRE_STOCKS = 1800  # 30 mins
    CACHE_EXPIRE_MARKET = 300   # 5 mins

config = Config()
