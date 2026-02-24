import sqlite3
import pandas as pd
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from core.config import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """SQLite 데이터베이스 관리를 담당하는 클래스 (Singleton)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.db_path = config.DB_PATH
            cls._instance._ensure_db_dir()
            cls._instance.init_db()
        return cls._instance

    def _ensure_db_dir(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def get_connection(self):
        return sqlite3.connect(self.db_path, timeout=30.0)

    def init_db(self):
        """필요한 테이블들을 생성"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. 종목 정보 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stocks (
                    code TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    market TEXT NOT NULL,
                    sector TEXT,
                    industry TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 2. 주가 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_prices (
                    code TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    change REAL,
                    PRIMARY KEY (code, date),
                    FOREIGN KEY (code) REFERENCES stocks(code)
                )
            ''')
            
            # 3. 추천 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT NOT NULL,
                    type TEXT NOT NULL, -- 'BUY', 'SELL', 'HOLD'
                    score REAL,
                    reason TEXT,
                    target_price REAL,
                    stop_loss REAL,
                    source TEXT, -- 'ML_ENSEMBLE', 'AI_AGENT'
                    detail_json TEXT, -- 전체 분석 결과 JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (code) REFERENCES stocks(code)
                )
            ''')
            # 기존 테이블 마이그레이션
            for migration in [
                "ALTER TABLE recommendations ADD COLUMN detail_json TEXT",
                "ALTER TABLE recommendations ADD COLUMN session_date DATE",
            ]:
                try:
                    cursor.execute(migration)
                except Exception:
                    pass  # 이미 존재하면 무시
            
            # 4. 백테스트 결과 테이블 (Phase 1 강화)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    total_return REAL,
                    win_rate REAL,
                    mdd REAL,
                    sharpe_ratio REAL,
                    start_date DATE,
                    end_date DATE,
                    parameters TEXT, -- JSON string
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 5. 관심 종목 테이블 (Phase 4 신규)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS watchlist (
                    code TEXT PRIMARY KEY,
                    name TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 6. 분석 이력 테이블 (Timeline용)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT NOT NULL,
                    tech_score REAL,
                    ml_score REAL,
                    sentiment_score REAL,
                    action TEXT,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (code) REFERENCES stocks(code)
                )
            ''')

    def save_analysis_history(self, res: Dict):
        """분석 결과 이력 저장"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analysis_history (code, tech_score, ml_score, sentiment_score, action, summary)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                res['code'], 
                res['tech_score'], 
                res['ml_score'], 
                res['sentiment_score'],
                res['ai_opinion']['action'],
                res['ai_opinion']['summary']
            ))
            conn.commit()

    def get_analysis_history(self, code: str, limit: int = 5) -> List[Dict]:
        """특정 종목의 최근 분석 이력 조회"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT tech_score, ml_score, sentiment_score, action, summary, created_at 
                FROM analysis_history 
                WHERE code = ? 
                ORDER BY created_at DESC LIMIT ?
            ''', (code, limit))
            rows = cursor.fetchall()
            return [
                {
                    'tech_score': r[0], 'ml_score': r[1], 'sentiment_score': r[2],
                    'action': r[3], 'summary': r[4], 'date': r[5]
                } for r in rows
            ]

    def get_recommendations_by_date(self, date_str: str) -> List[Dict]:
        """특정 날짜의 추천 종목 목록 반환 (detail_json 우선, 없으면 기본 구조로 폴백)"""
        import json
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT r.code, s.name, r.type, r.score, r.reason,
                       r.target_price, r.created_at, r.detail_json
                FROM recommendations r
                LEFT JOIN stocks s ON r.code = s.code
                WHERE r.session_date = ?
                ORDER BY r.score DESC
            ''', (date_str,))
            rows = cursor.fetchall()

        result = []
        for code, name, action, score, reason, target_price, created_at, detail_json in rows:
            if detail_json:
                try:
                    result.append(json.loads(detail_json))
                    continue
                except Exception:
                    pass
            # 폴백: detail_json 없는 구버전 데이터
            result.append({
                'code': code, 'name': name or code,
                'current_price': 0, 'change_pct': 0,
                'tech_score': score, 'ml_score': score, 'sentiment_score': 0,
                'sentiment_info': {}, 'indicators': {}, 'stats': {},
                'ai_opinion': {
                    'action': action, 'summary': reason or '',
                    'reasoning': '', 'target_price': target_price or 0,
                    'target_rationale': '', 'strength': '', 'weakness': '',
                },
            })
        return result

    def get_recommendation_dates(self, limit: int = 30) -> List[str]:
        """추천 데이터가 존재하는 날짜 목록 반환 (최근순)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT session_date
                FROM recommendations
                WHERE session_date IS NOT NULL
                ORDER BY session_date DESC
                LIMIT ?
            ''', (limit,))
            return [r[0] for r in cursor.fetchall()]

    def get_latest_recommendation_date(self) -> Optional[str]:
        """가장 최근 추천 날짜 반환"""
        dates = self.get_recommendation_dates(limit=1)
        return dates[0] if dates else None

    def save_stocks(self, df: pd.DataFrame):
        """종목 리스트 저장"""
        if df.empty: return
        # 중복 컬럼 방어: 스키마에 맞는 컬럼만 선택
        target_cols = ['code', 'name', 'market', 'sector', 'industry']
        save_df = df[[c for c in target_cols if c in df.columns]].copy()
        # 누락된 필수 컬럼 채우기
        for col in target_cols:
            if col not in save_df.columns:
                save_df[col] = ''
        with self.get_connection() as conn:
            save_df.to_sql('stocks', conn, if_exists='replace', index=False)

    def save_prices(self, code: str, df: pd.DataFrame):
        """주가 데이터 저장"""
        if df.empty: return
        df = df.copy()
        df['code'] = code
        df = df.reset_index()
        with self.get_connection() as conn:
            df.to_sql('stock_prices', conn, if_exists='append', index=False, 
                      method=None, chunksize=1000)

    def get_prices(self, code: str, start: str = None, end: str = None) -> pd.DataFrame:
        """저장된 주가 데이터 조회"""
        query = "SELECT * FROM stock_prices WHERE code = ?"
        params = [code]
        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)
        query += " ORDER BY date"
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            return df

    def add_to_watchlist(self, code: str, name: str):
        """관심 종목 추가"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT OR REPLACE INTO watchlist (code, name) VALUES (?, ?)', (code, name))
            conn.commit()

    def remove_from_watchlist(self, code: str):
        """관심 종목 삭제"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM watchlist WHERE code = ?', (code,))
            conn.commit()

    def get_watchlist(self) -> List[Dict]:
        """관심 종목 리스트 조회"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT code, name FROM watchlist ORDER BY added_at DESC')
            rows = cursor.fetchall()
            return [{'code': row[0], 'name': row[1]} for row in rows]

# Singleton instance
db_manager = DatabaseManager()
