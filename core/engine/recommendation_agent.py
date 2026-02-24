import json
import pandas as pd
import logging
from datetime import date
from typing import List, Dict, Any
from core.data.provider import data_provider
from core.engine.analysis_agent import analysis_agent
from core.data.database import db_manager

logger = logging.getLogger(__name__)

class RecommendationAgent:
    """분석된 데이터를 바탕으로 투자 종목을 추천하는 에이전트"""

    def get_recommendations(self, limit: int = 5, market: str = 'ALL', theme_keywords: List[str] = None) -> List[Dict[str, Any]]:
        """유망 종목 추천 리스트 생성 (테마 및 시장 필터 적용)"""
        logger.info(f"Generating recommendations (Market: {market}, Theme: {theme_keywords})...")
        
        # 1. 후보군 코드 선정
        if theme_keywords:
            # 테마 키워드가 있을 경우 관련 종목 추출 후 시장 랭킹으로 정렬
            theme_df = data_provider.get_stocks_by_theme(theme_keywords, market)
            theme_codes = set(theme_df['code'].tolist())
            # 거래량 상위 랭킹 순서로 정렬하여 유동성 높은 종목 우선 분석
            ranked_codes = data_provider.get_market_ranking(limit=200)
            candidate_codes = [c for c in ranked_codes if c in theme_codes]
            # 랭킹에 없는 테마 종목은 뒤에 추가
            candidate_codes += [c for c in theme_df['code'].tolist() if c not in set(candidate_codes)]
        else:
            # 없을 경우 시장 랭킹 기반
            candidate_codes = data_provider.get_market_ranking(limit=50)
            if market != 'ALL':
                # 시장 필터링 적용 (StockListing에서 가져온 데이터 활용)
                stock_list = data_provider.get_stock_list()
                candidate_codes = stock_list[(stock_list['code'].isin(candidate_codes)) & (stock_list['market'] == market)]['code'].tolist()

        if not candidate_codes:
            return []

        # 2. 종목명 매칭을 위한 전체 리스트 확보
        stock_list = data_provider.get_stock_list()
        
        results = []
        # 3. 선별된 후보군 분석 (최대 20개 종목)
        for code in candidate_codes[:20]: 
            stock_info = stock_list[stock_list['code'] == code]
            name = stock_info.iloc[0]['name'] if not stock_info.empty else code
            
            try:
                analysis = analysis_agent.analyze_stock(code, name)
                if "error" not in analysis:
                    results.append(analysis)
            except Exception as e:
                continue
                
        if not results:
            logger.warning("No successful analyses to recommend.")
            return []

        # 4. 종합 점수순 정렬 후 반환
        # sentiment_score(-100~100)를 0~100 스케일로 정규화 후 가중 합산
        def composite(x):
            normalized_sentiment = (x['sentiment_score'] + 100) / 2
            return x['tech_score'] * 0.3 + x['ml_score'] * 0.4 + normalized_sentiment * 0.3

        results.sort(key=composite, reverse=True)
        
        # 5. DB에 추천 결과 저장
        final_recs = results[:limit]
        self._save_to_db(final_recs)
        
        return final_recs

    def _save_to_db(self, recommendations: List[Dict]):
        """추천 결과를 날짜별로 저장 (동일 날짜+종목은 덮어쓰기)"""
        session_date = date.today().isoformat()
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            for rec in recommendations:
                normalized_sentiment = (rec['sentiment_score'] + 100) / 2
                composite_score = (
                    rec['tech_score'] * 0.3
                    + rec['ml_score'] * 0.4
                    + normalized_sentiment * 0.3
                )
                try:
                    detail_json = json.dumps(rec, ensure_ascii=False, default=str)
                except Exception:
                    detail_json = None

                # 동일 날짜 + 동일 종목 기존 데이터 삭제 후 재삽입 (UPSERT)
                cursor.execute(
                    'DELETE FROM recommendations WHERE code = ? AND session_date = ?',
                    (rec['code'], session_date)
                )
                cursor.execute('''
                    INSERT INTO recommendations
                        (code, type, score, reason, target_price, source, detail_json, session_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    rec['code'],
                    rec['ai_opinion'].get('action', 'HOLD'),
                    round(composite_score, 2),
                    rec['ai_opinion'].get('summary', ''),
                    rec['ai_opinion'].get('target_price', 0),
                    'AI_RECOMMENDER_V1',
                    detail_json,
                    session_date,
                ))
            conn.commit()
        logger.info(f"Saved {len(recommendations)} recommendations for {session_date}")

recommendation_agent = RecommendationAgent()
