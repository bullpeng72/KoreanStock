import json
import pandas as pd
import logging
from datetime import date
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from core.data.provider import data_provider
from core.engine.analysis_agent import analysis_agent
from core.data.database import db_manager

logger = logging.getLogger(__name__)


def _composite_score(x: Dict[str, Any]) -> float:
    """3-way 가중합 composite 점수 산출.

    ML 모델 활성 여부에 따라 가중치를 동적으로 조정:
      - 모델 있음: tech 0.40 + ml 0.35 + sentiment 0.25
      - 모델 없음 (fallback): tech 0.65 + sentiment 0.35
        (ML이 tech_score 복사본이면 이중 가중을 피하기 위해 ml 제외)
    """
    sentiment_raw  = x.get('sentiment_score', 0)
    sentiment_norm = max(0.0, min(100.0, (sentiment_raw + 100.0) / 2.0))
    ml_count = x.get('ml_model_count', 0)

    if ml_count == 0:
        # ML 모델 없음 — tech 위주
        return x.get('tech_score', 50.0) * 0.65 + sentiment_norm * 0.35
    return (
        x.get('tech_score', 50.0) * 0.40
        + x.get('ml_score',   50.0) * 0.35
        + sentiment_norm             * 0.25
    )


class RecommendationAgent:
    """분석된 데이터를 바탕으로 투자 종목을 추천하는 에이전트"""

    def get_recommendations(self, limit: int = 5, market: str = 'ALL', theme_keywords: List[str] = None, theme_label: str = '전체') -> List[Dict[str, Any]]:
        """유망 종목 추천 리스트 생성 (테마 및 시장 필터 적용)"""
        logger.info(f"Generating recommendations (Market: {market}, Theme: {theme_label})...")

        # 1. 후보군 코드 선정
        if theme_keywords:
            theme_df = data_provider.get_stocks_by_theme(theme_keywords, market)
            theme_codes = set(theme_df['code'].tolist())
            # 거래량 상위 랭킹 순서로 정렬하여 유동성 높은 종목 우선 분석 (market 필터 적용)
            ranked_codes = data_provider.get_market_ranking(limit=200, market=market)
            candidate_codes = [c for c in ranked_codes if c in theme_codes]
            # 랭킹에 없는 테마 종목은 뒤에 추가
            candidate_codes += [c for c in theme_df['code'].tolist() if c not in set(candidate_codes)]
        else:
            # get_market_ranking() 내부에서 market 필터 적용 — 별도 post-filter 불필요
            candidate_codes = data_provider.get_market_ranking(limit=100, market=market)

        if not candidate_codes:
            return []

        # 2. 종목명 매칭을 위한 전체 리스트 확보
        stock_list = data_provider.get_stock_list()

        # 3. 선별된 후보군 분석 (최대 30개 종목, 병렬 실행)
        target_codes = candidate_codes[:30]
        candidates = []
        for code in target_codes:
            stock_info = stock_list[stock_list['code'] == code]
            nm = stock_info.iloc[0]['name'] if not stock_info.empty else code
            candidates.append((code, nm))

        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._analyze_candidate, code, nm): code
                       for code, nm in candidates}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    res = future.result(timeout=60)   # 종목당 최대 60초 대기
                    if res is not None:
                        results.append(res)
                except FuturesTimeoutError:
                    logger.warning(f"Analysis timeout: {code}, skipping")
                except Exception as e:
                    logger.warning(f"Analysis error for {code}: {e}")

        if not results:
            logger.warning("No successful analyses to recommend.")
            return []

        # 4. composite 점수로 정렬
        results.sort(key=_composite_score, reverse=True)

        # 5. DB에 추천 결과 저장 (theme_label / market 메타 포함)
        final_recs = results[:limit]
        for rec in final_recs:
            rec['theme'] = theme_label
            rec['analysis_market'] = market
        self._save_to_db(final_recs)

        # 6. composite_score를 rec dict에 추가 (알림 표시용)
        for rec in final_recs:
            rec['composite_score'] = round(_composite_score(rec), 2)

        return final_recs

    def _analyze_candidate(self, code: str, name: str) -> Optional[Dict[str, Any]]:
        """단일 종목 분석 — ThreadPoolExecutor 워커에서 호출"""
        try:
            analysis = analysis_agent.analyze_stock(code, name)
            return analysis if "error" not in analysis else None
        except Exception as e:
            logger.warning(f"Analysis failed for {code} ({name}): {e}")
            return None

    def _save_to_db(self, recommendations: List[Dict]):
        """추천 결과를 날짜별로 저장 (동일 날짜+종목은 덮어쓰기)"""
        session_date = date.today().isoformat()
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            for rec in recommendations:
                composite = round(_composite_score(rec), 2)
                try:
                    detail_json = json.dumps(rec, ensure_ascii=False, default=str)
                except Exception as e:
                    logger.warning(f"JSON serialization failed for {rec.get('code', '?')}: {e}")
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
                    rec.get('ai_opinion', {}).get('action', 'HOLD'),
                    composite,
                    rec.get('ai_opinion', {}).get('summary', ''),
                    rec.get('ai_opinion', {}).get('target_price', 0),
                    'AI_RECOMMENDER_V2',
                    detail_json,
                    session_date,
                ))
            conn.commit()
        logger.info(f"Saved {len(recommendations)} recommendations for {session_date}")


recommendation_agent = RecommendationAgent()
