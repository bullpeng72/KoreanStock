import json
import pandas as pd
import logging
from datetime import date
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from koreanstocks.core.data.provider import data_provider
from koreanstocks.core.engine.analysis_agent import analysis_agent
from koreanstocks.core.data.database import db_manager

logger = logging.getLogger(__name__)

# 버킷 메타데이터 (이름 → 한국어 레이블)
BUCKET_LABELS: Dict[str, str] = {
    'volume':   '거래량 상위',
    'momentum': '상승 모멘텀',
    'rebound':  '반등 후보',
}

# 버킷별 분석 풀 비율
_BUCKET_RATIOS = [
    ('volume',   0.40),
    ('momentum', 0.35),
    ('rebound',  0.25),
]


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
        return x.get('tech_score', 50.0) * 0.65 + sentiment_norm * 0.35
    return (
        x.get('tech_score', 50.0) * 0.40
        + x.get('ml_score',   50.0) * 0.35
        + sentiment_norm             * 0.25
    )


def _apply_bucket_quota(
    results: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """버킷별 쿼터를 보장하며 최종 종목 선정.

    각 버킷에서 비율에 따라 우선 선정한 뒤,
    미달 버킷의 슬롯은 전체 점수 순으로 보충한다.
    섹터 편중도 버킷 내에서 max_per_sector(~33%)로 제한.
    """
    max_per_sector = max(1, round(limit / 3))

    # 버킷별 쿼터 계산 (합계가 limit이 되도록 조정)
    quotas: Dict[str, int] = {}
    assigned = 0
    for i, (bucket_name, ratio) in enumerate(_BUCKET_RATIOS):
        if i < len(_BUCKET_RATIOS) - 1:
            q = max(1, round(limit * ratio))
            quotas[bucket_name] = q
            assigned += q
        else:
            quotas[bucket_name] = max(1, limit - assigned)

    selected: List[Dict[str, Any]] = []
    selected_codes: set = set()
    sector_count: Dict[str, int] = {}

    def _pick(candidates: List[Dict[str, Any]], quota: int) -> List[Dict[str, Any]]:
        """candidates 중 쿼터만큼 섹터 다양성을 고려해 선정."""
        picked: List[Dict[str, Any]] = []
        deferred: List[Dict[str, Any]] = []
        for rec in candidates:
            if rec['code'] in selected_codes:
                continue
            sector = (rec.get('sector') or '').strip()
            if not sector or sector_count.get(sector, 0) < max_per_sector:
                picked.append(rec)
                if sector:
                    sector_count[sector] = sector_count.get(sector, 0) + 1
            else:
                deferred.append(rec)
            if len(picked) >= quota:
                break
        # 섹터 한도로 미달이면 보충
        if len(picked) < quota:
            for rec in deferred:
                if rec['code'] not in selected_codes and rec not in picked:
                    picked.append(rec)
                    if len(picked) >= quota:
                        break
        return picked

    # 버킷 순서대로 쿼터 선정
    for bucket_name, quota in quotas.items():
        bucket_results = sorted(
            [r for r in results if r.get('bucket') == bucket_name],
            key=_composite_score, reverse=True,
        )
        picks = _pick(bucket_results, quota)
        selected.extend(picks)
        selected_codes.update(r['code'] for r in picks)

    # 전체 limit 미달 시 남은 종목으로 보충 (점수 순)
    if len(selected) < limit:
        remaining = sorted(
            [r for r in results if r['code'] not in selected_codes],
            key=_composite_score, reverse=True,
        )
        picks = _pick(remaining, limit - len(selected))
        selected.extend(picks)

    return selected[:limit]


class RecommendationAgent:
    """분석된 데이터를 바탕으로 투자 종목을 추천하는 에이전트"""

    def get_recommendations(
        self,
        limit: int = 5,
        market: str = 'ALL',
        theme_keywords: List[str] = None,
        theme_label: str = '전체',
    ) -> List[Dict[str, Any]]:
        """유망 종목 추천 리스트 생성 (버킷 기반 후보군 구성)

        버킷 구성:
          volume   (40%) — 거래량 상위 유동성 안정주
          momentum (35%) — 상승 모멘텀 (+2%~+15%)
          rebound  (25%) — 거래량 상위 중 하락 반등 후보

        분석 풀: min(limit * 8, 80)  →  limit=5: 40개, limit=10: 80개
        """
        logger.info(f"Generating recommendations (Market: {market}, Theme: {theme_label})...")

        # 종목명 매칭용 전체 리스트 (루프 밖 1회)
        stock_list = data_provider.get_stock_list()

        # ── 1. 후보군 코드 선정 ─────────────────────────────────────
        if theme_keywords:
            # 테마 지정 시: 테마 종목 + 거래량 랭킹 교집합 우선, 나머지 추가
            theme_df    = data_provider.get_stocks_by_theme(theme_keywords, market)
            theme_codes = set(theme_df['code'].tolist())
            ranked      = data_provider.get_market_ranking(limit=200, market=market)
            candidate_codes = [c for c in ranked if c in theme_codes]
            candidate_codes += [c for c in theme_df['code'].tolist() if c not in set(candidate_codes)]
            # 테마 모드: 버킷 없이 단일 풀로 처리 (bucket='volume' 태그)
            code_bucket: Dict[str, str] = {c: 'volume' for c in candidate_codes}
        else:
            # 버킷 기반 후보군 구성
            buckets = data_provider.get_market_buckets(market)
            total_pool = min(limit * 8, 80)
            code_bucket = {}
            seen: set = set()

            for bucket_name, ratio in _BUCKET_RATIOS:
                pool_size = max(2, round(total_pool * ratio))
                count = 0
                for code in buckets.get(bucket_name, []):
                    if code not in seen and count < pool_size:
                        code_bucket[code] = bucket_name
                        seen.add(code)
                        count += 1

            candidate_codes = list(code_bucket.keys())

        if not candidate_codes:
            return []

        # ── 2. 종목명 해석 ──────────────────────────────────────────
        candidates: List[Tuple[str, str]] = []
        for code in candidate_codes:
            row = stock_list[stock_list['code'] == code]
            nm = row.iloc[0]['name'] if not row.empty else code
            candidates.append((code, nm))

        # ── 3. 병렬 분석 ────────────────────────────────────────────
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._analyze_candidate, code, nm): code
                for code, nm in candidates
            }
            for future in as_completed(futures):
                code = futures[future]
                try:
                    res = future.result(timeout=60)
                    if res is not None:
                        res['bucket'] = code_bucket.get(code, 'volume')
                        results.append(res)
                except FuturesTimeoutError:
                    logger.warning(f"Analysis timeout: {code}, skipping")
                except Exception as e:
                    logger.warning(f"Analysis error for {code}: {e}")

        if not results:
            logger.warning("No successful analyses to recommend.")
            return []

        # ── 4. 버킷 쿼터 + 섹터 다양성으로 최종 선정 ───────────────
        final_recs = _apply_bucket_quota(results, limit)
        for rec in final_recs:
            rec['theme']            = theme_label
            rec['analysis_market']  = market
            rec['composite_score']  = round(_composite_score(rec), 2)
            rec['bucket_label']     = BUCKET_LABELS.get(rec.get('bucket', 'volume'), '')
        self._save_to_db(final_recs)

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
