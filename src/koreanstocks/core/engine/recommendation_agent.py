import json
import logging
from collections import Counter
from datetime import date
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from koreanstocks.core.data.provider import data_provider
from koreanstocks.core.engine.analysis_agent import analysis_agent
from koreanstocks.core.data.database import db_manager
from koreanstocks.core.constants import BUCKET_DEFAULT, BUCKET_LABELS, BUCKET_RATIOS as _BUCKET_RATIOS

logger = logging.getLogger(__name__)


def _composite_score(x: Dict[str, Any]) -> float:
    """3-way 가중합 composite 점수 산출.

    ML 모델 활성 여부에 따라 가중치를 동적으로 조정:
      - 모델 있음: tech 0.40 + ml 0.35 + sentiment 0.25
      - 모델 없음 (fallback): tech 0.65 + sentiment 0.35
        (ML이 tech_score 복사본이면 이중 가중을 피하기 위해 ml 제외)
    sorted() key 함수로 사용되므로 예외 시 0.0 반환.
    """
    try:
        _ss = x.get('sentiment_score')
        sentiment_raw  = float(_ss) if _ss is not None else 0.0
        sentiment_norm = max(0.0, min(100.0, (sentiment_raw + 100.0) / 2.0))
        _mc = x.get('ml_model_count')
        ml_count  = int(_mc) if _mc is not None else 0
        _ts = x.get('tech_score')
        _ms = x.get('ml_score')
        tech_score = float(_ts) if _ts is not None else 50.0
        ml_score   = float(_ms) if _ms is not None else 50.0

        if ml_count == 0:
            return tech_score * 0.65 + sentiment_norm * 0.35
        return (
            tech_score  * 0.40
            + ml_score  * 0.35
            + sentiment_norm * 0.25
        )
    except (TypeError, ValueError) as e:
        logger.debug(f"_composite_score 계산 오류 (code={x.get('code', '?')}): {e}")
        return 0.0


def _apply_bucket_quota(
    results: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """버킷별 쿼터를 보장하며 최종 종목 선정.

    1단계: 각 버킷 자체 후보에서 쿼터만큼 선정 (섹터 다양성 반영).
    2단계: 쿼터 미달 버킷은 교차 버킷 잉여 후보로 보충 + 버킷 재태깅
           → 최종 결과에 3개 버킷이 모두 대표되도록 보장.
    3단계: 전체 limit 미달이면 점수 순으로 보충.
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
            if rec.get('code') in selected_codes:
                continue
            sector = (rec.get('sector') or '').strip() or '__unknown__'
            cnt = sector_count.get(sector, 0)
            if cnt < max_per_sector:
                picked.append(rec)
                sector_count[sector] = cnt + 1
            else:
                deferred.append(rec)
            if len(picked) >= quota:
                break
        # 섹터 한도로 미달이면 보충
        if len(picked) < quota:
            for rec in deferred:
                if rec.get('code') not in selected_codes:
                    picked.append(rec)
                    sector = (rec.get('sector') or '').strip() or '__unknown__'
                    sector_count[sector] = sector_count.get(sector, 0) + 1  # deferred: 섹터 한도 초과 허용
                    if len(picked) >= quota:
                        break
        return picked

    # ── 1단계: 버킷별 자체 후보 선정 ──────────────────────────────
    bucket_shortfall: Dict[str, int] = {}
    for bucket_name, quota in quotas.items():
        bucket_results = sorted(
            [r for r in results if r.get('bucket') == bucket_name],
            key=_composite_score, reverse=True,
        )
        picks = _pick(bucket_results, quota)
        selected.extend(picks)
        selected_codes.update(r.get('code') for r in picks if r.get('code'))

        shortfall = quota - len(picks)
        if shortfall > 0:
            bucket_shortfall[bucket_name] = shortfall
            logger.warning(
                f"버킷 '{bucket_name}' 후보 부족: {len(picks)}/{quota} "
                f"→ {shortfall}개 교차 버킷에서 보충 예정"
            )

    # ── 2단계: 미달 버킷 교차 보충 (잉여 후보를 재태깅) ───────────
    for bucket_name, needed in bucket_shortfall.items():
        cross_pool = sorted(
            [r for r in results if r.get('code') not in selected_codes],
            key=_composite_score, reverse=True,
        )
        cross_picks = _pick(cross_pool, needed)
        for r in cross_picks:
            r['bucket'] = bucket_name  # 미달 버킷으로 재태깅
        selected.extend(cross_picks)
        selected_codes.update(r.get('code') for r in cross_picks if r.get('code'))
        if cross_picks:
            logger.info(
                f"버킷 '{bucket_name}' 교차 보충 완료: "
                f"{len(cross_picks)}종목 (재태깅)"
            )
        else:
            logger.warning(f"버킷 '{bucket_name}' 교차 보충 불가 (전체 후보 소진)")

    # ── 3단계: 전체 limit 미달 시 잔여 종목으로 보충 (점수 순) ────
    if len(selected) < limit:
        remaining = sorted(
            [r for r in results if r.get('code') not in selected_codes],
            key=_composite_score, reverse=True,
        )
        picks = _pick(remaining, limit - len(selected))
        selected.extend(picks)
        selected_codes.update(r.get('code') for r in picks if r.get('code'))

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

        최종 추천에서 3개 버킷이 모두 대표되도록 쿼터를 보장한다.
        후보 부족 버킷은 교차 버킷 잉여 종목으로 보충 후 재태깅.
        """
        logger.info(f"Generating recommendations (Market: {market}, Theme: {theme_label})...")

        # 종목명 매칭용 전체 리스트 (루프 밖 1회)
        stock_list = data_provider.get_stock_list()

        # ── 1. 후보군 코드 선정 ─────────────────────────────────────
        if theme_keywords:
            # 테마 지정 시: 테마 종목 + 거래량 랭킹 교집합 우선, 나머지 추가
            theme_df    = data_provider.get_stocks_by_theme(theme_keywords, market)
            theme_codes = set(theme_df['code'].tolist()) if 'code' in theme_df.columns else set()
            ranked      = data_provider.get_market_ranking(limit=200, market=market)
            candidate_codes = [c for c in ranked if c in theme_codes]
            existing = set(candidate_codes)
            candidate_codes += [c for c in (theme_df['code'].tolist() if 'code' in theme_df.columns else []) if c not in existing]

            # 테마 모드에서도 버킷 분류 적용
            # 거래량·등락률 기반 버킷 맵을 가져와 테마 종목에 매핑
            market_buckets = data_provider.get_market_buckets(market)
            bucket_map: Dict[str, str] = {}
            for bname, codes in market_buckets.items():
                for code in codes:
                    if code not in bucket_map:
                        bucket_map[code] = bname
            # 버킷 맵에 없는 테마 종목은 기본 버킷으로 배정
            code_bucket: Dict[str, str] = {c: bucket_map.get(c, BUCKET_DEFAULT) for c in candidate_codes}
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

        # 후보군 버킷 분포 로깅
        pool_dist = Counter(code_bucket[c] for c in candidate_codes)
        logger.info(
            f"분석 후보군 {len(candidate_codes)}종목 "
            f"(volume={pool_dist.get('volume',0)}, "
            f"momentum={pool_dist.get('momentum',0)}, "
            f"rebound={pool_dist.get('rebound',0)})"
        )

        # ── 2. 종목명 해석 ──────────────────────────────────────────
        name_map: Dict[str, str] = (
            dict(zip(stock_list['code'], stock_list['name']))
            if 'code' in stock_list.columns and 'name' in stock_list.columns
            else {}
        )
        candidates: List[Tuple[str, str]] = []
        for code in candidate_codes:
            _nm = name_map.get(code)
            nm = _nm if isinstance(_nm, str) and _nm else code
            candidates.append((code, nm))

        # ── 3. 병렬 분석 ────────────────────────────────────────────
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self._analyze_candidate, code, nm): code
                for code, nm in candidates
            }
            for future in as_completed(futures):
                code = futures[future]
                try:
                    res = future.result(timeout=60)
                    if res is not None:
                        res['bucket'] = code_bucket.get(code, BUCKET_DEFAULT)
                        results.append(res)
                except FuturesTimeoutError:
                    future.cancel()
                    logger.warning(f"Analysis timeout: {code}, skipping")
                except Exception as e:
                    logger.warning(f"Analysis error for {code}: {e}")

        if not results:
            logger.warning("No successful analyses to recommend.")
            return []

        # 분석 완료 버킷 분포 로깅
        result_dist = Counter(r.get('bucket', '?') for r in results)
        logger.info(
            f"분석 완료 {len(results)}종목 "
            f"(volume={result_dist.get('volume',0)}, "
            f"momentum={result_dist.get('momentum',0)}, "
            f"rebound={result_dist.get('rebound',0)})"
        )

        # ── 4. 버킷 쿼터 + 섹터 다양성으로 최종 선정 ───────────────
        final_recs = _apply_bucket_quota(results, limit)
        for rec in final_recs:
            rec['theme']            = theme_label
            rec['analysis_market']  = market
            rec['composite_score']  = round(_composite_score(rec), 2)
            rec['bucket_label']     = BUCKET_LABELS.get(rec.get('bucket', BUCKET_DEFAULT), '')
        self._save_to_db(final_recs)

        # 최종 버킷 분포 로깅
        final_dist = Counter(r.get('bucket', '?') for r in final_recs)
        logger.info(
            f"최종 추천 {len(final_recs)}종목 버킷 분포: "
            f"volume={final_dist.get('volume',0)}, "
            f"momentum={final_dist.get('momentum',0)}, "
            f"rebound={final_dist.get('rebound',0)}"
        )

        return final_recs

    def _analyze_candidate(self, code: str, name: str) -> Optional[Dict[str, Any]]:
        """단일 종목 분석 — ThreadPoolExecutor 워커에서 호출"""
        try:
            analysis = analysis_agent.analyze_stock(code, name)
            return analysis if analysis is not None and "error" not in analysis else None
        except Exception as e:
            logger.warning(f"Analysis failed for {code} ({name}): {e}")
            return None

    def _save_to_db(self, recommendations: List[Dict]):
        """추천 결과를 날짜별로 저장 (동일 날짜+종목은 덮어쓰기)"""
        session_date = date.today().isoformat()
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            saved_count = 0
            for i, rec in enumerate(recommendations):
                sp = f'sp_rec_{i}'
                cursor.execute(f'SAVEPOINT {sp}')
                try:
                    _cs = rec.get('composite_score')
                    composite = _cs if _cs is not None else round(_composite_score(rec), 2)
                    try:
                        detail_json = json.dumps(rec, ensure_ascii=False, default=str)
                    except Exception as e:
                        logger.warning(f"JSON serialization failed for {rec.get('code', '?')}: {e}")
                        detail_json = None
                    ai_opinion = rec.get('ai_opinion') or {}
                    try:
                        target_price = float(ai_opinion.get('target_price') or 0)
                    except (TypeError, ValueError):
                        target_price = 0.0
                    cursor.execute(
                        'DELETE FROM recommendations WHERE code = ? AND session_date = ?',
                        (rec.get('code'), session_date)
                    )
                    cursor.execute('''
                        INSERT INTO recommendations
                            (code, type, score, reason, target_price, source, detail_json, session_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        rec.get('code'),
                        ai_opinion.get('action', 'HOLD'),
                        composite,
                        ai_opinion.get('summary', ''),
                        target_price,
                        'AI_RECOMMENDER_V2',
                        detail_json,
                        session_date,
                    ))
                    cursor.execute(f'RELEASE {sp}')
                    saved_count += 1
                except Exception as e:
                    cursor.execute(f'ROLLBACK TO {sp}')
                    cursor.execute(f'RELEASE {sp}')
                    logger.warning(f"DB 저장 실패 (code={rec.get('code', '?')}): {e}, 건너뜀")
            conn.commit()
        logger.info(f"Saved {saved_count}/{len(recommendations)} recommendations for {session_date}")


recommendation_agent = RecommendationAgent()
