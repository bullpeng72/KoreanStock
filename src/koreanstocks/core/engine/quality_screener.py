"""
우량주 스크리너
===============
펀더멘털 기반 장기(6개월~수년) 우량주 발굴.

가치주 스크리너(value_screener.py)와의 차이:
  - 철학  : 저평가 발굴(가치주) → 수익성·성장성 높은 우량 기업(우량주)
  - PER   : 상한 없음 — 비싸도 좋은 기업이면 포함
  - 핵심  : 고ROE + 고영업이익률 + 안정 성장 + 낮은 부채
  - 정렬  : quality_score 내림차순 (ROE·영업이익률·성장·부채·배당 가중합)

필터 체인:
  1. 영업이익 흑자 필수
  2. 영업이익률 ≥ op_margin_min  (저마진 기업 제거)
  3. ROE ≥ roe_min               (자본 효율 하한 — 가치주 8% → 12%)
  4. 영업이익 YoY ≥ yoy_min      (이익 역성장 제거)
  5. 부채비율 ≤ debt_max         (가치주 150% → 100%)
  6. PBR ≤ pbr_max               (느슨한 상한 — 성장 우량주 포함)

점수:
  quality_score (0~100) — ROE·영업이익률·영업이익YoY·부채비율·배당 가중합
"""

import logging
from datetime import date as _date
from typing import Dict, List, Optional, Tuple

from koreanstocks.core.data.fundamental_provider import fundamental_provider
from koreanstocks.core.data.provider import data_provider

logger = logging.getLogger(__name__)

# ── 기본 필터 임계값 ─────────────────────────────────────────────
DEFAULT_QUALITY_FILTERS = {
    "roe_min":          12.0,   # ROE 하한 (% — 가치주 8%보다 높음)
    "op_margin_min":    10.0,   # 영업이익률 하한 (% — 핵심 차별 기준)
    "yoy_min":         -10.0,   # 영업이익 YoY 하한 (% — 소폭 역성장 허용, 사이클 업종 대응)
    "debt_max":        100.0,   # 부채비율 상한 (% — 가치주 150% → 100%)
    "pbr_max":           6.0,   # PBR 상한 (성장주 포함을 위해 느슨하게)
}


# ── 우량 점수 ─────────────────────────────────────────────────────

def quality_score(f: Dict) -> float:
    """
    우량 점수 산출 (0~100).

    구성:
      ROE          30pt  높을수록 유리 (20%에서 만점) — 2개년 평균 사용 시 더 정확
      영업이익률    25pt  높을수록 유리 (20%에서 만점)
      영업이익 YoY  20pt  높을수록 유리 (30%에서 만점 — 변별력 강화)
      부채비율      15pt  낮을수록 유리 (0%에서 만점, 100% 이상→0pt)
      배당수익률    10pt  항상 포함 (무배당=0pt, 3%=만점) — 데이터 없음도 0pt 처리

    Note: 배당 항목은 데이터 없음을 0으로 처리하여 전 종목을 100pt 기준으로 비교.
    """
    parts: List[Tuple[float, float]] = []   # (earned, possible)

    roe = f.get("roe")
    if roe is not None:
        parts.append((min(30.0, max(0.0, roe / 20 * 30)), 30.0))

    op_margin = f.get("op_margin")
    if op_margin is not None:
        parts.append((min(25.0, max(0.0, op_margin / 20 * 25)), 25.0))

    yoy = f.get("op_income_yoy")
    if yoy is not None:
        # Fix 4: 만점 기준 20% → 30% (변별력 강화)
        parts.append((min(20.0, max(0.0, yoy / 30 * 20)), 20.0))

    debt = f.get("debt_ratio")
    if debt is not None:
        parts.append((max(0.0, 15 * (1 - min(debt, 100) / 100)), 15.0))

    # Fix 3: 배당 None → 0pt 처리 (무배당 vs 데이터 없음 혼동 제거, 전 종목 동일 기준)
    div = f.get("dividend_yield") or 0.0
    parts.append((min(10.0, max(0.0, div / 3 * 10)), 10.0))

    if not parts:
        return 50.0

    earned   = sum(p[0] for p in parts)
    possible = sum(p[1] for p in parts)
    return round(earned / possible * 100, 1)


# ── 스크리너 ─────────────────────────────────────────────────────

class QualityScreener:
    """
    펀더멘털 기반 우량주 스크리너.

    실행 흐름:
      1. 시가총액 상위 후보군 + ROE 사전 필터 (PER 상한 없음)
      2. 펀더멘털 병렬 수집 (fundamental_provider, 당일 SQLite 캐시)
      3. 하드 필터 통과 (영업이익률·ROE·YoY·부채비율·PBR)
      4. quality_score 내림차순 정렬
      5. 상위 limit개 반환

    캐시:
      동일 필터 조합의 스크리닝 결과를 당일(자정까지) 인메모리 캐시로 보관.
    """

    def __init__(self):
        self._cache: Dict[tuple, List[Dict]] = {}
        self._cache_date: Optional[_date] = None

    def screen(
        self,
        market: str = "ALL",
        roe_min: float = DEFAULT_QUALITY_FILTERS["roe_min"],
        op_margin_min: float = DEFAULT_QUALITY_FILTERS["op_margin_min"],
        yoy_min: float = DEFAULT_QUALITY_FILTERS["yoy_min"],
        debt_max: float = DEFAULT_QUALITY_FILTERS["debt_max"],
        pbr_max: float = DEFAULT_QUALITY_FILTERS["pbr_max"],
        candidate_limit: int = 200,
        limit: int = 20,
    ) -> List[Dict]:
        """
        우량주 스크리닝 실행.

        Args:
            market: ALL | KOSPI | KOSDAQ
            roe_min: ROE 하한 (%)
            op_margin_min: 영업이익률 하한 (%)
            yoy_min: 영업이익 YoY 하한 (%)
            debt_max: 부채비율 상한 (%)
            pbr_max: PBR 상한
            candidate_limit: 시가총액 상위 탐색 종목 수
            limit: 최종 반환 종목 수

        Returns:
            quality_score 내림차순 정렬된 Dict 리스트
        """
        # ── 당일 인메모리 캐시 확인 ──────────────────────────────────
        today = _date.today()
        if self._cache_date != today:
            self._cache.clear()
            self._cache_date = today

        cache_key = (market, roe_min, op_margin_min, yoy_min,
                     debt_max, pbr_max, candidate_limit, limit)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            logger.info(f"[QUALITY] 캐시 히트 — {len(cached)}종목 즉시 반환 (당일 동일 조건)")
            return [dict(r) for r in cached]

        logger.info(f"[QUALITY] 스크리닝 시작 (market={market}, limit={limit})")

        # 1. 후보 종목 — PER 상한 없음, ROE 사전 필터만 적용
        candidates = data_provider.get_value_candidates(
            limit=candidate_limit,
            market=market,
            per_max=9999,                          # PER 제한 없음
            roe_min=max(0.0, roe_min - 5),         # 관용 하한 (DART 재확인)
        )
        if not candidates:
            logger.warning("[QUALITY] 후보 종목 없음 — 스크리닝 중단")
            return []

        candidates = candidates[:candidate_limit]
        stock_list = data_provider.get_stock_list()
        logger.info(f"[QUALITY] 후보 {len(candidates)}종목 펀더멘털 수집 중...")

        # 2. 펀더멘털 병렬 수집
        fund_map = fundamental_provider.get_fundamentals_batch(candidates, max_workers=15)

        # 3. 필터 + 점수 산출
        passed: List[Dict] = []
        skipped_no_data = 0
        skipped_filter  = 0

        for code in candidates:
            f = fund_map.get(code, {})

            # Fix 5: ROE 2개년 평균 사용 (지속성 반영) — prev가 없으면 단일 연도
            roe_cur  = f.get("roe")
            roe_prev = f.get("roe_prev")
            if roe_cur is not None and roe_prev is not None:
                roe = round((roe_cur + roe_prev) / 2, 1)
            else:
                roe = roe_cur

            op_margin = f.get("op_margin")
            yoy       = f.get("op_income_yoy")
            debt      = f.get("debt_ratio")
            pbr       = f.get("pbr")
            op_pos    = f.get("op_income_positive", False)

            # 데이터 부족 → 스킵 (ROE와 op_margin 모두 없으면 판단 불가)
            if roe is None and op_margin is None:
                skipped_no_data += 1
                continue

            # ── 하드 필터 ───────────────────────────────────────────
            fail = False

            if not op_pos:
                fail = True

            # Fix 2: op_margin=None이면 핵심 기준 미확인 → 탈락
            if not fail and (op_margin is None or op_margin < op_margin_min):
                fail = True

            if not fail and roe is not None and roe < roe_min:
                fail = True

            # 영업이익 YoY (데이터 없으면 통과 허용)
            if not fail and yoy is not None and yoy < yoy_min:
                fail = True

            if not fail and debt is not None and debt > debt_max:
                fail = True

            if not fail and pbr is not None and pbr > pbr_max:
                fail = True

            if fail:
                skipped_filter += 1
                continue

            # 종목 메타
            row  = stock_list[stock_list["code"] == code]
            name = row.iloc[0]["name"]             if not row.empty else code
            mkt  = row.iloc[0].get("market", "")  if not row.empty else ""
            sect = row.iloc[0].get("sector", "")  if not row.empty else ""

            # Fix 5: 평균 ROE를 점수 계산에 반영
            f_for_score = {**f, "roe": roe}
            qscore = quality_score(f_for_score)

            passed.append({
                "code":           code,
                "name":           name,
                "market":         mkt,
                "sector":         sect,
                "roe":            roe,   # 평균 ROE 표시
                "op_margin":      op_margin,
                "op_income_yoy":  yoy,
                "debt_ratio":     debt,
                "pbr":            pbr,
                "dividend_yield": f.get("dividend_yield"),
                "quality_score":  qscore,
                "fundamentals":   {k: v for k, v in f.items() if k != "code"},
            })

        # 4. quality_score 내림차순 정렬
        passed.sort(key=lambda x: x["quality_score"], reverse=True)
        result = passed[:limit]

        logger.info(
            f"[QUALITY] 완료: 후보 {len(candidates)} → "
            f"데이터없음 {skipped_no_data} → "
            f"필터탈락 {skipped_filter} → "
            f"통과 {len(passed)} → 최종 {len(result)}"
        )

        self._cache[cache_key] = [dict(r) for r in result]
        return result

    def get_filter_defaults(self) -> Dict:
        """현재 기본 필터 임계값 반환 (API 노출용)."""
        return dict(DEFAULT_QUALITY_FILTERS)


quality_screener = QualityScreener()
