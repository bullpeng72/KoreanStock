"""
가치주 스크리너 (Phase 2)
===========================
펀더멘털 기반 중기(3~6개월) 가치주 발굴.

필터 체인:
  1. 영업이익 흑자 (가치함정 1차 방어)
  2. PER / PBR 상한
  3. ROE 하한 (자본 효율성)
  4. 부채비율 상한 (재무 안전성)
  5. 매출 역성장 하한 (가치함정 2차 방어)
  6. Piotroski F-Score 하한

점수:
  value_score (0~100) — PER·PBR·ROE·부채비율·영업이익YoY·배당수익률 가중합
  f_score     (0~9)   — 간소화 Piotroski (가용 데이터 기준)
"""

import logging
from datetime import date as _date
from typing import Dict, List, Optional, Tuple

from koreanstocks.core.data.fundamental_provider import fundamental_provider
from koreanstocks.core.data.provider import data_provider

logger = logging.getLogger(__name__)

# ── 업종별 PER 중앙값 (한국 시장 기준, 부분 매칭) ─────────────────
# value_score() sector_per_median 파라미터에 주입 — 업종 특성 반영
_SECTOR_PER_MEDIANS: Dict[str, float] = {
    "은행":    6.0,
    "금융":    7.0,
    "보험":    8.0,
    "증권":    9.0,
    "철강":    7.0,
    "조선":    9.0,
    "건설":    8.0,
    "화학":    9.0,
    "자동차":  9.0,
    "유통":   10.0,
    "음식료": 14.0,
    "전기전자": 15.0,
    "반도체":  18.0,
    "소프트웨어": 20.0,
    "의약품":  22.0,
    "바이오":  25.0,
    # 추가 업종 (기본값 12.0과 괴리가 큰 업종 우선)
    "제약":   20.0,   # 제약·바이오 고PER 업종
    "헬스케어": 22.0,
    "운수":    9.0,   # 항공·해운·육운
    "항공":    9.0,
    "해운":    7.0,
    "통신":   11.0,   # KT·SKT·LGU+
    "에너지": 10.0,   # 정유·가스
    "정유":    8.0,
    "기계":   10.0,   # 산업기계·장비
    "섬유":    8.0,   # 패션·의류
    "지주":    7.0,   # 지주회사 저PER
    "농업":   12.0,
    "서비스": 13.0,
    # KOSDAQ 업종 분류명 (KIND API 기준) — 기본값 12.0 대비 괴리 큰 업종
    "IT":     18.0,   # IT H/W·S/W·부품·통신 등 KOSDAQ IT 계열
    "오락":   22.0,   # 오락문화 (엔터테인먼트·공연)
    "게임":   22.0,   # 게임·모바일게임
    "인터넷": 20.0,   # 인터넷 플랫폼
    "콘텐츠": 20.0,   # 콘텐츠·스트리밍
    "미디어": 16.0,   # 방송·미디어
}

# 긴 키(구체적 업종) → 짧은 키 순으로 미리 정렬 — 매 호출마다 sorted() 재실행 방지
# "바이오의약품" 같은 복합 업종명이 더 구체적인 키("바이오")에 우선 매칭됨
_SECTOR_PER_SORTED: List[Tuple[str, float]] = sorted(
    _SECTOR_PER_MEDIANS.items(), key=lambda x: -len(x[0])
)


def _sector_per_median(sector) -> float:
    """업종명에서 PER 중앙값 반환. 매칭 없으면 기본값 12.0.
    NaN·None·비문자열 입력은 기본값으로 처리 (pandas NaN 방어).
    """
    if not isinstance(sector, str) or not sector:
        return 12.0
    for key, val in _SECTOR_PER_SORTED:
        if key in sector:
            return val
    return 12.0


# ── 기본 필터 임계값 ─────────────────────────────────────────────
DEFAULT_FILTERS = {
    "per_max":          25.0,   # PER 상한 (성장주 포함 여유)
    "pbr_max":           3.0,   # PBR 상한
    "roe_min":           8.0,   # ROE 하한 (%)
    "debt_max":        150.0,   # 부채비율 상한 (%)
    "revenue_yoy_min": -15.0,   # 매출 역성장 하한 (%) — 가치함정 방어
    "f_score_min":       4,     # Piotroski 최소 점수
}

# value_score() 내부 cap — DEFAULT_FILTERS와 단일 소스로 연결
# 필터 상한값이 변경되면 점수 cap도 자동 반영됨
_PBR_CAP  = DEFAULT_FILTERS["pbr_max"]    # PBR=cap 이상이면 0pt
_DEBT_CAP = DEFAULT_FILTERS["debt_max"]   # 부채비율=cap 이상이면 0pt


# ── Piotroski F-Score ────────────────────────────────────────────

def piotroski_score(
    f: Dict,
    roe_min: float = DEFAULT_FILTERS["roe_min"],
) -> Tuple[int, Dict[str, bool]]:
    """
    간소화 Piotroski F-Score (0~9점).

    네이버 coinfo에서 가져올 수 있는 데이터로 9개 항목을 평가한다.

    Args:
        f: 펀더멘털 데이터 딕셔너리
        roe_min: screen()의 roe_min 파라미터 — P2 threshold를 필터 기준과 동기화

    수익성 (3):
      P1  영업이익률 > 5%  (단순 흑자 대신 수익 질 확인 — 하드 필터 중복 방지)
      P2  ROE > roe_min+3% (필터 하한보다 충분히 높은 ROE — 하드 필터 중복 방지)
      P3  ROE 전년 대비 개선
    레버리지·안전성 (3):
      L1  부채비율 < 100%
      L2  부채비율 전년 대비 감소
      L3  배당 지급 이력 있음 (배당수익률 > 0)
    성장성·효율성 (3):
      E1  매출 성장 (YoY > 0)
      E2  영업이익 YoY > 0
      E3  매출 역성장 없음 (YoY > -5%)
    """
    def safe(key, default=None):
        v = f.get(key)
        return v if v is not None else default

    # 필터 하한보다 3%p 높은 기준 — screen() roe_min 변경 시 P2도 자동 연동
    p2_threshold = roe_min + 3

    checks: Dict[str, bool] = {
        # 수익성
        # P1: 영업이익률 5% 초과 — 단순 흑자(하드 필터 중복)보다 수익 질을 평가
        "P1_영업이익률5":    safe("op_margin", 0) > 5,
        # P2: ROE가 필터 하한보다 충분히 높음 — roe_min 파라미터로 screen()과 동기화
        "P2_ROE우량":        safe("roe_cur", safe("roe", 0)) > p2_threshold,
        "P3_ROE개선":        bool(f.get("roe_improved")),
        # 레버리지·안전성
        "L1_부채100미만":     safe("debt_ratio", 999) < 100,
        "L2_부채감소":        bool(f.get("debt_decreased")),
        "L3_배당지급":        safe("dividend_yield", 0) > 0,
        # 성장성·효율성
        "E1_매출성장":        safe("revenue_yoy", -999) > 0,   # 매출 YoY 성장 여부
        "E2_영업이익성장":    safe("op_income_yoy", -999) > 0,
        "E3_매출역성장없음":  safe("revenue_yoy", 0) > -5,   # 데이터 없으면 True — screen() None 통과 정책과 일관
    }
    score = sum(1 for v in checks.values() if v)
    return score, checks


# ── 가치 점수 ────────────────────────────────────────────────────

def value_score(
    f: Dict,
    sector_per_median: float = 12.0,
    pbr_cap: float = _PBR_CAP,
    debt_cap: float = _DEBT_CAP,
) -> float:
    """
    가치 점수 산출 (0~100).

    구성:
      PER      25pt  낮을수록 유리 (업종 중앙값 대비 상대 평가)
      PBR      15pt  낮을수록 유리 (pbr_cap 이상 → 0pt — screen()의 pbr_max와 동기화)
      ROE      20pt  높을수록 유리 (30%에서 만점, 2개년 평균)
      부채      15pt  낮을수록 유리 (debt_cap 이상 → 0pt — screen()의 debt_max와 동기화)
      영업YoY   30pt  -30%→0점, 0%→50점, +30%→만점 (완전 선형, 데이터 없으면 제외)
      배당수익률 10pt  배당 지급 이력 유리 (3%+ 만점, 데이터 없으면 denominator 제외)

    Args:
        pbr_cap:  PBR 점수 상한 (screen()의 pbr_max 전달 — 하드 필터와 점수 척도 동기화)
        debt_cap: 부채비율 점수 상한 (screen()의 debt_max 전달)
    """
    parts: List[Tuple[float, float]] = []   # (earned, possible)

    per = f.get("per")
    if per is not None and per > 0:
        ratio = min(per / max(sector_per_median, 1), 2.5)
        parts.append((max(0.0, 25 * (1 - ratio / 2.5)), 25.0))

    pbr = f.get("pbr")
    if pbr is not None and pbr > 0:
        # cap을 screen()의 pbr_max와 연동 — 하드 필터 상한에서 0pt로 변별력 강화
        parts.append((max(0.0, 15 * (1 - min(pbr, pbr_cap) / pbr_cap)), 15.0))

    roe = f.get("roe")
    if roe is not None:
        # 만점 기준 30% — 20%보다 넓은 범위에서 변별력 확보 (ROE=30%에서 20pt)
        parts.append((min(20.0, max(0.0, roe / 30 * 20)), 20.0))

    debt = f.get("debt_ratio")
    if debt is not None:
        # cap을 screen()의 debt_max와 연동 — 하드 필터 상한에서 0pt로 변별력 강화
        parts.append((max(0.0, 15 * (1 - min(debt, debt_cap) / debt_cap)), 15.0))

    opi_yoy = f.get("op_income_yoy")
    if opi_yoy is not None:
        # -30%~+30% 완전 선형: -30%→0점, 0%→50점, +30%→만점
        # 양단 클리핑으로 earned ≤ possible 보장 → value_score ≤ 100 유지
        raw = max(-30.0, min(30.0, opi_yoy)) / 30 * 15   # -15 ~ +15 보장
        parts.append((raw + 15.0, 30.0))                   # 0~30 범위로 이동 후 정규화

    # 배당수익률: 데이터 없으면 denominator 제외 (성장주 불이익 방지)
    div = f.get("dividend_yield")
    if div is not None:
        parts.append((min(10.0, max(0.0, div / 3 * 10)), 10.0))

    if not parts:
        # screen()의 hard filter를 통과한 종목은 PER/PBR/ROE/debt 중 하나 이상이
        # 반드시 non-None이므로 실제로는 도달하지 않는 안전 폴백
        return 50.0

    earned   = sum(p[0] for p in parts)
    possible = sum(p[1] for p in parts)
    return round(earned / possible * 100, 1)


# ── 스크리너 ─────────────────────────────────────────────────────

class ValueScreener:
    """
    펀더멘털 기반 가치주 스크리너.

    실행 흐름:
      1. 시가총액 상위 후보군 + PER/ROE 사전 필터
      2. 펀더멘털 병렬 수집 (fundamental_provider, 당일 SQLite 캐시)
      3. 하드 필터 통과
      4. value_score + f_score 복합 정렬
      5. 상위 limit개 반환

    캐시:
      동일 필터 조합의 스크리닝 결과를 당일(자정까지) 인메모리 캐시로 보관.
      재요청 시 1~2분 소요되는 DART 수집 없이 즉시 반환한다.
    """

    def __init__(self):
        self._cache: Dict[tuple, List[Dict]] = {}
        self._cache_date: Optional[_date] = None

    def screen(
        self,
        market: str = "ALL",
        per_max: float = DEFAULT_FILTERS["per_max"],
        pbr_max: float = DEFAULT_FILTERS["pbr_max"],
        roe_min: float = DEFAULT_FILTERS["roe_min"],
        debt_max: float = DEFAULT_FILTERS["debt_max"],
        revenue_yoy_min: float = DEFAULT_FILTERS["revenue_yoy_min"],
        f_score_min: int = DEFAULT_FILTERS["f_score_min"],
        candidate_limit: int = 200,
        limit: int = 20,
    ) -> List[Dict]:
        """
        가치주 스크리닝 실행.

        Args:
            market: ALL | KOSPI | KOSDAQ
            per_max: PER 상한
            pbr_max: PBR 상한
            roe_min: ROE 하한 (%)
            debt_max: 부채비율 상한 (%)
            revenue_yoy_min: 매출 YoY 하한 (%) — 가치함정 방어
            f_score_min: Piotroski 최소 점수
            candidate_limit: 시가총액 상위 몇 종목을 후보로 볼지
            limit: 최종 반환 종목 수

        Returns:
            Piotroski(40%) + value_score(60%) 복합 점수 내림차순 정렬된 Dict 리스트
        """
        # ── 당일 인메모리 캐시 확인 ───────────────────────────────────
        today = _date.today()
        if self._cache_date != today:
            self._cache.clear()
            self._cache_date = today

        # limit·f_score_min은 캐시 키에서 제외 — 동일 필터로 이 값만 다를 때 재스크리닝 방지
        # f_score_min은 반환 직전 사후 필터로 적용 (limit과 동일한 패턴)
        cache_key = (market, per_max, pbr_max, roe_min, debt_max,
                     revenue_yoy_min, candidate_limit)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            filtered = [r for r in cached if r["f_score"] >= f_score_min]
            logger.info(f"[VALUE] 캐시 히트 — {min(len(filtered), limit)}종목 반환 (당일 동일 조건)")
            return [dict(r) for r in filtered[:limit]]

        logger.info(f"[VALUE] 스크리닝 시작 (market={market}, limit={limit})")

        # 1. 후보 종목 (시가총액 상위 + PER/ROE 사전 필터 → 가치주 후보군)
        candidates = data_provider.get_value_candidates(
            limit=candidate_limit,
            market=market,
            per_max=per_max * 1.5,   # DART 없이 사전 필터 → 관용 상한으로 설정
            roe_min=max(0.0, roe_min - 5),  # 여유 있게 (DART 기준 재확인)
        )
        if not candidates:
            logger.warning("[VALUE] 후보 종목 없음 — 스크리닝 중단")
            return []

        stock_list = data_provider.get_stock_list()
        # 코드 기준 인덱스 — 루프 내 O(n²) boolean mask 대신 O(1) 조회
        stock_index = stock_list.drop_duplicates(subset=["code"]).set_index("code")
        logger.info(f"[VALUE] 후보 {len(candidates)}종목 펀더멘털 수집 중...")

        # 2. 펀더멘털 병렬 수집
        fund_map = fundamental_provider.get_fundamentals_batch(candidates, max_workers=15)

        # 3. 필터 + 점수 산출
        passed: List[Dict] = []
        skipped_no_data  = 0
        skipped_filter   = 0

        for code in candidates:
            f = fund_map.get(code, {})

            per    = f.get("per")
            pbr    = f.get("pbr")
            # ROE 2개년 평균 (지속성 반영 — quality_screener와 동일 방식)
            roe_cur  = f.get("roe")
            roe_prev = f.get("roe_prev")
            if roe_cur is not None and roe_prev is not None:
                roe = round((roe_cur + roe_prev) / 2, 1)
            else:
                roe = roe_cur
            debt   = f.get("debt_ratio")
            revyoy = f.get("revenue_yoy")
            op_pos = f.get("op_income_positive", False)

            # 데이터 부족 → 스킵
            # PER·PBR 모두 없거나, ROE·부채비율 모두 없으면 가치 판단 불가
            if (per is None and pbr is None) or (roe is None and debt is None):
                skipped_no_data += 1
                continue

            # ── 하드 필터 ──────────────────────────────────────
            fail = False

            # 영업이익 흑자 필수
            if not op_pos:
                fail = True

            # PER: 유효한 경우에만 체크 (음수 PER = 적자 기업)
            if not fail and per is not None and (per <= 0 or per > per_max):
                fail = True

            if not fail and pbr is not None and pbr > pbr_max:
                fail = True

            # 현재 연도 ROE 음수 → 즉시 탈락 (2개년 평균이 양수여도 현재 적자는 거부)
            if not fail and roe_cur is not None and roe_cur < 0:
                fail = True

            # 2개년 평균 ROE가 하한 미달 → 탈락
            if not fail and roe is not None and roe < roe_min:
                fail = True

            if not fail and debt is not None and debt > debt_max:
                fail = True

            # 매출 역성장 (데이터 없으면 통과 — 보수적 허용)
            if not fail and revyoy is not None and revyoy < revenue_yoy_min:
                fail = True

            if fail:
                skipped_filter += 1
                continue

            # 평균 ROE를 Piotroski·value 점수 계산에 반영
            # roe_cur 키를 별도 전달 → P2_ROE우량이 현재 연도 ROE 기준으로 판단
            f_for_score = {**f, "roe": roe, "roe_cur": roe_cur}

            # Piotroski — roe_min 전달로 P2 threshold를 필터 기준과 동기화
            fscore, fchecks = piotroski_score(f_for_score, roe_min=roe_min)

            # 종목 메타
            meta = stock_index.loc[code] if code in stock_index.index else None
            name = meta.get("name", code)    if meta is not None else code
            mkt  = meta.get("market", "")   if meta is not None else ""
            sect = meta.get("sector", "")   if meta is not None else ""

            vscore = value_score(
                f_for_score,
                sector_per_median=_sector_per_median(sect),
                pbr_cap=pbr_max,
                debt_cap=debt_max,
            )

            passed.append({
                "code":           code,
                "name":           name,
                "market":         mkt,
                "sector":         sect,
                "per":            per,
                "pbr":            pbr,
                "roe":            roe,   # 평균 ROE 표시
                "debt_ratio":     debt,
                "op_margin":      f.get("op_margin"),
                "revenue_yoy":    revyoy,
                "op_income_yoy":  f.get("op_income_yoy"),
                "dividend_yield": f.get("dividend_yield"),
                "f_score":        fscore,
                "f_checks":       fchecks,
                "value_score":    vscore,
                "fundamentals":   {
                    k: v for k, v in f.items()
                    if k not in ("code",)
                },
            })

        # 4. 정렬: f_score(40%) + value_score(60%) 복합 점수
        # 튜플 정렬 대신 단일 점수화 — value_score 차이를 충분히 반영
        passed.sort(
            key=lambda r: r["f_score"] / 9 * 40 + r["value_score"] * 0.60,
            reverse=True,
        )

        # passed 전체 캐시 (당일 유효) — limit·f_score_min 변경 시 재스크리닝 불필요
        self._cache[cache_key] = [dict(r) for r in passed]

        # 5. f_score_min 사후 필터 + limit 슬라이스
        f_filtered = [r for r in passed if r["f_score"] >= f_score_min]
        skipped_fscore = len(passed) - len(f_filtered)
        result = f_filtered[:limit]

        logger.info(
            f"[VALUE] 완료: 후보 {len(candidates)} → "
            f"데이터없음 {skipped_no_data} → "
            f"하드필터탈락 {skipped_filter} → "
            f"F-Score탈락 {skipped_fscore} → "
            f"통과 {len(passed) - skipped_fscore} → 최종 {len(result)}"
        )

        # 캐시 히트 경로와 동일하게 복사본 반환 (호출자 변경이 캐시에 영향 없도록)
        return [dict(r) for r in result]

    def get_filter_defaults(self) -> Dict:
        """현재 기본 필터 임계값 반환 (API 노출용)."""
        return dict(DEFAULT_FILTERS)


value_screener = ValueScreener()
