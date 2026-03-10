"""추천 결과 자동 검증 (Outcome Tracker)

추천 후 5·10·20 거래일이 경과하면 실제 주가를 수집해 성과를 기록합니다.
누적 데이터는 ML 재학습·프롬프트 개선·가중치 튜닝의 근거로 활용됩니다.

정답 기준:
  BUY  → 5/10/20 거래일 후 수익률 > 0%
  SELL → 5/10/20 거래일 후 수익률 < 0%
  HOLD → 5/10/20 거래일 후 손실 > HOLD_LOSS_THRESHOLD 를 넘지 않으면 정답
"""
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from koreanstocks.core.data.database import db_manager

logger = logging.getLogger(__name__)

# 검증 거래일 수 — horizon_config 및 DB 쿼리 컬럼 순서와 반드시 일치
HORIZONS: List[int] = [5, 10, 20]

# HOLD 정답 기준: 수익률이 이 값 초과이면 정답 (예: -5.0 → -5% 이상 손실은 오답)
HOLD_LOSS_THRESHOLD: float = -5.0

# 거래일 → 달력일 변환 파라미터 (공휴일·주말 여유 포함)
_LOOK_AHEAD_MULT: int = 2   # 거래일당 달력일 배율
_LOOK_AHEAD_BUF:  int = 10  # 추가 여유 일수


# ──────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────

def _look_ahead_days(n: int) -> int:
    """n 거래일 조회에 필요한 달력일 범위 계산."""
    return n * _LOOK_AHEAD_MULT + _LOOK_AHEAD_BUF


def _is_correct(action: str, return_pct: float) -> int:
    """추천 정답 여부: 1=정답, 0=오답"""
    if action == "BUY":
        return 1 if return_pct > 0 else 0
    if action == "SELL":
        return 1 if return_pct < 0 else 0
    # HOLD: HOLD_LOSS_THRESHOLD 초과 손실이 나지 않으면 정답
    return 1 if return_pct > HOLD_LOSS_THRESHOLD else 0


def _fetch_ohlcv(code: str, from_date: str, to_date: str) -> pd.DataFrame:
    """FDR로 OHLCV 조회. 공통 컬럼(close 포함) DataFrame 반환."""
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(code, from_date, to_date)
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        logger.warning(f"[{code}] FDR OHLCV 실패: {e}")

    return pd.DataFrame()


def _get_date_range(base_date: str, n: int) -> Optional[Tuple[str, str]]:
    """base_date 다음날부터 n 거래일 조회에 필요한 (start, end) 달력일 반환.
    아직 조회 가능한 날짜 범위가 없으면 None 반환.
    """
    base  = date.fromisoformat(base_date)
    far   = min(base + timedelta(days=_look_ahead_days(n)), date.today())
    start = (base + timedelta(days=1)).isoformat()
    end   = far.isoformat()
    return (start, end) if start <= end else None


def _get_price_after_n_trading_days(
    code: str, base_date: str, n: int
) -> Optional[Tuple[str, float]]:
    """base_date 이후 n번째 거래일의 (날짜, 종가) 반환. 아직 지나지 않으면 None."""
    date_range = _get_date_range(base_date, n)
    if date_range is None:
        return None  # 오늘 추천 → 아직 미래

    start, end = date_range
    df = _fetch_ohlcv(code, start, end)
    if df.empty or len(df) < n:
        return None  # 아직 n 거래일이 경과하지 않음

    actual_date = df.index[n - 1].strftime("%Y-%m-%d")
    close_price = float(df.iloc[n - 1]["close"])
    return actual_date, close_price


def _check_target_hit(
    code: str, base_date: str, n: int, target_price: float, action: str
) -> Optional[int]:
    """n 거래일 이내에 목표가 도달 여부 반환 (1=달성, 0=미달, None=데이터없음).

    BUY: n일 중 일중 고가(high) 기준 — 장중 한 번이라도 목표가 이상이면 달성
    SELL: n일 중 일중 저가(low) 기준 — 장중 한 번이라도 목표가 이하면 달성
    """
    date_range = _get_date_range(base_date, n)
    if date_range is None:
        return None

    start, end = date_range

    df = _fetch_ohlcv(code, start, end)
    if df.empty or len(df) < n:
        return None

    window = df.iloc[:n]
    if action == "BUY":
        peak = float(window["high"].max()) if "high" in window.columns else float(window["close"].max())
        return 1 if peak >= target_price else 0
    else:  # SELL
        trough = float(window["low"].min()) if "low" in window.columns else float(window["close"].min())
        return 1 if trough <= target_price else 0


# ──────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────

def record_outcomes() -> int:
    """아직 결과가 기록되지 않은 추천의 5·10·20거래일 후 성과를 DB에 업데이트.

    Returns:
        새로 업데이트(또는 삽입)된 추천 레코드 수.
    """
    # 20거래일 결과가 없는 모든 추천 조회
    # SELECT 뒤 price_Xd 컬럼 순서는 HORIZONS = [5, 10, 20] 과 반드시 일치
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                r.code,
                r.session_date,
                json_extract(r.detail_json, '$.ai_opinion.action')       AS action,
                json_extract(r.detail_json, '$.current_price')           AS entry_price,
                json_extract(r.detail_json, '$.ai_opinion.target_price') AS target_price,
                o.price_5d,   -- HORIZONS[0] = 5
                o.price_10d,  -- HORIZONS[1] = 10
                o.price_20d   -- HORIZONS[2] = 20
            FROM recommendations r
            LEFT JOIN recommendation_outcomes o
                   ON r.code = o.code AND r.session_date = o.session_date
            WHERE r.session_date IS NOT NULL
              AND r.detail_json  IS NOT NULL
              AND (o.correct_20d IS NULL)
            ORDER BY r.session_date ASC
            """
        )
        pending = cursor.fetchall()

    updated = 0
    for code, session_date, action, entry_price, target_price, p5, p10, p20 in pending:
        if entry_price is None or not action or action == "N/A":
            logger.debug(f"[{code}] {session_date} 스킵: entry_price={entry_price}, action={action}")
            continue

        try:
            entry_price = float(entry_price)
            if entry_price <= 0:
                logger.warning(f"[{code}] entry_price={entry_price} 비정상 — 스킵")
                continue
            target_price = float(target_price) if target_price else None
        except (ValueError, TypeError) as e:
            logger.warning(f"[{code}] 가격 변환 실패: entry={entry_price} — {e}")
            continue

        # HORIZONS 상수로 horizon_config 동적 생성 — HORIZONS 변경 시 자동 반영
        existing_by_n = dict(zip(HORIZONS, [p5, p10, p20]))
        horizon_config = [
            (n, existing_by_n[n], f"price_{n}d", f"return_{n}d", f"correct_{n}d")
            for n in HORIZONS
        ]
        updates: Dict = {}

        for n, existing_p, col_p, col_r, col_c in horizon_config:
            if existing_p is not None:
                continue  # 이미 기록됨

            result = _get_price_after_n_trading_days(code, session_date, n)
            if result is None:
                break  # 아직 n 거래일 미경과 → 더 긴 horizon도 의미 없음

            actual_date, price = result
            ret_pct = (price - entry_price) / entry_price * 100
            correct = _is_correct(action, ret_pct)

            updates[col_p] = price
            updates[col_r] = round(ret_pct, 2)
            updates[col_c] = correct

            logger.info(
                f"[{code}] {session_date} +{n}거래일({actual_date}): "
                f"{entry_price:,.0f}→{price:,.0f}원 ({ret_pct:+.1f}%) "
                f"{'✅' if correct else '❌'}"
            )

        # 20거래일 결과까지 완료 시 목표가 달성 여부 계산 (BUY/SELL 한정)
        # 기간 내 일중 고가(BUY)/저가(SELL) 기준으로 판정
        last_horizon = HORIZONS[-1]
        if f"price_{last_horizon}d" in updates and target_price and target_price > 0 \
                and action in ("BUY", "SELL"):
            hit = _check_target_hit(code, session_date, last_horizon, target_price, action)
            if hit is not None:
                updates["target_hit"] = hit

        if not updates:
            continue

        # DB UPSERT
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO recommendation_outcomes
                        (code, session_date, action, entry_price, target_price)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(code, session_date) DO NOTHING
                    """,
                    (code, session_date, action, entry_price, target_price),
                )
                set_clause = ", ".join(f"{k} = ?" for k in updates)
                values     = list(updates.values()) + [code, session_date]
                cursor.execute(
                    f"UPDATE recommendation_outcomes "
                    f"SET {set_clause}, updated_at = CURRENT_TIMESTAMP "
                    f"WHERE code = ? AND session_date = ?",
                    values,
                )
                conn.commit()
                updated += 1
            except Exception as e:
                logger.error(f"DB 저장 실패 [{code}, {session_date}]: {e}", exc_info=True)
                conn.rollback()

    logger.info(f"Outcome tracking 완료: {updated}건 업데이트")
    return updated


def get_outcome_stats(days: int = 90) -> Dict:
    """최근 N일간 추천 성과 집계 통계 반환."""
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(*)                                                       AS total,
                SUM(CASE WHEN correct_5d  IS NOT NULL THEN 1 ELSE 0 END)      AS ev_5d,
                SUM(CASE WHEN correct_10d IS NOT NULL THEN 1 ELSE 0 END)      AS ev_10d,
                SUM(CASE WHEN correct_20d IS NOT NULL THEN 1 ELSE 0 END)      AS ev_20d,
                AVG(CASE WHEN correct_5d  IS NOT NULL THEN correct_5d  END)   AS wr_5d,
                AVG(CASE WHEN correct_10d IS NOT NULL THEN correct_10d END)   AS wr_10d,
                AVG(CASE WHEN correct_20d IS NOT NULL THEN correct_20d END)   AS wr_20d,
                AVG(return_5d)                                                 AS ret_5d,
                AVG(return_10d)                                                AS ret_10d,
                AVG(return_20d)                                                AS ret_20d,
                SUM(CASE WHEN target_hit = 1 THEN 1 ELSE 0 END)              AS t_hits,
                SUM(CASE WHEN target_hit IS NOT NULL THEN 1 ELSE 0 END)       AS t_eval
            FROM recommendation_outcomes
            WHERE session_date >= date('now', ?)
            """,
            (f"-{days} days",),
        )
        row = cursor.fetchone()

    if not row or row[0] == 0:
        return {"total": 0}

    total, ev5, ev10, ev20, wr5, wr10, wr20, ret5, ret10, ret20, t_hits, t_eval = row
    return {
        "total":          total,
        "evaluated_5d":   ev5  or 0,
        "evaluated_10d":  ev10 or 0,
        "evaluated_20d":  ev20 or 0,
        "win_rate_5d":    round((wr5  or 0) * 100, 1),
        "win_rate_10d":   round((wr10 or 0) * 100, 1),
        "win_rate_20d":   round((wr20 or 0) * 100, 1),
        "avg_return_5d":  round(ret5  or 0, 2),
        "avg_return_10d": round(ret10 or 0, 2),
        "avg_return_20d": round(ret20 or 0, 2),
        "target_hit_rate": round(t_hits / t_eval * 100, 1) if t_eval and t_eval > 0 else None,
    }


def get_recent_outcomes(days: int = 90) -> List[Dict]:
    """최근 N일간 개별 추천 성과 목록 반환 (API·Telegram·CLI 표시용)."""
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                o.code,
                COALESCE(s.name, json_extract(r.detail_json, '$.name'), o.code) AS name,
                o.session_date,
                o.action,
                o.entry_price,
                o.target_price,
                o.price_5d,  o.return_5d,  o.correct_5d,
                o.price_10d, o.return_10d, o.correct_10d,
                o.price_20d, o.return_20d, o.correct_20d,
                o.target_hit
            FROM recommendation_outcomes o
            LEFT JOIN recommendations r
                   ON o.code = r.code AND o.session_date = r.session_date
            LEFT JOIN stocks s ON o.code = s.code
            WHERE o.session_date >= date('now', ?)
            ORDER BY o.session_date DESC, o.code
            """,
            (f"-{days} days",),
        )
        rows = cursor.fetchall()

    result = []
    for row in rows:
        (code, name, session_date, action,
         entry_price, target_price,
         p5,  r5,  c5,
         p10, r10, c10,
         p20, r20, c20,
         target_hit) = row
        result.append({
            "code":         code,
            "name":         name,
            "session_date": session_date,
            "action":       action,
            "entry_price":  entry_price,
            "target_price": target_price,
            "outcome_5d":  {"price": p5,  "return_pct": r5,  "correct": c5},
            "outcome_10d": {"price": p10, "return_pct": r10, "correct": c10},
            "outcome_20d": {"price": p20, "return_pct": r20, "correct": c20},
            "target_hit":   target_hit,
        })
    return result
