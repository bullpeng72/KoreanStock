"""
피처 개선 효과 검증 스크립트
===============================
세 가지 개선안을 순차 적용하여 RF 모델 R² 변화를 측정합니다.

Phase 0  Baseline  — 현재 22개 피처
Phase 1  +XS Rank  — 크로스섹셔널 퍼센타일 순위 피처 추가 (22개 → 44개)
Phase 2  +Fundamental — PBR·PER·배당수익률 추가 (PyKrx)
Phase 3  +Foreign/Inst — 외국인·기관 5일 누적 순매수 비율 추가 (PyKrx)

실행:
    python test_feature_improvements.py
"""

import os
import sys
import time
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from core.config import config
from core.data.provider import data_provider
from core.engine.indicators import indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("feature_test")

# ── 설정 ──────────────────────────────────────────────────────────
PERIOD       = "2y"
FUTURE_DAYS  = 5
TEST_RATIO   = 0.2
MIN_STOCKS   = 5
CACHE_PATH   = os.path.join(config.BASE_DIR, "data", "storage", "_pykrx_cache.pkl")

# 검증용 종목 (학습 종목 중 대표 30개 — PyKrx 호출 비용 절감)
# 전종목(146개) 사용 시 외국인/기관 API 호출에 ~5분 소요
TEST_STOCKS = [
    # KOSPI 대형주
    "005930", "000660", "035420", "005380", "051910", "006400", "035720",
    "068270", "105560", "055550", "086790", "012450", "090430", "128940",
    "036570", "009150", "010950", "003670", "034020", "000270",
    # KOSDAQ 대표
    "247540", "086520", "196170", "145020", "293490",
    "066970", "278280", "328130", "042700", "041510",
]

RF_PARAMS = dict(
    n_estimators=300, max_depth=4, min_samples_split=20,
    min_samples_leaf=15, max_features=0.5, random_state=42, n_jobs=-1,
)

BASE_FEATURE_COLS = [
    "rsi", "macd_diff", "price_sma_20_ratio", "vol_change",
    "price_sma_5_ratio", "rsi_change", "macd_diff_change",
    "bb_position", "bb_width", "vol_ratio",
    "stoch_k", "stoch_d", "cci", "atr_ratio", "candle_body", "obv_change",
    "return_1m", "return_3m", "high_52w_ratio", "mom_accel",
    "rs_vs_mkt_1m", "rs_vs_mkt_3m",
]

XS_RANK_COLS   = [f"{c}_xs" for c in BASE_FEATURE_COLS]
FUND_COLS      = ["pbr", "per", "div", "pbr_xs", "per_xs"]
FOREIGN_COLS   = ["foreign_5d_ratio", "inst_5d_ratio", "foreign_xs", "inst_xs"]


# ─────────────────────────────────────────────────────────────────
# 1. 기본 피처 수집 (현재 파이프라인과 동일)
# ─────────────────────────────────────────────────────────────────

def _fetch_market_returns(period: str) -> pd.DataFrame:
    raw = data_provider.get_ohlcv("KS11", period=period)
    if raw.empty:
        return pd.DataFrame()
    mkt = pd.DataFrame(index=raw.index)
    mkt["return_1m"] = raw["close"].pct_change(20)
    mkt["return_3m"] = raw["close"].pct_change(60)
    return mkt


def _build_base_features(df_ind: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """기존 22개 피처 추출 (train_models.build_features와 동일)"""
    feat = pd.DataFrame(index=df_ind.index)
    feat["rsi"]                = df_ind["rsi"]
    feat["macd_diff"]          = df_ind["macd_diff"]
    feat["price_sma_20_ratio"] = df_ind["close"] / df_ind["sma_20"]
    feat["vol_change"]         = df_ind["volume"].pct_change()
    feat["price_sma_5_ratio"]  = df_ind["close"] / df_ind["sma_5"]
    feat["rsi_change"]         = df_ind["rsi"].diff()
    feat["macd_diff_change"]   = df_ind["macd_diff"].diff()
    bb_range = (df_ind["bb_high"] - df_ind["bb_low"]).replace(0, np.nan)
    feat["bb_position"]        = (df_ind["close"] - df_ind["bb_low"]) / bb_range
    feat["bb_width"]           = bb_range / df_ind["bb_mid"]
    feat["vol_ratio"]          = df_ind["volume"] / df_ind["vol_sma_20"].replace(0, np.nan)
    for col in ["stoch_k", "stoch_d", "cci", "atr"]:
        if col in df_ind.columns:
            feat[col if col != "atr" else "atr_ratio"] = (
                df_ind[col] if col != "atr" else df_ind["atr"] / df_ind["close"]
            )
    feat["candle_body"]        = (df_ind["close"] - df_ind["open"]) / df_ind["open"]
    if "obv" in df_ind.columns:
        feat["obv_change"]     = df_ind["obv"].pct_change().clip(-1, 1)
    feat["return_1m"]          = df_ind["close"].pct_change(20)
    feat["return_3m"]          = df_ind["close"].pct_change(60)
    feat["high_52w_ratio"]     = df_ind["close"] / df_ind["close"].rolling(252, min_periods=60).max()
    feat["mom_accel"]          = feat["return_1m"] - feat["return_3m"] / 3.0
    if market_df is not None and not market_df.empty:
        aligned = market_df.reindex(feat.index).ffill()
        feat["rs_vs_mkt_1m"]  = (feat["return_1m"] - aligned.get("return_1m", 0)).fillna(0)
        feat["rs_vs_mkt_3m"]  = (feat["return_3m"] - aligned.get("return_3m", 0)).fillna(0)
    else:
        feat["rs_vs_mkt_1m"]  = 0.0
        feat["rs_vs_mkt_3m"]  = 0.0
    return feat.replace([np.inf, -np.inf], np.nan).dropna()


def collect_base_dataset(codes: List[str], period: str, future_days: int) -> pd.DataFrame:
    """전 종목 기본 피처 + raw_return 수집 → 날짜 인덱스 DataFrame"""
    market_df = _fetch_market_returns(period)
    frames = []
    for code in codes:
        try:
            df = data_provider.get_ohlcv(code, period=period)
            if df is None or df.empty or len(df) < 60:
                continue
            df_ind = indicators.calculate_all(df)
            if df_ind.empty:
                continue
            feat = _build_base_features(df_ind, market_df)
            if len(feat) <= future_days:
                continue
            close     = df_ind["close"].reindex(feat.index)
            future    = (close.shift(-future_days) - close) / close
            valid_idx = feat.index[:-future_days]
            res       = feat.loc[valid_idx].copy()
            res["raw_return"] = future.loc[valid_idx]
            res["code"]       = code
            valid = res.dropna()
            if not valid.empty:
                frames.append(valid)
                logger.info(f"  [{code}] {len(valid)}개 샘플")
        except Exception as e:
            logger.warning(f"  [{code}] 오류: {e}")
    if not frames:
        raise RuntimeError("수집된 샘플 없음")
    df_all = pd.concat(frames)
    # 크로스섹셔널 퍼센타일 순위 타깃
    df_all["target"] = df_all.groupby(df_all.index)["raw_return"].rank(pct=True) * 100.0
    # 날짜별 종목 수 필터
    cnt = df_all.groupby(df_all.index)["raw_return"].count()
    df_all = df_all[df_all.index.isin(cnt[cnt >= MIN_STOCKS].index)]
    return df_all


# ─────────────────────────────────────────────────────────────────
# 2. 크로스섹셔널 피처 추가 (Phase 1)
# ─────────────────────────────────────────────────────────────────

def add_xs_rank_features(df_all: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
    """날짜별 각 피처의 크로스섹셔널 퍼센타일 순위를 추가 피처로 생성"""
    for col in base_cols:
        if col in df_all.columns:
            df_all[f"{col}_xs"] = (
                df_all.groupby(df_all.index)[col].rank(pct=True) * 100.0
            )
    return df_all


# ─────────────────────────────────────────────────────────────────
# 3. PyKrx 펀더멘털 피처 수집 (Phase 2)
# ─────────────────────────────────────────────────────────────────

def fetch_fundamental_data(codes: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """종목별 PBR·PER·배당수익률 수집 (PyKrx)"""
    from pykrx import stock as krx
    cache = {}
    for i, code in enumerate(codes):
        try:
            df = krx.get_market_fundamental_by_date(start, end, code)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                cache[code] = df[["PBR", "PER", "DIV"]].rename(
                    columns={"PBR": "pbr", "PER": "per", "DIV": "div"}
                )
            if (i + 1) % 10 == 0:
                logger.info(f"  펀더멘털 수집 {i+1}/{len(codes)}")
        except Exception as e:
            logger.debug(f"  [{code}] 펀더멘털 오류: {e}")
    return cache


def add_fundamental_features(df_all: pd.DataFrame, fund_cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """DataFrame에 PBR·PER·배당수익률 컬럼 추가"""
    rows = []
    for code, grp in df_all.groupby("code"):
        if code in fund_cache:
            fdf = fund_cache[code].reindex(grp.index).ffill()
            grp = grp.copy()
            for col in ["pbr", "per", "div"]:
                grp[col] = fdf[col] if col in fdf.columns else np.nan
        rows.append(grp)
    df_merged = pd.concat(rows)
    # 크로스섹셔널 순위
    for col in ["pbr", "per"]:
        if col in df_merged.columns:
            df_merged[f"{col}_xs"] = (
                df_merged.groupby(df_merged.index)[col].rank(pct=True) * 100.0
            )
    return df_merged


# ─────────────────────────────────────────────────────────────────
# 4. PyKrx 외국인·기관 순매수 피처 수집 (Phase 3)
# ─────────────────────────────────────────────────────────────────

def fetch_trading_data(codes: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """종목별 외국인·기관 순매수 금액 수집 (PyKrx)"""
    from pykrx import stock as krx
    cache = {}
    for i, code in enumerate(codes):
        try:
            df = krx.get_market_trading_value_by_date(start, end, code)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                cache[code] = df[["외국인합계", "기관합계"]].rename(
                    columns={"외국인합계": "foreign_net", "기관합계": "inst_net"}
                )
            if (i + 1) % 10 == 0:
                logger.info(f"  거래 수집 {i+1}/{len(codes)}")
        except Exception as e:
            logger.debug(f"  [{code}] 거래 오류: {e}")
    return cache


def add_foreign_inst_features(
    df_all: pd.DataFrame,
    trade_cache: Dict[str, pd.DataFrame],
    ohlcv_cache: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """외국인·기관 5일 누적 순매수 비율 + 크로스섹셔널 순위 추가"""
    rows = []
    for code, grp in df_all.groupby("code"):
        grp = grp.copy()
        if code in trade_cache and code in ohlcv_cache:
            tdf = trade_cache[code].reindex(grp.index).ffill()
            odf = ohlcv_cache[code]
            # 거래대금 = 종가 × 거래량 (분모)
            turnover = (odf["close"] * odf["volume"]).reindex(grp.index).ffill()
            # 5일 누적 / 총 거래대금 5일 합계
            for col, feat_name in [("foreign_net", "foreign_5d_ratio"),
                                    ("inst_net",    "inst_5d_ratio")]:
                if col in tdf.columns:
                    roll_net = tdf[col].rolling(5, min_periods=1).sum()
                    roll_to  = turnover.rolling(5, min_periods=1).sum().replace(0, np.nan)
                    grp[feat_name] = (roll_net / roll_to).clip(-1, 1)
        rows.append(grp)
    df_merged = pd.concat(rows)
    # 크로스섹셔널 순위
    for col in ["foreign_5d_ratio", "inst_5d_ratio"]:
        if col in df_merged.columns:
            df_merged[f"{col.replace('_5d_ratio','_xs')}"] = (
                df_merged.groupby(df_merged.index)[col].rank(pct=True) * 100.0
            )
    return df_merged


# ─────────────────────────────────────────────────────────────────
# 5. 공통 학습/평가
# ─────────────────────────────────────────────────────────────────

def train_eval(df_all: pd.DataFrame, feature_cols: List[str], label: str) -> Dict:
    """지정 피처로 RF 학습 후 R²·RMSE 반환"""
    available = [c for c in feature_cols if c in df_all.columns]
    if not available:
        return {"label": label, "r2": None, "rmse": None, "n_features": 0}

    all_dates  = sorted(df_all.index.unique())
    n_dates    = len(all_dates)
    split_idx  = int(n_dates * (1.0 - TEST_RATIO))
    split_date = all_dates[split_idx]

    keep = available + ["target"]
    df_train = df_all[df_all.index <  split_date][keep].dropna()
    df_test  = df_all[df_all.index >= split_date][keep].dropna()

    if df_train.empty or df_test.empty:
        return {"label": label, "r2": None, "rmse": None, "n_features": len(available)}

    X_train, y_train = df_train[available].values, df_train["target"].values
    X_test,  y_test  = df_test[available].values,  df_test["target"].values

    baseline_rmse = float(np.sqrt(np.mean((y_test - y_train.mean()) ** 2)))

    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X_train)
    X_te    = scaler.transform(X_test)
    model   = RandomForestRegressor(**RF_PARAMS)
    t0      = time.time()
    model.fit(X_tr, y_train)
    elapsed = time.time() - t0

    y_pred   = model.predict(X_te)
    r2       = r2_score(y_test, y_pred)
    rmse     = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    train_r2 = r2_score(y_train, model.predict(X_tr))
    improv   = (baseline_rmse - rmse) / baseline_rmse * 100

    result = dict(
        label=label, n_features=len(available),
        r2=round(r2, 4), train_r2=round(train_r2, 4),
        gap=round(train_r2 - r2, 4),
        rmse=round(rmse, 4), baseline_rmse=round(baseline_rmse, 4),
        improvement=round(improv, 2), elapsed=round(elapsed, 1),
        train_samples=len(X_train), test_samples=len(X_test),
    )
    return result


def print_result(r: Dict):
    print(
        f"  {r['label']:<30} "
        f"피처={r['n_features']:>2}개  "
        f"R²={r['r2']:>+.4f}  "
        f"Train R²={r['train_r2']:>+.4f}  "
        f"gap={r['gap']:>.4f}  "
        f"개선={r['improvement']:>+.2f}%  "
        f"({r['elapsed']:.1f}s)"
    )


# ─────────────────────────────────────────────────────────────────
# 6. 메인 실행
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  피처 개선 효과 검증")
    print(f"  종목: {len(TEST_STOCKS)}개  |  기간: {PERIOD}  |  예측: {FUTURE_DAYS}거래일 후")
    print("=" * 70)

    # ── Step 1: 기본 OHLCV + 피처 수집 ────────────────────────────
    print("\n[1/4] 기본 피처 수집 중...")
    df_base = collect_base_dataset(TEST_STOCKS, PERIOD, FUTURE_DAYS)
    print(f"  총 샘플: {len(df_base)}, 날짜별 평균 종목: {df_base.groupby(df_base.index)['code'].count().mean():.1f}")

    # ── PyKrx 날짜 범위 계산 ──────────────────────────────────────
    dates    = sorted(df_base.index.unique())
    pykrx_start = dates[0].strftime("%Y%m%d")
    pykrx_end   = dates[-1].strftime("%Y%m%d")

    # ── Step 2: PyKrx 데이터 캐시 확인 ────────────────────────────
    fund_cache  = {}
    trade_cache = {}
    ohlcv_cache = {}

    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                saved = pickle.load(f)
            # 캐시 유효성: 종목 리스트가 동일한지 확인
            if saved.get("codes") == sorted(TEST_STOCKS):
                fund_cache  = saved.get("fund", {})
                trade_cache = saved.get("trade", {})
                ohlcv_cache = saved.get("ohlcv", {})
                print(f"\n  캐시 로드: 펀더멘털 {len(fund_cache)}개, 거래 {len(trade_cache)}개 종목")
            else:
                print("\n  종목 목록 변경 → 캐시 재수집")
        except Exception:
            pass

    if not fund_cache:
        print(f"\n[2/4] PyKrx 펀더멘털 수집 중 ({len(TEST_STOCKS)}종목)...")
        fund_cache = fetch_fundamental_data(TEST_STOCKS, pykrx_start, pykrx_end)

    if not trade_cache:
        print(f"\n[3/4] PyKrx 외국인/기관 수집 중 ({len(TEST_STOCKS)}종목)...")
        trade_cache = fetch_trading_data(TEST_STOCKS, pykrx_start, pykrx_end)

    # OHLCV 캐시 (외국인/기관 비율 분모 계산용)
    if not ohlcv_cache:
        for code in TEST_STOCKS:
            try:
                df = data_provider.get_ohlcv(code, period=PERIOD)
                if df is not None and not df.empty:
                    ohlcv_cache[code] = df
            except Exception:
                pass

    # 캐시 저장
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(dict(
            codes=sorted(TEST_STOCKS),
            fund=fund_cache, trade=trade_cache, ohlcv=ohlcv_cache,
        ), f)

    # ── Step 3: 피처 단계별 구축 ──────────────────────────────────
    print("\n[4/4] 단계별 학습 및 평가")
    print("-" * 70)

    results = []

    # Phase 0: Baseline
    r = train_eval(df_base, BASE_FEATURE_COLS, "Phase0 Baseline (22피처)")
    results.append(r); print_result(r)

    # Phase 1: + 크로스섹셔널 순위 피처
    df_xs = add_xs_rank_features(df_base.copy(), BASE_FEATURE_COLS)
    r = train_eval(df_xs, BASE_FEATURE_COLS + XS_RANK_COLS,
                   "Phase1 +XS Rank (44피처)")
    results.append(r); print_result(r)

    # Phase 2: + 펀더멘털
    df_fund = add_fundamental_features(df_xs.copy(), fund_cache)
    r = train_eval(df_fund, BASE_FEATURE_COLS + XS_RANK_COLS + FUND_COLS,
                   "Phase2 +Fundamental (49피처)")
    results.append(r); print_result(r)

    # Phase 3: + 외국인/기관
    df_fi = add_foreign_inst_features(df_fund.copy(), trade_cache, ohlcv_cache)
    r = train_eval(df_fi, BASE_FEATURE_COLS + XS_RANK_COLS + FUND_COLS + FOREIGN_COLS,
                   "Phase3 +Foreign/Inst (53피처)")
    results.append(r); print_result(r)

    # ── 요약 출력 ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  최종 요약")
    print("-" * 70)
    base_r2 = results[0]["r2"]
    for r in results:
        delta = (r["r2"] - base_r2) if r["r2"] is not None else None
        delta_str = f"  (Δ {delta:+.4f})" if delta is not None else ""
        print(f"  {r['label']:<30}  R²={r['r2']:>+.4f}{delta_str}")
    print("=" * 70)

    # ── 외국인/기관 피처의 실제 유효 비율 ─────────────────────────
    fi_cols_found = [c for c in FOREIGN_COLS if c in df_fi.columns]
    non_null = df_fi[fi_cols_found].notna().mean() * 100 if fi_cols_found else pd.Series()
    if not non_null.empty:
        print("\n  외국인/기관 피처 유효 데이터 비율:")
        for col, pct in non_null.items():
            print(f"    {col}: {pct:.1f}%")

    return results


if __name__ == "__main__":
    main()
