"""
공유 피처 추출 로직
===================
trainer.py (학습) 과 prediction_model.py (추론) 양쪽에서 import해 사용.
BASE_FEATURE_COLS 20개만 계산 — 미사용 중간 피처 제거로 수집 속도 개선.

피처 목록은 BASE_FEATURE_COLS 가 단일 소스(Single Source of Truth).
trainer.py / prediction_model.py 는 이 목록을 import 해 사용하며,
직접 하드코딩하지 않는다.
"""
import numpy as np
import pandas as pd
from koreanstocks.core.config import config

# ── 피처 목록 단일 소스 (trainer.py / prediction_model.py 양쪽 import) ──────
BASE_FEATURE_COLS = [
    # ── 변동성 / 추세 강도 ────────────────────────────────────
    'atr_ratio',            # rolling 60일 percentile (레짐 독립)
    'adx',                  # 추세 강도 (방향 무관)
    'bb_width',             # 볼린저 밴드 폭 (변동성 압축 감지)
    'bb_position',          # BB 내 가격 위치 (0=하단, 1=상단)
    # ── 중기 모멘텀 / 상대강도 ────────────────────────────────
    'rs_vs_mkt_3m',         # KOSPI 대비 3개월 초과수익
    'high_52w_ratio',       # 52주 고가 대비 현재가 (추세 위치)
    'mom_accel',            # 모멘텀 가속도 (1m - 3m/3)
    # ── 추세 / 가격 모멘텀 ────────────────────────────────────
    'macd_diff',            # MACD 다이버전스 (추세 전환)
    'macd_slope_5d',        # MACD 다이버전스 5일 기울기 (모멘텀 가속)
    'price_sma_5_ratio',    # 단기 추세 (가격/SMA5)
    # ── 반전/패턴 신호 ────────────────────────────────────────
    'fisher',               # Fisher Transform (극값=반전)
    'bullish_fractal_5d',   # Williams 강세 프랙탈 5일
    # ── 거래량 방향성 ─────────────────────────────────────────
    'mfi',                  # Money Flow Index (가격+거래량 통합)
    'vzo',                  # Volume Zone Oscillator
    'obv_trend',            # OBV 10일 모멘텀 rolling 20일 percentile (0~1)
    'low_52w_ratio',        # 52주 저가 대비 현재가 (반등 위치)
    # ── 극값 감지 / 반전 신호 ──────────────────────────────────
    'rsi',                  # RSI rolling 14일 percentile (0~1, 레짐 독립)
    'cci_pct',              # CCI rolling 20일 percentile (레짐 독립적 0~1)
    # ── 거시경제 ──────────────────────────────────────────────
    'vix_level',            # VIX 공포지수
    'sp500_1m',             # S&P500 1개월 수익률
]  # 20개 피처


def build_features(
    df: pd.DataFrame,
    market_df: pd.DataFrame = None,
    macro_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """지표(indicator) DataFrame → BASE_FEATURE_COLS 20개 추출.

    학습(trainer.py)과 추론(prediction_model.py)에서 동일 로직을 사용해
    train/serve 피처 불일치(feature skew)를 방지한다.

    Parameters
    ----------
    df         : indicators.calculate_all() 결과 (OHLCV + 기술적 지표 컬럼 포함)
    market_df  : 시장 지수 수익률 DataFrame (return_1m, return_3m 컬럼)
    macro_df   : 거시경제 DataFrame (vix_level, sp500_1m 컬럼)

    Returns
    -------
    DataFrame  : 유효 행만 포함 (NaN / ±inf 행 제거), 컬럼 = BASE_FEATURE_COLS 교집합
    """
    if df.empty:
        return df

    feat = pd.DataFrame(index=df.index)
    tdy  = config.TRADING_DAYS_PER_YEAR   # 252거래일

    # ── 변동성 / 추세 강도 ────────────────────────────────────
    feat['atr_ratio']   = (df['atr'] / df['close']).rolling(60).rank(pct=True)
    feat['adx']         = df['adx']

    bb_range            = (df['bb_high'] - df['bb_low']).replace(0, np.nan)
    feat['bb_position'] = (df['close'] - df['bb_low']) / bb_range
    feat['bb_width']    = (bb_range / df['bb_mid']).clip(0.01, 0.50)  # ±inf 방지

    # ── 중기 모멘텀 / 상대강도 ────────────────────────────────
    feat['high_52w_ratio'] = (
        df['close'] / df['close'].rolling(tdy, min_periods=60).max()
    )
    _return_1m = df['close'].pct_change(20)
    _return_3m = df['close'].pct_change(60)
    feat['mom_accel'] = _return_1m - _return_3m / 3.0

    if market_df is not None and not market_df.empty:
        aligned = market_df.reindex(feat.index).ffill()
        feat['rs_vs_mkt_3m'] = (_return_3m - aligned.get('return_3m', 0)).fillna(0)
    else:
        feat['rs_vs_mkt_3m'] = 0.0

    # ── 추세 / 가격 모멘텀 ────────────────────────────────────
    feat['macd_diff']         = df['macd_diff']
    feat['macd_slope_5d']     = df['macd_diff'].diff(5)
    feat['price_sma_5_ratio'] = df['close'] / df['sma_5']

    # ── 반전 / 패턴 신호 ─────────────────────────────────────
    if 'fisher' in df.columns:
        feat['fisher'] = df['fisher']
    if 'bullish_fractal' in df.columns:
        feat['bullish_fractal_5d'] = df['bullish_fractal'].rolling(5, min_periods=1).max()

    # ── 거래량 방향성 ─────────────────────────────────────────
    if 'mfi' in df.columns:
        feat['mfi'] = df['mfi']
    if 'vzo' in df.columns:
        feat['vzo'] = df['vzo']
    if 'obv' in df.columns:
        # OBV 10일 모멘텀 → rolling 20일 percentile (0~1)
        # clip(-1, 1) 대신 rank(pct=True) 사용: 급등 OBV(+300%)도 동등 신호 강도 유지
        feat['obv_trend'] = df['obv'].pct_change(10).rolling(20, min_periods=1).rank(pct=True)
    feat['low_52w_ratio'] = (
        df['close'] / df['close'].rolling(tdy, min_periods=60).min()
    )

    # ── 극값 감지 / 반전 신호 ─────────────────────────────────
    if 'rsi' in df.columns:
        # RSI rolling 14일 percentile: 0~1 레짐 독립 정규화
        # /100 단순 나눔 대비 분포가 균일해져 극값(과매도/과매수) 신호 강도 보존
        feat['rsi'] = df['rsi'].rolling(14, min_periods=1).rank(pct=True)
    if 'cci' in df.columns:
        # CCI rolling 20일 percentile: 레짐 독립적 0~1 정규화 (±100 이탈 극값 감지)
        feat['cci_pct'] = df['cci'].rolling(20, min_periods=1).rank(pct=True)

    # ── 거시경제 ──────────────────────────────────────────────
    if macro_df is not None and not macro_df.empty:
        aligned = macro_df.reindex(feat.index).ffill()
        # ffill 후에도 커버되지 않는 날짜(macro 시작 이전)는 중립값으로 채움
        feat['vix_level'] = aligned['vix_level'].fillna(20.0) if 'vix_level' in aligned.columns else 20.0
        feat['sp500_1m']  = aligned['sp500_1m'].fillna(0.0)   if 'sp500_1m'  in aligned.columns else 0.0
    else:
        feat['vix_level'] = 20.0
        feat['sp500_1m']  = 0.0

    return feat.replace([np.inf, -np.inf], np.nan).dropna()
