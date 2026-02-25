"""
ML 모델 학습 스크립트
======================
여러 종목의 OHLCV 데이터를 수집하고, 기술적 지표를 계산한 뒤
RandomForest / GradientBoosting / XGBoost 모델을 학습하여
모델(.pkl)과 스케일러(.pkl)를 함께 저장합니다.

실행 방법:
    python train_models.py
    python train_models.py --period 2y --future-days 5
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, r2_score
import joblib
import xgboost as xgb

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import config
from core.data.provider import data_provider
from core.engine.indicators import indicators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_models')

# ───────────────────────────── 설정 ─────────────────────────────

MODEL_DIR  = os.path.join(config.BASE_DIR, "models", "saved", "prediction_models")
PARAMS_DIR = os.path.join(config.BASE_DIR, "models", "saved", "model_params")

# 학습에 사용할 종목
# 구성 원칙:
#   - KOSPI 대형주 (기존) + KOSDAQ 대표주 + 미학습 섹터 추가
#   - 추천 후보 풀(거래량·등락률 상위)에 자주 등장하는 종목 유형을 최대한 포괄
DEFAULT_TRAINING_STOCKS = [
    # ── KOSPI 대형주 (기존 20개) ───────────────────────────────────
    '005930',  # 삼성전자       반도체
    '000660',  # SK하이닉스     반도체
    '035420',  # NAVER          인터넷
    '005380',  # 현대차         자동차
    '051910',  # LG화학         화학/2차전지
    '006400',  # 삼성SDI        2차전지
    '035720',  # 카카오         인터넷/플랫폼
    '068270',  # 셀트리온       바이오(KOSPI)
    '105560',  # KB금융         은행
    '055550',  # 신한지주       은행
    '003550',  # LG             지주
    '096770',  # SK이노베이션   에너지/2차전지
    '028260',  # 삼성물산       건설/무역
    '066570',  # LG전자         가전/부품
    '017670',  # SK텔레콤       통신
    '030200',  # KT             통신
    '032830',  # 삼성생명       보험
    '000270',  # 기아           자동차
    '012330',  # 현대모비스     자동차부품
    '011170',  # 롯데케미칼     화학

    # ── KOSPI 미학습 섹터 신규 추가 ───────────────────────────────
    '207940',  # 삼성바이오로직스  바이오 대형
    '326030',  # SK바이오팜        바이오 중형
    '259960',  # 크래프톤          게임
    '005490',  # POSCO홀딩스       철강
    '004020',  # 현대제철          철강
    '010140',  # 삼성중공업        조선
    '329180',  # HD현대중공업      조선
    '000720',  # 현대건설          건설
    '139480',  # 이마트            유통
    '097950',  # CJ제일제당        식품
    '006800',  # 미래에셋증권      증권
    '000810',  # 삼성화재          보험(손해)
    '032640',  # LG유플러스        통신

    # ── KOSDAQ 대표주 신규 추가 (고변동성·이벤트 드리븐 패턴) ──────
    '247540',  # 에코프로비엠      KOSDAQ 2차전지 소재
    '086520',  # 에코프로          KOSDAQ 2차전지 지주
    '196170',  # 알테오젠          KOSDAQ 바이오
    '068760',  # 셀트리온제약      KOSDAQ 바이오
    '145020',  # 휴젤              KOSDAQ 바이오/뷰티
    '293490',  # 카카오게임즈      KOSDAQ 게임
    '112040',  # 위메이드          KOSDAQ 게임/블록체인
]

# 모델별 하이퍼파라미터 (model_params JSON과 동일하게 유지)
MODEL_CONFIGS: Dict[str, dict] = {
    'random_forest': {
        'class': RandomForestRegressor,
        'params': dict(
            n_estimators=300,
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=15,
            max_features=0.5,
            random_state=42,
            n_jobs=-1,
        ),
    },
    'gradient_boosting': {
        'class': GradientBoostingRegressor,
        'params': dict(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            subsample=0.7,
            random_state=42,
        ),
    },
    'xgboost': {
        'class': xgb.XGBRegressor,
        'params': dict(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=15,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        ),
    },
}

FEATURE_COLS = [
    # 기존
    'rsi', 'macd_diff', 'price_sma_20_ratio', 'vol_change',
    # 추세 (multi-timeframe)
    'price_sma_5_ratio', 'rsi_change', 'macd_diff_change',
    # 볼린저 밴드
    'bb_position', 'bb_width',
    # 거래량
    'vol_ratio',
    # 모멘텀
    'stoch_k', 'stoch_d', 'cci',     # stoch_d 추가
    # 변동성
    'atr_ratio',
    # 당일 캔들
    'candle_body',
    # OBV 변화율
    'obv_change',
    # ── 모멘텀 팩터 (신규) ──────────────────────────
    'return_1m', 'return_3m',
    'high_52w_ratio',
    'mom_accel',
    # ── 시장 상대강도 (신규) ────────────────────────
    'rs_vs_mkt_1m', 'rs_vs_mkt_3m',
]

# ───────────────────────────── 데이터 수집 ─────────────────────────────

def build_features(df_ind: pd.DataFrame,
                   market_df: pd.DataFrame = None) -> pd.DataFrame:
    """지표 DataFrame에서 모델 입력 특성을 추출 (prediction_model._extract_features와 동일)"""
    feat = pd.DataFrame(index=df_ind.index)

    # ── 기존 4개 ──────────────────────────────────
    feat['rsi']               = df_ind['rsi']
    feat['macd_diff']         = df_ind['macd_diff']
    feat['price_sma_20_ratio'] = df_ind['close'] / df_ind['sma_20']
    feat['vol_change']        = df_ind['volume'].pct_change()

    # ── 추세 (multi-timeframe) ────────────────────
    feat['price_sma_5_ratio'] = df_ind['close'] / df_ind['sma_5']
    feat['rsi_change']        = df_ind['rsi'].diff()
    feat['macd_diff_change']  = df_ind['macd_diff'].diff()

    # ── 볼린저 밴드 ───────────────────────────────
    bb_range = (df_ind['bb_high'] - df_ind['bb_low']).replace(0, np.nan)
    feat['bb_position']       = (df_ind['close'] - df_ind['bb_low']) / bb_range
    feat['bb_width']          = bb_range / df_ind['bb_mid']

    # ── 거래량 ────────────────────────────────────
    feat['vol_ratio']         = df_ind['volume'] / df_ind['vol_sma_20'].replace(0, np.nan)

    # ── 모멘텀 ────────────────────────────────────
    feat['stoch_k']           = df_ind['stoch_k']
    feat['stoch_d']           = df_ind['stoch_d']                           # 신규
    feat['cci']               = df_ind['cci']

    # ── 변동성 ────────────────────────────────────
    feat['atr_ratio']         = df_ind['atr'] / df_ind['close']

    # ── 당일 캔들 ─────────────────────────────────
    feat['candle_body']       = (df_ind['close'] - df_ind['open']) / df_ind['open']

    # ── OBV 변화율 ────────────────────────────────
    feat['obv_change']        = df_ind['obv'].pct_change().clip(-1, 1)

    # ── 모멘텀 팩터 (신규) ────────────────────────
    feat['return_1m']         = df_ind['close'].pct_change(20)
    feat['return_3m']         = df_ind['close'].pct_change(60)
    feat['high_52w_ratio']    = df_ind['close'] / df_ind['close'].rolling(252, min_periods=60).max()
    feat['mom_accel']         = feat['return_1m'] - feat['return_3m'] / 3.0

    # ── 시장 상대강도 (신규) ──────────────────────
    if market_df is not None and not market_df.empty:
        aligned = market_df.reindex(feat.index, method='ffill')
        feat['rs_vs_mkt_1m'] = (feat['return_1m'] - aligned.get('return_1m', 0)).fillna(0)
        feat['rs_vs_mkt_3m'] = (feat['return_3m'] - aligned.get('return_3m', 0)).fillna(0)
    else:
        feat['rs_vs_mkt_1m'] = 0.0
        feat['rs_vs_mkt_3m'] = 0.0

    # inf(거래량 0, 분모 0 등) → NaN 치환 후 제거
    return feat.replace([np.inf, -np.inf], np.nan).dropna()


# 날짜별 크로스섹셔널 순위를 신뢰하기 위한 최소 종목 수
MIN_STOCKS_PER_DATE = 5


def _fetch_market_returns(symbol: str, period: str) -> pd.DataFrame:
    """시장 지수의 롤링 수익률 DataFrame 반환 (시장 상대강도 피처 계산용).

    컬럼: return_1m (pct_change 20), return_3m (pct_change 60)
    """
    try:
        raw = data_provider.get_ohlcv(symbol, period=period)
        if raw.empty:
            return pd.DataFrame()
        mkt = pd.DataFrame(index=raw.index)
        mkt['return_1m'] = raw['close'].pct_change(20)
        mkt['return_3m'] = raw['close'].pct_change(60)
        logger.info(f"  [시장] {symbol} 수익률 데이터 {len(mkt)}개 로드 완료")
        return mkt
    except Exception as e:
        logger.warning(f"  [시장] {symbol} 데이터 로드 실패: {e}")
        return pd.DataFrame()


def _collect_stock_features(code: str, period: str, future_days: int,
                             market_df: pd.DataFrame = None) -> pd.DataFrame:
    """단일 종목의 (날짜, 특성, 미래수익률) DataFrame 반환.

    반환 DataFrame: 날짜 인덱스, FEATURE_COLS 컬럼 + 'raw_return' 컬럼
    """
    try:
        df = data_provider.get_ohlcv(code, period=period)
        if df is None or df.empty or len(df) < 60:
            logger.warning(f"  [{code}] 데이터 부족 ({len(df) if df is not None else 0}행) — 건너뜀")
            return pd.DataFrame()

        df_ind = indicators.calculate_all(df)
        if df_ind.empty:
            return pd.DataFrame()

        feat = build_features(df_ind, market_df=market_df)
        if len(feat) <= future_days:
            return pd.DataFrame()

        # 미래 수익률 계산 (절대값, 아직 순위화하지 않음)
        close       = df_ind['close'].reindex(feat.index)
        future_ret  = (close.shift(-future_days) - close) / close

        # 마지막 future_days 행은 미래 종가가 없으므로 제외
        valid_idx   = feat.index[:-future_days]
        result      = feat.loc[valid_idx].copy()
        result['raw_return'] = future_ret.loc[valid_idx]

        valid = result.dropna()
        logger.info(f"  [{code}] {len(valid)}개 샘플 수집")
        return valid

    except Exception as exc:
        logger.error(f"  [{code}] 처리 오류: {exc}")
        return pd.DataFrame()


def fetch_train_test_samples(
    codes: List[str], period: str, future_days: int, test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """크로스섹셔널 상대 강도 순위를 타깃으로 하는 학습/검증 세트 수집.

    타깃 산출 방식
    ──────────────
    각 날짜마다 해당일에 데이터가 있는 모든 종목의 미래 수익률을
    퍼센타일 순위(0~100)로 변환한다.
    예) 40개 종목 중 수익률 상위 10%면 target ≈ 90

    장점
    ────
    • 시장 전체 방향(상승장/하락장)에 관계없이 상대적 강세·약세를 학습
    • 절대 수익률 예측보다 노이즈가 적어 R² 개선 효과 큼
    • 추천 목적(순위가 높은 종목 선별)과 타깃이 정합

    분할 방식
    ─────────
    날짜 공간을 전역으로 분할: 앞 (1-test_ratio) 기간 → 학습,
    뒤 test_ratio 기간 → 검증. 동일 날짜의 모든 종목이 같은 세트에 속한다.
    """
    # 1단계: KS11 시장 수익률 사전 로드 (시장 상대강도 피처용, 실패 시 None)
    market_df = _fetch_market_returns('KS11', period)
    if market_df.empty:
        logger.warning("KS11 시장 데이터 미수신 — rs_vs_mkt 피처는 0으로 채워집니다.")
        market_df = None

    # 2단계: 전 종목 특성 + 미래 수익률 수집 (날짜 인덱스 유지)
    frames = [_collect_stock_features(c, period, future_days, market_df) for c in codes]
    frames = [f for f in frames if not f.empty]

    if not frames:
        raise RuntimeError("수집된 학습 샘플이 없습니다. 네트워크 또는 종목 코드를 확인하세요.")

    df_all = pd.concat(frames)   # 날짜 인덱스, 비고유 (종목 여러 개)

    # 2단계: 날짜별 크로스섹셔널 퍼센타일 순위 산출 (0~100)
    df_all['target'] = (
        df_all.groupby(df_all.index)['raw_return']
        .rank(pct=True) * 100.0
    )

    # 날짜별 종목 수가 MIN_STOCKS_PER_DATE 미만인 날은 순위 신뢰도가 낮으므로 제외
    stocks_per_date = df_all.groupby(df_all.index)['raw_return'].count()
    valid_dates     = stocks_per_date[stocks_per_date >= MIN_STOCKS_PER_DATE].index
    df_all          = df_all[df_all.index.isin(valid_dates)]

    # 3단계: 날짜 기준 전역 시계열 분할
    all_dates  = sorted(df_all.index.unique())
    n_dates    = len(all_dates)
    split_idx  = int(n_dates * (1.0 - test_ratio))
    split_date = all_dates[split_idx]

    keep_cols = [c for c in FEATURE_COLS if c in df_all.columns] + ['target']
    df_train  = df_all[df_all.index <  split_date][keep_cols].dropna()
    df_test   = df_all[df_all.index >= split_date][keep_cols].dropna()

    logger.info(
        f"\n타깃: 크로스섹셔널 퍼센타일 순위 (0~100)"
        f"\n분할 기준일: {split_date.date()}"
        f"\n총 날짜: {n_dates} (학습 {split_idx}일 / 검증 {n_dates - split_idx}일)"
        f"\n총 샘플: 학습 {len(df_train)} / 검증 {len(df_test)}"
        f"\n날짜별 평균 종목 수: {stocks_per_date[valid_dates].mean():.1f}"
    )
    return df_train, df_test


# ───────────────────────────── 학습 및 저장 ─────────────────────────────

def train_and_save(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """모델 학습 → 평가 → 모델/스케일러/파라미터 저장.

    df_train / df_test 는 fetch_train_test_samples()가 반환한 시계열 분리 세트.
    """
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(PARAMS_DIR, exist_ok=True)

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train['target'].values

    if df_test.empty:
        logger.warning("검증 세트가 없습니다. 학습 세트 성능만 기록됩니다.")
        X_test, y_test = X_train, y_train
    else:
        X_test = df_test[FEATURE_COLS].values
        y_test = df_test['target'].values

    # 기준선 RMSE (학습 세트 평균으로 예측)
    baseline_rmse = np.sqrt(np.mean((y_test - y_train.mean()) ** 2))
    logger.info(f"\n기준선 RMSE: {baseline_rmse:.4f}")
    logger.info(f"학습 샘플: {len(X_train)}, 검증 샘플: {len(X_test)}\n")

    results = []

    for name, cfg in MODEL_CONFIGS.items():
        logger.info(f"{'─'*40}")
        logger.info(f"  학습 중: {name}")

        t0 = time.time()

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        model = cfg['class'](**cfg['params'])
        model.fit(X_tr, y_train)

        duration = time.time() - t0

        y_pred_tr  = model.predict(X_tr)
        train_r2   = r2_score(y_train, y_pred_tr)

        y_pred = model.predict(X_te)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)
        improvement = (baseline_rmse - rmse) / baseline_rmse * 100

        logger.info(f"  RMSE: {rmse:.4f}  (기준선 대비 {improvement:+.2f}%)")
        logger.info(f"  R²  : {r2:.4f}  (학습 R²: {train_r2:.4f}  과적합 gap: {train_r2 - r2:.4f})")
        logger.info(f"  소요: {duration:.1f}초")

        # 모델 + 스케일러 저장
        model_path  = os.path.join(MODEL_DIR,  f"{name}_model.pkl")
        scaler_path = os.path.join(MODEL_DIR,  f"{name}_scaler.pkl")
        joblib.dump(model,  model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"  저장: {model_path}")
        logger.info(f"  저장: {scaler_path}")

        # 파라미터 메타데이터 저장
        version = f"{name}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        meta = {
            "parameters": {
                k: v for k, v in cfg['params'].items()
                if k not in ('random_state', 'n_jobs', 'verbosity')
            },
            "training_samples":      int(len(X_train)),
            "test_rmse":             round(rmse, 4),
            "test_r2":               round(r2, 4),
            "training_duration":     round(duration, 1),
            "performance_improvement": round(improvement, 2),
            "saved_at":              datetime.now().isoformat(),
            "model_version":         version,
        }
        params_path = os.path.join(PARAMS_DIR, f"{name}_params.json")
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        results.append((name, rmse, r2, improvement))

    # 최종 요약
    logger.info(f"\n{'═'*40}")
    logger.info("  학습 완료 요약")
    logger.info(f"{'─'*40}")
    for name, rmse, r2, imp in results:
        logger.info(f"  {name:<22} RMSE={rmse:.4f}  R²={r2:.4f}  개선={imp:+.2f}%")
    logger.info(f"{'═'*40}")
    logger.info("✅ 모든 모델 저장 완료")
    logger.info(f"   경로: {MODEL_DIR}")


# ───────────────────────────── 진입점 ─────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='ML 주가 예측 모델 학습')
    parser.add_argument(
        '--period', default='2y',
        choices=['1y', '2y', '3m', '6m'],
        help='학습 데이터 기간 (기본값: 2y)'
    )
    parser.add_argument(
        '--future-days', type=int, default=5,
        help='예측 대상 기간 (거래일 수, 기본값: 5)'
    )
    parser.add_argument(
        '--stocks', nargs='+', default=None,
        help='학습에 사용할 종목 코드 (미지정 시 기본 종목 리스트 사용)'
    )
    parser.add_argument(
        '--test-ratio', type=float, default=0.2,
        help='검증 세트 비율 — 각 종목의 마지막 N%% 기간을 검증에 사용 (기본값: 0.2)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    stocks = args.stocks or DEFAULT_TRAINING_STOCKS

    logger.info("=" * 40)
    logger.info("  ML 모델 학습 시작")
    logger.info(f"  종목 수     : {len(stocks)}")
    logger.info(f"  데이터 기간 : {args.period}")
    logger.info(f"  예측 기간   : {args.future_days}거래일 후")
    logger.info(f"  검증 비율   : {args.test_ratio * 100:.0f}% (시계열 후반부)")
    logger.info(f"  타깃 변수   : {args.future_days}거래일 후 크로스섹셔널 퍼센타일 순위")
    logger.info(f"  피처 수     : {len(FEATURE_COLS)}개 (모멘텀 팩터 + 시장 상대강도 포함)")
    logger.info("=" * 40)

    logger.info("\n[1/2] 학습 데이터 수집 중...")
    df_train, df_test = fetch_train_test_samples(
        stocks, period=args.period,
        future_days=args.future_days, test_ratio=args.test_ratio
    )

    logger.info("\n[2/2] 모델 학습 및 저장 중...")
    train_and_save(df_train, df_test)
