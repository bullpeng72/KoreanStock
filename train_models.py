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
import pickle
import hashlib
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, r2_score
import joblib
import xgboost as xgb

from koreanstocks.core.config import config
from koreanstocks.core.data.provider import data_provider
from koreanstocks.core.engine.indicators import indicators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_models')

# ───────────────────────────── 설정 ─────────────────────────────

MODEL_DIR       = os.path.join(config.BASE_DIR, "models", "saved", "prediction_models")
PARAMS_DIR      = os.path.join(config.BASE_DIR, "models", "saved", "model_params")
PYKRX_CACHE_DIR = os.path.join(config.BASE_DIR, "data", "storage")

# 학습에 사용할 종목
# 구성 원칙:
#   - KOSPI 대형주 (기존) + KOSDAQ 대표주 + 미학습 섹터 추가
#   - 추천 후보 풀(거래량·등락률 상위)에 자주 등장하는 종목 유형을 최대한 포괄
DEFAULT_TRAINING_STOCKS = [
    # ── KOSPI 기존 대형주 (20개) ──────────────────────────────────
    '005930',  # 삼성전자          반도체
    '000660',  # SK하이닉스        반도체
    '035420',  # NAVER             인터넷
    '005380',  # 현대차            자동차
    '051910',  # LG화학            화학/2차전지
    '006400',  # 삼성SDI           2차전지
    '035720',  # 카카오            인터넷/플랫폼
    '068270',  # 셀트리온          바이오(KOSPI)
    '105560',  # KB금융            은행
    '055550',  # 신한지주          은행
    '003550',  # LG                지주
    '096770',  # SK이노베이션      에너지/2차전지
    '028260',  # 삼성물산          건설/무역
    '066570',  # LG전자            가전/부품
    '017670',  # SK텔레콤          통신
    '030200',  # KT                통신
    '032830',  # 삼성생명          보험
    '000270',  # 기아              자동차
    '012330',  # 현대모비스        자동차부품
    '011170',  # 롯데케미칼        화학

    # ── KOSPI 기존 미학습 섹터 (13개) ────────────────────────────
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

    # ── KOSPI 신규: 금융 (7개) ───────────────────────────────────
    '086790',  # 하나금융지주      은행
    '316140',  # 우리금융지주      은행
    '138040',  # 메리츠금융지주    금융
    '016360',  # 삼성증권          증권
    '039490',  # 키움증권          증권
    '001450',  # 현대해상          보험(손해)
    '005830',  # DB손해보험        보험(손해)

    # ── KOSPI 신규: 방산/항공 (4개) ──────────────────────────────
    '012450',  # 한화에어로스페이스  방산
    '047810',  # 한국항공우주       방산/항공
    '079550',  # LIG넥스원          방산
    '003490',  # 대한항공            항공

    # ── KOSPI 신규: 화장품 (2개) ─────────────────────────────────
    '090430',  # 아모레퍼시픽       화장품
    '051900',  # LG생활건강         화장품

    # ── KOSPI 신규: 제약 (5개) ───────────────────────────────────
    '128940',  # 한미약품           제약
    '000100',  # 유한양행           제약
    '006280',  # 녹십자             제약
    '185750',  # 종근당             제약
    '069620',  # 대웅제약           제약

    # ── KOSPI 신규: IT/게임/전자부품 (6개) ───────────────────────
    '036570',  # 엔씨소프트         게임
    '251270',  # 넷마블             게임
    '009150',  # 삼성전기           전자부품
    '011070',  # LG이노텍           전자부품
    '018260',  # 삼성SDS            IT서비스
    '323410',  # 카카오뱅크         인터넷은행

    # ── KOSPI 신규: 에너지/화학/소재 (6개) ───────────────────────
    '010950',  # S-Oil              정유
    '011780',  # 금호석유           화학
    '003670',  # 포스코퓨처엠       이차전지소재
    '020150',  # 일진머티리얼즈     이차전지소재
    '010060',  # OCI홀딩스          화학
    '028300',  # 에이치엘비         바이오

    # ── KOSPI 신규: 건설/중공업 (7개) ────────────────────────────
    '006360',  # GS건설             건설
    '028050',  # 삼성엔지니어링     엔지니어링
    '034020',  # 두산에너빌리티     에너지/원전
    '241560',  # 두산밥캣           기계
    '298040',  # 효성중공업         중공업/전력기기
    '010120',  # LS일렉트릭         전력기기
    '042660',  # 한화오션           조선

    # ── KOSPI 신규: 유통/물류/식품/해운 (6개) ────────────────────
    '023530',  # 롯데쇼핑           유통
    '086280',  # 현대글로비스       물류
    '000120',  # CJ대한통운         물류
    '004370',  # 농심               식품
    '271560',  # 오리온             식품
    '011200',  # HMM                해운

    # ── KOSPI 신규: 지주/기타 (8개) ──────────────────────────────
    '034730',  # SK                 지주
    '078930',  # GS                 에너지/유통
    '000880',  # 한화               방산/화학
    '103140',  # 풍산               비철금속
    '001040',  # CJ                 지주
    '112610',  # 씨에스윈드         풍력타워
    '298050',  # 효성첨단소재       소재
    '180640',  # 한진칼             항공지주

    # ── KOSDAQ 기존 대표주 (7개) ─────────────────────────────────
    '247540',  # 에코프로비엠       2차전지 소재
    '086520',  # 에코프로           2차전지 지주
    '196170',  # 알테오젠           바이오
    '068760',  # 셀트리온제약       바이오
    '145020',  # 휴젤               바이오/뷰티
    '293490',  # 카카오게임즈       게임
    '112040',  # 위메이드           게임/블록체인

    # ── KOSDAQ 신규: 이차전지소재 (5개) ──────────────────────────
    '066970',  # L&F               이차전지소재
    '278280',  # 천보               이차전지소재
    '348370',  # 엔켐               이차전지소재
    '121600',  # 나노신소재         이차전지소재
    '336370',  # 솔루스첨단소재     이차전지소재

    # ── KOSDAQ 신규: 바이오/제약 (8개) ───────────────────────────
    '096530',  # 씨젠               바이오/진단
    '095700',  # 제넥신             바이오
    '328130',  # 루닛               AI의료
    '338220',  # 뷰노               AI의료
    '086900',  # 메디톡스           바이오/뷰티
    '008930',  # 한미사이언스       제약
    '086450',  # 동국제약           제약
    '085670',  # 케어젠             바이오

    # ── KOSDAQ 신규: 의료기기 (4개) ──────────────────────────────
    '214150',  # 클래시스           의료기기/뷰티
    '214450',  # 파마리서치         제약/뷰티
    '048260',  # 오스템임플란트     의료기기
    '145720',  # 덴티움             의료기기

    # ── KOSDAQ 신규: 뷰티/생활 (5개) ─────────────────────────────
    '278470',  # 에이피알           뷰티
    '237880',  # 클리오             뷰티
    '161890',  # 한국콜마           뷰티
    '192820',  # 코스맥스           뷰티
    '018290',  # 브이티             뷰티

    # ── KOSDAQ 신규: 반도체장비/소재 (12개) ──────────────────────
    '036930',  # 주성엔지니어링     반도체장비
    '240810',  # 원익IPS            반도체장비
    '042700',  # 한미반도체         반도체장비
    '058470',  # 리노공업           반도체
    '095340',  # ISC                반도체
    '222800',  # 심텍               반도체PCB
    '067310',  # 하나마이크론       반도체패키징
    '319660',  # 피에스케이         반도체장비
    '090460',  # 비에이치           FPCB
    '183300',  # 코미코             반도체부품
    '102710',  # 이엔에프테크놀로지 반도체소재
    '029460',  # 케이씨텍           반도체장비

    # ── KOSDAQ 신규: 엔터/미디어 (5개) ───────────────────────────
    '041510',  # SM엔터테인먼트     엔터
    '035900',  # JYP엔터테인먼트   엔터
    '122870',  # YG엔터테인먼트    엔터
    '253450',  # 스튜디오드래곤    드라마제작
    '067160',  # 아프리카TV         미디어

    # ── KOSDAQ 신규: 게임 (6개) ──────────────────────────────────
    '263750',  # 펄어비스           게임
    '192080',  # 더블유게임즈       게임
    '069080',  # 웹젠               게임
    '078340',  # 컴투스             게임
    '225570',  # 넥슨게임즈         게임
    '095660',  # 네오위즈           게임

    # ── KOSDAQ 신규: IT서비스/장비 (5개) ─────────────────────────
    '030190',  # NICE평가정보       IT서비스
    '056190',  # 에스에프에이       장비
    '079940',  # 가비아             IT/도메인
    '054040',  # 포스코DX           IT/자동화
    '196180',  # 파크시스템스       반도체/측정

    # ── KOSDAQ 신규: 화학/소재/기타 (5개) ────────────────────────
    '025900',  # 동화기업           화학
    '005680',  # 코스모화학         화학
    '357780',  # 솔브레인홀딩스     반도체소재
    '091810',  # 두산테스나         반도체
    '101490',  # 에스앤에스텍       반도체부품
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

BASE_FEATURE_COLS = [
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

PYKRX_FEATURE_COLS = [
    # ── 펀더멘털 raw ──────────────────────────────
    'pbr', 'per', 'div',
    # ── 펀더멘털 크로스섹셔널 순위 (날짜별, 0~100) ──
    'pbr_xs', 'per_xs',
    # ── 외국인/기관 5일 누적 순매수 비율 ─────────
    'foreign_5d_ratio', 'inst_5d_ratio',
    # ── 외국인/기관 크로스섹셔널 순위 ────────────
    'foreign_xs', 'inst_xs',
]

FEATURE_COLS = BASE_FEATURE_COLS + PYKRX_FEATURE_COLS  # 22 + 9 = 31

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


def _fetch_pykrx_all_stocks(codes: List[str], period: str) -> Dict[str, Dict]:
    """PyKrx로 전 종목의 펀더멘털·외국인/기관 데이터를 일괄 로드 (파일 캐시 사용).

    반환: {code: {'fundamental': DataFrame, 'trading': DataFrame}}
    캐시: PYKRX_CACHE_DIR/pykrx_cache_{hash}.pkl (sorted codes + period 해시)
    """
    cache_key = hashlib.md5((','.join(sorted(codes)) + period).encode()).hexdigest()[:8]
    cache_file = os.path.join(PYKRX_CACHE_DIR, f"pykrx_cache_{cache_key}.pkl")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            logger.info(f"[PyKrx] 캐시 로드: {cache_file} ({len(cached)}종목)")
            return cached
        except Exception as e:
            logger.warning(f"[PyKrx] 캐시 로드 실패: {e}, 재수집합니다.")

    try:
        from pykrx import stock as pykrx_stock
    except ImportError:
        logger.warning("[PyKrx] pykrx 미설치 — 펀더멘털/투자자 피처 건너뜀")
        return {}

    period_map = {'1y': 365, '2y': 730, '3y': 1095, '3m': 90, '6m': 180}
    days = period_map.get(period, 730)
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days + 30)  # 여유 30일
    start_str = start_dt.strftime('%Y%m%d')
    end_str   = end_dt.strftime('%Y%m%d')

    result: Dict[str, Dict] = {}
    for i, code in enumerate(codes):
        try:
            fund_df  = pykrx_stock.get_market_fundamental_by_date(start_str, end_str, code)
            trade_df = pykrx_stock.get_market_trading_value_by_date(start_str, end_str, code)

            fund_df.index  = pd.to_datetime(fund_df.index)
            trade_df.index = pd.to_datetime(trade_df.index)

            fund_cols  = [c for c in ['PBR', 'PER', 'DIV'] if c in fund_df.columns]
            trade_cols = [c for c in ['외국인합계', '기관합계'] if c in trade_df.columns]

            result[code] = {
                'fundamental': fund_df[fund_cols]  if fund_cols  else pd.DataFrame(),
                'trading':     trade_df[trade_cols] if trade_cols else pd.DataFrame(),
            }
        except Exception as e:
            logger.debug(f"[PyKrx] {code} 수집 실패: {e}")
            result[code] = {'fundamental': pd.DataFrame(), 'trading': pd.DataFrame()}

        if (i + 1) % 20 == 0:
            logger.info(f"[PyKrx] {i+1}/{len(codes)} 종목 수집 완료")

    try:
        os.makedirs(PYKRX_CACHE_DIR, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"[PyKrx] 캐시 저장 완료: {cache_file}")
    except Exception as e:
        logger.warning(f"[PyKrx] 캐시 저장 실패: {e}")

    return result


def _collect_stock_features(code: str, period: str, future_days: int,
                             market_df: pd.DataFrame = None,
                             pykrx_cache: Optional[Dict] = None) -> pd.DataFrame:
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

        # ── PyKrx 피처 합산 (캐시에 해당 종목 데이터가 있을 경우) ────────
        if pykrx_cache and code in pykrx_cache:
            pkx = pykrx_cache[code]

            # 펀더멘털 (PBR, PER, DIV)
            fund_df = pkx.get('fundamental', pd.DataFrame())
            if not fund_df.empty:
                for src_col, dst_col in [('PBR', 'pbr'), ('PER', 'per'), ('DIV', 'div')]:
                    if src_col in fund_df.columns:
                        aligned = fund_df[src_col].reindex(result.index, method='ffill')
                        result[dst_col] = aligned

            # 외국인/기관 5일 누적 순매수 비율 (turnover 대비 정규화)
            trade_df = pkx.get('trading', pd.DataFrame())
            if not trade_df.empty:
                close_series = df_ind['close'].reindex(result.index)
                vol_series   = df_ind['volume'].reindex(result.index)
                turnover_5d  = (close_series * vol_series).rolling(5, min_periods=1).sum().replace(0, np.nan)
                for src_col, dst_col in [('외국인합계', 'foreign_5d_ratio'), ('기관합계', 'inst_5d_ratio')]:
                    if src_col in trade_df.columns:
                        aligned = trade_df[src_col].reindex(result.index, method='ffill').fillna(0)
                        net_5d  = aligned.rolling(5, min_periods=1).sum()
                        result[dst_col] = (net_5d / turnover_5d).clip(-0.5, 0.5).fillna(0)

        # 기술적 지표 NaN만 제거 (PyKrx 피처 NaN은 fetch_train_test_samples에서 처리)
        base_subset = [c for c in BASE_FEATURE_COLS + ['raw_return'] if c in result.columns]
        valid = result.dropna(subset=base_subset)
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

    # 2단계: PyKrx 펀더멘털/투자자 데이터 일괄 수집 (파일 캐시 활용)
    logger.info("[PyKrx] 펀더멘털/투자자 데이터 수집 중...")
    pykrx_cache = _fetch_pykrx_all_stocks(codes, period)
    if not pykrx_cache:
        logger.warning("[PyKrx] 데이터 없음 — 펀더멘털/투자자 피처는 중립값으로 처리됩니다.")

    # 3단계: 전 종목 특성 + 미래 수익률 수집 (날짜 인덱스 유지)
    frames = [_collect_stock_features(c, period, future_days, market_df, pykrx_cache)
              for c in codes]
    frames = [f for f in frames if not f.empty]

    if not frames:
        raise RuntimeError("수집된 학습 샘플이 없습니다. 네트워크 또는 종목 코드를 확인하세요.")

    df_all = pd.concat(frames)   # 날짜 인덱스, 비고유 (종목 여러 개)

    # 4단계: PyKrx 크로스섹셔널 순위 피처 계산 (날짜별 퍼센타일 순위)
    for raw_col, xs_col in [('pbr', 'pbr_xs'), ('per', 'per_xs'),
                             ('foreign_5d_ratio', 'foreign_xs'), ('inst_5d_ratio', 'inst_xs')]:
        if raw_col in df_all.columns:
            df_all[xs_col] = (
                df_all.groupby(df_all.index)[raw_col]
                .rank(pct=True) * 100.0
            )
        else:
            df_all[xs_col] = 50.0

    # PyKrx 피처 NaN 보충 및 없는 컬럼을 중립값으로 초기화
    for col in PYKRX_FEATURE_COLS:
        if col in df_all.columns:
            median_val = df_all[col].median()
            df_all[col] = df_all[col].fillna(median_val if pd.notna(median_val) else 50.0)
        else:
            df_all[col] = 50.0

    # 5단계: 날짜별 크로스섹셔널 퍼센타일 순위 산출 (0~100)
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
    logger.info(f"  피처 수     : {len(FEATURE_COLS)}개 (기술적{len(BASE_FEATURE_COLS)} + PyKrx{len(PYKRX_FEATURE_COLS)})")
    logger.info("=" * 40)

    logger.info("\n[1/2] 학습 데이터 수집 중...")
    df_train, df_test = fetch_train_test_samples(
        stocks, period=args.period,
        future_days=args.future_days, test_ratio=args.test_ratio
    )

    logger.info("\n[2/2] 모델 학습 및 저장 중...")
    train_and_save(df_train, df_test)
