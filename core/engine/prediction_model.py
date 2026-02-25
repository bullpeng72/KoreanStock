import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import os
import json

from core.config import config
from core.engine.indicators import indicators
from core.data.database import db_manager

logger = logging.getLogger(__name__)

class StockPredictionModel:
    """머신러닝 기반 주가 예측 모델 클래스 (앙상블)"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_weights = {}   # name → 1/RMSE 가중치 (성능 기반 앙상블용)
        # 절대 경로 설정
        self.model_dir = os.path.join(config.BASE_DIR, "models", "saved", "prediction_models")
        self.params_dir = os.path.join(config.BASE_DIR, "models", "saved", "model_params")
        # 시장 지수 당일 캐시 (KS11/KQ11 별도 캐싱, 상대강도 피처용)
        self._market_cache: Dict[str, Any] = {}  # symbol → {'df': DataFrame, 'date': str}
        self._load_existing_models()

    def _load_existing_models(self):
        """저장된 모델 및 스케일러 로드 (한 쌍이 모두 존재할 때만 활성화)"""
        model_names = ['random_forest', 'gradient_boosting', 'xgboost']
        
        if not os.path.exists(self.model_dir):
            logger.error(f"Model directory not found: {self.model_dir}")
            return

        for name in model_names:
            model_path = os.path.join(self.model_dir, f"{name}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{name}_scaler.pkl")
            
            # 모델과 스케일러가 모두 존재해야 로드 (정합성 유지)
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    loaded_model = joblib.load(model_path)
                    loaded_scaler = joblib.load(scaler_path)

                    self.models[name] = loaded_model
                    self.scalers[name] = loaded_scaler

                    # params JSON에서 test_rmse를 읽어 가중치 산출 (없으면 균등 가중치 1.0)
                    params_path = os.path.join(self.params_dir, f"{name}_params.json")
                    if os.path.exists(params_path):
                        with open(params_path, 'r', encoding='utf-8') as pf:
                            meta = json.load(pf)
                        rmse = float(meta.get("test_rmse", 1.0))
                        self.model_weights[name] = 1.0 / max(rmse, 0.01)
                    else:
                        self.model_weights[name] = 1.0

                    logger.info(f"✅ Loaded ML model & scaler: {name} (weight={self.model_weights[name]:.3f})")
                except Exception as e:
                    logger.error(f"❌ Error loading {name} package: {e}")
            else:
                missing = []
                if not os.path.exists(model_path): missing.append("model.pkl")
                if not os.path.exists(scaler_path): missing.append("scaler.pkl")
                logger.warning(f"⚠️ Skipping {name}: Missing {', '.join(missing)}")


    def _get_market_df(self, index_symbol: str = 'KS11') -> pd.DataFrame:
        """시장 지수 수익률 DataFrame 반환 (KS11=KOSPI, KQ11=KOSDAQ, 당일 캐싱).

        컬럼: return_1m (20d), return_3m (60d) — 인덱스: 날짜
        시장 데이터 미수신 시 빈 DataFrame 반환 (피처 fallback 처리).
        """
        from datetime import date as _date
        from core.data.provider import data_provider as _dp
        today = _date.today().isoformat()
        cached = self._market_cache.get(index_symbol, {})
        if cached.get('date') == today and not cached.get('df', pd.DataFrame()).empty:
            return cached['df']
        try:
            raw = _dp.get_ohlcv(index_symbol, period='2y')
            if not raw.empty:
                mkt = pd.DataFrame(index=raw.index)
                mkt['return_1m'] = raw['close'].pct_change(20)
                mkt['return_3m'] = raw['close'].pct_change(60)
                self._market_cache[index_symbol] = {'df': mkt, 'date': today}
                logger.debug(f"Market data ({index_symbol}) refreshed for relative strength features.")
                return mkt
        except Exception as e:
            logger.warning(f"Failed to fetch {index_symbol} market data: {e}")
        return pd.DataFrame()

    def prepare_features(self, df: pd.DataFrame,
                         market_df: pd.DataFrame = None) -> pd.DataFrame:
        """원본 OHLCV에서 지표를 계산한 뒤 특성(Feature) 생성"""
        if df.empty: return df
        df_ind = indicators.calculate_all(df)
        return self._extract_features(df_ind, market_df=market_df)

    def _extract_features(self, df: pd.DataFrame,
                          market_df: pd.DataFrame = None) -> pd.DataFrame:
        """이미 지표가 계산된 데이터프레임에서 특성(Feature)만 추출"""
        if df.empty: return df
        feat = pd.DataFrame(index=df.index)

        # ── 기존 4개 ──────────────────────────────────
        feat['rsi']               = df['rsi']
        feat['macd_diff']         = df['macd_diff']
        feat['price_sma_20_ratio'] = df['close'] / df['sma_20']
        feat['vol_change']        = df['volume'].pct_change()

        # ── 추세 (multi-timeframe) ────────────────────
        feat['price_sma_5_ratio'] = df['close'] / df['sma_5']
        feat['rsi_change']        = df['rsi'].diff()
        feat['macd_diff_change']  = df['macd_diff'].diff()

        # ── 볼린저 밴드 ───────────────────────────────
        bb_range = (df['bb_high'] - df['bb_low']).replace(0, np.nan)
        feat['bb_position']       = (df['close'] - df['bb_low']) / bb_range
        feat['bb_width']          = bb_range / df['bb_mid']

        # ── 거래량 ────────────────────────────────────
        feat['vol_ratio']         = df['volume'] / df['vol_sma_20'].replace(0, np.nan)

        # ── 모멘텀 (오실레이터) ───────────────────────
        if 'stoch_k' in df.columns:
            feat['stoch_k']       = df['stoch_k']
        if 'stoch_d' in df.columns:
            feat['stoch_d']       = df['stoch_d']
        if 'cci' in df.columns:
            feat['cci']           = df['cci']

        # ── 변동성 ────────────────────────────────────
        if 'atr' in df.columns:
            feat['atr_ratio']     = df['atr'] / df['close']

        # ── OBV 변화율 ────────────────────────────────
        if 'obv' in df.columns:
            feat['obv_change']    = df['obv'].pct_change().clip(-1, 1)

        # ── 당일 캔들 ────────────────────────────────
        feat['candle_body']       = (df['close'] - df['open']) / df['open']

        # ── 모멘텀 팩터 (신규) ────────────────────────
        feat['return_1m']         = df['close'].pct_change(20)
        feat['return_3m']         = df['close'].pct_change(60)
        feat['high_52w_ratio']    = df['close'] / df['close'].rolling(252, min_periods=60).max()
        feat['mom_accel']         = feat['return_1m'] - feat['return_3m'] / 3.0

        # ── 시장 상대강도 (신규) ──────────────────────
        # market_df 없으면 0(중립) 으로 채워 모델 입력 일관성 유지
        if market_df is not None and not market_df.empty:
            aligned = market_df.reindex(feat.index, method='ffill')
            feat['rs_vs_mkt_1m'] = (feat['return_1m'] - aligned.get('return_1m', 0)).fillna(0)
            feat['rs_vs_mkt_3m'] = (feat['return_3m'] - aligned.get('return_3m', 0)).fillna(0)
        else:
            feat['rs_vs_mkt_1m'] = 0.0
            feat['rs_vs_mkt_3m'] = 0.0

        # inf(거래량 0, 분모 0 등) → NaN 치환 후 제거
        return feat.replace([np.inf, -np.inf], np.nan).dropna()

    def predict(self, code: str, df: pd.DataFrame,
                df_with_indicators: pd.DataFrame = None,
                fallback_score: float = None) -> Dict[str, Any]:
        """앙상블 예측 수행. 순수 ML 점수만 반환 (sentiment 블렌딩은 호출 측에서 처리).

        Parameters
        ----------
        df_with_indicators : 이미 지표가 계산된 DataFrame (전달 시 재계산 생략)
        fallback_score     : ML 모델 없을 때 대체할 tech_score
        """
        # 종목 시장에 맞는 벤치마크 지수 선택 (KOSDAQ → KQ11, 그 외 → KS11)
        index_symbol = 'KS11'
        try:
            from core.data.provider import data_provider as _dp
            stock_list = _dp.get_stock_list()
            matched = stock_list[stock_list['code'] == code]
            if not matched.empty and matched.iloc[0].get('market') == 'KOSDAQ':
                index_symbol = 'KQ11'
        except Exception:
            pass
        market_df = self._get_market_df(index_symbol)
        if df_with_indicators is not None:
            features = self._extract_features(df_with_indicators, market_df=market_df)
        else:
            features = self.prepare_features(df, market_df=market_df)
        if features.empty:
            return {"error": "Insufficient data for ML prediction"}

        latest_x = features.iloc[-1:].values

        # RMSE 기반 가중 앙상블: w_i = 1/RMSE_i
        weighted_sum = 0.0
        total_weight = 0.0
        model_count  = 0
        for name, model in self.models.items():
            try:
                x = latest_x.copy()
                if name in self.scalers:
                    x = self.scalers[name].transform(x)
                p = float(model.predict(x)[0])
                w = self.model_weights.get(name, 1.0)
                weighted_sum += p * w
                total_weight += w
                model_count  += 1
            except Exception as e:
                logger.debug(f"[{name}] predict failed: {e}")
                continue

        if model_count == 0:
            # 저장된 모델이 없을 때: tech_score 폴백 → 피처 휴리스틱 순으로 대체
            if fallback_score is not None:
                score = round(float(np.clip(fallback_score, 0.0, 100.0)), 2)
                logger.warning(f"No ML models loaded for {code}. Using tech_score fallback: {score}")
                return {"ensemble_score": score, "model_count": 0, "note": "fallback_to_tech_score"}
            else:
                latest = features.iloc[-1]
                rsi = float(latest.get('rsi', 50))
                macd_diff = float(latest.get('macd_diff', 0))
                price_sma_ratio = float(latest.get('price_sma_20_ratio', 1.0))
                heuristic = 50.0 + (50.0 - rsi) * 0.3 + (10.0 if macd_diff > 0 else -10.0) + (price_sma_ratio - 1.0) * 50.0
                score = round(float(np.clip(heuristic, 0.0, 100.0)), 2)
                logger.warning(f"No ML models loaded for {code}. Using feature heuristic fallback: {score}")
                return {"ensemble_score": score, "model_count": 0, "note": "fallback_heuristic"}

        # RMSE 가중 앙상블 점수 (0~100)
        tech_ml_score = float(np.clip(weighted_sum / total_weight, 0.0, 100.0))
        return {
            "ensemble_score":     round(tech_ml_score, 2),
            "model_count":        model_count,
            "prediction_date":    datetime.now().strftime('%Y-%m-%d'),
        }

prediction_model = StockPredictionModel()
