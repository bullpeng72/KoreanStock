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
        # 절대 경로 설정
        self.model_dir = os.path.join(config.BASE_DIR, "models", "saved", "prediction_models")
        self.params_dir = os.path.join(config.BASE_DIR, "models", "saved", "model_params")
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
                    logger.info(f"✅ Loaded ML model & scaler: {name}")
                except Exception as e:
                    logger.error(f"❌ Error loading {name} package: {e}")
            else:
                missing = []
                if not os.path.exists(model_path): missing.append("model.pkl")
                if not os.path.exists(scaler_path): missing.append("scaler.pkl")
                logger.warning(f"⚠️ Skipping {name}: Missing {', '.join(missing)}")


    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """원본 OHLCV에서 지표를 계산한 뒤 특성(Feature) 생성"""
        if df.empty: return df
        df_ind = indicators.calculate_all(df)
        return self._extract_features(df_ind)

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """이미 지표가 계산된 데이터프레임에서 특성(Feature)만 추출"""
        if df.empty: return df
        features = pd.DataFrame(index=df.index)
        features['rsi'] = df['rsi']
        features['macd_diff'] = df['macd_diff']
        features['price_sma_20_ratio'] = df['close'] / df['sma_20']
        features['vol_change'] = df['volume'].pct_change()
        return features.dropna()

    def predict(self, code: str, df: pd.DataFrame, df_with_indicators: pd.DataFrame = None, fallback_score: float = None) -> Dict[str, Any]:
        """앙상블 예측 수행. df_with_indicators가 전달되면 지표 재계산을 생략한다."""
        if df_with_indicators is not None:
            features = self._extract_features(df_with_indicators)
        else:
            features = self.prepare_features(df)
        if features.empty:
            return {"error": "Insufficient data for ML prediction"}

        latest_x = features.iloc[-1:].values

        preds = []
        for name, model in self.models.items():
            try:
                x = latest_x
                if name in self.scalers:
                    x = self.scalers[name].transform(x)
                p = model.predict(x)[0]
                preds.append(p)
            except:
                continue

        if not preds:
            # 저장된 모델이 없을 때: tech_score 폴백 → 피처 휴리스틱 → 50.0 순으로 대체
            if fallback_score is not None:
                score = round(float(np.clip(fallback_score, 0.0, 100.0)), 2)
                logger.warning(f"No ML models loaded for {code}. Using tech_score fallback: {score}")
                return {"ensemble_score": score, "model_count": 0, "note": "fallback_to_tech_score"}

            latest = features.iloc[-1]
            rsi = float(latest.get('rsi', 50))
            macd_diff = float(latest.get('macd_diff', 0))
            price_sma_ratio = float(latest.get('price_sma_20_ratio', 1.0))
            # RSI 50 기준 대칭 점수, MACD 방향, 가격/SMA 위치를 가중 합산
            heuristic = 50.0 + (50.0 - rsi) * 0.3 + (10.0 if macd_diff > 0 else -10.0) + (price_sma_ratio - 1.0) * 50.0
            score = round(float(np.clip(heuristic, 0.0, 100.0)), 2)
            logger.warning(f"No ML models loaded for {code}. Using feature heuristic fallback: {score}")
            return {"ensemble_score": score, "model_count": 0, "note": "fallback_heuristic"}

        # 앙상블 점수 (0~100 범위로 클리핑)
        avg_pred = float(np.clip(np.mean(preds), 0.0, 100.0))
        return {
            "ensemble_score": avg_pred,
            "model_count": len(preds),
            "prediction_date": datetime.now().strftime('%Y-%m-%d')
        }

prediction_model = StockPredictionModel()
