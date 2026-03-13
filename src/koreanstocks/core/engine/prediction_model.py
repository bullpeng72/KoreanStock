import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from koreanstocks.core.config import config
from koreanstocks.core.constants import MIN_MODEL_AUC
from koreanstocks.core.engine.indicators import indicators
from koreanstocks.core.engine.features import build_features as _build_features, BASE_FEATURE_COLS
from koreanstocks.core.engine import tcn_model as _tcn

logger = logging.getLogger(__name__)

# 피처 목록 단일 소스는 features.BASE_FEATURE_COLS (중복 정의 제거)
_FEATURE_COLS = BASE_FEATURE_COLS
# 구버전(22/34/37/23/25/18/17) 구분용 문서 상수
_BASE_FEATURE_COUNT = len(BASE_FEATURE_COLS)
# AUC 가중치 하한선: AUC 기반 가중치 = (AUC - 0.5) / 상수
# AUC=0.55 → weight=0.05, AUC=0.60 → weight=0.10 (최대 2배 차이 허용)
_AUC_WEIGHT_FLOOR = 0.50   # AUC - 0.5가 이 값 이하면 weight=0 처리
# 모델 품질 게이트: test_auc 가 이 값 미만이면 모델 로드를 거부하고 tech_score 폴백 사용
# AUC < 0.52 = 랜덤(0.5)과 사실상 동등 → 예측력 없음
_MIN_MODEL_AUC = MIN_MODEL_AUC  # core/constants.py 단일 소스
# 하위 호환: 구버전 R² 기반 모델 로드 시 fallback용 (사용 안 함)
_MIN_MODEL_R2 = 0.0


class StockPredictionModel:
    """머신러닝 기반 주가 예측 모델 클래스 (앙상블)"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_weights = {}   # name → 1/RMSE 가중치 (성능 기반 앙상블용)
        self.calibrations: Dict[str, list] = {}  # name → 101분위수 배열 (predict_proba → 0~100)
        # 절대 경로 설정
        self.model_dir = Path(config.BASE_DIR) / "models" / "saved" / "prediction_models"
        self.params_dir = Path(config.BASE_DIR) / "models" / "saved" / "model_params"
        # 시장 지수 당일 캐시 (KS11/KQ11 별도 캐싱, 상대강도 피처용)
        self._market_cache: Dict[str, Any] = {}  # symbol → {'df': DataFrame, 'date': str}
        # TCN 딥러닝 모델 (선택적)
        self._tcn_loaded: Optional[dict] = None
        self._load_existing_models()

    def _load_existing_models(self):
        """저장된 모델 및 스케일러 로드 (한 쌍이 모두 존재할 때만 활성화)"""
        model_names = ['random_forest', 'gradient_boosting', 'lightgbm', 'catboost', 'xgboost_ranker']
        
        if not self.model_dir.exists():
            logger.error(f"Model directory not found: {self.model_dir}")
            return

        for name in model_names:
            model_path = self.model_dir / f"{name}_model.pkl"
            scaler_path = self.model_dir / f"{name}_scaler.pkl"

            # 모델과 스케일러가 모두 존재해야 로드 (정합성 유지)
            if model_path.exists() and scaler_path.exists():
                try:
                    loaded_model = joblib.load(model_path)
                    loaded_scaler = joblib.load(scaler_path)

                    # params JSON에서 품질 지표 확인 — 기준 미달 모델은 로드 거부
                    params_path = self.params_dir / f"{name}_params.json"
                    if params_path.exists():
                        with open(params_path, 'r', encoding='utf-8') as pf:
                            meta = json.load(pf)
                        model_type = meta.get("model_type", "regression")
                        if model_type in ("binary_classifier", "ranker"):
                            auc = float(meta.get("test_auc", 0.0))
                            if auc < _MIN_MODEL_AUC:
                                logger.warning(
                                    f"⚠️  {name} 품질 기준 미달 (test_auc={auc:.4f} < {_MIN_MODEL_AUC}) — "
                                    f"로드 건너뜀. tech_score 폴백으로 동작합니다."
                                )
                                continue
                            # AUC 기반 가중치: (AUC - 0.5) 에 비례
                            self.model_weights[name] = max(auc - _AUC_WEIGHT_FLOOR, 1e-6)
                            # 캘리브레이션 배열 로드 (101분위수) — 타입·범위·단조증가 검증
                            cal = meta.get("calibration")
                            if cal and len(cal) == 101:
                                try:
                                    cal_arr = np.array(cal, dtype=float)
                                    if np.all(np.isfinite(cal_arr)) and np.all(np.diff(cal_arr) >= 0):
                                        self.calibrations[name] = cal_arr.tolist()
                                    else:
                                        logger.warning(f"⚠️  {name} calibration 배열 손상 (NaN/Inf 또는 단조 감소) — 캘리브레이션 비활성화")
                                except (ValueError, TypeError) as _cal_err:
                                    logger.warning(f"⚠️  {name} calibration 파싱 실패: {_cal_err} — 비활성화")
                            label = "ranker" if model_type == "ranker" else "classifier"
                            logger.info(f"✅ Loaded {label}: {name} (auc={auc:.4f}, weight={self.model_weights[name]:.4f})")
                        else:
                            # 구버전 regression 모델 — R² 기준
                            r2   = float(meta.get("test_r2",   0.0))
                            rmse = float(meta.get("test_rmse", 30.0))
                            if r2 < _MIN_MODEL_R2:
                                logger.warning(f"⚠️  {name} 구버전 R² 미달 ({r2:.4f}) — 건너뜀.")
                                continue
                            self.model_weights[name] = 1.0 / max(rmse, 5.0)
                            logger.info(f"✅ Loaded regressor: {name} (r2={r2:.4f})")
                    else:
                        self.model_weights[name] = 0.05  # 파라미터 없으면 기본 가중치

                    self.models[name]  = loaded_model
                    self.scalers[name] = loaded_scaler
                except Exception as e:
                    logger.error(f"❌ Error loading {name} package: {e}")
            else:
                missing = []
                if not model_path.exists(): missing.append("model.pkl")
                if not scaler_path.exists(): missing.append("scaler.pkl")
                logger.warning(f"⚠️ Skipping {name}: Missing {', '.join(missing)}")

        # ── Softmax 정규화: AUC 미세 차이 과민도 완화 ──────────────────────
        # 선형 가중치(AUC-0.5)는 0.019 AUC 차이로 23% 가중치 격차 → 과도한 편중
        # temperature=5: 차이는 유지하되 최대/최소 격차를 ~8%로 압축
        if len(self.model_weights) > 1:
            _names = list(self.model_weights.keys())
            _raw   = np.array([self.model_weights[n] for n in _names])
            _exp   = np.exp(_raw * 5.0)
            _norm  = _exp / _exp.sum()
            for _n, _w in zip(_names, _norm.tolist()):
                self.model_weights[_n] = float(_w)
            logger.info(
                f"앙상블 가중치 softmax 정규화: "
                + ", ".join(f"{n}={self.model_weights[n]:.4f}" for n in _names)
            )

        # ── TCN 딥러닝 모델 로드 (선택적, PyTorch 필요) ────────────────────
        if _tcn.is_available():
            tcn_loaded = _tcn.load_tcn(self.model_dir, self.params_dir)
            if tcn_loaded is not None:
                meta = tcn_loaded["meta"]
                auc  = float(meta.get("test_auc", 0.0))
                if auc >= _MIN_MODEL_AUC:
                    self._tcn_loaded = tcn_loaded
                    self.model_weights["tcn"] = max(auc - _AUC_WEIGHT_FLOOR, 1e-6)
                    cal = meta.get("calibration")
                    if cal and len(cal) == 101:
                        try:
                            cal_arr = np.array(cal, dtype=float)
                            if np.all(np.isfinite(cal_arr)) and np.all(np.diff(cal_arr) >= 0):
                                self.calibrations["tcn"] = cal_arr.tolist()
                        except (ValueError, TypeError):
                            pass
                    logger.info(f"✅ Loaded classifier: tcn (auc={auc:.4f})")
                else:
                    logger.warning(
                        f"⚠️  tcn 품질 기준 미달 (test_auc={auc:.4f} < {_MIN_MODEL_AUC}) — 로드 건너뜀."
                    )


    def _get_market_df(self, index_symbol: str = 'KS11') -> pd.DataFrame:
        """시장 지수 수익률 DataFrame 반환 (KS11=KOSPI, KQ11=KOSDAQ, 당일 캐싱).

        컬럼: return_1m (20d), return_3m (60d) — 인덱스: 날짜
        FDR Yahoo Finance 경유 (^ 접두사 강제) → 실패 시 yfinance 폴백.
        """
        from datetime import date as _date
        from koreanstocks.core.data.provider import data_provider as _dp
        today = _date.today().isoformat()
        cached = self._market_cache.get(index_symbol, {})
        if cached.get('date') == today and not cached.get('df', pd.DataFrame()).empty:
            return cached['df']
        # FDR 1차 시도 — '^' 접두사로 Yahoo Finance 경유 강제 (KRX 직접 접근 차단 회피)
        yf_sym_primary = {'KS11': '^KS11', 'KQ11': '^KQ11'}.get(index_symbol, f'^{index_symbol}')
        try:
            raw = _dp.get_ohlcv(yf_sym_primary, period='2y')
            if not raw.empty:
                mkt = pd.DataFrame(index=raw.index)
                mkt['return_1m'] = raw['close'].pct_change(20)
                mkt['return_3m'] = raw['close'].pct_change(60)
                self._market_cache[index_symbol] = {'df': mkt, 'date': today}
                return mkt
        except Exception as e:
            logger.warning(f"Failed to fetch {yf_sym_primary} via FDR: {e}")
        # yfinance 폴백
        yf_map = {'KS11': '^KS11', 'KQ11': '^KQ11'}
        yf_sym = yf_map.get(index_symbol)
        if yf_sym:
            try:
                import yfinance as yf
                raw = yf.download(yf_sym, period='2y', progress=False)
                if not raw.empty:
                    close = raw.xs('Close', level=0, axis=1).iloc[:, 0] \
                        if isinstance(raw.columns, pd.MultiIndex) else raw['Close']
                    close.index = pd.to_datetime(close.index).tz_localize(None)
                    mkt = pd.DataFrame(index=close.index)
                    mkt['return_1m'] = close.pct_change(20)
                    mkt['return_3m'] = close.pct_change(60)
                    self._market_cache[index_symbol] = {'df': mkt, 'date': today}
                    logger.debug(f"Market data ({index_symbol}) loaded via yfinance fallback.")
                    return mkt
            except Exception as e2:
                logger.warning(f"yfinance fallback for {yf_sym} also failed: {e2}")
        return pd.DataFrame()

    # Phase 1: 확장된 거시 심볼 목록
    _MACRO_SYMBOLS = ['^VIX', '^GSPC', '^IXIC', '^TNX', '^IRX', 'GC=F', 'CL=F', '000300.SS']

    def _get_macro_df(self) -> pd.DataFrame:
        """거시경제 데이터 반환 (당일 캐싱).

        컬럼 (28개 피처 중 거시 10개 소스):
          vix_level, vix_change_5d,          — VIX 레벨·변화율
          sp500_1m, nasdaq_1m,               — 미국 증시
          tnx_level, tnx_change_1m,          — 미국채 10Y 금리
          yield_spread,                      — 장단기 스프레드 (10Y-3M)
          gold_1m, oil_1m,                   — 금·유가
          csi300_1m                          — 중국 CSI300
        인덱스: 날짜(tz-naive)
        """
        from datetime import date as _date
        today = _date.today().isoformat()
        cached = self._market_cache.get('__macro__', {})
        if cached.get('date') == today and not cached.get('df', pd.DataFrame()).empty:
            return cached['df']
        try:
            import yfinance as yf
            raw = yf.download(
                self._MACRO_SYMBOLS, period='2y', progress=False, auto_adjust=True
            )
            if not raw.empty:
                close = (
                    raw.xs('Close', level=0, axis=1)
                    if isinstance(raw.columns, pd.MultiIndex)
                    else raw[['Close']].rename(columns={'Close': self._MACRO_SYMBOLS[0]})
                )
                _idx = pd.to_datetime(close.index)
                macro = pd.DataFrame(
                    index=_idx.tz_convert(None) if _idx.tz is not None else _idx.tz_localize(None)
                )

                def _s(sym: str) -> pd.Series:
                    """심볼 컬럼 안전 조회 — 없으면 NaN 시리즈 반환."""
                    return close[sym] if sym in close.columns else pd.Series(
                        dtype=float, index=macro.index
                    )

                vix = _s('^VIX')
                macro['vix_level']     = vix.values
                macro['vix_change_5d'] = vix.pct_change(5, fill_method=None).values
                macro['sp500_1m']      = _s('^GSPC').pct_change(20, fill_method=None).values
                macro['nasdaq_1m']     = _s('^IXIC').pct_change(20, fill_method=None).values

                tnx = _s('^TNX')
                irx = _s('^IRX')
                macro['tnx_level']     = tnx.values
                macro['tnx_change_1m'] = tnx.diff(20).values           # 절대 변화 (pp)
                macro['yield_spread']  = (tnx - irx).values             # 10Y-3M 스프레드

                macro['gold_1m']       = _s('GC=F').pct_change(20, fill_method=None).values
                macro['oil_1m']        = _s('CL=F').pct_change(20, fill_method=None).values
                macro['csi300_1m']     = _s('000300.SS').pct_change(20, fill_method=None).values

                macro = macro.ffill()
                self._market_cache['__macro__'] = {'df': macro, 'date': today}
                logger.debug(
                    f"[MacroDF] 수집 완료: {len(macro)}행, "
                    f"컬럼={list(macro.columns)}"
                )
                return macro
        except Exception as e:
            logger.warning(f"Macro data fetch failed: {e}")
        return pd.DataFrame()

    def prepare_features(self, df: pd.DataFrame,
                         market_df: pd.DataFrame = None,
                         macro_df: pd.DataFrame = None) -> pd.DataFrame:
        """원본 OHLCV에서 지표를 계산한 뒤 특성(Feature) 생성"""
        if df.empty: return df
        df_ind = indicators.calculate_all(df)
        return self._extract_features(df_ind, market_df=market_df, macro_df=macro_df)

    def _extract_features(self, df: pd.DataFrame,
                          market_df: pd.DataFrame = None,
                          macro_df: pd.DataFrame = None) -> pd.DataFrame:
        """이미 지표가 계산된 데이터프레임에서 특성(Feature)만 추출.

        features.py의 build_features()에 위임 → trainer.py 학습 로직과 완전 동일.
        train/serve 피처 불일치(feature skew)를 구조적으로 방지.
        """
        return _build_features(df, market_df=market_df, macro_df=macro_df)

    def predict(self, code: str, df: pd.DataFrame,
                df_with_indicators: pd.DataFrame = None,
                fallback_score: float = None,
                market: str = '') -> Dict[str, Any]:
        """앙상블 예측 수행. 순수 ML 점수만 반환 (sentiment 블렌딩은 호출 측에서 처리).

        Parameters
        ----------
        df_with_indicators : 이미 지표가 계산된 DataFrame (전달 시 재계산 생략)
        fallback_score     : ML 모델 없을 때 대체할 tech_score
        market             : 'KOSDAQ' 이면 KQ11, 그 외 KS11 벤치마크 사용.
                             호출 측에서 전달하면 내부 get_stock_list() 중복 호출 생략.
        """
        # 종목 시장에 맞는 벤치마크 지수 선택 (KOSDAQ → KQ11, 그 외 → KS11)
        if market:
            index_symbol = 'KQ11' if market == 'KOSDAQ' else 'KS11'
        else:
            # market 미전달 시 stock_list 조회로 시장 탐지 (하위 호환)
            index_symbol = 'KS11'
            try:
                from koreanstocks.core.data.provider import data_provider as _dp
                _sl = _dp.get_stock_list()
                matched = _sl[_sl['code'] == code]
                if not matched.empty and matched.iloc[0].get('market') == 'KOSDAQ':
                    index_symbol = 'KQ11'
            except Exception:
                pass
        market_df = self._get_market_df(index_symbol)
        macro_df  = self._get_macro_df()
        if df_with_indicators is not None:
            features = self._extract_features(df_with_indicators, market_df=market_df, macro_df=macro_df)
        else:
            features = self.prepare_features(df, market_df=market_df, macro_df=macro_df)
        if features.empty:
            return {"error": "Insufficient data for ML prediction"}

        # _FEATURE_COLS 순서로 피처 선택 (학습 시 FEATURE_COLS와 동일 순서)
        missing_cols = [c for c in _FEATURE_COLS if c not in features.columns]
        if missing_cols:
            logger.warning(
                f"[{code}] 피처 {len(missing_cols)}개 누락 ({missing_cols}). "
                f"finta 미설치 또는 지표 계산 실패 → tech_score 폴백"
            )
            if fallback_score is not None:
                score = round(float(np.clip(fallback_score, 0.0, 100.0)), 2)
                return {"ensemble_score": score, "model_count": 0, "note": "fallback_missing_features"}
            return {"error": f"Missing features: {missing_cols}"}
        feat_cols = _FEATURE_COLS
        latest_x = features[feat_cols].iloc[-1:]  # DataFrame (1, n_features) — feature names 보존

        # ── 분류기 / 랜커 분리 앙상블 ───────────────────────────────────────
        # 분류기(predict_proba): 확률 기반 → calibration → 0~100
        # 랜커(predict):         raw score 기반 → calibration → 0~100
        # 최종: classifier 75% + ranker 25% 가중 평균 (랜커는 보조 신호)
        clf_sum, clf_weight = 0.0, 0.0
        rnk_sum, rnk_weight = 0.0, 0.0
        model_count = 0
        for name, model in self.models.items():
            try:
                scaler = self.scalers.get(name)
                if scaler is not None:
                    x = pd.DataFrame(scaler.transform(latest_x.values), columns=feat_cols)
                else:
                    x = latest_x.copy()
                cal = self.calibrations.get(name)
                w   = self.model_weights.get(name, 0.05)
                if hasattr(model, 'predict_proba'):
                    # 분류기: predict_proba → calibration percentile → 0~100
                    p_raw = float(model.predict_proba(x)[0, 1])
                    p = float(np.clip(np.searchsorted(cal, p_raw), 0, 100)) if cal else p_raw * 100.0
                    clf_sum    += p * w
                    clf_weight += w
                else:
                    # 랜커: raw score → calibration percentile → 0~100
                    p_raw = float(model.predict(x)[0])
                    p = float(np.clip(np.searchsorted(cal, p_raw), 0, 100)) if cal else 50.0
                    rnk_sum    += p * w
                    rnk_weight += w
                model_count += 1
            except Exception as e:
                logger.debug(f"[{name}] predict failed: {e}")
                continue

        # ── TCN 추론 (분류기 버킷에 포함) ──────────────────────────────────
        if self._tcn_loaded is not None:
            try:
                feat_matrix = features[feat_cols].values   # [T, F]
                prob = _tcn.predict_proba_tcn(self._tcn_loaded, feat_matrix)
                if prob is not None:
                    cal = self.calibrations.get("tcn")
                    w   = self.model_weights.get("tcn", 0.05)
                    p   = float(np.clip(np.searchsorted(cal, prob), 0, 100)) if cal else prob * 100.0
                    clf_sum    += p * w
                    clf_weight += w
                    model_count += 1
            except Exception as e:
                logger.debug(f"[tcn] predict failed: {e}")

        if model_count == 0:
            # 저장된 모델이 없을 때: tech_score 폴백
            if fallback_score is not None:
                score = round(float(np.clip(fallback_score, 0.0, 100.0)), 2)
                logger.warning(f"No ML models loaded for {code}. Using tech_score fallback: {score}")
                return {"ensemble_score": score, "model_count": 0, "note": "fallback_to_tech_score"}
            else:
                latest = features.iloc[-1]
                macd_diff = float(latest.get('macd_diff', 0))
                atr_ratio = float(latest.get('atr_ratio', 0.02))
                heuristic = 50.0 + (10.0 if macd_diff > 0 else -10.0) - atr_ratio * 200
                score = round(float(np.clip(heuristic, 0.0, 100.0)), 2)
                logger.warning(f"No ML models loaded for {code}. Using feature heuristic fallback: {score}")
                return {"ensemble_score": score, "model_count": 0, "note": "fallback_heuristic"}

        # ── 분류기 75% + 랜커 25% 가중 결합 ──────────────────────────────
        clf_score = clf_sum / clf_weight if clf_weight > 0 else 50.0
        rnk_score = rnk_sum / rnk_weight if rnk_weight > 0 else clf_score
        if clf_weight > 0 and rnk_weight > 0:
            ensemble_score = 0.75 * clf_score + 0.25 * rnk_score
        else:
            ensemble_score = clf_score if clf_weight > 0 else rnk_score
        ensemble_score = float(np.clip(ensemble_score, 0.0, 100.0))
        return {
            "ensemble_score":     round(ensemble_score, 2),
            "model_count":        model_count,
            "prediction_date":    datetime.now().strftime('%Y-%m-%d'),
        }

prediction_model = StockPredictionModel()
