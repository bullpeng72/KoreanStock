import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class IndicatorCalculator:
    """기술적 지표 계산 및 분석을 담당하는 클래스"""

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 주요 기술적 지표를 계산하여 데이터프레임에 추가"""
        if df.empty or len(df) < 30:
            return df
        
        df = df.copy()
        data_len = len(df)
        
        try:
            # 1. 이동평균 (Trend) - 데이터 길이에 따라 선택적 계산
            df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5, fillna=False)
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20, fillna=False)
            
            if data_len >= 60:
                df['sma_60'] = ta.trend.sma_indicator(df['close'], window=60, fillna=False)
            if data_len >= 120:
                df['sma_120'] = ta.trend.sma_indicator(df['close'], window=120, fillna=False)
            
            # 2. MACD (Trend) - 기본적으로 26일 이상이면 가능
            df['macd'] = ta.trend.macd(df['close'], fillna=False)
            df['macd_signal'] = ta.trend.macd_signal(df['close'], fillna=False)
            df['macd_diff'] = ta.trend.macd_diff(df['close'], fillna=False)
            
            # 3. RSI (Momentum) - 14일 이상이면 가능
            df['rsi'] = ta.momentum.rsi(df['close'], window=14, fillna=False)
            
            # 4. 볼린저 밴드 (Volatility) - 20일 이상이면 가능
            indicator_bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2, fillna=False)
            df['bb_high'] = indicator_bb.bollinger_hband()
            df['bb_mid'] = indicator_bb.bollinger_mavg()
            df['bb_low'] = indicator_bb.bollinger_lband()
            
            # 5. 거래량 지표 (Volume)
            df['vol_sma_20'] = ta.trend.sma_indicator(df['volume'], window=20, fillna=False)
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'], fillna=False)

            # 6. 스토캐스틱 (Momentum)
            stoch = ta.momentum.StochasticOscillator(
                df['high'], df['low'], df['close'], window=14, smooth_window=3, fillna=False
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

            # 7. CCI (Commodity Channel Index)
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20, fillna=False)

            # 8. ATR (Average True Range)
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=14, fillna=False
            )

            # 전략 수립에 필수적인 핵심 지표(RSI, MACD)가 생성되는 시점부터 데이터 유지
            # 장기 SMA가 NaN이더라도 dropna()로 인해 데이터가 통째로 날아가는 것을 방지하기 위해 
            # 필수 지표 컬럼들만 기준으로 dropna 수행
            essential_cols = ['rsi', 'macd', 'macd_signal', 'bb_mid']
            return df.dropna(subset=essential_cols)
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

    def get_composite_score(self, df: pd.DataFrame) -> float:
        """기술적 지표들을 종합하여 0~100 사이의 점수 산출

        구성 (최대 100pt):
          추세 (40pt)  : 단기SMA + 중기SMA60 + MACD
          모멘텀 (30pt): RSI 구간별 차등
          위치+거래량 (30pt): BB 위치(25pt, 추세 맥락 반영) + 거래량 확인(5pt)
        """
        if df.empty or 'rsi' not in df.columns:
            return 50.0

        latest = df.iloc[-1]

        # ── 1. 추세 점수 (40pt max) ─────────────────────────────────
        # sma_60 가용 여부에 따라 MACD 가중치를 조정하여 합계 40pt 유지
        trend_score = 0
        if latest['close'] > latest['sma_20']: trend_score += 10
        if latest['sma_5'] > latest['sma_20']: trend_score += 10

        try:
            sma_60_valid = pd.notna(latest['sma_60'])
        except KeyError:
            sma_60_valid = False

        if sma_60_valid:
            if latest['macd'] > latest['macd_signal']: trend_score += 15
            if latest['close'] > latest['sma_60']:     trend_score += 5  # 중기 추세 확인
        else:
            if latest['macd'] > latest['macd_signal']: trend_score += 20  # sma_60 없으면 MACD 집중

        # ── 2. 모멘텀 점수 (30pt max) ───────────────────────────────
        # RSI: 상승 추세 내 적정 구간(45~65) 최고점
        mom_score = 0
        rsi = latest['rsi']
        if 45 <= rsi <= 65:    mom_score += 30  # 상승 추세 내 적정 구간 (최적)
        elif 35 <= rsi < 45:   mom_score += 22  # 하락 완화, 반등 준비 구간
        elif 65 < rsi <= 75:   mom_score += 18  # 강한 상승 모멘텀 (과매수 경계)
        elif 30 <= rsi < 35:   mom_score += 12  # 과매도 근접, 주의 필요
        elif rsi > 75:         mom_score += 8   # 강한 과매수 (오버히팅)
        else:                  mom_score += 4   # RSI < 30: 깊은 과매도 (고위험)

        # ── 3. 가격 위치 + 거래량 확인 (30pt max) ───────────────────
        vol_score = 0
        bb_range = latest['bb_high'] - latest['bb_low']
        bb_pos   = (latest['close'] - latest['bb_low']) / bb_range if bb_range != 0 else 0.5

        # BB 위치 (25pt): MACD 방향에 따라 최적 구간 이동 (추세 맥락 반영)
        is_uptrend = latest['macd'] > latest['macd_signal']
        if is_uptrend:
            # 상승 추세: 중상단(0.4~0.75) 최적 — 추세 추종 구간
            if 0.4 <= bb_pos <= 0.75:   vol_score += 25
            elif 0.75 < bb_pos <= 0.9:  vol_score += 18
            elif 0.2 <= bb_pos < 0.4:   vol_score += 14
            elif bb_pos > 0.9:          vol_score += 8
            else:                        vol_score += 3   # 하단 이탈 (약세)
        else:
            # 하락/중립 추세: 중하단(0.2~0.5) 최적 — 반등 매수 구간
            if 0.2 <= bb_pos <= 0.5:    vol_score += 25
            elif 0.5 < bb_pos <= 0.7:   vol_score += 18
            elif 0.1 <= bb_pos < 0.2:   vol_score += 12
            elif 0.7 < bb_pos < 0.9:    vol_score += 8
            else:                        vol_score += 3   # 밴드 이탈: 고위험

        # 거래량 확인 (5pt): 20일 평균 대비 1.5배 이상 급증 시 추세 신뢰도 가점
        try:
            if latest['vol_sma_20'] > 0 and latest['volume'] / latest['vol_sma_20'] >= 1.5:
                vol_score += 5
        except (KeyError, TypeError, ZeroDivisionError):
            pass

        return float(trend_score + mom_score + vol_score)

indicators = IndicatorCalculator()
