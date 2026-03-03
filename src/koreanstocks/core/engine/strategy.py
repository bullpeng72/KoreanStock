import logging

import numpy as np
import pandas as pd

from koreanstocks.core.engine.indicators import indicators

logger = logging.getLogger(__name__)


class TechnicalStrategy:
    """기술적 지표를 기반으로 매수/매도 시그널을 생성하는 클래스"""

    def generate_signals(self, df: pd.DataFrame, strategy_type: str = "RSI") -> pd.Series:
        """
        다양한 전략에 따른 시그널 생성 (보유 로직 포함).

        pandas .iloc[i] 반복 대신 numpy 배열 직접 접근으로 성능 개선.
        """
        if df.empty:
            logger.warning(f"generate_signals({strategy_type}): 빈 DataFrame — 빈 시그널 반환")
            return pd.Series(dtype=int)

        n = len(df)
        signals_arr = np.zeros(n, dtype=int)

        if strategy_type == "RSI":
            rsi = df['rsi'].to_numpy(dtype=float, na_value=np.nan)
            pos = 0
            for i in range(n):
                if not np.isnan(rsi[i]):
                    if rsi[i] < 40:    # 과매도 기준 (30 → 40 완화)
                        pos = 1
                    elif rsi[i] > 60:  # 과매수 기준 (70 → 60 완화)
                        pos = 0
                signals_arr[i] = pos

        elif strategy_type == "MACD":
            macd     = df['macd'].to_numpy(dtype=float, na_value=np.nan)
            macd_sig = df['macd_signal'].to_numpy(dtype=float, na_value=np.nan)
            pos = 0
            for i in range(n):
                if np.isnan(macd[i]) or np.isnan(macd_sig[i]):
                    signals_arr[i] = pos
                    continue
                if macd[i] > macd_sig[i]:    # 골든크로스 → 매수
                    pos = 1
                elif macd[i] < macd_sig[i]:  # 데드크로스 → 매도
                    pos = 0
                signals_arr[i] = pos

        elif strategy_type == "COMPOSITE":
            rsi      = df['rsi'].to_numpy(dtype=float, na_value=np.nan)
            macd     = df['macd'].to_numpy(dtype=float, na_value=np.nan)
            macd_sig = df['macd_signal'].to_numpy(dtype=float, na_value=np.nan)
            pos = 0
            for i in range(n):
                if np.isnan(rsi[i]) or np.isnan(macd[i]):
                    signals_arr[i] = pos
                    continue
                # 복합 조건: RSI 낮으면서 MACD 골든크로스 시 매수
                if rsi[i] < 50 and macd[i] > macd_sig[i]:
                    pos = 1
                elif rsi[i] > 60 or macd[i] < macd_sig[i]:
                    pos = 0
                signals_arr[i] = pos

        else:
            logger.warning(f"generate_signals: 알 수 없는 strategy_type='{strategy_type}'")

        return pd.Series(signals_arr, index=df.index)


technical_strategy = TechnicalStrategy()
