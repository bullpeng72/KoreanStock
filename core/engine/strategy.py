import pandas as pd
import numpy as np
from core.engine.indicators import indicators

class TechnicalStrategy:
    """기술적 지표를 기반으로 매수/매도 시그널을 생성하는 클래스"""

    def generate_signals(self, df: pd.DataFrame, strategy_type: str = "RSI") -> pd.Series:
        """
        다양한 전략에 따른 시그널 생성 (보유 로직 포함)
        """
        if df.empty:
            return pd.Series(dtype=int)

        signals = pd.Series(0, index=df.index)
        current_position = 0  # 0: 미보유, 1: 보유

        if strategy_type == "RSI":
            for i in range(len(df)):
                rsi = df['rsi'].iloc[i]
                if pd.isna(rsi): continue

                if rsi < 40:  # 과매도 기준 완화 (30 -> 40)
                    current_position = 1
                elif rsi > 60:  # 과매수 기준 완화 (70 -> 60)
                    current_position = 0
                
                signals.iloc[i] = current_position
        
        elif strategy_type == "MACD":
            for i in range(len(df)):
                if pd.isna(df['macd'].iloc[i]) or pd.isna(df['macd_signal'].iloc[i]): continue
                
                # 골든크로스 시 매수
                if df['macd'].iloc[i] > df['macd_signal'].iloc[i]:
                    current_position = 1
                # 데드크로스 시 매도
                elif df['macd'].iloc[i] < df['macd_signal'].iloc[i]:
                    current_position = 0
                
                signals.iloc[i] = current_position
            
        elif strategy_type == "COMPOSITE":
            for i in range(len(df)):
                rsi = df['rsi'].iloc[i]
                macd = df['macd'].iloc[i]
                macd_sig = df['macd_signal'].iloc[i]
                
                if pd.isna(rsi) or pd.isna(macd): continue

                # 복합 조건: RSI가 낮으면서 MACD 골든크로스 시 매수
                if rsi < 50 and macd > macd_sig:
                    current_position = 1
                elif rsi > 60 or macd < macd_sig:
                    current_position = 0
                
                signals.iloc[i] = current_position

        return signals

technical_strategy = TechnicalStrategy()
