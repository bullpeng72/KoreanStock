import pandas as pd
import numpy as np
from typing import Dict, Any, List
from koreanstocks.core.config import config

class Backtester:
    """주식 투자 전략의 성과를 검증하는 백테스팅 엔진"""

    def __init__(self, initial_capital: float = 10000000.0):
        self.initial_capital = initial_capital
        self.fee = config.TRANSACTION_FEE
        self.tax = config.TAX_RATE

    def run(self, df: pd.DataFrame, signals: pd.Series, initial_capital: float = None) -> Dict[str, Any]:
        """
        백테스팅 실행
        :param df: OHLCV 데이터프레임
        :param signals: 매수/매도 시그널
        :param initial_capital: 초기 투자 금액 (None일 경우 클래스 기본값 사용)
        :return: 성과 지표 딕셔너리
        """
        capital = initial_capital if initial_capital is not None else self.initial_capital
        
        if df.empty or len(df) != len(signals):
            return {"error": "Invalid data or signals"}

        results = df.copy()
        results['signal'] = signals
        
        # 수익률 계산 (Daily Returns)
        results['pct_change'] = results['close'].pct_change()
        
        # 전략 수익률
        results['strategy_returns'] = results['signal'].shift(1) * results['pct_change']
        
        # 거래 비용 반영
        results['trade'] = results['signal'].diff().abs().fillna(0)
        cost_mask = results['trade'] > 0
        results.loc[cost_mask, 'strategy_returns'] -= (self.fee + self.tax)

        # 누적 수익률 및 자본금 계산
        strategy_returns = results['strategy_returns'].fillna(0)
        results['cum_returns'] = (1 + strategy_returns).cumprod()
        
        # 첫 번째 행을 원금(1.0)으로 초기화하여 그래프 가독성 향상
        if not results.empty:
            results.iloc[0, results.columns.get_loc('cum_returns')] = 1.0
            
        results['cum_capital'] = results['cum_returns'] * capital

        # 성과 지표
        total_return = (results['cum_returns'].iloc[-1] - 1) * 100
        rolling_max = results['cum_returns'].cummax()
        drawdown = results['cum_returns'] / rolling_max - 1
        mdd = drawdown.min() * 100

        win_rate = (results['strategy_returns'] > 0).sum() / (results['strategy_returns'] != 0).sum() if (results['strategy_returns'] != 0).sum() > 0 else 0
        std = results['strategy_returns'].std()
        sharpe = (results['strategy_returns'].mean() / std) * np.sqrt(config.TRADING_DAYS_PER_YEAR) if std != 0 else 0

        return {
            "total_return_pct": round(total_return, 2),
            "mdd_pct": round(mdd, 2),
            "win_rate": round(win_rate * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "final_capital": int(results['cum_capital'].iloc[-1]),
            "daily_results": results[['close', 'signal', 'cum_returns', 'cum_capital']]
        }

backtester = Backtester()
