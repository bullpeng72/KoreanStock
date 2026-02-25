import pytest
import pandas as pd
import numpy as np
from core.utils.backtester import Backtester


def _make_df(closes: list) -> pd.DataFrame:
    """Helper: OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    return pd.DataFrame({
        "open":   closes,
        "high":   closes,
        "low":    closes,
        "close":  closes,
        "volume": [1000] * n,
    })


@pytest.fixture
def bt() -> Backtester:
    return Backtester(initial_capital=1_000_000)


class TestInputValidation:
    def test_empty_dataframe_returns_error(self, bt: Backtester):
        result = bt.run(pd.DataFrame(), pd.Series(dtype=float))
        assert "error" in result

    def test_mismatched_signals_length_returns_error(self, bt: Backtester):
        df = _make_df([100, 110, 120])
        signals = pd.Series([1, 1])  # length mismatch
        result = bt.run(df, signals)
        assert "error" in result


class TestReturns:
    def test_buy_and_hold_on_rising_market(self, bt: Backtester):
        """All-buy signal on steadily rising prices should yield positive total return."""
        closes = [100 + i * 10 for i in range(20)]  # 100 → 290
        df = _make_df(closes)
        signals = pd.Series([1] * 20)
        result = bt.run(df, signals)
        assert result["total_return_pct"] > 0

    def test_short_on_falling_market(self, bt: Backtester):
        """Short signal (-1) on steadily falling prices should yield positive return."""
        closes = [200 - i * 5 for i in range(20)]  # 200 → 105
        df = _make_df(closes)
        signals = pd.Series([-1] * 20)
        result = bt.run(df, signals)
        assert result["total_return_pct"] > 0

    def test_no_position_yields_zero_return(self, bt: Backtester):
        """Signal of 0 throughout means no exposure — return should be ~0."""
        closes = [100, 110, 90, 105, 115]
        df = _make_df(closes)
        signals = pd.Series([0] * 5)
        result = bt.run(df, signals)
        assert result["total_return_pct"] == pytest.approx(0.0, abs=0.01)

    def test_final_capital_consistent_with_return(self, bt: Backtester):
        """final_capital should equal initial_capital × (1 + total_return / 100)."""
        closes = [100, 110, 121, 133]
        df = _make_df(closes)
        signals = pd.Series([1] * 4)
        result = bt.run(df, signals)
        expected = bt.initial_capital * (1 + result["total_return_pct"] / 100)
        assert result["final_capital"] == pytest.approx(expected, rel=0.01)


class TestRiskMetrics:
    def test_mdd_non_positive(self, bt: Backtester):
        """Max drawdown must always be <= 0."""
        closes = [100, 120, 90, 110, 80, 130]
        df = _make_df(closes)
        signals = pd.Series([1] * 6)
        result = bt.run(df, signals)
        assert result["mdd_pct"] <= 0

    def test_win_rate_between_0_and_100(self, bt: Backtester):
        closes = [100, 105, 102, 108, 100, 112]
        df = _make_df(closes)
        signals = pd.Series([1, 1, 0, 1, 1, 0])
        result = bt.run(df, signals)
        assert 0 <= result["win_rate"] <= 100

    def test_sharpe_zero_when_no_variance(self, bt: Backtester):
        """Flat price series → std=0 → sharpe should be 0, not an exception."""
        closes = [100] * 10
        df = _make_df(closes)
        signals = pd.Series([1] * 10)
        result = bt.run(df, signals)
        assert result["sharpe_ratio"] == pytest.approx(0.0, abs=0.01)


class TestTradingCosts:
    def test_trading_costs_reduce_returns(self, bt: Backtester):
        """Frequent signal changes incur costs, reducing return vs a static signal."""
        closes = [100 + i for i in range(20)]
        df = _make_df(closes)
        # Alternating signals → many trades → high costs
        alternating = pd.Series([1 if i % 2 == 0 else 0 for i in range(20)])
        static = pd.Series([1] * 20)
        result_alt = bt.run(df, alternating)
        result_static = bt.run(df, static)
        assert result_alt["total_return_pct"] < result_static["total_return_pct"]
