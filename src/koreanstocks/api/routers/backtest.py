"""백테스트 라우터 — GET /api/backtest"""
import logging
from fastapi import APIRouter, Depends, Query
from koreanstocks.api.dependencies import get_data_provider
from koreanstocks.core.utils.backtester import Backtester
from koreanstocks.core.engine.strategy import TechnicalStrategy
from koreanstocks.core.engine.indicators import indicators as ind_engine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["backtest"])

_STRATEGY_MAP = {
    "RSI": "RSI",
    "MACD": "MACD",
    "COMPOSITE": "COMPOSITE",
}


@router.get("/backtest")
def run_backtest(
    code: str = Query(..., description="종목 코드"),
    strategy: str = Query("RSI", description="전략명: RSI | MACD | COMPOSITE"),
    period: str = Query("1y", description="기간: 3m | 6m | 1y | 2y"),
    initial_capital: float = Query(10_000_000),
    dp=Depends(get_data_provider),
):
    """단일 종목 백테스트 실행"""
    try:
        df = dp.get_ohlcv(code, period=period)
        if df is None or df.empty:
            return {"error": f"데이터 없음: {code}"}

        # 기술적 지표 계산 — 올바른 메서드명: calculate_all
        df_ind = ind_engine.calculate_all(df)

        strategy_type = _STRATEGY_MAP.get(strategy.upper(), "RSI")
        signals = TechnicalStrategy().generate_signals(df_ind, strategy_type=strategy_type)

        backtester = Backtester(initial_capital=initial_capital)
        result = backtester.run(df_ind, signals, initial_capital=initial_capital)

        # daily_results (DataFrame) → JSON 직렬화
        daily_df = result.pop("daily_results", None)
        if daily_df is not None and not daily_df.empty:
            # B&H 누적 수익률 추가
            bnh_series = df_ind["close"] / df_ind["close"].iloc[0]

            result["dates"] = [str(d.date()) for d in daily_df.index]
            result["cum_returns"]     = daily_df["cum_returns"].round(4).tolist()
            result["cum_returns_bnh"] = bnh_series.reindex(daily_df.index).fillna(method="ffill").round(4).tolist()
            result["recent_rows"]     = daily_df.tail(10).reset_index().to_dict(orient="records")

            # 각 dict의 date key를 문자열로 변환
            for row in result["recent_rows"]:
                for k, v in row.items():
                    try:
                        row[k] = round(float(v), 4)
                    except (TypeError, ValueError):
                        row[k] = str(v)

            # B&H 수익률
            result["bnh_return_pct"] = round((bnh_series.iloc[-1] - 1) * 100, 2)

        return {"code": code, "strategy": strategy, "period": period, **result}
    except Exception as e:
        logger.error(f"백테스트 오류 [{code}]: {e}", exc_info=True)
        return {"error": str(e)}
