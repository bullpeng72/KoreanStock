"""시장 현황 라우터 — GET /api/market"""
import logging
from fastapi import APIRouter, Depends
from koreanstocks.api.dependencies import get_data_provider

logger = logging.getLogger(__name__)
router = APIRouter(tags=["market"])


@router.get("/market")
def get_market(dp=Depends(get_data_provider)):
    """KOSPI / KOSDAQ / USD/KRW 지수 현황"""
    try:
        indices = dp.get_market_indices()
        return {
            "KOSPI":  {"name": "KOSPI",  "close": indices.get("KOSPI", 0),   "change": indices.get("KOSPI_change", 0)},
            "KOSDAQ": {"name": "KOSDAQ", "close": indices.get("KOSDAQ", 0),  "change": indices.get("KOSDAQ_change", 0)},
            "USDKRW": {"name": "USD/KRW","close": indices.get("USD_KRW", 0), "change": None},
        }
    except Exception as e:
        logger.error(f"시장 지수 조회 오류: {e}")
        return {"error": str(e)}


@router.get("/market/ranking")
def get_ranking(limit: int = 20, market: str = "ALL", dp=Depends(get_data_provider)):
    """거래량·등락률 상위 종목 코드 목록"""
    try:
        codes = dp.get_market_ranking(limit=limit, market=market)
        stock_list = dp.get_stock_list()
        result = []
        for code in codes:
            row = stock_list[stock_list["code"] == code]
            name = row.iloc[0]["name"] if not row.empty else code
            result.append({"code": code, "name": name})
        return result
    except Exception as e:
        logger.error(f"랭킹 조회 오류: {e}")
        return {"error": str(e)}
