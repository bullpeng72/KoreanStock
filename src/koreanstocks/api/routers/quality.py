"""우량주 스크리너 API 라우터"""
import logging

from fastapi import APIRouter, Query

from koreanstocks.core.engine.quality_screener import quality_screener

logger = logging.getLogger(__name__)
router = APIRouter(tags=["quality"])


@router.get("/quality_stocks")
async def get_quality_stocks(
    market: str = Query("ALL", description="시장 필터: ALL | KOSPI | KOSDAQ"),
    roe_min: float = Query(12.0, description="ROE 하한 (%)"),
    op_margin_min: float = Query(10.0, description="영업이익률 하한 (%)"),
    yoy_min: float = Query(0.0, description="영업이익 YoY 하한 (%)"),
    debt_max: float = Query(100.0, description="부채비율 상한 (%)"),
    pbr_max: float = Query(6.0, description="PBR 상한"),
    candidate_limit: int = Query(200, description="시가총액 상위 탐색 종목 수 (100~500)"),
):
    """
    펀더멘털 기반 우량주 스크리닝.

    장기(6개월~) 관점으로 고ROE·고영업이익률·안정적 성장·건전한 재무구조를 갖춘
    종목을 우량 점수(quality_score)로 정렬하여 반환합니다.

    - **quality_score**: ROE·영업이익률·영업이익YoY·부채비율·배당 종합 점수 (0~100)
    - PER 상한 없음 — 비싸도 좋은 기업이면 포함 (가치주 스크리너와의 핵심 차이)
    """
    try:
        results = quality_screener.screen(
            market=market,
            roe_min=roe_min,
            op_margin_min=op_margin_min,
            yoy_min=yoy_min,
            debt_max=debt_max,
            pbr_max=pbr_max,
            candidate_limit=candidate_limit,
            limit=candidate_limit,
        )
        for r in results:
            r.pop("fundamentals", None)
        return {"stocks": results, "total": len(results)}
    except Exception as e:
        logger.error(f"[QUALITY API] 스크리닝 실패: {e}")
        return {"stocks": [], "total": 0, "error": str(e)}


@router.get("/quality_stocks/filters")
async def get_quality_filters():
    """현재 기본 필터 임계값 반환."""
    return quality_screener.get_filter_defaults()
