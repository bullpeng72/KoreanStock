"""종목 분석 라우터 — GET|POST /api/analysis/{code}"""
import logging
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from koreanstocks.api.dependencies import get_db, get_analysis_agent, get_data_provider

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])

_in_progress: set = set()


def _run_async(code: str, name: str):
    try:
        from koreanstocks.api.dependencies import get_analysis_agent
        get_analysis_agent().analyze_stock(code, name)
    except Exception as e:
        logger.error(f"[{code}] 백그라운드 분석 오류: {e}")
    finally:
        _in_progress.discard(code)


def _resolve_name(code: str, dp) -> str:
    stock_list = dp.get_stock_list()
    row = stock_list[stock_list["code"] == code]
    return row.iloc[0]["name"] if not row.empty else code


@router.get("/analysis/{code}")
def get_analysis(code: str, db=Depends(get_db)):
    """DB에 저장된 최신 종목 분석 결과 조회"""
    history = db.get_analysis_history(code, limit=1)
    if not history:
        raise HTTPException(status_code=404, detail=f"분석 데이터 없음: {code}")
    return history[0]


@router.get("/analysis/{code}/history")
def get_analysis_history(code: str, limit: int = 5, db=Depends(get_db)):
    """종목 분석 이력 (최근 N건)"""
    try:
        return db.get_analysis_history(code, limit=limit)
    except Exception as e:
        logger.error(f"[{code}] 분석 이력 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analysis/{code}", status_code=202)
def trigger_analysis_async(
    code: str,
    background_tasks: BackgroundTasks,
    dp=Depends(get_data_provider),
):
    """비동기 분석 트리거. 즉시 202 반환, DB에 결과 저장 후 GET으로 조회."""
    if code in _in_progress:
        return {"status": "already_running", "code": code}
    name = _resolve_name(code, dp)
    _in_progress.add(code)
    background_tasks.add_task(_run_async, code, name)
    return {"status": "started", "code": code, "name": name}


@router.post("/analysis/{code}/sync")
def run_analysis_sync(
    code: str,
    agent=Depends(get_analysis_agent),
    dp=Depends(get_data_provider),
):
    """동기 실시간 분석 (Watchlist 상세 분석용). 최대 60초 대기."""
    name = _resolve_name(code, dp)
    try:
        result = agent.analyze_stock(code, name)
        if result is None:
            raise HTTPException(status_code=500, detail="분석 결과 없음")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{code}] 동기 분석 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
