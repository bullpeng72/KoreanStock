"""관심 종목 라우터 — CRUD /api/watchlist"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from koreanstocks.api.dependencies import get_db, get_data_provider

logger = logging.getLogger(__name__)

router = APIRouter(tags=["watchlist"])


class WatchlistAdd(BaseModel):
    code: str
    name: str = ""


@router.get("/watchlist")
def get_watchlist(db=Depends(get_db)):
    """관심 종목 목록"""
    try:
        return db.get_watchlist()
    except Exception as e:
        logger.error(f"watchlist 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watchlist", status_code=201)
def add_to_watchlist(body: WatchlistAdd, db=Depends(get_db), dp=Depends(get_data_provider)):
    """관심 종목 추가. name 미입력 시 종목 리스트에서 자동 조회."""
    name = body.name
    if not name:
        stock_list = dp.get_stock_list()
        row = stock_list[stock_list["code"] == body.code]
        if not row.empty:
            name = row.iloc[0]["name"]
        else:
            # 폴백: 로컬 DB stocks 테이블 (오프라인·비거래일 안전)
            name = db.get_stock_name(body.code)
            if not name:
                logger.warning(f"watchlist 추가: 미인식 종목 코드 {body.code!r} — 분석 실행 시 오류 발생 가능")
                raise HTTPException(
                    status_code=404,
                    detail=f"종목 코드 {body.code!r}를 찾을 수 없습니다. 올바른 6자리 코드인지 확인하세요.",
                )
    db.add_to_watchlist(body.code, name)
    return {"code": body.code, "name": name}


@router.delete("/watchlist/{code}", status_code=204)
def remove_from_watchlist(code: str, db=Depends(get_db)):
    """관심 종목 삭제"""
    try:
        watchlist = db.get_watchlist()
        codes = [w["code"] for w in watchlist]
        if code not in codes:
            raise HTTPException(status_code=404, detail=f"관심 종목 없음: {code}")
        db.remove_from_watchlist(code)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"watchlist 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
