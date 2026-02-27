"""관심 종목 라우터 — CRUD /api/watchlist"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from koreanstocks.api.dependencies import get_db, get_data_provider

router = APIRouter(tags=["watchlist"])


class WatchlistAdd(BaseModel):
    code: str
    name: str = ""


@router.get("/watchlist")
def get_watchlist(db=Depends(get_db)):
    """관심 종목 목록"""
    return db.get_watchlist()


@router.post("/watchlist", status_code=201)
def add_to_watchlist(body: WatchlistAdd, db=Depends(get_db), dp=Depends(get_data_provider)):
    """관심 종목 추가. name 미입력 시 종목 리스트에서 자동 조회."""
    name = body.name
    if not name:
        stock_list = dp.get_stock_list()
        row = stock_list[stock_list["code"] == body.code]
        name = row.iloc[0]["name"] if not row.empty else body.code
    db.add_to_watchlist(body.code, name)
    return {"code": body.code, "name": name}


@router.delete("/watchlist/{code}", status_code=204)
def remove_from_watchlist(code: str, db=Depends(get_db)):
    """관심 종목 삭제"""
    watchlist = db.get_watchlist()
    codes = [w["code"] for w in watchlist]
    if code not in codes:
        raise HTTPException(status_code=404, detail=f"관심 종목 없음: {code}")
    db.remove_from_watchlist(code)
