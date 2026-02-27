"""추천 종목 라우터 — GET /api/recommendations, POST /api/recommendations/run"""
import logging
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, Query
from koreanstocks.api.dependencies import get_db, get_recommendation_agent

logger = logging.getLogger(__name__)
router = APIRouter(tags=["recommendations"])

_running: bool = False  # 중복 실행 방지 플래그
_run_theme_keywords = None
_run_theme_label = "전체"


def _run_analysis(limit: int, market: str, theme_keywords, theme_label: str):
    global _running, _run_theme_keywords, _run_theme_label
    _running = True
    try:
        from koreanstocks.api.dependencies import get_recommendation_agent
        agent = get_recommendation_agent()
        agent.get_recommendations(
            limit=limit, market=market,
            theme_keywords=theme_keywords, theme_label=theme_label,
        )
    except Exception as e:
        logger.error(f"백그라운드 분석 오류: {e}")
    finally:
        _running = False


@router.get("/recommendations")
def list_recommendations(
    date_str: Optional[str] = Query(None, alias="date", description="YYYY-MM-DD, 미입력 시 최신"),
    db=Depends(get_db),
):
    """날짜별 추천 종목 목록 반환. date 미입력 시 최근 세션 데이터."""
    target_date = date_str or db.get_latest_recommendation_date()
    if not target_date:
        return {"date": None, "recommendations": []}
    recs = db.get_recommendations_by_date(target_date)
    return {"date": target_date, "recommendations": recs}


@router.get("/recommendations/dates")
def recommendation_dates(limit: int = Query(30), db=Depends(get_db)):
    """추천 데이터가 존재하는 날짜 목록 (최근순)"""
    return db.get_recommendation_dates(limit=limit)


@router.get("/recommendations/history")
def recommendation_history(days: int = Query(14, ge=1, le=90), db=Depends(get_db)):
    """최근 N일 추천 이력 (히트맵용). 형식: [{code, name, score, action, date}]"""
    return db.get_recommendation_history(days=days)


@router.post("/recommendations/run", status_code=202)
def run_recommendations(
    background_tasks: BackgroundTasks,
    limit: int = Query(5),
    market: str = Query("ALL"),
    theme: str = Query("전체", description="테마: 전체 | AI/인공지능 | 반도체 | 이차전지 | 제약/바이오 | 로봇/자동화"),
):
    """새 추천 분석 실행 (백그라운드). 즉시 202 반환."""
    if _running:
        return {"status": "already_running", "message": "분석이 이미 진행 중입니다."}

    theme_map = {
        "AI/인공지능":  ["AI", "인공지능", "소프트웨어", "데이터"],
        "로봇/자동화":  ["로봇", "자동화", "기계", "장비"],
        "반도체":       ["반도체", "장비", "소재", "부품"],
        "이차전지":     ["배터리", "이차전지", "에너지", "화학"],
        "제약/바이오":  ["제약", "바이오", "의료", "생명"],
    }
    theme_keywords = theme_map.get(theme)

    background_tasks.add_task(_run_analysis, limit, market, theme_keywords, theme)
    return {"status": "started", "message": f"분석 시작 (market={market}, theme={theme}, limit={limit})"}


@router.get("/recommendations/status")
def analysis_status():
    """백그라운드 분석 실행 여부"""
    return {"running": _running}
