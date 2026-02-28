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
    force: bool = Query(False, description="오늘 결과가 있어도 강제 재분석"),
    db=Depends(get_db),
):
    """새 추천 분석 실행 (백그라운드). 즉시 202 반환.
    오늘 이미 분석 결과가 DB에 있으면 재사용 (force=true 시 강제 재실행).
    """
    if _running:
        return {"status": "already_running", "message": "분석이 이미 진행 중입니다."}

    # 오늘 이미 분석된 결과가 있으면 GPT 재호출 없이 DB 결과 재사용
    if not force:
        from datetime import date as _date
        today = _date.today().isoformat()
        existing = db.get_recommendations_by_date(today)
        if existing:
            return {
                "status": "cached",
                "message": f"오늘({today}) 분석 결과 {len(existing)}개가 이미 있습니다. 강제 재실행은 ?force=true를 사용하세요.",
            }

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


@router.get("/recommendations/outcomes")
def recommendation_outcomes(
    days: int = Query(90, ge=1, le=365, description="최근 N일간"),
):
    """추천 결과 성과 통계 및 개별 내역 반환.

    stats    — 기간 내 집계 통계 (정답률, 평균 수익률, 목표가 달성률)
    outcomes — 종목별 개별 성과 목록
    """
    from koreanstocks.core.utils.outcome_tracker import get_outcome_stats, get_recent_outcomes
    return {
        "stats":    get_outcome_stats(days=days),
        "outcomes": get_recent_outcomes(days=days),
    }
