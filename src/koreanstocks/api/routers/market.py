"""시장 현황 라우터 — GET /api/market"""
import logging
import math
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeout
from fastapi import APIRouter, Depends
from koreanstocks.api.dependencies import get_data_provider
from koreanstocks.core.config import config

logger = logging.getLogger(__name__)
router = APIRouter(tags=["market"])


def _safe_num(val, default=0):
    """NaN / Inf → JSON 안전 기본값으로 변환"""
    if val is None:
        return default
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


# ── 데이터 소스 헬스체크 헬퍼 ──────────────────────────────────────────────

def _run_check(fn, timeout: int = 12) -> dict:
    """단일 체크 함수를 타임아웃 내에 실행."""
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            result = ex.submit(fn).result(timeout=timeout)
        result.setdefault("latency_ms", int((time.time() - t0) * 1000))
        return result
    except _FuturesTimeout:
        return {"status": "error", "detail": f"타임아웃 ({timeout}초 초과)", "latency_ms": timeout * 1000}
    except Exception as e:
        return {"status": "error", "detail": str(e)[:120], "latency_ms": int((time.time() - t0) * 1000)}


def _chk_fdr_ohlcv():
    import FinanceDataReader as fdr
    start = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    df = fdr.DataReader('005930', start)
    if df.empty:
        return {"status": "warn", "detail": "데이터 없음 (장 전 또는 공휴일)"}
    last = df.index[-1].strftime('%Y-%m-%d')
    return {"status": "ok", "detail": f"삼성전자 최근 {len(df)}일 수신 (마지막: {last})"}


def _chk_fdr_listing():
    """FDR StockListing 직접 조회 상태 (KRX 정책 차단 여부 확인)."""
    import FinanceDataReader as fdr
    try:
        df = fdr.StockListing('KOSPI')
        if not df.empty:
            return {"status": "ok", "detail": f"KOSPI {len(df):,}종목 수신 (FDR 직접)"}
    except Exception as e:
        detail_suffix = f" ({type(e).__name__}: {str(e)[:80]})"
    else:
        detail_suffix = " (빈 DataFrame 반환)"
    return {
        "status": "warn",
        "detail": f"FDR StockListing 차단 중 (KRX 세션 정책) — KIND API 폴백 동작{detail_suffix}",
    }


def _chk_kind_api():
    """KIND API 직접 조회 — 현재 전종목 목록의 실질 주력 소스."""
    import requests
    from io import BytesIO
    import pandas as pd
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    r = requests.get(
        'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13',
        headers=headers, timeout=10,
    )
    r.raise_for_status()
    df_raw = pd.read_html(BytesIO(r.content), encoding='euc-kr')[0]
    kospi  = len(df_raw[df_raw['시장구분'] == '유가'])
    kosdaq = len(df_raw[df_raw['시장구분'] == '코스닥'])
    return {"status": "ok", "detail": f"KOSPI {kospi:,} · KOSDAQ {kosdaq:,}종목 수신"}


def _chk_naver_sise():
    """네이버 금융 시세 — 거래량 조회 2차 폴백 연결 상태 확인 (KOSPI 1페이지 샘플)."""
    import requests, re
    from bs4 import BeautifulSoup
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    r = requests.get(
        'https://finance.naver.com/sise/sise_market_sum.naver',
        params={'sosok': '0', 'page': '1'},
        headers=headers, timeout=10,
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    stocks = [a['href'].split('code=')[-1] for a in soup.select('table.type_2 td a[href*="code="]')]
    if not stocks:
        return {"status": "warn", "detail": "종목 파싱 실패 (HTML 구조 변경 가능성)"}
    pager = soup.select('.pgRR a')
    last_page = 0
    if pager:
        m = re.search(r'page=(\d+)', pager[-1]['href'])
        if m:
            last_page = int(m.group(1))
    return {"status": "ok", "detail": f"KOSPI 1p/{last_page}p — {len(stocks)}종목 수신 (code·거래량·등락률)"}


def _chk_fdr_index():
    import FinanceDataReader as fdr
    start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    df = fdr.DataReader('^KS11', start)
    if df.empty:
        return {"status": "warn", "detail": "지수 데이터 없음"}
    val = float(df.iloc[-1]['Close'])
    return {"status": "ok", "detail": f"KOSPI {val:,.2f}"}



def _chk_naver_news():
    import requests
    cid = config.NAVER_CLIENT_ID or ''
    sec = config.NAVER_CLIENT_SECRET or ''
    if not cid or not sec:
        return {"status": "warn", "detail": "API 키 미설정 (NAVER_CLIENT_ID / NAVER_CLIENT_SECRET)"}
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": sec}
    r = requests.get(
        "https://openapi.naver.com/v1/search/news.json",
        params={"query": "삼성전자 주가", "display": 1},
        headers=headers, timeout=6,
    )
    r.raise_for_status()
    total = r.json().get('total', 0)
    return {"status": "ok", "detail": f"뉴스 검색 정상 (검색 결과 약 {total:,}건)"}


def _chk_openai():
    import openai
    api_key = config.OPENAI_API_KEY or ''
    if not api_key:
        return {"status": "error", "detail": "API 키 미설정 (OPENAI_API_KEY)"}
    client = openai.OpenAI(api_key=api_key, timeout=10.0)
    resp = client.chat.completions.create(
        model=config.DEFAULT_MODEL,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
    model = resp.model
    return {"status": "ok", "detail": f"{model} 응답 정상"}


def _chk_dart():
    dart_key = config.DART_API_KEY or ''
    if not dart_key:
        return {"status": "warn", "detail": "미설정 (선택 항목 — 공시 수집 비활성)"}
    import requests

    # 1차: 기업 정보 조회
    r1 = requests.get(
        "https://opendart.fss.or.kr/api/company.json",
        params={"crtfc_key": dart_key, "corp_code": "00126380"},
        timeout=6,
    )
    r1_json = r1.json()
    if not (r1.status_code == 200 and r1_json.get('status') == '000'):
        return {"status": "warn", "detail": f"DART company.json 응답 이상 (status={r1.status_code})"}

    # 2차: 재무제표 단일 조회 (fnlttSinglAcnt) — 가치주 스크리너 핵심 소스
    prev_year = str(datetime.now().year - 1)
    r2 = requests.get(
        "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json",
        params={
            "crtfc_key": dart_key,
            "corp_code": "00126380",  # 삼성전자
            "bsns_year": prev_year,
            "reprt_code": "11011",    # 사업보고서
        },
        timeout=5,
    )
    r2_json = r2.json()
    if r2.status_code == 200 and r2_json.get('status') == '000':
        count = len(r2_json.get('list', []))
        return {"status": "ok", "detail": f"DART API 정상 (company.json + fnlttSinglAcnt {prev_year}년 {count}건)"}
    return {"status": "warn", "detail": f"company.json 정상 · fnlttSinglAcnt 응답 이상 (status={r2.status_code})"}


def _chk_sqlite():
    from koreanstocks.core.data.database import db_manager
    missing = []
    parts = []
    with db_manager.get_connection() as conn:
        cur = conn.cursor()
        for table in ("recommendations", "analysis_history"):
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                parts.append(f"{table} {count:,}건")
            except Exception:
                missing.append(table)
        try:
            cur.execute("SELECT COUNT(*) FROM fundamental_cache")
            fcache = cur.fetchone()[0]
            parts.append(f"fundamental_cache {fcache:,}건")
        except Exception:
            pass
    if missing:
        return {"status": "warn", "detail": f"테이블 없음: {', '.join(missing)} (DB 초기화 필요)"}
    return {"status": "ok", "detail": " · ".join(parts)}


def _chk_naver_fundamental():
    """네이버 금융 종목 메인 — PER·PBR·배당수익률 파싱 상태 확인 (가치주 스크리너 1차 소스)."""
    import requests
    from bs4 import BeautifulSoup

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    # KT&G(033780) — PER·PBR·배당수익률 모두 있는 안정적 종목
    r = requests.get(
        "https://finance.naver.com/item/main.naver",
        params={"code": "033780"},
        headers=headers, timeout=10,
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    def _em(sel):
        el = soup.select_one(sel)
        return el.get_text(strip=True) if el else None

    per = _em("em#_per")
    pbr = _em("em#_pbr")

    # 시가배당률: em#_dividend_rate(구) or 테이블 파싱(신)
    div = _em("em#_dividend_rate")
    if div is None:
        strong = soup.find("strong", string="시가배당률(%)")
        if strong:
            tr = strong.find_parent("tr")
            if tr:
                for td in tr.select("td"):
                    txt = td.get_text(strip=True).replace(",", "")
                    try:
                        float(txt)
                        div = txt
                        break
                    except ValueError:
                        pass

    fields = []
    if per:
        fields.append(f"PER {per}")
    if pbr:
        fields.append(f"PBR {pbr}")
    if div:
        fields.append(f"배당수익률 {div}%")

    if not fields:
        return {"status": "warn", "detail": "PER·PBR·배당 모두 파싱 실패 (HTML 구조 변경 가능성)"}
    missing = []
    if not per:
        missing.append("PER")
    if not pbr:
        missing.append("PBR")
    if not div:
        missing.append("배당수익률")
    status = "warn" if missing else "ok"
    detail = "KT&G 기준 " + " · ".join(fields)
    if missing:
        detail += f" (파싱 실패: {', '.join(missing)})"
    return {"status": status, "detail": detail}


def _chk_naver_coinfo():
    """네이버 금융 coinfo (wisereport) — ROE·부채비율·영업이익률 파싱 상태 확인 (가치주 스크리너 2차 소스)."""
    import requests
    from bs4 import BeautifulSoup

    # 삼성전자 — 데이터 안정적으로 제공
    code = "005930"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': f'https://finance.naver.com/item/coinfo.naver?code={code}',
    }
    url = f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}&target=finsum_Y"
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # 테이블에서 ROE·부채비율 행 탐색
    found = []
    for th in soup.find_all("th"):
        txt = th.get_text(strip=True)
        if "ROE" in txt or "부채비율" in txt or "영업이익률" in txt:
            found.append(txt.split("(")[0].strip())

    if not found:
        return {"status": "warn", "detail": "ROE·부채비율·영업이익률 행 파싱 실패 (wisereport 구조 변경 가능성)"}
    return {"status": "ok", "detail": f"wisereport 파싱 정상 — {', '.join(found[:3])} 확인"}


# 소스 메타데이터 정의 (id, name, category, description, used_for, fn) — 11개
_DATA_SOURCES = [
    {
        "id": "fdr_ohlcv",
        "name": "FinanceDataReader — OHLCV",
        "category": "개별 주가 (OHLCV)",
        "description": "종목별 일별 시가·고가·저가·종가·거래량 이력 + 거래일 온라인 실측",
        "source": "Yahoo Finance",
        "used_for": ["기술적 지표 계산 (RSI·MACD·BB 등)", "ML 피처 생성", "백테스팅",
                     "당일 거래일 온라인 실측 (is_trading_day 4단계)"],
        "fn": _chk_fdr_ohlcv,
    },
    {
        "id": "fdr_listing",
        "name": "FinanceDataReader — 종목 목록",
        "category": "전종목 목록",
        "description": "FDR StockListing 복구 여부 모니터링 — 현재 KRX 세션 정책으로 차단, KIND API가 실질 주력 소스",
        "source": "KRX Open API (FinanceDataReader)",
        "used_for": ["복구 시 후보군 유효성 검증", "종목명·섹터 매핑", "테마 필터링"],
        "fn": _chk_fdr_listing,
    },
    {
        "id": "kind_api",
        "name": "KIND API — 전종목 목록",
        "category": "전종목 목록 (대체 소스)",
        "description": "한국거래소 KIND 포털 — 세션 인증 없이 전체 상장 종목 목록 수신 가능",
        "source": "kind.krx.co.kr (KRX 공시 포털)",
        "used_for": ["종목 목록 주력 수집 (FDR KRX 차단 시 자동 대체)",
                     "후보군 구성", "종목명·섹터·업종 매핑"],
        "fn": _chk_kind_api,
    },
    {
        "id": "naver_sise",
        "name": "네이버 금융 시세",
        "category": "전종목 시세 (거래량·등락률)",
        "description": "전종목 거래량·등락률 실질 주력 소스 — FDR KRX 차단 후 자동 대체, BeautifulSoup 병렬 파싱 (~9초)",
        "source": "finance.naver.com/sise/sise_market_sum.naver",
        "used_for": ["버킷 구성 volume/momentum/rebound (get_market_buckets)",
                     "후보군 거래량·등락률 정렬 (get_market_ranking)"],
        "fn": _chk_naver_sise,
    },
    {
        "id": "fdr_index",
        "name": "FinanceDataReader — 시장 지수",
        "category": "시장 지수",
        "description": "KOSPI(^KS11)·KOSDAQ(^KQ11) 지수 및 USD/KRW 환율",
        "source": "Yahoo Finance",
        "used_for": ["대시보드 지수 표시", "GPT 시장 맥락 제공", "ML 상대강도 피처 (rs_vs_mkt)"],
        "fn": _chk_fdr_index,
    },
    {
        "id": "naver_news",
        "name": "Naver News API",
        "category": "뉴스·감성 데이터",
        "description": "종목명 기반 최신 뉴스 기사 수집 (sort=date, display=15)",
        "source": "네이버 검색 API (openapi.naver.com)",
        "used_for": ["뉴스 감성 점수 산출 (sentiment_score)", "GPT 분석 컨텍스트 제공"],
        "fn": _chk_naver_news,
    },
    {
        "id": "openai",
        "name": "OpenAI API",
        "category": "AI / LLM",
        "description": "GPT-4o-mini — 뉴스 감성 분석 및 투자 의견 생성",
        "source": "api.openai.com",
        "used_for": ["뉴스 감성 분석 (-100~100)", "AI 투자 의견 (action·summary·target_price)"],
        "fn": _chk_openai,
    },
    {
        "id": "dart",
        "name": "DART 공시 API",
        "category": "펀더멘털 데이터 (선택)",
        "description": "금융감독원 전자공시 — ROE·부채비율 재무제표 원천 수집 (DART_API_KEY 필요). company.json + fnlttSinglAcnt 이중 검증",
        "source": "opendart.fss.or.kr",
        "used_for": ["ROE·부채비율 직접 계산 (대차대조표)", "영업이익·매출액 YoY 계산",
                     "가치주·우량주 스크리너 3차 소스 (네이버 실패 시 폴백)"],
        "fn": _chk_dart,
    },
    {
        "id": "naver_fundamental",
        "name": "네이버 금융 — 종목 메인",
        "category": "펀더멘털 데이터",
        "description": "PER·PBR·EPS·배당수익률 수집 — 가치주·우량주 스크리너 1차 소스. 시가배당률(%) 테이블 파싱 방식 사용",
        "source": "finance.naver.com/item/main.naver",
        "used_for": ["PER·PBR 필터 (가치주·우량주 스크리너)", "배당수익률 (F-Score·value_score·quality_score)",
                     "펀더멘털 캐시 1차 수집"],
        "fn": _chk_naver_fundamental,
    },
    {
        "id": "naver_coinfo",
        "name": "네이버 금융 — coinfo (wisereport)",
        "category": "펀더멘털 데이터",
        "description": "ROE·부채비율·영업이익률·YoY 다년도 재무 요약 — 가치주·우량주 스크리너 2차 소스",
        "source": "navercomp.wisereport.co.kr (네이버 금융 coinfo iframe)",
        "used_for": ["ROE 필터 (가치주 ≥8%, 우량주 ≥12%)", "영업이익률 필터 (우량주 ≥10%)",
                     "부채비율 필터", "영업이익 YoY 계산"],
        "fn": _chk_naver_coinfo,
    },
    {
        "id": "sqlite",
        "name": "SQLite DB",
        "category": "로컬 저장소",
        "description": "추천 결과·분석 이력·관심 종목·성과 추적·펀더멘털 당일 캐시 저장",
        "source": "data/storage/stock_analysis.db",
        "used_for": ["추천 결과 저장·조회", "분석 이력 관리", "성과 추적 (5·10·20거래일)",
                     "펀더멘털 캐시 (fundamental_cache, 당일 유효)"],
        "fn": _chk_sqlite,
    },
]


@router.get("/macro_context")
def get_macro_context():
    """오늘의 거시경제 레짐 + 감성 점수 + 요약 (일별 캐시, 추가 API 호출 없음)."""
    try:
        from koreanstocks.core.engine.macro_news_agent import macro_news_agent
        ctx = macro_news_agent.get_macro_context()
        return {
            "macro_regime":          ctx.get("macro_regime",          "uncertain"),
            "macro_regime_label":    ctx.get("macro_regime_label",    "불확실"),
            "macro_sentiment_score": ctx.get("macro_sentiment_score", 0),
            "macro_summary":         ctx.get("macro_summary",         ""),
        }
    except Exception as e:
        logger.error(f"거시 컨텍스트 조회 오류: {e}")
        return {
            "macro_regime":          "uncertain",
            "macro_regime_label":    "불확실",
            "macro_sentiment_score": 0,
            "macro_summary":         "",
        }


@router.get("/market")
def get_market(dp=Depends(get_data_provider)):
    """KOSPI / KOSDAQ / USD/KRW 지수 현황"""
    try:
        indices = dp.get_market_indices()
        return {
            "KOSPI":  {"name": "KOSPI",  "close": _safe_num(indices.get("KOSPI")),         "change": _safe_num(indices.get("KOSPI_change"))},
            "KOSDAQ": {"name": "KOSDAQ", "close": _safe_num(indices.get("KOSDAQ")),        "change": _safe_num(indices.get("KOSDAQ_change"))},
            "USDKRW": {"name": "USD/KRW","close": _safe_num(indices.get("USD_KRW")),       "change": None},
        }
    except Exception as e:
        logger.error(f"시장 지수 조회 오류: {e}")
        return {"error": str(e)}


@router.get("/market/trading-day")
def get_trading_day(dp=Depends(get_data_provider)):
    """오늘이 한국 증시 거래일인지 여부 반환"""
    is_trading = dp.is_trading_day()
    return {
        "is_trading_day": is_trading,
        "date": datetime.now().strftime('%Y-%m-%d'),
    }


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


@router.get("/market/data-sources")
def check_data_sources():
    """데이터 수집 라이브러리별 연결 상태 헬스체크 (병렬 실행)"""
    results = []
    # 모든 소스를 병렬로 체크 (전체 타임아웃 15초)
    with ThreadPoolExecutor(max_workers=len(_DATA_SOURCES)) as pool:
        futures = {
            pool.submit(_run_check, src["fn"]): src
            for src in _DATA_SOURCES
        }
        completed = {}
        try:
            for future in as_completed(futures, timeout=15):
                src = futures[future]
                try:
                    check_result = future.result()
                except Exception as e:
                    check_result = {"status": "error", "detail": str(e)[:120], "latency_ms": 0}
                completed[src["id"]] = check_result
        except _FuturesTimeout:
            logger.warning("데이터 소스 헬스체크 전체 타임아웃 (15초 초과) — 미완료 소스는 오류로 처리")

    for src in _DATA_SOURCES:
        chk = completed.get(src["id"], {"status": "error", "detail": "결과 없음", "latency_ms": 0})
        results.append({
            "id":          src["id"],
            "name":        src["name"],
            "category":    src["category"],
            "description": src["description"],
            "source":      src["source"],
            "used_for":    src["used_for"],
            "status":      chk.get("status", "error"),
            "latency_ms":  chk.get("latency_ms", 0),
            "detail":      chk.get("detail", ""),
        })

    ok_count   = sum(1 for r in results if r["status"] == "ok")
    warn_count = sum(1 for r in results if r["status"] == "warn")
    err_count  = sum(1 for r in results if r["status"] == "error")

    return {
        "sources":    results,
        "summary":    {"ok": ok_count, "warn": warn_count, "error": err_count},
        "checked_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
