"""
펀더멘털 데이터 수집 모듈 (Phase 1)
=====================================
네이버 금융 + DART API를 통해 PER·PBR·ROE·부채비율·YoY 성장률을 수집한다.

수집 전략:
  1차 — 네이버 금융 종목 메인 (`em#_per`, `em#_pbr`, `em#_dividend_rate`)
  2차 — 네이버 금융 기업현황 coinfo (`finsum_Y`: ROE, 부채비율, 매출/영업이익 다년도)
  3차 — DART `fnlttSinglAcnt` (DART_API_KEY 있을 때만, 재무제표 원천 데이터)

캐시: SQLite fundamental_cache (당일 유효)
"""

import json
import logging
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as date_type
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

from koreanstocks.core.config import config
from koreanstocks.core.data.database import db_manager

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
}
_TIMEOUT = 10


def _to_float(text: Optional[str]) -> Optional[float]:
    """숫자 문자열 → float, 실패 시 None."""
    if not text:
        return None
    cleaned = text.strip().replace(",", "").replace("%", "").replace("배", "").replace("원", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


class FundamentalProvider:
    """
    종목 펀더멘털 데이터 수집.

    반환 키:
        per, pbr, eps, dividend_yield          (네이버 메인)
        roe, roe_prev                          (coinfo 최근/전년)
        debt_ratio, debt_ratio_prev            (coinfo)
        op_margin                              (coinfo 최근)
        revenue_yoy, op_income_yoy             (coinfo YoY %)
        op_income_positive                     (bool)
        revenue_cur, op_income_cur             (최근 연도 억원 단위)
        dart_revenue, dart_op_income,          (DART, 억원)
        dart_revenue_prev, dart_op_income_prev
    """

    # ── 공개 API ────────────────────────────────────────────────

    def get_fundamentals(self, code: str) -> Dict:
        """단일 종목 펀더멘털 반환 (당일 캐시 우선)."""
        today = date_type.today().isoformat()
        cached = self._load_cache(code, today)
        if cached:
            # 구 캐시에 dividend_yield 키 자체가 없으면 네이버 메인 재수집 시도
            # (None 값은 "무배당 확인됨"이므로 재수집 불필요 — is None 대신 not in 사용)
            if "dividend_yield" not in cached:
                try:
                    fresh = self._fetch_naver_main(cached.get("code", code))
                    if fresh.get("dividend_yield") is not None:
                        cached["dividend_yield"] = fresh["dividend_yield"]
                        self._save_cache(code, today, cached)
                except Exception:
                    pass
            return cached

        data = self._fetch(code)
        self._save_cache(code, today, data)
        return data

    def get_fundamentals_batch(
        self, codes: List[str], max_workers: int = 15
    ) -> Dict[str, Dict]:
        """여러 종목 병렬 수집. 반환: {code: fundamentals_dict}"""
        results: Dict[str, Dict] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.get_fundamentals, c): c for c in codes}
            for f in as_completed(futures):
                code = futures[f]
                try:
                    results[code] = f.result()
                except Exception as e:
                    logger.warning(f"[FUND] {code} 수집 실패: {e}")
                    results[code] = {"code": code}
        return results

    # ── 내부 수집 ────────────────────────────────────────────────

    def _fetch(self, code: str) -> Dict:
        result: Dict = {"code": code}

        # 1차: 네이버 메인 (PER, PBR, EPS, 배당수익률)
        try:
            result.update(self._fetch_naver_main(code))
        except Exception as e:
            logger.debug(f"[FUND] {code} 네이버 메인 실패: {e}")

        # 2차: 네이버 coinfo (ROE, 부채비율, YoY)
        try:
            result.update(self._fetch_naver_coinfo(code))
        except Exception as e:
            logger.debug(f"[FUND] {code} coinfo 실패: {e}")

        # 3차: DART 재무제표 (DART_API_KEY 있을 때만)
        if config.DART_API_KEY:
            try:
                dart_data = self._fetch_dart_financials(code)
                if dart_data:
                    # DART-only 필드는 항상 추가, 공유 필드는 DART가 실제 값(non-None)을 가진 경우만 덮어씀
                    # → DART가 None을 반환해도 coinfo의 유효값이 소멸되지 않도록 방어
                    for k, v in dart_data.items():
                        if v is not None or k not in result:
                            result[k] = v
            except Exception as e:
                logger.debug(f"[FUND] {code} DART 실패: {e}")

        return result

    def _fetch_naver_main(self, code: str) -> Dict:
        """네이버 금융 종목 메인 — PER·PBR·EPS·배당수익률."""
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        soup = BeautifulSoup(resp.text, "html.parser")

        def em(eid: str) -> Optional[float]:
            el = soup.find("em", id=eid)
            return _to_float(el.get_text()) if el else None

        # 시가배당률: em#_dividend_rate 엘리먼트 제거됨 → 테이블 직접 파싱
        # 네이버 메인 하단 투자지표 테이블의 '시가배당률(%)' 행에서 최근 값을 추출
        dividend_yield: Optional[float] = em("_dividend_rate")  # 구버전 폴백
        if dividend_yield is None:
            strong = soup.find("strong", string="시가배당률(%)")
            if strong:
                tr = strong.find_parent("tr")
                if tr:
                    # 테이블은 오래된 연도→최신 연도 순 — 마지막 non-None이 가장 최근 배당수익률
                    vals = [_to_float(td.get_text()) for td in tr.select("td")]
                    valid = [v for v in vals if v is not None]
                    if valid:
                        dividend_yield = valid[-1]

        return {
            "per":            em("_per"),
            "pbr":            em("_pbr"),
            "eps":            em("_eps"),
            "dividend_yield": dividend_yield,
        }

    def _fetch_naver_coinfo(self, code: str) -> Dict:
        """네이버 금융 기업현황 — 연간 재무 요약 (ROE·부채비율·YoY).

        실제 테이블은 wisereport iframe 안에 있으므로 iframe URL을 직접 요청한다.
        """
        url = (
            f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx"
            f"?cmp_cd={code}&target=finsum_Y"
        )
        headers = {**_HEADERS, "Referer": f"https://finance.naver.com/item/coinfo.naver?code={code}"}
        resp = requests.get(url, headers=headers, timeout=_TIMEOUT)
        soup = BeautifulSoup(resp.text, "html.parser")

        # 연간 재무 테이블 탐색 (IFRS vs GAAP 두 클래스 시도)
        table = (
            soup.select_one("table.tb_type1_ifrs")
            or soup.select_one("table.tb_type1")
        )
        if not table:
            return {}

        # 연도 컬럼 수 계산
        n_years = sum(
            1 for th in table.select("thead th")
            if re.match(r"^\d{4}$", th.get_text(strip=True))  # 정확히 4자리 숫자 — 예측치(E) 컬럼 제외
        )
        if n_years < 1:
            return {}

        # 행 레이블 → td값 리스트
        row_data: Dict[str, List[Optional[float]]] = {}
        for tr in table.select("tbody tr"):
            th = tr.select_one("th")
            if not th:
                continue
            label = th.get_text(strip=True)
            vals = [_to_float(td.get_text()) for td in tr.select("td")]
            # td 개수가 연도 수와 맞지 않으면 스킵
            # 중복 레이블이 있으면 첫 번째 행(지배지분 등) 우선 채택
            if label not in row_data and len(vals) >= n_years:
                row_data[label] = vals[:n_years]

        def get(label_sub: str, idx: int) -> Optional[float]:
            """label_sub과 정확히 일치하거나 'label_sub(...)' 형태인 행의 idx번째 값.
            정확히 일치하는 레이블 우선, 없으면 괄호형 변형(예: '영업이익(손실)')으로 폴백.
            '영업이익률'처럼 label_sub을 포함하지만 다른 지표인 행은 제외된다.
            """
            for lbl, vals in row_data.items():
                if lbl == label_sub and idx < len(vals):
                    return vals[idx]
            for lbl, vals in row_data.items():
                # 괄호형 변형만 허용: "영업이익(손실)" ← OK, "영업이익률" ← NG
                if lbl.startswith(label_sub + "(") and idx < len(vals):
                    return vals[idx]
            return None

        cur  = n_years - 1              # 최근 연도 인덱스
        prev = n_years - 2 if n_years >= 2 else None  # 전년도 (1개년이면 None)

        roe_cur    = get("ROE",      cur)
        roe_prev   = get("ROE",      prev) if prev is not None else None
        debt_cur   = get("부채비율", cur)
        debt_prev  = get("부채비율", prev) if prev is not None else None
        opm_cur    = get("영업이익률", cur)
        # 금융주(은행·증권·보험)는 "영업수익"을 사용하므로 "매출액" 실패 시 폴백
        rev_cur    = get("매출액", cur)
        if rev_cur is None:
            rev_cur = get("영업수익", cur)
        rev_prev   = (get("매출액", prev) if prev is not None else None)
        if prev is not None and rev_prev is None:
            rev_prev = get("영업수익", prev)
        opi_cur    = get("영업이익", cur)
        opi_prev   = get("영업이익", prev) if prev is not None else None

        def yoy(c, p) -> Optional[float]:
            if c is None or p is None or p == 0:
                return None
            return round((c - p) / abs(p) * 100, 1)

        return {
            "roe":               roe_cur,
            "roe_prev":          roe_prev,
            "debt_ratio":        debt_cur,
            "debt_ratio_prev":   debt_prev,
            "op_margin":         opm_cur,
            "revenue_cur":       rev_cur,
            "op_income_cur":     opi_cur,
            "revenue_yoy":       yoy(rev_cur,  rev_prev),   # 1개년이면 None
            "op_income_yoy":     yoy(opi_cur,  opi_prev),   # 1개년이면 None
            "op_income_positive": (opi_cur is not None and opi_cur > 0),
            "roe_improved":      (roe_cur is not None and roe_prev is not None and roe_cur > roe_prev),
            "debt_decreased":    (debt_cur is not None and debt_prev is not None and debt_cur < debt_prev),
        }

    def _fetch_dart_financials(self, code: str) -> Dict:
        """DART 단일회사 주요계정 — 매출액·영업이익 당기/전기."""
        from koreanstocks.core.engine.news_agent import news_agent  # corp_code 매핑 재사용
        corp_code = news_agent._get_dart_corp_code(code)
        if not corp_code:
            return {}

        year = date_type.today().year
        result: Dict = {}

        # 루프 밖에 정의 — 반복마다 재정의 불필요, 순수 함수
        def amt(items_list: list, keys: List[str], field: str = "thstrm_amount") -> Optional[float]:
            for k in keys:
                for item in items_list:
                    if k == item.get("account_nm"):   # .get() — "account_nm" 키 누락 방어
                        raw = item.get(field, "") or ""
                        v = _to_float(raw)   # _to_float 내부에서 이미 쉼표 제거
                        return round(v / 1e8, 1) if v is not None else None
            return None

        def dart_yoy(c, p) -> Optional[float]:
            if c is None or p is None or p == 0:
                return None
            return round((c - p) / abs(p) * 100, 1)

        # 사업보고서(11011) 우선, 없으면 전전년도 사업보고서로 폴백
        # 반기 보고서(11012)는 제외 — frmtrm_amount가 전년 동기(반기)여서
        # dart_yoy()가 상반기 YoY를 연간 YoY처럼 저장하는 오류 방지
        # (3월 이전에는 당해 사업보고서가 미제출인 경우가 많으므로 year-2 폴백 필요)
        for bsns_year, reprt_code in [
            (year - 1, "11011"),   # 전년도 사업보고서 (확정치)
            (year - 2, "11011"),   # 전전년도 사업보고서 (3월 이전 폴백)
        ]:
            try:
                resp = requests.get(
                    "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json",
                    params={
                        "crtfc_key": config.DART_API_KEY,
                        "corp_code": corp_code,
                        "bsns_year": str(bsns_year),
                        "reprt_code": reprt_code,
                    },
                    timeout=_TIMEOUT,
                )
                data = resp.json()
                if data.get("status") != "000" or not data.get("list"):
                    continue

                # 연결(CFS) 우선, 없으면 개별(OFS) 폴백
                cfs_items = [i for i in data["list"] if i.get("fs_div") == "CFS"]
                ofs_items = [i for i in data["list"] if i.get("fs_div") == "OFS"]
                use_items = cfs_items if cfs_items else ofs_items

                rev  = amt(use_items, ["매출액", "영업수익"])
                opi  = amt(use_items, ["영업이익", "영업이익(손실)"])          # 적자 이력 기업 계정명 포함
                net  = amt(use_items, ["당기순이익(손실)", "당기순이익"])
                revp = amt(use_items, ["매출액", "영업수익"], "frmtrm_amount")
                opip = amt(use_items, ["영업이익", "영업이익(손실)"], "frmtrm_amount")
                netp = amt(use_items, ["당기순이익(손실)", "당기순이익"], "frmtrm_amount")
                debt  = amt(use_items, ["부채총계"])
                debtp = amt(use_items, ["부채총계"], "frmtrm_amount")
                eq    = amt(use_items, ["자본총계"])
                eqp   = amt(use_items, ["자본총계"], "frmtrm_amount")

                # 재무상태표 & 손익 데이터가 하나라도 있으면 저장
                has_data = any(v is not None for v in [rev, opi, debt, eq, net])
                if not has_data:
                    continue

                # iter_result: 이번 연도 데이터 전용 dict — 예외 발생 시 연도간 혼합 방지
                # 처리가 완전히 성공한 경우에만 result에 반영한다
                iter_result: Dict = {}

                if rev is not None:
                    iter_result["dart_revenue"]      = rev
                    iter_result["dart_revenue_prev"] = revp
                    iter_result["revenue_yoy"]       = dart_yoy(rev, revp)

                # op_income_yoy·op_income_positive 는 rev와 무관 — opi 유무만 확인
                # 매출액 계정이 없는 지주회사 등에서도 영업이익 지표를 정상 추출하기 위함
                if opi is not None:
                    iter_result["dart_op_income"]      = opi
                    iter_result["dart_op_income_prev"] = opip
                    iter_result["op_income_yoy"]       = dart_yoy(opi, opip)
                    iter_result["op_income_positive"]  = opi > 0

                # 부채비율 (부채총계 / 자본총계 × 100)
                if debt is not None and eq is not None and eq != 0:
                    iter_result["debt_ratio"] = round(debt / eq * 100, 1)
                # 전년 부채비율
                if debtp is not None and eqp is not None and eqp != 0:
                    iter_result["debt_ratio_prev"] = round(debtp / eqp * 100, 1)
                # ROE = 당기순이익 / 자본총계 × 100
                if net is not None and eq is not None and eq != 0:
                    iter_result["roe"] = round(net / eq * 100, 1)
                if netp is not None and eqp is not None and eqp != 0:
                    iter_result["roe_prev"] = round(netp / eqp * 100, 1)
                # 영업이익률 = 영업이익 / 매출액 × 100
                if opi is not None and rev is not None and rev != 0:
                    iter_result["op_margin"] = round(opi / rev * 100, 1)
                # roe_improved / debt_decreased 파생 플래그
                roe_c, roe_p = iter_result.get("roe"), iter_result.get("roe_prev")
                if roe_c is not None and roe_p is not None:
                    iter_result["roe_improved"] = roe_c > roe_p
                dr_c, dr_p = iter_result.get("debt_ratio"), iter_result.get("debt_ratio_prev")
                if dr_c is not None and dr_p is not None:
                    iter_result["debt_decreased"] = dr_c < dr_p

                # has_data가 True여도 계산 가능한 지표가 없으면 (분모 없어 계산 불가 등)
                # year-2 폴백을 계속 시도
                if not iter_result:
                    continue

                result = iter_result
                break

            except Exception as e:
                logger.debug(f"[FUND] DART fnlttSinglAcnt 실패 ({bsns_year}): {e}")

        return result

    # ── 캐시 ────────────────────────────────────────────────────

    def _load_cache(self, code: str, today: str) -> Optional[Dict]:
        try:
            with db_manager.get_connection() as conn:
                row = conn.execute(
                    "SELECT data_json FROM fundamental_cache WHERE code=? AND cache_date=?",
                    (code, today),
                ).fetchone()
                if row:
                    return json.loads(row[0])
        except Exception:
            pass
        return None

    def _save_cache(self, code: str, today: str, data: Dict) -> None:
        try:
            with db_manager.get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO fundamental_cache(code, cache_date, data_json) "
                    "VALUES (?, ?, ?)",
                    (code, today, json.dumps(data, ensure_ascii=False, default=str)),
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"[FUND] 캐시 저장 실패: {e}")


fundamental_provider = FundamentalProvider()
