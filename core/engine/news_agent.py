import requests
import logging
import json
from datetime import datetime, date, timezone
from typing import List, Dict, Any
import openai
from core.config import config

logger = logging.getLogger(__name__)

class NewsAgent:
    """주식 관련 뉴스 수집 및 감성 분석을 담당하는 에이전트"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.naver_client_id = config.NAVER_CLIENT_ID
        self.naver_client_secret = config.NAVER_CLIENT_SECRET
        self._cache: Dict[str, Any] = {}   # key: "{종목명}_{YYYY-MM-DD}" — 당일 TTL 캐시

    def get_sentiment_score(self, stock_name: str) -> Dict[str, Any]:
        """특정 종목의 최신 뉴스를 분석하여 감성 점수 반환 (당일 결과 캐싱)"""
        cache_key = f"{stock_name}_{date.today().isoformat()}"
        if cache_key in self._cache:
            logger.debug(f"News cache hit: {stock_name}")
            return self._cache[cache_key]

        news_items = self._fetch_news(stock_name)
        if not news_items:
            result = {"sentiment_score": 0, "sentiment_label": "Neutral", "reason": "최근 뉴스 없음", "articles": []}
        else:
            result = self._analyze_sentiment_with_ai(stock_name, news_items)
            result["articles"] = news_items  # 원문 링크·날짜 포함 기사 목록 전달

        self._cache[cache_key] = result
        return result

    def _fetch_news(self, stock_name: str) -> List[Dict[str, str]]:
        """네이버 뉴스 API를 통해 뉴스 제목 + 날짜 수집.

        Returns: [{"title": str, "pubDate": str}, ...]
        """
        if not self.naver_client_id or not self.naver_client_secret:
            logger.warning("Naver API credentials missing.")
            return []

        # " 주가" 추가로 검색 정확도 향상 (동명 계열사 뉴스 혼입 방지)
        query = f"{stock_name} 주가"
        url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display=15&sort=date"
        headers = {
            "X-Naver-Client-Id": self.naver_client_id,
            "X-Naver-Client-Secret": self.naver_client_secret,
        }

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                items = response.json().get('items', [])
                result = []
                for item in items:
                    title = (
                        item.get('title', '')
                        .replace('<b>', '').replace('</b>', '')
                        .replace('&quot;', '"').replace('&amp;', '&')
                    )
                    pub_date = item.get('pubDate', '')
                    result.append({
                        "title":        title,
                        "link":         item.get('link', ''),          # Naver 뉴스 페이지
                        "originallink": item.get('originallink', ''),  # 원문 URL
                        "pubDate":      pub_date,
                        "days_ago":     self._days_ago_label(pub_date),
                    })
                return result
        except Exception as e:
            logger.error(f"News fetching error: {e}")

        return []

    @staticmethod
    def _days_ago_label(pub_date_str: str) -> str:
        """Naver pubDate 문자열('Thu, 19 Dec 2024 10:00:00 +0900')을 '오늘/N일 전' 형태로 변환"""
        try:
            pub_dt = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
            days = (datetime.now(timezone.utc) - pub_dt.astimezone(timezone.utc)).days
            if days == 0:
                return "오늘"
            elif days == 1:
                return "1일 전"
            else:
                return f"{days}일 전"
        except Exception:
            return ""

    def _analyze_sentiment_with_ai(self, stock_name: str, news_items: List[Dict[str, str]]) -> Dict[str, Any]:
        """GPT-4o-mini를 사용하여 뉴스 제목들의 투자 심리 분석.

        - 최신 뉴스에 더 높은 가중치 부여 (pubDate 기반 상대 날짜 포함)
        - temperature=0.1 으로 응답 일관성 확보
        """
        news_lines = []
        for item in news_items:
            age = self._days_ago_label(item.get('pubDate', ''))
            age_str = f" [{age}]" if age else ""
            news_lines.append(f"- {item['title']}{age_str}")
        titles_str = "\n".join(news_lines)

        prompt = f"""
        다음은 주식 종목 '{stock_name}'에 대한 최신 뉴스 제목들입니다 (최신순, 괄호 안은 게재 시점).
        이 뉴스들이 향후 주가에 미칠 영향을 분석하여 감성 점수(-100에서 100 사이)를 매겨줘.
        -100에 가까울수록 매우 부정적(악재), 100에 가까울수록 매우 긍정적(호재)입니다.
        **최신 뉴스(오늘·1일 전)에 더 높은 가중치를 부여**하여 점수를 산출해주세요.

        [뉴스 제목들]
        {titles_str}

        다음 형식의 JSON으로만 응답해줘:
        {{
            "sentiment_score": 점수(숫자),
            "sentiment_label": "Very Bullish/Bullish/Neutral/Bearish/Very Bearish",
            "reason": "점수 산출 근거 (한 문장, 최신 뉴스 위주 요약)",
            "top_news": "가장 영향력이 큰 최신 뉴스 한 줄 요약"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=config.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 금융 뉴스 분석 전문가입니다. 반드시 JSON 형식으로만 답변하세요."},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,   # 일관된 감성 점수 산출
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"sentiment_score": 0, "sentiment_label": "Neutral", "reason": "분석 실패"}

news_agent = NewsAgent()
