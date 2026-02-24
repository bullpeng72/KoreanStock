import requests
import logging
import json
from datetime import datetime
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

    def get_sentiment_score(self, stock_name: str) -> Dict[str, Any]:
        """특정 종목의 최신 뉴스를 분석하여 감성 점수 반환"""
        news_titles = self._fetch_news_titles(stock_name)
        if not news_titles:
            return {"sentiment_score": 0, "sentiment_label": "Neutral", "reason": "최근 뉴스 없음"}

        # GPT를 통한 뉴스 감성 분석
        return self._analyze_sentiment_with_ai(stock_name, news_titles)

    def _fetch_news_titles(self, stock_name: str) -> List[str]:
        """네이버 뉴스 API를 통해 뉴스 제목 수집"""
        if not self.naver_client_id or not self.naver_client_secret:
            logger.warning("Naver API credentials missing.")
            return []
            
        url = f"https://openapi.naver.com/v1/search/news.json?query={stock_name}&display=10&sort=sim"
        headers = {
            "X-Naver-Client-Id": self.naver_client_id,
            "X-Naver-Client-Secret": self.naver_client_secret
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                items = response.json().get('items', [])
                return [item['title'].replace('<b>', '').replace('</b>', '').replace('&quot;', '"') for item in items]
        except Exception as e:
            logger.error(f"News fetching error: {e}")
        
        return []

    def _analyze_sentiment_with_ai(self, stock_name: str, titles: List[str]) -> Dict[str, Any]:
        """GPT-4o-mini를 사용하여 뉴스 제목들의 투자 심리 분석"""
        titles_str = "\n- ".join(titles)
        prompt = f"""
        다음은 주식 종목 '{stock_name}'에 대한 최신 뉴스 제목들입니다.
        이 뉴스들이 향후 주가에 미칠 영향을 분석하여 감성 점수(-100에서 100 사이)를 매겨줘.
        -100에 가까울수록 매우 부정적(악재), 100에 가까울수록 매우 긍정적(호재)입니다.

        [뉴스 제목들]
        - {titles_str}

        다음 형식의 JSON으로만 응답해줘:
        {{
            "sentiment_score": 점수(숫자),
            "sentiment_label": "Very Bullish/Bullish/Neutral/Bearish/Very Bearish",
            "reason": "점수 산출 근거 (한 문장)",
            "top_news": "가장 영향력이 큰 뉴스 한 줄 요약"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=config.DEFAULT_MODEL,
                messages=[{"role": "system", "content": "당신은 금융 뉴스 분석 전문가입니다. 반드시 JSON 형식으로만 답변하세요."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"sentiment_score": 0, "sentiment_label": "Neutral", "reason": "분석 실패"}

news_agent = NewsAgent()
