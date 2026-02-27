import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, List
import openai
import json
from koreanstocks.core.config import config
from koreanstocks.core.data.provider import data_provider
from koreanstocks.core.engine.indicators import indicators
from koreanstocks.core.data.database import db_manager

logger = logging.getLogger(__name__)

class AnalysisAgent:
    """주식 데이터 분석 및 AI 의견 생성을 담당하는 에이전트"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def analyze_stock(self, code: str, name: str = "") -> Dict[str, Any]:
        """특정 종목에 대한 심층 분석 수행"""
        logger.info(f"Analyzing stock: {code} ({name})")
        
        # 1. 데이터 수집
        df = data_provider.get_ohlcv(code, period='1y')
        if df.empty:
            return {"error": f"No data found for {code}"}

        # 2. 기술적 지표 계산
        df_with_indicators = indicators.calculate_all(df)
        tech_score = indicators.get_composite_score(df_with_indicators)

        # 3. 뉴스 감성 분석 (ML 예측보다 먼저 수행하여 블렌딩에 활용)
        from koreanstocks.core.engine.news_agent import news_agent
        news_res = news_agent.get_sentiment_score(name or code, stock_code=code)
        sentiment_score = news_res.get("sentiment_score", 0)

        # 4. ML 예측 점수 산출 (순수 ML 앙상블; sentiment 블렌딩은 composite 단계에서 일원화)
        from koreanstocks.core.engine.prediction_model import prediction_model
        ml_res = prediction_model.predict(
            code, df,
            df_with_indicators=df_with_indicators,
            fallback_score=tech_score,
        )
        ml_raw_score   = ml_res.get("ensemble_score", tech_score)   # 순수 ML 앙상블 점수
        ml_model_count = ml_res.get("model_count", 0)               # 활성 모델 수 (composite 가중치 분기용)

        # 참고용: ML + 뉴스 감성 블렌딩 점수 (composite 계산에는 미사용, 표시 전용)
        _sentiment_norm_ref = max(0.0, min(100.0, (sentiment_score + 100.0) / 2.0))
        ml_blended = round(0.65 * ml_raw_score + 0.35 * _sentiment_norm_ref, 2)

        # 5. AI 분석 (최근 데이터 + 순수 ML 점수 + 뉴스 점수 기반)
        current_price = float(df_with_indicators.iloc[-1]['close'])
        ai_opinion = self._get_ai_opinion(name or code, df_with_indicators.tail(30), tech_score, ml_raw_score, news_res, current_price)
        
        # 6. 결과 정리
        latest = df_with_indicators.iloc[-1]

        # 시장 구분·섹터 정보 (stock_list 캐시에서 조회)
        stock_list = data_provider.get_stock_list()
        _row = stock_list[stock_list['code'] == code]
        market_val   = str(_row.iloc[0]['market'])   if not _row.empty else ''
        sector_val   = str(_row.iloc[0].get('sector',   '') or '') if not _row.empty else ''
        industry_val = str(_row.iloc[0].get('industry', '') or '') if not _row.empty else ''

        analysis_res = {
            "code": code,
            "name": name,
            "market": market_val,
            "sector": sector_val,
            "industry": industry_val,
            "current_price": float(latest['close']),
            "change_pct": float(latest['change']) * 100 if 'change' in latest else 0.0,
            "tech_score": tech_score,
            "ml_score":       round(ml_raw_score, 2),    # 순수 ML 앙상블 (composite 계산용)
            "ml_blended":     ml_blended,               # ML + 감성 블렌딩 참고값 (표시 전용)
            "ml_model_count": ml_model_count,           # 활성 모델 수 (0이면 fallback)
            "sentiment_score": sentiment_score,
            "sentiment_info": news_res,
            "stats": {
                "high_52w": float(df['high'].max()),
                "low_52w": float(df['low'].min()),
                "avg_vol": int(df['volume'].tail(20).mean()),
                "current_vol": int(latest['volume'])
            },
            "indicators": {
                "rsi": round(float(latest['rsi']), 2),
                "macd": round(float(latest['macd']), 2),
                "macd_sig": round(float(latest['macd_signal']), 2),
                "sma_20": round(float(latest['sma_20']), 0),
                "bb_pos": round(float((latest['close'] - latest['bb_low']) / (latest['bb_high'] - latest['bb_low'])) if (latest['bb_high'] - latest['bb_low']) != 0 else 0.5, 2)
            },
            "ai_opinion": ai_opinion,
            "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M')
        }

        # 7. 분석 이력 저장
        try:
            db_manager.save_analysis_history(analysis_res)
        except Exception as e:
            logger.error(f"Failed to save analysis history: {e}")

        return analysis_res

    def _get_ai_opinion(self, name: str, recent_df: pd.DataFrame, tech_score: float, ml_score: float, news_res: Dict, current_price: float = 0.0) -> Dict[str, Any]:
        """GPT-4o-mini를 사용한 정성적 분석 (ML 및 뉴스 감성 반영)"""
        try:
            # 최근 가격 흐름 요약 (종가, 거래량, 주요 지표 포함)
            indicator_cols = ['close', 'volume', 'rsi', 'macd', 'macd_signal', 'bb_low', 'bb_high']
            available_cols = [c for c in indicator_cols if c in recent_df.columns]
            price_summary = recent_df[available_cols].tail(10).round(2).to_string()

            latest = recent_df.iloc[-1]
            rsi_val = round(float(latest['rsi']), 1) if 'rsi' in latest else 'N/A'
            macd_val = round(float(latest['macd']), 2) if 'macd' in latest else 'N/A'
            macd_sig_val = round(float(latest['macd_signal']), 2) if 'macd_signal' in latest else 'N/A'
            macd_direction = "골든크로스(상승)" if (macd_val != 'N/A' and macd_sig_val != 'N/A' and macd_val > macd_sig_val) else "데드크로스(하락)"
            bb_pos = round(float((latest['close'] - latest['bb_low']) / (latest['bb_high'] - latest['bb_low'])), 2) if ('bb_high' in latest and (latest['bb_high'] - latest['bb_low']) != 0) else 'N/A'

            prompt = f"""
            주식 종목 '{name}'에 대한 데이터와 뉴스 심리를 바탕으로 심층 분석해줘.

            [점수 해석 기준]
            - 기술적 점수: 0~39 약세, 40~59 중립, 60~79 강세, 80~100 매우 강세
            - ML 점수: 향후 5거래일 크로스섹셔널 순위 예측 (0=전체 최하위 상대강도, 50=평균, 100=전체 최상위)
            - 뉴스 감성: -100~-50 매우 부정, -49~-1 부정, 0 중립, 1~50 긍정, 51~100 매우 긍정

            [정량 점수]
            - 기술적 지표 종합 점수: {tech_score}/100
            - 머신러닝 예측 점수: {ml_score}/100
            - 뉴스 감성 점수: {news_res.get('sentiment_score', 0)} (-100~100, 양수면 호재)

            [현재 기술적 지표]
            - 현재가: {int(current_price):,}원
            - RSI(14): {rsi_val} (30 이하: 과매도, 70 이상: 과매수)
            - MACD: {macd_val} / Signal: {macd_sig_val} → {macd_direction}
            - 볼린저 밴드 위치: {bb_pos} (0=하단, 0.5=중간, 1=상단)

            [최근 뉴스 요약]
            - 근거: {news_res.get('reason', '정보 없음')}
            - 주요 이슈: {news_res.get('top_news', '정보 없음')}

            [최근 10일 가격/지표 데이터]
            {price_summary}

            위 정보를 종합하여 다음 형식의 JSON으로만 응답해줘:
            {{
                "summary": "한 줄 요약",
                "strength": "강점 (최대 2개)",
                "weakness": "약점 (최대 2개)",
                "reasoning": "기술적 지표, ML 예측, 뉴스 심리를 모두 반영한 상세 추천 사유",
                "action": "BUY, HOLD, SELL 중 하나 (영문 대문자)",
                "target_price": "4주(20거래일) 목표가 (숫자만, 현재가 기준으로 BUY면 현재가 이상, SELL이면 현재가 이하로 설정)",
                "target_rationale": "목표가 산출의 구체적 근거"
            }}
            """

            response = self.client.chat.completions.create(
                model=config.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 한국 주식 시장 전문 퀀트 애널리스트입니다. 제공된 정량 데이터에 근거하여 객관적이고 일관된 투자 분석을 JSON 형식으로만 제공합니다."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=600,
            )
            result = json.loads(response.choices[0].message.content)

            # 데이터 정제: target_price를 숫자로 변환
            if 'target_price' in result:
                try:
                    price_str = str(result['target_price']).replace(',', '').replace('원', '').strip()
                    result['target_price'] = int(float(price_str))
                except (ValueError, TypeError):
                    result['target_price'] = 0

            # action ↔ target_price 일관성 보정
            if current_price > 0 and result.get('target_price', 0) > 0:
                tp = result['target_price']
                action = result.get('action', 'HOLD')
                if action == 'BUY' and tp < current_price * 0.98:
                    # 매수 의견인데 목표가가 현재가 대비 2% 이상 낮음 → 3% 상향 조정
                    result['target_price'] = int(current_price * 1.03)
                    logger.warning(f"[{name}] BUY but target_price below current. Auto-adjusted to {result['target_price']}")
                elif action == 'HOLD' and tp < current_price * 0.92:
                    # HOLD인데 목표가가 현재가 대비 8% 이상 하락 → SELL로 변경 (기존 3% → 8%)
                    result['action'] = 'SELL'
                    logger.warning(f"[{name}] HOLD but target_price significantly below current. Changed action to SELL.")
                elif action == 'SELL' and tp > current_price * 1.02:
                    # 매도 의견인데 목표가가 현재가 대비 2% 이상 높음 → 3% 하향 조정
                    result['target_price'] = int(current_price * 0.97)
                    logger.warning(f"[{name}] SELL but target_price above current. Auto-adjusted to {result['target_price']}")

            return result
        except Exception as e:
            logger.error(f"AI Analysis Error: {e}")
            return {"summary": "AI 분석 실패", "action": "N/A", "target_price": 0}

analysis_agent = AnalysisAgent()
