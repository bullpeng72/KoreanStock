import requests
import logging
from datetime import date
from core.config import config
import os

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """텔레그램 봇을 통한 알림 전송 클래스"""

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)

    def send_message(self, message: str, parse_mode: str = "Markdown"):
        """텍스트 메시지 전송"""
        if not self.enabled:
            logger.warning("Telegram notification is disabled (Token/ChatID missing)")
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": message
        }
        if parse_mode:
            data["parse_mode"] = parse_mode
        
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
            logger.info("Telegram message sent successfully.")
        except Exception as e:
            # Markdown 오류 시 일반 텍스트로 재시도
            if "400" in str(e) and parse_mode == "Markdown":
                logger.warning("Markdown failed, retrying with plain text...")
                self.send_message(message, parse_mode=None)
            else:
                logger.error(f"Failed to send telegram message: {e}")

    def notify_recommendation(self, rec_list: list, session_date: str = None):
        """추천 종목 리스트를 구조화하여 전송"""
        if not rec_list:
            return

        today = session_date or date.today().strftime('%Y-%m-%d')
        lines = [f"📊 <b>오늘의 AI 추천 — {today}</b>\n"]

        for i, rec in enumerate(rec_list, 1):
            ai = rec.get('ai_opinion', {})
            action = ai.get('action', 'HOLD')
            icon = {'BUY': '🟢', 'SELL': '🔴'}.get(action, '🟡')

            current_price = int(rec.get('current_price', 0))
            target_price = int(ai.get('target_price', 0))
            if current_price and target_price:
                upside = round((target_price - current_price) / current_price * 100, 1)
                upside_str = f"+{upside}%" if upside >= 0 else f"{upside}%"
                price_line = f"{current_price:,}원 → {target_price:,}원 ({upside_str})"
            else:
                price_line = f"{target_price:,}원" if target_price else "N/A"

            sentiment_info = rec.get('sentiment_info', {})
            sentiment_label = sentiment_info.get('sentiment_label', 'Neutral')

            lines.append(
                f"{i}. {icon} <b>{rec.get('name', rec.get('code', ''))} ({rec.get('code', '')})</b> — {action}\n"
                f"   점수: Tech {rec.get('tech_score','?')} · ML {rec.get('ml_score','?')} · News {rec.get('sentiment_score','?')} ({sentiment_label})\n"
                f"   목표가: {price_line}\n"
                f"   {ai.get('summary', '')}"
            )

        lines.append("\n💡 대시보드에서 상세 리포트를 확인하세요.")
        self.send_message("\n\n".join(lines), parse_mode="HTML")

notifier = TelegramNotifier()
