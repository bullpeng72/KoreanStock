import requests
import logging
from datetime import date
from koreanstocks.core.config import config
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
            # Markdown 오류 시 일반 텍스트로 재시도 (재귀 방지: parse_mode=None이면 재시도 안 함)
            if "400" in str(e) and parse_mode == "Markdown":
                logger.warning("Markdown failed, retrying with plain text...")
                try:
                    plain_data = {"chat_id": self.chat_id, "text": message}
                    requests.post(url, data=plain_data).raise_for_status()
                    logger.info("Telegram message sent successfully (plain text).")
                except Exception as e2:
                    logger.error(f"Failed to send telegram message (plain text fallback): {e2}")
            else:
                logger.error(f"Failed to send telegram message: {e}")

    def notify_recommendation(self, rec_list: list, session_date: str = None):
        """추천 종목 리스트를 구조화하여 전송"""
        if not rec_list:
            return

        today = session_date or date.today().strftime('%Y-%m-%d')
        sep = "─" * 26
        blocks = [f"📊 <b>AI 추천 리포트 — {today}</b>  ({len(rec_list)}종목)\n{sep}"]

        for i, rec in enumerate(rec_list, 1):
            ai     = rec.get('ai_opinion') or {}
            action = ai.get('action', 'HOLD')
            icon   = {'BUY': '🟢', 'SELL': '🔴'}.get(action, '🟡')

            # 종합 점수 + 바
            composite = rec.get('composite_score', 0) or 0
            filled    = min(10, int(composite / 10))
            score_bar = "█" * filled + "░" * (10 - filled)

            # 가격 및 등락률
            current_price = int(rec.get('current_price', 0))
            target_price  = int(ai.get('target_price', 0))
            change_pct    = rec.get('change_pct', 0) or 0
            change_arrow  = "▲" if change_pct >= 0 else "▼"
            change_str    = f"{change_arrow} {abs(change_pct):.1f}%"

            if current_price and target_price:
                upside     = (target_price - current_price) / current_price * 100
                upside_str = f"+{upside:.1f}%" if upside >= 0 else f"{upside:.1f}%"
                price_line = f"{current_price:,}원  →  {target_price:,}원 ({upside_str})"
            else:
                price_line = f"{current_price:,}원" if current_price else "N/A"

            # 점수 및 기술적 지표
            tech     = rec.get('tech_score', '?')
            ml       = round(rec.get('ml_score', 0) or 0)
            sent     = rec.get('sentiment_score', 0) or 0
            sent_str = f"+{sent}" if sent >= 0 else str(sent)
            indic    = rec.get('indicators') or {}
            rsi      = indic.get('rsi')
            rsi_str  = f"{rsi:.0f}" if rsi is not None else "?"

            # 뉴스 및 AI 의견 (GPT가 list로 반환할 수 있으므로 안전하게 변환)
            def _s(val) -> str:
                if isinstance(val, list):
                    return ' · '.join(str(v) for v in val if v)
                return str(val).strip() if val else ''

            sentiment_info = rec.get('sentiment_info') or {}
            top_news = _s(sentiment_info.get('top_news'))
            summary  = _s(ai.get('summary'))
            strength = _s(ai.get('strength'))

            lines = [
                f"{i}. {icon} <b>{rec.get('name', rec.get('code', ''))} ({rec.get('code', '')})</b>",
                f"   <b>{action}</b>  <code>[{score_bar}]</code> {composite:.1f}점  |  당일 {change_str}",
                f"   💰 {price_line}",
                f"   📊 Tech {tech} · ML {ml} · 감성 {sent_str} · RSI {rsi_str}",
            ]
            if top_news:
                lines.append(f"   📰 {top_news}")
            if strength:
                lines.append(f"   ✅ {strength}")
            if summary:
                lines.append(f"   💬 {summary}")

            blocks.append("\n".join(lines))

        blocks.append(f"{sep}\n💡 대시보드에서 상세 리포트를 확인하세요.")
        self.send_message("\n\n".join(blocks), parse_mode="HTML")

    def notify_performance_report(self, stats: dict, recent_outcomes: list):
        """지난 추천 성과 리포트를 텔레그램으로 전송.

        stats           — get_outcome_stats() 결과
        recent_outcomes — get_recent_outcomes() 결과 (최근 14일)
        """
        if not stats or stats.get("total", 0) == 0:
            return

        sep   = "─" * 26
        total = stats["total"]
        lines = [f"📈 <b>AI 추천 성과 리포트</b>\n{sep}",
                 f"📊 <b>최근 90일 통계</b> (총 {total}건)"]

        for n, label in [(5, "5거래일 "), (10, "10거래일"), (20, "20거래일")]:
            ev  = stats.get(f"evaluated_{n}d", 0)
            if ev == 0:
                continue
            wr  = stats.get(f"win_rate_{n}d",  0)
            ret = stats.get(f"avg_return_{n}d", 0)
            bar_filled = min(10, round(wr / 10))
            bar = "█" * bar_filled + "░" * (10 - bar_filled)
            lines.append(
                f"  {label}: <code>[{bar}]</code> 정답 {wr:.0f}%  "
                f"평균 {ret:+.1f}%  ({ev}건)"
            )

        thr = stats.get("target_hit_rate")
        if thr is not None:
            lines.append(f"  목표가 달성률: {thr:.0f}%")

        # 새로 집계 완료된 종목 (5거래일 결과가 있고 최근 2일 이내 업데이트)
        new_5d = [o for o in recent_outcomes
                  if o.get("outcome_5d", {}).get("return_pct") is not None][:5]
        if new_5d:
            lines.append(f"\n{sep}\n<b>새로 집계된 성과</b>")
            for o in new_5d:
                action = o.get("action", "?")
                icon   = {"BUY": "🟢", "SELL": "🔴"}.get(action, "🟡")
                r5     = o["outcome_5d"].get("return_pct")
                c5     = o["outcome_5d"].get("correct")
                hit    = "✅" if c5 == 1 else "❌"
                r5_str = f"{r5:+.1f}%" if r5 is not None else "-"
                lines.append(
                    f"  {hit} {icon} <b>{o['name']}</b>({o['code']}) "
                    f"[{o['session_date']}] {action} → 5d: {r5_str}"
                )

        lines.append(sep)
        self.send_message("\n".join(lines), parse_mode="HTML")


notifier = TelegramNotifier()
