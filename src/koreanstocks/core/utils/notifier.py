import requests
import logging
from collections import defaultdict
from datetime import date
from typing import Dict
from koreanstocks.core.config import config
from koreanstocks.core.constants import BUCKET_DEFAULT
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
            # Markdown/HTML 파싱 오류(400) 시 일반 텍스트로 재시도
            # (재귀 방지: parse_mode=None이면 재시도 안 함)
            if "400" in str(e) and parse_mode in ("Markdown", "HTML"):
                logger.warning(f"{parse_mode} parse failed, retrying with plain text...")
                try:
                    plain_data = {"chat_id": self.chat_id, "text": message}
                    requests.post(url, data=plain_data).raise_for_status()
                    logger.info("Telegram message sent successfully (plain text).")
                except Exception as e2:
                    logger.error(f"Failed to send telegram message (plain text fallback): {e2}")
            else:
                logger.error(f"Failed to send telegram message: {e}")

    # ── 포맷팅 헬퍼 ────────────────────────────────────────────────

    @staticmethod
    def _to_str(val) -> str:
        """list 또는 None을 안전하게 문자열로 변환."""
        if isinstance(val, list):
            return ' · '.join(str(v) for v in val if v)
        return str(val).strip() if val else ''

    @staticmethod
    def _score_bar(score: float, width: int = 10) -> str:
        """0~100 점수를 블록 바(████░░░░░░) 형태로 변환."""
        filled = min(width, round(score / 10))
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def _format_price_line(current_price: int, target_price: int) -> str:
        """현재가·목표가를 '현재가 → 목표가 (+X.X%)' 형태로 포맷."""
        if current_price and target_price:
            upside = (target_price - current_price) / current_price * 100
            upside_str = f"+{upside:.1f}%" if upside >= 0 else f"{upside:.1f}%"
            return f"{current_price:,}원  →  {target_price:,}원 ({upside_str})"
        return f"{current_price:,}원" if current_price else "N/A"

    def _format_stock_block(self, i: int, rec: dict) -> str:
        """추천 종목 1건을 HTML 형식 텍스트 블록으로 변환."""
        ai     = rec.get('ai_opinion') or {}
        action = ai.get('action', 'HOLD')
        icon   = {'BUY': '🟢', 'SELL': '🔴'}.get(action, '🟡')

        composite     = rec.get('composite_score', 0) or 0
        current_price = int(rec.get('current_price', 0))
        target_price  = int(ai.get('target_price', 0))
        change_pct    = rec.get('change_pct', 0) or 0
        change_arrow  = "▲" if change_pct >= 0 else "▼"
        change_str    = f"{change_arrow} {abs(change_pct):.1f}%"

        tech     = rec.get('tech_score', '?')
        ml       = round(rec.get('ml_score', 0) or 0)
        sent     = rec.get('sentiment_score', 0) or 0
        sent_str = f"+{sent}" if sent >= 0 else str(sent)
        indic    = rec.get('indicators') or {}
        rsi      = indic.get('rsi')
        rsi_str  = f"{rsi:.0f}" if rsi is not None else "?"

        sentiment_info = rec.get('sentiment_info') or {}
        top_news = self._to_str(sentiment_info.get('top_news'))
        summary  = self._to_str(ai.get('summary'))
        strength = self._to_str(ai.get('strength'))

        lines = [
            f"{i}. {icon} <b>{rec.get('name', rec.get('code', ''))} ({rec.get('code', '')})</b>",
            f"   <b>{action}</b>  <code>[{self._score_bar(composite)}]</code> {composite:.1f}점  |  당일 {change_str}",
            f"   💰 {self._format_price_line(current_price, target_price)}",
            f"   📊 Tech {tech} · ML {ml} · 감성 {sent_str} · RSI {rsi_str}",
        ]
        if top_news:
            lines.append(f"   📰 {top_news}")
        if strength:
            lines.append(f"   ✅ {strength}")
        if summary:
            lines.append(f"   💬 {summary}")

        return "\n".join(lines)

    @staticmethod
    def _format_period_stat(n: int, label: str, stats: dict) -> str | None:
        """N거래일 성과 통계 한 줄 포맷. 데이터 없으면 None 반환."""
        ev = stats.get(f"evaluated_{n}d", 0)
        if ev == 0:
            return None
        wr  = stats.get(f"win_rate_{n}d",  0)
        ret = stats.get(f"avg_return_{n}d", 0)
        bar = TelegramNotifier._score_bar(wr)
        return f"  {label}: <code>[{bar}]</code> 정답 {wr:.0f}%  평균 {ret:+.1f}%  ({ev}건)"

    @staticmethod
    def _format_outcome_line(o: dict) -> str:
        """개별 추천 성과 결과 한 줄 포맷."""
        action = o.get("action", "?")
        icon   = {"BUY": "🟢", "SELL": "🔴"}.get(action, "🟡")
        r5     = o["outcome_5d"].get("return_pct")
        c5     = o["outcome_5d"].get("correct")
        hit    = "✅" if c5 == 1 else "❌"
        r5_str = f"{r5:+.1f}%" if r5 is not None else "-"
        return (f"  {hit} {icon} <b>{o['name']}</b>({o['code']}) "
                f"[{o['session_date']}] {action} → 5d: {r5_str}")

    # ── 공개 알림 메서드 ────────────────────────────────────────────

    # 버킷별 섹션 헤더 (표시 순서 고정)
    _BUCKET_SECTION: Dict[str, str] = {
        'volume':   '📊 <b>거래량 상위</b> — 유동성 안정주',
        'momentum': '🚀 <b>상승 모멘텀</b> — 단기 상승 추세',
        'rebound':  '🔄 <b>반등 후보</b> — 저점 반등 기회',
    }

    def notify_recommendation(self, rec_list: list, session_date: str = None):
        """추천 종목 리스트를 버킷별로 그룹화하여 전송"""
        if not rec_list:
            return

        today = session_date or date.today().strftime('%Y-%m-%d')
        sep   = "─" * 26
        blocks = [f"📊 <b>AI 추천 리포트 — {today}</b>  ({len(rec_list)}종목)\n{sep}"]

        # 버킷별 그룹화 (bucket 필드 없는 종목은 기본 버킷으로 처리)
        by_bucket: Dict[str, list] = defaultdict(list)
        for rec in rec_list:
            by_bucket[rec.get('bucket', BUCKET_DEFAULT)].append(rec)

        i = 1
        for bucket_name, header in self._BUCKET_SECTION.items():
            recs = by_bucket.get(bucket_name, [])
            if not recs:
                continue
            blocks.append(header)
            for rec in recs:
                blocks.append(self._format_stock_block(i, rec))
                i += 1

        # 알 수 없는 버킷 처리 (방어 코드)
        known = set(self._BUCKET_SECTION.keys())
        for bucket_name, recs in by_bucket.items():
            if bucket_name not in known:
                for rec in recs:
                    blocks.append(self._format_stock_block(i, rec))
                    i += 1

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
            stat_line = self._format_period_stat(n, label, stats)
            if stat_line:
                lines.append(stat_line)

        thr = stats.get("target_hit_rate")
        if thr is not None:
            lines.append(f"  목표가 달성률: {thr:.0f}%")

        # 새로 집계 완료된 종목 (5거래일 결과가 있는 최근 5건)
        new_5d = [o for o in recent_outcomes
                  if o.get("outcome_5d", {}).get("return_pct") is not None][:5]
        if new_5d:
            lines.append(f"\n{sep}\n<b>새로 집계된 성과</b>")
            lines.extend(self._format_outcome_line(o) for o in new_5d)

        lines.append(sep)
        self.send_message("\n".join(lines), parse_mode="HTML")


notifier = TelegramNotifier()
