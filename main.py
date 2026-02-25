import streamlit as st
import logging
from core.config import config

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Korean Stocks AI Analysis",
    page_icon="📈",
    layout="wide",
)

def main():
    st.sidebar.title("📈 AI Stock Analyzer")
    
    menu = ["Dashboard", "My Watchlist", "AI Recommendations", "Backtest Viewer", "Automation & Settings"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    st.sidebar.markdown("---")
    st.sidebar.info("AI와 ML을 결합한 한국 주식 분석 시스템입니다.")
    st.sidebar.caption(f"v{config.VERSION}")

    if choice == "Dashboard":
        show_dashboard()
    elif choice == "My Watchlist":
        show_watchlist()
    elif choice == "AI Recommendations":
        show_recommendations()
    elif choice == "Backtest Viewer":
        show_backtest_viewer()
    elif choice == "Automation & Settings":
        show_automation_settings()

def show_watchlist():
    st.title("⭐️ My Watchlist")
    st.write("관심 종목의 최신 분석 상태를 모니터링합니다.")

    from core.data.database import db_manager
    from core.engine.analysis_agent import analysis_agent

    # 1. 신규 종목 추가 UI
    with st.expander("➕ 관심 종목 추가"):
        new_code = st.text_input("종목 코드 (예: 005930)")
        
        if st.button("Add to Watchlist"):
            if new_code:
                from core.data.provider import data_provider
                with st.spinner("종목 정보 조회 중..."):
                    # 1. 전체 종목 리스트에서 이름 찾기
                    stock_list = data_provider.get_stock_list()
                    matched_stock = stock_list[stock_list['code'] == new_code]
                    
                    final_name = ""
                    if not matched_stock.empty:
                        final_name = matched_stock.iloc[0]['name']
                    else:
                        final_name = f"종목({new_code})"
                    
                    # 2. DB 저장
                    db_manager.add_to_watchlist(new_code, final_name)
                    st.success(f"✅ [{final_name}] 등록 완료!")
                    st.rerun()
            else:
                st.warning("종목 코드를 입력해주세요.")

    # 2. 리스트 조회
    items = db_manager.get_watchlist()
    if not items:
        st.info("등록된 관심 종목이 없습니다.")
        return

    for item in items:
        with st.container(border=True):
            # 모바일 대응: 제목과 삭제 버튼을 상단에 배치
            head_col, del_col = st.columns([5, 1])
            head_col.subheader(f"⭐ {item['name'] or item['code']}")
            if del_col.button("🗑️", key=f"del_{item['code']}"):
                db_manager.remove_from_watchlist(item['code'])
                st.rerun()
            
            st.caption(f"종목 코드: `{item['code']}`")

            # 분석 실행 버튼 (가로를 꽉 채우도록)
            if st.button(f"🔍 실시간 심층 분석 실행", key=f"ana_{item['code']}", width='stretch'):
                with st.spinner("데이터 분석 중..."):
                    res = analysis_agent.analyze_stock(item['code'], item['name'])
                    if "error" not in res:
                        # 모바일에서는 세로로 쌓이는 Metric 구조가 유리
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("Tech", res['tech_score'])
                        m_col2.metric("ML", res['ml_score'])
                        m_col3.metric("News", res['sentiment_score'])
                        
                        st.info(f"💡 **AI 의견:** {res['ai_opinion']['summary']}")
                        
                        # 데이터 보강 섹션
                        with st.expander("📊 상세 통계 및 지표 데이터"):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.write("**가격/거래량 통계**")
                                st.write(f"- 52주 최고: {int(res['stats']['high_52w']):,}원")
                                st.write(f"- 52주 최저: {int(res['stats']['low_52w']):,}원")
                                st.write(f"- 20일 평균 거래량: {res['stats']['avg_vol']:,}")
                                st.write(f"- 현재 거래량 대비: {round(res['stats']['current_vol']/res['stats']['avg_vol']*100, 1)}%")
                            with c2:
                                st.write("**기술적 수치**")
                                st.write(f"- RSI(14): {res['indicators']['rsi']}")
                                st.write(f"- MACD: {res['indicators']['macd']}")
                                st.write(f"- 20일 이평선: {int(res['indicators']['sma_20']):,}원")
                                st.write(f"- BB 위치: {res['indicators']['bb_pos']} (0에 가까울수록 하단)")

                        st.write(f"**상세 사유:** {res['ai_opinion']['reasoning']}")
                        st.success(f"🎯 **목표가(4주):** {int(res['ai_opinion']['target_price']):,}원 ({res['ai_opinion'].get('target_rationale', '')})")

                        # 관련 뉴스 기사 목록
                        news_info = res.get('sentiment_info', {})
                        articles  = news_info.get('articles', [])
                        top_news  = news_info.get('top_news', '')
                        n_display = len(articles) if articles else (1 if top_news else 0)
                        if n_display:
                            with st.expander(f"📰 관련 뉴스 ({n_display}건)"):
                                if news_info.get('reason'):
                                    st.caption(f"💬 AI 종합: {news_info['reason']}")
                                if articles:
                                    for art in articles[:8]:
                                        url   = art.get('originallink') or art.get('link', '')
                                        title = art.get('title', '제목 없음')
                                        age   = art.get('days_ago', '')
                                        badge = f"  `{age}`" if age else ""
                                        if url:
                                            st.markdown(f"- [{title}]({url}){badge}")
                                        else:
                                            st.markdown(f"- {title}{badge}")
                                elif top_news:
                                    st.markdown(f"- {top_news}")
                    else:
                        st.error(res['error'])

            # 분석 히스토리
            with st.expander(f"📜 분석 이력 타임라인"):
                history = db_manager.get_analysis_history(item['code'])
                if history:
                    for h in history:
                        st.write(f"📅 **{h['date']}** | 의견: `{h['action']}`")
                        st.caption(f"Score: [Tech {h['tech_score']} | ML {h['ml_score']} | News {h['sentiment_score']}]")
                        st.write(f"💬 {h['summary']}")
                        st.markdown("---")
                else:
                    st.write("이전 분석 데이터가 없습니다.")

def show_automation_settings():
    st.title("⚙️ Automation & Settings")
    st.write("시스템 자동화 프로세스를 관리하고 테스트합니다.")

    st.subheader("🚀 Manual Automation Trigger")
    st.info("이 버튼을 누르면 '종목 갱신 -> 유망 종목 분석 -> 텔레그램 알림' 프로세스가 순차적으로 실행됩니다.")
    
    if st.button("Run Full Daily Update Process"):
        with st.spinner("자동화 프로세스 실행 중... (데이터 수집 및 분석 포함)"):
            try:
                from core.engine.scheduler import run_daily_update
                run_daily_update()
                st.success("✅ 자동화 프로세스가 성공적으로 완료되었습니다. 텔레그램 메시지를 확인하세요!")
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

    st.markdown("---")
    st.subheader("🔔 Telegram Configuration Status")
    import os
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if token and chat_id:
        st.success("텔레그램 설정이 완료되었습니다.")
    else:
        st.warning("텔레그램 설정(`.env`)이 누락되었습니다. 알림 기능을 사용하려면 설정이 필요합니다.")

def show_backtest_viewer():
    st.title("📈 Strategy Backtest Viewer")

    # ── 백테스트 개념 설명 ────────────────────────────────────────
    st.info(
        "**📚 백테스트(Backtest)란?**\n\n"
        '"이 투자 전략을 과거에 사용했다면 얼마나 벌었을까?" 를 가상으로 시뮬레이션하는 도구입니다.\n\n'
        "실제 돈은 전혀 사용되지 않으며, 과거 주가 데이터를 바탕으로 전략의 성과를 미리 검증합니다.\n\n"
        "⚠️ **주의:** 과거의 좋은 성과가 미래 수익을 보장하지는 않습니다."
    )

    # ── ① 종목 코드 ──────────────────────────────────────────────
    st.subheader("① 종목 선택")
    stock_code = st.text_input(
        "종목 코드 (6자리 숫자)",
        value="005930",
        help="네이버 증권 또는 증권사 앱에서 종목명을 검색하면 6자리 코드를 확인할 수 있습니다.",
    )
    st.caption("예시 코드: 삼성전자 **005930** · SK하이닉스 **000660** · NAVER **035420** · 카카오 **035720** · 현대차 **005380**")

    # ── ② 전략 선택 ──────────────────────────────────────────────
    st.subheader("② 매매 전략 선택")
    strategy_name = st.radio(
        "전략",
        ["RSI", "MACD", "COMPOSITE"],
        horizontal=True,
        label_visibility="collapsed",
        help="세 가지 전략 중 하나를 선택하세요. 아래 설명을 참고하세요.",
    )

    with st.expander("📖 전략별 상세 설명 펼치기 (처음이라면 꼭 읽어보세요)"):
        st.markdown("""
#### 🌡️ RSI 전략 — "싸게 사서 비쌀 때 팔기"

**RSI(상대강도지수)**는 주식이 얼마나 과열되었는지를 0~100 사이 숫자로 나타냅니다.
체온계처럼 생각하면 쉽습니다. 열이 너무 높으면 식혀야 하고, 너무 낮으면 걱정해야 하듯이,
RSI가 너무 높으면 과매수(비싼 상태), 너무 낮으면 과매도(싼 상태)입니다.

| RSI 값 | 상태 | 이 전략의 행동 |
|--------|------|----------------|
| **40 미만** | 너무 많이 팔려 저평가 상태 | 🟢 **매수** (쌀 때 구입) |
| 40 ~ 60 | 중립 구간 | ⏸️ 관망 (아무것도 안 함) |
| **60 초과** | 너무 많이 올라 고평가 상태 | 🔴 **매도** (비쌀 때 처분) |

> 💡 **비유:** 마트에서 평소 1,000원짜리 라면이 700원이 되면 사고, 1,500원이 되면 파는 것과 같습니다.

---

#### 📉 MACD 전략 — "추세의 방향을 따르기"

**MACD**는 단기 평균 가격과 장기 평균 가격의 차이를 이용해 추세 전환 시점을 포착합니다.
두 선이 교차하는 순간을 매매 신호로 삼습니다.

| 신호 | 의미 | 이 전략의 행동 |
|------|------|----------------|
| MACD선이 Signal선을 **위로 교차** (골든크로스) | 상승 추세 시작 | 🟢 **매수** |
| MACD선이 Signal선을 **아래로 교차** (데드크로스) | 하락 추세 시작 | 🔴 **매도** |

> 💡 **비유:** 일주일 평균 기온이 월평균 기온보다 올라가기 시작하면 봄이 오고 있다는 신호와 같습니다.

---

#### 🔐 COMPOSITE 전략 — "두 신호가 모두 동의할 때만 진입"

RSI와 MACD **두 지표가 동시에 매수 신호를 보낼 때만** 진입합니다.
하나라도 위험 신호를 보이면 즉시 빠져나옵니다.

| 조건 | 이 전략의 행동 |
|------|----------------|
| RSI < 50 **그리고** MACD 골든크로스 | 🟢 **매수** (둘 다 긍정적) |
| RSI > 60 **또는** MACD 데드크로스 | 🔴 **매도** (하나라도 부정적) |

> 💡 **비유:** 날씨 앱 두 개가 모두 "맑음"이라고 할 때만 소풍을 가는 신중한 전략입니다.
> 거래 횟수는 줄어들지만 진입 신뢰도가 높아집니다.
        """)

    # ── ③ 기간 및 금액 ───────────────────────────────────────────
    st.subheader("③ 분석 기간 및 투자 금액")
    col_period, col_amount = st.columns(2)

    period_map = {
        "3개월 (단기 검증)": "3m",
        "6개월 (중단기)": "6m",
        "1년 — 권장": "1y",
        "2년 (장기 검증)": "2y",
    }
    with col_period:
        period_label = st.selectbox(
            "분석 기간",
            list(period_map.keys()),
            index=2,
            help="기간이 길수록 전략의 신뢰도가 높아집니다. 최소 1년 이상을 권장합니다.",
        )
        period = period_map[period_label]
        st.caption("📌 기간이 짧으면 운이 좋은 결과처럼 보일 수 있습니다. 여러 기간으로 검증하세요.")

    with col_amount:
        investment_amount = st.number_input(
            "시뮬레이션 투자 금액 (원)",
            value=10_000_000,
            step=1_000_000,
            min_value=100_000,
            help="가상의 투자 금액입니다. 실제 돈이 사용되지 않습니다.",
        )
        st.caption("📌 실제 돈이 사용되지 않는 가상 시뮬레이션입니다.")

    # ── 실행 버튼 ────────────────────────────────────────────────
    if st.button("▶️ 백테스트 실행", type="primary"):
        with st.spinner("과거 데이터로 시뮬레이션 중..."):
            from core.data.provider import data_provider
            from core.engine.indicators import indicators
            from core.engine.strategy import technical_strategy
            from core.utils.backtester import backtester
            import plotly.graph_objects as go

            # 1. 데이터 로드 및 지표 계산
            df = data_provider.get_ohlcv(stock_code, period=period)
            if df.empty:
                st.error("❌ 데이터를 불러올 수 없습니다. 종목 코드를 다시 확인해 주세요.")
                return

            stock_list = data_provider.get_stock_list()
            matched = stock_list[stock_list['code'] == stock_code]
            stock_name = matched.iloc[0]['name'] if not matched.empty else stock_code

            df = indicators.calculate_all(df)

            # 2. 전략 시그널 생성 및 백테스팅
            signals = technical_strategy.generate_signals(df, strategy_type=strategy_name)
            res = backtester.run(df, signals, initial_capital=float(investment_amount))

            if "error" in res:
                st.error(res["error"])
                return

            # 3. 부가 계산
            bnh_return    = round((df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100, 2)
            profit_loss   = res['final_capital'] - investment_amount
            profit_delta  = f"+{profit_loss:,}원" if profit_loss >= 0 else f"{profit_loss:,}원"
            hold_days     = int(res['daily_results']['signal'].sum())
            start_date    = res['daily_results'].index[0].strftime('%Y-%m-%d')
            end_date      = res['daily_results'].index[-1].strftime('%Y-%m-%d')
            total         = res['total_return_pct']
            mdd           = res['mdd_pct']
            wr            = res['win_rate']

            # 4. 헤더 및 종합 판정
            st.divider()
            st.subheader(f"📊 결과: {stock_name} ({stock_code}) · {strategy_name} 전략")
            st.caption(f"분석 기간: {start_date} ~ {end_date}  ({len(df)}거래일 중 {hold_days}일 매수 포지션 유지)")

            if total >= 10:
                st.success(f"✅ 이 기간 **{strategy_name} 전략은 수익**을 냈습니다. (총 수익률 {total}%)")
            elif total >= 0:
                st.info(f"➡️ 이 기간 **{strategy_name} 전략은 소폭 수익 / 본전** 수준이었습니다. (총 수익률 {total}%)")
            else:
                st.warning(f"⚠️ 이 기간 **{strategy_name} 전략은 손실**을 기록했습니다. (총 수익률 {total}%)")

            # 5. 핵심 지표 4개
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "📈 총 수익률",
                f"{total}%",
                delta=f"단순보유 대비 {total - bnh_return:+.1f}%p",
                help=(
                    "전략대로 매매했을 때 얻은 누적 수익률입니다.\n\n"
                    "수수료(0.015%)와 거래세(0.18%)가 반영된 실질 수익률입니다.\n\n"
                    f"같은 기간 그냥 보유만 했을 때는 {bnh_return:+.1f}% 였습니다."
                ),
            )
            m2.metric(
                "📉 최대 낙폭 (MDD)",
                f"{mdd}%",
                help=(
                    "전략 운용 중 가장 고점이었을 때 대비 최대 얼마나 떨어졌는지를 나타냅니다.\n\n"
                    f"예: 1,000만원 투자 시 최악의 순간에 {abs(mdd):.1f}% 하락 = "
                    f"{int(investment_amount * abs(mdd) / 100):,}원 손실 경험\n\n"
                    "이 하락을 감정적으로 버틸 수 있어야 이 전략을 실전에서 쓸 수 있습니다."
                ),
            )
            m3.metric(
                "🎯 승률",
                f"{wr}%",
                help=(
                    "매수 포지션을 보유한 날 중 수익이 난 날의 비율입니다.\n\n"
                    "50% = 수익 날과 손실 날이 반반\n"
                    "60% 이상 = 양호 / 55% 이상 = 보통 / 50% 미만 = 불리"
                ),
            )
            m4.metric(
                "💰 최종 자산",
                f"{res['final_capital']:,}원",
                delta=profit_delta,
                help=(
                    f"초기 투자금 {investment_amount:,}원이 전략 운용 후 "
                    f"{res['final_capital']:,}원이 되었습니다.\n\n"
                    f"수익/손실: {profit_delta}"
                ),
            )

            # 단순 보유 비교 안내
            beat_bnh = total >= bnh_return
            st.markdown(
                f"> 📌 **단순 보유(Buy & Hold) 비교:** 같은 기간 아무것도 안 하고 보유만 했다면 **{bnh_return:+.1f}%** 였습니다.  \n"
                f"> {'이 전략이 단순 보유보다 **유리**했습니다. 🟢' if beat_bnh else '단순 보유가 이 전략보다 **유리**했습니다. 🟡'}"
            )

            # 6. 차트 (전략 vs 단순보유 vs 원금선)
            bnh_curve = df['close'] / df['close'].iloc[0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=res['daily_results'].index, y=res['daily_results']['cum_returns'],
                mode='lines', name=f'{strategy_name} 전략',
                line=dict(color='#00d4aa', width=2),
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=bnh_curve,
                mode='lines', name='단순 보유 (Buy & Hold)',
                line=dict(color='#aaaaaa', width=1.5, dash='dot'),
            ))
            fig.add_hline(
                y=1.0, line_dash="dash", line_color="#ff6b6b",
                annotation_text="원금선 (1.0)", annotation_position="bottom right",
            )
            fig.update_layout(
                title=f"{stock_name} ({stock_code}) · {strategy_name} 전략 vs 단순 보유",
                xaxis_title="날짜",
                yaxis_title="수익 배수  (1.0 = 원금 유지 / 1.1 = +10% / 0.9 = -10%)",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, width='stretch')

            # 7. 결과 해석 가이드
            with st.expander("📖 이 결과를 어떻게 읽어야 할까요? (해석 가이드)"):
                mdd_grade = "안전 ✅" if mdd > -10 else ("주의 🟡" if mdd > -25 else "위험 🔴")
                wr_grade  = "우수 ✅" if wr >= 60 else ("보통 🟡" if wr >= 50 else "낮음 🔴")
                ret_grade = "양호 ✅" if total >= 10 else ("보통 🟡" if total >= 0 else "손실 🔴")
                bnh_grade = "전략 우위 ✅" if beat_bnh else "보유 우위 🟡"

                st.markdown(f"""
#### 지표별 평가 요약

| 지표 | 이번 결과 | 기준 | 평가 |
|------|-----------|------|------|
| 총 수익률 | **{total}%** | 10% 이상이면 양호 | {ret_grade} |
| 최대 낙폭 (MDD) | **{mdd}%** | -10% 이내면 안전 | {mdd_grade} |
| 승률 | **{wr}%** | 60% 이상이면 우수 | {wr_grade} |
| 단순 보유 대비 | **{total - bnh_return:+.1f}%p** | 0 이상이면 전략이 유리 | {bnh_grade} |

---

#### ⚠️ 이것만은 꼭 기억하세요

1. **과거 성과 ≠ 미래 수익**: 이 결과는 과거 데이터 기준이며, 미래를 보장하지 않습니다.
2. **MDD가 핵심**: 총 수익률이 좋아도 MDD가 크면 중간에 공포로 인한 손절 위험이 있습니다.
   예) MDD -30%는 1,000만원 투자 시 최악의 순간에 **300만원 손실을 경험**한다는 의미입니다.
3. **여러 종목·기간에서 검증**: 한 종목에서 잘 된 전략이 다른 종목에서도 통한다는 보장은 없습니다.
4. **단순 보유와 비교 필수**: 전략이 "단순 보유보다 나은가?"를 항상 확인하세요.
   복잡한 전략이 아무것도 안 한 것보다 못한 경우가 많습니다.
5. **슬리피지 미반영**: 실제 매매에서는 원하는 가격에 정확히 체결되지 않을 수 있습니다.
                """)

            # 8. 원본 데이터 테이블
            st.subheader("📋 최근 10거래일 상세 데이터")
            st.caption(
                "signal: **1** = 매수 포지션 보유 중, **0** = 현금 보유 중 (장 밖)  |  "
                "cum_returns: 누적 수익 배수 (**1.0** = 원금 유지, **1.1** = +10%, **0.9** = -10%)"
            )
            st.dataframe(res['daily_results'].tail(10))

# ── 테마 분류 상수 & 헬퍼 ──────────────────────────────────────────────────
_THEME_KEYWORDS_MAP = {
    "AI/인공지능": ["AI", "인공지능", "소프트웨어", "데이터"],
    "로봇/자동화": ["로봇", "자동화", "기계", "장비"],
    "반도체":      ["반도체", "장비", "소재", "부품"],
    "이차전지":    ["배터리", "이차전지", "에너지", "화학"],
    "제약/바이오": ["제약", "바이오", "의료", "생명"],
}

def _get_rec_themes(rec: dict) -> list:
    """sector/industry/name 키워드 매칭으로 테마 분류 반환."""
    text = " ".join([
        rec.get('name', ''),
        rec.get('sector', ''),
        rec.get('industry', ''),
    ])
    matched = [label for label, kws in _THEME_KEYWORDS_MAP.items()
               if any(kw in text for kw in kws)]
    return matched if matched else ["기타"]


def _market_badge(market: str) -> str:
    """KOSPI/KOSDAQ 배지 HTML 반환."""
    color = "#1f6adb" if market == "KOSPI" else "#8b44db" if market == "KOSDAQ" else "#555"
    label = market if market in ("KOSPI", "KOSDAQ") else "─"
    return f"<span style='background:{color};color:white;padding:1px 7px;border-radius:4px;font-size:0.8em;font-weight:bold'>{label}</span>"


def show_dashboard():
    st.title("📊 Market Dashboard")
    
    # 1. 시장 지수 섹션 (모바일 대응: 작은 화면에선 세로로 표시)
    from core.data.provider import data_provider
    indices = data_provider.get_market_indices()
    
    idx_cols = st.columns(3)
    if indices:
        with idx_cols[0]:
            st.metric("KOSPI", f"{indices.get('KOSPI', 0):,.2f}", f"{indices.get('KOSPI_change', 0)*100:.2f}%")
        with idx_cols[1]:
            st.metric("KOSDAQ", f"{indices.get('KOSDAQ', 0):,.2f}", f"{indices.get('KOSDAQ_change', 0)*100:.2f}%")
        with idx_cols[2]:
            st.metric("USD/KRW", f"{indices.get('USD_KRW', 0):,.2f}")
    
    st.markdown("---")

    # 2. 내 관심 종목 요약 (Portfolio Summary)
    from core.data.database import db_manager
    st.subheader("📁 My Portfolio Summary")
    watchlist = db_manager.get_watchlist()
    
    if watchlist:
        # 간단한 요약 지표 (실전 수익률은 백테스팅 결과를 참조하거나 실시간 가격으로 계산)
        total_stocks = len(watchlist)
        col1, col2 = st.columns(2)
        col1.metric("Monitoring Stocks", f"{total_stocks}개")
        col2.info(f"💡 현재 {total_stocks}개의 종목을 밀착 감시 중입니다. Watchlist 메뉴에서 상세 분석을 실행하세요.")
    else:
        st.info("Watchlist에 종목을 추가하여 포트폴리오 관리를 시작하세요.")

    st.markdown("---")

    # 3. 날짜별 AI 분석 리포트
    st.subheader("🎯 AI Analysis Reports")

    from datetime import date as _date

    today_str = _date.today().isoformat()
    saved_dates = db_manager.get_recommendation_dates()

    if not saved_dates:
        st.info("저장된 추천 데이터가 없습니다. 'AI Recommendations' 메뉴에서 분석을 실행하세요.")
    else:
        # 날짜 선택 (기본: 오늘 또는 최근 날짜)
        date_options = sorted(set([today_str] + saved_dates), reverse=True)
        dash_col_date, dash_col_info = st.columns([2, 3])
        selected_dash_date = dash_col_date.selectbox(
            "날짜 선택",
            date_options,
            index=0,
            format_func=lambda d: f"{'★ 오늘  ' if d == today_str else ''}{d}",
            key="dash_date_selector"
        )

        recs = db_manager.get_recommendations_by_date(selected_dash_date)

        if recs:
            freshness = "오늘 분석" if selected_dash_date == today_str else f"{selected_dash_date} 분석"
            dash_col_info.markdown(
                f"<br><span style='color:#aaa'>{freshness} · {len(recs)}종목</span>",
                unsafe_allow_html=True
            )

            # 테마 필터 — rec['theme']에 저장된 분석 설정 기준으로 분류
            _stored_themes = sorted({r.get('theme', '전체') for r in recs})
            _show_dash_filter = len(_stored_themes) > 1 or (_stored_themes and _stored_themes[0] != '전체')
            _dash_theme = '전체'
            if _show_dash_filter:
                _dash_choices = ['전체'] + [t for t in _stored_themes if t != '전체']
                _dash_theme = st.radio(
                    "테마 필터", _dash_choices, horizontal=True, key="dash_theme_filter"
                )

            filtered_recs = recs if _dash_theme == '전체' else [
                r for r in recs if r.get('theme', '전체') == _dash_theme
            ]
            if not filtered_recs:
                st.info(f"'{_dash_theme}' 테마에 해당하는 종목이 없습니다.")
            for r in filtered_recs:
                action = r.get('ai_opinion', {}).get('action', 'HOLD')
                icon = {'BUY': '🟢', 'SELL': '🔴'}.get(action, '🟡')
                score = round(
                    r.get('tech_score', 0) * 0.3
                    + r.get('ml_score', 0) * 0.4
                    + (r.get('sentiment_score', 0) + 100) / 2 * 0.3, 1
                )
                market = r.get('market', '')
                theme_tag = r.get('theme', '')
                label = (
                    f"{icon} {r.get('name', r.get('code',''))} ({r.get('code','')}) "
                    f"{'[' + market + ']' if market else ''}"
                    f"{'  [' + theme_tag + ']' if theme_tag and theme_tag != '전체' else ''}  |  "
                    f"종합점수: {score}  |  목표가(4주): {int(r.get('ai_opinion',{}).get('target_price',0)):,}원"
                )
                with st.expander(label):
                    render_recommendation_card(r, key_prefix=f"dash_{r.get('code','')}_{selected_dash_date}")
        else:
            st.info(f"{selected_dash_date}의 추천 데이터가 없습니다. 'AI Recommendations' 메뉴에서 분석을 실행하세요.")

def render_recommendation_card(rec: dict, key_prefix: str = "rec"):
    """AI 추천 종목 카드를 렌더링하는 공통 함수"""
    ai = rec.get('ai_opinion', {})
    sentiment_info = rec.get('sentiment_info', {})
    action = ai.get('action', 'HOLD')
    action_color = {"BUY": "green", "SELL": "red"}.get(action, "gray")

    # 헤더 행: 현재가 / 점수 요약 / AI 의견 배지
    top_left, top_mid, top_right = st.columns([2, 3, 1])
    with top_left:
        market = rec.get('market', '')
        theme = rec.get('theme', '')
        meta_parts = []
        if market:
            meta_parts.append(_market_badge(market))
        if theme and theme != '전체':
            meta_parts.append(f"<span style='color:#aaa;font-size:0.82em'>{theme}</span>")
        if meta_parts:
            st.markdown(" &nbsp;".join(meta_parts), unsafe_allow_html=True)
        st.metric(
            "현재가",
            f"{int(rec.get('current_price', 0)):,}원",
            f"{rec.get('change_pct', 0):.2f}%"
        )
    with top_mid:
        s1, s2, s3 = st.columns(3)
        s1.metric("Tech", rec.get('tech_score', '-'))
        s2.metric("ML", rec.get('ml_score', '-'))
        s3.metric("News", rec.get('sentiment_score', '-'))
    with top_right:
        st.markdown(f"<br>", unsafe_allow_html=True)
        st.markdown(
            f"<span style='background:{'#1a7a1a' if action=='BUY' else '#7a1a1a' if action=='SELL' else '#555'};"
            f"color:white;padding:6px 14px;border-radius:6px;font-weight:bold;font-size:1.1em'>{action}</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 본문: 좌측 지표·뉴스 / 우측 AI 분석
    left, right = st.columns([1, 2])

    with left:
        st.markdown("**📊 기술적 지표**")
        ind = rec.get('indicators', {})
        if ind:
            st.write(f"- RSI(14): `{ind.get('rsi', 'N/A')}`")
            macd_dir = "▲ 골든크로스" if ind.get('macd', 0) > ind.get('macd_sig', 0) else "▼ 데드크로스"
            st.write(f"- MACD: `{ind.get('macd', 'N/A')}` {macd_dir}")
            st.write(f"- SMA20: `{int(ind.get('sma_20', 0)):,}원`")
            bb_pos = ind.get('bb_pos', 0.5)
            bb_label = "하단권" if bb_pos < 0.3 else "상단권" if bb_pos > 0.7 else "중간권"
            st.write(f"- BB 위치: `{bb_pos}` ({bb_label})")

        stats = rec.get('stats', {})
        if stats:
            st.markdown("**📈 52주 통계**")
            st.write(f"- 최고: `{int(stats.get('high_52w', 0)):,}원`")
            st.write(f"- 최저: `{int(stats.get('low_52w', 0)):,}원`")
            avg_vol = stats.get('avg_vol', 1)
            cur_vol = stats.get('current_vol', 0)
            vol_ratio = round(cur_vol / avg_vol * 100, 1) if avg_vol else 0
            st.write(f"- 거래량: 평균 대비 `{vol_ratio}%`")

        sentiment_score = rec.get('sentiment_score', 0)
        sentiment_color = "green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "gray"
        sentiment_label = sentiment_info.get('sentiment_label', 'Neutral')
        st.markdown("**📰 뉴스 심리**")
        st.markdown(f":{sentiment_color}[{sentiment_score} · {sentiment_label}]")
        if sentiment_info.get('reason'):
            st.caption(sentiment_info['reason'])

    with right:
        st.markdown("**🤖 AI 분석 요약**")
        st.info(ai.get('summary', '분석 내용 없음'))

        if ai.get('strength'):
            st.markdown(f"✅ **강점:** {ai['strength']}")
        if ai.get('weakness'):
            st.markdown(f"⚠️ **약점:** {ai['weakness']}")

        st.markdown("**📝 상세 추천 사유**")
        st.write(ai.get('reasoning', '분석 내용 없음'))

        try:
            target_price = int(ai.get('target_price', 0))
            if target_price > 0:
                current_price = int(rec.get('current_price', 0))
                upside = round((target_price - current_price) / current_price * 100, 1) if current_price else 0
                upside_str = f"(+{upside}%)" if upside >= 0 else f"({upside}%)"
                st.success(f"🎯 **목표가(4주): {target_price:,}원** {upside_str}")
                if ai.get('target_rationale'):
                    st.caption(f"근거: {ai['target_rationale']}")
        except (ValueError, TypeError):
            pass

    # ── 관련 뉴스 기사 목록 (전체 너비) ─────────────────────────────
    articles  = sentiment_info.get('articles', [])
    top_news  = sentiment_info.get('top_news', '')
    n_display = len(articles) if articles else (1 if top_news else 0)
    if n_display:
        with st.expander(f"📰 관련 뉴스 ({n_display}건)", expanded=False):
            reason = sentiment_info.get('reason', '')
            if reason:
                st.caption(f"💬 AI 종합 평가: {reason}")
            if articles:
                for art in articles[:8]:
                    url   = art.get('originallink') or art.get('link', '')
                    title = art.get('title', '제목 없음')
                    age   = art.get('days_ago', '')
                    badge = f"  `{age}`" if age else ""
                    if url:
                        st.markdown(f"- [{title}]({url}){badge}")
                    else:
                        st.markdown(f"- {title}{badge}")
            elif top_news:
                st.markdown(f"- {top_news}")


def _render_rec_fallback(code, action, score, reason, target_price):
    """detail_json이 없는 구버전 추천 데이터를 최소한으로 표시"""
    c1, c2 = st.columns([1, 2])
    c1.write(f"**추천 의견:** `{action}`")
    c1.write(f"**종합 점수:** {score}점")
    try:
        c1.write(f"**목표가(4주):** {int(target_price):,}원")
    except Exception:
        c1.write(f"**목표가(4주):** {target_price}원")
    c2.write(f"**분석 요약:** {reason}")
    st.caption("상세 데이터는 AI Recommendations 메뉴에서 새 추천을 생성하면 확인할 수 있습니다.")


def _show_recommendation_heatmap(db_manager):
    """추천 종목 지속성 히트맵을 렌더링하는 내부 함수"""
    import pandas as pd
    import plotly.graph_objects as go

    days_options = {"7일": 7, "14일": 14, "30일": 30}
    selected_days_label = st.radio(
        "조회 기간", list(days_options.keys()), index=1, horizontal=True
    )
    days = days_options[selected_days_label]

    history = db_manager.get_recommendation_history(days=days)
    if not history:
        st.info("히트맵을 그릴 추천 이력이 없습니다. 추천을 여러 날 실행하면 표시됩니다.")
        return

    df = pd.DataFrame(history)

    # ── 피벗: 행=종목, 열=날짜, 값=score ────────────────────────
    pivot = df.pivot_table(
        values='score', index=['name', 'code'],
        columns='date', aggfunc='first'
    )

    # action(BUY/HOLD/SELL) 피벗 (hover용)
    action_pivot = df.pivot_table(
        values='action', index=['name', 'code'],
        columns='date', aggfunc='first'
    )

    # 날짜 오름차순 정렬
    pivot       = pivot.sort_index(axis=1)
    action_pivot = action_pivot.reindex(columns=pivot.columns)

    # ── 연속 추천 일수 계산 ──────────────────────────────────────
    dates_sorted = list(pivot.columns)

    def _streak(row):
        """가장 최근 날짜부터 역순으로 연속 추천 일수"""
        count = 0
        for d in reversed(dates_sorted):
            if pd.notna(row.get(d)):
                count += 1
            else:
                break
        return count

    streak_map = {idx: _streak(pivot.loc[idx]) for idx in pivot.index}

    # ── Y축 레이블 (종목명 + 연속 일수 배지) ────────────────────
    def _label(idx):
        name, code = idx
        streak = streak_map[idx]
        badge = f"  🔥{streak}일" if streak >= 2 else ""
        return f"{name} ({code}){badge}"

    y_labels = [_label(idx) for idx in pivot.index]

    # ── Hover 텍스트 행렬 ────────────────────────────────────────
    hover = []
    for idx in pivot.index:
        row_hover = []
        for d in dates_sorted:
            score  = pivot.loc[idx, d]
            action = action_pivot.loc[idx, d] if idx in action_pivot.index else None
            if pd.notna(score):
                row_hover.append(
                    f"{idx[0]} ({idx[1]})<br>{d}<br>"
                    f"점수: {round(score, 1)}<br>의견: {action or 'N/A'}"
                )
            else:
                row_hover.append(f"{idx[0]} ({idx[1]})<br>{d}<br>미추천")
        hover.append(row_hover)

    # NaN → None (plotly heatmap용)
    z_values = pivot.values.tolist()
    for r in z_values:
        for i, v in enumerate(r):
            if pd.isna(v):
                r[i] = None

    # ── 차트 렌더링 ──────────────────────────────────────────────
    fig = go.Figure(go.Heatmap(
        z=z_values,
        x=dates_sorted,
        y=y_labels,
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        colorscale=[
            [0.0,  "#2d2d2d"],   # 낮은 점수 (어두운 회색)
            [0.4,  "#4a7c59"],   # 중간
            [0.7,  "#5cb85c"],   # 높음
            [1.0,  "#00e676"],   # 최고
        ],
        zmin=0, zmax=100,
        colorbar=dict(title="점수", tickvals=[0, 25, 50, 75, 100]),
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        title=f"종목 추천 지속성 히트맵 (최근 {days}일)",
        xaxis=dict(title="추천일", tickangle=-30, tickfont=dict(size=11)),
        yaxis=dict(title="", tickfont=dict(size=12), autorange="reversed"),
        height=max(300, len(pivot) * 42 + 100),
        margin=dict(l=20, r=20, t=50, b=40),
        template="plotly_dark",
    )
    st.plotly_chart(fig, width='stretch')

    # ── 연속 추천 상위 종목 요약 ─────────────────────────────────
    streaks = sorted(
        [(streak_map[idx], idx[0], idx[1]) for idx in pivot.index if streak_map[idx] >= 2],
        reverse=True
    )
    if streaks:
        st.markdown("**🔥 연속 추천 종목**")
        cols = st.columns(min(len(streaks), 4))
        for i, (days_cnt, name, code) in enumerate(streaks[:4]):
            cols[i].metric(f"{name} ({code})", f"{days_cnt}일 연속")


def _display_rec_list(recs: list, date_str: str, source_label: str):
    """추천 종목 리스트를 카드로 렌더링하는 내부 헬퍼"""
    st.caption(f"📅 {date_str}  |  {source_label}  |  {len(recs)}종목")

    # 테마 필터 — rec['theme']에 저장된 분석 설정 기준으로 분류
    _stored_themes = sorted({r.get('theme', '전체') for r in recs})
    _show_filter = len(_stored_themes) > 1 or (_stored_themes and _stored_themes[0] != '전체')
    _sel_theme = '전체'
    if _show_filter:
        _theme_choices = ['전체'] + [t for t in _stored_themes if t != '전체']
        _sel_theme = st.radio(
            "테마 필터", _theme_choices, horizontal=True,
            key=f"rec_theme_filter_{date_str}_{source_label}"
        )

    filtered = recs if _sel_theme == '전체' else [
        r for r in recs if r.get('theme', '전체') == _sel_theme
    ]
    if not filtered:
        st.info(f"'{_sel_theme}' 테마에 해당하는 종목이 없습니다.")
        return

    for i, rec in enumerate(filtered):
        action = rec.get('ai_opinion', {}).get('action', 'HOLD')
        icon = {'BUY': '🟢', 'SELL': '🔴'}.get(action, '🟡')
        market = rec.get('market', '')
        theme_tag = rec.get('theme', '')
        label = (
            f"{icon} {rec.get('name', rec.get('code', ''))} ({rec.get('code', '')}) "
            f"{'[' + market + ']' if market else ''}"
            f"{'  [' + theme_tag + ']' if theme_tag and theme_tag != '전체' else ''}  |  "
            f"Tech: {rec.get('tech_score', '-')} · ML: {rec.get('ml_score', '-')} · News: {rec.get('sentiment_score', '-')}"
        )
        with st.expander(label, expanded=(i == 0)):
            render_recommendation_card(rec, key_prefix=f"rec_{rec.get('code',i)}_{date_str}")


def show_recommendations():
    st.title("🤖 AI Recommended Stocks")

    from datetime import date as _date
    from core.data.database import db_manager
    from core.engine.recommendation_agent import recommendation_agent

    today_str = _date.today().isoformat()

    # ── Session State 초기화 ──────────────────────────────────────
    if 'rec_results' not in st.session_state:
        st.session_state['rec_results'] = None   # 현재 세션 분석 결과
    if 'rec_session_date' not in st.session_state:
        st.session_state['rec_session_date'] = None  # 세션 결과의 날짜
    if 'force_reanalyze' not in st.session_state:
        st.session_state['force_reanalyze'] = False  # 강제 재분석 트리거

    # ── 상단 컨트롤 행 ────────────────────────────────────────────
    ctrl_left, ctrl_right = st.columns([3, 1])

    with ctrl_left:
        # 저장된 날짜 목록 로드
        saved_dates = db_manager.get_recommendation_dates()
        date_options = sorted(set([today_str] + saved_dates), reverse=True)
        selected_date = st.selectbox(
            "조회 날짜",
            date_options,
            index=0,
            format_func=lambda d: f"{'★ 오늘  ' if d == today_str else ''}{d}"
        )

    with ctrl_right:
        st.markdown("<br>", unsafe_allow_html=True)
        run_new = st.button("🔄 새로 분석 실행", width='stretch')

    # ── 분석 설정 (접을 수 있게) ─────────────────────────────────
    with st.expander("⚙️ 분석 설정", expanded=False):
        col1, col2 = st.columns(2)
        selected_market = col1.selectbox("시장 선택", ["ALL", "KOSPI", "KOSDAQ"])
        theme_option = col2.selectbox(
            "관심 테마",
            ["전체", "AI/인공지능", "로봇/자동화", "반도체", "이차전지", "제약/바이오"]
        )

    theme_map = {
        "AI/인공지능": ["AI", "인공지능", "소프트웨어", "데이터"],
        "로봇/자동화": ["로봇", "자동화", "기계", "장비"],
        "반도체": ["반도체", "장비", "소재", "부품"],
        "이차전지": ["배터리", "이차전지", "에너지", "화학"],
        "제약/바이오": ["제약", "바이오", "의료", "생명"],
    }
    selected_themes = theme_map.get(theme_option, None)

    st.markdown("---")

    # ── 강제 재분석 플래그 처리 ('지금 재분석 실행' 버튼 클릭 후 rerun) ──
    if st.session_state.get('force_reanalyze'):
        st.session_state['force_reanalyze'] = False
        with st.spinner(f"[{theme_option}] 강제 재분석 중... 잠시만 기다려주세요."):
            recs = recommendation_agent.get_recommendations(
                limit=5, market=selected_market, theme_keywords=selected_themes, theme_label=theme_option
            )
        if recs:
            st.session_state['rec_results'] = recs
            st.session_state['rec_session_date'] = today_str
            st.success(f"✅ 재분석 완료 — {today_str}  |  {len(recs)}종목 저장됨")
            _display_rec_list(recs, today_str, "강제 재분석")
        else:
            st.error("추천 종목을 생성하지 못했습니다.")
        return

    # ── 새로 분석 실행 (DB 우선 조회) ─────────────────────────────
    if run_new:
        existing_today = db_manager.get_recommendations_by_date(today_str)
        if existing_today:
            # 오늘 DB 결과 있음 → 재분석 없이 표시
            st.session_state['rec_results'] = existing_today
            st.session_state['rec_session_date'] = today_str
            st.info(f"💾 오늘({today_str}) 저장된 분석 결과를 불러왔습니다.")
            _display_rec_list(existing_today, today_str, "오늘 저장된 데이터")
            with st.expander("🔁 강제 재분석 (기존 결과 덮어쓰기)", expanded=True):
                st.warning("⚠️ 재분석 시 AI 비결정성으로 추천 종목이 달라질 수 있습니다.")
                def _trigger_force_reanalyze():
                    st.session_state['force_reanalyze'] = True
                st.button("지금 재분석 실행", type="primary", on_click=_trigger_force_reanalyze)
        else:
            # 오늘 DB 결과 없음 → 새로 분석
            with st.spinner(f"[{theme_option}] 테마 분석 중... 잠시만 기다려주세요."):
                recs = recommendation_agent.get_recommendations(
                    limit=5, market=selected_market, theme_keywords=selected_themes, theme_label=theme_option
                )
            if recs:
                st.session_state['rec_results'] = recs
                st.session_state['rec_session_date'] = today_str
                st.success(f"✅ 분석 완료 — {today_str}  |  {len(recs)}종목 저장됨")
                _display_rec_list(recs, today_str, "방금 분석")
            else:
                st.error("추천 종목을 생성하지 못했습니다.")
        return  # 아래 조회 로직과 중복 렌더링 방지

    # ── 날짜별 결과 표시 ──────────────────────────────────────────
    # 오늘 날짜 + 세션 데이터 있으면 세션 캐시 우선 사용
    if selected_date == today_str and st.session_state.get('rec_results'):
        _display_rec_list(
            st.session_state['rec_results'], today_str, "세션 캐시 (재분석 불필요)"
        )
    else:
        # DB에서 해당 날짜 결과 조회
        recs = db_manager.get_recommendations_by_date(selected_date)
        if recs:
            source = "오늘 저장된 데이터" if selected_date == today_str else "저장된 히스토리"
            _display_rec_list(recs, selected_date, source)
            # 오늘 데이터를 DB에서 로드한 경우 세션에도 캐시
            if selected_date == today_str:
                st.session_state['rec_results'] = recs
                st.session_state['rec_session_date'] = today_str
        else:
            if selected_date == today_str:
                st.info("오늘 분석 결과가 없습니다. '🔄 새로 분석 실행' 버튼을 눌러 시작하세요.")
            else:
                st.info(f"{selected_date}의 저장된 추천 데이터가 없습니다.")

    # ── 추천 지속성 히트맵 ────────────────────────────────────────
    st.markdown("---")
    with st.expander("📅 추천 지속성 분석 (히트맵)", expanded=False):
        _show_recommendation_heatmap(db_manager)

if __name__ == "__main__":
    main()
