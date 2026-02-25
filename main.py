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
    st.write("선택한 종목과 전략의 과거 성과를 분석합니다.")

    col1, col2, col3 = st.columns(3)
    stock_code = col1.text_input("종목 코드 (6자리)", value="005930")
    strategy_name = col2.selectbox("전략 선택", ["RSI", "MACD", "COMPOSITE"], 
                                 help="분석에 사용할 기술적 지표 전략을 선택하세요.")
    period = col3.selectbox("분석 기간", ["1y", "2y", "3m", "6m"])
    
    with st.expander("ℹ️ 전략별 상세 설명"):
        st.markdown("""
        - **RSI (Relative Strength Index)**: 과매수/과매도 지표를 활용합니다. 
            - *매수*: RSI가 40 미만으로 떨어질 때 (저평가 구간 진입)
            - *매도*: RSI가 60 이상으로 올라갈 때 (고평가 구간 진입)
        - **MACD (Moving Average Convergence Divergence)**: 추세의 전환을 포착합니다.
            - *매수*: MACD 선이 Signal 선을 상향 돌파(Golden Cross)할 때
            - *매도*: MACD 선이 Signal 선을 하향 돌파(Dead Cross)할 때
        - **COMPOSITE (복합 전략)**: 여러 지표를 조합하여 신중하게 진입합니다.
            - *매수*: RSI가 50 미만이면서 동시에 MACD가 골든크로스일 때
            - *매도*: RSI가 60 이상이거나 MACD가 데드크로스일 때
        """)
    
    investment_amount = st.number_input("초기 투자 금액 (원)", value=10000000, step=1000000)

    if st.button("Run Backtest"):
        with st.spinner("과거 데이터 시뮬레이션 중..."):
            from core.data.provider import data_provider
            from core.engine.indicators import indicators
            from core.engine.strategy import technical_strategy
            from core.utils.backtester import backtester
            import plotly.graph_objects as go

            # 1. 데이터 로드 및 지표 계산
            df = data_provider.get_ohlcv(stock_code, period=period)
            if df.empty:
                st.error("데이터를 불러올 수 없습니다.")
                return

            stock_list = data_provider.get_stock_list()
            matched = stock_list[stock_list['code'] == stock_code]
            stock_name = matched.iloc[0]['name'] if not matched.empty else stock_code

            df = indicators.calculate_all(df)
            
            # 2. 전략 시그널 생성
            signals = technical_strategy.generate_signals(df, strategy_type=strategy_name)
            
            # 3. 백테스팅 실행
            res = backtester.run(df, signals, initial_capital=float(investment_amount))
            
            if "error" in res:
                st.error(res["error"])
                return

            # 4. 결과 표시 (Metric)
            st.subheader(f"{stock_name} ({stock_code})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("총 수익률", f"{res['total_return_pct']}%", 
                      help="투자 기간 동안의 누적 복리 수익률입니다. 수수료와 세금이 포함된 실전 수익률입니다.")
            m2.metric("최대 낙폭 (MDD)", f"{res['mdd_pct']}%", 
                      help="전략 실행 중 최고점 대비 발생한 최대 하락폭입니다. 리스크 측정 지표로 활용됩니다.")
            m3.metric("승률", f"{res['win_rate']}%", 
                      help="포지션을 보유한 거래일 중 수익이 발생한 날의 비율입니다.")
            m4.metric("최종 자산", f"{res['final_capital']:,}원", 
                      help="초기 투자 금액이 전략에 따라 운용된 후의 최종 평가 금액입니다.")

            start_date = res['daily_results'].index[0].strftime('%Y-%m-%d')
            end_date = res['daily_results'].index[-1].strftime('%Y-%m-%d')
            st.write(f"📊 **분석 요약:** `{start_date}` ~ `{end_date}` ({len(df)}거래일)")
            st.write(f"💡 **포지션 보유:** 총 {int(res['daily_results']['signal'].sum())}일 동안 매수 상태 유지")

            # 5. 차트 시각화
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res['daily_results'].index, y=res['daily_results']['cum_returns'],
                                     mode='lines', name='Strategy Cumulative Returns'))
            fig.update_layout(title=f"{stock_name} ({stock_code}) · {strategy_name} Strategy Performance",
                              xaxis_title="Date", yaxis_title="Cumulative Return (Base 1.0)",
                              template="plotly_dark")
            st.plotly_chart(fig, width='stretch')
            
            st.subheader("📋 상세 분석 데이터 (최근 10일)")
            st.dataframe(res['daily_results'].tail(10))

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
            for r in recs:
                action = r.get('ai_opinion', {}).get('action', 'HOLD')
                icon = {'BUY': '🟢', 'SELL': '🔴'}.get(action, '🟡')
                score = round(
                    r.get('tech_score', 0) * 0.3
                    + r.get('ml_score', 0) * 0.4
                    + (r.get('sentiment_score', 0) + 100) / 2 * 0.3, 1
                )
                label = (
                    f"{icon} {r.get('name', r.get('code',''))} ({r.get('code','')})  |  "
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
    st.plotly_chart(fig, use_container_width=True)

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
    for i, rec in enumerate(recs):
        action = rec.get('ai_opinion', {}).get('action', 'HOLD')
        icon = {'BUY': '🟢', 'SELL': '🔴'}.get(action, '🟡')
        label = (
            f"{icon} {rec.get('name', rec.get('code', ''))} ({rec.get('code', '')})  |  "
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
        run_new = st.button("🔄 새로 분석 실행", use_container_width=True)

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

    # ── 새로 분석 실행 ────────────────────────────────────────────
    if run_new:
        with st.spinner(f"[{theme_option}] 테마 분석 중... 잠시만 기다려주세요."):
            recs = recommendation_agent.get_recommendations(
                limit=5, market=selected_market, theme_keywords=selected_themes
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
