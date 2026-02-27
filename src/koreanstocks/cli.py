"""Typer CLI — koreanstocks serve / recommend / analyze / train / init"""
import typer
from typing import Optional

app = typer.Typer(
    name="koreanstocks",
    help="KOSPI·KOSDAQ 종목 자동 스크리닝 + 텔레그램 리포트 플랫폼",
    add_completion=False,
    rich_markup_mode="rich",
    invoke_without_command=True,   # 서브커맨드 없이 실행 가능
    no_args_is_help=False,         # 직접 처리
)


def _version_callback(value: bool):
    if value:
        try:
            import importlib.metadata
            ver = importlib.metadata.version("koreanstocks")
        except Exception:
            from koreanstocks import VERSION
            ver = VERSION
        typer.echo(f"koreanstocks {ver}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None, "--version", "-V",
        help="버전 정보 표시 후 종료",
        callback=_version_callback,
        is_eager=True,
    ),
):
    """
    [bold cyan]KoreanStocks[/bold cyan] — AI/ML 기반 한국 주식 분석 플랫폼

    [dim]KOSPI·KOSDAQ 종목을 기술적 지표, ML 앙상블, 뉴스 감성 분석으로
    자동 스크리닝하고 텔레그램 리포트를 발송합니다.[/dim]

    [bold]빠른 시작:[/bold]

    [green]  koreanstocks init[/green]       [dim]# .env 설정 파일 생성[/dim]
    [green]  koreanstocks serve[/green]      [dim]# 웹 대시보드 실행 (브라우저 자동 열림)[/dim]
    [green]  koreanstocks recommend[/green]  [dim]# 오늘의 추천 종목 분석[/dim]
    [green]  koreanstocks analyze 005930[/green]  [dim]# 삼성전자 단일 분석[/dim]

    서브커맨드 상세 도움말: [cyan]koreanstocks [커맨드] --help[/cyan]
    """
    # 서브커맨드 없이 실행됐을 때만 도움말 출력
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="바인딩 호스트 주소"),
    port: int = typer.Option(8000, help="포트 번호"),
    reload: bool = typer.Option(False, help="코드 변경 시 자동 재시작 [dim](개발용)[/dim]"),
    no_browser: bool = typer.Option(False, "--no-browser", help="브라우저 자동 실행 비활성화"),
):
    """
    [bold]FastAPI 서버 실행[/bold] — 웹 대시보드 및 Reveal.js 브리핑

    서버 시작 후 브라우저가 자동으로 열립니다.

    [bold]제공 URL:[/bold]
    [green]  /[/green]          일일 브리핑 슬라이드 (Reveal.js)
    [green]  /dashboard[/green] 인터랙티브 대시보드 (5탭)
    [green]  /docs[/green]      API 문서 (Swagger UI)

    [bold]예시:[/bold]
    [dim]  koreanstocks serve[/dim]
    [dim]  koreanstocks serve --port 8080[/dim]
    [dim]  koreanstocks serve --reload --no-browser[/dim]
    """
    import threading
    import time
    import webbrowser
    import uvicorn

    url = f"http://{host}:{port}/dashboard"
    typer.echo(f"서버 시작: {url}")

    if not no_browser:
        def _open_browser():
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(
        "koreanstocks.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def recommend(
    market: str = typer.Option("ALL", help="시장 필터: [cyan]ALL[/cyan] | KOSPI | KOSDAQ"),
    limit: int = typer.Option(5, help="추천 종목 수 (1~30)"),
):
    """
    [bold]추천 종목 분석 실행[/bold] — GitHub Actions 자동화 및 수동 실행용

    후보 종목 100개를 병렬 분석하여 상위 N개를 선정합니다.
    결과는 DB에 저장되고 텔레그램으로 알림이 전송됩니다.

    [bold]분석 파이프라인:[/bold]
    [dim]  후보 100종목 → 병렬 분석(최대 30) → 종합점수 산출 → DB저장 → 텔레그램 발송[/dim]

    [bold]예시:[/bold]
    [dim]  koreanstocks recommend[/dim]
    [dim]  koreanstocks recommend --market KOSPI --limit 10[/dim]
    [dim]  koreanstocks recommend --market KOSDAQ[/dim]
    """
    from koreanstocks.core.engine.scheduler import run_daily_update
    typer.echo(f"일일 업데이트 실행 (market={market}, limit={limit})...")
    run_daily_update()
    typer.echo("완료.")


@app.command()
def analyze(
    code: str = typer.Argument(..., help="종목 코드 6자리 (예: 005930)"),
):
    """
    [bold]단일 종목 심층 분석[/bold] — 기술지표·ML·뉴스 감성 통합 분석

    [bold]출력 항목:[/bold]
      기술점수 (Tech)  — RSI, MACD, BB 등 지표 기반 (0~100)
      ML점수           — 크로스섹셔널 퍼센타일 순위 예측 (0~100)
      감성점수 (News)  — 뉴스 GPT 감성 분석 (-100~+100)
      AI 의견          — BUY / HOLD / SELL + 한줄 요약 + 목표가

    [bold]예시:[/bold]
    [dim]  koreanstocks analyze 005930[/dim]   [dim]# 삼성전자[/dim]
    [dim]  koreanstocks analyze 000660[/dim]   [dim]# SK하이닉스[/dim]
    [dim]  koreanstocks analyze 035420[/dim]   [dim]# NAVER[/dim]
    """
    from koreanstocks.core.engine.analysis_agent import analysis_agent
    from koreanstocks.core.data.provider import data_provider

    stock_list = data_provider.get_stock_list()
    row = stock_list[stock_list["code"] == code]
    name = row.iloc[0]["name"] if not row.empty else code

    typer.echo(f"[{code}] {name} 분석 중...")
    result = analysis_agent.analyze_stock(code, name)

    if result:
        opinion = result.get("ai_opinion", {})
        action  = opinion.get("action", "-")
        typer.echo(f"  기술점수 : {result.get('tech_score', '-')}")
        typer.echo(f"  ML점수   : {result.get('ml_score', '-')}")
        typer.echo(f"  감성점수 : {result.get('sentiment_score', '-')}")
        typer.echo(f"  의견     : {action} — {opinion.get('summary', '')}")
        if opinion.get("target_price"):
            typer.echo(f"  목표가   : {int(opinion['target_price']):,}원")
    else:
        typer.echo("분석 실패.", err=True)
        raise typer.Exit(1)


@app.command()
def train(
    period: str = typer.Option("2y", help="학습 데이터 기간: 1y | [cyan]2y[/cyan]"),
    future_days: int = typer.Option(5, help="예측 대상 거래일 수"),
    test_ratio: float = typer.Option(0.2, help="검증 세트 비율 (0~1)"),
):
    """
    [bold]ML 모델 재학습[/bold] — RandomForest·GradientBoosting·XGBoost 앙상블

    [bold]사용 피처 (31개):[/bold]
    [dim]  기술지표 22개 — RSI, MACD, BB, Stochastic, CCI, ATR 등[/dim]
    [dim]  PyKrx 9개    — PBR, PER, DIV + 외국인/기관 5일 순매수 비율[/dim]

    [bold]타깃:[/bold] 5거래일 후 크로스섹셔널 퍼센타일 순위 (0~100)

    [bold]예시:[/bold]
    [dim]  koreanstocks train[/dim]
    [dim]  koreanstocks train --period 1y --future-days 10[/dim]
    [dim]  koreanstocks train --test-ratio 0.3[/dim]
    """
    import subprocess
    import sys
    from pathlib import Path

    # editable install: src/koreanstocks/cli.py → 3단계 상위 = 저장소 루트
    script = Path(__file__).parent.parent.parent / "train_models.py"
    if not script.exists():
        # 전역 설치(non-editable) 또는 경로 불일치 시 현재 디렉토리에서 탐색
        script = Path.cwd() / "train_models.py"
    if not script.exists():
        typer.echo(
            "train_models.py를 찾을 수 없습니다. "
            "프로젝트 루트 디렉토리에서 실행하거나 "
            ".env에 KOREANSTOCKS_BASE_DIR를 설정하세요.",
            err=True,
        )
        raise typer.Exit(1)

    cmd = [
        sys.executable, str(script),
        f"--period={period}",
        f"--future-days={future_days}",
        f"--test-ratio={test_ratio}",
    ]
    typer.echo(f"실행: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.command()
def init():
    """
    [bold]초기 설정[/bold] — .env 환경변수 템플릿 생성

    현재 디렉토리에 [cyan].env[/cyan] 파일을 생성합니다.
    이미 존재하는 경우 덮어쓰지 않습니다.

    [bold]필수 항목:[/bold]
    [dim]  OPENAI_API_KEY         — GPT-4o-mini (뉴스 감성 분석·AI 의견 생성)[/dim]
    [dim]                           발급: https://platform.openai.com/api-keys[/dim]
    [dim]  NAVER_CLIENT_ID/SECRET — 네이버 뉴스 검색 API (종목명 기반 최신 뉴스)[/dim]
    [dim]                           발급: https://developers.naver.com/apps[/dim]
    [dim]  TELEGRAM_BOT_TOKEN     — 텔레그램 봇 토큰 (추천 리포트 발송)[/dim]
    [dim]                           발급: @BotFather → /newbot[/dim]
    [dim]  TELEGRAM_CHAT_ID       — 텔레그램 수신 채팅방 ID[/dim]
    [dim]                           확인: api.telegram.org/bot{TOKEN}/getUpdates[/dim]

    [bold]선택 항목:[/bold]
    [dim]  DART_API_KEY           — 금융감독원 공시 수집 (선택, 없으면 뉴스만 분석)[/dim]
    [dim]                           발급: opendart.fss.or.kr (무료, 즉시 발급)[/dim]
    [dim]                           미설정 시 뉴스만으로 감성 분석 진행[/dim]

    [bold]예시:[/bold]
    [dim]  koreanstocks init     # .env 생성 후 편집기로 열어 키 입력[/dim]
    """
    from pathlib import Path

    env_file = Path(".env")
    if env_file.exists():
        typer.echo(".env 파일이 이미 존재합니다. 내용을 직접 편집하세요.")
        return

    template = (
        "# KoreanStocks 환경변수\n"
        "# koreanstocks init 으로 생성됨\n"
        "# 각 항목에 API 키를 입력한 뒤 저장하세요.\n"
        "\n"
        "# ── 필수 ────────────────────────────────────────────────────\n"
        "\n"
        "# OpenAI API 키 — GPT-4o-mini (뉴스 감성 분석, AI 의견 생성)\n"
        "# 발급: https://platform.openai.com/api-keys\n"
        "OPENAI_API_KEY=\n"
        "\n"
        "# 네이버 검색 API — 종목명 기반 최신 뉴스 수집\n"
        "# 발급: https://developers.naver.com/apps\n"
        "NAVER_CLIENT_ID=\n"
        "NAVER_CLIENT_SECRET=\n"
        "\n"
        "# 텔레그램 봇 — 일일 추천 리포트 발송\n"
        "# BOT_TOKEN: @BotFather 에서 /newbot 으로 발급\n"
        "# CHAT_ID:   봇에게 메시지를 보낸 뒤 https://api.telegram.org/bot<TOKEN>/getUpdates 로 확인\n"
        "TELEGRAM_BOT_TOKEN=\n"
        "TELEGRAM_CHAT_ID=\n"
        "\n"
        "# ── 선택 ────────────────────────────────────────────────────\n"
        "\n"
        "# DART Open API 키 — 금융감독원 공시 수집 (설정 시 감성 분석 품질 향상)\n"
        "# 미설정이어도 동작하며, 뉴스만으로 감성 분석을 진행합니다.\n"
        "# 발급: https://opendart.fss.or.kr → 개발자 센터 → API 신청 (무료, 즉시 발급)\n"
        "DART_API_KEY=\n"
        "\n"
        "# ── 시스템 ───────────────────────────────────────────────────\n"
        "\n"
        "# SQLite DB 경로 (기본값 그대로 사용 권장)\n"
        "DB_PATH=data/storage/stock_analysis.db\n"
        "\n"
        "# 프로젝트 루트 경로 (pip install -e . 로 editable 설치 시 자동 탐지됨)\n"
        "# 전역 설치(pip install koreanstocks) 사용 시 아래 주석을 해제하고 경로를 입력\n"
        "# KOREANSTOCKS_BASE_DIR=/path/to/KoreanStocks\n"
    )
    env_file.write_text(template, encoding="utf-8")
    typer.echo(".env 파일을 생성했습니다. API 키를 입력한 뒤 저장하세요.")
