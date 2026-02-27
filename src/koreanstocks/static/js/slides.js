/**
 * slides.js — API 데이터를 받아 Reveal.js 슬라이드 동적 생성
 */

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function fmt(n, digits = 0) {
  if (n == null) return "-";
  return Number(n).toLocaleString("ko-KR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function chgClass(v) { return v >= 0 ? "pos" : "neg"; }

function chgText(v) {
  if (v == null) return "-";
  return (v >= 0 ? "▲" : "▼") + " " + Math.abs(v).toFixed(2) + "%";
}

function badgeClass(action) {
  const a = (action || "").toUpperCase();
  if (a === "BUY")  return "badge badge-buy";
  if (a === "SELL") return "badge badge-sell";
  return "badge badge-hold";
}

function slScoreBar(label, value) {
  const pct   = Math.min(100, Math.max(0, value || 0));
  const color = pct >= 70 ? "var(--buy)" : pct >= 40 ? "var(--accent)" : "var(--hold)";
  return `
    <div class="sl-score-row">
      <span class="sl-score-label">${label}</span>
      <div class="sl-score-track">
        <div class="sl-score-fill" style="width:${pct}%;background:${color}"></div>
      </div>
      <span class="sl-score-val">${pct.toFixed(0)}</span>
    </div>`;
}

// ── 표지 슬라이드 ────────────────────────────────────────────────
function coverSlide(market, date, count) {
  const kospi  = market.KOSPI  || {};
  const kosdaq = market.KOSDAQ || {};
  const usd    = market.USDKRW || {};

  function mktCard(label, info) {
    if (!info || !info.close) return "";
    const chg = info.change || 0;
    return `
      <div class="sl-mkt-card">
        <div class="sl-mkt-label">${label}</div>
        <div class="sl-mkt-val">${fmt(info.close, 2)}</div>
        <div class="sl-mkt-chg ${chgClass(chg)}">${chgText(chg)}</div>
      </div>`;
  }

  return `
    <section>
      <div class="sl-cover">
        <div>
          <div class="cover-title">📊 오늘의 시장 브리핑</div>
          <div class="cover-sub">${date || ""}  ·  분석 종목 ${count}개</div>
        </div>
        <div class="sl-mkt-row">
          ${mktCard("KOSPI",   kospi)}
          ${mktCard("KOSDAQ",  kosdaq)}
          ${usd.close ? mktCard("USD/KRW", { close: usd.close, change: 0 }) : ""}
        </div>
        <div class="sl-cover-nav">
          ← / → 키 또는 터치로 탐색
          <span class="sl-divider">|</span>
          <a href="/dashboard">대시보드 →</a>
        </div>
      </div>
    </section>`;
}

// ── 종목 슬라이드 ────────────────────────────────────────────────
function stockSlide(rec) {
  const opinion  = rec.ai_opinion || {};
  const action   = opinion.action || "HOLD";
  const si       = rec.sentiment_info || {};
  const news     = si.headlines || si.articles?.slice(0,3).map(a => a.title) || [];
  const strength = opinion.strength || "";
  const weakness = opinion.weakness || "";
  const ml_score  = rec.ml_score != null ? rec.ml_score : (rec.tech_score || 0);
  const sent_norm = Math.min(100, Math.max(0, ((rec.sentiment_score || 0) + 100) / 2));

  const newsHtml = news.slice(0, 3).map(h =>
    `<div class="sl-news-item">· ${h}</div>`
  ).join("");

  const swHtml = (strength || weakness) ? `
    <div class="sl-sw-grid">
      ${strength ? `<div class="sl-sw-box">
        <div class="sl-sw-title buy">💪 강점</div>
        <div>${strength}</div>
      </div>` : ""}
      ${weakness ? `<div class="sl-sw-box">
        <div class="sl-sw-title hold">⚠️ 약점</div>
        <div>${weakness}</div>
      </div>` : ""}
    </div>` : "";

  return `
    <section>
      <div class="sl-stock">
        <!-- 헤더 -->
        <div class="sl-stock-header">
          <span class="sl-stock-name">${rec.name || rec.code}</span>
          <span class="sl-stock-code">${rec.code}</span>
          <span class="${badgeClass(action)}">${action}</span>
          <div class="sl-stock-price">
            <div class="sl-price-val">₩${fmt(rec.current_price)}</div>
            <div class="sl-price-chg ${chgClass(rec.change_pct)}">${chgText(rec.change_pct)}</div>
          </div>
        </div>

        <!-- 본문 2열 -->
        <div class="sl-stock-body">
          <!-- 좌측: 점수 + 목표가 + 뉴스 -->
          <div class="sl-col">
            ${slScoreBar("기술점수", rec.tech_score)}
            ${slScoreBar("ML점수",   ml_score)}
            ${slScoreBar("감성점수", sent_norm)}
            ${opinion.target_price ? `
              <div class="sl-target">🎯 목표가: ₩${fmt(opinion.target_price)}</div>` : ""}
            ${newsHtml ? `<div class="sl-news-wrap">${newsHtml}</div>` : ""}
          </div>

          <!-- 우측: AI 요약 + 강점/약점 -->
          <div class="sl-col">
            <div class="sl-ai-box">${opinion.summary || "분석 내용 없음"}</div>
            ${swHtml}
          </div>
        </div>
      </div>
    </section>`;
}

// ── 마지막 슬라이드 ──────────────────────────────────────────────
function navSlide() {
  return `
    <section>
      <div class="sl-cover">
        <h2 style="color:var(--accent);font-weight:800;margin:0 0 6px">📌 더 알아보기</h2>
        <div class="cover-sub">분석 결과를 대시보드에서 상세 확인하세요.</div>
        <div class="sl-mkt-row" style="gap:20px;margin-top:8px">
          <a href="/dashboard" class="sl-nav-card">
            <div class="sl-nav-icon">📈</div>
            <div class="sl-nav-title">대시보드</div>
            <div class="sl-nav-desc">Watchlist · 백테스트 · 수동 분석</div>
          </a>
          <a href="/docs" class="sl-nav-card">
            <div class="sl-nav-icon">📖</div>
            <div class="sl-nav-title">API 문서</div>
            <div class="sl-nav-desc">Swagger UI · 엔드포인트 테스트</div>
          </a>
        </div>
        <p class="sl-disclaimer">※ 본 분석은 투자 참고용이며 투자 결정의 책임은 투자자 본인에게 있습니다.</p>
      </div>
    </section>`;
}

// ── 테마 토글 ────────────────────────────────────────────────────
function updateRevealTheme() {
  const link = document.getElementById("reveal-theme");
  if (!link) return;
  const isDark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
  const base = "https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/theme/";
  link.href = base + (isDark ? "black.css" : "white.css");
}

function toggleTheme() {
  const root = document.documentElement;
  const next = (root.getAttribute("data-theme") || "dark") === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  localStorage.setItem("ks-theme", next);
  updateRevealTheme();
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = next === "dark" ? "☀️" : "🌙";
}

// ── 슬라이드 빌드 + Reveal 초기화 ───────────────────────────────
async function buildSlides() {
  const container = document.getElementById("slide-container");

  try {
    const [recData, marketData] = await Promise.all([
      fetchJSON("/api/recommendations"),
      fetchJSON("/api/market").catch(() => ({ KOSPI: {}, KOSDAQ: {} })),
    ]);

    const recs = recData.recommendations || [];
    const date = recData.date || "";

    let html = coverSlide(marketData, date, recs.length);
    html += recs.map(stockSlide).join("");
    html += navSlide();
    container.innerHTML = html;

  } catch (err) {
    container.innerHTML = `
      <section>
        <div class="sl-cover">
          <h2 style="color:var(--sell)">데이터 로드 실패</h2>
          <p>${err.message}</p>
          <p style="font-size:.85em;color:var(--muted)">
            서버가 실행 중인지 확인하거나
            <a href="/api/recommendations/run">분석을 실행</a>해 주세요.
          </p>
        </div>
      </section>`;
  }

  // 로딩 오버레이 숨기기
  const overlay = document.getElementById("loading-overlay");
  if (overlay) overlay.style.display = "none";

  // Reveal.js 5.x: initialize()는 최초 1회만, 이후 동적 슬라이드 추가 시 sync() 사용
  if (Reveal.isReady()) {
    Reveal.sync();
    Reveal.slide(0);
  } else {
    await Reveal.initialize({
      hash: true,
      controls: true,
      progress: true,
      center: false,
      transition: "slide",
      backgroundTransition: "fade",
      width: 1100,
      height: 700,
      margin: 0.05,
      minScale: 0.5,
      maxScale: 1.5,
    });
  }

  updateRevealTheme();
  const btn = document.getElementById("theme-toggle");
  if (btn) {
    const isDark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
    btn.textContent = isDark ? "☀️" : "🌙";
  }
}

buildSlides();
