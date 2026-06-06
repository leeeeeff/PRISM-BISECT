"""Home page — PRISM+BISECT hero banner, explanation, and navigation tiles."""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import streamlit.components.v1 as components

cfg = st.session_state.get('cfg') or {}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY GATE — hero splash until user clicks CTA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_qp = st.query_params
if _qp.get('entered'):
    st.session_state['app_entered'] = True
    st.session_state['_hero_mode'] = _qp.get('entered', 'demo')
    st.query_params.clear()
    st.rerun()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HERO — embedded via st.components (GSAP-style, iframe-optimized)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_HERO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --black:#000;--white:#fff;--off-white:#f5f5f7;--near-black:#1d1d1f;
  --muted-dark:rgba(255,255,255,0.45);--muted-light:rgba(29,29,31,0.45);
  --ease:cubic-bezier(0.16,1,0.3,1);
}
html{scroll-behavior:smooth}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Helvetica Neue',Arial,sans-serif;
  background:var(--black);color:var(--white);
  -webkit-font-smoothing:antialiased;overflow-x:hidden;
  transition:background-color 0.8s ease,color 0.8s ease;
}
body.light{background:var(--off-white);color:var(--near-black)}

/* Scenes */
.scene{width:100%;height:100vh;display:flex;align-items:center;
  justify-content:center;position:relative}
.scene-inner{width:100%;height:100%;display:flex;align-items:center;
  justify-content:center;padding:0 max(32px,8vw)}
.text-block{display:flex;flex-direction:column;gap:clamp(6px,1vw,12px);
  text-align:center;max-width:860px;width:100%}

/* Typography */
.wordmark{font-size:clamp(60px,11vw,128px);font-weight:600;letter-spacing:-0.04em;
  line-height:1;opacity:0;animation:rise 1.1s var(--ease) 0.1s forwards}
.tagline{font-size:clamp(15px,1.8vw,22px);font-weight:300;
  color:var(--muted-dark);margin-top:clamp(12px,1.8vw,20px);
  opacity:0;animation:rise 1.1s var(--ease) 0.38s forwards}
@keyframes rise{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}

.headline{font-size:clamp(32px,6vw,76px);font-weight:600;letter-spacing:-0.03em;
  line-height:1.06;opacity:0;transform:translateY(22px);
  transition:opacity 0.65s var(--ease),transform 0.65s var(--ease)}
.headline.massive{font-size:clamp(56px,11vw,136px);letter-spacing:-0.04em}
.headline:nth-child(2){transition-delay:0.18s}
.subline{font-size:clamp(14px,1.4vw,18px);font-weight:300;line-height:1.6;
  color:var(--muted-dark);margin-top:clamp(16px,2.2vw,28px);
  opacity:0;transform:translateY(12px);
  transition:opacity 0.65s var(--ease) 0.35s,transform 0.65s var(--ease) 0.35s}
.subline.light{color:var(--muted-light)}
.visible .headline,.visible .subline{opacity:1;transform:translateY(0)}

/* CTA */
.cta-block{display:flex;flex-direction:column;align-items:center;
  text-align:center;gap:0}
.cta-headline{font-size:clamp(30px,5vw,60px);font-weight:600;letter-spacing:-0.03em;
  line-height:1.05;color:var(--near-black);
  opacity:0;transform:translateY(20px);
  transition:opacity 0.7s var(--ease),transform 0.7s var(--ease)}
.cta-buttons{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;
  margin-top:clamp(32px,4vw,52px);
  opacity:0;transform:translateY(16px);
  transition:opacity 0.7s var(--ease) 0.2s,transform 0.7s var(--ease) 0.2s}
.cta-caption{margin-top:32px;font-size:11px;letter-spacing:0.04em;
  color:var(--muted-light);
  opacity:0;transition:opacity 0.7s ease 0.42s}
.visible .cta-headline,.visible .cta-buttons,.visible .cta-caption{
  opacity:1;transform:translateY(0)}

/* Buttons */
.btn{display:inline-flex;align-items:center;justify-content:center;
  height:48px;padding:0 28px;border-radius:100px;font-family:inherit;
  font-size:15px;font-weight:500;letter-spacing:0.01em;
  border:none;cursor:pointer;
  transition:transform 0.18s var(--ease),opacity 0.18s}
.btn:hover{transform:scale(1.025)}
.btn:active{transform:scale(0.975);opacity:0.85}
.btn-dark{background:var(--near-black);color:var(--white)}
.btn-outline{background:transparent;color:var(--near-black);
  border:1.5px solid rgba(29,29,31,0.35)}
.btn-outline:hover{border-color:var(--near-black)}

/* Scroll hint */
.scroll-hint{position:absolute;bottom:40px;left:50%;
  transform:translateX(-50%);color:rgba(255,255,255,0.3);
  opacity:0;animation:sfade 1s ease 1.5s both,bob 2.2s ease-in-out 1.5s infinite;
  transition:opacity 0.4s}
.scroll-hint.gone{opacity:0!important;animation:none}
@keyframes sfade{from{opacity:0}to{opacity:1}}
@keyframes bob{
  0%,100%{transform:translateX(-50%) translateY(0)}
  50%{transform:translateX(-50%) translateY(6px)}}

@media(max-width:600px){
  .cta-buttons{flex-direction:column;align-items:stretch}
  .btn{width:100%;height:52px}}
</style>
</head>
<body>

<section class="scene" id="s1">
  <div class="scene-inner" style="flex-direction:column;gap:0">
    <h1 class="wordmark">PRISM</h1>
    <p class="tagline">Isoform function. Decoded.</p>
  </div>
  <div class="scroll-hint" id="scroll-hint">
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none"
         stroke="currentColor" stroke-width="1.5">
      <path d="M12 5v14M5 12l7 7 7-7" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
</section>

<section class="scene" id="s2">
  <div class="scene-inner"><div class="text-block">
    <h2 class="headline">One gene.</h2>
    <h2 class="headline">Many proteins.</h2>
    <p class="subline">That's alternative splicing.</p>
  </div></div>
</section>

<section class="scene" id="s3">
  <div class="scene-inner"><div class="text-block">
    <h2 class="headline">Each isoform</h2>
    <h2 class="headline">carries a different function.</h2>
  </div></div>
</section>

<section class="scene" id="s4">
  <div class="scene-inner"><div class="text-block">
    <h2 class="headline">Under disease,</h2>
    <h2 class="headline">a different isoform takes over.</h2>
    <p class="subline">The function changes. Completely.</p>
  </div></div>
</section>

<section class="scene" id="s5">
  <div class="scene-inner"><div class="text-block">
    <h2 class="headline">No existing tool</h2>
    <h2 class="headline">could see the difference.</h2>
  </div></div>
</section>

<section class="scene" id="s6">
  <div class="scene-inner"><div class="text-block">
    <h2 class="headline massive">PRISM can.</h2>
    <p class="subline light">Sequence-based. 672 GO terms.<br>Zero-shot transfer.</p>
  </div></div>
</section>

<section class="scene" id="s7">
  <div class="scene-inner">
    <div class="cta-block">
      <p class="cta-headline">Explore isoform function.</p>
      <div class="cta-buttons">
        <button class="btn btn-dark" id="btn-demo">Explore demo data</button>
        <button class="btn btn-outline" id="btn-upload">Upload your data</button>
      </div>
      <p class="cta-caption">PRISM + BISECT &nbsp;·&nbsp; Lee et al. (2026) &nbsp;·&nbsp; Nature Machine Intelligence</p>
    </div>
  </div>
</section>

<script>
// ── IntersectionObserver: reveal scenes on scroll ─────────────────────────
const io = new IntersectionObserver(entries => {
  entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
}, { threshold: 0.45 });
document.querySelectorAll('#s2,#s3,#s4,#s5,#s6,#s7').forEach(s => io.observe(s));

// ── Background: dark → light when s6 appears ─────────────────────────────
const bgObs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      document.body.classList.add('light');
    } else if (e.boundingClientRect.top > 0) {
      document.body.classList.remove('light');
    }
  });
}, { threshold: 0.15 });
bgObs.observe(document.getElementById('s6'));

// ── Scroll hint fade ──────────────────────────────────────────────────────
let gone = false;
window.addEventListener('scroll', () => {
  if (!gone && window.scrollY > 50) {
    document.getElementById('scroll-hint').classList.add('gone');
    gone = true;
  }
}, { passive: true });

// ── CTA buttons ───────────────────────────────────────────────────────────
document.getElementById('btn-demo').addEventListener('click', () => {
  try { window.parent.location.href = '?entered=demo'; }
  catch(e) { window.location.href = '?entered=demo'; }
});
document.getElementById('btn-upload').addEventListener('click', () => {
  try { window.parent.location.href = '?entered=upload'; }
  catch(e) { window.location.href = '?entered=upload'; }
});
</script>
</body>
</html>"""

if not st.session_state.get('app_entered'):
    # ── Fullscreen hero mode ───────────────────────────────────────────────
    # CSS hides Streamlit chrome AND makes the component iframe cover the
    # entire viewport (position:fixed overrides the inline height attribute).
    # Only one iframe exists in hero mode, so the selector is unambiguous.
    st.markdown("""<style>
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    header[data-testid="stHeader"],
    #MainMenu, footer { display: none !important; }

    .main, .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"] {
        background: #000 !important;
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }

    iframe {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: 999999 !important;
        border: none !important;
    }
    </style>""", unsafe_allow_html=True)
    components.html(_HERO_HTML, height=800, scrolling=True)
    st.stop()

# ── Normal app mode (hero dismissed) ──────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0a0a0a 0%,#0f2942 60%,#0a1628 100%);
border-radius:16px;padding:48px 56px;margin-bottom:8px;color:white;text-align:center'>
  <div style='font-size:clamp(2.8rem,8vw,5rem);font-weight:700;
    letter-spacing:-0.04em;line-height:1;margin-bottom:12px'>PRISM</div>
  <div style='font-size:clamp(0.95rem,2vw,1.15rem);color:rgba(255,255,255,0.55);
    font-weight:300;letter-spacing:0.01em'>
    Isoform function. Decoded.
  </div>
  <div style='margin-top:18px;display:flex;justify-content:center;gap:32px;
    font-size:0.82rem;color:rgba(255,255,255,0.38);letter-spacing:0.04em'>
    <span>ESM-2 · 672 GO terms</span>
    <span>|</span>
    <span>Zero-shot transfer</span>
    <span>|</span>
    <span>Lee et al. 2026 · NMI</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WHAT THIS TOOL DOES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

c_prob, c_mid, c_prism, c_mid2, c_bisect = st.columns([10, 1, 10, 1, 10])

with c_prob:
    st.markdown("""
<div style='background:#fefce8;border-radius:12px;padding:24px 22px;height:100%'>
<div style='font-size:1.5rem;margin-bottom:10px'>❓</div>
<div style='font-size:1rem;font-weight:700;color:#92400e;margin-bottom:10px'>
왜 아이소폼 분석이 어려운가
</div>
<div style='font-size:0.875rem;color:#44403c;line-height:1.7'>
롱리드 싱글셀 시퀀싱은 수만 개의 <b>전사체 변이(아이소폼)</b>를 발견합니다.
그러나 대부분에 기능 주석이 없고, 기존 도구는 유전자 수준 주석만
아이소폼에 그대로 적용합니다.<br><br>
Novel 아이소폼(NIC·NNIC)은 Ensembl 주석이 아예 없어 Pfam·InterPro 기반
도구로는 처리 불가합니다.
</div>
</div>
""", unsafe_allow_html=True)

with c_mid:
    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:1.5rem;color:#94a3b8'>→</div>",
                unsafe_allow_html=True)

with c_prism:
    st.markdown("""
<div style='background:#eff6ff;border-radius:12px;padding:24px 22px;height:100%'>
<div style='font-size:1.5rem;margin-bottom:10px'>🔬</div>
<div style='font-size:1rem;font-weight:700;color:#1d4ed8;margin-bottom:10px'>
PRISM — 기능 예측
</div>
<div style='font-size:0.875rem;color:#1e3a5f;line-height:1.7'>
<b>ESM-2</b> 단백질 언어 모델로 아이소폼 고유 서열 특징을 추출,
<b>18~672개 GO BP 기능</b>을 동시에 예측합니다.<br><br>
Focal loss + 5-seed ensemble로 희소 라벨에서도 안정적으로 학습하며,
근육 학습 모델을 뇌에 <b>제로샷으로 전이</b>합니다 (AUPRC 0.672, 41-term panel).
Novel 아이소폼도 서열만 있으면 예측 가능합니다.
</div>
</div>
""", unsafe_allow_html=True)

with c_mid2:
    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:1.5rem;color:#94a3b8'>→</div>",
                unsafe_allow_html=True)

with c_bisect:
    st.markdown("""
<div style='background:#f0fdf4;border-radius:12px;padding:24px 22px;height:100%'>
<div style='font-size:1.5rem;margin-bottom:10px'>🎯</div>
<div style='font-size:1rem;font-weight:700;color:#15803d;margin-bottom:10px'>
BISECT — 기능 스위치 분류
</div>
<div style='font-size:0.875rem;color:#14532d;line-height:1.7'>
DTU 결과 + PRISM 스코어를 결합해 아이소폼별 <b>4-시나리오</b>를 분류합니다:
S1(기능 스위치), S2(발현 스위치), S3(신규 기능), S4(배경).<br><br>
AlphaFold pLDDT, Pfam 도메인, STRING PPI, phyloP 보존성 등
<b>15개 독립 증거 모듈</b>로 케이스별 등급을 산정합니다.
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4 KEY CAPABILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_CAPS = [
    ("🆕", "Novel 아이소폼 예측",
     "Ensembl 주석이 없는 NIC·NNIC 전사체도 ESM-2 서열 표현만으로 GO 기능 예측. "
     "InterPro·Pfam 기반 도구가 다루지 못하는 영역을 커버합니다.",
     "#f5f3ff", "#7c3aed"),
    ("🎯", "유전자 편향 극복",
     "아이소폼 고유 서열 특징을 사용해 gene-level label propagation 문제를 해소. "
     "같은 유전자의 아이소폼 간 점수 차이를 정량화합니다 (pos_bias 지표).",
     "#eff6ff", "#1d4ed8"),
    ("🔄", "DTU → 기능 변화 통합",
     "satuRn·DEXSeq·rMATS DTU 결과를 직접 연결해 어떤 기능이 획득/손실됐는지 정량화. "
     "GAIN/LOSS 기능 행렬을 세포 유형별로 분해합니다.",
     "#f0fdf4", "#15803d"),
    ("🧠", "672 GO 기능 모듈 지형도",
     "672 BP GO term을 Ward 계층 클러스터링으로 44개 기능 모듈로 조직화. "
     "63,994개 뇌 아이소폼을 모듈-수준 해상도로 분석합니다.",
     "#fff7ed", "#c2410c"),
]

cap_cols = st.columns(4)
for col, (icon, title, desc, bg, color) in zip(cap_cols, _CAPS):
    col.markdown(f"""
<div style='background:{bg};border-radius:10px;padding:20px 16px;height:100%'>
<div style='font-size:1.6rem;margin-bottom:8px'>{icon}</div>
<div style='font-size:0.92rem;font-weight:700;color:{color};margin-bottom:8px'>{title}</div>
<div style='font-size:0.8rem;color:#374151;line-height:1.65'>{desc}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GET STARTED
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.divider()
st.markdown("### 시작하기")

gs_demo, gs_upload = st.columns(2, gap="large")

with gs_demo:
    st.markdown("""
<div style='background:linear-gradient(135deg,#0f2942,#0f4c75);
border-radius:12px;padding:28px 28px 20px;color:white'>
<div style='font-size:1.4rem;font-weight:700;margin-bottom:10px'>
  📊 Demo 모드 — 논문 데이터 탐색
</div>
<div style='font-size:0.88rem;color:rgba(255,255,255,0.75);line-height:1.7;margin-bottom:14px'>
  별도 설치 없이 Brain 672 GO / Muscle 18 GO 사전 계산 결과를<br>
  인터랙티브하게 탐색합니다.<br><br>
  <b style='color:#7dd3fc'>논문 리뷰어 추천 경로:</b><br>
  사이드바 → Brain — Full (672 GO) 선택 → 아래 버튼으로 바로 이동
</div>
</div>
""", unsafe_allow_html=True)
    _d1, _d2 = st.columns(2)
    with _d1:
        if st.button("🏠 Analysis Hub 열기", use_container_width=True, key="home_demo_hub"):
            st.switch_page("pages/00_hub.py")
    with _d2:
        if st.button("🧫 BISECT 케이스 보기", use_container_width=True, key="home_demo_bisect"):
            st.switch_page("pages/07_bisect.py")

    # Landmark case shortcuts
    st.markdown("<div style='margin-top:8px;font-size:0.82rem;color:#64748b;font-weight:600'>⭐ Landmark case 바로 분석</div>", unsafe_allow_html=True)
    _lm1, _lm2, _lm3, _lm4 = st.columns(4)
    for _col, _gene in zip([_lm1, _lm2, _lm3, _lm4], ['NDUFS4', 'KIF21B', 'PTPRF', 'DLG1']):
        with _col:
            if st.button(_gene, key=f"home_lm_{_gene}", use_container_width=True):
                st.session_state['search_gene'] = _gene
                st.session_state['auto_search'] = True
                st.session_state['_targets_query_loaded'] = False
                st.switch_page("pages/05_targets.py")

with gs_upload:
    st.markdown("""
<div style='background:#f8fafc;border:2px solid #e2e8f0;
border-radius:12px;padding:28px 28px 20px'>
<div style='font-size:1.4rem;font-weight:700;color:#1e293b;margin-bottom:10px'>
  📤 Upload 모드 — 내 데이터 분석
</div>
<div style='font-size:0.88rem;color:#475569;line-height:1.7;margin-bottom:14px'>
  자체 롱리드 싱글셀 데이터를 분석합니다.<br>
  <code>prism-infer</code> 명령어로 NPY 스코어를 생성한 뒤 업로드하세요.<br><br>
  <b>필수:</b> PRISM 스코어 매트릭스 (N × GO, float32 NPY)<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;아이소폼 ID 목록 (NPY 또는 TXT)<br>
  <b>선택:</b> 유전자 ID · 아이소폼 타입 · DTU 결과 · SQANTI 분류
</div>
</div>
""", unsafe_allow_html=True)
    _u1, _u2 = st.columns(2)
    with _u1:
        if st.button("📖 Upload 가이드", use_container_width=True, key="home_upload_guide"):
            st.session_state['_show_upload_guide'] = True
    with _u2:
        if st.button("📂 사이드바 Upload 열기", use_container_width=True, key="home_upload_hint",
                     help="사이드바에서 Upload 모드 선택 후 파일을 업로드하세요"):
            st.info("👈 사이드바에서 **Upload (내 데이터)** 를 선택하고 파일을 업로드하세요.", icon="📂")

    if st.session_state.get('_show_upload_guide'):
        with st.expander("📖 Upload 가이드", expanded=True):
            st.markdown("""
| 파일 | 필수 | 형식 | 설명 |
|------|:----:|------|------|
| 스코어 매트릭스 | **필수** | `.npy` — `(N, GO)` float32 | PRISM 예측 결과 (0~1) |
| 아이소폼 ID | **필수** | `.npy` 또는 `.txt` | 전사체 ID, 행 순서 일치 |
| 유전자 ID | 권장 | `.npy` 또는 `.txt` | 유전자 심볼 또는 ENSG ID |
| 아이소폼 타입 | 선택 | `.npy` 또는 `.txt` — `known`/`nic`/`nnic` | Novel 분석 활성화 |
| DTU 결과 | 선택 | `.tsv` 또는 `.csv` | Condition 분석 활성화 |

**PRISM 스코어 생성 방법:**
```bash
conda activate isoform_env
prism-infer --input my_sequences.fasta --output my_scores.npy --panel brain_672
```
""")
            if st.button("닫기", key="upload_guide_close"):
                st.session_state['_show_upload_guide'] = False
                st.rerun()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADED — navigation tiles
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if cfg.get('score_matrix') is not None:
    import numpy as _np
    sm     = cfg['score_matrix']
    ids    = cfg['isoform_ids']
    thr    = cfg['score_threshold']
    tissue = cfg.get('tissue', '')
    has_dtu = cfg.get('dtu_df') is not None

    n_iso  = len(ids)
    n_go   = sm.shape[1]
    n_high = int((_np.asarray(sm) >= thr).any(axis=1).sum())

    st.divider()
    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("로드된 아이소폼",   f"{n_iso:,}")
    _m2.metric("GO 기능 패널",      f"{n_go}개")
    _m3.metric(f"고신뢰 (≥{thr})", f"{n_high:,}", f"{100*n_high/n_iso:.1f}%")
    _m4.metric("조직 패널",         tissue or cfg.get('mode', '—').capitalize())

    st.markdown("### 분석 페이지")

    _NAV = [
        ("🏠", "Analysis Hub",       "분석 워크플로우 · BISECT 케이스 · 빠른 조회",
         "pages/00_hub.py",         "#0f2942", "white"),
        ("📊", "QC & Overview",      "커버리지 · 시나리오 분류 · AUPRC 검증",
         "pages/01_qc.py",          "#eff6ff", "#1e293b"),
        ("🗺️", "Module Landscape",   "672 GO → 44 모듈 지형도",
         "pages/02_landscape.py",   "#f0fdf4", "#1e293b"),
        ("🔬", "Functional Patterns","GO 네트워크 · UMAP 클러스터",
         "pages/03_patterns.py",    "#fdf4ff", "#1e293b"),
        ("🔄", "Condition Analysis", "DTU GAIN/LOSS · GO Enrichment"
         + (" ✅" if has_dtu else " ⚠️ DTU필요"),
         "pages/04_condition.py",   "#fffbeb", "#1e293b"),
        ("🎯", "타겟 탐색",          "후보 발굴 · 모듈 지형도 · 바스켓 비교",
         "pages/05_target_hub.py",  "#faf5ff", "#1e293b"),
    ]

    _nc1, _nc2, _nc3 = st.columns(3)
    _nav_cols = [_nc1, _nc2, _nc3, _nc1, _nc2, _nc3]
    for col, (icon, name, desc, path, bg, fg) in zip(_nav_cols, _NAV):
        with col:
            st.markdown(
                f"<div style='background:{bg};border-radius:10px;padding:14px 16px;"
                f"margin-bottom:10px;min-height:90px'>"
                f"<span style='font-size:1.4rem'>{icon}</span>"
                f"<b style='color:{fg};font-size:0.9rem;margin-left:8px'>{name}</b><br>"
                f"<span style='color:{'rgba(255,255,255,0.7)' if bg=='#0f2942' else '#64748b'};"
                f"font-size:0.78rem'>{desc}</span></div>",
                unsafe_allow_html=True,
            )
            if st.button(f"▶ {name}", key=f"home_nav_{name.replace(' ','_')}",
                         use_container_width=True):
                st.switch_page(path)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL COMPARISON + INPUT FORMAT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

with st.expander("📊 기존 도구 대비 차별점", expanded=False):
    st.markdown("""
| 기능 | **PRISM+BISECT** | DIFFUSE (2022) | IsoformSwitchAnalyzeR | InterPro/Pfam |
|------|:---:|:---:|:---:|:---:|
| 아이소폼별 GO 기능 예측 | ✅ | △ (유전자 기준) | ❌ | ❌ |
| Novel isoform (NIC·NNIC) 지원 | ✅ | ❌ | ❌ | ❌ |
| Gene-level bias 극복 | ✅ | ❌ | — | — |
| DTU 결과 연동 + 기능 변화 정량화 | ✅ GAIN/LOSS | ❌ | ✅ (DTU만) | ❌ |
| Zero-shot 조직 전이 | ✅ | ❌ | — | △ |
| 672 GO term 기능 모듈 지형도 | ✅ 44 modules | ❌ | ❌ | ❌ |
| 인터랙티브 웹 분석 플랫폼 | ✅ | ❌ | ❌ | ❌ |
| 케이스 증거 등급화 (15 모듈) | ✅ BISECT | ❌ | ❌ | ❌ |

> ✅ 완전 지원 · △ 부분 지원 · ❌ 미지원 · — 해당 없음
""")

with st.expander("📁 Upload 모드 — 입력 파일 형식", expanded=False):
    st.markdown("""
| 파일 | 필수 | 형식 | 설명 |
|------|:----:|------|------|
| 스코어 매트릭스 | **필수** | `.npy` — `(N, GO)` float32 | PRISM 예측 결과 (0~1) |
| 아이소폼 ID | **필수** | `.npy` 또는 `.txt` — 1D 문자열 | 전사체 ID, 행 순서 일치 |
| 유전자 ID | 권장 | `.npy` 또는 `.txt` — 1D 문자열 | 유전자 심볼 또는 ENSG ID |
| 아이소폼 타입 | 선택 | `.npy` 또는 `.txt` — `known`/`nic`/`nnic` | Novel 분석 활성화 |
| DTU 결과 | 선택 | `.tsv` 또는 `.csv` | Condition 분석 활성화 |
""")

st.markdown("""
<div style='margin-top:24px;text-align:center;font-size:0.8rem;color:#94a3b8'>
Lee et al. (2026) · PRISM+BISECT · <i>Nature Machine Intelligence</i> (in review)
</div>
""", unsafe_allow_html=True)
