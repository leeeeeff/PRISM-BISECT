// DATA-INSTRUMENT 공유 차트 테마 — main.css :root 토큰과 동기. 모든 Plotly 페이지가 사용.
'use strict';
// T 는 CSS 커스텀 프로퍼티에서 파생 → 흑/백 테마 전환 시 refreshTheme() 로 갱신된다.
// (dark 값을 기본값으로 둬서 CSS var 를 못 읽는 상황에서도 안전)
const T = {
  ink900:'#0B0E14', ink800:'#0E121A', grid:'#1B2230', grid2:'#28324a',
  txt:'#C9D2DE', txtHi:'#F2F5FA', dim:'#66728A', signal:'#D9764A', amber:'#C99A52',
  trace:'#4FB4D0', ok:'#4FC08A', warn:'#E06A80',
};
// CSS 토큰 → T 키 매핑
const _T_VARS = {
  ink900:'--ink-900', ink800:'--ink-800', grid:'--grid', grid2:'--grid-2',
  txt:'--txt', txtHi:'--txt-hi', dim:'--txt-dim', signal:'--signal', amber:'--amber',
  trace:'--trace', ok:'--ok', warn:'--warn',
};
function refreshTheme(){
  const cs = getComputedStyle(document.documentElement);
  for(const [k, v] of Object.entries(_T_VARS)){
    const val = cs.getPropertyValue(v).trim();
    if(val) T[k] = val;
  }
}
window.refreshTheme = refreshTheme;
refreshTheme();   // 로드 시점의 활성 테마 반영(첫 렌더용)
const MONO = 'IBM Plex Mono, ui-monospace, monospace';
const PLOT_CFG = {displayModeBar:false, responsive:true};

// Okabe-Ito 색각이상-안전 다범주 팔레트(Okabe & Ito 2008) — Nature/Cell 계열 학술 표준, 적록색맹
// (protanopia/deuteranopia)·청황색맹(tritanopia)에서도 구분되고 흑백 인쇄에서도 명암으로 구별됨.
// 이전에 여러 페이지(mydata.js/bisect.js)가 각자 비슷하지만 미묘하게 다른 8색 배열을 하드코딩했던
// 것을 여기 하나로 통합(요청: "figure들의 색상 표현에 통일성과 일관성").
//
// OKABE_ITO_SAFE(6색) — 이 앱의 카테고리 팔레트(CLUSTER_PAL 등)는 T(위)와 달리 테마 전환에
// 반응하지 않는 고정값이므로, 다크 잉크 패널과 라이트 종이 패널 양쪽에서 전부 대비가 나와야
// 한다. 원 8색 중 검정(#000000)은 다크 패널(--ink-800 배경)에서 사실상 안 보이고, 노랑
// (#F0E442)은 라이트 패널(흰 배경)에서 선/마커로 쓰기엔 명도가 너무 높아 대비가 약하다 —
// 이 둘을 뺀 6색이 두 테마 모두에서 안전한 핵심 세트.
const OKABE_ITO_SAFE = ['#0072B2','#D55E00','#009E73','#E69F00','#CC79A7','#56B4E9'];
// OKABE_ITO_FULL(8색, 검정·노랑 포함) — 확실히 라이트/인쇄 전용인 맥락(예: 정적 export)에서만.
const OKABE_ITO_FULL = OKABE_ITO_SAFE.concat(['#000000', '#F0E442']);
// 이진 대비(증가/감소, gain/loss, pass/fail) 전용 — 적록 대신 청-주홍 조합
// ("적색과 녹색의 단독 대비는 피하라"는 색각이상 가이드라인 그대로 적용).
const DIVERGE_POS = '#0072B2', DIVERGE_NEG = '#D55E00';
// 연속형(그라디언트) 데이터는 Plotly 내장 Viridis 사용 — 지각적으로 균일하고 무지개(Jet) 아님.
const VIRIDIS = 'Viridis';

// n개 구분 색상 — n<=6 이면 OKABE_ITO_SAFE 그대로, 그 이상(예: 8개 brain 세포유형, 30개 GO
// subterm drill-down)이면 HSL 균등분할로 확장. 다크 잉크 패널·라이트 종이 패널 양쪽 대비를
// 위해 저채도 파스텔이 아니라 중간 채도·낮은 명도로 생성(mydata.js/bisect.js 공유).
function distinctPalette(n){
  if(n <= OKABE_ITO_SAFE.length) return OKABE_ITO_SAFE.slice(0, n);
  const out = [];
  for(let i=0;i<n;i++) out.push(`hsl(${Math.round(360*i/n)}, 60%, 40%)`);
  return out;
}

// 계측기 공통 레이아웃(축·격자·폰트를 CSS 토큰과 일치)
function inst(extra){
  const base = {
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:MONO, size:11, color:T.dim},
    margin:{l:50,r:12,t:8,b:34},
    xaxis:{gridcolor:T.grid, zerolinecolor:T.grid2, linecolor:T.grid,
           tickfont:{family:MONO,size:10,color:T.dim}, ticks:'outside', tickcolor:T.grid2, ticklen:4},
    yaxis:{gridcolor:T.grid, zerolinecolor:T.grid2, linecolor:T.grid,
           tickfont:{family:MONO,size:10,color:T.dim}, ticks:'outside', tickcolor:T.grid2, ticklen:4},
    hoverlabel:{bgcolor:T.ink800, bordercolor:T.signal, font:{family:MONO,size:11,color:T.txt}},
  };
  return Object.assign(base, extra||{});
}

// count-up (계측기 값 카운트업)
function countUp(el, to, dec){
  const dur=420, t0=performance.now(), from=0;
  function step(t){ const p=Math.min((t-t0)/dur,1); const e=1-Math.pow(1-p,3);
    el.textContent=(from+(to-from)*e).toFixed(dec); if(p<1) requestAnimationFrame(step); }
  requestAnimationFrame(step);
}
