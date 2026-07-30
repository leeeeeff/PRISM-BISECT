// Individual analysis flagship — search + panels (Plotly). Theme shared via instrument.js.
'use strict';

function goSearch(form){
  const q = form.q.value.trim();
  if(q) window.location = '/gene/' + encodeURIComponent(q);
}

// ── autocomplete ────────────────────────────────────────────
const qEl = document.getElementById('q');
const sugEl = document.getElementById('suggest');
let sugTimer = null;
if(qEl){
  qEl.addEventListener('input', () => {
    clearTimeout(sugTimer);
    const v = qEl.value.trim();
    if(v.length < 1){ sugEl.innerHTML=''; return; }
    sugTimer = setTimeout(async () => {
      const r = await fetch('/api/suggest?q=' + encodeURIComponent(v));
      const hits = await r.json();
      sugEl.innerHTML = hits.map(h => `<li onclick="pick('${h}')">${h}</li>`).join('');
    }, 120);
  });
  document.addEventListener('click', e => { if(!e.target.closest('.search-bar')) sugEl.innerHTML=''; });
}
function pick(g){
  window.location = '/gene/' + encodeURIComponent(g);
}

// 모듈 색 — 고정 hex 대신 T(theme.js/instrument.js 가 갱신) 에서 매 호출마다 파생 → 흑/백 전환 시 자동 추종.
function modColor(m){ return [T.signal, T.trace, T.amber, T.ok][((m%4)+4)%4]; }

// ── Gene Quick Report — open by default; expands to ONE axis panel at a
// time (S1/S3/BISECT/Diversity selected via tabs), ported from streamlit
// target_hub.py's card format. Colors derive from T (theme tokens), same
// convention as bisect.js TIER_COLOR_KEY. Auto-collapses once renderDetail()
// shows the full panel readout for a picked gene/isoform (qcCollapse()).
const QC_AXES = ['s1', 's3', 'bisect', 'diversity'];
const AXIS_COLOR_KEY = {s1: 'signal', s3: 'trace', bisect: 'trace', diversity: 'amber'};
let QC_DATA = null, _qcActiveAxis = 's1', _qcLoaded = false;
const qcIdx = {s1: 0, s3: 0, bisect: 0, diversity: 0};

// 접힘/펼침 — 기본 펼침(open), 처음 렌더 시 바로 fetch. 이후 토글은 손으로.
const qcToggle = document.getElementById('qc-toggle');
const qcBody = document.getElementById('qc-body');
if(qcToggle){
  qcToggle.addEventListener('click', () => {
    const expanding = qcBody.classList.contains('hidden');
    qcBody.classList.toggle('hidden');
    qcToggle.setAttribute('aria-expanded', String(expanding));
    qcToggle.querySelector('.qc-chevron').textContent = expanding ? '▾' : '▸';
    if(expanding && !_qcLoaded){ _qcLoaded = true; loadQuickCases(); }
  });
}
if(qcBody && !qcBody.classList.contains('hidden')){ _qcLoaded = true; loadQuickCases(); }
function qcCollapse(){
  if(!qcBody || qcBody.classList.contains('hidden')) return;
  qcBody.classList.add('hidden');
  qcToggle.setAttribute('aria-expanded', 'false');
  qcToggle.querySelector('.qc-chevron').textContent = '▸';
}
// 탭 선택 — 4개 패널을 동시에 쌓지 않고 하나만 표시.
const qcAxisTabs = document.getElementById('qc-axis-tabs');
if(qcAxisTabs){
  qcAxisTabs.addEventListener('click', e => {
    const b = e.target.closest('.ttab'); if(!b) return;
    qcShowAxis(b.dataset.axis);
  });
}
function qcShowAxis(ax){
  _qcActiveAxis = ax;
  document.querySelectorAll('#qc-axis-tabs .ttab').forEach(b => b.classList.toggle('on', b.dataset.axis === ax));
  document.querySelectorAll('.qc-axis-panel').forEach(p => p.classList.toggle('hidden', p.dataset.axis !== ax));
}

async function loadQuickCases(){
  const q = await (await fetch('/api/quick_cases')).json();
  if(q.error) return;
  QC_DATA = q;
  QC_AXES.forEach(ax => {
    const panel = document.querySelector(`.qc-axis-panel[data-axis="${ax}"]`);
    const tab = document.querySelector(`#qc-axis-tabs .ttab[data-axis="${ax}"]`);
    if(!panel) return;
    const m = (q.meta || {})[ax] || {label: ax, desc: ''};
    const col = T[AXIS_COLOR_KEY[ax]] || T.dim;
    panel.querySelector('.qc-axis-label').textContent = m.label;
    panel.querySelector('.qc-axis-label').style.color = col;
    panel.querySelector('.qc-axis-desc').textContent = m.desc;
    panel.style.setProperty('--qc-accent', col);
    panel.querySelector('.qc-prev').onclick = () => qcStep(ax, -1);
    panel.querySelector('.qc-next').onclick = () => qcStep(ax, 1);
    if(tab) tab.querySelector('.qc-tab-label').textContent = m.label;
    qcIdx[ax] = 0;
    renderQcCard(ax);
  });
}
function qcStep(ax, delta){
  const items = (QC_DATA && QC_DATA[ax]) || [];
  if(!items.length) return;
  qcIdx[ax] = (qcIdx[ax] + delta + items.length) % items.length;
  renderQcCard(ax);
}
function qcStat(item, ax){
  if(ax === 'diversity') return `Δrange ${item.score_range.toFixed(2)} · ${item.n_isoforms} isoforms`;
  if(ax === 'bisect') return `${item.bisect_tier} · ΔIF ${item.delta.toFixed(2)} · ${item.mechanism}`;
  return `max score ${item.max_score.toFixed(3)}`;
}
async function renderQcCard(ax){
  const panel = document.querySelector(`.qc-axis-panel[data-axis="${ax}"]`);
  if(!panel) return;
  const body = panel.querySelector('.qc-card-body');
  const posEl = panel.querySelector('.qc-pos');
  const items = (QC_DATA && QC_DATA[ax]) || [];
  if(!items.length){ body.innerHTML = '<p class="muted mono sm">no candidates for this axis</p>'; posEl.textContent = ''; return; }
  const i = qcIdx[ax], item = items[i];
  posEl.textContent = `${i + 1} / ${items.length}`;
  body.innerHTML = `<p class="muted mono sm">▸ loading ${item.gene}…</p>`;
  const d = await (await fetch('/api/quick_case_detail/' + encodeURIComponent(item.gene))).json();
  if(d.error){ body.innerHTML = `<p class="err mono sm">✕ ${d.error}</p>`; return; }
  body.innerHTML = qcCardHtml(item, d, ax);
  drawQcHeatmap(`qcHeat-${ax}`, d.heatmap);
}
function qcCardHtml(item, d, ax){
  const rows = d.isoforms.slice().sort((a, b) => b.max_go_score - a.max_go_score).slice(0, 8);
  const tbl = rows.map(r => `<tr onclick="pick('${d.gene}')">
      <td>${r.isoform_id}</td><td class="muted">${r.type}</td>
      <td class="tnum">${r.max_go_score.toFixed(3)}</td>
      <td class="muted">${(r.top_go && r.top_go.name || '').slice(0, 26)}</td>
      <td class="muted">${r.module_label}</td></tr>`).join('');
  return `
    <div class="qc-banner">
      <div>
        <span class="qc-gene-name">${d.gene}</span>${geneCardsLink(d.gene)}
        <span class="qc-substat mono sm">${d.n_isoforms} isoforms · ${qcStat(item, ax)}</span>
      </div>
      <button class="btn" onclick="pick('${d.gene}')">▸ full analysis</button>
    </div>
    <div class="qc-card-grid">
      <div class="qc-heat-col"><div id="qcHeat-${ax}"></div></div>
      <div class="qc-tbl-col"><table class="qc-iso-tbl">
        <thead><tr><th>isoform</th><th>type</th><th>score</th><th>top GO</th><th>module</th></tr></thead>
        <tbody>${tbl}</tbody></table></div>
      <div class="qc-metrics-col">
        ${qcMetric('Isoforms', d.n_isoforms)}
        ${qcMetric('High-conf', d.n_high_conf, `score>${d.high_conf_threshold}`)}
        ${qcMetric('Novel', d.n_novel, 'NIC+NNIC')}
        ${qcMetric('DTU sig.', d.n_dtu_sig, 'p<0.05')}
      </div>
    </div>`;
}
function qcMetric(k, v, sub){
  return `<div class="qc-metric"><div class="qc-metric-k">${k}</div><div class="qc-metric-v">${v}</div>
    ${sub ? `<div class="qc-metric-sub muted">${sub}</div>` : ''}</div>`;
}
// 컴팩트 히트맵 — 풀페이지 renderHeatmap() 과 달리 모듈 스트립/클릭 확장 없이 값만 빠르게 보여준다.
function drawQcHeatmap(divId, hm){
  const host = document.getElementById(divId);
  if(!host || !hm) return;
  Plotly.newPlot(divId, [{
    type: 'heatmap', z: hm.matrix, x: hm.go_names, y: hm.isoform_ids,
    // Viridis — sequential/perceptually-uniform, matches every other score-gradient figure in the
    // app (mydata.js UMAP max_score) instead of a bespoke grid→amber→signal ramp; also fixes a
    // separate latent issue the old ramp had to work around (T.ink900 alone would've broken
    // monotonicity in light theme) since Viridis is fixed regardless of theme.
    colorscale: VIRIDIS, zmin: 0, zmax: 1,
    showscale: true, colorbar: {thickness: 7, len: .55, tickfont: {family: MONO, size: 7, color: T.dim}, outlinewidth: 0},
    hovertemplate: '%{y}<br>%{x}<br>score %{z:.3f}<extra></extra>',
  }], inst({height: Math.max(160, 26 * hm.isoform_ids.length + 60), margin: {l: 110, r: 34, t: 6, b: 90},
    xaxis: {tickangle: -40, tickfont: {family: MONO, size: 8, color: T.dim}, automargin: true},
    yaxis: {tickfont: {family: MONO, size: 8, color: T.dim}, autorange: 'reversed', automargin: true}}), PLOT_CFG);
}

// ── result load ─────────────────────────────────────────────
async function loadResult(){
  const root = document.getElementById('result');
  if(!root) return;
  const kind = root.dataset.kind, key = root.dataset.key;
  if(kind === 'gene'){
    const prof = await (await fetch('/api/gene/' + encodeURIComponent(key))).json();
    if(prof.error) return;
    renderIsoList(prof);
    renderDetail(prof.detail, prof.isoforms);
  } else if(kind === 'isoform'){
    const prof = await (await fetch('/api/isoform/' + encodeURIComponent(key))).json();
    if(prof.error) return;
    // same gene isoform list as the gene-search path (§02 table + heatmap + sticky mini-summary) —
    // an isoform-id search shouldn't lose sibling context just because the query wasn't the gene symbol.
    const gp = await (await fetch('/api/gene/' + encodeURIComponent(prof.gene))).json();
    if(!gp.error) renderIsoList(gp);
    renderDetail(prof, gp.isoforms);
  }
}

let _geneProf = null;   // last-rendered gene profile — reused by the sticky mini-summary
function renderIsoList(prof){
  _geneProf = prof;
  const sec = document.getElementById('iso-list');
  document.getElementById('gene-readout').innerHTML =
    `<span><span class="k">GENE</span> <b>${prof.gene}</b>${geneCardsLink(prof.gene)}</span>` +
    `<span><span class="k">ISOFORMS</span> <b id="ro-niso">0</b></span>` +
    `<span><span class="k">REPRESENTATIVE</span> <b>${prof.representative}</b></span>` +
    `<span><span class="k">UNIVERSE</span> <b>brain·672-GO</b></span>`;
  countUp(document.getElementById('ro-niso'), prof.n_isoforms, 0);
  const tb = document.getElementById('iso-tbody');
  tb.innerHTML = prof.isoforms.map(iso => `
    <tr class="${iso.isoform_id===prof.representative?'is-rep':''}${Cart.has(iso.isoform_id)?' is-cart-selected':''}${iso.is_noncoding?' is-noncoding':''}"
        data-iso-id="${iso.isoform_id}" data-gene="${prof.gene}"
        onclick="loadIsoform('${iso.isoform_id}')">
      <td onclick="event.stopPropagation()"><input type="checkbox" class="iso-cart-chk"></td>
      <td>${iso.isoform_id}${iso.isoform_id===prof.representative?' <span class="tag">REP</span>':''}
        ${Cart.has(iso.isoform_id)?' <span class="tag cart-tag" title="already in cart">CART</span>':''}
        ${iso.is_noncoding?' <span class="iso-noncoding-flag" title="non-coding — zero ESM-2 embedding, no functional signal">⚠ non-coding</span>':''}${geneCardsLink(prof.gene)}</td>
      <td>L${iso.peak_layer}</td>
      <td>${iso.axis3_domain.toFixed(2)} <span class="muted" title="${iso.axis3_pct}th percentile (domain content) among brain-wide isoforms">p${iso.axis3_pct}</span></td>
      <td>${iso.axis5_length.toFixed(2)}</td>
      <td>${iso.max_go_score.toFixed(3)}</td>
      <td class="muted">${iso.top_go.name}</td>
    </tr>`).join('');
  sec.classList.remove('hidden');
  const cartStatus = document.getElementById('isoAddCartStatus');
  if(cartStatus) cartStatus.textContent = '';
  // 섹션을 먼저 표시한 뒤 히트맵을 그린다 — display:none 상태에서 Plotly 를 그리면
  // 컨테이너 clientWidth=0 → 기본 ~700px 로 폴백돼 박스 왼쪽에 치우쳐 렌더된다.
  renderHeatmap(prof.heatmap);
  setupIsoMiniSummaryObserver();
}

// marks whichever isoform is currently open in §03 (Readout panels) as visually distinct in the
// §02 table — separate from "is-rep" (the gene's default/most-complete isoform), since the user can
// browse to any sibling from the list, from search, or from the mini-summary below.
function markIsoListCurrent(id){
  document.querySelectorAll('#iso-tbody tr[data-iso-id]').forEach(tr => {
    tr.classList.toggle('is-current', tr.dataset.isoId === id);
  });
  const miniHost = document.getElementById('isoMiniSummary');
  if(miniHost && !miniHost.classList.contains('hidden')) renderIsoMiniSummary();
}

// sticky compact recap of the isoform list — appears (wide viewports only, see CSS) once the real
// §02 list scrolls out of view, so the other isoforms of this gene stay glanceable while reading
// §03 without scrolling back up; disappears again once §02 is back on screen.
let _isoMiniObserver = null;
function renderIsoMiniSummary(){
  const host = document.getElementById('isoMiniSummary');
  if(!host || !_geneProf) return;
  const rows = _geneProf.isoforms.map(iso => `
    <div class="ims-row${iso.isoform_id===_curId?' ims-cur':''}" onclick="loadIsoform('${iso.isoform_id}')">
      <div class="ims-line1"><span class="ims-id">${iso.isoform_id}${iso.is_noncoding?' ⚠':''}</span>
        <span class="ims-score tnum">${iso.max_go_score.toFixed(2)}</span></div>
      <div class="ims-go" title="${escapeIms(iso.top_go.name)}">${escapeIms(iso.top_go.name)}</div>
    </div>`).join('');
  host.innerHTML = `<div class="ims-gene">${_geneProf.gene}</div>${rows}`;
}
function escapeIms(s){ return String(s).replace(/"/g, '&quot;'); }
function setupIsoMiniSummaryObserver(){
  const target = document.querySelector('#iso-list table');
  const host = document.getElementById('isoMiniSummary');
  if(!target || !host) return;
  if(_isoMiniObserver) _isoMiniObserver.disconnect();
  renderIsoMiniSummary();
  _isoMiniObserver = new IntersectionObserver((entries) => {
    const visible = entries[0].isIntersecting;
    host.classList.toggle('hidden', visible);
    if(!visible) renderIsoMiniSummary();   // refresh (current isoform / cart state may have changed)
  }, {threshold: 0});
  _isoMiniObserver.observe(target);
}

// 체크박스는 이제 순수 "선택"(스테이징)일 뿐 카트 상태와 무관 — 체크만으로 자동으로 담기던
// 이전 동작이 처음 쓰는 사람에게 안 직관적이라는 피드백으로, "Add selected to CART" 버튼을 눌러야만
// 실제로 담긴다. 이미 카트에 있는지는 체크박스가 아니라 행의 CART 배지로 별도 표시.
function addSelectedToCart(){
  const rows = document.querySelectorAll('#iso-tbody tr[data-iso-id]');
  let n = 0;
  rows.forEach(tr => {
    const chk = tr.querySelector('.iso-cart-chk');
    if(chk && chk.checked && !Cart.has(tr.dataset.isoId)){
      Cart.add(tr.dataset.isoId, 'isoform', tr.dataset.gene);
      chk.checked = false;
      n++;
    }
  });
  const status = document.getElementById('isoAddCartStatus');
  if(status) status.textContent = n ? `+${n} added to cart` : 'nothing new selected';
}
Cart.onChange(() => {
  document.querySelectorAll('#iso-tbody tr[data-iso-id]').forEach(tr => {
    const id = tr.dataset.isoId, inCart = Cart.has(id);
    tr.classList.toggle('is-cart-selected', inCart);
    let badge = tr.querySelector('.cart-tag');
    if(inCart && !badge){
      const idCell = tr.children[1];
      idCell.insertAdjacentHTML('beforeend', ' <span class="tag cart-tag">CART</span>');
      badge = tr.querySelector('.cart-tag');
    } else if(!inCart && badge){
      badge.remove();
    }
    if(badge) badge.title = Cart.isInHub(id) ? 'In Hub — remove it from the Hub page to un-track' : 'already in cart';
  });
});

async function loadIsoform(id){
  const prof = await (await fetch('/api/isoform/' + encodeURIComponent(id))).json();
  if(prof.error) return;
  renderDetail(prof);   // sets _curId and calls markIsoListCurrent() — keeps §02/mini-summary in sync
                         // even when clicked from inside the mini-summary itself (already scrolled
                         // past §02, so the IntersectionObserver visibility state wouldn't otherwise change).
  window.scrollTo({top:document.getElementById('panels').offsetTop-60, behavior:'smooth'});
}

// state shared across panels
let _siblings=[], _curId=null, _axes=null, _axesSummary='';

async function renderDetail(d, siblings){
  if(typeof qcCollapse === 'function') qcCollapse();   // full analysis 뜨면 Quick Report 접기
  document.getElementById('panels').classList.remove('hidden');
  document.getElementById('detail-title').textContent = d.isoform_id;
  document.getElementById('detail-gene').innerHTML = 'gene ' + d.gene + geneCardsLink(d.gene);
  const peak = d.panel_D_trajectory.peak_layer;
  document.getElementById('detail-peak').textContent =
    'info peak ▸ L' + peak + ' (‖Z‖ ' + d.panel_D_trajectory.peak_value + ')';
  // non-coding transcript = ESM-2 zero-embedding by design, no real functional signal — every
  // panel below (GO prediction, axes, layer profile, 3D trajectory) is sigmoid(bias)-only noise
  // for this isoform ([[finding-brain672-noncoding-zero-embedding-flag-needed]]).
  const ncBadge = document.getElementById('detail-noncoding');
  if(ncBadge) ncBadge.classList.toggle('hidden', !d.is_noncoding);

  // siblings (gene isoforms) for axis comparison — fetch if not provided
  if(!siblings){
    const gp = await (await fetch('/api/gene/'+encodeURIComponent(d.gene))).json();
    siblings = gp.isoforms || [];
  }
  _siblings = siblings; _curId = d.isoform_id; _axes = d.panel_C_axes; _axesSummary = d.panel_C_summary || '';
  markIsoListCurrent(_curId);   // §02 list + mini-summary both key off _curId — keep the "currently
                                 // analyzed" row visually distinct as the user browses siblings.

  // I1 — top predicted function banner
  const top = d.panel_A_go[0];
  document.getElementById('topGoBanner').innerHTML = top ? `
    <div class="tgb-k">TOP PREDICTED FUNCTION</div>
    <div class="tgb-name">${top.name}</div>
    <div class="tgb-score"><span class="tnum">${top.score.toFixed(3)}</span>
      <span class="tgb-id">${top.go_id}</span></div>` : '';

  // I2 — Panel A: top-3 emphasized strip + colored bars
  const gA = d.panel_A_go;
  document.getElementById('topGo3').innerHTML = gA.slice(0,3).map((g,i)=>`
    <div class="tg3 r${i}"><span class="tg3-rank">${i+1}</span>
      <span class="tg3-name">${g.name}</span><span class="tg3-score tnum">${g.score.toFixed(3)}</span></div>`).join('');
  const go = gA.slice().reverse();  // horizontal bar bottom→top
  const nA = go.length;
  const barCols = go.map((_,i)=> (i>=nA-3) ? T.amber : T.signal);  // top-3 = amber
  Plotly.newPlot('panelA', [{
    type:'bar', orientation:'h', x: go.map(g=>g.score), y: go.map(g=>g.name),
    marker:{color:barCols, line:{width:0}}, hovertemplate:'%{y}<br>score %{x:.3f}<extra></extra>'
  }], inst({height:300, margin:{l:224,r:12,t:6,b:30},
       xaxis:{range:[0,1], title:{text:'prediction score',font:{size:10,color:T.dim}}, dtick:0.25}}), PLOT_CFG);

  // I4 — Panel C: axis selector + per-axis sibling comparison
  setupAxisSelector();

  // #7 — real sequence covariate panel (option B)
  _cov = d.covariates;
  renderCovariate();

  // E4 — domain architecture vs most-complete isoform
  renderDomain(d.isoform_id);

  // R4 — axis meaning
  document.getElementById('axisMeaning').innerHTML =
    `<div class="am-legend muted sm">p0–100 = this isoform's percentile rank among all brain-wide isoforms on each axis (p83 = higher than 83% of isoforms)</div>` +
    _axes.map(a=>{
    const hi=a.percentile>=66, lo=a.percentile<=33, col=hi?T.signal:(lo?T.trace:T.dim);
    return `<div class="am-row"><span class="am-ax" style="color:${col}">a${a.axis}</span>
      <span class="am-pct tnum" style="color:${col}" title="${a.percentile.toFixed(0)}th percentile among brain-wide isoforms">p${a.percentile.toFixed(0)}</span>
      <span class="am-txt">${a.meaning}</span></div>`;
  }).join('');

  // Panel D — combined layer profile (magnitude + discriminability), GO-driven
  _dTraj = d.panel_D_trajectory; _dPeak = peak; _dGlobalFisher = d.layer_fisher;
  setupLayerGo(gA);

  // Panel 3D — cohort overlay + centroid, R6 + option B
  _cur3d = _axes; _cur3dPeak = peak;
  setupCohortSelector(gA);
}

// ── I4: 8-axis fingerprint — radar (organic-chem descriptor style) ──
const AXIS_LABELS=['β-sheet/TM','LRR/Ig','Pro-turn','domain','helix-chg','length','KRAB-ZNF','acidic-hel'];
function setupAxisSelector(){
  const sel=document.getElementById('axisSel'); if(!sel) return;
  sel.innerHTML=[`<option value="radar">radar · all axes</option>`]
    .concat(AXIS_LABELS.map((l,i)=>`<option value="${i}">a${i} ${l}</option>`)).join('');
  sel.onchange=()=>renderAxisMap(sel.value);
  sel.value='radar'; renderAxisMap('radar');
}
function renderAxisMap(mode){
  const cur=_axes, sibs=_siblings;
  const theta=AXIS_LABELS.map((l,i)=>`a${i} ${l}`);
  if(mode==='radar'){
    const close=arr=>arr.concat([arr[0]]);            // 다각형 닫기
    const th=close(theta);
    const traces=[];
    // p50 reference ring
    traces.push({type:'scatterpolar', r:close(theta.map(()=>50)), theta:th, mode:'lines',
      line:{color:T.grid2,width:1,dash:'dot'}, hoverinfo:'skip', name:'population median (p50)'});
    // sibling fingerprints (thin grey)
    sibs.forEach(s=>{ if(s.isoform_id===_curId) return;
      traces.push({type:'scatterpolar', r:close(s.axis_pct), theta:th, mode:'lines',
        line:{color:'rgba(120,132,156,0.35)',width:1}, hoverinfo:'skip', showlegend:false});
    });
    // current isoform (filled)
    traces.push({type:'scatterpolar', r:close(cur.map(a=>a.percentile)), theta:th, mode:'lines+markers',
      fill:'toself', fillcolor:T.signal+'29', line:{color:T.signal,width:2},
      marker:{size:5,color:T.signal}, name:'this isoform',
      hovertemplate:'%{theta}<br>p%{r:.0f}<extra></extra>'});
    Plotly.newPlot('panelC', traces, {
      paper_bgcolor:'rgba(0,0,0,0)', height:340, margin:{l:40,r:40,t:24,b:24}, showlegend:false,
      font:{family:MONO,size:9,color:T.dim},
      polar:{bgcolor:T.ink800+'66',
        radialaxis:{range:[0,100], tickvals:[25,50,75,100], gridcolor:T.grid, angle:90,
          tickfont:{family:MONO,size:8,color:T.dim}, linecolor:T.grid},
        angularaxis:{gridcolor:T.grid, linecolor:T.grid, tickfont:{family:MONO,size:9,color:T.txt}}}}, PLOT_CFG);
    document.getElementById('axisMeaning').innerHTML =
      `<div class="am-analysis">${_axesSummary}</div>`+
      `<div class="am-analysis muted sm">Radial = population percentile (0–100). <b style="color:${T.signal}">Filled</b> = this isoform · thin grey = sibling isoforms · dotted ring = population median. Spokes far from center = axes where this isoform is extreme.</div>`;
  } else {
    // 단일 축 — 정렬된 lollipop(아이소폼별 행), 가시성↑ (#6)
    const a=+mode, curPct=cur[a].percentile;
    const rows=sibs.map(s=>({id:s.isoform_id, v:s.axis_pct[a], cur:s.isoform_id===_curId}))
      .sort((x,y)=>x.v-y.v);
    const rank=1+rows.filter(r=>r.v>curPct).length, n=rows.length;
    const yv=rows.map((_,i)=>i);
    const stems=[]; rows.forEach((r,i)=>{ stems.push({type:'line', x0:0,x1:r.v, y0:i,y1:i,
      line:{color:r.cur?T.signal:'rgba(120,132,156,0.4)', width:r.cur?2.5:1}}); });
    const dots={x:rows.map(r=>r.v), y:yv, mode:'markers', hovertemplate:'%{text}<br>p%{x:.0f}<extra></extra>',
      text:rows.map(r=>r.id), marker:{size:rows.map(r=>r.cur?13:8),
        color:rows.map(r=>r.cur?T.signal:'rgba(140,152,176,0.75)'),
        symbol:rows.map(r=>r.cur?'diamond':'circle'), line:{color:T.ink900,width:1}}, showlegend:false};
    const p50={type:'line', x0:50,x1:50,y0:-.5,y1:n-.5, line:{color:T.grid2,width:1,dash:'dot'}};
    Plotly.newPlot('panelC', [dots], inst({height:Math.max(200,26*n+60), margin:{l:120,r:16,t:16,b:38},
      shapes:stems.concat([p50]),
      xaxis:{range:[0,100], title:{text:'a'+a+' '+AXIS_LABELS[a]+' · population percentile (⋮ p50)',font:{size:10,color:T.dim}}, dtick:25},
      yaxis:{tickvals:yv, ticktext:rows.map(r=>r.id), tickfont:{family:MONO,size:9,
        color:T.dim}, range:[-.6,n-.4]}}), PLOT_CFG);
    document.getElementById('axisMeaning').innerHTML =
      `<div class="am-analysis"><b>a${a} ${AXIS_LABELS[a]}</b> — this isoform is at the `+
      `<b style="color:${T.signal}">◆ ${curPct.toFixed(0)}th percentile</b> (higher than ${curPct.toFixed(0)}% of brain-wide isoforms), `+
      `ranked <b>${rank}/${n}</b> among its ${n} sibling isoforms of this gene (sorted low→high). ${cur[a].meaning}</div>`;
  }
}

// ── #7: real sequence covariate panel (option B) ────────────────
let _cov=null;
function renderCovariate(){
  const host=document.getElementById('covPanel'); if(!host) return;
  if(!_cov || !_cov.available){
    host.innerHTML=`<p class="muted mono sm">covariates unavailable — ${(_cov&&_cov.reason)||'not built'}</p>`;
    return;
  }
  host.innerHTML=_cov.covariates.map((c,j)=>{
    const p=c.percentile;
    // sibling range from cov_pct[j]
    const sx=_siblings.map(s=>s.cov_pct&&s.cov_pct[j]).filter(v=>v!=null);
    const lo=sx.length?Math.min(...sx,p):p, hi=sx.length?Math.max(...sx,p):p;
    const col=p>=66?T.signal:(p<=33?T.trace:T.dim);
    const tier=p>=80?'high':(p>=60?'above median':(p<=20?'low':(p<=40?'below median':'typical')));
    // which pole the value sits toward, for plain-language read
    const pole=p>=60?c.hi:(p<=40?c.lo:'mid-range');
    return `<div class="cov-card">
      <div class="cov-top"><span class="cov-name">${c.label}</span>
        <span class="cov-p tnum" style="color:${col}">${c.value}<span class="cov-unit muted"> ${c.unit||''}</span></span></div>
      <div class="cov-track">
        <span class="cov-tick" style="left:50%"></span>
        <span class="cov-range" style="left:${lo}%;width:${Math.max(hi-lo,1)}%"></span>
        <span class="cov-cur" style="left:${p}%;background:${col}"></span></div>
      <div class="cov-scale"><span>${c.lo}</span><span>${c.hi}</span></div>
      <div class="cov-read">
        <span class="cov-tier" style="color:${col}">${tier}</span>
        <span class="muted">· <span title="${p.toFixed(0)}th percentile among brain-wide isoforms">${p.toFixed(0)}th percentile</span> · toward <b style="color:${col}">${pole}</b></span></div>
      ${c.note?`<div class="cov-note muted">${c.note}</div>`:''}
    </div>`;
  }).join('');
}

// ── E4: domain architecture — this isoform vs most-complete sibling ──
const DOMPAL=distinctPalette(10);   // Okabe-Ito, shared with mydata.js/bisect.js
// lost/gained: blue↔vermillion (DIVERGE_POS/NEG), not orange/green — colourblind-safe opposite pair.
const STCOL={lost:DIVERGE_NEG,gained:DIVERGE_POS,truncated:'#E69F00',intact:'#8A94A8'};
async function renderDomain(id){
  const host=document.getElementById('domPanel'); if(!host) return;
  host.innerHTML='<p class="muted mono sm">loading domain architecture…</p>';
  const d=await (await fetch('/api/domains/'+encodeURIComponent(id))).json();
  if(d.error || d.available===false){
    host.innerHTML=`<p class="muted mono sm">no domain annotation — ${(d&&d.reason)||'non-coding / novel / unmapped'} <span class="muted">(hmmscan Pfam coverage 35% of brain universe)</span></p>`;
    return;
  }
  // stable color per domain family (shared across both tracks)
  const fams=[...new Set([...(d.ref?d.ref.domains:[]),...d.this.domains].map(x=>x.name))];
  const famCol=Object.fromEntries(fams.map((f,i)=>[f,DOMPAL[i%DOMPAL.length]]));
  const maxLen=Math.max(d.ref?d.ref.len:1, d.this.len, 1);

  // ── summary line
  const nLost=d.changes.filter(c=>c.status==='lost').length,
        nTr=d.changes.filter(c=>c.status==='truncated').length,
        nGa=d.changes.filter(c=>c.status==='gained').length;
  const isRef=d.is_ref;
  let summary=isRef
    ? `<span class="dom-badge" style="border-color:${STCOL.intact};color:${STCOL.intact}">this is the most-complete isoform</span>`
    : [nLost&&`<span class="dom-badge" style="border-color:${STCOL.lost};color:${STCOL.lost}">${nLost} lost</span>`,
       nTr&&`<span class="dom-badge" style="border-color:${STCOL.truncated};color:${STCOL.truncated}">${nTr} truncated</span>`,
       nGa&&`<span class="dom-badge" style="border-color:${STCOL.gained};color:${STCOL.gained}">${nGa} gained</span>`]
      .filter(Boolean).join('') || `<span class="dom-badge" style="border-color:${STCOL.intact};color:${STCOL.intact}">architecture intact vs reference</span>`;

  // ── track figure (Plotly shapes): ref row (y=1), this row (y=0)
  const rows=[{iso:d.this.iso, dl:d.this.domains, y:0, lbl:'this'}];
  if(d.ref && !isRef) rows.unshift({iso:d.ref.iso, dl:d.ref.domains, y:1, lbl:'reference'});
  const shapes=[], anns=[];
  // baseline sequence bars
  rows.forEach(r=>{
    shapes.push({type:'rect', x0:0, x1:r.iso===d.this.iso?d.this.len:(d.ref?d.ref.len:1),
      y0:r.y-0.16, y1:r.y+0.16, fillcolor:'rgba(102,114,138,0.15)', line:{color:T.grid2,width:1}});
  });
  // domains
  rows.forEach(r=>{
    r.dl.forEach(dm=>{
      shapes.push({type:'rect', x0:dm.start, x1:dm.end, y0:r.y-0.22, y1:r.y+0.22,
        fillcolor:famCol[dm.name], line:{color:T.ink900,width:1}, opacity:0.9});
      if(dm.end-dm.start > maxLen*0.05)
        anns.push({x:(dm.start+dm.end)/2, y:r.y, text:dm.name, showarrow:false,
          font:{family:MONO,size:8,color:T.ink900}, xanchor:'center'});
    });
  });
  // ghost markers for lost domains on the 'this' row (show where they used to be)
  if(!isRef && d.ref){
    d.changes.filter(c=>c.status==='lost').forEach(c=>{
      (d.ref.domains.filter(dm=>dm.name===c.domain)).forEach(dm=>{
        shapes.push({type:'rect', x0:dm.start, x1:dm.end, y0:-0.22, y1:0.22,
          fillcolor:'rgba(217,118,74,0.10)', line:{color:STCOL.lost,width:1,dash:'dot'}});
      });
    });
  }
  // ── change table
  let tbl='';
  if(!isRef && d.changes.length){
    tbl=`<table class="dom-tbl"><thead><tr><th>domain</th><th>status</th><th>reference</th><th>this</th><th>retained</th><th></th></tr></thead><tbody>`;
    d.changes.slice().sort((a,b)=>a.retained-b.retained).forEach(c=>{
      const col=STCOL[c.status];
      const bar=`<span class="dom-ret"><span class="dom-ret-fill" style="width:${c.retained}%;background:${col}"></span></span>`;
      tbl+=`<tr><td class="dom-fam"><span class="dom-sw" style="background:${famCol[c.domain]||col}"></span>${c.domain}</td>`+
        `<td style="color:${col}">${c.status}</td>`+
        `<td class="tnum muted">${c.ref_len||'–'}</td><td class="tnum">${c.this_len||'–'}</td>`+
        `<td class="tnum" style="color:${col}">${c.retained}%</td><td style="width:80px">${bar}</td></tr>`;
    });
    tbl+='</tbody></table>';
  }
  host.innerHTML=`<div class="dom-summary">${summary}
      <span class="muted sm">reference = ${d.ref_id||'—'} (most complete of ${d.gene})</span>${geneCardsLink(d.gene)}</div>
    <div id="domFig"></div>${tbl}`;
  // plot into the freshly-injected #domFig node
  const yt=rows.map(r=>r.y), ytxt=rows.map(r=>r.lbl);
  Plotly.newPlot('domFig', [{x:[0],y:[0],mode:'markers',marker:{opacity:0},hoverinfo:'skip'}],
    inst({height:rows.length>1?150:96, margin:{l:80,r:14,t:8,b:34}, shapes, annotations:anns,
      xaxis:{range:[-maxLen*0.01,maxLen*1.02], title:{text:'residue position',font:{size:10,color:T.dim}}},
      yaxis:{tickvals:yt, ticktext:ytxt, range:[-0.6,rows.length>1?1.6:0.6],
        tickfont:{family:MONO,size:9,color:T.txt}, zeroline:false}}), PLOT_CFG);
}

// ── R6: 3D joint PCA trajectory — GO-matched isoforms + siblings + options ───
// "GO-matched isoforms" (was "cohort bundle" — renamed, the old name implied an external cohort;
// this is just the other isoforms in the same dataset that independently top-score for the
// selected GO term, excluding the isoform currently open) = go_cohort_trajectories() backend
// (variable/route names left as-is internally, only user-facing text renamed).
const SIBCOL=distinctPalette(8);   // Okabe-Ito, shared with mydata.js/bisect.js
// GO-matched isoforms — background-context lines, must stay visually secondary but still legible.
// Was a fixed rgba(150,160,180,0.10) (invisible on white), then `const GOMATCH_COLOR = T.dim` — but
// that captured T.dim's STRING value once at module load, so it never actually re-derived on a live
// theme switch (T mutates in place via refreshTheme(), a `const` string doesn't follow it); and even
// at its own theme's value, "dim" muted-text grey at 0.30 opacity on white is far below legible
// contrast (light --txt-dim #5B6880 on white --ink-800 #FFFFFF ≈ invisible at that opacity — the
// exact "bundle not visible in light theme" symptom). Computed fresh per render instead, using
// body-text contrast (T.txt, not the muted token) + a theme-conditional opacity boost, since white
// backgrounds eat faint colour much faster than the app's near-black one.
function _goMatchStyle(){
  const light = document.documentElement.dataset.theme === 'light';
  return {color: T.dim, opacity: light ? 0.28 : 0.16};
}
let _cur3d=null, _cur3dPeak=null, _cohort=null, _3dFisher=null;
const _3d={mode:'raw', grad:true, fisherColor:false, peaks:false, bundle:true, sibs:new Map()}; // sibs: id→traj

function setupCohortSelector(topGo){
  const sel=document.getElementById('cohortGo');
  if(sel){
    sel.innerHTML=[`<option value="">(no GO-matched isoforms · this isoform only)</option>`]
      .concat((topGo||[]).map(g=>`<option value="${g.go_id}">GO-matched: ${g.name} · ${g.score}</option>`)).join('');
    sel.onchange=async ()=>{
      if(!sel.value){ _cohort=null; _3dFisher=null; render3D(); return; }
      const [c, lf] = await Promise.all([
        fetch(`/api/go_cohort/${encodeURIComponent(sel.value)}?exclude=${encodeURIComponent(_curId)}&n=60`).then(r=>r.json()),
        fetch('/api/go_fisher/'+encodeURIComponent(sel.value)).then(r=>r.json()),
      ]);
      _cohort=c.error?null:c;
      _3dFisher=lf.error?null:lf;   // per-layer discriminability for the "colour by Fisher" toggle
      render3D();
    };
    if(topGo && topGo.length){ sel.value=topGo[0].go_id; sel.onchange(); } else { _cohort=null; _3dFisher=null; }
  }
  // sibling chips
  _3d.sibs.clear();
  const host=document.getElementById('p3dSibs');
  if(host){
    const others=_siblings.filter(s=>s.isoform_id!==_curId);
    host.innerHTML='<span class="sib-lbl">overlay siblings:</span>'+others.map((s,i)=>
      `<button class="sib-chip" data-id="${s.isoform_id}" data-ci="${i}">${s.isoform_id}</button>`).join('')
      || '';
  }
  render3D();
}
async function toggleSib(id, ci, btn){
  if(_3d.sibs.has(id)){ _3d.sibs.delete(id); btn.classList.remove('on'); btn.style.borderColor=''; }
  else {
    const d=await (await fetch('/api/iso_traj/'+encodeURIComponent(id))).json();
    if(d.error) return;
    _3d.sibs.set(id, {traj:d.traj, col:SIBCOL[ci%SIBCOL.length]});
    btn.classList.add('on'); btn.style.borderColor=SIBCOL[ci%SIBCOL.length]; btn.style.color=SIBCOL[ci%SIBCOL.length];
  }
  render3D();
}
function lineOf(traj, col, width, grad, colorArr){
  const xs=traj.map(p=>p[0]),ys=traj.map(p=>p[1]),zs=traj.map(p=>p[2]);
  const carr = colorArr || traj.map((_,i)=>i+1);
  return {type:'scatter3d', mode:'lines', x:xs,y:ys,z:zs,
    // gradient start was a fixed dark navy (#3a4a66) — fine on white, weak against the dark
    // theme's own panel; T.grid2 tracks whichever theme is active instead.
    line: grad ? {color:carr, colorscale:[[0,T.grid2],[1,col]], width} : {color:col, width},
    hoverinfo:'skip'};
}
function render3D(){
  if(!_cur3d) return;
  const key=(_3d.mode==='norm')?'trajectory_norm':'trajectory';
  const xs=_cur3d[0][key], ys=_cur3d[1][key], zs=_cur3d[2][key], n=xs.length, layers=xs.map((_,i)=>i+1);
  const useFisher = _3d.fisherColor && _3dFisher && _3dFisher.scores;
  const colorArr = useFisher ? _3dFisher.scores.map(v=>v==null?0:v) : layers;
  const cRange = useFisher ? [0.4, 0.9] : undefined;   // matches panelD's AUROC axis range
  const traces=[];
  // GO-matched isoforms — each drawn as its OWN trace (not one merged line) so the gradient/Fisher
  // colouring applies per-isoform like siblings do, not just as one flat colour for the whole set;
  // deliberately thin/low-opacity so they stay visually secondary to the main isoform + siblings.
  if(_3d.bundle && _cohort && _cohort.cohort && _cohort.cohort.length){
    const co=_cohort.cohort;
    const gm=_goMatchStyle();
    co.forEach(o => {
      const tr = lineOf(o.traj, gm.color, 1, _3d.grad || useFisher, colorArr);
      tr.opacity = gm.opacity;
      traces.push(tr);
    });
    const nL=co[0].traj.length, mx=[],my=[],mz=[];
    for(let l=0;l<nL;l++){ let a=0,b=0,c=0; co.forEach(o=>{a+=o.traj[l][0];b+=o.traj[l][1];c+=o.traj[l][2];});
      mx.push(a/co.length);my.push(b/co.length);mz.push(c/co.length); }
    // 고정 near-white 대신 T.txtHi(테마별 최대대비 텍스트색)에서 파생 — light 테마에서
    // 흰 배경 위 거의-흰색 선이 안 보이던 문제 수정(다크에서 텍스트가 밝듯, 라이트에선 어둡게).
    traces.push({type:'scatter3d', mode:'lines', x:mx,y:my,z:mz,
      line:{color:T.txtHi+'99', width:3, dash:'dash'}, name:'GO-matched mean',
      hovertemplate:'GO-matched mean · L%{text}<extra></extra>', text:mx.map((_,i)=>i+1)});
  }
  // selected siblings (#4)
  _3d.sibs.forEach((v,id)=> traces.push(lineOf(v.traj, v.col, 3, _3d.grad || useFisher, colorArr)));
  // this isoform — always on top, colored
  traces.push({type:'scatter3d', mode:'lines+markers', x:xs,y:ys,z:zs,
    line: (_3d.grad||useFisher) ? {color:colorArr, colorscale:[[0,T.trace],[1,T.signal]], width:7,
      cmin:cRange&&cRange[0], cmax:cRange&&cRange[1]} : {color:T.signal, width:7},
    marker:{size:3, color: (_3d.grad||useFisher)?colorArr:T.signal, colorscale:[[0,T.trace],[1,T.signal]],
      cmin:cRange&&cRange[0], cmax:cRange&&cRange[1],
      colorbar:(_3d.grad||useFisher)?{title:{text:useFisher?'discriminability (AUROC)':'layer',
        font:{family:MONO,size:9,color:T.dim}}, thickness:9, len:.5,
        tickfont:{family:MONO,size:8,color:T.dim}, outlinewidth:0}:undefined},
    hovertemplate:'L%{text}<br>PC1 %{x:.2f} PC2 %{y:.2f} PC3 %{z:.2f}<extra></extra>', text:layers});
  // L1 / L30 endpoints
  traces.push({type:'scatter3d', mode:'markers+text', x:[xs[0]],y:[ys[0]],z:[zs[0]],
    marker:{size:5,color:T.trace,symbol:'circle-open'}, text:['L1'], textposition:'top center',
    textfont:{family:MONO,size:9,color:T.trace}, hoverinfo:'skip'});
  traces.push({type:'scatter3d', mode:'markers+text', x:[xs[n-1]],y:[ys[n-1]],z:[zs[n-1]],
    marker:{size:6,color:T.signal}, text:['L30'], textposition:'top center',
    textfont:{family:MONO,size:9,color:T.signal}, hoverinfo:'skip'});
  // peak markers (#5)
  if(_3d.peaks){
    const ip=_cur3dPeak;
    traces.push({type:'scatter3d', mode:'markers+text', x:[xs[ip-1]],y:[ys[ip-1]],z:[zs[ip-1]],
      marker:{size:8,color:T.signal,symbol:'diamond'}, text:['info-peak L'+ip], textposition:'bottom center',
      textfont:{family:MONO,size:9,color:T.signal}, hoverinfo:'skip'});
    const fp=_dGlobalFisher&&_dGlobalFisher.peak_layer;
    if(fp) traces.push({type:'scatter3d', mode:'markers+text', x:[xs[fp-1]],y:[ys[fp-1]],z:[zs[fp-1]],
      marker:{size:9,color:T.trace,symbol:'cross'}, text:['discrim-peak L'+fp], textposition:'top center',
      textfont:{family:MONO,size:9,color:T.trace}, hoverinfo:'skip'});
  }
  // 고정 dark rgba 대신 현재 테마의 패널 배경(T.ink800)에서 파생 — light 에서도 옅은 slate 로 대응.
  const ax3={gridcolor:T.grid, zerolinecolor:T.grid2, showbackground:true,
    backgroundcolor:T.ink800+'80', color:T.dim, tickfont:{family:MONO,size:9,color:T.dim}};
  Plotly.newPlot('panel3D', traces, {
    paper_bgcolor:'rgba(0,0,0,0)', height:620, margin:{l:0,r:0,t:0,b:0}, showlegend:false,
    scene:{xaxis:{...ax3,title:'PC1'}, yaxis:{...ax3,title:'PC2'}, zaxis:{...ax3,title:'PC3'},
           camera:{eye:{x:1.6,y:1.6,z:1.2}}},
    font:{family:MONO,size:10,color:T.dim}}, PLOT_CFG);
}
// 3D control handlers
document.addEventListener('click', e=>{
  const b=e.target.closest('#pca3dToggle button');
  if(b){ document.querySelectorAll('#pca3dToggle button').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); _3d.mode=b.dataset.mode; render3D(); return; }
  const chip=e.target.closest('.sib-chip');
  if(chip){ toggleSib(chip.dataset.id, +chip.dataset.ci, chip); return; }
});
document.addEventListener('change', e=>{
  if(e.target.id==='optGrad'){ _3d.grad=e.target.checked; render3D(); }
  if(e.target.id==='optFisherColor'){ _3d.fisherColor=e.target.checked; render3D(); }
  if(e.target.id==='optPeaks'){ _3d.peaks=e.target.checked; render3D(); }
  if(e.target.id==='optBundle'){ _3d.bundle=e.target.checked; render3D(); }
});

// ── Panel D: combined layer profile (magnitude ‖Z‖ + discriminability) ──
let _dTraj=null, _dPeak=null, _dGlobalFisher=null;
function setupLayerGo(topGo){
  const sel=document.getElementById('layerGo'); if(!sel) return;
  sel.innerHTML=[`<option value="__domain__">discriminability: domain (global)</option>`]
    .concat((topGo||[]).map(g=>`<option value="${g.go_id}">discriminability: ${g.name} · ${g.score}</option>`)).join('');
  sel.onchange=async ()=>{
    const v=sel.value;
    if(v==='__domain__'){ renderLayerProfile(_dGlobalFisher); return; }
    const lf=await (await fetch('/api/go_fisher/'+encodeURIComponent(v))).json();
    renderLayerProfile(lf.error?null:lf);
  };
  if(topGo && topGo.length){ sel.value=topGo[0].go_id; sel.onchange(); }
  else renderLayerProfile(_dGlobalFisher);
}
function renderLayerProfile(fisherLf){
  const t=_dTraj, peak=_dPeak; if(!t) return;
  const L=Array.from({length:t.n_layers},(_,i)=>i+1);
  // 왼쪽축 — magnitude ‖Z‖ (this isoform) + population band
  const band={x:L.concat(L.slice().reverse()), y:t.ref_q75.concat(t.ref_q25.slice().reverse()),
    fill:'toself', fillcolor:'rgba(102,114,138,0.12)', line:{width:0}, hoverinfo:'skip', yaxis:'y', name:'pop 25–75%'};
  const med={x:L, y:t.ref_median, mode:'lines', line:{color:T.dim,dash:'dot',width:1}, yaxis:'y', name:'pop median'};
  const mag={x:L, y:t.magnitude, mode:'lines+markers', line:{color:T.signal,width:2.5}, marker:{size:4,color:T.signal},
    yaxis:'y', name:'magnitude ‖Z‖', hovertemplate:'L%{x} · ‖Z‖ %{y:.2f}<extra></extra>'};
  const magPk={x:[peak], y:[t.magnitude[peak-1]], mode:'markers', yaxis:'y',
    marker:{symbol:'triangle-up',size:13,color:T.signal,line:{color:T.ink900,width:1}}, name:'info-peak', hoverinfo:'skip'};
  const traces=[band,med,mag,magPk];
  // 오른쪽축 — discriminability (selected GO)
  if(fisherLf && fisherLf.scores){
    const s=fisherLf.scores, fpk=fisherLf.peak_layer;
    traces.push({x:L, y:s, mode:'lines+markers', line:{color:T.trace,width:2,dash:'solid'}, marker:{size:3,color:T.trace},
      yaxis:'y2', name:'discriminability', hovertemplate:'L%{x} · AUROC %{y:.3f}<extra></extra>'});
    traces.push({x:[fpk], y:[s[fpk-1]], mode:'markers', yaxis:'y2',
      marker:{symbol:'star',size:14,color:T.trace}, name:'discrim-peak', hoverinfo:'skip'});
  }
  Plotly.newPlot('panelD', traces, inst({height:340, margin:{l:54,r:54,t:10,b:40}, hovermode:'x unified',
    xaxis:{title:{text:'ESM-2 layer (1–30)',font:{size:10,color:T.dim}}, dtick:5,
           showspikes:true, spikemode:'across', spikecolor:T.grid2, spikethickness:1},
    yaxis:{title:{text:'magnitude ‖Z‖',font:{size:10,color:T.signal}}, tickfont:{family:MONO,size:10,color:T.signal},
           gridcolor:T.grid},
    yaxis2:{title:{text:'discriminability (AUROC)',font:{size:10,color:T.trace}}, tickfont:{family:MONO,size:10,color:T.trace},
            overlaying:'y', side:'right', showgrid:false, range:[0.4,0.9]},
    legend:{orientation:'h', y:1.16, font:{size:9,color:T.dim}}}), PLOT_CFG);
}

// ── I3: transcripts × GO heatmap + module click ────────────────
let _hm=null, _openMod=null;
function renderHeatmap(hm){
  const host=document.getElementById('geneHeatmap'); if(!host||!hm) return;
  _hm=hm; _openMod=null;
  // module color strip (just above cells) + label annotations on a separate row higher up
  // strip sits at y 1.01–1.035; labels at y 1.10 with a connector — clear gap so text ≠ figure
  // 라벨 배경(bgcolor) 은 현재 테마의 패널 배경(T.ink800)에 알파를 더해 파생 — light 에서도
  // 흰 배경 위에 진한 dark chip 이 뜨는 부조화를 피한다(8-digit hex alpha).
  const labelBg = T.ink800 + 'D9';
  const shapes=(hm.modules||[]).map((m,j)=>({type:'rect', xref:'x', yref:'paper',
    x0:j-0.5, x1:j+0.5, y0:1.012, y1:1.038, fillcolor:modColor(m), line:{width:0}}));
  const anns=[];
  [...new Set(hm.modules)].forEach(m=>{
    const cols=hm.modules.map((mm,j)=>mm===m?j:-1).filter(j=>j>=0);
    const mid=(cols[0]+cols[cols.length-1])/2;
    // module span bracket under the label
    shapes.push({type:'line', xref:'x', yref:'paper', x0:cols[0]-0.42, x1:cols[cols.length-1]+0.42,
      y0:1.052, y1:1.052, line:{color:modColor(m), width:1.5}});
    anns.push({x:mid, xref:'x', y:1.135, yref:'paper', showarrow:false,
      text:'▸ '+(hm.module_labels?hm.module_labels[m]:('module '+(m+1))),
      font:{family:MONO,size:9.5,color:modColor(m)}, xanchor:'center', align:'center',
      bgcolor:labelBg, borderpad:2});
  });
  Plotly.newPlot('geneHeatmap', [{
    type:'heatmap', z:hm.matrix, x:hm.go_names, y:hm.isoform_ids,
    // Viridis — same reasoning as drawQcHeatmap above (shared sequential-gradient convention +
    // theme-independent monotonicity).
    colorscale:VIRIDIS, zmin:0, zmax:1,
    colorbar:{title:{text:'score',font:{size:9,color:T.dim}}, tickfont:{family:MONO,size:9,color:T.dim},
      thickness:10, len:.7, outlinewidth:0},
    hovertemplate:'%{y}<br>%{x}<br>score %{z:.3f}<extra></extra>'
  }], inst({height:Math.max(300, 34*hm.isoform_ids.length+210), margin:{l:150,r:60,t:78,b:150}, shapes, annotations:anns,
    xaxis:{tickangle:-40, tickfont:{family:MONO,size:9,color:T.dim}, side:'bottom', automargin:true},
    yaxis:{tickfont:{family:MONO,size:9,color:T.dim}, autorange:'reversed', automargin:true}}), PLOT_CFG);
  document.getElementById('geneHeatmap').on('plotly_click', ev=>showModule(ev.points[0].pointIndex[1]));
}
function showModule(colIdx){
  const hm=_hm; if(!hm) return;
  const mod=hm.modules[colIdx];
  const det=document.getElementById('moduleDetail');
  if(_openMod===mod){ det.innerHTML=''; _openMod=null; return; }   // 같은 모듈 재클릭 → 접기 (#2)
  _openMod=mod;
  const cols=hm.modules.map((m,j)=>m===mod?j:-1).filter(j=>j>=0);
  const label=hm.module_labels?hm.module_labels[mod]:('module '+(mod+1));
  let html=`<div class="mod-head"><span class="mod-dot" style="background:${modColor(mod)}"></span>
    <b>Module ${mod+1}: ${label}</b> · ${cols.length} GO term(s)
    <button class="mod-close" onclick="closeModule()">✕ collapse</button></div>`;
  html+=`<table class="mod-tbl"><thead><tr><th>GO term</th>${hm.isoform_ids.map(i=>`<th>${i}</th>`).join('')}</tr></thead><tbody>`;
  cols.forEach(j=>{
    html+=`<tr><td class="mod-go">${hm.go_names[j]}</td>`+
      hm.isoform_ids.map((_,i)=>{const v=hm.matrix[i][j];
        return `<td class="tnum" style="color:${v>=0.5?T.signal:(v>=0.3?T.amber:T.dim)}">${v.toFixed(2)}</td>`;}).join('')+`</tr>`;
  });
  html+=`</tbody></table>`;
  det.innerHTML=html;
}
function closeModule(){ document.getElementById('moduleDetail').innerHTML=''; _openMod=null; }

// Quick Report 는 기본 펼침(loadQuickCases() 는 위쪽 초기화 블록에서 즉시 호출) —
// 여기선 검색 결과(있다면) 로드만 담당, renderDetail() 이 뜨면 Quick Report 는 자동으로 접힌다.
document.addEventListener('DOMContentLoaded', loadResult);
// 테마 전환 시 현재 뷰의 모든 Plotly 차트를 새 팔레트로 다시 그린다(theme.js 가 호출).
// Quick Report 는 한 번이라도 펼쳐서 로드된 경우에만 다시 그린다(접힌 채로 불필요한 fetch 방지).
window.__rerenderCharts = () => { loadResult(); if(_qcLoaded) loadQuickCases(); };
