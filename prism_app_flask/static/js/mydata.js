// 내 데이터 분석 대시보드 — tissue 요약 렌더 (instrument.js 테마 공유).
'use strict';

const tabs = document.getElementById('tissue-tabs');
if(tabs){
  tabs.addEventListener('click', e => {
    const b = e.target.closest('.ttab'); if(!b) return;
    tabs.querySelectorAll('.ttab').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');
    load(b.dataset.tissue);
  });
}

function statCard(k, v, sub){
  return `<div class="stat"><div class="stat-k">${k}</div>`+
         `<div class="stat-v tnum" data-count="${v}">${v}</div>`+
         (sub?`<div class="stat-sub muted">${sub}</div>`:'')+`</div>`;
}

async function load(tissue){
  const dash = document.getElementById('dash');
  document.getElementById('ds-readout').innerHTML = `<span class="muted mono">▸ measuring ${tissue}…</span>`;
  const s = await (await fetch('/api/summary/'+encodeURIComponent(tissue))).json();
  if(s.error){ document.getElementById('ds-readout').innerHTML = `<span class="err mono">✕ ${s.error}</span>`; return; }

  document.getElementById('ds-readout').innerHTML =
    `<span><span class="k">TISSUE</span> <b>${s.tissue}</b></span>`+
    `<span><span class="k">THRESHOLD</span> <b>${s.threshold}</b></span>`+
    `<span><span class="k">GO TERMS</span> <b>${s.n_go}</b></span>`+
    (s.dtu?`<span><span class="k">DTU</span> <b>${s.dtu.n_sig.toLocaleString()}</b> sig / ${s.dtu.conditions.length} cond</span>`:'');

  // stat cards
  const types = Object.entries(s.types||{}).map(([k,v])=>`${k} ${v.toLocaleString()}`).join(' · ');
  document.getElementById('stat-row').innerHTML =
    statCard('ISOFORMS', s.n_isoforms, s.n_genes? s.n_genes.toLocaleString()+' genes':'') +
    statCard('HIGH-CONF', s.n_highconf, s.highconf_pct+'% (max>thr)') +
    statCard('MEDIAN maxGO', s.median_max_score, 'mean n-high '+s.mean_n_high) +
    statCard('ISO TYPES', (s.types.known||0), types);
  document.querySelectorAll('.stat-v[data-count]').forEach(el=>{
    const to=parseFloat(el.dataset.count); countUp(el, to, Number.isInteger(to)?0:3);
    if(to>=1000) el.dataset.big=1;
  });

  // GO term score distribution (by gene) — which GO term the MLP was trained on, and how each
  // gene's isoforms score for it, matters more to a researcher than a generic max-score histogram.
  await loadGoTermSelector(tissue);

  // isoform triage — ranked list (supersedes the old S1-S4 scenario donut and the
  // "Functional-switch candidates (S1)" table, which ranked by max_score alone — the exact
  // magnitude-first approach this ranked list was built to replace)
  loadTriageRanked(tissue);

  // existing-annotation space vs PRISM-predicted space, as a 2-set Venn (area ∝ (gene,GO-term)
  // pair count) — replaces the old isoform-level "Case A/B/C" breakdown after feedback that the
  // naming wasn't intuitive and the gap-histogram wasn't the most useful cut of this data.
  // brain_672 only — muscle gene ids are versioned ENSG* and don't match the symbol-keyed
  // annotation lookup, so the backend returns null there rather than emit an artifact.
  _ngbData = s.novel_go_breakdown;
  renderNovelGoVenn();

  // structural-type pie (score>0.4, app-wide high-confidence threshold — fixed, not the QC threshold above)
  const th = s.type_high_conf;
  document.getElementById('typeThrLabel').textContent = th ? th.threshold : '0.4';
  if(th){
    const tlabels=['Known','NIC','NNIC'], tvals=[th.known, th.nic, th.nnic];
    // fixed Okabe-Ito hex (not T.trace/T.ok/T.signal) — matches the UMAP legend's isoform_type
    // colours exactly (dataset_summary.umap_points), which are also fixed hex; before this they
    // were two independently-tuned colour sets for the same 3 categories (an inconsistency).
    const tcols=[ISOTYPE_PAL.known, ISOTYPE_PAL.nic, ISOTYPE_PAL.nnic];
    Plotly.newPlot('typePie',[{type:'pie',hole:.35,labels:tlabels,values:tvals,
      marker:{colors:tcols,line:{color:T.ink900,width:2}},textinfo:'percent+label',
      textfont:{family:MONO,size:10,color:'#fff'},
      hovertemplate:'%{label}<br>%{value} isoforms (%{percent})<extra></extra>'}],
      inst({height:250,margin:{l:6,r:6,t:6,b:6},showlegend:false}),PLOT_CFG);
    document.getElementById('type-cap').textContent =
      `score>${th.threshold} (app default high-confidence threshold) · `+
      `Known ${th.known.toLocaleString()}/${th.total_known.toLocaleString()} · `+
      `NIC ${th.nic.toLocaleString()}/${th.total_nic.toLocaleString()} · `+
      `NNIC ${th.nnic.toLocaleString()}/${th.total_nnic.toLocaleString()}`;
    // per-type high-conf RATE, alongside the pie's per-type SHARE-of-high-conf — a different cut
    // of the same th payload (fills the panel's second column instead of leaving it blank).
    const ttotals = [th.total_known, th.total_nic, th.total_nnic];
    document.getElementById('typeBreakdownTbody').innerHTML = tlabels.map((lbl, i) => {
      const hc = tvals[i], tot = ttotals[i], pct = tot ? (100 * hc / tot).toFixed(1) : '0.0';
      return `<tr><td>${lbl}</td><td class="tnum">${hc.toLocaleString()}</td>`+
             `<td class="tnum">${tot.toLocaleString()}</td><td class="tnum">${pct}%</td></tr>`;
    }).join('');
  } else {
    document.getElementById('typePie').innerHTML = '<p class="muted mono sm">no structural-type data for this tissue</p>';
    document.getElementById('typeBreakdownTbody').innerHTML = '';
  }

  // landscape bar
  const ls=s.landscape.slice().reverse();
  Plotly.newPlot('landBar',[{type:'bar',orientation:'h',x:ls.map(l=>l.coverage),y:ls.map(l=>l.name),
    marker:{color:T.trace},hovertemplate:'%{y}<br>%{x} isoform (score>thr)<extra></extra>'}],
    inst({height:360,margin:{l:280,r:12,t:6,b:30},
      xaxis:{title:{text:'coverage · # high-confidence isoforms',font:{size:10,color:T.dim}}}}),PLOT_CFG);

  dash.dataset.active = tissue;
}

// ── Isoform triage — ranked list (docs/mydata_triage_design.md §3) ──────────
// Supersedes the S1-S4 (DTU × novel-GO) donut. Top-line badges = validated-consequence evidence
// only (domain-family identity, ORF change) + the novel-GO prediction — these are the sort keys
// (§3.2/3.3). Structural descriptor (disorder Δ etc.) is real, measured evidence but NOT a
// function claim — it lives in the expandable row, never top-line (§3.3, 2nd devils-advocate).
function triageEvidenceBadges(row){
  const parts = [];
  (row.domain_changes||[]).forEach(c => {
    parts.push(`<span class="badge match" title="vs canonical ${row.canonical_id} — family identity, not magnitude">✓ domain ${c.status} (${escapeHtml(c.domain)})</span>`);
  });
  if(row.orf_change){
    parts.push(`<span class="badge match" title="coding-status differs from canonical ${row.canonical_id}">✓ ORF ${row.orf_change}</span>`);
  }
  if(row.novel_go){
    const ng = row.novel_go;
    parts.push(`<span class="badge${ng.hiconf?' match':''}" title="PRISM prediction, gene not annotated with this GO term — see literature-audit caveat above, not confirmed function">novel-GO: ${escapeHtml(ng.go_name)} (${ng.score.toFixed(2)})</span>`);
  }
  if(!parts.length && row.descriptor){
    parts.push('<span class="badge" title="measured structural difference — describes what differs, not which function changes">△ structural descriptor</span>');
  }
  if(!parts.length && row.no_canonical_ref){
    parts.push('<span class="badge caveat" title="no CDS canonical for this gene, or canonical not indexed — domain/ORF comparison unavailable, not silently dropped">no protein reference</span>');
  }
  if(!parts.length){
    parts.push('<span class="badge">no detected change</span>');
  }
  return parts.join(' ');
}

const TRIAGE_STRUCT_LABEL = {known:'FSM', nic:'NIC', nnic:'NNIC'};

function renderTriageRow(row, i){
  const hasExpand = !!row.descriptor || (row.no_canonical_ref && !row.novel_go);
  const detailId = 'triageDetail'+i;
  let detail = '';
  if(row.descriptor){
    const dv = row.descriptor.disorder_delta;
    detail += `disorder Δ vs canonical (${row.canonical_id}): <b>${dv>0?'+':''}${dv}</b> — measured difference, not a function claim (see individual-analysis page for the full 8-axis coordinate).`;
  }
  if(row.no_canonical_ref){
    detail += (detail?' ':'') + 'no protein-coding canonical for this gene (no_CDS, or canonical not indexed) — domain/ORF comparison unavailable.';
  }
  return `
  <tr class="triage-row"${hasExpand?` onclick="toggleTriageDetail(${i})" style="cursor:pointer"`:''}>
    <td>${hasExpand?`<span id="${detailId}-arrow">▸</span>`:''}</td>
    <td>${row.gene}${geneCardsLink(row.gene)}</td>
    <td><a href="/gene/${encodeURIComponent(row.isoform_id)}" onclick="event.stopPropagation()" title="probe in individual analysis ▸">${row.isoform_id}</a></td>
    <td class="muted">${TRIAGE_STRUCT_LABEL[row.structural_type]||row.structural_type||''}</td>
    <td>${triageEvidenceBadges(row)}</td>
    <td class="tnum">${row.max_score.toFixed(3)}</td>
  </tr>${hasExpand?`
  <tr class="triage-detail hidden" id="${detailId}"><td></td><td colspan="5" class="muted sm">${detail}</td></tr>`:''}`;
}

function toggleTriageDetail(i){
  const row = document.getElementById('triageDetail'+i);
  const arrow = document.getElementById('triageDetail'+i+'-arrow');
  if(!row) return;
  const nowHidden = row.classList.toggle('hidden');
  if(arrow) arrow.textContent = nowHidden ? '▸' : '▾';
}

async function loadTriageRanked(tissue){
  const tbody = document.getElementById('triageTbody');
  const more = document.getElementById('triageMore');
  const countsEl = document.getElementById('triageKindCounts');
  if(!tbody) return;
  tbody.innerHTML = '<tr><td colspan="6" class="muted mono sm">▸ loading…</td></tr>';
  const consequence = document.getElementById('triageConsequence').value;
  const structType = document.getElementById('triageStructTypeChips').dataset.value || '';
  const sort = document.getElementById('triageSort').value;
  const params = new URLSearchParams({sort});
  if(consequence) params.set('consequence', consequence);
  if(structType) params.set('structural_type', structType);
  const r = await fetch('/api/summary/'+encodeURIComponent(tissue)+'/triage_ranked?'+params.toString());
  const d = await r.json();
  if(d.error){ tbody.innerHTML = `<tr><td colspan="6" class="err mono sm">✕ ${d.error}</td></tr>`; more.textContent=''; countsEl.textContent=''; return; }
  tbody.innerHTML = d.rows.map((row,i)=>renderTriageRow(row,i)).join('')
    || '<tr><td colspan="6" class="muted mono sm">no isoforms match this filter</td></tr>';
  const kc = d.kind_counts;
  // "total" is every alt isoform vs its canonical (kind='none' included) -- NOT all "changed": say
  // so explicitly and break out how many actually carry detected evidence, so the count next to
  // "no detected change" rows in the table never contradicts this line (was previously worded
  // "changed isoforms" even when e.g. sort=score surfaces mostly kind='none' rows in the visible page).
  const nWithChange = d.total - kc.none;
  more.textContent = d.truncated
    ? `showing top ${d.rows.length.toLocaleString()} of ${d.total.toLocaleString()} alt isoforms vs canonical `+
      `(${nWithChange.toLocaleString()} with detected change), sorted by ${d.sort}`
    : `${d.total.toLocaleString()} alt isoform(s) vs canonical (${nWithChange.toLocaleString()} with detected change)`;
  countsEl.textContent = `domain ${kc.domain.toLocaleString()} · ORF ${kc.orf.toLocaleString()} · `+
    `IDR/compositional ${kc.idr.toLocaleString()} · novel-GO only ${kc.novel_go.toLocaleString()} · `+
    `no detected change ${kc.none.toLocaleString()}`;
}

['triageConsequence','triageSort'].forEach(id=>{
  const el = document.getElementById(id);
  if(el) el.addEventListener('change', () =>
    loadTriageRanked(document.getElementById('dash').dataset.active || 'brain_672'));
});

// structural-type filter chips (single-select, radio-like — reuses .sib-chip toggle styling)
const triageStructChipsEl = document.getElementById('triageStructTypeChips');
if(triageStructChipsEl){
  triageStructChipsEl.addEventListener('click', e => {
    const btn = e.target.closest('.sib-chip'); if(!btn) return;
    triageStructChipsEl.querySelectorAll('.sib-chip').forEach(b=>b.classList.remove('on'));
    btn.classList.add('on');
    triageStructChipsEl.dataset.value = btn.dataset.value || '';
    loadTriageRanked(document.getElementById('dash').dataset.active || 'brain_672');
  });
}

// -- GO term score distribution: raw per-isoform spikes, gene-highlightable --
// Pick a GO term the MLP was trained on for this tissue -> every isoform's raw score for that
// column is drawn as a vertical spike (one real data point, no per-gene smoothing/KDE). Hover a
// spike to see which gene it belongs to and highlight every other isoform of that same gene;
// typing a gene name in the search box pins the same highlight so it survives mouse-out. The
// bold curve is a separate, gene-independent statistic (population KDE, peak-normalized) drawn
// as a plain 'scatter' (SVG) trace, which Plotly always layers above 'scattergl' (WebGL) traces
// regardless of add order -- so it stays on top of the raw spikes and any highlight.
//
// Beyond hover/search, two more highlight mechanisms layer on top (all coexist as separate
// overlay traces, rebuilt together by redrawGoDistOverlays() whenever any of their state changes):
//   - criterion chips (GENE_HL_PAL) -- "color on/off" toggles for 4 gene sets (top-score isoform's
//     gene / top-variance genes / single-isoform high-score genes / top-mean genes), each its own
//     colour + a gene list panel below the plot.
//   - click-to-lock -- clicking a raw spike OR a gene button inside a criterion panel pins a
//     single gene's highlight (teal) + an isoform-detail panel that survives mouse movement,
//     until: (a) a different gene is clicked, (b) its collapse button is pressed, or (c) — only
//     when the lock came from a criterion panel — that criterion's chip is switched off. (b)/(c)
//     going the OTHER way (collapse/other-gene-click while locked-via-criterion) also auto-flips
//     that criterion's chip off, but only when the lock actually moves to a DIFFERENT criterion
//     (or the plot) -- clicking another gene row within the SAME still-open criterion panel just
//     re-points the lock, it doesn't flicker that criterion's chip off then back on.
let _goDistTissue = null, _goTerms = [];

const GENE_HL_KEYS = ['top_score', 'top_variance', 'single_iso_high', 'top_mean'];
const GENE_HL_PAL = {
  top_score:       {color:'#CC79A7', label:'Top-score isoform’s gene'},        // reddish purple
  top_variance:    {color:'#56B4E9', label:'Top-variance genes (10)'},         // sky blue
  single_iso_high: {color:'#E69F00', label:'Single-isoform high-score genes'}, // orange
  top_mean:        {color:'#0072B2', label:'Top-mean genes (10)'},             // blue
};
const GENE_LOCK_COLOR = '#009E73';   // bluish green — distinct "locked" identity

// novel-GO Venn — "existing annotation" circle = DIVERGE_POS(blue), "PRISM prediction" circle =
// DIVERGE_NEG(vermillion), same warm/cool split used everywhere else in the app for
// known-baseline-vs-novel-signal. The two toggleable sub-rings inside the "new" lobe get their
// own distinct hues (not reused from the two main circles, so they read as a layer on top).
const NOVELGO_PAL = {known: DIVERGE_POS, predicted: DIVERGE_NEG, hiconf: '#E69F00', beats: '#CC79A7'};
let _ngbData = null;   // current /api/summary novel_go_breakdown payload — toggles re-render from
                       // this without refetching.

// isoform structural type — must match dataset_summary.umap_points' isoform_type color_meta
// exactly (backend emits the same 3 fixed hex values) so the UMAP legend and this page's type-pie
// always agree on which colour is Known/NIC/NNIC.
const ISOTYPE_PAL = {known: '#0072B2', nic: '#009E73', nnic: '#D55E00'};

// region labels/order shared by the bar plot and the Venn — single source of truth so both stay
// visually consistent (same colour per region, same reading order).
const NOVELGO_REGION_META = {
  known_only: {label: 'missed (annotated, not predicted)', color: NOVELGO_PAL.known},
  overlap:    {label: 'recovered (matches annotation)',    color: '#8A94A8'},
  new:        {label: 'new (beyond annotation)',           color: NOVELGO_PAL.predicted},
  new_hiconf: {label: 'new · top-ranked (score)',           color: NOVELGO_PAL.hiconf},
  new_beats:  {label: "new · exceeds gene's own known fn.", color: NOVELGO_PAL.beats},
};

// § Existing annotation vs PRISM prediction — left: bar plot of the 5 region sizes (full
// distribution at a glance, including the two "new" sub-slices) · right: area-proportional 2-set
// Venn (venn.js, D3) for existing-annotation vs PRISM-prediction, restricted to (gene, GO term)
// pairs on the 672-term panel. Both are clickable — a region opens the same isoform-list panel
// below (§ backend: dataset_summary.novel_go_region_isoforms, lazy-fetched per click, not part of
// the /api/summary payload since it can be thousands of rows).
function renderNovelGoVenn(){
  const ngb = _ngbData;
  const vennHost = document.getElementById('novelGoVenn'), barHost = document.getElementById('novelGoBar');
  if(!ngb || ngb.error){
    vennHost.innerHTML = '<p class="muted mono sm">no gene-symbol-matched annotation data for this tissue</p>';
    barHost.innerHTML = '';
    document.getElementById('novel-go-cap').textContent = '';
    document.getElementById('novelGoBPanel').classList.add('hidden');
    document.getElementById('novelGoRegionPanel').classList.add('hidden');
    return;
  }
  const c = ngb.counts;
  document.getElementById('novelGoHiconfThr').textContent = ngb.hiconf_threshold.toFixed(2);

  renderNovelGoBar(c);
  renderNovelGoVennDiagram(c);

  document.getElementById('novel-go-cap').textContent =
    `${ngb.n_genes_annotated_on_panel.toLocaleString()} genes have existing panel annotation `+
    `(${ngb.n_genes_annotation_gap.toLocaleString()} more have none at all — their predictions `+
    `are entirely "new" by definition) · recovered ${c.overlap.toLocaleString()} · `+
    `missed ${c.known_only.toLocaleString()} · new ${c.new.toLocaleString()} `+
    `(${c.new_hiconf.toLocaleString()} top-ranked, ${c.new_beats.toLocaleString()} exceed `+
    `the gene's own known function) — counts are (gene, GO term) pairs, threshold ${ngb.threshold}.`;

  const bPanel = document.getElementById('novelGoBPanel');
  if(ngb.top_beats_examples && ngb.top_beats_examples.length){
    bPanel.classList.remove('hidden');
    document.getElementById('novelgob-tbody').innerHTML = ngb.top_beats_examples.map(r=>`
      <tr onclick="location.href='/gene/'+encodeURIComponent('${r.isoform_id}')" title="probe in individual analysis ▸">
      <td>${r.gene}${geneCardsLink(r.gene)}</td><td>${r.isoform_id} <span class="muted">▸</span></td>
      <td class="muted">${escapeHtml(r.go_name)} (${r.novel_go})</td>
      <td>${r.novel_score.toFixed(3)}</td><td>${r.own_best_score.toFixed(3)}</td>
      <td>${(r.novel_score-r.own_best_score).toFixed(3)}</td></tr>`).join('');
  } else {
    bPanel.classList.add('hidden');
  }
}

// left panel — full distribution across all 5 regions, including the two "new" sub-slices
// (new_hiconf/new_beats are subsets of "new", drawn lighter/indented so the chart still reads as
// "3 partition bars + 2 zoom-in bars" rather than implying 5 independent, summable categories).
function renderNovelGoBar(c){
  const order = ['known_only', 'overlap', 'new', 'new_hiconf', 'new_beats'];
  const vals = order.map(k => c[k]);
  const labels = order.map(k => NOVELGO_REGION_META[k].label);
  const colors = order.map(k => NOVELGO_REGION_META[k].color);
  const opacities = order.map(k => (k === 'new_hiconf' || k === 'new_beats') ? .55 : .85);
  Plotly.newPlot('novelGoBar', [{
    type: 'bar', orientation: 'h', x: vals, y: labels, marker: {color: colors, opacity: opacities},
    text: vals.map(v => v.toLocaleString()), textposition: 'outside',
    textfont: {family: MONO, size: 10, color: T.txt},
    hovertemplate: '%{y}<br>%{x} (gene, GO term) pairs<extra></extra>',
  }], inst({height: 340, margin: {l: 190, r: 40, t: 10, b: 30},
    xaxis: {title: {text: '(gene, GO term) pairs', font: {size: 10, color: T.dim}}},
    yaxis: {tickfont: {family: MONO, size: 10, color: T.dim}, autorange: 'reversed'}}), PLOT_CFG);
  const gd = document.getElementById('novelGoBar');
  gd.removeAllListeners && gd.removeAllListeners('plotly_click');
  gd.on('plotly_click', ev => {
    const p = ev.points && ev.points[0]; if(!p) return;
    loadNovelGoRegion(order[p.pointIndex]);
  });
}

// right panel — venn.js (D3): true area-proportional 2-set layout (replaces the old hand-drawn
// Plotly-shape circles, which weren't area-proportional inside the "new"/"missed" wedges and had
// no per-region interactivity). Click a region (circle-only, overlap wedge, or B-only wedge) to
// open its isoform list below via loadNovelGoRegion().
function renderNovelGoVennDiagram(c){
  const host = document.getElementById('novelGoVenn');
  if(typeof venn === 'undefined' || typeof d3 === 'undefined'){
    host.innerHTML = '<p class="err mono sm">✕ venn.js/d3 failed to load (CDN unreachable?)</p>';
    return;
  }
  const knownTotal = c.known_only + c.overlap, predTotal = c.new + c.overlap;
  host.innerHTML = '';
  if(knownTotal <= 0 || predTotal <= 0){
    host.innerHTML = '<p class="muted mono sm">not enough data for a two-set diagram</p>';
    return;
  }
  const sets = [
    {sets: ['A'], size: knownTotal, label: 'existing annotation'},
    {sets: ['B'], size: predTotal, label: 'PRISM prediction'},
    {sets: ['A', 'B'], size: c.overlap},
  ];
  const chart = venn.VennDiagram().width(Math.min(460, host.clientWidth || 460)).height(340);
  const sel = d3.select('#novelGoVenn').datum(sets).call(chart);

  // colour per set (existing app convention: known=blue, predicted=vermillion) + custom label text
  // (region name + count) instead of venn.js's default label-only text.
  const regionOf = d => d.sets.length === 2 ? 'overlap' : (d.sets[0] === 'A' ? 'known_only' : 'new');
  const countOf = d => d.sets.length === 2 ? c.overlap : (d.sets[0] === 'A' ? c.known_only : c.new);
  sel.selectAll('.venn-circle path')
    .style('fill', d => d.sets[0] === 'A' ? NOVELGO_PAL.known : NOVELGO_PAL.predicted)
    .style('stroke', d => d.sets[0] === 'A' ? NOVELGO_PAL.known : NOVELGO_PAL.predicted);
  sel.selectAll('.venn-area text')
    .style('fill', T.txt).style('font-size', '10px')
    .text(d => `${d.sets.length === 2 ? 'recovered' : (d.sets[0] === 'A' ? 'missed' : 'new')} — ${countOf(d).toLocaleString()}`);
  // d3@7 click listeners receive (event, d), not (d) — passing the event where a
  // datum is expected made regionOf() throw on d.sets.length (venn regions never opened).
  sel.selectAll('.venn-area').on('click', (event, d) => loadNovelGoRegion(regionOf(d)));
}

let _novelGoRegionTissue = () => (document.getElementById('dash').dataset.active || 'brain_672');

// Manual literature audit (WebSearch + UniProt REST, 2026-07-29 — see memory
// finding-novel-go-literature-verification-hiconf.md / finding-novel-go-contradiction-rate-by-score-bin.md)
// of 81 sampled "new" (gene, GO term) pairs across score 0.5–1.0. These genes' predicted GO theme
// directly contradicted the gene's UniProt/PubMed-documented function (not just "no evidence
// found" — an active mismatch, e.g. paralog/compartment/complex-location confusion). This is a
// spot-check sample, not full dataset coverage — absence from this set is not a validation.
const NOVELGO_LIT_CONTRADICTED = new Set([
  'AACS','ELOVL4','CHAMP1','B4GALT6','ST8SIA1','ST8SIA5','GABRB2','GABRG2','H3-3A','MRPL15',
  'RPS11','UBE2V2','RPL23AP7','MCM3AP','APOLD1','ATP1A4','ATP8A2','ATR','BLOC1S2','ACAT2',
  'AP2S1','ATP13A4',
]);
// Same audit sample, the genes on the other side: clean gene-level match to UniProt/PubMed-
// documented function (still gene-level only, never isoform-specific — see caveat text above).
// Excludes soft/imprecise matches from the same audit (e.g. ADORA1 wrong GPCR sub-branch,
// AP1M1/ALG1/ARHGEF4/BBS2/ATP10B/ATG4A/BEST4 partial-fit calls) — only unambiguous hits are marked.
const NOVELGO_LIT_MATCHED = new Set([
  'AARS1','ACSF2','ACSS1','ADAT1','ADGRE5','ADHFE1','ADSS1','ADSS2','AK1','AK2','AK8',
  'ALDH1B1','ALDH2','ALG10','ANKIB1','ACVR2B','ARHGAP17','ASIC4','ATP6V0A2','BRAF','BET1L',
  'B3GALT9','B4GALT2','B4GALT3','BPNT2','ADORA2B','AMPD3',
]);

async function loadNovelGoRegion(region){
  const panel = document.getElementById('novelGoRegionPanel');
  const title = document.getElementById('novelGoRegionTitle');
  const tbody = document.getElementById('novelGoRegionTbody');
  const more = document.getElementById('novelGoRegionMore');
  panel.classList.remove('hidden');
  title.textContent = '§ ' + (NOVELGO_REGION_META[region] ? NOVELGO_REGION_META[region].label : region);
  tbody.innerHTML = '<tr><td colspan="4" class="muted mono sm">▸ loading…</td></tr>';
  more.textContent = '';
  const tissue = _novelGoRegionTissue();
  const r = await fetch('/api/summary/' + encodeURIComponent(tissue) + '/novel_go_region?region=' + encodeURIComponent(region));
  const d = await r.json();
  if(d.error){ tbody.innerHTML = `<tr><td colspan="4" class="err mono sm">✕ ${d.error}</td></tr>`; return; }
  tbody.innerHTML = d.isoforms.map(iso => {
    let flag = '';
    if(NOVELGO_LIT_CONTRADICTED.has(iso.gene)){
      flag = ` <span class="badge caveat" style="padding:.05rem .5rem;font-size:.65rem"
          title="Manual literature check found this gene's real (UniProt/PubMed-documented) function contradicts this predicted GO theme — likely paralog/compartment/complex-location confusion, not a validated novel function.">⚠ lit-contradicted</span>`;
    } else if(NOVELGO_LIT_MATCHED.has(iso.gene)){
      flag = ` <span class="badge match" style="padding:.05rem .5rem;font-size:.65rem"
          title="Manual literature check found this gene's real (UniProt/PubMed-documented) function matches this predicted GO theme's broad class — gene-level only, not isoform-specific confirmation.">✓ lit-matched</span>`;
    }
    return `
    <tr onclick="location.href='/gene/'+encodeURIComponent('${iso.isoform_id}')" title="probe in individual analysis ▸">
      <td>${iso.gene}${geneCardsLink(iso.gene)}${flag}</td><td>${iso.isoform_id} <span class="muted">▸</span></td>
      <td class="muted">${escapeHtml(iso.go_name)} (${iso.go_id})</td>
      <td class="tnum">${iso.score.toFixed(3)}</td></tr>`;
  }).join('')
    || '<tr><td colspan="4" class="muted mono sm">no pairs in this region</td></tr>';
  more.textContent = d.truncated
    ? `showing top ${d.isoforms.length.toLocaleString()} of ${d.total.toLocaleString()} (gene, GO term) pairs, sorted by score`
    : `${d.total.toLocaleString()} (gene, GO term) pair(s)`;
}
const novelGoRegionCloseEl = document.getElementById('novelGoRegionClose');
if(novelGoRegionCloseEl) novelGoRegionCloseEl.addEventListener('click', () =>
  document.getElementById('novelGoRegionPanel').classList.add('hidden'));

// the two checkboxes are now direct shortcuts into the same region-list panel (their sub-slices
// of "new" no longer have a nested-circle drawing to toggle — venn.js draws only the 2 base sets +
// intersection, so a 3rd/4th nested subset would need its own separate mini-figure; a direct list
// is simpler and gives the same "how many, which ones" answer more usefully).
const novelGoHiconfEl = document.getElementById('novelGoHiconf');
if(novelGoHiconfEl) novelGoHiconfEl.addEventListener('change', () => {
  if(novelGoHiconfEl.checked) loadNovelGoRegion('new_hiconf');
  else document.getElementById('novelGoRegionPanel').classList.add('hidden');
});
const novelGoBeatsEl = document.getElementById('novelGoBeats');
if(novelGoBeatsEl) novelGoBeatsEl.addEventListener('change', () => {
  if(novelGoBeatsEl.checked) loadNovelGoRegion('new_beats');
  else document.getElementById('novelGoRegionPanel').classList.add('hidden');
});

async function loadGoTermSelector(tissue){
  const sel = document.getElementById('goTermSel');
  if(_goDistTissue !== tissue){
    _goDistTissue = tissue;
    _goTerms = await (await fetch('/api/summary/'+encodeURIComponent(tissue)+'/go_terms')).json();
    sel.innerHTML = _goTerms.map(t=>`<option value="${t.go_id}">${escapeHtml(t.name)} (${t.go_id})</option>`).join('');
  }
  const goId = sel.value || (_goTerms[0] && _goTerms[0].go_id);
  if(goId) await loadGoDistribution(tissue, goId);
}

// State kept around so hover/search/toggles/lock can highlight gene(s) without re-fetching.
let _goDist = null, _goDistGeneRows = null, _goDistGeneLower = null, _goDistPinnedGene = null;
let _goDistHighlights = null;             // /go_distribution/highlights payload for the current go_id
let _goDistHlActive = new Set();          // criterion keys currently "colour on"
let _goDistLock = null;                   // {gene, source: 'plot' | one of GENE_HL_KEYS} | null
let _goDistHoverTraceAdded = false;       // transient hover trace present beyond the persistent overlays

async function loadGoDistribution(tissue, goId){
  const host = document.getElementById('goDistPlot');
  host.innerHTML = '<p class="muted mono sm">loading...</p>';
  const [d, hl] = await Promise.all([
    fetch('/api/summary/'+encodeURIComponent(tissue)+'/go_distribution?go_id='+encodeURIComponent(goId)).then(r=>r.json()),
    fetch('/api/summary/'+encodeURIComponent(tissue)+'/go_distribution/highlights?go_id='+encodeURIComponent(goId)).then(r=>r.json()),
  ]);
  if(d.error){ host.innerHTML = `<p class="err mono sm">error: ${d.error}</p>`; return; }
  host.innerHTML = '<div id="goDistCanvas"></div>';
  _goDist = d;
  _goDistHighlights = hl.error ? null : hl;
  _goDistPinnedGene = null;
  _goDistHlActive = new Set();
  _goDistLock = null;
  _goDistHoverTraceAdded = false;

  // gene -> [row indices] lookup, built once per load (used by hover + search + toggle + lock).
  _goDistGeneRows = {};
  d.isoform_genes.forEach((g, i) => {
    (_goDistGeneRows[g] || (_goDistGeneRows[g] = [])).push(i);
  });
  _goDistGeneLower = {};
  Object.keys(_goDistGeneRows).forEach(g => { _goDistGeneLower[g.toLowerCase()] = g; });

  document.getElementById('goDistGeneList').innerHTML =
    Object.keys(_goDistGeneRows).sort().map(g => `<option value="${g}">`).join('');
  document.getElementById('goDistGeneSearch').value = '';
  document.getElementById('goDistGeneStatus').textContent = '';
  renderGoDistHlRow();
  document.getElementById('goDistHlPanels').innerHTML = '';
  renderGoDistLockPanel();
  renderGoDistBase();
}

// non-coding transcripts have zero ESM-2 embedding by design \u2192 sigmoid(bias)-only predictions,
// identical across all 672 GO terms, carrying no functional signal
// ([[finding-brain672-noncoding-zero-embedding-flag-needed]]). Hidden by default so the plot
// reflects genuine sequence\u2192function signal; togglable for anyone who wants to see the "wall".
let _goDistHideNoncoding = true;
const NONCODING_COLOR = '#5A6478';
// base trace layout is fixed regardless of hide/show so overlay-trace indices never shift:
// 0 coding-lines \u00b7 1 coding-markers \u00b7 2 noncoding-lines \u00b7 3 noncoding-markers.
// The population KDE curve is deliberately NOT one of these \u2014 it's re-added as the LAST trace
// every time redrawGoDistOverlays() runs (below), so it always draws on top of every highlight
// overlay regardless of how many are active. Relying on 'scatter' (SVG) always compositing above
// 'scattergl' (WebGL) wasn't reliable enough in practice once overlay line count/opacity grew.
const GODIST_BASE_TRACE_COUNT = 4;

function _goDistSpikeXY(idx){
  const x = new Array(idx.length*3), y = new Array(idx.length*3);
  idx.forEach((i,k) => {
    const s = _goDist.isoform_scores[i];
    x[k*3]=s; x[k*3+1]=s; x[k*3+2]=null;
    y[k*3]=0; y[k*3+1]=1; y[k*3+2]=null;
  });
  return {x, y};
}

// population KDE curve, built fresh each redraw and always appended LAST (see GODIST_BASE_TRACE_COUNT
// note) so it renders on top of every isoform line / highlight overlay instead of getting lost in them.
function _goDistKdeTrace(){
  return {type:'scatter', mode:'lines', x:_goDist.grid, y:_goDist.aggregate,
    line:{color:T.signal, width:2.5}, name:'all isoforms (population, gene-independent)',
    hovertemplate:'score %{x:.2f}<br>relative density %{y:.2f}<extra></extra>'};
}

// (re)draws the base plot (raw spikes + hover-target markers + population KDE) from the already
// -fetched `_goDist` and current `_goDistHideNoncoding` \u2014 no refetch, so the hide/show checkbox
// can call this directly. Re-wires hover/click every call (Plotly.newPlot replaces the div).
function renderGoDistBase(){
  const d = _goDist;
  const n = d.isoform_scores.length;
  const isNc = d.is_noncoding || new Array(n).fill(false);
  const codingIdx = [], noncodingIdx = [];
  for(let i=0;i<n;i++) (isNc[i] ? noncodingIdx : codingIdx).push(i);
  const shownNcIdx = _goDistHideNoncoding ? [] : noncodingIdx;

  const cLines = _goDistSpikeXY(codingIdx), ncLines = _goDistSpikeXY(shownNcIdx);
  const cCustom = codingIdx.map(i => [d.isoform_genes[i], d.isoform_ids[i], false]);
  const ncCustom = shownNcIdx.map(i => [d.isoform_genes[i], d.isoform_ids[i], true]);

  // `visible:false` (not just empty x/y) when there's nothing to show for a scattergl trace \u2014 some
  // Plotly.js versions corrupt the shared WebGL canvas (garbled figure, other traces including the
  // KDE curve added later via addTraces silently failing to paint) when a scattergl trace is fed
  // zero-length data arrays while sibling scattergl traces in the same plot are non-empty. This is
  // exactly the "hide non-coding" state (shownNcIdx=[] whenever the toggle is on or a GO term simply
  // has no non-coding isoforms) \u2014 keep the trace present (fixed GODIST_BASE_TRACE_COUNT index
  // invariant, see below) but invisible instead of empty.
  const ncVisible = shownNcIdx.length > 0;
  Plotly.newPlot('goDistCanvas', [
    {type:'scattergl', mode:'lines', x:cLines.x, y:cLines.y, line:{color:T.trace, width:.8},
     opacity:.08, hoverinfo:'skip', showlegend:false, name:'isoforms (raw)'},
    {type:'scattergl', mode:'markers', x:codingIdx.map(i=>d.isoform_scores[i]), y:new Array(codingIdx.length).fill(1),
     marker:{size:6, color:T.trace}, opacity:.18, customdata:cCustom, showlegend:false,
     hovertemplate:'%{customdata[0]} \u00b7 %{customdata[1]}<br>score %{x:.3f}<extra></extra>'},
    {type:'scattergl', mode:'lines', x:ncLines.x, y:ncLines.y, line:{color:NONCODING_COLOR, width:.9},
     opacity:.35, hoverinfo:'skip', showlegend:false, name:'non-coding (no signal)', visible:ncVisible},
    {type:'scattergl', mode:'markers', x:shownNcIdx.map(i=>d.isoform_scores[i]), y:new Array(shownNcIdx.length).fill(1),
     marker:{size:6, color:NONCODING_COLOR}, opacity:.28, customdata:ncCustom, showlegend:false, visible:ncVisible,
     hovertemplate:'%{customdata[0]} \u00b7 %{customdata[1]}<br>\u26a0 non-coding, zero-embedding, no signal<extra></extra>'},
  ], inst({height:320, margin:{l:20,r:16,t:10,b:36}, showlegend:true,
    legend:{font:{size:9,color:T.dim}, orientation:'h', y:-0.24},
    xaxis:{title:{text:'GO score',font:{size:10,color:T.dim}}, range:[0,1], dtick:0.1},
    yaxis:{visible:false, range:[0, GODIST_Y_TOP]}, annotations:[]}), PLOT_CFG);

  const gd = document.getElementById('goDistCanvas');
  gd.on('plotly_hover', ev => {
    const p = ev.points && ev.points[0];
    if(!p || !p.customdata) return;
    const [gene, , isNonCoding] = p.customdata;
    showGoDistHoverTrace(gene);
    document.getElementById('goDistGeneStatus').textContent = isNonCoding
      ? `${gene}: non-coding \u2014 zero ESM-2 embedding, no signal (hover)`
      : `${gene}: ${(_goDistGeneRows[gene]||[]).length.toLocaleString()} isoforms (hover)`;
  });
  gd.on('plotly_unhover', () => {
    hideGoDistHoverTrace();
    document.getElementById('goDistGeneStatus').textContent = goDistStatusText();
  });
  gd.on('plotly_click', ev => {
    const p = ev.points && ev.points[0];
    if(!p || !p.customdata) return;
    setGoDistLock(p.customdata[0], 'plot');
  });

  const ncNote = d.n_noncoding
    ? ` \u00b7 ${d.n_noncoding.toLocaleString()} non-coding isoforms (zero ESM-2 embedding, no signal) `+
      (_goDistHideNoncoding ? 'hidden by default' : 'shown in grey')
    : '';
  document.getElementById('go-dist-cap').textContent =
    `${d.go_name} (${d.go_id}) \u00b7 ${(d.n_isoforms_total-d.n_noncoding).toLocaleString()} coding isoforms shown as raw spikes `+
    `(1 tick each, exact score, no smoothing) \u00b7 hover a spike or search a gene to highlight all its `+
    `isoforms, click a spike (or a gene below) to lock it \u00b7 highlighted genes get a leader-line `+
    `label above the plot pulling their (possibly scattered) isoforms together \u00b7 thick curve = `+
    `population KDE (peak-normalized), gene-independent, always drawn on top${ncNote}`;

  redrawGoDistOverlays();
}

// -- overlay trace management -------------------------------------------------------------
// Everything beyond the GODIST_BASE_TRACE_COUNT base traces (coding/non-coding raw spikes,
// hover-target markers, population KDE) is rebuilt together here: search-pin, each active
// criterion's gene set, and the click-lock -- in
// that fixed order, so re-toggling one never disturbs the others' colours. The transient hover
// trace (added by showGoDistHoverTrace) always lives on top of this block and is tracked
// separately since it comes and goes far more often (every mouse move).
function _goDistSpikeTrace(genes, color, width, name){
  const hx = [], hy = [];
  genes.forEach(g => (_goDistGeneRows[g] || []).forEach(i => {
    const s = _goDist.isoform_scores[i]; hx.push(s, s, null); hy.push(0, 1, null);
  }));
  return {type:'scattergl', mode:'lines', x:hx, y:hy, line:{color, width},
          hoverinfo:'skip', showlegend:false, name};
}

// -- leader-line gene labels ---------------------------------------------------------------
// A highlighted gene's isoforms are scattered along the score axis (not spatially clustered like
// UMAP points), so a colour on a bunch of thin 1px lines is easy to miss — this pulls a line from
// each of the gene's spike tops up to ONE shared anchor point above the plot and writes the gene
// name there once, same idea as the UMAP cluster leader-line labels (drawUmap/_umapClusterAnnotations)
// but converging FROM multiple scattered x-positions TO one point instead of one centroid outward.
const GODIST_LABEL_TIERS = [1.06, 1.15, 1.24];   // persistent highlights cycle through these 3
const GODIST_HOVER_TIER = 1.34;                   // hover always gets its own top slot (transient,
                                                   // must never collide with a persistent label)
const GODIST_Y_TOP = GODIST_HOVER_TIER + 0.08;

function _goDistGeneAnchorX(gene){
  const idx = _goDistGeneRows[gene] || [];
  if(!idx.length) return 0.5;
  let sum = 0;
  idx.forEach(i => sum += _goDist.isoform_scores[i]);
  return Math.min(0.96, Math.max(0.04, sum / idx.length));
}

function _goDistGeneLeaderAnnotations(gene, color, tierY, bold){
  const idx = _goDistGeneRows[gene] || [];
  if(!idx.length) return [];
  const ax = _goDistGeneAnchorX(gene);
  const anns = idx.map(i => ({
    x: _goDist.isoform_scores[i], y: 1, ax, ay: tierY, axref:'x', ayref:'y',
    text:'', showarrow:true, arrowhead:0, arrowwidth: bold?1.4:1, arrowcolor:color, opacity:.6,
  }));
  anns.push({
    x: ax, y: tierY, text: escapeHtml(gene), showarrow:false,
    font:{size: bold?10:8.5, color}, bgcolor:'rgba(11,14,20,.82)', bordercolor:'transparent',
    xanchor:'center', yanchor:'bottom', yshift:2,
  });
  return anns;
}

function _goDistHlGeneList(key){
  if(!_goDistHighlights) return [];
  if(key === 'top_score') return _goDistHighlights.top_score ? [_goDistHighlights.top_score.gene] : [];
  if(key === 'top_variance') return _goDistHighlights.top_variance.map(r => r.gene);
  if(key === 'single_iso_high') return _goDistHighlights.single_iso_high.map(r => r.gene);
  if(key === 'top_mean') return _goDistHighlights.top_mean.map(r => r.gene);
  return [];
}

let _goDistBaseAnnotations = [];   // persistent leader-labels (pin/criteria/lock) — hover merges
                                    // its own transient label on top of this, never replaces it.
let _goDistTierCounter = 0;        // round-robins GODIST_LABEL_TIERS across whatever's active,
                                    // in a fixed order (pin, then criteria, then lock) so re-toggling
                                    // one source doesn't reshuffle another's already-placed label.

function redrawGoDistOverlays(){
  const gd = document.getElementById('goDistCanvas');
  if(!gd || !gd.data) return;
  if(gd.data.length > GODIST_BASE_TRACE_COUNT){
    Plotly.deleteTraces(gd, Array.from({length: gd.data.length - GODIST_BASE_TRACE_COUNT}, (_, i) => i + GODIST_BASE_TRACE_COUNT));
  }
  _goDistHoverTraceAdded = false;   // any transient hover trace was just wiped along with the rest
  const traces = [];
  _goDistTierCounter = 0;
  const anns = [];
  const nextTier = () => GODIST_LABEL_TIERS[(_goDistTierCounter++) % GODIST_LABEL_TIERS.length];

  if(_goDistPinnedGene && _goDistGeneRows[_goDistPinnedGene]){
    traces.push(_goDistSpikeTrace([_goDistPinnedGene], T.ok, 2, _goDistPinnedGene + ' (pinned)'));
    anns.push(..._goDistGeneLeaderAnnotations(_goDistPinnedGene, T.ok, nextTier(), true));
  }
  GENE_HL_KEYS.forEach(key => {
    if(!_goDistHlActive.has(key)) return;
    const genes = _goDistHlGeneList(key);
    if(!genes.length) return;
    traces.push(_goDistSpikeTrace(genes, GENE_HL_PAL[key].color, 1.3, GENE_HL_PAL[key].label));
    genes.forEach(g => anns.push(..._goDistGeneLeaderAnnotations(g, GENE_HL_PAL[key].color, nextTier(), false)));
  });
  if(_goDistLock && _goDistGeneRows[_goDistLock.gene]){
    traces.push(_goDistSpikeTrace([_goDistLock.gene], GENE_LOCK_COLOR, 2.6, _goDistLock.gene + ' (locked)'));
    anns.push(..._goDistGeneLeaderAnnotations(_goDistLock.gene, GENE_LOCK_COLOR, nextTier(), true));
  }
  traces.push(_goDistKdeTrace());   // always last → always drawn on top of every highlight above
  Plotly.addTraces(gd, traces);
  _goDistBaseAnnotations = anns;
  Plotly.relayout(gd, {annotations: anns});
}

// hover's trace is inserted just BEFORE the KDE trace (which redrawGoDistOverlays always leaves as
// the last one) so the KDE curve stays on top even while a hover highlight is showing; the exact
// index is tracked since it's no longer simply "the last trace" once this insertion happens.
let _goDistHoverTraceIndex = null;

function showGoDistHoverTrace(gene){
  const gd = document.getElementById('goDistCanvas');
  if(!gd || !gd.data || !_goDistGeneRows[gene]) return;
  hideGoDistHoverTrace();
  const insertAt = gd.data.length - 1;   // just before the current last trace (the KDE curve)
  Plotly.addTraces(gd, [_goDistSpikeTrace([gene], T.ok, 2, gene + ' (hover)')], [insertAt]);
  _goDistHoverTraceIndex = insertAt;
  _goDistHoverTraceAdded = true;
  const hoverAnns = _goDistGeneLeaderAnnotations(gene, T.ok, GODIST_HOVER_TIER, true);
  Plotly.relayout(gd, {annotations: _goDistBaseAnnotations.concat(hoverAnns)});
}
function hideGoDistHoverTrace(){
  const gd = document.getElementById('goDistCanvas');
  if(!gd || !gd.data || !_goDistHoverTraceAdded) return;
  Plotly.deleteTraces(gd, _goDistHoverTraceIndex);
  _goDistHoverTraceIndex = null;
  _goDistHoverTraceAdded = false;
  Plotly.relayout(gd, {annotations: _goDistBaseAnnotations});
}
function goDistStatusText(){
  if(_goDistPinnedGene) return `${_goDistPinnedGene}: ${_goDistGeneRows[_goDistPinnedGene].length.toLocaleString()} isoforms (pinned via search)`;
  if(_goDistLock) return `${_goDistLock.gene}: ${(_goDistGeneRows[_goDistLock.gene]||[]).length.toLocaleString()} isoforms (locked)`;
  return '';
}

// -- criterion chips ("color on/off") -----------------------------------------------------
// dynamic label — bakes the live threshold / truncation state into the chip text itself so a
// user doesn't have to open the panel to discover e.g. "capped at 500" (devils-advocate M2).
function _goDistHlChipLabel(key){
  if(key === 'single_iso_high' && _goDistHighlights){
    const thr = _goDistHighlights.single_iso_thr;
    const n = _goDistHlGeneList(key).length;
    const capped = _goDistHighlights.single_iso_high_truncated ? `, capped@${n}` : '';
    return `Single-isoform high-score (≥${thr}${capped})`;
  }
  return GENE_HL_PAL[key].label;
}

function renderGoDistHlRow(){
  const row = document.getElementById('goDistHlRow');
  if(!row) return;
  row.innerHTML = GENE_HL_KEYS.map(key => {
    const pal = GENE_HL_PAL[key];
    const genes = _goDistHlGeneList(key);
    return `<button type="button" class="gene-hl-chip" data-crit="${key}" style="--hl:${pal.color}"
      title="${genes.length} gene(s)">${escapeHtml(_goDistHlChipLabel(key))}</button>`;
  }).join('');
}

function renderHlPanel(key){
  const panels = document.getElementById('goDistHlPanels');
  let panel = document.getElementById('hlPanel_' + key);
  if(!panel){
    panel = document.createElement('div');
    panel.className = 'gene-hl-panel';
    panel.id = 'hlPanel_' + key;
    panel.style.setProperty('--hl', GENE_HL_PAL[key].color);
    panels.appendChild(panel);
  }
  const genes = _goDistHlGeneList(key);
  panel.innerHTML = `<div class="gene-hl-head" title="Clicking a gene locks it (teal) until: a `+
    `different gene is clicked, \u25be collapse is pressed, or this chip is turned off \u2014 turning this `+
    `chip off while it holds the lock clears the lock too.">`+
    `<b>${escapeHtml(_goDistHlChipLabel(key))}</b>`+
    `<span class="muted">${genes.length} gene(s) \u00b7 click a gene to lock it in the plot</span></div>`+
    `<div class="gene-hl-genes">${genes.map(g => {
      const locked = _goDistLock && _goDistLock.gene === g && _goDistLock.source === key;
      return `<button type="button" class="gene-hl-gene-btn${locked ? ' locked' : ''}" `+
        `style="--hl:${GENE_HL_PAL[key].color}" data-gene="${escapeHtml(g)}" data-crit="${key}">${escapeHtml(g)}</button>`;
    }).join('')}</div>`;
}

function setCriterionActive(key, on, fromAutoOff){
  if(on){
    // radio-style — only one criterion "colour on" (and one highlight panel) at a time, per request;
    // switching to a new one turns off whatever else was active first (also releases its lock, same
    // as an explicit "color off" click on it would — see the `on` branch below).
    Array.from(_goDistHlActive).filter(k => k !== key).forEach(k => setCriterionActive(k, false, fromAutoOff));
    _goDistHlActive.add(key);
  } else {
    _goDistHlActive.delete(key);
  }
  document.querySelectorAll(`.gene-hl-chip[data-crit="${key}"]`).forEach(b => b.classList.toggle('on', on));
  if(on){
    renderHlPanel(key);
  } else {
    const panel = document.getElementById('hlPanel_' + key);
    if(panel) panel.remove();
    // explicit "color off" click on an already-locked criterion clears the lock too (one of the
    // three unlock conditions for a criterion-sourced lock) -- but not when THIS call is itself
    // the auto-off reaction to the lock having already moved elsewhere (avoids re-entrant clear).
    if(!fromAutoOff && _goDistLock && _goDistLock.source === key){
      _goDistLock = null;
      renderGoDistLockPanel();
    }
  }
  redrawGoDistOverlays();
}

// -- click-to-lock --------------------------------------------------------------------------
function setGoDistLock(gene, source){
  if(!_goDistGeneRows[gene]) return;
  const prevSource = _goDistLock ? _goDistLock.source : null;
  _goDistLock = {gene, source};
  renderGoDistLockPanel();
  if(prevSource && prevSource !== 'plot' && prevSource !== source && _goDistHlActive.has(prevSource)){
    setCriterionActive(prevSource, false, true);   // also redraws overlays + re-renders that panel's list
  } else {
    redrawGoDistOverlays();
  }
  // keep the (possibly still-open) source criterion panel's "locked" chip in sync
  if(_goDistHlActive.has(source)) renderHlPanel(source);
}

function clearGoDistLock(){
  const prevSource = _goDistLock ? _goDistLock.source : null;
  _goDistLock = null;
  renderGoDistLockPanel();
  if(prevSource && prevSource !== 'plot' && _goDistHlActive.has(prevSource)){
    setCriterionActive(prevSource, false, true);
  } else {
    redrawGoDistOverlays();
  }
}

function renderGoDistLockPanel(){
  const panel = document.getElementById('goDistLockPanel');
  if(!panel) return;
  if(!_goDistLock){ panel.classList.add('hidden'); panel.innerHTML = ''; return; }
  const gene = _goDistLock.gene;
  const isNc = _goDist.is_noncoding || [];
  const rows = (_goDistGeneRows[gene] || []).map(i => ({
    isoform_id: _goDist.isoform_ids[i], score: _goDist.isoform_scores[i], noncoding: !!isNc[i],
  })).sort((a, b) => b.score - a.score);
  panel.classList.remove('hidden');
  panel.innerHTML = `<div class="gene-hl-head"><b>\u25b8 ${escapeHtml(gene)}</b>`+
    `<span class="muted">${rows.length.toLocaleString()} isoform(s) \u00b7 locked</span>`+
    `<button type="button" class="btn ghost sm" id="goDistLockCollapse" style="margin-left:auto">\u25be collapse</button></div>`+
    `<table><thead><tr><th>isoform</th><th>score</th></tr></thead><tbody>`+
    rows.map(r => `<tr><td>${escapeHtml(r.isoform_id)}${r.noncoding
        ? ' <span class="muted" title="zero ESM-2 embedding by design, no functional signal">\u26a0 non-coding</span>' : ''}</td>`+
      `<td>${r.score.toFixed(3)}</td></tr>`).join('')+
    `</tbody></table>`;
  document.getElementById('goDistLockCollapse').addEventListener('click', clearGoDistLock);
}

const goDistHlRowEl = document.getElementById('goDistHlRow');
if(goDistHlRowEl){
  goDistHlRowEl.addEventListener('click', e => {
    const b = e.target.closest('.gene-hl-chip'); if(!b) return;
    const key = b.dataset.crit;
    setCriterionActive(key, !_goDistHlActive.has(key), false);
  });
}
const goDistHlPanelsEl = document.getElementById('goDistHlPanels');
if(goDistHlPanelsEl){
  goDistHlPanelsEl.addEventListener('click', e => {
    const b = e.target.closest('.gene-hl-gene-btn'); if(!b) return;
    setGoDistLock(b.dataset.gene, b.dataset.crit);
  });
}

const goTermSel = document.getElementById('goTermSel');
if(goTermSel){
  goTermSel.addEventListener('change', () => {
    loadGoDistribution(document.getElementById('dash').dataset.active || 'brain_672', goTermSel.value);
  });
}

const goDistGeneSearch = document.getElementById('goDistGeneSearch');
if(goDistGeneSearch){
  goDistGeneSearch.addEventListener('input', () => {
    const raw = goDistGeneSearch.value.trim();
    const status = document.getElementById('goDistGeneStatus');
    if(!raw){ _goDistPinnedGene = null; redrawGoDistOverlays(); status.textContent = ''; return; }
    const key = _goDistGeneLower ? _goDistGeneLower[raw.toLowerCase()] : null;
    if(!key){
      _goDistPinnedGene = null; redrawGoDistOverlays();
      status.textContent = `"${raw}" not found for this GO term`;
      return;
    }
    _goDistPinnedGene = key;
    redrawGoDistOverlays();
    status.textContent = `${key}: ${_goDistGeneRows[key].length.toLocaleString()} isoforms (pinned via search)`;
  });
}

const goDistHideNoncodingEl = document.getElementById('goDistHideNoncoding');
if(goDistHideNoncodingEl){
  goDistHideNoncodingEl.addEventListener('change', () => {
    _goDistHideNoncoding = goDistHideNoncodingEl.checked;
    if(_goDist) renderGoDistBase();   // re-draw from already-fetched data, no refetch needed
  });
}

// ── § FIG · GO-Score space explorer — brain·672 참조 3종, 옵션 선택으로 하나씩 ──
// (tissue 탭과 무관 — 항상 brain-672 사전계산 reference. 모두 한번에 그리지 않고 선택한 것만.)
//
// UMAP 선택 UX: 범례 클릭(Plotly 기본 동작)으로 카테고리를 켜고 끄는 것 자체가 "선택" —
// 클릭할 때마다 즉시 목록을 갱신하지 않는다. 대신 "Selected isoform list" 버튼을 눌렀을 때만
// 그 시점에 켜져 있는 트레이스들을 읽어 한 번에 조회한다 — 선택(legend)과 조회(button)를
// 의도적으로 분리. 축 범위는 항상 고정 — 범례 on/off 로 안 보이는 점이 생겨도 Plotly 가
// autorange 로 화면을 다시 잡지 않도록 xaxis/yaxis range 를 명시적으로 박아둔다.
let _figxKind = 'umap';
let _umapColorBy = 'max_go';
// score-gradient "overlay" — genuinely independent of _umapColorBy (which categorical partition is
// selected in the dropdown): the backend always returns cluster_id/clusters/max_score alongside
// whatever colour_by was requested (dataset_summary.umap_points), so this never needs its own fetch,
// it just changes how drawUmap() renders the ALREADY-fetched payload. Was previously implemented by
// overwriting _umapColorBy to 'max_score' itself (via a since-removed _umapPrevColorBy save/restore
// dance) — that forced a full refetch+redraw on every toggle (felt like a page reload) and discarded
// whichever colour-by the user was actually looking at. Mutually exclusive with the dropdown's own
// 'max_score' option (that one still shows a plain continuous view + threshold-slider selection,
// unrelated to the module filter chips here) — forced off whenever the dropdown is set to it.
let _umapScoreOn = false;
let _umapSpace = 'go_score';         // 'go_score' (672-dim prediction output) | 'axis8' (ESM-2 8-axis L30, pre-prediction)
let _umapFilter = null;              // {colorBy, values} — last isoform-list query (for download)
let _umapTraceKeys = [];             // gd.data[i] → category key (cluster id / type), parallel array
let _umapClusterAnnotations = [];    // all cluster leader-line labels (order matches _umapClusterKeys)
let _umapClusterKeys = [];           // cluster id per entry in _umapClusterAnnotations
let _umapRawScores = null;           // raw max_score array, kept for instant slider-driven restyle
let _umapClustersMeta = [];          // last-drawn u.clusters (id/label/count/subterms) — read by the
                                      // legend double-click handler to gate drill-down on cluster size
let _umapSubclusterId = null;        // non-null while drilled into one cluster's top-5 GO subterms
                                      // (max_go mode only — dropdown itself always stays on 'max_go')
const UMAP_SUBCLUSTER_MIN_COUNT = 500;  // mirrors dataset_summary.SUBCLUSTER_MIN_COUNT

// score-gradient mode's module/subterm on-off filter — no Plotly legend exists there (single
// continuous-colour trace, not one trace per category), so this is tracked separately and driven
// by chips (renderUmapModFilter). Reset whenever the category set changes (tissue/space/drill
// level) via the signature check in drawUmapScoreGradient.
let _umapModuleOn = null;            // Set of category keys currently "on"
let _umapModuleOnSig = null;         // joined category-key list the above Set was built for
let _umapLastResponse = null;        // last /api/umap payload, cached so chip toggles re-render
                                      // locally instead of re-fetching

const UMAP_SPACE_CAP = {
  go_score: 'Each point = one isoform (brain-672 precomputed 20K sample), positioned by cosine-distance UMAP over its 672-dim GO score vector — the MLP prediction head\'s output. Clusters here group isoforms by predicted function.',
  axis8: 'Each point = one isoform (same 20K sample), positioned by cosine-distance UMAP over its 8-axis L30 embedding — ESM-2\'s 30-layer sequence representation, projected to 8 axes, before the GO-prediction head ever sees it. Clusters here group isoforms by sequence representation, independent of what the model predicts.',
};
const FIGX_CAP = {
  bubble: '44 functional modules — X = mean brain zero-shot AUPRC (prediction quality) · Y = Novel(NIC+NNIC) enrichment vs background · bubble size = high-confidence isoform count (module_score>0.3).',
  corr: 'GO-GO Pearson correlation matrix, GO terms sorted by their 44-module assignment (module boundaries = black gridlines). Downsampled for render speed.',
};
// shared Okabe-Ito palette (instrument.js) — theme-fixed 6-colour safe subset (excludes black/yellow,
// each of which fails contrast on one of the two themes — see instrument.js header comment).
// distinctPalette() (also instrument.js) extends it past 6 categories (up to N_SUBTERMS=30, or the
// 8 top-level UMAP clusters here) via HSL — shared with bisect.js's CELLTYPE_PAL/ISO_PAL so an
// 8-category palette looks the same on every page, not independently re-derived per file.
const CLUSTER_PAL = OKABE_ITO_SAFE;
const UMAP_N_CLUSTERS = 8;   // mirrors dataset_summary.N_UMAP_CLUSTERS
const UMAP_CLUSTER_PAL = distinctPalette(UMAP_N_CLUSTERS);

function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

async function loadFigx(kind){
  _figxKind = kind;
  const host = document.getElementById('figxPlot');
  document.getElementById('figx-cap').textContent = kind === 'umap' ? (UMAP_SPACE_CAP[_umapSpace]||'') : (FIGX_CAP[kind] || '');
  document.getElementById('umapCtrl').classList.toggle('hidden', kind !== 'umap');
  document.getElementById('umapHint').classList.toggle('hidden', kind !== 'umap');
  document.getElementById('umapListRow').classList.toggle('hidden', kind !== 'umap');
  document.getElementById('umapScoreToggleWrap').classList.toggle('hidden', kind !== 'umap');
  if(kind !== 'umap') hideUmapIsoPanel();
  host.innerHTML = '<p class="muted mono sm">▸ loading…</p>';
  if(kind === 'umap'){
    await loadUmap();
  } else if(kind === 'bubble'){
    const b = await (await fetch('/api/module_landscape/bubble')).json();
    drawBubble(b);
  } else if(kind === 'corr'){
    const c = await (await fetch('/api/module_landscape/corr')).json();
    if(c.error){ host.innerHTML = `<p class="err mono sm">✕ ${c.error}</p>`; return; }
    drawCorr(c);
  }
}

// effective color_by actually sent to the API — drilled-in state (entered/left via the module
// button / breadcrumb, not the dropdown) always needs subterm membership regardless of whichever
// categorical partition the dropdown is on, since drilling is inherently a module-cluster concept;
// otherwise send whatever the dropdown is set to. _umapScoreOn never affects this — the score
// overlay reuses whatever was already fetched (cluster_id/clusters/max_score are unconditional
// fields in every /api/umap response), so toggling it never triggers a fetch.
function _umapEffectiveColorBy(){
  return (_umapSubclusterId != null) ? 'max_go_sub' : _umapColorBy;
}

// keeps the umap-ctrl chrome (label toggle, threshold slider, breadcrumb, drill button, module
// filter row, score-toggle checkbox state/enabled-ness) in sync with the current mode — called both
// after a fetch (loadUmap) and after a pure local re-render (score-toggle flip, no fetch needed).
function _umapSyncCtrlUI(){
  const drilled = _umapSubclusterId != null;
  document.getElementById('umapLabelToggleWrap').classList.toggle('hidden',
    _umapColorBy !== 'max_go' || _umapScoreOn || drilled);
  document.getElementById('umapScoreHost').classList.toggle('hidden', _umapColorBy !== 'max_score');
  document.getElementById('umapBreadcrumb').classList.toggle('hidden', !drilled);
  document.getElementById('umapModuleGoBtn').classList.toggle('hidden',
    !((_umapColorBy === 'max_go' || _umapScoreOn) && !drilled));
  document.getElementById('umapModFilterRow').classList.toggle('hidden', !_umapScoreOn);
  const scoreToggle = document.getElementById('umapScoreToggle');
  const scoreToggleWrap = document.getElementById('umapScoreToggleWrap');
  if(scoreToggle){
    scoreToggle.checked = _umapScoreOn;
    scoreToggle.disabled = (_umapColorBy === 'max_score');
    if(scoreToggleWrap){
      scoreToggleWrap.classList.toggle('is-disabled', scoreToggle.disabled);
      scoreToggleWrap.title = scoreToggle.disabled
        ? 'already showing a continuous score view via the colour-by dropdown'
        : 'overlay PRISM score as a colour gradient, independent of the module colour-by dropdown';
    }
  }
}

async function loadUmap(){
  const host = document.getElementById('figxPlot');
  document.getElementById('figx-cap').textContent = UMAP_SPACE_CAP[_umapSpace] || '';
  _umapSyncCtrlUI();
  hideUmapIsoPanel();
  const effColorBy = _umapEffectiveColorBy();
  let url = '/api/umap?color_by='+encodeURIComponent(effColorBy)+'&space='+encodeURIComponent(_umapSpace);
  if(effColorBy === 'max_go_sub') url += '&cluster_id='+encodeURIComponent(_umapSubclusterId);
  const u = await (await fetch(url)).json();
  if(u.error){
    host.innerHTML = `<p class="err mono sm">✕ ${u.error}</p>`;
    if(effColorBy === 'max_go_sub'){ _umapSubclusterId = null; document.getElementById('umapBreadcrumb').classList.add('hidden'); }
    return;
  }
  if(effColorBy === 'max_go_sub' && u.parent_cluster){
    document.getElementById('umapBreadcrumbLabel').textContent =
      `▸ #${u.parent_cluster.id+1} ${u.parent_cluster.label} (n=${u.parent_cluster.count.toLocaleString()}) `+
      `— top-${u.parent_cluster.n_subterms} GO subterms`;
  }
  drawUmap(u);
}

// reuse the existing canvas div if the umap figure is already showing (score-toggle flips, module
// chip toggles) instead of tearing it down and recreating it — only rebuilt fresh when navigating
// into the umap figure from elsewhere (loadFigx already blanks #figxPlot with a loading message
// first in that case, so the div genuinely won't exist yet).
function _umapEnsureCanvas(){
  let gd = document.getElementById('figxUmapCanvas');
  if(!gd){
    document.getElementById('figxPlot').innerHTML = '<div id="figxUmapCanvas"></div>';
    gd = document.getElementById('figxUmapCanvas');
  }
  return gd;
}

// bottom margin fixed large enough to fit the horizontal legend's worst case (max_go: up to 9
// entries, some full-length GO-term labels) even when it wraps onto multiple lines — measured
// empirically at 105px (2 lines, >=1050px-wide panel) up to 181px (4 lines, ~900px-wide panel).
// Previously b:56 — nowhere near enough, so the wrapped legend rows rendered past the plot's own
// height and were invisible (only a sliver of the topmost row peeked out at the very bottom edge).
// Same margin/height applied to both categorical (drawUmap) and score-gradient
// (drawUmapScoreGradient) render paths so the plot area never resizes when toggling between them.
const UMAP_LEGEND_MARGIN_B = 190;
const UMAP_PLOT_HEIGHT = 560 + (UMAP_LEGEND_MARGIN_B - 56);

// fixed axis range from the FULL point set — computed once per draw and never touched again,
// so toggling legend categories on/off only changes colour, never the view/zoom.
function _umapFixedRange(vals){
  let lo = Infinity, hi = -Infinity;
  for(const v of vals){ if(v < lo) lo = v; if(v > hi) hi = v; }
  const pad = (hi - lo) * 0.06 || 1;
  return [lo - pad, hi + pad];
}

// score-gradient render path — a single continuous-colour (Viridis) scatter trace over whichever
// points are currently "on" (module-, or if drilled, GO-subterm-level filter, chips below the
// plot) — "on/off" is purely a visibility filter (points shown or hidden), not a recolour; no
// module-boundary outline is drawn (was a convex hull per module — removed per request, since with
// only one colour channel available it fought the score gradient and wasn't needed once "on/off"
// already means "show/hide" rather than "highlight vs. not").
function drawUmapScoreGradient(u, xRange, yRange){
  const gd = _umapEnsureCanvas();
  const drilled = _umapSubclusterId != null;

  let catKeys, catColorOf, catLabelOf, membershipOf;
  if(!drilled){
    catKeys = (u.clusters||[]).filter(c => c.count > 0 && c.id !== -1).map(c => String(c.id));
    catColorOf = k => UMAP_CLUSTER_PAL[catKeys.indexOf(k) % UMAP_CLUSTER_PAL.length];
    catLabelOf = k => { const c = (u.clusters||[]).find(x => String(x.id) === k); return c ? `#${c.id+1} ${c.label}` : k; };
    membershipOf = i => String(u.cluster_id[i]);
  } else {
    const meta = u.color_meta || {};
    catKeys = Object.keys(meta).filter(k => k !== 'other' && k !== 'etc_go');
    const subPal = distinctPalette(catKeys.length);
    catColorOf = k => subPal[catKeys.indexOf(k)];
    catLabelOf = k => meta[k] ? meta[k].label : k;
    membershipOf = i => u.color[i];
  }

  const keySig = (drilled ? 'sub:'+_umapSubclusterId+':' : 'top:') + catKeys.join(',');
  if(_umapModuleOnSig !== keySig){
    _umapModuleOn = new Set(catKeys);
    _umapModuleOnSig = keySig;
  }

  const idx = [];
  for(let i = 0; i < u.x.length; i++){
    const k = membershipOf(i);
    if(k != null && _umapModuleOn.has(k)) idx.push(i);
  }
  const traces = [{
    type:'scattergl', mode:'markers', x:idx.map(i=>u.x[i]), y:idx.map(i=>u.y[i]),
    text:idx.map(i=>u.isoform_id[i]),
    marker:{size:3.5, opacity:.6, color:idx.map(i=>u.max_score[i]),
      colorscale:VIRIDIS, showscale:true, cmin:0, cmax:1,
      colorbar:{thickness:10,len:.6,tickfont:{family:MONO,size:9,color:T.dim}}},
    hovertemplate:'%{text}<br>max_score %{marker.color:.3f}<extra></extra>',
  }];

  Plotly.newPlot(gd, traces, inst({height:UMAP_PLOT_HEIGHT, margin:{l:12,r:70,t:10,b:UMAP_LEGEND_MARGIN_B,autoexpand:false},
    showlegend:false,
    xaxis:{showticklabels:false, showgrid:false, zeroline:false, range:xRange, autorange:false},
    yaxis:{showticklabels:false, showgrid:false, zeroline:false, range:yRange, autorange:false}}), PLOT_CFG);

  renderUmapModFilter(catKeys, catColorOf, catLabelOf);
}

function renderUmapModFilter(catKeys, catColorOf, catLabelOf){
  const row = document.getElementById('umapModFilterRow');
  row.classList.remove('hidden');
  row.innerHTML = catKeys.map(k => `
    <label>
      <input type="checkbox" data-modkey="${escapeHtml(k)}" ${_umapModuleOn.has(k)?'checked':''}>
      <span class="sw" style="--sw:${catColorOf(k)}"></span>
      <span>${escapeHtml(catLabelOf(k))}</span>
    </label>`).join('') +
    '<span class="muted" style="margin-left:.4rem">on = points shown (score-coloured), off = hidden</span>';
  row.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    cb.addEventListener('change', () => {
      const k = cb.dataset.modkey;
      if(cb.checked) _umapModuleOn.add(k); else _umapModuleOn.delete(k);
      if(_umapLastResponse) drawUmapScoreGradient(_umapLastResponse, _umapFixedRange(_umapLastResponse.x), _umapFixedRange(_umapLastResponse.y));
    });
  });
}

function drawUmap(u){
  _umapEnsureCanvas();
  const cb = u.color_by;
  let traces;
  _umapTraceKeys = [];
  _umapClusterAnnotations = [];
  _umapClusterKeys = [];
  _umapRawScores = null;
  _umapClustersMeta = u.clusters || [];
  _umapLastResponse = u;

  const xRange = _umapFixedRange(u.x), yRange = _umapFixedRange(u.y);
  // centre of the whole point cloud — cluster labels are pushed outward from here (§ leader lines)
  const cx0 = (xRange[0]+xRange[1])/2, cy0 = (yRange[0]+yRange[1])/2;

  if(_umapScoreOn){
    // score overlay is independent of _umapColorBy (the categorical partition the dropdown is on) —
    // it always renders the module/subterm partition (cluster_id/clusters, unconditional fields in
    // every /api/umap response) continuously coloured by score, regardless of what colour_by was
    // requested for this payload. Fully separate render path (single continuous trace + on/off
    // visibility chips) from the categorical one below, so it's split out.
    drawUmapScoreGradient(u, xRange, yRange);
    return;
  }

  if(cb === 'max_score'){
    _umapRawScores = u.color;
    // sequential/continuous data → Viridis (perceptually uniform, colourblind-safe, not a
    // rainbow/Jet map — matches the requested academic-figure convention for gradients).
    traces = [{type:'scattergl', mode:'markers', x:u.x, y:u.y, text:u.isoform_id,
      marker:{size:3.5, opacity:.55, color:u.color,
        colorscale:VIRIDIS, showscale:true,
        colorbar:{thickness:10,len:.6,tickfont:{family:MONO,size:9,color:T.dim}}},
      hovertemplate:'%{text}<br>max_score %{marker.color:.3f}<extra></extra>'}];
  } else {
    // categorical: max_go(=GO cluster), max_go_sub(=drilled-in subterm), isoform_type —
    // one trace per category, so Plotly's own legend click (toggle to 'legendonly') IS the on/off
    // selection mechanism.
    let catKeys, catLabel, catColor, catOpacity = () => .55;
    if(cb === 'max_go'){
      catKeys = (u.clusters||[]).filter(c=>c.count>0).map(c=>String(c.id));
      // id=-1 = non-coding pseudo-cluster (data-missing, not a functional cluster — no "#N" prefix,
      // fixed grey, excluded from the normal palette cycle so it never collides with a real cluster).
      catLabel = k => { const m=(u.color_meta||{})[k]; return k==='-1' ? (m?m.label:'non-coding') : (m ? `#${+k+1} ${m.label}` : 'Cluster '+(+k+1)); };
      catColor = (k,i) => k==='-1' ? T.grid2 : UMAP_CLUSTER_PAL[i % UMAP_CLUSTER_PAL.length];
    } else if(cb === 'max_go_sub'){
      // 2-level drill-down: up to N_SUBTERMS individual GO-term categories (own cluster's members
      // only, ranked by within-cluster z-score) + 'etc_go' (member, but its own true top-scoring
      // GO term didn't make the top-N — was silently mis-bucketed into one of the N before this
      // fix) + 'other' (rest of the cloud entirely, dimmed). Same fixed axis range as the
      // top-level view, only colour changes.
      const meta = u.color_meta || {};
      const goKeys = Object.keys(meta).filter(k => k !== 'other' && k !== 'etc_go');
      const subPal = distinctPalette(goKeys.length);
      catKeys = goKeys.concat(meta.etc_go ? ['etc_go'] : []).concat(meta.other ? ['other'] : []);
      catLabel = k => k === 'other' ? 'other clusters' : k === 'etc_go' ? meta.etc_go.label : `${meta[k].label} (z=${meta[k].z})`;
      catColor = (k,i) => k === 'other' ? T.grid2 : k === 'etc_go' ? T.amber : subPal[goKeys.indexOf(k)];
      catOpacity = k => k === 'other' ? .12 : k === 'etc_go' ? .22 : .68;
    } else {   // isoform_type
      catKeys = [...new Set(u.color)];
      catLabel = k => k;
      catColor = k => (u.color_meta||{})[k] || T.trace;
    }
    traces = catKeys.map((k,i) => {
      const idx = u.color.map((c,j)=>String(c)===k?j:-1).filter(j=>j>=0);
      if(!idx.length) return null;
      _umapTraceKeys.push(k);
      const nm = catLabel(k);
      return {type:'scattergl', mode:'markers', name:nm,
        x: idx.map(j=>u.x[j]), y: idx.map(j=>u.y[j]),
        text: idx.map(j=>u.isoform_id[j]),
        marker:{size:3.5, opacity:catOpacity(k), color:catColor(k,i)},
        hovertemplate:'%{text}<br>'+escapeHtml(nm)+'<extra></extra>'};
    }).filter(Boolean);
  }

  if(cb === 'max_go'){
    // leader-line labels (no background box) — text sits pushed away from the cloud centre,
    // a thin coloured line (matching that cluster's own colour) connects it back to the
    // actual centroid, so text never sits directly on top of the points underneath it.
    // non-coding(id=-1)은 기능 클러스터가 아니라 데이터 결측이라 leader-line 라벨 대상에서 제외
    // (legend 토글로는 여전히 켜고 끌 수 있음, catKeys에 남아있음).
    (u.clusters||[]).filter(c=>c.count>0 && c.id!==-1).forEach((c,i) => {
      const color = UMAP_CLUSTER_PAL[i % UMAP_CLUSTER_PAL.length];
      let dx = c.x - cx0, dy = c.y - cy0;
      const mag = Math.hypot(dx, dy) || 1;
      dx /= mag; dy /= mag;
      _umapClusterAnnotations.push({
        x:c.x, y:c.y, ax: dx*88, ay: -dy*88, axref:'pixel', ayref:'pixel', standoff:6,
        text:`#${c.id+1} ${escapeHtml(c.label)}`,
        showarrow:true, arrowhead:0, arrowwidth:1, arrowcolor:color,
        font:{size:9,color:color}, bgcolor:'rgba(0,0,0,0)', bordercolor:'transparent',
      });
      _umapClusterKeys.push(String(c.id));
    });
  }

  const gd = document.getElementById('figxUmapCanvas');
  const showLabels = document.getElementById('umapLabelToggle').checked;
  // margins are FIXED and identical across every colour-by mode (autoexpand:false) — reserving
  // room for both the bottom legend (categorical modes) and the right colorbar (max_score) at all
  // times, whether or not that mode is currently showing one, so the plotted-point rectangle never
  // resizes/shifts when switching mode (previously Plotly auto-expanded margins per-mode, which
  // visibly nudged the whole plot on every toggle).
  Plotly.newPlot(gd, traces, inst({height:UMAP_PLOT_HEIGHT, margin:{l:12,r:70,t:10,b:UMAP_LEGEND_MARGIN_B,autoexpand:false},
    showlegend: cb !== 'max_score', legend:{font:{size:9,color:T.dim}, orientation:'h', y:-0.08},
    annotations: (cb === 'max_go' && showLabels) ? _umapClusterAnnotations : [],
    xaxis:{showticklabels:false, showgrid:false, zeroline:false, range:xRange, autorange:false},
    yaxis:{showticklabels:false, showgrid:false, zeroline:false, range:yRange, autorange:false}}), PLOT_CFG);

  // legend click toggles a trace's visibility (Plotly's default) — after it settles, re-filter
  // the leader-line labels to only the clusters that are still "colour on".
  gd.removeAllListeners && gd.removeAllListeners('plotly_restyle');
  gd.on('plotly_restyle', () => updateClusterLabelVisibility(gd));

  // double-click a legend entry → Plotly's native isolate-trace behaviour (show only this one,
  // legendonly the rest) — previously intercepted to trigger the module→GO-term drill-down
  // instead, which testing found unintuitive; drill-down is now a separate explicit button
  // (viewModuleGoTerms(), wired below umapCtrl) that acts on whichever module(s) are currently
  // "colour on" in the legend, however that state was reached (single click, double-click
  // isolate, or left alone). No listener needed here — native behaviour already fires
  // plotly_restyle, which the listener above already uses to keep cluster labels in sync.
}

// keeps cluster labels in sync with which legend entries are currently on — a label only shows
// while its cluster's colour is on, per request (was previously always-on regardless of legend).
function updateClusterLabelVisibility(gd){
  if(_umapColorBy !== 'max_go' || !gd || !gd.data) return;
  if(!document.getElementById('umapLabelToggle').checked){ return; }   // fully hidden by the checkbox already
  const visible = new Set(gd.data.map((tr,i)=> tr.visible === 'legendonly' ? null : _umapTraceKeys[i]).filter(v=>v!=null));
  const anns = _umapClusterAnnotations.filter((a,i) => visible.has(_umapClusterKeys[i]));
  Plotly.relayout(gd, {annotations: anns});
}

// ── "View GO terms in this module" — reads which legend entries are currently "colour on" (same
// mechanism viewUmapSelection uses) and, if that's exactly one large-enough/drillable module,
// enters the module→GO-term drill-down for it. Replaces the old legend-double-click trigger
// (unintuitive — double-click now does Plotly's native isolate instead, see drawUmap).
function viewModuleGoTerms(){
  if((_umapColorBy !== 'max_go' && !_umapScoreOn) || _umapSubclusterId != null) return;
  const cap = document.getElementById('figx-cap');
  // score-gradient mode has no per-module legend trace (single continuous scatter) — selection
  // there comes from the module-filter chips instead (_umapModuleOn), same "exactly one on" rule.
  let onKeys;
  if(_umapScoreOn){
    onKeys = _umapModuleOn ? Array.from(_umapModuleOn) : [];
  } else {
    const gd = document.getElementById('figxUmapCanvas');
    if(!gd || !gd.data) return;
    onKeys = gd.data
      .map((tr, i) => (tr.visible === 'legendonly') ? null : _umapTraceKeys[i])
      .filter(v => v != null);
  }
  if(onKeys.length === 0){
    cap.textContent = 'no module is colour-on — click a legend entry (or double-click to isolate one) first';
    return;
  }
  if(onKeys.length > 1){
    cap.textContent = `${onKeys.length} modules are colour-on — isolate exactly one (double-click its `+
      `legend entry, or click the others off) to view its GO terms`;
    return;
  }
  const meta = _umapClustersMeta.find(c => String(c.id) === String(onKeys[0]));
  if(!meta || meta.count < UMAP_SUBCLUSTER_MIN_COUNT){
    cap.textContent = meta ? `cluster #${meta.id+1} too small to drill down (n=${meta.count} < ${UMAP_SUBCLUSTER_MIN_COUNT})` : '';
    return;
  }
  if(meta.id === -1){
    cap.textContent = `non-coding isoforms have no ESM-2 signal (zero-embedding by design) — there is no `+
      `subcategory structure to drill into, this isn't a functional cluster`;
    return;
  }
  if(!meta.drillable){
    cap.textContent = `cluster #${meta.id+1} has no real subcategory structure (entropy=${meta.subterm_entropy.toFixed(2)} `+
      `bits) — its members share near-identical predictions, likely a model prediction-collapse `+
      `artifact rather than genuine functional diversity`;
    return;
  }
  _umapSubclusterId = meta.id;
  loadUmap();
}

// ── "Selected isoform list" — reads which legend entries are currently ON (not toggled to
// 'legendonly') and queries the union of those. Deliberately NOT wired to the legend click
// itself — selecting and listing are separate actions. ──
async function viewUmapSelection(){
  // three distinct sources depending on current mode: the dropdown's own continuous max_score view
  // (threshold slider), the score-overlay's module/subterm chips (_umapModuleOn, no Plotly legend
  // exists there — single continuous trace), or the categorical Plotly legend (default). Was
  // previously gated on _umapEffectiveColorBy() === 'max_score', which that function could never
  // actually return (it always funnelled max_score → max_go for fetch purposes) — dead code, the
  // threshold-slider selection path was unreachable even in the dropdown's own max_score mode.
  if(_umapColorBy === 'max_score' && !_umapScoreOn){
    await loadUmapIsoforms('max_score', [document.getElementById('umapScoreThr').value]);
    return;
  }
  let values;
  if(_umapScoreOn){
    values = _umapModuleOn ? Array.from(_umapModuleOn) : [];
  } else {
    const gd = document.getElementById('figxUmapCanvas');
    if(!gd || !gd.data){ return; }
    values = gd.data
      .map((tr, i) => (tr.visible === 'legendonly') ? null : _umapTraceKeys[i])
      .filter(v => v != null);
  }
  if(!values.length){
    const panel = document.getElementById('umapIsoPanel');
    panel.classList.remove('hidden');
    document.getElementById('umapIsoTitle').textContent = '§ nothing selected — turn a category on in the legend first';
    document.getElementById('umapIsoTbody').innerHTML = '';
    return;
  }
  await loadUmapIsoforms(_umapEffectiveColorBy(), values);
}

async function loadUmapIsoforms(colorBy, values){
  _umapFilter = {colorBy, values};
  const panel = document.getElementById('umapIsoPanel'), tbody = document.getElementById('umapIsoTbody'),
        title = document.getElementById('umapIsoTitle');
  panel.classList.remove('hidden');
  title.textContent = '§ loading…';
  tbody.innerHTML = '';
  let url = '/api/umap/isoforms?color_by='+encodeURIComponent(colorBy)+
    '&values='+encodeURIComponent(values.join(','))+'&space='+encodeURIComponent(_umapSpace);
  if(colorBy === 'max_go_sub') url += '&cluster_id='+encodeURIComponent(_umapSubclusterId);
  const r = await fetch(url);
  const d = await r.json();
  if(d.error){ title.textContent = '✕ '+d.error; return; }
  title.textContent = `§ ${d.label} · ${d.n.toLocaleString()} isoform(s)`;
  tbody.innerHTML = d.isoforms.map(iso => `
    <tr onclick="location.href='/gene/'+encodeURIComponent('${iso.isoform_id}')" title="probe in individual analysis ▸">
      <td>${iso.gene_id}${geneCardsLink(iso.gene_id)}</td><td>${iso.isoform_id} <span class="muted">▸</span></td>
      <td class="muted">${iso.max_go_name}</td><td>${iso.max_score.toFixed(3)}</td>
      <td>${iso.isoform_type}</td></tr>`).join('');
}

function hideUmapIsoPanel(){
  document.getElementById('umapIsoPanel').classList.add('hidden');
  _umapFilter = null;
}

const umapSpaceSel = document.getElementById('umapSpace');
if(umapSpaceSel){
  umapSpaceSel.addEventListener('change', e => {
    _umapSpace = e.target.value;
    _umapSubclusterId = null;
    loadUmap();
  });
}
const umapColorBySel = document.getElementById('umapColorBy');
if(umapColorBySel){
  umapColorBySel.addEventListener('change', e => {
    _umapColorBy = e.target.value;
    _umapSubclusterId = null;
    // the dropdown's own 'max_score' already IS a continuous score view — the overlay toggle would
    // be redundant/conflicting on top of it, so it's forced off (and disabled, see _umapSyncCtrlUI).
    if(_umapColorBy === 'max_score') _umapScoreOn = false;
    loadUmap();
  });
}
// top-right "score gradient" — independent overlay on top of whichever colour-by is active, not a
// colour-by value itself (see _umapScoreOn declaration above). Toggling it never refetches: every
// /api/umap payload already carries cluster_id/clusters/max_score regardless of what colour_by was
// requested, so this just re-renders the already-cached response through the other branch of
// drawUmap() — no network round trip, no lost categorical view, no more "feels like a reload".
const umapScoreToggle = document.getElementById('umapScoreToggle');
if(umapScoreToggle){
  umapScoreToggle.addEventListener('change', () => {
    _umapScoreOn = umapScoreToggle.checked;
    if(!_umapScoreOn && _umapSubclusterId != null && _umapColorBy !== 'max_go'){
      // drilled in while viewing some other colour-by + the overlay — drilling is inherently a
      // module concept, so leaving the overlay (with a non-module colour-by underneath) also leaves
      // the drill-down; this is the one case that still needs a fresh fetch.
      _umapSubclusterId = null;
      loadUmap();
      return;
    }
    _umapSyncCtrlUI();
    if(_umapLastResponse) drawUmap(_umapLastResponse);
  });
}
const umapBreadcrumbBack = document.getElementById('umapBreadcrumbBack');
if(umapBreadcrumbBack){
  umapBreadcrumbBack.addEventListener('click', () => {
    _umapSubclusterId = null;
    loadUmap();
  });
}
const umapLabelToggle = document.getElementById('umapLabelToggle');
if(umapLabelToggle){
  umapLabelToggle.addEventListener('change', () => {
    const gd = document.getElementById('figxUmapCanvas');
    if(!gd) return;
    if(umapLabelToggle.checked) updateClusterLabelVisibility(gd);
    else Plotly.relayout(gd, {annotations: []});
  });
}
const umapScoreThr = document.getElementById('umapScoreThr');
if(umapScoreThr){
  // instant visual feedback — dim points below the threshold directly on the UMAP as the
  // slider moves, rather than only reflecting the change once "Selected isoform list" is pressed.
  umapScoreThr.addEventListener('input', () => {
    const thr = +umapScoreThr.value;
    document.getElementById('umapScoreThrVal').textContent = thr.toFixed(2);
    if(_umapColorBy !== 'max_score' || !_umapRawScores) return;
    const gd = document.getElementById('figxUmapCanvas');
    if(!gd || !gd.data) return;
    const opacity = _umapRawScores.map(s => s >= thr ? 0.75 : 0.05);
    Plotly.restyle(gd, {'marker.opacity': [opacity]}, [0]);
  });
}
const umapListBtn = document.getElementById('umapListBtn');
if(umapListBtn) umapListBtn.addEventListener('click', viewUmapSelection);
const umapModuleGoBtn = document.getElementById('umapModuleGoBtn');
if(umapModuleGoBtn) umapModuleGoBtn.addEventListener('click', viewModuleGoTerms);
const umapDlBtn = document.getElementById('umapDownloadBtn');
if(umapDlBtn) umapDlBtn.addEventListener('click', () => {
  if(!_umapFilter) return;
  let url = '/api/umap/isoforms?color_by='+encodeURIComponent(_umapFilter.colorBy)+
    '&values='+encodeURIComponent(_umapFilter.values.join(','))+'&space='+encodeURIComponent(_umapSpace)+'&format=csv';
  if(_umapFilter.colorBy === 'max_go_sub') url += '&cluster_id='+encodeURIComponent(_umapSubclusterId);
  window.location.href = url;
});

function drawBubble(b){
  const host = document.getElementById('figxPlot');
  host.innerHTML = '<div id="figxBubbleCanvas"></div>';
  const mods = b.modules||[];
  if(!mods.length){ host.innerHTML = '<p class="muted mono sm">module reference data unavailable</p>'; return; }
  const cats = [...new Set(mods.map(m=>m.category))];
  const traces = cats.map(cat=>{
    const rows = mods.filter(m=>m.category===cat);
    return {type:'scatter', mode:'markers+text', name:cat,
      x: rows.map(m=>m.mean_auprc), y: rows.map(m=>m.enrichment),
      text: rows.map(m=>m.label), textposition:'top center', textfont:{size:8,color:T.dim},
      marker:{size: rows.map(m=>Math.max(6, Math.sqrt(m.n_isoforms))), sizemode:'diameter',
        color: rows[0].color, opacity:.82, line:{color:T.ink900,width:1}},
      customdata: rows.map(m=>[m.module_id, m.n_isoforms, m.novel_pct]),
      hovertemplate:'%{text}<br>AUPRC %{x:.3f} · enrichment %{y:.2f}<br>'+
        'n=%{customdata[1]} · novel %{customdata[2]:.1f}%<extra></extra>'};
  });
  const bg = (b.meta&&b.meta.background_novel_frac)||0;
  Plotly.newPlot('figxBubbleCanvas', traces, inst({height:560, margin:{l:60,r:14,t:10,b:44},
    showlegend:true, legend:{font:{size:9,color:T.dim}, orientation:'h', y:-0.14},
    shapes:[{type:'line', x0:0,x1:1,xref:'paper', y0:0,y1:0, line:{color:T.grid2,width:1,dash:'dash'}}],
    annotations:[{x:0.02,y:0,xref:'paper',yref:'y',text:`background novel ${bg.toFixed(1)}%`,
      showarrow:false, font:{size:9,color:T.dim}, xanchor:'left', yanchor:'bottom'}],
    xaxis:{title:{text:'mean brain AUPRC (zero-shot prediction quality)',font:{size:10,color:T.dim}}},
    yaxis:{title:{text:'novel(NIC+NNIC) enrichment vs background',font:{size:10,color:T.dim}}}}), PLOT_CFG);
}

function drawCorr(c){
  const host = document.getElementById('figxPlot');
  host.innerHTML = '<div id="figxCorrCanvas"></div>';
  const shapes = (c.module_boundaries||[]).flatMap(b=>[
    {type:'line', x0:b,x1:b, y0:0,y1:c.n_shown-1, line:{color:T.grid2,width:.5}},
    {type:'line', y0:b,y1:b, x0:0,x1:c.n_shown-1, line:{color:T.grid2,width:.5}},
  ]);
  // diverging: blue↔vermillion (DIVERGE_POS/NEG) instead of red-blue/red-green, per the
  // colourblind-safe diverging convention — neutral/white stop placed at the actual r=0 fraction
  // of the [zmin,zmax] range (0.5/1.5=0.333), not the scale midpoint (r=0 isn't centred in
  // [-0.5, 1.0]).
  Plotly.newPlot('figxCorrCanvas', [{type:'heatmap', z:c.z, zmin:-0.5, zmax:1.0,
    colorscale:[[0,DIVERGE_POS],[0.333,'#F0F0EC'],[1,DIVERGE_NEG]],
    colorbar:{title:{text:'Pearson r',font:{size:9,color:T.dim}}, thickness:10,
      tickfont:{family:MONO,size:9,color:T.dim}}}],
    inst({height:600, margin:{l:12,r:12,t:10,b:12}, shapes,
      xaxis:{showticklabels:false, title:{text:'GO terms (module-sorted, '+c.n_shown+'/'+c.n_go+' shown)',font:{size:10,color:T.dim}}},
      yaxis:{showticklabels:false, autorange:'reversed'}}), PLOT_CFG);
}

const figxTabs = document.getElementById('figx-tabs');
if(figxTabs){
  figxTabs.addEventListener('click', e=>{
    const b = e.target.closest('.ttab'); if(!b) return;
    figxTabs.querySelectorAll('.ttab').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');
    loadFigx(b.dataset.fig);
  });
}

document.addEventListener('DOMContentLoaded', ()=> {
  load(document.getElementById('dash').dataset.active || 'brain_672');
  loadFigx('umap');
});
// 테마 전환 시 현재 뷰(tissue 대시보드 + 열려있는 figx)를 새 팔레트로 다시 그린다.
window.__rerenderCharts = () => { load(document.getElementById('dash').dataset.active || 'brain_672'); loadFigx(_figxKind); };
