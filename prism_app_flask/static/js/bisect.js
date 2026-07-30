// BISECT 케이스 탐색기 — 필터·정렬 테이블 + 상세 drawer (개별분석 교차링크).
'use strict';

let CASES = [], SUM = {}, sortKey = 'tier', sortDir = 1, _openCase = null;
// tier 우선순위(A 계열이 최상위 evidence) — 기본 정렬 근거
const TIER_RANK = {'A-DR':0, 'A-BP':1, 'B':2, 'C':3, 'D':4};

// tier → T 토큰 매핑 — 서버 고정 hex(bisect_cases.py TIER_META) 대신 테마 토큰으로 렌더.
// light 테마의 --signal/--amber/--trace 는 이미 흰 배경 대비를 위해 더 어둡게 정의돼 있어
// (main.css :root[data-theme="light"]) 이 매핑만으로 두 테마 모두 가독성이 확보된다.
const TIER_COLOR_KEY = {'A-DR':'signal', 'A-BP':'amber', 'B':'trace', 'C':'dim', 'D':'grid2'};
function tierBadge(t){
  const col = T[TIER_COLOR_KEY[t]] || T.dim;
  return `<span class="tier-badge" style="border-color:${col};color:${col}">${t||'—'}</span>`;
}
function deltaBar(d){
  const w = Math.min(Math.abs(d)*100, 100), col = d>=0 ? T.signal : T.trace;
  return `<span class="dbar"><i style="width:${w}%;background:${col}"></i></span><span class="tnum">${d.toFixed(2)}</span>`;
}
// 도메인 획득(+, 초록)/소실(−, 빨강)을 개별 chip 으로 — ';' 나열을 시각적으로 분해하고
// 방향(gained/lost) 정보를 명시한다.
function domainChips(c){
  const split = s => String(s||'').split(';').map(x=>x.trim()).filter(Boolean);
  const lost = split(c.domains_lost), gained = split(c.domains_gained);
  if(!lost.length && !gained.length) return '<span class="muted">—</span>';
  const chip = (name, cls, sign) =>
    `<span class="dom-tag ${cls}" title="${sign==='+'?'gained':'lost'} domain: ${name}">${sign} ${name}</span>`;
  return lost.map(d=>chip(d,'lost','−')).join('') + gained.map(d=>chip(d,'gain','+')).join('');
}

async function boot(){
  const r = await (await fetch('/api/bisect/cases')).json();
  CASES = r.cases; SUM = r.summary;
  document.getElementById('bi-readout').innerHTML =
    `<span><span class="k">CASES</span> <b>${SUM.n}</b></span>`+
    `<span><span class="k">TIERS</span> <b>${Object.entries(SUM.tiers).map(([k,v])=>k+':'+v).join(' · ')}</b></span>`+
    `<span><span class="k">CELL TYPES</span> <b>${Object.keys(SUM.cell_types).length}</b></span>`;
  fill('f-cell', SUM.cell_types); fill('f-tier', SUM.tiers); fill('f-mech', SUM.mechanisms);
  ['bi-search','f-cell','f-tier','f-mech'].forEach(id=>
    document.getElementById(id).addEventListener('input', render));
  document.querySelectorAll('thead th[data-s]').forEach(th=>th.addEventListener('click',()=>{
    const k=th.dataset.s; sortDir = (sortKey===k)? -sortDir : 1; sortKey=k; render();
  }));
  render();
  drawMechCell(SUM.mech_cell_rows||[]);
}

// 세포유형 palette — 카테고리 색은 테마와 무관하게 고정(다른 페이지의 다범주 팔레트와 동일 관례).
// Okabe-Ito(instrument.js distinctPalette) 로 통일 — mydata.js 의 UMAP_CLUSTER_PAL 과 동일 소스.
const CELLTYPE_PAL = distinctPalette(8);

// ── Cases by mechanism × cell type — streamlit 00_hub.py 이관 (§00 overview) ──
function drawMechCell(rows){
  const host=document.getElementById('mechCellBar'); if(!host) return;
  if(!rows.length){ host.innerHTML='<p class="muted mono sm">no mechanism data</p>'; return; }
  const mechs=[...new Set(rows.map(r=>r.mechanism))];
  const cellTypes=[...new Set(rows.map(r=>r.cell_type))];
  // mechanism 별 총합 내림차순 — 가장 흔한 기전이 위로
  const totals=Object.fromEntries(mechs.map(m=>[m, rows.filter(r=>r.mechanism===m).reduce((s,r)=>s+r.n,0)]));
  mechs.sort((a,b)=>totals[b]-totals[a]);
  const traces=cellTypes.map((ct,i)=>({
    type:'bar', orientation:'h', name:ct,
    y:mechs, x:mechs.map(m=>{ const r=rows.find(r=>r.mechanism===m&&r.cell_type===ct); return r?r.n:0; }),
    marker:{color:CELLTYPE_PAL[i%CELLTYPE_PAL.length]},
    hovertemplate:'%{y}<br>'+ct+': %{x} case(s)<extra></extra>',
  }));
  Plotly.newPlot('mechCellBar', traces, inst({
    barmode:'stack', height:260, margin:{l:150,r:14,t:10,b:34},
    xaxis:{title:{text:'cases',font:{size:10,color:T.dim}}},
    yaxis:{tickfont:{family:MONO,size:10,color:T.dim}, automargin:true},
    legend:{orientation:'h', y:-0.22, font:{size:9,color:T.dim}},
    showlegend:true,
  }), PLOT_CFG);
}

function fill(id, counts){
  const el=document.getElementById(id);
  Object.entries(counts).forEach(([k,v])=>{
    const o=document.createElement('option'); o.value=k; o.textContent=`${k} (${v})`; el.appendChild(o);
  });
}

function render(){
  const q=document.getElementById('bi-search').value.trim().toUpperCase();
  const fc=document.getElementById('f-cell').value, ft=document.getElementById('f-tier').value,
        fm=document.getElementById('f-mech').value;
  let rows = CASES.filter(c=>
    (!q || (c.gene||'').toUpperCase().includes(q)) &&
    (!fc || c.cell_type===fc) && (!ft || c.tier===ft) && (!fm || c.mechanism===fm));
  rows.sort((a,b)=>{
    if(sortKey==='tier'){
      // tier-A 우선 → 동일 tier 내 |ΔIF| 내림차순
      const r=(TIER_RANK[a.tier]??9)-(TIER_RANK[b.tier]??9);
      if(r) return r*sortDir;
      return (Math.abs(b.delta)-Math.abs(a.delta));
    }
    let x=a[sortKey], y=b[sortKey];
    if(typeof x==='number') return (x-y)*sortDir;
    return String(x||'').localeCompare(String(y||''))*sortDir;
  });
  document.getElementById('bi-count').textContent = `${rows.length} / ${CASES.length} cases`;
  document.getElementById('bi-tbody').innerHTML = rows.map(c=>`
    <tr onclick='showCase(${JSON.stringify(c.gene)}, ${JSON.stringify(c.cell_type)})'>
      <td>${c.gene}${geneCardsLink(c.gene)}</td><td class="muted">${c.cell_type}</td><td>${tierBadge(c.tier)}</td>
      <td>${deltaBar(c.delta)}</td><td class="muted">${c.mechanism||'—'}</td>
      <td class="bi-domains">${domainChips(c)}</td>
    </tr>`).join('');
}

function fmt(v, dec){ return (v===null||v===undefined||v==='')?'—':(typeof v==='number'?v.toFixed(dec||2):v); }
function txLink(id){ return id?`<a class="tx" href="/gene/${encodeURIComponent(id)}">${id} ▸</a>`:'—'; }

// GO 리스트(dict 배열)를 미니 bar 로 렌더 — R6a 버그 수정(스칼라→리스트)
function goList(arr, key){
  if(!Array.isArray(arr) || !arr.length) return '<span class="muted">—</span>';
  return arr.slice(0,5).map(g=>{
    const v = key ? g[key] : g.score;
    const w = Math.min(Math.abs(v)*100, 100);
    const col = (key==='delta') ? (v>=0?T.ok:T.warn) : T.signal;
    const val = (key==='delta'?(v>=0?'+':''):'')+ (v!=null?v.toFixed(2):'—');
    return `<div class="goli"><span class="gname">${g.go_name||g.go_id}</span>
      <span class="gbar"><i style="width:${w}%;background:${col}"></i></span>
      <span class="gval tnum">${val}</span></div>`;
  }).join('');
}

function chip(txt, col){ return txt&&txt!=='—' ? `<span class="bd-chip"${col?` style="border-color:${col};color:${col}"`:''}>${txt}</span>` : ''; }
function pval(p){ if(p==null||p==='') return '—'; const n=+p; return n<1e-3 ? n.toExponential(1) : n.toFixed(4); }
// ';' 로 구분된 원본 문자열(예: af_gained_confident "Clathrin (pLDDT=72.97);TPR_14 (...)")을
// 그대로 노출하지 않고 한 줄씩 나눠 보여준다 — bd-cell 안에서 세미콜론이 그대로 보이던 문제.
function semiList(s){
  const items = String(s||'').split(';').map(x=>x.trim()).filter(Boolean);
  if(!items.length) return '<span class="muted">—</span>';
  return `<div class="bd-semilist">${items.map(x=>`<span class="bd-semi-item">${x}</span>`).join('')}</div>`;
}
// domain 목록(name+func) → bd-grid 셀 — bio_report.domains 가 유일 출처(narrative 에서는 중복 제거).
// 세미콜론 나열 대신 semiList 와 같은 줄바꿈 포맷으로 통일.
function domainFuncList(arr){
  if(!Array.isArray(arr) || !arr.length) return '<span class="muted">—</span>';
  return `<div class="bd-semilist">${arr.map(d=>
    `<span class="bd-semi-item">${d.name} <span class="muted sm">(${d.func})</span></span>`).join('')}</div>`;
}

// ── 생물학적 기능 예측 리포트 (Streamlit BISECT 서사 리포트 이관·정제) ────────
// 중복 제거 원칙: 숫자·사실은 위쪽 bd-grid 에 한 번만 — 여기 narrative 는 해석 절만 남긴다.
// headline 이 맨 위 한 줄 결론(abstract 처럼 먼저 읽히는) 역할, narrative 는 라벨별 짧은 행.
function bioReport(r){
  if(!r) return '';
  const conf = r.confidence||{}, tier = r.tier||{};
  const pathway = (r.pathway||[]).map((s,i)=>
    `<span class="br-step">${s}</span>${i<r.pathway.length-1?'<span class="br-arrow">→</span>':''}`).join('');
  const regs = (r.regulators||[]).map(g=>{
    // blue/vermillion, not green/red — colourblind-safe diverging pair (also redundantly
    // encoded by the ↑/↓ glyph itself, but the colour should still be safe on its own).
    const up = g.direction==='up', col = up?DIVERGE_POS:DIVERGE_NEG, arr = up?'↑':'↓';
    return `<div class="br-reg" style="border-left-color:${col}" title="${g.desc||''}">
      <b>${g.gene}</b>${g.known?'':' <span class="br-novel">●</span>'}
      <span class="br-cat">[${g.cat}]</span>
      <span class="br-lfc" style="color:${col}">${arr} ${(+g.logFC).toFixed(3)}</span>
      <span class="muted br-p">−log10p ${(+g.neg_log10_padj).toFixed(1)}</span></div>`;
  }).join('') || '<div class="muted mono sm">no regulator data</div>';
  const ctx = r.context||{}, m = ctx.mech||{};
  const ctxRows = [
    ctx.tss ? `TSS <b>${ctx.tss}</b>${ctx.tss_bp?` (${ctx.tss_bp>0?'+':''}${ctx.tss_bp}bp)`:''}` : '',
    ctx.apa ? `APA <b>${ctx.apa}</b>${ctx.apa_bp?` (${ctx.apa_bp>0?'+':''}${ctx.apa_bp}bp)`:''}` : '',
    m.label ? `mechanism <b style="color:${m.color}">${m.label}</b>` : '',
  ].filter(Boolean).map(x=>`<div class="br-ctx-row">${x}</div>`).join('');
  const narrative = (r.narrative||[]).map(n=>`<div class="br-note">
      <span class="br-note-ico" title="${n.label}">${icon(n.icon)}</span>
      <span class="br-note-label">${n.label}</span>
      <span class="br-note-text">${n.text}</span></div>`).join('')
      || '<p class="muted mono sm">no additional interpretive notes</p>';
  const tierIcon = tier.icon ? `<span class="br-tier-ico">${icon(tier.icon)}</span>` : '';

  return `<div class="bio-report">
    <div class="br-head">
      <span class="br-title">${icon('clipboard-list')} Biological Function Prediction Report · Integrated Analysis</span>
      <span class="br-badges">
        <span class="br-tier" style="background:${tier.color}">${tierIcon}${tier.label}</span>
        <span class="br-conf" style="background:${conf.color}" title="evidence modules ${conf.n}/${conf.max}">confidence: ${conf.label}</span>
      </span>
    </div>

    ${r.headline ? `<div class="br-headline">${r.headline}</div>` : ''}

    <div class="br-pathway"><div class="br-sub">${icon('telescope')} Isoform-Switch Causal Pathway</div>
      <div class="br-path-flow">${pathway}</div></div>

    <div class="br-2col">
      <div><div class="br-sub">TF / ASF Activity Change (AD vs CT)</div>${regs}
        <div class="muted mono sm" style="margin-top:.35rem">● = newly identified regulator · ↑/↓ = increased/decreased in AD</div></div>
      <div><div class="br-sub">Promoter · APA Context</div>${ctxRows||'<div class="muted mono sm">—</div>'}
        ${m.detail?`<div class="br-mech-detail">${m.detail}</div>`:''}</div>
    </div>

    <div class="br-narrative"><div class="br-sub">${icon('lightbulb')} Interpretive Notes</div>${narrative}</div>
    <div class="br-foot">PRISM+BISECT auto-generated · Lee et al. (2026) · requires experimental validation</div>
  </div>`;
}

async function showCase(gene, cell){
  _openCase = {gene, cell};   // 테마 전환 시 재렌더 대상
  const c = await (await fetch(`/api/bisect/case/${encodeURIComponent(gene)}?cell_type=${encodeURIComponent(cell)}`)).json();
  const el=document.getElementById('bi-detail');
  if(c.error){ el.innerHTML=`<p class="err">${c.error}</p>`; return; }

  // mechanism summary chips
  const chips = [
    chip(c.m16_mechanism, '#C99A52'), chip(c.m14_event_type), chip(c.mechanism_type),
    c.m15_nmd_switch?chip('NMD switch','#FF4D6D'):'', c.stage2_pass?chip('stage-2 pass','#4FC08A'):'',
    (c.prism_match_ct==='exact'&&c.prism_match_ad==='exact')?chip('PRISM exact-match','#4FB4D0'):''
  ].filter(Boolean).join('');

  el.innerHTML = `
    <div class="bd-head"><h2>${c.gene}</h2>${geneCardsLink(c.gene)}${tierBadge(c.bisect_tier)}
      <span class="badge">${c.cell_type}</span></div>
    <div class="bd-chips">${chips}</div>

    <div class="bd-switch">
      <div class="sw ct"><div class="sw-k">CT · control isoform</div>${txLink(c.ct_transcript_id)}
        <div class="mono sm">top GO · ${fmt(c.prism_ct_max_go)} (${fmt(c.prism_ct_max_score,3)})</div>
        <div class="mono sm">${fmt(c.ct_transcript_id? (tracks_len(c,'ct')) : '')}</div></div>
      <div class="sw-arrow">→<br>ΔIF ${fmt(c.delta,2)}<br><span class="muted sm">ΔGO ${fmt(c.prism_delta_min,2)}…${fmt(c.prism_delta_max,2)}</span></div>
      <div class="sw ad"><div class="sw-k">AD · switch isoform</div>${txLink(c.ad_transcript_id)}
        <div class="mono sm">top GO · ${fmt(c.prism_ad_max_go)} (${fmt(c.prism_ad_max_score,3)})</div>
        <div class="mono sm">${fmt(c.ad_transcript_id? (tracks_len(c,'ad')) : '')}</div></div>
    </div>

    <div class="bd-sec">Isoform usage (DTU) · dominant isoform per condition</div>
    ${dtuSection(c.dtu, c.cell_type)}

    <div class="bd-fullrow"><div class="bd-sec">IDR &amp; domain architecture · CT vs AD</div><div id="bd-track"></div>
      <div class="mono sm muted">line = per-residue disorder (metapredict) · boxes = Pfam domains · CT top / AD bottom</div></div>
    <div class="bd-fullrow"><div class="bd-sec">Regulatory volcano · SF / TF activity (AD vs CT)</div><div id="bd-volcano"></div>
      <div class="mono sm muted">${(c.regulators||[]).length} regulator(s) · top ${fmt(c.top_regulators&&c.top_regulators.gene)} (logFC ${fmt(c.top_regulators&&c.top_regulators.logFC,2)})</div></div>

    <div class="bd-sec">PRISM functional shift</div>
    <div class="bd-go2">
      <div><div class="go-k">CT top GO</div>${goList(c.prism_ct_top_go)}</div>
      <div><div class="go-k">AD top GO</div>${goList(c.prism_ad_top_go)}</div>
      <div><div class="go-k gain">gained (AD−CT)</div>${goList(c.prism_gain_go,'delta')}</div>
      <div><div class="go-k loss">lost (AD−CT)</div>${goList(c.prism_loss_go,'delta')}</div>
    </div>

    <div class="bd-sec">Domain &amp; structural consequence</div>
    <div class="bd-grid c4">
      ${cell4('domains lost', domainFuncList(c.bio_report&&c.bio_report.domains&&c.bio_report.domains.lost), c.domains_lost?DIVERGE_NEG:'')}
      ${cell4('domains gained', domainFuncList(c.bio_report&&c.bio_report.domains&&c.bio_report.domains.gained), c.domains_gained?DIVERGE_POS:'')}
      ${cell4('ΔpLDDT (AF)', fmt(c.af_delta_plddt,1))}
      ${cell4('pLDDT CT→AD', fmt(c.af_ct_plddt_mean,1)+'→'+fmt(c.af_ad_plddt_mean,1))}
      ${cell4('confident lost', semiList(c.af_lost_confident))}
      ${cell4('confident gained', semiList(c.af_gained_confident))}
      ${cell4('high-pLDDT frac', fmt(c.af_ct_plddt_high_frac,2)+'→'+fmt(c.af_ad_plddt_high_frac,2))}
      ${cell4('seq validation', fmt(c.seq_val_conclusion)||fmt(c.seq_val_identity)||'—')}
    </div>

    <div class="bd-sec">Genomic architecture · transcript event</div>
    <div class="bd-grid c4">
      ${cell4('promoter (TSS)', fmt(c.tss_class)+(c.tss_diff_bp?(' · '+c.tss_diff_bp+'bp'):''))}
      ${cell4('poly-A (TTS)', fmt(c.tts_class)+(c.tts_diff_bp?(' · '+c.tts_diff_bp+'bp'):''))}
      ${cell4('APA', fmt(c.apa_class))}
      ${cell4('NAT / antisense', c.nat?'YES':'no')}
      ${cell4('NMD (CT→AD)', (c.ct_nmd?'NMD':'stable')+'→'+(c.ad_nmd?'NMD':'stable'))}
      ${cell4('NMD relevant', c.nmd_relevant?'YES':'no')}
      ${cell4('young L1 CDS', c.young_l1_cds?'YES':'no')}
      ${cell4('DTU', 'p='+pval(c.dtu_p)+' · '+fmt(c.dtu_method))}
    </div>

    <div class="bd-sec">Conservation &amp; interaction network</div>
    <div class="bd-grid c4">
      ${cell4('conservation class', fmt(c.cons_ad_class))}
      ${cell4('phyloP (AD)', fmt(c.cons_ad_phylop,2)+(c.cons_background_phylop?(' vs bg '+fmt(c.cons_background_phylop,2)):''))}
      ${cell4('PPI verdict', fmt(c.ppi_verdict), c.ppi_verdict==='SUPPORTED'?'#4FC08A':(c.ppi_verdict==='UNSUPPORTED'?'#66728A':''))}
      ${cell4('top PPI partner', fmt(c.ppi_top_partner)+(c.ppi_top_score?(' · '+c.ppi_top_score):''))}
    </div>

    <div class="bd-sec">Mechanism of Origin &amp; Sequence-to-Function Prediction · Narrative Report</div>
    ${bioReport(c.bio_report)}`;
  drawTracks(c.tracks);
  drawVolcano(c.regulators);
  if(c.dtu){ drawDtuStack(c.dtu); drawDominantGo(c.dtu); }
  el.scrollIntoView({behavior:'smooth', block:'nearest'});
}
function tracks_len(c, k){ const t=c.tracks&&c.tracks[k]; return t&&t.len? (t.len+' aa'):''; }
function cell4(k,v,col){ return `<div class="bd-cell"><div class="bd-k">${k}</div><div class="bd-v"${col?` style="color:${col}"`:''}>${v}</div></div>`; }

// ── CT→AD DTU: 실제 조건별 아이소폼 사용률 stacked chart + dominant-isoform GO 비교 ──
// c.dtu 가 없으면(muscle 세포유형 또는 커버되지 않은 유전자) 차트 대신 솔직한 empty-state만 표시 —
// 있는 것처럼 꾸미지 않는다.
const MUSCLE_CELL_TYPES = new Set(['Cardiomyocyte', 'Skeletal_muscle']);
function dtuSection(dtu, cellType){
  if(!dtu){
    const reason = MUSCLE_CELL_TYPES.has(cellType)
      ? 'DTU breakdown not available for this cell type — no CT/AD comparison is defined for it.'
      : 'No per-isoform usage data available for this gene in the current DTU source.';
    return `<p class="muted mono sm" style="padding:.4rem 0">${reason}</p>`;
  }
  const dom = dtu.dominant || {};
  return `<div class="bd-2col">
    <div><div id="bd-dtustack"></div></div>
    <div><div id="bd-domgo"></div>
      <div class="mono sm muted">dominant isoform · CT ${dom.ct?dom.ct.id:'—'} (${dom.ct?(dom.ct.frac*100).toFixed(1):'—'}%)
        · AD ${dom.ad?dom.ad.id:'—'} (${dom.ad?(dom.ad.frac*100).toFixed(1):'—'}%)
        ${dom.ct && dom.ad && dom.ct.id===dom.ad.id ? '<br><span class="sig">same isoform dominant in both conditions — usage share shifts, function does not</span>' : ''}</div></div>
  </div>`;
}

const ISO_PAL = distinctPalette(8);   // Okabe-Ito, same source as CELLTYPE_PAL/UMAP_CLUSTER_PAL
function drawDtuStack(dtu){
  const host=document.getElementById('bd-dtustack'); if(!host) return;
  const isos = dtu.isoforms||[];
  if(!isos.length){ host.innerHTML='<p class="muted mono sm">no isoform usage data</p>'; return; }
  const traces = isos.map((iso,i)=>({
    type:'bar', x:['CT','AD'], y:[iso.ct_frac, iso.ad_frac], name:iso.id,
    marker:{color:ISO_PAL[i%ISO_PAL.length]},
    hovertemplate:iso.id+'<br>%{x}: %{y:.1%}<extra></extra>',
  }));
  Plotly.newPlot('bd-dtustack', traces, inst({
    barmode:'stack', height:280, margin:{l:44,r:14,t:46,b:34},
    title:{text:'isoform usage · χ² p = '+pval(dtu.chi_pval)+' · FDR q = '+pval(dtu.chi_padj),
           font:{size:10,color:T.dim}},
    yaxis:{title:{text:'usage fraction',font:{size:10,color:T.dim}}, tickformat:'.0%', range:[0,1]},
    xaxis:{tickfont:{family:MONO,size:11,color:T.txt}},
    legend:{orientation:'h', y:-0.2, font:{size:8,color:T.dim}},
    showlegend: isos.length<=8,
  }), PLOT_CFG);
}

// dominant isoform 이 CT/AD 에서 다를 수 있다(진짜 발견) — CT-dominant top-3 ∪ AD-dominant
// top-3 GO(최대 6개)에 대해, 두 이소폼 각각의 *실제* score 를 나란히 비교한다. 어느 한쪽의
// top-3 에만 들었다고 다른 쪽을 0으로 채우지 않는다(go_compare 는 이미 전체 score row에서
// 직접 조회된 값 — precompute/build_bisect_dtu.py 참조).
function drawDominantGo(dtu){
  const host=document.getElementById('bd-domgo'); if(!host) return;
  const dom = dtu.dominant||{}, cmp = dom.go_compare||[];
  if(!cmp.length){ host.innerHTML='<p class="muted mono sm">no GO prediction for dominant isoform(s)</p>'; return; }
  // go_compare arrives pre-ordered CT-dominant top-3 first, then AD-dominant top-3 (see
  // build_bisect_dtu.py go_compare_for) — reversing the y-axis autorange makes the FIRST
  // array item render at the TOP of the chart, so it reads "CT top-3 on top, AD top-3 below"
  // instead of Plotly's default bottom-up category order.
  const names = cmp.map(g=>g.name);
  const ctVals = cmp.map(g=>g.ct_score ?? 0), adVals = cmp.map(g=>g.ad_score ?? 0);
  const ctMaxIdx = ctVals.indexOf(Math.max(...ctVals));
  const adMaxIdx = adVals.indexOf(Math.max(...adVals));
  const starText = (vals, maxIdx) => vals.map((_,i) => i===maxIdx ? '★' : '');
  Plotly.newPlot('bd-domgo', [
    {type:'bar', orientation:'h', name:'CT-dominant', y:names, x:ctVals, marker:{color:T.trace},
     text:starText(ctVals, ctMaxIdx), textposition:'outside', textfont:{color:T.trace,size:11}, cliponaxis:false},
    {type:'bar', orientation:'h', name:'AD-dominant', y:names, x:adVals, marker:{color:T.signal},
     text:starText(adVals, adMaxIdx), textposition:'outside', textfont:{color:T.signal,size:11}, cliponaxis:false},
  ], inst({
    barmode:'group', height:280, margin:{l:170,r:36,t:46,b:34},
    title:{text:'dominant-isoform PRISM GO score · CT top-3 (top) / AD top-3 (bottom) · ★ = highest score', font:{size:10,color:T.dim}},
    xaxis:{title:{text:'PRISM score',font:{size:10,color:T.dim}}, range:[0,1.1]},
    yaxis:{tickfont:{family:MONO,size:9,color:T.dim}, automargin:true, autorange:'reversed'},
    legend:{orientation:'h', y:-0.2, font:{size:9,color:T.dim}},
  }), PLOT_CFG);
}

// ── IDR + domain 아키텍처 모식도 (CT vs AD) ──────────────────
function drawTracks(tr){
  const host=document.getElementById('bd-track'); if(!host) return;
  if(!tr || (!tr.ct && !tr.ad)){ host.innerHTML='<p class="muted mono sm">no sequence track</p>'; return; }
  const traces=[], shapes=[], anns=[];
  const rows=[['ct',tr.ct,'#35C6E8',1],['ad',tr.ad,'#FF5C1A',0]];
  let maxLen=1;
  rows.forEach(([k,s])=>{ if(s&&s.len) maxLen=Math.max(maxLen,s.len); });
  rows.forEach(([k,s,col,yb])=>{
    if(!s) return;
    const y=yb; // ct=1, ad=0
    // disorder line
    if(s.disorder && s.disorder.length){
      const n=s.disorder.length, xs=s.disorder.map((_,i)=>i+1);
      traces.push({x:xs, y:s.disorder.map(d=>y+ d*0.34), mode:'lines',
        line:{color:col,width:1.2}, name:k.toUpperCase()+' disorder',
        hovertemplate:k.toUpperCase()+' aa%{x} · disorder %{customdata:.2f}<extra></extra>',
        customdata:s.disorder});
      // IDR 임계(0.5) 위 영역 음영
    }
    // sequence backbone
    shapes.push({type:'line', x0:1, x1:s.len, y0:y, y1:y, line:{color:'#28324a',width:1}});
    // domain boxes — label height staggers alternately (0.16 / 0.30) so two domains close
    // together in residue-space don't render their name labels on top of each other.
    (s.domains||[]).forEach((d,di)=>{
      shapes.push({type:'rect', x0:d.start, x1:d.end, y0:y-0.09, y1:y+0.09,
        fillcolor:col, opacity:.28, line:{color:col,width:1}});
      const lift = di % 2 === 0 ? 0.16 : 0.30;
      anns.push({x:(d.start+d.end)/2, y:y+lift, text:d.name, showarrow:false,
        font:{family:MONO,size:9,color:col}});
    });
    anns.push({x:0, y:y, xref:'x', yref:'y', text:k.toUpperCase()+' '+s.len+'aa',
      showarrow:false, xanchor:'right', font:{family:MONO,size:10,color:col}, xshift:-6});
  });
  // full-width row now (was a half-width .bd-2col column) — taller canvas + wider yaxis
  // headroom for the staggered domain labels above.
  Plotly.newPlot('bd-track', traces, inst({height:220, margin:{l:60,r:24,t:12,b:32}, shapes, annotations:anns,
    showlegend:false,
    xaxis:{title:{text:'residue',font:{size:10,color:T.dim}}, range:[-maxLen*0.06,maxLen*1.02]},
    yaxis:{range:[-0.4,1.75], tickvals:[0,1], ticktext:['AD','CT'],
           tickfont:{family:MONO,size:10,color:T.dim}}}), PLOT_CFG);
}

// ── SF/TF regulator volcano ─────────────────────────────────
function drawVolcano(regs){
  const host=document.getElementById('bd-volcano'); if(!host) return;
  if(!regs || !regs.length){ host.innerHTML='<p class="muted mono sm">no regulators</p>'; return; }
  const up=regs.filter(r=>r.logFC>=0), dn=regs.filter(r=>r.logFC<0);
  const mk=(arr,col)=>({x:arr.map(r=>r.logFC), y:arr.map(r=>r.neg_log10_padj),
    text:arr.map(r=>r.gene), mode:'markers+text', textposition:'top center',
    textfont:{family:MONO,size:9,color:T.dim}, marker:{size:9,color:col,line:{color:T.ink900,width:1}},
    hovertemplate:'%{text}<br>logFC %{x:.2f} · −log10padj %{y:.1f}<extra></extra>'});
  // full-width row now (was a half-width .bd-2col column) — a bit taller, and the extra
  // horizontal room means gene-name text labels on nearby points collide less.
  Plotly.newPlot('bd-volcano',[mk(up,T.signal), mk(dn,T.trace)],
    inst({height:280, margin:{l:48,r:24,t:16,b:44}, showlegend:false,
      shapes:[{type:'line',x0:0,x1:0,yref:'paper',y0:0,y1:1,line:{color:T.grid2,width:1,dash:'dot'}}],
      xaxis:{title:{text:'logFC (AD vs CT regulator activity)',font:{size:10,color:T.dim}}, zeroline:false},
      yaxis:{title:{text:'−log10(padj)',font:{size:10,color:T.dim}}}}), PLOT_CFG);
}

document.addEventListener('DOMContentLoaded', boot);
// 테마 전환 시 표 + 열려있는 케이스 리포트(Plotly 포함)를 새 팔레트로 다시 그린다.
window.__rerenderCharts = () => {
  render(); drawMechCell(SUM.mech_cell_rows||[]);
  if(_openCase) showCase(_openCase.gene, _openCase.cell);
};
