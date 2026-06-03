"""Page 0 — Analysis Hub: Entry point with workflow overview and quick navigation."""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Analysis Hub — PRISM", layout="wide")

# Ensure session state is initialised
if 'analysis_step' not in st.session_state:
    st.session_state['analysis_step'] = {}
if 'basket_genes' not in st.session_state:
    st.session_state['basket_genes'] = []
if 'active_module' not in st.session_state:
    st.session_state['active_module'] = None
if 'search_gene' not in st.session_state:
    st.session_state['search_gene'] = ''

st.session_state['analysis_step']['hub'] = True

cfg = st.session_state.get('cfg', {})
sm   = cfg.get('score_matrix')
ids  = cfg.get('isoform_ids')
typs = cfg.get('isoform_types')
gins = cfg.get('gene_ids')
dtu  = cfg.get('dtu_df')

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0f2942 0%,#1a4a6e 60%,#0f4c75 100%);
border-radius:14px;padding:36px 48px 28px;margin-bottom:20px'>
<h1 style='color:white;margin:0;font-size:2rem'>🧬 PRISM Analysis Hub</h1>
<p style='color:#93c5fd;margin:8px 0 0;font-size:1rem'>
Isoform Function Prediction · Module Landscape · Disease Target Discovery
</p>
</div>
""", unsafe_allow_html=True)

# ── Data status summary ───────────────────────────────────────────────────────
tissue = cfg.get('tissue', '')
has_data = sm is not None and ids is not None

col_status1, col_status2, col_status3, col_status4 = st.columns(4)

with col_status1:
    if has_data:
        n = sm.shape[0]
        st.metric("Isoforms loaded", f"{n:,}", delta="ready")
    else:
        st.metric("Isoforms loaded", "—", delta="No data")

with col_status2:
    if has_data:
        n_go = sm.shape[1]
        st.metric("GO terms", f"{n_go}", delta=tissue or "—")
    else:
        st.metric("GO terms", "—")

with col_status3:
    if dtu is not None:
        st.metric("DTU events", f"{len(dtu):,}", delta="loaded")
    else:
        st.metric("DTU events", "—", delta="not loaded")

with col_status4:
    basket = st.session_state.get('basket_genes', [])
    st.metric("Gene basket", f"{len(basket)}", delta="saved targets")

st.divider()

# ── Workflow diagram ──────────────────────────────────────────────────────────
st.subheader("📋 분석 워크플로우")

steps = st.session_state.get('analysis_step', {})

WORKFLOW = [
    {
        'key': 'qc',
        'page': '01_qc',
        'icon': '📊',
        'title': 'QC & Overview',
        'desc': '데이터 커버리지 확인, GO term 분포, Scenario 분류 (1-4)',
        'when': '데이터 로드 직후',
        'output': '기능 예측 정확도 (AUPRC), 고신뢰 아이소폼 수',
    },
    {
        'key': 'landscape',
        'page': '02_landscape',
        'icon': '🗺️',
        'title': 'Module Landscape',
        'desc': '672 GO term 계층 클러스터링 → 44개 기능 모듈 전체 지형도',
        'when': 'Brain-672 패널 선택 후',
        'output': '아이소폼별 주요 기능 모듈, Novel isoform 농화 모듈',
    },
    {
        'key': 'patterns',
        'page': '03_patterns',
        'icon': '🔬',
        'title': 'Functional Patterns',
        'desc': 'GO co-occurrence 네트워크, UMAP 클러스터 탐색',
        'when': '모듈 지형 파악 후',
        'output': '기능적으로 연관된 GO term 군집, 이상치 아이소폼',
    },
    {
        'key': 'condition',
        'page': '04_condition',
        'icon': '🔄',
        'title': 'Condition Analysis',
        'desc': 'DTU × 기능 모듈 전환 — 질환 조건에서 기능 변화',
        'when': 'DTU 데이터 있을 때',
        'output': 'GAIN/LOSS 유전자, 교차 모듈 전환 후보',
        'requires_dtu': True,
    },
    {
        'key': 'targets',
        'page': '05_targets',
        'icon': '🎯',
        'title': 'Target Analysis',
        'desc': '특정 유전자/아이소폼 심화 분석 — 4-Scenario + 모듈 귀속',
        'when': '후보 타겟 선정 후',
        'output': 'Isoform case report, 모듈×DTU 히트맵',
    },
]

# Render workflow steps as cards
for i, step in enumerate(WORKFLOW):
    done = steps.get(step['key'], False)
    requires_dtu = step.get('requires_dtu', False)
    available = not requires_dtu or (dtu is not None)

    bg = '#f0fdf4' if done else ('#f9fafb' if available else '#fef2f2')
    border = '#16a34a' if done else ('#6b7280' if available else '#fca5a5')
    status_icon = '✅' if done else ('⬜' if available else '⚠️ DTU 필요')
    status_text = '완료' if done else ('대기 중' if available else 'DTU 데이터 필요')

    with st.container():
        st.markdown(f"""
<div style='background:{bg};border-radius:10px;padding:14px 20px;
border-left:4px solid {border};margin-bottom:10px;display:flex;align-items:flex-start'>
<div style='font-size:1.5rem;margin-right:14px;margin-top:2px'>{step['icon']}</div>
<div style='flex:1'>
  <div style='display:flex;align-items:center;gap:10px'>
    <b style='font-size:1rem'>Step {i+1}: {step['title']}</b>
    <span style='font-size:0.75rem;background:{"#dcfce7" if done else "#f3f4f6"};
    border-radius:10px;padding:2px 8px;color:{"#15803d" if done else "#6b7280"}'>{status_text}</span>
  </div>
  <div style='color:#374151;font-size:0.875rem;margin-top:4px'>{step['desc']}</div>
  <div style='color:#6b7280;font-size:0.8rem;margin-top:4px'>
    ▸ <b>언제</b>: {step['when']} &nbsp;|&nbsp; ▸ <b>산출물</b>: {step['output']}
  </div>
</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Quick Stats (if data loaded) ──────────────────────────────────────────────
if has_data:
    st.subheader("📈 데이터 Quick View")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        if typs is not None:
            type_arr = np.array(typs, dtype=str)
            unique_types, counts = np.unique(type_arr, return_counts=True)
            fig_pie = px.pie(
                values=counts,
                names=unique_types,
                title="Isoform type distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10),
                                  showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Isoform types not available — load a dataset with type annotations.")

    with col_chart2:
        # Score distribution histogram
        max_scores = sm.max(axis=1)
        fig_hist = px.histogram(
            x=max_scores,
            nbins=50,
            title="Max PRISM score per isoform",
            labels={'x': 'Max score', 'y': 'Count'},
            color_discrete_sequence=['#3b82f6'],
        )
        thr = cfg.get('score_threshold', 0.4)
        fig_hist.add_vline(x=thr, line_dash='dash', line_color='red',
                           annotation_text=f'thr={thr}', annotation_position='top right')
        n_high = int((max_scores >= thr).sum())
        fig_hist.update_layout(
            height=280,
            margin=dict(t=40, b=10, l=10, r=10),
            annotations=[dict(
                x=thr, y=0, text=f'{n_high:,} high-conf',
                showarrow=False, yanchor='bottom', font=dict(size=10, color='red'),
                xanchor='left',
            )],
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Gene basket quick access
    if basket:
        st.subheader("🧬 Gene Basket")
        cols = st.columns(min(len(basket), 6))
        for i, gene in enumerate(basket[:6]):
            with cols[i]:
                if st.button(f"🎯 {gene}", key=f'hub_basket_{i}', use_container_width=True):
                    st.session_state['search_gene'] = gene
                    st.info(f"'{gene}' → Target Analysis 페이지에서 상세 분석을 확인하세요.")

else:
    # No data: show onboarding
    st.subheader("🚀 시작하기")
    st.markdown("""
사이드바에서 **데이터 소스**를 선택하세요:

| 옵션 | 설명 | 바로 시작 가능? |
|------|------|---------------|
| **Demo (paper data)** | 논문의 사전 계산 결과 탐색 (muscle 36K / brain 64K) | ✅ 즉시 |
| **Upload my data** | 내 NPY 스코어 매트릭스 업로드 | 파일 준비 필요 |

**데모 추천 경로**: Brain — Full Module Landscape (672 GO terms) 선택 → Step 2 (Landscape) → Step 4 (Condition)
""")

    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        st.markdown("""
<div style='background:#eff6ff;border-radius:8px;padding:16px;border:1px solid #bfdbfe'>
<b>논문 리뷰어용</b><br>
Demo 모드에서 Brain 672 패널 선택 →<br>
Module Landscape → NDUFS4/KIF21B/PTPRF 검색
</div>
""", unsafe_allow_html=True)
    with col_demo2:
        st.markdown("""
<div style='background:#f0fdf4;border-radius:8px;padding:16px;border:1px solid #bbf7d0'>
<b>연구자용 (내 데이터)</b><br>
Upload 모드 → NPY 스코어 + ID 업로드 →<br>
QC → Landscape → 조건별 분석
</div>
""", unsafe_allow_html=True)

# ── BISECT Featured Cases ─────────────────────────────────────────────────────

st.divider()
st.markdown("""
<div style='display:flex;align-items:center;gap:12px;margin-bottom:4px'>
<h3 style='margin:0'>🔬 BISECT Featured Cases</h3>
<span style='background:#0f2942;color:#93c5fd;border-radius:6px;padding:3px 10px;
font-size:0.75rem;font-weight:600;letter-spacing:0.05em'>PAPER DEMO DATA</span>
</div>
""", unsafe_allow_html=True)
st.caption(
    "이것은 **논문 사전 계산 결과**입니다 — 내 데이터 분석 결과가 아닙니다. "
    "AD vs CT 조건에서 isoform 기능 전환이 검증된 84개 케이스 (BISECT PASS). "
    "유전자 클릭 → Target Analysis 페이지 자동 이동 + 검색 실행."
)


@st.cache_data(show_spinner=False)
def _load_bisect_cases():
    p = Path(__file__).parents[2] / 'data' / 'demo' / 'bisect_cases.json'
    if not p.exists():
        return None
    with open(p) as f:
        cases = json.load(f)
    df = pd.DataFrame(cases)
    # Clean up
    df['domains_lost']   = df['domains_lost'].fillna('').astype(str)
    df['domains_gained'] = df['domains_gained'].fillna('').astype(str)
    df['delta_abs']      = df['delta'].abs()
    df['mechanism_label'] = df['mechanism_type'].map({
        'alternative_promoter':   'Alt. Promoter',
        'transcriptional':        'Transcriptional',
        'epigenetic_derepression':'Epigenetic',
        'alternative_splicing':   'Alt. Splicing',
    }).fillna(df['mechanism_type'])
    df['ppi_str'] = df['ppi_verdict'].map({'SUPPORTED': '✅ PPI', 'UNSUPPORTED': ''}).fillna('')
    df['cons_str'] = df['cons_ad_class'].map({
        'highly_conserved': '🌿', 'conserved': '🌱', 'low': '',
    }).fillna('')
    return df


_bisect_df = _load_bisect_cases()

if _bisect_df is not None:
    # ── Summary metrics ───────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("BISECT PASS", "84")
    mc2.metric("Genes", str(_bisect_df['gene'].nunique()))
    mc3.metric("Cell types", str(_bisect_df['cell_type'].nunique()))
    mc4.metric("PPI supported", str((_bisect_df['ppi_verdict'] == 'SUPPORTED').sum()))

    col_chart, col_table = st.columns([1, 2])

    with col_chart:
        mech_ct = _bisect_df.groupby(['mechanism_label','cell_type']).size().reset_index(name='n')
        fig_mech = px.bar(
            mech_ct, x='n', y='mechanism_label', color='cell_type',
            orientation='h',
            title='Cases by mechanism × cell type',
            labels={'n': 'Cases', 'mechanism_label': '', 'cell_type': 'Cell type'},
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=260,
        )
        fig_mech.update_layout(
            margin=dict(t=36, b=10, l=10, r=10),
            legend=dict(orientation='h', y=-0.25, font=dict(size=10)),
            xaxis=dict(title_font_size=11),
        )
        st.plotly_chart(fig_mech, use_container_width=True)

    with col_table:
        # Top cases by |delta|
        top_cases = _bisect_df.sort_values('delta_abs', ascending=False).head(15)
        disp = top_cases[['gene','cell_type','delta','mechanism_label','domains_lost','ppi_str']].rename(columns={
            'gene': '유전자', 'cell_type': '세포 유형', 'delta': 'dIF',
            'mechanism_label': '메커니즘', 'domains_lost': '손실 도메인', 'ppi_str': 'PPI',
        })
        st.dataframe(disp.style.background_gradient(subset=['dIF'], cmap='RdBu_r'),
                     use_container_width=True, hide_index=True, height=260)

    # ── Landmark case cards ───────────────────────────────────────────────
    st.markdown("**⭐ Landmark Cases**")

    LANDMARKS = [
        {
            'gene': 'NDUFS4',
            'title': 'Complex I collapse (Epigenetic)',
            'desc': 'NDUS4 도메인 손실 → Complex I 어셈블리 불가 · PRISM M6→M17 전환 · 50× 기능 스코어 변화',
            'color': '#fef3c7', 'border': '#d97706',
        },
        {
            'gene': 'KIF21B',
            'title': 'Kinesin motor switch (Alt. Splicing)',
            'desc': 'NIC-M43(골지/막 수송) LOSS → NNIC-M44(미세소관) GAIN · Excitatory neurons · WD40 cargo binding 변화',
            'color': '#eff6ff', 'border': '#2563eb',
        },
        {
            'gene': 'PTPRF',
            'title': 'Phosphatase → adhesion (Alt. Promoter)',
            'desc': 'M36(신경 분화) → M35(세포 부착) 교차 모듈 전환 · 4 cell types · 포스파타제 도메인 9개 손실',
            'color': '#f0fdf4', 'border': '#16a34a',
        },
        {
            'gene': 'DLG1',
            'title': 'Synaptic scaffold gain (Alt. Splicing)',
            'desc': 'PDZ/SH3/Guanylate_kin 도메인 획득 · OPC · M13/M35 이중 모듈 · PPI SUPPORTED',
            'color': '#fdf4ff', 'border': '#9333ea',
        },
    ]

    lm_cols = st.columns(4)
    for i, lm in enumerate(LANDMARKS):
        # Find in dataframe
        row = _bisect_df[_bisect_df['gene'] == lm['gene']]
        delta_val = f"{row['delta'].values[0]:+.3f}" if not row.empty else '—'
        with lm_cols[i]:
            st.markdown(f"""
<div style='background:{lm["color"]};border:1.5px solid {lm["border"]};border-radius:10px;
padding:12px 14px;height:160px;overflow:hidden'>
<b style='font-size:1rem'>{lm["gene"]}</b>
<span style='font-size:0.8rem;color:#6b7280;margin-left:6px'>dIF {delta_val}</span><br>
<span style='font-size:0.75rem;color:{lm["border"]};font-weight:600'>{lm["title"]}</span><br>
<span style='font-size:0.75rem;color:#374151;line-height:1.3'>{lm["desc"]}</span>
</div>
""", unsafe_allow_html=True)
            if st.button(f"🔍 {lm['gene']} 분석", key=f'hub_lm_{i}', use_container_width=True):
                st.session_state['search_gene'] = lm['gene']
                st.session_state['auto_search'] = True
                st.session_state['_targets_query_loaded'] = False
                st.toast(f"{lm['gene']} → Target Analysis 페이지에서 자동 검색됩니다. 왼쪽 사이드바에서 'Targets'를 클릭하세요.")

    # ── Full case browser ─────────────────────────────────────────────────
    with st.expander("📋 전체 84 케이스 브라우저", expanded=False):
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            _search = st.text_input("유전자명 검색", placeholder="FANCA, DMD ...", key='bisect_search')
        with col_f2:
            _mech_filter = st.selectbox(
                "메커니즘 필터",
                ['전체'] + sorted(_bisect_df['mechanism_label'].unique().tolist()),
                key='bisect_mech',
            )

        _fdf = _bisect_df.copy()
        if _search:
            _fdf = _fdf[_fdf['gene'].str.upper().str.contains(_search.upper())]
        if _mech_filter != '전체':
            _fdf = _fdf[_fdf['mechanism_label'] == _mech_filter]

        _show_cols = ['gene', 'cell_type', 'delta', 'mechanism_label',
                      'domains_lost', 'domains_gained', 'ppi_str', 'cons_str']
        _rename = {
            'gene': '유전자', 'cell_type': '세포 유형', 'delta': 'dIF',
            'mechanism_label': '메커니즘', 'domains_lost': '손실 도메인',
            'domains_gained': '획득 도메인', 'ppi_str': 'PPI', 'cons_str': '보존성',
        }
        st.dataframe(
            _fdf[_show_cols].rename(columns=_rename).sort_values('dIF', key=abs, ascending=False),
            use_container_width=True, hide_index=True,
            height=min(400, 35 * len(_fdf) + 38),
        )
        st.caption(f"{len(_fdf)}/{len(_bisect_df)} 케이스 표시")

        # Gene select → basket
        _gene_options = sorted(_fdf['gene'].unique().tolist())
        _sel_gene = st.selectbox("유전자 선택 → Target Analysis 연결",
                                  ['—'] + _gene_options, key='bisect_gene_select')
        _bc1, _bc2 = st.columns(2)
        if _sel_gene != '—':
            with _bc1:
                if st.button("🔍 Target Analysis에서 분석", key='bisect_goto_target'):
                    st.session_state['search_gene'] = _sel_gene
                    st.session_state['auto_search'] = True
                    st.session_state['_targets_query_loaded'] = False
                    st.toast(f"{_sel_gene} → Target Analysis 페이지를 열면 자동 검색됩니다.")
            with _bc2:
                if st.button("➕ 바스켓에 추가", key='bisect_add_basket'):
                    _b = st.session_state.get('basket_genes', [])
                    if _sel_gene not in _b:
                        _b.append(_sel_gene)
                        st.session_state['basket_genes'] = _b
                    st.toast(f"{_sel_gene} 바스켓 추가 완료")

else:
    st.info("BISECT case data not found (`prism_app/data/demo/bisect_cases.json`). Run demo data setup first.")
