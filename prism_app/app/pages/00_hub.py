"""Page 0 — Analysis Hub: Entry point with workflow overview and quick navigation."""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import numpy as np
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
