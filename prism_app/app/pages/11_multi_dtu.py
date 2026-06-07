"""11_multi_dtu.py — Multi-gene DTU condition comparison."""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

@st.cache_data(show_spinner=False)
def _load_diu_for_dtu() -> 'pd.DataFrame | None':
    """Load 8 cell-type DIU CSVs (gene_name, chi_significant, usage_direction, delta_usage, cell_type)."""
    _diu_dir = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/06_DIU')
    _cell_names = [
        'Excitatory_neuron', 'Inhibitory_neuron', 'Astrocyte', 'Microglia',
        'Oligodendrocyte', 'OPC', 'Vascular_cell', 'Lymphocyte',
    ]
    _frames = []
    for _ct in _cell_names:
        _fp = _diu_dir / f'DIU_by_condition_{_ct}.csv'
        if _fp.exists():
            try:
                _df = pd.read_csv(_fp)
                _df['cell_type'] = _ct.replace('_', ' ')
                _frames.append(_df)
            except Exception:
                pass
    return pd.concat(_frames, ignore_index=True) if _frames else None

from prism_app.app.components.basket import init_basket, get_analysis_cases

init_basket()

cfg      = st.session_state.get('cfg') or {}
dtu      = cfg.get('dtu_df')
gene_ids = cfg.get('gene_ids')

st.markdown("## 🔄 DTU 조건 다중 분석")
st.caption(
    "여러 유전자의 조건별 DTU (Differential Transcript Usage) 이벤트를 비교합니다. "
    "|ΔIF|가 클수록 조건 간 아이소폼 전환이 강하게 일어납니다."
)
st.divider()

if dtu is None:
    st.warning("DTU 데이터 없음 — 사이드바에서 DTU 파일(.tsv)을 업로드하세요.")
    st.stop()

# Detect column names
_gc  = next((c for c in ['gene_id', 'gene', 'geneID'] if c in dtu.columns), None)
_cc  = next((c for c in ['condition', 'comparison', 'contrast'] if c in dtu.columns), None)
_dfc = next((c for c in ['delta_IF', 'delta_if', 'dIF', 'deltaPSI'] if c in dtu.columns), None)
_ic  = next((c for c in ['isoform_id', 'transcript_id', 'isoformID'] if c in dtu.columns), None)

if not all([_gc, _cc, _dfc]):
    st.error(f"DTU 파일 컬럼 확인 실패. 감지된 컬럼: {list(dtu.columns)}")
    st.stop()

# ── Gene selection ────────────────────────────────────────────────────────────
_dtu_cases = [c for c in get_analysis_cases()
              if c.get('mode') == 'multi' and c.get('item_type') == 'gene'
              and c.get('axis') == 'dtu']
_gene_pool = gene_ids
_all_unique = list(dict.fromkeys(str(g) for g in _gene_pool)) if _gene_pool is not None else []
# Also include genes in DTU file
_dtu_genes = sorted(dtu[_gc].dropna().astype(str).unique().tolist())
_pool_combined = sorted(set(_all_unique) | set(_dtu_genes))

_col_left, _col_right = st.columns([3, 2])
with _col_left:
    st.markdown("#### 유전자 선택")
    if _dtu_cases:
        _case_opts = {f"케이스 #{c['case_id']} — {', '.join(c['items'][:3])}{'...' if len(c['items'])>3 else ''}": c
                      for c in _dtu_cases}
        _sel_case = st.selectbox("저장된 케이스 불러오기",
                                 ["— 직접 선택 —"] + list(_case_opts.keys()),
                                 key='mdtu_case_sel')
        _default_genes = _case_opts[_sel_case]['items'] if _sel_case != "— 직접 선택 —" else []
    else:
        st.caption("저장된 DTU 케이스 없음 — 아래서 직접 선택하세요.")
        _default_genes = []

    _gene_sel = st.multiselect(
        "분석할 유전자 (다중 선택 · 검색 가능)",
        options=_pool_combined,
        default=[g for g in _default_genes if g in _pool_combined],
        key='mdtu_gene_sel',
        placeholder="유전자명 입력하여 검색...",
    )

with _col_right:
    if _gene_sel:
        _badge_html = ' '.join(
            f"<span style='background:#1e40af;color:white;border-radius:4px;"
            f"padding:3px 10px;font-size:0.82rem'>{g}</span>"
            for g in _gene_sel
        )
        st.markdown("#### 선택된 유전자")
        st.markdown(_badge_html, unsafe_allow_html=True)

st.divider()

# ── Analysis ──────────────────────────────────────────────────────────────────
if not _gene_sel:
    st.info("위에서 유전자를 선택하면 DTU 조건 분석이 표시됩니다.")
    st.stop()

# Filter DTU data for selected genes
_dtu_filt = dtu[dtu[_gc].astype(str).str.upper().isin([g.upper() for g in _gene_sel])].copy()
if _dtu_filt.empty:
    st.info("선택된 유전자에 DTU 이벤트가 없습니다.")
    st.stop()

# Aggregate: max |ΔIF| per (gene, condition)
_dtu_filt['|ΔIF|'] = _dtu_filt[_dfc].abs()
_dtu_agg = (_dtu_filt.groupby([_gc, _cc])['|ΔIF|']
             .max().reset_index()
             .rename(columns={_gc: '유전자', _cc: '조건'}))
_dtu_agg['|ΔIF|'] = _dtu_agg['|ΔIF|'].round(3)

# Bar chart
_fig_bar = px.bar(
    _dtu_agg, x='조건', y='|ΔIF|', color='유전자',
    barmode='group', height=460, title="DTU 조건별 |ΔIF| 최대값 비교",
)
_fig_bar.update_layout(plot_bgcolor='white', xaxis_tickangle=-30)
_fig_bar.add_hline(y=0.1, line_dash='dash', line_color='grey',
                   annotation_text="|ΔIF| = 0.1")
st.plotly_chart(_fig_bar, use_container_width=True, key='mdtu_bar')

# Pivot heatmap
_pivot = _dtu_agg.pivot_table(index='유전자', columns='조건', values='|ΔIF|',
                               aggfunc='max').fillna(0)
_fig_hmap = px.imshow(
    _pivot, title="유전자 × 조건 |ΔIF| 히트맵",
    color_continuous_scale='Oranges', aspect='auto',
    height=max(320, 70 * len(_gene_sel)),
)
_fig_hmap.update_layout(xaxis_tickangle=-30)
st.plotly_chart(_fig_hmap, use_container_width=True, key='mdtu_hmap')

st.markdown("#### 조건별 |ΔIF| 피벗 테이블")
st.dataframe(
    _pivot.style.background_gradient(cmap='Oranges'),
    use_container_width=True,
)

# ── DTU × PRISM Functional Consequence Scatter ───────────────────────────────
_sm_mat  = cfg.get('score_matrix')
_iso_ids = cfg.get('isoform_ids')
_g_ids   = cfg.get('gene_ids')

if _sm_mat is not None and _iso_ids is not None and _g_ids is not None and _ic:
    st.divider()
    st.markdown("#### DTU × PRISM 기능 변화 연관성")
    st.caption(
        "x축: 조건 간 아이소폼 전환 강도 |ΔIF| · "
        "y축: 유전자 내 중앙값 대비 최대 ΔPRISM score. "
        "우상단 = 발현 전환이 강하고 기능 변화도 큰 케이스 (BISECT S1 후보)."
    )

    _iso_arr  = np.asarray(_iso_ids, dtype=str)
    _gene_arr = np.asarray(_g_ids,   dtype=str)
    _sm_arr   = np.asarray(_sm_mat,  dtype=float)

    _scatter_rows = []
    for _sg in _gene_sel:
        _g_mask = np.char.upper(_gene_arr) == _sg.upper()
        if not _g_mask.any():
            continue
        _g_sm     = _sm_arr[_g_mask]     # (n_iso, n_go)
        _g_median = np.median(_g_sm, axis=0)  # (n_go,) — gene reference

        _dtu_gene = _dtu_filt[_dtu_filt[_gc].astype(str).str.upper() == _sg.upper()]
        for _, _row in _dtu_gene.iterrows():
            _iso_name = str(_row[_ic])
            _iso_idx  = np.where(_iso_arr == _iso_name)[0]
            if not len(_iso_idx):
                _iso_idx = np.where(np.char.upper(_iso_arr) == _iso_name.upper())[0]
            if not len(_iso_idx):
                continue
            _delta_prism = float((_sm_arr[_iso_idx[0]] - _g_median).max())
            _scatter_rows.append({
                '유전자':     _sg,
                '아이소폼':   _iso_name,
                '조건':       str(_row[_cc]),
                '|ΔIF|':      round(abs(float(_row[_dfc])), 3),
                'max ΔPRISM': round(_delta_prism, 3),
            })

    if _scatter_rows:
        _sdf = pd.DataFrame(_scatter_rows)
        _fig_sc = px.scatter(
            _sdf, x='|ΔIF|', y='max ΔPRISM',
            color='유전자', symbol='조건',
            hover_data=['아이소폼', '조건', '유전자'],
            title="DTU 강도 vs PRISM 기능 변화 (유전자 내 중앙값 기준)",
            height=460,
        )
        _fig_sc.add_hline(y=0,   line_dash='dash', line_color='#94a3b8', line_width=1)
        _fig_sc.add_vline(x=0.1, line_dash='dash', line_color='#94a3b8', line_width=1,
                          annotation_text="|ΔIF|=0.1", annotation_font_size=11)
        _fig_sc.update_layout(plot_bgcolor='white')
        st.plotly_chart(_fig_sc, use_container_width=True, key='mdtu_prism_scatter')

        from scipy.stats import spearmanr as _spearmanr
        _r, _p = _spearmanr(_sdf['|ΔIF|'], _sdf['max ΔPRISM'])
        _n_pts  = len(_sdf)
        _q1 = (_sdf['|ΔIF|'] >= 0.1) & (_sdf['max ΔPRISM'] > 0)
        st.caption(
            f"Spearman r = **{_r:.3f}** (p = {_p:.2e}, n = {_n_pts:,}) · "
            f"우상단 고신뢰 케이스: {int(_q1.sum())}개 "
            f"(|ΔIF| ≥ 0.1 ∩ ΔPRISM > 0)"
        )
    else:
        st.info("선택된 유전자의 DTU 아이소폼과 PRISM 데이터 간 매칭 결과 없음 "
                "(아이소폼 ID 형식이 다를 수 있음)")

# Per-gene isoform breakdown (if isoform column exists)
if _ic:
    st.divider()
    st.markdown("#### 아이소폼 수준 DTU 상세")
    _sel_gene_detail = st.selectbox("유전자 선택", _gene_sel, key='mdtu_detail_gene')
    _gene_detail = _dtu_filt[_dtu_filt[_gc].astype(str).str.upper() == _sel_gene_detail.upper()]
    if not _gene_detail.empty:
        _cols_show = [c for c in [_ic, _cc, _dfc] if c in _gene_detail.columns]
        _detail_df = _gene_detail[_cols_show].copy()
        _detail_df[_dfc] = _detail_df[_dfc].round(3)
        _detail_df = _detail_df.sort_values(_dfc, key=abs, ascending=False)
        st.dataframe(_detail_df, use_container_width=True, hide_index=True)

# ── Cell-type DIU 재현성 ──────────────────────────────────────────────────────
_diu_full = _load_diu_for_dtu()
if _diu_full is not None:
    st.divider()
    with st.expander("🔬 세포유형 DTU 재현성 — 8개 뇌 세포유형 간 일관성", expanded=True):
        _CELL_ORDER = [
            'Excitatory neuron', 'Inhibitory neuron', 'Astrocyte', 'Microglia',
            'Oligodendrocyte', 'OPC', 'Vascular cell', 'Lymphocyte',
        ]
        # ── 1. Gene-level: 유전자 × 세포유형 유의 DTU 건수 히트맵
        st.markdown("##### 유전자 × 세포유형 — 유의 DTU 이벤트 수")
        _diu_sel = _diu_full[
            _diu_full['gene_name'].str.upper().isin([g.upper() for g in _gene_sel])
            & (_diu_full['chi_significant'] == True)
        ].copy()

        if _diu_sel.empty:
            st.info("선택된 유전자에서 DIU 데이터(chi_significant=True) 없음")
        else:
            _ct_present = [c for c in _CELL_ORDER if c in _diu_sel['cell_type'].unique()]
            _gene_ct = (
                _diu_sel.groupby(['gene_name', 'cell_type'])
                .size().reset_index(name='n_sig')
                .pivot_table(index='gene_name', columns='cell_type', values='n_sig', fill_value=0)
            )
            _gene_ct = _gene_ct[[c for c in _ct_present if c in _gene_ct.columns]]

            _fig_g = px.imshow(
                _gene_ct,
                color_continuous_scale='Blues',
                labels={'color': '유의 DTU 수'},
                aspect='auto',
                height=max(280, 50 * len(_gene_ct) + 80),
                title="유전자별 세포유형 DTU 재현성",
            )
            _fig_g.update_layout(
                xaxis_tickangle=-25,
                margin=dict(t=40, b=40, l=10, r=10),
            )
            st.plotly_chart(_fig_g, use_container_width=True, key='mdtu_diu_gene_hmap')

            # Concordance summary per gene
            _conc_rows = []
            for _g in _gene_sel:
                _g_sig = _diu_sel[_diu_sel['gene_name'].str.upper() == _g.upper()]
                _n_ct  = _g_sig['cell_type'].nunique()
                _n_ad  = int((_g_sig['usage_direction'].str.contains('AD_enriched', na=False)).sum())
                _n_ct2 = int((_g_sig['usage_direction'].str.contains('CT_enriched', na=False)).sum())
                _conc  = round(max(_n_ad, _n_ct2) / _n_ct, 2) if _n_ct > 0 else 0.0
                _conc_rows.append({
                    '유전자': _g, '재현 세포유형': _n_ct,
                    'AD-enriched': _n_ad, 'CT-enriched': _n_ct2,
                    'Concordance': _conc,
                    '주요 방향': ('AD↑' if _n_ad >= _n_ct2 else 'CT↑') if _n_ct > 0 else '—',
                })
            _conc_df = pd.DataFrame(_conc_rows).sort_values('재현 세포유형', ascending=False)
            st.dataframe(
                _conc_df.style.background_gradient(subset=['재현 세포유형', 'Concordance'],
                                                   cmap='Greens', vmin=0),
                use_container_width=True, hide_index=True,
            )

        # ── 2. 단일 유전자 선택 시 아이소폼 × 세포유형 히트맵
        if len(_gene_sel) >= 1:
            st.markdown("##### 아이소폼 수준 — 세포유형별 ΔIF")
            _detail_g2 = st.selectbox("아이소폼 히트맵 유전자", _gene_sel,
                                      key='mdtu_diu_iso_sel')
            _iso_diu = _diu_full[
                _diu_full['gene_name'].str.upper() == _detail_g2.upper()
            ].copy()
            _sig2_mask = _iso_diu['chi_significant'] == True
            _pivot_iso = (
                _iso_diu[_sig2_mask]
                .pivot_table(
                    index='transcript_name', columns='cell_type',
                    values='delta_usage', aggfunc='first',
                ).fillna(0)
            )
            _ct_iso = [c for c in _CELL_ORDER if c in _pivot_iso.columns]
            _pivot_iso = _pivot_iso[[c for c in _ct_iso if c in _pivot_iso.columns]]

            if _pivot_iso.empty:
                st.caption(f"{_detail_g2}: 유의 DIU 없음")
            else:
                _fig_iso = px.imshow(
                    _pivot_iso,
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0,
                    zmin=-0.5, zmax=0.5,
                    labels={'color': 'ΔIF'},
                    aspect='auto',
                    height=max(260, 40 * len(_pivot_iso) + 80),
                    title=f"{_detail_g2} — 아이소폼 × 세포유형 ΔIF",
                )
                _fig_iso.update_layout(
                    xaxis_tickangle=-25,
                    margin=dict(t=40, b=40, l=10, r=10),
                    coloraxis_colorbar=dict(title='ΔIF', len=0.6, thickness=12),
                )
                st.plotly_chart(_fig_iso, use_container_width=True, key='mdtu_diu_iso_hmap')
                st.caption(
                    "빨강 = AD에서 아이소폼 사용 증가 · 파랑 = CT에서 증가. "
                    "chi_significant=True인 이벤트만 표시."
                )
