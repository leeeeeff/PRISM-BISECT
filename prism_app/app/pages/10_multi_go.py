"""10_multi_go.py — Multi-gene GO term score comparison."""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from prism_app.app.components.basket import init_basket, get_analysis_cases

init_basket()

cfg       = st.session_state.get('cfg') or {}
sm        = cfg.get('score_matrix')
ids       = cfg.get('isoform_ids')
gins      = cfg.get('gene_ids')
go_terms  = cfg.get('go_terms', [])
go_names  = cfg.get('go_names', {})
thr       = cfg.get('score_threshold', 0.4)
has_data  = sm is not None and ids is not None

st.markdown("## 🧬 GO term 다중 분석")
st.caption(
    "여러 유전자의 GO term 기능 점수를 비교합니다. "
    "각 유전자의 아이소폼 중 최고 점수를 유전자 대표 점수로 사용합니다."
)
st.divider()

if not has_data:
    st.warning("데이터를 먼저 로드하세요 (사이드바 → Demo 또는 Upload).")
    st.stop()

if not go_terms:
    st.warning("GO term 정보가 없습니다. 데이터 로드 상태를 확인하세요.")
    st.stop()

# ── Gene selection ────────────────────────────────────────────────────────────
_go_cases = [c for c in get_analysis_cases()
             if c.get('mode') == 'multi' and c.get('item_type') == 'gene'
             and c.get('axis') == 'go']
_gene_pool = cfg.get('gene_ids')
_all_unique = list(dict.fromkeys(str(g) for g in _gene_pool)) if _gene_pool is not None else []

_col_left, _col_right = st.columns([3, 2])
with _col_left:
    st.markdown("#### 유전자 선택")
    if _go_cases:
        _case_opts = {f"케이스 #{c['case_id']} — {', '.join(c['items'][:3])}{'...' if len(c['items'])>3 else ''}": c
                      for c in _go_cases}
        _sel_case = st.selectbox("저장된 케이스 불러오기",
                                 ["— 직접 선택 —"] + list(_case_opts.keys()),
                                 key='mgo_case_sel')
        _default_genes = _case_opts[_sel_case]['items'] if _sel_case != "— 직접 선택 —" else []
    else:
        st.caption("저장된 GO 케이스 없음 — 아래서 직접 선택하세요.")
        _default_genes = []

    _gene_sel = st.multiselect(
        "분석할 유전자 (다중 선택 · 검색 가능)",
        options=_all_unique,
        default=[g for g in _default_genes if g in _all_unique],
        key='mgo_gene_sel',
        placeholder="유전자명 입력하여 검색...",
    )

    _top_k = st.slider("상위 GO term 수", min_value=5, max_value=min(30, len(go_terms)),
                        value=15, key='mgo_topk')

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
    st.info("위에서 유전자를 선택하면 GO term 분석이 표시됩니다.")
    st.stop()

_gins_arr = np.array(gins, dtype=str) if gins is not None else None
_ids_arr  = np.array(ids, dtype=str)

# Build gene → max score vector
_gene_scores: dict = {}
_missing = []
for _g in _gene_sel:
    _gm = np.array([x.upper() == _g.upper()
                    for x in (_gins_arr if _gins_arr is not None else _ids_arr)])
    if not _gm.any():
        _missing.append(_g)
        continue
    _gene_scores[_g] = sm[_gm].max(axis=0)

if _missing:
    st.warning(f"데이터에서 찾을 수 없는 유전자: {', '.join(_missing)}")

if not _gene_scores:
    st.info("선택된 유전자에 점수 데이터가 없습니다.")
    st.stop()

# Union of top-K GO indices
_union_gi: set = set()
for _g, _scores in _gene_scores.items():
    _union_gi |= set(np.argsort(_scores)[-_top_k:].tolist())
_union_gi_sorted = sorted(_union_gi)
_go_cols = [go_names.get(go_terms[i], go_terms[i])[:30]
            for i in _union_gi_sorted if i < len(go_terms)]

# Build heatmap matrix
_hmap_rows, _hmap_idx = [], []
for _g in _gene_sel:
    if _g in _gene_scores:
        _hmap_rows.append([float(_gene_scores[_g][i])
                           for i in _union_gi_sorted if i < len(go_terms)])
        _hmap_idx.append(_g)

_hmap_df = pd.DataFrame(_hmap_rows, index=_hmap_idx, columns=_go_cols)

# Heatmap
_fig_hmap = px.imshow(
    _hmap_df,
    title=f"GO Score 히트맵 (상위 {_top_k} GO 합집합, {len(_gene_sel)}개 유전자)",
    color_continuous_scale='Blues', zmin=0, zmax=1, aspect='auto',
    height=max(350, 70 * len(_gene_sel)),
)
_fig_hmap.update_layout(
    coloraxis_colorbar=dict(title='Score'), xaxis_tickangle=-45
)
st.plotly_chart(_fig_hmap, use_container_width=True, key='mgo_hmap')
_fig_hmap.add_shape(type='line', x0=-0.5, x1=len(_go_cols)-0.5,
                    y0=0, y1=0, line=dict(color='grey', dash='dot'))

# Bar chart for inter-gene comparison
_bar_rows = []
for _g in _hmap_idx:
    for _col, _val in zip(_go_cols, _hmap_df.loc[_g]):
        _bar_rows.append({'유전자': _g, 'GO term': _col, 'Score': round(float(_val), 3)})

_fig_bar = px.bar(
    pd.DataFrame(_bar_rows), x='GO term', y='Score', color='유전자',
    barmode='group', height=420, title="GO Score 그룹 비교",
)
_fig_bar.update_layout(xaxis_tickangle=-45, plot_bgcolor='white',
                       yaxis=dict(range=[0, 1.05]))
_fig_bar.add_hline(y=thr, line_dash='dash', line_color='grey',
                   annotation_text=f"thr={thr}")
st.plotly_chart(_fig_bar, use_container_width=True, key='mgo_bar')

# Top GO term per gene table
st.markdown("#### 유전자별 상위 GO term")
_top_rows = []
for _g in _gene_sel:
    if _g not in _gene_scores:
        continue
    _scores_g = _gene_scores[_g]
    _top_idxs = np.argsort(_scores_g)[-5:][::-1]
    for _rank, _idx in enumerate(_top_idxs, 1):
        if _idx < len(go_terms):
            _top_rows.append({
                '유전자': _g, '순위': _rank,
                'GO term': go_names.get(go_terms[_idx], go_terms[_idx])[:40],
                'Score': round(float(_scores_g[_idx]), 3),
                '고신뢰': '✅' if float(_scores_g[_idx]) >= thr else '',
            })

if _top_rows:
    st.dataframe(
        pd.DataFrame(_top_rows).style.background_gradient(subset=['Score'], cmap='Blues'),
        use_container_width=True, hide_index=True,
    )
