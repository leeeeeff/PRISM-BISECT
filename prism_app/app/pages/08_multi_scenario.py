"""08_multi_scenario.py — Multi-gene Scenario distribution comparison."""
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
from prism_app.app.components.search import gene_search_autocomplete

init_basket()

cfg   = st.session_state.get('cfg') or {}
sm    = cfg.get('score_matrix')
ids   = cfg.get('isoform_ids')
gins  = cfg.get('gene_ids')
has_data = sm is not None and ids is not None

st.markdown("## 📊 Scenario 다중 분석")
st.caption(
    "여러 유전자의 Scenario 분포(S1 기능스위치 / S2 발현스위치 / S3 신규기능 / S4 배경)를 "
    "한눈에 비교합니다."
)
st.divider()

# ── Gene selection ────────────────────────────────────────────────────────────
_sc_cases = [c for c in get_analysis_cases()
             if c.get('mode') == 'multi' and c.get('item_type') == 'gene'
             and c.get('axis') == 'scenario']
_gene_pool = cfg.get('gene_ids')
_all_unique = list(dict.fromkeys(str(g) for g in _gene_pool)) if _gene_pool is not None else []

_col_left, _col_right = st.columns([3, 2])
with _col_left:
    st.markdown("#### 유전자 선택")
    if _sc_cases:
        _case_opts = {f"케이스 #{c['case_id']} — {', '.join(c['items'][:3])}{'...' if len(c['items'])>3 else ''}": c
                      for c in _sc_cases}
        _sel_case = st.selectbox("저장된 케이스 불러오기",
                                 ["— 직접 선택 —"] + list(_case_opts.keys()),
                                 key='msc_case_sel')
        if _sel_case != "— 직접 선택 —":
            _default_genes = _case_opts[_sel_case]['items']
        else:
            _default_genes = []
    else:
        st.caption("저장된 Scenario 케이스 없음 — 사이드바에서 생성하거나 아래서 직접 선택하세요.")
        _default_genes = []

    _gene_sel = st.multiselect(
        "분석할 유전자 (다중 선택 · 검색 가능)",
        options=_all_unique,
        default=[g for g in _default_genes if g in _all_unique],
        key='msc_gene_sel',
        placeholder="유전자명 입력하여 검색...",
    )

with _col_right:
    if _gene_sel:
        st.markdown("#### 선택된 유전자")
        _badge_html = ' '.join(
            f"<span style='background:#1e40af;color:white;border-radius:4px;"
            f"padding:3px 10px;font-size:0.82rem'>{g}</span>"
            for g in _gene_sel
        )
        st.markdown(_badge_html, unsafe_allow_html=True)

st.divider()

# ── Analysis ──────────────────────────────────────────────────────────────────
if not _gene_sel:
    st.info("위에서 유전자를 선택하면 Scenario 분포 분석이 표시됩니다.")
    st.stop()

_classified = st.session_state.get('classified_df')
if _classified is None:
    st.warning("Scenario 분류 데이터 없음 — 먼저 **시나리오 & 분석** 페이지에서 분류를 실행하세요.")
    st.stop()

if not has_data:
    st.warning("데이터를 먼저 로드하세요 (사이드바 → Demo 또는 Upload).")
    st.stop()

_gins_arr = np.array(gins, dtype=str) if gins is not None else None
_ids_arr  = np.array(ids, dtype=str)
_sc_labels = {1: 'S1 기능스위치', 2: 'S2 발현스위치', 3: 'S3 신규기능', 4: 'S4 배경'}
_sc_colors = {'S1 기능스위치': '#ef4444', 'S2 발현스위치': '#f97316',
              'S3 신규기능': '#22c55e', 'S4 배경': '#94a3b8'}

_sc_rows = []
_missing = []
for _g in _gene_sel:
    _gm = np.array([x.upper() == _g.upper()
                    for x in (_gins_arr if _gins_arr is not None else _ids_arr)])
    if not _gm.any():
        _missing.append(_g)
        continue
    _gcls = _classified[_classified['isoform_id'].isin(_ids_arr[_gm])]
    for _sc in [1, 2, 3, 4]:
        _sc_rows.append({'유전자': _g, 'Scenario': _sc_labels[_sc],
                         '아이소폼 수': int((_gcls['scenario'] == _sc).sum())})

if _missing:
    st.warning(f"다음 유전자는 데이터에서 찾을 수 없습니다: {', '.join(_missing)}")

if not _sc_rows:
    st.info("선택된 유전자에 Scenario 데이터가 없습니다.")
    st.stop()

_sc_df = pd.DataFrame(_sc_rows)

# Stacked bar chart
_fig = px.bar(
    _sc_df, x='유전자', y='아이소폼 수', color='Scenario',
    barmode='stack', color_discrete_map=_sc_colors,
    title=f"유전자별 Scenario 분포 ({len(_gene_sel)}개 유전자)",
    height=480,
)
_fig.update_layout(plot_bgcolor='white', xaxis_title='유전자', yaxis_title='아이소폼 수')
st.plotly_chart(_fig, use_container_width=True, key='msc_bar')

# Pivot table
_pivot = _sc_df.pivot(index='유전자', columns='Scenario', values='아이소폼 수').fillna(0)
_pivot['전체'] = _pivot.sum(axis=1)
_s1s2s3 = [c for c in _pivot.columns if c not in ['S4 배경', '전체']]
if _s1s2s3:
    _pivot['지배 Scenario'] = _pivot[_s1s2s3].idxmax(axis=1)
_pivot = _pivot.reset_index()

st.markdown("#### Scenario 피벗 테이블")
st.dataframe(
    _pivot.style.background_gradient(subset=[c for c in _pivot.columns
                                             if c in _sc_labels.values()], cmap='Oranges'),
    use_container_width=True, hide_index=True,
)

# Per-gene percentage breakdown
st.markdown("#### Scenario 비율 (%) 비교")
_pct_rows = []
for _g in _gene_sel:
    _sub = _sc_df[_sc_df['유전자'] == _g]
    _tot = _sub['아이소폼 수'].sum()
    if _tot > 0:
        for _, _row in _sub.iterrows():
            _pct_rows.append({'유전자': _g, 'Scenario': _row['Scenario'],
                              '비율 (%)': round(_row['아이소폼 수'] / _tot * 100, 1)})

if _pct_rows:
    _fig_pct = px.bar(
        pd.DataFrame(_pct_rows), x='유전자', y='비율 (%)', color='Scenario',
        barmode='stack', color_discrete_map=_sc_colors,
        title="유전자별 Scenario 비율 (%)", height=380,
    )
    _fig_pct.update_layout(plot_bgcolor='white', yaxis=dict(range=[0, 105]))
    st.plotly_chart(_fig_pct, use_container_width=True, key='msc_pct')
