"""09_multi_module.py — Multi-gene functional module comparison."""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd
import streamlit as st
import plotly.express as px

from prism_app.app.components.basket import init_basket, get_analysis_cases

init_basket()

cfg  = st.session_state.get('cfg') or {}
REPORTS = Path(__file__).parents[3] / 'reports'
_mod_path = REPORTS / 'brain_isoform_modules.tsv'

st.markdown("## 🗂️ 기능 모듈 다중 분석")
st.caption(
    "여러 유전자의 기능 모듈 소속을 비교합니다. "
    "같은 모듈에 속하는 유전자들은 유사한 세포 기능을 가질 가능성이 높습니다."
)
st.divider()

if not _mod_path.exists():
    st.error(f"모듈 데이터 파일을 찾을 수 없습니다: {_mod_path}")
    st.stop()

@st.cache_data(show_spinner=False)
def _load_module_df() -> pd.DataFrame:
    return pd.read_csv(_mod_path, sep='\t')

_mod_df = _load_module_df()

# ── Gene selection ────────────────────────────────────────────────────────────
_mod_cases = [c for c in get_analysis_cases()
              if c.get('mode') == 'multi' and c.get('item_type') == 'gene'
              and c.get('axis') == 'module']
_gene_pool = cfg.get('gene_ids')
_all_unique = list(dict.fromkeys(str(g) for g in _gene_pool)) if _gene_pool is not None else []
# Also use genes in module file
_mod_genes = sorted(_mod_df['gene'].unique().tolist()) if 'gene' in _mod_df.columns else []
_pool_combined = sorted(set(_all_unique) | set(_mod_genes))

_col_left, _col_right = st.columns([3, 2])
with _col_left:
    st.markdown("#### 유전자 선택")
    if _mod_cases:
        _case_opts = {f"케이스 #{c['case_id']} — {', '.join(c['items'][:3])}{'...' if len(c['items'])>3 else ''}": c
                      for c in _mod_cases}
        _sel_case = st.selectbox("저장된 케이스 불러오기",
                                 ["— 직접 선택 —"] + list(_case_opts.keys()),
                                 key='mmod_case_sel')
        _default_genes = _case_opts[_sel_case]['items'] if _sel_case != "— 직접 선택 —" else []
    else:
        st.caption("저장된 모듈 케이스 없음 — 아래서 직접 선택하세요.")
        _default_genes = []

    _gene_sel = st.multiselect(
        "분석할 유전자 (다중 선택 · 검색 가능)",
        options=_pool_combined,
        default=[g for g in _default_genes if g in _pool_combined],
        key='mmod_gene_sel',
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
    st.info("위에서 유전자를 선택하면 기능 모듈 분석이 표시됩니다.")
    st.stop()

_mod_rows = []
_missing = []
for _g in _gene_sel:
    _gmod = _mod_df[_mod_df['gene'].str.upper() == _g.upper()] if 'gene' in _mod_df.columns else pd.DataFrame()
    if _gmod.empty:
        _missing.append(_g)
        continue
    for _, _mr in _gmod.iterrows():
        _lbl = str(_mr.get('module_label', ''))
        _mod_rows.append({
            '유전자': _g,
            '모듈': f"M{int(_mr['primary_module'])} {_lbl[:20]}",
            '모듈 번호': int(_mr['primary_module']),
            '스코어': float(_mr.get('module_score', 1.0)),
        })

if _missing:
    st.warning(f"모듈 데이터 없는 유전자: {', '.join(_missing)}")

if not _mod_rows:
    st.info("선택된 유전자에 모듈 데이터가 없습니다.")
    st.stop()

_df = pd.DataFrame(_mod_rows)
_pivot = _df.groupby(['유전자', '모듈'])['스코어'].max().unstack(fill_value=0)

# Heatmap
_fig_hmap = px.imshow(
    _pivot,
    title=f"유전자 × 기능 모듈 히트맵 ({len(_gene_sel)}개 유전자)",
    color_continuous_scale='Blues', aspect='auto',
    height=max(350, 75 * len(_gene_sel)),
)
_fig_hmap.update_layout(xaxis_tickangle=-40)
st.plotly_chart(_fig_hmap, use_container_width=True, key='mmod_hmap')

# Dominant module per gene
st.markdown("#### 지배 모듈 요약")
_dom_df = _pivot.idxmax(axis=1).rename('지배 모듈').reset_index()
_dom_df['모듈 스코어'] = [
    float(_pivot.loc[_g, _dom_df[_dom_df['유전자'] == _g]['지배 모듈'].values[0]])
    for _g in _dom_df['유전자']
]
st.dataframe(_dom_df.style.background_gradient(subset=['모듈 스코어'], cmap='Blues'),
             use_container_width=True, hide_index=True)

# Bar chart of module scores
_fig_bar = px.bar(
    _df, x='모듈', y='스코어', color='유전자',
    barmode='group', title="기능 모듈 스코어 비교",
    height=420,
)
_fig_bar.update_layout(plot_bgcolor='white', xaxis_tickangle=-35)
st.plotly_chart(_fig_bar, use_container_width=True, key='mmod_bar')

# Co-occurrence: genes sharing same dominant module
st.markdown("#### 모듈 공유 유전자 클러스터")
_dom_map: dict = {}
for _, _row in _dom_df.iterrows():
    _mod = _row['지배 모듈']
    _dom_map.setdefault(_mod, []).append(_row['유전자'])
for _mod_key, _mod_genes in sorted(_dom_map.items()):
    _clr = '#22c55e' if len(_mod_genes) > 1 else '#94a3b8'
    _gene_str = ' · '.join(_mod_genes)
    st.markdown(
        f"<span style='background:{_clr};color:white;border-radius:4px;"
        f"padding:3px 8px;font-size:0.8rem'>{_mod_key}</span>&nbsp; {_gene_str}",
        unsafe_allow_html=True,
    )
