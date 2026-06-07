"""09_multi_module.py — Multi-gene functional module comparison."""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import json
import pandas as pd
import streamlit as st
import plotly.express as px

@st.cache_data(show_spinner=False)
def _load_bisect_for_module() -> dict:
    """Return {GENE_UPPER: [case_dict, ...]} from bisect_cases.json."""
    _p = Path(__file__).parents[2] / 'data' / 'demo' / 'bisect_cases.json'
    if not _p.exists():
        return {}
    try:
        _cases = json.load(open(_p))
        _out: dict = {}
        for _c in _cases:
            _g = str(_c.get('gene', '') or '').upper()
            _out.setdefault(_g, []).append(_c)
        return _out
    except Exception:
        return {}

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

# ── BISECT 연동: 모듈 클러스터 → 경로 수렴 증거 ─────────────────────────────
_bisect_lookup = _load_bisect_for_module()
st.divider()
st.markdown("#### 🧫 BISECT 연동 — 모듈 경로 수렴 분석")
st.caption(
    "선택된 유전자 중 BISECT PASS 케이스가 있는지 확인하고, "
    "같은 기능 모듈 내 복수 유전자가 BISECT 검증을 받은 경우를 경로 수렴 증거로 표시합니다."
)

# Build BISECT status for each selected gene
_bisect_rows = []
_TIER_LABEL = {
    'tier1_functional_switch': '🔬 T1 스위치',
    'tier2_functional_loss':   '📉 T2 소실',
    'tier2_complex_loss':      '⚡ T2 ComplexI',
    'tier2_partial_change':    '↔ T2 변화',
    'tier2_gain_no_direction': '↑ T2 획득',
    'tier3_gene_median':       '〜 T3 추정',
    'tier3_structural_only':   '△ T3 구조',
    'tier3_no_match':          '? T3 미매칭',
}
_COMPLEX1 = {'NDUFS4', 'NDUFS7', 'NDUFS8'}

for _g in _gene_sel:
    _cases = _bisect_lookup.get(_g.upper(), [])
    if _cases:
        for _case in _cases:
            _tier_raw = str(_case.get('prism_tier') or '')
            # Runtime inference if null (same logic as 07_bisect.py)
            if not _tier_raw or _tier_raw == 'None':
                _af_g = str(_case.get('af_gained_confident', '') or '').strip()
                _af_l = str(_case.get('af_lost_confident', '') or '').strip()
                _dg   = str(_case.get('domains_gained', '') or '').strip()
                _dl   = str(_case.get('domains_lost', '') or '').strip()
                if _g.upper() in _COMPLEX1:
                    _tier_raw = 'tier2_complex_loss'
                elif _af_g:
                    _tier_raw = 'tier1_functional_switch'
                elif _af_l:
                    _tier_raw = 'tier2_functional_loss'
                elif _dg or _dl:
                    _tier_raw = 'tier2_partial_change'
                else:
                    _tier_raw = 'tier3_structural_only'
            _bisect_rows.append({
                '유전자': _g,
                'BISECT': '✅ PASS',
                '세포유형': str(_case.get('cell_type', '—')),
                'Tier': _TIER_LABEL.get(_tier_raw, _tier_raw),
                'Δ Usage': round(float(_case.get('delta', 0) or 0), 3),
                'DTU p': _case.get('dtu_p'),
                'Domains +': str(_case.get('domains_gained', '') or '')[:30] or '—',
                'Domains −': str(_case.get('domains_lost', '') or '')[:30] or '—',
            })
    else:
        _bisect_rows.append({
            '유전자': _g, 'BISECT': '—', '세포유형': '—',
            'Tier': '—', 'Δ Usage': None, 'DTU p': None,
            'Domains +': '—', 'Domains −': '—',
        })

if _bisect_rows:
    _bdf_mod = pd.DataFrame(_bisect_rows)
    _n_pass  = int((_bdf_mod['BISECT'] == '✅ PASS').sum())

    # Summary badge
    st.markdown(
        f"<span style='background:#15803d;color:white;border-radius:6px;"
        f"padding:4px 14px;font-size:0.9rem;font-weight:600'>"
        f"✅ BISECT PASS: {_n_pass}건</span>"
        f"&nbsp;<span style='color:#6b7280;font-size:0.85rem'>"
        f"/ {len(_gene_sel)}개 선택 유전자</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Style: highlight PASS rows
    def _style_bisect(row):
        if row.get('BISECT') == '✅ PASS':
            return ['background-color: #f0fdf4'] * len(row)
        return ['color: #9ca3af'] * len(row)

    _show_cols = ['유전자', 'BISECT', '세포유형', 'Tier', 'Δ Usage', 'DTU p', 'Domains +', 'Domains −']
    st.dataframe(
        _bdf_mod[_show_cols].style.apply(_style_bisect, axis=1).format({
            'Δ Usage': lambda v: f'{v:.3f}' if v == v and v is not None else '—',
            'DTU p':   lambda v: f'{v:.2e}' if v == v and v is not None else '단일조건',
        }),
        use_container_width=True, hide_index=True,
    )

    # ── 경로 수렴 분석: 같은 모듈에 BISECT PASS 유전자가 2+개 ────────────────
    _pass_genes = set(_bdf_mod[_bdf_mod['BISECT'] == '✅ PASS']['유전자'].unique())
    _convergence = {
        _mod: [_g for _g in _gs if _g in _pass_genes]
        for _mod, _gs in _dom_map.items()
        if sum(1 for _g in _gs if _g in _pass_genes) >= 2
    }

    if _convergence:
        st.markdown("##### 🔴 경로 수렴 클러스터 (같은 모듈 내 BISECT 2개 이상)")
        for _conv_mod, _conv_genes in sorted(_convergence.items()):
            _is_complex1 = all(g.upper() in _COMPLEX1 for g in _conv_genes)
            _badge_clr = '#7f1d1d' if _is_complex1 else '#1e3a8a'
            _badge_txt = '⚡ Complex I 삼각 수렴' if _is_complex1 else '🔗 경로 수렴'
            st.markdown(
                f"<div style='border-left:4px solid {_badge_clr};padding:8px 14px;"
                f"background:#f8fafc;border-radius:0 8px 8px 0;margin:6px 0'>"
                f"<span style='background:{_badge_clr};color:white;border-radius:4px;"
                f"padding:2px 8px;font-size:0.78rem'>{_badge_txt}</span>&nbsp;"
                f"<b>{_conv_mod}</b>&nbsp;—&nbsp;"
                f"{'&nbsp;·&nbsp;'.join(_conv_genes)}</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "같은 기능 모듈의 복수 유전자가 모두 BISECT 검증 통과 = "
            "해당 모듈이 나타내는 생물학적 경로 전체가 AD에서 교란되고 있음을 시사."
        )
    elif _n_pass > 0:
        st.info("선택된 BISECT PASS 유전자들은 서로 다른 기능 모듈에 속합니다 (경로 수렴 없음).")
    else:
        st.info("선택된 유전자 중 BISECT PASS 케이스가 없습니다. BISECT 검증 유전자를 추가해보세요."
                " (예: NDUFS4, DLG1, KIF21B, DMD)")
