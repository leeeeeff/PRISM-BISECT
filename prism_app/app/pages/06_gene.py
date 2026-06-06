"""06_gene.py — Individual Gene Analysis: gene-level overview before isoform drill-down."""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from prism_app.app.components.basket import add_to_gene_basket, basket_gene_ids
from prism_app.app.components.search import gene_search_autocomplete

REPORTS = Path(__file__).parents[3] / 'reports'

# ── Session data ──────────────────────────────────────────────────────────────
cfg           = st.session_state.get('cfg') or {}
sm            = cfg.get('score_matrix')
ids           = cfg.get('isoform_ids')
genes         = cfg.get('gene_ids')
go_terms      = cfg.get('go_terms', [])
gnames        = cfg.get('go_names', {})
thr           = cfg.get('score_threshold', 0.4)
dtu_df        = cfg.get('dtu_df')
isoform_types = cfg.get('isoform_types')

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1e1b4b,#312e81,#4338ca);
border-radius:14px;padding:28px 48px 20px;margin-bottom:20px'>
<span style='color:#a5b4fc;font-size:0.75rem;letter-spacing:0.1em'>개별 분석</span>
<h1 style='color:white;margin:4px 0 2px;font-size:1.9rem'>🧬 유전자 분석</h1>
<p style='color:rgba(255,255,255,0.65);margin:8px 0 0;font-size:0.9rem'>
유전자 레벨 통계 · 아이소폼 분포 · 기능 모듈 · GO 프로파일 → 개별 아이소폼 분석 연동
</p></div>
""", unsafe_allow_html=True)

if sm is None:
    st.warning("데이터를 먼저 로드하세요 — 사이드바에서 Demo 또는 Upload 선택")
    st.stop()

ids_arr  = np.asarray(ids, dtype=str)
gene_arr = np.asarray(genes, dtype=str) if genes is not None else ids_arr
sm_arr   = np.asarray(sm, dtype=float)
typ_arr  = np.asarray(isoform_types, dtype=str) if isoform_types is not None else np.full(len(ids_arr), '')

# ── Classified cache ──────────────────────────────────────────────────────────
classified = st.session_state.get('classified_df')
if classified is None:
    from prism_app.core.classifier import classify_isoforms
    with st.spinner("아이소폼 분류 중…"):
        classified = classify_isoforms(
            sm, ids, genes, go_terms, score_threshold=thr, dtu_df=dtu_df,
        )
        st.session_state['classified_df'] = classified

# ── Module lookup ─────────────────────────────────────────────────────────────
def _load_mod_tsv():
    p = REPORTS / 'brain_isoform_modules.tsv'
    if p.exists():
        try:
            return pd.read_csv(p, sep='\t')
        except Exception:
            pass
    return None

_mod_df = st.session_state.get('_gene_mod_df_cache')
if _mod_df is None:
    _mod_df = _load_mod_tsv()
if _mod_df is not None:
    st.session_state['_gene_mod_df_cache'] = _mod_df

# ── BISECT case lookup ────────────────────────────────────────────────────────
def _load_bisect_genes() -> set:
    import json
    p = Path(__file__).parents[2] / 'data' / 'demo' / 'bisect_cases.json'
    if p.exists():
        try:
            return {c.get('gene', '').upper() for c in json.load(open(p))}
        except Exception:
            pass
    return set()

_bisect_genes = st.session_state.get('_bisect_gene_set_cache')
if _bisect_genes is None:
    _bisect_genes = _load_bisect_genes()
st.session_state['_bisect_gene_set_cache'] = _bisect_genes

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH UI
# ══════════════════════════════════════════════════════════════════════════════
_init_gene = str(st.session_state.get('search_gene', ''))
_gene_pool = cfg.get('gene_ids')  # ndarray of gene names per isoform

_query = gene_search_autocomplete(
    'gene_page_query',
    _gene_pool,
    label="유전자 검색",
    placeholder="예: NDUFS4, KIF21B, PTPRF — 입력하면 연관 유전자 자동완성",
    max_suggestions=8,
    pre_fill=_init_gene,
)
st.session_state['search_gene'] = _query

if not _query.strip():
    # Show quick-access basket items
    _bids = basket_gene_ids()
    if _bids:
        st.caption("바스켓에서 빠르게 선택:")
        _bcols = st.columns(min(len(_bids), 6))
        for _bi, _bg in enumerate(_bids[:6]):
            with _bcols[_bi]:
                if st.button(f"🧬 {_bg}", key=f'gene_bk_{_bi}', use_container_width=True):
                    st.session_state['search_gene'] = _bg
                    st.session_state['gene_page_query'] = _bg
                    st.rerun()
    st.info("유전자명을 입력하거나 바스켓에서 선택하세요.")
    st.stop()

# ── Match gene ────────────────────────────────────────────────────────────────
_q = _query.strip().upper()
_mask = np.char.upper(gene_arr) == _q
if not _mask.any():
    _mask = np.array([_q in str(g).upper() for g in gene_arr])
if not _mask.any():
    st.warning(f"'{_query}' — 데이터셋에서 찾을 수 없습니다.")
    st.stop()

# If multiple genes matched, pick one
_matched_genes = np.unique(gene_arr[_mask])
if len(_matched_genes) > 1:
    _chosen = st.selectbox("여러 유전자 매칭 — 선택:", sorted(_matched_genes), key='gene_page_sel')
    _mask = np.char.upper(gene_arr) == _chosen.upper()
    _gene_name = _chosen
else:
    _gene_name = str(_matched_genes[0])

_hit_ids   = ids_arr[_mask]
_hit_sm    = sm_arr[_mask]
_hit_types = typ_arr[_mask]
_max_per   = _hit_sm.max(axis=1)

# ══════════════════════════════════════════════════════════════════════════════
# GENE REPORT HEADER
# ══════════════════════════════════════════════════════════════════════════════
_n_iso     = int(_mask.sum())
_n_high    = int((_max_per >= thr).sum())
_bisect_ok = _gene_name.upper() in _bisect_genes

st.markdown(f"""
<div style='background:linear-gradient(90deg,#1e1b4b,#312e81);border-radius:12px;
padding:16px 24px 12px;margin-bottom:16px'>
<span style='color:#a5b4fc;font-size:0.8rem'>GENE REPORT</span>
<h2 style='color:white;margin:4px 0 2px;font-size:1.7rem'>{_gene_name.upper()}</h2>
<span style='color:#c7d2fe;font-size:0.9rem'>
{_n_iso}개 아이소폼 · 고신뢰 {_n_high}개
{"· <b>BISECT PASS ✅</b>" if _bisect_ok else ""}
</span></div>
""", unsafe_allow_html=True)

# ── Auto-tag values for basket add ───────────────────────────────────────────
_cls_pre = st.session_state.get('classified_df')
_auto_tag_scenario = None
_auto_tag_module   = None
if _cls_pre is not None:
    _pre_mask = _cls_pre['isoform_id'].isin(ids_arr[_mask])
    _pre_cls  = _cls_pre[_pre_mask]
    if len(_pre_cls):
        _sc_counts = _pre_cls['scenario'].value_counts()
        _auto_tag_scenario = int(_sc_counts.idxmin()) if len(_sc_counts) else None
        # pick dominant non-S4 scenario if exists
        for _s in [1, 2, 3]:
            if _s in _sc_counts.index and _sc_counts[_s] > 0:
                _auto_tag_scenario = _s
                break
if _mod_df is not None:
    _gmod_pre = _mod_df[_mod_df['gene'].str.upper() == _gene_name.upper()]
    if len(_gmod_pre):
        _auto_tag_module = int(_gmod_pre['primary_module'].mode().iloc[0])

# ── Top action row ────────────────────────────────────────────────────────────
_ac1, _ac2, _ac3 = st.columns(3)
with _ac1:
    _in_basket = _gene_name.upper() in [g.upper() for g in basket_gene_ids()]
    if _in_basket:
        st.success(f"✅ 바스켓에 있음")
    else:
        if st.button("➕ 유전자 바스켓 추가", key='gene_add_basket', use_container_width=True):
            add_to_gene_basket(
                _gene_name, source_page='gene_page',
                tag_scenario=_auto_tag_scenario,
                tag_module=_auto_tag_module,
            )
            st.toast(f"✅ {_gene_name} 바스켓 추가")
            st.rerun()
with _ac2:
    if st.button("🔬 아이소폼 상세 분석으로 →", key='gene_goto_isoform', use_container_width=True):
        st.session_state['search_gene'] = _gene_name
        st.session_state['gene_from_gene_page'] = True
        st.switch_page("pages/06_isoform.py")
with _ac3:
    if st.button("🎯 타겟 탐색에서 보기", key='gene_goto_target', use_container_width=True):
        st.session_state['search_gene'] = _gene_name
        st.session_state['auto_search'] = True
        st.switch_page("pages/05_targets.py")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Scenario Distribution
# ══════════════════════════════════════════════════════════════════════════════
_cls_gene = classified[classified['isoform_id'].isin(_hit_ids)] if len(classified) else pd.DataFrame()

col_s, col_m, col_g = st.columns([1, 1, 2])

with col_s:
    st.markdown("##### Scenario 분포")
    if len(_cls_gene):
        _sc_counts = _cls_gene['scenario'].value_counts().to_dict()
        _sc_labels = {1: '🔴 S1 기능스위치', 2: '🟠 S2 발현스위치',
                      3: '🟢 S3 신규기능', 4: '⬜ S4 배경'}
        _sc_colors = {1: '#ef4444', 2: '#f97316', 3: '#22c55e', 4: '#94a3b8'}
        for _si in [1, 2, 3, 4]:
            _cnt = _sc_counts.get(_si, 0)
            if _cnt > 0:
                st.markdown(
                    f"<span style='background:{_sc_colors[_si]};color:white;"
                    f"border-radius:4px;padding:2px 8px;font-size:0.82rem'>"
                    f"{_sc_labels[_si]}: {_cnt}</span>",
                    unsafe_allow_html=True,
                )
    else:
        st.caption("분류 데이터 없음")

with col_m:
    st.markdown("##### 기능 모듈")
    if _mod_df is not None:
        _g_mods = _mod_df[_mod_df['gene'].str.upper() == _gene_name.upper()]
        if len(_g_mods):
            _pmod = _g_mods['primary_module'].mode().iloc[0]
            _plab = _g_mods[_g_mods['primary_module'] == _pmod]['module_label'].iloc[0]
            st.metric(f"M{int(_pmod)}", _plab[:40] if _plab else '—')
        else:
            st.caption("모듈 데이터 없음")
    else:
        st.caption("모듈 파일 없음")
    if dtu_df is not None:
        _dtu_hit = dtu_df[dtu_df['gene_id'].str.upper() == _gene_name.upper()]
        _n_dtu = len(_dtu_hit['condition'].unique()) if len(_dtu_hit) else 0
        st.metric("DTU 발생 조건 수", str(_n_dtu))

# ── Top GO terms ──────────────────────────────────────────────────────────────
with col_g:
    st.markdown("##### Top GO 기능 (유전자 최고 아이소폼 기준)")
    if go_terms:
        _gene_max_go = _hit_sm.max(axis=0)
        _top3_idx    = np.argsort(_gene_max_go)[-5:][::-1]
        _top3_rows   = [
            {'GO term': gnames.get(go_terms[i], go_terms[i])[:45],
             'Score':   round(float(_gene_max_go[i]), 3),
             '고신뢰':  '✅' if float(_gene_max_go[i]) >= thr else ''}
            for i in _top3_idx if i < len(go_terms)
        ]
        st.dataframe(pd.DataFrame(_top3_rows), hide_index=True, use_container_width=True,
                     height=220)
    else:
        st.caption("GO term 데이터 없음")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Isoform Ranking
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("##### 아이소폼 랭킹 (max PRISM score 기준)")

_iso_rows = []
for _i, _iid in enumerate(_hit_ids):
    _top_go_i = int(np.argmax(_hit_sm[_i]))
    _tgo_name = gnames.get(go_terms[_top_go_i], go_terms[_top_go_i])[:35] if go_terms else '—'
    _scen     = int(_cls_gene[_cls_gene['isoform_id'] == _iid]['scenario'].iloc[0]) \
                if len(_cls_gene) and _iid in _cls_gene['isoform_id'].values else 4
    _iso_rows.append({
        '아이소폼':   _iid,
        'Max score':  round(float(_max_per[_i]), 3),
        'Type':       _hit_types[_i],
        'Scenario':   f"S{_scen}",
        'Top GO':     _tgo_name,
    })

_iso_df = (pd.DataFrame(_iso_rows)
           .sort_values('Max score', ascending=False)
           .reset_index(drop=True))

_ic1, _ic2 = st.columns([3, 2])
with _ic1:
    st.dataframe(
        _iso_df.style.background_gradient(subset=['Max score'], cmap='Blues', vmin=0, vmax=1),
        use_container_width=True, hide_index=True,
        height=min(500, 50 + 35 * min(15, len(_iso_df))),
    )
with _ic2:
    if len(_iso_df) <= 30:
        _fig_rank = px.bar(
            _iso_df.head(20), x='Max score', y='아이소폼',
            orientation='h', color='Scenario',
            color_discrete_map={'S1': '#ef4444', 'S2': '#f97316',
                                'S3': '#22c55e', 'S4': '#94a3b8'},
            title=f"아이소폼 Max Score (상위 20개)",
            height=min(500, 80 + 22 * min(20, len(_iso_df))),
        )
        _fig_rank.update_layout(
            plot_bgcolor='white', yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=True,
        )
        _fig_rank.add_vline(x=thr, line_dash='dash', line_color='grey',
                            annotation_text=f"thr={thr}")
        st.plotly_chart(_fig_rank, use_container_width=True, key='gene_iso_rank')
    else:
        st.caption(f"{len(_iso_df)}개 아이소폼 — 차트는 30개 이하 유전자만 표시")

st.divider()

# ── Proceed to isoform analysis ───────────────────────────────────────────────
st.markdown(
    "유전자 레벨 분석 후, 개별 아이소폼을 선택해서 GO 프로파일 · GAIN/LOSS · Within-gene 비교를 확인하세요."
)
if st.button("🔬 개별 아이소폼 분석으로 이동 →", key='gene_to_isoform_bottom',
             use_container_width=False, type='primary'):
    st.session_state['search_gene'] = _gene_name
    st.session_state['gene_from_gene_page'] = True
    st.switch_page("pages/06_isoform.py")
