"""06_isoform.py — Individual Isoform Analysis: GO profile, GAIN/LOSS, within-gene comparison."""
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

@st.cache_data(show_spinner=False)
def _load_iso_diu_data() -> 'pd.DataFrame | None':
    """Load 8 cell-type DIU CSVs for isoform-level concordance display."""
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

@st.cache_data(show_spinner=False)
def _load_iso_splice_div() -> dict:
    """Return {GENE_UPPER: {'dist': float, 'frac': float}} from splicing_delta_v2 (muscle)."""
    _SD   = Path(__file__).parents[3] / 'hMuscle/results_isoform/features/splicing/splicing_delta_v2.npy'
    _GENE = Path(__file__).parents[3] / 'hMuscle/model/my_gene_list_fixed.npy'
    _SYM  = Path(__file__).parents[3] / 'hMuscle/data/raw_data/data/id_lists/ensembl_to_symbol.txt'
    if not _SD.exists() or not _GENE.exists():
        return {}
    _sd   = np.load(_SD).astype(np.float32)
    _graw = np.load(_GENE, allow_pickle=True)
    _sym_map: dict = {}
    if _SYM.exists():
        for _line in open(_SYM):
            _parts = _line.strip().split('\t')
            if len(_parts) >= 2:
                _sym_map[_parts[0]] = _parts[1]
    _genes = np.array([_sym_map.get(str(g), str(g)) for g in _graw])
    _out: dict = {}
    for _ug in np.unique(_genes):
        _idx = np.where(_genes == _ug)[0]
        if len(_idx) < 2:
            continue
        _g_sd = _sd[_idx]
        _diffs = []
        for _i in range(len(_idx)):
            for _j in range(_i + 1, len(_idx)):
                _diffs.append(float(np.linalg.norm(_g_sd[_i] - _g_sd[_j])))
        _frac = float(((_g_sd != 0).any(axis=1)).mean())
        _out[_ug.upper()] = {'dist': max(_diffs), 'frac': _frac}
    return _out

from prism_app.app.components.analysis_context import render_analysis_context, context_for_report
from prism_app.app.components.basket import add_to_gene_basket, add_to_isoform_basket, basket_gene_ids
from prism_app.app.components.search import gene_search_autocomplete, isoform_search_autocomplete

REPORTS = Path(__file__).parents[3] / 'reports'

# ── Session data ──────────────────────────────────────────────────────────────
cfg           = st.session_state.get('cfg') or {}
sm            = cfg.get('score_matrix')
ids           = cfg.get('isoform_ids')
genes         = cfg.get('gene_ids')
go_terms      = cfg.get('go_terms', [])
gnames        = cfg.get('go_names', {})
thr           = cfg.get('score_threshold', 0.4)
isoform_types = cfg.get('isoform_types')
dtu_df        = cfg.get('dtu_df')
tissue        = cfg.get('tissue', '')

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#052e16,#14532d,#166534);
border-radius:14px;padding:28px 48px 20px;margin-bottom:20px'>
<h1 style='color:white;margin:0;font-size:1.9rem'>🔬 개별 아이소폼 분석</h1>
<p style='color:rgba(255,255,255,0.7);margin:8px 0 0;font-size:0.9rem'>
아이소폼/유전자 검색 · GO 기능 프로파일 · GAIN/LOSS 분석 · within-gene 비교
</p></div>
""", unsafe_allow_html=True)

if sm is None:
    st.warning("데이터를 먼저 로드하세요 — 사이드바에서 Demo 또는 Upload 선택")
    st.stop()

ids_arr   = np.asarray(ids, dtype=str)
gene_arr  = np.asarray(genes, dtype=str) if genes is not None else ids_arr
sm_arr    = np.asarray(sm, dtype=float)
typ_arr   = np.asarray(isoform_types, dtype=str) if isoform_types is not None else np.full(len(ids_arr), '')

# ── Module lookup (lazy load) ─────────────────────────────────────────────────
def _get_go_mod_map():
    _mods = st.session_state.get('user_modules') or st.session_state.get('brain672_modules')
    if _mods is None:
        _p = REPORTS / 'brain_go_modules_672.json'
        if _p.exists():
            import json as _json
            try:
                _mods = _json.loads(_p.read_text())
                st.session_state['brain672_modules'] = _mods
            except Exception:
                return {}
    return (_mods or {}).get('go_module_map', {})

# ── Classified lookup ─────────────────────────────────────────────────────────
classified = st.session_state.get('classified_df')
if classified is None:
    from prism_app.core.classifier import classify_isoforms
    with st.spinner("아이소폼 분류 중…"):
        classified = classify_isoforms(
            sm, ids, genes, go_terms, score_threshold=thr, dtu_df=dtu_df,
        )
        st.session_state['classified_df'] = classified

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH UI
# ══════════════════════════════════════════════════════════════════════════════

# Sync session_state search
# gene_from_gene_page = True means we arrived from 06_gene.py with gene pre-loaded
_from_gene_page_init = st.session_state.get('gene_from_gene_page', False)
_init_val = str(st.session_state.get('search_gene', '') or
                st.session_state.get('deepdive_iso_id', ''))

st.markdown("### 유전자 / 아이소폼 검색")
_gene_pool_iso  = cfg.get('gene_ids')    # unique gene names (from gene_ids array)
_isoform_pool   = cfg.get('isoform_ids') # isoform ID array

# Determine if the init value looks like an isoform ID or a gene name
_is_init_iso = '-' in _init_val and not _gene_pool_iso is None

st.caption("유전자명 또는 아이소폼 ID 입력 — 입력하는 동안 연관 항목이 자동 표시됩니다")
_query = gene_search_autocomplete(
    'iso_search_input',
    _isoform_pool if _is_init_iso else _gene_pool_iso,
    label="유전자명 또는 아이소폼 ID",
    placeholder="예: KIF21B, NDUFS4, NDUFS4-201",
    max_suggestions=8,
    pre_fill=_init_val,
)
_search_btn = st.button("🔍 분석", key='iso_search_btn')

if _query:
    st.session_state['search_gene'] = _query

if not _query:
    # Show quick-access candidates when no search
    st.info("유전자명 또는 아이소폼 ID를 입력하고 🔍 분석 버튼을 누르세요.")
    if classified is not None and len(classified):
        st.markdown("**빠른 접근 — 주요 후보:**")
        _quick = classified.sort_values('max_score', ascending=False).head(8)
        _qcols = st.columns(4)
        for _qi, (_, _qrow) in enumerate(_quick.iterrows()):
            _qg = str(_qrow.get('gene_id', _qrow['isoform_id']))
            if _qcols[_qi % 4].button(
                f"🔬 {_qg}", key=f"quick_{_qi}", use_container_width=True,
                help=f"score={_qrow['max_score']:.3f}",
            ):
                st.session_state['search_gene'] = _qg
                st.rerun()
    st.stop()

# ── Match isoforms ────────────────────────────────────────────────────────────
_q_upper = _query.upper()
# Try exact gene match first
_mask = np.char.upper(gene_arr) == _q_upper
if not _mask.any():
    _mask = np.char.upper(ids_arr) == _q_upper
if not _mask.any():
    _mask = np.array([_q_upper in str(g).upper() for g in gene_arr])
if not _mask.any():
    _mask = np.array([_q_upper in str(i).upper() for i in ids_arr])
if not _mask.any():
    st.warning(f"**'{_query}'**을(를) 데이터에서 찾지 못했습니다.")
    st.stop()

_hit_ids   = ids_arr[_mask]
_hit_sm    = sm_arr[_mask]
_hit_types = typ_arr[_mask]
_hit_genes = gene_arr[_mask]
_gene_name = str(np.unique(_hit_genes)[0]) if len(np.unique(_hit_genes)) == 1 else _query

# If multiple genes matched → pick one
if len(np.unique(_hit_genes)) > 1:
    _chosen_gene = st.selectbox(
        "여러 유전자가 검색됐습니다 — 분석할 유전자 선택:",
        options=sorted(np.unique(_hit_genes)),
        key='iso_gene_select',
    )
    _mask2 = np.char.upper(_hit_genes) == _chosen_gene.upper()
    _hit_ids   = _hit_ids[_mask2]
    _hit_sm    = _hit_sm[_mask2]
    _hit_types = _hit_types[_mask2]
    _hit_genes = _hit_genes[_mask2]
    _gene_name = _chosen_gene

_n_iso = len(_hit_ids)

# ── Gene landing card ─────────────────────────────────────────────────────────
# Show a compact gene-level summary before isoform selection.
# Shown always; helps orient the user before drilling into one isoform.
_from_gene_page = st.session_state.get('gene_from_gene_page', False)

_gene_max_per = _hit_sm.max(axis=1) if _n_iso > 0 else np.array([])
_n_high       = int((_gene_max_per >= thr).sum()) if len(_gene_max_per) else 0

# Module info
_gene_mod_label = ''
_gene_mod_id    = None
_mod_df_iso     = st.session_state.get('_gene_mod_df_cache')
if _mod_df_iso is None:
    _mod_path = Path(__file__).parents[3] / 'reports' / 'brain_isoform_modules.tsv'
    if _mod_path.exists():
        try:
            _mod_df_iso = pd.read_csv(_mod_path, sep='\t')
            st.session_state['_gene_mod_df_cache'] = _mod_df_iso
        except Exception:
            pass
if _mod_df_iso is not None:
    _gm = _mod_df_iso[_mod_df_iso['gene'].str.upper() == _gene_name.upper()]
    if len(_gm):
        _gene_mod_id = int(_gm['primary_module'].mode().iloc[0])
        _gene_mod_label = _gm[_gm['primary_module'] == _gene_mod_id]['module_label'].iloc[0]

# Scenario counts
_cls_g = classified[classified['isoform_id'].isin(_hit_ids)] if len(classified) else pd.DataFrame()
_sc_map = {1: '🔴 S1', 2: '🟠 S2', 3: '🟢 S3', 4: '⬜ S4'}
_sc_summary = ' · '.join(
    f"{_sc_map.get(int(s), f'S{s}')}: {int(c)}"
    for s, c in _cls_g['scenario'].value_counts().sort_index().items()
) if len(_cls_g) else '분류 데이터 없음'

with st.container():
    st.markdown(f"""
<div style='background:linear-gradient(90deg,#1e1b4b,#312e81);border-radius:10px;
padding:14px 20px 10px;margin-bottom:14px'>
<span style='color:#a5b4fc;font-size:0.75rem'>GENE SUMMARY</span>
<h3 style='color:white;margin:2px 0;font-size:1.3rem'>{_gene_name.upper()}</h3>
<span style='color:#c7d2fe;font-size:0.85rem'>
{_n_iso}개 아이소폼 · 고신뢰 {_n_high}개
{"· M" + str(_gene_mod_id) + " " + _gene_mod_label[:30] if _gene_mod_id else ""}
</span><br>
<span style='color:#e0e7ff;font-size:0.8rem'>Scenario: {_sc_summary}</span>
</div>
""", unsafe_allow_html=True)

    _lc1, _lc2 = st.columns([1, 1])
    with _lc1:
        if st.button("🧬 유전자 분석 페이지로", key='iso_back_to_gene',
                     use_container_width=True):
            st.session_state['gene_from_gene_page'] = False
            st.switch_page("pages/06_gene.py")
    with _lc2:
        _in_bk = _gene_name.upper() in [g.upper() for g in basket_gene_ids()]
        if _in_bk:
            st.success(f"✅ 바스켓에 있음")
        else:
            if st.button("➕ 유전자 바스켓 추가", key='iso_lc_add_basket',
                         use_container_width=True):
                add_to_gene_basket(_gene_name, source_page='isoform')
                st.toast(f"✅ {_gene_name} 바스켓 추가")
                st.rerun()

st.divider()

# ── Isoform selector (if multiple) ────────────────────────────────────────────
_deepdive_iso = str(st.session_state.get('deepdive_iso_id', ''))
if _deepdive_iso and _deepdive_iso in _hit_ids:
    _selected_iso_idx = int(np.where(_hit_ids == _deepdive_iso)[0][0])
else:
    _selected_iso_idx = 0
    st.session_state['deepdive_iso_id'] = ''

if _n_iso > 1:
    _sel_iso = st.selectbox(
        "아이소폼 선택 (상세 분석 대상)",
        options=_hit_ids.tolist(),
        index=_selected_iso_idx,
        key='iso_selector',
    )
    _iso_idx = int(np.where(_hit_ids == _sel_iso)[0][0])
else:
    _sel_iso = _hit_ids[0]
    _iso_idx = 0

_iso_scores = _hit_sm[_iso_idx]
_iso_type   = str(_hit_types[_iso_idx])

# ══════════════════════════════════════════════════════════════════════════════
# ISOFORM HEADER CARD
# ══════════════════════════════════════════════════════════════════════════════
_max_score  = float(_iso_scores.max())
_top_go_idx = int(_iso_scores.argmax())
_top_go_id  = go_terms[_top_go_idx] if go_terms else '—'
_top_go_nm  = gnames.get(_top_go_id, _top_go_id)[:40]
_n_high     = int((_iso_scores >= thr).sum())

# Scenario badge
_scen_badge = ''
if classified is not None and len(classified):
    _cls_row = classified[classified['isoform_id'] == _sel_iso]
    if len(_cls_row):
        _scen = int(_cls_row.iloc[0].get('scenario', 4))
        _scen_label = str(_cls_row.iloc[0].get('scenario_label', f'Scenario {_scen}'))
        _scen_color = str(_cls_row.iloc[0].get('scenario_color', '#94a3b8'))
        _scen_badge = (
            f"<span style='background:{_scen_color};color:white;border-radius:4px;"
            f"padding:2px 8px;font-size:0.78rem;font-weight:600'>{_scen_label}</span>"
        )

_type_badge = (
    f"<span style='background:#7c3aed;color:white;border-radius:4px;"
    f"padding:2px 7px;font-size:0.78rem'>{_iso_type}</span>"
    if _iso_type and _iso_type not in ('', 'nan') else ''
)

st.markdown(
    f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;"
    f"padding:18px 24px;margin:12px 0'>"
    f"<span style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em'>"
    f"아이소폼 분석 리포트</span><br>"
    f"<span style='font-size:1.4rem;font-weight:700;color:#0f172a'>{_sel_iso}</span>"
    f"&nbsp;&nbsp;{_type_badge}&nbsp;{_scen_badge}<br>"
    f"<span style='font-size:0.85rem;color:#475569'>유전자: <b>{_gene_name}</b> · "
    f"아이소폼 {_iso_idx+1}/{_n_iso}</span></div>",
    unsafe_allow_html=True,
)

_hm1, _hm2, _hm3, _hm4 = st.columns(4)
_hm1.metric("Max score",  f"{_max_score:.3f}")
_hm2.metric("Top GO",     _top_go_nm[:22])
_hm3.metric(f"고신뢰 GO (≥{thr})", _n_high)
_hm4.metric("유전자 아이소폼", f"{_n_iso}개")

render_analysis_context(cfg)

# ── Basket add ────────────────────────────────────────────────────────────────
_ba1, _ba2, _ba3 = st.columns([2, 2, 6])
if _ba1.button("➕ 유전자 바스켓 추가", key='iso_add_gene_basket'):
    added = add_to_gene_basket(_gene_name, source_page='isoform')
    st.toast(f"✅ {_gene_name} 바스켓 추가" if added else f"⚠️ {_gene_name} 이미 바스켓에 있음")
if _ba2.button("🎯 타겟 탐색으로", key='iso_goto_hub'):
    st.switch_page("pages/05_target_hub.py")

# ══════════════════════════════════════════════════════════════════════════════
# TABS: GO Profile | GAIN/LOSS | Within-Gene | Module | DTU
# ══════════════════════════════════════════════════════════════════════════════
_t_go, _t_gl, _t_wg, _t_mod = st.tabs([
    "📊 GO 프로파일",
    "🔺 GAIN / LOSS",
    "🧬 Within-Gene 비교",
    "🧩 모듈 배정",
])

# ── Tab 1: GO Profile ─────────────────────────────────────────────────────────
with _t_go:
    if len(go_terms) == 0:
        st.caption("GO term 정보 없음")
    else:
        _go_df = pd.DataFrame({
            'GO_ID': go_terms,
            'GO': [gnames.get(g, g)[:40] for g in go_terms],
            'Score': _iso_scores.tolist(),
        }).sort_values('Score', ascending=False).reset_index(drop=True)

        _top_n = 15
        _plot_df = _go_df.head(_top_n)
        _fig_go = px.bar(
            _plot_df, x='GO', y='Score',
            color='Score', color_continuous_scale='RdYlGn', range_color=[0, 1],
            title=f"{_sel_iso} — GO 기능 프로파일  (상위 {_top_n}/{len(go_terms)}개)",
            height=320,
        )
        _fig_go.add_hline(y=thr, line_dash='dash', line_color='grey',
                           annotation_text=f"threshold ({thr})")
        _fig_go.update_layout(
            xaxis_tickangle=-40, xaxis_tickfont_size=9,
            coloraxis_showscale=False,
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(t=50, b=100, l=10, r=10),
        )
        st.plotly_chart(_fig_go, use_container_width=True)

        # High-confidence table
        _high_df = _go_df[_go_df['Score'] >= thr]
        if len(_high_df):
            st.markdown(f"**고신뢰 GO term ({len(_high_df)}개):**")
            st.dataframe(_high_df[['GO', 'Score', 'GO_ID']].reset_index(drop=True),
                         use_container_width=True, hide_index=True, height=180)

        # Full profile expander
        with st.expander(f"📋 전체 {len(_go_df)}개 GO term 보기"):
            st.dataframe(_go_df, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 GO 프로파일 CSV", key='dl_go_csv',
                data=_go_df.to_csv(index=False).encode(),
                file_name=f"{_sel_iso}_go_profile.csv", mime="text/csv",
            )

# ── Tab 2: GAIN / LOSS vs gene median ────────────────────────────────────────
with _t_gl:
    if _n_iso < 2:
        st.info("같은 유전자의 다른 아이소폼이 없어 GAIN/LOSS 비교를 할 수 없습니다.")
    elif len(go_terms) == 0:
        st.caption("GO term 정보 없음")
    else:
        _gene_median = np.median(_hit_sm, axis=0)
        _delta       = _iso_scores - _gene_median
        _delta_df = pd.DataFrame({
            'GO': [gnames.get(g, g)[:35] for g in go_terms],
            'Δ score': _delta.tolist(),
            'This isoform': _iso_scores.tolist(),
            'Gene median': _gene_median.tolist(),
        }).sort_values('Δ score', key=abs, ascending=False).reset_index(drop=True)

        _top20 = _delta_df.head(20)
        _colors = ['#ef4444' if d > 0 else '#3b82f6' for d in _top20['Δ score']]
        _fig_gl = go.Figure(go.Bar(
            x=_top20['Δ score'], y=_top20['GO'],
            orientation='h',
            marker_color=_colors,
            hovertemplate='%{y}<br>Δ=%{x:.3f}<extra></extra>',
        ))
        _fig_gl.update_layout(
            title=f"GAIN/LOSS vs 유전자 중앙값 (상위 20개 |Δ|)",
            xaxis=dict(title='Δ score (이 아이소폼 − 유전자 중앙값)', zeroline=True,
                       zerolinecolor='#94a3b8'),
            plot_bgcolor='white', paper_bgcolor='white',
            height=max(300, 22 * len(_top20) + 80),
            margin=dict(l=200, r=40, t=50, b=40),
        )
        st.plotly_chart(_fig_gl, use_container_width=True)

        _gain_n = int((_delta_df['Δ score'] > 0.05).sum())
        _loss_n = int((_delta_df['Δ score'] < -0.05).sum())
        _g1, _g2 = st.columns(2)
        _g1.metric("GAIN (Δ > 0.05)", f"{_gain_n}개 GO term")
        _g2.metric("LOSS (Δ < -0.05)", f"{_loss_n}개 GO term")

        with st.expander("📋 전체 Δ 테이블"):
            st.dataframe(_delta_df, use_container_width=True, hide_index=True)

# ── Tab 3: Within-Gene Comparison ─────────────────────────────────────────────
with _t_wg:
    if _n_iso < 2:
        st.info("같은 유전자의 다른 아이소폼이 없습니다.")
    elif len(go_terms) == 0:
        st.caption("GO term 정보 없음")
    else:
        try:
            from prism_app.visualization.scatter import build_within_gene_chart
            _wg_fig, _wg_div = build_within_gene_chart(
                gene_name=_gene_name,
                isoform_ids=_hit_ids,
                score_matrix=_hit_sm,
                go_terms=go_terms,
                go_names=gnames,
                gene_ids=_hit_genes,
                chart_type='bar',
                title=f"{_gene_name}: 아이소폼별 GO 점수 비교",
            )
            st.plotly_chart(_wg_fig, use_container_width=True)

            if _wg_div:
                st.markdown(
                    f"**최대 기능 분기:** `{_wg_div['go_term']}` 에서 "
                    f"`{_wg_div['iso_high']}` ({_wg_div['score_high']:.3f}) vs "
                    f"`{_wg_div['iso_low']}` ({_wg_div['score_low']:.3f}), "
                    f"Δ = {_wg_div['max_delta']:.3f}"
                )

            # Heatmap alternative
            with st.expander("🔥 히트맵으로 보기"):
                _hm_fig, _ = build_within_gene_chart(
                    gene_name=_gene_name,
                    isoform_ids=_hit_ids,
                    score_matrix=_hit_sm,
                    go_terms=go_terms,
                    go_names=gnames,
                    gene_ids=_hit_genes,
                    chart_type='heatmap',
                )
                st.plotly_chart(_hm_fig, use_container_width=True)
        except Exception as _e:
            st.error(f"Within-gene 차트 오류: {_e}")

# ── Tab 4: Module Assignment ───────────────────────────────────────────────────
with _t_mod:
    _go_mod_map = _get_go_mod_map()
    if not _go_mod_map:
        st.info("모듈 배정 정보가 없습니다. Brain — Full (672 GO) 데이터셋 또는 모듈 생성 후 사용 가능합니다.")
    elif len(go_terms) == 0:
        st.caption("GO term 정보 없음")
    else:
        _mod_scores: dict = {}
        for _gi, _gs in enumerate(_iso_scores):
            _gid = go_terms[_gi]
            _mid = _go_mod_map.get(_gid)
            if _mid is not None:
                if _mid not in _mod_scores:
                    _mod_scores[_mid] = []
                _mod_scores[_mid].append(float(_gs))

        if _mod_scores:
            _mod_rows = [
                {'Module': f"M{m}", 'GO terms': len(sc), 'Mean score': round(np.mean(sc), 3),
                 'Max score': round(max(sc), 3)}
                for m, sc in sorted(_mod_scores.items(), key=lambda x: -np.mean(x[1]))
            ]
            _mod_df = pd.DataFrame(_mod_rows)
            st.dataframe(
                _mod_df.style.background_gradient(subset=['Mean score', 'Max score'],
                                                   cmap='RdYlGn', vmin=0, vmax=1),
                use_container_width=True, hide_index=True,
            )

            # Top modules
            _top_mods = [r['Module'] for r in _mod_rows if r['Max score'] >= thr]
            if _top_mods:
                st.markdown("**주요 모듈:** " + " · ".join(f"`{m}`" for m in _top_mods[:8]))

            # Mini bar chart
            _fig_mod_bar = px.bar(
                _mod_df.head(15), x='Module', y='Mean score',
                color='Mean score', color_continuous_scale='RdYlGn', range_color=[0, 1],
                title="모듈별 평균 PRISM 점수 (상위 15개)",
                height=300,
            )
            _fig_mod_bar.add_hline(y=thr, line_dash='dash', line_color='grey')
            _fig_mod_bar.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                coloraxis_showscale=False,
                margin=dict(t=40, b=40, l=10, r=10),
            )
            st.plotly_chart(_fig_mod_bar, use_container_width=True)
        else:
            st.caption("로드된 GO term에 대한 모듈 배정 정보가 없습니다.")

# ── DTU context (if available) ────────────────────────────────────────────────
if dtu_df is not None:
    with st.expander("🔄 DTU 연동 정보"):
        _id_cols = [c for c in ['isoform_id', 'transcript_id', 'isoform', 'id'] if c in dtu_df.columns]
        if _id_cols:
            _dtu_match = dtu_df[dtu_df[_id_cols[0]].astype(str).str.upper() == _sel_iso.upper()]
            if len(_dtu_match):
                st.dataframe(_dtu_match, use_container_width=True, hide_index=True)
            else:
                _gene_dtu = dtu_df[dtu_df[_id_cols[0]].astype(str).str.upper().isin(
                    [i.upper() for i in _hit_ids])]
                if len(_gene_dtu):
                    st.caption(f"이 아이소폼에 대한 직접 DTU 기록은 없지만 "
                               f"같은 유전자의 {len(_gene_dtu)}개 아이소폼에 DTU 기록이 있습니다:")
                    st.dataframe(_gene_dtu.head(10), use_container_width=True, hide_index=True)
                else:
                    st.caption("이 유전자에 대한 DTU 기록이 없습니다.")

# ── Splice Diversity + Cell-type Concordance ─────────────────────────────────
_iso_diu_all = _load_iso_diu_data()
_iso_sd_map  = _load_iso_splice_div()

_sd_c1, _sd_c2 = st.columns([1, 2])
with _sd_c1:
    st.markdown("##### Splice Diversity")
    _sd_entry = _iso_sd_map.get(_gene_name.upper())
    if _sd_entry:
        _sd_dist = _sd_entry['dist']
        _sd_color = ('#16a34a' if _sd_dist > 1.0
                     else '#d97706' if _sd_dist > 0.1
                     else '#6b7280')
        _sd_label = ('엑손 변이 강함' if _sd_dist > 1.0
                     else '부분 엑손 변이' if _sd_dist > 0.1
                     else 'UTR/미세 변이')
        st.markdown(
            f"<span style='font-size:1.5rem;font-weight:700;color:{_sd_color}'>{_sd_dist:.3f}</span>"
            f"<span style='font-size:0.8rem;color:#6b7280;margin-left:6px'>{_sd_label}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"유전자 내 최대 pairwise splice_delta L2 거리 ({_gene_name})")
    else:
        st.caption("splice_delta: 근육 모델 전용 (brain 아이소폼 미지원)")

with _sd_c2:
    st.markdown("##### 세포유형별 ΔIF")
    if _iso_diu_all is not None:
        _CELL_ORDER_ISO = [
            'Excitatory neuron', 'Inhibitory neuron', 'Astrocyte', 'Microglia',
            'Oligodendrocyte', 'OPC', 'Vascular cell', 'Lymphocyte',
        ]
        _iso_rows = _iso_diu_all[
            _iso_diu_all['transcript_name'].str.upper() == _sel_iso.upper()
        ].copy()
        if _iso_rows.empty:
            st.caption(f"{_sel_iso}: DIU 데이터 없음 (transcript_name 미매칭)")
        else:
            _ct_vals = {
                row['cell_type']: row['delta_usage']
                for _, row in _iso_rows.iterrows()
            }
            _ct_df = pd.DataFrame([
                {
                    '세포유형': ct,
                    'ΔIF': _ct_vals.get(ct, 0),
                    '유의': '●' if (
                        _iso_rows[_iso_rows['cell_type'] == ct]['chi_significant'].any()
                    ) else '○',
                }
                for ct in _CELL_ORDER_ISO
            ])
            _ct_df['방향'] = _ct_df['ΔIF'].apply(
                lambda v: 'AD↑' if v > 0.05 else ('CT↑' if v < -0.05 else '—'))
            _n_sig_iso = int((_iso_rows['chi_significant'] == True).sum())
            _ct_df_show = _ct_df[_ct_df['ΔIF'] != 0]
            if not _ct_df_show.empty:
                _fig_iso_bar = px.bar(
                    _ct_df, x='세포유형', y='ΔIF',
                    color='ΔIF',
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0,
                    range_color=[-0.5, 0.5],
                    height=220,
                    title=f"{_sel_iso} — 세포유형별 ΔIF (유의 {_n_sig_iso}건)",
                )
                _fig_iso_bar.update_layout(
                    plot_bgcolor='white',
                    margin=dict(t=30, b=40, l=10, r=10),
                    showlegend=False,
                    xaxis_tickangle=-25,
                    coloraxis_showscale=False,
                )
                _fig_iso_bar.add_hline(y=0, line_dash='solid', line_color='#94a3b8', line_width=1)
                st.plotly_chart(_fig_iso_bar, use_container_width=True, key='iso_diu_bar')
                st.caption("빨강 = AD-enriched · 파랑 = CT-enriched · ● = chi_significant")
            else:
                st.caption(f"{_sel_iso}: 세포유형 ΔIF = 0 (유의 이벤트 없음)")
    else:
        st.caption("세포유형별 DIU 데이터 없음 (내부 분석 환경 전용)")

st.divider()

# ── Case report download ───────────────────────────────────────────────────────
with st.expander("📄 케이스 리포트 다운로드"):
    _cr_lines = [
        f"# PRISM 아이소폼 분석 리포트\n",
        f"## {_sel_iso}  ({_gene_name})\n",
        f"- **Type**: {_iso_type}",
        f"- **Max score**: {_max_score:.3f}",
        f"- **Top GO**: {_top_go_nm} ({_top_go_id})",
        f"- **고신뢰 GO term**: {_n_high}개 (threshold={thr})\n",
        f"## GO 프로파일 (상위 15개)\n",
    ]
    if len(go_terms):
        _go_df_tmp = pd.DataFrame({
            'GO': [gnames.get(g, g)[:40] for g in go_terms],
            'Score': _iso_scores.tolist(),
        }).sort_values('Score', ascending=False).head(15)
        for _, _r in _go_df_tmp.iterrows():
            _cr_lines.append(f"- {_r['GO']}: {_r['Score']:.3f}")
    _cr_lines.append(context_for_report(cfg))
    _cr_md = '\n'.join(_cr_lines)
    st.download_button(
        "📥 Markdown 리포트 다운로드", key='dl_case_report',
        data=_cr_md.encode(),
        file_name=f"{_sel_iso}_report.md", mime="text/markdown",
    )
