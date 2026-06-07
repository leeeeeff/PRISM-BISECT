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


@st.cache_data(show_spinner=False)
def _load_gene_splice_diversity() -> dict:
    """Return {GENE_UPPER: {'dist': float, 'frac': float}} from splicing_delta_v2."""
    _SD   = Path(__file__).parents[3] / 'hMuscle/results_isoform/features/splicing/splicing_delta_v2.npy'
    _GENE = Path(__file__).parents[3] / 'hMuscle/model/my_gene_list_fixed.npy'
    _SYM  = (Path(__file__).parents[3]
             / 'hMuscle/data/raw_data/data/id_lists/ensembl_to_symbol.txt')
    if not _SD.exists() or not _GENE.exists():
        return {}
    _sd    = np.load(_SD).astype(np.float32)
    _graw  = np.load(_GENE, allow_pickle=True)
    _genes = [x.decode() if isinstance(x, bytes) else str(x) for x in _graw]
    _smap: dict = {}
    if _SYM.exists():
        with open(_SYM) as _f:
            next(_f)
            for _ln in _f:
                _p = _ln.strip().split()
                if len(_p) >= 5:
                    _smap[_p[0]] = _p[4]
    _syms = [_smap.get(g.split('.')[0], g.split('.')[0]) for g in _genes]
    _g2i: dict = {}
    for _i, _g in enumerate(_syms):
        _g2i.setdefault(_g, []).append(_i)
    _out: dict = {}
    for _g, _idxs in _g2i.items():
        if len(_idxs) < 2:
            continue
        _d   = _sd[_idxs]
        _mx  = 0.0
        for _a in range(len(_idxs)):
            for _b in range(_a + 1, len(_idxs)):
                _dist = float(np.linalg.norm(_d[_a] - _d[_b]))
                if _dist > _mx:
                    _mx = _dist
        _nz  = _d[_d != 0]
        _fr  = float(((_nz > -1) & (_nz < 1)).sum() / max(len(_nz), 1)) if len(_nz) else 0.0
        _out[_g.upper()] = {'dist': round(_mx, 3), 'frac': round(_fr, 3)}
    return _out

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

_hdr_col, _bisect_btn_col = st.columns([5, 1])
with _hdr_col:
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
with _bisect_btn_col:
    if _bisect_ok:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("🧫 BISECT\nCases →", key='gene_goto_bisect', use_container_width=True,
                     help="BISECT Cases 페이지에서 이 유전자의 증거 패키지 보기"):
            st.session_state['bisect_filter_gene'] = _gene_name
            st.switch_page("pages/07_bisect.py")

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
# SECTION 1.5: Functional Spread (max_delta)
# 유전자 내 아이소폼 간 GO score 최대 차 — PRISM isoform-level 기여 수치화
# cosine similarity 기반 FDI와 달리 magnitude 차이를 직접 포착
# ══════════════════════════════════════════════════════════════════════════════
if _n_iso >= 2 and go_terms:
    _per_go_delta = _hit_sm.max(axis=0) - _hit_sm.min(axis=0)  # (n_go,)
    _max_delta    = float(_per_go_delta.max())
    _argmax_go    = int(np.argmax(_per_go_delta))
    _max_go_name  = (gnames.get(go_terms[_argmax_go], go_terms[_argmax_go])[:42]
                     if _argmax_go < len(go_terms) else '—')

    if _max_delta >= 0.3:
        _fd_color, _fd_label = '#16a34a', '기능 분기 유의 ↑'
    elif _max_delta >= 0.1:
        _fd_color, _fd_label = '#d97706', '기능 분기 중간'
    else:
        _fd_color, _fd_label = '#94a3b8', '기능 분기 낮음'

    _fd1, _fd2 = st.columns([1, 2])
    with _fd1:
        st.markdown(
            f"<div style='border:1px solid {_fd_color};border-radius:8px;"
            f"padding:12px 16px;background:{_fd_color}18'>"
            f"<div style='color:{_fd_color};font-size:0.72rem;font-weight:600;"
            f"letter-spacing:0.05em'>FUNCTIONAL SPREAD</div>"
            f"<div style='font-size:2rem;font-weight:700;color:{_fd_color};line-height:1.1'>"
            f"{_max_delta:.3f}</div>"
            f"<div style='font-size:0.8rem;color:#64748b;margin-top:2px'>{_fd_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with _fd2:
        st.markdown(
            f"<div style='padding:10px 0 4px'>"
            f"<div style='font-size:0.78rem;color:#94a3b8'>최대 분기 GO term</div>"
            f"<div style='font-size:0.97rem;font-weight:600;color:#1e293b;margin:2px 0'>"
            f"{_max_go_name}</div>"
            f"<div style='font-size:0.75rem;color:#94a3b8'>"
            f"max(score_max − score_min) across {len(go_terms)} GO terms · "
            f"{_n_iso}개 아이소폼 비교"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1.6: Splice Diversity — splice_delta 기반 구조 다양성
# ══════════════════════════════════════════════════════════════════════════════
_sd_lookup = _load_gene_splice_diversity()
_sd_entry  = _sd_lookup.get(_gene_name.upper()) if _sd_lookup else None
if _sd_entry and _n_iso >= 2:
    _sd_dist = _sd_entry['dist']
    _sd_frac = _sd_entry['frac']
    if _sd_dist > 1.0:
        _sd_color, _sd_label = '#16a34a', '엑손 변이 강함 ↑'
    elif _sd_dist > 0.1:
        _sd_color, _sd_label = '#d97706', '부분 엑손 변이'
    else:
        _sd_color, _sd_label = '#94a3b8', '미세 / UTR 변이'

    _spc1, _spc2, _spc3 = st.columns(3)
    with _spc1:
        st.markdown(
            f"<div style='border:1px solid {_sd_color};border-radius:8px;"
            f"padding:10px 14px;background:{_sd_color}12'>"
            f"<div style='color:{_sd_color};font-size:0.72rem;font-weight:600;"
            f"letter-spacing:0.05em'>SPLICE DIVERSITY</div>"
            f"<div style='font-size:1.9rem;font-weight:700;color:{_sd_color};line-height:1.1'>"
            f"{_sd_dist:.3f}</div>"
            f"<div style='font-size:0.79rem;color:#64748b;margin-top:2px'>{_sd_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with _spc2:
        st.markdown(
            f"<div style='padding:10px 0 4px'>"
            f"<div style='font-size:0.78rem;color:#94a3b8'>A5SS / A3SS 비율</div>"
            f"<div style='font-size:1.5rem;font-weight:700;color:#1e293b;margin:2px 0'>"
            f"{_sd_frac:.1%}</div>"
            f"<div style='font-size:0.74rem;color:#94a3b8'>"
            f"부분 엑손 경계 이동 비율 (0%=clean SE, 100%=A5SS/A3SS 지배)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with _spc3:
        st.markdown(
            f"<div style='padding:10px 0 4px'>"
            f"<div style='font-size:0.78rem;color:#94a3b8'>해석</div>"
            f"<div style='font-size:0.84rem;color:#334155;margin:4px 0;line-height:1.4'>"
            + (
                "ESM-2 mean-pooling이 이 유전자의 아이소폼을 구분하지 못할 가능성 있음 "
                "(DIFF_SPLICE 후보)" if _sd_dist > 0.5
                else "아이소폼 간 splice 차이가 작음 — ESM-2가 충분히 구분 가능"
            )
            + f"</div>"
            f"<div style='font-size:0.72rem;color:#94a3b8'>"
            f"max pairwise splice_delta L2 · {_n_iso}개 아이소폼</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
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

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Cell-type Concordance (8 cell types × AD/CT DTU)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _load_diu_all_celltypes() -> 'pd.DataFrame | None':
    """Load 8 cell-type DIU CSVs and concatenate with cell_type label."""
    _diu_dir = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/06_DIU')
    _cell_types = [
        'Excitatory_neuron', 'Inhibitory_neuron', 'Astrocyte', 'Microglia',
        'Oligodendrocyte', 'OPC', 'Vascular_cell', 'Lymphocyte',
    ]
    _frames = []
    for _ct in _cell_types:
        _fp = _diu_dir / f'DIU_by_condition_{_ct}.csv'
        if _fp.exists():
            try:
                _df = pd.read_csv(_fp)
                _df['cell_type'] = _ct.replace('_', ' ')
                _frames.append(_df)
            except Exception:
                pass
    return pd.concat(_frames, ignore_index=True) if _frames else None

_diu_all = _load_diu_all_celltypes()

with st.expander("🔬 세포유형별 DTU 일관성 (Cell-type Concordance)", expanded=True):
    if _diu_all is None:
        st.info("세포유형별 DIU 데이터에 접근할 수 없습니다 (내부 분석 환경 전용).")
    else:
        _gene_diu = _diu_all[_diu_all['gene_name'].str.upper() == _gene_name.upper()].copy()
        if _gene_diu.empty:
            st.caption(f"{_gene_name}의 세포유형별 DTU 이벤트 없음")
        else:
            # Pivot: rows = isoform, cols = cell_type, values = delta_usage
            _sig_mask = _gene_diu['chi_significant'] == True
            _pivot_du = (
                _gene_diu[_sig_mask]
                .pivot_table(index='transcript_name', columns='cell_type',
                             values='delta_usage', aggfunc='first')
                .fillna(0)
            )
            _cell_order = [c for c in
                ['Excitatory neuron', 'Inhibitory neuron', 'Astrocyte', 'Microglia',
                 'Oligodendrocyte', 'OPC', 'Vascular cell', 'Lymphocyte']
                if c in _pivot_du.columns]
            _pivot_du = _pivot_du[[c for c in _cell_order if c in _pivot_du.columns]]

            if not _pivot_du.empty:
                # Concordance score: fraction of significant cell types with same direction
                _concord = {}
                for _iso in _pivot_du.index:
                    _vals = _pivot_du.loc[_iso]
                    _nonzero = _vals[_vals != 0]
                    if len(_nonzero) >= 2:
                        _pos = int((_nonzero > 0).sum())
                        _neg = int((_nonzero < 0).sum())
                        _dom = max(_pos, _neg) / len(_nonzero)
                        _concord[_iso] = round(_dom, 2)
                    elif len(_nonzero) == 1:
                        _concord[_iso] = 1.0
                    else:
                        _concord[_iso] = 0.0

                # Summary text
                _dir_counts = _gene_diu[_sig_mask]['usage_direction'].value_counts()
                _n_ad  = int(_dir_counts.get('AD_enriched', 0))
                _n_ct  = int(_dir_counts.get('CT_enriched', 0))
                _n_sig_total = int(_sig_mask.sum())
                _n_celltypes = _gene_diu[_sig_mask]['cell_type'].nunique()
                st.caption(
                    f"**{_gene_name}**: {len(_pivot_du)}개 아이소폼 · "
                    f"유의 DTU {_n_sig_total}건 ({_n_celltypes}개 세포유형) · "
                    f"AD-enriched {_n_ad}건 · CT-enriched {_n_ct}건"
                )

                # Heatmap
                _fig_ct = px.imshow(
                    _pivot_du,
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0,
                    zmin=-0.5, zmax=0.5,
                    title=f"{_gene_name} — 세포유형별 ΔUTF (유의한 이벤트만)",
                    labels={'color': 'delta_usage'},
                    aspect='auto',
                    height=max(280, 45 * len(_pivot_du) + 80),
                )
                _fig_ct.update_layout(
                    xaxis_tickangle=-30,
                    coloraxis_colorbar=dict(title='ΔIF', len=0.6),
                )
                st.plotly_chart(_fig_ct, use_container_width=True, key='gene_ct_heatmap')

                # Concordance table
                _conc_df = pd.DataFrame([
                    {'아이소폼': k,
                     'Concordance': v,
                     'N 세포유형': int((_pivot_du.loc[k] != 0).sum())}
                    for k, v in _concord.items()
                ]).sort_values('Concordance', ascending=False).reset_index(drop=True)
                st.dataframe(
                    _conc_df.style.background_gradient(subset=['Concordance'],
                                                        cmap='Greens', vmin=0, vmax=1),
                    use_container_width=True, hide_index=True, height=200,
                )
                st.caption(
                    "Concordance = 유의한 세포유형 중 같은 방향(AD-enriched 또는 CT-enriched)의 비율. "
                    "1.0 = 모든 세포유형에서 일관된 방향."
                )
            else:
                st.caption("유의한 DTU 이벤트 없음 (chi_significant = True 조건 없음)")

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
