"""05_target_hub.py — Target Analysis landing: candidate discovery, basket, module map."""
import sys
import json
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

REPORTS = Path(__file__).parents[3] / 'reports'

from prism_app.app.components.basket import (
    add_to_gene_basket, basket_gene_ids, get_basket_genes, clear_basket,
)

# ── DTU canonical/alternative chart helper ────────────────────────────────────

def _build_dtu_card_chart(
    gene: str,
    hit_ids,
    hit_scores,
    hit_types,
    dtu_df,
    go_terms_list,
    go_names_dict,
    score_thr: float,
):
    """Canonical vs alternative isoform DTU bar chart for a gene card.

    두 가지 모드:
    - DTU 있음: delta_IF 막대 (조건 A→B 전환량, canonical 기준선 표시)
    - DTU 없음: PRISM max score 막대 (canonical=Known 최고점, fallback)

    Returns a Plotly Figure or None if data insufficient.
    """
    n_iso = len(hit_ids)
    if n_iso == 0:
        return None

    max_scores   = hit_scores.max(axis=1)   # (n_iso,)
    max_go_idxs  = hit_scores.argmax(axis=1) # (n_iso,) — index into go_terms_list
    types_arr    = np.asarray(hit_types, dtype=str)

    # ── Canonical isoform identification ──────────────────────────────────────
    known_mask = np.char.lower(types_arr) == 'known'
    if known_mask.any():
        canon_idx = int(np.argmax(np.where(known_mask, max_scores, -1.0)))
    else:
        canon_idx = int(np.argmax(max_scores))
    canon_id = str(hit_ids[canon_idx])

    # Short ID labels
    def _short(iid: str) -> str:
        parts = str(iid).split('.')
        return parts[0][-14:] if len(parts[0]) > 14 else parts[0]

    short_ids = [_short(str(i)) for i in hit_ids]

    # ── Per-isoform max GO term name mapping ──────────────────────────────────
    def _go_short(idx: int) -> str:
        """짧은 GO 이름 (16자 이하). 긴 접두어 제거."""
        if not go_terms_list:
            return ''
        gid  = go_terms_list[int(idx)]
        name = go_names_dict.get(gid, gid)
        for pfx in ('negative regulation of ', 'positive regulation of ',
                    'regulation of ', 'establishment of '):
            if name.lower().startswith(pfx):
                name = name[len(pfx):]
                break
        return name[:16]

    def _go_full(idx: int) -> str:
        """호버용 전체 GO 이름."""
        if not go_terms_list:
            return ''
        gid  = go_terms_list[int(idx)]
        name = go_names_dict.get(gid, gid)
        return f"{gid}: {name}"

    _id_to_go_short = {str(hit_ids[i]): _go_short(max_go_idxs[i]) for i in range(n_iso)}
    _id_to_go_full  = {str(hit_ids[i]): _go_full(max_go_idxs[i])  for i in range(n_iso)}

    # ── Mode A: DTU data available ────────────────────────────────────────────
    if dtu_df is not None:
        _id_col  = next(
            (c for c in ['isoform_id', 'transcript_id', 'featureID', 'id'] if c in dtu_df.columns),
            None,
        )
        _dif_col = next(
            (c for c in ['delta_IF', 'delta_if', 'dIF', 'deltaPSI', 'logFC'] if c in dtu_df.columns),
            None,
        )
        _pval_col = next(
            (c for c in ['padj', 'pvalue', 'FDR', 'adj.p.value', 'p_val_adj'] if c in dtu_df.columns),
            None,
        )
        _cond_col = next(
            (c for c in ['condition', 'comparison', 'contrast', 'group'] if c in dtu_df.columns),
            None,
        )
        # Direct CT/AD IF columns (output from satuRn, IsoformSwitchAnalyzeR, etc.)
        _ct_col = next(
            (c for c in ['ct_if', 'IF_CT', 'IF_ctrl', 'control_IF', 'meanCT', 'meanControl',
                          'isoform_fraction_ctrl', 'dIF_ctrl'] if c in dtu_df.columns),
            None,
        )
        _ad_col = next(
            (c for c in ['ad_if', 'IF_AD', 'IF_treat', 'disease_IF', 'meanAD', 'meanDisease',
                          'isoform_fraction_treat', 'dIF_treat'] if c in dtu_df.columns),
            None,
        )

        if _id_col is not None and _dif_col is not None:
            _ids_upper = {str(i).upper(): str(i) for i in hit_ids}
            _matched   = dtu_df[dtu_df[_id_col].astype(str).str.upper().isin(_ids_upper)]

            if len(_matched) >= 1:
                _rows = []
                for _orig_id in hit_ids:
                    _sub = _matched[_matched[_id_col].astype(str).str.upper() == str(_orig_id).upper()]
                    if len(_sub) == 0:
                        _rows.append({'isoform_id': str(_orig_id), 'delta_if': 0.0,
                                      'pval': 1.0, 'condition': '—', 'in_dtu': False,
                                      'ct_if': None, 'ad_if': None})
                    else:
                        _sub = _sub.copy()
                        _sub['abs_dif'] = _sub[_dif_col].abs()
                        _best = _sub.loc[_sub['abs_dif'].idxmax()]
                        _pv   = float(_best[_pval_col]) if _pval_col else 1.0
                        _cond = str(_best[_cond_col])   if _cond_col else '—'
                        _rows.append({
                            'isoform_id': str(_orig_id),
                            'delta_if':   float(_best[_dif_col]),
                            'pval':       _pv,
                            'condition':  _cond,
                            'in_dtu':     True,
                            'ct_if': float(_best[_ct_col]) if _ct_col else None,
                            'ad_if': float(_best[_ad_col]) if _ad_col else None,
                        })

                _df = pd.DataFrame(_rows)
                _df['short_id']    = [_short(i) for i in _df['isoform_id']]
                _df['is_canon']    = _df['isoform_id'] == canon_id
                _df['type']        = [str(t) for t in types_arr]
                _df['max_score']   = [float(s) for s in max_scores]
                _df['sig']         = _df['pval'] < 0.05
                _df['max_go_name'] = _df['isoform_id'].map(_id_to_go_short).fillna('')
                _df['max_go_full'] = _df['isoform_id'].map(_id_to_go_full).fillna('')

                # ── CT/AD fraction computation ────────────────────────────
                _has_direct = (
                    _ct_col is not None and _ad_col is not None
                    and _df['ct_if'].notna().any()
                )
                if _has_direct:
                    _df['ct_frac'] = _df['ct_if'].fillna(1.0 / n_iso)
                    _df['ad_frac'] = _df['ad_if'].fillna(
                        (_df['ct_frac'] + _df['delta_if']).clip(lower=0)
                    )
                    _est_note = ''
                else:
                    # Uniform CT baseline → renormalized AD
                    _ct_base = 1.0 / n_iso
                    _df['ct_frac'] = _ct_base
                    _df['ad_frac'] = (_ct_base + _df['delta_if']).clip(lower=0)
                    _sum_ad = _df['ad_frac'].sum()
                    if _sum_ad > 0:
                        _df['ad_frac'] /= _sum_ad
                    _est_note = ' *CT 균등 추정'

                # Sort: canonical first, then by delta_if desc
                _df = pd.concat([
                    _df[_df['is_canon']],
                    _df[~_df['is_canon']].sort_values('delta_if', ascending=False),
                ], ignore_index=True)

                fig = go.Figure()

                # CT bars
                fig.add_trace(go.Bar(
                    name='CT (Control)',
                    x=_df['short_id'],
                    y=_df['ct_frac'],
                    marker_color='#93c5fd',
                    marker_line=dict(
                        color=['#1e3a8a' if r['is_canon'] else '#60a5fa'
                               for _, r in _df.iterrows()],
                        width=[2.5 if r['is_canon'] else 0.5
                               for _, r in _df.iterrows()],
                    ),
                    hovertemplate=(
                        '<b>%{x}</b><br>CT 사용 비율: %{y:.3f}'
                        '<extra>CT</extra>'
                    ),
                ))

                # AD bars
                fig.add_trace(go.Bar(
                    name='AD (Disease)',
                    x=_df['short_id'],
                    y=_df['ad_frac'],
                    marker_color='#fca5a5',
                    marker_line=dict(
                        color=['#991b1b' if r['is_canon'] else '#f87171'
                               for _, r in _df.iterrows()],
                        width=[2.5 if r['is_canon'] else 0.5
                               for _, r in _df.iterrows()],
                    ),
                    text=['* ' if r['sig'] and r['in_dtu'] else ''
                          for _, r in _df.iterrows()],
                    textposition='outside',
                    textfont=dict(size=10, color='#991b1b'),
                    hovertemplate=(
                        '<b>%{x}</b><br>AD 사용 비율: %{y:.3f}<br>'
                        'ΔIF: %{customdata[0]:+.3f}  p=%{customdata[1]:.2e}'
                        '<extra>AD</extra>'
                    ),
                    customdata=_df[['delta_if', 'pval']].values,
                ))

                # PRISM score overlay (right y-axis) — markers + GO name label
                fig.add_trace(go.Scatter(
                    name='PRISM (max GO)',
                    x=_df['short_id'],
                    y=_df['max_score'],
                    mode='markers+text',
                    yaxis='y2',
                    marker=dict(
                        size=[15 if r['is_canon'] else 9 for _, r in _df.iterrows()],
                        color='#7c3aed',
                        symbol=['star' if r['is_canon'] else 'circle'
                                for _, r in _df.iterrows()],
                        opacity=0.9,
                    ),
                    text=_df['max_go_name'],
                    textposition='top center',
                    textfont=dict(size=6.5, color='#5b21b6'),
                    customdata=list(zip(
                        _df['max_go_full'],
                        _df['max_score'],
                    )),
                    hovertemplate=(
                        'PRISM %{customdata[1]:.3f}<br>'
                        '<b>%{customdata[0]}</b>'
                        '<extra>PRISM</extra>'
                    ),
                ))

                _cond_str = _df['condition'].iloc[-1] if _cond_col else ''
                _cond_sfx = f'  [{_cond_str}]' if _cond_str and _cond_str != '—' else ''
                _title = (
                    f"이소폼 사용 비율 CT vs AD · {gene.upper()}"
                    f"{_cond_sfx}{_est_note}"
                )

                fig.update_layout(
                    title=dict(text=_title, font_size=10, x=0),
                    height=310,
                    barmode='group',
                    bargap=0.22,
                    bargroupgap=0.04,
                    margin=dict(t=32, b=58, l=10, r=14),
                    plot_bgcolor='#f8fafc',
                    paper_bgcolor='white',
                    legend=dict(
                        orientation='h', y=-0.24, x=0,
                        font=dict(size=8), bgcolor='rgba(0,0,0,0)',
                        traceorder='normal',
                    ),
                    xaxis=dict(
                        title='', tickangle=-28,
                        tickfont=dict(size=8.5), showgrid=False,
                    ),
                    yaxis=dict(
                        title=dict(text='사용 비율 (IF)', font=dict(size=9)),
                        gridcolor='#e2e8f0',
                        range=[0, min(1.0, _df[['ct_frac', 'ad_frac']].max().max() * 1.25)],
                    ),
                    yaxis2=dict(
                        title=dict(text='PRISM', font=dict(size=9, color='#7c3aed')),
                        overlaying='y', side='right',
                        range=[0, 1.32],   # 텍스트 라벨 공간 확보
                        showgrid=False,
                        tickfont=dict(size=8, color='#7c3aed'),
                    ),
                )

                # Canonical annotation (★)
                _canon_short = _short(canon_id)
                if _canon_short in _df['short_id'].values:
                    _ct_val = float(_df.loc[_df['short_id'] == _canon_short, 'ct_frac'].iloc[0])
                    fig.add_annotation(
                        x=_canon_short, y=_ct_val,
                        text='★', showarrow=False,
                        font=dict(size=13, color='#1e3a8a'),
                        yshift=14,
                    )

                return fig

    # ── Mode B: No DTU — PRISM score comparison bar ───────────────────────────
    if n_iso < 2:
        return None

    _colors_fallback = []
    for _i, (_sid, _t) in enumerate(zip(short_ids, types_arr)):
        if _i == canon_idx:
            _colors_fallback.append('#3b82f6')
        elif str(_t).lower() in ('nic', 'nnic'):
            _colors_fallback.append('#f59e0b')
        else:
            _colors_fallback.append('#94a3b8')

    _order = np.argsort(max_scores)[::-1]

    # Per-isoform max GO label for hover
    _hover_b = [
        f"{short_ids[i]}<br>"
        f"PRISM {float(max_scores[i]):.3f}<br>"
        f"<b>{_id_to_go_full.get(str(hit_ids[i]), '')}</b>"
        for i in _order
    ]
    # Short GO name for bar text
    _bar_text_b = [_id_to_go_short.get(str(hit_ids[i]), '') for i in _order]

    fig = go.Figure(go.Bar(
        x=[short_ids[i] for i in _order],
        y=[float(max_scores[i]) for i in _order],
        marker_color=[_colors_fallback[i] for i in _order],
        marker_line=dict(
            color=['#1e3a8a' if i == canon_idx else 'rgba(0,0,0,0)' for i in _order],
            width=[2.5 if i == canon_idx else 0 for i in _order],
        ),
        text=_bar_text_b,
        textposition='outside',
        textfont=dict(size=7, color='#374151'),
        hovertext=_hover_b,
        hoverinfo='text',
    ))
    fig.add_hline(y=score_thr, line_dash='dash', line_color='#dc2626',
                  annotation_text=f'thr={score_thr}', annotation_position='top right',
                  annotation_font_size=9)

    _canon_short = short_ids[canon_idx]
    fig.add_annotation(
        x=_canon_short,
        y=float(max_scores[canon_idx]),
        text='★ Canonical',
        showarrow=True, arrowhead=2, arrowsize=0.8, arrowcolor='#3b82f6',
        font=dict(size=9, color='#3b82f6'), yshift=14,
    )

    fig.update_layout(
        title=dict(
            text=f"PRISM max GO score · {gene.upper()}  [DTU 없음]",
            font_size=10, x=0,
        ),
        height=300,
        margin=dict(t=32, b=50, l=10, r=10),
        plot_bgcolor='#f8fafc',
        paper_bgcolor='white',
        showlegend=False,
        xaxis=dict(title='', tickangle=-30, tickfont=dict(size=8.5), showgrid=False),
        yaxis=dict(
            title=dict(text='Max PRISM score', font=dict(size=9)),
            range=[0, 1.35], gridcolor='#e2e8f0',
        ),
        bargap=0.35,
    )
    fig.add_annotation(
        xref='paper', yref='paper', x=1.0, y=1.07,
        text=(
            "<span style='color:#3b82f6'>■</span> Canonical  "
            "<span style='color:#f59e0b'>■</span> Novel  "
            "<span style='color:#94a3b8'>■</span> Other"
        ),
        showarrow=False, font=dict(size=8.5), align='right', xanchor='right',
    )
    return fig


# ── Session data ──────────────────────────────────────────────────────────────
cfg            = st.session_state.get('cfg') or {}
sm             = cfg.get('score_matrix')
ids            = cfg.get('isoform_ids')
genes          = cfg.get('gene_ids')
go_terms       = cfg.get('go_terms', [])
gnames         = cfg.get('go_names', {})
thr            = cfg.get('score_threshold', 0.4)
isoform_types  = cfg.get('isoform_types')
dtu_df         = cfg.get('dtu_df')
tissue         = cfg.get('tissue', '')

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1e1b4b,#312e81,#1e3a5f);
border-radius:14px;padding:32px 48px 24px;margin-bottom:20px'>
<h1 style='color:white;margin:0;font-size:2rem'>🎯 Target Analysis</h1>
<p style='color:rgba(255,255,255,0.7);margin:8px 0 0;font-size:0.95rem'>
후보 아이소폼 동적 발굴 · 바스켓 비교 · 모듈 분포 필터링
</p></div>
""", unsafe_allow_html=True)

if sm is None:
    st.info(
        "**데이터가 로드되지 않았습니다.**\n\n"
        "사이드바에서 **Demo** (Brain — Full 672 GO 권장) 또는 "
        "**Upload** 모드를 선택하세요."
    )
    st.divider()
    _c1, _c2, _c3 = st.columns(3)
    for _col, (_icon, _title, _desc, _page) in zip(
        [_c1, _c2, _c3],
        [
            ("📊", "QC & Overview", "데이터 로드 후 아이소폼 분류 현황 확인", "pages/01_qc.py"),
            ("🗺️", "Module Landscape", "672 GO → 44 기능 모듈 지형도", "pages/02_landscape.py"),
            ("📋", "시나리오 & 분석", "4-시나리오 분류 및 BISECT 케이스", "pages/05_targets.py"),
        ],
    ):
        with _col:
            st.markdown(f"""<div style='background:#f8fafc;border-radius:10px;
            padding:20px;text-align:center'>
            <div style='font-size:2rem'>{_icon}</div>
            <b style='color:#1e293b'>{_title}</b><br>
            <span style='font-size:0.8rem;color:#64748b'>{_desc}</span></div>""",
            unsafe_allow_html=True)
            if st.button(f"▶ {_title}", key=f"hub_nav_{_title}", use_container_width=True):
                st.switch_page(_page)
    st.stop()

# ── Build classified_df ────────────────────────────────────────────────────────
classified = st.session_state.get('classified_df')
if classified is None:
    from prism_app.core.classifier import classify_isoforms
    with st.spinner("아이소폼 분류 중…"):
        classified = classify_isoforms(
            sm, ids, genes, go_terms,
            score_threshold=thr, dtu_df=dtu_df,
        )
    st.session_state['classified_df'] = classified

ids_arr   = np.asarray(ids, dtype=str)
gene_arr  = np.asarray(genes, dtype=str) if genes is not None else ids_arr
sm_arr    = np.asarray(sm, dtype=float)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Quick Card Panel
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<span style='font-size:1.1rem;font-weight:700;color:#1e293b'>🔍 핵심 케이스 Quick Card</span>"
    "<span style='font-size:0.82rem;color:#64748b;margin-left:10px'>"
    "4개 분석 축에서 자동 선정된 핵심 케이스 · ◀▶ 로 탐색하고 바로 분석을 시작하세요</span>",
    unsafe_allow_html=True,
)

def _add_to_basket(gene_name: str, tag_scenario: object = None):
    add_to_gene_basket(str(gene_name), source_page='target_hub', tag_scenario=tag_scenario)

# ── Enrich classified with isoform_type ──────────────────────────────────────
if isoform_types is not None and 'isoform_type' not in classified.columns:
    _type_arr = np.asarray(isoform_types, dtype=str)
    _type_map = {str(ids_arr[i]): _type_arr[i] for i in range(len(ids_arr))}
    classified = classified.copy()
    classified['isoform_type'] = classified['isoform_id'].map(_type_map).fillna('')
    st.session_state['classified_df'] = classified

# ── Build candidate DataFrames per axis ──────────────────────────────────────
_s1_cands = (classified[classified['scenario'] == 1]
             .sort_values('max_score', ascending=False).head(12).reset_index(drop=True))
_s3_cands = (classified[classified['scenario'] == 3]
             .sort_values('max_score', ascending=False).head(12).reset_index(drop=True))

if 'gene_id' in classified.columns:
    def _div_stats(g):
        if len(g) < 2:
            return None
        top_i = g['max_score'].idxmax()
        return pd.Series({
            'gene_id':        g['gene_id'].iloc[0],
            'isoform_id':     g.loc[top_i, 'isoform_id'],
            'n_isoforms':     len(g),
            'score_range':    round(float(g['max_score'].max() - g['max_score'].min()), 4),
            'max_score':      float(g.loc[top_i, 'max_score']),
            'max_go':         g.loc[top_i, 'max_go'],
            'scenario_color': g.loc[top_i, 'scenario_color'],
            'isoform_type':   g.loc[top_i, 'isoform_type'] if 'isoform_type' in g.columns else '',
        })
    _div_cands = (
        classified.groupby('gene_id', sort=False).apply(_div_stats)
        .dropna().sort_values('score_range', ascending=False)
        .head(12).reset_index(drop=True)
    )
else:
    _div_cands = pd.DataFrame()

_dtu_cands = pd.DataFrame()
if dtu_df is not None and 'gene_id' in classified.columns:
    from prism_app.core.classifier import _find_col as _fc
    try:
        _id_col  = _fc(dtu_df, ['isoform_id', 'transcript_id', 'isoform', 'id'])
        _dif_col = _fc(dtu_df, ['delta_if', 'deltaIF', 'dIF', 'delta'], required=False)
        if _dif_col:
            _dtu_abs = dtu_df.copy()
            _dtu_abs['abs_dif'] = _dtu_abs[_dif_col].abs()
            _merge_cols = ['isoform_id', 'gene_id', 'max_score', 'max_go', 'scenario_color']
            if 'isoform_type' in classified.columns:
                _merge_cols.append('isoform_type')
            _dtu_cands = (
                _dtu_abs.merge(
                    classified[_merge_cols],
                    left_on=_id_col, right_on='isoform_id', how='inner',
                )
                .sort_values(['abs_dif', 'max_score'], ascending=[False, False])
                .drop_duplicates('gene_id').head(12).reset_index(drop=True)
            )
    except Exception:
        pass

# ── Session state for carousel indices ───────────────────────────────────────
for _ik in ['_cidx_s1', '_cidx_s3', '_cidx_div', '_cidx_dtu']:
    if _ik not in st.session_state:
        st.session_state[_ik] = 0

# ── Gene Quick Report renderer (same format as 05_targets._render_gene_landing) ──
@st.cache_data(show_spinner=False)
def _load_module_umap_data():
    """Load precomputed UMAP coords + isoform-module mapping (brain_672 only)."""
    _demo = Path(__file__).parents[2] / 'data' / 'demo'
    _cp   = _demo / 'umap_coords.npy'
    _sp   = _demo / 'umap_sample_idx.npy'
    _mp   = Path(__file__).parents[3] / 'reports' / 'brain_isoform_modules.tsv'
    if not (_cp.exists() and _sp.exists() and _mp.exists()):
        return None, None, None
    return (np.load(_cp), np.load(_sp),
            pd.read_csv(_mp, sep='\t'))


def _build_module_umap_figure(gene: str, hit_ids, hit_scores):
    """Module UMAP: background colored by primary_module, gene isoforms highlighted.

    brain_672 only — uses precomputed 20K UMAP coords.
    Off-sample isoforms are approximated via cosine-NN averaging.
    Falls back to None (caller shows heatmap) if data unavailable.
    """
    if tissue not in ('brain_672', 'brain_expanded'):
        return None

    _coords, _samp_idx, _df_imod = _load_module_umap_data()
    if _coords is None:
        return None

    # Build id → module mappings
    _id2mod   = dict(zip(_df_imod['isoform_id'], _df_imod['primary_module']))
    _id2label = dict(zip(_df_imod['isoform_id'], _df_imod['module_label']))

    _samp_ids  = ids_arr[_samp_idx]
    _samp_mods = np.array([_id2mod.get(str(s), -1) for s in _samp_ids])

    # 44-module color palette
    _pal = (px.colors.qualitative.Alphabet
            + px.colors.qualitative.Dark24
            + px.colors.qualitative.Light24)
    _all_mods   = sorted(int(m) for m in np.unique(_samp_mods) if m >= 0)
    _mod_color  = {m: _pal[i % len(_pal)] for i, m in enumerate(_all_mods)}

    fig_umap = go.Figure()

    # ── Background: one trace per module ─────────────────────────────────
    for _m in _all_mods:
        _msk = _samp_mods == _m
        _first = int(np.where(_msk)[0][0])
        _mlbl  = str(_id2label.get(str(_samp_ids[_first]), f'M{_m}'))[:24]
        fig_umap.add_trace(go.Scatter(
            x=_coords[_msk, 0], y=_coords[_msk, 1],
            mode='markers',
            marker=dict(size=2.5, color=_mod_color[_m], opacity=0.22),
            name=f'M{_m}: {_mlbl}',
            showlegend=False,
            hovertemplate=f'M{_m}: {_mlbl}<extra></extra>',
        ))

    # ── Gene isoform positions ─────────────────────────────────────────
    _gene_idx = np.where(np.isin(ids_arr, hit_ids))[0]
    _samp_set = set(_samp_idx.tolist())
    _samp_pos = {int(v): k for k, v in enumerate(_samp_idx.tolist())}

    _in_samp  = [i for i in _gene_idx if i in _samp_set]
    _off_samp = [i for i in _gene_idx if i not in _samp_set]

    _gx, _gy, _giso, _gmods, _gapprox = [], [], [], [], []

    for _gi in _in_samp:
        _si = _samp_pos[_gi]
        _gx.append(float(_coords[_si, 0])); _gy.append(float(_coords[_si, 1]))
        _giso.append(str(ids_arr[_gi]))
        _gmods.append(int(_id2mod.get(str(ids_arr[_gi]), -1)))
        _gapprox.append(False)

    if _off_samp:
        _ss = sm_arr[_samp_idx].astype(np.float32)
        _su = _ss / (np.linalg.norm(_ss, axis=1, keepdims=True) + 1e-8)
        for _gi in _off_samp:
            _v  = sm_arr[_gi].astype(np.float32)
            _sims = _su @ (_v / (np.linalg.norm(_v) + 1e-8))
            _top5 = np.argsort(_sims)[-5:]
            _gx.append(float(_coords[_top5, 0].mean()))
            _gy.append(float(_coords[_top5, 1].mean()))
            _giso.append(str(ids_arr[_gi]))
            _gmods.append(int(_id2mod.get(str(ids_arr[_gi]), -1)))
            _gapprox.append(True)

    if not _gx:
        return None

    _max_s    = hit_scores.max(axis=1)
    _g_colors = [_mod_color.get(_m, '#2563eb') for _m in _gmods]
    _g_sizes  = [16 + 8 * float(_max_s[_i] if _i < len(_max_s) else 0)
                 for _i in range(len(_gx))]
    _g_sym    = ['diamond' if _a else 'circle' for _a in _gapprox]
    _g_short  = [_iso.split('.')[-1] if '.' in _iso else _iso[-8:] for _iso in _giso]
    _g_modlbl = [
        f"M{_m}: {_id2label.get(_iso, '')[:22]}" if _m >= 0 else '—'
        for _iso, _m in zip(_giso, _gmods)
    ]

    fig_umap.add_trace(go.Scatter(
        x=_gx, y=_gy,
        mode='markers+text',
        marker=dict(size=_g_sizes, color=_g_colors, opacity=0.95,
                    symbol=_g_sym, line=dict(width=2.5, color='white')),
        text=_g_short,
        textposition='top center',
        textfont=dict(size=9, color='#1e293b'),
        customdata=list(zip(_giso, _g_modlbl)),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'Module: %{customdata[1]}<extra></extra>'
        ),
        name=gene.upper(),
        showlegend=True,
    ))

    # ── Module label annotations for gene's modules ───────────────────
    _gene_uniq_mods = list(dict.fromkeys(_m for _m in _gmods if _m >= 0))
    for _m in _gene_uniq_mods[:5]:
        _mmsk = _samp_mods == _m
        if not _mmsk.any():
            continue
        _cx = float(_coords[_mmsk, 0].mean())
        _cy = float(_coords[_mmsk, 1].mean())
        _mlbl = str(_id2label.get(str(_samp_ids[np.where(_mmsk)[0][0]]), f'M{_m}'))[:22]
        fig_umap.add_annotation(
            x=_cx, y=_cy,
            text=f'<b>M{_m}</b>: {_mlbl}',
            showarrow=False,
            font=dict(size=8.5, color='#1e293b'),
            bgcolor='rgba(255,255,255,0.88)',
            bordercolor=_mod_color.get(_m, '#666'),
            borderwidth=1.5, borderpad=3,
        )

    fig_umap.update_layout(
        title=dict(text=f'Module UMAP — {gene.upper()}', font_size=12),
        height=340,
        margin=dict(t=36, b=10, l=10, r=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor='#f8fafc', paper_bgcolor='white',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.8)', font=dict(size=9)),
    )
    return fig_umap


def _render_gene_quick_report(gene: str, scenario_badge: str = '', badge_color: str = '#64748b', key_prefix: str = ''):
    gene_upper = gene.upper()
    if gene_arr is not None:
        _mask = np.char.upper(gene_arr) == gene_upper
    else:
        _mask = np.char.upper(ids_arr).startswith(gene_upper)
    if not _mask.any():
        _mask = np.array([gene_upper in str(g).upper() for g in (gene_arr if gene_arr is not None else ids_arr)])
    if not _mask.any():
        st.warning(f"'{gene}' 데이터에서 찾을 수 없습니다.")
        return

    _hit_ids    = ids_arr[_mask]
    _hit_scores = sm_arr[_mask]
    _hit_types  = (np.asarray(isoform_types, dtype=str)[_mask]
                   if isoform_types is not None else np.full(_mask.sum(), ''))
    _max_per_iso = _hit_scores.max(axis=1)

    # ── Header (identical to _render_gene_landing) ────────────────────────
    _badge_html = (
        f"<div style='position:absolute;top:12px;right:16px;background:{badge_color};"
        f"color:white;border-radius:20px;padding:3px 12px;font-size:0.73rem;"
        f"font-weight:700'>{scenario_badge}</div>"
        if scenario_badge else ''
    )
    st.markdown(f"""
<div style='background:linear-gradient(90deg,#0f2942,#1e3a5f);border-radius:12px;
padding:16px 24px 12px;margin-bottom:16px;position:relative'>
  {_badge_html}
  <span style='color:#93c5fd;font-size:0.8rem'>GENE QUICK REPORT</span>
  <h2 style='color:white;margin:4px 0 2px;font-size:1.6rem'>{gene.upper()}</h2>
  <span style='color:#bfdbfe;font-size:0.9rem'>{_mask.sum()} isoforms · {len(go_terms)} GO terms</span>
</div>
""", unsafe_allow_html=True)

    _lc_hm, _lc_table, _lc_met = st.columns([2, 2, 1])

    # ── Module UMAP (brain_672) / PRISM score heatmap fallback ─────────────
    with _lc_hm:
        _umap_fig = _build_module_umap_figure(gene, _hit_ids, _hit_scores)
        if _umap_fig is not None:
            st.plotly_chart(_umap_fig, use_container_width=True, key=f"{key_prefix}_umap")
        else:
            # Fallback: PRISM score heatmap (non-brain_672 / no module data)
            _top_idx = np.argsort(_hit_scores.max(axis=0))[-min(12, len(go_terms)):][::-1]
            _top_names = [gnames.get(go_terms[i], go_terms[i])[:20] for i in _top_idx]
            _fig_hm = px.imshow(
                _hit_scores[:, _top_idx],
                x=_top_names, y=[str(_i) for _i in _hit_ids],
                color_continuous_scale='Blues', aspect='auto',
                title="PRISM score heatmap", zmin=0, zmax=1,
            )
            _fig_hm.update_layout(height=340, margin=dict(t=36, b=10, l=10, r=10),
                                   coloraxis_showscale=False)
            _fig_hm.update_xaxes(tickangle=-40, tickfont_size=9)
            st.plotly_chart(_fig_hm, use_container_width=True, key=f"{key_prefix}_hm")

    # ── Isoform table ─────────────────────────────────────────────────────
    with _lc_table:
        _top_go_per = [go_terms[np.argmax(_hit_scores[i])] for i in range(len(_hit_ids))]
        _top_nm_per = [gnames.get(g, g)[:28] for g in _top_go_per]

        # Module lookup (session state → JSON file, same pattern as _render_gene_landing)
        _umods = st.session_state.get('user_modules') or st.session_state.get('brain672_modules')
        if _umods is None:
            _mj = Path(__file__).parents[3] / 'reports' / 'brain_go_modules_672.json'
            if not _mj.exists():
                _mj = Path(__file__).parents[2] / 'data' / 'demo' / 'brain_go_modules_672.json'
            if _mj.exists():
                try:
                    _umods = json.loads(_mj.read_text())
                    st.session_state['brain672_modules'] = _umods
                except Exception:
                    pass
        _mod_labels = []
        if _umods:
            _gm_map = _umods.get('go_module_map', {})
            for i in range(len(_hit_ids)):
                _tg  = go_terms[np.argmax(_hit_scores[i])]
                _mid = _gm_map.get(_tg)
                _mod_labels.append(f"M{_mid}" if _mid else "—")
        else:
            _mod_labels = ["—"] * len(_hit_ids)

        _df_land = pd.DataFrame({
            'Isoform':   _hit_ids,
            'Type':      _hit_types,
            'Max score': _max_per_iso.round(3),
            'Top GO':    _top_nm_per,
            'Module':    _mod_labels,
        }).sort_values('Max score', ascending=False)

        st.dataframe(
            _df_land.style.background_gradient(subset=['Max score'], cmap='Blues'),
            use_container_width=True, hide_index=True,
            height=min(320, 35 * len(_df_land) + 38),
        )

        _uniq_mods = list(dict.fromkeys(m for m in _mod_labels if m != "—"))
        if _uniq_mods:
            st.markdown("**Primary modules:** " + " · ".join(f"`{m}`" for m in _uniq_mods))

        with st.expander("📊 GO score heatmap (확장)"):
            _top2_idx  = np.argsort(_hit_scores.max(axis=0))[-min(15, len(go_terms)):][::-1]
            _top2_nms  = [gnames.get(go_terms[i], go_terms[i])[:22] for i in _top2_idx]
            _fig_hm2 = px.imshow(
                _hit_scores[:, _top2_idx], x=_top2_nms,
                y=[str(_i) for _i in _hit_ids],
                color_continuous_scale='Blues', aspect='auto', zmin=0, zmax=1,
            )
            _fig_hm2.update_layout(
                height=max(160, 28 * len(_hit_ids) + 50),
                margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_showscale=False,
            )
            _fig_hm2.update_xaxes(tickangle=-45, tickfont_size=9)
            st.plotly_chart(_fig_hm2, use_container_width=True, key=f"{key_prefix}_hm2")

    # ── Metrics ───────────────────────────────────────────────────────────
    with _lc_met:
        st.metric("Isoforms", str(_mask.sum()))
        _n_high = int((_max_per_iso >= thr).sum())
        st.metric("High-conf", str(_n_high), delta=f"≥{thr}")
        _novel_t = {'nic', 'nnic'}
        _n_novel = int(sum(str(t).lower() in _novel_t for t in _hit_types))
        st.metric("Novel", str(_n_novel))
        if dtu_df is not None:
            _ids_up = set(str(x).upper() for x in _hit_ids)
            _n_dtu  = 0
            for _dc in ['isoform_id', 'transcript_id', 'feature']:
                if _dc in dtu_df.columns:
                    _n_dtu = dtu_df[dtu_df[_dc].str.upper().isin(_ids_up)].shape[0]
                    break
            st.metric("DTU events", str(_n_dtu))

    # ── DTU canonical vs alternative chart (full-width row below the 3 columns) ──
    _dtu_fig = _build_dtu_card_chart(
        gene        = gene,
        hit_ids     = _hit_ids,
        hit_scores  = _hit_scores,
        hit_types   = _hit_types,
        dtu_df      = dtu_df,
        go_terms_list = go_terms,
        go_names_dict = gnames,
        score_thr   = thr,
    )
    if _dtu_fig is not None:
        st.plotly_chart(_dtu_fig, use_container_width=True,
                        key=f"{key_prefix}_dtu_card_chart")

# ── Carousel renderer ─────────────────────────────────────────────────────────
def _quick_card_panel(cands: pd.DataFrame, idx_key: str, badge: str,
                      badge_color: str, tab_key: str = '',
                      tag_scenario: object = None):
    if len(cands) == 0:
        st.info("해당 조건의 케이스가 없습니다.")
        return
    total = len(cands)
    idx   = max(0, min(st.session_state.get(idx_key, 0), total - 1))
    row   = cands.iloc[idx]
    gene  = str(row.get('gene_id', row['isoform_id']))
    iso   = str(row['isoform_id'])

    # Navigation row
    _nv1, _nv2, _nv3 = st.columns([1, 4, 1])
    with _nv1:
        if st.button("◀ 이전", key=f"qprev_{tab_key}", use_container_width=True,
                     disabled=(idx == 0)):
            st.session_state[idx_key] = idx - 1
            st.rerun()
    with _nv2:
        st.markdown(
            f"<div style='text-align:center;padding:4px 0;font-size:0.85rem;color:#64748b'>"
            f"케이스 <b style='color:#1e293b'>{idx + 1}</b> / {total}"
            f" &nbsp;—&nbsp; <b style='color:#1e293b'>{gene}</b></div>",
            unsafe_allow_html=True,
        )
    with _nv3:
        if st.button("다음 ▶", key=f"qnext_{tab_key}", use_container_width=True,
                     disabled=(idx >= total - 1)):
            st.session_state[idx_key] = idx + 1
            st.rerun()

    # Quick Report (same format as _render_gene_landing in 05_targets.py)
    _render_gene_quick_report(gene, scenario_badge=badge, badge_color=badge_color,
                              key_prefix=f"{tab_key}_{idx}")

    # Action buttons
    _ba, _bb = st.columns(2)
    with _ba:
        if st.button("➕ 바스켓 추가", key=f"qcard_bsk_{tab_key}", use_container_width=True):
            _add_to_basket(gene, tag_scenario=tag_scenario)
            st.rerun()
    with _bb:
        if st.button("🔬 상세 분석", key=f"qcard_ana_{tab_key}",
                     use_container_width=True, type="primary"):
            st.session_state['search_gene'] = gene
            st.session_state['deepdive_iso_id'] = iso
            st.switch_page("pages/06_isoform.py")

    # Per-gene interpretation report
    try:
        from prism_app.reports.interpreter import interpret_target_gene
        from prism_app.app.components.report_panel import render_report_panel as _rrp_hub
        _cl_hub = st.session_state.get('classified_df')
        _dtu_hub = st.session_state.get('cfg', {}).get('dtu_df')
        _gene_report = interpret_target_gene(
            gene, classified_df=_cl_hub, dtu_df=_dtu_hub,
            cfg=st.session_state.get('cfg'))
        _rrp_hub(_gene_report,
                 section_name=f"{gene} 분석",
                 download_filename=f"prism_{gene}_report.md",
                 key=f"hub_{tab_key}_{idx}")
    except Exception:
        pass

    # Candidate strip
    with st.expander(f"전체 {total}개 케이스 목록", expanded=False):
        _ncols = min(6, total)
        _strip_cols = st.columns(_ncols)
        for _ci in range(total):
            _crow   = cands.iloc[_ci]
            _cg     = str(_crow.get('gene_id', _crow['isoform_id']))
            _csc    = float(_crow.get('max_score', 0))
            _active = (_ci == idx)
            with _strip_cols[_ci % _ncols]:
                if st.button(
                    f"{'▶ ' if _active else ''}{_cg[:9]}\n{_csc:.2f}",
                    key=f"qstrip_{tab_key}_{_ci}",
                    use_container_width=True,
                    type="primary" if _active else "secondary",
                ):
                    st.session_state[idx_key] = _ci
                    st.rerun()

# ── 4 Tabs ────────────────────────────────────────────────────────────────────
_tab_s1, _tab_s3, _tab_div, _tab_dtu = st.tabs([
    f"🔴 Scenario 1 · 기능 스위치  ({len(_s1_cands)})",
    f"🟢 Scenario 3 · 신규 기능  ({len(_s3_cands)})",
    f"⚡ 아이소폼 분기  ({len(_div_cands)})",
    f"🔄 DTU 변화 상위  ({len(_dtu_cands)})",
])

with _tab_s1:
    st.caption("DTU 변화 + 신규 GO 기능 예측 동시 충족 — 기능 스위치 유력 후보 아이소폼")
    _quick_card_panel(_s1_cands, '_cidx_s1', 'S1 · 기능 스위치', '#dc2626', 's1', tag_scenario=1)

with _tab_s3:
    st.caption("DTU 독립적 고신뢰 PRISM 스코어 — 발현 변화 없이 기능적으로 특이적인 아이소폼")
    _quick_card_panel(_s3_cands, '_cidx_s3', 'S3 · 신규 기능', '#16a34a', 's3', tag_scenario=3)

with _tab_div:
    if len(_div_cands):
        st.caption("같은 유전자 내 아이소폼 간 PRISM 스코어 차이 최대 — 기능 다양성 집중 탐색 대상")
        _quick_card_panel(_div_cands, '_cidx_div', '고분기', '#d97706', 'div')
    else:
        st.info("유전자 ID 정보가 없어 within-gene 분기 분석을 할 수 없습니다.")

with _tab_dtu:
    if len(_dtu_cands):
        st.caption("|ΔIF| 절댓값 상위 유전자 중 PRISM 고신뢰 아이소폼 — 기능 연관 발현 전환 후보")
        _quick_card_panel(_dtu_cands, '_cidx_dtu', 'DTU 상위', '#2563eb', 'dtu')
    else:
        st.info(
            "DTU 데이터가 없습니다. 사이드바에서 DTU 결과 파일을 업로드하거나 "
            "**Brain** 데모 데이터셋을 선택하세요."
        )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Basket
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
_basket_n_g = len(basket_gene_ids())
_basket_n_i = len(st.session_state.get('basket_isoforms', []))
st.subheader(f"🧺 분석 바스켓 (유전자 {_basket_n_g}개 · 아이소폼 {_basket_n_i}개)")

if _basket_n_g == 0 and _basket_n_i == 0:
    st.info("위에서 ➕ 버튼으로 후보 유전자를 추가하세요. 바스켓 상세 관리는 Analysis Hub에서 할 수 있습니다.")
else:
    _bk_ids = basket_gene_ids()
    if _bk_ids:
        st.caption("유전자: " + " · ".join(f"**{g}**" for g in _bk_ids[:8])
                   + ("…" if len(_bk_ids) > 8 else ""))
    _bk_cols = st.columns(2)
    with _bk_cols[0]:
        if st.button("📋 Hub에서 바스켓 상세 관리", key='tghub_basket_goto_hub',
                     use_container_width=True):
            st.switch_page("pages/00_hub.py")
    with _bk_cols[1]:
        if st.button("🗑️ 바스켓 전체 초기화", key='tghub_basket_clear',
                     use_container_width=True):
            clear_basket()
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Module Map + Filter
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("🗺️ 모듈 분포 & 후보 필터")
st.caption(
    "5개 독립 축으로 후보를 좁히면 모듈 버블차트 위에 해당 아이소폼이 오버레이됩니다. "
    "모든 조건은 AND로 결합됩니다 — 각 축은 서로 다른 차원을 측정하므로 충돌하지 않습니다."
)

@st.cache_data(show_spinner=False)
def _load_mod_data():
    mod_j = REPORTS / 'brain_go_modules_672.json'
    iso_t = REPORTS / 'brain_isoform_modules.tsv'
    if not mod_j.exists() or not iso_t.exists():
        return None, None
    with open(mod_j) as f:
        md = json.load(f)
    df = pd.read_csv(iso_t, sep='\t')
    return md, df

_mod_data, _df_iso_ref = _load_mod_data()
_has_modules = _mod_data is not None and _df_iso_ref is not None

if _has_modules:
    # Per-module stats from isoform reference
    _df_merged = _df_iso_ref.merge(
        classified[['isoform_id', 'scenario', 'max_score']].rename(
            columns={'isoform_id': 'isoform_id'}),
        on='isoform_id', how='left',
    )
    _mods_dict = _mod_data.get('modules', {})

    _mod_stats = (
        _df_merged.groupby('primary_module')
        .agg(
            n_iso=('isoform_id', 'count'),
            mean_score=('max_score', 'mean'),
            n_s1=('scenario', lambda x: int((x == 1).sum())),
            n_s3=('scenario', lambda x: int((x == 3).sum())),
        )
        .reset_index()
    )
    _mod_stats['label'] = _mod_stats['primary_module'].apply(
        lambda m: _mods_dict.get(str(m), {}).get('label', f'M{m}')[:22]
    )
    # Grid layout (7 columns)
    _n_cols = 7
    _mod_stats = _mod_stats.reset_index(drop=True)
    _mod_stats['gx'] = _mod_stats.index % _n_cols
    _mod_stats['gy'] = _mod_stats.index // _n_cols

    # Filter panel + chart side by side
    _flt_col, _map_col = st.columns([1, 3])

    # Load BISECT PASS gene set once
    @st.cache_data(show_spinner=False)
    def _load_bisect_pass_genes() -> set:
        _bj = Path(__file__).parents[2] / 'data' / 'demo' / 'bisect_cases.json'
        if not _bj.exists():
            return set()
        import json as _json
        with open(_bj) as _bf:
            _cases = _json.load(_bf)
        return {str(c.get('gene', '')).upper() for c in _cases if c.get('gene')}

    _bisect_pass_genes = _load_bisect_pass_genes()

    with _flt_col:
        st.markdown(
            "<div style='font-size:0.88rem;font-weight:700;color:#1e293b;"
            "margin-bottom:6px'>후보 필터 — 5개 독립 축</div>",
            unsafe_allow_html=True,
        )

        # ① 기능 변화 유형 (single-select → 다른 축과 AND 가능)
        _SCENARIO_OPTS = {
            '전체 (필터 없음)': None,
            '🔴 S1 기능 스위치 (DTU+ & GO+)': 1,
            '🟠 S2 발현 변화 (DTU+ & GO-)':  2,
            '🟢 S3 구성적 신기능 (DTU- & GO+)': 3,
            '⚪ S4 배경/미분류':               4,
        }
        _f_scenario_label = st.selectbox(
            "① 기능 변화 유형",
            options=list(_SCENARIO_OPTS.keys()),
            index=0,
            key='f_scenario',
            help="시나리오는 DTU 유무 × 신규 GO 예측 유무의 2×2 조합입니다. "
                 "selectbox이므로 다른 조건과 AND로 결합됩니다.",
        )
        _f_scenario_val = _SCENARIO_OPTS[_f_scenario_label]

        st.markdown(
            "<div style='height:1px;background:#e2e8f0;margin:8px 0'></div>",
            unsafe_allow_html=True,
        )

        # ② 서열 기원 — 신규 미주석 아이소폼 여부 (시나리오와 독립)
        _f_novel = st.checkbox(
            "② 🆕 신규 아이소폼 (NIC/NNIC)",
            value=False, key='f_novel',
            help="IsoQuant 분류: NIC(Novel In-Catalog), NNIC(Novel Not-In-Catalog). "
                 "주석에 없는 이소폼만 선택합니다. S1~S4 모든 시나리오에 존재할 수 있습니다.",
        )

        # ③ PRISM 예측 신뢰도 — 점수 기준 (시나리오·기원과 독립)
        _f_high = st.checkbox(
            f"③ ⭐ 고신뢰 (score ≥ {thr})",
            value=False, key='f_high',
            help=f"PRISM 최대 GO score ≥ {thr}인 아이소폼. "
                 "신규 기능(GO+) 여부와는 별개로, 예측 점수 자체의 절대값 기준입니다.",
        )

        # ④ 사용량 변화 증거 — DTU 유의성 (시나리오·점수와 독립)
        if dtu_df is not None:
            _f_dtu = st.checkbox(
                "④ 🔄 DTU 유의 변화",
                value=False, key='f_dtu',
                help="실험적으로 아이소폼 사용 비율이 유의하게 변한 것(DTU)이 확인된 아이소폼. "
                     "S3(DTU-)처럼 점수는 높아도 DTU 없는 경우와 구분됩니다.",
            )
        else:
            _f_dtu = False
            st.caption("④ DTU 데이터 없음")

        # ⑤ BISECT 다중 검증 — 13-모듈 분석 통과 (모든 축과 독립)
        _bisect_available = len(_bisect_pass_genes) > 0
        _f_bisect = st.checkbox(
            f"⑤ 🏆 BISECT PASS 유전자 ({len(_bisect_pass_genes)}개)"
            if _bisect_available else "⑤ BISECT 데이터 없음",
            value=False, key='f_bisect',
            disabled=not _bisect_available,
            help="구조·진화·조절·기능 13개 모듈을 통과한 BISECT PASS 케이스의 유전자. "
                 "PRISM 점수·시나리오·DTU와 별개로 다중 증거로 검증된 후보입니다.",
        )

        # Apply all filters (AND logic — each axis is independent)
        _any_filter = (_f_scenario_val is not None or _f_novel or _f_high or _f_dtu or _f_bisect)

        if _any_filter:
            _fmask = pd.Series([True] * len(classified), index=classified.index)

            if _f_scenario_val is not None:
                _fmask &= classified['scenario'] == _f_scenario_val

            if _f_novel and 'isoform_type' in classified.columns:
                _fmask &= classified['isoform_type'].str.upper().isin(['NIC', 'NNIC'])

            if _f_high:
                _fmask &= classified['max_score'] >= thr

            if _f_dtu and 'dtu_flag' in classified.columns:
                _fmask &= classified['dtu_flag'] == True

            if _f_bisect and _bisect_available and 'gene_id' in classified.columns:
                _fmask &= classified['gene_id'].str.upper().isin(_bisect_pass_genes)

            _filt_ids = set(classified[_fmask]['isoform_id'])
            _filt_n   = len(_filt_ids)

            # Show active filter summary
            _active_labels = []
            if _f_scenario_val is not None:
                _active_labels.append(_f_scenario_label.split('(')[0].strip())
            if _f_novel:    _active_labels.append("NIC/NNIC")
            if _f_high:     _active_labels.append(f"score≥{thr}")
            if _f_dtu:      _active_labels.append("DTU+")
            if _f_bisect:   _active_labels.append("BISECT PASS")

            st.markdown(
                f"<div style='background:#f0fdf4;border-radius:6px;padding:8px 10px;"
                f"font-size:0.82rem;color:#15803d;margin-top:8px'>"
                f"✅ <b>{_filt_n:,}개</b> 아이소폼<br>"
                f"<span style='font-size:0.76rem;color:#374151'>"
                f"조건: {' &amp; '.join(_active_labels)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if st.button("➕ 필터 결과 바스켓 추가 (상위 유전자)", key='filt_basket_add'):
                if 'gene_id' in classified.columns:
                    _filt_genes = (
                        classified[_fmask]
                        .sort_values('max_score', ascending=False)
                        .drop_duplicates('gene_id')['gene_id']
                        .head(10).tolist()
                    )
                    for _fg in _filt_genes:
                        add_to_gene_basket(str(_fg), source_page='target_hub')
                    st.rerun()
        else:
            _filt_ids = None

    with _map_col:
        _fig_mod = go.Figure()

        # Base bubbles
        _size_scale = np.sqrt(_mod_stats['n_iso'].clip(lower=1)) / np.sqrt(_mod_stats['n_iso'].max()) * 60 + 10
        _fig_mod.add_trace(go.Scatter(
            x=_mod_stats['gx'],
            y=_mod_stats['gy'],
            mode='markers+text',
            marker=dict(
                size=_size_scale,
                color=_mod_stats['mean_score'],
                colorscale='RdYlGn', cmin=0, cmax=1,
                colorbar=dict(title='Mean<br>Score', thickness=12, len=0.7),
                line=dict(width=1.5, color='white'),
                opacity=0.85,
            ),
            text=_mod_stats['label'],
            textposition='bottom center',
            textfont=dict(size=7, color='#374151'),
            hovertemplate=(
                '<b>M%{customdata[0]}</b> — %{text}<br>'
                'N=%{customdata[1]:,}<br>'
                'Mean score=%{marker.color:.3f}<br>'
                'S1=%{customdata[2]}  S3=%{customdata[3]}'
                '<extra></extra>'
            ),
            customdata=_mod_stats[['primary_module', 'n_iso', 'n_s1', 'n_s3']].values,
            name='모듈',
        ))

        # Overlay: filtered isoforms
        if _filt_ids:
            _fiso = _df_iso_ref[_df_iso_ref['isoform_id'].isin(_filt_ids)].copy()
            _fiso = _fiso.merge(
                _mod_stats[['primary_module', 'gx', 'gy']],
                on='primary_module', how='left',
            ).dropna(subset=['gx', 'gy'])
            if len(_fiso):
                _rng = np.random.default_rng(42)
                _fiso['jx'] = _fiso['gx'] + _rng.uniform(-0.25, 0.25, len(_fiso))
                _fiso['jy'] = _fiso['gy'] + _rng.uniform(-0.25, 0.25, len(_fiso))
                _fig_mod.add_trace(go.Scatter(
                    x=_fiso['jx'], y=_fiso['jy'],
                    mode='markers',
                    marker=dict(size=5, color='#ef4444', opacity=0.55,
                                symbol='circle'),
                    name=f"필터 결과 ({len(_fiso):,}개)",
                    hovertext=_fiso['isoform_id'].astype(str),
                    hovertemplate='%{hovertext}<extra></extra>',
                ))

        _fig_mod.update_layout(
            title="기능 모듈 분포도 (44 modules)",
            height=420,
            plot_bgcolor='#f8fafc',
            paper_bgcolor='white',
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            margin=dict(l=10, r=40, t=40, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=-0.12),
        )
        st.plotly_chart(_fig_mod, use_container_width=True)

else:
    # Non-brain_672: show top-GO bar chart as alternative
    st.info(
        "모듈 지형도는 **Brain — Full (672 GO)** 데이터셋에서 사용 가능합니다.\n\n"
        "현재 데이터의 상위 GO term 분포를 표시합니다."
    )
    if len(go_terms) > 0:
        _go_mean = sm_arr.mean(axis=0)
        _top_idx = np.argsort(_go_mean)[-20:][::-1]
        _fig_go = px.bar(
            x=[gnames.get(go_terms[i], go_terms[i])[:30] for i in _top_idx],
            y=[float(_go_mean[i]) for i in _top_idx],
            labels={'x': 'GO term', 'y': '평균 PRISM score'},
            color=[float(_go_mean[i]) for i in _top_idx],
            color_continuous_scale='RdYlGn', range_color=[0, 1],
            title="전체 아이소폼 GO term 평균 점수 (상위 20개)",
            height=350,
        )
        _fig_go.update_layout(xaxis_tickangle=-40, showlegend=False,
                               plot_bgcolor='white', paper_bgcolor='white',
                               coloraxis_showscale=False)
        _fig_go.add_hline(y=thr, line_dash='dash', line_color='grey',
                           annotation_text=f"threshold ({thr})")
        st.plotly_chart(_fig_go, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — Navigation Cards
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("▶ 분석 섹션으로 이동")

_nd1, _nd2, _nd3 = st.columns(3)

with _nd1:
    st.markdown("""<div style='background:#fef2f2;border-radius:10px;
    padding:20px 16px;min-height:120px'>
    <div style='font-size:1.6rem'>📋</div>
    <b style='color:#991b1b'>시나리오 분류 & BISECT</b><br>
    <span style='font-size:0.8rem;color:#64748b'>
    S1~S4 시나리오별 아이소폼 테이블 · BISECT 84 케이스 심층 분석
    </span></div>""", unsafe_allow_html=True)
    if st.button("▶ 시나리오 & BISECT", key='nav_scenarios', use_container_width=True):
        st.switch_page("pages/05_targets.py")

with _nd2:
    st.markdown("""<div style='background:#f0fdf4;border-radius:10px;
    padding:20px 16px;min-height:120px'>
    <div style='font-size:1.6rem'>🔬</div>
    <b style='color:#15803d'>개별 아이소폼 분석</b><br>
    <span style='font-size:0.8rem;color:#64748b'>
    아이소폼 검색 · GO 프로파일 · GAIN/LOSS · within-gene 비교
    </span></div>""", unsafe_allow_html=True)
    if st.button("▶ 개별 아이소폼 분석", key='nav_isoform', use_container_width=True):
        st.switch_page("pages/06_isoform.py")

with _nd3:
    st.markdown("""<div style='background:#eff6ff;border-radius:10px;
    padding:20px 16px;min-height:120px'>
    <div style='font-size:1.6rem'>🗺️</div>
    <b style='color:#1d4ed8'>Module Landscape</b><br>
    <span style='font-size:0.8rem;color:#64748b'>
    672 GO → 44 모듈 전체 지형도 · 모듈 클러스터 탐색
    </span></div>""", unsafe_allow_html=True)
    if st.button("▶ Module Landscape", key='nav_landscape', use_container_width=True):
        st.switch_page("pages/02_landscape.py")
