"""Page 4 — Individual Isoform Analysis (Modules D1 + D2)."""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from prism_app.core.classifier import get_scenario_candidates, IsoformScenario
from prism_app.app.components.interpretation import render_data_context_banner

st.set_page_config(page_title="Individual Analysis — PRISM", layout="wide")
st.title("🔬 Individual Isoform Analysis")
st.caption("4-Scenario classification, isoform search, and case report.")

with st.expander("📖 이 페이지 사용법", expanded=False):
    st.markdown("""
**Individual Analysis** 페이지는 아이소폼을 시나리오별로 탐색하고, 특정 유전자/아이소폼의 상세 리포트를 생성합니다.

**시나리오 탭 활용법:**
- **S1 (빨강)** DTU + 신규 GO 예측 → 조건 의존적 기능 변화. DTU 파일이 있을 때만 등장합니다.
- **S2 (주황)** DTU + GO 변화 없음 → 발현량만 바뀐 스위치.
- **S3 (초록)** DTU 없음 + 신규 GO 예측 → 항상 발현되는 신규 기능 아이소폼. 논문의 541개 뇌 아이소폼이 여기에 해당합니다.
- **S4 (회색)** 특이사항 없음 → 배경 아이소폼.

각 탭에서 **CSV 다운로드** 버튼으로 해당 시나리오 후보 목록을 저장할 수 있습니다.

**검색 탭 활용법:**
1. 아이소폼 ID (예: `NDUFS4-201`) 또는 유전자 이름 (예: `NDUFS4`, `KIF21B`)을 입력합니다.
2. 매칭된 아이소폼의 GO 스코어 막대 차트와 시나리오 카드를 확인합니다.
3. **"Download case report (Markdown)"** 버튼으로 개별 케이스 리포트를 `.md` 파일로 저장합니다.

> Score 임계값(사이드바 슬라이더)에 따라 시나리오 분류가 달라집니다. 0.5가 기본값입니다.
    """)

# ── Data ─────────────────────────────────────────────────────────────────────
cfg = st.session_state.get('cfg', {})
sm  = cfg.get('score_matrix')
if sm is None:
    st.warning("No data loaded. Return to the main page."); st.stop()

render_data_context_banner(cfg)

# ── Linked View: UMAP cluster filter ─────────────────────────────────────────
_cluster_filter = st.session_state.get('umap_cluster_filter')
if _cluster_filter:
    _cname = _cluster_filter.get('cluster_name', 'Selected cluster')
    _cids  = set(_cluster_filter.get('isoform_ids', []))
    _cn    = _cluster_filter.get('n_isoforms', 0)
    col_lv, col_lv_clear = st.columns([5, 1])
    with col_lv:
        st.markdown(
            f"""<div style='background:#fef9c3;border-left:4px solid #eab308;
            padding:10px 16px;border-radius:6px;margin:4px 0 12px 0;font-size:0.87rem'>
            🔗 <b>Functional Map 연동 활성</b> — 클러스터: <b>{_cname}</b>
            ({_cn:,}개 아이소폼) · 아래 시나리오 탭이 이 클러스터로 필터됩니다.
            </div>""",
            unsafe_allow_html=True,
        )
    with col_lv_clear:
        if st.button("✖ 필터 해제", key='clear_cluster_filter'):
            del st.session_state['umap_cluster_filter']
            st.rerun()
else:
    _cids = None

ids    = cfg['isoform_ids']
genes  = cfg.get('gene_ids')
go     = cfg['go_terms']
gnames = cfg['go_names']
thr    = cfg['score_threshold']

classified = st.session_state.get('classified_df')
if classified is None:
    from prism_app.core.classifier import classify_isoforms
    classified = classify_isoforms(sm, ids, genes, go,
                                   score_threshold=thr,
                                   dtu_df=cfg.get('dtu_df'))
    st.session_state['classified_df'] = classified

# ── Scenario filter tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab_search, tab_bisect = st.tabs([
    "🔴 Scenario 1: Functional Switch",
    "🟠 Scenario 2: Expression Switch",
    "🟢 Scenario 3: Constitutive Novel",
    "⬜ Scenario 4: Background",
    "🔍 Search Isoform",
    "🧫 BISECT Cases",
])

SCENARIO_DESCS = {
    1: "DTU detected + novel GO function predicted → highest priority for experimental follow-up.",
    2: "DTU detected but no novel GO function → structural isoform change without detected function gain.",
    3: "No DTU but novel GO function predicted → constitutively expressed novel function (Use Case B).",
    4: "No DTU, no novel GO prediction → low-priority background isoforms.",
}


def _render_scenario_table(scenario_id: int) -> None:
    desc = SCENARIO_DESCS[scenario_id]
    st.info(desc, icon=["🔴","🟠","🟢","⬜"][scenario_id - 1])

    cands = get_scenario_candidates(classified, scenario_id, min_score=thr)

    # Apply cluster filter if active (linked view from UMAP)
    _active_cluster_ids = st.session_state.get('umap_cluster_filter', {}).get('isoform_ids')
    if _active_cluster_ids:
        _active_set = set(_active_cluster_ids)
        cands = cands[cands['isoform_id'].isin(_active_set)]

    if cands.empty:
        if scenario_id in (1, 2) and cfg.get('dtu_df') is None:
            st.markdown(
                f"""<div style='background:#fffbeb;border-left:4px solid #f59e0b;
                padding:16px 20px;border-radius:8px;margin:8px 0'>
                <b>⚠️ Scenario {scenario_id}가 비어있는 이유</b><br><br>
                이 시나리오는 <b>DTU (Differential Transcript Usage)</b> 분석 결과가 필요합니다.
                DTU 분석은 두 조건(예: 질병 vs. 정상) 간에 아이소폼 사용 비율이 통계적으로
                달라진 전사체를 식별합니다.<br><br>
                <b>활성화 방법:</b><br>
                1. satuRn / DEXSeq / IsoformSwitchAnalyzeR 등으로 DTU 분석 실행<br>
                2. 사이드바 → <b>Upload 모드</b> → DTU 결과 파일(.tsv) 업로드<br>
                3. 필요 컬럼: <code>isoform_id</code>, <code>delta_IF</code> (또는 <code>dIF</code>), <code>pvalue</code><br><br>
                Demo 데이터는 단일 조건이므로 DTU를 계산할 수 없습니다.
                <b>Scenario 3 (신규 기능)</b>은 DTU 없이도 분석 가능합니다.
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.write(f"현재 설정(Score > {thr})에서 이 시나리오에 해당하는 아이소폼이 없습니다. 임계값을 낮춰보세요.")
        return

    st.metric("Isoforms in scenario", len(cands))
    disp = cands[['isoform_id', 'gene_id', 'max_score', 'max_go', 'n_high_go',
                  'novel_go_terms', 'dtu_pvalue']].copy()
    disp['max_go'] = disp['max_go'].map(lambda g: f"{g}: {gnames.get(g,'')[:35]}")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # Download button
    csv = cands.to_csv(index=False).encode('utf-8')
    st.download_button(
        f"Download Scenario {scenario_id} candidates (CSV)",
        csv,
        f"scenario_{scenario_id}_candidates.csv",
        "text/csv",
        key=f"dl_scenario_{scenario_id}",
    )


with tab1: _render_scenario_table(1)
with tab2: _render_scenario_table(2)
with tab3: _render_scenario_table(3)
with tab4: _render_scenario_table(4)


def _build_case_report_md(row, go_df: pd.DataFrame, gnames: dict, thr: float) -> str:
    high = go_df[go_df['Score'] > thr]
    high_lines = "\n".join(
        f"| {r['GO_ID']} | {gnames.get(r['GO_ID'], r['GO'])[:50]} | {r['Score']:.3f} |"
        for _, r in high.iterrows()
    )
    return f"""# PRISM Case Report: {row['isoform_id']}

## Classification
- **Scenario**: {row['scenario']} — {row['scenario_label']}
- **Gene**: {row['gene_id']}
- **Max GO score**: {row['max_score']:.3f} ({gnames.get(row['max_go'], row['max_go'])})
- **DTU p-value**: {row['dtu_pvalue'] if row['dtu_pvalue'] is not None else 'N/A'}
- **DTU flag**: {row['dtu_flag']}

## High-Confidence GO Predictions (score > {thr})

| GO ID | Function | Score |
|-------|----------|-------|
{high_lines if high_lines else '| — | No high-confidence predictions | — |'}

## Novel GO terms (absent from existing annotation)
{row['novel_go_terms'] or 'None detected'}

---
*Generated by PRISM v0.1.0 · Lee et al. (2026)*
"""


# ── Isoform search / case report ─────────────────────────────────────────────
with tab_search:
    st.subheader("Isoform Case Report")
    if _cids:
        st.caption(f"🔗 Functional Map 연동: {len(_cids):,}개 아이소폼 대상 검색 중")
    query = st.text_input("Search by isoform ID or gene name",
                          placeholder="e.g. NDUFS4-201, KIF21B, tr319500")

    if query:
        ids_arr = np.asarray(ids, dtype=str)
        mask = np.array([query.lower() in i.lower() for i in ids_arr])
        if genes is not None:
            genes_arr = np.asarray(genes, dtype=str)
            mask |= np.array([query.lower() in g.lower() for g in genes_arr])
        if _cids:
            cluster_mask = np.array([iso in _cids for iso in ids_arr])
            mask = mask & cluster_mask

        if mask.sum() == 0:
            st.warning(f"No isoforms matching '{query}'")
        else:
            hits = classified[classified['isoform_id'].str.contains(query, case=False, na=False)
                              | classified['gene_id'].str.contains(query, case=False, na=False)]
            st.write(f"**{len(hits)} isoforms found**")

            for _, row in hits.iterrows():
                iso_idx = np.where(ids_arr == row['isoform_id'])[0]
                if len(iso_idx) == 0:
                    continue
                idx = iso_idx[0]

                with st.expander(f"📋 {row['isoform_id']}  —  {row['scenario_label']}", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Max score",     f"{row['max_score']:.3f}")
                    c2.metric("Top GO",        gnames.get(row['max_go'], row['max_go'])[:35])
                    c3.metric("Scenario",      str(row['scenario']))

                    # Per-GO score bar chart
                    go_scores = sm[idx]
                    go_df = pd.DataFrame({'GO': [gnames.get(g, g)[:35] for g in go],
                                          'Score': go_scores,
                                          'GO_ID': go})
                    go_df = go_df.sort_values('Score', ascending=False)
                    fig = px.bar(go_df, x='GO', y='Score',
                                 color='Score', color_continuous_scale='RdYlGn',
                                 range_color=[0, 1],
                                 title=f"GO score profile: {row['isoform_id']}",
                                 height=320)
                    fig.update_layout(xaxis_tickangle=-40,
                                      showlegend=False,
                                      plot_bgcolor='white')
                    fig.add_hline(y=float(cfg['score_threshold']),
                                  line_dash='dash', line_color='grey',
                                  annotation_text=f"threshold ({cfg['score_threshold']})")
                    _safe_iso_key = row['isoform_id'].replace('/', '_').replace('.', '_')
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"search_go_chart_{_safe_iso_key}")

                    # Summary table
                    high_go_df = go_df[go_df['Score'] > thr][['GO', 'Score', 'GO_ID']]
                    if not high_go_df.empty:
                        st.write("**High-confidence GO predictions:**")
                        st.dataframe(high_go_df, use_container_width=True, hide_index=True)

                    dtu_pval = row.get('dtu_pvalue')
                    if dtu_pval is not None and not (isinstance(dtu_pval, float) and np.isnan(dtu_pval)):
                        st.write(f"**DTU p-value**: {float(dtu_pval):.2e}")

                    # Markdown report download
                    md = _build_case_report_md(row, go_df, gnames, thr)
                    st.download_button(
                        "Download case report (Markdown)",
                        md.encode('utf-8'),
                        f"case_report_{_safe_iso_key}.md",
                        "text/markdown",
                        key=f"dl_case_report_{_safe_iso_key}",
                    )

# ── BISECT Cases tab ──────────────────────────────────────────────────────────
with tab_bisect:
    import json
    from pathlib import Path

    _BISECT_PATH = Path(__file__).parents[3] / 'prism_app' / 'data' / 'demo' / 'bisect_cases.json'

    st.subheader("BISECT PASS Cases")
    st.caption("15-모듈 파이프라인을 통과한 기능 스위치 후보 케이스 (stage2_pass = True)")

    if not _BISECT_PATH.exists():
        st.warning("bisect_cases.json not found in demo data directory.")
        st.stop()

    with open(_BISECT_PATH) as _f:
        _bisect_raw = json.load(_f)

    _bdf = pd.DataFrame(_bisect_raw)

    # ── Cross-link: build S1 gene → isoform map ───────────────────────────────
    _s1_genes = set()
    _gene_to_rows = {}   # gene_id → list of classified rows (for PRISM chart)
    if classified is not None:
        _s1 = classified[classified['scenario'] == 1]
        _s1_genes = set(_s1['gene_id'].dropna().tolist())
        for _g, _grp in _s1.groupby('gene_id'):
            _gene_to_rows[_g] = _grp

    # ── Summary metrics ───────────────────────────────────────────────────────
    _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
    _mc1.metric("PASS Cases",    len(_bdf))
    _mc2.metric("Cell Types",    _bdf['cell_type'].nunique())
    _mc3.metric("Domain Gains",  int((_bdf['domains_gained'].fillna('') != '').sum()))
    _mc4.metric("NAT Overlap",   int(_bdf['nat'].fillna(False).sum()))
    _mc5.metric("S1 교차 유전자", len(_s1_genes & set(_bdf['gene'].dropna())))

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    _fcol1, _fcol2 = st.columns([2, 2])
    with _fcol1:
        _ct_opts = sorted(_bdf['cell_type'].dropna().unique().tolist())
        _ct_sel  = st.multiselect("Cell type 필터", _ct_opts, default=_ct_opts, key='bisect_ct')
    with _fcol2:
        _bq = st.text_input("유전자 검색", placeholder="예: KIF21B, DLG1, DMD", key='bisect_gene_q')

    _bdf_filt = _bdf[_bdf['cell_type'].isin(_ct_sel)] if _ct_sel else _bdf
    if _bq:
        _bdf_filt = _bdf_filt[_bdf_filt['gene'].str.contains(_bq, case=False, na=False)]

    # ── Summary table ─────────────────────────────────────────────────────────
    _col_map = {
        'gene': 'Gene', 'cell_type': 'Cell Type',
        'delta': 'Δ Usage', 'dtu_p': 'DTU p-val',
        'domains_gained': 'Domains Gained', 'domains_lost': 'Domains Lost',
        'nat': 'NAT', 'young_l1_cds': 'L1-CDS',
        'ppi_verdict': 'PPI', 'af_ad_plddt_mean': 'pLDDT',
        'cons_ad_phylop': 'phyloP',
    }
    _show_cols_raw = ['gene', 'cell_type', 'delta', 'dtu_p',
                      'domains_gained', 'domains_lost',
                      'ppi_verdict', 'af_ad_plddt_mean', 'cons_ad_phylop',
                      'nat', 'young_l1_cds']
    _show_cols = [c for c in _show_cols_raw if c in _bdf_filt.columns]
    _bdf_show  = _bdf_filt[_show_cols].rename(columns=_col_map).copy()

    def _highlight_s1_row(row):
        if row.get('Gene', '') in _s1_genes:
            return ['background-color: #fef9c3'] * len(row)
        return [''] * len(row)

    if _s1_genes:
        st.caption("🟡 강조 = 현재 데이터셋 Scenario 1 교차 유전자 | pLDDT ≥ 70 = AlphaFold 신뢰 구조")
        st.dataframe(
            _bdf_show.style.apply(_highlight_s1_row, axis=1).format(
                {'Δ Usage': '{:.3f}', 'pLDDT': '{:.1f}', 'phyloP': '{:.3f}',
                 'DTU p-val': '{:.2e}'},
                na_rep='—',
            ),
            use_container_width=True, hide_index=True,
        )
    else:
        st.dataframe(_bdf_show, use_container_width=True, hide_index=True)

    # ── Per-case expanders — only rendered when a gene search is active ──────
    # Rendering all 84 expanders at once is expensive (domain maps, DTU charts,
    # IGV iframes). Gate on search query so the table always stays fast.
    if not _bdf_filt.empty:
        st.divider()
        if not _bq:
            st.info(
                f"위 표에서 **{len(_bdf_filt)}건**의 PASS 케이스를 확인하세요. "
                "유전자 이름을 **위 검색창에 입력**하면 도메인 구조·DTU·IGV 등 "
                "상세 분석이 펼쳐집니다.  예) `KIF21B`, `DLG1`, `NDUFS4`",
                icon="🔍",
            )
        else:
            st.markdown(f"**케이스 상세** — '{_bq}' 검색 결과 {len(_bdf_filt)}건")

    # ── DTU lookup dict (cached) — built once, O(1) per gene lookup ──────────
    @st.cache_data(show_spinner=False)
    def _build_dtu_lookup(dtu_bytes: bytes) -> dict:
        import io
        _d = pd.read_csv(io.BytesIO(dtu_bytes), sep='\t')
        _lk: dict = {}
        if 'condition' in _d.columns:
            for (_g, _c), _grp in _d.groupby(['gene_id', 'condition']):
                _lk[(_g, _c)] = _grp.reset_index(drop=True)
        else:
            for _g, _grp in _d.groupby('gene_id'):
                _lk[(_g, '')] = _grp.reset_index(drop=True)
        return _lk

    _dtu_src = cfg.get('dtu_df')
    _dtu_lookup: dict = {}
    if _dtu_src is not None:
        _dtu_lookup = _build_dtu_lookup(_dtu_src.to_csv(index=False).encode())

    if _bq and not _bdf_filt.empty:
        for _, _brow in _bdf_filt.iterrows():
            _gene = _brow.get('gene', '?')
            _ct   = _brow.get('cell_type', '?')
            _is_s1 = _gene in _s1_genes
            _title = f"{'🟡 ' if _is_s1 else '🧫 '}{_gene} — {_ct}"
            if _is_s1:
                _title += "  ·  🔴 Scenario 1 PASS"

            with st.expander(_title, expanded=bool(_bq)):
                # ── Isoform pair ──────────────────────────────────────────────
                _ct_tx = str(_brow.get('ct_transcript_id') or '').strip()
                _ad_tx = str(_brow.get('ad_transcript_id') or '').strip()
                _safe_ct_key = _ct.replace(' ', '_').replace('-', '_')
                if _ct_tx or _ad_tx:
                    st.markdown(
                        f"<div style='background:#f8fafc;border-radius:6px;"
                        f"padding:6px 12px;font-size:0.82rem;color:#475569;margin-bottom:8px'>"
                        f"🔵 CT: <code>{_ct_tx or '—'}</code> &nbsp;→&nbsp; "
                        f"🔴 AD: <code>{_ad_tx or '—'}</code></div>",
                        unsafe_allow_html=True,
                    )

                # ── DTU Δ Usage bar chart (O(1) lookup via cached dict) ───────
                _CT_COND_MAP = {'Excitatory': 'Excitatory neuron',
                                'Inhibitory': 'Inhibitory neuron'}
                _dtu_cond = _CT_COND_MAP.get(_ct, _ct)
                _g_dtu = _dtu_lookup.get((_gene, _dtu_cond),
                         _dtu_lookup.get((_gene, ''), pd.DataFrame()))
                if not _g_dtu.empty:
                    _g_dtu = _g_dtu.sort_values('delta_IF').reset_index(drop=True)
                    _g_dtu['role'] = _g_dtu['isoform_id'].map(
                        lambda iso: ('CT (Control)' if iso == _ct_tx
                                     else ('AD (Disease)' if iso == _ad_tx
                                           else 'Other isoform'))
                    )
                    _g_dtu['label'] = _g_dtu['delta_IF'].map(lambda v: f'{v:+.3f}')
                    _fig_dtu = px.bar(
                        _g_dtu,
                        x='isoform_id', y='delta_IF',
                        color='role',
                        color_discrete_map={
                            'CT (Control)':   '#3b82f6',
                            'AD (Disease)':   '#ef4444',
                            'Other isoform':  '#94a3b8',
                        },
                        title=f"Δ Usage (AD − CT) — {_gene}  ·  {_ct}",
                        labels={'delta_IF': 'Δ Usage (AD − CT)', 'isoform_id': ''},
                        text='label',
                        height=max(240, len(_g_dtu) * 30 + 80),
                    )
                    _fig_dtu.add_hline(y=0, line_color='#1e293b', line_width=1.2)
                    _fig_dtu.update_traces(textposition='outside', textfont_size=9)
                    _fig_dtu.update_layout(
                        xaxis_tickangle=-35,
                        legend_title='',
                        plot_bgcolor='white',
                        yaxis=dict(gridcolor='#f0f0f0'),
                        margin=dict(t=40, b=80, l=10, r=10),
                    )
                    st.plotly_chart(_fig_dtu, use_container_width=True,
                                    key=f"dtu_usage_{_gene}_{_safe_ct_key}")

                # ── Row 1: core metrics (6-col) ───────────────────────────────
                _r1c1, _r1c2, _r1c3, _r1c4, _r1c5, _r1c6 = st.columns(6)
                _delta = _brow.get('delta')
                _r1c1.metric("Δ Usage (AD−CT)",
                             f"{float(_delta):.3f}" if _delta is not None else "N/A")
                _dtu_p = _brow.get('dtu_p')
                _r1c2.metric("DTU p-value",
                             f"{float(_dtu_p):.2e}" if _dtu_p else "N/A")
                _ct_plddt = _brow.get('af_ct_plddt_mean')
                _r1c3.metric("CT pLDDT",
                             f"{float(_ct_plddt):.1f}" if _ct_plddt else "N/A",
                             help="AlphaFold pLDDT for Control transcript")
                _ad_plddt = _brow.get('af_ad_plddt_mean')
                _r1c4.metric("AD pLDDT",
                             f"{float(_ad_plddt):.1f}" if _ad_plddt else "N/A",
                             delta="신뢰" if _ad_plddt and float(_ad_plddt) >= 70 else None,
                             help="AlphaFold pLDDT for AD transcript")
                _dplddt = _brow.get('af_delta_plddt')
                _r1c5.metric("ΔpLDDT (AD−CT)",
                             f"{float(_dplddt):+.1f}" if _dplddt else "N/A",
                             help="Positive = AD isoform more structured")
                _phylo = _brow.get('cons_ad_phylop')
                _r1c6.metric("phyloP (AD exon)",
                             f"{float(_phylo):.3f}" if _phylo else "N/A",
                             help="Mean phyloP100way for AD-specific exon")

                # ── Row 2: domain changes with AlphaFold confidence ───────────
                _dg     = str(_brow.get('domains_gained')       or '').strip()
                _dl     = str(_brow.get('domains_lost')         or '').strip()
                _af_gd  = str(_brow.get('af_gained_confident')  or '').strip()
                _af_ld  = str(_brow.get('af_lost_confident')    or '').strip()

                _dc1, _dc2 = st.columns(2)
                with _dc1:
                    _dg_items = [d for d in _dg.split(';') if d]
                    _dg_html  = ''.join(
                        f"<code style='background:#dcfce7;padding:1px 5px;"
                        f"border-radius:3px'>{d}</code> " for d in _dg_items
                    ) if _dg_items else '<span style="color:#94a3b8">없음</span>'
                    _af_gd_html = (
                        f"<br><span style='font-size:0.75rem;color:#15803d'>"
                        f"🏗 AlphaFold 확인: {_af_gd}</span>"
                    ) if _af_gd else ''
                    st.markdown(
                        f"<div style='background:#f0fdf4;border-left:3px solid #22c55e;"
                        f"padding:8px 12px;border-radius:4px;font-size:0.85rem'>"
                        f"<b>도메인 획득 (AD)</b><br>{_dg_html}{_af_gd_html}</div>",
                        unsafe_allow_html=True,
                    )
                with _dc2:
                    _dl_items = [d for d in _dl.split(';') if d]
                    _dl_html  = ''.join(
                        f"<code style='background:#fee2e2;padding:1px 5px;"
                        f"border-radius:3px'>{d}</code> " for d in _dl_items
                    ) if _dl_items else '<span style="color:#94a3b8">없음</span>'
                    _af_ld_html = (
                        f"<br><span style='font-size:0.75rem;color:#b91c1c'>"
                        f"🏗 AlphaFold 확인: {_af_ld}</span>"
                    ) if _af_ld else ''
                    st.markdown(
                        f"<div style='background:#fef2f2;border-left:3px solid #ef4444;"
                        f"padding:8px 12px;border-radius:4px;font-size:0.85rem'>"
                        f"<b>도메인 손실 (CT)</b><br>{_dl_html}{_af_ld_html}</div>",
                        unsafe_allow_html=True,
                    )

                # ── Row 3: PPI + conservation detail ─────────────────────────
                _ppi_v  = str(_brow.get('ppi_verdict')    or '').strip()
                _ppi_p  = str(_brow.get('ppi_top_partner')or '').strip()
                _ppi_s  = _brow.get('ppi_top_score')
                _ppi_n  = _brow.get('ppi_n_string_hits')
                _cons_c = str(_brow.get('cons_ad_class')  or '').strip()
                _cons_bg= _brow.get('cons_background_phylop')
                _top_reg= str(_brow.get('top_regulators') or '').strip()

                _det_items = []
                if _ppi_v:
                    _ppi_clr = '#15803d' if _ppi_v == 'SUPPORTED' else '#b91c1c'
                    _ppi_txt = f"PPI: <b style='color:{_ppi_clr}'>{_ppi_v}</b>"
                    if _ppi_p:
                        _ppi_txt += f" (top: {_ppi_p}"
                        if _ppi_s:
                            _ppi_txt += f" score={int(_ppi_s)}"
                        _ppi_txt += f", n={int(_ppi_n) if _ppi_n else '?'})"
                    _det_items.append(_ppi_txt)
                if _phylo:
                    _cons_txt = f"Conservation: phyloP={float(_phylo):.3f}"
                    if _cons_c:
                        _cons_txt += f" ({_cons_c})"
                    if _cons_bg:
                        _cons_txt += f" | bg={float(_cons_bg):.3f}"
                    _det_items.append(_cons_txt)
                if _top_reg:
                    _det_items.append(f"Top regulators: {_top_reg}")

                if _det_items:
                    st.markdown(
                        "<div style='background:#f8fafc;border-radius:6px;padding:8px 12px;"
                        "font-size:0.82rem;color:#374151;margin:8px 0;line-height:1.8'>"
                        + "<br>".join(_det_items) + "</div>",
                        unsafe_allow_html=True,
                    )

                # ── Row 4: module verdict badges ──────────────────────────────
                _mod_items = []
                _seq_id = _brow.get('seq_val_identity')
                if _seq_id and float(_seq_id) > 0:
                    _mod_items.append(
                        f"<span style='background:#eff6ff;color:#1d4ed8;"
                        f"border-radius:4px;padding:2px 8px;font-size:0.8rem'>"
                        f"Seq identity: {float(_seq_id):.1%}</span>"
                    )
                _mech = str(_brow.get('mechanism_type') or '').strip()
                if _mech:
                    _mod_items.append(
                        f"<span style='background:#faf5ff;color:#7e22ce;"
                        f"border-radius:4px;padding:2px 8px;font-size:0.8rem'>"
                        f"Mechanism: {_mech}</span>"
                    )
                _tss = str(_brow.get('tss_class') or '').strip()
                _tss_bp = _brow.get('tss_diff_bp')
                if _tss:
                    _tss_txt = f"TSS: {_tss}"
                    if _tss_bp:
                        _tss_txt += f" ({int(float(_tss_bp)):+d}bp)"
                    _mod_items.append(
                        f"<span style='background:#fff7ed;color:#c2410c;"
                        f"border-radius:4px;padding:2px 8px;font-size:0.8rem'>"
                        f"{_tss_txt}</span>"
                    )
                _apa = str(_brow.get('apa_class') or '').strip()
                if _apa:
                    _apa_bp = _brow.get('tts_diff_bp')
                    _apa_txt = f"APA: {_apa}"
                    if _apa_bp:
                        _apa_txt += f" ({int(float(_apa_bp)):+d}bp)"
                    _mod_items.append(
                        f"<span style='background:#ecfdf5;color:#065f46;"
                        f"border-radius:4px;padding:2px 8px;font-size:0.8rem'>"
                        f"{_apa_txt}</span>"
                    )
                _ad_nmd = _brow.get('ad_nmd')
                _ct_nmd = _brow.get('ct_nmd')
                if _ad_nmd or _ct_nmd:
                    _nmd_txt = "NMD: "
                    _nmd_txt += ("AD✓" if _ad_nmd else "AD✗") + " / "
                    _nmd_txt += ("CT✓" if _ct_nmd else "CT✗")
                    _mod_items.append(
                        f"<span style='background:#fef9c3;color:#854d0e;"
                        f"border-radius:4px;padding:2px 8px;font-size:0.8rem'>"
                        f"{_nmd_txt}</span>"
                    )
                if _mod_items:
                    st.markdown(
                        "<div style='margin:10px 0 6px;display:flex;gap:6px;flex-wrap:wrap'>"
                        + "".join(_mod_items) + "</div>",
                        unsafe_allow_html=True,
                    )

                # ── Domain structure + IGV genomic view ───────────────────────
                _BISECT_OUT = Path(__file__).parents[3] / \
                    'Final_analysis' / 'pipeline_bioanalysis' / 'outputs'
                _dmap = _BISECT_OUT / f"{_gene}_{_ct}" / "domain_map.png"
                if _dmap.exists():
                    st.divider()
                    st.markdown(
                        "<div style='font-size:0.88rem;font-weight:600;"
                        "color:#1e293b;margin-bottom:4px'>"
                        "🗺️ 도메인 구조 변화 (CT → AD)</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Primary pair: 🔵 `{_ct_tx or '—'}` (Control) → "
                        f"🔴 `{_ad_tx or '—'}` (AD). "
                        "BISECT m11 AlphaFold + Pfam 기반 도메인 GAIN/LOSS 주석."
                    )
                    st.image(str(_dmap), use_column_width=True)

                # ── IGV / UCSC quick links ────────────────────────────────────
                st.divider()
                st.markdown(
                    "<div style='font-size:0.88rem;font-weight:600;"
                    "color:#1e293b;margin-bottom:6px'>"
                    "🧬 유전체 뷰 (외부 링크)</div>",
                    unsafe_allow_html=True,
                )
                _ql1, _ql2, _ql3 = st.columns(3)
                _igv_url  = f"https://igv.org/app/?genome=hg38&locus={_gene}"
                _ucsc_url = (f"https://genome.ucsc.edu/cgi-bin/hgTracks"
                             f"?db=hg38&position={_gene}&knownGene=pack"
                             f"&wgEncodeGencodeCompV45=pack")
                _ens_url  = (f"https://www.ensembl.org/Homo_sapiens/Gene/Summary"
                             f"?q={_gene};db=core")
                _ql1.link_button("🔬 IGV Web (hg38)", _igv_url)
                _ql2.link_button("🌐 UCSC Genome Browser", _ucsc_url)
                _ql3.link_button("🧫 Ensembl Gene View", _ens_url)
                st.caption(
                    f"핵심 전사체: 🔵 CT `{_ct_tx or '—'}` / "
                    f"🔴 AD `{_ad_tx or '—'}` — 각 브라우저에서 해당 전사체 ID로 검색하세요."
                )

                # ── Warnings ──────────────────────────────────────────────────
                if _brow.get('nat'):
                    st.warning("⚠️ NAT (Natural Antisense Transcript) overlap 감지됨")
                if _brow.get('young_l1_cds'):
                    st.warning("⚠️ Young L1 retrotransposon이 CDS 내 삽입됨")
                if _brow.get('nmd_relevant'):
                    st.info("ℹ️ NMD (Nonsense-Mediated Decay) 관련 구조 감지됨")

                # ── Option C: PRISM GO score chart for S1-overlapping genes ──
                if _is_s1 and _gene in _gene_to_rows:
                    st.divider()
                    st.markdown(
                        "<div style='background:#fef9c3;border-left:3px solid #eab308;"
                        "padding:8px 12px;border-radius:4px;font-size:0.85rem;margin-bottom:8px'>"
                        "🔴 <b>Scenario 1 확인됨</b> — 현재 데이터셋에서 DTU + 신규 GO 예측 교차 검증"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    _gene_rows = _gene_to_rows[_gene]
                    # Show top isoforms by max_score
                    _top_isos = _gene_rows.nlargest(min(5, len(_gene_rows)), 'max_score')
                    for _, _irow in _top_isos.iterrows():
                        _iso_idx_arr = np.where(np.asarray(ids, dtype=str) == _irow['isoform_id'])[0]
                        if len(_iso_idx_arr) == 0:
                            continue
                        _iso_idx = _iso_idx_arr[0]
                        _go_scores = sm[_iso_idx]
                        _go_df = pd.DataFrame({
                            'GO': [gnames.get(g, g)[:30] for g in go],
                            'Score': _go_scores,
                            'GO_ID': go,
                        }).sort_values('Score', ascending=False)
                        _fig = px.bar(
                            _go_df, x='GO', y='Score',
                            color='Score', color_continuous_scale='RdYlGn',
                            range_color=[0, 1],
                            title=f"PRISM GO scores: {_irow['isoform_id']} (max={_irow['max_score']:.3f})",
                            height=260,
                        )
                        _fig.update_layout(
                            xaxis_tickangle=-35,
                            showlegend=False,
                            plot_bgcolor='white',
                            margin=dict(t=40, b=60, l=10, r=10),
                        )
                        _fig.add_hline(
                            y=float(thr), line_dash='dash', line_color='grey',
                            annotation_text=f"threshold ({thr})",
                        )
                        _safe_irow_key = _irow['isoform_id'].replace('/', '_').replace('.', '_')
                        st.plotly_chart(_fig, use_container_width=True,
                                        key=f"bisect_s1_go_{_gene}_{_safe_ct_key}_{_safe_irow_key}")

    st.divider()
    st.download_button(
        "Download BISECT PASS cases (CSV)",
        _bdf_filt.to_csv(index=False).encode('utf-8'),
        "bisect_pass_cases.csv", "text/csv",
    )
