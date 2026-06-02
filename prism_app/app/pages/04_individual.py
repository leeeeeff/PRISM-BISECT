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
                    st.plotly_chart(fig, use_container_width=True)

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
                        f"case_report_{row['isoform_id'].replace('/','_')}.md",
                        "text/markdown",
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

    # Rename columns for display
    _col_map = {
        'gene': 'Gene', 'cell_type': 'Cell Type',
        'delta': 'Δ Usage (AD−CT)', 'dtu_p': 'DTU p-value',
        'domains_gained': 'Domains Gained', 'domains_lost': 'Domains Lost',
        'nat': 'NAT overlap', 'young_l1_cds': 'Young L1 in CDS',
    }
    _bdf_disp = _bdf.rename(columns=_col_map)

    # Summary metrics
    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
    _mc1.metric("PASS Cases", len(_bdf))
    _mc2.metric("Cell Types", _bdf['cell_type'].nunique())
    _mc3.metric("Domain Gains", int((_bdf['domains_gained'].fillna('') != '').sum()))
    _mc4.metric("NAT Overlap", int(_bdf['nat'].fillna(False).sum()))

    st.divider()

    # Cell type filter
    _ct_opts = sorted(_bdf['cell_type'].dropna().unique().tolist())
    _ct_sel  = st.multiselect("Cell type 필터", _ct_opts, default=_ct_opts, key='bisect_ct')
    _bdf_filt = _bdf[_bdf['cell_type'].isin(_ct_sel)] if _ct_sel else _bdf

    # Gene search
    _bq = st.text_input("유전자 검색", placeholder="예: KIF21B, NDUFS4", key='bisect_gene_q')
    if _bq:
        _bdf_filt = _bdf_filt[_bdf_filt['gene'].str.contains(_bq, case=False, na=False)]

    # Cross-link with PRISM Scenario 1
    _s1_genes = set()
    if classified is not None:
        _s1_genes = set(classified[classified['scenario'] == 1]['gene_id'].dropna().tolist())

    def _highlight_s1(row):
        if row.get('Gene', row.get('gene', '')) in _s1_genes:
            return ['background-color: #fef9c3'] * len(row)
        return [''] * len(row)

    _show_cols = [c for c in ['Gene', 'Cell Type', 'Δ Usage (AD−CT)', 'DTU p-value',
                               'Domains Gained', 'Domains Lost', 'NAT overlap', 'Young L1 in CDS']
                  if c in _bdf_disp.columns]
    _bdf_show = _bdf_filt.rename(columns=_col_map)[_show_cols].copy()

    if _s1_genes:
        st.caption("🟡 강조 표시 = 현재 데이터셋의 Scenario 1과 겹치는 유전자")
        st.dataframe(
            _bdf_show.style.apply(_highlight_s1, axis=1),
            use_container_width=True, hide_index=True,
        )
    else:
        st.dataframe(_bdf_show, use_container_width=True, hide_index=True)

    # Per-case expander for domain details
    if _bq and not _bdf_filt.empty:
        st.divider()
        st.markdown("**케이스 상세**")
        for _, _brow in _bdf_filt.iterrows():
            with st.expander(f"🧫 {_brow.get('gene','?')} — {_brow.get('cell_type','?')}", expanded=True):
                _bc1, _bc2, _bc3 = st.columns(3)
                _bc1.metric("Δ Usage (AD−CT)", f"{_brow.get('delta', float('nan')):.3f}")
                _bc2.metric("DTU p-value",
                            f"{float(_brow.get('dtu_p', 1)):.2e}" if _brow.get('dtu_p') else "N/A")
                _bc3.metric("BISECT Stage2", "✅ PASS")

                _dg = str(_brow.get('domains_gained') or '').strip()
                _dl = str(_brow.get('domains_lost')  or '').strip()
                if _dg:
                    st.markdown(f"**도메인 획득 (AD 아이소폼):** `{_dg}`")
                if _dl:
                    st.markdown(f"**도메인 손실 (CT 아이소폼):** `{_dl}`")
                if _brow.get('nat'):
                    st.warning("⚠️ NAT (Natural Antisense Transcript) overlap 감지됨")
                if _brow.get('young_l1_cds'):
                    st.warning("⚠️ Young L1 retrotransposon이 CDS 내 삽입됨")

    st.divider()
    st.download_button(
        "Download BISECT PASS cases (CSV)",
        _bdf_filt.to_csv(index=False).encode('utf-8'),
        "bisect_pass_cases.csv", "text/csv",
    )
