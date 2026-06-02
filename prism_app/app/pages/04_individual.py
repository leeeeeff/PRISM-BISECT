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

# ── Bio-report helper (called inside BISECT expanders) ───────────────────────
_DOMAIN_FUNC_MAP = {
    'Kinesin':       'microtubule-based motor activity (ATP-dependent)',
    'WD40':          'β-propeller scaffold for protein–protein interactions',
    'PDZ':           'synaptic scaffolding, C-terminal peptide binding',
    'SAM':           'oligomerization / RNA-binding (context-dependent)',
    'SH3':           'proline-rich sequence binding, signaling assembly',
    'SH2':           'phosphotyrosine binding, downstream signaling',
    'RRM':           'RNA recognition motif, post-transcriptional regulation',
    'Microtub_bd':   'direct microtubule binding and stabilization',
    'NDUS4':         'NADH:ubiquinone oxidoreductase (Complex I) assembly',
    'RVT_1':         'reverse-transcriptase / RNA-dependent DNA polymerase',
    'DUF5082':       'domain of unknown function (DUF5082)',
    'ANAPC4_WD40':   'APC/C complex scaffold, cell-cycle regulation',
    'Nup160':        'nuclear pore complex, nucleocytoplasmic transport',
    'PH':            'phosphoinositide binding, membrane recruitment',
    'Guanylate_kin': 'guanylate kinase activity, scaffolding at PSD',
    'RhoGAP':        'Rho GTPase-activating protein, cytoskeleton regulation',
    'RhoGEF':        'Rho guanine-nucleotide exchange factor',
    'Pkinase':       'serine/threonine protein kinase, signal transduction',
    'CARD':          'caspase recruitment domain, apoptosis regulation',
    'FN3':           'fibronectin type-III fold, cell adhesion',
    'EGF':           'EGF receptor binding, proliferation signaling',
    'BEACH':         'lysosome/endosome biogenesis regulation',
    'GRAM':          'membrane association with PH domain',
}


def _build_bio_report_html(
    brow: dict,
    gene: str,
    ct_type: str,
    ct_tx: str,
    ad_tx: str,
    ct_scores,
    ad_scores,
    go_ids: list,
    go_names: dict,
    threshold: float,
) -> str:
    """Return styled HTML biological prediction report from BISECT evidence."""
    import ast

    # ── Extract fields ────────────────────────────────────────────────────────
    delta   = brow.get('delta')
    dtu_p   = brow.get('dtu_p')
    dg      = str(brow.get('domains_gained') or '').strip()
    dl      = str(brow.get('domains_lost')   or '').strip()
    ppi_v   = str(brow.get('ppi_verdict')    or '').strip()
    ppi_p   = str(brow.get('ppi_top_partner')or '').strip()
    ppi_s   = brow.get('ppi_top_score')
    phylo   = brow.get('cons_ad_phylop')
    cons_c  = str(brow.get('cons_ad_class')  or '').strip()
    mech    = str(brow.get('mechanism_type') or '').strip()
    tss_cls = str(brow.get('tss_class')      or '').strip()
    apa_cls = str(brow.get('apa_class')      or '').strip()
    tss_bp  = brow.get('tss_diff_bp')
    apa_bp  = brow.get('tts_diff_bp')
    reg_raw = str(brow.get('top_regulators') or '').strip()
    ad_nmd  = brow.get('ad_nmd')
    ct_nmd  = brow.get('ct_nmd')
    af_ad   = brow.get('af_ad_plddt_mean')

    # parse top regulator name
    reg_name = ''
    if reg_raw and reg_raw not in ('None', ''):
        try:
            _rd = ast.literal_eval(reg_raw)
            reg_name = _rd.get('gene', '')
            reg_logfc = _rd.get('logFC', None)
        except Exception:
            reg_name = ''

    dg_list = [d for d in dg.split(';') if d]
    dl_list = [d for d in dl.split(';') if d]

    def _domain_func(d):
        for k, v in _DOMAIN_FUNC_MAP.items():
            if k.lower() in d.lower():
                return v
        return 'function uncharacterised'

    # ── PRISM top GO terms ────────────────────────────────────────────────────
    def _top_go(scores, n=3):
        if scores is None:
            return []
        idxs = np.argsort(scores)[-n:][::-1]
        return [(go_ids[i], go_names.get(go_ids[i], go_ids[i]), float(scores[i]))
                for i in idxs if scores[i] > 0.15]

    ct_top = _top_go(ct_scores)
    ad_top = _top_go(ad_scores)
    ct_go_ids_set = {g for g, _, _ in ct_top}
    ad_go_ids_set = {g for g, _, _ in ad_top}
    gained_go = [(g, n, s) for g, n, s in ad_top if g not in ct_go_ids_set]
    lost_go   = [(g, n, s) for g, n, s in ct_top if g not in ad_go_ids_set]

    # ── Confidence score ──────────────────────────────────────────────────────
    ev_count = sum([
        bool(delta and abs(float(delta)) > 0.1),
        bool(dtu_p and float(dtu_p) < 1e-5),
        bool(dg_list),
        bool(dl_list),
        ppi_v == 'SUPPORTED',
        bool(phylo and float(phylo) > 1.0),
        bool(gained_go or lost_go),
    ])
    conf_label = ['Low', 'Low', 'Moderate', 'Moderate', 'High', 'High', 'Very High'][min(ev_count, 6)]
    conf_color = {'Low': '#ef4444', 'Moderate': '#f59e0b', 'High': '#22c55e', 'Very High': '#15803d'}[conf_label]

    # ── Narrative sentences ───────────────────────────────────────────────────
    lines = []

    # 1. Isoform switch
    try:
        dv = float(delta)
    except Exception:
        dv = None
    if dv is not None:
        direction = '감소하며 대체됨' if dv < 0 else '증가함'
        lines.append(
            f"알츠하이머 조건 {ct_type} 세포에서 <b>{ct_tx or 'CT 이소폼'}</b>의 "
            f"사용 비율이 <b>Δ = {dv:+.3f}</b>로 {direction}하고 "
            f"<b>{ad_tx or 'AD 이소폼'}</b>으로 전환이 관측되었다"
            + (f" (DTU p = {float(dtu_p):.2e})" if dtu_p else "") + "."
        )

    # 2. Structural domain change
    if dg_list:
        gained_descs = '; '.join(
            f"<b>{d}</b> ({_domain_func(d)})" for d in dg_list
        )
        lines.append(f"AD 이소폼은 {gained_descs} 도메인을 새로 획득하여 기능적 다양성이 증가한다.")
    if dl_list:
        lost_descs = '; '.join(
            f"<b>{d}</b> ({_domain_func(d)})" for d in dl_list
        )
        lines.append(f"반면 {lost_descs} 도메인이 제거됨으로써 정상 이소폼의 주요 기능적 역량이 소실된다.")

    # 3. PRISM functional shift
    if gained_go:
        gfstr = ', '.join(f"{n[:35]} ({s:.3f})" for _, n, s in gained_go[:2])
        lines.append(
            f"PRISM GO 기능 예측에서 AD 이소폼은 정상 이소폼에는 없는 "
            f"<b>{gfstr}</b> 기능 공간을 새로 점유한다."
        )
    if lost_go:
        lfstr = ', '.join(f"{n[:35]} ({s:.3f})" for _, n, s in lost_go[:2])
        lines.append(
            f"정상 이소폼에서 높았던 <b>{lfstr}</b> 기능 점수가 AD 이소폼에서 유의미하게 낮아져, "
            f"질병 전환에 의한 기능 소실이 시사된다."
        )

    # 4. PPI
    if ppi_v == 'SUPPORTED' and ppi_p:
        ppi_score_str = f" (STRING score = {int(float(ppi_s))})" if ppi_s else ""
        lines.append(
            f"STRING PPI 분석에서 AD 이소폼은 <b>{ppi_p}</b>와의 상호작용이 예측되며"
            f"{ppi_score_str}, 이는 {ct_type} 내 새로운 단백질 복합체 형성 가능성을 시사한다."
        )

    # 5. AlphaFold
    if af_ad:
        try:
            af_val = float(af_ad)
            qual = "구조적으로 신뢰도 높은 (pLDDT ≥ 70)" if af_val >= 70 else "부분적으로 무질서한"
            lines.append(
                f"AlphaFold 구조 예측에서 AD 이소폼은 {qual} 단백질로 예측된다 (pLDDT = {af_val:.1f})."
            )
        except Exception:
            pass

    # 6. Conservation
    if phylo:
        try:
            phv = float(phylo)
            cs = ("고보존 — 100-way vertebrate alignment에서 강한 purifying selection" if phv > 1.5
                  else ("중간 보존" if phv > 0.5 else "낮은 보존 — 최근 진화적 혁신 가능성"))
            lines.append(
                f"AD 특이적 엑손의 보존성 (phyloP100way = {phv:.3f}, {cs})은 "
                f"{'이 서열의 기능적 중요성을 강하게 지지한다' if phv > 1.5 else '추가적인 기능 검증이 필요함을 시사한다'}."
            )
        except Exception:
            pass

    # 7. Regulatory mechanism
    _mech_ko = {
        'alternative_splicing': '선택적 스플라이싱 (exon inclusion/exclusion)',
        'alternative_tss':      '대체 프로모터 사용 (alternative TSS)',
        'alternative_apa':      '대체 폴리아데닐화 (alternative APA)',
        'intron_retention':     '인트론 유지 (intron retention)',
    }
    if mech:
        mech_desc = _mech_ko.get(mech, mech)
        tss_note = f" TSS 차이: {int(float(tss_bp)):+d}bp" if tss_bp else ""
        apa_note = f" APA 차이: {int(float(apa_bp)):+d}bp" if apa_bp else ""
        reg_note = f" 핵심 조절 인자: <b>{reg_name}</b>" if reg_name else ""
        lines.append(f"전사체 생성 기전: <b>{mech_desc}</b>.{tss_note}{apa_note}{reg_note}")

    # 8. NMD caveat
    if ad_nmd and str(ad_nmd).lower() not in ('false', ''):
        lines.append(
            "⚠️ AD 이소폼은 NMD (Nonsense-Mediated Decay) 감수성 구조를 포함하므로, "
            "단백질 번역 여부를 Ribo-seq 또는 질량분석으로 검증해야 한다."
        )

    # ── HTML assembly (inline styles only — no CSS classes) ──────────────────
    _TD_L = "style='padding:4px 10px;color:#6b7280;font-size:0.83rem;white-space:nowrap;vertical-align:top'"
    _TD_V = "style='padding:4px 10px;font-weight:700;font-size:0.83rem;vertical-align:top'"
    _TD_C = "style='padding:4px 10px;font-size:0.75rem;color:#9ca3af;vertical-align:top'"

    def _tag(text, bg, fg='#1e293b'):
        return (f"<code style='background:{bg};color:{fg};padding:2px 6px;"
                f"border-radius:3px;font-size:0.82rem'>{text}</code>")

    evid_rows_html = ''
    if delta:
        evid_rows_html += f"<tr><td {_TD_L}>Δ Usage (AD−CT)</td><td {_TD_V}>{float(delta):+.3f}</td><td {_TD_C}>DTU</td></tr>"
    if dtu_p:
        evid_rows_html += f"<tr><td {_TD_L}>DTU p-value</td><td {_TD_V}>{float(dtu_p):.2e}</td><td {_TD_C}>DTU</td></tr>"
    if dg_list:
        evid_rows_html += f"<tr><td {_TD_L}>도메인 획득</td><td {_TD_V}>{'&nbsp;·&nbsp;'.join(dg_list)}</td><td {_TD_C}>Structure</td></tr>"
    if dl_list:
        evid_rows_html += f"<tr><td {_TD_L}>도메인 손실</td><td {_TD_V}>{'&nbsp;·&nbsp;'.join(dl_list)}</td><td {_TD_C}>Structure</td></tr>"
    if ppi_v:
        _ppi_clr = '#15803d' if ppi_v == 'SUPPORTED' else '#b91c1c'
        evid_rows_html += f"<tr><td {_TD_L}>PPI support</td><td {_TD_V}><span style='color:{_ppi_clr}'>{ppi_v}</span></td><td {_TD_C}>Interaction</td></tr>"
    if phylo:
        evid_rows_html += f"<tr><td {_TD_L}>phyloP (AD exon)</td><td {_TD_V}>{float(phylo):.3f}&nbsp;<span style='color:#9ca3af;font-size:0.75rem'>({cons_c or '?'})</span></td><td {_TD_C}>Conservation</td></tr>"
    if mech:
        evid_rows_html += f"<tr><td {_TD_L}>기전</td><td {_TD_V}>{_mech_ko.get(mech, mech)}</td><td {_TD_C}>Regulation</td></tr>"
    if not evid_rows_html:
        evid_rows_html = f"<tr><td {_TD_L} colspan='3'>증거 데이터 없음</td></tr>"

    def _go_badges(top_list, bg, border):
        if not top_list:
            return "<span style='color:#9ca3af;font-size:0.82rem'>데이터 없음</span>"
        return ''.join(
            f"<div style='background:{bg};border-left:3px solid {border};"
            f"border-radius:4px;padding:5px 8px;margin:3px 0;font-size:0.83rem'>"
            f"<b>{n[:36]}</b>&nbsp;&nbsp;"
            f"<span style='color:#64748b'>{s:.3f}</span></div>"
            for _, n, s in top_list[:3]
        )

    domain_gained_li = ''.join(
        f"<div style='margin:4px 0;font-size:0.83rem'>"
        f"{_tag(d, '#dcfce7', '#14532d')}"
        f"<span style='color:#374151;margin-left:6px'>{_domain_func(d)}</span></div>"
        for d in dg_list
    ) or "<div style='color:#9ca3af;font-size:0.83rem;padding:4px 0'>변화 없음</div>"

    domain_lost_li = ''.join(
        f"<div style='margin:4px 0;font-size:0.83rem'>"
        f"{_tag(d, '#fee2e2', '#7f1d1d')}"
        f"<span style='color:#374151;margin-left:6px'>{_domain_func(d)}</span></div>"
        for d in dl_list
    ) or "<div style='color:#9ca3af;font-size:0.83rem;padding:4px 0'>변화 없음</div>"

    interp_html = ''.join(
        f"<p style='margin:0 0 10px 0;font-size:0.86rem;line-height:1.7;color:#1e293b'>{l}</p>"
        for l in lines
    ) or "<p style='color:#9ca3af;font-size:0.86rem'>해석 데이터 불충분</p>"

    return (
        f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;"
        f"padding:20px 22px;margin:14px 0;font-family:Arial,sans-serif'>"

        # ── Header ──
        f"<table width='100%' cellspacing='0' cellpadding='0' style='margin-bottom:14px'><tr>"
        f"<td style='vertical-align:middle'>"
        f"<span style='font-size:1.0rem;font-weight:700;color:#1e293b'>"
        f"📋 생물학적 기능 예측 리포트</span>"
        f"&nbsp;<span style='font-size:0.88rem;color:#0ea5e9;font-weight:700'>{gene}</span>"
        f"&nbsp;<span style='font-size:0.85rem;color:#64748b'>· {ct_type}</span>"
        f"</td>"
        f"<td style='text-align:right;vertical-align:middle;white-space:nowrap'>"
        f"<span style='background:{conf_color};color:white;padding:4px 14px;"
        f"border-radius:12px;font-size:0.8rem;font-weight:700'>신뢰도: {conf_label}</span>"
        f"</td></tr></table>"

        # ── Row 1: Evidence table | Domain changes ──
        f"<table width='100%' cellspacing='0' cellpadding='0' style='margin-bottom:14px'><tr>"
        f"<td width='50%' style='vertical-align:top;padding-right:12px'>"
        f"<div style='font-size:0.75rem;font-weight:700;color:#374151;text-transform:uppercase;"
        f"letter-spacing:0.5px;margin-bottom:8px'>📊 증거 요약</div>"
        f"<table width='100%' cellspacing='0' style='border-collapse:collapse'>{evid_rows_html}</table>"
        f"</td>"
        f"<td width='50%' style='vertical-align:top;padding-left:12px;"
        f"border-left:1px solid #e2e8f0'>"
        f"<div style='font-size:0.75rem;font-weight:700;color:#374151;text-transform:uppercase;"
        f"letter-spacing:0.5px;margin-bottom:8px'>🔩 도메인 기능 변화</div>"
        f"<div style='font-size:0.78rem;color:#15803d;font-weight:600;margin-bottom:4px'>▲ 획득 (AD 이소폼)</div>"
        f"{domain_gained_li}"
        f"<div style='font-size:0.78rem;color:#dc2626;font-weight:600;margin:10px 0 4px'>▼ 손실 (CT 이소폼)</div>"
        f"{domain_lost_li}"
        f"</td></tr></table>"

        # ── Row 2: CT GO | AD GO ──
        f"<table width='100%' cellspacing='0' cellpadding='0' style='margin-bottom:14px'><tr>"
        f"<td width='50%' style='vertical-align:top;padding-right:8px'>"
        f"<div style='background:#eff6ff;border-radius:6px;padding:10px 12px'>"
        f"<div style='font-size:0.78rem;font-weight:700;color:#1d4ed8;margin-bottom:6px'>"
        f"🔵 Control 이소폼 TOP GO"
        f"<span style='font-weight:400;color:#94a3b8;font-size:0.72rem;display:block'>{(ct_tx or '—')[:35]}</span>"
        f"</div>"
        f"{_go_badges(ct_top, '#dbeafe', '#3b82f6')}"
        f"</div></td>"
        f"<td width='50%' style='vertical-align:top;padding-left:8px'>"
        f"<div style='background:#fef2f2;border-radius:6px;padding:10px 12px'>"
        f"<div style='font-size:0.78rem;font-weight:700;color:#dc2626;margin-bottom:6px'>"
        f"🔴 AD 이소폼 TOP GO"
        f"<span style='font-weight:400;color:#94a3b8;font-size:0.72rem;display:block'>{(ad_tx or '—')[:35]}</span>"
        f"</div>"
        f"{_go_badges(ad_top, '#fee2e2', '#ef4444')}"
        f"</div></td>"
        f"</tr></table>"

        # ── Narrative ──
        f"<div style='background:white;border:1px solid #e2e8f0;border-radius:8px;"
        f"padding:16px 18px;margin-bottom:10px'>"
        f"<div style='font-size:0.85rem;font-weight:700;color:#1e293b;margin-bottom:12px;"
        f"padding-bottom:8px;border-bottom:2px solid #f1f5f9'>🧬 종합 해석 및 기능 예측</div>"
        f"{interp_html}"
        f"</div>"

        # ── Footer ──
        f"<div style='font-size:0.72rem;color:#9ca3af;text-align:right'>"
        f"PRISM+BISECT 자동 생성 · Lee et al. (2026) · 실험적 검증 필요</div>"
        f"</div>"
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
        # Normalise gene column name (gene_id / gene / geneID / gene_name)
        _gene_col = next(
            (c for c in _d.columns
             if c.lower() in ('gene_id', 'gene', 'geneid', 'gene_name', 'gene_symbol')),
            None,
        )
        if _gene_col is None:
            return _lk
        if _gene_col != 'gene_id':
            _d = _d.rename(columns={_gene_col: 'gene_id'})
        # Normalise delta_IF column name
        _dif_col = next(
            (c for c in _d.columns
             if c.lower() in ('delta_if', 'dif', 'deltif', 'delta_usage', 'delta')),
            None,
        )
        if _dif_col and _dif_col != 'delta_IF':
            _d = _d.rename(columns={_dif_col: 'delta_IF'})
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

                # ── Proportion chart + GO comparison + Bio report ────────────
                _ids_arr_b = np.asarray(ids, dtype=str)
                _ct_idx_b  = np.where(_ids_arr_b == _ct_tx)[0]
                _ad_idx_b  = np.where(_ids_arr_b == _ad_tx)[0]
                _ct_go_scores = sm[_ct_idx_b[0]] if len(_ct_idx_b) > 0 else None
                _ad_go_scores = sm[_ad_idx_b[0]] if len(_ad_idx_b) > 0 else None

                st.divider()

                # 1. Proportion chart — estimated isoform usage ratio CT vs AD
                st.markdown("**📊 Control vs Disease Transcript Usage**")
                if not _g_dtu.empty and (_ct_tx or _ad_tx):
                    _n_iso = len(_g_dtu)
                    _prop = _g_dtu[['isoform_id', 'delta_IF']].copy()
                    _prop['ct_frac'] = 1.0 / _n_iso
                    _prop['ad_frac'] = (_prop['ct_frac'] + _prop['delta_IF']).clip(lower=0)
                    _sum_ad = _prop['ad_frac'].sum()
                    if _sum_ad > 0:
                        _prop['ad_frac'] /= _sum_ad

                    # Collapse non-focal isoforms into 'Other'
                    _focal_set = {_ct_tx, _ad_tx}
                    _prop_rows = []
                    _other_ct = _other_ad = 0.0
                    for _, _pr in _prop.iterrows():
                        _iso = _pr['isoform_id']
                        if _iso in _focal_set:
                            _label = (f'🔵 CT: {_iso[:22]}' if _iso == _ct_tx
                                      else f'🔴 AD: {_iso[:22]}')
                            _prop_rows.append({'Condition': 'Control (CT)', 'Label': _label,
                                               'Fraction': _pr['ct_frac'], 'IsoType': _iso})
                            _prop_rows.append({'Condition': 'Disease (AD)', 'Label': _label,
                                               'Fraction': _pr['ad_frac'], 'IsoType': _iso})
                        else:
                            _other_ct += _pr['ct_frac']
                            _other_ad += _pr['ad_frac']
                    _prop_rows.append({'Condition': 'Control (CT)', 'Label': '◻ Other isoforms',
                                       'Fraction': _other_ct, 'IsoType': 'other'})
                    _prop_rows.append({'Condition': 'Disease (AD)', 'Label': '◻ Other isoforms',
                                       'Fraction': _other_ad, 'IsoType': 'other'})

                    _prop_df2 = pd.DataFrame(_prop_rows)
                    _ct_label = f'🔵 CT: {_ct_tx[:22]}'
                    _ad_label = f'🔴 AD: {_ad_tx[:22]}'
                    _color_map_prop = {
                        _ct_label:           '#3b82f6',
                        _ad_label:           '#ef4444',
                        '◻ Other isoforms':  '#cbd5e1',
                    }
                    _cat_order_prop = [_ct_label, _ad_label, '◻ Other isoforms']

                    _fig_prop = px.bar(
                        _prop_df2,
                        x='Condition', y='Fraction', color='Label',
                        barmode='stack',
                        color_discrete_map=_color_map_prop,
                        category_orders={'Label': _cat_order_prop},
                        title=f'Transcript Usage — {_gene}  ·  {_ct} (추정)',
                        labels={'Fraction': '이소폼 사용 비율 (추정)', 'Condition': ''},
                        height=310,
                    )
                    _fig_prop.update_layout(
                        plot_bgcolor='white',
                        yaxis=dict(tickformat='.0%', range=[0, 1.05], gridcolor='#f0f0f0'),
                        legend_title='이소폼',
                        margin=dict(t=38, b=30, l=10, r=10),
                        bargap=0.35,
                    )
                    _fig_prop.update_yaxes(tickformat='.0%')
                    st.caption(
                        "CT 조건 균등 baseline(1/n) + DTU delta_IF로 비율 추정. "
                        "핵심 이소폼 쌍만 강조 표시됩니다."
                    )
                    st.plotly_chart(_fig_prop, use_container_width=True,
                                    key=f"prop_{_gene}_{_safe_ct_key}")

                elif _ct_tx or _ad_tx:
                    # Fallback: DTU 데이터 없을 때 BISECT delta로 방향성 표시
                    _dv = float(_brow.get('delta') or 0)
                    _fb_rows = []
                    if _ct_tx:
                        _fb_rows.append({'이소폼': f'CT: {_ct_tx[:30]}',
                                         '역할': '🔵 CT (Control)',
                                         'Δ Usage': _dv})
                    if _ad_tx:
                        _fb_rows.append({'이소폼': f'AD: {_ad_tx[:30]}',
                                         '역할': '🔴 AD (Disease)',
                                         'Δ Usage': -_dv})
                    if _fb_rows:
                        _fb_df = pd.DataFrame(_fb_rows)
                        _fig_fb = px.bar(
                            _fb_df, x='이소폼', y='Δ Usage', color='역할',
                            color_discrete_map={
                                '🔵 CT (Control)': '#3b82f6',
                                '🔴 AD (Disease)': '#ef4444',
                            },
                            title=f'BISECT Δ Usage 방향 — {_gene} · {_ct}',
                            labels={'Δ Usage': 'Δ Usage (AD − CT)', '이소폼': ''},
                            height=260,
                        )
                        _fig_fb.add_hline(y=0, line_color='#1e293b', line_width=1.2)
                        _fig_fb.update_layout(
                            plot_bgcolor='white',
                            yaxis=dict(gridcolor='#f0f0f0'),
                            legend_title='',
                            margin=dict(t=38, b=40, l=10, r=10),
                        )
                        st.caption(
                            "DTU 상세 데이터 없음 — BISECT delta로 방향성만 표시. "
                            "Brain 조직 선택 시 전체 이소폼 비율 차트로 전환됩니다."
                        )
                        st.plotly_chart(_fig_fb, use_container_width=True,
                                        key=f"prop_fb_{_gene}_{_safe_ct_key}")

                # 2. GO function comparison chart — CT vs AD isoform
                if _ct_go_scores is not None or _ad_go_scores is not None:
                    _top_n_go = 6
                    _union_idx: set = set()
                    if _ct_go_scores is not None:
                        _union_idx |= set(np.argsort(_ct_go_scores)[-_top_n_go:].tolist())
                    if _ad_go_scores is not None:
                        _union_idx |= set(np.argsort(_ad_go_scores)[-_top_n_go:].tolist())

                    _cmp_rows = []
                    for _gi in sorted(_union_idx):
                        _gn = gnames.get(go[_gi], go[_gi])[:38]
                        if _ct_go_scores is not None:
                            _cmp_rows.append({'GO term': _gn, 'Score': float(_ct_go_scores[_gi]),
                                              'Isoform': 'CT', 'IsoLabel': f'🔵 CT ({(_ct_tx or "—")[:18]})'})
                        if _ad_go_scores is not None:
                            _cmp_rows.append({'GO term': _gn, 'Score': float(_ad_go_scores[_gi]),
                                              'Isoform': 'AD', 'IsoLabel': f'🔴 AD ({(_ad_tx or "—")[:18]})'})

                    if _cmp_rows:
                        _cmp_df = pd.DataFrame(_cmp_rows)
                        _go_order = (
                            _cmp_df.groupby('GO term')['Score'].max()
                            .sort_values(ascending=False).index.tolist()
                        )
                        _ct_iso_label = f'🔵 CT ({(_ct_tx or "—")[:18]})'
                        _ad_iso_label = f'🔴 AD ({(_ad_tx or "—")[:18]})'
                        _fig_cmp = px.bar(
                            _cmp_df, x='GO term', y='Score', color='IsoLabel',
                            barmode='group',
                            color_discrete_map={
                                _ct_iso_label: '#3b82f6',
                                _ad_iso_label: '#ef4444',
                            },
                            category_orders={
                                'GO term': _go_order,
                                'IsoLabel': [_ct_iso_label, _ad_iso_label],
                            },
                            title=f'PRISM GO Score — CT vs AD Isoform · {_gene}',
                            labels={'Score': 'PRISM Score', 'GO term': '', 'IsoLabel': '이소폼'},
                            height=330,
                        )
                        _fig_cmp.add_hline(
                            y=float(thr), line_dash='dash', line_color='#94a3b8',
                            annotation_text=f'threshold ({thr})',
                        )
                        _fig_cmp.update_layout(
                            xaxis_tickangle=-38,
                            plot_bgcolor='white',
                            yaxis=dict(range=[0, 1.05], gridcolor='#f0f0f0'),
                            legend_title='',
                            margin=dict(t=38, b=80, l=10, r=10),
                        )
                        st.markdown("**🧬 GO Function Score: CT vs AD Isoform 비교**")
                        st.caption(
                            "두 이소폼의 PRISM GO 점수 상위 항목을 합집합하여 기능 차이를 비교합니다. "
                            "점수 차이가 큰 GO term이 이소폼 스위치로 인한 기능 변화 후보입니다."
                        )
                        st.plotly_chart(_fig_cmp, use_container_width=True,
                                        key=f"go_cmp_{_gene}_{_safe_ct_key}")

                # 3. Biological prediction report
                _bio_html = _build_bio_report_html(
                    brow=_brow, gene=_gene, ct_type=_ct,
                    ct_tx=_ct_tx, ad_tx=_ad_tx,
                    ct_scores=_ct_go_scores, ad_scores=_ad_go_scores,
                    go_ids=go, go_names=gnames, threshold=thr,
                )
                st.markdown(_bio_html, unsafe_allow_html=True)

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
