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

st.set_page_config(page_title="Target Analysis — PRISM", layout="wide")
st.title("🎯 Target Analysis")
st.caption(
    "아이소폼별 GO 기능 예측 결과를 4가지 시나리오로 분류하고, "
    "BISECT 파이프라인이 검증한 질병 특이적 기능 스위치 케이스를 심층 탐색합니다."
)

with st.expander("📖 이 페이지 완전 가이드 — 처음이라면 먼저 읽어보세요", expanded=False):
    st.markdown("""
### 이 페이지에서 무엇을 할 수 있나요?

PRISM이 예측한 아이소폼별 GO(Gene Ontology) 기능 점수를 바탕으로,
어떤 아이소폼이 **질병 특이적으로 기능이 바뀌었는지** 탐색합니다.
크게 세 가지 분석 경로를 제공합니다:

---

#### 🟦 경로 1 — 시나리오별 탐색 (탭 S1~S4)

PRISM은 모든 아이소폼을 아래 4가지 시나리오로 분류합니다.

| 시나리오 | 조건 | 의미 | 우선순위 |
|----------|------|------|----------|
| 🔴 **S1: 기능 스위치** | DTU 있음 + 신규 GO 예측 | 질병에서 기능 자체가 바뀐 아이소폼 | 최우선 실험 대상 |
| 🟠 **S2: 발현 스위치** | DTU 있음 + GO 변화 없음 | 발현 비율만 바뀐 구조적 전환 | 중간 |
| 🟢 **S3: 항상 신규 기능** | DTU 없음 + 신규 GO 예측 | 조건 무관하게 새 기능을 가진 아이소폼 | 논문 주 발견 (뇌 541개) |
| ⬜ **S4: 배경** | DTU 없음 + GO 변화 없음 | 특이사항 없는 배경 아이소폼 | 낮음 |

> **S1·S2는 DTU 파일이 있어야 활성화됩니다.** 데모 모드에서는 S3·S4만 이용 가능합니다.

---

#### 🔍 경로 2 — 아이소폼 검색 (Search Isoform 탭)

특정 유전자나 아이소폼 ID를 검색하면 GO 점수 막대 차트와 케이스 리포트(Markdown)를 확인할 수 있습니다.
예: `NDUFS4`, `KIF21B`, `DLG1`

---

#### 🧫 경로 3 — BISECT Cases 심층 탐색

BISECT 파이프라인은 15개 모듈(구조·PPI·계통보존·규제 인자 등)로 각 유전자를 검증하여,
생물학적으로 의미 있는 **84개의 PASS 케이스**를 선정합니다.
각 케이스에서 확인할 수 있는 내용:

- **Volcano Plot** — 어떤 TF·ASF 인자가 AD에서 발현이 바뀌었는지 통계적으로 시각화
- **TF/ASF 활성 변화 막대 차트** — 핵심 전사·스플라이싱 인자의 logFC 방향
- **도메인 구조 변화** — 어떤 단백질 도메인이 획득/손실되었는지
- **GO 기능 비교** — CT 이소폼 vs AD 이소폼의 기능 공간 차이
- **종합 해석 리포트** — 인과 경로부터 PPI·보존성까지 통합 분석

---

#### 💡 핵심 용어 정리

| 용어 | 의미 |
|------|------|
| **GO score (0~1)** | PRISM이 예측한 GO term 해당 확률. 0.5 이상이면 유의미한 기능 예측 |
| **DTU (Δ Usage)** | 두 조건 간 아이소폼 사용 비율 차이. ±0.1 이상이면 의미있는 전환 |
| **pLDDT** | AlphaFold 구조 예측 신뢰도. 70 이상이면 구조적으로 신뢰 가능 |
| **logFC** | 조절 인자의 발현 배수 변화 (log₂). 양수=AD에서 증가, 음수=감소 |
| **phyloP** | 척추동물 100종 보존도. 1.5 이상이면 강한 purifying selection |
| **-log₁₀(p-adj)** | 보정된 p-값의 -log₁₀ 변환. 2 이상이면 p < 0.01에 해당 |

> Score 임계값은 사이드바 슬라이더에서 조절 가능합니다 (기본값 0.5).
    """)

# ── Data ─────────────────────────────────────────────────────────────────────
cfg = st.session_state.get('cfg', {})
if 'analysis_step' not in st.session_state: st.session_state['analysis_step'] = {}
st.session_state['analysis_step']['targets'] = True
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
    1: ("🔴 **Scenario 1 — 기능 스위치 (Functional Switch)**\n\n"
        "DTU(Differential Transcript Usage) 분석에서 통계적으로 유의미한 아이소폼 비율 변화가 확인되고, "
        "PRISM 예측에서 기존 주석에 없는 **신규 GO 기능**이 검출된 케이스입니다. "
        "두 조건이 동시에 충족되므로 질병 특이적 기능 변화 후보 중 **최우선 실험 검증 대상**입니다."),
    2: ("🟠 **Scenario 2 — 발현 스위치 (Expression Switch)**\n\n"
        "DTU에서 아이소폼 비율이 유의미하게 달라졌지만, GO 기능 점수는 조건 간 차이가 작습니다. "
        "단백질 서열·도메인 구성이 달라졌을 가능성이 있으나, "
        "현재 GO term 범위 내에서 기능 변화는 감지되지 않았습니다. "
        "도메인 수준 분석(BISECT) 또는 확장된 GO term 세트로 추가 검증을 권장합니다."),
    3: ("🟢 **Scenario 3 — 항상 신규 기능 (Constitutive Novel Function)**\n\n"
        "두 조건 간 아이소폼 비율 차이(DTU)는 없지만, PRISM이 기존 주석에 없는 **신규 GO 기능**을 높은 점수로 예측합니다. "
        "조건과 무관하게 항상 발현되는 기능적으로 독특한 아이소폼입니다. "
        "본 연구의 뇌 데이터에서 **541개의 novel isoform**이 이 카테고리에 해당하며, "
        "여러 세포 유형에서 반복 확인된 케이스일수록 신뢰도가 높습니다."),
    4: ("⬜ **Scenario 4 — 배경 (Background)**\n\n"
        "DTU와 신규 GO 예측 모두 임계값 미달입니다. "
        "현재 설정(score > threshold, DTU p < 0.05)으로는 특이사항이 없는 배경 아이소폼입니다. "
        "임계값을 낮추거나, 더 넓은 GO term 세트를 사용하면 일부가 S1~S3으로 전환될 수 있습니다."),
}


def _render_scenario_table(scenario_id: int) -> None:
    st.markdown(SCENARIO_DESCS[scenario_id])

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
                데모 데이터는 단일 조건이므로 DTU를 계산할 수 없습니다.
                <b>Scenario 3 (신규 기능)</b>은 DTU 없이도 바로 분석 가능합니다.
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info(
                f"현재 설정(GO score 임계값 > {thr})에서 이 시나리오에 해당하는 아이소폼이 없습니다. "
                "사이드바에서 임계값을 낮추면 더 많은 후보가 나타납니다.",
                icon="ℹ️",
            )
        return

    _c1, _c2, _c3 = st.columns(3)
    _c1.metric(
        "후보 아이소폼 수",
        f"{len(cands):,}",
        help="현재 GO score 임계값 기준으로 이 시나리오에 해당하는 아이소폼 총수",
    )
    _c2.metric(
        "관련 유전자 수",
        f"{cands['gene_id'].nunique():,}" if 'gene_id' in cands.columns else "—",
        help="이 시나리오에 아이소폼이 하나 이상 속한 고유 유전자 수",
    )
    _c3.metric(
        "최고 GO 점수",
        f"{cands['max_score'].max():.3f}" if not cands.empty else "—",
        help="이 시나리오 후보 중 가장 높은 PRISM GO 예측 점수",
    )

    disp = cands[['isoform_id', 'gene_id', 'max_score', 'max_go', 'n_high_go',
                  'novel_go_terms', 'dtu_pvalue']].copy()
    disp['max_go'] = disp['max_go'].map(lambda g: f"{g}: {gnames.get(g,'')[:35]}")
    disp = disp.rename(columns={
        'isoform_id':    '아이소폼 ID',
        'gene_id':       '유전자',
        'max_score':     '최고 GO 점수',
        'max_go':        '최고 기능 (GO)',
        'n_high_go':     f'GO ≥{thr} 개수',
        'novel_go_terms':'신규 GO (주석 외)',
        'dtu_pvalue':    'DTU p-value',
    })
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption(
        f"**최고 GO 점수**: PRISM이 예측한 GO term 중 가장 높은 확률값 (1.0에 가까울수록 확신도 높음) | "
        f"**GO ≥{thr} 개수**: 임계값을 넘는 GO term 수 | "
        "**신규 GO**: 기존 유전자 주석에 없는 새로운 기능 예측 GO term"
    )

    # Download button
    csv = cands.to_csv(index=False).encode('utf-8')
    st.download_button(
        f"📥 Scenario {scenario_id} 후보 목록 다운로드 (CSV)",
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
    # Pre-populate from sidebar persistent search or hub basket
    _default_query = st.session_state.get('search_gene', '')
    if _default_query and not st.session_state.get('_targets_query_loaded'):
        st.session_state['_targets_query_loaded'] = True
        st.session_state['targets_search_key'] = _default_query

    query = st.text_input(
        "Search by isoform ID or gene name",
        value=st.session_state.get('targets_search_key', _default_query),
        placeholder="e.g. NDUFS4-201, KIF21B, tr319500",
        key='targets_gene_input',
    )
    # Write back to shared session state so sidebar reflects current query
    if query:
        st.session_state['search_gene'] = query

    # ── Module × DTU gene-level view ──────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _load_module_dtu_data():
        """Load pre-computed module assignments and DTU data for module×DTU view."""
        import json
        mod_path = Path(__file__).parents[3] / 'reports' / 'brain_isoform_modules.tsv'
        dtu_path = Path(__file__).parents[2] / 'data' / 'demo' / 'brain_dtu.tsv'
        mod_j    = Path(__file__).parents[3] / 'reports' / 'brain_go_modules_672.json'

        df_mod = pd.read_csv(mod_path, sep='\t') if mod_path.exists() else None
        df_dtu = pd.read_csv(dtu_path, sep='\t') if dtu_path.exists() else None
        modules_dict = json.loads(mod_j.read_text())['modules'] if mod_j.exists() else {}
        return df_mod, df_dtu, modules_dict

    def _render_gene_module_dtu(gene_query: str) -> None:
        """Show module assignment + DTU heatmap for all isoforms of a gene."""
        df_mod, df_dtu, mod_dict = _load_module_dtu_data()
        if df_mod is None:
            return

        gene_mod = df_mod[df_mod['gene'].str.upper() == gene_query.upper()]
        if gene_mod.empty:
            return

        st.markdown("---")
        st.markdown(f"### 🧩 기능 모듈 × 조건 분석 — *{gene_query.upper()}*")
        st.caption(
            "각 아이소폼의 모듈 배정(PRISM 672-term)과 DTU(AD vs CT) 이벤트를 통합 시각화합니다. "
            "모듈이 다른 아이소폼 간 dIF 방향 변화 = 기능 스위치 신호."
        )

        col_mod, col_dtu = st.columns([1, 1])

        with col_mod:
            st.markdown("**모듈 배정 (module_score 기준)**")
            gene_mod_sorted = gene_mod.sort_values('module_score', ascending=False).head(15)
            gene_mod_sorted['mod_label'] = gene_mod_sorted['primary_module'].apply(
                lambda m: f"M{int(m)}: {mod_dict.get(str(int(m)),{}).get('label','').split('/')[0].strip()[:30]}"
            )
            _type_colors = {'known': '#2196F3', 'nic': '#FF9800', 'nnic': '#E91E63'}
            gene_mod_sorted['color'] = gene_mod_sorted['type'].map(_type_colors).fillna('#9E9E9E')

            fig_mod = px.bar(
                gene_mod_sorted,
                x='module_score', y='isoform_id',
                color='type',
                color_discrete_map=_type_colors,
                orientation='h',
                text='mod_label',
                labels={'module_score': 'Module score', 'isoform_id': ''},
                height=max(280, len(gene_mod_sorted) * 36),
            )
            fig_mod.update_traces(textposition='inside', textfont=dict(size=9))
            fig_mod.add_vline(x=0.3, line_dash='dash', line_color='red',
                              annotation_text='고신뢰도', annotation_font_size=9)
            fig_mod.update_layout(
                margin=dict(l=10, r=10, t=10, b=30),
                legend=dict(title='Type', orientation='h', y=1.02),
                yaxis=dict(autorange='reversed'),
            )
            st.plotly_chart(fig_mod, use_container_width=True, key=f"mod_bar_{gene_query}")
            st.caption("파랑=Known · 주황=NIC · 분홍=NNIC. 막대 안 텍스트 = 배정 모듈명.")

        with col_dtu:
            st.markdown("**DTU 이벤트 (|dIF| > 0.05, p < 0.1)**")

            _dtu_source = cfg.get('dtu_df') if cfg.get('dtu_df') is not None else df_dtu
            if _dtu_source is None:
                st.info("DTU 데이터 없음.")
            else:
                # Normalize column names
                _dtu = _dtu_source.copy()
                for _old, _new in [('dIF','delta_IF'),('padj','pvalue'),('p_adj','pvalue')]:
                    if _old in _dtu.columns and _new not in _dtu.columns:
                        _dtu = _dtu.rename(columns={_old: _new})

                gene_dtu = _dtu[
                    _dtu['isoform_id'].str.upper().str.startswith(gene_query.upper()) |
                    (_dtu.get('gene_id', pd.Series(dtype=str)).str.upper() == gene_query.upper()
                     if 'gene_id' in _dtu.columns else False)
                ].copy()

                if gene_dtu.empty:
                    st.info(f"DTU 데이터에서 {gene_query} isoform을 찾을 수 없습니다.")
                else:
                    # Merge with module info
                    gene_dtu = gene_dtu.merge(
                        gene_mod[['isoform_id', 'primary_module', 'type']],
                        on='isoform_id', how='left'
                    )
                    gene_dtu['mod_label'] = gene_dtu['primary_module'].apply(
                        lambda m: f"M{int(m)}" if pd.notna(m) else '?'
                    )
                    gene_dtu['sig'] = (
                        gene_dtu['delta_IF'].abs() > 0.05
                    ) & (gene_dtu['pvalue'] < 0.1 if 'pvalue' in gene_dtu.columns else True)
                    gene_dtu_sig = gene_dtu[gene_dtu['sig']].copy()

                    if gene_dtu_sig.empty:
                        st.info("유의한 DTU 이벤트 없음 (|dIF|>0.05, p<0.1).")
                    else:
                        # Shorten isoform ID for display
                        gene_dtu_sig['iso_short'] = gene_dtu_sig['isoform_id'].apply(
                            lambda x: x if len(x) <= 20 else x[:8]+'…'+x[-6:]
                        )
                        has_cond = 'condition' in gene_dtu_sig.columns

                        if has_cond:
                            # Heatmap: isoforms × conditions
                            pivot = gene_dtu_sig.pivot_table(
                                index='iso_short', columns='condition',
                                values='delta_IF', aggfunc='mean'
                            ).fillna(0)
                            annot_mod = gene_dtu_sig.groupby('iso_short')['mod_label'].first()

                            fig_heat = px.imshow(
                                pivot,
                                color_continuous_scale='RdBu_r',
                                color_continuous_midpoint=0,
                                zmin=-0.5, zmax=0.5,
                                labels={'x': '조건 (cell type)', 'y': '아이소폼', 'color': 'dIF'},
                                height=max(280, len(pivot) * 40),
                                aspect='auto',
                            )
                            # Annotate module assignments on y-axis
                            for i, iso in enumerate(pivot.index):
                                ml = annot_mod.get(iso, '')
                                fig_heat.add_annotation(
                                    x=-0.5, y=i,
                                    text=ml, showarrow=False,
                                    xref='x', yref='y',
                                    xanchor='right', font=dict(size=8, color='#555'),
                                )
                            fig_heat.update_layout(
                                margin=dict(l=60, r=10, t=10, b=60),
                                xaxis_tickangle=-35,
                                coloraxis_colorbar=dict(title='dIF', len=0.6),
                            )
                            st.plotly_chart(fig_heat, use_container_width=True,
                                            key=f"dtu_heat_{gene_query}")
                            st.caption(
                                "빨강(dIF > 0): AD에서 해당 isoform 사용 증가. "
                                "파랑(dIF < 0): 감소. "
                                "왼쪽 레이블 = 배정 모듈(M번호)."
                            )
                        else:
                            fig_dtu_bar = px.bar(
                                gene_dtu_sig.sort_values('delta_IF'),
                                x='delta_IF', y='iso_short', orientation='h',
                                color='delta_IF',
                                color_continuous_scale='RdBu_r',
                                color_continuous_midpoint=0,
                                text='mod_label',
                                labels={'delta_IF': 'dIF (AD−CT)', 'iso_short': ''},
                                height=max(280, len(gene_dtu_sig) * 36),
                            )
                            fig_dtu_bar.update_traces(textposition='inside', textfont=dict(size=9))
                            fig_dtu_bar.add_vline(x=0, line_color='grey', line_width=1)
                            fig_dtu_bar.update_layout(margin=dict(l=10, r=10, t=10, b=30))
                            st.plotly_chart(fig_dtu_bar, use_container_width=True,
                                            key=f"dtu_bar_{gene_query}")

                        # Key switch callout
                        switch_isos = gene_dtu_sig.copy()
                        if 'condition' in switch_isos.columns:
                            switch_isos = switch_isos.groupby('isoform_id').agg(
                                mean_dIF=('delta_IF','mean'),
                                mod_label=('mod_label','first'),
                                type=('type','first'),
                            ).reset_index()
                        gain_isos = switch_isos[switch_isos['mean_dIF' if 'mean_dIF' in switch_isos.columns else 'delta_IF'] > 0.1]
                        loss_isos = switch_isos[switch_isos['mean_dIF' if 'mean_dIF' in switch_isos.columns else 'delta_IF'] < -0.1]

                        if len(gain_isos) > 0 and len(loss_isos) > 0:
                            gain_mods = gain_isos['mod_label'].unique().tolist()
                            loss_mods = loss_isos['mod_label'].unique().tolist()
                            if set(gain_mods) != set(loss_mods):
                                st.warning(
                                    f"**⚡ 모듈 간 기능 스위치 감지**: "
                                    f"GAIN 모듈 {gain_mods} ↔ LOSS 모듈 {loss_mods} — "
                                    f"서로 다른 기능 영역 간 isoform 교환."
                                )

        # Detailed table
        with st.expander("상세 데이터 테이블"):
            merged_detail = gene_mod.merge(
                (_dtu_source[_dtu_source['isoform_id'].str.upper().str.startswith(gene_query.upper())]
                 if _dtu_source is not None else pd.DataFrame()),
                on='isoform_id', how='left'
            ) if df_dtu is not None or cfg.get('dtu_df') is not None else gene_mod

            show_cols = [c for c in ['isoform_id','type','primary_module','module_label',
                                      'module_score','condition','delta_IF','pvalue']
                         if c in merged_detail.columns]
            st.dataframe(merged_detail[show_cols].sort_values('module_score', ascending=False),
                         use_container_width=True, hide_index=True)

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

            # Gene-level module × DTU view (shown when query matches a gene name)
            _is_gene_query = (
                genes is not None and
                any(query.upper() == str(g).upper() for g in np.asarray(genes, dtype=str))
            )
            if _is_gene_query or (len(hits) > 1 and not '-' in query):
                _render_gene_module_dtu(query)

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

# ── Regulatory knowledge base (TF / ASF / Epigenetic) ────────────────────────
# category: 'TF' | 'ASF' | 'Epigenetic' | 'RBP'
# known: True = established AD/disease literature; False = newly observed in BISECT
_REGULATOR_KB: dict = {
    'STAT1':   ('TF',         True,  'AD 신경염증 핵심 전사인자; 미세아교·흥분성 뉴런에서 억제됨 (Baranzini 2020)'),
    'REST':    ('TF',         True,  '신경보호 전사억제인자; AD에서 발현 감소 → 시냅스 유전자 억제 해제 (Lu 2014 Cell)'),
    'CREB1':   ('TF',         True,  '신경 생존·LTP 전사인자; AD에서 인산화 감소 → 기억 형성 장애 (Saura 2004)'),
    'SP1':     ('TF',         True,  'Tau·APP 프로모터에 직접 결합; AD 취약성 인자 (Citron 2008)'),
    'SP3':     ('TF',         True,  'SP1 길항 전사인자; AD에서 SP1 대비 과발현 → 프로모터 경쟁 (Black 2001)'),
    'SRSF5':   ('ASF',        True,  'Serine/Arginine Splicing Factor 5; AD 관련 스플라이싱 재편 (Raj 2018)'),
    'SRSF7':   ('ASF',        True,  'tau exon 10 포함 조절; FTLD-Tau 관련 (Jiang 1998)'),
    'RBFOX1':  ('ASF',        True,  '뇌 특이적 ASF; 신경 발달·AD 취약 exon 조절 (Bhatt 2020)'),
    'HDAC2':   ('Epigenetic', True,  'AD에서 히스톤 H3K27 탈아세틸화 과활성 → 신경 유전자 억제 (Gräff 2012)'),
    'SIRT1':   ('Epigenetic', True,  'AD에서 NAD+-의존 탈아세틸화 감소 → p53·NF-κB 과활성 (Kim 2007)'),
    'KLF9':    ('TF',         False, '새로 발견; 억제성 전사인자 후보, 산화 스트레스 반응 조절'),
    'YBX1':    ('RBP',        False, 'Y-box RNA 결합 단백질; 스플라이싱·번역 조절, AD 역할 미확립'),
    'HNRNPK':  ('ASF',        False, 'hnRNP K; pre-mRNA 스플라이싱·수송 조절, AD 연관 신규 발견'),
    'E2F3':    ('TF',         False, '세포주기·아포프토시스 전사인자; AD 신경세포 재진입 관련 가능성'),
    'SETDB2':  ('Epigenetic', False, 'H3K9me3 methyltransferase; 이형성질 억제 → 비정상 유전자 발현'),
}

_MECHANISM_KO: dict = {
    'alternative_promoter':   ('대체 프로모터', '#7c3aed',
                                '다른 프로모터 활성화로 전사 시작 위치가 이동. '
                                'N-말단 구조가 달라져 신호 펩타이드·막 결합 도메인 변화 가능.'),
    'alternative_splicing':   ('선택적 스플라이싱', '#0ea5e9',
                                'exon inclusion/exclusion으로 도메인 구성이 직접 변화. '
                                'ASF(SRSF, RBFOX 등)의 결합 부위 변화가 주요 원인.'),
    'transcriptional':        ('전사 조절 변화', '#d97706',
                                '동일 프로모터에서 TF 결합 변화로 전사량이 조절됨. '
                                'TF 활성 변화가 아이소폼 비율 변화의 직접 원인.'),
    'epigenetic_derepression': ('후성유전학적 탈억제', '#dc2626',
                                'HDAC 과활성 또는 DNA 메틸화 변화로 억제되어 있던 엑손이 개방됨. '
                                '염색질 접근성 변화가 스플라이싱 패턴을 재편함.'),
    'intron_retention':       ('인트론 유지', '#059669',
                                '스플라이싱 효율 저하로 인트론이 성숙 mRNA에 잔존. '
                                'NMD 위험 증가; 단백질 번역 여부 검증 필요.'),
}


def _parse_regulators(raw: str) -> list:
    """Parse BISECT top_regulators string → list of dicts."""
    import ast
    result = []
    if not raw or str(raw) in ('None', ''):
        return result
    for p in str(raw).split(';'):
        p = p.strip()
        if not p:
            continue
        try:
            result.append(ast.literal_eval(p))
        except Exception:
            pass
    return result


@st.cache_data(show_spinner=False)
def _load_case_sig_regs(gene: str, cell_type: str) -> list:
    """Load significant_regulators from case analysis.json (up to 14 per case)."""
    import json as _json
    _base = Path(__file__).parents[3] / 'Final_analysis' / 'pipeline_bioanalysis' / 'outputs'
    _aj = _base / f"{gene}_{cell_type}" / 'analysis.json'
    if not _aj.exists():
        return []
    try:
        with open(_aj) as _f:
            _d = _json.load(_f)
        _m8 = _d.get('m8_regulatory_context', {}) or {}
        return _m8.get('significant_regulators', []) or []
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _load_all_sig_regulators(bisect_path: str) -> list:
    """Load all significant_regulators from every BISECT case analysis.json."""
    import json as _json
    _base = Path(__file__).parents[3] / 'Final_analysis' / 'pipeline_bioanalysis' / 'outputs'
    try:
        with open(bisect_path) as _f:
            _cases = _json.load(_f)
    except Exception:
        return []
    _rows = []
    for _c in _cases:
        _g  = _c.get('gene', '')
        _ct = _c.get('cell_type', '')
        _aj = _base / f"{_g}_{_ct}" / 'analysis.json'
        if not _aj.exists():
            continue
        try:
            with open(_aj) as _f:
                _d = _json.load(_f)
            _m8 = _d.get('m8_regulatory_context', {}) or {}
            for _r in (_m8.get('significant_regulators', []) or []):
                _rows.append({
                    'Gene':         _r.get('gene', ''),
                    'logFC':        float(_r.get('logFC', 0)),
                    '-log10(padj)': float(_r.get('neg_log10_padj', 0)),
                    'Direction':    _r.get('direction', '').capitalize(),
                    'Case':         _g,
                    'CellType':     _ct,
                })
        except Exception:
            pass
    return _rows


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
    ad_nmd  = brow.get('ad_nmd')
    ct_nmd  = brow.get('ct_nmd')
    af_ad   = brow.get('af_ad_plddt_mean')
    af_ct   = brow.get('af_ct_plddt_mean')
    af_delta= brow.get('af_delta_plddt')

    # Parse regulators using shared helper
    all_regs  = _parse_regulators(str(brow.get('top_regulators') or ''))
    reg_name  = all_regs[0].get('gene', '') if all_regs else ''

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
        bool(all_regs),                                    # regulatory evidence
        bool(mech and mech != 'transcriptional'),          # mechanism specificity
    ])
    conf_label = ['Low', 'Low', 'Moderate', 'Moderate', 'High', 'High', 'Very High',
                  'Very High', 'Very High'][min(ev_count, 8)]
    conf_color = {'Low': '#ef4444', 'Moderate': '#f59e0b',
                  'High': '#22c55e', 'Very High': '#15803d'}[conf_label]

    # ── Regulatory context ────────────────────────────────────────────────────
    known_regs  = [r for r in all_regs if _REGULATOR_KB.get(r['gene'], (None, None))[1] is True]
    novel_regs  = [r for r in all_regs if _REGULATOR_KB.get(r['gene'], (None, None))[1] is False]
    mech_info   = _MECHANISM_KO.get(mech, ('', '#64748b', ''))

    # ── Narrative sentences ───────────────────────────────────────────────────
    lines = []

    # 0. Causal origin (upstream mechanism)
    if mech and all_regs:
        mech_ko = mech_info[0] or mech
        top_reg = all_regs[0]
        top_reg_name = top_reg.get('gene', '')
        top_dir  = '활성 증가' if top_reg.get('direction') == 'up' else '억제'
        top_lfc  = top_reg.get('logFC', 0)
        kb_desc  = _REGULATOR_KB.get(top_reg_name, ('', '', ''))[2]
        lines.append(
            f"이 아이소폼 전환의 상류 원인으로 <b>{mech_ko}</b> 기전이 예측된다. "
            f"핵심 조절 인자 <b>{top_reg_name}</b> (logFC = {float(top_lfc):+.3f}, AD에서 {top_dir})"
            + (f" — {kb_desc}" if kb_desc else "") + "."
        )

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
        gained_descs = '; '.join(f"<b>{d}</b> ({_domain_func(d)})" for d in dg_list)
        lines.append(f"AD 이소폼은 {gained_descs} 도메인을 새로 획득하여 기능적 다양성이 증가한다.")
    if dl_list:
        lost_descs = '; '.join(f"<b>{d}</b> ({_domain_func(d)})" for d in dl_list)
        lines.append(f"반면 {lost_descs} 도메인이 제거됨으로써 정상 이소폼의 주요 기능적 역량이 소실된다.")

    # 3. Structural stability (AlphaFold)
    if af_ad and af_ct:
        try:
            af_a = float(af_ad)
            af_c = float(af_ct)
            af_d = float(af_delta) if af_delta else af_a - af_c
            q_ad = "고신뢰 구조 (pLDDT ≥ 70)" if af_a >= 70 else "부분 무질서 구조 (pLDDT < 70)"
            q_ct = "고신뢰 구조" if af_c >= 70 else "무질서 포함"
            stab_interp = (
                "AD 이소폼이 CT 이소폼보다 더 안정된 구조를 형성한다" if af_d > 5
                else ("CT 이소폼이 구조적으로 더 안정적이며 AD 이소폼은 무질서 증가" if af_d < -5
                      else "두 이소폼의 구조적 안정성이 유사하다")
            )
            lines.append(
                f"AlphaFold 구조 예측: CT 이소폼 pLDDT = {af_c:.1f} ({q_ct}), "
                f"AD 이소폼 pLDDT = {af_a:.1f} ({q_ad}), ΔpLDDT = {af_d:+.1f}. "
                f"{stab_interp}."
            )
        except Exception:
            pass
    elif af_ad:
        try:
            af_val = float(af_ad)
            qual = "구조적으로 신뢰도 높은 (pLDDT ≥ 70)" if af_val >= 70 else "부분적으로 무질서한"
            lines.append(
                f"AlphaFold 구조 예측에서 AD 이소폼은 {qual} 단백질로 예측된다 (pLDDT = {af_val:.1f})."
            )
        except Exception:
            pass

    # 4. PRISM functional shift
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

    # 5. PPI
    if ppi_v == 'SUPPORTED' and ppi_p:
        ppi_score_str = f" (STRING score = {int(float(ppi_s))})" if ppi_s else ""
        lines.append(
            f"STRING PPI 분석에서 AD 이소폼은 <b>{ppi_p}</b>와의 상호작용이 예측되며"
            f"{ppi_score_str}, 이는 {ct_type} 내 새로운 단백질 복합체 형성 가능성을 시사한다."
        )

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

    # 7. Regulatory mechanism (upgraded with KB descriptions)
    if mech:
        mech_ko_n = mech_info[0] or mech
        mech_detail = mech_info[2]
        tss_note = f" TSS 차이: {int(float(tss_bp)):+d}bp" if tss_bp else ""
        apa_note = f" APA 차이: {int(float(apa_bp)):+d}bp" if apa_bp else ""
        reg_note = f" 핵심 조절 인자: <b>{reg_name}</b>" if reg_name else ""
        lines.append(
            f"전사체 생성 기전: <b>{mech_ko_n}</b>.{tss_note}{apa_note}{reg_note} "
            + (f"— {mech_detail}" if mech_detail else "")
        )

    # 8. TF/ASF regulatory interpretation
    if known_regs:
        k_str = '; '.join(
            f"<b>{r['gene']}</b> ({r['direction']}, logFC={float(r.get('logFC',0)):+.3f})"
            for r in known_regs[:3]
        )
        lines.append(
            f"기존 AD 연관 전사·스플라이싱 인자의 활성 변화: {k_str}. "
            "이 인자들의 발현 변화가 해당 유전자좌의 아이소폼 전환을 직접 유도했을 가능성이 높다."
        )
    if novel_regs:
        n_str = '; '.join(
            f"<b>{r['gene']}</b> ({r['direction']}, logFC={float(r.get('logFC',0)):+.3f})"
            for r in novel_regs
        )
        kb_descs = '; '.join(
            _REGULATOR_KB.get(r['gene'], ('', '', ''))[2]
            for r in novel_regs if _REGULATOR_KB.get(r['gene'], ('', '', ''))[2]
        )
        lines.append(
            f"새로 발견된 조절 인자 후보: {n_str}. "
            + (f"이 인자들의 AD 특이적 역할은 아직 확립되지 않았으나 ({kb_descs}), "
               "현 데이터에서 통계적으로 유의미한 발현 변화가 관측된다." if kb_descs else "")
        )

    # 9. NMD caveat
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
        evid_rows_html += f"<tr><td {_TD_L}>기전</td><td {_TD_V}>{mech_info[0] or mech}</td><td {_TD_C}>Regulation</td></tr>"
    if all_regs:
        _reg_short = ', '.join(
            f"{r['gene']}({'↑' if r.get('direction')=='up' else '↓'})"
            for r in all_regs[:3]
        )
        evid_rows_html += f"<tr><td {_TD_L}>TF / ASF</td><td {_TD_V} style='font-size:0.78rem'>{_reg_short}</td><td {_TD_C}>Regulator</td></tr>"
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

    # ── Regulatory origin HTML block ──────────────────────────────────────────
    def _reg_badge(r):
        g = r.get('gene', '?')
        d = r.get('direction', '')
        lfc = float(r.get('logFC', 0))
        neg_p = float(r.get('neg_log10_padj', 0))
        kb = _REGULATOR_KB.get(g, ('TF', None, ''))
        cat   = kb[0] or 'TF'
        known = kb[1]
        bg    = '#fee2e2' if d == 'down' else '#dcfce7'
        border= '#ef4444' if d == 'down' else '#22c55e'
        arrow = '↓' if d == 'down' else '↑'
        star  = '' if known else ' 🟠'
        return (
            f"<div style='background:{bg};border-left:3px solid {border};"
            f"border-radius:4px;padding:5px 10px;margin:3px 0;font-size:0.82rem'>"
            f"<b>{g}</b>{star}&nbsp;"
            f"<span style='color:#64748b;font-size:0.75rem'>[{cat}]</span>&nbsp;"
            f"<span style='font-weight:700'>{arrow} {lfc:+.3f}</span>&nbsp;"
            f"<span style='color:#9ca3af;font-size:0.72rem'>-log10p={neg_p:.1f}</span>"
            f"</div>"
        )

    reg_badges_html = ''.join(_reg_badge(r) for r in all_regs[:5])
    if not reg_badges_html:
        reg_badges_html = "<div style='color:#9ca3af;font-size:0.82rem'>조절 인자 데이터 없음</div>"

    mech_ko_label = mech_info[0] or mech or '—'
    mech_clr      = mech_info[1]

    # Causal pathway arrow (upstream → downstream)
    _pathway_steps = []
    if mech:
        _pathway_steps.append(f"<b style='color:{mech_clr}'>{mech_ko_label}</b>")
    if all_regs:
        _regs_short = ', '.join(r['gene'] for r in all_regs[:3])
        _pathway_steps.append(f"TF/ASF 활성 변화 ({_regs_short})")
    if tss_cls and tss_cls not in ('same_promoter', ''):
        _pathway_steps.append(f"전사 시작 위치 이동 ({tss_cls})")
    if apa_cls and apa_cls not in ('same_apa', ''):
        _pathway_steps.append(f"3′ 처리 변화 ({apa_cls})")
    _pathway_steps.append("아이소폼 비율 전환 (DTU)")
    if dg_list or dl_list:
        _pathway_steps.append("도메인 구성 변화")
    if gained_go or lost_go:
        _pathway_steps.append("GO 기능 공간 재편")
    pathway_html = " &rarr; ".join(
        f"<span style='background:#f1f5f9;padding:2px 6px;border-radius:3px;"
        f"font-size:0.78rem'>{s}</span>"
        for s in _pathway_steps
    )

    reg_origin_html = (
        f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;"
        f"padding:14px 16px;margin-bottom:14px'>"
        f"<div style='font-size:0.75rem;font-weight:700;color:#374151;text-transform:uppercase;"
        f"letter-spacing:0.5px;margin-bottom:10px'>🔭 아이소폼 전환 인과 경로</div>"
        # Pathway arrows
        f"<div style='margin-bottom:10px;line-height:2'>{pathway_html}</div>"
        # Two-column: regulators | mechanism details
        f"<table width='100%' cellspacing='0' cellpadding='0'><tr>"
        f"<td width='50%' style='vertical-align:top;padding-right:10px'>"
        f"<div style='font-size:0.75rem;color:#374151;font-weight:600;margin-bottom:4px'>"
        f"TF / ASF 활성 변화 (AD vs CT)</div>"
        f"{reg_badges_html}"
        f"<div style='font-size:0.7rem;color:#9ca3af;margin-top:4px'>"
        f"🟠 = 새로 발견된 인자 · ↑/↓ = AD에서 증가/감소</div>"
        f"</td>"
        f"<td width='50%' style='vertical-align:top;padding-left:10px;"
        f"border-left:1px solid #d1fae5'>"
        f"<div style='font-size:0.75rem;color:#374151;font-weight:600;margin-bottom:4px'>"
        f"프로모터 · APA 컨텍스트</div>"
        + (
            f"<div style='font-size:0.82rem;margin:2px 0'>"
            f"TSS: <b>{tss_cls or '—'}</b>"
            + (f" ({int(float(tss_bp)):+d}bp)" if tss_bp else "") + "</div>"
            if tss_cls else ""
        )
        + (
            f"<div style='font-size:0.82rem;margin:2px 0'>"
            f"APA: <b>{apa_cls or '—'}</b>"
            + (f" ({int(float(apa_bp)):+d}bp)" if apa_bp else "") + "</div>"
            if apa_cls else ""
        )
        + (
            f"<div style='font-size:0.82rem;margin:6px 0 2px;color:#7c3aed'>"
            f"기전: <b>{mech_ko_label}</b></div>"
            f"<div style='font-size:0.75rem;color:#6b7280'>{mech_info[2]}</div>"
            if mech else ""
        )
        + f"</td></tr></table>"
        f"</div>"
    )

    return (
        f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;"
        f"padding:20px 22px;margin:14px 0;font-family:Arial,sans-serif'>"

        # ── Header ──
        f"<table width='100%' cellspacing='0' cellpadding='0' style='margin-bottom:14px'><tr>"
        f"<td style='vertical-align:middle'>"
        f"<span style='font-size:1.0rem;font-weight:700;color:#1e293b'>"
        f"📋 생물학적 기능 예측 리포트 — 통합 분석</span>"
        f"&nbsp;<span style='font-size:0.88rem;color:#0ea5e9;font-weight:700'>{gene}</span>"
        f"&nbsp;<span style='font-size:0.85rem;color:#64748b'>· {ct_type}</span>"
        f"</td>"
        f"<td style='text-align:right;vertical-align:middle;white-space:nowrap'>"
        f"<span style='background:{conf_color};color:white;padding:4px 14px;"
        f"border-radius:12px;font-size:0.8rem;font-weight:700'>신뢰도: {conf_label}</span>"
        f"</td></tr></table>"

        # ── Regulatory origin (causal pathway) ──
        + reg_origin_html

        # ── Row 1: Evidence table | Domain changes ──
        + f"<table width='100%' cellspacing='0' cellpadding='0' style='margin-bottom:14px'><tr>"
        f"<td width='50%' style='vertical-align:top;padding-right:12px'>"
        f"<div style='font-size:0.75rem;font-weight:700;color:#374151;text-transform:uppercase;"
        f"letter-spacing:0.5px;margin-bottom:8px'>📊 증거 요약</div>"
        f"<table width='100%' cellspacing='0' style='border-collapse:collapse'>{evid_rows_html}</table>"
        f"</td>"
        f"<td width='50%' style='vertical-align:top;padding-left:12px;"
        f"border-left:1px solid #e2e8f0'>"
        f"<div style='font-size:0.75rem;font-weight:700;color:#374151;text-transform:uppercase;"
        f"letter-spacing:0.5px;margin-bottom:8px'>🔩 도메인·구조 기능 변화</div>"
        f"<div style='font-size:0.78rem;color:#15803d;font-weight:600;margin-bottom:4px'>▲ 획득 (AD 이소폼)</div>"
        f"{domain_gained_li}"
        f"<div style='font-size:0.78rem;color:#dc2626;font-weight:600;margin:10px 0 4px'>▼ 손실 (CT 이소폼)</div>"
        f"{domain_lost_li}"
        + (
            f"<div style='font-size:0.78rem;color:#7e22ce;margin-top:8px'>"
            f"ΔpLDDT = {float(af_delta):+.1f} "
            f"({'AD 더 안정' if float(af_delta)>0 else 'CT 더 안정'})</div>"
            if af_delta else ""
        )
        + f"</td></tr></table>"

        # ── Row 2: CT GO | AD GO ──
        + f"<table width='100%' cellspacing='0' cellpadding='0' style='margin-bottom:14px'><tr>"
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

    st.subheader("🧫 BISECT PASS Cases — 84개 기능 스위치 검증 케이스")
    st.markdown(
        """
        **BISECT** (Biological Isoform-Switch Evidence Characterization Tool)는 15개의 독립 분석 모듈을 통해
        각 유전자의 아이소폼 전환이 실제로 **생물학적 의미**를 가지는지 다층적으로 검증합니다.

        아래 표는 두 단계 검증(stage1: 통계 + stage2: 생물학 증거)을 모두 통과한 **84개 케이스**입니다.
        🟡 **노란색 행** = 도메인 구조 변화(단백질 기능 변화 직접 증거)가 확인된 고신뢰 케이스입니다.

        > 유전자 이름을 검색창에 입력하면 도메인 구조 그림, 규제 인자 분석, 생물학 리포트가 펼쳐집니다.
        """
    )

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
    _mc1.metric(
        "BISECT PASS 케이스",
        len(_bdf),
        help="15-모듈 파이프라인을 통과한 기능 스위치 후보 총수",
    )
    _mc2.metric(
        "세포 유형 수",
        _bdf['cell_type'].nunique(),
        help="분석된 고유 세포 유형 수 (Excitatory/Inhibitory 뉴런 등)",
    )
    _mc3.metric(
        "도메인 변화 케이스",
        int((_bdf['domains_gained'].fillna('') != '').sum()),
        help="Pfam + AlphaFold로 단백질 도메인 획득이 확인된 케이스 수 — 가장 직접적인 기능 변화 증거",
    )
    _mc4.metric(
        "NAT 중복 케이스",
        int(_bdf['nat'].fillna(False).sum()),
        help="Natural Antisense Transcript(NAT)와 게놈 위치가 겹치는 케이스 — 발현 조절 복잡성 추가 증거",
    )
    _mc5.metric(
        "S1 교차 유전자",
        len(_s1_genes & set(_bdf['gene'].dropna())),
        help="현재 업로드 데이터의 Scenario 1 목록과 BISECT PASS 케이스가 겹치는 유전자 수 (DTU 데이터 있을 때만 표시)",
    )

    st.divider()

    # ── Global TF / ASF / Epigenetic violin plot ──────────────────────────────
    with st.expander("📊 전체 케이스 — 조절 인자 활성 변화 분석 (Volcano + Violin)", expanded=False):
        st.markdown("""
**이 섹션은 BISECT 파이프라인이 84개 케이스에서 감지한 TF·ASF·후성유전 조절 인자들이
AD vs. CT 조건에서 어떻게 달라졌는지를 전체적으로 조망합니다.**

- **Volcano Plot** (위): X축 = 발현 변화 크기(logFC), Y축 = 통계적 유의성(-log₁₀ p-adj).
  오른쪽 위 = AD에서 유의미하게 증가한 인자 | 왼쪽 위 = 유의미하게 감소한 인자.
  점선 내부(|logFC| < 0.1 또는 p > 0.01) = 유의미하지 않은 변화.
- **Violin Plot** (아래): 같은 인자가 여러 케이스에서 어떤 logFC 분포를 보이는지 시각화.
  3개 이상의 케이스에서 반복 감지된 인자만 표시합니다.
- **● 원 = 기존 AD 문헌에 알려진 인자** | **◆ 다이아몬드 = 이 연구에서 새로 발견된 인자**
        """)

        # Build global regulator dataframe
        _glob_rows = []
        for _gc in _bisect_raw:
            _gregs = _parse_regulators(_gc.get('top_regulators', ''))
            for _r in _gregs:
                _gene_r = _r.get('gene', '')
                _kb = _REGULATOR_KB.get(_gene_r, (None, None, ''))
                _cat   = _kb[0] or 'TF'
                _known = _kb[1] if _kb[1] is not None else False
                _glob_rows.append({
                    'Regulator':  _gene_r,
                    'logFC':      float(_r.get('logFC', 0)),
                    'Direction':  _r.get('direction', '').capitalize(),
                    'Category':   _cat,
                    'Knowledge':  '🔵 Known AD' if _known else '🟠 Novel',
                    '-log10(padj)': float(_r.get('neg_log10_padj', 0)),
                    'Case':       _gc.get('gene', ''),
                    'CellType':   _gc.get('cell_type', ''),
                })
        if _glob_rows:
            _gdf = pd.DataFrame(_glob_rows)

            # ── Global Volcano (full sig_regulators from analysis.json) ──────────
            _all_sig_rows = _load_all_sig_regulators(str(_BISECT_PATH))
            if _all_sig_rows:
                _gvdf = pd.DataFrame(_all_sig_rows)
                _gvdf['Category'] = _gvdf['Gene'].map(
                    lambda _g: _REGULATOR_KB.get(_g, ('TF', None, ''))[0] or 'TF'
                )
                _gvdf['Knowledge'] = _gvdf['Gene'].map(
                    lambda _g: (
                        '🔵 Known AD' if _REGULATOR_KB.get(_g, (None, None))[1] is True
                        else ('🟠 Novel' if _REGULATOR_KB.get(_g, (None, None))[1] is False
                              else '⚪ Unknown')
                    )
                )
                # Label known regulators with high significance
                _gvdf['Label'] = _gvdf.apply(
                    lambda _row: (
                        _row['Gene']
                        if (_REGULATOR_KB.get(_row['Gene'], (None, None))[1] is True
                            and float(_row['-log10(padj)']) > 10)
                        else ''
                    ), axis=1
                )
                _fig_gvol = px.scatter(
                    _gvdf,
                    x='logFC', y='-log10(padj)',
                    color='Direction',
                    symbol='Knowledge',
                    color_discrete_map={'Up': '#ef4444', 'Down': '#3b82f6'},
                    symbol_map={
                        '🔵 Known AD': 'circle',
                        '🟠 Novel':    'diamond',
                        '⚪ Unknown':  'square',
                    },
                    text='Label',
                    hover_data=['Gene', 'Category', 'Knowledge', 'Case', 'CellType'],
                    title='Volcano Plot — TF/ASF Activity (AD vs CT) · 26 Cases · 전체 Significant Regulators',
                    labels={'logFC': 'logFC (AD vs CT)', '-log10(padj)': '-log₁₀(p-adj)'},
                    height=430,
                )
                _fig_gvol.update_traces(
                    textposition='top center',
                    textfont=dict(size=9, color='#1e293b'),
                    marker=dict(size=8, opacity=0.75),
                )
                _fig_gvol.add_vline(x=0.1,  line_dash='dash', line_color='#94a3b8', line_width=1)
                _fig_gvol.add_vline(x=-0.1, line_dash='dash', line_color='#94a3b8', line_width=1)
                _fig_gvol.add_hline(y=2.0,  line_dash='dash', line_color='#94a3b8', line_width=1)
                _fig_gvol.add_vline(x=0,    line_color='#374151', line_width=1.2)
                _fig_gvol.update_layout(
                    plot_bgcolor='white',
                    xaxis=dict(gridcolor='#f0f0f0'),
                    yaxis=dict(gridcolor='#f0f0f0'),
                    legend_title='',
                    margin=dict(t=50, b=20, l=10, r=10),
                    font=dict(size=11),
                )
                st.plotly_chart(_fig_gvol, use_container_width=True, key='glob_volcano')
                st.caption(
                    f"Volcano: 26개 케이스 analysis.json의 모든 significant_regulators "
                    f"({len(_all_sig_rows)}개 관측). X=logFC, Y=-log₁₀(p-adj). "
                    "점선: |logFC|=0.1, -log₁₀p=2. ● = 기존 AD 연관, ◆ = 새로 발견. "
                    "레이블 = -log₁₀p > 10인 Known 인자."
                )
                st.divider()

            # Show violin only for regulators appearing ≥3 times (else strip plot)
            _freq = _gdf['Regulator'].value_counts()
            _violin_regs = _freq[_freq >= 3].index.tolist()
            _strip_regs  = _freq[_freq < 3].index.tolist()

            if _violin_regs:
                _gdf_v = _gdf[_gdf['Regulator'].isin(_violin_regs)].copy()
                # Sort regulators: Known first, then by median logFC desc
                _reg_order = (
                    _gdf_v.groupby('Regulator')['logFC'].median()
                    .sort_values(ascending=False).index.tolist()
                )
                _fig_vio = px.violin(
                    _gdf_v,
                    x='Regulator', y='logFC',
                    color='Direction',
                    box=True, points='all',
                    color_discrete_map={'Up': '#ef4444', 'Down': '#3b82f6'},
                    category_orders={
                        'Regulator': _reg_order,
                        'Direction': ['Up', 'Down'],
                    },
                    hover_data=['Case', 'CellType', 'Category', 'Knowledge', '-log10(padj)'],
                    title='TF / ASF / Epigenetic logFC Distribution (AD vs CT) — 84 BISECT Cases',
                    labels={'logFC': 'logFC (AD vs CT)', 'Regulator': ''},
                    height=380,
                )
                _fig_vio.add_hline(y=0, line_dash='dash', line_color='#374151', line_width=1.5)
                _fig_vio.update_layout(
                    plot_bgcolor='white',
                    yaxis=dict(gridcolor='#f0f0f0', zeroline=False),
                    legend_title='Direction (AD vs CT)',
                    margin=dict(t=45, b=60, l=10, r=10),
                    font=dict(size=11),
                )
                st.plotly_chart(_fig_vio, use_container_width=True, key='glob_violin')
                st.caption(
                    "바이올린 = logFC 분포 | 박스 = IQR | 점 = 개별 케이스. "
                    "n≥3인 인자만 바이올린으로 표시. "
                    "Red = AD에서 활성 증가, Blue = AD에서 억제."
                )

            # Known vs Novel summary bar
            _know_summ = _gdf.groupby(['Knowledge', 'Direction']).size().reset_index(name='N')
            if not _know_summ.empty:
                _fig_ks = px.bar(
                    _know_summ,
                    x='Knowledge', y='N', color='Direction',
                    barmode='group',
                    color_discrete_map={'Up': '#ef4444', 'Down': '#3b82f6'},
                    title='Known vs Novel 조절인자 방향성',
                    labels={'N': '케이스 수', 'Knowledge': ''},
                    height=260,
                )
                _fig_ks.update_layout(
                    plot_bgcolor='white', legend_title='',
                    margin=dict(t=38, b=20, l=10, r=10),
                )
                st.plotly_chart(_fig_ks, use_container_width=True, key='glob_known_bar')

            # Regulator knowledge table
            _kb_df = (
                _gdf.groupby(['Regulator', 'Category', 'Knowledge'])
                .agg(N_cases=('Case', 'count'),
                     mean_logFC=('logFC', 'mean'),
                     max_neg_log10p=('-log10(padj)', 'max'))
                .reset_index()
                .sort_values('N_cases', ascending=False)
            )
            _kb_df['설명'] = _kb_df['Regulator'].map(
                lambda g: _REGULATOR_KB.get(g, ('', '', ''))[2]
            )
            st.markdown("**조절 인자 요약표**")
            st.dataframe(
                _kb_df.rename(columns={
                    'Regulator': '인자', 'Category': '분류',
                    'Knowledge': '기존 AD 연관',
                    'N_cases': 'N케이스', 'mean_logFC': '평균 logFC',
                    'max_neg_log10p': '최대 -log10p',
                }).round({'평균 logFC': 3, '최대 -log10p': 1}),
                use_container_width=True, hide_index=True,
            )

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

    def _highlight_bisect_row(row):
        # Highlight if: has domain change (BISECT-intrinsic) OR Scenario 1 cross-link
        _dg = str(row.get('Domains Gained', '') or '').strip()
        _dl = str(row.get('Domains Lost',   '') or '').strip()
        _has_domain = bool(_dg or _dl)
        _is_s1 = row.get('Gene', '') in _s1_genes
        if _has_domain or _is_s1:
            return ['background-color: #fef9c3'] * len(row)
        return [''] * len(row)

    _caption_parts = ["🟡 강조 = 도메인 구조 변화 확인 케이스 (단백질 기능 변화 직접 증거)"]
    if _s1_genes:
        _caption_parts.append("또는 Scenario 1 교차 유전자")
    st.caption(" | ".join(_caption_parts))
    st.dataframe(
        _bdf_show.style.apply(_highlight_bisect_row, axis=1).format(
            {'Δ Usage': '{:.3f}', 'pLDDT': '{:.1f}', 'phyloP': '{:.3f}',
             'DTU p-val': '{:.2e}'},
            na_rep='—',
        ),
        use_container_width=True, hide_index=True,
    )

    with st.expander("📋 표 컬럼 설명 — 약어가 낯설다면 펼쳐보세요", expanded=False):
        st.markdown("""
| 컬럼 | 의미 | 해석 기준 |
|------|------|-----------|
| **Δ Usage** | AD − CT 조건 간 아이소폼 사용 비율 차이 | ±0.1 이상이면 의미 있는 전환 |
| **DTU p-val** | 아이소폼 비율 차이 통계 검정 p-value | < 0.05 (또는 < 1e-5) 이면 유의미 |
| **Domains Gained** | AD 아이소폼에서 새로 생긴 Pfam 도메인 | 도메인 획득 = 기능 추가 직접 증거 |
| **Domains Lost** | CT 아이소폼에서 제거된 Pfam 도메인 | 도메인 손실 = 정상 기능 소실 |
| **PPI** | STRING PPI 데이터베이스 기반 상호작용 지원 여부 | SUPPORTED = 새로운 단백질 파트너 예측 |
| **pLDDT** | AlphaFold 구조 예측 신뢰도 점수 (0~100) | ≥70 = 신뢰 가능한 구조, <50 = 무질서 영역 |
| **phyloP** | 100종 척추동물 서열 보존도 (phyloP100way) | >1.5 = 강한 진화적 선택압 (기능적으로 중요) |
| **NAT** | Natural Antisense Transcript 게놈 중복 여부 | True = 반대 가닥 전사체와 겹침 (조절 복잡성) |
| **L1-CDS** | 젊은 L1 레트로트랜스포존 CDS 삽입 여부 | True = 이동성 유전 요소 삽입 (이형성 검토 필요) |
        """)

    # ── Per-case expanders — only rendered when a gene search is active ──────
    # Rendering all 84 expanders at once is expensive (domain maps, DTU charts,
    # IGV iframes). Gate on search query so the table always stays fast.
    if not _bdf_filt.empty:
        st.divider()
        if not _bq:
            st.info(
                f"위 표에서 **{len(_bdf_filt)}건**의 PASS 케이스를 확인할 수 있습니다. "
                "**유전자 이름을 위 검색창에 입력**하면 아래 섹션들이 펼쳐집니다:\n\n"
                "▸ Volcano Plot (어떤 TF·ASF가 얼마나 바뀌었는지)  "
                "▸ 도메인 구조 변화 그림  "
                "▸ DTU Δ Usage 막대 차트  "
                "▸ GO 기능 비교 (CT vs AD 이소폼)  "
                "▸ 종합 생물학 해석 리포트\n\n"
                "추천 검색어: `KIF21B` `DLG1` `NDUFS4` `DMD`",
                icon="🔍",
            )
        else:
            st.markdown(
                f"**케이스 상세 분석** — '{_bq}' 검색 결과 **{len(_bdf_filt)}건** "
                "| 아래 각 케이스를 클릭해 상세 분석을 확인하세요."
            )

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
                # ── Isoform pair + reading guide ──────────────────────────────
                _ct_tx = str(_brow.get('ct_transcript_id') or '').strip()
                _ad_tx = str(_brow.get('ad_transcript_id') or '').strip()
                _safe_ct_key = _ct.replace(' ', '_').replace('-', '_')

                # Case reading guide banner
                st.markdown(
                    "<div style='background:#f0f9ff;border-left:4px solid #0ea5e9;"
                    "padding:10px 14px;border-radius:6px;font-size:0.83rem;"
                    "color:#0c4a6e;margin-bottom:10px'>"
                    "<b>📖 이 케이스 읽는 법</b> &nbsp;—&nbsp; "
                    "①&nbsp;<b>Volcano/Bar</b>: 어떤 TF·ASF가 얼마나 변했는지 "
                    "②&nbsp;<b>Δ Usage 차트</b>: 아이소폼 비율이 얼마나 바뀌었는지 "
                    "③&nbsp;<b>도메인 구조</b>: 어떤 단백질 기능이 추가/제거됐는지 "
                    "④&nbsp;<b>GO 비교</b>: CT vs AD 이소폼의 기능 공간 차이 "
                    "⑤&nbsp;<b>종합 리포트</b>: 인과 경로 전체 서사"
                    "</div>",
                    unsafe_allow_html=True,
                )
                if _ct_tx or _ad_tx:
                    st.markdown(
                        f"<div style='background:#f8fafc;border-radius:6px;"
                        f"padding:6px 12px;font-size:0.82rem;color:#475569;margin-bottom:8px'>"
                        f"분석 대상 이소폼 쌍 &nbsp;|&nbsp; "
                        f"🔵 <b>Control (CT)</b>: <code>{_ct_tx or '—'}</code> "
                        f"&nbsp;→&nbsp; "
                        f"🔴 <b>AD (Disease)</b>: <code>{_ad_tx or '—'}</code></div>",
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
                        title=f"② 아이소폼 사용 비율 변화 (Δ Usage = AD − CT) — {_gene} · {_ct}",
                        labels={
                            'delta_IF':   'Δ Usage (AD − CT)  ·  양수 = AD에서 증가, 음수 = CT에서 우세',
                            'isoform_id': '아이소폼 ID',
                        },
                        text='label',
                        height=max(260, len(_g_dtu) * 32 + 90),
                    )
                    _fig_dtu.add_hline(y=0, line_color='#1e293b', line_width=1.2)
                    _fig_dtu.update_traces(textposition='outside', textfont_size=9)
                    _fig_dtu.update_layout(
                        xaxis_tickangle=-35,
                        legend_title='이소폼 역할',
                        plot_bgcolor='white',
                        yaxis=dict(gridcolor='#f0f0f0'),
                        margin=dict(t=45, b=80, l=10, r=10),
                    )
                    st.plotly_chart(_fig_dtu, use_container_width=True,
                                    key=f"dtu_usage_{_gene}_{_safe_ct_key}")
                    st.caption(
                        "🔵 CT (Control) 이소폼은 정상 조건에서 우세 (Δ Usage < 0). "
                        "🔴 AD (Disease) 이소폼은 알츠하이머 조건에서 비율 증가 (Δ Usage > 0). "
                        "|Δ Usage| ≥ 0.1이면 통계적으로 의미 있는 전환으로 판단합니다."
                    )

                # ── Row 1: core metrics (6-col) ───────────────────────────────
                st.markdown(
                    "<div style='font-size:0.78rem;color:#6b7280;margin:12px 0 4px'>"
                    "📊 <b>핵심 정량 지표</b> — 각 지표 위에 마우스를 올리면 설명이 나옵니다</div>",
                    unsafe_allow_html=True,
                )
                _r1c1, _r1c2, _r1c3, _r1c4, _r1c5, _r1c6 = st.columns(6)
                _delta = _brow.get('delta')
                _r1c1.metric(
                    "Δ Usage (AD−CT)",
                    f"{float(_delta):.3f}" if _delta is not None else "N/A",
                    help="AD 조건에서 이 아이소폼 사용 비율 − CT 조건 비율. ±0.1 이상이면 유의미한 전환.",
                )
                _dtu_p = _brow.get('dtu_p')
                _r1c2.metric(
                    "DTU p-value",
                    f"{float(_dtu_p):.2e}" if _dtu_p else "N/A",
                    help="아이소폼 비율 차이 통계 검정 p-value. 1e-5 미만이면 매우 유의미.",
                )
                _ct_plddt = _brow.get('af_ct_plddt_mean')
                _r1c3.metric(
                    "CT pLDDT",
                    f"{float(_ct_plddt):.1f}" if _ct_plddt else "N/A",
                    help="AlphaFold2로 예측한 Control 이소폼 구조 신뢰도. 70↑ = 신뢰, 50↓ = 무질서",
                )
                _ad_plddt = _brow.get('af_ad_plddt_mean')
                _r1c4.metric(
                    "AD pLDDT",
                    f"{float(_ad_plddt):.1f}" if _ad_plddt else "N/A",
                    delta="구조 신뢰" if _ad_plddt and float(_ad_plddt) >= 70 else None,
                    help="AlphaFold2로 예측한 AD 이소폼 구조 신뢰도. 70 이상이면 실험 가능한 구조 가짐",
                )
                _dplddt = _brow.get('af_delta_plddt')
                _r1c5.metric(
                    "ΔpLDDT (AD−CT)",
                    f"{float(_dplddt):+.1f}" if _dplddt else "N/A",
                    help="양수 = AD 이소폼이 CT보다 더 안정된 구조 | 음수 = AD 이소폼이 더 무질서함",
                )
                _phylo = _brow.get('cons_ad_phylop')
                _r1c6.metric(
                    "phyloP (AD exon)",
                    f"{float(_phylo):.3f}" if _phylo else "N/A",
                    help="AD 특이적 엑손의 100종 척추동물 보존도. 1.5↑ = 강한 기능적 선택압 하에 있음",
                )

                # ── Row 2: domain changes with AlphaFold confidence ───────────
                st.markdown(
                    "<div style='font-size:0.82rem;font-weight:600;color:#1e293b;"
                    "margin:14px 0 4px'>③ 단백질 도메인 구조 변화 "
                    "<span style='font-weight:400;color:#6b7280;font-size:0.75rem'>"
                    "— Pfam 도메인 데이터베이스 + AlphaFold2 구조 확인</span></div>",
                    unsafe_allow_html=True,
                )
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

                # ── Regulatory Origin Analysis ────────────────────────────────
                _case_regs = _parse_regulators(_brow.get('top_regulators', ''))
                _case_mech = str(_brow.get('mechanism_type') or '').strip()
                _case_tss  = str(_brow.get('tss_class')      or '').strip()
                _case_apa  = str(_brow.get('apa_class')       or '').strip()

                if _case_regs or _case_mech:
                    st.divider()
                    st.markdown(
                        "**① 조절 인자 분석 — 아이소폼 전환의 상류 원인 (Regulatory Origin)**"
                    )
                    st.caption(
                        "아이소폼 비율이 달라진 직접적 원인(TF 활성, 스플라이싱 인자, 후성유전 변화)을 역추적합니다. "
                        "Volcano에서 왼쪽 위 = AD에서 억제된 인자, 오른쪽 위 = AD에서 활성화된 인자. "
                        "🔵 원 = 기존 AD 문헌에 알려진 인자, 🟠 다이아몬드 = 이 연구에서 새로 발견된 인자."
                    )

                    _mech_info = _MECHANISM_KO.get(_case_mech, ('', '#64748b', ''))
                    _mech_ko_label, _mech_color, _mech_desc = _mech_info

                    # Mechanism banner
                    if _case_mech:
                        _tss_note = ''
                        _apa_note = ''
                        _tss_bp_v = _brow.get('tss_diff_bp')
                        _apa_bp_v = _brow.get('tts_diff_bp')
                        if _case_tss and _case_tss not in ('same_promoter', ''):
                            _tss_note = f" · TSS: <b>{_case_tss}</b>"
                            if _tss_bp_v:
                                _tss_note += f" ({int(float(_tss_bp_v)):+d}bp)"
                        if _case_apa and _case_apa not in ('same_apa', ''):
                            _apa_note = f" · APA: <b>{_case_apa}</b>"
                            if _apa_bp_v:
                                _apa_note += f" ({int(float(_apa_bp_v)):+d}bp)"
                        st.markdown(
                            f"<div style='background:#faf5ff;border-left:4px solid {_mech_color};"
                            f"padding:10px 14px;border-radius:6px;margin-bottom:10px;"
                            f"font-size:0.86rem'>"
                            f"<b style='color:{_mech_color}'>⚙️ {_mech_ko_label}</b>"
                            f"{_tss_note}{_apa_note}<br>"
                            f"<span style='color:#374151'>{_mech_desc}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # TF/ASF bar chart (logFC, colored by direction, annotated)
                    if _case_regs:
                        _rrr = []
                        for _r in _case_regs:
                            _rg = _r.get('gene', '?')
                            _kb = _REGULATOR_KB.get(_rg, ('TF', None, ''))
                            _rrr.append({
                                'Gene':       _rg,
                                'logFC':      float(_r.get('logFC', 0)),
                                'Direction':  _r.get('direction', '').capitalize(),
                                'Category':   _kb[0] or 'TF',
                                'Known':      '🔵 Known AD' if _kb[1] else '🟠 Novel',
                                '-log10p':    float(_r.get('neg_log10_padj', 0)),
                                'Label':      f"{_rg}\n({_kb[0] or 'TF'})",
                            })
                        _rdf = pd.DataFrame(_rrr).sort_values('logFC')

                        # ── Per-case Volcano (full sig_regs from analysis.json) ──
                        _sig_regs_full = _load_case_sig_regs(_gene, _ct)
                        _vol_src = _sig_regs_full if _sig_regs_full else _case_regs
                        _vrows = []
                        for _vr in _vol_src:
                            _vg   = _vr.get('gene', '?')
                            _vlfc = float(_vr.get('logFC', 0))
                            _vnlp = float(_vr.get('neg_log10_padj', 0))
                            _vdir = _vr.get('direction', '').capitalize()
                            _kb2  = _REGULATOR_KB.get(_vg, ('TF', None, ''))
                            _vknown = (
                                '🔵 Known AD' if _kb2[1] is True
                                else ('🟠 Novel' if _kb2[1] is False else '⚪ Unknown')
                            )
                            _vrows.append({
                                'Gene':         _vg,
                                'logFC':        _vlfc,
                                '-log10(padj)': _vnlp,
                                'Direction':    _vdir,
                                'Category':     _kb2[0] or 'TF',
                                'Knowledge':    _vknown,
                                'Label': _vg if (_kb2[1] is True or _vnlp > 20) else '',
                            })
                        if _vrows:
                            _vdf = pd.DataFrame(_vrows)
                            _fig_vol = px.scatter(
                                _vdf,
                                x='logFC', y='-log10(padj)',
                                color='Direction',
                                symbol='Knowledge',
                                color_discrete_map={'Up': '#ef4444', 'Down': '#3b82f6'},
                                symbol_map={
                                    '🔵 Known AD': 'circle',
                                    '🟠 Novel':    'diamond',
                                    '⚪ Unknown':  'square',
                                },
                                text='Label',
                                hover_data=['Gene', 'Category', 'Knowledge'],
                                title=f'Volcano — TF/ASF Activity · {_gene} · {_ct}',
                                labels={
                                    'logFC':        'logFC (AD vs CT)',
                                    '-log10(padj)': '-log₁₀(p-adj)',
                                },
                                height=360,
                            )
                            _fig_vol.update_traces(
                                textposition='top center',
                                textfont=dict(size=9, color='#1e293b'),
                                marker=dict(size=10, opacity=0.85),
                            )
                            _fig_vol.add_vline(x=0.1,  line_dash='dash', line_color='#94a3b8', line_width=1)
                            _fig_vol.add_vline(x=-0.1, line_dash='dash', line_color='#94a3b8', line_width=1)
                            _fig_vol.add_hline(y=2.0,  line_dash='dash', line_color='#94a3b8', line_width=1)
                            _fig_vol.add_vline(x=0,    line_color='#374151', line_width=1.2)
                            _fig_vol.update_layout(
                                plot_bgcolor='white',
                                xaxis=dict(gridcolor='#f0f0f0'),
                                yaxis=dict(gridcolor='#f0f0f0'),
                                legend_title='',
                                margin=dict(t=42, b=20, l=10, r=10),
                                font=dict(size=11),
                            )
                            st.plotly_chart(
                                _fig_vol, use_container_width=True,
                                key=f"vol_{_gene}_{_safe_ct_key}",
                            )
                            st.caption(
                                f"Volcano: X=logFC (AD vs CT), Y=-log₁₀(p-adj). "
                                f"점선: |logFC|=0.1, -log₁₀p=2. 총 {len(_vrows)}개 인자"
                                + (" (analysis.json 전체)" if _sig_regs_full else " (top regulators)")
                                + ". ● = 기존 AD, ◆ = 신규 발견."
                            )

                        _r_color = {
                            row['Gene']: ('#ef4444' if row['Direction'] == 'Up' else '#3b82f6')
                            for _, row in _rdf.iterrows()
                        }
                        _fig_reg = px.bar(
                            _rdf,
                            x='logFC', y='Gene',
                            orientation='h',
                            color='Direction',
                            color_discrete_map={'Up': '#ef4444', 'Down': '#3b82f6'},
                            text=_rdf['Gene'].map(
                                lambda g: _REGULATOR_KB.get(g, ('', '', ''))[0]
                            ),
                            hover_data=['Category', 'Known', '-log10p'],
                            title=f'TF / ASF 활성 변화 — {_gene}  ·  {_ct} (AD vs CT logFC)',
                            labels={'logFC': 'logFC (AD vs CT)', 'Gene': ''},
                            height=max(200, len(_rrr) * 52 + 80),
                        )
                        _fig_reg.add_vline(x=0, line_color='#374151', line_width=1.5,
                                           line_dash='dash')
                        _fig_reg.update_traces(textposition='outside', textfont_size=9)
                        _fig_reg.update_layout(
                            plot_bgcolor='white',
                            xaxis=dict(gridcolor='#f0f0f0'),
                            legend_title='',
                            margin=dict(t=40, b=20, l=60, r=90),
                        )
                        st.plotly_chart(_fig_reg, use_container_width=True,
                                        key=f"reg_chart_{_gene}_{_safe_ct_key}")

                        # Known vs Novel annotation cards
                        _known_here   = [r for r in _case_regs
                                         if _REGULATOR_KB.get(r['gene'], (None, None))[1] is True]
                        _novel_here   = [r for r in _case_regs
                                         if _REGULATOR_KB.get(r['gene'], (None, None))[1] is False]
                        _unknown_here = [r for r in _case_regs
                                         if r['gene'] not in _REGULATOR_KB]

                        _ann_parts = []
                        if _known_here:
                            _kh_str = '; '.join(
                                f"<b>{r['gene']}</b> ({r['direction']}, logFC={r['logFC']:+.3f}) "
                                f"— {_REGULATOR_KB[r['gene']][2]}"
                                for r in _known_here
                            )
                            _ann_parts.append(
                                f"<div style='background:#eff6ff;border-left:3px solid #3b82f6;"
                                f"padding:8px 12px;border-radius:4px;margin:4px 0;"
                                f"font-size:0.82rem'>"
                                f"<b style='color:#1d4ed8'>🔵 기존 AD 연관 인자</b><br>{_kh_str}</div>"
                            )
                        if _novel_here:
                            _nh_str = '; '.join(
                                f"<b>{r['gene']}</b> ({r['direction']}, logFC={r['logFC']:+.3f}) "
                                f"— {_REGULATOR_KB[r['gene']][2]}"
                                for r in _novel_here
                            )
                            _ann_parts.append(
                                f"<div style='background:#fff7ed;border-left:3px solid #f59e0b;"
                                f"padding:8px 12px;border-radius:4px;margin:4px 0;"
                                f"font-size:0.82rem'>"
                                f"<b style='color:#b45309'>🟠 새로 발견된 조절 인자</b><br>{_nh_str}</div>"
                            )
                        if _unknown_here:
                            _uh_str = ', '.join(
                                f"{r['gene']} ({r['direction']}, logFC={r['logFC']:+.3f})"
                                for r in _unknown_here
                            )
                            _ann_parts.append(
                                f"<div style='background:#f8fafc;border-left:3px solid #94a3b8;"
                                f"padding:8px 12px;border-radius:4px;margin:4px 0;"
                                f"font-size:0.82rem'>"
                                f"<b style='color:#475569'>⚪ 기타 인자</b>: {_uh_str}</div>"
                            )
                        if _ann_parts:
                            st.markdown("".join(_ann_parts), unsafe_allow_html=True)

                # ── Proportion chart + GO comparison + Bio report ────────────
                _ids_arr_b = np.asarray(ids, dtype=str)
                _ct_idx_b  = np.where(_ids_arr_b == _ct_tx)[0]
                _ad_idx_b  = np.where(_ids_arr_b == _ad_tx)[0]
                _ct_go_scores = sm[_ct_idx_b[0]] if len(_ct_idx_b) > 0 else None
                _ad_go_scores = sm[_ad_idx_b[0]] if len(_ad_idx_b) > 0 else None

                st.divider()

                # 1. Proportion chart — estimated isoform usage ratio CT vs AD
                st.markdown("**② 아이소폼 사용 비율 추정 — Control(CT) vs Disease(AD)**")
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
                        "CT 조건에서 모든 이소폼이 균등하게 발현된다고 가정(1/n)한 뒤, "
                        "DTU Δ Usage를 더해 AD 조건의 비율을 추정합니다. "
                        "🔵 CT 이소폼이 정상에서 주로 쓰이다가, 🔴 AD 이소폼으로 전환되는 비율 변화를 시각화합니다. "
                        "핵심 이소폼 쌍만 강조하며 나머지는 '◻ Other'로 묶습니다."
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
                        st.markdown(
                            "**④ GO 기능 예측 점수 비교 — CT 이소폼 vs AD 이소폼**"
                        )
                        st.caption(
                            "PRISM이 예측한 GO term 기능 점수(0~1)를 두 이소폼 간 나란히 비교합니다. "
                            "🔵 CT 이소폼과 🔴 AD 이소폼에서 높이 차이가 큰 GO term이 "
                            "이소폼 전환으로 인한 **기능 변화 후보**입니다. "
                            f"점선(threshold = {thr})을 넘는 항목만 신뢰도 높은 예측으로 간주합니다."
                        )
                        st.plotly_chart(_fig_cmp, use_container_width=True,
                                        key=f"go_cmp_{_gene}_{_safe_ct_key}")

                # 3. Biological prediction report (⑤)
                st.markdown(
                    "**⑤ 종합 생물학 해석 리포트**&nbsp;"
                    "<span style='font-size:0.8rem;color:#6b7280;font-weight:400'>"
                    "— PRISM+BISECT 15개 모듈 분석을 인과 경로 형태로 통합 서술</span>",
                    unsafe_allow_html=True,
                )
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
                        "③ 단백질 도메인 구조 변화 지도 (CT → AD)</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"🔵 Control 이소폼: `{_ct_tx or '—'}` &nbsp;→&nbsp; "
                        f"🔴 AD 이소폼: `{_ad_tx or '—'}`. "
                        "각 막대가 하나의 이소폼을 나타내며, 색상 블록이 Pfam 도메인 위치입니다. "
                        "초록 = AD에서 새로 생긴 도메인(GAIN) | 빨강 = CT에서 사라진 도메인(LOSS). "
                        "AlphaFold pLDDT ≥ 70인 도메인만 구조적으로 신뢰 가능합니다."
                    )
                    st.image(str(_dmap), use_column_width=True)

                # ── IGV / UCSC quick links ────────────────────────────────────
                st.divider()
                st.markdown(
                    "<div style='font-size:0.88rem;font-weight:600;"
                    "color:#1e293b;margin-bottom:2px'>"
                    "🧬 유전체 브라우저 — 엑손 구조 직접 확인 (외부 링크)</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "아래 버튼으로 외부 유전체 브라우저를 열어 실제 엑손 배치와 전사체 구조를 확인하세요. "
                    "IGV / UCSC에서 아이소폼 ID로 검색하면 CT·AD 전사체의 엑손 차이를 시각적으로 비교할 수 있습니다."
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
