"""Page 1 — Summary Dashboard (Modules A1 + A2 + A3)."""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from prism_app.reports.coverage import generate_coverage_report
from prism_app.reports.novel_summary import generate_novel_summary
from prism_app.reports.validation import generate_validation_report
from prism_app.core.classifier import classify_isoforms, scenario_summary

st.set_page_config(page_title="Overview — PRISM", layout="wide")
st.title("📊 Overview")
st.caption("Coverage report, GO-term distribution, and scenario summary.")

with st.expander("📖 이 페이지 사용법", expanded=False):
    st.markdown("""
**Overview** 페이지는 PRISM 예측 결과를 4개 섹션으로 요약합니다.

| 섹션 | 설명 | 주목할 점 |
|------|------|-----------|
| **A1 · Coverage** | 전체 아이소폼 수 · 타입별(Known/NIC/NNIC) 분포 · 고신뢰 예측 비율 | Score > 임계값인 아이소폼 비율 확인 |
| **D1 · 4-Scenario** | PRISM 점수 + DTU 결과 조합으로 4가지 기능 시나리오 분류 | S1(기능 스위치) > S3(구성적 신규 기능) 순으로 우선 분석 |
| **A3 · Novel** | NIC/NNIC 아이소폼 중 새로운 GO 기능이 예측된 아이소폼 목록 | DTU 파일 없이도 신규 기능 후보 발굴 가능 |
| **A2 · Validation** | UniProt 주석 대비 PRISM 예측 정확도 (AUPRC + 95% CI) | 랜덤 분류기 기준(0.5)과 비교; 0.7 이상이면 양호 |

**4-시나리오 분류 기준:**
- **S1** DTU + 신규 GO 예측 → 기능 변화 아이소폼 스위치 (최우선 후보)
- **S2** DTU + 신규 GO 없음 → 발현량 변화만 있는 스위치
- **S3** DTU 없음 + 신규 GO 예측 → 조건 무관 신규 기능 (Use Case B)
- **S4** 둘 다 없음 → 배경 아이소폼

> DTU 파일을 업로드하지 않으면 모든 아이소폼은 DTU(-) 처리되어 S3/S4만 존재합니다.
    """)

# ── Get data from session ────────────────────────────────────────────────────
cfg = st.session_state.get('cfg', {})
sm  = cfg.get('score_matrix')
if sm is None:
    st.warning("No data loaded. Return to the main page and select a data source.")
    st.stop()

ids   = cfg['isoform_ids']
types = cfg.get('isoform_types')
genes = cfg.get('gene_ids')
go    = cfg['go_terms']
gnames= cfg['go_names']
thr   = cfg['score_threshold']
dtu   = cfg.get('dtu_df')

# ── Coverage Report ──────────────────────────────────────────────────────────
st.subheader("A1 · Coverage Summary")
st.caption("아이소폼 타입별 분포와 고신뢰 GO 예측(Score > 임계값) 커버리지를 확인합니다.")

with st.spinner("Computing coverage report…"):
    rep = generate_coverage_report(sm, ids, types, go, gnames, score_threshold=thr)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total isoforms",    f"{rep.total_isoforms:,}")
c2.metric("Known (Ensembl)",   f"{rep.n_known:,}", f"{100*rep.n_known/max(1,rep.total_isoforms):.1f}%")
c3.metric("NIC",               f"{rep.n_nic:,}",   f"{100*rep.n_nic/max(1,rep.total_isoforms):.1f}%")
c4.metric("NNIC",              f"{rep.n_nnic:,}",  f"{100*rep.n_nnic/max(1,rep.total_isoforms):.1f}%")
c5.metric(f"Score>{thr} (any GO)", f"{rep.n_with_any_high:,}", f"{rep.pct_with_any_high:.1f}%")

# Type breakdown pie
with_high = {
    'Known': rep.n_known_with_high,
    'NIC':   rep.n_nic_with_high,
    'NNIC':  rep.n_nnic_with_high,
}
fig_pie = px.pie(
    names=list(with_high.keys()),
    values=list(with_high.values()),
    title=f"Isoforms with score>{thr} by structural type",
    color_discrete_map={'Known': '#4c72b0', 'NIC': '#55a868', 'NNIC': '#c44e52'},
    hole=0.35,
)
fig_pie.update_traces(textinfo='percent+label')

# Per-GO bar chart
go_df = pd.DataFrame(rep.per_go).sort_values('n_high', ascending=False)
fig_go = px.bar(
    go_df.head(20), x='name', y='n_high',
    title=f"Top GO terms by isoform count (score > {thr})",
    labels={'name': 'GO Term', 'n_high': f'N isoforms (score>{thr})'},
    color='mean_score',
    color_continuous_scale='RdYlGn',
    range_color=[0, 1],
)
fig_go.update_layout(xaxis_tickangle=-40, height=380)

col_a, col_b = st.columns([1, 2])
with col_a:
    st.plotly_chart(fig_pie, use_container_width=True)
with col_b:
    st.plotly_chart(fig_go, use_container_width=True)

st.divider()

# ── Scenario Summary ─────────────────────────────────────────────────────────
st.subheader("D1 · 4-Scenario Classification")
st.caption("각 아이소폼을 DTU 유무 × 신규 GO 예측 유무로 4가지 시나리오에 분류합니다. Scenario 1·3이 실험 후보의 우선순위입니다.")

with st.spinner("Classifying isoforms…"):
    annot = cfg.get('existing_annotations')
    classified = classify_isoforms(
        sm, ids, genes, go,
        existing_annotations=annot,
        dtu_df=dtu,
        score_threshold=thr,
    )
    st.session_state['classified_df'] = classified   # share with other pages

summ = scenario_summary(classified)

colors = ['#e63946', '#f4a261', '#2a9d8f', '#adb5bd']
fig_sc = px.bar(
    summ, x='scenario_label', y='count',
    text='pct',
    title='Isoform Scenario Distribution',
    color='scenario_label',
    color_discrete_sequence=colors,
    labels={'scenario_label': '', 'count': 'N isoforms'},
)
fig_sc.update_traces(texttemplate='%{text}%', textposition='outside')
fig_sc.update_layout(showlegend=False, height=380, xaxis_tickangle=-15)

col_c, col_d = st.columns([2, 1])
with col_c:
    st.plotly_chart(fig_sc, use_container_width=True)
with col_d:
    st.dataframe(summ[['scenario', 'scenario_label', 'count', 'pct']],
                 use_container_width=True, hide_index=True)

st.divider()

# ── Novel Isoform Summary ────────────────────────────────────────────────────
st.subheader("A3 · Novel Isoform Function Predictions")
st.caption("NIC/NNIC 아이소폼 중 기존 Ensembl 주석에 없는 GO 기능이 예측된 아이소폼을 GO 기능별로 집계합니다.")

if types is not None and np.isin(np.asarray(types, dtype=str), ['nic', 'nnic', 'novel']).any():
    with st.spinner("Summarising novel isoform functions…"):
        novel_rep = generate_novel_summary(sm, ids, types, go, gnames, score_threshold=thr)

    nc1, nc2, nc3 = st.columns(3)
    nc1.metric("Total novel isoforms", f"{novel_rep.total_novel:,}")
    nc2.metric(f"With score>{thr}", f"{novel_rep.n_novel_with_any_high:,}", f"{novel_rep.pct_novel_with_high:.1f}%")
    nc3.metric("GO terms with novel predictions", novel_rep.n_prism18_terms_with_novel + novel_rep.n_extended_terms_with_novel)

    st.dataframe(novel_rep.to_dataframe(), use_container_width=True, hide_index=True)
else:
    st.info("Isoform type labels not provided — novel isoform summary unavailable. "
            "Upload an isoform_types file to enable this section.")

st.divider()

# ── Known Annotation Validation (A2) ─────────────────────────────────────────
st.subheader("A2 · Known Annotation Validation (AUPRC)")
st.caption("UniProt/UniProtKB 주석을 정답으로 사용해 PRISM 예측의 정확도를 GO 기능별 AUPRC로 평가합니다. 유전자 ID 파일이 있어야 활성화됩니다.")

@st.cache_data(show_spinner="Computing AUPRC validation…")
def _run_validation(sm_bytes, sm_shape, ids_list, genes_list, go_list, gnames_json,
                    thr, n_bootstrap, mode):
    import json, numpy as np
    from prism_app.reports.validation import generate_validation_report
    sm_arr  = np.frombuffer(sm_bytes, dtype=np.float32).reshape(sm_shape)
    ids_arr = np.array(ids_list)
    genes_arr = np.array(genes_list) if genes_list else None
    go_names  = json.loads(gnames_json)
    return generate_validation_report(
        sm_arr, ids_arr, go_list, go_names,
        gene_ids=genes_arr,
        n_bootstrap=n_bootstrap,
    )

if genes is None:
    st.info(
        "Upload a **Gene ID** file in the sidebar to enable AUPRC validation. "
        "PRISM maps gene symbols → GO annotations to compute precision metrics.",
        icon="ℹ️",
    )
else:
    import json as _json

    n_boot = 200 if cfg.get('mode') == 'demo' else 100
    val_rep = _run_validation(
        sm.astype(np.float32).tobytes(),
        sm.shape,
        list(np.asarray(ids, dtype=str)),
        list(np.asarray(genes, dtype=str)),
        list(go),
        _json.dumps(gnames),
        thr,
        n_boot,
        cfg.get('mode', 'upload'),
    )

    if val_rep is None:
        st.warning(
            "No GO annotation overlap found. "
            "AUPRC validation requires gene symbols matching the bundled annotation "
            "(UniProt/UniProtKB BP terms). Ensembl gene IDs are automatically converted.",
            icon="⚠️",
        )
    else:
        va1, va2, va3, va4 = st.columns(4)
        va1.metric("Annotated isoforms", f"{val_rep.n_isoforms_with_annotation:,}")
        va2.metric("GO terms evaluated",  val_rep.n_go_terms)
        va3.metric("Macro AUPRC",         f"{val_rep.macro_auprc:.4f}")
        va4.metric("95% CI",
                   f"[{val_rep.macro_auprc_ci[0]:.4f}, {val_rep.macro_auprc_ci[1]:.4f}]")

        per_go_df = val_rep.to_dataframe()

        col_val1, col_val2 = st.columns([3, 2])

        with col_val1:
            fig_auprc = px.bar(
                per_go_df.head(18).sort_values('auprc', ascending=True),
                x='auprc', y='name',
                orientation='h',
                color='auprc',
                color_continuous_scale='RdYlGn',
                range_color=[0, 1],
                title="Per-GO AUPRC (known annotation validation)",
                labels={'auprc': 'AUPRC', 'name': ''},
                height=max(300, len(per_go_df) * 22),
                text='auprc',
            )
            fig_auprc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_auprc.add_vline(x=0.5, line_dash='dash', line_color='grey',
                                annotation_text='random (0.5)')
            fig_auprc.update_layout(coloraxis_showscale=False, yaxis_tickfont_size=11)
            st.plotly_chart(fig_auprc, use_container_width=True)

        with col_val2:
            st.caption(
                f"**{val_rep.n_isoforms_with_annotation:,}** isoforms have known GO annotations. "
                f"Macro AUPRC **{val_rep.macro_auprc:.4f}** "
                f"(95% CI: {val_rep.macro_auprc_ci[0]:.4f}–{val_rep.macro_auprc_ci[1]:.4f}) "
                f"across {val_rep.n_go_terms} GO terms evaluated with ≥2 positives.\n\n"
                "A random classifier scores 0.5; PRISM achieves **0.70** on muscle "
                "(Lee et al. 2026, §3.3)."
            )
            st.dataframe(
                per_go_df[['name', 'auprc', 'n_pos']].rename(
                    columns={'name': 'GO Term', 'auprc': 'AUPRC', 'n_pos': 'Positives'}
                ),
                use_container_width=True,
                hide_index=True,
                height=min(400, len(per_go_df) * 36 + 40),
            )

        csv_val = per_go_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download validation metrics (CSV)",
            csv_val, "auprc_validation.csv", "text/csv",
        )
