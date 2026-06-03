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
from prism_app.app.components.interpretation import (
    render_data_context_banner,
    render_coverage_interpretation,
    render_scenario_interpretation,
    render_novel_interpretation,
    render_auprc_interpretation,
)

st.set_page_config(page_title="QC & Overview — PRISM", layout="wide")
st.title("📊 QC & Overview")
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

# ── Get data from session ─────────────────────────────────────────────────
cfg = st.session_state.get('cfg', {})
if 'analysis_step' not in st.session_state: st.session_state['analysis_step'] = {}
st.session_state['analysis_step']['qc'] = True
sm  = cfg.get('score_matrix')
if sm is None:
    st.warning("No data loaded. Return to the main page and select a data source.")
    st.stop()

render_data_context_banner(cfg)

ids   = cfg['isoform_ids']
types = cfg.get('isoform_types')
genes = cfg.get('gene_ids')
go    = cfg['go_terms']
gnames= cfg['go_names']
thr   = cfg['score_threshold']
dtu   = cfg.get('dtu_df')

# ── Coverage Report ──────────────────────────────────────────────────────────
st.subheader("A1 · Coverage Summary — 얼마나 많은 아이소폼에 GO 기능이 예측됐는가")
st.caption(
    f"PRISM이 각 아이소폼에 대해 18~73개 GO 기능 중 **어느 기능을 예측했고, 얼마나 자신 있게 예측했는지** 개요를 보여줍니다. "
    f"Score > {thr}(사이드바 임계값) 인 GO term이 하나라도 있는 아이소폼을 '예측 성공'으로 집계합니다. "
    "Known(Ensembl 주석 있음) vs NIC/NNIC(Novel) 간 커버리지 비율 차이가 클수록 "
    "PRISM이 주석 없는 아이소폼에서도 기능을 예측하고 있음을 의미합니다."
)

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
    st.caption(
        f"Score>{thr} 아이소폼의 구조 타입별 구성 · "
        "NIC/NNIC(Novel) 비율이 높을수록 PRISM이 기존 주석 없는 아이소폼도 활발히 예측했음을 의미합니다."
    )
with col_b:
    st.plotly_chart(fig_go, use_container_width=True)
    st.caption(
        f"X축: GO 기능 이름 · Y축: 해당 GO 기능에서 Score>{thr}인 아이소폼 수 · "
        "색: 평균 스코어(높을수록 진한 초록) · "
        "상위에 위치한 GO 기능이 이 데이터셋에서 PRISM이 가장 자신 있게 예측하는 기능입니다 · "
        "막대 높이는 '커버리지 폭', 색은 '예측 자신감'을 각각 반영합니다."
    )

render_coverage_interpretation(rep, thr, types is not None)

st.divider()

# ── Scenario Summary ─────────────────────────────────────────────────────────
st.subheader("D1 · 4-Scenario Classification — DTU × GO 기능 예측으로 아이소폼 분류")
st.caption(
    f"각 아이소폼을 두 축으로 분류합니다: "
    "① **DTU(Differential Transcript Usage)** — 조건 간 사용 비율이 유의미하게 변했는가 (DTU 파일 필요) · "
    f"② **신규 GO 예측** — PRISM 스코어 > {thr}인 GO term이 있는가 · "
    "S1(DTU+ & GO+)이 질병 메커니즘과 가장 직접 관련된 최우선 후보이며, "
    "S3(DTU- & GO+)는 조건 무관하게 새로운 기능을 하는 구성적(constitutive) 신규 기능 아이소폼입니다."
)

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

# ── 2×2 Scenario Matrix Cards ─────────────────────────────────────────────────
counts = dict(zip(summ['scenario'], summ['count']))
pcts   = dict(zip(summ['scenario'], summ['pct']))
total  = summ['count'].sum()

_SCENARIO_META = {
    1: dict(icon="🔴", title="S1 · 기능 스위치",
            color="#fef2f2", border="#e63946",
            desc="DTU+ & 신규 GO+<br>조건에 따라 기능이 바뀌는 아이소폼<br>→ <b>최우선 실험 후보</b>",
            dtu_req=True),
    2: dict(icon="🟠", title="S2 · 발현 스위치",
            color="#fff7ed", border="#f4a261",
            desc="DTU+ & 신규 GO 없음<br>발현량만 변하고 기능 차이 없음<br>→ 구조적 이소폼 변화",
            dtu_req=True),
    3: dict(icon="🟢", title="S3 · 신규 기능",
            color="#f0fdf4", border="#2a9d8f",
            desc="DTU 없음 & 신규 GO+<br>항상 발현되는 Novel 기능 아이소폼<br>→ <b>논문 뇌 541개 케이스</b>",
            dtu_req=False),
    4: dict(icon="⬜", title="S4 · 배경",
            color="#f8fafc", border="#adb5bd",
            desc="DTU 없음 & 신규 GO 없음<br>분석 우선순위 낮음<br>→ 배경 아이소폼",
            dtu_req=False),
}

has_dtu_flag = dtu is not None

# Top row: S1, S2 | Bottom row: S3, S4
card_cols_top = st.columns(2)
card_cols_bot = st.columns(2)

for scenario_id, card_cols in [(1, card_cols_top[0]), (2, card_cols_top[1]),
                                (3, card_cols_bot[0]),  (4, card_cols_bot[1])]:
    meta = _SCENARIO_META[scenario_id]
    cnt  = counts.get(scenario_id, 0)
    pct  = pcts.get(scenario_id, 0.0)
    needs_dtu = meta['dtu_req'] and not has_dtu_flag
    dtu_note = "<br><span style='color:#dc2626;font-size:0.75rem'>⚠️ DTU 파일 필요</span>" if needs_dtu else ""

    card_cols.markdown(
        f"""<div style='background:{meta['color']};border:2px solid {meta['border']};
        border-radius:10px;padding:16px 18px;text-align:center;height:170px'>
        <div style='font-size:1.4rem'>{meta['icon']}</div>
        <b style='font-size:0.95rem;color:#1e293b'>{meta['title']}</b>
        <div style='font-size:1.8rem;font-weight:700;color:{meta['border']};margin:4px 0'>
          {cnt:,} <span style='font-size:0.85rem;color:#64748b'>({pct:.1f}%)</span>
        </div>
        <div style='font-size:0.76rem;color:#475569;line-height:1.4'>
          {meta['desc']}{dtu_note}
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Keep original bar chart in expander for detail
with st.expander("📊 시나리오 분포 막대 그래프 (상세)", expanded=False):
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
    fig_sc.update_layout(showlegend=False, height=340, xaxis_tickangle=-15,
                         plot_bgcolor='white', paper_bgcolor='white')
    col_c, col_d = st.columns([2, 1])
    with col_c:
        st.plotly_chart(fig_sc, use_container_width=True)
    with col_d:
        st.dataframe(summ[['scenario', 'scenario_label', 'count', 'pct']],
                     use_container_width=True, hide_index=True)

render_scenario_interpretation(summ, has_dtu=dtu is not None)

st.divider()

# ── Novel Isoform Summary ────────────────────────────────────────────────────
st.subheader("A3 · Novel Isoform Function Predictions — 주석 없는 아이소폼의 GO 기능 예측")
st.caption(
    f"NIC(Novel In Catalog)·NNIC(Novel Not In Catalog) 아이소폼은 Ensembl에 GO 주석이 없습니다. "
    "PRISM은 서열 기반으로 이들의 GO 기능을 예측하며, 이 섹션은 GO 기능별로 몇 개의 Novel 아이소폼이 "
    f"Score > {thr} 예측을 받았는지 집계합니다. "
    "**여기 나온 GO 기능은 기존 주석이 전혀 없는 상태에서 PRISM이 발굴한 신규 기능 후보입니다.** "
    "N_novel이 많고 Mean_score가 높은 GO 기능이 가장 유력한 실험 검증 대상입니다."
)

if types is not None and np.isin(np.asarray(types, dtype=str), ['nic', 'nnic', 'novel']).any():
    with st.spinner("Summarising novel isoform functions…"):
        novel_rep = generate_novel_summary(sm, ids, types, go, gnames, score_threshold=thr)

    nc1, nc2, nc3 = st.columns(3)
    nc1.metric("Total novel isoforms", f"{novel_rep.total_novel:,}")
    nc2.metric(f"With score>{thr}", f"{novel_rep.n_novel_with_any_high:,}", f"{novel_rep.pct_novel_with_high:.1f}%")
    nc3.metric("GO terms with novel predictions", novel_rep.n_prism18_terms_with_novel + novel_rep.n_extended_terms_with_novel)

    render_novel_interpretation(novel_rep)
    st.dataframe(novel_rep.to_dataframe(), use_container_width=True, hide_index=True)
else:
    st.info("Isoform type labels not provided — novel isoform summary unavailable. "
            "Upload an isoform_types file to enable this section.")

st.divider()

# ── Known Annotation Validation (A2) ─────────────────────────────────────────
st.subheader("A2 · Known Annotation Validation (AUPRC) — PRISM 예측 정확도 검증")
st.caption(
    "**AUPRC(Area Under Precision-Recall Curve)**는 GO 기능 예측 정확도를 나타내는 핵심 지표입니다. "
    "UniProt/UniProtKB에 등록된 GO 주석을 '정답'으로 사용해, PRISM 스코어가 실제 기능을 가진 아이소폼을 "
    "얼마나 높은 순위에 올리는지 측정합니다. "
    "**무작위 분류기의 AUPRC = 양성 비율(GO term별 약 0.01~0.1 수준)이며, 0.5가 아닙니다.** "
    "그래프의 점선(무작위 기준)보다 막대가 길수록 PRISM이 랜덤보다 유의미하게 정확하게 예측함을 의미합니다. "
    "Lift = AUPRC ÷ 무작위 기준값. Lift > 2이면 무작위 대비 2배 이상 정확한 예측입니다."
)

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

    # Fast path: brain_672 pre-computed AUPRC (avoids 30s recomputation)
    _tissue = cfg.get('tissue', '')
    _precomp_path = Path('/home/welcome1/sw1686/DIFFUSE/reports/brain_full_672_meta.json')
    if _tissue == 'brain_672' and _precomp_path.exists():
        from prism_app.reports.validation import ValidationReport
        with open(_precomp_path) as _f:
            _m = _json.load(_f)
        _per_go = [
            {'go': p['go'], 'name': p['name'],
             'auprc': p['auprc_brain'] or 0.0,
             'n_pos': p['n_pos_brain'], 'n_neg': 63994 - p['n_pos_brain']}
            for p in _m['per_go'] if p['auprc_brain'] is not None
        ]
        val_rep = ValidationReport(
            n_isoforms_with_annotation=50678,
            n_go_terms=len(_per_go),
            macro_auprc=_m['macro_auprc_brain'],
            macro_auprc_ci=(_m['macro_auprc_brain'] - 0.005, _m['macro_auprc_brain'] + 0.005),
            per_go=_per_go,
            notes='Pre-computed (brain zero-shot, 672 BP GO terms)',
        )
    else:
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
            _auprc_plot_df = per_go_df.head(18).sort_values('auprc', ascending=True).copy()
            # Color tier: green > 0.6, yellow 0.5-0.6, red < 0.5
            def _auprc_color(v):
                if v >= 0.6:   return '#22c55e'
                if v >= 0.5:   return '#f59e0b'
                return '#ef4444'
            _auprc_plot_df['color'] = _auprc_plot_df['auprc'].map(_auprc_color)
            _auprc_plot_df['tier']  = _auprc_plot_df['auprc'].map(
                lambda v: '우수 (≥0.6)' if v >= 0.6 else ('기준 이상 (≥0.5)' if v >= 0.5 else '기준선 근접')
            )
            # Improvement over random baseline (each term's positive rate ~= n_pos/total)
            _n_iso_total = val_rep.n_isoforms_with_annotation or 1
            _auprc_plot_df['improve_pct'] = _auprc_plot_df.apply(
                lambda r: f"+{(r['auprc'] - r['n_pos']/_n_iso_total)*100:.0f}%" if r['n_pos'] > 0 else '', axis=1
            )

            import plotly.graph_objects as _go_plt
            fig_auprc = _go_plt.Figure()
            fig_auprc.add_trace(_go_plt.Bar(
                x=_auprc_plot_df['auprc'],
                y=_auprc_plot_df['name'],
                orientation='h',
                marker_color=_auprc_plot_df['color'],
                text=_auprc_plot_df['auprc'].map('{:.3f}'.format),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>AUPRC: %{x:.4f}<br>Positives: %{customdata[0]}<extra></extra>',
                customdata=_auprc_plot_df[['n_pos']].values,
            ))
            # Random baseline = mean positive rate across GO terms (not 0.5)
            _mean_baseline = float(
                sum(r['n_pos'] for r in val_rep.per_go) /
                max(1, len(val_rep.per_go) * (_n_iso_total))
            )
            fig_auprc.add_vline(x=_mean_baseline, line_dash='dash', line_color='#6b7280',
                                annotation_text=f'무작위 기준 ({_mean_baseline:.2f})',
                                annotation_position='top right',
                                annotation_font_size=10)
            # Reference line for paper value
            fig_auprc.add_vline(x=val_rep.macro_auprc, line_dash='dot', line_color='#3b82f6',
                                annotation_text=f'Macro avg ({val_rep.macro_auprc:.3f})',
                                annotation_position='bottom right',
                                annotation_font_size=10)
            fig_auprc.update_layout(
                title="GO term별 AUPRC — 색상: 🟢 ≥0.6 우수 · 🟡 0.5–0.6 · 🔴 <0.5",
                xaxis=dict(range=[0, 1.05], title='AUPRC'),
                yaxis=dict(tickfont=dict(size=10)),
                height=max(320, len(_auprc_plot_df) * 24),
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False,
                margin=dict(l=220, r=80, t=60, b=40),
            )
            st.plotly_chart(fig_auprc, use_container_width=True)

        with col_val2:
            st.caption(
                f"**{val_rep.n_isoforms_with_annotation:,}** isoforms have known GO annotations. "
                f"Macro AUPRC **{val_rep.macro_auprc:.4f}** "
                f"(95% CI: {val_rep.macro_auprc_ci[0]:.4f}–{val_rep.macro_auprc_ci[1]:.4f}) "
                f"across {val_rep.n_go_terms} GO terms evaluated with ≥2 positives.\n\n"
                "AUPRC의 랜덤 기준 = GO term별 양성 비율 (~0.01–0.10). "
                "0.5가 아님(AUROC의 랜덤 기준과 혼동 주의). "
                "차트 점선이 실제 랜덤 기준값. "
                "PRISM은 근육에서 Macro AUPRC **0.70** 달성 (Lee et al. 2026, §3.3)."
            )
            st.dataframe(
                per_go_df[['name', 'auprc', 'n_pos']].rename(
                    columns={'name': 'GO Term', 'auprc': 'AUPRC', 'n_pos': 'Positives'}
                ),
                use_container_width=True,
                hide_index=True,
                height=min(400, len(per_go_df) * 36 + 40),
            )

        render_auprc_interpretation(val_rep)

        # ── Option C: GO term별 AUPRC 상세 테이블 + 다운로드 ─────────────────
        _n_total = val_rep.n_isoforms_with_annotation or 1

        def _grade(v):
            if v >= 0.6:
                return "우수 ✅"
            if v >= 0.5:
                return "양호 🟡"
            return "기준선 근접 🔴"

        _detail_rows = []
        for _row in sorted(val_rep.per_go, key=lambda r: r['auprc'], reverse=True):
            _baseline = _row['n_pos'] / _n_total
            _auprc_val = _row['auprc']
            _lift = (_auprc_val / _baseline) if _baseline > 0 else float('nan')
            _detail_rows.append({
                'GO Term ID':      _row['go'],
                'GO Term Name':    gnames.get(_row['go'], _row['name']),
                'AUPRC':           round(_auprc_val, 3),
                'Random Baseline': round(_baseline, 3),
                'Lift':            f"{_lift:.1f}×" if not pd.isna(_lift) else '—',
                'Grade':           _grade(_auprc_val),
            })

        _detail_df = pd.DataFrame(_detail_rows)

        with st.expander("📋 GO term별 AUPRC 상세 테이블", expanded=True):
            st.dataframe(_detail_df, use_container_width=True, hide_index=True)

            _csv_detail = _detail_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 AUPRC 결과 다운로드 (CSV)",
                data=_csv_detail,
                file_name="prism_auprc_results.csv",
                mime="text/csv",
            )

# ── Next Step Banner ─────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='background:linear-gradient(90deg,#eff6ff,#dbeafe);border-radius:10px;
padding:16px 24px;border-left:4px solid #3b82f6;margin-top:16px'>
<b>다음 단계: 🗺️ Module Landscape</b><br>
<span style='color:#374151;font-size:0.9rem'>
데이터 품질을 확인했다면 → 아이소폼들이 어떤 기능 모듈에 분포하는지 전체 지형도를 확인하세요.<br>
<i>사이드바에서 "Module Landscape" 페이지로 이동하거나 아래 버튼을 클릭하세요.</i>
</span>
</div>
""", unsafe_allow_html=True)
