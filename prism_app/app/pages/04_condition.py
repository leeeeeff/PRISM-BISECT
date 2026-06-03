"""Page 3 — Condition Analysis (Modules C1 + C2): DTU-linked functional switch."""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as pgo

from prism_app.pipeline.dtu_connector import (
    compute_functional_consequence,
    build_consequence_pivot,
    consequence_summary,
)
from prism_app.pipeline.go_enrichment import enrich_dtu_switched
from prism_app.app.components.interpretation import render_data_context_banner, render_condition_interpretation

st.set_page_config(page_title="Condition Analysis — PRISM", layout="wide")
st.title("🔄 Condition Analysis")
st.caption("DTU-linked functional switch: which isoforms gain or lose function between conditions?")

with st.expander("📖 이 페이지 사용법", expanded=False):
    st.markdown("""
**Condition Analysis** 페이지는 DTU(Differential Transcript Usage) 결과와 PRISM 스코어를 결합하여
조건 간(예: 질병 vs 정상) 기능 변화를 분석합니다.

**사용 방법:**
1. 사이드바에서 **DTU 결과 TSV 파일을 업로드**하세요. (없으면 이 페이지는 비활성화됩니다)
2. 사이드바 슬라이더로 DTU p-value 임계값과 최소 |delta_IF| 기준을 조정합니다.

**DTU 파일 필수 컬럼** (아래 중 하나의 이름이면 자동 인식):

| 항목 | 허용 컬럼명 |
|------|------------|
| 아이소폼 ID | `isoform_id`, `transcriptID`, `featureID` |
| 발현 변화량 | `delta_IF`, `dIF`, `deltaPSI`, `logFC` |
| 유의도 | `pvalue`, `padj`, `FDR`, `adj.p.value` |
| 조건명 | `condition`, `comparison`, `contrast` |

**호환 도구:** satuRn · DEXSeq · IsoformSwitchAnalyzeR · rMATS

| 탭 | 설명 |
|----|------|
| **GAIN/LOSS/NEUTRAL** | 유전자별 GO 기능 획득(GAIN)/손실(LOSS)/변화없음(NEUTRAL) 분류. 같은 조건에서 delta_IF 양수/음수 아이소폼 쌍의 GO 스코어 차이로 계산 |
| **GO Enrichment** | DTU 유의미한 아이소폼들에서 어떤 GO 기능이 집중적으로 변하는지 초기하 분포 검정 (BH-FDR 보정) |
| **Sankey** | 조건 간 아이소폼 클러스터 이동을 흐름도로 시각화 |
| **Gene Detail** | 특정 유전자 검색 → GO 기능별 GAIN/LOSS 막대 차트 |
| **Full Results** | 전체 분석 결과 테이블 다운로드 |
    """)

# ── Session data ──────────────────────────────────────────────────────────────
cfg = st.session_state.get('cfg', {})
if 'analysis_step' not in st.session_state: st.session_state['analysis_step'] = {}
st.session_state['analysis_step']['condition'] = True
sm  = cfg.get('score_matrix')
if sm is None:
    st.warning("No data loaded. Return to the main page."); st.stop()

render_data_context_banner(cfg)

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

dtu_df = cfg.get('dtu_df')

# ── No DTU: show placeholder ─────────────────────────────────────────────────
if dtu_df is None:
    st.info(
        "No DTU results loaded. Upload a DTU file in the sidebar to enable condition analysis.\n\n"
        "**Expected format (TSV/CSV)**:\n"
        "```\nisoform_id  condition  delta_IF  pvalue\n"
        "ENST00000123  disease  -0.35  0.001\n```\n\n"
        "Compatible with: **satuRn**, **DEXSeq**, **IsoformSwitchAnalyzeR** output.",
        icon="📂",
    )
    if classified is not None:
        s1 = classified[classified['scenario'] == 1]
        st.subheader("Preview: Scenario 1 candidates (requires DTU data)")
        if s1.empty:
            st.write("No Scenario 1 isoforms without DTU data — upload DTU to activate.")
        else:
            st.dataframe(s1[['isoform_id', 'gene_id', 'max_score', 'max_go',
                              'dtu_pvalue', 'scenario_label']].head(20),
                         use_container_width=True, hide_index=True)
    st.stop()

# ── DTU loaded ────────────────────────────────────────────────────────────────
st.success(f"DTU 데이터 로드 완료: **{len(dtu_df):,}** 아이소폼-조건 레코드.")

with st.expander("📐 분석 원리 — GAIN/LOSS/NEUTRAL은 무엇을 의미하는가", expanded=True):
    st.markdown("""
### 핵심 개념: 아이소폼 쌍 비교를 통한 GO 기능 방향성 판단

**분석 단위는 "유전자별 아이소폼 쌍"입니다.**
DTU는 조건 간(예: 질병 vs 정상) 아이소폼 사용 비율(Isoform Fraction, IF)의 변화를 측정합니다.
각 유전자에서 BISECT는 두 아이소폼을 짝지어 비교합니다:

| 역할 | 정의 | delta_IF 부호 |
|------|------|:---:|
| **↑ 질병 우세 아이소폼** | 질병 조건에서 IF가 **증가**한 아이소폼 | + |
| **↓ 대체된 아이소폼** | 질병 조건에서 IF가 **감소**한 아이소폼 | − |

각 (유전자, GO term) 쌍에 대해 **PRISM 스코어 차이**를 계산합니다:

```
score_delta = PRISM_score(↑질병 우세 아이소폼, GO term)
            − PRISM_score(↓대체된 아이소폼,    GO term)
```

| 판정 | 조건 | 생물학적 의미 |
|------|------|-------------|
| **GAIN** (초록) | score_delta > +임계값 | 질병에서 우세해진 아이소폼이 이 GO 기능을 **더 높게** 예측받음 → 세포가 이 기능을 새롭게 획득 |
| **LOSS** (빨강) | score_delta < −임계값 | 대체된 아이소폼이 이 GO 기능을 **더 높게** 예측받았었음 → 세포가 이 기능을 상실 |
| **NEUTRAL** (회색) | \|score_delta\| ≤ 임계값 | 두 아이소폼 간 해당 GO 기능 차이 없음 |

> **예시**: 유전자 NDUFS4에서 정상 조건 아이소폼(NDUFS4-201)의 미토콘드리아 호흡 GO 스코어가 0.8이고,
> 질병 조건 우세 아이소폼(NDUFS4-204)의 스코어가 0.3이라면
> → score_delta = 0.3 − 0.8 = **−0.5** → **LOSS** (이 유전자는 질병에서 미토콘드리아 호흡 기능을 손실)

### 중요 구분

- **주체는 "유전자"**: GAIN/LOSS는 개별 아이소폼 자체가 아니라 **유전자 수준 기능 변화**를 표현합니다
- **기전은 "아이소폼 쌍"**: 실제로는 두 아이소폼의 PRISM 스코어 차이가 판정 근거입니다
- **방향 기준**: `delta_IF > 0`인 아이소폼이 "질병 우세"로 정의됩니다 (사이드바 슬라이더로 |delta_IF| 최솟값 조정 가능)
- **유전자 수준 집계**: 한 유전자에 여러 DTU 이벤트가 있으면, 가장 유의미한 이벤트(|delta_IF| 최대) 기준으로 판정합니다
    """)

# ── Cell type / condition filter ──────────────────────────────────────────────
_all_conditions = sorted(dtu_df['condition'].dropna().unique().tolist()) if 'condition' in dtu_df.columns else []
if _all_conditions:
    _sel_conditions = st.sidebar.multiselect(
        "Cell type / Condition 필터",
        options=_all_conditions,
        default=_all_conditions,
        help="분석에 포함할 조건(세포 유형)을 선택합니다. 전체 선택 = 합산 분석",
        key='cond_celltype',
    )
    if _sel_conditions and set(_sel_conditions) != set(_all_conditions):
        dtu_df = dtu_df[dtu_df['condition'].isin(_sel_conditions)].copy()
        st.caption(f"🔍 필터 적용: {', '.join(_sel_conditions)} ({len(dtu_df):,}개 레코드)")

# Global p-value threshold (shared across tabs)
pval_thr = st.sidebar.slider(
    "DTU p-value threshold", 0.001, 0.1, 0.05, 0.005,
    help="Applied in all Condition Analysis tabs",
    key='cond_pval',
)
delta_thr = st.sidebar.slider(
    "Min |delta_IF| to call up/down", 0.05, 0.5, 0.1, 0.05,
    key='cond_delta',
)

tab_matrix, tab_enrich, tab_sankey, tab_genes, tab_table = st.tabs([
    "🗂️ Functional Consequence Matrix",
    "📊 GO Enrichment",
    "🔀 Sankey: Scenario Flow",
    "🧬 Gene-Level Detail",
    "📋 Full Results Table",
])

# ── Cached consequence computation ───────────────────────────────────────────
@st.cache_data(show_spinner="Computing functional consequences…")
def _compute_consequences(dtu_bytes, sm_bytes, sm_shape, ids_list, go_list,
                          gnames_json, pval_thr, delta_thr, score_thr):
    import json, numpy as np, pandas as pd, io
    from prism_app.pipeline.dtu_connector import compute_functional_consequence
    dtu  = pd.read_json(io.BytesIO(dtu_bytes))
    sm_a = np.frombuffer(sm_bytes, dtype=np.float32).reshape(sm_shape)
    ids_a = np.array(ids_list)
    gnames = json.loads(gnames_json)
    return compute_functional_consequence(
        dtu, sm_a, ids_a, go_list, gnames,
        score_threshold=score_thr,
        dtu_pval_threshold=pval_thr,
        delta_if_threshold=delta_thr,
    )


@st.cache_data(show_spinner="Running GO enrichment…")
def _compute_enrichment(dtu_bytes, sm_bytes, sm_shape, ids_list, go_list,
                        gnames_json, pval_thr, score_thr, direction):
    import json, numpy as np, pandas as pd, io
    from prism_app.pipeline.go_enrichment import enrich_dtu_switched
    dtu  = pd.read_json(io.BytesIO(dtu_bytes))
    sm_a = np.frombuffer(sm_bytes, dtype=np.float32).reshape(sm_shape)
    ids_a = np.array(ids_list)
    gnames = json.loads(gnames_json)
    return enrich_dtu_switched(dtu, sm_a, ids_a, go_list, gnames,
                                score_threshold=score_thr,
                                dtu_pval_threshold=pval_thr,
                                direction=direction)


import json as _json
_dtu_bytes   = dtu_df.to_json().encode()
_sm_bytes    = sm.astype(np.float32).tobytes()
_ids_list    = list(np.asarray(ids, dtype=str))
_go_list     = list(go)
_gnames_json = _json.dumps(gnames)

conseq_df = _compute_consequences(
    _dtu_bytes, _sm_bytes, sm.shape, _ids_list, _go_list,
    _gnames_json, pval_thr, delta_thr, thr,
)

# ── Tab 1: Functional Consequence Matrix ──────────────────────────────────────
with tab_matrix:
    st.subheader("유전자 × GO 기능 변화 매트릭스 (Functional Consequence Matrix)")
    st.caption(
        "각 유전자에서 DTU로 감지된 아이소폼 쌍의 PRISM 스코어 차이로 GO 기능 방향성(GAIN/LOSS/NEUTRAL)을 판정합니다. "
        "**GAIN** (초록): 질병 우세 아이소폼이 이 GO 기능을 더 높이 예측받음 — 세포가 이 기능을 새롭게 획득.  "
        "**LOSS** (빨강): 대체된 아이소폼이 이 GO 기능을 더 높이 예측받았었음 — 세포가 이 기능을 상실.  "
        "**NEUTRAL** (회색): 두 아이소폼 간 스코어 차이가 임계값 미만."
    )

    if conseq_df.empty:
        st.warning("유의미한 DTU 이벤트가 없습니다. p-value 임계값을 완화하거나 |delta_IF| 최솟값을 낮춰 보세요.")
    else:
        summ = consequence_summary(conseq_df)

        # ── Chart ①: GO term별 GAIN/LOSS/NEUTRAL 분포 ────────────────────────
        st.markdown("**① GO term별 기능 변화 분포** — 어떤 GO 기능에서 스위치가 집중되는가")
        fig_summ = px.bar(
            summ.head(18),
            x='go_name', y=['GAIN', 'LOSS', 'NEUTRAL'],
            barmode='stack',
            title="GO term별 GAIN / LOSS / NEUTRAL 유전자 수",
            labels={'value': '유전자 수', 'go_name': 'GO 기능', 'variable': '판정'},
            color_discrete_map={'GAIN': '#2a9d8f', 'LOSS': '#e63946', 'NEUTRAL': '#adb5bd'},
            height=380,
        )
        fig_summ.update_layout(xaxis_tickangle=-35, legend_title='')
        st.plotly_chart(fig_summ, use_container_width=True)
        st.caption(
            "X축: 각 GO 기능 항목 · Y축: 해당 GO 기능에서 GAIN/LOSS/NEUTRAL 판정을 받은 **유전자 수** · "
            "막대가 클수록 이 조건에서 그 GO 기능을 가진 아이소폼 스위치가 많이 일어남을 의미 · "
            "NEUTRAL이 압도적이면 DTU는 일어났지만 GO 기능 차이는 적다는 뜻 (발현 스위치 우세)"
        )

        # ── Chart ②: GAIN/LOSS only ───────────────────────────────────────────
        _gl = summ[['go_name', 'GAIN', 'LOSS']].copy()
        _gl['total'] = _gl['GAIN'] + _gl['LOSS']
        _gl = _gl[_gl['total'] > 0].sort_values('total', ascending=True)

        if not _gl.empty:
            st.markdown("**② 기능 스위치 집중 GO term** (NEUTRAL 제외)")
            fig_gl = px.bar(
                _gl,
                x=['GAIN', 'LOSS'],
                y='go_name',
                orientation='h',
                barmode='group',
                title="실제 기능 변화가 있는 GO term — GAIN vs LOSS 유전자 수",
                labels={'value': '유전자 수', 'go_name': '', 'variable': ''},
                color_discrete_map={'GAIN': '#2a9d8f', 'LOSS': '#e63946'},
                height=max(300, len(_gl) * 28),
                text_auto=True,
            )
            fig_gl.update_traces(textposition='outside', textfont_size=10)
            fig_gl.update_layout(
                yaxis_tickfont_size=11,
                legend_title='',
                plot_bgcolor='white',
                xaxis=dict(gridcolor='#f0f0f0'),
            )
            st.plotly_chart(fig_gl, use_container_width=True)
            st.caption(
                "NEUTRAL을 제외하고 실제 GO 기능이 변한 유전자만 집계 · "
                "GAIN 막대가 크면: 이 조건에서 해당 GO 기능을 새로 획득한 유전자가 많음 · "
                "LOSS 막대가 크면: 이 조건에서 해당 GO 기능을 잃은 유전자가 많음 · "
                "양쪽이 비슷하면: 기능 스위치가 방향 편향 없이 일어나는 중"
            )

            # ── Chart ③: Bias diverging bar ──────────────────────────────────
            st.markdown("**③ GO term별 기능 변화 방향 편향도** — GAIN 우세인가, LOSS 우세인가")
            _bias = _gl.copy()
            _bias['bias'] = (_bias['GAIN'] - _bias['LOSS']) / _bias['total']
            _bias = _bias.sort_values('bias')
            _bias['bias_label'] = _bias['bias'].map(
                lambda v: f"{'GAIN' if v >= 0 else 'LOSS'} 편향  {abs(v):.0%}"
            )

            fig_bias = px.bar(
                _bias,
                x='bias', y='go_name',
                orientation='h',
                color='bias',
                color_continuous_scale=[[0, '#e63946'], [0.5, '#f1f5f9'], [1, '#2a9d8f']],
                range_color=[-1, 1],
                title="기능 방향 편향도 — (GAIN − LOSS) / (GAIN + LOSS)",
                labels={'bias': '편향도 (−1 = 전부 LOSS · 0 = 균형 · +1 = 전부 GAIN)', 'go_name': ''},
                text='bias_label',
                height=max(300, len(_bias) * 28),
            )
            fig_bias.update_traces(textposition='outside', textfont_size=9)
            fig_bias.update_layout(
                yaxis_tickfont_size=11,
                coloraxis_showscale=False,
                plot_bgcolor='white',
                xaxis=dict(gridcolor='#f0f0f0', range=[-1.25, 1.25]),
            )
            fig_bias.add_vline(x=0, line_color='#64748b', line_width=1.5)
            st.plotly_chart(fig_bias, use_container_width=True)
            st.caption(
                "편향도 = (GAIN 유전자 수 − LOSS 유전자 수) / (GAIN + LOSS 유전자 수) · "
                "초록(+1 방향): 이 GO 기능은 질병에서 **획득 경향** — 새로운 기능을 가진 아이소폼이 우세해짐 · "
                "빨강(−1 방향): 이 GO 기능은 질병에서 **손실 경향** — 기존 기능을 가진 아이소폼이 대체됨 · "
                "0 근처: 획득과 손실이 비슷한 규모로 일어나는 기능 (유전자마다 방향이 다름)"
            )

        # ── Chart ④: Gene × GO score delta heatmap ──────────────────────────
        pivot = build_consequence_pivot(conseq_df)

        if not pivot.empty:
            st.markdown("**④ 개별 유전자 × GO 기능 스코어 차이 히트맵**")
            top_n = st.slider("표시할 최대 유전자 수", 10, 80, 30, 5, key='mat_top_n')
            top_genes = pivot.abs().max(axis=1).nlargest(top_n).index
            pivot_view = pivot.loc[top_genes]

            fig_heat = px.imshow(
                pivot_view,
                color_continuous_scale=[[0, '#e63946'], [0.5, '#f8fafc'], [1, '#2a9d8f']],
                color_continuous_midpoint=0,
                zmin=-0.6, zmax=0.6,
                aspect='auto',
                title=f"|스코어 차이| 상위 {top_n} 유전자 × GO 기능 히트맵",
                labels={'color': 'score_delta\n(+: GAIN · −: LOSS)'},
                height=max(350, len(pivot_view) * 20),
            )
            fig_heat.update_traces(xgap=1, ygap=1)
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption(
                "각 셀 색상 = score_delta (질병 우세 아이소폼 PRISM 스코어 − 대체된 아이소폼 PRISM 스코어) · "
                "**초록(양수)**: 질병에서 우세해진 아이소폼이 이 GO 기능을 더 높이 예측받음 → GAIN · "
                "**빨강(음수)**: 대체된 아이소폼이 이 GO 기능을 더 높이 예측받았었음 → LOSS · "
                "흰색(0 근처): 두 아이소폼 간 GO 기능 차이 없음 (NEUTRAL) · "
                "Y축: |score_delta| 최댓값 기준 상위 유전자 정렬 (변화가 가장 큰 유전자가 위에)"
            )

        # ── Summary metrics ────────────────────────────────────────────────────
        st.markdown("---")
        n_gain = int((conseq_df['consequence'] == 'GAIN').sum())
        n_loss = int((conseq_df['consequence'] == 'LOSS').sum())
        n_neut = int((conseq_df['consequence'] == 'NEUTRAL').sum())
        n_total = n_gain + n_loss + n_neut
        g1, g2, g3 = st.columns(3)
        g1.metric(
            "GAIN 이벤트",  n_gain,
            help=f"질병 우세 아이소폼 PRISM 스코어 > 대체 아이소폼 PRISM 스코어 · 전체의 {n_gain/max(n_total,1):.0%}",
        )
        g2.metric(
            "LOSS 이벤트",  n_loss,
            help=f"질병 우세 아이소폼 PRISM 스코어 < 대체 아이소폼 PRISM 스코어 · 전체의 {n_loss/max(n_total,1):.0%}",
        )
        g3.metric(
            "NEUTRAL 이벤트", n_neut,
            help=f"두 아이소폼 간 스코어 차이 < 임계값 · 전체의 {n_neut/max(n_total,1):.0%}",
        )
        st.caption(
            f"이벤트 단위 = (유전자, GO term) 쌍 · 총 {n_total:,}개 · "
            f"GAIN이 압도적이면 이 조건이 새로운 기능을 활성화하는 방향, LOSS가 압도적이면 기존 기능 상실이 주요 변화"
        )
        render_condition_interpretation(n_gain, n_loss, n_neut)

        # Per-gene expander
        with st.expander("유전자별 상세 결과 보기 (GAIN/LOSS만)"):
            st.caption(
                "up_isoform = 질병 우세 아이소폼 (delta_IF > 0) · "
                "down_isoform = 대체된 아이소폼 (delta_IF < 0) · "
                "score_delta = up_score − down_score · "
                "양수 = 질병 우세 아이소폼이 이 GO 기능을 더 높이 가짐(GAIN), 음수 = 그 반대(LOSS)"
            )
            disp = conseq_df[conseq_df['consequence'].isin(['GAIN', 'LOSS'])][
                ['gene_id', 'condition', 'go_name', 'up_isoform', 'down_isoform',
                 'up_score', 'down_score', 'score_delta', 'consequence']
            ].sort_values('score_delta', key=abs, ascending=False)
            st.dataframe(disp, use_container_width=True, hide_index=True)

        csv_c = conseq_df.to_csv(index=False).encode('utf-8')
        st.download_button("결과 다운로드 (CSV)", csv_c,
                            "functional_consequences.csv", "text/csv")

# ── Tab 2: GO Enrichment ──────────────────────────────────────────────────────
with tab_enrich:
    st.subheader("DTU 스위치 아이소폼의 GO 기능 농집 분석 (GO Enrichment)")
    st.caption(
        "**질문**: DTU로 발현이 바뀐 아이소폼들이 특정 GO 기능에 집중되어 있는가? "
        "초기하 분포 검정(Hypergeometric test)으로 배경 대비 과대표현된 GO term을 탐지합니다. "
        "FDR 보정(Benjamini-Hochberg). "
        "**fold_enrichment**: 관측 비율 / 기대 비율 — 1보다 크면 우연보다 많이 관찰됨."
    )

    direction = st.radio(
        "분석할 아이소폼 방향",
        ['both', 'up', 'down'], horizontal=True,
        help="up = 질병에서 사용 증가(delta_IF > 0) · down = 사용 감소(delta_IF < 0) · both = 둘 다",
    )
    fdr_cut   = st.slider("FDR threshold for significance", 0.01, 0.5, 0.1, 0.01)

    enrich_df = _compute_enrichment(
        _dtu_bytes, _sm_bytes, sm.shape, _ids_list, _go_list,
        _gnames_json, pval_thr, thr, direction,
    )

    if enrich_df.empty:
        st.info("No enriched GO terms found. Try relaxing the DTU p-value threshold "
                "or the score threshold.")
    else:
        sig_enrich = enrich_df[enrich_df['FDR'] <= fdr_cut]
        st.metric("Significant GO terms (FDR ≤ threshold)", len(sig_enrich))

        # Bar chart
        plot_df = sig_enrich if not sig_enrich.empty else enrich_df.head(15)
        fig_e = px.bar(
            plot_df.head(15).sort_values('fold_enrichment', ascending=True),
            x='fold_enrichment', y='GO_term',
            orientation='h',
            color='FDR',
            color_continuous_scale='Blues_r',
            range_color=[0, fdr_cut * 2],
            title=f"GO enrichment: DTU {direction} isoforms (FDR ≤ {fdr_cut})",
            labels={'fold_enrichment': 'Fold enrichment', 'GO_term': ''},
            height=max(300, len(plot_df.head(15)) * 28),
            text='observed',
        )
        fig_e.update_traces(texttemplate='n=%{text}', textposition='outside')
        fig_e.update_layout(yaxis_tickfont_size=11, coloraxis_colorbar_title='FDR')
        st.plotly_chart(fig_e, use_container_width=True)
        st.caption(
            "X축: fold_enrichment (관측 아이소폼 수 / 기대 아이소폼 수) — 1.0 이상이면 이 GO 기능이 DTU 아이소폼에 우연보다 많이 집중됨 · "
            "색: FDR (진한 파랑 = 낮은 FDR = 더 유의미) · 막대 끝 숫자(n=): 실제 관측된 아이소폼 수 · "
            "fold_enrichment가 크더라도 n이 매우 작으면 통계적 신뢰도 제한됨"
        )

        # Full table
        st.dataframe(
            enrich_df[['GO_ID', 'GO_term', 'observed', 'expected',
                        'fold_enrichment', 'pvalue', 'FDR']].round(5),
            use_container_width=True, hide_index=True,
        )

        csv_e = enrich_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download enrichment results (CSV)", csv_e,
                            "go_enrichment_dtu.csv", "text/csv")

# ── Tab 3: Sankey — scenario flow ─────────────────────────────────────────────
with tab_sankey:
    st.subheader("조건 간 시나리오 전환 흐름도 (Sankey)")
    st.caption(
        "두 조건(예: 정상 vs 질병) 간에 아이소폼들이 4-시나리오 분류에서 어떻게 이동하는지 보여줍니다. "
        "**밴드 두께**: 해당 경로를 따라 이동한 아이소폼 수 · "
        "S1(기능 스위치) → S1 유지이면 양쪽 조건 모두 DTU+GO 변화 · "
        "S4(배경) → S1 이동이면 조건 변화로 기능 스위치가 새로 활성화됨을 의미합니다."
    )

    from prism_app.pipeline.dtu_connector import parse_dtu_result
    try:
        parsed_dtu = parse_dtu_result(dtu_df.copy())
    except ValueError as e:
        st.error(f"DTU parse error: {e}")
        parsed_dtu = None

    if parsed_dtu is not None and 'condition' in parsed_dtu.columns \
            and parsed_dtu['condition'].notna().any():

        # Merge DTU with scenario labels
        merged_sc = parsed_dtu.merge(
            classified[['isoform_id', 'scenario', 'scenario_label', 'gene_id']],
            on='isoform_id', how='left',
        )
        merged_sc['pvalue'] = merged_sc.get('pvalue', pd.Series(dtype=float))
        if 'pvalue' in merged_sc.columns and merged_sc['pvalue'].notna().any():
            sig_sc = merged_sc[merged_sc['pvalue'] <= pval_thr]
        else:
            sig_sc = merged_sc

        cond_vals = sig_sc['condition'].dropna().unique()
        if len(cond_vals) >= 2:
            cond_a = st.selectbox("Condition A (source)", sorted(cond_vals), index=0)
            cond_b = st.selectbox("Condition B (target)", sorted(cond_vals),
                                   index=min(1, len(cond_vals) - 1))

            sc_labels = {1: 'S1: Func Switch', 2: 'S2: Expr Switch',
                         3: 'S3: Const Novel', 4: 'S4: Background'}
            COLORS = ['#e63946', '#f4a261', '#2a9d8f', '#adb5bd']

            grp_a = sig_sc[sig_sc['condition'] == cond_a]['scenario'].value_counts()
            grp_b = sig_sc[sig_sc['condition'] == cond_b]['scenario'].value_counts()

            node_labels = [f"{sc_labels.get(s, f'S{s}')} [{cond_a}]" for s in [1,2,3,4]] + \
                          [f"{sc_labels.get(s, f'S{s}')} [{cond_b}]" for s in [1,2,3,4]]

            # All isoforms in condition A that also appear in condition B
            isos_a = set(sig_sc[sig_sc['condition'] == cond_a]['isoform_id'])
            isos_b = set(sig_sc[sig_sc['condition'] == cond_b]['isoform_id'])
            shared  = isos_a & isos_b

            source, target, value, link_labels = [], [], [], []
            for i, s in enumerate([1, 2, 3, 4]):
                for j, t in enumerate([1, 2, 3, 4]):
                    # Isoforms that are scenario s in cond_a AND scenario t in cond_b
                    a_df = sig_sc[(sig_sc['condition'] == cond_a) & (sig_sc['scenario'] == s)]
                    b_df = sig_sc[(sig_sc['condition'] == cond_b) & (sig_sc['scenario'] == t)]
                    v = len(set(a_df['isoform_id']) & set(b_df['isoform_id']) & shared)
                    if v > 0:
                        source.append(i)
                        target.append(j + 4)
                        value.append(v)
                        link_labels.append(f"{v} isoforms")

            if source:
                fig_s = pgo.Figure(pgo.Sankey(
                    arrangement='freeform',
                    node=dict(
                        pad=18, thickness=22,
                        label=node_labels,
                        color=COLORS * 2,
                        hovertemplate='%{label}<br>%{value} isoforms<extra></extra>',
                    ),
                    link=dict(
                        source=source, target=target, value=value,
                        label=link_labels,
                        color='rgba(150,150,150,0.3)',
                    ),
                ))
                fig_s.update_layout(
                    title_text=f"Scenario transitions: {cond_a} → {cond_b}  "
                               f"({len(shared)} shared DTU isoforms)",
                    height=450,
                )
                st.plotly_chart(fig_s, use_container_width=True)

                st.caption(
                    "Only isoforms with DTU records in *both* conditions are shown. "
                    "S1/S2 require DTU; S3/S4 are score-only."
                )
            else:
                st.info("No isoforms appear in both conditions.")
        else:
            st.info(
                f"Sankey requires ≥ 2 conditions. "
                f"Found only: {list(cond_vals)}. "
                "Check that the DTU file has a 'condition' column."
            )
    else:
        st.info(
            "No 'condition' column found in the DTU file. "
            "Sankey diagram requires a condition label per row."
        )

# ── Tab 4: Gene-Level Detail ──────────────────────────────────────────────────
with tab_genes:
    st.subheader("유전자별 상세 분석 — 아이소폼 스위치와 GO 기능 변화")
    st.caption(
        "특정 유전자를 검색하면 해당 유전자 내 아이소폼 스위치의 방향과 영향을 확인합니다. "
        "**막대 방향**: 오른쪽(양수) = 질병 우세 아이소폼이 이 GO 기능을 더 높이 가짐(GAIN), "
        "왼쪽(음수) = 대체된 아이소폼이 더 높게 가졌었음(LOSS). "
        "**막대 길이**: 두 아이소폼 간 PRISM 스코어 차이의 크기 (0~1 범위)."
    )

    gene_query = st.text_input("Gene symbol or ID", placeholder="e.g. NDUFS4, KIF21B, DLG1")

    if gene_query and not conseq_df.empty:
        gene_hits = conseq_df[
            conseq_df['gene_id'].str.contains(gene_query, case=False, na=False)
        ] if 'gene_id' in conseq_df.columns else pd.DataFrame()

        if gene_hits.empty:
            # Try matching via classified_df
            clf_hits = classified[
                classified['gene_id'].str.contains(gene_query, case=False, na=False)
                | classified['isoform_id'].str.contains(gene_query, case=False, na=False)
            ]
            if not clf_hits.empty:
                st.write(f"**{len(clf_hits)} isoforms** for gene '{gene_query}':")
                st.dataframe(
                    clf_hits[['isoform_id', 'gene_id', 'max_score', 'max_go',
                               'dtu_flag', 'scenario_label']],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.warning(f"No isoforms matching '{gene_query}'.")
        else:
            st.write(f"**{len(gene_hits)} consequence records** for '{gene_query}':")

            # Per-GO consequence table
            c_table = gene_hits[['go_name', 'condition', 'up_isoform', 'down_isoform',
                                  'up_score', 'down_score', 'score_delta', 'consequence']]
            st.dataframe(
                c_table.sort_values('score_delta', key=abs, ascending=False),
                use_container_width=True, hide_index=True,
            )

            # Bar chart: score delta per GO term
            fig_gene = px.bar(
                gene_hits.sort_values('score_delta'),
                x='score_delta', y='go_name',
                orientation='h',
                color='consequence',
                color_discrete_map={'GAIN': '#2a9d8f', 'LOSS': '#e63946', 'NEUTRAL': '#adb5bd',
                                    'UNKNOWN': '#cccccc'},
                title=f"{gene_query} — GO 기능별 변화 방향 (score_delta = 질병 우세 아이소폼 − 대체 아이소폼)",
                labels={'score_delta': 'score_delta  (오른쪽: GAIN · 왼쪽: LOSS)', 'go_name': 'GO 기능'},
                height=max(300, len(gene_hits) * 26),
            )
            fig_gene.add_vline(x=0, line_color='#374151', line_width=1.5)
            st.plotly_chart(fig_gene, use_container_width=True)
            st.caption(
                f"유전자 {gene_query}의 DTU 이벤트에서: "
                "초록 막대(GAIN) = 질병 아이소폼이 이 GO 기능을 더 높이 예측받음 → 기능 획득 · "
                "빨강 막대(LOSS) = 정상 아이소폼이 이 GO 기능을 더 높이 예측받았었음 → 기능 상실 · "
                "막대 길이 = score_delta 절댓값 (클수록 두 아이소폼 간 기능 차이가 뚜렷함)"
            )
    elif gene_query:
        st.info("No consequence data available. Check that DTU data is loaded.")

# ── Tab 5: Full Results Table ─────────────────────────────────────────────────
with tab_table:
    st.subheader("Merged DTU + PRISM Consequence Results")

    view_mode = st.radio("View", ['Consequence matrix', 'Scenario table'], horizontal=True)

    if view_mode == 'Consequence matrix':
        if not conseq_df.empty:
            disp = conseq_df[['gene_id', 'condition', 'go_name', 'up_isoform',
                               'down_isoform', 'up_score', 'down_score',
                               'score_delta', 'consequence']]
            st.dataframe(disp, use_container_width=True, hide_index=True)
            csv1 = disp.to_csv(index=False).encode('utf-8')
            st.download_button("Download consequence table (CSV)", csv1,
                                "dtu_consequences.csv", "text/csv")
        else:
            st.info("No consequence data — relax thresholds.")
    else:
        # Scenario table merged with DTU
        from prism_app.pipeline.dtu_connector import parse_dtu_result
        try:
            parsed = parse_dtu_result(dtu_df.copy())
            merged_full = parsed.merge(
                classified[['isoform_id', 'gene_id', 'scenario', 'scenario_label',
                             'max_score', 'max_go', 'novel_go_flag']],
                on='isoform_id', how='left',
            )
            disp_cols = [c for c in ['isoform_id', 'gene_id', 'condition', 'delta_IF',
                                      'pvalue', 'scenario', 'scenario_label',
                                      'max_score', 'max_go', 'novel_go_flag']
                         if c in merged_full.columns]
            st.dataframe(merged_full[disp_cols], use_container_width=True, hide_index=True)
            csv2 = merged_full[disp_cols].to_csv(index=False).encode('utf-8')
            st.download_button("Download scenario table (CSV)", csv2,
                                "dtu_scenarios.csv", "text/csv")
        except ValueError as e:
            st.error(f"DTU parse error: {e}")

# ── Next Step Banner ─────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='background:linear-gradient(90deg,#fdf4ff,#fae8ff);border-radius:10px;
padding:16px 24px;border-left:4px solid #9333ea;margin-top:16px'>
<b>다음 단계: 🎯 Target Analysis</b><br>
<span style='color:#374151;font-size:0.9rem'>
조건별 모듈 전환 유전자를 확인했다면 → 사이드바 Gene Search에 유전자명 입력 →
Target Analysis 페이지에서 이소폼별 모듈 스코어 + DTU 히트맵으로 메커니즘을 검증하세요.
</span>
</div>
""", unsafe_allow_html=True)
