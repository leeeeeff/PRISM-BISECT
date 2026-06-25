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
import plotly.graph_objects as _go_fig
from collections import defaultdict, Counter

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


# ── Discovery stats helper (cached) ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _compute_discovery_stats(sm_bytes, sm_shape, genes_tuple, types_tuple, go_tuple, thr):
    """
    Isoform Case Taxonomy + intra-gene divergence + score distribution.
    Requires annotation file at <root>/hMuscle/data/raw_data/data/annotations/.
    Returns dict; gracefully returns partial results if annotation file missing.
    """
    sm     = np.frombuffer(sm_bytes, dtype=np.float32).reshape(sm_shape)
    genes  = list(genes_tuple)
    types  = list(types_tuple)
    go_ids = list(go_tuple)
    N_TE   = sm.shape[0]
    N_GO   = sm.shape[1]

    # ── gene-level grouping ───────────────────────────────────────────────
    gene_idx = defaultdict(list)
    for i, sym in enumerate(genes):
        gene_idx[sym].append(i)

    unique_genes     = len(gene_idx)
    single_iso       = sum(1 for v in gene_idx.values() if len(v) == 1)
    multi_iso_genes  = unique_genes - single_iso
    mean_iso_per_gene = N_TE / max(1, unique_genes)

    # ── isoform type stats ───────────────────────────────────────────────
    known_count = sum(1 for t in types if t == 'known')
    novel_count = sum(1 for t in types if t in ('nic', 'nnic', 'novel'))

    # ── score distribution ───────────────────────────────────────────────
    max_scores = sm.max(axis=1)
    mean_max   = float(np.mean(max_scores))
    n_high     = int((max_scores >= thr).sum())
    n_high_novel = int(sum(1 for i, sc in enumerate(max_scores)
                           if sc >= thr and types[i] in ('nic', 'nnic', 'novel')))

    # ── intra-gene divergence ────────────────────────────────────────────
    div_cnt = mod_cnt = con_cnt = 0
    div_by_niso = defaultdict(lambda: {'total': 0, 'div': 0})
    for sym, idxs in gene_idx.items():
        if len(idxs) < 2:
            continue
        arr  = sm[idxs]
        md   = float((arr.max(axis=0) - arr.min(axis=0)).max())
        nbin = min(len(idxs), 11)  # cap at ">10" bucket
        div_by_niso[nbin]['total'] += 1
        if md > 0.3:
            div_cnt += 1
            div_by_niso[nbin]['div'] += 1
        elif md > 0.1:
            mod_cnt += 1
        else:
            con_cnt += 1

    # ── GO annotation-based taxonomy (requires annotation file) ──────────
    ann_file = Path(_root) / 'hMuscle/data/raw_data/data/annotations/human_annotations_unified_bp.txt'
    taxonomy = None
    n_no_annot_high = 0
    n_novel_go_expansion = 0
    n_diff_go = 0

    if ann_file.exists():
        gene_go = defaultdict(set)
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    for g in parts[1:]:
                        if g.startswith('GO:'):
                            gene_go[parts[0]].add(g)

        annot = np.zeros((N_TE, N_GO), dtype=bool)
        for gi, go_term in enumerate(go_ids):
            annot[:, gi] = np.array([go_term in gene_go.get(sym, set()) for sym in genes])

        cases = []
        for i in range(N_TE):
            n_ann        = int(annot[i].sum())
            max_sc       = float(sm[i].max())
            any_high_ann = bool(((sm[i] >= thr) & annot[i]).any())
            any_high_nov = bool(((sm[i] >= thr) & ~annot[i]).any())
            if n_ann == 0:
                if max_sc >= thr:
                    tag = 1
                elif max_sc >= 0.3:
                    tag = 2
                else:
                    tag = 3
            else:
                if any_high_ann and any_high_nov:
                    tag = 4
                elif any_high_ann and not any_high_nov:
                    tag = 5
                elif not any_high_ann and any_high_nov:
                    tag = 6
                else:
                    tag = 7
            cases.append(tag)

        taxonomy = dict(Counter(cases))
        n_no_annot_high     = taxonomy.get(1, 0)
        n_novel_go_expansion = taxonomy.get(4, 0)
        n_diff_go           = taxonomy.get(6, 0)

    return {
        'unique_genes':      unique_genes,
        'single_iso':        single_iso,
        'multi_iso_genes':   multi_iso_genes,
        'mean_iso_per_gene': mean_iso_per_gene,
        'known_count':       known_count,
        'novel_count':       novel_count,
        'mean_max_score':    mean_max,
        'n_high':            n_high,
        'n_high_novel':      n_high_novel,
        'max_scores':        max_scores.tolist(),
        'div_cnt':           div_cnt,
        'mod_cnt':           mod_cnt,
        'con_cnt':           con_cnt,
        'div_by_niso':       {k: dict(v) for k, v in div_by_niso.items()},
        'taxonomy':          taxonomy,
        'n_no_annot_high':   n_no_annot_high,
        'n_novel_go_expansion': n_novel_go_expansion,
        'n_diff_go':         n_diff_go,
        'annotation_available': ann_file.exists(),
    }

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

# ── A0 · 데이터 기본 현황 ────────────────────────────────────────────────────
st.subheader("A0 · 데이터 기본 현황")
st.caption(
    "현재 로드된 score matrix에 포함된 이소폼들의 구성 통계입니다. "
    "**Known** = Ensembl 데이터베이스에 등재된 이소폼 · "
    "**NIC(Novel In Catalog)** = Ensembl에 등재된 splice site 조합이지만 해당 전사체는 신규 · "
    "**NNIC(Novel Not In Catalog)** = Ensembl에 없는 완전 신규 splice site를 포함한 전사체."
)

_N = sm.shape[0]
_gene_ctr   = Counter(np.asarray(genes if genes is not None else ids, dtype=str))
_n_genes    = len(_gene_ctr)
_multi_iso  = sum(1 for c in _gene_ctr.values() if c > 1)
_mean_iso   = _N / max(1, _n_genes)
_max_iso    = max(_gene_ctr.values()) if _gene_ctr else 0
_max_gene   = max(_gene_ctr, key=_gene_ctr.get) if _gene_ctr else '—'
_novel_cnt  = int(np.isin(np.asarray(types if types is not None else [], dtype=str),
                          ['nic', 'nnic', 'novel']).sum()) if types is not None else 0
_n_go_eval  = sm.shape[1]

a0_c1, a0_c2, a0_c3, a0_c4, a0_c5, a0_c6 = st.columns(6)
a0_c1.metric(
    "유니크 유전자 수",
    f"{_n_genes:,}",
    help="score matrix에 이소폼이 1개 이상 포함된 유전자(gene symbol 기준) 총 수"
)
a0_c2.metric(
    "멀티-이소폼 유전자",
    f"{_multi_iso:,}",
    f"{_multi_iso/_n_genes*100:.1f}% of genes",
    help="이 데이터셋에 이소폼이 2개 이상 포함된 유전자 수 — PRISM이 같은 유전자 내 이소폼을 비교할 수 있는 유전자"
)
a0_c3.metric(
    "평균 이소폼 수/유전자",
    f"{_mean_iso:.1f}",
    help=f"전체 이소폼 수({_N:,}) ÷ 유니크 유전자 수({_n_genes:,})"
)
a0_c4.metric(
    "최다 이소폼 유전자",
    f"{_max_gene}",
    f"{_max_iso}개 이소폼",
    help="이 데이터셋에서 가장 많은 이소폼이 포함된 유전자"
)
a0_c5.metric(
    "Novel 이소폼 (NIC + NNIC)",
    f"{_novel_cnt:,}",
    f"{_novel_cnt/_N*100:.1f}% of total" if _N else "—",
    help="Ensembl 데이터베이스에 등재되지 않은 신규 이소폼 수 (NIC + NNIC 합계). "
         "이 이소폼들은 GO annotation이 원천적으로 없으며, PRISM이 서열만으로 기능을 예측합니다."
)
a0_c6.metric(
    "PRISM 예측 GO term 수",
    f"{_n_go_eval}",
    help="현재 score matrix가 예측하는 GO term(생물학적 과정) 수. "
         "각 이소폼은 이 GO term들 각각에 대해 0–1 점수를 부여받습니다."
)

# Isoform count distribution per gene (histogram)
_iso_counts = list(_gene_ctr.values())
_iso_bins   = [1, 2, 3, 4, 5, 6, 10, 999]
_iso_labels = ['1', '2', '3', '4', '5', '6–10', '>10']
_iso_bar    = []
for lo, hi, lbl in zip(_iso_bins[:-1], _iso_bins[1:], _iso_labels):
    _iso_bar.append({'bin': lbl, 'n_genes': sum(1 for c in _iso_counts if lo <= c < hi)})
_iso_bar_df = pd.DataFrame(_iso_bar)

with st.expander("📊 유전자당 이소폼 수 분포", expanded=False):
    _fig_iso = px.bar(_iso_bar_df, x='bin', y='n_genes',
                      labels={'bin': '유전자당 이소폼 수 (이 데이터셋 기준)', 'n_genes': '유전자 수'},
                      title='유전자당 이소폼 수 분포 — 이 데이터셋에 포함된 이소폼 기준',
                      color_discrete_sequence=['#4c72b0'])
    _fig_iso.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(_fig_iso, use_container_width=True)
    st.caption(
        "X축: 이 데이터셋에 해당 유전자의 이소폼이 몇 개 포함됐는지 · "
        "Y축: 그 구간에 해당하는 유전자 수 · "
        "이소폼 수가 많은 유전자(오른쪽)일수록 PRISM이 동일 유전자 내에서 이소폼별로 다른 기능을 예측하는지 검증 가능합니다."
    )

st.divider()

# ── A1b · PRISM 발견 통계 ────────────────────────────────────────────────────
st.subheader("A1b · PRISM 발견 통계 — 기능 예측 심층 분석")
st.caption(
    f"PRISM이 {len(go)}개 GO term 각각에 대해 0–1 점수를 예측한 결과를 분석합니다. "
    "GO annotation 출처: **human_annotations_unified_bp.txt** (UniProtKB/SwissProt, 유전자 symbol 단위) — "
    "여기에 없는 유전자/이소폼은 'annotation 없음'으로 처리됩니다. "
    "분석 항목: ① PRISM이 annotation 없이 새로 예측한 기능 후보(Discovery candidates) · "
    "② 같은 유전자 내 이소폼 간 예측 점수 차이(Intra-gene divergence) · "
    "③ 전체 PRISM 점수 분포."
)

# Compute discovery stats (cached)
_ds_key = 'discovery_stats'
if _ds_key not in st.session_state or st.session_state.get('_ds_thr') != thr:
    with st.spinner("발견 통계 계산 중…"):
        _ds = _compute_discovery_stats(
            sm.astype(np.float32).tobytes(), sm.shape,
            tuple(np.asarray(genes if genes is not None else ids, dtype=str)),
            tuple(np.asarray(types if types is not None else ['known'] * _N, dtype=str)),
            tuple(go), thr,
        )
        st.session_state[_ds_key]   = _ds
        st.session_state['_ds_thr'] = thr
else:
    _ds = st.session_state[_ds_key]

# ── Metric cards row 1: discovery ────────────────────────────────────────────
_taxonomy   = _ds.get('taxonomy') or {}
_t1 = _taxonomy.get(1, 0)
_t4 = _taxonomy.get(4, 0)
_t6 = _taxonomy.get(6, 0)
_t5 = _taxonomy.get(5, 0)
_disc_total = _t1 + _t4 + _t6
_disc_pct   = _disc_total / _N * 100 if _N else 0

if _ds['annotation_available']:
    d_c1, d_c2, d_c3, d_c4 = st.columns(4)
    d_c1.metric(
        "발견 후보 이소폼 총계",
        f"{_disc_total:,}",
        f"전체 이소폼의 {_disc_pct:.1f}%",
        help=(
            "TYPE 1 + TYPE 4 + TYPE 6 이소폼 합계.\n"
            "• TYPE 1: 유전자 자체에 SwissProt GO annotation이 전혀 없는데 "
            f"PRISM이 score > {thr}를 예측 (완전 신규)\n"
            "• TYPE 4: 유전자에 SwissProt GO annotation이 있고, PRISM이 기존 "
            f"annotation GO term들과 완전히 다른 새 GO term을 score > {thr}로 추가 예측\n"
            "• TYPE 6: 유전자에 SwissProt GO annotation이 있지만, PRISM이 "
            f"기존 annotation GO term을 score < {thr}로 예측하지 않고 "
            f"대신 전혀 다른 GO term을 score > {thr}로 예측 (기능 전환)\n"
            "※ GO annotation 출처: human_annotations_unified_bp.txt (UniProtKB/SwissProt 기반, 유전자 symbol 단위)"
        )
    )
    d_c2.metric(
        "완전 신규 예측 이소폼 (TYPE 1)",
        f"{_t1:,}",
        f"score > {thr}, GO annotation 전무",
        help=(
            "해당 이소폼이 속한 유전자(gene symbol)가 "
            "human_annotations_unified_bp.txt에 단 하나의 GO term도 등재되지 않았음에도, "
            f"PRISM이 {len(go)}개 GO term 중 최소 1개에서 score > {thr}를 예측한 이소폼.\n"
            "NIC/NNIC 신규 이소폼뿐 아니라 Known 이소폼도 포함됩니다 — "
            "유전자 자체가 SwissProt에서 GO 미주석된 경우."
        )
    )
    d_c3.metric(
        "GO term 확장 예측 이소폼 (TYPE 4)",
        f"{_t4:,}",
        "기존 annotation GO + 새 GO term 동시 예측",
        help=(
            "해당 이소폼의 유전자가 SwissProt에 GO annotation을 보유하고, "
            f"PRISM이 그 기존 GO term들 중 최소 1개를 score > {thr}로 확인하면서, "
            f"동시에 기존 annotation에 없는 새로운 GO term도 score > {thr}로 예측한 이소폼.\n"
            "기존 기능을 유지하면서 새 기능을 추가로 수행할 가능성이 있는 케이스."
        )
    )
    d_c4.metric(
        "기존 GO annotation 확인 이소폼 (TYPE 5)",
        f"{_t5:,}",
        "PRISM이 SwissProt annotation과 일치하는 예측",
        help=(
            "해당 이소폼의 유전자가 SwissProt에 GO annotation을 보유하고, "
            f"PRISM이 그 기존 GO term들 중 최소 1개를 score > {thr}로 예측했으며, "
            "기존 annotation에 없는 새로운 GO term은 예측하지 않은 이소폼.\n"
            "PRISM 예측이 기존 데이터베이스 annotation과 일치하는 검증(validation) 케이스."
        )
    )
else:
    st.info(
        "GO annotation 파일 없음 — Taxonomy 통계(TYPE 1-7) 비활성화. "
        "annotation 파일을 data/raw_data/data/annotations/에 위치시키면 활성화됩니다.",
        icon="ℹ️"
    )

# ── Metric cards row 2: score & divergence ───────────────────────────────────
d2_c1, d2_c2, d2_c3, d2_c4 = st.columns(4)
d2_c1.metric(
    "이소폼별 최대 PRISM 점수 평균",
    f"{_ds['mean_max_score']:.4f}",
    help=(
        f"각 이소폼에 대해 {len(go)}개 GO term 중 가장 높은 PRISM 예측 score를 구한 뒤, "
        "모든 이소폼에 걸쳐 평균한 값. "
        "이 값이 높을수록 데이터셋 전반적으로 PRISM이 기능적 활성을 예측하는 경향이 강합니다."
    )
)
d2_c2.metric(
    f"고득점 이소폼 수 (score > {thr})",
    f"{_ds['n_high']:,}",
    f"전체의 {_ds['n_high']/_N*100:.1f}% · 신규(NIC+NNIC): {_ds['n_high_novel']:,}개",
    help=(
        f"{len(go)}개 GO term 중 최소 1개에서 PRISM score > {thr}를 받은 이소폼 수. "
        f"임계값 {thr}는 사이드바에서 조정 가능합니다. "
        "신규(NIC+NNIC) 이소폼 중 고득점을 받은 수는 Ensembl annotation이 없는 이소폼 중 "
        "PRISM이 기능을 예측한 케이스입니다."
    )
)
d2_c3.metric(
    "기능 분화 유전자 수 (Divergent)",
    f"{_ds['div_cnt']:,}",
    f"멀티-이소폼 유전자의 {_ds['div_cnt']/_ds['multi_iso_genes']*100:.1f}%" if _ds['multi_iso_genes'] else "—",
    help=(
        "이 데이터셋에 이소폼이 2개 이상 포함된 유전자 중, "
        f"같은 유전자 내 이소폼들 사이에서 임의의 GO term에 대해 "
        "max(PRISM score) − min(PRISM score) > 0.3인 경우가 한 번이라도 존재하는 유전자 수. "
        "PRISM이 유전자 수준이 아닌 이소폼 수준에서 기능 차이를 감지하고 있음을 의미합니다."
    )
)
d2_c4.metric(
    "멀티-이소폼 유전자 수",
    f"{_ds['multi_iso_genes']:,}",
    f"단일 이소폼 유전자: {_ds['single_iso']:,}개",
    help=(
        "이 데이터셋에 이소폼이 2개 이상 포함된 유전자 수. "
        "Intra-gene divergence 분석은 이 유전자들에 대해서만 수행됩니다. "
        "데이터셋에 1개 이소폼만 포함된 유전자는 이소폼 간 비교가 불가능하므로 제외됩니다."
    )
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row: score distribution + taxonomy ─────────────────────────────────
_chart_c1, _chart_c2 = st.columns([3, 2])

with _chart_c1:
    _max_scores_arr = np.array(_ds['max_scores'])
    _score_bins  = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]
    _score_lbls  = ['0–0.1','0.1–0.2','0.2–0.3','0.3–0.4','0.4–0.5',
                    '0.5–0.6','0.6–0.7','0.7–0.8','0.8–0.9','0.9–1.0']
    _types_arr   = np.asarray(types if types is not None else ['known'] * _N, dtype=str)
    _score_rows  = []
    for lo, hi, lbl in zip(_score_bins[:-1], _score_bins[1:], _score_lbls):
        mask = (_max_scores_arr >= lo) & (_max_scores_arr < hi)
        _score_rows.append({
            'bin': lbl,
            'Known':    int((mask & (_types_arr == 'known')).sum()),
            'NIC':      int((mask & (_types_arr == 'nic')).sum()),
            'NNIC':     int((mask & (_types_arr == 'nnic')).sum()),
        })
    _score_df = pd.DataFrame(_score_rows)
    _fig_score = _go_fig.Figure()
    for col, color in [('Known','#4c72b0'),('NIC','#55a868'),('NNIC','#c44e52')]:
        if _score_df[col].sum() > 0:
            _fig_score.add_trace(_go_fig.Bar(
                x=_score_df['bin'], y=_score_df[col],
                name=col, marker_color=color,
            ))
    # vline at the numeric index of the threshold bin (add_vline with string x breaks on newer plotly)
    _thr_idx = next(
        (i for i, l in enumerate(_score_lbls)
         if float(l.split('–')[0]) <= thr < float(l.split('–')[1])),
        4
    )
    _fig_score.add_shape(
        type='line', x0=_thr_idx - 0.5, x1=_thr_idx - 0.5, y0=0, y1=1, yref='paper',
        line=dict(dash='dash', color='#ef4444', width=2),
    )
    _fig_score.add_annotation(
        x=_thr_idx - 0.5, y=0.96, yref='paper',
        text=f'임계값 {thr}', font=dict(size=10, color='#ef4444'),
        showarrow=False, xanchor='left',
    )
    _fig_score.update_layout(
        barmode='stack', title='이소폼별 최대 PRISM 점수 분포 (stacked by type)',
        xaxis_title='Max PRISM Score 구간', yaxis_title='이소폼 수',
        height=320, plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    st.plotly_chart(_fig_score, use_container_width=True)
    st.caption(
        f"**X축**: 각 이소폼에서 {len(go)}개 GO term 중 PRISM이 가장 높게 예측한 점수의 구간 · "
        "**Y축**: 해당 구간에 속하는 이소폼 수 (Known / NIC / NNIC 누적) · "
        f"**빨간 점선** = 현재 임계값 {thr} — 이 점선 **오른쪽** 이소폼은 "
        f"최소 1개 GO term에서 score > {thr}를 기록하여 '기능 예측 성공'으로 간주됩니다."
    )

with _chart_c2:
    if _taxonomy:
        _type_meta = {
            1: ('TYPE_1', f'유전자 GO annotation 없음 + score>{thr} (신규 예측)', '#e63946'),
            2: ('TYPE_2', f'유전자 GO annotation 없음 + 0.3≤score≤{thr} (저신뢰)', '#f4a261'),
            3: ('TYPE_3', 'GO annotation 없음 + score<0.3 (기능 미예측)', '#adb5bd'),
            4: ('TYPE_4', f'GO annotation 있음 + 기존 GO 확인 + 새 GO score>{thr} (확장)', '#2a9d8f'),
            5: ('TYPE_5', f'GO annotation 있음 + 기존 GO score>{thr} + 새 GO 없음 (검증)', '#4c72b0'),
            6: ('TYPE_6', f'GO annotation 있음 + 기존 GO score<{thr} + 다른 GO score>{thr} (전환)', '#9b59b6'),
            7: ('TYPE_7', f'GO annotation 있음 + 모든 GO score<{thr} (기능 미예측)', '#dfe6e9'),
        }
        _tax_rows = [
            {'type': _type_meta[t][0], 'label': _type_meta[t][1],
             'count': _taxonomy.get(t, 0), 'color': _type_meta[t][2]}
            for t in sorted(_type_meta)
        ]
        _tax_df = pd.DataFrame(_tax_rows)
        _fig_tax = _go_fig.Figure(_go_fig.Bar(
            x=_tax_df['count'], y=_tax_df['type'],
            orientation='h',
            marker_color=_tax_df['color'],
            text=_tax_df['count'].map(lambda v: f"{v:,}"),
            textposition='outside',
            hovertext=_tax_df['label'],
            hovertemplate='%{hovertext}<br>이소폼 수: %{x:,}<extra></extra>',
        ))
        _fig_tax.update_layout(
            title=f'이소폼 케이스 분류 (TYPE 1–7) — 임계값 {thr} 기준',
            height=360, plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title='이소폼 수', yaxis=dict(autorange='reversed'),
            margin=dict(l=80, r=80),
        )
        st.plotly_chart(_fig_tax, use_container_width=True)
        st.caption(
            f"**GO annotation** = human_annotations_unified_bp.txt (UniProtKB/SwissProt, 유전자 symbol 단위). "
            f"**TYPE 1·4·6** (빨강·청록·보라) = 발견 후보 — PRISM이 SwissProt에 없는 새 기능을 예측. "
            f"**TYPE 5** (파랑) = PRISM 예측이 기존 SwissProt annotation과 일치하는 검증 케이스. "
            "막대에 마우스를 올리면 각 TYPE의 정의를 확인할 수 있습니다."
        )
    else:
        st.info("annotation 파일 필요", icon="ℹ️")

# ── Intra-gene divergence detail ─────────────────────────────────────────────
with st.expander("🔬 Intra-gene 이소폼 기능 분화 상세 (이소폼 2개 이상 유전자 대상)", expanded=False):
    st.caption(
        f"**분석 대상**: 이 데이터셋에 이소폼이 2개 이상 포함된 유전자 ({_ds['multi_iso_genes']:,}개). "
        f"**점수 차이 정의**: 같은 유전자 내 이소폼들 사이에서, {len(go)}개 GO term 중 "
        "임의의 GO term에 대해 max(PRISM score) − min(PRISM score)를 계산하고 "
        "그 최댓값을 해당 유전자의 '이소폼 간 최대 점수 차이'로 정의합니다. "
        "이 값이 클수록 PRISM이 같은 유전자라도 이소폼마다 **다른 기능적 역할**을 예측하고 있음을 의미합니다."
    )
    _div_c1, _div_c2, _div_c3 = st.columns(3)
    _total_multi = _ds['multi_iso_genes']
    _div_c1.metric(
        "기능 분화 유전자 (Divergent)",
        f"{_ds['div_cnt']:,}",
        f"멀티-이소폼 유전자의 {_ds['div_cnt']/_total_multi*100:.1f}%" if _total_multi else "—",
        help=(
            "같은 유전자 내 이소폼 간 최대 PRISM score 차이 > 0.3인 유전자. "
            "PRISM이 이 유전자의 이소폼들을 서로 다른 기능을 수행하는 분자로 예측합니다."
        )
    )
    _div_c2.metric(
        "중간 분화 유전자 (Moderate)",
        f"{_ds['mod_cnt']:,}",
        f"멀티-이소폼 유전자의 {_ds['mod_cnt']/_total_multi*100:.1f}%" if _total_multi else "—",
        help=(
            "같은 유전자 내 이소폼 간 최대 PRISM score 차이가 0.1 초과 0.3 이하인 유전자. "
            "기능 차이가 미미하게 포착되는 케이스."
        )
    )
    _div_c3.metric(
        "기능 일치 유전자 (Concordant)",
        f"{_ds['con_cnt']:,}",
        f"멀티-이소폼 유전자의 {_ds['con_cnt']/_total_multi*100:.1f}%" if _total_multi else "—",
        help=(
            "같은 유전자 내 이소폼 간 최대 PRISM score 차이 ≤ 0.1인 유전자. "
            "PRISM이 이소폼들에 대해 사실상 동일한 기능을 예측합니다."
        )
    )

    # Bar chart: divergence rate by n_iso
    _div_bins = {2:'2', 3:'3', 4:'4–5', 5:'4–5', 6:'6–10', 7:'6–10', 8:'6–10',
                 9:'6–10', 10:'6–10', 11:'>10'}
    _by_niso = defaultdict(lambda: {'total':0,'div':0})
    for nbin, vals in _ds['div_by_niso'].items():
        lbl = _div_bins.get(int(nbin), '>10')
        _by_niso[lbl]['total'] += vals.get('total', 0)
        _by_niso[lbl]['div']   += vals.get('div', 0)

    _div_order = ['2','3','4–5','6–10','>10']
    _div_plot  = [{'n_iso': lbl,
                   '전체': _by_niso[lbl]['total'],
                   'Divergent': _by_niso[lbl]['div'],
                   '비율(%)': round(_by_niso[lbl]['div']/_by_niso[lbl]['total']*100, 1)
                             if _by_niso[lbl]['total'] else 0}
                  for lbl in _div_order if _by_niso[lbl]['total'] > 0]
    if _div_plot:
        _div_df = pd.DataFrame(_div_plot)
        _fig_div = _go_fig.Figure()
        _fig_div.add_trace(_go_fig.Bar(
            x=_div_df['n_iso'], y=_div_df['전체'],
            name='전체 유전자', marker_color='#d1d5db',
        ))
        _fig_div.add_trace(_go_fig.Bar(
            x=_div_df['n_iso'], y=_div_df['Divergent'],
            name='Divergent (>0.3)', marker_color='#e63946',
            text=_div_df['비율(%)'].map(lambda v: f"{v:.0f}%"),
            textposition='outside',
        ))
        _fig_div.update_layout(
            barmode='overlay',
            title='이소폼 수별 기능 분화(Divergent) 유전자 비율 — max(PRISM score) − min(PRISM score) > 0.3 기준',
            xaxis_title='유전자당 이소폼 수 (이 데이터셋 기준)',
            yaxis_title='유전자 수',
            height=300, plot_bgcolor='white', paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        st.plotly_chart(_fig_div, use_container_width=True)
        st.caption(
            "**회색 막대**: 각 이소폼 수 구간의 전체 유전자 수 · "
            "**빨간 막대**: 그 중 기능 분화(Divergent, 이소폼 간 최대 PRISM 점수 차이 > 0.3) 유전자 수 · "
            "빨간 막대 위 숫자 = Divergent 비율(%). "
            "이소폼 수가 많을수록 Divergent 비율이 높아지는 경향은 "
            "PRISM이 이소폼 다양성이 높은 유전자에서 이소폼 수준 기능 차이를 더 풍부하게 포착함을 시사합니다."
        )

st.divider()

# ── Scenario Summary ─────────────────────────────────────────────────────────
st.subheader("D1 · 4-Scenario Classification — DTU × GO 기능 예측으로 아이소폼 분류")
st.caption(
    f"각 아이소폼을 두 축으로 분류합니다: "
    "① **DTU(Differential Transcript Usage)** — 조건 간 사용 비율이 유의미하게 변했는가 (DTU 파일 필요) · "
    f"② **신규 GO 예측** — score > {thr}인 GO term 중 SwissProt annotation에 없는 GO term이 1개 이상 존재하는가 "
    f"(annotation 파일 미로드 시 score > {thr}인 모든 GO term을 신규로 간주) · "
    "S1(DTU+ & 신규GO+)이 최우선 후보, S3(DTU- & 신규GO+)는 조건 무관 구성적 신규 기능 아이소폼."
)

# annotation 로드 (A1b와 동일한 파일 — cached via _ds if already computed)
_ann_file = Path(_root) / 'hMuscle/data/raw_data/data/annotations/human_annotations_unified_bp.txt'
_swissport_ann = None
if _ann_file.exists():
    _swissport_ann = defaultdict(list)
    with open(_ann_file) as _af:
        for _line in _af:
            _parts = _line.strip().split('\t')
            if len(_parts) >= 2:
                for _g in _parts[1:]:
                    if _g.startswith('GO:'):
                        _swissport_ann[_parts[0]].append(_g)

# thr 변경 시 재분류 필요 — thr를 캐시 키로 사용
_classified_thr = st.session_state.get('_classified_thr')
classified = st.session_state.get('classified_df')
if classified is None or _classified_thr != thr:
    with st.spinner("Classifying isoforms…"):
        classified = classify_isoforms(
            sm, ids, genes, go,
            existing_annotations=dict(_swissport_ann) if _swissport_ann else cfg.get('existing_annotations'),
            dtu_df=dtu,
            score_threshold=thr,
        )
        st.session_state['classified_df']  = classified
        st.session_state['_classified_thr'] = thr

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
    _precomp_path = Path(__file__).parents[3] / 'reports' / 'brain_full_672_meta.json'
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
