"""Page 2 — Functional Map: GO-Score UMAP + Heatmap (Modules B1 + A4)."""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import numpy as np

from prism_app.visualization.umap_plot import (
    build_umap_figure, compute_umap_coords, load_precomputed_coords,
)
from prism_app.visualization.heatmap import build_type_go_heatmap, build_go_cooccurrence_network
from prism_app.visualization.scatter import build_within_gene_chart
from prism_app.app.components.interpretation import (
    render_data_context_banner, render_umap_interpretation, render_within_gene_interpretation,
)

st.set_page_config(page_title="Functional Map — PRISM", layout="wide")
st.title("🗺️ Functional Map")
st.caption("GO-score space UMAP, isoform-type × GO heatmap, and within-gene comparison.")

with st.expander("📖 이 페이지 사용법", expanded=False):
    st.markdown("""
**Functional Map** 페이지는 아이소폼의 GO 기능 공간을 시각화합니다.

| 탭 | 내용 | 활용법 |
|----|------|--------|
| **UMAP** | 각 아이소폼을 GO 스코어 벡터(18~73차원)로 2D 공간에 투영 | 비슷한 기능의 아이소폼이 클러스터를 형성하는지 확인; 아이소폼 타입·시나리오·최고 GO로 색칠 변경 가능 |
| **Type × GO Heatmap** | 아이소폼 타입(Known/NIC/NNIC)별로 각 GO 기능의 평균 스코어를 히트맵으로 표시 | Novel 아이소폼(NIC/NNIC)이 Known과 다른 GO 패턴을 보이는지 확인 |
| **Within-Gene** | 같은 유전자의 모든 아이소폼 간 GO 스코어 차이를 비교 | 유전자 이름 입력 후 Plot 클릭; 특정 아이소폼이 다른 아이소폼과 뚜렷이 구별되면 기능 스위치 후보 |
| **GO Network** | GO 기능 간 공동예측 상관관계를 네트워크로 시각화 | Pearson r 임계값 조절; 연결된 GO들은 같이 예측되는 경향 (기능 모듈) |

> UMAP은 첫 실행 후 캐싱됩니다. 15,000개 초과 시 무작위 샘플링합니다.
> umap-learn 충돌 시 t-SNE로 자동 대체됩니다 (결과 품질은 유사).
    """)

# ── Data ─────────────────────────────────────────────────────────────────────
cfg = st.session_state.get('cfg', {})
sm  = cfg.get('score_matrix')
if sm is None:
    st.warning("No data loaded. Return to the main page."); st.stop()

render_data_context_banner(cfg)

ids    = cfg['isoform_ids']
types  = cfg.get('isoform_types')
genes  = cfg.get('gene_ids')
go     = cfg['go_terms']
gnames = cfg['go_names']
thr    = cfg['score_threshold']
mode   = cfg.get('mode', 'demo')
classified = st.session_state.get('classified_df')

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_umap, tab_heat, tab_gene, tab_net = st.tabs([
    "UMAP", "Type × GO Heatmap", "Within-Gene", "GO Network",
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: UMAP
# ─────────────────────────────────────────────────────────────────────────────
with tab_umap:
    st.subheader("GO-Score Space UMAP")
    st.caption("GO 스코어 벡터를 2D로 투영합니다. 색상 옵션을 바꿔가며 클러스터 패턴을 분석하세요.")

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    color_by = col_opt1.selectbox(
        "Colour by", ['isoform_type', 'scenario', 'max_go', 'max_score'], index=0,
    )
    point_sz = col_opt2.slider("Point size", 2, 10, 4)
    opacity  = col_opt3.slider("Opacity", 0.2, 1.0, 0.7, 0.05)

    _MAX_UMAP = 15_000   # cap for reasonable render speed

    @st.cache_data(show_spinner="임베딩 계산 중 (첫 실행만 소요)…")
    def _get_umap_coords(sm_bytes, sm_shape, n_total, seed):
        import numpy as np
        sm_arr = np.frombuffer(sm_bytes, dtype=np.float32).reshape(sm_shape)
        rng = np.random.default_rng(seed)
        if n_total > _MAX_UMAP:
            idx = np.sort(rng.choice(n_total, _MAX_UMAP, replace=False))
        else:
            idx = np.arange(n_total)
        coords, method = compute_umap_coords(sm_arr[idx], random_state=seed)
        return coords, idx, method

    n_total = sm.shape[0]
    coords, sample_idx, embed_method = _get_umap_coords(
        sm.astype(np.float32).tobytes(), sm.shape, n_total, seed=42
    )

    sampled_ids        = np.asarray(ids)[sample_idx]
    sampled_sm         = sm[sample_idx]
    sampled_classified = (classified.iloc[sample_idx].reset_index(drop=True)
                          if classified is not None else None)

    note_parts = [f"투영 방법: **{embed_method}**"]
    if n_total > _MAX_UMAP:
        note_parts.append(f"{_MAX_UMAP:,}개 샘플링 (전체 {n_total:,}개)")
    if embed_method == 't-SNE':
        note_parts.append("⚠️ umap-learn을 사용할 수 없어 t-SNE로 대체됨 (로컬 환경에서 TF/protobuf 충돌). `run_app.sh`로 실행하면 UMAP 사용 가능.")
    st.caption(" · ".join(note_parts))

    fig_umap = build_umap_figure(
        coords, sampled_ids,
        color_by=color_by,
        metadata_df=sampled_classified,
        go_terms=go, go_names=gnames,
        score_matrix=sampled_sm,
        point_size=point_sz,
        opacity=opacity,
    )
    st.plotly_chart(fig_umap, use_container_width=True)
    st.caption(
        "Each point = one isoform projected into 2D using its GO-score vector (UMAP, cosine distance). "
        "Isoforms with similar predicted functions cluster together."
    )
    render_umap_interpretation(embed_method, n_total, len(sample_idx), color_by)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Type × GO Heatmap
# ─────────────────────────────────────────────────────────────────────────────
with tab_heat:
    st.subheader("Mean PRISM Score by Isoform Type and GO Term")
    st.caption("Known·NIC·NNIC 각 타입에서 GO 기능별 평균 예측 점수를 비교합니다. 아이소폼 타입 파일이 있어야 표시됩니다.")

    if types is None:
        st.info("Isoform type labels not available. Upload isoform_types file.")
    else:
        fig_heat = build_type_go_heatmap(sm, types, go, gnames)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "Each cell shows the mean PRISM score for isoforms of that structural type "
            "in the given GO term. Known = Ensembl-annotated; NIC = Novel In Catalog; NNIC = Novel Not In Catalog."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Within-Gene Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab_gene:
    st.subheader("Within-Gene Isoform × GO Score Comparison")
    st.caption("유전자 이름(예: NDUFS4, KIF21B, DLG1)을 입력하고 Plot 버튼을 클릭하면 해당 유전자의 모든 아이소폼 간 GO 스코어를 비교합니다.")

    gene_input = st.text_input(
        "Gene symbol", value="DLG1",
        placeholder="e.g. NDUFS4, KIF21B, DLG1",
    )
    chart_type = st.radio("Chart type", ['bar', 'heatmap', 'parallel'], horizontal=True)

    if st.button("Plot"):
        with st.spinner(f"Building chart for {gene_input}…"):
            fig_gene = build_within_gene_chart(
                gene_name=gene_input,
                isoform_ids=ids,
                score_matrix=sm,
                go_terms=go,
                go_names=gnames,
                gene_ids=genes,
                chart_type=chart_type,
            )
        st.plotly_chart(fig_gene, use_container_width=True)
        # Count isoforms for this gene
        ids_arr = np.asarray(ids, dtype=str)
        if genes is not None:
            g_arr = np.asarray(genes, dtype=str)
            n_gene_iso = int((g_arr == gene_input).sum())
        else:
            n_gene_iso = int(np.array([gene_input.lower() in i.lower() for i in ids_arr]).sum())
        if n_gene_iso > 0:
            render_within_gene_interpretation(gene_input, n_gene_iso)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: GO Co-occurrence Network
# ─────────────────────────────────────────────────────────────────────────────
with tab_net:
    st.subheader("GO Term Co-prediction Network")
    min_corr = st.slider("Minimum Pearson r for edge", 0.1, 0.9, 0.3, 0.05)

    if st.button("Build network"):
        from prism_app.visualization.heatmap import build_go_cooccurrence_network
        with st.spinner("Computing GO co-occurrence…"):
            fig_net = build_go_cooccurrence_network(sm, go, gnames, min_corr=min_corr)
        st.plotly_chart(fig_net, use_container_width=True)
        st.caption(
            "Nodes = GO terms; edges drawn when Pearson r ≥ threshold across all isoform scores. "
            "Clustered nodes are co-predicted functions (e.g. muscle contraction + sarcomere organization)."
        )
