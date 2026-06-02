"""Page 5 — Advanced Analytics: Cross-Tissue Comparison (Module E1)."""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Advanced — PRISM", layout="wide")
st.title("🔭 Advanced Analytics")
st.caption("Cross-tissue isoform function comparison and multi-dataset analysis.")

with st.expander("📖 이 페이지 사용법", expanded=False):
    st.markdown("""
**Advanced** 페이지는 단일 데이터셋 분석을 넘어선 비교 분석 도구를 제공합니다.

| 탭 | 설명 | 필요 데이터 |
|----|------|------------|
| **Cross-Tissue** | 근육과 뇌(또는 업로드 데이터) 간 GO 기능 스코어 분포를 비교합니다. 조직 특이적 vs 공통 기능을 파악합니다. | Demo 데이터 기본 제공; 업로드도 가능 |
| **Expression Filter** | CPM(Counts Per Million) 기반 발현량 필터와 PRISM 스코어를 결합합니다. 낮은 발현량에서 나온 예측을 제거합니다. | 카운트 매트릭스 TSV 업로드 필요 |
| **NMD Risk Screen** | Nonsense-Mediated Decay(NMD) 위험이 있는 아이소폼을 필터링합니다. NMD 대상이면 단백질이 만들어지지 않으므로 기능 예측이 무의미할 수 있습니다. | NMD JSON 파일 업로드 필요 |

**Cross-Tissue 탭 활용법:**
1. Tissue A·B 각각 데이터 소스를 선택합니다.
2. **"Run Comparison"** 클릭 → 공통 GO 기능(18개)에서의 평균 스코어 차이를 분석합니다.
3. 산점도에서 대각선에서 멀리 떨어진 GO 기능이 조직 특이적입니다.
    """)

DEMO_DIR = Path(__file__).parents[2] / 'data' / 'demo'

from prism_app.core.go_utils import TISSUE_PRESETS, GO_FULL_NAMES
from prism_app.app.components.interpretation import render_data_context_banner

cfg_main = st.session_state.get('cfg', {})
render_data_context_banner(cfg_main)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_cross, tab_expr, tab_nmd = st.tabs([
    "🔁 Cross-Tissue Comparison",
    "📉 Expression Filter",
    "⚠️ NMD Risk Screen",
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Cross-Tissue Comparison (E1)
# ─────────────────────────────────────────────────────────────────────────────
with tab_cross:
    st.subheader("조직 간 GO 기능 비교 (Cross-Tissue Comparison)")
    st.caption(
        "두 조직의 PRISM 평균 스코어를 GO 기능별로 비교해 **조직 특이적 기능**과 **공통 기능**을 구분합니다. "
        "**Delta (A−B)**: 조직 A의 평균 스코어 − 조직 B의 평균 스코어 · 양수 = A에 더 높게 예측됨 · "
        "**산점도의 대각선(y=x)**: 두 조직에서 동일한 스코어 → 공통 기능 · "
        "대각선 위쪽 = B 특이적, 아래쪽 = A 특이적 · "
        "PRISM의 Zero-shot 전이 성능을 시각적으로 확인하려면 근육(train)과 뇌(zero-shot)를 비교하세요."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Tissue A**")
        tissue_a = st.selectbox("Select tissue A", ['Muscle (demo)', 'Brain (demo)', 'Upload'],
                                 key='tissue_a')
    with col_b:
        st.markdown("**Tissue B**")
        tissue_b = st.selectbox("Select tissue B", ['Brain (demo)', 'Muscle (demo)', 'Upload'],
                                 key='tissue_b', index=0)

    @st.cache_data(show_spinner="Loading tissue data…")
    def _load_tissue(label: str) -> dict:
        if label == 'Muscle (demo)':
            sm    = np.load(DEMO_DIR / 'muscle_scores.npy')
            ids   = np.load(DEMO_DIR / 'muscle_ids.npy',      allow_pickle=True)
            types = np.load(DEMO_DIR / 'muscle_types.npy',    allow_pickle=True)
            go    = list(TISSUE_PRESETS['muscle'].keys())
            return dict(sm=sm, ids=ids, types=types, go=go,
                        gnames=TISSUE_PRESETS['muscle'], name='Muscle')
        elif label == 'Brain (demo)':
            sm    = np.load(DEMO_DIR / 'brain_full_scores.npy')
            ids   = np.load(DEMO_DIR / 'brain_full_ids.npy',   allow_pickle=True)
            types = np.load(DEMO_DIR / 'brain_full_types.npy', allow_pickle=True)
            go    = list(TISSUE_PRESETS['brain'].keys())
            return dict(sm=sm, ids=ids, types=types, go=go,
                        gnames=TISSUE_PRESETS['brain'], name='Brain')
        return {}

    # Upload option
    def _upload_tissue(label_suffix: str) -> dict:
        sm_file   = st.file_uploader(f"Score matrix (.npy)", type=['npy'], key=f'sm_{label_suffix}')
        ids_file  = st.file_uploader(f"Isoform IDs (.npy/.txt)", type=['npy','txt'], key=f'ids_{label_suffix}')
        if sm_file and ids_file:
            sm  = np.load(sm_file, allow_pickle=True)
            ids = np.load(ids_file, allow_pickle=True) if ids_file.name.endswith('.npy') \
                  else np.array(ids_file.read().decode().strip().splitlines())
            return dict(sm=sm, ids=ids, types=None, go=None, gnames=GO_FULL_NAMES,
                        name='Custom')
        return {}

    data_a = _load_tissue(tissue_a) if tissue_a != 'Upload' else _upload_tissue('a')
    data_b = _load_tissue(tissue_b) if tissue_b != 'Upload' else _upload_tissue('b')

    if not data_a or not data_b:
        st.info("Select or upload both tissue datasets to compare.")
    else:
        go_a = set(data_a.get('go') or [])
        go_b = set(data_b.get('go') or [])
        shared_go = sorted(go_a & go_b)

        if not shared_go:
            st.warning("No shared GO terms between the two tissues.")
        else:
            st.info(f"Shared GO terms: **{len(shared_go)}** | "
                    f"{data_a['name']}: {len(data_a['sm']):,} isoforms | "
                    f"{data_b['name']}: {len(data_b['sm']):,} isoforms")

            score_thr = st.slider("Score threshold for 'high'", 0.1, 0.9, 0.5, 0.05,
                                   key='cross_thr')

            go_a_idx = [list(data_a['go']).index(g) for g in shared_go if g in data_a['go']]
            go_b_idx = [list(data_b['go']).index(g) for g in shared_go if g in data_b['go']]
            sm_a_sh  = data_a['sm'][:, go_a_idx]
            sm_b_sh  = data_b['sm'][:, go_b_idx]
            gnames_shared = data_a['gnames']
            mean_a = sm_a_sh.mean(axis=0)
            mean_b = sm_b_sh.mean(axis=0)

            compare_df = pd.DataFrame({
                'GO_ID':  shared_go,
                'GO_term': [gnames_shared.get(g, g)[:40] for g in shared_go],
                f'Mean_{data_a["name"]}': mean_a,
                f'Mean_{data_b["name"]}': mean_b,
                'Delta (A−B)': mean_a - mean_b,
            }).sort_values('Delta (A−B)', key=abs, ascending=False)

            fig_bar = px.bar(
                compare_df,
                x='GO_term', y=['Delta (A−B)', f'Mean_{data_a["name"]}', f'Mean_{data_b["name"]}'],
                barmode='group',
                title=f"GO score comparison: {data_a['name']} vs {data_b['name']}",
                labels={'value': 'Mean PRISM score', 'GO_term': ''},
                color_discrete_map={
                    'Delta (A−B)': '#555',
                    f'Mean_{data_a["name"]}': '#4c72b0',
                    f'Mean_{data_b["name"]}': '#c44e52',
                },
                height=400,
            )
            fig_bar.update_layout(xaxis_tickangle=-35, legend_title='')
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption(
                f"막대 색: 파랑 = {data_a['name']} 평균 스코어 · 빨강 = {data_b['name']} 평균 스코어 · "
                f"회색(Delta A−B) = 두 조직 간 차이 · 위쪽(양수)이면 {data_a['name']}에서 더 높게 예측됨 · "
                "Delta가 크고 두 조직의 절대 스코어가 모두 높으면 공통으로 중요한 기능 · "
                "Delta가 크고 한쪽만 높으면 해당 조직 특이적 기능"
            )

            fig_scatter = px.scatter(
                compare_df,
                x=f'Mean_{data_a["name"]}',
                y=f'Mean_{data_b["name"]}',
                text='GO_term',
                color='Delta (A−B)',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0,
                title=f"GO score correlation: {data_a['name']} vs {data_b['name']}",
                labels={
                    f'Mean_{data_a["name"]}': f'Mean score ({data_a["name"]})',
                    f'Mean_{data_b["name"]}': f'Mean score ({data_b["name"]})',
                },
                height=420,
            )
            mx = float(max(mean_a.max(), mean_b.max())) * 1.05
            fig_scatter.add_shape(type='line', x0=0, y0=0, x1=mx, y1=mx,
                                   line=dict(dash='dash', color='grey', width=1))
            fig_scatter.update_traces(textposition='top center', textfont_size=9)
            fig_scatter.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(
                f"X축: {data_a['name']} GO 평균 스코어 · Y축: {data_b['name']} GO 평균 스코어 · "
                "**점선(y=x 대각선)**: 두 조직에서 동일한 스코어인 기능 (공통 기능) · "
                "대각선 위쪽 점: B에서 더 높게 예측됨(B 특이적) · 아래쪽: A 특이적 · "
                "대각선에서 멀리 떨어질수록 조직 특이성이 강함 · "
                "오른쪽 위 모서리 근처: 두 조직 모두에서 높게 예측된 공통 핵심 기능"
            )

            st.subheader("조직 특이적 GO 기능")
            delta_thr_ct = st.slider("Min |delta| to call tissue-specific", 0.01, 0.1, 0.03, 0.005)

            a_specific  = compare_df[compare_df['Delta (A−B)'] >  delta_thr_ct]
            b_specific  = compare_df[compare_df['Delta (A−B)'] < -delta_thr_ct]
            shared_both = compare_df[compare_df['Delta (A−B)'].abs() <= delta_thr_ct]

            col1, col2, col3 = st.columns(3)
            col1.metric(f"{data_a['name']}-specific GO terms", len(a_specific))
            col2.metric(f"{data_b['name']}-specific GO terms", len(b_specific))
            col3.metric("Shared (|delta| < threshold)", len(shared_both))

            with st.expander(f"{data_a['name']}-enriched GO terms"):
                st.dataframe(a_specific[['GO_term', 'Delta (A−B)',
                                          f'Mean_{data_a["name"]}',
                                          f'Mean_{data_b["name"]}']],
                             use_container_width=True, hide_index=True)
            with st.expander(f"{data_b['name']}-enriched GO terms"):
                st.dataframe(b_specific[['GO_term', 'Delta (A−B)',
                                          f'Mean_{data_a["name"]}',
                                          f'Mean_{data_b["name"]}']],
                             use_container_width=True, hide_index=True)

            csv = compare_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download comparison table (CSV)", csv,
                                "cross_tissue_comparison.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Expression × Score Joint Filter (E4)
# ─────────────────────────────────────────────────────────────────────────────
with tab_expr:
    st.subheader("발현량 × PRISM 스코어 통합 필터 (Expression × Score Joint Filter)")
    st.caption(
        "**왜 이 필터가 필요한가**: PRISM 스코어가 높아도 실제 발현량이 매우 낮으면 "
        "해당 아이소폼이 세포에서 실제로 작동하지 않을 수 있습니다(위양성 위험). "
        "CPM(Counts Per Million) ≥ 임계값 AND PRISM 스코어 ≥ 임계값 조건을 동시에 만족하는 "
        "아이소폼만 최종 후보로 선별합니다. "
        "**CPM 임계값 가이드**: CPM 1 = 평균 발현 수준, CPM 5 = 고발현, CPM 0.1 = 저발현 허용."
    )

    cfg = st.session_state.get('cfg', {})
    sm_main = cfg.get('score_matrix')
    ids_main = cfg.get('isoform_ids')

    count_file = st.file_uploader(
        "Count matrix (isoform × sample, TSV/CSV, first column = isoform ID)",
        type=['tsv', 'csv', 'txt'],
        key='expr_count',
    )

    cpm_thr   = st.slider("Minimum CPM (expression threshold)", 0.1, 10.0, 1.0, 0.1)
    score_thr2 = st.slider("Minimum PRISM score threshold", 0.1, 0.9, 0.5, 0.05,
                            key='expr_score')

    if count_file is None:
        st.info("Upload a count matrix to enable expression filtering. "
                "The main data (sidebar) must also be loaded.")
    elif sm_main is None:
        st.warning("Load a dataset from the sidebar first.")
    else:
        sep = '\t' if count_file.name.endswith('.tsv') else ','
        counts = pd.read_csv(count_file, sep=sep, index_col=0)

        # CPM normalisation
        lib_sizes = counts.sum(axis=0)
        cpm = counts.div(lib_sizes, axis=1) * 1e6
        mean_cpm = cpm.mean(axis=1)

        # Align to score matrix IDs
        ids_arr = np.asarray(ids_main, dtype=str)
        expressed_mask = np.array([
            mean_cpm.get(iso, 0.0) >= cpm_thr for iso in ids_arr
        ])
        high_score_mask = (sm_main > score_thr2).any(axis=1)
        joint_mask = expressed_mask & high_score_mask

        c1, c2, c3 = st.columns(3)
        c1.metric("High PRISM score", int(high_score_mask.sum()))
        c2.metric(f"Expressed (CPM ≥ {cpm_thr})", int(expressed_mask.sum()))
        c3.metric("Both (joint filter)", int(joint_mask.sum()))

        # Show filtered candidates
        from prism_app.core.classifier import classify_isoforms
        go    = cfg['go_terms']
        gnames = cfg['go_names']

        joint_ids   = ids_arr[joint_mask]
        joint_sm    = sm_main[joint_mask]
        joint_genes = cfg.get('gene_ids')
        if joint_genes is not None:
            joint_genes = np.asarray(joint_genes)[joint_mask]

        clf = classify_isoforms(joint_sm, joint_ids, joint_genes, go,
                                 score_threshold=score_thr2)

        st.write(f"**{len(clf):,} isoforms pass joint filter:**")
        st.dataframe(
            clf[['isoform_id','gene_id','max_score','max_go','scenario_label']].head(50),
            use_container_width=True, hide_index=True
        )

        csv2 = clf.to_csv(index=False).encode('utf-8')
        st.download_button("Download filtered candidates (CSV)", csv2,
                            "expression_filtered_candidates.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: NMD Risk Screening (E3)
# ─────────────────────────────────────────────────────────────────────────────
with tab_nmd:
    st.subheader("NMD 위험 아이소폼 스크리닝 (NMD Risk Screening)")
    st.caption(
        "**NMD(Nonsense-Mediated mRNA Decay)**: 조기 종결 코돈(PTC)이 있는 mRNA는 세포 내 분해 메커니즘에 의해 "
        "단백질을 만들기 전에 제거됩니다. NMD 위험이 있는 아이소폼은 PRISM 스코어가 높아도 "
        "실제로는 단백질이 만들어지지 않으므로 기능 예측 해석에 주의가 필요합니다. "
        "**'High-score + NMD risk'** 아이소폼은 실험 검증 전에 NMD 억제제(e.g. cycloheximide) 처리 후 "
        "단백질 발현 여부를 확인하는 것이 권장됩니다."
    )

    from prism_app.core.nmd_filter import load_nmd_screening, add_nmd_flags

    cfg = st.session_state.get('cfg', {})
    classified = st.session_state.get('classified_df')

    # Try to load default NMD screening results
    nmd_json_candidates = [
        Path('reports') / 'nmd_screening_20260516.json',
        Path('reports') / 'nmd_screening.json',
    ]
    nmd_data = None
    for p in nmd_json_candidates:
        if p.exists():
            nmd_data = load_nmd_screening(str(p))
            st.success(f"NMD screening loaded: {len(nmd_data):,} isoforms screened.")
            break

    if nmd_data is None:
        st.info(
            "No NMD screening results found. "
            "Expected at `reports/nmd_screening_20260516.json`.\n\n"
            "NMD screening requires CDS annotation and PTC detection (SQANTI3 or custom)."
        )
    elif classified is None:
        st.warning("Load a dataset from the sidebar and visit Overview first to classify isoforms.")
    else:
        flagged = add_nmd_flags(classified, nmd_data)

        n_nmd_risk = int(flagged['nmd_risk'].sum()) if 'nmd_risk' in flagged.columns else 0
        n_high_nmd = int((flagged['nmd_risk'] & (flagged['max_score'] > 0.5)).sum()) \
                     if 'nmd_risk' in flagged.columns else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Isoforms screened", len(nmd_data))
        c2.metric("NMD-susceptible", n_nmd_risk)
        c3.metric("High-score + NMD risk (caution)", n_high_nmd,
                  delta="False positive risk" if n_high_nmd > 0 else None,
                  delta_color="inverse")

        if 'nmd_risk' in flagged.columns:
            risky = flagged[flagged['nmd_risk'] & (flagged['max_score'] > 0.5)]
            if not risky.empty:
                st.warning(f"{len(risky)} high-confidence predictions may be NMD-susceptible:")
                st.dataframe(
                    risky[['isoform_id','gene_id','max_score','max_go',
                            'scenario_label','nmd_risk']].head(30),
                    use_container_width=True, hide_index=True
                )
