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

# ── Gene Landing Card (auto_search) — shown ABOVE tabs ───────────────────────
_auto_gene = st.session_state.get('search_gene', '')
if st.session_state.get('auto_search') and _auto_gene and sm is not None:
    st.session_state['auto_search'] = False
    _render_gene_landing(
        _auto_gene, sm, ids, genes, go, gnames,
        cfg.get('score_threshold', 0.4), cfg.get('dtu_df'), cfg,
    )

@st.cache_data(show_spinner=False)
def _load_umap_precomp():
    """Load precomputed UMAP coords and sample indices (brain_672 only)."""
    demo = Path(__file__).parents[2] / 'data' / 'demo'
    cp = demo / 'umap_coords.npy'
    sp = demo / 'umap_sample_idx.npy'
    if cp.exists() and sp.exists():
        return np.load(cp), np.load(sp)
    return None, None


def _build_umap_figure(gene: str, hit_ids, hit_scores, hit_types,
                        sm, ids_arr, iso_types, tissue: str) -> object:
    """Build Plotly UMAP scatter with the target gene highlighted.

    Strategy:
    - brain_672: use precomputed 20K coords as background; project off-sample
      gene isoforms via cosine nearest-neighbour averaging.
    - other/upload: compute fresh UMAP on random subsample + gene isoforms.
    """
    import plotly.graph_objects as go_plotly
    from plotly.subplots import make_subplots

    TYPE_COLORS = {'known': '#94a3b8', 'nic': '#f97316', 'nnic': '#8b5cf6', '': '#94a3b8'}
    GENE_TYPE_COLORS = {'known': '#2563eb', 'nic': '#ea580c', 'nnic': '#7c3aed', '': '#2563eb'}

    coords_pre, samp_idx = _load_umap_precomp()
    use_precomp = (tissue == 'brain_672' and coords_pre is not None
                   and samp_idx is not None and sm is not None
                   and sm.shape[0] == len(ids_arr))

    if use_precomp:
        # Background: 20K sampled isoforms
        bg_ids   = ids_arr[samp_idx]
        bg_types = np.array(iso_types, dtype=str)[samp_idx] if iso_types is not None else np.full(len(samp_idx), '')
        bg_x, bg_y = coords_pre[:, 0], coords_pre[:, 1]

        # Find gene isoform indices in the full array
        gene_global_idx = np.where(np.isin(ids_arr, hit_ids))[0]
        # Which are in the UMAP sample?
        samp_set = set(samp_idx.tolist())
        in_samp  = [i for i in gene_global_idx if i in samp_set]
        off_samp = [i for i in gene_global_idx if i not in samp_set]

        gene_x, gene_y = [], []
        gene_labels, gene_type_list = [], []

        # Exact coords for in-sample isoforms
        samp_pos = {v: k for k, v in enumerate(samp_idx)}
        for gi in in_samp:
            si = samp_pos[gi]
            gene_x.append(float(coords_pre[si, 0]))
            gene_y.append(float(coords_pre[si, 1]))
            gene_labels.append(str(ids_arr[gi]))
            gene_type_list.append(str(iso_types[gi]) if iso_types is not None else '')

        # NN approximation for off-sample isoforms
        if off_samp:
            samp_scores = sm[samp_idx].astype(np.float32)  # (20K, n_go)
            samp_norms  = np.linalg.norm(samp_scores, axis=1, keepdims=True) + 1e-8
            samp_unit   = samp_scores / samp_norms
            for gi in off_samp:
                v = sm[gi].astype(np.float32)
                vn = np.linalg.norm(v) + 1e-8
                sims = samp_unit @ (v / vn)
                top5 = np.argsort(sims)[-5:]
                ax = float(coords_pre[top5, 0].mean())
                ay = float(coords_pre[top5, 1].mean())
                gene_x.append(ax); gene_y.append(ay)
                gene_labels.append(str(ids_arr[gi]) + ' ≈')
                gene_type_list.append(str(iso_types[gi]) if iso_types is not None else '')

    else:
        # Fresh mini-UMAP: subsample + gene isoforms
        try:
            from umap import UMAP
        except ImportError:
            return None

        n_total = sm.shape[0] if sm is not None else 0
        if n_total == 0:
            return None

        n_bg = min(3000, n_total)
        rng  = np.random.default_rng(42)
        bg_idx = rng.choice(n_total, size=n_bg, replace=False)
        gene_global_idx = np.where(np.isin(ids_arr, hit_ids))[0]

        all_idx = np.concatenate([bg_idx, gene_global_idx])
        all_idx_unique, inv = np.unique(all_idx, return_inverse=True)
        X = sm[all_idx_unique].astype(np.float32)

        with st.spinner("UMAP 계산 중 (15-30초)…"):
            emb = UMAP(n_components=2, random_state=42, n_neighbors=15,
                       min_dist=0.1, metric='cosine').fit_transform(X)

        bg_pos_in_unique  = np.where(np.isin(all_idx_unique, bg_idx))[0]
        gene_pos_in_unique = np.where(np.isin(all_idx_unique, gene_global_idx))[0]

        bg_x = emb[bg_pos_in_unique, 0]; bg_y = emb[bg_pos_in_unique, 1]
        bg_ids_local   = ids_arr[all_idx_unique[bg_pos_in_unique]]
        bg_types       = (np.array(iso_types, dtype=str)[all_idx_unique[bg_pos_in_unique]]
                          if iso_types is not None else np.full(len(bg_pos_in_unique), ''))

        gene_x = emb[gene_pos_in_unique, 0].tolist()
        gene_y = emb[gene_pos_in_unique, 1].tolist()
        gene_labels = ids_arr[all_idx_unique[gene_pos_in_unique]].tolist()
        gene_type_list = (np.array(iso_types, dtype=str)[all_idx_unique[gene_pos_in_unique]].tolist()
                          if iso_types is not None else [''] * len(gene_pos_in_unique))
        bg_ids = bg_ids_local

    # Build Plotly figure
    fig = go_plotly.Figure()

    # Background traces by type
    for ttype, tcolor in TYPE_COLORS.items():
        tmask = np.array([str(t) == ttype for t in bg_types]) if len(bg_types) > 0 else np.array([])
        if not tmask.any():
            continue
        fig.add_trace(go_plotly.Scatter(
            x=bg_x[tmask] if isinstance(bg_x, np.ndarray) else [bg_x[i] for i, m in enumerate(tmask) if m],
            y=bg_y[tmask] if isinstance(bg_y, np.ndarray) else [bg_y[i] for i, m in enumerate(tmask) if m],
            mode='markers',
            marker=dict(size=3, color=tcolor, opacity=0.25),
            name=ttype or 'other',
            showlegend=False,
            hoverinfo='skip',
        ))

    # Gene isoform — single consolidated trace with customdata for on_select
    max_s = hit_scores.max(axis=1)
    g_colors  = [GENE_TYPE_COLORS.get(str(t), '#2563eb') for t in gene_type_list]
    g_sizes   = [14 + 6 * (float(max_s[i]) if i < len(max_s) else 0.0) for i in range(len(gene_x))]
    g_symbols = ['diamond' if '≈' in str(lab) else 'circle' for lab in gene_labels]
    g_texts   = [str(lab).replace(' ≈', '') for lab in gene_labels]
    g_scores  = [float(max_s[i]) if i < len(max_s) else 0.0 for i in range(len(gene_x))]
    # customdata: [[isoform_id, type, max_score], ...]
    g_custom  = [[str(lab).replace(' ≈', ''), str(t), f"{s:.3f}"]
                 for lab, t, s in zip(gene_labels, gene_type_list, g_scores)]

    fig.add_trace(go_plotly.Scatter(
        x=gene_x, y=gene_y,
        mode='markers+text',
        marker=dict(
            size=g_sizes, color=g_colors, opacity=0.92,
            symbol=g_symbols, line=dict(width=2, color='white'),
        ),
        text=g_texts,
        textposition='top center',
        textfont=dict(size=9, color='#1e293b'),
        customdata=g_custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "type: %{customdata[1]}<br>"
            "max score: %{customdata[2]}"
            "<br><i>클릭하여 바스켓에 추가</i><extra></extra>"
        ),
        name='gene_isoforms',
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=f"Score space UMAP — {gene.upper()}", font_size=12),
        height=340,
        margin=dict(t=36, b=10, l=10, r=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor='#f8fafc',
        paper_bgcolor='white',
    )
    if use_precomp:
        fig.add_annotation(
            text="● known  ● NIC  ● NNIC  ◆ approx.",
            xref='paper', yref='paper', x=0.01, y=0.01,
            showarrow=False, font=dict(size=9, color='#64748b'),
            align='left',
        )
    return fig


def _render_gene_landing(gene: str, sm, ids, genes, go, gnames, thr, dtu_df, cfg_local=None):
    """Inline gene report card — shown above tabs when arriving from Hub/sidebar."""
    _cfg = cfg_local or {}
    ids_arr  = np.array(ids,   dtype=str)
    gene_arr = np.array(genes, dtype=str) if genes is not None else None
    _ityp = _cfg.get('isoform_types')
    iso_types_arr = np.array(_ityp, dtype=str) if _ityp is not None else None
    tissue = _cfg.get('tissue', '')

    # Find matching isoforms
    gene_upper = gene.upper()
    if gene_arr is not None:
        mask = np.char.upper(gene_arr) == gene_upper
    else:
        mask = np.char.upper(ids_arr).startswith(gene_upper)
    if not mask.any():
        mask = np.array([gene_upper in str(g).upper() for g in (gene_arr if gene_arr is not None else ids_arr)])
    if not mask.any():
        st.warning(f"'{gene}' not found in loaded dataset.")
        return

    hit_ids    = ids_arr[mask]
    hit_scores = sm[mask]
    hit_types  = iso_types_arr[mask] if iso_types_arr is not None else np.full(mask.sum(), '')
    max_per_iso = hit_scores.max(axis=1)

    st.markdown(f"""
<div style='background:linear-gradient(90deg,#0f2942,#1e3a5f);border-radius:12px;
padding:16px 24px 12px;margin-bottom:16px'>
<span style='color:#93c5fd;font-size:0.8rem'>GENE QUICK REPORT</span>
<h2 style='color:white;margin:4px 0 2px;font-size:1.6rem'>{gene.upper()}</h2>
<span style='color:#bfdbfe;font-size:0.9rem'>{mask.sum()} isoforms · {len(go)} GO terms</span>
</div>
""", unsafe_allow_html=True)

    lc_umap, lc_table, lc_metrics = st.columns([2, 2, 1])

    # ── Mini UMAP ─────────────────────────────────────────────────────────
    with lc_umap:
        fig_umap = _build_umap_figure(
            gene, hit_ids, hit_scores, hit_types,
            sm, ids_arr, iso_types_arr, tissue,
        )
        if fig_umap is not None:
            umap_event = st.plotly_chart(
                fig_umap, use_container_width=True,
                key='umap_landing', on_select='rerun',
                selection_mode='points',
            )
            # ── Click handler: add selected isoform to basket ─────────────
            sel_pts = getattr(getattr(umap_event, 'selection', None), 'points', [])
            if sel_pts:
                for pt in sel_pts:
                    cd = pt.get('customdata')
                    if cd and len(cd) >= 1:
                        sel_iso = str(cd[0])
                        sel_type = str(cd[1]) if len(cd) > 1 else ''
                        sel_score = str(cd[2]) if len(cd) > 2 else ''
                        # Add to basket_isoforms
                        basket_iso = st.session_state.get('basket_isoforms', [])
                        if sel_iso not in basket_iso:
                            basket_iso.append(sel_iso)
                            st.session_state['basket_isoforms'] = basket_iso
                        # Also update highlight_isoforms for Condition page
                        st.session_state['highlight_isoforms'] = basket_iso[:]
                        st.toast(f"✅ {sel_iso} 바스켓 추가 (type={sel_type}, score={sel_score})")

            # Show basket_isoforms actions
            basket_iso = st.session_state.get('basket_isoforms', [])
            if basket_iso:
                st.markdown(
                    f"<div style='background:#f0fdf4;border-radius:6px;padding:6px 10px;"
                    f"font-size:0.8rem;color:#15803d;margin-top:4px'>"
                    f"🧬 선택된 이소폼 ({len(basket_iso)}): "
                    f"<b>{', '.join(basket_iso[:5])}{'…' if len(basket_iso)>5 else ''}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                ci1, ci2 = st.columns(2)
                with ci1:
                    if st.button("🔄 Condition 분석에서 하이라이트", key='umap_goto_cond',
                                 use_container_width=True):
                        st.session_state['highlight_isoforms'] = basket_iso[:]
                        st.toast("Condition 페이지에서 해당 이소폼이 강조됩니다.")
                with ci2:
                    if st.button("🗑️ 선택 초기화", key='umap_clear_iso', use_container_width=True):
                        st.session_state['basket_isoforms'] = []
                        st.session_state['highlight_isoforms'] = []
                        st.rerun()
        else:
            # Fallback: score heatmap
            import plotly.express as px_local
            top_go_idx   = np.argsort(hit_scores.max(axis=0))[-min(12, len(go)):][::-1]
            top_go_names = [gnames.get(go[i], go[i])[:20] for i in top_go_idx]
            fig_hm = px_local.imshow(
                hit_scores[:, top_go_idx],
                x=top_go_names, y=[str(i) for i in hit_ids],
                color_continuous_scale='Blues', aspect='auto',
                title="PRISM score heatmap", zmin=0, zmax=1,
            )
            fig_hm.update_layout(height=340, margin=dict(t=36,b=10,l=10,r=10),
                                  coloraxis_showscale=False)
            fig_hm.update_xaxes(tickangle=-40, tickfont_size=9)
            st.plotly_chart(fig_hm, use_container_width=True)

    # ── Isoform table ─────────────────────────────────────────────────────
    with lc_table:
        top_go_per_iso   = [go[np.argmax(hit_scores[i])] for i in range(len(hit_ids))]
        top_name_per_iso = [gnames.get(g, g)[:28] for g in top_go_per_iso]

        _user_mods = st.session_state.get('user_modules') or st.session_state.get('brain672_modules')
        mod_labels = []
        if _user_mods:
            go_mod_map = _user_mods.get('go_module_map', {})
            for i in range(len(hit_ids)):
                top_g = go[np.argmax(hit_scores[i])]
                mid   = go_mod_map.get(top_g)
                mod_labels.append(f"M{mid}" if mid else "—")
        else:
            mod_labels = ["—"] * len(hit_ids)

        df_land = pd.DataFrame({
            'Isoform':   hit_ids,
            'Type':      hit_types,
            'Max score': max_per_iso.round(3),
            'Top GO':    top_name_per_iso,
            'Module':    mod_labels,
        }).sort_values('Max score', ascending=False)

        st.dataframe(
            df_land.style.background_gradient(subset=['Max score'], cmap='Blues'),
            use_container_width=True, hide_index=True,
            height=min(320, 35 * len(df_land) + 38),
        )

        unique_mods = list(dict.fromkeys(m for m in mod_labels if m != "—"))
        if unique_mods:
            st.markdown("**Primary modules:** " + " · ".join(f"`{m}`" for m in unique_mods))

        # Score heatmap in expander
        with st.expander("📊 GO score heatmap"):
            import plotly.express as px_hm
            top_go_idx   = np.argsort(hit_scores.max(axis=0))[-min(15, len(go)):][::-1]
            top_go_names = [gnames.get(go[i], go[i])[:22] for i in top_go_idx]
            fig_hm2 = px_hm.imshow(
                hit_scores[:, top_go_idx], x=top_go_names,
                y=[str(i) for i in hit_ids],
                color_continuous_scale='Blues', aspect='auto', zmin=0, zmax=1,
            )
            fig_hm2.update_layout(height=max(160, 28*len(hit_ids)+50),
                                   margin=dict(t=10,b=10,l=10,r=10),
                                   coloraxis_showscale=False)
            fig_hm2.update_xaxes(tickangle=-45, tickfont_size=9)
            st.plotly_chart(fig_hm2, use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────
    with lc_metrics:
        st.metric("Isoforms", str(mask.sum()))
        n_high = int((max_per_iso >= thr).sum())
        st.metric("High-conf", str(n_high), delta=f"≥{thr}")
        novel_types = {'nic', 'nnic'}
        n_novel = int(sum(str(t).lower() in novel_types for t in hit_types))
        st.metric("Novel", str(n_novel))
        if dtu_df is not None:
            _ids_upper = set(str(x).upper() for x in hit_ids)
            n_dtu = 0
            for col in ['isoform_id', 'transcript_id', 'feature']:
                if col in dtu_df.columns:
                    n_dtu = dtu_df[dtu_df[col].str.upper().isin(_ids_upper)].shape[0]
                    break
            st.metric("DTU events", str(n_dtu))

    st.caption(
        "UMAP: 배경 = 20K 무작위 샘플(gray), 강조 = 검색 유전자 이소폼(대형). "
        "◆ = 샘플 외 이소폼(cosine NN 근사 위치). "
        "↓ 탭에서 Scenario 분류 · 모듈×DTU 히트맵 · BISECT 케이스 확인."
    )
    st.divider()

# ── Scenario filter tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab_search = st.tabs([
    "🔴 Scenario 1: Functional Switch",
    "🟠 Scenario 2: Expression Switch",
    "🟢 Scenario 3: Constitutive Novel",
    "⬜ Scenario 4: Background",
    "🔍 Search Isoform",
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
    # Pre-populate from sidebar / Hub auto_search
    _default_query = st.session_state.get('search_gene', '')
    _auto_search = st.session_state.get('auto_search', False)

    if _default_query and not st.session_state.get('_targets_query_loaded'):
        st.session_state['_targets_query_loaded'] = True
        st.session_state['targets_search_key'] = _default_query

    if _auto_search and _default_query:
        st.info(f"🔍 **자동 검색**: '{_default_query}'  (Hub에서 연결됨)")
        st.session_state['auto_search'] = False

    query = st.text_input(
        "Search by isoform ID or gene name",
        value=st.session_state.get('targets_search_key', _default_query),
        placeholder="e.g. NDUFS4-201, KIF21B, tr319500",
        key='targets_gene_input',
    )
    # Write back to shared session state so sidebar reflects current query
    if query:
        st.session_state['search_gene'] = query
        st.session_state['targets_search_key'] = query

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
