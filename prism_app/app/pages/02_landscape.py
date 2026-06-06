"""Page 6 — Functional Module Landscape (672 BP GO terms → 44 modules)."""
import sys
import json
from pathlib import Path

_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import fisher_exact

st.set_page_config(page_title="Module Landscape — PRISM", layout="wide")
st.title("🧩 Functional Module Landscape")
st.caption(
    "672 BP GO terms × 63,994 brain isoforms → 44 functional modules  "
    "(Ward hierarchical clustering, silhouette = 0.401 · within-module r = 0.818)"
)

with st.expander("📖 이 페이지 사용법", expanded=False):
    st.markdown("""
이 페이지는 PRISM이 예측한 **기능 모듈 구조**를 분석합니다. 사이드바에서 **Brain — Full Module Landscape**를 선택하면 모든 탭이 활성화됩니다.

| 탭 | 핵심 질문 | 활용법 |
|----|----------|--------|
| **📍 모듈 배정** | "내 아이소폼이 어떤 기능 영역에 분포하나?" | 어떤 모듈에 아이소폼이 몰려있는지, 고신뢰도 배정 비율 확인 |
| **🔬 Novel 기능 후보** | "Novel 아이소폼이 특정 기능에 농축되어 있나?" | Fisher's exact test → 유의한 모듈 확인 후 후보 아이소폼 다운로드 |
| **🧬 유전자 기능 다양성** | "같은 유전자에서 기능이 다른 아이소폼이 있나?" | 여러 모듈에 걸친 유전자 = 기능 스위치 후보 |
| **🔄 조건별 기능 변화** | "질병/조건에서 어떤 기능 모듈이 바뀌나?" | DTU 데이터와 교차 분석 (brain demo 자동 로드) |
| **📊 참조 지형도** | "44개 모듈은 무엇을 뜻하나?" | GO-GO 상관 행렬, 버블 차트로 전체 맥락 파악 |

> **모듈 점수(module_score)**: 해당 모듈 GO term들의 평균 PRISM 예측값.
> **고신뢰도 기준**: module_score > 0.3 (상위 35%)
    """)

# ── Data loader ────────────────────────────────────────────────────────────────
REPORTS = Path(__file__).parents[3] / 'reports'

@st.cache_data(show_spinner="모듈 데이터 로딩…")
def _load_refs():
    missing = []
    for p in [REPORTS / 'brain_go_modules_672.json',
              REPORTS / 'brain_full_672_meta.json',
              REPORTS / 'brain_isoform_modules.tsv']:
        if not p.exists():
            missing.append(str(p))
    if missing:
        return None, None, None, missing

    with open(REPORTS / 'brain_go_modules_672.json') as f:
        mod_data = json.load(f)
    with open(REPORTS / 'brain_full_672_meta.json') as f:
        meta = json.load(f)
    df_iso = pd.read_csv(REPORTS / 'brain_isoform_modules.tsv', sep='\t')
    return mod_data, meta, df_iso, []


@st.cache_data(show_spinner="상관 행렬 로딩…")
def _load_corr():
    p = REPORTS / 'brain_go_corr_672.npy'
    return np.load(p).astype(np.float32) if p.exists() else None


mod_data, meta, df_iso_ref, missing = _load_refs()

# ── Sidebar cfg ────────────────────────────────────────────────────────────────
cfg      = st.session_state.get('cfg', {})
tissue   = cfg.get('tissue', '')
sm       = cfg.get('score_matrix')
iso_ids  = cfg.get('isoform_ids')
iso_types= cfg.get('isoform_types')
gene_ids = cfg.get('gene_ids')
dtu_df   = cfg.get('dtu_df')
go_terms = cfg.get('go_terms', [])

# Determine module source: brain_672 precomputed > user_modules > fallback partial
_user_mods = st.session_state.get('user_modules')

if tissue == 'brain_672':
    if missing:
        st.error(
            "Brain-672 필요 파일 없음:\n\n"
            "```bash\nconda activate isoform_env\n"
            "python scripts/build_go_modules.py\n"
            "python scripts/assign_isoform_modules.py\n```"
        )
        st.stop()
    modules = mod_data['modules']
    per_go  = {p['go']: p for p in meta['per_go']}
    USE_PRECOMP = True
elif _user_mods is not None:
    # Use user-generated modules (Upload mode after clustering)
    modules = _user_mods['modules']
    per_go  = {}   # no per-GO AUPRC data for user modules
    USE_PRECOMP = False
    if mod_data is not None:
        # Store brain_672 in session state for Target page module lookups
        st.session_state['brain672_modules'] = mod_data
else:
    # Partial mode: use brain_672 modules for overlap if available, else empty
    if mod_data is not None:
        modules = mod_data['modules']
        per_go  = {p['go']: p for p in meta['per_go']}
    else:
        modules = {}
        per_go  = {}
    USE_PRECOMP = False

# ── Module assignment (real-time for non-brain_672) ────────────────────────────
@st.cache_data(show_spinner="모듈 배정 계산 중…")
def _assign_modules(_sm, go_terms_tuple, _modules_json):
    """Assign isoforms to 44 modules from their score vectors."""
    modules_local = json.loads(_modules_json)
    go_terms_list = list(go_terms_tuple)
    go_idx = {g: i for i, g in enumerate(go_terms_list)}

    mod_ids = sorted(modules_local.keys(), key=int)
    n_mods  = len(mod_ids)
    n_iso   = len(_sm)

    if n_mods == 0:
        return (np.zeros(n_iso, dtype=int),
                np.zeros(n_iso, dtype=np.float32), 0)

    mod_score_mat = np.zeros((n_iso, n_mods), dtype=np.float32)
    overlap_counts = []
    for j, mid_str in enumerate(mod_ids):
        go_list = modules_local[mid_str]['go_ids']
        overlap = [go_idx[g] for g in go_list if g in go_idx]
        overlap_counts.append(len(overlap))
        if overlap:
            mod_score_mat[:, j] = _sm[:, overlap].mean(axis=1)

    primary_idx   = np.argmax(mod_score_mat, axis=1)
    primary_module = np.array([int(mod_ids[i]) for i in primary_idx])
    module_score   = mod_score_mat[np.arange(n_iso), primary_idx]

    n_overlap_total = sum(overlap_counts)
    return primary_module, module_score, n_overlap_total


def _get_assignment_df():
    """Return df with isoform assignments + types, from precomputed or real-time."""
    if USE_PRECOMP:
        df = df_iso_ref.copy()
        df['high_conf'] = df['module_score'] > 0.3
        return df, True

    if sm is None or iso_ids is None or len(go_terms) == 0 or not modules:
        return None, False

    modules_json = json.dumps(modules)
    primary, mod_score, n_overlap = _assign_modules(sm, tuple(go_terms), modules_json)

    df = pd.DataFrame({
        'isoform_id':     np.asarray(iso_ids, dtype=str),
        'primary_module': primary,
        'module_score':   mod_score,
    })
    if iso_types is not None:
        df['type'] = np.asarray(iso_types, dtype=str)
    if gene_ids is not None:
        df['gene'] = np.asarray(gene_ids, dtype=str)

    df['module_label'] = df['primary_module'].apply(
        lambda m: modules.get(str(m), {}).get('label', f'M{m}').split('/')[0].strip()[:35]
    )
    df['high_conf'] = df['module_score'] > 0.3
    return df, n_overlap > 0


# ─────────────────────────────────────────────────────────────────────────────
# Top banner
# ─────────────────────────────────────────────────────────────────────────────
if USE_PRECOMP:
    st.success("Brain — Full Module Landscape (672 GO terms) 데이터 로드됨.")
    df_main = df_iso_ref.copy()
    df_main['high_conf'] = df_main['module_score'] > 0.3
elif _user_mods is not None and sm is not None:
    # User-generated modules
    from prism_app.core.clustering import assign_isoforms_to_modules
    _pm, _ms = assign_isoforms_to_modules(sm, go_terms, _user_mods)
    df_main = pd.DataFrame({
        'isoform_id':     np.asarray(iso_ids, dtype=str),
        'primary_module': _pm,
        'module_score':   _ms,
    })
    if iso_types is not None:
        df_main['type'] = np.asarray(iso_types, dtype=str)
    if gene_ids is not None:
        df_main['gene'] = np.asarray(gene_ids, dtype=str)
    df_main['module_label'] = df_main['primary_module'].apply(
        lambda m: modules.get(str(m), {}).get('label', f'M{m}').split('/')[0].strip()[:35]
    )
    df_main['high_conf'] = df_main['module_score'] > 0.3
    n_mods = _user_mods['n_modules']
    sil    = _user_mods['best_silhouette']
    st.info(
        f"자동 생성 모듈 적용됨: **{n_mods}개 모듈** (silhouette={sil:.3f}, GO term {len(go_terms)}개). "
        "사이드바에서 '재생성'으로 k를 조정할 수 있습니다."
    )
elif sm is not None:
    df_main, ok = _get_assignment_df()
    if ok:
        n_ov = sum(1 for g in go_terms if any(g in m.get('go_ids', []) for m in modules.values()))
        st.info(f"사용자 데이터 ({tissue}): GO term {len(go_terms)}개 중 672-모듈과 **{n_ov}개** 겹침. 부분 배정 완료.")
        if n_ov == 0:
            st.caption("💡 사이드바에서 **모듈 자동 생성**을 실행하면 전체 Landscape 분석이 활성화됩니다.")
    else:
        st.warning("모듈 배정 실패: score matrix 또는 GO term 목록을 확인하세요.")
        df_main = None
else:
    st.info("사이드바에서 **Brain — Full Module Landscape** 를 선택하거나 데이터를 업로드하면 분석이 활성화됩니다.")
    df_main = None

# ─────────────────────────────────────────────────────────────────────────────
# Reference banner (always show key numbers)
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("GO terms", f"{meta['n_go']:,}")
c2.metric("기능 모듈", mod_data['n_modules'])
c3.metric("Silhouette", f"{mod_data['best_silhouette']:.3f}")
c4.metric("Brain AUPRC", f"{meta['macro_auprc_brain']:.4f}")
c5.metric("Brain isoforms", f"{meta['n_isoforms_brain']:,}")
st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_ref, tab_assign, tab_novel, tab_div, tab_cond = st.tabs([
    "📊 참조 지형도",
    "📍 모듈 배정",
    "🔬 Novel 기능 후보",
    "🧬 유전자 기능 다양성",
    "🔄 조건별 기능 변화",
])


# ═════════════════════════════════════════════════════════════════════════════
# Tab 1: Module Assignment Overview
# ═════════════════════════════════════════════════════════════════════════════
with tab_assign:
    st.subheader("아이소폼 기능 모듈 배정 결과")
    st.caption(
        "각 아이소폼을 44개 기능 모듈 중 예측 스코어가 가장 높은 모듈에 배정합니다. "
        "**module_score > 0.3 = 고신뢰도** (상위 35%)."
    )

    if df_main is None:
        st.info("데이터를 로드하면 분석이 표시됩니다.")
        st.stop()

    # Key metrics
    n_total   = len(df_main)
    n_hi      = df_main['high_conf'].sum()
    top_mod_n = df_main['primary_module'].value_counts().idxmax()
    top_mod_l = modules.get(str(top_mod_n), {}).get('label', '')[:35]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("전체 아이소폼", f"{n_total:,}")
    m2.metric("고신뢰도 배정 (score>0.3)", f"{n_hi:,}", f"{100*n_hi/n_total:.1f}%")
    m3.metric("최다 배정 모듈", f"M{top_mod_n}")
    m4.metric("모듈 레이블", top_mod_l[:20] + "…" if len(top_mod_l) > 20 else top_mod_l)

    st.divider()

    col_l, col_r = st.columns(2)

    # Left: top modules by isoform count
    with col_l:
        st.markdown("**상위 20개 모듈 — 아이소폼 수**")
        top20 = (
            df_main.groupby('primary_module')
            .agg(n_iso=('isoform_id', 'count'),
                 mean_score=('module_score', 'mean'))
            .reset_index()
            .sort_values('n_iso', ascending=False)
            .head(20)
        )
        top20['label'] = top20['primary_module'].apply(
            lambda m: f"M{m}: {modules.get(str(m),{}).get('label','').split('/')[0].strip()[:25]}"
        )
        top20['auprc'] = top20['primary_module'].apply(
            lambda m: float(np.mean([
                per_go[g]['auprc_brain']
                for g in modules.get(str(m), {}).get('go_ids', [])
                if g in per_go and per_go[g].get('auprc_brain')
            ])) if modules.get(str(m)) else 0.0
        )

        fig_assign = px.bar(
            top20.sort_values('n_iso'),
            x='n_iso', y='label', orientation='h',
            color='auprc', color_continuous_scale='Viridis',
            labels={'n_iso': '# 아이소폼', 'label': '', 'auprc': 'Mean AUPRC'},
            height=480,
        )
        fig_assign.update_layout(
            coloraxis_colorbar=dict(title='AUPRC', len=0.6),
            margin=dict(l=200, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_assign, use_container_width=True)
        st.caption("색이 진할수록 해당 모듈의 brain zero-shot 예측 성능이 높음.")

    # Right: module_score distribution
    with col_r:
        st.markdown("**모듈 점수 분포 (배정 신뢰도)**")
        fig_hist = px.histogram(
            df_main, x='module_score', nbins=60,
            color_discrete_sequence=['#4c72b0'],
            labels={'module_score': 'Module score (해당 모듈 GO term 평균 PRISM 예측값)'},
            height=240,
        )
        fig_hist.add_vline(x=0.3, line_dash='dash', line_color='red',
                           annotation_text='고신뢰도 기준 (0.3)')
        fig_hist.update_layout(margin=dict(t=20, b=40))
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(
            "대부분의 아이소폼이 낮은 모듈 점수를 가집니다. "
            "module_score > 0.3 아이소폼만 high-confidence 해석에 사용하세요."
        )

        # Type breakdown if available
        if 'type' in df_main.columns:
            st.markdown("**고신뢰도 배정 아이소폼 타입 분포**")
            type_df = (
                df_main[df_main['high_conf']]
                .groupby('type').size().reset_index(name='count')
            )
            type_df['pct'] = 100 * type_df['count'] / type_df['count'].sum()
            fig_type = px.pie(
                type_df, values='count', names='type',
                color='type',
                color_discrete_map={'known':'#2196F3','nic':'#FF9800','nnic':'#E91E63'},
                height=200,
            )
            fig_type.update_traces(textinfo='label+percent', textfont_size=11)
            fig_type.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_type, use_container_width=True)

    # Full assignment table (filterable)
    with st.expander("전체 모듈 배정 테이블 (필터링 + 다운로드)"):
        min_score = st.slider("최소 module_score", 0.0, 0.6, 0.3, 0.05, key='tbl_score')
        show_cols = ['isoform_id', 'primary_module', 'module_label', 'module_score']
        if 'gene' in df_main.columns:
            show_cols = ['isoform_id', 'gene'] + show_cols[1:]
        if 'type' in df_main.columns:
            show_cols.append('type')

        tbl = df_main[df_main['module_score'] >= min_score][show_cols].sort_values(
            'module_score', ascending=False
        )
        st.write(f"**{len(tbl):,}개** 아이소폼 (score ≥ {min_score})")
        st.dataframe(tbl.head(500), use_container_width=True, hide_index=True)
        csv_assign = tbl.to_csv(index=False).encode()
        st.download_button(
            "다운로드 (CSV)", csv_assign,
            file_name=f"module_assignment_{tissue or 'data'}.csv",
            mime="text/csv",
        )


# ═════════════════════════════════════════════════════════════════════════════
# Tab 2: Novel Isoform Functional Candidates
# ═════════════════════════════════════════════════════════════════════════════
with tab_novel:
    st.subheader("Novel 아이소폼 기능 농축 분석")
    st.caption(
        "NIC/NNIC 아이소폼이 특정 기능 모듈에 유의하게 농축되어 있는지 Fisher's exact test. "
        "**유의한 모듈 = 실험적 검증 우선순위**"
    )

    if df_main is None:
        st.info("데이터를 로드하면 분석이 표시됩니다.")
    elif 'type' not in df_main.columns:
        st.warning("아이소폼 타입(known/nic/nnic) 정보가 없습니다. 사이드바에서 isoform types 파일을 업로드하세요.")
    else:
        min_score_nov = st.slider(
            "최소 module_score (고신뢰도 필터)",
            0.0, 0.5, 0.3, 0.05, key='nov_score',
        )
        df_hi = df_main[df_main['module_score'] > min_score_nov].copy()

        bg_novel = len(df_hi[df_hi['type'].isin(['nic', 'nnic'])]) / max(len(df_hi), 1)
        bg_known = 1 - bg_novel

        st.info(
            f"고신뢰도 아이소폼: **{len(df_hi):,}개** / Background NIC+NNIC 비율: **{100*bg_novel:.1f}%**"
        )

        # Fisher's exact per module
        enrich_rows = []
        for mid, grp in df_hi.groupby('primary_module'):
            n = len(grp)
            if n < 20:
                continue
            n_nov = len(grp[grp['type'].isin(['nic', 'nnic'])])
            n_kn  = n - n_nov
            bg_n  = int(len(df_hi) * bg_novel)
            bg_k  = len(df_hi) - bg_n

            _, pval = fisher_exact([[n_nov, n_kn], [bg_n, bg_k]], alternative='greater')
            enrich_rows.append({
                'module':      int(mid),
                'label':       modules.get(str(int(mid)), {}).get('label', '').split('/')[0].strip()[:40],
                'n_isoforms':  n,
                'n_novel':     n_nov,
                'novel_pct':   100 * n_nov / n,
                'bg_pct':      100 * bg_novel,
                'enrichment':  (n_nov / n) / max(bg_novel, 1e-6),
                'pval':        pval,
            })

        if not enrich_rows:
            st.warning("각 모듈에 20개 이상 아이소폼이 없어 검정 불가합니다.")
        else:
            enrich_df = pd.DataFrame(enrich_rows).sort_values('enrichment', ascending=False)

            # FDR correction (Benjamini-Hochberg)
            n_tests = len(enrich_df)
            sorted_pvals = enrich_df['pval'].values
            rank = np.argsort(sorted_pvals)
            fdr = np.zeros(n_tests)
            for i, r in enumerate(rank):
                fdr[r] = sorted_pvals[r] * n_tests / (i + 1)
            fdr = np.minimum.accumulate(fdr[::-1])[::-1]
            enrich_df['fdr'] = fdr[np.argsort(rank)]
            enrich_df['significant'] = enrich_df['fdr'] < 0.05

            n_sig = enrich_df['significant'].sum()

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("검정된 모듈", len(enrich_df))
            col_stat2.metric("유의한 모듈 (FDR<0.05)", int(n_sig),
                             delta="Novel 농축" if n_sig > 0 else None)
            col_stat3.metric("Background NIC+NNIC", f"{100*bg_novel:.1f}%")

            # Enrichment bar chart
            fig_enrich = px.bar(
                enrich_df.sort_values('enrichment', ascending=True).tail(25),
                x='enrichment', y='label', orientation='h',
                color='significant',
                color_discrete_map={True: '#E53935', False: '#9E9E9E'},
                hover_data={
                    'n_isoforms': True, 'n_novel': True,
                    'novel_pct': ':.1f', 'pval': ':.2e', 'fdr': ':.3f',
                    'significant': True,
                },
                labels={'enrichment': '농축비 (NIC+NNIC 비율 / background)', 'label': ''},
                height=550,
            )
            fig_enrich.add_vline(x=1.0, line_dash='dash', line_color='grey',
                                 annotation_text='background (1.0×)')
            fig_enrich.update_layout(
                legend=dict(title='FDR < 0.05', orientation='h', y=1.02),
                margin=dict(l=280, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_enrich, use_container_width=True)
            st.caption(
                "빨간 막대: NIC+NNIC 비율이 background 대비 통계적으로 유의하게 높은 모듈 (FDR < 0.05). "
                "1.0보다 클수록 Novel 아이소폼이 풍부. 실험 검증 우선순위 선정에 활용하세요."
            )

            # Significant module details
            if n_sig > 0:
                sig_mods = enrich_df[enrich_df['significant']].sort_values('enrichment', ascending=False)
                st.markdown("### 유의한 모듈 — 후보 아이소폼 목록")

                for _, row in sig_mods.iterrows():
                    mid = int(row['module'])
                    with st.expander(
                        f"M{mid}: {row['label']} — "
                        f"{row['novel_pct']:.1f}% Novel ({row['n_novel']} isoforms, "
                        f"enrichment {row['enrichment']:.2f}×, FDR={row['fdr']:.3f})"
                    ):
                        st.session_state['active_module'] = mid
                        cands = df_hi[
                            (df_hi['primary_module'] == mid) &
                            (df_hi['type'].isin(['nic', 'nnic']))
                        ].sort_values('module_score', ascending=False)

                        show_c = ['isoform_id', 'module_score', 'type']
                        if 'gene' in cands.columns:
                            show_c = ['isoform_id', 'gene', 'module_score', 'type']
                        st.dataframe(cands[show_c].head(100),
                                     use_container_width=True, hide_index=True)

                        # Module GO terms
                        go_list = modules.get(str(mid), {}).get('go_ids', [])
                        st.markdown(f"**포함 GO term ({len(go_list)}개):**")
                        go_rows = []
                        for g in go_list:
                            p = per_go.get(g, {})
                            go_rows.append({
                                'GO ID': g,
                                'Name': meta['go_names'].get(g, g)[:50],
                                'Brain AUPRC': p.get('auprc_brain', None),
                            })
                        st.dataframe(
                            pd.DataFrame(go_rows).sort_values('Brain AUPRC', ascending=False),
                            use_container_width=True, hide_index=True,
                        )

                # Download all novel candidates from sig modules
                sig_mod_ids = set(int(r['module']) for _, r in sig_mods.iterrows())
                cands_all = df_hi[
                    df_hi['primary_module'].isin(sig_mod_ids) &
                    df_hi['type'].isin(['nic', 'nnic'])
                ].sort_values('module_score', ascending=False)

                csv_cands = cands_all.to_csv(index=False).encode()
                st.download_button(
                    f"📥 유의한 모듈 Novel 후보 전체 다운로드 ({len(cands_all)}개, CSV)",
                    csv_cands,
                    file_name="novel_module_candidates.csv",
                    mime="text/csv",
                )

            # Full enrichment table
            with st.expander("전체 모듈 농축 테이블"):
                st.dataframe(
                    enrich_df[['module', 'label', 'n_isoforms', 'n_novel',
                                'novel_pct', 'enrichment', 'pval', 'fdr', 'significant']]
                    .sort_values('enrichment', ascending=False),
                    use_container_width=True, hide_index=True,
                )


# ═════════════════════════════════════════════════════════════════════════════
# Tab 3: Gene Functional Diversity
# ═════════════════════════════════════════════════════════════════════════════
with tab_div:
    st.subheader("유전자별 기능 다양성 분석")
    st.caption(
        "같은 유전자에서 서로 다른 기능 모듈로 배정된 아이소폼이 존재하는지 분석합니다. "
        "**다중 모듈 유전자 = 기능 스위치 후보** → 실험 검증 우선순위"
    )

    if df_main is None:
        st.info("데이터를 로드하면 분석이 표시됩니다.")
    elif 'gene' not in df_main.columns:
        st.warning("유전자 ID(gene_ids) 정보가 없습니다. 사이드바에서 gene IDs 파일을 업로드하세요.")
    else:
        min_score_div = st.slider("최소 module_score", 0.0, 0.5, 0.3, 0.05, key='div_score')
        df_div = df_main[df_main['module_score'] > min_score_div].copy()

        # Compute per-gene diversity
        gene_stats = (
            df_div.groupby('gene')
            .agg(
                n_isoforms   = ('isoform_id', 'count'),
                n_modules    = ('primary_module', 'nunique'),
                module_ids   = ('primary_module', lambda x: sorted(x.unique().tolist())),
                mean_score   = ('module_score', 'mean'),
            )
            .reset_index()
        )
        if 'type' in df_div.columns:
            has_novel = df_div.groupby('gene')['type'].apply(
                lambda t: t.isin(['nic', 'nnic']).any()
            ).rename('has_novel').reset_index()
            gene_stats = gene_stats.merge(has_novel, on='gene', how='left')
        else:
            gene_stats['has_novel'] = False

        gene_stats['module_labels'] = gene_stats['module_ids'].apply(
            lambda mids: ' / '.join(
                modules.get(str(m), {}).get('label', f'M{m}').split('/')[0].strip()[:20]
                for m in mids[:3]
            ) + (' …' if len(mids) > 3 else '')
        )

        multi_gene = gene_stats[gene_stats['n_modules'] >= 2]
        n_genes_total = len(gene_stats)
        n_multi = len(multi_gene)

        m1, m2, m3 = st.columns(3)
        m1.metric("분석된 유전자", f"{n_genes_total:,}")
        m2.metric("다중 모듈 유전자 (≥2)", f"{n_multi:,}", f"{100*n_multi/max(n_genes_total,1):.1f}%")
        m3.metric("최대 모듈 수 (단일 유전자)", int(gene_stats['n_modules'].max()))

        # Scatter: n_isoforms vs n_modules
        st.markdown("**유전자별 아이소폼 수 × 기능 모듈 다양성**")
        scatter_df = gene_stats[gene_stats['n_isoforms'] >= 2].copy()

        fig_div = px.scatter(
            scatter_df,
            x='n_isoforms', y='n_modules',
            color='has_novel',
            color_discrete_map={True: '#E91E63', False: '#2196F3'},
            size='mean_score',
            hover_data={'gene': True, 'n_isoforms': True, 'n_modules': True,
                        'module_labels': True},
            labels={
                'n_isoforms': '아이소폼 수 (gene)',
                'n_modules':  '배정된 모듈 수',
                'has_novel':  'Novel 아이소폼 포함',
                'mean_score': 'Mean module score',
            },
            opacity=0.6,
            height=420,
        )
        # Highlight region: few isoforms, many modules
        fig_div.add_hrect(y0=3, y1=scatter_df['n_modules'].max() + 0.5,
                          fillcolor='rgba(255,200,0,0.05)',
                          line_width=0, annotation_text='고다양성 영역',
                          annotation_position='top left',
                          annotation_font=dict(size=10, color='#b45309'))
        fig_div.update_layout(legend=dict(orientation='h', y=1.02))
        st.plotly_chart(fig_div, use_container_width=True)
        st.caption(
            "분홍색: Novel 아이소폼(NIC/NNIC) 포함 유전자. "
            "파란색: Known 아이소폼만. "
            "**오른쪽 위(많은 아이소폼 + 많은 모듈)**: 기능적으로 복잡한 유전자. "
            "**왼쪽 위(적은 아이소폼 + 많은 모듈)**: 강한 기능 특이성 → 가장 흥미로운 후보."
        )

        # Top diversity genes table
        st.markdown("**기능 다양성 상위 50개 유전자**")
        top_div = gene_stats.sort_values(['n_modules', 'mean_score'], ascending=[False, False]).head(50)
        show_div = ['gene', 'n_isoforms', 'n_modules', 'module_labels', 'mean_score']
        if 'has_novel' in top_div.columns:
            show_div.append('has_novel')
        st.dataframe(top_div[show_div].rename(columns={
            'gene': '유전자', 'n_isoforms': '아이소폼 수',
            'n_modules': '모듈 수', 'module_labels': '배정 모듈 (상위 3)',
            'mean_score': '평균 모듈 점수', 'has_novel': 'Novel 포함',
        }), use_container_width=True, hide_index=True)

        csv_div = gene_stats.sort_values('n_modules', ascending=False).to_csv(index=False).encode()
        st.download_button(
            "📥 전체 유전자 기능 다양성 테이블 다운로드",
            csv_div, file_name="gene_functional_diversity.csv", mime="text/csv",
        )

        # Quick basket add
        top_genes = top_div['gene'].head(8).tolist() if 'gene' in top_div.columns else []
        if top_genes:
            st.markdown("**후보 바스켓에 추가** (클릭 → 사이드바에 저장 → Target Analysis에서 분석)")
            _bcols = st.columns(min(len(top_genes), 8))
            for _bi, _bg in enumerate(top_genes):
                with _bcols[_bi]:
                    if st.button(f"➕{_bg}", key=f'basket_div_{_bi}', use_container_width=True):
                        _b = st.session_state.get('basket_genes', [])
                        if _bg not in _b:
                            _b.append(_bg)
                            st.session_state['basket_genes'] = _b
                        st.session_state['search_gene'] = _bg
                        st.toast(f"{_bg} → 바스켓 추가 완료")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 4: Condition × Module Analysis
# ═════════════════════════════════════════════════════════════════════════════
with tab_cond:
    st.subheader("조건별 기능 모듈 변화 (DTU × Module)")
    st.caption(
        "DTU(Differential Transcript Usage) 결과와 모듈 배정을 교차 분석합니다. "
        "어떤 기능 모듈에서 질병/조건에 따른 아이소폼 스위치가 일어나는지 확인합니다."
    )

    _dtu_local = dtu_df if dtu_df is not None else None

    # Try demo brain_dtu.tsv if no DTU in cfg
    if _dtu_local is None:
        demo_dtu_path = Path(__file__).parents[2] / 'data' / 'demo' / 'brain_dtu.tsv'
        if demo_dtu_path.exists():
            _dtu_local = pd.read_csv(demo_dtu_path, sep='\t')

    if _dtu_local is None:
        st.warning(
            "DTU 데이터가 없습니다. 사이드바에서 DTU results (.tsv)를 업로드하거나 "
            "Brain 데모 패널을 선택하세요.\n\n"
            "**필요 컬럼**: isoform_id, delta_IF (또는 dIF), pvalue (또는 padj), condition (선택)"
        )
    elif df_main is None:
        st.info("사이드바에서 데이터를 먼저 로드하세요.")
    else:
        # Normalize column names
        col_map = {}
        for alt in ['dIF', 'deltaIF', 'delta_if']:
            if alt in _dtu_local.columns:
                col_map[alt] = 'delta_IF'
        for alt in ['padj', 'adj_pval', 'p_adj', 'FDR']:
            if alt in _dtu_local.columns:
                col_map[alt] = 'pvalue'
        dtu_norm = _dtu_local.rename(columns=col_map)

        if 'delta_IF' not in dtu_norm.columns or 'pvalue' not in dtu_norm.columns:
            st.error(f"필요 컬럼 없음. 현재 컬럼: {list(dtu_norm.columns)}")
        else:
            # Merge with module assignments
            dtu_mod = dtu_norm.merge(
                df_main[['isoform_id', 'primary_module', 'module_label', 'module_score']],
                on='isoform_id', how='inner',
            )

            pval_thr = st.slider("유의 p값 기준", 0.001, 0.1, 0.05, 0.001, key='dtu_pval')
            dif_thr  = st.slider("|delta_IF| 기준 (최소 변화량)", 0.05, 0.3, 0.1, 0.05)

            sig_dtu = dtu_mod[
                (dtu_mod['pvalue'] < pval_thr) &
                (dtu_mod['delta_IF'].abs() > dif_thr)
            ]

            m1c, m2c, m3c = st.columns(3)
            m1c.metric("DTU 유전자-아이소폼 쌍", f"{len(dtu_norm):,}")
            m2c.metric("유의한 DTU 이벤트", f"{len(sig_dtu):,}")
            m3c.metric("영향받은 모듈", sig_dtu['primary_module'].nunique())

            if len(sig_dtu) == 0:
                st.info("유의한 DTU 이벤트가 없습니다. 임계값을 조정해보세요.")
            else:
                has_condition = 'condition' in sig_dtu.columns

                if has_condition:
                    conditions = sig_dtu['condition'].unique().tolist()
                    sel_cond = st.multiselect("조건 선택", conditions, default=conditions[:3])
                    sig_plot = sig_dtu[sig_dtu['condition'].isin(sel_cond)]
                else:
                    sig_plot = sig_dtu

                # Per-module gain / loss counts
                sig_plot = sig_plot.copy()
                sig_plot['direction'] = sig_plot['delta_IF'].apply(
                    lambda d: 'GAIN (usage ↑)' if d > 0 else 'LOSS (usage ↓)'
                )

                if has_condition:
                    mod_counts = (
                        sig_plot.groupby(['primary_module', 'module_label', 'condition', 'direction'])
                        .size().reset_index(name='n')
                    )
                    top_mods = (
                        sig_plot.groupby('primary_module')['isoform_id'].count()
                        .sort_values(ascending=False).head(15).index.tolist()
                    )
                    mod_counts = mod_counts[mod_counts['primary_module'].isin(top_mods)]

                    fig_dtu = px.bar(
                        mod_counts,
                        x='n', y='module_label', color='direction',
                        facet_col='condition', facet_col_wrap=3,
                        color_discrete_map={
                            'GAIN (usage ↑)': '#43A047',
                            'LOSS (usage ↓)': '#E53935',
                        },
                        barmode='group', orientation='h',
                        labels={'n': 'DTU 이벤트 수', 'module_label': ''},
                        height=500,
                    )
                    fig_dtu.update_layout(
                        legend=dict(orientation='h', y=1.04),
                        margin=dict(l=250),
                    )
                else:
                    mod_counts = (
                        sig_plot.groupby(['primary_module', 'module_label', 'direction'])
                        .size().reset_index(name='n')
                    )
                    top_mods = (
                        sig_plot.groupby('primary_module')['isoform_id'].count()
                        .sort_values(ascending=False).head(20).index.tolist()
                    )
                    mod_counts = mod_counts[mod_counts['primary_module'].isin(top_mods)]
                    mod_counts.loc[mod_counts['direction'] == 'LOSS (usage ↓)', 'n'] *= -1

                    fig_dtu = px.bar(
                        mod_counts,
                        x='n', y='module_label', color='direction',
                        color_discrete_map={
                            'GAIN (usage ↑)': '#43A047',
                            'LOSS (usage ↓)': '#E53935',
                        },
                        orientation='h',
                        labels={'n': 'GAIN (+) / LOSS (−) 이벤트 수', 'module_label': ''},
                        height=500,
                    )
                    fig_dtu.add_vline(x=0, line_color='grey', line_width=1)

                st.plotly_chart(fig_dtu, use_container_width=True)
                st.caption(
                    "초록(GAIN): 조건/질병군에서 해당 모듈 기능을 가진 아이소폼 사용 증가. "
                    "빨강(LOSS): 사용 감소. "
                    "한 모듈에서 GAIN/LOSS가 동시에 크면 **기능 스위치(isoform switch within a module)**."
                )

                csv_dtu = sig_dtu.to_csv(index=False).encode()
                st.download_button(
                    "📥 유의한 DTU × 모듈 결과 다운로드",
                    csv_dtu, file_name="dtu_module_analysis.csv", mime="text/csv",
                )


# ═════════════════════════════════════════════════════════════════════════════
# Tab 5: Reference Landscape (static, always available)
# ═════════════════════════════════════════════════════════════════════════════
with tab_ref:
    st.subheader("44개 기능 모듈 — 참조 지형도")
    st.caption(
        "Brain 672-term 사전 계산 결과. 44개 모듈의 생물학적 의미와 구조를 파악하기 위한 참조 뷰."
    )

    sub_ref1, sub_ref2 = st.tabs(["버블 차트 (AUPRC × Novel 농축)", "GO-GO 상관 행렬"])

    with sub_ref1:
        # Bubble chart (precomputed from df_iso_ref)
        df_bubble = df_iso_ref.copy()
        df_bubble['high_conf'] = df_bubble['module_score'] > 0.3
        df_hi_b = df_bubble[df_bubble['high_conf']]
        bg_b = len(df_hi_b[df_hi_b['type'].isin(['nic', 'nnic'])]) / max(len(df_hi_b), 1)

        def _get_cat(mid):
            m = int(mid)
            if m in {36,37}: return 'Neuronal development'
            if m in {13,14}: return 'Synaptic / GPCR'
            if m in {11,12}: return 'Ion transport'
            if m in {35}:    return 'Cell adhesion'
            if m in {23,24,25}: return 'Immune'
            if m in {4,34}:  return 'Cell cycle'
            if m in {1,2}:   return 'Transcription / DNA'
            if m in {10,8}:  return 'RNA processing'
            return 'General'

        b_rows = []
        for mid_str, minfo in modules.items():
            mid = int(mid_str)
            go_list = minfo['go_ids']
            auprc_v = [per_go[g]['auprc_brain'] for g in go_list if g in per_go and per_go[g].get('auprc_brain')]
            if not auprc_v: continue
            sub = df_hi_b[df_hi_b['primary_module'] == mid]
            if len(sub) < 10: continue
            nov_f = len(sub[sub['type'].isin(['nic','nnic'])]) / len(sub)
            b_rows.append({
                'module_id': mid,
                'label': f"M{mid}: {minfo['label'].split('/')[0].strip()[:28]}",
                'mean_auprc': float(np.mean(auprc_v)),
                'enrichment': (nov_f - bg_b) / max(bg_b, 1e-6),
                'n_isoforms': len(sub),
                'novel_pct':  100 * nov_f,
                'category':   _get_cat(mid_str),
            })

        bdf = pd.DataFrame(b_rows)
        cat_colors = {
            'Neuronal development': '#1565C0', 'Synaptic / GPCR': '#0097A7',
            'Ion transport': '#00897B', 'Cell adhesion': '#43A047',
            'Immune': '#E53935', 'Cell cycle': '#F57F17',
            'Transcription / DNA': '#6A1B9A', 'RNA processing': '#AD1457',
            'General': '#9E9E9E',
        }

        if not bdf.empty:
            fig_b = px.scatter(
                bdf, x='mean_auprc', y='enrichment',
                size='n_isoforms', color='category',
                color_discrete_map=cat_colors, text='label',
                hover_data={'module_id': True, 'n_isoforms': True,
                            'novel_pct': ':.1f', 'mean_auprc': ':.3f', 'label': False},
                size_max=40, height=550,
                labels={'mean_auprc': 'Mean Brain AUPRC (zero-shot 예측 품질)',
                        'enrichment': 'NIC+NNIC 농축비 vs background'},
            )
            fig_b.add_hline(y=0, line_dash='dash', line_color='gray',
                            annotation_text=f'background ({100*bg_b:.1f}%)')
            fig_b.add_vline(x=0.4, line_dash='dot', line_color='lightgray', annotation_text='AUPRC=0.4')
            fig_b.update_traces(textposition='top center', textfont=dict(size=8))
            st.plotly_chart(fig_b, use_container_width=True)
            st.caption(
                "X축이 높을수록 해당 기능 모듈의 예측 성능이 좋음. "
                "Y축이 높을수록 Novel 아이소폼이 집중됨. "
                "오른쪽 위 모듈이 연구 가치가 가장 높음."
            )

    with sub_ref2:
        R = _load_corr()
        if R is None:
            st.info("상관 행렬 없음 (`reports/brain_go_corr_672.npy`).")
        else:
            go_ids_list = meta['go_ids']
            mod_ids_sorted = sorted(modules.keys(), key=int)
            sorted_go_idx, mod_boundaries = [], []
            for mid_str in mod_ids_sorted:
                idx = [go_ids_list.index(g) for g in modules[mid_str]['go_ids'] if g in go_ids_list]
                sorted_go_idx.extend(idx)
                mod_boundaries.append(len(sorted_go_idx))

            R_sorted = R[np.ix_(sorted_go_idx, sorted_go_idx)]
            n = len(sorted_go_idx)
            step = max(1, n // 200)
            R_ds = R_sorted[::step, ::step]
            bounds_ds = [b // step for b in mod_boundaries if b // step < R_ds.shape[0]]

            fig_hm = go.Figure(go.Heatmap(
                z=R_ds, colorscale='RdBu_r', zmin=-0.5, zmax=1.0,
                colorbar=dict(title='Pearson r', len=0.6),
            ))
            for b in bounds_ds[:-1]:
                fig_hm.add_shape(type='line', x0=b, x1=b, y0=0, y1=R_ds.shape[0]-1,
                                 line=dict(color='black', width=0.5))
                fig_hm.add_shape(type='line', y0=b, y1=b, x0=0, x1=R_ds.shape[1]-1,
                                 line=dict(color='black', width=0.5))
            fig_hm.update_layout(
                title='GO-GO Pearson 상관 행렬 (44 모듈 경계선 포함)',
                height=650,
                xaxis=dict(showticklabels=False, title='GO terms'),
                yaxis=dict(showticklabels=False, title='GO terms', autorange='reversed'),
                margin=dict(l=50, r=30, t=60, b=50),
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption(
                "빨강 블록(대각): 모듈 내 GO term들이 같이 높게 예측됨 (within r=0.818). "
                "흰색/파랑: 서로 독립적인 기능 모듈 (cross r=0.277)."
            )

# ── Next Step Banner ─────────────────────────────────────────────────────────
st.divider()
_has_dtu = bool(st.session_state.get('cfg', {}).get('dtu_df') is not None)
if _has_dtu:
    st.markdown("""
<div style='background:linear-gradient(90deg,#f0fdf4,#dcfce7);border-radius:10px;
padding:16px 24px;border-left:4px solid #16a34a;margin-top:16px'>
<b>다음 단계 (DTU 데이터 있음): 🔄 Condition Analysis</b><br>
<span style='color:#374151;font-size:0.9rem'>
AD vs CT 조건에서 모듈 전환이 일어나는 유전자를 확인하세요.<br>
관심 모듈 발견 시 → 사이드바 바스켓에 저장 → Condition 페이지에서 심화 분석
</span>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div style='background:linear-gradient(90deg,#eff6ff,#dbeafe);border-radius:10px;
padding:16px 24px;border-left:4px solid #3b82f6;margin-top:16px'>
<b>다음 단계: 🔬 Functional Patterns</b><br>
<span style='color:#374151;font-size:0.9rem'>
모듈 구조를 파악했다면 → GO co-occurrence 네트워크로 기능 패턴을 심화 탐색하세요.
</span>
</div>
""", unsafe_allow_html=True)
