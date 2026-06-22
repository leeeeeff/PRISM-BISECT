"""
BISECT 세포 맥락 분석 — Isoform Switch × Cell Type × Braak
KIF21B/NDUFS4/DLG1 switch의 세포 타입 특이성 + Braak 단계 상관
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

cfg = st.session_state.get('cfg', {})

CELLTYPE_TSV = Path('/home/welcome1/sw1686/DIFFUSE/reports/bisect_celltype/celltype_isoform_switch.tsv')
BRAAK_TSV    = Path('/home/welcome1/sw1686/DIFFUSE/reports/braak_correlation_results.tsv')
C18_TSV      = Path('/home/welcome1/sw1686/DIFFUSE/reports/c18_deep_dive/c18_vs_c10c11_diff.tsv')
LAMP5_TSV    = Path('/home/welcome1/sw1686/DIFFUSE/reports/lamp5_kit_validation/lamp5_subtypes_comparison.tsv')
LAMP5_DONOR  = Path('/home/welcome1/sw1686/DIFFUSE/reports/lamp5_kit_validation/c15_per_donor_expression.tsv')

SIG_MARKS  = {'***': '★★★', '**': '★★', '*': '★', '†': '◆', 'ns': ''}
SIG_COLORS = {'***': '#c0392b', '**': '#e74c3c', '*': '#e67e22', '†': '#f39c12', 'ns': '#555566'}
COND_PAL   = {'AD': '#e74c3c', 'Control': '#3498db', 'Active control': '#f39c12'}
CT_ORDER   = [
    'Excitatory_neuron', 'Inhibitory_neuron', 'Oligodendrocyte',
    'OPC', 'Astrocyte', 'Microglia', 'Vascular_cell', 'Lymphocyte',
]

st.title("🔬 BISECT 세포 맥락 분석")
st.caption(
    "핵심 AD isoform switch의 세포 타입 특이성 | Braak 단계 상관 | "
    "C18 심층 분석 | LAMP5/KIT 검증"
)


@st.cache_data
def load_celltype():
    return pd.read_csv(CELLTYPE_TSV, sep='\t') if CELLTYPE_TSV.exists() else pd.DataFrame()


@st.cache_data
def load_braak():
    return pd.read_csv(BRAAK_TSV, sep='\t') if BRAAK_TSV.exists() else pd.DataFrame()


@st.cache_data
def load_c18():
    return pd.read_csv(C18_TSV, sep='\t') if C18_TSV.exists() else pd.DataFrame()


@st.cache_data
def load_lamp5():
    sub   = pd.read_csv(LAMP5_TSV,   sep='\t') if LAMP5_TSV.exists()   else pd.DataFrame()
    donor = pd.read_csv(LAMP5_DONOR, sep='\t') if LAMP5_DONOR.exists() else pd.DataFrame()
    return sub, donor


tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 세포 타입별 Switch",
    "📈 Braak 상관",
    "🔭 C18 심층 분석",
    "🔬 LAMP5/KIT 검증",
])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("세포 타입별 핵심 Isoform Switch 유의성")
    st.caption("DIU permutation 검정 (n=10,000) | 빨강 계열 = AD isoform switch 유의")

    ct_df = load_celltype()

    if ct_df.empty:
        st.warning("celltype_isoform_switch.tsv 없음")
    else:
        TARGET_GENES = ['KIF21B', 'NDUFS4', 'NDUFS7', 'NDUFS8', 'DLG1']
        genes_avail = [g for g in TARGET_GENES if g in ct_df['gene'].values]
        ct_df = ct_df[ct_df['gene'].isin(genes_avail)].copy()

        # ── 히트맵: cell_type × gene, 색=-log10(p) ───────────────────────────
        pivot_p = ct_df.pivot_table(index='cell_type', columns='gene', values='p', aggfunc='min')
        pivot_p = pivot_p.reindex(columns=genes_avail)
        ct_rows = [c for c in CT_ORDER if c in pivot_p.index] + \
                  [c for c in pivot_p.index if c not in CT_ORDER]
        pivot_p = pivot_p.reindex(ct_rows)
        log_p = -np.log10(pivot_p.fillna(1).clip(lower=1e-50))

        pivot_sig = ct_df.pivot_table(index='cell_type', columns='gene', values='sig', aggfunc='first')
        pivot_sig = pivot_sig.reindex(index=ct_rows, columns=genes_avail)
        ann_text = [[
            SIG_MARKS.get(str(pivot_sig.loc[r, c]), '') if not pd.isna(pivot_sig.loc[r, c]) else ''
            for c in genes_avail
        ] for r in ct_rows]

        fig_hm = go.Figure(go.Heatmap(
            z=log_p.values,
            x=genes_avail,
            y=[r.replace('_', ' ') for r in ct_rows],
            colorscale='YlOrRd',
            zmin=0, zmax=12,
            text=ann_text,
            texttemplate='%{text}',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{y} — %{x}</b><br>−log₁₀(p) = %{z:.2f}<extra></extra>',
            colorbar=dict(title='−log₁₀(p)', len=0.8, tickfont=dict(color='white')),
        ))
        fig_hm.update_layout(
            height=380,
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='white', size=11),
            margin=dict(t=30, b=60, l=160, r=40),
            xaxis=dict(side='top'),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("★★★ p<0.001 | ★★ p<0.01 | ★ p<0.05 | ◆ p<0.10 | 흰칸=검출 없음")

        # ── 그룹 막대: 선택 유전자의 AD vs CT ratio per cell type ─────────────
        st.divider()
        sel_gene = st.selectbox(
            "유전자 선택 (isoform switch AD/CT 비율)",
            options=genes_avail,
            index=0,
            key='tab1_gene_sel',
        )
        gene_df = ct_df[ct_df['gene'] == sel_gene].copy()
        gene_df = gene_df.sort_values('delta', ascending=False)
        gene_df['ct_label'] = gene_df['cell_type'].str.replace('_', ' ')

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='AD isoform ratio',
            x=gene_df['ct_label'], y=gene_df['AD_ratio'],
            marker_color='#e74c3c',
            customdata=gene_df[['n_AD', 'p', 'sig']].values,
            hovertemplate='<b>%{x}</b><br>AD ratio=%{y:.3f}<br>n=%{customdata[0]}<br>p=%{customdata[1]:.4f} %{customdata[2]}<extra></extra>',
        ))
        fig_bar.add_trace(go.Bar(
            name='CT isoform ratio',
            x=gene_df['ct_label'], y=gene_df['CT_ratio'],
            marker_color='#3498db',
            customdata=gene_df[['n_CT']].values,
            hovertemplate='<b>%{x}</b><br>CT ratio=%{y:.3f}<br>n=%{customdata[0]}<extra></extra>',
        ))
        for _, row in gene_df[gene_df['sig'].isin(['***', '**', '*', '†'])].iterrows():
            fig_bar.add_annotation(
                x=row['ct_label'],
                y=max(row['AD_ratio'], row['CT_ratio']) + 0.03,
                text=SIG_MARKS.get(row['sig'], ''),
                showarrow=False, font=dict(size=13, color='white'),
            )
        fig_bar.update_layout(
            height=340, barmode='group',
            xaxis_title='', yaxis_title='Isoform usage ratio',
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='white', size=11),
            legend=dict(orientation='h', yanchor='bottom', y=1.01),
            margin=dict(t=40, b=60, l=60, r=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("""
**핵심 발견:**
- **KIF21B** — Excitatory ★★★, Inhibitory ★ → **뉴런 특이적** (Oligo/Astro/Microglia 없음)
- **NDUFS7** — 범세포형 (Exc + Inh + Oligo + Astro + OPC) → 전신 Complex I 이상
- **DLG1** — Exc + Inh + Oligo + OPC → 시냅스-미엘린 연결
- **NDUFS8** — Inh + OPC 우선
- **NDUFS4** — Excitatory 원발성
""")

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Braak Stage × Isoform Switch 상관분석")
    st.caption("Spearman r | Bootstrap 95% CI | Bonferroni 보정")

    braak_df = load_braak()

    if braak_df.empty:
        st.warning("braak_correlation_results.tsv 없음")
    else:
        braak_df = braak_df.copy()
        braak_df['sig'] = braak_df['p_bonferroni'].apply(
            lambda p: '**' if p < 0.01 else ('*' if p < 0.05 else ('†' if p < 0.10 else 'ns'))
        )

        # ── 버블 차트 ──────────────────────────────────────────────────────────
        fig_bub = go.Figure()
        for sig_lv in ['**', '*', '†', 'ns']:
            grp = braak_df[braak_df['sig'] == sig_lv]
            if grp.empty:
                continue
            fig_bub.add_trace(go.Scatter(
                x=grp['gene'], y=grp['spearman_r'],
                mode='markers+text',
                name=f"{SIG_MARKS.get(sig_lv, sig_lv)} ({sig_lv})",
                marker=dict(
                    color=SIG_COLORS[sig_lv],
                    size=grp['n_donors'] * 3,
                    line=dict(color='white', width=0.8),
                    opacity=0.85,
                ),
                text=grp['ct_isoform'].str[:15],
                textposition='top center',
                textfont=dict(size=8, color='#aaaaaa'),
                customdata=grp[['ct_isoform', 'n_donors', 'p_value', 'p_bonferroni', 'description']].values,
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'Isoform: %{customdata[0]}<br>'
                    'r = %{y:.3f}  n = %{customdata[1]}<br>'
                    'p = %{customdata[2]:.4f}  Bonf = %{customdata[3]:.4f}<br>'
                    '<i>%{customdata[4]}</i><extra></extra>'
                ),
            ))
        fig_bub.add_hline(y=0,    line_width=1, line_color='white', opacity=0.3)
        fig_bub.add_hline(y=0.5,  line_dash='dot', line_color='#e74c3c', opacity=0.5,
                           annotation_text='r=0.5',  annotation_position='right')
        fig_bub.add_hline(y=-0.5, line_dash='dot', line_color='#3498db', opacity=0.5,
                           annotation_text='r=−0.5', annotation_position='right')
        fig_bub.update_layout(
            height=400,
            xaxis_title='유전자', yaxis_title='Spearman r (isoform ratio vs Braak B)',
            yaxis=dict(range=[-1, 1]),
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='white', size=11),
            legend=dict(orientation='h', yanchor='bottom', y=1.01),
            margin=dict(t=30, b=60, l=70, r=80),
        )
        st.plotly_chart(fig_bub, use_container_width=True)
        st.caption("버블 크기 = 도너 수 비례 | 점선: |r|=0.5 기준")

        # ── CI 막대 ────────────────────────────────────────────────────────────
        has_ci = ('ci_95_lo' in braak_df.columns and braak_df['ci_95_lo'].notna().any())
        if has_ci:
            st.divider()
            st.subheader("Bootstrap 95% CI")
            ci_df = braak_df.dropna(subset=['ci_95_lo', 'ci_95_hi']).copy()
            ci_df = ci_df.sort_values('spearman_r')
            fig_ci = go.Figure()
            for _, row in ci_df.iterrows():
                c = SIG_COLORS.get(row['sig'], '#888')
                fig_ci.add_trace(go.Scatter(
                    x=[row['ci_95_lo'], row['spearman_r'], row['ci_95_hi']],
                    y=[row['gene']] * 3,
                    mode='lines+markers',
                    line=dict(color=c, width=3),
                    marker=dict(size=[8, 14, 8], color=c,
                                symbol=['line-ew', 'circle', 'line-ew']),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{row['gene']}</b><br>"
                        f"r={row['spearman_r']:.3f} [{row['ci_95_lo']:.3f}, {row['ci_95_hi']:.3f}]"
                        "<extra></extra>"
                    ),
                ))
            fig_ci.add_vline(x=0, line_width=1, line_color='white', opacity=0.4)
            fig_ci.update_layout(
                height=max(200, len(ci_df) * 55),
                xaxis_title='Spearman r', xaxis=dict(range=[-1, 1]),
                yaxis_title='',
                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                font=dict(color='white', size=11),
                margin=dict(t=20, b=50, l=100, r=40),
            )
            st.plotly_chart(fig_ci, use_container_width=True)

        with st.expander("전체 수치"):
            show_cols = ['gene', 'ct_isoform', 'n_donors', 'spearman_r',
                         'p_value', 'p_bonferroni', 'sig', 'description']
            avail = [c for c in show_cols if c in braak_df.columns]
            st.dataframe(braak_df[avail].sort_values('p_value'), use_container_width=True)

        st.markdown("""
**결과 해석:**
- **KIF21B**: r=−0.527, n=10, p=0.117 (경향) — Braak 높을수록 motor domain isoform 감소
- **NDUFS4**: r=−0.308, n=13 — 방향 일치, 유의성 미확보
- **한계**: Braak B (0–3) 4단계 해상도 → Roman numeral (I–VI) 코호트 확장 시 개선 가능
""")

# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("C18 — L4 IT atypical (PRSS12⁺) 심층 분석")
    st.caption("C18 vs L4 IT 평균(C10+C11) 마커 유전자 발현 비교 | log₂FC 기준")

    c18_df = load_c18()

    if c18_df.empty:
        st.warning("c18_vs_c10c11_diff.tsv 없음")
    else:
        c18_df = c18_df.copy()
        c18_df = c18_df.sort_values('log2FC_C18_vs_C10C11', ascending=False)

        n_show = st.slider("표시 유전자 수 (상위/하위 각각)", 5, 20, 10, 1, key='c18_n')
        top    = c18_df.head(n_show)
        bottom = c18_df.tail(n_show)
        plot_df = pd.concat([top, bottom]).drop_duplicates('gene').sort_values('log2FC_C18_vs_C10C11')

        colors_bar = ['#e74c3c' if v > 0 else '#3498db' for v in plot_df['log2FC_C18_vs_C10C11']]

        fig_c18 = go.Figure(go.Bar(
            x=plot_df['log2FC_C18_vs_C10C11'],
            y=plot_df['gene'],
            orientation='h',
            marker_color=colors_bar,
            customdata=plot_df[['C18', 'C10_C11_mean']].values,
            hovertemplate=(
                '<b>%{y}</b><br>'
                'log₂FC = %{x:.3f}<br>'
                'C18 = %{customdata[0]:.3f}<br>'
                'C10/C11 avg = %{customdata[1]:.3f}<extra></extra>'
            ),
        ))
        fig_c18.add_vline(x=0,  line_width=1, line_color='white', opacity=0.4)
        fig_c18.add_vline(x=1,  line_dash='dot', line_color='#e74c3c', opacity=0.4)
        fig_c18.add_vline(x=-1, line_dash='dot', line_color='#3498db', opacity=0.4)
        fig_c18.update_layout(
            height=max(350, len(plot_df) * 22),
            xaxis_title='log₂FC  C18 vs L4 IT (C10+C11 평균)',
            yaxis_title='',
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='white', size=10),
            margin=dict(l=100, r=60, t=20, b=50),
        )
        st.plotly_chart(fig_c18, use_container_width=True)
        st.caption("빨강: C18 과발현 | 파랑: L4 IT 평균 과발현 | 점선: |log₂FC|=1")

        st.divider()
        st.subheader("C18 vs L4 IT 발현 산점도")
        top_genes = c18_df.head(5)['gene'].tolist() + c18_df.tail(5)['gene'].tolist()
        max_val = max(c18_df['C18'].max(), c18_df['C10_C11_mean'].max()) * 1.05

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val], mode='lines',
            line=dict(color='white', width=1, dash='dot'),
            showlegend=False, hoverinfo='none',
        ))
        fig_sc.add_trace(go.Scatter(
            x=c18_df['C10_C11_mean'], y=c18_df['C18'],
            mode='markers',
            marker=dict(
                color=c18_df['log2FC_C18_vs_C10C11'],
                colorscale='RdBu_r', cmin=-3, cmax=3,
                size=7,
                colorbar=dict(title='log₂FC', tickfont=dict(color='white')),
                line=dict(color='white', width=0.3),
            ),
            text=c18_df['gene'],
            customdata=c18_df[['log2FC_C18_vs_C10C11']].values,
            hovertemplate='<b>%{text}</b><br>C10/C11=%{x:.3f}  C18=%{y:.3f}<br>log₂FC=%{customdata[0]:.2f}<extra></extra>',
        ))
        for g in top_genes:
            row = c18_df[c18_df['gene'] == g]
            if not row.empty:
                fig_sc.add_annotation(
                    x=row['C10_C11_mean'].iloc[0], y=row['C18'].iloc[0],
                    text=g, showarrow=True, arrowhead=2, arrowwidth=1,
                    arrowcolor='white', font=dict(size=9, color='white'),
                    ax=20, ay=-20,
                )
        fig_sc.update_layout(
            height=440,
            xaxis_title='L4 IT 평균 발현 (C10+C11)',
            yaxis_title='C18 발현',
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='white', size=11),
            margin=dict(l=70, r=40, t=20, b=60),
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption("대각선 위 = C18 과발현 | 색상: 빨강=C18 특이, 파랑=L4 IT 일반")

        st.markdown("""
**C18 해석:**
- **PRSS12 과발현**: 일반 L4 IT(C10, C11) 대비 20× 이상 → C18의 분자적 정의 마커
- **ETV1/BCL11A 억제**: 일반 L4 IT 정체성 마커 소실 → 비전형적 분화 궤도
- **AD 내 축적(+1.46%, p=0.0099)**: PRSS12⁺ 세포가 특이적으로 AD 뇌에 증가
- **PRSS12 기능**: 신경 세린 프로테아제 → ECM 리모델링 / 시냅스 정리 이상 가능성
""")

# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("LAMP5/KIT (C15) — Novel AD-enriched Interneuron 검증")
    st.caption("C15 vs C13(LAMP5/NDNF) 비교 | per-donor 발현 | ivy cell / CAA 가설")

    lamp5_sub, lamp5_donor = load_lamp5()

    if lamp5_sub.empty and lamp5_donor.empty:
        st.warning("lamp5_kit_validation/ 데이터 없음")
    else:
        col1, col2 = st.columns([3, 2])

        with col1:
            markers_to_plot = [m for m in ['KIT', 'PRSS12', 'ADARB2', 'LAMP5']
                               if not lamp5_donor.empty and m in lamp5_donor.columns]
            sel_marker = st.selectbox(
                "마커 선택", options=markers_to_plot, index=0, key='lamp5_marker_sel',
            )
            if not lamp5_donor.empty:
                fig_box = go.Figure()
                for cond, color in [('AD', '#e74c3c'), ('Control', '#3498db')]:
                    grp = lamp5_donor[lamp5_donor['condition'] == cond]
                    if grp.empty:
                        continue
                    fig_box.add_trace(go.Box(
                        y=grp[sel_marker], x=[cond] * len(grp),
                        name=cond, marker_color=color,
                        boxpoints='all', jitter=0.3, pointpos=0,
                        marker=dict(size=9, line=dict(color='white', width=1)),
                        customdata=grp[['donor', 'n_cells']].values,
                        hovertemplate=(
                            '<b>%{customdata[0]}</b><br>'
                            f'{sel_marker}=%{{y:.3f}}<br>'
                            'n_cells=%{customdata[1]}<extra></extra>'
                        ),
                    ))
                fig_box.update_layout(
                    height=340,
                    title=f"C15 {sel_marker} per-donor",
                    yaxis_title=f'Mean {sel_marker} expression',
                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                    font=dict(color='white', size=11),
                    margin=dict(l=60, r=20, t=50, b=40),
                )
                st.plotly_chart(fig_box, use_container_width=True)
                st.caption("KIT 발현 수준 자체는 AD≈CT (p=0.645) → **세포 수 비율**이 증가한 것.")

        with col2:
            if not lamp5_sub.empty:
                radar_markers = [m for m in
                                 ['KIT', 'PRSS12', 'KIF21B', 'LAMP5', 'ADARB2', 'NDNF', 'LHX6', 'SST', 'PVALB']
                                 if m in lamp5_sub.columns]
                labels_map = {13: 'C13 LAMP5/NDNF', 15: 'C15 LAMP5/KIT'}
                colors_rad = ['#3498db', '#e74c3c']
                fig_rad = go.Figure()
                for i, cl in enumerate(lamp5_sub['cluster'].tolist()):
                    row = lamp5_sub[lamp5_sub['cluster'] == cl]
                    if row.empty:
                        continue
                    vals = [float(row[m].iloc[0]) for m in radar_markers]
                    vmax = max(vals) if max(vals) > 0 else 1
                    vals_norm = [v / vmax for v in vals] + [vals[0] / vmax]
                    theta = radar_markers + [radar_markers[0]]
                    fig_rad.add_trace(go.Scatterpolar(
                        r=vals_norm, theta=theta,
                        fill='toself',
                        name=labels_map.get(cl, f'C{cl}'),
                        line_color=colors_rad[i % len(colors_rad)],
                        opacity=0.7,
                    ))
                fig_rad.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1],
                                        tickfont=dict(color='#888', size=8)),
                        angularaxis=dict(tickfont=dict(color='white', size=10)),
                        bgcolor='#0e1117',
                    ),
                    height=340,
                    paper_bgcolor='#0e1117',
                    font=dict(color='white', size=10),
                    legend=dict(x=0.8, y=1.1),
                    margin=dict(t=50, b=20, l=20, r=20),
                    title=dict(text='LAMP5 서브타입 마커 프로파일', font=dict(color='white', size=12)),
                )
                st.plotly_chart(fig_rad, use_container_width=True)

        st.markdown("""
**LAMP5/KIT (C15) 해석:**

| 항목 | C13 LAMP5/NDNF | C15 LAMP5/KIT |
|------|---------------|--------------|
| 위치 | L1–L2 | L1 혈관 주변 |
| 문헌 | Mossink 2022 (AD 소실) | **Novel** |
| AD 변화 | Δ−0.8% (ns) | **+1.04% (★)** |
| KIT 발현 | 0.41 | **2.08** (5×) |
| PRSS12 | 0.027 | **0.56** (20×) |

- **Ivy cell 가설**: L1 혈관 주변 억제 뉴런 → CAA(혈관 아밀로이드) 반응 → AD 내 반응성 증가
- **PRSS12 공동발현**: C18(L4 atypical)과 분자적 연결점 → 별도 손상 경로?
- **논문 서술**: "Novel AD-enriched LAMP5/KIT interneuron, distinct from LAMP5/NDNF neurogliaform cells"
""")
