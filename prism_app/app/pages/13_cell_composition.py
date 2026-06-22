"""
세포 구성 분석 — Cell Composition (AD vs CT)
30개 Leiden 클러스터 서브타입 × AD/CT 구성 변화
코호트 배치 효과 검증 결과 포함
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

cfg = st.session_state.get('cfg', {})

SUBTYPE_TSV   = Path('/home/welcome1/sw1686/DIFFUSE/reports/final_subtype_classification.tsv')
COHORT_TSV    = Path('/home/welcome1/sw1686/DIFFUSE/reports/cohort_batch_check/cohort_batch_results.tsv')
DONOR_TSV     = Path('/home/welcome1/sw1686/DIFFUSE/reports/cohort_batch_check/per_donor_pct.tsv')
LAYER_TSV     = Path('/home/welcome1/sw1686/DIFFUSE/reports/layer_annotation/layer_annotation_results.tsv')
LAYER_MWU_TSV = Path('/home/welcome1/sw1686/DIFFUSE/reports/layer_annotation/layer_composition_mwu.tsv')
LAYER_ADC_TSV = Path('/home/welcome1/sw1686/DIFFUSE/reports/layer_annotation/layer_composition_ad_ct.tsv')

SIG_MARKS  = {'***': '★★★', '**': '★★', '*': '★', '†': '◆', 'ns': ''}
SIG_COLORS = {'***': '#c0392b', '**': '#e74c3c', '*': '#e67e22', '†': '#f39c12', 'ns': '#555566'}
SIG_CAPTION = "★★★ p<0.001 | ★★ p<0.01 | ★ p<0.05 | ◆ p<0.10 | 회색 ns"
COHORT_PAL = {'PO': '#e67e22', 'SMC': '#3498db'}
COND_PAL   = {'AD': '#e74c3c', 'Control': '#3498db', 'Active control': '#f39c12'}


@st.cache_data
def load_data():
    sub    = pd.read_csv(SUBTYPE_TSV,   sep='\t') if SUBTYPE_TSV.exists()   else pd.DataFrame()
    cohort = pd.read_csv(COHORT_TSV,    sep='\t') if COHORT_TSV.exists()    else pd.DataFrame()
    donor  = pd.read_csv(DONOR_TSV,     sep='\t') if DONOR_TSV.exists()     else pd.DataFrame()
    layer  = pd.read_csv(LAYER_TSV,     sep='\t') if LAYER_TSV.exists()     else pd.DataFrame()
    mwu    = pd.read_csv(LAYER_MWU_TSV, sep='\t') if LAYER_MWU_TSV.exists() else pd.DataFrame()
    adc    = pd.read_csv(LAYER_ADC_TSV, sep='\t') if LAYER_ADC_TSV.exists() else pd.DataFrame()
    return sub, cohort, donor, layer, mwu, adc


st.title("🧬 세포 구성 분석 — AD vs Control")
st.caption("30개 Leiden 클러스터 서브타입별 AD/CT 비율 변화 | PO+SMC 코호트 배치 효과 검증 포함")

sub_df, cohort_df, donor_df, layer_df, mwu_df, adc_df = load_data()

if sub_df.empty:
    st.error("final_subtype_classification.tsv 파일이 없습니다.")
    st.stop()

if not layer_df.empty:
    layer_map = dict(zip(layer_df['cluster'].astype(str), layer_df['layer_label']))
    sub_df['layer'] = sub_df['cluster'].astype(str).map(layer_map).fillna('')

# ── 상단 핵심 지표 ─────────────────────────────────────────────────────────────
n_total   = len(sub_df)
n_sig     = int(sub_df['sig'].isin(['***', '**', '*']).sum())
n_tend    = int((sub_df['sig'] == '†').sum())
n_ad_up   = int((sub_df['sig'].isin(['***', '**', '*']) & (sub_df['delta'] > 0)).sum())
n_ad_down = int((sub_df['sig'].isin(['***', '**', '*']) & (sub_df['delta'] < 0)).sum())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("전체 클러스터", n_total)
m2.metric("유의 클러스터", n_sig, f"★ {n_sig}/{n_total}")
m3.metric("AD 증가", n_ad_up,   delta="↑ enriched", delta_color="inverse")
m4.metric("AD 감소", n_ad_down, delta="↓ depleted", delta_color="normal")
m5.metric("경향 (†)",  n_tend)
st.divider()

# ── 탭 구성 ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 전체 서브타입 요약",
    "🔍 유의 클러스터 상세",
    "🔬 코호트 배치 검증",
    "📋 전체 분류표",
    "🗂️ 피질 층별 구성 변화",
])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("전체 30 클러스터 AD vs CT 구성 변화")

    ct_filter = st.multiselect(
        "세포 타입 필터",
        options=sorted(sub_df['cell_type'].unique()),
        default=sorted(sub_df['cell_type'].unique()),
        key='tab1_ct_filter',
    )
    df_show = sub_df[sub_df['cell_type'].isin(ct_filter)].copy()
    df_show['label']  = 'C' + df_show['cluster'].astype(str) + ' ' + df_show['subtype']
    df_show['color']  = df_show['sig'].map(SIG_COLORS)
    df_show['mark']   = df_show['sig'].map(SIG_MARKS)
    df_show = df_show.sort_values(['cell_type', 'delta'])

    # ── 가로 막대 차트 ─────────────────────────────────────────────────────────
    fig_bar = go.Figure()
    for ct in df_show['cell_type'].unique():
        sub = df_show[df_show['cell_type'] == ct].copy()
        fig_bar.add_trace(go.Bar(
            x=sub['delta'],
            y=sub['label'],
            orientation='h',
            name=ct,
            marker_color=sub['color'],
            text=sub['mark'],
            textposition='outside',
            customdata=sub[['p', 'AD_pct', 'CT_pct', 'markers', 'sig']].values,
            hovertemplate=(
                '<b>%{y}</b><br>'
                'Δ = %{x:+.2f}%<br>'
                'p = %{customdata[0]:.4f} %{customdata[4]}<br>'
                'AD %{customdata[1]:.1f}%  CT %{customdata[2]:.1f}%<br>'
                '%{customdata[3]}<extra></extra>'
            ),
        ))

    fig_bar.add_vline(x=0, line_width=1, line_color='white', opacity=0.4)
    fig_bar.update_layout(
        height=max(500, len(df_show) * 22),
        barmode='relative',
        xaxis_title='Δ% (AD − Control)',
        yaxis_title='',
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='white', size=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.01),
        margin=dict(l=250, r=80, t=40, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(SIG_CAPTION + " | 색상: 빨강=유의증가, 파랑=유의감소, 노랑=경향, 회색=ns")

    # ── 화산 플롯 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("화산 플롯 — Δ% vs 유의성")

    vdf = sub_df.copy()
    vdf['neg_log_p'] = -np.log10(vdf['p'].clip(lower=1e-5))
    vdf['label']     = 'C' + vdf['cluster'].astype(str) + ' ' + vdf['subtype']
    vdf['color']     = vdf['sig'].map(SIG_COLORS)
    vdf['size']      = vdf['n'].apply(lambda n: max(6, min(20, n / 1000)))

    fig_vol = go.Figure()
    for sig_level in ['***', '**', '*', '†', 'ns']:
        grp = vdf[vdf['sig'] == sig_level]
        if grp.empty:
            continue
        fig_vol.add_trace(go.Scatter(
            x=grp['delta'],
            y=grp['neg_log_p'],
            mode='markers+text',
            name=f"{SIG_MARKS.get(sig_level, sig_level)} ({sig_level})",
            marker=dict(
                color=SIG_COLORS[sig_level],
                size=grp['size'],
                line=dict(width=0.5, color='white'),
                opacity=0.85,
            ),
            text=grp.apply(lambda r: f"C{r['cluster']}" if r['sig'] != 'ns' else '', axis=1),
            textposition='top center',
            textfont=dict(size=9, color='white'),
            customdata=grp[['label', 'p', 'AD_pct', 'CT_pct', 'n']].values,
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Δ = %{x:+.2f}%<br>'
                'p = %{customdata[1]:.4f}<br>'
                'AD %{customdata[2]:.1f}%  CT %{customdata[3]:.1f}%<br>'
                'n = %{customdata[4]:,}<extra></extra>'
            ),
        ))

    fig_vol.add_hline(y=-np.log10(0.05), line_dash='dot', line_color='#f39c12',
                       annotation_text='p=0.05', annotation_position='right')
    fig_vol.add_hline(y=-np.log10(0.01), line_dash='dot', line_color='#e74c3c',
                       annotation_text='p=0.01', annotation_position='right')
    fig_vol.add_vline(x=0, line_width=1, line_color='white', opacity=0.3)

    fig_vol.update_layout(
        height=480,
        xaxis_title='Δ% (AD − Control)',
        yaxis_title='−log₁₀(p)',
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='white', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.01),
        margin=dict(l=60, r=60, t=20, b=50),
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    st.caption("점 크기: 세포 수 비례 | 점선: 유의 임계값")

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("유의 클러스터 상세 분석")
    sig_df = sub_df[sub_df['sig'].isin(['**', '*', '†'])].sort_values('p')

    for _, r in sig_df.iterrows():
        direction = "↑ AD 증가" if r['delta'] > 0 else "↓ AD 감소"
        color = "#e74c3c" if r['delta'] > 0 else "#3498db"
        layer_txt = r.get('layer', '') or '-'
        st.markdown(f"""
<div style="border-left:4px solid {color};padding:10px;margin:8px 0;
     background:#1a1a2e;border-radius:4px;">
<b>C{r['cluster']} {r['subtype']}</b> &nbsp; {SIG_MARKS.get(r['sig'],'')} &nbsp;
<span style="color:{color}">{direction}</span><br>
<small>
Δ{r['delta']:+.2f}% &nbsp;|&nbsp; p={r['p']:.4f} &nbsp;|&nbsp;
AD={r['AD_pct']:.1f}% vs CT={r['CT_pct']:.1f}%<br>
세포 타입: {r['cell_type']} &nbsp;|&nbsp; Layer: {layer_txt} &nbsp;|&nbsp; 마커: {r['markers']}<br>
{r.get('note','')}</small></div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
### 해석 요약

| 클러스터 | 서브타입 | Δ% | 의미 |
|---------|---------|-----|------|
| **C18** ★★ | L4 IT atypical (PRSS12⁺) | +1.46% | L4 이상 뉴런 AD 내 축적. PRSS12 = 신경 세린 프로테아제 |
| **C9** ★ | Inhibitory SST | −1.28% | SST interneuron 감소. AD 억제 회로 손상과 일치 |
| **C15** ★ | Inhibitory LAMP5/KIT | +1.04% | L1 ivy cell 증가 (novel). 혈관 주변 아밀로이드(CAA) 반응 가능성 |
| **C19** ★ | Excitatory L5 ET | −1.60% | L5 투사 뉴런 감소. 피질-피질하 연결 손상 |
| **C11** ◆ | Excitatory L4 IT | +1.12% | L4 excitatory 증가 경향 |

> **주의**: 코호트 분리 검증에서 PO와 SMC 모두 동일 방향 확인 → 배치 효과 아님
""")

# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("PO vs SMC 코호트 배치 효과 검증")
    st.caption("3'v4(PO, n=9 AD+CT) vs 3'v3(SMC, n=16 AD+CT) 라이브러리 차이가 결과를 오염시키는가?")

    if cohort_df.empty:
        st.warning("cohort_batch_results.tsv 없음")
    else:
        sig_clusters = sorted(cohort_df['cluster'].unique(), key=int)

        for cl in sig_clusters:
            sub = cohort_df[cohort_df['cluster'] == cl].copy()
            subtype_name = sub['subtype'].iloc[0]
            st.markdown(f"#### C{cl} — {subtype_name}")

            col1, col2, col3 = st.columns(3)
            for col, cohort in zip([col1, col2, col3], ['PO', 'SMC', 'ALL']):
                r = sub[sub['cohort'] == cohort]
                if len(r) == 0:
                    continue
                r = r.iloc[0]
                col.metric(
                    label=f"{cohort} (AD={r['n_AD']}, CT={r['n_CT']})",
                    value=f"Δ{r['delta']:+.2f}%",
                    delta=f"p={r['p']:.3f} {r['sig']} {r['direction']}",
                    delta_color='inverse' if r['sig'] in ['**', '*', '†'] else 'off',
                )

            po_row  = sub[sub['cohort'] == 'PO']
            smc_row = sub[sub['cohort'] == 'SMC']
            if len(po_row) > 0 and len(smc_row) > 0:
                po_dir  = po_row.iloc[0]['direction']
                smc_dir = smc_row.iloc[0]['direction']
                if po_dir == smc_dir:
                    st.success(f"✅ 방향 일관성: PO({po_dir}) = SMC({smc_dir}) — 배치 효과 아님")
                else:
                    st.error(f"⚠️ 방향 불일치: PO({po_dir}) ≠ SMC({smc_dir}) — 추가 확인 필요")

            # per-donor scatter
            if not donor_df.empty:
                d = donor_df[donor_df['cluster'] == cl].copy()
                if not d.empty:
                    fig_d = go.Figure()
                    for cohort, coh_color in COHORT_PAL.items():
                        for cond in ['AD', 'Control']:
                            grp = d[(d['cohort'] == cohort) & (d['condition'] == cond)]
                            if grp.empty:
                                continue
                            fig_d.add_trace(go.Box(
                                y=grp['pct'],
                                x=[f"{cohort} {cond}"] * len(grp),
                                name=f"{cohort} {cond}",
                                marker_color=COND_PAL.get(cond, '#888888'),
                                boxpoints='all',
                                jitter=0.3,
                                pointpos=0,
                                marker=dict(
                                    size=8,
                                    symbol='circle',
                                    line=dict(
                                        color='white' if cohort == 'PO' else '#444',
                                        width=1.5,
                                    ),
                                ),
                                line=dict(color=coh_color),
                                customdata=grp[['donor', 'n_cells']].values,
                                hovertemplate=(
                                    '%{customdata[0]}<br>'
                                    '%{y:.2f}% &nbsp; n=%{customdata[1]}<extra></extra>'
                                ),
                            ))

                    fig_d.update_layout(
                        height=280,
                        title=f"C{cl} 도너별 구성 비율",
                        yaxis_title="구성 비율 (%)",
                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                        font=dict(color='white', size=10),
                        showlegend=False,
                        margin=dict(l=50, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_d, use_container_width=True)

            st.divider()

        st.markdown("""
**결론**: 유의 클러스터(C18, C19, C9, C15)는 PO와 SMC 두 코호트 모두에서 동일한 방향의
변화를 보임. 통합 분석에서 유의성이 확보된 것은 power 향상 때문이며 배치 효과가 아님.
""")

# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("전체 30 클러스터 분류표")
    show_cols = ['cluster', 'cell_type', 'subtype', 'layer', 'n',
                 'AD_pct', 'CT_pct', 'delta', 'p', 'sig', 'markers', 'confidence', 'note']
    avail = [c for c in show_cols if c in sub_df.columns]
    tbl = sub_df[avail].sort_values('p', key=lambda x: x.fillna(1)).copy()

    def _row_style(row):
        color = (
            'rgba(231,76,60,0.15)' if row.get('sig') in ['**', '*']
            else 'rgba(243,156,18,0.10)' if row.get('sig') == '†'
            else ''
        )
        return [f'background-color:{color}' if color else '' for _ in row]

    st.dataframe(
        tbl.style.apply(_row_style, axis=1),
        use_container_width=True, height=700,
    )
    st.download_button(
        "📥 TSV 다운로드",
        data=tbl.to_csv(sep='\t', index=False),
        file_name='cell_subtype_composition_AD_CT.tsv',
        mime='text/tab-separated-values',
    )

# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("피질 층별 흥분성 뉴런 구성 변화")
    st.caption("L2/3 · L4 · L5 · L6 서브클러스터 내 AD vs CT 비율 (흥분성 뉴런 전체 대비 %)")

    if mwu_df.empty or adc_df.empty:
        st.warning("layer_composition_*.tsv 파일 없음")
        st.stop()

    # ── MWU 막대 차트 ──────────────────────────────────────────────────────────
    mwu = mwu_df.copy()
    mwu['sig'] = mwu['MWU_p'].apply(
        lambda p: '**' if p < 0.01 else ('*' if p < 0.05 else ('†' if p < 0.10 else 'ns'))
    )
    mwu['color'] = mwu.apply(
        lambda r: '#e74c3c' if (r['delta_pct'] > 0 and r['sig'] != 'ns')
        else '#3498db' if (r['delta_pct'] < 0 and r['sig'] != 'ns')
        else '#555566', axis=1
    )
    mwu['mark'] = mwu['sig'].map(SIG_MARKS)
    mwu_sorted = mwu.sort_values('delta_pct')

    fig_mwu = go.Figure(go.Bar(
        x=mwu_sorted['delta_pct'],
        y=mwu_sorted['cluster_layer'],
        orientation='h',
        marker_color=mwu_sorted['color'],
        text=mwu_sorted['mark'],
        textposition='outside',
        customdata=mwu_sorted[['AD_mean_pct', 'CT_mean_pct', 'MWU_p', 'sig']].values,
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Δ = %{x:+.2f}%<br>'
            'AD %{customdata[0]:.2f}%  CT %{customdata[1]:.2f}%<br>'
            'p = %{customdata[2]:.4f} %{customdata[3]}<extra></extra>'
        ),
    ))
    fig_mwu.add_vline(x=0, line_width=1, line_color='white', opacity=0.4)
    fig_mwu.update_layout(
        height=max(300, len(mwu) * 30),
        xaxis_title='Δ% AD − Control (흥분성 뉴런 내 비율)',
        yaxis_title='',
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='white', size=11),
        margin=dict(l=160, r=80, t=20, b=40),
    )
    st.plotly_chart(fig_mwu, use_container_width=True)

    # ── 샘플별 산점도 ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("샘플별 층 클러스터 비율 분포")

    # 클러스터 열 파악 (C\d+_ 패턴)
    pct_cols = [c for c in adc_df.columns if c.startswith('C') and '_' in c]
    selected_col = st.selectbox(
        "층 클러스터 선택",
        options=pct_cols,
        index=0,
        format_func=lambda c: c.replace('_pct', '').replace('_', ' '),
        key='tab5_col_select',
    )

    adc = adc_df.copy()
    adc['cond_color'] = adc['condition'].map(COND_PAL)

    fig_sc = go.Figure()
    for cond in ['AD', 'Control']:
        grp = adc[adc['condition'] == cond]
        if grp.empty:
            continue
        fig_sc.add_trace(go.Box(
            y=grp[selected_col],
            x=[cond] * len(grp),
            name=cond,
            marker_color=COND_PAL.get(cond, '#888'),
            boxpoints='all',
            jitter=0.3,
            pointpos=0,
            marker=dict(size=9, line=dict(color='white', width=1)),
            customdata=grp[['sample', 'braak_B', 'total_exc']].values,
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                '%{y:.2f}%<br>'
                'Braak B=%{customdata[1]}<br>'
                'total Exc=%{customdata[2]}<extra></extra>'
            ),
        ))

    col_label = selected_col.replace('_pct', '').replace('_', ' ')
    fig_sc.update_layout(
        height=380,
        title=f"{col_label} — AD vs Control 분포",
        yaxis_title="흥분성 뉴런 내 비율 (%)",
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        font=dict(color='white', size=11),
        margin=dict(l=60, r=40, t=50, b=50),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Braak 상관 ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Braak 병기 상관")
    ad_only = adc[adc['condition'] == 'AD']
    if not ad_only.empty and 'braak_B' in ad_only.columns:
        fig_br = go.Figure()
        for cohort, col in COHORT_PAL.items():
            grp = ad_only[ad_only['sample'].str.startswith(cohort)]
            if grp.empty:
                continue
            fig_br.add_trace(go.Scatter(
                x=grp['braak_B'],
                y=grp[selected_col],
                mode='markers',
                name=cohort,
                marker=dict(color=col, size=10, line=dict(color='white', width=1)),
                customdata=grp[['sample']].values,
                hovertemplate='%{customdata[0]}<br>Braak=%{x}  %{y:.2f}%<extra></extra>',
            ))
        fig_br.update_layout(
            height=340,
            xaxis_title='Braak B stage',
            yaxis_title=f'{col_label} (%)',
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
            font=dict(color='white', size=11),
            margin=dict(l=60, r=40, t=20, b=50),
        )
        st.plotly_chart(fig_br, use_container_width=True)
        st.caption("AD 샘플만 표시. Braak B와 층별 클러스터 비율의 상관 관계.")

# ── 페이지 간 이동 ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### 관련 페이지")
col_nav1, col_nav2, col_nav3 = st.columns(3)
col_nav1.page_link("pages/12_brain3d.py",        label="🧠 뇌 3D 세포 지도",  help="3D 해부학적 지도 + UMAP 공간")
col_nav2.page_link("pages/14_bisect_context.py",  label="🔬 BISECT 세포 맥락", help="세포 타입별 아이소폼 스위치")
col_nav3.page_link("pages/07_bisect.py",          label="🧫 BISECT Cases",     help="개별 케이스 상세 분석")
