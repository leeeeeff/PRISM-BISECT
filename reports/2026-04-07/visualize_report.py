"""
DIFFUSE 연구 분석 레포트 (2026-04-07) — 성능 비교 시각화
research_report_20260407.md + milestone_report_20260407.md 기반
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 공통 스타일 ────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm
_noto_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(_noto_path)
_fp = fm.FontProperties(fname=_noto_path)
_font_name = _fp.get_name()

plt.rcParams.update({
    'font.family': _font_name,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

GO_LABELS = {
    'GO_0006936': 'GO_0006936\n(Muscle contraction)',
    'GO_0006412': 'GO_0006412\n(Translation)',
    'GO_0006096': 'GO_0006096\n(Glycolysis)',
    'GO_0022900': 'GO_0022900\n(ETC)',
}
GO_KEYS   = list(GO_LABELS.keys())
GO_SHORT  = ['0006936\n근육수축', '0006412\n번역', '0006096\n해당과정', '0022900\nETC']
RANDOM_AUPRC = [0.016, 0.019, 0.002, 0.008]

VERSION_COLORS = {
    'v4-3':  '#4e79a7',
    'v5-2':  '#f28e2b',
    'v5-3':  '#e15759',
    'v5-4':  '#76b7b2',
    'v5-5':  '#59a14f',
}

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1 — 전체 버전 Macro-AUPRC 변화 추이
# ═══════════════════════════════════════════════════════════════════════════════
def chart1_macro_auprc():
    versions = ['v4-3', 'v5-2', 'v5-3', 'v5-4', 'v5-5']
    macro    = [0.178,   0.221,  0.076,  0.231,  0.3717]
    colors   = [VERSION_COLORS[v] for v in versions]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(versions, macro, color=colors, width=0.55, edgecolor='white', linewidth=1.2)

    # 값 표시
    for bar, val in zip(bars, macro):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.006,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # v5-3 회귀 강조
    ax.annotate('⚠ 회귀\nmargin 0.3→0.1\n오진', xy=(2, macro[2]+0.005),
                xytext=(2.45, 0.16),
                arrowprops=dict(arrowstyle='->', color='#e15759', lw=1.5),
                color='#e15759', fontsize=9, ha='center')

    # v5-5 최고 성능 강조
    ax.annotate('★ 최고 성능\n+60.9% vs v5-4\ncoverage 기반\n동적 n_batches', xy=(4, macro[4]+0.005),
                xytext=(3.5, 0.34),
                arrowprops=dict(arrowstyle='->', color='#59a14f', lw=1.5),
                color='#59a14f', fontsize=9, ha='center')

    ax.set_ylim(0, 0.46)
    ax.set_ylabel('Macro-AUPRC (4개 GO term 평균)')
    ax.set_title('Chart 1 — 전체 버전 Macro-AUPRC 변화 추이\n(Primary Metric: AUPRC [R9.1] — Positive Collapse 탐지 가능한 지표)')
    ax.set_xlabel('모델 버전')
    ax.axhline(y=0.178, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(4.6, 0.180, 'v4-3 기준선', fontsize=8.5, color='gray', va='bottom')

    legend_text = (
        "• Macro-AUPRC: 4개 GO term AUPRC의 산술 평균 (전체 모델 성능 요약)\n"
        "• v5-3: margin 0.3→0.1 변경 오진으로 전 GO term 동시 악화 (−57%)\n"
        "• v5-5: coverage 기반 동적 n_batches 도입 → v4-3 대비 +109%, v5-4 대비 +61%\n"
        "• Random baseline (클래스 비율 기준 AUPRC): GO_0006936=0.016, GO_0006096=0.002"
    )
    fig.text(0.5, -0.04, legend_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', alpha=0.8))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(os.path.join(OUT_DIR, 'chart1_macro_auprc_trend.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Chart 1 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2 — GO term별 AUPRC 버전 비교 (grouped bar)
# ═══════════════════════════════════════════════════════════════════════════════
def chart2_per_goterm_auprc():
    versions = ['v4-3', 'v5-2', 'v5-3', 'v5-4', 'v5-5']
    # rows: GO term, cols: version
    data = {
        'GO_0006936': [0.296, 0.410, 0.119, 0.191, 0.2507],
        'GO_0006412': [0.082, 0.089, 0.075, 0.093, 0.2091],
        'GO_0006096': [0.258, 0.354, 0.095, 0.297, 0.6698],
        'GO_0022900': [0.074, 0.031, 0.016, 0.341, 0.3572],
    }

    n_go  = 4
    n_ver = 5
    x     = np.arange(n_go)
    w     = 0.14
    offsets = np.linspace(-(n_ver-1)/2, (n_ver-1)/2, n_ver) * w

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, ver in enumerate(versions):
        vals = [data[go][i] for go in GO_KEYS]
        bars = ax.bar(x + offsets[i], vals, width=w, label=ver,
                      color=VERSION_COLORS[ver], edgecolor='white', linewidth=0.8)

    # random baseline
    for j, (gk, rv) in enumerate(zip(GO_KEYS, RANDOM_AUPRC)):
        ax.hlines(rv, j-0.4, j+0.4, colors='gray', linestyles=':', linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(GO_SHORT, fontsize=10)
    ax.set_ylabel('AUPRC')
    ax.set_title('Chart 2 — GO Term별 AUPRC 버전 비교\n(점선: 각 GO term의 random baseline ≈ positive class 비율)')
    ax.legend(title='버전', ncol=5, loc='upper left',
              bbox_to_anchor=(0, -0.02), frameon=True)
    ax.set_ylim(0, 0.80)

    # v5-5 GO_0006096 강조
    ax.annotate('+125%\nvs v5-4', xy=(2 + offsets[4], 0.6698),
                xytext=(2.5, 0.68),
                arrowprops=dict(arrowstyle='->', color='#59a14f', lw=1.5),
                color='#59a14f', fontsize=9, fontweight='bold')

    legend_text = (
        "• GO_0006096 (Glycolysis, 76 positive): v5-5에서 0.297→0.670, +125% 폭발적 개선\n"
        "• GO_0022900 (ETC): v5-4에서 0.031→0.341 극적 개선 후 v5-5에서 유지(0.357)\n"
        "• GO_0006412 (Translation): 모든 버전에서 낮음 — 세포 전반 기능으로 이소폼 수준 차별화 어려움\n"
        "• 점선(gray): 각 GO term의 random AUPRC. 이 값 이상일 때만 의미있는 예측으로 인정"
    )
    fig.text(0.5, -0.10, legend_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', alpha=0.8))
    plt.tight_layout(rect=[0, 0.14, 1, 1])
    fig.savefig(os.path.join(OUT_DIR, 'chart2_goterm_auprc_versions.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Chart 2 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3 — CRF 파괴 효과: Phase 2 vs Final (v1→v4-3 vs v5-2)
# ═══════════════════════════════════════════════════════════════════════════════
def chart3_crf_damage():
    # 4 GO terms x versions with Phase2 / Final
    versions_old = ['v1', 'v3', 'v4-3']
    # Phase 2 AUPRC
    p2 = {
        'GO_0006936': [0.101, 0.387, 0.296],
        'GO_0006412': [0.025, 0.052, 0.082],
        'GO_0006096': [0.527, 0.493, 0.258],
        'GO_0022900': [0.191, 0.087, 0.074],
    }
    # Final (CRF) AUPRC
    final_crf = {
        'GO_0006936': [0.019, 0.017, 0.017],
        'GO_0006412': [0.020, 0.019, 0.019],
        'GO_0006096': [0.002, 0.002, 0.002],
        'GO_0022900': [0.008, 0.009, 0.008],
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 5.5), sharey=False)
    fig.suptitle('Chart 3 — CRF 파괴 효과: Phase 2 예측 vs CRF 적용 후 최종 예측\n'
                 '(v1~v4-3 전 버전에서 CRF Phase 3가 Phase 2의 예측 품질을 random 수준으로 붕괴)',
                 fontsize=13, fontweight='bold')

    x = np.arange(len(versions_old))
    for ax, gk, gs in zip(axes, GO_KEYS, GO_SHORT):
        p2_vals    = p2[gk]
        final_vals = final_crf[gk]
        rv = RANDOM_AUPRC[GO_KEYS.index(gk)]

        ax.bar(x - 0.18, p2_vals,    width=0.34, label='Phase 2 AUPRC',  color='#4e79a7', alpha=0.9)
        ax.bar(x + 0.18, final_vals, width=0.34, label='CRF 후 Final',   color='#e15759', alpha=0.9)
        ax.axhline(rv, color='gray', linestyle=':', linewidth=1.2)
        ax.text(2.5, rv + 0.005, f'random\n({rv:.3f})', fontsize=7.5, color='gray')

        ax.set_title(f'GO {gs}', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(versions_old, fontsize=9)
        ax.set_ylabel('AUPRC' if ax == axes[0] else '')

        # 손실률 표시 (v3 기준)
        loss = (p2_vals[1] - final_vals[1]) / p2_vals[1] * 100
        ax.text(1, max(p2_vals[1], final_vals[1]) + 0.01,
                f'−{loss:.0f}%', color='#e15759', fontsize=9, ha='center', fontweight='bold')

    axes[0].legend(loc='upper left', fontsize=8)
    # v5-2 개선 화살표 (마지막 패널에)
    axes[-1].annotate('v5-2에서\nCRF 제거\n→ 해결', xy=(2.5, 0.008), xytext=(2.5, 0.06),
                      arrowprops=dict(arrowstyle='->', color='#59a14f', lw=1.5),
                      color='#59a14f', fontsize=8.5, ha='center')

    legend_text = (
        "• Phase 2 (파란 막대): Focal + Triplet Loss 결합 학습 후 예측 품질\n"
        "• CRF 후 Final (빨간 막대): 원본 DIFFUSE CRF(gene-level message passing) 적용 후 최종 출력\n"
        "• CRF는 within-gene isoform 간 점수를 평균화 → discriminative score 소멸 → random 수준으로 붕괴\n"
        "• v5-2에서 CRF를 Expression Label Propagation으로 교체하여 근본 해결"
    )
    fig.text(0.5, -0.06, legend_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', alpha=0.8))
    plt.tight_layout(rect=[0, 0.10, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, 'chart3_crf_damage.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Chart 3 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4 — Phase별 Embedding 품질: v5-2 vs v5-3 (GO_0006936)
# ═══════════════════════════════════════════════════════════════════════════════
def chart4_phase_embedding_v52_v53():
    phases = ['Ph0\n(untrained)', 'Ph1\n(triplet)', 'Ph1.5\n(linear)', 'Ph2\n(joint)', 'Final\n(LP)']
    sil_v52 = [0.162, 0.473, 0.523, 0.733, None]
    sil_v53 = [0.031, 0.045, 0.144, 0.316, None]
    pred_v52 = [0.469, 0.678, 0.696, 0.775, 0.805]
    pred_v53 = [0.527, 0.433, 0.661, 0.706, 0.712]

    x = np.arange(len(phases))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Chart 4 — Phase별 Embedding 품질 비교: v5-2 vs v5-3 (GO_0006936, 근육수축)\n'
                 'margin=0.3(v5-2) vs margin=0.1(v5-3) 효과 비교 — Silhouette & PredAUROC',
                 fontsize=12, fontweight='bold')

    # Silhouette
    sil_v52_plot = [v if v is not None else np.nan for v in sil_v52]
    sil_v53_plot = [v if v is not None else np.nan for v in sil_v53]
    ax1.plot(x[:4], sil_v52_plot[:4], 'o-', color='#f28e2b', linewidth=2.2, markersize=8, label='v5-2 (margin=0.3)')
    ax1.plot(x[:4], sil_v53_plot[:4], 's--', color='#e15759', linewidth=2.2, markersize=8, label='v5-3 (margin=0.1)')
    ax1.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax1.fill_between(x[:4], sil_v53_plot[:4], sil_v52_plot[:4], alpha=0.12, color='#f28e2b')
    ax1.set_xticks(x[:4])
    ax1.set_xticklabels(phases[:4])
    ax1.set_ylabel('Silhouette Score (-1 ~ +1)')
    ax1.set_title('Embedding 분리도 (Silhouette)')
    ax1.legend()
    ax1.set_ylim(-0.05, 0.85)
    # 주석
    ax1.annotate(f'v5-2: +0.473\n(v5-3×10배)', xy=(1, 0.473), xytext=(1.4, 0.58),
                 arrowprops=dict(arrowstyle='->', lw=1.2), fontsize=8.5, color='#f28e2b')
    ax1.annotate(f'v5-3: +0.045', xy=(1, 0.045), xytext=(0.1, -0.03),
                 arrowprops=dict(arrowstyle='->', lw=1.2), fontsize=8.5, color='#e15759')

    # PredAUROC
    ax2.plot(x, pred_v52, 'o-', color='#f28e2b', linewidth=2.2, markersize=8, label='v5-2 (margin=0.3)')
    ax2.plot(x, pred_v53, 's--', color='#e15759', linewidth=2.2, markersize=8, label='v5-3 (margin=0.1)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(phases)
    ax2.set_ylabel('Prediction AUROC')
    ax2.set_title('예측 AUROC (각 Phase 직후 평가)')
    ax2.legend()
    ax2.set_ylim(0.4, 0.88)
    ax2.axhline(0.5, color='gray', linestyle=':', linewidth=1)
    ax2.text(4.1, 0.805, '0.805\n★', color='#f28e2b', fontsize=9, fontweight='bold')
    ax2.text(4.1, 0.712, '0.712', color='#e15759', fontsize=9)

    legend_text = (
        "• Silhouette: 임베딩 공간에서 양성/음성 클러스터 분리도 (-1~+1). 양수일수록 잘 분리됨\n"
        "• margin=0.3(v5-2): 15 epoch 내내 gradient 소멸 없이 유지 → Ph1 Silhouette +0.473 (v5-3의 10배)\n"
        "• margin=0.1(v5-3): epoch 7~9에서 active ratio 0.9~1.7%로 조기 gradient 소멸 → Ph1 Silhouette +0.045\n"
        "• 핵심 역설: v5-2에서 margin_sat=0.0% (한 번도 만족 안 됨)인데도 최고 성능 → 'frustrated triplet' 효과"
    )
    fig.text(0.5, -0.06, legend_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', alpha=0.8))
    plt.tight_layout(rect=[0, 0.10, 1, 0.92])
    fig.savefig(os.path.join(OUT_DIR, 'chart4_phase_embedding_v52_v53.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Chart 4 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 5 — v5-5 전체 GO term Phase별 Silhouette + AUPRC (heatmap + bar)
# ═══════════════════════════════════════════════════════════════════════════════
def chart5_v55_phase_quality():
    phases = ['Ph0', 'Ph1', 'Ph1.5', 'Ph2']
    # silhouette
    sil = np.array([
        [+0.027, +0.129, +0.363, -0.050],   # GO_0006936
        [-0.019, -0.038, +0.155, +0.363],   # GO_0006412
        [-0.123, +0.397, +0.604, +0.719],   # GO_0006096
        [-0.265, -0.267, -0.198, +0.585],   # GO_0022900
    ])
    lin_auroc = [0.772, 0.818, 0.929, 0.888]
    final_auprc = [0.2507, 0.2091, 0.6698, 0.3572]

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle('Chart 5 — v5-5 Phase별 Embedding 품질 전체 GO term\n'
                 '(Silhouette heatmap + Ph2 Linear AUROC + 최종 AUPRC)',
                 fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.5, 1, 1], wspace=0.4)

    # Heatmap
    ax1 = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list('rg', ['#e15759', '#ffffff', '#59a14f'])
    im = ax1.imshow(sil, cmap=cmap, vmin=-0.3, vmax=0.75, aspect='auto')
    ax1.set_xticks(range(len(phases)))
    ax1.set_xticklabels(phases)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(GO_SHORT, fontsize=9)
    ax1.set_title('Phase별 Silhouette Score')
    plt.colorbar(im, ax=ax1, shrink=0.8, label='Silhouette')
    for i in range(4):
        for j in range(4):
            val = sil[i, j]
            color = 'white' if abs(val) > 0.4 else 'black'
            ax1.text(j, i, f'{val:+.3f}', ha='center', va='center',
                     fontsize=9.5, color=color, fontweight='bold')
    # GO_0006096 Ph2 강조
    rect = plt.Rectangle((3-0.5, 2-0.5), 1, 1, linewidth=2.5, edgecolor='gold', facecolor='none')
    ax1.add_patch(rect)
    ax1.text(3, 2 - 0.65, '★최고', ha='center', fontsize=8, color='gold', fontweight='bold')

    # Ph2 Linear AUROC
    ax2 = fig.add_subplot(gs[1])
    colors_bar = [VERSION_COLORS['v5-2'], VERSION_COLORS['v5-3'], '#59a14f', VERSION_COLORS['v5-4']]
    bars2 = ax2.barh(range(4), lin_auroc, color=['#4e79a7','#f28e2b','#59a14f','#76b7b2'],
                     height=0.55, edgecolor='white')
    for bar, val in zip(bars2, lin_auroc):
        ax2.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=9.5, fontweight='bold')
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(GO_SHORT, fontsize=9)
    ax2.set_xlim(0.6, 1.0)
    ax2.axvline(0.85, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.text(0.851, 3.6, '>0.85\n=우수', fontsize=7.5, color='gray')
    ax2.set_title('Ph2\nLinear AUROC', fontsize=10)
    ax2.set_xlabel('AUROC')

    # Final AUPRC
    ax3 = fig.add_subplot(gs[2])
    bars3 = ax3.barh(range(4), final_auprc, color='#59a14f', height=0.55, edgecolor='white', alpha=0.9)
    for bar, val, rv in zip(bars3, final_auprc, RANDOM_AUPRC):
        ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9.5, fontweight='bold', color='#59a14f')
        ax3.axvline(rv, ymin=(bar.get_y()-0.05)/4, ymax=(bar.get_y()+bar.get_height()+0.05)/4,
                    color='gray', linestyle=':', linewidth=1.2)
    ax3.set_yticks(range(4))
    ax3.set_yticklabels(GO_SHORT, fontsize=9)
    ax3.set_xlim(0, 0.82)
    ax3.set_title('최종 AUPRC\n(v5-5)', fontsize=10)
    ax3.set_xlabel('AUPRC')

    legend_text = (
        "• Silhouette heatmap: 빨강(-) = embedding이 역방향 구조, 흰색(0) = random, 초록(+) = 잘 분리\n"
        "• GO_0006096 (★): Ph2 Silhouette=+0.719, LinAUROC=0.929 → 전 버전 통틀어 최고 품질\n"
        "• GO_0022900: Ph0~Ph1.5 Silhouette 음수였다가 Ph2에서 +0.585로 회복 → Phase 2 AUPRC early stop 효과\n"
        "• Linear AUROC >0.85 기준선: encoder가 이미 선형 분리 가능 수준의 표현을 학습했음을 의미"
    )
    fig.text(0.5, -0.06, legend_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', alpha=0.8))
    plt.tight_layout(rect=[0, 0.10, 1, 0.92])
    fig.savefig(os.path.join(OUT_DIR, 'chart5_v55_phase_quality.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Chart 5 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 6 — Positive Collapse: v5-4 vs v5-5 예측 분포 비교
# ═══════════════════════════════════════════════════════════════════════════════
def chart6_positive_collapse():
    go_terms = GO_SHORT
    # v5-4
    v54_pct = [100.0, 99.9, 91.5, 100.0]
    v54_std  = [0.062, 0.045, 0.122, 0.018]
    v54_auprc = [0.191, 0.093, 0.297, 0.341]
    # v5-5
    v55_pct  = [99.7, 58.0, 100.0, 100.0]
    v55_std  = [0.1103, 0.1108, 0.0369, 0.0280]
    v55_auprc = [0.2507, 0.2091, 0.6698, 0.3572]

    x = np.arange(4)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle('Chart 6 — Positive Collapse 상태: v5-4 vs v5-5 예측 분포 비교\n'
                 '(score >0.5 예측 비율, 예측 분포 표준편차, 최종 AUPRC)',
                 fontsize=12, fontweight='bold')

    w = 0.35
    # >0.5 예측 비율
    ax = axes[0]
    ax.bar(x - w/2, v54_pct, width=w, label='v5-4', color='#76b7b2', alpha=0.85)
    ax.bar(x + w/2, v55_pct, width=w, label='v5-5', color='#59a14f', alpha=0.85)
    ax.axhline(100, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(1.9, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(3.6, 2.5, '실제\npositive\n비율~2%', fontsize=7.5, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(go_terms, fontsize=9)
    ax.set_ylabel('score > 0.5 예측 비율 (%)')
    ax.set_title('Positive Collapse 지표\n(>0.5 예측 비율)')
    ax.legend()
    # GO_0006412 개선 강조
    ax.annotate('58%\n(정상화!)', xy=(1 + w/2, 58), xytext=(1.8, 70),
                arrowprops=dict(arrowstyle='->', color='#59a14f', lw=1.5),
                color='#59a14f', fontsize=9, fontweight='bold')

    # 표준편차
    ax = axes[1]
    ax.bar(x - w/2, v54_std, width=w, label='v5-4', color='#76b7b2', alpha=0.85)
    ax.bar(x + w/2, v55_std, width=w, label='v5-5', color='#59a14f', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(go_terms, fontsize=9)
    ax.set_ylabel('예측 score 표준편차')
    ax.set_title('예측 분포 퍼짐 (std)\n(높을수록 선택적 예측)')
    ax.legend()

    # AUPRC
    ax = axes[2]
    ax.bar(x - w/2, v54_auprc, width=w, label='v5-4', color='#76b7b2', alpha=0.85)
    ax.bar(x + w/2, v55_auprc, width=w, label='v5-5', color='#59a14f', alpha=0.85)
    for j, (v4, v5) in enumerate(zip(v54_auprc, v55_auprc)):
        change = (v5 - v4) / v4 * 100
        color = '#59a14f' if change > 0 else '#e15759'
        ax.text(j + w/2, v5 + 0.01, f'{change:+.0f}%', ha='center', fontsize=8.5,
                color=color, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(go_terms, fontsize=9)
    ax.set_ylabel('AUPRC')
    ax.set_title('최종 AUPRC\n(v5-4 → v5-5 변화율)')
    ax.legend()

    legend_text = (
        "• Positive Collapse: 모든/대부분의 이소폼을 양성으로 예측 → Recall↑ but Precision↓, AUPRC 의미 없음\n"
        "• GO_0006412: v5-4에서 99.9% → v5-5에서 58%로 정상화 → AUPRC 0.093→0.2091 (+125%)\n"
        "• GO_0006096: 100% positive 예측이지만 ranking이 정확 → AUPRC=0.670 (ranking metric이므로 가능)\n"
        "• 표준편차가 높을수록 모델이 선별적 예측(confidence 차별화), 낮을수록 전체를 비슷하게 예측"
    )
    fig.text(0.5, -0.06, legend_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', alpha=0.8))
    plt.tight_layout(rect=[0, 0.10, 1, 0.92])
    fig.savefig(os.path.join(OUT_DIR, 'chart6_positive_collapse.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Chart 6 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 7 — n_batches coverage 비교 및 v5-4 → v5-5 인과 분석
# ═══════════════════════════════════════════════════════════════════════════════
def chart7_coverage_analysis():
    go_terms = GO_SHORT
    n_pos = [840, 3046, 287, 1027]
    cov_v52   = [6.1,  1.7,  17.8, 5.0]
    cov_v54   = [15.2, 4.2,  44.6, 12.5]
    cov_v55   = [6.1,  4.2,  17.8, 6.2]
    nb_v55    = [20, 50, 20, 25]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Chart 7 — n_batches Coverage 분석: 버전별 비교 및 동적 할당 원리\n'
                 '(Coverage = n_batches × 256 / n_pos, target=6.0x)',
                 fontsize=12, fontweight='bold')

    x = np.arange(4)
    w = 0.25

    # Coverage 비교
    ax = axes[0]
    ax.bar(x - w, cov_v52, width=w, label='v5-2 (n=20 fixed)', color='#f28e2b', alpha=0.85)
    ax.bar(x,     cov_v54, width=w, label='v5-4 (n=50 fixed)', color='#76b7b2', alpha=0.85)
    ax.bar(x + w, cov_v55, width=w, label='v5-5 (dynamic)', color='#59a14f', alpha=0.85)
    ax.axhline(6.0, color='navy', linestyle='--', linewidth=1.8, label='Target coverage (6x)')
    ax.set_xticks(x)
    ax.set_xticklabels(go_terms, fontsize=9)
    ax.set_ylabel('Coverage (× per epoch)')
    ax.set_title('Phase 1 epoch당 Coverage 비교\n(같은 양성 이소폼을 epoch당 몇 회 학습)')
    ax.legend(fontsize=8.5)
    ax.set_ylim(0, 55)
    # 44.6x 강조
    ax.annotate('44.6×\n→ 과적합\n유발', xy=(2, 44.6), xytext=(2.6, 46),
                arrowprops=dict(arrowstyle='->', color='#e15759', lw=1.5),
                color='#e15759', fontsize=8.5, ha='center')

    # v5-4 vs v5-5 GO_0006096 인과 분석
    ax = axes[1]
    phases = ['Ph1\nSilhouette', 'Ph1.5\nSilhouette', 'Ph2\nSilhouette', 'Ph2\nLinAUROC×10', 'Final\nAUPRC×10']
    v54_vals = [0.459, 0.511, 0.195, 0.7*10/10, 0.297]
    v54_vals_scaled = [0.459, 0.511, 0.195, 0.7, 0.297]   # LinAUROC 그대로
    v55_vals_scaled = [0.397, 0.604, 0.719, 0.929, 0.6698]

    xp = np.arange(len(phases))
    ax.plot(xp, v54_vals_scaled, 'o-', color='#76b7b2', linewidth=2.2, markersize=8,
            label='v5-4 (n=50, cov=44.6x)')
    ax.plot(xp, v55_vals_scaled, 's-', color='#59a14f', linewidth=2.2, markersize=8,
            label='v5-5 (n=20, cov=17.8x)')
    ax.fill_between(xp, v54_vals_scaled, v55_vals_scaled, alpha=0.12, color='#59a14f')
    ax.set_xticks(xp)
    ax.set_xticklabels(phases, fontsize=9)
    ax.set_ylabel('값 (AUPRC는 원래 범위)')
    ax.set_title('GO_0006096: v5-4 vs v5-5 각 Phase 품질\n(n_batches 50→20, coverage 44.6×→17.8×)')
    ax.legend(fontsize=8.5)
    ax.text(4, 0.6698 + 0.01, '★ AUPRC\n0.6698', color='#59a14f', fontsize=9, fontweight='bold', ha='center')
    ax.text(4, 0.297 - 0.04, '0.297', color='#76b7b2', fontsize=9, ha='center')

    legend_text = (
        "• v5-2 GO_0006096: n=20이지만 n_pos=287이 작아 coverage=17.8x — v5-5와 동일, 성능 우수\n"
        "• v5-4 GO_0006096: n=50 → coverage=44.6x → 과도한 반복으로 triplet diversity 감소 → Phase 1 과적합\n"
        "• v5-5 동적 공식: n = clip(ceil(n_pos × 6.0 / 256), 20, 50) → 각 GO term별 target coverage 6x 목표\n"
        "• 핵심 발견: 학습량(n_batches) 감소가 GO_0006096 AUPRC를 0.297→0.670으로 폭발적 개선 (과적합 해소)"
    )
    fig.text(0.5, -0.06, legend_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', alpha=0.8))
    plt.tight_layout(rect=[0, 0.10, 1, 0.92])
    fig.savefig(os.path.join(OUT_DIR, 'chart7_coverage_analysis.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Chart 7 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 8 — LabelProp alpha 선택 효과 (v5-5)
# ═══════════════════════════════════════════════════════════════════════════════
def chart8_labelprop_alpha():
    alphas = [0.0, 0.2, 0.3, 0.5]
    data = {
        'GO_0006936': [0.2507, 0.2091, 0.1702, 0.0882],
        'GO_0006412': [0.2091, 0.1990, 0.1847, 0.1563],
        'GO_0006096': [0.6698, 0.6282, 0.6275, 0.6313],
        'GO_0022900': [0.3438, 0.3486, 0.3525, 0.3572],
    }
    best_alpha = [0.0, 0.0, 0.0, 0.5]

    colors_go = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.suptitle('Chart 8 — LabelProp Alpha 선택 효과 (v5-5)\n'
                 '(Expression KNN graph propagation 강도 α ∈ {0.0, 0.2, 0.3, 0.5})',
                 fontsize=12, fontweight='bold')

    for i, (gk, gs) in enumerate(zip(GO_KEYS, GO_SHORT)):
        vals = data[gk]
        ax.plot(alphas, vals, 'o-', color=colors_go[i], linewidth=2,
                markersize=8, label=f'GO {gs.split(chr(10))[0]}')
        # 선택된 alpha 표시
        best_i = alphas.index(best_alpha[i])
        ax.scatter([best_alpha[i]], [vals[best_i]], s=150, color=colors_go[i],
                   edgecolors='black', linewidth=2, zorder=5)

    ax.set_xlabel('LabelProp alpha (α)')
    ax.set_ylabel('AUPRC')
    ax.set_title('')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xticks(alphas)
    ax.set_xticklabels([f'α={a}\n({"비활성" if a==0 else "전파"})' for a in alphas])

    # 범례 마커
    ax.scatter([], [], s=150, color='gray', edgecolors='black', linewidth=2,
               label='▶ 선택된 alpha (AUPRC 최대)', zorder=5)
    ax.legend(fontsize=9)

    # GO_0022900 설명
    ax.annotate('GO_0022900만\nα=0.5에서\n개선', xy=(0.5, 0.3572), xytext=(0.35, 0.38),
                arrowprops=dict(arrowstyle='->', color='#e15759', lw=1.3),
                fontsize=9, color='#e15759')

    legend_text = (
        "• alpha=0.0: LabelProp 비활성 (Phase 2 score 그대로 사용)\n"
        "• alpha>0: score_final = (1-α)×score_ph2 + α×neighbor_mean (이소폼 발현 이웃의 평균 점수로 보완)\n"
        "• 3개 GO term(근육수축·번역·해당과정): alpha=0.0 최적 → Phase 2 score가 이미 최적 상태\n"
        "• GO_0022900 (ETC)만 alpha=0.5 최적 → expression neighbor 정보가 전자전달계 예측에 실제로 기여"
    )
    fig.text(0.5, -0.06, legend_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', alpha=0.8))
    plt.tight_layout(rect=[0, 0.10, 1, 0.92])
    fig.savefig(os.path.join(OUT_DIR, 'chart8_labelprop_alpha.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Chart 8 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 9 — 전체 버전 히스토리 AUPRC 히트맵 (v1~v5-5)
# ═══════════════════════════════════════════════════════════════════════════════
def chart9_full_history_heatmap():
    versions = ['v1\nPhase2', 'v1\nCRF후', 'v3\nPhase2', 'v3\nCRF후',
                'v4-3\nPhase2', 'v4-3\nCRF후', 'v5-2\n★', 'v5-3\n↓', 'v5-4', 'v5-5\n★★']
    data = np.array([
        # GO_0006936
        [0.101, 0.019, 0.387, 0.017, 0.296, 0.017, 0.410, 0.119, 0.191, 0.2507],
        # GO_0006412
        [0.025, 0.020, 0.052, 0.019, 0.082, 0.019, 0.089, 0.075, 0.093, 0.2091],
        # GO_0006096
        [0.527, 0.002, 0.493, 0.002, 0.258, 0.002, 0.354, 0.095, 0.297, 0.6698],
        # GO_0022900
        [0.191, 0.008, 0.087, 0.009, 0.074, 0.008, 0.031, 0.016, 0.341, 0.3572],
    ])

    fig, ax = plt.subplots(figsize=(15, 5))
    fig.suptitle('Chart 9 — 전체 버전 AUPRC 히트맵 (v1 ~ v5-5)\n'
                 '(CRF 파괴 → CRF 제거 → 성능 안정화 전 과정 시각화)',
                 fontsize=12, fontweight='bold')

    cmap = LinearSegmentedColormap.from_list('auprc',
           ['#f0f0f0', '#ffffcc', '#a1dab4', '#41b6c4', '#225ea8'])
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=0.70, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8, label='AUPRC')

    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions, fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(GO_SHORT, fontsize=9)

    # 값 표시
    for i in range(4):
        for j in range(len(versions)):
            val = data[i, j]
            color = 'white' if val > 0.4 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8.5, color=color)

    # CRF 구간 강조
    for j in [1, 3, 5]:
        rect = plt.Rectangle((j-0.5, -0.5), 1, 4, linewidth=2,
                              edgecolor='#e15759', facecolor='#e15759', alpha=0.12)
        ax.add_patch(rect)
    ax.text(3.0, -0.7, '← CRF 적용 버전 (빨간 음영) →',
            ha='center', fontsize=9, color='#e15759', fontweight='bold')

    # v5-2, v5-5 강조
    for j, label in [(6, 'v5-2\n(CRF 제거)'), (9, 'v5-5\n(최고)')]:
        rect = plt.Rectangle((j-0.5, -0.5), 1, 4, linewidth=2.5,
                              edgecolor='gold', facecolor='none')
        ax.add_patch(rect)

    legend_text = (
        "• 빨간 음영(CRF 후): 전 GO term에서 AUPRC가 random 수준(≤0.020)으로 붕괴\n"
        "• 금색 테두리: v5-2(CRF 제거, 첫 실질적 성능), v5-5(현재 최고 성능)\n"
        "• GO_0006096 세로 컬럼: v3 Phase2=0.493 → CRF=0.002 → v5-5=0.670 (역대 최고)\n"
        "• v5-3는 회귀 구간 — margin 0.3→0.1 오진으로 전 GO term 동시 악화"
    )
    fig.text(0.5, -0.10, legend_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', alpha=0.8))
    plt.tight_layout(rect=[0, 0.12, 1, 0.92])
    fig.savefig(os.path.join(OUT_DIR, 'chart9_full_history_heatmap.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Chart 9 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 10 — AUROC vs AUPRC 차이: Positive Collapse 감지 능력
# ═══════════════════════════════════════════════════════════════════════════════
def chart10_auroc_vs_auprc():
    # v5-4 GO term별 AUROC vs AUPRC
    go_labels = GO_SHORT
    auroc_v54 = [0.685, 0.653, 0.720, 0.869]
    auprc_v54 = [0.191, 0.093, 0.297, 0.341]
    auroc_v55 = [0.7225, 0.6147, 0.8837, 0.8166]
    auprc_v55 = [0.2507, 0.2091, 0.6698, 0.3572]
    # >0.5 예측 비율 v5-4
    collapse_v54 = [100.0, 99.9, 91.5, 100.0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Chart 10 — AUROC vs AUPRC: Positive Collapse 감지 능력 차이\n'
                 '(v5-4 / v5-5 기준, AUROC는 class imbalance에서 model failure를 못 잡음)',
                 fontsize=12, fontweight='bold')

    # v5-4: AUROC vs AUPRC scatter
    ax = axes[0]
    sc = ax.scatter(auroc_v54, auprc_v54, s=[c*3 for c in collapse_v54],
                    c=collapse_v54, cmap='RdYlGn_r', vmin=50, vmax=100,
                    edgecolors='black', linewidth=1.2, zorder=5, alpha=0.85)
    plt.colorbar(sc, ax=ax, label='>0.5 예측 비율 (%)')
    for i, gs in enumerate(go_labels):
        ax.annotate(gs.split('\n')[0], (auroc_v54[i], auprc_v54[i]),
                    textcoords='offset points', xytext=(8, 4), fontsize=8.5)
    ax.axhline(0.02, color='gray', linestyle=':', linewidth=1.2)
    ax.text(0.88, 0.025, 'random\nbaseline', fontsize=7.5, color='gray')
    ax.set_xlabel('AUROC')
    ax.set_ylabel('AUPRC')
    ax.set_title('v5-4: AUROC vs AUPRC 산포도\n(원 크기 & 색: >0.5 예측 비율)')
    ax.set_xlim(0.6, 0.93)
    ax.set_ylim(0, 0.42)
    # 설명 화살표
    ax.annotate('AUROC=0.685\n→ 그럭저럭\n실제 100%\npositive 예측!',
                xy=(0.685, 0.191), xytext=(0.63, 0.32),
                arrowprops=dict(arrowstyle='->', color='#e15759', lw=1.3),
                color='#e15759', fontsize=8, ha='center')

    # v5-4 vs v5-5 AUROC와 AUPRC 변화 비교
    ax = axes[1]
    x = np.arange(4)
    w = 0.2
    bars_auroc_v54 = ax.bar(x - 1.5*w, auroc_v54, width=w, label='v5-4 AUROC', color='#76b7b2', alpha=0.75)
    bars_auroc_v55 = ax.bar(x - 0.5*w, auroc_v55, width=w, label='v5-5 AUROC', color='#4e79a7', alpha=0.75)
    bars_auprc_v54 = ax.bar(x + 0.5*w, auprc_v54, width=w, label='v5-4 AUPRC', color='#f28e2b', alpha=0.85)
    bars_auprc_v55 = ax.bar(x + 1.5*w, auprc_v55, width=w, label='v5-5 AUPRC', color='#59a14f', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(go_labels, fontsize=9)
    ax.set_ylabel('값')
    ax.set_title('v5-4 vs v5-5: AUROC와 AUPRC 동시 비교\n(AUPRC는 버전간 큰 차이, AUROC는 상대적으로 안정)')
    ax.legend(ncol=2, fontsize=8.5, loc='upper right')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6)

    legend_text = (
        "• AUROC: class imbalance에서 model failure 감지 못함 (GO_0006936 v5-4: AUROC=0.685인데 100% positive 예측)\n"
        "• AUPRC: Precision-Recall 균형을 직접 측정 → positive collapse 즉시 반영\n"
        "• 산포도 원 크기/색: >0.5 예측 비율 (클수록=빨갈수록 positive collapse 심각)\n"
        "• v5-4→v5-5: AUROC는 소폭 변화, AUPRC는 GO_0006096에서 0.297→0.670 (+125%) 대폭 개선"
    )
    fig.text(0.5, -0.06, legend_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', alpha=0.8))
    plt.tight_layout(rect=[0, 0.10, 1, 0.92])
    fig.savefig(os.path.join(OUT_DIR, 'chart10_auroc_vs_auprc.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Chart 10 saved.")


# ─── 실행 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Saving charts to: {OUT_DIR}")
    chart1_macro_auprc()
    chart2_per_goterm_auprc()
    chart3_crf_damage()
    chart4_phase_embedding_v52_v53()
    chart5_v55_phase_quality()
    chart6_positive_collapse()
    chart7_coverage_analysis()
    chart8_labelprop_alpha()
    chart9_full_history_heatmap()
    chart10_auroc_vs_auprc()
    print("\n모든 차트 생성 완료!")
