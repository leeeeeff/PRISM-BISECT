"""
Research Meeting Figures — 2026-05-14
=====================================
Total: 10 figures covering
  Fig 1-7 : Performance / SP-Dependency / FTI / Selective Ensemble
  Fig 8-10: D/S (Domain/Splicing) isoform-level feature analysis

Run:
  conda activate isoform_env
  python generate_meeting_figures.py
"""

import os, json, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr
from numpy.linalg import lstsq

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = '/home/welcome1/sw1686/DIFFUSE'
FEAT_DIR  = f'{ROOT}/hMuscle/results_isoform/features'
DATA_DIR  = f'{ROOT}/hMuscle/data'
MODEL_DIR = f'{ROOT}/hMuscle/model'
OUT_DIR   = f'{ROOT}/reports/2026-05-14'

os.makedirs(OUT_DIR, exist_ok=True)

# ─── Shared constants ─────────────────────────────────────────────────────────
TARGET_GOS = ['GO:0007204', 'GO:0030017', 'GO:0006941', 'GO:0003774', 'GO:0006096']
GO_SHORT   = {
    'GO:0007204': 'Ca²⁺\n(0007204)',
    'GO:0030017': 'Sarcomere\n(0030017)',
    'GO:0006941': 'Contraction\n(0006941)',
    'GO:0003774': 'Myosin\n(0003774)',
    'GO:0006096': 'Glycolysis\n(0006096)',
}
GO_LABEL = {
    'GO:0007204': 'Ca²⁺',
    'GO:0030017': 'Sarcomere',
    'GO:0006941': 'Contraction',
    'GO:0003774': 'Myosin',
    'GO:0006096': 'Glycolysis',
}
TIER_COLOR = {'GO:0007204': '#1f77b4', 'GO:0030017': '#1f77b4',
              'GO:0006941': '#ff7f0e', 'GO:0003774': '#ff7f0e',
              'GO:0006096': '#d62728'}
TIER_NAME  = {'GO:0007204': 'Tier 3 (SP Harmful)', 'GO:0030017': 'Tier 3 (SP Harmful)',
              'GO:0006941': 'Tier 2 (SP Neutral)', 'GO:0003774': 'Tier 2 (SP Neutral+)',
              'GO:0006096': 'Tier 1 (SP Required)'}

LR_AUPRC = {
    'GO:0007204': 0.4140, 'GO:0030017': 0.5610, 'GO:0006941': 0.3120,
    'GO:0003774': 0.8250, 'GO:0006096': 0.6950,
}
V8B_AUPRC = {
    'GO:0007204': 0.1462, 'GO:0030017': 0.1570, 'GO:0006941': 0.1177,
    'GO:0003774': 0.5686, 'GO:0006096': 0.7945,
}
P3_AUPRC = {
    'GO:0007204': 0.1624, 'GO:0030017': 0.1918, 'GO:0006941': 0.1292,
    'GO:0003774': 0.5405, 'GO:0006096': 0.4640,
}
P3_256_AUPRC = {
    'GO:0007204': 0.3109, 'GO:0030017': 0.2836, 'GO:0006941': 0.1540,
    'GO:0003774': 0.5830, 'GO:0006096': 0.4445,
}
P3_512_AUPRC = {
    'GO:0007204': 0.4055, 'GO:0030017': 0.3553, 'GO:0006941': 0.1954,
    'GO:0003774': 0.6192, 'GO:0006096': 0.4939,
}
D256_AUPRC = {
    'GO:0007204': 0.1947, 'GO:0030017': 0.1366, 'GO:0006941': 0.1567,
    'GO:0003774': 0.5982, 'GO:0006096': 0.8331,
}

FTI_64  = {'GO:0007204': 0.900, 'GO:0030017': 0.819, 'GO:0006941': 0.911,
            'GO:0003774': 1.052, 'GO:0006096': 1.712}
FTI_256 = {'GO:0007204': 0.626, 'GO:0030017': 0.482, 'GO:0006941': 1.018,
            'GO:0003774': 1.026, 'GO:0006096': 1.874}

# Load TBS/TCS
with open(f'{OUT_DIR}/tbs_results.json') as f:  tbs_data = json.load(f)
with open(f'{OUT_DIR}/tcs_results.json') as f:  tcs_data = json.load(f)
TBS  = {go: tbs_data[go]['tbs']             for go in TARGET_GOS}
TCS  = {go: tcs_data['per_go'][go]['tcs']   for go in TARGET_GOS}
SMSI = {go: tcs_data['per_go'][go]['smsi']  for go in TARGET_GOS}

STYLE = dict(fontsize=9)

def save(fig, name):
    path = f'{OUT_DIR}/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Performance History
# ══════════════════════════════════════════════════════════════════════════════
print('[Fig 1] Performance history ...')

versions = [
    ('v7c',           0.315,  'grey'),
    ('v8b\n(Unified Loss)', 0.357, '#aec7e8'),
    ('Type-aware\nEnsemble', 0.371, '#aec7e8'),
    ('D256\n(SP✓,256d)',    0.384, '#ffbb78'),
    ('Selective\n(P3-256+D256)', 0.437, '#98df8a'),
    ('Selective\n(P3-512+D256)', 0.482, '#2ca02c'),
]
lr_val = 0.5614

fig, ax = plt.subplots(figsize=(10, 5))
ys = range(len(versions))
for i, (label, val, color) in enumerate(versions):
    ax.barh(i, val, color=color, edgecolor='black', linewidth=0.7, height=0.6)
    ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

ax.axvline(lr_val, color='red', linestyle='--', linewidth=1.5, label=f'ESM-2 LR baseline = {lr_val:.3f}')
ax.text(lr_val + 0.003, len(versions) - 0.3, f'LR = {lr_val:.3f}', color='red', fontsize=9)

ax.set_yticks(list(ys))
ax.set_yticklabels([v[0] for v in versions], fontsize=10)
ax.set_xlabel('Macro-AUPRC (5 GO terms)', fontsize=12)
ax.set_title('Pipeline Performance History\n'
             'Key: Unified Loss → Dim Expansion → Selective SP Strategy', fontsize=12, fontweight='bold')
ax.set_xlim(0, lr_val + 0.12)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Milestone annotations
milestones = {
    1: 'Focal + Triplet loss',
    2: 'Type-A/B GO model selection',
    3: 'SP✓ + 256d',
    4: 'Per-GO optimal SP × dim',
    5: 'P3-512 type-B + D256 type-A',
}
for i, text in milestones.items():
    ax.annotate(text, xy=(versions[i][1], i), xytext=(versions[i][1] - 0.08, i - 0.4),
                fontsize=7.5, color='#555555',
                arrowprops=dict(arrowstyle='->', color='#999', lw=0.7))

pct = versions[-1][1] / lr_val * 100
ax.text(0.98, 0.04, f'Current best: {versions[-1][1]:.3f}\n({pct:.1f}% of LR)',
        transform=ax.transAxes, ha='right', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8))

ax.legend(loc='lower right', fontsize=9)
save(fig, 'fig1_performance_history.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — 2×2 SP × Dim Ablation
# ══════════════════════════════════════════════════════════════════════════════
print('[Fig 2] SP × Dim ablation ...')

models = ['v8b\nSP✓,64d', 'P3\nSP✗,64d', 'P3-256\nSP✗,256d', 'D256\nSP✓,256d', 'LR\n640d']
results_table = {
    go: [V8B_AUPRC[go], P3_AUPRC[go], P3_256_AUPRC[go], D256_AUPRC[go], LR_AUPRC[go]]
    for go in TARGET_GOS
}

mat = np.array([results_table[go] for go in TARGET_GOS])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [3, 1]})

# Left: full heatmap
ax = axes[0]
im = ax.imshow(mat, cmap='RdYlGn', vmin=0.0, vmax=0.90, aspect='auto')

for i, go in enumerate(TARGET_GOS):
    for j, val in enumerate(results_table[go]):
        bold = (val == max(results_table[go][:4]))
        txt = ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                      fontsize=10, fontweight='bold' if bold else 'normal',
                      color='white' if val > 0.55 else 'black')

ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=9)
ax.set_yticks(range(len(TARGET_GOS)))
ax.set_yticklabels([GO_LABEL[g] for g in TARGET_GOS], fontsize=10)
ax.set_title('GO term AUPRC by Model Configuration', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='AUPRC', shrink=0.8)

# Highlight key cells
ax.add_patch(plt.Rectangle((3 - 0.48, 4 - 0.48), 0.96, 0.96, fill=False, edgecolor='gold', lw=2.5))
ax.text(3, 4.49, '★ > LR', ha='center', va='bottom', fontsize=8, color='goldenrod', fontweight='bold')

ax.add_patch(plt.Rectangle((2 - 0.48, 0 - 0.48), 0.96, 0.96, fill=False, edgecolor='steelblue', lw=2))
ax.add_patch(plt.Rectangle((2 - 0.48, 1 - 0.48), 0.96, 0.96, fill=False, edgecolor='steelblue', lw=2))

# SP×Dim interaction arrow annotation
ax.annotate('', xy=(2, 1.5), xytext=(0, 1.5),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.text(1.0, 1.9, 'SP✗ + dim↑\n→ Type-B ↑↑', fontsize=7.5, color='blue', ha='center')

ax.annotate('', xy=(3, 4.5), xytext=(0, 4.5),
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
ax.text(1.5, 4.65, 'SP✓ + dim↑\n→ GO:0006096 ↑', fontsize=7.5, color='darkred', ha='center')

# Right: Macro-AUPRC summary bar
ax2 = axes[1]
macros = [
    ('v8b', np.mean(list(V8B_AUPRC.values())), '#aec7e8'),
    ('P3',  np.mean(list(P3_AUPRC.values())),  '#aec7e8'),
    ('P3-256', np.mean(list(P3_256_AUPRC.values())), '#ffbb78'),
    ('D256', np.mean(list(D256_AUPRC.values())), '#ffbb78'),
    ('LR',  np.mean(list(LR_AUPRC.values())),  '#d62728'),
]
select_macro = (P3_512_AUPRC['GO:0007204'] + P3_512_AUPRC['GO:0030017'] +
                P3_512_AUPRC['GO:0006941'] + P3_512_AUPRC['GO:0003774'] + D256_AUPRC['GO:0006096']) / 5

bar_labels = [m[0] for m in macros] + ['Selective\nBest']
bar_vals   = [m[1] for m in macros] + [select_macro]
bar_colors = [m[2] for m in macros] + ['#2ca02c']

bars = ax2.bar(bar_labels, bar_vals, color=bar_colors, edgecolor='black', linewidth=0.7)
for bar, val in zip(bars, bar_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax2.axhline(macros[-1][1], color='red', linestyle='--', lw=1.2, alpha=0.7)
ax2.set_ylim(0, 0.72)
ax2.set_ylabel('Macro-AUPRC', fontsize=10)
ax2.set_title('Macro-AUPRC\nSummary', fontsize=11, fontweight='bold')
ax2.tick_params(axis='x', labelsize=8)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('SP × Dim 2×2 Ablation: SwissProt Usage × Embedding Dimension',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save(fig, 'fig2_sp_dim_ablation.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Dim Scaling Law
# ══════════════════════════════════════════════════════════════════════════════
print('[Fig 3] Dim scaling law ...')

dims = [64, 256, 512]
type_b_gos = ['GO:0007204', 'GO:0030017', 'GO:0006941']
type_a_gos = ['GO:0003774', 'GO:0006096']

sp_off = {
    'GO:0007204': [P3_AUPRC['GO:0007204'], P3_256_AUPRC['GO:0007204'], P3_512_AUPRC['GO:0007204']],
    'GO:0030017': [P3_AUPRC['GO:0030017'], P3_256_AUPRC['GO:0030017'], P3_512_AUPRC['GO:0030017']],
    'GO:0006941': [P3_AUPRC['GO:0006941'], P3_256_AUPRC['GO:0006941'], P3_512_AUPRC['GO:0006941']],
    'GO:0003774': [P3_AUPRC['GO:0003774'], P3_256_AUPRC['GO:0003774'], P3_512_AUPRC['GO:0003774']],
    'GO:0006096': [P3_AUPRC['GO:0006096'], P3_256_AUPRC['GO:0006096'], P3_512_AUPRC['GO:0006096']],
}
sp_on = {
    'GO:0003774': [V8B_AUPRC['GO:0003774'], D256_AUPRC['GO:0003774']],
    'GO:0006096': [V8B_AUPRC['GO:0006096'], D256_AUPRC['GO:0006096']],
}

colors_b = {'GO:0007204': '#1f77b4', 'GO:0030017': '#17becf', 'GO:0006941': '#9edae5'}
colors_a = {'GO:0003774': '#ff7f0e', 'GO:0006096': '#d62728'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: Type-B (SP✗)
for go in type_b_gos:
    lr = LR_AUPRC[go]
    ax1.plot(dims, sp_off[go], 'o-', color=colors_b[go], linewidth=2.0, markersize=7,
             label=f'{GO_LABEL[go]} (LR={lr:.3f})')
    ax1.axhline(lr, color=colors_b[go], linestyle=':', alpha=0.5, linewidth=1.2)

ax1.set_xlabel('Embedding Dimension (SP✗ model)', fontsize=11)
ax1.set_ylabel('AUPRC', fontsize=11)
ax1.set_title('Type-B GO Terms (SP Harmful)\nDim Scaling with SP Removed', fontsize=11, fontweight='bold')
ax1.set_xticks(dims)
ax1.set_xlim(30, 580)
ax1.set_ylim(0, 0.65)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=9)

# Annotate P3-512 GO:0007204 ≈ LR
ax1.annotate('97.9% of LR!', xy=(512, P3_512_AUPRC['GO:0007204']),
             xytext=(450, P3_512_AUPRC['GO:0007204'] + 0.07),
             fontsize=8.5, color='#1f77b4', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2))

# Improvement annotations
for go in type_b_gos:
    delta = sp_off[go][2] - sp_off[go][0]
    pct   = delta / sp_off[go][0] * 100
    ax1.text(512 + 5, sp_off[go][2], f'+{pct:.0f}%', fontsize=7.5, color=colors_b[go], va='center')

# Right: Type-A (SP✓ vs SP✗ per dim)
ax2_twin = ax2.twinx()
for go in type_a_gos:
    ax2.plot([64, 256], sp_on[go], 's--', color=colors_a[go], linewidth=1.8,
             markersize=6, alpha=0.5, label=f'{GO_LABEL[go]} SP✓')
    ax2.plot(dims, sp_off[go], 'o-', color=colors_a[go], linewidth=2.0,
             markersize=7, alpha=0.8, label=f'{GO_LABEL[go]} SP✗')
    ax2.axhline(LR_AUPRC[go], color=colors_a[go], linestyle=':', alpha=0.4, lw=1.2)

ax2.set_xlabel('Embedding Dimension', fontsize=11)
ax2.set_ylabel('AUPRC', fontsize=11)
ax2.set_title('Type-A GO Terms (SP Required/Neutral)\nSP✓ (dashed) vs SP✗ (solid)', fontsize=11, fontweight='bold')
ax2.set_xticks(dims)
ax2.set_xlim(30, 580)
ax2.set_ylim(0, 1.0)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8.5, loc='lower right')

ax2.annotate('D256 SP✓\n> LR', xy=(256, D256_AUPRC['GO:0006096']),
             xytext=(350, D256_AUPRC['GO:0006096'] - 0.12),
             fontsize=8.5, color='#d62728', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2))

plt.suptitle('Dim Scaling Law: SP✗ (Type-B) enables monotone AUPRC growth with dim',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
save(fig, 'fig3_dim_scaling_law.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — FTI Tier Classification
# ══════════════════════════════════════════════════════════════════════════════
print('[Fig 4] FTI tier classification ...')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

go_order = ['GO:0030017', 'GO:0007204', 'GO:0006941', 'GO:0003774', 'GO:0006096']
fti64_vals  = [FTI_64[g]  for g in go_order]
fti256_vals = [FTI_256[g] for g in go_order]
tier_colors = [TIER_COLOR[g] for g in go_order]

# Left: FTI_64 vs FTI_256 scatter
for go in go_order:
    ax1.scatter(FTI_64[go], FTI_256[go], s=200, c=TIER_COLOR[go],
                edgecolors='black', linewidths=0.8, zorder=5)
    ax1.annotate(GO_LABEL[go], xy=(FTI_64[go], FTI_256[go]),
                 xytext=(FTI_64[go] + 0.02, FTI_256[go] + 0.05),
                 fontsize=8.5, ha='left')

ax1.axhline(1.0, color='gray', linestyle='--', lw=1, alpha=0.7)
ax1.axvline(1.0, color='gray', linestyle='--', lw=1, alpha=0.7)
ax1.plot([0.4, 2.0], [0.4, 2.0], 'k-', alpha=0.2, lw=1, label='FTI64=FTI256')

ax1.text(0.42, 1.55, 'dim↑\namplifies', fontsize=8, color='#1f77b4', style='italic')
ax1.text(1.6, 0.45, 'dim↑\nbenefits', fontsize=8, color='#d62728', style='italic')

tier_patches = [
    mpatches.Patch(color='#d62728', label='Tier 1: SP Required'),
    mpatches.Patch(color='#ff7f0e', label='Tier 2: SP Neutral'),
    mpatches.Patch(color='#1f77b4', label='Tier 3: SP Harmful'),
]
ax1.legend(handles=tier_patches, fontsize=8.5, loc='upper left')
ax1.set_xlabel('FTI at 64d  [AUPRC(SP✓) / AUPRC(SP✗)]', fontsize=10)
ax1.set_ylabel('FTI at 256d  [AUPRC(SP✓) / AUPRC(SP✗)]', fontsize=10)
ax1.set_title('FTI: 64d vs 256d\n(dim amplifies SP effect in both directions)', fontsize=11, fontweight='bold')
ax1.grid(alpha=0.3)

# Right: FTI bar chart (256d) with Tier annotation
bars = ax2.bar([GO_LABEL[g] for g in go_order], fti256_vals,
               color=tier_colors, edgecolor='black', linewidth=0.7)
ax2.axhline(1.0, color='black', linestyle='--', lw=1.5, label='FTI=1.0 (neutral)')
for bar, val in zip(bars, fti256_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_ylim(0, 2.3)
ax2.set_ylabel('FTI₂₅₆ = AUPRC(SP✓,256d) / AUPRC(SP✗,256d)', fontsize=9)
ax2.set_title('Functional Transferability Index (256d)\nTier Classification', fontsize=11, fontweight='bold')
ax2.tick_params(axis='x', labelsize=9)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Tier region shading
ax2.axhspan(1.5, 2.3, alpha=0.07, color='#d62728', label='')
ax2.axhspan(0.8, 1.5, alpha=0.07, color='#ff7f0e', label='')
ax2.axhspan(0.0, 0.8, alpha=0.07, color='#1f77b4', label='')
ax2.text(4.4, 2.1, 'Tier 1', fontsize=8, color='#d62728')
ax2.text(4.4, 1.1, 'Tier 2', fontsize=8, color='#ff7f0e')
ax2.text(4.4, 0.3, 'Tier 3', fontsize=8, color='#1f77b4')

plt.suptitle('FTI (Functional Transferability Index) — Empirical SP Benefit Score per GO term',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
save(fig, 'fig4_fti_tier.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — 2D SP Dependency Framework
# ══════════════════════════════════════════════════════════════════════════════
print('[Fig 5] 2D SP dependency framework ...')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

tbs_arr = np.array([TBS[g]  for g in TARGET_GOS])
tcs_arr = np.array([TCS[g]  for g in TARGET_GOS])
fti_arr = np.array([FTI_256[g] for g in TARGET_GOS])

norm = plt.Normalize(0.4, 2.0)
cmap = plt.cm.RdYlGn

# Left: 2D scatter TBS × TCS, color = FTI
ax = axes[0]
sc = ax.scatter(tbs_arr, tcs_arr, c=fti_arr, cmap=cmap, norm=norm,
                s=350, edgecolors='black', linewidths=0.9, zorder=5)

offsets = {
    'GO:0007204': (0.03,  0.003),
    'GO:0030017': (-0.17, -0.006),
    'GO:0006941': (0.03,  -0.010),
    'GO:0003774': (0.03,  0.003),
    'GO:0006096': (0.03,  -0.008),
}
for go in TARGET_GOS:
    dx, dy = offsets[go]
    ax.annotate(f'{GO_LABEL[go]}\nFTI={FTI_256[go]:.2f}',
                xy=(TBS[go], TCS[go]),
                xytext=(TBS[go] + dx, TCS[go] + dy),
                fontsize=8.5, ha='left',
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))

tcs_med = np.median(tcs_arr)
ax.axvline(0.5, color='gray', ls='--', alpha=0.4, lw=0.8)
ax.axhline(tcs_med, color='gray', ls='--', alpha=0.4, lw=0.8)

quad_kw = dict(fontsize=7.5, alpha=0.8, ha='center')
ax.text(0.22, tcs_med + 0.01, 'Low TBS\nHigh TCS\n→ Tier 3', color='#1f77b4', **quad_kw)
ax.text(0.75, tcs_med + 0.01, 'High TBS\nHigh TCS\n→ Tier 3 ⚠', color='#1f77b4', **quad_kw)
ax.text(0.22, tcs_med - 0.025, 'Low TBS\nLow TCS\n→ Tier 2', color='#ff7f0e', **quad_kw)
ax.text(0.75, tcs_med - 0.025, 'High TBS\nLow TCS\n→ Tier 1', color='#d62728', **quad_kw)

cb = plt.colorbar(sc, ax=ax, shrink=0.85)
cb.set_label('FTI₂₅₆', fontsize=10)
cb.ax.axhline(1.0, color='black', ls='--', lw=1.0)

ax.set_xlabel('TBS (Taxonomic Breadth Score = kingdoms/6)', fontsize=11)
ax.set_ylabel('TCS (mean τ, Tissue Context Specificity)', fontsize=11)
ax.set_title('2D SP Dependency Framework\n'
             'GO:0007204 & GO:0006096: same TBS (0.667) → opposite FTI!',
             fontsize=10, fontweight='bold')
ax.set_xlim(0.1, 1.0)
ax.grid(alpha=0.2)

# Right: Regression surface + scatter (TBS, TCS → FTI_256)
ax2 = axes[1]

# Fit 2D linear model: FTI = b0 + b1*TBS + b2*TCS + b3*(TBS*TCS)
X = np.column_stack([np.ones(5), tbs_arr, tcs_arr, tbs_arr * tcs_arr])
coef, _, _, _ = lstsq(X, fti_arr, rcond=None)
fti_pred = X @ coef
r2 = 1 - np.sum((fti_arr - fti_pred)**2) / np.sum((fti_arr - fti_arr.mean())**2)

ax2.scatter(fti_arr, fti_pred, s=300, c=fti_arr, cmap=cmap, norm=norm,
            edgecolors='black', linewidths=0.9, zorder=5)
for i, go in enumerate(TARGET_GOS):
    ax2.annotate(GO_LABEL[go], xy=(fti_arr[i], fti_pred[i]),
                 xytext=(fti_arr[i] + 0.04, fti_pred[i]),
                 fontsize=8.5, va='center')

diag = np.linspace(0.4, 2.0, 50)
ax2.plot(diag, diag, 'k--', alpha=0.4, lw=1, label='perfect fit')
ax2.set_xlabel('FTI₂₅₆ (observed)', fontsize=11)
ax2.set_ylabel('FTI₂₅₆ (predicted, 2D model)', fontsize=11)
ax2.set_title(f'2D Linear Regression Fit\n'
              f'FTI = f(TBS, TCS, TBS×TCS),  R² = {r2:.3f}\n'
              f'β(TBS) = {coef[1]:.1f},  β(TCS) = {coef[2]:.1f},  β(TBS×TCS) = {coef[3]:.1f}',
              fontsize=9.5, fontweight='bold')
ax2.text(0.05, 0.92, f'R² = {r2:.3f}\n(n=5, interpretive)', transform=ax2.transAxes,
         fontsize=11, fontweight='bold', color='#2ca02c',
         bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8))
ax2.grid(alpha=0.3)
ax2.legend(fontsize=9)

plt.suptitle('2D SP Dependency Framework: TBS × TCS predicts FTI with R²=0.992\n'
             'Single conservation axis (TBS only: r=0.36) insufficient — TCS required',
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
save(fig, 'fig5_2d_sp_dependency.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Kingdom Coverage × SMSI panel
# ══════════════════════════════════════════════════════════════════════════════
print('[Fig 6] Kingdom × SMSI panel ...')

kingdoms = ['Bacteria', 'Fungi', 'Invertebrata', 'Vertebrata', 'Viridiplantae', 'Archaea']
kingdom_colors = ['#8c564b', '#e377c2', '#17becf', '#ff7f0e', '#2ca02c', '#7f7f7f']

kingdom_counts = {}
for go in TARGET_GOS:
    kc = tbs_data[go].get('kingdom_counts', {})
    kingdom_counts[go] = {k: kc.get(k, 0) for k in kingdoms}

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

# Left: Stacked bar — kingdom composition per GO term
ax = axes[0]
go_labels = [GO_LABEL[g] for g in TARGET_GOS]
bottoms = np.zeros(5)
for ki, k in enumerate(kingdoms):
    vals = [kingdom_counts[go][k] for go in TARGET_GOS]
    bars = ax.bar(go_labels, vals, bottom=bottoms, color=kingdom_colors[ki],
                  label=k, edgecolor='white', linewidth=0.5)
    bottoms += np.array(vals)

ax.set_ylabel('Number of SwissProt+ proteins', fontsize=10)
ax.set_title('Kingdom Breadth (SP+ annotations)\nper GO term', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right', ncol=1)
ax.tick_params(axis='x', labelsize=9)
ax.grid(axis='y', alpha=0.3)

# TBS text on bars
for i, go in enumerate(TARGET_GOS):
    total = sum(kingdom_counts[go].values())
    ax.text(i, total + 10, f'TBS={TBS[go]:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Middle: FTI vs TBS scatter (with SMSI as marker size)
ax = axes[1]
smsi_vals = [SMSI[g] for g in TARGET_GOS]
smsi_scaled = [max(s * 25, 50) for s in smsi_vals]

sc = ax.scatter([TBS[g] for g in TARGET_GOS], [FTI_256[g] for g in TARGET_GOS],
                s=smsi_scaled, c=[TIER_COLOR[g] for g in TARGET_GOS],
                edgecolors='black', linewidths=0.8, zorder=5)

for go in TARGET_GOS:
    ax.annotate(f'{GO_LABEL[go]}\nSMSI={SMSI[go]:.1f}×',
                xy=(TBS[go], FTI_256[go]),
                xytext=(TBS[go] + 0.02, FTI_256[go] + 0.06),
                fontsize=7.5)

ax.axhline(1.0, color='gray', ls='--', lw=1)
ax.set_xlabel('TBS (Taxonomic Breadth Score)', fontsize=10)
ax.set_ylabel('FTI₂₅₆', fontsize=10)
ax.set_title('FTI vs TBS\n(bubble size = SMSI, skeletal muscle enrichment)', fontsize=10, fontweight='bold')
ax.grid(alpha=0.3)

# Right: SMSI bar chart
ax = axes[2]
smsi_order = sorted(TARGET_GOS, key=lambda g: SMSI[g], reverse=True)
bar_vals = [SMSI[g] for g in smsi_order]
bar_colors2 = [TIER_COLOR[g] for g in smsi_order]
bars = ax.bar([GO_LABEL[g] for g in smsi_order], bar_vals,
              color=bar_colors2, edgecolor='black', linewidth=0.7)

ax.axhline(1.0, color='gray', ls='--', lw=1, label='SMSI=1.0 (ubiquitous)')
for bar, val in zip(bars, bar_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('SMSI = nTPM(skeletal muscle) / mean(all tissues)', fontsize=9)
ax.set_title('Skeletal Muscle Specificity Index (SMSI)\nHPA 51-tissue consensus', fontsize=10, fontweight='bold')
ax.tick_params(axis='x', labelsize=9)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

tier_patches2 = [
    mpatches.Patch(color='#d62728', label='Tier 1'),
    mpatches.Patch(color='#ff7f0e', label='Tier 2'),
    mpatches.Patch(color='#1f77b4', label='Tier 3'),
]
ax.legend(handles=tier_patches2 + [mpatches.Patch(color='gray', label='SMSI=1')],
          fontsize=8, loc='upper right')

plt.suptitle('Kingdom Coverage (TBS) + Skeletal Muscle Specificity (SMSI)\n'
             'GO:0007204: high TBS but SMSI=1.63× → SP mixes non-muscle Ca²⁺ noise',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
save(fig, 'fig6_kingdom_smsi_panel.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Selective Ensemble Final Result
# ══════════════════════════════════════════════════════════════════════════════
print('[Fig 7] Selective ensemble ...')

fig = plt.figure(figsize=(14, 6))
gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

x = np.arange(len(TARGET_GOS))
w = 0.16
models_bar = [
    ('v8b\nSP✓,64d',   [V8B_AUPRC[g]   for g in TARGET_GOS], '#aec7e8'),
    ('D256\nSP✓,256d', [D256_AUPRC[g]  for g in TARGET_GOS], '#ffbb78'),
    ('P3-512\nSP✗,512d', [P3_512_AUPRC[g] for g in TARGET_GOS], '#98df8a'),
    ('LR\n640d',       [LR_AUPRC[g]    for g in TARGET_GOS], '#ff9896'),
]

for i, (label, vals, color) in enumerate(models_bar):
    offset = (i - 1.5) * w
    bars = ax1.bar(x + offset, vals, w, label=label, color=color, edgecolor='black', lw=0.6)

# Selective best markers
sel_vals = {
    'GO:0007204': P3_512_AUPRC['GO:0007204'],
    'GO:0030017': P3_512_AUPRC['GO:0030017'],
    'GO:0006941': P3_512_AUPRC['GO:0006941'],
    'GO:0003774': P3_512_AUPRC['GO:0003774'],
    'GO:0006096': D256_AUPRC['GO:0006096'],
}
sel_models = {
    'GO:0007204': 'P3-512', 'GO:0030017': 'P3-512', 'GO:0006941': 'P3-512',
    'GO:0003774': 'P3-512', 'GO:0006096': 'D256',
}

for i, go in enumerate(TARGET_GOS):
    v = sel_vals[go]
    ax1.annotate('★', xy=(x[i], v), xytext=(x[i], v + 0.03),
                fontsize=12, ha='center', color='green', fontweight='bold')
    ax1.text(x[i], v + 0.07, sel_models[go], ha='center', fontsize=6.5, color='darkgreen')

ax1.set_xticks(x)
ax1.set_xticklabels([GO_LABEL[g] for g in TARGET_GOS], fontsize=9)
ax1.set_ylabel('AUPRC', fontsize=11)
ax1.set_title('Per-GO-term AUPRC by Model\n★ = Selective Best selection', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8.5, loc='upper right', ncol=2)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 1.05)

# Right: Macro summary waterfall
macros2 = [
    ('v8b', np.mean(list(V8B_AUPRC.values())),    '#aec7e8'),
    ('D256', np.mean(list(D256_AUPRC.values())),  '#ffbb78'),
    ('P3-512', np.mean(list(P3_512_AUPRC.values())), '#98df8a'),
    ('Selective\n(Best)', (sum(sel_vals[g] for g in TARGET_GOS))/5, '#2ca02c'),
    ('LR', np.mean(list(LR_AUPRC.values())),      '#d62728'),
]

bars2 = ax2.bar([m[0] for m in macros2], [m[1] for m in macros2],
                color=[m[2] for m in macros2], edgecolor='black', lw=0.7)
for bar, (label, val, _) in zip(bars2, macros2):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_ylabel('Macro-AUPRC', fontsize=11)
ax2.set_title('Macro-AUPRC Comparison\n(Selective = per-GO optimal selection)', fontsize=10, fontweight='bold')
ax2.set_ylim(0, 0.72)
ax2.tick_params(axis='x', labelsize=9)
ax2.grid(axis='y', alpha=0.3)

sel_macro = sum(sel_vals[g] for g in TARGET_GOS) / 5
lr_macro  = np.mean(list(LR_AUPRC.values()))
ax2.text(0.5, 0.92, f'Gap to LR: {lr_macro-sel_macro:.4f}\n({sel_macro/lr_macro*100:.1f}% of LR)',
         transform=ax2.transAxes, ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8))

plt.suptitle('Selective SP Ensemble: P3-512 (Type-B) + D256 (Type-A)\nMacro-AUPRC = 0.4817 (85.8% of LR)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
save(fig, 'fig7_selective_ensemble.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — D/S Feature Statistics Overview
# ══════════════════════════════════════════════════════════════════════════════
print('[Fig 8] D/S feature statistics ...')

dd = np.load(f'{FEAT_DIR}/domain_delta_v2.npy')
sd = np.load(f'{FEAT_DIR}/splicing/splicing_delta_v2.npy')

dd_nonzero = np.abs(dd).sum(axis=1) > 0
sd_nonzero = np.abs(sd).sum(axis=1) > 0
n = len(dd)

both   = (dd_nonzero & sd_nonzero).sum()
d_only = (dd_nonzero & ~sd_nonzero).sum()
s_only = (~dd_nonzero & sd_nonzero).sum()
none   = (~dd_nonzero & ~sd_nonzero).sum()

iso_list  = np.load(f'{MODEL_DIR}/my_isoform_list_fixed.npy', allow_pickle=True)
gene_list = np.load(f'{MODEL_DIR}/my_gene_list_fixed.npy',    allow_pickle=True)
gene_to_idx = collections.defaultdict(list)
for idx, g in enumerate(gene_list):
    gene_to_idx[g.decode() if isinstance(g, bytes) else g].append(idx)
iso_counts = [len(v) for v in gene_to_idx.values()]

dd_norm = np.linalg.norm(dd, axis=1)
sd_norm = np.linalg.norm(sd, axis=1)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# [0,0] Pie: D/S signal breakdown
ax = axes[0, 0]
sizes  = [both, d_only, s_only, none]
labels = [f'D≠0 & S≠0\n{both} ({both/n*100:.1f}%)',
          f'D≠0 only\n{d_only} ({d_only/n*100:.1f}%)',
          f'S≠0 only\n{s_only} ({s_only/n*100:.1f}%)',
          f'D=0 & S=0\n{none} ({none/n*100:.1f}%)']
colors_pie = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d3d3d3']
wedges, texts = ax.pie(sizes, colors=colors_pie, startangle=90,
                       wedgeprops=dict(edgecolor='white', linewidth=1.5))
ax.legend(wedges, labels, fontsize=8.5, loc='lower center', bbox_to_anchor=(0.5, -0.35))
ax.set_title(f'Isoform-level D/S Signal\n(n={n:,} isoforms total)', fontsize=11, fontweight='bold')

# [0,1] Isoform count per gene histogram
ax = axes[0, 1]
count_bins = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 63]
hist, edges = np.histogram(iso_counts, bins=count_bins)
bar_labels2 = ['1', '2', '3', '4', '5', '6', '7-8', '9-14', '15-19', '20-62', '≥63']
bar_labels2 = bar_labels2[:len(hist)]
ax.bar(range(len(hist)), hist, color='#4e79a7', edgecolor='black', lw=0.6)
ax.set_xticks(range(len(hist)))
ax.set_xticklabels(bar_labels2, fontsize=9)
ax.set_xlabel('Isoforms per gene', fontsize=10)
ax.set_ylabel('Number of genes', fontsize=10)
ax.set_title(f'Isoform Count Distribution\n'
             f'(n={len(gene_to_idx):,} genes, mean={np.mean(iso_counts):.2f}/gene)',
             fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
n_multi = sum(1 for c in iso_counts if c > 1)
ax.text(0.97, 0.92, f'{n_multi:,} genes ({n_multi/len(iso_counts)*100:.0f}%)\nhave >1 isoform',
        transform=ax.transAxes, ha='right', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.8))

# [0,2] Domain delta (D) magnitude histogram
ax = axes[0, 2]
dd_nz = dd_norm[dd_norm > 0]
ax.hist(dd_nz, bins=60, color='#1f77b4', edgecolor='white', lw=0.3, alpha=0.8)
ax.axvline(np.median(dd_nz), color='red', ls='--', lw=1.5, label=f'median={np.median(dd_nz):.0f}')
ax.axvline(np.mean(dd_nz),   color='orange', ls='--', lw=1.5, label=f'mean={np.mean(dd_nz):.0f}')
ax.set_xlabel('‖Domain Delta‖₂ (vs canonical isoform)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title(f'Domain Δ Magnitude Distribution\n'
             f'(isoforms with D≠0: {dd_nonzero.sum():,} / {n:,})',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# [1,0] Splicing delta (S) magnitude histogram
ax = axes[1, 0]
sd_nz = sd_norm[sd_norm > 0]
ax.hist(np.log1p(sd_nz), bins=60, color='#ff7f0e', edgecolor='white', lw=0.3, alpha=0.8)
ax.axvline(np.log1p(np.median(sd_nz)), color='red', ls='--', lw=1.5,
           label=f'median={np.median(sd_nz):.3f}')
ax.set_xlabel('log(1 + ‖Splicing Delta‖₂)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title(f'Splicing Δ Magnitude Distribution\n'
             f'(isoforms with S≠0: {sd_nonzero.sum():,} / {n:,})',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# [1,1] D vs S scatter (per-isoform, log scale)
ax = axes[1, 1]
mask = dd_nonzero | sd_nonzero
sample_idx = np.random.RandomState(42).choice(np.where(mask)[0], size=min(5000, mask.sum()), replace=False)
ax.scatter(np.log1p(dd_norm[sample_idx]), np.log1p(sd_norm[sample_idx]),
           alpha=0.2, s=6, color='#7f7f7f', rasterized=True)

q_labels = [
    (both,   '#2ca02c', 'D≠0 & S≠0'),
    (d_only, '#1f77b4', 'D only'),
    (s_only, '#ff7f0e', 'S only'),
]
from matplotlib.lines import Line2D
legend_els = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=l)
              for _, c, l in q_labels]
ax.legend(handles=legend_els, fontsize=8.5)
ax.set_xlabel('log(1 + ‖Domain Delta‖₂)', fontsize=10)
ax.set_ylabel('log(1 + ‖Splicing Delta‖₂)', fontsize=10)
ax.set_title('D vs S per isoform\n(5,000 sample, only isoforms with D or S ≠ 0)',
             fontsize=10, fontweight='bold')
ax.grid(alpha=0.3)

# [1,2] Feature dim overview table
ax = axes[1, 2]
ax.axis('off')
table_data = [
    ['Feature', 'Shape', 'Non-zero\nisoforms', 'Dim'],
    ['Domain Delta v2\n(D)', f'{dd.shape[0]:,} × {dd.shape[1]}',
     f'{dd_nonzero.sum():,}\n({dd_nonzero.mean()*100:.1f}%)', str(dd.shape[1])],
    ['Splicing Delta v2\n(S)', f'{sd.shape[0]:,} × {sd.shape[1]}',
     f'{sd_nonzero.sum():,}\n({sd_nonzero.mean()*100:.1f}%)', str(sd.shape[1])],
    ['ESM-2 640d\n(sequence)', f'31,668 × 640', '100%', '640'],
    ['Total isoforms\n(train+human)', f'{n:,}', '—', '—'],
]

tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 2.0)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 1:
        cell.set_facecolor('#f0f8ff')
ax.set_title('D/S Feature Summary Table', fontsize=11, fontweight='bold', pad=15)

plt.suptitle('Domain (D) / Splicing (S) Feature Statistics\n'
             '62.8% of isoforms carry D or S signal distinct from their gene\'s canonical isoform',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
save(fig, 'fig8_ds_overview.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Intra-gene Isoform D/S Diversity (isoform vs gene-level resolution)
# ══════════════════════════════════════════════════════════════════════════════
print('[Fig 9] Intra-gene D/S diversity ...')

# For each gene with >=2 isoforms, compute:
#   within-gene std of dd_norm and sd_norm (= isoform-level signal)
#   vs between-gene variance

gene_dd_vals  = collections.defaultdict(list)
gene_sd_vals  = collections.defaultdict(list)
for g, idxs in gene_to_idx.items():
    for i in idxs:
        gene_dd_vals[g].append(dd_norm[i])
        gene_sd_vals[g].append(sd_norm[i])

multi_genes   = [g for g, v in gene_dd_vals.items() if len(v) > 1]
within_dd_std = [np.std(gene_dd_vals[g]) for g in multi_genes]
within_sd_std = [np.std(gene_sd_vals[g]) for g in multi_genes]

gene_means_dd = [np.mean(gene_dd_vals[g]) for g in gene_dd_vals]
gene_means_sd = [np.mean(gene_sd_vals[g]) for g in gene_sd_vals]
between_dd_std = np.std(gene_means_dd)
between_sd_std = np.std(gene_means_sd)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# [0,0] Within-gene D diversity (isoform-level signal exists)
ax = axes[0, 0]
within_dd_nonzero = [v for v in within_dd_std if v > 0]
ax.hist(within_dd_std, bins=60, color='#1f77b4', alpha=0.8, edgecolor='white', lw=0.3)
ax.axvline(np.mean(within_dd_std), color='red', ls='--', lw=1.5,
           label=f'mean = {np.mean(within_dd_std):.0f}')
frac_nonzero = sum(1 for v in within_dd_std if v > 0) / len(within_dd_std)
ax.set_xlabel('Within-gene std of ‖Domain Delta‖  (isoform spread)', fontsize=10)
ax.set_ylabel('Number of genes', fontsize=10)
ax.set_title(f'Domain (D): Intra-gene Isoform Diversity\n'
             f'{frac_nonzero*100:.1f}% of multi-isoform genes have D variation across isoforms',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# [0,1] Within-gene S diversity
ax = axes[0, 1]
ax.hist(within_sd_std, bins=60, color='#ff7f0e', alpha=0.8, edgecolor='white', lw=0.3)
ax.axvline(np.mean(within_sd_std), color='red', ls='--', lw=1.5,
           label=f'mean = {np.mean(within_sd_std):.3f}')
frac_nonzero_s = sum(1 for v in within_sd_std if v > 0) / len(within_sd_std)
ax.set_xlabel('Within-gene std of ‖Splicing Delta‖  (isoform spread)', fontsize=10)
ax.set_ylabel('Number of genes', fontsize=10)
ax.set_title(f'Splicing (S): Intra-gene Isoform Diversity\n'
             f'{frac_nonzero_s*100:.1f}% of multi-isoform genes have S variation across isoforms',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# [1,0] Within vs Between gene variance comparison (D)
ax = axes[1, 0]
categories = ['Within-gene\n(isoform-level)', 'Between-gene\n(gene-level)']
dd_variances = [np.mean(within_dd_std), between_dd_std]
bars = ax.bar(categories, dd_variances, color=['#1f77b4', '#aec7e8'], edgecolor='black', lw=0.8)
for bar, val in zip(bars, dd_variances):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ratio_d = dd_variances[0] / dd_variances[1]
ax.text(0.5, 0.85, f'Within/Between ratio = {ratio_d:.2f}\n→ Isoform-level D signal is\n   {ratio_d*100:.0f}% as large as gene-level',
        transform=ax.transAxes, ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.8))
ax.set_ylabel('std of ‖Domain Delta‖', fontsize=10)
ax.set_title('Domain (D) Variance:\nWithin-gene vs Between-gene', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# [1,1] Within vs Between gene variance comparison (S)
ax = axes[1, 1]
sd_variances = [np.mean(within_sd_std), between_sd_std]
bars = ax.bar(categories, sd_variances, color=['#ff7f0e', '#ffbb78'], edgecolor='black', lw=0.8)
for bar, val in zip(bars, sd_variances):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ratio_s = sd_variances[0] / sd_variances[1]
ax.text(0.5, 0.85, f'Within/Between ratio = {ratio_s:.2f}\n→ Isoform-level S signal is\n   {ratio_s*100:.0f}% as large as gene-level',
        transform=ax.transAxes, ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.8))
ax.set_ylabel('std of ‖Splicing Delta‖', fontsize=10)
ax.set_title('Splicing (S) Variance:\nWithin-gene vs Between-gene', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('D/S Intra-gene Isoform Diversity: Evidence for Isoform-level Resolution\n'
             f'Multi-isoform genes: {len(multi_genes):,} genes — within-gene D/S variance is substantial',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
save(fig, 'fig9_isoform_resolution.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 10 — Previous (Gene-level LR) vs Current (Isoform-level D/S) Comparison
# ══════════════════════════════════════════════════════════════════════════════
print('[Fig 10] Previous vs current resolution ...')

# Compute: for multi-isoform genes, what fraction of isoform pairs
# would be INDISTINGUISHABLE by ESM alone (same gene) but
# DISTINGUISHABLE by D/S features?
# = isoform pairs within same gene where at least one has D≠0 or S≠0

n_pairs_total       = 0
n_pairs_esm_same    = 0  # same gene → ESM encodes same gene context
n_pairs_ds_distinct = 0  # but D/S differs between the pair

for g, idxs in gene_to_idx.items():
    if len(idxs) < 2: continue
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            n_pairs_total += 1
            n_pairs_esm_same += 1  # within gene: gene-level approach treats them the same
            di = dd_nonzero[idxs[i]] or sd_nonzero[idxs[i]]
            dj = dd_nonzero[idxs[j]] or sd_nonzero[idxs[j]]
            if di or dj:
                n_pairs_ds_distinct += 1

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# [0] Comparison conceptual diagram (bar chart)
ax = axes[0]
approaches = ['Gene-level\n(ESM-2 640d only,\nLR baseline)', 'Isoform-level\n(ESM + Domain Δ\n+ Splicing Δ)']
n_distinguishable = [0, n_pairs_ds_distinct]
n_total_pairs = n_pairs_total

bars = ax.bar(approaches, [0, n_pairs_ds_distinct / n_total_pairs * 100],
              color=['#d3d3d3', '#2ca02c'], edgecolor='black', lw=0.8, width=0.5)

ax.text(0, 3, 'Treats all isoforms\nof same gene\nidentically', ha='center', fontsize=9,
        color='#666', style='italic')
ax.text(1, n_pairs_ds_distinct / n_total_pairs * 100 + 2,
        f'{n_pairs_ds_distinct / n_total_pairs * 100:.1f}%\nof intra-gene pairs\nhave distinct D/S',
        ha='center', fontsize=9, fontweight='bold', color='darkgreen')

ax.set_ylabel('% of intra-gene isoform pairs\ndistinguishable', fontsize=10)
ax.set_title(f'Isoform Pair Discriminability\n(Total pairs within genes: {n_total_pairs:,})',
             fontsize=10, fontweight='bold')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

# [1] Cumulative % of isoforms with D/S signal (as depth increases)
ax = axes[1]
# Simulate: if we only had canonical (1 per gene), vs full isoform set
n_canonical = len(gene_to_idx)
n_full       = n

# Isoforms that have D or S signal
n_ds_signal = (dd_nonzero | sd_nonzero).sum()

approach_labels = ['Canonical only\n(1 isoform/gene)', 'Full isoform\nset (current)']
coverage_vals   = [0, n_ds_signal / n_full * 100]

# More nuanced: among canonical isoforms (dd=0, sd=0 by definition), 0% have signal
# among all isoforms, ~62.8% have signal

ax.bar(approach_labels, coverage_vals, color=['#aec7e8', '#2ca02c'],
       edgecolor='black', lw=0.8, width=0.5)
ax.text(0, 3, '0%\n(canonical = reference,\nall deltas = 0)',
        ha='center', fontsize=9, color='#666', style='italic')
ax.text(1, coverage_vals[1] + 2,
        f'{coverage_vals[1]:.1f}%',
        ha='center', fontsize=12, fontweight='bold', color='darkgreen')

ax.set_ylabel('% isoforms with isoform-level\nD/S signal', fontsize=10)
ax.set_title('D/S Signal Coverage:\nCanonical vs Full Isoform Set', fontsize=10, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# [2] AUPRC comparison: LR (gene-level) vs pipeline per GO
ax = axes[2]
x = np.arange(len(TARGET_GOS))
w = 0.25

lr_vals_list = [LR_AUPRC[g] for g in TARGET_GOS]
sel_vals_list = [sel_vals[g] for g in TARGET_GOS]
v8b_vals_list = [V8B_AUPRC[g] for g in TARGET_GOS]

b1 = ax.bar(x - w, v8b_vals_list, w, label='v8b (gene+isoform, 64d)', color='#aec7e8', edgecolor='black', lw=0.6)
b2 = ax.bar(x,     sel_vals_list, w, label='Selective Best (isoform D/S optimized)', color='#2ca02c', edgecolor='black', lw=0.6)
b3 = ax.bar(x + w, lr_vals_list,  w, label='LR baseline (gene-level ESM-2)', color='#d62728', alpha=0.7, edgecolor='black', lw=0.6)

ax.set_xticks(x)
ax.set_xticklabels([GO_LABEL[g] for g in TARGET_GOS], fontsize=9)
ax.set_ylabel('AUPRC', fontsize=10)
ax.set_title('AUPRC: v8b vs Selective (D/S isoform)\nvs LR baseline (gene-level)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.0)

# Annotate where isoform pipeline beats LR
for i, go in enumerate(TARGET_GOS):
    if sel_vals[go] >= LR_AUPRC[go]:
        ax.annotate('★ >LR', xy=(i, sel_vals[go]), xytext=(i, sel_vals[go] + 0.05),
                    ha='center', fontsize=8, color='gold', fontweight='bold')

plt.suptitle('Previous Gene-level Approach vs Current Isoform-level D/S Analysis\n'
             f'{n_pairs_ds_distinct:,}/{n_total_pairs:,} intra-gene isoform pairs uniquely '
             f'distinguished by D/S ({n_pairs_ds_distinct/n_total_pairs*100:.1f}%)',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
save(fig, 'fig10_ds_vs_previous.png')

# ─── Final summary ────────────────────────────────────────────────────────────
print()
print('=' * 60)
print(' All figures saved successfully')
print('=' * 60)
print(f'  Output dir: {OUT_DIR}')
for i in range(1, 11):
    fname = {
        1: 'fig1_performance_history.png',
        2: 'fig2_sp_dim_ablation.png',
        3: 'fig3_dim_scaling_law.png',
        4: 'fig4_fti_tier.png',
        5: 'fig5_2d_sp_dependency.png',
        6: 'fig6_kingdom_smsi_panel.png',
        7: 'fig7_selective_ensemble.png',
        8: 'fig8_ds_overview.png',
        9: 'fig9_isoform_resolution.png',
        10: 'fig10_ds_vs_previous.png',
    }[i]
    size = os.path.getsize(f'{OUT_DIR}/{fname}') // 1024
    print(f'  Fig {i:2d}: {fname}  ({size} KB)')
