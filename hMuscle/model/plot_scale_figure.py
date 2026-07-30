#!/usr/bin/env python3
"""Generate S_Scale figure: ESM-2 scale dependence of δ_layer benefit."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

DATA = '../../reports/exp_a_scale/scale_results.json'
OUT_DIR = '../../reports/figures'
os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA) as f:
    results = json.load(f)

scales = ['ESM2_8M', 'ESM2_35M', 'ESM2_150M']
labels = ['8M\n(6 layers)', '35M\n(12 layers)', '150M\n(30 layers)']
prism_vals = [results[s]['prism']['All MF'] for s in scales]
v17f_vals  = [results[s]['v17f']['All MF']  for s in scales]

# L2_Structural
prism_l2 = [results[s]['prism'].get('L2_Structural', np.nan) for s in scales]
v17f_l2  = [results[s]['v17f'].get('L2_Structural', np.nan)  for s in scales]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor('white')

x = np.arange(len(scales))
w = 0.35
c_prism = '#5B6BAF'
c_v17f  = '#E07040'

for ax, prism, v17f, title in [
    (ax1, prism_vals, v17f_vals, 'All MF AUPRC'),
    (ax2, prism_l2,  v17f_l2,   'L2_Structural AUPRC'),
]:
    b1 = ax.bar(x - w/2, prism, w, label='PRISM (frozen)', color=c_prism, alpha=0.85, edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + w/2, v17f,  w, label='v17f (δ_layer)', color=c_v17f, alpha=0.85, edgecolor='white', linewidth=0.5)

    for bar, val in zip(b1, prism):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.3f}',
                ha='center', va='bottom', fontsize=7.5, color=c_prism, fontweight='bold')
    for bar, val, pval in zip(b2, v17f, prism):
        delta = val - pval
        label_text = f'{val:.3f}\n({delta:+.3f})'
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, label_text,
                ha='center', va='bottom', fontsize=7, color=c_v17f, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel('ESM-2 model scale', fontsize=11)
    ax.set_ylabel('Macro AUPRC', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.82)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend(fontsize=9, frameon=False, loc='upper left')

    # Annotate 150M gain
    ax.annotate('p(δ_layer) emerges\nwith 30-layer depth',
                xy=(2 + w/2, v17f[2] + 0.015), fontsize=8,
                ha='center', color=c_v17f, style='italic',
                xytext=(2 + w/2, v17f[2] + 0.08),
                arrowprops=dict(arrowstyle='->', color=c_v17f, lw=1.5))

plt.suptitle('ESM-2 Scale Dependence of δ_layer Benefit\n(Brain zero-shot, 82 MF GO terms)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

for ext in ['pdf', 'png']:
    out = f'{OUT_DIR}/S_Scale_esm2_depth.{ext}'
    plt.savefig(out, bbox_inches='tight', dpi=200)
    print(f'[Saved] {out}')

print('\nSummary:')
for s, lab, p, v in zip(scales, labels, prism_vals, v17f_vals):
    print(f'  {s:12s}: PRISM={p:.4f}  v17f={v:.4f}  Δ={v-p:+.4f}')
