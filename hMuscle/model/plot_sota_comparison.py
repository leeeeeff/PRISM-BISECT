#!/usr/bin/env python3
"""Generate S_SOTA figure: method comparison bar chart (brain MF AUPRC)."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RESULTS_FILE = '../../reports/exp_e_sota/sota_results.json'
OUT_DIR = '../../reports/figures'
os.makedirs(OUT_DIR, exist_ok=True)

with open(RESULTS_FILE) as f:
    results = json.load(f)

# Display order and labels
METHODS = [
    ('E0_kNN_L30',     'ESM-2\nk-NN'),
    ('E0b_GeneMean',   'ESM-2\nGene-mean'),
    ('E1_BLAST_Sprot', 'BLAST+\nSwissProt'),
    ('Domain_LR',      'Domain\nLR'),
    ('PRISM',          'PRISM\n(frozen)'),
    ('v17f',           'v17f\n(δ_layer)'),
]

keys    = [m[0] for m in METHODS]
labels  = [m[1] for m in METHODS]
all_mf  = [results.get(k, {}).get('All MF') or float('nan') for k in keys]
l2_str  = [results.get(k, {}).get('L2_Structural') or float('nan') for k in keys]

x = np.arange(len(keys))
w = 0.35

c_all  = '#5B6BAF'
c_l2   = '#E07040'
c_hl   = '#2A7A2A'  # green highlight for v17f

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('white')

bars_all = ax.bar(x - w/2, all_mf, w, label='All MF (82 terms)', color=c_all, alpha=0.85,
                  edgecolor='white', linewidth=0.5)
bars_l2  = ax.bar(x + w/2, l2_str, w, label='L2_Structural (33 terms)', color=c_l2, alpha=0.85,
                  edgecolor='white', linewidth=0.5)

# Highlight v17f bars
last_idx = len(keys) - 1
bars_all[last_idx].set_edgecolor(c_hl)
bars_all[last_idx].set_linewidth(2.5)
bars_l2[last_idx].set_edgecolor(c_hl)
bars_l2[last_idx].set_linewidth(2.5)

for bars, vals, color in [(bars_all, all_mf, c_all), (bars_l2, l2_str, c_l2)]:
    for bar, val in zip(bars, vals):
        if np.isnan(val): continue
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.3f}',
                ha='center', va='bottom', fontsize=7.5, color=color, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Macro AUPRC', fontsize=12)
ax.set_title('PRISM+v17f vs. SOTA baselines\n(Brain zero-shot, 82 MF GO terms)',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 0.85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=9)
ax.axhline(0, color='black', linewidth=0.5)
ax.legend(fontsize=10, frameon=False, loc='upper left')

# PRISM reference line
prism_val = results.get('PRISM', {}).get('All MF', 0.596)
ax.axhline(prism_val, color=c_all, linewidth=1, linestyle='--', alpha=0.4)

# v17f annotation
v17f_all = results.get('v17f', {}).get('All MF', 0.717)
ax.annotate(f'v17f: {v17f_all:.3f}',
            xy=(last_idx - w/2, v17f_all + 0.01), fontsize=9,
            ha='center', color=c_hl, fontweight='bold',
            xytext=(last_idx - w/2, v17f_all + 0.07),
            arrowprops=dict(arrowstyle='->', color=c_hl, lw=1.5))

plt.tight_layout()

for ext in ['pdf', 'png']:
    out = f'{OUT_DIR}/S_SOTA_comparison.{ext}'
    plt.savefig(out, bbox_inches='tight', dpi=200)
    print(f'[Saved] {out}')

print('\nSummary table:')
print(f'{"Method":<20} {"All MF":>9} {"L2_Struct":>11}')
print('-' * 42)
for key, label in METHODS:
    vals = results.get(key, {})
    a = vals.get('All MF') or float('nan')
    l = vals.get('L2_Structural') or float('nan')
    print(f'{label.replace(chr(10), " "):<20} {a:>9.4f} {l:>11.4f}')
