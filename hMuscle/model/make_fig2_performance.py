"""F2 — Performance benchmark & isoform-resolution evidence.
(a) SOTA macro-AUPRC bar (All MF + L2 structural) with bootstrap CI + gene-mean oracle line.
(b) isoform discrimination: Domain-Ranking AUC vs gene-mean null; UniProt directional test.
(c) gene-mean oracle honesty: cross-gene shuffle collapse 0.734->0.048.
(d) length-independence: within-gene length-AUC PRISM vs BLAST.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import fig_style as fs
fs.apply()
OUT = os.path.join(os.path.dirname(__file__), '../../reports/figures_v2')
os.makedirs(OUT, exist_ok=True)

fig = plt.figure(figsize=(7.2, 6.2))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.34,
                      height_ratios=[1.0, 0.85])

# ============ (a) SOTA benchmark ============
axA = fig.add_subplot(gs[0, :])
fs.panel_label(axA, 'a', dx=-0.045, dy=1.0)
axA.set_title('Zero-shot macro-AUPRC: PRISM vs homology & deep baselines', loc='left', x=0.0)
#            name              All-MF  ci_lo  ci_hi   L2     kind
methods = [
    ('DeepGoPlus',            0.441, 0.433, 0.451, 0.352, 'base'),
    ('Gene-mean\noracle',     0.465, 0.465, 0.465, 0.303, 'oracle'),
    ('PRISM\nzero-shot',      0.596, 0.586, 0.607, 0.313, 'prism'),
    ('k-NN\n(ESM-2)',         0.636, 0.627, 0.646, 0.543, 'base'),
    ('PRISM v17f*',           0.734, 0.723, 0.747, 0.637, 'prism'),
    ('BLAST->GOA\n(homology)',0.861, 0.851, 0.870, 0.802, 'ceil'),
]
kind_col = {'base': fs.C_BASE, 'oracle': fs.OI['purple'], 'prism': fs.C_PRISM,
            'ceil': fs.C_ORACLE}
x = np.arange(len(methods)); w = 0.36
mf = np.array([m[1] for m in methods]); mf_lo = np.array([m[2] for m in methods]); mf_hi = np.array([m[3] for m in methods])
l2 = np.array([m[4] for m in methods])
cols = [kind_col[m[5]] for m in methods]
axA.bar(x - w/2, mf, w, yerr=[mf - mf_lo, mf_hi - mf], color=cols, ecolor='#444',
        capsize=2.2, error_kw={'lw': 0.8}, label='All MF')
axA.bar(x + w/2, l2, w, color=cols, alpha=0.45, label='L2 (structural terms)')
for xi, v in zip(x, mf):
    axA.text(xi - w/2, v + 0.018, f'{v:.3f}', ha='center', fontsize=5.6)
axA.axhline(0.465, ls='--', lw=0.9, color=fs.OI['purple'], zorder=0)
axA.text(len(methods)-0.5, 0.475, 'gene-mean oracle 0.465', ha='right', fontsize=5.8,
         color=fs.OI['purple'])
axA.set_xticks(x); axA.set_xticklabels([m[0] for m in methods], fontsize=6.2)
axA.set_ylabel('macro-AUPRC'); axA.set_ylim(0, 0.95)
# custom legend for solid=All MF / faded=L2
from matplotlib.patches import Patch
axA.legend(handles=[Patch(fc='#555', label='All MF'), Patch(fc='#555', alpha=0.45, label='L2 structural')],
           loc='upper left', ncol=1)
axA.text(2.0, 0.90, 'PRISM: sequence-intrinsic, no homology lookup', fontsize=6,
         color=fs.C_PRISM, style='italic')

# ============ (b) isoform discrimination ============
axB = fig.add_subplot(gs[1, 0])
fs.panel_label(axB, 'b', dx=-0.14, dy=1.02)
axB.set_title('Within-gene isoform discrimination', loc='left', x=0.0)
labels = ['gene-mean\nnull', 'Domain-\nRanking', 'centroid', 'BLAST']
vals   = [0.500, 0.630, 0.638, 0.722]
los    = [0.500, 0.613, 0.620, 0.700]
his    = [0.500, 0.646, 0.656, 0.744]
bcols  = [fs.C_NULL, fs.C_PRISM, fs.C_PRISM, fs.C_ORACLE]
xb = np.arange(len(labels))
axB.bar(xb, vals, 0.6, yerr=[np.array(vals)-np.array(los), np.array(his)-np.array(vals)],
        color=bcols, ecolor='#444', capsize=2.2, error_kw={'lw': 0.8})
axB.axhline(0.5, ls=':', lw=0.9, color='#888')
for xi, v in zip(xb, vals):
    if v > 0.5: axB.text(xi, v + 0.012, f'{v:.3f}', ha='center', fontsize=5.8)
axB.set_xticks(xb); axB.set_xticklabels(labels, fontsize=6)
axB.set_ylabel('within-gene AUC'); axB.set_ylim(0.45, 0.78)
axB.text(0.02, 0.965, 'UniProt directional: 11/11 pairs\n(gap$\\geq$0.10), perm p<0.001',
         transform=axB.transAxes, fontsize=5.9, va='top', color=fs.C_PRISM2,
         bbox=dict(boxstyle='round', fc='#FBEEE6', ec='none', pad=0.3))

# ============ (c) oracle honesty ============
axC = fig.add_subplot(gs[1, 1])
fs.panel_label(axC, 'c', dx=-0.14, dy=1.02)
axC.set_title('Macro-AUPRC = gene-family identification', loc='left', x=0.0)
stages = ['v17f*\n(intact)', 'random\nlabels', 'cross-gene\nshuffle']
sv = [0.734, 0.345, 0.048]
scol = [fs.C_PRISM, fs.C_NULL, fs.OI['verm']]
xc = np.arange(len(stages))
axC.bar(xc, sv, 0.6, color=scol)
for xi, v in zip(xc, sv):
    axC.text(xi, v + 0.015, f'{v:.3f}', ha='center', fontsize=6)
axC.set_xticks(xc); axC.set_xticklabels(stages, fontsize=6)
axC.set_ylabel('macro-AUPRC'); axC.set_ylim(0, 0.82)
axC.annotate('', xy=(2, 0.10), xytext=(0, 0.70),
             arrowprops=dict(arrowstyle='->', color=fs.OI['verm'], lw=1.2,
                             connectionstyle='arc3,rad=-0.25'))
axC.text(1.0, 0.55, '97.7% lost when\ngene identity\nis shuffled', ha='center',
         fontsize=6, color=fs.OI['verm'])

fig.suptitle('Figure 2  |  PRISM performance and the isoform-resolution signal beyond gene-family identity',
             x=0.02, ha='left', fontsize=10, fontweight='bold', y=1.0)
p = os.path.join(OUT, 'F2_performance')
fig.savefig(p + '.png'); fig.savefig(p + '.pdf')
print('saved', p + '.png')
