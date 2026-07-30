"""F4 — PCA axis biology: evr-inverse principle + 3-stage verification + layer-cluster x axis.
(a) STEP1 evr-inverse: explained-variance-ratio vs within-gene variance fraction per axis.
(b) 3-stage verification reasoning flow (schematic).
(c) layer-attribution cluster (Early/Mid/Final) x dominant PCA axis distribution.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import fig_style as fs
fs.apply()
ROOT = os.path.join(os.path.dirname(__file__), '../..')
OUT = os.path.join(ROOT, 'reports/figures_v2')
SCR = '/tmp/claude-1811/-home-welcome1-sw1686-DIFFUSE/79479b99-e4cd-447b-b678-41709c5db7dd/scratchpad'

vd = json.load(open(os.path.join(ROOT, 'reports/v20b_pca_interp/variance_decomp.json')))
evr = np.array(vd['explained_var_ratio'])
wf = np.array([np.mean(vd['per_axis'][str(a)]['within_frac_by_layer']) for a in range(8)])
N, G = 63994, 18514
null_wf = (N - G) / (N - 1)
M = np.load(os.path.join(SCR, 'bin_axis_M.npy'))   # (3,8) bin x axis

fig = plt.figure(figsize=(7.2, 6.6))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05], hspace=0.44, wspace=0.36)

# ===== (a) evr-inverse scatter =====
axA = fig.add_subplot(gs[0, 0])
fs.panel_label(axA, 'a', dx=-0.17, dy=1.02)
axA.set_title('evr-inverse: top-variance axis is\nleast isoform-discriminative', loc='left', x=0.0, fontsize=8.2)
for a in range(8):
    axA.scatter(evr[a] * 100, wf[a], s=70, color=fs.AXCOL[a], edgecolor='k', lw=0.5, zorder=3)
    axA.annotate(f'ax{a}', (evr[a] * 100, wf[a]), fontsize=6, ha='center', va='center',
                 color='w' if a not in (6,) else 'k', zorder=4)
axA.axhline(null_wf, ls='--', lw=0.9, color='#888')
axA.text(4.3, null_wf - 0.015, f'random-assignment null {null_wf:.2f}', fontsize=6, color='#666',
         va='top', ha='right')
from scipy.stats import spearmanr
rho, pv = spearmanr(evr, wf)
axA.text(0.97, 0.62, 'all axes far below null\n= gene-family dominated', transform=axA.transAxes,
         fontsize=6, color='#666', ha='right', va='top')
axA.annotate('ax0: highest evr,\nlowest within-frac', (evr[0]*100, wf[0]), xytext=(3.1, 0.36),
             fontsize=6, color=fs.OI['blue'], ha='center',
             arrowprops=dict(arrowstyle='->', color=fs.OI['blue'], lw=0.7))
axA.annotate('ax3/ax6 carry the\nmost isoform signal\n(low evr, high within-frac)',
             (evr[6]*100, wf[6]), xytext=(2.4, 0.50), fontsize=6, color=fs.OI['verm'], ha='center',
             arrowprops=dict(arrowstyle='->', color=fs.OI['verm'], lw=0.7))
axA.set_xlabel('explained-variance ratio (%)'); axA.set_ylabel('within-gene variance fraction')
axA.set_xlim(1.0, 4.7); axA.set_ylim(0.20, 0.75)

# ===== (b) 3-stage reasoning flow =====
axB = fig.add_subplot(gs[0, 1]); axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis('off')
fs.panel_label(axB, 'b', dx=-0.02, dy=1.02)
axB.set_title('3-stage axis-biology verification', loc='left', x=0.0)
def sbox(y, t, c):
    axB.add_patch(FancyBboxPatch((0.4, y), 9.2, 1.9, boxstyle='round,pad=0.05,rounding_size=0.15',
                                 fc=c, ec='#666', lw=0.7))
    axB.text(5.0, y + 0.95, t, ha='center', va='center', fontsize=6.3)
sbox(7.4, 'STEP 1 · null-anchored variance decomposition\nwithin-frac vs random null 0.71 -> all 8 axes gene-family dominated', '#D7E9F7')
sbox(4.4, 'STEP 2 · gene-level vs within-gene-residual Spearman\nresidual test removes gene-family / SLiM confounds', '#CFE8DE')
sbox(1.4, 'STEP 3 · domain-family tercile log-odds enrichment\nassigns biology to weak (low-evr) axes', '#F5E6C4')
axB.add_patch(FancyArrowPatch((5, 7.4), (5, 6.3), arrowstyle='-|>', mutation_scale=12, color='#555', lw=1.2))
axB.add_patch(FancyArrowPatch((5, 4.4), (5, 3.3), arrowstyle='-|>', mutation_scale=12, color='#555', lw=1.2))
axB.text(5.0, 0.5, 'ax0 soluble-b/TM · ax1 LRR/Ig · ax2 pro-turn · ax3 multidomain\nax4 helix/cys · ax5 linker · ax6 KRAB-ZNF · ax7 acidic-helix',
         fontsize=4.9, ha='center', color='#555')

# ===== (c) layer-cluster x axis heatmap =====
axC = fig.add_subplot(gs[1, :])
fs.panel_label(axC, 'c', dx=-0.04, dy=1.02)
axC.set_title('Layer-attribution cluster $\\times$ dominant PCA axis (per-isoform, N=63,994)', loc='left', x=0.0)
im = axC.imshow(M * 100, aspect='auto', cmap='YlGnBu', vmin=0)
axC.set_yticks(range(3)); axC.set_yticklabels(['Early\n[L1-10]', 'Mid\n[L11-20]', 'Final\n[L21-29]'], fontsize=7)
axC.set_xticks(range(8)); axC.set_xticklabels(fs.AXLAB, rotation=35, ha='right', fontsize=6)
for bi in range(3):
    for a in range(8):
        v = M[bi, a] * 100
        axC.text(a, bi, f'{v:.0f}', ha='center', va='center', fontsize=6,
                 color='w' if v > 22 else '#333')
cb = fig.colorbar(im, ax=axC, fraction=0.025, pad=0.01)
cb.set_label('% of isoforms in cluster', fontsize=6.5); cb.ax.tick_params(labelsize=6)
# highlight Mid=ax3 (delta_layer confirmation)
axC.add_patch(plt.Rectangle((3 - 0.5, 1 - 0.5), 1, 1, fill=False, ec=fs.OI['verm'], lw=2))
axC.text(3, 1 - 0.62, r'Mid$\to$ax3 = $\delta_{layer}$ site', ha='center', fontsize=5.8,
         color=fs.OI['verm'], va='bottom')

fig.suptitle('Figure 4  |  Biological identity of ESM-2 trajectory axes and their coupling to layer-attribution clusters',
             x=0.02, ha='left', fontsize=9.5, fontweight='bold', y=1.0)
p = os.path.join(OUT, 'F4_axis')
fig.savefig(p + '.png'); fig.savefig(p + '.pdf')
print('saved', p + '.png', '| rho', round(rho, 3))
