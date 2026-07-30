"""F7 — Case study (NDUFS4, Complex I): dual isoform-trajectory representation.
(a) NON-NORMALIZED raw projection onto multidomain axis-3: monotonic fan-out = layer
    magnitude drift (boundary artifact), obscures where divergence occurs.
(b) Z-NORMALIZED trajectory: early convergence (shared gene identity) -> mid-late
    divergence peak (domain architecture splits) -> partial reconvergence.
(c) within-gene spread(layer) raw vs z: why normalization is required.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fig_style as fs
fs.apply()
ROOT = os.path.join(os.path.dirname(__file__), '../..')
OUT = os.path.join(ROOT, 'reports/figures_v2')
SCR = '/tmp/claude-1811/-home-welcome1-sw1686-DIFFUSE/79479b99-e4cd-447b-b678-41709c5db7dd/scratchpad'
d = np.load(os.path.join(SCR, 'f7_ndufs4.npz'), allow_pickle=True)
raw, z, ids = d['raw'], d['z'], [str(x) for x in d['ids']]
L = np.arange(1, 31)
# highlight canonical Ensembl isoforms
hi = {i: (nm, c) for i, (nm, c) in
      [(ids.index('NDUFS4-201'), ('NDUFS4-201 (canonical)', fs.C_PRISM)),
       (ids.index('NDUFS4-204'), ('NDUFS4-204', fs.OI['verm']))] if nm in ids or True}
# most-divergent novel isoform (max |z| at late layers)
novel = [i for i in range(len(ids)) if i not in hi]
div_i = novel[int(np.argmax(np.abs(z[novel, 20:]).mean(1)))]
hi[div_i] = (ids[div_i][:18] + ' (novel)', fs.OI['green'])

fig = plt.figure(figsize=(7.2, 3.3))
gs = GridSpec(1, 3, figure=fig, wspace=0.42, width_ratios=[1, 1, 0.85])

def plot_bundle(ax, T, ylab, title):
    for i in range(T.shape[0]):
        if i in hi: continue
        ax.plot(L, T[i], color=fs.C_NULL, lw=0.8, alpha=0.35, zorder=1)
    for i, (nm, c) in hi.items():
        ax.plot(L, T[i], color=c, lw=1.8, zorder=3, label=nm)
    ax.axvline(15, ls='--', lw=0.7, color='#bbb')
    ax.set_xlabel('ESM-2 layer'); ax.set_ylabel(ylab); ax.set_title(title, loc='left', x=0.0, fontsize=8)
    ax.set_xticks([1, 10, 20, 30])

axA = fig.add_subplot(gs[0]); fs.panel_label(axA, 'a', dx=-0.22, dy=1.05)
plot_bundle(axA, raw, 'raw projection (ax3)', 'Non-normalized:\nmagnitude drift')
axA.text(0.03, 0.96, 'monotonic fan-out\n= boundary artifact', transform=axA.transAxes,
         fontsize=6, color='#777', va='top')

axB = fig.add_subplot(gs[1]); fs.panel_label(axB, 'b', dx=-0.20, dy=1.05)
plot_bundle(axB, z, 'z-normalized (ax3)', 'Z-normalized:\nconvergence $\\to$ divergence')
zsp = z.std(0); pk = int(np.argmax(zsp)) + 1
axB.axvline(pk, ls=':', lw=1.0, color=fs.OI['verm'])
axB.text(pk + 0.5, axB.get_ylim()[1] * 0.9, f'divergence\npeak L{pk}', fontsize=5.8, color=fs.OI['verm'])
axB.legend(fontsize=5.2, loc='lower left', handlelength=1.2)

axC = fig.add_subplot(gs[2]); fs.panel_label(axC, 'c', dx=-0.24, dy=1.05)
axC.set_title('Within-gene spread', loc='left', x=0.0, fontsize=8)
rn = raw.std(0) / raw.std(0).max(); zn = z.std(0) / z.std(0).max()
axC.plot(L, rn, color='#777', lw=1.5, label='raw (norm.)')
axC.plot(L, zn, color=fs.OI['verm'], lw=1.5, label='z-normalized')
axC.axvline(pk, ls=':', lw=0.9, color=fs.OI['verm'])
axC.set_xlabel('ESM-2 layer'); axC.set_ylabel('rel. isoform spread'); axC.set_xticks([1, 10, 20, 30])
axC.legend(fontsize=6, loc='upper left')
axC.text(0.5, 0.02, 'raw keeps rising (drift);\nz-norm peaks then\nreconverges', transform=axC.transAxes,
         fontsize=5.6, color='#555', va='bottom', ha='center')

fig.suptitle('Figure 7  |  Case study NDUFS4 (Complex I): isoforms converge early and diverge at mid-to-late layers on the multidomain axis  ·  BISECT Tier 2 (functional loss)',
             x=0.02, ha='left', fontsize=8.0, fontweight='bold', y=1.04)
p = os.path.join(OUT, 'F7_case')
fig.savefig(p + '.png'); fig.savefig(p + '.pdf')
print('saved', p + '.png', '| z divergence peak L', pk, '| highlighted', [ids[i] for i in hi])
