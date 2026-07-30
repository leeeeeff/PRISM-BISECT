"""F5 — The encoding boundary: domain-level captured, motif/non-canonical direction not.
(a) per-type zero-shot AUPRC gradient: SameDomain > NoDomain > PartialTrunc ~ DomainLoss.
(b) canonical vs non-canonical resolution: mean GO-correlation & top-consistency (S1 vs S2).
(c) where S2 (non-canonical) failures concentrate: domain-type composition + SLiM note.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fig_style as fs
fs.apply()
ROOT = os.path.join(os.path.dirname(__file__), '../..')
OUT = os.path.join(ROOT, 'reports/figures_v2')

# --- data ---
rows = [l.split('\t') for l in open(os.path.join(ROOT, 'reports/isoform_resolution_full/type_auprc_stats.tsv')).read().splitlines()]
hdr = rows[0]; d = {r[0]: dict(zip(hdr, r)) for r in rows[1:]}
def g(t, k): return float(d[t][k])
att = json.load(open(os.path.join(ROOT, 'reports/isoform_resolution_full/go_attribution_summary.json')))
mech = json.load(open(os.path.join(ROOT, 'reports/isoform_resolution_full/go_attribution_mech_summary.json')))

fig = plt.figure(figsize=(7.2, 3.4))
gs = GridSpec(1, 3, figure=fig, wspace=0.5, width_ratios=[1.05, 0.95, 1.0])

# ===== (a) per-type AUPRC =====
axA = fig.add_subplot(gs[0])
fs.panel_label(axA, 'a', dx=-0.22, dy=1.04)
axA.set_title('Domain sharing $\\to$\nbest prediction', loc='left', x=0.0, fontsize=8)
types = ['Type3_SameDomain', 'Type0_NoDomain', 'Type2_PartialTrunc', 'Type1_DomainLoss']
tlab = ['same\ndomain', 'no\ndomain', 'partial\ntrunc.', 'domain\nloss']
vals = [g(t, 'auprc') for t in types]
lo = [g(t, 'ci_lo') for t in types]; hi = [g(t, 'ci_hi') for t in types]
tcol = [fs.OI['green'], fs.C_BASE, fs.OI['orange'], fs.OI['verm']]
y = np.arange(len(types))[::-1]
axA.barh(y, vals, 0.62, xerr=[np.array(vals) - lo, np.array(hi) - vals],
         color=tcol, ecolor='#444', capsize=2.2, error_kw={'lw': 0.8})
for yi, v, t in zip(y, vals, types):
    axA.text(v + 0.012, yi, f'{v:.3f}', va='center', fontsize=6)
axA.set_yticks(y); axA.set_yticklabels(tlab, fontsize=6.5)
axA.set_xlabel('zero-shot AUPRC'); axA.set_xlim(0.5, 0.87)
axA.axvline(0.5, ls=':', lw=0.8, color='#999')
axA.text(0.97, 0.06, 'tiers (i)/(ii):\ndomain encoded', transform=axA.transAxes,
         fontsize=5.6, color=fs.OI['green'], ha='right', va='bottom')

# ===== (b) canonical vs non-canonical =====
axB = fig.add_subplot(gs[1])
fs.panel_label(axB, 'b', dx=-0.24, dy=1.04)
axB.set_title('Non-canonical switch =\nresolution boundary', loc='left', x=0.0, fontsize=8)
grp = ['S1: canonical\ndominant', 'S2: non-canonical\nshould dominate']
corr = [att['mean_go_corr_s1'], att['mean_go_corr_s2']]
cons = [att['top_consistency_s1'], att['top_consistency_s2']]
x = np.arange(2); w = 0.36
axB.bar(x - w/2, corr, w, color=[fs.OI['blue'], fs.OI['verm']], label='mean GO corr.')
axB.bar(x + w/2, cons, w, color=[fs.OI['blue'], fs.OI['verm']], alpha=0.45, label='top-GO consistency')
axB.axhline(0, color='k', lw=0.7)
for xi, c in zip(x - w/2, corr): axB.text(xi, c + (0.04 if c > 0 else -0.05), f'{c:.2f}', ha='center', va='bottom' if c > 0 else 'top', fontsize=6)
for xi, c in zip(x + w/2, cons): axB.text(xi, c + 0.04, f'{c:.2f}', ha='center', fontsize=6)
axB.set_xticks(x)
axB.set_xticklabels([f"S1: canonical\ndominant\n(n={att['scenario1_n']})",
                     f"S2: non-canonical\nshould dominate\n(n={att['scenario2_n']})"], fontsize=6.0)
axB.set_ylabel('correlation / consistency'); axB.set_ylim(-0.45, 1.18)
axB.legend(fontsize=5.8, loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.0))

# ===== (c) S2 mechanism composition =====
axC = fig.add_subplot(gs[2])
fs.panel_label(axC, 'c', dx=-0.20, dy=1.04)
axC.set_title('Residual failures where\ndomains are shared', loc='left', x=0.0, fontsize=8)
s2 = mech['s2_type_fractions']
order = ['Type3_SameDomain', 'Type0_NoDomain', 'Type1_DomainLoss', 'Type2_PartialTrunc']
olab = ['same domain', 'no domain', 'domain loss', 'partial trunc.']
fr = [s2[t] * 100 for t in order]
ocol = [fs.OI['green'], fs.C_BASE, fs.OI['verm'], fs.OI['orange']]
axC.bar(range(4), fr, 0.62, color=ocol)
for i, v in enumerate(fr): axC.text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=6)
axC.set_xticks(range(4)); axC.set_xticklabels(olab, rotation=30, ha='right', fontsize=6)
axC.set_ylabel('% of S2 (non-canonical) cases'); axC.set_ylim(0, 52)
axC.text(0.02, 0.94, 'tier (iii): SLiM/motif-level\ndirection not encoded\n(58 genes differ by SLiM only)',
         transform=axC.transAxes, fontsize=5.6, color=fs.OI['verm'], va='top',
         bbox=dict(boxstyle='round', fc='#FBEEE6', ec='none', pad=0.3))

fig.suptitle('Figure 5  |  The encoding boundary: PRISM captures domain architecture but not motif-level functional direction',
             x=0.02, ha='left', fontsize=9.3, fontweight='bold', y=1.06)
p = os.path.join(OUT, 'F5_boundary')
fig.savefig(p + '.png'); fig.savefig(p + '.pdf')
print('saved', p + '.png')
