"""F6 — BISECT: multi-evidence isoform-switch characterization pipeline + case statistics.
(a) pipeline schematic: PRISM delta -> 15 evidence modules -> mechanism tier.
(b) tier distribution across 83 characterized cases.
(c) switch mechanism composition + structural-consequence stats.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from collections import Counter
import fig_style as fs
fs.apply()
ROOT = os.path.join(os.path.dirname(__file__), '../..')
OUT = os.path.join(ROOT, 'reports/figures_v2')
rows = list(csv.DictReader(open(os.path.join(ROOT, 'reports/supplementary_table_S_bisect_83cases.tsv')), delimiter='\t'))
N = len(rows)
tier_c = Counter(r['prism_tier'] for r in rows)
mech_c = Counter(r['mechanism_type'] for r in rows)
n_domloss = sum(1 for r in rows if r.get('domains_lost', '').strip())
n_af = sum(1 for r in rows if r.get('af_ad_plddt_mean', 'NA') not in ('NA', ''))
n_ppi = sum(1 for r in rows if r.get('ppi_verdict', 'NA') not in ('NA', ''))
n_dtu = sum(1 for r in rows if r.get('dtu_p', 'NA') not in ('NA', '') and r.get('dtu_note','')=='dtu_tested')

fig = plt.figure(figsize=(7.2, 6.2))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], hspace=0.5, wspace=0.42)

# ===== (a) pipeline schematic =====
axA = fig.add_subplot(gs[0, :]); axA.set_xlim(0, 10); axA.set_ylim(0, 4); axA.axis('off')
fs.panel_label(axA, 'a', dx=-0.01, dy=1.0)
axA.set_title('BISECT: multi-evidence isoform-switch characterization', loc='left', x=0.0)
def bx(x, y, w, h, t, c, fs_=6):
    axA.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                 fc=c, ec='#777', lw=0.7)); axA.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=fs_)
def ar(x1, y1, x2, y2):
    axA.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=10, color='#555', lw=1.1))
bx(0.15, 1.5, 1.5, 1.0, 'PRISM\n$\\delta$-score\nswitch pair', '#D7E9F7')
ar(1.65, 2.0, 2.25, 2.0)
# evidence modules column
mods = [('DTU / DRIMSeq', n_dtu), ('Pfam domain loss', n_domloss), ('AlphaFold pLDDT', n_af),
        ('PPI interface', n_ppi), ('conservation phyloP', None), ('TSS / APA origin', None),
        ('NMD / NAT', None), ('convergence', None)]
for i, (m, cnt) in enumerate(mods):
    yy = 3.55 - (i % 4) * 0.95; xx = 2.3 + (i // 4) * 1.95
    lab = m + (f'\n(n={cnt})' if cnt is not None else '')
    bx(xx, yy - 0.42, 1.85, 0.78, lab, '#CFE8DE', fs_=5.4)
axA.text(4.2, 3.9, '15 evidence modules', ha='center', fontsize=6, fontweight='bold')
ar(6.25, 2.0, 6.9, 2.0)
bx(6.95, 1.4, 1.5, 1.2, 'evidence\nintegration', '#F5E6C4')
ar(8.45, 2.0, 9.0, 2.0)
bx(9.0, 1.2, 0.95, 1.6, 'mechanism\ntier\n1 / 2 / 3', '#F5D9C4', fs_=5.6)

# ===== (b) tier distribution =====
axB = fig.add_subplot(gs[1, 0])
fs.panel_label(axB, 'b', dx=-0.16, dy=1.03)
axB.set_title(f'Mechanism-tier distribution (n={N})', loc='left', x=0.0, fontsize=8.2)
order = ['tier1_functional_switch', 'tier2_functional_loss', 'tier2_complex_loss',
         'tier2_partial_change', 'tier3_gene_median', 'tier3_structural_only']
olab = ['T1 functional switch', 'T2 functional loss', 'T2 complex loss',
        'T2 partial change', 'T3 gene-median', 'T3 structural only']
tcol = {'tier1': fs.OI['verm'], 'tier2': fs.OI['blue'], 'tier3': fs.C_BASE}
cols = [tcol[o.split('_')[0]] for o in order]
vals = [tier_c.get(o, 0) for o in order]
y = np.arange(len(order))[::-1]
axB.barh(y, vals, 0.66, color=cols)
for yi, v in zip(y, vals):
    if v: axB.text(v + 0.5, yi, str(v), va='center', fontsize=6.5)
axB.set_yticks(y); axB.set_yticklabels(olab, fontsize=6.2)
axB.set_xlabel('cases'); axB.set_xlim(0, 58)
from matplotlib.patches import Patch
axB.legend(handles=[Patch(fc=fs.OI['verm'], label='Tier 1 (switch)'),
                    Patch(fc=fs.OI['blue'], label='Tier 2 (loss)'),
                    Patch(fc=fs.C_BASE, label='Tier 3 (quant.)')], fontsize=5.8, loc='lower right')

# ===== (c) mechanism + structural stats =====
axC = fig.add_subplot(gs[1, 1])
fs.panel_label(axC, 'c', dx=-0.16, dy=1.03)
axC.set_title('Switch mechanism & structural consequence', loc='left', x=0.0, fontsize=8.2)
morder = ['alternative_promoter', 'transcriptional', 'epigenetic_derepression', 'alternative_splicing']
mlab = ['alt.\npromoter', 'transcript-\nional', 'epigenetic\nderepr.', 'alt.\nsplicing']
mcol = [fs.OI['blue'], fs.OI['green'], fs.OI['purple'], fs.OI['orange']]
mv = [mech_c.get(m, 0) for m in morder]
axC.bar(range(4), mv, 0.62, color=mcol)
for i, v in enumerate(mv): axC.text(i, v + 0.7, str(v), ha='center', fontsize=6.5)
axC.set_xticks(range(4)); axC.set_xticklabels(mlab, fontsize=6)
axC.set_ylabel('cases'); axC.set_ylim(0, 55)
axC.text(0.97, 0.95, f'{n_domloss}/{N} cases lose\n$\\geq$1 Pfam domain\n({n_domloss/N*100:.0f}%)',
         transform=axC.transAxes, ha='right', va='top', fontsize=6, color=fs.OI['verm'],
         bbox=dict(boxstyle='round', fc='#FBEEE6', ec='none', pad=0.3))

fig.suptitle('Figure 6  |  BISECT translates PRISM isoform-switch signals into mechanistic, multi-evidence hypotheses',
             x=0.02, ha='left', fontsize=9.3, fontweight='bold', y=1.0)
p = os.path.join(OUT, 'F6_bisect')
fig.savefig(p + '.png'); fig.savefig(p + '.pdf')
print('saved', p + '.png', '| domloss', n_domloss, 'dtu', n_dtu, 'af', n_af, 'ppi', n_ppi)
