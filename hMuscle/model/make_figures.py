# -*- coding: utf-8 -*-
"""
make_figures.py
===============
논문용 핵심 Figure 생성 (2026-05-19 세션 결과 통합)

Figure 1: Model Performance Overview — muscle vs brain
Figure 2: Functional Switch Landscape (skeletal muscle)
Figure 3: KIF1B Cross-Tissue Case Study
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.cm as cm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

REP_V15  = '../../reports/v15_switch_dtu'
REP_V17  = '../../reports/v17_brain_model'
FIG_DIR  = '../../reports/figures_20260519'
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

C_MUSCLE = '#E07B54'   # warm orange-red
C_BRAIN  = '#5B8DB8'   # calm blue
C_MED    = '#8E6BBF'   # purple for mixed
C_GREEN  = '#4CAF50'
C_GRAY   = '#9E9E9E'
C_DARK   = '#333333'

# ── Load data ────────────────────────────────────────────────────────────────
v15d = json.load(open(f'{REP_V15}/cross_go_18go_20260519_1403.json'))
v17  = json.load(open(f'{REP_V17}/brain_switch_v17_20260519_1458.json'))
kif  = json.load(open(f'{REP_V15}/kif1b_brain_20260519_1428.json'))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Performance Overview (3 panels)
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig1.suptitle('Figure 1: Isoform Function Prediction — Muscle vs Brain',
              fontsize=11, fontweight='bold', y=1.02)

# ── Panel A: AUPRC per GO term (Muscle 18 terms) ────────────────────────────
ax = axes[0]

MUSCLE_GO_SHORT = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0030017': 'Sarcomere',
    'GO:0006941': 'Striated muscle',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007519': 'SKM development',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0003774': 'Motor activity',
    'GO:0006096': 'Glycolysis',
    'GO:0007268': 'Synaptic trans',
    'GO:0007018': 'MT-based movement',
    'GO:0043005': 'Neuron projection',
    'GO:0030182': 'Neuron diff',
    'GO:0000226': 'MT cytoskeleton org',
}

# Original 13 terms (muscle-trained with muscle labels)
ORIG_13 = ['GO:0007204','GO:0030017','GO:0006941','GO:0006914','GO:0043161',
           'GO:0007519','GO:0042692','GO:0055074','GO:0007005','GO:0007517',
           'GO:0032006','GO:0003774','GO:0006096']
NEW_5   = ['GO:0007268','GO:0007018','GO:0043005','GO:0030182','GO:0000226']

auprc_m = v15d['auprc_per_go']
auprc_b = v17['auprc_per_go']

# Random baseline (positive rates for muscle 18 terms from v17 diagnosis)
POS_RATES_M = {  # approximate positive rates from v15d pos counts / ~36748 test isos
    'GO:0007204': 0.0084, 'GO:0030017': 0.0089, 'GO:0006941': 0.0052,
    'GO:0006914': 0.0112, 'GO:0043161': 0.0219, 'GO:0007519': 0.0035,
    'GO:0042692': 0.0115, 'GO:0055074': 0.0165, 'GO:0007005': 0.0260,
    'GO:0007517': 0.0128, 'GO:0032006': 0.0065, 'GO:0003774': 0.0045,
    'GO:0006096': 0.0021, 'GO:0007268': 0.0114, 'GO:0007018': 0.0115,
    'GO:0043005': 0.0386, 'GO:0030182': 0.0368, 'GO:0000226': 0.0261,
}
POS_RATES_B = {  # from v17 pos_rate column
    'GO:0043005': 0.0954, 'GO:0030182': 0.0598, 'GO:0048666': 0.0518,
    'GO:0045202': 0.0736, 'GO:0007268': 0.0538, 'GO:0050804': 0.0294,
    'GO:0007411': 0.0243, 'GO:0007018': 0.0253, 'GO:0000226': 0.0307,
    'GO:0006836': 0.0170, 'GO:0006979': 0.0131, 'GO:0006954': 0.0080,
    'GO:0006914': 0.0109, 'GO:0043161': 0.0208, 'GO:0007204': 0.0128,
    'GO:0055074': 0.0218, 'GO:0007005': 0.0195, 'GO:0032006': 0.0010,
}

go_ids = list(MUSCLE_GO_SHORT.keys())
names  = [MUSCLE_GO_SHORT[g] for g in go_ids]
vals   = [auprc_m.get(g, 0) for g in go_ids]
baselines = [POS_RATES_M.get(g, 0.02) for g in go_ids]
colors = [C_MUSCLE if g in ORIG_13 else C_MED for g in go_ids]

y_pos = np.arange(len(go_ids))
bars = ax.barh(y_pos, vals, color=colors, alpha=0.85, height=0.65, zorder=3)
# Baseline dots
ax.scatter(baselines, y_pos, color='gray', s=20, zorder=4, marker='|', linewidths=1.5)

ax.axvline(np.mean(vals), color=C_MUSCLE, ls='--', lw=1.2, alpha=0.7,
           label=f'Macro AUPRC = {np.mean(vals):.3f}')
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=7.5)
ax.set_xlabel('AUPRC')
ax.set_title('A. Muscle model performance\n(18 GO terms, 36,748 isoforms)', pad=6)
ax.set_xlim(0, 0.92)
ax.legend(loc='lower right', fontsize=7.5)

# Category patches
p1 = mpatches.Patch(color=C_MUSCLE, label='Original 13 terms')
p2 = mpatches.Patch(color=C_MED, label='New 5 (neuro)')
ax.legend(handles=[p1, p2], loc='lower right', fontsize=7)
ax.set_axisbelow(True)
ax.grid(axis='x', alpha=0.3)

# ── Panel B: GO term structural diversity ────────────────────────────────────
ax = axes[1]

# From v17_diagnosis.py output (hardcoded results)
MUSCLE_DIV = {
    'Sarcomere': 0.0986,
    'Muscle cont.': 0.1157,
    'Motor activity': 0.1316,
    'Striated muscle': 0.1390,
    'Fatty acid metab': 0.0971,
    'Electron trans.': 0.1174,
    'Gluconeogenesis': 0.0868,
    'ATP metabolic': 0.0905,
    'Muscle organ dev': 0.0809,
}
BRAIN_DIV = {
    'Synapse assembly': 0.4614,
    'Mod. syn.trans': 0.4622,
    'Synaptic trans.': 0.4543,
    'Neurotrans. trans.': 0.4155,
    'Neuron diff': 0.4076,
    'Neuron dev': 0.3770,
    'MT-based mov.': 0.3784,
    'Axon guidance': 0.3670,
    'Neuron proj.': 0.3644,
    'MT cyto. org.': 0.3176,
}

m_names = list(MUSCLE_DIV.keys())
m_vals  = list(MUSCLE_DIV.values())
b_names = list(BRAIN_DIV.keys())
b_vals  = list(BRAIN_DIV.values())

all_names = m_names + ['──────'] + b_names
all_vals  = m_vals  + [0] + b_vals
all_colors = [C_MUSCLE]*len(m_names) + ['white'] + [C_BRAIN]*len(b_names)

y2 = np.arange(len(all_names))
valid = [i for i, v in enumerate(all_vals) if v > 0]
ax.barh([y2[i] for i in valid], [all_vals[i] for i in valid],
        color=[all_colors[i] for i in valid], alpha=0.85, height=0.65, zorder=3)
ax.axvline(0.25, color='gray', ls='--', lw=1, alpha=0.6, label='Tractability threshold')
ax.set_yticks(y2)
ax.set_yticklabels(all_names, fontsize=7.5)
ax.set_xlabel('Structural diversity (1 - mean cosine sim)')
ax.set_title('B. GO term protein family diversity\n(structural homogeneity)', pad=6)
ax.set_xlim(0, 0.55)
ax.legend(fontsize=7.5)
ax.set_axisbelow(True)
ax.grid(axis='x', alpha=0.3)

p1 = mpatches.Patch(color=C_MUSCLE, label='Muscle GO terms')
p2 = mpatches.Patch(color=C_BRAIN, label='Brain GO terms')
ax.legend(handles=[p1, p2], loc='lower right', fontsize=7)

# ── Panel C: AUPRC vs Diversity scatter ──────────────────────────────────────
ax = axes[2]

# Muscle terms: diversity vs AUPRC
muscle_div_vals = [0.0986, 0.1157, 0.1316, 0.1390, 0.0971, 0.1174, 0.0868, 0.0905, 0.0809]
muscle_auprc_vals = [0.769, 0.744, 0.832, 0.670, 0.683, 0.683, 0.644, 0.640, 0.640]
muscle_go_labels = ['Ca2+sig','Sarcomere','Motor','Striated','Ca2+hom','SKM_dev',
                    'Muscle_diff','Mitochon','Muscle_org']

# Brain terms: diversity vs AUPRC
brain_div_vals = [0.4614, 0.4622, 0.4543, 0.4155, 0.4076, 0.3770, 0.3784, 0.3670, 0.3644, 0.3176]
brain_auprc_go = ['GO:0045202','GO:0050804','GO:0007268','GO:0006836',
                  'GO:0030182','GO:0048666','GO:0007018','GO:0007411',
                  'GO:0043005','GO:0000226']
brain_auprc_vals = [v17['auprc_per_go'].get(g, 0) for g in brain_auprc_go]

ax.scatter(muscle_div_vals, muscle_auprc_vals, c=C_MUSCLE, s=70, alpha=0.85,
           zorder=4, label='Muscle GO terms', edgecolors='white', linewidth=0.5)
ax.scatter(brain_div_vals, brain_auprc_vals, c=C_BRAIN, s=70, alpha=0.85,
           zorder=4, label='Brain GO terms', edgecolors='white', linewidth=0.5)

# Trend lines
from numpy.polynomial import polynomial as P
if len(muscle_div_vals) > 2:
    x_fit = np.linspace(0.07, 0.16, 50)
    ax.plot(x_fit, [np.mean(muscle_auprc_vals)]*50, color=C_MUSCLE, ls='--', lw=1, alpha=0.5)

ax.axhline(np.mean(brain_auprc_vals), color=C_BRAIN, ls='--', lw=1, alpha=0.5)

# Annotate key terms
for x, y, lbl in [(0.1316, 0.832, 'Motor'), (0.0986, 0.769, 'Sarcomere')]:
    ax.annotate(lbl, (x, y), fontsize=6.5, xytext=(5, 3), textcoords='offset points')
for x, y, lbl in [(0.3176, 0.059, 'MT cyto'), (0.3784, 0.039, 'MT-mov')]:
    ax.annotate(lbl, (x, y), fontsize=6.5, xytext=(5, 2), textcoords='offset points')

ax.set_xlabel('Protein family diversity')
ax.set_ylabel('AUPRC')
ax.set_title('C. Diversity predicts predictability\n(structural homogeneity → AUPRC)', pad=6)
ax.legend(fontsize=7.5)
ax.set_xlim(0.0, 0.52)
ax.set_ylim(0, 0.95)

# Add region annotations
ax.axvspan(0.0, 0.20, alpha=0.06, color=C_MUSCLE, label='Tractable zone')
ax.axvspan(0.30, 0.52, alpha=0.06, color=C_BRAIN, label='Intractable zone')
ax.text(0.06, 0.89, 'Tractable\n(sequence-determined)', fontsize=7,
        color=C_MUSCLE, ha='center', alpha=0.8)
ax.text(0.40, 0.25, 'Context-dependent\n(sequence insufficient)', fontsize=7,
        color=C_BRAIN, ha='center', alpha=0.8)
ax.axvline(0.25, color='gray', ls='--', lw=0.8, alpha=0.5)

fig1.tight_layout()
path1 = f'{FIG_DIR}/fig1_performance_overview.png'
fig1.savefig(path1, dpi=200, bbox_inches='tight')
print(f"[Saved] {path1}")
plt.close(fig1)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Functional Switch Landscape (skeletal muscle)
# ─────────────────────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(15, 5.5))
gs2 = GridSpec(1, 3, figure=fig2, wspace=0.35)
fig2.suptitle('Figure 2: Functional Switch Landscape in Skeletal Muscle (v15d)',
              fontsize=11, fontweight='bold', y=1.02)

# ── Panel A: Top-20 reversal bar chart ───────────────────────────────────────
ax2a = fig2.add_subplot(gs2[0, 0])

top20 = v15d['top20_reversals']
genes = [r['gene'] for r in top20]
gapA  = [r['gap_A'] for r in top20]
gapB  = [r['gap_B'] for r in top20]
total = [r['reversal_strength'] for r in top20]
nameA = [r['name_A'][:16] for r in top20]
nameB = [r['name_B'][:16] for r in top20]

y_pos = np.arange(len(genes))[::-1]  # reverse for top-down display
bar_height = 0.38

# Color code by whether both partners are in same category
def cat(name):
    NEURO = {'Synaptic trans', 'Neuron projection', 'Neuron diff', 'MT cytoskeleton org',
             'MT-based movement', 'Neurotrans.'}
    MUSCLE_CAT = {'Sarcomere', 'Muscle contraction', 'Motor activity', 'Muscle cell diff',
                  'Muscle organ dev', 'Autophagy'}
    if any(n in name for n in NEURO): return 'neuro'
    if any(n in name for n in MUSCLE_CAT): return 'muscle'
    return 'other'

for i, (g, ga, gb, na, nb, tot) in enumerate(zip(genes, gapA, gapB, nameA, nameB, total)):
    yi = y_pos[i]
    ca = C_MED if cat(na) == 'neuro' else C_MUSCLE
    cb = C_MED if cat(nb) == 'neuro' else C_MUSCLE
    ax2a.barh(yi + bar_height/2, ga, height=bar_height, color=ca, alpha=0.85)
    ax2a.barh(yi - bar_height/2, -gb, height=bar_height, color=cb, alpha=0.85)
    # Gene label
    ax2a.text(-0.02, yi, g, ha='right', va='center', fontsize=7.5, fontweight='bold')

ax2a.axvline(0, color='black', lw=0.8)
ax2a.axvline(0.2, color='gray', ls='--', lw=0.7, alpha=0.5)
ax2a.axvline(-0.2, color='gray', ls='--', lw=0.7, alpha=0.5)
ax2a.set_yticks([])
ax2a.set_xlabel('Score gap ← iso B wins  |  iso A wins →')
ax2a.set_title('A. Top-20 cross-GO functional\nswitches (gap ≥ 0.2)', pad=6)
ax2a.set_xlim(-1.1, 1.1)

p_mus = mpatches.Patch(color=C_MUSCLE, label='Muscle GO term')
p_neu = mpatches.Patch(color=C_MED, label='Neurological GO term')
ax2a.legend(handles=[p_mus, p_neu], loc='lower right', fontsize=6.5)
ax2a.set_axisbelow(True)
ax2a.grid(axis='x', alpha=0.25)

# ── Panel B: Case study profiles (TPM1, KIF2A, TMOD4) ───────────────────────
ax2b = fig2.add_subplot(gs2[0, 1])

GO_SHORT_MAP = {
    'Sarcomere': 'Sarcomere', 'Mitochondrion org': 'Mito.org',
    'Motor activity': 'Motor', 'Autophagy': 'Autophagy',
    'Muscle cell diff': 'Mus.diff', 'Synaptic transmission': 'Synaptic',
    'MT cytoskeleton org': 'MT.cyto', 'Muscle contraction': 'Mus.cont',
    'MT-based movement': 'MT.mov', 'Neuron projection': 'Neur.proj',
    'Neuron diff': 'Neur.diff', 'Muscle organ dev': 'Mus.org',
}

cases = [
    ('TPM1',   'ENST00000360005', 'ENST00000449372', 'Sarcomere',        'Mito.org'),
    ('KIF2A',  'ENST00000676271', 'ENST00000676413', 'Motor',            'Autophagy'),
    ('TMOD4',  'ENST00000380011', 'ENST00000585680', 'Mus.diff',         'Motor'),
    ('DMD',    'ENST00000357033', 'ENST00000360679', 'Mus.diff',         'Synaptic'),
]

# Get actual scores from reversals
reversal_data = {r['gene']: r for r in top20}

bar_width = 0.35
x_case = np.arange(len(cases))
offset = 0.22

case_names = [c[0] for c in cases]
gapA_case = [reversal_data.get(g, {}).get('gap_A', 0) for g, *_ in cases]
gapB_case = [reversal_data.get(g, {}).get('gap_B', 0) for g, *_ in cases]
nameA_case = [reversal_data.get(g, {}).get('name_A', '')[:12] for g, *_ in cases]
nameB_case = [reversal_data.get(g, {}).get('name_B', '')[:12] for g, *_ in cases]

bars_a = ax2b.bar(x_case - offset/2, gapA_case, bar_width*0.85,
                  color=C_MUSCLE, alpha=0.85, label='Isoform A wins')
bars_b = ax2b.bar(x_case + offset/2, gapB_case, bar_width*0.85,
                  color=C_MED, alpha=0.85, label='Isoform B wins')

ax2b.axhline(0.2, color='gray', ls='--', lw=0.8, alpha=0.6, label='Min gap (0.2)')
ax2b.set_xticks(x_case)
ax2b.set_xticklabels(case_names, fontsize=8.5, fontweight='bold')
ax2b.set_ylabel('Score gap')
ax2b.set_title('B. Case study profiles\n(selected cross-GO reversals)', pad=6)
ax2b.legend(fontsize=7.5)
ax2b.set_ylim(0, 1.1)

# Add GO term labels above bars
for i, (na, nb) in enumerate(zip(nameA_case, nameB_case)):
    ax2b.text(i - offset/2, gapA_case[i] + 0.03, na[:10], ha='center',
              va='bottom', fontsize=6, color=C_MUSCLE, rotation=70)
    ax2b.text(i + offset/2, gapB_case[i] + 0.03, nb[:10], ha='center',
              va='bottom', fontsize=6, color=C_MED, rotation=70)

ax2b.set_axisbelow(True)
ax2b.grid(axis='y', alpha=0.3)

# ── Panel C: Summary statistics ───────────────────────────────────────────────
ax2c = fig2.add_subplot(gs2[0, 2])

# Cross-GO reversal statistics
stats_labels = [
    '123 genes\nwith reversal',
    '807 reversal\npairs total',
    '5 dual-category\n(muscle + neuro)',
    'TPM1 #1\n(total=1.204)',
    'DMD found\n(ALS/Duchenne)',
]
stats_vals   = [123, 807, 5, 1.204, 1.030]
stats_colors = [C_MUSCLE, C_MUSCLE, C_MED, '#e74c3c', '#e74c3c']

# Make a summary text-based panel
ax2c.axis('off')
ax2c.text(0.5, 0.97, 'C. Summary Statistics', ha='center', va='top',
          fontsize=9, fontweight='bold', transform=ax2c.transAxes)

rows = [
    ('Total genes analyzed', '8,402', C_DARK),
    ('Genes with ≥1 reversal', '123 (1.5%)', C_MUSCLE),
    ('Total reversal pairs', '807', C_MUSCLE),
    ('Reversal gap threshold', '≥ 0.20', C_GRAY),
    ('', '', C_GRAY),
    ('Top reversal gene', 'TPM1 (1.204)', '#e74c3c'),
    ('Top cross-tissue', 'DMD (1.030)', '#e74c3c'),
    ('Motor ↔ Neuro', 'KIF2A, KIF13A', C_MED),
    ('Striated ↔ Neuro', 'CAMK2D, MARK1', C_MED),
    ('', '', C_GRAY),
    ('Macro AUPRC (18 GO)', '0.702', C_MUSCLE),
    ('Muscle-only (13 GO)', '0.700', C_MUSCLE),
    ('Neuro terms (5 GO)', '0.709', C_MED),
    ('', '', C_GRAY),
    ('DTU-concordant switches', '905 / 3622', C_GREEN),
    ('go_type A terms', '7 / 18', C_GREEN),
    ('go_type B terms', '11 / 18', C_MUSCLE),
]

y_start = 0.90
for label, val, color in rows:
    if not label:
        y_start -= 0.025
        continue
    ax2c.text(0.05, y_start, label, ha='left', va='top',
              fontsize=8, color=C_DARK, transform=ax2c.transAxes)
    ax2c.text(0.95, y_start, val, ha='right', va='top',
              fontsize=8, fontweight='bold', color=color, transform=ax2c.transAxes)
    ax2c.plot([0.02, 0.98], [y_start - 0.005, y_start - 0.005],
              transform=ax2c.transAxes, color='#eee', lw=0.5, clip_on=False)
    y_start -= 0.052

fig2.tight_layout()
path2 = f'{FIG_DIR}/fig2_functional_switch_landscape.png'
fig2.savefig(path2, dpi=200, bbox_inches='tight')
print(f"[Saved] {path2}")
plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: KIF1B Cross-Tissue Analysis
# ─────────────────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))
fig3.suptitle('Figure 3: KIF1B Cross-Tissue Isoform Function Analysis',
              fontsize=11, fontweight='bold', y=1.02)

GO_NAMES_18 = [
    'Ca2+ sig.', 'Sarcomere', 'Mus. cont.', 'Autophagy', 'Proteasome',
    'SKM dev', 'Mus. diff', 'Ca2+ hom.', 'Mito.org', 'Mus.org',
    'TOR sig.', 'Motor act.', 'Glycolysis', 'Synaptic', 'MT-mov.',
    'Neur.proj', 'Neur.diff', 'MT.cyto',
]
GO_IDS_18 = list(kif['auprc_per_go'].keys())

alpha_scores = list(kif['kif1b_profiles']['ENST00000377093']['scores'].values())
beta_scores  = list(kif['kif1b_profiles']['ENST00000377086']['scores'].values())

# Brain-biased isoforms from v15d KIF1B profile
# From v15d JSON target_gene_profiles, isoforms sorted by max_score
kif1b_v15 = v15d['target_gene_profiles']['KIF1B']['isoforms']
# Deduplicate (JSON has duplicates)
seen_enst = set()
kif1b_unique = []
for iso in kif1b_v15:
    if iso['enst'] not in seen_enst:
        seen_enst.add(iso['enst'])
        kif1b_unique.append(iso)

# Get brain-biased isoform (highest Neuron projection score)
brain_biased = max(kif1b_unique, key=lambda x: x['scores'].get('GO:0043005', 0))
brain_biased_scores = [brain_biased['scores'].get(g, 0) for g in GO_IDS_18]
brain_biased_enst = brain_biased['enst']

# ── Panel A: GO profile bar chart ────────────────────────────────────────────
ax3a = axes3[0]

x = np.arange(len(GO_NAMES_18))
w = 0.28

bars_a = ax3a.bar(x - w, alpha_scores, w, label=f'KIF1Bα (SKM 57%)', color=C_MUSCLE, alpha=0.85)
bars_bb = ax3a.bar(x,     brain_biased_scores, w,
                   label=f'Brain-biased\n({brain_biased_enst[:14]}..)', color=C_MED, alpha=0.85)
bars_b = ax3a.bar(x + w, beta_scores, w, label='KIF1Bβ (Brain 74%)', color=C_BRAIN, alpha=0.85)

ax3a.axhline(0.20, color='gray', ls='--', lw=0.8, alpha=0.5, label='Gap threshold')
ax3a.set_xticks(x)
ax3a.set_xticklabels(GO_NAMES_18, rotation=45, ha='right', fontsize=6.5)
ax3a.set_ylabel('GO term score')
ax3a.set_title('A. KIF1B isoform GO profiles\n(muscle-trained model)', pad=6)
ax3a.legend(fontsize=6.5, loc='upper right')
ax3a.set_ylim(0, 1.0)
ax3a.set_axisbelow(True)
ax3a.grid(axis='y', alpha=0.3)

# ── Panel B: Tissue expression comparison ────────────────────────────────────
ax3b = axes3[1]

tissues_alpha = {
    'Skeletal\nmuscle': kif['kif1b_profiles']['ENST00000377093']['hpa_skm'],
    'Brain\n(avg)': kif['kif1b_profiles']['ENST00000377093']['hpa_brain'],
}
tissues_beta = {
    'Skeletal\nmuscle': kif['kif1b_profiles']['ENST00000377086']['hpa_skm'],
    'Brain\n(avg)': kif['kif1b_profiles']['ENST00000377086']['hpa_brain'],
}

# TPM bar chart
tiss_labels = list(tissues_alpha.keys())
alpha_tpm = list(tissues_alpha.values())
beta_tpm  = list(tissues_beta.values())

x_tiss = np.arange(len(tiss_labels))
w_tiss = 0.35
ax3b.bar(x_tiss - w_tiss/2, alpha_tpm, w_tiss, label='KIF1Bα', color=C_MUSCLE, alpha=0.85)
ax3b.bar(x_tiss + w_tiss/2, beta_tpm,  w_tiss, label='KIF1Bβ', color=C_BRAIN,  alpha=0.85)

ax3b.set_xticks(x_tiss)
ax3b.set_xticklabels(tiss_labels, fontsize=9)
ax3b.set_ylabel('HPA expression (TPM)')
ax3b.set_title('B. Tissue expression (HPA)\nKIF1Bα vs KIF1Bβ', pad=6)
ax3b.legend(fontsize=8)
ax3b.set_axisbelow(True)
ax3b.grid(axis='y', alpha=0.3)

# Add text annotations
ax3b.text(0 - w_tiss/2, alpha_tpm[0] + 1, f'{alpha_tpm[0]:.1f}',
          ha='center', fontsize=7.5, color=C_MUSCLE)
ax3b.text(0 + w_tiss/2, beta_tpm[0] + 0.5, f'{beta_tpm[0]:.1f}',
          ha='center', fontsize=7.5, color=C_BRAIN)
ax3b.text(1 - w_tiss/2, alpha_tpm[1] + 0.5, f'{alpha_tpm[1]:.1f}',
          ha='center', fontsize=7.5, color=C_MUSCLE)
ax3b.text(1 + w_tiss/2, beta_tpm[1] + 1, f'{beta_tpm[1]:.1f}',
          ha='center', fontsize=7.5, color=C_BRAIN)

# ── Panel C: Key insight summary (muscle model limitation) ───────────────────
ax3c = axes3[2]

# Selected GO terms for α vs β comparison
sel_gos = ['Motor act.', 'MT-mov.', 'MT.cyto', 'Synaptic', 'Neur.proj', 'Neur.diff']
sel_ids = ['GO:0003774', 'GO:0007018', 'GO:0000226', 'GO:0007268', 'GO:0043005', 'GO:0030182']
alpha_sel = [kif['kif1b_profiles']['ENST00000377093']['scores'].get(g, 0) for g in sel_ids]
beta_sel  = [kif['kif1b_profiles']['ENST00000377086']['scores'].get(g, 0) for g in sel_ids]
bb_sel    = [brain_biased['scores'].get(g, 0) for g in sel_ids]

x_sel = np.arange(len(sel_gos))
w_sel = 0.26

ax3c.bar(x_sel - w_sel, alpha_sel, w_sel, label='KIF1Bα (muscle)', color=C_MUSCLE, alpha=0.85)
ax3c.bar(x_sel,         bb_sel,   w_sel, label='Brain-biased (in muscle data)', color=C_MED, alpha=0.85)
ax3c.bar(x_sel + w_sel, beta_sel,  w_sel, label='KIF1Bβ (brain, injected)', color=C_BRAIN, alpha=0.85)

ax3c.set_xticks(x_sel)
ax3c.set_xticklabels(sel_gos, rotation=30, ha='right', fontsize=8)
ax3c.set_ylabel('GO term score')
ax3c.set_title('C. Key GO term comparison\n(muscle model → brain isoform blind spot)', pad=6)
ax3c.legend(fontsize=7, loc='upper right')
ax3c.set_ylim(0, 1.0)

# Annotation: model blind spot
ax3c.annotate('Model blind spot:\nKIF1Bβ near-zero\n(not in muscle training)',
              xy=(5 + w_sel, 0.04), xytext=(4.5, 0.45),
              arrowprops=dict(arrowstyle='->', color=C_BRAIN, lw=1.5),
              fontsize=7, color=C_BRAIN, ha='center',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=C_BRAIN, alpha=0.8))

ax3c.set_axisbelow(True)
ax3c.grid(axis='y', alpha=0.3)

fig3.tight_layout()
path3 = f'{FIG_DIR}/fig3_kif1b_cross_tissue.png'
fig3.savefig(path3, dpi=200, bbox_inches='tight')
print(f"[Saved] {path3}")
plt.close(fig3)

print(f"\nAll figures saved to: {FIG_DIR}")
print("\nFigure summary:")
print("  Fig 1: Model performance (AUPRC + diversity + AUPRC vs diversity scatter)")
print("  Fig 2: Functional switch landscape (top-20 reversals + case studies)")
print("  Fig 3: KIF1B cross-tissue analysis (GO profiles + expression + blind spot)")
print("\nALL DONE")
