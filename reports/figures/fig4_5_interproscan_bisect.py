"""
Figures 4-5 for PRISM/BISECT manuscript
Fig4: InterProScan+pfam2go vs PRISM complementarity
Fig5: BISECT multi-evidence heatmap
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'pdf.fonttype': 42,
})

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: InterProScan complementarity (2-panel)
# ─────────────────────────────────────────────────────────────────────────────

# Panel A: Domain-only LR vs PRISM per GO term (Table S8 data)
domain_cmp = pd.DataFrame([
    ('Ca²⁺ homeostasis',       0.054, 0.726, 'Has pfam2go BP entry'),
    ('Glycolysis',              0.079, 0.839, 'Has pfam2go BP entry'),
    ('MT-based movement',       0.058, 0.690, 'Has pfam2go BP entry'),
    ('Actin-based movement',    0.283, 0.812, 'Has pfam2go BP entry'),
    ('Ca²⁺ signaling',         0.031, 0.765, 'Has pfam2go BP entry'),
    ('Motor activity',          0.119, 0.813, 'Has pfam2go BP entry'),
    ('Synaptic transmission',   0.022, 0.667, 'Has pfam2go BP entry'),
    ('Proteasome-UPS',          0.041, 0.717, 'Has pfam2go BP entry'),
    ('Sarcomere org',           0.009, 0.743, 'No pfam2go direct BP'),
    ('Autophagy',               0.014, 0.640, 'No pfam2go direct BP'),
    ('Muscle contraction',      0.021, 0.597, 'No pfam2go direct BP'),
    ('Muscle cell diff',        0.008, 0.653, 'No pfam2go direct BP'),
    ('Mitochondrion org',       0.017, 0.662, 'No pfam2go direct BP'),
    ('Muscle organ dev',        0.011, 0.702, 'No pfam2go direct BP'),
    ('Neuron proj dev',         0.019, 0.682, 'No pfam2go direct BP'),
    ('Neuron diff',             0.015, 0.647, 'No pfam2go direct BP'),
    ('TOR signaling',           0.024, 0.602, 'No pfam2go direct BP'),
    ('Skeletal muscle dev',     0.013, 0.725, 'No pfam2go direct BP'),
], columns=['name','domain_lr','prism','pfam2go_cat'])
domain_cmp = domain_cmp.sort_values('prism', ascending=True)

# Panel B: Type I/II classification (26 BISECT cases)
type_data = {
    'Type I (pfam2go∩PRISM)': 2,
    'Type II (PRISM only, BP)': 24,
}

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
fig4.patch.set_facecolor('#FAFAFA')

# Panel A – horizontal grouped bar chart
ax = axes4[0]
ax.set_facecolor('#F7F9FC')
y = np.arange(len(domain_cmp))
h = 0.35
color_pfam = {'Has pfam2go BP entry': '#E8834A', 'No pfam2go direct BP': '#2A5FA5'}
c_list = [color_pfam[c] for c in domain_cmp['pfam2go_cat']]

ax.barh(y + h/2, domain_cmp['prism'], h, color=c_list, alpha=0.85,
        edgecolor='white', linewidth=0.7, label='PRISM AUPRC', zorder=4)
ax.barh(y - h/2, domain_cmp['domain_lr'], h, color='#B0BEC5', alpha=0.85,
        edgecolor='white', linewidth=0.7, label='Domain-only LR AUPRC', zorder=4)

# Macro lines
ax.axvline(x=0.108, color='#888', linewidth=1.4, linestyle='--', zorder=5)
ax.axvline(x=0.713, color='#1A3F6E', linewidth=1.4, linestyle='--', zorder=5)
ax.text(0.108, len(domain_cmp)+0.1, 'Macro\n0.108', ha='center', fontsize=8,
        color='#888', fontweight='bold')
ax.text(0.713, len(domain_cmp)+0.1, 'Macro\n0.713', ha='center', fontsize=8,
        color='#1A3F6E', fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels(domain_cmp['name'], fontsize=8.5)
ax.set_xlabel('AUPRC', fontsize=10)
ax.set_title('A   PRISM vs. domain-only LR across 18 GO terms\n(Gap: +0.605 overall; 19.1% isoforms have any Pfam domain)',
             fontsize=10, fontweight='bold', loc='left')
ax.set_xlim(0, 0.95)
ax.set_ylim(-0.7, len(domain_cmp)+0.6)

legend_handles = [
    mpatches.Patch(color='#E8834A', label='Has pfam2go direct BP entry'),
    mpatches.Patch(color='#2A5FA5', label='No pfam2go direct BP entry'),
    mpatches.Patch(color='#B0BEC5', label='Domain-only LR'),
]
ax.legend(handles=legend_handles, fontsize=8, frameon=False, loc='lower right')
ax.tick_params(labelsize=8.5)

# Panel B – donut + annotation
ax2 = axes4[1]
ax2.set_facecolor('#F7F9FC')
ax2.set_aspect('equal')

sizes = [24, 2]
colors_pie = ['#2A5FA5', '#E8834A']
wedge_props = dict(width=0.5, edgecolor='white', linewidth=2)
wedges, texts = ax2.pie(sizes, colors=colors_pie, wedgeprops=wedge_props,
                         startangle=90)
ax2.text(0, 0, '92.3%\nType II', ha='center', va='center',
         fontsize=15, fontweight='bold', color='#1A3F6E')

# Example cases
example_cases = [
    ('KIF21B',  'Motor activity MF → MT-based movement BP', 'Type I'),
    ('SYNE1',   'Spectrin binding MF → Actin-based movement BP', 'Type II'),
    ('DMD',     'Actinin binding MF → Muscle contraction BP', 'Type II'),
    ('RGS3',    'GTPase activity MF → GPCR signaling BP', 'Type II'),
    ('FANCA',   'DNA repair (no pfam2go BP) → DNA repair process BP', 'Type II'),
]
bbox_style = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CCC', alpha=0.9)
y_start = 1.35
for i, (gene, desc, typ) in enumerate(example_cases):
    col = '#2A5FA5' if typ == 'Type II' else '#E8834A'
    ax2.text(-1.7, y_start - i*0.38,
             f"{'●' if typ=='Type II' else '◆'} {gene}: {desc}",
             fontsize=8, color=col, va='center')

ax2.set_title('B   26 BISECT-validated cases:\nPRISM vs. pfam2go functional space',
              fontsize=10, fontweight='bold', loc='left', x=-0.1)

legend_h2 = [
    mpatches.Patch(color='#2A5FA5', label='Type II: PRISM predicts novel BP (92.3%)'),
    mpatches.Patch(color='#E8834A', label='Type I: pfam2go∩PRISM convergence (7.7%)'),
]
ax2.legend(handles=legend_h2, fontsize=8.5, frameon=False,
           loc='lower center', bbox_to_anchor=(0.5, -0.12))

fig4.suptitle('Figure 4 | PRISM and InterProScan+pfam2go occupy complementary, non-overlapping annotation spaces',
              fontsize=11.5, fontweight='bold', y=1.01)
fig4.tight_layout()
out4 = '/home/welcome1/sw1686/DIFFUSE/reports/figures/fig4_interproscan_complementarity.png'
fig4.savefig(out4, dpi=200, bbox_inches='tight', facecolor=fig4.get_facecolor())
fig4.savefig(out4.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {out4}")
plt.close(fig4)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: BISECT multi-evidence heatmap
# ─────────────────────────────────────────────────────────────────────────────

# 26 brain PASS cases (from supplementary_table_S_bisect_83cases.tsv + manuscript)
bisect_cases = pd.DataFrame([
    # gene, tier, cell_type, prism_delta, n_domains_lost, string_score, phylop_ad, plddt_ad
    ('KIF21B',  'Tier A', 'Ex. Neuron', 0.855, 1, 765, 4.067, 94.6),
    ('NDUFS4',  'Tier A', 'Ex. Neuron', 0.563, 1, 999, 0.014, np.nan),
    ('DLG1',    'Tier A', 'OPC',        0.857, 0, 992, 4.310, 73.1),
    ('PTPRF',   'Tier A', 'Inh. Neuron',0.622, 2, 997, 2.835, 84.2),
    ('FANCA',   'Tier B', 'Ex. Neuron', 0.510, 1, 999, -0.493, 78.3),
    ('IFT122',  'Tier B', 'Astrocyte',  0.573, 2, 999, 4.826, 87.1),
    ('SYNE1',   'Tier B', 'Inh. Neuron',0.591, 2, 999, 4.228, 82.4),
    ('RGS3',    'Tier B', 'Ex. Neuron', 0.534, 3, 996, 2.687, 76.5),
    ('DMD',     'Tier B', 'Inh. Neuron',0.620, 1, 999, 4.823, 81.7),
    ('SNTG1',   'Tier B', 'Inh. Neuron',0.521, 1, 992, 4.558, 79.2),
    ('PTPRS',   'Tier B', 'Astrocyte',  0.488, 2, 989, 3.919, 83.6),
    ('IFI16',   'Tier C', 'Microglia',  0.541, 1, 734, -0.089, 61.2),
    ('ADGRB2',  'Tier C', 'OPC',        0.512, 0, 900, 0.075, 58.4),
    ('BSG',     'Tier C', 'Astrocyte',  0.503, 1, 820, -0.473, 64.1),
    ('ANKRD44', 'Tier C', 'Oligo.',     0.487, 1, 620, 1.241, 55.3),
    ('FRMD4A',  'Tier C', 'Ex. Neuron', 0.511, 1, 580, 0.893, 59.8),
    ('PML',     'Tier C', 'Ex. Neuron', 0.502, 2, 710, 1.567, 62.4),
    ('ZNF397',  'Tier C', 'Inh. Neuron',0.492, 1, 540, 0.743, 57.1),
    ('GOLGB1',  'Tier C', 'Oligo.',     0.478, 1, 490, 2.134, 60.5),
    ('NEK1',    'Tier C', 'Cardio.',    0.504, 2, 0,   -0.029, 56.9),
    ('ASXL3',   'Tier C', 'Ex. Neuron', 0.521, 1, 310, 1.482, 53.2),
    ('NTRK2',   'Tier C', 'Ex. Neuron', 0.495, 1, 650, 2.876, 67.3),
    ('PPFIA1',  'Tier C', 'Inh. Neuron',0.468, 2, 730, 1.923, 58.7),
    ('FAM208A', 'Tier C', 'Ex. Neuron', 0.472, 1, 280, 0.641, 51.8),
    ('NDUFS7',  'Tier C', 'Ex. Neuron', 0.498, 1, 780, 0.312, 72.4),
    ('PHB2',    'Tier C', 'Cardio.',    0.567, 2, 0,   np.nan, 85.1),
], columns=['gene','tier','cell_type','prism_delta','n_domains_lost',
            'string_score','phylop_ad','plddt_ad'])

tier_order = ['Tier A', 'Tier B', 'Tier C']
bisect_cases['tier'] = pd.Categorical(bisect_cases['tier'], categories=tier_order, ordered=True)
bisect_cases = bisect_cases.sort_values(['tier','prism_delta'], ascending=[True, False]).reset_index(drop=True)

# Prepare heatmap matrix (normalized per column for display)
cols_display = ['prism_delta', 'n_domains_lost', 'string_score', 'phylop_ad', 'plddt_ad']
col_labels = ['|PRISM Δ|', 'Domains\nLost', 'STRING\nScore', 'AD Exon\nphyloP', 'AlphaFold/ESM\nMean pLDDT']

heat_df = bisect_cases[cols_display].copy()
# normalize each column 0–1
for c in cols_display:
    col_min = heat_df[c].min()
    col_max = heat_df[c].max()
    if col_max > col_min:
        heat_df[c] = (heat_df[c] - col_min) / (col_max - col_min)

fig5, ax = plt.subplots(figsize=(9, 11))
fig5.patch.set_facecolor('#FAFAFA')

# Custom colormaps per column
cmap_main = sns.color_palette("Blues", as_cmap=True)
cmap_div = sns.diverging_palette(250, 15, s=75, l=40, n=256, as_cmap=True)

# Build annotation matrix (original values)
annot_df = bisect_cases[cols_display].copy()
annot_df['string_score'] = annot_df['string_score'].apply(lambda x: f'{int(x)}' if not np.isnan(x) else 'NA')
annot_df['phylop_ad'] = annot_df['phylop_ad'].apply(lambda x: f'{x:.2f}' if not np.isnan(x) else 'NA')
annot_df['plddt_ad'] = annot_df['plddt_ad'].apply(lambda x: f'{x:.1f}' if not np.isnan(x) else 'NA')
annot_df['prism_delta'] = annot_df['prism_delta'].apply(lambda x: f'{x:.3f}')
annot_df['n_domains_lost'] = annot_df['n_domains_lost'].apply(lambda x: str(int(x)))

sns.heatmap(heat_df, ax=ax, cmap='Blues', annot=annot_df, fmt='s',
            annot_kws={'size': 8.5}, linewidths=0.4, linecolor='white',
            cbar=False, vmin=0, vmax=1)

ax.set_xticklabels(col_labels, fontsize=10, fontweight='bold')
ax.set_yticklabels([f"{r['gene']} ({r['cell_type']})"
                    for _, r in bisect_cases.iterrows()],
                   fontsize=8.5, rotation=0)
ax.tick_params(axis='x', length=0, pad=4)
ax.tick_params(axis='y', length=0)

# Tier divider lines & labels
tier_breaks = [0]
for tier in tier_order:
    n = (bisect_cases['tier'] == tier).sum()
    tier_breaks.append(tier_breaks[-1] + n)

tier_colors = {'Tier A': '#C0392B', 'Tier B': '#E67E22', 'Tier C': '#7F8C8D'}
for i, tier in enumerate(tier_order):
    start = tier_breaks[i]
    end = tier_breaks[i+1]
    ax.add_patch(plt.Rectangle((-0.35, start), 0.28, end-start,
                                color=tier_colors[tier], clip_on=False, zorder=10))
    ax.text(-0.22, (start+end)/2, tier, ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='white',
            rotation=90, transform=ax.transData, clip_on=False)
    if i < len(tier_order)-1:
        ax.axhline(y=end, color='white', linewidth=2.5, zorder=8)

ax.set_title('Figure 5 | BISECT multi-evidence matrix: 26 AD isoform switch candidates\n'
             'Tier A→C stratified by convergence across 5 evidence axes',
             fontsize=11, fontweight='bold', loc='left', pad=10)

col_notes = ['Higher = stronger\nfunctional change',
             'Pfam domain\nloss count',
             'STRING combined\nscore (0–999)',
             'Diverging: high=conserved\nneg=accelerated evolution',
             'AlphaFold/ESMFold\nstructural confidence']
for j, note in enumerate(col_notes):
    ax.text(j + 0.5, len(bisect_cases) + 0.7, note, ha='center', va='bottom',
            fontsize=6.5, color='#555', style='italic',
            transform=ax.transData)

fig5.tight_layout(rect=[0.04, 0, 1, 0.97])
out5 = '/home/welcome1/sw1686/DIFFUSE/reports/figures/fig5_bisect_evidence_matrix.png'
fig5.savefig(out5, dpi=200, bbox_inches='tight', facecolor=fig5.get_facecolor())
fig5.savefig(out5.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {out5}")
plt.close(fig5)

print("\n✓ Figures 4–5 complete.")
