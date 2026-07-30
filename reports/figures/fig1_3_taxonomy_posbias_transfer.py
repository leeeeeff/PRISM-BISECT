"""
Figures 1-3 for PRISM/BISECT manuscript
Fig1: Three-Case Taxonomy (sep_cosine vs delta AUPRC)
Fig2: pos_bias distribution vs noise floor
Fig3: Cross-tissue zero-shot transfer
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'pdf.fonttype': 42,
})
PALETTE = {'A': '#E8834A', 'B-high': '#4C72B0', 'B-mid': '#64B5CD', 'B-low': '#A8D5E2'}

# ─────────────────────────────────────────────────────────────────────────────
# DATA: from manuscript tables & JSON outputs
# ─────────────────────────────────────────────────────────────────────────────

# Fig 1 – Three-Case Taxonomy data (from typeAB JSON + sarcopenia_final JSON)
taxonomy_data = pd.DataFrame([
    # go_id, name_short, type, sep_cosine, lr_auprc, prism_auprc, delta, case
    ('GO:0006096', 'Glycolysis',         'A', 0.737, 0.695, 0.671, -0.023, 'Case 1'),
    ('GO:0003774', 'Motor activity',      'A', 0.167, 0.825, 0.813, -0.013, 'Case 1'),
    ('GO:0007519', 'Skeletal muscle dev', 'B', 0.018, 0.587, 0.725,  0.138, 'Case 2'),
    ('GO:0030017', 'Sarcomere org',       'B', 0.044, 0.564, 0.743,  0.179, 'Case 2'),
    ('GO:0032006', 'TOR signaling',       'B', 0.040, 0.510, 0.602,  0.092, 'Case 2'),
    ('GO:0006914', 'Autophagy',           'B', 0.031, 0.285, 0.640,  0.354, 'Case 3'),
    ('GO:0043161', 'Proteasome-UPS',      'B', 0.037, 0.362, 0.717,  0.356, 'Case 3'),
    ('GO:0042692', 'Muscle cell diff',    'B', 0.024, 0.232, 0.653,  0.421, 'Case 3'),
    ('GO:0055074', 'Ca²⁺ homeostasis',   'B', 0.025, 0.251, 0.726,  0.475, 'Case 3'),
    ('GO:0007005', 'Mitochondrion org',   'B', 0.025, 0.238, 0.662,  0.424, 'Case 3'),
    ('GO:0007517', 'Muscle organ dev',    'B', 0.019, 0.237, 0.702,  0.465, 'Case 3'),
    ('GO:0006941', 'Muscle contraction',  'B', 0.037, 0.310, 0.597,  0.287, 'Case 3'),
    ('GO:0007204', 'Ca²⁺ signaling',     'B', 0.056, 0.415, 0.765,  0.350, 'Case 3'),
], columns=['go','name','type','sep_cosine','lr_auprc','prism_auprc','delta','case'])

# Fig 2 – pos_bias per GO term (Table 2 from manuscript)
posbias_data = pd.DataFrame([
    ('Muscle contraction',  'B', 1.902, 0.895, 1.592, '***'),
    ('Skeletal muscle dev', 'B', 1.778, 0.873, 1.955, '***'),
    ('Motor activity',      'A', 1.435, 0.693, 1.582, '***'),
    ('Sarcomere org',       'B', 1.176, 0.730, 1.224, '***'),
    ('Proteasome-UPS',      'B', 0.957, 0.549, 0.796, '***'),
    ('Muscle cell diff',    'B', 0.824, 0.654, 1.045, '***'),
    ('Muscle organ dev',    'B', 0.805, 0.834, 1.239, '***'),
    ('Mitochondrion org',   'B', 0.879, 0.502, 0.731, '***'),
    ('Ca²⁺ homeostasis',   'B', 0.764, 0.414, 0.700, '***'),
    ('Autophagy',           'B', 0.724, 0.409, 0.748, '***'),
    ('TOR signaling',       'B', 0.699, 0.241, 0.625, '*'),
    ('Glycolysis',          'A', 0.663, 0.019, 1.172, 'n.s.'),
    ('Ca²⁺ signaling',     'B', 0.475, 0.130, 0.465, 'n.s.'),
], columns=['name','type','pos_bias','ci_lo','ci_hi','sig'])
posbias_data = posbias_data.sort_values('pos_bias', ascending=False)

# Fig 3 – Cross-tissue transfer (exp_A_results.json)
transfer_data = pd.DataFrame([
    # name, esm2, prism18, related
    ('Neuron proj dev',          0.063, 0.567, True),
    ('Neuron differentiation',   0.082, 0.529, True),
    ('Axon development',         0.038, 0.398, True),
    ('Ca²⁺ homeostasis',        0.042, 0.447, True),
    ('Learning or memory',       0.021, 0.140, True),
    ('Neuron development',       0.072, 0.497, True),
    ('Reg. Ca²⁺ transmembrane', 0.009, 0.130, True),
    ('Neuropeptide signaling',   0.103, 0.036, False),
    ('K⁺ ion transport',        0.054, 0.018, False),
    ('K⁺ transmembrane transp', 0.029, 0.023, False),
    ('Immune response',          0.042, 0.027, False),
    ('GPCR signaling (Gi)',      0.091, 0.085, False),
    ('Receptor endocytosis',     0.047, 0.022, False),
    ('Cell recognition',         0.037, 0.031, False),
], columns=['name','esm2_640','prism_18','related'])


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Three-Case Taxonomy
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5.5))
fig1.patch.set_facecolor('#FAFAFA')

case_colors = {'Case 1': '#E8834A', 'Case 2': '#64B5CD', 'Case 3': '#2A5FA5'}
case_markers = {'Case 1': 'D', 'Case 2': 's', 'Case 3': 'o'}
case_sizes = {'Case 1': 110, 'Case 2': 110, 'Case 3': 110}

ax = axes1[0]
ax.set_facecolor('#F7F9FC')
for _, row in taxonomy_data.iterrows():
    ax.scatter(row['sep_cosine'], row['delta'],
               color=case_colors[row['case']],
               marker=case_markers[row['case']],
               s=case_sizes[row['case']],
               edgecolors='white', linewidth=1.2, zorder=5, alpha=0.92)
    if row['name'] in ['Glycolysis', 'Motor activity', 'Ca²⁺ homeostasis',
                        'Muscle cell diff', 'Autophagy']:
        ax.annotate(row['name'], (row['sep_cosine'], row['delta']),
                    fontsize=7.5, ha='left', va='bottom',
                    xytext=(4, 3), textcoords='offset points', color='#333')

# threshold line
ax.axvline(x=0.060, color='#C0392B', linewidth=1.5, linestyle='--', alpha=0.8, zorder=4)
ax.text(0.062, 0.47, 'τ = 0.060\n(LOOCV 13/13)', fontsize=8, color='#C0392B', va='top')
ax.axhline(y=0, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)

# regression for Case 3 only
c3 = taxonomy_data[taxonomy_data['case']=='Case 3']
z = np.polyfit(c3['sep_cosine'], c3['delta'], 1)
xline = np.linspace(c3['sep_cosine'].min()-0.005, c3['sep_cosine'].max()+0.005, 50)
ax.plot(xline, np.polyval(z, xline), color='#2A5FA5', linewidth=1.4,
        linestyle='-', alpha=0.5, zorder=3)

ax.set_xlabel('Positive-class cosine separability (sep_cosine)', fontsize=10)
ax.set_ylabel('ΔAUPRC (PRISM − LR)', fontsize=10)
ax.set_title('A   Embedding geometry predicts model advantage', fontsize=11, fontweight='bold', loc='left')
ax.set_xlim(-0.02, 0.80)
ax.set_ylim(-0.10, 0.54)

patches = [mpatches.Patch(color=case_colors[c], label=c) for c in ['Case 1','Case 2','Case 3']]
ax.legend(handles=patches, fontsize=8.5, frameon=False, loc='upper right')

# Panel B: LR AUPRC vs delta (alternative view)
ax2 = axes1[1]
ax2.set_facecolor('#F7F9FC')
for _, row in taxonomy_data.iterrows():
    ax2.scatter(row['lr_auprc'], row['prism_auprc'],
                color=case_colors[row['case']],
                marker=case_markers[row['case']],
                s=110, edgecolors='white', linewidth=1.2, zorder=5, alpha=0.92)

# diagonal reference
lims = [0.2, 0.9]
ax2.plot(lims, lims, 'k--', linewidth=1, alpha=0.4, zorder=2, label='PRISM = LR')
ax2.fill_between(lims, lims, [0.9, 0.9], alpha=0.05, color='#2A5FA5')
ax2.text(0.22, 0.86, 'PRISM advantage zone', fontsize=8, color='#2A5FA5', alpha=0.7)

ax2.set_xlabel('Logistic Regression AUPRC', fontsize=10)
ax2.set_ylabel('PRISM AUPRC', fontsize=10)
ax2.set_title('B   PRISM vs. LR per GO term', fontsize=11, fontweight='bold', loc='left')
ax2.set_xlim(0.18, 0.88)
ax2.set_ylim(0.50, 0.90)
ax2.legend(fontsize=8.5, frameon=False)

for ax in axes1:
    ax.tick_params(labelsize=9)

fig1.suptitle('Figure 1 | Data geometry determines PRISM utility: Three-Case Taxonomy',
              fontsize=12, fontweight='bold', y=1.01)
fig1.tight_layout()
out1 = '/home/welcome1/sw1686/DIFFUSE/reports/figures/fig1_three_case_taxonomy.png'
fig1.savefig(out1, dpi=200, bbox_inches='tight', facecolor=fig1.get_facecolor())
fig1.savefig(out1.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {out1}")
plt.close(fig1)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: pos_bias with noise floor
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(12, 5.5))
fig2.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#F7F9FC')

bar_colors = []
for _, row in posbias_data.iterrows():
    if row['type'] == 'A':
        bar_colors.append('#E8834A')
    elif row['sig'] == '***':
        bar_colors.append('#2A5FA5')
    elif row['sig'] == '*':
        bar_colors.append('#5B8DB8')
    else:
        bar_colors.append('#A8C8DC')

x = np.arange(len(posbias_data))
bars = ax.bar(x, posbias_data['pos_bias'], color=bar_colors,
              edgecolor='white', linewidth=0.8, zorder=4, width=0.65)

# CI error bars
ci_lo_err = posbias_data['pos_bias'] - posbias_data['ci_lo']
ci_hi_err = posbias_data['ci_hi'] - posbias_data['pos_bias']
ax.errorbar(x, posbias_data['pos_bias'],
            yerr=[ci_lo_err.clip(0), ci_hi_err.clip(0)],
            fmt='none', color='#333', linewidth=1.2, capsize=3, zorder=6)

# Reference lines
ax.axhline(y=0.240, color='#E74C3C', linewidth=2, linestyle='--', zorder=5,
           label='Shuffled-label null (0.240 ± 0.048)')
ax.axhline(y=0.898, color='#27AE60', linewidth=2, linestyle='--', zorder=5,
           label='Random ceiling (0.898 ± 0.041)')
ax.fill_between([-0.5, len(posbias_data)-0.5], 0.192, 0.288,
                color='#E74C3C', alpha=0.08, zorder=2)
ax.fill_between([-0.5, len(posbias_data)-0.5], 0.857, 0.939,
                color='#27AE60', alpha=0.06, zorder=2)

# XGBoost reference point
ax.axhline(y=0.027, color='#F39C12', linewidth=1.5, linestyle='-.', zorder=5,
           label='XGBoost (gene-level memorization, 0.027)')

# significance stars
for i, (_, row) in enumerate(posbias_data.iterrows()):
    if row['sig'] != 'n.s.':
        ax.text(i, row['pos_bias'] + 0.07, row['sig'],
                ha='center', va='bottom', fontsize=8.5, color='#222', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(posbias_data['name'], rotation=38, ha='right', fontsize=9)
ax.set_ylabel('pos_bias (within-gene isoform discrimination score)', fontsize=10)
ax.set_title('Figure 2 | PRISM achieves isoform-level discrimination exceeding gene-memorization null\n'
             '(XGBoost pos_bias = 0.027; PRISM macro = 1.006)',
             fontsize=11, fontweight='bold', loc='left')
ax.set_ylim(-0.05, 2.3)
ax.set_xlim(-0.5, len(posbias_data)-0.5)

legend_patches = [
    mpatches.Patch(color='#2A5FA5', label='Type-B (q < 0.001)'),
    mpatches.Patch(color='#5B8DB8', label='Type-B (q < 0.05)'),
    mpatches.Patch(color='#A8C8DC', label='Type-B (n.s.)'),
    mpatches.Patch(color='#E8834A', label='Type-A'),
]
h, l = ax.get_legend_handles_labels()
ax.legend(handles=legend_patches + h, fontsize=8.5, frameon=False,
          loc='upper right', ncol=2)

ax.tick_params(labelsize=9)
fig2.tight_layout()
out2 = '/home/welcome1/sw1686/DIFFUSE/reports/figures/fig2_posbias_discrimination.png'
fig2.savefig(out2, dpi=200, bbox_inches='tight', facecolor=fig2.get_facecolor())
fig2.savefig(out2.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {out2}")
plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Cross-tissue zero-shot transfer
# ─────────────────────────────────────────────────────────────────────────────
# Sort: related first (descending prism), then unrelated (ascending)
related = transfer_data[transfer_data['related']].sort_values('prism_18', ascending=False)
unrelated = transfer_data[~transfer_data['related']].sort_values('prism_18', ascending=False)
df3 = pd.concat([related, unrelated]).reset_index(drop=True)

fig3, ax = plt.subplots(figsize=(14, 5.5))
fig3.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#F7F9FC')

x = np.arange(len(df3))
w = 0.35
b1 = ax.bar(x - w/2, df3['esm2_640'], w, label='ESM-2 640-dim (raw)', 
            color='#A0B9D0', edgecolor='white', linewidth=0.8, zorder=4)
b2 = ax.bar(x + w/2, df3['prism_18'], w, label='PRISM-18 (muscle-trained)', 
            color='#2A5FA5', edgecolor='white', linewidth=0.8, zorder=4)

# improvement annotations for "related" group
for i, (_, row) in enumerate(df3.iterrows()):
    if row['related'] and row['prism_18'] > row['esm2_640'] * 1.5:
        ratio = row['prism_18'] / max(row['esm2_640'], 0.001)
        ax.annotate(f'{ratio:.1f}×', xy=(i + w/2, row['prism_18']),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=8, fontweight='bold', color='#1A3F6E')

# divider line between related/unrelated
divider_x = len(related) - 0.5
ax.axvline(x=divider_x, color='#888', linewidth=1.5, linestyle=':', zorder=6)
ax.text(divider_x - 0.3, 0.58, 'Functionally\nRelated', ha='right',
        fontsize=9, color='#2A5FA5', fontweight='bold')
ax.text(divider_x + 0.3, 0.58, 'Functionally\nUnrelated', ha='left',
        fontsize=9, color='#888', fontweight='bold')

# shading
ax.axvspan(-0.5, divider_x, alpha=0.05, color='#2A5FA5', zorder=1)
ax.axvspan(divider_x, len(df3)-0.5, alpha=0.04, color='gray', zorder=1)

ax.set_xticks(x)
ax.set_xticklabels(df3['name'], rotation=38, ha='right', fontsize=8.5)
ax.set_ylabel('AUPRC (zero-shot brain evaluation)', fontsize=10)
ax.set_title('Figure 3 | Cross-tissue transfer is selective: functionally related GO terms transfer up to 10×\n'
             '(muscle-trained PRISM-18 vs. raw ESM-2-640 on 63,994 brain isoforms)',
             fontsize=11, fontweight='bold', loc='left')
ax.set_ylim(0, 0.66)
ax.set_xlim(-0.5, len(df3)-0.5)
ax.legend(fontsize=9.5, frameon=False, loc='upper right')
ax.tick_params(labelsize=9)

fig3.tight_layout()
out3 = '/home/welcome1/sw1686/DIFFUSE/reports/figures/fig3_cross_tissue_transfer.png'
fig3.savefig(out3, dpi=200, bbox_inches='tight', facecolor=fig3.get_facecolor())
fig3.savefig(out3.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {out3}")
plt.close(fig3)

print("\n✓ Figures 1–3 complete.")
