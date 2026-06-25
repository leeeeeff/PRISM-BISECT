"""
Visualize C18 isoform ratio analysis results.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

IN_TSV  = Path('reports/c18_deep_dive/c18_isoform_ratios.tsv')
OUT_DIR = Path('reports/c18_deep_dive')
SUM_TSV = Path('reports/c18_deep_dive/c18_isoform_ratio_summary.tsv')

df = pd.read_csv(IN_TSV, sep='\t')
df_sum = pd.read_csv(SUM_TSV, sep='\t')

# Map condition labels
COND_COLOR = {'AD': '#D62728', 'Control': '#1F77B4', 'Active control': '#9467BD'}
CLUSTER_ORDER = ['C10', 'C11', 'C18', 'C19']
CLUSTER_LABELS = {'C10': 'C10 L4-IT (canonical)', 'C11': 'C11 L4-IT (canonical)',
                  'C18': 'C18 L4-IT atypical\n(AD-enriched *)', 'C19': 'C19 L5-ET'}

GENES = ['NDUFS8', 'DCAF5', 'NDUFS7', 'ZNF736', 'RPS3', 'NDUFAF5', 'ZNF582', 'NDUFS4']
GENE_LABELS = {
    'NDUFS8':  'NDUFS8 (Complex I, A-BP)',
    'NDUFS7':  'NDUFS7 (Complex I, A-BP)',
    'NDUFS4':  'NDUFS4 (Complex I, A-BP)',
    'NDUFAF5': 'NDUFAF5 (Complex I, A-DR)',
    'DCAF5':   'DCAF5 (Ub-Prot, A-DR)',
    'ZNF736':  'ZNF736 (KRAB-ZFP, A-DR)',
    'ZNF582':  'ZNF582 (KRAB-ZFP, A-DR)',
    'RPS3':    'RPS3 (RNA Metab, A-DR)',
}

n_genes = len(GENES)
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
axes = axes.flatten()

for ax, gene in zip(axes, GENES):
    gdf = df[df['gene'] == gene]
    x_pos = 0
    xticks, xlabels = [], []
    for cluster in CLUSTER_ORDER:
        cdf = gdf[gdf['cluster'] == cluster]
        if cdf.empty:
            x_pos += 1
            continue
        for cond in ['AD', 'Control', 'Active control']:
            pts = cdf[cdf['condition'] == cond]['ad_ratio'].values
            if len(pts) == 0:
                continue
            jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(pts))
            ax.scatter([x_pos + jitter[i] for i in range(len(pts))], pts,
                       color=COND_COLOR[cond], alpha=0.7, s=30, zorder=3)
            mean_val = pts.mean()
            ax.hlines(mean_val, x_pos - 0.25, x_pos + 0.25,
                      colors=COND_COLOR[cond], linewidths=2, zorder=4)
            x_pos += 0.4
        xticks.append(x_pos - 0.6)
        xlabels.append(CLUSTER_LABELS.get(cluster, cluster))
        # shade C18
        if cluster == 'C18':
            ax.axvspan(x_pos - 1.35, x_pos - 0.1, alpha=0.08, color='red')
        x_pos += 0.2

    ax.set_title(GENE_LABELS.get(gene, gene), fontsize=8, fontweight='bold')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=6, rotation=20, ha='right')
    ax.set_ylabel('AD-isoform ratio', fontsize=7)
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Legend
patches = [mpatches.Patch(color=c, label=l) for l, c in COND_COLOR.items()]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.03))
fig.suptitle('AD-isoform ratio by cluster and condition\n(C18 L4-IT atypical = AD-enriched, MWU p=0.0099)',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0.04, 1, 1])
out = OUT_DIR / 'c18_isoform_ratio_plot.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')

# Print key findings
print('\n=== Key findings ===')
highlight_genes = ['NDUFS8', 'DCAF5']
for gene in highlight_genes:
    gsum = df_sum[df_sum['gene'] == gene]
    c18 = gsum[gsum['cluster'] == 'C18']
    c10 = gsum[gsum['cluster'] == 'C10']
    c11 = gsum[gsum['cluster'] == 'C11']
    if not c18.empty:
        c18_delta = c18['AD_minus_CT'].values[0]
        canonical = np.nanmean([c10['AD_minus_CT'].values[0] if not c10.empty else np.nan,
                                 c11['AD_minus_CT'].values[0] if not c11.empty else np.nan])
        print(f'{gene}: C18 AD-CT={c18_delta:.3f}, C10/C11 avg={canonical:.3f}, '
              f'C18-specific Δ={c18_delta - canonical:.3f}')
        ad_n = c18['AD_n_donors'].values[0]
        ct_n = c18['CT_n_donors'].values[0]
        print(f'  (AD donors in C18: {ad_n}, CT donors: {ct_n})')
