#!/usr/bin/env python3
"""
Visualize ESM-2 vs splice_delta complementarity
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from collections import defaultdict

# Load data
splice_delta = np.load('/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/features/splicing/splicing_delta_v2.npy')
esm2_emb = np.load('/home/welcome1/sw1686/DIFFUSE/hMuscle/data/esm2_embeddings_t30_150M.npy')
gene_ids = np.load('/home/welcome1/sw1686/DIFFUSE/hMuscle/model/my_gene_list_fixed.npy', allow_pickle=True)

# Build gene -> isoform mapping
gene_to_isoforms = defaultdict(list)
for i, gene_id in enumerate(gene_ids):
    gene_to_isoforms[gene_id].append(i)

multi_iso_genes = {g: iso_list for g, iso_list in gene_to_isoforms.items() if len(iso_list) >= 2}

# Compute within-gene distances
esm2_dists = []
splice_dists = []

for gene_id, iso_indices in multi_iso_genes.items():
    esm2_vecs = esm2_emb[iso_indices]
    splice_vecs = splice_delta[iso_indices]
    
    esm2_pdist = pdist(esm2_vecs, metric='cosine')
    splice_pdist = pdist(splice_vecs, metric='cityblock')
    
    esm2_dists.append(esm2_pdist.mean())
    splice_dists.append(splice_pdist.mean())

esm2_dists = np.array(esm2_dists)
splice_dists = np.array(splice_dists)

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Scatter plot with density
ax = axes[0]
scatter = ax.scatter(esm2_dists, splice_dists, alpha=0.3, s=10, c='#2E86AB', edgecolors='none')
ax.set_xlabel('Within-gene ESM-2 distance (cosine)', fontsize=12)
ax.set_ylabel('Within-gene splice_delta distance (L1)', fontsize=12)
ax.set_title('ESM-2 vs splice_delta Within-Gene Separation\n(n=8,569 multi-isoform genes)', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')

# Add Pearson r
from scipy.stats import pearsonr, spearmanr
pearson_r, pearson_p = pearsonr(esm2_dists, splice_dists)
spearman_r, spearman_p = spearmanr(esm2_dists, splice_dists)
ax.text(0.95, 0.95, f'Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_r:.3f}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Highlight ESM-2 failure zone
esm2_low_thresh = np.percentile(esm2_dists, 25)
splice_high_thresh = np.percentile(splice_dists, 75)
ax.axvline(esm2_low_thresh, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='ESM-2 P25')
ax.axhline(splice_high_thresh, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='splice P75')
ax.fill_between([0, esm2_low_thresh], 0, splice_high_thresh, alpha=0.1, color='gray')
ax.fill_between([0, esm2_low_thresh], splice_high_thresh, ax.get_ylim()[1], alpha=0.2, color='red',
                label=f'ESM-2 fails, splice wins (n={((esm2_dists < esm2_low_thresh) & (splice_dists > splice_high_thresh)).sum()})')
ax.legend(loc='lower right', fontsize=9)

# Plot 2: 2D histogram
ax = axes[1]
hist, xedges, yedges = np.histogram2d(esm2_dists, splice_dists, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax.imshow(hist.T, origin='lower', extent=extent, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xlabel('Within-gene ESM-2 distance (cosine)', fontsize=12)
ax.set_ylabel('Within-gene splice_delta distance (L1)', fontsize=12)
ax.set_title('2D Histogram (Density)', fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Gene count', fontsize=10)

# Plot 3: Marginal distributions
ax = axes[2]
ax.hist(esm2_dists, bins=50, alpha=0.6, label='ESM-2 distance', color='#2E86AB', density=True)
# Scale splice_dists to same range for comparison
splice_dists_scaled = splice_dists / splice_dists.max() * esm2_dists.max()
ax.hist(splice_dists_scaled, bins=50, alpha=0.6, label='splice_delta distance (scaled)', color='#A23B72', density=True)
ax.set_xlabel('Distance (arbitrary units)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Marginal Distributions', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/home/welcome1/sw1686/DIFFUSE/reports/esm2_splice_complementarity.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/welcome1/sw1686/DIFFUSE/reports/esm2_splice_complementarity.pdf', bbox_inches='tight')
print("✓ Saved: /home/welcome1/sw1686/DIFFUSE/reports/esm2_splice_complementarity.png")
print("✓ Saved: /home/welcome1/sw1686/DIFFUSE/reports/esm2_splice_complementarity.pdf")
