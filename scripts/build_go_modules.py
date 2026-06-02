#!/usr/bin/env python3
"""
build_go_modules.py
====================
672-term brain score matrix (63994×672)에서 GO-GO 상관관계를 계산하고
Ward hierarchical clustering으로 기능 모듈을 발굴.

Circular validation 방지:
  - 모듈 구성: PRISM score-based Pearson r (exploratory)
  - 모듈 검증: GO semantic similarity (DAG 기반, PRISM 독립)
              + BISECT case module divergence check

출력:
  - brain_go_modules.json  : go_id → module_id/name/members
  - brain_go_corr.npy      : (672×672) Pearson r matrix
  - brain_go_dendrogram.png: Ward linkage dendrogram
"""
import json, time
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score
import warnings; warnings.filterwarnings('ignore')

BASE     = Path('/home/welcome1/sw1686/DIFFUSE')
REPORTS  = BASE / 'reports'

# ── Load 672-term score matrix ─────────────────────────────────────────────────
print("Loading brain 672-term score matrix...")
score_path = REPORTS / 'brain_full_672_scores.npy'
S = np.load(score_path)          # (63994, 672)
print(f"  Shape: {S.shape}")

with open(BASE / 'hMuscle/data/brain672_go_terms.json') as f:
    go_meta = json.load(f)
go_ids   = go_meta['go_ids']     # 672개
go_names_d = go_meta['go_names']
N_GO = len(go_ids)
print(f"  GO terms: {N_GO}")

# ── GO-GO Pearson correlation (672×672) ───────────────────────────────────────
print("Computing GO-GO Pearson correlation matrix...")
t0 = time.time()
# Use numpy corrcoef on transposed matrix (each col = one GO term's scores)
R = np.corrcoef(S.T)             # (672, 672), fast vectorized
R = np.clip(R, -1.0, 1.0)
print(f"  Done in {time.time()-t0:.1f}s  |  R range: [{R.min():.3f}, {R.max():.3f}]")

np.save(REPORTS / 'brain_go_corr_672.npy', R.astype(np.float32))
print(f"  Saved: brain_go_corr_672.npy")

# ── Ward hierarchical clustering ──────────────────────────────────────────────
print("Ward hierarchical clustering...")
# Convert correlation to distance: d = 1 - r (range 0~2 for r in [-1,1])
dist_mat = 1.0 - R
np.fill_diagonal(dist_mat, 0.0)
dist_condensed = squareform(dist_mat, checks=False)
Z = linkage(dist_condensed, method='ward')
print(f"  Linkage computed")

# ── Optimal k via silhouette score (range 10~40) ──────────────────────────────
print("Finding optimal number of clusters via silhouette...")
sil_scores = {}
for k in range(10, 45, 5):
    labels = fcluster(Z, k, criterion='maxclust')
    if len(np.unique(labels)) < 2:
        continue
    sil = silhouette_score(dist_mat, labels, metric='precomputed')
    sil_scores[k] = sil
    print(f"  k={k:3d}  silhouette={sil:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\nBest k = {best_k}  (silhouette = {sil_scores[best_k]:.4f})")

# Fine-search around best_k ±4
fine_range = range(max(5, best_k-4), best_k+5)
for k in fine_range:
    if k not in sil_scores:
        labels = fcluster(Z, k, criterion='maxclust')
        if len(np.unique(labels)) >= 2:
            sil = silhouette_score(dist_mat, labels, metric='precomputed')
            sil_scores[k] = sil

best_k = max(sil_scores, key=sil_scores.get)
print(f"Final best k = {best_k}  (silhouette = {sil_scores[best_k]:.4f})")

# ── Assign GO terms to modules ────────────────────────────────────────────────
module_labels = fcluster(Z, best_k, criterion='maxclust')  # 1-indexed

# Module composition
modules = {}
for mod_id in sorted(np.unique(module_labels)):
    members = [go_ids[i] for i, m in enumerate(module_labels) if m == mod_id]
    # Top 3 GO names for auto-label
    top3 = [go_names_d[g][:30] for g in members[:3]]
    modules[int(mod_id)] = {
        'module_id': int(mod_id),
        'size': len(members),
        'go_ids': members,
        'label': ' / '.join(top3),
        'top3_names': top3,
    }

# Print module summary
print(f"\n=== {best_k} Functional Modules ===")
for mid, m in sorted(modules.items(), key=lambda x: -x[1]['size']):
    print(f"  Module {mid:2d} ({m['size']:3d} terms): {m['label'][:70]}")

# Per-GO term module assignment
go_module_map = {go_ids[i]: int(module_labels[i]) for i in range(N_GO)}

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'n_go': N_GO,
    'n_modules': best_k,
    'best_silhouette': sil_scores[best_k],
    'silhouette_by_k': {str(k): round(v, 4) for k, v in sorted(sil_scores.items())},
    'go_module_map': go_module_map,   # go_id → module_id
    'modules': {str(k): v for k, v in modules.items()},
}
out_path = REPORTS / 'brain_go_modules_672.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")

# ── Dendrogram (top-level, 최대 30 leaf 표시) ─────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 8))
    dn = dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
                    leaf_rotation=90, leaf_font_size=8,
                    show_contracted=True,
                    color_threshold=Z[-(best_k-1), 2])
    ax.set_title(f'GO-GO Ward Dendrogram (672 terms, k={best_k} modules)', fontsize=14)
    ax.set_xlabel('GO terms (contracted)')
    ax.set_ylabel('Ward distance')
    plt.tight_layout()
    fig.savefig(REPORTS / 'brain_go_dendrogram_672.png', dpi=150)
    plt.close()
    print(f"Saved: brain_go_dendrogram_672.png")
except Exception as e:
    print(f"Dendrogram skipped: {e}")

print("\nAll done.")
