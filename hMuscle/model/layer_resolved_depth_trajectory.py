#!/usr/bin/env python
"""
Option A — Layer-resolved depth trajectory of the POOLED edit signal (interpretability MAP).

Purpose: resolve the framework's single "B1 encoding" box into a 30-step depth curve, and
adjudicate devils-advocate Attack 1 ("B1 nearly lossless") as far as CACHED (pooled) data allow.

Design (brain = primary/clean tissue, FRAMEWORK.md §8d):
  For each within-gene canonical-anchored pair, at each ESM-2 layer l=1..30:
      dphi^(l) = phi^(l)[other_idx] - phi^(l)[canonical_idx]      (pooled 640-d difference)
  Then:
   (1) Depth curve of DOMAIN decodability: gene-disjoint logistic probe AUROC(l) for domain_binary.
   (2) Magnitude trajectory: mean ||dphi^(l)||_2 across depth, split domain vs non-domain, and
       within non-domain by edit-size tercile (small = SLiM-candidate).
   (3) delta_layer comparison: AUROC using [phi30], [phi30 || (phi30-phi15)], and (phi30-phi15) alone
       -> does the B2a depth-contrast add decodable domain signal beyond the last layer?

HARD LIMITATION (stated up front): every layer here is ALREADY mean-pooled, so this cannot
separate "B1 destroyed it" from "B2b pooled it away at each depth". The clean per-residue test
is deferred to Option B (needs re-extraction). This pass answers: WHERE along depth does the
pooled edit signal emerge, and is small-edit signal EVER present in pooled space at any depth.

Read-only on all inputs. Resource-limited for the shared server.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
BRAIN = ROOT / 'hMuscle/data/brain_isoquant_esm2/full'
PAIRS = ROOT / 'reports/severity_pairs/brain_severity_pairs_scored.tsv'
OUT = ROOT / 'reports/model_interpretability_map'
OUT.mkdir(exist_ok=True)
NL = 30
SEED = 0
rng = np.random.default_rng(SEED)

def layer_path(l): return BRAIN / f'brain_full_esm2_layer{l:02d}_t30_150M.npy'

print('[load] pair table', flush=True)
df = pd.read_csv(PAIRS, sep='\t')
ca = df['canonical_idx'].to_numpy(); ot = df['other_idx'].to_numpy()
y = df['domain_binary'].astype(int).to_numpy()
genes = df['gene'].to_numpy()
size = df['size'].to_numpy().astype(float)
n = len(df)
print(f'  pairs={n} genes={df.gene.nunique()} domain_prev={y.mean():.3f}', flush=True)

# unique rows actually needed (memory discipline): load only these per layer
need = np.unique(np.concatenate([ca, ot]))
remap = {r: i for i, r in enumerate(need)}
ca_l = np.array([remap[r] for r in ca]); ot_l = np.array([remap[r] for r in ot])
print(f'  unique embedding rows needed={len(need)} / 63994', flush=True)

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(np.zeros(n), y, groups=genes))

def cv_auroc(X, target):
    """gene-disjoint 5-fold logistic AUROC on standardized X."""
    oof = np.zeros(len(target))
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=300, C=0.5, tol=1e-3)
        clf.fit(sc.transform(X[tr]), target[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(target, oof)

# ---- pass 1: per-layer dphi, decodability + magnitude ----
small = size <= np.quantile(size, 0.25)          # SLiM-candidate edits (<=~69 aa)
nd = (y == 0)
rows = []
phi15 = phi30 = None
for l in range(1, NL + 1):
    emb = np.load(layer_path(l), mmap_mode='r')[need].astype(np.float32)
    d = emb[ot_l] - emb[ca_l]                     # (n, 640) canonical-anchored pooled diff
    if l == 15: phi15 = d.copy()
    if l == 30: phi30 = d.copy()
    au = cv_auroc(d, y)
    mag = np.linalg.norm(d, axis=1)
    rows.append(dict(layer=l, auroc_domain=au,
                     mag_domain=float(mag[y == 1].mean()),
                     mag_nondomain=float(mag[nd].mean()),
                     mag_nd_small=float(mag[nd & small].mean()),
                     mag_nd_large=float(mag[nd & ~small].mean())))
    print(f'  L{l:02d}  AUROC_domain={au:.4f}  '
          f'|dphi| dom={mag[y==1].mean():6.3f} nd={mag[nd].mean():6.3f} '
          f'nd_small={mag[nd&small].mean():6.3f} nd_large={mag[nd&~small].mean():6.3f}', flush=True)
    del emb, d
res = pd.DataFrame(rows)

# ---- pass 2: delta_layer (B2a) comparison ----
print('\n[B2a] delta_layer = phi30 - phi15 comparison', flush=True)
dlayer = phi30 - phi15
au_30 = cv_auroc(phi30, y)
au_dl = cv_auroc(dlayer, y)
au_cat = cv_auroc(np.concatenate([phi30, dlayer], axis=1), y)
print(f'  AUROC[phi30]        = {au_30:.4f}', flush=True)
print(f'  AUROC[phi30-phi15]  = {au_dl:.4f}   (delta_layer alone)', flush=True)
print(f'  AUROC[phi30 || dl]  = {au_cat:.4f}   (concat; B2a incremental = {au_cat-au_30:+.4f})', flush=True)

res.to_csv(OUT / 'layer_depth_trajectory.tsv', sep='\t', index=False)
with open(OUT / 'layer_depth_trajectory_summary.txt', 'w') as f:
    f.write(f'brain pairs n={n} genes={df.gene.nunique()} domain_prev={y.mean():.3f}\n')
    f.write(f'peak AUROC layer = L{int(res.loc[res.auroc_domain.idxmax(),"layer"])} '
            f'({res.auroc_domain.max():.4f}); L1={res.auroc_domain.iloc[0]:.4f} '
            f'L15={res.loc[res.layer==15,"auroc_domain"].iloc[0]:.4f} '
            f'L30={res.loc[res.layer==30,"auroc_domain"].iloc[0]:.4f}\n')
    f.write(f'phi30={au_30:.4f} delta_layer={au_dl:.4f} concat={au_cat:.4f} '
            f'B2a_incremental={au_cat-au_30:+.4f}\n')
print('\n[done] wrote layer_depth_trajectory.tsv', flush=True)
