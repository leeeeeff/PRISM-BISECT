#!/usr/bin/env python3
"""
compute_delta_features.py
==========================
기존 RNA(9-dim) + LOC(8-dim) features를 gene-mean-subtracted DELTA로 변환.

진단 근거:
  RNA B/W=2.35, LOC B/W=3.23 → gene-level biased (절대값 인코딩)
  splice_delta B/W=0.90       → isoform-specific (상대값 인코딩)

DELTA 변환: f_delta(i) = f(i) - mean_{j∈gene(i)} f(j)
  → B/W가 ~0.9 수준으로 감소 예상
  → 이소폼이 유전자 평균 대비 어떻게 다른지 인코딩

출력:
  hMuscle/results_isoform/features/rna/rna_delta_test.npy   (36748 × 9)
  hMuscle/results_isoform/features/rna/rna_delta_train.npy  (31668 × 9)
  hMuscle/results_isoform/features/loc/loc_delta_test.npy   (36748 × 8)
  hMuscle/results_isoform/features/loc/loc_delta_train.npy  (31668 × 8)
"""

import os, json
import numpy as np
from collections import defaultdict

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL   = os.path.join(BASE, 'model')
ID_DIR  = os.path.join(BASE, 'data/raw_data/data/id_lists')
RNA_DIR = os.path.join(BASE, 'results_isoform/features/rna')
LOC_DIR = os.path.join(BASE, 'results_isoform/features/loc')


def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def bw_ratio(X, gene_ids):
    gene_to_idx = defaultdict(list)
    for i, g in enumerate(gene_ids):
        gene_to_idx[g].append(i)
    multi = [(g, idxs) for g, idxs in gene_to_idx.items() if len(idxs) >= 2]
    if not multi:
        return float('nan')
    gene_means, within = [], []
    for g, idxs in multi:
        rows = X[idxs]
        gene_means.append(rows.mean(0))
        within.append(rows.var(0).mean())
    bg = np.var(gene_means, axis=0).mean()
    wg = np.mean(within)
    return float(bg / (wg + 1e-10))


def compute_delta(X, gene_ids):
    """isoform feature → gene-mean-subtracted delta."""
    gene_to_idx = defaultdict(list)
    for i, g in enumerate(gene_ids):
        gene_to_idx[g].append(i)

    X_delta = X.copy()
    n_single = 0
    for g, idxs in gene_to_idx.items():
        if len(idxs) == 1:
            X_delta[idxs[0]] = 0.0  # single isoform: no relative info
            n_single += 1
        else:
            gene_mean = X[idxs].mean(axis=0)
            for idx in idxs:
                X_delta[idx] = X[idx] - gene_mean
    print(f"  Single-isoform genes zeroed: {n_single}")
    return X_delta


# ── Test set ──────────────────────────────────────────────────────────────────
print("[1] Loading test features...")
te_gene = load_ids(os.path.join(MODEL, 'my_gene_list_fixed.npy'))
X_rna_te = np.load(os.path.join(RNA_DIR, 'rna_features_test.npy')).astype(np.float32)
X_loc_te = np.load(os.path.join(LOC_DIR, 'loc_features_test.npy')).astype(np.float32)

print(f"  RNA test shape: {X_rna_te.shape}")
print(f"  LOC test shape: {X_loc_te.shape}")
print(f"\nBefore DELTA transform:")
print(f"  RNA B/W: {bw_ratio(X_rna_te, te_gene):.3f}")
print(f"  LOC B/W: {bw_ratio(X_loc_te, te_gene):.3f}")

print("\nComputing test RNA delta...")
X_rna_delta_te = compute_delta(X_rna_te, te_gene)
print("Computing test LOC delta...")
X_loc_delta_te = compute_delta(X_loc_te, te_gene)

print(f"\nAfter DELTA transform:")
print(f"  RNA B/W: {bw_ratio(X_rna_delta_te, te_gene):.3f}")
print(f"  LOC B/W: {bw_ratio(X_loc_delta_te, te_gene):.3f}")


# ── Train set ─────────────────────────────────────────────────────────────────
print("\n[2] Loading train features...")
tr_iso  = load_ids(os.path.join(ID_DIR, 'train_isoform_list.npy'))
tr_gene = load_ids(os.path.join(ID_DIR, 'train_gene_list.npy'))
X_rna_tr = np.load(os.path.join(RNA_DIR, 'rna_features_train.npy')).astype(np.float32)
X_loc_tr = np.load(os.path.join(LOC_DIR, 'loc_features_train.npy')).astype(np.float32)

print(f"  RNA train shape: {X_rna_tr.shape}")
print(f"  LOC train shape: {X_loc_tr.shape}")
print(f"\nBefore DELTA transform:")
print(f"  RNA B/W: {bw_ratio(X_rna_tr, tr_gene):.3f}")
print(f"  LOC B/W: {bw_ratio(X_loc_tr, tr_gene):.3f}")

print("\nComputing train RNA delta...")
X_rna_delta_tr = compute_delta(X_rna_tr, tr_gene)
print("Computing train LOC delta...")
X_loc_delta_tr = compute_delta(X_loc_tr, tr_gene)

print(f"\nAfter DELTA transform:")
print(f"  RNA B/W: {bw_ratio(X_rna_delta_tr, tr_gene):.3f}")
print(f"  LOC B/W: {bw_ratio(X_loc_delta_tr, tr_gene):.3f}")


# ── Save ──────────────────────────────────────────────────────────────────────
print("\n[3] Saving delta features...")
np.save(os.path.join(RNA_DIR, 'rna_delta_test.npy'),  X_rna_delta_te)
np.save(os.path.join(RNA_DIR, 'rna_delta_train.npy'), X_rna_delta_tr)
np.save(os.path.join(LOC_DIR, 'loc_delta_test.npy'),  X_loc_delta_te)
np.save(os.path.join(LOC_DIR, 'loc_delta_train.npy'), X_loc_delta_tr)

stats = {
    'rna_before_bw_test':   bw_ratio(X_rna_te, te_gene),
    'rna_after_bw_test':    bw_ratio(X_rna_delta_te, te_gene),
    'loc_before_bw_test':   bw_ratio(X_loc_te, te_gene),
    'loc_after_bw_test':    bw_ratio(X_loc_delta_te, te_gene),
    'rna_delta_test_shape':  list(X_rna_delta_te.shape),
    'loc_delta_test_shape':  list(X_loc_delta_te.shape),
    'rna_delta_train_shape': list(X_rna_delta_tr.shape),
    'loc_delta_train_shape': list(X_loc_delta_tr.shape),
}
with open(os.path.join(RNA_DIR, 'delta_feature_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

print("DONE")
print(json.dumps(stats, indent=2))
