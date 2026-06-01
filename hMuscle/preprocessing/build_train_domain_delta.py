# -*- coding: utf-8 -*-
"""
build_train_domain_delta.py
===========================
Training set (human_train)에 대한 domain_delta 계산

domain_delta[i] = sign(domain_matrix[i] - domain_matrix[canonical_of_gene_i])
canonical: gene 내에서 nonzero domain count가 가장 많은 isoform

출력:
  ../results_isoform/features/train_domain_delta_sign.npy  (31668, 251) float32
  (swissprot: all zeros → swissprot_domain_delta_sign.npy)
"""

import numpy as np
import os

DOMAIN_TRAIN  = '../data/raw_data/data/domains/human_domain_train.npy'
GENE_LIST     = '../data/raw_data/data/id_lists/train_gene_list.npy'
SW_DOMAIN     = '../data/raw_data/data/domains/swissprot_domain_train.npy'

OUT_TRAIN  = '../results_isoform/features/train_domain_delta_sign.npy'
OUT_SW     = '../results_isoform/features/swissprot_domain_delta_sign.npy'

os.makedirs('../results_isoform/features', exist_ok=True)

print("[1] Loading training domain matrix...")
dm = np.load(DOMAIN_TRAIN).astype(np.float32)  # (31668, 251)
genes = np.load(GENE_LIST, allow_pickle=True)
genes = np.array([g.decode('utf-8') if isinstance(g, bytes) else g for g in genes])
print(f"    domain_train: {dm.shape}, unique genes: {len(np.unique(genes))}")

print("[2] Computing domain_delta for human_train...")
delta = np.zeros_like(dm, dtype=np.float32)

unique_genes = np.unique(genes)
n_canonical_found = 0

for gene in unique_genes:
    idxs = np.where(genes == gene)[0]
    if len(idxs) == 1:
        # 단일 isoform → delta = 0 (canonical 자신)
        continue

    # canonical: nonzero count 최대
    nonzero_counts = (dm[idxs] != 0).sum(axis=1)
    canonical_local = np.argmax(nonzero_counts)
    canonical_idx = idxs[canonical_local]
    canonical_vec = dm[canonical_idx]

    for idx in idxs:
        if idx == canonical_idx:
            continue
        delta[idx] = dm[idx] - canonical_vec

    n_canonical_found += 1

print(f"    Genes with multiple isoforms: {n_canonical_found}")
nonzero_rows = (np.abs(delta).sum(axis=1) > 0).sum()
print(f"    Nonzero delta rows: {nonzero_rows} / {len(delta)}")

# Sign transform
delta_sign = np.sign(delta).astype(np.float32)
print(f"    Sign values — -1: {(delta_sign==-1).sum()}, 0: {(delta_sign==0).sum()}, +1: {(delta_sign==1).sum()}")

np.save(OUT_TRAIN, delta_sign)
print(f"[3] Saved: {OUT_TRAIN} {delta_sign.shape}")

print("[4] Creating swissprot domain delta (all zeros)...")
dm_sw = np.load(SW_DOMAIN)
sw_delta = np.zeros((dm_sw.shape[0], dm_sw.shape[1]), dtype=np.float32)
np.save(OUT_SW, sw_delta)
print(f"    Saved: {OUT_SW} {sw_delta.shape}")

print("\n[Done] domain_delta sign files ready.")
print(f"  train:     {OUT_TRAIN}")
print(f"  swissprot: {OUT_SW}")
print(f"  test:      ../results_isoform/features/domain_delta.npy (sign transform needed)")
