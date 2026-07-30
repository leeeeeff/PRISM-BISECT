#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sota_final_benchmark_knn_v2_truebrain.py
==========================================
TISSUE-MISLABELING BUGFIX RERUN (2026-07-18), FOCUSED SCOPE.
sota_final_benchmark.py (the script that produced the manuscript's headline
k-NN ESM-2 number, 0.636 [0.621,0.652] All MF / 0.543 L2_Structural) loaded
my_gene_list_fixed.npy (36,748 isoforms) as its "brain" test set -- these are
MUSCLE data. This is a DIFFERENT k-NN implementation from exp_e_sota_comparison.py's
E0_kNN_L30 (which gave a since-superseded, buggy 0.599 -- see memory
session_20260629_sota_benchmark), so the exp_e_sota_comparison_v2_truebrain.py rerun
from 2026-07-14 (E0_kNN_L30 = 0.5216) does NOT correct this specific number.

This script reruns ONLY the kNN_ESM2 section of sota_final_benchmark.py against the
TRUE brain isoform set (brain_full_gene_names.npy / brain_full_ids.npy / TRUE-brain
ESM-2 L30 embeddings, 63,994 isoforms / 18,514 genes), keeping the training side
(esm2_train_human_layer30, train_gene_list.npy, Y_tr_v construction) byte-for-byte
identical to the original script. D0/D1/D2/v17f*/DeepFRI/Domain_LR already have
correct true-brain values from other reruns in this family (exp_d_finetune,
v17f_star_bootstrap, exp_e_sota/deepfri_auprc.json) and are NOT recomputed here.

Original (mislabeled-as-brain, actually muscle): sota_final_benchmark.py -- left unmodified.
"""

import os, json, gzip, time
import numpy as np
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
BRAIN_DIR = '../data/brain_isoquant_esm2/full'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/truebrain_rerun_20260714/sota_final_benchmark_knn'
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOT = 500
K_NN   = 5

print("=" * 70)
print("  SOTA k-NN ESM-2 rerun — TRUE BRAIN (63,994 isoforms)")
print("=" * 70)

# ── 1. Load embeddings ───────────────────────────────────────────────
print("\n[1] Loading embeddings...")
X30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X30_te = np.load(f'{BRAIN_DIR}/brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
print(f"  Train: {X30_tr.shape}  Test (TRUE brain): {X30_te.shape}")

# ── 2. Gene IDs ──────────────────────────────────────────────────────
print("\n[2] Loading gene IDs...")

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
tr_sym2idx   = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

# TRUE BRAIN: gene names already symbols, no ENSG2SYM mapping needed.
te_genes_raw = np.load(f'{BRAIN_DIR}/brain_full_gene_names.npy', allow_pickle=True)
te_sym_list  = [clean(g) for g in te_genes_raw]
gene2idxs_te = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs_te[g].append(i)
te_genes_arr = np.array(list(gene2idxs_te.keys()))
print(f"  Train: {len(tr_genes)} isoforms / Test: {len(te_sym_list)} isoforms ({len(te_genes_arr)} genes)")

# ── 3. GO labels ─────────────────────────────────────────────────────
print("\n[3] Loading GO MF labels (82 terms)...")
sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id: sym2id[syn] = p[1]

tr_ids    = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_ids)
go_tr  = defaultdict(set)
go_all = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        gid, go_id, cat_raw = p[1], p[2], p[7]
        if cat_raw != 'Function': continue
        go_all[go_id].add(gid)
        if gid in tr_id_set: go_tr[go_id].add(gid)

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])

def build_Y_tr(terms):
    rows = []
    for go_id in terms:
        pos_ids  = go_tr[go_id]
        pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in pos_ids}
        y = np.zeros(len(tr_genes), dtype=np.float32)
        for sym in pos_syms:
            for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
        rows.append(y)
    return np.stack(rows, axis=1)

def build_Y_te(terms):
    return np.stack([
        np.array([1.0 if sym2id.get(s, '__') in go_all[go_id] else 0.0
                  for s in te_sym_list], dtype=np.float32)
        for go_id in terms
    ], axis=1)

Y_tr = build_Y_tr(mf_terms)
Y_te = build_Y_te(mf_terms)
valid_mask = Y_te.sum(0) >= 2
mf_valid   = [go for go, v in zip(mf_terms, valid_mask) if v]
Y_tr_v = Y_tr[:, valid_mask]
Y_te_v = Y_te[:, valid_mask]
print(f"  MF terms: {len(mf_terms)} total / {valid_mask.sum()} valid (≥2 positives)")

L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural': L2_TERMS.add(p[0])

l2_idx = [i for i, go in enumerate(mf_valid) if go in L2_TERMS]
print(f"  L2_Structural terms: {len(l2_idx)}/{len(mf_valid)}")

# ── 4. Bootstrap CI infrastructure (identical to original) ────────────
rng = np.random.default_rng(42)
_gene_idx_arrays = [np.array(gene2idxs_te[g], dtype=np.int32) for g in te_genes_arr]
_n_genes = len(te_genes_arr)

def fast_auprc(y_true, y_score):
    order  = np.argsort(-y_score, kind='quicksort')
    y_true = y_true[order]
    n_pos  = y_true.sum()
    if n_pos == 0: return float('nan')
    tp     = np.cumsum(y_true)
    prec   = tp / np.arange(1, len(y_true) + 1, dtype=np.float64)
    rec    = tp / n_pos
    delta_r = np.diff(rec, prepend=0.0)
    return float(np.dot(prec, delta_r))

def bootstrap_ci(preds, Y, l2_idx, B=N_BOOT):
    l2_set = set(l2_idx)
    boot_all, boot_l2 = [], []
    n_valid = Y.shape[1]
    t0 = time.time()
    for b in range(B):
        g_sample = rng.integers(0, _n_genes, size=_n_genes)
        bidxs    = np.concatenate([_gene_idx_arrays[g] for g in g_sample])
        Yb = Y[bidxs]; pb = preds[bidxs]
        aps_all, aps_l2 = [], []
        for j in range(n_valid):
            if Yb[:, j].sum() < 2: continue
            ap = fast_auprc(Yb[:, j], pb[:, j])
            if not np.isnan(ap):
                aps_all.append(ap)
                if j in l2_set: aps_l2.append(ap)
        if aps_all:  boot_all.append(float(np.mean(aps_all)))
        if aps_l2:   boot_l2.append(float(np.mean(aps_l2)))
        if (b + 1) % 100 == 0:
            print(f"    boot {b+1}/{B}  [{time.time()-t0:.0f}s]", flush=True)
    ci_all = [float(np.percentile(boot_all, 2.5)), float(np.percentile(boot_all, 97.5))]
    ci_l2  = [float(np.percentile(boot_l2,  2.5)), float(np.percentile(boot_l2,  97.5))]
    return ci_all, ci_l2

def macro_ap(preds, Y, l2_idx):
    all_ap = [fast_auprc(Y[:, j], preds[:, j])
              for j in range(Y.shape[1]) if Y[:, j].sum() >= 2]
    l2_ap  = [fast_auprc(Y[:, j], preds[:, j])
              for j in l2_idx if Y[:, j].sum() >= 2]
    return float(np.nanmean(all_ap)), float(np.nanmean(l2_ap)) if l2_ap else float('nan')

# ── 5. k-NN ESM-2 (identical algorithm, TRUE-brain query) ──────────────
print("\n[k-NN] ESM-2 L30 k-NN retrieval (k=5, cosine), TRUE brain query...")
t0 = time.time()

def l2norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)

X_tr_n = l2norm(X30_tr)
X_te_n = l2norm(X30_te)
BATCH  = 512
knn_preds = np.zeros((len(te_sym_list), len(mf_valid)), dtype=np.float32)
for start in range(0, len(te_sym_list), BATCH):
    end     = min(start + BATCH, len(te_sym_list))
    sim     = X_te_n[start:end] @ X_tr_n.T
    top_idx = np.argpartition(sim, -K_NN, axis=1)[:, -K_NN:]
    for bi in range(end - start):
        knn_preds[start + bi] = Y_tr_v[top_idx[bi]].mean(axis=0)
    if start % (BATCH * 20) == 0:
        print(f"  k-NN {start}/{len(te_sym_list)}", flush=True)

pt_all, pt_l2 = macro_ap(knn_preds, Y_te_v, l2_idx)
print(f"  Point: All MF={pt_all:.4f}  L2={pt_l2:.4f}  [{time.time()-t0:.0f}s]")
print(f"  Bootstrap CI (B={N_BOOT})...")
ci_all, ci_l2 = bootstrap_ci(knn_preds, Y_te_v, l2_idx)
print(f"  k-NN TRUE-BRAIN: All={pt_all:.4f} {ci_all}  L2={pt_l2:.4f} {ci_l2}")

result = {
    'method': 'kNN_ESM2_truebrain',
    'point_all': pt_all, 'ci_all': ci_all,
    'point_l2':  pt_l2,  'ci_l2':  ci_l2,
    'note': 'k=5 cosine, L30 ESM-2, isoform-level retrieval, TRUE brain query (63994 isoforms)',
    'n_isoforms': len(te_sym_list),
    'n_genes': len(te_genes_arr),
    'muscle_mislabeled_reference': {
        'point_all': 0.636, 'ci_all': [0.621, 0.652],
        'point_l2': 0.543, 'ci_l2': [0.519, 0.567],
        'n_isoforms': 36748,
    },
}
with open(f'{OUT_DIR}/results.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n[Saved] {OUT_DIR}/results.json")
