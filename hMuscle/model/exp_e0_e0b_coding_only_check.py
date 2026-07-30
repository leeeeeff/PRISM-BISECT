#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_e0_e0b_coding_only_check.py
---------------------------------------
Verification rerun (2026-07-29): does the non-coding zero-embedding contamination
(finding-brain672-noncoding-zero-embedding-flag-needed) affect Table S2b's
ESM-2 k-NN (E0) and ESM-2 Gene-mean retrieval (E0b) true-brain numbers the same
way it affected DR-AUC (0.775) and macro AUPRC (0.647)?

E0 logic reproduced byte-for-byte from sota_final_benchmark_knn_v2_truebrain.py
(the canonical ‡-marked source of Table S2b's 0.561 k-NN row).
E0b logic reproduced byte-for-byte from exp_e_sota_comparison_v2_truebrain.py
(the non-‡ source of Table S2b's 0.404 Gene-mean row, reused from the 2026-07-14 rerun).

Both scored over the FULL 63,994-isoform true-brain test set (reproducing the
manuscript's exact reported numbers as a sanity check), then re-scored restricted
to the 53,826 protein-coding isoforms only (brain_full_mask.npy), matching the
methodology already applied to DR-AUC/macro-AUPRC.
"""
import os, gzip, time, json
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
BRAIN_DIR = '../data/brain_isoquant_esm2/full'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/truebrain_rerun_20260714/exp_e0_e0b_coding_only'
os.makedirs(OUT_DIR, exist_ok=True)

K = 5
BATCH = 512

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)

def eval_auprc(scores_2d, labels_2d, l2_idx):
    aps_all = [average_precision_score(labels_2d[:, j], scores_2d[:, j])
               for j in range(labels_2d.shape[1]) if labels_2d[:, j].sum() >= 2]
    aps_l2  = [average_precision_score(labels_2d[:, j], scores_2d[:, j])
               for j in l2_idx if labels_2d[:, j].sum() >= 2]
    return float(np.mean(aps_all)), (float(np.mean(aps_l2)) if aps_l2 else float('nan'))

print("=" * 65)
print("  E0/E0b coding-only contamination check (true brain)")
print("=" * 65)

print("\n[1] Loading embeddings + coding mask...")
X_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X_te = np.load(f'{BRAIN_DIR}/brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
coding_mask = np.load(f'{BRAIN_DIR}/brain_full_mask.npy').astype(bool)
print(f"  Train: {X_tr.shape}  Test: {X_te.shape}  Coding: {coding_mask.sum()}/{len(coding_mask)}")

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
tr_sym2idx   = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

te_genes_raw = np.load(f'{BRAIN_DIR}/brain_full_gene_names.npy', allow_pickle=True)
te_sym_list  = [clean(g) for g in te_genes_raw]
assert len(te_sym_list) == len(coding_mask)

print("\n[2] Loading GO MF labels (82 terms)...")
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
go_tr, go_all = defaultdict(set), defaultdict(set)
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
Y_te_full = build_Y_te(mf_terms)
# valid-term mask defined on FULL set (matches both canonical scripts)
valid_mask = Y_te_full.sum(0) >= 2
mf_valid   = [go for go, v in zip(mf_terms, valid_mask) if v]
Y_tr_v     = Y_tr[:, valid_mask]
Y_te_v     = Y_te_full[:, valid_mask]
print(f"  MF terms: {len(mf_terms)} total / {valid_mask.sum()} valid (>=2 positives, full-set def)")

L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural': L2_TERMS.add(p[0])
l2_idx = [i for i, go in enumerate(mf_valid) if go in L2_TERMS]
print(f"  L2_Structural terms: {len(l2_idx)}/{len(mf_valid)}")

X_tr_n = l2norm(X_tr)
X_te_n = l2norm(X_te)

# ── E0: k-NN (k=5 individual isoforms) ──────────────────────────
print("\n[E0] k-NN ESM-2 retrieval...")
t0 = time.time()
knn_scores = np.zeros((len(te_sym_list), len(mf_valid)), dtype=np.float32)
for start in range(0, len(te_sym_list), BATCH):
    end = min(start + BATCH, len(te_sym_list))
    sim = X_te_n[start:end] @ X_tr_n.T
    top_idx = np.argpartition(sim, -K, axis=1)[:, -K:]
    for bi in range(end - start):
        knn_scores[start + bi] = Y_tr_v[top_idx[bi]].mean(axis=0)
print(f"  done [{time.time()-t0:.0f}s]")

# ── E0b: Gene-mean (k=5 gene centroids) ─────────────────────────
print("\n[E0b] Gene-mean ESM-2 retrieval...")
t0 = time.time()
gene_centroids, centroid_labels, centroid_genes = {}, [], []
centroid_matrix = []
for sym, idxs in tr_sym2idx.items():
    cent = X_tr[idxs].mean(axis=0)
    lbl  = Y_tr_v[idxs].max(axis=0)
    centroid_matrix.append(cent); centroid_labels.append(lbl); centroid_genes.append(sym)
centroid_matrix = np.array(centroid_matrix, dtype=np.float32)
centroid_labels = np.array(centroid_labels, dtype=np.float32)
centroid_n = l2norm(centroid_matrix)

gm_scores = np.zeros((len(te_sym_list), len(mf_valid)), dtype=np.float32)
for start in range(0, len(te_sym_list), BATCH):
    end = min(start + BATCH, len(te_sym_list))
    sim = X_te_n[start:end] @ centroid_n.T
    top_idx = np.argpartition(sim, -K, axis=1)[:, -K:]
    for bi in range(end - start):
        gm_scores[start + bi] = centroid_labels[top_idx[bi]].mean(axis=0)
print(f"  done [{time.time()-t0:.0f}s]")

# ── Evaluate: full (contaminated, sanity check vs manuscript) vs coding-only ──
print("\n[Eval] Full 63,994 vs coding-only 53,826...")
results = {}
for name, scores in [('E0_kNN', knn_scores), ('E0b_GeneMean', gm_scores)]:
    full_all, full_l2 = eval_auprc(scores, Y_te_v, l2_idx)
    cod_all,  cod_l2  = eval_auprc(scores[coding_mask], Y_te_v[coding_mask], l2_idx)
    results[name] = {
        'full_all_mf': full_all, 'full_l2': full_l2,
        'coding_only_all_mf': cod_all, 'coding_only_l2': cod_l2,
        'delta_all_mf': cod_all - full_all, 'delta_l2': cod_l2 - full_l2,
    }
    print(f"  {name}: full All MF={full_all:.4f} L2={full_l2:.4f}  |  "
          f"coding-only All MF={cod_all:.4f} L2={cod_l2:.4f}  |  "
          f"delta All MF={cod_all-full_all:+.4f} L2={cod_l2-full_l2:+.4f}")

with open(f'{OUT_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[Saved] {OUT_DIR}/results.json")
print("ALL DONE")
