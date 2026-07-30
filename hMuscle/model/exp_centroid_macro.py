#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_centroid_macro.py  (Option B / question ①)
Decisive remaining compute: does the training-free ESM-2 centroid-similarity method, which TIES
PRISM on the isoform-resolution axis (DR-AUC 0.638 vs 0.630), also match PRISM on the MACRO axis
(v17f* 0.734), or does it fall to the level of the retrieval baselines (k-NN 0.636, gene-mean 0.465)?
Answers whether PRISM's *deployable calibrated multi-term predictor* role (the macro axis) is real.
Reuses the exact centroid recipe of exp_domain_ranking_baselines.py (L30, per-GO training-positive
centroid, cosine). Macro = mean average_precision over valid MF terms (same as manuscript)."""
import os, gzip, json
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
PRISM_PREDS = '../../reports/v17f_star_bootstrap/v17f_star_preds.npy'
OUT = '../../reports/domain_ranking_validation/centroid_macro.json'


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in te_genes_raw]
n_iso = len(te_sym_list)

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

go_all_mf = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606' or p[7] != 'Function': continue
        go_all_mf[p[2]].add(p[1])

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])

Y_te = np.stack([
    np.array([1.0 if sym2id.get(s, '__') in go_all_mf[go_id] else 0.0 for s in te_sym_list], dtype=np.float32)
    for go_id in mf_terms], axis=1)
valid_mask = Y_te.sum(0) >= 2
mf_valid = [go for go, v in zip(mf_terms, valid_mask) if v]
Y_te_v = Y_te[:, valid_mask]
print(f"valid MF terms: {len(mf_valid)}", flush=True)

X30_te = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)
X30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)

tr_genes = [clean(g) for g in np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)]
tr_id_set = set(sym2id.get(g, g) for g in tr_genes)
go_tr = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606' or p[7] != 'Function': continue
        if p[1] in tr_id_set: go_tr[p[2]].add(p[1])

Xtr_n = normalize(X30_tr, axis=1)
Xte_n = normalize(X30_te, axis=1)

centroid_preds = np.zeros((n_iso, len(mf_valid)), dtype=np.float32)
for j, go_id in enumerate(mf_valid):
    pos_gene_ids = go_tr.get(go_id, set())
    pos_idx = [i for i, g in enumerate(tr_genes) if sym2id.get(g, g) in pos_gene_ids]
    if not pos_idx:
        continue
    c = Xtr_n[pos_idx].mean(0, keepdims=True)
    c /= (np.linalg.norm(c) + 1e-8)
    centroid_preds[:, j] = (Xte_n @ c.T).ravel()


def macro_ap(P):
    aps = []
    for j in range(P.shape[1]):
        y = Y_te_v[:, j]
        if y.sum() < 1 or y.sum() == len(y):
            continue
        aps.append(average_precision_score(y, P[:, j]))
    return float(np.mean(aps)), len(aps)


c_macro, nA = macro_ap(centroid_preds)
prism = np.load(PRISM_PREDS).astype(np.float32)
if prism.shape[1] != len(mf_valid):
    prism = prism[:, valid_mask]
p_macro, _ = macro_ap(prism)

res = {'centroid_sim_macro_AUPRC': c_macro, 'v17f_star_macro_AUPRC_sanity': p_macro,
       'n_terms': nA, 'reference': {'knn_retrieval': 0.636, 'gene_mean_retrieval': 0.465,
                                    'v17f_star_manuscript': 0.734, 'centroid_DR_AUC': 0.638, 'prism_DR_AUC': 0.630}}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, 'w'), indent=2)
print("\n=== centroid-sim MACRO AUPRC (Option B q①) ===")
print(f"  centroid-sim MACRO AUPRC : {c_macro:.4f}   (n={nA} terms)")
print(f"  v17f* MACRO (sanity)     : {p_macro:.4f}   (manuscript 0.734)")
print(f"  reference: k-NN 0.636 | gene-mean 0.465 | v17f* 0.734")
print(f"  --> centroid-sim on DR ties PRISM (0.638 vs 0.630); on MACRO it is {c_macro:.3f}")
print(f"[saved] {OUT}")
