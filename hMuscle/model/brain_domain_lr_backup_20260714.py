#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_domain_lr.py
------------------
Brain zero-shot Domain LR for 82 MF terms.
Train: muscle domain matrix + muscle GO labels
Test:  same test isoforms (36748), evaluated with brain GO labels

Comparison with muscle result (mf_domain_vs_prism.tsv):
  muscle domain_lr macro = 0.1625
  muscle prism     macro = 0.5962

Goal: compute brain-side domain_lr to show consistent PRISM dominance.
"""

import os, sys, json, gzip
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/v_expanded_gomf'

# ── Load MF 82 terms from existing results ───────────────────────────────────
print("[1] Loading MF 82 terms from mf_domain_vs_prism.tsv...")
mf_terms = []
mf_prism_brain = {}
with open(f'{OUT_DIR}/mf_domain_vs_prism.tsv') as f:
    header = f.readline().strip().split('\t')
    for line in f:
        p = line.strip().split('\t')
        if len(p) < 8: continue
        go_id = p[0]
        prism_brain = float(p[5]) if p[5] else None  # prism_brain column
        mf_terms.append(go_id)
        mf_prism_brain[go_id] = prism_brain
print(f"  {len(mf_terms)} MF terms loaded")

# ── sym2id mapping ────────────────────────────────────────────────────────────
print("[2] Building sym → Entrez ID map...")
sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id:
                        sym2id[syn] = p[1]
print(f"  {len(sym2id):,} symbols mapped")

# ── gene2go — both training-restricted (Y_tr) and all (Y_te brain) ───────────
print("[3] Parsing gene2go...")
go_genes_tr  = defaultdict(set)
go_genes_all = defaultdict(set)

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_syms = [str(g).replace("b'","").replace("'","").strip() for g in tr_genes_raw]
tr_ids  = [sym2id.get(s, s) for s in tr_syms]
tr_id_set = set(tr_ids)

with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        gid, go_id, cat_raw = p[1], p[2], p[7]
        if cat_raw != 'Function': continue  # MF only
        go_genes_all[go_id].add(gid)
        if gid in tr_id_set:
            go_genes_tr[go_id].add(gid)
print(f"  MF terms in gene2go: {len(go_genes_all)}")

# ── Gene ID lists ─────────────────────────────────────────────────────────────
print("[4] Loading gene ID lists...")
# Train
tr_sym2idx = defaultdict(list)
for i, s in enumerate(tr_syms):
    tr_sym2idx[s].append(i)

# Test: ENSG → symbol
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

def clean_sym(raw):
    s = str(raw)
    for ch in ["b'", "'", '"', ' ']: s = s.replace(ch, '')
    return s

te_gene_raw = [clean_sym(x) for x in np.load('my_gene_list_fixed.npy', allow_pickle=True)]
te_sym_list = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_gene_raw]
n_mapped = sum(1 for g in te_gene_raw if g.split('.')[0] in ENSG2SYM)
print(f"  Test genes: {len(te_sym_list)}, mapped: {n_mapped}")

# ── Label builders ────────────────────────────────────────────────────────────
def build_labels_train(go_id):
    pos_ids  = go_genes_tr[go_id]
    pos_syms = {s for s, gid in zip(tr_syms, tr_ids) if gid in pos_ids}
    y = np.zeros(len(tr_syms), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []):
            y[idx] = 1.0
    return y

def build_labels_test_brain(go_id):
    pos_ids = go_genes_all[go_id]
    return np.array([
        1.0 if sym2id.get(s, '__NONE__') in pos_ids else 0.0
        for s in te_sym_list
    ], dtype=np.float32)

# ── Domain features ───────────────────────────────────────────────────────────
print("[5] Loading domain matrices...")
X_tr_dom = np.load(f'{FEAT_DIR}/domain_matrix_proper_train.npy').astype(np.float32)
X_te_dom = np.load(f'{FEAT_DIR}/domain_matrix_proper_test.npy').astype(np.float32)
print(f"  Train: {X_tr_dom.shape}  Test: {X_te_dom.shape}")

scaler = MaxAbsScaler()
X_tr_s = scaler.fit_transform(X_tr_dom)
X_te_s = scaler.transform(X_te_dom)

# ── Run Domain LR for each MF term ───────────────────────────────────────────
print("\n[6] Brain Domain LR for 82 MF terms...\n")
print(f"{'go_id':<14} {'n_pos_tr':>9} {'n_pos_te_brain':>15} {'domain_lr_brain':>16} {'prism_brain':>12} {'Δ':>8}")
print("-" * 80)

results = []
auprc_domain_list = []
auprc_prism_list  = []

for go_id in mf_terms:
    y_tr = build_labels_train(go_id)
    y_te = build_labels_test_brain(go_id)

    n_pos_tr = int(y_tr.sum())
    n_pos_te = int(y_te.sum())
    prism_brain = mf_prism_brain.get(go_id)

    if n_pos_tr < 5 or n_pos_te < 2:
        print(f"{go_id:<14} {n_pos_tr:>9} {n_pos_te:>15} {'SKIP':>16}")
        results.append({'go_id': go_id, 'n_pos_tr': n_pos_tr, 'n_pos_te_brain': n_pos_te,
                        'domain_lr_brain': None, 'prism_brain': prism_brain})
        continue

    clf = LogisticRegression(class_weight='balanced', C=1.0, max_iter=500,
                             solver='liblinear', random_state=42)
    clf.fit(X_tr_s, y_tr)
    proba = clf.predict_proba(X_te_s)[:, 1]
    auprc = float(average_precision_score(y_te, proba))

    delta = (prism_brain - auprc) if prism_brain else float('nan')
    auprc_domain_list.append(auprc)
    if prism_brain:
        auprc_prism_list.append(prism_brain)

    print(f"{go_id:<14} {n_pos_tr:>9} {n_pos_te:>15} {auprc:>16.4f} {prism_brain:>12.4f} {delta:>+8.4f}")
    results.append({'go_id': go_id, 'n_pos_tr': n_pos_tr, 'n_pos_te_brain': n_pos_te,
                    'domain_lr_brain': auprc, 'prism_brain': prism_brain, 'delta': delta})

# ── Summary ───────────────────────────────────────────────────────────────────
valid = [r for r in results if r['domain_lr_brain'] is not None]
macro_domain_brain = float(np.mean([r['domain_lr_brain'] for r in valid]))
macro_prism_brain  = float(np.mean([r['prism_brain'] for r in valid if r['prism_brain']]))
n_prism_wins = sum(1 for r in valid if r['prism_brain'] and r['prism_brain'] > r['domain_lr_brain'])

print("-" * 80)
print(f"\n{'='*65}")
print(f"  BRAIN ZERO-SHOT COMPARISON (MF {len(valid)} valid terms)")
print(f"{'='*65}")
print(f"  Domain LR (brain labels):  {macro_domain_brain:.4f}")
print(f"  PRISM (brain zero-shot):   {macro_prism_brain:.4f}")
print(f"  Δ PRISM - Domain LR:      +{macro_prism_brain - macro_domain_brain:.4f}")
print(f"  PRISM wins: {n_prism_wins}/{len(valid)}")
print(f"\n  Muscle comparison (from mf_domain_vs_prism_summary.json):")
print(f"  Domain LR (muscle):        0.1625")
print(f"  PRISM (muscle):            0.5962")
print(f"  Δ muscle:                 +0.4337")

# ── Save ──────────────────────────────────────────────────────────────────────
out_json = {
    'n_valid_terms': len(valid),
    'macro_domain_lr_brain': macro_domain_brain,
    'macro_prism_brain': macro_prism_brain,
    'delta_prism_vs_domain_brain': macro_prism_brain - macro_domain_brain,
    'prism_wins_brain': n_prism_wins,
    'muscle_domain_lr': 0.1625,
    'muscle_prism': 0.5962,
    'per_term': results
}
out_path = f'{OUT_DIR}/brain_domain_lr_summary.json'
json.dump(out_json, open(out_path, 'w'), indent=2)
print(f"\n  [Saved] {out_path}")
