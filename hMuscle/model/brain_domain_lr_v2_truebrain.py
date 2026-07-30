#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_domain_lr_v2_truebrain.py
----------------------------------
TISSUE-MISLABELING BUGFIX RERUN (2026-07-14).
Original brain_domain_lr.py's OWN internal print statement says "BRAIN ZERO-SHOT COMPARISON"
but loads my_gene_list_fixed.npy / domain_matrix_proper_test.npy -- MUSCLE data (36748
isoforms) -- this was the CLEAREST case of the tissue-mislabeling bug, producing the
manuscript's "brain Domain LR 0.0319" figure. This rerun re-points the TEST side at the
TRUE brain isoform set: brain_full_gene_names.npy (63994 isoforms / 18514 unique genes,
symbols not ENSG) and domain_matrix_brain_full.npy (built in Step 2 of this rerun, same
512-Pfam vocab as domain_matrix_proper_test.npy). ADDITIONALLY: the original script's
"prism_brain" comparison column was read from a STALE mf_domain_vs_prism.tsv column that
itself holds the OLD muscle-mislabeled-as-brain PRISM numbers -- this rerun instead computes
the comparison directly from the TRUE-brain v17f* predictions
(reports/truebrain_rerun_20260714/v17f_star_bootstrap/v17f_star_preds.npy + Y_te.npy,
produced by v17f_star_bootstrap_ci_v2_truebrain.py), so the PRISM reference is also corrected,
not just the Domain LR side. Training side (train_gene_list.npy, domain_matrix_proper_train.npy)
is UNCHANGED.

Original (mislabeled-as-brain, actually muscle): brain_domain_lr.py, backed up at
brain_domain_lr_backup_20260714.py before this rerun.

Brain zero-shot Domain LR for 82 MF terms.
Train: muscle domain matrix + muscle GO labels
Test:  TRUE brain isoforms (63994), evaluated with brain GO labels

Comparison with muscle result (mf_domain_vs_prism.tsv, OLD/mislabeled reference):
  muscle domain_lr macro = 0.1625
  muscle prism     macro = 0.5962  (this was actually muscle-as-"brain", now corrected)

Goal: compute TRUE brain-side domain_lr AND true brain-side PRISM (v17f*) to show
whether PRISM dominance over Domain LR still holds on genuinely cross-tissue data.
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
BRAIN_DIR = '../data/brain_isoquant_esm2/full'
FEAT_DIR  = '../results_isoform/features'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/truebrain_rerun_20260714/v_expanded_gomf'
os.makedirs(OUT_DIR, exist_ok=True)
V17F_STAR_DIR = '../../reports/truebrain_rerun_20260714/v17f_star_bootstrap'
TERMS_SRC_DIR = '../../reports/v_expanded_gomf'  # term LIST is tissue-independent, reused

# ── Load MF 82 terms from existing results (term identities are tissue-independent) ──
print("[1] Loading MF term list from mf_domain_vs_prism.tsv (term IDs only, tissue-independent)...")
mf_terms = []
mf_prism_brain_STALE = {}  # OLD muscle-mislabeled-as-brain column -- kept only for old-vs-new print
with open(f'{TERMS_SRC_DIR}/mf_domain_vs_prism.tsv') as f:
    header = f.readline().strip().split('\t')
    for line in f:
        p = line.strip().split('\t')
        if len(p) < 8: continue
        go_id = p[0]
        prism_brain_stale = float(p[5]) if p[5] else None  # OLD muscle-mislabeled column
        mf_terms.append(go_id)
        mf_prism_brain_STALE[go_id] = prism_brain_stale
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

# TRUE BRAIN: gene names already symbols (e.g. 'A1BG'), no ENSG2SYM mapping needed.
te_sym_list = [clean_sym(x) for x in np.load(f'{BRAIN_DIR}/brain_full_gene_names.npy', allow_pickle=True)]
print(f"  Test genes (TRUE BRAIN): {len(te_sym_list)}")

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
X_tr_dom = np.load(f'{FEAT_DIR}/domain_matrix_proper_train.npy').astype(np.float32)  # train UNCHANGED
X_te_dom = np.load(f'{FEAT_DIR}/domain_matrix_brain_full.npy').astype(np.float32)    # TRUE BRAIN, same 512-Pfam vocab
print(f"  Train: {X_tr_dom.shape}  Test (TRUE BRAIN): {X_te_dom.shape}")

scaler = MaxAbsScaler()
X_tr_s = scaler.fit_transform(X_tr_dom)
X_te_s = scaler.transform(X_te_dom)

# ── Load TRUE BRAIN v17f* predictions (NOT the stale TSV column) ─────────────
# The original script's "prism_brain" reference came from mf_domain_vs_prism.tsv's
# prism_brain column, itself populated by the muscle-mislabeled-as-brain pipeline.
# This rerun instead computes the true-brain PRISM (v17f*) per-term AUPRC directly
# from v17f_star_bootstrap_ci_v2_truebrain.py's saved predictions, which are already
# aligned to brain_full_gene_names.npy / mf_terms (same term list, same row order).
print("[5b] Loading TRUE BRAIN v17f* predictions for PRISM comparison...")
v17f_star_preds = np.load(f'{V17F_STAR_DIR}/v17f_star_preds.npy').astype(np.float32)  # (63994, len(mf_terms))
Y_te_v17f       = np.load(f'{V17F_STAR_DIR}/Y_te.npy').astype(np.float32)             # (63994, len(mf_terms))
print(f"  v17f* preds: {v17f_star_preds.shape}  Y_te: {Y_te_v17f.shape}")

mf_prism_brain_TRUE = {}
for j, go_id in enumerate(mf_terms):
    if Y_te_v17f[:, j].sum() >= 2:
        mf_prism_brain_TRUE[go_id] = float(average_precision_score(Y_te_v17f[:, j], v17f_star_preds[:, j]))
    else:
        mf_prism_brain_TRUE[go_id] = None
print(f"  Computed TRUE-brain PRISM AUPRC for {sum(1 for v in mf_prism_brain_TRUE.values() if v is not None)} valid terms")

# ── Run Domain LR for each MF term ───────────────────────────────────────────
print("\n[6] TRUE Brain Domain LR for MF terms...\n")
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
    prism_brain = mf_prism_brain_TRUE.get(go_id)

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

    prism_disp = f"{prism_brain:.4f}" if prism_brain is not None else "N/A"
    delta_disp = f"{delta:+.4f}" if prism_brain is not None else "N/A"
    print(f"{go_id:<14} {n_pos_tr:>9} {n_pos_te:>15} {auprc:>16.4f} {prism_disp:>12} {delta_disp:>8}")
    results.append({'go_id': go_id, 'n_pos_tr': n_pos_tr, 'n_pos_te_brain': n_pos_te,
                    'domain_lr_brain': auprc, 'prism_brain': prism_brain, 'delta': delta})

# ── Summary ───────────────────────────────────────────────────────────────────
valid = [r for r in results if r['domain_lr_brain'] is not None]
macro_domain_brain = float(np.mean([r['domain_lr_brain'] for r in valid]))
macro_prism_brain  = float(np.mean([r['prism_brain'] for r in valid if r['prism_brain']]))
n_prism_wins = sum(1 for r in valid if r['prism_brain'] and r['prism_brain'] > r['domain_lr_brain'])

print("-" * 80)
print(f"\n{'='*65}")
print(f"  TRUE BRAIN ZERO-SHOT COMPARISON (MF {len(valid)} valid terms)")
print(f"{'='*65}")
print(f"  Domain LR (TRUE brain labels):  {macro_domain_brain:.4f}")
print(f"  PRISM (TRUE brain zero-shot):   {macro_prism_brain:.4f}")
print(f"  Δ PRISM - Domain LR:           +{macro_prism_brain - macro_domain_brain:.4f}")
print(f"  PRISM wins: {n_prism_wins}/{len(valid)}")
print(f"\n  OLD reference (muscle-mislabeled-as-brain, from mf_domain_vs_prism.tsv / this")
print(f"  project's memory session_20260627_reviewer_response.md):")
print(f"  Domain LR (muscle, mislabeled 'brain'):  0.1625")
print(f"  PRISM   (muscle, mislabeled 'brain'):     0.5962")
print(f"  Δ (muscle-mislabeled):                   +0.4337")
print(f"  Reported 'brain Domain LR' in manuscript (also muscle-mislabeled): 0.0319")

# ── Save ──────────────────────────────────────────────────────────────────────
out_json = {
    'n_valid_terms': len(valid),
    'macro_domain_lr_truebrain': macro_domain_brain,
    'macro_prism_truebrain': macro_prism_brain,
    'delta_prism_vs_domain_truebrain': macro_prism_brain - macro_domain_brain,
    'prism_wins_truebrain': n_prism_wins,
    'OLD_muscle_mislabeled_domain_lr': 0.1625,
    'OLD_muscle_mislabeled_prism': 0.5962,
    'OLD_muscle_mislabeled_manuscript_brain_domain_lr_figure': 0.0319,
    'per_term': results
}
out_path = f'{OUT_DIR}/brain_domain_lr_summary.json'
json.dump(out_json, open(out_path, 'w'), indent=2)
print(f"\n  [Saved] {out_path}")
