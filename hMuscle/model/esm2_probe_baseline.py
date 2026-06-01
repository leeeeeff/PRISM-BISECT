# -*- coding: utf-8 -*-
# esm2_probe_baseline.py
# Option A: ESM-2 + balanced LR (human + swissprot)
# Option B: ESM-2 + balanced LR (human only)
# Purpose: establish what the complex PFN pipeline adds over a simple linear probe

import os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils_Full import generate_label

# ------------------------------------------------------------------
# GO terms (same as v8 experiments)
# ------------------------------------------------------------------
GO_TERMS = [
    'GO:0007204',
    'GO:0030017',
    'GO:0006941',
    'GO:0003774',
    'GO:0006096',
]

# ------------------------------------------------------------------
# v8b results for comparison (from previous experiments)
# ------------------------------------------------------------------
V8B_AUPRC = {
    'GO:0007204': 0.1462,
    'GO:0030017': 0.1787,
    'GO:0006941': 0.1177,
    'GO:0003774': None,
    'GO:0006096': None,
}

ORACLE_AUPRC = {
    'GO:0007204': 0.2312,
    'GO:0030017': 0.2979,
    'GO:0006941': 0.1825,
    'GO:0003774': None,
    'GO:0006096': None,
}

DATA_DIR = '../data'

def load(name):
    path = os.path.join(DATA_DIR, name)
    arr = np.load(path).astype(np.float32)
    print("  Loaded {} {}".format(name, arr.shape))
    return arr

def load_ids(path):
    return [x.decode('utf-8') if isinstance(x, bytes) else x
            for x in np.load(path, allow_pickle=True)]


# ------------------------------------------------------------------
# Load all data once
# ------------------------------------------------------------------
print("=" * 60)
print(" ESM-2 Linear Probe Baseline")
print(" Option A: human + swissprot | Option B: human only")
print("=" * 60)

print("\n[1] Loading ESM-2 embeddings (t30_150M)...")
X_train_human  = load('esm2_train_human_t30_150M.npy')        # (31668, 640)
X_train_swiss  = load('esm2_train_swissprot_t30_150M.npy')    # (82703, 640)
X_test_esm2    = load('esm2_embeddings_t30_150M.npy')         # (36748, 640)

X_train_human_mask  = load('esm2_train_human_t30_150M_mask.npy')
X_train_swiss_mask  = load('esm2_train_swissprot_t30_150M_mask.npy')
X_test_esm2_mask    = load('esm2_embeddings_t30_150M_mask.npy')

print("\n[2] Loading gene/isoform IDs...")
X_train_geneid       = load_ids('../data/raw_data/data/id_lists/train_gene_list.npy')
X_train_geneid_swiss = load_ids('../data/raw_data/data/id_lists/train_swissprot_list.npy')
X_test_geneid        = load_ids('my_gene_list_fixed.npy')
X_test_isoid         = load_ids('my_isoform_list_fixed.npy')

print("\n[3] Loading annotations...")
ANNOT_DIR = '../data/raw_data/data/annotations/'


# ------------------------------------------------------------------
# Run baseline for each GO term
# ------------------------------------------------------------------
results = []

for go_term in GO_TERMS:
    print("\n" + "=" * 60)
    print(" GO term: {}".format(go_term))
    print("=" * 60)

    # Load positive gene set
    positive_Gene = []
    for fname in ['human_annotations.txt', 'swissprot_annotations.txt']:
        fpath = ANNOT_DIR + fname
        if not os.path.exists(fpath):
            print("  WARNING: {} not found".format(fpath))
            continue
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if go_term in parts[1:]:
                    positive_Gene.append(parts[0])
    print("  positive_Gene: {} genes".format(len(positive_Gene)))

    # ---- Option A: human + swissprot combined ----
    print("\n  [Option A] human + swissprot combined...")
    y_train_A, y_test, _, _, _, _, X_esm_A_comb, _ = generate_label(
        X_train_human, X_train_human_mask,
        X_train_swiss, X_train_swiss_mask,
        X_train_geneid, X_train_geneid_swiss,
        X_test_geneid, positive_Gene)

    n_pos_A = int((y_train_A == 1).sum())
    n_pos_test = int((y_test == 1).sum())
    print("  y_train_A: pos={} neg={} ratio={:.3f}%".format(
        n_pos_A, int((y_train_A == 0).sum()), n_pos_A / len(y_train_A) * 100))
    print("  y_test:    pos={} neg={} ratio={:.3f}%".format(
        n_pos_test, int((y_test == 0).sum()), n_pos_test / len(y_test) * 100))

    random_auprc = n_pos_test / len(y_test) if len(y_test) > 0 else 0

    if n_pos_A == 0 or n_pos_test == 0:
        print("  SKIP: no positives in train or test")
        results.append({'go': go_term, 'auprc_A': None, 'auprc_B': None,
                        'auroc_A': None, 'auroc_B': None, 'random': random_auprc})
        continue

    # X_esm_A_comb is mask from generate_label, not ESM2 — call again with ESM2
    _, _, _, _, _, _, X_train_esm2_A, _ = generate_label(
        X_train_human, X_train_human,
        X_train_swiss, X_train_swiss,
        X_train_geneid, X_train_geneid_swiss,
        X_test_geneid, positive_Gene)

    t0 = time.time()
    scaler_A = StandardScaler()
    X_tr_A_scaled = scaler_A.fit_transform(X_train_esm2_A)
    X_te_scaled   = scaler_A.transform(X_test_esm2)

    clf_A = LogisticRegression(class_weight='balanced', max_iter=1000,
                               solver='lbfgs', C=1.0, n_jobs=-1)
    clf_A.fit(X_tr_A_scaled, y_train_A)
    proba_A = clf_A.predict_proba(X_te_scaled)[:, 1]

    auprc_A = average_precision_score(y_test, proba_A)
    auroc_A = roc_auc_score(y_test, proba_A)
    print("  Option A AUPRC={:.4f} AUROC={:.4f} ({:.1f}s)".format(
        auprc_A, auroc_A, time.time() - t0))

    # ---- Option B: human only ----
    print("\n  [Option B] human only...")
    dummy_swiss   = np.zeros((0, X_train_swiss.shape[1]), dtype=np.float32)
    dummy_ids     = []

    y_train_B, _, _, _, _, _, X_train_esm2_B, _ = generate_label(
        X_train_human, X_train_human,
        dummy_swiss, dummy_swiss,
        X_train_geneid, dummy_ids,
        X_test_geneid, positive_Gene)

    n_pos_B = int((y_train_B == 1).sum())
    print("  y_train_B: pos={} neg={} ratio={:.3f}%".format(
        n_pos_B, int((y_train_B == 0).sum()), n_pos_B / len(y_train_B) * 100))

    t0 = time.time()
    scaler_B = StandardScaler()
    X_tr_B_scaled = scaler_B.fit_transform(X_train_esm2_B)
    X_te_B_scaled = scaler_B.transform(X_test_esm2)

    clf_B = LogisticRegression(class_weight='balanced', max_iter=1000,
                               solver='lbfgs', C=1.0, n_jobs=-1)
    clf_B.fit(X_tr_B_scaled, y_train_B)
    proba_B = clf_B.predict_proba(X_te_B_scaled)[:, 1]

    auprc_B = average_precision_score(y_test, proba_B)
    auroc_B = roc_auc_score(y_test, proba_B)
    print("  Option B AUPRC={:.4f} AUROC={:.4f} ({:.1f}s)".format(
        auprc_B, auroc_B, time.time() - t0))

    results.append({
        'go': go_term,
        'auprc_A': auprc_A, 'auroc_A': auroc_A,
        'auprc_B': auprc_B, 'auroc_B': auroc_B,
        'random': random_auprc,
    })


# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
print("\n" + "=" * 90)
print(" RESULTS SUMMARY")
print("=" * 90)
print("{:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}".format(
    "GO term", "A_AUPRC", "A_AUROC", "B_AUPRC", "B_AUROC",
    "v8b", "Oracle", "Random"))
print("-" * 90)

for r in results:
    go = r['go']
    def fmt(v): return "{:.4f}".format(v) if v is not None else "  N/A "
    v8b_val  = V8B_AUPRC.get(go)
    orc_val  = ORACLE_AUPRC.get(go)
    print("{:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10.4f}".format(
        go,
        fmt(r['auprc_A']), fmt(r['auroc_A']),
        fmt(r['auprc_B']), fmt(r['auroc_B']),
        fmt(v8b_val), fmt(orc_val),
        r['random']))

print("=" * 90)
print("\nInterpretation:")
print("  A_AUPRC > v8b  → pipeline hurts vs simple LR (SwissProt may help)")
print("  B_AUPRC > A    → SwissProt causes distribution shift")
print("  B_AUPRC > v8b  → pipeline adds no value over human-only ESM-2 LR")
print("  Oracle >> A,B  → problem is train distribution, not model complexity")
