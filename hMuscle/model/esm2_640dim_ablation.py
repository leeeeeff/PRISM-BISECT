# -*- coding: utf-8 -*-
"""
esm2_640dim_ablation.py

640-dim ESM-2 (human-only) 위에서 v9-1 피처 기여도 3-way ablation
gene-stratified GroupKFold (K=5) CV on test set

Variants:
  A: 640-dim ESM-2 only
  B: 640-dim + domain_delta_v2 sign (251-dim)   → 891-dim
  C: 640-dim + domain_delta_v2 + splicing_delta_v2 (150-dim)  → 1041-dim

Bonus:
  A_train→test: train(31k human) → test evaluation (sanity vs Option B 0.561)

비교 기준:
  Option B (human-only, train→test): Macro 0.561
  v8b best ensemble: Macro 0.370
"""

import os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# utils_Full.MAPPING_DICT: ENSG00000xxxxx → gene_symbol
from utils_Full import MAPPING_DICT

# ---- Config ----
GO_TERMS = [
    'GO:0007204', 'GO:0030017', 'GO:0006941', 'GO:0003774', 'GO:0006096'
]
K_FOLDS = 5
DATA_DIR = '../data'
FEAT_DIR = '../results_isoform/features'
RESULTS_FILE = 'esm2_640dim_ablation_results.txt'
ANNOT_DIR = '../data/raw_data/data/annotations/'

# Reference numbers from status report (Section 3.2)
OPTB_AUPRC = {
    'GO:0007204': 0.414, 'GO:0030017': 0.561, 'GO:0006941': 0.312,
    'GO:0003774': 0.825, 'GO:0006096': 0.695
}
V8B_AUPRC = {
    'GO:0007204': 0.146, 'GO:0030017': 0.157, 'GO:0006941': 0.118,
    'GO:0003774': 0.569, 'GO:0006096': 0.795
}


# ============================================================
# Utils
# ============================================================
def load_npy(path, label=""):
    arr = np.load(path, allow_pickle=True).astype(np.float32)
    tag = label or os.path.basename(path)
    print("  [load] {} {} dtype={} range=[{:.2f}, {:.2f}]".format(
        tag, arr.shape, arr.dtype, arr.min(), arr.max()))
    return arr

def load_ids(path):
    raw = np.load(path, allow_pickle=True)
    return [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in raw]

def ensg_to_symbol(gid):
    """ENSG00000xxx.14 → gene symbol via MAPPING_DICT. Falls back to gid itself."""
    base = gid.split('.')[0]
    return MAPPING_DICT.get(base, MAPPING_DICT.get(gid, gid))

def get_positive_genes(go_term):
    """Collect positive gene symbols from human + swissprot annotation files."""
    pos = set()
    for fname in ['human_annotations.txt', 'swissprot_annotations.txt']:
        fpath = os.path.join(ANNOT_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > 1 and go_term in parts[1:]:
                    pos.add(parts[0])
    return pos

def make_labels_test(test_gene_ids, positive_genes):
    """
    Label test isoforms via ENSG→symbol mapping.
    Consecutive identical gIDs (= isoforms of same gene) inherit the same label.
    Returns (y, group_arr) where group_arr contains gene symbols for GroupKFold.
    """
    labels, groups = [], []
    last_gid, last_label, last_sym = None, 0, None
    for gid in test_gene_ids:
        if gid != last_gid:
            sym = ensg_to_symbol(gid)
            last_label = 1 if sym in positive_genes else 0
            last_gid, last_sym = gid, sym
        labels.append(last_label)
        groups.append(last_sym)
    return np.array(labels, dtype=np.int32), np.array(groups)

def make_labels_train(train_gene_ids, positive_genes):
    """Train IDs are already gene symbols (NAT2, ADA, ...). No mapping needed."""
    return np.array([1 if g in positive_genes else 0 for g in train_gene_ids], dtype=np.int32)

def lr_auprc_auroc(X_tr, y_tr, X_te, y_te):
    """Fit balanced LR on train, evaluate on test."""
    if y_tr.sum() == 0 or y_te.sum() == 0:
        return np.nan, np.nan
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    clf = LogisticRegression(class_weight='balanced', max_iter=1000,
                             solver='lbfgs', C=1.0, n_jobs=-1)
    clf.fit(X_tr_s, y_tr)
    prob = clf.predict_proba(X_te_s)[:, 1]
    return average_precision_score(y_te, prob), roc_auc_score(y_te, prob)

def cv_auprc_auroc(X, y, groups, k=5):
    """Gene-stratified GroupKFold CV — returns mean AUPRC, AUROC across folds."""
    gkf = GroupKFold(n_splits=k)
    auprc_list, auroc_list = [], []
    for tr_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        ap, auroc = lr_auprc_auroc(X_tr, y_tr, X_val, y_val)
        if not np.isnan(ap):
            auprc_list.append(ap)
            auroc_list.append(auroc)
    return (np.mean(auprc_list) if auprc_list else np.nan,
            np.mean(auroc_list) if auroc_list else np.nan,
            len(auprc_list))


# ============================================================
# Load data
# ============================================================
print("=" * 70)
print(" ESM-2 640-dim Ablation  |  gene-stratified 5-fold CV on test set")
print("=" * 70)

print("\n[1] Test set features...")
X_esm2   = load_npy(os.path.join(DATA_DIR, 'esm2_embeddings_t30_150M.npy'), "ESM2_test(640)")
X_dd_raw = load_npy(os.path.join(FEAT_DIR, 'domain_delta_v2.npy'), "dd_v2(251)")
X_splice = load_npy(os.path.join(FEAT_DIR, 'splicing/splicing_delta_v2.npy'), "splice_v2(150)")

X_dd_sign = np.sign(X_dd_raw).astype(np.float32)
print("  [sign] dd_v2 → unique values: {}".format(sorted(np.unique(X_dd_sign).tolist())))

print("\n[2] Test IDs...")
test_gene_ids = load_ids('my_gene_list_fixed.npy')
test_gene_arr = np.array(test_gene_ids)
N = len(test_gene_ids)
print("  N_test={} unique_genes={}".format(N, len(set(test_gene_ids))))

# Shape sanity
assert X_esm2.shape[0] == N, f"ESM2 N mismatch: {X_esm2.shape[0]} vs {N}"
assert X_dd_sign.shape[0] == N, f"dd N mismatch"
assert X_splice.shape[0] == N, f"splice N mismatch"

# Feature matrices for 3 variants
feat_A = X_esm2                                                    # 640
feat_B = np.concatenate([X_esm2, X_dd_sign],         axis=1)      # 891
feat_C = np.concatenate([X_esm2, X_dd_sign, X_splice], axis=1)    # 1041
print("\n  Variant dims: A={}, B={}, C={}".format(
    feat_A.shape[1], feat_B.shape[1], feat_C.shape[1]))

print("\n[3] Train data (for sanity check)...")
train_path = os.path.join(DATA_DIR, 'esm2_train_human.npy')
if not os.path.exists(train_path):
    train_path = os.path.join(DATA_DIR, 'esm2_train_human_t30_150M.npy')
if os.path.exists(train_path):
    X_train = load_npy(train_path, "ESM2_train(640)")
    train_gene_ids = load_ids('../data/raw_data/data/id_lists/train_gene_list.npy')
    do_train_test = True
    print("  train N={} unique_genes={}".format(len(train_gene_ids), len(set(train_gene_ids))))
else:
    do_train_test = False
    print("  [skip] train ESM-2 not found")


# ============================================================
# Run ablation
# ============================================================
print("\n" + "=" * 70)
print(" Running experiments...")
print("=" * 70)

results = []

for go_term in GO_TERMS:
    print("\n[GO] {}".format(go_term))
    t_start = time.time()

    pos_genes = get_positive_genes(go_term)
    y_test, group_arr = make_labels_test(test_gene_ids, pos_genes)
    n_pos = int(y_test.sum())
    random_auprc = n_pos / N

    row = {'go': go_term, 'n_pos': n_pos, 'N': N, 'random': random_auprc}

    if n_pos == 0:
        print("  [SKIP] no positives in test set (pos_genes={})".format(len(pos_genes)))
        for k in ['A', 'B', 'C', 'A_auroc', 'B_auroc', 'C_auroc', 'A_tt', 'A_tt_auroc']:
            row[k] = np.nan
        results.append(row)
        continue

    print("  y_test: pos={} / {} ({:.3f}%) | pos_genes={} | unique_groups={}".format(
        n_pos, N, 100 * random_auprc, len(pos_genes), len(set(group_arr))))

    # ---- Variant A, B, C: gene-stratified CV on test set ----
    for variant, X_feat in [('A', feat_A), ('B', feat_B), ('C', feat_C)]:
        t1 = time.time()
        ap, auroc, n_folds = cv_auprc_auroc(X_feat, y_test, group_arr, k=K_FOLDS)
        row[variant] = ap
        row[variant + '_auroc'] = auroc
        print("  [{} CV({} folds)] AUPRC={:.4f} AUROC={:.4f}  ({:.1f}s)".format(
            variant, n_folds, ap, auroc, time.time() - t1))

    # ---- Bonus: train→test for A only (sanity vs Option B 0.561) ----
    if do_train_test:
        t1 = time.time()
        y_train = make_labels_train(train_gene_ids, pos_genes)
        ap_tt, auroc_tt = lr_auprc_auroc(X_train, y_train, X_esm2, y_test)
        row['A_tt'] = ap_tt
        row['A_tt_auroc'] = auroc_tt
        print("  [A train→test]         AUPRC={:.4f} AUROC={:.4f}  ({:.1f}s)".format(
            ap_tt, auroc_tt, time.time() - t1))
    else:
        row['A_tt'] = np.nan
        row['A_tt_auroc'] = np.nan

    print("  [Total {:.1f}s]".format(time.time() - t_start))
    results.append(row)


# ============================================================
# Summary
# ============================================================
def fmt(v):
    return "{:.4f}".format(v) if not (v is None or np.isnan(v)) else "  NaN "

lines = []
lines.append("\n" + "=" * 108)
lines.append(" ESM-2 640-dim Ablation Results")
lines.append(" gene-stratified GroupKFold (K={}) CV on test set".format(K_FOLDS))
lines.append("=" * 108)
lines.append("{:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8} {:>8} {:>8}".format(
    "GO term",
    "A_AUPRC", "B_AUPRC", "C_AUPRC",
    "A_AUROC", "B_AUROC", "C_AUROC",
    "A_train→test",
    "OptB_ref", "v8b_ref", "Random"))
lines.append("-" * 108)

macro = {k: [] for k in ['A', 'B', 'C', 'A_tt']}

for r in results:
    go = r['go']
    for k in ['A', 'B', 'C', 'A_tt']:
        v = r.get(k, np.nan)
        if not np.isnan(v):
            macro[k].append(v)
    line = "{:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>12} {:>8} {:>8} {:>8}".format(
        go,
        fmt(r.get('A')),   fmt(r.get('B')),   fmt(r.get('C')),
        fmt(r.get('A_auroc')), fmt(r.get('B_auroc')), fmt(r.get('C_auroc')),
        fmt(r.get('A_tt')),
        fmt(OPTB_AUPRC.get(go)), fmt(V8B_AUPRC.get(go)), fmt(r['random']))
    lines.append(line)

lines.append("-" * 108)
macro_A  = np.mean(macro['A'])  if macro['A']  else np.nan
macro_B  = np.mean(macro['B'])  if macro['B']  else np.nan
macro_C  = np.mean(macro['C'])  if macro['C']  else np.nan
macro_tt = np.mean(macro['A_tt']) if macro['A_tt'] else np.nan
macro_optb = np.mean(list(OPTB_AUPRC.values()))
macro_v8b  = np.mean(list(V8B_AUPRC.values()))

lines.append("{:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>12} {:>8} {:>8}".format(
    "MACRO",
    fmt(macro_A), fmt(macro_B), fmt(macro_C),
    "", "", "",
    fmt(macro_tt),
    fmt(macro_optb), fmt(macro_v8b)))
lines.append("=" * 108)

lines.append("\n[Interpretation]")
lines.append("  A (640 only, CV)      Macro={:.4f}".format(macro_A))
lines.append("  B (640+dd, CV)        Macro={:.4f}  Δ vs A: {:+.4f}".format(macro_B, macro_B - macro_A))
lines.append("  C (640+dd+spl, CV)    Macro={:.4f}  Δ vs A: {:+.4f}  Δ vs B: {:+.4f}".format(
    macro_C, macro_C - macro_A, macro_C - macro_B))
if not np.isnan(macro_tt):
    lines.append("  A (train→test)        Macro={:.4f}  [ref: Option B honest = {:.3f}]".format(
        macro_tt, macro_optb))
    lines.append("  CV-overfit factor: A_CV/A_tt = {:.2f}x".format(macro_A / macro_tt if macro_tt > 0 else float('nan')))
lines.append("")
lines.append("  Decision rules:")
lines.append("  B > A + 0.01 → domain_delta_v2 (canonical 교정) adds value beyond ESM-2")
lines.append("  C > B + 0.01 → splicing_delta_v2 adds value beyond ESM-2 + domain")
lines.append("  A_CV ≫ A_train→test → intra-test CV is optimistic (compare C_CV fairly)")
lines.append("  A_CV ≈ A_train→test → CV results are reliable estimates")

output = "\n".join(lines)
print("\n\n" + output)

with open(RESULTS_FILE, 'w') as f:
    f.write(output)
print("\n[Saved] {}".format(RESULTS_FILE))
