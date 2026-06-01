# -*- coding: utf-8 -*-
"""
esm2_dim_sweep.py

ESM-2 640-dim에서 PCA 차원 축소 후 LR AUPRC 포화 곡선 측정
목적: 파이프라인의 Phase 1 ESM-2 projection 목표 차원 결정

평가 방식: train (31K human-only) → test (동일 split as Option B=0.561)
비교 기준: 640-dim full (0.561) vs 파이프라인 64-dim (0.370)

차원 테스트: [32, 64, 96, 128, 192, 256, 384, 512, 640]
추가 분석:
  - PCA explained variance curve
  - 각 GO term별 포화점 (AUPRC ≥ 0.95 × max 최소 dim)
"""

import os, sys, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_Full import MAPPING_DICT

GO_TERMS = [
    'GO:0007204', 'GO:0030017', 'GO:0006941', 'GO:0003774', 'GO:0006096'
]
DIMS      = [32, 64, 96, 128, 192, 256, 384, 512, 640]
DATA_DIR  = '../data'
ANNOT_DIR = '../data/raw_data/data/annotations/'
RESULTS_FILE = 'esm2_dim_sweep_results.txt'

# Reference AUPRC for comparison
V8B_AUPRC  = {'GO:0007204':0.146,'GO:0030017':0.157,'GO:0006941':0.118,
               'GO:0003774':0.569,'GO:0006096':0.795}
OPTB_AUPRC = {'GO:0007204':0.414,'GO:0030017':0.561,'GO:0006941':0.312,
               'GO:0003774':0.825,'GO:0006096':0.695}


# ---- Utils ----
def load_npy(path, label=""):
    arr = np.load(path, allow_pickle=True).astype(np.float32)
    tag = label or os.path.basename(path)
    print("  [load] {} {}".format(tag, arr.shape))
    return arr

def load_ids(path):
    raw = np.load(path, allow_pickle=True)
    return [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in raw]

def ensg_to_symbol(gid):
    base = gid.split('.')[0]
    return MAPPING_DICT.get(base, MAPPING_DICT.get(gid, gid))

def get_positive_genes(go_term):
    pos = set()
    with open(os.path.join(ANNOT_DIR, 'human_annotations.txt')) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    return pos

def make_labels_train(train_gene_ids, pos_genes):
    return np.array([1 if g in pos_genes else 0
                     for g in train_gene_ids], dtype=np.int32)

def make_labels_test(test_gene_ids, pos_genes):
    labels, last_gid, last_label = [], None, 0
    for gid in test_gene_ids:
        if gid != last_gid:
            sym = ensg_to_symbol(gid)
            last_label = 1 if sym in pos_genes else 0
            last_gid = gid
        labels.append(last_label)
    return np.array(labels, dtype=np.int32)

def lr_auprc(X_tr, y_tr, X_te, y_te):
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


# ---- Load data ----
print("=" * 65)
print(" ESM-2 Dimension Sweep — PCA + LR AUPRC (train→test)")
print(" Dims: {}".format(DIMS))
print("=" * 65)

print("\n[1] Load ESM-2 embeddings...")
X_train = load_npy(os.path.join(DATA_DIR, 'esm2_train_human.npy'),       "train(640)")
X_test  = load_npy(os.path.join(DATA_DIR, 'esm2_embeddings_t30_150M.npy'),"test(640)")

print("\n[2] Load IDs...")
train_gene_ids = load_ids('../data/raw_data/data/id_lists/train_gene_list.npy')
test_gene_ids  = load_ids('my_gene_list_fixed.npy')
print("  train N={} unique_genes={}".format(
    len(train_gene_ids), len(set(train_gene_ids))))
print("  test  N={} unique_genes={}".format(
    len(test_gene_ids), len(set(test_gene_ids))))


# ---- PCA fit on train ----
print("\n[3] PCA on train (640-dim)...")
t0 = time.time()
pca_full = PCA(n_components=640, random_state=42)
pca_full.fit(X_train)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
print("  PCA fit: {:.1f}s".format(time.time()-t0))
print("  Explained variance:")
for n in [32, 64, 96, 128, 192, 256, 384, 512, 640]:
    print("    {:3d} PCs → {:.2f}%".format(n, cumvar[n-1]*100))

# Find minimum dim for common thresholds
for thr in [0.90, 0.95, 0.99]:
    n_min = int(np.searchsorted(cumvar, thr)) + 1
    print("  {:.0f}% variance → {} PCs".format(thr*100, n_min))


# ---- Dim sweep ----
print("\n[4] Dimension sweep (train→test LR)...")
results = {go: {} for go in GO_TERMS}
y_tests, y_trains = {}, {}

for go_term in GO_TERMS:
    pos = get_positive_genes(go_term)
    y_trains[go_term] = make_labels_train(train_gene_ids, pos)
    y_tests[go_term]  = make_labels_test(test_gene_ids, pos)
    n_pos = int(y_tests[go_term].sum())
    print("  {} pos={}/{}".format(go_term, n_pos, len(y_tests[go_term])))

for dim in DIMS:
    t_dim = time.time()
    # Project: fit PCA on train, transform both
    pca = PCA(n_components=dim, random_state=42)
    X_tr_pca = pca.fit_transform(X_train)
    X_te_pca = pca.transform(X_test)

    print("\n  --- dim={} ---".format(dim))
    for go_term in GO_TERMS:
        t1 = time.time()
        ap, auroc = lr_auprc(X_tr_pca, y_trains[go_term],
                             X_te_pca,  y_tests[go_term])
        results[go_term][dim] = (ap, auroc)
        ref = OPTB_AUPRC.get(go_term, 0)
        pct = (ap / ref * 100) if ref > 0 else 0
        print("    {} AUPRC={:.4f} ({:5.1f}% of 640d ref) AUROC={:.4f}  ({:.1f}s)".format(
            go_term, ap, pct, auroc, time.time()-t1))

    print("  dim={} total {:.1f}s".format(dim, time.time()-t_dim))


# ---- Summary ----
lines = []
lines.append("\n" + "=" * 100)
lines.append(" ESM-2 Dimension Sweep Results (PCA + balanced LR, human-only train→test)")
lines.append("=" * 100)

# Header
hdr = "{:<14}".format("GO term")
for dim in DIMS:
    hdr += " {:>7}".format("d={}".format(dim))
hdr += "  {:>7}  {:>7}".format("640ref", "v8b_ref")
lines.append(hdr)
lines.append("-" * 100)

macro_per_dim = {dim: [] for dim in DIMS}

for go_term in GO_TERMS:
    row = "{:<14}".format(go_term)
    for dim in DIMS:
        ap = results[go_term].get(dim, (np.nan,))[0]
        row += " {:>7.4f}".format(ap) if not np.isnan(ap) else " {:>7}".format("NaN")
        macro_per_dim[dim].append(ap)
    row += "  {:>7.4f}  {:>7.4f}".format(
        OPTB_AUPRC.get(go_term, 0), V8B_AUPRC.get(go_term, 0))
    lines.append(row)

lines.append("-" * 100)
macro_row = "{:<14}".format("MACRO")
for dim in DIMS:
    vals = [v for v in macro_per_dim[dim] if not np.isnan(v)]
    m = np.mean(vals) if vals else np.nan
    macro_row += " {:>7.4f}".format(m) if not np.isnan(m) else " {:>7}".format("NaN")
macro_row += "  {:>7.4f}  {:>7.4f}".format(
    np.mean(list(OPTB_AUPRC.values())),
    np.mean(list(V8B_AUPRC.values())))
lines.append(macro_row)
lines.append("=" * 100)

# PCA variance summary
lines.append("\nPCA Explained Variance:")
for n in DIMS:
    lines.append("  {:3d} PCs → {:.2f}%".format(n, cumvar[n-1]*100))

# Saturation analysis per GO term (95% of 640-dim AUPRC)
lines.append("\nSaturation Analysis (first dim to reach ≥95% of 640-dim AUPRC):")
for go_term in GO_TERMS:
    ref640 = results[go_term].get(640, (np.nan,))[0]
    if np.isnan(ref640):
        continue
    thr = ref640 * 0.95
    sat_dim = None
    for dim in DIMS:
        ap = results[go_term].get(dim, (np.nan,))[0]
        if not np.isnan(ap) and ap >= thr:
            sat_dim = dim
            break
    lines.append("  {}: 640d={:.4f}, 95%thr={:.4f}, saturation@dim={}".format(
        go_term, ref640, thr, sat_dim))

# Pipeline recommendation
macro640 = np.mean([results[g].get(640,(np.nan,))[0] for g in GO_TERMS
                    if not np.isnan(results[g].get(640,(np.nan,))[0])])
lines.append("\n[Pipeline Dim Recommendation]")
for dim in DIMS:
    vals = [results[g].get(dim,(np.nan,))[0] for g in GO_TERMS]
    vals = [v for v in vals if not np.isnan(v)]
    m = np.mean(vals) if vals else np.nan
    pct_of_640 = (m / macro640 * 100) if macro640 > 0 else 0
    lines.append("  dim={:3d}: Macro={:.4f}  ({:.1f}% of 640-dim)".format(
        dim, m, pct_of_640))
lines.append("  v8b pipeline (64-dim non-PCA): Macro=0.3570  (63.6% of 640-dim via PCA)")
lines.append("  Target: dim where Macro ≥ 90% of 640-dim PCA → use this as Phase 1 output size")

output = "\n".join(lines)
print("\n" + output)
with open(RESULTS_FILE, 'w') as f:
    f.write(output)
print("\n[Saved] {}".format(RESULTS_FILE))
