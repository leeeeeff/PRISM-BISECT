"""
v10_xgb_bias_check.py — XGBoost bias_score & fair comparison
============================================================
목적: DA 비판 대응 — XGB > v10-B 주장의 3가지 공정성 검증

1. bias_score: XGB가 gene-level shortcut에 의존하는가?
   bias_score = within_gene_score_std / global_score_std
   < 0.10: gene-level shortcut 의심
   > 0.30: isoform-specific

2. Protocol fair comparison: v10-B 5-seed bootstrap CI 계산
   (vs XGB single-seed 비교의 통계적 공정성)

3. within-gene ranking: GO-positive gene 내에서
   isoform ranking이 의미있는가?

출력:
  reports/xgb_baseline/xgb_bias_check_{ts}.json
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
ts = time.strftime('%Y%m%d_%H%M')
OUT_DIR   = '../../reports/xgb_baseline'
os.makedirs(OUT_DIR, exist_ok=True)

SEED    = 42
N_BOOT  = 500
np.random.seed(SEED)

# Same GO terms as v10_xgb_baseline
GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0030017': 'Sarcomere org',
    'GO:0006941': 'Muscle contraction',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0003774': 'Motor activity',
    'GO:0006096': 'Glycolysis',
}

# v10-B known values (multi-seed ensemble)
V10B_KNOWN = {
    'GO:0007204': 0.7653, 'GO:0030017': 0.7426, 'GO:0006941': 0.5968,
    'GO:0006914': 0.6397, 'GO:0043161': 0.7174, 'GO:0007519': 0.7250,
    'GO:0042692': 0.6526, 'GO:0055074': 0.7255, 'GO:0007005': 0.6624,
    'GO:0007517': 0.7017, 'GO:0032006': 0.6023, 'GO:0003774': 0.8128,
    'GO:0006096': 0.6712,
}

print("=" * 65)
print(" XGBoost Bias Check & Fair Comparison")
print("=" * 65)

# Load data
X_tr_esm2 = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te_esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

te_isoid  = load_ids('my_isoform_list_fixed.npy')
te_geneid = load_ids('my_gene_list_fixed.npy')
tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

te_sym      = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_geneid]
te_genebase = [g.split('.')[0] for g in te_geneid]
tr_sym      = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_geneid]
te_genes    = np.array(te_genebase)

print(f"  Train: {len(tr_geneid)}, Test: {len(te_isoid)}")

def load_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te, pos

def bootstrap_ci(y_true, y_score, gene_ids, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    unique_genes = np.unique(gene_ids)
    base_auprc = float(average_precision_score(y_true, y_score))
    gene_to_idxs = defaultdict(list)
    for i, g in enumerate(gene_ids):
        gene_to_idxs[g].append(i)
    gene_to_idxs = {g: np.array(idxs) for g, idxs in gene_to_idxs.items()}
    boot = []
    for _ in range(n_boot):
        gs = rng.choice(unique_genes, size=len(unique_genes), replace=True)
        idx = np.concatenate([gene_to_idxs[g] for g in gs])
        if y_true[idx].sum() == 0: continue
        boot.append(average_precision_score(y_true[idx], y_score[idx]))
    boot = np.array(boot)
    return base_auprc, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def compute_bias_score(y_te, scores, gene_ids):
    """
    bias_score = within_gene_std / global_std
    Low (<0.1): gene-level shortcut (within-gene rankings are uniform)
    High (>0.3): isoform-specific differentiation
    """
    global_std = scores.std()
    if global_std < 1e-10:
        return np.nan

    # Per-gene within-gene std
    gene_to_idxs_te = defaultdict(list)
    for i, g in enumerate(gene_ids):
        gene_to_idxs_te[g].append(i)

    within_stds = []
    for g, idxs in gene_to_idxs_te.items():
        if len(idxs) < 2:
            continue
        within_stds.append(scores[idxs].std())

    if len(within_stds) == 0:
        return np.nan

    return float(np.mean(within_stds) / global_std)


def compute_within_gene_ranking_quality(y_te, scores, gene_ids, pos_set):
    """
    For GO-positive genes with ≥2 isoforms:
    Does the model rank the 'canonical' (most expressed?) isoform higher?
    Proxy: within-gene score variance for positive vs negative genes.
    """
    gene_to_idxs_te = defaultdict(list)
    for i, g in enumerate(gene_ids):
        gene_to_idxs_te[g].append(i)

    pos_within_stds = []
    neg_within_stds = []
    te_sym_arr = np.array(te_sym)

    for g, idxs in gene_to_idxs_te.items():
        if len(idxs) < 2:
            continue
        gene_syms = set(te_sym_arr[idxs])
        is_pos = any(s in pos_set for s in gene_syms)
        wstd = scores[idxs].std()
        if is_pos:
            pos_within_stds.append(wstd)
        else:
            neg_within_stds.append(wstd)

    return {
        'pos_within_std': float(np.mean(pos_within_stds)) if pos_within_stds else np.nan,
        'neg_within_std': float(np.mean(neg_within_stds)) if neg_within_stds else np.nan,
        'n_pos_genes': len(pos_within_stds),
        'n_neg_genes': len(neg_within_stds),
    }


# Prepare scaled data once
sc_global = StandardScaler()
X_tr_sc = sc_global.fit_transform(X_tr_esm2)
X_te_sc = sc_global.transform(X_te_esm2)

if HAS_XGB:
    # Use same params as v10_xgb_baseline
    pass

all_results = {}

print("\n" + "=" * 65)
print(" Bias Score Analysis per GO Term")
print("=" * 65)
print(f"\n{'GO Term':<20} {'BiasXGB':>9} {'BiasLR':>9} {'AUPRC_XGB':>10} {'AUPRC_v10B':>11} {'CI_overlap':>12}")
print("-" * 78)

for go, go_name in GO_TERMS.items():
    y_tr, y_te, pos_set = load_labels(go)
    if y_tr.sum() < 10 or y_te.sum() < 5:
        continue

    pos_weight = (y_tr == 0).sum() / max(y_tr.sum(), 1)

    # XGB prediction
    if HAS_XGB:
        clf_xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=float(pos_weight), eval_metric='aucpr',
            use_label_encoder=False, random_state=SEED, n_jobs=4, verbosity=0,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        clf_xgb = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                             learning_rate=0.05, random_state=SEED)

    clf_xgb.fit(X_tr_sc, y_tr)
    xgb_scores = clf_xgb.predict_proba(X_te_sc)[:, 1]

    # LR prediction
    clf_lr = LogisticRegression(class_weight='balanced', C=1.0,
                                 max_iter=1000, random_state=SEED)
    clf_lr.fit(X_tr_sc, y_tr)
    lr_scores = clf_lr.predict_proba(X_te_sc)[:, 1]

    # AUPRC with CI
    xgb_auprc, xgb_lo, xgb_hi = bootstrap_ci(y_te, xgb_scores, te_genes)
    lr_auprc, lr_lo, lr_hi     = bootstrap_ci(y_te, lr_scores, te_genes)

    # Bias scores
    bias_xgb = compute_bias_score(y_te, xgb_scores, te_genes)
    bias_lr  = compute_bias_score(y_te, lr_scores,  te_genes)

    # Within-gene analysis
    wgr_xgb = compute_within_gene_ranking_quality(y_te, xgb_scores, te_genes, pos_set)
    wgr_lr  = compute_within_gene_ranking_quality(y_te, lr_scores,  te_genes, pos_set)

    # CI overlap with v10-B known
    v10b_val = V10B_KNOWN.get(go, np.nan)
    ci_overlap = (xgb_lo <= v10b_val <= xgb_hi) if not np.isnan(v10b_val) else None

    print(f"{go_name:<20} {bias_xgb:>9.4f} {bias_lr:>9.4f} "
          f"{xgb_auprc:>10.4f} {v10b_val:>11.4f} "
          f"{'OVERLAP' if ci_overlap else 'SEPARATE':>12}")

    all_results[go] = {
        'go_name': go_name,
        'bias_xgb': round(bias_xgb, 4) if not np.isnan(bias_xgb) else None,
        'bias_lr': round(bias_lr, 4) if not np.isnan(bias_lr) else None,
        'xgb_auprc': round(xgb_auprc, 4),
        'xgb_ci': [round(xgb_lo, 4), round(xgb_hi, 4)],
        'lr_auprc': round(lr_auprc, 4),
        'lr_ci': [round(lr_lo, 4), round(lr_hi, 4)],
        'v10b_auprc': v10b_val,
        'v10b_in_xgb_ci': ci_overlap,
        'xgb_within_gene': wgr_xgb,
        'lr_within_gene': wgr_lr,
    }

print("-" * 78)

# Summary
bias_xgb_vals = [v['bias_xgb'] for v in all_results.values() if v['bias_xgb'] is not None]
bias_lr_vals  = [v['bias_lr']  for v in all_results.values() if v['bias_lr']  is not None]
overlap_count = sum(1 for v in all_results.values() if v['v10b_in_xgb_ci'])

print(f"\n  Mean bias_score XGB: {np.mean(bias_xgb_vals):.4f}")
print(f"  Mean bias_score LR : {np.mean(bias_lr_vals):.4f}")
print(f"  v10-B in XGB CI    : {overlap_count}/{len(all_results)} GO terms")

interpretation = "GENE-LEVEL SHORTCUT" if np.mean(bias_xgb_vals) < 0.15 else \
                 ("ISOFORM-SPECIFIC" if np.mean(bias_xgb_vals) > 0.30 else "MIXED")
print(f"\n  XGB bias interpretation: {interpretation}")

if overlap_count > len(all_results) // 2:
    print(f"  → v10-B and XGB are NOT statistically different (CI overlap {overlap_count}/13)")
else:
    print(f"  → XGB significantly outperforms v10-B (CI overlap only {overlap_count}/13)")

# Save
out_path = f'{OUT_DIR}/xgb_bias_check_{ts}.json'
with open(out_path, 'w') as f:
    json.dump({'results': all_results, 'timestamp': ts,
               'summary': {
                   'mean_bias_xgb': round(float(np.mean(bias_xgb_vals)), 4),
                   'mean_bias_lr':  round(float(np.mean(bias_lr_vals)), 4),
                   'v10b_in_xgb_ci_count': overlap_count,
                   'n_go_terms': len(all_results),
               }}, f, indent=2)
print(f"\nFINAL: {out_path}")
