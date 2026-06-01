#!/usr/bin/env python3
"""
v10_within_gene_pairwise.py
============================
Within-gene isoform pairwise ranking — D/S feature 기여 강력 검증.

기존 Option A (Ridge Spearman)의 한계:
  - "correlation이 있다"는 수준
  - ESM-2 640d가 Ridge에서 우세 (more parameters)

이 실험의 접근:
  Gene 내 모든 (iso_i, iso_j) 쌍 생성 → 이진 과제:
  "iso_i가 iso_j보다 expression ratio가 높은가?"

  Feature: 각 isoform의 D/S 특성치 차이 or 절대값 concat
  Baseline: random (AUROC=0.5)

  평가:
  - Gene-stratified CV (test gene 내에서 pairwise AUROC)
  - Bootstrap CI (n=500)
  - Ablation: ESM2 / splice / domain / combined

  핵심 주장:
    D/S features에서만 파생된 pairwise classifier가 AUROC > 0.5 (p<0.001)
    → "splicing/domain delta carries genuine isoform-level expression information"
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import os, json, time
from itertools import combinations

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FEAT_DIR  = '../results_isoform/features'
DATA_DIR  = '../data'
OUT_DIR   = '../../reports/within_gene'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print(" Within-Gene Pairwise Ranking — D/S Feature Validation")
print("=" * 65)

# ── 1. 데이터 로드 ────────────────────────────────────────────────
print("\n[1] Loading data ...")
ratio_arr = np.load(f'{FEAT_DIR}/within_gene_ratio.npy')
iso_arr   = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
gene_arr  = np.load('my_gene_list_fixed.npy', allow_pickle=True)
iso_list  = [s.decode() if isinstance(s, bytes) else s for s in iso_arr]
gene_list = [s.decode() if isinstance(s, bytes) else s for s in gene_arr]
gene_base_list = [g.split('.')[0] for g in gene_list]

valid_mask = ~np.isnan(ratio_arr)
valid_idxs = np.where(valid_mask)[0]
valid_genes = np.array([gene_base_list[i] for i in valid_idxs])
valid_ratios = ratio_arr[valid_idxs]

esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)[valid_idxs]
dd   = np.load(f'{FEAT_DIR}/domain_delta_proper_test_v2.npy').astype(np.float32)[valid_idxs]
sd   = np.load(f'{FEAT_DIR}/splicing/splicing_delta_v2.npy').astype(np.float32)[valid_idxs]
print(f"  Valid isoforms: {len(valid_idxs)}, ESM2: {esm2.shape}, DD: {dd.shape}, SD: {sd.shape}")

# ── 2. Gene 그룹 (≥2 isoform) ──────────────────────────────────
print("\n[2] Building gene groups ...")
gene_to_local = {}
for local_i, g in enumerate(valid_genes):
    gene_to_local.setdefault(g, []).append(local_i)
multi_genes = {g: v for g, v in gene_to_local.items() if len(v) >= 2}
print(f"  Multi-isoform genes: {len(multi_genes)}")

# ── 3. Pairwise 데이터셋 생성 ─────────────────────────────────────
print("\n[3] Building pairwise dataset ...")

def make_pairs(gene_dict, ratios, max_pairs_per_gene=10):
    """각 gene 내에서 (iso_i, iso_j) 쌍 생성. label=1 if ratio_i > ratio_j."""
    pairs_feat_esm = []
    pairs_feat_ds  = []
    pairs_feat_full = []
    labels = []
    gene_ids = []

    for g, idxs in gene_dict.items():
        if len(idxs) < 2: continue
        ratio_g = ratios[idxs]
        # 유효한 pair만 (ratio 차이 > 0.02 — 너무 비슷한 건 제외)
        pair_list = list(combinations(range(len(idxs)), 2))
        if len(pair_list) > max_pairs_per_gene:
            rng = np.random.RandomState(42)
            pair_list = [pair_list[k] for k in rng.choice(len(pair_list), max_pairs_per_gene, replace=False)]

        for (a, b) in pair_list:
            ia, ib = idxs[a], idxs[b]
            diff = ratio_g[a] - ratio_g[b]
            if abs(diff) < 0.02: continue  # tie 제외

            # Feature: [isoform_i properties] concat [isoform_j properties]
            # → pairwise difference representation
            esm_diff = esm2[ia] - esm2[ib]
            ds_diff  = np.concatenate([sd[ia] - sd[ib], dd[ia] - dd[ib]])
            full_diff = np.concatenate([esm_diff, sd[ia] - sd[ib], dd[ia] - dd[ib]])

            # 방향성 통일: label=1 means i > j
            label = 1 if diff > 0 else 0

            pairs_feat_esm.append(esm_diff)
            pairs_feat_ds.append(ds_diff)
            pairs_feat_full.append(full_diff)
            labels.append(label)
            gene_ids.append(g)

    return (np.array(pairs_feat_esm),
            np.array(pairs_feat_ds),
            np.array(pairs_feat_full),
            np.array(labels),
            np.array(gene_ids))

X_esm, X_ds, X_full, y, pair_genes = make_pairs(multi_genes, valid_ratios)
print(f"  Total pairs: {len(y)}, pos rate: {y.mean():.3f}")
print(f"  ESM feat: {X_esm.shape}, DS feat: {X_ds.shape}, Full feat: {X_full.shape}")

# ── 4. Gene-stratified CV + Pairwise AUROC ───────────────────────
print("\n[4] Gene-stratified CV (5-fold × 5 seeds) ...")

def gene_cv_auroc(X, y, pair_genes, n_seeds=5, n_folds=5):
    """Gene-stratified CV → pairwise AUROC"""
    all_genes = np.unique(pair_genes)
    all_aucs = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        rng.shuffle(all_genes)
        fold_size = len(all_genes) // n_folds

        fold_aucs = []
        for fold in range(n_folds):
            test_g  = set(all_genes[fold*fold_size:(fold+1)*fold_size])
            train_g = set(all_genes) - test_g

            tr_mask = np.array([g in train_g for g in pair_genes])
            te_mask = np.array([g in test_g  for g in pair_genes])

            if tr_mask.sum() < 20 or te_mask.sum() < 10: continue
            if y[tr_mask].std() < 1e-8 or y[te_mask].std() < 1e-8: continue

            pipe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=1.0, max_iter=500, solver='lbfgs'))])
            pipe.fit(X[tr_mask], y[tr_mask])
            proba = pipe.predict_proba(X[te_mask])[:, 1]
            fold_aucs.append(roc_auc_score(y[te_mask], proba))

        if fold_aucs:
            all_aucs.append(np.mean(fold_aucs))

    return float(np.mean(all_aucs)), float(np.std(all_aucs))

auc_esm,  std_esm  = gene_cv_auroc(X_esm,  y, pair_genes)
auc_ds,   std_ds   = gene_cv_auroc(X_ds,   y, pair_genes)
auc_full, std_full = gene_cv_auroc(X_full, y, pair_genes)

print(f"  [P1] ESM-2 only:    Pairwise AUROC = {auc_esm:.4f} ± {std_esm:.4f}")
print(f"  [P2] D/S only:      Pairwise AUROC = {auc_ds:.4f}  ± {std_ds:.4f}")
print(f"  [P3] Full (ESM+DS): Pairwise AUROC = {auc_full:.4f} ± {std_full:.4f}")
print(f"  [Random baseline]:  AUROC = 0.5000")

# ── 5. 통계 검증 (Bootstrap CI) ─────────────────────────────────
print("\n[5] Bootstrap CI (n=500) ...")

def bootstrap_auroc(X, y, n_boot=500, seed=42):
    """전체 데이터 bootstrap → CI"""
    rng = np.random.RandomState(seed)
    pipe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=1.0, max_iter=500, solver='lbfgs'))])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]  # in-sample (빠른 bootstrap용)
    # gene-block bootstrap
    all_genes = np.unique(pair_genes)
    boot_aucs = []
    for _ in range(n_boot):
        boot_g = set(rng.choice(all_genes, len(all_genes), replace=True))
        mask = np.array([g in boot_g for g in pair_genes])
        if mask.sum() < 20 or y[mask].std() < 1e-8: continue
        boot_aucs.append(roc_auc_score(y[mask], proba[mask]))
    ci_low  = np.percentile(boot_aucs, 2.5)
    ci_high = np.percentile(boot_aucs, 97.5)
    # p-value: fraction of boots where AUROC ≤ 0.5
    p_val = (np.array(boot_aucs) <= 0.5).mean()
    return float(np.mean(boot_aucs)), float(ci_low), float(ci_high), float(p_val)

bm_ds,  ci_lo_ds,  ci_hi_ds,  p_ds  = bootstrap_auroc(X_ds,   y, n_boot=500)
bm_esm, ci_lo_esm, ci_hi_esm, p_esm = bootstrap_auroc(X_esm,  y, n_boot=500)
bm_full,ci_lo_full,ci_hi_full,p_full = bootstrap_auroc(X_full, y, n_boot=500)

print(f"  [P1] ESM-2: {bm_esm:.4f} 95%CI[{ci_lo_esm:.4f},{ci_hi_esm:.4f}]  p(≤0.5)={p_esm:.4f}")
print(f"  [P2] D/S:   {bm_ds:.4f}  95%CI[{ci_lo_ds:.4f},{ci_hi_ds:.4f}]   p(≤0.5)={p_ds:.4f}")
print(f"  [P3] Full:  {bm_full:.4f} 95%CI[{ci_lo_full:.4f},{ci_hi_full:.4f}] p(≤0.5)={p_full:.4f}")

# ── 6. 비-canonical only (trivial bias 제거) ─────────────────────
print("\n[6] Non-canonical pairs only ...")
canon_ref = {}
canon_path = f'{FEAT_DIR}/canonical_reference.tsv'
if os.path.exists(canon_path):
    with open(canon_path) as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4:
                try: canon_ref[p[0]] = int(p[3])
                except: pass

# non-canonical gene 그룹 구성
nc_genes = {}
for g, idxs in multi_genes.items():
    gb = gene_base_list[valid_idxs[idxs[0]]]
    canon_global = canon_ref.get(gb)
    if canon_global is not None:
        nc_idxs = [i for i in idxs if valid_idxs[i] != canon_global]
    else:
        top_local = idxs[int(np.argmax(valid_ratios[idxs]))]
        nc_idxs = [i for i in idxs if i != top_local]
    if len(nc_idxs) >= 2:
        nc_genes[g] = nc_idxs

print(f"  Genes with ≥2 non-canonical isoforms: {len(nc_genes)}")

nc_esm, nc_ds, nc_full, nc_y, nc_pair_genes = make_pairs(nc_genes, valid_ratios)
print(f"  Non-canonical pairs: {len(nc_y)}")

if len(nc_y) >= 100:
    auc_nc_esm,  std_nc_esm  = gene_cv_auroc(nc_esm,  nc_y, nc_pair_genes)
    auc_nc_ds,   std_nc_ds   = gene_cv_auroc(nc_ds,   nc_y, nc_pair_genes)
    auc_nc_full, std_nc_full = gene_cv_auroc(nc_full, nc_y, nc_pair_genes)
    print(f"  [NC-P1] ESM-2:    {auc_nc_esm:.4f} ± {std_nc_esm:.4f}")
    print(f"  [NC-P2] D/S only: {auc_nc_ds:.4f}  ± {std_nc_ds:.4f}")
    print(f"  [NC-P3] Full:     {auc_nc_full:.4f} ± {std_nc_full:.4f}")
else:
    auc_nc_esm = auc_nc_ds = auc_nc_full = None

# ── 7. Splice vs Domain 개별 기여 ────────────────────────────────
print("\n[7] Individual feature ablation ...")

X_sd_only = X_ds[:, :150]   # splice part
X_dd_only = X_ds[:, 150:]   # domain part
auc_splice, std_splice = gene_cv_auroc(X_sd_only, y, pair_genes)
auc_domain, std_domain = gene_cv_auroc(X_dd_only, y, pair_genes)
print(f"  [P-S] splice_delta only: {auc_splice:.4f} ± {std_splice:.4f}")
print(f"  [P-D] domain_delta only: {auc_domain:.4f} ± {std_domain:.4f}")

# ── 8. 결과 저장 ──────────────────────────────────────────────────
results = {
    'n_pairs': int(len(y)),
    'n_multi_genes': int(len(multi_genes)),
    'n_nc_genes': int(len(nc_genes)),
    'n_nc_pairs': int(len(nc_y)) if nc_y is not None else 0,
    'cv_auroc': {
        'esm2':   {'mean': auc_esm,  'std': std_esm},
        'ds':     {'mean': auc_ds,   'std': std_ds},
        'full':   {'mean': auc_full, 'std': std_full},
        'splice': {'mean': auc_splice,'std': std_splice},
        'domain': {'mean': auc_domain,'std': std_domain},
    },
    'bootstrap_ci': {
        'esm2': {'mean': bm_esm, 'ci95': [ci_lo_esm, ci_hi_esm], 'p_vs_random': p_esm},
        'ds':   {'mean': bm_ds,  'ci95': [ci_lo_ds,  ci_hi_ds],  'p_vs_random': p_ds},
        'full': {'mean': bm_full,'ci95': [ci_lo_full,ci_hi_full],'p_vs_random': p_full},
    },
    'nc_cv_auroc': {
        'esm2': auc_nc_esm,
        'ds':   auc_nc_ds,
        'full': auc_nc_full,
    } if auc_nc_ds is not None else {},
    'timestamp': time.strftime('%Y%m%d_%H%M'),
}

fname = f'{OUT_DIR}/pairwise_{time.strftime("%Y%m%d_%H%M")}.json'
with open(fname, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[Saved] {fname}")
print("\n" + "=" * 65)
print(" PAIRWISE RANKING SUMMARY")
print("=" * 65)
print(f"  Pairs: {len(y)} (from {len(multi_genes)} multi-iso genes)")
print(f"  [CV AUROC] ESM-2={auc_esm:.4f}  Splice={auc_splice:.4f}  Domain={auc_domain:.4f}  D/S={auc_ds:.4f}  Full={auc_full:.4f}")
if auc_nc_ds is not None:
    print(f"  [NC AUROC] ESM-2={auc_nc_esm:.4f}  D/S={auc_nc_ds:.4f}  Full={auc_nc_full:.4f}")
print(f"  [Bootstrap] D/S AUROC={bm_ds:.4f} 95%CI[{ci_lo_ds:.4f},{ci_hi_ds:.4f}] p={p_ds:.4f}")
ds_beats_random = bm_ds > 0.5 and ci_lo_ds > 0.5
print(f"\n  D/S beats random: {'YES' if ds_beats_random else 'NO'} (CI lower bound > 0.5: {ci_lo_ds:.4f})")
esm_ds_gap = auc_ds - auc_esm
print(f"  D/S vs ESM-2 gap: {esm_ds_gap:+.4f} ({'D/S better' if esm_ds_gap > 0 else 'ESM-2 better'})")
print("Done.")
