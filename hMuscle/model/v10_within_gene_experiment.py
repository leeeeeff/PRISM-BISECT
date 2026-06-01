#!/usr/bin/env python3
"""
v10_within_gene_experiment.py  [Option A]
==========================================
Within-gene isoform expression ratio 예측 실험.

핵심 질문:
  - domain_delta + splice_delta가 within-gene ratio 예측에 기여하는가?
  - ESM-2 abs는 이 task에서 유용한가?
  - trivial bias (canonical=delta0=dominant) 제거 후에도 신호 유지되는가?

평가: Gene-stratified Spearman correlation
      (각 gene 내에서 predicted_score vs ratio Spearman → macro 평균)

Ablation:
  [1] |splice_delta| alone  (trivial proxy)
  [2] |domain_delta| alone
  [3] ESM-2 abs alone
  [4] splice + domain (D/S only)
  [5] ESM-2 + splice + domain (full)
  [6] non-canonical only — trivial bias 제거 후 D/S 기여 확인
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os, json, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FEAT_DIR  = '../results_isoform/features'
DATA_DIR  = '../data'
COUNTS    = '../data/counts_transcript.txt'
OUT_DIR   = '../../reports/within_gene'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print(" Within-Gene Expression Ratio Prediction Experiment [Option A]")
print("=" * 65)

# ── 1. ratio 로드 ────────────────────────────────────────────────
print("\n[1] Loading within-gene ratios ...")
ratio_arr = np.load(f'{FEAT_DIR}/within_gene_ratio.npy')
iso_arr   = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
gene_arr  = np.load('my_gene_list_fixed.npy', allow_pickle=True)
iso_list  = [s.decode() if isinstance(s, bytes) else s for s in iso_arr]
gene_list = [s.decode() if isinstance(s, bytes) else s for s in gene_arr]

# NaN = 우리 counts에 없는 isoform
valid_mask = ~np.isnan(ratio_arr)
valid_idxs = np.where(valid_mask)[0]
print(f"  Valid isoforms (with ratio): {valid_mask.sum()} / {len(ratio_arr)}")

# gene 매핑
gene_base_list = [g.split('.')[0] for g in gene_list]
valid_genes = [gene_base_list[i] for i in valid_idxs]
valid_ratios = ratio_arr[valid_idxs]

# ── 2. Feature 로드 ───────────────────────────────────────────────
print("\n[2] Loading features ...")
esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)[valid_idxs]
dd   = np.load(f'{FEAT_DIR}/domain_delta_proper_test_v2.npy').astype(np.float32)[valid_idxs]
sd   = np.load(f'{FEAT_DIR}/splicing/splicing_delta_v2.npy').astype(np.float32)[valid_idxs]

print(f"  ESM-2: {esm2.shape}, domain_delta: {dd.shape}, splice_delta: {sd.shape}")

# scalar proxy features
esm2_norm = np.linalg.norm(esm2, axis=1, keepdims=True)
dd_norm   = np.abs(dd).sum(axis=1, keepdims=True)
sd_norm   = np.abs(sd).sum(axis=1, keepdims=True)

# ── 3. Gene 그룹 구성 (≥2 isoform인 gene만) ─────────────────────
print("\n[3] Grouping by gene ...")
gene_to_idxs = {}
for i, g in enumerate(valid_genes):
    gene_to_idxs.setdefault(g, []).append(i)
multi_genes = {g: idxs for g, idxs in gene_to_idxs.items() if len(idxs) >= 2}
print(f"  Multi-isoform genes: {len(multi_genes)}")

# ── 4. Canonical isoform 정보 ─────────────────────────────────────
print("\n[4] Loading canonical reference ...")
canon_ref = {}
canon_path = f'{FEAT_DIR}/canonical_reference.tsv'
if os.path.exists(canon_path):
    with open(canon_path) as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4:
                try:
                    canon_ref[p[0]] = int(p[3])
                except: pass
print(f"  Canonical entries: {len(canon_ref)}")

# valid_idxs 기준으로 global_idx → local_idx 매핑
global_to_local = {g: l for l, g in enumerate(valid_idxs)}


# ── 5. Within-gene Spearman 계산 함수 ────────────────────────────
def within_gene_spearman(features, genes_dict, ratios):
    """각 gene 내에서 feature와 ratio의 Spearman 계산 → macro 평균"""
    rhos = []
    for g, idxs in genes_dict.items():
        if len(idxs) < 2: continue
        feat_g = features[idxs].ravel() if features.ndim == 1 else features[idxs]
        ratio_g = ratios[idxs]
        if feat_g.ndim > 1:
            # multi-dim: Ridge prediction으로 single score 추출
            # (단순 내적 대신 gene-내 정규화 후 norm 사용)
            feat_g = np.linalg.norm(feat_g, axis=1)
        if ratio_g.std() < 1e-8: continue
        r, _ = spearmanr(feat_g, ratio_g)
        if not np.isnan(r):
            rhos.append(r)
    return np.mean(rhos), np.std(rhos), len(rhos)


# ── 6. Ablation 1: scalar proxy per feature ───────────────────────
print("\n[5] Ablation (scalar norm Spearman within gene) ...")
results = {}

r_sd, std_sd, n_sd = within_gene_spearman(sd_norm.ravel(), multi_genes, valid_ratios)
r_dd, std_dd, n_dd = within_gene_spearman(dd_norm.ravel(), multi_genes, valid_ratios)
r_esm, std_esm, n_esm = within_gene_spearman(esm2_norm.ravel(), multi_genes, valid_ratios)

print(f"  [A1] |splice_delta| alone:  macro Spearman = {r_sd:.4f}  ± {std_sd:.4f}  (n={n_sd})")
print(f"  [A2] |domain_delta| alone:  macro Spearman = {r_dd:.4f}  ± {std_dd:.4f}  (n={n_dd})")
print(f"  [A3] |ESM-2|       alone:   macro Spearman = {r_esm:.4f}  ± {std_esm:.4f}  (n={n_esm})")

results.update({
    'A1_splice_macro_spearman': float(r_sd),
    'A2_domain_macro_spearman': float(r_dd),
    'A3_esm2_macro_spearman': float(r_esm),
})


# ── 7. Ablation 2: Ridge regression (learn optimal combination) ───
print("\n[6] Ridge regression (learned combination) ...")

def ridge_within_gene_spearman(X_feat, genes_dict, ratios):
    """
    Gene-stratified Ridge:
      각 gene 그룹을 순서대로 test로 삼고 나머지 학습 (LOO gene CV)
      → 너무 느릴 수 있으므로 랜덤 80:20 gene split 5회 평균
    """
    gene_list_all = list(genes_dict.keys())
    macro_rhos = []
    for seed in range(5):
        rng = np.random.RandomState(seed)
        rng.shuffle(gene_list_all)
        n_train = int(len(gene_list_all) * 0.8)
        train_genes = set(gene_list_all[:n_train])
        test_genes  = set(gene_list_all[n_train:])

        tr_idxs = [i for g in train_genes for i in genes_dict[g]]
        te_idxs = [i for g in test_genes  for i in genes_dict[g]]

        if len(tr_idxs) < 10 or len(te_idxs) < 10: continue

        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
        pipe.fit(X_feat[tr_idxs], ratios[tr_idxs])
        preds = pipe.predict(X_feat[te_idxs])

        te_genes_arr = [valid_genes[i] for i in te_idxs]
        rhos = []
        for g in test_genes:
            g_local = [k for k, vg in enumerate(te_genes_arr) if vg == g]
            if len(g_local) < 2: continue
            ratio_g = ratios[[te_idxs[k] for k in g_local]]
            pred_g  = preds[g_local]
            if ratio_g.std() < 1e-8: continue
            r, _ = spearmanr(pred_g, ratio_g)
            if not np.isnan(r):
                rhos.append(r)
        if rhos:
            macro_rhos.append(np.mean(rhos))

    return float(np.mean(macro_rhos)), float(np.std(macro_rhos))

# splice + domain (D/S only)
X_ds = np.hstack([sd, dd])
r_ds, std_ds = ridge_within_gene_spearman(X_ds, multi_genes, valid_ratios)
print(f"  [A4] splice + domain (D/S): macro Spearman = {r_ds:.4f} ± {std_ds:.4f}")

# ESM-2 alone
r_esm_ridge, std_esm_ridge = ridge_within_gene_spearman(esm2, multi_genes, valid_ratios)
print(f"  [A5] ESM-2 alone:           macro Spearman = {r_esm_ridge:.4f} ± {std_esm_ridge:.4f}")

# Full: ESM-2 + domain + splice
X_full = np.hstack([esm2, dd, sd])
r_full, std_full = ridge_within_gene_spearman(X_full, multi_genes, valid_ratios)
print(f"  [A6] Full (ESM+D+S):        macro Spearman = {r_full:.4f} ± {std_full:.4f}")

results.update({
    'A4_DS_ridge_macro': float(r_ds),
    'A5_esm2_ridge_macro': float(r_esm_ridge),
    'A6_full_ridge_macro': float(r_full),
})


# ── 8. Trivial bias 제거: non-canonical only ─────────────────────
print("\n[7] Non-canonical isoforms only (remove trivial bias) ...")

# 각 gene에서 canonical을 제외한 isoform만 남기기
non_canon_genes = {}
for g, local_idxs in multi_genes.items():
    # gene_base → canonical global idx 조회
    global_idxs = [valid_idxs[i] for i in local_idxs]
    gb = gene_base_list[global_idxs[0]]  # 모두 같은 gene
    canon_global = canon_ref.get(gb)
    if canon_global is None:
        # canonical 정보 없으면 ratio 최고를 제외
        ratio_g = valid_ratios[local_idxs]
        top_local = local_idxs[int(np.argmax(ratio_g))]
        nc_idxs = [i for i in local_idxs if i != top_local]
    else:
        nc_idxs = [i for i in local_idxs if valid_idxs[i] != canon_global]

    if len(nc_idxs) >= 2:
        non_canon_genes[g] = nc_idxs

print(f"  Genes with ≥2 non-canonical isoforms: {len(non_canon_genes)}")

if len(non_canon_genes) >= 100:
    r_sd_nc, std_sd_nc, n_nc = within_gene_spearman(sd_norm.ravel(), non_canon_genes, valid_ratios)
    r_dd_nc, std_dd_nc, _    = within_gene_spearman(dd_norm.ravel(), non_canon_genes, valid_ratios)
    r_esm_nc, std_esm_nc, _  = within_gene_spearman(esm2_norm.ravel(), non_canon_genes, valid_ratios)
    print(f"  [NC-A] |splice_delta| alone:  Spearman = {r_sd_nc:.4f} ± {std_sd_nc:.4f}")
    print(f"  [NC-B] |domain_delta| alone:  Spearman = {r_dd_nc:.4f} ± {std_dd_nc:.4f}")
    print(f"  [NC-C] |ESM-2|       alone:   Spearman = {r_esm_nc:.4f} ± {std_esm_nc:.4f}")

    X_ds_nc = np.hstack([sd, dd])
    r_ds_nc, std_ds_nc = ridge_within_gene_spearman(X_ds_nc, non_canon_genes, valid_ratios)
    X_full_nc = np.hstack([esm2, dd, sd])
    r_full_nc, std_full_nc = ridge_within_gene_spearman(X_full_nc, non_canon_genes, valid_ratios)
    print(f"  [NC-D] splice + domain:       Spearman = {r_ds_nc:.4f} ± {std_ds_nc:.4f}")
    print(f"  [NC-E] Full (ESM+D+S):        Spearman = {r_full_nc:.4f} ± {std_full_nc:.4f}")

    results.update({
        'NC_splice_macro': float(r_sd_nc),
        'NC_domain_macro': float(r_dd_nc),
        'NC_esm2_macro': float(r_esm_nc),
        'NC_DS_ridge_macro': float(r_ds_nc),
        'NC_full_ridge_macro': float(r_full_nc),
        'n_non_canon_genes': int(len(non_canon_genes)),
    })
else:
    print("  [SKIP] Too few non-canonical genes")


# ── 9. 저장 ──────────────────────────────────────────────────────
results['timestamp'] = time.strftime('%Y%m%d_%H%M')
results['n_multi_genes'] = int(len(multi_genes))
results['n_valid_isoforms'] = int(valid_mask.sum())

fname = f'{OUT_DIR}/experiment_{time.strftime("%Y%m%d_%H%M")}.json'
with open(fname, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[Saved] {fname}")
print("\n" + "=" * 65)
print(" EXPERIMENT SUMMARY")
print("=" * 65)
print(f"  [GLOBAL Spearman]   splice_delta={r_sd:.4f}  domain={r_dd:.4f}  ESM={r_esm:.4f}")
print(f"  [Ridge Regression]  D/S only={r_ds:.4f}  ESM only={r_esm_ridge:.4f}  Full={r_full:.4f}")
if 'NC_splice_macro' in results:
    print(f"  [Non-canonical]     splice={results['NC_splice_macro']:.4f}  domain={results['NC_domain_macro']:.4f}  ESM={results['NC_esm2_macro']:.4f}")
    print(f"  [Non-canonical]     D/S ridge={results['NC_DS_ridge_macro']:.4f}  Full ridge={results['NC_full_ridge_macro']:.4f}")

ds_vs_esm = r_ds - r_esm_ridge
print(f"\n  D/S contribution over ESM-2: {ds_vs_esm:+.4f}")
if 'NC_full_ridge_macro' in results:
    nc_ds_vs_esm = results['NC_DS_ridge_macro'] - results['NC_esm2_macro']
    print(f"  D/S contribution (non-canonical): {nc_ds_vs_esm:+.4f}")

feasible = r_ds > 0.05 and ('NC_DS_ridge_macro' not in results or results['NC_DS_ridge_macro'] > 0.05)
print(f"\n  Verdict: {'GENUINE CONTRIBUTION' if feasible else 'TRIVIAL BIAS ONLY'}")
print("Done.")
