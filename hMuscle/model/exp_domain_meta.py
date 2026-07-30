"""
exp_domain_meta.py — Option B (revised): Domain-Augmented Meta-Model
=======================================================================
PRISM predictions + domain_delta를 결합한 stacked generalization.

Input: concat[PRISM_preds(82), domain_delta(512)] → 594 dim
Meta-model: Ridge regression per GO term (gene-stratified 5-fold CV)
Goal: does domain information improve prediction beyond PRISM alone?

Split: gene-level 5-fold (no isoform leakage)
Evaluation: Macro AUPRC per feature type, with CI
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import json

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
OUT_DIR = ROOT / "reports/isoform_resolution_full"
FEAT_DIR = ROOT / "hMuscle/results_isoform/features"

print("[1] 데이터 로드...")
preds = np.load(ROOT / "reports/v17f_star_bootstrap/v17f_star_preds.npy")   # (36748, 82)
Y = np.load(ROOT / "reports/v17f_star_bootstrap/Y_te.npy")                   # (36748, 82)
dm_v3 = np.load(FEAT_DIR / "domain_matrix_proper_test_v3.npy")               # (36748, 512)
dom_delta_v3 = np.load(FEAT_DIR / "domain_delta_proper_test_v3.npy")         # (36748, 512)
label_conf = np.load(OUT_DIR / "label_confidence.npy")                        # (36748, 82)

iso_raw = np.load(ROOT / "hMuscle/model/my_isoform_list_fixed.npy", allow_pickle=True)
gene_raw = np.load(ROOT / "hMuscle/model/my_gene_list_fixed.npy", allow_pickle=True)
iso_list = [x.decode() if isinstance(x, bytes) else x for x in iso_raw]
gene_list = [x.decode() if isinstance(x, bytes) else x for x in gene_raw]
gene_arr = np.array(gene_list)
print(f"  Loaded: preds{preds.shape} Y{Y.shape} dm_v3{dm_v3.shape} dom_delta{dom_delta_v3.shape}")

import csv
ft_arr = []
with open(OUT_DIR / "full_isoform_feature_types.tsv") as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        ft_arr.append(row['feature_type'])
ft_arr = np.array(ft_arr)

# ── Gene-stratified 5-fold split ─────────────────────────────────────────
print("[2] Gene-stratified 5-fold split 생성...")
genes_uniq = np.array(sorted(set(gene_list)))
rng = np.random.default_rng(42)
gene_fold = np.zeros(len(genes_uniq), dtype=int)
perm = rng.permutation(len(genes_uniq))
for k, idx in enumerate(perm):
    gene_fold[idx] = k % 5
gene_to_fold = {g: gene_fold[i] for i, g in enumerate(genes_uniq)}
iso_fold = np.array([gene_to_fold[g] for g in gene_list])

# ── Meta-feature sets to compare ─────────────────────────────────────────
# F0: PRISM predictions only (baseline)
# F1: PRISM + domain_delta (v3)
# F2: PRISM + domain_matrix (binary presence)
# F3: PRISM + domain_delta + label_confidence

feature_sets = {
    'F0_PRISM': preds,                                              # (36748, 82)
    'F1_PRISM+delta': np.hstack([preds, dom_delta_v3]),            # (36748, 594)
    'F2_PRISM+dm': np.hstack([preds, dm_v3]),                      # (36748, 594)
    'F3_PRISM+delta+conf': np.hstack([preds, dom_delta_v3, label_conf]),  # (36748, 676)
}

def macro_auprc_type(y_all, yp_all, ft_all, min_pos=2):
    results = {}
    types = ['Type0_NoDomain','Type1_DomainLoss','Type2_PartialTrunc','Type3_SameDomain','Overall']
    for t in types:
        if t == 'Overall':
            mask = np.ones(len(ft_all), dtype=bool)
        else:
            mask = (ft_all == t)
        if mask.sum() < 10:
            continue
        yt = y_all[mask]; ypp = yp_all[mask]
        valid = yt.sum(0) >= min_pos
        if valid.sum() == 0:
            continue
        vals = [average_precision_score(yt[:,j], ypp[:,j]) for j in np.where(valid)[0]]
        results[t] = float(np.mean(vals))
    return results

print("[3] 5-fold CV meta-model 평가...")
print("    (Ridge regression per GO term, 5-fold gene-stratified)")
print()

all_results = {}
for fname, X_meta in feature_sets.items():
    print(f"  === {fname} (dim={X_meta.shape[1]}) ===")
    # Collect OOF predictions
    oof_preds = np.zeros_like(Y, dtype=float)
    for fold in range(5):
        tr_mask = (iso_fold != fold)
        te_mask = (iso_fold == fold)
        X_tr, Y_tr = X_meta[tr_mask], Y[tr_mask]
        X_te = X_meta[te_mask]
        # Per-GO logistic regression (binary output)
        preds_fold = np.zeros((te_mask.sum(), 82))
        for j in range(82):
            y_tr = Y_tr[:, j]
            pos = y_tr.sum()
            if pos < 5:
                preds_fold[:, j] = X_tr[:, j].mean() if fname == 'F0_PRISM' else X_tr[:, j % 82].mean()
                continue
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            try:
                clf = LogisticRegression(C=0.1, max_iter=200, solver='lbfgs')
                clf.fit(X_s, y_tr.astype(int))
                preds_fold[:, j] = clf.predict_proba(X_te_s)[:, 1]
            except Exception:
                preds_fold[:, j] = y_tr.mean()
        oof_preds[te_mask] = preds_fold

    res = macro_auprc_type(Y, oof_preds.astype(np.float32), ft_arr)
    all_results[fname] = res
    for t, v in sorted(res.items()):
        print(f"    {t:25s}: {v:.4f}")
    print()

# ── Summary table ─────────────────────────────────────────────────────────
print("\n[4] 비교 요약 (Delta vs F0_PRISM)")
print(f"{'Feature':30s} | {'Type1':8s} | {'Type2':8s} | {'Type3':8s} | {'Type0':8s} | {'Overall':8s}")
print("-" * 85)
base = all_results.get('F0_PRISM', {})
for fname, res in all_results.items():
    t0 = res.get('Type0_NoDomain', 0)
    t1 = res.get('Type1_DomainLoss', 0)
    t2 = res.get('Type2_PartialTrunc', 0)
    t3 = res.get('Type3_SameDomain', 0)
    ov = res.get('Overall', 0)
    if fname == 'F0_PRISM':
        print(f"  {fname:28s} | {t1:8.4f} | {t2:8.4f} | {t3:8.4f} | {t0:8.4f} | {ov:8.4f}")
    else:
        d1 = t1 - base.get('Type1_DomainLoss', 0)
        d2 = t2 - base.get('Type2_PartialTrunc', 0)
        d3 = t3 - base.get('Type3_SameDomain', 0)
        d0 = t0 - base.get('Type0_NoDomain', 0)
        dv = ov - base.get('Overall', 0)
        print(f"  {fname:28s} | {t1:6.4f}({d1:+.3f}) | {t2:6.4f}({d2:+.3f}) | {t3:6.4f}({d3:+.3f}) | {t0:6.4f}({d0:+.3f}) | {ov:6.4f}({dv:+.3f})")

with open(OUT_DIR / "domain_meta_results.json", 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {OUT_DIR}/domain_meta_results.json")

# ── Option C Design Recommendation ──────────────────────────────────────
print("\n[5] Option C 아키텍처 설계 권고")
f1 = all_results.get('F1_PRISM+delta', {})
f0 = all_results.get('F0_PRISM', {})
d1 = f1.get('Type1_DomainLoss', 0) - f0.get('Type1_DomainLoss', 0)
print(f"  domain_delta 추가 효과 (Type1): Δ{d1:+.4f}")
if abs(d1) > 0.01:
    print("  → 유의미한 효과: domain_delta를 PRISM 입력에 통합 권장")
    print("  → 아키텍처: concat[δ_layer(640), L30(640), domain_delta_v3(512)] → Dense(256)→...")
else:
    print("  → 효과 미미: ESM-2가 도메인 정보를 이미 내포 (domain_delta 중복)")
    print("  → 권장: domain_matrix를 GO-term-specific attention weight로만 활용 (설명 도구)")
