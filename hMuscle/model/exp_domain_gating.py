"""
exp_domain_gating.py — Option C: Post-hoc Domain-Gated Prediction
=======================================================================
label_confidence.npy를 이미 계산된 v17f* predictions에 post-hoc gate로 적용.

gate(i,j) = alpha + (1 - alpha) * clip(label_conf(i,j), 0, 1)
pred_gated(i,j) = pred(i,j) * gate(i,j)

분석:
1. alpha sweep [0.0, 0.3, 0.5, 0.7, 1.0]에서 최적 alpha 탐색
2. Feature Type별 AUPRC 변화 측정 (Type1 target: domain-loss 보정)
3. Threshold-independent: AUPRC primary metric
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
import json

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
OUT_DIR = ROOT / "reports/isoform_resolution_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("[1] 데이터 로드...")
preds = np.load(ROOT / "reports/v17f_star_bootstrap/v17f_star_preds.npy")  # (36748, 82)
Y = np.load(ROOT / "reports/v17f_star_bootstrap/Y_te.npy")                 # (36748, 82)
label_conf = np.load(OUT_DIR / "label_confidence.npy")                      # (36748, 82)

print(f"  preds: {preds.shape}, Y: {Y.shape}, label_conf: {label_conf.shape}")
print(f"  label_conf range: [{label_conf.min():.3f}, {label_conf.max():.3f}]")

# Feature type per isoform
import csv
ft_arr = []
with open(OUT_DIR / "full_isoform_feature_types.tsv") as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        ft_arr.append(row['feature_type'])
ft_arr = np.array(ft_arr)
print(f"  Feature types: {np.unique(ft_arr, return_counts=True)}")

# ---- Helper: AUPRC per type ----
def auprc_by_type(y_true, y_pred, ft, n_boot=200, seed=0):
    types = ['Type0_NoDomain', 'Type1_DomainLoss', 'Type2_PartialTrunc',
             'Type3_SameDomain']
    rng = np.random.default_rng(seed)
    result = {}
    for t in types:
        mask = (ft == t)
        if mask.sum() < 10:
            continue
        yt = y_true[mask]  # (n_t, 82)
        yp = y_pred[mask]  # (n_t, 82)
        # Macro AUPRC (terms with >=2 positives)
        valid = (yt.sum(0) >= 2)
        if valid.sum() == 0:
            continue
        auprcs = [average_precision_score(yt[:, j], yp[:, j]) for j in np.where(valid)[0]]
        observed = float(np.mean(auprcs))
        # Bootstrap CI (200 isoform-level resample)
        boot_vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(yt), size=len(yt))
            byt, byp = yt[idx], yp[idx]
            bv = (byt.sum(0) >= 2)
            if bv.sum() == 0:
                boot_vals.append(observed)
                continue
            boot_vals.append(float(np.mean([average_precision_score(byt[:, j], byp[:, j])
                                             for j in np.where(bv)[0]])))
        lo, hi = float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))
        result[t] = {'auprc': observed, 'ci_lo': lo, 'ci_hi': hi, 'n': int(mask.sum())}
    # Overall
    valid_overall = (y_true.sum(0) >= 2)
    auprcs_all = [average_precision_score(y_true[:, j], y_pred[:, j])
                  for j in np.where(valid_overall)[0]]
    result['Overall'] = {'auprc': float(np.mean(auprcs_all)), 'n': y_true.shape[0]}
    return result


print("\n[2] Baseline (alpha=1.0, no gating) AUPRC...")
baseline = auprc_by_type(Y, preds, ft_arr)
print("  Type → Baseline AUPRC:")
for t, v in sorted(baseline.items()):
    ci = f"[{v.get('ci_lo', 0):.4f},{v.get('ci_hi', 0):.4f}]" if 'ci_lo' in v else ""
    print(f"    {t:25s}: {v['auprc']:.4f} {ci} n={v['n']}")

print("\n[3] Gate 적용 — alpha sweep...")
# gate(i,j): alpha=1.0 → no effect; alpha=0.0 → full confidence weighting
# gate clips label_conf to [0,1] before scaling
lc_clipped = np.clip(label_conf, 0.0, 1.0)  # (36748, 82)

alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
sweep_results = {}

for alpha in alphas:
    gate = alpha + (1.0 - alpha) * lc_clipped          # (36748, 82), range [alpha, 1.0]
    preds_gated = np.clip(preds * gate, 0.0, 1.0)      # (36748, 82)
    res = auprc_by_type(Y, preds_gated, ft_arr, n_boot=100)
    sweep_results[alpha] = res
    print(f"  alpha={alpha:.1f}: Type1={res.get('Type1_DomainLoss',{}).get('auprc',0):.4f} "
          f"Type2={res.get('Type2_PartialTrunc',{}).get('auprc',0):.4f} "
          f"Type3={res.get('Type3_SameDomain',{}).get('auprc',0):.4f} "
          f"Overall={res.get('Overall',{}).get('auprc',0):.4f}")

# Find best alpha per type
print("\n[4] Best alpha per type:")
for t in ['Type1_DomainLoss', 'Type2_PartialTrunc', 'Type3_SameDomain', 'Overall']:
    best_alpha = max(alphas, key=lambda a: sweep_results[a].get(t, {}).get('auprc', 0))
    best_auprc = sweep_results[best_alpha].get(t, {}).get('auprc', 0)
    base_auprc = baseline.get(t, {}).get('auprc', 0)
    delta = best_auprc - base_auprc
    print(f"  {t:25s}: best alpha={best_alpha:.1f} → {best_auprc:.4f} (Δ{delta:+.4f} vs baseline {base_auprc:.4f})")

print("\n[5] Asymmetric gating — positive-label only gating...")
# 핵심 아이디어: gate는 positive label에만 적용 (negatives는 그대로)
# For positive-labeled isoforms: down-weight domain-inconsistent positives
# For negative-labeled isoforms: keep as-is
alpha_best = 0.4
gate_pos = alpha_best + (1.0 - alpha_best) * lc_clipped  # (36748, 82)

# Apply only to positive predictions that have high prediction score (top decile)
# Interpretation: "if we predicted positive but domain is inconsistent → reduce score"
preds_asym = preds.copy()
high_pred = (preds > 0.3)  # predicted-positive region
preds_asym[high_pred] = np.clip(preds[high_pred] * gate_pos[high_pred], 0.0, 1.0)

res_asym = auprc_by_type(Y, preds_asym, ft_arr, n_boot=200)
print("  Asymmetric gate (alpha=0.4, apply to pred>0.3):")
for t, v in sorted(res_asym.items()):
    base = baseline.get(t, {}).get('auprc', 0)
    ci = f"[{v.get('ci_lo', 0):.4f},{v.get('ci_hi', 0):.4f}]" if 'ci_lo' in v else ""
    delta = v['auprc'] - base
    print(f"    {t:25s}: {v['auprc']:.4f} {ci} Δ{delta:+.4f}")

print("\n[6] Type1 deep dive — per-GO impact...")
t1_mask = (ft_arr == 'Type1_DomainLoss')
yt1 = Y[t1_mask]
pp1_base = preds[t1_mask]

alpha_test = 0.4
gate1 = alpha_test + (1.0 - alpha_test) * lc_clipped[t1_mask]
pp1_gated = np.clip(pp1_base * gate1, 0.0, 1.0)

valid_j = (yt1.sum(0) >= 2)
print(f"  Type1 valid GO terms: {valid_j.sum()}/82")
improvements = []
for j in np.where(valid_j)[0]:
    base_ap = average_precision_score(yt1[:, j], pp1_base[:, j])
    gate_ap = average_precision_score(yt1[:, j], pp1_gated[:, j])
    improvements.append((j, gate_ap - base_ap, base_ap, gate_ap))

improvements.sort(key=lambda x: -x[1])
print("  Top-5 improved GO terms (Type1):")
for j, delta, ba, ga in improvements[:5]:
    print(f"    GO[{j}]: {ba:.4f} → {ga:.4f} (Δ{delta:+.4f})")
print("  Top-5 degraded GO terms (Type1):")
for j, delta, ba, ga in improvements[-5:]:
    print(f"    GO[{j}]: {ba:.4f} → {ga:.4f} (Δ{delta:+.4f})")

# Save results
out = {
    'baseline': baseline,
    'sweep': {str(a): sweep_results[a] for a in alphas},
    'asymmetric_gate': res_asym,
}
with open(OUT_DIR / "domain_gating_results.json", 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {OUT_DIR}/domain_gating_results.json")
