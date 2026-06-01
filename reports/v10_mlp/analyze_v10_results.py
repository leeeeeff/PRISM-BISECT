"""
analyze_v10_results.py — v10-MLP 결과 분석 및 테이블 출력
"""
import json, glob, os
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
GO_ORDER = ['GO:0006096', 'GO:0003774', 'GO:0007204', 'GO:0030017', 'GO:0006941']
GO_LABELS = {
    'GO:0006096': 'Glycolysis (A)',
    'GO:0003774': 'Motor Activity (A)',
    'GO:0007204': 'Ca2+ Signal (B)',
    'GO:0030017': 'Sarcomere (B)',
    'GO:0006941': 'Muscle Contr (B)',
}

# Load the latest JSON with all 5 GO terms
jsons = sorted(glob.glob(f'{RESULTS_DIR}/v10_mlp_results_*.json'))
latest = None
for j in reversed(jsons):
    with open(j) as f:
        d = json.load(f)
    if len(d['results']) == 5:
        latest = d
        print(f"Loaded: {j}  ({len(d['results'])} GO terms)")
        break

if not latest:
    # Merge single-GO results
    print("No 5-GO file found. Merging single results...")
    merged = {}
    for j in jsons:
        with open(j) as f:
            d = json.load(f)
        for r in d['results']:
            merged[r['go']] = r
    latest = {'results': [merged[g] for g in GO_ORDER if g in merged],
              'baselines': d['baselines']}
    print(f"  Merged {len(latest['results'])} GO terms")

results = {r['go']: r for r in latest['results']}
baselines = latest['baselines']

# ─── Table 1: Main Results ────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  TABLE 1 — v10-MLP Ablation (train→test AUPRC)")
print("=" * 90)
print(f"{'GO Term':<22} {'v10-A':>7} {'v10-B':>7} {'v10-C':>7} | {'LR':>7} {'D1_MLP':>7} {'v8b':>7}")
print(f"{'':22} {'(E+D_cnn)':>7} {'(E only)':>7} {'(E+D_dns)':>7} | {'ref':>7} {'ref':>7} {'ref':>7}")
print("-" * 90)

v10A_vals, v10B_vals, v10C_vals = [], [], []
lr_vals, d1_vals, v8b_vals = [], [], []

for go in GO_ORDER:
    if go not in results:
        continue
    r = results[go]
    lr  = baselines['LR'][go]
    d1  = baselines['D1_MLP'][go]
    v8b = baselines['v8b'][go]
    label = GO_LABELS[go]
    print(f"{label:<22} {r['v10A_auprc']:>7.4f} {r['v10B_auprc']:>7.4f} {r['v10C_auprc']:>7.4f} | "
          f"{lr:>7.4f} {d1:>7.4f} {v8b:>7.4f}")
    v10A_vals.append(r['v10A_auprc']); v10B_vals.append(r['v10B_auprc'])
    v10C_vals.append(r['v10C_auprc']); lr_vals.append(lr)
    d1_vals.append(d1); v8b_vals.append(v8b)

if v10A_vals:
    print("-" * 90)
    mA = np.mean(v10A_vals); mB = np.mean(v10B_vals); mC = np.mean(v10C_vals)
    mLR = np.mean(lr_vals);  mD1 = np.mean(d1_vals); mv8 = np.mean(v8b_vals)
    print(f"{'Macro':<22} {mA:>7.4f} {mB:>7.4f} {mC:>7.4f} | {mLR:>7.4f} {mD1:>7.4f} {mv8:>7.4f}")

# ─── Table 2: Splicing Contribution (Phase 2 CV) ─────────────────────────────
print("\n" + "=" * 70)
print("  TABLE 2 — Splicing BiGRU Contribution (test CV)")
print("=" * 70)
print(f"{'GO Term':<22} {'emb_only':>10} {'emb+splice':>12} {'delta':>8} {'LR_ref':>8}")
print("-" * 70)

cv_emb_vals, cv_spl_vals = [], []
for go in GO_ORDER:
    if go not in results:
        continue
    r = results[go]
    lr = baselines['LR'][go]
    delta = r['v10D_cv_splice'] - r['v10D_cv_emb']
    print(f"{GO_LABELS[go]:<22} {r['v10D_cv_emb']:>10.4f} {r['v10D_cv_splice']:>12.4f} "
          f"{delta:>+8.4f} {lr:>8.4f}")
    cv_emb_vals.append(r['v10D_cv_emb']); cv_spl_vals.append(r['v10D_cv_splice'])

if cv_emb_vals:
    print("-" * 70)
    mce = np.mean(cv_emb_vals); mcs = np.mean(cv_spl_vals)
    print(f"{'Macro':<22} {mce:>10.4f} {mcs:>12.4f} {mcs-mce:>+8.4f}")

# ─── Table 3: Ablation Interpretation ────────────────────────────────────────
if v10A_vals:
    print("\n" + "=" * 60)
    print("  TABLE 3 — Ablation Summary (Macro AUPRC)")
    print("=" * 60)
    print(f"  v10-A (ESM + domain_CNN, E+D_cnn): {mA:.4f}")
    print(f"  v10-B (ESM only, E):                {mB:.4f}")
    print(f"  v10-C (ESM + domain_Dense, E+D_dns):{mC:.4f}")
    print(f"  LR baseline:                         {mLR:.4f}")
    print(f"  D1_MLP (simple MLP reference):       {mD1:.4f}")
    print(f"  v8b-PFN (previous best):             {mv8:.4f}")
    print()
    print(f"  domain_CNN contribution (A-B):        {mA-mB:+.4f}")
    print(f"  Conv1D vs Dense (A-C):                {mA-mC:+.4f}")
    print(f"  v10-A vs LR:                          {mA-mLR:+.4f} ({(mA/mLR-1)*100:+.1f}%)")
    print(f"  v10-A vs D1_MLP:                      {mA-mD1:+.4f} ({(mA/mD1-1)*100:+.1f}%)")
    print(f"  v10-A vs v8b:                         {mA-mv8:+.4f} ({(mA/mv8-1)*100:+.1f}%)")
    print()
    print(f"  splicing_BiGRU macro contribution:    {mcs-mce:+.4f}")

    # Gate re-check
    print("\n" + "=" * 60)
    print("  GATE CHECK")
    print("=" * 60)
    print(f"  [Primary] v10-A > LR (0.5615):  {'✅ PASS' if mA > 0.5615 else '❌ FAIL'} ({mA:.4f})")
    print(f"  [Domain]  A > B (domain adds):  {'✅ PASS' if mA > mB else '❌ FAIL'} (Δ={mA-mB:+.4f})")
    print(f"  [Conv1D]  A > C (CNN>Dense):    {'✅ PASS' if mA > mC else '❌ FAIL'} (Δ={mA-mC:+.4f})")
    if cv_spl_vals:
        print(f"  [Splice]  splice>emb (BiGRU):   {'✅ PASS' if mcs > mce else '❌ FAIL'} (Δ={mcs-mce:+.4f})")
