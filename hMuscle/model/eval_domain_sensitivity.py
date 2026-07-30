"""
eval_domain_sensitivity.py — 도메인 중요도 가중 아이소폼 기능 소실 평가
=========================================================================
평가 철학:
  기존 AUPRC는 gene-level label noise 때문에 모델 개선을 포착 못한다.
  대신, "핵심 도메인이 소실된 아이소폼을 모델이 해당 GO term에서
  canonical보다 낮게 예측하는가?"를 직접 평가한다.

케이스 선별 기준:
  1. Type1/Type2 아이소폼 (도메인 소실 또는 부분 절단)
  2. canonical vs isoform 간 domain matrix 차이 벡터:
       dm_lost(i) = dm[canonical_gene(i)] - dm[i]   (≥0 for Type1)
  3. GO term j에 대한 Pfam 도메인 중요도 (test set 기반):
       pfam_imp(d, j) = log[P(pfam_d | GO_j+) / P(pfam_d | GO_j-)]  (clipped)
  4. lost_domain_importance(i, j) = dot(dm_lost(i), pfam_imp[:, j])
     → 이 값이 클수록 "핵심 도메인 소실"

평가 지표 (Domain Sensitivity Rate, DSR):
  - Core loss (top 25%): score(isoform, j) < score(canonical, j) 의 비율
  - Peripheral loss (middle 50%): 같은 기준
  - No loss (Type3, 같은 도메인):  false positive rate (기준선)

비교:
  - v17f*            (baseline, no confidence weighting)
  - v17f_confweight  (alpha=0.5)
  - v17f_confweight_a02 (alpha=0.2, 강한 억제)
"""

import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import csv, json

ROOT     = Path("/home/welcome1/sw1686/DIFFUSE")
OUT_DIR  = ROOT / "reports/isoform_resolution_full"
FEAT_DIR = ROOT / "hMuscle/results_isoform/features"

print("[0] 데이터 로드...")
Y        = np.load(ROOT / "reports/v17f_star_bootstrap/Y_te.npy")     # (36748, 82)
dm       = np.load(FEAT_DIR / "domain_matrix_proper_test_v3.npy")     # (36748, 512)

gene_raw = np.load(ROOT / "hMuscle/model/my_gene_list_fixed.npy", allow_pickle=True)
iso_raw  = np.load(ROOT / "hMuscle/model/my_isoform_list_fixed.npy", allow_pickle=True)
gene_list = [x.decode() if isinstance(x, bytes) else x for x in gene_raw]
iso_list  = [x.decode() if isinstance(x, bytes) else x for x in iso_raw]
gene_arr  = np.array(gene_list)
n_iso, n_go = Y.shape

ft_arr = []
dc_arr = []
with open(OUT_DIR / "full_isoform_feature_types.tsv") as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        ft_arr.append(row['feature_type'])
        dc_arr.append(int(row['domain_count']))
ft_arr = np.array(ft_arr)
dc_arr = np.array(dc_arr)

mf_terms = []
with open(ROOT / "reports/v_expanded_gomf/mf_domain_vs_prism.tsv") as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])
mf_terms = np.array(mf_terms)

gene_to_idx = defaultdict(list)
for i, g in enumerate(gene_list):
    gene_to_idx[g].append(i)

# ── 1. Pfam GO-importance 계산 (test set 기반) ────────────────────────────
print("[1] Pfam GO-importance 계산 (test set 기반)...")
# For each (Pfam domain d, GO term j):
# pfam_imp(d, j) = log[ (pos_dom[j][d] + eps) / (neg_dom[j][d] + eps) ]
# where pos_dom[j][d] = fraction of GO_j+ isoforms that have domain d

eps = 1e-3
n_pfam = dm.shape[1]
pfam_imp = np.zeros((n_pfam, n_go), dtype=np.float32)  # (512, 82)

for j in range(n_go):
    pos_mask = Y[:, j] > 0
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos < 5: continue
    pos_rate = (dm[pos_mask].sum(0) + eps) / (n_pos + eps)  # (512,)
    neg_rate = (dm[neg_mask].sum(0) + eps) / (n_neg + eps)  # (512,)
    pfam_imp[:, j] = np.clip(np.log(pos_rate / neg_rate), -5, 5)

print(f"  pfam_imp shape: {pfam_imp.shape}, nonzero: {(pfam_imp != 0).sum()}")

# ── 2. 유전자별 canonical isoform 결정 (max domain count per gene) ─────────
print("[2] Canonical isoform 결정 (max domain count per gene)...")
gene_canonical = {}  # gene → global idx of canonical (most domains)
for g, idxs in gene_to_idx.items():
    idxs_arr = np.array(idxs)
    dc_g = dc_arr[idxs_arr]
    max_dc = dc_g.max()
    # canonical = first isoform with max domain count
    canon_local = np.where(dc_g == max_dc)[0][0]
    gene_canonical[g] = idxs_arr[canon_local]

# ── 3. 케이스 선별: Type1/2 아이소폼 × GO term 쌍 ─────────────────────────
print("[3] Type1/2 케이스 선별 + lost_domain_importance 계산...")

cases = []  # dict per case
for g, idxs in gene_to_idx.items():
    idxs_arr = np.array(idxs)
    canon_idx = gene_canonical[g]
    dm_canon  = dm[canon_idx]           # (512,) canonical domain profile
    y_canon   = Y[canon_idx]            # (82,) canonical GO labels

    for i in idxs_arr:
        if i == canon_idx: continue
        ft = ft_arr[i]
        if 'Type1' not in ft and 'Type2' not in ft: continue

        dm_i    = dm[i]
        dm_lost = (dm_canon - dm_i).clip(0)  # lost domains (≥0)
        if dm_lost.sum() < 1e-6: continue    # no domain actually lost

        for j in range(n_go):
            # Only evaluate GO terms where canonical is positive (gene has this GO)
            if y_canon[j] < 1: continue

            # lost domain importance for this (isoform, GO) pair
            ldi = float(np.dot(dm_lost, pfam_imp[:, j]))

            # secondary: which specific domains are lost + their importance
            lost_dom_idxs = np.where(dm_lost > 0)[0]
            imp_per_lost_dom = pfam_imp[lost_dom_idxs, j] if len(lost_dom_idxs) > 0 else np.array([])
            max_single_dom_imp = float(imp_per_lost_dom.max()) if len(imp_per_lost_dom) > 0 else 0.0

            cases.append({
                'gene': g,
                'iso_idx': i,
                'canon_idx': canon_idx,
                'go_idx': j,
                'feature_type': ft,
                'ldi': ldi,           # lost domain importance (aggregate)
                'max_dom_imp': max_single_dom_imp,  # single most important lost domain
                'n_lost_domains': int(dm_lost.sum()),
            })

cases_arr = np.array(cases)
ldi_vals  = np.array([c['ldi'] for c in cases])
print(f"  Total cases (Type1/2 isoform × positive GO term): {len(cases)}")
print(f"  LDI stats: mean={ldi_vals.mean():.3f}, std={ldi_vals.std():.3f}")
print(f"  LDI > 0 (some positive importance): {(ldi_vals > 0).sum()} ({(ldi_vals>0).mean()*100:.1f}%)")

# ── 4. LDI 기반 계층화 ────────────────────────────────────────────────────
print("[4] LDI 계층화...")
# Stratify by LDI into Core / Peripheral / Near-zero
q25, q50, q75 = np.percentile(ldi_vals, [25, 50, 75])
print(f"  LDI percentiles: Q25={q25:.3f}, Q50={q50:.3f}, Q75={q75:.3f}")

core_mask       = ldi_vals >= q75       # top 25% = core domain loss
peripheral_mask = (ldi_vals >= q25) & (ldi_vals < q75)  # middle 50%
minimal_mask    = ldi_vals < q25        # bottom 25% = minimal importance

# Also: max single domain importance stratification
max_imp_vals = np.array([c['max_dom_imp'] for c in cases])
q75_maximp = np.percentile(max_imp_vals, 75)
critical_dom_mask = max_imp_vals >= q75_maximp  # single critical domain lost

print(f"  Core loss (LDI ≥ Q75={q75:.3f}): n={core_mask.sum()}")
print(f"  Peripheral (Q25-Q75):            n={peripheral_mask.sum()}")
print(f"  Minimal (LDI < Q25={q25:.3f}): n={minimal_mask.sum()}")
print(f"  Critical single domain (≥Q75 max_imp): n={critical_dom_mask.sum()}")

# ── 5. Type3 케이스 (False Positive Baseline) ─────────────────────────────
print("[5] Type3 baseline cases (same domain, should NOT split)...")
type3_cases = []
for g, idxs in gene_to_idx.items():
    idxs_arr = np.array(idxs)
    canon_idx = gene_canonical[g]
    y_canon   = Y[canon_idx]

    for i in idxs_arr:
        if i == canon_idx: continue
        if 'Type3' not in ft_arr[i]: continue
        for j in range(n_go):
            if y_canon[j] < 1: continue
            type3_cases.append({'iso_idx': i, 'canon_idx': canon_idx, 'go_idx': j})

print(f"  Type3 baseline cases: {len(type3_cases)}")

# ── 6. Domain Sensitivity Rate (DSR) 계산 함수 ───────────────────────────
def compute_dsr(case_list, pred_matrix, label=""):
    """
    DSR = P(score(isoform, j) < score(canonical, j))
    = fraction where model correctly ranks isoform below canonical
    """
    correct = 0
    tie     = 0
    total   = len(case_list)
    if total == 0: return float('nan'), float('nan'), 0

    for c in case_list:
        s_iso   = pred_matrix[c['iso_idx'], c['go_idx']]
        s_canon = pred_matrix[c['canon_idx'], c['go_idx']]
        if s_iso < s_canon:
            correct += 1
        elif s_iso == s_canon:
            tie += 1

    dsr = correct / total
    margin = np.mean([pred_matrix[c['canon_idx'], c['go_idx']] - pred_matrix[c['iso_idx'], c['go_idx']]
                      for c in case_list])
    return dsr, float(margin), total

# Also compute DSR split by core/peripheral/minimal
def compute_dsr_stratified(cases_full, ldi_values, pred_matrix):
    masks = {
        'core (LDI≥Q75)':     ldi_values >= q75,
        'peripheral (Q25-Q75)': (ldi_values >= q25) & (ldi_values < q75),
        'minimal (LDI<Q25)':  ldi_values < q25,
        'critical_dom':        max_imp_vals >= q75_maximp,
    }
    results = {}
    for name, mask in masks.items():
        sub = [cases_full[k] for k in np.where(mask)[0]]
        dsr, margin, n = compute_dsr(sub, pred_matrix)
        results[name] = {'dsr': dsr, 'margin': margin, 'n': n}
    return results

# ── 7. 모델별 DSR 비교 ────────────────────────────────────────────────────
print("\n[7] 모델별 Domain Sensitivity Rate 비교...")

model_preds = {}

# v17f* (baseline)
p_base = np.load(ROOT / "reports/v17f_star_bootstrap/v17f_star_preds.npy")
model_preds['v17f* (baseline)'] = p_base

# v17f_confweight (alpha=0.5)
cw_path = ROOT / "reports/v17f_confweight_preds.npy"
if cw_path.exists():
    model_preds['confweight_a05'] = np.load(cw_path)

# v17f_confweight_a02 (alpha=0.2)
cw02_path = ROOT / "reports/v17f_confweight_a02_preds.npy"
if cw02_path.exists():
    model_preds['confweight_a02'] = np.load(cw02_path)

print(f"\n{'Model':<22} {'Overall DSR':>12} {'Margin':>9} {'n':>8}")
print("-" * 55)
overall_results = {}
for mname, preds in model_preds.items():
    dsr, margin, n = compute_dsr(cases, preds)
    overall_results[mname] = {'dsr': dsr, 'margin': margin, 'n': n}
    print(f"  {mname:<20} {dsr:>11.4f} {margin:>8.4f} {n:>8}")

# Type3 baseline
dsr_t3, margin_t3, n_t3 = compute_dsr(type3_cases, p_base)
print(f"\n  Type3 (FP baseline)  {dsr_t3:>11.4f} {margin_t3:>8.4f} {n_t3:>8}")
print(f"  (Expected DSR(Type3) ≈ 0.5 if no bias)")

# Type3 DSR per model (to check if confidence weighting affects Type3)
print(f"\n  Type3 DSR per model (should not change much):")
for mname, preds in model_preds.items():
    dsr_t3m, _, _ = compute_dsr(type3_cases, preds)
    print(f"    {mname:<22}: {dsr_t3m:.4f}")

# DSR gap = Type1/2 DSR - Type3 DSR (larger gap = better domain sensitivity)
print(f"\n  Domain Sensitivity Gap (DSR_type12 - DSR_type3):")
for mname, preds in model_preds.items():
    dsr_type12, _, _ = compute_dsr(cases, preds)
    dsr_type3m, _, _ = compute_dsr(type3_cases, preds)
    gap = dsr_type12 - dsr_type3m
    print(f"    {mname:<22}: {gap:+.4f}  (type12={dsr_type12:.4f}, type3={dsr_type3m:.4f})")

# Stratified
print(f"\n{'Stratified DSR by LDI level':}")
print(f"{'Strata':<22} {'v17f*':>8} {'CW_a05':>8} {'CW_a02':>8}")
print("-" * 50)
strat_base = compute_dsr_stratified(cases, ldi_vals, p_base)
strat_a05  = compute_dsr_stratified(cases, ldi_vals, model_preds.get('confweight_a05', p_base))
strat_a02  = compute_dsr_stratified(cases, ldi_vals, model_preds.get('confweight_a02', p_base))

for strat_name in strat_base:
    b   = strat_base[strat_name]['dsr']
    a05 = strat_a05[strat_name]['dsr']
    a02 = strat_a02[strat_name]['dsr']
    n   = strat_base[strat_name]['n']
    print(f"  {strat_name:<20} {b:>7.4f} {a05:>7.4f} {a02:>7.4f}  (n={n})")

# ── 8. 개별 케이스: 이전 모델이 못 분리했지만 새 모델이 분리한 케이스 ──────
print("\n[8] 새 모델이 새로 분리한 케이스 (confweight_a02 > v17f*)...")
if 'confweight_a02' in model_preds:
    p_new = model_preds['confweight_a02']

    newly_correct = []
    newly_wrong   = []
    for c in cases:
        s_iso_base  = p_base[c['iso_idx'], c['go_idx']]
        s_can_base  = p_base[c['canon_idx'], c['go_idx']]
        s_iso_new   = p_new[c['iso_idx'], c['go_idx']]
        s_can_new   = p_new[c['canon_idx'], c['go_idx']]
        base_correct = s_iso_base < s_can_base
        new_correct  = s_iso_new  < s_can_new
        if not base_correct and new_correct:
            newly_correct.append(c)
        elif base_correct and not new_correct:
            newly_wrong.append(c)

    print(f"  Cases newly CORRECT in confweight_a02: {len(newly_correct)} "
          f"({len(newly_correct)/len(cases)*100:.2f}%)")
    print(f"  Cases newly WRONG in confweight_a02:   {len(newly_wrong)} "
          f"({len(newly_wrong)/len(cases)*100:.2f}%)")
    net_gain = len(newly_correct) - len(newly_wrong)
    print(f"  Net gain: {net_gain:+d}")

    # Newly correct: LDI distribution
    ldi_newly_correct = [c['ldi'] for c in newly_correct]
    ldi_newly_wrong   = [c['ldi'] for c in newly_wrong]
    if ldi_newly_correct:
        print(f"\n  Newly correct LDI: mean={np.mean(ldi_newly_correct):.3f} ± {np.std(ldi_newly_correct):.3f}")
    if ldi_newly_wrong:
        print(f"  Newly wrong LDI:   mean={np.mean(ldi_newly_wrong):.3f} ± {np.std(ldi_newly_wrong):.3f}")

    # Among CORE domain loss cases: new vs baseline DSR
    core_cases = [cases[k] for k in np.where(core_mask)[0]]
    core_newly_correct = [c for c in core_cases
                          if not (p_base[c['iso_idx'], c['go_idx']] < p_base[c['canon_idx'], c['go_idx']])
                          and (p_new[c['iso_idx'], c['go_idx']] < p_new[c['canon_idx'], c['go_idx']])]
    core_newly_wrong   = [c for c in core_cases
                          if (p_base[c['iso_idx'], c['go_idx']] < p_base[c['canon_idx'], c['go_idx']])
                          and not (p_new[c['iso_idx'], c['go_idx']] < p_new[c['canon_idx'], c['go_idx']])]
    print(f"\n  Core domain loss (LDI≥Q75):")
    print(f"    Newly correct: {len(core_newly_correct)}")
    print(f"    Newly wrong:   {len(core_newly_wrong)}")
    if core_cases:
        print(f"    Net gain in core cases: {len(core_newly_correct)-len(core_newly_wrong):+d} "
              f"/ {len(core_cases)} ({(len(core_newly_correct)-len(core_newly_wrong))/len(core_cases)*100:+.2f}%)")

# ── 9. GO term별 DSR 변화 (어떤 GO에서 개선?) ─────────────────────────────
print("\n[9] GO term별 DSR 변화 (top improved GO terms)...")
if 'confweight_a02' in model_preds:
    p_new = model_preds['confweight_a02']
    go_dsr_base = {}
    go_dsr_new  = {}
    for j in range(n_go):
        cases_j = [c for c in cases if c['go_idx'] == j]
        if len(cases_j) < 10: continue
        dsr_b, _, _ = compute_dsr(cases_j, p_base)
        dsr_n, _, _ = compute_dsr(cases_j, p_new)
        go_dsr_base[j] = dsr_b
        go_dsr_new[j]  = dsr_n

    deltas = {j: go_dsr_new[j] - go_dsr_base[j] for j in go_dsr_base}
    print(f"  {'GO term':<20} {'baseline DSR':>13} {'new DSR':>10} {'Δ':>6} n")
    for j, delta in sorted(deltas.items(), key=lambda x: -x[1])[:10]:
        cases_j = [c for c in cases if c['go_idx'] == j]
        print(f"  {mf_terms[j]:<20} {go_dsr_base[j]:>12.4f} {go_dsr_new[j]:>9.4f} {delta:>+5.3f} {len(cases_j)}")

    print(f"\n  {'GO term':<20} {'baseline DSR':>13} {'new DSR':>10} {'Δ':>6} n  (top degraded)")
    for j, delta in sorted(deltas.items(), key=lambda x: x[1])[:5]:
        cases_j = [c for c in cases if c['go_idx'] == j]
        print(f"  {mf_terms[j]:<20} {go_dsr_base[j]:>12.4f} {go_dsr_new[j]:>9.4f} {delta:>+5.3f} {len(cases_j)}")

# ── 10. 저장 ─────────────────────────────────────────────────────────────
print("\n[10] 요약 저장...")
out = {
    'n_cases_total': len(cases),
    'n_cases_core': int(core_mask.sum()),
    'n_cases_peripheral': int(peripheral_mask.sum()),
    'n_cases_type3_baseline': len(type3_cases),
    'ldi_q25': float(q25), 'ldi_q50': float(q50), 'ldi_q75': float(q75),
    'models': {}
}
for mname, preds in model_preds.items():
    dsr, margin, n = compute_dsr(cases, preds)
    strat = compute_dsr_stratified(cases, ldi_vals, preds)
    out['models'][mname] = {
        'overall_dsr': dsr, 'margin': margin,
        'stratified': {k: v for k, v in strat.items()}
    }
out['type3_baseline_dsr'] = dsr_t3

with open(OUT_DIR / "domain_sensitivity_eval.json", 'w') as f:
    json.dump(out, f, indent=2, default=float)
print(f"  Saved: {OUT_DIR}/domain_sensitivity_eval.json")
