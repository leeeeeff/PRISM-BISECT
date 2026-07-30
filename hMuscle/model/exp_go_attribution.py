"""
exp_go_attribution.py — GO Attribution Structure Analysis
==========================================================
Gene X에 GO term A, B, C가 할당될 때:
  Scenario 1: Canonical 아이소폼 하나가 A+B+C 모두 담당 (다기능 단백질)
  Scenario 2: 아이소폼-1→A, 아이소폼-2→B, 아이소폼-3→C (기능 분화)

분석 지표:
  1. Within-gene GO-score Spearman correlation
     - 모든 GO term이 같은 아이소폼에 집중 → 높은 상관관계 (Scenario 1)
     - GO term별로 다른 아이소폼이 top → 낮거나 음의 상관관계 (Scenario 2)

  2. Top-isoform consistency
     - 모든 GO term에서 동일 아이소폼이 top → Scenario 1
     - GO term별로 다른 아이소폼이 top → Scenario 2

  3. Domain-stratified analysis
     - Domain variation 있는 유전자 vs 없는 유전자에서 위 패턴 비교
     - Type1 (domain-loss) 아이소폼 포함 유전자: 더 강한 Scenario 2 예상

  4. Functional specificity index (FSI)
     - FSI(gene) = 1 - mean(pairwise GO-score correlation within gene)
     - FSI=0: 모든 GO term이 동일 아이소폼에 집중 (Scenario 1)
     - FSI=1: GO term들이 서로 다른 아이소폼에 분화 (Scenario 2)
"""

import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import average_precision_score
import csv, json

ROOT     = Path("/home/welcome1/sw1686/DIFFUSE")
OUT_DIR  = ROOT / "reports/isoform_resolution_full"
FEAT_DIR = ROOT / "hMuscle/results_isoform/features"

print("[1] 데이터 로드...")
preds = np.load(ROOT / "reports/v17f_star_bootstrap/v17f_star_preds.npy")   # (36748, 82)
Y     = np.load(ROOT / "reports/v17f_star_bootstrap/Y_te.npy")               # (36748, 82)

gene_raw = np.load(ROOT / "hMuscle/model/my_gene_list_fixed.npy", allow_pickle=True)
iso_raw  = np.load(ROOT / "hMuscle/model/my_isoform_list_fixed.npy", allow_pickle=True)
gene_list = [x.decode() if isinstance(x, bytes) else x for x in gene_raw]
iso_list  = [x.decode() if isinstance(x, bytes) else x for x in iso_raw]
gene_arr  = np.array(gene_list)

dm = np.load(FEAT_DIR / "domain_matrix_proper_test_v3.npy")                  # (36748, 512)

ft_arr = []
with open(OUT_DIR / "full_isoform_feature_types.tsv") as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        ft_arr.append(row['feature_type'])
ft_arr = np.array(ft_arr)

n_iso, n_go = preds.shape
print(f"  isoforms={n_iso}, GO terms={n_go}")

# ── 2. Multi-isoform gene 목록 구축 ────────────────────────────────────
print("[2] Multi-isoform genes with ≥2 positive GO terms...")
genes_uniq = list(set(gene_list))
gene_to_idx = {}
for i, g in enumerate(gene_list):
    gene_to_idx.setdefault(g, []).append(i)

multi_genes = {g: idxs for g, idxs in gene_to_idx.items() if len(idxs) >= 2}
print(f"  Multi-isoform genes: {len(multi_genes)}")

# ── 3. Within-gene GO-score correlation analysis ───────────────────────
print("[3] Within-gene GO-score Spearman correlation...")

results = []

for g, idxs in multi_genes.items():
    idxs = np.array(idxs)
    n_iso_g = len(idxs)
    if n_iso_g < 2: continue

    # GO terms that are positive for this gene
    y_g = Y[idxs]  # (n_iso_g, n_go)
    # Gene-level positive GO terms (any isoform labeled positive)
    pos_go = np.where(y_g.any(0))[0]  # indices of positive GO terms
    n_pos_go = len(pos_go)
    if n_pos_go < 2: continue  # need at least 2 GO terms

    scores_g = preds[idxs][:, pos_go]  # (n_iso_g, n_pos_go)

    # ── 3a. Pairwise GO-score Spearman correlation ──────────────────
    # For each pair of GO terms: Spearman ρ of scores across isoforms
    if n_iso_g < 3:
        # Can't compute meaningful correlation with only 2 isoforms
        mean_go_corr = float('nan')
    else:
        go_corrs = []
        for a in range(n_pos_go):
            for b in range(a+1, n_pos_go):
                sa = scores_g[:, a]; sb = scores_g[:, b]
                if sa.std() < 1e-6 or sb.std() < 1e-6: continue
                rho, _ = stats.spearmanr(sa, sb)
                go_corrs.append(rho)
        mean_go_corr = float(np.mean(go_corrs)) if go_corrs else float('nan')

    # ── 3b. Top-isoform consistency ──────────────────────────────────
    # For each GO term, which isoform ranks #1?
    top_iso_per_go = scores_g.argmax(0)  # (n_pos_go,)
    # Are they all the same isoform?
    mode_top = int(stats.mode(top_iso_per_go, keepdims=True).mode[0])
    top_consistency = (top_iso_per_go == mode_top).mean()  # 1.0 = Scenario 1

    # ── 3c. Functional Specificity Index ────────────────────────────
    # FSI = 1 - mean(pairwise GO-score correlation)
    fsi = 1.0 - mean_go_corr if not np.isnan(mean_go_corr) else float('nan')

    # ── 3d. Score variance decomposition ────────────────────────────
    # Within-gene variance of scores: between-GO vs within-GO across isoforms
    # If scores vary more across GO terms than across isoforms → Scenario 1
    # If scores vary more across isoforms within each GO → Scenario 2
    score_var_across_isoforms = scores_g.var(0).mean()   # mean var across isoforms, per GO
    score_var_across_go       = scores_g.var(1).mean()   # mean var across GOs, per isoform
    specialization_ratio = score_var_across_isoforms / (score_var_across_go + 1e-8)
    # High ratio → scores vary more across isoforms per GO → different isoforms for different GOs
    # Low ratio → scores vary more across GO terms per isoform → one isoform, many GO scores

    # ── 3e. Domain structure ─────────────────────────────────────────
    ft_g = ft_arr[idxs]
    has_domain_loss = ('Type1_DomainLoss' in ft_g) or ('Type2_PartialTrunc' in ft_g)
    n_types = len(set(ft_g))
    canonical_dc = dm[idxs].sum(1).max()

    results.append({
        'gene': g,
        'n_isoforms': n_iso_g,
        'n_pos_go': n_pos_go,
        'mean_go_corr': mean_go_corr,
        'fsi': fsi,
        'top_consistency': top_consistency,
        'specialization_ratio': specialization_ratio,
        'has_domain_loss': has_domain_loss,
        'n_types': n_types,
        'canonical_dc': canonical_dc,
    })

results = [r for r in results if not np.isnan(r['mean_go_corr'])]
print(f"  Analyzed genes: {len(results)}")

# ── 4. 시나리오 분류 ──────────────────────────────────────────────────
print("[4] Scenario classification...")

go_corrs   = np.array([r['mean_go_corr'] for r in results])
fsi_vals   = np.array([r['fsi'] for r in results])
top_cons   = np.array([r['top_consistency'] for r in results])
spec_ratio = np.array([r['specialization_ratio'] for r in results])
dom_loss   = np.array([r['has_domain_loss'] for r in results])

# Thresholds
HIGH_CORR  = 0.7  # Scenario 1: all GO terms track together
LOW_CORR   = 0.3  # Scenario 2: GO terms go to different isoforms

s1_mask = (go_corrs >= HIGH_CORR) & (top_cons >= 0.8)  # canonical-driven
s2_mask = (go_corrs <  LOW_CORR)                         # isoform-specialized
mixed   = ~s1_mask & ~s2_mask

print(f"\n  Scenario 1 (canonical carries all GO): n={s1_mask.sum()} ({s1_mask.mean()*100:.1f}%)")
print(f"  Scenario 2 (isoforms specialize in GO): n={s2_mask.sum()} ({s2_mask.mean()*100:.1f}%)")
print(f"  Mixed/intermediate:                     n={mixed.sum()}  ({mixed.mean()*100:.1f}%)")

print(f"\n  Mean GO-score correlation:")
print(f"    Overall:    {go_corrs.mean():.4f} ± {go_corrs.std():.4f}")
print(f"    S1 genes:   {go_corrs[s1_mask].mean():.4f}")
print(f"    S2 genes:   {go_corrs[s2_mask].mean():.4f}")

print(f"\n  Top-isoform consistency:")
print(f"    Overall:    {top_cons.mean():.4f}")
print(f"    S1 genes:   {top_cons[s1_mask].mean():.4f}")
print(f"    S2 genes:   {top_cons[s2_mask].mean():.4f}")

# ── 5. Domain variation과의 관계 ─────────────────────────────────────
print("\n[5] Domain variation → GO specialization 관계...")
dom_loss_mask = dom_loss == True
no_dom_loss   = dom_loss == False

print(f"\n  Domain-variable genes (with Type1/2):")
print(f"    n = {dom_loss_mask.sum()}")
print(f"    Mean GO correlation = {go_corrs[dom_loss_mask].mean():.4f}")
print(f"    S2 fraction = {s2_mask[dom_loss_mask].mean()*100:.1f}%")
print(f"\n  Domain-invariant genes:")
print(f"    n = {no_dom_loss.sum()}")
print(f"    Mean GO correlation = {go_corrs[no_dom_loss].mean():.4f}")
print(f"    S2 fraction = {s2_mask[no_dom_loss].mean()*100:.1f}%")

mwu_stat, mwu_p = stats.mannwhitneyu(go_corrs[dom_loss_mask], go_corrs[no_dom_loss], alternative='less')
print(f"\n  MWU (domain-var < no-domain-var in GO correlation): p = {mwu_p:.4e}")

# ── 6. n_pos_go별 분포 ────────────────────────────────────────────────
print("\n[6] n_pos_go (GO term 수)별 시나리오 분포...")
n_pos_go_arr = np.array([r['n_pos_go'] for r in results])
for n in [2, 3, 4, 5]:
    mask = (n_pos_go_arr == n)
    if mask.sum() == 0: continue
    print(f"  n_pos_go={n}: n={mask.sum()}, S1={s1_mask[mask].mean()*100:.1f}%, "
          f"S2={s2_mask[mask].mean()*100:.1f}%, mean_corr={go_corrs[mask].mean():.4f}")
mask6p = (n_pos_go_arr >= 6)
if mask6p.sum() > 0:
    print(f"  n_pos_go≥6: n={mask6p.sum()}, S1={s1_mask[mask6p].mean()*100:.1f}%, "
          f"S2={s2_mask[mask6p].mean()*100:.1f}%, mean_corr={go_corrs[mask6p].mean():.4f}")

# ── 7. 극단 사례 예시 ─────────────────────────────────────────────────
print("\n[7] Scenario 2 대표 유전자 (GO term별 다른 아이소폼이 top)...")
s2_indices = np.where(s2_mask)[0]
s2_results = sorted([results[i] for i in s2_indices],
                     key=lambda x: x['mean_go_corr'])[:10]
for r in s2_results:
    g = r['gene']
    idxs = np.array(gene_to_idx[g])
    y_g = Y[idxs]; scores_g = preds[idxs]
    pos_go = np.where(y_g.any(0))[0]
    top_iso = scores_g[:, pos_go].argmax(0)
    # Which iso is top for which GO?
    iso_names = [iso_list[i][:25] for i in idxs]
    go_str = ' '.join([f"GO[{j}]→iso{t}" for j, t in zip(pos_go[:4], top_iso[:4])])
    g_short = g.split('.')[0]
    print(f"  {g_short}: corr={r['mean_go_corr']:.3f}, {r['n_isoforms']}iso, "
          f"{r['n_pos_go']}GO | {go_str}")

print("\n[8] Scenario 1 대표 유전자 (canonical이 모든 GO 담당)...")
s1_indices = np.where(s1_mask)[0]
s1_results = sorted([results[i] for i in s1_indices],
                     key=lambda x: -x['mean_go_corr'])[:10]
for r in s1_results:
    g = r['gene']
    g_short = g.split('.')[0]
    print(f"  {g_short}: corr={r['mean_go_corr']:.3f}, {r['n_isoforms']}iso, "
          f"{r['n_pos_go']}GO, dom_loss={r['has_domain_loss']}")

# ── 8. 요약 통계 저장 ─────────────────────────────────────────────────
print("\n[9] 요약 저장...")
summary = {
    'n_analyzed': len(results),
    'scenario1_n': int(s1_mask.sum()),
    'scenario1_frac': float(s1_mask.mean()),
    'scenario2_n': int(s2_mask.sum()),
    'scenario2_frac': float(s2_mask.mean()),
    'mixed_n': int(mixed.sum()),
    'mean_go_corr_overall': float(go_corrs.mean()),
    'mean_go_corr_s1': float(go_corrs[s1_mask].mean()),
    'mean_go_corr_s2': float(go_corrs[s2_mask].mean()),
    'mean_go_corr_dom_var': float(go_corrs[dom_loss_mask].mean()),
    'mean_go_corr_no_dom_var': float(go_corrs[no_dom_loss].mean()),
    'mwu_p_domain_vs_nodom': float(mwu_p),
    'top_consistency_overall': float(top_cons.mean()),
    'top_consistency_s1': float(top_cons[s1_mask].mean()),
    'top_consistency_s2': float(top_cons[s2_mask].mean()),
}
with open(OUT_DIR / "go_attribution_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {OUT_DIR}/go_attribution_summary.json")

# TSV 저장 (gene-level)
with open(OUT_DIR / "go_attribution_per_gene.tsv", 'w') as f:
    header = ['gene','n_isoforms','n_pos_go','mean_go_corr','fsi',
              'top_consistency','specialization_ratio','has_domain_loss','scenario']
    f.write('\t'.join(header) + '\n')
    for r, s1, s2 in zip(results, s1_mask, s2_mask):
        sc = 'S1' if s1 else ('S2' if s2 else 'Mixed')
        row = [r['gene'], str(r['n_isoforms']), str(r['n_pos_go']),
               f"{r['mean_go_corr']:.4f}", f"{r['fsi']:.4f}",
               f"{r['top_consistency']:.4f}", f"{r['specialization_ratio']:.4f}",
               str(r['has_domain_loss']), sc]
        f.write('\t'.join(row) + '\n')
print(f"  Saved: {OUT_DIR}/go_attribution_per_gene.tsv")
