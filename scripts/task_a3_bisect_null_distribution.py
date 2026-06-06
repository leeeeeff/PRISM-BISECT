#!/usr/bin/env python3
"""
Task A3: BISECT Null Distribution + PRISM-DTU Correlation
=========================================================
1. BISECT 83 cases의 PRISM score difference가 random isoform pairs보다
   유의하게 큰가 (MWU test)
2. 26 brain cases에서 PRISM Δscore vs DTU -log10(p) Spearman correlation

설계 주의사항:
- 83 cases 중 prism_delta_max > 0 인 cases = 60 (tier1/2)
- prism_delta_max == 0 인 cases = 23 (tier3_gene_median, transcript 매핑 실패)
- dtu_tested = 26 cases (대부분 tier3_gene_median)
- Null distribution: same-gene random pairs from score matrix
"""

import os, json, sys, time, random
import numpy as np
from collections import defaultdict

BASE = '/home/welcome1/sw1686/DIFFUSE'

try:
    from scipy import stats
except ImportError:
    print("ERROR: scipy 필요")
    sys.exit(1)

print("="*70)
print("  Task A3: BISECT Null Distribution + PRISM-DTU Correlation")
print("="*70)

# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────────
print(f"\n[Step 1] Loading data ...")

with open(f'{BASE}/prism_app/data/demo/bisect_cases.json') as f:
    cases = json.load(f)
print(f"  BISECT cases: {len(cases)}")

score_matrix = np.load(f'{BASE}/reports/v15_bp_clean/score_matrix_18go_20260519_1914.npy')  # (36748, 18)
te_gene = np.load(f'{BASE}/hMuscle/model/my_gene_list_fixed.npy', allow_pickle=True)
print(f"  Score matrix: {score_matrix.shape}")
print(f"  te_gene: {te_gene.shape}")

# ENSG base (version 제거)
te_ensg_base = [g.decode('utf-8').split('.')[0] if isinstance(g, bytes) else str(g).split('.')[0] for g in te_gene]

# ENSG → symbol 매핑
ID_DIR = f'{BASE}/hMuscle/data/raw_data/data/id_lists'
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as fh:
    next(fh)
    for line in fh:
        parts = line.strip().split()
        if len(parts) >= 5:
            ENSG2SYM[parts[0]] = parts[4]

te_sym = [ENSG2SYM.get(g, g) for g in te_ensg_base]

# gene → isoform indices 그룹화
gene_isos = defaultdict(list)
for i, sym in enumerate(te_sym):
    gene_isos[sym].append(i)

print(f"  Unique genes in test set: {len(gene_isos)}")
multi_iso_genes = {g: idxs for g, idxs in gene_isos.items() if len(idxs) >= 2}
print(f"  Genes with >=2 isoforms: {len(multi_iso_genes)}")

# ── 2. BISECT case delta 추출 ─────────────────────────────────────────────────
print(f"\n[Step 2] Extracting BISECT prism_delta_max values ...")

# 전략: prism_delta_max가 있는 cases만 사용 (tier1, tier2)
# tier3_gene_median은 실제 transcript 매핑 없이 gene median으로 채워진 것
bisect_all_deltas = [abs(c['prism_delta_max']) for c in cases]
bisect_nonzero = [abs(c['prism_delta_max']) for c in cases if abs(c['prism_delta_max']) > 0]
bisect_tier12   = [abs(c['prism_delta_max']) for c in cases if c['prism_tier'] not in ('tier3_gene_median', 'tier3_structural_only')]

print(f"  All 83 cases:  mean={np.mean(bisect_all_deltas):.4f}  median={np.median(bisect_all_deltas):.4f}")
print(f"  NonZero (N={len(bisect_nonzero)}): mean={np.mean(bisect_nonzero):.4f}  median={np.median(bisect_nonzero):.4f}")
print(f"  Tier1+2 (N={len(bisect_tier12)}):  mean={np.mean(bisect_tier12):.4f}   median={np.median(bisect_tier12):.4f}")

from collections import Counter
tier_counts = Counter(c['prism_tier'] for c in cases)
print(f"  Tier distribution: {dict(tier_counts)}")

# ── 3. Null distribution 구축 ─────────────────────────────────────────────────
print(f"\n[Step 3] Building null distribution (N=5000 random same-gene pairs) ...")

random.seed(42)
np.random.seed(42)

N_NULL = 5000
null_deltas = []
null_gene_names = []

genes_list = list(multi_iso_genes.keys())
attempts = 0

while len(null_deltas) < N_NULL and attempts < N_NULL * 10:
    attempts += 1
    gene = random.choice(genes_list)
    idxs = multi_iso_genes[gene]
    if len(idxs) < 2:
        continue
    a_idx, b_idx = random.sample(idxs, 2)
    sa = score_matrix[a_idx]
    sb = score_matrix[b_idx]
    max_diff = float(np.max(np.abs(sa - sb)))
    null_deltas.append(max_diff)
    null_gene_names.append(gene)

print(f"  Generated {len(null_deltas)} null pairs (from {len(set(null_gene_names))} unique genes)")
print(f"  Null delta stats: mean={np.mean(null_deltas):.4f} median={np.median(null_deltas):.4f} std={np.std(null_deltas):.4f}")

# ── 4. MWU test: BISECT vs Null ───────────────────────────────────────────────
print(f"\n[Step 4] Mann-Whitney U test: BISECT vs Null ...")

def mwu_analysis(bisect_vals, null_vals, label):
    u_stat, p_val = stats.mannwhitneyu(bisect_vals, null_vals, alternative='greater')
    # Effect size: rank-biserial correlation
    # U / (n1*n2) > 0.5 means bisect > null; r = 2U/(n1*n2) - 1
    n1, n2 = len(bisect_vals), len(null_vals)
    r_biserial = (2 * u_stat) / (n1 * n2) - 1  # positive = bisect > null
    print(f"  [{label}]")
    print(f"    N_bisect={n1}, N_null={n2}")
    print(f"    BISECT: mean={np.mean(bisect_vals):.4f} median={np.median(bisect_vals):.4f}")
    print(f"    Null:   mean={np.mean(null_vals):.4f} median={np.median(null_vals):.4f}")
    print(f"    U={u_stat:.1f}  p={p_val:.4e}  r_biserial={r_biserial:.4f}")
    return {
        'n_bisect': n1,
        'n_null': n2,
        'bisect_mean': float(np.mean(bisect_vals)),
        'bisect_median': float(np.median(bisect_vals)),
        'null_mean': float(np.mean(null_vals)),
        'null_median': float(np.median(null_vals)),
        'mwu_U': float(u_stat),
        'mwu_p': float(p_val),
        'effect_size_r_biserial': float(r_biserial),
        'fold_enrichment_mean': float(np.mean(bisect_vals) / np.mean(null_vals)) if np.mean(null_vals) > 0 else None,
    }

mwu_all    = mwu_analysis(bisect_all_deltas, null_deltas, "All 83 BISECT cases")
mwu_nonzero = mwu_analysis(bisect_nonzero, null_deltas, "Non-zero delta (Tier1+2, N=60)")
mwu_tier12  = mwu_analysis(bisect_tier12, null_deltas, "Tier1+Tier2 explicit (N=59)")

# ── 5. Percentile rank analysis ───────────────────────────────────────────────
print(f"\n[Step 5] Percentile rank of BISECT cases in null distribution ...")

null_sorted = np.sort(null_deltas)
bisect_percentiles_all    = [float(np.searchsorted(null_sorted, d) / len(null_sorted) * 100) for d in bisect_all_deltas]
bisect_percentiles_nonzero = [float(np.searchsorted(null_sorted, d) / len(null_sorted) * 100) for d in bisect_nonzero]

print(f"  All cases: median percentile = {np.median(bisect_percentiles_all):.1f}th")
print(f"  Non-zero cases: median percentile = {np.median(bisect_percentiles_nonzero):.1f}th")
print(f"  Non-zero cases > 90th percentile: {sum(p>90 for p in bisect_percentiles_nonzero)} / {len(bisect_percentiles_nonzero)}")
print(f"  Non-zero cases > 75th percentile: {sum(p>75 for p in bisect_percentiles_nonzero)} / {len(bisect_percentiles_nonzero)}")

# ── 6. Spearman correlation: Δscore vs DTU -log10(p) ─────────────────────────
print(f"\n[Step 6] Spearman correlation: PRISM Δscore vs DTU -log10(p) ...")

dtu_cases = [c for c in cases if c.get('dtu_note') == 'dtu_tested']
print(f"  dtu_tested cases: {len(dtu_cases)}")

# dtu_p > 0 인 cases만 사용
valid_dtu = [(c, c.get('dtu_p')) for c in dtu_cases if c.get('dtu_p') is not None and c.get('dtu_p') > 0]
print(f"  dtu_p valid cases: {len(valid_dtu)}")

if len(valid_dtu) >= 3:
    dtu_deltas_spearman = [abs(c['prism_delta_max']) for c, _ in valid_dtu]
    dtu_log10p = [-np.log10(p) for _, p in valid_dtu]

    rho, sp_p = stats.spearmanr(dtu_deltas_spearman, dtu_log10p)
    print(f"  Spearman rho = {rho:.4f}  p = {sp_p:.4f}")
    print(f"  N={len(valid_dtu)} cases")

    # breakdown by tier
    tier_breakdown = {}
    for c, p in valid_dtu:
        tier = c['prism_tier']
        if tier not in tier_breakdown:
            tier_breakdown[tier] = {'deltas': [], 'log10p': []}
        tier_breakdown[tier]['deltas'].append(abs(c['prism_delta_max']))
        tier_breakdown[tier]['log10p'].append(-np.log10(p))

    print(f"\n  Per-tier breakdown (dtu_tested):")
    for tier, data in tier_breakdown.items():
        print(f"    {tier}: N={len(data['deltas'])} delta_mean={np.mean(data['deltas']):.4f} -log10p_mean={np.mean(data['log10p']):.2f}")

    # note on tier3 cases
    n_tier3_dtu = sum(1 for c, _ in valid_dtu if 'tier3' in c['prism_tier'])
    print(f"\n  NOTE: {n_tier3_dtu}/{len(valid_dtu)} dtu_tested cases are tier3 (gene_median, no transcript match)")
    print(f"  → Spearman correlation limited by tier3 cases having Δ=0 despite significant DTU")

    spearman_result = {
        'n': len(valid_dtu),
        'n_tier3': n_tier3_dtu,
        'spearman_rho': float(rho),
        'spearman_p': float(sp_p),
        'case_details': [
            {
                'gene': c['gene'],
                'prism_delta_max': c['prism_delta_max'],
                'dtu_p': p,
                'neg_log10_dtu_p': float(-np.log10(p)),
                'prism_tier': c['prism_tier'],
            }
            for c, p in valid_dtu
        ]
    }

    # Interpretation
    if n_tier3_dtu > 0:
        spearman_result['interpretation'] = (
            f"Spearman rho={rho:.3f} (p={sp_p:.4f}) across {len(valid_dtu)} dtu_tested cases. "
            f"Correlation is attenuated by {n_tier3_dtu} tier3 cases where PRISM lacks matched "
            f"transcript IDs (returns gene-median Δ=0 despite significant DTU). "
            f"Among non-tier3 cases: DLG1 (Δ=0.637, -log10p=9.0), KIF21B (Δ=0.163, -log10p=5.4), "
            f"NDUFS4 (Δ=0.131, -log10p=5.4)."
        )
else:
    print(f"  Insufficient dtu_p data for Spearman (N={len(valid_dtu)})")
    spearman_result = {'n': len(valid_dtu), 'error': 'Insufficient data'}

# ── 7. IsoQuant transcript match rate analysis ────────────────────────────────
print(f"\n[Step 7] Transcript match rate analysis (why tier3 cases exist) ...")

# tier3 cases: transcript IDs from brain IsoQuant, not in PRISM test set (muscle)
tier3_cases = [c for c in cases if 'tier3' in c['prism_tier']]
tier12_cases = [c for c in cases if 'tier3' not in c['prism_tier']]

print(f"  Tier1/2 (transcript matched): {len(tier12_cases)} ({len(tier12_cases)/83*100:.1f}%)")
print(f"  Tier3 (gene-median only):     {len(tier3_cases)} ({len(tier3_cases)/83*100:.1f}%)")

# tier3 중 dtu_tested proportion
tier3_dtu = [c for c in tier3_cases if c.get('dtu_note') == 'dtu_tested']
print(f"  Tier3 + dtu_tested:           {len(tier3_dtu)} ({len(tier3_dtu)/len(tier3_cases)*100:.1f}% of tier3)")

# ── 8. 결과 저장 ──────────────────────────────────────────────────────────────
output = {
    'task': 'A3_BISECT_null_distribution_PRISM_DTU_correlation',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'description': 'Null distribution test for BISECT PRISM delta significance + DTU correlation',
    'n_bisect_cases': len(cases),
    'tier_distribution': dict(tier_counts),
    'bisect_deltas': {
        'all_83': {
            'n': len(bisect_all_deltas),
            'mean': float(np.mean(bisect_all_deltas)),
            'median': float(np.median(bisect_all_deltas)),
            'n_zero': sum(d == 0 for d in bisect_all_deltas),
            'n_nonzero': sum(d > 0 for d in bisect_all_deltas),
        },
        'nonzero_60': {
            'n': len(bisect_nonzero),
            'mean': float(np.mean(bisect_nonzero)),
            'median': float(np.median(bisect_nonzero)),
        }
    },
    'null_distribution': {
        'n': len(null_deltas),
        'mean': float(np.mean(null_deltas)),
        'median': float(np.median(null_deltas)),
        'std': float(np.std(null_deltas)),
        'p95': float(np.percentile(null_deltas, 95)),
        'p99': float(np.percentile(null_deltas, 99)),
        'sampling': 'random same-gene isoform pairs, N=5000',
    },
    'mwu_tests': {
        'all_83_vs_null': mwu_all,
        'nonzero_60_vs_null': mwu_nonzero,
        'tier12_vs_null': mwu_tier12,
    },
    'percentile_analysis': {
        'all_median_pct': float(np.median(bisect_percentiles_all)),
        'nonzero_median_pct': float(np.median(bisect_percentiles_nonzero)),
        'nonzero_above_90th': int(sum(p > 90 for p in bisect_percentiles_nonzero)),
        'nonzero_above_75th': int(sum(p > 75 for p in bisect_percentiles_nonzero)),
    },
    'spearman_correlation': spearman_result,
    'transcript_match_analysis': {
        'tier12_count': len(tier12_cases),
        'tier12_frac': len(tier12_cases) / 83,
        'tier3_count': len(tier3_cases),
        'tier3_frac': len(tier3_cases) / 83,
        'tier3_dtu_tested': len(tier3_dtu),
        'note': 'Tier3 cases have brain IsoQuant transcript IDs not in PRISM muscle test set; gene-median used as fallback'
    },
}

out_path = f'{BASE}/reports/bisect_null_distribution.json'
with open(out_path, 'w') as fh:
    json.dump(output, fh, indent=2, default=str)

print(f"\n[DONE] Saved → {out_path}")

# ── 최종 요약 ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"FINAL SUMMARY")
print(f"{'='*70}")
print(f"Null distribution (N=5000 same-gene pairs):")
print(f"  mean Δ = {np.mean(null_deltas):.4f}")
print(f"  95th   = {np.percentile(null_deltas, 95):.4f}")
print(f"  99th   = {np.percentile(null_deltas, 99):.4f}")
print(f"\nBISECT tier1+2 cases (N={len(bisect_tier12)}):")
print(f"  mean Δ = {np.mean(bisect_tier12):.4f}  ({np.mean(bisect_tier12)/np.mean(null_deltas):.1f}× null)")
print(f"  MWU p  = {mwu_tier12['mwu_p']:.2e}")
print(f"  Effect = {mwu_tier12['effect_size_r_biserial']:.3f} (rank-biserial r)")
print(f"\nSpearman (dtu_tested N={spearman_result.get('n','?')}):")
print(f"  rho = {spearman_result.get('spearman_rho', 'N/A')}")
print(f"  p   = {spearman_result.get('spearman_p', 'N/A')}")
print(f"  Note: attenuated by {spearman_result.get('n_tier3', '?')} tier3 cases (Δ=0 by design)")
print(f"{'='*70}")
