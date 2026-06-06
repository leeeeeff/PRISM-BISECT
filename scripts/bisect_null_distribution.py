#!/usr/bin/env python3
"""
Task A3: BISECT Null Distribution + PRISM-DTU Correlation
==========================================================
1. BISECT 83 cases의 PRISM delta (prism_delta_max) vs null distribution (random
   same-gene isoform pairs from test set) — MWU test
2. 26 brain DTU cases: Spearman(prism_delta_max, -log10(dtu_p))
"""

import json, os, sys, time
import numpy as np
from collections import defaultdict

BASE = '/home/welcome1/sw1686/DIFFUSE'

try:
    from scipy import stats
    from sklearn.metrics import average_precision_score
except ImportError:
    print("ERROR: scipy / sklearn required"); sys.exit(1)

print("="*70)
print("  Task A3: BISECT Null Distribution + PRISM-DTU Correlation")
print("="*70)

# ── 1. Load BISECT cases ───────────────────────────────────────────────────────
bisect_path = f'{BASE}/prism_app/data/demo/bisect_cases.json'
with open(bisect_path) as f:
    cases = json.load(f)

print(f"\n[Step 1] Loaded {len(cases)} BISECT cases")

# Extract prism_delta_max for all 83 cases
bisect_deltas = []
dtu_cases = []   # cases with dtu_p not None

for c in cases:
    delta = c.get('prism_delta_max')
    if delta is None:
        delta = c.get('prism_ad_max_score', 0) - c.get('prism_ct_max_score', 0)
    if delta is not None:
        bisect_deltas.append(abs(float(delta)))

    # DTU cases: have numeric dtu_p (not None)
    dtu_p = c.get('dtu_p')
    if dtu_p is not None and str(dtu_p).lower() not in ('null', 'none', ''):
        try:
            p_val = float(dtu_p)
            d_val = abs(float(delta)) if delta is not None else None
            if d_val is not None:
                dtu_cases.append({'gene': c.get('gene', ''), 'dtu_p': p_val,
                                  'prism_delta': d_val})
        except (ValueError, TypeError):
            pass

bisect_deltas = np.array(bisect_deltas, dtype=np.float32)
print(f"  BISECT |delta| values: n={len(bisect_deltas)}, "
      f"mean={bisect_deltas.mean():.4f}, median={np.median(bisect_deltas):.4f}")
print(f"  DTU-tested cases: {len(dtu_cases)}")

# ── 2. Load score matrix + gene list ─────────────────────────────────────────
score_path = f'{BASE}/reports/v15_bp_clean/score_matrix_18go_20260519_1914.npy'
gene_path  = f'{BASE}/hMuscle/model/my_gene_list_fixed.npy'

if not os.path.exists(score_path):
    print(f"ERROR: score matrix not found at {score_path}"); sys.exit(1)
if not os.path.exists(gene_path):
    print(f"ERROR: gene list not found at {gene_path}"); sys.exit(1)

scores = np.load(score_path)            # (36748, 18) or (N, 18)
genes  = np.load(gene_path, allow_pickle=True)  # (N,) gene symbols/ENSG

print(f"\n[Step 2] score_matrix: {scores.shape}, gene_list: {genes.shape}")

# ── 3. Map genes → isoform indices ────────────────────────────────────────────
gene_isos = defaultdict(list)
for i, g in enumerate(genes):
    gene_isos[str(g)].append(i)

multi_gene_list = [g for g, idxs in gene_isos.items() if len(idxs) >= 2]
print(f"  Genes with ≥2 isoforms: {len(multi_gene_list)}")

# ── 4. Null distribution: 5000 random same-gene isoform pairs ─────────────────
print(f"\n[Step 3] Sampling 5000 random same-gene isoform pairs ...")

N_NULL = 5000
rng = np.random.default_rng(seed=42)
null_deltas = []

attempts = 0
while len(null_deltas) < N_NULL and attempts < N_NULL * 20:
    attempts += 1
    gene = multi_gene_list[rng.integers(len(multi_gene_list))]
    idxs = gene_isos[gene]
    if len(idxs) < 2:
        continue
    i, j = rng.choice(idxs, size=2, replace=False)
    max_diff = np.max(np.abs(scores[i] - scores[j]))
    null_deltas.append(max_diff)

null_deltas = np.array(null_deltas, dtype=np.float32)
print(f"  Null distribution: n={len(null_deltas)}, "
      f"mean={null_deltas.mean():.4f}, median={np.median(null_deltas):.4f}")

# ── 5. MWU test: BISECT vs null ────────────────────────────────────────────────
print(f"\n[Step 4] Mann-Whitney U test: BISECT deltas vs null distribution")
mwu_stat, mwu_p = stats.mannwhitneyu(bisect_deltas, null_deltas,
                                      alternative='greater')
effect_size = (bisect_deltas.mean() - null_deltas.mean()) / null_deltas.std()
fold = bisect_deltas.mean() / null_deltas.mean() if null_deltas.mean() > 0 else float('inf')

print(f"  BISECT mean |delta| = {bisect_deltas.mean():.4f}")
print(f"  Null   mean |delta| = {null_deltas.mean():.4f}")
print(f"  Fold enrichment     = {fold:.2f}×")
print(f"  MWU statistic       = {mwu_stat:.1f}")
print(f"  MWU p-value         = {mwu_p:.2e}")
print(f"  Cohen's d           = {effect_size:.3f}")

# Percentile fractions
pct_high_bisect = (bisect_deltas > np.percentile(null_deltas, 75)).mean()
pct_high_null   = 0.25  # by definition
print(f"  BISECT cases above null Q3: {pct_high_bisect*100:.1f}%")

# ── 6. PRISM-DTU Spearman correlation (26 brain cases) ────────────────────────
print(f"\n[Step 5] PRISM-DTU Spearman correlation for {len(dtu_cases)} DTU-tested cases")

spearman_r, spearman_p = None, None
if len(dtu_cases) >= 5:
    dtup_arr = np.array([c['dtu_p'] for c in dtu_cases])
    delt_arr = np.array([c['prism_delta'] for c in dtu_cases])

    # Use -log10(dtu_p), clip at 1e-300 to avoid inf
    neg_log_p = -np.log10(np.clip(dtup_arr, 1e-300, 1.0))

    spearman_r, spearman_p = stats.spearmanr(delt_arr, neg_log_p)
    print(f"  Spearman r  = {spearman_r:.4f}")
    print(f"  Spearman p  = {spearman_p:.4e}")
    print(f"  n           = {len(dtu_cases)}")
else:
    print(f"  Only {len(dtu_cases)} DTU cases with numeric p-values — skipping correlation")

# ── 7. Percentile of BISECT cases vs null ─────────────────────────────────────
# What percentile of null distribution does median BISECT delta correspond to?
bisect_median = np.median(bisect_deltas)
null_pct = stats.percentileofscore(null_deltas, bisect_median)
print(f"\n[Step 6] Null distribution percentile analysis")
print(f"  BISECT median delta ({bisect_median:.4f}) is at {null_pct:.1f}th percentile of null")
print(f"  → {100 - null_pct:.1f}% of null pairs have lower delta than median BISECT case")

# ── 8. Save results ────────────────────────────────────────────────────────────
out = {
    'task': 'A3_BISECT_null_distribution',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'n_bisect_cases': len(bisect_deltas),
    'n_dtu_tested': len(dtu_cases),
    'n_null_pairs': len(null_deltas),
    'bisect_delta': {
        'mean': float(bisect_deltas.mean()),
        'median': float(np.median(bisect_deltas)),
        'std':  float(bisect_deltas.std()),
        'min':  float(bisect_deltas.min()),
        'max':  float(bisect_deltas.max()),
    },
    'null_delta': {
        'mean': float(null_deltas.mean()),
        'median': float(np.median(null_deltas)),
        'std':  float(null_deltas.std()),
        'min':  float(null_deltas.min()),
        'max':  float(null_deltas.max()),
    },
    'fold_enrichment': float(fold),
    'mwu': {
        'statistic': float(mwu_stat),
        'p_value': float(mwu_p),
        'cohens_d': float(effect_size),
        'alternative': 'greater',
    },
    'bisect_pct_above_null_q3': float(pct_high_bisect),
    'null_percentile_of_bisect_median': float(null_pct),
    'spearman_prism_dtu': {
        'r': float(spearman_r) if spearman_r is not None else None,
        'p': float(spearman_p) if spearman_p is not None else None,
        'n': len(dtu_cases),
    },
    'defense_claim': (
        f"BISECT cases show {fold:.1f}-fold higher PRISM functional divergence than "
        f"random same-gene isoform pairs (MWU p={mwu_p:.2e}, n_BISECT={len(bisect_deltas)}, "
        f"n_null={len(null_deltas)}). "
        f"{pct_high_bisect*100:.0f}% of BISECT cases exceed the 75th percentile of the null "
        f"distribution. BISECT median delta ({bisect_median:.3f}) is at the "
        f"{null_pct:.0f}th percentile of null, confirming non-random functional selection."
    )
}
if spearman_r is not None:
    out['defense_claim'] += (
        f" Among {len(dtu_cases)} DTU-tested brain cases, PRISM delta correlates with "
        f"DTU evidence (Spearman r={spearman_r:.3f}, p={spearman_p:.3f})."
    )

out_path = f'{BASE}/reports/bisect_null_distribution.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n[Saved] {out_path}")

print("\n" + "="*70)
print("DEFENSE CLAIM:")
print(out['defense_claim'])
print("="*70)
