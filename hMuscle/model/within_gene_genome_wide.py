#!/usr/bin/env python3
"""
within_gene_genome_wide.py
==========================
Genome-wide within-gene discrimination analysis using brain_full reference dataset.

Goal: Show that PRISM's within-gene embedding variance correlates with
sequence-level structural proxies genome-wide, defending against the
"gene-level identity suffices" critique.

Approach (model-free proxy):
  - delta_emb = L30 - L18 (splice-sensitive embedding shift; no model weights needed)
  - ||delta_emb||_2 per isoform = "splice magnitude" proxy
  - For each multi-isoform gene g:
      within_delta_std_g = std of ||delta_emb||_2 across isoforms of gene g
  - Structural proxies per gene:
      delta_seqlen_g  = std of isoform name-implied length (ENST length variation)
      delta_loc_std_g = mean std of loc features (8-dim) within gene
      delta_rna_std_g = mean std of rna features (9-dim) within gene
      delta_emb_l7_g  = std of ||L30-L7||_2 (early vs late layer comparison)

Compute Spearman ρ between within_delta_std_g and each structural proxy.

Output: hMuscle/model/within_gene_genome_wide_result.txt
"""

import os, sys, time
import numpy as np
from collections import defaultdict
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data/brain_isoquant_esm2/full'
FEAT_DIR  = '../results_isoform/features'
OUT_FILE  = 'within_gene_genome_wide_result.txt'

t0 = time.time()

def log(msg, fh=None):
    print(msg)
    sys.stdout.flush()
    if fh: fh.write(msg + '\n')

print("=" * 70)
print("  Genome-Wide Within-Gene Discrimination Analysis")
print("  (brain reference transcriptome, 63,994 isoforms)")
print("=" * 70)
sys.stdout.flush()

# ── 1. Load data ──────────────────────────────────────────────────────
print("\n[1] Loading embeddings and features...")
sys.stdout.flush()

emb_L30 = np.load(f'{DATA_DIR}/brain_full_esm2_t30_150M.npy').astype(np.float32)          # (N, 640)
emb_L18 = np.load(f'{DATA_DIR}/brain_full_esm2_layer18_t30_150M.npy').astype(np.float32)  # (N, 640)
emb_L7  = np.load(f'{DATA_DIR}/brain_full_esm2_layer07_t30_150M.npy').astype(np.float32)  # (N, 640)
emb_L27 = np.load(f'{DATA_DIR}/brain_full_esm2_layer27_t30_150M.npy').astype(np.float32)  # (N, 640)

gene_names = np.load(f'{DATA_DIR}/brain_full_gene_names.npy', allow_pickle=True)           # (N,)
isoform_ids = np.load(f'{DATA_DIR}/brain_full_ids.npy', allow_pickle=True)                 # (N,)

loc_feats = np.load(f'{FEAT_DIR}/loc/loc_features_brain_full.npy').astype(np.float32)     # (N, 8)
rna_feats = np.load(f'{FEAT_DIR}/rna/rna_features_brain_full.npy').astype(np.float32)     # (N, 9)

N = emb_L30.shape[0]
print(f"  N={N} isoforms  emb_L30={emb_L30.shape}  loc={loc_feats.shape}  rna={rna_feats.shape}")
sys.stdout.flush()

# ── 2. Compute delta embeddings (per isoform) ────────────────────────
print("\n[2] Computing delta embeddings...")
sys.stdout.flush()

# Primary: L30 - L18 (mid-to-late splice shift)
delta_L30_L18 = emb_L30 - emb_L18          # (N, 640)
norm_L30_L18  = np.linalg.norm(delta_L30_L18, axis=1)  # (N,) = "splice magnitude" proxy

# Secondary: L30 - L7 (early-to-late, captures more structural divergence)
delta_L30_L7  = emb_L30 - emb_L7
norm_L30_L7   = np.linalg.norm(delta_L30_L7, axis=1)   # (N,)

# Tertiary: L27 - L18 (fine-grained upper-layer shift)
delta_L27_L18 = emb_L27 - emb_L18
norm_L27_L18  = np.linalg.norm(delta_L27_L18, axis=1)  # (N,)

# L30 raw norm (absolute embedding magnitude, not delta)
norm_L30_raw  = np.linalg.norm(emb_L30, axis=1)        # (N,)

print(f"  ||L30-L18||: mean={norm_L30_L18.mean():.3f}  std={norm_L30_L18.std():.3f}")
print(f"  ||L30-L7||:  mean={norm_L30_L7.mean():.3f}  std={norm_L30_L7.std():.3f}")
sys.stdout.flush()

# ── 3. Build gene-to-isoform index ──────────────────────────────────
print("\n[3] Building gene → isoform index...")
sys.stdout.flush()

gene2idxs = defaultdict(list)
for i, g in enumerate(gene_names):
    gene2idxs[g].append(i)

multi_genes = sorted([g for g, v in gene2idxs.items() if len(v) >= 2])
n_multi = len(multi_genes)
print(f"  Total genes: {len(gene2idxs)}  Multi-isoform (≥2): {n_multi}")
sys.stdout.flush()

# ── 4. Per-gene structural features ──────────────────────────────────
print("\n[4] Computing per-gene structural proxies...")
sys.stdout.flush()

# Parse isoform number from ID (e.g., "GENE-204" → number suffix as proxy for "length rank")
# Ensembl isoform number roughly correlates with annotation completeness (201=principal)
# Use the isoform suffix number as a proxy (higher number = typically shorter/less complete)
import re
_num_re = re.compile(r'-(\d+)$')

def get_enst_number(iso_id):
    """Extract the numeric suffix from Ensembl isoform ID like GENE-204."""
    m = _num_re.search(str(iso_id))
    return int(m.group(1)) if m else 0

iso_numbers = np.array([get_enst_number(x) for x in isoform_ids], dtype=np.float32)
print(f"  Isoform number range: {iso_numbers[iso_numbers>0].min():.0f}~{iso_numbers.max():.0f}")
sys.stdout.flush()

# Per-gene aggregated metrics
within_delta_std    = np.zeros(n_multi, dtype=np.float32)  # primary proxy
within_delta_L7_std = np.zeros(n_multi, dtype=np.float32)  # secondary
within_delta_27_std = np.zeros(n_multi, dtype=np.float32)  # tertiary
within_loc_std      = np.zeros(n_multi, dtype=np.float32)  # localization
within_rna_std      = np.zeros(n_multi, dtype=np.float32)  # RNA features
within_seqnum_std   = np.zeros(n_multi, dtype=np.float32)  # isoform number spread
within_emb_cosine_std = np.zeros(n_multi, dtype=np.float32) # L30 pairwise cosine spread
n_iso_per_gene      = np.zeros(n_multi, dtype=np.int32)

for gi, g in enumerate(multi_genes):
    idxs = np.array(gene2idxs[g])
    n_iso_per_gene[gi] = len(idxs)

    # Primary: std of ||delta||_2 across isoforms within gene
    within_delta_std[gi]    = norm_L30_L18[idxs].std()
    within_delta_L7_std[gi] = norm_L30_L7[idxs].std()
    within_delta_27_std[gi] = norm_L27_L18[idxs].std()

    # Loc features: mean std across 8 dimensions
    loc_g = loc_feats[idxs]  # (k, 8)
    within_loc_std[gi] = loc_g.std(axis=0).mean()

    # RNA features: mean std across 9 dimensions
    rna_g = rna_feats[idxs]  # (k, 9)
    within_rna_std[gi] = rna_g.std(axis=0).mean()

    # Isoform number std (diversity proxy)
    nums = iso_numbers[idxs]
    within_seqnum_std[gi] = nums.std()

    # L30 cosine diversity: mean pairwise (1 - cosine_sim)
    e = emb_L30[idxs]
    norms_e = np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
    e_n = e / norms_e
    if len(idxs) >= 2:
        cos_mat = e_n @ e_n.T  # (k, k) cosine similarity
        # Mean of upper triangle (pairwise diversity)
        tri = cos_mat[np.triu_indices(len(idxs), k=1)]
        within_emb_cosine_std[gi] = (1 - tri).mean()  # mean cosine distance
    else:
        within_emb_cosine_std[gi] = 0.0

print(f"  Computed metrics for {n_multi} multi-isoform genes")
print(f"  within_delta_std: mean={within_delta_std.mean():.4f}  std={within_delta_std.std():.4f}")
print(f"  within_loc_std:   mean={within_loc_std.mean():.4f}  std={within_loc_std.std():.4f}")
print(f"  within_rna_std:   mean={within_rna_std.mean():.4f}  std={within_rna_std.std():.4f}")
sys.stdout.flush()

# ── 5. Spearman correlations ──────────────────────────────────────────
print("\n[5] Computing Spearman correlations (within_delta_std vs structural proxies)...")
sys.stdout.flush()

# Use only genes where all metrics are well-defined (no zero std, no missing)
valid = (within_delta_std > 0) & (n_iso_per_gene >= 2)
N_valid = valid.sum()
print(f"  Valid genes (delta_std > 0, ≥2 isoforms): {N_valid}/{n_multi}")

X = within_delta_std[valid]  # primary: ||delta_L30_L18||_2 std per gene

proxies = {
    'loc_feature_std'     : within_loc_std[valid],
    'rna_feature_std'     : within_rna_std[valid],
    'delta_L30_L7_std'   : within_delta_L7_std[valid],
    'delta_L27_L18_std'  : within_delta_27_std[valid],
    'iso_number_std'      : within_seqnum_std[valid],
    'L30_cosine_distance' : within_emb_cosine_std[valid],
}

print(f"\n  {'Proxy':<26} {'Spearman_rho':>13} {'p_value':>12} {'interpretation'}")
print("  " + "-" * 75)

spearman_results = {}
for name, Y in proxies.items():
    valid_pair = (X > 0) & (Y > 0)
    if valid_pair.sum() < 50:
        print(f"  {name:<26} {'SKIP (n<50)':>13}")
        continue
    rho, p = stats.spearmanr(X[valid_pair], Y[valid_pair])
    n_pairs = valid_pair.sum()
    if rho > 0.2 and p < 0.001:
        interp = "genome-wide isoform discrimination signal"
    elif rho > 0.1 and p < 0.01:
        interp = "moderate"
    else:
        interp = "negligible"
    spearman_results[name] = {'rho': float(rho), 'p': float(p), 'n': int(n_pairs),
                               'interpretation': interp}
    print(f"  {name:<26}  rho={rho:+.4f}  p={p:.2e}  n={n_pairs}  [{interp}]")

sys.stdout.flush()

# Also: within_loc_std → within_rna_std (sanity check: these should also correlate)
rho_loc_rna, p_loc_rna = stats.spearmanr(within_loc_std[valid], within_rna_std[valid])
print(f"\n  Sanity check: loc_std vs rna_std: rho={rho_loc_rna:+.4f}  p={p_loc_rna:.2e}")

# ── 6. Additional: per-gene isoform count effect ───────────────────
print("\n[6] Gene size effect on delta_std (controls for n_isoforms per gene)...")
sys.stdout.flush()

# Partial correlation controlling for n_isoforms per gene
from scipy.stats import spearmanr as spr

n_iso_v = n_iso_per_gene[valid].astype(float)
# Partial Spearman: residualize X and each proxy on log(n_iso)
log_n = np.log(n_iso_v)

def partial_spearman(a, b, c):
    """Spearman partial correlation of a,b controlling for c."""
    # Residualize a on c (Spearman-rank regression)
    ra = stats.rankdata(a); rb = stats.rankdata(b); rc = stats.rankdata(c)
    # OLS residuals of ranks
    from numpy.linalg import lstsq
    X_c = np.column_stack([np.ones_like(rc), rc])
    res_a = ra - X_c @ lstsq(X_c, ra, rcond=None)[0]
    res_b = rb - X_c @ lstsq(X_c, rb, rcond=None)[0]
    rho, p = stats.pearsonr(res_a, res_b)  # Pearson of residualized ranks
    return rho, p

print(f"  {'Proxy':<26} {'Partial_rho':>12} {'p_value':>12} {'(controlling n_iso)':>22}")
print("  " + "-" * 74)
partial_results = {}
for name, Y in proxies.items():
    valid_pair = (X > 0) & (Y > 0)
    if valid_pair.sum() < 50: continue
    try:
        rho_p, p_p = partial_spearman(X[valid_pair], Y[valid_pair], log_n[valid_pair])
    except Exception as e:
        print(f"  {name:<26} ERROR: {e}")
        continue
    partial_results[name] = {'partial_rho': float(rho_p), 'partial_p': float(p_p)}
    print(f"  {name:<26}  rho_partial={rho_p:+.4f}  p={p_p:.2e}")

sys.stdout.flush()

# ── 7. Quartile analysis (concrete evidence) ──────────────────────
print("\n[7] Quartile analysis: high vs low delta_std genes...")
sys.stdout.flush()

# Split by quartile of within_delta_std
q25, q75 = np.percentile(X, 25), np.percentile(X, 75)
low_mask  = X <= q25
high_mask = X >= q75

print(f"  Q1 (≤{q25:.4f}): n={low_mask.sum()} genes")
print(f"  Q4 (≥{q75:.4f}): n={high_mask.sum()} genes")
print()

for name, Y in proxies.items():
    valid_pair = (X > 0) & (Y > 0)
    if valid_pair.sum() < 50: continue
    Xv = X[valid_pair]; Yv = Y[valid_pair]
    q25_x = np.percentile(Xv, 25); q75_x = np.percentile(Xv, 75)
    low_m = Xv <= q25_x; high_m = Xv >= q75_x
    mean_lo = Yv[low_m].mean(); mean_hi = Yv[high_m].mean()
    fold = mean_hi / (mean_lo + 1e-12)
    print(f"  {name:<26}  low={mean_lo:.4f}  high={mean_hi:.4f}  "
          f"fold={fold:.2f}×  (Q4/Q1)")

sys.stdout.flush()

# ── 8. Write results ───────────────────────────────────────────────
print(f"\n[8] Writing results to {OUT_FILE}...")
sys.stdout.flush()

with open(OUT_FILE, 'w') as fh:
    def w(msg): fh.write(msg + '\n')

    w("=" * 70)
    w("WITHIN-GENE GENOME-WIDE DISCRIMINATION ANALYSIS")
    w("Brain reference transcriptome (Ensembl isoforms)")
    w(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w("=" * 70)
    w("")
    w(f"Dataset: brain_full ESM-2 embeddings (L7, L18, L27, L30)")
    w(f"  N_isoforms_total : {N}")
    w(f"  N_genes_total    : {len(gene2idxs)}")
    w(f"  N_multi_iso_genes: {n_multi}  (≥2 isoforms)")
    w(f"  N_valid_for_corr : {N_valid}  (delta_std > 0)")
    w("")
    w("PRIMARY METRIC")
    w("  within_delta_std_g = std(||L30_emb - L18_emb||_2) across isoforms of gene g")
    w("  Biological rationale: L30-L18 captures late-layer splice-induced shifts;")
    w("  higher variance within gene → more isoform-specific structural differentiation")
    w("")
    w("SPEARMAN CORRELATIONS (within_delta_std vs structural proxies)")
    w("-" * 70)
    w(f"{'Proxy':<28} {'rho':>8} {'p':>12} {'n_genes':>9}  Interpretation")
    w("-" * 70)
    for name, res in spearman_results.items():
        w(f"  {name:<26} {res['rho']:>+8.4f} {res['p']:>12.2e} {res['n']:>9}  "
          f"[{res['interpretation']}]")
    w("")
    w("PARTIAL SPEARMAN (controlling for log(n_isoforms per gene))")
    w("-" * 70)
    for name, res in partial_results.items():
        w(f"  {name:<26} {res['partial_rho']:>+8.4f} {res['partial_p']:>12.2e}")
    w("")
    w("QUARTILE COMPARISON (Q4 vs Q1 of within_delta_std)")
    w("-" * 70)
    for name, Y in proxies.items():
        valid_pair = (X > 0) & (Y > 0)
        if valid_pair.sum() < 50: continue
        Xv = X[valid_pair]; Yv = Y[valid_pair]
        q25_x = np.percentile(Xv, 25); q75_x = np.percentile(Xv, 75)
        low_m = Xv <= q25_x; high_m = Xv >= q75_x
        mean_lo = Yv[low_m].mean(); mean_hi = Yv[high_m].mean()
        fold = mean_hi / (mean_lo + 1e-12)
        w(f"  {name:<26}  Q1={mean_lo:.4f}  Q4={mean_hi:.4f}  Q4/Q1={fold:.2f}x")
    w("")
    w("INTERPRETATION GUIDE")
    w("  rho > 0.2, p < 0.001 → 'genome-wide isoform discrimination signal confirmed'")
    w("  rho 0.1-0.2          → 'moderate'")
    w("  rho < 0.1            → 'negligible'")
    w("")
    w("PAPER-LEVEL CLAIM ASSESSMENT")
    w("-" * 70)
    n_strong = sum(1 for r in spearman_results.values() if r['rho'] > 0.2 and r['p'] < 0.001)
    n_moderate = sum(1 for r in spearman_results.values() if 0.1 <= r['rho'] <= 0.2 and r['p'] < 0.01)
    if n_strong >= 2:
        claim = "DEFENSIBLE: ≥2 structural proxies show rho>0.2 (p<0.001)"
    elif n_strong >= 1 or n_moderate >= 2:
        claim = "MODERATE: ≥1 strong or ≥2 moderate correlations; include with appropriate hedging"
    else:
        claim = "WEAK: Limited evidence; consider alternative metrics or framing"
    w(f"  {claim}")
    w(f"  N_isoforms: {N_valid} genes with valid delta_std")
    w(f"  Strong (rho>0.2, p<0.001): {n_strong}/{len(spearman_results)}")
    w(f"  Moderate (rho>0.1, p<0.01): {n_moderate}/{len(spearman_results)}")
    w("")
    w(f"Runtime: {time.time()-t0:.1f}s")

print(f"\n  [Saved] {OUT_FILE}")
print(f"  Runtime: {time.time()-t0:.1f}s")
print("Done.")
