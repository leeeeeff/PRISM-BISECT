#!/usr/bin/env python3
"""
exp_within_gene_layer_divergence.py
====================================
DATASET-WIDE, WITHIN-GENE trajectory divergence — a gene-family-confound-free
replacement for the cross-gene "convergent evolution" statistic (which was a
ribosomal gene-family clustering artifact).

QUESTION (S0 reframe):
  Do isoforms of the SAME gene that differ in Pfam domain architecture diverge
  MORE across the ESM-2 layer trajectory than isoforms that share architecture,
  and is that divergence concentrated at mid-layers (L15-20) — matching the
  per-layer Fisher peak distribution and the delta_layer = L30 - L15 design?

  Within-gene => gene identity is constant for both members of every pair =>
  gene-family confound is structurally removed (same discipline as Domain-Ranking).

LOCKED PREDICTION (predict-before-you-look; Platt/HARKing guard):
  H1: ratio_L = mean(dist | domain-different) / mean(dist | domain-same) is > 1
      at all layers, and PEAKS at mid-layers (L15-20); relatively lower at L1 and L30.
  H0 (null): shuffling the domain-diff/same pair labels yields ratio_L ~ 1.0, FLAT.
  Confound: divergence could be driven by sequence-length difference, not domain
      content -> control by partial Spearman at the peak layer.

Metric is scale-invariant by construction: the ratio at each layer normalizes out
per-layer embedding-norm growth (numerator and denominator are at the same layer).

Data (muscle held-out test set, 36,748 isoforms — SAME set as Domain-Ranking 0.630
and Type 0-3 stratification, so results are directly comparable to Results 4b):
  - per-isoform gene:    my_gene_list_fixed.npy               (36748,)
  - Pfam domain matrix:  domain_matrix_proper_test.npy        (36748, 512) binary
  - per-layer ESM-2:     esm2_layer_{01..30}_t30_150M.npy     (36748, 640) each
  - length proxy:        my_sequence_matrix_fixed.npy         (36748, 6000) -> nonzero count
"""
import os, json, time
import numpy as np
from collections import defaultdict
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

DATA_DIR = "../data"
OUT_DIR  = "../../reports/within_gene_layer_divergence"
os.makedirs(OUT_DIR, exist_ok=True)
N_LAYERS = 30
SEED     = 42
B_NULL   = 1000
rng      = np.random.default_rng(SEED)

print("=" * 72)
print("  Within-gene per-layer trajectory divergence (domain-diff vs domain-same)")
print("  LOCKED PREDICTION: ratio>1 all layers, PEAK at mid (L15-20), null flat ~1.0")
print("=" * 72)

# ── 1. Load alignment ─────────────────────────────────────────────────
genes = np.load("my_gene_list_fixed.npy", allow_pickle=True)
genes = np.array([str(g) for g in genes])
DM = np.load("../results_isoform/features/domain_matrix_proper_test.npy")
DM = (DM > 0).astype(np.int8)                      # binarize
seqmat = np.load("my_sequence_matrix_fixed.npy")   # (36748, 6000)
lengths = (seqmat != 0).sum(1).astype(np.int32)
del seqmat
N = len(genes)
assert DM.shape[0] == N == len(lengths), (DM.shape, N, len(lengths))
print(f"[1] N={N} isoforms, domain dim={DM.shape[1]}, "
      f"domain-bearing isoforms={(DM.sum(1)>0).sum()}")

# ── 2. Build within-gene pairs ────────────────────────────────────────
gene2idx = defaultdict(list)
for i, g in enumerate(genes):
    gene2idx[g].append(i)
multi = {g: ix for g, ix in gene2idx.items() if len(ix) >= 2}

A, Bp, is_diff, len_diff, ham = [], [], [], [], []
MAX_PAIRS_PER_GENE = 60   # cap combinatorial blow-up for high-isoform genes
for g, ix in multi.items():
    ix = np.array(ix)
    pairs = [(a, b) for k, a in enumerate(ix) for b in ix[k + 1:]]
    if len(pairs) > MAX_PAIRS_PER_GENE:
        sel = rng.choice(len(pairs), MAX_PAIRS_PER_GENE, replace=False)
        pairs = [pairs[s] for s in sel]
    for a, b in pairs:
        h = int(np.abs(DM[a] - DM[b]).sum())        # Hamming (domain set diff)
        A.append(a); Bp.append(b)
        is_diff.append(1 if h > 0 else 0)
        ham.append(h)
        len_diff.append(abs(int(lengths[a] - lengths[b])))
A = np.array(A); Bp = np.array(Bp)
is_diff = np.array(is_diff); ham = np.array(ham); len_diff = np.array(len_diff)
# domain-same restricted to BOTH having >=1 real domain (matched, informative)
both_dom = (DM[A].sum(1) > 0) & (DM[Bp].sum(1) > 0)
same_mask = (is_diff == 0) & both_dom
diff_mask = (is_diff == 1)
print(f"[2] within-gene pairs: total={len(A):,}  "
      f"domain-diff={diff_mask.sum():,}  domain-same(both>=1 dom)={same_mask.sum():,}")

# ── 3. Per-layer distances ────────────────────────────────────────────
def cos_dist(E, a, b):
    ea = E[a]; eb = E[b]
    na = np.linalg.norm(ea, axis=1) + 1e-9
    nb = np.linalg.norm(eb, axis=1) + 1e-9
    return 1.0 - (ea * eb).sum(1) / (na * nb)

dist = np.zeros((len(A), N_LAYERS), dtype=np.float32)
t0 = time.time()
for L in range(1, N_LAYERS + 1):
    E = np.load(f"{DATA_DIR}/esm2_layer_{L:02d}_t30_150M.npy")
    dist[:, L - 1] = cos_dist(E, A, Bp)
    del E
    if L % 6 == 0:
        print(f"    layer {L:02d}/30  ({time.time()-t0:.0f}s)")
print(f"[3] distances computed ({time.time()-t0:.0f}s)")

# ── 4. Layer-wise ratio + shuffle null ────────────────────────────────
d_diff = dist[diff_mask]
d_same = dist[same_mask]
ratio = d_diff.mean(0) / d_same.mean(0)            # (30,)

# shuffle null: permute diff/same labels within the pooled (diff+same) set
pool_idx = np.where(diff_mask | same_mask)[0]
pool_dist = dist[pool_idx]
n_diff = int(diff_mask.sum())
null_ratio = np.zeros((B_NULL, N_LAYERS), dtype=np.float32)
for b in range(B_NULL):
    perm = rng.permutation(len(pool_idx))
    dd = pool_dist[perm[:n_diff]].mean(0)
    ss = pool_dist[perm[n_diff:]].mean(0)
    null_ratio[b] = dd / ss
null_lo = np.percentile(null_ratio, 2.5, axis=0)
null_hi = np.percentile(null_ratio, 97.5, axis=0)
# empirical p per layer (two-sided): fraction of null as extreme as observed
p_layer = np.array([
    (np.abs(null_ratio[:, L] - 1.0) >= abs(ratio[L] - 1.0)).mean()
    for L in range(N_LAYERS)])

peak_L = int(np.argmax(ratio)) + 1
print(f"\n[4] LAYER-WISE RATIO (domain-diff / domain-same cosine distance):")
print(f"    {'L':>3} {'ratio':>7} {'null95CI':>18} {'p':>8}")
for L in range(N_LAYERS):
    star = " <-- PEAK" if L + 1 == peak_L else ""
    print(f"    {L+1:>3} {ratio[L]:7.3f}  [{null_lo[L]:.3f},{null_hi[L]:.3f}]"
          f"  {p_layer[L]:8.4g}{star}")
print(f"\n    PEAK layer = L{peak_L}  ratio={ratio[peak_L-1]:.3f}  "
      f"(L1={ratio[0]:.3f}, L30={ratio[29]:.3f})")

# ── 5. Length confound control at peak layer ──────────────────────────
# among domain-diff pairs: does mid-layer divergence track domain-set diff (Hamming)
# after controlling for length difference?
dl = len_diff[diff_mask].astype(float)
hm = ham[diff_mask].astype(float)
dp = d_diff[:, peak_L - 1].astype(float)
rho_raw, p_raw = stats.spearmanr(hm, dp)
# partial Spearman(dp, hm | dl): residualize both on rank(dl)
def resid_on(x, ctrl):
    rx = stats.rankdata(x); rc = stats.rankdata(ctrl)
    beta = np.polyfit(rc, rx, 1)
    return rx - (beta[0] * rc + beta[1])
r_dp = resid_on(dp, dl); r_hm = resid_on(hm, dl)
rho_par, p_par = stats.spearmanr(r_dp, r_hm)
rho_len, p_len = stats.spearmanr(dl, dp)
print(f"\n[5] LENGTH CONFOUND CONTROL at L{peak_L} (domain-diff pairs, n={len(dp):,}):")
print(f"    Spearman(divergence, |domain Hamming|)           = {rho_raw:+.4f} (p={p_raw:.2g})")
print(f"    Spearman(divergence, |length diff|)              = {rho_len:+.4f} (p={p_len:.2g})")
print(f"    PARTIAL Spearman(divergence, Hamming | len diff) = {rho_par:+.4f} (p={p_par:.2g})")

# ── 6. Relative-divergence trajectory (normalize each pair by its L1 dist) ──
rel_diff = (d_diff / (d_diff[:, [0]] + 1e-9)).mean(0)
rel_same = (d_same / (d_same[:, [0]] + 1e-9)).mean(0)
print(f"\n[6] Relative divergence vs L1 at L{peak_L}: "
      f"domain-diff={rel_diff[peak_L-1]:.3f}x, domain-same={rel_same[peak_L-1]:.3f}x")

# ── 7. Save ───────────────────────────────────────────────────────────
out = {
    "n_pairs_total": int(len(A)),
    "n_domain_diff": int(diff_mask.sum()),
    "n_domain_same_both": int(same_mask.sum()),
    "ratio_by_layer": [float(x) for x in ratio],
    "null_lo": [float(x) for x in null_lo],
    "null_hi": [float(x) for x in null_hi],
    "p_by_layer": [float(x) for x in p_layer],
    "peak_layer": peak_L,
    "ratio_peak": float(ratio[peak_L - 1]),
    "ratio_L1": float(ratio[0]),
    "ratio_L30": float(ratio[29]),
    "length_control": {
        "spearman_divergence_hamming": [float(rho_raw), float(p_raw)],
        "spearman_divergence_lendiff": [float(rho_len), float(p_len)],
        "partial_spearman_hamming_given_len": [float(rho_par), float(p_par)],
        "peak_layer": peak_L,
    },
    "rel_divergence_diff_by_layer": [float(x) for x in rel_diff],
    "rel_divergence_same_by_layer": [float(x) for x in rel_same],
    "B_null": B_NULL, "seed": SEED,
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"\nSaved: {OUT_DIR}/results.json")
