#!/usr/bin/env python3
"""
exp_within_gene_length_matched.py
=================================
S4 rigor gate for exp_within_gene_layer_divergence.py.

The domain-diff/domain-same ratio (~3-4x, peak L17) could be a pure LENGTH
artifact: domain gain/loss changes sequence length, and within-pair length
difference correlates with trajectory divergence (rho=+0.176).

DISCRIMINATING TEST (predict-before-you-look):
  Stratify pairs into |length-diff| bins. Within each length-matched bin,
  recompute the domain-diff / domain-same cosine-distance ratio at the peak
  layer (L17) and at L1/L30.
  H_domain: ratio stays > 1 within length-matched bins  -> domain content adds
            trajectory signal beyond length (size).
  H_length: ratio -> ~1 within length-matched bins       -> divergence is purely
            a size/length signal (domain-diff just has bigger length diffs).
"""
import os, json
import numpy as np
from collections import defaultdict
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "8")

DATA_DIR = "../data"
OUT = "../../reports/within_gene_layer_divergence"
rng = np.random.default_rng(42)

genes = np.array([str(g) for g in np.load("my_gene_list_fixed.npy", allow_pickle=True)])
DM = (np.load("../results_isoform/features/domain_matrix_proper_test.npy") > 0).astype(np.int8)
lengths = (np.load("my_sequence_matrix_fixed.npy") != 0).sum(1).astype(np.int32)
N = len(genes)

gene2idx = defaultdict(list)
for i, g in enumerate(genes):
    gene2idx[g].append(i)

A, Bp, is_diff, len_diff = [], [], [], []
for g, ix in gene2idx.items():
    if len(ix) < 2:
        continue
    ix = np.array(ix)
    pairs = [(a, b) for k, a in enumerate(ix) for b in ix[k + 1:]]
    if len(pairs) > 60:
        pairs = [pairs[s] for s in rng.choice(len(pairs), 60, replace=False)]
    for a, b in pairs:
        h = int(np.abs(DM[a] - DM[b]).sum())
        A.append(a); Bp.append(b); is_diff.append(1 if h > 0 else 0)
        len_diff.append(abs(int(lengths[a] - lengths[b])))
A = np.array(A); Bp = np.array(Bp)
is_diff = np.array(is_diff); len_diff = np.array(len_diff)
both_dom = (DM[A].sum(1) > 0) & (DM[Bp].sum(1) > 0)
diff_mask = is_diff == 1
same_mask = (is_diff == 0) & both_dom

def cos_dist(E, a, b):
    ea, eb = E[a], E[b]
    return 1.0 - (ea * eb).sum(1) / (
        (np.linalg.norm(ea, axis=1) + 1e-9) * (np.linalg.norm(eb, axis=1) + 1e-9))

d = {}
for L in (1, 17, 30):
    E = np.load(f"{DATA_DIR}/esm2_layer_{L:02d}_t30_150M.npy")
    d[L] = cos_dist(E, A, Bp).astype(np.float32)
    del E

# length-diff bins (quantile edges from the pooled diff+same set)
pool = diff_mask | same_mask
edges = np.quantile(len_diff[pool], [0, .2, .4, .6, .8, 1.0])
edges[-1] += 1
print("=" * 72)
print("  Length-matched stratification (S4 gate)")
print("  H_domain: ratio>1 persists within length bins | H_length: ratio->1")
print("=" * 72)
print(f"\n  |len-diff| quantile edges: {[int(e) for e in edges]}")
print(f"\n  {'len-diff bin':>16} {'n_diff':>7} {'n_same':>7}"
      f" {'ratio_L1':>9} {'ratio_L17':>10} {'ratio_L30':>10}")

rows = []
for k in range(len(edges) - 1):
    lo, hi = edges[k], edges[k + 1]
    binm = (len_diff >= lo) & (len_diff < hi)
    dm = binm & diff_mask
    sm = binm & same_mask
    if dm.sum() < 30 or sm.sum() < 30:
        print(f"  [{int(lo):>5},{int(hi):>5}) too few (d={dm.sum()},s={sm.sum()})")
        continue
    r = {}
    for L in (1, 17, 30):
        r[L] = float(d[L][dm].mean() / d[L][sm].mean())
    rows.append({"bin": [int(lo), int(hi)], "n_diff": int(dm.sum()),
                 "n_same": int(sm.sum()), "ratio_L1": r[1],
                 "ratio_L17": r[17], "ratio_L30": r[30]})
    print(f"  [{int(lo):>5},{int(hi):>5}) {dm.sum():>7} {sm.sum():>7}"
          f" {r[1]:>9.3f} {r[17]:>10.3f} {r[30]:>10.3f}")

# also: same-length-ONLY subset (|len diff| == 0) — the cleanest test
zerolen = len_diff == 0
dz, sz = zerolen & diff_mask, zerolen & same_mask
print(f"\n  EXACT same-length pairs: domain-diff n={dz.sum()}, domain-same n={sz.sum()}")
if dz.sum() >= 30 and sz.sum() >= 30:
    for L in (1, 17, 30):
        print(f"    L{L}: ratio = {d[L][dz].mean()/d[L][sz].mean():.3f}")

with open(f"{OUT}/length_matched.json", "w") as f:
    json.dump({"len_edges": [int(e) for e in edges], "bins": rows,
               "same_length_n_diff": int(dz.sum()),
               "same_length_n_same": int(sz.sum()),
               "same_length_ratio_L17": (
                   float(d[17][dz].mean() / d[17][sz].mean())
                   if dz.sum() >= 30 and sz.sum() >= 30 else None)}, f, indent=1)
print(f"\nSaved: {OUT}/length_matched.json")
