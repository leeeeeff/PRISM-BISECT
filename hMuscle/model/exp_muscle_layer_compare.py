"""Quick comparison: muscle L1->L30 vs L7->L30 start-layer effect."""
import numpy as np, time
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA = "../data"; SEED = 42
t0 = time.time()

te_gene = np.load("my_gene_list_fixed.npy", allow_pickle=True)
e2s = {}
with open(f"{DATA}/raw_data/data/id_lists/ensembl_to_symbol.txt") as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            e2s[p[0]] = p[4]
te_sym = [e2s.get(str(g.decode() if isinstance(g,bytes) else g).split(".")[0],
                  str(g.decode() if isinstance(g,bytes) else g).split(".")[0])
          for g in te_gene]

ANNOT = f"{DATA}/raw_data/data/annotations/human_annotations_unified_bp.txt"
all_pos = set()
with open(ANNOT) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) > 1:
            all_pos.add(p[0])

pos_idx  = sorted({i for i, s in enumerate(te_sym) if s in all_pos})
neg_pool = [i for i, s in enumerate(te_sym) if s not in all_pos]
rng = np.random.default_rng(SEED)
n_neg = max(1, min(len(pos_idx), len(neg_pool), 15000-len(pos_idx)))
neg_idx = rng.choice(neg_pool, size=n_neg, replace=False).tolist()
subset_idx = np.array(sorted(pos_idx + list(neg_idx)))
N = len(subset_idx)
sub_sym = [te_sym[i] for i in subset_idx]
print(f"N={N}  [{time.time()-t0:.1f}s]")

def lload(L):
    a = np.load(f"{DATA}/esm2_layer_{L:02d}_t30_150M.npy", mmap_mode="r")
    return np.asarray(a[subset_idx], dtype=np.float32)

L1  = lload(1)
L7  = lload(7)
L30 = lload(30)
l1_orig = np.linalg.norm(L1.astype(np.float64), axis=1)
print(f"Layers loaded  [{time.time()-t0:.1f}s]")

gene2 = defaultdict(list)
for li, sym in enumerate(sub_sym):
    gene2[sym].append(li)
multi = {g: v for g, v in gene2.items() if len(v) >= 2}
print(f"Multi-isoform genes: {len(multi)}")

def run(sa, ea, lab):
    def zn(a): return (a - a.mean(0)) / (a.std(0) + 1e-6)
    flat = np.concatenate([zn(sa), zn(ea)], axis=0)
    pca  = PCA(3, random_state=SEED, svd_solver="randomized")
    c    = pca.fit_transform(flat)
    cs, ce = c[:N], c[N:]
    s, e, d = [], [], []
    for g, isos in multi.items():
        for ii in range(len(isos)):
            for jj in range(ii+1, len(isos)):
                a, b = isos[ii], isos[jj]
                s.append(np.linalg.norm(cs[a] - cs[b]))
                e.append(np.linalg.norm(ce[a] - ce[b]))
                d.append(abs(l1_orig[a] - l1_orig[b]))
    s = np.array(s); e = np.array(e); d = np.array(d)
    rs, _ = spearmanr(d, s)
    re, _ = spearmanr(d, e)
    ratio  = e.mean() / s.mean()
    dr     = re - rs
    print(f"  [{lab:12s}]  ratio={ratio:.3f}  rho_s={rs:.3f}  rho_e={re:.3f}  delta_rho={dr:+.3f}")
    return ratio, dr

print("\nMuscle:")
r1, dr1 = run(L1,  L30, "L1 -> L30")
r7, dr7 = run(L7,  L30, "L7 -> L30")

print(f"\n{'='*58}")
print(f"  {'':20s} {'ratio':>7} {'delta_rho':>11}")
print(f"  {'-'*38}")
print(f"  {'Muscle L1->L30':20s} {r1:>7.3f} {dr1:>+11.3f}")
print(f"  {'Muscle L7->L30':20s} {r7:>7.3f} {dr7:>+11.3f}")
print(f"  {'Brain  L7->L30':20s} {'1.865':>7} {'-0.053':>11}  (reference)")
print(f"{'='*58}")

if abs(dr7 - (-0.053)) < abs(dr1 - (-0.053)):
    print("\n  → Muscle L7->L30 closer to Brain: Δρ gap is START-LAYER effect")
else:
    print("\n  → Gap persists even at L7: DATASET-SPECIFIC difference (brain ≠ muscle biology)")
