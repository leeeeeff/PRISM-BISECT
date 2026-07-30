"""
exp_fluid_stage1i_within_gene_gap.py
=====================================
Within-gene isoform separation: does fluid trajectory encode
per-isoform functional divergence *within the same gene* better
than L30 mean-pool alone?

Method
------
1.  Load narrow-GO pilot subset (N ~ 8000), extract 30-layer traj +
    L30 mean-pool for each isoform.
2.  Fit joint PCA (K=8) so curve_vec is 240-D.
3.  Also fit L30 PCA (240-D) for matched-dim comparison.
4.  Enumerate all same-gene isoform pairs (i, j).
5.  For each pair:
      d_fluid  = || curve_vec_240 ||
      d_L30    = || L30_pca_240 ||    (matched dim)
      d_L30raw = || L30_raw_640 ||
6.  Also load PRISM cached predictions (score_matrix_18go) if a pair's
    two isoforms fall in the test set --> compute max GO-score gap.
7.  Statistical tests:
      - median d_fluid vs d_L30 (paired Wilcoxon)
      - Spearman correlation of d_fluid with PRISM max GO gap
        vs Spearman correlation of d_L30 with PRISM max GO gap
      - top-decile PRISM-gap pairs: fluid rank vs L30 rank
"""

import os, json, time, gc, glob
import numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon, spearmanr
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = "../data"
ANNOT_FILE = "../data/raw_data/data/annotations/human_annotations_unified_bp.txt"
ID_DIR     = "../data/raw_data/data/id_lists"
OUT_DIR    = "../../reports/fluid_stage1"
PRISM_SCORE = "../../reports/v15_bp_clean/score_matrix_18go_20260519_1914.npy"

N_LAYERS = 30
EMB_DIM  = 640
SEED     = 42
K_PCA    = 8

NARROW_GO = [
    "GO:0006096", "GO:0006099", "GO:0006120", "GO:0006754",
    "GO:0006635", "GO:0006418", "GO:0045214", "GO:0030048",
    "GO:0007018", "GO:0007015", "GO:0007204", "GO:0006816",
    "GO:0006888", "GO:0000398", "GO:0006414",
]


def load_ids(p):
    a = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def load_e2s():
    m = {}
    with open(f"{ID_DIR}/ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                m[p[0]] = p[4]
    return m


def load_go_pos(go):
    pos = set()
    with open(ANNOT_FILE) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and go in parts[1:]:
                pos.add(parts[0])
    return pos


def build_trajectory(idx_subset):
    N = len(idx_subset)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        p = f"{DATA_DIR}/esm2_layer_{L:02d}_t30_150M.npy"
        arr = np.load(p, mmap_mode="r")
        traj[:, L - 1, :] = np.asarray(arr[idx_subset], dtype=np.float32)
        del arr
    return traj


def main():
    t0 = time.time()
    te_iso  = load_ids("my_isoform_list_fixed.npy")
    te_gene = load_ids("my_gene_list_fixed.npy")
    e2s = load_e2s()
    te_sym = [e2s.get(g.split(".")[0], g.split(".")[0]) for g in te_gene]
    print(f"[{time.time()-t0:5.1f}s] N_TE={len(te_iso)}")

    go_pos = {go: load_go_pos(go) for go in NARROW_GO}
    all_pos = set().union(*go_pos.values())
    pos_idx = sorted({i for i, s in enumerate(te_sym) if s in all_pos})
    neg_pool = [i for i, s in enumerate(te_sym) if s not in all_pos]
    n_neg = min(len(pos_idx), len(neg_pool))
    rng = np.random.default_rng(SEED)
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False).tolist()
    subset_idx = np.array(sorted(pos_idx + list(neg_idx)))
    N = len(subset_idx)
    sub_gene = [te_gene[i] for i in subset_idx]
    sub_sym  = [te_sym[i]  for i in subset_idx]
    print(f"[{time.time()-t0:5.1f}s] pilot N={N}")

    # same-gene pair index within subset
    sym_to_idx = defaultdict(list)
    for i, s in enumerate(sub_sym):
        sym_to_idx[s].append(i)
    pairs = []
    for s, idxs in sym_to_idx.items():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pairs.append((idxs[a], idxs[b], s))
    print(f"[{time.time()-t0:5.1f}s] same-gene pairs: {len(pairs)}")

    # build features
    print(f"[{time.time()-t0:5.1f}s] building trajectory ...")
    traj = build_trajectory(subset_idx)
    L30 = traj[:, -1, :].copy()

    # fluid PCA
    flat = traj.reshape(N * N_LAYERS, EMB_DIM)
    pca_fluid = PCA(n_components=K_PCA, random_state=SEED,
                    svd_solver="randomized")
    flat_red = pca_fluid.fit_transform(flat)
    curve_vec = flat_red.reshape(N, N_LAYERS * K_PCA)   # (N, 240)
    del flat, flat_red, traj
    gc.collect()

    # L30 PCA to matched dim
    pca_l30 = PCA(n_components=N_LAYERS * K_PCA,
                  random_state=SEED, svd_solver="randomized")
    L30_pca = pca_l30.fit_transform(L30)                # (N, 240)

    # PRISM score matrix (36748 x 18 in full test set index)
    prism_scores = None
    if os.path.exists(PRISM_SCORE):
        prism_scores = np.load(PRISM_SCORE)  # (36748, 18)
        print(f"[{time.time()-t0:5.1f}s] PRISM score shape {prism_scores.shape}")

    # compute distances
    print(f"[{time.time()-t0:5.1f}s] computing pair distances ...")
    d_fluid  = np.empty(len(pairs), dtype=np.float64)
    d_l30_p  = np.empty(len(pairs), dtype=np.float64)
    d_l30_r  = np.empty(len(pairs), dtype=np.float64)
    prism_gap = np.full(len(pairs), np.nan, dtype=np.float64)
    for k, (a, b, s) in enumerate(pairs):
        d_fluid[k]  = np.linalg.norm(curve_vec[a] - curve_vec[b])
        d_l30_p[k]  = np.linalg.norm(L30_pca[a]   - L30_pca[b])
        d_l30_r[k]  = np.linalg.norm(L30[a]       - L30[b])
        if prism_scores is not None:
            # map subset index back to test set idx
            i_a = int(subset_idx[a])
            i_b = int(subset_idx[b])
            gap = np.abs(prism_scores[i_a] - prism_scores[i_b]).max()
            prism_gap[k] = gap

    # normalize distances to same scale (percentile rank) for fair comparison
    def pctrank(x):
        r = np.argsort(np.argsort(x))
        return r / (len(x) - 1)
    r_fluid  = pctrank(d_fluid)
    r_l30_p  = pctrank(d_l30_p)
    r_l30_r  = pctrank(d_l30_r)

    # stats
    med_fluid = float(np.median(d_fluid))
    med_l30p  = float(np.median(d_l30_p))
    med_l30r  = float(np.median(d_l30_r))
    print(f"  median d_fluid ={med_fluid:.3f}")
    print(f"  median d_L30_pc={med_l30p:.3f}")
    print(f"  median d_L30_rw={med_l30r:.3f}")

    # paired Wilcoxon on ranks (fluid vs L30_pca)
    W1, p_w1 = wilcoxon(r_fluid, r_l30_p, alternative="greater")
    W2, p_w2 = wilcoxon(r_fluid, r_l30_r, alternative="greater")
    print(f"  Wilcoxon r_fluid > r_L30_pca:  W={W1:.1f}  p={p_w1:.3e}")
    print(f"  Wilcoxon r_fluid > r_L30_raw:  W={W2:.1f}  p={p_w2:.3e}")

    # Spearman correlation with PRISM gap
    corr_report = None
    if prism_scores is not None:
        valid = ~np.isnan(prism_gap)
        if valid.sum() >= 30:
            sp_f, pf = spearmanr(d_fluid[valid], prism_gap[valid])
            sp_p, pp = spearmanr(d_l30_p[valid], prism_gap[valid])
            sp_r, pr = spearmanr(d_l30_r[valid], prism_gap[valid])
            print(f"  Spearman d_fluid ~ PRISM_gap:  rho={sp_f:+.3f}  p={pf:.2e}")
            print(f"  Spearman d_L30_pc ~ PRISM_gap: rho={sp_p:+.3f}  p={pp:.2e}")
            print(f"  Spearman d_L30_rw ~ PRISM_gap: rho={sp_r:+.3f}  p={pr:.2e}")
            corr_report = dict(fluid=[float(sp_f), float(pf)],
                               l30_pca=[float(sp_p), float(pp)],
                               l30_raw=[float(sp_r), float(pr)],
                               n_valid=int(valid.sum()))

    # top-decile PRISM-gap pairs: does fluid rank consistently exceed L30?
    top_decile_report = None
    if prism_scores is not None:
        thr = np.nanpercentile(prism_gap, 90)
        mask = prism_gap >= thr
        n_top = int(mask.sum())
        if n_top >= 10:
            top_fluid_med = float(np.median(r_fluid[mask]))
            top_l30_med   = float(np.median(r_l30_p[mask]))
            top_diff      = top_fluid_med - top_l30_med
            print(f"  top-decile PRISM-gap (n={n_top}):  "
                  f"med r_fluid={top_fluid_med:.3f}  "
                  f"med r_L30_pca={top_l30_med:.3f}  "
                  f"diff={top_diff:+.3f}")
            top_decile_report = dict(n_top=n_top,
                                     top_fluid_med=top_fluid_med,
                                     top_l30_pca_med=top_l30_med,
                                     top_diff=top_diff)

    # diagnostic figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].hist(d_fluid, bins=60, alpha=0.5, label="fluid 240-D")
    axs[0].hist(d_l30_p, bins=60, alpha=0.5, label="L30-PCA 240-D")
    axs[0].hist(d_l30_r, bins=60, alpha=0.5, label="L30 raw 640-D")
    axs[0].set_xlabel("same-gene pair distance")
    axs[0].legend(); axs[0].set_title("distance distributions")
    if prism_scores is not None:
        v = ~np.isnan(prism_gap)
        axs[1].scatter(prism_gap[v], d_fluid[v], s=4, alpha=0.4,
                       label="fluid")
        axs[1].set_xlabel("PRISM max-GO gap")
        axs[1].set_ylabel("d_fluid")
        axs[1].set_title("fluid dist vs PRISM gap")
        axs[2].scatter(prism_gap[v], d_l30_p[v], s=4, alpha=0.4,
                       label="L30-PCA", color="orange")
        axs[2].set_xlabel("PRISM max-GO gap")
        axs[2].set_ylabel("d_L30_pca")
        axs[2].set_title("L30-PCA dist vs PRISM gap")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/within_gene_gap.png", dpi=140)
    plt.close(fig)

    # save
    report = dict(
        pilot_N=int(N),
        n_pairs=len(pairs),
        median_distances=dict(fluid=med_fluid,
                              L30_pca=med_l30p,
                              L30_raw=med_l30r),
        wilcoxon=dict(fluid_vs_L30_pca_p=float(p_w1),
                      fluid_vs_L30_raw_p=float(p_w2)),
        prism_gap_corr=corr_report,
        prism_top_decile=top_decile_report,
    )
    with open(f"{OUT_DIR}/within_gene_gap.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved: {OUT_DIR}/within_gene_gap.json")
    print(f"saved: {OUT_DIR}/within_gene_gap.png")
    print(f"total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
