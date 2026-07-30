"""
exp_fluid_stage1_curve_cluster.py
==================================
Stage 1 pilot for the fluid-vector trajectory idea.

Question: If we treat each isoform's ESM-2 L1..L30 mean-pooled embeddings
as a 30-point trajectory in 640-D, do isoforms sharing the same GO term
form visible *bundles* (curve clusters) without any supervised training?

If bundles do exist and enrich a target GO term above baseline (p<0.01),
Stage 2 (continuous fluid via Flow Matching) is worth building.
If not, the whole framework is dead here.

Pipeline
--------
1.  Load 30 pooled-layer embeddings for the 36,748 test isoforms (mmap).
2.  Build labels for 3 pilot GO terms (gene-level annotation from
    human_annotations_unified_bp.txt).
3.  Subset: for each GO, positives + a size-matched random negative pool
    drawn from the same gene distribution (weakly matched, since labels
    are gene-level anyway).
4.  Joint PCA on all pilot (isoform x layer) points --> shared K=8 axes.
5.  Reduce each trajectory to (30, 8) and flatten to a 240-D curve vector.
6.  KMeans (k_clusters = 12) on curve vectors.
7.  For each cluster and each pilot GO:
      - GO+ fraction inside cluster vs global baseline
      - hypergeometric p-value
      - 1000x bootstrap CI on GO+ fraction
8.  Save: cluster assignments, purity table, 3D trajectory PNG per GO.

Decision gate
-------------
GO wins if at least one cluster has GO+ fraction >= 2x baseline
AND hypergeometric p < 0.01 after Bonferroni across k_clusters * n_go tests.
"""

import os, json, time, gc, sys
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import hypergeom
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = "../data"
ANNOT_FILE = "../data/raw_data/data/annotations/human_annotations_unified_bp.txt"
ID_DIR     = "../data/raw_data/data/id_lists"
OUT_DIR    = "../../reports/fluid_stage1"
os.makedirs(OUT_DIR, exist_ok=True)

N_LAYERS      = 30
EMB_DIM       = 640
K_PCA         = 8
K_CLUSTERS    = 12
N_BOOTSTRAP   = 1000
SEED          = 42

PILOT_GO = {
    "GO:0006941": "Muscle contraction",
    "GO:0007005": "Mitochondrion organization",
    "GO:0006096": "Glycolysis",
}

np.random.seed(SEED)


# ------------------------------------------------------------------ helpers
def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def load_ensg_to_symbol():
    m = {}
    with open(f"{ID_DIR}/ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                m[parts[0]] = parts[4]
    return m


def load_go_positive_symbols(go_term):
    pos = set()
    with open(ANNOT_FILE) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    return pos


def build_trajectory(idx_subset):
    """Return (len(idx_subset), N_LAYERS, EMB_DIM) float32 tensor."""
    N = len(idx_subset)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        p = f"{DATA_DIR}/esm2_layer_{L:02d}_t30_150M.npy"
        layer = np.load(p, mmap_mode="r")
        traj[:, L - 1, :] = np.asarray(layer[idx_subset], dtype=np.float32)
        del layer
    return traj


def hypergeom_p(k, N, K, n):
    """Prob >= k successes in draw of n, from pop N containing K positives."""
    return float(hypergeom.sf(k - 1, N, K, n))


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()

    # ---- 1. IDs and labels
    te_iso  = load_ids("my_isoform_list_fixed.npy")
    te_gene = load_ids("my_gene_list_fixed.npy")
    ensg2sym = load_ensg_to_symbol()
    te_sym = [ensg2sym.get(g.split(".")[0], g.split(".")[0]) for g in te_gene]

    N_TE = len(te_iso)
    print(f"[{time.time()-t0:5.1f}s] loaded {N_TE} test isoforms")

    go_pos_syms = {go: load_go_positive_symbols(go) for go in PILOT_GO}
    for go, name in PILOT_GO.items():
        n_iso_pos = sum(1 for s in te_sym if s in go_pos_syms[go])
        print(f"          {go} ({name:32s})  positives_in_test={n_iso_pos}")

    # ---- 2. Pilot subset: union of positives across 3 GOs + neg pool
    pos_union_idx = sorted({i for i, s in enumerate(te_sym)
                            if any(s in go_pos_syms[go] for go in PILOT_GO)})
    all_pos_syms = set().union(*go_pos_syms.values())
    neg_idx_pool = [i for i, s in enumerate(te_sym) if s not in all_pos_syms]
    n_neg = min(len(pos_union_idx), len(neg_idx_pool))
    rng = np.random.default_rng(SEED)
    neg_idx = rng.choice(neg_idx_pool, size=n_neg, replace=False).tolist()

    subset_idx = np.array(sorted(pos_union_idx + list(neg_idx)))
    N = len(subset_idx)
    print(f"[{time.time()-t0:5.1f}s] pilot subset N={N} "
          f"(pos_union={len(pos_union_idx)}, neg={n_neg})")

    subset_sym = [te_sym[i] for i in subset_idx]
    subset_iso = [te_iso[i] for i in subset_idx]

    # per-GO subset labels
    y = {}
    for go in PILOT_GO:
        y[go] = np.array([1 if s in go_pos_syms[go] else 0 for s in subset_sym],
                         dtype=np.int32)

    # ---- 3. Trajectory tensor
    print(f"[{time.time()-t0:5.1f}s] building trajectory tensor "
          f"({N} x {N_LAYERS} x {EMB_DIM}) ...")
    traj = build_trajectory(subset_idx)   # (N, 30, 640)
    print(f"[{time.time()-t0:5.1f}s] tensor built, "
          f"{traj.nbytes/1e9:.2f} GB")

    # ---- 4. Joint PCA across all (isoform x layer) points
    flat = traj.reshape(N * N_LAYERS, EMB_DIM)
    print(f"[{time.time()-t0:5.1f}s] joint PCA on {flat.shape[0]} pts -> {K_PCA} axes")
    pca = PCA(n_components=K_PCA, random_state=SEED, svd_solver="randomized")
    flat_red = pca.fit_transform(flat)              # (N*30, K_PCA)
    traj_red = flat_red.reshape(N, N_LAYERS, K_PCA)
    print(f"          explained_var_sum={pca.explained_variance_ratio_.sum():.3f}")
    del flat, flat_red
    gc.collect()

    # ---- 5. Cluster curves
    curve_vec = traj_red.reshape(N, N_LAYERS * K_PCA)   # (N, 240)
    print(f"[{time.time()-t0:5.1f}s] KMeans k={K_CLUSTERS} on curve vectors")
    km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=SEED)
    cid = km.fit_predict(curve_vec)

    # ---- 6. Purity + hypergeometric
    print(f"[{time.time()-t0:5.1f}s] bundle GO-purity")
    purity = {go: [] for go in PILOT_GO}
    baseline = {go: float(y[go].mean()) for go in PILOT_GO}

    n_tests = K_CLUSTERS * len(PILOT_GO)
    alpha_bonf = 0.01 / n_tests
    winners = []

    for c in range(K_CLUSTERS):
        mask = (cid == c)
        n_c = int(mask.sum())
        for go in PILOT_GO:
            k_pos = int(y[go][mask].sum())
            K_pos = int(y[go].sum())
            frac  = k_pos / max(n_c, 1)
            base  = baseline[go]
            p_hg  = hypergeom_p(k_pos, N, K_pos, n_c) if k_pos > 0 else 1.0
            # bootstrap CI on frac
            if n_c >= 2:
                boot = []
                y_sub = y[go][mask]
                for _ in range(N_BOOTSTRAP):
                    b = rng.choice(n_c, size=n_c, replace=True)
                    boot.append(y_sub[b].mean())
                lo, hi = np.percentile(boot, [2.5, 97.5])
            else:
                lo, hi = float("nan"), float("nan")
            row = dict(cluster=c, go=go, name=PILOT_GO[go],
                       n_cluster=n_c, k_pos=k_pos, K_pos=K_pos, N=N,
                       frac=frac, baseline=base,
                       ratio_to_baseline=frac / base if base > 0 else float("nan"),
                       p_hyper=p_hg, ci_lo=float(lo), ci_hi=float(hi),
                       bonferroni_alpha=alpha_bonf)
            purity[go].append(row)
            if frac >= 2.0 * base and p_hg < alpha_bonf and k_pos >= 5:
                winners.append(row)

    # ---- 7. Save cluster assignment + purity table
    ts = time.strftime("%Y%m%d_%H%M")
    np.savez(f"{OUT_DIR}/curve_cluster_{ts}.npz",
             subset_idx=subset_idx,
             cluster_id=cid,
             curve_vec_240=curve_vec.astype(np.float32),
             traj_red=traj_red.astype(np.float32),
             pca_expl_var=pca.explained_variance_ratio_)

    summary = dict(
        pilot_go=PILOT_GO,
        N=N,
        k_pca=K_PCA,
        k_clusters=K_CLUSTERS,
        baseline=baseline,
        n_tests=n_tests,
        bonferroni_alpha=alpha_bonf,
        purity=purity,
        winners=winners,
        elapsed_s=time.time() - t0,
    )
    with open(f"{OUT_DIR}/purity_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{time.time()-t0:5.1f}s] winners (cluster,GO): {len(winners)}")
    for w in winners:
        print(f"    cluster {w['cluster']:2d} :: {w['go']} {w['name']:32s}"
              f"  frac={w['frac']:.3f} (base={w['baseline']:.3f},"
              f" x{w['ratio_to_baseline']:.2f}, p={w['p_hyper']:.2e},"
              f" 95CI [{w['ci_lo']:.3f},{w['ci_hi']:.3f}], n={w['n_cluster']})")

    # ---- 8. 3D visualization per GO
    for go in PILOT_GO:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for c in range(K_CLUSTERS):
            mask = (cid == c)
            if mask.sum() == 0:
                continue
            mean_traj = traj_red[mask].mean(axis=0)   # (30, K_PCA)
            frac = y[go][mask].mean() if mask.sum() > 0 else 0.0
            col = plt.cm.viridis(min(1.0, frac / (2 * baseline[go] + 1e-9)))
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2],
                    color=col, lw=2,
                    label=f"c{c} n={mask.sum()} f={frac:.2f}")
            ax.scatter(mean_traj[0, 0], mean_traj[0, 1], mean_traj[0, 2],
                       color=col, marker="o", s=40)
            ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1], mean_traj[-1, 2],
                       color=col, marker="X", s=60)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title(f"Mean L1->L30 trajectory per cluster\n{go} {PILOT_GO[go]}"
                     f" (baseline={baseline[go]:.3f})")
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.05, 1.0))
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/trajectory_3d_{go.replace(':','_')}_{ts}.png",
                    dpi=140)
        plt.close(fig)

    print(f"[{time.time()-t0:5.1f}s] done. results in {OUT_DIR}")
    return summary


if __name__ == "__main__":
    main()
