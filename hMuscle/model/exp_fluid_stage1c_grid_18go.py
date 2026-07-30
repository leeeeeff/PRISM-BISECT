"""
exp_fluid_stage1c_grid_18go.py
===============================
Grid sensitivity + 18-GO expansion for Stage-1 fluid pilot.

For each (K_PCA, K_CLUSTERS, seed) triple:
    1. joint PCA on (isoform x layer) points
    2. KMeans on 30 * K_PCA flattened curve vectors
    3. per-GO purity (18 BP terms) + hypergeometric p
       (Bonferroni-adjusted for total tests within triple)

Aggregation per GO
------------------
    stability(g) = fraction of grid points where at least one cluster
                   satisfies frac >= 2*baseline AND p < alpha_bonf AND k+ >= 5
    typical_max_ratio(g) = median over grid of max cluster ratio
    typical_n_sig(g)     = median over grid of significant-cluster count

Decision
--------
GO is *robust* if stability(g) >= 0.80.
"""

import os, json, time, gc
import numpy as np
from collections import defaultdict
from scipy.stats import hypergeom
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = "../data"
ANNOT_FILE = "../data/raw_data/data/annotations/human_annotations_unified_bp.txt"
ID_DIR     = "../data/raw_data/data/id_lists"
OUT_DIR    = "../../reports/fluid_stage1"
os.makedirs(OUT_DIR, exist_ok=True)

N_LAYERS  = 30
EMB_DIM   = 640
SEED_BASE = 42

GO_TERMS = {
    "GO:0007204": "Ca2+ signaling",
    "GO:0045214": "Sarcomere organization",
    "GO:0006941": "Muscle contraction",
    "GO:0006914": "Autophagy",
    "GO:0043161": "Proteasome-UPS",
    "GO:0007519": "Skeletal muscle dev",
    "GO:0042692": "Muscle cell diff",
    "GO:0055074": "Ca2+ homeostasis",
    "GO:0007005": "Mitochondrion org",
    "GO:0007517": "Muscle organ dev",
    "GO:0032006": "TOR signaling",
    "GO:0030048": "Actin-based movement",
    "GO:0006096": "Glycolysis",
    "GO:0007268": "Synaptic transmission",
    "GO:0007018": "MT-based movement",
    "GO:0031175": "Neuron proj development",
    "GO:0030182": "Neuron diff",
    "GO:0000226": "MT cytoskeleton org",
}

GRID_KPCA     = [4, 8, 12, 16]
GRID_KCLUS    = [8, 12, 16, 20]
GRID_SEEDS    = [42, 137, 271]


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
    e2s     = load_e2s()
    te_sym  = [e2s.get(g.split(".")[0], g.split(".")[0]) for g in te_gene]
    N_TE = len(te_iso)
    print(f"[{time.time()-t0:5.1f}s] loaded {N_TE} test isoforms")

    # 18-GO positive sets
    go_pos = {go: load_go_pos(go) for go in GO_TERMS}
    pos_counts = {go: sum(1 for s in te_sym if s in go_pos[go]) for go in GO_TERMS}
    print(f"[{time.time()-t0:5.1f}s] per-GO positives in test:")
    for go in GO_TERMS:
        print(f"          {go} {GO_TERMS[go]:32s}  n_pos={pos_counts[go]}")

    all_pos = set().union(*go_pos.values())
    pos_idx = sorted({i for i, s in enumerate(te_sym) if s in all_pos})
    neg_pool = [i for i, s in enumerate(te_sym) if s not in all_pos]

    # matched neg pool (same size as positive union, cap ~20k)
    n_neg = min(len(pos_idx), len(neg_pool), 20000 - len(pos_idx))
    rng = np.random.default_rng(SEED_BASE)
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False).tolist()

    subset_idx = np.array(sorted(pos_idx + list(neg_idx)))
    N = len(subset_idx)
    sub_sym = [te_sym[i] for i in subset_idx]
    print(f"[{time.time()-t0:5.1f}s] pilot subset N={N} "
          f"(pos_union={len(pos_idx)}, neg={n_neg})")

    y = {go: np.array([1 if s in go_pos[go] else 0 for s in sub_sym],
                      dtype=np.int32) for go in GO_TERMS}
    baseline = {go: float(y[go].mean()) for go in GO_TERMS}

    # trajectory tensor (one-shot)
    print(f"[{time.time()-t0:5.1f}s] building trajectory tensor ...")
    traj = build_trajectory(subset_idx)
    print(f"[{time.time()-t0:5.1f}s] traj {traj.shape} {traj.nbytes/1e9:.2f} GB")

    K_PCA_MAX = max(GRID_KPCA)
    flat = traj.reshape(N * N_LAYERS, EMB_DIM)
    print(f"[{time.time()-t0:5.1f}s] joint PCA on {flat.shape[0]} pts "
          f"-> {K_PCA_MAX} axes (fit once, slice per K)")
    pca = PCA(n_components=K_PCA_MAX, random_state=SEED_BASE,
              svd_solver="randomized")
    flat_red_full = pca.fit_transform(flat)
    print(f"          explained_var_sum(K={K_PCA_MAX})="
          f"{pca.explained_variance_ratio_.sum():.3f}")
    del flat, traj
    gc.collect()

    # per-K_PCA reduced tensor cache
    traj_red_by_k = {}
    for k_pca in GRID_KPCA:
        traj_red_by_k[k_pca] = flat_red_full[:, :k_pca].reshape(N, N_LAYERS, k_pca)
    del flat_red_full
    gc.collect()

    # grid loop
    n_go = len(GO_TERMS)
    grid_records = []
    detailed_per_go = defaultdict(list)   # go -> list of (grid_pt_key, max_ratio, n_sig)

    for k_pca in GRID_KPCA:
        for k_clus in GRID_KCLUS:
            for seed in GRID_SEEDS:
                gpt = (k_pca, k_clus, seed)
                t1 = time.time()
                traj_red = traj_red_by_k[k_pca]
                curve_vec = traj_red.reshape(N, N_LAYERS * k_pca)
                km = KMeans(n_clusters=k_clus, n_init=10, random_state=seed)
                cid = km.fit_predict(curve_vec)

                n_tests = k_clus * n_go
                alpha_bonf = 0.01 / n_tests

                # per-GO stats for this grid point
                for go in GO_TERMS:
                    K_pos = int(y[go].sum())
                    max_ratio = 0.0
                    n_sig = 0
                    best_row = None
                    for c in range(k_clus):
                        mask = (cid == c)
                        n_c = int(mask.sum())
                        if n_c == 0:
                            continue
                        k_p = int(y[go][mask].sum())
                        frac = k_p / n_c
                        ratio = frac / baseline[go] if baseline[go] > 0 else 0.0
                        p_hg = (float(hypergeom.sf(k_p - 1, N, K_pos, n_c))
                                if k_p > 0 else 1.0)
                        if (ratio >= 2.0 and p_hg < alpha_bonf and k_p >= 5):
                            n_sig += 1
                        if ratio > max_ratio:
                            max_ratio = ratio
                            best_row = dict(cluster=c, k_pos=k_p, n=n_c,
                                            frac=frac, ratio=ratio, p=p_hg)
                    detailed_per_go[go].append(dict(
                        grid=list(gpt), max_ratio=max_ratio, n_sig=n_sig,
                        best=best_row,
                    ))

                grid_records.append(dict(
                    grid=list(gpt), n_tests=n_tests,
                    bonferroni=alpha_bonf, elapsed_s=time.time() - t1,
                ))
                print(f"[{time.time()-t0:5.1f}s] grid K_PCA={k_pca} "
                      f"K_CLUS={k_clus} seed={seed}  ({time.time()-t1:.1f}s)")

    # aggregate stability per GO
    aggregate = {}
    n_grid = len(grid_records)
    for go in GO_TERMS:
        rows = detailed_per_go[go]
        max_ratios = [r["max_ratio"] for r in rows]
        n_sigs     = [r["n_sig"]     for r in rows]
        stability  = float(np.mean([1 if ns >= 1 else 0 for ns in n_sigs]))
        aggregate[go] = dict(
            name=GO_TERMS[go],
            n_positives_in_pilot=int(y[go].sum()),
            baseline=baseline[go],
            n_grid_points=n_grid,
            stability=stability,                        # fraction with n_sig >= 1
            typical_max_ratio_median=float(np.median(max_ratios)),
            typical_max_ratio_min=float(np.min(max_ratios)),
            typical_max_ratio_max=float(np.max(max_ratios)),
            typical_n_sig_median=float(np.median(n_sigs)),
            typical_n_sig_max=int(np.max(n_sigs)),
            robust=(stability >= 0.80),
        )

    # console summary
    print()
    print("=" * 78)
    print(f"18-GO Stability across {n_grid} grid points "
          f"(K_PCA x K_CLUS x seed = {len(GRID_KPCA)}x{len(GRID_KCLUS)}x{len(GRID_SEEDS)})")
    print("=" * 78)
    print(f"{'GO':<11}{'name':<28}{'n_pos':>7}{'base':>8}"
          f"{'stab':>7}{'ratio_med':>11}{'nsig_med':>10}{'ROBUST':>8}")
    for go in sorted(GO_TERMS, key=lambda g: -aggregate[g]["stability"]):
        a = aggregate[go]
        flag = "YES" if a["robust"] else "-"
        print(f"{go:<11}{a['name'][:27]:<28}{a['n_positives_in_pilot']:>7}"
              f"{a['baseline']:>8.3f}{a['stability']:>7.2f}"
              f"{a['typical_max_ratio_median']:>11.2f}"
              f"{a['typical_n_sig_median']:>10.1f}{flag:>8}")

    ts = time.strftime("%Y%m%d_%H%M")
    with open(f"{OUT_DIR}/grid_18go_{ts}.json", "w") as f:
        json.dump(dict(
            grid_axes=dict(K_PCA=GRID_KPCA, K_CLUSTERS=GRID_KCLUS,
                           seeds=GRID_SEEDS),
            n_grid_points=n_grid,
            pilot_N=N, pilot_pos_union=len(pos_idx),
            baseline=baseline,
            aggregate=aggregate,
            per_gridpoint=grid_records,
            detailed_per_go=detailed_per_go,
        ), f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else o)
    print(f"\nSaved: {OUT_DIR}/grid_18go_{ts}.json")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
