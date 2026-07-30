"""
exp_fluid_stage1g_l30_shootout.py
=================================
Falsification experiment: does the fluid curve_vec add value over
mean-pool L30 alone?

Three competing methods, identical pilot subset (narrow-GO union) and
identical grid points, identical Bonferroni threshold per grid point:

    (1) fluid     : joint PCA on (isoform x layer) points, reshape to
                    30*K_PCA vector.
    (2) L30_pca   : PCA on L30 mean-pool only, K_PCA*30 axes (matched
                    dimensionality to fluid).
    (3) L30_raw   : raw L30 mean-pool 640-D (no PCA).

Decision gate
-------------
    fluid_winners >= L30_pca_winners + 3   AND  >= L30_raw_winners + 3
        -> fluid framework has novel value
    otherwise
        -> curve_vec is a re-projection of L30
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

N_LAYERS = 30
EMB_DIM  = 640
SEED_BASE = 42

NARROW_GO = {
    "GO:0006096": "Glycolysis",
    "GO:0006099": "TCA cycle",
    "GO:0006120": "Complex I NADH ox",
    "GO:0006754": "ATP biosynthesis",
    "GO:0006635": "FA beta-oxidation",
    "GO:0006418": "tRNA aminoacylation",
    "GO:0045214": "Sarcomere organization",
    "GO:0030048": "Actin-based movement",
    "GO:0007018": "MT-based movement",
    "GO:0007015": "Actin filament org",
    "GO:0007204": "Ca2+ signaling",
    "GO:0006816": "Ca2+ transport",
    "GO:0006888": "ER-Golgi transport",
    "GO:0000398": "mRNA splicing spliceosome",
    "GO:0006414": "Translational elongation",
}

# reduced grid for tractability (still balanced across axes)
GRID_KPCA  = [4, 8, 16]
GRID_KCLUS = [8, 16]
GRID_SEEDS = [42, 137]


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


def score_grid_point(feat_matrix, y_dict, n_go, k_clus, seed, N, baseline):
    km = KMeans(n_clusters=k_clus, n_init=5, random_state=seed)
    cid = km.fit_predict(feat_matrix)
    n_tests = k_clus * n_go
    alpha = 0.01 / n_tests
    per_go_sig = {}
    for go, y in y_dict.items():
        K_pos = int(y.sum())
        n_sig = 0
        for c in range(k_clus):
            m = (cid == c)
            n_c = int(m.sum())
            if n_c == 0:
                continue
            k_p = int(y[m].sum())
            frac = k_p / n_c
            ratio = frac / baseline[go] if baseline[go] > 0 else 0.0
            if k_p > 0:
                p_hg = float(hypergeom.sf(k_p - 1, N, K_pos, n_c))
            else:
                p_hg = 1.0
            if ratio >= 2.0 and p_hg < alpha and k_p >= 5:
                n_sig += 1
        per_go_sig[go] = n_sig
    return per_go_sig, alpha


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
    rng = np.random.default_rng(SEED_BASE)
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False).tolist()
    subset_idx = np.array(sorted(pos_idx + list(neg_idx)))
    N = len(subset_idx)
    sub_sym = [te_sym[i] for i in subset_idx]
    print(f"[{time.time()-t0:5.1f}s] pilot N={N} (pos_union={len(pos_idx)})")

    y = {go: np.array([1 if s in go_pos[go] else 0 for s in sub_sym],
                      dtype=np.int32) for go in NARROW_GO}
    baseline = {go: float(y[go].mean()) for go in NARROW_GO}
    n_go = len(NARROW_GO)

    # ---- build trajectory
    print(f"[{time.time()-t0:5.1f}s] building trajectory ...")
    traj = build_trajectory(subset_idx)
    print(f"[{time.time()-t0:5.1f}s] traj {traj.shape}")
    L30 = traj[:, -1, :].copy()                     # (N, 640)

    # ---- fluid PCA fit at K_MAX
    K_MAX = max(GRID_KPCA)
    flat = traj.reshape(N * N_LAYERS, EMB_DIM)
    print(f"[{time.time()-t0:5.1f}s] fluid joint PCA K={K_MAX}")
    pca_fluid = PCA(n_components=K_MAX, random_state=SEED_BASE,
                    svd_solver="randomized")
    flat_red = pca_fluid.fit_transform(flat)
    print(f"          fluid PCA expl_var={pca_fluid.explained_variance_ratio_.sum():.3f}")
    del flat, traj
    gc.collect()

    # ---- L30 PCA fit at K_MAX*30 (matched to fluid max dim)
    L30_TARGET_DIM = K_MAX * N_LAYERS   # = 480 for K=16
    L30_target = min(L30_TARGET_DIM, L30.shape[1])
    print(f"[{time.time()-t0:5.1f}s] L30 PCA K={L30_target}")
    pca_l30 = PCA(n_components=L30_target, random_state=SEED_BASE,
                  svd_solver="randomized")
    L30_red = pca_l30.fit_transform(L30)
    print(f"          L30 PCA expl_var({L30_target})={pca_l30.explained_variance_ratio_.sum():.3f}")

    # ---- grid loop across methods
    fluid_traj_red_by_k = {k: flat_red[:, :k].reshape(N, N_LAYERS, k)
                           for k in GRID_KPCA}
    del flat_red
    gc.collect()

    winners = {"fluid": defaultdict(int),
               "L30_pca": defaultdict(int),
               "L30_raw": defaultdict(int)}
    n_grid = 0
    per_gp = []

    for k_pca in GRID_KPCA:
        for k_clus in GRID_KCLUS:
            for seed in GRID_SEEDS:
                t1 = time.time()
                # method 1: fluid
                cv_fluid = fluid_traj_red_by_k[k_pca].reshape(
                    N, N_LAYERS * k_pca)
                sig_f, _ = score_grid_point(cv_fluid, y, n_go, k_clus, seed,
                                            N, baseline)
                # method 2: L30 PCA matched to same dim as fluid
                dim_matched = k_pca * N_LAYERS
                cv_l30p = L30_red[:, :dim_matched]
                sig_p, _ = score_grid_point(cv_l30p, y, n_go, k_clus, seed,
                                            N, baseline)
                # method 3: raw 640-D L30 (k_pca irrelevant to feature)
                sig_r, _ = score_grid_point(L30, y, n_go, k_clus, seed,
                                            N, baseline)

                for go in NARROW_GO:
                    winners["fluid"][go]   += (1 if sig_f[go] >= 1 else 0)
                    winners["L30_pca"][go] += (1 if sig_p[go] >= 1 else 0)
                    winners["L30_raw"][go] += (1 if sig_r[go] >= 1 else 0)

                per_gp.append(dict(
                    grid=[k_pca, k_clus, seed],
                    n_sig_fluid={g: sig_f[g] for g in NARROW_GO},
                    n_sig_L30_pca={g: sig_p[g] for g in NARROW_GO},
                    n_sig_L30_raw={g: sig_r[g] for g in NARROW_GO},
                ))
                n_grid += 1
                print(f"[{time.time()-t0:5.1f}s] grid k_pca={k_pca} "
                      f"k_clus={k_clus} seed={seed}  ({time.time()-t1:.1f}s)")

    # ---- aggregate: per-GO stability across grid per method
    stab = {m: {} for m in winners}
    total_robust = {m: 0 for m in winners}
    for m in winners:
        for go in NARROW_GO:
            s = winners[m][go] / n_grid
            stab[m][go] = s
            if s >= 0.80:
                total_robust[m] += 1

    print()
    print("=" * 78)
    print(f"L30 shootout (narrow-GO 15 terms, {n_grid} grid points)")
    print("=" * 78)
    print(f"{'method':<10}{'robust_count':>15}{'ratio_of_total':>17}")
    for m in ["fluid", "L30_pca", "L30_raw"]:
        print(f"{m:<10}{total_robust[m]:>15}{total_robust[m]/n_go:>17.2f}")

    print()
    print(f"{'GO':<11}{'name':<28}"
          f"{'fluid':>9}{'L30_pca':>10}{'L30_raw':>10}")
    for go in sorted(NARROW_GO, key=lambda g: -stab["fluid"][g]):
        print(f"{go:<11}{NARROW_GO[go][:27]:<28}"
              f"{stab['fluid'][go]:>9.2f}"
              f"{stab['L30_pca'][go]:>10.2f}"
              f"{stab['L30_raw'][go]:>10.2f}")

    # ---- decision gate
    diff_pca = total_robust["fluid"] - total_robust["L30_pca"]
    diff_raw = total_robust["fluid"] - total_robust["L30_raw"]
    print()
    print(f"fluid - L30_pca = {diff_pca:+d}   fluid - L30_raw = {diff_raw:+d}")
    if diff_pca >= 3 and diff_raw >= 3:
        print("DECISION: FLUID HAS NOVEL VALUE (both diffs >= +3)")
    elif max(diff_pca, diff_raw) >= 2:
        print("DECISION: FLUID BORDERLINE (some novelty, weak margin)")
    else:
        print("DECISION: FLUID = L30 RE-PROJECTION (novelty NOT proven)")

    ts = time.strftime("%Y%m%d_%H%M")
    with open(f"{OUT_DIR}/l30_shootout_{ts}.json", "w") as f:
        json.dump(dict(
            narrow_go=NARROW_GO,
            grid=dict(K_PCA=GRID_KPCA, K_CLUSTERS=GRID_KCLUS,
                      seeds=GRID_SEEDS),
            n_grid_points=n_grid,
            baseline=baseline,
            stability=stab,
            total_robust=total_robust,
            diff_pca=diff_pca,
            diff_raw=diff_raw,
            decision=("FLUID_NOVEL" if diff_pca >= 3 and diff_raw >= 3
                      else ("BORDERLINE" if max(diff_pca, diff_raw) >= 2
                            else "FLUID_EQUALS_L30")),
            per_gridpoint=per_gp,
        ), f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else o)
    print(f"\nsaved: {OUT_DIR}/l30_shootout_{ts}.json")
    print(f"total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
