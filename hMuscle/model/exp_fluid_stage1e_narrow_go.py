"""
exp_fluid_stage1e_narrow_go.py
==============================
Re-run Stage 1 grid on a set of *narrow, mechanistically specific* GO
terms — the ones that are essentially one or two enzymatic /
structural functions, avoiding broad regulatory / developmental terms.

Rationale
---------
In the 18-BP scan only 6 GO terms were robust across the 48-point
grid, and they were exactly the mechanistically specific ones
(glycolysis, sarcomere, MT / actin movement, Ca signaling, cytoskeleton
org).  If the fluid framework's real handle is functional specificity,
a curated narrow-specific GO set should yield a much higher robust
fraction, and also cover complementary axes (Complex I, TCA, tRNA
aminoacylation, splicing, translation, vesicle transport).
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

NARROW_GO = {
    # --- enzyme / metabolism (very specific) ---
    "GO:0006096": "Glycolysis",
    "GO:0006099": "TCA cycle",
    "GO:0006120": "Complex I NADH ox",
    "GO:0006754": "ATP biosynthesis",
    "GO:0006635": "FA beta-oxidation",
    "GO:0006418": "tRNA aminoacylation",
    # --- structural / cytoskeleton (specific) ---
    "GO:0045214": "Sarcomere organization",
    "GO:0030048": "Actin-based movement",
    "GO:0007018": "MT-based movement",
    "GO:0007015": "Actin filament org",
    # --- signaling / transport (specific) ---
    "GO:0007204": "Ca2+ signaling",
    "GO:0006816": "Ca2+ transport",
    "GO:0006888": "ER-Golgi transport",
    # --- RNA / translation (specific) ---
    "GO:0000398": "mRNA splicing spliceosome",
    "GO:0006414": "Translational elongation",
}

GRID_KPCA  = [4, 8, 12, 16]
GRID_KCLUS = [8, 12, 16, 20]
GRID_SEEDS = [42, 137, 271]


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

    go_pos = {go: load_go_pos(go) for go in NARROW_GO}
    pos_counts = {go: sum(1 for s in te_sym if s in go_pos[go]) for go in NARROW_GO}
    print(f"[{time.time()-t0:5.1f}s] narrow-GO positives in test:")
    for go in NARROW_GO:
        print(f"          {go} {NARROW_GO[go]:32s}  n_pos={pos_counts[go]}")

    all_pos = set().union(*go_pos.values())
    pos_idx = sorted({i for i, s in enumerate(te_sym) if s in all_pos})
    neg_pool = [i for i, s in enumerate(te_sym) if s not in all_pos]
    n_neg = min(len(pos_idx), len(neg_pool))
    rng = np.random.default_rng(SEED_BASE)
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False).tolist()

    subset_idx = np.array(sorted(pos_idx + list(neg_idx)))
    N = len(subset_idx)
    sub_sym = [te_sym[i] for i in subset_idx]
    print(f"[{time.time()-t0:5.1f}s] pilot subset N={N} "
          f"(pos_union={len(pos_idx)}, neg={n_neg})")

    y = {go: np.array([1 if s in go_pos[go] else 0 for s in sub_sym],
                      dtype=np.int32) for go in NARROW_GO}
    baseline = {go: float(y[go].mean()) for go in NARROW_GO}

    print(f"[{time.time()-t0:5.1f}s] building trajectory ...")
    traj = build_trajectory(subset_idx)
    print(f"[{time.time()-t0:5.1f}s] traj {traj.shape} "
          f"{traj.nbytes/1e9:.2f} GB")

    K_MAX = max(GRID_KPCA)
    flat = traj.reshape(N * N_LAYERS, EMB_DIM)
    print(f"[{time.time()-t0:5.1f}s] joint PCA (K={K_MAX}) on {flat.shape[0]} pts")
    pca = PCA(n_components=K_MAX, random_state=SEED_BASE,
              svd_solver="randomized")
    flat_red_full = pca.fit_transform(flat)
    print(f"          expl_var_sum(K={K_MAX})="
          f"{pca.explained_variance_ratio_.sum():.3f}")
    del flat, traj
    gc.collect()

    traj_red_by_k = {k: flat_red_full[:, :k].reshape(N, N_LAYERS, k)
                     for k in GRID_KPCA}
    del flat_red_full
    gc.collect()

    n_go = len(NARROW_GO)
    grid_records = []
    detailed = defaultdict(list)
    per_gp_details = []

    for k_pca in GRID_KPCA:
        for k_clus in GRID_KCLUS:
            for seed in GRID_SEEDS:
                gpt = (k_pca, k_clus, seed)
                t1 = time.time()
                cv = traj_red_by_k[k_pca].reshape(N, N_LAYERS * k_pca)
                km = KMeans(n_clusters=k_clus, n_init=10, random_state=seed)
                cid = km.fit_predict(cv)

                n_tests = k_clus * n_go
                alpha_bonf = 0.01 / n_tests

                gp_rows = []
                for go in NARROW_GO:
                    K_pos = int(y[go].sum())
                    max_ratio = 0.0
                    n_sig = 0
                    best = None
                    for c in range(k_clus):
                        m = (cid == c)
                        n_c = int(m.sum())
                        if n_c == 0:
                            continue
                        k_p = int(y[go][m].sum())
                        frac  = k_p / n_c
                        ratio = frac / baseline[go] if baseline[go] > 0 else 0.0
                        p_hg  = (float(hypergeom.sf(k_p - 1, N, K_pos, n_c))
                                 if k_p > 0 else 1.0)
                        if ratio >= 2.0 and p_hg < alpha_bonf and k_p >= 5:
                            n_sig += 1
                        if ratio > max_ratio:
                            max_ratio = ratio
                            best = dict(cluster=c, k_p=k_p, n=n_c,
                                        frac=frac, ratio=ratio, p=p_hg)
                    detailed[go].append(dict(grid=list(gpt),
                                             max_ratio=max_ratio,
                                             n_sig=n_sig,
                                             best=best))
                    gp_rows.append(dict(go=go, name=NARROW_GO[go],
                                        max_ratio=max_ratio,
                                        n_sig=n_sig, best=best))

                per_gp_details.append(dict(grid=list(gpt),
                                           n_tests=n_tests,
                                           bonferroni=alpha_bonf,
                                           rows=gp_rows))
                grid_records.append(dict(grid=list(gpt),
                                         elapsed_s=time.time() - t1))
                print(f"[{time.time()-t0:5.1f}s] grid "
                      f"K_PCA={k_pca} K_CLUS={k_clus} seed={seed}  "
                      f"({time.time()-t1:.1f}s)")

    # aggregate
    aggregate = {}
    n_grid = len(grid_records)
    for go in NARROW_GO:
        rows = detailed[go]
        max_ratios = [r["max_ratio"] for r in rows]
        n_sigs     = [r["n_sig"]     for r in rows]
        stab       = float(np.mean([1 if ns >= 1 else 0 for ns in n_sigs]))
        aggregate[go] = dict(
            name=NARROW_GO[go],
            n_pos=int(y[go].sum()),
            baseline=baseline[go],
            n_grid=n_grid,
            stability=stab,
            ratio_med=float(np.median(max_ratios)),
            ratio_min=float(np.min(max_ratios)),
            ratio_max=float(np.max(max_ratios)),
            n_sig_med=float(np.median(n_sigs)),
            n_sig_max=int(np.max(n_sigs)),
            robust=(stab >= 0.80),
        )

    # print summary
    print()
    print("=" * 80)
    print(f"Narrow-GO ({n_go} terms) stability across {n_grid} grid points")
    print("=" * 80)
    print(f"{'GO':<11}{'name':<28}{'n_pos':>7}{'base':>8}"
          f"{'stab':>7}{'ratio_med':>11}{'nsig_med':>10}{'ROBUST':>8}")
    for go in sorted(NARROW_GO, key=lambda g: -aggregate[g]["stability"]):
        a = aggregate[go]
        flag = "YES" if a["robust"] else "-"
        print(f"{go:<11}{a['name'][:27]:<28}{a['n_pos']:>7}"
              f"{a['baseline']:>8.3f}{a['stability']:>7.2f}"
              f"{a['ratio_med']:>11.2f}"
              f"{a['n_sig_med']:>10.1f}{flag:>8}")

    n_robust = sum(1 for go in NARROW_GO if aggregate[go]["robust"])
    print(f"\nrobust: {n_robust}/{n_go} = {n_robust/n_go*100:.0f}%")

    ts = time.strftime("%Y%m%d_%H%M")
    with open(f"{OUT_DIR}/narrow_go_{ts}.json", "w") as f:
        json.dump(dict(
            narrow_go=NARROW_GO,
            grid_axes=dict(K_PCA=GRID_KPCA, K_CLUSTERS=GRID_KCLUS,
                           seeds=GRID_SEEDS),
            n_grid_points=n_grid,
            pilot_N=N,
            pilot_pos_union=len(pos_idx),
            baseline=baseline,
            aggregate=aggregate,
            per_gridpoint=per_gp_details,
            robust_count=n_robust,
        ), f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else o)
    print(f"\nsaved: {OUT_DIR}/narrow_go_{ts}.json")
    print(f"total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
