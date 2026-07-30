"""
exp_fluid_stage1f_complex1_subflow.py
=====================================
Within-gene bifurcation analysis for Complex I subunits.

Question
--------
Do multiple isoforms of the same Complex I subunit gene form a *single*
compact fluid trajectory (i.e. isoform-invariant PLM signal, mean-pool
would work) or do they diverge along L1..L30 (i.e. fluid genuinely
resolves isoform-level information)?

Genes analyzed
--------------
    Core BISECT trio: NDUFS4, NDUFS7, NDUFS8
    Broader Complex I family (all NDUFA/B/C/S/V/AF subunits present)
    Controls: TPM1 (structural muscle), KIF21B (motor), DOCK11 (GEF)

Analysis
--------
1.  Extract L1..L30 mean-pooled trajectories for every isoform of the
    target gene set (from full 36,748 test isoforms).
2.  Fit joint PCA (K=6) on all target-gene layer points.
3.  Per gene:
     - Plot all isoform trajectories in 3D PCA (color per isoform).
     - Within-gene layer-wise dispersion: mean pairwise Euclidean
       distance among isoform points at each of 30 layers.
     - Identify bifurcation layer L* = argmax layer dispersion (only
       meaningful if gene has >= 2 isoforms).
4.  Complex I *family* bundling:
     - Gene centroid = mean of that gene's isoform curve_vec.
     - Complex I core subunit family: are their centroids closer to
       each other than to controls?  (silhouette score / centroid
       distance heatmap.)
5.  Save per-gene reports, dispersion CSV, bifurcation summary, 3D PNGs.
"""

import os, json, time, gc, re
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = "../data"
ID_DIR     = "../data/raw_data/data/id_lists"
OUT_DIR    = "../../reports/fluid_stage1"
os.makedirs(OUT_DIR, exist_ok=True)

N_LAYERS = 30
EMB_DIM  = 640
SEED     = 42
K_PCA    = 6

# Focus set
CORE_BISECT   = ["NDUFS4", "NDUFS7", "NDUFS8"]
COMPLEX_I_FAM = [
    # NDUFS (7 core catalytic subunits, S = supernumerary now numbering 1-8)
    "NDUFS1", "NDUFS2", "NDUFS3", "NDUFS4", "NDUFS5",
    "NDUFS6", "NDUFS7", "NDUFS8",
    # NDUFA (accessory, 1-13)
    "NDUFA1", "NDUFA2", "NDUFA3", "NDUFA4", "NDUFA5",
    "NDUFA6", "NDUFA7", "NDUFA8", "NDUFA9", "NDUFA10",
    "NDUFA11", "NDUFA12", "NDUFA13",
    # NDUFB (membrane arm 1-11)
    "NDUFB1", "NDUFB2", "NDUFB3", "NDUFB4", "NDUFB5",
    "NDUFB6", "NDUFB7", "NDUFB8", "NDUFB9", "NDUFB10", "NDUFB11",
    # NDUFC (1-2)
    "NDUFC1", "NDUFC2",
    # NDUFV (matrix arm 1-3)
    "NDUFV1", "NDUFV2", "NDUFV3",
    # NDUFAF (assembly factors 1-8)
    "NDUFAF1", "NDUFAF2", "NDUFAF3", "NDUFAF4",
    "NDUFAF5", "NDUFAF6", "NDUFAF7", "NDUFAF8",
]
CONTROLS = ["TPM1", "KIF21B", "DOCK11", "DLG1", "ACTB"]

FOCUS_GENES = list(dict.fromkeys(CORE_BISECT + COMPLEX_I_FAM + CONTROLS))


def load_ids(p):
    a = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def load_e2s():
    m = {}
    with open(f"{ID_DIR}/ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                m[parts[0]] = parts[4]
    return m


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

    focus_set = set(FOCUS_GENES)
    focus_idx = [i for i, s in enumerate(te_sym) if s in focus_set]
    focus_sym = [te_sym[i] for i in focus_idx]
    focus_iso = [te_iso[i] for i in focus_idx]

    print(f"[{time.time()-t0:5.1f}s] focus isoforms: {len(focus_idx)} "
          f"(from {len(FOCUS_GENES)} target genes)")
    from collections import Counter
    sym_counts = Counter(focus_sym)
    print(f"  genes with ≥2 isoforms:")
    for s, n in sym_counts.most_common():
        if n >= 2:
            tag = "  [BISECT]" if s in CORE_BISECT else \
                  ("  [CTRL]"   if s in CONTROLS   else "")
            print(f"    {s:10s}  {n} isoforms{tag}")

    # trajectory
    print(f"[{time.time()-t0:5.1f}s] building trajectory ...")
    traj = build_trajectory(np.array(focus_idx))
    print(f"[{time.time()-t0:5.1f}s] traj {traj.shape} "
          f"{traj.nbytes/1e9:.3f} GB")

    # joint PCA
    N = traj.shape[0]
    flat = traj.reshape(N * N_LAYERS, EMB_DIM)
    pca  = PCA(n_components=K_PCA, random_state=SEED, svd_solver="randomized")
    flat_red = pca.fit_transform(flat)
    print(f"  joint PCA expl_var={pca.explained_variance_ratio_.sum():.3f}")
    traj_red = flat_red.reshape(N, N_LAYERS, K_PCA)
    del traj, flat
    gc.collect()

    # curve vec 30 * K_PCA
    curve_vec = traj_red.reshape(N, N_LAYERS * K_PCA)

    # ---- per-gene analysis ----
    per_gene = {}
    dispersion_table = {}

    for sym in FOCUS_GENES:
        member_idx = [i for i, s in enumerate(focus_sym) if s == sym]
        n = len(member_idx)
        if n == 0:
            continue
        pts_traj = traj_red[member_idx]           # (n, 30, K_PCA)
        pts_cv   = curve_vec[member_idx]          # (n, 30*K_PCA)
        pts_iso  = [focus_iso[i] for i in member_idx]

        # layer-wise within-gene dispersion (mean pairwise Euclidean)
        disp = np.zeros(N_LAYERS)
        for L in range(N_LAYERS):
            if n >= 2:
                d = pdist(pts_traj[:, L, :])
                disp[L] = d.mean()
        bifurc_L = int(np.argmax(disp)) if n >= 2 else -1

        # centroid in curve_vec space
        centroid = pts_cv.mean(axis=0)
        centroid_traj = pts_traj.mean(axis=0)     # (30, K_PCA)

        per_gene[sym] = dict(
            n_isoforms=n,
            iso_ids=pts_iso,
            centroid=centroid.tolist(),
            centroid_traj=centroid_traj.tolist(),
            layer_dispersion=disp.tolist(),
            bifurcation_layer=bifurc_L,
            bifurcation_dispersion=float(disp.max()),
            mean_dispersion=float(disp.mean()),
        )
        dispersion_table[sym] = disp

    # ---- family bundling: silhouette on Complex I vs controls ----
    print(f"\n[{time.time()-t0:5.1f}s] family bundling silhouette")
    gene_centroids = {}
    for sym, r in per_gene.items():
        if r["n_isoforms"] >= 1:
            gene_centroids[sym] = np.array(r["centroid"])
    fam_labels = []
    fam_pts    = []
    for sym, c in gene_centroids.items():
        if sym in COMPLEX_I_FAM:
            fam_labels.append("complexI")
        elif sym in CONTROLS:
            fam_labels.append("control")
        else:
            continue
        fam_pts.append(c)
    fam_pts = np.array(fam_pts)
    fam_labels_arr = np.array(fam_labels)
    n_ci = int((fam_labels_arr == "complexI").sum())
    n_ct = int((fam_labels_arr == "control").sum())
    if n_ci >= 2 and n_ct >= 2:
        sil = silhouette_score(fam_pts, fam_labels_arr)
    else:
        sil = float("nan")
    print(f"  complexI genes={n_ci}, control genes={n_ct}, "
          f"silhouette={sil:.3f}")

    # centroid distance matrix (subset)
    sym_list = list(gene_centroids.keys())
    C = np.stack([gene_centroids[s] for s in sym_list])
    D = squareform(pdist(C))
    dist_mat = dict(genes=sym_list, matrix=D.tolist())

    # ---- bifurcation table (Complex I trio + rest) ----
    print(f"\n[{time.time()-t0:5.1f}s] within-gene bifurcation summary")
    print(f"  {'gene':12s}  {'n_iso':>5s}  {'bifL':>4s}  "
          f"{'disp@bifL':>10s}  {'mean_disp':>10s}  category")
    for sym in FOCUS_GENES:
        if sym not in per_gene:
            continue
        r = per_gene[sym]
        if r["n_isoforms"] < 2:
            continue
        cat = ("BISECT" if sym in CORE_BISECT else
               ("ComplexI" if sym in COMPLEX_I_FAM else "control"))
        print(f"  {sym:12s}  {r['n_isoforms']:>5d}  "
              f"{r['bifurcation_layer']:>4d}  "
              f"{r['bifurcation_dispersion']:>10.3f}  "
              f"{r['mean_dispersion']:>10.3f}  {cat}")

    # ---- 3D visualization: BISECT trio + one control side by side ----
    fig_paths = []
    plot_targets = [g for g in CORE_BISECT + ["TPM1", "DOCK11"]
                    if g in per_gene and per_gene[g]["n_isoforms"] >= 2]
    for sym in plot_targets:
        r = per_gene[sym]
        # get trajectories
        member_idx = [i for i, s in enumerate(focus_sym) if s == sym]
        pts_traj = traj_red[member_idx]
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        cmap = plt.cm.tab20
        for k in range(pts_traj.shape[0]):
            col = cmap(k % 20)
            ax.plot(pts_traj[k, :, 0], pts_traj[k, :, 1], pts_traj[k, :, 2],
                    color=col, lw=1.5, alpha=0.9,
                    label=focus_iso[member_idx[k]][:20])
            ax.scatter(pts_traj[k, 0, 0], pts_traj[k, 0, 1], pts_traj[k, 0, 2],
                       color=col, marker="o", s=30)
            ax.scatter(pts_traj[k, -1, 0], pts_traj[k, -1, 1], pts_traj[k, -1, 2],
                       color=col, marker="X", s=60)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title(f"{sym}  n_iso={r['n_isoforms']}  "
                     f"bifL={r['bifurcation_layer']}  "
                     f"max_disp={r['bifurcation_dispersion']:.2f}")
        if pts_traj.shape[0] <= 8:
            ax.legend(fontsize=6, loc="upper left",
                      bbox_to_anchor=(1.05, 1.0))
        fig.tight_layout()
        outp = f"{OUT_DIR}/subflow_{sym}.png"
        fig.savefig(outp, dpi=140)
        plt.close(fig)
        fig_paths.append(outp)

    # dispersion overlay figure (all multi-iso genes)
    fig, ax = plt.subplots(figsize=(10, 6))
    for sym in FOCUS_GENES:
        if sym not in per_gene or per_gene[sym]["n_isoforms"] < 2:
            continue
        disp = dispersion_table[sym]
        if sym in CORE_BISECT:
            ax.plot(range(1, N_LAYERS + 1), disp, color="crimson", lw=2,
                    marker="o", label=f"{sym} [BISECT]")
        elif sym in COMPLEX_I_FAM:
            ax.plot(range(1, N_LAYERS + 1), disp, color="tab:blue",
                    alpha=0.35, lw=1)
        elif sym in CONTROLS:
            ax.plot(range(1, N_LAYERS + 1), disp, color="tab:orange",
                    lw=2, marker="s", label=f"{sym} [CTRL]")
    ax.set_xlabel("ESM-2 layer L")
    ax.set_ylabel("within-gene mean pairwise distance (PCA-6 space)")
    ax.set_title("Layer-wise within-gene dispersion")
    ax.legend(fontsize=8, loc="best")
    ax.axvspan(15, 27, alpha=0.08, color="gray", label="mid-late (L15-27)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/subflow_dispersion.png", dpi=140)
    plt.close(fig)
    fig_paths.append(f"{OUT_DIR}/subflow_dispersion.png")

    # save
    with open(f"{OUT_DIR}/complex1_subflow.json", "w") as f:
        json.dump(dict(
            pilot_N=int(N),
            k_pca=K_PCA,
            genes_analyzed=list(per_gene.keys()),
            per_gene={s: {k: v for k, v in r.items()
                         if k not in ("iso_ids",)}
                     for s, r in per_gene.items()},
            centroid_distance_matrix=dist_mat,
            silhouette_complexI_vs_control=float(sil),
            n_complexI_centroids=n_ci,
            n_control_centroids=n_ct,
        ), f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else o)
    print(f"\nsaved: {OUT_DIR}/complex1_subflow.json")
    print(f"figures: {fig_paths}")
    print(f"total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
