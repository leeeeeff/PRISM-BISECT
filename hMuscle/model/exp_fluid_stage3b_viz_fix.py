"""
exp_fluid_stage3b_viz_fix.py
=============================
Fixed master trajectory visualization:

- Uses *layer-normalized* joint PCA coords (top 3 PCs) so early layers
  are not compressed near the origin — bundles remain distinguishable
  from L1 onwards, not only at L30.
- Two side-by-side 3D panels on one figure:
    (A) bundle-coloured: each of the 6 bundles gets a distinct tab10
        colour, so bundle identity is unambiguous.
    (B) layer-signal-coloured: per-segment cividis (blue -> yellow)
        by GO-specific Fisher signal.
- Layer index markers (L1, L5, ..., L30) along each bundle's mean
  trajectory, with explicit L1 (circle) / L30 (X) start/end symbols.
- Prominent colorbar with tick labels for panel B.
- Sampling restricted to GO+ members of the winner cluster, so two
  GOs sharing a cluster will show different sample paths (different
  members).
"""

import os, json, time, gc
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = "../data"
ANNOT_FILE = "../data/raw_data/data/annotations/human_annotations_unified_bp.txt"
ID_DIR     = "../data/raw_data/data/id_lists"
OUT_DIR    = "../../reports/fluid_stage3"
os.makedirs(OUT_DIR, exist_ok=True)

N_LAYERS = 30
EMB_DIM  = 640
SEED     = 42
K_PCA    = 8
K_CLUS   = 16

FEATURED = {
    "GO:0007204": dict(name="Ca2+ signaling",           expected_peakL=11),
    "GO:0006414": dict(name="Translational elongation", expected_peakL=12),
    "GO:0000398": dict(name="mRNA splicing",            expected_peakL=14),
    "GO:0006418": dict(name="tRNA aminoacylation",      expected_peakL=19),
    "GO:0006635": dict(name="FA beta-oxidation",        expected_peakL=28),
    "GO:0006120": dict(name="Complex I NADH ox",        expected_peakL=30),
}

GO_TERMS_ALL = {
    "GO:0006974": "DNA damage response", "GO:0035556": "Intracellular signal",
    "GO:0006508": "Proteolysis", "GO:0043161": "Proteasome-UPS",
    "GO:0006281": "DNA repair", "GO:0000226": "MT cytoskeleton org",
    "GO:0005975": "Carbohydrate metabolism", "GO:0055074": "Ca2+ homeostasis",
    "GO:0000165": "MAPK cascade", "GO:0000398": "mRNA splicing",
    "GO:0006417": "Regulation of translation", "GO:0007015": "Actin filament org",
    "GO:0007204": "Ca2+ signaling", "GO:0007059": "Chromosome segregation",
    "GO:0007265": "Ras signaling", "GO:0007018": "MT-based movement",
    "GO:0006816": "Ca2+ transport", "GO:0006888": "ER-Golgi transport",
    "GO:0006402": "mRNA catabolism", "GO:0006486": "Protein glycosylation",
    "GO:0006914": "Autophagy", "GO:0006470": "Dephosphorylation",
    "GO:0006836": "Neurotransmitter transp", "GO:0006414": "Translational elongation",
    "GO:0030048": "Actin-based movement", "GO:0032465": "Cytokinesis",
    "GO:0006906": "Vesicle fusion", "GO:0006418": "tRNA aminoacylation",
    "GO:0006754": "ATP biosynthesis", "GO:0006635": "FA beta-oxidation",
    "GO:0006120": "Complex I NADH ox", "GO:0045214": "Sarcomere organization",
    "GO:0006096": "Glycolysis", "GO:0006099": "TCA cycle",
}


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


def per_layer_fisher(traj, y_bool):
    scores = np.zeros(N_LAYERS)
    p = y_bool; n = ~y_bool
    if p.sum() < 5 or n.sum() < 5:
        return scores
    for L in range(N_LAYERS):
        pts = traj[:, L, :]
        mu_p = pts[p].mean(axis=0); mu_n = pts[n].mean(axis=0)
        v_p  = pts[p].var(axis=0);  v_n  = pts[n].var(axis=0)
        scores[L] = ((mu_p - mu_n) ** 2).sum() / ((v_p + v_n).sum() + 1e-9)
    return scores


def colored_line3d(pts, values, cmap, lw=1.5, alpha=0.85):
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_vals = (values[:-1] + values[1:]) / 2.0
    lc = Line3DCollection(segments, cmap=cmap, linewidths=lw, alpha=alpha,
                          norm=plt.Normalize(vmin=0.0, vmax=1.0))
    lc.set_array(seg_vals)
    return lc


def main():
    t0 = time.time()
    te_iso  = load_ids("my_isoform_list_fixed.npy")
    te_gene = load_ids("my_gene_list_fixed.npy")
    e2s = load_e2s()
    te_sym = [e2s.get(g.split(".")[0], g.split(".")[0]) for g in te_gene]
    print(f"[{time.time()-t0:5.1f}s] N_TE={len(te_iso)}")

    go_pos = {go: load_go_pos(go) for go in GO_TERMS_ALL}
    all_pos = set().union(*go_pos.values())
    pos_idx = sorted({i for i, s in enumerate(te_sym) if s in all_pos})
    neg_pool = [i for i, s in enumerate(te_sym) if s not in all_pos]
    n_neg = min(len(pos_idx), len(neg_pool), 15000 - len(pos_idx))
    rng = np.random.default_rng(SEED)
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False).tolist()
    subset_idx = np.array(sorted(pos_idx + list(neg_idx)))
    N = len(subset_idx)
    sub_sym = [te_sym[i] for i in subset_idx]
    print(f"[{time.time()-t0:5.1f}s] pilot N={N}")

    y = {go: np.array([1 if s in go_pos[go] else 0 for s in sub_sym],
                      dtype=bool) for go in FEATURED}

    print(f"[{time.time()-t0:5.1f}s] building trajectory ...")
    traj = build_trajectory(subset_idx)
    print(f"[{time.time()-t0:5.1f}s] traj {traj.shape}")

    # Fisher signal per featured GO
    for go in FEATURED:
        s = per_layer_fisher(traj, y[go])
        FEATURED[go]["observed_peakL"] = int(s.argmax()) + 1
        FEATURED[go]["curve_norm"] = (s - s.min()) / (s.max() - s.min() + 1e-12)

    # layer normalization
    layer_mean = traj.mean(axis=0)
    layer_std  = traj.std(axis=0) + 1e-6
    traj_norm  = (traj - layer_mean) / layer_std
    del traj
    gc.collect()

    # ---- LAYER-NORMALIZED joint PCA (BOTH clustering + viz)
    flat_nm = traj_norm.reshape(N * N_LAYERS, EMB_DIM)
    K_MAX = 16
    print(f"[{time.time()-t0:5.1f}s] joint NORM PCA (K={K_MAX})")
    pca_nm = PCA(n_components=K_MAX, random_state=SEED, svd_solver="randomized")
    reduced_nm = pca_nm.fit_transform(flat_nm).reshape(N, N_LAYERS, K_MAX)
    del flat_nm
    gc.collect()
    coords = reduced_nm[:, :, :3]                                    # (N, 30, 3)
    print(f"          expl_var(K=3)={pca_nm.explained_variance_ratio_[:3].sum():.3f}")

    cv_nm = reduced_nm[:, :, :K_PCA].reshape(N, N_LAYERS * K_PCA)
    km = KMeans(n_clusters=K_CLUS, n_init=5, random_state=SEED)
    cid = km.fit_predict(cv_nm)
    print(f"[{time.time()-t0:5.1f}s] KMeans done")

    baseline = {go: float(y[go].mean()) for go in FEATURED}
    for go in FEATURED:
        best_c = None; best_ratio = 0
        for c in range(K_CLUS):
            m = (cid == c)
            n_c = int(m.sum())
            if n_c < 10:
                continue
            k_p = int(y[go][m].sum())
            if k_p < 5:
                continue
            frac = k_p / n_c
            ratio = frac / baseline[go] if baseline[go] > 0 else 0
            if ratio > best_ratio:
                best_ratio = ratio
                best_c = c
        FEATURED[go]["cluster"] = int(best_c) if best_c is not None else -1
        FEATURED[go]["ratio"] = float(best_ratio)
        FEATURED[go]["n_members"] = int((cid == best_c).sum()) if best_c is not None else 0

    # ---- rendering ----
    print(f"\n[{time.time()-t0:5.1f}s] rendering fixed master figure (2 panels)")

    # bundle-specific palette (tab10)
    BUNDLE_COLORS = plt.cm.tab10(np.linspace(0.0, 0.95, len(FEATURED)))

    # background subsample idx
    N_BG = 400
    N_SAMPLE = 25
    rng2 = np.random.default_rng(SEED)
    bg_idx = rng2.choice(N, size=N_BG, replace=False)

    fig = plt.figure(figsize=(22, 11))
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    axB = fig.add_subplot(1, 2, 2, projection="3d")
    for ax in (axA, axB):
        ax.set_facecolor("white")

    # ---- (1) background on both panels
    for ax in (axA, axB):
        for i in bg_idx:
            pts = coords[i]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color="lightgray", lw=0.35, alpha=0.20)

    # helper: layer index annotation along a reference trajectory
    LAYER_LABELS = [1, 5, 10, 15, 20, 25, 30]

    for k, (go, spec) in enumerate(FEATURED.items()):
        c = spec["cluster"]
        if c < 0:
            continue
        # GO+ members of the winner cluster
        cluster_mask  = (cid == c)
        go_pos_mask   = y[go]
        good_mask     = cluster_mask & go_pos_mask
        good_idx      = np.where(good_mask)[0]
        if len(good_idx) < 5:
            good_idx = np.where(cluster_mask)[0]
        chosen = rng2.choice(good_idx,
                             size=min(N_SAMPLE, len(good_idx)),
                             replace=False)
        bundle_mean = coords[cluster_mask].mean(axis=0)
        col = BUNDLE_COLORS[k]

        # -- panel A: bundle-colored trajectories --
        for i in chosen:
            pts = coords[i]
            axA.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                     color=col, lw=1.1, alpha=0.55)
        # bundle mean thick
        axA.plot(bundle_mean[:, 0], bundle_mean[:, 1], bundle_mean[:, 2],
                 color=col, lw=3.0, alpha=1.0,
                 label=(f"{go} {spec['name']}\n"
                        f"peakL={spec['observed_peakL']}  "
                        f"c{c} n={spec['n_members']} "
                        f"({spec['ratio']:.1f}×)"))
        # L1 marker (circle)
        axA.scatter(bundle_mean[0, 0], bundle_mean[0, 1], bundle_mean[0, 2],
                    color=col, marker="o", s=110, edgecolors="black",
                    linewidths=1.5, zorder=6)
        # L30 marker (X)
        axA.scatter(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                    color=col, marker="X", s=160, edgecolors="black",
                    linewidths=1.5, zorder=6)
        # layer index annotations on this bundle's mean trajectory
        for L in LAYER_LABELS:
            j = L - 1
            axA.text(bundle_mean[j, 0], bundle_mean[j, 1], bundle_mean[j, 2],
                     f" L{L}", fontsize=6.5, color="black", alpha=0.7)

        # -- panel B: layer-signal-colored trajectories --
        for i in chosen:
            pts = coords[i]
            lc = colored_line3d(pts, spec["curve_norm"],
                                cmap=plt.cm.cividis, lw=1.2, alpha=0.75)
            axB.add_collection3d(lc)
        axB.plot(bundle_mean[:, 0], bundle_mean[:, 1], bundle_mean[:, 2],
                 color="black", lw=2.4, alpha=0.9)
        axB.scatter(bundle_mean[0, 0], bundle_mean[0, 1], bundle_mean[0, 2],
                    color="black", marker="o", s=90, edgecolors="white",
                    linewidths=1.2, zorder=6)
        axB.scatter(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                    color="black", marker="X", s=140, edgecolors="white",
                    linewidths=1.5, zorder=6)
        # bundle labels near L30 with box
        axB.text(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                 f" {spec['name']}\n peakL={spec['observed_peakL']}",
                 fontsize=8, color="black",
                 bbox=dict(facecolor="white", edgecolor="black",
                           boxstyle="round,pad=0.15", alpha=0.85))
        for L in LAYER_LABELS:
            j = L - 1
            axB.text(bundle_mean[j, 0], bundle_mean[j, 1], bundle_mean[j, 2],
                     f" L{L}", fontsize=6.5, color="black", alpha=0.6)

    axA.set_xlabel("PC1 (layer-norm joint PCA)")
    axA.set_ylabel("PC2")
    axA.set_zlabel("PC3")
    axA.set_title("(A) Bundle-coloured trajectories\n"
                  "○ = L1 start, X = L30 end")
    axA.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.05))

    axB.set_xlabel("PC1"); axB.set_ylabel("PC2"); axB.set_zlabel("PC3")
    axB.set_title("(B) Layer-signal colouring (per-GO Fisher normalized)\n"
                  "blue = low signal, yellow = critical layer")

    # -- colorbar for panel B --
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis,
                               norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axB, shrink=0.6, pad=0.05,
                      orientation="vertical")
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0 (min)", "0.25", "0.5", "0.75", "1 (peak)"])
    cb.set_label("per-layer Fisher signal (per-GO normalized)")

    fig.suptitle(
        "Master fluid trajectory — layer-normalized PCA coords.\n"
        "Panel A: each of 6 GO bundles gets a distinct colour, L1 (○) and "
        "L30 (X) markers per bundle mean; layer indices L1/5/10/15/20/25/30 "
        "annotated.\n"
        "Panel B: same trajectories with per-GO Fisher signal colouring — "
        "yellow band marks the critical layer for each GO."
    , fontsize=11)
    fig.tight_layout(rect=[0, 0.01, 1, 0.94])

    ts = time.strftime("%Y%m%d_%H%M")
    fig.savefig(f"{OUT_DIR}/master_trajectory_v2_{ts}.png", dpi=180)
    plt.close(fig)
    print(f"saved: {OUT_DIR}/master_trajectory_v2_{ts}.png")

    # ---- per-GO grid with bundle color + Fisher color side-by-side ----
    print(f"[{time.time()-t0:5.1f}s] rendering per-GO panel v2 grid")
    fig, axs = plt.subplots(2, 3, figsize=(19, 11),
                            subplot_kw={"projection": "3d"})
    axs = axs.ravel()
    for k, (go, spec) in enumerate(FEATURED.items()):
        ax = axs[k]
        col = BUNDLE_COLORS[k]
        for i in bg_idx[:150]:
            pts = coords[i]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color="lightgray", lw=0.3, alpha=0.25)
        c = spec["cluster"]
        if c < 0:
            ax.set_title(f"{go} — no winner cluster")
            continue
        cluster_mask = (cid == c)
        good_idx = np.where(cluster_mask & y[go])[0]
        if len(good_idx) < 5:
            good_idx = np.where(cluster_mask)[0]
        chosen = rng2.choice(good_idx,
                             size=min(N_SAMPLE, len(good_idx)),
                             replace=False)
        for i in chosen:
            pts = coords[i]
            lc = colored_line3d(pts, spec["curve_norm"],
                                cmap=plt.cm.cividis, lw=1.4, alpha=0.85)
            ax.add_collection3d(lc)
        bundle_mean = coords[cluster_mask].mean(axis=0)
        ax.plot(bundle_mean[:, 0], bundle_mean[:, 1], bundle_mean[:, 2],
                color=col, lw=2.6, alpha=0.95)
        ax.scatter(bundle_mean[0, 0], bundle_mean[0, 1], bundle_mean[0, 2],
                   color=col, marker="o", s=100,
                   edgecolors="black", linewidths=1.2, zorder=6)
        ax.text(bundle_mean[0, 0], bundle_mean[0, 1], bundle_mean[0, 2],
                "  L1", fontsize=8, color="black")
        ax.scatter(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                   color=col, marker="X", s=150,
                   edgecolors="black", linewidths=1.5, zorder=6)
        ax.text(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                "  L30", fontsize=8, color="black")
        for L in [10, 20]:
            j = L - 1
            ax.scatter(bundle_mean[j, 0], bundle_mean[j, 1], bundle_mean[j, 2],
                       color=col, marker="s", s=40,
                       edgecolors="black", linewidths=1.0, zorder=6)
            ax.text(bundle_mean[j, 0], bundle_mean[j, 1], bundle_mean[j, 2],
                    f"  L{L}", fontsize=7, color="black")
        # highlight peakL with star
        pl = spec["observed_peakL"] - 1
        ax.scatter(bundle_mean[pl, 0], bundle_mean[pl, 1], bundle_mean[pl, 2],
                   color="red", marker="*", s=180, edgecolors="black",
                   linewidths=1.2, zorder=7)
        ax.text(bundle_mean[pl, 0], bundle_mean[pl, 1], bundle_mean[pl, 2],
                f"  peakL={spec['observed_peakL']}",
                fontsize=8, color="red", weight="bold")
        ax.set_title(f"{go}: {spec['name']}\n"
                     f"cluster c{c}  ratio={spec['ratio']:.2f}×  "
                     f"n={spec['n_members']}", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

    # add a single shared colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis,
                               norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axs.tolist(), shrink=0.55, pad=0.02,
                      orientation="vertical")
    cb.set_label("per-layer Fisher signal (per-GO normalized 0..1)")
    fig.suptitle(
        "Per-GO panels (v2): star = peak layer, ○=L1, □=L10/L20, "
        "X=L30 markers along bundle mean.",
        fontsize=11)
    fig.savefig(f"{OUT_DIR}/master_panels_v2_{ts}.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT_DIR}/master_panels_v2_{ts}.png")
    print(f"total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
