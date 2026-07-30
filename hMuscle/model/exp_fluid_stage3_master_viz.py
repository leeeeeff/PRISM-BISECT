"""
exp_fluid_stage3_master_viz.py
==============================
Master visualization of fluid trajectories for the "typed-GO" set.

Concept
-------
Show every isoform's L1..L30 mean-pooled embedding as a 30-point
polyline in a shared 3D joint-PCA space. Isoforms that belong to a
statistically significant GO bundle get their line coloured per-segment
by the layer-specific Fisher signal for that GO — blue where the GO
receives little differentiation from that layer, yellow where the GO
receives the most.

Result: a single figure showing (a) random L1 dispersion converging
into GO-specific bundles by L30 and (b) the layer of "critical
information intake" for each GO reads off directly as the position of
the yellow hot band along its trajectory.

Design choices
--------------
- 3D projection: top-3 PCs of joint PCA (unnormalized, so L1 vs L30
  natural scale differences reveal convergence)
- Colouring: matplotlib cividis colormap (blue -> yellow, color-vision
  friendly). Normalized per-GO by max so the *peak* layer is always
  yellow.
- Bundles overlaid: 6 curated GOs spanning early/mid/late peak layer.
- Background: subsampled 500 non-bundle isoforms as light gray.
- Bundle mean trajectory drawn as thicker white-outlined black line
  with terminal label.
"""

import os, json, time, gc, glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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

# --- 6 curated GO bundles spanning peak-layer positions ------------------
FEATURED = {
    # (GO, name, target_peak_L for annotation), ordered mid to late
    "GO:0007204": dict(name="Ca2+ signaling",          expected_peakL=11),
    "GO:0006414": dict(name="Translational elongation", expected_peakL=12),
    "GO:0000398": dict(name="mRNA splicing",           expected_peakL=14),
    "GO:0006418": dict(name="tRNA aminoacylation",     expected_peakL=19),
    "GO:0006635": dict(name="FA beta-oxidation",       expected_peakL=28),
    "GO:0006120": dict(name="Complex I NADH ox",       expected_peakL=30),
}

# same 34-GO catalog as stage 2b
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
    p = y_bool
    n = ~y_bool
    if p.sum() < 5 or n.sum() < 5:
        return scores
    for L in range(N_LAYERS):
        pts = traj[:, L, :]
        mu_p = pts[p].mean(axis=0); mu_n = pts[n].mean(axis=0)
        v_p  = pts[p].var(axis=0);  v_n  = pts[n].var(axis=0)
        scores[L] = ((mu_p - mu_n) ** 2).sum() / ((v_p + v_n).sum() + 1e-9)
    return scores


def colored_line3d(pts, values, cmap, lw=1.4, alpha=0.85):
    """Return Line3DCollection with per-segment color = values midpoint."""
    segments = np.stack([pts[:-1], pts[1:]], axis=1)                # (n-1, 2, 3)
    seg_vals = (values[:-1] + values[1:]) / 2.0                     # (n-1,)
    lc = Line3DCollection(segments, cmap=cmap, linewidths=lw,
                          alpha=alpha, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    lc.set_array(seg_vals)
    return lc


def main():
    t0 = time.time()
    te_iso  = load_ids("my_isoform_list_fixed.npy")
    te_gene = load_ids("my_gene_list_fixed.npy")
    e2s = load_e2s()
    te_sym = [e2s.get(g.split(".")[0], g.split(".")[0]) for g in te_gene]
    print(f"[{time.time()-t0:5.1f}s] N_TE={len(te_iso)}")

    # ---- rebuild pilot subset identical to stage 2b
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

    # per-featured-GO membership
    y = {}
    for go in FEATURED:
        y[go] = np.array([1 if s in go_pos[go] else 0 for s in sub_sym],
                         dtype=bool)

    # ---- trajectory
    print(f"[{time.time()-t0:5.1f}s] building trajectory ...")
    traj = build_trajectory(subset_idx)                              # (N, 30, 640)
    print(f"[{time.time()-t0:5.1f}s] traj {traj.shape}")

    # ---- per-featured GO Fisher signal (using RAW traj, not normalized)
    print(f"[{time.time()-t0:5.1f}s] Fisher signal per featured GO")
    sig = {}
    for go in FEATURED:
        s = per_layer_fisher(traj, y[go])
        sig[go] = s
        peakL = int(s.argmax()) + 1
        FEATURED[go]["observed_peakL"] = peakL
        # normalize to [0,1] for coloring
        FEATURED[go]["curve_norm"] = (s - s.min()) / (s.max() - s.min() + 1e-12)
        print(f"     {go}  {FEATURED[go]['name']:28s}"
              f"  observed_peakL={peakL:>2d}"
              f"  expected={FEATURED[go]['expected_peakL']}")

    # ---- layer-norm PCA for clustering (matches stage 2/2b)
    layer_mean = traj.mean(axis=0)
    layer_std  = traj.std(axis=0) + 1e-6
    traj_norm = (traj - layer_mean) / layer_std
    flat_nm = traj_norm.reshape(N * N_LAYERS, EMB_DIM)
    print(f"[{time.time()-t0:5.1f}s] norm joint PCA (K=16) for clustering")
    pca_nm = PCA(n_components=16, random_state=SEED, svd_solver="randomized")
    reduced_nm = pca_nm.fit_transform(flat_nm).reshape(N, N_LAYERS, 16)
    cv_nm = reduced_nm[:, :, :K_PCA].reshape(N, N_LAYERS * K_PCA)
    del flat_nm
    gc.collect()
    km = KMeans(n_clusters=K_CLUS, n_init=5, random_state=SEED)
    cid = km.fit_predict(cv_nm)
    print(f"[{time.time()-t0:5.1f}s] KMeans done")

    # ---- UNNORM joint PCA for visualization coords
    flat_un = traj.reshape(N * N_LAYERS, EMB_DIM)
    print(f"[{time.time()-t0:5.1f}s] unnorm joint PCA (K=3) for viz coords")
    pca_un = PCA(n_components=3, random_state=SEED, svd_solver="randomized")
    coords = pca_un.fit_transform(flat_un).reshape(N, N_LAYERS, 3)
    del flat_un, traj
    gc.collect()
    print(f"          expl_var(K=3)={pca_un.explained_variance_ratio_.sum():.3f}")

    # ---- per-featured GO: find winner cluster (highest ratio)
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
        print(f"  {go}  cluster={FEATURED[go]['cluster']:>3d}"
              f"  ratio={best_ratio:.2f}"
              f"  n_members={FEATURED[go]['n_members']}")

    # ---- MASTER FIGURE ----
    print(f"\n[{time.time()-t0:5.1f}s] rendering master figure")
    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # (1) background — 500 random isoforms as gray polylines
    N_BG = 500
    rng2 = np.random.default_rng(SEED)
    bg_idx = rng2.choice(N, size=N_BG, replace=False)
    for i in bg_idx:
        pts = coords[i]                                              # (30, 3)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color="lightgray", lw=0.35, alpha=0.28)

    # (2) featured bundles
    N_SAMPLE_PER_BUNDLE = 25
    cmap = plt.cm.cividis                                            # blue -> yellow
    label_positions = []
    for go, spec in FEATURED.items():
        c = spec["cluster"]
        if c < 0:
            continue
        member_idx = np.where(cid == c)[0]
        chosen = rng2.choice(member_idx,
                             size=min(N_SAMPLE_PER_BUNDLE, len(member_idx)),
                             replace=False)
        curve_norm = spec["curve_norm"]                              # (30,)
        for i in chosen:
            pts = coords[i]
            lc = colored_line3d(pts, curve_norm, cmap=cmap,
                                lw=1.6, alpha=0.75)
            ax.add_collection3d(lc)

        # bundle mean trajectory in solid dark
        bundle_mean = coords[member_idx].mean(axis=0)
        ax.plot(bundle_mean[:, 0], bundle_mean[:, 1], bundle_mean[:, 2],
                color="black", lw=2.4, alpha=0.9)
        # start / end markers
        ax.scatter(bundle_mean[0, 0], bundle_mean[0, 1], bundle_mean[0, 2],
                   color="black", marker="o", s=50, edgecolors="white",
                   linewidths=1.0, zorder=5)
        ax.scatter(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                   color="black", marker="X", s=110, edgecolors="white",
                   linewidths=1.5, zorder=5)
        # text label near L30
        label = (f"{spec['name']}\n"
                 f"peakL={spec['observed_peakL']}  "
                 f"cluster c{c}  n={spec['n_members']}\n"
                 f"ratio={spec['ratio']:.2f}×")
        ax.text(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                label, fontsize=8, color="black",
                bbox=dict(facecolor="white", edgecolor="black",
                          boxstyle="round,pad=0.2", alpha=0.85))
        label_positions.append(bundle_mean[-1])

    ax.set_xlabel("PC1 (joint 30-layer PCA, unnormalized)")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(
        "Fluid trajectory master view — every isoform's L1..L30 path in "
        "shared 3D PCA\n"
        "Background gray = random 500 non-bundle isoforms; "
        "coloured lines = 6 GO-significant bundles.\n"
        "Line colour along a trajectory (blue → yellow) = per-layer Fisher "
        "signal for that GO (yellow = critical-information layer)."
    )

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08,
                      orientation="vertical")
    cb.set_label("per-layer Fisher signal (per-GO normalized 0=low, 1=peak)")

    fig.tight_layout()
    ts = time.strftime("%Y%m%d_%H%M")
    fig.savefig(f"{OUT_DIR}/master_trajectory_{ts}.png", dpi=180)
    plt.close(fig)
    print(f"saved: {OUT_DIR}/master_trajectory_{ts}.png")

    # ---- companion small-multiples: one panel per featured GO ----
    print(f"[{time.time()-t0:5.1f}s] rendering per-GO panel grid")
    fig, axs = plt.subplots(2, 3, figsize=(18, 11),
                            subplot_kw={"projection": "3d"})
    axs = axs.ravel()
    for k, (go, spec) in enumerate(FEATURED.items()):
        ax = axs[k]
        # background: 200 random iso
        for i in bg_idx[:200]:
            pts = coords[i]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color="lightgray", lw=0.35, alpha=0.28)
        c = spec["cluster"]
        if c < 0:
            ax.set_title(f"{go} — no winner cluster")
            continue
        member_idx = np.where(cid == c)[0]
        chosen = rng2.choice(member_idx,
                             size=min(N_SAMPLE_PER_BUNDLE, len(member_idx)),
                             replace=False)
        for i in chosen:
            pts = coords[i]
            lc = colored_line3d(pts, spec["curve_norm"], cmap=cmap,
                                lw=1.6, alpha=0.85)
            ax.add_collection3d(lc)
        bundle_mean = coords[member_idx].mean(axis=0)
        ax.plot(bundle_mean[:, 0], bundle_mean[:, 1], bundle_mean[:, 2],
                color="black", lw=2.2, alpha=0.9)
        ax.scatter(bundle_mean[0, 0], bundle_mean[0, 1], bundle_mean[0, 2],
                   color="black", marker="o", s=40)
        ax.scatter(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                   color="black", marker="X", s=90)
        ax.set_title(f"{go}: {spec['name']}\n"
                     f"peakL={spec['observed_peakL']}  "
                     f"n_members={spec['n_members']}  "
                     f"ratio={spec['ratio']:.2f}×", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

    fig.suptitle("Per-GO bundle panels — line color intensity marks the "
                 "critical layer where the GO signal peaks", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/master_panels_{ts}.png", dpi=170)
    plt.close(fig)
    print(f"saved: {OUT_DIR}/master_panels_{ts}.png")

    with open(f"{OUT_DIR}/master_viz_{ts}.json", "w") as f:
        json.dump({go: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in spec.items()}
                   for go, spec in FEATURED.items()}, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, np.integer) else o)
    print(f"total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
