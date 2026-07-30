"""
exp_fluid_stage3d_panels_annotated.py
=====================================
Enhanced per-GO panel grid (successor to master_panels_20260706_2021).

Adds:
- Layer index markers L1/L5/L10/L15/L20/L25/L30 with text along the
  bundle mean trajectory (like alt_C figure).
- Red star at peakL, with a short annotation describing the semantic
  role of that layer group in ESM-2:

      L1-6   : raw sequence composition, amino-acid identity
      L7-12  : local motifs & sequence periodicity
      L13-18 : secondary structure, short-range contacts
      L19-24 : domain identity, tertiary contacts
      L25-30 : functional / semantic / GO-level information

  (Consensus from ESM-2 layer probing studies — Rives et al. 2021,
   Meier et al. 2021, and follow-up interpretability papers.)

- Prominent shared colorbar with explicit tick labels (0/0.25/0.5/0.75/1)
  labelled "per-layer Fisher signal (per-GO normalized)".
- Bottom horizontal legend strip showing the 5 layer groups with their
  semantic roles so the reader can decode any peakL directly.
- Uses UNNORMALIZED joint PCA (top-3 = 86% var) for the coordinate
  system to match master_trajectory_20260706_2021.png.
"""

import os, time, gc, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = "../data"
ANNOT_FILE = "../data/raw_data/data/annotations/human_annotations_unified_bp.txt"
ID_DIR     = "../data/raw_data/data/id_lists"
OUT_DIR    = "../../reports/fluid_stage3"

N_LAYERS = 30
EMB_DIM  = 640
SEED     = 42
K_PCA    = 8
K_CLUS   = 16

# Semantic layer groups (ESM-2 layer probing consensus)
LAYER_GROUPS = [
    (1, 6,   "L1-6",   "raw sequence composition, AA identity",       "#FEE9B4"),
    (7, 12,  "L7-12",  "local motifs, k-mer periodicity",             "#F9C270"),
    (13, 18, "L13-18", "secondary structure, short contacts",         "#F19A46"),
    (19, 24, "L19-24", "domain identity, tertiary contacts",          "#D96F30"),
    (25, 30, "L25-30", "functional / semantic / GO-level",             "#A6431F"),
]


def layer_group_name(L):
    for lo, hi, tag, meaning, _c in LAYER_GROUPS:
        if lo <= L <= hi:
            return tag, meaning
    return "?", "?"


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
    s = np.zeros(N_LAYERS)
    p = y_bool; n = ~y_bool
    if p.sum() < 5 or n.sum() < 5:
        return s
    for L in range(N_LAYERS):
        pts = traj[:, L, :]
        mu_p = pts[p].mean(axis=0); mu_n = pts[n].mean(axis=0)
        v_p  = pts[p].var(axis=0);  v_n  = pts[n].var(axis=0)
        s[L] = ((mu_p - mu_n) ** 2).sum() / ((v_p + v_n).sum() + 1e-9)
    return s


def colored_line3d(pts, values, cmap, lw=1.5, alpha=0.85):
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_vals = (values[:-1] + values[1:]) / 2.0
    lc = Line3DCollection(segs, cmap=cmap, linewidths=lw, alpha=alpha,
                          norm=plt.Normalize(vmin=0.0, vmax=1.0))
    lc.set_array(seg_vals)
    return lc


def main():
    t0 = time.time()
    te_iso  = load_ids("my_isoform_list_fixed.npy")
    te_gene = load_ids("my_gene_list_fixed.npy")
    e2s = load_e2s()
    te_sym = [e2s.get(g.split(".")[0], g.split(".")[0]) for g in te_gene]

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

    for go in FEATURED:
        s = per_layer_fisher(traj, y[go])
        FEATURED[go]["observed_peakL"] = int(s.argmax()) + 1
        FEATURED[go]["curve_norm"] = (s - s.min()) / (s.max() - s.min() + 1e-12)

    # unnorm PCA for viz coords (matches v1 file convention)
    flat_un = traj.reshape(N * N_LAYERS, EMB_DIM)
    pca_un = PCA(n_components=3, random_state=SEED, svd_solver="randomized")
    coords_un = pca_un.fit_transform(flat_un).reshape(N, N_LAYERS, 3)
    print(f"[{time.time()-t0:5.1f}s] unnorm top3 var="
          f"{pca_un.explained_variance_ratio_.sum():.3f}")

    # norm PCA + KMeans (identifies bundles same as stage 2)
    layer_mean = traj.mean(axis=0); layer_std = traj.std(axis=0) + 1e-6
    traj_norm = (traj - layer_mean) / layer_std
    del traj, flat_un
    gc.collect()
    flat_nm = traj_norm.reshape(N * N_LAYERS, EMB_DIM)
    pca_nm = PCA(n_components=16, random_state=SEED, svd_solver="randomized")
    red_nm = pca_nm.fit_transform(flat_nm).reshape(N, N_LAYERS, 16)
    del flat_nm
    gc.collect()
    cv_nm = red_nm[:, :, :K_PCA].reshape(N, N_LAYERS * K_PCA)
    km = KMeans(n_clusters=K_CLUS, n_init=5, random_state=SEED)
    cid = km.fit_predict(cv_nm)

    baseline = {go: float(y[go].mean()) for go in FEATURED}
    for go in FEATURED:
        best_c = None; best_ratio = 0
        for c in range(K_CLUS):
            m = (cid == c)
            n_c = int(m.sum())
            if n_c < 10: continue
            k_p = int(y[go][m].sum())
            if k_p < 5: continue
            frac = k_p / n_c
            ratio = frac / baseline[go] if baseline[go] > 0 else 0
            if ratio > best_ratio:
                best_ratio = ratio; best_c = c
        FEATURED[go]["cluster"] = int(best_c) if best_c is not None else -1
        FEATURED[go]["ratio"] = float(best_ratio)
        FEATURED[go]["n_members"] = int((cid == best_c).sum()) if best_c is not None else 0

    # ---- render enhanced panels ----
    BUNDLE_COLORS = plt.cm.tab10(np.linspace(0.0, 0.95, len(FEATURED)))
    LAYER_LABELS = [1, 5, 10, 15, 20, 25, 30]
    N_BG = 150
    N_SAMPLE = 25
    rng2 = np.random.default_rng(SEED)
    bg_idx = rng2.choice(N, size=N_BG, replace=False)

    print(f"\n[{time.time()-t0:5.1f}s] rendering annotated panels ...")
    # 2x3 grid + wide colorbar column + bottom legend
    fig = plt.figure(figsize=(21, 13))
    gs = fig.add_gridspec(3, 4,
                          height_ratios=[5, 5, 0.9],
                          width_ratios=[1, 1, 1, 0.06],
                          hspace=0.25, wspace=0.15)

    ax_axes = []
    for r in range(2):
        for c in range(3):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            ax_axes.append(ax)

    for k, (go, spec) in enumerate(FEATURED.items()):
        ax = ax_axes[k]
        # background
        for i in bg_idx:
            pts = coords_un[i]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color="lightgray", lw=0.3, alpha=0.22)

        c = spec["cluster"]
        if c < 0:
            ax.set_title(f"{go} — no winner cluster")
            continue
        cluster_mask = (cid == c)
        good_idx = np.where(cluster_mask & y[go])[0]
        if len(good_idx) < 5:
            good_idx = np.where(cluster_mask)[0]
        chosen = np.random.default_rng(SEED + k).choice(
            good_idx, size=min(N_SAMPLE, len(good_idx)), replace=False)

        # trajectories with Fisher color
        for i in chosen:
            pts = coords_un[i]
            lc = colored_line3d(pts, spec["curve_norm"],
                                cmap=plt.cm.cividis, lw=1.4, alpha=0.85)
            ax.add_collection3d(lc)

        # bundle mean thick line + markers
        bundle_mean = coords_un[cluster_mask].mean(axis=0)
        col = BUNDLE_COLORS[k]
        ax.plot(bundle_mean[:, 0], bundle_mean[:, 1], bundle_mean[:, 2],
                color=col, lw=2.6, alpha=0.95)

        # L1 marker (circle)
        ax.scatter(bundle_mean[0, 0], bundle_mean[0, 1], bundle_mean[0, 2],
                   color=col, marker="o", s=110,
                   edgecolors="black", linewidths=1.4, zorder=6)
        ax.text(bundle_mean[0, 0], bundle_mean[0, 1], bundle_mean[0, 2],
                "  L1", fontsize=8.5, color="black", weight="bold")

        # intermediate layer markers
        for L in [5, 10, 15, 20, 25]:
            j = L - 1
            ax.scatter(bundle_mean[j, 0], bundle_mean[j, 1], bundle_mean[j, 2],
                       color=col, marker="s", s=42,
                       edgecolors="black", linewidths=1.0, zorder=6)
            ax.text(bundle_mean[j, 0], bundle_mean[j, 1], bundle_mean[j, 2],
                    f"  L{L}", fontsize=7.2, color="black", alpha=0.85)

        # L30 marker (X)
        ax.scatter(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                   color=col, marker="X", s=150,
                   edgecolors="black", linewidths=1.4, zorder=6)
        ax.text(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                "  L30", fontsize=8.5, color="black", weight="bold")

        # peakL red star + semantic annotation
        pl = spec["observed_peakL"] - 1
        tag, meaning = layer_group_name(spec["observed_peakL"])
        ax.scatter(bundle_mean[pl, 0], bundle_mean[pl, 1], bundle_mean[pl, 2],
                   color="red", marker="*", s=210,
                   edgecolors="black", linewidths=1.3, zorder=8)
        ax.text(bundle_mean[pl, 0], bundle_mean[pl, 1], bundle_mean[pl, 2],
                f"  ★ peakL={spec['observed_peakL']}\n"
                f"  [{tag}]\n"
                f"  {meaning}",
                fontsize=7.5, color="red", weight="bold",
                bbox=dict(facecolor="white", edgecolor="red",
                          boxstyle="round,pad=0.2", alpha=0.9))

        ax.set_title(f"{go}: {spec['name']}\n"
                     f"cluster c{c}  ratio={spec['ratio']:.2f}×  "
                     f"n={spec['n_members']}", fontsize=10)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.set_zlabel("PC3", fontsize=8)

    # ---- shared colorbar in gs[:, 3]
    ax_cb = fig.add_subplot(gs[:2, 3])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis,
                               norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation="vertical")
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0 (min)", "0.25", "0.5", "0.75", "1 (peak layer)"])
    cb.set_label("per-layer Fisher signal (per-GO normalized 0..1)\n"
                 "blue → yellow gradient = "
                 "low → critical layer contribution",
                 fontsize=9)

    # ---- bottom semantic legend strip
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.set_xlim(0, 30)
    ax_leg.set_ylim(0, 1)
    ax_leg.set_yticks([])
    ax_leg.set_xticks(range(1, 31))
    ax_leg.set_xticklabels([str(i) if i in [1, 5, 10, 15, 20, 25, 30]
                            else "" for i in range(1, 31)],
                           fontsize=8)
    ax_leg.set_xlabel("ESM-2 layer L", fontsize=9)
    ax_leg.set_title("Layer semantic reference (ESM-2 layer probing "
                     "consensus)",
                     fontsize=10, loc="left")

    for lo, hi, tag, meaning, color in LAYER_GROUPS:
        rect = Rectangle((lo - 0.5, 0.15), hi - lo + 1, 0.70,
                         facecolor=color, edgecolor="black", alpha=0.85,
                         linewidth=1.0)
        ax_leg.add_patch(rect)
        ax_leg.text((lo + hi) / 2.0, 0.55, tag, ha="center", va="center",
                    fontsize=10, weight="bold", color="black")
        ax_leg.text((lo + hi) / 2.0, 0.28, meaning, ha="center", va="center",
                    fontsize=7.5, color="black", style="italic")

    fig.suptitle("Per-GO fluid trajectory panels (annotated) — "
                 "unnormalized joint PCA-3D (86% var).\n"
                 "○=L1 start, □=L5/10/15/20/25, X=L30, "
                 "★=peakL with semantic layer group annotation.",
                 fontsize=12)

    ts = time.strftime("%Y%m%d_%H%M")
    fig.savefig(f"{OUT_DIR}/master_panels_annotated_{ts}.png",
                dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT_DIR}/master_panels_annotated_{ts}.png")
    print(f"total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
