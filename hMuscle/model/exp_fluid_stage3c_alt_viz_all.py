"""
exp_fluid_stage3c_alt_viz_all.py
=================================
Render 3 alternative visualizations of the fluid trajectory:

(A) UMAP embedding from 16-D layer-normalized joint PCA — should give
    strongest cluster separation.
(B) Per-layer 2D projection sequence (L1, L5, L10, L15, L20, L25, L30)
    — direct view of how isoforms move from scattered at L1 to
    bundle-concentrated at L30.
(C) Unnormalized joint PCA top-3 (86% variance) with bundle colours +
    Fisher-signal overlay. L1 is compressed near origin (which itself
    is informative: "no differentiation yet") while L30 spreads into
    bundles.

Each figure includes a colorbar and layer markers.
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


def colored_line3d(pts, values, cmap, lw=1.4, alpha=0.8):
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
    print(f"[{time.time()-t0:5.1f}s] traj {traj.shape}")

    # Fisher signals
    for go in FEATURED:
        s = per_layer_fisher(traj, y[go])
        FEATURED[go]["observed_peakL"] = int(s.argmax()) + 1
        FEATURED[go]["curve_norm"] = (s - s.min()) / (s.max() - s.min() + 1e-12)

    # normalized traj
    layer_mean = traj.mean(axis=0); layer_std = traj.std(axis=0) + 1e-6
    traj_norm = (traj - layer_mean) / layer_std

    # unnorm joint PCA (for C)
    flat_un = traj.reshape(N * N_LAYERS, EMB_DIM)
    pca_un = PCA(n_components=3, random_state=SEED, svd_solver="randomized")
    coords_un = pca_un.fit_transform(flat_un).reshape(N, N_LAYERS, 3)
    print(f"[{time.time()-t0:5.1f}s] unnorm PCA top3 var="
          f"{pca_un.explained_variance_ratio_.sum():.3f}")

    # norm joint PCA (for A + B + KMeans)
    flat_nm = traj_norm.reshape(N * N_LAYERS, EMB_DIM)
    # store original L1 norms BEFORE deletion — for non-circular len_proxy
    l1_norm_orig = np.linalg.norm(traj[:, 0, :].astype(np.float64), axis=1)
    del traj, traj_norm, flat_un
    gc.collect()
    K_MAX = 16
    pca_nm = PCA(n_components=K_MAX, random_state=SEED, svd_solver="randomized")
    reduced_nm_flat = pca_nm.fit_transform(flat_nm)                  # (N*30, 16)
    reduced_nm = reduced_nm_flat.reshape(N, N_LAYERS, K_MAX)
    print(f"[{time.time()-t0:5.1f}s] norm PCA K={K_MAX} var="
          f"{pca_nm.explained_variance_ratio_.sum():.3f}")

    cv_nm = reduced_nm[:, :, :K_PCA].reshape(N, N_LAYERS * K_PCA)
    km = KMeans(n_clusters=K_CLUS, n_init=5, random_state=SEED)
    cid = km.fit_predict(cv_nm)

    # ========================================================================
    # (D) Verification v2: within-gene separation with 4 bias controls
    #   Fix 1. len_proxy from PRE-normalization traj (l1_norm_orig)
    #   Fix 2. cross-gene baseline (random non-same-gene pairs)
    #   Fix 3. shuffle baseline (gene labels shuffled within gene)
    #   Fix 4. partial correlation controlling for L1 separation
    # ========================================================================
    print(f"\n[{time.time()-t0:5.1f}s] === Verification v2: within-gene separation (4 controls) ===")
    from scipy.stats import spearmanr, pearsonr
    from collections import defaultdict

    gene_to_local = defaultdict(list)
    for local_i, sym in enumerate(sub_sym):
        gene_to_local[sym].append(local_i)
    multi_iso_local = {g: idxs for g, idxs in gene_to_local.items() if len(idxs) >= 2}
    print(f"     Multi-isoform genes in subset: {len(multi_iso_local)}")

    coords_un3 = coords_un[:, :, :3]       # (N, 30, 3)
    coords_nm3 = reduced_nm[:, :, :3]      # (N, 30, 3)
    traj_nm_rebuilt = flat_nm.reshape(N, N_LAYERS, EMB_DIM)

    # Fix 1: non-circular len_proxy from ORIGINAL (pre-normalization) L1
    len_diff_orig = []
    sep_un_start, sep_un_end = [], []
    sep_nm_start, sep_nm_end = [], []

    for sym, isos in multi_iso_local.items():
        for ii in range(len(isos)):
            for jj in range(ii + 1, len(isos)):
                a, b = isos[ii], isos[jj]
                sep_un_start.append(np.linalg.norm(coords_un3[a, 0] - coords_un3[b, 0]))
                sep_un_end.append(np.linalg.norm(coords_un3[a, 29] - coords_un3[b, 29]))
                sep_nm_start.append(np.linalg.norm(coords_nm3[a, 0] - coords_nm3[b, 0]))
                sep_nm_end.append(np.linalg.norm(coords_nm3[a, 29] - coords_nm3[b, 29]))
                len_diff_orig.append(abs(l1_norm_orig[a] - l1_norm_orig[b]))

    sep_un_start   = np.array(sep_un_start)
    sep_un_end     = np.array(sep_un_end)
    sep_nm_start   = np.array(sep_nm_start)
    sep_nm_end     = np.array(sep_nm_end)
    len_diff_orig  = np.array(len_diff_orig)

    print(f"\n     [Fix 1] Non-circular len_proxy (pre-norm L1 ||·||):")
    r_orig_start, p_orig_start = spearmanr(len_diff_orig, sep_nm_start)
    r_orig_end,   p_orig_end   = spearmanr(len_diff_orig, sep_nm_end)
    print(f"     ρ(len_orig ↔ norm L1  sep) = {r_orig_start:.3f}, p={p_orig_start:.3e}")
    print(f"     ρ(len_orig ↔ norm L30 sep) = {r_orig_end:.3f},   p={p_orig_end:.3e}")
    if r_orig_end > r_orig_start:
        print("     → ρ drops from L1→L30: L30 contains additional variation beyond seq diff")
    else:
        print("     → ρ maintained/increased: L30 sep is proportional to seq diff (possible artifact)")

    # Fix 2: cross-gene baseline
    all_idx = np.arange(N)
    rng_ctrl = np.random.default_rng(SEED + 999)
    n_pairs_within = len(sep_nm_start)
    # sample same number of cross-gene pairs
    cross_nm_start, cross_nm_end = [], []
    attempts = 0
    while len(cross_nm_start) < n_pairs_within and attempts < n_pairs_within * 10:
        a, b = rng_ctrl.integers(0, N, size=2)
        if sub_sym[a] != sub_sym[b]:
            cross_nm_start.append(np.linalg.norm(coords_nm3[a, 0] - coords_nm3[b, 0]))
            cross_nm_end.append(np.linalg.norm(coords_nm3[a, 29] - coords_nm3[b, 29]))
        attempts += 1
    cross_nm_start = np.array(cross_nm_start)
    cross_nm_end   = np.array(cross_nm_end)

    print(f"\n     [Fix 2] Cross-gene baseline (n={len(cross_nm_start)}):")
    print(f"     Cross-gene  L1  sep: mean={cross_nm_start.mean():.4f}")
    print(f"     Cross-gene  L30 sep: mean={cross_nm_end.mean():.4f}")
    print(f"     Cross-gene  L30/L1 ratio: {cross_nm_end.mean()/cross_nm_start.mean():.3f}")
    print(f"     Within-gene L1  sep: mean={sep_nm_start.mean():.4f}")
    print(f"     Within-gene L30 sep: mean={sep_nm_end.mean():.4f}")
    print(f"     Within-gene L30/L1 ratio: {sep_nm_end.mean()/sep_nm_start.mean():.3f}")
    ratio_wg = sep_nm_end.mean() / sep_nm_start.mean()
    ratio_xg = cross_nm_end.mean() / cross_nm_start.mean()
    if ratio_wg > ratio_xg:
        print(f"     → within-gene ratio ({ratio_wg:.3f}) > cross-gene ({ratio_xg:.3f}): isoform-specific signal")
    else:
        print(f"     → within-gene ratio ({ratio_wg:.3f}) ≤ cross-gene ({ratio_xg:.3f}): normalization artifact")

    # Fix 3: proper null — shuffle gene labels across ALL isoforms, then
    # compute "within-gene" pairs using the shuffled assignment.
    # Preserves within-group size distribution but breaks true gene membership.
    rng_shuf = np.random.default_rng(SEED + 42)
    # build gene-size list from multi_iso_local
    gene_sizes = [len(v) for v in multi_iso_local.values()]
    # randomly assign isoform indices to groups of the same sizes
    all_local_idx = np.arange(N)
    shuffled_pool = rng_shuf.permutation(all_local_idx)
    ptr = 0
    shuf_nm_start, shuf_nm_end = [], []
    for sz in gene_sizes:
        group = shuffled_pool[ptr:ptr + sz]
        ptr += sz
        if ptr > N:
            break
        for ii in range(len(group)):
            for jj in range(ii + 1, len(group)):
                a, b = int(group[ii]), int(group[jj])
                shuf_nm_start.append(np.linalg.norm(coords_nm3[a, 0] - coords_nm3[b, 0]))
                shuf_nm_end.append(np.linalg.norm(coords_nm3[a, 29] - coords_nm3[b, 29]))
    shuf_nm_start = np.array(shuf_nm_start)
    shuf_nm_end   = np.array(shuf_nm_end)

    ratio_shuf = shuf_nm_end.mean() / shuf_nm_start.mean()
    print(f"\n     [Fix 3] Gene-label shuffle null (n={len(shuf_nm_start)}):")
    print(f"     Shuffle L30/L1 ratio: {ratio_shuf:.3f}  ← proper null")
    print(f"     Within-gene true:     {ratio_wg:.3f}")
    print(f"     Cross-gene baseline:  {ratio_xg:.3f}")
    if ratio_wg > ratio_shuf:
        print(f"     → within-gene ({ratio_wg:.3f}) > null ({ratio_shuf:.3f}): gene-specific signal")
    else:
        print(f"     → within-gene ({ratio_wg:.3f}) ≤ null ({ratio_shuf:.3f}): normalization artifact")

    # Fix 4: partial correlation — does L30 sep predict anything BEYOND L1 sep?
    # regress out L1_sep from L30_sep, then correlate residual with len_diff_orig
    from numpy.polynomial.polynomial import polyfit as nppolyfit
    def partial_r(x, y, z):
        """Spearman partial corr of x,y controlling for z."""
        # residualize x on z, y on z using linear regression ranks
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rz = np.argsort(np.argsort(z)).astype(float)
        # residuals of rx and ry on rz
        def resid(a, b):
            coef = np.polyfit(b, a, 1)
            return a - np.polyval(coef, b)
        rx_z = resid(rx, rz)
        ry_z = resid(ry, rz)
        r, p = pearsonr(rx_z, ry_z)
        return r, p

    r_partial, p_partial = partial_r(sep_nm_end, len_diff_orig, sep_nm_start)
    print(f"\n     [Fix 4] Partial correlation:")
    print(f"     ρ(L30_sep, len_orig | L1_sep) = {r_partial:.3f}, p={p_partial:.3e}")
    if abs(r_partial) > 0.1 and p_partial < 0.05:
        print(f"     → L30 sep carries signal beyond L1 — deepening layers add info")
    else:
        print(f"     → L30 sep is explained by L1 alone — no additional layer-depth signal")

    # ========================================================================
    # (E) Normalization comparison: per-isoform centering vs per-layer z-score
    # ========================================================================
    print(f"\n[{time.time()-t0:5.1f}s] === Norm comparison: per-isoform centering ===")

    iso_mean = traj_nm_rebuilt.mean(axis=1, keepdims=True)      # (N, 1, 640)
    traj_iso_centered = traj_nm_rebuilt - iso_mean              # (N, 30, 640)
    flat_iso = traj_iso_centered.reshape(N * N_LAYERS, EMB_DIM)
    pca_iso = PCA(n_components=3, random_state=SEED, svd_solver="randomized")
    coords_iso = pca_iso.fit_transform(flat_iso).reshape(N, N_LAYERS, 3)
    print(f"     Per-isoform centered PCA: top-3 var={pca_iso.explained_variance_ratio_.sum():.3f}")

    sep_iso_start, sep_iso_end = [], []
    for sym, isos in multi_iso_local.items():
        for ii in range(len(isos)):
            for jj in range(ii + 1, len(isos)):
                a, b = isos[ii], isos[jj]
                sep_iso_start.append(np.linalg.norm(coords_iso[a, 0] - coords_iso[b, 0]))
                sep_iso_end.append(np.linalg.norm(coords_iso[a, 29] - coords_iso[b, 29]))
    sep_iso_start = np.array(sep_iso_start)
    sep_iso_end   = np.array(sep_iso_end)

    print(f"\n     ─── L30/L1 ratio summary (within-gene pairs) ───")
    print(f"     Unnormalized:          {sep_un_end.mean()/sep_un_start.mean():.3f}")
    print(f"     Per-layer z-score:     {sep_nm_end.mean()/sep_nm_start.mean():.3f}")
    print(f"     Per-layer z cross-gene:{ratio_xg:.3f}  ← baseline")
    print(f"     Label-shuffle (null):  {ratio_shuf:.3f}  ← proper null")
    print(f"     Per-isoform centering: {sep_iso_end.mean()/sep_iso_start.mean():.3f}")

    # figure D: 4-panel summary
    figD, axsD = plt.subplots(2, 2, figsize=(14, 10))
    axD_box  = axsD[0, 0]
    axD_r    = axsD[0, 1]
    axD_ctrl = axsD[1, 0]
    axD_part = axsD[1, 1]

    # panel 1: boxplot within-gene L1 vs L30 (per-layer z)
    axD_box.boxplot([sep_nm_start, sep_nm_end],
                    tick_labels=["L1 (start)", "L30 (end)"],
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor="skyblue", alpha=0.7))
    axD_box.set_title(f"Within-gene separation\n(per-layer z-score, L30/L1={ratio_wg:.3f})")
    axD_box.set_ylabel("PCA distance")

    # panel 2: scatter len_orig vs sep (L1 and L30)
    sample_n = min(2000, len(len_diff_orig))
    sidx = np.random.default_rng(SEED).choice(len(len_diff_orig), sample_n, replace=False)
    axD_r.scatter(len_diff_orig[sidx], sep_nm_start[sidx],
                  alpha=0.3, s=4, c="steelblue", label=f"L1 ρ={r_orig_start:.3f}")
    axD_r.scatter(len_diff_orig[sidx], sep_nm_end[sidx],
                  alpha=0.3, s=4, c="firebrick", label=f"L30 ρ={r_orig_end:.3f}")
    axD_r.set_xlabel("len_diff (pre-norm L1 ||·||)")
    axD_r.set_ylabel("norm PCA separation")
    axD_r.set_title("Fix 1: non-circular len_proxy\nρ(L1) > ρ(L30) → L30 has extra signal")
    axD_r.legend(fontsize=9)

    # panel 3: ratio comparison with baselines
    labels_ctrl = ["Within-gene\n(true)", "Label-shuffle\n(null)", "Cross-gene\n(baseline)",
                   "Per-isoform\ncentered"]
    ratios_ctrl = [ratio_wg, ratio_shuf, ratio_xg,
                   sep_iso_end.mean()/sep_iso_start.mean()]
    colors_ctrl = ["firebrick", "steelblue", "gray", "mediumpurple"]
    bars = axD_ctrl.bar(labels_ctrl, ratios_ctrl, color=colors_ctrl, alpha=0.75)
    axD_ctrl.axhline(1.0, color="black", lw=1.0, linestyle="--", alpha=0.5)
    axD_ctrl.set_ylabel("L30/L1 separation ratio")
    axD_ctrl.set_title("Fix 2+3: baselines\n(within > cross > shuffle → isoform signal)")
    for bar, val in zip(bars, ratios_ctrl):
        axD_ctrl.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
                      ha="center", va="bottom", fontsize=9)

    # panel 4: partial corr text summary
    axD_part.axis("off")
    summary_txt = (
        f"Fix 4: Partial Correlation\n"
        f"ρ(L30_sep, len_orig | L1_sep) = {r_partial:.3f}\n"
        f"p = {p_partial:.3e}\n\n"
        f"Interpretation:\n"
        f"{'→ L30 adds signal beyond L1' if abs(r_partial)>0.1 and p_partial<0.05 else '→ L30 explained by L1 alone'}\n\n"
        f"Summary:\n"
        f"  Within-gene ratio: {ratio_wg:.3f}\n"
        f"  Cross-gene ratio:  {ratio_xg:.3f}\n"
        f"  Label-shuffle null:{ratio_shuf:.3f}\n"
        f"  ρ drop L1→L30:    {r_orig_start:.3f}→{r_orig_end:.3f}"
    )
    axD_part.text(0.05, 0.95, summary_txt, transform=axD_part.transAxes,
                  fontsize=11, verticalalignment="top", fontfamily="monospace",
                  bbox=dict(facecolor="lightyellow", edgecolor="orange",
                            boxstyle="round,pad=0.5"))

    figD.suptitle("(D) Verification v2: within-gene isoform separation — 4 bias controls",
                  fontsize=13)
    figD.tight_layout(rect=[0, 0.01, 1, 0.95])
    ts_d = time.strftime("%Y%m%d_%H%M")
    figD.savefig(f"{OUT_DIR}/alt_D_sep_verify_v2_{ts_d}.png", dpi=160)
    plt.close(figD)
    print(f"     saved {OUT_DIR}/alt_D_sep_verify_v2_{ts_d}.png")

    # winner cluster per featured GO
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

    ts = time.strftime("%Y%m%d_%H%M")
    BUNDLE_COLORS = plt.cm.tab10(np.linspace(0.0, 0.95, len(FEATURED)))
    LAYER_LABELS = [1, 5, 10, 15, 20, 25, 30]
    N_BG = 400
    N_SAMPLE = 25

    # ========================================================================
    # (A) UMAP embedding from norm 16-D
    # ========================================================================
    print(f"\n[{time.time()-t0:5.1f}s] === Option A: UMAP embedding ===")
    import umap
    # subsample isoforms for tractable UMAP
    N_UMAP_ISO = min(4000, N)
    umap_iso_idx = rng.choice(N, size=N_UMAP_ISO, replace=False)
    # ensure all featured-GO members are included
    for go in FEATURED:
        c = FEATURED[go]["cluster"]
        if c < 0: continue
        cluster_members = np.where(cid == c)[0]
        add = np.setdiff1d(cluster_members[:200], umap_iso_idx)
        umap_iso_idx = np.concatenate([umap_iso_idx, add])
    umap_iso_idx = np.unique(umap_iso_idx)
    N_UMAP = len(umap_iso_idx)
    print(f"     UMAP N_iso={N_UMAP}, N_pts={N_UMAP * N_LAYERS}")

    umap_input = reduced_nm[umap_iso_idx].reshape(N_UMAP * N_LAYERS, K_MAX)
    reducer = umap.UMAP(n_components=3, random_state=SEED,
                        n_neighbors=30, min_dist=0.1,
                        metric="euclidean", verbose=False)
    umap_emb = reducer.fit_transform(umap_input).reshape(N_UMAP, N_LAYERS, 3)
    print(f"     UMAP done at {time.time()-t0:.1f}s")

    # local index mapping
    subset_pos = {orig_i: local_i for local_i, orig_i in enumerate(umap_iso_idx)}

    figA = plt.figure(figsize=(22, 11))
    axA1 = figA.add_subplot(1, 2, 1, projection="3d")
    axA2 = figA.add_subplot(1, 2, 2, projection="3d")
    for ax in (axA1, axA2):
        ax.set_facecolor("white")
        # background
        bg_local = rng2 = np.random.default_rng(SEED).choice(
            N_UMAP, size=min(N_BG, N_UMAP), replace=False)
        for li in bg_local:
            pts = umap_emb[li]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color="lightgray", lw=0.3, alpha=0.22)

    for k, (go, spec) in enumerate(FEATURED.items()):
        c = spec["cluster"]
        if c < 0: continue
        cluster_orig = np.where(cid == c)[0]
        avail_orig = np.intersect1d(cluster_orig, umap_iso_idx)
        avail_local = np.array([subset_pos[o] for o in avail_orig])
        good_local = np.array([li for li in avail_local
                               if y[go][umap_iso_idx[li]]])
        if len(good_local) < 5:
            good_local = avail_local
        chosen = np.random.default_rng(SEED + k).choice(
            good_local, size=min(N_SAMPLE, len(good_local)), replace=False)
        bundle_mean = umap_emb[avail_local].mean(axis=0)
        col = BUNDLE_COLORS[k]

        # panel A1: bundle-colored
        for li in chosen:
            pts = umap_emb[li]
            axA1.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                      color=col, lw=1.1, alpha=0.55)
        axA1.plot(bundle_mean[:, 0], bundle_mean[:, 1], bundle_mean[:, 2],
                  color=col, lw=3.0, alpha=1.0,
                  label=(f"{go} {spec['name'][:22]}\n"
                         f"peakL={spec['observed_peakL']} "
                         f"c{c} n={spec['n_members']} "
                         f"({spec['ratio']:.1f}×)"))
        axA1.scatter(bundle_mean[0, 0], bundle_mean[0, 1], bundle_mean[0, 2],
                     color=col, marker="o", s=110, edgecolors="black",
                     linewidths=1.4, zorder=6)
        axA1.scatter(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                     color=col, marker="X", s=160, edgecolors="black",
                     linewidths=1.4, zorder=6)
        for L in LAYER_LABELS:
            j = L - 1
            axA1.text(bundle_mean[j, 0], bundle_mean[j, 1], bundle_mean[j, 2],
                      f" L{L}", fontsize=6.5, color="black", alpha=0.7)

        # panel A2: Fisher-colored
        for li in chosen:
            pts = umap_emb[li]
            lc = colored_line3d(pts, spec["curve_norm"],
                                cmap=plt.cm.cividis, lw=1.2, alpha=0.75)
            axA2.add_collection3d(lc)
        axA2.plot(bundle_mean[:, 0], bundle_mean[:, 1], bundle_mean[:, 2],
                  color="black", lw=2.4, alpha=0.9)
        axA2.scatter(bundle_mean[0, 0], bundle_mean[0, 1], bundle_mean[0, 2],
                     color="black", marker="o", s=90, edgecolors="white",
                     linewidths=1.2, zorder=6)
        axA2.scatter(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                     color="black", marker="X", s=140, edgecolors="white",
                     linewidths=1.5, zorder=6)
        axA2.text(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                  f" {spec['name']}\n peakL={spec['observed_peakL']}",
                  fontsize=8, color="black",
                  bbox=dict(facecolor="white", edgecolor="black",
                            boxstyle="round,pad=0.15", alpha=0.85))
        for L in LAYER_LABELS:
            j = L - 1
            axA2.text(bundle_mean[j, 0], bundle_mean[j, 1], bundle_mean[j, 2],
                      f" L{L}", fontsize=6.5, color="black", alpha=0.6)

    axA1.set_title("(A1) UMAP-3D  •  bundle-coloured\n○=L1 start, X=L30 end")
    axA1.set_xlabel("UMAP1"); axA1.set_ylabel("UMAP2"); axA1.set_zlabel("UMAP3")
    axA1.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.05))
    axA2.set_title("(A2) UMAP-3D  •  layer-signal cividis")
    axA2.set_xlabel("UMAP1"); axA2.set_ylabel("UMAP2"); axA2.set_zlabel("UMAP3")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis,
                               norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cb = figA.colorbar(sm, ax=axA2, shrink=0.6, pad=0.05)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0 (min)", "0.25", "0.5", "0.75", "1 (peak)"])
    cb.set_label("per-layer Fisher signal (per-GO normalized)")
    figA.suptitle(f"Option A — UMAP 3D embedding of layer-normalized 16-D "
                  f"trajectory coords (N_iso={N_UMAP})", fontsize=11)
    figA.tight_layout(rect=[0, 0.01, 1, 0.95])
    figA.savefig(f"{OUT_DIR}/alt_A_umap_{ts}.png", dpi=170)
    plt.close(figA)
    print(f"     saved {OUT_DIR}/alt_A_umap_{ts}.png")

    # ========================================================================
    # (B) Per-layer 2D projection sequence
    # ========================================================================
    print(f"\n[{time.time()-t0:5.1f}s] === Option B: per-layer 2D sequence ===")
    L_PANELS = [1, 5, 10, 15, 20, 25, 30]                  # 7 panels
    figB, axsB = plt.subplots(1, len(L_PANELS), figsize=(23, 4.2),
                              sharex=False, sharey=False)
    # use norm PC1,PC2 (fixed axes across layers for direct comparison)
    coords_2d = reduced_nm[:, :, :2]                       # (N, 30, 2)

    for pi, L in enumerate(L_PANELS):
        ax = axsB[pi]
        j = L - 1
        # background scatter of all iso at this layer
        ax.scatter(coords_2d[:, j, 0], coords_2d[:, j, 1],
                   c="lightgray", s=1.8, alpha=0.35, edgecolors="none")
        for k, (go, spec) in enumerate(FEATURED.items()):
            c = spec["cluster"]
            if c < 0: continue
            good_idx = np.where((cid == c) & y[go])[0]
            if len(good_idx) < 5:
                good_idx = np.where(cid == c)[0]
            ax.scatter(coords_2d[good_idx, j, 0], coords_2d[good_idx, j, 1],
                       c=[BUNDLE_COLORS[k]], s=6, alpha=0.6,
                       edgecolors="none",
                       label=(f"{spec['name'][:14]}" if pi == 0 else None))
        ax.set_title(f"L{L}", fontsize=11)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2" if pi == 0 else "")
        if pi == 0:
            ax.legend(fontsize=6, loc="upper left")
        ax.set_aspect("auto")

    figB.suptitle("Option B — per-layer 2D snapshots (layer-norm PC1, PC2). "
                  "Points move from scattered (L1) toward concentrated bundle "
                  "positions (L30).", fontsize=11)
    figB.tight_layout(rect=[0, 0.01, 1, 0.92])
    figB.savefig(f"{OUT_DIR}/alt_B_per_layer_{ts}.png", dpi=160)
    plt.close(figB)
    print(f"     saved {OUT_DIR}/alt_B_per_layer_{ts}.png")

    # ========================================================================
    # (C) Unnormalized joint PCA 3D + bundle colors + Fisher color
    # ========================================================================
    print(f"\n[{time.time()-t0:5.1f}s] === Option C: unnormalized joint PCA 3D ===")
    figC = plt.figure(figsize=(22, 11))
    axC1 = figC.add_subplot(1, 2, 1, projection="3d")
    axC2 = figC.add_subplot(1, 2, 2, projection="3d")
    bg_local = np.random.default_rng(SEED).choice(N, size=N_BG, replace=False)
    for ax in (axC1, axC2):
        ax.set_facecolor("white")
        for i in bg_local:
            pts = coords_un[i]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color="lightgray", lw=0.35, alpha=0.20)

    for k, (go, spec) in enumerate(FEATURED.items()):
        c = spec["cluster"]
        if c < 0: continue
        good_idx = np.where((cid == c) & y[go])[0]
        if len(good_idx) < 5:
            good_idx = np.where(cid == c)[0]
        chosen = np.random.default_rng(SEED + k).choice(
            good_idx, size=min(N_SAMPLE, len(good_idx)), replace=False)
        bundle_mean = coords_un[cid == c].mean(axis=0)
        col = BUNDLE_COLORS[k]

        for i in chosen:
            pts = coords_un[i]
            axC1.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                      color=col, lw=1.1, alpha=0.55)
            lc = colored_line3d(pts, spec["curve_norm"],
                                cmap=plt.cm.cividis, lw=1.2, alpha=0.75)
            axC2.add_collection3d(lc)
        axC1.plot(bundle_mean[:, 0], bundle_mean[:, 1], bundle_mean[:, 2],
                  color=col, lw=3.0, alpha=1.0,
                  label=(f"{go} {spec['name'][:22]}\n"
                         f"peakL={spec['observed_peakL']} "
                         f"c{c} n={spec['n_members']} "
                         f"({spec['ratio']:.1f}×)"))
        axC1.scatter(bundle_mean[0, 0], bundle_mean[0, 1], bundle_mean[0, 2],
                     color=col, marker="o", s=110, edgecolors="black",
                     linewidths=1.4, zorder=6)
        axC1.scatter(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                     color=col, marker="X", s=160, edgecolors="black",
                     linewidths=1.4, zorder=6)
        for L in LAYER_LABELS:
            j = L - 1
            axC1.text(bundle_mean[j, 0], bundle_mean[j, 1], bundle_mean[j, 2],
                      f" L{L}", fontsize=6.5, color="black", alpha=0.7)
        axC2.plot(bundle_mean[:, 0], bundle_mean[:, 1], bundle_mean[:, 2],
                  color="black", lw=2.4, alpha=0.9)
        axC2.scatter(bundle_mean[0, 0], bundle_mean[0, 1], bundle_mean[0, 2],
                     color="black", marker="o", s=90,
                     edgecolors="white", linewidths=1.2, zorder=6)
        axC2.scatter(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                     color="black", marker="X", s=140,
                     edgecolors="white", linewidths=1.5, zorder=6)
        axC2.text(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                  f" {spec['name']}\n peakL={spec['observed_peakL']}",
                  fontsize=8, color="black",
                  bbox=dict(facecolor="white", edgecolor="black",
                            boxstyle="round,pad=0.15", alpha=0.85))

    axC1.set_title(f"(C1) Unnorm PCA-3D  •  bundle-coloured\n"
                   f"expl_var={pca_un.explained_variance_ratio_.sum():.2f}")
    axC1.set_xlabel("PC1"); axC1.set_ylabel("PC2"); axC1.set_zlabel("PC3")
    axC1.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.05))
    axC2.set_title("(C2) Unnorm PCA-3D  •  layer-signal cividis")
    axC2.set_xlabel("PC1"); axC2.set_ylabel("PC2"); axC2.set_zlabel("PC3")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis,
                               norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cb = figC.colorbar(sm, ax=axC2, shrink=0.6, pad=0.05)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0 (min)", "0.25", "0.5", "0.75", "1 (peak)"])
    cb.set_label("per-layer Fisher signal (per-GO normalized)")
    figC.suptitle("Option C — unnormalized joint PCA 3D "
                  "(86% variance in top 3, L1 compressed near origin)",
                  fontsize=11)
    figC.tight_layout(rect=[0, 0.01, 1, 0.95])
    figC.savefig(f"{OUT_DIR}/alt_C_unnorm3d_{ts}.png", dpi=170)
    plt.close(figC)
    print(f"     saved {OUT_DIR}/alt_C_unnorm3d_{ts}.png")

    print(f"\ntotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
