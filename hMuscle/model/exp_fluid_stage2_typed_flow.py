"""
exp_fluid_stage2_typed_flow.py
==============================
Stage 2 (revised): the point is NOT "beat L30 at overall winner count".
It is:

    (i)  Classify GO terms by *which layer their functional signal peaks*.
         late-dominated GO → fluid ≈ L30 (uninteresting, easy case)
         mid-/early-dominated GO → fluid may carry information L30 buries
         (interesting, "encoded but not expressed at output")

    (ii) Apply per-layer z-score normalization so joint PCA gives every
         layer equal say. Compare layer decomposition before vs after.

    (iii) Detect *side-branch events* in bundle trajectories:
         isoforms that briefly diverge from the bundle mean at some
         mid layer and return by L30 → candidate "encoded but not
         expressed" isoform-level features.

Deliverables
------------
    - Per-GO 30-layer Fisher discriminant signal profile
    - GO type classification (early / mid / late / flat)
    - Layer-norm vs unnorm PCA layer decomposition (frac_late compare)
    - Per-type stability shootout (norm-fluid vs unnorm-fluid vs L30 raw)
    - Side-branch top-20 isoforms per winner bundle
    - Visualizations: (a) layer signal heatmap grouped by type
                      (b) bundle tube 3D + side-branch overlay
"""

import os, json, time, gc
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
OUT_DIR    = "../../reports/fluid_stage2"
os.makedirs(OUT_DIR, exist_ok=True)

N_LAYERS = 30
EMB_DIM  = 640
SEED     = 42
K_PCA    = 8
K_CLUS_LIST = [12, 16]
GRID_SEEDS  = [42, 137]

# 34 usable specific BP terms (all with gene_hits >= 30)
GO_TERMS = {
    "GO:0006974": "DNA damage response",
    "GO:0035556": "Intracellular signal",
    "GO:0006508": "Proteolysis",
    "GO:0043161": "Proteasome-UPS",
    "GO:0006281": "DNA repair",
    "GO:0000226": "MT cytoskeleton org",
    "GO:0005975": "Carbohydrate metabolism",
    "GO:0055074": "Ca2+ homeostasis",
    "GO:0000165": "MAPK cascade",
    "GO:0000398": "mRNA splicing",
    "GO:0006417": "Regulation of translation",
    "GO:0007015": "Actin filament org",
    "GO:0007204": "Ca2+ signaling",
    "GO:0007059": "Chromosome segregation",
    "GO:0007265": "Ras signaling",
    "GO:0007018": "MT-based movement",
    "GO:0006816": "Ca2+ transport",
    "GO:0006888": "ER-Golgi transport",
    "GO:0006402": "mRNA catabolism",
    "GO:0006486": "Protein glycosylation",
    "GO:0006914": "Autophagy",
    "GO:0006470": "Dephosphorylation",
    "GO:0006836": "Neurotransmitter transp",
    "GO:0006414": "Translational elongation",
    "GO:0030048": "Actin-based movement",
    "GO:0032465": "Cytokinesis",
    "GO:0006906": "Vesicle fusion",
    "GO:0006418": "tRNA aminoacylation",
    "GO:0006754": "ATP biosynthesis",
    "GO:0006635": "FA beta-oxidation",
    "GO:0006120": "Complex I NADH ox",
    "GO:0045214": "Sarcomere organization",
    "GO:0006096": "Glycolysis",
    "GO:0006099": "TCA cycle",
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


def score_grid(feat, y_dict, k_clus, seed, N, baseline):
    km = KMeans(n_clusters=k_clus, n_init=5, random_state=seed)
    cid = km.fit_predict(feat)
    n_tests = k_clus * len(y_dict)
    alpha = 0.01 / n_tests
    sig = {}
    for go, y in y_dict.items():
        K_pos = int(y.sum())
        n_sig_c = 0
        for c in range(k_clus):
            m = (cid == c)
            n_c = int(m.sum())
            if n_c == 0:
                continue
            k_p = int(y[m].sum())
            if k_p == 0:
                continue
            frac = k_p / n_c
            ratio = frac / baseline[go] if baseline[go] > 0 else 0
            p_hg = float(hypergeom.sf(k_p - 1, N, K_pos, n_c))
            if ratio >= 2.0 and p_hg < alpha and k_p >= 5:
                n_sig_c += 1
        sig[go] = n_sig_c
    return sig, cid


def per_layer_fisher(traj, y):
    """Return 30-vec of Fisher score (GO+ vs GO-) per layer."""
    scores = np.zeros(N_LAYERS)
    pos_mask = y.astype(bool)
    neg_mask = ~pos_mask
    if pos_mask.sum() < 5 or neg_mask.sum() < 5:
        return scores
    for L in range(N_LAYERS):
        pts = traj[:, L, :]                             # (N, 640)
        mu_p = pts[pos_mask].mean(axis=0)
        mu_n = pts[neg_mask].mean(axis=0)
        v_p  = pts[pos_mask].var(axis=0)
        v_n  = pts[neg_mask].var(axis=0)
        num  = ((mu_p - mu_n) ** 2).sum()
        den  = (v_p + v_n).sum() + 1e-9
        scores[L] = num / den
    return scores


def classify_go_type(sig_curve):
    """early / mid / late / flat based on argmax of 30-vec."""
    if sig_curve.max() < 1e-9:
        return "flat"
    p = int(np.argmax(sig_curve)) + 1              # 1..30
    ratio = sig_curve.max() / (sig_curve.min() + 1e-12)
    if ratio < 2.0:
        return "flat"
    if p <= 10:
        return "early"
    elif p <= 20:
        return "mid"
    else:
        return "late"


def main():
    t0 = time.time()
    te_iso  = load_ids("my_isoform_list_fixed.npy")
    te_gene = load_ids("my_gene_list_fixed.npy")
    e2s = load_e2s()
    te_sym = [e2s.get(g.split(".")[0], g.split(".")[0]) for g in te_gene]
    print(f"[{time.time()-t0:5.1f}s] N_TE={len(te_iso)}")

    go_pos = {go: load_go_pos(go) for go in GO_TERMS}
    all_pos = set().union(*go_pos.values())
    pos_idx = sorted({i for i, s in enumerate(te_sym) if s in all_pos})
    neg_pool = [i for i, s in enumerate(te_sym) if s not in all_pos]
    n_neg = min(len(pos_idx), len(neg_pool), 15000 - len(pos_idx))
    rng = np.random.default_rng(SEED)
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False).tolist()
    subset_idx = np.array(sorted(pos_idx + list(neg_idx)))
    N = len(subset_idx)
    sub_sym = [te_sym[i] for i in subset_idx]
    sub_iso = [te_iso[i] for i in subset_idx]
    print(f"[{time.time()-t0:5.1f}s] pilot N={N} (pos_union={len(pos_idx)})")

    y = {go: np.array([1 if s in go_pos[go] else 0 for s in sub_sym],
                      dtype=np.int32) for go in GO_TERMS}
    baseline = {go: float(y[go].mean()) for go in GO_TERMS}
    y_go_counts = {go: int(y[go].sum()) for go in GO_TERMS}
    print(f"[{time.time()-t0:5.1f}s] per-GO positive count in pilot:")
    for go in sorted(GO_TERMS, key=lambda g: -y_go_counts[g]):
        print(f"     {go}  {GO_TERMS[go]:32s}  n={y_go_counts[go]}")

    # ---- trajectory
    print(f"\n[{time.time()-t0:5.1f}s] building trajectory ...")
    traj = build_trajectory(subset_idx)
    L30 = traj[:, -1, :].copy()
    print(f"[{time.time()-t0:5.1f}s] traj {traj.shape}")

    # ---- per-GO layer signal profile
    print(f"[{time.time()-t0:5.1f}s] per-GO Fisher signal profile (30 layers)")
    sig_curves = {}
    for go in GO_TERMS:
        sig_curves[go] = per_layer_fisher(traj, y[go])

    # normalize each curve to sum-1 (so peak comparison is fair)
    def normed(v):
        s = v.sum()
        return v / s if s > 1e-9 else v
    sig_curves_norm = {go: normed(sig_curves[go]) for go in GO_TERMS}

    go_type = {go: classify_go_type(sig_curves[go]) for go in GO_TERMS}
    print(f"[{time.time()-t0:5.1f}s] GO type distribution:")
    print(Counter(go_type.values()))
    for tp in ["early", "mid", "late", "flat"]:
        members = [g for g in GO_TERMS if go_type[g] == tp]
        print(f"  [{tp}] n={len(members)}")
        for g in members:
            peakL = int(np.argmax(sig_curves[g])) + 1
            peak_val = float(sig_curves[g].max())
            print(f"     peakL={peakL:2d}  peak_score={peak_val:.4f}  "
                  f"{g}  {GO_TERMS[g]}")

    # ---- LAYER NORMALIZATION
    print(f"\n[{time.time()-t0:5.1f}s] applying per-layer z-score normalization")
    layer_mean = traj.mean(axis=0)               # (30, 640)
    layer_std  = traj.std(axis=0) + 1e-6
    traj_norm  = (traj - layer_mean) / layer_std

    # ---- Fit joint PCA (both unnorm and norm)
    flat_un = traj.reshape(N * N_LAYERS, EMB_DIM)
    flat_nm = traj_norm.reshape(N * N_LAYERS, EMB_DIM)
    print(f"[{time.time()-t0:5.1f}s] joint PCA (K=16) unnorm ...")
    K_MAX = 16
    pca_un = PCA(n_components=K_MAX, random_state=SEED, svd_solver="randomized")
    reduced_un = pca_un.fit_transform(flat_un).reshape(N, N_LAYERS, K_MAX)
    print(f"          expl_var={pca_un.explained_variance_ratio_.sum():.3f}")
    print(f"[{time.time()-t0:5.1f}s] joint PCA (K=16) NORM ...")
    pca_nm = PCA(n_components=K_MAX, random_state=SEED, svd_solver="randomized")
    reduced_nm = pca_nm.fit_transform(flat_nm).reshape(N, N_LAYERS, K_MAX)
    print(f"          expl_var={pca_nm.explained_variance_ratio_.sum():.3f}")

    # layer decomposition on both bases
    def layer_decomp(traj_arr, pca_obj):
        contrib = np.zeros((K_MAX, N_LAYERS))
        for L in range(N_LAYERS):
            proj = pca_obj.transform(traj_arr[:, L, :])
            contrib[:, L] = proj.var(axis=0)
        contrib_norm = contrib / (contrib.sum(axis=1, keepdims=True) + 1e-12)
        # frac_late@top-8
        top8_ev = pca_obj.explained_variance_ratio_[:8].sum()
        late_ev = 0.0
        for i in range(8):
            if contrib_norm[i, 20:].sum() > 0.5:
                late_ev += pca_obj.explained_variance_ratio_[i]
        return contrib_norm, late_ev / top8_ev
    contrib_un, frac_late_un = layer_decomp(traj, pca_un)
    contrib_nm, frac_late_nm = layer_decomp(traj_norm, pca_nm)
    print(f"[{time.time()-t0:5.1f}s] layer decomp: "
          f"UNNORM frac_late@top8={frac_late_un:.3f}  "
          f"NORM frac_late@top8={frac_late_nm:.3f}")

    # ---- per-type shootout: build curve_vec at K_PCA and grid-run KMeans
    print(f"\n[{time.time()-t0:5.1f}s] per-type stability shootout (K_PCA={K_PCA})")
    cv_un = reduced_un[:, :, :K_PCA].reshape(N, N_LAYERS * K_PCA)
    cv_nm = reduced_nm[:, :, :K_PCA].reshape(N, N_LAYERS * K_PCA)

    win_counts = {"fluid_un": defaultdict(int),
                  "fluid_nm": defaultdict(int),
                  "L30_raw":  defaultdict(int)}
    n_grid = 0
    for k_clus in K_CLUS_LIST:
        for seed in GRID_SEEDS:
            t1 = time.time()
            sig_un, cid_un = score_grid(cv_un, y, k_clus, seed, N, baseline)
            sig_nm, cid_nm = score_grid(cv_nm, y, k_clus, seed, N, baseline)
            sig_r,  cid_r  = score_grid(L30,   y, k_clus, seed, N, baseline)
            for go in GO_TERMS:
                if sig_un[go] >= 1: win_counts["fluid_un"][go] += 1
                if sig_nm[go] >= 1: win_counts["fluid_nm"][go] += 1
                if sig_r[go]  >= 1: win_counts["L30_raw"][go]  += 1
            n_grid += 1
            print(f"  grid k_clus={k_clus} seed={seed}  ({time.time()-t1:.1f}s)")

    # per-type aggregate
    stability = {m: {go: win_counts[m][go] / n_grid for go in GO_TERMS}
                 for m in win_counts}
    per_type = defaultdict(lambda: {"un": [], "nm": [], "r": [], "gos": []})
    for go in GO_TERMS:
        tp = go_type[go]
        per_type[tp]["un"].append(stability["fluid_un"][go])
        per_type[tp]["nm"].append(stability["fluid_nm"][go])
        per_type[tp]["r"].append(stability["L30_raw"][go])
        per_type[tp]["gos"].append(go)

    print(f"\nType    n_go  fluid_un  fluid_nm  L30_raw   winner")
    type_summary = {}
    for tp in ["early", "mid", "late", "flat"]:
        d = per_type.get(tp)
        if d is None or len(d["un"]) == 0:
            continue
        avg = {k: float(np.mean(d[k])) for k in ["un", "nm", "r"]}
        rob = {k: sum(1 for x in d[k] if x >= 0.80) for k in ["un", "nm", "r"]}
        winner = max(avg, key=avg.get)
        type_summary[tp] = dict(n_go=len(d["un"]),
                                avg=avg, robust=rob,
                                gos=d["gos"], winner=winner)
        print(f"  {tp:6s}{len(d['un']):>5d}"
              f"{avg['un']:>10.2f}{avg['nm']:>10.2f}{avg['r']:>10.2f}"
              f"    ({winner}) rob({rob['un']}/{rob['nm']}/{rob['r']})")

    # ---- SIDE-BRANCH DETECTION ----
    # find the best mid-type GO winner cluster (from NORM fluid),
    # detect isoforms with side-branch score >> 0
    print(f"\n[{time.time()-t0:5.1f}s] side-branch detection")
    side_report = {}
    for tp in ["mid", "late", "early"]:
        gos = per_type.get(tp, {}).get("gos", [])
        if not gos:
            continue
        # pick GO with highest stability under norm
        top_go = max(gos, key=lambda g: stability["fluid_nm"][g])
        if stability["fluid_nm"][top_go] < 0.5:
            continue
        # locate cluster of that GO from k=16 seed=42 (deterministic)
        km = KMeans(n_clusters=16, n_init=5, random_state=SEED)
        cid = km.fit_predict(cv_nm)
        K_pos = int(y[top_go].sum())
        best_c = None; best_ratio = 0
        for c in range(16):
            m = (cid == c)
            n_c = int(m.sum())
            if n_c < 10:
                continue
            k_p = int(y[top_go][m].sum())
            frac = k_p / n_c
            ratio = frac / baseline[top_go] if baseline[top_go] > 0 else 0
            if ratio > best_ratio and k_p >= 5:
                best_ratio = ratio
                best_c = c
        if best_c is None:
            continue
        member_idx = np.where(cid == best_c)[0]
        # bundle mean trajectory in K_PCA-space
        bundle_mean = reduced_nm[member_idx].mean(axis=0)   # (30, K_MAX)
        # per-isoform per-layer deviation
        # take first 3 PC axes for stable geometry
        dev = np.linalg.norm(reduced_nm[member_idx][:, :, :3]
                              - bundle_mean[:, :3][None, :, :], axis=2)
        # side-branch score = max_L dev - dev at L30
        sb = dev.max(axis=1) - dev[:, -1]
        order = np.argsort(-sb)
        # find peak L per iso
        peak_L = dev.argmax(axis=1) + 1

        top_side_branch = []
        for k in range(min(20, len(order))):
            i_sub = member_idx[order[k]]
            top_side_branch.append(dict(
                rank=k + 1,
                iso=sub_iso[i_sub], gene=sub_sym[i_sub],
                sb_score=float(sb[order[k]]),
                peak_layer=int(peak_L[order[k]]),
                is_go_pos=bool(y[top_go][i_sub]),
            ))
        side_report[tp] = dict(
            top_go=top_go, top_go_name=GO_TERMS[top_go],
            cluster=int(best_c), n_members=len(member_idx),
            ratio=float(best_ratio),
            top20=top_side_branch,
        )
        print(f"  [{tp}] top GO={top_go} ({GO_TERMS[top_go]}) "
              f"cluster c{best_c} n={len(member_idx)}")
        print(f"     top-10 side-branch isoforms "
              f"(rank | iso | gene | sb_score | peak_L | GO+):")
        for t in top_side_branch[:10]:
            g = "+" if t["is_go_pos"] else "-"
            print(f"       {t['rank']:2d}  {t['iso']:20s}  "
                  f"{t['gene']:12s}  sb={t['sb_score']:+.2f}  "
                  f"peakL={t['peak_layer']:2d}  GO{g}")

    # ---- VISUALIZATIONS ----
    ts = time.strftime("%Y%m%d_%H%M")

    # A: layer signal heatmap grouped by type
    print(f"\n[{time.time()-t0:5.1f}s] plotting layer signal heatmap")
    ordered_gos, ordered_types = [], []
    for tp in ["early", "mid", "late", "flat"]:
        members = [g for g in GO_TERMS if go_type[g] == tp]
        members.sort(key=lambda g: np.argmax(sig_curves[g]))
        for g in members:
            ordered_gos.append(g)
            ordered_types.append(tp)
    signal_mat = np.stack([sig_curves_norm[g] for g in ordered_gos])

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(signal_mat, aspect="auto", cmap="magma", origin="upper")
    ax.set_xlabel("ESM-2 layer L (1..30)")
    ax.set_yticks(range(len(ordered_gos)))
    ax.set_yticklabels([f"[{t[:3]}] {GO_TERMS[g]}"
                        for g, t in zip(ordered_gos, ordered_types)],
                       fontsize=7)
    ax.set_xticks(range(0, 30, 2))
    ax.set_xticklabels(range(1, 31, 2))
    ax.set_title(f"per-GO layer Fisher signal (row-normalized), "
                 f"grouped by type\n"
                 f"UNNORM PCA frac_late@top8={frac_late_un:.2f}, "
                 f"NORM PCA frac_late@top8={frac_late_nm:.2f}")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/typed_layer_heatmap_{ts}.png", dpi=140)
    plt.close(fig)

    # B: bundle tube visualization for each type winner (up to 3 panels)
    print(f"[{time.time()-t0:5.1f}s] plotting bundle tubes with side-branches")
    for tp, rep in side_report.items():
        top_go = rep["top_go"]
        cid_arr = KMeans(n_clusters=16, n_init=5,
                         random_state=SEED).fit_predict(cv_nm)
        member_idx = np.where(cid_arr == rep["cluster"])[0]
        # bundle mean + 5 top side-branches
        bundle_mean = reduced_nm[member_idx].mean(axis=0)
        # random 30 members for context
        context_n = min(30, len(member_idx))
        ctx_idx = np.random.default_rng(SEED).choice(
            member_idx, size=context_n, replace=False)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        # context (gray thin lines)
        for k in ctx_idx:
            pts = reduced_nm[k, :, :3]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color="lightgray", lw=0.6, alpha=0.6)
        # bundle mean (thick blue)
        ax.plot(bundle_mean[:, 0], bundle_mean[:, 1], bundle_mean[:, 2],
                color="tab:blue", lw=3.5, label="bundle mean")
        ax.scatter(bundle_mean[0, 0], bundle_mean[0, 1], bundle_mean[0, 2],
                   color="tab:blue", marker="o", s=60)
        ax.scatter(bundle_mean[-1, 0], bundle_mean[-1, 1], bundle_mean[-1, 2],
                   color="tab:blue", marker="X", s=100)
        # top-5 side-branch (crimson)
        for k in range(min(5, len(rep["top20"]))):
            iso_row = rep["top20"][k]
            # find its subset index
            iso_matches = [i for i in member_idx if sub_iso[i] == iso_row["iso"]]
            if not iso_matches:
                continue
            i_sub = iso_matches[0]
            pts = reduced_nm[i_sub, :, :3]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color="crimson", lw=2, alpha=0.85,
                    label=(f"{iso_row['gene']} sb={iso_row['sb_score']:.1f}"
                           if k < 3 else None))
            ax.scatter(pts[iso_row["peak_layer"] - 1, 0],
                       pts[iso_row["peak_layer"] - 1, 1],
                       pts[iso_row["peak_layer"] - 1, 2],
                       color="crimson", marker="*", s=110)
        ax.set_title(f"[{tp}] {top_go} {GO_TERMS[top_go]}\n"
                     f"cluster c{rep['cluster']} n={rep['n_members']} "
                     f"ratio={rep['ratio']:.2f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/bundle_tube_{tp}_{ts}.png", dpi=140)
        plt.close(fig)

    # ---- SAVE ----
    with open(f"{OUT_DIR}/typed_flow_{ts}.json", "w") as f:
        json.dump(dict(
            pilot_N=int(N),
            pilot_pos_union=len(pos_idx),
            go_terms=GO_TERMS,
            go_positive_counts=y_go_counts,
            baseline=baseline,
            frac_late_at_top8=dict(unnorm=float(frac_late_un),
                                   norm=float(frac_late_nm)),
            expl_var_top16=dict(
                unnorm=float(pca_un.explained_variance_ratio_.sum()),
                norm=float(pca_nm.explained_variance_ratio_.sum()),
            ),
            sig_curves={g: sig_curves[g].tolist() for g in GO_TERMS},
            go_type=go_type,
            stability=stability,
            type_summary=type_summary,
            side_branch_report=side_report,
        ), f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else o)
    print(f"\nsaved: {OUT_DIR}/typed_flow_{ts}.json")
    print(f"saved: {OUT_DIR}/typed_layer_heatmap_{ts}.png")
    print(f"saved: {OUT_DIR}/bundle_tube_*_{ts}.png "
          f"(one per active GO type)")
    print(f"total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
