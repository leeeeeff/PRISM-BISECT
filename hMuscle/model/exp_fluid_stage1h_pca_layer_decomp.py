"""
exp_fluid_stage1h_pca_layer_decomp.py
=====================================
Decompose the joint PCA basis by *source layer*.

Question
--------
When we fit joint PCA on all (isoform x layer) points from L1..L30,
does the top-K basis capture information distributed *across* layers,
or is it dominated by the last layers (L28-L30)?

If the top 8 PCs are essentially just L28-L30 axes, then curve_vec is
a re-projection of L30 (matches devils-advocate concern).

Method
------
1.  Fit joint PCA (K=16) on all (isoform x layer) points from the
    narrow-GO pilot subset.
2.  For each PC_i and each layer L, compute:
      var_contrib(i, L) = var over isoforms of PC_i.project(layer_L_point)
    This measures how much variance PC_i captures *at layer L*.
3.  Normalize per-PC so contribution sums to 1 across layers.
4.  Report per-PC layer-of-max-contribution + heatmap.
"""

import os, json, time, gc
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = "../data"
ANNOT_FILE = "../data/raw_data/data/annotations/human_annotations_unified_bp.txt"
ID_DIR     = "../data/raw_data/data/id_lists"
OUT_DIR    = "../../reports/fluid_stage1"

N_LAYERS = 30
EMB_DIM  = 640
SEED     = 42
K_PCA    = 16

NARROW_GO = [
    "GO:0006096", "GO:0006099", "GO:0006120", "GO:0006754",
    "GO:0006635", "GO:0006418", "GO:0045214", "GO:0030048",
    "GO:0007018", "GO:0007015", "GO:0007204", "GO:0006816",
    "GO:0006888", "GO:0000398", "GO:0006414",
]


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
    e2s = load_e2s()
    te_sym = [e2s.get(g.split(".")[0], g.split(".")[0]) for g in te_gene]
    print(f"[{time.time()-t0:5.1f}s] N_TE={len(te_iso)}")

    go_pos = {go: load_go_pos(go) for go in NARROW_GO}
    all_pos = set().union(*go_pos.values())
    pos_idx = sorted({i for i, s in enumerate(te_sym) if s in all_pos})
    neg_pool = [i for i, s in enumerate(te_sym) if s not in all_pos]
    n_neg = min(len(pos_idx), len(neg_pool))
    rng = np.random.default_rng(SEED)
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False).tolist()
    subset_idx = np.array(sorted(pos_idx + list(neg_idx)))
    N = len(subset_idx)
    print(f"[{time.time()-t0:5.1f}s] pilot N={N}")

    print(f"[{time.time()-t0:5.1f}s] building trajectory ...")
    traj = build_trajectory(subset_idx)
    print(f"[{time.time()-t0:5.1f}s] traj {traj.shape}")

    # joint PCA
    flat = traj.reshape(N * N_LAYERS, EMB_DIM)
    pca = PCA(n_components=K_PCA, random_state=SEED, svd_solver="randomized")
    pca.fit(flat)
    print(f"          expl_var={pca.explained_variance_ratio_.sum():.3f}")

    # project each layer independently
    var_contrib = np.zeros((K_PCA, N_LAYERS), dtype=np.float64)  # (PC, Layer)
    for L in range(N_LAYERS):
        layer_pts = traj[:, L, :]                  # (N, 640)
        proj = pca.transform(layer_pts)            # (N, K_PCA)
        var_contrib[:, L] = proj.var(axis=0)
    # normalize per-PC across layers so rows sum to 1
    row_sum = var_contrib.sum(axis=1, keepdims=True)
    var_contrib_norm = var_contrib / (row_sum + 1e-12)

    # per-PC dominant layer
    dom_layer = var_contrib_norm.argmax(axis=1)     # (K_PCA,)
    # peak concentration = max normalized value
    peak_frac = var_contrib_norm.max(axis=1)        # (K_PCA,)

    # summary: for each PC, which layer group dominates?
    print(f"\n{'PC':>4}{'expl_var':>10}{'dom_L':>7}{'peak_frac':>11}"
          f"{'L1-10':>8}{'L11-20':>9}{'L21-30':>9}"
          f"{'top3_L':>18}")
    layer_groups = {
        "L1-10":  slice(0, 10),
        "L11-20": slice(10, 20),
        "L21-30": slice(20, 30),
    }
    per_pc = []
    for i in range(K_PCA):
        row = var_contrib_norm[i]
        top3 = np.argsort(-row)[:3] + 1
        grp_fracs = {g: float(row[s].sum()) for g, s in layer_groups.items()}
        print(f"{i:>4}"
              f"{pca.explained_variance_ratio_[i]:>10.3f}"
              f"{int(dom_layer[i])+1:>7d}"
              f"{peak_frac[i]:>11.3f}"
              f"{grp_fracs['L1-10']:>8.3f}"
              f"{grp_fracs['L11-20']:>9.3f}"
              f"{grp_fracs['L21-30']:>9.3f}"
              f"   {list(top3)}")
        per_pc.append(dict(pc=i, expl_var=float(pca.explained_variance_ratio_[i]),
                           dom_layer=int(dom_layer[i]) + 1,
                           peak_frac=float(peak_frac[i]),
                           frac_L1_L10=grp_fracs['L1-10'],
                           frac_L11_L20=grp_fracs['L11-20'],
                           frac_L21_L30=grp_fracs['L21-30'],
                           top3_layers=[int(x) for x in top3]))

    # weighted by expl_var: what fraction of top-8 total variance comes from L21-30 dominated PCs?
    ev = pca.explained_variance_ratio_
    top8_ev = ev[:8].sum()
    late_dom_ev_top8 = 0.0
    for i in range(8):
        if dom_layer[i] >= 20:
            late_dom_ev_top8 += ev[i]
    frac_late = late_dom_ev_top8 / top8_ev
    print(f"\nof top-8 PC total expl_var, {frac_late*100:.1f}% comes from "
          f"PCs whose dominant layer is L21-30")

    # devils verdict
    if frac_late >= 0.80:
        print("VERDICT: top-8 PCs are LATE-LAYER dominated "
              "(curve_vec ≈ L21-30 combo). Devils' 'PCA is L30 rebottle' partially confirmed.")
    elif frac_late >= 0.50:
        print("VERDICT: top-8 PCs are MIXED but skewed late.")
    else:
        print("VERDICT: top-8 PCs span layers, "
              "including early L1-L10 information. Devils' concern refuted.")

    # heatmap
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(var_contrib_norm, aspect="auto",
                   cmap="viridis", origin="lower")
    ax.set_xlabel("layer L (1..30)")
    ax.set_ylabel("PC index (top-16)")
    ax.set_title(f"per-PC layer variance contribution "
                 f"(row-normalized)  frac_late@top8={frac_late:.2f}")
    ax.set_xticks(range(0, 30, 2))
    ax.set_xticklabels(range(1, 31, 2))
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/pca_layer_heatmap.png", dpi=140)
    plt.close(fig)

    with open(f"{OUT_DIR}/pca_layer_decomp.json", "w") as f:
        json.dump(dict(
            k_pca=K_PCA,
            pilot_N=int(N),
            expl_var=[float(x) for x in ev],
            per_pc=per_pc,
            frac_late_dominated_top8=float(frac_late),
            var_contrib_row_normalized=var_contrib_norm.tolist(),
        ), f, indent=2)
    print(f"\nsaved: {OUT_DIR}/pca_layer_decomp.json")
    print(f"saved: {OUT_DIR}/pca_layer_heatmap.png")
    print(f"total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
