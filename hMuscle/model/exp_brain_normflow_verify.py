"""
exp_brain_normflow_verify.py
============================
Replication of the normalized residual flow signal verification
on BRAIN isoform data (63994 isoforms, layers 7/18/27/30).

Muscle result (reference):
  Within-gene L30/L1 ratio : 2.204
  Gene-label shuffle null  : 1.808
  Cross-gene baseline      : 1.827
  Partial ρ(L30|L1_sep)   : 0.271

Question: do the same 4 Fix controls hold in brain zero-shot?
  L_start = L7, L_end = L30 (same principle, 4 checkpoint layers)
"""

import os, time
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BRAIN_DIR = "../data/brain_isoquant_esm2/full"
OUT_DIR   = "../../reports/fluid_stage3"
os.makedirs(OUT_DIR, exist_ok=True)

EMB_DIM  = 640
SEED     = 42

LAYER_FILES = {
    7:  f"{BRAIN_DIR}/brain_full_esm2_layer07_t30_150M.npy",
    18: f"{BRAIN_DIR}/brain_full_esm2_layer18_t30_150M.npy",
    27: f"{BRAIN_DIR}/brain_full_esm2_layer27_t30_150M.npy",
    30: f"{BRAIN_DIR}/brain_full_esm2_t30_150M.npy",
}
LAYER_KEYS = sorted(LAYER_FILES.keys())   # [7, 18, 27, 30]
N_LAYERS   = len(LAYER_KEYS)


def partial_r(x, y, z):
    """Rank-based partial correlation of x,y controlling for z."""
    def resid(a, b):
        coef = np.polyfit(b, a, 1)
        return a - np.polyval(coef, b)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rz = np.argsort(np.argsort(z)).astype(float)
    r, p = pearsonr(resid(rx, rz), resid(ry, rz))
    return r, p


def main():
    t0 = time.time()

    # ── load ──────────────────────────────────────────────────────────────
    print("Loading brain data...")
    genes = np.load(f"{BRAIN_DIR}/brain_full_gene_names.npy", allow_pickle=True)
    genes = np.array([str(g) for g in genes])
    N = len(genes)
    print(f"  N={N}, unique genes={len(np.unique(genes))}")

    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for li, lkey in enumerate(LAYER_KEYS):
        arr = np.load(LAYER_FILES[lkey], mmap_mode="r")
        traj[:, li, :] = arr
        print(f"  L{lkey} loaded: {arr.shape}")

    # ── normalization ──────────────────────────────────────────────────────
    print(f"[{time.time()-t0:.1f}s] Normalizing...")
    l1_norm_orig = np.linalg.norm(traj[:, 0, :].astype(np.float64), axis=1)

    layer_mean = traj.mean(axis=0)               # (4, 640)
    layer_std  = traj.std(axis=0) + 1e-6
    traj_norm  = (traj - layer_mean) / layer_std  # (N, 4, 640)
    del traj

    # ── joint PCA ─────────────────────────────────────────────────────────
    print(f"[{time.time()-t0:.1f}s] PCA...")
    flat_nm = traj_norm.reshape(N * N_LAYERS, EMB_DIM)
    pca = PCA(n_components=3, random_state=SEED, svd_solver="randomized")
    coords = pca.fit_transform(flat_nm).reshape(N, N_LAYERS, 3)
    print(f"  norm PCA top-3 var={pca.explained_variance_ratio_.sum():.3f}")
    traj_nm_rebuilt = flat_nm.reshape(N, N_LAYERS, EMB_DIM)

    # ── within-gene grouping ───────────────────────────────────────────────
    gene_to_idx = defaultdict(list)
    for i, g in enumerate(genes):
        gene_to_idx[g].append(i)
    multi_iso = {g: v for g, v in gene_to_idx.items() if len(v) >= 2}
    print(f"[{time.time()-t0:.1f}s] Multi-isoform genes: {len(multi_iso)}")

    # ── Fix 1: within-gene pairs + non-circular len_proxy ─────────────────
    print(f"\n[{time.time()-t0:.1f}s] Computing pairwise separations...")
    sep_start, sep_end, len_diff = [], [], []

    for g, isos in multi_iso.items():
        for ii in range(len(isos)):
            for jj in range(ii + 1, len(isos)):
                a, b = isos[ii], isos[jj]
                sep_start.append(np.linalg.norm(coords[a, 0] - coords[b, 0]))   # L7
                sep_end.append(np.linalg.norm(coords[a, -1] - coords[b, -1]))   # L30
                len_diff.append(abs(l1_norm_orig[a] - l1_norm_orig[b]))

    sep_start = np.array(sep_start)
    sep_end   = np.array(sep_end)
    len_diff  = np.array(len_diff)
    ratio_wg  = sep_end.mean() / sep_start.mean()

    print(f"\n  [Brain: within-gene]")
    print(f"  L7  (start) sep: mean={sep_start.mean():.4f}")
    print(f"  L30 (end)   sep: mean={sep_end.mean():.4f}")
    print(f"  L30/L7 ratio:    {ratio_wg:.3f}")

    r_start, p_start = spearmanr(len_diff, sep_start)
    r_end,   p_end   = spearmanr(len_diff, sep_end)
    print(f"\n  [Fix 1] Non-circular len_proxy:")
    print(f"  ρ(len_orig ↔ L7  sep) = {r_start:.3f}, p={p_start:.3e}")
    print(f"  ρ(len_orig ↔ L30 sep) = {r_end:.3f},   p={p_end:.3e}")
    delta_rho = r_end - r_start
    print(f"  Δρ = {delta_rho:+.3f}  {'(ρ drops → extra L30 signal)' if delta_rho < 0 else '(ρ increases → artifact risk)'}")

    # ── Fix 2: cross-gene baseline ─────────────────────────────────────────
    rng = np.random.default_rng(SEED + 999)
    n_pairs = min(len(sep_start), 50000)
    cross_start, cross_end = [], []
    attempts = 0
    while len(cross_start) < n_pairs and attempts < n_pairs * 10:
        a, b = rng.integers(0, N, size=2)
        if genes[a] != genes[b]:
            cross_start.append(np.linalg.norm(coords[a, 0] - coords[b, 0]))
            cross_end.append(np.linalg.norm(coords[a, -1] - coords[b, -1]))
        attempts += 1
    cross_start = np.array(cross_start)
    cross_end   = np.array(cross_end)
    ratio_xg = cross_end.mean() / cross_start.mean()

    print(f"\n  [Fix 2] Cross-gene baseline (n={len(cross_start)}):")
    print(f"  Cross-gene L30/L7 ratio: {ratio_xg:.3f}")
    print(f"  Within-gene true:        {ratio_wg:.3f}")
    if ratio_wg > ratio_xg:
        print(f"  → within > cross ({ratio_wg:.3f} > {ratio_xg:.3f}): isoform-specific signal ✓")
    else:
        print(f"  → within ≤ cross ({ratio_wg:.3f} ≤ {ratio_xg:.3f}): no isoform specificity ✗")

    # ── Fix 3: gene-label shuffle null ────────────────────────────────────
    rng_s = np.random.default_rng(SEED + 42)
    gene_sizes = [len(v) for v in multi_iso.values()]
    shuffled_pool = rng_s.permutation(N)
    ptr = 0
    shuf_start, shuf_end = [], []
    for sz in gene_sizes:
        if ptr + sz > N:
            break
        group = shuffled_pool[ptr:ptr + sz]
        ptr += sz
        for ii in range(len(group)):
            for jj in range(ii + 1, len(group)):
                a, b = int(group[ii]), int(group[jj])
                shuf_start.append(np.linalg.norm(coords[a, 0] - coords[b, 0]))
                shuf_end.append(np.linalg.norm(coords[a, -1] - coords[b, -1]))
    shuf_start = np.array(shuf_start)
    shuf_end   = np.array(shuf_end)
    ratio_shuf = shuf_end.mean() / shuf_start.mean()

    print(f"\n  [Fix 3] Gene-label shuffle null (n={len(shuf_start)}):")
    print(f"  Shuffle null L30/L7:  {ratio_shuf:.3f}")
    print(f"  Within-gene true:     {ratio_wg:.3f}")
    if ratio_wg > ratio_shuf:
        print(f"  → within > null ({ratio_wg:.3f} > {ratio_shuf:.3f}): gene structure signal ✓")
    else:
        print(f"  → within ≤ null ({ratio_wg:.3f} ≤ {ratio_shuf:.3f}): normalization artifact ✗")

    # ── Fix 4: partial correlation ─────────────────────────────────────────
    r_part, p_part = partial_r(sep_end, len_diff, sep_start)
    print(f"\n  [Fix 4] Partial correlation:")
    print(f"  ρ(L30_sep, len_orig | L7_sep) = {r_part:.3f}, p={p_part:.3e}")
    if abs(r_part) > 0.1 and p_part < 0.05:
        print(f"  → trajectory adds info beyond start point ✓")
    else:
        print(f"  → L30 sep explained by L7 alone ✗")

    # ── summary vs muscle ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {'Metric':<32} {'Muscle':>8} {'Brain':>8}")
    print(f"  {'-'*48}")
    print(f"  {'Within-gene L_end/L_start ratio':<32} {'2.204':>8} {ratio_wg:>8.3f}")
    print(f"  {'Gene-label shuffle null':<32} {'1.808':>8} {ratio_shuf:>8.3f}")
    print(f"  {'Cross-gene baseline':<32} {'1.827':>8} {ratio_xg:>8.3f}")
    print(f"  {'Δρ (Fix1: L_start→L_end)':<32} {'-0.009':>8} {delta_rho:>+8.3f}")
    print(f"  {'Partial ρ (Fix4)':<32} {'0.271':>8} {r_part:>8.3f}")
    print(f"{'='*55}")

    # ── figure ────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # panel 1: boxplot L7 vs L30 within-gene
    axs[0].boxplot([sep_start, sep_end], tick_labels=["L7 (start)", "L30 (end)"],
                   showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="salmon", alpha=0.7))
    axs[0].set_title(f"Brain: within-gene separation\nL30/L7={ratio_wg:.3f}", fontsize=11)
    axs[0].set_ylabel("PCA distance (norm)")

    # panel 2: ratio comparison with baselines
    labels = ["Within-gene\n(true)", "Label-shuffle\n(null)", "Cross-gene\n(base)"]
    ratios = [ratio_wg, ratio_shuf, ratio_xg]
    colors = ["firebrick", "gray", "steelblue"]
    bars = axs[1].bar(labels, ratios, color=colors, alpha=0.75)
    axs[1].axhline(1.0, color="black", lw=1.0, linestyle="--", alpha=0.5)
    axs[1].set_ylabel("L30/L7 ratio")
    axs[1].set_title("Fix 2+3: baselines", fontsize=11)
    for bar, val in zip(bars, ratios):
        axs[1].text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    # panel 3: muscle vs brain comparison
    muscle_vals = [2.204, 1.808, 1.827, 0.271]
    brain_vals  = [ratio_wg, ratio_shuf, ratio_xg, r_part]
    xlabels     = ["WG ratio", "Null ratio", "XG ratio", "Partial ρ"]
    x = np.arange(len(xlabels))
    w = 0.35
    axs[2].bar(x - w/2, muscle_vals, w, label="Muscle", color="steelblue", alpha=0.75)
    axs[2].bar(x + w/2, brain_vals,  w, label="Brain",  color="salmon",    alpha=0.75)
    axs[2].set_xticks(x); axs[2].set_xticklabels(xlabels)
    axs[2].set_title("Muscle vs Brain replication", fontsize=11)
    axs[2].legend(fontsize=9)
    axs[2].axhline(0, color="black", lw=0.8)

    fig.suptitle("Brain: normalized residual flow verification (L7→L30, 4 checkpoints)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    ts = time.strftime("%Y%m%d_%H%M")
    out_path = f"{OUT_DIR}/brain_normflow_verify_{ts}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"\n  saved: {out_path}")
    print(f"  total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
