"""
plot_trajectory_flow_3d.py
==========================
fluid_stage2 스타일 3D bundle-tube 시각화, 18 BP GO에 적용.

각 패널 (per GO):
  회색 얇은 선: GO-positive 단백질 30개 sample 궤적 (context cloud)
  파란 두꺼운 선: pos 그룹 bundle mean 궤적
  빨간 얇은 점선: neg 그룹 bundle mean 궤적 (대조)
  파란 O: L1 (시작), 파란 X: L30 (끝)
  금색 별: Fisher peak layer 위치

두 버전:
  Non-normalized: Z_te 그대로 (이미 joint PCA)
  Normalized: per-layer z-score 후 시각화
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from pathlib import Path

SEED = 42
rng  = np.random.default_rng(SEED)

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
DATA      = ROOT / "data"
ID_DIR    = DATA / "raw_data/data/id_lists"
ANNOT_DIR = DATA / "raw_data/data/annotations"
PROBE_DIR = ROOT.parent / "reports" / "layer_probe"
CACHE_DIR = ROOT.parent / "reports" / "v20_cache"
OUT_DIR   = ROOT.parent / "reports" / "curve_sweep"

GO_18 = {
    "GO:0007204": "Ca2+ signaling",
    "GO:0045214": "Sarcomere org",
    "GO:0006941": "Muscle contract",
    "GO:0006914": "Autophagy",
    "GO:0043161": "Proteasome-UPS",
    "GO:0007519": "Skeletal musc dev",
    "GO:0042692": "Muscle cell diff",
    "GO:0055074": "Ca2+ homeostasis",
    "GO:0007005": "Mitochondrion org",
    "GO:0007517": "Muscle organ dev",
    "GO:0032006": "TOR signaling",
    "GO:0030048": "Actin-based mov",
    "GO:0006096": "Glycolysis",
    "GO:0007268": "Synaptic transm",
    "GO:0007018": "MT-based mov",
    "GO:0031175": "Neuron proj dev",
    "GO:0030182": "Neuron diff",
    "GO:0000226": "MT cytosk org",
}
MID_GOs = {"GO:0007204", "GO:0007018", "GO:0000226"}

N_CONTEXT = 20   # context lines per panel (fluid_stage2 used 30)


def load_fisher():
    all_lr = {}
    for fname in ["layer_probe_v15d_terms_results.json",
                  "layer_probe_expanded_results.json",
                  "layer_probe_results.json"]:
        p = PROBE_DIR / fname
        if p.exists():
            d = json.load(open(p))
            all_lr.update(d["lr_auprc"])
    return {go: np.array(all_lr[go], dtype=np.float32) for go in GO_18}


def load_labels(go, sym_te):
    pos = set()
    with open(ANNOT_DIR / "human_annotations_unified_bp.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 1 and go in parts[1:]:
                pos.add(parts[0])
    return np.array([s in pos for s in sym_te], dtype=bool)


def draw_panel(ax, Z, y_mask, fisher_scores, go_name, is_mid, is_first=False):
    """Draw a single 3D bundle-tube panel."""
    pos_idx = np.where(y_mask)[0]
    neg_idx = np.where(~y_mask)[0]

    if len(pos_idx) == 0:
        ax.set_axis_off(); return

    # Bundle mean (pos)
    bm_pos = Z[pos_idx, :, :3].mean(axis=0)

    # Set axis limits based on mean trajectory bbox (avoid outlier scaling)
    pad_frac = 0.6
    for dim, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        vals = bm_pos[:, dim]
        lo, hi = vals.min(), vals.max()
        rng_ = hi - lo + 1e-6
        setter(lo - pad_frac * rng_, hi + pad_frac * rng_)

    # Context: N random pos proteins (clipped to axis bbox visually via zorder)
    n_ctx = min(N_CONTEXT, len(pos_idx))
    ctx_idx = rng.choice(pos_idx, size=n_ctx, replace=False)
    for k in ctx_idx:
        pts = Z[k, :, :3]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color="#B0B0B0", lw=0.7, alpha=0.55)

    # Bundle mean (pos) — thick blue
    ax.plot(bm_pos[:, 0], bm_pos[:, 1], bm_pos[:, 2],
            color="tab:blue", lw=3.5, label=f"pos mean (n={len(pos_idx)})",
            zorder=5)
    ax.scatter(bm_pos[0, 0], bm_pos[0, 1], bm_pos[0, 2],
               color="tab:blue", marker="o", s=90, edgecolors='white', lw=1.2,
               zorder=8)
    ax.scatter(bm_pos[-1, 0], bm_pos[-1, 1], bm_pos[-1, 2],
               color="tab:blue", marker="X", s=140, edgecolors='white', lw=1.2,
               zorder=8)

    # Bundle mean (neg) — dashed crimson
    if len(neg_idx) > 0:
        bm_neg = Z[neg_idx, :, :3].mean(axis=0)
        ax.plot(bm_neg[:, 0], bm_neg[:, 1], bm_neg[:, 2],
                color="crimson", lw=2.0, ls='--', alpha=0.9,
                label=f"neg mean (n={len(neg_idx)})", zorder=4)

    # Fisher peak star (on pos bundle)
    peak = int(np.argmax(fisher_scores))   # 0-indexed
    ax.scatter(bm_pos[peak, 0], bm_pos[peak, 1], bm_pos[peak, 2],
               color="gold", marker="*", s=280, edgecolors='darkorange', lw=1.5,
               zorder=9, label=f"peak L{peak+1}")

    # Title
    title_color = '#B71C1C' if is_mid else '#1A237E'
    mid_tag = ' [MID]' if is_mid else ''
    ax.set_title(f"{go_name}{mid_tag}\n(pos={len(pos_idx)}  peak=L{peak+1})",
                 fontsize=9, color=title_color,
                 fontweight='bold' if is_mid else 'normal', pad=6)
    ax.set_xlabel('PC1', fontsize=7, labelpad=-2)
    ax.set_ylabel('PC2', fontsize=7, labelpad=-2)
    ax.set_zlabel('PC3', fontsize=7, labelpad=-2)
    ax.tick_params(labelsize=6, pad=0)

    # View angle: match fluid_stage2 style (elevation ≈ 20°, azimuth ≈ -60°)
    ax.view_init(elev=20, azim=-60)

    # Show legend only in first panel
    if is_first:
        ax.legend(fontsize=6, loc='upper left', framealpha=0.85)


def make_fig(Z, sym_te, fisher, title_suffix, fname_prefix, normalize_layers=False):
    """Split 18 GOs into 3 figures (6 GOs each, 2x3 grid, larger subplots)."""
    if normalize_layers:
        Z = Z.copy()
        for L in range(30):
            mu = Z[:, L, :].mean(0)
            sd = Z[:, L, :].std(0) + 1e-8
            Z[:, L, :] = (Z[:, L, :] - mu) / sd

    go_list = list(GO_18.keys())
    groups = [go_list[0:6], go_list[6:12], go_list[12:18]]

    for gi, gos in enumerate(groups):
        fig = plt.figure(figsize=(24, 14))
        fig.suptitle(
            f"ESM-2 Rectified Trajectory Flow (3D PC1-PC2-PC3)  —  "
            f"Group {gi+1}/3  —  {title_suffix}\n"
            f"Blue solid = pos bundle mean  |  Crimson dashed = neg bundle mean  |  "
            f"Gray = {N_CONTEXT} individual pos trajectories  |  "
            f"O=L1 start  X=L30 end  *=Fisher peak layer",
            fontsize=13, fontweight='bold', y=0.99
        )
        for i, go in enumerate(gos):
            ax = fig.add_subplot(2, 3, i + 1, projection='3d')
            y_mask = load_labels(go, sym_te)
            draw_panel(ax, Z, y_mask, fisher[go],
                       GO_18[go], go in MID_GOs,
                       is_first=(gi == 0 and i == 0))

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_png = OUT_DIR / f"{fname_prefix}_g{gi+1}.png"
        fig.savefig(out_png, dpi=140, bbox_inches='tight')
        fig.savefig(out_png.with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f"[saved] {out_png}")


def main():
    print("[plot_trajectory_flow_3d] Loading data...")

    te_gene = np.load(MODEL_DIR / "my_gene_list_fixed.npy", allow_pickle=True)
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

    def clean(raw):
        s = str(raw)
        for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
        return s

    sym_te = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
              for g in te_gene]

    print(f"  Loading Z_te ...")
    Z = np.load(CACHE_DIR / "Z_te.npy")   # (36748, 30, 8)
    print(f"  Z_te shape: {Z.shape}")

    print("  Loading Fisher scores...")
    fisher = load_fisher()

    print("\n[1] 3D bundle-tube — Non-normalized (raw Z from joint PCA)...")
    make_fig(Z, sym_te, fisher,
             title_suffix="Non-normalized (joint PCA space, raw Z)",
             fname_prefix="fig_flow3d_nonnorm",
             normalize_layers=False)

    print("\n[2] 3D bundle-tube — Normalized (per-layer z-score)...")
    make_fig(Z, sym_te, fisher,
             title_suffix="Normalized (per-layer z-score across all proteins)",
             fname_prefix="fig_flow3d_norm",
             normalize_layers=True)

    print("\n[done] 6 figures saved to", OUT_DIR)
    print("  fig_flow3d_nonnorm_g{1,2,3}.png  — 3D bundle tube (raw Z), 6 GOs each")
    print("  fig_flow3d_norm_g{1,2,3}.png     — 3D bundle tube (z-scored), 6 GOs each")


if __name__ == "__main__":
    main()
