"""
plot_trajectory_flow.py
=======================
18 GO term별 ESM-2 rectified trajectory flow 시각화.

핵심 아이디어:
  ESM-2 30 layers = Euler discretization of ODE dX/dt = v(X,t)
  Z[i, L, k] = PCA(z-score(X_L^(i))) → 각 단백질의 per-layer 궤적 좌표

시각화 두 버전:
  [Non-normalized] Z_te 그대로 사용 (이미 joint PCA로 공통 공간에 투영)
    → 각 GO의 positive/negative 단백질 그룹별 평균 궤적 (PC1 + PC2)
    → 레이어별 절대 위치 표현

  [Normalized] 각 레이어 L에서 전체 단백질 기준 per-layer z-score 적용
    Z_norm[i,L,k] = (Z[i,L,k] - mean_L(k)) / std_L(k)
    → 레이어 간 스케일 차이 제거, 그룹별 상대적 편차만 표현

각 패널 구성:
  - 상단 bar: Fisher per-layer LR AUPRC (레이어별 판별력)
  - 중단 line: pos/neg 그룹별 평균 PC1 궤적
  - 우측 line (inset): 평균 PC2 궤적
  - 하이라이트: window w=7 선택 영역 (GO별 peak ± 7)
  - MID GO 표시 (빨간 타이틀)
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
DATA      = ROOT / "data"
MODEL_DIR = ROOT / "model"
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
LAYERS  = np.arange(1, 31)
W       = 7   # highlight window


def load_fisher() -> dict[str, np.ndarray]:
    all_lr = {}
    for fname in ["layer_probe_v15d_terms_results.json",
                  "layer_probe_expanded_results.json",
                  "layer_probe_results.json"]:
        p = PROBE_DIR / fname
        if p.exists():
            d = json.load(open(p))
            all_lr.update(d["lr_auprc"])
    return {go: np.array(all_lr[go], dtype=np.float32) for go in GO_18}


def load_labels(go, sym_te) -> np.ndarray:
    pos = set()
    with open(ANNOT_DIR / "human_annotations_unified_bp.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 1 and go in parts[1:]:
                pos.add(parts[0])
    return np.array([s in pos for s in sym_te], dtype=bool)


def make_fig(Z: np.ndarray, sym_te, fisher, title_suffix: str, fname: str,
             normalize_layers: bool = False):
    """
    Z: (N, 30, 8)
    normalize_layers: True → per-layer z-score before plotting
    """
    if normalize_layers:
        Z = Z.copy()
        for L in range(30):
            mu = Z[:, L, :].mean(0)
            sd = Z[:, L, :].std(0) + 1e-8
            Z[:, L, :] = (Z[:, L, :] - mu) / sd

    # ── 3×6 grid ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(
        f"ESM-2 Rectified Trajectory Flow per GO — {title_suffix}\n"
        f"Solid=positive proteins, Dashed=negative proteins  |  "
        f"Window ±{W} (highlight) = v20b GO-specific selection",
        fontsize=12, fontweight='bold', y=1.01
    )

    gs_outer = gridspec.GridSpec(3, 6, figure=fig, hspace=0.55, wspace=0.35)

    go_list = list(GO_18.keys())

    for idx, go in enumerate(go_list):
        row, col = divmod(idx, 6)
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_outer[row, col],
            height_ratios=[1, 3], hspace=0.08
        )
        ax_fish = fig.add_subplot(gs_inner[0])   # Fisher bar (top)
        ax_traj = fig.add_subplot(gs_inner[1])   # Trajectory line (bottom)

        go_name = GO_18[go]
        fs      = fisher[go]                      # (30,) Fisher AUPRC per layer
        peak    = int(np.argmax(fs))              # 0-indexed
        is_mid  = go in MID_GOs

        # ── Labels ───────────────────────────────────────────────────
        y_mask = load_labels(go, sym_te)          # bool (N,)
        n_pos  = y_mask.sum()
        n_neg  = (~y_mask).sum()

        if n_pos == 0:
            ax_fish.axis('off'); ax_traj.axis('off')
            continue

        # ── Per-layer mean PC1 & PC2 ─────────────────────────────────
        Z_pos = Z[y_mask]     # (n_pos, 30, 8)
        Z_neg = Z[~y_mask]    # (n_neg, 30, 8)

        mean_pos_pc1 = Z_pos[:, :, 0].mean(0)   # (30,)
        mean_neg_pc1 = Z_neg[:, :, 0].mean(0)
        std_pos_pc1  = Z_pos[:, :, 0].std(0)
        std_neg_pc1  = Z_neg[:, :, 0].std(0)

        mean_pos_pc2 = Z_pos[:, :, 1].mean(0)
        mean_neg_pc2 = Z_neg[:, :, 1].mean(0)

        # ── Window highlight ─────────────────────────────────────────
        lo = max(0, peak - W)
        hi = min(29, peak + W)

        # ── Fisher bar (top) ────────────────────────────────────────
        bar_col = ['#EF5350' if is_mid else '#90A4AE'] * 30
        for L in range(lo, hi + 1):
            bar_col[L] = '#FF7043' if is_mid else '#5C6BC0'
        ax_fish.bar(LAYERS, fs, color=bar_col, width=0.8, alpha=0.85)
        ax_fish.axvline(peak + 1, color='red', lw=1.0, alpha=0.7)
        ax_fish.set_xlim(0.5, 30.5)
        ax_fish.tick_params(labelbottom=False, labelsize=5)
        ax_fish.set_ylabel('Fisher\nAUPRC', fontsize=5, labelpad=1)
        ax_fish.yaxis.set_major_locator(plt.MaxNLocator(2))

        title_color = '#B71C1C' if is_mid else '#1A237E'
        mid_tag = ' [MID]' if is_mid else ''
        ax_fish.set_title(f"{go_name}{mid_tag}\n(pos={n_pos})",
                          fontsize=7, color=title_color,
                          fontweight='bold' if is_mid else 'normal', pad=2)

        # ── Trajectory (bottom) ──────────────────────────────────────
        # Window shade
        ax_traj.axvspan(lo + 0.5, hi + 1.5, alpha=0.12,
                        color='#FF7043' if is_mid else '#5C6BC0',
                        label=f'win±{W}')

        # PC1 trajectories
        ax_traj.plot(LAYERS, mean_pos_pc1, color='#E53935', lw=1.8,
                     label=f'pos PC1 (n={n_pos})', zorder=3)
        ax_traj.fill_between(LAYERS,
                              mean_pos_pc1 - 0.3 * std_pos_pc1,
                              mean_pos_pc1 + 0.3 * std_pos_pc1,
                              color='#E53935', alpha=0.12, zorder=2)
        ax_traj.plot(LAYERS, mean_neg_pc1, color='#1565C0', lw=1.4,
                     ls='--', label=f'neg PC1 (n={n_neg})', zorder=3)
        ax_traj.fill_between(LAYERS,
                              mean_neg_pc1 - 0.3 * std_neg_pc1,
                              mean_neg_pc1 + 0.3 * std_neg_pc1,
                              color='#1565C0', alpha=0.08, zorder=2)

        # PC2 as thin dotted lines (secondary)
        ax_traj.plot(LAYERS, mean_pos_pc2, color='#EF9A9A', lw=0.9,
                     ls=':', alpha=0.8, label='pos PC2')
        ax_traj.plot(LAYERS, mean_neg_pc2, color='#90CAF9', lw=0.9,
                     ls=':', alpha=0.8, label='neg PC2')

        # Peak marker
        ax_traj.axvline(peak + 1, color='gray', lw=0.8, alpha=0.5, ls='--')

        ax_traj.set_xlim(0.5, 30.5)
        ax_traj.set_xlabel('ESM-2 Layer', fontsize=6)
        ax_traj.set_ylabel('PC value', fontsize=6)
        ax_traj.tick_params(labelsize=5)
        ax_traj.yaxis.set_major_locator(plt.MaxNLocator(4))

        if idx == 0:
            ax_traj.legend(fontsize=4.5, loc='upper left',
                           ncol=2, framealpha=0.7)

    plt.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
    plt.savefig(OUT_DIR / fname.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[saved] {OUT_DIR}/{fname}")


def make_fig_2d(Z: np.ndarray, sym_te, fisher, title_suffix: str, fname: str,
                normalize_layers: bool = False):
    """
    2D PC1-PC2 trajectory plot: layer를 색상으로 표현 (flow 방향 시각화).
    """
    if normalize_layers:
        Z = Z.copy()
        for L in range(30):
            mu = Z[:, L, :].mean(0)
            sd = Z[:, L, :].std(0) + 1e-8
            Z[:, L, :] = (Z[:, L, :] - mu) / sd

    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(
        f"ESM-2 Trajectory Flow — PC1 vs PC2 Space — {title_suffix}\n"
        f"Color = Layer (L1=dark → L30=bright)  |  Circle=pos, Triangle=neg  |  Arrow=flow direction",
        fontsize=12, fontweight='bold', y=1.01
    )

    cmap   = plt.get_cmap('plasma')
    norm_c = Normalize(vmin=1, vmax=30)
    go_list = list(GO_18.keys())

    axes = []
    for idx in range(18):
        ax = fig.add_subplot(3, 6, idx + 1)
        axes.append(ax)

    for idx, go in enumerate(go_list):
        ax = axes[idx]
        go_name = GO_18[go]
        fs      = fisher[go]
        peak    = int(np.argmax(fs))
        is_mid  = go in MID_GOs

        y_mask = load_labels(go, sym_te)
        n_pos  = y_mask.sum()
        if n_pos == 0:
            ax.axis('off'); continue

        Z_pos = Z[y_mask]
        Z_neg = Z[~y_mask]

        traj_pos = Z_pos[:, :, :2].mean(0)   # (30, 2) mean trajectory in PC1-PC2
        traj_neg = Z_neg[:, :, :2].mean(0)

        # Plot trajectories: scatter + arrows
        for L in range(30):
            c = cmap(norm_c(L + 1))
            # Position dots
            ax.scatter(traj_pos[L, 0], traj_pos[L, 1],
                       color=c, s=28, marker='o', zorder=3, linewidths=0)
            ax.scatter(traj_neg[L, 0], traj_neg[L, 1],
                       color=c, s=18, marker='^', zorder=3,
                       alpha=0.7, linewidths=0)

        # Draw flow arrows every 5 layers
        for L in range(0, 28, 4):
            c = cmap(norm_c(L + 1))
            dx_p = traj_pos[L+1, 0] - traj_pos[L, 0]
            dy_p = traj_pos[L+1, 1] - traj_pos[L, 1]
            ax.annotate('', xy=(traj_pos[L+1, 0], traj_pos[L+1, 1]),
                        xytext=(traj_pos[L, 0], traj_pos[L, 1]),
                        arrowprops=dict(arrowstyle='->', color=c,
                                        lw=1.2, mutation_scale=8))

        # Connect dots with thin line
        ax.plot(traj_pos[:, 0], traj_pos[:, 1], color='#E53935', lw=0.7,
                alpha=0.5, zorder=2)
        ax.plot(traj_neg[:, 0], traj_neg[:, 1], color='#1565C0', lw=0.7,
                ls='--', alpha=0.4, zorder=2)

        # Mark peak layer
        ax.scatter(traj_pos[peak, 0], traj_pos[peak, 1],
                   s=80, marker='*', color='gold',
                   edgecolors='darkorange', lw=0.8, zorder=5,
                   label=f'peak L{peak+1}')

        title_color = '#B71C1C' if is_mid else '#1A237E'
        mid_tag = ' [MID]' if is_mid else ''
        ax.set_title(f"{go_name}{mid_tag}", fontsize=7,
                     color=title_color,
                     fontweight='bold' if is_mid else 'normal')
        ax.set_xlabel('PC1', fontsize=6)
        ax.set_ylabel('PC2', fontsize=6)
        ax.tick_params(labelsize=5)

        if idx == 0:
            ax.legend(fontsize=5, loc='upper right')

    # Colorbar for layer
    sm = ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.01, fraction=0.01)
    cbar.set_label('ESM-2 Layer', fontsize=9)
    cbar.set_ticks([1, 10, 20, 30])

    plt.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
    plt.savefig(OUT_DIR / fname.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[saved] {OUT_DIR}/{fname}")


def main():
    print("[plot_trajectory_flow] Loading data...")

    # IDs
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

    sym_te = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in te_gene]

    # Z cache
    print("  Loading Z_te (36748, 30, 8)...")
    Z = np.load(CACHE_DIR / "Z_te.npy")   # (36748, 30, 8)

    # Fisher
    print("  Loading Fisher scores...")
    fisher = load_fisher()

    # ── Figure set 1: Layer-line plots (PC1 + PC2 vs layer) ──────────
    print("\n[1] Layer-line trajectory — Non-normalized...")
    make_fig(Z, sym_te, fisher,
             title_suffix="Non-normalized (joint PCA space, raw Z)",
             fname="fig_traj_line_nonnorm.png",
             normalize_layers=False)

    print("[2] Layer-line trajectory — Normalized (per-layer z-score)...")
    make_fig(Z, sym_te, fisher,
             title_suffix="Normalized (per-layer z-score across proteins)",
             fname="fig_traj_line_norm.png",
             normalize_layers=True)

    # ── Figure set 2: 2D PC1-PC2 flow plots ─────────────────────────
    print("\n[3] 2D flow — Non-normalized...")
    make_fig_2d(Z, sym_te, fisher,
                title_suffix="Non-normalized",
                fname="fig_traj_2d_nonnorm.png",
                normalize_layers=False)

    print("[4] 2D flow — Normalized (per-layer z-score)...")
    make_fig_2d(Z, sym_te, fisher,
                title_suffix="Normalized",
                fname="fig_traj_2d_norm.png",
                normalize_layers=True)

    print("\n[done] 4 figures saved to", OUT_DIR)
    print("  fig_traj_line_nonnorm.png  — PC1/PC2 vs layer (raw Z)")
    print("  fig_traj_line_norm.png     — PC1/PC2 vs layer (per-layer z-scored)")
    print("  fig_traj_2d_nonnorm.png    — 2D PC1-PC2 flow (raw Z)")
    print("  fig_traj_2d_norm.png       — 2D PC1-PC2 flow (per-layer z-scored)")


if __name__ == "__main__":
    main()
