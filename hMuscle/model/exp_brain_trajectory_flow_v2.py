"""
exp_brain_trajectory_flow_v2.py
================================
v2 개선:
1. Pair grid에 pairwise distance-vs-layer inset 추가
2. Bracket bundle: negative reference 축소, peak layer 강조
3. Case 요약을 서브패널에 함께 표기
"""
from __future__ import annotations

import json, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from pathlib import Path
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family'    : ['DejaVu Sans'],
    'font.size'      : 10,
    'axes.titlesize' : 11,
    'axes.labelsize' : 10,
    'axes.unicode_minus': False,
})

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
BRAIN_DIR = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
BRAIN_LABEL_DIR = BRAIN_DIR
OUT_DIR = ROOT / "reports" / "brain_flow"
IN_JSON = OUT_DIR / "brain_case_summary.json"
Z_PATH = OUT_DIR / "brain_Z_te_30x8.npy"

N_LAYERS = 30

GO_18 = {
    "GO:0007204": "Ca2+ signaling",        "GO:0045214": "Sarcomere org",
    "GO:0006941": "Muscle contraction",    "GO:0006914": "Autophagy",
    "GO:0043161": "Proteasome / UPS",      "GO:0007519": "Skeletal muscle dev",
    "GO:0042692": "Muscle cell diff",      "GO:0055074": "Ca2+ homeostasis",
    "GO:0007005": "Mitochondrion org",     "GO:0007517": "Muscle organ dev",
    "GO:0032006": "TOR signaling",         "GO:0030048": "Actin-based movement",
    "GO:0006096": "Glycolysis",            "GO:0007268": "Synaptic transmission",
    "GO:0007018": "Microtubule movement",  "GO:0031175": "Neuron proj development",
    "GO:0030182": "Neuron differentiation","GO:0000226": "MT cytoskeleton org",
}


# ── Pair grid with distance inset ─────────────────────────────────

def plot_pair_grid(Z, pairs, kind, out_name):
    fig = plt.figure(figsize=(20, 12))
    if kind == "conv":
        header = ("Brain — Convergent evolution   |   "
                  "Two isoforms from different genes converge to the same GO subspace   |   "
                  "Inset: pairwise distance vs layer (red dot = convergence layer)")
        col_a = '#1E88E5'; col_b = '#43A047'
    else:
        header = ("Brain — Functional divergence   |   "
                  "Two isoforms from different genes diverge into different GO subspaces   |   "
                  "Inset: pairwise distance vs layer (red dot = divergence layer)")
        col_a = '#1E88E5'; col_b = '#D81B60'
    fig.suptitle(header, fontsize=13, fontweight='bold', y=0.995)

    for i, p in enumerate(pairs[:6]):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        pts_a = Z[p["i_a"], :, :3]
        pts_b = Z[p["i_b"], :, :3]

        # Per-layer pairwise distance
        d_seq = np.sqrt(((Z[p["i_a"]] - Z[p["i_b"]]) ** 2).sum(-1))
        if kind == "conv":
            # convergence layer: first layer where distance < 50% of L1
            target = d_seq[0] * 0.5
            conv_L = -1
            for L in range(1, 30):
                if d_seq[L] < target:
                    conv_L = L + 1; break
        else:
            # divergence layer: first layer where distance > 2× L1
            target = d_seq[0] * 2.0
            conv_L = -1
            for L in range(1, 30):
                if d_seq[L] > target:
                    conv_L = L + 1; break

        # 3D trajectories
        ax.plot(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2],
                color=col_a, lw=2.4, alpha=0.95, label='Iso A', zorder=6)
        ax.plot(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2],
                color=col_b, lw=2.4, alpha=0.95, label='Iso B', zorder=6)
        for pts, col in [(pts_a, col_a), (pts_b, col_b)]:
            ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2],
                       color=col, marker='o', s=90,
                       edgecolors='white', lw=1.2, zorder=8)
            ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                       color=col, marker='X', s=130,
                       edgecolors='white', lw=1.2, zorder=8)

        # Key layer marker
        if conv_L > 0:
            L_idx = conv_L - 1
            for pts in [pts_a, pts_b]:
                ax.scatter(pts[L_idx, 0], pts[L_idx, 1], pts[L_idx, 2],
                           color='red', marker='*', s=220,
                           edgecolors='black', lw=1.2, zorder=10)

        # Title
        if kind == "conv":
            gene_a = p.get('gene_a', '?')
            gene_b = p.get('gene_b', '?')
            go_name = p.get('name', '?')
            title = (f"{gene_a} vs {gene_b}   ({go_name})\n"
                     f"d(L1)={p['d_L1']:.2f} → d(L30)={p['d_L30']:.2f}   "
                     f"conv@L{conv_L}   ratio={p['d_L30']/p['d_L1']:.3f}")
        else:
            go_a_name = GO_18.get(p.get('gos_a', ['?'])[0], '?')
            go_b_name = GO_18.get(p.get('gos_b', ['?'])[0], '?')
            title = (f"{p['gene_a']} ({go_a_name})\n"
                     f"vs {p['gene_b']} ({go_b_name})\n"
                     f"d(L1)={p['d_L1']:.2f} → d(L30)={p['d_L30']:.2f}   "
                     f"div@L{conv_L}   ratio={p['d_L30']/p['d_L1']:.2f}×")
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel('PC1', fontsize=8, labelpad=-2)
        ax.set_ylabel('PC2', fontsize=8, labelpad=-2)
        ax.set_zlabel('PC3', fontsize=8, labelpad=-2)
        ax.tick_params(labelsize=6, pad=0)
        ax.view_init(elev=20, azim=-60)
        if i == 0:
            ax.legend(fontsize=8, loc='upper left', framealpha=0.85)

        # Distance-vs-layer inset
        ax_ins = fig.add_axes([
            ax.get_position().x1 - 0.055,
            ax.get_position().y0 + 0.015,
            0.06, 0.05
        ])
        ax_ins.plot(range(1, 31), d_seq, color='#37474F', lw=1.4)
        if conv_L > 0:
            ax_ins.scatter([conv_L], [d_seq[conv_L - 1]],
                            color='red', s=35, zorder=6, edgecolors='black', lw=0.5)
        ax_ins.set_xlim(1, 30)
        ax_ins.tick_params(labelsize=5, pad=0)
        ax_ins.set_title('d(a,b) vs L', fontsize=6, pad=1)
        ax_ins.grid(True, alpha=0.35, linestyle=':')

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT_DIR / out_name, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_name}")


# ── Bracket bundle (cleaner) ──────────────────────────────────────

def load_labels_gene():
    lab = np.load(BRAIN_LABEL_DIR / "brain_full_labels.npy")
    gene_names = np.load(BRAIN_LABEL_DIR / "brain_full_gene_names.npy",
                          allow_pickle=True)
    mask = np.load(BRAIN_LABEL_DIR / "brain_full_mask.npy")
    coding_idx = np.where(mask == 1)[0]
    return lab, gene_names, coding_idx


def draw_bundle_panel(ax, Z, coding_idx, labels, go, name, peak_L,
                      n_context=8, show_neg=False):
    gi = list(GO_18.keys()).index(go)
    pos = np.where(labels[:, gi] == 1)[0]
    pos = np.intersect1d(pos, coding_idx)
    neg = np.where(labels[:, gi] == 0)[0]
    neg = np.intersect1d(neg, coding_idx)
    if len(pos) < 5:
        ax.set_axis_off(); return

    # Ensure enough negatives sampled but only draw a few
    ctx_pos = rng.choice(pos, size=min(n_context, len(pos)), replace=False)

    for oi in ctx_pos:
        pts = Z[oi, :, :3]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color='#4FC3F7', lw=0.6, alpha=0.5, zorder=2)

    # Positive bundle mean (thick blue)
    bm_pos = Z[pos, :, :3].mean(axis=0)
    ax.plot(bm_pos[:, 0], bm_pos[:, 1], bm_pos[:, 2],
            color='#0D47A1', lw=3.0, alpha=0.98, zorder=5,
            label=f'pos bundle (n={len(pos)})')

    # Optional negative bundle mean (dashed red, thin)
    if show_neg and len(neg) > 20:
        neg_sub = rng.choice(neg, size=min(2000, len(neg)), replace=False)
        bm_neg = Z[neg_sub, :, :3].mean(axis=0)
        ax.plot(bm_neg[:, 0], bm_neg[:, 1], bm_neg[:, 2],
                color='#C62828', lw=1.5, ls='--', alpha=0.8, zorder=4,
                label=f'neg bundle (n={len(neg_sub)})')

    # Peak layer marker
    ax.scatter(bm_pos[peak_L - 1, 0], bm_pos[peak_L - 1, 1], bm_pos[peak_L - 1, 2],
               color='#FFC107', marker='*', s=340,
               edgecolors='darkorange', lw=1.6, zorder=8,
               label=f'peak L{peak_L}')

    # Start / end markers
    ax.scatter(bm_pos[0, 0], bm_pos[0, 1], bm_pos[0, 2],
               color='#0D47A1', marker='o', s=90,
               edgecolors='white', lw=1.2, zorder=7)
    ax.scatter(bm_pos[-1, 0], bm_pos[-1, 1], bm_pos[-1, 2],
               color='#0D47A1', marker='X', s=140,
               edgecolors='white', lw=1.2, zorder=7)

    # Layer labels at 1, peak_L, 30
    for L_ann in [1, peak_L, 30]:
        j = L_ann - 1
        ax.text(bm_pos[j, 0], bm_pos[j, 1], bm_pos[j, 2],
                f'  L{L_ann}', fontsize=8, color='#0D47A1',
                fontweight='bold', zorder=11)

    ax.set_title(f'{name}   peak L{peak_L}  (n_pos={len(pos)})',
                 fontsize=11, pad=4)
    ax.set_xlabel('PC1', fontsize=9, labelpad=-2)
    ax.set_ylabel('PC2', fontsize=9, labelpad=-2)
    ax.set_zlabel('PC3', fontsize=9, labelpad=-2)
    ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=20, azim=-60)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)


def plot_bracket_v2(Z, coding_idx, labels, bracket_go, out_name):
    """bracket_go: dict {Early/Mid/Late: [(go, peak_L, name), ...]}"""
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Brain — Early / Mid / Late layer signal-specialized GO bundles\n'
                 'Thick blue = positive-class bundle mean trajectory   |   '
                 'Dashed red = negative reference bundle mean',
                 fontsize=14, fontweight='bold', y=0.995)

    order = ['Early (L1-10)', 'Mid (L11-20)', 'Late (L21-30)']
    for row, bkey in enumerate(order):
        entries = bracket_go[bkey][:2]
        for col, e in enumerate(entries):
            ax = fig.add_subplot(3, 2, row * 2 + col + 1, projection='3d')
            draw_bundle_panel(ax, Z, coding_idx, labels,
                              e['go'], e['name'], e['peak_layer'],
                              n_context=6, show_neg=True)

    # Row labels
    for i, bkey in enumerate(order):
        fig.text(0.01, 0.83 - i * 0.30, bkey, fontsize=14, fontweight='bold',
                 rotation=90, va='center', ha='left', color='#333')

    fig.tight_layout(rect=[0.03, 0.02, 1, 0.96])
    fig.savefig(OUT_DIR / out_name, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_name}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("[1] Loading Z, labels, cases…")
    Z = np.load(Z_PATH)
    print(f"  Z: {Z.shape}")
    labels, gene_names, coding_idx = load_labels_gene()

    summary = json.load(open(IN_JSON))
    conv = summary["convergent_top"]
    div = summary["divergent_top"]
    bracket_raw = summary["bracket_top2"]

    print("[2] Plot conv pairs v2 …")
    plot_pair_grid(Z, conv, "conv", "fig_brain_convergent_pairs_v2.png")

    if div:
        print("[3] Plot div pairs v2 …")
        plot_pair_grid(Z, div, "div", "fig_brain_divergent_pairs_v2.png")

    print("[4] Plot bracket bundle v2 …")
    plot_bracket_v2(Z, coding_idx, labels, bracket_raw,
                     "fig_brain_bundle_by_bracket_v2.png")

    print(f"[done] {time.time()-t0:.1f}s. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
