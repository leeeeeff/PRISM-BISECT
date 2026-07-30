"""
plot_convdiv_per_case.py
========================
각 convergent/divergent 케이스별로 하나의 figure에 4개 sub-panel:

  [Top-Left]  Non-normalized 3D trajectory flow
              start/end markers + Fisher peak + conv/div layer
  [Top-Right] Non-normalized per-layer Fisher signal for isoforms' GOs
  [Bot-Left]  Normalized (per-layer z-score) 3D trajectory flow
  [Bot-Right] Normalized (0-1 per GO) per-layer Fisher signal

총 6 convergent + 6 divergent = 12 figures.
"""
from __future__ import annotations

import json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from pathlib import Path
warnings.filterwarnings('ignore')

SEED = 42
ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
DATA      = ROOT / "data"
ID_DIR    = DATA / "raw_data/data/id_lists"
ANNOT_DIR = DATA / "raw_data/data/annotations"
PROBE_DIR = ROOT.parent / "reports" / "layer_probe"
CACHE_DIR = ROOT.parent / "reports" / "v20_cache"
OUT_DIR   = ROOT.parent / "reports" / "curve_sweep" / "cases"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GO_18 = {
    "GO:0007204": "Ca2+ signal",           "GO:0045214": "Sarcomere org",
    "GO:0006941": "Muscle contract",       "GO:0006914": "Autophagy",
    "GO:0043161": "Proteasome-UPS",        "GO:0007519": "Skeletal musc dev",
    "GO:0042692": "Muscle cell diff",      "GO:0055074": "Ca2+ homeostasis",
    "GO:0007005": "Mitochondrion org",     "GO:0007517": "Muscle organ dev",
    "GO:0032006": "TOR signaling",         "GO:0030048": "Actin-based mov",
    "GO:0006096": "Glycolysis",            "GO:0007268": "Synaptic transm",
    "GO:0007018": "MT-based mov",          "GO:0031175": "Neuron proj dev",
    "GO:0030182": "Neuron diff",           "GO:0000226": "MT cytosk org",
}

GO_PALETTE = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA',
              '#00ACC1', '#795548', '#EC407A', '#FFB300', '#5E35B1']


def load_all():
    te_gene = np.load(MODEL_DIR / "my_gene_list_fixed.npy", allow_pickle=True)
    te_iso  = np.load(MODEL_DIR / "my_isoform_list_fixed.npy", allow_pickle=True)
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
    iso_te = [clean(i) for i in te_iso]
    Z = np.load(CACHE_DIR / "Z_te.npy")

    go_labels = {}
    for go in GO_18:
        pos = set()
        with open(ANNOT_DIR / "human_annotations_unified_bp.txt") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > 1 and go in parts[1:]:
                    pos.add(parts[0])
        go_labels[go] = np.array([s in pos for s in sym_te], dtype=bool)

    fisher_raw = {}
    for fname in ["layer_probe_v15d_terms_results.json",
                  "layer_probe_expanded_results.json",
                  "layer_probe_results.json"]:
        p = PROBE_DIR / fname
        if p.exists():
            d = json.load(open(p))
            for k, v in d["lr_auprc"].items():
                if k in GO_18:
                    fisher_raw[k] = np.array(v, dtype=np.float32)

    return Z, sym_te, iso_te, go_labels, fisher_raw


def isoform_gos(i, go_labels):
    return [go for go, labels in go_labels.items() if labels[i]]


def normalize_z(Z):
    Z2 = Z.copy()
    for L in range(30):
        mu = Z2[:, L, :].mean(0)
        sd = Z2[:, L, :].std(0) + 1e-8
        Z2[:, L, :] = (Z2[:, L, :] - mu) / sd
    return Z2


# ── Drawing helpers ────────────────────────────────────────────────

def draw_traj_3d(ax, pts_a, pts_b, conv_or_div_L, fisher_peak_L,
                 label_a, label_b, kind, title_prefix):
    """3D trajectory subplot."""
    # Trajectory A (blue)
    ax.plot(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2],
            color='tab:blue', lw=2.2, label=f'A: {label_a}', zorder=4)
    ax.scatter(pts_a[0, 0], pts_a[0, 1], pts_a[0, 2],
               color='tab:blue', marker='o', s=90, edgecolors='white', lw=1.0,
               zorder=6, label='A: L1 start')
    ax.scatter(pts_a[-1, 0], pts_a[-1, 1], pts_a[-1, 2],
               color='tab:blue', marker='X', s=130, edgecolors='white', lw=1.0,
               zorder=6, label='A: L30 end')

    # Trajectory B (green)
    ax.plot(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2],
            color='seagreen', lw=2.2, label=f'B: {label_b}', zorder=4)
    ax.scatter(pts_b[0, 0], pts_b[0, 1], pts_b[0, 2],
               color='seagreen', marker='o', s=90, edgecolors='white', lw=1.0,
               zorder=6, label='B: L1 start')
    ax.scatter(pts_b[-1, 0], pts_b[-1, 1], pts_b[-1, 2],
               color='seagreen', marker='X', s=130, edgecolors='white', lw=1.0,
               zorder=6, label='B: L30 end')

    # Fisher peak marker (both trajectories, gold)
    if 0 < fisher_peak_L <= 30:
        Lp = fisher_peak_L - 1
        for pts in (pts_a, pts_b):
            ax.scatter(pts[Lp, 0], pts[Lp, 1], pts[Lp, 2],
                       color='gold', marker='*', s=240,
                       edgecolors='darkorange', lw=1.2, zorder=7)
        ax.plot([], [], color='gold', marker='*', ms=12, ls='',
                markeredgecolor='darkorange',
                label=f'Fisher peak L{fisher_peak_L}')

    # Conv/Div layer marker (red star with black edge)
    if conv_or_div_L > 0:
        Lc = conv_or_div_L - 1
        for pts in (pts_a, pts_b):
            ax.scatter(pts[Lc, 0], pts[Lc, 1], pts[Lc, 2],
                       color='red', marker='*', s=240,
                       edgecolors='black', lw=1.2, zorder=8)
        # Draw connecting line at conv/div layer
        ax.plot([pts_a[Lc, 0], pts_b[Lc, 0]],
                [pts_a[Lc, 1], pts_b[Lc, 1]],
                [pts_a[Lc, 2], pts_b[Lc, 2]],
                color='red', lw=1.5, alpha=0.7, ls=':', zorder=5)
        tag = 'conv' if kind == 'conv' else 'div'
        ax.plot([], [], color='red', marker='*', ms=12, ls='',
                markeredgecolor='black',
                label=f'{tag} layer L{conv_or_div_L}')

    # Layer number annotations (L5, L10, L15, L20, L25)
    for L in [5, 10, 15, 20, 25]:
        j = L - 1
        ax.text(pts_a[j, 0], pts_a[j, 1], pts_a[j, 2],
                f' L{L}', fontsize=6, color='navy', alpha=0.75)

    ax.set_title(title_prefix, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)
    ax.set_zlabel('PC3', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=20, azim=-60)
    ax.legend(fontsize=6.5, loc='upper left', framealpha=0.85)


def draw_fisher_panel(ax, gos_a, gos_b, fisher_raw, conv_or_div_L,
                      normalized: bool, title_prefix):
    """Per-layer Fisher signal for the GOs each isoform is annotated with."""
    all_gos = list(dict.fromkeys(gos_a + gos_b))   # preserve order, deduplicate
    if not all_gos:
        ax.text(0.5, 0.5, 'No GO annotations from the 18 tracked terms',
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title(title_prefix, fontsize=11, fontweight='bold')
        return

    layers = np.arange(1, 31)
    peak_layers = []

    for gi, go in enumerate(all_gos):
        curve = fisher_raw[go].copy()
        if normalized:
            curve = (curve - curve.min()) / (curve.max() - curve.min() + 1e-9)

        color = GO_PALETTE[gi % len(GO_PALETTE)]
        # membership tag: A only / B only / both
        in_a, in_b = go in gos_a, go in gos_b
        if in_a and in_b:
            tag, ls = 'A∩B', '-'
        elif in_a:
            tag, ls = 'A only', '--'
        else:
            tag, ls = 'B only', ':'

        ax.plot(layers, curve, color=color, lw=2.0, ls=ls,
                label=f'{tag} · {go} {GO_18[go]}')
        peak = int(np.argmax(fisher_raw[go])) + 1
        peak_layers.append((go, peak, curve[peak-1]))
        ax.scatter([peak], [curve[peak-1]], color=color, marker='*',
                   s=140, edgecolors='black', lw=0.8, zorder=6)

    # Conv/Div layer vertical line
    if conv_or_div_L > 0:
        ax.axvline(conv_or_div_L, color='red', lw=1.5, alpha=0.6, ls='-.',
                   label=f'conv/div layer L{conv_or_div_L}')

    ax.set_xlabel('ESM-2 Layer', fontsize=9)
    ax.set_ylabel(('normalized Fisher (0-1 per GO)' if normalized
                   else 'Fisher (LR AUPRC per layer)'),
                  fontsize=9)
    ax.set_xlim(0.5, 30.5)
    ax.set_xticks(np.arange(1, 31, 3))
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best', framealpha=0.85)
    ax.set_title(title_prefix, fontsize=11, fontweight='bold', pad=6)


# ── Case figure builder ────────────────────────────────────────────

def make_case_figure(p, Z, Z_norm, sym_te, iso_te, go_labels, fisher_raw,
                     kind: str, case_idx: int):
    """Build one 2×2 figure for a single case."""
    i_a, i_b = p['i_a'], p['i_b']
    gos_a = isoform_gos(i_a, go_labels)
    gos_b = isoform_gos(i_b, go_labels)
    gene_a = p.get('gene_a', p.get('gene', ''))
    gene_b = p.get('gene_b', p.get('gene', ''))

    conv_or_div_L = p.get('conv_layer', p.get('div_layer', -1))
    # Fisher peak of the shared GO (for conv) or first shared GO (for div)
    if kind == 'conv':
        shared_go = p['go']
    else:
        shared = p.get('shared_gos', [])
        shared_go = shared[0] if shared else (gos_a[0] if gos_a else None)
    fisher_peak_L = int(np.argmax(fisher_raw[shared_go])) + 1 if shared_go else -1

    # Case-level header info
    if kind == 'conv':
        header = (f"Case Conv#{case_idx}  —  {GO_18[p['go']]}  ({p['go']})\n"
                  f"Gene A: {gene_a} ({iso_te[i_a]})  vs  "
                  f"Gene B: {gene_b} ({iso_te[i_b]})\n"
                  f"d(L1)={p['d_L1']:.2f} → d(L30)={p['d_L30']:.2f}   "
                  f"conv_layer=L{p['conv_layer']}   "
                  f"Fisher_peak=L{fisher_peak_L}")
    else:
        n_shared = len(p.get('shared_gos', []))
        header = (f"Case Div#{case_idx}  —  Gene: {gene_a}  |  shared_GO={n_shared}\n"
                  f"Isoform A: {iso_te[i_a]}   vs   Isoform B: {iso_te[i_b]}\n"
                  f"d(L1)={p['d_L1']:.2f} → d(L30)={p['d_L30']:.2f}   "
                  f"div_layer=L{p['div_layer']}   "
                  f"Fisher_peak=L{fisher_peak_L}")

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(header, fontsize=12, fontweight='bold', y=0.995)

    # ── Top-Left: Non-normalized 3D trajectory ──────────────────────
    ax_nn_3d = fig.add_subplot(2, 2, 1, projection='3d')
    draw_traj_3d(ax_nn_3d,
                 Z[i_a, :, :3], Z[i_b, :, :3],
                 conv_or_div_L, fisher_peak_L,
                 gene_a, gene_b, kind,
                 '(A) Non-normalized 3D trajectory (raw Z)')

    # ── Top-Right: Non-normalized per-layer Fisher signal ───────────
    ax_nn_fs = fig.add_subplot(2, 2, 2)
    draw_fisher_panel(ax_nn_fs, gos_a, gos_b, fisher_raw,
                      conv_or_div_L, normalized=False,
                      title_prefix='(B) Non-normalized per-layer Fisher signal')

    # ── Bot-Left: Normalized 3D trajectory ──────────────────────────
    ax_n_3d = fig.add_subplot(2, 2, 3, projection='3d')
    draw_traj_3d(ax_n_3d,
                 Z_norm[i_a, :, :3], Z_norm[i_b, :, :3],
                 conv_or_div_L, fisher_peak_L,
                 gene_a, gene_b, kind,
                 '(C) Normalized 3D trajectory (per-layer z-score)')

    # ── Bot-Right: Normalized per-layer Fisher signal ───────────────
    ax_n_fs = fig.add_subplot(2, 2, 4)
    draw_fisher_panel(ax_n_fs, gos_a, gos_b, fisher_raw,
                      conv_or_div_L, normalized=True,
                      title_prefix='(D) Normalized per-layer Fisher signal (0-1 per GO)')

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    out_name = f"fig_case_{kind}_{case_idx:02d}.png"
    fig.savefig(OUT_DIR / out_name, dpi=140, bbox_inches='tight')
    fig.savefig((OUT_DIR / out_name).with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_name}")


def main():
    print("[load]")
    Z, sym_te, iso_te, go_labels, fisher_raw = load_all()
    print("[normalize Z per layer]")
    Z_norm = normalize_z(Z)

    print("[load pairs from convergence_divergence.json]")
    js = json.load(open(ROOT.parent / "reports" / "curve_sweep" /
                        "convergence_divergence.json"))
    conv_pairs = js["convergent_top"]
    div_pairs  = js["divergent_top"]

    print(f"\n=== Convergent cases ({len(conv_pairs)}) ===")
    for i, p in enumerate(conv_pairs, 1):
        make_case_figure(p, Z, Z_norm, sym_te, iso_te, go_labels, fisher_raw,
                         kind='conv', case_idx=i)

    print(f"\n=== Divergent cases ({len(div_pairs)}) ===")
    for i, p in enumerate(div_pairs, 1):
        make_case_figure(p, Z, Z_norm, sym_te, iso_te, go_labels, fisher_raw,
                         kind='div', case_idx=i)

    print(f"\n[done] {len(conv_pairs) + len(div_pairs)} figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
