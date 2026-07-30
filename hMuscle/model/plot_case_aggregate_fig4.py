"""
plot_case_aggregate_fig4.py
============================
Manuscript Fig 4 — aggregate 12-case summary panel (2×6):
Top row  : 6 CONV cases (different genes, same GO)
Bottom   : 6 DIV cases  (different genes, different GO)

Each subpanel: 3D joint-PCA trajectory with case A & B + target GO bundle mean
Compact size for main-text figure.
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa
from sklearn.decomposition import PCA
from pathlib import Path

plt.rcParams.update({
    'font.family'    : ['DejaVu Sans'],
    'font.size'      : 10,
    'axes.titlesize' : 11,
    'axes.linewidth' : 1.0,
    'axes.unicode_minus': False,
})

SEED = 42
rng = np.random.default_rng(SEED)

ROOT      = Path("/home/welcome1/sw1686/DIFFUSE")
MODEL_DIR = ROOT / "hMuscle/model"
DATA      = ROOT / "hMuscle/data"
ID_DIR    = DATA / "raw_data/data/id_lists"
ANNOT_DIR = DATA / "raw_data/data/annotations"
CACHE     = ROOT / "reports/v20_cache"
OUT       = ROOT / "reports/curve_sweep"

GO_18 = {
    "GO:0007204": "Ca2+ signaling",        "GO:0045214": "Sarcomere org.",
    "GO:0006941": "Muscle contract.",      "GO:0006914": "Autophagy",
    "GO:0043161": "Proteasome/UPS",        "GO:0007519": "Skeletal m. dev.",
    "GO:0042692": "Muscle cell diff.",     "GO:0055074": "Ca2+ homeost.",
    "GO:0007005": "Mitochondrion org.",    "GO:0007517": "Muscle organ dev.",
    "GO:0032006": "TOR signaling",         "GO:0030048": "Actin movement",
    "GO:0006096": "Glycolysis",            "GO:0007268": "Synaptic transm.",
    "GO:0007018": "MT movement",           "GO:0031175": "Neuron proj. dev.",
    "GO:0030182": "Neuron diff.",          "GO:0000226": "MT cytosk. org.",
}

N_LAYERS  = 30
EMB_DIM   = 640
N_BUNDLE  = 30

COLOR_A       = '#1E88E5'
COLOR_B       = '#43A047'
COLOR_BUNDLE  = '#3A3A3A'
COLOR_PEAK    = '#FFC107'
COLOR_KEY     = '#D81B60'


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


def load_ids():
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
    te_gene = np.load(MODEL_DIR / "my_gene_list_fixed.npy", allow_pickle=True)
    te_iso  = np.load(MODEL_DIR / "my_isoform_list_fixed.npy", allow_pickle=True)
    sym_te = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
              for g in te_gene]
    return sym_te, [clean(i) for i in te_iso]


def load_go_labels(sym_te):
    go_labels = {}
    for go in GO_18:
        pos = set()
        with open(ANNOT_DIR / "human_annotations_unified_bp.txt") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > 1 and go in parts[1:]:
                    pos.add(parts[0])
        go_labels[go] = np.array([s in pos for s in sym_te], dtype=bool)
    return go_labels


def collect_subset(cases, go_labels):
    idx_set = set()
    for c in cases:
        idx_set.add(c['i_a']); idx_set.add(c['i_b'])
    go_bundle_idx = {}
    for go in GO_18:
        pos_idx = np.where(go_labels[go])[0]
        n = min(N_BUNDLE, len(pos_idx))
        if n > 0:
            chosen = rng.choice(pos_idx, size=n, replace=False)
            go_bundle_idx[go] = chosen.tolist()
            idx_set.update(chosen.tolist())
        else:
            go_bundle_idx[go] = []
    subset_idx = np.array(sorted(idx_set), dtype=int)
    orig2local = {oi: li for li, oi in enumerate(subset_idx)}
    return subset_idx, orig2local, go_bundle_idx


def build_trajectory(subset_idx):
    N = len(subset_idx)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        path = DATA / f"esm2_layer_{L:02d}_t30_150M.npy"
        arr = np.load(path, mmap_mode="r")
        traj[:, L-1, :] = arr[subset_idx].astype(np.float32)
        del arr
    return traj


def pca_coords(traj):
    N = traj.shape[0]
    flat = traj.reshape(N * N_LAYERS, EMB_DIM)
    pca = PCA(n_components=3, random_state=SEED)
    reduced = pca.fit_transform(flat).reshape(N, N_LAYERS, 3)
    return reduced.astype(np.float32)


def draw_case(ax, coords, orig2local, go_bundle_idx,
              i_a, i_b, primary_A, primary_B,
              key_L, is_conv, title):
    la = orig2local[i_a]; lb = orig2local[i_b]
    A = coords[la]; B = coords[lb]

    # Bundle means for target GOs
    for go in {primary_A, primary_B}:
        if go is None or go not in go_bundle_idx: continue
        idxs = [orig2local[oi] for oi in go_bundle_idx[go] if oi in orig2local]
        if len(idxs) < 5: continue
        bm = coords[idxs].mean(axis=0)
        ax.plot(bm[:, 0], bm[:, 1], bm[:, 2],
                color=COLOR_BUNDLE, lw=1.3, alpha=0.55, ls=':', zorder=3)

    # Case A (blue)
    ax.plot(A[:, 0], A[:, 1], A[:, 2],
            color=COLOR_A, lw=2.0, alpha=0.95, zorder=6)
    ax.scatter(A[0, 0], A[0, 1], A[0, 2], color=COLOR_A, marker='o', s=50,
               edgecolors='white', linewidths=1.0, zorder=8)
    ax.scatter(A[-1, 0], A[-1, 1], A[-1, 2], color=COLOR_A, marker='X', s=80,
               edgecolors='white', linewidths=1.0, zorder=8)

    # Case B (green)
    ax.plot(B[:, 0], B[:, 1], B[:, 2],
            color=COLOR_B, lw=2.0, alpha=0.95, zorder=6)
    ax.scatter(B[0, 0], B[0, 1], B[0, 2], color=COLOR_B, marker='o', s=50,
               edgecolors='white', linewidths=1.0, zorder=8)
    ax.scatter(B[-1, 0], B[-1, 1], B[-1, 2], color=COLOR_B, marker='X', s=80,
               edgecolors='white', linewidths=1.0, zorder=8)

    # Key layer marker
    if key_L > 0:
        Lc = key_L - 1
        ax.scatter(A[Lc, 0], A[Lc, 1], A[Lc, 2],
                   color=COLOR_KEY, marker='*', s=150,
                   edgecolors='black', lw=0.8, zorder=10)
        ax.scatter(B[Lc, 0], B[Lc, 1], B[Lc, 2],
                   color=COLOR_KEY, marker='*', s=150,
                   edgecolors='black', lw=0.8, zorder=10)

    ax.set_title(title, fontsize=9.5, pad=2,
                 color='#1A237E' if is_conv else '#B71C1C',
                 fontweight='bold')
    ax.set_xlabel('PC1', fontsize=7, labelpad=-6)
    ax.set_ylabel('PC2', fontsize=7, labelpad=-6)
    ax.set_zlabel('PC3', fontsize=7, labelpad=-6)
    ax.tick_params(labelsize=6, pad=0)
    ax.view_init(elev=20, azim=-60)


def main():
    print("[1] Loading IDs & labels …")
    sym_te, iso_te = load_ids()
    go_labels = load_go_labels(sym_te)

    print("[2] Loading case JSON …")
    d = json.load(open(ROOT / "reports/curve_sweep/convergence_divergence.json"))
    conv = d["convergent_top"]
    div = d["divergent_top"]
    all_cases = [('conv', p) for p in conv] + [('div', p) for p in div]

    print("[3] Building subset & raw ESM-2 trajectories …")
    subset_idx, orig2local, go_bundle_idx = collect_subset(
        [p for _, p in all_cases], go_labels)
    print(f"    subset size: {len(subset_idx)}")
    traj = build_trajectory(subset_idx)
    print(f"    traj shape: {traj.shape}")

    print("[4] Joint PCA …")
    coords = pca_coords(traj)
    del traj

    # ── Figure ─────────────────────────────────────────────────────
    print("[5] Rendering aggregate figure …")
    fig = plt.figure(figsize=(24, 9))
    fig.suptitle(
        "Trajectory case studies — twelve pairs demonstrating "
        "isoform-resolution dynamics inaccessible to mean-pooling",
        fontsize=15, fontweight='bold', y=0.995)

    # CONV row (top)
    for i, p in enumerate(conv):
        ax = fig.add_subplot(2, 6, i + 1, projection='3d')
        primary = p['go']
        title = (f"CONV #{i+1}  |  {sym_te[p['i_a']]} vs {sym_te[p['i_b']]}\n"
                 f"→ {GO_18[primary]}\n"
                 f"d(L1)={p['d_L1']:.1f} → d(L30)={p['d_L30']:.1f}"
                 f"   conv@L{p['conv_layer']}")
        draw_case(ax, coords, orig2local, go_bundle_idx,
                  p['i_a'], p['i_b'], primary, primary,
                  p['conv_layer'], is_conv=True, title=title)

    # DIV row (bottom)
    for i, p in enumerate(div):
        ax = fig.add_subplot(2, 6, i + 7, projection='3d')
        primary_A = p['gos_a'][0] if p.get('gos_a') else None
        primary_B = p['gos_b'][0] if p.get('gos_b') else None
        title = (f"DIV #{i+1}  |  {sym_te[p['i_a']]} vs {sym_te[p['i_b']]}\n"
                 f"→ {GO_18[primary_A] if primary_A else '?'} vs "
                 f"{GO_18[primary_B] if primary_B else '?'}\n"
                 f"d(L1)={p['d_L1']:.1f} → d(L30)={p['d_L30']:.1f}"
                 f"   div@L{p['div_layer']}")
        draw_case(ax, coords, orig2local, go_bundle_idx,
                  p['i_a'], p['i_b'], primary_A, primary_B,
                  p['div_layer'], is_conv=False, title=title)

    # Legend at bottom
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COLOR_A, lw=2.4, label='Case A trajectory'),
        Line2D([0], [0], color=COLOR_B, lw=2.4, label='Case B trajectory'),
        Line2D([0], [0], color=COLOR_BUNDLE, lw=1.3, ls=':',
               label='Target GO bundle mean (n≥5 isoforms)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='white', markersize=8, label='L1 (start)', lw=0),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='gray',
               markeredgecolor='white', markersize=9, label='L30 (end)', lw=0),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLOR_KEY,
               markeredgecolor='black', markersize=13,
               label='Convergence/Divergence layer', lw=0),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, 0.00), fontsize=10, framealpha=0.9)

    # Section separators
    fig.text(0.005, 0.735,
             'CONVERGENT\nEVOLUTION\n(diff gene\nsame GO)',
             fontsize=11, fontweight='bold', color='#1A237E',
             ha='left', va='center', linespacing=1.3)
    fig.text(0.005, 0.335,
             'FUNCTIONAL\nDIVERGENCE\n(diff gene\ndiff GO)',
             fontsize=11, fontweight='bold', color='#B71C1C',
             ha='left', va='center', linespacing=1.3)

    fig.tight_layout(rect=[0.02, 0.045, 1, 0.965])
    fig.savefig(OUT / "fig_case_aggregate_fig4.png", dpi=140,
                bbox_inches='tight')
    fig.savefig(OUT / "fig_case_aggregate_fig4.pdf",
                bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {OUT}/fig_case_aggregate_fig4.png")


if __name__ == "__main__":
    main()
