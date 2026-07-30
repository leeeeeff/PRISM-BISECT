"""
plot_convdiv_fluid3.py
======================
Convergent / Divergent trajectory 결과를 fluid_stage3 스타일로 재시각화.

각 시나리오별 2 figure (non-norm / norm), 각 figure = 1×2 subplot:
  (L1) Bundle-coloured: 각 pair 고유 색상, background=gray context
  (L2) Layer-signal cividis: pair가 공유한 GO의 Fisher signal로 세그먼트 색칠

각 isoform의 최종 assigned GO를 궤적 끝점에 textbox로 표기.
"""
from __future__ import annotations

import json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path

warnings.filterwarnings('ignore')

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

N_BG        = 250     # background context trajectories
BUNDLE_PALETTE = ['#E53935', '#1E88E5', '#43A047', '#FB8C00',
                  '#8E24AA', '#00ACC1']
LAYER_LABELS   = [1, 10, 20, 30]


# ── Load ───────────────────────────────────────────────────────────

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

    sym_te = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
              for g in te_gene]
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

    fisher = {}
    for fname in ["layer_probe_v15d_terms_results.json",
                  "layer_probe_expanded_results.json",
                  "layer_probe_results.json"]:
        p = PROBE_DIR / fname
        if p.exists():
            d = json.load(open(p))
            for k, v in d["lr_auprc"].items():
                if k in GO_18:
                    arr = np.array(v, dtype=np.float32)
                    fisher[k] = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    return Z, sym_te, iso_te, go_labels, fisher


# ── Helpers ────────────────────────────────────────────────────────

def isoform_gos(i, go_labels):
    """Return list of GO IDs that this isoform is positively annotated for."""
    return [go for go, labels in go_labels.items() if labels[i]]


def format_go_short(go_list, max_n=2):
    """Short label of GO assignments."""
    if not go_list: return "no-GO"
    names = [GO_18[g] for g in go_list[:max_n]]
    extra = f" +{len(go_list)-max_n}" if len(go_list) > max_n else ""
    return "; ".join(names) + extra


def colored_line3d(pts, values, cmap, lw=1.4, alpha=0.85):
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_vals = (values[:-1] + values[1:]) / 2.0
    lc = Line3DCollection(segs, cmap=cmap, linewidths=lw, alpha=alpha,
                          norm=plt.Normalize(vmin=0.0, vmax=1.0))
    lc.set_array(seg_vals)
    return lc


# ── Main draw ──────────────────────────────────────────────────────

def draw_scenario(Z, sym_te, iso_te, go_labels, fisher,
                  pairs, kind, normalize_layers, out_fname):
    """
    kind: 'conv' or 'div'
    normalize_layers: True → per-layer z-score before plotting
    """
    Z_use = Z.copy()
    if normalize_layers:
        for L in range(30):
            mu = Z_use[:, L, :].mean(0)
            sd = Z_use[:, L, :].std(0) + 1e-8
            Z_use[:, L, :] = (Z_use[:, L, :] - mu) / sd

    fig = plt.figure(figsize=(24, 12))
    ax_bundle = fig.add_subplot(1, 2, 1, projection='3d')
    ax_fisher = fig.add_subplot(1, 2, 2, projection='3d')

    for ax in (ax_bundle, ax_fisher):
        ax.set_facecolor('white')

    # Background context (gray)
    bg_idx = rng.choice(len(Z_use), size=min(N_BG, len(Z_use)), replace=False)
    for i in bg_idx:
        pts = Z_use[i, :, :3]
        ax_bundle.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                       color='lightgray', lw=0.35, alpha=0.18)
        ax_fisher.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                       color='lightgray', lw=0.35, alpha=0.18)

    # Each pair
    for k, p in enumerate(pairs):
        col = BUNDLE_PALETTE[k % len(BUNDLE_PALETTE)]
        i_a, i_b = p['i_a'], p['i_b']
        pts_a = Z_use[i_a, :, :3]
        pts_b = Z_use[i_b, :, :3]

        # ── L1: Bundle-coloured (each pair a distinct color) ─────────
        ax_bundle.plot(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2],
                       color=col, lw=2.4, alpha=0.9)
        ax_bundle.plot(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2],
                       color=col, lw=2.4, alpha=0.9, ls='--')

        # Start/end markers
        for pts in (pts_a, pts_b):
            ax_bundle.scatter(pts[0, 0], pts[0, 1], pts[0, 2],
                              color=col, marker='o', s=95,
                              edgecolors='black', linewidths=1.2, zorder=6)
            ax_bundle.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                              color=col, marker='X', s=140,
                              edgecolors='black', linewidths=1.2, zorder=6)

        # Layer text labels on trajectory a (only some layers)
        for L in LAYER_LABELS:
            j = L - 1
            ax_bundle.text(pts_a[j, 0], pts_a[j, 1], pts_a[j, 2],
                           f'  L{L}', fontsize=6, color=col, alpha=0.85,
                           fontweight='bold')

        # Convergence/divergence marker
        conv_L = p.get('conv_layer', p.get('div_layer', -1))
        if conv_L > 0:
            L_idx = conv_L - 1
            mid_pt = (pts_a[L_idx] + pts_b[L_idx]) / 2
            ax_bundle.scatter(mid_pt[0], mid_pt[1], mid_pt[2],
                              color='yellow', marker='*', s=250,
                              edgecolors='black', linewidths=1.5, zorder=9)

        # GO labels at end of each trajectory
        gos_a = isoform_gos(i_a, go_labels)
        gos_b = isoform_gos(i_b, go_labels)
        gene_a = p.get('gene_a', p.get('gene', ''))
        gene_b = p.get('gene_b', p.get('gene', ''))

        label_a = f"{gene_a}\n[{format_go_short(gos_a)}]"
        label_b = f"{gene_b}\n[{format_go_short(gos_b)}]"
        # only bundle panel
        ax_bundle.text(pts_a[-1, 0], pts_a[-1, 1], pts_a[-1, 2] + 0.5,
                       label_a, fontsize=6, color='black',
                       bbox=dict(facecolor='white', edgecolor=col,
                                 boxstyle='round,pad=0.15', alpha=0.85),
                       zorder=10)
        ax_bundle.text(pts_b[-1, 0], pts_b[-1, 1], pts_b[-1, 2] - 0.5,
                       label_b, fontsize=6, color='black',
                       bbox=dict(facecolor='white', edgecolor=col,
                                 boxstyle='round,pad=0.15', alpha=0.85),
                       zorder=10)

        # ── R2: Layer-signal cividis ─────────────────────────────────
        if kind == 'conv':
            fisher_go = p['go']
        else:
            shared = p.get('shared_gos', [])
            fisher_go = shared[0] if shared else None
        if fisher_go and fisher_go in fisher:
            fs = fisher[fisher_go]
            lc_a = colored_line3d(pts_a, fs, plt.cm.cividis, lw=2.2, alpha=0.9)
            lc_b = colored_line3d(pts_b, fs, plt.cm.cividis, lw=2.2, alpha=0.9)
            ax_fisher.add_collection3d(lc_a)
            ax_fisher.add_collection3d(lc_b)
        else:
            # fallback: use gray
            ax_fisher.plot(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2],
                           color='#555555', lw=2.2, alpha=0.85)
            ax_fisher.plot(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2],
                           color='#555555', lw=2.2, alpha=0.85, ls='--')

        # Markers on right panel too (using black)
        for pts in (pts_a, pts_b):
            ax_fisher.scatter(pts[0, 0], pts[0, 1], pts[0, 2],
                              color='black', marker='o', s=60,
                              edgecolors='white', linewidths=0.8, zorder=6)
            ax_fisher.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                              color='black', marker='X', s=90,
                              edgecolors='white', linewidths=0.8, zorder=6)
        if conv_L > 0:
            L_idx = conv_L - 1
            mid_pt = (pts_a[L_idx] + pts_b[L_idx]) / 2
            ax_fisher.scatter(mid_pt[0], mid_pt[1], mid_pt[2],
                              color='red', marker='*', s=180,
                              edgecolors='black', linewidths=1.0, zorder=9)

        # Panel-label per pair (right side legend-ish, only in bundle panel)
        pair_label = (f"{gene_a} × {gene_b}"
                      if kind == 'conv' else
                      f"{p.get('gene', '')} : {p['iso_a']} × {p['iso_b']}")
        if kind == 'conv':
            extra = f" — {GO_18[p['go']]}  conv@L{p['conv_layer']}"
        else:
            extra = f" — div@L{p['div_layer']}  GO∩={len(p.get('shared_gos', []))}"
        ax_bundle.plot([], [], color=col, lw=2.4, label=pair_label + extra)

    # ── Cosmetics ────────────────────────────────────────────────────
    ax_bundle.set_title(
        f"({'C' if not normalize_layers else 'D'}1) "
        f"{'Non-normalized' if not normalize_layers else 'Normalized'} 3D PCA  "
        f"•  bundle-coloured  •  solid=isoform A, dashed=isoform B  "
        f"•  ★=yellow ({'conv' if kind=='conv' else 'div'} layer)",
        fontsize=10, pad=10
    )
    ax_bundle.set_xlabel('PC1'); ax_bundle.set_ylabel('PC2'); ax_bundle.set_zlabel('PC3')
    ax_bundle.legend(fontsize=6, loc='upper left', bbox_to_anchor=(-0.02, 1.02),
                     framealpha=0.85, ncol=1)

    ax_fisher.set_title(
        f"({'C' if not normalize_layers else 'D'}2) "
        f"{'Non-normalized' if not normalize_layers else 'Normalized'} 3D PCA  "
        f"•  layer-signal cividis (per-GO Fisher AUPRC 0-1)  "
        f"•  ★=red ({'conv' if kind=='conv' else 'div'} layer)",
        fontsize=10, pad=10
    )
    ax_fisher.set_xlabel('PC1'); ax_fisher.set_ylabel('PC2'); ax_fisher.set_zlabel('PC3')

    # Colorbar for right panel
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis,
                                norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_fisher, shrink=0.55, pad=0.05)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0 (Fisher min)", "0.25", "0.5", "0.75", "1 (peak)"])
    cb.set_label("per-layer Fisher signal (normalized 0-1 per GO)")

    kind_full = 'Convergent Evolution (different genes, same GO)' if kind == 'conv' \
                else 'Divergent Isoforms (same gene, different splicing branches)'
    fig.suptitle(
        f"{kind_full} — fluid_stage3 style  |  "
        f"{'Non-normalized' if not normalize_layers else 'Normalized (per-layer z-score)'} PCA",
        fontsize=13, fontweight='bold', y=0.99
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT_DIR / out_fname, dpi=150, bbox_inches='tight')
    fig.savefig((OUT_DIR / out_fname).with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {OUT_DIR}/{out_fname}")


def main():
    print("[load]")
    Z, sym_te, iso_te, go_labels, fisher = load_all()

    print("[load pairs from convergence_divergence.json]")
    js = json.load(open(OUT_DIR / "convergence_divergence.json"))
    conv_pairs = js["convergent_top"]
    div_pairs  = js["divergent_top"]

    print(f"  conv: {len(conv_pairs)}  div: {len(div_pairs)}")

    # We need d_seq from computation. Add on-the-fly
    for p in conv_pairs + div_pairs:
        if 'd_seq' not in p:
            d_seq = np.sqrt(((Z[p['i_a']] - Z[p['i_b']]) ** 2).sum(-1))
            p['d_seq'] = d_seq.tolist()

    # ── Convergent: non-norm + norm ─────────────────────────────────
    print("\n[1] Convergent — Non-normalized (fluid3 style)")
    draw_scenario(Z, sym_te, iso_te, go_labels, fisher, conv_pairs,
                  kind='conv', normalize_layers=False,
                  out_fname="fig_conv_fluid3_nonnorm.png")

    print("[2] Convergent — Normalized (fluid3 style)")
    draw_scenario(Z, sym_te, iso_te, go_labels, fisher, conv_pairs,
                  kind='conv', normalize_layers=True,
                  out_fname="fig_conv_fluid3_norm.png")

    # ── Divergent: non-norm + norm ──────────────────────────────────
    print("\n[3] Divergent — Non-normalized (fluid3 style)")
    draw_scenario(Z, sym_te, iso_te, go_labels, fisher, div_pairs,
                  kind='div', normalize_layers=False,
                  out_fname="fig_div_fluid3_nonnorm.png")

    print("[4] Divergent — Normalized (fluid3 style)")
    draw_scenario(Z, sym_te, iso_te, go_labels, fisher, div_pairs,
                  kind='div', normalize_layers=True,
                  out_fname="fig_div_fluid3_norm.png")

    print("\n[done] 4 fluid_stage3 style figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
