"""
analyze_convergence_divergence.py
=================================
두 가지 궤적 현상을 데이터에서 자동 발굴 + 시각화:

[A] Convergent evolution
    - 다른 유전자 + 같은 GO
    - L1 distance 큼 (서로 다른 시퀀스)
    - L30 distance 작음 (기능적 수렴)
    - convergence_layer = distance가 L1의 50%로 처음 감소한 layer

[B] Divergent isoforms
    - 같은 유전자의 서로 다른 isoform
    - L1 distance 작음 (공통 서열 배경)
    - L30 distance 큼 (splicing으로 인한 발산)
    - divergence_layer = distance가 처음 유의미하게 상승한 layer

각 시나리오별 top-6 후보를 3D 궤적으로 시각화.
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa
from collections import defaultdict
from pathlib import Path
from itertools import combinations

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
    "GO:0007204": "Ca2+ signaling",       "GO:0045214": "Sarcomere org",
    "GO:0006941": "Muscle contract",      "GO:0006914": "Autophagy",
    "GO:0043161": "Proteasome-UPS",       "GO:0007519": "Skeletal musc dev",
    "GO:0042692": "Muscle cell diff",     "GO:0055074": "Ca2+ homeostasis",
    "GO:0007005": "Mitochondrion org",    "GO:0007517": "Muscle organ dev",
    "GO:0032006": "TOR signaling",        "GO:0030048": "Actin-based mov",
    "GO:0006096": "Glycolysis",           "GO:0007268": "Synaptic transm",
    "GO:0007018": "MT-based mov",         "GO:0031175": "Neuron proj dev",
    "GO:0030182": "Neuron diff",          "GO:0000226": "MT cytosk org",
}
MID_GOs = {"GO:0007204", "GO:0007018", "GO:0000226"}

SAMPLE_PER_GO   = 40    # convergent 탐색용 isoform 샘플 수
MAX_PAIRS_KEEP  = 6     # 각 시나리오에서 시각화할 top-N


# ── Load ───────────────────────────────────────────────────────────

def load_all():
    print("[load] IDs, Z, labels...")
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

    Z = np.load(CACHE_DIR / "Z_te.npy")   # (N, 30, 8)
    print(f"  Z: {Z.shape}, N_iso: {len(iso_te)}")

    # GO labels (gene-level)
    go_labels = {}
    for go in GO_18:
        pos = set()
        with open(ANNOT_DIR / "human_annotations_unified_bp.txt") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > 1 and go in parts[1:]:
                    pos.add(parts[0])
        go_labels[go] = np.array([s in pos for s in sym_te], dtype=bool)
    return Z, sym_te, iso_te, go_labels


# ── Distance helpers ───────────────────────────────────────────────

def pairwise_dist_at_layer(Z_subset, L):
    """Euclidean distance matrix at layer L (using all 8 PCs)."""
    X = Z_subset[:, L, :]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    return D


def convergence_layer(d_seq, threshold_ratio=0.5):
    """First layer where distance drops below threshold_ratio × L1 distance."""
    if d_seq[0] < 1e-6:
        return -1
    target = d_seq[0] * threshold_ratio
    for L in range(1, 30):
        if d_seq[L] < target:
            return L + 1   # 1-indexed
    return -1


def divergence_layer(d_seq, threshold_ratio=2.0):
    """First layer where distance rises above threshold_ratio × L1 distance."""
    if d_seq[0] < 1e-6:
        return -1
    target = d_seq[0] * threshold_ratio
    for L in range(1, 30):
        if d_seq[L] > target:
            return L + 1
    return -1


# ── Analysis A: Convergent evolution ───────────────────────────────

def find_convergent_pairs(Z, sym_te, iso_te, go_labels):
    """
    For each GO: sample K positive isoforms (from ≥2 different genes),
    compute pairwise distances at L1 and L30,
    rank pairs by (d_L1 - d_L30) subject to d_L1 > median and d_L30 < median.
    """
    print("\n[A] Searching convergent pairs...")
    candidates = []

    for go, name in GO_18.items():
        mask = go_labels[go]
        pos_idx = np.where(mask)[0]
        if len(pos_idx) < 10: continue

        # Restrict to isoforms from distinct genes
        gene_to_isos = defaultdict(list)
        for i in pos_idx:
            gene_to_isos[sym_te[i]].append(i)
        distinct_gene_isos = [ivs[0] for ivs in gene_to_isos.values()]  # 1 per gene
        if len(distinct_gene_isos) < 10: continue

        # Sample
        n_samp = min(SAMPLE_PER_GO, len(distinct_gene_isos))
        samp_idx = rng.choice(distinct_gene_isos, size=n_samp, replace=False)
        Z_sub = Z[samp_idx]

        D_L1  = pairwise_dist_at_layer(Z_sub, 0)
        D_L30 = pairwise_dist_at_layer(Z_sub, 29)

        for a, b in combinations(range(n_samp), 2):
            d1  = D_L1[a, b]
            d30 = D_L30[a, b]
            # Convergent filter (Z scale: L1 median~7.3, L30 median~12.2)
            # Require: L1 above population median AND L30 below population p25
            if d1 < 8.0 or d30 > 6.0: continue
            score = d1 - d30
            i_a, i_b = samp_idx[a], samp_idx[b]

            # per-layer distance sequence
            d_seq = np.sqrt(((Z[i_a] - Z[i_b]) ** 2).sum(-1))
            conv_L = convergence_layer(d_seq, 0.5)

            candidates.append({
                "go": go, "go_name": name,
                "is_mid": go in MID_GOs,
                "i_a": int(i_a), "i_b": int(i_b),
                "gene_a": sym_te[i_a], "gene_b": sym_te[i_b],
                "iso_a": iso_te[i_a], "iso_b": iso_te[i_b],
                "d_L1": float(d1), "d_L30": float(d30),
                "score": float(score),
                "conv_layer": int(conv_L),
                "d_seq": d_seq.tolist(),
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    # Deduplicate genes to get diverse examples
    seen = set()
    top = []
    for c in candidates:
        key = tuple(sorted([c["gene_a"], c["gene_b"]]))
        if key in seen: continue
        seen.add(key)
        top.append(c)
        if len(top) >= MAX_PAIRS_KEEP: break
    print(f"  Found {len(candidates)} convergent candidates; top {len(top)} kept")
    return top


# ── Analysis B: Divergent isoforms ─────────────────────────────────

def find_divergent_isoforms(Z, sym_te, iso_te, go_labels):
    """
    Find same-gene isoform pairs where L1 distance is small but L30 is large.
    Rank by (d_L30 - d_L1); require d_L1 < 1.5 and d_L30 > 3.0.
    """
    print("\n[B] Searching divergent isoform pairs...")
    gene_to_idx = defaultdict(list)
    for i, s in enumerate(sym_te):
        gene_to_idx[s].append(i)

    multi_iso_genes = {g: v for g, v in gene_to_idx.items() if len(v) >= 2}
    print(f"  Multi-isoform genes: {len(multi_iso_genes)}")

    candidates = []
    for gene, idxs in multi_iso_genes.items():
        if len(idxs) > 30:
            idxs = list(rng.choice(idxs, 30, replace=False))
        Z_sub = Z[idxs]
        D_L1  = pairwise_dist_at_layer(Z_sub, 0)
        D_L30 = pairwise_dist_at_layer(Z_sub, 29)

        for a, b in combinations(range(len(idxs)), 2):
            d1  = D_L1[a, b]
            d30 = D_L30[a, b]
            # Divergent filter (Z scale): same-gene must start close, end far
            # Population medians: L1~7.3, L30~12.2. Same-gene expected < median at L1.
            if d1 > 3.0 or d30 < 10.0: continue
            score = d30 - d1
            i_a, i_b = idxs[a], idxs[b]
            d_seq = np.sqrt(((Z[i_a] - Z[i_b]) ** 2).sum(-1))
            div_L = divergence_layer(d_seq, 2.0)

            # Assigned GOs
            shared_gos = [go for go in GO_18
                          if go_labels[go][i_a] and go_labels[go][i_b]]

            candidates.append({
                "gene": gene,
                "i_a": int(i_a), "i_b": int(i_b),
                "iso_a": iso_te[i_a], "iso_b": iso_te[i_b],
                "d_L1": float(d1), "d_L30": float(d30),
                "score": float(score),
                "div_layer": int(div_L),
                "shared_gos": shared_gos,
                "d_seq": d_seq.tolist(),
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    top = []
    for c in candidates:
        if c["gene"] in seen: continue
        seen.add(c["gene"])
        top.append(c)
        if len(top) >= MAX_PAIRS_KEEP: break
    print(f"  Found {len(candidates)} divergent candidates; top {len(top)} kept")
    return top


# ── Visualization ──────────────────────────────────────────────────

def viz_pairs(Z, pairs, kind: str, fname: str):
    """kind='conv' or 'div'."""
    fig = plt.figure(figsize=(24, 14))
    if kind == "conv":
        fig.suptitle(
            "Convergent Evolution — Different genes, Same GO\n"
            "Blue vs Green: two isoforms from different genes converging at later layers  |  "
            "Bottom-right inset: per-layer pairwise distance (red dot = convergence layer)",
            fontsize=13, fontweight='bold', y=0.995)
    else:
        fig.suptitle(
            "Isoform Divergence — Same gene, different splicing branches\n"
            "Blue vs Green: two isoforms of the same gene diverging at later layers  |  "
            "Bottom-right inset: per-layer pairwise distance (red dot = divergence layer)",
            fontsize=13, fontweight='bold', y=0.995)

    for i, p in enumerate(pairs):
        # 3D panel
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        pts_a = Z[p["i_a"], :, :3]
        pts_b = Z[p["i_b"], :, :3]

        # Trajectory a (blue)
        ax.plot(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2],
                color='tab:blue', lw=2.2, alpha=0.9,
                label=f"{p.get('gene_a', p.get('gene', ''))} / {p['iso_a']}")
        ax.scatter(pts_a[0, 0], pts_a[0, 1], pts_a[0, 2],
                   color='tab:blue', marker='o', s=80, edgecolors='white', lw=0.8,
                   zorder=5)
        ax.scatter(pts_a[-1, 0], pts_a[-1, 1], pts_a[-1, 2],
                   color='tab:blue', marker='X', s=110, edgecolors='white', lw=0.8,
                   zorder=5)

        # Trajectory b (green)
        ax.plot(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2],
                color='seagreen', lw=2.2, alpha=0.9,
                label=f"{p.get('gene_b', p.get('gene', ''))} / {p['iso_b']}")
        ax.scatter(pts_b[0, 0], pts_b[0, 1], pts_b[0, 2],
                   color='seagreen', marker='o', s=80, edgecolors='white', lw=0.8,
                   zorder=5)
        ax.scatter(pts_b[-1, 0], pts_b[-1, 1], pts_b[-1, 2],
                   color='seagreen', marker='X', s=110, edgecolors='white', lw=0.8,
                   zorder=5)

        # Mark convergence/divergence layer
        conv_or_div_L = p.get("conv_layer", p.get("div_layer", -1))
        if conv_or_div_L > 0:
            L_idx = conv_or_div_L - 1
            ax.scatter(pts_a[L_idx, 0], pts_a[L_idx, 1], pts_a[L_idx, 2],
                       color='red', marker='*', s=180,
                       edgecolors='darkred', lw=1.0, zorder=8,
                       label=f'{"conv" if kind=="conv" else "div"} @ L{conv_or_div_L}')
            ax.scatter(pts_b[L_idx, 0], pts_b[L_idx, 1], pts_b[L_idx, 2],
                       color='red', marker='*', s=180,
                       edgecolors='darkred', lw=1.0, zorder=8)

        # Title
        if kind == "conv":
            title = (f"{p['go_name']} {'[MID]' if p['is_mid'] else ''}\n"
                     f"{p['gene_a']} vs {p['gene_b']}\n"
                     f"d(L1)={p['d_L1']:.2f} → d(L30)={p['d_L30']:.2f}"
                     f"  conv@L{p['conv_layer']}")
            title_color = '#B71C1C' if p['is_mid'] else '#1A237E'
        else:
            n_go = len(p.get('shared_gos', []))
            title = (f"{p['gene']}\n"
                     f"{p['iso_a']} vs {p['iso_b']}\n"
                     f"d(L1)={p['d_L1']:.2f} → d(L30)={p['d_L30']:.2f}"
                     f"  div@L{p['div_layer']}  |  GO∩={n_go}")
            title_color = '#1A237E'
        ax.set_title(title, fontsize=8, color=title_color, pad=4)
        ax.set_xlabel('PC1', fontsize=7)
        ax.set_ylabel('PC2', fontsize=7)
        ax.set_zlabel('PC3', fontsize=7)
        ax.tick_params(labelsize=5)
        ax.view_init(elev=20, azim=-60)
        ax.legend(fontsize=6, loc='upper left')

        # Inset: pairwise distance sequence
        ax_ins = fig.add_axes([
            ax.get_position().x1 - 0.055,
            ax.get_position().y0 + 0.015,
            0.06, 0.055
        ])
        d_seq = np.array(p['d_seq'])
        ax_ins.plot(range(1, 31), d_seq, color='#455A64', lw=1.2)
        if conv_or_div_L > 0:
            ax_ins.axvline(conv_or_div_L, color='red', lw=1.2, alpha=0.7)
            ax_ins.scatter([conv_or_div_L], [d_seq[conv_or_div_L-1]],
                           color='red', s=25, zorder=5)
        ax_ins.set_xlim(1, 30)
        ax_ins.tick_params(labelsize=5)
        ax_ins.set_title('d(a,b) vs L', fontsize=6, pad=1)
        ax_ins.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / fname, dpi=140, bbox_inches='tight')
    fig.savefig((OUT_DIR / fname).with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {OUT_DIR}/{fname}")


def main():
    Z, sym_te, iso_te, go_labels = load_all()

    conv_pairs = find_convergent_pairs(Z, sym_te, iso_te, go_labels)
    div_pairs  = find_divergent_isoforms(Z, sym_te, iso_te, go_labels)

    if conv_pairs:
        viz_pairs(Z, conv_pairs, "conv", "fig_convergent_bundles.png")
    else:
        print("[warn] No convergent pairs met threshold")

    if div_pairs:
        viz_pairs(Z, div_pairs, "div",  "fig_divergent_isoforms.png")
    else:
        print("[warn] No divergent pairs met threshold")

    # Save summary JSON
    summary = {
        "convergent_top": [{k: v for k, v in c.items() if k != "d_seq"}
                           for c in conv_pairs],
        "divergent_top":  [{k: v for k, v in c.items() if k != "d_seq"}
                           for c in div_pairs],
    }
    with open(OUT_DIR / "convergence_divergence.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {OUT_DIR}/convergence_divergence.json")


if __name__ == "__main__":
    main()
