"""
exp_brain_trajectory_flow.py
============================
뇌 데이터 trajectory analysis (요구: 브레인 L01-L30 layer 임베딩 필요).

1. Convergent (diff-gene, same-GO) & Divergent (diff-gene, diff-GO) case 발굴
2. Early/Mid/Late layer 신호 특화 대표 GO term 선정 (Fisher peak 위치 기반)
3. 3D trajectory flow — bundle 표현

Output: reports/brain_flow/
  fig_brain_conv_top.png / fig_brain_div_top.png
  fig_brain_bundle_early_go.png / _mid_ / _late_
  brain_case_summary.json
"""
from __future__ import annotations

import json, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.decomposition import PCA
from collections import defaultdict
from itertools import combinations
from pathlib import Path
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family'    : ['DejaVu Sans'],
    'font.size'      : 11,
    'axes.titlesize' : 13,
    'axes.labelsize' : 11,
    'axes.unicode_minus': False,
})

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
BRAIN_DIR = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
BRAIN_LABEL_DIR = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
OUT_DIR = ROOT / "reports" / "brain_flow"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS = 30
K_PCA_PER_LAYER = 8
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


def check_prereq():
    missing = []
    for L in range(1, N_LAYERS + 1):
        p = BRAIN_DIR / f"brain_full_esm2_layer{L:02d}_t30_150M.npy"
        if not p.exists():
            missing.append(L)
    if missing:
        print(f"[warn] Missing brain layers: {missing}")
        return False
    print(f"[ok] All 30 brain layers present in {BRAIN_DIR}")
    return True


def load_brain_layers_and_pca(coding_mask):
    """Load all 30 brain layers → joint per-layer PCA → Z (N, 30, 8)."""
    N_all = len(coding_mask)
    coding_idx = np.where(coding_mask == 1)[0]
    print(f"  N_all={N_all}, coding={len(coding_idx)}")

    Z = np.zeros((N_all, N_LAYERS, K_PCA_PER_LAYER), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        arr = np.load(BRAIN_DIR / f"brain_full_esm2_layer{L:02d}_t30_150M.npy",
                       mmap_mode='r')
        X = arr[coding_idx].astype(np.float32)
        pca = PCA(n_components=K_PCA_PER_LAYER, random_state=SEED)
        Xr = pca.fit_transform(X)
        Z[coding_idx, L - 1, :] = Xr
        del arr, X
        if L % 5 == 0:
            print(f"    L{L:02d} PCA done  var_top8={pca.explained_variance_ratio_.sum():.3f}")
    return Z


def pairwise_dist_at_layer(Z_subset, L):
    X = Z_subset[:, L, :]
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))


def load_labels():
    """Load brain 18 BP GO labels."""
    lab = np.load(BRAIN_LABEL_DIR / "brain_full_labels.npy")
    ids = np.load(BRAIN_LABEL_DIR / "brain_full_ids.npy", allow_pickle=True)
    genes = np.load(BRAIN_LABEL_DIR / "brain_full_gene_names.npy",
                    allow_pickle=True)
    return lab, ids, genes


def find_convergent(Z, coding_idx, labels, gene_names):
    """diff gene, same GO — L1 far, L30 close."""
    candidates = []
    for gi, (go, name) in enumerate(GO_18.items()):
        pos_idx = np.where(labels[:, gi] == 1)[0]
        pos_idx = np.intersect1d(pos_idx, coding_idx)
        if len(pos_idx) < 15: continue
        gene_to_iso = defaultdict(list)
        for i in pos_idx:
            gene_to_iso[str(gene_names[i])].append(int(i))
        rep = [g[0] for g in gene_to_iso.values()]
        if len(rep) < 10: continue
        n_samp = min(40, len(rep))
        samp = rng.choice(rep, size=n_samp, replace=False)
        Zs = Z[samp]
        D1  = pairwise_dist_at_layer(Zs, 0)
        D30 = pairwise_dist_at_layer(Zs, 29)
        for a, b in combinations(range(n_samp), 2):
            d1, d30 = D1[a, b], D30[a, b]
            if d1 < 8.0 or d30 > 6.0: continue
            i_a, i_b = int(samp[a]), int(samp[b])
            candidates.append({
                "go": go, "name": name,
                "i_a": i_a, "i_b": i_b,
                "gene_a": str(gene_names[i_a]),
                "gene_b": str(gene_names[i_b]),
                "d_L1": float(d1), "d_L30": float(d30),
                "score": float(d1 - d30),
            })
    candidates.sort(key=lambda x: x["score"], reverse=True)
    seen = set(); top = []
    for c in candidates:
        key = tuple(sorted([c["gene_a"], c["gene_b"]]))
        if key in seen: continue
        seen.add(key); top.append(c)
        if len(top) >= 6: break
    return top


def find_divergent(Z, coding_idx, labels, gene_names):
    """diff gene, diff GO — L1 close, L30 far. Brain-adaptive thresholds."""
    N_go = len(GO_18)
    with_any = np.where(labels.sum(axis=1) > 0)[0]
    with_any = np.intersect1d(with_any, coding_idx)
    print(f"  brain isoforms with any 18-BP GO: {len(with_any)}")
    # one representative per gene
    seen_gene = set(); rep_idxs = []
    for i in with_any:
        g = str(gene_names[i])
        if g in seen_gene: continue
        seen_gene.add(g); rep_idxs.append(int(i))
    print(f"  distinct-gene reps: {len(rep_idxs)}")
    N_SUB = min(2500, len(rep_idxs))
    if len(rep_idxs) > N_SUB:
        rep_idxs = list(rng.choice(rep_idxs, size=N_SUB, replace=False))

    Zs = Z[rep_idxs]
    D1  = pairwise_dist_at_layer(Zs, 0)
    D30 = pairwise_dist_at_layer(Zs, 29)

    # Brain-adaptive thresholds: scan for L1 percentile-based cutoff
    iu = np.triu_indices(len(rep_idxs), 1)
    d1_all = D1[iu]; d30_all = D30[iu]
    d1_p25 = float(np.percentile(d1_all, 25))
    d30_p75 = float(np.percentile(d30_all, 75))
    d30_p95 = float(np.percentile(d30_all, 95))
    print(f"  brain pair distribution: L1 p25={d1_p25:.2f}, "
          f"L30 p75={d30_p75:.2f} p95={d30_p95:.2f}")
    # Divergent: L1 in bottom quartile AND L30 in top quartile
    D1_TH = d1_p25          # was 5.0
    D30_TH = d30_p75        # was 15.0
    print(f"  Divergent thresholds: d(L1) ≤ {D1_TH:.2f} AND d(L30) ≥ {D30_TH:.2f}")

    candidates = []
    for a in range(len(rep_idxs)):
        for b in range(a + 1, len(rep_idxs)):
            d1, d30 = D1[a, b], D30[a, b]
            if d1 > D1_TH or d30 < D30_TH: continue
            i_a, i_b = rep_idxs[a], rep_idxs[b]
            gos_a = set(np.where(labels[i_a] == 1)[0].tolist())
            gos_b = set(np.where(labels[i_b] == 1)[0].tolist())
            if not gos_a or not gos_b or (gos_a & gos_b):
                continue
            candidates.append({
                "i_a": int(i_a), "i_b": int(i_b),
                "gene_a": str(gene_names[i_a]),
                "gene_b": str(gene_names[i_b]),
                "gos_a": [list(GO_18.keys())[i] for i in sorted(gos_a)],
                "gos_b": [list(GO_18.keys())[i] for i in sorted(gos_b)],
                "d_L1": float(d1), "d_L30": float(d30),
                "score": float(d30 - d1),
            })
    candidates.sort(key=lambda x: x["score"], reverse=True)
    seen_gene_pair = set(); seen_go_pair = set(); top = []
    for c in candidates:
        gpair = tuple(sorted([c["gene_a"], c["gene_b"]]))
        if gpair in seen_gene_pair: continue
        go_pair = tuple(sorted([c["gos_a"][0], c["gos_b"][0]]))
        if go_pair in seen_go_pair: continue
        seen_gene_pair.add(gpair); seen_go_pair.add(go_pair)
        top.append(c)
        if len(top) >= 6: break
    return top


def find_peak_by_bracket(Z, coding_idx, labels, gene_names):
    """For each GO, compute a proxy Fisher signal per layer:
       ratio = between-class-mean-distance / within-class-mean-distance
       Peak layer = argmax.
       Bracket by Early/Mid/Late; pick top-2 GOs per bracket."""
    per_go_peak = {}
    for gi, (go, name) in enumerate(GO_18.items()):
        pos = np.where(labels[:, gi] == 1)[0]
        pos = np.intersect1d(pos, coding_idx)
        neg = np.where(labels[:, gi] == 0)[0]
        neg = np.intersect1d(neg, coding_idx)
        if len(pos) < 30 or len(neg) < 30: continue
        n_samp = min(200, len(pos), len(neg))
        pos_s = rng.choice(pos, size=n_samp, replace=False)
        neg_s = rng.choice(neg, size=n_samp, replace=False)
        signal = np.zeros(N_LAYERS)
        for L in range(N_LAYERS):
            mu_p = Z[pos_s, L, :].mean(0)
            mu_n = Z[neg_s, L, :].mean(0)
            sig_p = Z[pos_s, L, :].std(0) + 1e-6
            sig_n = Z[neg_s, L, :].std(0) + 1e-6
            between = np.linalg.norm(mu_p - mu_n)
            within = np.sqrt((sig_p ** 2 + sig_n ** 2).mean())
            signal[L] = between / within
        peak = int(np.argmax(signal)) + 1
        per_go_peak[go] = {
            "name": name,
            "peak_layer": peak,
            "peak_signal": float(signal.max()),
            "signal_curve": [float(x) for x in signal],
        }
    # bucket
    bracket = {"Early (L1-10)": [], "Mid (L11-20)": [], "Late (L21-30)": []}
    for go, info in per_go_peak.items():
        pl = info["peak_layer"]
        if pl <= 10: bracket["Early (L1-10)"].append((go, info))
        elif pl <= 20: bracket["Mid (L11-20)"].append((go, info))
        else: bracket["Late (L21-30)"].append((go, info))
    # sort each bracket by peak_signal desc, top-2
    for k in bracket:
        bracket[k].sort(key=lambda kv: kv[1]["peak_signal"], reverse=True)
    return per_go_peak, bracket


def draw_bundle_panel(ax, Z, coding_idx, labels, gene_names,
                      go, name, peak_L, n_context=12, n_neg=12):
    gi = list(GO_18.keys()).index(go)
    pos = np.where(labels[:, gi] == 1)[0]
    neg = np.where(labels[:, gi] == 0)[0]
    pos = np.intersect1d(pos, coding_idx)
    neg = np.intersect1d(neg, coding_idx)

    if len(pos) < 5:
        ax.set_axis_off(); ax.set_title(f'{name} (no signal)', fontsize=9)
        return

    ctx_pos = rng.choice(pos, size=min(n_context, len(pos)), replace=False)
    ctx_neg = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)

    for oi in ctx_pos:
        pts = Z[oi, :, :3]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color='#1E88E5', lw=0.8, alpha=0.35, zorder=2)
    for oi in ctx_neg:
        pts = Z[oi, :, :3]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color='crimson', lw=0.5, alpha=0.25, zorder=2)

    bm_pos = Z[ctx_pos, :, :3].mean(axis=0)
    ax.plot(bm_pos[:, 0], bm_pos[:, 1], bm_pos[:, 2],
            color='#0D47A1', lw=3.0, alpha=0.95, zorder=5,
            label='pos mean')

    bm_neg = Z[ctx_neg, :, :3].mean(axis=0)
    ax.plot(bm_neg[:, 0], bm_neg[:, 1], bm_neg[:, 2],
            color='#B71C1C', lw=2.0, alpha=0.9, ls='--', zorder=4,
            label='neg mean')

    # peak marker
    ax.scatter(bm_pos[peak_L - 1, 0], bm_pos[peak_L - 1, 1], bm_pos[peak_L - 1, 2],
               color='#FFC107', marker='*', s=280,
               edgecolors='darkorange', lw=1.4, zorder=8,
               label=f'peak L{peak_L}')
    ax.scatter(bm_pos[0, 0], bm_pos[0, 1], bm_pos[0, 2],
               color='#0D47A1', marker='o', s=80,
               edgecolors='white', lw=1.0, zorder=7)
    ax.scatter(bm_pos[-1, 0], bm_pos[-1, 1], bm_pos[-1, 2],
               color='#0D47A1', marker='X', s=110,
               edgecolors='white', lw=1.0, zorder=7)

    ax.set_title(f'{name}\npeak L{peak_L}  ({len(pos)} pos, {len(neg)} neg)',
                 fontsize=10, pad=4)
    ax.set_xlabel('PC1', fontsize=8); ax.set_ylabel('PC2', fontsize=8); ax.set_zlabel('PC3', fontsize=8)
    ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=20, azim=-60)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.85)


def plot_pair_grid(Z, pairs, kind, out_name):
    """Plot conv or div case pairs as 2×3 3D panels."""
    fig = plt.figure(figsize=(18, 11))
    if kind == "conv":
        header = ("Brain — Convergent evolution: different genes → same GO   |   "
                  "3D joint-PCA trajectory (per-layer K=8 top-3 PCs)")
    else:
        header = ("Brain — Functional divergence: different genes → different GO   |   "
                  "3D joint-PCA trajectory")
    fig.suptitle(header, fontsize=13, fontweight='bold', y=0.995)

    for i, p in enumerate(pairs[:6]):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        pts_a = Z[p["i_a"], :, :3]
        pts_b = Z[p["i_b"], :, :3]
        ax.plot(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2],
                color='#1E88E5', lw=2.4, alpha=0.95, label='A', zorder=6)
        ax.plot(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2],
                color='#43A047', lw=2.4, alpha=0.95, label='B', zorder=6)
        for pts, col in [(pts_a, '#1E88E5'), (pts_b, '#43A047')]:
            ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2],
                       color=col, marker='o', s=90,
                       edgecolors='white', lw=1.2, zorder=8)
            ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                       color=col, marker='X', s=130,
                       edgecolors='white', lw=1.2, zorder=8)
        if kind == "conv":
            title = (f"{p.get('gene_a','?')} vs {p.get('gene_b','?')}\n"
                     f"{p.get('name','?')}\n"
                     f"d(L1)={p['d_L1']:.2f} → d(L30)={p['d_L30']:.2f}")
        else:
            go_a = p.get('gos_a', ['?'])[0]
            go_b = p.get('gos_b', ['?'])[0]
            go_a_name = GO_18.get(go_a, go_a)
            go_b_name = GO_18.get(go_b, go_b)
            title = (f"{p['gene_a']} ({go_a_name})\n"
                     f"vs {p['gene_b']} ({go_b_name})\n"
                     f"d(L1)={p['d_L1']:.2f} → d(L30)={p['d_L30']:.2f}")
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.set_zlabel('PC3', fontsize=8)
        ax.tick_params(labelsize=6, pad=0)
        ax.view_init(elev=20, azim=-60)
        if i == 0:
            ax.legend(fontsize=8, loc='upper left', framealpha=0.85)

    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig(OUT_DIR / out_name, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_name}")


def plot_bracket(Z, coding_idx, labels, gene_names, bracket, out_name):
    """3 rows × top-2 = 6 panels."""
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle(f'Brain — GO bundle 3D trajectory flow by Fisher-peak bracket\n'
                 f'Blue = GO-positive isoforms, Red dashed = negative reference',
                 fontsize=14, fontweight='bold', y=0.995)
    row_idx = 0
    for bkey, gos in bracket.items():
        for col, (go, info) in enumerate(gos[:2]):
            ax = fig.add_subplot(3, 2, row_idx * 2 + col + 1, projection='3d')
            draw_bundle_panel(ax, Z, coding_idx, labels, gene_names,
                              go, info["name"], info["peak_layer"])
        row_idx += 1

    # Add row headers
    for i, bkey in enumerate(list(bracket.keys())):
        fig.text(0.02, 0.83 - i * 0.31, bkey, fontsize=13, fontweight='bold',
                 rotation=90, va='center', ha='left', color='#333')
    fig.tight_layout(rect=[0.03, 0.02, 1, 0.96])
    fig.savefig(OUT_DIR / out_name, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_name}")


def main():
    t0 = time.time()
    if not check_prereq():
        print("ERROR: brain layers incomplete. Exit.")
        return

    print("\n[1] Load labels & coding mask …")
    labels, ids, gene_names = load_labels()
    print(f"  labels shape: {labels.shape}")
    coding_mask = np.load(BRAIN_LABEL_DIR / "brain_full_mask.npy")
    coding_idx = np.where(coding_mask == 1)[0]
    print(f"  coding: {len(coding_idx)}")

    print("\n[2] Per-layer PCA (30 × K=8) …")
    Z = load_brain_layers_and_pca(coding_mask)
    np.save(OUT_DIR / "brain_Z_te_30x8.npy", Z)
    print(f"  Z: {Z.shape}  [{time.time()-t0:.1f}s]")

    print("\n[3] Find convergent evolution pairs (diff gene, same GO) …")
    conv = find_convergent(Z, coding_idx, labels, gene_names)
    print(f"  Top {len(conv)}:")
    for i, c in enumerate(conv, 1):
        print(f"    #{i} {c['gene_a']} vs {c['gene_b']} ({c['name']})  "
              f"d(L1)={c['d_L1']:.2f} → d(L30)={c['d_L30']:.2f}")

    print("\n[4] Find divergent-function pairs (diff gene, diff GO) …")
    div = find_divergent(Z, coding_idx, labels, gene_names)
    print(f"  Top {len(div)}:")
    for i, d in enumerate(div, 1):
        print(f"    #{i} {d['gene_a']} vs {d['gene_b']}  "
              f"d(L1)={d['d_L1']:.2f} → d(L30)={d['d_L30']:.2f}")

    print("\n[5] Early/Mid/Late Fisher peak brackets …")
    per_go_peak, bracket = find_peak_by_bracket(Z, coding_idx, labels, gene_names)
    print("  Bracket counts:")
    for k, gs in bracket.items():
        print(f"    {k}: {len(gs)} GO — {[g[1]['name'] for g in gs[:3]]}")

    print("\n[6] Plot bracket bundle 3D flow …")
    plot_bracket(Z, coding_idx, labels, gene_names, bracket,
                 "fig_brain_bundle_by_bracket.png")

    print("\n[7] Plot conv/div case pairs (2×3 grid each) …")
    plot_pair_grid(Z, conv, "conv", "fig_brain_convergent_pairs.png")
    if div:
        plot_pair_grid(Z, div, "div", "fig_brain_divergent_pairs.png")
    else:
        print("  [skip] no divergent pairs")

    # Save case summary
    summary = {
        "convergent_top": conv,
        "divergent_top": div,
        "peak_by_go": per_go_peak,
        "bracket_top2": {k: [{"go": go, **info} for go, info in v[:2]]
                          for k, v in bracket.items()},
    }
    with open(OUT_DIR / "brain_case_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {(time.time()-t0)/60:.1f} min. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
