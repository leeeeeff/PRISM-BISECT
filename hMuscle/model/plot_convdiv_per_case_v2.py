"""
plot_convdiv_per_case_v2.py
===========================
케이스별 2x2 figure with 실제 non-normalized vs normalized PCA 구축.

각 case별 2×2 패널:
  (A) Non-normalized 3D: raw ESM-2 → joint PCA
      - Case A/B 궤적 (colored)
      - Target GO bundle (translucent gray = 대표 아이소폼들)
      - start/end/conv_or_div/Fisher_peak 마커
  (B) Non-normalized 3D + layer-signal cividis (Fisher-colored)
  (C) Normalized 3D: per-layer z-score → joint PCA
      - 동일 요소, per-layer 정규화된 공간
  (D) Normalized 3D + layer-signal cividis

Subset: 12 case × 2 iso = 24, + 50 대표 아이소폼 × 18 GO = 900, 총 ~924.
Raw ESM-2 layer files 30개를 subset만 추출하여 joint PCA에 사용.
"""
from __future__ import annotations

import json, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA
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
OUT_DIR   = ROOT.parent / "reports" / "curve_sweep" / "cases_v2"
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

N_LAYERS       = 30
EMB_DIM        = 640
N_GO_BUNDLE    = 40   # 대표 아이소폼 per GO
CASE_COLORS    = ('#1E88E5', '#43A047')   # A=blue, B=green


# ── Load ───────────────────────────────────────────────────────────

def clean_bytestr(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


def load_gene_symbols():
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
    te_gene = np.load(MODEL_DIR / "my_gene_list_fixed.npy", allow_pickle=True)
    te_iso  = np.load(MODEL_DIR / "my_isoform_list_fixed.npy", allow_pickle=True)
    sym_te = [ENSG2SYM.get(clean_bytestr(g).split('.')[0],
                           clean_bytestr(g).split('.')[0]) for g in te_gene]
    iso_te = [clean_bytestr(i) for i in te_iso]
    return sym_te, iso_te


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


def load_fisher_curves():
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
    fisher_norm = {}
    for go, curve in fisher_raw.items():
        fisher_norm[go] = (curve - curve.min()) / (curve.max() - curve.min() + 1e-9)
    return fisher_raw, fisher_norm


def isoform_gos(i, go_labels):
    return [go for go, labels in go_labels.items() if labels[i]]


# ── Subset assembly ────────────────────────────────────────────────

def collect_subset_indices(cases, go_labels):
    """Union of case isoforms + N_GO_BUNDLE random positives per GO."""
    idx_set = set()
    for c in cases:
        idx_set.add(c['i_a']); idx_set.add(c['i_b'])
    go_bundle_idx = {}
    for go in GO_18:
        pos_idx = np.where(go_labels[go])[0]
        n = min(N_GO_BUNDLE, len(pos_idx))
        if n > 0:
            chosen = rng.choice(pos_idx, size=n, replace=False)
            go_bundle_idx[go] = chosen.tolist()
            idx_set.update(chosen.tolist())
        else:
            go_bundle_idx[go] = []
    subset_idx = np.array(sorted(idx_set), dtype=int)
    orig2local = {oi: li for li, oi in enumerate(subset_idx)}
    return subset_idx, orig2local, go_bundle_idx


def build_trajectory(subset_idx: np.ndarray) -> np.ndarray:
    """Load raw ESM-2 layer embeddings for subset → (N_sub, 30, 640)."""
    N = len(subset_idx)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        path = DATA / f"esm2_layer_{L:02d}_t30_150M.npy"
        arr = np.load(path, mmap_mode="r")
        traj[:, L-1, :] = arr[subset_idx].astype(np.float32)
        del arr
    return traj


def pca_coords(traj: np.ndarray, normalize_layers: bool, n_components=3):
    """Trajectory → 3D joint PCA coords."""
    if normalize_layers:
        traj = traj.copy()
        for L in range(N_LAYERS):
            mu = traj[:, L, :].mean(0)
            sd = traj[:, L, :].std(0) + 1e-6
            traj[:, L, :] = (traj[:, L, :] - mu) / sd
    N = traj.shape[0]
    flat = traj.reshape(N * N_LAYERS, EMB_DIM)
    pca = PCA(n_components=n_components, random_state=SEED)
    reduced = pca.fit_transform(flat).reshape(N, N_LAYERS, n_components)
    print(f"   PCA (norm={normalize_layers}): expl_var_top3 = "
          f"{pca.explained_variance_ratio_.sum():.3f}")
    return reduced.astype(np.float32)


# ── Drawing ────────────────────────────────────────────────────────

def colored_line3d(pts, values, cmap, lw=1.4, alpha=0.85):
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_vals = (values[:-1] + values[1:]) / 2.0
    lc = Line3DCollection(segs, cmap=cmap, linewidths=lw, alpha=alpha,
                          norm=plt.Normalize(vmin=0.0, vmax=1.0))
    lc.set_array(seg_vals)
    return lc


def draw_panel(ax, coords, orig2local, go_bundle_idx, target_gos,
               case_A_orig, case_B_orig,
               conv_or_div_L, fisher_peak_L,
               kind, mode: str, fisher_curve=None):
    """
    Draw a single 3D panel.
    mode: 'bundle' (color-coded case, gray target GO bundle context)
          'fisher' (case colored by Fisher signal, gray context)
    fisher_curve: (30,) normalized Fisher curve for the shared GO (mode='fisher')
    """
    # Target GO bundle context (translucent gray)
    bundle_isos = set()
    for go in target_gos:
        for oi in go_bundle_idx.get(go, []):
            if oi in orig2local:
                bundle_isos.add(oi)

    for oi in bundle_isos:
        li = orig2local[oi]
        pts = coords[li]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color='#909090', lw=0.55, alpha=0.28, zorder=2)

    # Bundle mean of each target GO (thicker translucent gray)
    for gi, go in enumerate(target_gos):
        oi_list = [oi for oi in go_bundle_idx.get(go, [])
                   if oi in orig2local]
        if len(oi_list) < 5: continue
        li_list = [orig2local[oi] for oi in oi_list]
        bm = coords[li_list].mean(axis=0)
        ax.plot(bm[:, 0], bm[:, 1], bm[:, 2],
                color='#454545', lw=1.6, alpha=0.55, ls=':', zorder=3,
                label=f'GO bundle mean: {GO_18[go]}' if gi < 2 else None)

    # Case trajectories
    for cidx, (case_orig, col_default) in enumerate(
            [(case_A_orig, CASE_COLORS[0]), (case_B_orig, CASE_COLORS[1])]):
        li = orig2local[case_orig]
        pts = coords[li]

        if mode == 'bundle':
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=col_default, lw=2.6, alpha=0.95,
                    label=f"Case {'A' if cidx == 0 else 'B'}",
                    zorder=6)
        elif mode == 'fisher' and fisher_curve is not None:
            lc = colored_line3d(pts, fisher_curve, plt.cm.cividis,
                                lw=2.6, alpha=0.95)
            ax.add_collection3d(lc)
        else:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=col_default, lw=2.6, alpha=0.95, zorder=6)

        # Start/end markers
        marker_col = col_default if mode == 'bundle' else 'black'
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2],
                   color=marker_col, marker='o', s=100,
                   edgecolors='white', linewidths=1.2, zorder=8)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                   color=marker_col, marker='X', s=140,
                   edgecolors='white', linewidths=1.2, zorder=8)

        # Fisher peak marker (gold star)
        if 0 < fisher_peak_L <= 30:
            Lp = fisher_peak_L - 1
            ax.scatter(pts[Lp, 0], pts[Lp, 1], pts[Lp, 2],
                       color='gold', marker='*', s=260,
                       edgecolors='darkorange', lw=1.4, zorder=9)

        # Conv/Div layer marker (red star)
        if conv_or_div_L > 0:
            Lc = conv_or_div_L - 1
            ax.scatter(pts[Lc, 0], pts[Lc, 1], pts[Lc, 2],
                       color='red', marker='*', s=260,
                       edgecolors='black', lw=1.4, zorder=10)

        # Layer labels at key layers (only on case A trajectory)
        if cidx == 0 and mode == 'bundle':
            for L in [1, 10, 20, 30]:
                j = L - 1
                ax.text(pts[j, 0], pts[j, 1], pts[j, 2],
                        f'  L{L}', fontsize=6.5, color=col_default,
                        alpha=0.9, fontweight='bold', zorder=11)

    # Legend markers for peak / conv layer
    if mode == 'bundle':
        if fisher_peak_L > 0:
            ax.plot([], [], color='gold', marker='*', ms=13, ls='',
                    markeredgecolor='darkorange',
                    label=f'Fisher peak L{fisher_peak_L}')
        if conv_or_div_L > 0:
            tag = 'conv' if kind == 'conv' else 'div'
            ax.plot([], [], color='red', marker='*', ms=13, ls='',
                    markeredgecolor='black',
                    label=f'{tag}_layer L{conv_or_div_L}')

    ax.set_xlabel('PC1', fontsize=8, labelpad=0)
    ax.set_ylabel('PC2', fontsize=8, labelpad=0)
    ax.set_zlabel('PC3', fontsize=8, labelpad=0)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=20, azim=-60)
    if mode == 'bundle':
        ax.legend(fontsize=6.5, loc='upper left', framealpha=0.85)


# ── Case figure builder ────────────────────────────────────────────

def make_case_figure(p, kind, case_idx,
                     coords_nn, coords_n, orig2local, go_bundle_idx,
                     go_labels, iso_te, fisher_raw, fisher_norm):
    i_a, i_b = p['i_a'], p['i_b']
    gos_a = isoform_gos(i_a, go_labels)
    gos_b = isoform_gos(i_b, go_labels)
    target_gos = list(dict.fromkeys(gos_a + gos_b))   # unique, ordered
    gene_a = p.get('gene_a', p.get('gene', ''))
    gene_b = p.get('gene_b', p.get('gene', ''))

    conv_or_div_L = p.get('conv_layer', p.get('div_layer', -1))
    if kind == 'conv':
        primary_go = p['go']
    else:
        shared = p.get('shared_gos', [])
        primary_go = shared[0] if shared else (gos_a[0] if gos_a else None)
    fisher_peak_L = int(np.argmax(fisher_raw[primary_go])) + 1 \
                    if primary_go and primary_go in fisher_raw else -1
    fisher_curve = fisher_norm.get(primary_go) if primary_go else None

    # Header
    def fmt_gos(gs):
        if not gs: return "no-GO"
        return "; ".join([GO_18[g] for g in gs[:3]]) + \
               (f" +{len(gs)-3}" if len(gs) > 3 else "")

    if kind == 'conv':
        header = (
            f"[CONV Case #{case_idx}]  Shared GO: {GO_18[p['go']]}  ({p['go']})\n"
            f"A: {gene_a} ({iso_te[i_a]})  →  GOs: {fmt_gos(gos_a)}\n"
            f"B: {gene_b} ({iso_te[i_b]})  →  GOs: {fmt_gos(gos_b)}\n"
            f"d(L1)={p['d_L1']:.2f} → d(L30)={p['d_L30']:.2f}   "
            f"conv_layer=L{p['conv_layer']}   "
            f"primary Fisher peak: L{fisher_peak_L}"
        )
    else:
        header = (
            f"[DIV Case #{case_idx}]  Gene: {gene_a}  |  "
            f"shared GO count = {len(p.get('shared_gos', []))}\n"
            f"A: {iso_te[i_a]}   →  GOs: {fmt_gos(gos_a)}\n"
            f"B: {iso_te[i_b]}   →  GOs: {fmt_gos(gos_b)}\n"
            f"d(L1)={p['d_L1']:.2f} → d(L30)={p['d_L30']:.2f}   "
            f"div_layer=L{p['div_layer']}   "
            f"primary Fisher peak: L{fisher_peak_L}"
        )

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(header, fontsize=11, fontweight='bold', y=0.995)

    # (A) Non-normalized, bundle-coloured
    ax_A = fig.add_subplot(2, 2, 1, projection='3d')
    draw_panel(ax_A, coords_nn, orig2local, go_bundle_idx, target_gos,
               i_a, i_b, conv_or_div_L, fisher_peak_L,
               kind, mode='bundle')
    ax_A.set_title(
        '(A) Non-normalized 3D  •  bundle-coloured\n'
        '(gray=target GO bundle context, dotted=GO bundle mean)',
        fontsize=10, pad=6)

    # (B) Non-normalized, layer-signal cividis
    ax_B = fig.add_subplot(2, 2, 2, projection='3d')
    draw_panel(ax_B, coords_nn, orig2local, go_bundle_idx, target_gos,
               i_a, i_b, conv_or_div_L, fisher_peak_L,
               kind, mode='fisher', fisher_curve=fisher_curve)
    ax_B.set_title(
        '(B) Non-normalized 3D  •  layer-signal cividis\n'
        f'(case colored by Fisher signal of {GO_18[primary_go] if primary_go else "N/A"})',
        fontsize=10, pad=6)

    # (C) Normalized, bundle-coloured
    ax_C = fig.add_subplot(2, 2, 3, projection='3d')
    draw_panel(ax_C, coords_n, orig2local, go_bundle_idx, target_gos,
               i_a, i_b, conv_or_div_L, fisher_peak_L,
               kind, mode='bundle')
    ax_C.set_title(
        '(C) Normalized (per-layer z-score) 3D  •  bundle-coloured',
        fontsize=10, pad=6)

    # (D) Normalized, layer-signal cividis
    ax_D = fig.add_subplot(2, 2, 4, projection='3d')
    draw_panel(ax_D, coords_n, orig2local, go_bundle_idx, target_gos,
               i_a, i_b, conv_or_div_L, fisher_peak_L,
               kind, mode='fisher', fisher_curve=fisher_curve)
    ax_D.set_title(
        '(D) Normalized 3D  •  layer-signal cividis',
        fontsize=10, pad=6)

    # Colorbar shared for (B) and (D)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis,
                                norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=[ax_B, ax_D], shrink=0.5, pad=0.02, location='right')
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0 (Fisher min)", "0.25", "0.5", "0.75", "1 (peak)"])
    cb.set_label("per-layer Fisher signal (0-1 per GO)")

    fig.tight_layout(rect=[0, 0.01, 0.94, 0.93])
    out_name = f"fig_case_{kind}_{case_idx:02d}.png"
    fig.savefig(OUT_DIR / out_name, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_name}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("[1] Loading IDs, GO labels, Fisher curves...")
    sym_te, iso_te = load_gene_symbols()
    go_labels = load_go_labels(sym_te)
    fisher_raw, fisher_norm = load_fisher_curves()

    print("[2] Loading case pairs...")
    js = json.load(open(ROOT.parent / "reports" / "curve_sweep" /
                        "convergence_divergence.json"))
    conv_pairs = js["convergent_top"]
    div_pairs  = js["divergent_top"]
    all_cases = [('conv', p) for p in conv_pairs] + [('div', p) for p in div_pairs]
    print(f"   {len(conv_pairs)} conv + {len(div_pairs)} div = {len(all_cases)}")

    print("[3] Assembling subset...")
    subset_idx, orig2local, go_bundle_idx = collect_subset_indices(
        [p for _, p in all_cases], go_labels)
    print(f"   Subset size: {len(subset_idx)}")

    print(f"[4] Loading raw ESM-2 trajectories for subset "
          f"({len(subset_idx)}×{N_LAYERS}×{EMB_DIM})...")
    traj = build_trajectory(subset_idx)
    print(f"   traj shape: {traj.shape}  [{time.time()-t0:.1f}s]")

    print("[5] Joint PCA (non-normalized)...")
    coords_nn = pca_coords(traj, normalize_layers=False)
    print("[6] Joint PCA (normalized: per-layer z-score)...")
    coords_n  = pca_coords(traj, normalize_layers=True)
    del traj
    print(f"   both PCAs done [{time.time()-t0:.1f}s]")

    print(f"\n[7] Rendering {len(all_cases)} case figures...")
    for i, (kind, p) in enumerate([('conv', p) for p in conv_pairs], 1):
        make_case_figure(p, 'conv', i, coords_nn, coords_n,
                         orig2local, go_bundle_idx,
                         go_labels, iso_te, fisher_raw, fisher_norm)
    for i, p in enumerate(div_pairs, 1):
        make_case_figure(p, 'div', i, coords_nn, coords_n,
                         orig2local, go_bundle_idx,
                         go_labels, iso_te, fisher_raw, fisher_norm)

    print(f"\n[done] {len(all_cases)} figures in {OUT_DIR}")
    print(f"       Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
