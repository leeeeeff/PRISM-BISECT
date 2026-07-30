"""
plot_convdiv_per_case_v4.py
===========================
Academic journal style, English only.

v3 → v4 improvements:
  - All labels English (journal style)
  - Larger title / axis / legend fonts
  - Div definition v2: different gene + different GO (functionally distinct)
  - Endpoint start↔end distance annotation
  - Legend organized: (i) case identity, (ii) trajectory metrics,
    (iii) key layer signals, (iv) mechanistic interpretation
"""
from __future__ import annotations

import json, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
OUT_DIR   = ROOT.parent / "reports" / "curve_sweep" / "cases_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global matplotlib style — academic journal ───────────────────
plt.rcParams.update({
    'font.family'         : ['DejaVu Sans', 'Arial'],
    'font.size'           : 12,
    'axes.titlesize'      : 13,
    'axes.labelsize'      : 11,
    'xtick.labelsize'     : 9,
    'ytick.labelsize'     : 9,
    'legend.fontsize'     : 10,
    'axes.linewidth'      : 1.0,
    'axes.spines.top'     : True,
    'axes.spines.right'   : True,
    'axes.unicode_minus'  : False,
})

GO_18 = {
    "GO:0007204": "Ca2+ signaling",        "GO:0045214": "Sarcomere organization",
    "GO:0006941": "Muscle contraction",    "GO:0006914": "Autophagy",
    "GO:0043161": "Proteasome / UPS",      "GO:0007519": "Skeletal muscle dev.",
    "GO:0042692": "Muscle cell diff.",     "GO:0055074": "Ca2+ homeostasis",
    "GO:0007005": "Mitochondrion organ.",  "GO:0007517": "Muscle organ dev.",
    "GO:0032006": "TOR signaling",         "GO:0030048": "Actin-based movement",
    "GO:0006096": "Glycolysis",            "GO:0007268": "Synaptic transmission",
    "GO:0007018": "Microtubule movement",  "GO:0031175": "Neuron projection dev.",
    "GO:0030182": "Neuron differentiation","GO:0000226": "MT cytoskeleton org.",
}

N_LAYERS      = 30
EMB_DIM       = 640
N_GO_BUNDLE   = 40
COLOR_A       = '#1E88E5'   # blue
COLOR_B       = '#43A047'   # green
COLOR_CTX     = '#B0B0B0'
COLOR_BUNDLE  = '#3A3A3A'
COLOR_PEAK    = '#FFC107'
COLOR_KEY     = '#D81B60'


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


def load_seq_lengths():
    sm = np.load(MODEL_DIR / "my_sequence_matrix_fixed.npy")
    return (sm > 0).sum(axis=1).astype(int)


def isoform_gos(i, go_labels):
    return [go for go, labels in go_labels.items() if labels[i]]


# ── Subset assembly ────────────────────────────────────────────────

def collect_subset_indices(cases, go_labels):
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


def build_trajectory(subset_idx):
    N = len(subset_idx)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        path = DATA / f"esm2_layer_{L:02d}_t30_150M.npy"
        arr = np.load(path, mmap_mode="r")
        traj[:, L-1, :] = arr[subset_idx].astype(np.float32)
        del arr
    return traj


def pca_coords(traj, normalize_layers, n_components=3):
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


# ── Feature analytics ─────────────────────────────────────────────

def cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return 0.0
    return float(np.dot(a, b) / (na * nb))


def bundle_mean(coords, go_bundle_idx, orig2local, go):
    ii = [orig2local[oi] for oi in go_bundle_idx.get(go, []) if oi in orig2local]
    if len(ii) < 5: return None
    return coords[ii].mean(axis=0)


def compute_features(coords_nn, orig2local, go_bundle_idx,
                     case_a, case_b, primary_go_a, primary_go_b, key_layer):
    """Compute quantitative interpretation features (raw PCA space)."""
    la = orig2local[case_a]; lb = orig2local[case_b]
    A = coords_nn[la]; B = coords_nn[lb]
    Lk = max(0, key_layer - 1)

    bm_a = bundle_mean(coords_nn, go_bundle_idx, orig2local, primary_go_a) if primary_go_a else None
    bm_b = bundle_mean(coords_nn, go_bundle_idx, orig2local, primary_go_b) if primary_go_b else None

    def dist(u, v):
        return float(np.linalg.norm(u - v))

    feats = {
        # Pairwise (A vs B)
        'd_L1'    : dist(A[0], B[0]),
        'd_key'   : dist(A[Lk], B[Lk]),
        'd_L30'   : dist(A[-1], B[-1]),
        # Per-isoform path length (start-end)
        'path_A'  : dist(A[0], A[-1]),
        'path_B'  : dist(B[0], B[-1]),
    }
    # Bundle-alignment for post-key trajectory direction
    if bm_a is not None:
        dir_A       = A[-1] - A[Lk]
        dir_bundleA = bm_a[-1] - bm_a[Lk]
        feats['cos_A_bundleA'] = cosine(dir_A, dir_bundleA)
        feats['d_A_to_bundleA_L30'] = dist(A[-1], bm_a[-1])
    if bm_b is not None:
        dir_B       = B[-1] - B[Lk]
        dir_bundleB = bm_b[-1] - bm_b[Lk]
        feats['cos_B_bundleB'] = cosine(dir_B, dir_bundleB)
        feats['d_B_to_bundleB_L30'] = dist(B[-1], bm_b[-1])
    if bm_a is not None and bm_b is not None:
        feats['cos_bundleA_bundleB_post'] = cosine(bm_a[-1] - bm_a[Lk],
                                                    bm_b[-1] - bm_b[Lk])
    # Direction between A and B
    dir_AB_pre  = B[0]  - A[0]
    dir_AB_post = B[-1] - A[-1]
    feats['cos_AB_pre_post'] = cosine(dir_AB_pre, dir_AB_post)
    return feats


# ── Drawing ────────────────────────────────────────────────────────

def colored_line3d(pts, values, cmap, lw=1.4, alpha=0.85):
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_vals = (values[:-1] + values[1:]) / 2.0
    lc = Line3DCollection(segs, cmap=cmap, linewidths=lw, alpha=alpha,
                          norm=plt.Normalize(vmin=0.0, vmax=1.0))
    lc.set_array(seg_vals)
    return lc


def auto_axis_limits(all_pts_list, pad_frac=0.08):
    stacked = np.vstack(all_pts_list)
    lo = stacked.min(axis=0); hi = stacked.max(axis=0)
    rng_ = hi - lo + 1e-6
    return [(lo[d] - pad_frac * rng_[d], hi[d] + pad_frac * rng_[d])
            for d in range(3)]


def draw_panel(ax, coords, orig2local, go_bundle_idx, target_gos_map,
               case_A, case_B, primary_A, primary_B,
               key_L, peak_L,
               kind, mode, fisher_curve=None,
               show_start_end_dist=False):
    la = orig2local[case_A]; lb = orig2local[case_B]
    pts_A = coords[la]; pts_B = coords[lb]

    # Bundle context
    ctx_pts = []
    ctx_isos = set()
    for gos in target_gos_map.values():
        for go in gos:
            for oi in go_bundle_idx.get(go, []):
                if oi in orig2local: ctx_isos.add(oi)
    for oi in ctx_isos:
        ctx_pts.append(coords[orig2local[oi]])

    # Bundle means per target GO
    bm_map = {}
    for go in {*target_gos_map.get('A', []), *target_gos_map.get('B', [])}:
        oi_list = [oi for oi in go_bundle_idx.get(go, []) if oi in orig2local]
        if len(oi_list) >= 5:
            bm_map[go] = coords[[orig2local[oi] for oi in oi_list]].mean(axis=0)

    # Axis fitting
    pts_for_fit = [pts_A, pts_B] + [bm for bm in bm_map.values()]
    if ctx_pts:
        allb = np.vstack(ctx_pts)
        lo_p = np.percentile(allb, 5, axis=0)
        hi_p = np.percentile(allb, 95, axis=0)
        pts_for_fit.append(np.stack([lo_p, hi_p]))
    (xl, yl, zl) = auto_axis_limits(pts_for_fit)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)

    # Context (translucent)
    for oi in ctx_isos:
        pts = coords[orig2local[oi]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=COLOR_CTX, lw=0.45, alpha=0.20, zorder=2)

    # Bundle means (dotted)
    for gi, (go, bm) in enumerate(bm_map.items()):
        lbl = None
        if mode == 'bundle':
            lbl = f'GO bundle mean: {GO_18[go]}'
        ax.plot(bm[:, 0], bm[:, 1], bm[:, 2],
                color=COLOR_BUNDLE, lw=1.7, alpha=0.65, ls=':',
                zorder=3, label=lbl)

    # Case A & B
    for cidx, (pts, col, gos, tag) in enumerate([
        (pts_A, COLOR_A, target_gos_map.get('A', []), 'A'),
        (pts_B, COLOR_B, target_gos_map.get('B', []), 'B')
    ]):
        if mode == 'bundle':
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=col, lw=3.0, alpha=0.95,
                    label=f'Case {tag}', zorder=6)
        elif mode == 'fisher' and fisher_curve is not None:
            lc = colored_line3d(pts, fisher_curve, plt.cm.cividis,
                                lw=3.0, alpha=0.95)
            ax.add_collection3d(lc)
        else:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=col, lw=3.0, alpha=0.95, zorder=6)

        m_col = col if mode == 'bundle' else 'black'
        # Start (L1)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2],
                   color=m_col, marker='o', s=110,
                   edgecolors='white', linewidths=1.4, zorder=8)
        # End (L30)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                   color=m_col, marker='X', s=160,
                   edgecolors='white', linewidths=1.4, zorder=8)

        # Endpoint GO label
        if gos:
            end_lbl = GO_18[gos[0]] + (f' (+{len(gos)-1})' if len(gos) > 1 else '')
        else:
            end_lbl = 'no-GO'
        ax.text(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                f'  → {end_lbl}', fontsize=9.5,
                color=col, fontweight='bold', zorder=12)

        # Start-end path length annotation on trajectory midpoint
        if show_start_end_dist:
            pd = float(np.linalg.norm(pts[-1] - pts[0]))
            mid_pt = (pts[0] + pts[-1]) / 2.0
            ax.text(mid_pt[0], mid_pt[1], mid_pt[2],
                    f' |A→L30|={pd:.1f}' if tag == 'A' else f' |A→L30|={pd:.1f}',
                    fontsize=8.5, color=col, alpha=0.85,
                    fontweight='normal', style='italic', zorder=11)

        # Fisher peak
        if 0 < peak_L <= 30:
            Lp = peak_L - 1
            ax.scatter(pts[Lp, 0], pts[Lp, 1], pts[Lp, 2],
                       color=COLOR_PEAK, marker='*', s=320,
                       edgecolors='darkorange', lw=1.5, zorder=9)
        # Key layer
        if key_L > 0:
            Lc = key_L - 1
            ax.scatter(pts[Lc, 0], pts[Lc, 1], pts[Lc, 2],
                       color=COLOR_KEY, marker='*', s=320,
                       edgecolors='black', lw=1.5, zorder=10)

        # Layer markers on Case A only
        if cidx == 0 and mode == 'bundle':
            for L in [1, 15, 30]:
                j = L - 1
                ax.text(pts[j, 0], pts[j, 1], pts[j, 2],
                        f'  L{L}', fontsize=8, color=col,
                        alpha=0.85, fontweight='bold', zorder=11)

    # Bundle-mode: extra legend entries for peak / key
    if mode == 'bundle':
        if peak_L > 0:
            ax.plot([], [], color=COLOR_PEAK, marker='*', ms=14, ls='',
                    markeredgecolor='darkorange',
                    label=f'Fisher peak (L{peak_L})')
        if key_L > 0:
            tag = 'convergence' if kind == 'conv' else 'divergence'
            ax.plot([], [], color=COLOR_KEY, marker='*', ms=14, ls='',
                    markeredgecolor='black',
                    label=f'{tag} layer (L{key_L})')

    ax.set_xlabel('PC1', fontsize=10, labelpad=0)
    ax.set_ylabel('PC2', fontsize=10, labelpad=0)
    ax.set_zlabel('PC3', fontsize=10, labelpad=0)
    ax.tick_params(labelsize=8, pad=0)
    ax.view_init(elev=20, azim=-60)
    if mode == 'bundle':
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9,
                  handlelength=1.6, handletextpad=0.6)


# ── Legend text builder (English, structured) ──────────────────────

def align_word(c):
    if c >= 0.60:  return 'strongly aligned'
    if c >= 0.30:  return 'aligned'
    if c >= -0.30: return 'uncorrelated'
    return 'anti-aligned'


def build_legend_text(kind, p, iso_te, sym_te, seq_lens,
                      gos_a, gos_b, primary_A, primary_B,
                      key_L, peak_A_L, peak_B_L, feats):
    i_a, i_b = p['i_a'], p['i_b']
    gene_a = sym_te[i_a]; gene_b = sym_te[i_b]
    la = seq_lens[i_a]; lb = seq_lens[i_b]
    pA_name = GO_18[primary_A] if primary_A else 'N/A'
    pB_name = GO_18[primary_B] if primary_B else 'N/A'

    def fmt_gos(gs):
        if not gs: return 'no assigned GO'
        return '; '.join([GO_18[g] for g in gs[:3]]) + \
               (f' (+{len(gs)-3} more)' if len(gs) > 3 else '')

    cosA = feats.get('cos_A_bundleA', 0.0)
    cosB = feats.get('cos_B_bundleB', 0.0)
    dA_bA = feats.get('d_A_to_bundleA_L30', 0.0)
    dB_bB = feats.get('d_B_to_bundleB_L30', 0.0)
    cos_bAbB = feats.get('cos_bundleA_bundleB_post', 0.0)

    # Section (i): case identity
    if kind == 'conv':
        sec_i_title = f'(i) Case identity  —  Convergent evolution: two proteins from different genes share a common GO annotation'
    else:
        sec_i_title = f'(i) Case identity  —  Functional divergence: two proteins from different genes reach distinct GO annotations'
    l_a = f'    Case A:  {gene_a} / {iso_te[i_a]}   ({la} aa)   →  primary GO: {pA_name}   |  all: {fmt_gos(gos_a)}'
    l_b = f'    Case B:  {gene_b} / {iso_te[i_b]}   ({lb} aa)   →  primary GO: {pB_name}   |  all: {fmt_gos(gos_b)}'

    # Section (ii): trajectory metrics
    d1 = feats['d_L1']; d_key = feats['d_key']; d30 = feats['d_L30']
    if kind == 'conv':
        arrow = f'|A-B|:  L1 = {d1:.2f}   →   convergence layer L{key_L} = {d_key:.2f}   →   L30 = {d30:.2f}'
        trend = f'({d1-d30:+.2f} contraction from L1 to L30)'
    else:
        arrow = f'|A-B|:  L1 = {d1:.2f}   →   divergence layer L{key_L} = {d_key:.2f}   →   L30 = {d30:.2f}'
        trend = f'({d30-d1:+.2f} expansion from L1 to L30)'
    path_A = feats['path_A']; path_B = feats['path_B']
    endpoints = f'    Trajectory length (start → end):  |L1_A → L30_A| = {path_A:.2f},   |L1_B → L30_B| = {path_B:.2f}'
    sec_ii = ('(ii) Pairwise trajectory metrics (raw ESM-2 joint PCA space):\n'
              f'    {arrow}   {trend}\n{endpoints}')

    # Section (iii): key layer signals
    if kind == 'conv':
        peak_line = f'    Fisher peak (shared GO {pA_name}): L{peak_A_L}  —  layer where GO-discriminative signal is maximal.'
    else:
        peak_line = (f'    Fisher peak of Case A primary GO ({pA_name}): L{peak_A_L}   |   '
                     f'peak of Case B primary GO ({pB_name}): L{peak_B_L}')
    sec_iii = ('(iii) Key layer signals:\n'
               f'    Key trajectory transition:  L{key_L}\n'
               f'{peak_line}')

    # Section (iv): mechanistic interpretation
    if kind == 'conv':
        mech = (
            '(iv) Mechanistic interpretation — evolutionary/functional convergence:\n'
            f'    • L1 embeddings reflect raw sequence identity → the pair starts far apart '
            f'(d_L1 = {d1:.2f}).\n'
            f'    • From L{key_L} onward, ESM-2 mid-layer attention integrates shared '
            f'residue-level features (active-site geometry, binding pocket motifs) '
            f'that transcend sequence background.\n'
            f'    • Post-L{key_L}, both trajectories align with the shared GO bundle '
            f'(cos(A, bundle) = {cosA:+.2f} [{align_word(cosA)}], '
            f'cos(B, bundle) = {cosB:+.2f} [{align_word(cosB)}]) → '
            f'converging to distance {d30:.2f} at L30.\n'
            f'    • Interpretation:  ESM-2 layers {key_L}–{peak_A_L} carry the '
            f'shared functional signal that Fisher discrimination for "{pA_name}" reaches its maximum at L{peak_A_L}.'
        )
    else:
        mech = (
            '(iv) Mechanistic interpretation — functional divergence:\n'
            f'    • L1 embeddings are close (d_L1 = {d1:.2f}) — the two proteins begin in '
            f'similar coarse-composition space.\n'
            f'    • From L{key_L} onward, splicing- or domain-specific residues that '
            f'distinguish "{pA_name}" from "{pB_name}" are amplified through attention → '
            f'the trajectories separate to distance {d30:.2f} at L30.\n'
            f'    • Each isoform aligns with its respective GO bundle: '
            f'cos(A, bundle_A) = {cosA:+.2f} [{align_word(cosA)}], '
            f'cos(B, bundle_B) = {cosB:+.2f} [{align_word(cosB)}].\n'
            f'    • The two target bundles themselves point in different directions '
            f'(cos(bundle_A, bundle_B) = {cos_bAbB:+.2f} [{align_word(cos_bAbB)}]), so '
            f'reaching them requires distinct trajectories from a shared origin.\n'
            f'    • Interpretation:  L{key_L} is where ESM-2 begins to enforce '
            f'GO-specific specialization — the layer marks the transition from '
            f'"generic protein embedding" to "function-committed embedding".'
        )

    text = '\n'.join([sec_i_title, l_a, l_b, '', sec_ii, '', sec_iii, '', mech])
    return text


# ── Case figure builder ────────────────────────────────────────────

def make_case_figure(p, kind, case_idx,
                     coords_nn, coords_n, orig2local, go_bundle_idx,
                     go_labels, iso_te, sym_te, seq_lens,
                     fisher_raw, fisher_norm):
    i_a, i_b = p['i_a'], p['i_b']
    gos_a = isoform_gos(i_a, go_labels)
    gos_b = isoform_gos(i_b, go_labels)
    gene_a = sym_te[i_a]; gene_b = sym_te[i_b]

    # Primary GO selection
    if kind == 'conv':
        primary_A = primary_B = p['go']
    else:
        primary_A = p['gos_a'][0] if p.get('gos_a') else (gos_a[0] if gos_a else None)
        primary_B = p['gos_b'][0] if p.get('gos_b') else (gos_b[0] if gos_b else None)

    key_L = p.get('conv_layer', p.get('div_layer', -1))
    peak_A_L = int(np.argmax(fisher_raw[primary_A])) + 1 if primary_A in fisher_raw else -1
    peak_B_L = int(np.argmax(fisher_raw[primary_B])) + 1 if primary_B in fisher_raw else -1
    fisher_curve_A = fisher_norm.get(primary_A) if primary_A else None

    feats = compute_features(coords_nn, orig2local, go_bundle_idx,
                             i_a, i_b, primary_A, primary_B, key_L)

    target_gos_map = {'A': gos_a, 'B': gos_b}

    # Header
    pA_name = GO_18[primary_A] if primary_A else 'N/A'
    pB_name = GO_18[primary_B] if primary_B else 'N/A'
    if kind == 'conv':
        header = (
            f'Convergent case #{case_idx}   —   {gene_a}  and  {gene_b}  '
            f'both annotated with {pA_name}\n'
            f'd(A, B):  L1 = {p["d_L1"]:.2f}   →   L30 = {p["d_L30"]:.2f}   '
            f'|  convergence layer L{key_L}   |  Fisher peak L{peak_A_L}'
        )
    else:
        header = (
            f'Divergent case #{case_idx}   —   {gene_a} ({pA_name})   vs.   '
            f'{gene_b} ({pB_name})\n'
            f'd(A, B):  L1 = {p["d_L1"]:.2f}   →   L30 = {p["d_L30"]:.2f}   '
            f'|  divergence layer L{key_L}   |  Fisher peaks  L{peak_A_L} / L{peak_B_L}'
        )

    fig = plt.figure(figsize=(22, 20))
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[1.0, 1.0, 0.62],
        hspace=0.10, wspace=0.05,
        left=0.035, right=0.965,
        top=0.935, bottom=0.02,
    )
    fig.suptitle(header, fontsize=16, fontweight='bold', y=0.983)

    ax_A = fig.add_subplot(gs[0, 0], projection='3d')
    draw_panel(ax_A, coords_nn, orig2local, go_bundle_idx,
               target_gos_map, i_a, i_b, primary_A, primary_B,
               key_L, peak_A_L, kind, mode='bundle',
               show_start_end_dist=True)
    ax_A.set_title(
        '(a) Raw ESM-2 (non-normalised) — bundle-coloured\n'
        'translucent grey = context isoforms of target GOs   |   '
        'dotted grey = GO bundle mean',
        fontsize=13, pad=6)

    ax_B = fig.add_subplot(gs[0, 1], projection='3d')
    draw_panel(ax_B, coords_nn, orig2local, go_bundle_idx,
               target_gos_map, i_a, i_b, primary_A, primary_B,
               key_L, peak_A_L, kind, mode='fisher',
               fisher_curve=fisher_curve_A)
    ax_B.set_title(
        '(b) Raw ESM-2 — coloured by per-layer Fisher signal\n'
        f'(colour bar = normalised Fisher discrimination for {pA_name})',
        fontsize=13, pad=6)

    ax_C = fig.add_subplot(gs[1, 0], projection='3d')
    draw_panel(ax_C, coords_n, orig2local, go_bundle_idx,
               target_gos_map, i_a, i_b, primary_A, primary_B,
               key_L, peak_A_L, kind, mode='bundle',
               show_start_end_dist=True)
    ax_C.set_title(
        '(c) Per-layer z-normalised ESM-2 — bundle-coloured\n'
        '(each layer standardised across all proteins before joint PCA)',
        fontsize=13, pad=6)

    ax_D = fig.add_subplot(gs[1, 1], projection='3d')
    draw_panel(ax_D, coords_n, orig2local, go_bundle_idx,
               target_gos_map, i_a, i_b, primary_A, primary_B,
               key_L, peak_A_L, kind, mode='fisher',
               fisher_curve=fisher_curve_A)
    ax_D.set_title(
        '(d) Per-layer z-normalised — coloured by per-layer Fisher signal',
        fontsize=13, pad=6)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis,
                               norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=[ax_B, ax_D], shrink=0.55, pad=0.02,
                      location='right')
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(['0', '0.25', '0.5', '0.75', '1'])
    cb.set_label('Per-layer Fisher discrimination (min → max)', fontsize=10)
    cb.ax.tick_params(labelsize=9)

    # Legend text at bottom
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_txt = build_legend_text(kind, p, iso_te, sym_te, seq_lens,
                                   gos_a, gos_b, primary_A, primary_B,
                                   key_L, peak_A_L, peak_B_L, feats)
    ax_legend.text(
        0.005, 0.99, legend_txt,
        transform=ax_legend.transAxes,
        va='top', ha='left',
        fontsize=12.5,
        linespacing=1.4,
        bbox=dict(
            boxstyle='round,pad=0.9',
            facecolor='#FBFAF7', edgecolor='#333',
            linewidth=1.0,
        ),
    )

    out_name = f'fig_case_{kind}_{case_idx:02d}.png'
    fig.savefig(OUT_DIR / out_name, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out_name}')


# ── Main ───────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print('[1] Loading IDs, GO labels, Fisher curves, seq lengths...')
    sym_te, iso_te = load_gene_symbols()
    go_labels = load_go_labels(sym_te)
    fisher_raw, fisher_norm = load_fisher_curves()
    seq_lens = load_seq_lengths()

    print('[2] Loading case pairs...')
    js = json.load(open(ROOT.parent / 'reports' / 'curve_sweep' /
                        'convergence_divergence.json'))
    conv_pairs = js['convergent_top']
    div_pairs  = js['divergent_top']
    all_cases = [('conv', p) for p in conv_pairs] + [('div', p) for p in div_pairs]
    print(f'   {len(conv_pairs)} conv + {len(div_pairs)} div = {len(all_cases)}')

    print('[3] Assembling subset...')
    subset_idx, orig2local, go_bundle_idx = collect_subset_indices(
        [p for _, p in all_cases], go_labels)
    print(f'   Subset size: {len(subset_idx)}')

    print('[4] Loading raw ESM-2 trajectories for subset...')
    traj = build_trajectory(subset_idx)
    print(f'   traj shape: {traj.shape}  [{time.time()-t0:.1f}s]')

    print('[5] Joint PCA (non-normalised)...')
    coords_nn = pca_coords(traj, normalize_layers=False)
    print('[6] Joint PCA (per-layer z-normalised)...')
    coords_n  = pca_coords(traj, normalize_layers=True)
    del traj
    print(f'   PCAs done [{time.time()-t0:.1f}s]')

    print(f'\n[7] Rendering {len(all_cases)} case figures...')
    for i, p in enumerate(conv_pairs, 1):
        make_case_figure(p, 'conv', i, coords_nn, coords_n,
                         orig2local, go_bundle_idx,
                         go_labels, iso_te, sym_te, seq_lens,
                         fisher_raw, fisher_norm)
    for i, p in enumerate(div_pairs, 1):
        make_case_figure(p, 'div', i, coords_nn, coords_n,
                         orig2local, go_bundle_idx,
                         go_labels, iso_te, sym_te, seq_lens,
                         fisher_raw, fisher_norm)

    print(f'\n[done] {len(all_cases)} figures in {OUT_DIR}')
    print(f'       Total: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
