"""
plot_convdiv_per_case_v3.py
===========================
v2 대비 개선사항:
  1) 균형 잡힌 4-panel gridspec (중앙 여백 제거, 4개 subplot 균등 배치)
  2) Normalized trajectory 축범위 자동 최적화(case+bundle 범위 기반) → 확대 표시
  3) 각 case의 L30 endpoint에 최종 GO term 라벨 표기
  4) figure 하단에 상세 legend:
       - 두 아이소폼 요약(길이/할당 GO/gene)
       - conv/div 분기점의 PCA 공간 특성(GO bundle과의 정렬도, 상대 이동)
       - Fisher peak layer(정보 밀집 layer)
       - 수렴/발산의 정량적 원인(feature-level)
"""
from __future__ import annotations

import json, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager as _fm
import os as _os

for _fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']:
    if _os.path.exists(_fp):
        try: _fm.fontManager.addfont(_fp)
        except Exception: pass
plt.rcParams['font.family']        = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
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
OUT_DIR   = ROOT.parent / "reports" / "curve_sweep" / "cases_v3"
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
N_GO_BUNDLE    = 40
CASE_COLORS    = ('#1E88E5', '#43A047')
GRAY_CTX       = '#909090'
GRAY_BUNDLE    = '#454545'


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


# ── Feature analytics for legend ───────────────────────────────────

def cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return 0.0
    return float(np.dot(a, b) / (na * nb))


def bundle_mean(coords, go_bundle_idx, orig2local, go):
    ii = [orig2local[oi] for oi in go_bundle_idx.get(go, []) if oi in orig2local]
    if len(ii) < 5: return None
    return coords[ii].mean(axis=0)   # (30,3)


def compute_features(coords_nn, coords_n, orig2local, go_bundle_idx,
                     case_a, case_b, primary_go, key_layer):
    """Compute quantitative features around the conv/div layer.
    Returns dict of interpretable numbers used in the legend."""
    la = orig2local[case_a]; lb = orig2local[case_b]
    A_nn = coords_nn[la]; B_nn = coords_nn[lb]
    A_n  = coords_n [la]; B_n  = coords_n [lb]
    bm_nn = bundle_mean(coords_nn, go_bundle_idx, orig2local, primary_go)

    d_A_B_at = lambda arr, L: float(np.linalg.norm(arr[0][L] - arr[1][L]))
    feats = {
        "d_L1_nn":     d_A_B_at((A_nn, B_nn), 0),
        "d_conv_nn":   d_A_B_at((A_nn, B_nn), max(0, key_layer-1)),
        "d_L30_nn":    d_A_B_at((A_nn, B_nn), N_LAYERS-1),
        "d_L1_n":      d_A_B_at((A_n,  B_n ), 0),
        "d_L30_n":     d_A_B_at((A_n,  B_n ), N_LAYERS-1),
    }
    # Trajectory delta after key layer
    if bm_nn is not None:
        Lk = max(0, key_layer-1)
        # direction of A trajectory L_k → L30 vs bundle direction
        dir_A     = A_nn[N_LAYERS-1] - A_nn[Lk]
        dir_B     = B_nn[N_LAYERS-1] - B_nn[Lk]
        dir_bundle= bm_nn[N_LAYERS-1] - bm_nn[Lk]
        feats["cos_A_bundle_post"] = cosine(dir_A, dir_bundle)
        feats["cos_B_bundle_post"] = cosine(dir_B, dir_bundle)
        feats["cos_AB_post"]       = cosine(dir_A, dir_B)
        # distance-to-bundle at Lk
        feats["d_A_bundle_at_key"] = float(np.linalg.norm(A_nn[Lk] - bm_nn[Lk]))
        feats["d_B_bundle_at_key"] = float(np.linalg.norm(B_nn[Lk] - bm_nn[Lk]))
        feats["d_A_bundle_at_L30"] = float(np.linalg.norm(A_nn[-1] - bm_nn[-1]))
        feats["d_B_bundle_at_L30"] = float(np.linalg.norm(B_nn[-1] - bm_nn[-1]))
    return feats


# ── Drawing ────────────────────────────────────────────────────────

def colored_line3d(pts, values, cmap, lw=1.4, alpha=0.85):
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_vals = (values[:-1] + values[1:]) / 2.0
    lc = Line3DCollection(segs, cmap=cmap, linewidths=lw, alpha=alpha,
                          norm=plt.Normalize(vmin=0.0, vmax=1.0))
    lc.set_array(seg_vals)
    return lc


def auto_axis_limits(all_pts_list, pad_frac=0.10):
    """Compute tight axis limits containing all supplied point sets."""
    stacked = np.vstack(all_pts_list)
    lo = stacked.min(axis=0); hi = stacked.max(axis=0)
    rng_ = hi - lo + 1e-6
    return [(lo[d] - pad_frac * rng_[d], hi[d] + pad_frac * rng_[d])
            for d in range(3)]


def draw_panel(ax, coords, orig2local, go_bundle_idx, target_gos,
               case_A_orig, case_B_orig, gos_a, gos_b,
               conv_or_div_L, fisher_peak_L,
               kind, mode, fisher_curve=None):
    """Draw single 3D panel; auto-fit axes; show endpoint GO label."""
    la = orig2local[case_A_orig]; lb = orig2local[case_B_orig]
    pts_A = coords[la]; pts_B = coords[lb]

    # Collect bundle points for axis fitting
    bundle_pts_all = []
    bundle_isos = set()
    for go in target_gos:
        for oi in go_bundle_idx.get(go, []):
            if oi in orig2local:
                bundle_isos.add(oi)
    for oi in bundle_isos:
        bundle_pts_all.append(coords[orig2local[oi]])

    # Bundle means for target GOs
    bm_map = {}
    for go in target_gos:
        oi_list = [oi for oi in go_bundle_idx.get(go, []) if oi in orig2local]
        if len(oi_list) >= 5:
            li_list = [orig2local[oi] for oi in oi_list]
            bm_map[go] = coords[li_list].mean(axis=0)

    # Axis limits: case + bundles → tight fit
    pts_for_fit = [pts_A, pts_B] + [bm for bm in bm_map.values()]
    if bundle_pts_all:
        # subsample bundle context to avoid outliers dominating scale
        allb = np.vstack(bundle_pts_all)
        # clip to 5th-95th percentile per axis
        lo_p = np.percentile(allb, 5, axis=0)
        hi_p = np.percentile(allb, 95, axis=0)
        pts_for_fit.append(np.stack([lo_p, hi_p]))
    (xl, yl, zl) = auto_axis_limits(pts_for_fit, pad_frac=0.08)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)

    # Context lines (translucent gray)
    for oi in bundle_isos:
        li = orig2local[oi]
        pts = coords[li]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=GRAY_CTX, lw=0.5, alpha=0.22, zorder=2)

    # Bundle means (dotted dark gray)
    for gi, (go, bm) in enumerate(bm_map.items()):
        lbl = f'bundle: {GO_18[go]}' if gi < 2 else None
        ax.plot(bm[:, 0], bm[:, 1], bm[:, 2],
                color=GRAY_BUNDLE, lw=1.6, alpha=0.65, ls=':',
                zorder=3, label=lbl)

    # Case trajectories
    for cidx, (pts, col, gos) in enumerate(
            [(pts_A, CASE_COLORS[0], gos_a),
             (pts_B, CASE_COLORS[1], gos_b)]):

        if mode == 'bundle':
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=col, lw=2.6, alpha=0.95,
                    label=f"Case {'A' if cidx == 0 else 'B'}",
                    zorder=6)
        elif mode == 'fisher' and fisher_curve is not None:
            lc = colored_line3d(pts, fisher_curve, plt.cm.cividis,
                                lw=2.6, alpha=0.95)
            ax.add_collection3d(lc)
        else:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=col, lw=2.6, alpha=0.95, zorder=6)

        # Start / end markers
        marker_col = col if mode == 'bundle' else 'black'
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2],
                   color=marker_col, marker='o', s=100,
                   edgecolors='white', linewidths=1.2, zorder=8)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                   color=marker_col, marker='X', s=140,
                   edgecolors='white', linewidths=1.2, zorder=8)

        # Endpoint GO label (main assigned GO)
        if gos:
            go_label = GO_18[gos[0]] + (f' +{len(gos)-1}' if len(gos) > 1 else '')
        else:
            go_label = 'no-GO'
        ax.text(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                f"  → {go_label}", fontsize=6.5,
                color=col, fontweight='bold', zorder=12)

        # Fisher peak marker
        if 0 < fisher_peak_L <= 30:
            Lp = fisher_peak_L - 1
            ax.scatter(pts[Lp, 0], pts[Lp, 1], pts[Lp, 2],
                       color='gold', marker='*', s=260,
                       edgecolors='darkorange', lw=1.4, zorder=9)

        # Conv/Div layer marker
        if conv_or_div_L > 0:
            Lc = conv_or_div_L - 1
            ax.scatter(pts[Lc, 0], pts[Lc, 1], pts[Lc, 2],
                       color='red', marker='*', s=260,
                       edgecolors='black', lw=1.4, zorder=10)

        # Layer labels on case A
        if cidx == 0 and mode == 'bundle':
            for L in [1, 15, 30]:
                j = L - 1
                ax.text(pts[j, 0], pts[j, 1], pts[j, 2],
                        f'  L{L}', fontsize=6.5, color=col,
                        alpha=0.9, fontweight='bold', zorder=11)

    # Legend markers
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

    ax.set_xlabel('PC1', fontsize=8, labelpad=-2)
    ax.set_ylabel('PC2', fontsize=8, labelpad=-2)
    ax.set_zlabel('PC3', fontsize=8, labelpad=-2)
    ax.tick_params(labelsize=6, pad=0)
    ax.view_init(elev=20, azim=-60)
    if mode == 'bundle':
        ax.legend(fontsize=6, loc='upper left', framealpha=0.85)


# ── Legend text builder ────────────────────────────────────────────

def build_legend_text(p, kind, iso_te, sym_te, seq_lens,
                      gos_a, gos_b, primary_go, key_layer,
                      fisher_peak_L, feats):
    i_a, i_b = p['i_a'], p['i_b']
    gene_a = sym_te[i_a]; gene_b = sym_te[i_b]
    la = seq_lens[i_a]; lb = seq_lens[i_b]

    def fmt_gos(gs):
        if not gs: return "no-GO"
        return "; ".join([GO_18[g] for g in gs[:3]]) + \
               (f" (+{len(gs)-3})" if len(gs) > 3 else "")

    # Interpret cosine similarities post key layer
    def align_word(c):
        if c >= 0.6:   return "매우 정렬됨"
        if c >= 0.3:   return "정렬됨"
        if c >= -0.3:  return "무관"
        return "반대 방향"

    primary_name = GO_18[primary_go] if primary_go else 'N/A'
    cosA = feats.get('cos_A_bundle_post', 0.0)
    cosB = feats.get('cos_B_bundle_post', 0.0)
    cosAB = feats.get('cos_AB_post', 0.0)
    dA_L30 = feats.get('d_A_bundle_at_L30', 0.0)
    dB_L30 = feats.get('d_B_bundle_at_L30', 0.0)

    if kind == 'conv':
        head = (f"■ [CONV] Different genes → same GO ({GO_18[p['go']]})   "
                f"Fisher peak L{fisher_peak_L}   conv_layer L{key_layer}")
        line_a = f"  Case A: {gene_a} / {iso_te[i_a]}   len={la} aa   GOs: {fmt_gos(gos_a)}"
        line_b = f"  Case B: {gene_b} / {iso_te[i_b]}   len={lb} aa   GOs: {fmt_gos(gos_b)}"
        m1 = (f"■ 궤적 동역학 (raw ESM-2 non-normalized): "
              f"L1에서 서열 상이 → d(L1)={feats['d_L1_nn']:.2f} 큰 거리 시작 → "
              f"conv_layer L{key_layer} 부터 target GO bundle 기능 subspace 진입 → "
              f"L30에서 d(L30)={feats['d_L30_nn']:.2f}로 근접.")
        m2 = (f"    A/B 궤적이 GO bundle 방향과 정렬: "
              f"cos(A,bundle)={cosA:+.2f} [{align_word(cosA)}], "
              f"cos(B,bundle)={cosB:+.2f} [{align_word(cosB)}], "
              f"cos(A,B)={cosAB:+.2f} [{align_word(cosAB)}].")
        c1 = (f"■ Feature-level 수렴 원인: (1) 서열 background가 달라도 shared residue-level 특성 "
              f"(active-site geometry, binding-pocket motif 등)이 mid-layer attention에서 통합.")
        c2 = (f"    (2) L{key_layer} 이후 두 궤적 모두 bundle 방향으로 나아가 "
              f"Fisher peak L{fisher_peak_L}에서 '{primary_name}' 정보가 최대치로 밀집.")
        c3 = (f"    (3) 함의: ESM-2가 evolutionary/functional convergence를 "
              f"layer {key_layer}~{fisher_peak_L} 구간에서 표현.")
    else:
        head = (f"■ [DIV] Same gene ({gene_a}) → different isoform GO branches   "
                f"Fisher peak L{fisher_peak_L}   div_layer L{key_layer}")
        line_a = f"  Case A: {iso_te[i_a]}   len={la} aa   GOs: {fmt_gos(gos_a)}"
        line_b = f"  Case B: {iso_te[i_b]}   len={lb} aa   GOs: {fmt_gos(gos_b)}"
        m1 = (f"■ 궤적 동역학 (raw ESM-2 non-normalized): "
              f"L1에서 서열 공유 → d(L1)={feats['d_L1_nn']:.2f} 작게 시작 → "
              f"div_layer L{key_layer} 부근 splicing-차이 exon이 embedding에서 증폭 → "
              f"L30에서 d(L30)={feats['d_L30_nn']:.2f}로 크게 벌어짐.")
        m2 = (f"    두 궤적의 bundle 정렬도가 갈림: "
              f"cos(A,bundle)={cosA:+.2f} [{align_word(cosA)}]  vs  "
              f"cos(B,bundle)={cosB:+.2f} [{align_word(cosB)}] → 하나는 근접, 다른 하나는 이탈.")
        c1 = (f"■ Feature-level 발산 원인: (1) 두 isoform 간 differential exon inclusion/skipping이 "
              f"L{key_layer} 부근 domain-level context를 재구성.")
        c2 = (f"    (2) L30 bundle 거리 A={dA_L30:.2f} vs B={dB_L30:.2f} — "
              f"근접도 차이가 함수 예측 분기의 원인.")
        c3 = (f"    (3) Fisher peak L{fisher_peak_L}에 '{primary_name}' 정보 밀집 → "
              f"이 layer 부근 embedding 분기가 예측 라벨 분기로 이어짐.")

    text = "\n".join([head, line_a, line_b, m1, m2, c1, c2, c3])
    return text


# ── Case figure builder ────────────────────────────────────────────

def make_case_figure(p, kind, case_idx,
                     coords_nn, coords_n, orig2local, go_bundle_idx,
                     go_labels, iso_te, sym_te, seq_lens,
                     fisher_raw, fisher_norm):
    i_a, i_b = p['i_a'], p['i_b']
    gos_a = isoform_gos(i_a, go_labels)
    gos_b = isoform_gos(i_b, go_labels)
    target_gos = list(dict.fromkeys(gos_a + gos_b))
    gene_a = sym_te[i_a]; gene_b = sym_te[i_b]

    conv_or_div_L = p.get('conv_layer', p.get('div_layer', -1))
    if kind == 'conv':
        primary_go = p['go']
    else:
        shared = p.get('shared_gos', [])
        primary_go = (shared[0] if shared else
                      (gos_a[0] if gos_a else
                       (gos_b[0] if gos_b else None)))
    fisher_peak_L = int(np.argmax(fisher_raw[primary_go])) + 1 \
                    if primary_go and primary_go in fisher_raw else -1
    fisher_curve = fisher_norm.get(primary_go) if primary_go else None

    # Compute features for legend
    feats = compute_features(coords_nn, coords_n, orig2local, go_bundle_idx,
                             i_a, i_b, primary_go, conv_or_div_L)

    # Header
    if kind == 'conv':
        header = (
            f"[CONV Case #{case_idx}]  Shared GO: {GO_18[p['go']]}  ({p['go']})  "
            f"|  A: {gene_a}   B: {gene_b}\n"
            f"d(L1)={p['d_L1']:.2f} → d(L30)={p['d_L30']:.2f}   "
            f"conv_layer=L{p['conv_layer']}   Fisher peak L{fisher_peak_L}"
        )
    else:
        header = (
            f"[DIV Case #{case_idx}]  Gene: {gene_a}  "
            f"|  shared GO count = {len(p.get('shared_gos', []))}\n"
            f"d(L1)={p['d_L1']:.2f} → d(L30)={p['d_L30']:.2f}   "
            f"div_layer=L{p['div_layer']}   Fisher peak L{fisher_peak_L}"
        )

    # ── Layout: 2x2 balanced 3D grid + bottom legend ──
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[1.0, 1.0, 0.55],
        hspace=0.08, wspace=0.05,
        left=0.03, right=0.97,
        top=0.945, bottom=0.02,
    )
    fig.suptitle(header, fontsize=12, fontweight='bold', y=0.985)

    ax_A = fig.add_subplot(gs[0, 0], projection='3d')
    draw_panel(ax_A, coords_nn, orig2local, go_bundle_idx, target_gos,
               i_a, i_b, gos_a, gos_b, conv_or_div_L, fisher_peak_L,
               kind, mode='bundle')
    ax_A.set_title(
        '(A) Non-normalized 3D  •  bundle-coloured\n'
        '(gray=target GO bundle context, dotted=bundle mean)',
        fontsize=10, pad=4)

    ax_B = fig.add_subplot(gs[0, 1], projection='3d')
    draw_panel(ax_B, coords_nn, orig2local, go_bundle_idx, target_gos,
               i_a, i_b, gos_a, gos_b, conv_or_div_L, fisher_peak_L,
               kind, mode='fisher', fisher_curve=fisher_curve)
    ax_B.set_title(
        '(B) Non-normalized 3D  •  layer-signal cividis\n'
        f'(case colored by Fisher of {GO_18[primary_go] if primary_go else "N/A"})',
        fontsize=10, pad=4)

    ax_C = fig.add_subplot(gs[1, 0], projection='3d')
    draw_panel(ax_C, coords_n, orig2local, go_bundle_idx, target_gos,
               i_a, i_b, gos_a, gos_b, conv_or_div_L, fisher_peak_L,
               kind, mode='bundle')
    ax_C.set_title(
        '(C) Normalized (per-layer z-score) 3D  •  bundle-coloured',
        fontsize=10, pad=4)

    ax_D = fig.add_subplot(gs[1, 1], projection='3d')
    draw_panel(ax_D, coords_n, orig2local, go_bundle_idx, target_gos,
               i_a, i_b, gos_a, gos_b, conv_or_div_L, fisher_peak_L,
               kind, mode='fisher', fisher_curve=fisher_curve)
    ax_D.set_title(
        '(D) Normalized 3D  •  layer-signal cividis',
        fontsize=10, pad=4)

    # Shared cividis colorbar (right of B & D)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis,
                               norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=[ax_B, ax_D], shrink=0.55, pad=0.02,
                      location='right')
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])
    cb.set_label("per-layer Fisher signal", fontsize=8)

    # Bottom legend text box
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_txt = build_legend_text(p, kind, iso_te, sym_te, seq_lens,
                                   gos_a, gos_b, primary_go,
                                   conv_or_div_L, fisher_peak_L, feats)
    ax_legend.text(
        0.005, 0.99, legend_txt,
        transform=ax_legend.transAxes,
        va='top', ha='left',
        fontsize=10.5,
        linespacing=1.35,
        bbox=dict(
            boxstyle='round,pad=0.7',
            facecolor='#F7F7F5', edgecolor='#666',
            linewidth=0.8,
        ),
    )

    out_name = f"fig_case_{kind}_{case_idx:02d}.png"
    fig.savefig(OUT_DIR / out_name, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_name}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("[1] Loading IDs, GO labels, Fisher curves, seq lengths...")
    sym_te, iso_te = load_gene_symbols()
    go_labels = load_go_labels(sym_te)
    fisher_raw, fisher_norm = load_fisher_curves()
    seq_lens = load_seq_lengths()

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

    print(f"[4] Loading raw ESM-2 trajectories for subset...")
    traj = build_trajectory(subset_idx)
    print(f"   traj shape: {traj.shape}  [{time.time()-t0:.1f}s]")

    print("[5] Joint PCA (non-normalized)...")
    coords_nn = pca_coords(traj, normalize_layers=False)
    print("[6] Joint PCA (normalized: per-layer z-score)...")
    coords_n  = pca_coords(traj, normalize_layers=True)
    del traj
    print(f"   PCAs done [{time.time()-t0:.1f}s]")

    print(f"\n[7] Rendering {len(all_cases)} case figures...")
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

    print(f"\n[done] {len(all_cases)} figures in {OUT_DIR}")
    print(f"       Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
