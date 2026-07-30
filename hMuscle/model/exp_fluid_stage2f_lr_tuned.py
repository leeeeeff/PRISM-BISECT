"""
exp_fluid_stage2f_lr_tuned.py
==============================
Improved LR diagnostic over Stage 2e:
  - max_iter=1000, C-tuning {0.1, 1.0, 10.0} via inner CV
  - AUPRC evaluation (gene-level, as before)
  - NEW: within-gene isoform separation metrics (isoform-level)

Within-gene metrics (key addition)
------------------------------------
AUPRC is a GENE-level metric: it measures whether GO-positive genes rank
above GO-negative genes, regardless of which isoform. The critical new
question is whether curve_vec_norm better SEPARATES isoforms WITHIN the
same gene — especially for Type 3 (same-domain-count, motif-dependent) genes.

Metric A — Within-gene pairwise L2 distance
  For each multi-isoform gene G, compute mean pairwise L2 distance between
  isoform embeddings. Compare L30 vs curve vs concat.
  Hypothesis: curve_vec_norm amplifies within-gene distance for Type 3 genes
  (captures motif-level variation that L30 averages out).

Metric B — Within-gene prediction spread (LR score range)
  For each gene G with ≥2 isoforms, compute max−min of predicted probability
  from the best-C LR model. Compare L30 vs curve vs concat, stratified by
  Type (3 = same domain count, 1/2 = domain count varies).

Both metrics are averaged over genes, not isoforms, to prevent bias from
gene size.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL = ROOT / "model"
FEAT_DIR = ROOT / "results_isoform" / "features"
OUT_DIR = ROOT.parent / "reports" / "fluid_stage2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d_%H%M")

ANNOT_FILE = DATA / "raw_data/data/annotations/human_annotations_unified_bp.txt"
E2S_FILE   = DATA / "raw_data/data/id_lists/ensembl_to_symbol.txt"

N_LAYERS = 30
EMB_DIM  = 640
K_PCA    = 8        # → curve dim = 8×30 = 240
SEED     = 42
N_FOLDS  = 5
C_GRID   = [0.1, 1.0, 10.0]

MID_TYPE_GOS = {
    "GO:0035556","GO:0006281","GO:0000165","GO:0006417",
    "GO:0007204","GO:0006816","GO:0006470","GO:0006414",
    "GO:0006906","GO:0006418",
}

GO_TERMS = {
    "GO:0006974":"DNA damage response",    "GO:0035556":"Intracellular signal",
    "GO:0006508":"Proteolysis",            "GO:0043161":"Proteasome-UPS",
    "GO:0006281":"DNA repair",             "GO:0000226":"MT cytoskeleton org",
    "GO:0005975":"Carbohydrate metabolism","GO:0055074":"Ca2+ homeostasis",
    "GO:0000165":"MAPK cascade",           "GO:0000398":"mRNA splicing",
    "GO:0006417":"Reg. of translation",    "GO:0007015":"Actin filament org",
    "GO:0007204":"Ca2+ signaling",         "GO:0007059":"Chromosome segregation",
    "GO:0007265":"Ras signaling",          "GO:0007018":"MT-based movement",
    "GO:0006816":"Ca2+ transport",         "GO:0006888":"ER-Golgi transport",
    "GO:0006402":"mRNA catabolism",        "GO:0006486":"Protein glycosylation",
    "GO:0006914":"Autophagy",              "GO:0006470":"Dephosphorylation",
    "GO:0006836":"Neurotransmitter transp","GO:0006414":"Translational elongation",
    "GO:0030048":"Actin-based movement",   "GO:0032465":"Cytokinesis",
    "GO:0006906":"Vesicle fusion",         "GO:0006418":"tRNA aminoacylation",
    "GO:0006754":"ATP biosynthesis",       "GO:0006635":"FA beta-oxidation",
    "GO:0006120":"Complex I NADH ox",      "GO:0045214":"Sarcomere organization",
    "GO:0006096":"Glycolysis",             "GO:0006099":"TCA cycle",
}


def load_ids(p):
    a = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def load_gene_symbol_map():
    m = {}
    with open(E2S_FILE) as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                m[p[0]] = p[4]
    return m


def load_go_positives():
    gene2go = defaultdict(set)
    with open(ANNOT_FILE) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1:
                gene2go[parts[0]] = set(parts[1:])
    return gene2go


def build_trajectory(idx: np.ndarray) -> np.ndarray:
    N = len(idx)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        p = DATA / f"esm2_layer_{L:02d}_t30_150M.npy"
        arr = np.load(p, mmap_mode="r")
        traj[:, L - 1, :] = arr[idx].astype(np.float32)
        del arr
    return traj


def build_curve_vec_norm(traj: np.ndarray, pca: PCA | None = None):
    """Per-layer z-score → joint PCA. Returns (curve_vec, fitted_pca)."""
    N = traj.shape[0]
    traj_norm = np.empty_like(traj)
    for L in range(N_LAYERS):
        mu = traj[:, L, :].mean(0)
        sd = traj[:, L, :].std(0) + 1e-6
        traj_norm[:, L, :] = (traj[:, L, :] - mu) / sd
    flat = traj_norm.reshape(N * N_LAYERS, EMB_DIM)
    if pca is None:
        pca = PCA(n_components=K_PCA, random_state=SEED)
        reduced = pca.fit_transform(flat)
    else:
        reduced = pca.transform(flat)
    curve = reduced.reshape(N, N_LAYERS * K_PCA)
    return curve.astype(np.float32), pca


def best_C_auprc(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Inner CV to pick C; return (best_auprc, best_C)."""
    if y.sum() < N_FOLDS or (1 - y).sum() < N_FOLDS:
        return float("nan"), 1.0
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    best_auprc, best_C = -1.0, 1.0
    for C in C_GRID:
        scores = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                    random_state=SEED)
            lr.fit(Xtr, y[tr])
            prob = lr.predict_proba(Xte)[:, 1]
            scores.append(average_precision_score(y[te], prob))
        mean = float(np.mean(scores))
        if mean > best_auprc:
            best_auprc, best_C = mean, C
    return best_auprc, best_C


def train_lr_full(X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    """Train on full data, return predicted probabilities."""
    sc = StandardScaler()
    Xsc = sc.fit_transform(X)
    lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=SEED)
    lr.fit(Xsc, y)
    return lr.predict_proba(Xsc)[:, 1]


# ── Within-gene metrics ────────────────────────────────────────────────────────

def within_gene_distance(feat: np.ndarray, gene_list: np.ndarray) -> dict:
    """
    For each gene with ≥2 isoforms in the PILOT, compute mean pairwise L2
    distance between isoform feature vectors.
    Returns {gene_sym: mean_distance}.
    """
    gene2idxs = defaultdict(list)
    for i, g in enumerate(gene_list):
        gene2idxs[g].append(i)

    result = {}
    for g, idxs in gene2idxs.items():
        if len(idxs) < 2:
            continue
        vecs = feat[idxs]                    # (k, dim)
        # pairwise L2
        dists = []
        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                dists.append(float(np.linalg.norm(vecs[a] - vecs[b])))
        result[g] = float(np.mean(dists))
    return result


def within_gene_pred_spread(probs: np.ndarray, gene_list: np.ndarray) -> dict:
    """max−min predicted probability per gene (≥2 isoforms)."""
    gene2idxs = defaultdict(list)
    for i, g in enumerate(gene_list):
        gene2idxs[g].append(i)
    return {
        g: float(probs[idxs].max() - probs[idxs].min())
        for g, idxs in gene2idxs.items()
        if len(idxs) >= 2
    }


def main():
    t0 = time.time()

    print("[1] Loading data...")
    iso_ids  = load_ids(str(MODEL / "my_isoform_list_fixed.npy"))
    gene_ids = load_ids(str(MODEL / "my_gene_list_fixed.npy"))
    gene_ids_base = [g.split(".")[0] for g in gene_ids]
    n_iso = len(iso_ids)

    e2s = load_gene_symbol_map()
    symbols = [e2s.get(g, g) for g in gene_ids_base]
    gene2go = load_go_positives()

    # ── Pilot subset ───────────────────────────────────────────────
    print("[2] Building pilot subset...")
    pos_mask = np.zeros(n_iso, dtype=bool)
    y_dict = {}
    for go in GO_TERMS:
        go_pos_genes = {sym for sym, gos in gene2go.items() if go in gos}
        y = np.array([sym in go_pos_genes for sym in symbols], dtype=np.int8)
        y_dict[go] = y
        pos_mask |= y.astype(bool)

    pos_idx = np.where(pos_mask)[0]
    rng = np.random.default_rng(SEED)
    neg_idx = rng.choice(np.where(~pos_mask)[0],
                         size=min(len(pos_idx) * 2, (~pos_mask).sum()),
                         replace=False)
    pilot_idx = np.sort(np.concatenate([pos_idx, neg_idx]))
    N = len(pilot_idx)
    print(f"   pilot N={N}")

    # ── Type-3 flags ───────────────────────────────────────────────
    print("[3] Computing Type-3 flags...")
    dm_path = FEAT_DIR / "domain_matrix_proper_test.npy"
    if dm_path.exists():
        dm = np.load(dm_path, mmap_mode="r")
        domain_counts = np.array(dm.sum(1)).ravel()
        del dm
        gene2dc = defaultdict(list)
        for i, g in enumerate(gene_ids_base):
            gene2dc[g].append(domain_counts[i])
        gene_dc_range = {g: (max(v) - min(v)) for g, v in gene2dc.items()}
        is_type3_full = np.array(
            [gene_dc_range.get(g, 0) == 0 for g in gene_ids_base], dtype=bool)
        print(f"   Type-3 isoforms (full): {is_type3_full.sum()} / {n_iso} "
              f"({is_type3_full.mean():.1%})")
    else:
        print("   [warn] domain matrix missing, no Type-3 split")
        is_type3_full = np.zeros(n_iso, dtype=bool)

    is_type3_pilot = is_type3_full[pilot_idx]
    symbols_pilot  = np.array(symbols)[pilot_idx]
    iso_ids_pilot  = np.array(iso_ids)[pilot_idx]

    # Type-3 gene sets for within-gene analysis
    type3_genes = {g for g, dr in gene_dc_range.items() if dr == 0}
    type12_genes = set(gene_dc_range.keys()) - type3_genes

    # ── Build representations ───────────────────────────────────────
    print(f"[4] Building trajectory (N={N} × 30 layers)...")
    traj = build_trajectory(pilot_idx)
    x_L30   = traj[:, -1, :]
    print(f"   [{time.time()-t0:.1f}s] Building curve_vec_norm...")
    x_curve, _ = build_curve_vec_norm(traj)
    x_concat    = np.concatenate([x_L30, x_curve], axis=1)
    print(f"   L30={x_L30.shape} curve={x_curve.shape} concat={x_concat.shape}  "
          f"[{time.time()-t0:.1f}s]")

    # ── AUPRC evaluation ───────────────────────────────────────────
    print("[5] LR AUPRC with C-tuning...")
    results = []
    best_C_per_go = {}

    for go, go_name in GO_TERMS.items():
        y = y_dict[go][pilot_idx]
        go_type = "mid" if go in MID_TYPE_GOS else "non-mid"
        if y.sum() < 5:
            continue

        r = {"go": go, "go_name": go_name, "go_type": go_type, "n_pos": int(y.sum())}

        r["auprc_L30"],    r["C_L30"]    = best_C_auprc(x_L30,    y)
        r["auprc_curve"],  r["C_curve"]  = best_C_auprc(x_curve,  y)
        r["auprc_concat"], r["C_concat"] = best_C_auprc(x_concat,  y)
        r["gain_concat"]  = r["auprc_concat"] - r["auprc_L30"]

        # Type-3 only
        y_t3 = y[is_type3_pilot]
        if y_t3.sum() >= 5:
            r["auprc_L30_T3"],  _ = best_C_auprc(x_L30[is_type3_pilot],   y_t3)
            r["auprc_concat_T3"],_ = best_C_auprc(x_concat[is_type3_pilot],y_t3)
            r["gain_concat_T3"] = r["auprc_concat_T3"] - r["auprc_L30_T3"]
        else:
            r["auprc_L30_T3"] = r["auprc_concat_T3"] = r["gain_concat_T3"] = float("nan")

        best_C_per_go[go] = r["C_concat"]
        results.append(r)
        print(f"   {'[MID]' if go_type=='mid' else '     '} {go} "
              f"L30={r['auprc_L30']:.3f}(C={r['C_L30']}) "
              f"concat={r['auprc_concat']:.3f}(C={r['C_concat']}) "
              f"gain={r['gain_concat']:+.3f} | "
              f"T3_concat={r.get('auprc_concat_T3', float('nan')):.3f} "
              f"gain_T3={r.get('gain_concat_T3', float('nan')):+.3f}")

    # ── Within-gene metrics (using best-C model trained on full pilot) ──
    print(f"\n[6] Within-gene separation metrics  [{time.time()-t0:.1f}s]")

    # Use the first GO's best C for representative model
    rep_go = "GO:0000165"   # MAPK cascade [MID] — representative mid-type
    y_rep = y_dict[rep_go][pilot_idx]
    C_rep = best_C_per_go.get(rep_go, 1.0)

    prob_L30   = train_lr_full(x_L30,    y_rep, C_rep)
    prob_curve = train_lr_full(x_curve,  y_rep, C_rep)
    prob_concat= train_lr_full(x_concat, y_rep, C_rep)

    # A. Within-gene pairwise L2 distance
    wg_dist_L30   = within_gene_distance(x_L30,    symbols_pilot)
    wg_dist_curve = within_gene_distance(x_curve,  symbols_pilot)
    wg_dist_concat= within_gene_distance(x_concat, symbols_pilot)

    def split_dist(d: dict):
        t3  = [v for g, v in d.items() if g in type3_genes]
        t12 = [v for g, v in d.items() if g in type12_genes]
        return np.array(t3), np.array(t12)

    dist_L30_t3, dist_L30_t12     = split_dist(wg_dist_L30)
    dist_curve_t3, dist_curve_t12 = split_dist(wg_dist_curve)
    dist_concat_t3,dist_concat_t12= split_dist(wg_dist_concat)

    # B. Within-gene prediction spread
    spread_L30   = within_gene_pred_spread(prob_L30,    symbols_pilot)
    spread_curve = within_gene_pred_spread(prob_curve,  symbols_pilot)
    spread_concat= within_gene_pred_spread(prob_concat, symbols_pilot)

    def split_spread(d: dict):
        t3  = [v for g, v in d.items() if g in type3_genes]
        t12 = [v for g, v in d.items() if g in type12_genes]
        return np.array(t3), np.array(t12)

    sp_L30_t3, sp_L30_t12       = split_spread(spread_L30)
    sp_curve_t3, sp_curve_t12   = split_spread(spread_curve)
    sp_concat_t3, sp_concat_t12 = split_spread(spread_concat)

    # MWU: Type-3 distance/spread vs Type-1/2
    def mwu_greater(a, b):
        if len(a) < 3 or len(b) < 3:
            return float("nan"), float("nan")
        u, p = mannwhitneyu(a, b, alternative="greater")
        return float(u), float(p)

    print("\n  ─── WITHIN-GENE PAIRWISE L2 DISTANCE ───────────────────────────")
    for name, t3, t12 in [
        ("L30   ", dist_L30_t3,    dist_L30_t12),
        ("curve ", dist_curve_t3,  dist_curve_t12),
        ("concat", dist_concat_t3, dist_concat_t12),
    ]:
        u, p = mwu_greater(t3, t12)
        print(f"    {name}  Type3 mean={np.mean(t3):.3f}  "
              f"Type1/2 mean={np.mean(t12):.3f}  "
              f"ratio={np.mean(t3)/np.mean(t12):.3f}  "
              f"MWU p(T3>T12)={p:.4f}")

    print("\n  ─── WITHIN-GENE PREDICTION SPREAD (MAPK [MID]) ─────────────────")
    for name, t3, t12 in [
        ("L30   ", sp_L30_t3,    sp_L30_t12),
        ("curve ", sp_curve_t3,  sp_curve_t12),
        ("concat", sp_concat_t3, sp_concat_t12),
    ]:
        u, p = mwu_greater(t3, t12)
        print(f"    {name}  Type3 mean={np.mean(t3):.4f}  "
              f"Type1/2 mean={np.mean(t12):.4f}  "
              f"ratio={np.mean(t3)/max(np.mean(t12),1e-9):.3f}  "
              f"MWU p(T3>T12)={p:.4f}")

    # ── Summary by GO type ─────────────────────────────────────────
    df = pd.DataFrame(results)
    print(f"\n{'='*65}")
    print("AUPRC SUMMARY (LR tuned, max_iter=1000)")
    print(f"{'='*65}")
    for gtype in ["mid", "non-mid"]:
        sub = df[df["go_type"] == gtype]
        gc  = sub["gain_concat"].dropna()
        gt3 = sub["gain_concat_T3"].dropna()
        print(f"  {gtype:7s} ({len(sub)} GOs): "
              f"gain_concat mean={gc.mean():+.4f}  "
              f"gain_T3 mean={gt3.mean():+.4f}  "
              f"n_pos_gain={( gc>0).sum()}/{len(gc)}")

    # ── Save ────────────────────────────────────────────────────────
    out = {
        "stamp": STAMP, "pilot_N": N,
        "type3_gene_count": len(type3_genes),
        "type12_gene_count": len(type12_genes),
        "auprc_results": results,
        "within_gene_dist": {
            "L30_T3_mean":    float(np.mean(dist_L30_t3)),
            "L30_T12_mean":   float(np.mean(dist_L30_t12)),
            "curve_T3_mean":  float(np.mean(dist_curve_t3)),
            "curve_T12_mean": float(np.mean(dist_curve_t12)),
            "concat_T3_mean": float(np.mean(dist_concat_t3)),
            "concat_T12_mean":float(np.mean(dist_concat_t12)),
        },
        "within_gene_spread": {
            "L30_T3_mean":    float(np.mean(sp_L30_t3)),
            "curve_T3_mean":  float(np.mean(sp_curve_t3)),
            "concat_T3_mean": float(np.mean(sp_concat_t3)),
            "L30_T12_mean":   float(np.mean(sp_L30_t12)),
            "curve_T12_mean": float(np.mean(sp_curve_t12)),
            "concat_T12_mean":float(np.mean(sp_concat_t12)),
        },
    }
    json_path = OUT_DIR / f"lr_tuned_{STAMP}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[write] {json_path}")
    print(f"[done]  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
