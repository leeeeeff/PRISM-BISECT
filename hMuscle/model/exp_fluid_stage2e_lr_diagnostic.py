"""
exp_fluid_stage2e_lr_diagnostic.py
====================================
Diagnostic: does curve_vec_norm contain classification-relevant information
beyond L30 mean-pool, specifically for mid-type GOs and Type 3 isoforms?

Motivation
----------
- Feature attribution (exp_feature_attribution.py) showed R²=0.1856 for
  explicit features; 81.4% of functional divergence lives in ESM-2 internal
  encodings (SLiM, PTM, 2° structure, evolutionary patterns).
- Type 3 isoforms (same domain count, different motif/arrangement): PRISM
  ratio ≈ 1.000 — currently fails.
- Stage 2 typed-flow: 10/34 GOs have Fisher discriminant peak at L11-L20
  (mid-type). These are exactly the GOs where PTM/SLiM-level information
  concentrates in ESM-2.
- Hypothesis: curve_vec_norm (per-layer z-score normalized, joint PCA)
  captures mid-layer signal → improves AUPRC on mid-type GOs → specifically
  benefits Type 3 (motif-dependent) isoforms.

Design
------
Feature representations (trained with LogisticRegression, 5-fold CV):
  x_L30    : L30 mean-pool (640-dim)
  x_curve  : curve_vec_norm PCA (K_PCA×N_LAYERS dims, K_PCA=8 → 240)
  x_concat : L30 + curve_vec_norm (880-dim)

Stratification:
  A. All mid-type GOs (10) vs non-mid GOs (24)
  B. Within each GO: Type 3 isoforms (same-domain-count within gene)
     vs Type 1/2 (domain-count-varying)

Output: reports/fluid_stage2/lr_diagnostic_STAMP.json + .txt
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
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
K_PCA    = 8        # PCA axes per layer → curve vector dim = K_PCA × N_LAYERS = 240
SEED     = 42
N_FOLDS  = 5
MAX_ITER = 200

# 10 mid-type GOs (peak Fisher discriminant at L11-L20)
MID_TYPE_GOS = {
    "GO:0035556", "GO:0006281", "GO:0000165", "GO:0006417",
    "GO:0007204", "GO:0006816", "GO:0006470", "GO:0006414",
    "GO:0006906", "GO:0006418",
}

# Full 34-GO catalog from Stage 2
GO_TERMS = {
    "GO:0006974": "DNA damage response",    "GO:0035556": "Intracellular signal",
    "GO:0006508": "Proteolysis",            "GO:0043161": "Proteasome-UPS",
    "GO:0006281": "DNA repair",             "GO:0000226": "MT cytoskeleton org",
    "GO:0005975": "Carbohydrate metabolism","GO:0055074": "Ca2+ homeostasis",
    "GO:0000165": "MAPK cascade",           "GO:0000398": "mRNA splicing",
    "GO:0006417": "Reg. of translation",    "GO:0007015": "Actin filament org",
    "GO:0007204": "Ca2+ signaling",         "GO:0007059": "Chromosome segregation",
    "GO:0007265": "Ras signaling",          "GO:0007018": "MT-based movement",
    "GO:0006816": "Ca2+ transport",         "GO:0006888": "ER-Golgi transport",
    "GO:0006402": "mRNA catabolism",        "GO:0006486": "Protein glycosylation",
    "GO:0006914": "Autophagy",              "GO:0006470": "Dephosphorylation",
    "GO:0006836": "Neurotransmitter transp","GO:0006414": "Translational elongation",
    "GO:0030048": "Actin-based movement",   "GO:0032465": "Cytokinesis",
    "GO:0006906": "Vesicle fusion",         "GO:0006418": "tRNA aminoacylation",
    "GO:0006754": "ATP biosynthesis",       "GO:0006635": "FA beta-oxidation",
    "GO:0006120": "Complex I NADH ox",      "GO:0045214": "Sarcomere organization",
    "GO:0006096": "Glycolysis",             "GO:0006099": "TCA cycle",
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
    """gene_symbol → set of GO IDs (from annotation file)."""
    gene2go = defaultdict(set)
    with open(ANNOT_FILE) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1:
                gene2go[parts[0]] = set(parts[1:])
    return gene2go


def build_trajectory_subset(idx: np.ndarray) -> np.ndarray:
    """Stack layers → (N, 30, 640), float32."""
    N = len(idx)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        p = DATA / f"esm2_layer_{L:02d}_t30_150M.npy"
        arr = np.load(p, mmap_mode="r")
        traj[:, L - 1, :] = arr[idx].astype(np.float32)
        del arr
    return traj


def build_curve_vec_norm(traj: np.ndarray, k_pca: int = K_PCA) -> np.ndarray:
    """Per-layer z-score → joint PCA → (N, k_pca × 30)."""
    N = traj.shape[0]
    traj_norm = np.empty_like(traj)
    for L in range(N_LAYERS):
        mu = traj[:, L, :].mean(0)
        sd = traj[:, L, :].std(0) + 1e-6
        traj_norm[:, L, :] = (traj[:, L, :] - mu) / sd
    flat = traj_norm.reshape(N * N_LAYERS, EMB_DIM)
    pca = PCA(n_components=k_pca, random_state=SEED)
    reduced = pca.fit_transform(flat)          # (N*30, k_pca)
    curve = reduced.reshape(N, N_LAYERS * k_pca)  # (N, 240)
    return curve.astype(np.float32)


def cv_auprc(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS) -> float:
    """5-fold stratified CV AUPRC."""
    if y.sum() < n_folds or (1 - y).sum() < n_folds:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        lr = LogisticRegression(max_iter=MAX_ITER, C=1.0,
                                solver="lbfgs", random_state=SEED)
        lr.fit(X_tr, y[tr])
        prob = lr.predict_proba(X_te)[:, 1]
        scores.append(average_precision_score(y[te], prob))
    return float(np.mean(scores))


def main():
    t0 = time.time()

    # ── load isoform index ──────────────────────────────────────────
    print("[1] Loading isoform/gene lists...")
    iso_ids  = load_ids(str(MODEL / "my_isoform_list_fixed.npy"))
    gene_ids = load_ids(str(MODEL / "my_gene_list_fixed.npy"))
    gene_ids_base = [g.split(".")[0] for g in gene_ids]
    n_iso = len(iso_ids)
    print(f"   n_iso={n_iso}")

    e2s = load_gene_symbol_map()
    # Map gene ENSG → symbol
    symbols = [e2s.get(g, g) for g in gene_ids_base]

    gene2go = load_go_positives()

    # ── build pilot subset (pos_union ∪ matched_neg) ────────────────
    print("[2] Building pilot subset...")
    pos_mask = np.zeros(n_iso, dtype=bool)
    y_dict   = {}
    for go in GO_TERMS:
        go_pos_genes = {sym for sym, gos in gene2go.items() if go in gos}
        y = np.array([sym in go_pos_genes for sym in symbols], dtype=np.int8)
        y_dict[go] = y
        pos_mask |= y.astype(bool)

    pos_idx = np.where(pos_mask)[0]
    rng = np.random.default_rng(SEED)
    neg_idx_all = np.where(~pos_mask)[0]
    neg_idx = rng.choice(neg_idx_all, size=min(len(pos_idx) * 2, len(neg_idx_all)), replace=False)
    pilot_idx = np.sort(np.concatenate([pos_idx, neg_idx]))
    N = len(pilot_idx)
    print(f"   pilot N={N} (pos_union={len(pos_idx)}, neg={len(neg_idx)})")

    # ── domain matrix → Type 3 flags ────────────────────────────────
    print("[3] Computing Type-3 (same-domain) flags...")
    dm_path = FEAT_DIR / "domain_matrix_proper_test.npy"
    if dm_path.exists():
        dm = np.load(dm_path, mmap_mode="r")       # (n_iso, 512)
        domain_counts = np.array(dm.sum(1)).ravel()  # (n_iso,)
        del dm
        # Per-gene domain count range
        gene2dc = defaultdict(list)
        for i, g in enumerate(gene_ids_base):
            gene2dc[g].append(domain_counts[i])
        gene_dc_range = {g: (max(v) - min(v)) for g, v in gene2dc.items()}
        # Type 3: same domain count across all gene isoforms
        is_type3 = np.array(
            [gene_dc_range.get(g, 0) == 0 for g in gene_ids_base], dtype=bool
        )
        print(f"   Type-3 isoforms: {is_type3.sum()} / {n_iso} "
              f"({is_type3.mean():.1%})")
    else:
        print(f"   [warn] domain matrix not found at {dm_path}; skipping Type-3 split")
        is_type3 = np.zeros(n_iso, dtype=bool)

    # ── build trajectory & representations ──────────────────────────
    print(f"[4] Building trajectory for pilot ({N} isoforms × 30 layers)...")
    traj = build_trajectory_subset(pilot_idx)
    print(f"   traj shape: {traj.shape}  [{time.time()-t0:.1f}s]")

    print("[5] Building curve_vec_norm (per-layer z-score + joint PCA)...")
    x_curve = build_curve_vec_norm(traj)
    x_L30   = traj[:, -1, :]                    # last layer = L30
    x_concat = np.concatenate([x_L30, x_curve], axis=1)
    print(f"   x_L30={x_L30.shape} x_curve={x_curve.shape} x_concat={x_concat.shape}")
    print(f"   [{time.time()-t0:.1f}s]")

    # Pilot-level masks
    type3_pilot = is_type3[pilot_idx]
    type3_frac  = type3_pilot.mean()

    # ── LR evaluation per GO ─────────────────────────────────────────
    print("[6] Running LR evaluation...")
    results = []

    for go, go_name in GO_TERMS.items():
        y_full = y_dict[go]
        y      = y_full[pilot_idx]
        go_type = "mid" if go in MID_TYPE_GOS else "non-mid"
        n_pos  = int(y.sum())

        if n_pos < 5:
            continue

        row = {"go": go, "go_name": go_name, "go_type": go_type, "n_pos": n_pos}

        # ── all pilot ─────────────────────
        row["auprc_L30"]    = cv_auprc(x_L30,    y)
        row["auprc_curve"]  = cv_auprc(x_curve,  y)
        row["auprc_concat"] = cv_auprc(x_concat, y)
        row["gain_curve"]   = row["auprc_curve"]  - row["auprc_L30"]
        row["gain_concat"]  = row["auprc_concat"] - row["auprc_L30"]

        # ── Type 3 isoforms only ──────────
        y_t3 = y[type3_pilot]
        if y_t3.sum() >= 5:
            row["auprc_L30_type3"]    = cv_auprc(x_L30[type3_pilot],    y_t3)
            row["auprc_curve_type3"]  = cv_auprc(x_curve[type3_pilot],  y_t3)
            row["auprc_concat_type3"] = cv_auprc(x_concat[type3_pilot], y_t3)
            row["gain_concat_type3"]  = row["auprc_concat_type3"] - row["auprc_L30_type3"]
        else:
            for k in ["auprc_L30_type3","auprc_curve_type3","auprc_concat_type3","gain_concat_type3"]:
                row[k] = float("nan")

        results.append(row)
        print(f"   {go_type:7s} {go} ({go_name[:22]:22s}) "
              f"n_pos={n_pos:4d} | "
              f"L30={row['auprc_L30']:.3f} "
              f"curve={row['auprc_curve']:.3f} "
              f"concat={row['auprc_concat']:.3f} "
              f"gain={row['gain_concat']:+.3f} | "
              f"T3_concat={row['auprc_concat_type3']:.3f} gain_T3={row['gain_concat_type3']:+.3f}"
              if not np.isnan(row.get("auprc_concat_type3", float("nan")))
              else
              f"   {go_type:7s} {go} ({go_name[:22]:22s}) "
              f"n_pos={n_pos:4d} | "
              f"L30={row['auprc_L30']:.3f} "
              f"curve={row['auprc_curve']:.3f} "
              f"concat={row['auprc_concat']:.3f} "
              f"gain={row['gain_concat']:+.3f} | T3=n/a"
        )

    # ── Summary by GO type ───────────────────────────────────────────
    df = pd.DataFrame(results)
    print(f"\n{'='*72}")
    print(f"SUMMARY  (pilot N={N}, Type-3 frac={type3_frac:.1%})")
    print(f"{'='*72}")
    for gtype in ["mid", "non-mid"]:
        sub = df[df["go_type"] == gtype]
        if sub.empty:
            continue
        print(f"\n  GO type = {gtype} (n={len(sub)} GOs)")
        for col, label in [
            ("auprc_L30",       "L30 AUPRC       "),
            ("auprc_curve",     "curve AUPRC     "),
            ("auprc_concat",    "concat AUPRC    "),
            ("gain_concat",     "Δ concat-L30    "),
            ("gain_concat_type3", "Δ concat-L30 T3"),
        ]:
            vals = sub[col].dropna()
            print(f"    {label}: mean={vals.mean():+.4f}  "
                  f"median={vals.median():+.4f}  n={len(vals)}")

    # ── Write outputs ─────────────────────────────────────────────────
    json_path = OUT_DIR / f"lr_diagnostic_{STAMP}.json"
    txt_path  = OUT_DIR / f"lr_diagnostic_{STAMP}.txt"

    with open(json_path, "w") as f:
        json.dump({"results": results, "pilot_N": N,
                   "type3_frac": float(type3_frac), "stamp": STAMP}, f, indent=2)

    # human-readable full table
    with open(txt_path, "w") as f:
        f.write(df.to_string(index=False))

    print(f"\n[write] {json_path}")
    print(f"[write] {txt_path}")
    print(f"[done]  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
