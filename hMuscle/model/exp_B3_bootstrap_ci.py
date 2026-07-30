"""
exp_B3_bootstrap_ci.py
======================
Bootstrap CI (n=1000) for macro AUPRC and T3/T12 spread ratio,
computed from existing score matrices.

Models: v15d, v17f*, v19, v20b w=5, v20b w=7  (18 BP GO).
For each bootstrap: sample GO indices with replacement OR sample isoforms
  with replacement (per-GO AUPRC recomputed).

Two-tier bootstrap for T3/T12:
  - resample isoforms → recompute spread per gene → mean T3, T12 → ratio.
"""
from __future__ import annotations

import os, json, time
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = ROOT / "model"
FEAT_DIR = ROOT / "results_isoform" / "features"
ID_DIR = DATA / "raw_data/data/id_lists"
ANNOT_DIR = DATA / "raw_data/data/annotations"
REP_DIR = ROOT.parent / "reports"
OUT_DIR = REP_DIR / "exp_B3_bootstrap_ci"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GO_18 = [
    "GO:0007204", "GO:0045214", "GO:0006941", "GO:0006914",
    "GO:0043161", "GO:0007519", "GO:0042692", "GO:0055074",
    "GO:0007005", "GO:0007517", "GO:0032006", "GO:0030048",
    "GO:0006096", "GO:0007268", "GO:0007018", "GO:0031175",
    "GO:0030182", "GO:0000226",
]
N_BOOT = 1000
SEED = 42


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']:
        s = s.replace(c, '')
    return s


def load_labels(sym_te):
    Y = np.zeros((len(sym_te), len(GO_18)), dtype=np.int8)
    for gi, go in enumerate(GO_18):
        pos = set()
        with open(ANNOT_DIR / "human_annotations_unified_bp.txt") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > 1 and go in parts[1:]:
                    pos.add(parts[0])
        Y[:, gi] = np.array([1 if s in pos else 0 for s in sym_te],
                            dtype=np.int8)
    return Y


def type3_gene_set(te_gene):
    dm = np.load(FEAT_DIR / "domain_matrix_proper_test.npy", mmap_mode="r")
    dc = np.array(dm.sum(1)).ravel(); del dm
    base = [clean(g).split('.')[0] for g in te_gene]
    g2dc = defaultdict(list)
    for i, g in enumerate(base):
        g2dc[g].append(dc[i])
    rng = {g: max(v) - min(v) for g, v in g2dc.items()}
    return {g for g, r in rng.items() if r == 0}


def macro_auprc(S, Y, iso_idx):
    """S: (N, 18), Y: (N, 18)."""
    aps = []
    for gi in range(len(GO_18)):
        y = Y[iso_idx, gi]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        s = S[iso_idx, gi]
        aps.append(average_precision_score(y, s))
    return float(np.mean(aps)) if aps else 0.0


def t3_t12_ratio(S, Y, iso_idx, sym_te, type3_genes):
    """
    Within-gene predicted-value spread ratio.
    Compute per GO, then average.
    """
    sym_sub = [sym_te[i] for i in iso_idx]
    ratios = []
    for gi in range(len(GO_18)):
        preds = S[iso_idx, gi]
        g2i = defaultdict(list)
        for k, g in enumerate(sym_sub):
            g2i[g].append(k)
        spreads = {g: float(preds[idxs].max() - preds[idxs].min())
                   for g, idxs in g2i.items() if len(idxs) >= 2}
        if not spreads:
            continue
        t3 = [v for g, v in spreads.items() if g in type3_genes]
        t12 = [v for g, v in spreads.items() if g not in type3_genes]
        if not t3 or not t12:
            continue
        m_t3, m_t12 = np.mean(t3), np.mean(t12)
        if m_t12 < 1e-9:
            continue
        ratios.append(m_t3 / m_t12)
    return float(np.mean(ratios)) if ratios else 0.0


def bootstrap_ci(S, Y, sym_te, type3_genes, n_boot=N_BOOT, seed=SEED):
    N = S.shape[0]
    rng = np.random.default_rng(seed)
    ap_list, t_list = [], []
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        ap_list.append(macro_auprc(S, Y, idx))
        t_list.append(t3_t12_ratio(S, Y, idx, sym_te, type3_genes))
    ap = np.array(ap_list)
    tr = np.array(t_list)
    return {
        "macro_auprc": {
            "point": float(np.mean(ap)),
            "ci_lo": float(np.percentile(ap, 2.5)),
            "ci_hi": float(np.percentile(ap, 97.5)),
        },
        "t3_t12_ratio": {
            "point": float(np.mean(tr)),
            "ci_lo": float(np.percentile(tr, 2.5)),
            "ci_hi": float(np.percentile(tr, 97.5)),
        },
    }


def main():
    t0 = time.time()
    print("[1] IDs & labels…")
    te_gene = np.load(MODEL_DIR / "my_gene_list_fixed.npy", allow_pickle=True)
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                ENSG2SYM[p[0]] = p[4]
    sym_te = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
              for g in te_gene]
    Y = load_labels(sym_te)
    type3_genes = type3_gene_set(te_gene)

    print(f"   N_te={len(sym_te)}   Type-3 genes: {len(type3_genes)}")

    paths = [
        ("v15d", REP_DIR / "v15_bp_clean/score_matrix_18go_20260519_1914.npy"),
        ("v17f-BP", REP_DIR / "v17f_bp/v17f_bp_score_matrix.npy"),
        ("v19", REP_DIR / "v19_curve/v19_score_matrix.npy"),
        ("v20b_w5", REP_DIR / "v20b/w5_score_matrix.npy"),
        ("v20b_w7", REP_DIR / "v20b/w7_score_matrix.npy"),
    ]
    results = {}
    for name, p in paths:
        if not p.exists():
            print(f"  {name}: MISSING")
            continue
        S = np.load(p)
        if S.shape[1] != len(GO_18):
            print(f"  {name}: SHAPE MISMATCH ({S.shape})")
            continue
        print(f"\n  Bootstrapping {name} …")
        t1 = time.time()
        ci = bootstrap_ci(S, Y, sym_te, type3_genes, N_BOOT, SEED)
        ci["elapsed"] = round(time.time() - t1, 1)
        results[name] = ci
        ap = ci["macro_auprc"]
        tr = ci["t3_t12_ratio"]
        print(f"    macro AUPRC = {ap['point']:.4f}  [{ap['ci_lo']:.4f}, {ap['ci_hi']:.4f}]")
        print(f"    T3/T12      = {tr['point']:.4f}  [{tr['ci_lo']:.4f}, {tr['ci_hi']:.4f}]")

    out = {"n_boot": N_BOOT, "seed": SEED, "models": results,
           "elapsed_sec": time.time() - t0}
    with open(OUT_DIR / "bootstrap_ci.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {OUT_DIR/'bootstrap_ci.json'}")
    print(f"[elapsed] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
