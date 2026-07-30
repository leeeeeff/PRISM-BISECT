"""
exp_B3b_paired_bootstrap.py
============================
Paired bootstrap on (T3/T12_v20b_w5 − T3/T12_v15d) and (AUPRC_v20b_w5 − AUPRC_v15d).

For each bootstrap resample of test isoforms, compute both metrics for
BOTH models on the SAME resample, then take the difference. This controls
for the shared source of variance (identity of test isoforms).

Also computes pairwise CI for v20b w=5 vs {v15d, v17f-BP, v19, v20b w=7}.
"""
from __future__ import annotations

import json, time
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import average_precision_score

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
MODEL_DIR = ROOT / "hMuscle/model"
FEAT_DIR = ROOT / "hMuscle/results_isoform/features"
ID_DIR = ROOT / "hMuscle/data/raw_data/data/id_lists"
ANNOT_DIR = ROOT / "hMuscle/data/raw_data/data/annotations"
REP_DIR = ROOT / "reports"
OUT_DIR = REP_DIR / "exp_B3b_paired_bootstrap"
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
    aps = []
    for gi in range(len(GO_18)):
        y = Y[iso_idx, gi]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        s = S[iso_idx, gi]
        aps.append(average_precision_score(y, s))
    return float(np.mean(aps)) if aps else 0.0


def t3_t12_ratio(S, Y, iso_idx, sym_te_arr, type3_mask_gene, gene_id_arr):
    """
    gene_id_arr: integer gene index per isoform (precomputed).
    type3_mask_gene: bool array[len(unique_genes)] indicating type-3 gene.
    """
    sub_gid = gene_id_arr[iso_idx]
    ratios = []
    for gi in range(len(GO_18)):
        preds = S[iso_idx, gi]
        # spread per gene: max - min
        order = np.argsort(sub_gid, kind='stable')
        sg = sub_gid[order]; sp = preds[order]
        # find gene boundaries
        boundaries = np.concatenate([[0], 1 + np.where(np.diff(sg) != 0)[0],
                                     [len(sg)]])
        gene_spreads = []
        gene_ids_here = []
        for k in range(len(boundaries) - 1):
            a, b = boundaries[k], boundaries[k + 1]
            if b - a < 2: continue
            gene_spreads.append(sp[a:b].max() - sp[a:b].min())
            gene_ids_here.append(sg[a])
        if not gene_spreads: continue
        gs = np.array(gene_spreads)
        gid = np.array(gene_ids_here)
        t3_mask = type3_mask_gene[gid]
        t3 = gs[t3_mask]; t12 = gs[~t3_mask]
        if len(t3) == 0 or len(t12) == 0: continue
        m_t3 = t3.mean(); m_t12 = t12.mean()
        if m_t12 < 1e-9: continue
        ratios.append(m_t3 / m_t12)
    return float(np.mean(ratios)) if ratios else 0.0


def paired_bootstrap(S_A, S_B, Y, sym_te_arr, type3_mask_gene, gene_id_arr,
                     n_boot=N_BOOT, seed=SEED):
    N = S_A.shape[0]
    rng = np.random.default_rng(seed)
    ap_diff, tr_diff = [], []
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        ap_A = macro_auprc(S_A, Y, idx)
        ap_B = macro_auprc(S_B, Y, idx)
        tr_A = t3_t12_ratio(S_A, Y, idx, sym_te_arr, type3_mask_gene, gene_id_arr)
        tr_B = t3_t12_ratio(S_B, Y, idx, sym_te_arr, type3_mask_gene, gene_id_arr)
        ap_diff.append(ap_A - ap_B)
        tr_diff.append(tr_A - tr_B)
    ap_d = np.array(ap_diff)
    tr_d = np.array(tr_diff)
    return {
        "auprc_diff": {
            "mean": float(np.mean(ap_d)),
            "ci_lo": float(np.percentile(ap_d, 2.5)),
            "ci_hi": float(np.percentile(ap_d, 97.5)),
            "p_gt_0": float((ap_d <= 0).mean()),
        },
        "t3t12_diff": {
            "mean": float(np.mean(tr_d)),
            "ci_lo": float(np.percentile(tr_d, 2.5)),
            "ci_hi": float(np.percentile(tr_d, 97.5)),
            "p_gt_0": float((tr_d <= 0).mean()),
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

    # Encode genes as integer IDs, and precompute type-3 mask over unique genes
    unique_genes = sorted(set(sym_te))
    gene_to_id = {g: i for i, g in enumerate(unique_genes)}
    gene_id_arr = np.array([gene_to_id[s] for s in sym_te], dtype=np.int32)
    type3_mask_gene = np.array([g in type3_genes for g in unique_genes],
                                dtype=bool)
    sym_te_arr = np.array(sym_te, dtype=object)
    print(f"   N_te={len(sym_te)}, unique_genes={len(unique_genes)},"
          f"  Type-3 genes: {int(type3_mask_gene.sum())}")

    paths = {
        "v15d":    REP_DIR / "v15_bp_clean/score_matrix_18go_20260519_1914.npy",
        "v17f-BP": REP_DIR / "v17f_bp/v17f_bp_score_matrix.npy",
        "v19":     REP_DIR / "v19_curve/v19_score_matrix.npy",
        "v20b_w5": REP_DIR / "v20b/w5_score_matrix.npy",
        "v20b_w7": REP_DIR / "v20b/w7_score_matrix.npy",
    }
    mats = {k: np.load(v) for k, v in paths.items() if v.exists()}

    baseline = "v20b_w5"
    others = ["v15d", "v17f-BP", "v19", "v20b_w7"]
    S_base = mats[baseline]

    results = {}
    for other in others:
        print(f"\n  Paired bootstrap: {baseline} vs {other} …")
        t1 = time.time()
        S_other = mats[other]
        r = paired_bootstrap(S_base, S_other, Y, sym_te_arr,
                             type3_mask_gene, gene_id_arr,
                             N_BOOT, SEED)
        r["elapsed"] = round(time.time() - t1, 1)
        results[f"{baseline}_minus_{other}"] = r
        ap = r["auprc_diff"]
        tr = r["t3t12_diff"]
        print(f"    ΔAUPRC = {ap['mean']:+.4f}  [{ap['ci_lo']:+.4f}, {ap['ci_hi']:+.4f}]  "
              f"p(diff ≤ 0)={ap['p_gt_0']:.4f}")
        print(f"    ΔT3/T12 = {tr['mean']:+.4f}  [{tr['ci_lo']:+.4f}, {tr['ci_hi']:+.4f}]  "
              f"p(diff ≤ 0)={tr['p_gt_0']:.4f}")

    out = {"n_boot": N_BOOT, "seed": SEED, "baseline": baseline,
           "comparisons": results, "elapsed_sec": time.time() - t0}
    with open(OUT_DIR / "paired_ci.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {OUT_DIR/'paired_ci.json'}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
