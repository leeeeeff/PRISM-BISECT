"""
eval_brain_isoform_resolution.py
=================================
Evaluate ISOFORM-LEVEL RESOLUTION on brain zero-shot, not gene-level AUPRC.

Motivation:
  Gene-mean baseline achieves brain AUPRC ~0.811 → most macro-AUPRC signal is
  gene-level information. If v20b trajectory concat successfully injects
  isoform-level signal, gene-level AUPRC may DROP while isoform-resolution
  metrics IMPROVE. This is the true axis of v20b's contribution.

Metrics (all computed against brain gene grouping):
  1. Within-gene spread mean:    mean over genes of (max_pred - min_pred)
                                 across isoforms of that gene.
  2. Within-gene ranking AUC:    per GO, for each gene with mixed pos/neg
                                 isoforms, AUC of "positive isoform > negative
                                 isoform within same gene". Aggregate mean.
  3. Within-gene Kendall tau:    per GO, correlation of prediction rank with
                                 label rank within each mixed gene.

Compared:
  - v15d brain zero-shot: /home/.../reports/v15d_brain_eval/brain_full_score_matrix_20260519_2125.npy
  - v20b brain w=5:       /home/.../reports/v20b_brain/w5_brain_score_matrix.npy
  - v20b brain w=7:       /home/.../reports/v20b_brain/w7_brain_score_matrix.npy  (may not exist yet)
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
BRAIN_DIR = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
OUT_DIR = ROOT / "reports" / "v20b_brain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GO_18 = [
    "GO:0007204", "GO:0045214", "GO:0006941", "GO:0006914",
    "GO:0043161", "GO:0007519", "GO:0042692", "GO:0055074",
    "GO:0007005", "GO:0007517", "GO:0032006", "GO:0030048",
    "GO:0006096", "GO:0007268", "GO:0007018", "GO:0031175",
    "GO:0030182", "GO:0000226",
]

MODELS = [
    ("v15d_brain", ROOT / "reports/v15d_brain_eval/brain_full_score_matrix_20260519_2125.npy"),
    ("v20b_w5",    OUT_DIR / "w5_brain_score_matrix.npy"),
    ("v20b_w7",    OUT_DIR / "w7_brain_score_matrix.npy"),
]


def within_gene_spread(preds: np.ndarray, gene_of: np.ndarray) -> dict:
    """
    preds: (N, K) K = 18 GOs
    gene_of: (N,) gene symbol per isoform
    Returns per-GO within-gene spread mean, coefficient of variation, and
    stratified by isoform count.
    """
    K = preds.shape[1]
    g2idx = defaultdict(list)
    for i, g in enumerate(gene_of):
        g2idx[g].append(i)
    multi = [g for g, idx in g2idx.items() if len(idx) >= 2]
    genes_2 = [g for g in multi if len(g2idx[g]) == 2]
    genes_3p = [g for g in multi if len(g2idx[g]) >= 3]
    print(f"  multi-iso genes: {len(multi)} "
          f"(n_iso=2: {len(genes_2)}, n_iso>=3: {len(genes_3p)})")

    def group_spread(genes, k):
        arr = []
        for g in genes:
            idx = g2idx[g]
            s = preds[idx, k]
            arr.append(float(s.max() - s.min()))
        return arr

    def group_cv(genes, k):
        arr = []
        for g in genes:
            idx = g2idx[g]
            s = preds[idx, k]
            mu = s.mean()
            if mu > 1e-9:
                arr.append(float(s.std() / mu))
        return arr

    per_go = {"spread_all": [], "spread_2": [], "spread_3p": [],
              "cv_all": []}
    for k in range(K):
        sp_all = group_spread(multi, k)
        sp_2   = group_spread(genes_2, k)
        sp_3p  = group_spread(genes_3p, k)
        cv     = group_cv(multi, k)
        per_go["spread_all"].append(float(np.mean(sp_all))     if sp_all else float("nan"))
        per_go["spread_2"].append(  float(np.mean(sp_2))       if sp_2   else float("nan"))
        per_go["spread_3p"].append( float(np.mean(sp_3p))      if sp_3p  else float("nan"))
        per_go["cv_all"].append(    float(np.mean(cv))         if cv     else float("nan"))

    return {
        "per_go": per_go,
        "macro_spread_all":  float(np.nanmean(per_go["spread_all"])),
        "macro_spread_2":    float(np.nanmean(per_go["spread_2"])),
        "macro_spread_3p":   float(np.nanmean(per_go["spread_3p"])),
        "macro_cv":          float(np.nanmean(per_go["cv_all"])),
        "n_multi_iso_genes": len(multi),
        "n_iso2": len(genes_2), "n_iso3p": len(genes_3p),
    }


def within_gene_ranking(preds: np.ndarray, labels: np.ndarray,
                        gene_of: np.ndarray) -> dict:
    """
    For each GO, restrict to genes with >=2 isoforms AND mixed labels
    (>=1 pos, >=1 neg within the gene). Compute per-gene pairwise ordering
    accuracy: probability that P(pos_iso) > P(neg_iso).
    Aggregate mean across such genes.
    """
    K = preds.shape[1]
    g2idx = defaultdict(list)
    for i, g in enumerate(gene_of):
        g2idx[g].append(i)

    per_go_auc = []
    per_go_n = []
    for k in range(K):
        pair_correct = 0
        pair_total = 0
        for g, idx in g2idx.items():
            if len(idx) < 2:
                continue
            y = labels[idx, k]
            p = preds[idx, k]
            npos = int(y.sum())
            nneg = len(y) - npos
            if npos == 0 or nneg == 0:
                continue
            pos_p = p[y == 1]
            neg_p = p[y == 0]
            # count strictly greater (0.5 for ties)
            for pp in pos_p:
                for nn in neg_p:
                    if pp > nn:
                        pair_correct += 1
                    elif pp == nn:
                        pair_correct += 0.5
                    pair_total += 1
        auc = float(pair_correct / pair_total) if pair_total > 0 else float("nan")
        per_go_auc.append(auc)
        per_go_n.append(pair_total)
    valid = [a for a in per_go_auc if not np.isnan(a)]
    return {
        "per_go_ranking_auc": per_go_auc,
        "per_go_pair_n": per_go_n,
        "macro_ranking_auc": float(np.mean(valid)) if valid else float("nan"),
        "n_valid_gos": len(valid),
    }


def build_brain_labels():
    """Build brain (N, 18) label matrix using brain gene names → BP annotation file."""
    ANNOT = ROOT / "hMuscle/data/raw_data/data/annotations/human_annotations_unified_bp.txt"
    go_pos = {go: set() for go in GO_18}
    with open(ANNOT) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            sym = parts[0]
            for go in parts[1:]:
                if go in go_pos:
                    go_pos[go].add(sym)
    syms = np.load(BRAIN_DIR / "brain_full_gene_names.npy", allow_pickle=True)
    syms = np.array([str(s) for s in syms])
    N = len(syms)
    Y = np.zeros((N, len(GO_18)), dtype=np.int8)
    for k, go in enumerate(GO_18):
        pos_set = go_pos[go]
        Y[:, k] = np.array([1 if s in pos_set else 0 for s in syms], dtype=np.int8)
    return Y, syms


def main():
    print("[1] Load brain gene symbols + build labels (18 GO)")
    Y, syms = build_brain_labels()
    print(f"  N_brain={len(syms)}")
    for k, go in enumerate(GO_18):
        print(f"  {go}: n_pos={int(Y[:, k].sum())}")

    results = {}
    for name, path in MODELS:
        if not Path(path).exists():
            print(f"\n[skip] {name}: {path} does not exist yet")
            continue
        print(f"\n=== {name} : {path} ===")
        preds = np.load(path)
        print(f"  shape={preds.shape}")
        assert preds.shape[0] == len(syms), f"row count mismatch for {name}"
        # Some model outputs may have >18 GOs; align to first 18 assuming same ordering
        if preds.shape[1] != len(GO_18):
            print(f"  [warn] pred has {preds.shape[1]} cols, expected {len(GO_18)}. "
                  f"Taking first {len(GO_18)}.")
            preds = preds[:, :len(GO_18)]

        print("  [a] within-gene spread ...")
        sp = within_gene_spread(preds, syms)
        print(f"    macro spread all  = {sp['macro_spread_all']:.4f} "
              f"(n_multi={sp['n_multi_iso_genes']})")
        print(f"    macro spread n=2  = {sp['macro_spread_2']:.4f} "
              f"(n_genes={sp['n_iso2']})")
        print(f"    macro spread n>=3 = {sp['macro_spread_3p']:.4f} "
              f"(n_genes={sp['n_iso3p']})")
        print(f"    macro CV (all)    = {sp['macro_cv']:.4f}")

        print("  [b] within-gene ranking AUC ...")
        rk = within_gene_ranking(preds, Y, syms)
        print(f"    macro ranking AUC = {rk['macro_ranking_auc']:.4f} "
              f"(over {rk['n_valid_gos']} GOs with mixed genes)")
        for k, go in enumerate(GO_18):
            n = rk["per_go_pair_n"][k]
            a = rk["per_go_ranking_auc"][k]
            print(f"      {go}: AUC={a:.4f}  (n_pairs={n})")

        results[name] = {
            "path": str(path),
            "spread": sp,
            "ranking": rk,
        }

    with open(OUT_DIR / "brain_isoform_resolution_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {OUT_DIR/'brain_isoform_resolution_metrics.json'}")

    # Delta table
    if "v15d_brain" in results:
        base = results["v15d_brain"]
        base_sp   = base['spread']['macro_spread_all']
        base_sp2  = base['spread']['macro_spread_2']
        base_sp3  = base['spread']['macro_spread_3p']
        base_cv   = base['spread']['macro_cv']
        print("\n=== Δ vs v15d brain zero-shot (isoform-resolution) ===")
        print(f"{'Model':12s}  {'Spread_all':>10s}  {'Δ':>8s}  "
              f"{'Spread_n=2':>10s}  {'Δ':>8s}  "
              f"{'Spread_n>=3':>11s}  {'Δ':>8s}  "
              f"{'CV':>8s}  {'Δ':>8s}")
        for name, r in results.items():
            s  = r['spread']
            sp = s['macro_spread_all']
            sp2 = s['macro_spread_2']
            sp3 = s['macro_spread_3p']
            cv  = s['macro_cv']
            print(f"{name:12s}  {sp:10.4f}  {sp-base_sp:+8.4f}  "
                  f"{sp2:10.4f}  {sp2-base_sp2:+8.4f}  "
                  f"{sp3:11.4f}  {sp3-base_sp3:+8.4f}  "
                  f"{cv:8.4f}  {cv-base_cv:+8.4f}")


if __name__ == "__main__":
    main()
