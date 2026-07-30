"""
exp_B1_within_gene_ranking.py
=============================
Within-gene isoform-pair GO ranking AUC:

For each GO and each gene with ≥2 test isoforms, form isoform pairs.
A pair (i, j) has "label direction" if only ONE is a GO-positive (or has
different domain/functional evidence). Compute the AUC of "predict-which-
isoform-carries-the-GO" using each model's score.

Compared models (using existing score matrices, 18 BP GO):
  v15d MLP (baseline)     — reports/v15_bp_clean/score_matrix_18go_*.npy
  v17f* (L30 ∥ δ=L30-L15) — reports/v17f_bp/v17f_bp_score_matrix.npy
  v19  curve_vec_norm     — reports/v19_curve/v19_score_matrix.npy
  v20b w=5 (BEST)         — reports/v20b/w5_score_matrix.npy
  v20b w=7                — reports/v20b/w7_score_matrix.npy
  ESM-2 L30 raw + LR probe — recompute quickly

Output: reports/exp_B1_within_gene_ranking/results.json + per-GO table
"""
from __future__ import annotations

import os, json, time
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = ROOT / "model"
ID_DIR = DATA / "raw_data/data/id_lists"
ANNOT_DIR = DATA / "raw_data/data/annotations"
REP_DIR = ROOT.parent / "reports"
OUT_DIR = REP_DIR / "exp_B1_within_gene_ranking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GO_18 = [
    "GO:0007204", "GO:0045214", "GO:0006941", "GO:0006914",
    "GO:0043161", "GO:0007519", "GO:0042692", "GO:0055074",
    "GO:0007005", "GO:0007517", "GO:0032006", "GO:0030048",
    "GO:0006096", "GO:0007268", "GO:0007018", "GO:0031175",
    "GO:0030182", "GO:0000226",
]


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']:
        s = s.replace(c, '')
    return s


def load_ids():
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                ENSG2SYM[p[0]] = p[4]
    te_gene = np.load(MODEL_DIR / "my_gene_list_fixed.npy", allow_pickle=True)
    te_iso = np.load(MODEL_DIR / "my_isoform_list_fixed.npy", allow_pickle=True)
    sym_te = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
              for g in te_gene]
    return sym_te, [clean(i) for i in te_iso]


def load_labels_te(sym_te):
    """gene-level positive set → (N_te, 18) 0/1."""
    Y = np.zeros((len(sym_te), len(GO_18)), dtype=np.int8)
    for gi, go in enumerate(GO_18):
        pos = set()
        with open(ANNOT_DIR / "human_annotations_unified_bp.txt") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > 1 and go in parts[1:]:
                    pos.add(parts[0])
        Y[:, gi] = np.array([1 if s in pos else 0 for s in sym_te], dtype=np.int8)
    return Y


def within_gene_ranking_auc(scores, Y, sym_te, dom_counts):
    """
    For each GO g, form within-gene isoform pairs where the two isoforms
    differ in domain-count evidence:
      pair (i, j) from same gene
      set y_pair = 1 if scores[i, g] > scores[j, g] is CORRECT,
                   where correctness is defined by dom_counts (proxy for
                   which isoform "carries" more of the domain evidence).
    We evaluate: does score_g[i] - score_g[j] correlate with
                  dom_counts[i]  - dom_counts[j]  ?
    We use pair-wise ranking AUC:
      For each GO, collect all (i, j) same-gene pairs where dom_counts differ.
      y = 1 if dom_counts[i] > dom_counts[j], else 0 (skip ties).
      pred = scores[i, g] - scores[j, g].
      AUC = roc_auc_score(y, pred).
    Return per-GO AUC + macro mean.
    """
    g2i = defaultdict(list)
    for i, g in enumerate(sym_te):
        g2i[g].append(i)

    aucs = []
    per_go = {}
    for gi, go in enumerate(GO_18):
        y_pair, s_pair = [], []
        n_pairs = 0
        for gene, idxs in g2i.items():
            if len(idxs) < 2:
                continue
            # only genes where GO is positive
            if Y[idxs[0], gi] == 0:
                continue
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    if dom_counts[i] == dom_counts[j]:
                        continue
                    label = 1 if dom_counts[i] > dom_counts[j] else 0
                    diff = scores[i, gi] - scores[j, gi]
                    y_pair.append(label)
                    s_pair.append(diff)
                    n_pairs += 1
        if n_pairs < 20 or len(set(y_pair)) < 2:
            per_go[go] = {"auc": None, "n_pairs": n_pairs}
            continue
        auc = float(roc_auc_score(y_pair, s_pair))
        per_go[go] = {"auc": round(auc, 4), "n_pairs": n_pairs}
        aucs.append(auc)
    macro = float(np.mean(aucs)) if aucs else 0.0
    return macro, per_go


def load_score_matrices():
    """Return dict of model_name → score_matrix (N_te, 18)."""
    mats = {}
    paths = [
        ("v15d", REP_DIR / "v15_bp_clean/score_matrix_18go_20260519_1914.npy"),
        ("v17f-BP", REP_DIR / "v17f_bp/v17f_bp_score_matrix.npy"),
        ("v19", REP_DIR / "v19_curve/v19_score_matrix.npy"),
        ("v20b_w5", REP_DIR / "v20b/w5_score_matrix.npy"),
        ("v20b_w7", REP_DIR / "v20b/w7_score_matrix.npy"),
    ]
    for name, p in paths:
        if p.exists():
            arr = np.load(p)
            print(f"  {name:12s}  shape={arr.shape}   file={p.name}")
            if arr.shape[1] != len(GO_18):
                print(f"    [WARN] {arr.shape[1]} cols vs {len(GO_18)} expected")
            mats[name] = arr
        else:
            print(f"  {name:12s}  MISSING ({p})")
    return mats


def build_l30_lr_baseline(Y_te):
    """Quick LR baseline on ESM-2 L30 → produce score matrix."""
    print("\n  Building LR baseline on L30 …")
    X_tr = np.load(DATA / "esm2_train_human_t30_150M.npy").astype(np.float32)
    X_te = np.load(DATA / "esm2_embeddings_t30_150M.npy").astype(np.float32)

    # Train labels
    tr_gene = np.load(ID_DIR / "train_gene_list.npy", allow_pickle=True)
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                ENSG2SYM[p[0]] = p[4]
    sym_tr = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
              for g in tr_gene]

    Y_tr = np.zeros((len(sym_tr), len(GO_18)), dtype=np.int8)
    for gi, go in enumerate(GO_18):
        pos = set()
        with open(ANNOT_DIR / "human_annotations_unified_bp.txt") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > 1 and go in parts[1:]:
                    pos.add(parts[0])
        Y_tr[:, gi] = np.array([1 if s in pos else 0 for s in sym_tr],
                                dtype=np.int8)

    scores = np.zeros((X_te.shape[0], len(GO_18)), dtype=np.float32)
    for gi in range(len(GO_18)):
        if Y_tr[:, gi].sum() < 5:
            continue
        m = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        m.fit(X_tr, Y_tr[:, gi])
        scores[:, gi] = m.predict_proba(X_te)[:, 1]
    return scores


def main():
    t0 = time.time()
    print("[1] IDs & labels …")
    sym_te, iso_te = load_ids()
    Y = load_labels_te(sym_te)
    print(f"   N_te={len(sym_te)}, GO_18={Y.sum(axis=0).tolist()}")

    print("\n[2] Domain counts (proxy for isoform-specific evidence) …")
    dm = np.load(ROOT / "results_isoform/features/domain_matrix_proper_test.npy",
                 mmap_mode="r")
    dc = np.array(dm.sum(1)).ravel()
    del dm
    print(f"   dom_counts: min={dc.min()} max={dc.max()} median={np.median(dc):.1f}")

    print("\n[3] Loading score matrices …")
    mats = load_score_matrices()

    print("\n[4] LR baseline on L30 …")
    lr_scores = build_l30_lr_baseline(Y)
    mats["LR_L30"] = lr_scores

    print("\n[5] Compute within-gene ranking AUC per model …")
    results = {}
    for name, S in mats.items():
        if S.shape[1] != len(GO_18):
            print(f"   {name:12s}: SKIP (shape mismatch)")
            continue
        macro, per_go = within_gene_ranking_auc(S, Y, sym_te, dc)
        results[name] = {"macro_auc": round(macro, 4), "per_go": per_go}
        print(f"   {name:12s}  macro within-gene AUC = {macro:.4f}")

    out = {"models": results,
           "n_te": len(sym_te),
           "GO_18": GO_18,
           "elapsed_sec": time.time() - t0}
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Console summary
    print(f"\n{'='*60}")
    print(f"  Within-gene ranking AUC summary")
    print(f"{'='*60}")
    print(f"  {'Model':15s}  {'macro AUC':>10s}")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["macro_auc"],
                          reverse=True):
        print(f"  {name:15s}  {r['macro_auc']:>10.4f}")
    print(f"\n[saved] {OUT_DIR/'results.json'}")
    print(f"[elapsed] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
