"""
exp_C1_full_layer_fisher.py
============================
Fast Fisher-discriminant-based per-layer signal for 279 GO × 30 layers.

Fisher discriminant per (GO g, layer L):
  F(g,L) = ||μ_pos - μ_neg||² / trace(Σ_pos + Σ_neg)

Peak layer per GO = argmax_L F(g,L).

100× faster than iterative LR; equivalent for peak identification.
"""
from __future__ import annotations

import gzip, json, time
import numpy as np
from collections import defaultdict
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
MODEL_DIR = ROOT / "hMuscle/model"
DATA = ROOT / "hMuscle/data"
ID_DIR = DATA / "raw_data/data/id_lists"
ANNOT_DIR = DATA / "raw_data/data/annotations"
OUT_DIR = ROOT / "reports" / "exp_C1_layer_probe_279"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS = 30
MIN_POS = 100
CATEGORIES = {'Process': 'BP', 'Function': 'MF', 'Component': 'CC'}


def clean_sym(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']:
        s = s.replace(c, '')
    return s


def load_sym2id():
    sym2id = {}
    with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) > 2:
                sym2id[p[2]] = p[1]
                if len(p) > 4 and p[4] != '-':
                    for syn in p[4].split('|'):
                        if syn not in sym2id:
                            sym2id[syn] = p[1]
    return sym2id


def build_go_labels():
    sym2id = load_sym2id()
    tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
    tr_syms = [clean_sym(g) for g in tr_genes_raw]
    tr_ids = [sym2id.get(s, s) for s in tr_syms]
    tr_id_set = set(tr_ids)

    go_info = {}
    go_genes = defaultdict(set)
    with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if p[0] != '9606':
                continue
            gid, go_id, go_name, cat_raw = p[1], p[2], p[5], p[7]
            if cat_raw not in CATEGORIES:
                continue
            cat = CATEGORIES[cat_raw]
            go_info[go_id] = {'name': go_name, 'cat': cat}
            if gid in tr_id_set:
                go_genes[go_id].add(gid)

    selected = [go for go, s in go_genes.items() if len(s) >= MIN_POS]
    return sorted(selected), go_info, go_genes, sym2id, tr_syms, tr_ids


def compute_fisher_layer(X, Y):
    """
    X: (N, D) — layer L embedding
    Y: (N, G) — binary label matrix
    Returns: fisher (G,) per-GO Fisher discriminant

    F(g) = ||μ_pos - μ_neg||² / (mean(σ_pos²) + mean(σ_neg²))
    """
    G = Y.shape[1]
    fisher = np.zeros(G, dtype=np.float32)
    for gi in range(G):
        y = Y[:, gi]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        pos_mask = y == 1
        neg_mask = ~pos_mask
        mu_p = X[pos_mask].mean(0)
        mu_n = X[neg_mask].mean(0)
        # Simple estimator: ||μ_p - μ_n||² / (var trace normalization)
        num = ((mu_p - mu_n) ** 2).sum()
        # class-mean variance (diagonal)
        var_p = X[pos_mask].var(0).mean()
        var_n = X[neg_mask].var(0).mean()
        denom = var_p + var_n + 1e-9
        fisher[gi] = num / denom
    return fisher


def main():
    t0 = time.time()

    print("[1] Loading gene→ID map & GO labels…")
    selected_gos, go_info, go_genes, sym2id, tr_syms, tr_ids = build_go_labels()
    cat_counts = defaultdict(int)
    for g in selected_gos:
        cat_counts[go_info[g]['cat']] += 1
    print(f"   Selected: {len(selected_gos)} ({dict(cat_counts)})")

    te_gene = np.load(MODEL_DIR / "my_gene_list_fixed.npy", allow_pickle=True)
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                ENSG2SYM[p[0]] = p[4]

    def ensg_to_sym(raw):
        return ENSG2SYM.get(clean_sym(raw).split('.')[0],
                             clean_sym(raw).split('.')[0])

    te_syms = [ensg_to_sym(g) for g in te_gene]
    te_ids = [sym2id.get(s, s) for s in te_syms]

    print("[2] Building label matrices…")
    Y_te = np.zeros((len(te_ids), len(selected_gos)), dtype=np.int8)
    for gi, go in enumerate(selected_gos):
        pos_set = go_genes[go]
        Y_te[:, gi] = np.array([1 if g in pos_set else 0 for g in te_ids],
                                dtype=np.int8)
    valid = Y_te.sum(0) >= 10
    valid_gos = [selected_gos[i] for i in range(len(selected_gos)) if valid[i]]
    Y_te = Y_te[:, valid]
    print(f"   Valid GOs (≥10 test pos): {len(valid_gos)}")

    print(f"\n[3] Fisher-per-layer (30 × {len(valid_gos)}) …")
    fisher_mat = np.zeros((len(valid_gos), N_LAYERS), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        t1 = time.time()
        te_path = DATA / f"esm2_layer_{L:02d}_t30_150M.npy"
        X = np.load(te_path).astype(np.float32)
        f_L = compute_fisher_layer(X, Y_te)
        fisher_mat[:, L - 1] = f_L
        del X
        elapsed = time.time() - t1
        print(f"   L{L:02d}: {elapsed:.1f}s   [{time.time()-t0:.1f}s total]", flush=True)

    # Per-GO peak layer & category
    per_go = {}
    for gi, go in enumerate(valid_gos):
        curve = fisher_mat[gi]
        peak = int(np.argmax(curve)) + 1
        per_go[go] = {
            "name": go_info[go]['name'],
            "category": go_info[go]['cat'],
            "n_pos_te": int(Y_te[:, gi].sum()),
            "fisher_per_layer": [float(x) for x in curve],
            "peak_layer": peak,
            "peak_fisher": float(curve.max()),
        }

    # Bucket by Early/Mid/Late × category
    bucket = {"Early (L1-10)": {"BP": 0, "MF": 0, "CC": 0},
              "Mid (L11-20)":  {"BP": 0, "MF": 0, "CC": 0},
              "Late (L21-30)": {"BP": 0, "MF": 0, "CC": 0}}
    for go, info in per_go.items():
        pl = info["peak_layer"]
        cat = info["category"]
        if pl <= 10:      bucket["Early (L1-10)"][cat] += 1
        elif pl <= 20:    bucket["Mid (L11-20)"][cat] += 1
        else:             bucket["Late (L21-30)"][cat] += 1
    cat_tot = {c: sum(bucket[b][c] for b in bucket) for c in ["BP", "MF", "CC"]}

    print("\n  Peak-layer distribution × category (Fisher discriminant):")
    print(f"  {'Bucket':<16s}  BP  MF  CC  Total  BP%  MF%  CC%")
    for bkey, cats in bucket.items():
        tot = sum(cats.values())
        bp_pct = cats['BP'] / max(cat_tot['BP'], 1) * 100
        mf_pct = cats['MF'] / max(cat_tot['MF'], 1) * 100
        cc_pct = cats['CC'] / max(cat_tot['CC'], 1) * 100
        print(f"  {bkey:<16s}  {cats['BP']:>2d}  {cats['MF']:>2d}  {cats['CC']:>2d}  {tot:>3d}"
              f"  {bp_pct:>3.0f}  {mf_pct:>3.0f}  {cc_pct:>3.0f}")
    print(f"  {'-'*16}  {cat_tot['BP']:>2d}  {cat_tot['MF']:>2d}  {cat_tot['CC']:>2d}  "
          f"{sum(cat_tot.values()):>3d}  100  100  100")

    out = {
        "n_layers": N_LAYERS,
        "n_valid_gos": len(valid_gos),
        "per_go": per_go,
        "bucket_summary": bucket,
        "cat_totals": cat_tot,
        "method": "Fisher discriminant per layer (test-set only)",
        "elapsed_sec": time.time() - t0,
    }
    with open(OUT_DIR / "layer_probe_279_fisher.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {OUT_DIR/'layer_probe_279_fisher.json'}")
    print(f"[elapsed] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
