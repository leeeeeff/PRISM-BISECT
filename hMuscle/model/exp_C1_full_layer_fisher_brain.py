"""
exp_C1_full_layer_fisher_brain.py
==================================
Brain-tissue version of exp_C1_full_layer_fisher.py.

Reuses 279-GO selection built from the MUSCLE training gene list (so the GO set
is identical between muscle and brain — apple-to-apple).

Fisher discriminant per (GO g, layer L) on brain ESM-2 embeddings:
  F(g,L) = ||mu_pos - mu_neg||^2 / (mean(var_pos) + mean(var_neg))
Peak layer per GO = argmax_L F(g,L).

Resource-aware:
  - CPU-only (numpy). Threads capped via OMP/MKL/OPENBLAS env.
  - nice +10 to avoid crowding shared server.
  - Loads one layer at a time (never all 30 in RAM at once).
"""
from __future__ import annotations

import os
# Cap BLAS threads BEFORE numpy import — shared server, be polite.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "8"

import gzip, json, time
import numpy as np
from collections import defaultdict
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

try:
    os.nice(10)
except Exception:
    pass

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
MODEL_DIR = ROOT / "hMuscle/model"
DATA = ROOT / "hMuscle/data"
BRAIN_DIR = DATA / "brain_isoquant_esm2/full"
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


def build_go_labels_from_muscle_train():
    """Use MUSCLE training genes to define the 279-GO set — identical to
    exp_C1_full_layer_fisher.py so brain result is directly comparable."""
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
    return sorted(selected), go_info, go_genes, sym2id


def compute_fisher_layer(X, Y):
    """
    X: (N, D) — layer L embedding
    Y: (N, G) — binary label matrix
    F(g) = ||mu_p - mu_n||^2 / (mean(var_p) + mean(var_n))
    """
    G = Y.shape[1]
    fisher = np.zeros(G, dtype=np.float32)
    for gi in range(G):
        y = Y[:, gi]
        s = y.sum()
        if s == 0 or s == len(y):
            continue
        pos_mask = y == 1
        neg_mask = ~pos_mask
        mu_p = X[pos_mask].mean(0)
        mu_n = X[neg_mask].mean(0)
        num = ((mu_p - mu_n) ** 2).sum()
        var_p = X[pos_mask].var(0).mean()
        var_n = X[neg_mask].var(0).mean()
        denom = var_p + var_n + 1e-9
        fisher[gi] = num / denom
    return fisher


def main():
    t0 = time.time()

    print("[1] Build 279 GO set from MUSCLE training genes (apple-to-apple)...")
    selected_gos, go_info, go_genes, sym2id = build_go_labels_from_muscle_train()
    cat_counts = defaultdict(int)
    for g in selected_gos:
        cat_counts[go_info[g]['cat']] += 1
    print(f"   279-GO seed set: {len(selected_gos)} ({dict(cat_counts)})")

    print("[2] Load BRAIN gene symbols -> NCBI IDs -> per-isoform labels...")
    brain_syms = np.load(BRAIN_DIR / "brain_full_gene_names.npy",
                         allow_pickle=True)
    brain_syms = [clean_sym(s) for s in brain_syms]
    brain_ids = [sym2id.get(s, s) for s in brain_syms]
    N_brain = len(brain_ids)
    print(f"   N_brain isoforms = {N_brain}")

    Y_br = np.zeros((N_brain, len(selected_gos)), dtype=np.int8)
    for gi, go in enumerate(selected_gos):
        pos_set = go_genes[go]
        Y_br[:, gi] = np.array([1 if g in pos_set else 0 for g in brain_ids],
                                dtype=np.int8)
    valid = Y_br.sum(0) >= 10
    valid_gos = [selected_gos[i] for i in range(len(selected_gos)) if valid[i]]
    Y_br = Y_br[:, valid]
    val_cat = defaultdict(int)
    for g in valid_gos:
        val_cat[go_info[g]['cat']] += 1
    print(f"   Valid GOs (>=10 brain pos): {len(valid_gos)} ({dict(val_cat)})")

    print(f"\n[3] Fisher-per-layer (30 x {len(valid_gos)}) on BRAIN...")
    fisher_mat = np.zeros((len(valid_gos), N_LAYERS), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        t1 = time.time()
        p = BRAIN_DIR / f"brain_full_esm2_layer{L:02d}_t30_150M.npy"
        X = np.load(p).astype(np.float32)
        f_L = compute_fisher_layer(X, Y_br)
        fisher_mat[:, L - 1] = f_L
        del X
        elapsed = time.time() - t1
        print(f"   L{L:02d}: {elapsed:.1f}s   [{time.time()-t0:.1f}s total]",
              flush=True)

    per_go = {}
    for gi, go in enumerate(valid_gos):
        curve = fisher_mat[gi]
        peak = int(np.argmax(curve)) + 1
        per_go[go] = {
            "name": go_info[go]['name'],
            "category": go_info[go]['cat'],
            "n_pos_br": int(Y_br[:, gi].sum()),
            "fisher_per_layer": [float(x) for x in curve],
            "peak_layer": peak,
            "peak_fisher": float(curve.max()),
        }

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

    print("\n  BRAIN peak-layer distribution x category (Fisher):")
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
        "tissue": "brain",
        "n_layers": N_LAYERS,
        "n_valid_gos": len(valid_gos),
        "per_go": per_go,
        "bucket_summary": bucket,
        "cat_totals": cat_tot,
        "method": "Fisher discriminant per layer (brain isoforms, muscle-defined 279-GO seed)",
        "elapsed_sec": time.time() - t0,
    }
    with open(OUT_DIR / "layer_probe_279_fisher_brain.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {OUT_DIR/'layer_probe_279_fisher_brain.json'}")
    print(f"[elapsed] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
