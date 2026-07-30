"""
exp_C1_full_layer_probe_279.py  (v2)
=====================================
Full 279-GO (BP+MF+CC) layer-wise LR probe on ESM-2 muscle test set.

Labels: derived from gene2go.gz (BP/MF/CC categories) matching the 279-GO
        set used by v_expanded_gomf.py.

For each GO g, compute per-layer AUPRC of a linear probe → identify Fisher peak.
Aggregate distribution: Early(L1-10)/Mid(L11-20)/Late(L21-30) × category.
"""
from __future__ import annotations

import gzip, json, time
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
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
    """Build GO→positive-gene-set from gene2go.gz for BP/MF/CC."""
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

    # Match v_expanded_gomf.py: select GO with ≥ MIN_POS training positives
    selected = [go for go, s in go_genes.items() if len(s) >= MIN_POS]
    return sorted(selected), go_info, go_genes, sym2id, tr_syms, tr_ids


def main():
    t0 = time.time()

    print("[1] Loading gene→ID map & GO labels …")
    selected_gos, go_info, go_genes, sym2id, tr_syms, tr_ids = build_go_labels()

    cat_counts = defaultdict(int)
    for g in selected_gos:
        cat_counts[go_info[g]['cat']] += 1
    print(f"   Selected GO ({MIN_POS}+ positives): {len(selected_gos)}")
    for c, n in sorted(cat_counts.items()):
        print(f"     {c}: {n}")

    # Test set — ENSG IDs, need ENSG → symbol → Entrez conversion
    te_gene = np.load(MODEL_DIR / "my_gene_list_fixed.npy", allow_pickle=True)
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                ENSG2SYM[p[0]] = p[4]

    def ensg_to_sym(raw):
        s = clean_sym(raw).split('.')[0]
        return ENSG2SYM.get(s, s)

    te_syms = [ensg_to_sym(g) for g in te_gene]
    te_ids = [sym2id.get(s, s) for s in te_syms]
    n_te_matched = sum(1 for i in te_ids if not i.startswith('ENSG'))
    print(f"   Test ID mapping: {n_te_matched}/{len(te_ids)} matched to Entrez")

    # Build label matrices
    print("\n[2] Building label matrices …")
    N_TR = len(tr_ids); N_TE = len(te_ids)
    N_GO = len(selected_gos)
    Y_tr = np.zeros((N_TR, N_GO), dtype=np.int8)
    Y_te = np.zeros((N_TE, N_GO), dtype=np.int8)
    for gi, go in enumerate(selected_gos):
        pos_gene_set = go_genes[go]
        Y_tr[:, gi] = np.array([1 if g in pos_gene_set else 0 for g in tr_ids],
                                dtype=np.int8)
        Y_te[:, gi] = np.array([1 if g in pos_gene_set else 0 for g in te_ids],
                                dtype=np.int8)
    print(f"   Y_tr shape={Y_tr.shape}, positives min={Y_tr.sum(0).min()} "
          f"median={int(np.median(Y_tr.sum(0)))} max={Y_tr.sum(0).max()}")
    print(f"   Y_te shape={Y_te.shape}, positives min={Y_te.sum(0).min()} "
          f"median={int(np.median(Y_te.sum(0)))} max={Y_te.sum(0).max()}")

    # Filter GO with ≥ 10 test positives
    valid_mask = Y_te.sum(0) >= 10
    print(f"   Filtering GO with ≥ 10 test positives: {int(valid_mask.sum())}/{N_GO}")
    valid_gos = [selected_gos[i] for i in range(N_GO) if valid_mask[i]]
    Y_tr = Y_tr[:, valid_mask]
    Y_te = Y_te[:, valid_mask]
    N_GO_valid = len(valid_gos)
    cat_counts_v = defaultdict(int)
    for g in valid_gos:
        cat_counts_v[go_info[g]['cat']] += 1
    print(f"   Valid cat: {dict(cat_counts_v)}")

    print(f"\n[3] Per-layer LR probe (30 layers × {N_GO_valid} GO)")
    per_go = {go: {"category": go_info[go]['cat'],
                   "name": go_info[go]['name'],
                   "n_pos_tr": int(Y_tr[:, i].sum()),
                   "n_pos_te": int(Y_te[:, i].sum()),
                   "auprc_per_layer": [0.0] * N_LAYERS}
              for i, go in enumerate(valid_gos)}

    for L in range(1, N_LAYERS + 1):
        t1 = time.time()
        tr_path = DATA / f"esm2_train_human_layer{L:02d}_t30_150M.npy"
        te_path = DATA / f"esm2_layer_{L:02d}_t30_150M.npy"
        if not (tr_path.exists() and te_path.exists()):
            print(f"   [skip] L{L:02d}: tr={tr_path.exists()} te={te_path.exists()}")
            continue
        X_tr = np.load(tr_path).astype(np.float32)
        X_te = np.load(te_path).astype(np.float32)

        for gi, go in enumerate(valid_gos):
            y_tr = Y_tr[:, gi]; y_te = Y_te[:, gi]
            m = LogisticRegression(max_iter=200, C=1.0,
                                    class_weight="balanced", n_jobs=1,
                                    solver='liblinear')
            m.fit(X_tr, y_tr)
            s = m.predict_proba(X_te)[:, 1]
            ap = float(average_precision_score(y_te, s))
            per_go[go]["auprc_per_layer"][L - 1] = ap

        del X_tr, X_te
        elapsed = time.time() - t1
        print(f"   Layer L{L:02d} done ({elapsed:.1f}s)  "
              f"[{time.time()-t0:.1f}s total]", flush=True)

        if L % 5 == 0:
            with open(OUT_DIR / "layer_probe_279_intermediate.json", "w") as f:
                json.dump({"per_go": per_go, "layer_done": L,
                           "valid_gos": valid_gos}, f, indent=2)

    # Post-process
    print("\n[4] Peak layer + Early/Mid/Late × category distribution …")
    for go, info in per_go.items():
        arr = np.array(info["auprc_per_layer"])
        info["peak_layer"] = int(np.argmax(arr)) + 1
        info["peak_auprc"] = float(arr.max())

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
    print("\n  Peak-layer distribution × category:")
    print(f"  {'Bucket':<16s}  BP  MF  CC  Total")
    for bkey, cats in bucket.items():
        tot = sum(cats.values())
        print(f"  {bkey:<16s}  {cats['BP']:>2d}  {cats['MF']:>2d}  {cats['CC']:>2d}  {tot:>3d}")
    print(f"  {'-'*16}  {cat_tot['BP']:>2d}  {cat_tot['MF']:>2d}  {cat_tot['CC']:>2d}  {sum(cat_tot.values()):>3d}")

    out = {
        "n_layers": N_LAYERS,
        "min_pos": MIN_POS,
        "n_valid_gos": N_GO_valid,
        "per_go": per_go,
        "bucket_summary": bucket,
        "cat_totals": cat_tot,
        "elapsed_sec": time.time() - t0,
    }
    with open(OUT_DIR / "layer_probe_279.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {OUT_DIR/'layer_probe_279.json'}")
    print(f"[elapsed] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
