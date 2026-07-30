"""
exp_B2_window_ablation.py
=========================
Curve window ablation: w ∈ {0, 3, 10} 추가 훈련.
w=0(L30 only, ≈ v15d), w=3(중간), w=10(넓은 window).
w=5, w=7은 이미 학습되었으므로 재활용.

기존 v20b_window_sweep.py의 학습 코드를 재사용.
"""
from __future__ import annotations

import os, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks
import warnings; warnings.filterwarnings('ignore')

# Import shared components from v20b sweep
import sys
sys.path.insert(0, str(Path(__file__).parent))
from v20b_window_sweep import (
    GO_18, MID_GOs, load_fisher_peaks, window_curve,
    build_model, run_ensemble, load_labels, within_gene_spread,
    N_SEEDS,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = ROOT / "model"
FEAT_DIR = ROOT / "results_isoform" / "features"
ID_DIR = DATA / "raw_data/data/id_lists"
CACHE_DIR = ROOT.parent / "reports" / "v20_cache"
OUT_DIR = ROOT.parent / "reports" / "exp_B2_window_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W_LIST = [0, 3, 10]


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']:
        s = s.replace(c, '')
    return s


def run_w(w, peaks, Z_tr, Z_te, X_L30_tr, X_L30_te,
          sym_tr, sym_te, type3_genes):
    print(f"\n{'='*70}")
    print(f"  Ablation w={w}  (±{w} layers around Fisher peak)")
    if w == 0:
        print(f"  Note: w=0 → L30 only (no curve injection, ≈ v15d)")
    print(f"{'='*70}")

    auprc_list, spread_results = [], []
    score_matrix = np.zeros((Z_te.shape[0], len(GO_18)), dtype=np.float32)

    for gi, (go, go_name) in enumerate(GO_18.items()):
        t1 = time.time()
        peak = peaks[go]
        flag = "[MID]" if go in MID_GOs else "     "

        if w == 0:
            # L30 only baseline
            X_tr = X_L30_tr
            X_te = X_L30_te
            n_layers_used = 0
            win_idx = []
        else:
            c_tr_raw, win_idx = window_curve(Z_tr, peak, w)
            c_te_raw, _ = window_curve(Z_te, peak, w)
            sc = StandardScaler()
            c_tr = sc.fit_transform(c_tr_raw).astype(np.float32)
            c_te = sc.transform(c_te_raw).astype(np.float32)
            X_tr = np.concatenate([X_L30_tr, c_tr], axis=1)
            X_te = np.concatenate([X_L30_te, c_te], axis=1)
            n_layers_used = len(win_idx)

        input_dim = X_tr.shape[1]
        y_tr, y_te = load_labels(go, sym_tr, sym_te)
        if y_te.sum() == 0:
            continue

        preds = run_ensemble(X_tr, y_tr, X_te, input_dim)
        score_matrix[:, gi] = preds
        auprc = float(average_precision_score(y_te, preds))
        auprc_list.append(auprc)

        spread = within_gene_spread(preds, sym_te)
        t3_sp = [v for g, v in spread.items() if g in type3_genes]
        t12_sp = [v for g, v in spread.items() if g not in type3_genes]
        t3_t12 = float(np.mean(t3_sp) / np.mean(t12_sp)) if t3_sp and t12_sp else 0.0

        spread_results.append({
            "go": go, "go_name": go_name,
            "is_mid": go in MID_GOs,
            "peak_layer_1idx": peak + 1,
            "win_layers": [l + 1 for l in win_idx],
            "n_layers_used": n_layers_used,
            "input_dim": input_dim,
            "auprc": round(auprc, 4),
            "T3_spread_mean": round(float(np.mean(t3_sp)) if t3_sp else 0, 5),
            "T12_spread_mean": round(float(np.mean(t12_sp)) if t12_sp else 0, 5),
            "T3_T12_ratio": round(t3_t12, 4),
        })

        print(f"  [{gi+1:2d}/{len(GO_18)}] {flag} {go} {go_name[:20]:20s}  "
              f"dim={input_dim}  AUPRC={auprc:.4f}  T3/T12={t3_t12:.4f}  "
              f"({time.time()-t1:.0f}s)")

    macro = float(np.mean(auprc_list))
    all_ratios = [r["T3_T12_ratio"] for r in spread_results if r["T3_T12_ratio"] > 0]
    mid_ratios = [r["T3_T12_ratio"] for r in spread_results
                  if r["is_mid"] and r["T3_T12_ratio"] > 0]
    t3_all = float(np.mean(all_ratios)) if all_ratios else 0.0
    t3_mid = float(np.mean(mid_ratios)) if mid_ratios else 0.0

    print(f"\n  === w={w} Summary ===")
    print(f"  Macro AUPRC = {macro:.4f}  T3/T12 ALL = {t3_all:.4f}  MID = {t3_mid:.4f}")

    result = {
        "w": w,
        "n_layers_max": 0 if w == 0 else 2 * w + 1,
        "macro_auprc": round(macro, 4),
        "t3_t12_ratio_all": round(t3_all, 4),
        "t3_t12_ratio_mid": round(t3_mid, 4),
        "per_go": spread_results,
    }
    with open(OUT_DIR / f"w{w}_results.json", "w") as f:
        json.dump(result, f, indent=2)
    np.save(OUT_DIR / f"w{w}_score_matrix.npy", score_matrix)
    return result


def main():
    t0 = time.time()
    print(f"[exp_B2] Curve window ablation — w ∈ {W_LIST}")

    print("\n[1] Fisher peaks…")
    peaks = load_fisher_peaks()

    print("\n[2] IDs…")
    tr_gene = np.load(ID_DIR / "train_gene_list.npy", allow_pickle=True)
    te_gene = np.load(MODEL_DIR / "my_gene_list_fixed.npy", allow_pickle=True)
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                ENSG2SYM[p[0]] = p[4]
    sym_tr = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
              for g in tr_gene]
    sym_te = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
              for g in te_gene]
    print(f"   Train={len(sym_tr)}  Test={len(sym_te)}")

    print("\n[3] Type-3 genes…")
    dm = np.load(FEAT_DIR / "domain_matrix_proper_test.npy", mmap_mode="r")
    dc = np.array(dm.sum(1)).ravel(); del dm
    te_gene_base = [clean(g).split('.')[0] for g in te_gene]
    gene2dc = defaultdict(list)
    for i, g in enumerate(te_gene_base):
        gene2dc[g].append(dc[i])
    dc_range = {g: max(v) - min(v) for g, v in gene2dc.items()}
    type3_genes = {g for g, r in dc_range.items() if r == 0}
    print(f"   Type-3: {len(type3_genes)} genes")

    print("\n[4] Z cache & L30…")
    Z_tr = np.load(CACHE_DIR / "Z_tr.npy")
    Z_te = np.load(CACHE_DIR / "Z_te.npy")
    X_L30_tr = np.load(DATA / "esm2_train_human_t30_150M.npy").astype(np.float32)
    X_L30_te = np.load(DATA / "esm2_embeddings_t30_150M.npy").astype(np.float32)
    print(f"   Setup done [{time.time()-t0:.1f}s]")

    all_results = {}
    for w in W_LIST:
        r = run_w(w, peaks, Z_tr, Z_te, X_L30_tr, X_L30_te,
                  sym_tr, sym_te, type3_genes)
        all_results[f"w{w}"] = r

    with open(OUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[done] total elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
