#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_f_plm_fisher_sweep.py
--------------------------
Per-layer Fisher discriminant sweep, generalized across PLM architectures
(companion to exp_C1_full_layer_fisher.py, which was ESM-2 150M / 279-term only).

Purpose: find the layer where GO-discriminative signal peaks (Fisher discriminant),
INSTEAD OF assuming a fixed 50% depth ratio (exp_f_plm_scale_scan.py's assumption).
Uses the SAME 82-term MF label set as the concat AUPRC benchmark for direct
comparability (not the 279-term BP+MF+CC set used in the original ESM-2-only script).

Fisher discriminant per (GO g, layer L):
  F(g,L) = ||mu_pos - mu_neg||^2 / (var_pos + var_neg)

Computed independently on TEST embeddings (matches original exp_C1 methodology)
AND on TRAIN embeddings (robustness check against single-split peak-layer noise —
peak-layer selection is a design decision, not a benchmarked metric, but a peak
that only appears in one split is weaker evidence than one that appears in both).

Usage:
  python3 exp_f_plm_fisher_sweep.py --tag t30_150M     --n_layers 30
  python3 exp_f_plm_fisher_sweep.py --tag prot_t5_xl   --n_layers 24
  python3 exp_f_plm_fisher_sweep.py --tag ankh_base    --n_layers 48
"""
import argparse, json, os, time
import numpy as np
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from exp_f_plm_scale_scan import load_labels

DATA_DIR = '../data'
OUT_DIR  = '../../reports/exp_f_plm_scale'
os.makedirs(OUT_DIR, exist_ok=True)


def compute_fisher_layer(X, Y):
    """F(g) = ||mu_p - mu_n||^2 / (var_p + var_n).  Identical formula to exp_C1."""
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
        num = ((mu_p - mu_n) ** 2).sum()
        var_p = X[pos_mask].var(0).mean()
        var_n = X[neg_mask].var(0).mean()
        fisher[gi] = num / (var_p + var_n + 1e-9)
    return fisher


def layer_path(tag, L, split):
    if split == 'train':
        return f'{DATA_DIR}/esm2_train_human_layer{L:02d}_{tag}.npy'
    else:
        return f'{DATA_DIR}/esm2_layer_{L:02d}_{tag}.npy'


def sweep(tag, n_layers, Y, valid_mask, split):
    fisher_mat = np.zeros((n_layers, Y.shape[1]), dtype=np.float32)
    missing = []
    for L in range(1, n_layers + 1):
        p = layer_path(tag, L, split)
        if not os.path.exists(p):
            missing.append(L)
            continue
        X = np.load(p).astype(np.float32)
        fisher_mat[L - 1] = compute_fisher_layer(X, Y)
        del X
    if missing:
        print(f"  [WARN] {split}: missing layers {missing}")
    return fisher_mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--n_layers', type=int, required=True)
    args = ap.parse_args()
    tag, n_layers = args.tag, args.n_layers

    print("=" * 70)
    print(f"  Per-layer Fisher sweep: tag={tag}  n_layers={n_layers}")
    print("  Label set: 82-MF (same as concat AUPRC benchmark)")
    print("=" * 70)

    t0 = time.time()
    Y_tr, Y_te, valid_mask = load_labels()
    Y_tr, Y_te = Y_tr[:, valid_mask], Y_te[:, valid_mask]
    n_go = Y_tr.shape[1]
    print(f"  Valid GO terms: {n_go}  (train N={Y_tr.shape[0]}, test N={Y_te.shape[0]})")

    print(f"\n[test split]")
    fisher_te = sweep(tag, n_layers, Y_te, valid_mask, 'test')
    print(f"[train split]")
    fisher_tr = sweep(tag, n_layers, Y_tr, valid_mask, 'train')

    def summarize(fisher_mat, label):
        # mean Fisher across GO terms per layer (z-normalized per-GO before averaging,
        # so high-Fisher terms don't dominate the aggregate curve)
        valid_L = fisher_mat.sum(axis=1) > 0
        norm = fisher_mat.copy()
        for gi in range(norm.shape[1]):
            col = norm[:, gi]
            if col[valid_L].std() > 1e-9:
                norm[:, gi] = (col - col[valid_L].mean()) / col[valid_L].std()
        mean_curve = norm[:, :].mean(axis=1)
        mean_curve[~valid_L] = -np.inf
        peak_L = int(np.argmax(mean_curve)) + 1
        peak_ratio = peak_L / n_layers * 100
        per_go_peak = [int(np.argmax(fisher_mat[:, gi])) + 1 if fisher_mat[:, gi].sum() > 0 else -1
                       for gi in range(fisher_mat.shape[1])]
        median_per_go_peak = float(np.median([p for p in per_go_peak if p > 0]))
        print(f"  [{label}] aggregate peak: L{peak_L} ({peak_ratio:.0f}% depth)  "
              f"median per-GO peak: L{median_per_go_peak:.1f} ({median_per_go_peak/n_layers*100:.0f}% depth)")
        return {
            'peak_layer_aggregate': peak_L,
            'peak_ratio_pct_aggregate': round(peak_ratio, 1),
            'median_per_go_peak_layer': median_per_go_peak,
            'median_per_go_peak_ratio_pct': round(median_per_go_peak / n_layers * 100, 1),
            'mean_curve_zscore': [float(x) if np.isfinite(x) else None for x in mean_curve],
        }

    print()
    res_te = summarize(fisher_te, 'TEST')
    res_tr = summarize(fisher_tr, 'TRAIN')

    out = {
        'tag': tag, 'n_layers': n_layers, 'n_go': n_go,
        'test': res_te, 'train': res_tr,
        'agreement': abs(res_te['peak_layer_aggregate'] - res_tr['peak_layer_aggregate']) <= max(2, n_layers // 10),
        'elapsed_sec': time.time() - t0,
    }
    outpath = f'{OUT_DIR}/fisher_sweep_{tag}.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  train/test peak agreement (within 10% depth): {out['agreement']}")
    print(f"  Saved: {outpath}")
    print(f"  [elapsed] {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
