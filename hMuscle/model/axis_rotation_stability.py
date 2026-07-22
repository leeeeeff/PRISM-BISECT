#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""axis_rotation_stability.py  (Option B: is the map's 'identified vs grey' axis boundary robust?)

The interpretability map names axes 0,1,3,5,6 ('identified') and leaves 2,7 grey. §12 argued axis2/7
sit in a near-degenerate low-variance block (eigenvalue ratios ~1.0-1.2) where individual PCA
*directions* are rotation-unstable, so naming them is fragile UNLESS pinned by an external anchor.
This tests that claim empirically AND checks whether the 'identified' axes 1/6 are any more stable
(they are named on similarly weak composition: axis1 |r|<=0.29, axis6 <=0.30, vs axis2 0.26).

METHOD — split-half re-extraction of the exact v20b joint-PCA (interp_v20b_pca_axes.py):
  muscle train trajectory (N=31668, 30 layers, 640-d), z-scored per layer with the SAVED layer stats
  (isolates PCA sampling variability from z-score variability). Split ISOFORMS into 2 halves (all 30
  layers of an isoform stay together -> no layer leakage). Fit PCA(8) on each half. For each original
  full-fit axis w_k (W_axes_8x640.npy):
    direction stability  s_k = max_j |cos(w_k, v_j^half)|         (is the individual direction preserved?)
    subspace  stability  u_k = || Proj_span(V^half) w_k ||        (is the 8-d block preserved?)
  Decisive signature of rotation-instability: u_k ~ 1 (block preserved) BUT s_k low (direction rotates
  within the block). A well-separated OR externally-anchored axis has s_k ~ 1. Averaged over 3 splits.

Read-only. nice/threads limited. ~2.5 GB load.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'; os.environ['MKL_NUM_THREADS'] = '4'
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
DATA = ROOT / 'hMuscle/data'
INTERP = ROOT / 'reports/v20b_pca_interp'
N_LAYERS, EMB, K = 30, 640, 8


def main():
    W = np.load(INTERP / 'W_axes_8x640.npy').astype(np.float64)      # (8,640) full-fit axes
    evr = [0.04294, 0.02921, 0.02192, 0.01813, 0.01776, 0.01548, 0.01461, 0.01301]
    mu = np.load(INTERP / 'layer_stats_mu.npy').astype(np.float32)   # (30,640)
    sd = np.load(INTERP / 'layer_stats_sd.npy').astype(np.float32)

    # load + z-score trajectory (N,30,640) with saved stats
    a0 = np.load(DATA / 'esm2_train_human_layer01_t30_150M.npy', mmap_mode='r')
    N = a0.shape[0]; del a0
    print(f"[load] muscle train N={N}, {N_LAYERS} layers ...", flush=True)
    tn = np.empty((N, N_LAYERS, EMB), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        arr = np.load(DATA / f'esm2_train_human_layer{L:02d}_t30_150M.npy').astype(np.float32)
        tn[:, L - 1, :] = (arr - mu[L - 1]) / sd[L - 1]
        del arr
    print(f"[z-scored] tn {tn.shape}", flush=True)

    def fit_half(iso_idx):
        flat = tn[iso_idx].reshape(len(iso_idx) * N_LAYERS, EMB)
        p = PCA(n_components=K, svd_solver='randomized', random_state=0).fit(flat)
        return p.components_.astype(np.float64)                       # (8,640) orthonormal rows

    def cos_and_subspace(Wref, V):
        # V rows orthonormal -> subspace projector energy = sum_j (w.v_j)^2 ; sqrt = norm of projection
        s = np.zeros(K); u = np.zeros(K); match = np.zeros(K, dtype=int)
        for k in range(K):
            c = np.abs(V @ Wref[k])                                   # |cos| to each half-axis (Wref[k] unit)
            j = int(np.argmax(c)); s[k] = c[j]; match[k] = j
            u[k] = np.sqrt(float((c ** 2).sum()))
        return s, u, match

    S = np.zeros((3, K)); U = np.zeros((3, K))          # DISJOINT half-A vs half-B (no shared data)
    Sf = np.zeros((3, K))                                 # full-fit W vs half (shares data; upper bound)
    rng = np.random.default_rng(42)
    for t in range(3):
        perm = rng.permutation(N); h1 = perm[:N // 2]; h2 = perm[N // 2:]
        Va = fit_half(h1); Vb = fit_half(h2)
        # DECISIVE: half-A axes vs half-B subspace (fully disjoint isoforms)
        s, u, _ = cos_and_subspace(Va, Vb)
        S[t] = s; U[t] = u
        # reference: full-fit W vs each half (data overlap -> optimistic)
        sa, _, _ = cos_and_subspace(W, Va); sb, _, _ = cos_and_subspace(W, Vb)
        Sf[t] = (sa + sb) / 2
        print(f"[split {t}] done", flush=True)

    s_mean = S.mean(0); s_sd = S.std(0); u_mean = U.mean(0); sf_mean = Sf.mean(0)
    names = {0: 'β-sheet/hydrophobic', 1: 'length/size(weak)', 2: 'Pro-turn(GREY)',
             3: 'DOMAIN(used,anchored)', 4: '(unnamed)', 5: 'length', 6: 'inv-domain/low-cplx',
             7: 'acidic-helix(GREY)'}
    print("\n" + "=" * 92)
    print("AXIS ROTATION STABILITY — split-half PCA re-extraction (full-fit axis vs half-fit subspace)")
    print("s_k = individual DIRECTION stability (max|cos|); u_k = 8-d SUBSPACE stability (proj norm)")
    print("=" * 92)
    print(f"{'axis':>4} {'evr':>7} {'gap':>5} {'A-vs-B dir s_k':>16} {'subspace u_k':>12} {'W-vs-half':>10}  name")
    for k in range(K):
        gap = evr[k] / evr[k + 1] if k + 1 < K else float('nan')
        flag = '  <-- rotates in block' if (u_mean[k] > 0.9 and s_mean[k] < 0.8) else ''
        print(f"{k:>4} {evr[k]:>7.4f} {gap:>5.2f} {s_mean[k]:>10.3f}±{s_sd[k]:.3f} {u_mean[k]:>12.3f} {sf_mean[k]:>10.3f}  {names[k]}{flag}")
    print("-" * 92)
    print("Read: s_k~1 = direction preserved (stable/anchored). u_k~1 & s_k low = subspace kept but")
    print("      individual axis ROTATES within a near-degenerate block -> naming that direction is fragile.")


if __name__ == '__main__':
    main()
