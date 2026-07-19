#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_axis_projection_muscle.py
==================================
Muscle counterpart of reports/v20b_pca_interp/Z_brain_Nx30x8.npy (which exists
for brain but not muscle -- the joint PCA basis W was muscle-TRAIN-fit but the
saved per-layer axis projections were only ever computed for brain, used for
BISECT/trajectory validation). Applies the same W (8x640) + per-layer z-score
(mu/sd) to muscle's 30-layer mean-pooled ESM-2 embeddings.

출력: reports/v20b_pca_interp/Z_muscle_Nx30x8.npy  (36748, 30, 8)
"""
from pathlib import Path
import numpy as np

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/v20b_pca_interp/Z_muscle_Nx30x8.npy'


def main():
    W = np.load(ROOT / 'reports/v20b_pca_interp/W_axes_8x640.npy')  # (8, 640)
    mu = np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_mu.npy')  # (30, 640)
    sd = np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_sd.npy')  # (30, 640)

    n_iso = np.load(ROOT / 'hMuscle/model/my_isoform_list_fixed.npy', allow_pickle=True).shape[0]
    Z = np.zeros((n_iso, 30, 8), dtype=np.float32)

    for layer in range(1, 31):
        emb = np.load(ROOT / f'hMuscle/data/esm2_layer_{layer:02d}_t30_150M.npy').astype(np.float32)
        z = (emb - mu[layer - 1]) / sd[layer - 1]
        Z[:, layer - 1, :] = z @ W.T
        print(f"  layer {layer:02d} done", flush=True)

    np.save(OUT, Z)
    print(f"[Save] {OUT} {Z.shape}")


if __name__ == '__main__':
    main()
