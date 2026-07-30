#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_go_layer_fisher.py
========================
per-GO × per-layer Fisher discriminability. 각 GO term g, 각 레이어 L 에서
"GO-양성 vs 음성 아이소폼을 표현이 얼마나 분리하는가"를 Fisher LDA(8축 Z[:,L,:])
방향 투영의 AUROC 로 측정한다.

라벨: gene-level GO 주석(human_annotations_unified_bp) → 아이소폼 gene-inherited.
  (⚠️ gene-level 이므로 판별력은 서술용 참조이지 배포 성능 주장이 아님 — UI 에 명시)

출력: prism_app_flask/data/isoform_index/brain/go_layer_fisher.npy  (672, 30) float32
      go_layer_fisher_ids.json  (컬럼=go_ids 순서, meta.go_ids 와 동일)

실행: conda activate isoform_env; python prism_app_flask/precompute/build_go_layer_fisher.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
IDX = ROOT / 'prism_app_flask/data/isoform_index/brain'
MIN_POS = 20


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """rank 기반 AUROC (Mann-Whitney U)."""
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    Z = np.load(ROOT / 'reports/v20b_pca_interp/Z_brain_Nx30x8.npy')  # (N,30,8)
    N, nL, nA = Z.shape
    genes = [str(g) for g in np.load(IDX / 'gene_ids.npy', allow_pickle=True)]
    meta = json.loads((IDX / 'meta.json').read_text())
    go_ids = meta['go_ids']

    # gene → GO set
    ann = {}
    for line in (ROOT / 'prism_app/data/annotations/human_annotations_unified_bp.txt').read_text().splitlines():
        p = line.split('\t')
        if len(p) >= 2:
            ann[p[0]] = set(p[1:])
    gene_go = [ann.get(g, set()) for g in genes]

    out = np.full((len(go_ids), nL), np.nan, dtype=np.float32)
    for gi, gid in enumerate(go_ids):
        y = np.fromiter((1 if gid in gg else 0 for gg in gene_go), dtype=np.int8, count=N)
        if y.sum() < MIN_POS or (N - y.sum()) < MIN_POS:
            continue
        pos, neg = (y == 1), (y == 0)
        for L in range(nL):
            X = Z[:, L, :].astype(np.float64)          # (N,8)
            mu1, mu0 = X[pos].mean(0), X[neg].mean(0)
            Sw = np.cov(X[pos].T) * (pos.sum() - 1) + np.cov(X[neg].T) * (neg.sum() - 1)
            Sw += np.eye(nA) * 1e-3                     # ridge for stability
            try:
                w = np.linalg.solve(Sw, mu1 - mu0)
            except np.linalg.LinAlgError:
                continue
            out[gi, L] = auroc(y, X @ w)
        if (gi + 1) % 50 == 0:
            print(f'  {gi+1}/{len(go_ids)} …')

    np.save(IDX / 'go_layer_fisher.npy', out)
    (IDX / 'go_layer_fisher_ids.json').write_text(json.dumps(go_ids))
    valid = int((~np.isnan(out[:, 0])).sum())
    print(f'done. ({valid}/{len(go_ids)} GO computed) → {IDX/"go_layer_fisher.npy"} '
          f'({out.nbytes/1e3:.0f} KB)')


if __name__ == '__main__':
    main()
