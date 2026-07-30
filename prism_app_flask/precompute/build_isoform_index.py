#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_isoform_index.py
======================
개별 유전자/아이소폼 분석(flagship)이 O(1)로 조회할 per-isoform 리치 인덱스를 생성한다.

설계 근거 (탐색으로 확정 — plan `curious-cuddling-adleman.md` 참조):
  - 브레인이 통일 좌표계다. 다음 3자가 모두 같은 63,994 순서로 정렬됨:
      * GO 예측 점수      prism_app/data/demo/brain_full_672_scores.npy   (63994, 672)
      * 8축 × 30레이어 PCA reports/v20b_pca_interp/Z_brain_Nx30x8.npy      (63994, 30, 8)
      * gene symbol / iso brain_full_672_gene_ids.npy / brain_full_672_ids.npy
  - Z_brain 은 검증된 projection 으로 이미 저장돼 있다(재계산 불필요):
      Z[:,L,:] == ((emb_L - layer_mu[L]) / layer_sd[L]) @ W_axes.T   (정확히 재현 확인)
  - covariate npz(a1a2, 36,748 fixed pop)는 muscle/fixed 좌표계 → 브레인엔 직접 안 붙음(secondary).

이 스크립트가 파생하는 것(작고 빠른 것만; 대용량 원본 Z/scores 는 런타임에 mmap 참조):
  axis_pos      (N, 8)   = Z.mean(axis=1)                 각 축에서 이 isoform 의 위치(30층 평균)
  axis_pos_pct  (N, 8)   = 축별 percentile rank           분포 대비 어디인지
  traj_mag      (N, 30)  = ||Z[:, L, :]||_2               레이어별 정보량 곡선(trajectory flow)
  peak_layer    (N,)     = argmax_L traj_mag + 1          정보 peak 레이어(1-indexed)
  gene_to_rows.json      symbol -> [row idx...]           유전자 검색 → 아이소폼 목록
  id_to_row.json         isoform id -> row idx            아이소폼 직접 검색
  meta.json              축 라벨/해석, GO id·name, 경로   패널 렌더 메타

실행:
  conda activate isoform_env
  python prism_app_flask/precompute/build_isoform_index.py
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
DEMO = ROOT / 'prism_app/data/demo'
PCA = ROOT / 'reports/v20b_pca_interp'
OUT = ROOT / 'prism_app_flask/data/isoform_index/brain'

# 축 라벨: axis_covariate_partial_corr.py 의 canonical AXIS_LABEL (muscle 에서 특성화).
# 브레인 축 정체성은 부분적으로 다를 수 있어 meta 에 tissue_caveat 를 함께 넣는다.
AXIS_LABEL = {
    0: 'axis0 · beta-sheet / TM (gene-level)',
    1: 'axis1 · LRR / Ig (weak)',
    2: 'axis2 · Pro-turn order (weak)',
    3: 'axis3 · domain content (KNOWN)',
    4: 'axis4 · helix-charge (weak)',
    5: 'axis5 · length (KNOWN)',
    6: 'axis6 · KRAB-ZNF / inverse-domain (weak)',
    7: 'axis7 · acidic-helical (weak)',
}


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """열별 percentile rank (0..100). 동점은 평균 순위."""
    out = np.empty_like(x, dtype=np.float32)
    n = x.shape[0]
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind='mergesort')
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n)
        out[:, j] = (ranks / max(n - 1, 1) * 100.0).astype(np.float32)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print('[1/5] loading brain arrays…')
    iso_ids = np.load(DEMO / 'brain_full_672_ids.npy', allow_pickle=True)
    gene_ids = np.load(DEMO / 'brain_full_672_gene_ids.npy', allow_pickle=True)
    Z = np.load(PCA / 'Z_brain_Nx30x8.npy')  # (N, 30, 8)
    N, n_layers, n_axes = Z.shape
    assert iso_ids.shape[0] == N == gene_ids.shape[0], (iso_ids.shape, N, gene_ids.shape)
    print(f'    N={N:,}  layers={n_layers}  axes={n_axes}')

    print('[2/5] deriving axis position + trajectory…')
    axis_pos = Z.mean(axis=1).astype(np.float32)              # (N, 8)
    axis_pos_pct = percentile_rank(axis_pos)                  # (N, 8)
    traj_mag = np.linalg.norm(Z, axis=2).astype(np.float32)   # (N, 30)
    peak_layer = (traj_mag.argmax(axis=1) + 1).astype(np.int16)  # 1-indexed

    # 레이어별 population 통계(axis별) — 3D normalized 궤적용(원본 normalize_layers 방식)
    layer_axis_mean = Z.mean(axis=0).astype(np.float32)       # (30, 8)
    layer_axis_std = (Z.std(axis=0) + 1e-6).astype(np.float32)  # (30, 8)

    # per-layer 판별력(Fisher-like): layer_depth_trajectory.tsv 의 domain AUROC(30 layer)
    layer_fisher = None
    ldt = ROOT / 'reports/model_interpretability_map/layer_depth_trajectory.tsv'
    if ldt.exists():
        import csv
        rows = list(csv.DictReader(ldt.open(), delimiter='\t'))
        layer_fisher = [round(float(r['auroc_domain']), 4) for r in rows]  # 30 values

    print('[3/5] building id/gene lookup…')
    iso_str = [str(x) for x in iso_ids]
    gene_str = [str(x) for x in gene_ids]
    id_to_row = {iso: i for i, iso in enumerate(iso_str)}
    gene_to_rows: dict[str, list[int]] = {}
    for i, g in enumerate(gene_str):
        gene_to_rows.setdefault(g, []).append(i)

    print('[4/5] loading GO term order/names…')
    go_json = json.loads((ROOT / 'hMuscle/data/brain672_go_terms.json').read_text())
    go_names = go_json.get('go_names', {})
    go_ids = list(go_names.keys())  # brain_full_672_scores.npy 컬럼 순서와 일치(문서화된 규약)
    assert len(go_ids) == 672, len(go_ids)

    # 축별 17-feature 상관(panel C tooltip 심화용)
    axes_bio = {}
    try:
        axes_bio = json.loads((PCA / 'axes_vs_biology.json').read_text())
    except Exception as e:  # noqa: BLE001
        print(f'    (axes_vs_biology.json 없음: {e})')

    print('[5/5] saving index →', OUT)
    np.save(OUT / 'axis_pos.npy', axis_pos)
    np.save(OUT / 'axis_pos_pct.npy', axis_pos_pct)
    np.save(OUT / 'traj_mag.npy', traj_mag)
    np.save(OUT / 'peak_layer.npy', peak_layer)
    np.save(OUT / 'layer_axis_mean.npy', layer_axis_mean)
    np.save(OUT / 'layer_axis_std.npy', layer_axis_std)
    (OUT / 'id_to_row.json').write_text(json.dumps(id_to_row))
    (OUT / 'gene_to_rows.json').write_text(json.dumps(gene_to_rows))
    np.save(OUT / 'isoform_ids.npy', np.asarray(iso_str, dtype=object), allow_pickle=True)
    np.save(OUT / 'gene_ids.npy', np.asarray(gene_str, dtype=object), allow_pickle=True)

    meta = {
        'universe': 'brain',
        'n_isoforms': int(N),
        'n_layers': int(n_layers),
        'n_axes': int(n_axes),
        'n_genes': len(gene_to_rows),
        'axis_labels': {str(k): v for k, v in AXIS_LABEL.items()},
        'tissue_caveat': (
            '축 정체성 라벨은 muscle severity-covariate 분석에서 특성화됨. '
            'axis3(domain)·axis5(length)·axis0(beta-sheet, gene-level)은 tissue 간 안정적, '
            '나머지 약축은 브레인에서 정체성이 다를 수 있음.'),
        'axes_vs_biology': axes_bio,
        'go_ids': go_ids,
        'go_names': go_names,
        'layer_fisher': layer_fisher,   # per-layer domain-discriminability AUROC (30)
        'layer_fisher_label': 'per-layer domain discriminability (AUROC)',
        # 런타임 mmap 참조 경로(중복 저장 회피)
        'ref_scores': str((DEMO / 'brain_full_672_scores.npy').resolve()),
        'ref_Z': str((PCA / 'Z_brain_Nx30x8.npy').resolve()),
    }
    (OUT / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    print('done. files:')
    for p in sorted(OUT.iterdir()):
        print(f'   {p.name:22s} {p.stat().st_size/1e6:8.2f} MB')


if __name__ == '__main__':
    main()
