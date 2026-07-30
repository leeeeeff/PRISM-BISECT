"""Lazy, memoized loaders for the per-isoform rich index.

인덱스 배열은 프로세스 최초 접근 시 1회 로드해 모듈 전역에 캐시한다.
대용량 원본(Z tensor, GO score matrix)은 mmap 으로 열어 RAM 상주를 피한다.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from prism_app_flask import config


@lru_cache(maxsize=4)
def _universe_dir(universe: str) -> Path:
    d = config.INDEX_DIR / universe
    if not d.exists():
        raise FileNotFoundError(
            f'isoform index for universe={universe!r} not found at {d}. '
            f'Run: python prism_app_flask/precompute/build_isoform_index.py')
    return d


@lru_cache(maxsize=4)
def load_index(universe: str = config.DEFAULT_UNIVERSE) -> dict:
    """universe 인덱스 전체를 로드(작은 배열은 RAM, 큰 원본은 mmap)."""
    d = _universe_dir(universe)
    meta = json.loads((d / 'meta.json').read_text())
    return {
        'meta': meta,
        'axis_pos': np.load(d / 'axis_pos.npy'),
        'axis_pos_pct': np.load(d / 'axis_pos_pct.npy'),
        'traj_mag': np.load(d / 'traj_mag.npy'),
        'peak_layer': np.load(d / 'peak_layer.npy'),
        'layer_axis_mean': np.load(d / 'layer_axis_mean.npy'),   # (30, 8)
        'layer_axis_std': np.load(d / 'layer_axis_std.npy'),     # (30, 8)
        'isoform_ids': np.load(d / 'isoform_ids.npy', allow_pickle=True),
        'gene_ids': np.load(d / 'gene_ids.npy', allow_pickle=True),
        'id_to_row': json.loads((d / 'id_to_row.json').read_text()),
        'gene_to_rows': json.loads((d / 'gene_to_rows.json').read_text()),
        # 대용량 원본 — mmap 참조
        'scores': np.load(meta['ref_scores'], mmap_mode='r'),   # (N, n_go)
        'Z': np.load(meta['ref_Z'], mmap_mode='r'),             # (N, 30, 8)
    }


def gene_symbols(universe: str = config.DEFAULT_UNIVERSE) -> list[str]:
    """검색 자동완성용 정렬된 유전자 심볼 목록."""
    return sorted(load_index(universe)['gene_to_rows'].keys())


_CODING_MASK_PATH = config.ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_mask.npy'
_CODING_IDS_PATH = config.ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy'


@lru_cache(maxsize=1)
def coding_status_by_id() -> dict | None:
    """isoform_id -> bool(coding). Aligned by ID string, not row position — this isoform-index
    universe and the brain_full_mask.npy file are built by separate precompute pipelines, so row
    order isn't guaranteed to match even though both cover the same 63,994-isoform universe.
    None if the mask files aren't available (mydata.js dashboard already established this is
    brain_672-only — [[finding-brain672-noncoding-zero-embedding-flag-needed]])."""
    if not (_CODING_MASK_PATH.exists() and _CODING_IDS_PATH.exists()):
        return None
    mask = np.load(_CODING_MASK_PATH).astype(bool)
    ids = np.load(_CODING_IDS_PATH, allow_pickle=True)
    ids = [x.decode() if isinstance(x, bytes) else str(x) for x in ids]
    return dict(zip(ids, mask.tolist()))


@lru_cache(maxsize=2)
def brain_domains(universe: str = config.DEFAULT_UNIVERSE) -> dict:
    """isoform id → [{name,start,end,evalue}] (build_brain_domains.py). 없으면 {}."""
    d = _universe_dir(universe)
    p = d / 'domains.json'
    return json.loads(p.read_text()) if p.exists() else {}


@lru_cache(maxsize=2)
def canonical_map(universe: str = config.DEFAULT_UNIVERSE) -> dict:
    """gene → {id, row, source} canonical isoform (build_canonical_map.py). 없으면 {}."""
    d = _universe_dir(universe)
    p = d / 'canonical.json'
    return json.loads(p.read_text()) if p.exists() else {}


@lru_cache(maxsize=2)
def brain_covariates(universe: str = config.DEFAULT_UNIVERSE) -> dict | None:
    """실제 서열 covariate (build_brain_covariates.py). 없으면 None."""
    d = _universe_dir(universe)
    p = d / 'covariates.npz'
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {'values': z['values'], 'pct': z['pct'], 'mask': z['mask'],
            'names': [str(x) for x in z['names']]}


@lru_cache(maxsize=2)
def go_layer_fisher(universe: str = config.DEFAULT_UNIVERSE) -> dict | None:
    """per-GO × per-layer Fisher discriminability (build_go_layer_fisher.py)."""
    d = _universe_dir(universe)
    p = d / 'go_layer_fisher.npy'
    if not p.exists():
        return None
    mat = np.load(p)                                        # (n_go, 30)
    go_ids = json.loads((d / 'go_layer_fisher_ids.json').read_text())
    return {'mat': mat, 'go_row': {g: i for i, g in enumerate(go_ids)}}
