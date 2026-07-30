"""44개 기능 모듈 참조 지형도 — brain-672 전용, 데이터셋(=tissue 선택)과 무관한 **정적 참조**.

streamlit `02_landscape.py` Tab5 "참조 지형도"(버블 차트 + GO-GO 상관 행렬) 이관.
소스: reports/brain_go_modules_672.json · brain_full_672_meta.json · brain_isoform_modules.tsv ·
      brain_go_corr_672.npy(demo 사본 우선) — 전부 사전계산돼 있어 요청마다 재계산 불필요, lru_cache 로 1회 로드.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from prism_app_flask import config

REPORTS = config.ROOT / 'reports'
DEMO = config.DEMO_DIR

# 모듈 id → 상위 카테고리 (02_landscape.py `_get_cat` 이관, 버블 차트 색 그룹)
_CATEGORY_BY_MODULE = {
    36: 'Neuronal development', 37: 'Neuronal development',
    13: 'Synaptic / GPCR', 14: 'Synaptic / GPCR',
    11: 'Ion transport', 12: 'Ion transport',
    35: 'Cell adhesion',
    23: 'Immune', 24: 'Immune', 25: 'Immune',
    4: 'Cell cycle', 34: 'Cell cycle',
    1: 'Transcription / DNA', 2: 'Transcription / DNA',
    10: 'RNA processing', 8: 'RNA processing',
}
CATEGORY_COLORS = {
    'Neuronal development': '#1565C0', 'Synaptic / GPCR': '#0097A7',
    'Ion transport': '#00897B', 'Cell adhesion': '#43A047',
    'Immune': '#E53935', 'Cell cycle': '#F57F17',
    'Transcription / DNA': '#6A1B9A', 'RNA processing': '#AD1457',
    'General': '#9E9E9E',
}


def _category(mid: int) -> str:
    return _CATEGORY_BY_MODULE.get(mid, 'General')


@lru_cache(maxsize=1)
def _load_refs():
    mod_p, meta_p, iso_p = (REPORTS / 'brain_go_modules_672.json',
                             REPORTS / 'brain_full_672_meta.json',
                             REPORTS / 'brain_isoform_modules.tsv')
    if not (mod_p.exists() and meta_p.exists() and iso_p.exists()):
        return None
    mod_data = json.loads(mod_p.read_text())
    meta = json.loads(meta_p.read_text())
    df_iso = pd.read_csv(iso_p, sep='\t')
    return {'modules': mod_data['modules'], 'meta': meta, 'df_iso': df_iso}


@lru_cache(maxsize=1)
def _load_corr() -> np.ndarray | None:
    for p in (DEMO / 'brain_go_corr_672.npy', REPORTS / 'brain_go_corr_672.npy'):
        if p.exists():
            return np.load(p).astype(np.float32)
    return None


@lru_cache(maxsize=1)
def bubble_data() -> list[dict]:
    """44 모듈 버블 차트: X=평균 brain AUPRC(예측 품질) · Y=Novel(NIC+NNIC) 농축비 · size=고신뢰 아이소폼 수."""
    refs = _load_refs()
    if refs is None:
        return []
    modules, meta, df_iso = refs['modules'], refs['meta'], refs['df_iso']
    per_go = {p['go']: p for p in meta['per_go']}

    df_hi = df_iso[df_iso['module_score'] > 0.3]
    novel_types = {'nic', 'nnic'}
    bg = len(df_hi[df_hi['type'].isin(novel_types)]) / max(len(df_hi), 1)

    rows = []
    for mid_str, minfo in modules.items():
        mid = int(mid_str)
        auprc_v = [per_go[g]['auprc_brain'] for g in minfo['go_ids']
                   if g in per_go and per_go[g].get('auprc_brain')]
        if not auprc_v:
            continue
        sub = df_hi[df_hi['primary_module'] == mid]
        if len(sub) < 10:
            continue
        nov_f = len(sub[sub['type'].isin(novel_types)]) / len(sub)
        cat = _category(mid)
        rows.append({
            'module_id': mid,
            'label': f"M{mid}: {minfo['label'].split('/')[0].strip()[:28]}",
            'mean_auprc': round(float(np.mean(auprc_v)), 4),
            'enrichment': round((nov_f - bg) / max(bg, 1e-6), 4),
            'n_isoforms': int(len(sub)),
            'novel_pct': round(100 * nov_f, 2),
            'category': cat,
            'color': CATEGORY_COLORS.get(cat, CATEGORY_COLORS['General']),
        })
    return sorted(rows, key=lambda r: -r['mean_auprc'])


@lru_cache(maxsize=1)
def bubble_meta() -> dict:
    refs = _load_refs()
    if refs is None:
        return {'background_novel_frac': 0.0}
    df_hi = refs['df_iso'][refs['df_iso']['module_score'] > 0.3]
    novel_types = {'nic', 'nnic'}
    bg = len(df_hi[df_hi['type'].isin(novel_types)]) / max(len(df_hi), 1)
    return {'background_novel_frac': round(100 * bg, 2)}


@lru_cache(maxsize=1)
def corr_matrix(downsample: int = 220) -> dict | None:
    """44 모듈 경계로 정렬된 GO-GO Pearson 상관 행렬 — 렌더 속도를 위해 다운샘플."""
    refs = _load_refs()
    R = _load_corr()
    if refs is None or R is None:
        return None
    go_ids_list = refs['meta']['go_ids']
    modules = refs['modules']
    mod_ids_sorted = sorted(modules.keys(), key=int)
    go_idx = {g: i for i, g in enumerate(go_ids_list)}

    sorted_go_idx, mod_boundaries = [], []
    for mid_str in mod_ids_sorted:
        idx = [go_idx[g] for g in modules[mid_str]['go_ids'] if g in go_idx]
        sorted_go_idx.extend(idx)
        mod_boundaries.append(len(sorted_go_idx))

    R_sorted = R[np.ix_(sorted_go_idx, sorted_go_idx)]
    n = len(sorted_go_idx)
    step = max(1, n // downsample)
    R_ds = R_sorted[::step, ::step]
    bounds_ds = [b // step for b in mod_boundaries if b // step < R_ds.shape[0]]

    return {
        'z': np.round(R_ds, 3).tolist(),
        'n_go': n,
        'n_shown': int(R_ds.shape[0]),
        'module_boundaries': bounds_ds,
    }
