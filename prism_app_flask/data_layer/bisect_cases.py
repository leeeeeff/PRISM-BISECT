"""BISECT 케이스 로더 — bisect_cases.json 을 로드/요약.

101 케이스, 필드 풍부(tier, mechanism, CT/AD transcript, plDDT, PPI, conservation,
PRISM top-GO). 개별분석 flagship 과 ct/ad transcript_id 로 교차링크된다.
"""
from __future__ import annotations

import json
from functools import lru_cache

from prism_app_flask import config

CASES_PATH = config.DEMO_DIR / 'bisect_cases.json'
TRACKS_PATH = config.INDEX_DIR / 'bisect_tracks.json'
DTU_PATH = config.INDEX_DIR / 'bisect_dtu.json'

TIER_META = {
    'A-DR': ('Tier A · domain-resolved', '#FF5C1A'),
    'A-BP': ('Tier A · biological-process', '#FFB020'),
    'B':    ('Tier B', '#35C6E8'),
    'C':    ('Tier C', '#66728A'),
    'D':    ('Tier D', '#3a4356'),
}

# mechanism_type(전사체 생성 기전) → 표시 라벨. streamlit 00_hub.py 'Cases by mechanism × cell type' 이관.
# m16_mechanism(도메인 손실/획득 등 서열결과 분류)과는 다른 축 — summary()['mechanisms'] 는 그쪽.
MECHANISM_TYPE_LABELS = {
    'alternative_promoter':    'Alt. Promoter',
    'transcriptional':         'Transcriptional',
    'epigenetic_derepression': 'Epigenetic',
    'alternative_splicing':    'Alt. Splicing',
}


@lru_cache(maxsize=1)
def all_cases() -> list:
    if not CASES_PATH.exists():
        return []
    return json.loads(CASES_PATH.read_text())


@lru_cache(maxsize=1)
def _tracks() -> dict:
    """CT/AD disorder(IDR) + domain 아키텍처 트랙 (build_bisect_tracks.py)."""
    if not TRACKS_PATH.exists():
        return {}
    return json.loads(TRACKS_PATH.read_text())


@lru_cache(maxsize=1)
def _dtu() -> dict:
    """조건별 실제 아이소폼 사용률 + chi-square DTU p-value (build_bisect_dtu.py).
    brain 세포유형 케이스만 존재(muscle/누락 유전자는 키 자체가 없음 — detail()에서 None)."""
    if not DTU_PATH.exists():
        return {}
    return json.loads(DTU_PATH.read_text())


def parse_regulators(raw) -> list:
    """top_regulators 문자열(';'-join python-dict-repr) → SF/TF volcano 용 리스트."""
    import ast
    out = []
    if not raw or str(raw) in ('None', ''):
        return out
    for p in str(raw).split(';'):
        p = p.strip()
        if not p:
            continue
        try:
            d = ast.literal_eval(p)
            out.append({
                'gene': d.get('gene', ''),
                'logFC': float(d.get('logFC', 0)),
                'neg_log10_padj': float(d.get('neg_log10_padj', 0)),
                'direction': str(d.get('direction', '')).lower(),
            })
        except Exception:  # noqa: BLE001
            pass
    return out


@lru_cache(maxsize=1)
def summary() -> dict:
    cases = all_cases()
    def counts(field):
        c: dict = {}
        for x in cases:
            k = str(x.get(field) or '—')
            c[k] = c.get(k, 0) + 1
        return dict(sorted(c.items(), key=lambda kv: -kv[1]))
    # mechanism_type × cell_type 교차표 — long-format rows(클라이언트에서 stacked bar 로 pivot).
    mech_cell: dict = {}
    for x in cases:
        mech = MECHANISM_TYPE_LABELS.get(str(x.get('mechanism_type') or ''), str(x.get('mechanism_type') or '—'))
        ct = str(x.get('cell_type') or '—')
        mech_cell.setdefault(mech, {})
        mech_cell[mech][ct] = mech_cell[mech].get(ct, 0) + 1
    mech_cell_rows = [
        {'mechanism': mech, 'cell_type': ct, 'n': n}
        for mech, by_ct in mech_cell.items() for ct, n in by_ct.items()
    ]

    return {
        'n': len(cases),
        'tiers': counts('bisect_tier'),
        'cell_types': counts('cell_type'),
        'mechanisms': counts('m16_mechanism'),
        'tier_meta': {k: {'label': v[0], 'color': v[1]} for k, v in TIER_META.items()},
        'mech_cell_rows': mech_cell_rows,
    }


def _compact(c: dict) -> dict:
    """테이블 행에 필요한 필드만."""
    return {
        'gene': c.get('gene'), 'cell_type': c.get('cell_type'),
        'tier': c.get('bisect_tier'), 'delta': c.get('delta'),
        'mechanism': c.get('m16_mechanism'), 'event': c.get('m14_event_type'),
        'domains_lost': c.get('domains_lost') or '', 'domains_gained': c.get('domains_gained') or '',
        'delta_plddt': c.get('af_delta_plddt'), 'ppi': c.get('ppi_verdict'),
        'cons': c.get('cons_ad_class') or '', 'nmd': bool(c.get('m15_nmd_switch')),
        'ct_transcript_id': c.get('ct_transcript_id'), 'ad_transcript_id': c.get('ad_transcript_id'),
    }


def table() -> list:
    return [_compact(c) for c in all_cases()]


def detail(gene: str, cell_type: str | None = None) -> dict:
    from prism_app_flask.data_layer import bio_report as br
    for c in all_cases():
        if c.get('gene') == gene and (cell_type is None or c.get('cell_type') == cell_type):
            out = dict(c)
            out['regulators'] = parse_regulators(c.get('top_regulators'))
            out['tracks'] = _tracks().get(f"{c.get('gene')}_{c.get('cell_type')}", {})
            out['dtu'] = _dtu().get(f"{c.get('gene')}_{c.get('cell_type')}")
            out['bio_report'] = br.build_report(out, out['regulators'])
            return out
    return {'error': f'case {gene!r} not found'}
