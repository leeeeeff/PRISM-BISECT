#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bisect_dtu.py
====================
BISECT 브레인 케이스(gene, cell_type)에 대해 실제 조건별(CT/AD) 아이소폼 사용률(usage
fraction) + 유전자수준 chi-square DTU p-value를 precompute한다.

기존 bisect_cases.json 은 큐레이션된 ct_transcript_id/ad_transcript_id 2개와 스칼라
delta 하나만 갖고 있어 "진짜 dominant isoform"(사용률 1등)을 알 수 없다 — 실제로는
큐레이션 쌍과 다를 수 있다(검증: ZCCHC17/Oligodendrocyte, 6개 이소폼 중
transcript54898.chr1.nnic 가 CT 85.6%→AD 55.0%로 지배적, 큐레이션 쌍(ZCCHC17-205/206)
과 다름 — 이건 버그가 아니라 진짜 생물학적 발견).

소스: /home/dhkim1674/Project_AD_with_refTSS_novel/06_DIU/DIU_by_condition_{cell_type}.csv
  columns: transcript_name, gene_name, AD, Control, chi_pval, chi_padj, chi_significant,
           delta_usage, usage_direction
  (AD/Control = 그 조건에서의 아이소폼 사용률, chi_pval/padj = 유전자수준 검정 — gene 내
  모든 행에 동일값)

cell_type 이름 매핑: bisect_cases.json 의 Excitatory/Inhibitory 는 CSV 파일명이
Excitatory_neuron/Inhibitory_neuron — 나머지는 동일.

dominant isoform 의 top-3 GO 는 기존 GO score matrix(brain universe, id_to_row)에서
isoform_profile._top_go() 를 그대로 재사용 — 새 계산 없음.

go_compare: CT-dominant top-3 ∪ AD-dominant top-3 GO(최대 6개, 중복 제거)에 대해, 두
아이소폼 모두의 실제 점수를 조회한다(어느 한쪽의 top-3 에만 들었다고 다른 쪽 점수를 0으로
채우지 않는다 — 전체 score row에서 해당 go_id 컬럼을 직접 읽는다, isoform_profile._top_go
와 동일 소스인 idx['scores']).

출력: prism_app_flask/data/isoform_index/bisect_dtu.json
  { "GENE_CELLTYPE": {
      "chi_pval": float, "chi_padj": float,
      "isoforms": [{"id","ct_frac","ad_frac","direction"}, ...],   # max(ct,ad) desc
      "dominant": {
        "ct": {"id","frac","top_go":[{go_id,go_name,score}x3] | null},
        "ad": {"id","frac","top_go":[...] | null},
        "go_compare": [{"go_id","name","ct_score","ad_score"}, ...]  # union of both top-3, both isoforms' real scores
      }
    }, ... }

실행: conda activate isoform_env; python prism_app_flask/precompute/build_bisect_dtu.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
sys.path.insert(0, str(ROOT))

CASES = ROOT / 'prism_app/data/demo/bisect_cases.json'
DIU_DIR = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/06_DIU')
OUT = ROOT / 'prism_app_flask/data/isoform_index/bisect_dtu.json'

CT_MAP = {
    'Excitatory': 'Excitatory_neuron',
    'Inhibitory': 'Inhibitory_neuron',
    'Astrocyte': 'Astrocyte',
    'Microglia': 'Microglia',
    'Oligodendrocyte': 'Oligodendrocyte',
    'OPC': 'OPC',
}


def load_diu(diu_cell_type):
    path = DIU_DIR / f'DIU_by_condition_{diu_cell_type}.csv'
    by_gene = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            by_gene.setdefault(row['gene_name'], []).append(row)
    return by_gene


def top_go_for(iso_id, idx, top_go_fn):
    row = idx['id_to_row'].get(iso_id)
    if row is None:
        return None
    return top_go_fn(row, idx, k=3)


def scores_for(iso_id, go_ids, idx, go_col):
    """iso_id 의 실제 score row 에서 go_ids 각각의 점수를 직접 조회 (top-k 제한 없음)."""
    row = idx['id_to_row'].get(iso_id)
    if row is None:
        return {g: None for g in go_ids}
    s = idx['scores'][row]
    return {g: (round(float(s[go_col[g]]), 4) if g in go_col else None) for g in go_ids}


def go_compare_for(dom_ct_id, dom_ad_id, ct_top, ad_top, idx, go_col):
    """CT-dominant top-3 ∪ AD-dominant top-3 GO(최대 6개)에 대해 두 아이소폼의 실제 점수를 나란히."""
    name_by_go = {}
    for g in (ct_top or []):
        name_by_go.setdefault(g['go_id'], g['name'])
    for g in (ad_top or []):
        name_by_go.setdefault(g['go_id'], g['name'])
    if not name_by_go:
        return []
    union_ids = list(name_by_go.keys())
    ct_scores = scores_for(dom_ct_id, union_ids, idx, go_col)
    ad_scores = scores_for(dom_ad_id, union_ids, idx, go_col)
    return [{'go_id': g, 'name': name_by_go[g], 'ct_score': ct_scores.get(g), 'ad_score': ad_scores.get(g)}
            for g in union_ids]


def main():
    from prism_app_flask.data_layer import loaders
    from prism_app_flask.data_layer.isoform_profile import _top_go

    cases = json.loads(CASES.read_text())
    idx = loaders.load_index('brain')
    go_col = {g: i for i, g in enumerate(idx['meta']['go_ids'])}

    diu_cache = {}
    out = {}
    n_covered, n_skipped = 0, 0

    for c in cases:
        gene, cell_type = c.get('gene'), c.get('cell_type')
        diu_ct = CT_MAP.get(cell_type)
        if diu_ct is None:
            n_skipped += 1
            continue
        if diu_ct not in diu_cache:
            diu_cache[diu_ct] = load_diu(diu_ct)
        rows = diu_cache[diu_ct].get(gene)
        if not rows:
            n_skipped += 1
            continue

        isoforms = []
        for r in rows:
            try:
                ad_frac = float(r['AD'])
                ct_frac = float(r['Control'])
            except (TypeError, ValueError):
                continue
            isoforms.append({
                'id': r['transcript_name'], 'ct_frac': round(ct_frac, 4),
                'ad_frac': round(ad_frac, 4), 'direction': r.get('usage_direction', ''),
            })
        if not isoforms:
            n_skipped += 1
            continue
        isoforms.sort(key=lambda x: max(x['ct_frac'], x['ad_frac']), reverse=True)

        dom_ct = max(isoforms, key=lambda x: x['ct_frac'])
        dom_ad = max(isoforms, key=lambda x: x['ad_frac'])

        def _pf(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        ct_top = top_go_for(dom_ct['id'], idx, _top_go)
        ad_top = top_go_for(dom_ad['id'], idx, _top_go)

        out[f'{gene}_{cell_type}'] = {
            'chi_pval': _pf(rows[0].get('chi_pval')),
            'chi_padj': _pf(rows[0].get('chi_padj')),
            'isoforms': isoforms,
            'dominant': {
                'ct': {'id': dom_ct['id'], 'frac': dom_ct['ct_frac'], 'top_go': ct_top},
                'ad': {'id': dom_ad['id'], 'frac': dom_ad['ad_frac'], 'top_go': ad_top},
                'go_compare': go_compare_for(dom_ct['id'], dom_ad['id'], ct_top, ad_top, idx, go_col),
            },
        }
        n_covered += 1

    OUT.write_text(json.dumps(out, indent=None))
    print(f'covered: {n_covered}, skipped: {n_skipped}, total cases: {len(cases)}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
