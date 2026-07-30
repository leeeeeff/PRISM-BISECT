#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_bisect_tracks.py
======================
BISECT 케이스별 CT/AD 서열의 IDR(disorder) + 도메인 아키텍처 트랙을 precompute.

소스: Final_analysis/pipeline_bioanalysis/outputs/{gene}_{cell_type}/analysis.json
  - ct_seq / ad_seq          (단백질 서열)
  - ct_domains / ad_domains   (Pfam 도메인 + ali_from/ali_to 위치)
disorder: metapredict v3 (per-residue), 계측 schematic 용으로 라운딩 저장.

출력: prism_app_flask/data/isoform_index/bisect_tracks.json
  { "GENE_CELLTYPE": { "ct": {len, disorder:[...], domains:[{name,start,end}]},
                       "ad": {...} }, ... }

실행: conda activate isoform_env; python prism_app_flask/precompute/build_bisect_tracks.py
"""
import json
from pathlib import Path

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
CASES = ROOT / 'prism_app/data/demo/bisect_cases.json'
OUTBASE = ROOT / 'Final_analysis/pipeline_bioanalysis/outputs'
OUT = ROOT / 'prism_app_flask/data/isoform_index/bisect_tracks.json'


def domain_track(dom_list):
    out = []
    for d in dom_list or []:
        out.append({'name': d.get('domain') or d.get('pfam_family') or '?',
                    'start': int(d.get('ali_from', 0)), 'end': int(d.get('ali_to', 0)),
                    'evalue': d.get('evalue')})
    return out


def main():
    import metapredict as meta
    cases = json.loads(CASES.read_text())
    tracks = {}
    n_ok = 0
    for i, c in enumerate(cases):
        gene, ct = c.get('gene'), c.get('cell_type')
        aj = OUTBASE / f'{gene}_{ct}' / 'analysis.json'
        if not aj.exists():
            continue
        d = json.loads(aj.read_text())
        entry = {}
        for side in ('ct', 'ad'):
            seq = d.get(f'{side}_seq')
            if isinstance(seq, dict):        # {'seq': '...', 'length': N, 'source': ...}
                seq = seq.get('seq') or ''
            if not seq or not isinstance(seq, str):
                continue
            try:
                dis = meta.predict_disorder(seq)
                dis = [round(float(x), 3) for x in dis]
            except Exception as e:  # noqa: BLE001
                dis = []
                print(f'  ! {gene}_{ct} {side} disorder fail: {e}')
            entry[side] = {'len': len(seq), 'disorder': dis,
                           'domains': domain_track(d.get(f'{side}_domains'))}
        if entry:
            tracks[f'{gene}_{ct}'] = entry
            n_ok += 1
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(cases)} …')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tracks))
    print(f'done. {n_ok} cases → {OUT} ({OUT.stat().st_size/1e6:.2f} MB)')


if __name__ == '__main__':
    main()
