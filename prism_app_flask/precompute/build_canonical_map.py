#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_canonical_map.py
======================
Brain 인덱스에 gene → canonical isoform 매핑(canonical.json)을 심는다. triage ranked-list가
"각 isoform을 자기 유전자의 canonical(MANE/APPRIS)과 비교"하기 위한 anchor. 이전에는 앱이
isoform_profile.domain_architecture 에서 "max domain-span sibling"을 ref로 썼는데, 그건 구조적으로
loss-편향(교락)이라 proper canonical 로 교체하기 위함.

source: hMuscle/results_isoform/features/canonical_reference_brain.tsv
        (18,514 gene, 4-tier: MANE > Ensembl_canonical > APPRIS > longest_CDS; no_CDS 는 canonical 없음)

출력: prism_app_flask/data/isoform_index/brain/canonical.json
  { gene_name: {"id": canonical_iso_id, "row": <int index-row>, "source": "MANE"|... } }
  — canonical 이 이 인덱스에 실제로 존재하는(row 매핑 가능한) 유전자만 수록.

실행: conda activate isoform_env; python prism_app_flask/precompute/build_canonical_map.py
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
IDX = ROOT / 'prism_app_flask/data/isoform_index/brain'
TSV = ROOT / 'hMuscle/results_isoform/features/canonical_reference_brain.tsv'


def main():
    iso_ids = [str(x) for x in np.load(IDX / 'isoform_ids.npy', allow_pickle=True)]
    id_to_row = {i: r for r, i in enumerate(iso_ids)}

    out = {}
    n_rows, n_present, n_no_cds = 0, 0, 0
    with TSV.open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            n_rows += 1
            gene = row['gene_name']
            cid = row['canonical_iso_id']
            src = row.get('canonical_source', '')
            if src == 'no_CDS' or not cid or cid == 'NA':
                n_no_cds += 1
                continue
            r = id_to_row.get(cid)
            if r is None:
                continue  # canonical not detected in this brain index → no anchor
            out[gene] = {'id': cid, 'row': r, 'source': src}
            n_present += 1

    (IDX / 'canonical.json').write_text(json.dumps(out))
    print(f'canonical_reference rows: {n_rows} | no_CDS(skip): {n_no_cds} '
          f'| written (canonical present in index): {n_present} → canonical.json')


if __name__ == '__main__':
    main()
