#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_brain_domains.py
======================
brain 아이소폼별 Pfam 도메인 아키텍처(이름 + 위치).

crosswalk: A1BG-204 →[brain_only.gtf name↔ENST]→ ENST →[hmmscan_brain.domtblout]→ domains.
커버리지 ≈ 35.9%. + charge/aromatic covariate 확장(E1, cheap)도 여기서 함께.

출력:
  prism_app_flask/data/isoform_index/brain/domains.json   { A1BG-204: [{name,start,end,evalue}] }
  covariates.npz 에 charge_nterm, aromatic_nterm 2열 추가(재저장, disorder 등 기존 유지).

실행: conda activate isoform_env; python prism_app_flask/precompute/build_brain_domains.py
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
IDX = ROOT / 'prism_app_flask/data/isoform_index/brain'
GTF = ROOT / 'hMuscle/data/brain_esm2/brain_only.gtf'
PEP = ROOT / 'hMuscle/data/brain_esm2/brain_only_transcripts.fa.transdecoder.pep'
DOMTBL = ROOT / 'hMuscle/results_isoform/features/hmmscan_brain.domtblout'
NTERM = 60

CHARGE = {'D': -1, 'E': -1, 'K': +1, 'R': +1, 'H': +0.1}
AROM = set('FWY')


def name2enst():
    m = {}
    for line in GTF.open():
        if '\ttranscript\t' not in line:
            continue
        mn = re.search(r'transcript_name "([^"]+)"', line)
        me = re.search(r'transcript_id "([^"]+)"', line)
        if mn and me:
            m[mn.group(1)] = me.group(1)
    return m


def enst_seq():
    seqs, cur, buf = {}, None, []
    def flush():
        nonlocal cur, buf
        if cur and buf:
            s = ''.join(buf).replace('*', '')
            if cur not in seqs or len(s) > len(seqs[cur]):
                seqs[cur] = s
        buf = []
    for line in PEP.open():
        if line.startswith('>'):
            flush(); mm = re.match(r'>(\S+)', line)
            cur = re.sub(r'\.p\d+$', '', mm.group(1)) if mm else None
        else:
            buf.append(line.strip())
    flush()
    return seqs


def main():
    bid = [str(x) for x in np.load(ROOT / 'prism_app/data/demo/brain_full_672_ids.npy', allow_pickle=True)]
    n2e = name2enst()
    print('gtf name→enst', len(n2e))

    # domtblout → ENST → domains
    print('parsing domtblout…')
    enst_dom = {}
    for line in DOMTBL.open():
        if line.startswith('#'):
            continue
        p = line.split()
        if len(p) < 19:
            continue
        q = re.sub(r'\.p\d+$', '', p[3])
        try:
            ie = float(p[12]); af = int(p[17]); at = int(p[18])
        except ValueError:
            continue
        if ie > 1e-3:            # i-Evalue 유의성 컷
            continue
        enst_dom.setdefault(q, []).append({'name': p[0], 'start': af, 'end': at, 'evalue': ie})

    # A1BG-204 → domains (merge overlapping same-family hits)
    domains = {}
    covd = 0
    for b in bid:
        e = n2e.get(b)
        ds = enst_dom.get(e)
        if not ds:
            continue
        ds = sorted(ds, key=lambda d: d['start'])
        merged = []
        for d in ds:
            if merged and d['name'] == merged[-1]['name'] and d['start'] <= merged[-1]['end'] + 10:
                merged[-1]['end'] = max(merged[-1]['end'], d['end'])
                merged[-1]['evalue'] = min(merged[-1]['evalue'], d['evalue'])
            else:
                merged.append(dict(d))
        domains[b] = [{'name': d['name'], 'start': d['start'], 'end': d['end'],
                       'evalue': float(f"{d['evalue']:.1e}")} for d in merged]
        covd += 1
    (IDX / 'domains.json').write_text(json.dumps(domains))
    print(f'domains: {covd}/{len(bid)} ({100*covd/len(bid):.1f}%) → domains.json')

    # E1 — charge/aromatic covariate 확장 (기존 npz 유지 + 2열 추가)
    print('extending covariates (charge, aromatic)…')
    seqs = enst_seq()
    N = len(bid)
    charge = np.full(N, np.nan, np.float32)
    arom = np.full(N, np.nan, np.float32)
    for i, b in enumerate(bid):
        e = n2e.get(b); s = seqs.get(e) if e else None
        if not s:
            continue
        nt = s[:NTERM]
        if nt:
            charge[i] = sum(CHARGE.get(a, 0) for a in nt) / len(nt)
            arom[i] = sum(1 for a in nt if a in AROM) / len(nt)

    old = np.load(IDX / 'covariates.npz', allow_pickle=True)
    vals = old['values']; names = list(old['names'])
    new_vals = np.concatenate([vals, charge[:, None], arom[:, None]], axis=1)
    names += ['charge_nterm', 'aromatic_nterm']

    def pct_rank(col):
        out = np.full(col.shape, np.nan, np.float32); m = ~np.isnan(col)
        if m.sum() < 2:
            return out
        v = col[m]; order = np.argsort(v, kind='mergesort'); r = np.empty(len(v)); r[order] = np.arange(len(v))
        out[m] = (r / (len(v) - 1) * 100).astype(np.float32)
        return out
    new_pct = np.stack([pct_rank(new_vals[:, j]) for j in range(new_vals.shape[1])], axis=1).astype(np.float32)
    mask = ~np.isnan(new_vals[:, 0])
    np.savez(IDX / 'covariates.npz', values=new_vals, pct=new_pct, mask=mask, names=np.array(names))
    print(f'covariates now {new_vals.shape[1]} cols: {names}')


if __name__ == '__main__':
    main()
