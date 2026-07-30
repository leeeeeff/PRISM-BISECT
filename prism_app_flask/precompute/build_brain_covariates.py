#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_brain_covariates.py
=========================
brain universe(63,994 A1BG-204) 아이소폼의 실제 서열 covariate를 계산.

서열 crosswalk (옵션 B — 실제 소싱):
  A1BG-204 (brain_full_672_ids) --[brain_only.gtf: transcript_name↔ENST]--> ENST
    --[brain_only_transcripts.fa.transdecoder.pep]--> protein sequence
커버리지 ≈ 46.5% (29,786/63,994) — 나머지는 non-coding/novel(단백질 ORF 없음) → NaN.

covariate (a1a2 fixed-pop npz 스타일):
  disorder_frac : metapredict v3 per-residue > 0.5 비율 (full length)
  helix_nterm   : Chou-Fasman P(α) 평균, N-말단 60 aa
  sheet_nterm   : Chou-Fasman P(β) 평균, N-말단 60 aa
  hydro_nterm   : Kyte-Doolittle hydropathy 평균, N-말단 60 aa

출력: prism_app_flask/data/isoform_index/brain/covariates.npz
  values (N,4) float32 (NaN=서열없음) · names[4] · mask(N,) bool · pct(N,4)

실행: conda activate isoform_env; python prism_app_flask/precompute/build_brain_covariates.py
"""
import re
from pathlib import Path

import numpy as np

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
IDX = ROOT / 'prism_app_flask/data/isoform_index/brain'
GTF = ROOT / 'hMuscle/data/brain_esm2/brain_only.gtf'
PEP = ROOT / 'hMuscle/data/brain_esm2/brain_only_transcripts.fa.transdecoder.pep'
NTERM = 60

KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
      'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
PA = {'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,'H':1.00,'I':1.08,
      'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,'T':0.83,'W':1.08,'Y':0.69,'V':1.06}
PB = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,'H':0.87,'I':1.60,
      'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,'T':1.19,'W':1.37,'Y':1.47,'V':1.70}


def mean_scale(seq, scale):
    vals = [scale[a] for a in seq if a in scale]
    return float(np.mean(vals)) if vals else np.nan


def pct_rank(col):
    out = np.full(col.shape, np.nan, np.float32)
    m = ~np.isnan(col)
    if m.sum() < 2:
        return out
    v = col[m]
    order = np.argsort(v, kind='mergesort')
    r = np.empty(len(v)); r[order] = np.arange(len(v))
    out[m] = (r / (len(v) - 1) * 100).astype(np.float32)
    return out


def main():
    import metapredict as meta
    bid = [str(x) for x in np.load(ROOT /
           'prism_app/data/demo/brain_full_672_ids.npy', allow_pickle=True)]
    N = len(bid)

    print('[1/4] name→ENST from gtf…')
    name2enst = {}
    for line in GTF.open():
        if '\ttranscript\t' not in line:
            continue
        mn = re.search(r'transcript_name "([^"]+)"', line)
        me = re.search(r'transcript_id "([^"]+)"', line)
        if mn and me:
            name2enst[mn.group(1)] = me.group(1)

    print('[2/4] ENST→seq from transdecoder pep…')
    enst_seq = {}
    cur, buf = None, []
    def flush():
        nonlocal cur, buf
        if cur and buf:
            s = ''.join(buf).replace('*', '')
            if cur not in enst_seq or len(s) > len(enst_seq[cur]):
                enst_seq[cur] = s
        buf = []
    for line in PEP.open():
        if line.startswith('>'):
            flush(); m = re.match(r'>(\S+)', line)
            cur = re.sub(r'\.p\d+$', '', m.group(1)) if m else None
        else:
            buf.append(line.strip())
    flush()

    print('[3/4] computing covariates (metapredict + composition)…')
    vals = np.full((N, 4), np.nan, np.float32)   # disorder_frac, helix, sheet, hydro
    covered = 0
    for i, b in enumerate(bid):
        e = name2enst.get(b)
        seq = enst_seq.get(e) if e else None
        if not seq:
            continue
        nt = seq[:NTERM]
        try:
            dis = meta.predict_disorder(seq)
            disfrac = float(np.mean(np.asarray(dis) > 0.5))
        except Exception:
            disfrac = np.nan
        vals[i] = [disfrac, mean_scale(nt, PA), mean_scale(nt, PB), mean_scale(nt, KD)]
        covered += 1
        if covered % 3000 == 0:
            print(f'    {covered} computed (at idx {i})…')

    print('[4/4] percentiles + save…')
    pct = np.stack([pct_rank(vals[:, j]) for j in range(4)], axis=1).astype(np.float32)
    mask = ~np.isnan(vals[:, 0])
    np.savez(IDX / 'covariates.npz',
             values=vals, pct=pct, mask=mask,
             names=np.array(['disorder_frac', 'helix_nterm', 'sheet_nterm', 'hydro_nterm']))
    print(f'done. covered {covered}/{N} ({100*covered/N:.1f}%) → {IDX/"covariates.npz"}')


if __name__ == '__main__':
    main()
