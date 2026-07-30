#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adds helix_nterm / sheet_nterm / hydro_nterm to a1a2_isoform_covariates.npz -- same per-isoform
design as disorder_nterm (mean Chou-Fasman/Kyte-Doolittle over first-60 residues), NOT the pairwise
"Delta" framing Gemini's proposal used verbatim (that framing requires an explicit long/short pair,
which doesn't exist in this population -- same generalization issue already resolved for
nterm_deviates/disorder_nterm). Composition-only, no re-extraction needed (fast).
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
import re
import numpy as np

ROOT = '/home/welcome1/sw1686/DIFFUSE'
MODEL_DIR = f'{ROOT}/hMuscle/model'
DATA_DIR = f'{ROOT}/hMuscle/data'
COV_FILE = f'{ROOT}/reports/model_interpretability_map/a1a2_isoform_covariates.npz'
NTERM_WIN = 60
MAX_LEN = 1022
TYPE_RANK = {'complete': 4, '5prime_partial': 3, '3prime_partial': 2, 'internal': 1}

HELIX = {'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,'H':1.00,
         'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,'T':0.83,'W':1.08,
         'Y':0.69,'V':1.06}
SHEET = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,'H':0.87,
         'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,'T':1.19,'W':1.37,
         'Y':1.47,'V':1.70}
HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
         'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,
         'Y':-1.3,'V':4.2}


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


def parse_test_pep(pep_path, max_len=MAX_LEN):
    records = {}
    cur_id = cur_meta = None
    cur_seq = []

    def flush():
        nonlocal cur_id, cur_meta, cur_seq
        if cur_id is None: return
        seq = ''.join(cur_seq).replace('*', '').strip()
        if not seq: return
        rank, score, length = cur_meta
        prev = records.get(cur_id)
        if prev is None or (rank, score, length) > prev[:3]:
            records[cur_id] = (rank, score, length, seq)

    with open(pep_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush(); cur_seq = []
                m_id = re.match(r'>(\S+)', line)
                m_type = re.search(r'ORF type:(\S+)', line)
                m_score = re.search(r'score=([\d.]+)', line)
                m_len = re.search(r'len:(\d+)', line)
                if not m_id: cur_id = None; continue
                raw_id = m_id.group(1)
                cur_id = re.sub(r'\.p\d+$', '', raw_id)
                orf_type = m_type.group(1) if m_type else 'internal'
                score = float(m_score.group(1)) if m_score else 0.0
                length = int(m_len.group(1)) if m_len else 0
                rank = TYPE_RANK.get(orf_type.split('(')[0], 1)
                cur_meta = (rank, score, length)
            else:
                cur_seq.append(line)
    flush()
    return {k: v[3][:max_len] for k, v in records.items()}


te_iso_raw = np.load(f'{MODEL_DIR}/my_isoform_list_fixed.npy', allow_pickle=True)
te_iso_list = [clean(x) for x in te_iso_raw]
n_iso = len(te_iso_list)
seqs = parse_test_pep(f'{DATA_DIR}/top30k_isoforms.pep')

helix_nterm = np.zeros(n_iso, dtype=np.float32)
sheet_nterm = np.zeros(n_iso, dtype=np.float32)
hydro_nterm = np.zeros(n_iso, dtype=np.float32)
for i, iso_id in enumerate(te_iso_list):
    s = seqs.get(iso_id, '')
    if not s: continue
    w = s[:NTERM_WIN]
    helix_nterm[i] = np.mean([HELIX.get(a, 1.0) for a in w])
    sheet_nterm[i] = np.mean([SHEET.get(a, 1.0) for a in w])
    hydro_nterm[i] = np.mean([HYDRO.get(a, 0.0) for a in w])

existing = dict(np.load(COV_FILE))
existing.update(helix_nterm=helix_nterm, sheet_nterm=sheet_nterm, hydro_nterm=hydro_nterm)
np.savez(COV_FILE, **existing)
print(f'added helix_nterm/sheet_nterm/hydro_nterm -> {COV_FILE}')
print(f'  helix_nterm mean={helix_nterm[helix_nterm>0].mean():.3f}')
print(f'  sheet_nterm mean={sheet_nterm[sheet_nterm>0].mean():.3f}')
print(f'  hydro_nterm mean={hydro_nterm[hydro_nterm!=0].mean():.3f}')
