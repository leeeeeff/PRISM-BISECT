#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_isoform_level_covariates_a1a2pop.py

Generalizes domain_ranking_validation.py's within-gene, gene-mean-immune ranking-AUC design
(currently ONLY tested against Pfam domain count) to two NEW per-isoform covariates, computed
natively on the A1/A2 evaluation population (my_gene_list_fixed.npy / my_isoform_list_fixed.npy,
36,748 isoforms) -- NOT reused from severity_pairs_scored.tsv, which turned out to be a DIFFERENT
isoform catalog (brain/muscle isoquant assembly, e.g. 'A1BG-204') with ZERO id overlap with this
population (Bambu-assembled ids, e.g. 'BambuTx10'). Same covariate INTENT as severity_pairs
(nterm_overlap, disorder_frac), redefined as true per-isoform scalars (not pair/edit-region
concepts, which don't generalize to PRISM's standalone per-isoform scoring) so they slot into
compute_domain_ranking_auc's exact median-split-within-gene design unchanged:

  nterm_deviates : binary, 1 if this isoform's first-60aa differs from the gene's most common
                   (mode) first-60aa among its own test-set isoforms (0 if only 1 unique N-term
                   in the gene, i.e. no isoform-level N-terminal variation to rank).
  disorder_nterm : continuous, mean metapredict disorder score over the first 60 residues.

Output: n_domains, nterm_deviates, disorder_nterm arrays (36748,) aligned to my_isoform_list_fixed
        index order, saved to reports/model_interpretability_map/a1a2_isoform_covariates.npz
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
import re
import sys
from collections import defaultdict, Counter

import numpy as np

ROOT = '/home/welcome1/sw1686/DIFFUSE'
MODEL_DIR = f'{ROOT}/hMuscle/model'
DATA_DIR = f'{ROOT}/hMuscle/data'
ID_DIR = f'{DATA_DIR}/raw_data/data/id_lists'
OUT = f'{ROOT}/reports/model_interpretability_map/a1a2_isoform_covariates.npz'
NTERM_WIN = 60
MAX_LEN = 1022
TYPE_RANK = {'complete': 4, '5prime_partial': 3, '3prime_partial': 2, 'internal': 1}


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']:
        s = s.replace(c, '')
    return s


def parse_test_pep(pep_path, max_len=MAX_LEN):
    records = {}
    cur_id = cur_meta = None
    cur_seq = []

    def flush():
        nonlocal cur_id, cur_meta, cur_seq
        if cur_id is None:
            return
        seq = ''.join(cur_seq).replace('*', '').strip()
        if not seq:
            return
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
                if not m_id:
                    cur_id = None; continue
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


def main():
    te_genes_raw = np.load(f'{MODEL_DIR}/my_gene_list_fixed.npy', allow_pickle=True)
    ENSG2SYM = {}
    with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 5:
                ENSG2SYM[p[0]] = p[4]
    te_sym_list = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in te_genes_raw]

    te_iso_raw = np.load(f'{MODEL_DIR}/my_isoform_list_fixed.npy', allow_pickle=True)
    te_iso_list = [clean(x) for x in te_iso_raw]
    n_iso = len(te_iso_list)
    print(f'n_iso={n_iso}  n_genes={len(set(te_sym_list))}')

    gene2idxs = defaultdict(list)
    for i, g in enumerate(te_sym_list):
        gene2idxs[g].append(i)

    print('Parsing sequences...')
    seqs = parse_test_pep(f'{DATA_DIR}/top30k_isoforms.pep')
    seq_by_idx = [seqs.get(iso_id, '') for iso_id in te_iso_list]
    resolved = sum(1 for s in seq_by_idx if s)
    print(f'  resolved {resolved}/{n_iso}')

    # ---- nterm_deviates: within-gene mode-based N-terminal deviation ----
    print('Computing nterm_deviates (within-gene mode of first-60aa)...')
    nterm_deviates = np.zeros(n_iso, dtype=np.float32)
    for g, idxs in gene2idxs.items():
        if len(idxs) < 2:
            continue
        nterms = [seq_by_idx[i][:NTERM_WIN] for i in idxs]
        if not any(nterms):
            continue
        counts = Counter(nterms)
        mode_nterm, _ = counts.most_common(1)[0]
        for i, nt in zip(idxs, nterms):
            nterm_deviates[i] = float(nt != mode_nterm and nt != '')

    # ---- disorder_nterm: metapredict mean disorder over first 60 residues ----
    print('Computing disorder_nterm (metapredict, first 60 residues)...')
    import metapredict as meta
    disorder_nterm = np.zeros(n_iso, dtype=np.float32)
    uniq_seqs = {}
    for s in seq_by_idx:
        if s and s not in uniq_seqs:
            uniq_seqs[s] = None
    print(f'  {len(uniq_seqs)} unique sequences to score')
    for i, s in enumerate(uniq_seqs):
        uniq_seqs[s] = meta.predict_disorder(s)
        if i % 2000 == 0:
            print(f'    {i}/{len(uniq_seqs)}', flush=True)
    for i, s in enumerate(seq_by_idx):
        if s and s in uniq_seqs and uniq_seqs[s] is not None:
            d = uniq_seqs[s]
            disorder_nterm[i] = float(np.mean(d[:min(NTERM_WIN, len(d))]))

    # ---- n_domains: reuse existing domain_matrix (already computed for THIS population) ----
    domain_mat = np.load(f'{ROOT}/hMuscle/results_isoform/features/domain_matrix_proper_test.npy')
    n_domains = domain_mat.sum(axis=1).astype(np.float32)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, n_domains=n_domains, nterm_deviates=nterm_deviates, disorder_nterm=disorder_nterm)
    print(f'\n[stats] n_domains: mean={n_domains.mean():.2f} nonzero-genes-with-var=?')
    print(f'[stats] nterm_deviates: sum={nterm_deviates.sum():.0f} ({100*nterm_deviates.mean():.1f}%)')
    print(f'[stats] disorder_nterm: mean={disorder_nterm[disorder_nterm>0].mean():.3f}')
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
