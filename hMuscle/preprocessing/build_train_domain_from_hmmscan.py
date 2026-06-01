#!/usr/bin/env python3
"""
build_train_domain_from_hmmscan.py
====================================
hmmscan domtblout 결과 → train binary Pfam presence/absence matrix

입력:
  results_isoform/features/hmmscan_train.domtblout
  data/raw_data/data/id_lists/train_isoform_list.npy
  data/raw_data/data/id_lists/train_gene_list.npy
  results_isoform/features/domain_pfam_vocab_v2.txt

출력:
  results_isoform/features/train_domain_matrix_hmmscan.npy  (31668, 512) float32
  results_isoform/features/train_domain_delta_hmmscan.npy   (31668, 512) float32
"""

import numpy as np
import re
from collections import defaultdict
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

OUT_DIR = 'results_isoform/features'
DOMTBL = f'{OUT_DIR}/hmmscan_train.domtblout'

# ── 1. hmmscan domtblout 파싱 ─────────────────────────────────────
print("[1] Parsing hmmscan domtblout ...")
nm_to_pfams = defaultdict(set)
n_lines = 0
with open(DOMTBL) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 22:
            continue
        n_lines += 1
        pfam_acc = parts[1].split('.')[0]   # PF00096.28 → PF00096
        nm_id = parts[3]                     # query sequence ID
        ievalue = float(parts[12])           # independent E-value
        if ievalue < 1e-5 and pfam_acc.startswith('PF'):
            nm_to_pfams[nm_id].add(pfam_acc)

print(f"  Parsed {n_lines} domain hits")
print(f"  NM IDs with Pfam: {len(nm_to_pfams)}")

# ── 2. Top-512 vocab 로드 ─────────────────────────────────────────
print("[2] Loading top-512 Pfam vocabulary ...")
pfam_to_col = {}
with open(f'{OUT_DIR}/domain_pfam_vocab_v2.txt') as f:
    for line in f:
        col, pfam, freq = line.strip().split('\t')
        pfam_to_col[pfam] = int(col)
N_PFAM = len(pfam_to_col)

# ── 3. Train isoform list 매칭 ────────────────────────────────────
print("[3] Matching train isoforms ...")
train_ids = np.load('data/raw_data/data/id_lists/train_isoform_list.npy', allow_pickle=True)
train_ids = [s.decode() if isinstance(s, bytes) else s for s in train_ids]
train_genes = np.load('data/raw_data/data/id_lists/train_gene_list.npy', allow_pickle=True)
train_genes = [s.decode() if isinstance(s, bytes) else s for s in train_genes]

train_pfams = []
n_found = 0
for iso in train_ids:
    nm_parts = [x.strip() for x in iso.split(',')]
    pfams = set()
    for nm in nm_parts:
        pfams.update(nm_to_pfams.get(nm, set()))
    train_pfams.append(pfams)
    if pfams:
        n_found += 1

print(f"  With >= 1 Pfam: {n_found}/{len(train_ids)} ({n_found/len(train_ids)*100:.1f}%)")

# ── 4. Binary presence matrix ─────────────────────────────────────
print("[4] Building binary presence matrix ...")
N = len(train_ids)
dm = np.zeros((N, N_PFAM), dtype=np.float32)
for i, pfam_set in enumerate(train_pfams):
    for p in pfam_set:
        if p in pfam_to_col:
            dm[i, pfam_to_col[p]] = 1.0

nz = (dm != 0).any(axis=1).sum()
all_pfams = set(p for s in train_pfams for p in s)
overlap = all_pfams & set(pfam_to_col.keys())
print(f"  Matrix: {dm.shape}, {nz}/{N} ({nz/N*100:.1f}%) with domains")
print(f"  Vocab overlap: {len(overlap)}/512 ({len(overlap)/512*100:.1f}%)")

# ── 5. Gene-level canonical delta ────────────────────────────────
print("[5] Computing canonical delta ...")
from collections import defaultdict
gene_to_idxs = defaultdict(list)
for i, gene in enumerate(train_genes):
    gene_to_idxs[gene].append(i)

gene_to_canon = {}
for gene, idxs in gene_to_idxs.items():
    domain_counts = [dm[i].sum() for i in idxs]
    gene_to_canon[gene] = idxs[int(np.argmax(domain_counts))]

delta = np.zeros_like(dm)
for i, gene in enumerate(train_genes):
    cidx = gene_to_canon.get(gene)
    if cidx is not None:
        delta[i] = dm[i] - dm[cidx]

nz_delta = (delta != 0).any(axis=1).sum()
print(f"  Delta isoforms: {nz_delta}/{N} ({nz_delta/N*100:.1f}%)")
print(f"  Gains: {int((delta>0).sum())}, Losses: {int((delta<0).sum())}")

# ── 6. 저장 ──────────────────────────────────────────────────────
np.save(f'{OUT_DIR}/train_domain_matrix_hmmscan.npy', dm)
np.save(f'{OUT_DIR}/train_domain_delta_hmmscan.npy', delta)
print(f"[6] Saved:")
print(f"  {OUT_DIR}/train_domain_matrix_hmmscan.npy")
print(f"  {OUT_DIR}/train_domain_delta_hmmscan.npy")
print()
print("=" * 60)
print(f"  Domain presence: {nz}/{N} = {nz/N*100:.1f}%")
print(f"  Domain delta:    {nz_delta}/{N} = {nz_delta/N*100:.1f}%")
print(f"  Vocab overlap:   {len(overlap)}/512 = {len(overlap)/512*100:.1f}%")
print("Done.")
