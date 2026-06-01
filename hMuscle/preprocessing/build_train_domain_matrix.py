#!/usr/bin/env python3
"""
build_train_domain_matrix.py
============================
CDD 어노테이션 (human_CDD_query_results.txt + human_isoform_dm.txt) →
train isoform binary Pfam presence/absence matrix

입력:
  data/raw_data/data/raw_data/domain_data/human_CDD_query_results.txt
  data/raw_data/data/raw_data/domain_data/human_isoform_dm.txt
  data/raw_data/data/id_lists/train_isoform_list.npy
  data/raw_data/data/id_lists/train_gene_list.npy
  results_isoform/features/domain_pfam_vocab_v2.txt  (512 Pfam vocab, test set 기준)

출력:
  results_isoform/features/train_domain_matrix_proper.npy  (31668, 512) float32
  results_isoform/features/train_domain_delta_proper.npy   (31668, 512) float32
"""

import numpy as np
import re
from collections import defaultdict, Counter
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

OUT_DIR = 'results_isoform/features'

# ── 1. CDD accession → Pfam ID 역매핑 ────────────────────────────
print("[1] Building CDD accession → Pfam ID mapping ...")
acc_to_pfam = {}
pfam_pattern = re.compile(r'^pfam(\d+)$')
with open('data/raw_data/data/raw_data/domain_data/human_CDD_query_results.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 8:
            continue
        try:
            acc = int(parts[2])
            pm = pfam_pattern.match(parts[7].strip())
            if pm:
                acc_to_pfam[acc] = f'PF{int(pm.group(1)):05d}'
        except (ValueError, IndexError):
            pass
print(f"  CDD acc → Pfam: {len(acc_to_pfam)} mappings")

# ── 2. NM_ID → Pfam IDs (human_isoform_dm.txt 기반) ──────────────
print("[2] Parsing human_isoform_dm.txt ...")
nm_to_pfams = defaultdict(set)
with open('data/raw_data/data/raw_data/domain_data/human_isoform_dm.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 3 or not parts[2].strip():
            continue
        nm_id = parts[1]
        accs = [int(x) for x in parts[2].split() if x.strip().isdigit()]
        pfams = {acc_to_pfam[a] for a in accs if a in acc_to_pfam}
        if pfams:
            nm_to_pfams[nm_id].update(pfams)
print(f"  NM IDs with Pfam annotation: {len(nm_to_pfams)}")

# ── 3. Top-512 vocab 로드 (test set 기준, 동일 feature space 유지) ──
print("[3] Loading top-512 Pfam vocabulary ...")
pfam_to_col = {}
with open(f'{OUT_DIR}/domain_pfam_vocab_v2.txt') as f:
    for line in f:
        col, pfam, freq = line.strip().split('\t')
        pfam_to_col[pfam] = int(col)
N_PFAM = len(pfam_to_col)
print(f"  Loaded {N_PFAM} Pfam vocab entries")

# ── 4. Train isoform list 로드 & 매칭 ────────────────────────────
print("[4] Matching train isoforms ...")
train_ids = np.load('data/raw_data/data/id_lists/train_isoform_list.npy', allow_pickle=True)
train_ids = [s.decode() if isinstance(s, bytes) else s for s in train_ids]
train_genes = np.load('data/raw_data/data/id_lists/train_gene_list.npy', allow_pickle=True)
train_genes = [s.decode() if isinstance(s, bytes) else s for s in train_genes]

train_pfams = []
n_found = n_miss = n_multi = 0
for iso in train_ids:
    # comma-separated NM_ IDs (일부 isoform)
    nm_parts = [x.strip() for x in iso.split(',')]
    pfams = set()
    for nm in nm_parts:
        pfams.update(nm_to_pfams.get(nm, set()))
    train_pfams.append(pfams)
    if len(nm_parts) > 1:
        n_multi += 1
    if pfams:
        n_found += 1
    else:
        n_miss += 1

print(f"  Matched: {n_found}/{len(train_ids)} ({n_found/len(train_ids)*100:.1f}%) with >= 1 Pfam domain")
print(f"  No annotation: {n_miss}, Multi-NM entries: {n_multi}")

# ── 5. Binary presence matrix ─────────────────────────────────────
print("[5] Building binary presence matrix ...")
N = len(train_ids)
dm_train = np.zeros((N, N_PFAM), dtype=np.float32)
for i, pfam_set in enumerate(train_pfams):
    for p in pfam_set:
        if p in pfam_to_col:
            dm_train[i, pfam_to_col[p]] = 1.0

nz = (dm_train != 0).any(axis=1).sum()
print(f"  Matrix: {dm_train.shape}, {nz}/{N} ({nz/N*100:.1f}%) isoforms with domains in top-512 vocab")

# vocab coverage
train_pfams_all = set()
for s in train_pfams:
    train_pfams_all.update(s)
overlap = train_pfams_all & set(pfam_to_col.keys())
print(f"  Pfam vocab overlap: {len(overlap)}/{N_PFAM} ({len(overlap)/N_PFAM*100:.1f}%) dims active in train")

# ── 6. Gene-level canonical delta (gene symbol 기반) ──────────────
print("[6] Computing canonical delta (gene symbol, most-domain canonical) ...")
# gene symbol → canonical isoform idx (가장 많은 도메인을 가진 isoform)
gene_to_canon = {}
gene_to_idxs = defaultdict(list)
for i, gene in enumerate(train_genes):
    gene_to_idxs[gene].append(i)

for gene, idxs in gene_to_idxs.items():
    domain_counts = [dm_train[i].sum() for i in idxs]
    canon_local = idxs[int(np.argmax(domain_counts))]
    gene_to_canon[gene] = canon_local

delta_train = np.zeros_like(dm_train)
n_no_canon = 0
for i, gene in enumerate(train_genes):
    cidx = gene_to_canon.get(gene)
    if cidx is not None:
        delta_train[i] = dm_train[i] - dm_train[cidx]
    else:
        n_no_canon += 1

nz_delta = (delta_train != 0).any(axis=1).sum()
gains = (delta_train > 0).sum()
losses = (delta_train < 0).sum()
genes_multi = sum(1 for idxs in gene_to_idxs.values() if len(idxs) > 1)
print(f"  Genes with >1 isoform: {genes_multi}")
print(f"  Delta isoforms: {nz_delta}/{N} ({nz_delta/N*100:.1f}%)")
print(f"  Domain gains: {int(gains)}, losses: {int(losses)}, no_canon: {n_no_canon}")

# ── 7. 저장 ──────────────────────────────────────────────────────
np.save(f'{OUT_DIR}/train_domain_matrix_proper.npy', dm_train)
np.save(f'{OUT_DIR}/train_domain_delta_proper.npy', delta_train)
print(f"[7] Saved:")
print(f"  {OUT_DIR}/train_domain_matrix_proper.npy")
print(f"  {OUT_DIR}/train_domain_delta_proper.npy")

print()
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"  Train domain presence: {nz}/{N} = {nz/N*100:.1f}%")
print(f"  Train domain delta:    {nz_delta}/{N} = {nz_delta/N*100:.1f}%")
print(f"  Vocab overlap (train/test): {len(overlap)}/{N_PFAM} = {len(overlap)/N_PFAM*100:.1f}%")
print(f"  Canonical method: most-domain per gene symbol")
print("Done.")
