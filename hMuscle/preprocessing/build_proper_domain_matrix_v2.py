#!/usr/bin/env python3
"""
build_proper_domain_matrix_v2.py
=================================
domain_list.txt (ENST text IDs) → binary presence/absence matrix
ID 기반 직접 매칭 (정수 역매핑 우회)

출력:
  results_isoform/features/domain_matrix_proper_test_v2.npy  (36748, 512) float32
  results_isoform/features/domain_delta_proper_test_v2.npy   (36748, 512) float32
  results_isoform/features/domain_pfam_vocab_v2.txt           512 Pfam IDs
"""

import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import pearsonr
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

OUT_DIR = 'results_isoform/features'
N_PFAM = 512

# ── 1. domain_list.txt ID 기반 파싱 ──────────────────────────────
print("[1] Parsing domain_list.txt ...")
enst_to_pfams = {}
with open('data/domain/domain_list.txt') as f:
    for l in f:
        parts = l.strip().split('\t')
        raw_id = parts[0]
        base_id = raw_id.split('.p')[0] if '.p' in raw_id else raw_id
        pfams = set(parts[1].split()) if len(parts) > 1 and parts[1].strip() else set()
        enst_to_pfams[base_id] = pfams
print(f"  {len(enst_to_pfams)} isoform entries, {sum(1 for v in enst_to_pfams.values() if v)} with domains")

# ── 2. my_isoform_list_fixed와 ID 매칭 ──────────────────────────
print("[2] Matching to isoform list ...")
iso_list = np.load('model/my_isoform_list_fixed.npy', allow_pickle=True)
iso_list = [s.decode() if isinstance(s, bytes) else s for s in iso_list]

iso_pfams = []
n_found = n_bambu = n_miss = 0
for iso in iso_list:
    if 'BambuTx' in iso:
        iso_pfams.append(set()); n_bambu += 1
    elif iso in enst_to_pfams:
        iso_pfams.append(enst_to_pfams[iso]); n_found += 1
    else:
        iso_pfams.append(set()); n_miss += 1

print(f"  Matched ENST: {n_found}, BambuTx (no annot): {n_bambu}, Unmatched: {n_miss}")
n_with_domain = sum(1 for s in iso_pfams if s)
print(f"  Isoforms with >= 1 domain: {n_with_domain}/{len(iso_pfams)} ({n_with_domain/len(iso_pfams)*100:.1f}%)")

# ── 3. Pfam vocabulary (top-512) ─────────────────────────────────
print("[3] Building Pfam vocabulary ...")
pfam_freq = Counter()
for s in iso_pfams:
    pfam_freq.update(s)
print(f"  Unique Pfam IDs: {len(pfam_freq)}")

top_pfams = [p for p, _ in pfam_freq.most_common(N_PFAM)]
pfam_to_col = {p: c for c, p in enumerate(top_pfams)}
print(f"  Selected top-{N_PFAM}, min frequency: {pfam_freq[top_pfams[-1]]}")

with open(f'{OUT_DIR}/domain_pfam_vocab_v2.txt', 'w') as f:
    for c, p in enumerate(top_pfams):
        f.write(f"{c}\t{p}\t{pfam_freq[p]}\n")

# ── 4. Binary presence matrix ─────────────────────────────────────
print("[4] Building binary presence matrix ...")
N = len(iso_list)
dm_proper = np.zeros((N, N_PFAM), dtype=np.float32)
for i, pfam_set in enumerate(iso_pfams):
    for p in pfam_set:
        if p in pfam_to_col:
            dm_proper[i, pfam_to_col[p]] = 1.0

nz = (dm_proper != 0).any(axis=1).sum()
print(f"  Matrix: {dm_proper.shape}, {nz}/{N} ({nz/N*100:.1f}%) isoforms with domains")

# ── 5. MANE canonical delta ────────────────────────────────────────
print("[5] Computing MANE canonical delta ...")
canon = pd.read_csv('results_isoform/features/canonical_reference.tsv', sep='\t')
gene_list = np.load('model/my_gene_list_fixed.npy', allow_pickle=True)
gene_list = [g.decode() if isinstance(g, bytes) else g for g in gene_list]
gene_base = [g.split('.')[0] for g in gene_list]
gene_to_cidx = dict(zip(canon['gene_base'].str.split('.').str[0],
                        canon['canonical_iso_idx'].astype(int)))

delta_proper = np.zeros_like(dm_proper)
n_no_canon = 0
for i, gb in enumerate(gene_base):
    cidx = gene_to_cidx.get(gb)
    if cidx is not None and cidx < N:
        delta_proper[i] = dm_proper[i] - dm_proper[cidx]
    else:
        n_no_canon += 1

nz_delta = (delta_proper != 0).any(axis=1).sum()
gains = (delta_proper > 0).sum()
losses = (delta_proper < 0).sum()
print(f"  Delta isoforms: {nz_delta}/{N} ({nz_delta/N*100:.1f}%)")
print(f"  Domain gains: {int(gains)}, losses: {int(losses)}, no_canon: {n_no_canon}")

# ── 6. Pearson orthogonality vs ESM-2 ────────────────────────────
print("[6] Pearson(ESM2_delta, DD_proper) ...")
esm2 = np.load('data/esm2_embeddings_t30_150M.npy').astype(np.float32)
esm2_delta_mat = np.zeros_like(esm2)
for i, gb in enumerate(gene_base):
    cidx = gene_to_cidx.get(gb)
    if cidx is not None and cidx < N:
        esm2_delta_mat[i] = esm2[i] - esm2[cidx]

esm2_norm = np.linalg.norm(esm2_delta_mat, axis=1)
dd_norm = np.abs(delta_proper).sum(axis=1)
nc_mask = esm2_norm > 0.01

r_all, p_all = pearsonr(esm2_norm, dd_norm)
r_nc,  p_nc  = pearsonr(esm2_norm[nc_mask], dd_norm[nc_mask])
print(f"  All isoforms:      r={r_all:.4f} (p={p_all:.2e})")
print(f"  Non-canonical (n={nc_mask.sum()}): r={r_nc:.4f} (p={p_nc:.2e})")

# ── 7. 저장 ───────────────────────────────────────────────────────
np.save(f'{OUT_DIR}/domain_matrix_proper_test_v2.npy', dm_proper)
np.save(f'{OUT_DIR}/domain_delta_proper_test_v2.npy',  delta_proper)
print(f"[7] Saved:")
print(f"  {OUT_DIR}/domain_matrix_proper_test_v2.npy")
print(f"  {OUT_DIR}/domain_delta_proper_test_v2.npy")
print(f"  {OUT_DIR}/domain_pfam_vocab_v2.txt")

print()
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"  Test domain presence: {nz}/{N} = {nz/N*100:.1f}%")
print(f"  Test domain delta: {nz_delta}/{N} = {nz_delta/N*100:.1f}%")
print(f"  ESM-2 orthogonality r_nc = {r_nc:.4f}")
print(f"    {'INDEPENDENT' if abs(r_nc) < 0.3 else 'CORRELATED'} (threshold=0.3)")
print("Done.")
