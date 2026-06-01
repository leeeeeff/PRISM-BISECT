#!/usr/bin/env python3
"""
build_proper_domain_matrix.py
==============================
domain_matrix.npy (pre-padded integer sequence) → 올바른 Pfam presence/absence 행렬 재구축

배경:
  기존 domain_matrix.npy = pre-padded sequence of Pfam integer IDs
    domain_matrix[i] = [0, 0, ..., 0, id1, id2, id3]  (right-aligned)
  → delta = matrix[i] - matrix[canonical]  은 의미 없음 (ID 빼기이므로)

  domain_list.txt = 텍스트 Pfam ID 목록
    isoform_id <TAB> PF00096 PF00069 ...

  human_domain_train.npy = 같은 pre-padded format (train isoforms)

목표:
  1. Pfam text ID → integer ID 역매핑 재구성
     (domain_list.txt text + domain_matrix.npy integer 정렬로 추론)
  2. 공통 Pfam vocabulary 결정 (top-N by frequency from train+test)
  3. 올바른 binary presence/absence 행렬 생성
     test : (36748, N_pfam)
     train: (31668, N_pfam)
  4. MANE canonical 기반 delta 계산
     domain_delta_proper_test.npy  (36748, N_pfam) float32
     domain_delta_proper_train.npy (31668, N_pfam) float32

출력:
  results_isoform/features/domain_matrix_proper_test.npy   (36748, N_pfam)
  results_isoform/features/domain_matrix_proper_train.npy  (31668, N_pfam)
  results_isoform/features/domain_delta_proper_test.npy    (36748, N_pfam)
  results_isoform/features/domain_delta_proper_train.npy   (31668, N_pfam)
  results_isoform/features/domain_pfam_vocab.txt           N_pfam Pfam IDs (index-ordered)
  results_isoform/features/pfam_to_int_mapping.json        {Pfam_ID: integer}

실행:
  cd hMuscle
  conda run -n isoform_env python preprocessing/build_proper_domain_matrix.py
"""

import numpy as np
import os, json, time
from collections import Counter, defaultdict

# ── 경로 ──────────────────────────────────────────────────────────────────────
DOMAIN_LIST_TEST  = 'data/domain/domain_list.txt'
DOMAIN_MATRIX_TEST  = 'data/domain/domain_matrix.npy'
DOMAIN_MATRIX_TRAIN = 'data/raw_data/data/domains/human_domain_train.npy'

ISO_LIST_FILE  = 'model/my_isoform_list_fixed.npy'
GENE_LIST_FILE = 'model/my_gene_list_fixed.npy'
TRAIN_ISO_FILE = 'data/raw_data/data/id_lists/train_isoform_list.npy'
TRAIN_GENE_FILE= 'data/raw_data/data/id_lists/train_gene_list.npy'
CANONICAL_REF  = 'results_isoform/features/canonical_reference.tsv'

OUT_DIR = 'results_isoform/features'
os.makedirs(OUT_DIR, exist_ok=True)

N_PFAM = 512   # 상위 N Pfam 패밀리 사용 (251보다 크게, full vocabulary 허용)

print("=" * 70)
print(" Step 1: domain_list.txt 파싱 (test 이소폼 Pfam IDs)")
print("=" * 70)

# test isoform 순서 로드
iso_list = np.load(ISO_LIST_FILE, allow_pickle=True)
iso_list = [s.decode() if isinstance(s, bytes) else s for s in iso_list]
iso_to_idx = {iso: i for i, iso in enumerate(iso_list)}

# domain_list.txt 파싱: isoform_id.p1 → [PF00096, PF00069, ...]
#   isoform_id 형식: ENST00000394825.6.p1 → base ID: ENST00000394825.6
domain_list_pfam = {}   # iso_id (without .p suffix) → list of Pfam IDs

with open(DOMAIN_LIST_TEST) as f:
    for line in f:
        parts = line.strip().split('\t')
        raw_id = parts[0]
        # .p1, .p2 suffix 제거
        iso_id = raw_id.split('.p')[0] if '.p' in raw_id else raw_id
        pfam_ids = parts[1].split() if len(parts) > 1 and parts[1].strip() else []
        domain_list_pfam[iso_id] = pfam_ids

print(f"  Parsed {len(domain_list_pfam)} isoforms from domain_list.txt")
has_domain = sum(1 for v in domain_list_pfam.values() if v)
print(f"  Isoforms with >= 1 Pfam domain: {has_domain} ({has_domain/len(domain_list_pfam)*100:.1f}%)")

# 전체 Pfam ID 빈도 (test)
pfam_freq_test = Counter()
for pfams in domain_list_pfam.values():
    pfam_freq_test.update(pfams)
print(f"  Unique Pfam IDs in test: {len(pfam_freq_test)}")
print(f"  Top-5 Pfam IDs: {pfam_freq_test.most_common(5)}")

print()
print("=" * 70)
print(" Step 2: Pfam text ID → integer ID 역매핑 재구성")
print("=" * 70)

# domain_matrix.npy (pre-padded integer sequences) 로드
dm_test = np.load(DOMAIN_MATRIX_TEST)
print(f"  domain_matrix_test: {dm_test.shape}")

# 정렬: iso_list의 순서와 domain_list.txt의 순서가 같다고 가정
# (domain_dataset.py가 같은 isoform list로 만들었으므로)
pfam_to_int = {}    # PF00096 → 5159
int_to_pfam = {}    # 5159 → PF00096

n_reconstructed = 0
n_mismatch = 0

for i, iso_id in enumerate(iso_list):
    pfams = domain_list_pfam.get(iso_id, [])
    if not pfams:
        continue

    row = dm_test[i]
    # pre-padded: non-zero values at the END
    nonzero_vals = row[row != 0].tolist()

    # 길이가 매칭되는 경우만 사용 (정확한 정렬 확인)
    if len(nonzero_vals) != len(pfams):
        n_mismatch += 1
        continue

    # 순서대로 매핑
    for pfam_id, int_id in zip(pfams, nonzero_vals):
        int_id = int(int_id)
        if pfam_id not in pfam_to_int:
            pfam_to_int[pfam_id] = int_id
            int_to_pfam[int_id] = pfam_id
            n_reconstructed += 1
        elif pfam_to_int[pfam_id] != int_id:
            # 충돌: 같은 Pfam이 다른 integer에 매핑
            pass  # 첫 번째 매핑 유지

print(f"  Reconstructed {n_reconstructed} Pfam→integer mappings")
print(f"  Length mismatches (skipped): {n_mismatch}")
print(f"  Coverage: {len(pfam_to_int)}/{len(pfam_freq_test)} Pfam IDs ({len(pfam_to_int)/len(pfam_freq_test)*100:.1f}%)")

# 매핑 저장
with open(f'{OUT_DIR}/pfam_to_int_mapping.json', 'w') as f:
    json.dump(pfam_to_int, f)
print(f"  Saved: {OUT_DIR}/pfam_to_int_mapping.json")

print()
print("=" * 70)
print(" Step 3: Train 이소폼 Pfam 역디코딩")
print("=" * 70)

dm_train = np.load(DOMAIN_MATRIX_TRAIN)
print(f"  human_domain_train: {dm_train.shape}")

# train 이소폼의 integer ID를 Pfam text로 역변환
train_pfam_sets = []  # list of set(Pfam IDs) per train isoform
n_decoded = 0
n_unknown_int = 0

for i in range(len(dm_train)):
    row = dm_train[i]
    nonzero_ints = row[row != 0].tolist()
    pfam_set = set()
    for int_id in nonzero_ints:
        int_id = int(int_id)
        if int_id in int_to_pfam:
            pfam_set.add(int_to_pfam[int_id])
        else:
            n_unknown_int += 1
    train_pfam_sets.append(pfam_set)
    if pfam_set:
        n_decoded += 1

pfam_freq_train = Counter()
for s in train_pfam_sets:
    pfam_freq_train.update(s)

print(f"  Train isoforms with decoded domains: {n_decoded}/{len(dm_train)} ({n_decoded/len(dm_train)*100:.1f}%)")
print(f"  Unknown integer IDs: {n_unknown_int}")
print(f"  Unique Pfam IDs in train: {len(pfam_freq_train)}")

print()
print("=" * 70)
print(f" Step 4: 공통 Pfam vocabulary 선정 (top-{N_PFAM})")
print("=" * 70)

# train+test 합산 빈도 기준 상위 N_PFAM 선택
pfam_freq_all = Counter()
pfam_freq_all.update(pfam_freq_train)
pfam_freq_all.update(pfam_freq_test)

top_pfams = [pfam for pfam, _ in pfam_freq_all.most_common(N_PFAM)]
pfam_to_col = {pfam: col for col, pfam in enumerate(top_pfams)}

print(f"  Total unique Pfam IDs (train+test): {len(pfam_freq_all)}")
print(f"  Selected top-{N_PFAM} vocabulary")
print(f"  Coverage (test): {sum(1 for p in pfam_freq_test if p in pfam_to_col)}/{len(pfam_freq_test)} Pfam IDs")
print(f"  Top-5 in vocabulary: {top_pfams[:5]}")
print(f"  Min frequency in vocabulary: {pfam_freq_all[top_pfams[-1]]}")

# 저장
with open(f'{OUT_DIR}/domain_pfam_vocab.txt', 'w') as f:
    for col, pfam in enumerate(top_pfams):
        f.write(f"{col}\t{pfam}\t{pfam_freq_all[pfam]}\n")
print(f"  Saved: {OUT_DIR}/domain_pfam_vocab.txt")

print()
print("=" * 70)
print(f" Step 5: 올바른 binary presence matrix 생성")
print("=" * 70)

# Test domain matrix
N_test = len(iso_list)
dm_test_proper = np.zeros((N_test, N_PFAM), dtype=np.float32)
for i, iso_id in enumerate(iso_list):
    pfams = domain_list_pfam.get(iso_id, [])
    for p in pfams:
        if p in pfam_to_col:
            dm_test_proper[i, pfam_to_col[p]] = 1.0

nz_test = (dm_test_proper != 0).any(axis=1).sum()
print(f"  Test  matrix: {dm_test_proper.shape}, isoforms with domains: {nz_test} ({nz_test/N_test*100:.1f}%)")

# Train domain matrix
N_train = len(dm_train)
dm_train_proper = np.zeros((N_train, N_PFAM), dtype=np.float32)
for i, pfam_set in enumerate(train_pfam_sets):
    for p in pfam_set:
        if p in pfam_to_col:
            dm_train_proper[i, pfam_to_col[p]] = 1.0

nz_train = (dm_train_proper != 0).any(axis=1).sum()
print(f"  Train matrix: {dm_train_proper.shape}, isoforms with domains: {nz_train} ({nz_train/N_train*100:.1f}%)")

np.save(f'{OUT_DIR}/domain_matrix_proper_test.npy',  dm_test_proper)
np.save(f'{OUT_DIR}/domain_matrix_proper_train.npy', dm_train_proper)
print(f"  Saved: domain_matrix_proper_test.npy, domain_matrix_proper_train.npy")

print()
print("=" * 70)
print(" Step 6: MANE canonical 기반 delta 계산")
print("=" * 70)

# Test delta
import pandas as pd
canon = pd.read_csv(CANONICAL_REF, sep='\t')
gene_list_test = np.load(GENE_LIST_FILE, allow_pickle=True)
gene_list_test = [g.decode() if isinstance(g, bytes) else g for g in gene_list_test]
gene_base_test = [g.split('.')[0] for g in gene_list_test]
gene_to_cidx = dict(zip(canon['gene_base'].str.split('.').str[0], canon['canonical_iso_idx'].astype(int)))

delta_test = np.zeros_like(dm_test_proper)
n_no_canon = 0
for i, gb in enumerate(gene_base_test):
    cidx = gene_to_cidx.get(gb)
    if cidx is not None and cidx < N_test:
        delta_test[i] = dm_test_proper[i] - dm_test_proper[cidx]
    else:
        n_no_canon += 1

nz_delta_test = (delta_test != 0).any(axis=1).sum()
print(f"  Test delta: {delta_test.shape}")
print(f"    Isoforms with domain change: {nz_delta_test} ({nz_delta_test/N_test*100:.1f}%)")
print(f"    No canonical found: {n_no_canon}")
gains = (delta_test > 0).sum(); losses = (delta_test < 0).sum()
print(f"    Domain gains: {gains}, losses: {losses}")

# Train delta
train_gene = np.load(TRAIN_GENE_FILE, allow_pickle=True)
train_gene = [g.decode() if isinstance(g, bytes) else g for g in train_gene]
from collections import defaultdict

gene_to_train_idxs = defaultdict(list)
for i, g in enumerate(train_gene):
    gene_to_train_idxs[g].append(i)

# Train canonical: gene 내 domain 수 가장 많은 isoform
delta_train = np.zeros_like(dm_train_proper)
n_multi_gene = 0
for g, idxs in gene_to_train_idxs.items():
    if len(idxs) == 1:
        continue
    n_multi_gene += 1
    domain_counts = dm_train_proper[idxs].sum(axis=1)
    canonical_local = int(np.argmax(domain_counts))
    canonical_idx = idxs[canonical_local]
    for idx in idxs:
        if idx != canonical_idx:
            delta_train[idx] = dm_train_proper[idx] - dm_train_proper[canonical_idx]

nz_delta_train = (delta_train != 0).any(axis=1).sum()
print(f"  Train delta: {delta_train.shape}")
print(f"    Genes with >1 isoform: {n_multi_gene}")
print(f"    Isoforms with domain change: {nz_delta_train} ({nz_delta_train/N_train*100:.1f}%)")

np.save(f'{OUT_DIR}/domain_delta_proper_test.npy',  delta_test)
np.save(f'{OUT_DIR}/domain_delta_proper_train.npy', delta_train)
print(f"  Saved: domain_delta_proper_test.npy, domain_delta_proper_train.npy")

print()
print("=" * 70)
print(" Step 7: Pearson(ESM-2 delta, proper domain delta) 재검증")
print("=" * 70)
from scipy.stats import pearsonr

esm2 = np.load('data/esm2_embeddings_t30_150M.npy').astype(np.float32)
esm2_delta = np.zeros_like(esm2)
for i, gb in enumerate(gene_base_test):
    cidx = gene_to_cidx.get(gb)
    if cidx is not None and cidx < N_test:
        esm2_delta[i] = esm2[i] - esm2[cidx]
esm2_norm = np.linalg.norm(esm2_delta, axis=1)

dd_proper_norm = delta_test.sum(axis=1)  # L1 norm of binary delta
dd_proper_abs  = np.abs(delta_test).sum(axis=1)

# non-canonical only
nc_mask = esm2_norm > 0.01
r_all, p_all = pearsonr(esm2_norm, dd_proper_abs)
r_nc,  p_nc  = pearsonr(esm2_norm[nc_mask], dd_proper_abs[nc_mask])
print(f"  All isoforms (n={nc_mask.sum()+int((~nc_mask).sum())}): Pearson(ESM2_delta, DD_proper) = {r_all:.4f} (p={p_all:.2e})")
print(f"  Non-canonical (n={nc_mask.sum()}):                      Pearson(ESM2_delta, DD_proper) = {r_nc:.4f} (p={p_nc:.2e})")

# 이전 broken delta와 비교
dd_broken = np.load('results_isoform/features/domain_delta_v2.npy')
dd_broken_norm = np.abs(dd_broken).sum(axis=1)
r_broken, _ = pearsonr(esm2_norm, dd_broken_norm)
r_broken_nc, _ = pearsonr(esm2_norm[nc_mask], dd_broken_norm[nc_mask])
print(f"  [Reference] Broken delta_v2: r_all={r_broken:.4f}, r_nc={r_broken_nc:.4f}")

print()
print("=" * 70)
print(" 완료 요약")
print("=" * 70)
print(f"  Pfam vocabulary: {N_PFAM} families")
print(f"  Test proper matrix: {dm_test_proper.shape}, {nz_test} isoforms ({nz_test/N_test*100:.1f}%) with domains")
print(f"  Train proper matrix: {dm_train_proper.shape}, {nz_train} isoforms ({nz_train/N_train*100:.1f}%) with domains")
print(f"  Test domain delta: {nz_delta_test} isoforms ({nz_delta_test/N_test*100:.1f}%) with changes")
print(f"  Train domain delta: {nz_delta_train} isoforms ({nz_delta_train/N_train*100:.1f}%) with changes")
print(f"  ESM-2 orthogonality: r_nc={r_nc:.4f} (proper) vs r_nc={r_broken_nc:.4f} (broken)")
print()
print("Done.")
