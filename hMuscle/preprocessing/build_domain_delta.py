# -*- coding: utf-8 -*-
"""
build_domain_delta.py
=====================
Alt 1: Domain Delta 특징 계산

domain_delta[i] = domain_matrix[i] - domain_matrix[canonical_of_gene_i]

canonical isoform 결정 기준:
  1. 해당 유전자의 isoform 중 domain 개수 (non-zero) 가장 많은 것
  2. 동점 시: 먼저 나오는 isoform (index 작은 것)

출력:
  ../results/domain/domain_delta.npy      (36748, 251) float32
  ../results/domain/iso_gene_map.txt      isoform_id → gene_id 매핑
  ../results/domain/canonical_map.txt     gene_id → canonical_isoform_idx

실행:
  cd hMuscle/preprocessing/
  python build_domain_delta.py
"""

import numpy as np
import re
import os
import sys

# ── 경로 ──────────────────────────────────────────────────────────────
ISO_LIST_FILE    = '../model/my_isoform_list_fixed.npy'
DOMAIN_MATRIX    = '../results/domain/domain_matrix.npy'
GTF_FILE         = '../data/cleaned_annotations.gtf'

OUT_DELTA        = '../results_isoform/features/domain_delta.npy'
OUT_ISO_GENE_MAP = '../results_isoform/features/iso_gene_map.txt'
OUT_CANONICAL    = '../results_isoform/features/canonical_map.txt'


def parse_gtf_transcript_gene(gtf_path):
    """GTF에서 transcript_id → gene_id 매핑 반환"""
    print("[GTF] Parsing {} ...".format(gtf_path))
    mapping = {}
    with open(gtf_path) as f:
        for line in f:
            if '\ttranscript\t' not in line:
                continue
            # gene_id 추출
            m_gene = re.search(r'gene_id "([^"]+)"', line)
            m_tx   = re.search(r'transcript_id "([^"]+)"', line)
            if m_gene and m_tx:
                tx   = m_tx.group(1)
                gene = m_gene.group(1)
                mapping[tx] = gene
    print("[GTF] {} transcripts mapped".format(len(mapping)))
    return mapping


def build_iso_gene_array(iso_list, tx_gene_map):
    """isoform_list 각 항목의 gene_id 반환 (없으면 None)"""
    iso_gene = []
    missing = 0
    for iso_bytes in iso_list:
        iso_id = iso_bytes.decode() if isinstance(iso_bytes, bytes) else iso_bytes
        # ENST ID에 버전 suffix가 다를 수 있음 (e.g. ENST000.10 vs ENST000.10.1)
        gene = tx_gene_map.get(iso_id)
        if gene is None:
            # 버전 앞부분만으로 재시도
            base_id = iso_id.split('.')[0]
            for k, v in tx_gene_map.items():
                if k.startswith(base_id):
                    gene = v
                    break
        if gene is None:
            missing += 1
        iso_gene.append(gene)
    print("[Map] {} isoforms mapped | {} unmapped".format(
        len(iso_list) - missing, missing))
    return iso_gene


def find_canonical_per_gene(iso_gene, domain_matrix):
    """
    각 유전자의 canonical isoform index 결정.
    기준: domain 수 (non-zero columns) 최대.
    """
    from collections import defaultdict
    gene_to_idxs = defaultdict(list)
    for i, gene in enumerate(iso_gene):
        if gene is not None:
            gene_to_idxs[gene].append(i)

    gene_to_canonical = {}
    for gene, idxs in gene_to_idxs.items():
        # 각 isoform의 domain 개수 (non-zero)
        domain_counts = [(domain_matrix[i] != 0).sum() for i in idxs]
        best_local_idx = int(np.argmax(domain_counts))
        gene_to_canonical[gene] = idxs[best_local_idx]

    print("[Canonical] {} genes, {} canonical isoforms".format(
        len(gene_to_idxs), len(gene_to_canonical)))
    return gene_to_canonical


def compute_domain_delta(iso_gene, gene_to_canonical, domain_matrix):
    """domain_delta[i] = domain_matrix[i] - domain_matrix[canonical_i]"""
    n = len(iso_gene)
    delta = np.zeros_like(domain_matrix, dtype=np.float32)

    no_gene_count = 0
    for i, gene in enumerate(iso_gene):
        if gene is None:
            no_gene_count += 1
            continue
        can_idx = gene_to_canonical.get(gene)
        if can_idx is None:
            no_gene_count += 1
            continue
        delta[i] = domain_matrix[i].astype(np.float32) - domain_matrix[can_idx].astype(np.float32)

    print("[Delta] Computed. Zero-delta rows (no gene/canonical): {}".format(no_gene_count))
    print("[Delta] Non-zero delta rows: {}".format((delta.sum(axis=1) != 0).sum()))
    return delta


def main():
    # 1. 파일 로드
    print("[Load] Isoform list: {}".format(ISO_LIST_FILE))
    iso_list = np.load(ISO_LIST_FILE, allow_pickle=True)
    print("  shape: {}".format(iso_list.shape))

    print("[Load] Domain matrix: {}".format(DOMAIN_MATRIX))
    domain_matrix = np.load(DOMAIN_MATRIX, allow_pickle=True)
    print("  shape: {} dtype: {}".format(domain_matrix.shape, domain_matrix.dtype))

    # 2. GTF 파싱
    tx_gene_map = parse_gtf_transcript_gene(GTF_FILE)

    # 3. isoform → gene 매핑
    iso_gene = build_iso_gene_array(iso_list, tx_gene_map)

    # 4. gene별 canonical isoform
    gene_to_canonical = find_canonical_per_gene(iso_gene, domain_matrix)

    # 5. domain_delta 계산
    delta = compute_domain_delta(iso_gene, gene_to_canonical, domain_matrix)

    # 6. 저장
    os.makedirs(os.path.dirname(OUT_DELTA) or '.', exist_ok=True)
    np.save(OUT_DELTA, delta)
    print("[Save] domain_delta.npy saved: shape={}".format(delta.shape))

    # 7. 매핑 파일 저장 (디버그용)
    with open(OUT_ISO_GENE_MAP, 'w') as f:
        for i, (iso_bytes, gene) in enumerate(zip(iso_list, iso_gene)):
            iso_id = iso_bytes.decode() if isinstance(iso_bytes, bytes) else iso_bytes
            f.write("{}\t{}\t{}\n".format(i, iso_id, gene or 'UNKNOWN'))

    with open(OUT_CANONICAL, 'w') as f:
        for gene, can_idx in gene_to_canonical.items():
            iso_id = iso_list[can_idx].decode() if isinstance(iso_list[can_idx], bytes) else iso_list[can_idx]
            f.write("{}\t{}\t{}\n".format(gene, can_idx, iso_id))

    print("[Done] domain_delta.npy: {}".format(delta.shape))
    print("  Stored at: {}".format(OUT_DELTA))

    # 8. 검증 샘플
    print("\n[Sanity Check] First 5 delta rows (non-zero check):")
    for i in range(5):
        nz = (delta[i] != 0).sum()
        gene = iso_gene[i]
        iso_id = iso_list[i].decode() if isinstance(iso_list[i], bytes) else iso_list[i]
        print("  [{}] {} | gene={} | non-zero delta domains={}".format(
            i, iso_id, gene, nz))


if __name__ == '__main__':
    main()
