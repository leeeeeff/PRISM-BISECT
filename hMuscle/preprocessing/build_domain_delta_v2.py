# -*- coding: utf-8 -*-
"""
build_domain_delta_v2.py
========================
Step 2: GENCODE canonical 기반 domain_delta 재계산

기존 build_domain_delta.py와 차이:
  - canonical = MANE_Select / Ensembl_canonical / APPRIS / longest_CDS (3-tier)
  - canonical_reference.tsv에서 결정된 canonical 사용
  - domain_matrix 자체는 변경 없음, delta 기준만 교체

출력:
  ../results_isoform/features/domain_delta_v2.npy      (36748, 251) float32
  ../results_isoform/features/canonical_map_v2.txt      gene_id → canonical 정보

실행:
  cd hMuscle/preprocessing/
  python build_domain_delta_v2.py
"""

import numpy as np
import os

# ── 경로 ──────────────────────────────────────────────────────────────
DOMAIN_MATRIX     = '../results/domain/domain_matrix.npy'
CANONICAL_REF     = '../results_isoform/features/canonical_reference.tsv'
ISO_LIST_FILE     = '../model/my_isoform_list_fixed.npy'

OUT_DELTA         = '../results_isoform/features/domain_delta_v2.npy'
OUT_CANONICAL_MAP = '../results_isoform/features/canonical_map_v2.txt'


def load_canonical_reference(ref_path):
    """
    canonical_reference.tsv 로드
    
    Returns:
        gene_to_canonical: {gene_versioned: (canonical_idx, canonical_iso_id, source)}
    """
    print("[Ref] Loading {} ...".format(ref_path))
    gene_to_canonical = {}
    
    with open(ref_path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                gene_versioned = parts[1]
                canonical_idx  = int(parts[3])
                canonical_src  = parts[4]
                canonical_iso  = parts[5]
                gene_to_canonical[gene_versioned] = (canonical_idx, canonical_iso, canonical_src)
    
    print("  Loaded {} gene → canonical mappings".format(len(gene_to_canonical)))
    return gene_to_canonical


def compute_domain_delta_v2(iso_list, gene_list, gene_to_canonical, domain_matrix):
    """domain_delta[i] = domain_matrix[i] - domain_matrix[canonical_i]"""
    n = len(iso_list)
    delta = np.zeros_like(domain_matrix, dtype=np.float32)
    
    no_canonical = 0
    source_counts = {}
    
    for i in range(n):
        gene = gene_list[i]
        can_info = gene_to_canonical.get(gene)
        
        if can_info is None:
            no_canonical += 1
            continue
        
        can_idx, can_iso, source = can_info
        delta[i] = domain_matrix[i].astype(np.float32) - domain_matrix[can_idx].astype(np.float32)
        source_counts[source] = source_counts.get(source, 0) + 1
    
    n_nonzero = (np.abs(delta).sum(axis=1) != 0).sum()
    print("[Delta] Computed. Zero-delta (no canonical): {}".format(no_canonical))
    print("[Delta] Non-zero delta rows: {} ({:.1f}%)".format(n_nonzero, n_nonzero / n * 100))
    print("[Delta] Source distribution (isoform-level):")
    for src, cnt in sorted(source_counts.items()):
        print("    {}: {}".format(src, cnt))
    
    return delta


def main():
    print("=" * 60)
    print("Step 2: Domain Delta v2 (GENCODE canonical)")
    print("=" * 60)
    
    # Load
    iso_list = np.load(ISO_LIST_FILE, allow_pickle=True)
    iso_list = [s.decode() if isinstance(s, bytes) else s for s in iso_list]
    
    gene_list = np.load('../model/my_gene_list_fixed.npy', allow_pickle=True)
    gene_list = [g.decode() if isinstance(g, bytes) else g for g in gene_list]
    
    domain_matrix = np.load(DOMAIN_MATRIX, allow_pickle=True)
    print("[Load] domain_matrix: {} dtype={}".format(domain_matrix.shape, domain_matrix.dtype))
    
    gene_to_canonical = load_canonical_reference(CANONICAL_REF)
    
    # Compute
    delta = compute_domain_delta_v2(iso_list, gene_list, gene_to_canonical, domain_matrix)
    
    # Save
    os.makedirs(os.path.dirname(OUT_DELTA) or '.', exist_ok=True)
    np.save(OUT_DELTA, delta)
    print("[Save] {} shape={}".format(OUT_DELTA, delta.shape))
    
    # Save canonical map (for exon matrix to reuse)
    with open(OUT_CANONICAL_MAP, 'w') as f:
        for gene, (can_idx, can_iso, source) in gene_to_canonical.items():
            f.write("{}\t{}\t{}\t{}\n".format(gene, can_idx, can_iso, source))
    print("[Save] {}".format(OUT_CANONICAL_MAP))
    
    # Compare with v1
    old_delta_path = '../results_isoform/features/domain_delta.npy'
    if os.path.exists(old_delta_path):
        old_delta = np.load(old_delta_path)
        
        old_d0 = np.all(old_delta == 0, axis=1)
        new_d0 = np.all(delta == 0, axis=1)
        
        d0_to_d1 = (old_d0 & ~new_d0).sum()
        d1_to_d0 = (~old_d0 & new_d0).sum()
        
        print("\n[Comparison] vs domain_delta v1:")
        print("  Old D0 (domain same): {}".format(old_d0.sum()))
        print("  New D0 (domain same): {}".format(new_d0.sum()))
        print("  D0→D1 (was same, now different): {}".format(d0_to_d1))
        print("  D1→D0 (was different, now same): {}".format(d1_to_d0))
        print("  Net D0 change: {:+d}".format(new_d0.sum() - old_d0.sum()))
    
    print("\n[Done]")


if __name__ == '__main__':
    main()
