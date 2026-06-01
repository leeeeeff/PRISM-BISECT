# -*- coding: utf-8 -*-
"""
build_exon_matrix_v2.py
=======================
Step 3: Exon matrix 고도화

기존 build_exon_matrix.py 대비 개선:
  1. Canonical = GENCODE v44 기반 (canonical_reference.tsv 사용)
  2. Exon clustering: 좌표 overlap 기반 exon cluster 생성
  3. Constitutive exon cluster 제거 (모든 isoform에 동일 → delta=0, 무정보)
  4. Continuous value: exon_length / max_length_in_cluster (0~1)
     - 완전 포함=1.0, 완전 제외=0.0, alt splice site=0<x<1
  5. 동적 MAX_EXONS: constitutive 제거 후 가변 cluster만 선택

출력:
  ../results_isoform/features/splicing/exon_matrix_v2.npy       (36748, MAX_VAR) float32
  ../results_isoform/features/splicing/splicing_delta_v2.npy    (36748, MAX_VAR) float32
  ../results_isoform/features/splicing/exon_meta_v2.npz         메타정보
  ../results_isoform/features/splicing/splicing_stats_v2.txt    통계

실행:
  cd hMuscle/preprocessing/
  python build_exon_matrix_v2.py
"""

import numpy as np
import re
import os
from collections import defaultdict

# ── 경로 ──────────────────────────────────────────────────────────────
ISO_LIST_FILE    = '../model/my_isoform_list_fixed.npy'
GTF_FILE         = '../data/cleaned_annotations.gtf'
CANONICAL_REF    = '../results_isoform/features/canonical_reference.tsv'

OUT_DIR          = '../results_isoform/features/splicing/'
OUT_EXON_MATRIX  = OUT_DIR + 'exon_matrix_v2.npy'
OUT_SPLICE_DELTA = OUT_DIR + 'splicing_delta_v2.npy'
OUT_META         = OUT_DIR + 'exon_meta_v2.npz'
OUT_STATS        = OUT_DIR + 'splicing_stats_v2.txt'

MAX_VAR_EXONS = 150   # 가변 exon cluster 상한 (constitutive 제거 후)


# ── GTF 파싱 ──────────────────────────────────────────────────────────

def parse_gtf_exons(gtf_path):
    """GTF에서 transcript별 exon 좌표 파싱."""
    print("[GTF] Parsing exons from {} ...".format(gtf_path))
    tx_exons = defaultdict(list)
    tx_gene  = {}
    
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            feature = parts[2]
            attrs   = parts[8]
            
            m_tx   = re.search(r'transcript_id "([^"]+)"', attrs)
            m_gene = re.search(r'gene_id "([^"]+)"', attrs)
            if not m_tx or not m_gene:
                continue
            
            tx_id   = m_tx.group(1)
            gene_id = m_gene.group(1)
            tx_gene[tx_id] = gene_id
            
            if feature == 'exon':
                chrom = parts[0]
                start = int(parts[3])
                end   = int(parts[4])
                tx_exons[tx_id].append((chrom, start, end))
    
    print("[GTF] {} transcripts | {} with exons".format(
        len(tx_gene), len(tx_exons)))
    return tx_exons, tx_gene


# ── Exon Clustering ──────────────────────────────────────────────────

def build_exon_clusters(exon_set):
    """
    Genomic overlap 기반 exon clustering.
    겹치는 exon들을 하나의 cluster로 묶음.
    
    Args:
        exon_set: set of (chrom, start, end)
    
    Returns:
        clusters: list of sets, each set contains (chrom, start, end) tuples
    """
    if not exon_set:
        return []
    
    # chrom별 분리 후 좌표 기준 정렬
    chrom_exons = defaultdict(list)
    for chrom, start, end in exon_set:
        chrom_exons[chrom].append((start, end))
    
    clusters = []
    for chrom in sorted(chrom_exons.keys()):
        sorted_exons = sorted(chrom_exons[chrom])
        
        # interval merging으로 overlapping group 생성
        current_cluster_start = sorted_exons[0][0]
        current_cluster_end = sorted_exons[0][1]
        current_members = [(chrom, sorted_exons[0][0], sorted_exons[0][1])]
        
        for start, end in sorted_exons[1:]:
            if start <= current_cluster_end:
                # overlap → 같은 cluster
                current_cluster_end = max(current_cluster_end, end)
                current_members.append((chrom, start, end))
            else:
                # no overlap → 새 cluster
                clusters.append(set(current_members))
                current_cluster_start = start
                current_cluster_end = end
                current_members = [(chrom, start, end)]
        
        clusters.append(set(current_members))
    
    return clusters


# ── Exon Matrix 구축 ──────────────────────────────────────────────────

def build_exon_matrix_v2(iso_list, tx_exons, tx_gene_map, gene_list,
                         max_var_exons=MAX_VAR_EXONS):
    """
    Continuous exon matrix 생성.
    
    각 유전자 내:
      1. 모든 isoform의 exon을 수집
      2. Overlap 기반 exon cluster 생성
      3. Constitutive cluster 제거 (모든 isoform에 동일 exon)
      4. Variable cluster에 대해: exon_length / max_length_in_cluster
    """
    n = len(iso_list)
    
    print("[ExonMatrix] Building v2 (continuous, cluster-based) ...")
    
    # gene → isoform indices
    gene_to_idxs = defaultdict(list)
    for i, gene in enumerate(gene_list):
        gene_to_idxs[gene].append(i)
    
    # isoform → exons (GTF version matching)
    iso_exon_map = {}
    missing_tx = 0
    for i, iso_id in enumerate(iso_list):
        exons = tx_exons.get(iso_id)
        if exons is None:
            # version matching
            base = iso_id.split('.')[0]
            for k in tx_exons.keys():
                if k.split('.')[0] == base or k.startswith(base):
                    exons = tx_exons[k]
                    break
        if exons is None:
            missing_tx += 1
            iso_exon_map[i] = set()
        else:
            iso_exon_map[i] = set(tuple(e) for e in exons)
    
    print("  Missing transcripts: {}".format(missing_tx))
    
    # Gene별 cluster 생성 및 variable exon 식별
    gene_var_clusters = {}  # gene → [cluster_info]
    total_constitutive = 0
    total_variable = 0
    
    for gene, idxs in gene_to_idxs.items():
        if len(idxs) < 2:
            # 단일 isoform 유전자: 모든 exon이 constitutive
            gene_var_clusters[gene] = []
            continue
        
        # 유전자 내 모든 exon 수집
        all_exons = set()
        for idx in idxs:
            all_exons.update(iso_exon_map.get(idx, set()))
        
        if not all_exons:
            gene_var_clusters[gene] = []
            continue
        
        # Cluster 생성
        clusters = build_exon_clusters(all_exons)
        
        # Constitutive vs variable 분류
        var_clusters = []
        for cluster in clusters:
            # 각 isoform이 이 cluster에서 어떤 exon을 포함하는지 확인
            iso_patterns = []
            for idx in idxs:
                iso_exons = iso_exon_map.get(idx, set())
                # cluster 내에서 overlap하는 exon 찾기
                overlapping = cluster & iso_exons
                if overlapping:
                    # 가장 긴 exon의 길이 사용
                    max_len = max(e[2] - e[1] + 1 for e in overlapping)
                    iso_patterns.append(max_len)
                else:
                    iso_patterns.append(0)
            
            # 모든 isoform에서 동일한 패턴이면 constitutive
            if len(set(iso_patterns)) > 1:
                # Variable cluster
                cluster_max_len = max(e[2] - e[1] + 1 for e in cluster)
                var_clusters.append({
                    'exons': cluster,
                    'max_len': cluster_max_len,
                    'variance': np.var(iso_patterns),
                })
                total_variable += 1
            else:
                total_constitutive += 1
        
        # variance 기준 정렬, 상위 max_var_exons 선택
        var_clusters.sort(key=lambda x: -x['variance'])
        gene_var_clusters[gene] = var_clusters[:max_var_exons]
    
    print("  Total clusters: constitutive={}, variable={}".format(
        total_constitutive, total_variable))
    
    # 실제 사용 가변 cluster 수 통계
    var_counts = [len(v) for v in gene_var_clusters.values()]
    if var_counts:
        print("  Variable clusters per gene: mean={:.1f}, median={:.0f}, max={}, p95={:.0f}".format(
            np.mean(var_counts), np.median(var_counts),
            max(var_counts), np.percentile(var_counts, 95)))
    
    # 실제 필요 차원 결정
    actual_max = max(var_counts) if var_counts else 0
    dim = min(actual_max, max_var_exons)
    print("  Matrix dimension: {} (capped at {})".format(dim, max_var_exons))
    
    # Matrix 구축
    exon_matrix = np.zeros((n, dim), dtype=np.float32)
    
    for gene, idxs in gene_to_idxs.items():
        var_cls = gene_var_clusters.get(gene, [])
        
        for col_idx, vc in enumerate(var_cls):
            if col_idx >= dim:
                break
            
            cls_exons = vc['exons']
            cluster_max_len = vc['max_len']
            
            for iso_idx in idxs:
                iso_exons = iso_exon_map.get(iso_idx, set())
                overlapping = cls_exons & iso_exons
                
                if overlapping:
                    # 가장 긴 overlap exon의 길이 / cluster 최대 길이
                    max_overlap_len = max(e[2] - e[1] + 1 for e in overlapping)
                    exon_matrix[iso_idx, col_idx] = max_overlap_len / cluster_max_len
                # else: 0.0 (default)
    
    print("[ExonMatrix] shape={} | non-zero rows={}".format(
        exon_matrix.shape, (exon_matrix.sum(axis=1) > 0).sum()))
    
    return exon_matrix, gene_var_clusters, dim


# ── Splicing Delta ───────────────────────────────────────────────────

def compute_splicing_delta_v2(exon_matrix, gene_list, gene_to_canonical):
    """splicing_delta[i] = exon_matrix[i] - exon_matrix[canonical_i]"""
    n = exon_matrix.shape[0]
    delta = np.zeros_like(exon_matrix, dtype=np.float32)
    
    no_canon = 0
    for i in range(n):
        gene = gene_list[i]
        can_info = gene_to_canonical.get(gene)
        
        if can_info is None:
            no_canon += 1
            continue
        
        can_idx = can_info[0]
        delta[i] = exon_matrix[i] - exon_matrix[can_idx]
    
    n_nonzero = (np.abs(delta).sum(axis=1) != 0).sum()
    print("[SpliceDelta] Non-zero rows: {} ({:.1f}%) | No-canonical: {}".format(
        n_nonzero, n_nonzero / n * 100, no_canon))
    
    return delta


# ── Canonical Reference 로드 ────────────────────────────────────────

def load_canonical_reference(ref_path):
    """canonical_reference.tsv 로드"""
    gene_to_canonical = {}
    with open(ref_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                gene_versioned = parts[1]
                canonical_idx  = int(parts[3])
                canonical_src  = parts[4]
                canonical_iso  = parts[5]
                gene_to_canonical[gene_versioned] = (canonical_idx, canonical_iso, canonical_src)
    return gene_to_canonical


# ── 메인 ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Step 3: Exon Matrix v2 (continuous, cluster-based)")
    print("=" * 60)
    
    # 1. Load
    iso_list = np.load(ISO_LIST_FILE, allow_pickle=True)
    iso_list = [s.decode() if isinstance(s, bytes) else s for s in iso_list]
    
    gene_list = np.load('../model/my_gene_list_fixed.npy', allow_pickle=True)
    gene_list = [g.decode() if isinstance(g, bytes) else g for g in gene_list]
    
    print("[Load] {} isoforms, {} unique genes".format(len(iso_list), len(set(gene_list))))
    
    # 2. GTF 파싱
    tx_exons, tx_gene_map = parse_gtf_exons(GTF_FILE)
    
    # 3. Exon matrix v2
    exon_matrix, gene_var_clusters, dim = build_exon_matrix_v2(
        iso_list, tx_exons, tx_gene_map, gene_list,
        max_var_exons=MAX_VAR_EXONS)
    
    # 4. Canonical 로드
    gene_to_canonical = load_canonical_reference(CANONICAL_REF)
    print("[Canonical] Loaded {} gene → canonical".format(len(gene_to_canonical)))
    
    # 5. Splicing delta v2
    splice_delta = compute_splicing_delta_v2(exon_matrix, gene_list, gene_to_canonical)
    
    # 6. 저장
    np.save(OUT_EXON_MATRIX, exon_matrix)
    np.save(OUT_SPLICE_DELTA, splice_delta)
    print("[Save] exon_matrix_v2.npy: {}".format(exon_matrix.shape))
    print("[Save] splicing_delta_v2.npy: {}".format(splice_delta.shape))
    
    # 7. 메타 저장
    gene_names = sorted(set(gene_list))
    gene_n_var = np.array([len(gene_var_clusters.get(g, [])) for g in gene_names])
    np.savez(OUT_META,
             gene_names=np.array(gene_names, dtype=object),
             gene_n_var_exon=gene_n_var,
             dim=dim)
    
    # 8. 통계
    stats_lines = []
    stats_lines.append("Exon Matrix v2 Statistics")
    stats_lines.append("========================")
    stats_lines.append("Shape:               {}".format(exon_matrix.shape))
    stats_lines.append("Non-zero rows:       {}".format((exon_matrix.sum(axis=1) > 0).sum()))
    stats_lines.append("Value range:         [{:.3f}, {:.3f}]".format(
        exon_matrix[exon_matrix > 0].min() if (exon_matrix > 0).any() else 0,
        exon_matrix.max()))
    stats_lines.append("Fractional values (0<x<1): {}".format(
        ((exon_matrix > 0) & (exon_matrix < 1)).sum()))
    stats_lines.append("")
    stats_lines.append("Splicing Delta v2 Statistics")
    stats_lines.append("===========================")
    stats_lines.append("Non-zero delta rows: {}".format(
        (np.abs(splice_delta).sum(axis=1) != 0).sum()))
    stats_lines.append("Mean |delta|:        {:.4f}".format(np.abs(splice_delta).mean()))
    stats_lines.append("Fractional deltas:   {}".format(
        ((np.abs(splice_delta) > 0) & (np.abs(splice_delta) < 1)).sum()))
    
    with open(OUT_STATS, 'w') as f:
        f.write('\n'.join(stats_lines) + '\n')
    print("[Stats] Written to {}".format(OUT_STATS))
    
    # 9. v1과 비교
    old_sd_path = OUT_DIR + 'splicing_delta.npy'
    if os.path.exists(old_sd_path):
        old_sd = np.load(old_sd_path)
        old_s0 = np.all(old_sd == 0, axis=1)
        new_s0 = np.all(splice_delta == 0, axis=1)
        
        print("\n[Comparison] vs splicing_delta v1:")
        print("  Old S0: {}".format(old_s0.sum()))
        print("  New S0: {}".format(new_s0.sum()))
        print("  S0→S1 (newly detected): {}".format((old_s0 & ~new_s0).sum()))
        print("  S1→S0 (no longer detected): {}".format((~old_s0 & new_s0).sum()))
    
    for line in stats_lines:
        print("  " + line)
    
    print("\n[Done]")


if __name__ == '__main__':
    main()
