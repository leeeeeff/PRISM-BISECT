# -*- coding: utf-8 -*-
"""
build_train_splicing_delta.py
==============================
Step 2-4: Train NM_ ID → 공유 Exon Cluster 공간 → splicing_delta 생성

파이프라인:
  1. GENCODE v44 GTF 파싱 → gene_name 기준 exon 좌표
  2. 공유 exon cluster space 구축 (gene별 150-dim, test와 동일 차원)
  3. NM_ → ENST (nm_to_enst_mapping.tsv 활용)
  4. ENST exon 좌표 → cluster 할당
  5. MANE canonical 기준 splicing_delta 계산

출력:
  ../results_isoform/features/splicing/train_exon_matrix.npy    (31668, 150)
  ../results_isoform/features/splicing/train_splicing_delta.npy (31668, 150)
  ../results_isoform/features/splicing/train_splicing_stats.txt

실행:
  python build_nm_enst_mapping.py   # 먼저 실행 (Step 1)
  python build_train_splicing_delta.py
"""

import gzip
import os
import re
from collections import defaultdict
import numpy as np

# ── 경로 ──────────────────────────────────────────────────────────────
TRAIN_ISO_FILE  = '../data/raw_data/data/id_lists/train_isoform_list.npy'
TRAIN_GENE_FILE = '../data/raw_data/data/id_lists/train_gene_list.npy'
GENCODE_GTF     = '../data/refdata-gex-GRCh38-2024-A/genes/genes.gtf.gz'
MANE_FILE       = '../data/MANE_summary.txt.gz'
NM_ENST_MAP     = '../results_isoform/features/splicing/nm_to_enst_mapping.tsv'

OUT_DIR         = '../results_isoform/features/splicing/'
OUT_EXON_MAT    = OUT_DIR + 'train_exon_matrix.npy'
OUT_SPLICE_DELTA= OUT_DIR + 'train_splicing_delta.npy'
OUT_STATS       = OUT_DIR + 'train_splicing_stats.txt'

MAX_VAR_EXONS = 150


# ── Step A: NM_→ENST 매핑 로드 ────────────────────────────────────────

def load_nm_enst_mapping(map_file):
    """nm_to_enst_mapping.tsv → {nm_base: enst_base}"""
    if not os.path.exists(map_file):
        raise FileNotFoundError(f"Mapping not found: {map_file}\n"
                                f"Run build_nm_enst_mapping.py first.")
    mapping = {}
    with open(map_file) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[1]:
                mapping[parts[0]] = parts[1]  # nm_id → enst_id
    print(f"[NM→ENST] Loaded {len(mapping)} valid mappings")
    return mapping


# ── Step B: GENCODE v44 파싱 ───────────────────────────────────────────

def parse_gencode_gtf(gtf_path, enst_ids_needed=None):
    """
    GENCODE v44 GTF → transcript_id별 exon 좌표 목록
    Args:
        enst_ids_needed: set of ENST IDs to include (None = all)
    Returns:
        tx_exons: {enst_base: [(chrom, start, end), ...]}
        tx_gene:  {enst_base: gene_name}
    """
    print(f"[GENCODE] Parsing {gtf_path} ...")
    tx_exons = defaultdict(list)
    tx_gene  = {}

    opener = gzip.open if gtf_path.endswith('.gz') else open
    with opener(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            feature = parts[2]
            attrs   = parts[8]

            m_tx   = re.search(r'transcript_id "([^"]+)"', attrs)
            m_gene = re.search(r'gene_name "([^"]+)"', attrs)
            if not m_tx or not m_gene:
                continue

            enst_ver  = m_tx.group(1)
            enst_base = enst_ver.split('.')[0]
            gene_name = m_gene.group(1)

            if enst_ids_needed and enst_base not in enst_ids_needed:
                continue

            tx_gene[enst_base] = gene_name

            if feature == 'exon':
                chrom = parts[0]
                start = int(parts[3])
                end   = int(parts[4])
                tx_exons[enst_base].append((chrom, start, end))

    print(f"[GENCODE] {len(tx_gene)} transcripts | {len(tx_exons)} with exons")
    return tx_exons, tx_gene


# ── Step C: MANE canonical 로드 ───────────────────────────────────────

def load_mane_canonical(mane_file):
    """
    MANE_summary.txt.gz → {gene_name: enst_base} (MANE Select canonical per gene)
    """
    canon = {}
    with gzip.open(mane_file, 'rt') as f:
        header = f.readline().strip().split('\t')
        enst_col = header.index('Ensembl_nuc')
        sym_col  = header.index('symbol')
        for line in f:
            parts = line.strip().split('\t')
            sym   = parts[sym_col]
            enst  = parts[enst_col].split('.')[0]
            if enst.startswith('ENST'):
                canon[sym] = enst
    print(f"[MANE] Loaded {len(canon)} gene→canonical ENST mappings")
    return canon


# ── Step D: 유전자별 Exon Cluster 구축 ────────────────────────────────

def build_exon_clusters(exon_set):
    """
    Genomic overlap 기반 exon clustering (build_exon_matrix_v2.py 로직 재사용).
    겹치는 exons → 하나의 cluster로 묶음.
    Returns: list of frozensets, each containing (chrom,start,end) tuples
    """
    if not exon_set:
        return []

    chrom_exons = defaultdict(list)
    for exon in exon_set:
        chrom_exons[exon[0]].append(exon)

    clusters = []
    for chrom, exons in chrom_exons.items():
        exons.sort(key=lambda x: x[1])
        current_cluster = [exons[0]]
        current_end = exons[0][2]

        for ex in exons[1:]:
            if ex[1] <= current_end:
                current_cluster.append(ex)
                current_end = max(current_end, ex[2])
            else:
                clusters.append(frozenset(current_cluster))
                current_cluster = [ex]
                current_end = ex[2]
        clusters.append(frozenset(current_cluster))

    return clusters


def build_gene_cluster_space(gene_name, gene_enst_list, tx_exons, max_var=MAX_VAR_EXONS):
    """
    하나의 유전자에 대해 가변 exon cluster 공간 구축.
    Args:
        gene_name:      gene symbol
        gene_enst_list: [enst_base, ...] — 이 유전자의 모든 ENST 이소폼
        tx_exons:       {enst_base: [(chrom,start,end),...]}
        max_var:        최대 가변 exon cluster 수
    Returns:
        var_clusters:   list of dicts {exons, max_len, variance} (최대 max_var개)
    """
    if not gene_enst_list:
        return []

    # 이 유전자의 모든 exon 수집
    all_exons = set()
    for enst in gene_enst_list:
        for ex in tx_exons.get(enst, []):
            all_exons.add(ex)

    clusters = build_exon_clusters(all_exons)
    if not clusters:
        return []

    # 각 cluster가 각 isoform에 포함되는지 계산 → 분산 기반 variable 선택
    total_constitutive = 0
    total_variable = 0
    var_clusters = []

    for cluster in clusters:
        iso_patterns = []
        for enst in gene_enst_list:
            iso_exons = set(tx_exons.get(enst, []))
            iso_patterns.append(1 if cluster & iso_exons else 0)

        if len(set(iso_patterns)) == 1:
            total_constitutive += 1
            continue

        total_variable += 1
        cluster_max_len = max(ex[2] - ex[1] + 1 for ex in cluster)
        var_clusters.append({
            'exons':     cluster,
            'max_len':   cluster_max_len,
            'variance':  float(np.var(iso_patterns)),
        })

    # variance 기준 정렬, 상위 max_var 선택
    var_clusters.sort(key=lambda x: -x['variance'])
    return var_clusters[:max_var]


# ── Step E: NM_ 이소폼을 cluster 공간에 할당 ──────────────────────────

def assign_to_clusters(iso_exons_set, var_clusters):
    """
    하나의 이소폼(iso_exons_set)을 가변 exon cluster 공간에 할당.
    cluster와 overlap 시 exon_len/max_len (continuous, 0~1).
    Returns: np.array shape=(len(var_clusters),) float32
    """
    vec = np.zeros(len(var_clusters), dtype=np.float32)
    for col_idx, vc in enumerate(var_clusters):
        cls_exons    = vc['exons']
        overlapping  = cls_exons & iso_exons_set
        if overlapping:
            max_overlap_len = max(ex[2] - ex[1] + 1 for ex in overlapping)
            vec[col_idx] = max_overlap_len / vc['max_len']
    return vec


# ── 메인 ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Train Splicing Delta — Shared GENCODE Exon Cluster Space")
    print("=" * 60)

    # 1. Train 데이터 로드
    train_iso  = np.load(TRAIN_ISO_FILE,  allow_pickle=True)
    train_gene = np.load(TRAIN_GENE_FILE, allow_pickle=True)
    train_iso  = [x.decode() if isinstance(x, bytes) else x for x in train_iso]
    train_gene = [x.decode() if isinstance(x, bytes) else x for x in train_gene]
    n = len(train_iso)
    print(f"[Load] {n} train isoforms, {len(set(train_gene))} unique genes")

    # NM_ base ID 추출 (버전 제거, 쉼표 구분 지원)
    # "NM_001206729,NM_181690" → ["NM_001206729", "NM_181690"]
    train_iso_parts = []  # list of lists
    for x in train_iso:
        parts = [nm.strip().split('.')[0] for nm in x.split(',')]
        train_iso_parts.append(parts)

    # 2. NM_→ENST 매핑 로드
    nm_to_enst = load_nm_enst_mapping(NM_ENST_MAP)

    def resolve_nm(nm_parts_list):
        """쉼표 구분 NM_ ID 목록 중 첫 번째 매핑 반환."""
        for nm_base in nm_parts_list:
            enst = nm_to_enst.get(nm_base)
            if enst:
                return enst
        return None

    # 3. 매핑된 ENST IDs 수집 (필요한 것만 GTF에서 파싱)
    enst_needed = set(nm_to_enst.values())
    print(f"[ENST needed] {len(enst_needed)} unique ENST IDs")

    # 4. GENCODE v44 파싱 (필요한 ENST만)
    tx_exons, tx_gene = parse_gencode_gtf(GENCODE_GTF, enst_ids_needed=enst_needed)

    # 5. gene별 ENST 그룹핑 (gene_name 기준)
    gene_to_enst = defaultdict(list)
    for enst, gname in tx_gene.items():
        gene_to_enst[gname].append(enst)
    print(f"[Gene groups] {len(gene_to_enst)} genes with ENST data")

    # 6. MANE canonical
    mane_canonical = load_mane_canonical(MANE_FILE)

    # 7. gene별 cluster space 구축 + train 이소폼 할당
    # 메모리 효율: gene별로 처리 후 결과 집계
    exon_matrix   = np.zeros((n, MAX_VAR_EXONS), dtype=np.float32)
    gene_clusters = {}   # {gene_name: var_clusters_list}

    print("[Building exon matrix] Processing genes ...")
    gene_list_unique = sorted(set(train_gene))
    progress_step = max(1, len(gene_list_unique) // 20)

    for gi, gene_name in enumerate(gene_list_unique):
        if gi % progress_step == 0:
            print(f"  [{gi}/{len(gene_list_unique)}] {gene_name} ...", flush=True)

        # 이 유전자의 모든 GENCODE ENST
        gene_enst_all = gene_to_enst.get(gene_name, [])
        if not gene_enst_all:
            continue

        # 가변 exon cluster 구축
        var_clusters = build_gene_cluster_space(
            gene_name, gene_enst_all, tx_exons, max_var=MAX_VAR_EXONS)
        gene_clusters[gene_name] = var_clusters
        if not var_clusters:
            continue

        # 이 유전자의 train 이소폼들을 cluster 공간에 할당
        dim = len(var_clusters)
        for iso_idx in [i for i, g in enumerate(train_gene) if g == gene_name]:
            enst_id = resolve_nm(train_iso_parts[iso_idx])
            if not enst_id:
                continue
            iso_exons_list = tx_exons.get(enst_id, [])
            if not iso_exons_list:
                continue
            iso_exons_set = frozenset(iso_exons_list)
            vec = assign_to_clusters(iso_exons_set, var_clusters)
            exon_matrix[iso_idx, :dim] = vec[:dim]

    print(f"[ExonMatrix] shape={exon_matrix.shape} | "
          f"non-zero rows={(exon_matrix.sum(axis=1) > 0).sum()}")

    # 8. Canonical index 결정 (train 이소폼 중 MANE Select)
    gene_to_canon_idx = {}
    for gene_name in set(train_gene):
        # MANE canonical ENST
        mane_enst = mane_canonical.get(gene_name)
        # train 이소폼 중 MANE canonical 매핑
        best_idx = None
        for i, g in enumerate(train_gene):
            if g != gene_name:
                continue
            enst_id = resolve_nm(train_iso_parts[i])
            if mane_enst and enst_id == mane_enst:
                best_idx = i
                break
        # MANE not found → max exon count isoform
        if best_idx is None:
            gene_idxs = [i for i, g in enumerate(train_gene) if g == gene_name]
            if gene_idxs:
                best_idx = max(gene_idxs,
                               key=lambda i: exon_matrix[i].sum())
        gene_to_canon_idx[gene_name] = best_idx

    # 9. Splicing delta 계산
    splice_delta = np.zeros_like(exon_matrix)
    no_canon = 0
    for i in range(n):
        gene_name = train_gene[i]
        can_idx   = gene_to_canon_idx.get(gene_name)
        if can_idx is None:
            no_canon += 1
            continue
        splice_delta[i] = exon_matrix[i] - exon_matrix[can_idx]

    n_nonzero = (np.abs(splice_delta).sum(axis=1) != 0).sum()
    print(f"[SpliceDelta] Non-zero rows: {n_nonzero} ({n_nonzero/n*100:.1f}%) | "
          f"No-canonical: {no_canon}")

    # 10. 저장
    np.save(OUT_EXON_MAT,     exon_matrix.astype(np.float32))
    np.save(OUT_SPLICE_DELTA, splice_delta.astype(np.float32))
    print(f"[Save] {OUT_EXON_MAT}: {exon_matrix.shape}")
    print(f"[Save] {OUT_SPLICE_DELTA}: {splice_delta.shape}")

    # 11. Match rate 통계
    mapped_count = sum(
        1 for parts in train_iso_parts if resolve_nm(parts)
    )
    exon_covered = int((exon_matrix.sum(axis=1) > 0).sum())
    delta_nonzero = int(n_nonzero)

    stats = [
        "Train Splicing Delta Statistics",
        "================================",
        f"Total train isoforms:          {n}",
        f"NM_→ENST mapped:               {mapped_count} ({mapped_count/n*100:.1f}%)",
        f"NM_→ENST unmapped (zero-filled):{n-mapped_count} ({(n-mapped_count)/n*100:.1f}%)",
        f"Exon matrix non-zero rows:     {exon_covered} ({exon_covered/n*100:.1f}%)",
        f"Splicing delta non-zero rows:  {delta_nonzero} ({delta_nonzero/n*100:.1f}%)",
        f"Matrix shape:                  {exon_matrix.shape}",
        f"Mean |delta|:                  {np.abs(splice_delta).mean():.4f}",
        f"Max |delta|:                   {np.abs(splice_delta).max():.4f}",
    ]
    with open(OUT_STATS, 'w') as f:
        f.write('\n'.join(stats) + '\n')
    for s in stats:
        print(f"  {s}")

    print("\n[Done] train_splicing_delta.npy ready for v10-E integration")


if __name__ == '__main__':
    main()
