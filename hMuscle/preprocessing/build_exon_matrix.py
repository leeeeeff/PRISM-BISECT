# -*- coding: utf-8 -*-
"""
build_exon_matrix.py
====================
Alt 4: Splicing Pattern (Exon Inclusion) 특징 계산

각 유전자의 고유 exon 집합을 기준으로 isoform × exon binary matrix 생성.
canonical isoform과의 delta(차이) 계산.

출력:
  ../results/splicing/exon_matrix.npy         (36748, MAX_EXONS) int8   — binary 포함 여부
  ../results/splicing/splicing_delta.npy       (36748, MAX_EXONS) float32 — canonical과의 차이
  ../results/splicing/exon_meta.npz            gene별 exon 메타정보
  ../results/splicing/splicing_stats.txt       요약 통계

파라미터:
  MAX_EXONS = 50  (유전자당 최대 exon 수, 초과 시 상위 50개 선택)

실행:
  cd hMuscle/preprocessing/
  python build_exon_matrix.py
"""

import numpy as np
import re
import os
from collections import defaultdict

# ── 경로 ──────────────────────────────────────────────────────────────
ISO_LIST_FILE    = '../model/my_isoform_list_fixed.npy'
GTF_FILE         = '../data/cleaned_annotations.gtf'
ISO_GENE_MAP     = '../results_isoform/features/iso_gene_map.txt'  # build_domain_delta.py 출력 재사용

OUT_DIR          = '../results_isoform/features/splicing/'
OUT_EXON_MATRIX  = OUT_DIR + 'exon_matrix.npy'
OUT_SPLICE_DELTA = OUT_DIR + 'splicing_delta.npy'
OUT_META         = OUT_DIR + 'exon_meta.npz'
OUT_STATS        = OUT_DIR + 'splicing_stats.txt'

MAX_EXONS = 50   # 유전자당 최대 exon 수 (p95 커버)


# ── GTF 파싱 ──────────────────────────────────────────────────────────

def parse_gtf_exons(gtf_path):
    """
    GTF에서 transcript별 exon 목록 파싱.
    Returns:
      tx_exons: dict {transcript_id: [(chrom, start, end), ...]}
      tx_gene:  dict {transcript_id: gene_id}
    """
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


# ── Gene별 exon 정규화 ─────────────────────────────────────────────────

def build_gene_exon_universe(tx_exons, tx_gene, isoform_ids):
    """
    각 유전자의 고유 exon 집합을 구축.
    (chrom, start, end) 튜플을 정렬된 인덱스로 매핑.

    Returns:
      gene_exon_idx: {gene_id: {exon_tuple: local_index}}
      gene_exon_list: {gene_id: [exon_tuples...]} (정렬됨)
    """
    print("[Universe] Building gene-level exon universe ...")

    # 대상 isoform의 gene 목록 (GTF 기반 전체 사용)
    gene_exons = defaultdict(set)
    for tx_id, exons in tx_exons.items():
        gene = tx_gene.get(tx_id)
        if gene is None:
            continue
        for exon in exons:
            gene_exons[gene].add(exon)

    # 각 gene의 exon을 start 기준 정렬
    gene_exon_list = {}
    gene_exon_idx  = {}
    for gene, exon_set in gene_exons.items():
        sorted_exons = sorted(exon_set, key=lambda x: (x[0], x[1], x[2]))
        gene_exon_list[gene] = sorted_exons
        gene_exon_idx[gene]  = {e: i for i, e in enumerate(sorted_exons)}

    exon_counts = [len(v) for v in gene_exon_list.values()]
    print("[Universe] {} genes | exon count: min={} max={} median={:.0f} p95={:.0f}".format(
        len(gene_exon_list),
        min(exon_counts) if exon_counts else 0,
        max(exon_counts) if exon_counts else 0,
        np.median(exon_counts) if exon_counts else 0,
        np.percentile(exon_counts, 95) if exon_counts else 0
    ))

    return gene_exon_list, gene_exon_idx


# ── Exon matrix 구축 ──────────────────────────────────────────────────

def build_exon_matrix(iso_list, tx_exons, tx_gene_map, gene_exon_list,
                      gene_exon_idx, max_exons=MAX_EXONS):
    """
    isoform × exon binary matrix 생성.
    각 유전자별 exon에 max_exons 개까지 binary 포함 여부.

    전략: 유전자별 exon이 max_exons 초과 시
      → 해당 유전자 내 most variable exons (isoforms 간 최대 분산) 상위 max_exons 선택
    """
    n = len(iso_list)
    exon_matrix = np.zeros((n, max_exons), dtype=np.int8)

    # gene별 선택 exon 인덱스 캐싱 (가변 exon 선택)
    gene_selected_exons = {}
    missing_tx = 0

    def get_selected_exons(gene):
        """유전자의 exon 중 max_exons개 선택 (분산 기반 or 처음 max_exons)"""
        if gene in gene_selected_exons:
            return gene_selected_exons[gene]
        exon_list = gene_exon_list.get(gene, [])
        if len(exon_list) <= max_exons:
            selected = list(range(len(exon_list)))
        else:
            # 모든 isoform에서 각 exon의 포함 여부 수집
            gene_txs = [tx for tx, g in tx_gene_map.items() if g == gene]
            exon_variance = []
            for local_idx, exon in enumerate(exon_list):
                inclusion = [1 if exon in set(tx_exons.get(tx, [])) else 0
                             for tx in gene_txs]
                var = np.var(inclusion) if inclusion else 0
                exon_variance.append((var, local_idx))
            # 분산 높은 상위 max_exons 선택
            exon_variance.sort(reverse=True)
            selected = sorted([idx for _, idx in exon_variance[:max_exons]])
        gene_selected_exons[gene] = selected
        return selected

    for i, iso_bytes in enumerate(iso_list):
        iso_id = iso_bytes.decode() if isinstance(iso_bytes, bytes) else iso_bytes
        gene = tx_gene_map.get(iso_id)

        if gene is None:
            # 버전 매칭 시도
            base = iso_id.split('.')[0]
            for k, v in tx_gene_map.items():
                if k.startswith(base):
                    gene = v
                    iso_id_for_exon = k
                    break
            else:
                missing_tx += 1
                continue
        else:
            iso_id_for_exon = iso_id

        # 이 isoform의 exon 집합
        this_exons = set(tx_exons.get(iso_id_for_exon, []))
        exon_list  = gene_exon_list.get(gene, [])
        selected   = get_selected_exons(gene)

        for bit_pos, local_idx in enumerate(selected):
            if local_idx < len(exon_list):
                exon = exon_list[local_idx]
                exon_matrix[i, bit_pos] = 1 if exon in this_exons else 0

    print("[ExonMatrix] shape={} | missing transcripts={}".format(
        exon_matrix.shape, missing_tx))
    included_rows = (exon_matrix.sum(axis=1) > 0).sum()
    print("[ExonMatrix] Rows with ≥1 exon included: {}".format(included_rows))
    return exon_matrix, gene_selected_exons


# ── Splicing delta 계산 ───────────────────────────────────────────────

def compute_splicing_delta(iso_list, tx_gene_map, exon_matrix, gene_to_canonical):
    """
    splicing_delta[i] = exon_matrix[i] - exon_matrix[canonical_i]
    canonical = domain delta와 동일 기준 (domain 수 최대)
    """
    n = len(iso_list)
    delta = np.zeros((n, exon_matrix.shape[1]), dtype=np.float32)

    no_canon = 0
    for i, iso_bytes in enumerate(iso_list):
        iso_id = iso_bytes.decode() if isinstance(iso_bytes, bytes) else iso_bytes
        gene   = tx_gene_map.get(iso_id)
        if gene is None:
            # 버전 매칭
            base = iso_id.split('.')[0]
            for k, v in tx_gene_map.items():
                if k.startswith(base):
                    gene = v
                    break

        if gene is None:
            no_canon += 1
            continue
        can_idx = gene_to_canonical.get(gene)
        if can_idx is None:
            no_canon += 1
            continue
        delta[i] = exon_matrix[i].astype(np.float32) - exon_matrix[can_idx].astype(np.float32)

    print("[SpliceDelta] Non-zero rows: {} | No-canonical: {}".format(
        (delta.sum(axis=1) != 0).sum(), no_canon))
    return delta


def load_iso_gene_map(iso_gene_map_file):
    """build_domain_delta.py가 생성한 iso_gene_map.txt 로드"""
    if not os.path.exists(iso_gene_map_file):
        return None
    mapping = {}
    with open(iso_gene_map_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3 and parts[2] != 'UNKNOWN':
                mapping[parts[1]] = parts[2]
    print("[Map] Loaded {} iso→gene mappings from {}".format(
        len(mapping), iso_gene_map_file))
    return mapping


def load_canonical_map(canonical_file):
    """build_domain_delta.py가 생성한 canonical_map.txt 로드"""
    if not os.path.exists(canonical_file):
        return None
    mapping = {}
    with open(canonical_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                mapping[parts[0]] = int(parts[1])
    print("[Canonical] Loaded {} gene→canonical mappings".format(len(mapping)))
    return mapping


# ── 메인 ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 로드
    print("[Load] Isoform list: {}".format(ISO_LIST_FILE))
    iso_list = np.load(ISO_LIST_FILE, allow_pickle=True)
    print("  shape: {}".format(iso_list.shape))

    # 2. GTF 파싱
    tx_exons, tx_gene_map = parse_gtf_exons(GTF_FILE)

    # 3. iso→gene 매핑 (build_domain_delta 결과 재사용 or GTF에서 직접)
    if os.path.exists(ISO_GENE_MAP):
        saved_map = load_iso_gene_map(ISO_GENE_MAP)
        if saved_map:
            tx_gene_map.update(saved_map)

    # 4. gene별 exon universe 구축
    gene_exon_list, gene_exon_idx = build_gene_exon_universe(
        tx_exons, tx_gene_map, iso_list)

    # 5. exon matrix 생성
    exon_matrix, gene_selected_exons = build_exon_matrix(
        iso_list, tx_exons, tx_gene_map, gene_exon_list, gene_exon_idx,
        max_exons=MAX_EXONS)

    # 6. canonical isoform 로드 (domain delta 재사용)
    canonical_map_file = '../results_isoform/features/canonical_map.txt'
    gene_to_canonical = load_canonical_map(canonical_map_file)
    if gene_to_canonical is None:
        print("[WARN] canonical_map.txt not found. Run build_domain_delta.py first.")
        print("       Falling back: canonical = first isoform per gene")
        from collections import defaultdict
        gene_first = {}
        for i, iso_bytes in enumerate(iso_list):
            iso_id = iso_bytes.decode() if isinstance(iso_bytes, bytes) else iso_bytes
            gene   = tx_gene_map.get(iso_id)
            if gene and gene not in gene_first:
                gene_first[gene] = i
        gene_to_canonical = gene_first

    # 7. splicing delta 계산
    splice_delta = compute_splicing_delta(
        iso_list, tx_gene_map, exon_matrix, gene_to_canonical)

    # 8. 저장
    np.save(OUT_EXON_MATRIX,  exon_matrix.astype(np.int8))
    np.save(OUT_SPLICE_DELTA, splice_delta.astype(np.float32))
    print("[Save] exon_matrix.npy:    {}".format(exon_matrix.shape))
    print("[Save] splicing_delta.npy: {}".format(splice_delta.shape))

    # 9. 메타 저장
    gene_names  = list(gene_exon_list.keys())
    gene_n_exon = np.array([len(gene_exon_list[g]) for g in gene_names])
    np.savez(OUT_META,
             gene_names=np.array(gene_names, dtype=object),
             gene_n_exon=gene_n_exon)
    print("[Save] exon_meta.npz saved")

    # 10. 통계
    with open(OUT_STATS, 'w') as f:
        f.write("Exon Matrix Statistics\n")
        f.write("======================\n")
        f.write("Shape:             {}\n".format(exon_matrix.shape))
        f.write("Non-zero rows:     {}\n".format((exon_matrix.sum(axis=1) > 0).sum()))
        f.write("Max exons per row: {}\n".format(exon_matrix.sum(axis=1).max()))
        f.write("Mean exons/row:    {:.2f}\n".format(exon_matrix.sum(axis=1).mean()))
        f.write("\nSplicing Delta Statistics\n")
        f.write("=========================\n")
        f.write("Non-zero delta rows: {}\n".format((splice_delta.sum(axis=1) != 0).sum()))
        f.write("Mean |delta|:        {:.4f}\n".format(np.abs(splice_delta).mean()))
    print("[Stats] Written to {}".format(OUT_STATS))

    # 11. 검증 샘플
    print("\n[Sanity Check] First 5 isoforms:")
    for i in range(5):
        iso_id = iso_list[i].decode() if isinstance(iso_list[i], bytes) else iso_list[i]
        n_exon = int(exon_matrix[i].sum())
        n_delta_nz = int((splice_delta[i] != 0).sum())
        gene = tx_gene_map.get(iso_id, 'UNKNOWN')
        print("  [{}] {} | gene={} | exons={} | delta_nz={}".format(
            i, iso_id, gene, n_exon, n_delta_nz))

    print("\n[Done] Splicing features saved to {}".format(OUT_DIR))


if __name__ == '__main__':
    main()
