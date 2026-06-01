# -*- coding: utf-8 -*-
"""
build_canonical_reference.py
============================
Step 1: GENCODE v44 기반 canonical isoform 참조 테이블 생성

3-tier canonical 결정:
  Priority 1: MANE_Select (NCBI/EBI joint, 임상 표준)
  Priority 2: Ensembl_canonical (GENCODE/Ensembl 전산)
  Priority 3: Longest CDS in dataset (fallback)

입력:
  GENCODE v44 GTF (genes.gtf.gz)
  my_isoform_list_fixed.npy
  my_gene_list_fixed.npy
  top30k_isoforms.pep (CDS 길이 산출용)

출력:
  ../results_isoform/features/canonical_reference.tsv
    columns: gene_base | gene_versioned | canonical_enst_base |
             canonical_iso_idx | canonical_source | canonical_iso_id

실행:
  cd hMuscle/preprocessing/
  python build_canonical_reference.py
"""

import numpy as np
import re
import os
import gzip
from collections import defaultdict

# ── 경로 ──────────────────────────────────────────────────────────────
GENCODE_GTF   = '../data/refdata-gex-GRCh38-2024-A/genes/genes.gtf.gz'
ISO_LIST_FILE = '../model/my_isoform_list_fixed.npy'
GENE_LIST_FILE = '../model/my_gene_list_fixed.npy'
PEP_FILE      = '../data/top30k_isoforms.pep'
OUT_FILE      = '../results_isoform/features/canonical_reference.tsv'


# ── 1. GENCODE v44 파싱 ──────────────────────────────────────────────

def parse_gencode_canonical(gtf_gz_path):
    """
    GENCODE GTF에서 각 유전자의 MANE_Select / Ensembl_canonical transcript 추출.
    
    Returns:
        gene_mane:      {gene_base: enst_base}  — MANE_Select
        gene_ensembl:   {gene_base: enst_base}  — Ensembl_canonical
        gene_appris:    {gene_base: enst_base}  — appris_principal_1 (추가 fallback)
    """
    print("[GENCODE] Parsing {} ...".format(gtf_gz_path))
    
    gene_mane = {}
    gene_ensembl = {}
    gene_appris = {}
    n_transcripts = 0
    
    with gzip.open(gtf_gz_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if '\ttranscript\t' not in line:
                continue
            
            n_transcripts += 1
            
            m_gene = re.search(r'gene_id "([^"]+)"', line)
            m_tx   = re.search(r'transcript_id "([^"]+)"', line)
            if not m_gene or not m_tx:
                continue
            
            gene_base = m_gene.group(1).split('.')[0]
            tx_base   = m_tx.group(1).split('.')[0]
            
            if 'MANE_Select' in line:
                gene_mane[gene_base] = tx_base
            
            if 'Ensembl_canonical' in line:
                gene_ensembl[gene_base] = tx_base
            
            # APPRIS: appris_principal_1 > appris_principal_2 > ...
            if 'appris_principal_1' in line and gene_base not in gene_appris:
                gene_appris[gene_base] = tx_base
    
    print("  Transcripts scanned: {}".format(n_transcripts))
    print("  MANE_Select: {} genes".format(len(gene_mane)))
    print("  Ensembl_canonical: {} genes".format(len(gene_ensembl)))
    print("  APPRIS_principal_1: {} genes".format(len(gene_appris)))
    
    return gene_mane, gene_ensembl, gene_appris


# ── 2. PEP 파일에서 CDS 길이 추출 ───────────────────────────────────

def parse_pep_lengths(pep_path):
    """
    PEP 파일에서 isoform별 최장 ORF 길이(아미노산 수) 반환.
    
    Returns:
        iso_cds_len: {isoform_id: int (aa length)}
    """
    print("[PEP] Parsing {} ...".format(pep_path))
    
    TYPE_RANK = {'complete': 4, '5prime_partial': 3, '3prime_partial': 2, 'internal': 1}
    records = {}  # isoform_id → (rank, score, length, seq_len)
    
    cur_id = None
    cur_meta = None
    cur_seq = []
    
    def flush():
        nonlocal cur_id, cur_meta, cur_seq
        if cur_id is None:
            return
        seq = ''.join(cur_seq).replace('*', '').strip()
        if not seq:
            return
        rank, score, length = cur_meta
        seq_len = len(seq)
        prev = records.get(cur_id)
        if prev is None or (rank, score, seq_len) > prev[:3]:
            records[cur_id] = (rank, score, seq_len)
    
    with open(pep_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush()
                cur_seq = []
                m_id    = re.match(r'>(\S+)', line)
                m_type  = re.search(r'ORF type:(\S+)', line)
                m_score = re.search(r'score=([\d.]+)', line)
                m_len   = re.search(r'len:(\d+)', line)
                
                if not m_id:
                    cur_id = None
                    continue
                
                raw_id  = m_id.group(1)
                cur_id  = re.sub(r'\.p\d+$', '', raw_id)
                
                orf_type = m_type.group(1) if m_type else 'internal'
                score    = float(m_score.group(1)) if m_score else 0.0
                length   = int(m_len.group(1)) if m_len else 0
                rank     = TYPE_RANK.get(orf_type.split('(')[0], 1)
                cur_meta = (rank, score, length)
            else:
                cur_seq.append(line)
    
    flush()
    
    iso_cds_len = {iso_id: rec[2] for iso_id, rec in records.items()}
    print("  Parsed {} isoforms with CDS lengths".format(len(iso_cds_len)))
    
    return iso_cds_len


# ── 3. 3-Tier Canonical 결정 ─────────────────────────────────────────

def determine_canonical(iso_list, gene_list,
                        gene_mane, gene_ensembl, gene_appris,
                        iso_cds_len):
    """
    3-tier canonical 결정.
    
    Returns:
        results: list of dicts per gene
    """
    print("\n[Canonical] Determining 3-tier canonical ...")
    
    # gene → [list of (iso_idx, iso_id)] 매핑
    gene_to_isoforms = defaultdict(list)
    for i, (iso_id, gene) in enumerate(zip(iso_list, gene_list)):
        gene_to_isoforms[gene].append((i, iso_id))
    
    # 우리 데이터의 ENST base set
    iso_base_to_idx = {}  # enst_base → [(iso_idx, iso_id_versioned)]
    for i, iso_id in enumerate(iso_list):
        if iso_id.startswith('ENST'):
            base = iso_id.split('.')[0]
            if base not in iso_base_to_idx:
                iso_base_to_idx[base] = []
            iso_base_to_idx[base].append((i, iso_id))
    
    results = []
    counts = {'MANE': 0, 'Ensembl': 0, 'APPRIS': 0, 'longest_CDS': 0, 'no_CDS': 0}
    
    for gene_versioned in sorted(set(gene_list)):
        gene_base = gene_versioned.split('.')[0]
        isoforms = gene_to_isoforms[gene_versioned]
        
        canonical_idx = None
        canonical_iso = None
        source = None
        canonical_enst_base = None
        
        # Priority 1: MANE_Select
        mane_enst = gene_mane.get(gene_base)
        if mane_enst and mane_enst in iso_base_to_idx:
            # 데이터에 존재하는 첫 번째 match
            candidates = iso_base_to_idx[mane_enst]
            # 해당 유전자 소속인 것만
            gene_isoform_idxs = set(idx for idx, _ in isoforms)
            for idx, iso_id in candidates:
                if idx in gene_isoform_idxs:
                    canonical_idx = idx
                    canonical_iso = iso_id
                    canonical_enst_base = mane_enst
                    source = 'MANE'
                    break
        
        # Priority 2: Ensembl_canonical
        if canonical_idx is None:
            ens_enst = gene_ensembl.get(gene_base)
            if ens_enst and ens_enst in iso_base_to_idx:
                gene_isoform_idxs = set(idx for idx, _ in isoforms)
                for idx, iso_id in iso_base_to_idx[ens_enst]:
                    if idx in gene_isoform_idxs:
                        canonical_idx = idx
                        canonical_iso = iso_id
                        canonical_enst_base = ens_enst
                        source = 'Ensembl'
                        break
        
        # Priority 2.5: APPRIS_principal_1 (Ensembl도 없으면)
        if canonical_idx is None:
            appris_enst = gene_appris.get(gene_base)
            if appris_enst and appris_enst in iso_base_to_idx:
                gene_isoform_idxs = set(idx for idx, _ in isoforms)
                for idx, iso_id in iso_base_to_idx[appris_enst]:
                    if idx in gene_isoform_idxs:
                        canonical_idx = idx
                        canonical_iso = iso_id
                        canonical_enst_base = appris_enst
                        source = 'APPRIS'
                        break
        
        # Priority 3: Longest CDS in dataset
        if canonical_idx is None:
            best_len = -1
            for idx, iso_id in isoforms:
                cds_len = iso_cds_len.get(iso_id, 0)
                if cds_len > best_len:
                    best_len = cds_len
                    canonical_idx = idx
                    canonical_iso = iso_id
                    canonical_enst_base = iso_id.split('.')[0] if iso_id.startswith('ENST') else iso_id
                    source = 'longest_CDS' if best_len > 0 else 'no_CDS'
        
        counts[source] += 1
        
        results.append({
            'gene_base': gene_base,
            'gene_versioned': gene_versioned,
            'canonical_enst_base': canonical_enst_base,
            'canonical_iso_idx': canonical_idx,
            'canonical_source': source,
            'canonical_iso_id': canonical_iso,
        })
    
    print("  Total genes: {}".format(len(results)))
    for src, cnt in sorted(counts.items()):
        print("    {}: {} ({:.1f}%)".format(src, cnt, cnt / len(results) * 100))
    
    return results


# ── 4. 저장 ──────────────────────────────────────────────────────────

def save_results(results, out_path):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    
    with open(out_path, 'w') as f:
        header = ['gene_base', 'gene_versioned', 'canonical_enst_base',
                  'canonical_iso_idx', 'canonical_source', 'canonical_iso_id']
        f.write('\t'.join(header) + '\n')
        for r in results:
            f.write('\t'.join(str(r[h]) for h in header) + '\n')
    
    print("\n[Save] {} ({} genes)".format(out_path, len(results)))


# ── 메인 ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 1: GENCODE v44 Canonical Reference Table")
    print("=" * 60)
    
    # Load data
    iso_list = np.load(ISO_LIST_FILE, allow_pickle=True)
    iso_list = [s.decode() if isinstance(s, bytes) else s for s in iso_list]
    
    gene_list = np.load(GENE_LIST_FILE, allow_pickle=True)
    gene_list = [g.decode() if isinstance(g, bytes) else g for g in gene_list]
    
    print("[Load] {} isoforms, {} genes ({} unique)".format(
        len(iso_list), len(gene_list), len(set(gene_list))))
    
    # Parse GENCODE
    gene_mane, gene_ensembl, gene_appris = parse_gencode_canonical(GENCODE_GTF)
    
    # Parse CDS lengths
    iso_cds_len = parse_pep_lengths(PEP_FILE)
    
    # Determine canonical
    results = determine_canonical(
        iso_list, gene_list,
        gene_mane, gene_ensembl, gene_appris,
        iso_cds_len)
    
    # Save
    save_results(results, OUT_FILE)
    
    # Sanity check: compare with old canonical
    old_canonical_file = '../results_isoform/features/canonical_map.txt'
    if os.path.exists(old_canonical_file):
        old_canonical = {}
        with open(old_canonical_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    old_canonical[parts[0].split('.')[0]] = parts[2].split('.')[0]
        
        match = 0
        mismatch = 0
        for r in results:
            old_can = old_canonical.get(r['gene_base'])
            new_can = r['canonical_enst_base']
            if old_can and new_can:
                if old_can == new_can:
                    match += 1
                else:
                    mismatch += 1
        
        print("\n[Comparison] vs old canonical (Pfam-max):")
        print("  Match: {}".format(match))
        print("  Changed: {} ({:.1f}%)".format(
            mismatch, mismatch / (match + mismatch) * 100 if (match + mismatch) > 0 else 0))
    
    print("\n[Done]")


if __name__ == '__main__':
    main()
