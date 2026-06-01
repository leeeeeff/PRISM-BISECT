# -*- coding: utf-8 -*-
"""
build_nm_enst_mapping.py
========================
Step 1: NM_ → ENST 포괄적 매핑 테이블 구축

두 소스 합산:
  [A] MANE_summary.txt.gz  — 고신뢰도 MANE Select (38.7% coverage)
  [B] Biomart bulk query    — 나머지 61.3% 커버 (ENST ↔ RefSeq_mRNA)

출력:
  ../results_isoform/features/splicing/nm_to_enst_mapping.tsv
    컬럼: nm_id  enst_id  source  gene_name

실행:
  cd hMuscle/preprocessing/
  python build_nm_enst_mapping.py
"""

import gzip
import os
import time
import urllib.request
import urllib.parse

# ── 경로 ──────────────────────────────────────────────────────────────
MANE_FILE    = '../data/MANE_summary.txt.gz'
TRAIN_ISO    = '../data/raw_data/data/id_lists/train_isoform_list.npy'
OUT_DIR      = '../results_isoform/features/splicing/'
OUT_MAPPING  = OUT_DIR + 'nm_to_enst_mapping.tsv'

BIOMART_URL  = 'https://www.ensembl.org/biomart/martservice'

# ── Biomart XML query ──────────────────────────────────────────────────
BIOMART_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1"
       uniqueRows="0" count="" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Attribute name="ensembl_transcript_id"/>
    <Attribute name="refseq_mrna"/>
    <Attribute name="external_gene_name"/>
    <Attribute name="transcript_biotype"/>
  </Dataset>
</Query>"""


def load_mane_mapping(mane_file):
    """MANE_summary.txt.gz → {nm_base: (enst_base, gene_name)}"""
    mapping = {}
    with gzip.open(mane_file, 'rt') as f:
        header = f.readline().strip().split('\t')
        nm_col   = header.index('RefSeq_nuc')
        enst_col = header.index('Ensembl_nuc')
        sym_col  = header.index('symbol')
        for line in f:
            parts = line.strip().split('\t')
            nm_ver   = parts[nm_col]
            enst_ver = parts[enst_col]
            symbol   = parts[sym_col]
            nm_base   = nm_ver.split('.')[0]
            enst_base = enst_ver.split('.')[0]
            if nm_base.startswith('NM_') and enst_base.startswith('ENST'):
                mapping[nm_base] = (enst_base, symbol, 'MANE')
    print(f"[MANE] Loaded {len(mapping)} NM_→ENST mappings")
    return mapping


def query_biomart(xml_query, max_retries=3, timeout=300):
    """Biomart REST API로 bulk query, TSV 반환."""
    params = urllib.parse.urlencode({'query': xml_query}).encode()
    for attempt in range(max_retries):
        try:
            print(f"[Biomart] Query attempt {attempt+1}/{max_retries} ...")
            req = urllib.request.Request(BIOMART_URL, data=params,
                                         headers={'User-Agent': 'DIFFUSE/1.0'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read().decode('utf-8')
            print(f"[Biomart] Downloaded {len(content)//1024} KB")
            return content
        except Exception as e:
            print(f"[Biomart] Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
    return None


def parse_biomart_tsv(content):
    """Biomart TSV → {nm_base: (enst_base, gene_name)}"""
    mapping = {}
    lines = content.strip().split('\n')
    if not lines:
        return mapping
    # header: Transcript stable ID, RefSeq mRNA ID, Gene name, Transcript type
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        enst_base = parts[0].split('.')[0]
        nm_ver    = parts[1].strip()
        gene_name = parts[2].strip() if len(parts) > 2 else ''
        if not nm_ver.startswith('NM_'):
            continue
        nm_base = nm_ver.split('.')[0]
        if nm_base not in mapping:
            mapping[nm_base] = (enst_base, gene_name, 'biomart')
    print(f"[Biomart] Parsed {len(mapping)} NM_→ENST mappings")
    return mapping


def main():
    import numpy as np
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load train NM_ IDs
    # 일부 엔트리는 쉼표 구분 다중 ID (예: "NM_001206729,NM_181690") → 개별 NM_ 추출
    train_iso_raw = np.load(TRAIN_ISO, allow_pickle=True)
    train_ids_all = set()
    for x in train_iso_raw:
        raw = (x.decode() if isinstance(x, bytes) else x)
        for nm in raw.split(','):
            nm_base = nm.strip().split('.')[0]
            if nm_base.startswith('NM_') or nm_base.startswith('NR_'):
                train_ids_all.add(nm_base)
    train_ids = sorted(train_ids_all)
    print(f"[Train] {len(train_iso_raw)} entries → {len(train_ids)} unique NM_ IDs after comma expansion")

    # 2. MANE Select mapping
    mane_map = load_mane_mapping(MANE_FILE)

    # 3. Biomart bulk mapping
    biomart_map = {}
    biomart_cache = OUT_DIR + 'biomart_raw.tsv'
    if os.path.exists(biomart_cache):
        print(f"[Biomart] Using cached file: {biomart_cache}")
        with open(biomart_cache) as f:
            content = f.read()
        biomart_map = parse_biomart_tsv(content)
    else:
        content = query_biomart(BIOMART_XML)
        if content:
            with open(biomart_cache, 'w') as f:
                f.write(content)
            biomart_map = parse_biomart_tsv(content)
        else:
            print("[WARN] Biomart query failed — using MANE only")

    # 4. 합산 (MANE 우선)
    final_map = {}
    final_map.update(biomart_map)
    final_map.update(mane_map)   # MANE가 biomart보다 고신뢰도

    # 5. Coverage 계산
    covered = [nid for nid in train_ids if nid in final_map]
    not_covered = [nid for nid in train_ids if nid not in final_map]
    print(f"\n[Coverage]")
    print(f"  Total train NM_ IDs:  {len(train_ids)}")
    print(f"  Covered:              {len(covered)} ({len(covered)/len(train_ids)*100:.1f}%)")
    print(f"  Not covered (→ zero): {len(not_covered)} ({len(not_covered)/len(train_ids)*100:.1f}%)")
    mane_cnt  = sum(1 for nid in covered if final_map[nid][2] == 'MANE')
    biom_cnt  = sum(1 for nid in covered if final_map[nid][2] == 'biomart')
    print(f"  From MANE:            {mane_cnt}")
    print(f"  From Biomart:         {biom_cnt}")

    # 6. 저장 (train IDs만 출력)
    with open(OUT_MAPPING, 'w') as f:
        f.write("nm_id\tenst_id\tgene_name\tsource\n")
        for nid in train_ids:
            if nid in final_map:
                enst, gname, src = final_map[nid]
                f.write(f"{nid}\t{enst}\t{gname}\t{src}\n")
            else:
                f.write(f"{nid}\t\t\tunmapped\n")

    print(f"\n[Save] {OUT_MAPPING} ({len(train_ids)} rows)")
    if not_covered[:5]:
        print(f"[Sample unmapped] {not_covered[:5]}")


if __name__ == '__main__':
    main()
