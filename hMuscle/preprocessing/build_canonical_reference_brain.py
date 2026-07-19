# -*- coding: utf-8 -*-
"""
build_canonical_reference_brain.py
===================================
Brain counterpart of build_canonical_reference.py (muscle). Rebuilds the
canonical-isoform reference table for the brain_isoquant "full" set that was
used (but not persisted) in the 2026-07-20 canonical-anchored severity
regression session (see memory finding-severity-regression-canonical-anchored.md).

Brain isoquant isoform IDs follow GENCODE's transcript_name convention
(e.g. "A1BG-204") rather than gene_id/transcript_id, so the same GENCODE v44
GTF is re-parsed keyed on gene_name/transcript_name.

4-tier canonical priority (brain has no PEP-derived CDS-length table for the
"full" set, so the longest_CDS fallback uses protein length from
brain_full_proteins.fa directly):
  Priority 1: MANE_Select
  Priority 2: Ensembl_canonical
  Priority 3: appris_principal_1
  Priority 4: longest protein sequence present in the dataset
  (no isoform of the gene has a protein sequence -> no_CDS, gene excluded)

입력:
  GENCODE v44 GTF (hMuscle/data/refdata-gex-GRCh38-2024-A/genes/genes.gtf.gz)
  hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy
  hMuscle/data/brain_isoquant_esm2/full/brain_full_gene_names.npy
  reports/truebrain_rerun_20260714/data/brain_full_proteins.fa

출력:
  hMuscle/results_isoform/features/canonical_reference_brain.tsv
    columns: gene_name | canonical_iso_idx | canonical_iso_id | canonical_source
"""

import numpy as np
import re
import os
import gzip
from collections import defaultdict

GENCODE_GTF = '../data/refdata-gex-GRCh38-2024-A/genes/genes.gtf.gz'
IDS_FILE = '../data/brain_isoquant_esm2/full/brain_full_ids.npy'
GENE_FILE = '../data/brain_isoquant_esm2/full/brain_full_gene_names.npy'
PEP_FILE = '../../reports/truebrain_rerun_20260714/data/brain_full_proteins.fa'
OUT_FILE = '../results_isoform/features/canonical_reference_brain.tsv'


def parse_gencode_canonical_by_name(gtf_gz_path):
    """Same as build_canonical_reference.py Step 1, keyed on gene_name/transcript_name."""
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

            m_gene = re.search(r'gene_name "([^"]+)"', line)
            m_tx = re.search(r'transcript_name "([^"]+)"', line)
            if not m_gene or not m_tx:
                continue

            gene_name = m_gene.group(1)
            tx_name = m_tx.group(1)

            if 'MANE_Select' in line:
                gene_mane[gene_name] = tx_name

            if 'Ensembl_canonical' in line:
                gene_ensembl[gene_name] = tx_name

            if 'appris_principal_1' in line and gene_name not in gene_appris:
                gene_appris[gene_name] = tx_name

    print("  Transcripts scanned: {}".format(n_transcripts))
    print("  MANE_Select: {} genes".format(len(gene_mane)))
    print("  Ensembl_canonical: {} genes".format(len(gene_ensembl)))
    print("  APPRIS_principal_1: {} genes".format(len(gene_appris)))

    return gene_mane, gene_ensembl, gene_appris


def parse_pep_lengths(pep_path):
    """brain_full_proteins.fa -> {iso_id (transcript_name, .p# stripped): protein length}."""
    print("[PEP] Parsing {} ...".format(pep_path))

    lengths = {}
    cur_id = None
    cur_len = 0

    def flush():
        if cur_id is None:
            return
        prev = lengths.get(cur_id, -1)
        if cur_len > prev:
            lengths[cur_id] = cur_len

    with open(pep_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush()
                raw_id = line[1:].split()[0]
                cur_id = re.sub(r'\.p\d+$', '', raw_id)
                cur_len = 0
            else:
                cur_len += len(line.strip())
    flush()

    print("  Parsed {} isoforms with protein sequences".format(len(lengths)))
    return lengths


def determine_canonical(iso_ids, gene_names, gene_mane, gene_ensembl, gene_appris, iso_len):
    print("\n[Canonical] Determining 4-tier canonical (brain) ...")

    gene_to_isoforms = defaultdict(list)
    for i, (iso_id, gene) in enumerate(zip(iso_ids, gene_names)):
        gene_to_isoforms[gene].append((i, iso_id))

    results = []
    counts = {'MANE': 0, 'Ensembl': 0, 'APPRIS': 0, 'longest_CDS': 0, 'no_CDS': 0}

    for gene in sorted(gene_to_isoforms.keys()):
        isoforms = gene_to_isoforms[gene]
        gene_iso_ids = {iso_id for _, iso_id in isoforms}

        canonical_idx = None
        canonical_iso = None
        source = None

        mane_tx = gene_mane.get(gene)
        if mane_tx and mane_tx in gene_iso_ids:
            for idx, iso_id in isoforms:
                if iso_id == mane_tx:
                    canonical_idx, canonical_iso, source = idx, iso_id, 'MANE'
                    break

        if canonical_idx is None:
            ens_tx = gene_ensembl.get(gene)
            if ens_tx and ens_tx in gene_iso_ids:
                for idx, iso_id in isoforms:
                    if iso_id == ens_tx:
                        canonical_idx, canonical_iso, source = idx, iso_id, 'Ensembl'
                        break

        if canonical_idx is None:
            appris_tx = gene_appris.get(gene)
            if appris_tx and appris_tx in gene_iso_ids:
                for idx, iso_id in isoforms:
                    if iso_id == appris_tx:
                        canonical_idx, canonical_iso, source = idx, iso_id, 'APPRIS'
                        break

        if canonical_idx is None:
            best_len = -1
            for idx, iso_id in isoforms:
                L = iso_len.get(iso_id, -1)
                if L > best_len:
                    best_len = L
                    canonical_idx, canonical_iso = idx, iso_id
            source = 'longest_CDS' if best_len > 0 else 'no_CDS'

        counts[source] += 1
        if source == 'no_CDS':
            canonical_idx = None
            canonical_iso = None

        results.append({
            'gene_name': gene,
            'canonical_iso_idx': canonical_idx if canonical_idx is not None else '',
            'canonical_iso_id': canonical_iso if canonical_iso is not None else '',
            'canonical_source': source,
        })

    print("  Total genes: {}".format(len(results)))
    for src, cnt in sorted(counts.items()):
        print("    {}: {} ({:.1f}%)".format(src, cnt, cnt / len(results) * 100))

    return results


def save_results(results, out_path):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        header = ['gene_name', 'canonical_iso_idx', 'canonical_iso_id', 'canonical_source']
        f.write('\t'.join(header) + '\n')
        for r in results:
            f.write('\t'.join(str(r[h]) for h in header) + '\n')
    print("\n[Save] {} ({} genes)".format(out_path, len(results)))


def main():
    print("=" * 60)
    print("Brain Canonical Reference Table (GENCODE v44, name-keyed)")
    print("=" * 60)

    iso_ids = np.load(IDS_FILE, allow_pickle=True)
    iso_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in iso_ids]

    gene_names = np.load(GENE_FILE, allow_pickle=True)
    gene_names = [g.decode() if isinstance(g, bytes) else str(g) for g in gene_names]

    print("[Load] {} isoforms, {} genes ({} unique)".format(
        len(iso_ids), len(gene_names), len(set(gene_names))))

    gene_mane, gene_ensembl, gene_appris = parse_gencode_canonical_by_name(GENCODE_GTF)
    iso_len = parse_pep_lengths(PEP_FILE)

    results = determine_canonical(iso_ids, gene_names, gene_mane, gene_ensembl, gene_appris, iso_len)
    save_results(results, OUT_FILE)

    print("\n[Done]")


if __name__ == '__main__':
    main()
