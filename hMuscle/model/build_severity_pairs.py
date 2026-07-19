#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_severity_pairs.py
========================
Step 0 rebuild (2026-07-20 session's canonical-anchored severity regression had
no persisted pair-table/script — see memory finding-severity-regression-canonical-anchored.md
and approach-isoform-local-change-axis-design.md). Reconstructs, for muscle and
brain independently, the canonical-anchored within-gene pair set + 5 structural
covariates per pair (size, domain_binary, nterm_overlap, disorder_frac,
resync_failure_binary). The coherence-projection severity SCORE (gene-disjoint
5-fold mean-pool L15+L30 projection) is added by a separate downstream script
(build_severity_score.py) so this table is reusable for any future scoring
scheme without recomputation.

Pair convention: long = longer sequence of the pair, short = shorter (ties broken
long=canonical). Covariates are computed on (long, short) regardless of which one
is canonical; `canonical_is_lo` records whether canonical happens to be the
shorter member (needed for the devils-advocate C1 stratification check).

출력 (committed, not ephemeral):
  reports/severity_pairs/muscle_severity_pairs.tsv
  reports/severity_pairs/brain_severity_pairs.tsv
"""
import re
import os
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

import numpy as np

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT_DIR = ROOT / 'reports/severity_pairs'

TOPIDP = {'A': 0.06, 'R': 0.180, 'N': 0.007, 'D': 0.192, 'C': 0.02, 'Q': 0.318, 'E': 0.736,
          'G': 0.166, 'H': 0.303, 'I': -0.486, 'L': -0.326, 'K': 0.586, 'M': -0.397,
          'F': -0.697, 'P': 0.987, 'S': 0.341, 'T': 0.059, 'W': -0.884, 'Y': -0.510, 'V': -0.121}

NTERM_WINDOW = 60


def strip_orf(name):
    return re.sub(r'\.p\d+$', '', name)


def parse_pep_sequences(pep_path):
    """Best-ORF-per-isoform sequence text (completeness-rank > score > length),
    same selection rule as hMuscle/preprocessing/build_canonical_reference.py."""
    TYPE_RANK = {'complete': 4, '5prime_partial': 3, '3prime_partial': 2, 'internal': 1}
    best = {}  # iso_id -> (rank, score, seq)

    cur_id, cur_meta, cur_seq = None, None, []

    def flush():
        if cur_id is None:
            return
        seq = ''.join(cur_seq).replace('*', '').strip()
        if not seq:
            return
        rank, score = cur_meta
        prev = best.get(cur_id)
        if prev is None or (rank, score, len(seq)) > (prev[0], prev[1], len(prev[2])):
            best[cur_id] = (rank, score, seq)

    with open(pep_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush()
                cur_seq = []
                m_id = re.match(r'>(\S+)', line)
                m_type = re.search(r'ORF type:(\S+)', line)
                m_score = re.search(r'score=([\d.]+)', line)
                raw_id = m_id.group(1)
                cur_id = strip_orf(raw_id)
                orf_type = m_type.group(1) if m_type else 'internal'
                score = float(m_score.group(1)) if m_score else 0.0
                rank = TYPE_RANK.get(orf_type.split('(')[0], 1)
                cur_meta = (rank, score)
            else:
                cur_seq.append(line)
    flush()
    return {k: v[2] for k, v in best.items()}


def parse_fasta_sequences(fa_path):
    """Plain FASTA (one longest-kept sequence per stripped id)."""
    seqs = {}
    cur_id, cur_seq = None, []

    def flush():
        if cur_id is None:
            return
        seq = ''.join(cur_seq).strip()
        if not seq:
            return
        prev = seqs.get(cur_id)
        if prev is None or len(seq) > len(prev):
            seqs[cur_id] = seq

    with open(fa_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush()
                cur_seq = []
                cur_id = strip_orf(line[1:].split()[0])
            else:
                cur_seq.append(line)
    flush()
    return seqs


def load_canonical_muscle():
    """gene_versioned -> canonical_iso_idx (int or None)."""
    path = ROOT / 'hMuscle/results_isoform/features/canonical_reference.tsv'
    m = {}
    with open(path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = dict(zip(header, line.rstrip('\n').split('\t')))
            src = parts['canonical_source']
            idx = parts['canonical_iso_idx']
            m[parts['gene_versioned']] = int(idx) if src != 'no_CDS' and idx != '' else None
    return m


def load_canonical_brain():
    """gene_name -> canonical_iso_idx (int or None)."""
    path = ROOT / 'hMuscle/results_isoform/features/canonical_reference_brain.tsv'
    m = {}
    with open(path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = dict(zip(header, line.rstrip('\n').split('\t')))
            src = parts['canonical_source']
            idx = parts['canonical_iso_idx']
            m[parts['gene_name']] = int(idx) if src != 'no_CDS' and idx != '' else None
    return m


def changed_intervals(long_s, short_s):
    """SequenceMatcher opcodes -> (list of (i1,i2) changed intervals on long_s,
    total changed residue count, list of non-'equal' opcodes in order)."""
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    ops = sm.get_opcodes()
    ivs, changed = [], 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == 'equal':
            continue
        changed += max(i2 - i1, j2 - j1)
        if i2 > i1:
            ivs.append((i1, i2))
    return ivs, changed, ops


def resync_failure_binary(ops, long_len):
    """First change position -> longest single contiguous 'equal' block downstream,
    as a fraction of downstream length. Binarized at ratio < 0.5 (majority of the
    downstream sequence is NOT covered by one clean resynced block).
    Fixes the tautology bug from the 2026-07-20 session (last-change-based proxy
    was always 1.0 by construction) by anchoring on the FIRST change instead."""
    first_change_start = None
    for tag, i1, i2, j1, j2 in ops:
        if tag != 'equal':
            first_change_start = i1
            break
    if first_change_start is None:
        return None  # no change at all (shouldn't happen, pairs are pre-filtered)
    downstream_len = long_len - first_change_start
    if downstream_len <= 0:
        return None
    max_equal = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == 'equal' and i1 >= first_change_start:
            max_equal = max(max_equal, i2 - i1)
    ratio = max_equal / downstream_len
    return int(ratio < 0.5)


def build_pairs(tissue, iso_ids, gene_ids, canonical_map, seqs, domain_mat):
    gene_to_isoidx = defaultdict(list)
    for i, g in enumerate(gene_ids):
        gene_to_isoidx[g].append(i)

    rows = []
    n_genes_multi = 0
    n_no_canonical = 0
    n_missing_seq = 0

    for gene, idxs in gene_to_isoidx.items():
        if len(idxs) < 2:
            continue
        can_idx = canonical_map.get(gene)
        if can_idx is None or can_idx not in idxs:
            n_no_canonical += 1
            continue
        n_genes_multi += 1
        can_id = iso_ids[can_idx]
        if can_id not in seqs:
            n_missing_seq += 1
            continue

        for other_idx in idxs:
            if other_idx == can_idx:
                continue
            other_id = iso_ids[other_idx]
            if other_id not in seqs:
                n_missing_seq += 1
                continue

            seq_can, seq_other = seqs[can_id], seqs[other_id]
            if seq_can == seq_other:
                continue

            if len(seq_can) >= len(seq_other):
                long_idx, short_idx = can_idx, other_idx
                long_s, short_s = seq_can, seq_other
                canonical_is_lo = False
            else:
                long_idx, short_idx = other_idx, can_idx
                long_s, short_s = seq_other, seq_can
                canonical_is_lo = True

            ivs, size, ops = changed_intervals(long_s, short_s)
            if size == 0 or not ivs:
                continue

            dom_diff = int(np.any(domain_mat[long_idx] != domain_mat[short_idx]))
            nterm = int(any(i1 < NTERM_WINDOW for i1, i2 in ivs))
            changed_res_idx = [i for (a, b) in ivs for i in range(a, b)]
            disorder = float(np.mean([TOPIDP.get(long_s[i], 0.0) for i in changed_res_idx]))
            resync_fail = resync_failure_binary(ops, len(long_s))
            if resync_fail is None:
                continue

            rows.append({
                'tissue': tissue,
                'gene': gene,
                'canonical_idx': can_idx,
                'other_idx': other_idx,
                'long_idx': long_idx,
                'short_idx': short_idx,
                'canonical_is_lo': int(canonical_is_lo),
                'size': size,
                'domain_binary': dom_diff,
                'nterm_overlap': nterm,
                'disorder_frac': disorder,
                'resync_failure_binary': resync_fail,
            })

    print(f"[{tissue}] multi-isoform genes with resolved canonical: {n_genes_multi} "
          f"(skipped no-canonical genes: {n_no_canonical}, missing-seq events: {n_missing_seq})")
    print(f"[{tissue}] pairs built: {len(rows)}")
    return rows


def save_rows(rows, out_path):
    os.makedirs(out_path.parent, exist_ok=True)
    cols = ['tissue', 'gene', 'canonical_idx', 'other_idx', 'long_idx', 'short_idx',
            'canonical_is_lo', 'size', 'domain_binary', 'nterm_overlap',
            'disorder_frac', 'resync_failure_binary']
    with open(out_path, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in rows:
            f.write('\t'.join(str(r[c]) for c in cols) + '\n')
    print(f"[Save] {out_path} ({len(rows)} rows)")


def main():
    # ---- Muscle ----
    iso_m = np.load(ROOT / 'hMuscle/model/my_isoform_list_fixed.npy', allow_pickle=True)
    iso_m = [s.decode() if isinstance(s, bytes) else str(s) for s in iso_m]
    gene_m = np.load(ROOT / 'hMuscle/model/my_gene_list_fixed.npy', allow_pickle=True)
    gene_m = [g.decode() if isinstance(g, bytes) else str(g) for g in gene_m]
    dom_m = np.load(ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy')
    canon_m = load_canonical_muscle()
    seqs_m = parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
    print(f"[muscle] {len(iso_m)} isoforms, {len(set(gene_m))} genes, "
          f"{len(seqs_m)} sequences parsed, domain_mat {dom_m.shape}")
    rows_m = build_pairs('muscle', iso_m, gene_m, canon_m, seqs_m, dom_m)
    save_rows(rows_m, OUT_DIR / 'muscle_severity_pairs.tsv')

    # ---- Brain ----
    iso_b = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)
    iso_b = [s.decode() if isinstance(s, bytes) else str(s) for s in iso_b]
    gene_b = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_gene_names.npy', allow_pickle=True)
    gene_b = [g.decode() if isinstance(g, bytes) else str(g) for g in gene_b]
    dom_b = np.load(ROOT / 'hMuscle/results_isoform/features/domain_matrix_brain_full.npy')
    canon_b = load_canonical_brain()
    seqs_b = parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    print(f"[brain] {len(iso_b)} isoforms, {len(set(gene_b))} genes, "
          f"{len(seqs_b)} sequences parsed, domain_mat {dom_b.shape}")
    rows_b = build_pairs('brain', iso_b, gene_b, canon_b, seqs_b, dom_b)
    save_rows(rows_b, OUT_DIR / 'brain_severity_pairs.tsv')


if __name__ == '__main__':
    main()
