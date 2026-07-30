#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_celltype_expr_features.py
----------------------------------
Per-cell-type isoform expression fraction features for v18b.

Input:
  tx_counts_by_cell_type.csv  — 63,994 isoforms × 8 cell types (IsoQuant ID format)
  my_isoform_list_fixed.npy   — 36,748 test isoform IDs (BambuTx or ENST.version)
  my_gene_list_fixed.npy      — 36,748 test gene IDs (ENSG.version)
  train_isoform_list.npy      — train isoform IDs
  extended_annotation...tsv   — ENST → transcript_name mapping

Feature design:
  cell_frac[i, k] = count(isoform_i, celltype_k) / sum_k(count(isoform_i, celltype_k))
  → 8-dim simplex vector: "in which cell types is this isoform expressed?"
  → zero-vector for unmapped isoforms (no expression evidence)

  Additionally: isoform_usage_frac[i, k] =
    count(isoform_i, celltype_k) / sum_j_same_gene(count(isoform_j, celltype_k))
  → "what fraction of gene's expression comes from this isoform in cell type k?"

Output per split (train/test):
  celltype_expr_frac_{split}.npy       — shape (N, 8), normalized counts
  celltype_isoform_usage_{split}.npy   — shape (N, 8), isoform/gene fraction
  Combined: celltype_features_{split}.npy — shape (N, 16)

Saved to: hMuscle/results_isoform/features/celltype/
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR    = '../model'
ID_DIR       = '../data/raw_data/data/id_lists'
FEAT_DIR     = '../results_isoform/features/celltype'
ANNOT_TSV    = '/home/dhkim1674/Project_AD_with_refTSS_novel/02_Isoquant_Output/' \
               'extended_annotation_including_refTSS_umi10_donor3_supported_novel_tx_with_gene_name_for_novel_tx.tsv'
COUNT_CSV    = '/home/dhkim1674/Project_AD_with_refTSS_novel/04_Counts/Long_Read/' \
               'Cell_Type/counts_by_cell_type/tx_counts_by_cell_type.csv'
BAMBU_GTF    = '../data/cleaned_annotations.gtf'

os.makedirs(FEAT_DIR, exist_ok=True)

CELL_TYPES = ['Astrocyte', 'Excitatory neuron', 'Inhibitory neuron', 'Lymphocyte',
              'Microglia', 'OPC', 'Oligodendrocyte', 'Vascular cell']

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


print("=" * 65)
print("  v18b Preprocessing: Cell-Type Expression Features")
print("=" * 65)

# ── 1. Load count matrix ──────────────────────────────────────
print("\n[1] Loading count matrix...")
df_counts = pd.read_csv(COUNT_CSV)
print(f"  Shape: {df_counts.shape}")
print(f"  Cell types: {df_counts.columns[2:].tolist()}")

# Index by transcript_name
df_counts = df_counts.set_index('transcript_name')
count_mat = df_counts[CELL_TYPES].values.astype(np.float32)  # (63994, 8)
tx_names  = np.array(df_counts.index.tolist())               # transcript_name strings

# Build tx_name → row_idx lookup
txname2row = {name: i for i, name in enumerate(tx_names)}

# Gene → set of row indices (for isoform usage fraction)
gene2rows = {}
for i, (tx, gene) in enumerate(zip(df_counts.index, df_counts['gene_name'])):
    gene2rows.setdefault(gene, []).append(i)
gene2rows = {k: np.array(v) for k, v in gene2rows.items()}

print(f"  Unique transcript names: {len(txname2row)}")
print(f"  Unique genes: {len(gene2rows)}")


# ── 2. Build ENST → transcript_name mapping ──────────────────
print("\n[2] Building ENST → transcript_name mapping...")
# Load IsoQuant annotation — only need transcript rows
df_annot = pd.read_csv(
    ANNOT_TSV, sep='\t',
    usecols=['feature', 'transcript_id', 'transcript_name', 'gene_name'],
    dtype=str
)
df_annot = df_annot[df_annot['feature'] == 'transcript'].drop_duplicates('transcript_id')
enst2txname = df_annot.set_index('transcript_id')['transcript_name'].to_dict()
enst2gene   = df_annot.set_index('transcript_id')['gene_name'].to_dict()
print(f"  Mapped ENST IDs: {len(enst2txname)}")


# ── 3. Build BambuTx → gene_name mapping from GTF ────────────
print("\n[3] Building BambuTx → gene_name from Bambu GTF...")
bambu2gene = {}
with open(BAMBU_GTF) as f:
    for line in f:
        if '\ttranscript\t' not in line: continue
        # gene_id "ENSG00000228794.13"; transcript_id "BambuTx2";
        tid = None; gid = None
        for part in line.split(';'):
            part = part.strip()
            if 'transcript_id' in part:
                tid = part.split('"')[1].strip()
            elif 'gene_id' in part:
                gid = part.split('"')[1].strip()
        if tid and gid and tid.startswith('BambuTx'):
            bambu2gene[tid] = gid.split('.')[0]  # strip version: ENSG00000228794

print(f"  BambuTx → ENSG mappings: {len(bambu2gene)}")

# Build ENSG → gene_name from annotation
ensg2genename = df_annot.dropna(subset=['gene_name']).drop_duplicates('transcript_id')
# Also need gene_id → gene_name; use IsoQuant GTF
# Alternatively build from df_counts gene_name column and ENST2gene
ensg2sym = {}
for enst, txname in enst2txname.items():
    gene = enst2gene.get(enst, '')
    if gene:
        ensg2sym[gene] = enst2gene.get(enst, gene)

# Load Ensembl symbol file for ENSG → symbol
ensg2sym_file = {}
sym_path = os.path.join(ID_DIR, 'ensembl_to_symbol.txt')
if os.path.exists(sym_path):
    with open(sym_path) as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 5:
                ensg2sym_file[p[0]] = p[4]
print(f"  ENSG → symbol mappings: {len(ensg2sym_file)}")


def lookup_row(iso_id: str) -> tuple:
    """
    Returns (row_idx_in_count_mat, gene_name_str) for an isoform ID.
    row_idx = -1 if not found.
    """
    if iso_id.startswith('ENST'):
        enst_noversion = iso_id.split('.')[0]
        txname = enst2txname.get(enst_noversion)
        if txname is None:
            return -1, None
        row = txname2row.get(txname, -1)
        gene = enst2gene.get(enst_noversion)
        return row, gene

    elif iso_id.startswith('BambuTx'):
        ensg_noversion = bambu2gene.get(iso_id)
        if ensg_noversion is None:
            return -1, None
        # Get gene_name (symbol) from ENSG
        symbol = ensg2sym_file.get(ensg_noversion, ensg_noversion)
        return -1, symbol  # no per-isoform row, but we have gene

    return -1, None


def gene_expr_frac(gene_name):
    """
    Returns (8,) cell-type expression fraction for a gene, or zeros if not found.
    """
    if gene_name and gene_name in gene2rows:
        rows = gene2rows[gene_name]
        counts = count_mat[rows].sum(axis=0)  # (8,)
        total = counts.sum()
        if total > 0:
            return counts / total
    return np.zeros(8, dtype=np.float32)


def build_features(iso_ids, gene_syms, split_name):
    """
    Build (N, 16) feature matrix for a split.
    First 8 dims: cell-type expression fraction (direct isoform or gene proxy).
    Next 8 dims: isoform-to-gene usage fraction per cell type (zero if gene-proxy).

    iso_ids   — list of isoform ID strings (ENST.version, BambuTx#, NM_#, etc.)
    gene_syms — list of gene symbol strings (e.g., 'ADA', 'NAT2')
                OR list of raw ENSG IDs (for test set; will be handled via ENSG2SYM)
    """
    N = len(iso_ids)
    expr_frac     = np.zeros((N, 8), dtype=np.float32)
    isoform_usage = np.zeros((N, 8), dtype=np.float32)

    n_direct  = 0
    n_gene    = 0
    n_missing = 0

    for i, iso_id in enumerate(iso_ids):
        gene_sym_raw = clean(gene_syms[i]) if gene_syms is not None else ''
        # Resolve gene symbol: could be gene symbol directly or ENSG
        if gene_sym_raw.startswith('ENSG'):
            gene_sym = ensg2sym_file.get(gene_sym_raw.split('.')[0], gene_sym_raw)
        else:
            gene_sym = gene_sym_raw

        row, gene_name_from_id = lookup_row(iso_id)
        # Prefer gene name from ID lookup (more precise), fallback to gene_syms
        eff_gene = gene_name_from_id or gene_sym

        if row >= 0:
            # Direct isoform hit in count matrix
            counts = count_mat[row]  # (8,)
            total = counts.sum()
            if total > 0:
                expr_frac[i] = counts / total
                if eff_gene and eff_gene in gene2rows:
                    gene_rows   = gene2rows[eff_gene]
                    gene_counts = count_mat[gene_rows].sum(axis=0)
                    safe_denom  = np.where(gene_counts > 0, gene_counts, 1.0)
                    isoform_usage[i] = counts / safe_denom
            n_direct += 1

        elif eff_gene:
            # Gene-level proxy (BambuTx, NM_, or unmapped ENST)
            expr_frac[i] = gene_expr_frac(eff_gene)
            if expr_frac[i].sum() > 0:
                n_gene += 1
            else:
                n_missing += 1
        else:
            n_missing += 1

    print(f"  [{split_name}] N={N}: direct={n_direct}, gene_proxy={n_gene}, missing={n_missing}")
    print(f"    Coverage: {(n_direct + n_gene) / N * 100:.1f}%")

    combined = np.concatenate([expr_frac, isoform_usage], axis=1)  # (N, 16)
    return expr_frac, isoform_usage, combined


# ── 4. Process test split ─────────────────────────────────────
print("\n[4] Processing test split...")
te_iso_raw  = np.load(f'{MODEL_DIR}/my_isoform_list_fixed.npy', allow_pickle=True)
te_gene_raw = np.load(f'{MODEL_DIR}/my_gene_list_fixed.npy',    allow_pickle=True)
te_iso_ids  = [clean(x) for x in te_iso_raw]
te_gene_ids = te_gene_raw  # raw, will be cleaned in lookup

expr_frac_te, usage_te, combined_te = build_features(te_iso_ids, te_gene_ids, 'test')

np.save(f'{FEAT_DIR}/celltype_expr_frac_test.npy',    expr_frac_te)
np.save(f'{FEAT_DIR}/celltype_isoform_usage_test.npy', usage_te)
np.save(f'{FEAT_DIR}/celltype_features_test.npy',      combined_te)
print(f"  Saved test features: {combined_te.shape}")


# ── 5. Process train split ────────────────────────────────────
print("\n[5] Processing train split...")
tr_iso_path  = f'{ID_DIR}/train_isoform_list.npy'
tr_gene_path = f'{ID_DIR}/train_gene_list.npy'

if os.path.exists(tr_iso_path):
    tr_iso_raw  = np.load(tr_iso_path,  allow_pickle=True)
    tr_gene_raw = np.load(tr_gene_path, allow_pickle=True)
    tr_iso_ids  = [clean(x) for x in tr_iso_raw]

    expr_frac_tr, usage_tr, combined_tr = build_features(tr_iso_ids, tr_gene_raw, 'train')
    np.save(f'{FEAT_DIR}/celltype_expr_frac_train.npy',    expr_frac_tr)
    np.save(f'{FEAT_DIR}/celltype_isoform_usage_train.npy', usage_tr)
    np.save(f'{FEAT_DIR}/celltype_features_train.npy',      combined_tr)
    print(f"  Saved train features: {combined_tr.shape}")
else:
    print(f"  [WARN] Train isoform list not found at {tr_iso_path}")
    print("  Checking alternative paths...")
    # Try to find train isoforms from ESM-2 embedding file dimensions
    x_tr = np.load(f'{MODEL_DIR}/../data/esm2_train_human_layer30_t30_150M.npy', mmap_mode='r')
    print(f"  Train ESM-2 shape: {x_tr.shape} — no isoform IDs available for train split")
    print("  Skipping train split (test features only)")


# ── 6. Summary statistics ─────────────────────────────────────
print("\n[6] Feature statistics (test)...")
print(f"  expr_frac    : mean={expr_frac_te.mean():.4f}, std={expr_frac_te.std():.4f}, "
      f"nonzero={( expr_frac_te.sum(axis=1) > 0).mean()*100:.1f}%")
print(f"  isoform_usage: mean={usage_te.mean():.4f}, std={usage_te.std():.4f}, "
      f"nonzero={(usage_te.sum(axis=1) > 0).mean()*100:.1f}%")

# Per-cell-type expression profile
print("\n  Per cell-type expression (mean over all isoforms):")
for k, ct in enumerate(CELL_TYPES):
    print(f"    {ct:25s}: {expr_frac_te[:, k].mean():.4f}")

print("\n  Done.")
