#!/usr/bin/env python3
"""
mc2_blast_uncovered_auprc.py
============================
MC2: Compute PRISM v17f* AUPRC on BLAST-uncovered isoforms (31.5%).

Research question: Does PRISM add value on isoforms BLAST cannot annotate?
If PRISM achieves substantial AUPRC on those isoforms, it provides complementary
information to BLAST beyond gene-identification.

BLAST-uncovered = isoforms where no BLAST hit yields an MF GO annotation
(BLAST scores = 0 for all 81 valid MF terms).
"""

import os, sys, gzip, json
import numpy as np
from collections import defaultdict
import re

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
DATA_DIR   = os.path.join(ROOT, 'hMuscle/data')
ID_DIR     = os.path.join(DATA_DIR, 'raw_data/data/id_lists')
ANNOT_DIR  = os.path.join(DATA_DIR, 'raw_data/data/annotations')
SP_DIR     = os.path.join(ROOT, 'reports/exp_e_sota/swissprot_db')
GOA_FILE   = os.path.join(SP_DIR, 'goa_human.gaf.gz')
BLAST_OUT  = os.path.join(ROOT, 'reports/benchmark_external/blast_goa/blast_hits.tsv')
PRISM_PREDS = os.path.join(ROOT, 'reports/sota_final_benchmark/prism_preds.npy')

print("=== MC2: BLAST-uncovered subset AUPRC analysis ===", flush=True)

# ── 1. GOA MF annotations ─────────────────────────────────────────────────
print("\n[1] Loading GOA...", flush=True)
uniprot2go = defaultdict(set)
with gzip.open(GOA_FILE, 'rt') as fh:
    for line in fh:
        if line.startswith('!'): continue
        parts = line.strip().split('\t')
        if len(parts) < 9: continue
        uid  = parts[1]; go_id = parts[4]; aspect = parts[8]; qual = parts[3]
        if aspect != 'F' or 'NOT' in qual: continue
        uniprot2go[uid].add(go_id)
print(f"  {len(uniprot2go)} UniProt entries with MF GO", flush=True)

# ── 2. Load labels ───────────────────────────────────────────────────────
print("\n[2] Loading labels...", flush=True)

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

ENSG2SYM = {}
with open(os.path.join(ID_DIR, 'ensembl_to_symbol.txt')) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]
te_isoforms  = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
te_iso_list  = [clean(x) for x in te_isoforms]
n_test       = len(te_iso_list)
print(f"  Test isoforms: {n_test}", flush=True)

sym2id = {}
with gzip.open(os.path.join(ANNOT_DIR, 'Homo_sapiens.gene_info.gz'), 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id: sym2id[syn] = p[1]

go_all = defaultdict(set)
with gzip.open(os.path.join(ANNOT_DIR, 'gene2go.gz'), 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        gid, go_id, cat_raw = p[1], p[2], p[7]
        if cat_raw != 'Function': continue
        go_all[go_id].add(gid)

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 1: mf_terms.append(p[0])

Y_te = np.stack([
    np.array([1.0 if sym2id.get(s, '__') in go_all[go_id] else 0.0
              for s in te_sym_list], dtype=np.float32)
    for go_id in mf_terms
], axis=1)
valid_mask = Y_te.sum(0) >= 2
mf_valid   = [go for go, v in zip(mf_terms, valid_mask) if v]
Y_te_v     = Y_te[:, valid_mask]
print(f"  {len(mf_valid)}/{len(mf_terms)} valid MF terms", flush=True)

L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural': L2_TERMS.add(p[0])
l2_idx = [i for i, go in enumerate(mf_valid) if go in L2_TERMS]
l2_set = set(l2_idx)

# ── 3. Parse BLAST hits ──────────────────────────────────────────────────
print("\n[3] Parsing BLAST hits...", flush=True)
_p_suffix = re.compile(r'\.p\d+$')

query_go_score = defaultdict(lambda: defaultdict(float))
with open(BLAST_OUT) as fh:
    for line in fh:
        parts = line.strip().split('\t')
        if len(parts) < 3: continue
        raw_qid = parts[0]
        qid     = _p_suffix.sub('', raw_qid)
        sseqid  = parts[1]; bitscore = float(parts[2])
        evalue  = float(parts[3]) if len(parts) > 3 else 1.0
        if evalue > 1e-3: continue
        sp_parts = sseqid.split('|')
        uniprot_acc = sp_parts[1] if len(sp_parts) >= 2 else sseqid
        for go_id in uniprot2go.get(uniprot_acc, []):
            if bitscore > query_go_score[qid][go_id]:
                query_go_score[qid][go_id] = bitscore

# Build BLAST prediction matrix
blast_preds = np.zeros((n_test, len(mf_valid)), dtype=np.float32)
for i, iso_id in enumerate(te_iso_list):
    d = query_go_score.get(iso_id, {})
    if d:
        for j, go_id in enumerate(mf_valid):
            v = d.get(go_id, 0.0)
            if v > 0: blast_preds[i, j] = v

# Identify covered vs uncovered
covered_mask   = blast_preds.max(axis=1) > 0    # has any BLAST GO prediction
uncovered_mask = ~covered_mask

n_covered   = covered_mask.sum()
n_uncovered = uncovered_mask.sum()
print(f"  BLAST covered:   {n_covered:,} / {n_test:,} ({n_covered/n_test:.1%})", flush=True)
print(f"  BLAST uncovered: {n_uncovered:,} / {n_test:,} ({n_uncovered/n_test:.1%})", flush=True)

# ── 4. Load PRISM v17f* predictions ─────────────────────────────────────
print("\n[4] Loading PRISM v17f* predictions...", flush=True)
prism_preds = np.load(PRISM_PREDS)
print(f"  PRISM preds shape: {prism_preds.shape}", flush=True)

# The PRISM preds may have 81 valid terms; align to mf_valid order
# Check if shapes match
assert prism_preds.shape == (n_test, len(mf_valid)), \
    f"Shape mismatch: prism {prism_preds.shape} vs expected ({n_test}, {len(mf_valid)})"

# ── 5. Compute AUPRC ─────────────────────────────────────────────────────
def fast_auprc(y_true, y_score):
    order  = np.argsort(-y_score, kind='quicksort')
    y_true = y_true[order]
    n_pos  = y_true.sum()
    if n_pos == 0: return float('nan')
    tp   = np.cumsum(y_true)
    prec = tp / np.arange(1, len(y_true) + 1, dtype=np.float64)
    rec  = tp / n_pos
    return float(np.dot(prec, np.diff(rec, prepend=0.0)))

def macro_ap_subset(preds_mat, Y_mat, row_mask, l2_set=None):
    preds_sub = preds_mat[row_mask]
    Y_sub     = Y_mat[row_mask]
    aps_all, aps_l2 = [], []
    for j in range(Y_sub.shape[1]):
        if Y_sub[:, j].sum() < 2: continue
        ap = fast_auprc(Y_sub[:, j], preds_sub[:, j])
        if np.isnan(ap): continue
        aps_all.append(ap)
        if l2_set and j in l2_set: aps_l2.append(ap)
    all_ap = float(np.mean(aps_all)) if aps_all else float('nan')
    l2_ap  = float(np.mean(aps_l2)) if aps_l2 else float('nan')
    return all_ap, l2_ap, len(aps_all)

print("\n[5] Computing AUPRC by coverage group...", flush=True)

# PRISM on ALL isoforms (sanity check)
prism_all, prism_l2_all, n_terms = macro_ap_subset(prism_preds, Y_te_v, np.ones(n_test, dtype=bool), l2_set)
print(f"\n  PRISM v17f* ALL ({n_test:,} isoforms):     All MF={prism_all:.4f}  L2={prism_l2_all:.4f}  ({n_terms} valid terms)")

# PRISM on COVERED isoforms
prism_cov, prism_l2_cov, n_terms_cov = macro_ap_subset(prism_preds, Y_te_v, covered_mask, l2_set)
print(f"  PRISM v17f* COVERED ({n_covered:,}):         All MF={prism_cov:.4f}  L2={prism_l2_cov:.4f}  ({n_terms_cov} valid terms)")

# PRISM on UNCOVERED isoforms
prism_unc, prism_l2_unc, n_terms_unc = macro_ap_subset(prism_preds, Y_te_v, uncovered_mask, l2_set)
print(f"  PRISM v17f* UNCOVERED ({n_uncovered:,}):       All MF={prism_unc:.4f}  L2={prism_l2_unc:.4f}  ({n_terms_unc} valid terms)")

# BLAST on COVERED isoforms (sanity: should be 0.861)
blast_cov, blast_l2_cov, _ = macro_ap_subset(blast_preds, Y_te_v, covered_mask, l2_set)
print(f"\n  BLAST covered-only AUPRC:                 All MF={blast_cov:.4f}  L2={blast_l2_cov:.4f}")
blast_all_check, _, _ = macro_ap_subset(blast_preds, Y_te_v, np.ones(n_test, dtype=bool), l2_set)
print(f"  BLAST all-isoforms AUPRC:                 All MF={blast_all_check:.4f} (expected 0.861)")

# BLAST on UNCOVERED (should be near 0 or chance since scores are 0)
blast_unc, blast_l2_unc, _ = macro_ap_subset(blast_preds, Y_te_v, uncovered_mask, l2_set)
print(f"  BLAST uncovered AUPRC:                    All MF={blast_unc:.4f}  L2={blast_l2_unc:.4f} (expected ~0 by construction)")

# ── 6. Check positive class distribution in uncovered ──────────────────
print("\n[6] Positive label counts in uncovered subset...", flush=True)
Y_unc = Y_te_v[uncovered_mask]
pos_per_term = Y_unc.sum(0)
terms_with_pos = (pos_per_term >= 2).sum()
frac_pos = Y_unc.mean()
print(f"  Terms with ≥2 positives in uncovered subset: {terms_with_pos}/{len(mf_valid)}")
print(f"  Mean positive rate in uncovered subset: {frac_pos:.3f} (vs overall {Y_te_v.mean():.3f})")

# ── 7. Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUMMARY: MC2 BLAST-uncovered subset analysis")
print("=" * 65)
print(f"  N isoforms (total):    {n_test:,}")
print(f"  BLAST covered:         {n_covered:,} ({n_covered/n_test:.1%})")
print(f"  BLAST uncovered:       {n_uncovered:,} ({n_uncovered/n_test:.1%})")
print()
print(f"  {'Method':<35} {'All MF':>8} {'L2':>8}")
print(f"  {'-'*35} {'-'*8} {'-'*8}")
print(f"  {'PRISM v17f* (all isoforms)':<35} {prism_all:>8.4f} {prism_l2_all:>8.4f}")
print(f"  {'PRISM v17f* (BLAST-covered subset)':<35} {prism_cov:>8.4f} {prism_l2_cov:>8.4f}")
print(f"  {'PRISM v17f* (BLAST-uncovered subset)':<35} {prism_unc:>8.4f} {prism_l2_unc:>8.4f}")
print(f"  {'BLAST→GOA (all isoforms)':<35} {blast_all_check:>8.4f}   (0 for uncovered)")
print(f"  {'BLAST→GOA (covered-only)':<35} {blast_cov:>8.4f} {blast_l2_cov:>8.4f}")
print(f"  {'BLAST→GOA (uncovered=0)':<35}  0.0000  0.0000 (by construction)")
print()
print(f"  PRISM vs BLAST on uncovered: PRISM {prism_unc:.4f} vs BLAST 0.0000")
if prism_unc > 0:
    print(f"  → PRISM provides {prism_unc:.4f} AUPRC where BLAST gives 0")

# Save results
result = {
    'n_total': int(n_test),
    'n_blast_covered': int(n_covered),
    'n_blast_uncovered': int(n_uncovered),
    'pct_uncovered': float(n_uncovered / n_test),
    'prism_all': float(prism_all),
    'prism_l2_all': float(prism_l2_all),
    'prism_covered': float(prism_cov),
    'prism_l2_covered': float(prism_l2_cov),
    'prism_uncovered': float(prism_unc),
    'prism_l2_uncovered': float(prism_l2_unc),
    'blast_all': float(blast_all_check),
    'blast_covered': float(blast_cov),
    'blast_uncovered': 0.0,
    'n_valid_terms_uncovered': int(n_terms_unc),
    'frac_positive_uncovered': float(frac_pos),
}

OUT = '../../reports/mc2_blast_uncovered.json'
with open(OUT, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n  Results saved → {OUT}")
