#!/usr/bin/env python3
"""
exp_isoform_discrimination.py
==============================
Within-gene isoform discrimination evaluation.

Standard gene-level AUPRC (as in SOTA benchmark) cannot distinguish methods
that make isoform-specific predictions from methods that assign identical
scores to all isoforms of the same gene. This script evaluates:

1. Intra-gene prediction variance (CV) — how much does each method
   differentiate between isoforms of the same gene?

2. Within-gene AUC (sequence-length proxy) — for each gene, rank
   isoforms by prediction score; ground truth = longer protein is
   "more complete" → expected to retain more functional domains.
   AUC measures whether predicted ranking agrees with length ranking.

3. Delta score |max - min| within gene for each GO term — measures
   isoform-specific signal magnitude.

4. BISECT-case discrimination — for curated isoform-switch genes,
   does the method rank the functional isoform above the truncated one?

Methods compared:
  PRISM v17f*, BLAST→GOA, k-NN (ESM-2), gene-mean, random

Output: reports/exp_isoform_discrimination/results.json + table.tsv
"""

import os, sys, gzip, json, re, time
import numpy as np
from collections import defaultdict
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/exp_isoform_discrimination'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("  Within-Gene Isoform Discrimination Evaluation")
print("=" * 70)

# ── 1. Load test set identity ─────────────────────────────────────────
print("\n[1] Loading test set...")

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]
te_isoforms  = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
te_iso_list  = [clean(x) for x in te_isoforms]
n_isoforms   = len(te_iso_list)

gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs[g].append(i)
multi_gene_list = [g for g, idxs in gene2idxs.items() if len(idxs) >= 2]
print(f"  {n_isoforms} isoforms, {len(gene2idxs)} genes")
print(f"  Genes with ≥2 isoforms: {len(multi_gene_list)}")

# ── 2. Load GO labels ─────────────────────────────────────────────────
print("\n[2] Loading GO MF labels...")
sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id: sym2id[syn] = p[1]

go_all = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
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
        if len(p) >= 6: mf_terms.append(p[0])

Y_te = np.stack([
    np.array([1.0 if sym2id.get(s, '__') in go_all[go_id] else 0.0
              for s in te_sym_list], dtype=np.float32)
    for go_id in mf_terms
], axis=1)
valid_mask = Y_te.sum(0) >= 2
mf_valid   = [go for go, v in zip(mf_terms, valid_mask) if v]
Y_te_v     = Y_te[:, valid_mask]  # (36748, 81)
n_terms    = len(mf_valid)
print(f"  {n_terms} valid MF terms")

# L2 subset
L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural': L2_TERMS.add(p[0])
l2_idx = [i for i, go in enumerate(mf_valid) if go in L2_TERMS]

# ── 3. Load protein sequence lengths (proxy for domain completeness) ──
print("\n[3] Loading sequence lengths (functional completeness proxy)...")
pep_file = '../data/top30k_isoforms.pep'
seq_lens = {}
cur_id, cur_len = None, 0
with open(pep_file) as fh:
    for line in fh:
        line = line.strip()
        if line.startswith('>'):
            if cur_id: seq_lens[cur_id] = cur_len
            # Header: >BambuTx10.p1 GENE.BambuTx10~~...
            raw_id = line[1:].split()[0]
            # Strip .p1 suffix → BambuTx10
            cur_id = re.sub(r'\.p\d+$', '', raw_id)
            cur_len = 0
        else:
            cur_len += len(line)
if cur_id: seq_lens[cur_id] = cur_len

# Align with test set
iso_lengths = np.array([seq_lens.get(iso, 0) for iso in te_iso_list], dtype=np.float32)
print(f"  Lengths loaded: {(iso_lengths > 0).sum()}/{n_isoforms}")
print(f"  Length range: {iso_lengths[iso_lengths>0].min():.0f}~{iso_lengths.max():.0f} aa")

# ── 4. Load prediction matrices ───────────────────────────────────────
print("\n[4] Loading prediction matrices...")

# PRISM v17f* (82 columns, includes 1 invalid → use valid_mask)
prism_raw = np.load('../../reports/v17f_star_bootstrap/v17f_star_preds.npy').astype(np.float32)
# Check alignment: prism has 82 columns, Y_te_v has 81 (valid terms)
if prism_raw.shape[1] == 82 and valid_mask.shape[0] == 82:
    prism_preds = prism_raw[:, valid_mask]
elif prism_raw.shape[1] == len(mf_valid):
    prism_preds = prism_raw
else:
    raise ValueError(f"PRISM shape mismatch: {prism_raw.shape}, valid={valid_mask.sum()}")
print(f"  PRISM v17f*: {prism_preds.shape}")

# BLAST→GOA (rebuild from blast_hits.tsv + GOA)
print("  Building BLAST score matrix...")
GOA_FILE = '../../reports/exp_e_sota/swissprot_db/goa_human.gaf.gz'
BLAST_TSV = '../../reports/benchmark_external/blast_goa/blast_hits.tsv'

uniprot2go_mf = defaultdict(set)
with gzip.open(GOA_FILE, 'rt') as fh:
    for line in fh:
        if line.startswith('!'): continue
        parts = line.strip().split('\t')
        if len(parts) < 9: continue
        uid = parts[1]; go_id = parts[4]; aspect = parts[8]; qual = parts[3]
        if aspect != 'F' or 'NOT' in qual: continue
        uniprot2go_mf[uid].add(go_id)

_p_suffix = re.compile(r'\.p\d+$')
query_go_score = defaultdict(lambda: defaultdict(float))
with open(BLAST_TSV) as fh:
    for line in fh:
        parts = line.strip().split('\t')
        if len(parts) < 4: continue
        qid = _p_suffix.sub('', parts[0])
        bitscore = float(parts[2])
        evalue = float(parts[3])
        if evalue > 1e-3: continue
        sp_parts = parts[1].split('|')
        uid = sp_parts[1] if len(sp_parts) >= 2 else parts[1]
        for go_id in uniprot2go_mf.get(uid, []):
            if bitscore > query_go_score[qid][go_id]:
                query_go_score[qid][go_id] = bitscore

blast_preds = np.zeros((n_isoforms, n_terms), dtype=np.float32)
for i, iso_id in enumerate(te_iso_list):
    d = query_go_score.get(iso_id, {})
    for j, go_id in enumerate(mf_valid):
        blast_preds[i, j] = d.get(go_id, 0.0)
print(f"  BLAST→GOA: {blast_preds.shape}")

# k-NN (ESM-2 L30, k=5)
print("  Building k-NN predictions...")
X30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X30_te = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes = [clean(g) for g in tr_genes_raw]
tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

tr_ids = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_ids)
go_tr = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        gid, go_id, cat_raw = p[1], p[2], p[7]
        if cat_raw != 'Function': continue
        if gid in tr_id_set: go_tr[go_id].add(gid)

Y_tr_v = np.stack([
    np.array([1.0 if any(sym2id.get(g, g) in go_tr[go_id]
                          for g in tr_genes[i:i+1]) else 0.0
              for i in range(len(tr_genes))], dtype=np.float32)
    for go_id in mf_valid
], axis=1)

# Cosine similarity → k-NN
from sklearn.preprocessing import normalize
Xtr_n = normalize(X30_tr)
Xte_n = normalize(X30_te)
CHUNK = 4096; K = 5
knn_preds = np.zeros((n_isoforms, n_terms), dtype=np.float32)
for start in range(0, n_isoforms, CHUNK):
    end = min(start + CHUNK, n_isoforms)
    sims = Xte_n[start:end] @ Xtr_n.T
    topk_idx = np.argpartition(-sims, K, axis=1)[:, :K]
    for i_local in range(end - start):
        knn_preds[start + i_local] = Y_tr_v[topk_idx[i_local]].mean(0)
print(f"  k-NN: {knn_preds.shape}")

# Gene-mean (train set gene centroids)
tr_gene_set_unique = sorted(set(tr_genes))
gene_centroid = {}
for g in tr_gene_set_unique:
    idxs = tr_sym2idx[g]
    gene_centroid[g] = X30_tr[idxs].mean(0)
gene_label = {}
for g in tr_gene_set_unique:
    gid = sym2id.get(g, g)
    gene_label[g] = np.array([1.0 if gid in go_tr[go_id] else 0.0
                               for go_id in mf_valid], dtype=np.float32)

centroids_mat = np.stack(list(gene_centroid.values()))
centroid_genes = list(gene_centroid.keys())
cent_n = normalize(centroids_mat)
gene_mean_preds = np.zeros((n_isoforms, n_terms), dtype=np.float32)
for start in range(0, n_isoforms, CHUNK):
    end = min(start + CHUNK, n_isoforms)
    sims = Xte_n[start:end] @ cent_n.T
    best_gene_idx = np.argmax(sims, axis=1)
    for i_local in range(end - start):
        g_best = centroid_genes[best_gene_idx[i_local]]
        gene_mean_preds[start + i_local] = gene_label[g_best]
print(f"  Gene-mean: {gene_mean_preds.shape}")

# Random baseline
rng = np.random.default_rng(42)
random_preds = rng.random((n_isoforms, n_terms)).astype(np.float32)

methods = {
    'PRISM_v17f*': prism_preds,
    'BLAST_GOA':   blast_preds,
    'kNN_ESM2':    knn_preds,
    'Gene_mean':   gene_mean_preds,
    'Random':      random_preds,
}

# ── 5. Metric A: Intra-gene prediction variance (CV) ─────────────────
print("\n[5] Metric A: Intra-gene prediction variance...")
print("  (CV = std/mean within gene across isoforms; higher = more discriminative)")

cv_results = {}
delta_results = {}
for name, preds in methods.items():
    cvs, deltas = [], []
    for g in multi_gene_list:
        idxs = gene2idxs[g]
        p_g = preds[idxs]   # (k, n_terms)
        # Only for terms where gene is positive (labeled functional)
        gene_pos_terms = np.where(Y_te_v[idxs[0]] > 0)[0]
        if len(gene_pos_terms) == 0: continue
        p_pos = p_g[:, gene_pos_terms]  # (k, n_pos_terms)
        # CV per term, then mean
        means = p_pos.mean(0)
        stds  = p_pos.std(0)
        # Avoid div-by-zero
        valid = means > 0
        if valid.sum() > 0:
            cv = (stds[valid] / means[valid]).mean()
            cvs.append(float(cv))
        # Delta (max - min) per positive term, mean
        delta = (p_pos.max(0) - p_pos.min(0)).mean()
        deltas.append(float(delta))
    cv_results[name]    = float(np.mean(cvs)) if cvs else 0.0
    delta_results[name] = float(np.mean(deltas)) if deltas else 0.0
    print(f"  {name:20s}: mean CV={cv_results[name]:.4f}  mean Δ={delta_results[name]:.4f}")

# ── 6. Metric B: Within-gene length-proxy AUC ────────────────────────
print("\n[6] Metric B: Within-gene ranking vs sequence length (AUC)...")
print("  (AUC > 0.5 = method ranks longer/more-complete isoforms higher)")
print("  (BLAST should approach 0.5 since it assigns same score to same-gene isoforms)")

from sklearn.metrics import roc_auc_score

def within_gene_length_auc(preds_mat, gene_list, gene2idxs, iso_lengths, Y_te_v, n_min_isoforms=2):
    """
    For each gene with ≥n_min_isoforms and variable lengths:
    - Ground truth: rank by length (longer = more complete = proxy for functional)
    - Prediction: method's predicted score for the gene's positive GO terms
    Compute AUC per gene per positive term, aggregate.
    """
    aucs = []
    for g in gene_list:
        idxs = gene2idxs[g]
        if len(idxs) < n_min_isoforms: continue
        lengths = iso_lengths[idxs]
        if lengths.std() < 1: continue  # all same length → skip
        gene_pos_terms = np.where(Y_te_v[idxs[0]] > 0)[0]
        if len(gene_pos_terms) == 0: continue
        p_g = preds_mat[idxs]  # (k, n_terms)
        len_binary = (lengths >= np.median(lengths)).astype(float)  # longer half = positive
        if len_binary.sum() == 0 or len_binary.sum() == len(idxs): continue
        # AUC for each positive GO term
        for t in gene_pos_terms:
            scores = p_g[:, t]
            if scores.std() < 1e-8:
                aucs.append(0.5)  # constant predictions → random
                continue
            try:
                auc = roc_auc_score(len_binary, scores)
                aucs.append(auc)
            except:
                pass
    return np.mean(aucs) if aucs else 0.5, len(aucs)

wg_aucs = {}
for name, preds in methods.items():
    auc, n = within_gene_length_auc(preds, multi_gene_list, gene2idxs, iso_lengths, Y_te_v)
    wg_aucs[name] = auc
    print(f"  {name:20s}: within-gene length AUC = {auc:.4f}  (N={n})")

# ── 7. Metric C: % genes where method agrees with BLAST on top isoform ─
print("\n[7] Metric C: Top-isoform agreement with BLAST (per gene, per GO term)...")
print("  (which fraction of genes: method ranks same isoform #1 as BLAST?)")

def top1_agreement_with_blast(preds_mat, blast_mat, gene_list, gene2idxs, Y_te_v):
    agrees, total = 0, 0
    for g in gene_list:
        idxs = gene2idxs[g]
        if len(idxs) < 2: continue
        gene_pos_terms = np.where(Y_te_v[idxs[0]] > 0)[0]
        blast_g = blast_mat[idxs]
        pred_g  = preds_mat[idxs]
        for t in gene_pos_terms:
            blast_scores = blast_g[:, t]
            pred_scores  = pred_g[:, t]
            if blast_scores.max() == blast_scores.min(): continue  # BLAST tied → skip
            if pred_scores.max() == pred_scores.min(): continue
            blast_top = np.argmax(blast_scores)
            pred_top  = np.argmax(pred_scores)
            agrees += int(blast_top == pred_top)
            total  += 1
    return agrees / total if total > 0 else 0.0, total

top1_agree = {}
for name, preds in methods.items():
    if name == 'BLAST_GOA': continue
    agr, n = top1_agreement_with_blast(preds, blast_preds, multi_gene_list, gene2idxs, Y_te_v)
    top1_agree[name] = agr
    print(f"  {name:20s}: top-1 agreement with BLAST = {agr:.4f}  (N={n})")

# ── 8. Metric D: BLAST within-gene uniformity ────────────────────────
print("\n[8] Metric D: BLAST uniformity (fraction of genes with IDENTICAL scores)...")
n_uniform_genes, n_multi_genes = 0, 0
for g in multi_gene_list:
    idxs = gene2idxs[g]
    if len(idxs) < 2: continue
    b_g = blast_preds[idxs]
    gene_pos_terms = np.where(Y_te_v[idxs[0]] > 0)[0]
    if len(gene_pos_terms) == 0: continue
    b_pos = b_g[:, gene_pos_terms]
    # Uniform if std across isoforms ≈ 0
    if b_pos.std(axis=0).max() < 1e-6:
        n_uniform_genes += 1
    n_multi_genes += 1
print(f"  BLAST: {n_uniform_genes}/{n_multi_genes} genes ({100*n_uniform_genes/n_multi_genes:.1f}%) "
      f"have IDENTICAL scores for ALL positive GO terms")

# ── 9. BISECT-based discrimination ───────────────────────────────────
print("\n[9] Metric E: BISECT curated case discrimination...")
# Load BISECT results to find dominant/subordinate isoform pairs
bisect_genes = {
    # Gene: (functional_iso_hint, truncated_iso_hint, relevant_GO_terms)
    # From BISECT analysis: dominant isoform should score higher
    'NDUFS4': None,  # load from BISECT data
    'NDUFS7': None,
    'NDUFS8': None,
    'DLG1':   None,
    'DOCK11': None,
}

# Load BISECT case file if available
bisect_file = '../../Final_analysis/pipeline_bioanalysis/cases_v1x_rerun.csv'
if os.path.exists(bisect_file):
    import csv
    with open(bisect_file) as f:
        reader = csv.DictReader(f)
        bisect_cases = list(reader)
    print(f"  Loaded {len(bisect_cases)} BISECT cases")

    # For each BISECT case: find gene in test set, compare isoform predictions
    bisect_discrimination = []
    for case in bisect_cases[:20]:  # sample first 20
        gene_name = case.get('gene_name', case.get('gene', ''))
        if not gene_name or gene_name not in gene2idxs: continue
        idxs = gene2idxs[gene_name]
        if len(idxs) < 2: continue
        gene_pos_terms = np.where(Y_te_v[idxs[0]] > 0)[0]
        if len(gene_pos_terms) == 0: continue

        n_terms_this = len(gene_pos_terms)
        for name, preds in methods.items():
            p_g = preds[idxs][:, gene_pos_terms]
            p_range = (p_g.max(0) - p_g.min(0)).mean()
            bisect_discrimination.append({
                'gene': gene_name, 'method': name, 'prediction_range': float(p_range)
            })

    if bisect_discrimination:
        from collections import defaultdict
        method_ranges = defaultdict(list)
        for d in bisect_discrimination:
            method_ranges[d['method']].append(d['prediction_range'])
        print("  Mean prediction range per method (BISECT genes):")
        for nm, ranges in method_ranges.items():
            print(f"    {nm:20s}: {np.mean(ranges):.4f}")
else:
    print(f"  BISECT case file not found: {bisect_file}")
    bisect_cases = []

# ── 10. Compile and save ──────────────────────────────────────────────
print("\n[10] Summary...")
print(f"\n{'Method':<20} {'CV':>8} {'Δ score':>9} {'WG-AUC':>8} {'Uniform%':>9}")
print("-" * 60)
for name in methods:
    uniform_pct = 100.0 if name == 'BLAST_GOA' else 0.0  # shown above for BLAST
    if name == 'BLAST_GOA':
        uniform_pct = 100 * n_uniform_genes / n_multi_genes
    print(f"{name:<20} {cv_results[name]:>8.4f} {delta_results[name]:>9.4f} "
          f"{wg_aucs[name]:>8.4f} {'N/A':>9}")

results = {
    'intra_gene_cv': cv_results,
    'intra_gene_delta': delta_results,
    'within_gene_length_auc': wg_aucs,
    'blast_uniform_fraction': n_uniform_genes / n_multi_genes if n_multi_genes > 0 else 0,
    'n_multi_gene': n_multi_genes,
    'n_bisect_cases_tested': len([d for d in bisect_discrimination if d['method'] == 'PRISM_v17f*']),
}
with open(f'{OUT_DIR}/results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"\nSaved → {OUT_DIR}/results.json")

# ── TSV table ─────────────────────────────────────────────────────────
import csv
with open(f'{OUT_DIR}/isoform_discrimination_table.tsv', 'w', newline='') as fh:
    writer = csv.writer(fh, delimiter='\t')
    writer.writerow(['Method', 'Intra_gene_CV', 'Mean_delta', 'WG_length_AUC', 'BLAST_uniform_pct'])
    blast_unif = f"{100*n_uniform_genes/n_multi_genes:.1f}%" if n_multi_genes > 0 else 'N/A'
    for name in methods:
        writer.writerow([
            name, f"{cv_results[name]:.4f}", f"{delta_results[name]:.4f}",
            f"{wg_aucs[name]:.4f}",
            blast_unif if name == 'BLAST_GOA' else 'N/A'
        ])
print(f"Saved → {OUT_DIR}/isoform_discrimination_table.tsv")
