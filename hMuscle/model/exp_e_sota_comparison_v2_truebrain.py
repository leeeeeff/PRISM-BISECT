#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_e_sota_comparison_v2_truebrain.py
---------------------------------------
TISSUE-MISLABELING BUGFIX RERUN (2026-07-14).
Original exp_e_sota_comparison.py loaded my_gene_list_fixed.npy / esm2_layer_30_t30_150M.npy
as its "test" set -- these are MUSCLE data (BambuTx IDs, 36748 isoforms), despite being
described in the manuscript as "brain zero-shot". This rerun re-points the TEST side only
at the TRUE brain isoform set (brain_full_gene_names.npy / brain_full_esm2_layer30 / brain_full_ids.npy,
63994 isoforms / 18514 unique genes, IsoQuant IDs like A1BG-204). Training side
(train_gene_list.npy etc.) is UNCHANGED. The E1 BLAST query FASTA is also re-pointed:
original built its query set from brain_esm2/brain_only_transcripts.fa.transdecoder.pep,
which covers ONLY the 32,851 NOVEL brain transcripts (not the full 63,994 isoform set) --
this rerun instead uses the TRUE-BRAIN COMPLETE protein FASTA built in Step 1 of this
rerun (reports/truebrain_rerun_20260714/data/brain_full_proteins.fa, 53826/63994=84.1%
protein-coding coverage spanning BOTH novel and known/annotated isoforms, matching
brain_full_mask.npy exactly), in true brain_full_ids.npy embedding order. The cached
Swiss-Prot BLAST DB / GOA (reports/exp_e_sota/swissprot_db/) is tissue-agnostic and reused
as-is per task instructions -- only the query side changes.

Original (mislabeled-as-brain, actually muscle): exp_e_sota_comparison.py, backed up
at exp_e_sota_comparison_backup_20260714.py before this rerun was created.

exp_e_sota_comparison.py
========================
Exp E: SOTA comparison baselines for brain MF GO prediction (82 terms).

Baselines:
  E0  — ESM-2 k-NN retrieval (k=5, cosine, train→test, no learning)
  E1  — BLAST + Swiss-Prot GO transfer (best-hit, e-value < 1e-4)
  E2  — Domain LR (Pfam-based logistic regression, already computed: 0.1625 All MF)

Reference (pre-computed):
  PRISM (v15d_bp_clean, frozen L30 MLP) = 0.596 All MF
  v17f  (δ_layer T_ψ + Stage 2)         = 0.717 All MF

All AUPRC values computed on the TRUE brain 82 MF GO term test set
(63,994 brain isoforms / 53,826 protein-coding, gene2go annotations, ≥2 positives per term).
"""

import os, sys, gzip, time, json, subprocess, urllib.request
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────
# Paths (mirror v17f_layer_delta.py)
# ─────────────────────────────────────────────────────────────────
DATA_DIR   = '../data'
BRAIN_DIR  = '../data/brain_isoquant_esm2/full'
ID_DIR     = '../data/raw_data/data/id_lists'
ANNOT_DIR  = '../data/raw_data/data/annotations'
FEAT_DIR   = '../results_isoform/features'
TRUEBRAIN_PEP = '../../reports/truebrain_rerun_20260714/data/brain_full_proteins.fa'
OUT_DIR    = '../../reports/truebrain_rerun_20260714/exp_e_sota'
os.makedirs(OUT_DIR, exist_ok=True)

LAYER_B   = 30
LAYER_A   = 15

# ─────────────────────────────────────────────────────────────────
# Reference values from prior experiments
# ─────────────────────────────────────────────────────────────────
PRISM_REF   = {'All MF': 0.5962, 'L2_Structural': 0.3127}
V17F_REF    = {'All MF': 0.7173, 'L2_Structural': 0.6219}
DOMAIN_LR   = {'All MF': 0.1625, 'L2_Structural': None}

print("=" * 65)
print("  Exp E: SOTA comparison — brain MF GO prediction (82 terms)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Load ESM-2 embeddings
# ─────────────────────────────────────────────────────────────────
print("\n[1] Loading ESM-2 L30 embeddings...")

X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l30_te = np.load(f'{BRAIN_DIR}/brain_full_esm2_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)

print(f"  Train L30: {X_l30_tr.shape}  Test L30: {X_l30_te.shape}")

# ─────────────────────────────────────────────────────────────────
# 2. IDs and gene symbols
# ─────────────────────────────────────────────────────────────────
print("\n[2] Loading gene IDs...")

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

# TRUE BRAIN: gene names already symbols (e.g. 'A1BG'), no ENSG2SYM mapping needed.
te_genes_raw = np.load(f'{BRAIN_DIR}/brain_full_gene_names.npy', allow_pickle=True)
te_sym_list  = [clean(g) for g in te_genes_raw]
# te_enst_list renamed conceptually to "te_iso_list" (IsoQuant IDs, e.g. A1BG-204) --
# kept variable name te_enst_list for minimal-diff compatibility with rest of script,
# used only for FASTA query matching below (Section E1).
te_enst_list = [clean(g) for g in np.load(f'{BRAIN_DIR}/brain_full_ids.npy', allow_pickle=True)]

gene_arr_tr = np.array(tr_genes)
gene_arr_te = np.array(te_sym_list)

print(f"  Train: {len(tr_genes)} isoforms  Test: {len(te_sym_list)} isoforms")

# ─────────────────────────────────────────────────────────────────
# 3. GO labels (82 MF terms — same as v17f)
# ─────────────────────────────────────────────────────────────────
print("\n[3] Loading GO MF labels...")

sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id:
                        sym2id[syn] = p[1]

go_genes_tr  = defaultdict(set)
go_genes_all = defaultdict(set)
tr_ids       = [sym2id.get(g, g) for g in tr_genes]
tr_id_set    = set(tr_ids)
tr_sym2idx   = defaultdict(list)
for i, g in enumerate(tr_genes):
    tr_sym2idx[g].append(i)

with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        gid, go_id, cat_raw = p[1], p[2], p[7]
        if cat_raw != 'Function': continue
        go_genes_all[go_id].add(gid)
        if gid in tr_id_set:
            go_genes_tr[go_id].add(gid)

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])

def build_Y_tr(go_id):
    pos_ids  = go_genes_tr[go_id]
    pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in pos_ids}
    y = np.zeros(len(tr_genes), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []):
            y[idx] = 1.0
    return y

def build_Y_te(go_id):
    pos_ids = go_genes_all[go_id]
    return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0
                     for s in te_sym_list], dtype=np.float32)

Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)
Y_te = np.stack([build_Y_te(go) for go in mf_terms], axis=1)

valid_mask = Y_te.sum(0) >= 2
mf_valid   = [go for go, v in zip(mf_terms, valid_mask) if v]
Y_te_v     = Y_te[:, valid_mask]
Y_tr_v     = Y_tr[:, valid_mask]
print(f"  {len(mf_terms)} MF terms  |  valid (≥2 positives): {valid_mask.sum()}")

# H2 classification
L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural':
            L2_TERMS.add(p[0])

l2_idx = [i for i, go in enumerate(mf_valid) if go in L2_TERMS]
print(f"  L2_Structural terms: {len(l2_idx)}/{len(mf_valid)}")

def eval_auprc(scores_2d, labels_2d, l2_idx):
    """Macro AUPRC for All and L2_Structural subsets."""
    n_terms = labels_2d.shape[1]
    auprcs_all = []
    for j in range(n_terms):
        y = labels_2d[:, j]
        if y.sum() < 2: continue
        auprcs_all.append(average_precision_score(y, scores_2d[:, j]))
    auprc_all = float(np.mean(auprcs_all))

    auprcs_l2 = []
    for j in l2_idx:
        y = labels_2d[:, j]
        if y.sum() < 2: continue
        auprcs_l2.append(average_precision_score(y, scores_2d[:, j]))
    auprc_l2 = float(np.mean(auprcs_l2)) if auprcs_l2 else float('nan')

    return auprc_all, auprc_l2

# ─────────────────────────────────────────────────────────────────
# E0: ESM-2 k-NN GO label retrieval
# ─────────────────────────────────────────────────────────────────
print("\n[E0] ESM-2 k-NN GO retrieval (k=5, cosine similarity)...")
t0 = time.time()
K = 5

# L2-normalise for cosine similarity
def l2norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms

X_tr_n = l2norm(X_l30_tr)
X_te_n = l2norm(X_l30_te)

# Batch cosine similarity to avoid OOM (36,748 × 31,668 float32 ~ 4.3GB)
BATCH = 512
knn_scores = np.zeros((len(te_sym_list), Y_tr_v.shape[1]), dtype=np.float32)

for start in range(0, len(te_sym_list), BATCH):
    end  = min(start + BATCH, len(te_sym_list))
    sim  = X_te_n[start:end] @ X_tr_n.T  # (batch, n_train)
    top_k_idx = np.argpartition(sim, -K, axis=1)[:, -K:]
    for bi, te_i in enumerate(range(start, end)):
        nbrs = top_k_idx[bi]
        knn_scores[te_i] = Y_tr_v[nbrs].mean(axis=0)
    if start % (BATCH * 10) == 0:
        print(f"  k-NN progress: {start}/{len(te_sym_list)}", flush=True)

auprc_knn_all, auprc_knn_l2 = eval_auprc(knn_scores, Y_te_v, l2_idx)
t_knn = time.time() - t0
print(f"  E0 k-NN: All MF={auprc_knn_all:.4f}  L2_Structural={auprc_knn_l2:.4f}  [{t_knn:.0f}s]")

# ─────────────────────────────────────────────────────────────────
# E0b: Gene-mean ESM-2 baseline (cross-gene average → transfer)
# ─────────────────────────────────────────────────────────────────
print("\n[E0b] Gene-mean ESM-2 baseline (PRISM equivalent without MLP)...")
t0 = time.time()

# Compute gene centroids in training set
gene_centroids = {}
for sym, idxs in tr_sym2idx.items():
    gene_centroids[sym] = X_l30_tr[idxs].mean(axis=0)

# For each test isoform, score = max cosine similarity to a training gene centroid
# that shares the same GO term label
# This is the "gene-average retrieval" baseline
centroid_matrix = []
centroid_labels = []
centroid_genes  = []
for sym, cent in gene_centroids.items():
    idxs = tr_sym2idx[sym]
    lbl  = Y_tr_v[idxs].max(axis=0)  # gene-level label: positive if any isoform is positive
    centroid_matrix.append(cent)
    centroid_labels.append(lbl)
    centroid_genes.append(sym)

centroid_matrix = np.array(centroid_matrix, dtype=np.float32)
centroid_labels = np.array(centroid_labels, dtype=np.float32)
centroid_n      = l2norm(centroid_matrix)

genemean_scores = np.zeros((len(te_sym_list), Y_tr_v.shape[1]), dtype=np.float32)
for start in range(0, len(te_sym_list), BATCH):
    end  = min(start + BATCH, len(te_sym_list))
    sim  = X_te_n[start:end] @ centroid_n.T
    top_k_idx = np.argpartition(sim, -K, axis=1)[:, -K:]
    for bi in range(end - start):
        nbrs = top_k_idx[bi]
        genemean_scores[start + bi] = centroid_labels[nbrs].mean(axis=0)

auprc_gm_all, auprc_gm_l2 = eval_auprc(genemean_scores, Y_te_v, l2_idx)
t_gm = time.time() - t0
print(f"  E0b Gene-mean: All MF={auprc_gm_all:.4f}  L2_Structural={auprc_gm_l2:.4f}  [{t_gm:.0f}s]")

# ─────────────────────────────────────────────────────────────────
# E1: BLAST + Swiss-Prot GO transfer
# ─────────────────────────────────────────────────────────────────
print("\n[E1] BLAST + Swiss-Prot GO transfer...")

# SPROT_DIR points at the EXISTING cache (tissue-agnostic Swiss-Prot + GOA + BLAST DB,
# already built by the original muscle-mislabeled run) -- reused as-is per task
# instructions ("BLAST DB is ALREADY cached and tissue-agnostic ... reuse it as-is").
SPROT_DIR   = '../../reports/exp_e_sota/swissprot_db'
SPROT_FASTA = f'{SPROT_DIR}/uniprot_sprot.fasta.gz'
# NOTE (bugfix, pre-existing in original muscle script too, confirmed via diff): this
# constant used to point at 'goa_uniprot_sprot.gaf.gz', a filename that was NEVER actually
# cached (the EBI URL for it now 404s) -- the file that IS cached and IS used successfully
# elsewhere (domain_ranking_validation.py) is 'goa_human.gaf.gz' (same GAF format, human-only
# Swiss-Prot GOA). Repointing here to the file that actually exists.
GOA_FILE    = f'{SPROT_DIR}/goa_human.gaf.gz'
BLAST_DB    = f'{SPROT_DIR}/swissprot'
QUERY_FASTA = f'{OUT_DIR}/brain_test_proteins_truebrain.fasta'
BLAST_OUT   = f'{OUT_DIR}/blast_results_truebrain.tsv'
BLAST_BIN   = '/opt/blast/ncbi-blast-2.12.0+/bin/blastp'

os.makedirs(SPROT_DIR, exist_ok=True)

def download(url, dest):
    if os.path.exists(dest):
        print(f"  [skip] already exists: {os.path.basename(dest)}")
        return
    print(f"  Downloading {os.path.basename(dest)} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Done: {os.path.getsize(dest) // 1024 // 1024} MB")

# Download Swiss-Prot FASTA
download(
    'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz',
    SPROT_FASTA
)

# Download Swiss-Prot GOA (EBI)
download(
    'https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_sprot.gaf.gz',
    GOA_FILE
)

# Build BLAST database (only if needed)
if not os.path.exists(f'{BLAST_DB}.phr'):
    print("  Building BLAST database...")
    makedb_cmd = [
        '/opt/blast/ncbi-blast-2.12.0+/bin/makeblastdb',
        '-in', f'<(gunzip -c {SPROT_FASTA})',
        '-dbtype', 'prot',
        '-out', BLAST_DB,
        '-title', 'swissprot'
    ]
    # makeblastdb doesn't support process substitution directly; use a temp file
    SPROT_UNZIP = f'{SPROT_DIR}/uniprot_sprot.fasta'
    if not os.path.exists(SPROT_UNZIP):
        print("  Decompressing Swiss-Prot FASTA...")
        subprocess.run(f'gunzip -k {SPROT_FASTA}', shell=True, check=True)
    subprocess.run([
        '/opt/blast/ncbi-blast-2.12.0+/bin/makeblastdb',
        '-in', SPROT_UNZIP, '-dbtype', 'prot', '-out', BLAST_DB
    ], check=True)
    print("  BLAST database built.")
else:
    print("  [skip] BLAST database exists")

# Extract TRUE brain test protein sequences in ESM-2 embedding order.
# NOTE (bugfix): the original script built its query index from
# brain_esm2/brain_only_transcripts.fa.transdecoder.pep, which covers ONLY the
# 32,851 NOVEL brain transcripts -- known/annotated isoforms (the majority of the
# 63,994-isoform brain_full set) were silently written as MISSING. This rerun uses
# the COMPLETE true-brain protein FASTA from Step 1 of this rerun
# (brain_full_proteins.fa, 53826/63994 coding, built from BOTH TransDecoder-predicted
# novel ORFs AND canonical GENCODE protein sequences for known isoforms), indexed
# directly by brain_full_ids.npy IDs (e.g. 'A1BG-204') -- no ENST stripping needed.
if not os.path.exists(QUERY_FASTA):
    print("  Extracting TRUE brain test protein sequences (complete 63994-isoform set)...")

    # Build index: isoform_id (e.g. A1BG-204) → sequence, from brain_full_proteins.fa
    # (headers are '>{iso_id}.p1')
    pep_index = {}
    current_id, current_seq = None, []
    with open(TRUEBRAIN_PEP) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if current_id:
                    pep_index[current_id] = ''.join(current_seq)
                raw_id = line[1:].split()[0]
                current_id = raw_id.rsplit('.p', 1)[0] if '.p' in raw_id else raw_id
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            pep_index[current_id] = ''.join(current_seq)

    print(f"  Protein sequences indexed: {len(pep_index)} entries (true-brain complete FASTA)")

    written, missing = 0, 0
    with open(QUERY_FASTA, 'w') as out:
        for i, iso_id in enumerate(te_enst_list):
            if iso_id in pep_index:
                seq = pep_index[iso_id]
                # Write as index_i so we can map back to embedding order
                out.write(f'>{i}|{iso_id}\n{seq}\n')
                written += 1
            else:
                # Write dummy entry to preserve index alignment (non-coding per SQANTI3)
                out.write(f'>{i}|{iso_id}|MISSING\n')
                missing += 1
    print(f"  Written: {written}  Missing: {missing}")
else:
    print("  [skip] query FASTA exists")
    # Count written/missing
    written = sum(1 for line in open(QUERY_FASTA) if line.startswith('>') and 'MISSING' not in line)
    missing = sum(1 for line in open(QUERY_FASTA) if 'MISSING' in line)
    print(f"  Written: {written}  Missing: {missing}")

# Run BLAST
if not os.path.exists(BLAST_OUT):
    print("  Running BLAST (this may take 20-60 minutes)...")
    # Filter out MISSING entries for BLAST
    QUERY_VALID = f'{OUT_DIR}/brain_test_proteins_valid.fasta'
    if not os.path.exists(QUERY_VALID):
        subprocess.run(
            f"grep -A 1 -v MISSING {QUERY_FASTA} | grep -v '^--$' > {QUERY_VALID}",
            shell=True
        )
    t_blast = time.time()
    subprocess.run([
        BLAST_BIN,
        '-query', QUERY_VALID,
        '-db', BLAST_DB,
        '-out', BLAST_OUT,
        '-outfmt', '6 qseqid sseqid pident length evalue bitscore',
        '-evalue', '1e-4',
        '-max_target_seqs', '5',
        '-num_threads', '16',
        '-seg', 'yes'
    ], check=True)
    print(f"  BLAST done in {(time.time()-t_blast)/60:.1f} min")
else:
    print("  [skip] BLAST results exist")

# Parse GOA to get Swiss-Prot ID → GO MF terms
print("  Parsing Swiss-Prot GOA for MF terms...")
sprot_go_mf = defaultdict(set)
mf_go_set   = set(mf_valid)

with gzip.open(GOA_FILE, 'rt') as f:
    for line in f:
        if line.startswith('!'): continue
        p = line.strip().split('\t')
        if len(p) < 9: continue
        # Columns: DB, DB_ID, DB_Symbol, Qualifier, GO_ID, ..., Aspect
        db, db_id, aspect, go_id = p[0], p[1], p[8], p[4]
        if aspect != 'F': continue  # MF only
        if 'NOT' in p[3]: continue  # skip NOT qualifiers
        if go_id in mf_go_set:
            # Swiss-Prot accession (P12345 form)
            sprot_go_mf[db_id].add(go_id)

print(f"  Swiss-Prot entries with MF GO terms: {len(sprot_go_mf)}")

# Parse BLAST results and transfer GO terms
print("  Transferring GO terms from BLAST hits...")
# blast_out format: qseqid sseqid pident length evalue bitscore
# qseqid = {index}|{enst_id}
# sseqid = sp|P12345|GENE_HUMAN or tr|... format

best_hit = {}  # index → (sseqid, evalue, bitscore)
if os.path.exists(BLAST_OUT):
    with open(BLAST_OUT) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 6: continue
            qid, sid, evalue = p[0], p[1], float(p[4])
            idx = int(qid.split('|')[0])
            if idx not in best_hit or evalue < best_hit[idx][1]:
                best_hit[idx] = (sid, evalue, float(p[5]))

    # Extract Swiss-Prot accession from sseqid
    def extract_accession(sid):
        # sp|P12345|GENE_HUMAN → P12345
        parts = sid.split('|')
        if len(parts) >= 2:
            return parts[1]
        return sid

    n_hit_with_go = 0
    blast_scores = np.zeros((len(te_sym_list), len(mf_valid)), dtype=np.float32)

    for idx in range(len(te_sym_list)):
        if idx not in best_hit:
            continue
        sid, evalue, bscore = best_hit[idx]
        acc = extract_accession(sid)
        go_terms = sprot_go_mf.get(acc, set())
        if go_terms:
            n_hit_with_go += 1
        for j, go in enumerate(mf_valid):
            if go in go_terms:
                blast_scores[idx, j] = 1.0 - evalue  # higher for better hits

    n_hits = len(best_hit)
    print(f"  BLAST hits: {n_hits}/{len(te_sym_list)} ({n_hits/len(te_sym_list):.1%})")
    print(f"  Hits with relevant MF GO: {n_hit_with_go} ({n_hit_with_go/max(n_hits,1):.1%})")

    auprc_blast_all, auprc_blast_l2 = eval_auprc(blast_scores, Y_te_v, l2_idx)
    print(f"  E1 BLAST: All MF={auprc_blast_all:.4f}  L2_Structural={auprc_blast_l2:.4f}")
else:
    print("  BLAST output not found — skipping E1 evaluation")
    auprc_blast_all, auprc_blast_l2 = float('nan'), float('nan')

# ─────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────
results = {
    'E0_kNN_L30':    {'All MF': auprc_knn_all, 'L2_Structural': auprc_knn_l2},
    'E0b_GeneMean':  {'All MF': auprc_gm_all,  'L2_Structural': auprc_gm_l2},
    'E1_BLAST_Sprot':{'All MF': auprc_blast_all, 'L2_Structural': auprc_blast_l2},
    'Domain_LR':     DOMAIN_LR,
    'PRISM':         PRISM_REF,
    'v17f':          V17F_REF,
}

print("\n" + "=" * 65)
print("  SOTA Comparison — Brain MF AUPRC (82 terms)")
print("=" * 65)
print(f"  {'Method':<22} {'All MF':>10} {'L2_Struct':>12}")
print(f"  {'-'*22} {'-'*10} {'-'*12}")
for name, vals in results.items():
    all_mf = vals.get('All MF', float('nan'))
    l2     = vals.get('L2_Structural', float('nan'))
    l2_s   = f'{l2:.4f}' if l2 and not np.isnan(l2) else 'N/A'
    print(f"  {name:<22} {all_mf:>10.4f} {l2_s:>12}")

print(f"\n  Reference gains vs. best SOTA baseline (E0 k-NN):")
if not np.isnan(auprc_knn_all):
    print(f"    PRISM vs kNN:  {PRISM_REF['All MF']:.4f} vs {auprc_knn_all:.4f}  "
          f"(Δ={PRISM_REF['All MF']-auprc_knn_all:+.4f})")
    print(f"    v17f  vs kNN:  {V17F_REF['All MF']:.4f} vs {auprc_knn_all:.4f}  "
          f"(Δ={V17F_REF['All MF']-auprc_knn_all:+.4f})")

with open(f'{OUT_DIR}/sota_results.json', 'w') as f:
    # Convert any nan to null for JSON
    def clean_val(v):
        if isinstance(v, float) and np.isnan(v): return None
        return v
    json.dump({k: {kk: clean_val(vv) for kk, vv in val.items()}
               for k, val in results.items()}, f, indent=2)

print(f"\n  Results saved: {OUT_DIR}/sota_results.json")
