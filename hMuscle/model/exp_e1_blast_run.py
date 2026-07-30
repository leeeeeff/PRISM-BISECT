#!/usr/bin/env python3
"""
exp_e1_blast_run.py
===================
Run BLAST + Swiss-Prot GO transfer for Exp E1.
Pre-reqs: uniprot_sprot.fasta.gz + goa_human.gaf.gz already downloaded.
"""

import os, sys, subprocess, json, time, gzip
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/exp_e_sota'
SPROT_DIR = f'{OUT_DIR}/swissprot_db'

SPROT_FASTA = f'{SPROT_DIR}/uniprot_sprot.fasta.gz'
SPROT_UNZIP = f'{SPROT_DIR}/uniprot_sprot.fasta'
GOA_FILE    = f'{SPROT_DIR}/goa_human.gaf.gz'
BLAST_DB    = f'{SPROT_DIR}/swissprot'
BLAST_BIN   = '/opt/blast/ncbi-blast-2.12.0+/bin/blastp'
QUERY_FASTA = f'{OUT_DIR}/brain_test_proteins.fasta'
QUERY_VALID = f'{OUT_DIR}/brain_test_proteins_valid.fasta'
BLAST_OUT   = f'{OUT_DIR}/blast_results.tsv'

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

# ─────────────────────────────────────────────────────────────────
# 1. Load MF term list (82 terms)
# ─────────────────────────────────────────────────────────────────
mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 1: mf_terms.append(p[0])

mf_terms_set = set(mf_terms)
print(f"MF terms: {len(mf_terms)}")

# ─────────────────────────────────────────────────────────────────
# 2. Load brain test isoform IDs + GO labels
# ─────────────────────────────────────────────────────────────────
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]
te_enst_list = [clean(g) for g in te_genes_raw]
n_test = len(te_sym_list)
print(f"Test isoforms: {n_test}")

# Build GO labels
sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]

go_genes_all = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        if p[7] != 'Function': continue
        if p[2] in mf_terms_set:
            go_genes_all[p[2]].add(p[1])

def build_Y_te(go_id):
    pos_ids = go_genes_all[go_id]
    return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0
                     for s in te_sym_list], dtype=np.float32)

Y_te = np.stack([build_Y_te(go) for go in mf_terms], axis=1)
valid_mask = Y_te.sum(0) >= 2
mf_valid   = [go for go, v in zip(mf_terms, valid_mask) if v]
Y_te_v     = Y_te[:, valid_mask]
n_valid    = valid_mask.sum()
print(f"Valid MF terms (≥2 positives): {n_valid}")

# H2 L2_Structural index
L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural':
            L2_TERMS.add(p[0])
l2_idx = [i for i, go in enumerate(mf_valid) if go in L2_TERMS]

def eval_auprc(scores, labels, l2_idx):
    auprcs = [average_precision_score(labels[:, j], scores[:, j])
              for j in range(labels.shape[1]) if labels[:, j].sum() >= 2]
    all_mf = float(np.mean(auprcs))
    l2_ap  = [average_precision_score(labels[:, j], scores[:, j])
              for j in l2_idx if labels[:, j].sum() >= 2]
    l2_mf  = float(np.mean(l2_ap)) if l2_ap else float('nan')
    return all_mf, l2_mf

# ─────────────────────────────────────────────────────────────────
# 3. Parse human GOA for MF terms
# ─────────────────────────────────────────────────────────────────
print("\nParsing human GOA (goa_human.gaf.gz)...")
sprot_go_mf = defaultdict(set)

with gzip.open(GOA_FILE, 'rt') as f:
    for line in f:
        if line.startswith('!'): continue
        p = line.strip().split('\t')
        if len(p) < 9: continue
        # GAF format: db, db_obj_id, db_obj_symbol, qualifier, go_id, ..., aspect
        db_id  = p[1]   # UniProt accession
        qual   = p[3]   # qualifier (NOT, etc.)
        go_id  = p[4]
        aspect = p[8]   # F=MF, P=BP, C=CC
        if aspect != 'F': continue
        if 'NOT' in qual: continue
        if go_id in mf_terms_set:
            sprot_go_mf[db_id].add(go_id)

print(f"  Human UniProt entries with relevant MF GO: {len(sprot_go_mf)}")

# ─────────────────────────────────────────────────────────────────
# 4. Decompress Swiss-Prot FASTA + Build BLAST DB
# ─────────────────────────────────────────────────────────────────
if not os.path.exists(SPROT_UNZIP):
    print("\nDecompressing Swiss-Prot FASTA...")
    subprocess.run(f'gunzip -k {SPROT_FASTA}', shell=True, check=True)
    print(f"  Decompressed: {os.path.getsize(SPROT_UNZIP)//1024//1024} MB")

if not os.path.exists(f'{BLAST_DB}.phr'):
    print("\nBuilding BLAST database...")
    t0 = time.time()
    subprocess.run([
        '/opt/blast/ncbi-blast-2.12.0+/bin/makeblastdb',
        '-in', SPROT_UNZIP, '-dbtype', 'prot', '-out', BLAST_DB
    ], check=True)
    print(f"  Done in {time.time()-t0:.0f}s")
else:
    print("\nBLAST database already exists")

# ─────────────────────────────────────────────────────────────────
# 5. Extract brain test sequences (in embedding order)
# ─────────────────────────────────────────────────────────────────
if not os.path.exists(QUERY_VALID):
    print("\nExtracting brain test protein sequences...")
    PEP_FILE = f'{DATA_DIR}/brain_esm2/brain_only_transcripts.fa.transdecoder.pep'
    pep_by_enst = {}
    current_id, current_seq = None, []
    with open(PEP_FILE) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if current_id:
                    pep_by_enst[current_id.split('.')[0]] = (current_id, ''.join(current_seq))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            pep_by_enst[current_id.split('.')[0]] = (current_id, ''.join(current_seq))

    written, missing = 0, 0
    with open(QUERY_FASTA, 'w') as fq, open(QUERY_VALID, 'w') as fv:
        for i, enst_raw in enumerate(te_enst_list):
            enst = enst_raw.split('.')[0]
            if enst in pep_by_enst:
                pid, seq = pep_by_enst[enst]
                fq.write(f'>{i}|{enst}\n{seq}\n')
                fv.write(f'>{i}|{enst}\n{seq}\n')
                written += 1
            else:
                fq.write(f'>{i}|{enst}|MISSING\n')
                missing += 1
    print(f"  Written: {written}  Missing: {missing}")
else:
    written = sum(1 for line in open(QUERY_VALID) if line.startswith('>'))
    print(f"\nQuery FASTA exists: {written} sequences")

# ─────────────────────────────────────────────────────────────────
# 6. Run BLAST
# ─────────────────────────────────────────────────────────────────
if not os.path.exists(BLAST_OUT):
    print("\nRunning BLAST (this takes 20-60 min)...", flush=True)
    t0 = time.time()
    subprocess.run([
        BLAST_BIN,
        '-query', QUERY_VALID, '-db', BLAST_DB,
        '-out', BLAST_OUT,
        '-outfmt', '6 qseqid sseqid pident length evalue bitscore',
        '-evalue', '1e-4',
        '-max_target_seqs', '5',
        '-num_threads', '16',
        '-seg', 'yes'
    ], check=True)
    t_blast = time.time() - t0
    n_hits = sum(1 for _ in open(BLAST_OUT))
    print(f"  BLAST done in {t_blast/60:.1f} min  |  {n_hits} hits")
else:
    n_hits = sum(1 for _ in open(BLAST_OUT))
    print(f"\nBLAST results exist: {n_hits} hits")

# ─────────────────────────────────────────────────────────────────
# 7. Transfer GO terms and evaluate
# ─────────────────────────────────────────────────────────────────
print("\nTransferring GO annotations from BLAST hits...")

# Parse best hit per query
best_hit = {}
with open(BLAST_OUT) as f:
    for line in f:
        p = line.strip().split('\t')
        if len(p) < 6: continue
        qid, sid, evalue = p[0], p[1], float(p[4])
        idx = int(qid.split('|')[0])
        if idx not in best_hit or evalue < best_hit[idx][1]:
            best_hit[idx] = (sid, evalue, float(p[5]))

def extract_acc(sid):
    # sp|P12345|GENE_HUMAN → P12345
    parts = sid.split('|')
    return parts[1] if len(parts) >= 2 else sid

# Build score matrix
blast_scores = np.zeros((n_test, n_valid), dtype=np.float32)
n_with_go = 0

for idx in range(n_test):
    if idx not in best_hit: continue
    sid, evalue, bscore = best_hit[idx]
    acc = extract_acc(sid)
    go_set = sprot_go_mf.get(acc, set())
    if go_set: n_with_go += 1
    # Score: 1-evalue for hits (higher = better); 0 for no hit/no GO
    for j, go in enumerate(mf_valid):
        if go in go_set:
            blast_scores[idx, j] = bscore / 1000.0  # normalise bitscore

n_hits_total = len(best_hit)
print(f"  Total hits: {n_hits_total}/{n_test} ({n_hits_total/n_test:.1%})")
print(f"  Hits with relevant MF GO: {n_with_go} ({n_with_go/max(n_hits_total,1):.1%})")

auprc_blast_all, auprc_blast_l2 = eval_auprc(blast_scores, Y_te_v, l2_idx)
print(f"\n  E1 BLAST+Swiss-Prot: All MF={auprc_blast_all:.4f}  L2_Structural={auprc_blast_l2:.4f}")

# ─────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────
results = {
    'E0_kNN_L30':     {'All MF': 0.5992, 'L2_Structural': 0.4979},
    'E0b_GeneMean':   {'All MF': 0.4651, 'L2_Structural': 0.3029},
    'E1_BLAST_Sprot': {'All MF': auprc_blast_all, 'L2_Structural': auprc_blast_l2,
                       'n_hits': int(n_hits_total), 'n_with_GO': int(n_with_go)},
    'Domain_LR':      {'All MF': 0.1625, 'L2_Structural': None},
    'PRISM':          {'All MF': 0.5962, 'L2_Structural': 0.3127},
    'v17f':           {'All MF': 0.7173, 'L2_Structural': 0.6219},
}

with open(f'{OUT_DIR}/sota_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("  SOTA Comparison Summary")
print("=" * 60)
print(f"  {'Method':<22} {'All MF':>10} {'L2_Struct':>12}")
print(f"  {'-'*22} {'-'*10} {'-'*12}")
for name, vals in results.items():
    a = vals.get('All MF', float('nan'))
    l = vals.get('L2_Structural') or float('nan')
    l_s = f'{l:.4f}' if not (isinstance(l, float) and np.isnan(l)) else 'N/A'
    print(f"  {name:<22} {a:>10.4f} {l_s:>12}")
print(f"\nResults saved: {OUT_DIR}/sota_results.json")
