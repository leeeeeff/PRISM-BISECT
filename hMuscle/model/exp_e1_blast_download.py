#!/usr/bin/env python3
"""
exp_e1_blast_download.py
========================
Step 1: Download Swiss-Prot FASTA + build BLAST DB + get GO annotations.
Run BEFORE exp_e1_blast_eval.py.

GOA source: UniProt REST API (Swiss-Prot MF GO terms as TSV).
This avoids the large goa_uniprot_all.gaf.gz download.
"""

import os, sys, subprocess, urllib.request, json, time, gzip
import numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/exp_e_sota'
SPROT_DIR = f'{OUT_DIR}/swissprot_db'
os.makedirs(SPROT_DIR, exist_ok=True)

SPROT_FASTA = f'{SPROT_DIR}/uniprot_sprot.fasta.gz'
SPROT_UNZIP = f'{SPROT_DIR}/uniprot_sprot.fasta'
GO_TSV      = f'{SPROT_DIR}/swissprot_go_mf.tsv'
BLAST_DB    = f'{SPROT_DIR}/swissprot'
BLAST_BIN   = '/opt/blast/ncbi-blast-2.12.0+/bin/blastp'
QUERY_FASTA = f'{OUT_DIR}/brain_test_proteins.fasta'
QUERY_VALID = f'{OUT_DIR}/brain_test_proteins_valid.fasta'
BLAST_OUT   = f'{OUT_DIR}/blast_results.tsv'

# ─────────────────────────────────────────────────────────────────
# 1. Download Swiss-Prot FASTA
# ─────────────────────────────────────────────────────────────────
if not os.path.exists(SPROT_UNZIP) and not os.path.exists(SPROT_FASTA):
    url = 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz'
    print(f"[1] Downloading Swiss-Prot FASTA (~270MB)...", flush=True)
    t0 = time.time()
    urllib.request.urlretrieve(url, SPROT_FASTA)
    print(f"    Done: {os.path.getsize(SPROT_FASTA)//1024//1024} MB in {time.time()-t0:.0f}s")
elif os.path.exists(SPROT_UNZIP):
    print(f"[1] Swiss-Prot FASTA already decompressed")
else:
    print(f"[1] Swiss-Prot FASTA already downloaded: {os.path.getsize(SPROT_FASTA)//1024//1024} MB")

# Decompress if needed
if not os.path.exists(SPROT_UNZIP):
    print("    Decompressing...", flush=True)
    subprocess.run(f'gunzip -k {SPROT_FASTA}', shell=True, check=True)

# ─────────────────────────────────────────────────────────────────
# 2. Download Swiss-Prot GO MF annotations via UniProt REST API
# ─────────────────────────────────────────────────────────────────
if not os.path.exists(GO_TSV):
    print("\n[2] Downloading Swiss-Prot GO MF annotations via UniProt REST API...")
    print("    (fields: accession + go_f — this may take 2-5 min)")
    url = ('https://rest.uniprot.org/uniprotkb/stream?'
           'format=tsv&query=reviewed:true&fields=accession,go_f')
    t0 = time.time()

    import http.client
    from urllib.parse import urlparse
    parsed = urlparse(url)
    conn   = http.client.HTTPSConnection(parsed.netloc)
    conn.request('GET', parsed.path + '?' + parsed.query,
                 headers={'User-Agent': 'PRISM-research/1.0'})
    resp = conn.getresponse()
    print(f"    HTTP {resp.status}")
    if resp.status != 200:
        print(f"    ERROR: {resp.status} {resp.reason}")
        sys.exit(1)

    with open(GO_TSV, 'wb') as f:
        total = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if total % (10 * 1024 * 1024) == 0:
                print(f"    Downloaded {total//1024//1024} MB...", flush=True)
    print(f"    Done: {os.path.getsize(GO_TSV)//1024//1024} MB in {time.time()-t0:.0f}s")
else:
    print(f"[2] GO MF TSV already exists: {os.path.getsize(GO_TSV)//1024//1024} MB")

# Parse GO TSV
print("\n[3] Parsing GO MF TSV...")
# Format: accession\tGene Ontology (molecular function)
# GO column: "GO:0003674 [molecular_function]; GO:0004012 [ATP-dependent chromatin remodeling]; ..."
mf_terms_set = set()

# Load 82 MF terms
ANNOT_TSV = '../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv'
with open(ANNOT_TSV) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 1:
            mf_terms_set.add(p[0])

sprot_go_mf = {}  # accession → set of GO MF IDs
n_parsed = 0
with open(GO_TSV, encoding='utf-8', errors='replace') as f:
    next(f)  # skip header
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) < 2:
            continue
        acc, go_col = p[0], p[1] if len(p) > 1 else ''
        go_terms = set()
        for entry in go_col.split(';'):
            entry = entry.strip()
            if entry.startswith('GO:'):
                go_id = entry.split(' ')[0]
                if go_id in mf_terms_set:
                    go_terms.add(go_id)
        if go_terms:
            sprot_go_mf[acc] = go_terms
        n_parsed += 1

print(f"    Parsed {n_parsed} Swiss-Prot entries")
print(f"    Entries with relevant MF GO terms: {len(sprot_go_mf)}")

# Save parsed dict for quick reuse
with open(f'{SPROT_DIR}/sprot_go_mf.json', 'w') as f:
    json.dump({k: list(v) for k, v in sprot_go_mf.items()}, f)
print(f"    Saved: {SPROT_DIR}/sprot_go_mf.json")

# ─────────────────────────────────────────────────────────────────
# 3. Build BLAST database
# ─────────────────────────────────────────────────────────────────
if not os.path.exists(f'{BLAST_DB}.phr'):
    print("\n[4] Building BLAST database...")
    t0 = time.time()
    subprocess.run([
        '/opt/blast/ncbi-blast-2.12.0+/bin/makeblastdb',
        '-in', SPROT_UNZIP, '-dbtype', 'prot', '-out', BLAST_DB
    ], check=True)
    print(f"    Done in {time.time()-t0:.0f}s")
else:
    print("\n[4] BLAST database already exists")

# ─────────────────────────────────────────────────────────────────
# 4. Extract brain test protein sequences in embedding order
# ─────────────────────────────────────────────────────────────────
def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

if not os.path.exists(QUERY_VALID):
    print("\n[5] Extracting brain test protein sequences...")
    PEP_FILE = f'{DATA_DIR}/brain_esm2/brain_only_transcripts.fa.transdecoder.pep'
    ENSG2SYM = {}
    with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 5:
                ENSG2SYM[p[0]] = p[4]
    te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
    te_enst_list = [clean(g) for g in te_genes_raw]

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
    print(f"    Written: {written}  Missing: {missing}")
else:
    written = sum(1 for line in open(QUERY_VALID) if line.startswith('>'))
    print(f"\n[5] Query FASTA already exists ({written} sequences)")

# ─────────────────────────────────────────────────────────────────
# 5. Run BLAST
# ─────────────────────────────────────────────────────────────────
if not os.path.exists(BLAST_OUT):
    print("\n[6] Running BLAST (blastp, ~20-60 min)...", flush=True)
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
    print(f"    BLAST done in {t_blast/60:.1f} min  |  {n_hits} hits")
else:
    n_hits = sum(1 for _ in open(BLAST_OUT))
    print(f"\n[6] BLAST results already exist ({n_hits} hits)")

print("\n[Done] Ready for evaluation — run exp_e1_blast_eval.py")
