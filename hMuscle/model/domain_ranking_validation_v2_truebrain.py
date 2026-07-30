#!/usr/bin/env python3
"""
domain_ranking_validation_v2_truebrain.py
--------------------------------------------
TISSUE-MISLABELING BUGFIX RERUN (2026-07-14).
Original domain_ranking_validation.py loaded my_gene_list_fixed.npy / my_isoform_list_fixed.npy
/ domain_matrix_proper_test.npy / esm2_layer_30_t30_150M.npy as its "brain test set" -- these
are MUSCLE data (BambuTx IDs, 36748 isoforms), despite being described as validating against
the "brain test set". This rerun re-points the TEST side only at the TRUE brain isoform set
(brain_full_gene_names.npy / brain_full_ids.npy / domain_matrix_brain_full.npy (built in Step 2
of this rerun, same 512-Pfam vocab as domain_matrix_proper_test.npy) / brain_full_esm2_layer30),
and PRISM_PREDS at the TRUE-brain v17f* predictions from v17f_star_bootstrap_ci_v2_truebrain.py.
BLAST_TSV is re-pointed at the true-brain diamond-blastp hits produced by a new
run_blast_goa_truebrain.py (mirroring reports/benchmark_external/blast_goa/run_blast_goa.py,
which is the ACTUAL authoritative BLAST->GOA pipeline this script depends on -- NOT
exp_e_sota_comparison.py's own separate/unused blastp attempt, which produced an empty
blast_results.tsv on the muscle side). Query = TRUE brain complete protein FASTA
(53826/63994 coding isoforms), same diamond DB (cached, tissue-agnostic Swiss-Prot).
Training side (train_gene_list.npy, esm2_train_human_layer30) is UNCHANGED. GOA_FILE (cached,
tissue-agnostic Swiss-Prot GOA) is reused as-is per task instructions.

Original (mislabeled-as-brain, actually muscle): domain_ranking_validation.py, backed up
at domain_ranking_validation_backup_20260714.py before this rerun.

domain_ranking_validation.py
==============================
Per-locus domain ranking validation.

목적: Gene-mean oracle 비판에 대한 핵심 반증.
  - 논리: gene-mean oracle은 같은 유전자의 모든 아이소폼에 동일한 점수를 줌 → AUC=0.5
  - PRISM이 도메인 완전한 아이소폼을 도메인 절단된 것보다 높이 평가한다면
    → within-gene domain-ranking AUC > 0.5 → gene classifier가 아님을 증명

분석 설계:
  - TRUE 뇌 테스트 세트 (63,994 아이소폼 / 53,826 protein-coding)의 hmmscan Pfam 도메인
    개수를 ground truth로 사용
  - 같은 유전자 내에서 도메인 수가 다른 아이소폼 쌍을 식별
  - 각 방법이 도메인 많은 아이소폼에 더 높은 점수를 주는지 AUC로 평가
  - 비교: PRISM v17f*, BLAST→GOA, k-NN, Gene-mean oracle, Random

출력: reports/truebrain_rerun_20260714/domain_ranking_validation/results.json
"""

import os, sys, re, json, gzip
import numpy as np
from collections import defaultdict
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = '../../reports/truebrain_rerun_20260714/domain_ranking_validation'
os.makedirs(OUT_DIR, exist_ok=True)

DATA_DIR  = '../data'
BRAIN_DIR = '../data/brain_isoquant_esm2/full'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'

DOMAIN_MAT   = '../results_isoform/features/domain_matrix_brain_full.npy'  # 63994×512 binary Pfam, TRUE BRAIN
PRISM_PREDS  = '../../reports/truebrain_rerun_20260714/v17f_star_bootstrap/v17f_star_preds.npy'
BLAST_TSV    = '../../reports/truebrain_rerun_20260714/blast_goa_truebrain/blast_hits.tsv'
GOA_FILE     = '../../reports/exp_e_sota/swissprot_db/goa_human.gaf.gz'  # cached, tissue-agnostic, reused as-is
PEP_FILE     = '../../reports/truebrain_rerun_20260714/data/brain_full_proteins.fa'

print("=" * 70)
print("  Per-Locus Domain Ranking Validation")
print("  (Against Gene-Mean Oracle: AUC=0.5 null)")
print("=" * 70)

# ── 1. Brain isoform identities ───────────────────────────────────────
print("\n[1] Loading brain test set identities...")

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

# TRUE BRAIN: gene names already symbols (e.g. 'A1BG'), no ENSG2SYM mapping needed.
te_genes_raw = np.load(f'{BRAIN_DIR}/brain_full_gene_names.npy', allow_pickle=True)
te_sym_list  = [clean(g) for g in te_genes_raw]
te_isoforms  = np.load(f'{BRAIN_DIR}/brain_full_ids.npy', allow_pickle=True)
te_iso_list  = [clean(x) for x in te_isoforms]
n_iso        = len(te_iso_list)

iso2idx   = {iso: i for i, iso in enumerate(te_iso_list)}
gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs[g].append(i)

print(f"  {n_iso} isoforms, {len(gene2idxs)} genes")
print(f"  Multi-isoform genes (≥2): {sum(1 for g, v in gene2idxs.items() if len(v)>=2)}")

# ── 2. Pfam domain count per isoform (muscle test set) ───────────────
print("\n[2] Loading Pfam domain counts from domain_matrix_proper_test.npy...")
# 36748 × 512 binary matrix (row = isoform, col = Pfam family)
# Row sum = number of distinct Pfam domain families detected
domain_mat   = np.load(DOMAIN_MAT)
iso_n_domains = domain_mat.sum(axis=1).astype(np.int32)
covered = (iso_n_domains > 0).sum()
print(f"  Matrix: {domain_mat.shape}")
print(f"  With ≥1 Pfam domain: {covered}/{n_iso} ({100*covered/n_iso:.1f}%)")
print(f"  Domain count dist: 0={(iso_n_domains==0).sum()} 1={(iso_n_domains==1).sum()} "
      f"2={(iso_n_domains==2).sum()} 3+={(iso_n_domains>=3).sum()}")

# ── 3. Load MF GO labels ──────────────────────────────────────────────
print("\n[3] Loading MF GO labels...")
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

go_all_mf = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        gid, go_id, cat_raw = p[1], p[2], p[7]
        if cat_raw != 'Function': continue
        go_all_mf[go_id].add(gid)

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])

Y_te = np.stack([
    np.array([1.0 if sym2id.get(s,'__') in go_all_mf[go_id] else 0.0
              for s in te_sym_list], dtype=np.float32)
    for go_id in mf_terms
], axis=1)

valid_mask = Y_te.sum(0) >= 2
mf_valid = [go for go, v in zip(mf_terms, valid_mask) if v]
Y_te_v   = Y_te[:, valid_mask]
print(f"  {len(mf_valid)} valid MF terms")

# ── 4. Load prediction matrices ───────────────────────────────────────
print("\n[4] Loading prediction matrices...")

prism_raw   = np.load(PRISM_PREDS).astype(np.float32)
prism_preds = prism_raw[:, valid_mask] if prism_raw.shape[1] == 82 else prism_raw
print(f"  PRISM v17f*: {prism_preds.shape}")

# Gene-mean oracle: each isoform gets its gene's mean v17f* prediction
# → AUC must be 0.5 by construction (all isoforms of a gene get identical score)
gene_mean_preds = np.zeros_like(prism_preds)
for g, idxs in gene2idxs.items():
    gene_mean = prism_preds[idxs].mean(0)
    for i in idxs: gene_mean_preds[i] = gene_mean
print(f"  Gene-mean oracle: computed (collapses within-gene variation)")

# BLAST→GOA
print("  Building BLAST score matrix...")
uniprot2go_mf = defaultdict(set)
with gzip.open(GOA_FILE, 'rt') as fh:
    for line in fh:
        if line.startswith('!'): continue
        parts = line.strip().split('\t')
        if len(parts) < 9: continue
        uid, go_id, aspect, qual = parts[1], parts[4], parts[8], parts[3]
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
        if float(parts[3]) > 1e-3: continue
        sp_parts = parts[1].split('|')
        uid = sp_parts[1] if len(sp_parts) >= 2 else parts[1]
        for go_id in uniprot2go_mf.get(uid, []):
            if bitscore > query_go_score[qid][go_id]:
                query_go_score[qid][go_id] = bitscore

blast_preds = np.zeros((n_iso, len(mf_valid)), dtype=np.float32)
for i, iso_id in enumerate(te_iso_list):
    for j, go_id in enumerate(mf_valid):
        blast_preds[i, j] = query_go_score.get(iso_id, {}).get(go_id, 0.0)
print(f"  BLAST→GOA: {blast_preds.shape}")

# k-NN
print("  Building k-NN predictions...")
X30_tr  = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X30_te  = np.load(f'{BRAIN_DIR}/brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes = [clean(g) for g in tr_genes_raw]
tr_id_set = set(sym2id.get(g, g) for g in tr_genes)

go_tr = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0]!='9606': continue
        gid, go_id, cat_raw = p[1], p[2], p[7]
        if cat_raw!='Function': continue
        if gid in tr_id_set: go_tr[go_id].add(gid)

tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)
Y_tr_v = np.stack([
    np.array([1.0 if sym2id.get(tr_genes[i], tr_genes[i]) in go_tr[go_id] else 0.0
              for i in range(len(tr_genes))], dtype=np.float32)
    for go_id in mf_valid
], axis=1)

from sklearn.preprocessing import normalize
Xtr_n = normalize(X30_tr); Xte_n = normalize(X30_te)
CHUNK = 4096; K = 5
knn_preds = np.zeros((n_iso, len(mf_valid)), dtype=np.float32)
for start in range(0, n_iso, CHUNK):
    end = min(start + CHUNK, n_iso)
    sims = Xte_n[start:end] @ Xtr_n.T
    topk_idx = np.argpartition(-sims, K, axis=1)[:, :K]
    for il in range(end - start):
        knn_preds[start + il] = Y_tr_v[topk_idx[il]].mean(0)
print(f"  k-NN: {knn_preds.shape}")

rng = np.random.default_rng(42)
random_preds = rng.random((n_iso, len(mf_valid))).astype(np.float32)

methods = {
    'PRISM_v17f*':      prism_preds,
    'Gene_mean_oracle': gene_mean_preds,
    'BLAST_GOA':        blast_preds,
    'kNN_ESM2':         knn_preds,
    'Random':           random_preds,
}

# ── 5. CORE ANALYSIS: Within-gene domain ranking AUC ──────────────────
print("\n[5] CORE: Within-gene domain ranking AUC")
print("  Ground truth: within each gene, isoforms with MORE Pfam domains")
print("  should be ranked HIGHER (= more functionally complete)")
print("  Gene-mean oracle MUST score AUC=0.5 (all isoforms get identical scores)")

from sklearn.metrics import roc_auc_score

def compute_domain_ranking_auc(preds_mat, gene2idxs, iso_n_domains, Y_te_v, n_min_pairs=2):
    """
    For each multi-isoform gene where isoforms differ in domain count:
    - Binary label: has_more_domains (domain_count > median domain_count for that gene)
    - Score: predicted functional score for each positive GO term
    - AUC per (gene, GO_term) → aggregate
    Returns: (mean_AUC, n_pairs_evaluated, per_gene_results)
    """
    aucs = []
    gene_results = []

    for g, idxs in gene2idxs.items():
        if len(idxs) < 2: continue
        domains = iso_n_domains[idxs]
        if domains.std() < 0.1: continue  # all isoforms same domain count → skip

        gene_pos_terms = np.where(Y_te_v[idxs[0]] > 0)[0]
        if len(gene_pos_terms) == 0: continue

        # Binary: 1 if above-median domain count in this gene
        med = np.median(domains)
        domain_binary = (domains > med).astype(float)
        if domain_binary.sum() == 0 or domain_binary.sum() == len(idxs): continue

        p_g = preds_mat[idxs]  # (k, n_terms)
        gene_aucs = []
        for t in gene_pos_terms:
            scores = p_g[:, t]
            if scores.std() < 1e-8:
                gene_aucs.append(0.5)  # tied → random
                aucs.append(0.5)
                continue
            try:
                auc = roc_auc_score(domain_binary, scores)
                gene_aucs.append(auc)
                aucs.append(auc)
            except:
                pass

        if gene_aucs:
            gene_results.append({
                'gene': g,
                'n_isoforms': len(idxs),
                'domain_range': f"{domains.min()}-{domains.max()}",
                'mean_auc': float(np.mean(gene_aucs)),
                'n_terms': len(gene_aucs),
            })

    return np.mean(aucs) if aucs else 0.5, len(aucs), gene_results

print()
results_by_method = {}
for name, preds in methods.items():
    auc, n, gene_res = compute_domain_ranking_auc(preds, gene2idxs, iso_n_domains, Y_te_v)
    results_by_method[name] = {'domain_ranking_auc': auc, 'n_pairs': n, 'gene_results': gene_res}
    sig = "*** vs gene-mean" if (name == 'PRISM_v17f*' and auc > results_by_method.get('Gene_mean_oracle', {}).get('domain_ranking_auc', 0.5)) else ""
    print(f"  {name:<22}: Domain-Ranking AUC = {auc:.4f}  (N={n:,} term-pairs) {sig}")

# ── 6. Bootstrap CI for PRISM and Gene-mean ───────────────────────────
print("\n[6] Bootstrap CI (B=500, gene-level block)")

def bootstrap_domain_ranking(preds_mat, gene2idxs, iso_n_domains, Y_te_v, n_boot=500, seed=42):
    rng_b = np.random.default_rng(seed)
    gene_list = [g for g, idxs in gene2idxs.items()
                 if len(idxs) >= 2
                 and iso_n_domains[idxs].std() >= 0.1
                 and Y_te_v[idxs[0]].sum() > 0]
    boots = []
    for _ in range(n_boot):
        sampled = rng_b.choice(gene_list, size=len(gene_list), replace=True)
        auc, n, _ = compute_domain_ranking_auc(
            preds_mat,
            {g: gene2idxs[g] for g in sampled},
            iso_n_domains, Y_te_v
        )
        boots.append(auc)
    boots = np.array(boots)
    return float(boots.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

for name in ['PRISM_v17f*', 'Gene_mean_oracle', 'BLAST_GOA']:
    preds = methods[name]
    mean, lo, hi = bootstrap_domain_ranking(preds, gene2idxs, iso_n_domains, Y_te_v, n_boot=500)
    print(f"  {name:<22}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")
    results_by_method[name]['bootstrap_auc']   = mean
    results_by_method[name]['bootstrap_ci_lo'] = lo
    results_by_method[name]['bootstrap_ci_hi'] = hi

# ── 7. Top genes: domain differentiation ─────────────────────────────
print("\n[7] Genes with strongest PRISM domain discrimination...")
prism_gene_res = results_by_method['PRISM_v17f*']['gene_results']
top_genes = sorted(prism_gene_res, key=lambda x: x['mean_auc'], reverse=True)[:10]
bottom_genes = sorted(prism_gene_res, key=lambda x: x['mean_auc'])[:5]

print("  Top-10 genes (PRISM correctly ranks domain-complete isoform):")
for g in top_genes:
    print(f"    {g['gene']:<12} AUC={g['mean_auc']:.3f}  isoforms={g['n_isoforms']}  "
          f"domain_range={g['domain_range']}  terms={g['n_terms']}")

print("  Bottom-5 genes (PRISM ranks domain-truncated isoform higher):")
for g in bottom_genes:
    print(f"    {g['gene']:<12} AUC={g['mean_auc']:.3f}  isoforms={g['n_isoforms']}  "
          f"domain_range={g['domain_range']}  terms={g['n_terms']}")

# ── 8. Specific BISECT case check ─────────────────────────────────────
print("\n[8] BISECT-validated Complex I genes (domain intact > truncated?)...")
bisect_check = [
    ('NDUFS4', 'oxidoreductase'),
    ('NDUFS7', 'oxidoreductase'),
    ('NDUFS8', 'iron-sulfur'),
    ('DLG1',   'synaptic transmission'),
    ('KIF21B', 'microtubule'),
    ('ERCC6L2','DNA repair'),
]
for gene_sym, go_hint in bisect_check:
    if gene_sym not in gene2idxs: continue
    idxs = gene2idxs[gene_sym]
    domains = iso_n_domains[idxs]
    scores  = prism_preds[idxs]  # (k, n_terms)
    isos_g  = [te_iso_list[i] for i in idxs]
    print(f"  {gene_sym}: {len(idxs)} isoforms | domain counts: {domains.tolist()}")
    for ii, (iso, d, sc) in enumerate(zip(isos_g, domains, scores.mean(1))):
        print(f"    [{ii}] {iso:<35} domains={d}  mean_score={sc:.4f}")

# ── 9. Save ───────────────────────────────────────────────────────────
print("\n[9] Saving results...")
summary = {
    'analysis': 'per_locus_domain_ranking_auc',
    'n_brain_isoforms': n_iso,
    'n_multi_isoform_genes': sum(1 for g, v in gene2idxs.items() if len(v)>=2),
    'n_genes_with_domain_variation': sum(
        1 for g, idxs in gene2idxs.items()
        if len(idxs)>=2 and iso_n_domains[idxs].std() >= 0.1
    ),
    'interpro_coverage_pct': float(100 * covered / n_iso),
    'results': {
        name: {k: v for k, v in r.items() if k != 'gene_results'}
        for name, r in results_by_method.items()
    },
    'top_genes': top_genes,
    'bottom_genes': bottom_genes,
}
out_path = f'{OUT_DIR}/results.json'
with open(out_path, 'w') as fh:
    json.dump(summary, fh, indent=2)
print(f"  Saved → {out_path}")

# ── 10. Final interpretation ──────────────────────────────────────────
prism_auc  = results_by_method['PRISM_v17f*']['domain_ranking_auc']
oracle_auc = results_by_method['Gene_mean_oracle']['domain_ranking_auc']
blast_auc  = results_by_method['BLAST_GOA']['domain_ranking_auc']

print("\n" + "="*70)
print("  SUMMARY: Per-Locus Domain Ranking AUC")
print("="*70)
print(f"  PRISM v17f*:       {prism_auc:.4f}")
print(f"  Gene-mean oracle:  {oracle_auc:.4f}  ← must be 0.5 (identical within-gene scores)")
print(f"  BLAST→GOA:         {blast_auc:.4f}")
print(f"  Gap (PRISM-Oracle): {prism_auc-oracle_auc:+.4f}")
if prism_auc > oracle_auc:
    print("  ✓ PRISM EXCEEDS gene-mean oracle on within-gene domain ranking")
    print("  → PRISM is not a pure gene-level classifier")
else:
    print("  ✗ PRISM does not exceed gene-mean oracle on within-gene domain ranking")
print("="*70)
