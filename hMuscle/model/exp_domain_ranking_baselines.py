"""
MR1 defense: compute Domain-Ranking AUC for
  (1) sequence length baseline
  (2) raw ESM-2 L30 L2-norm
  (3) raw ESM-2 L30 centroid similarity (per-GO class centroid from training set)
  (4) PRISM D0 (frozen ESM-2, task-trained 82 MF, no delta)
vs PRISM v17f* (0.630) and Gene-mean oracle (0.500)

Paper-critic MR1: "What is Domain-Ranking AUC for raw ESM-2 and length baseline?"
"""

import os, sys, gzip, re, json
import numpy as np
from collections import defaultdict
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR     = '../data'
ID_DIR       = '../data/raw_data/data/id_lists'
ANNOT_DIR    = '../data/raw_data/data/annotations'
DOMAIN_MAT   = '../results_isoform/features/domain_matrix_proper_test.npy'
PRISM_PREDS  = '../../reports/v17f_star_bootstrap/v17f_star_preds.npy'
D0_PREDS     = '../../reports/exp_e_sota/d0_preds.npy'
PEP_FILE     = '../data/top30k_isoforms.pep'
OUT_DIR      = '../../reports/domain_ranking_validation'
os.makedirs(OUT_DIR, exist_ok=True)


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


print("="*65)
print("  Domain-Ranking AUC — Extended Baseline Comparison")
print("="*65)

# ── 1. Identities ─────────────────────────────────────────────────────
print("\n[1] Loading identities...")
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in te_genes_raw]
te_isoforms  = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
te_iso_list  = [clean(x) for x in te_isoforms]
n_iso        = len(te_iso_list)

iso2idx   = {iso: i for i, iso in enumerate(te_iso_list)}
gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs[g].append(i)
print(f"  {n_iso} isoforms, {len(gene2idxs)} genes")

# ── 2. Domain matrix ──────────────────────────────────────────────────
print("\n[2] Pfam domain counts...")
domain_mat    = np.load(DOMAIN_MAT)
iso_n_domains = domain_mat.sum(axis=1).astype(np.int32)

# ── 3. GO labels ──────────────────────────────────────────────────────
print("\n[3] GO labels...")
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
mf_valid   = [go for go, v in zip(mf_terms, valid_mask) if v]
Y_te_v     = Y_te[:, valid_mask]
print(f"  {len(mf_valid)} valid MF terms")

# ── 4. Embeddings ─────────────────────────────────────────────────────
print("\n[4] Loading ESM-2 embeddings...")
X30_te = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)
X30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
print(f"  Brain L30: {X30_te.shape}  Train L30: {X30_tr.shape}")

# ── 5. Sequence lengths from PEP file ─────────────────────────────────
print("\n[5] Sequence lengths from PEP file...")
iso2len = {}
cur_id, cur_len = None, 0
with open(PEP_FILE) as f:
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if cur_id: iso2len[cur_id] = cur_len
            cur_id  = line[1:].split()[0]
            cur_len = 0
        else:
            cur_len += len(line)
if cur_id: iso2len[cur_id] = cur_len

seq_lengths = np.array([iso2len.get(iso, 0) for iso in te_iso_list], dtype=np.float32)
covered_len = (seq_lengths > 0).sum()
print(f"  Length coverage: {covered_len}/{n_iso} ({100*covered_len/n_iso:.1f}%)")
print(f"  Mean length: {seq_lengths[seq_lengths>0].mean():.0f} aa  Median: {np.median(seq_lengths[seq_lengths>0]):.0f} aa")

# ── 6. Build prediction matrices ──────────────────────────────────────
print("\n[6] Building prediction matrices for baselines...")

# (a) PRISM v17f*
prism_preds = np.load(PRISM_PREDS).astype(np.float32)
if prism_preds.shape[1] != len(mf_valid):
    prism_preds = prism_preds[:, valid_mask]
print(f"  PRISM v17f*: {prism_preds.shape}")

# (b) Gene-mean oracle
gene_mean_preds = np.zeros_like(prism_preds)
for g, idxs in gene2idxs.items():
    gm = prism_preds[idxs].mean(0)
    for i in idxs: gene_mean_preds[i] = gm

# (c) Length baseline: broadcast length as scalar score across all GO terms
#     (simulates "longer isoforms rank higher")
length_preds = np.tile(seq_lengths[:, None], (1, len(mf_valid)))

# (d) Raw ESM-2 L2-norm: isoforms with richer embeddings may rank higher
esm2_norms = np.linalg.norm(X30_te, axis=1, keepdims=True)
norm_preds  = np.tile(esm2_norms, (1, len(mf_valid)))
print(f"  ESM-2 norm range: {esm2_norms.min():.2f} – {esm2_norms.max():.2f}")

# (e) Raw ESM-2 centroid similarity: per-GO centroid from training positives
#     Most principled "raw ESM-2 no training" baseline
print("  Building ESM-2 centroid similarity (per-GO training positives)...")
tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
tr_id_set    = set(sym2id.get(g, g) for g in tr_genes)

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

Xtr_n = normalize(X30_tr, axis=1)
Xte_n = normalize(X30_te, axis=1)

centroid_preds = np.zeros((n_iso, len(mf_valid)), dtype=np.float32)
for j, go_id in enumerate(mf_valid):
    pos_gene_ids = go_tr.get(go_id, set())
    pos_idx = [i for i, g in enumerate(tr_genes)
               if sym2id.get(g, g) in pos_gene_ids]
    if len(pos_idx) == 0:
        continue
    centroid = Xtr_n[pos_idx].mean(0, keepdims=True)  # (1, 640)
    centroid /= (np.linalg.norm(centroid) + 1e-8)
    centroid_preds[:, j] = (Xte_n @ centroid.T).ravel()
print(f"  Centroid similarity: {centroid_preds.shape}")

# (f) PRISM D0 (task-trained on 82 MF, no delta) if available
d0_available = os.path.exists(D0_PREDS)
if d0_available:
    d0_preds = np.load(D0_PREDS).astype(np.float32)
    if d0_preds.shape[1] != len(mf_valid):
        d0_preds = d0_preds[:, valid_mask]
    print(f"  PRISM D0: {d0_preds.shape}")
else:
    print(f"  PRISM D0: NOT FOUND at {D0_PREDS} (skipping)")
    d0_preds = None

# ── 7. Core: Domain-Ranking AUC ──────────────────────────────────────
print("\n[7] Computing Domain-Ranking AUC for each method...")

def compute_dr_auc(preds_mat, gene2idxs, iso_n_domains, Y_te_v):
    aucs = []
    for g, idxs in gene2idxs.items():
        if len(idxs) < 2: continue
        domains = iso_n_domains[idxs]
        if domains.std() < 0.1: continue
        gene_pos = np.where(Y_te_v[idxs[0]] > 0)[0]
        if len(gene_pos) == 0: continue
        med = np.median(domains)
        domain_bin = (domains > med).astype(float)
        if domain_bin.sum() == 0 or domain_bin.sum() == len(idxs): continue
        p_g = preds_mat[idxs]
        for t in gene_pos:
            scores = p_g[:, t]
            if scores.std() < 1e-8:
                aucs.append(0.5)
                continue
            try:
                aucs.append(roc_auc_score(domain_bin, scores))
            except:
                pass
    return np.mean(aucs) if aucs else 0.5, len(aucs)


methods = {
    'PRISM v17f*':          prism_preds,
    'Gene-mean oracle':     gene_mean_preds,
    'ESM-2 centroid sim':   centroid_preds,
    'ESM-2 L2-norm':        norm_preds,
    'Sequence length':      length_preds,
}
if d0_preds is not None:
    methods['PRISM D0 (no δ)'] = d0_preds

print()
auc_results = {}
for name, preds in methods.items():
    auc, n = compute_dr_auc(preds, gene2idxs, iso_n_domains, Y_te_v)
    auc_results[name] = auc
    print(f"  {name:<25}: {auc:.4f}  (N={n:,})")

# ── 8. Bootstrap CIs ──────────────────────────────────────────────────
print("\n[8] Bootstrap CIs (B=500)...")

def bootstrap_dr(preds_mat, gene2idxs, iso_n_domains, Y_te_v, n_boot=500, seed=42):
    rng_b = np.random.default_rng(seed)
    gene_list = [g for g, idxs in gene2idxs.items()
                 if len(idxs) >= 2
                 and iso_n_domains[idxs].std() >= 0.1
                 and Y_te_v[idxs[0]].sum() > 0]
    boots = []
    for _ in range(n_boot):
        sampled = rng_b.choice(gene_list, len(gene_list), replace=True)
        auc, _ = compute_dr_auc(preds_mat,
                                  {g: gene2idxs[g] for g in sampled},
                                  iso_n_domains, Y_te_v)
        boots.append(auc)
    boots = np.array(boots)
    return float(boots.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


ci_results = {}
for name in ['PRISM v17f*', 'Gene-mean oracle', 'ESM-2 centroid sim', 'Sequence length']:
    m, lo, hi = bootstrap_dr(methods[name], gene2idxs, iso_n_domains, Y_te_v, n_boot=500)
    ci_results[name] = (m, lo, hi)
    print(f"  {name:<25}: {m:.4f} [{lo:.4f}, {hi:.4f}]")

# ── 9. Final table ────────────────────────────────────────────────────
print("\n" + "="*65)
print("  DOMAIN-RANKING AUC — FINAL TABLE")
print("  Ground truth: Pfam domain count  |  Random null = 0.500")
print("="*65)
print(f"  {'Method':<28}  {'AUC':>6}  {'95% CI'}")
print("  " + "-"*60)

ordered = [
    'Gene-mean oracle',
    'Sequence length',
    'ESM-2 L2-norm',
    'ESM-2 centroid sim',
    'PRISM D0 (no δ)',
    'PRISM v17f*',
]
for name in ordered:
    if name not in auc_results: continue
    auc = auc_results[name]
    if name in ci_results:
        m, lo, hi = ci_results[name]
        print(f"  {name:<28}  {auc:.4f}  [{lo:.4f}, {hi:.4f}]")
    else:
        print(f"  {name:<28}  {auc:.4f}  [no CI]")

print()
prism_auc = auc_results.get('PRISM v17f*', 0)
raw_auc   = auc_results.get('ESM-2 centroid sim', 0)
len_auc   = auc_results.get('Sequence length', 0)
oracle    = auc_results.get('Gene-mean oracle', 0)

print(f"  PRISM v17f* gap above gene-mean:    {prism_auc - oracle:+.4f}")
print(f"  PRISM v17f* gap above raw ESM-2:    {prism_auc - raw_auc:+.4f}")
print(f"  PRISM v17f* gap above length:       {prism_auc - len_auc:+.4f}")

print("\n  MANUSCRIPT DEFENSE (MR1):")
if prism_auc - raw_auc > 0.01:
    print(f"  ✓ PRISM ({prism_auc:.3f}) > raw ESM-2 ({raw_auc:.3f}) by +{prism_auc-raw_auc:.3f}")
    print(f"    → Task-specific training ADDS isoform-discrimination signal (+{prism_auc-raw_auc:.3f})")
else:
    print(f"  ⚠ PRISM ({prism_auc:.3f}) ≈ raw ESM-2 ({raw_auc:.3f})")
    print(f"    → Domain-ranking may be driven by ESM-2 representation, not training")

if prism_auc - len_auc > 0.01:
    print(f"  ✓ PRISM ({prism_auc:.3f}) > length ({len_auc:.3f}) by +{prism_auc-len_auc:.3f}")
    print(f"    → PRISM is not dominated by sequence length")
else:
    print(f"  ⚠ PRISM ({prism_auc:.3f}) ≈ length ({len_auc:.3f}) — length confound present")

# Save
out = {
    'method_aucs': {k: float(v) for k, v in auc_results.items()},
    'bootstrap_cis': {k: {'mean': m, 'lo': lo, 'hi': hi}
                      for k, (m, lo, hi) in ci_results.items()},
}
with open(f'{OUT_DIR}/baseline_comparison.json', 'w') as fh:
    json.dump(out, fh, indent=2)
print(f"\n  Saved: {OUT_DIR}/baseline_comparison.json")
