#!/usr/bin/env python3
"""
deepfri_eval_v2_truebrain.py
------------------------------
TISSUE-MISLABELING BUGFIX RERUN (2026-07-14).
Original deepfri_eval.py evaluated against my_gene_list_fixed.npy -- MUSCLE data
(36748 isoforms), despite being described as "brain zero-shot". This rerun evaluates
against the TRUE brain gene list (brain_full_gene_names.npy, 63994 isoforms / 18514
unique genes) and the predictions produced by deepfri_predict_v2_truebrain.py
(reports/truebrain_rerun_20260714/exp_e_sota/deepfri_truebrain_MF_pred_scores.json).
Original (mislabeled-as-brain, actually muscle): deepfri_eval.py, backed up at
deepfri_eval_backup_20260714.py before this rerun.

deepfri_eval.py
===============
Parse DeepFRI output → compute AUPRC on our 82 MF terms.
Run AFTER deepfri predict.py completes.
"""

import os, gzip, json
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
BRAIN_DIR = '../data/brain_isoquant_esm2/full'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/truebrain_rerun_20260714/exp_e_sota'

PRED_JSON = f'{OUT_DIR}/deepfri_truebrain_MF_pred_scores.json'
OUT_JSON  = f'{OUT_DIR}/deepfri_auprc.json'

# ── Load DeepFRI predictions ──────────────────────────────────────
print(f"[1] Loading DeepFRI predictions: {PRED_JSON}")
with open(PRED_JSON) as f:
    raw = json.load(f)

pdb_chains = raw['pdb_chains']    # list of protein names
Y_hat      = np.array(raw['Y_hat'])   # (n_prot, n_goterms)
goterms    = raw['goterms']       # list of GO term IDs
gonames    = raw['gonames']

print(f"  Proteins: {len(pdb_chains)}  DeepFRI GO terms: {len(goterms)}")
print(f"  Y_hat shape: {Y_hat.shape}")

# Build index from chain name to row in Y_hat
# chain format: "0|BambuTx10", "1|BambuTx100", etc.
chain2idx = {}
for i, chain in enumerate(pdb_chains):
    idx = int(chain.split('|')[0])
    chain2idx[idx] = i

# ── Load our 82 MF terms ──────────────────────────────────────────
print("\n[2] Loading our 82 MF terms...")
mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])
mf_terms_set = set(mf_terms)

# Check overlap between DeepFRI GO terms and our 82 MF terms
deepfri_mf_set = set(goterms)
overlap = mf_terms_set & deepfri_mf_set
print(f"  Our 82 MF terms: {len(mf_terms)}")
print(f"  DeepFRI MF terms: {len(goterms)}")
print(f"  Overlap: {len(overlap)} terms ({100*len(overlap)/len(mf_terms):.1f}% of our terms)")

# Map DeepFRI GO term index to our term index
deepfri_go2idx = {go: i for i, go in enumerate(goterms)}

# ── Load GO labels ────────────────────────────────────────────────
print("\n[3] Loading GO labels...")

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
n_test = len(te_sym_list)

sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2: sym2id[p[2]] = p[1]

go_genes_all = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        if p[7] != 'Function': continue
        go_genes_all[p[2]].add(p[1])

def build_Y_te(go_id):
    pos_ids = go_genes_all[go_id]
    return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0
                     for s in te_sym_list], dtype=np.float32)

Y_te = np.stack([build_Y_te(go) for go in mf_terms], axis=1)
valid_mask = Y_te.sum(0) >= 2
Y_te_v    = Y_te[:, valid_mask]
mf_valid  = [go for go, v in zip(mf_terms, valid_mask) if v]
print(f"  Valid MF terms (≥2 positives): {valid_mask.sum()}")

# H2 layer groups
H2_LAYERS = {}
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12: H2_LAYERS[p[0]] = p[11]
L2_idxs = [i for i, go in enumerate(mf_valid) if H2_LAYERS.get(go) == 'L2_Structural']
L4_idxs = [i for i, go in enumerate(mf_valid) if H2_LAYERS.get(go) == 'L4_CellState']
print(f"  L2_Structural: {len(L2_idxs)}  L4_CellState: {len(L4_idxs)}")

# ── Build score matrix ────────────────────────────────────────────
print("\n[4] Building DeepFRI score matrix (n_test × n_valid_terms)...")
scores = np.zeros((n_test, len(mf_valid)), dtype=np.float32)

# For each valid term, find if it's in DeepFRI's vocabulary
terms_covered = 0
for j, go in enumerate(mf_valid):
    if go in deepfri_go2idx:
        df_idx = deepfri_go2idx[go]
        terms_covered += 1
        # Fill scores from Y_hat rows corresponding to each test isoform
        for test_i in range(n_test):
            if test_i in chain2idx:
                scores[test_i, j] = float(Y_hat[chain2idx[test_i], df_idx])

n_isoforms_predicted = len(chain2idx)
coverage_pct = 100 * n_isoforms_predicted / n_test
terms_coverage_pct = 100 * terms_covered / len(mf_valid)

print(f"  Isoforms with DeepFRI predictions: {n_isoforms_predicted}/{n_test} ({coverage_pct:.1f}%)")
print(f"  Valid MF terms in DeepFRI vocab: {terms_covered}/{len(mf_valid)} ({terms_coverage_pct:.1f}%)")

# ── Evaluate AUPRC ────────────────────────────────────────────────
print("\n[5] Computing AUPRC...")

# Full eval (zeros for uncovered terms/isoforms)
aps_all = [average_precision_score(Y_te_v[:, j], scores[:, j])
           for j in range(Y_te_v.shape[1]) if Y_te_v[:, j].sum() >= 2]
l2_aps  = [average_precision_score(Y_te_v[:, j], scores[:, j])
           for j in L2_idxs if Y_te_v[:, j].sum() >= 2]
l4_aps  = [average_precision_score(Y_te_v[:, j], scores[:, j])
           for j in L4_idxs if Y_te_v[:, j].sum() >= 2]

auprc_all = float(np.mean(aps_all))
auprc_l2  = float(np.mean(l2_aps)) if l2_aps else float('nan')
auprc_l4  = float(np.mean(l4_aps)) if l4_aps else float('nan')

# Covered-only eval (only terms in DeepFRI vocab)
covered_idxs = [j for j, go in enumerate(mf_valid) if go in deepfri_go2idx]
aps_cov = [average_precision_score(Y_te_v[:, j], scores[:, j])
           for j in covered_idxs if Y_te_v[:, j].sum() >= 2]
auprc_cov = float(np.mean(aps_cov)) if aps_cov else float('nan')

print(f"\n  DeepFRI (CNN-LM, seq-only):")
print(f"    All 82 MF (full eval):     {auprc_all:.4f}")
print(f"    All 82 MF (covered only):  {auprc_cov:.4f}  [{terms_covered} terms]")
print(f"    L2_Structural:             {auprc_l2:.4f}")
print(f"    L4_CellState:              {auprc_l4:.4f}")

# ── Comparison table ──────────────────────────────────────────────
refs = {
    'PRISM':          (0.5962, 0.3127),
    'k-NN ESM-2':     (0.5992, 0.4979),
    'Gene-mean ESM2': (0.4651, 0.3029),
    'D0 (frozen L30)': (None, None),  # fill from d0_bootstrap_ci.json
    'v17f':           (0.7173, 0.6219),
}

d0_json = f'../../reports/truebrain_rerun_20260714/exp_d_finetune/d0_bootstrap_ci.json'
if os.path.exists(d0_json):
    with open(d0_json) as f:
        d0_data = json.load(f)
    refs['D0 (frozen L30)'] = (d0_data['D0']['all_mf'], d0_data['D0']['l2'])

print(f"\n{'Method':<28} {'All MF':>8} {'L2_Struct':>10}")
print(f"{'-'*48}")
for name, (a, l) in refs.items():
    a_s = f'{a:.4f}' if a is not None else 'pending'
    l_s = f'{l:.4f}' if l is not None else 'pending'
    print(f"  {name:<26} {a_s:>8} {l_s:>10}")
print(f"  {'DeepFRI CNN-LM':<26} {auprc_all:>8.4f} {auprc_l2:>10.4f}")
print(f"  {'v17f':<26} {0.7173:>8.4f} {0.6219:>10.4f}")

# ── Save ──────────────────────────────────────────────────────────
results = {
    'DeepFRI_CNN_LM': {
        'all_mf': auprc_all,
        'all_mf_covered_only': auprc_cov,
        'l2_structural': auprc_l2,
        'l4_cellstate': auprc_l4,
        'n_test': n_test,
        'n_isoforms_predicted': n_isoforms_predicted,
        'coverage_isoforms_pct': coverage_pct,
        'n_terms_in_vocab': terms_covered,
        'n_valid_terms': len(mf_valid),
        'coverage_terms_pct': terms_coverage_pct,
    },
    'reference': {
        'PRISM':      {'all_mf': 0.5962, 'l2': 0.3127},
        'k_NN_ESM2':  {'all_mf': 0.5992, 'l2': 0.4979},
        'Gene_mean':  {'all_mf': 0.4651, 'l2': 0.3029},
        'v17f':       {'all_mf': 0.7173, 'l2': 0.6219},
    }
}

if os.path.exists(d0_json):
    results['reference']['D0'] = {'all_mf': d0_data['D0']['all_mf'], 'l2': d0_data['D0']['l2']}

with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[Done] {OUT_JSON}")
