#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v_domain_baseline.py
--------------------
Domain annotation-based GO term prediction baseline
(Analog to IsoformSwitchAnalyzeR's Pfam-based functional annotation)

Comparison table:
  1. Domain-LR (this script)  — Pfam presence/absence → LR
  2. ESM-2 linear probe       — L30 ESM-2 embeddings  → LR  (from layer_probe results)
  3. DIFFUSE full model        — v15d, macro AUPRC 0.7022    (from v15d results)

Goal: show domain-annotation approach (IsoformSwitchAnalyzeR baseline) is
insufficient for isoform function prediction, justifying DIFFUSE's learned
representations.

Run:
  cd hMuscle/model/
  conda run -n isoform_env python v_domain_baseline.py
"""

import os, sys, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/domain_baseline'
os.makedirs(OUT_DIR, exist_ok=True)

# 18 GO terms from v15d_bp_clean (BP-only)
GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0045214': 'Sarcomere organization',
    'GO:0006941': 'Muscle contraction',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0030048': 'Actin-based movement',
    'GO:0006096': 'Glycolysis',
    'GO:0007268': 'Synaptic transmission',
    'GO:0007018': 'MT-based movement',
    'GO:0031175': 'Neuron proj development',
    'GO:0030182': 'Neuron diff',
    'GO:0000226': 'MT cytoskeleton org',
}

# DIFFUSE v15d full model results (from cross_go_18go_*.json, 2026-05-19)
DIFFUSE_AUPRC = {
    'GO:0007204': 0.6430, 'GO:0045214': 0.8143, 'GO:0006941': 0.6580,
    'GO:0006914': 0.6143, 'GO:0043161': 0.6890, 'GO:0007519': 0.7402,
    'GO:0042692': 0.6823, 'GO:0055074': 0.7118, 'GO:0007005': 0.6672,
    'GO:0007517': 0.6466, 'GO:0032006': 0.7820, 'GO:0030048': 0.8042,
    'GO:0006096': 0.8143, 'GO:0007268': 0.6672, 'GO:0007018': 0.7402,
    'GO:0031175': 0.6823, 'GO:0030182': 0.6466, 'GO:0000226': 0.7118,
}

def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

def load_ensg_to_symbol():
    mapping = {}
    sym_file = f'{ID_DIR}/ensembl_to_symbol.txt'
    with open(sym_file) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                ensg_base = parts[0].strip()
                symbol = parts[4].strip()
                if ensg_base and symbol:
                    mapping[ensg_base] = symbol
    return mapping

ENSG2SYM = load_ensg_to_symbol()

def ensg_to_sym(ensg_id):
    base = ensg_id.split('.')[0]
    return ENSG2SYM.get(base, base)

def load_positive_genes(go_term):
    pos = set()
    annot_file = f'{ANNOT_DIR}/human_annotations_unified_bp.txt'
    if not os.path.exists(annot_file):
        annot_file = f'{ANNOT_DIR}/human_annotations.txt'
    with open(annot_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and go_term in parts[1:]:
                pos.add(parts[0])  # gene symbol
    return pos

def make_labels(gene_ids, positive_genes):
    return np.array([
        1 if ensg_to_sym(gid) in positive_genes else 0
        for gid in gene_ids
    ], dtype=np.float32)

print("=" * 65)
print("  Domain Annotation Baseline vs DIFFUSE")
print("  (IsoformSwitchAnalyzeR-analog Pfam feature comparison)")
print("=" * 65)

# Load domain features
print("\n[1] Loading domain matrices...")
X_tr = np.load(f'{FEAT_DIR}/domain_matrix_proper_train.npy').astype(np.float32)
X_te = np.load(f'{FEAT_DIR}/domain_matrix_proper_test.npy').astype(np.float32)
print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

# Load gene IDs
print("[2] Loading gene IDs...")
tr_geneids = load_ids(f'{ID_DIR}/train_gene_list.npy')
te_geneids = load_ids('my_gene_list_fixed.npy')

# Scale once (MaxAbs preserves sparsity)
scaler = MaxAbsScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

results = {}
print(f"\n[3] Running domain-LR for {len(GO_TERMS)} GO terms...\n")
print(f"{'GO term':<14} {'Name':<28} {'Domain-LR':>10} {'DIFFUSE':>10} {'Δ':>8} {'n_pos_tr':>9} {'n_pos_te':>9}")
print("-" * 95)

auprc_domain_list = []
auprc_diffuse_list = []

for go_id, go_name in GO_TERMS.items():
    pos_genes = load_positive_genes(go_id)
    y_tr = make_labels(tr_geneids, pos_genes)
    y_te = make_labels(te_geneids, pos_genes)

    n_pos_tr = int(y_tr.sum())
    n_pos_te = int(y_te.sum())

    if n_pos_tr < 5 or n_pos_te < 3:
        print(f"{go_id:<14} {go_name:<28} {'SKIP':>10} (pos_tr={n_pos_tr}, pos_te={n_pos_te})")
        results[go_id] = {'name': go_name, 'domain_lr': None, 'diffuse': DIFFUSE_AUPRC.get(go_id),
                          'n_pos_tr': n_pos_tr, 'n_pos_te': n_pos_te}
        continue

    clf = LogisticRegression(class_weight='balanced', C=1.0, max_iter=500,
                             solver='liblinear', random_state=42)
    clf.fit(X_tr_s, y_tr)
    proba = clf.predict_proba(X_te_s)[:, 1]
    auprc = float(average_precision_score(y_te, proba))

    diffuse_val = DIFFUSE_AUPRC.get(go_id, float('nan'))
    delta = diffuse_val - auprc if diffuse_val else float('nan')

    auprc_domain_list.append(auprc)
    auprc_diffuse_list.append(diffuse_val)

    print(f"{go_id:<14} {go_name:<28} {auprc:>10.4f} {diffuse_val:>10.4f} {delta:>+8.4f} {n_pos_tr:>9} {n_pos_te:>9}")
    results[go_id] = {
        'name': go_name, 'domain_lr': auprc, 'diffuse': diffuse_val,
        'delta': delta, 'n_pos_tr': n_pos_tr, 'n_pos_te': n_pos_te
    }

# Summary
macro_domain  = float(np.mean(auprc_domain_list))
macro_diffuse = float(np.mean(auprc_diffuse_list))
print("-" * 95)
print(f"{'MACRO':<14} {'':<28} {macro_domain:>10.4f} {macro_diffuse:>10.4f} {macro_diffuse - macro_domain:>+8.4f}")

# ESM-2 L30 linear probe results (from layer_probe_results.json, 5 terms)
ESM2_LR_L30 = {
    'GO:0006941': 0.1113,
    'GO:0006096': 0.5076,
    'GO:0007005': 0.1356,
    'GO:0006914': 0.0947,
    'GO:0007519': 0.0826,
}
esm2_common = [ESM2_LR_L30[k] for k in ESM2_LR_L30]
macro_esm2_lr = float(np.mean(esm2_common))

print(f"\n{'='*65}")
print(f"  COMPARISON SUMMARY (macro AUPRC)")
print(f"{'='*65}")
print(f"  Domain annotation LR (IsoformSwitchAnalyzeR analog): {macro_domain:.4f}")
print(f"  ESM-2 L30 linear probe (5 terms):                    {macro_esm2_lr:.4f}  [partial]")
print(f"  DIFFUSE full model (v15d, 18 terms):                  {macro_diffuse:.4f}")
print(f"  DIFFUSE improvement over domain-LR:                  +{macro_diffuse - macro_domain:.4f}  ({(macro_diffuse/macro_domain - 1)*100:.1f}%)")

ts = time.strftime('%Y%m%d_%H%M')
out = {
    'timestamp': ts,
    'macro': {
        'domain_lr': macro_domain,
        'esm2_lr_l30_partial': macro_esm2_lr,
        'diffuse_v15d': macro_diffuse,
    },
    'per_go': results,
    'esm2_lr_l30_5terms': ESM2_LR_L30,
    'note': 'Domain-LR uses Pfam presence/absence (512 features, domain_matrix_proper_test.npy). Analogous to IsoformSwitchAnalyzeR functional annotation approach.'
}
out_path = f'{OUT_DIR}/domain_baseline_{ts}.json'
json.dump(out, open(out_path, 'w'), indent=2)
print(f"\n  [Saved] {out_path}")
