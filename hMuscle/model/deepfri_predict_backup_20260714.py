#!/usr/bin/env python3
"""
deepfri_predict.py
==================
Run DeepFRI (sequence-only CNN) on brain test isoforms → compute AUPRC.

Prerequisites:
  - deepfri_setup.sh completed (DeepFRI installed, models downloaded)
  - Run with deepfri_env: /home/welcome1/miniconda3/envs/deepfri_env/bin/python

Usage:
  /home/welcome1/miniconda3/envs/deepfri_env/bin/python deepfri_predict.py
"""

import os, sys, json, gzip, time
import numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DEEPFRI_DIR = './DeepFRI'
MODEL_DIR   = f'{DEEPFRI_DIR}/trained_models'
DATA_DIR    = '../data'
ID_DIR      = '../data/raw_data/data/id_lists'
ANNOT_DIR   = '../data/raw_data/data/annotations'
OUT_DIR     = '../../reports/exp_e_sota'
PEP_FILE    = f'{DATA_DIR}/top30k_isoforms.pep'

FASTA_OUT   = f'{OUT_DIR}/brain_test_deepfri.fasta'
PRED_OUT    = f'{OUT_DIR}/deepfri_predictions.json'
RESULT_OUT  = f'{OUT_DIR}/deepfri_auprc.json'

os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, DEEPFRI_DIR)

# ── 1. Load test isoform list ─────────────────────────────────────
print("[1] Loading test isoforms...")

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

te_iso_raw = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
te_isos    = [clean(x) for x in te_iso_raw]
n_test     = len(te_isos)
print(f"  Test isoforms: {n_test}")

# ── 2. Extract protein sequences from top30k_isoforms.pep ─────────
print(f"[2] Parsing {PEP_FILE}...")

import re

TYPE_RANK = {'complete': 4, '5prime_partial': 3, '3prime_partial': 2, 'internal': 1}

pep_by_id = {}
cur_id, cur_meta, cur_seq = None, None, []

def flush_pep():
    global cur_id, cur_meta, cur_seq
    if cur_id is None: return
    seq = ''.join(cur_seq).replace('*', '').strip()
    if not seq: return
    rank, score = cur_meta
    prev = pep_by_id.get(cur_id)
    if prev is None or (rank, score) > prev[:2]:
        pep_by_id[cur_id] = (rank, score, seq[:1022])

with open(PEP_FILE) as f:
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('>'):
            flush_pep(); cur_seq = []
            m = re.match(r'>(\S+)', line)
            if not m: cur_id = None; continue
            raw = m.group(1)
            base = re.sub(r'\.p\d+$', '', raw)
            cur_id = base
            m_type = re.search(r'ORF type:(\S+)', line)
            m_sc   = re.search(r'score=([\d.]+)', line)
            orf_t  = m_type.group(1).split('(')[0] if m_type else 'internal'
            rank   = TYPE_RANK.get(orf_t, 1)
            score  = float(m_sc.group(1)) if m_sc else 0.0
            cur_meta = (rank, score)
        else:
            cur_seq.append(line)
flush_pep()

seqs_final = {k: v[2] for k, v in pep_by_id.items()}
print(f"  PEP unique IDs: {len(seqs_final)}")

# Match test isoforms to sequences
te_seq_list = []
found = 0
for iso in te_isos:
    base = re.sub(r'\.\d+$', '', iso)  # strip Ensembl version
    seq = seqs_final.get(iso) or seqs_final.get(base)
    if seq:
        te_seq_list.append(seq)
        found += 1
    else:
        te_seq_list.append(None)

print(f"  Sequences found: {found}/{n_test} ({100*found/n_test:.1f}%)")

# ── 3. Write FASTA for DeepFRI ────────────────────────────────────
print(f"[3] Writing FASTA: {FASTA_OUT}")
with open(FASTA_OUT, 'w') as f:
    for i, (iso, seq) in enumerate(zip(te_isos, te_seq_list)):
        if seq:
            f.write(f'>{i}|{iso}\n{seq}\n')

n_fasta = sum(1 for s in te_seq_list if s is not None)
print(f"  Written {n_fasta} sequences")

# ── 4. Run DeepFRI prediction (CNN sequence-only model) ───────────
print(f"\n[4] Running DeepFRI CNN predictions (MF)...")
print(f"  Model dir: {MODEL_DIR}")

if not os.path.exists(PRED_OUT):
    try:
        from deepfrier.Predictor import Predictor

        # CNN-LM_MF model = sequence-only (no PDB structure needed)
        model_path = f'{MODEL_DIR}/CNN-LM_MF0.05'
        if not os.path.exists(f'{model_path}.hdf5'):
            # Try alternative naming
            available = [f for f in os.listdir(MODEL_DIR) if 'MF' in f and f.endswith('.hdf5')]
            print(f"  Available MF models: {available}")
            model_path = os.path.join(MODEL_DIR, available[0].replace('.hdf5', '')) if available else None

        if model_path:
            pred = Predictor(model_name=model_path, gcn=False)
            pred.predict(FASTA_OUT)

            # Save raw predictions
            go_preds = {}
            for i, (name, scores) in enumerate(pred.test_preds.items()):
                go_preds[name] = {go: float(s) for go, s in zip(pred.go_terms, scores)}

            with open(PRED_OUT, 'w') as f:
                json.dump(go_preds, f, indent=2)
            print(f"  Predictions saved: {PRED_OUT}")
        else:
            print("  ERROR: No MF model found in MODEL_DIR")
            sys.exit(1)

    except Exception as e:
        print(f"  DeepFRI prediction failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
else:
    print(f"  Predictions already exist: {PRED_OUT}")
    with open(PRED_OUT) as f:
        go_preds = json.load(f)
    print(f"  Loaded {len(go_preds)} predictions")

# ── 5. Load GO labels and evaluate ───────────────────────────────
print("\n[5] Loading GO labels and evaluating...")
from sklearn.metrics import average_precision_score

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]

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

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])
mf_terms_set = set(mf_terms)

def build_Y_te(go_id):
    pos_ids = go_genes_all[go_id]
    return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0
                     for s in te_sym_list], dtype=np.float32)

Y_te = np.stack([build_Y_te(go) for go in mf_terms], axis=1)
valid_mask = Y_te.sum(0) >= 2
Y_te_v  = Y_te[:, valid_mask]
mf_valid = [go for go, v in zip(mf_terms, valid_mask) if v]

# H2 layer groups
H2_LAYERS = {}
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12: H2_LAYERS[p[0]] = p[11]
L2_idxs = [i for i, go in enumerate(mf_valid) if H2_LAYERS.get(go) == 'L2_Structural']

# Build DeepFRI score matrix
print("  Building score matrix from DeepFRI predictions...")
scores = np.zeros((n_test, len(mf_valid)), dtype=np.float32)

for key, preds in go_preds.items():
    idx = int(key.split('|')[0])
    for j, go in enumerate(mf_valid):
        if go in preds:
            scores[idx, j] = preds[go]

# Isoforms with no DeepFRI prediction get score=0 (no sequence coverage)
n_predicted = sum(1 for k in go_preds.keys() if '|' in k)
print(f"  Isoforms with DeepFRI predictions: {n_predicted}/{n_test}")

# Coverage mask: only isoforms with sequences
has_seq = np.array([1 if s else 0 for s in te_seq_list], dtype=bool)
print(f"  Coverage: {has_seq.sum()} isoforms with sequences")

# Evaluate on all isoforms (zeros for missing → conservative)
aps_all = [average_precision_score(Y_te_v[:, j], scores[:, j])
           for j in range(Y_te_v.shape[1]) if Y_te_v[:, j].sum() >= 2]
l2_aps  = [average_precision_score(Y_te_v[:, j], scores[:, j])
           for j in L2_idxs if Y_te_v[:, j].sum() >= 2]

auprc_all = float(np.mean(aps_all))
auprc_l2  = float(np.mean(l2_aps)) if l2_aps else float('nan')

print(f"\n  DeepFRI (CNN-LM, seq-only): All MF = {auprc_all:.4f}  L2 = {auprc_l2:.4f}")
print(f"  Coverage: {has_seq.sum()}/{n_test} ({100*has_seq.mean():.1f}%)")

# ── 6. Save results ───────────────────────────────────────────────
results = {
    'method': 'DeepFRI_CNN_LM_seq_only',
    'n_test': n_test,
    'n_with_sequences': int(has_seq.sum()),
    'coverage_pct': float(100 * has_seq.mean()),
    'auprc_all_mf': auprc_all,
    'auprc_l2_structural': auprc_l2,
    'reference': {
        'PRISM': {'all_mf': 0.5962, 'l2': 0.3127},
        'v17f':  {'all_mf': 0.7173, 'l2': 0.6219},
        'k_NN_ESM2': {'all_mf': 0.5992, 'l2': 0.4979},
        'D0': {'all_mf': None, 'l2': None}  # filled after d0_bootstrap_ci.json
    }
}

with open(RESULT_OUT, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[Done] {RESULT_OUT}")
print("\n=== SOTA Comparison Summary ===")
print(f"  {'Method':<25} {'All MF':>8} {'L2_Struct':>10}")
print(f"  {'-'*43}")
print(f"  {'PRISM':<25} {0.5962:>8.4f} {0.3127:>10.4f}")
print(f"  {'k-NN ESM-2':<25} {0.5992:>8.4f} {0.4979:>10.4f}")
print(f"  {'DeepFRI CNN-LM':<25} {auprc_all:>8.4f} {auprc_l2:>10.4f}")
print(f"  {'v17f (PRISM+δ_layer)':<25} {0.7173:>8.4f} {0.6219:>10.4f}")
