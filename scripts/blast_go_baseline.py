"""
ESM-2 k-NN GO Transfer Baseline
=================================
BLAST 없이 ESM-2 cosine similarity를 sequence homology proxy로 사용.
SwissProt (82703개) 전체를 reference DB로 사용하여
test isoforms (36748개)에 대해 GO 전달.

두 가지 variant:
  (A) knn_swissprot_5  : SwissProt DB k=5 cosine NN → majority vote
  (B) knn_train_5      : Human train set k=5 cosine NN → majority vote

두 baseline 모두 계산하여 더 강한 것을 보고.
"""

import numpy as np
import json
import os
import sys
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

BASE = '/home/welcome1/sw1686/DIFFUSE'
DATA_DIR  = f'{BASE}/hMuscle/data'
ANNOT_DIR = f'{BASE}/hMuscle/data/raw_data/data/annotations'
ID_DIR    = f'{BASE}/hMuscle/data/raw_data/data/id_lists'
OUT_DIR   = f'{BASE}/reports'

os.makedirs(OUT_DIR, exist_ok=True)

# ── 18 GO terms (v15d_bp_clean 동일) ─────────────────────────────────────────
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
GO_KEYS = list(GO_TERMS.keys())
N_GO = len(GO_KEYS)

# ── Load embeddings ──────────────────────────────────────────────────────────
print("Loading embeddings...")
X_te = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)   # (36748, 640)
X_tr = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)  # (31668, 640)
X_sw = np.load(f'{DATA_DIR}/esm2_train_swissprot_t30_150M.npy').astype(np.float32)  # (82703, 640)
print(f"  Test: {X_te.shape}  Train: {X_tr.shape}  SwissProt: {X_sw.shape}")

# ── Load IDs ─────────────────────────────────────────────────────────────────
def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

sw_list  = load_ids(f'{ID_DIR}/train_swissprot_list.npy')     # (82703,)
tr_iso   = load_ids(f'{ID_DIR}/train_isoform_list.npy')       # (31668,)
te_iso   = load_ids('my_isoform_list_fixed.npy')              # (36748,)
te_gene  = load_ids('my_gene_list_fixed.npy')                 # (36748,)

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

te_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_gene]

# ── Build GO label dicts ─────────────────────────────────────────────────────
print("Building GO label dicts...")

# Human gene → GO set
gene_go = {}
with open(f'{ANNOT_DIR}/human_annotations_unified_bp.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > 1:
            gene_go[parts[0]] = set(parts[1:])

# SwissProt entry → GO set
sw_go = {}
with open(f'{ANNOT_DIR}/swissprot_annotations.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > 1:
            sw_go[parts[0]] = set(parts[1:])

# ── Ground-truth test labels (gene-level, same as v15d) ──────────────────────
def get_te_labels(go_term):
    return np.array([1 if go_term in gene_go.get(sym, set()) else 0
                     for sym in te_sym], dtype=np.float32)

# ── Normalize embeddings for cosine similarity ───────────────────────────────
print("Normalizing embeddings...")
X_te_n  = normalize(X_te,  norm='l2')
X_tr_n  = normalize(X_tr,  norm='l2')
X_sw_n  = normalize(X_sw,  norm='l2')

# ── k-NN GO transfer: batch cosine similarity ────────────────────────────────
K = 5
BATCH = 512  # test isoforms per batch

def knn_go_transfer_scores(X_query_n, X_ref_n, ref_go_list, ref_has_labels,
                            go_term, k=K, batch=BATCH):
    """
    For each query, find k nearest neighbors in ref,
    return fraction of neighbors annotated with go_term (soft score).
    ref_go_list: list of sets (GO annotations per ref entry)
    ref_has_labels: bool array, True if ref has annotation info
    """
    n_query = X_query_n.shape[0]
    scores = np.zeros(n_query, dtype=np.float32)

    # Precompute which ref entries have the GO term
    ref_pos = np.array([go_term in gos for gos in ref_go_list], dtype=np.float32)

    for start in range(0, n_query, batch):
        end = min(start + batch, n_query)
        sim = X_query_n[start:end] @ X_ref_n.T   # (batch, n_ref)
        # For each query, top-k indices
        topk_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
        for i, qidx in enumerate(range(start, end)):
            nn_pos = ref_pos[topk_idx[i]].mean()  # fraction positive
            scores[qidx] = nn_pos
    return scores

# ── Variant A: SwissProt k-NN ─────────────────────────────────────────────────
print(f"\n=== Variant A: SwissProt k-NN (k={K}) ===")
sw_go_list = [sw_go.get(sid, set()) for sid in sw_list]
sw_ref_has = np.array([sid in sw_go for sid in sw_list], dtype=bool)
print(f"  SwissProt entries with GO annotations: {sw_ref_has.sum()} / {len(sw_list)}")

auprc_sw = []
per_go_sw = []
for go_term in GO_KEYS:
    y_te = get_te_labels(go_term)
    if y_te.sum() == 0:
        auprc_sw.append(0.0)
        per_go_sw.append({'go': go_term, 'auprc': 0.0, 'n_pos': 0})
        continue
    scores = knn_go_transfer_scores(X_te_n, X_sw_n, sw_go_list, sw_ref_has,
                                     go_term, k=K)
    ap = average_precision_score(y_te, scores)
    auprc_sw.append(ap)
    per_go_sw.append({'go': go_term, 'name': GO_TERMS[go_term],
                      'auprc': round(float(ap), 4), 'n_pos': int(y_te.sum())})
    print(f"  {go_term} ({GO_TERMS[go_term][:20]:20s}): AUPRC={ap:.4f}  n_pos={int(y_te.sum())}")

macro_sw = float(np.mean([x for x in auprc_sw if x > 0]))
macro_sw_all = float(np.mean(auprc_sw))
print(f"\n  Macro AUPRC (non-zero GO only): {macro_sw:.4f}")
print(f"  Macro AUPRC (all 18 GO):        {macro_sw_all:.4f}")

# ── Variant B: Human train k-NN ───────────────────────────────────────────────
print(f"\n=== Variant B: Human train k-NN (k={K}) ===")

def load_tr_sym():
    tr_gene = load_ids(f'{ID_DIR}/train_gene_list.npy')
    return [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_gene]

tr_sym = load_tr_sym()
tr_go_list = [gene_go.get(sym, set()) for sym in tr_sym]
tr_ref_has = np.array([sym in gene_go for sym in tr_sym], dtype=bool)
print(f"  Train entries with GO annotations: {tr_ref_has.sum()} / {len(tr_sym)}")

auprc_tr = []
per_go_tr = []
for go_term in GO_KEYS:
    y_te = get_te_labels(go_term)
    if y_te.sum() == 0:
        auprc_tr.append(0.0)
        per_go_tr.append({'go': go_term, 'auprc': 0.0, 'n_pos': 0})
        continue
    scores = knn_go_transfer_scores(X_te_n, X_tr_n, tr_go_list, tr_ref_has,
                                     go_term, k=K)
    ap = average_precision_score(y_te, scores)
    auprc_tr.append(ap)
    per_go_tr.append({'go': go_term, 'name': GO_TERMS[go_term],
                      'auprc': round(float(ap), 4), 'n_pos': int(y_te.sum())})
    print(f"  {go_term} ({GO_TERMS[go_term][:20]:20s}): AUPRC={ap:.4f}  n_pos={int(y_te.sum())}")

macro_tr = float(np.mean([x for x in auprc_tr if x > 0]))
macro_tr_all = float(np.mean(auprc_tr))
print(f"\n  Macro AUPRC (non-zero GO only): {macro_tr:.4f}")
print(f"  Macro AUPRC (all 18 GO):        {macro_tr_all:.4f}")

# ── Best homology baseline ────────────────────────────────────────────────────
best_macro = max(macro_sw_all, macro_tr_all)
best_method = 'esm2_knn_swissprot_5' if macro_sw_all >= macro_tr_all else 'esm2_knn_train_5'
PRISM_AUPRC = 0.7022
DOMAIN_LR   = 0.0156

print(f"\n{'='*60}")
print(f"  Best homology baseline:  {best_method}")
print(f"  Best macro AUPRC:        {best_macro:.4f}")
print(f"  Domain-only LR:          {DOMAIN_LR:.4f}")
print(f"  PRISM:                   {PRISM_AUPRC:.4f}")
print(f"  PRISM vs homology fold:  {PRISM_AUPRC / best_macro:.1f}x")
print(f"  PRISM vs domain-LR fold: {PRISM_AUPRC / DOMAIN_LR:.1f}x")
print(f"{'='*60}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'method_a': 'esm2_knn_swissprot_5',
    'method_b': 'esm2_knn_train_5',
    'macro_auprc_swissprot_all18': round(macro_sw_all, 4),
    'macro_auprc_swissprot_nonzero': round(macro_sw, 4),
    'macro_auprc_train_all18': round(macro_tr_all, 4),
    'macro_auprc_train_nonzero': round(macro_tr, 4),
    'best_homology_baseline': best_method,
    'best_macro_auprc': round(best_macro, 4),
    'per_go_swissprot': per_go_sw,
    'per_go_train': per_go_tr,
    'comparison': {
        'domain_only_lr':          DOMAIN_LR,
        'homology_knn_best':       round(best_macro, 4),
        'prism':                   PRISM_AUPRC,
        'prism_vs_homology_fold':  round(PRISM_AUPRC / best_macro, 2),
        'prism_vs_domain_lr_fold': round(PRISM_AUPRC / DOMAIN_LR, 2),
    },
    'note': (
        'ESM-2 cosine k-NN (k=5) as sequence-homology GO transfer proxy. '
        'No BLAST available; ESM-2 encodes sequence identity, so cosine NN '
        'approximates sequence similarity-based GO transfer. '
        'SwissProt variant uses 82703 cross-species annotated proteins as DB; '
        'Train variant uses 31668 human training isoforms.'
    )
}

out_path = f'{OUT_DIR}/blast_go_baseline.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n[Saved] {out_path}")
