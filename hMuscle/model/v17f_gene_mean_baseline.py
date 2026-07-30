#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_gene_mean_baseline.py  (Option A)
---------------------------------------
Gene-mean baseline: identical architecture to v17f* but EVERY isoform of
the same gene receives the same embedding (gene-level mean).

If gene-mean AUPRC ~ v17f* (0.734):  v17f* is a gene-level classifier.
If gene-mean AUPRC << v17f* (0.734): per-isoform variation is informative.

Interpretation threshold:
  gap < 0.01  → essentially gene-level
  gap 0.01-0.03 → partial isoform signal
  gap > 0.03  → genuine isoform-specific information
"""

import os, json, gzip, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/v17f_gene_mean_baseline'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13, 21, 99]
BATCH_MLP  = 512
EPOCHS_MLP = 60
LAYER_A    = 15
LAYER_B    = 30
VSTAR_REF  = 0.7343

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  Gene-mean baseline: can the model learn WITHOUT isoform signal?")
print("=" * 65)

# ── 1. Load embeddings ────────────────────────────────────────────
print("\n[1] Loading embeddings...")
X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l15_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
X_l30_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l15_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
print(f"  Train: {X_l30_tr.shape}  Test: {X_l30_te.shape}")

# ── 2. IDs ────────────────────────────────────────────────────────
print("\n[2] Loading gene IDs and building gene-mean embeddings...")
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]

# Build gene->indices maps
tr_gene2idxs = defaultdict(list)
for i, g in enumerate(tr_genes): tr_gene2idxs[g].append(i)

te_gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_list): te_gene2idxs[g].append(i)

print(f"  Train: {len(tr_genes)} isoforms, {len(tr_gene2idxs)} genes")
print(f"  Test:  {len(te_sym_list)} isoforms, {len(te_gene2idxs)} genes")

# ── 3. Gene-mean embeddings ───────────────────────────────────────
# Replace each isoform's embedding with its gene's mean embedding
def make_gene_mean(X, gene2idxs, genes):
    """Replace each isoform embedding with gene-level mean."""
    X_gm = np.zeros_like(X)
    for gene, idxs in gene2idxs.items():
        gm = X[idxs].mean(axis=0)
        for i in idxs:
            X_gm[i] = gm
    return X_gm

print("  Computing gene-mean embeddings (train)...")
X_l30_tr_gm = make_gene_mean(X_l30_tr, tr_gene2idxs, tr_genes)
X_l15_tr_gm = make_gene_mean(X_l15_tr, tr_gene2idxs, tr_genes)

print("  Computing gene-mean embeddings (test)...")
X_l30_te_gm = make_gene_mean(X_l30_te, te_gene2idxs, te_sym_list)
X_l15_te_gm = make_gene_mean(X_l15_te, te_gene2idxs, te_sym_list)

# Gene-mean delta
delta_tr_gm = (X_l30_tr_gm - X_l15_tr_gm).astype(np.float32)
delta_te_gm = (X_l30_te_gm - X_l15_te_gm).astype(np.float32)
scaler = MaxAbsScaler()
delta_tr_gm_s = scaler.fit_transform(delta_tr_gm).astype(np.float32)
delta_te_gm_s = scaler.transform(delta_te_gm).astype(np.float32)

# Verify: check that within-gene variance is now 0
sample_gene = list(tr_gene2idxs.keys())[0]
idxs_sample = tr_gene2idxs[sample_gene]
if len(idxs_sample) > 1:
    within_var = X_l30_tr_gm[idxs_sample].std(axis=0).mean()
    print(f"  Within-gene std after gene-mean (should be 0): {within_var:.6f}")

tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

# ── 4. GO labels ──────────────────────────────────────────────────
print("\n[3] Loading GO labels...")
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

tr_ids    = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_ids)
go_genes_tr  = defaultdict(set)
go_genes_all = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606' or p[7] != 'Function': continue
        go_genes_all[p[2]].add(p[1])
        if p[1] in tr_id_set: go_genes_tr[p[2]].add(p[1])

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
        for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
    return y

def build_Y_te(go_id):
    pos_ids = go_genes_all[go_id]
    return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0
                     for s in te_sym_list], dtype=np.float32)

Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)
Y_te = np.stack([build_Y_te(go) for go in mf_terms], axis=1)
valid_mask = Y_te.sum(0) >= 2

L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural': L2_TERMS.add(p[0])

valid_idx = [i for i in range(len(mf_terms)) if valid_mask[i]]
l2_valid  = [i for i in range(len(mf_terms)) if mf_terms[i] in L2_TERMS and valid_mask[i]]
print(f"  {len(mf_terms)} MF | valid: {valid_mask.sum()} | L2: {len(l2_valid)}")

# ── 5. Train + Evaluate ───────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"\n  GPU: {gpus[0].name}")

def build_mlp(n_go=82):
    inp_d = layers.Input(shape=(640,))
    inp_e = layers.Input(shape=(640,))
    x     = layers.Concatenate()([inp_d, inp_e])
    x     = layers.Dense(256, activation='relu')(x)
    x     = layers.BatchNormalization()(x)
    x     = layers.Dropout(0.2)(x)
    x     = layers.Dense(128, activation='relu')(x)
    out   = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model([inp_d, inp_e], out)

focal_fn  = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
n_go      = len(mf_terms)
all_preds = []

print(f"\n[4] 5-seed ensemble (gene-mean inputs, {EPOCHS_MLP} epochs)...")
t0 = time.time()
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm    = np.random.permutation(len(delta_tr_gm_s))
    n_val   = int(len(delta_tr_gm_s) * 0.1)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    mlp = build_mlp(n_go=n_go)
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
    mlp.fit(
        [delta_tr_gm_s[tr_idx], X_l30_tr_gm[tr_idx]], Y_tr[tr_idx],
        validation_data=([delta_tr_gm_s[val_idx], X_l30_tr_gm[val_idx]], Y_tr[val_idx]),
        epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    preds_i = mlp.predict([delta_te_gm_s, X_l30_te_gm], batch_size=1024, verbose=0)
    all_preds.append(preds_i)
    aps = [average_precision_score(Y_te[:, i], preds_i[:, i])
           for i in valid_idx if Y_te[:, i].sum() >= 2]
    print(f"  seed={seed}  AUPRC={np.mean(aps):.4f}")

preds = np.mean(all_preds, axis=0)
print(f"  Ensemble done: {time.time()-t0:.0f}s")

# ── 6. Evaluate ───────────────────────────────────────────────────
def macro_auprc(preds, idxs):
    aps = [average_precision_score(Y_te[:, i], preds[:, i])
           for i in idxs if Y_te[:, i].sum() >= 2]
    return float(np.mean(aps)) if aps else float('nan')

auprc_all = macro_auprc(preds, valid_idx)
auprc_l2  = macro_auprc(preds, l2_valid)
gap       = VSTAR_REF - auprc_all

if gap < 0.01:
    verdict = "GENE-LEVEL CLASSIFIER: gene-mean ~ v17f*. Isoform variation contributes nothing."
elif gap < 0.03:
    verdict = f"PARTIAL ISOFORM SIGNAL: gap={gap:.3f}. Per-isoform info adds modest value."
else:
    verdict = f"GENUINE ISOFORM SIGNAL: gap={gap:.3f}. v17f* captures isoform-specific information."

print(f"\n{'='*65}")
print(f"  Gene-mean Baseline Results")
print(f"{'='*65}")
print(f"  Gene-mean  All MF: {auprc_all:.4f}  L2: {auprc_l2:.4f}")
print(f"  v17f*      All MF: {VSTAR_REF:.4f}  (reference)")
print(f"  Gap (v17f* - gene-mean): {gap:+.4f}")
print(f"  Verdict: {verdict}")
print(f"{'='*65}")

results = {
    'gene_mean_auprc_all_mf': auprc_all,
    'gene_mean_auprc_l2': auprc_l2,
    'vstar_ref': VSTAR_REF,
    'gap': gap,
    'verdict': verdict,
}
with open(f'{OUT_DIR}/results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"\n[Saved] {OUT_DIR}/results.json")
