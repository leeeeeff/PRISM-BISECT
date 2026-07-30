#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_abl_no_tpsi.py
-------------------
Ablation: identical to v17f but WITHOUT T_ψ triplet training.
Stage 2 MLP receives concat[δ_layer(640), φ_L30(640)] = 1280-dim directly.

Isolates the T_ψ contribution within the exact v17f data setup (31,668 isoforms).
Compare to v17f=0.717/0.622 and v17e=0.639/0.487 (v17d-based ablation).
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
OUT_DIR   = '../../reports/v17f_abl_no_tpsi'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13, 21, 99]
BATCH_MLP  = 512
EPOCHS_MLP = 60
LAYER_A    = 15
LAYER_B    = 30

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  v17f Ablation: no T_ψ — concat[δ(640), L30(640)]")
print("=" * 65)

# ── 1. Data ───────────────────────────────────────────────────────
print("\n[1] Loading embeddings...")
X_tr_l30 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_tr_l15 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
X_te_l30 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_te_l15 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
print(f"  Train: {X_tr_l30.shape}  Test: {X_te_l30.shape}")

# ── 2. δ_layer ───────────────────────────────────────────────────
print("\n[2] Computing δ_layer = L30 − L15...")
delta_tr = (X_tr_l30 - X_tr_l15).astype(np.float32)
delta_te = (X_te_l30 - X_te_l15).astype(np.float32)
scaler   = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler.transform(delta_te).astype(np.float32)

# ── 3. IDs ────────────────────────────────────────────────────────
print("\n[3] Loading gene IDs...")
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
print(f"  Train: {len(tr_genes)}  Test: {len(te_sym_list)}")

# ── 4. GO labels ──────────────────────────────────────────────────
print("\n[4] Loading GO labels (82 MF)...")
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

tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

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

# ── 5. TF + Stage 2 MLP (no T_ψ) ─────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"\n  GPU: {gpus[0].name}")

def build_mlp_no_tpsi(delta_dim=640, esm_dim=640, n_go=82):
    inp_d = layers.Input(shape=(delta_dim,))
    inp_e = layers.Input(shape=(esm_dim,))
    x = layers.Concatenate()([inp_d, inp_e])      # 1280-dim
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model([inp_d, inp_e], out, name='MLP_no_tpsi')

focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
n_go     = len(mf_terms)

print(f"\n[5] 5-seed ensemble (no T_ψ, {EPOCHS_MLP} epochs)...")
t0 = time.time()
all_preds = []
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm    = np.random.permutation(len(delta_tr_s))
    n_val   = int(len(delta_tr_s) * 0.1)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    mlp = build_mlp_no_tpsi(n_go=n_go)
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
    mlp.fit(
        [delta_tr_s[tr_idx], X_tr_l30[tr_idx]], Y_tr[tr_idx],
        validation_data=([delta_tr_s[val_idx], X_tr_l30[val_idx]], Y_tr[val_idx]),
        epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    preds_i = mlp.predict([delta_te_s, X_te_l30], batch_size=1024, verbose=0)
    all_preds.append(preds_i)
    # quick per-seed eval
    aps = [average_precision_score(Y_te[:,i], preds_i[:,i])
           for i in valid_idx if Y_te[:,i].sum() >= 2]
    print(f"  seed={seed}  AUPRC={np.mean(aps):.4f}")

preds = np.mean(all_preds, axis=0)
print(f"  Ensemble done: {time.time()-t0:.0f}s")

# ── 6. Evaluate ───────────────────────────────────────────────────
def macro_auprc(preds, idxs):
    aps = [average_precision_score(Y_te[:,i], preds[:,i])
           for i in idxs if Y_te[:,i].sum() >= 2]
    return float(np.mean(aps)) if aps else float('nan')

auprc_all = macro_auprc(preds, valid_idx)
auprc_l2  = macro_auprc(preds, l2_valid)

V17F_ALL = 0.7173; V17F_L2 = 0.6219
V17E_ALL = 0.6389; V17E_L2 = 0.4865

print(f"\n{'='*65}")
print(f"  Ablation results (no T_ψ, v17f data 31,668 isoforms):")
print(f"  All MF AUPRC:     {auprc_all:.4f}  (v17f={V17F_ALL}  Δ={auprc_all-V17F_ALL:+.4f})")
print(f"  L2_Structural:    {auprc_l2:.4f}  (v17f={V17F_L2}  Δ={auprc_l2-V17F_L2:+.4f})")
print(f"\n  Reference ablation v17e (same no-T_ψ, v17d data 28,040):")
print(f"    All MF: {V17E_ALL}    L2: {V17E_L2}")
print(f"{'='*65}")

results = {
    'architecture': {
        'method': 'no_T_psi_v17f_data',
        'input': 'concat[delta_layer(640_scaled), ESM2_L30(640)] = 1280',
        'mlp': '1280→256(ReLU)→BN→Drop(0.2)→128(ReLU)→82(sigmoid)',
        'loss': 'BinaryFocalCrossentropy(gamma=2)',
        'triplet': 'none',
        'train_isoforms': X_tr_l30.shape[0],
        'seeds': SEEDS,
    },
    'auprc_all_mf': auprc_all,
    'auprc_l2_structural': auprc_l2,
    'v17f_ref': {'all_mf': V17F_ALL, 'l2': V17F_L2},
    'v17e_ref': {'all_mf': V17E_ALL, 'l2': V17E_L2},
    'delta_vs_v17f': {'all_mf': auprc_all - V17F_ALL, 'l2': auprc_l2 - V17F_L2},
    'T_psi_contribution_estimate': {
        'all_mf': V17F_ALL - auprc_all,
        'l2': V17F_L2 - auprc_l2,
        'note': 'v17f - this ablation = T_psi + data_size combined'
    },
}
with open(f'{OUT_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[Saved] {OUT_DIR}/results.json")
