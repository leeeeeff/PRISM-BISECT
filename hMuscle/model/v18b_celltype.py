#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v18b_celltype.py
----------------
v17f + cell-type expression features (8 or 16 dims).

Architecture:
  Stage 1 (T_ψ): unchanged — δ_layer(640) → T_ψ(64) via triplet learning
  Stage 2 MLP: [T_ψ(64) + φ_L30(640) + cell_frac(16)] = 720-dim input

Feature:
  celltype_features_{split}.npy — shape (N, 16)
    dims 0-7:  expr_frac: normalized expression fraction across 8 cell types
               "in which cell types is this isoform expressed?"
    dims 8-15: isoform_usage_frac: isoform / gene expression per cell type
               "what fraction of gene expression comes from this isoform?"

WHY these features help where RNA/LOC delta did not:
  RNA/LOC delta are derived from protein sequence features, which are partially
  redundant with δ_layer (ESM-2 already encodes much of this).
  Cell-type expression is ORTHOGONAL to sequence: it encodes where the isoform
  is deployed, not what it structurally is. ESM-2 has zero access to cell context.
  This is the missing signal for L4_CellState and L3_CellType GO terms.

Key design issue: train set (muscle, NM_) has zero cell-type expression in brain.
  Train: gene-level brain expression proxy (78.7% coverage; remaining zero-filled)
  Test:  isoform-level where available (94.4% coverage), gene-proxy otherwise.
  → The domain gap is EXPECTED. We test whether brain cell-type signal
    at inference time lifts L4/L3 GO terms even when train signal is imperfect.

V17f baseline: All MF 0.7198 / L2_Structural 0.6219
V18a fail:     All MF 0.7104 / L2_Structural 0.6065 (RNA+LOC redundant)
Expected gain: L4_CellState and L3_CellType from cell-context features
"""

import os, gzip, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
FEAT_DIR  = '../results_isoform/features'
OUT_DIR   = '../../reports/v18b_celltype'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS        = [42, 7, 13, 21, 99]
MARGIN       = 0.3
BATCH_T      = 512
EPOCHS_T     = 50
BATCH_MLP    = 512
EPOCHS_MLP   = 60
EMBED_DIM_T  = 64
MAX_GO_TRIPS = 30000
GO_TRIPS_PER = 300

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


print("=" * 65)
print("  v18b: v17f + Cell-Type Expression Features (16-dim)")
print("=" * 65)

# ── 1. Embeddings + cell-type features ───────────────────────
print("\n[1] Loading embeddings and features...")
X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X_l15_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer15_t30_150M.npy').astype(np.float32)
X_l30_te = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)
X_l15_te = np.load(f'{DATA_DIR}/esm2_layer_15_t30_150M.npy').astype(np.float32)

delta_tr = (X_l30_tr - X_l15_tr).astype(np.float32)
delta_te = (X_l30_te - X_l15_te).astype(np.float32)
scaler_d = MaxAbsScaler()
delta_tr_s = scaler_d.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler_d.transform(delta_te).astype(np.float32)

# Cell-type expression features (16-dim per isoform)
ct_tr = np.load(f'{FEAT_DIR}/celltype/celltype_features_train.npy').astype(np.float32)
ct_te = np.load(f'{FEAT_DIR}/celltype/celltype_features_test.npy').astype(np.float32)

# Separate scalers for expr_frac (0-7) and isoform_usage (8-15)
# Both are already in [0,1] by construction but MaxAbsScaler preserves that
scaler_ct = MaxAbsScaler()
ct_tr_s = scaler_ct.fit_transform(ct_tr).astype(np.float32)
ct_te_s = scaler_ct.transform(ct_te).astype(np.float32)

AUX_DIM = ct_tr.shape[1]  # 16

print(f"  Train: ESM-2 {X_l30_tr.shape} | CellType {ct_tr.shape}")
print(f"  Test:  ESM-2 {X_l30_te.shape} | CellType {ct_te.shape}")
print(f"  AUX_DIM: {AUX_DIM}")
# Report coverage
ct_te_expr = ct_te[:, :8]
ct_te_usage = ct_te[:, 8:]
print(f"  Test coverage: expr_frac nonzero={(ct_te_expr.sum(1)>0).mean()*100:.1f}%, "
      f"usage nonzero={(ct_te_usage.sum(1)>0).mean()*100:.1f}%")
ct_tr_expr = ct_tr[:, :8]
print(f"  Train coverage: expr_frac nonzero={(ct_tr_expr.sum(1)>0).mean()*100:.1f}%")

# ── 2. IDs and GO labels (same as v17f_layer_breakdown) ───────
tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
gene_arr_tr  = np.array(tr_genes)

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
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id: sym2id[syn] = p[1]

go_genes_tr  = defaultdict(set)
go_genes_all = defaultdict(set)
tr_ids       = [sym2id.get(g, g) for g in tr_genes]
tr_id_set    = set(tr_ids)

with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        gid, go_id, cat_raw = p[1], p[2], p[7]
        if cat_raw != 'Function': continue
        go_genes_all[go_id].add(gid)
        if gid in tr_id_set: go_genes_tr[go_id].add(gid)

te_ids = [sym2id.get(g, g) for g in te_sym_list]

# Load MF terms and PRISM brain reference (same source as v17f_layer_breakdown)
MF_REF_PATH = '../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv'
H2_CLASS_PATH = '../../reports/v_expanded_gomf/h2_layer_classification.tsv'

MF_TERMS = []; prism_ref = {}
with open(MF_REF_PATH) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 7:
            MF_TERMS.append(p[0])
            try: prism_ref[p[0]] = float(p[6])
            except: pass

h2_layer_map = {}
with open(H2_CLASS_PATH) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12: h2_layer_map[p[0]] = p[11]

def get_h2(go_id):
    lyr = h2_layer_map.get(go_id, 'Unknown')
    if lyr != 'Unknown': return lyr
    ap = prism_ref.get(go_id, 0)
    if ap < 0.60:   return 'L2_Structural*'
    elif ap < 0.75: return 'L1_Generic_mid'
    else:           return 'L1_Generic_high'

print(f"\n  {len(MF_TERMS)} MF terms | Triplets building...")

# ── 3. Triplets (same as v17f_layer_breakdown) ────────────────
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

Y_tr = np.stack([build_Y_tr(go) for go in MF_TERMS], axis=1)
Y_te = np.stack([build_Y_te(go) for go in MF_TERMS], axis=1)
valid_mask = Y_te.sum(0) >= 2

rng_trip = np.random.default_rng(42)
trip_a, trip_p, trip_n = [], [], []
for k, go_id in enumerate(MF_TERMS):
    y_k = Y_tr[:, k]
    pos_idxs = np.where(y_k == 1)[0]; neg_idxs = np.where(y_k == 0)[0]
    if len(pos_idxs) < 5 or len(neg_idxs) < 10: continue
    if len(trip_a) >= MAX_GO_TRIPS: break
    n_anchor = min(GO_TRIPS_PER, len(pos_idxs))
    for a_idx in rng_trip.choice(pos_idxs, n_anchor, replace=False):
        a_gene    = tr_genes[a_idx]
        cross_pos = pos_idxs[gene_arr_tr[pos_idxs] != a_gene]
        cross_neg = neg_idxs[gene_arr_tr[neg_idxs] != a_gene]
        if len(cross_pos) < 2 or len(cross_neg) < 2: continue
        trip_a.append(a_idx)
        trip_p.append(int(rng_trip.choice(cross_pos)))
        trip_n.append(int(rng_trip.choice(cross_neg)))

trip_a = np.array(trip_a, dtype=np.int32)
trip_p = np.array(trip_p, dtype=np.int32)
trip_n = np.array(trip_n, dtype=np.int32)
print(f"  {len(MF_TERMS)} MF terms | {len(trip_a)} triplets")

n_labels  = len(MF_TERMS)
valid_te  = np.where(valid_mask)[0]

# ── 4. H2 layer mapping ──────────────────────────────────────
LAYER_ORDER = ['L2_Structural', 'L2_Structural*', 'L4_CellState', 'L3_CellType',
               'L1_Generic_mid', 'L1_Generic_high']
layer2idxs = defaultdict(list)
for i, go in enumerate(MF_TERMS):
    if valid_mask[i]:
        layer2idxs[get_h2(go)].append(i)

# ── 5. TensorFlow model ───────────────────────────────────────
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
print(f"\n  GPU: {gpus[0].name if gpus else 'CPU'}")

from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks

def build_triplet_net(input_dim, embed_dim):
    inp = keras.Input(shape=(input_dim,))
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(embed_dim, activation=None)(x)
    x   = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    return models.Model(inp, x)

def build_mlp(t_dim, esm_dim, aux_dim, n_labels):
    inp_t   = keras.Input(shape=(t_dim,))
    inp_esm = keras.Input(shape=(esm_dim,))
    inp_aux = keras.Input(shape=(aux_dim,))
    x = layers.Concatenate()([inp_t, inp_esm, inp_aux])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n_labels, activation='sigmoid')(x)
    return models.Model([inp_t, inp_esm, inp_aux], out)

def triplet_loss_fn(emb, T, margin=MARGIN):
    a_e = tf.gather(emb, T[:,0])
    p_e = tf.gather(emb, T[:,1])
    n_e = tf.gather(emb, T[:,2])
    d_ap = tf.reduce_sum(tf.square(a_e - p_e), axis=1)
    d_an = tf.reduce_sum(tf.square(a_e - n_e), axis=1)
    loss = tf.reduce_mean(tf.maximum(d_ap - d_an + margin, 0.0))
    active = tf.reduce_mean(tf.cast(d_ap - d_an + margin > 0, tf.float32))
    return loss, active

# GO labels for MLP training (same as v17f_layer_breakdown)
n_labels = len(MF_TERMS)
valid_te = np.where(valid_mask)[0]
print(f"  {n_labels} GO terms | valid test: {len(valid_te)}")

# ── 6. Training loop ─────────────────────────────────────────
print(f"\n[Stage 1] T_ψ triplet training (δ_layer, same as v17f)...")

from tensorflow.keras.losses import BinaryFocalCrossentropy

def build_T_psi():
    inp = layers.Input(shape=(640,))
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(EMBED_DIM_T, activation='relu')(x)
    out = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    return models.Model(inp, out)

def triplet_loss_cosine(embs, a, p, n):
    ea = tf.gather(embs, a); ep = tf.gather(embs, p); en = tf.gather(embs, n)
    return tf.reduce_mean(tf.maximum(
        (1 - tf.reduce_sum(ea*ep, 1)) - (1 - tf.reduce_sum(ea*en, 1)) + MARGIN, 0))

tf.random.set_seed(42)
T_psi   = build_T_psi()
opt_T   = optimizers.Adam(1e-3)
dlt_tf  = tf.constant(delta_tr_s, dtype=tf.float32)
n_trip  = len(trip_a)
n_batch = max(1, n_trip // BATCH_T)
final_active = 0.0
for epoch in range(EPOCHS_T):
    perm = np.random.permutation(n_trip); el = 0.0
    for b in range(n_batch):
        bi = perm[b*BATCH_T:(b+1)*BATCH_T]
        with tf.GradientTape() as tape:
            embs = T_psi(dlt_tf, training=True)
            loss = triplet_loss_cosine(embs, trip_a[bi], trip_p[bi], trip_n[bi])
        grads = tape.gradient(loss, T_psi.trainable_variables)
        opt_T.apply_gradients(zip(grads, T_psi.trainable_variables))
        el += float(loss)
    if (epoch+1) % 10 == 0:
        embs_np = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
        ea = embs_np[trip_a]; ep2 = embs_np[trip_p]; en2 = embs_np[trip_n]
        final_active = ((1-(ea*ep2).sum(1))-(1-(ea*en2).sum(1))+MARGIN > 0).mean()
        print(f"  Epoch {epoch+1:3d} | loss={el/n_batch:.4f} | active={final_active:.2%}")

t_embed_tr = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
t_embed_te = T_psi.predict(delta_te_s, batch_size=1024, verbose=0)
print(f"  T_ψ done.")

print(f"\n[Stage 2] MLP ensemble with cell-type features...")
t0 = time.time()
focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
all_preds = []
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm   = np.random.permutation(len(t_embed_tr))
    n_val  = int(len(t_embed_tr) * 0.1)
    val_i  = perm[:n_val]; tr_i = perm[n_val:]
    mlp = build_mlp(EMBED_DIM_T, 640, AUX_DIM, n_labels)
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
    mlp.fit(
        [t_embed_tr[tr_i], X_l30_tr[tr_i], ct_tr_s[tr_i]], Y_tr[tr_i],
        validation_data=([t_embed_tr[val_i], X_l30_tr[val_i], ct_tr_s[val_i]], Y_tr[val_i]),
        epochs=EPOCHS_MLP, batch_size=BATCH_MLP, verbose=0,
        callbacks=[callbacks.ReduceLROnPlateau(patience=10, factor=0.5, verbose=0),
                   callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
    )
    preds = mlp.predict([t_embed_te, X_l30_te, ct_te_s], batch_size=1024, verbose=0)
    all_preds.append(preds)
    print(f"  Seed {seed} done.")

preds_ens = np.mean(all_preds, axis=0)
np.save(f'{OUT_DIR}/v18b_preds_ensemble.npy', preds_ens)

# ── 7. Evaluation ─────────────────────────────────────────────
def compute_auprc(preds, Y, idx_list):
    scores = []
    for i in idx_list:
        if Y[:, i].sum() < 1: continue
        scores.append(average_precision_score(Y[:, i], preds[:, i]))
    return np.mean(scores) if scores else 0.0

# v17f reference
v17f_path = '../../reports/v17f_bootstrap/v17f_preds_ensemble.npy'
v17f_preds = np.load(v17f_path) if os.path.exists(v17f_path) else None

all_mf_v18b = compute_auprc(preds_ens, Y_te, valid_te)
print(f"\n  v18b: All MF={all_mf_v18b:.4f}  [{time.time()-t0:.0f}s]")

import csv
print("\n" + "=" * 72)
print("  v18b Per-H2 Breakdown  [v17f + Cell-Type Expression (16-dim)]")
print("=" * 72)
sep = "  " + "-" * 70
print(f"  {'Layer':25s}  {'n':>4}  {'PRISM':>6}  {'v17f':>6}  {'v18b':>6}  {'Δ(v18-v17)':>10}")
print(sep)

rows = []
LAYER_ORDER = ['L2_Structural', 'L2_Structural*', 'L4_CellState', 'L3_CellType',
               'L1_Generic_mid', 'L1_Generic_high']
for layer in LAYER_ORDER:
    idxs = layer2idxs.get(layer, [])
    if not idxs: continue
    n = len(idxs)
    v18b_val = compute_auprc(preds_ens, Y_te, idxs)
    v17f_val = compute_auprc(v17f_preds, Y_te, idxs) if v17f_preds is not None else 0.0
    prism_val = np.mean([prism_ref.get(MF_TERMS[i], 0) for i in idxs])
    delta = v18b_val - v17f_val
    print(f"  {layer:25s}  {n:>4}  {prism_val:.4f}  {v17f_val:.4f}  {v18b_val:.4f}  {delta:+.4f}")
    rows.append({'layer': layer, 'n': n, 'prism': prism_val, 'v17f': v17f_val,
                 'v18b': v18b_val, 'delta': delta})

all_v17f = compute_auprc(v17f_preds, Y_te, valid_te) if v17f_preds is not None else 0.7198
all_prism = np.mean([prism_ref.get(MF_TERMS[i], 0) for i in valid_te])
print(sep)
print(f"  {'ALL MF':25s}  {len(valid_te):>4}  {all_prism:.4f}  {all_v17f:.4f}  {all_mf_v18b:.4f}  {all_mf_v18b-all_v17f:+.4f}")
print("=" * 72)

print(f"\n  Summary comparison:")
print(f"    PRISM v15d:  All MF=0.5249  L2_Struct=0.3127")
print(f"    v17f:        All MF=0.7198  L2_Struct=0.6219")
print(f"    v18a (RNA+LOC):  All MF=0.7104  (FAIL: -0.0094 vs v17f)")
print(f"    v18b (CellType): All MF={all_mf_v18b:.4f}")

breakdown_path_out = f'{OUT_DIR}/layer_breakdown.tsv'
with open(breakdown_path_out, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['layer','n','prism','v17f','v18b','delta'], delimiter='\t')
    writer.writeheader(); writer.writerows(rows)
print(f"\n  [Saved] {breakdown_path_out}")
print("  Done.")
