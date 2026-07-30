#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_layer_breakdown.py
-----------------------
Per-H2-layer AUPRC breakdown for:
  B1: real δ_layer  (already have preds in v17f_bootstrap/)
  B2: random δ      (re-run, save per-term preds)
  PRISM brain ref   (from mf_domain_vs_prism.tsv)
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
OUT_DIR   = '../../reports/v17f_layer_breakdown'
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
print("  v17f Per-Layer Breakdown (real δ vs random δ)")
print("=" * 65)

# ── Load data ──────────────────────────────────────────────────
print("\n[1] Loading data...")
X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X_l15_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer15_t30_150M.npy').astype(np.float32)
X_l30_te = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)
X_l15_te = np.load(f'{DATA_DIR}/esm2_layer_15_t30_150M.npy').astype(np.float32)

delta_tr = (X_l30_tr - X_l15_tr).astype(np.float32)
delta_te = (X_l30_te - X_l15_te).astype(np.float32)
scaler   = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler.transform(delta_te).astype(np.float32)

# Random δ — matched norm
rng_rand  = np.random.default_rng(999)
rand_tr   = rng_rand.standard_normal(delta_tr_s.shape).astype(np.float32)
rand_te   = rng_rand.standard_normal(delta_te_s.shape).astype(np.float32)
norms_tr  = np.linalg.norm(delta_tr_s, axis=1, keepdims=True)
norms_te  = np.linalg.norm(delta_te_s, axis=1, keepdims=True)
rand_tr   = rand_tr / (np.linalg.norm(rand_tr, axis=1, keepdims=True) + 1e-8) * norms_tr
rand_te   = rand_te / (np.linalg.norm(rand_te, axis=1, keepdims=True) + 1e-8) * norms_te

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

mf_terms = []; prism_ref = {}
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 7:
            mf_terms.append(p[0])
            try: prism_ref[p[0]] = float(p[6])
            except: pass

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

rng_trip = np.random.default_rng(42)
trip_a, trip_p, trip_n = [], [], []
for k, go_id in enumerate(mf_terms):
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
        trip_a.append(a_idx); trip_p.append(int(rng_trip.choice(cross_pos))); trip_n.append(int(rng_trip.choice(cross_neg)))
trip_a = np.array(trip_a, dtype=np.int32)
trip_p = np.array(trip_p, dtype=np.int32)
trip_n = np.array(trip_n, dtype=np.int32)
print(f"  {len(mf_terms)} MF terms | {len(trip_a)} triplets")

# ── H2 layer mapping ────────────────────────────────────────────
h2_layer = {}
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12: h2_layer[p[0]] = p[11]

def get_h2(go_id):
    lyr = h2_layer.get(go_id, 'Unknown')
    if lyr != 'Unknown': return lyr
    ap = prism_ref.get(go_id, 0)
    if ap < 0.60:   return 'L2_Structural*'
    elif ap < 0.75: return 'L1_Generic_mid'
    else:           return 'L1_Generic_high'

layer2idxs = defaultdict(list)
for i, go in enumerate(mf_terms):
    if valid_mask[i]: layer2idxs[get_h2(go)].append(i)

LAYER_ORDER = ['L2_Structural', 'L2_Structural*', 'L4_CellState', 'L1_Generic_mid', 'L1_Generic_high']

# ── TF ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices(gpus[0], 'GPU')

def build_T_psi():
    inp = layers.Input(shape=(640,))
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(EMBED_DIM_T, activation='relu')(x)
    out = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    return models.Model(inp, out)

def build_mlp():
    inp_t   = layers.Input(shape=(EMBED_DIM_T,))
    inp_esm = layers.Input(shape=(640,))
    x = layers.Concatenate()([inp_t, inp_esm])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(len(mf_terms), activation='sigmoid')(x)
    return models.Model([inp_t, inp_esm], out)

def triplet_loss_fn(embs, a, p, n):
    ea = tf.gather(embs, a); ep = tf.gather(embs, p); en = tf.gather(embs, n)
    return tf.reduce_mean(tf.maximum(
        (1 - tf.reduce_sum(ea*ep, 1)) - (1 - tf.reduce_sum(ea*en, 1)) + MARGIN, 0))

def run_and_save(delta_tr_in, delta_te_in, label):
    tf.random.set_seed(42)
    T_psi   = build_T_psi()
    opt_T   = optimizers.Adam(1e-3)
    dlt_tf  = tf.constant(delta_tr_in, dtype=tf.float32)
    n_trip  = len(trip_a); n_batch = max(1, n_trip // BATCH_T)
    final_active = 0.0
    for epoch in range(EPOCHS_T):
        perm = np.random.permutation(n_trip); el = 0.0
        for b in range(n_batch):
            bi = perm[b*BATCH_T:(b+1)*BATCH_T]
            with tf.GradientTape() as tape:
                embs = T_psi(dlt_tf, training=True)
                loss = triplet_loss_fn(embs, trip_a[bi], trip_p[bi], trip_n[bi])
            grads = tape.gradient(loss, T_psi.trainable_variables)
            opt_T.apply_gradients(zip(grads, T_psi.trainable_variables))
            el += float(loss)
        if (epoch + 1) % 10 == 0:
            embs_np = T_psi.predict(delta_tr_in, batch_size=1024, verbose=0)
            ea = embs_np[trip_a]; ep2 = embs_np[trip_p]; en2 = embs_np[trip_n]
            final_active = ((1-(ea*ep2).sum(1))-(1-(ea*en2).sum(1))+MARGIN > 0).mean()
            print(f"  [{label}] Epoch {epoch+1:3d} | loss={el/n_batch:.4f} | active={final_active:.2%}")
    T_tr = T_psi.predict(delta_tr_in, batch_size=1024, verbose=0)
    T_te = T_psi.predict(delta_te_in, batch_size=1024, verbose=0)
    focal_fn  = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
    all_preds = []
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm    = np.random.permutation(len(T_tr))
        n_val   = int(len(T_tr) * 0.1); val_idx = perm[:n_val]; tr_idx = perm[n_val:]
        mlp     = build_mlp()
        mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
        mlp.fit(
            [T_tr[tr_idx], X_l30_tr[tr_idx]], Y_tr[tr_idx],
            validation_data=([T_tr[val_idx], X_l30_tr[val_idx]], Y_tr[val_idx]),
            epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0
        )
        all_preds.append(mlp.predict([T_te, X_l30_te], batch_size=1024, verbose=0))
    preds = np.mean(all_preds, axis=0)
    np.save(f'{OUT_DIR}/preds_{label}.npy', preds)
    tf.keras.backend.clear_session()
    return preds

# ── Run random-δ ────────────────────────────────────────────────
print("\n[B] Running random-δ model...")
preds_rand = run_and_save(rand_tr, rand_te, 'rand')

# ── Load real-δ preds (v17f bootstrap) ─────────────────────────
print("\n[A] Loading real δ_layer preds (v17f bootstrap)...")
preds_real = np.load('../../reports/v17f_bootstrap/v17f_preds_ensemble.npy')
print(f"  Loaded: {preds_real.shape}")

# ── Per-layer AUPRC ─────────────────────────────────────────────
print("\n\n" + "=" * 75)
print("  PER-H2-LAYER AUPRC BREAKDOWN")
print("=" * 75)
print(f"  {'Layer':<22} {'n':>4}  {'PRISM':>7}  {'real_δ':>7}  {'rand_δ':>7}  {'Δ_real':>7}  {'Δ_rand':>7}  {'δ_contrib':>9}")
print("  " + "-" * 70)

rows = []
for lyr in LAYER_ORDER:
    idxs = layer2idxs.get(lyr, [])
    if not idxs: continue
    prism_ap = np.mean([prism_ref[mf_terms[i]] for i in idxs if mf_terms[i] in prism_ref])
    real_ap  = np.mean([average_precision_score(Y_te[:,i], preds_real[:,i]) for i in idxs])
    rand_ap  = np.mean([average_precision_score(Y_te[:,i], preds_rand[:,i]) for i in idxs])
    d_real   = real_ap - prism_ap
    d_rand   = rand_ap - prism_ap
    contrib  = real_ap - rand_ap   # directional contribution of δ_layer
    rows.append((lyr, len(idxs), prism_ap, real_ap, rand_ap, d_real, d_rand, contrib))
    print(f"  {lyr:<22} {len(idxs):>4}  {prism_ap:>7.4f}  {real_ap:>7.4f}  {rand_ap:>7.4f}  {d_real:>+7.4f}  {d_rand:>+7.4f}  {contrib:>+9.4f}")

# All MF
all_valid = [i for i in range(len(mf_terms)) if valid_mask[i]]
prism_all = np.mean([prism_ref[mf_terms[i]] for i in all_valid if mf_terms[i] in prism_ref])
real_all  = np.mean([average_precision_score(Y_te[:,i], preds_real[:,i]) for i in all_valid])
rand_all  = np.mean([average_precision_score(Y_te[:,i], preds_rand[:,i]) for i in all_valid])
print("  " + "-" * 70)
print(f"  {'ALL MF':<22} {len(all_valid):>4}  {prism_all:>7.4f}  {real_all:>7.4f}  {rand_all:>7.4f}  {real_all-prism_all:>+7.4f}  {rand_all-prism_all:>+7.4f}  {real_all-rand_all:>+9.4f}")

print("\n  Column guide:")
print("  PRISM    = brain zero-shot baseline (v15d)")
print("  real_δ   = v17f (δ_layer = L30-L15)")
print("  rand_δ   = same T_ψ architecture, random direction δ (capacity baseline)")
print("  Δ_real   = real_δ - PRISM  (total improvement)")
print("  Δ_rand   = rand_δ - PRISM  (T_ψ capacity contribution)")
print("  δ_contrib= real_δ - rand_δ (directional content of δ_layer)")
print("=" * 75)

# Save TSV
with open(f'{OUT_DIR}/layer_breakdown.tsv', 'w') as f:
    f.write('layer\tn\tprism\treal_delta\trand_delta\tdelta_real\tdelta_rand\tdelta_contrib\n')
    for row in rows:
        f.write('\t'.join(str(x) for x in row) + '\n')
print(f"\n  [Saved] {OUT_DIR}/layer_breakdown.tsv")
print("  Done.")
