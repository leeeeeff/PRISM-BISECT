#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_bootstrap_ci.py
--------------------
Gene-level bootstrap CI (B=1000) for v17f layer-contrast delta.
Identical training pipeline to v17f; CI computed over test gene resampling.
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
OUT_DIR   = '../../reports/v17f_bootstrap'
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
LAYER_A      = 15
LAYER_B      = 30
B_BOOTSTRAP  = 1000

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print(f"  v17f Bootstrap CI  (gene-level, B={B_BOOTSTRAP})")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Data
# ─────────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
X_esm_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l15_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
X_esm_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l15_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_A:02d}_t30_150M.npy').astype(np.float32)

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
gene_arr_tr  = np.array(tr_genes)

gene2idxs_tr = defaultdict(list)
for i, g in enumerate(tr_genes):
    gene2idxs_tr[g].append(i)

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]
gene_arr_te  = np.array(te_sym_list)

gene2idxs_te = defaultdict(list)
for i, g in enumerate(te_sym_list):
    gene2idxs_te[g].append(i)

te_unique_genes = list(gene2idxs_te.keys())
print(f"  Train: {len(tr_genes)} isoforms, {len(gene2idxs_tr)} genes")
print(f"  Test:  {len(te_sym_list)} isoforms, {len(gene2idxs_te)} genes ({len(te_unique_genes)} unique)")

# ─────────────────────────────────────────────────────────────────
# 2. Layer-contrast delta
# ─────────────────────────────────────────────────────────────────
print("\n[2] Computing δ_layer = L30 − L15...")
delta_tr = (X_esm_tr - X_l15_tr).astype(np.float32)
delta_te = (X_esm_te - X_l15_te).astype(np.float32)

scaler = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler.transform(delta_te).astype(np.float32)
print(f"  Train δ: mean_norm={np.linalg.norm(delta_tr, axis=1).mean():.2f}  zero=0")

# ─────────────────────────────────────────────────────────────────
# 3. GO labels
# ─────────────────────────────────────────────────────────────────
print("\n[3] Loading GO labels (82 MF terms)...")
sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id:
                        sym2id[syn] = p[1]

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
        if gid in tr_id_set:
            go_genes_tr[go_id].add(gid)

mf_terms, prism_brain_ref = [], {}
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) < 6: continue
        mf_terms.append(p[0])
        prism_brain_ref[p[0]] = float(p[5]) if p[5] else None

tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes):
    tr_sym2idx[g].append(i)

def build_Y_tr(go_id):
    pos_ids  = go_genes_tr[go_id]
    pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in pos_ids}
    y = np.zeros(len(tr_genes), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []):
            y[idx] = 1.0
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
        if len(p) >= 12 and p[11] == 'L2_Structural':
            L2_TERMS.add(p[0])

valid_idx = [i for i in range(len(mf_terms)) if valid_mask[i]]
l2_idx    = [i for i in range(len(mf_terms)) if mf_terms[i] in L2_TERMS]
l2_valid  = [i for i in l2_idx if valid_mask[i]]
print(f"  {len(mf_terms)} MF terms | valid: {valid_mask.sum()} | L2_Structural: {len(l2_valid)}")

# ─────────────────────────────────────────────────────────────────
# 4. Triplets
# ─────────────────────────────────────────────────────────────────
print("\n[4] Mining triplets...")
rng = np.random.default_rng(42)
trip_a, trip_p, trip_n = [], [], []

for k, go_id in enumerate(mf_terms):
    y_k      = Y_tr[:, k]
    pos_idxs = np.where(y_k == 1)[0]
    neg_idxs = np.where(y_k == 0)[0]
    if len(pos_idxs) < 5 or len(neg_idxs) < 10:
        continue
    if len(trip_a) >= MAX_GO_TRIPS:
        break
    n_anchor = min(GO_TRIPS_PER, len(pos_idxs))
    for a_idx in rng.choice(pos_idxs, n_anchor, replace=False):
        a_gene    = tr_genes[a_idx]
        cross_pos = pos_idxs[gene_arr_tr[pos_idxs] != a_gene]
        cross_neg = neg_idxs[gene_arr_tr[neg_idxs] != a_gene]
        if len(cross_pos) < 2 or len(cross_neg) < 2:
            continue
        trip_a.append(a_idx)
        trip_p.append(int(rng.choice(cross_pos)))
        trip_n.append(int(rng.choice(cross_neg)))

trip_a = np.array(trip_a, dtype=np.int32)
trip_p = np.array(trip_p, dtype=np.int32)
trip_n = np.array(trip_n, dtype=np.int32)
print(f"  {len(trip_a)} triplets")

# ─────────────────────────────────────────────────────────────────
# 5. TensorFlow + Train
# ─────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"\n  GPU: {gpus[0].name}")

def build_T_psi(delta_dim=640, embed_dim=64):
    inp = layers.Input(shape=(delta_dim,))
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(embed_dim, activation='relu')(x)
    out = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    return models.Model(inp, out, name='T_psi_v17f')

def build_mlp_stage2(esm_dim=640, t_dim=64, n_go=82):
    inp_t   = layers.Input(shape=(t_dim,))
    inp_esm = layers.Input(shape=(esm_dim,))
    x = layers.Concatenate()([inp_t, inp_esm])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model([inp_t, inp_esm], out, name='MLP_v17f')

def triplet_loss_fn(embs, a, p, n, margin=0.3):
    ea = tf.gather(embs, a); ep = tf.gather(embs, p); en = tf.gather(embs, n)
    d_pos = 1.0 - tf.reduce_sum(ea * ep, axis=1)
    d_neg = 1.0 - tf.reduce_sum(ea * en, axis=1)
    return tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))

# Stage 1: T_ψ
print(f"\n[5] Stage 1: T_ψ ({EPOCHS_T} epochs)...")
t0 = time.time()
tf.random.set_seed(42)
T_psi    = build_T_psi()
opt_T    = optimizers.Adam(1e-3)
delta_tf = tf.constant(delta_tr_s, dtype=tf.float32)
n_triplets = len(trip_a)
n_batches  = max(1, n_triplets // BATCH_T)

for epoch in range(EPOCHS_T):
    perm = np.random.permutation(n_triplets)
    el   = 0.0
    for b in range(n_batches):
        bi = perm[b*BATCH_T:(b+1)*BATCH_T]
        with tf.GradientTape() as tape:
            embs = T_psi(delta_tf, training=True)
            loss = triplet_loss_fn(embs, trip_a[bi], trip_p[bi], trip_n[bi], MARGIN)
        grads = tape.gradient(loss, T_psi.trainable_variables)
        opt_T.apply_gradients(zip(grads, T_psi.trainable_variables))
        el += float(loss)
    if (epoch + 1) % 10 == 0:
        embs_np = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
        ea = embs_np[trip_a]; ep2 = embs_np[trip_p]; en2 = embs_np[trip_n]
        active = ((1-(ea*ep2).sum(1)) - (1-(ea*en2).sum(1)) + MARGIN > 0).mean()
        print(f"  Epoch {epoch+1:3d} | active={active:.2%}")

print(f"  T_ψ: {time.time()-t0:.0f}s")
T_tr = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
T_te = T_psi.predict(delta_te_s, batch_size=1024, verbose=0)

# Stage 2: ensemble
print(f"\n[6] Stage 2: 5-seed ensemble ({EPOCHS_MLP} epochs)...")
t0 = time.time()
focal_fn  = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
n_go      = len(mf_terms)
all_preds = []
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm    = np.random.permutation(len(T_tr))
    n_val   = int(len(T_tr) * 0.1)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    mlp     = build_mlp_stage2(n_go=n_go)
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
    mlp.fit(
        [T_tr[tr_idx], X_esm_tr[tr_idx]], Y_tr[tr_idx],
        validation_data=([T_tr[val_idx], X_esm_tr[val_idx]], Y_tr[val_idx]),
        epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    all_preds.append(mlp.predict([T_te, X_esm_te], batch_size=1024, verbose=0))
    print(f"  seed={seed} done")

preds_v17f = np.mean(all_preds, axis=0)
print(f"  Ensemble: {preds_v17f.shape}  [{time.time()-t0:.0f}s]")
np.save(f'{OUT_DIR}/v17f_preds_ensemble.npy', preds_v17f)
np.save(f'{OUT_DIR}/v17f_Y_te.npy', Y_te)
print(f"  [Saved] predictions to {OUT_DIR}/")

# ─────────────────────────────────────────────────────────────────
# 7. Point estimates
# ─────────────────────────────────────────────────────────────────
def macro_auprc(preds, Y, idxs):
    vals = []
    for i in idxs:
        if Y[:, i].sum() >= 2:
            vals.append(average_precision_score(Y[:, i], preds[:, i]))
    return float(np.mean(vals)) if vals else float('nan'), len(vals)

pt_all, n_all = macro_auprc(preds_v17f, Y_te, valid_idx)
pt_l2,  n_l2  = macro_auprc(preds_v17f, Y_te, l2_valid)

print(f"\n[7] Point estimates (verify vs original):")
print(f"  v17f All MF:     {pt_all:.4f}  (original: 0.7171)")
print(f"  v17f L2_Struct:  {pt_l2:.4f}  (original: 0.6156)")
print(f"  PRISM All MF:    0.5962  PRISM L2: 0.3501")

# ─────────────────────────────────────────────────────────────────
# 8. Gene-level bootstrap CI
# ─────────────────────────────────────────────────────────────────
print(f"\n[8] Gene-level bootstrap CI (B={B_BOOTSTRAP})...")
t0   = time.time()
rng2 = np.random.default_rng(0)
gene_to_te_idx = {g: np.array(idxs) for g, idxs in gene2idxs_te.items()}
N_genes        = len(te_unique_genes)

boot_all = np.zeros(B_BOOTSTRAP)
boot_l2  = np.zeros(B_BOOTSTRAP)
valid_b  = {'all': 0, 'l2': 0}

for b in range(B_BOOTSTRAP):
    sampled_genes = rng2.choice(te_unique_genes, size=N_genes, replace=True)
    boot_idxs     = np.concatenate([gene_to_te_idx[g] for g in sampled_genes])
    Y_b    = Y_te[boot_idxs]
    P_b    = preds_v17f[boot_idxs]

    vals_all = []
    for i in valid_idx:
        if Y_b[:, i].sum() >= 2:
            vals_all.append(average_precision_score(Y_b[:, i], P_b[:, i]))
    if vals_all:
        boot_all[b] = np.mean(vals_all); valid_b['all'] += 1

    vals_l2 = []
    for i in l2_valid:
        if Y_b[:, i].sum() >= 2:
            vals_l2.append(average_precision_score(Y_b[:, i], P_b[:, i]))
    if vals_l2:
        boot_l2[b] = np.mean(vals_l2); valid_b['l2'] += 1

    if (b + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta     = elapsed / (b + 1) * (B_BOOTSTRAP - b - 1)
        print(f"  Bootstrap {b+1:4d}/{B_BOOTSTRAP}  "
              f"mean_all={boot_all[:b+1][boot_all[:b+1]>0].mean():.4f}  "
              f"mean_l2={boot_l2[:b+1][boot_l2[:b+1]>0].mean():.4f}  "
              f"ETA={eta:.0f}s")

print(f"\n  Bootstrap done: {time.time()-t0:.0f}s")
print(f"  Valid bootstrap iterations: all={valid_b['all']}  l2={valid_b['l2']}")

ci_all = np.percentile(boot_all[boot_all > 0], [2.5, 97.5])
ci_l2  = np.percentile(boot_l2[boot_l2   > 0], [2.5, 97.5])

PRISM_ALL = 0.5962; PRISM_L2 = 0.3501
V17D_ALL  = 0.6825; V17D_L2  = 0.5586

print(f"\n{'='*70}")
print(f"  Bootstrap CI Results (gene-level, B={B_BOOTSTRAP})")
print(f"{'='*70}")
print(f"\n  Metric                                          Point                95% CI")
print(f"  -------------------------------------------------------------------------")
print(f"  v17f All MF 82 terms AUPRC                     {pt_all:.4f}  [{ci_all[0]:.4f}, {ci_all[1]:.4f}]")
print(f"  v17f L2_Structural 33 terms AUPRC              {pt_l2:.4f}  [{ci_l2[0]:.4f}, {ci_l2[1]:.4f}]")
print(f"\n  Reference (fixed, single run, no CI):")
print(f"    PRISM All MF: {PRISM_ALL}    v17d All MF: {V17D_ALL}")
print(f"    PRISM L2:     {PRISM_L2}    v17d L2:     {V17D_L2}")
print(f"\n  Δ(v17f - PRISM) point estimates:")
print(f"    All MF: {pt_all-PRISM_ALL:+.4f}     L2: {pt_l2-PRISM_L2:+.4f}")
print(f"  Δ(v17f - v17d) point estimates:")
print(f"    All MF: {pt_all-V17D_ALL:+.4f}     L2: {pt_l2-V17D_L2:+.4f}")
print(f"\n  CI lower bound > PRISM?")
print(f"    All MF: {ci_all[0]:.4f} > {PRISM_ALL} → {'YES ✓' if ci_all[0] > PRISM_ALL else 'NO ✗'}")
print(f"    L2:     {ci_l2[0]:.4f} > {PRISM_L2}  → {'YES ✓' if ci_l2[0] > PRISM_L2 else 'NO ✗'}")
print(f"\n  CI lower bound > v17d?")
print(f"    All MF: {ci_all[0]:.4f} > {V17D_ALL} → {'YES ✓' if ci_all[0] > V17D_ALL else 'NO ✗'}")
print(f"    L2:     {ci_l2[0]:.4f} > {V17D_L2} → {'YES ✓' if ci_l2[0] > V17D_L2 else 'NO ✗'}")

results = {
    'point_all_mf': pt_all, 'point_l2': pt_l2,
    'ci_all_mf': list(ci_all), 'ci_l2': list(ci_l2),
    'prism_ref': {'all_mf': PRISM_ALL, 'l2': PRISM_L2},
    'v17d_ref':  {'all_mf': V17D_ALL,  'l2': V17D_L2},
    'ci_lower_beats_prism': {'all': bool(ci_all[0] > PRISM_ALL), 'l2': bool(ci_l2[0] > PRISM_L2)},
    'ci_lower_beats_v17d':  {'all': bool(ci_all[0] > V17D_ALL),  'l2': bool(ci_l2[0] > V17D_L2)},
    'B': B_BOOTSTRAP, 'valid_boot': valid_b,
}
json.dump(results, open(f'{OUT_DIR}/v17f_bootstrap_ci.json', 'w'), indent=2)
print(f"\n  [Saved] {OUT_DIR}/v17f_bootstrap_ci.json")

print("\n" + "=" * 65)
print("  v17f Bootstrap CI — COMPLETE")
print("=" * 65)
