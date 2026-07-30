#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17g_hybrid_delta.py
--------------------
Hybrid delta: gene-mean for multi-isoform genes, layer-contrast for single-isoform.

Hypothesis: The two delta signals are complementary —
  δ_gene = φ(i) − μ_gene captures RELATIVE within-gene isoform variation
    Best for: genes with ≥2 isoforms where within-gene contrast is meaningful
    Weakness: single-isoform genes (26.9% train) → δ=0 → T_ψ uninformative

  δ_layer = L30(i) − L15(i) captures ABSOLUTE within-sequence depth progression
    Best for: all isoforms regardless of gene membership
    Advantage: v17f showed this is stronger for L2_Structural (+0.057 vs v17d)

Hybrid design:
  δ_hybrid[i] = δ_gene_scaled[i]  if gene has ≥2 isoforms in dataset
               δ_layer_scaled[i]  if gene has 1 isoform (single-iso)

Scaling: separate MaxAbsScaler per type to ensure comparable [-1,1] range.
All isoforms are eligible T_ψ anchors.

Test question: does hybrid δ beat pure δ_layer (v17f)?
  If yes → gene-mean adds complementary info beyond layer contrast for multi-iso genes
  If no  → δ_layer is sufficient; gene-mean adds noise above it
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
OUT_DIR   = '../../reports/v17g_hybrid'
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

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  v17g Hybrid Delta  (gene-mean + layer-contrast)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Load embeddings (both L30/L15 for layer delta, L30 for gene-mean)
# ─────────────────────────────────────────────────────────────────
print("\n[1] Loading ESM-2 embeddings...")
X_esm_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l15_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
X_esm_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l15_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
print(f"  ESM2 L{LAYER_B} train: {X_esm_tr.shape}  test: {X_esm_te.shape}")

# ─────────────────────────────────────────────────────────────────
# 2. Gene IDs
# ─────────────────────────────────────────────────────────────────
print("\n[2] Loading gene IDs...")
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

multi_mask_tr = np.array([len(gene2idxs_tr[g]) > 1 for g in tr_genes])
multi_mask_te = np.array([len(gene2idxs_te[g]) > 1 for g in te_sym_list])
n_multi_tr = multi_mask_tr.sum()
n_multi_te = multi_mask_te.sum()
print(f"  Train: {len(tr_genes)} isoforms, {len(gene2idxs_tr)} genes, "
      f"{n_multi_tr} multi-iso ({n_multi_tr/len(tr_genes):.1%})")
print(f"  Test:  {len(te_sym_list)} isoforms, {len(gene2idxs_te)} genes, "
      f"{n_multi_te} multi-iso ({n_multi_te/len(te_sym_list):.1%})")

# ─────────────────────────────────────────────────────────────────
# 3. Compute hybrid delta
# ─────────────────────────────────────────────────────────────────
print("\n[3] Computing δ_hybrid...")
print("  Multi-iso: δ_gene = φ(i) − μ_gene(i)")
print("  Single-iso: δ_layer = L30(i) − L15(i)")

# Gene-mean delta (zero for single-iso)
delta_gene_tr = np.zeros_like(X_esm_tr)
for gene, idxs in gene2idxs_tr.items():
    if len(idxs) > 1:
        mu = X_esm_tr[idxs].mean(axis=0)
        for i in idxs:
            delta_gene_tr[i] = X_esm_tr[i] - mu

delta_gene_te = np.zeros_like(X_esm_te)
for gene, idxs in gene2idxs_te.items():
    if len(idxs) > 1:
        mu = X_esm_te[idxs].mean(axis=0)
        for i in idxs:
            delta_gene_te[i] = X_esm_te[i] - mu

# Layer delta (all isoforms)
delta_layer_tr = (X_esm_tr - X_l15_tr).astype(np.float32)
delta_layer_te = (X_esm_te - X_l15_te).astype(np.float32)

# Scale each type independently, then combine
scaler_gene  = MaxAbsScaler()
scaler_layer = MaxAbsScaler()

# Fit gene scaler on multi-iso values only
delta_gene_multi = delta_gene_tr[multi_mask_tr]
scaler_gene.fit(delta_gene_multi)

# Fit layer scaler on all layer values
scaler_layer.fit(delta_layer_tr)

dg_tr_s = scaler_gene.transform(delta_gene_tr).astype(np.float32)
dg_te_s = scaler_gene.transform(delta_gene_te).astype(np.float32)
dl_tr_s = scaler_layer.transform(delta_layer_tr).astype(np.float32)
dl_te_s = scaler_layer.transform(delta_layer_te).astype(np.float32)

# Hybrid: gene-mean for multi-iso, layer-contrast for single-iso
delta_hybrid_tr = np.where(multi_mask_tr[:, None], dg_tr_s, dl_tr_s).astype(np.float32)
delta_hybrid_te = np.where(multi_mask_te[:, None], dg_te_s, dl_te_s).astype(np.float32)

n_gene_used_tr  = multi_mask_tr.sum()
n_layer_used_tr = (~multi_mask_tr).sum()
n_gene_used_te  = multi_mask_te.sum()
n_layer_used_te = (~multi_mask_te).sum()
print(f"  Train: {n_gene_used_tr} used δ_gene ({n_gene_used_tr/len(tr_genes):.1%}), "
      f"{n_layer_used_tr} used δ_layer ({n_layer_used_tr/len(tr_genes):.1%})")
print(f"  Test:  {n_gene_used_te} used δ_gene ({n_gene_used_te/len(te_sym_list):.1%}), "
      f"{n_layer_used_te} used δ_layer ({n_layer_used_te/len(te_sym_list):.1%})")
print(f"  All hybrid δ non-zero: {(np.linalg.norm(delta_hybrid_tr, axis=1) > 1e-6).all()}")

# ─────────────────────────────────────────────────────────────────
# 4. GO labels
# ─────────────────────────────────────────────────────────────────
print("\n[4] Loading GO labels (82 MF terms)...")
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
print(f"  {len(mf_terms)} MF terms | valid: {valid_mask.sum()}")

L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural':
            L2_TERMS.add(p[0])
l2_idx = [i for i, go in enumerate(mf_terms) if go in L2_TERMS]
l4_idx = [i for i, go in enumerate(mf_terms) if go not in L2_TERMS][:10]

# ─────────────────────────────────────────────────────────────────
# 5. Triplet mining (all isoforms eligible)
# ─────────────────────────────────────────────────────────────────
print("\n[5] Mining GO-label triplets (all isoforms eligible)...")
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
print(f"  GO-label triplets: {len(trip_a)}")

# single-iso anchor fraction
single_anc = (~multi_mask_tr[trip_a]).sum()
print(f"  Single-iso anchors (δ_layer): {single_anc}/{len(trip_a)} ({single_anc/len(trip_a):.1%})")

# ─────────────────────────────────────────────────────────────────
# 6. TensorFlow
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
    inp = layers.Input(shape=(delta_dim,), name='hybrid_delta_input')
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(embed_dim, activation='relu')(x)
    out = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    return models.Model(inp, out, name='T_psi_v17g')

def build_mlp_stage2(esm_dim=640, t_dim=64, n_go=82):
    inp_t   = layers.Input(shape=(t_dim,))
    inp_esm = layers.Input(shape=(esm_dim,))
    x = layers.Concatenate()([inp_t, inp_esm])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model([inp_t, inp_esm], out, name='MLP_v17g')

def triplet_loss_fn(embs, a, p, n, margin=0.3):
    ea = tf.gather(embs, a); ep = tf.gather(embs, p); en = tf.gather(embs, n)
    d_pos = 1.0 - tf.reduce_sum(ea * ep, axis=1)
    d_neg = 1.0 - tf.reduce_sum(ea * en, axis=1)
    return tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))

# ─────────────────────────────────────────────────────────────────
# 7. Stage 1: T_ψ on δ_hybrid
# ─────────────────────────────────────────────────────────────────
print(f"\n[6] Stage 1: T_ψ on δ_hybrid ({EPOCHS_T} epochs)...")
t0 = time.time()
tf.random.set_seed(42)
T_psi      = build_T_psi()
opt_T      = optimizers.Adam(1e-3)
delta_tf   = tf.constant(delta_hybrid_tr, dtype=tf.float32)
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
        embs_np = T_psi.predict(delta_hybrid_tr, batch_size=1024, verbose=0)
        ea = embs_np[trip_a]; ep2 = embs_np[trip_p]; en2 = embs_np[trip_n]
        active = ((1-(ea*ep2).sum(1)) - (1-(ea*en2).sum(1)) + MARGIN > 0).mean()
        print(f"  Epoch {epoch+1:3d} | loss={el/n_batches:.4f} | active={active:.2%}")

print(f"  T_ψ training: {time.time()-t0:.0f}s")
T_tr = T_psi.predict(delta_hybrid_tr, batch_size=1024, verbose=0)
T_te = T_psi.predict(delta_hybrid_te, batch_size=1024, verbose=0)
print(f"  T_tr norm: {np.linalg.norm(T_tr, axis=1).mean():.4f}  "
      f"T_te norm: {np.linalg.norm(T_te, axis=1).mean():.4f}")

# ─────────────────────────────────────────────────────────────────
# 8. Stage 2: MLP_θ
# ─────────────────────────────────────────────────────────────────
print(f"\n[7] Stage 2: MLP_θ ({EPOCHS_MLP} epochs, {len(SEEDS)} seeds)...")
t0       = time.time()
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
    pred = mlp.predict([T_te, X_esm_te], batch_size=1024, verbose=0)
    all_preds.append(pred)
    print(f"  seed={seed} done")

preds_ensemble = np.mean(all_preds, axis=0)
print(f"  Ensemble: {preds_ensemble.shape}  [{time.time()-t0:.0f}s]")

# ─────────────────────────────────────────────────────────────────
# 9. Evaluation
# ─────────────────────────────────────────────────────────────────
print(f"\n[8] Evaluation...")

def eval_subset(preds, Y, term_indices, label):
    vals = []
    for i in term_indices:
        y_t = Y[:, i]; y_p = preds[:, i]
        if y_t.sum() >= 2:
            vals.append(float(average_precision_score(y_t, y_p)))
    m = float(np.mean(vals)) if vals else float('nan')
    print(f"  {label}: macro AUPRC = {m:.4f}  (n={len(vals)} terms)")
    return m

valid_idx = [i for i in range(n_go) if valid_mask[i]]
l2_valid  = [i for i in l2_idx if valid_mask[i]]
l4_valid  = [i for i in l4_idx if valid_mask[i]]

macro_all = eval_subset(preds_ensemble, Y_te, valid_idx, f"All MF ({n_go} terms)      ")
macro_l2  = eval_subset(preds_ensemble, Y_te, l2_valid,  f"L2_Structural ({len(l2_valid)} MF)  ")
macro_l4  = eval_subset(preds_ensemble, Y_te, l4_valid,  f"L4_CellState sample {len(l4_valid)}")

PRISM_REF = 0.5962; PRISM_L2 = 0.3501
V17D_REF  = 0.6825; V17D_L2  = 0.5586
V17F_REF  = 0.7171; V17F_L2  = 0.6156

print(f"\n  {'='*72}")
print(f"  v17 Full Ablation Series (82 MF terms, brain zero-shot):")
print(f"  {'Method':<55} {'All MF':>7}  {'L2_Str':>7}")
print(f"  {'-'*72}")
print(f"  {'PRISM v15d (brain ref)':<55} {PRISM_REF:>7.4f}  {PRISM_L2:>7.4f}")
print(f"  {'v17d T_ψ(δ_gene)  [transductive, multi-iso only]':<55} {V17D_REF:>7.4f}  {V17D_L2:>7.4f}")
print(f"  {'v17f T_ψ(δ_layer) [inductive, all isoforms]':<55} {V17F_REF:>7.4f}  {V17F_L2:>7.4f}")
print(f"  {'v17g T_ψ(δ_hybrid)[gene-mean+layer fallback] [this]':<55} {macro_all:>7.4f}  {macro_l2:>7.4f}")
print(f"\n  Δ(v17g - PRISM): All MF {macro_all-PRISM_REF:+.4f}  L2 {macro_l2-PRISM_L2:+.4f}")
print(f"  Δ(v17g - v17d):  All MF {macro_all-V17D_REF:+.4f}  L2 {macro_l2-V17D_L2:+.4f}")
print(f"  Δ(v17g - v17f):  All MF {macro_all-V17F_REF:+.4f}  L2 {macro_l2-V17F_L2:+.4f}")
print(f"\n  Interpretation:")
if macro_l2 > V17F_L2 + 0.005:
    print(f"  → Hybrid BEATS v17f (+{macro_l2-V17F_L2:.4f} L2): gene-mean adds complementary signal")
elif abs(macro_l2 - V17F_L2) <= 0.005:
    print(f"  → Hybrid ≈ v17f (gap {macro_l2-V17F_L2:+.4f} L2): δ_layer alone is sufficient")
else:
    print(f"  → Hybrid BELOW v17f ({macro_l2-V17F_L2:+.4f} L2): mixed scaling disrupts T_ψ learning")

# ─────────────────────────────────────────────────────────────────
# 10. Per-type analysis: single-iso vs multi-iso performance
# ─────────────────────────────────────────────────────────────────
print(f"\n  Single-iso vs multi-iso test isoform breakdown:")
single_te_mask = ~multi_mask_te
multi_te_mask  = multi_mask_te
print(f"  Test: {single_te_mask.sum()} single-iso, {multi_te_mask.sum()} multi-iso")

# ─────────────────────────────────────────────────────────────────
# 11. Save
# ─────────────────────────────────────────────────────────────────
per_term = []
for i, go_id in enumerate(mf_terms):
    y_t = Y_te[:, i]; y_p = preds_ensemble[:, i]
    ap  = float(average_precision_score(y_t, y_p)) if y_t.sum() >= 2 else None
    per_term.append({
        'go_id':       go_id,
        'prism_brain': prism_brain_ref.get(go_id),
        'v17g_auprc':  ap,
        'delta_vs_prism': (ap - prism_brain_ref[go_id]) if ap and prism_brain_ref.get(go_id) else None,
        'delta_vs_v17f':  (ap - V17F_REF) if ap else None,
        'is_l2_structural': go_id in L2_TERMS,
    })

results = {
    'macro_v17g_all_mf':        macro_all,
    'macro_v17g_l2_structural': macro_l2,
    'macro_v17g_l4_sample':     macro_l4,
    'reference_series': {
        'prism': {'all_mf': PRISM_REF, 'l2': PRISM_L2},
        'v17d':  {'all_mf': V17D_REF,  'l2': V17D_L2},
        'v17f':  {'all_mf': V17F_REF,  'l2': V17F_L2},
    },
    'delta_v17g_vs_prism': {'all_mf': macro_all - PRISM_REF, 'l2': macro_l2 - PRISM_L2},
    'delta_v17g_vs_v17f':  {'all_mf': macro_all - V17F_REF,  'l2': macro_l2 - V17F_L2},
    'n_triplets': int(len(trip_a)),
    'architecture': {
        'delta_type':     'hybrid: gene-mean (multi-iso) + layer-contrast (single-iso)',
        'T_psi_dims':     '640→256→64(L2-norm)',
        'stage2_input':   'concat[T_psi(hybrid)(64), ESM-2-L30(640)] = 704',
        'scaler':         'separate MaxAbsScaler per delta type',
        'inductive':      True,
    },
    'per_term': per_term,
}
json.dump(results, open(f'{OUT_DIR}/v17g_results.json', 'w'), indent=2)
print(f"\n  [Saved] {OUT_DIR}/v17g_results.json")

with open(f'{OUT_DIR}/v17g_per_term.tsv', 'w') as f:
    f.write('go_id\tprism_brain\tv17g_auprc\tdelta_vs_prism\tdelta_vs_v17f\tis_l2_structural\n')
    for r in per_term:
        pb  = f"{r['prism_brain']:.4f}"      if r['prism_brain']    is not None else 'NA'
        v17g = f"{r['v17g_auprc']:.4f}"     if r['v17g_auprc']     is not None else 'NA'
        dp   = f"{r['delta_vs_prism']:.4f}" if r['delta_vs_prism'] is not None else 'NA'
        df   = f"{r['delta_vs_v17f']:.4f}"  if r['delta_vs_v17f']  is not None else 'NA'
        f.write(f"{r['go_id']}\t{pb}\t{v17g}\t{dp}\t{df}\t{r['is_l2_structural']}\n")
print(f"  [Saved] {OUT_DIR}/v17g_per_term.tsv")

print("\n" + "=" * 65)
print("  v17g Hybrid Delta — COMPLETE")
print("=" * 65)
