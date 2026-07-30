#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17c_two_stage.py
-----------------
Two-stage T architecture — v17b improvement to fix Stage 1 training signal.

v17b failure diagnosis:
  - ESM cosine threshold 0.70 too strict → only 15 fallback triplets from 80.9% non-domain pool
  - T_ψ active triplet ratio 0.32% at epoch 5 → immediate collapse
  - v17b AUPRC 0.602 came entirely from ESM-2 in Stage 2, not T_ψ signal

v17c changes (two targeted fixes):
  1. ESM cosine threshold relaxed: pos 0.70→0.45, neg 0.30→0.20
     Expected: 20,000+ ESM triplets (vs 15 in v17b)
  2. GO-label-based triplets added [3c]:
     Cross-gene pairs from same GO-positive training set → positive pair
     Cross-gene isoforms from non-positive set → negative pair
     Note: GO labels inform MINING only — T_ψ training loss is triplet loss, NOT GO
           supervision. Gradient decoupling principle preserved: T_ψ has no
           backpropagation path to gene-level GO labels.
     Expected: 5,000-10,000 additional cross-gene functional-similarity triplets

Architecture (same as v17b):
  Stage 1 — T_ψ (Pfam-only structural embedding):
    Pfam[512] → Dense(256,ReLU) → BN → Drop(0.3) → Dense(64,ReLU) → L2-norm → [64]
    Loss: triplet (margin=0.3, cosine) on hybrid triplets [3a]+[3b]+[3c]

  Stage 2 — MLP_θ (T_ψ frozen, ESM-2 bypass):
    concat[T_ψ(64), ESM-2(640)] = [704] → Dense(256,ReLU) → BN → Drop(0.2)
    → Dense(128,ReLU) → Dense(n_go, sigmoid)
    Loss: BinaryFocalCrossentropy(γ=2)

Success criterion:
  - Active triplet ratio > 5% sustained past epoch 10
  - macro AUPRC > 0.602 (v17b baseline) AND L2_Structural > 0.470 (v17b)
"""

import os, sys, json, gzip, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/v17c_two_stage'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS            = [42, 7, 13, 21, 99]
MARGIN           = 0.3
BATCH_T          = 512      # larger batch → more triplet diversity per step
EPOCHS_T         = 50       # more epochs — more triplets → more to learn
BATCH_MLP        = 512
EPOCHS_MLP       = 60
EMBED_DIM_T      = 64
MAX_PFAM_TRIPS   = 30000
MAX_ESM_TRIPS    = 40000
MAX_GO_TRIPS     = 20000
ESM_SIM_POS      = 0.45     # relaxed from v17b's 0.70
ESM_SIM_NEG      = 0.20     # relaxed from v17b's 0.30
GO_TRIPS_PER_TERM = 200     # max triplets per GO term

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  v17c Two-Stage T  (relaxed ESM threshold + GO-label triplets)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Load features
# ─────────────────────────────────────────────────────────────────
print("\n[1] Loading features...")
X_esm_tr = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_esm_te = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
X_dom_tr = np.load(f'{FEAT_DIR}/domain_matrix_proper_train.npy').astype(np.float32)
X_dom_te = np.load(f'{FEAT_DIR}/domain_matrix_proper_test.npy').astype(np.float32)
print(f"  ESM2  train: {X_esm_tr.shape}  test: {X_esm_te.shape}")
print(f"  Pfam  train: {X_dom_tr.shape}  test: {X_dom_te.shape}")

esm_norms_tr    = np.linalg.norm(X_esm_tr, axis=1, keepdims=True) + 1e-9
X_esm_tr_normed = X_esm_tr / esm_norms_tr

# ─────────────────────────────────────────────────────────────────
# 2. Gene IDs
# ─────────────────────────────────────────────────────────────────
print("\n[2] Loading gene IDs...")
tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
gene_arr     = np.array(tr_genes)

gene2idxs = defaultdict(list)
for i, g in enumerate(tr_genes):
    gene2idxs[g].append(i)
unique_genes = list(gene2idxs.keys())
print(f"  {len(tr_genes)} isoforms, {len(unique_genes)} unique genes")

# ─────────────────────────────────────────────────────────────────
# 3. GO labels — needed for [3c] GO-label triplets
# ─────────────────────────────────────────────────────────────────
print("\n[3] Loading GO labels for triplet mining...")

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

mf_terms, prism_muscle_ref, prism_brain_ref = [], {}, {}
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) < 6: continue
        mf_terms.append(p[0])
        prism_muscle_ref[p[0]] = float(p[4]) if p[4] else None
        prism_brain_ref[p[0]]  = float(p[5]) if p[5] else None
print(f"  {len(mf_terms)} MF terms loaded")

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

Y_tr_all = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)  # (31668, 82)
print(f"  Y_tr_all: {Y_tr_all.shape}")

# ─────────────────────────────────────────────────────────────────
# 4. Hybrid triplet mining: [4a] Pfam + [4b] ESM cosine + [4c] GO-label
# ─────────────────────────────────────────────────────────────────
print("\n[4] Hybrid triplet mining...")
dom_bool        = X_dom_tr > 0
isoform_has_dom = dom_bool.any(1)
n_dom           = isoform_has_dom.sum()
print(f"  Isoforms with domains: {n_dom} / {len(tr_genes)} ({n_dom/len(tr_genes):.1%})")

rng = np.random.default_rng(42)
trip_a, trip_p, trip_n = [], [], []

def pfam_jaccard(v1, v2):
    i = (v1 & v2).sum()
    u = (v1 | v2).sum()
    return float(i) / float(u + 1e-9)

# [4a] Pfam Jaccard triplets
print("  [4a] Pfam Jaccard triplets...")
anchors_dom = np.where(isoform_has_dom)[0]
anchor_sample_pfam = rng.choice(anchors_dom,
                                min(len(anchors_dom), MAX_PFAM_TRIPS // 6),
                                replace=False)
for a_idx in anchor_sample_pfam:
    a_gene = tr_genes[a_idx]
    a_dom  = dom_bool[a_idx]
    pool_mask = (gene_arr != a_gene) & isoform_has_dom
    pool_idxs = np.where(pool_mask)[0]
    if len(pool_idxs) < 10:
        continue
    cands = rng.choice(pool_idxs, min(150, len(pool_idxs)), replace=False)
    sims  = np.array([pfam_jaccard(a_dom, dom_bool[c]) for c in cands])
    pos_m = sims > 0.05
    neg_m = sims == 0.0
    if not pos_m.any() or not neg_m.any():
        continue
    p_idx = cands[np.argmax(sims * pos_m + (-1e9) * ~pos_m)]
    n_idx = rng.choice(cands[np.where(neg_m)[0]])
    trip_a.append(a_idx); trip_p.append(p_idx); trip_n.append(n_idx)
pfam_count = len(trip_a)
print(f"  Pfam Jaccard triplets: {pfam_count}")

# [4b] ESM-2 cosine fallback — relaxed thresholds (0.45 / 0.20)
print(f"  [4b] ESM-2 cosine triplets (pos>={ESM_SIM_POS}, neg<={ESM_SIM_NEG})...")
n_esm_anchors = min(len(tr_genes), MAX_ESM_TRIPS // 8)
anchors_esm   = rng.choice(np.arange(len(tr_genes)), n_esm_anchors, replace=False)

for a_idx in anchors_esm:
    if len(trip_a) - pfam_count >= MAX_ESM_TRIPS:
        break
    a_gene = tr_genes[a_idx]
    pool_mask = gene_arr != a_gene
    pool_idxs = np.where(pool_mask)[0]
    if len(pool_idxs) < 20:
        continue
    cands = rng.choice(pool_idxs, min(200, len(pool_idxs)), replace=False)
    a_esm = X_esm_tr_normed[a_idx]
    c_esm = X_esm_tr_normed[cands]
    sims  = c_esm @ a_esm

    pos_m = sims >= ESM_SIM_POS
    neg_m = sims <= ESM_SIM_NEG
    if not pos_m.any() or not neg_m.any():
        continue
    p_idx = cands[np.argmax(sims * pos_m + (-1e9) * ~pos_m)]
    n_idx = cands[np.argmin(sims * neg_m + (1e9) * ~neg_m)]
    trip_a.append(a_idx); trip_p.append(p_idx); trip_n.append(n_idx)

esm_count = len(trip_a) - pfam_count
print(f"  ESM-2 cosine triplets: {esm_count}")

# [4c] GO-label-based cross-gene triplets
# Anchor + same-GO-positive (different gene) = positive pair
# Anchor + GO-negative (different gene)       = negative pair
# T_ψ training remains triplet loss only — no GO supervision backprop
print("  [4c] GO-label cross-gene triplets...")
go_start = len(trip_a)

for k, go_id in enumerate(mf_terms):
    y_k     = Y_tr_all[:, k]
    pos_idxs = np.where(y_k == 1)[0]
    neg_idxs = np.where(y_k == 0)[0]
    if len(pos_idxs) < 10 or len(neg_idxs) < 10:
        continue
    if len(trip_a) - go_start >= MAX_GO_TRIPS:
        break

    n_anchor = min(GO_TRIPS_PER_TERM, len(pos_idxs))
    anchors_k = rng.choice(pos_idxs, n_anchor, replace=False)

    for a_idx in anchors_k:
        a_gene = tr_genes[a_idx]
        cross_pos = pos_idxs[gene_arr[pos_idxs] != a_gene]
        cross_neg = neg_idxs[gene_arr[neg_idxs] != a_gene]
        if len(cross_pos) < 2 or len(cross_neg) < 2:
            continue
        p_idx = rng.choice(cross_pos)
        n_idx = rng.choice(cross_neg)
        trip_a.append(a_idx); trip_p.append(p_idx); trip_n.append(n_idx)

go_count = len(trip_a) - pfam_count - esm_count
print(f"  GO-label triplets: {go_count}")
print(f"  Total triplets: {len(trip_a)}  (Pfam: {pfam_count}, ESM: {esm_count}, GO: {go_count})")

trip_a = np.array(trip_a, dtype=np.int32)
trip_p = np.array(trip_p, dtype=np.int32)
trip_n = np.array(trip_n, dtype=np.int32)

# ─────────────────────────────────────────────────────────────────
# 5. Test set GO labels
# ─────────────────────────────────────────────────────────────────
print("\n[5] Building test GO labels...")
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

def build_Y_te_brain(go_id):
    pos_ids = go_genes_all[go_id]
    return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0
                     for s in te_sym_list], dtype=np.float32)

Y_te = np.stack([build_Y_te_brain(go) for go in mf_terms], axis=1)
valid_mask = Y_te.sum(0) >= 2
print(f"  Y_te: {Y_te.shape}  valid terms: {valid_mask.sum()}")

L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural':
            L2_TERMS.add(p[0])
l2_idx = [i for i, go in enumerate(mf_terms) if go in L2_TERMS]
l4_idx = [i for i, go in enumerate(mf_terms) if go not in L2_TERMS][:10]
print(f"  L2_Structural: {len(l2_idx)}  L4_sample: {len(l4_idx)}")

# ─────────────────────────────────────────────────────────────────
# 6. TensorFlow setup
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
else:
    print("\n  Running on CPU")

# ─────────────────────────────────────────────────────────────────
# 7. Build models
# ─────────────────────────────────────────────────────────────────
def build_T_psi(pfam_dim=512, embed_dim=64):
    inp = layers.Input(shape=(pfam_dim,), name='pfam_input')
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(embed_dim, activation='relu')(x)
    out = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1), name='l2_norm')(x)
    return models.Model(inputs=inp, outputs=out, name='T_psi_v17c')

def build_mlp_stage2(esm_dim=640, t_dim=64, n_go=82):
    inp_t   = layers.Input(shape=(t_dim,),   name='t_psi_input')
    inp_esm = layers.Input(shape=(esm_dim,), name='esm2_input')
    x = layers.Concatenate(name='concat_t_esm')([inp_t, inp_esm])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model(inputs=[inp_t, inp_esm], outputs=out, name='MLP_theta_v17c')

def triplet_loss_fn(embeddings, a_idx, p_idx, n_idx, margin=0.3):
    ea = tf.gather(embeddings, a_idx)
    ep = tf.gather(embeddings, p_idx)
    en = tf.gather(embeddings, n_idx)
    d_pos = 1.0 - tf.reduce_sum(ea * ep, axis=1)
    d_neg = 1.0 - tf.reduce_sum(ea * en, axis=1)
    return tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))

# ─────────────────────────────────────────────────────────────────
# 8. Stage 1: Train T_ψ
# ─────────────────────────────────────────────────────────────────
print(f"\n[6] Stage 1: Training T_ψ ({EPOCHS_T} epochs, {BATCH_T} batch)...")
t0 = time.time()
tf.random.set_seed(42)
T_psi = build_T_psi(pfam_dim=X_dom_tr.shape[1], embed_dim=EMBED_DIM_T)
opt_T = optimizers.Adam(1e-3)

X_dom_tr_tf = tf.constant(X_dom_tr, dtype=tf.float32)
n_triplets  = len(trip_a)
n_batches   = max(1, n_triplets // BATCH_T)

for epoch in range(EPOCHS_T):
    perm       = np.random.permutation(n_triplets)
    epoch_loss = 0.0
    for b in range(n_batches):
        batch_idx = perm[b * BATCH_T: (b + 1) * BATCH_T]
        ba = trip_a[batch_idx]
        bp = trip_p[batch_idx]
        bn = trip_n[batch_idx]
        with tf.GradientTape() as tape:
            embs = T_psi(X_dom_tr_tf, training=True)
            loss = triplet_loss_fn(embs, ba, bp, bn, MARGIN)
        grads = tape.gradient(loss, T_psi.trainable_variables)
        opt_T.apply_gradients(zip(grads, T_psi.trainable_variables))
        epoch_loss += float(loss)

    if (epoch + 1) % 5 == 0:
        embs_np     = T_psi.predict(X_dom_tr, batch_size=1024, verbose=0)
        ea = embs_np[trip_a]; ep = embs_np[trip_p]; en = embs_np[trip_n]
        d_pos       = 1 - (ea * ep).sum(1)
        d_neg       = 1 - (ea * en).sum(1)
        active_frac = ((d_pos - d_neg + MARGIN) > 0).mean()
        print(f"  Epoch {epoch+1:3d} | loss={epoch_loss/n_batches:.4f} | active={active_frac:.2%}")

print(f"  T_ψ training: {time.time()-t0:.0f}s")
T_tr = T_psi.predict(X_dom_tr, batch_size=1024, verbose=0)
T_te = T_psi.predict(X_dom_te, batch_size=1024, verbose=0)
print(f"  T_tr: {T_tr.shape}  T_te: {T_te.shape}")

# ─────────────────────────────────────────────────────────────────
# 9. Stage 2: Train MLP_θ on concat[T_ψ (frozen), ESM-2]
# ─────────────────────────────────────────────────────────────────
print(f"\n[7] Stage 2: MLP_θ on concat[T_ψ, ESM-2] ({EPOCHS_MLP} epochs, {len(SEEDS)} seeds)...")
t0 = time.time()

n_go     = len(mf_terms)
focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

all_preds = []
for seed in SEEDS:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    perm    = np.random.permutation(len(T_tr))
    n_val   = int(len(T_tr) * 0.1)
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]

    mlp = build_mlp_stage2(esm_dim=X_esm_tr.shape[1], t_dim=EMBED_DIM_T, n_go=n_go)
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
    mlp.fit(
        [T_tr[tr_idx], X_esm_tr[tr_idx]], Y_tr_all[tr_idx],
        validation_data=([T_tr[val_idx], X_esm_tr[val_idx]], Y_tr_all[val_idx]),
        epochs=EPOCHS_MLP,
        batch_size=BATCH_MLP,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    pred = mlp.predict([T_te, X_esm_te], batch_size=1024, verbose=0)
    all_preds.append(pred)
    print(f"  seed={seed} done")

preds_ensemble = np.mean(all_preds, axis=0)
print(f"  Ensemble shape: {preds_ensemble.shape}  [{time.time()-t0:.0f}s]")

# ─────────────────────────────────────────────────────────────────
# 10. Evaluation
# ─────────────────────────────────────────────────────────────────
print(f"\n[8] Evaluation...")

def eval_subset(preds, Y, term_indices, label):
    auprc_list = []
    for i in term_indices:
        y_true = Y[:, i]
        y_pred = preds[:, i]
        if y_true.sum() < 2: continue
        auprc_list.append(float(average_precision_score(y_true, y_pred)))
    macro = float(np.mean(auprc_list)) if auprc_list else float('nan')
    print(f"  {label}: macro AUPRC = {macro:.4f}  (n={len(auprc_list)} terms)")
    return macro, auprc_list

valid_idx = [i for i in range(n_go) if valid_mask[i]]
l2_valid  = [i for i in l2_idx if valid_mask[i]]
l4_valid  = [i for i in l4_idx if valid_mask[i]]

macro_all, auprc_all = eval_subset(preds_ensemble, Y_te, valid_idx, f"All MF ({n_go} terms)      ")
macro_l2,  auprc_l2  = eval_subset(preds_ensemble, Y_te, l2_valid,  f"L2_Structural ({len(l2_valid)} MF)  ")
macro_l4,  auprc_l4  = eval_subset(preds_ensemble, Y_te, l4_valid,  f"L4_CellState sample {len(l4_valid)}")

ref_brain  = np.mean([v for v in prism_brain_ref.values()  if v is not None])
ref_muscle = np.mean([v for v in prism_muscle_ref.values() if v is not None])

print(f"\n  {'='*65}")
print(f"  Comparison (MF {len(valid_idx)} valid terms):")
print(f"  {'Method':<42} {'Macro AUPRC':>12}")
print(f"  {'-'*55}")
print(f"  {'PRISM v15d (muscle, ref)':<42} {ref_muscle:>12.4f}")
print(f"  {'PRISM v15d (brain zero-shot, ref)':<42} {ref_brain:>12.4f}")
print(f"  {'v17  T_ψ(ESM+Pfam)+MLP (FAILED)':<42} {'0.3543':>12}")
print(f"  {'v17b T_ψ(Pfam)+MLP(T+ESM) (ESM bypass)':<42} {'0.6022':>12}")
print(f"  {'v17c T_ψ(Pfam)+MLP(T+ESM) [this run]':<42} {macro_all:>12.4f}")
print(f"    {'└─ L2_Structural':<40} {macro_l2:>12.4f}  (v17b: 0.4699)")
print(f"    {'└─ L4_CellState sample':<40} {macro_l4:>12.4f}  (v17b: 0.7415)")
print(f"\n  Triplets: Pfam={pfam_count:,}, ESM={esm_count:,}, GO={go_count:,}, Total={len(trip_a):,}")
print(f"  Δ(v17c - PRISM brain): {macro_all-ref_brain:+.4f}")
print(f"  Δ(v17c - v17b):        {macro_all-0.6022:+.4f}")

# ─────────────────────────────────────────────────────────────────
# 11. Save
# ─────────────────────────────────────────────────────────────────
per_term = []
for i, go_id in enumerate(mf_terms):
    y_true = Y_te[:, i]
    y_pred = preds_ensemble[:, i]
    v17c_auprc = float(average_precision_score(y_true, y_pred)) if y_true.sum() >= 2 else None
    per_term.append({
        'go_id':              go_id,
        'prism_brain':        prism_brain_ref.get(go_id),
        'v17b_baseline':      None,   # reference
        'v17c_auprc':         v17c_auprc,
        'delta_v17c_prism':   (v17c_auprc - prism_brain_ref[go_id])
                              if v17c_auprc and prism_brain_ref.get(go_id) else None,
        'is_l2_structural':   go_id in L2_TERMS,
    })

results = {
    'macro_v17c_all_mf':        macro_all,
    'macro_v17c_l2_structural': macro_l2,
    'macro_v17c_l4_sample':     macro_l4,
    'prism_brain_ref':          ref_brain,
    'v17b_all_mf_ref':          0.6022,
    'delta_v17c_vs_prism':      macro_all - ref_brain,
    'delta_v17c_vs_v17b':       macro_all - 0.6022,
    'n_valid_terms':            len(valid_idx),
    'triplet_pfam':             int(pfam_count),
    'triplet_esm':              int(esm_count),
    'triplet_go':               int(go_count),
    'triplet_total':            int(len(trip_a)),
    'esm_sim_pos_threshold':    ESM_SIM_POS,
    'esm_sim_neg_threshold':    ESM_SIM_NEG,
    'per_term': per_term,
}
json.dump(results, open(f'{OUT_DIR}/v17c_results.json', 'w'), indent=2)
print(f"\n  [Saved] {OUT_DIR}/v17c_results.json")

with open(f'{OUT_DIR}/v17c_per_term.tsv', 'w') as f:
    f.write('go_id\tprism_brain\tv17c_auprc\tdelta_v17c_prism\tis_l2_structural\n')
    for r in per_term:
        pb    = f"{r['prism_brain']:.4f}"  if r['prism_brain']  is not None else 'NA'
        v17c  = f"{r['v17c_auprc']:.4f}"  if r['v17c_auprc']   is not None else 'NA'
        delta = f"{r['delta_v17c_prism']:.4f}" if r['delta_v17c_prism'] is not None else 'NA'
        f.write(f"{r['go_id']}\t{pb}\t{v17c}\t{delta}\t{r['is_l2_structural']}\n")
print(f"  [Saved] {OUT_DIR}/v17c_per_term.tsv")

print("\n" + "=" * 65)
print("  v17c Two-Stage T — COMPLETE")
print("=" * 65)
