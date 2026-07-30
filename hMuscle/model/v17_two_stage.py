#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17_two_stage.py
----------------
Two-stage T architecture for isoform-level GO prediction.

Stage 1 — T_ψ (independent training):
  Input : ESM-2 [640] + Pfam binary [512] = [1152]
  Output: 256-dim structural embedding (L2-normalised)
  Loss  : cross-gene triplet (margin=0.3, cosine distance)
  Triplet: Pfam Jaccard similarity → cross-gene positive/negative
           NO within-gene pairs (anti-gene-bias, [R2.1])

Stage 2 — MLP_θ (T_ψ frozen):
  Input : T(i) [256, frozen]
  Output: n_go sigmoid predictions
  Loss  : BinaryFocalCrossentropy(γ=2) — identical to v15d_bp_clean

Key property: Stage 1 and Stage 2 gradients are fully decoupled.
Gene-level GO supervision cannot backpropagate into T_ψ.

Evaluation targets:
  - MF 82 terms overall (baseline: muscle PRISM 0.5962, brain 0.5912)
  - L2_Structural 33 terms (all MF binding; expected improvement)
  - L4_CellState sample 10 terms (expected no improvement — ceiling check)
"""

import os, sys, json, gzip, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/v17_two_stage'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13, 21, 99]
MARGIN     = 0.3
N_GO_EVAL  = 82     # MF terms
BATCH_T    = 256    # Stage 1 batch size
EPOCHS_T   = 30     # Stage 1 epochs
BATCH_MLP  = 512    # Stage 2 batch size
EPOCHS_MLP = 50     # Stage 2 epochs

# ─────────────────────────────────────────────────────────────────
# 0. Utility
# ─────────────────────────────────────────────────────────────────
def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  v17 Two-Stage T Architecture")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Load features
# ─────────────────────────────────────────────────────────────────
print("\n[1] Loading features...")
X_esm_tr = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_esm_te = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
X_dom_tr = np.load(f'{FEAT_DIR}/domain_matrix_proper_train.npy').astype(np.float32)
X_dom_te = np.load(f'{FEAT_DIR}/domain_matrix_proper_test.npy').astype(np.float32)
print(f"  ESM2 train: {X_esm_tr.shape}  test: {X_esm_te.shape}")
print(f"  Pfam train: {X_dom_tr.shape}  test: {X_dom_te.shape}")

# Concatenate for T_ψ input
X_T_tr = np.concatenate([X_esm_tr, X_dom_tr], axis=1)   # (31668, 1152)
X_T_te = np.concatenate([X_esm_te, X_dom_te], axis=1)   # (36748, 1152)
print(f"  T_ψ input train: {X_T_tr.shape}  test: {X_T_te.shape}")

# ─────────────────────────────────────────────────────────────────
# 2. Gene IDs — for cross-gene constraint
# ─────────────────────────────────────────────────────────────────
print("\n[2] Loading gene IDs...")
tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]

gene2idxs = defaultdict(list)
for i, g in enumerate(tr_genes):
    gene2idxs[g].append(i)
unique_genes = list(gene2idxs.keys())
print(f"  {len(tr_genes)} isoforms, {len(unique_genes)} unique genes")

# ─────────────────────────────────────────────────────────────────
# 3. Triplet mining (offline, Pfam Jaccard similarity)
# ─────────────────────────────────────────────────────────────────
print("\n[3] Mining cross-gene triplets (Pfam Jaccard)...")
dom_bool = X_dom_tr > 0   # (31668, 512) bool

# Only anchor isoforms WITH at least 1 domain
anchors = np.where(dom_bool.any(1))[0]
print(f"  Anchors with domains: {len(anchors)}")

# For efficiency: compute block-wise Pfam cosine sim (binary → dot / union)
# We'll build triplets by sampling, not exhaustive pairwise
np.random.seed(42)
MAX_TRIPLETS = 60000
triplets_a, triplets_p, triplets_n = [], [], []

# Build gene-level lookup
gene_arr = np.array(tr_genes)          # string array
isoform_has_domain = dom_bool.any(1)   # (31668,) bool

def pfam_jaccard(v1, v2):
    intersection = (v1 & v2).sum()
    union = (v1 | v2).sum()
    return float(intersection) / float(union + 1e-9)

# Sample strategy: for each anchor, find best cross-gene pos and neg
rng = np.random.default_rng(42)
anchor_sample = rng.choice(anchors, min(len(anchors), MAX_TRIPLETS // 10), replace=False)

for a_idx in anchor_sample:
    a_gene = tr_genes[a_idx]
    a_dom  = dom_bool[a_idx]

    # Cross-gene pool: isoforms from different genes WITH domains
    pool_mask = (gene_arr != a_gene) & isoform_has_domain
    pool_idxs = np.where(pool_mask)[0]
    if len(pool_idxs) < 10:
        continue

    # Sample 100 candidates
    cands = rng.choice(pool_idxs, min(100, len(pool_idxs)), replace=False)
    sims  = np.array([pfam_jaccard(a_dom, dom_bool[c]) for c in cands])

    pos_mask = sims > 0.05   # share at least one Pfam domain
    neg_mask = sims == 0.0   # no shared Pfam domain

    if not pos_mask.any() or not neg_mask.any():
        continue

    # Hard-positive (highest sim), easy-negative (sim=0)
    p_idx = cands[np.argmax(sims * pos_mask + (-np.inf) * ~pos_mask)]
    n_idx = cands[np.random.choice(np.where(neg_mask)[0])]

    triplets_a.append(a_idx)
    triplets_p.append(p_idx)
    triplets_n.append(n_idx)

triplets_a = np.array(triplets_a)
triplets_p = np.array(triplets_p)
triplets_n = np.array(triplets_n)
print(f"  Generated {len(triplets_a)} triplets")
if len(triplets_a) < 500:
    print("  WARNING: very few triplets — Pfam coverage may be too sparse")
    print("  Falling back to ESM-2 cosine similarity for triplets...")
    # ESM-2 based triplets for isoforms without sufficient domain overlap
    anchors_all = np.arange(len(tr_genes))
    rng2 = np.random.default_rng(123)
    for a_idx in rng2.choice(anchors_all, min(6000, len(anchors_all)), replace=False):
        a_gene = tr_genes[a_idx]
        pool_mask = (gene_arr != a_gene)
        pool_idxs = np.where(pool_mask)[0]
        cands = rng2.choice(pool_idxs, min(50, len(pool_idxs)), replace=False)
        a_esm = X_esm_tr[a_idx] / (np.linalg.norm(X_esm_tr[a_idx]) + 1e-9)
        c_esm = X_esm_tr[cands]
        c_norms = np.linalg.norm(c_esm, axis=1, keepdims=True) + 1e-9
        sims = c_esm / c_norms @ a_esm
        if len(sims) < 4: continue
        p_idx = cands[np.argmax(sims)]
        n_idx = cands[np.argmin(sims)]
        triplets_a = np.append(triplets_a, a_idx)
        triplets_p = np.append(triplets_p, p_idx)
        triplets_n = np.append(triplets_n, n_idx)
    print(f"  After ESM-2 fallback: {len(triplets_a)} triplets")

print(f"  Final triplets: {len(triplets_a)}")

# ─────────────────────────────────────────────────────────────────
# 4. GO labels (MF 82 terms, same as brain_domain_lr.py)
# ─────────────────────────────────────────────────────────────────
print("\n[4] Building GO labels (MF 82 terms)...")

# sym2id
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

# gene2go
go_genes_tr  = defaultdict(set)
go_genes_all = defaultdict(set)
tr_ids = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_ids)

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

# MF 82 terms (from mf_domain_vs_prism.tsv)
mf_terms, prism_muscle_ref, prism_brain_ref = [], {}, {}
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) < 6: continue
        mf_terms.append(p[0])
        prism_muscle_ref[p[0]] = float(p[4]) if p[4] else None   # prism_muscle col
        prism_brain_ref[p[0]]  = float(p[5]) if p[5] else None   # prism_brain col
print(f"  {len(mf_terms)} MF terms")

# Test gene mapping
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in te_genes_raw]

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

def build_Y_te_brain(go_id):
    pos_ids = go_genes_all[go_id]
    return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0
                     for s in te_sym_list], dtype=np.float32)

# Build label matrices
Y_tr = np.stack([build_Y_tr(go)         for go in mf_terms], axis=1)   # (31668, 82)
Y_te = np.stack([build_Y_te_brain(go)   for go in mf_terms], axis=1)   # (36748, 82)
valid_mask = (Y_te.sum(0) >= 2)                                          # ≥2 positives in test
print(f"  Y_tr: {Y_tr.shape}  Y_te: {Y_te.shape}  valid terms: {valid_mask.sum()}")

# L2_Structural term indices
L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural':
            L2_TERMS.add(p[0])
l2_idx = [i for i, go in enumerate(mf_terms) if go in L2_TERMS]
l4_idx = [i for i, go in enumerate(mf_terms) if go not in L2_TERMS][:10]
print(f"  L2_Structural indices: {len(l2_idx)}  L4_sample: {len(l4_idx)}")

# ─────────────────────────────────────────────────────────────────
# 5. TensorFlow setup
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
# 6. Build T_ψ model
# ─────────────────────────────────────────────────────────────────
def build_T_psi(input_dim=1152, embed_dim=256):
    inp = layers.Input(shape=(input_dim,))
    x   = layers.Dense(512, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(embed_dim, activation='relu')(x)
    out = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    return models.Model(inputs=inp, outputs=out, name='T_psi')

def triplet_loss_fn(embeddings, a_idx, p_idx, n_idx, margin=0.3):
    ea = tf.gather(embeddings, a_idx)
    ep = tf.gather(embeddings, p_idx)
    en = tf.gather(embeddings, n_idx)
    d_pos = 1.0 - tf.reduce_sum(ea * ep, axis=1)
    d_neg = 1.0 - tf.reduce_sum(ea * en, axis=1)
    loss  = tf.maximum(d_pos - d_neg + margin, 0.0)
    return tf.reduce_mean(loss)

def build_mlp(n_go=82, embed_dim=256):
    inp = layers.Input(shape=(embed_dim,))
    x   = layers.Dense(128, activation='relu')(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model(inputs=inp, outputs=out, name='MLP_theta')

# ─────────────────────────────────────────────────────────────────
# 7. Train T_ψ (Stage 1) — one set of weights shared across seeds
# ─────────────────────────────────────────────────────────────────
print(f"\n[5] Stage 1: Training T_ψ  ({EPOCHS_T} epochs, {BATCH_T} batch)...")
t0 = time.time()
tf.random.set_seed(42)
T_psi = build_T_psi()
opt_T = optimizers.Adam(1e-3)

# Triplet dataset
t_a = tf.constant(triplets_a, dtype=tf.int32)
t_p = tf.constant(triplets_p, dtype=tf.int32)
t_n = tf.constant(triplets_n, dtype=tf.int32)
X_T_tr_tf = tf.constant(X_T_tr, dtype=tf.float32)

n_triplets = len(triplets_a)
n_batches  = max(1, n_triplets // BATCH_T)

for epoch in range(EPOCHS_T):
    perm = np.random.permutation(n_triplets)
    epoch_loss = 0.0
    for b in range(n_batches):
        batch_idx = perm[b * BATCH_T: (b + 1) * BATCH_T]
        ba = t_a.numpy()[batch_idx]
        bp = t_p.numpy()[batch_idx]
        bn = t_n.numpy()[batch_idx]
        with tf.GradientTape() as tape:
            embs = T_psi(X_T_tr_tf, training=True)
            loss = triplet_loss_fn(embs, ba, bp, bn, MARGIN)
        grads = tape.gradient(loss, T_psi.trainable_variables)
        opt_T.apply_gradients(zip(grads, T_psi.trainable_variables))
        epoch_loss += float(loss)
    if (epoch + 1) % 5 == 0:
        active_frac = 0.0
        embs_np = T_psi.predict(X_T_tr, batch_size=1024, verbose=0)
        ea = embs_np[t_a.numpy()]
        ep = embs_np[t_p.numpy()]
        en = embs_np[t_n.numpy()]
        d_pos = 1 - (ea * ep).sum(1)
        d_neg = 1 - (ea * en).sum(1)
        active_frac = ((d_pos - d_neg + MARGIN) > 0).mean()
        print(f"  Epoch {epoch+1:3d} | loss={epoch_loss/n_batches:.4f} | active={active_frac:.2%}")

print(f"  T_ψ training: {time.time()-t0:.0f}s")

# Get T embeddings (frozen)
T_tr = T_psi.predict(X_T_tr, batch_size=1024, verbose=0)   # (31668, 256)
T_te = T_psi.predict(X_T_te, batch_size=1024, verbose=0)   # (36748, 256)
print(f"  T_tr: {T_tr.shape}  T_te: {T_te.shape}")

# ─────────────────────────────────────────────────────────────────
# 8. Stage 2: Train MLP_θ on frozen T (multi-seed ensemble)
# ─────────────────────────────────────────────────────────────────
print(f"\n[6] Stage 2: MLP_θ on frozen T  ({EPOCHS_MLP} epochs, {len(SEEDS)} seeds)...")
t0 = time.time()

n_go     = len(mf_terms)
focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

val_frac = 0.1
idx_all  = np.arange(len(T_tr))

all_preds = []
for seed in SEEDS:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    perm    = np.random.permutation(len(T_tr))
    n_val   = int(len(T_tr) * val_frac)
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]

    mlp = build_mlp(n_go=n_go)
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)

    mlp.fit(
        T_tr[tr_idx], Y_tr[tr_idx],
        validation_data=(T_tr[val_idx], Y_tr[val_idx]),
        epochs=EPOCHS_MLP,
        batch_size=BATCH_MLP,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        ],
        verbose=0
    )
    pred = mlp.predict(T_te, batch_size=1024, verbose=0)
    all_preds.append(pred)
    print(f"  seed={seed} done")

preds_ensemble = np.mean(all_preds, axis=0)   # (36748, 82)
print(f"  Ensemble shape: {preds_ensemble.shape}  [{time.time()-t0:.0f}s]")

# ─────────────────────────────────────────────────────────────────
# 9. Evaluation
# ─────────────────────────────────────────────────────────────────
print(f"\n[7] Evaluation...")

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

macro_all, auprc_all   = eval_subset(preds_ensemble, Y_te, valid_idx, "All MF (82 terms)      ")
macro_l2,  auprc_l2    = eval_subset(preds_ensemble, Y_te, l2_valid,  "L2_Structural (33 MF)  ")
macro_l4,  auprc_l4    = eval_subset(preds_ensemble, Y_te, l4_valid,  "L4_CellState sample    ")

# Reference baseline (PRISM v15d)
ref_muscle = np.mean([v for v in prism_muscle_ref.values() if v is not None])
ref_brain  = np.mean([v for v in prism_brain_ref.values()  if v is not None])

print(f"\n  {'='*60}")
print(f"  Comparison (MF {len(valid_idx)} valid terms):")
print(f"  {'Method':<35} {'Macro AUPRC':>12}")
print(f"  {'-'*48}")
print(f"  {'PRISM v15d (muscle, ref)':<35} {ref_muscle:>12.4f}")
print(f"  {'PRISM v15d (brain zero-shot, ref)':<35} {ref_brain:>12.4f}")
print(f"  {'v17 T_ψ+MLP_θ (brain labels)':<35} {macro_all:>12.4f}")
print(f"  {'  └─ L2_Structural 33 terms':<35} {macro_l2:>12.4f}")
print(f"  {'  └─ L4_CellState sample 10':<35} {macro_l4:>12.4f}")
delta = macro_all - ref_brain
print(f"\n  Δ(v17 - PRISM brain): {delta:+.4f}")
print(f"  Expected: L2 improve, L4 unchanged")

# ─────────────────────────────────────────────────────────────────
# 10. Per-term results
# ─────────────────────────────────────────────────────────────────
per_term = []
for i, go_id in enumerate(mf_terms):
    y_true = Y_te[:, i]
    y_pred = preds_ensemble[:, i]
    n_pos  = int(y_true.sum())
    if n_pos < 2:
        auprc_v17 = None
    else:
        auprc_v17 = float(average_precision_score(y_true, y_pred))
    layer = 'L2_Structural' if go_id in L2_TERMS else 'other'
    per_term.append({
        'go_id': go_id,
        'layer': layer,
        'n_pos_te': n_pos,
        'v17_auprc': auprc_v17,
        'prism_brain': prism_brain_ref.get(go_id),
        'delta_v17_prism': (auprc_v17 - prism_brain_ref[go_id])
            if (auprc_v17 and prism_brain_ref.get(go_id)) else None
    })

# Save
out = {
    'macro_v17_all_mf': macro_all,
    'macro_v17_l2_structural': macro_l2,
    'macro_v17_l4_sample': macro_l4,
    'prism_muscle_ref': ref_muscle,
    'prism_brain_ref': ref_brain,
    'delta_v17_vs_prism_brain': delta,
    'n_triplets': int(len(triplets_a)),
    'architecture': {
        'stage1': 'T_psi: Dense(1152->512,ReLU)+BN+Drop(0.3)->Dense(256,ReLU)->L2norm',
        'stage1_loss': f'triplet margin={MARGIN} cosine, cross-gene only',
        'stage2': 'MLP_theta: Dense(256->128,ReLU)+Drop(0.2)->Dense(64,ReLU)->Dense(82,sigmoid)',
        'stage2_loss': 'BinaryFocalCrossentropy(gamma=2)',
        'seeds': SEEDS
    },
    'per_term': per_term
}
out_path = f'{OUT_DIR}/v17_results.json'
json.dump(out, open(out_path, 'w'), indent=2)
print(f"\n  [Saved] {out_path}")

# Per-term TSV
with open(f'{OUT_DIR}/v17_per_term.tsv', 'w') as f:
    f.write("go_id\tlayer\tn_pos_te\tv17_auprc\tprism_brain\tdelta\n")
    for r in per_term:
        v17 = f"{r['v17_auprc']:.4f}" if r['v17_auprc'] is not None else 'NA'
        pb  = f"{r['prism_brain']:.4f}" if r['prism_brain'] is not None else 'NA'
        d   = f"{r['delta_v17_prism']:.4f}" if r['delta_v17_prism'] is not None else 'NA'
        f.write(f"{r['go_id']}\t{r['layer']}\t{r['n_pos_te']}\t{v17}\t{pb}\t{d}\n")
print(f"  [Saved] {OUT_DIR}/v17_per_term.tsv")
print(f"\n{'='*65}")
print(f"  v17 Two-Stage T — COMPLETE")
print(f"{'='*65}")
