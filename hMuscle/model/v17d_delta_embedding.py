#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17d_delta_embedding.py
-----------------------
Two-stage T architecture using ESM-2 delta embedding as T_ψ input.

Background — the mean-pooling information loss problem:
  ESM-2 produces per-residue embeddings, then mean-pools across all L residues.
  φ(i) = mean_pool(ESM-2(sequence_i))  ∈ ℝ^640

  This mean pool loses positional specificity: which exons are present/absent,
  where structural changes occur. The between-gene signal (gene identity)
  dominates φ(i), suppressing within-gene isoform-specific dimensions.

  splice_delta (v15d_splice) attempted to recover this:
    δ_i = φ(i) − mean_gene(φ)
  But it FAILED due to gradient direction conflict: GO labels are gene-level,
  so the model couldn't determine which isoform's δ should be "positive direction".
  (AUPRC collapsed from 0.7022 → 0.4573)

v17d design — the two-stage fix for gradient conflict:
  Stage 1: T_ψ(δ_i) learns structural embedding of within-gene differential signal
           Training loss: GO-label cross-gene triplet (from v17c design; 70% active)
           NO GO supervision in Stage 1 → gradient conflict impossible
  Stage 2: MLP_θ(concat[T_ψ(δ_i)(64), φ(i)(640)]) → GO predictions
           T_ψ is FROZEN → Stage 2 GO gradients cannot reach T_ψ

Why T_ψ(δ_i) captures mean-pooling lost information:
  - δ_i removes between-gene variance (gene identity signal)
  - δ_i amplifies within-gene variance (isoform-specific dimensions)
  - PRISM's pos_bias (within-gene variance 5×10⁻⁴) proves these dimensions exist
  - T_ψ(δ_i) organises this differential signal by functional similarity
  - Stage 2 sees both the absolute (φ) and differential (T_ψ(δ)) signal

Why this differs from v17c (T_ψ on Pfam binary):
  - Pfam binary is domain inventory → no positional information → subset of ESM-2
  - δ_i is the ESM-2 isoform-specific residual → amplifies what mean pool kept of
    the positional signal → genuinely complementary to φ(i) in Stage 2

Delta computation:
  Training:  δ_tr[i] = φ_tr[i] − mean(φ_tr[j] for j in same gene as i in train set)
  Test:      δ_te[i] = φ_te[i] − mean(φ_te[j] for j in same gene as i in test set)
  Note: test gene means are computed from test set only (no train-test leakage;
        gene-grouped CV ensures test genes are entirely unseen in training)
  Single-isoform genes: δ = 0 → T_ψ(0) ≈ zero contribution (expected behaviour)

Architecture:
  Stage 1 — T_ψ (delta structural embedding, Pfam-free):
    δ_i [640] → Dense(256,ReLU) → BN → Drop(0.3) → Dense(64,ReLU) → L2-norm → [64]
    Loss: GO-label cross-gene triplet (same-GO positive / non-positive negative)
    Triplets computed on δ_i space, cross-gene constraint enforced

  Stage 2 — MLP_θ (T_ψ frozen, ESM-2 bypass):
    concat[T_ψ(δ_i)(64), φ(i)(640)] = [704] → Dense(256,ReLU) → BN → Drop(0.2)
    → Dense(128,ReLU) → Dense(n_go, sigmoid)
    Loss: BinaryFocalCrossentropy(γ=2), 5-seed ensemble

Success criterion vs v17c baseline (0.600 overall, 0.472 L2_Structural):
  - Active triplet ratio > 5% → T_ψ genuinely learning
  - Overall AUPRC improvement relative to v17c
  - L2_Structural AUPRC improvement > +0.01 (vs v17c 0.472)
"""

import os, json, gzip, time
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
OUT_DIR   = '../../reports/v17d_delta'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS          = [42, 7, 13, 21, 99]
MARGIN         = 0.3
BATCH_T        = 512
EPOCHS_T       = 50
BATCH_MLP      = 512
EPOCHS_MLP     = 60
EMBED_DIM_T    = 64
MAX_GO_TRIPS   = 30000
GO_TRIPS_PER   = 300

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  v17d Two-Stage T  (δ_i = ESM-2 within-gene delta embedding)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Load ESM-2 embeddings
# ─────────────────────────────────────────────────────────────────
print("\n[1] Loading ESM-2 embeddings...")
X_esm_tr = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_esm_te = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
print(f"  ESM2  train: {X_esm_tr.shape}  test: {X_esm_te.shape}")

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

n_multi_tr = sum(1 for idxs in gene2idxs_tr.values() if len(idxs) > 1)
n_multi_te = sum(1 for idxs in gene2idxs_te.values() if len(idxs) > 1)
print(f"  Train: {len(tr_genes)} isoforms, {len(gene2idxs_tr)} genes, "
      f"{n_multi_tr} multi-isoform genes ({n_multi_tr/len(gene2idxs_tr):.1%})")
print(f"  Test:  {len(te_sym_list)} isoforms, {len(gene2idxs_te)} genes, "
      f"{n_multi_te} multi-isoform genes ({n_multi_te/len(gene2idxs_te):.1%})")

# ─────────────────────────────────────────────────────────────────
# 3. Compute delta embeddings: δ_i = φ(i) - μ_gene(i)
# ─────────────────────────────────────────────────────────────────
print("\n[3] Computing delta embeddings (δ_i = φ(i) − μ_gene)...")

delta_tr = np.zeros_like(X_esm_tr)
for gene, idxs in gene2idxs_tr.items():
    gene_mean = X_esm_tr[idxs].mean(axis=0)
    for i in idxs:
        delta_tr[i] = X_esm_tr[i] - gene_mean

delta_te = np.zeros_like(X_esm_te)
for gene, idxs in gene2idxs_te.items():
    gene_mean = X_esm_te[idxs].mean(axis=0)
    for i in idxs:
        delta_te[i] = X_esm_te[i] - gene_mean

# Report delta statistics
delta_norms_tr = np.linalg.norm(delta_tr, axis=1)
n_zero_tr = (delta_norms_tr < 1e-6).sum()
print(f"  Train delta: mean_norm={delta_norms_tr.mean():.4f}  "
      f"std={delta_norms_tr.std():.4f}  zero(single-iso)={n_zero_tr} ({n_zero_tr/len(tr_genes):.1%})")
delta_norms_te = np.linalg.norm(delta_te, axis=1)
n_zero_te = (delta_norms_te < 1e-6).sum()
print(f"  Test  delta: mean_norm={delta_norms_te.mean():.4f}  "
      f"std={delta_norms_te.std():.4f}  zero(single-iso)={n_zero_te} ({n_zero_te/len(te_sym_list):.1%})")

# Scale delta for T_ψ input (mean pool zeroed; scale to unit range)
scaler = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler.transform(delta_te).astype(np.float32)

# ─────────────────────────────────────────────────────────────────
# 4. GO labels
# ─────────────────────────────────────────────────────────────────
print("\n[4] Loading GO labels...")
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

Y_tr = np.stack([build_Y_tr(go)        for go in mf_terms], axis=1)
Y_te = np.stack([build_Y_te_brain(go)  for go in mf_terms], axis=1)
valid_mask = Y_te.sum(0) >= 2
print(f"  {len(mf_terms)} MF terms | Y_tr: {Y_tr.shape} | valid: {valid_mask.sum()}")

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
# 5. Triplet mining: GO-label cross-gene on DELTA space
# ─────────────────────────────────────────────────────────────────
print("\n[5] Mining GO-label triplets on delta space...")
rng = np.random.default_rng(42)

# Only anchor on multi-isoform genes — single-iso δ=0 are uninformative anchors
multi_iso_mask = np.array([len(gene2idxs_tr[g]) > 1 for g in tr_genes])
print(f"  Multi-isoform isoforms (δ≠0): {multi_iso_mask.sum()} / {len(tr_genes)} "
      f"({multi_iso_mask.mean():.1%})")

trip_a, trip_p, trip_n = [], [], []

for k, go_id in enumerate(mf_terms):
    y_k      = Y_tr[:, k]
    pos_idxs = np.where((y_k == 1) & multi_iso_mask)[0]   # only multi-iso positives
    neg_idxs = np.where(y_k == 0)[0]
    if len(pos_idxs) < 5 or len(neg_idxs) < 10:
        continue
    if len(trip_a) >= MAX_GO_TRIPS:
        break

    n_anchor = min(GO_TRIPS_PER, len(pos_idxs))
    for a_idx in rng.choice(pos_idxs, n_anchor, replace=False):
        a_gene     = tr_genes[a_idx]
        cross_pos  = pos_idxs[gene_arr_tr[pos_idxs] != a_gene]
        cross_neg  = neg_idxs[gene_arr_tr[neg_idxs] != a_gene]
        if len(cross_pos) < 2 or len(cross_neg) < 2:
            continue
        trip_a.append(a_idx)
        trip_p.append(int(rng.choice(cross_pos)))
        trip_n.append(int(rng.choice(cross_neg)))

trip_a = np.array(trip_a, dtype=np.int32)
trip_p = np.array(trip_p, dtype=np.int32)
trip_n = np.array(trip_n, dtype=np.int32)
print(f"  GO-label triplets: {len(trip_a)}")
print(f"  Anchor zero-δ fraction: {(delta_norms_tr[trip_a] < 1e-6).mean():.1%}  "
      f"(should be ~0 — multi-iso filter)")

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
else:
    print("\n  CPU mode")

# ─────────────────────────────────────────────────────────────────
# 7. Build models
# ─────────────────────────────────────────────────────────────────
def build_T_psi(delta_dim=640, embed_dim=64):
    """T_ψ embeds the within-gene differential signal δ_i."""
    inp = layers.Input(shape=(delta_dim,), name='delta_input')
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(embed_dim, activation='relu')(x)
    out = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1), name='l2_norm')(x)
    return models.Model(inputs=inp, outputs=out, name='T_psi_v17d')

def build_mlp_stage2(esm_dim=640, t_dim=64, n_go=82):
    """Stage 2: full ESM-2 (absolute) + T_ψ(δ) (differential)."""
    inp_t   = layers.Input(shape=(t_dim,),   name='t_delta_input')
    inp_esm = layers.Input(shape=(esm_dim,), name='esm2_input')
    x = layers.Concatenate()([inp_t, inp_esm])   # [704]
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model(inputs=[inp_t, inp_esm], outputs=out, name='MLP_theta_v17d')

def triplet_loss_fn(embeddings, a, p, n, margin=0.3):
    ea = tf.gather(embeddings, a)
    ep = tf.gather(embeddings, p)
    en = tf.gather(embeddings, n)
    d_pos = 1.0 - tf.reduce_sum(ea * ep, axis=1)
    d_neg = 1.0 - tf.reduce_sum(ea * en, axis=1)
    return tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))

# ─────────────────────────────────────────────────────────────────
# 8. Stage 1: Train T_ψ on δ_i
# ─────────────────────────────────────────────────────────────────
print(f"\n[6] Stage 1: Training T_ψ on δ_i ({EPOCHS_T} epochs, {BATCH_T} batch)...")
t0 = time.time()
tf.random.set_seed(42)
T_psi    = build_T_psi(delta_dim=delta_tr_s.shape[1], embed_dim=EMBED_DIM_T)
opt_T    = optimizers.Adam(1e-3)
delta_tf = tf.constant(delta_tr_s, dtype=tf.float32)

n_triplets = len(trip_a)
n_batches  = max(1, n_triplets // BATCH_T)

for epoch in range(EPOCHS_T):
    perm       = np.random.permutation(n_triplets)
    epoch_loss = 0.0
    for b in range(n_batches):
        batch_idx = perm[b * BATCH_T: (b + 1) * BATCH_T]
        ba = trip_a[batch_idx]; bp = trip_p[batch_idx]; bn = trip_n[batch_idx]
        with tf.GradientTape() as tape:
            embs = T_psi(delta_tf, training=True)
            loss = triplet_loss_fn(embs, ba, bp, bn, MARGIN)
        grads = tape.gradient(loss, T_psi.trainable_variables)
        opt_T.apply_gradients(zip(grads, T_psi.trainable_variables))
        epoch_loss += float(loss)

    if (epoch + 1) % 5 == 0:
        embs_np     = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
        ea = embs_np[trip_a]; ep = embs_np[trip_p]; en = embs_np[trip_n]
        d_pos       = 1 - (ea * ep).sum(1)
        d_neg       = 1 - (ea * en).sum(1)
        active_frac = ((d_pos - d_neg + MARGIN) > 0).mean()
        print(f"  Epoch {epoch+1:3d} | loss={epoch_loss/n_batches:.4f} | active={active_frac:.2%}")

print(f"  T_ψ training: {time.time()-t0:.0f}s")
T_tr = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
T_te = T_psi.predict(delta_te_s, batch_size=1024, verbose=0)
print(f"  T_tr: {T_tr.shape}  T_te: {T_te.shape}")

# Check: T_te norms for zero-delta (single-iso) vs multi-iso isoforms
delta_norms_te_s = np.linalg.norm(delta_te_s, axis=1)
zero_mask_te     = delta_norms_te_s < 1e-6
t_norms_te       = np.linalg.norm(T_te, axis=1)
print(f"  T_te norm: multi-iso={t_norms_te[~zero_mask_te].mean():.4f}  "
      f"single-iso(δ=0)={t_norms_te[zero_mask_te].mean():.4f}")

# ─────────────────────────────────────────────────────────────────
# 9. Stage 2: MLP_θ on concat[T_ψ(δ_i), φ(i)]
# ─────────────────────────────────────────────────────────────────
print(f"\n[7] Stage 2: MLP_θ concat[T_ψ(δ), ESM-2] ({EPOCHS_MLP} epochs, {len(SEEDS)} seeds)...")
t0 = time.time()

n_go     = len(mf_terms)
focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

all_preds = []
for seed in SEEDS:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    perm    = np.random.permutation(len(T_tr))
    n_val   = int(len(T_tr) * 0.1)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]

    mlp = build_mlp_stage2(esm_dim=X_esm_tr.shape[1], t_dim=EMBED_DIM_T, n_go=n_go)
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
# 10. Evaluation
# ─────────────────────────────────────────────────────────────────
print(f"\n[8] Evaluation...")

def eval_subset(preds, Y, term_indices, label):
    auprc_list = []
    for i in term_indices:
        y_true = Y[:, i]; y_pred = preds[:, i]
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

print(f"\n  {'='*70}")
print(f"  Comparison (MF {len(valid_idx)} valid terms):")
print(f"  {'Method':<48} {'AUPRC':>8}  {'L2_Str':>7}  {'L4_Cel':>7}")
print(f"  {'-'*72}")
print(f"  {'PRISM v15d (muscle, ref)':<48} {ref_muscle:>8.4f}")
print(f"  {'PRISM v15d (brain zero-shot, ref)':<48} {ref_brain:>8.4f}")
print(f"  {'v17  T_ψ(ESM+Pfam) — FAILED':<48} {'0.3543':>8}  {'0.1783':>7}  {'0.5639':>7}")
print(f"  {'v17b T_ψ(Pfam)+MLP(T+ESM)':<48} {'0.6022':>8}  {'0.4699':>7}  {'0.7415':>7}")
print(f"  {'v17c T_ψ(Pfam)+MLP(T+ESM) w/ GO-trips':<48} {'0.6003':>8}  {'0.4721':>7}  {'0.7349':>7}")
print(f"  {'v17d T_ψ(δ_i)+MLP(T+ESM)  [this run]':<48} {macro_all:>8.4f}  {macro_l2:>7.4f}  {macro_l4:>7.4f}")
print(f"\n  Δ(v17d - PRISM brain): {macro_all-ref_brain:+.4f}")
print(f"  Δ(v17d - v17c):        {macro_all-0.6003:+.4f}")
print(f"  Δ(v17d - v17c) L2:     {macro_l2-0.4721:+.4f}  ← key metric")
print(f"\n  Triplets: {len(trip_a):,} GO-label (delta-space, multi-iso anchors only)")

# ─────────────────────────────────────────────────────────────────
# 11. Save
# ─────────────────────────────────────────────────────────────────
per_term = []
for i, go_id in enumerate(mf_terms):
    y_true   = Y_te[:, i]; y_pred = preds_ensemble[:, i]
    v17d_ap  = float(average_precision_score(y_true, y_pred)) if y_true.sum() >= 2 else None
    per_term.append({
        'go_id':            go_id,
        'prism_brain':      prism_brain_ref.get(go_id),
        'v17c_baseline':    None,
        'v17d_auprc':       v17d_ap,
        'delta_v17d_prism': (v17d_ap - prism_brain_ref[go_id])
                            if v17d_ap and prism_brain_ref.get(go_id) else None,
        'is_l2_structural': go_id in L2_TERMS,
    })

results = {
    'macro_v17d_all_mf':        macro_all,
    'macro_v17d_l2_structural': macro_l2,
    'macro_v17d_l4_sample':     macro_l4,
    'prism_brain_ref':          ref_brain,
    'v17c_ref':                 {'all_mf': 0.6003, 'l2_structural': 0.4721, 'l4_sample': 0.7349},
    'delta_v17d_vs_prism':      macro_all - ref_brain,
    'delta_v17d_vs_v17c':       macro_all - 0.6003,
    'delta_l2_vs_v17c':         macro_l2  - 0.4721,
    'n_triplets':               int(len(trip_a)),
    'n_valid_terms':            len(valid_idx),
    'architecture': {
        'T_psi_input': 'delta_i = phi(i) - gene_mean_phi',
        'T_psi_dims':  '640→256→64(L2-norm)',
        'stage2_input': 'concat[T_psi(delta)(64), ESM-2(640)] = 704',
        'triplet_type': 'GO-label cross-gene, multi-iso anchors only',
    },
    'per_term': per_term,
}
json.dump(results, open(f'{OUT_DIR}/v17d_results.json', 'w'), indent=2)
print(f"\n  [Saved] {OUT_DIR}/v17d_results.json")

with open(f'{OUT_DIR}/v17d_per_term.tsv', 'w') as f:
    f.write('go_id\tprism_brain\tv17d_auprc\tdelta_v17d_prism\tis_l2_structural\n')
    for r in per_term:
        pb    = f"{r['prism_brain']:.4f}"  if r['prism_brain']  is not None else 'NA'
        v17d  = f"{r['v17d_auprc']:.4f}"  if r['v17d_auprc']   is not None else 'NA'
        delta = f"{r['delta_v17d_prism']:.4f}" if r['delta_v17d_prism'] is not None else 'NA'
        f.write(f"{r['go_id']}\t{pb}\t{v17d}\t{delta}\t{r['is_l2_structural']}\n")
print(f"  [Saved] {OUT_DIR}/v17d_per_term.tsv")

print("\n" + "=" * 65)
print("  v17d Delta Embedding — COMPLETE")
print("=" * 65)
