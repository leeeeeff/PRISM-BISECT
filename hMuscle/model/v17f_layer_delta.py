#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_layer_delta.py
-------------------
Inductive two-stage T architecture using ESM-2 layer-contrast delta.

v17d solved the gradient-conflict problem for gene-mean delta but requires
TRANSDUCTIVE inference: δ_gene_i = φ(i) − μ_gene(i) needs gene siblings.
Single-isoform genes (26.9% train) always get δ=0; novel isoforms from
unseen genes cannot be scored differentially.

v17f replaces the transductive gene-mean with an INDUCTIVE layer-contrast:

  δ_layer_i = φ_L30(i) − φ_L15(i)   [640-dim, per-sequence]

Rationale — ESM-2 layer semantics:
  L15 (mid-layer): local structural features — secondary structure, amino
    acid chemistry, hydrophobic patches — determined by local sequence context.
  L30 (final layer): global functional embedding — evolutionary protein
    family context, domain topology, long-range co-evolutionary signals.
  δ_layer = L30 − L15 captures "how much global functional context emerges
    from this sequence's structural features" — the representation delta
    across depth, which is isoform-specific and context-independent.

Inductive property:
  δ_layer_i is computed purely from isoform i's own sequence.
  No gene siblings needed. All 100% of isoforms have non-zero δ_layer.
  Directly applicable to novel isoforms at inference time.

Key difference from v17d:
  - All isoforms can be T_ψ anchors (no multi-iso filter)
  - δ_layer encodes WITHIN-SEQUENCE depth contrast (absolute)
    vs δ_gene which encodes WITHIN-GENE variation (relative)
  - Tests: does layer-contrast signal approach the informativeness of
    gene-mean delta for MF prediction?

Architecture (identical to v17d except delta source):
  Stage 1 — T_ψ (layer-contrast delta embedding):
    δ_layer_i [640] → Dense(256,ReLU) → BN → Drop(0.3) → Dense(64,ReLU) → L2-norm → [64]
    Loss: GO-label cross-gene triplet
  Stage 2 — MLP_θ (T_ψ frozen, ESM-2 bypass):
    concat[T_ψ(δ_layer)(64), φ_L30(640)] = [704] → Dense(256→128→82, focal γ=2)
    5-seed ensemble
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
OUT_DIR   = '../../reports/v17f_layer_delta'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS       = [42, 7, 13, 21, 99]
MARGIN      = 0.3
BATCH_T     = 512
EPOCHS_T    = 50
BATCH_MLP   = 512
EPOCHS_MLP  = 60
EMBED_DIM_T = 64
MAX_GO_TRIPS = 30000
GO_TRIPS_PER = 300
LAYER_A     = 15   # "structural" layer
LAYER_B     = 30   # "functional" layer (== final ESM-2 layer)

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print(f"  v17f Layer-Contrast Delta  (δ = L{LAYER_B} − L{LAYER_A})")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Load ESM-2 embeddings (L30 == main embedding)
# ─────────────────────────────────────────────────────────────────
print("\n[1] Loading ESM-2 layer embeddings...")

# Train
X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l15_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
# Test (brain zero-shot)
X_l30_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l15_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_A:02d}_t30_150M.npy').astype(np.float32)

X_esm_tr = X_l30_tr   # L30 == final ESM-2 representation
X_esm_te = X_l30_te
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

n_multi_tr = sum(1 for idxs in gene2idxs_tr.values() if len(idxs) > 1)
n_multi_te = sum(1 for idxs in gene2idxs_te.values() if len(idxs) > 1)
print(f"  Train: {len(tr_genes)} isoforms, {len(gene2idxs_tr)} genes, "
      f"{n_multi_tr} multi-isoform ({n_multi_tr/len(gene2idxs_tr):.1%})")
print(f"  Test:  {len(te_sym_list)} isoforms, {len(gene2idxs_te)} genes, "
      f"{n_multi_te} multi-isoform ({n_multi_te/len(gene2idxs_te):.1%})")

# ─────────────────────────────────────────────────────────────────
# 3. Compute layer-contrast delta: δ_layer_i = L30(i) - L15(i)
# ─────────────────────────────────────────────────────────────────
print(f"\n[3] Computing δ_layer = L{LAYER_B}(i) − L{LAYER_A}(i)  [inductive, per-sequence]...")

delta_tr = (X_l30_tr - X_l15_tr).astype(np.float32)
delta_te = (X_l30_te - X_l15_te).astype(np.float32)

delta_norms_tr = np.linalg.norm(delta_tr, axis=1)
delta_norms_te = np.linalg.norm(delta_te, axis=1)
n_zero_tr = (delta_norms_tr < 1e-6).sum()
n_zero_te = (delta_norms_te < 1e-6).sum()

print(f"  Train δ_layer: mean_norm={delta_norms_tr.mean():.2f}  zero={n_zero_tr} ({n_zero_tr/len(tr_genes):.1%})")
print(f"  Test  δ_layer: mean_norm={delta_norms_te.mean():.2f}  zero={n_zero_te} ({n_zero_te/len(te_sym_list):.1%})")
print(f"  Compare: v17d gene-mean δ had mean_norm=0.38 (train), 0.52 (test)")

scaler = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler.transform(delta_te).astype(np.float32)

# ─────────────────────────────────────────────────────────────────
# 4. GO labels (MF 82 terms — same as v17d/v17e)
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

def build_Y_te_brain(go_id):
    pos_ids = go_genes_all[go_id]
    return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0
                     for s in te_sym_list], dtype=np.float32)

Y_tr = np.stack([build_Y_tr(go)       for go in mf_terms], axis=1)
Y_te = np.stack([build_Y_te_brain(go) for go in mf_terms], axis=1)
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
# 5. Triplet mining: GO-label cross-gene on LAYER-DELTA space
#    All isoforms can be anchors (no zero-delta exclusion)
# ─────────────────────────────────────────────────────────────────
print("\n[5] Mining GO-label triplets on δ_layer space...")
rng = np.random.default_rng(42)

# All isoforms are eligible anchors (δ_layer is non-zero for all)
n_zero_anchor = (delta_norms_tr < 1e-6).sum()
print(f"  Zero-δ anchors (excluded in v17d, included here): {n_zero_anchor} ({n_zero_anchor/len(tr_genes):.1%})")

trip_a, trip_p, trip_n = [], [], []

for k, go_id in enumerate(mf_terms):
    y_k      = Y_tr[:, k]
    pos_idxs = np.where(y_k == 1)[0]   # all positives (not just multi-iso)
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
# 7. Models
# ─────────────────────────────────────────────────────────────────
def build_T_psi(delta_dim=640, embed_dim=64):
    inp = layers.Input(shape=(delta_dim,), name='delta_layer_input')
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(embed_dim, activation='relu')(x)
    out = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1), name='l2_norm')(x)
    return models.Model(inputs=inp, outputs=out, name='T_psi_v17f')

def build_mlp_stage2(esm_dim=640, t_dim=64, n_go=82):
    inp_t   = layers.Input(shape=(t_dim,),   name='t_layer_input')
    inp_esm = layers.Input(shape=(esm_dim,), name='esm2_input')
    x = layers.Concatenate()([inp_t, inp_esm])   # [704]
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model(inputs=[inp_t, inp_esm], outputs=out, name='MLP_theta_v17f')

def triplet_loss_fn(embeddings, a, p, n, margin=0.3):
    ea = tf.gather(embeddings, a)
    ep = tf.gather(embeddings, p)
    en = tf.gather(embeddings, n)
    d_pos = 1.0 - tf.reduce_sum(ea * ep, axis=1)
    d_neg = 1.0 - tf.reduce_sum(ea * en, axis=1)
    return tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))

# ─────────────────────────────────────────────────────────────────
# 8. Stage 1: Train T_ψ on δ_layer
# ─────────────────────────────────────────────────────────────────
print(f"\n[6] Stage 1: T_ψ on δ_layer ({EPOCHS_T} epochs)...")
t0 = time.time()
tf.random.set_seed(42)
T_psi    = build_T_psi(delta_dim=640, embed_dim=EMBED_DIM_T)
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

    if (epoch + 1) % 10 == 0:
        embs_np     = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
        ea = embs_np[trip_a]; ep = embs_np[trip_p]; en = embs_np[trip_n]
        d_pos       = 1 - (ea * ep).sum(1)
        d_neg       = 1 - (ea * en).sum(1)
        active_frac = ((d_pos - d_neg + MARGIN) > 0).mean()
        print(f"  Epoch {epoch+1:3d} | loss={epoch_loss/n_batches:.4f} | active={active_frac:.2%}")

print(f"  T_ψ training: {time.time()-t0:.0f}s")
T_tr = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
T_te = T_psi.predict(delta_te_s, batch_size=1024, verbose=0)
print(f"  T_tr norm: {np.linalg.norm(T_tr, axis=1).mean():.4f}  T_te norm: {np.linalg.norm(T_te, axis=1).mean():.4f}")

# ─────────────────────────────────────────────────────────────────
# 9. Stage 2: MLP_θ on concat[T_ψ(δ_layer), φ_L30]
# ─────────────────────────────────────────────────────────────────
print(f"\n[7] Stage 2: MLP_θ ({EPOCHS_MLP} epochs, {len(SEEDS)} seeds)...")
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

macro_all, _ = eval_subset(preds_ensemble, Y_te, valid_idx, f"All MF ({n_go} terms)      ")
macro_l2,  _ = eval_subset(preds_ensemble, Y_te, l2_valid,  f"L2_Structural ({len(l2_valid)} MF)  ")
macro_l4,  _ = eval_subset(preds_ensemble, Y_te, l4_valid,  f"L4_CellState sample {len(l4_valid)}")

ref_brain = np.mean([v for v in prism_brain_ref.values() if v is not None])

PRISM_REF = 0.5962; PRISM_L2 = 0.3501
V17D_REF  = 0.6825; V17D_L2  = 0.5586
V17E_REF  = 0.6389; V17E_L2  = 0.4865

print(f"\n  {'='*72}")
print(f"  v17 Ablation Series (82 MF terms, brain zero-shot):")
print(f"  {'Method':<52} {'All MF':>7}  {'L2_Str':>7}")
print(f"  {'-'*70}")
print(f"  {'PRISM v15d (brain ref)':<52} {PRISM_REF:>7.4f}  {PRISM_L2:>7.4f}")
print(f"  {'v17e concat[δ_gene(640)+ESM(640)] — no T_ψ':<52} {V17E_REF:>7.4f}  {V17E_L2:>7.4f}")
print(f"  {'v17d T_ψ(δ_gene)+Stage2[T(64)+ESM(640)]':<52} {V17D_REF:>7.4f}  {V17D_L2:>7.4f}")
print(f"  {'v17f T_ψ(δ_layer)+Stage2[T(64)+ESM(640)]  [this]':<52} {macro_all:>7.4f}  {macro_l2:>7.4f}")
print(f"\n  Δ(v17f - PRISM):   All MF: {macro_all-PRISM_REF:+.4f}  L2: {macro_l2-PRISM_L2:+.4f}")
print(f"  Δ(v17f - v17e):    All MF: {macro_all-V17E_REF:+.4f}  L2: {macro_l2-V17E_L2:+.4f}")
print(f"  Δ(v17f - v17d):    All MF: {macro_all-V17D_REF:+.4f}  L2: {macro_l2-V17D_L2:+.4f}")
print(f"\n  Interpretation:")
if macro_l2 > V17E_L2:
    print(f"  → δ_layer adds value over v17e ({macro_l2-V17E_L2:+.4f} L2): layer contrast is informative")
else:
    print(f"  → δ_layer not better than v17e ({macro_l2-V17E_L2:+.4f} L2): gene-mean reference essential")
if macro_l2 > PRISM_L2:
    print(f"  → L2_Structural improved vs PRISM ({macro_l2-PRISM_L2:+.4f})")
gap_to_v17d = V17D_L2 - macro_l2
if gap_to_v17d < 0.02:
    print(f"  → v17f ≈ v17d (gap {gap_to_v17d:.4f}): layer delta ≈ gene-mean delta in L2 power")
elif gap_to_v17d < 0.05:
    print(f"  → v17f close to v17d (gap {gap_to_v17d:.4f}): layer delta partially recovers gene-mean benefit")
else:
    print(f"  → v17f substantially below v17d (gap {gap_to_v17d:.4f}): gene-specific reference is key")

# ─────────────────────────────────────────────────────────────────
# 11. Save
# ─────────────────────────────────────────────────────────────────
per_term = []
for i, go_id in enumerate(mf_terms):
    y_true = Y_te[:, i]; y_pred = preds_ensemble[:, i]
    ap = float(average_precision_score(y_true, y_pred)) if y_true.sum() >= 2 else None
    per_term.append({
        'go_id':        go_id,
        'prism_brain':  prism_brain_ref.get(go_id),
        'v17f_auprc':   ap,
        'delta_vs_prism': (ap - prism_brain_ref[go_id])
                           if ap and prism_brain_ref.get(go_id) else None,
        'delta_vs_v17d':  (ap - V17D_REF) if ap else None,
        'is_l2_structural': go_id in L2_TERMS,
    })

results = {
    'macro_v17f_all_mf':        macro_all,
    'macro_v17f_l2_structural': macro_l2,
    'macro_v17f_l4_sample':     macro_l4,
    'prism_brain_ref':          ref_brain,
    'reference_series': {
        'prism': {'all_mf': PRISM_REF, 'l2': PRISM_L2},
        'v17e':  {'all_mf': V17E_REF,  'l2': V17E_L2},
        'v17d':  {'all_mf': V17D_REF,  'l2': V17D_L2},
    },
    'delta_v17f_vs_prism': {'all_mf': macro_all - PRISM_REF, 'l2': macro_l2 - PRISM_L2},
    'delta_v17f_vs_v17e':  {'all_mf': macro_all - V17E_REF,  'l2': macro_l2 - V17E_L2},
    'delta_v17f_vs_v17d':  {'all_mf': macro_all - V17D_REF,  'l2': macro_l2 - V17D_L2},
    'n_triplets':  int(len(trip_a)),
    'n_valid_terms': len(valid_idx),
    'architecture': {
        'T_psi_input':   f'delta_layer = L{LAYER_B}(i) - L{LAYER_A}(i) [inductive]',
        'T_psi_dims':    '640→256→64(L2-norm)',
        'stage2_input':  'concat[T_psi(delta_layer)(64), ESM-2-L30(640)] = 704',
        'triplet_type':  'GO-label cross-gene, all isoforms eligible',
        'key_property':  'inductive: no gene siblings required',
    },
    'per_term': per_term,
}
json.dump(results, open(f'{OUT_DIR}/v17f_results.json', 'w'), indent=2)
print(f"\n  [Saved] {OUT_DIR}/v17f_results.json")

with open(f'{OUT_DIR}/v17f_per_term.tsv', 'w') as f:
    f.write('go_id\tprism_brain\tv17f_auprc\tdelta_vs_prism\tdelta_vs_v17d\tis_l2_structural\n')
    for r in per_term:
        pb  = f"{r['prism_brain']:.4f}"      if r['prism_brain']    is not None else 'NA'
        v17f = f"{r['v17f_auprc']:.4f}"     if r['v17f_auprc']     is not None else 'NA'
        dp   = f"{r['delta_vs_prism']:.4f}" if r['delta_vs_prism'] is not None else 'NA'
        dd   = f"{r['delta_vs_v17d']:.4f}"  if r['delta_vs_v17d']  is not None else 'NA'
        f.write(f"{r['go_id']}\t{pb}\t{v17f}\t{dp}\t{dd}\t{r['is_l2_structural']}\n")
print(f"  [Saved] {OUT_DIR}/v17f_per_term.tsv")

print("\n" + "=" * 65)
print("  v17f Layer-Contrast Delta — COMPLETE")
print("=" * 65)
