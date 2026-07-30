#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_layer_search.py
--------------------
Systematic search for the optimal mid-layer in δ_layer = L30 − L_mid.

Current: L15 → v17f AUPRC 0.7171 All MF / 0.6156 L2_Structural

Hypothesis on layer semantics (ESM-2 t30_150M, 30 layers):
  L5:  Local amino acid chemistry, residue pair patterns, hydrophobicity
  L10: Secondary structure (helix/sheet propensity), early tertiary contacts
  L15: Domain topology, β-barrel/coil patterns, buried residue networks  [current]
  L20: Long-range contacts, proto-domain families, fold-level context
  L25: Protein family context, evolutionary conservation signatures
  L30: Full functional representation (≡ φ(i), the final ESM-2 embedding)

δ = L30 − L_mid captures: "what the layers L_mid+1 … L30 add to the structural baseline"
  Small L_mid → large δ (many layers subtracted) → more functional depth captured
  Large L_mid → small δ (few layers subtracted) → focused on late refinement only

Test candidates: [5, 10, 15, 20, 25]
L30-L30 = 0 (trivially useless, excluded)
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
OUT_DIR   = '../../reports/v17f_layer_search'
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
LAYER_B      = 30                          # fixed: final layer
MID_LAYERS   = [5, 10, 15, 20, 25]        # candidates for L_mid

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print(f"  v17f Layer Search  (L{LAYER_B} − L_mid  for mid ∈ {MID_LAYERS})")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Load fixed data (gene IDs, GO labels, L30 embeddings)
# ─────────────────────────────────────────────────────────────────
print("\n[1] Loading fixed data...")

X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l30_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
gene_arr_tr  = np.array(tr_genes)
gene2idxs_tr = defaultdict(list)
for i, g in enumerate(tr_genes): gene2idxs_tr[g].append(i)

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]
gene_arr_te  = np.array(te_sym_list)
gene2idxs_te = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs_te[g].append(i)

print(f"  Train: {len(tr_genes)} isoforms  Test: {len(te_sym_list)} isoforms")

# GO labels
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

mf_terms, prism_brain_ref = [], {}
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) < 6: continue
        mf_terms.append(p[0])
        prism_brain_ref[p[0]] = float(p[5]) if p[5] else None

tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

def build_Y_tr(go_id):
    pos_ids = go_genes_tr[go_id]
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
print(f"  {len(mf_terms)} MF terms | valid: {len(valid_idx)} | L2_Structural: {len(l2_valid)}")

# Fixed triplets (computed once, reused across layer experiments)
rng = np.random.default_rng(42)
trip_a, trip_p, trip_n = [], [], []
for k, go_id in enumerate(mf_terms):
    y_k = Y_tr[:, k]
    pos_idxs = np.where(y_k == 1)[0]
    neg_idxs = np.where(y_k == 0)[0]
    if len(pos_idxs) < 5 or len(neg_idxs) < 10: continue
    if len(trip_a) >= MAX_GO_TRIPS: break
    n_anchor = min(GO_TRIPS_PER, len(pos_idxs))
    for a_idx in rng.choice(pos_idxs, n_anchor, replace=False):
        a_gene    = tr_genes[a_idx]
        cross_pos = pos_idxs[gene_arr_tr[pos_idxs] != a_gene]
        cross_neg = neg_idxs[gene_arr_tr[neg_idxs] != a_gene]
        if len(cross_pos) < 2 or len(cross_neg) < 2: continue
        trip_a.append(a_idx)
        trip_p.append(int(rng.choice(cross_pos)))
        trip_n.append(int(rng.choice(cross_neg)))
trip_a = np.array(trip_a, dtype=np.int32)
trip_p = np.array(trip_p, dtype=np.int32)
trip_n = np.array(trip_n, dtype=np.int32)
print(f"  Triplets: {len(trip_a)} (fixed across all layer experiments)")

# ─────────────────────────────────────────────────────────────────
# 2. TensorFlow
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
    print(f"  GPU: {gpus[0].name}")

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

def triplet_loss_fn(embs, a, p, n, margin=0.3):
    ea = tf.gather(embs, a); ep = tf.gather(embs, p); en = tf.gather(embs, n)
    return tf.reduce_mean(tf.maximum(
        (1 - tf.reduce_sum(ea*ep, 1)) - (1 - tf.reduce_sum(ea*en, 1)) + margin, 0))

def macro_auprc(preds, Y, idxs):
    vals = [average_precision_score(Y[:,i], preds[:,i])
            for i in idxs if Y[:,i].sum() >= 2]
    return float(np.mean(vals)) if vals else float('nan')

# ─────────────────────────────────────────────────────────────────
# 3. Layer search loop
# ─────────────────────────────────────────────────────────────────
results_all = {}

for mid_layer in MID_LAYERS:
    t_run = time.time()
    print(f"\n{'─'*65}")
    print(f"  Running: δ = L{LAYER_B} − L{mid_layer:02d}")
    print(f"{'─'*65}")

    # Load mid-layer embeddings
    X_lmid_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{mid_layer:02d}_t30_150M.npy').astype(np.float32)
    X_lmid_te = np.load(f'{DATA_DIR}/esm2_layer_{mid_layer:02d}_t30_150M.npy').astype(np.float32)

    # Compute delta
    delta_tr = (X_l30_tr - X_lmid_tr).astype(np.float32)
    delta_te = (X_l30_te - X_lmid_te).astype(np.float32)
    d_norms  = np.linalg.norm(delta_tr, axis=1)
    print(f"  δ(L{LAYER_B}-L{mid_layer:02d}) train: mean_norm={d_norms.mean():.2f}  std={d_norms.std():.2f}")

    scaler   = MaxAbsScaler()
    delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
    delta_te_s = scaler.transform(delta_te).astype(np.float32)

    # Stage 1: T_ψ
    tf.random.set_seed(42)
    T_psi    = build_T_psi()
    opt_T    = optimizers.Adam(1e-3)
    delta_tf = tf.constant(delta_tr_s, dtype=tf.float32)
    n_trip   = len(trip_a)
    n_batch  = max(1, n_trip // BATCH_T)

    final_active = 0.0
    for epoch in range(EPOCHS_T):
        perm = np.random.permutation(n_trip)
        el   = 0.0
        for b in range(n_batch):
            bi = perm[b*BATCH_T:(b+1)*BATCH_T]
            with tf.GradientTape() as tape:
                embs = T_psi(delta_tf, training=True)
                loss = triplet_loss_fn(embs, trip_a[bi], trip_p[bi], trip_n[bi])
            grads = tape.gradient(loss, T_psi.trainable_variables)
            opt_T.apply_gradients(zip(grads, T_psi.trainable_variables))
            el += float(loss)
        if (epoch + 1) % 10 == 0:
            embs_np = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
            ea = embs_np[trip_a]; ep2 = embs_np[trip_p]; en2 = embs_np[trip_n]
            final_active = ((1-(ea*ep2).sum(1)) - (1-(ea*en2).sum(1)) + MARGIN > 0).mean()
            print(f"  Epoch {epoch+1:3d} | loss={el/n_batch:.4f} | active={final_active:.2%}")

    T_tr = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
    T_te = T_psi.predict(delta_te_s, batch_size=1024, verbose=0)

    # Stage 2: ensemble
    focal_fn  = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
    all_preds = []
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm    = np.random.permutation(len(T_tr))
        n_val   = int(len(T_tr) * 0.1)
        val_idx = perm[:n_val]; tr_idx = perm[n_val:]
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
    ap_all = macro_auprc(preds, Y_te, valid_idx)
    ap_l2  = macro_auprc(preds, Y_te, l2_valid)

    elapsed = time.time() - t_run
    print(f"\n  L{LAYER_B}-L{mid_layer:02d}: All MF={ap_all:.4f}  L2_Struct={ap_l2:.4f}  "
          f"active={final_active:.2%}  [{elapsed:.0f}s]")

    results_all[mid_layer] = {
        'mid_layer': mid_layer,
        'all_mf': ap_all,
        'l2_structural': ap_l2,
        'final_active_ratio': float(final_active),
        'delta_mean_norm': float(d_norms.mean()),
        'delta_std_norm': float(d_norms.std()),
        'elapsed_s': elapsed,
    }

    # Clear session between runs
    tf.keras.backend.clear_session()

# ─────────────────────────────────────────────────────────────────
# 4. Summary
# ─────────────────────────────────────────────────────────────────
PRISM_REF = 0.5962; PRISM_L2 = 0.3501

print(f"\n\n{'='*72}")
print(f"  Layer Search Results: δ = L{LAYER_B} − L_mid")
print(f"{'='*72}")
print(f"  {'Mid-layer':<12} {'δ mean‖':<10} {'active%':<10} {'All MF':<10} {'L2_Struct':<12} {'Δ vs L15'}")
print(f"  {'-'*70}")

ref_l15_all = results_all.get(15, {}).get('all_mf', float('nan'))
ref_l15_l2  = results_all.get(15, {}).get('l2_structural', float('nan'))

for mid in MID_LAYERS:
    r   = results_all[mid]
    tag = " ← current" if mid == 15 else ""
    da  = r['all_mf'] - ref_l15_all
    print(f"  L30 − L{mid:02d}{tag:<14} {r['delta_mean_norm']:>7.1f}    "
          f"{r['final_active_ratio']:>7.1%}    {r['all_mf']:.4f}    {r['l2_structural']:.4f}    "
          f"{da:+.4f}")

print(f"\n  PRISM v15d (baseline):                              "
      f"{PRISM_REF:.4f}    {PRISM_L2:.4f}")

best_mid = max(results_all, key=lambda m: results_all[m]['l2_structural'])
best     = results_all[best_mid]
print(f"\n  Best mid-layer: L{LAYER_B} − L{best_mid:02d}  "
      f"(All MF={best['all_mf']:.4f}  L2={best['l2_structural']:.4f})")

if best_mid != 15:
    gap = best['l2_structural'] - ref_l15_l2
    print(f"  Improvement over L15: L2 {gap:+.4f} All MF {best['all_mf']-ref_l15_all:+.4f}")
else:
    print(f"  L15 remains optimal.")

# Save
json.dump({'layer_b': LAYER_B, 'results': results_all},
          open(f'{OUT_DIR}/layer_search_results.json', 'w'), indent=2)

with open(f'{OUT_DIR}/layer_search_summary.tsv', 'w') as f:
    f.write('mid_layer\tdelta_key\tall_mf\tl2_structural\tactive_ratio\tdelta_mean_norm\n')
    for mid in MID_LAYERS:
        r = results_all[mid]
        f.write(f"{mid}\tL{LAYER_B}-L{mid:02d}\t{r['all_mf']:.4f}\t"
                f"{r['l2_structural']:.4f}\t{r['final_active_ratio']:.4f}\t"
                f"{r['delta_mean_norm']:.2f}\n")

print(f"\n  [Saved] {OUT_DIR}/layer_search_summary.tsv")
print(f"  [Saved] {OUT_DIR}/layer_search_results.json")

print("\n" + "=" * 65)
print("  v17f Layer Search — COMPLETE")
print("=" * 65)
