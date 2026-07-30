#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v18a_rna_loc.py
---------------
v17f + RNA_delta(9) + LOC_delta(8) as Stage 2 additional features.

Architecture:
  Stage 1 (T_ψ): unchanged — δ_layer(640) → T_ψ(64) via triplet learning
  Stage 2 MLP: [T_ψ(64) + φ_L30(640) + RNA_delta(9) + LOC_delta(8)] → 721-dim input

Feature choices:
  RNA_delta: isoform vs canonical difference in transcript-level RNA features
    (UTR lengths, CDS fraction, uORF, Kozak, ARE, rare codon, NMD proxy)
    → captures splice-induced changes in mRNA structure and stability
  LOC_delta: isoform vs canonical difference in subcellular localization features
    (MTS, signal peptide, NLS, TM domain, GPI, amphipathic)
    → captures splice-induced changes in protein routing and compartment

WHY delta features over absolute:
  Absolute features carry gene-level signal (e.g., all kinase isoforms have
  similar codon usage). Delta features are isoform-specific by construction,
  directly encoding what changes when a particular exon is included/excluded.
  This is analogous to δ_layer = L30 - L15 but in RNA/protein-localization space.

V17f baseline: All MF 0.7171 / L2_Structural 0.6156
Expected benefit: L2_Structural (LOC features for compartment GO terms) > L1_Generic
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
OUT_DIR   = '../../reports/v18a_rna_loc'
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
print("  v18a: v17f + RNA_delta(9) + LOC_delta(8)")
print("=" * 65)

# ── 1. Embeddings ─────────────────────────────────────────────
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

# RNA_delta and LOC_delta features (isoform-specific changes)
rna_d_tr = np.load(f'{FEAT_DIR}/rna/rna_delta_train.npy').astype(np.float32)
rna_d_te = np.load(f'{FEAT_DIR}/rna/rna_delta_test.npy').astype(np.float32)
loc_d_tr = np.load(f'{FEAT_DIR}/loc/loc_delta_train.npy').astype(np.float32)
loc_d_te = np.load(f'{FEAT_DIR}/loc/loc_delta_test.npy').astype(np.float32)

# Normalize auxiliary features
scaler_rna = MaxAbsScaler()
rna_d_tr_s = scaler_rna.fit_transform(rna_d_tr).astype(np.float32)
rna_d_te_s = scaler_rna.transform(rna_d_te).astype(np.float32)

scaler_loc = MaxAbsScaler()
loc_d_tr_s = scaler_loc.fit_transform(loc_d_tr).astype(np.float32)
loc_d_te_s = scaler_loc.transform(loc_d_te).astype(np.float32)

print(f"  Train: ESM-2 {X_l30_tr.shape} | RNA_delta {rna_d_tr.shape} | LOC_delta {loc_d_tr.shape}")
print(f"  Test:  ESM-2 {X_l30_te.shape} | RNA_delta {rna_d_te.shape} | LOC_delta {loc_d_te.shape}")

AUX_DIM = rna_d_tr.shape[1] + loc_d_tr.shape[1]  # 9 + 8 = 17
print(f"  Auxiliary feature dim: {AUX_DIM} (RNA_delta={rna_d_tr.shape[1]}, LOC_delta={loc_d_tr.shape[1]})")

# ── 2. IDs and GO labels ──────────────────────────────────────
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

L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural': L2_TERMS.add(p[0])

valid_idx = [i for i in range(len(mf_terms)) if valid_mask[i]]
l2_valid  = [i for i in range(len(mf_terms)) if mf_terms[i] in L2_TERMS and valid_mask[i]]
print(f"  {len(mf_terms)} MF terms | valid: {len(valid_idx)} | L2: {len(l2_valid)}")

# Triplets (fixed, same as v17f)
rng = np.random.default_rng(42)
trip_a, trip_p, trip_n = [], [], []
for k, go_id in enumerate(mf_terms):
    y_k = Y_tr[:, k]
    pos_idxs = np.where(y_k == 1)[0]; neg_idxs = np.where(y_k == 0)[0]
    if len(pos_idxs) < 5 or len(neg_idxs) < 10: continue
    if len(trip_a) >= MAX_GO_TRIPS: break
    n_anchor = min(GO_TRIPS_PER, len(pos_idxs))
    for a_idx in rng.choice(pos_idxs, n_anchor, replace=False):
        a_gene = tr_genes[a_idx]
        cross_pos = pos_idxs[gene_arr_tr[pos_idxs] != a_gene]
        cross_neg = neg_idxs[gene_arr_tr[neg_idxs] != a_gene]
        if len(cross_pos) < 2 or len(cross_neg) < 2: continue
        trip_a.append(a_idx); trip_p.append(int(rng.choice(cross_pos))); trip_n.append(int(rng.choice(cross_neg)))
trip_a = np.array(trip_a, dtype=np.int32)
trip_p = np.array(trip_p, dtype=np.int32)
trip_n = np.array(trip_n, dtype=np.int32)
print(f"  Triplets: {len(trip_a)}")

# ── 3. TF ─────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
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
    inp_t   = layers.Input(shape=(EMBED_DIM_T,))        # T_ψ(δ_layer)
    inp_esm = layers.Input(shape=(640,))                # φ_L30
    inp_aux = layers.Input(shape=(AUX_DIM,))            # RNA_delta + LOC_delta
    x = layers.Concatenate()([inp_t, inp_esm, inp_aux]) # 64+640+17 = 721
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(len(mf_terms), activation='sigmoid')(x)
    return models.Model([inp_t, inp_esm, inp_aux], out)

def triplet_loss_fn(embs, a, p, n):
    ea = tf.gather(embs, a); ep = tf.gather(embs, p); en = tf.gather(embs, n)
    return tf.reduce_mean(tf.maximum(
        (1 - tf.reduce_sum(ea*ep, 1)) - (1 - tf.reduce_sum(ea*en, 1)) + MARGIN, 0))

def macro_auprc(preds, Y, idxs):
    vals = [average_precision_score(Y[:,i], preds[:,i])
            for i in idxs if Y[:,i].sum() >= 2]
    return float(np.mean(vals)) if vals else float('nan')

# Concatenated auxiliary features for Stage 2
aux_tr = np.concatenate([rna_d_tr_s, loc_d_tr_s], axis=1).astype(np.float32)
aux_te = np.concatenate([rna_d_te_s, loc_d_te_s], axis=1).astype(np.float32)

# ── 4. Stage 1: T_ψ (unchanged from v17f) ────────────────────
print("\n[Stage 1] T_ψ triplet training (δ_layer, same as v17f)...")
tf.random.set_seed(42)
T_psi    = build_T_psi()
opt_T    = optimizers.Adam(1e-3)
delta_tf = tf.constant(delta_tr_s, dtype=tf.float32)
n_trip   = len(trip_a); n_batch = max(1, n_trip // BATCH_T)

final_active = 0.0
for epoch in range(EPOCHS_T):
    perm = np.random.permutation(n_trip); el = 0.0
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
        final_active = ((1-(ea*ep2).sum(1))-(1-(ea*en2).sum(1))+MARGIN > 0).mean()
        print(f"  Epoch {epoch+1:3d} | loss={el/n_batch:.4f} | active={final_active:.2%}")

T_tr = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
T_te = T_psi.predict(delta_te_s, batch_size=1024, verbose=0)
print(f"  T_ψ done. Final active: {final_active:.2%}")

# ── 5. Stage 2: Ensemble with auxiliary features ──────────────
print("\n[Stage 2] MLP ensemble with RNA_delta + LOC_delta...")
focal_fn  = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
all_preds = []
t0 = time.time()
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm    = np.random.permutation(len(T_tr))
    n_val   = int(len(T_tr) * 0.1); val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    mlp     = build_mlp()
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
    mlp.fit(
        [T_tr[tr_idx], X_l30_tr[tr_idx], aux_tr[tr_idx]], Y_tr[tr_idx],
        validation_data=([T_tr[val_idx], X_l30_tr[val_idx], aux_tr[val_idx]], Y_tr[val_idx]),
        epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    all_preds.append(mlp.predict([T_te, X_l30_te, aux_te], batch_size=1024, verbose=0))
    print(f"  Seed {seed} done.")

preds = np.mean(all_preds, axis=0)
ap_all = macro_auprc(preds, Y_te, valid_idx)
ap_l2  = macro_auprc(preds, Y_te, l2_valid)

print(f"\n  v18a: All MF={ap_all:.4f}  L2_Struct={ap_l2:.4f}  [{time.time()-t0:.0f}s]")
np.save(f'{OUT_DIR}/v18a_preds_ensemble.npy', preds)
np.save(f'{OUT_DIR}/v18a_Y_te.npy', Y_te)

# ── 6. Per-H2-layer breakdown ─────────────────────────────────
from collections import defaultdict as dd2
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

layer2idxs = dd2(list)
for i, go in enumerate(mf_terms):
    if valid_mask[i]: layer2idxs[get_h2(go)].append(i)

V17F = {
    'L2_Structural': 0.6219, 'L2_Structural*': 0.7135,
    'L4_CellState': 0.5764, 'L1_Generic_mid': 0.7873, 'L1_Generic_high': 0.8593,
}

print(f"\n{'='*72}")
print(f"  v18a Per-H2 Breakdown  [v17f + RNA_delta + LOC_delta]")
print(f"{'='*72}")
print(f"  {'Layer':<22} {'n':>4}  {'PRISM':>7}  {'v17f':>7}  {'v18a':>7}  {'Δ(v18-v17)':>11}")
print("  " + "-" * 65)
LAYER_ORDER = ['L2_Structural', 'L2_Structural*', 'L4_CellState', 'L1_Generic_mid', 'L1_Generic_high']
for lyr in LAYER_ORDER:
    idxs = layer2idxs.get(lyr, [])
    if not idxs: continue
    prism_ap = np.mean([prism_ref[mf_terms[i]] for i in idxs if mf_terms[i] in prism_ref])
    v18a_ap  = np.mean([average_precision_score(Y_te[:,i], preds[:,i]) for i in idxs])
    v17f_ap  = V17F.get(lyr, float('nan'))
    d = v18a_ap - v17f_ap
    print(f"  {lyr:<22} {len(idxs):>4}  {prism_ap:.4f}   {v17f_ap:.4f}   {v18a_ap:.4f}   {d:>+.4f}")
print("  " + "-" * 65)
all_v18 = macro_auprc(preds, Y_te, valid_idx)
print(f"  {'ALL MF':<22} {len(valid_idx):>4}  {'0.5249':>7}   {'0.7198':>7}   {all_v18:.4f}   {all_v18-0.7198:>+.4f}")
print(f"{'='*72}")
