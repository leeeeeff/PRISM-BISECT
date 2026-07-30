#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_splice_diagnostic.py
--------------------------
Two diagnostic tests for the "δ_layer is splice-specific" hypothesis.

Test A: δ amplifies within-gene differences more than between-gene differences
  For same-gene isoform pairs: d_δ vs d_L30 vs d_L15
  If δ is splice-specific: d_δ > d_L30 for same-gene pairs
  (δ disproportionately amplifies splice-driven intra-gene variation)

Test B: Random-δ ablation control
  Replace δ_layer with isotropic noise of same per-sample L2 norm
  Run the full T_ψ + Stage 2 pipeline
  If random-δ performance ≈ v17f: improvement is model capacity, not splice signal
  If random-δ performance ≈ PRISM: δ_layer carries genuine information

Test C: Sequence length difference as splice proxy
  For same-gene pairs: |len_i - len_j| / max(len_i, len_j) correlates with d_δ?
  Larger splice difference → larger δ difference expected

V17f reference: All MF 0.7171 / L2_Struct 0.6156
PRISM reference: All MF 0.5962 / L2_Struct 0.3501
"""

import os, json, gzip, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
from scipy.stats import pearsonr, spearmanr
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
SEQ_DIR   = '../data/raw_data/data/sequences'
OUT_DIR   = '../../reports/v17f_splice_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS     = [42, 7, 13, 21, 99]
MARGIN    = 0.3
BATCH_T   = 512
EPOCHS_T  = 50
BATCH_MLP = 512
EPOCHS_MLP = 60
EMBED_DIM_T = 64
MAX_GO_TRIPS = 30000
GO_TRIPS_PER = 300

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  v17f Splice Diagnostic")
print("  Tests: δ amplification + random-δ ablation + seq-length proxy")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Load embeddings and sequences
# ─────────────────────────────────────────────────────────────────
print("\n[1] Loading embeddings...")

X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X_l15_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer15_t30_150M.npy').astype(np.float32)
X_l30_te = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)
X_l15_te = np.load(f'{DATA_DIR}/esm2_layer_15_t30_150M.npy').astype(np.float32)

delta_tr = (X_l30_tr - X_l15_tr).astype(np.float32)
delta_te = (X_l30_te - X_l15_te).astype(np.float32)

scaler = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler.transform(delta_te).astype(np.float32)

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes = [clean(g) for g in tr_genes_raw]

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]
gene_arr_tr  = np.array(tr_genes)
gene_arr_te  = np.array(te_sym_list)

gene2idxs_te = defaultdict(list)
for i, g in enumerate(te_sym_list):
    gene2idxs_te[g].append(i)

multi_genes = {g: v for g, v in gene2idxs_te.items() if len(v) >= 2}
print(f"  Test: {len(te_sym_list)} isoforms | multi-iso genes: {len(multi_genes)} | "
      f"isoforms in multi: {sum(len(v) for v in multi_genes.values())}")

# ─────────────────────────────────────────────────────────────────
# Test A: Within-gene vs between-gene distance comparison
# ─────────────────────────────────────────────────────────────────
print("\n[A] δ Within-Gene vs Between-Gene Distance Analysis")
print("─" * 60)

def cos_dist(u, v):
    """Cosine distance = 1 - cosine_similarity."""
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-10 or nv < 1e-10: return 1.0
    return 1.0 - float(np.dot(u, v) / (nu * nv))

# Sample within-gene pairs (all pairs in multi-iso genes)
within_pairs = []
for g, idxs in multi_genes.items():
    for ii in range(len(idxs)):
        for jj in range(ii + 1, len(idxs)):
            within_pairs.append((idxs[ii], idxs[jj]))
within_pairs = within_pairs[:5000]  # cap for speed

# Sample between-gene pairs (same count, random)
rng = np.random.default_rng(42)
all_idxs = np.arange(len(te_sym_list))
between_pairs = []
genes_used = set()
attempts = 0
while len(between_pairs) < len(within_pairs) and attempts < 50000:
    i, j = rng.choice(all_idxs, 2, replace=False)
    if te_sym_list[i] != te_sym_list[j]:
        between_pairs.append((int(i), int(j)))
    attempts += 1
between_pairs = between_pairs[:len(within_pairs)]

print(f"  Pairs: {len(within_pairs)} within-gene, {len(between_pairs)} between-gene")

def compute_pair_distances(pairs, X_l30, X_l15, delta):
    d_l30 = []; d_l15 = []; d_dlt = []
    for (i, j) in pairs:
        d_l30.append(cos_dist(X_l30[i], X_l30[j]))
        d_l15.append(cos_dist(X_l15[i], X_l15[j]))
        d_dlt.append(cos_dist(delta[i], delta[j]))
    return np.array(d_l30), np.array(d_l15), np.array(d_dlt)

w_l30, w_l15, w_dlt = compute_pair_distances(within_pairs, X_l30_te, X_l15_te, delta_te)
b_l30, b_l15, b_dlt = compute_pair_distances(between_pairs, X_l30_te, X_l15_te, delta_te)

print(f"\n  WITHIN-GENE pairs (same gene, different isoform):")
print(f"    d(L30):   mean={w_l30.mean():.4f}  std={w_l30.std():.4f}")
print(f"    d(L15):   mean={w_l15.mean():.4f}  std={w_l15.std():.4f}")
print(f"    d(δ):     mean={w_dlt.mean():.4f}  std={w_dlt.std():.4f}")
print(f"\n  BETWEEN-GENE pairs (different genes):")
print(f"    d(L30):   mean={b_l30.mean():.4f}  std={b_l30.std():.4f}")
print(f"    d(L15):   mean={b_l15.mean():.4f}  std={b_l15.std():.4f}")
print(f"    d(δ):     mean={b_dlt.mean():.4f}  std={b_dlt.std():.4f}")

# Key ratio: within/between distance ratio
# Lower ratio = distances more compressed within gene (shared scaffold dominates)
# If δ is splice-specific: δ within/between ratio should be HIGHER than L30's ratio
#   (δ preserves more within-gene discrimination)
print(f"\n  Within/Between distance ratio (higher = more discriminative within genes):")
print(f"    d(L30):  {w_l30.mean()/b_l30.mean():.4f}")
print(f"    d(L15):  {w_l15.mean()/b_l15.mean():.4f}")
print(f"    d(δ):    {w_dlt.mean()/b_dlt.mean():.4f}")
print(f"\n  Interpretation:")
print(f"    If d(δ) within/between HIGHER than d(L30): δ amplifies within-gene differences")
print(f"    If d(δ) within/between LOWER: δ compresses within-gene differences")

# ─────────────────────────────────────────────────────────────────
# Test C: L30 norm difference as protein size proxy
# (Brain dataset has no raw sequence files; use L30 embedding norm
#  as proxy for sequence complexity — larger proteins → larger norms)
# ─────────────────────────────────────────────────────────────────
print("\n[C] L30 Norm Difference vs δ Distance (size proxy for splice)")
print("─" * 60)

l30_norms = np.linalg.norm(X_l30_te, axis=1)

splice_proxy = []
delta_dists  = []
for g, idxs in multi_genes.items():
    for ii in range(len(idxs)):
        for jj in range(ii + 1, len(idxs)):
            i, j = idxs[ii], idxs[jj]
            n_i, n_j = l30_norms[i], l30_norms[j]
            if n_i > 0 and n_j > 0:
                rel_diff = abs(n_i - n_j) / max(n_i, n_j)
                splice_proxy.append(rel_diff)
                delta_dists.append(cos_dist(delta_te[i], delta_te[j]))

splice_proxy = np.array(splice_proxy)
delta_dists  = np.array(delta_dists)
print(f"  Pairs analyzed: {len(splice_proxy)}")
print(f"  L30 norm diff: mean={splice_proxy.mean():.3f}  std={splice_proxy.std():.3f}")
print(f"  δ distance:    mean={delta_dists.mean():.3f}  std={delta_dists.std():.3f}")

r_pearson, p_pearson = pearsonr(splice_proxy, delta_dists)
r_spearman, p_spearman = spearmanr(splice_proxy, delta_dists)
print(f"\n  Pearson r(seq_len_diff, δ_dist)  = {r_pearson:.4f}  p={p_pearson:.4e}")
print(f"  Spearman ρ(seq_len_diff, δ_dist) = {r_spearman:.4f}  p={p_spearman:.4e}")
print(f"\n  Interpretation:")
print(f"    r > 0: longer splice difference → larger δ distance (splice-specific)")
print(f"    r ≈ 0: δ distance unrelated to splice difference (generic)")

# ─────────────────────────────────────────────────────────────────
# 2. Build GO labels for Test B
# ─────────────────────────────────────────────────────────────────
print("\n[2] Building GO labels for Test B (random-δ ablation)...")

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
tr_ids    = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_ids)

with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        gid, go_id, cat_raw = p[1], p[2], p[7]
        if cat_raw != 'Function': continue
        go_genes_all[go_id].add(gid)
        if gid in tr_id_set: go_genes_tr[go_id].add(gid)

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])

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

# Fixed triplets
rng_trip = np.random.default_rng(42)
trip_a, trip_p, trip_n = [], [], []
for k, go_id in enumerate(mf_terms):
    y_k = Y_tr[:, k]
    pos_idxs = np.where(y_k == 1)[0]; neg_idxs = np.where(y_k == 0)[0]
    if len(pos_idxs) < 5 or len(neg_idxs) < 10: continue
    if len(trip_a) >= MAX_GO_TRIPS: break
    n_anchor = min(GO_TRIPS_PER, len(pos_idxs))
    for a_idx in rng_trip.choice(pos_idxs, n_anchor, replace=False):
        a_gene = tr_genes[a_idx]
        cross_pos = pos_idxs[gene_arr_tr[pos_idxs] != a_gene]
        cross_neg = neg_idxs[gene_arr_tr[neg_idxs] != a_gene]
        if len(cross_pos) < 2 or len(cross_neg) < 2: continue
        trip_a.append(a_idx); trip_p.append(int(rng_trip.choice(cross_pos))); trip_n.append(int(rng_trip.choice(cross_neg)))
trip_a = np.array(trip_a, dtype=np.int32)
trip_p = np.array(trip_p, dtype=np.int32)
trip_n = np.array(trip_n, dtype=np.int32)
print(f"  Triplets: {len(trip_a)}")

# ─────────────────────────────────────────────────────────────────
# 3. TensorFlow setup
# ─────────────────────────────────────────────────────────────────
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

def run_pipeline(delta_tr_input, delta_te_input, label):
    """Full T_ψ + Stage 2 pipeline for a given delta."""
    t0 = time.time()
    tf.random.set_seed(42)
    T_psi   = build_T_psi()
    opt_T   = optimizers.Adam(1e-3)
    dlt_tf  = tf.constant(delta_tr_input, dtype=tf.float32)
    n_trip  = len(trip_a)
    n_batch = max(1, n_trip // BATCH_T)

    final_active = 0.0
    for epoch in range(EPOCHS_T):
        perm = np.random.permutation(n_trip)
        el   = 0.0
        for b in range(n_batch):
            bi = perm[b*BATCH_T:(b+1)*BATCH_T]
            with tf.GradientTape() as tape:
                embs = T_psi(dlt_tf, training=True)
                loss = triplet_loss_fn(embs, trip_a[bi], trip_p[bi], trip_n[bi])
            grads = tape.gradient(loss, T_psi.trainable_variables)
            opt_T.apply_gradients(zip(grads, T_psi.trainable_variables))
            el += float(loss)
        if (epoch + 1) % 10 == 0:
            embs_np = T_psi.predict(delta_tr_input, batch_size=1024, verbose=0)
            ea = embs_np[trip_a]; ep2 = embs_np[trip_p]; en2 = embs_np[trip_n]
            final_active = ((1-(ea*ep2).sum(1)) - (1-(ea*en2).sum(1)) + MARGIN > 0).mean()
            print(f"  [{label}] Epoch {epoch+1:3d} | loss={el/n_batch:.4f} | active={final_active:.2%}")

    T_tr = T_psi.predict(delta_tr_input, batch_size=1024, verbose=0)
    T_te = T_psi.predict(delta_te_input, batch_size=1024, verbose=0)

    focal_fn  = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
    all_preds = []
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm    = np.random.permutation(len(T_tr))
        n_val   = int(len(T_tr) * 0.1)
        val_idx = perm[:n_val]; tr_idx = perm[n_val:]
        mlp = build_mlp()
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

    preds  = np.mean(all_preds, axis=0)
    ap_all = macro_auprc(preds, Y_te, valid_idx)
    ap_l2  = macro_auprc(preds, Y_te, l2_valid)
    elapsed = time.time() - t0
    print(f"\n  [{label}] All MF={ap_all:.4f}  L2_Struct={ap_l2:.4f}  active={final_active:.2%}  [{elapsed:.0f}s]")
    tf.keras.backend.clear_session()
    return ap_all, ap_l2, float(final_active)

# ─────────────────────────────────────────────────────────────────
# Test B1: Real δ_layer (replication)
# ─────────────────────────────────────────────────────────────────
print("\n[B1] Real δ_layer (v17f replication, L30-L15)")
print("─" * 60)
b1_all, b1_l2, b1_act = run_pipeline(delta_tr_s, delta_te_s, "real_δ")

# ─────────────────────────────────────────────────────────────────
# Test B2: Random δ (same per-sample L2 norm, random direction)
# ─────────────────────────────────────────────────────────────────
print("\n[B2] Random δ ablation (same norm, random direction)")
print("─" * 60)

rng_rand = np.random.default_rng(999)
rand_tr  = rng_rand.standard_normal(delta_tr_s.shape).astype(np.float32)
rand_te  = rng_rand.standard_normal(delta_te_s.shape).astype(np.float32)

# Match per-sample norm of real delta
real_norms_tr = np.linalg.norm(delta_tr_s, axis=1, keepdims=True)
real_norms_te = np.linalg.norm(delta_te_s, axis=1, keepdims=True)
rand_tr = rand_tr / (np.linalg.norm(rand_tr, axis=1, keepdims=True) + 1e-8) * real_norms_tr
rand_te = rand_te / (np.linalg.norm(rand_te, axis=1, keepdims=True) + 1e-8) * real_norms_te

print(f"  Random δ train: mean_norm={np.linalg.norm(rand_tr, axis=1).mean():.3f}  "
      f"(vs real: {np.linalg.norm(delta_tr_s, axis=1).mean():.3f})")

b2_all, b2_l2, b2_act = run_pipeline(rand_tr, rand_te, "rand_δ")

# ─────────────────────────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────────────────────────
PRISM_ALL = 0.5962; PRISM_L2 = 0.3501
V17F_ALL  = 0.7171; V17F_L2  = 0.6156

print(f"\n\n{'='*70}")
print(f"  SPLICE DIAGNOSTIC — FINAL RESULTS")
print(f"{'='*70}")

print(f"\n  [A] Within-Gene vs Between-Gene δ Distance Ratio")
print(f"  ─────────────────────────────────────────────────")
print(f"  Space   | within mean | between mean | ratio (w/b)")
print(f"  --------|-------------|--------------|------------")
print(f"  d(L15)  | {w_l15.mean():.4f}      | {b_l15.mean():.4f}       | {w_l15.mean()/b_l15.mean():.4f}")
print(f"  d(L30)  | {w_l30.mean():.4f}      | {b_l30.mean():.4f}       | {w_l30.mean()/b_l30.mean():.4f}")
print(f"  d(δ)    | {w_dlt.mean():.4f}      | {b_dlt.mean():.4f}       | {w_dlt.mean()/b_dlt.mean():.4f}")

if w_dlt.mean()/b_dlt.mean() > w_l30.mean()/b_l30.mean():
    print(f"  → δ has HIGHER within/between ratio than L30: δ amplifies intra-gene variation ✓")
else:
    print(f"  → δ has LOWER/EQUAL within/between ratio: δ does NOT preferentially amplify splice differences ✗")

print(f"\n  [C] Sequence Length Difference vs δ Distance (splice proxy)")
print(f"  ─────────────────────────────────────────────────────────────")
print(f"  Pearson r  = {r_pearson:.4f}  (p={p_pearson:.2e})")
print(f"  Spearman ρ = {r_spearman:.4f}  (p={p_spearman:.2e})")
if r_pearson > 0.05 and p_pearson < 0.05:
    print(f"  → Positive correlation: larger splice difference → larger δ distance ✓")
else:
    print(f"  → No significant correlation: δ distance does not track sequence-level splice differences ✗")

print(f"\n  [B] Random-δ Ablation Control")
print(f"  ────────────────────────────────────────────────────────────")
print(f"  Model                | All MF  | L2_Struct | active%")
print(f"  ---------------------|---------|-----------|--------")
print(f"  PRISM v15d (ref)     | {PRISM_ALL:.4f} | {PRISM_L2:.4f}   |  —")
print(f"  v17f (real δ)        | {V17F_ALL:.4f} | {V17F_L2:.4f}   |  80-85%")
print(f"  B1: δ replication    | {b1_all:.4f} | {b1_l2:.4f}   |  {b1_act:.1%}")
print(f"  B2: random δ         | {b2_all:.4f} | {b2_l2:.4f}   |  {b2_act:.1%}")

gap_b1_b2_all = b1_all - b2_all
gap_b1_b2_l2  = b1_l2  - b2_l2
print(f"\n  Real δ vs Random δ: All MF Δ={gap_b1_b2_all:+.4f}  L2 Δ={gap_b1_b2_l2:+.4f}")

if gap_b1_b2_all > 0.02:
    print(f"  → δ_layer carries genuine directional information beyond model capacity ✓")
    print(f"     'splice-specific' hypothesis SUPPORTED")
elif gap_b1_b2_all > 0:
    print(f"  → δ_layer modestly outperforms random: partial splice-specific signal")
    print(f"     'splice-specific' hypothesis WEAKLY SUPPORTED")
else:
    print(f"  → Random δ ≥ real δ: improvement is purely model capacity ✗")
    print(f"     'splice-specific' hypothesis REJECTED — rethink interpretation")

results = {
    'test_A': {
        'within_gene': {'d_l15': float(w_l15.mean()), 'd_l30': float(w_l30.mean()), 'd_delta': float(w_dlt.mean())},
        'between_gene': {'d_l15': float(b_l15.mean()), 'd_l30': float(b_l30.mean()), 'd_delta': float(b_dlt.mean())},
        'within_between_ratio': {'l15': float(w_l15.mean()/b_l15.mean()), 'l30': float(w_l30.mean()/b_l30.mean()), 'delta': float(w_dlt.mean()/b_dlt.mean())},
    },
    'test_C': {
        'pearson_r': float(r_pearson), 'pearson_p': float(p_pearson),
        'spearman_r': float(r_spearman), 'spearman_p': float(p_spearman),
        'n_pairs': len(splice_proxy),
    },
    'test_B': {
        'real_delta': {'all_mf': b1_all, 'l2_struct': b1_l2, 'active': b1_act},
        'random_delta': {'all_mf': b2_all, 'l2_struct': b2_l2, 'active': b2_act},
        'delta_vs_random': {'all_mf': gap_b1_b2_all, 'l2_struct': gap_b1_b2_l2},
    }
}

json.dump(results, open(f'{OUT_DIR}/diagnostic_results.json', 'w'), indent=2)
print(f"\n  [Saved] {OUT_DIR}/diagnostic_results.json")
print("=" * 70)
print("  v17f Splice Diagnostic — COMPLETE")
print("=" * 70)
