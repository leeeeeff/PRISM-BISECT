#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17d_bootstrap_ci.py
---------------------
Bootstrap 95% CI for v17d AUPRC improvement claims.

Validates two claims in natcomm_v0.md [R9.4]:
  1. All MF 82 terms: v17d AUPRC 0.6825 vs PRISM 0.5962 (Δ=+0.0863)
  2. L2_Structural 33 terms: v17d AUPRC 0.5586 vs PRISM 0.3501 (Δ=+0.2085)

Method:
  - Re-run v17d (fixed seeds → reproducible preds)
  - Bootstrap B=1000 iterations at GENE level (conservative; accounts for
    within-gene correlation between isoforms)
  - Report 2.5%–97.5% percentile CI for AUPRC and for Δ(v17d-PRISM)

Gene-level bootstrap: sample test genes with replacement, include ALL
isoforms for each sampled gene. This preserves within-gene structure and
is the appropriate unit of independence (gene-grouped CV).
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
OUT_DIR   = '../../reports/v17d_bootstrap'
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
B_BOOTSTRAP  = 1000
BOOTSTRAP_SEED = 0

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  v17d Bootstrap CI  (gene-level, B=1000)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1-3. Data loading (identical to v17d)
# ─────────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
X_esm_tr = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_esm_te = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

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
print(f"  Test:  {len(te_sym_list)} isoforms, {len(gene2idxs_te)} genes "
      f"({len(te_unique_genes)} unique)")

print("\n[2] Computing delta embeddings...")
delta_tr = np.zeros_like(X_esm_tr)
for gene, idxs in gene2idxs_tr.items():
    gm = X_esm_tr[idxs].mean(axis=0)
    for i in idxs: delta_tr[i] = X_esm_tr[i] - gm

delta_te = np.zeros_like(X_esm_te)
for gene, idxs in gene2idxs_te.items():
    gm = X_esm_te[idxs].mean(axis=0)
    for i in idxs: delta_te[i] = X_esm_te[i] - gm

scaler     = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler.transform(delta_te).astype(np.float32)

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
l2_idx    = [i for i, go in enumerate(mf_terms) if go in L2_TERMS and valid_mask[i]]
valid_idx = [i for i in range(len(mf_terms)) if valid_mask[i]]

print(f"  {len(mf_terms)} MF terms | valid: {len(valid_idx)} | L2_Structural: {len(l2_idx)}")

print("\n[4] Mining triplets...")
rng = np.random.default_rng(42)
multi_iso_mask = np.array([len(gene2idxs_tr[g]) > 1 for g in tr_genes])
trip_a, trip_p, trip_n = [], [], []
for k, go_id in enumerate(mf_terms):
    y_k      = Y_tr[:, k]
    pos_idxs = np.where((y_k == 1) & multi_iso_mask)[0]
    neg_idxs = np.where(y_k == 0)[0]
    if len(pos_idxs) < 5 or len(neg_idxs) < 10: continue
    if len(trip_a) >= MAX_GO_TRIPS: break
    for a_idx in rng.choice(pos_idxs, min(GO_TRIPS_PER, len(pos_idxs)), replace=False):
        a_gene    = tr_genes[a_idx]
        cross_pos = pos_idxs[gene_arr_tr[pos_idxs] != a_gene]
        cross_neg = neg_idxs[gene_arr_tr[neg_idxs] != a_gene]
        if len(cross_pos) < 2 or len(cross_neg) < 2: continue
        trip_a.append(a_idx); trip_p.append(int(rng.choice(cross_pos)))
        trip_n.append(int(rng.choice(cross_neg)))
trip_a = np.array(trip_a, dtype=np.int32)
trip_p = np.array(trip_p, dtype=np.int32)
trip_n = np.array(trip_n, dtype=np.int32)
print(f"  {len(trip_a)} triplets")

# ─────────────────────────────────────────────────────────────────
# 5. TF + models (identical to v17d)
# ─────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"\n  GPU: {gpus[0].name}")

def build_T_psi(delta_dim=640, embed_dim=64):
    inp = layers.Input(shape=(delta_dim,))
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(embed_dim, activation='relu')(x)
    out = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    return models.Model(inputs=inp, outputs=out)

def build_mlp_stage2(esm_dim=640, t_dim=64, n_go=82):
    inp_t   = layers.Input(shape=(t_dim,))
    inp_esm = layers.Input(shape=(esm_dim,))
    x = layers.Concatenate()([inp_t, inp_esm])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model(inputs=[inp_t, inp_esm], outputs=out)

def triplet_loss_fn(embeddings, a, p, n, margin=0.3):
    ea = tf.gather(embeddings, a); ep = tf.gather(embeddings, p); en = tf.gather(embeddings, n)
    d_pos = 1.0 - tf.reduce_sum(ea * ep, axis=1)
    d_neg = 1.0 - tf.reduce_sum(ea * en, axis=1)
    return tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))

# ─────────────────────────────────────────────────────────────────
# 6. Stage 1: T_ψ (identical to v17d)
# ─────────────────────────────────────────────────────────────────
print(f"\n[5] Stage 1: T_ψ ({EPOCHS_T} epochs)...")
t0 = time.time()
tf.random.set_seed(42)
T_psi    = build_T_psi(delta_dim=delta_tr_s.shape[1], embed_dim=EMBED_DIM_T)
opt_T    = optimizers.Adam(1e-3)
delta_tf = tf.constant(delta_tr_s, dtype=tf.float32)
n_batches = max(1, len(trip_a) // BATCH_T)
for epoch in range(EPOCHS_T):
    perm = np.random.permutation(len(trip_a))
    for b in range(n_batches):
        bidx = perm[b*BATCH_T:(b+1)*BATCH_T]
        ba = trip_a[bidx]; bp = trip_p[bidx]; bn = trip_n[bidx]
        with tf.GradientTape() as tape:
            embs = T_psi(delta_tf, training=True)
            loss = triplet_loss_fn(embs, ba, bp, bn, MARGIN)
        grads = tape.gradient(loss, T_psi.trainable_variables)
        opt_T.apply_gradients(zip(grads, T_psi.trainable_variables))
    if (epoch + 1) % 10 == 0:
        embs_np = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
        ea = embs_np[trip_a]; ep_np = embs_np[trip_p]; en_np = embs_np[trip_n]
        d_pos = 1 - (ea * ep_np).sum(1); d_neg = 1 - (ea * en_np).sum(1)
        print(f"  Epoch {epoch+1:3d} | active={((d_pos - d_neg + MARGIN) > 0).mean():.2%}")
print(f"  T_ψ: {time.time()-t0:.0f}s")

T_tr = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
T_te = T_psi.predict(delta_te_s, batch_size=1024, verbose=0)

# ─────────────────────────────────────────────────────────────────
# 7. Stage 2: 5-seed ensemble (identical to v17d)
# ─────────────────────────────────────────────────────────────────
print(f"\n[6] Stage 2: 5-seed ensemble ({EPOCHS_MLP} epochs)...")
t0       = time.time()
n_go     = len(mf_terms)
focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
all_preds = []
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm = np.random.permutation(len(T_tr))
    n_val = int(len(T_tr) * 0.1)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    mlp = build_mlp_stage2(esm_dim=X_esm_tr.shape[1], t_dim=EMBED_DIM_T, n_go=n_go)
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
    mlp.fit([T_tr[tr_idx], X_esm_tr[tr_idx]], Y_tr[tr_idx],
            validation_data=([T_tr[val_idx], X_esm_tr[val_idx]], Y_tr[val_idx]),
            epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0)
    pred = mlp.predict([T_te, X_esm_te], batch_size=1024, verbose=0)
    all_preds.append(pred)
    print(f"  seed={seed} done")
preds_v17d = np.mean(all_preds, axis=0)
print(f"  Ensemble: {preds_v17d.shape}  [{time.time()-t0:.0f}s]")

# Save preds for future use
np.save(f'{OUT_DIR}/v17d_preds_ensemble.npy', preds_v17d)
np.save(f'{OUT_DIR}/Y_te.npy', Y_te)
print(f"  [Saved] predictions to {OUT_DIR}/")

# ─────────────────────────────────────────────────────────────────
# 8. Point estimates (verify reproduction matches original)
# ─────────────────────────────────────────────────────────────────
def macro_auprc(preds, Y, term_indices):
    auprcs = []
    for i in term_indices:
        yt = Y[:, i]; yp = preds[:, i]
        if yt.sum() >= 2:
            auprcs.append(average_precision_score(yt, yp))
    return float(np.mean(auprcs)) if auprcs else float('nan')

# PRISM point estimates from reference TSV
prism_preds_ref = {}

point_v17d_all = macro_auprc(preds_v17d, Y_te, valid_idx)
point_v17d_l2  = macro_auprc(preds_v17d, Y_te, l2_idx)
point_prism_all = np.mean([v for v in prism_brain_ref.values() if v is not None])
point_prism_l2  = np.mean([prism_brain_ref[mf_terms[i]]
                            for i in l2_idx if prism_brain_ref.get(mf_terms[i])])

print(f"\n[7] Point estimates (verify vs original):")
print(f"  v17d  All MF:      {point_v17d_all:.4f}  (original: 0.6825)")
print(f"  v17d  L2_Struct:   {point_v17d_l2:.4f}  (original: 0.5586)")
print(f"  PRISM All MF ref:  {point_prism_all:.4f}  (original: 0.5962)")
print(f"  PRISM L2 ref:      {point_prism_l2:.4f}  (original: 0.3501)")

# ─────────────────────────────────────────────────────────────────
# 9. Gene-level bootstrap CI
# ─────────────────────────────────────────────────────────────────
print(f"\n[8] Gene-level bootstrap CI (B={B_BOOTSTRAP})...")
t0  = time.time()
rng = np.random.default_rng(BOOTSTRAP_SEED)

# Build gene → isoform index mapping for test set
te_gene_list = te_unique_genes   # N_genes
gene_to_te_idx = {g: np.array(idxs) for g, idxs in gene2idxs_te.items()}
N_genes = len(te_gene_list)

boot_all_v17d  = np.zeros(B_BOOTSTRAP)
boot_l2_v17d   = np.zeros(B_BOOTSTRAP)
boot_all_prism = np.zeros(B_BOOTSTRAP)
boot_l2_prism  = np.zeros(B_BOOTSTRAP)

# PRISM per-isoform predictions (gene-mean prediction, same for all isoforms in gene)
# For PRISM, we approximate per-isoform predictions from known per-term AUPRCs.
# Since we don't have PRISM's actual isoform-level predictions, we use Y_te labels
# + PRISM AUPRC as a reference and compute CI for v17d improvement (delta).
# Alternative: bootstrap only v17d, report CI for v17d AUPRC itself.

# We report:
# (A) CI for v17d AUPRC (gene-level bootstrap of v17d predictions)
# (B) CI for Δ(v17d - PRISM): this requires PRISM isoform predictions.
#     Since PRISM predictions are unavailable, we report (A) only, and note that
#     Δ CI requires paired predictions.

for b in range(B_BOOTSTRAP):
    # Sample N_genes genes with replacement
    sampled_genes = rng.choice(te_gene_list, size=N_genes, replace=True)
    # Build isoform index array (with repetitions)
    boot_idxs = np.concatenate([gene_to_te_idx[g] for g in sampled_genes])

    Y_boot    = Y_te[boot_idxs]
    P_v17d_b  = preds_v17d[boot_idxs]

    # Macro AUPRC over valid terms (valid_mask still applies)
    all_auprcs = []
    l2_auprcs  = []
    for i in valid_idx:
        yt = Y_boot[:, i]; yp = P_v17d_b[:, i]
        if yt.sum() >= 2 and (yt != yt[0]).any():  # need at least 2 classes
            ap = average_precision_score(yt, yp)
            all_auprcs.append(ap)
            if i in set(l2_idx): l2_auprcs.append(ap)

    boot_all_v17d[b] = np.mean(all_auprcs) if all_auprcs else float('nan')
    boot_l2_v17d[b]  = np.mean(l2_auprcs)  if l2_auprcs  else float('nan')

    if (b + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta     = elapsed / (b + 1) * (B_BOOTSTRAP - b - 1)
        print(f"  Bootstrap {b+1:4d}/{B_BOOTSTRAP}  "
              f"mean_all={np.nanmean(boot_all_v17d[:b+1]):.4f}  "
              f"mean_l2={np.nanmean(boot_l2_v17d[:b+1]):.4f}  "
              f"ETA={eta:.0f}s")

print(f"\n  Bootstrap done: {time.time()-t0:.0f}s")

# Remove NaN iterations
valid_b_all = ~np.isnan(boot_all_v17d)
valid_b_l2  = ~np.isnan(boot_l2_v17d)
print(f"  Valid bootstrap iterations: all={valid_b_all.sum()}  l2={valid_b_l2.sum()}")

ci_all_lo, ci_all_hi = np.percentile(boot_all_v17d[valid_b_all], [2.5, 97.5])
ci_l2_lo,  ci_l2_hi  = np.percentile(boot_l2_v17d[valid_b_l2],   [2.5, 97.5])

# ─────────────────────────────────────────────────────────────────
# 10. Report
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Bootstrap CI Results (gene-level, B={B_BOOTSTRAP})")
print(f"{'='*70}")
print(f"\n  {'Metric':<45} {'Point':>7}  {'95% CI':>20}")
print(f"  {'-'*73}")
print(f"  {'v17d All MF 82 terms AUPRC':<45} {point_v17d_all:>7.4f}  "
      f"[{ci_all_lo:.4f}, {ci_all_hi:.4f}]")
print(f"  {'v17d L2_Structural 33 terms AUPRC':<45} {point_v17d_l2:>7.4f}  "
      f"[{ci_l2_lo:.4f}, {ci_l2_hi:.4f}]")
print(f"\n  PRISM reference (fixed — single run, no CI):")
print(f"    All MF: {point_prism_all:.4f}")
print(f"    L2:     {point_prism_l2:.4f}")
print(f"\n  Δ(v17d - PRISM) point estimates:")
print(f"    All MF Δ:      {point_v17d_all - point_prism_all:+.4f}")
print(f"    L2 Structural: {point_v17d_l2  - point_prism_l2:+.4f}")
print(f"\n  CI lower bound > PRISM?")
print(f"    All MF:  {ci_all_lo:.4f} > {point_prism_all:.4f} → "
      f"{'YES ✓' if ci_all_lo > point_prism_all else 'NO ✗'}")
print(f"    L2:      {ci_l2_lo:.4f}  > {point_prism_l2:.4f}  → "
      f"{'YES ✓' if ci_l2_lo > point_prism_l2 else 'NO ✗'}")

# ─────────────────────────────────────────────────────────────────
# 11. Save
# ─────────────────────────────────────────────────────────────────
results = {
    'point_estimates': {
        'v17d_all_mf':       point_v17d_all,
        'v17d_l2_structural': point_v17d_l2,
        'prism_all_mf':      point_prism_all,
        'prism_l2':          point_prism_l2,
        'delta_all_mf':      point_v17d_all - point_prism_all,
        'delta_l2':          point_v17d_l2  - point_prism_l2,
    },
    'bootstrap': {
        'method':        'gene-level sampling with replacement',
        'B':             B_BOOTSTRAP,
        'valid_B_all':   int(valid_b_all.sum()),
        'valid_B_l2':    int(valid_b_l2.sum()),
        'ci_all_mf':     [round(float(ci_all_lo), 4), round(float(ci_all_hi), 4)],
        'ci_l2_structural': [round(float(ci_l2_lo), 4), round(float(ci_l2_hi), 4)],
        'ci_lower_exceeds_prism_all': bool(ci_all_lo > point_prism_all),
        'ci_lower_exceeds_prism_l2':  bool(ci_l2_lo  > point_prism_l2),
    },
    'boot_all_v17d':  boot_all_v17d[valid_b_all].tolist(),
    'boot_l2_v17d':   boot_l2_v17d[valid_b_l2].tolist(),
}
json.dump(results, open(f'{OUT_DIR}/v17d_bootstrap_ci.json', 'w'), indent=2)
print(f"\n  [Saved] {OUT_DIR}/v17d_bootstrap_ci.json")
print("\n" + "=" * 65)
print("  v17d Bootstrap CI — COMPLETE")
print("=" * 65)
