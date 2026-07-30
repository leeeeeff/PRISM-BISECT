#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17d_muscle.py
--------------
v17d delta embedding applied to muscle 18 BP GO terms.

Background:
  v17d achieved AUPRC 0.6825 on brain MF 82 terms (vs PRISM 0.5962, Δ=+0.0863)
  and L2_Structural 0.5586 (vs PRISM 0.3501, Δ=+0.2085).

  PRISM v15d_bp_clean achieved AUPRC 0.7022 on muscle 18 BP terms.
  This script asks: does the δ_i embedding also improve over PRISM on muscle?

  If YES: δ_i generalises across tissue contexts (muscle / brain).
  If NO:  L2_Structural improvement is brain/MF-specific; δ_i requires high
          between-isoform functional diversity to contribute.

Architecture: identical to v17d.
  Stage 1 — T_ψ(δ_i): triplet on muscle GO labels, cross-gene.
  Stage 2 — MLP(concat[T_ψ(64), ESM-2(640)], n_go=18).

Reference (PRISM v15d_bp_clean, 2026-05-19):
  Overall: 0.7022
  Per-term: see PRISM_MUSCLE_REF below
"""

import os, json, time, gzip
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/v17d_muscle'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS        = [42, 7, 13, 21, 99]
MARGIN       = 0.3
BATCH_T      = 512
EPOCHS_T     = 50
BATCH_MLP    = 512
EPOCHS_MLP   = 80
EMBED_DIM_T  = 64
GO_TRIPS_PER = 300
MAX_GO_TRIPS = 30000

# 18 BP muscle GO terms (from v15d_bp_clean.py)
GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0045214': 'Sarcomere organization',
    'GO:0006941': 'Muscle contraction',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0030048': 'Actin-based movement',
    'GO:0006096': 'Glycolysis',
    'GO:0007268': 'Synaptic transmission',
    'GO:0007018': 'MT-based movement',
    'GO:0031175': 'Neuron proj development',
    'GO:0030182': 'Neuron diff',
    'GO:0000226': 'MT cytoskeleton org',
}
GO_KEYS  = list(GO_TERMS.keys())
GO_NAMES = list(GO_TERMS.values())

# PRISM v15d_bp_clean reference (reports/v15_bp_clean/cross_go_18go_20260519_1914.json)
PRISM_MUSCLE_REF = {
    'GO:0007204': 0.6884,
    'GO:0045214': 0.8667,
    'GO:0006941': 0.7016,
    'GO:0006914': 0.6600,
    'GO:0043161': 0.7772,
    'GO:0007519': 0.7775,
    'GO:0042692': 0.6740,
    'GO:0055074': 0.6729,
    'GO:0007005': 0.6873,
    'GO:0007517': 0.6401,
    'GO:0032006': 0.4959,
    'GO:0030048': 0.7356,
    'GO:0006096': 0.8143,
    'GO:0007268': 0.6672,
    'GO:0007018': 0.7402,
    'GO:0031175': 0.6823,
    'GO:0030182': 0.6466,
    'GO:0000226': 0.7118,
}
PRISM_MUSCLE_MACRO = 0.7022

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  v17d Muscle — δ_i embedding on 18 BP GO terms")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Load ESM-2 embeddings
# ─────────────────────────────────────────────────────────────────
print("\n[1] Loading ESM-2 embeddings...")
X_esm_tr = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_esm_te = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
print(f"  ESM2 train: {X_esm_tr.shape}  test: {X_esm_te.shape}")

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
      f"{n_multi_tr} multi-iso ({n_multi_tr/len(gene2idxs_tr):.1%})")
print(f"  Test:  {len(te_sym_list)} isoforms, {len(gene2idxs_te)} genes, "
      f"{n_multi_te} multi-iso ({n_multi_te/len(gene2idxs_te):.1%})")

# ─────────────────────────────────────────────────────────────────
# 3. Delta embeddings
# ─────────────────────────────────────────────────────────────────
print("\n[3] Computing δ_i = φ(i) − μ_gene...")
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

delta_norms_tr = np.linalg.norm(delta_tr, axis=1)
n_zero_tr = (delta_norms_tr < 1e-6).sum()
print(f"  Train δ: mean_norm={delta_norms_tr.mean():.4f}  zero={n_zero_tr} ({n_zero_tr/len(tr_genes):.1%})")

scaler      = MaxAbsScaler()
delta_tr_s  = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s  = scaler.transform(delta_te).astype(np.float32)

# ─────────────────────────────────────────────────────────────────
# 4. Muscle GO labels (unified BP annotations)
# ─────────────────────────────────────────────────────────────────
print("\n[4] Loading muscle BP GO labels...")
BP_ANNOT_FILE = f'{ANNOT_DIR}/human_annotations_unified_bp.txt'

go_pos_syms = defaultdict(set)  # go_id → set of gene symbols
with open(BP_ANNOT_FILE) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2: continue
        sym = parts[0]
        for go_id in parts[1:]:
            go_pos_syms[go_id].add(sym)

tr_sym_set = set(tr_genes)
te_sym_set = set(te_sym_list)

def build_Y_bp(go_id, sym_list):
    pos = go_pos_syms[go_id]
    return np.array([1.0 if s in pos else 0.0 for s in sym_list], dtype=np.float32)

Y_tr = np.stack([build_Y_bp(go, tr_genes)    for go in GO_KEYS], axis=1)
Y_te = np.stack([build_Y_bp(go, te_sym_list) for go in GO_KEYS], axis=1)
valid_mask = Y_te.sum(0) >= 2
print(f"  {len(GO_KEYS)} BP terms | Y_tr: {Y_tr.shape} | valid: {valid_mask.sum()}")

for i, go_id in enumerate(GO_KEYS):
    npos_tr = int(Y_tr[:, i].sum()); npos_te = int(Y_te[:, i].sum())
    print(f"  {go_id}  {GO_NAMES[i][:22]:22s}  tr_pos={npos_tr:5d}  te_pos={npos_te:5d}  "
          f"PRISM_ref={PRISM_MUSCLE_REF.get(go_id, float('nan')):.4f}")

# ─────────────────────────────────────────────────────────────────
# 5. Triplet mining (muscle GO labels, delta space)
# ─────────────────────────────────────────────────────────────────
print("\n[5] Mining GO-label triplets on delta space (muscle terms)...")
rng = np.random.default_rng(42)

multi_iso_mask = np.array([len(gene2idxs_tr[g]) > 1 for g in tr_genes])
print(f"  Multi-isoform isoforms (δ≠0): {multi_iso_mask.sum()} / {len(tr_genes)} "
      f"({multi_iso_mask.mean():.1%})")

trip_a, trip_p, trip_n = [], [], []
for k, go_id in enumerate(GO_KEYS):
    y_k      = Y_tr[:, k]
    pos_idxs = np.where((y_k == 1) & multi_iso_mask)[0]
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
print(f"  Triplets: {len(trip_a)}")

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
# 7. Build models (identical to v17d)
# ─────────────────────────────────────────────────────────────────
def build_T_psi(delta_dim=640, embed_dim=64):
    inp = layers.Input(shape=(delta_dim,), name='delta_input')
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(embed_dim, activation='relu')(x)
    out = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1), name='l2_norm')(x)
    return models.Model(inputs=inp, outputs=out, name='T_psi_v17d_muscle')

def build_mlp_stage2(esm_dim=640, t_dim=64, n_go=18):
    inp_t   = layers.Input(shape=(t_dim,),   name='t_delta_input')
    inp_esm = layers.Input(shape=(esm_dim,), name='esm2_input')
    x = layers.Concatenate()([inp_t, inp_esm])   # [704]
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model(inputs=[inp_t, inp_esm], outputs=out, name='MLP_theta_v17d_muscle')

def triplet_loss_fn(embeddings, a, p, n, margin=0.3):
    ea = tf.gather(embeddings, a)
    ep = tf.gather(embeddings, p)
    en = tf.gather(embeddings, n)
    d_pos = 1.0 - tf.reduce_sum(ea * ep, axis=1)
    d_neg = 1.0 - tf.reduce_sum(ea * en, axis=1)
    return tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))

# ─────────────────────────────────────────────────────────────────
# 8. Stage 1: Train T_ψ on muscle δ_i
# ─────────────────────────────────────────────────────────────────
print(f"\n[6] Stage 1: T_ψ on δ_i ({EPOCHS_T} epochs, muscle BP triplets)...")
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

delta_norms_te_s = np.linalg.norm(delta_te_s, axis=1)
zero_mask_te     = delta_norms_te_s < 1e-6
t_norms_te       = np.linalg.norm(T_te, axis=1)
print(f"  T_te norm: multi-iso={t_norms_te[~zero_mask_te].mean():.4f}  "
      f"single-iso(δ=0)={t_norms_te[zero_mask_te].mean():.4f}")

# ─────────────────────────────────────────────────────────────────
# 9. Stage 2: MLP_θ on muscle 18 BP terms
# ─────────────────────────────────────────────────────────────────
print(f"\n[7] Stage 2: MLP on concat[T_ψ(64), ESM-2(640)] ({len(SEEDS)} seeds)...")
t0       = time.time()
n_go     = len(GO_KEYS)
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
print(f"\n[8] Evaluation (18 BP muscle terms)...")
valid_idx = [i for i in range(n_go) if valid_mask[i]]

auprc_list = []
per_term   = []
for i, go_id in enumerate(GO_KEYS):
    y_true = Y_te[:, i]; y_pred = preds_ensemble[:, i]
    if y_true.sum() < 2:
        per_term.append({'go_id': go_id, 'prism_ref': PRISM_MUSCLE_REF.get(go_id),
                         'v17d_muscle_auprc': None, 'delta_vs_prism': None})
        continue
    ap = float(average_precision_score(y_true, y_pred))
    auprc_list.append(ap)
    delta_vs_prism = ap - PRISM_MUSCLE_REF.get(go_id, float('nan'))
    per_term.append({'go_id': go_id, 'go_name': GO_NAMES[i],
                     'prism_ref': PRISM_MUSCLE_REF.get(go_id),
                     'v17d_muscle_auprc': ap,
                     'delta_vs_prism': delta_vs_prism})

macro_v17d_muscle = float(np.mean(auprc_list))
n_improved  = sum(1 for r in per_term if r['delta_vs_prism'] is not None and r['delta_vs_prism'] > 0)
n_regressed = sum(1 for r in per_term if r['delta_vs_prism'] is not None and r['delta_vs_prism'] < 0)

print(f"\n  {'='*70}")
print(f"  Muscle 18 BP Results:")
print(f"  {'Method':<45} {'Macro AUPRC':>12}")
print(f"  {'-'*58}")
print(f"  {'PRISM v15d_bp_clean (2026-05-19)':<45} {PRISM_MUSCLE_MACRO:>12.4f}")
print(f"  {'v17d_muscle (δ embedding)':<45} {macro_v17d_muscle:>12.4f}")
print(f"\n  Δ(v17d_muscle - PRISM): {macro_v17d_muscle - PRISM_MUSCLE_MACRO:+.4f}")
print(f"  Improved: {n_improved}/{len(auprc_list)} terms")
print(f"  Regressed: {n_regressed}/{len(auprc_list)} terms")

print(f"\n  Per-term breakdown:")
print(f"  {'GO ID':<14} {'Name':<24} {'PRISM':>7}  {'v17d':>7}  {'Δ':>7}")
print(f"  {'-'*65}")
for r in sorted(per_term, key=lambda x: -(x['delta_vs_prism'] or -99)):
    if r['v17d_muscle_auprc'] is None: continue
    print(f"  {r['go_id']:<14} {r.get('go_name','')[:24]:<24} "
          f"{r['prism_ref']:>7.4f}  {r['v17d_muscle_auprc']:>7.4f}  "
          f"{r['delta_vs_prism']:>+7.4f}")

# Generalisability verdict
if macro_v17d_muscle > PRISM_MUSCLE_MACRO + 0.01:
    verdict = "δ embedding generalises to muscle (BP) — tissue-agnostic improvement"
elif macro_v17d_muscle > PRISM_MUSCLE_MACRO - 0.01:
    verdict = "δ embedding neutral on muscle — neither improved nor degraded"
else:
    verdict = "δ embedding degrades muscle performance — brain/MF-specific benefit"
print(f"\n  Generalisability verdict: {verdict}")

# ─────────────────────────────────────────────────────────────────
# 11. Save
# ─────────────────────────────────────────────────────────────────
results = {
    'macro_v17d_muscle':       macro_v17d_muscle,
    'prism_muscle_ref':        PRISM_MUSCLE_MACRO,
    'delta_vs_prism':          macro_v17d_muscle - PRISM_MUSCLE_MACRO,
    'n_improved':              n_improved,
    'n_regressed':             n_regressed,
    'n_terms':                 len(auprc_list),
    'generalisability_verdict': verdict,
    'brain_comparison': {
        'v17d_brain_all_mf':       0.6825,
        'v17d_brain_l2_structural': 0.5586,
        'prism_brain_ref':          0.5962,
    },
    'architecture': {
        'T_psi_input': 'delta_i = phi(i) - gene_mean_phi (640-dim)',
        'T_psi_dims':  '640→256→64(L2-norm)',
        'stage2_input': 'concat[T_psi(delta)(64), ESM-2(640)] = 704',
        'triplet_type': 'muscle BP GO-label, cross-gene',
        'n_triplets':   int(len(trip_a)),
    },
    'per_term': per_term,
}
json.dump(results, open(f'{OUT_DIR}/v17d_muscle_results.json', 'w'), indent=2)

with open(f'{OUT_DIR}/v17d_muscle_per_term.tsv', 'w') as f:
    f.write('go_id\tgo_name\tprism_ref\tv17d_muscle\tdelta_vs_prism\n')
    for r in per_term:
        prism = f"{r['prism_ref']:.4f}"   if r['prism_ref']          is not None else 'NA'
        v17d  = f"{r['v17d_muscle_auprc']:.4f}" if r['v17d_muscle_auprc'] is not None else 'NA'
        delta = f"{r['delta_vs_prism']:.4f}"    if r['delta_vs_prism']    is not None else 'NA'
        f.write(f"{r['go_id']}\t{r.get('go_name','')}\t{prism}\t{v17d}\t{delta}\n")

print(f"\n  [Saved] {OUT_DIR}/v17d_muscle_results.json")
print(f"  [Saved] {OUT_DIR}/v17d_muscle_per_term.tsv")
print("\n" + "=" * 65)
print("  v17d Muscle — COMPLETE")
print("=" * 65)
