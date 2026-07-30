#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17e_ablation_direct_concat.py
-------------------------------
Ablation of v17d: no T_ψ, no triplet loss.

Research question: Does the T_ψ compression (640→64) and triplet training add
value over simply concatenating δ_i raw with φ(i)?

Design:
  Input: concat[δ_i(640), φ(i)(640)] = [1280]
  MLP: Dense(256,ReLU) → BN → Dropout(0.2) → Dense(128,ReLU) → Dense(82, sigmoid)
  Loss: BinaryFocalCrossentropy(γ=2)
  Ensemble: 5 seeds

If v17e ≈ v17d: T_ψ compression is unnecessary, δ alone is sufficient.
If v17e < v17d: T_ψ triplet organisation genuinely helps (especially L2_Structural).
If v17e > v17d: T_ψ compression is a bottleneck; raw δ is better.
"""

import os, json, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/v17e_ablation'
os.makedirs(OUT_DIR, exist_ok=True)

import gzip

SEEDS      = [42, 7, 13, 21, 99]
BATCH_MLP  = 512
EPOCHS_MLP = 60

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  v17e Ablation: concat[δ(640), φ(640)] — no T_ψ, no triplet")
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
# 3. Compute delta embeddings
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

delta_norms_tr = np.linalg.norm(delta_tr, axis=1)
n_zero_tr = (delta_norms_tr < 1e-6).sum()
print(f"  Train δ: mean_norm={delta_norms_tr.mean():.4f}  zero={n_zero_tr} ({n_zero_tr/len(tr_genes):.1%})")
delta_norms_te = np.linalg.norm(delta_te, axis=1)
n_zero_te = (delta_norms_te < 1e-6).sum()
print(f"  Test  δ: mean_norm={delta_norms_te.mean():.4f}  zero={n_zero_te} ({n_zero_te/len(te_sym_list):.1%})")

# Scale delta (same as v17d for fair comparison)
scaler = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler.transform(delta_te).astype(np.float32)

# ─────────────────────────────────────────────────────────────────
# 4. Concatenate [δ, φ] → 1280-dim
# ─────────────────────────────────────────────────────────────────
print("\n[4] Building concat[δ(640), φ(640)] = [1280] inputs...")
X_concat_tr = np.concatenate([delta_tr_s, X_esm_tr], axis=1).astype(np.float32)
X_concat_te = np.concatenate([delta_te_s, X_esm_te], axis=1).astype(np.float32)
print(f"  Train: {X_concat_tr.shape}  Test: {X_concat_te.shape}")

# ─────────────────────────────────────────────────────────────────
# 5. GO labels (82 MF terms — same as v17d)
# ─────────────────────────────────────────────────────────────────
print("\n[5] Loading GO labels (82 MF terms)...")
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
# 7. Model: single-stage MLP on concat[δ, φ]
# ─────────────────────────────────────────────────────────────────
def build_mlp_concat(input_dim=1280, n_go=82):
    inp = layers.Input(shape=(input_dim,), name='concat_delta_esm')
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model(inputs=inp, outputs=out, name='MLP_v17e')

# ─────────────────────────────────────────────────────────────────
# 8. Train (5-seed ensemble)
# ─────────────────────────────────────────────────────────────────
print(f"\n[6] Training MLP on concat[δ, φ] ({EPOCHS_MLP} epochs, {len(SEEDS)} seeds)...")
t0       = time.time()
n_go     = len(mf_terms)
focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

all_preds = []
for seed in SEEDS:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    perm    = np.random.permutation(len(X_concat_tr))
    n_val   = int(len(X_concat_tr) * 0.1)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]

    mlp = build_mlp_concat(input_dim=X_concat_tr.shape[1], n_go=n_go)
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
    mlp.fit(
        X_concat_tr[tr_idx], Y_tr[tr_idx],
        validation_data=(X_concat_tr[val_idx], Y_tr[val_idx]),
        epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    pred = mlp.predict(X_concat_te, batch_size=1024, verbose=0)
    all_preds.append(pred)
    print(f"  seed={seed} done")

preds_ensemble = np.mean(all_preds, axis=0)
print(f"  Ensemble: {preds_ensemble.shape}  [{time.time()-t0:.0f}s]")

# ─────────────────────────────────────────────────────────────────
# 9. Evaluation
# ─────────────────────────────────────────────────────────────────
print(f"\n[7] Evaluation...")

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

ref_brain = np.mean([v for v in prism_brain_ref.values() if v is not None])

# v17d reference
V17D_ALL = 0.6825; V17D_L2 = 0.5586; V17D_L4 = 0.7939

print(f"\n  {'='*72}")
print(f"  Ablation Comparison (MF {len(valid_idx)} valid terms, brain zero-shot):")
print(f"  {'Method':<50} {'All MF':>8}  {'L2_Str':>7}  {'L4_Cel':>7}")
print(f"  {'-'*74}")
print(f"  {'PRISM v15d (brain, ref)':<50} {ref_brain:>8.4f}  {'0.3501':>7}  {'0.7569':>7}")
print(f"  {'v17d T_ψ(δ)+Stage2[T(64)+ESM(640)]':<50} {V17D_ALL:>8.4f}  {V17D_L2:>7.4f}  {V17D_L4:>7.4f}")
print(f"  {'v17e (ablation) concat[δ(640)+ESM(640)]':<50} {macro_all:>8.4f}  {macro_l2:>7.4f}  {macro_l4:>7.4f}")
print(f"\n  Δ(v17e - PRISM):  {macro_all - ref_brain:+.4f}  L2: {macro_l2 - 0.3501:+.4f}")
print(f"  Δ(v17e - v17d):   {macro_all - V17D_ALL:+.4f}  L2: {macro_l2 - V17D_L2:+.4f}")

interpretation = ""
if macro_l2 >= V17D_L2 - 0.01:
    interpretation = "→ T_ψ compression adds little; raw δ is sufficient"
elif macro_l2 < V17D_L2 - 0.02:
    interpretation = "→ T_ψ triplet organisation genuinely helps L2_Structural"
else:
    interpretation = "→ marginal difference; T_ψ contribution borderline"
print(f"\n  Interpretation: {interpretation}")

# Per-term comparison
per_term = []
for i, go_id in enumerate(mf_terms):
    y_true = Y_te[:, i]; y_pred = preds_ensemble[:, i]
    v17e_ap = float(average_precision_score(y_true, y_pred)) if y_true.sum() >= 2 else None
    per_term.append({
        'go_id':            go_id,
        'prism_brain':      prism_brain_ref.get(go_id),
        'v17d_auprc':       None,   # fill manually if needed
        'v17e_auprc':       v17e_ap,
        'is_l2_structural': go_id in L2_TERMS,
    })

# L2 per-term breakdown
l2_v17e = [(r['go_id'], r['v17e_auprc'], r['prism_brain'])
           for r in per_term if r['is_l2_structural'] and r['v17e_auprc'] is not None]
print(f"\n  L2_Structural per-term (v17e vs PRISM):")
print(f"  {'GO ID':<15} {'v17e':>7}  {'PRISM':>7}  {'Δ':>7}")
for go_id, v17e_ap, prism_ap in sorted(l2_v17e, key=lambda x: -(x[1] or 0)):
    d = f"{v17e_ap - prism_ap:+.4f}" if prism_ap else "  N/A"
    print(f"  {go_id:<15} {v17e_ap:>7.4f}  {prism_ap if prism_ap else 0:>7.4f}  {d:>7}")

# ─────────────────────────────────────────────────────────────────
# 10. Save
# ─────────────────────────────────────────────────────────────────
results = {
    'macro_v17e_all_mf':        macro_all,
    'macro_v17e_l2_structural': macro_l2,
    'macro_v17e_l4_sample':     macro_l4,
    'prism_brain_ref':          ref_brain,
    'v17d_ref': {
        'all_mf': V17D_ALL, 'l2_structural': V17D_L2, 'l4_sample': V17D_L4
    },
    'delta_v17e_vs_prism':      macro_all - ref_brain,
    'delta_v17e_vs_v17d':       macro_all - V17D_ALL,
    'delta_l2_v17e_vs_v17d':    macro_l2  - V17D_L2,
    'architecture': {
        'method': 'direct_concat_no_Tpsi',
        'input': 'concat[delta_i(640_scaled), ESM2(640)] = 1280',
        'mlp': '1280→256(ReLU)→BN→Drop(0.2)→128(ReLU)→82(sigmoid)',
        'loss': 'BinaryFocalCrossentropy(gamma=2)',
        'triplet': 'none',
    },
    'interpretation': interpretation,
    'per_term': per_term,
}
json.dump(results, open(f'{OUT_DIR}/v17e_results.json', 'w'), indent=2)

with open(f'{OUT_DIR}/v17e_per_term.tsv', 'w') as f:
    f.write('go_id\tprism_brain\tv17e_auprc\tdelta_v17e_prism\tis_l2_structural\n')
    for r in per_term:
        pb   = f"{r['prism_brain']:.4f}"  if r['prism_brain'] is not None else 'NA'
        v17e = f"{r['v17e_auprc']:.4f}"  if r['v17e_auprc']  is not None else 'NA'
        d    = f"{r['v17e_auprc'] - r['prism_brain']:.4f}" \
               if (r['v17e_auprc'] and r['prism_brain']) else 'NA'
        f.write(f"{r['go_id']}\t{pb}\t{v17e}\t{d}\t{r['is_l2_structural']}\n")

print(f"\n  [Saved] {OUT_DIR}/v17e_results.json")
print(f"  [Saved] {OUT_DIR}/v17e_per_term.tsv")
print("\n" + "=" * 65)
print("  v17e Ablation — COMPLETE")
print("=" * 65)
