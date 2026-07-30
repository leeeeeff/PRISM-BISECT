#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_b2_bootstrap.py
--------------------
B2 (concat[L30, L25]) 5-seed ensemble + gene-level bootstrap CI (B=1000).
Question: Is L25 significantly better than v17f* (L15 delta)?
v17f* point = 0.7343, CI = [0.723, 0.747]
B2 point = 0.7449 (from capacity_baselines, 5-seed)
If B2 CI lower > 0.747: L25 is strictly better → change canonical layer
If B2 CI overlaps v17f* CI: not significant → retain L15 delta (interpretable)
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
OUT_DIR   = '../../reports/v17f_b2_bootstrap'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13, 21, 99]
BATCH_MLP  = 512
EPOCHS_MLP = 60
LAYER_B    = 30
LAYER_2    = 25       # B2: concat[L30, L25]
N_BOOT     = 1000
VSTAR_REF  = 0.7343
VSTAR_CI_U = 0.7470   # v17f* CI upper bound

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print(f"  B2 bootstrap CI: concat[L30, L{LAYER_2}]  B={N_BOOT}")
print(f"  v17f* ref: {VSTAR_REF:.4f}  CI upper: {VSTAR_CI_U:.4f}")
print("=" * 65)

# ── 1. Load embeddings ────────────────────────────────────────────
print("\n[1] Loading embeddings...")
X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l25_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_2:02d}_t30_150M.npy').astype(np.float32)
X_l30_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l25_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_2:02d}_t30_150M.npy').astype(np.float32)
print(f"  Train: {X_l30_tr.shape}  Test: {X_l30_te.shape}")

# ── 2. Scale L25 ──────────────────────────────────────────────────
sc = MaxAbsScaler()
X_l25_tr_s = sc.fit_transform(X_l25_tr).astype(np.float32)
X_l25_te_s = sc.transform(X_l25_te).astype(np.float32)

# ── 3. Gene IDs ───────────────────────────────────────────────────
print("\n[2] Loading gene IDs...")
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
tr_sym2idx   = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]
gene2idxs_te = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs_te[g].append(i)
te_genes_arr = np.array(list(gene2idxs_te.keys()))
print(f"  Test: {len(te_sym_list)} isoforms, {len(te_genes_arr)} genes")

# ── 4. GO labels ──────────────────────────────────────────────────
print("\n[3] Loading GO labels...")
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

tr_ids    = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_ids)
go_genes_tr  = defaultdict(set)
go_genes_all = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606' or p[7] != 'Function': continue
        go_genes_all[p[2]].add(p[1])
        if p[1] in tr_id_set: go_genes_tr[p[2]].add(p[1])

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])

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
print(f"  {len(mf_terms)} MF | valid: {valid_mask.sum()} | L2: {len(l2_valid)}")

# ── 5. Train ──────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')

def build_mlp(n_go=82):
    inp_a = layers.Input(shape=(640,))
    inp_b = layers.Input(shape=(640,))
    x     = layers.Concatenate()([inp_a, inp_b])
    x     = layers.Dense(256, activation='relu')(x)
    x     = layers.BatchNormalization()(x)
    x     = layers.Dropout(0.2)(x)
    x     = layers.Dense(128, activation='relu')(x)
    out   = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model([inp_a, inp_b], out)

focal_fn  = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
n_go      = len(mf_terms)
all_preds = []

print(f"\n[4] Training B2 (concat[L30, L{LAYER_2}]) — 5 seeds, {EPOCHS_MLP} epochs...")
t0 = time.time()
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm    = np.random.permutation(len(X_l30_tr))
    n_val   = int(len(X_l30_tr) * 0.1)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    mlp = build_mlp(n_go=n_go)
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
    mlp.fit(
        [X_l30_tr[tr_idx], X_l25_tr_s[tr_idx]], Y_tr[tr_idx],
        validation_data=([X_l30_tr[val_idx], X_l25_tr_s[val_idx]], Y_tr[val_idx]),
        epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    preds_i = mlp.predict([X_l30_te, X_l25_te_s], batch_size=1024, verbose=0)
    all_preds.append(preds_i)
    aps = [average_precision_score(Y_te[:, i], preds_i[:, i])
           for i in valid_idx if Y_te[:, i].sum() >= 2]
    print(f"  seed={seed}  AUPRC={np.mean(aps):.4f}")

preds = np.mean(all_preds, axis=0)
print(f"  Ensemble done: {time.time()-t0:.0f}s")
np.save(f'{OUT_DIR}/B2_preds.npy', preds)
np.save(f'{OUT_DIR}/Y_te.npy', Y_te)

# ── 6. Point estimate ─────────────────────────────────────────────
def macro_auprc(p, idxs):
    aps = [average_precision_score(Y_te[:, i], p[:, i])
           for i in idxs if Y_te[:, i].sum() >= 2]
    return float(np.mean(aps)) if aps else float('nan')

point_all = macro_auprc(preds, valid_idx)
point_l2  = macro_auprc(preds, l2_valid)
print(f"\n  B2 point: All MF={point_all:.4f}  L2={point_l2:.4f}")

# ── 7. Gene-level bootstrap CI ───────────────────────────────────
print(f"\n[5] Gene-level bootstrap CI (B={N_BOOT})...")
rng = np.random.default_rng(42)

boot_all, boot_l2 = [], []
for b in range(N_BOOT):
    sampled_genes = rng.choice(te_genes_arr, size=len(te_genes_arr), replace=True)
    boot_idxs = np.concatenate([gene2idxs_te[g] for g in sampled_genes])
    Y_b    = Y_te[boot_idxs]
    preds_b = preds[boot_idxs]
    valid_b = [i for i in valid_idx if Y_b[:, i].sum() >= 2]
    l2_b    = [i for i in l2_valid  if Y_b[:, i].sum() >= 2]
    if valid_b:
        aps = [average_precision_score(Y_b[:, i], preds_b[:, i]) for i in valid_b]
        boot_all.append(np.mean(aps))
    if l2_b:
        aps = [average_precision_score(Y_b[:, i], preds_b[:, i]) for i in l2_b]
        boot_l2.append(np.mean(aps))
    if (b+1) % 200 == 0:
        print(f"  {b+1}/{N_BOOT} done...")

ci_all = [float(np.percentile(boot_all, 2.5)), float(np.percentile(boot_all, 97.5))]
ci_l2  = [float(np.percentile(boot_l2,  2.5)), float(np.percentile(boot_l2,  97.5))]

# ── 8. Summary ────────────────────────────────────────────────────
overlap_all = ci_all[0] < VSTAR_CI_U and ci_all[1] > VSTAR_REF
verdict = "CI OVERLAP: B2(L25) NOT significantly better than v17f*(L15)" if overlap_all else \
          "B2(L25) CI lower > v17f* CI upper: L25 strictly better"

print(f"\n{'='*65}")
print(f"  B2 (concat[L30, L{LAYER_2}]) bootstrap CI results")
print(f"{'='*65}")
print(f"  B2  point:  All MF={point_all:.4f}  L2={point_l2:.4f}")
print(f"  B2  CI 95%: All MF=[{ci_all[0]:.4f}, {ci_all[1]:.4f}]")
print(f"              L2    =[{ci_l2[0]:.4f},  {ci_l2[1]:.4f}]")
print(f"")
print(f"  v17f* ref:  All MF={VSTAR_REF:.4f}  CI=[0.7227, {VSTAR_CI_U:.4f}]")
print(f"")
print(f"  Overlap (All MF CIs): {overlap_all}")
print(f"  Verdict: {verdict}")
print(f"{'='*65}")

results = {
    'B2_layer': LAYER_2,
    'description': f'concat[L30, L{LAYER_2}]',
    'point_all_mf': point_all,
    'point_l2': point_l2,
    'ci_all_mf': ci_all,
    'ci_l2': ci_l2,
    'B': N_BOOT,
    'vstar_ref': {'point': VSTAR_REF, 'ci': [0.7227, VSTAR_CI_U]},
    'ci_overlap_all_mf': overlap_all,
    'verdict': verdict,
}
with open(f'{OUT_DIR}/results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"\n[Saved] {OUT_DIR}/results.json")
