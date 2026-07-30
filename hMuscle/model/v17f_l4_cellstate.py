#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_l4_cellstate.py
---------------------
L4_CellState (112 terms, BP56+CC54) seq-only evaluation.
L3_CellType result: PRISM 0.222 → v17f* C0 0.622 (LOC/RNA did NOT help).
Question: Does L4 show similar seq-limit recovery, or does it remain truly hard?
If L4 >> 0.289 (PRISM): original H2 classification was entirely PRISM-artifact.
If L4 ≈ 0.289: L4 is the genuine sequence-only floor.
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
OUT_DIR   = '../../reports/v17f_l4_cellstate'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13, 21, 99]
BATCH_MLP  = 512
EPOCHS_MLP = 60
N_BOOT     = 500
L4_BASELINE_PRISM = 0.2888  # mean brain AUPRC from h2_layer_classification.tsv

print("=" * 65)
print("  L4_CellState (112 terms) seq-only evaluation")
print(f"  PRISM baseline: {L4_BASELINE_PRISM:.4f}")
print(f"  L3 reference:   C0=0.6218  (PRISM 0.2219)")
print("=" * 65)

# ── 1. Load embeddings ───────────────────────────────────────────
print("\n[1] Loading embeddings...")
X30_tr  = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X15_tr  = np.load(f'{DATA_DIR}/esm2_train_human_layer15_t30_150M.npy').astype(np.float32)
X30_te  = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)
X15_te  = np.load(f'{DATA_DIR}/esm2_layer_15_t30_150M.npy').astype(np.float32)
print(f"  Train: {X30_tr.shape}  Test: {X30_te.shape}")

sc = MaxAbsScaler()
delta_tr = sc.fit_transform(X30_tr - X15_tr).astype(np.float32)
delta_te = sc.transform(X30_te - X15_te).astype(np.float32)

# ── 2. Gene IDs ──────────────────────────────────────────────────
print("\n[2] Loading gene IDs...")
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

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

# ── 3. L4_CellState terms & labels ──────────────────────────────
print("\n[3] Loading L4_CellState terms...")
import csv
l4_terms, l4_cats = [], []
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        if row['layer'] == 'L4_CellState':
            l4_terms.append(row['go_id'])
            l4_cats.append(row['cat'])
print(f"  L4 terms: {len(l4_terms)} (BP={l4_cats.count('BP')}, CC={l4_cats.count('CC')})")

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
go_tr  = defaultdict(set)
go_all = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        go_all[p[2]].add(p[1])
        if p[1] in tr_id_set: go_tr[p[2]].add(p[1])

def build_labels(terms):
    Y_tr_list, Y_te_list = [], []
    for go_id in terms:
        pos_ids  = go_tr[go_id]
        pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in pos_ids}
        y_tr = np.zeros(len(tr_genes), dtype=np.float32)
        for sym in pos_syms:
            for idx in tr_sym2idx.get(sym, []): y_tr[idx] = 1.0
        Y_tr_list.append(y_tr)
        pos_all = go_all[go_id]
        y_te = np.array([1.0 if sym2id.get(s, '__') in pos_all else 0.0
                         for s in te_sym_list], dtype=np.float32)
        Y_te_list.append(y_te)
    return np.stack(Y_tr_list, axis=1), np.stack(Y_te_list, axis=1)

print("  Building label matrices (112 terms)...")
Y_tr, Y_te = build_labels(l4_terms)
valid_idx  = [i for i in range(len(l4_terms)) if Y_te[:, i].sum() >= 2]
l4_bp_idx  = [i for i in valid_idx if l4_cats[i] == 'BP']
l4_cc_idx  = [i for i in valid_idx if l4_cats[i] == 'CC']
print(f"  Valid: {len(valid_idx)} (BP={len(l4_bp_idx)}, CC={len(l4_cc_idx)})")

# ── 4. Model ─────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')

focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
n_out    = len(l4_terms)

def build_mlp():
    d   = layers.Input(shape=(640,))
    l30 = layers.Input(shape=(640,))
    x   = layers.Concatenate()([d, l30])
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(128, activation='relu')(x)
    return models.Model([d, l30], layers.Dense(n_out, activation='sigmoid')(x))

# ── 5. Train ─────────────────────────────────────────────────────
print(f"\n[4] Training L4 seq-only (C0) — 5 seeds, {EPOCHS_MLP} epochs...")
all_preds = []
t0 = time.time()
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm  = np.random.permutation(len(X30_tr))
    n_val = int(len(X30_tr) * 0.1)
    vi, ti = perm[:n_val], perm[n_val:]
    mlp = build_mlp()
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
    mlp.fit(
        [delta_tr[ti], X30_tr[ti]], Y_tr[ti],
        validation_data=([delta_tr[vi], X30_tr[vi]], Y_tr[vi]),
        epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    p = mlp.predict([delta_te, X30_te], batch_size=1024, verbose=0)
    all_preds.append(p)

    def mean_ap(idxs): return np.mean([average_precision_score(Y_te[:,i],p[:,i]) for i in idxs if Y_te[:,i].sum()>=2])
    print(f"  seed={seed}  all={mean_ap(valid_idx):.4f}  BP={mean_ap(l4_bp_idx):.4f}  CC={mean_ap(l4_cc_idx):.4f}")

preds = np.mean(all_preds, axis=0)
print(f"  Ensemble done: {time.time()-t0:.0f}s")
np.save(f'{OUT_DIR}/L4_preds.npy', preds)
np.save(f'{OUT_DIR}/Y_te.npy', Y_te)

# ── 6. Point + bootstrap CI ──────────────────────────────────────
def macro_ap(idxs):
    aps = [average_precision_score(Y_te[:,i], preds[:,i]) for i in idxs if Y_te[:,i].sum()>=2]
    return float(np.mean(aps)) if aps else float('nan')

point_all = macro_ap(valid_idx)
point_bp  = macro_ap(l4_bp_idx)
point_cc  = macro_ap(l4_cc_idx)

print(f"\n[5] Gene-level bootstrap CI (B={N_BOOT})...")
rng = np.random.default_rng(42)
boot_all, boot_bp, boot_cc = [], [], []
for b in range(N_BOOT):
    sg    = rng.choice(te_genes_arr, size=len(te_genes_arr), replace=True)
    bidxs = np.concatenate([gene2idxs_te[g] for g in sg])
    Yb = Y_te[bidxs]; pb = preds[bidxs]
    vb   = [i for i in valid_idx  if Yb[:,i].sum()>=2]
    bpb  = [i for i in l4_bp_idx  if Yb[:,i].sum()>=2]
    ccb  = [i for i in l4_cc_idx  if Yb[:,i].sum()>=2]
    if vb:  boot_all.append(np.mean([average_precision_score(Yb[:,i],pb[:,i]) for i in vb]))
    if bpb: boot_bp.append(np.mean([average_precision_score(Yb[:,i],pb[:,i]) for i in bpb]))
    if ccb: boot_cc.append(np.mean([average_precision_score(Yb[:,i],pb[:,i]) for i in ccb]))
    if (b+1) % 100 == 0: print(f"  {b+1}/{N_BOOT}")

ci_all = [float(np.percentile(boot_all, 2.5)), float(np.percentile(boot_all, 97.5))]
ci_bp  = [float(np.percentile(boot_bp,  2.5)), float(np.percentile(boot_bp,  97.5))]
ci_cc  = [float(np.percentile(boot_cc,  2.5)), float(np.percentile(boot_cc,  97.5))]

# ── 7. Summary ───────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  L2/L3/L4 COMPARISON — v17f* vs PRISM baseline")
print(f"{'='*65}")
print(f"  {'Layer':12s}  {'PRISM':>8s}  {'v17f*':>8s}  {'95%CI':>22s}  {'Δ':>8s}")
print(f"  {'L2_Struct':12s}  {'0.313':>8s}  {'0.633':>8s}  {'[0.617, 0.654]':>22s}  {'+0.320':>8s}")
print(f"  {'L3_CellType':12s}  {'0.222':>8s}  {'0.622':>8s}  {'[see L3 report]':>22s}  {'+0.400':>8s}")
print(f"  {'L4_CellState':12s}  {L4_BASELINE_PRISM:>8.4f}  {point_all:>8.4f}  [{ci_all[0]:.4f}, {ci_all[1]:.4f}]  {point_all-L4_BASELINE_PRISM:>+8.4f}")
print(f"  {'  BP':12s}  {'0.3057':>8s}  {point_bp:>8.4f}  [{ci_bp[0]:.4f}, {ci_bp[1]:.4f}]")
print(f"  {'  CC':12s}  {'0.2768':>8s}  {point_cc:>8.4f}  [{ci_cc[0]:.4f}, {ci_cc[1]:.4f}]")
print(f"{'='*65}")

verdict = (
    "L4 remains hard (seq-limit confirmed)" if point_all < 0.45 else
    "L4 partially recovered (original classification was artifact)" if point_all < 0.55 else
    "L4 largely recovered (PRISM-artifact classification)"
)
print(f"  Verdict: {verdict}")

results = {
    'layer': 'L4_CellState',
    'n_terms': len(l4_terms), 'n_bp': l4_cats.count('BP'), 'n_cc': l4_cats.count('CC'),
    'prism_baseline': {'all': L4_BASELINE_PRISM, 'bp': 0.3057, 'cc': 0.2768},
    'v17f_seq': {
        'point_all': point_all, 'ci_all': ci_all,
        'point_bp':  point_bp,  'ci_bp':  ci_bp,
        'point_cc':  point_cc,  'ci_cc':  ci_cc,
    },
    'l3_ref': {'prism': 0.2219, 'v17f_seq': 0.6218},
    'l2_ref': {'prism': 0.3127, 'v17f_seq': 0.633},
    'verdict': verdict,
    'B': N_BOOT,
}
with open(f'{OUT_DIR}/results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"[Saved] {OUT_DIR}/results.json")
