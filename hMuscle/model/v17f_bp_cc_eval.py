#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_bp_cc_eval.py
------------------
v17f* architecture (concat[delta_L30-L15, L30]) trained separately on
BP (103 terms) and CC (93 terms) GO domains.
Question: Does delta_layer benefit generalise beyond MF to BP/CC?
MF baseline: v17f* = 0.734 (vs PRISM 0.531, +0.203)
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
OUT_DIR   = '../../reports/v17f_bp_cc_eval'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13, 21, 99]
BATCH_MLP  = 512
EPOCHS_MLP = 60
N_BOOT     = 500

print("=" * 65)
print("  v17f* BP/CC evaluation  (delta[L30-L15] + L30)")
print(f"  MF baseline: 0.734  PRISM BP: 0.372  PRISM CC: 0.367")
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

# ── 3. GO labels ─────────────────────────────────────────────────
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

tr_ids   = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_ids)
go_genes_tr  = defaultdict(set)
go_genes_all = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        go_genes_all[p[2]].add(p[1])
        if p[1] in tr_id_set: go_genes_tr[p[2]].add(p[1])

# ── 4. Load term lists ───────────────────────────────────────────
import csv
all_terms_info = {}
with open('../../reports/v_expanded_gomf/expanded_go_per_term.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        all_terms_info[row['go_id']] = row

bp_terms = [r['go_id'] for r in all_terms_info.values()
            if r['cat']=='BP' and int(r['n_pos_te'])>=2 and int(r['n_pos_tr'])>=2]
cc_terms = [r['go_id'] for r in all_terms_info.values()
            if r['cat']=='CC' and int(r['n_pos_te'])>=2 and int(r['n_pos_tr'])>=2]
print(f"  BP: {len(bp_terms)} terms, CC: {len(cc_terms)} terms")

def build_Y(terms, namespace_filter):
    Y_tr_list, Y_te_list = [], []
    for go_id in terms:
        pos_ids = go_genes_tr[go_id]
        pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in pos_ids}
        y_tr = np.zeros(len(tr_genes), dtype=np.float32)
        for sym in pos_syms:
            for idx in tr_sym2idx.get(sym, []): y_tr[idx] = 1.0
        Y_tr_list.append(y_tr)

        pos_all = go_genes_all[go_id]
        y_te = np.array([1.0 if sym2id.get(s, '__') in pos_all else 0.0
                         for s in te_sym_list], dtype=np.float32)
        Y_te_list.append(y_te)
    return np.stack(Y_tr_list, axis=1), np.stack(Y_te_list, axis=1)

print("  Building BP label matrices...")
Y_bp_tr, Y_bp_te = build_Y(bp_terms, 'BP')
print("  Building CC label matrices...")
Y_cc_tr, Y_cc_te = build_Y(cc_terms, 'CC')

# ── 5. Model ─────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')

def build_mlp(n_out):
    inp_delta = layers.Input(shape=(640,))
    inp_l30   = layers.Input(shape=(640,))
    x = layers.Concatenate()([inp_delta, inp_l30])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_out, activation='sigmoid')(x)
    return models.Model([inp_delta, inp_l30], out)

focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

def train_eval(Y_tr, Y_te, terms, label):
    n_out = len(terms)
    valid_idx = [i for i in range(n_out) if Y_te[:, i].sum() >= 2]
    print(f"\n[Train] {label}: {n_out} terms, valid_test={len(valid_idx)}, 5 seeds")
    all_preds = []
    t0 = time.time()
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm  = np.random.permutation(len(X30_tr))
        n_val = int(len(X30_tr) * 0.1)
        vi, ti = perm[:n_val], perm[n_val:]
        mlp = build_mlp(n_out)
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
        aps = [average_precision_score(Y_te[:, i], p[:, i])
               for i in valid_idx if Y_te[:, i].sum() >= 2]
        print(f"  seed={seed}  AUPRC={np.mean(aps):.4f}")
    preds = np.mean(all_preds, axis=0)
    print(f"  Ensemble done: {time.time()-t0:.0f}s")

    # Point estimate
    aps_all = [average_precision_score(Y_te[:, i], preds[:, i])
               for i in valid_idx if Y_te[:, i].sum() >= 2]
    point = float(np.mean(aps_all))

    # Gene-level bootstrap CI
    rng = np.random.default_rng(42)
    boot = []
    for b in range(N_BOOT):
        sg = rng.choice(te_genes_arr, size=len(te_genes_arr), replace=True)
        bidxs = np.concatenate([gene2idxs_te[g] for g in sg])
        Yb = Y_te[bidxs]; pb = preds[bidxs]
        vb = [i for i in valid_idx if Yb[:, i].sum() >= 2]
        if vb:
            boot.append(np.mean([average_precision_score(Yb[:, i], pb[:, i]) for i in vb]))
        if (b+1) % 100 == 0: print(f"  boot {b+1}/{N_BOOT}")
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    print(f"  {label}: point={point:.4f}  95%CI=[{ci[0]:.4f}, {ci[1]:.4f}]")
    np.save(f'{OUT_DIR}/{label}_preds.npy', preds)
    return point, ci

Y_te_bp, Y_te_cc = Y_bp_te, Y_cc_te

point_bp, ci_bp = train_eval(Y_bp_tr, Y_bp_te, bp_terms, 'BP')
point_cc, ci_cc = train_eval(Y_cc_tr, Y_cc_te, cc_terms, 'CC')

# ── 6. Summary ───────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  v17f* DOMAIN COMPARISON")
print(f"{'='*65}")
print(f"  {'Domain':6s}  {'v17f*':>8s}  {'95%CI':>20s}  {'PRISM':>8s}  {'Δ':>8s}")
print(f"  {'MF':6s}  {'0.734':>8s}  {'[0.723, 0.747]':>20s}  {'0.531':>8s}  {'+0.203':>8s}")
print(f"  {'BP':6s}  {point_bp:8.4f}  [{ci_bp[0]:.4f}, {ci_bp[1]:.4f}]  {'0.372':>8s}  {point_bp-0.372:+8.4f}")
print(f"  {'CC':6s}  {point_cc:8.4f}  [{ci_cc[0]:.4f}, {ci_cc[1]:.4f}]  {'0.367':>8s}  {point_cc-0.367:+8.4f}")
print(f"{'='*65}")

results = {
    'MF': {'point': 0.734, 'ci': [0.723, 0.747], 'prism_baseline': 0.531, 'n_terms': 82},
    'BP': {'point': point_bp, 'ci': ci_bp, 'prism_baseline': 0.372, 'n_terms': len(bp_terms)},
    'CC': {'point': point_cc, 'ci': ci_cc, 'prism_baseline': 0.367, 'n_terms': len(cc_terms)},
    'B': N_BOOT,
    'architecture': 'concat[delta(L30-L15, MaxAbsScaled), L30] -> Dense(256,relu)->BN->Drop(0.2)->Dense(128)->Dense(n,sig)',
}
with open(f'{OUT_DIR}/results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"[Saved] {OUT_DIR}/results.json")
