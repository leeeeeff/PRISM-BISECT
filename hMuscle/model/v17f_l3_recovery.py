#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_l3_recovery.py
--------------------
L3_CellType (23 terms, BP9+CC14) recovery experiment.
Baseline: PRISM mean brain AUPRC = 0.222
Three conditions:
  C0: seq-only (delta[L30-L15] + L30)  — same as v17f* but for L3 terms
  C1: seq + LOC(8)                     — add subcellular localization signals
  C2: seq + LOC(8) + RNA(9)            — add mRNA structure features
Question: How much of L3's poor performance is recoverable from sequence-derivable features?
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
FEAT_DIR  = '../results_isoform/features'
OUT_DIR   = '../../reports/v17f_l3_recovery'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13, 21, 99]
BATCH_MLP  = 512
EPOCHS_MLP = 60
N_BOOT     = 500
L3_BASELINE_PRISM = 0.2219

print("=" * 65)
print("  L3_CellType (23 terms) recovery experiment")
print(f"  PRISM baseline: {L3_BASELINE_PRISM:.4f}")
print("  C0: seq-only  C1: +LOC  C2: +LOC+RNA")
print("=" * 65)

# ── 1. Load embeddings ───────────────────────────────────────────
print("\n[1] Loading embeddings & aux features...")
X30_tr  = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X15_tr  = np.load(f'{DATA_DIR}/esm2_train_human_layer15_t30_150M.npy').astype(np.float32)
X30_te  = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)
X15_te  = np.load(f'{DATA_DIR}/esm2_layer_15_t30_150M.npy').astype(np.float32)

sc_delta = MaxAbsScaler()
delta_tr = sc_delta.fit_transform(X30_tr - X15_tr).astype(np.float32)
delta_te = sc_delta.transform(X30_te - X15_te).astype(np.float32)

# LOC features (8-dim)
loc_tr_raw = np.load(f'{FEAT_DIR}/loc/loc_features_train.npy').astype(np.float32)
loc_te_raw = np.load(f'{FEAT_DIR}/loc/loc_features_test.npy').astype(np.float32)
sc_loc = MaxAbsScaler()
loc_tr = sc_loc.fit_transform(loc_tr_raw)
loc_te = sc_loc.transform(loc_te_raw)

# RNA features (9-dim)
rna_tr_raw = np.load(f'{FEAT_DIR}/rna/rna_features_train.npy').astype(np.float32)
rna_te_raw = np.load(f'{FEAT_DIR}/rna/rna_features_test.npy').astype(np.float32)
sc_rna = MaxAbsScaler()
rna_tr = sc_rna.fit_transform(rna_tr_raw)
rna_te = sc_rna.transform(rna_te_raw)

print(f"  Train: {X30_tr.shape}, LOC: {loc_tr.shape}, RNA: {rna_tr.shape}")
print(f"  Test:  {X30_te.shape}, LOC: {loc_te.shape}, RNA: {rna_te.shape}")

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

# ── 3. L3_CellType terms & labels ───────────────────────────────
print("\n[3] Loading L3_CellType terms...")
import csv
l3_terms, l3_cats = [], []
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        if row['layer'] == 'L3_CellType':
            l3_terms.append(row['go_id'])
            l3_cats.append(row['cat'])
print(f"  L3 terms: {len(l3_terms)} (BP={l3_cats.count('BP')}, CC={l3_cats.count('CC')})")

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

Y_tr, Y_te = build_labels(l3_terms)
valid_idx = [i for i in range(len(l3_terms)) if Y_te[:, i].sum() >= 2]
l3_bp_idx = [i for i in valid_idx if l3_cats[i] == 'BP']
l3_cc_idx = [i for i in valid_idx if l3_cats[i] == 'CC']
print(f"  Valid: {len(valid_idx)} (BP={len(l3_bp_idx)}, CC={len(l3_cc_idx)})")

# ── 4. Model builders ────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')

focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
n_out    = len(l3_terms)

def build_mlp_c0():  # seq-only: delta(640) + L30(640)
    d = layers.Input(shape=(640,))
    l = layers.Input(shape=(640,))
    x = layers.Concatenate()([d, l])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    return models.Model([d, l], layers.Dense(n_out, activation='sigmoid')(x))

def build_mlp_c1():  # seq + LOC(8)
    d = layers.Input(shape=(640,))
    l = layers.Input(shape=(640,))
    a = layers.Input(shape=(8,))
    x = layers.Concatenate()([d, l, a])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    return models.Model([d, l, a], layers.Dense(n_out, activation='sigmoid')(x))

def build_mlp_c2():  # seq + LOC(8) + RNA(9)
    d  = layers.Input(shape=(640,))
    l  = layers.Input(shape=(640,))
    a  = layers.Input(shape=(8,))
    r  = layers.Input(shape=(9,))
    x  = layers.Concatenate()([d, l, a, r])
    x  = layers.Dense(256, activation='relu')(x)
    x  = layers.BatchNormalization()(x)
    x  = layers.Dropout(0.2)(x)
    x  = layers.Dense(128, activation='relu')(x)
    return models.Model([d, l, a, r], layers.Dense(n_out, activation='sigmoid')(x))

# ── 5. Training function ─────────────────────────────────────────
def train_condition(cond_name, build_fn, get_tr_inputs, get_te_inputs):
    print(f"\n[Train] {cond_name} — 5 seeds")
    all_preds = []
    t0 = time.time()
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm  = np.random.permutation(len(X30_tr))
        n_val = int(len(X30_tr) * 0.1)
        vi, ti = perm[:n_val], perm[n_val:]

        tr_inp = [x[ti] for x in get_tr_inputs()]
        va_inp = [x[vi] for x in get_tr_inputs()]

        mlp = build_fn()
        mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
        mlp.fit(
            tr_inp, Y_tr[ti],
            validation_data=(va_inp, Y_tr[vi]),
            epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0
        )
        p = mlp.predict(get_te_inputs(), batch_size=1024, verbose=0)
        all_preds.append(p)
        aps = [average_precision_score(Y_te[:, i], p[:, i])
               for i in valid_idx if Y_te[:, i].sum() >= 2]
        print(f"  seed={seed}  all={np.mean(aps):.4f}  "
              f"BP={np.mean([average_precision_score(Y_te[:,i],p[:,i]) for i in l3_bp_idx]):.4f}  "
              f"CC={np.mean([average_precision_score(Y_te[:,i],p[:,i]) for i in l3_cc_idx]):.4f}")
    preds = np.mean(all_preds, axis=0)
    print(f"  Done: {time.time()-t0:.0f}s")

    def macro_auprc(idxs):
        aps = [average_precision_score(Y_te[:,i], preds[:,i]) for i in idxs if Y_te[:,i].sum()>=2]
        return float(np.mean(aps)) if aps else float('nan')

    point_all = macro_auprc(valid_idx)
    point_bp  = macro_auprc(l3_bp_idx)
    point_cc  = macro_auprc(l3_cc_idx)

    # Gene-level bootstrap
    rng = np.random.default_rng(42)
    boot_all, boot_bp, boot_cc = [], [], []
    for b in range(N_BOOT):
        sg = rng.choice(te_genes_arr, size=len(te_genes_arr), replace=True)
        bidxs = np.concatenate([gene2idxs_te[g] for g in sg])
        Yb = Y_te[bidxs]; pb = preds[bidxs]
        vb  = [i for i in valid_idx  if Yb[:,i].sum()>=2]
        bpb = [i for i in l3_bp_idx  if Yb[:,i].sum()>=2]
        ccb = [i for i in l3_cc_idx  if Yb[:,i].sum()>=2]
        if vb:  boot_all.append(np.mean([average_precision_score(Yb[:,i], pb[:,i]) for i in vb]))
        if bpb: boot_bp.append(np.mean([average_precision_score(Yb[:,i], pb[:,i]) for i in bpb]))
        if ccb: boot_cc.append(np.mean([average_precision_score(Yb[:,i], pb[:,i]) for i in ccb]))
        if (b+1) % 100 == 0: print(f"  boot {b+1}/{N_BOOT}")

    ci_all = [float(np.percentile(boot_all, 2.5)), float(np.percentile(boot_all, 97.5))]
    ci_bp  = [float(np.percentile(boot_bp,  2.5)), float(np.percentile(boot_bp,  97.5))]
    ci_cc  = [float(np.percentile(boot_cc,  2.5)), float(np.percentile(boot_cc,  97.5))]

    print(f"  {cond_name}  all={point_all:.4f}[{ci_all[0]:.4f},{ci_all[1]:.4f}]  "
          f"BP={point_bp:.4f}[{ci_bp[0]:.4f},{ci_bp[1]:.4f}]  "
          f"CC={point_cc:.4f}[{ci_cc[0]:.4f},{ci_cc[1]:.4f}]")
    np.save(f'{OUT_DIR}/{cond_name}_preds.npy', preds)
    return {'point_all': point_all, 'ci_all': ci_all,
            'point_bp': point_bp, 'ci_bp': ci_bp,
            'point_cc': point_cc, 'ci_cc': ci_cc}

# ── 6. Run three conditions ──────────────────────────────────────
res_c0 = train_condition(
    'C0_seq',
    build_mlp_c0,
    lambda: [delta_tr, X30_tr],
    lambda: [delta_te, X30_te]
)
res_c1 = train_condition(
    'C1_seq_loc',
    build_mlp_c1,
    lambda: [delta_tr, X30_tr, loc_tr],
    lambda: [delta_te, X30_te, loc_te]
)
res_c2 = train_condition(
    'C2_seq_loc_rna',
    build_mlp_c2,
    lambda: [delta_tr, X30_tr, loc_tr, rna_tr],
    lambda: [delta_te, X30_te, loc_te, rna_te]
)

# ── 7. Summary ───────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  L3_CellType RECOVERY RESULTS")
print(f"{'='*65}")
print(f"  {'Condition':20s}  {'All':>8s}  {'BP':>8s}  {'CC':>8s}")
print(f"  {'PRISM baseline':20s}  {L3_BASELINE_PRISM:>8.4f}  {'0.1716':>8s}  {'0.2542':>8s}")
for name, res in [('C0 seq-only', res_c0), ('C1 +LOC', res_c1), ('C2 +LOC+RNA', res_c2)]:
    print(f"  {name:20s}  {res['point_all']:>8.4f}  {res['point_bp']:>8.4f}  {res['point_cc']:>8.4f}")
print(f"{'='*65}")

results = {
    'prism_baseline': {'point_all': L3_BASELINE_PRISM, 'point_bp': 0.1716, 'point_cc': 0.2542},
    'C0_seq':         res_c0,
    'C1_seq_loc':     res_c1,
    'C2_seq_loc_rna': res_c2,
    'n_l3_terms': len(l3_terms),
    'n_bp': l3_cats.count('BP'),
    'n_cc': l3_cats.count('CC'),
    'B': N_BOOT,
    'note': 'LOC: 8-dim subcellular signals; RNA: 9-dim mRNA structure features',
}
with open(f'{OUT_DIR}/results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"[Saved] {OUT_DIR}/results.json")
