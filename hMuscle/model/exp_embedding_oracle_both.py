#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_embedding_oracle_both.py  (Option B: apples-to-apples gene-mean-EMBEDDING oracle)
=====================================================================================
The manuscript's muscle gene-mean oracle 0.803 is the "gene-mean EMBEDDING" oracle
(§203: "replacing each isoform's embedding with its gene-mean embedding"). The brain
number reported so far (0.761) is the gene-mean-of-PREDICTIONS oracle — a methodological
mismatch flagged by devils-advocate. This script computes the SAME embedding-oracle
recipe on BOTH tissues with ONE shared v17f* ensemble, so muscle should reproduce ~0.803
(recipe validation) and brain yields the apples-to-apples value.

Recipe (MaxAbsScaler and gene-mean are both linear, so they commute): for each test set,
replace each isoform's model-input features (scaled delta_layer, raw L30) with its parent
gene's mean, then predict with the trained ensemble = "gene-mean embedding" oracle.

Sanity: normal (un-collapsed) AUPRC should reproduce muscle 0.734 / brain 0.647.
"""
import os, gzip, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'   # RTX 4090 ptxas workaround (rerun bug #1)
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = '../data'
BRAIN_DIR = '../data/brain_isoquant_esm2/full'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/truebrain_rerun_20260714/exp_embedding_oracle'
os.makedirs(OUT_DIR, exist_ok=True)
SEEDS = [42, 7, 13, 21, 99]; EPOCHS = 60; BATCH = 512; LA, LB = 15, 30


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


# ── train ──────────────────────────────────────────────────────────
X_tr_l30 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LB:02d}_t30_150M.npy').astype(np.float32)
X_tr_l15 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LA:02d}_t30_150M.npy').astype(np.float32)
delta_tr = (X_tr_l30 - X_tr_l15).astype(np.float32)
scaler = MaxAbsScaler(); delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
tr_genes = [clean(g) for g in np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)]

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
tr_ids = [sym2id.get(g, g) for g in tr_genes]; tr_id_set = set(tr_ids)
go_genes_tr, go_genes_all = defaultdict(set), defaultdict(set)
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
L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural': L2_TERMS.add(p[0])
tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)


def build_Y_tr(go_id):
    pos_ids = go_genes_tr[go_id]
    pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in pos_ids}
    y = np.zeros(len(tr_genes), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
    return y


Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)


def build_Y_te(te_syms):
    return np.stack([np.array([1.0 if sym2id.get(s, '__') in go_genes_all[go] else 0.0
                               for s in te_syms], dtype=np.float32) for go in mf_terms], axis=1)


def load_testset(kind):
    if kind == 'muscle':
        l30 = np.load(f'{DATA_DIR}/esm2_layer_{LB:02d}_t30_150M.npy').astype(np.float32)
        l15 = np.load(f'{DATA_DIR}/esm2_layer_{LA:02d}_t30_150M.npy').astype(np.float32)
        graw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
        syms = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in graw]
    else:
        l30 = np.load(f'{BRAIN_DIR}/brain_full_esm2_layer{LB:02d}_t30_150M.npy').astype(np.float32)
        l15 = np.load(f'{BRAIN_DIR}/brain_full_esm2_layer{LA:02d}_t30_150M.npy').astype(np.float32)
        syms = [clean(g) for g in np.load(f'{BRAIN_DIR}/brain_full_gene_names.npy', allow_pickle=True)]
    ds = scaler.transform((l30 - l15).astype(np.float32)).astype(np.float32)
    Y = build_Y_te(syms)
    gi = np.unique(np.array(syms), return_inverse=True)[1]
    return {'ds': ds, 'l30': l30, 'Y': Y, 'gi': gi, 'syms': syms}


def gene_mean_collapse(feat, gi):
    """replace each row with its gene-mean (parent-gene centroid)."""
    K = gi.max() + 1
    sums = np.zeros((K, feat.shape[1]), np.float64); np.add.at(sums, gi, feat)
    cnt = np.bincount(gi, minlength=K).astype(np.float64)
    return (sums / cnt[:, None])[gi].astype(np.float32)


print("[load] muscle + brain test sets...", flush=True)
TS = {'muscle': load_testset('muscle'), 'brain': load_testset('brain')}
for k, d in TS.items():
    print(f"  {k}: {d['ds'].shape[0]} isoforms, {len(set(d['syms']))} genes", flush=True)

# ── model ──────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)


def build_mlp(n_go):
    inp_d = layers.Input(shape=(640,)); inp_e = layers.Input(shape=(640,))
    x = layers.Concatenate()([inp_d, inp_e])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model([inp_d, inp_e], out)


n_go = len(mf_terms)
focal = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
print(f"[train] 5-seed ensemble ({n_go} MF)...", flush=True)
t0 = time.time()
pred_norm = {k: [] for k in TS}; pred_oracle = {k: [] for k in TS}
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm = np.random.permutation(len(delta_tr_s)); nval = int(len(delta_tr_s) * 0.1)
    vi, ti = perm[:nval], perm[nval:]
    mlp = build_mlp(n_go); mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal)
    mlp.fit([delta_tr_s[ti], X_tr_l30[ti]], Y_tr[ti],
            validation_data=([delta_tr_s[vi], X_tr_l30[vi]], Y_tr[vi]),
            epochs=EPOCHS, batch_size=BATCH,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                        restore_best_weights=True)], verbose=0)
    for k, d in TS.items():
        pred_norm[k].append(mlp.predict([d['ds'], d['l30']], batch_size=1024, verbose=0))
        gm_ds = gene_mean_collapse(d['ds'], d['gi']); gm_l30 = gene_mean_collapse(d['l30'], d['gi'])
        pred_oracle[k].append(mlp.predict([gm_ds, gm_l30], batch_size=1024, verbose=0))
    print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

res = {}
for k, d in TS.items():
    Y = d['Y']; vmask = Y.sum(0) >= 2
    vidx = [i for i in range(n_go) if vmask[i]]
    l2idx = [i for i in range(n_go) if mf_terms[i] in L2_TERMS and vmask[i]]
    pn = np.mean(pred_norm[k], 0); po = np.mean(pred_oracle[k], 0)

    def macro(P, idxs):
        return float(np.mean([average_precision_score(Y[:, i], P[:, i]) for i in idxs]))
    res[k] = {
        'n_iso': int(Y.shape[0]), 'n_valid_mf': len(vidx), 'n_l2': len(l2idx),
        'normal_all_mf': macro(pn, vidx), 'normal_l2': macro(pn, l2idx),
        'embedding_oracle_all_mf': macro(po, vidx), 'embedding_oracle_l2': macro(po, l2idx),
    }
    res[k]['oracle_gap_all_mf'] = res[k]['embedding_oracle_all_mf'] - res[k]['normal_all_mf']

json.dump(res, open(f'{OUT_DIR}/results.json', 'w'), indent=2)
print("\n" + "=" * 68)
print(" GENE-MEAN EMBEDDING ORACLE (apples-to-apples, same ensemble)")
print("=" * 68)
for k in ['muscle', 'brain']:
    r = res[k]
    print(f" {k:6s}  normal AllMF={r['normal_all_mf']:.4f} L2={r['normal_l2']:.4f}"
          f"  | EMB-ORACLE AllMF={r['embedding_oracle_all_mf']:.4f} L2={r['embedding_oracle_l2']:.4f}"
          f"  (gap {r['oracle_gap_all_mf']:+.4f})")
print(" sanity: muscle normal≈0.734, muscle oracle≈0.803 (recipe validation);")
print("         brain normal≈0.647, brain oracle = apples-to-apples answer.")
print(f"[saved] {OUT_DIR}/results.json")
