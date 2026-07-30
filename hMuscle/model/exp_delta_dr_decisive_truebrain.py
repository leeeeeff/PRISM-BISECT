#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_delta_dr_decisive_truebrain.py
----------------------------------
DECISIVE test for the delta_layer (L30-L15) contribution on TRUE BRAIN.

Context: the capacity baselines (v17f_capacity_baselines_v2_truebrain.py) showed that
on true brain, MACRO GO-AUPRC is near-identical whether the second feature is delta,
raw-L15, or L25 (B0 0.659 / B1 0.655 / B2 0.659). But macro AUPRC is a gene-level
quantity (gene-mean oracle already ~0.81), so it is the WRONG place to look for the
delta's claimed contribution, which is WITHIN-GENE isoform discrimination.

This script computes, for identical MLPs / seeds / labels, BOTH macro AUPRC and
within-gene Domain-Ranking AUC for:
  L30_only     : single layer L30 (640-dim)           <- "one good layer" competitor
  L15_only     : single layer L15 (640-dim)           <- "one good layer" competitor
  delta_only   : L30-L15 alone (640-dim)              <- the contrast, by itself
  cat[L30,delta]: concat[L30, L30-L15] (1280)         <- v17f* reference
  cat[L30,L15] : concat[L30, L15_raw] (1280)          <- B1 (no subtraction)

Decisive question: does the CONTRAST (delta / cat-delta) beat a single layer on
within-gene DR-AUC? If yes -> delta earns its keep on brain (just not on macro).
If DR-AUC is also tied -> honest negative: delta adds nothing on brain.

DR-AUC computation replicated verbatim from domain_ranking_validation_v2_truebrain.py.
"""

import os, json, gzip, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
BRAIN_DIR = '../data/brain_isoquant_esm2/full'
DOMAIN_MAT= '../results_isoform/features/domain_matrix_brain_full.npy'
OUT_DIR   = '../../reports/truebrain_rerun_20260714/exp_delta_dr_decisive'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 7, 13, 21, 99]
EPOCHS_MLP = 60
BATCH_MLP  = 512

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 72)
print("  DECISIVE: delta_layer within-gene DR-AUC vs single-layer (TRUE BRAIN)")
print("=" * 72)

# ── 1. Embeddings ─────────────────────────────────────────────────────
print("\n[1] Loading embeddings L15, L30 (train muscle, test brain)...")
X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X_l15_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer15_t30_150M.npy').astype(np.float32)
X_l30_te = np.load(f'{BRAIN_DIR}/brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
X_l15_te = np.load(f'{BRAIN_DIR}/brain_full_esm2_layer15_t30_150M.npy').astype(np.float32)
print(f"  train {X_l30_tr.shape}  test {X_l30_te.shape}")

def maxabs(tr, te):
    sc = MaxAbsScaler(); return sc.fit_transform(tr).astype(np.float32), sc.transform(te).astype(np.float32)

delta_tr, delta_te = (X_l30_tr - X_l15_tr).astype(np.float32), (X_l30_te - X_l15_te).astype(np.float32)
delta_tr_s, delta_te_s = maxabs(delta_tr, delta_te)
l15_tr_s,  l15_te_s    = maxabs(X_l15_tr, X_l15_te)

# ── 2. IDs + GO labels (verbatim from capacity/domain_ranking scripts) ─
print("\n[2] IDs + GO labels...")
tr_genes = [clean(g) for g in np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)]
te_sym_list = [clean(g) for g in np.load(f'{BRAIN_DIR}/brain_full_gene_names.npy', allow_pickle=True)]
te_iso_list = [clean(x) for x in np.load(f'{BRAIN_DIR}/brain_full_ids.npy', allow_pickle=True)]
n_iso = len(te_iso_list)

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
    return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0 for s in te_sym_list], dtype=np.float32)

Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)
Y_te = np.stack([build_Y_te(go) for go in mf_terms], axis=1)
valid_mask = Y_te.sum(0) >= 2
valid_idx  = [i for i in range(len(mf_terms)) if valid_mask[i]]

L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural': L2_TERMS.add(p[0])
l2_valid = [i for i in range(len(mf_terms)) if mf_terms[i] in L2_TERMS and valid_mask[i]]
print(f"  {len(mf_terms)} MF | valid {int(valid_mask.sum())} | L2 {len(l2_valid)}")

# ── 3. Domain-ranking prerequisites ───────────────────────────────────
gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs[g].append(i)
iso_n_domains = np.load(DOMAIN_MAT).sum(axis=1).astype(np.int32)
Y_te_v = Y_te[:, valid_mask]  # (n_iso, n_valid), column order == valid_idx order

def compute_domain_ranking_auc(preds_v, gene2idxs, iso_n_domains, Y_te_v):
    """preds_v: (n_iso, n_valid) aligned to Y_te_v columns. Verbatim logic."""
    aucs = []
    for g, idxs in gene2idxs.items():
        if len(idxs) < 2: continue
        idxs = np.array(idxs)
        domains = iso_n_domains[idxs]
        if domains.std() < 0.1: continue
        gene_pos_terms = np.where(Y_te_v[idxs[0]] > 0)[0]
        if len(gene_pos_terms) == 0: continue
        med = np.median(domains)
        domain_binary = (domains > med).astype(float)
        if domain_binary.sum() == 0 or domain_binary.sum() == len(idxs): continue
        p_g = preds_v[idxs]
        for t in gene_pos_terms:
            scores = p_g[:, t]
            if scores.std() < 1e-8:
                aucs.append(0.5); continue
            try: aucs.append(roc_auc_score(domain_binary, scores))
            except: pass
    return (float(np.mean(aucs)) if aucs else 0.5), len(aucs)

# ── 4. TF MLPs ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
n_go = len(mf_terms)

def build_mlp1(dim):
    inp = layers.Input(shape=(dim,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model(inp, out)

def build_mlp2(da, db):
    ia, ib = layers.Input(shape=(da,)), layers.Input(shape=(db,))
    x = layers.Concatenate()([ia, ib])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model([ia, ib], out)

# config: (name, kind, train_inputs, test_inputs)
CONFIGS = [
    ('L30_only',      'single', X_l30_tr,   X_l30_te),
    ('L15_only',      'single', l15_tr_s,   l15_te_s),
    ('delta_only',    'single', delta_tr_s, delta_te_s),
    ('cat[L30,delta]','concat', (X_l30_tr, delta_tr_s), (X_l30_te, delta_te_s)),  # v17f* ref
    ('cat[L30,L15]',  'concat', (X_l30_tr, l15_tr_s),   (X_l30_te, l15_te_s)),    # B1
]

def macro(preds, idxs):
    aps = [average_precision_score(Y_te[:, i], preds[:, i]) for i in idxs if Y_te[:, i].sum() >= 2]
    return float(np.mean(aps)) if aps else float('nan')

results = {}
print("\n[3] Training + evaluating (macro AUPRC AND within-gene DR-AUC)...")
for name, kind, tr_in, te_in in CONFIGS:
    t0 = time.time(); all_preds = []
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm = np.random.permutation(len(X_l30_tr))
        nv = int(len(X_l30_tr) * 0.1); vi, ti = perm[:nv], perm[nv:]
        if kind == 'single':
            m = build_mlp1(tr_in.shape[1]); m.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
            m.fit(tr_in[ti], Y_tr[ti], validation_data=(tr_in[vi], Y_tr[vi]),
                  epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                  verbose=0)
            all_preds.append(m.predict(te_in, batch_size=1024, verbose=0))
        else:
            a_tr, b_tr = tr_in; a_te, b_te = te_in
            m = build_mlp2(a_tr.shape[1], b_tr.shape[1]); m.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
            m.fit([a_tr[ti], b_tr[ti]], Y_tr[ti], validation_data=([a_tr[vi], b_tr[vi]], Y_tr[vi]),
                  epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                  verbose=0)
            all_preds.append(m.predict([a_te, b_te], batch_size=1024, verbose=0))
    preds = np.mean(all_preds, axis=0)
    auprc_all = macro(preds, valid_idx)
    auprc_l2  = macro(preds, l2_valid)
    dr_auc, n_pairs = compute_domain_ranking_auc(preds[:, valid_mask], gene2idxs, iso_n_domains, Y_te_v)
    results[name] = {'macro_all_mf': auprc_all, 'macro_l2': auprc_l2,
                     'domain_ranking_auc': dr_auc, 'dr_n_pairs': n_pairs,
                     'elapsed_s': round(time.time()-t0)}
    print(f"  {name:<16} macro_All={auprc_all:.4f}  macro_L2={auprc_l2:.4f}  DR-AUC={dr_auc:.4f} (n={n_pairs})  [{results[name]['elapsed_s']}s]")

# ── 5. Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"  {'Config':<16} {'macro All':>10} {'macro L2':>9} {'DR-AUC':>8}")
print("  " + "-" * 46)
for name, r in results.items():
    print(f"  {name:<16} {r['macro_all_mf']:>10.4f} {r['macro_l2']:>9.4f} {r['domain_ranking_auc']:>8.4f}")

dr = {k: results[k]['domain_ranking_auc'] for k in results}
best_single = max(dr['L30_only'], dr['L15_only'])
delta_edge  = dr['cat[L30,delta]'] - best_single
print(f"\n  Best single-layer DR-AUC: {best_single:.4f}")
print(f"  v17f* cat[L30,delta] DR-AUC: {dr['cat[L30,delta]']:.4f}")
print(f"  delta contrast edge over best single layer: {delta_edge:+.4f}")
verdict = ("DELTA EARNS KEEP within-gene (contrast > single layer on DR-AUC)"
           if delta_edge > 0.02 else
           "DELTA NEUTRAL within-gene on brain (single layer ~ contrast on DR-AUC)")
print(f"  Verdict: {verdict}")

results['_meta'] = {'best_single_dr': best_single, 'delta_edge_dr': delta_edge, 'verdict': verdict,
                    'note': 'macro is gene-level (oracle~0.81); DR-AUC is the within-gene axis where delta is claimed to matter'}
json.dump(results, open(f'{OUT_DIR}/results.json', 'w'), indent=2)
print(f"\n[Saved] {OUT_DIR}/results.json")
