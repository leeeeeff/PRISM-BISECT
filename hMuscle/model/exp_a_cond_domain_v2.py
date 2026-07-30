#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_a_cond_domain_v2.py
=========================
Option A (fast version): Conditional Domain Auxiliary Loss via Keras .fit()

Key design:
  - Use Keras .fit() for GPU efficiency (vs slow GradientTape loop in v1)
  - Combined model output: [pred_go (n_go), pred_domain_count (1)]
  - Custom loss: focal(Y_go) + lambda_aux * conditional_MSE(domain_count, type12_mask)
  - Conditional masking: type12_mask = 1.0 only for domain_count < max_for_gene isoforms

Sweep: lambda_aux in [0.05, 0.1, 0.3] with 3 seeds each
  (baseline DR-AUC=0.630 from v17f* is already known; hurdle=0.638 centroid)

Goal: Does DR-AUC improve beyond 0.638 (ESM-2 centroid baseline)?
  If yes → conditional auxiliary loss adds isoform-domain-level value to MLP

S2 safety: Type 3 isoforms (same domain, motif-dependent, 4574 training cases)
  receive mask=0.0 → no domain aux gradient → motif-level discrimination preserved
"""

import os, json, time, gzip
import numpy as np
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
FEAT_DIR  = '../results_isoform/features'
OUT_DIR   = '../../reports/exp_a_cond_domain_aux'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13]   # 3 seeds for quick test (full: 5)
BATCH_SIZE = 512
EPOCHS     = 50
PATIENCE   = 10
LAYER_A    = 15
LAYER_B    = 30

LAMBDAS    = [0.05, 0.1, 0.3]  # skip 0.0 — baseline DR-AUC=0.630 already known

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  Option A v2: Conditional Domain Aux (Keras .fit)")
print(f"  λ sweep: {LAMBDAS}  seeds: {SEEDS}")
print("=" * 65, flush=True)

# ── 1. Data ───────────────────────────────────────────────────────────
print("\n[1] Loading embeddings...", flush=True)
X_tr_l30 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_tr_l15 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
X_te_l30 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_te_l15 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
print(f"  Train: {X_tr_l30.shape}  Test: {X_te_l30.shape}", flush=True)

from sklearn.preprocessing import MaxAbsScaler
delta_tr   = (X_tr_l30 - X_tr_l15).astype(np.float32)
delta_te   = (X_te_l30 - X_te_l15).astype(np.float32)
scaler     = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler.transform(delta_te).astype(np.float32)

# ── 2. Domain matrix + Type 1/2 mask ──────────────────────────────────
print("\n[2] Building Type 1/2 mask...", flush=True)
dm_tr = np.load(f'{FEAT_DIR}/domain_matrix_proper_train.npy').astype(np.float32)
dc_tr = dm_tr.sum(1)  # (31668,) domain count

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
gene2iso_tr  = defaultdict(list)
for i, g in enumerate(tr_genes): gene2iso_tr[g].append(i)

type12_mask_tr = np.zeros(len(tr_genes), dtype=np.float32)
for g, idxs in gene2iso_tr.items():
    if len(idxs) < 2: continue
    max_c = dc_tr[idxs].max()
    for idx in idxs:
        if dc_tr[idx] < max_c:
            type12_mask_tr[idx] = 1.0

n_type12 = int(type12_mask_tr.sum())
print(f"  Type 1/2 (domain_loss): {n_type12} / {len(tr_genes)}", flush=True)

# ── 3. GO labels ───────────────────────────────────────────────────────
print("\n[3] Loading GO labels...", flush=True)
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]

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

Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)   # (31668, n_go)
Y_te = np.stack([build_Y_te(go) for go in mf_terms], axis=1)   # (36748, n_go)
valid_mask = Y_te.sum(0) >= 2

L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural': L2_TERMS.add(p[0])

valid_idx = [i for i in range(len(mf_terms)) if valid_mask[i]]
l2_valid  = [i for i in range(len(mf_terms)) if mf_terms[i] in L2_TERMS and valid_mask[i]]
n_go = len(mf_terms)
print(f"  {n_go} MF | valid: {valid_mask.sum()} | L2: {len(l2_valid)}", flush=True)

# ── 4. Domain-Ranking AUC helper ──────────────────────────────────────
import pandas as pd
bisect_df  = pd.read_csv('../../reports/supplementary_table_S_bisect_83cases.tsv', sep='\t')
te_iso_raw = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
te_iso_list = [clean(s) for s in te_iso_raw]
iso2idx    = {iso: i for i, iso in enumerate(te_iso_list)}

def domain_ranking_auc(preds):
    from sklearn.metrics import roc_auc_score
    scores_ct, scores_ad = [], []
    for _, row in bisect_df.iterrows():
        iso_ct = str(row.get('iso_ct', '')).strip()
        iso_ad = str(row.get('iso_ad', '')).strip()
        go_id  = str(row.get('go_id', '')).strip()
        if iso_ct not in iso2idx or iso_ad not in iso2idx: continue
        if go_id not in mf_terms: continue
        gi = mf_terms.index(go_id)
        scores_ct.append(preds[iso2idx[iso_ct], gi])
        scores_ad.append(preds[iso2idx[iso_ad], gi])
    if len(scores_ct) < 10: return float('nan'), len(scores_ct)
    labels = [1]*len(scores_ct) + [0]*len(scores_ad)
    return float(roc_auc_score(labels, scores_ct + scores_ad)), len(scores_ct)

from sklearn.metrics import average_precision_score

def macro_auprc(preds, idxs):
    aps = [average_precision_score(Y_te[:,i], preds[:,i])
           for i in idxs if Y_te[:,i].sum() >= 2]
    return float(np.mean(aps)) if aps else float('nan')

# ── 5. TF / Keras setup ───────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"\n  GPU: {gpus[0].name}", flush=True)
else:
    print("\n  WARNING: No GPU found, running on CPU", flush=True)

# ── 6. Custom combined loss ────────────────────────────────────────────
class ConditionalDomainAuxLoss(tf.keras.losses.Loss):
    """
    y_true: (batch, n_go + 1 + 1)  [Y_go | domain_count | type12_mask]
    y_pred: (batch, n_go + 1)       [pred_go | pred_domain_count]
    """
    def __init__(self, lambda_aux, n_go, **kwargs):
        super().__init__(**kwargs)
        self.lambda_aux = lambda_aux
        self.n_go = n_go

    def call(self, y_true, y_pred):
        y_go_true = y_true[:, :self.n_go]
        dc_true   = y_true[:, self.n_go:self.n_go+1]
        mask      = y_true[:, self.n_go+1:self.n_go+2]

        pred_go = y_pred[:, :self.n_go]
        pred_dc = y_pred[:, self.n_go:]

        # Focal cross-entropy for GO terms
        focal_fn  = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
        loss_go   = focal_fn(y_go_true, pred_go)

        # Conditional domain-count MSE (only Type 1/2)
        mse       = tf.square(pred_dc - dc_true)          # (batch, 1)
        mse_cond  = mse * mask                             # zero for Type 3
        n_pos     = tf.maximum(tf.reduce_sum(mask), 1.0)
        loss_aux  = tf.reduce_sum(mse_cond) / n_pos

        return loss_go + self.lambda_aux * loss_aux

def build_model(n_go, delta_dim=640, esm_dim=640):
    inp_d = layers.Input(shape=(delta_dim,), name='delta')
    inp_e = layers.Input(shape=(esm_dim,), name='l30')
    x  = layers.Concatenate()([inp_d, inp_e])
    x  = layers.Dense(256, activation='relu')(x)
    x  = layers.BatchNormalization()(x)
    x  = layers.Dropout(0.2)(x)
    h  = layers.Dense(128, activation='relu')(x)
    go_out = layers.Dense(n_go, activation='sigmoid', name='go_out')(h)
    # Domain auxiliary head from 256-dim branch
    dc_out = layers.Dense(64, activation='relu')(x)
    dc_out = layers.Dense(1, activation='linear', name='dc_out')(dc_out)
    # Concatenate outputs for single-output Keras loss
    out = layers.Concatenate(name='combined')([go_out, dc_out])  # (n_go+1,)
    return models.Model([inp_d, inp_e], out)

# Combined training labels
# Y_combined: [Y_go | domain_count | type12_mask]  shape (31668, n_go+2)
Y_combined = np.concatenate([
    Y_tr,
    dc_tr[:, None].astype(np.float32),
    type12_mask_tr[:, None].astype(np.float32)
], axis=1)

# ── 7. Lambda sweep ───────────────────────────────────────────────────
sweep_results = {}

for lam in LAMBDAS:
    print(f"\n{'─'*65}")
    print(f"  λ_aux = {lam}  ({len(SEEDS)} seeds)")
    print(f"{'─'*65}", flush=True)
    t0 = time.time()
    all_preds = []

    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm    = np.random.permutation(len(delta_tr_s))
        n_val   = int(len(delta_tr_s) * 0.1)
        val_idx = perm[:n_val]; tr_idx = perm[n_val:]

        model = build_model(n_go)
        loss  = ConditionalDomainAuxLoss(lambda_aux=lam, n_go=n_go)
        model.compile(optimizer=optimizers.Adam(1e-3), loss=loss)

        cb = [callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                       restore_best_weights=True)]

        model.fit(
            [delta_tr_s[tr_idx], X_tr_l30[tr_idx]],
            Y_combined[tr_idx],
            validation_data=(
                [delta_tr_s[val_idx], X_tr_l30[val_idx]],
                Y_combined[val_idx]
            ),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=cb, verbose=0
        )

        out_te = model.predict([delta_te_s, X_te_l30], batch_size=1024, verbose=0)
        preds_i = out_te[:, :n_go]  # drop the domain_count prediction
        all_preds.append(preds_i)

        aps = [average_precision_score(Y_te[:,i], preds_i[:,i])
               for i in valid_idx if Y_te[:,i].sum() >= 2]
        dr, n_pairs = domain_ranking_auc(preds_i)
        print(f"    seed={seed}  AUPRC={np.mean(aps):.4f}  DR-AUC={dr:.4f}", flush=True)

    preds  = np.mean(all_preds, axis=0)
    a_all  = macro_auprc(preds, valid_idx)
    a_l2   = macro_auprc(preds, l2_valid)
    dr_auc, n_pairs = domain_ranking_auc(preds)
    elapsed = time.time() - t0

    sweep_results[str(lam)] = {
        'lambda_aux': lam,
        'auprc_all':  a_all,
        'auprc_l2':   a_l2,
        'domain_ranking_auc': dr_auc,
        'n_dr_pairs': n_pairs,
        'time_s':     elapsed,
    }

    print(f"\n  λ={lam}  All={a_all:.4f}  L2={a_l2:.4f}  DR-AUC={dr_auc:.4f} (n={n_pairs})")
    print(f"  vs baseline v17f*: All=0.734 L2=0.637 DR-AUC=0.630 | centroid DR=0.638")
    print(f"  Time: {elapsed:.0f}s", flush=True)

# ── 8. Summary ────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  RESULTS SUMMARY")
print(f"{'='*65}")
print(f"  λ       All MF    L2       DR-AUC  | vs centroid(0.638)")
for lam_str, r in sweep_results.items():
    beat = '✓' if r['domain_ranking_auc'] > 0.638 else '✗'
    print(f"  {lam_str:<6}  {r['auprc_all']:.4f}    {r['auprc_l2']:.4f}   {r['domain_ranking_auc']:.4f}  {beat}")
print(f"\n  Baseline v17f*: All=0.734  L2=0.637  DR-AUC=0.630")
print(f"  Centroid hurdle: DR-AUC=0.638")
print(f"  S2 safety: {len(tr_genes)-n_type12} Type 3 isoforms excluded from aux loss")
print(f"{'='*65}", flush=True)

with open(f'{OUT_DIR}/results_v2.json', 'w') as f:
    json.dump({
        'sweep': sweep_results,
        'metadata': {
            'n_train': len(tr_genes),
            'n_type12_in_aux': n_type12,
            'n_type3_excluded': len(tr_genes) - n_type12,
            'seeds': SEEDS,
            'lambdas_tested': LAMBDAS,
            'baseline_v17f_star': {'auprc_all': 0.734, 'auprc_l2': 0.637, 'dr_auc': 0.630},
            'centroid_baseline': {'dr_auc': 0.638},
        }
    }, f, indent=2)

print(f"\n  Saved: {OUT_DIR}/results_v2.json")
