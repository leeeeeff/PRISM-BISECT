#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_a_conditional_domain_aux.py
================================
Option A: v17f* + Conditional Domain Auxiliary Loss

Architecture:
  Base: same as v17f_abl_no_tpsi (no T_ψ triplet)
    [δ_layer(640), φ_L30(640)] → Dense(256,relu) → BN → Drop(0.2) → Dense(128,relu) → Dense(82,sigmoid)

  Additional domain auxiliary head (branching from 256-dim hidden):
    256-dim → Dense(64,relu) → Dense(1, linear)  ← domain_count regression

  Conditional masking:
    - Type 1/2 isoforms: domain_count < max_for_gene (domain loss relative to canonical)
    - Type 3 isoforms: domain_count == max_for_gene (same domain as canonical)
    - Auxiliary loss applies ONLY to Type 1/2 → preserves motif-level S2 discrimination

  Loss:
    focal = BinaryFocalCrossentropy(gamma=2)(Y_go, pred_go)
    domain_loss = MSE(pred_count, true_count) * type12_mask  (only Type 1/2)
    total = focal + lambda_aux * mean(domain_loss_masked)

S2 safety justification (from exp_feature_attribution.py results):
  - PRISM currently has minimal motif-level discrimination (2.8% of same-domain pairs >0.05 gap)
  - Conditional auxiliary loss ONLY affects Type 1/2 isoforms (domain_count_diff > 0)
  - Type 3 isoforms (same domain, motif-dependent) see no additional gradient → preserved

Research question:
  Does conditional domain auxiliary loss improve Domain-Ranking AUC from 0.630 → >0.638?
  (0.638 = ESM-2 centroid baseline, the MLP-contribution hurdle)

Outputs:
  reports/exp_a_cond_domain_aux/
    results.json   — AUPRC + Domain-Ranking AUC + lambda sweep
    run.log
"""

import os, json, time
import numpy as np
from collections import defaultdict
import gzip, warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = '../data'
ID_DIR     = '../data/raw_data/data/id_lists'
ANNOT_DIR  = '../data/raw_data/data/annotations'
FEAT_DIR   = '../results_isoform/features'
OUT_DIR    = '../../reports/exp_a_cond_domain_aux'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13, 21, 99]
BATCH_MLP  = 512
EPOCHS_MLP = 60
LAYER_A    = 15
LAYER_B    = 30
LAMBDA_AUX = 0.1   # start point; sweep 0.0, 0.05, 0.1, 0.3

# ── helpers ──────────────────────────────────────────────────────────
def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  Option A: Conditional Domain Auxiliary Loss")
print(f"  λ_aux = {LAMBDA_AUX} (sweep 0, 0.05, 0.1, 0.3)")
print("=" * 65)

# ── 1. Data ───────────────────────────────────────────────────────────
print("\n[1] Loading embeddings...")
X_tr_l30 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_tr_l15 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
X_te_l30 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_te_l15 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
print(f"  Train: {X_tr_l30.shape}  Test: {X_te_l30.shape}")

print("\n[2] Computing δ_layer = L30 − L15...")
from sklearn.preprocessing import MaxAbsScaler
delta_tr   = (X_tr_l30 - X_tr_l15).astype(np.float32)
delta_te   = (X_te_l30 - X_te_l15).astype(np.float32)
scaler     = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler.transform(delta_te).astype(np.float32)

# ── 3. Domain matrix + Type 1/2 mask ─────────────────────────────────
print("\n[3] Building Type 1/2 mask (conditional auxiliary)...")
dm_tr = np.load(f'{FEAT_DIR}/domain_matrix_proper_train.npy').astype(np.float32)  # (31668,512)
dm_te = np.load(f'{FEAT_DIR}/domain_matrix_proper_test.npy').astype(np.float32)   # (36748,512)
dc_tr = dm_tr.sum(1)  # domain count per training isoform
dc_te = dm_te.sum(1)

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
gene2iso_tr  = defaultdict(list)
for i, g in enumerate(tr_genes): gene2iso_tr[g].append(i)

# Type 1/2 mask: domain_count < max for that gene, multi-isoform only
type12_mask_tr = np.zeros(len(tr_genes), dtype=np.float32)
for g, idxs in gene2iso_tr.items():
    if len(idxs) < 2: continue
    max_c = dc_tr[idxs].max()
    for idx in idxs:
        if dc_tr[idx] < max_c:
            type12_mask_tr[idx] = 1.0

n_type12 = type12_mask_tr.sum()
print(f"  Train Type 1/2 (domain_loss): {n_type12:.0f} / {len(tr_genes)} = {n_type12/len(tr_genes)*100:.1f}%")
print(f"  Type 3 + no-domain (excluded from aux loss): {len(tr_genes)-n_type12:.0f}")

# ── 4. Gene IDs & GO labels ───────────────────────────────────────────
print("\n[4] Loading GO labels (82 MF)...")
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

# ── 5. Domain-Ranking data (BISECT 83 cases) ──────────────────────────
print("\n[5] Loading Domain-Ranking data...")
import pandas as pd
bisect_path = '../../reports/supplementary_table_S_bisect_83cases.tsv'
bisect_df   = pd.read_csv(bisect_path, sep='\t')
# Map test isoform list
te_iso_raw  = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
te_iso_list = [clean(s) for s in te_iso_raw]
iso2idx     = {iso: i for i, iso in enumerate(te_iso_list)}

def domain_ranking_auc(preds):
    """AUC for ranking CT isoform higher than AD isoform within-gene (n=83 pairs)."""
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
    scores  = scores_ct + scores_ad
    return float(roc_auc_score(labels, scores)), len(scores_ct)

# ── 6. TF setup ───────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"\n  GPU: {gpus[0].name}")

focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
n_go     = len(mf_terms)

from sklearn.metrics import average_precision_score

def macro_auprc(preds, idxs):
    aps = [average_precision_score(Y_te[:,i], preds[:,i])
           for i in idxs if Y_te[:,i].sum() >= 2]
    return float(np.mean(aps)) if aps else float('nan')

# ── 7. Build model with conditional domain auxiliary head ──────────────
def build_model_with_domain_head(delta_dim=640, esm_dim=640, n_go=82):
    """
    Returns a model with two outputs:
      go_out:     (n, n_go)  — GO prediction (main objective)
      domain_out: (n, 1)     — domain count regression (auxiliary, conditional)
    """
    inp_d = layers.Input(shape=(delta_dim,), name='delta')
    inp_e = layers.Input(shape=(esm_dim,), name='esm_l30')

    x = layers.Concatenate()([inp_d, inp_e])      # 1280-dim
    h256 = layers.Dense(256, activation='relu', name='hidden_256')(x)
    h256 = layers.BatchNormalization()(h256)
    h256 = layers.Dropout(0.2)(h256)
    h128 = layers.Dense(128, activation='relu', name='hidden_128')(h256)

    go_out     = layers.Dense(n_go, activation='sigmoid', name='go_out')(h128)
    domain_out = layers.Dense(64, activation='relu', name='domain_h64')(h256)
    domain_out = layers.Dense(1, activation='linear', name='domain_count')(domain_out)

    return models.Model([inp_d, inp_e], [go_out, domain_out], name='PRISM_DomainAux')

# ── 8. Lambda sweep ───────────────────────────────────────────────────
lambdas   = [0.0, 0.05, 0.1, 0.3]
sweep_results = {}

for lam in lambdas:
    print(f"\n{'─'*65}")
    print(f"  λ_aux = {lam}")
    print(f"{'─'*65}")
    t0 = time.time()
    all_preds = []

    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm    = np.random.permutation(len(delta_tr_s))
        n_val   = int(len(delta_tr_s) * 0.1)
        val_idx = perm[:n_val]; tr_idx = perm[n_val:]

        model = build_model_with_domain_head(n_go=n_go)
        opt   = optimizers.Adam(1e-3)

        # Prepare tensors
        X_tr_d  = tf.constant(delta_tr_s[tr_idx])
        X_tr_e  = tf.constant(X_tr_l30[tr_idx])
        Y_tr_go = tf.constant(Y_tr[tr_idx])
        dc_tr_t = tf.constant(dc_tr[tr_idx, None].astype(np.float32))  # (n,1)
        mask_t  = tf.constant(type12_mask_tr[tr_idx, None].astype(np.float32))

        X_val_d  = tf.constant(delta_tr_s[val_idx])
        X_val_e  = tf.constant(X_tr_l30[val_idx])
        Y_val_go = tf.constant(Y_tr[val_idx])
        dc_val_t = tf.constant(dc_tr[val_idx, None].astype(np.float32))
        mask_val = tf.constant(type12_mask_tr[val_idx, None].astype(np.float32))

        n_train   = len(tr_idx)
        n_batches = (n_train + BATCH_MLP - 1) // BATCH_MLP

        best_val_loss = np.inf
        best_weights  = None
        patience      = 10
        pat_cnt       = 0

        for epoch in range(EPOCHS_MLP):
            # Shuffle
            perm_e = tf.random.shuffle(tf.range(n_train))
            epoch_loss = 0.0

            for b in range(n_batches):
                bidx = perm_e[b*BATCH_MLP:(b+1)*BATCH_MLP]
                xd   = tf.gather(X_tr_d, bidx)
                xe   = tf.gather(X_tr_e, bidx)
                ygo  = tf.gather(Y_tr_go, bidx)
                dc_b = tf.gather(dc_tr_t, bidx)
                msk  = tf.gather(mask_t, bidx)

                with tf.GradientTape() as tape:
                    go_pred, dc_pred = model([xd, xe], training=True)
                    loss_go  = focal_fn(ygo, go_pred)
                    # Conditional domain auxiliary loss (only Type 1/2)
                    if lam > 0:
                        mse_all  = tf.square(dc_pred - dc_b)       # (batch,1)
                        mse_cond = mse_all * msk                    # zero for Type 3
                        # Normalize by # of Type 1/2 in batch (avoid division by zero)
                        n_type12_b = tf.maximum(tf.reduce_sum(msk), 1.0)
                        loss_aux   = tf.reduce_sum(mse_cond) / n_type12_b
                        loss       = loss_go + lam * loss_aux
                    else:
                        loss = loss_go

                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss += float(loss)

            # Validation loss (focal only for model selection)
            go_val, _ = model([X_val_d, X_val_e], training=False)
            val_loss  = float(focal_fn(Y_val_go, go_val))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = model.get_weights()
                pat_cnt = 0
            else:
                pat_cnt += 1
                if pat_cnt >= patience:
                    break

        model.set_weights(best_weights)
        go_pred_te, _ = model([delta_te_s, X_te_l30], training=False)
        preds_i = go_pred_te.numpy()
        all_preds.append(preds_i)

        aps = [average_precision_score(Y_te[:,i], preds_i[:,i])
               for i in valid_idx if Y_te[:,i].sum() >= 2]
        print(f"    seed={seed}  AUPRC={np.mean(aps):.4f}  (epoch={epoch+1})")

    preds   = np.mean(all_preds, axis=0)
    a_all   = macro_auprc(preds, valid_idx)
    a_l2    = macro_auprc(preds, l2_valid)
    dr_auc, n_pairs = domain_ranking_auc(preds)
    elapsed = time.time() - t0

    sweep_results[lam] = {
        'lambda_aux': lam,
        'auprc_all':  a_all,
        'auprc_l2':   a_l2,
        'domain_ranking_auc': dr_auc,
        'n_dr_pairs': n_pairs,
        'time_s': elapsed,
    }

    print(f"\n  λ={lam}  All MF={a_all:.4f}  L2={a_l2:.4f}  DR-AUC={dr_auc:.4f} (n={n_pairs})")
    print(f"  Reference: v17f* All=0.734  L2=0.637  DR-AUC=0.630  centroid=0.638")

# ── 9. Summary ────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  CONDITIONAL DOMAIN AUXILIARY LOSS SWEEP RESULTS")
print(f"{'='*65}")
print(f"  λ       All MF    L2       DR-AUC")
for lam, r in sweep_results.items():
    marker = ' ← BEST' if r['domain_ranking_auc'] == max(v['domain_ranking_auc'] for v in sweep_results.values()) else ''
    print(f"  {lam:<6}  {r['auprc_all']:.4f}    {r['auprc_l2']:.4f}   {r['domain_ranking_auc']:.4f}{marker}")

print(f"\n  Baseline: v17f*  All=0.734  L2=0.637  DR-AUC=0.630")
print(f"  Hurdle (centroid): DR-AUC=0.638 — beating this proves MLP adds domain-level value")
print(f"\n  S2 safety: conditional mask excluded {len(tr_genes)-n_type12:.0f} Type 3 isoforms")
print(f"  from auxiliary loss → motif-level discrimination preserved")
print(f"{'='*65}")

with open(f'{OUT_DIR}/results.json', 'w') as f:
    json.dump({
        'sweep': sweep_results,
        'metadata': {
            'n_train': len(tr_genes),
            'n_type12': int(n_type12),
            'n_type3_excluded': int(len(tr_genes) - n_type12),
            'baseline_v17f_star': {'auprc_all': 0.734, 'auprc_l2': 0.637, 'dr_auc': 0.630},
            'centroid_hurdle': {'dr_auc': 0.638},
            'design_note': (
                'Conditional auxiliary loss applied ONLY to Type 1/2 isoforms '
                '(domain_count < max for gene). Type 3 (same domain, S2 motif-dependent) '
                'excluded to preserve motif-level sensitivity.'
            )
        }
    }, f, indent=2)

print(f"\n  Results saved to {OUT_DIR}/results.json")
