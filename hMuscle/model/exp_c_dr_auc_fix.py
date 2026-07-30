#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_c_dr_auc_fix.py
====================
Option C: Fix DR-AUC computation for Option A comparison.

Problem: exp_a_cond_domain_v2.py used wrong DR-AUC function (BISECT TSV column names
         iso_ct/iso_ad/go_id don't exist → n_pairs=0 → NaN for all λ)

Correct method: compute_domain_ranking_auc() from domain_ranking_validation.py
  - Within-gene domain-count ranking AUC
  - Ground truth: domain_count > gene_median → binary label
  - Score: PRISM prediction per GO term
  - Aggregate across (gene, GO_term) pairs

Plan:
  Part 1: Verify baseline PRISM v17f* DR-AUC = 0.630 using correct method
  Part 2: Retrain Option A λ=0.05 (3 seeds), save preds, compute correct DR-AUC
  Report: Does explicit domain supervision improve DR-AUC beyond 0.630?
"""

import os, json, time, gzip
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
FEAT_DIR  = '../results_isoform/features'
BOOT_DIR  = '../../reports/v17f_star_bootstrap'
OUT_DIR   = '../../reports/exp_c_dr_auc_fix'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13]
BATCH_SIZE = 512
EPOCHS     = 50
PATIENCE   = 10
LAYER_A    = 15
LAYER_B    = 30
LAMBDA_TEST = 0.05   # least-bad Option A λ (others produced even worse AUPRC)

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  Option C: Fix DR-AUC Computation")
print(f"  Correct method: compute_domain_ranking_auc (within-gene)")
print("=" * 65, flush=True)

# ── 1. Gene/isoform identity ──────────────────────────────────────────
print("\n[1] Loading identities...", flush=True)
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]
n_iso = len(te_sym_list)

gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs[g].append(i)
print(f"  {n_iso} isoforms, {len(gene2idxs)} genes", flush=True)

# ── 2. Pfam domain counts ─────────────────────────────────────────────
print("\n[2] Domain counts...", flush=True)
domain_mat    = np.load(f'{FEAT_DIR}/domain_matrix_proper_test.npy')
iso_n_domains = domain_mat.sum(axis=1).astype(np.int32)
print(f"  domain_mat: {domain_mat.shape}, ≥1 domain: {(iso_n_domains>0).sum()}", flush=True)

# ── 3. GO labels ──────────────────────────────────────────────────────
print("\n[3] GO labels (from v17f_star_bootstrap Y_te.npy)...", flush=True)
Y_te_all = np.load(f'{BOOT_DIR}/Y_te.npy')   # (36748, 82)
valid_mask = Y_te_all.sum(0) >= 2
Y_te_v    = Y_te_all[:, valid_mask]
print(f"  Y_te: {Y_te_all.shape} → valid: {valid_mask.sum()} terms", flush=True)

# ── 4. Correct DR-AUC function ───────────────────────────────────────
def compute_domain_ranking_auc(preds_mat, gene2idxs, iso_n_domains, Y_te_v):
    """Within-gene domain-count ranking AUC (from domain_ranking_validation.py)."""
    aucs = []
    for g, idxs in gene2idxs.items():
        if len(idxs) < 2: continue
        domains = iso_n_domains[idxs]
        if domains.std() < 0.1: continue
        gene_pos_terms = np.where(Y_te_v[idxs[0]] > 0)[0]
        if len(gene_pos_terms) == 0: continue
        med = np.median(domains)
        domain_binary = (domains > med).astype(float)
        if domain_binary.sum() == 0 or domain_binary.sum() == len(idxs): continue
        p_g = preds_mat[idxs]
        for t in gene_pos_terms:
            scores = p_g[:, t]
            if scores.std() < 1e-8:
                aucs.append(0.5)
                continue
            try:
                aucs.append(roc_auc_score(domain_binary, scores))
            except:
                pass
    return float(np.mean(aucs)) if aucs else 0.5, len(aucs)

# ── 5. Part 1: Verify baseline v17f* DR-AUC ──────────────────────────
print("\n[5] PART 1: Verify PRISM v17f* baseline DR-AUC...", flush=True)
prism_raw   = np.load(f'{BOOT_DIR}/v17f_star_preds.npy').astype(np.float32)
# Apply same valid_mask to align with Y_te_v
prism_v = prism_raw[:, valid_mask] if prism_raw.shape[1] == Y_te_all.shape[1] else prism_raw

dr_prism, n_prism = compute_domain_ranking_auc(prism_v, gene2idxs, iso_n_domains, Y_te_v)
print(f"  PRISM v17f* DR-AUC = {dr_prism:.4f}  (N={n_prism:,} term-pairs)")
print(f"  Expected: 0.6296  [Published in natcomm_v0.md as 0.630]")

# Gene-mean oracle (must = 0.500)
gene_mean = np.zeros_like(prism_v)
for g, idxs in gene2idxs.items():
    m = prism_v[idxs].mean(0)
    for i in idxs: gene_mean[i] = m
dr_gm, n_gm = compute_domain_ranking_auc(gene_mean, gene2idxs, iso_n_domains, Y_te_v)
print(f"  Gene-mean oracle  DR-AUC = {dr_gm:.4f}  (expected: 0.500)")

# ── 6. Part 2: Retrain Option A λ=0.05, save preds, compute DR-AUC ──
print(f"\n[6] PART 2: Retrain Option A λ={LAMBDA_TEST} (3 seeds) + correct DR-AUC...", flush=True)

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"  GPU: {gpus[0].name}", flush=True)
else:
    print("  WARNING: No GPU", flush=True)

# Load embeddings
print("  Loading embeddings...", flush=True)
X_tr_l30 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_tr_l15 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
X_te_l30 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_te_l15 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_A:02d}_t30_150M.npy').astype(np.float32)

from sklearn.preprocessing import MaxAbsScaler
delta_tr   = (X_tr_l30 - X_tr_l15).astype(np.float32)
delta_te   = (X_te_l30 - X_te_l15).astype(np.float32)
scaler     = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler.transform(delta_te).astype(np.float32)

# Domain mask for training
dm_tr = np.load(f'{FEAT_DIR}/domain_matrix_proper_train.npy').astype(np.float32)
dc_tr = dm_tr.sum(1)
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
print(f"  Type 1/2 mask: {n_type12}/{len(tr_genes)}", flush=True)

# GO labels for training
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
n_go = len(mf_terms)

tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

def build_Y_tr(go_id):
    pos_ids  = go_genes_tr[go_id]
    pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in pos_ids}
    y = np.zeros(len(tr_genes), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
    return y

Y_tr     = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)
Y_combined = np.concatenate([
    Y_tr,
    dc_tr[:, None].astype(np.float32),
    type12_mask_tr[:, None].astype(np.float32)
], axis=1)

# valid_idx for AUPRC
valid_idx_tr = [i for i in range(n_go) if Y_te_all[:,i].sum() >= 2]
l2_terms_set = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural': l2_terms_set.add(p[0])
l2_valid_idx = [i for i in range(n_go) if mf_terms[i] in l2_terms_set and Y_te_all[:,i].sum() >= 2]

class ConditionalDomainAuxLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_aux, n_go, **kwargs):
        super().__init__(**kwargs)
        self.lambda_aux = lambda_aux
        self.n_go = n_go

    def call(self, y_true, y_pred):
        y_go_true = y_true[:, :self.n_go]
        dc_true   = y_true[:, self.n_go:self.n_go+1]
        mask      = y_true[:, self.n_go+1:self.n_go+2]
        pred_go   = y_pred[:, :self.n_go]
        pred_dc   = y_pred[:, self.n_go:]
        focal_fn  = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
        loss_go   = focal_fn(y_go_true, pred_go)
        mse_cond  = tf.square(pred_dc - dc_true) * mask
        n_pos     = tf.maximum(tf.reduce_sum(mask), 1.0)
        loss_aux  = tf.reduce_sum(mse_cond) / n_pos
        return loss_go + self.lambda_aux * loss_aux

def build_model(n_go):
    inp_d = layers.Input(shape=(640,), name='delta')
    inp_e = layers.Input(shape=(640,), name='l30')
    x  = layers.Concatenate()([inp_d, inp_e])
    x  = layers.Dense(256, activation='relu')(x)
    x  = layers.BatchNormalization()(x)
    x  = layers.Dropout(0.2)(x)
    h  = layers.Dense(128, activation='relu')(x)
    go_out = layers.Dense(n_go, activation='sigmoid', name='go_out')(h)
    dc_br  = layers.Dense(64, activation='relu')(x)
    dc_out = layers.Dense(1, activation='linear', name='dc_out')(dc_br)
    out = layers.Concatenate(name='combined')([go_out, dc_out])
    return models.Model([inp_d, inp_e], out)

# Train λ=0.05 (3 seeds)
t0 = time.time()
all_preds = []
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm    = np.random.permutation(len(delta_tr_s))
    n_val   = int(len(delta_tr_s) * 0.1)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]

    model = build_model(n_go)
    loss  = ConditionalDomainAuxLoss(lambda_aux=LAMBDA_TEST, n_go=n_go)
    model.compile(optimizer=optimizers.Adam(1e-3), loss=loss)
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                   restore_best_weights=True)]
    model.fit(
        [delta_tr_s[tr_idx], X_tr_l30[tr_idx]], Y_combined[tr_idx],
        validation_data=([delta_tr_s[val_idx], X_tr_l30[val_idx]], Y_combined[val_idx]),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=0
    )
    out_te = model.predict([delta_te_s, X_te_l30], batch_size=1024, verbose=0)
    preds_i = out_te[:, :n_go]
    all_preds.append(preds_i)

    # Per-seed quick stats
    preds_v = preds_i[:, valid_mask]
    dr_seed, n_s = compute_domain_ranking_auc(preds_v, gene2idxs, iso_n_domains, Y_te_v)
    aps = [average_precision_score(Y_te_all[:,i], preds_i[:,i])
           for i in valid_idx_tr if Y_te_all[:,i].sum() >= 2]
    print(f"    seed={seed}  AUPRC={np.mean(aps):.4f}  DR-AUC={dr_seed:.4f}  (N={n_s})", flush=True)

# Ensemble
preds_ens = np.mean(all_preds, axis=0)
np.save(f'{OUT_DIR}/opt_a_lambda005_preds.npy', preds_ens)

preds_ens_v = preds_ens[:, valid_mask]
dr_opt_a, n_opt_a = compute_domain_ranking_auc(preds_ens_v, gene2idxs, iso_n_domains, Y_te_v)
a_all = float(np.mean([average_precision_score(Y_te_all[:,i], preds_ens[:,i])
                        for i in valid_idx_tr if Y_te_all[:,i].sum() >= 2]))
a_l2  = float(np.mean([average_precision_score(Y_te_all[:,i], preds_ens[:,i])
                        for i in l2_valid_idx if Y_te_all[:,i].sum() >= 2]))
elapsed = time.time() - t0

# ── 7. Summary ────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  OPTION C RESULTS SUMMARY")
print(f"{'='*65}")
print(f"\n  Baseline verification (correct compute_domain_ranking_auc):")
print(f"    PRISM v17f*     DR-AUC = {dr_prism:.4f}  (expected: 0.6296)")
print(f"    Gene-mean oracle DR-AUC = {dr_gm:.4f}   (expected: 0.5000)")
print(f"\n  Option A λ={LAMBDA_TEST} (3 seeds, {elapsed:.0f}s):")
print(f"    AUPRC All = {a_all:.4f}  (baseline: 0.734, degradation: {0.734-a_all:.3f})")
print(f"    AUPRC L2  = {a_l2:.4f}  (baseline: 0.637)")
print(f"    DR-AUC    = {dr_opt_a:.4f}  (baseline: {dr_prism:.4f}, Δ={dr_opt_a-dr_prism:+.4f})")
print(f"    N pairs   = {n_opt_a:,}")
print(f"\n  Centroid DR-AUC hurdle: 0.638")
beat = "✓ BEATS centroid" if dr_opt_a > 0.638 else "✗ does NOT beat centroid"
print(f"  Option A λ=0.05: {beat}")
print(f"\n  CONCLUSION:")
if dr_opt_a <= dr_prism:
    print(f"  → DR-AUC DEGRADES with explicit domain supervision (Δ={dr_opt_a-dr_prism:+.4f})")
    print(f"  → Combined with AUPRC crash (Δ={0.734-a_all:-.3f}), Option A is definitively rejected")
    print(f"  → DR-AUC 0.630 is an EMERGENT property of GO-function learning (not improvable via structural supervision)")
else:
    print(f"  → DR-AUC improves to {dr_opt_a:.4f} but AUPRC crashes to {a_all:.4f}")
    print(f"  → DR-AUC gain is NOT worth catastrophic AUPRC loss")
print(f"{'='*65}", flush=True)

# Save results
out_json = {
    'part1_baseline_verification': {
        'prism_v17f_star': {'dr_auc': float(dr_prism), 'n_pairs': int(n_prism), 'expected': 0.6296},
        'gene_mean_oracle': {'dr_auc': float(dr_gm), 'n_pairs': int(n_gm), 'expected': 0.5000},
    },
    'part2_option_a_lambda005': {
        'lambda_aux': LAMBDA_TEST,
        'seeds': SEEDS,
        'auprc_all': float(a_all),
        'auprc_l2': float(a_l2),
        'dr_auc': float(dr_opt_a),
        'n_pairs': int(n_opt_a),
        'delta_dr_auc': float(dr_opt_a - dr_prism),
        'delta_auprc': float(a_all - 0.734),
        'time_s': float(elapsed),
    },
    'conclusion': {
        'beats_centroid_hurdle': bool(dr_opt_a > 0.638),
        'dr_auc_improves': bool(dr_opt_a > dr_prism),
        'verdict': 'REJECTED' if a_all < 0.700 else 'MARGINAL',
    }
}
with open(f'{OUT_DIR}/results.json', 'w') as f:
    json.dump(out_json, f, indent=2)
print(f"\n  Saved: {OUT_DIR}/results.json")
print(f"  Saved: {OUT_DIR}/opt_a_lambda005_preds.npy")
