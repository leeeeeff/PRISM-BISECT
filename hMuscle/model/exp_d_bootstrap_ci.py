#!/usr/bin/env python3
"""
exp_d_bootstrap_ci.py
=====================
D0 (frozen ESM-2 L30) 5-seed → bootstrap CI on AUPRC.
Also computes v17f vs D0 significance.

Output: reports/exp_d_finetune/d0_bootstrap_ci.json
"""

import os, gzip, json, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/exp_d_finetune'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13, 21, 99]
EPOCHS_MLP = 60
N_BOOT     = 1000
RNG        = np.random.default_rng(0)

# ── GO labels ─────────────────────────────────────────────────────
print("[1] Loading GO labels...")

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

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

go_genes_all = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        if p[7] != 'Function': continue
        go_genes_all[p[2]].add(p[1])

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])

def build_Y_te(go_id):
    pos_ids = go_genes_all[go_id]
    return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0
                     for s in te_sym_list], dtype=np.float32)

Y_te = np.stack([build_Y_te(go) for go in mf_terms], axis=1)
valid_mask = Y_te.sum(0) >= 2
Y_te_v    = Y_te[:, valid_mask]
mf_valid  = [go for go, v in zip(mf_terms, valid_mask) if v]

# H2 layer groups
H2_LAYERS = {}
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12: H2_LAYERS[p[0]] = p[11]

L2_idxs = [i for i, go in enumerate(mf_valid) if H2_LAYERS.get(go) == 'L2_Structural']
L4_idxs = [i for i, go in enumerate(mf_valid) if H2_LAYERS.get(go) == 'L4_CellState']

# Training GO labels
tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
tr_ids = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_ids)

go_genes_tr = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        if p[7] != 'Function': continue
        if p[1] in tr_id_set: go_genes_tr[p[2]].add(p[1])

tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

def build_Y_tr(go_id):
    pos_ids  = go_genes_tr[go_id]
    pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in pos_ids}
    y = np.zeros(len(tr_genes), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
    return y

Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)

print(f"  MF terms total: {len(mf_terms)}  valid (test ≥2): {valid_mask.sum()}")
print(f"  L2_Structural: {len(L2_idxs)}  L4_CellState: {len(L4_idxs)}")

# ── Load pre-computed L30 embeddings ─────────────────────────────
print("\n[2] Loading pre-computed L30 embeddings (D0)...")
X_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X_te = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)
print(f"  train: {X_tr.shape}  test: {X_te.shape}")

scaler = MaxAbsScaler()
X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
X_te_s = scaler.transform(X_te).astype(np.float32)

# ── Build PRISM MLP ───────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def build_prism_mlp(n_in, n_out):
    inp = tf.keras.Input(shape=(n_in,))
    x   = tf.keras.layers.Dense(256, activation='relu')(inp)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(128, activation='relu')(x)
    x   = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_out, activation='sigmoid')(x)
    return tf.keras.Model(inp, out)

# ── Train 5 seeds, save per-seed predictions ──────────────────────
print("\n[3] Training 5 seeds for D0...")
n_terms = Y_tr.shape[1]
seed_preds = []
t0 = time.time()

for seed in SEEDS:
    tf.random.set_seed(seed); np.random.seed(seed)
    model = build_prism_mlp(X_tr_s.shape[1], n_terms)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)
    )
    model.fit(X_tr_s, Y_tr, epochs=EPOCHS_MLP, batch_size=512, verbose=0)
    pred = model.predict(X_te_s, verbose=0)  # (n_te, n_terms)
    seed_preds.append(pred[:, valid_mask])   # only valid terms
    elapsed = time.time() - t0
    print(f"  Seed {seed}: done  [{elapsed:.0f}s]")

# Ensemble average (same as production)
pred_avg = np.mean(seed_preds, axis=0)  # (n_te, n_valid)

# ── Point estimate ────────────────────────────────────────────────
def auprc_summary(pred, labels, l2_idx, l4_idx):
    aps_all = [average_precision_score(labels[:, j], pred[:, j])
               for j in range(labels.shape[1]) if labels[:, j].sum() >= 2]
    l2_aps  = [average_precision_score(labels[:, j], pred[:, j])
               for j in l2_idx if labels[:, j].sum() >= 2]
    l4_aps  = [average_precision_score(labels[:, j], pred[:, j])
               for j in l4_idx if labels[:, j].sum() >= 2]
    return {
        'all_mf': float(np.mean(aps_all)),
        'l2': float(np.mean(l2_aps)) if l2_aps else float('nan'),
        'l4': float(np.mean(l4_aps)) if l4_aps else float('nan'),
        'per_term': [float(x) for x in aps_all]
    }

d0_point = auprc_summary(pred_avg, Y_te_v, L2_idxs, L4_idxs)
print(f"\n  D0 point estimate: All={d0_point['all_mf']:.4f}  L2={d0_point['l2']:.4f}  L4={d0_point['l4']:.4f}")

# ── Bootstrap CI (resample test isoforms, N=1000) ─────────────────
print(f"\n[4] Bootstrap CI (n={N_BOOT} resamples)...")
n_te = Y_te_v.shape[0]
boot_all = np.zeros(N_BOOT)
boot_l2  = np.zeros(N_BOOT)

for b in range(N_BOOT):
    idx = RNG.integers(0, n_te, size=n_te)
    pred_b = pred_avg[idx]
    labs_b = Y_te_v[idx]
    aps = [average_precision_score(labs_b[:, j], pred_b[:, j])
           for j in range(labs_b.shape[1]) if labs_b[:, j].sum() >= 2]
    boot_all[b] = np.mean(aps)
    l2_aps = [average_precision_score(labs_b[:, j], pred_b[:, j])
              for j in L2_idxs if labs_b[:, j].sum() >= 2]
    boot_l2[b] = np.mean(l2_aps) if l2_aps else float('nan')

ci_all = (float(np.percentile(boot_all, 2.5)), float(np.percentile(boot_all, 97.5)))
ci_l2  = (float(np.nanpercentile(boot_l2, 2.5)), float(np.nanpercentile(boot_l2, 97.5)))

print(f"  D0 All MF: {d0_point['all_mf']:.4f}  95% CI [{ci_all[0]:.4f}, {ci_all[1]:.4f}]")
print(f"  D0 L2:     {d0_point['l2']:.4f}  95% CI [{ci_l2[0]:.4f}, {ci_l2[1]:.4f}]")

# ── v17f reference (from saved bootstrap in v17f_bootstrap_ci.py) ─
V17F_ALL = 0.7173
V17F_L2  = 0.6219
PRISM_ALL = 0.5962
PRISM_L2  = 0.3127

# One-sample test: is v17f significantly > D0?
# Using bootstrap distribution of D0 as null, check if V17F > upper CI
gap_all = V17F_ALL - d0_point['all_mf']
gap_l2  = V17F_L2  - d0_point['l2']
p_val_all = float(np.mean(boot_all >= V17F_ALL))  # P(bootstrap D0 >= v17f)
p_val_l2  = float(np.mean(boot_l2  >= V17F_L2))

print(f"\n  v17f All MF = {V17F_ALL:.4f}  gap from D0 = {gap_all:+.4f}  P(D0 ≥ v17f) = {p_val_all:.4f}")
print(f"  v17f L2     = {V17F_L2:.4f}  gap from D0 = {gap_l2:+.4f}  P(D0 ≥ v17f) = {p_val_l2:.4f}")

# ── D1 values (from prior run) ─────────────────────────────────────
D1_ALL = 0.6606; D1_L2 = 0.5691
gap_d1_all = D1_ALL - d0_point['all_mf']
gap_d1_l2  = D1_L2  - d0_point['l2']
p_val_d1_all = float(np.mean(boot_all >= D1_ALL))
p_val_d1_l2  = float(np.mean(boot_l2  >= D1_L2))

print(f"\n  D1 All MF = {D1_ALL:.4f}  gap from D0 = {gap_d1_all:+.4f}  P(D0 ≥ D1) = {p_val_d1_all:.4f}")
print(f"  D1 L2     = {D1_L2:.4f}  gap from D0 = {gap_d1_l2:+.4f}  P(D0 ≥ D1) = {p_val_d1_l2:.4f}")

# ── Save ──────────────────────────────────────────────────────────
results = {
    'D0': {
        'all_mf': d0_point['all_mf'], 'l2': d0_point['l2'], 'l4': d0_point['l4'],
        'ci_all_95': ci_all, 'ci_l2_95': ci_l2,
        'n_boot': N_BOOT, 'seeds': SEEDS
    },
    'D1': {
        'all_mf': D1_ALL, 'l2': D1_L2,
        'delta_all': gap_d1_all, 'delta_l2': gap_d1_l2,
        'p_D0_ge_D1_all': p_val_d1_all, 'p_D0_ge_D1_l2': p_val_d1_l2
    },
    'v17f': {
        'all_mf': V17F_ALL, 'l2': V17F_L2,
        'delta_all_vs_D0': gap_all, 'delta_l2_vs_D0': gap_l2,
        'p_D0_ge_v17f_all': p_val_all, 'p_D0_ge_v17f_l2': p_val_l2
    },
    'PRISM': {'all_mf': PRISM_ALL, 'l2': PRISM_L2},
    'manuscript_text': {
        'D0_vs_v17f_all': f"v17f ({V17F_ALL:.3f}) vs D0 ({d0_point['all_mf']:.3f}), Δ={gap_all:+.3f}, P={p_val_all:.4f}",
        'D0_CI_all': f"{d0_point['all_mf']:.3f} (95% CI: {ci_all[0]:.3f}–{ci_all[1]:.3f})",
        'D0_CI_l2':  f"{d0_point['l2']:.3f} (95% CI: {ci_l2[0]:.3f}–{ci_l2[1]:.3f})"
    }
}

out_path = f'{OUT_DIR}/d0_bootstrap_ci.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[Done] {out_path}")
print("\n===  MANUSCRIPT SUMMARY  ===")
print(f"  D0 All MF: {results['manuscript_text']['D0_CI_all']}")
print(f"  D0 L2:     {results['manuscript_text']['D0_CI_l2']}")
print(f"  v17f vs D0 (All MF): {results['manuscript_text']['D0_vs_v17f_all']}")
