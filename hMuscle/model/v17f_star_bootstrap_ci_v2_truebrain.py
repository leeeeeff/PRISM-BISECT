#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_star_bootstrap_ci_v2_truebrain.py
---------------------------------------
TISSUE-MISLABELING BUGFIX RERUN (2026-07-14).
Original v17f_star_bootstrap_ci.py loaded my_gene_list_fixed.npy / esm2_layer_{NN}_t30_150M.npy
as its "test" set -- these are MUSCLE data (BambuTx/NM_ IDs, 36748 isoforms),
despite this script being the actual source of the manuscript's headline
"brain zero-shot v17f*" number (0.7343 all-MF / 0.6366 L2_Structural,
CI[0.7227,0.7470] / CI[0.6169,0.6543]). This rerun re-points the TEST side
only at the TRUE brain isoform set (hMuscle/data/brain_isoquant_esm2/full/brain_full_*,
IsoQuant IDs like A1BG-204, 63994 isoforms / 18514 unique genes). Training side
(train_gene_list.npy etc.) is UNCHANGED.

This script also writes v17f_star_preds.npy + Y_te.npy, which
domain_ranking_validation.py and bisect_prism_score_comparison.py load
downstream -- their _v2_truebrain reruns depend on this script's output.

Original (mislabeled-as-brain, actually muscle): v17f_star_bootstrap_ci.py, backed up
at v17f_star_bootstrap_ci_backup_20260714.py before this rerun was created.

Gene-level bootstrap CI (B=1000) for v17f* (no T_psi).
Architecture: concat[delta_layer(640_scaled), L30(640)] -> 1280-dim MLP.
"""

import os, json, gzip, time
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
OUT_DIR   = '../../reports/truebrain_rerun_20260714/v17f_star_bootstrap'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS       = [42, 7, 13, 21, 99]
BATCH_MLP   = 512
EPOCHS_MLP  = 60
LAYER_A     = 15
LAYER_B     = 30
B_BOOTSTRAP = 1000

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print(f"  v17f* Bootstrap CI (no T_psi, gene-level, B={B_BOOTSTRAP})")
print("=" * 65)

# ── 1. Data ───────────────────────────────────────────────────────
print("\n[1] Loading data...")
X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l15_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
X_l30_te = np.load(f'{BRAIN_DIR}/brain_full_esm2_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l15_te = np.load(f'{BRAIN_DIR}/brain_full_esm2_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
print(f"  Train: {X_l30_tr.shape}  Test: {X_l30_te.shape}")

# ── 2. delta_layer ────────────────────────────────────────────────
print("\n[2] Computing delta_layer = L30 - L15...")
delta_tr   = (X_l30_tr - X_l15_tr).astype(np.float32)
delta_te   = (X_l30_te - X_l15_te).astype(np.float32)
scaler     = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler.transform(delta_te).astype(np.float32)

# ── 3. IDs ────────────────────────────────────────────────────────
print("\n[3] Loading gene IDs...")
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
gene_arr_tr  = np.array(tr_genes)

# TRUE BRAIN: brain_full_gene_names.npy already contains gene SYMBOLS (e.g. 'A1BG'),
# not ENSG IDs, so no ENSG2SYM mapping step is needed here (unlike the muscle test set).
te_genes_raw = np.load(f'{BRAIN_DIR}/brain_full_gene_names.npy', allow_pickle=True)
te_sym_list  = [clean(g) for g in te_genes_raw]
gene_arr_te  = np.array(te_sym_list)

gene2idxs_tr = defaultdict(list)
for i, g in enumerate(tr_genes): gene2idxs_tr[g].append(i)

gene2idxs_te = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs_te[g].append(i)

te_unique_genes = list(gene2idxs_te.keys())
print(f"  Train: {len(tr_genes)} isoforms, {len(gene2idxs_tr)} genes")
print(f"  Test:  {len(te_sym_list)} isoforms, {len(te_unique_genes)} unique genes")

# ── 4. GO labels ──────────────────────────────────────────────────
print("\n[4] Loading GO labels (82 MF terms)...")
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
print(f"  {len(mf_terms)} MF | valid: {valid_mask.sum()} | L2_Structural: {len(l2_valid)}")

# ── 5. TF + 5-seed ensemble (no T_psi) ───────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"\n  GPU: {gpus[0].name}")

def build_mlp_no_tpsi(delta_dim=640, esm_dim=640, n_go=82):
    inp_d = layers.Input(shape=(delta_dim,))
    inp_e = layers.Input(shape=(esm_dim,))
    x     = layers.Concatenate()([inp_d, inp_e])
    x     = layers.Dense(256, activation='relu')(x)
    x     = layers.BatchNormalization()(x)
    x     = layers.Dropout(0.2)(x)
    x     = layers.Dense(128, activation='relu')(x)
    out   = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model([inp_d, inp_e], out, name='MLP_no_tpsi')

focal_fn  = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
n_go      = len(mf_terms)
all_preds = []

print(f"\n[5] 5-seed ensemble (no T_psi, {EPOCHS_MLP} epochs)...")
t0 = time.time()
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm    = np.random.permutation(len(delta_tr_s))
    n_val   = int(len(delta_tr_s) * 0.1)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    mlp = build_mlp_no_tpsi(n_go=n_go)
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
    mlp.fit(
        [delta_tr_s[tr_idx], X_l30_tr[tr_idx]], Y_tr[tr_idx],
        validation_data=([delta_tr_s[val_idx], X_l30_tr[val_idx]], Y_tr[val_idx]),
        epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    preds_i = mlp.predict([delta_te_s, X_l30_te], batch_size=1024, verbose=0)
    all_preds.append(preds_i)
    aps = [average_precision_score(Y_te[:, i], preds_i[:, i])
           for i in valid_idx if Y_te[:, i].sum() >= 2]
    print(f"  seed={seed}  AUPRC={np.mean(aps):.4f}")

preds_star = np.mean(all_preds, axis=0)
print(f"  Ensemble done: {time.time()-t0:.0f}s")
np.save(f'{OUT_DIR}/v17f_star_preds.npy', preds_star)
np.save(f'{OUT_DIR}/Y_te.npy', Y_te)

# ── 6. Point estimates ────────────────────────────────────────────
def macro_auprc(preds, Y, idxs):
    vals = [average_precision_score(Y[:, i], preds[:, i])
            for i in idxs if Y[:, i].sum() >= 2]
    return float(np.mean(vals)) if vals else float('nan'), len(vals)

pt_all, n_all = macro_auprc(preds_star, Y_te, valid_idx)
pt_l2,  n_l2  = macro_auprc(preds_star, Y_te, l2_valid)

print(f"\n[6] Point estimates:")
print(f"  v17f* All MF:     {pt_all:.4f}  (ablation ref: 0.7325)")
print(f"  v17f* L2_Struct:  {pt_l2:.4f}  (ablation ref: 0.6333)")

# ── 7. Gene-level bootstrap CI ────────────────────────────────────
print(f"\n[7] Gene-level bootstrap CI (B={B_BOOTSTRAP})...")
t0   = time.time()
rng2 = np.random.default_rng(0)
gene_to_te_idx = {g: np.array(idxs) for g, idxs in gene2idxs_te.items()}
N_genes        = len(te_unique_genes)

boot_all = np.zeros(B_BOOTSTRAP)
boot_l2  = np.zeros(B_BOOTSTRAP)
valid_b  = {'all': 0, 'l2': 0}

for b in range(B_BOOTSTRAP):
    sampled_genes = rng2.choice(te_unique_genes, size=N_genes, replace=True)
    boot_idxs     = np.concatenate([gene_to_te_idx[g] for g in sampled_genes])
    Y_b = Y_te[boot_idxs]
    P_b = preds_star[boot_idxs]

    vals_all = [average_precision_score(Y_b[:, i], P_b[:, i])
                for i in valid_idx if Y_b[:, i].sum() >= 2]
    if vals_all:
        boot_all[b] = np.mean(vals_all); valid_b['all'] += 1

    vals_l2 = [average_precision_score(Y_b[:, i], P_b[:, i])
               for i in l2_valid if Y_b[:, i].sum() >= 2]
    if vals_l2:
        boot_l2[b] = np.mean(vals_l2); valid_b['l2'] += 1

    if (b + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta     = elapsed / (b + 1) * (B_BOOTSTRAP - b - 1)
        print(f"  Bootstrap {b+1:4d}/{B_BOOTSTRAP}  "
              f"mean_all={boot_all[:b+1][boot_all[:b+1]>0].mean():.4f}  "
              f"mean_l2={boot_l2[:b+1][boot_l2[:b+1]>0].mean():.4f}  "
              f"ETA={eta:.0f}s")

print(f"\n  Bootstrap done: {time.time()-t0:.0f}s")

ci_all = np.percentile(boot_all[boot_all > 0], [2.5, 97.5])
ci_l2  = np.percentile(boot_l2[boot_l2   > 0], [2.5, 97.5])

# NOTE: these reference constants are the OLD muscle-mislabeled-as-brain numbers
# (kept only so this script's console output shows old-vs-new side by side for
# quick sanity-checking; they are NOT used in any TRUE-brain computation above).
PRISM_ALL = 0.5962;  PRISM_L2  = 0.3501
V17F_ALL  = 0.7173;  V17F_L2   = 0.6219
D0_ALL    = 0.6597;  D0_L2     = 0.5207

print(f"\n{'='*70}")
print(f"  Bootstrap CI Results — v17f* (no T_psi, B={B_BOOTSTRAP})")
print(f"{'='*70}")
print(f"\n  Metric                           Point       95% CI")
print(f"  -------------------------------------------------------")
print(f"  v17f* All MF (81 valid)          {pt_all:.4f}  [{ci_all[0]:.4f}, {ci_all[1]:.4f}]")
print(f"  v17f* L2_Structural (33 valid)   {pt_l2:.4f}  [{ci_l2[0]:.4f}, {ci_l2[1]:.4f}]")
print(f"\n  Reference (no CI):")
print(f"    PRISM:          {PRISM_ALL}  / L2: {PRISM_L2}")
print(f"    D0 frozen L30:  {D0_ALL}  / L2: {D0_L2}")
print(f"    v17f (T_psi):   {V17F_ALL}  / L2: {V17F_L2}")
print(f"\n  CI lower bound comparisons:")
print(f"    > PRISM All MF?  {ci_all[0]:.4f} > {PRISM_ALL} → {'YES' if ci_all[0] > PRISM_ALL else 'NO'}")
print(f"    > PRISM L2?      {ci_l2[0]:.4f} > {PRISM_L2}  → {'YES' if ci_l2[0] > PRISM_L2 else 'NO'}")
print(f"    > D0 All MF?     {ci_all[0]:.4f} > {D0_ALL} → {'YES' if ci_all[0] > D0_ALL else 'NO'}")
print(f"    > v17f All MF?   {ci_all[0]:.4f} > {V17F_ALL} → {'YES' if ci_all[0] > V17F_ALL else 'NO'}")

results = {
    'model': 'v17f_star_no_tpsi',
    'point_all_mf':  pt_all,
    'point_l2':      pt_l2,
    'ci_all_mf':     list(ci_all),
    'ci_l2':         list(ci_l2),
    'references': {
        'prism':  {'all_mf': PRISM_ALL, 'l2': PRISM_L2},
        'd0':     {'all_mf': D0_ALL,    'l2': D0_L2},
        'v17f':   {'all_mf': V17F_ALL,  'l2': V17F_L2},
    },
    'ci_lower_beats': {
        'prism_all':  bool(ci_all[0] > PRISM_ALL),
        'prism_l2':   bool(ci_l2[0]  > PRISM_L2),
        'd0_all':     bool(ci_all[0] > D0_ALL),
        'v17f_all':   bool(ci_all[0] > V17F_ALL),
    },
    'B': B_BOOTSTRAP,
    'valid_boot': valid_b,
    'n_terms': {'all_valid': n_all, 'l2_valid': n_l2},
}
with open(f'{OUT_DIR}/v17f_star_ci.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"\n  [Saved] {OUT_DIR}/v17f_star_ci.json")

print("\n" + "=" * 65)
print("  v17f* Bootstrap CI — COMPLETE")
print("=" * 65)
