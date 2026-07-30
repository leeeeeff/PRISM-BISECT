#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_capacity_baselines.py
--------------------------
Equal-capacity baselines for v17f* (no T_psi, concat[L30, delta]=1280-dim).

Question: Is delta_layer = L30-L15 special, or is any 1280-dim input equally good?

Baselines tested (all 1280-dim input, identical MLP, identical hyperparameters):
  B0: concat[L30, delta=L30-L15]   <- v17f* reference (should match 0.734)
  B1: concat[L30, L15_raw]          <- same layers, NO subtraction (is subtraction key?)
  B2: concat[L30, L25]              <- arbitrary layer near L30 (is L15 special?)
  B3: concat[L30, random-Gaussian]  <- pure capacity (is ANY 640-dim signal sufficient?)

Scaling: second component MaxAbsScaler in all cases (matches v17f* preprocessing).
         L30 is raw (unscaled) in all cases.

Interpretation:
  B3 ~ v17f* : gain is pure capacity artifact -> delta_layer novelty refuted
  B2 ~ v17f* : any layer works -> L15 is arbitrary -> novelty refuted
  B1 ~ v17f* : subtraction unnecessary -> novelty partially refuted
  B1 < v17f* : delta = L30-L15 is specifically better than raw L15 concat
  All B < v17f* : delta_layer is genuinely special -> novelty defended
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
OUT_DIR   = '../../reports/v17f_capacity_baselines'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13, 21, 99]
BATCH_MLP  = 512
EPOCHS_MLP = 60
VSTAR_REF  = 0.7343   # v17f* bootstrap point estimate

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 70)
print("  Equal-capacity baselines for v17f* (delta_layer novelty test)")
print("=" * 70)

# ── 1. Load ALL needed embeddings ─────────────────────────────────
print("\n[1] Loading embeddings (L15, L25, L30)...")
X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X_l15_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer15_t30_150M.npy').astype(np.float32)
X_l25_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer25_t30_150M.npy').astype(np.float32)
X_l30_te = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)
X_l15_te = np.load(f'{DATA_DIR}/esm2_layer_15_t30_150M.npy').astype(np.float32)
X_l25_te = np.load(f'{DATA_DIR}/esm2_layer_25_t30_150M.npy').astype(np.float32)
print(f"  Train: {X_l30_tr.shape}  Test: {X_l30_te.shape}")

# ── 2. Prepare second components (all MaxAbsScaler-scaled) ────────
print("\n[2] Preparing second components...")

# B0: delta = L30-L15 (v17f* reference)
delta_tr = (X_l30_tr - X_l15_tr).astype(np.float32)
delta_te = (X_l30_te - X_l15_te).astype(np.float32)
sc0 = MaxAbsScaler(); delta_tr_s = sc0.fit_transform(delta_tr).astype(np.float32); delta_te_s = sc0.transform(delta_te).astype(np.float32)

# B1: L15 raw (no subtraction), MaxAbsScaler
sc1 = MaxAbsScaler(); l15_tr_s = sc1.fit_transform(X_l15_tr).astype(np.float32); l15_te_s = sc1.transform(X_l15_te).astype(np.float32)

# B2: L25 (arbitrary nearby layer), MaxAbsScaler
sc2 = MaxAbsScaler(); l25_tr_s = sc2.fit_transform(X_l25_tr).astype(np.float32); l25_te_s = sc2.transform(X_l25_te).astype(np.float32)

# B3: random Gaussian (pure capacity baseline), fixed seed
rng_b = np.random.default_rng(777)
rand_tr = rng_b.standard_normal((X_l30_tr.shape[0], 640)).astype(np.float32)
rand_te = rng_b.standard_normal((X_l30_te.shape[0], 640)).astype(np.float32)
sc3 = MaxAbsScaler(); rand_tr_s = sc3.fit_transform(rand_tr).astype(np.float32); rand_te_s = sc3.transform(rand_te).astype(np.float32)

BASELINES = [
    ('B0_delta_L30-L15',  delta_tr_s, delta_te_s, 'concat[L30, L30-L15]  <- v17f* ref'),
    ('B1_L15_raw',        l15_tr_s,   l15_te_s,   'concat[L30, L15_raw]  <- no subtraction'),
    ('B2_L25',            l25_tr_s,   l25_te_s,   'concat[L30, L25]      <- arbitrary layer'),
    ('B3_random',         rand_tr_s,  rand_te_s,  'concat[L30, random]   <- pure capacity'),
]

# ── 3. IDs + GO labels ────────────────────────────────────────────
print("\n[3] Loading IDs and GO labels...")
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in te_genes_raw]

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

# ── 4. TF setup ───────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"\n  GPU: {gpus[0].name}")

def build_mlp(dim_a=640, dim_b=640, n_go=82):
    inp_a = layers.Input(shape=(dim_a,))
    inp_b = layers.Input(shape=(dim_b,))
    x     = layers.Concatenate()([inp_a, inp_b])
    x     = layers.Dense(256, activation='relu')(x)
    x     = layers.BatchNormalization()(x)
    x     = layers.Dropout(0.2)(x)
    x     = layers.Dense(128, activation='relu')(x)
    out   = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model([inp_a, inp_b], out)

focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
n_go     = len(mf_terms)

def macro_auprc(preds, idxs):
    aps = [average_precision_score(Y_te[:, i], preds[:, i])
           for i in idxs if Y_te[:, i].sum() >= 2]
    return float(np.mean(aps)) if aps else float('nan')

# ── 5. Run all baselines ──────────────────────────────────────────
results_all = {}
print(f"\n[4] Running {len(BASELINES)} baselines × 5 seeds each...")

for bname, X2_tr, X2_te, desc in BASELINES:
    print(f"\n  === {bname}: {desc} ===")
    t0 = time.time()
    all_preds = []
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm    = np.random.permutation(len(X_l30_tr))
        n_val   = int(len(X_l30_tr) * 0.1)
        val_idx = perm[:n_val]; tr_idx = perm[n_val:]
        mlp = build_mlp(n_go=n_go)
        mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
        mlp.fit(
            [X_l30_tr[tr_idx], X2_tr[tr_idx]], Y_tr[tr_idx],
            validation_data=([X_l30_tr[val_idx], X2_tr[val_idx]], Y_tr[val_idx]),
            epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0
        )
        preds_i = mlp.predict([X_l30_te, X2_te], batch_size=1024, verbose=0)
        all_preds.append(preds_i)
        aps = [average_precision_score(Y_te[:, i], preds_i[:, i])
               for i in valid_idx if Y_te[:, i].sum() >= 2]
        print(f"    seed={seed}  AUPRC={np.mean(aps):.4f}")

    preds = np.mean(all_preds, axis=0)
    auprc_all = macro_auprc(preds, valid_idx)
    auprc_l2  = macro_auprc(preds, l2_valid)
    delta_vs_vstar = auprc_all - VSTAR_REF
    results_all[bname] = {
        'description': desc,
        'all_mf': auprc_all,
        'l2_structural': auprc_l2,
        'delta_vs_vstar': delta_vs_vstar,
        'elapsed_s': round(time.time() - t0),
    }
    print(f"    Ensemble: All MF={auprc_all:.4f}  L2={auprc_l2:.4f}  Δ_vs_v17f*={delta_vs_vstar:+.4f}")

# ── 6. Summary table ──────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Equal-Capacity Baseline Summary")
print(f"{'='*70}")
print(f"  {'Model':<28} {'All MF':>7}  {'L2':>7}  {'Δ vs v17*':>10}  Interpretation")
print(f"  {'-'*67}")

THRESH_EQUIV = 0.005   # within 0.005 = "equivalent"

interp_map = {
    'B0_delta_L30-L15':  '(v17f* reference)',
    'B1_L15_raw':        'subtraction effect',
    'B2_L25':            'L15-specificity test',
    'B3_random':         'pure capacity test',
}

for bname, r in results_all.items():
    d = r['delta_vs_vstar']
    if abs(d) <= THRESH_EQUIV:
        sig = 'EQUIVALENT  <- capacity artifact'
    elif d > THRESH_EQUIV:
        sig = 'BETTER (unexpected)'
    else:
        sig = f'LOWER by {abs(d):.3f} <- delta_layer adds value'
    print(f"  {bname:<28} {r['all_mf']:>7.4f}  {r['l2_structural']:>7.4f}  {d:>+10.4f}  {sig}")

print(f"\n  v17f* (bootstrap ref):          0.7343  0.6366  baseline")
print(f"\n  Threshold for 'equivalent': ±{THRESH_EQUIV}")

# Overall verdict
b3 = results_all['B3_random']['delta_vs_vstar']
b2 = results_all['B2_L25']['delta_vs_vstar']
b1 = results_all['B1_L15_raw']['delta_vs_vstar']

if abs(b3) <= THRESH_EQUIV:
    verdict = "CAPACITY ARTIFACT: random noise ~ delta_layer. delta novelty REFUTED."
elif abs(b2) <= THRESH_EQUIV:
    verdict = "LAYER ARBITRARY: L25 ~ delta_layer. L15 selection is post-hoc. PARTIALLY REFUTED."
elif abs(b1) <= THRESH_EQUIV:
    verdict = "SUBTRACTION UNNECESSARY: raw L15 ~ delta_layer. L15 info matters, subtraction doesn't."
else:
    verdict = "DELTA DEFENDED: all baselines significantly lower. delta=L30-L15 is specifically valuable."

print(f"\n  Overall verdict: {verdict}")
print(f"{'='*70}")

with open(f'{OUT_DIR}/results.json', 'w') as fh:
    json.dump({
        'baselines': results_all,
        'vstar_ref': VSTAR_REF,
        'threshold_equiv': THRESH_EQUIV,
        'verdict': verdict,
    }, fh, indent=2)
print(f"\n[Saved] {OUT_DIR}/results.json")
