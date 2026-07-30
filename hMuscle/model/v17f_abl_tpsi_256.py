#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_abl_tpsi_256.py
--------------------
Ablation: T_psi with larger output dimension (640 -> 256) instead of v17f's 64.

Isolates bottleneck vs triplet effects by decoupling:
  - v17f   : T_psi(640->64)  + triplet  -> Stage2([64,  640]=704)   delta=9%
  - THIS   : T_psi(640->256) + triplet  -> Stage2([256, 640]=896)   delta=28.6%
  - v17f*  : no T_psi        + no tripl -> Stage2([640, 640]=1280)  delta=50%

If THIS > v17f* : triplet organization is genuinely useful; bottleneck was the problem.
If THIS ~ v17f* : bottleneck explains everything; triplet adds nothing.
If THIS < v17f  : larger T_psi dimension hurts (unlikely).
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
OUT_DIR   = '../../reports/v17f_abl_tpsi_256'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS        = [42, 7, 13, 21, 99]
MARGIN       = 0.3
BATCH_T      = 512
EPOCHS_T     = 50
BATCH_MLP    = 512
EPOCHS_MLP   = 60
EMBED_DIM_T  = 256          # KEY: 640->256 instead of v17f's 640->64
MAX_GO_TRIPS = 30000
GO_TRIPS_PER = 300
LAYER_A      = 15
LAYER_B      = 30

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print(f"  v17f Ablation: T_psi(640->{EMBED_DIM_T}) — bottleneck vs triplet")
print(f"  delta share in Stage2: {EMBED_DIM_T}/{EMBED_DIM_T+640} = {EMBED_DIM_T/(EMBED_DIM_T+640):.1%}")
print(f"  (v17f=9%, THIS={EMBED_DIM_T/(EMBED_DIM_T+640):.1%}, v17f*=50%)")
print("=" * 65)

# ── 1. Data ───────────────────────────────────────────────────────
print("\n[1] Loading embeddings...")
X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l15_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
X_l30_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l15_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
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
tr_sym2idx   = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]
print(f"  Train: {len(tr_genes)}  Test: {len(te_sym_list)}")

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

# ── 5. Triplets (same as v17f) ────────────────────────────────────
print("\n[5] Mining triplets...")
rng = np.random.default_rng(42)
trip_a, trip_p, trip_n = [], [], []
for k, go_id in enumerate(mf_terms):
    y_k      = Y_tr[:, k]
    pos_idxs = np.where(y_k == 1)[0]
    neg_idxs = np.where(y_k == 0)[0]
    if len(pos_idxs) < 5 or len(neg_idxs) < 10: continue
    if len(trip_a) >= MAX_GO_TRIPS: break
    n_anchor = min(GO_TRIPS_PER, len(pos_idxs))
    for a_idx in rng.choice(pos_idxs, n_anchor, replace=False):
        a_gene    = tr_genes[a_idx]
        cross_pos = pos_idxs[gene_arr_tr[pos_idxs] != a_gene]
        cross_neg = neg_idxs[gene_arr_tr[neg_idxs] != a_gene]
        if len(cross_pos) < 2 or len(cross_neg) < 2: continue
        trip_a.append(a_idx)
        trip_p.append(int(rng.choice(cross_pos)))
        trip_n.append(int(rng.choice(cross_neg)))

trip_a = np.array(trip_a, dtype=np.int32)
trip_p = np.array(trip_p, dtype=np.int32)
trip_n = np.array(trip_n, dtype=np.int32)
print(f"  {len(trip_a)} triplets")

# ── 6. TF + Stage 1: T_psi(640->256) ─────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"\n  GPU: {gpus[0].name}")

def build_T_psi(delta_dim=640, embed_dim=256):
    inp = layers.Input(shape=(delta_dim,))
    x   = layers.Dense(512, activation='relu')(inp)   # wider hidden for larger output
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(embed_dim, activation='relu')(x)
    out = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    return models.Model(inp, out, name=f'T_psi_{embed_dim}')

def build_mlp_stage2(t_dim=256, esm_dim=640, n_go=82):
    inp_t   = layers.Input(shape=(t_dim,))
    inp_esm = layers.Input(shape=(esm_dim,))
    x = layers.Concatenate()([inp_t, inp_esm])    # 896-dim
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model([inp_t, inp_esm], out, name='MLP_256')

def triplet_loss_fn(embs, a, p, n, margin=0.3):
    ea = tf.gather(embs, a); ep = tf.gather(embs, p); en = tf.gather(embs, n)
    d_pos = 1.0 - tf.reduce_sum(ea * ep, axis=1)
    d_neg = 1.0 - tf.reduce_sum(ea * en, axis=1)
    return tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))

print(f"\n[6] Stage 1: T_psi(640->{EMBED_DIM_T}) triplet training ({EPOCHS_T} epochs)...")
t0 = time.time()
tf.random.set_seed(42)
T_psi    = build_T_psi(embed_dim=EMBED_DIM_T)
opt_T    = optimizers.Adam(1e-3)
delta_tf = tf.constant(delta_tr_s, dtype=tf.float32)
n_triplets = len(trip_a)
n_batches  = max(1, n_triplets // BATCH_T)

for epoch in range(EPOCHS_T):
    perm = np.random.permutation(n_triplets)
    for b in range(n_batches):
        bi = perm[b*BATCH_T:(b+1)*BATCH_T]
        with tf.GradientTape() as tape:
            embs = T_psi(delta_tf, training=True)
            loss = triplet_loss_fn(embs, trip_a[bi], trip_p[bi], trip_n[bi], MARGIN)
        grads = tape.gradient(loss, T_psi.trainable_variables)
        opt_T.apply_gradients(zip(grads, T_psi.trainable_variables))
    if (epoch + 1) % 10 == 0:
        embs_np = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
        ea = embs_np[trip_a]; ep2 = embs_np[trip_p]; en2 = embs_np[trip_n]
        active = ((1-(ea*ep2).sum(1)) - (1-(ea*en2).sum(1)) + MARGIN > 0).mean()
        print(f"  Epoch {epoch+1:3d} | active={active:.2%}")

print(f"  T_psi done: {time.time()-t0:.0f}s")
T_tr = T_psi.predict(delta_tr_s, batch_size=1024, verbose=0)
T_te = T_psi.predict(delta_te_s, batch_size=1024, verbose=0)
print(f"  T_psi output: train={T_tr.shape}  test={T_te.shape}")

# ── 7. Stage 2: 5-seed ensemble ───────────────────────────────────
focal_fn  = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
n_go      = len(mf_terms)
all_preds = []

print(f"\n[7] Stage 2: 5-seed ensemble ({EPOCHS_MLP} epochs)...")
t0 = time.time()
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm    = np.random.permutation(len(T_tr))
    n_val   = int(len(T_tr) * 0.1)
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    mlp = build_mlp_stage2(t_dim=EMBED_DIM_T, n_go=n_go)
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
    mlp.fit(
        [T_tr[tr_idx], X_l30_tr[tr_idx]], Y_tr[tr_idx],
        validation_data=([T_tr[val_idx], X_l30_tr[val_idx]], Y_tr[val_idx]),
        epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    preds_i = mlp.predict([T_te, X_l30_te], batch_size=1024, verbose=0)
    all_preds.append(preds_i)
    aps = [average_precision_score(Y_te[:, i], preds_i[:, i])
           for i in valid_idx if Y_te[:, i].sum() >= 2]
    print(f"  seed={seed}  AUPRC={np.mean(aps):.4f}")

preds = np.mean(all_preds, axis=0)
print(f"  Ensemble done: {time.time()-t0:.0f}s")

# ── 8. Evaluate ───────────────────────────────────────────────────
def macro_auprc(preds, idxs):
    aps = [average_precision_score(Y_te[:, i], preds[:, i])
           for i in idxs if Y_te[:, i].sum() >= 2]
    return float(np.mean(aps)) if aps else float('nan')

auprc_all = macro_auprc(preds, valid_idx)
auprc_l2  = macro_auprc(preds, l2_valid)

V17F_ALL  = 0.7173; V17F_L2  = 0.6219
VSTAR_ALL = 0.7325; VSTAR_L2 = 0.6333
PRISM_ALL = 0.5962; PRISM_L2 = 0.3501

print(f"\n{'='*65}")
print(f"  T_psi(640->{EMBED_DIM_T}) Ablation — bottleneck vs triplet isolation")
print(f"{'='*65}")
print(f"  delta share in Stage2: {EMBED_DIM_T}/{EMBED_DIM_T+640}={EMBED_DIM_T/(EMBED_DIM_T+640):.1%}")
print(f"")
print(f"  Model                All MF    L2_Struct   delta_share")
print(f"  --------------------------------------------------------")
print(f"  PRISM (baseline)     {PRISM_ALL:.4f}   {PRISM_L2:.4f}      —")
print(f"  v17f (T_psi->64)     {V17F_ALL:.4f}   {V17F_L2:.4f}      9%")
print(f"  THIS (T_psi->256)    {auprc_all:.4f}   {auprc_l2:.4f}      {EMBED_DIM_T/(EMBED_DIM_T+640):.1%}")
print(f"  v17f* (no T_psi)     {VSTAR_ALL:.4f}   {VSTAR_L2:.4f}      50%")
print(f"")
if auprc_all > VSTAR_ALL:
    verdict = "Triplet HELPS: 256-dim T_psi > v17f* -> use T_psi with larger dim"
elif auprc_all > V17F_ALL:
    verdict = "Mixed: bottleneck matters, triplet marginal. v17f* still preferred."
elif auprc_all >= V17F_ALL - 0.005:
    verdict = "Neutral: 256-dim ~= v17f. Bottleneck size is not the key factor."
else:
    verdict = "Unexpected: larger T_psi dim hurts. Triplet or architecture issue."
print(f"  Verdict: {verdict}")
print(f"{'='*65}")

results = {
    'architecture': {
        'method': 'T_psi_256_v17f_data',
        'T_psi': f'640->512(ReLU)->BN->Drop(0.3)->{EMBED_DIM_T}(ReLU)->L2norm',
        'stage2_input': f'concat[T_psi({EMBED_DIM_T}), L30(640)] = {EMBED_DIM_T+640}',
        'delta_share_pct': round(EMBED_DIM_T/(EMBED_DIM_T+640)*100, 1),
        'triplet': 'cross-gene cosine margin=0.3',
        'train_isoforms': X_l30_tr.shape[0],
        'seeds': SEEDS,
    },
    'auprc_all_mf': auprc_all,
    'auprc_l2_structural': auprc_l2,
    'references': {
        'v17f_64dim':   {'all_mf': V17F_ALL,  'l2': V17F_L2,  'delta_share_pct': 9.1},
        'v17f_star':    {'all_mf': VSTAR_ALL,  'l2': VSTAR_L2, 'delta_share_pct': 50.0},
        'prism':        {'all_mf': PRISM_ALL,  'l2': PRISM_L2},
    },
    'delta_vs_v17f':    {'all_mf': auprc_all - V17F_ALL,  'l2': auprc_l2 - V17F_L2},
    'delta_vs_v17fstar':{'all_mf': auprc_all - VSTAR_ALL, 'l2': auprc_l2 - VSTAR_L2},
    'verdict': verdict,
}
with open(f'{OUT_DIR}/results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"\n[Saved] {OUT_DIR}/results.json")
