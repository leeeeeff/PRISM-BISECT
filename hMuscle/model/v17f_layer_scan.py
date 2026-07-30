#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_layer_scan.py
------------------
Full scan: concat[L30, L_k] for k in 1..29, identical 1280-dim MLP.
Dual metric per layer:
  1. Macro AUPRC (population-level, 81 valid MF terms)
  2. pos_bias = mean(within-gene AUROC) - 0.5
     > 0: model discriminates isoforms WITHIN same gene
     = 0: model is equivalent to gene-mean baseline

pos_bias > 0 with high AUPRC = genuine isoform-level resolution.
High AUPRC but pos_bias ~ 0 = gene-level classifier, not isoform discriminator.

Uses 3 seeds per layer (fast scan); top layers re-run with 5 seeds if needed.
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
OUT_DIR   = '../../reports/v17f_layer_scan'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS_SCAN = [42, 7, 13]     # 3 seeds for fast scan
BATCH_MLP  = 512
EPOCHS_MLP = 60
LAYER_B    = 30
SCAN_LAYERS = list(range(1, 30))   # L1 through L29

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 70)
print(f"  Layer scan: concat[L30, L_k] k=1..29  |  dual metric: AUPRC + pos_bias")
print("=" * 70)

# ── 1. Load L30 (fixed) ───────────────────────────────────────────
print("\n[1] Loading L30 embeddings (fixed)...")
X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_l30_te = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
print(f"  Train: {X_l30_tr.shape}  Test: {X_l30_te.shape}")

# ── 2. IDs ────────────────────────────────────────────────────────
print("\n[2] Loading gene IDs...")
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
tr_sym2idx   = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]
te_arr       = np.array(te_sym_list)

gene2idxs_te = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs_te[g].append(i)
multi_genes  = {g: np.array(idxs) for g, idxs in gene2idxs_te.items() if len(idxs) >= 2}
print(f"  Test genes with ≥2 isoforms: {len(multi_genes)} genes, "
      f"{sum(len(v) for v in multi_genes.values())} isoforms")

# ── 3. GO labels ──────────────────────────────────────────────────
print("\n[3] Loading GO labels...")
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

# ── 4. pos_bias computation ───────────────────────────────────────
def compute_pos_bias(preds, Y, valid_idx, multi_genes):
    """
    For each (GO term, gene with >=2 isoforms):
      - if gene has both positive and negative isoforms for this GO term
      - compute AUROC of model's predictions within that gene's isoforms
    pos_bias = mean(AUROC) - 0.5
    > 0 means model ranks positive isoforms above negative within same gene
    """
    within_aucs = []
    for go_i in valid_idx:
        y_go = Y[:, go_i]
        p_go = preds[:, go_i]
        for gene, idxs in multi_genes.items():
            y_sub = y_go[idxs]
            p_sub = p_go[idxs]
            if y_sub.sum() == 0 or y_sub.sum() == len(idxs):
                continue  # no discrimination possible (all pos or all neg)
            try:
                auc = roc_auc_score(y_sub, p_sub)
                within_aucs.append(auc)
            except Exception:
                pass
    if not within_aucs:
        return float('nan'), 0
    return float(np.mean(within_aucs)) - 0.5, len(within_aucs)

# ── 5. TF setup ───────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers as kl, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"\n  GPU: {gpus[0].name}")

def build_mlp(n_go=82):
    inp_a = kl.Input(shape=(640,))
    inp_b = kl.Input(shape=(640,))
    x     = kl.Concatenate()([inp_a, inp_b])
    x     = kl.Dense(256, activation='relu')(x)
    x     = kl.BatchNormalization()(x)
    x     = kl.Dropout(0.2)(x)
    x     = kl.Dense(128, activation='relu')(x)
    out   = kl.Dense(n_go, activation='sigmoid')(x)
    return models.Model([inp_a, inp_b], out)

focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
n_go     = len(mf_terms)

def run_ensemble(X2_tr, X2_te, seeds):
    all_preds = []
    for seed in seeds:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm    = np.random.permutation(len(X_l30_tr))
        n_val   = int(len(X_l30_tr) * 0.1)
        v_idx   = perm[:n_val]; t_idx = perm[n_val:]
        mlp = build_mlp(n_go=n_go)
        mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
        mlp.fit(
            [X_l30_tr[t_idx], X2_tr[t_idx]], Y_tr[t_idx],
            validation_data=([X_l30_tr[v_idx], X2_tr[v_idx]], Y_tr[v_idx]),
            epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0
        )
        all_preds.append(mlp.predict([X_l30_te, X2_te], batch_size=1024, verbose=0))
    return np.mean(all_preds, axis=0)

def macro_auprc(preds, idxs):
    aps = [average_precision_score(Y_te[:, i], preds[:, i])
           for i in idxs if Y_te[:, i].sum() >= 2]
    return float(np.mean(aps)) if aps else float('nan')

# ── 6. Scan ───────────────────────────────────────────────────────
print(f"\n[4] Scanning L1..L29 ({len(SCAN_LAYERS)} layers, {len(SEEDS_SCAN)} seeds each)...")
print(f"  {'Layer':<8} {'AUPRC_all':>10} {'L2':>8} {'pos_bias':>10} {'n_pairs':>8}  elapsed")
print(f"  {'-'*60}")

scan_results = {}
t_total = time.time()

for layer_k in SCAN_LAYERS:
    t0 = time.time()
    # Load Lk embeddings
    X_lk_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer{layer_k:02d}_t30_150M.npy').astype(np.float32)
    X_lk_te = np.load(f'{DATA_DIR}/esm2_layer_{layer_k:02d}_t30_150M.npy').astype(np.float32)

    # Scale (MaxAbsScaler on Lk)
    sc = MaxAbsScaler()
    X_lk_tr_s = sc.fit_transform(X_lk_tr).astype(np.float32)
    X_lk_te_s = sc.transform(X_lk_te).astype(np.float32)

    preds      = run_ensemble(X_lk_tr_s, X_lk_te_s, SEEDS_SCAN)
    auprc_all  = macro_auprc(preds, valid_idx)
    auprc_l2   = macro_auprc(preds, l2_valid)
    pb, n_pairs = compute_pos_bias(preds, Y_te, valid_idx, multi_genes)
    elapsed    = time.time() - t0

    scan_results[layer_k] = {
        'layer_k': layer_k,
        'auprc_all_mf': auprc_all,
        'auprc_l2': auprc_l2,
        'pos_bias': pb,
        'n_within_gene_pairs': n_pairs,
        'elapsed_s': round(elapsed),
    }
    print(f"  L{layer_k:<7} {auprc_all:>10.4f} {auprc_l2:>8.4f} {pb:>+10.4f} {n_pairs:>8}  {elapsed:.0f}s")

    # Save incrementally
    with open(f'{OUT_DIR}/scan_results.json', 'w') as fh:
        json.dump(scan_results, fh, indent=2)

# ── 7. Summary ────────────────────────────────────────────────────
best_auprc = max(scan_results.values(), key=lambda x: x['auprc_all_mf'])
best_pb    = max(scan_results.values(), key=lambda x: x['pos_bias'])

print(f"\n{'='*70}")
print(f"  Layer Scan Summary (concat[L30, L_k], {len(SEEDS_SCAN)} seeds)")
print(f"{'='*70}")
print(f"  Best AUPRC:    L{best_auprc['layer_k']}  All MF={best_auprc['auprc_all_mf']:.4f}  pos_bias={best_auprc['pos_bias']:+.4f}")
print(f"  Best pos_bias: L{best_pb['layer_k']}     All MF={best_pb['auprc_all_mf']:.4f}  pos_bias={best_pb['pos_bias']:+.4f}")
print(f"\n  Reference values (5 seeds):")
print(f"    v17f* = concat[L30, delta=L30-L15]: AUPRC=0.7343")
print(f"    B1    = concat[L30, L15_raw]:        AUPRC=0.7334")
print(f"    B2    = concat[L30, L25]:            AUPRC=0.7449")
print(f"\n  Interpretation:")
if best_auprc['pos_bias'] > 0.02:
    print(f"    Best-AUPRC model (L{best_auprc['layer_k']}) has genuine isoform discrimination (pos_bias={best_auprc['pos_bias']:+.4f})")
else:
    print(f"    Best-AUPRC model (L{best_auprc['layer_k']}) may be gene-level classifier (pos_bias={best_auprc['pos_bias']:+.4f} ~ 0)")
if best_pb['layer_k'] != best_auprc['layer_k']:
    print(f"    TRADEOFF: best pos_bias (L{best_pb['layer_k']}) ≠ best AUPRC (L{best_auprc['layer_k']})")
    print(f"    → isoform-discriminating layer ≠ population-AUPRC-maximizing layer")
else:
    print(f"    ALIGNED: same layer maximizes both AUPRC and pos_bias")
print(f"\n  Total elapsed: {time.time()-t_total:.0f}s")
print(f"{'='*70}")
print(f"\n[Saved] {OUT_DIR}/scan_results.json")
