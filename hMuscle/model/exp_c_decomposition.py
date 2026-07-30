#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_c_decomposition.py
-----------------------
Three-component gain decomposition figure for Nature Methods.

Runs v17e_layer: concat[δ_layer(640), φ_L30(640)] without T_ψ.
This isolates the raw δ_layer signal contribution without triplet organization.

Components:
  (1) δ_layer signal alone    = v17e_layer − PRISM
  (2) T_ψ organization alone  = random-δ − PRISM   (from existing layer_breakdown.tsv)
  (3) Synergy (superadditive) = v17f − PRISM − component(1) − component(2)
      OR = v17f − v17e_layer − (random-δ − PRISM)

Final figure: stacked bar chart per H2 layer group.
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
OUT_DIR   = '../../reports/exp_c_decomposition'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13, 21, 99]
EPOCHS_MLP = 60
BATCH      = 512

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  Experiment C: δ_layer Gain Decomposition")
print("=" * 65)

# ── Load data ────────────────────────────────────────────────────
print("\n[1] Loading embeddings...")
X_l30_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X_l15_tr = np.load(f'{DATA_DIR}/esm2_train_human_layer15_t30_150M.npy').astype(np.float32)
X_l30_te = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)
X_l15_te = np.load(f'{DATA_DIR}/esm2_layer_15_t30_150M.npy').astype(np.float32)

delta_tr = (X_l30_tr - X_l15_tr).astype(np.float32)
delta_te = (X_l30_te - X_l15_te).astype(np.float32)
scaler   = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
delta_te_s = scaler.transform(delta_te).astype(np.float32)

# v17e_layer input: concat[δ_layer, φ_L30]
X_concat_tr = np.concatenate([delta_tr_s, X_l30_tr], axis=1).astype(np.float32)
X_concat_te = np.concatenate([delta_te_s, X_l30_te], axis=1).astype(np.float32)
print(f"  concat shape: train={X_concat_tr.shape}  test={X_concat_te.shape}")

# ── GO labels ────────────────────────────────────────────────────
print("\n[2] Loading GO labels...")
tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
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
tr_ids = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_ids)
go_genes_tr = defaultdict(set); go_genes_all = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        if p[7] != 'Function': continue
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
n_labels = len(mf_terms)
valid_idx = [i for i in range(n_labels) if valid_mask[i]]

# H2 layer classification
H2_LAYER = {}
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12: H2_LAYER[p[0]] = p[11]

layer_groups_order = ['L2_Structural', 'L2_Structural*', 'L4_CellState',
                      'L1_Generic_mid', 'L1_Generic_high']
layer_idxs = {
    k: [i for i, go in enumerate(mf_terms) if H2_LAYER.get(go) == k and valid_mask[i]]
    for k in layer_groups_order
}
all_valid = [i for i in range(n_labels) if valid_mask[i]]
layer_idxs['All MF'] = all_valid

# ── TensorFlow ───────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers as klayers, models as kmodels, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
focal = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

def build_mlp(input_dim, n_go):
    inp = klayers.Input(shape=(input_dim,))
    x   = klayers.Dense(256, activation='relu')(inp)
    x   = klayers.BatchNormalization()(x)
    x   = klayers.Dropout(0.3)(x)
    x   = klayers.Dense(128, activation='relu')(x)
    x   = klayers.Dropout(0.2)(x)
    out = klayers.Dense(n_go, activation='sigmoid')(x)
    return kmodels.Model(inp, out)

def eval_layer_groups(preds, label):
    results = {}
    for group, idxs in layer_idxs.items():
        if not idxs: continue
        aps = [average_precision_score(Y_te[:,i], preds[:,i])
               for i in idxs if Y_te[:,i].sum() >= 2]
        m = float(np.mean(aps)) if aps else float('nan')
        results[group] = m
    return results

# ── Run v17e_layer: concat[δ_layer, φ_L30] no T_ψ ───────────────
print(f"\n[3] Training v17e_layer (concat[δ_layer(640), φ_L30(640)] no T_ψ)...")
t0 = time.time()
all_preds = []
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm  = np.random.permutation(len(X_concat_tr))
    n_val = int(len(X_concat_tr) * 0.1)
    vi = perm[:n_val]; ti = perm[n_val:]
    mlp = build_mlp(X_concat_tr.shape[1], n_labels)
    mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal)
    mlp.fit(X_concat_tr[ti], Y_tr[ti],
            validation_data=(X_concat_tr[vi], Y_tr[vi]),
            epochs=EPOCHS_MLP, batch_size=BATCH,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0)
    all_preds.append(mlp.predict(X_concat_te, batch_size=1024, verbose=0))
    print(f"  seed {seed} done")

v17e_layer_preds = np.mean(all_preds, axis=0)
v17e_layer_results = eval_layer_groups(v17e_layer_preds, 'v17e_layer')
print(f"\n  v17e_layer results:  [{time.time()-t0:.0f}s]")
for k, v in v17e_layer_results.items():
    print(f"    {k:<25} {v:.4f}")

# ── Load existing reference numbers ─────────────────────────────
print("\n[4] Loading reference results (PRISM, random-δ, v17f)...")
ref = {}
with open('../../reports/v17f_layer_breakdown/layer_breakdown.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 3:
            ref[p[0]] = {'prism': float(p[2]), 'real_delta': float(p[3]),
                         'rand_delta': float(p[4])}

prism_results = {'All MF': 0.5962}
v17f_results  = {'All MF': 0.7173}
rand_results  = {'All MF': 0.6416}
for group in layer_groups_order:
    if group in ref:
        prism_results[group] = ref[group]['prism']
        v17f_results[group]  = ref[group]['real_delta']
        rand_results[group]  = ref[group]['rand_delta']

# All MF for random-δ: use rand from layer_breakdown weighted mean
all_mf_rand = np.mean([ref[g]['rand_delta'] for g in layer_groups_order if g in ref])

print("\n  Reference values:")
print(f"  {'Group':<25} {'PRISM':>7} {'v17e_L':>7} {'rand_δ':>7} {'v17f':>7}")
for group in ['All MF'] + layer_groups_order:
    p = prism_results.get(group, float('nan'))
    e = v17e_layer_results.get(group, float('nan'))
    r = rand_results.get(group, float('nan'))
    f = v17f_results.get(group, float('nan'))
    print(f"  {group:<25} {p:>7.4f} {e:>7.4f} {r:>7.4f} {f:>7.4f}")

# ── Save JSON ────────────────────────────────────────────────────
decomposition_data = {}
for group in ['All MF'] + layer_groups_order:
    p = prism_results.get(group, float('nan'))
    e = v17e_layer_results.get(group, float('nan'))
    r = rand_results.get(group, float('nan'))
    f = v17f_results.get(group, float('nan'))
    decomposition_data[group] = {
        'prism':     p,
        'v17e_layer': e,      # δ_layer signal alone (no T_ψ)
        'rand_delta': r,      # T_ψ capacity (random δ)
        'v17f':      f,       # full method
        'delta_layer_signal': e - p,
        'tpsi_capacity':      r - p,
        'v17f_gain':          f - p,
        'synergy':            (f - p) - (e - p) - (r - p),  # can be negative if subadditive
    }
json.dump(decomposition_data, open(f'{OUT_DIR}/decomposition_data.json', 'w'), indent=2)

# ── Generate decomposition figure ────────────────────────────────
print("\n[5] Generating decomposition figure...")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('δ_layer + T_ψ: Three-Component Gain Decomposition\n'
                 'Brain zero-shot, 82 MF GO terms', fontsize=13, fontweight='bold')

    groups_plot = [g for g in ['L2_Structural', 'L2_Structural*', 'L4_CellState',
                                'L1_Generic_mid', 'L1_Generic_high']
                   if g in decomposition_data]

    # Panel A: Absolute AUPRC by model
    ax = axes[0]
    x    = np.arange(len(groups_plot))
    w    = 0.2
    colors = {'PRISM': '#9E9E9E', 'v17e_layer': '#42A5F5',
              'rand_delta': '#FFA726', 'v17f': '#EF5350'}
    labels_map = {'PRISM': 'PRISM (baseline)', 'v17e_layer': 'v17e_layer (δ_layer, no T_ψ)',
                  'rand_delta': 'random-δ (T_ψ capacity)', 'v17f': 'v17f (δ_layer + T_ψ)'}
    for j, (model, col) in enumerate(colors.items()):
        vals = [decomposition_data[g][model if model != 'rand_delta' else 'rand_delta']
                for g in groups_plot]
        bars = ax.bar(x + (j-1.5)*w, vals, w, label=labels_map[model],
                      color=col, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    short_labels = {'L2_Structural': 'L2_Struct', 'L2_Structural*': 'L2_Struct*',
                    'L4_CellState': 'L4_Cell', 'L1_Generic_mid': 'L1_mid',
                    'L1_Generic_high': 'L1_high'}
    ax.set_xticklabels([short_labels.get(g, g) for g in groups_plot], rotation=25, ha='right')
    ax.set_ylabel('Macro AUPRC (brain zero-shot)')
    ax.set_title('(A) Absolute Performance per GO Difficulty Layer')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    # Panel B: Stacked gain bars
    ax = axes[1]
    comp_colors = {'δ_layer signal\n(v17e_layer − PRISM)': '#42A5F5',
                   'T_ψ capacity\n(random-δ − PRISM)': '#FFA726',
                   'Synergy\n(v17f − both)': '#EF5350'}

    for j, group in enumerate(groups_plot):
        d  = decomposition_data[group]
        p  = d['prism']
        c1 = d['delta_layer_signal']    # δ_layer signal
        c2 = d['tpsi_capacity']         # T_ψ capacity
        c3 = d['synergy']               # synergy/interaction

        bottoms = [0, 0, 0]
        # PRISM bar (gray, full height)
        ax.bar(j, p, 0.6, color='#9E9E9E', alpha=0.4, label='PRISM' if j==0 else '')
        # δ_layer gain
        ax.bar(j, c1, 0.6, bottom=p, color='#42A5F5', alpha=0.85,
               label='δ_layer signal' if j==0 else '')
        # T_ψ capacity (stacked)
        ax.bar(j, c2, 0.6, bottom=p + c1, color='#FFA726', alpha=0.85,
               label='T_ψ capacity' if j==0 else '')
        # Synergy
        if c3 > 0.003:
            ax.bar(j, c3, 0.6, bottom=p + c1 + c2, color='#EF5350', alpha=0.85,
                   label='Synergy' if j==0 else '')
        elif c3 < -0.003:
            ax.bar(j, abs(c3), 0.6, bottom=p + c1 + c2 - abs(c3),
                   color='#B71C1C', alpha=0.5, label='Overlap (−)' if j==0 else '')
        # v17f marker
        ax.plot([j-0.32, j+0.32], [d['v17f'], d['v17f']], 'k-', linewidth=1.5,
                label='v17f total' if j==0 else '')

    ax.set_xticks(range(len(groups_plot)))
    ax.set_xticklabels([short_labels.get(g, g) for g in groups_plot],
                        rotation=25, ha='right')
    ax.set_ylabel('AUPRC')
    ax.set_title('(B) Stacked Gain Decomposition\n(δ_layer signal + T_ψ capacity + synergy)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    fig.savefig(f'{OUT_DIR}/decomposition_figure.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(f'{OUT_DIR}/decomposition_figure.png', bbox_inches='tight', dpi=150)
    print(f"[Saved] {OUT_DIR}/decomposition_figure.pdf")
except Exception as e:
    print(f"  Figure error: {e}")
    import traceback; traceback.print_exc()

print(f"\n[Saved] {OUT_DIR}/decomposition_data.json")
print("\n" + "=" * 65)
print("  Experiment C: COMPLETE")
print("=" * 65)
