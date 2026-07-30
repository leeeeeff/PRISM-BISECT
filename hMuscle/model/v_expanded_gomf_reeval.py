# -*- coding: utf-8 -*-
"""
v_expanded_gomf_reeval.py
=========================
Bug fix: rebuild Y_te using full gene2go (not training-restricted).
Uses saved preds_brain_expanded.npy — no retraining needed.

Bug in original: go_genes only contained training gene IDs, so
build_labels_test produced all-zero labels for brain test genes.
Fix: parse gene2go for ALL human genes to build brain label matrix.
"""

import os, json, gzip, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v_expanded_gomf'

MIN_POS    = 100
CATEGORIES = {'Process': 'BP', 'Function': 'MF', 'Component': 'CC'}

# ── Step 1: sym2id ────────────────────────────────────────────────────────────
def clean_sym(raw):
    s = str(raw)
    for ch in ["b'", "'", '"', ' ']: s = s.replace(ch, '')
    return s

sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id:
                        sym2id[syn] = p[1]
print(f"  sym2id: {len(sym2id):,} entries")

# ── Step 2: training gene list ────────────────────────────────────────────────
tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_syms = [clean_sym(g) for g in tr_genes_raw]
tr_ids  = [sym2id.get(s, s) for s in tr_syms]
tr_id_set = set(tr_ids)
print(f"  Training genes: {len(tr_id_set):,}")

# ── Step 3: gene2go — training-restricted for term selection ─────────────────
go_info      = {}
go_genes_tr  = defaultdict(set)   # training-restricted (for selected_terms)
go_genes_all = defaultdict(set)   # ALL human genes (for Y_te)

with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        gid, go_id, go_name, cat_raw = p[1], p[2], p[5], p[7]
        if cat_raw not in CATEGORIES: continue
        cat = CATEGORIES[cat_raw]
        go_info[go_id] = {'name': go_name, 'cat': cat}
        go_genes_all[go_id].add(gid)        # all human genes
        if gid in tr_id_set:
            go_genes_tr[go_id].add(gid)     # training only

# ── Step 4: selected_terms (same as original run) ────────────────────────────
selected_terms = sorted([go_id for go_id, gs in go_genes_tr.items()
                         if len(gs) >= MIN_POS])
N_GO = len(selected_terms)
cat_counts = defaultdict(int)
for go_id in selected_terms:
    cat_counts[go_info[go_id]['cat']] += 1
print(f"  Selected GO terms: {N_GO} total")
for cat, n in sorted(cat_counts.items()):
    print(f"    {cat}: {n}")

# ── Step 5: Load embeddings ───────────────────────────────────────────────────
print("  Loading embeddings...")
X_tr = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

tr_sym_list = [clean_sym(x) for x in np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)]
te_gene_raw = [clean_sym(x) for x in np.load('my_gene_list_fixed.npy', allow_pickle=True)]

# ENSG2SYM: Ensembl ID (no version) → gene symbol
ID_DIR_PATH = f'../data/raw_data/data/id_lists'
ENSG2SYM = {}
with open(f'{ID_DIR_PATH}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]   # p[0]=ENSG_base, p[4]=symbol

# brain test: ENSG(version) → ENSG(base) → symbol
te_sym_list = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_gene_raw]
n_mapped = sum(1 for g in te_gene_raw if g.split('.')[0] in ENSG2SYM)
print(f"  Brain gene mapping: {n_mapped}/{len(te_gene_raw)} ENSG → symbol")

tr_sym2idx = defaultdict(list)
for i, s in enumerate(tr_sym_list):
    tr_sym2idx[s].append(i)

# ── Step 6: Build label matrices ─────────────────────────────────────────────
print("  Building label matrices...")

def build_labels_train(go_id):
    pos_ids  = go_genes_tr[go_id]
    pos_syms = {s for s, gid in zip(tr_syms, tr_ids) if gid in pos_ids}
    y = np.zeros(len(X_tr), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []):
            y[idx] = 1.0
    return y

def build_labels_test_fixed(go_id):
    # FIX: ENSG → symbol → Entrez ID → gene2go_all
    pos_ids = go_genes_all[go_id]
    y = np.array([1.0 if sym2id.get(s, '__NONE__') in pos_ids else 0.0
                  for s in te_sym_list], dtype=np.float32)
    return y

Y_tr = np.stack([build_labels_train(go_id)       for go_id in selected_terms], axis=1)
Y_te = np.stack([build_labels_test_fixed(go_id)  for go_id in selected_terms], axis=1)

print(f"  Y_tr: {Y_tr.shape}  pos/term: min={Y_tr.sum(0).min():.0f} "
      f"median={np.median(Y_tr.sum(0)):.0f} max={Y_tr.sum(0).max():.0f}")
print(f"  Y_te: {Y_te.shape}  pos/term: min={Y_te.sum(0).min():.0f} "
      f"median={np.median(Y_te.sum(0)):.0f} max={Y_te.sum(0).max():.0f}")
n_nonzero_te = (Y_te.sum(0) > 0).sum()
print(f"  Y_te: terms with ≥1 positive: {n_nonzero_te}/{N_GO}")

# ── Step 7: Load saved PRISM predictions ─────────────────────────────────────
print("  Loading saved PRISM predictions...")
preds_te = np.load(f'{OUT_DIR}/preds_brain_expanded.npy').astype(np.float32)
print(f"  preds_te shape: {preds_te.shape}")

# Need val predictions for muscle — rerun last seed quickly
np.random.seed(42)
idx = np.random.permutation(len(X_tr))
val_frac = 0.1
n_val = int(len(X_tr) * val_frac)
val_idx = idx[:n_val]
tr_idx  = idx[n_val:]

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except: pass

def build_model(n_go):
    inp = layers.Input(shape=(640,))
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model(inputs=inp, outputs=out)

print("  Training one seed for muscle val predictions...")
tf.random.set_seed(42)
m_last = build_model(N_GO)
m_last.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
cb = [callbacks.EarlyStopping(monitor='val_loss', patience=10,
                              restore_best_weights=True, verbose=0),
      callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)]
m_last.fit(X_tr[tr_idx], Y_tr[tr_idx], epochs=80, batch_size=512,
           validation_data=(X_tr[val_idx], Y_tr[val_idx]),
           callbacks=cb, verbose=0)
preds_tr_val = m_last.predict(X_tr[val_idx], batch_size=1024, verbose=0)
print("  Muscle val predictions done.")

# ── Step 8: ESM-2 LR baseline ────────────────────────────────────────────────
print("  Computing ESM-2 LR baseline...")
lr_auprc_te  = {}
lr_auprc_val = {}

for i, go_id in enumerate(selected_terms):
    y_tr_i  = Y_tr[tr_idx, i]
    y_val_i = Y_tr[val_idx, i]
    y_te_i  = Y_te[:, i]
    if y_tr_i.sum() < 5 or y_val_i.sum() < 2:
        lr_auprc_te[go_id] = lr_auprc_val[go_id] = 0.0
        continue
    try:
        lr = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
        lr.fit(X_tr[tr_idx], y_tr_i)
        lr_auprc_val[go_id] = average_precision_score(
            y_val_i, lr.predict_proba(X_tr[val_idx])[:, 1])
        lr_auprc_te[go_id] = average_precision_score(
            y_te_i, lr.predict_proba(X_te)[:, 1]) if y_te_i.sum() > 0 else 0.0
    except Exception:
        lr_auprc_te[go_id] = lr_auprc_val[go_id] = 0.0

# ── Step 9: Collect results ───────────────────────────────────────────────────
print("  Computing AUPRC per GO term...")
results = []
for i, go_id in enumerate(selected_terms):
    y_val_i = Y_tr[val_idx, i]
    y_te_i  = Y_te[:, i]
    cat  = go_info[go_id]['cat']
    name = go_info[go_id]['name']
    n_pos_tr = int(Y_tr[:, i].sum())

    prism_val = average_precision_score(y_val_i, preds_tr_val[:, i]) \
                if y_val_i.sum() > 1 else 0.0
    prism_te  = average_precision_score(y_te_i, preds_te[:, i]) \
                if y_te_i.sum() > 0 else 0.0
    lr_val = lr_auprc_val.get(go_id, 0.0)
    lr_te  = lr_auprc_te.get(go_id, 0.0)

    results.append({
        'go_id':        go_id,
        'name':         name,
        'cat':          cat,
        'n_pos_tr':     n_pos_tr,
        'n_pos_te':     int(y_te_i.sum()),
        'prism_muscle': round(prism_val, 4),
        'prism_brain':  round(prism_te,  4),
        'lr_muscle':    round(lr_val,    4),
        'lr_brain':     round(lr_te,     4),
        'delta_muscle': round(prism_val - lr_val, 4),
        'delta_brain':  round(prism_te  - lr_te,  4),
    })

import pandas as pd
df = pd.DataFrame(results)
df.to_csv(f'{OUT_DIR}/expanded_go_per_term.tsv', sep='\t', index=False)

# ── Step 10: Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Results: {len(df)} GO terms")
print(f"{'='*70}")

for cat in ['BP', 'MF', 'CC']:
    sub   = df[df['cat'] == cat]
    b_sub = sub[sub['n_pos_te'] > 0]
    if len(sub) == 0: continue

    m_prism = sub['prism_muscle'].mean()
    m_lr    = sub['lr_muscle'].mean()
    b_prism = b_sub['prism_brain'].mean() if len(b_sub) > 0 else float('nan')
    b_lr    = b_sub['lr_brain'].mean()    if len(b_sub) > 0 else float('nan')
    b_delta = b_sub['delta_brain'].mean() if len(b_sub) > 0 else float('nan')
    b_n50   = (b_sub['prism_brain'] > 0.5).sum()

    print(f"\n  [{cat}] n={len(sub)} terms (brain n>0: {len(b_sub)})")
    print(f"    Muscle — PRISM: {m_prism:.4f}  LR: {m_lr:.4f}  Δ: {m_prism-m_lr:+.4f}")
    print(f"    Brain  — PRISM: {b_prism:.4f}  LR: {b_lr:.4f}  Δ: {b_delta:+.4f}")
    print(f"    Brain AUPRC>0.5: {b_n50}/{len(b_sub)} terms")

    if len(b_sub) >= 5:
        top5 = b_sub.nlargest(5, 'prism_brain')[['go_id','name','prism_brain','lr_brain','delta_brain']]
        print(f"    Top-5 brain:")
        for _, r in top5.iterrows():
            print(f"      {r['go_id']} {r['name'][:40]:<40}  "
                  f"PRISM={r['prism_brain']:.3f} LR={r['lr_brain']:.3f} Δ={r['delta_brain']:+.3f}")

# ── Step 11: Transfer Zone ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Transfer Zone Classification (brain zero-shot)")
print(f"{'='*70}")
df_brain = df[df['n_pos_te'] > 0].copy()
conditions = {
    'Transfer_success':  (df_brain['prism_brain'] > 0.5)  & (df_brain['delta_brain'] > 0.05),
    'ESM2_sufficient':   (df_brain['prism_brain'] > 0.4)  & (df_brain['delta_brain'].abs() < 0.05),
    'Transfer_failure':  (df_brain['prism_brain'] < 0.4),
    'MLP_hurts':         (df_brain['delta_brain'] < -0.05),
}
for zone, mask in conditions.items():
    sub_z = df_brain[mask]
    cd = sub_z['cat'].value_counts().to_dict()
    print(f"  {zone:<22} n={len(sub_z):3d}  "
          f"BP={cd.get('BP',0)} MF={cd.get('MF',0)} CC={cd.get('CC',0)}")

# ── Step 12: MF specific ──────────────────────────────────────────────────────
df_mf = df[(df['cat'] == 'MF') & (df['n_pos_te'] > 0)].copy()
df_mf.sort_values('prism_brain', ascending=False).to_csv(
    f'{OUT_DIR}/mf_terms_ranked.tsv', sep='\t', index=False)
print(f"\n  MF (n>0 in brain): {len(df_mf)}/{(df['cat']=='MF').sum()} terms")
if len(df_mf) > 0:
    print(f"  MF brain — PRISM: {df_mf['prism_brain'].mean():.4f}  "
          f"LR: {df_mf['lr_brain'].mean():.4f}  Δ: {df_mf['delta_brain'].mean():+.4f}")
    print(f"  MF Δ>0.05 (MLP adds): {(df_mf['delta_brain']>0.05).sum()}/{len(df_mf)}")
    print(f"  MF Δ<-0.05 (MLP hurts): {(df_mf['delta_brain']<-0.05).sum()}/{len(df_mf)}")

# ── Step 13: Save summary JSON ────────────────────────────────────────────────
summary = {
    'config': {'min_pos': MIN_POS, 'n_go': N_GO, 'n_seeds': 5,
               'cat_counts': dict(cat_counts), 'bug_fix': 'Y_te rebuilt with full gene2go'},
    'macro': {
        cat: {
            'n_terms':      int(cat_counts.get(cat, 0)),
            'n_brain_pos':  int((df[(df['cat']==cat)]['n_pos_te'] > 0).sum()),
            'muscle_prism': float(df[df['cat']==cat]['prism_muscle'].mean()),
            'muscle_lr':    float(df[df['cat']==cat]['lr_muscle'].mean()),
            'brain_prism':  float(df[(df['cat']==cat)&(df['n_pos_te']>0)]['prism_brain'].mean())
                            if (df[(df['cat']==cat)&(df['n_pos_te']>0)]).shape[0] > 0 else 0.0,
            'brain_lr':     float(df[(df['cat']==cat)&(df['n_pos_te']>0)]['lr_brain'].mean())
                            if (df[(df['cat']==cat)&(df['n_pos_te']>0)]).shape[0] > 0 else 0.0,
            'brain_n_gt05': int((df[(df['cat']==cat)&(df['n_pos_te']>0)]['prism_brain']>0.5).sum()),
        } for cat in ['BP', 'MF', 'CC']
    },
    'timestamp': time.strftime('%Y-%m-%d %H:%M')
}
with open(f'{OUT_DIR}/expanded_go_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved: {OUT_DIR}/expanded_go_per_term.tsv")
print(f"         {OUT_DIR}/expanded_go_summary.json")
print(f"         {OUT_DIR}/mf_terms_ranked.tsv")
print(f"\n{'='*70}")
print(f"  REEVAL DONE")
print(f"{'='*70}")
