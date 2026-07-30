# -*- coding: utf-8 -*-
"""
v_expanded_gomf.py
==================
PRISM with expanded GO term set: BP + MF + CC (from gene2go.gz)

핵심 목적:
  1. BP만이 아닌 MF / CC 영역에서 PRISM 성능 측정
  2. 카테고리별 ESM-2 기여 vs MLP 기여 분리
  3. InterPro MF 영역과 직접 비교 가능한 결과 생성
  4. Zero-shot 전이가 GO 카테고리별로 어떻게 다른지 전수 조사

설계:
  - GO term 선택: gene2go.gz에서 human(taxid=9606), n_positive >= MIN_POS
  - 훈련: v15d_bp_clean 동일 MLP (640→256→128→64→N_GO sigmoid, focal γ=2)
  - 평가: muscle held-out + brain zero-shot, 카테고리별 breakdown
  - ESM-2 LR baseline: 각 GO term에 대해 sklearn LR로 별도 평가
"""

import os, json, gzip, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("  Using GPU:0")
    except: pass

# ── 설정 ──────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v_expanded_gomf'
os.makedirs(OUT_DIR, exist_ok=True)

MIN_POS   = 100    # training set에서 최소 positive gene 수
N_SEEDS   = 5
CATEGORIES = {'Process': 'BP', 'Function': 'MF', 'Component': 'CC'}

print("="*70)
print("  PRISM Expanded GO: BP + MF + CC (gene2go)")
print(f"  MIN_POS = {MIN_POS}")
print("="*70)

# ── Step 1: gene symbol → Entrez ID 매핑 ────────────────────────────────────
def load_sym2id():
    sym2id = {}
    with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) > 2:
                sym2id[p[2]] = p[1]
                # synonyms
                if len(p) > 4 and p[4] != '-':
                    for syn in p[4].split('|'):
                        if syn not in sym2id:
                            sym2id[syn] = p[1]
    return sym2id

def clean_sym(raw):
    s = str(raw)
    for ch in ["b'", "'", '"', ' ']:
        s = s.replace(ch, '')
    return s

sym2id = load_sym2id()
print(f"  Gene symbol→ID map: {len(sym2id):,} entries")

# ── Step 2: training gene list 로드 및 ID 변환 ───────────────────────────────
tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_syms = [clean_sym(g) for g in tr_genes_raw]
tr_ids  = [sym2id.get(s, s) for s in tr_syms]
tr_id_set = set(tr_ids)
print(f"  Training genes: {len(tr_id_set):,}")

# ── Step 3: gene2go 파싱 → GO term별 positive gene set ──────────────────────
go_info    = {}   # go_id → {'name': str, 'cat': str ('BP'/'MF'/'CC')}
go_genes   = defaultdict(set)  # go_id → set of training gene IDs

with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606':
            continue
        gid, go_id, go_name, cat_raw = p[1], p[2], p[5], p[7]
        if cat_raw not in CATEGORIES:
            continue
        cat = CATEGORIES[cat_raw]
        go_info[go_id] = {'name': go_name, 'cat': cat}
        if gid in tr_id_set:
            go_genes[go_id].add(gid)

# ── Step 4: GO term 선택 (n_positive >= MIN_POS) ────────────────────────────
selected_terms = []
for go_id, gene_set in go_genes.items():
    if len(gene_set) >= MIN_POS:
        selected_terms.append(go_id)

selected_terms = sorted(selected_terms)
N_GO = len(selected_terms)

cat_counts = defaultdict(int)
for go_id in selected_terms:
    cat_counts[go_info[go_id]['cat']] += 1

print(f"\n  Selected GO terms: {N_GO} total")
for cat, n in sorted(cat_counts.items()):
    print(f"    {cat}: {n}")

# ── Step 5: ESM-2 임베딩 로드 ────────────────────────────────────────────────
print(f"\n  Loading ESM-2 embeddings...")
X_tr = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [clean_sym(x) for x in arr]

tr_sym_list = load_ids(f'{ID_DIR}/train_gene_list.npy')
te_sym_list = load_ids('my_gene_list_fixed.npy')
te_iso_list = load_ids('my_isoform_list_fixed.npy')

# symbol → training row index 매핑
tr_sym2idx = defaultdict(list)
for i, s in enumerate(tr_sym_list):
    tr_sym2idx[s].append(i)

# ── Step 6: 훈련 레이블 행렬 구성 ────────────────────────────────────────────
print(f"\n  Building label matrix ({len(X_tr)} × {N_GO})...")

def build_labels_train(go_id):
    pos_ids = go_genes[go_id]
    pos_syms = {s for s, gid in zip(tr_syms, tr_ids) if gid in pos_ids}
    y = np.zeros(len(X_tr), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []):
            y[idx] = 1.0
    return y

def build_labels_test(go_id):
    pos_ids = go_genes[go_id]
    pos_syms = {s for s, gid in zip(tr_syms, tr_ids) if gid in pos_ids}
    # test: gene-level match
    y = np.array([1.0 if s in pos_syms else 0.0 for s in te_sym_list],
                 dtype=np.float32)
    return y

Y_tr = np.stack([build_labels_train(go_id) for go_id in selected_terms], axis=1)
Y_te = np.stack([build_labels_test(go_id)  for go_id in selected_terms], axis=1)

print(f"  Y_tr shape: {Y_tr.shape}  positives/term: "
      f"min={Y_tr.sum(0).min():.0f} median={np.median(Y_tr.sum(0)):.0f} "
      f"max={Y_tr.sum(0).max():.0f}")

# ── Step 7: PRISM MLP 훈련 (multi-label focal loss) ─────────────────────────
def build_model(n_go):
    inp = layers.Input(shape=(640,))
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64,  activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model(inputs=inp, outputs=out)

print(f"\n  Training PRISM ({N_GO} GO terms, {N_SEEDS} seeds)...")
t0 = time.time()

seed_preds_te = []
for seed in range(N_SEEDS):
    tf.random.set_seed(seed * 137 + 42)
    np.random.seed(seed * 137 + 42)

    m = build_model(N_GO)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)
    )
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)
    ]
    idx = np.random.permutation(len(X_tr))
    m.fit(X_tr[idx], Y_tr[idx], epochs=80, batch_size=512,
          validation_split=0.1, callbacks=cb, verbose=0)
    seed_preds_te.append(m.predict(X_te, batch_size=1024, verbose=0))
    print(f"    seed {seed} done")

preds_te = np.mean(seed_preds_te, axis=0)  # (N_isoforms, N_GO)
print(f"  Training done: {time.time()-t0:.0f}s")

# 훈련 세트 예측 (held-out은 CV 대신 마지막 seed 모델로 근사)
m_last = build_model(N_GO)
m_last.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
idx = np.arange(len(X_tr))
n_val = int(len(X_tr) * 0.1)
val_idx = idx[-n_val:]
tr_idx  = idx[:-n_val]
m_last.fit(X_tr[tr_idx], Y_tr[tr_idx], epochs=80, batch_size=512,
           validation_data=(X_tr[val_idx], Y_tr[val_idx]),
           callbacks=cb, verbose=0)
preds_tr_val = m_last.predict(X_tr[val_idx], batch_size=1024, verbose=0)

# ── Step 8: ESM-2 LR baseline (카테고리별 기여 분리) ────────────────────────
print(f"\n  Computing ESM-2 LR baseline...")
lr_auprc_te  = {}
lr_auprc_val = {}

for i, go_id in enumerate(selected_terms):
    y_tr_i  = Y_tr[tr_idx, i]
    y_val_i = Y_tr[val_idx, i]
    y_te_i  = Y_te[:, i]

    if y_tr_i.sum() < 5 or y_val_i.sum() < 2:
        lr_auprc_te[go_id]  = 0.0
        lr_auprc_val[go_id] = 0.0
        continue
    try:
        lr = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
        lr.fit(X_tr[tr_idx], y_tr_i)
        lr_auprc_val[go_id] = average_precision_score(y_val_i,
                                  lr.predict_proba(X_tr[val_idx])[:, 1])
        lr_auprc_te[go_id]  = average_precision_score(y_te_i,
                                  lr.predict_proba(X_te)[:, 1]) \
                              if y_te_i.sum() > 0 else 0.0
    except Exception:
        lr_auprc_te[go_id]  = 0.0
        lr_auprc_val[go_id] = 0.0

# ── Step 9: 결과 집계 ─────────────────────────────────────────────────────────
print(f"\n  Computing AUPRC per GO term...")
results = []

for i, go_id in enumerate(selected_terms):
    y_val_i = Y_tr[val_idx, i]
    y_te_i  = Y_te[:, i]
    cat     = go_info[go_id]['cat']
    name    = go_info[go_id]['name']
    n_pos_tr = int(Y_tr[:, i].sum())

    prism_val = average_precision_score(y_val_i, preds_tr_val[:, i]) \
                if y_val_i.sum() > 1 else 0.0
    prism_te  = average_precision_score(y_te_i,  preds_te[:, i]) \
                if y_te_i.sum() > 0 else 0.0

    lr_val = lr_auprc_val.get(go_id, 0.0)
    lr_te  = lr_auprc_te.get(go_id, 0.0)

    results.append({
        'go_id':       go_id,
        'name':        name,
        'cat':         cat,
        'n_pos_tr':    n_pos_tr,
        'n_pos_te':    int(y_te_i.sum()),
        'prism_muscle': round(prism_val, 4),
        'prism_brain':  round(prism_te,  4),
        'lr_muscle':    round(lr_val,    4),
        'lr_brain':     round(lr_te,     4),
        'delta_muscle': round(prism_val - lr_val,   4),
        'delta_brain':  round(prism_te  - lr_te,    4),
    })

# ── Step 10: 카테고리별 요약 출력 ────────────────────────────────────────────
import pandas as pd
df = pd.DataFrame(results)
df.to_csv(f'{OUT_DIR}/expanded_go_per_term.tsv', sep='\t', index=False)

print(f"\n{'='*70}")
print(f"  Results: {len(df)} GO terms")
print(f"{'='*70}")

for cat in ['BP', 'MF', 'CC']:
    sub = df[df['cat'] == cat]
    if len(sub) == 0:
        continue
    # muscle held-out
    m_prism = sub['prism_muscle'].mean()
    m_lr    = sub['lr_muscle'].mean()
    m_delta = sub['delta_muscle'].mean()
    # brain zero-shot
    b_sub   = sub[sub['n_pos_te'] > 0]
    b_prism = b_sub['prism_brain'].mean() if len(b_sub) > 0 else float('nan')
    b_lr    = b_sub['lr_brain'].mean()    if len(b_sub) > 0 else float('nan')
    b_delta = b_sub['delta_brain'].mean() if len(b_sub) > 0 else float('nan')
    b_n50   = (b_sub['prism_brain'] > 0.5).sum()

    print(f"\n  [{cat}] n={len(sub)} terms")
    print(f"    Muscle  — PRISM: {m_prism:.4f}  LR: {m_lr:.4f}  Δ(MLP): {m_delta:+.4f}")
    print(f"    Brain   — PRISM: {b_prism:.4f}  LR: {b_lr:.4f}  Δ(MLP): {b_delta:+.4f}")
    print(f"    Brain n>0: {len(b_sub)} terms, AUPRC>0.5: {b_n50}")

    # top 5 / bottom 5 for brain AUPRC
    if len(b_sub) >= 5:
        top5 = b_sub.nlargest(5, 'prism_brain')[['go_id','name','prism_brain','lr_brain','delta_brain']]
        bot5 = b_sub.nsmallest(5, 'prism_brain')[['go_id','name','prism_brain','lr_brain','delta_brain']]
        print(f"    Top-5 brain:")
        for _, r in top5.iterrows():
            print(f"      {r['go_id']} {r['name'][:40]:<40} "
                  f"PRISM={r['prism_brain']:.3f} LR={r['lr_brain']:.3f} "
                  f"Δ={r['delta_brain']:+.3f}")
        print(f"    Bot-5 brain:")
        for _, r in bot5.iterrows():
            print(f"      {r['go_id']} {r['name'][:40]:<40} "
                  f"PRISM={r['prism_brain']:.3f} LR={r['lr_brain']:.3f} "
                  f"Δ={r['delta_brain']:+.3f}")

# ── Step 11: Transfer 성공/실패 분류 ─────────────────────────────────────────
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
    cat_dist = sub_z['cat'].value_counts().to_dict()
    print(f"  {zone:<22} n={len(sub_z):3d}  "
          f"BP={cat_dist.get('BP',0)} MF={cat_dist.get('MF',0)} CC={cat_dist.get('CC',0)}")

# ── Step 12: MF 특화 분석 (InterPro 비교 준비) ───────────────────────────────
df_mf = df[(df['cat'] == 'MF') & (df['n_pos_te'] > 0)].copy()
df_mf_sorted = df_mf.sort_values('prism_brain', ascending=False)
df_mf_sorted.to_csv(f'{OUT_DIR}/mf_terms_ranked.tsv', sep='\t', index=False)

print(f"\n  MF terms full ranking saved: {OUT_DIR}/mf_terms_ranked.tsv")
print(f"  MF: mean PRISM brain={df_mf['prism_brain'].mean():.4f}, "
      f"mean ESM-2 LR brain={df_mf['lr_brain'].mean():.4f}")
print(f"  MF: Δ>0.05 (MLP adds value) = "
      f"{(df_mf['delta_brain']>0.05).sum()}/{len(df_mf)} terms")
print(f"  MF: Δ<-0.05 (MLP hurts)     = "
      f"{(df_mf['delta_brain']<-0.05).sum()}/{len(df_mf)} terms")

# ── Step 13: 전체 결과 저장 ──────────────────────────────────────────────────
summary = {
    'config': {'min_pos': MIN_POS, 'n_go': N_GO, 'n_seeds': N_SEEDS,
               'cat_counts': dict(cat_counts)},
    'macro': {
        cat: {
            'n_terms': int(cat_counts.get(cat, 0)),
            'muscle_prism': float(df[df['cat']==cat]['prism_muscle'].mean()),
            'muscle_lr':    float(df[df['cat']==cat]['lr_muscle'].mean()),
            'brain_prism':  float(df[(df['cat']==cat)&(df['n_pos_te']>0)]['prism_brain'].mean()) if len(df[(df['cat']==cat)&(df['n_pos_te']>0)])>0 else 0.0,
            'brain_lr':     float(df[(df['cat']==cat)&(df['n_pos_te']>0)]['lr_brain'].mean()) if len(df[(df['cat']==cat)&(df['n_pos_te']>0)])>0 else 0.0,
            'brain_n_gt05': int((df[(df['cat']==cat)&(df['n_pos_te']>0)]['prism_brain']>0.5).sum()),
        } for cat in ['BP', 'MF', 'CC']
    },
    'timestamp': time.strftime('%Y-%m-%d %H:%M')
}
with open(f'{OUT_DIR}/expanded_go_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

np.save(f'{OUT_DIR}/preds_brain_expanded.npy', preds_te)

print(f"\n  Saved:")
print(f"    {OUT_DIR}/expanded_go_per_term.tsv")
print(f"    {OUT_DIR}/expanded_go_summary.json")
print(f"    {OUT_DIR}/mf_terms_ranked.tsv")
print(f"    {OUT_DIR}/preds_brain_expanded.npy")
print(f"\n{'='*70}")
print(f"  DONE")
print(f"{'='*70}")
