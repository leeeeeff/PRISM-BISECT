# -*- coding: utf-8 -*-
"""
v15d_splice.py
==============
v15d + splice_delta: Gated Residual Fusion

설계 근거:
  - ESM-2가 within-gene isoform 구분에 실패하는 2,078개 유전자(DIFF_SPLICE_SAME_PROT)에서
    splice_delta가 유일한 구조적 판별자 역할.
  - Gated residual로 canonical isoforms(splice_delta=0)에서 v15d와 동일한 동작 보장.
  - 경량 splice branch (Dense 32): avg 3.2 active dims에 맞는 적정 용량.

아키텍처:
  Stream A (v15d-identical, 640→128):
    Dense(256, ReLU) → BN → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → h_esm(128)

  Stream B (new, 150→32):
    Dense(32, ReLU) → Dropout(0.1) → h_splice(32)

  Gated Fusion (→ 128):
    gate_logits = Dense(128)( Concat([h_esm, h_splice]) )
    gate        = sigmoid(gate_logits)
    h_splice_up = Dense(128)(h_splice)          # 32 → 128 projection
    h_fused     = h_esm + gate * h_splice_up
    # splice_delta=0 → h_splice≈0 → h_splice_up≈0 → h_fused≈h_esm (v15d behavior)

  Output head (identical to v15d):
    Dense(64, ReLU) → Dense(1, sigmoid)

Loss: BinaryFocalCrossentropy(gamma=2.0)   [R1.1 — 유지]
Seeds: 5 (ensemble)

평가:
  - Macro AUPRC: ALL / DIFF_SPLICE / Normal 서브셋
  - pos_bias: DIFF_SPLICE 서브셋 (개선 목표)
  - gate activation: DIFF_SPLICE vs Normal 비교

실행:
  cd hMuscle/model
  conda activate isoform_env
  nohup python v15d_splice.py > ../../logs_isoform/v15d_splice_$(date +%Y%m%d_%H%M).log 2>&1 &
"""

import os, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("  Using GPU:0")
    except: pass

# ── 경로 ────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
SD_DIR    = '../results_isoform/features/splicing'
OUT_DIR   = '../../reports/v15d_splice'
os.makedirs(OUT_DIR, exist_ok=True)

ANNOT_FILE = f'{ANNOT_DIR}/human_annotations_unified_bp.txt'
SD_PATH    = f'{SD_DIR}/splicing_delta_v2.npy'
TRAIN_SD_PATH = f'{SD_DIR}/train_splicing_delta.npy'

N_SEEDS      = 5
BASELINE_AUPRC = 0.7022   # v15d reference

GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0045214': 'Sarcomere organization',
    'GO:0006941': 'Muscle contraction',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0030048': 'Actin-based movement',
    'GO:0006096': 'Glycolysis',
    'GO:0007268': 'Synaptic transmission',
    'GO:0007018': 'MT-based movement',
    'GO:0031175': 'Neuron proj development',
    'GO:0030182': 'Neuron diff',
    'GO:0000226': 'MT cytoskeleton org',
}
GO_KEYS  = list(GO_TERMS.keys())
GO_NAMES = list(GO_TERMS.values())
N_GO     = len(GO_KEYS)


# ── ID 로더 ──────────────────────────────────────────────────────────────────
def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


print("=" * 70)
print("  v15d_splice — Gated Residual Fusion (ESM-2 + splice_delta)")
print(f"  Baseline: v15d macro AUPRC = {BASELINE_AUPRC}")
print("=" * 70)

# ── 1. 데이터 로드 ────────────────────────────────────────────────────────────
print("\n[1] Loading features...")
te_iso  = load_ids('my_isoform_list_fixed.npy')
te_gene = load_ids('my_gene_list_fixed.npy')
tr_gene = load_ids(f'{ID_DIR}/train_gene_list.npy')

X_tr_esm = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te_esm = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

X_te_sd = np.load(SD_PATH).astype(np.float32)
if os.path.exists(TRAIN_SD_PATH):
    X_tr_sd = np.load(TRAIN_SD_PATH).astype(np.float32)
    print(f"  Train splice_delta loaded: {X_tr_sd.shape}")
else:
    # Train splice_delta 없으면 zero로 채움 (ESM-2 stream만 학습됨)
    X_tr_sd = np.zeros((X_tr_esm.shape[0], X_te_sd.shape[1]), dtype=np.float32)
    print(f"  WARN: train splice_delta not found → zeros ({X_tr_sd.shape})")

ESM_DIM = X_tr_esm.shape[1]   # 640
SD_DIM  = X_te_sd.shape[1]    # 150

print(f"  Train ESM-2: {X_tr_esm.shape} | splice_delta: {X_tr_sd.shape}")
print(f"  Test  ESM-2: {X_te_esm.shape} | splice_delta: {X_te_sd.shape}")

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
tr_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_gene]
te_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_gene]

# ── 2. Zero-ESM2 / Normal 서브셋 구축 ────────────────────────────────────────
print("\n[2] Building zero-ESM2 subsets...")
g2idx_te = defaultdict(list)
for i, g in enumerate(te_gene):
    g2idx_te[g].append(i)

multi_genes = {g: idxs for g, idxs in g2idx_te.items() if len(idxs) >= 2}

diff_splice_genes, normal_genes = [], []
for g, idxs in multi_genes.items():
    embs   = X_te_esm[idxs]
    deltas = X_te_sd[idxs]
    sims   = cosine_similarity(embs)
    np.fill_diagonal(sims, 0)
    max_dist = 1 - sims.max()

    iso_unique = len(set(te_iso[i] for i in idxs))
    if max_dist < 0.001 and iso_unique >= len(idxs) * 0.5:
        delta_diff = not np.allclose(deltas, deltas[0])
        if delta_diff:
            diff_splice_genes.append(g)
        # else: skip IDENTICAL_ALL / PURE_DUPLICATE
    else:
        normal_genes.append(g)

def genes_to_idx(gl):
    out = []
    for g in gl:
        out.extend(g2idx_te.get(g, []))
    return sorted(set(out))

ds_idx  = genes_to_idx(diff_splice_genes)
nm_idx  = genes_to_idx(normal_genes)
all_idx = list(range(len(te_iso)))

print(f"  DIFF_SPLICE genes: {len(diff_splice_genes)} ({len(ds_idx)} isoforms)")
print(f"  Normal genes:      {len(normal_genes)} ({len(nm_idx)} isoforms)")

# ── 3. 모델 정의 ─────────────────────────────────────────────────────────────
def build_model(esm_dim, sd_dim):
    """
    Gated Residual Fusion model.

    Stream A: ESM-2(640) → Dense(256)→BN→Drop(0.3)→Dense(128)→Drop(0.2) → h_esm(128)
    Stream B: splice_delta(150) → Dense(32)→Drop(0.1) → h_splice(32)
    Gate:     sigmoid( Dense(128)([h_esm, h_splice]) )
    Fused:    h_esm + gate * Dense(128)(h_splice)
    Head:     Dense(64,relu) → Dense(1,sigmoid)
    """
    inp_esm = layers.Input(shape=(esm_dim,), name='esm2')
    inp_sd  = layers.Input(shape=(sd_dim,),  name='splice_delta')

    # Stream A — v15d identical
    h = layers.Dense(256, activation='relu')(inp_esm)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(128, activation='relu')(h)
    h_esm = layers.Dropout(0.2)(h)                          # (128,)

    # Stream B — lightweight splice encoder
    # use_bias=False: splice_delta=0 → h_splice=0 → h_splice_up=0 → h_fused=h_esm exactly
    s = layers.Dense(32, activation='relu', use_bias=False)(inp_sd)
    h_splice = layers.Dropout(0.1)(s)                       # (32,)

    # Gated fusion — concat only for gate computation, not as output
    concat_for_gate = layers.Concatenate()([h_esm, h_splice])  # (160,)
    gate = layers.Dense(128, activation='sigmoid',
                        name='gate')(concat_for_gate)           # (128,)

    h_splice_up = layers.Dense(128, use_bias=False,
                                name='splice_proj')(h_splice)   # (32→128)
    h_fused = layers.Add()([h_esm,
                             layers.Multiply()([gate, h_splice_up])])  # (128,)

    # Output head
    h_out = layers.Dense(64, activation='relu')(h_fused)
    out   = layers.Dense(1,  activation='sigmoid')(h_out)

    return models.Model([inp_esm, inp_sd], out)


# ── 4. 라벨 로드 ─────────────────────────────────────────────────────────────
def load_labels(go_term):
    pos = set()
    with open(ANNOT_FILE) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) > 1 and go_term in p[1:]:
                pos.add(p[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te


# ── 5. pos_bias 계산 (evaluation.py 로직 재현) ───────────────────────────────
def compute_pos_bias(preds, labels, gene_list_te, subset_idx=None):
    """
    pos_bias_score = mean(within_gene_score_std, positive genes) / global_score_std
    subset_idx: None → all, else only these indices
    """
    if subset_idx is not None:
        preds  = preds[subset_idx]
        labels = labels[subset_idx]
        genes  = [gene_list_te[i] for i in subset_idx]
    else:
        genes = gene_list_te

    eps = 1e-10
    global_std = preds.std() + eps
    df_tmp = pd.DataFrame({'gene': genes, 'score': preds, 'label': labels})
    multi  = df_tmp.groupby('gene').filter(lambda g: len(g) >= 2)
    pos_g  = df_tmp[df_tmp['label'] == 1]['gene'].unique()
    pos_multi = multi[multi['gene'].isin(pos_g)]
    if len(pos_multi) == 0:
        return np.nan
    within_stds = pos_multi.groupby('gene')['score'].std().dropna()
    return float(within_stds.mean() / global_std)


# ── 6. 훈련 + 평가 ────────────────────────────────────────────────────────────
def train_and_eval(y_tr, y_te, seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    m = build_model(ESM_DIM, SD_DIM)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    )
    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                monitor='val_loss'),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor='val_loss'),
    ]
    m.fit(
        [X_tr_esm, X_tr_sd], y_tr,
        epochs=100, batch_size=256,
        validation_split=0.1, callbacks=cb, verbose=0,
    )
    preds = m.predict([X_te_esm, X_te_sd], batch_size=1024, verbose=0).flatten()

    # gate activation (평균) — DIFF_SPLICE vs Normal
    gate_layer  = m.get_layer('gate')
    gate_model  = models.Model(m.inputs, gate_layer.output)
    gate_vals   = gate_model.predict([X_te_esm, X_te_sd],
                                      batch_size=1024, verbose=0)  # (N, 128)
    gate_mean   = gate_vals.mean(axis=1)   # (N,) scalar per isoform
    gate_ds     = gate_mean[ds_idx].mean() if ds_idx else np.nan
    gate_nm     = gate_mean[nm_idx].mean() if nm_idx else np.nan

    auprc_all = average_precision_score(y_te, preds) if y_te.sum() > 0 else 0.0
    return preds, auprc_all, gate_ds, gate_nm


# ── 7. 전체 GO term 루프 ──────────────────────────────────────────────────────
print("\n[3] Training v15d_splice (5 seeds × 18 GO terms)...")
score_matrix = np.zeros((len(te_iso), N_GO), dtype=np.float32)
auprc_rows   = []
t0 = time.time()

# Baseline DIFF_SPLICE / Normal 참조 (zero_esm2_baseline.py 결과)
BASELINE_DS  = 0.8078
BASELINE_NM  = 0.6481

for j, (go, name) in enumerate(GO_TERMS.items()):
    y_tr, y_te = load_labels(go)
    n_pos_tr   = int(y_tr.sum())
    n_pos_te   = int(y_te.sum())
    print(f"\n  [{j+1:2d}/18] {go} ({name}) | tr_pos={n_pos_tr}, te_pos={n_pos_te}", flush=True)

    if n_pos_tr < 5:
        print(f"         SKIP (< 5 train positives)")
        continue

    seed_preds   = []
    seed_auprcs  = []
    seed_gate_ds = []
    seed_gate_nm = []

    for seed in range(N_SEEDS):
        preds, auprc, g_ds, g_nm = train_and_eval(y_tr, y_te, seed)
        seed_preds.append(preds)
        seed_auprcs.append(auprc)
        seed_gate_ds.append(g_ds)
        seed_gate_nm.append(g_nm)
        print(f"         seed={seed} AUPRC={auprc:.4f} "
              f"gate_DS={g_ds:.3f} gate_NM={g_nm:.3f}", flush=True)

    ens_preds   = np.mean(seed_preds, axis=0)
    ens_auprc   = average_precision_score(y_te, ens_preds) if n_pos_te > 0 else 0.0
    score_matrix[:, j] = ens_preds

    # AUPRC 서브셋
    y_ds = y_te[ds_idx]; p_ds = ens_preds[ds_idx]
    y_nm = y_te[nm_idx]; p_nm = ens_preds[nm_idx]
    auprc_ds = average_precision_score(y_ds, p_ds) if y_ds.sum() > 0 else None
    auprc_nm = average_precision_score(y_nm, p_nm) if y_nm.sum() > 0 else None

    # pos_bias 서브셋
    pb_ds = compute_pos_bias(ens_preds, y_te, te_gene, subset_idx=ds_idx)
    pb_nm = compute_pos_bias(ens_preds, y_te, te_gene, subset_idx=nm_idx)

    avg_gate_ds = float(np.nanmean(seed_gate_ds))
    avg_gate_nm = float(np.nanmean(seed_gate_nm))

    ds_str = f"{auprc_ds:.4f}" if auprc_ds is not None else "  N/A "
    nm_str = f"{auprc_nm:.4f}" if auprc_nm is not None else "  N/A "
    print(f"         Ensemble AUPRC={ens_auprc:.4f} "
          f"| DS={ds_str:>7} NM={nm_str:>7} "
          f"| pb_DS={pb_ds:.4f} pb_NM={pb_nm:.4f} "
          f"| gate DS={avg_gate_ds:.3f} NM={avg_gate_nm:.3f}")

    auprc_rows.append({
        'go': go, 'name': name,
        'auprc_all':  ens_auprc,
        'auprc_ds':   auprc_ds,
        'auprc_nm':   auprc_nm,
        'pos_bias_ds': pb_ds,
        'pos_bias_nm': pb_nm,
        'gate_ds':    avg_gate_ds,
        'gate_nm':    avg_gate_nm,
        'n_pos_tr': n_pos_tr, 'n_pos_te': n_pos_te,
    })

# ── 8. 결과 요약 ──────────────────────────────────────────────────────────────
elapsed = time.time() - t0
valid_all = [r['auprc_all']                         for r in auprc_rows if r['auprc_all']  is not None]
valid_ds  = [r['auprc_ds']                          for r in auprc_rows if r['auprc_ds']   is not None]
valid_nm  = [r['auprc_nm']                          for r in auprc_rows if r['auprc_nm']   is not None]
valid_pb_ds = [r['pos_bias_ds'] for r in auprc_rows if r['pos_bias_ds'] is not None and not np.isnan(r['pos_bias_ds'])]
valid_pb_nm = [r['pos_bias_nm'] for r in auprc_rows if r['pos_bias_nm'] is not None and not np.isnan(r['pos_bias_nm'])]
valid_gate_ds = [r['gate_ds'] for r in auprc_rows if r['gate_ds'] is not None]
valid_gate_nm = [r['gate_nm'] for r in auprc_rows if r['gate_nm'] is not None]

macro_all = np.mean(valid_all) if valid_all else 0
macro_ds  = np.mean(valid_ds)  if valid_ds  else 0
macro_nm  = np.mean(valid_nm)  if valid_nm  else 0
mean_pb_ds  = np.mean(valid_pb_ds)  if valid_pb_ds  else 0
mean_pb_nm  = np.mean(valid_pb_nm)  if valid_pb_nm  else 0
mean_gds = np.mean(valid_gate_ds) if valid_gate_ds else 0
mean_gnm = np.mean(valid_gate_nm) if valid_gate_nm else 0

print(f"\n{'='*70}")
print(f"  v15d_splice RESULTS (elapsed: {elapsed/60:.1f} min)")
print(f"{'='*70}")
print(f"\n  AUPRC Comparison:")
print(f"  {'Metric':<30} {'v15d (baseline)':>18} {'v15d_splice':>14} {'Δ':>8}")
print(f"  {'-'*72}")
print(f"  {'Macro AUPRC (ALL)':<30} {BASELINE_AUPRC:>18.4f} {macro_all:>14.4f} {macro_all-BASELINE_AUPRC:>+8.4f}")
print(f"  {'Macro AUPRC (DIFF_SPLICE)':<30} {BASELINE_DS:>18.4f} {macro_ds:>14.4f} {macro_ds-BASELINE_DS:>+8.4f}")
print(f"  {'Macro AUPRC (Normal)':<30} {BASELINE_NM:>18.4f} {macro_nm:>14.4f} {macro_nm-BASELINE_NM:>+8.4f}")

print(f"\n  pos_bias (DIFF_SPLICE):    {mean_pb_ds:.4f}  (baseline ≈ 0, goal: > 0)")
print(f"  pos_bias (Normal):         {mean_pb_nm:.4f}")

print(f"\n  Gate activation:")
print(f"  DIFF_SPLICE mean gate:  {mean_gds:.4f}")
print(f"  Normal      mean gate:  {mean_gnm:.4f}")
print(f"  Ratio DS/NM:            {mean_gds/mean_gnm:.3f}x  (>1 = gate activates more for zero-ESM2 genes)")

print(f"\n  REGRESSION CHECK: {'PASS' if macro_all >= BASELINE_AUPRC - 0.005 else 'FAIL'}")
print(f"    (≥ {BASELINE_AUPRC - 0.005:.4f} required, got {macro_all:.4f})")

# ── 9. 저장 ──────────────────────────────────────────────────────────────────
timestamp = time.strftime('%Y%m%d_%H%M')
score_path = f'{OUT_DIR}/score_matrix_splice_{timestamp}.npy'
np.save(score_path, score_matrix)

meta = {
    'model':           'v15d_splice',
    'timestamp':       timestamp,
    'baseline_auprc':  BASELINE_AUPRC,
    'macro_auprc_all': macro_all,
    'macro_auprc_ds':  macro_ds,
    'macro_auprc_nm':  macro_nm,
    'baseline_ds':     BASELINE_DS,
    'baseline_nm':     BASELINE_NM,
    'mean_pos_bias_ds': mean_pb_ds,
    'mean_pos_bias_nm': mean_pb_nm,
    'mean_gate_ds':    mean_gds,
    'mean_gate_nm':    mean_gnm,
    'gate_ratio_ds_nm': float(mean_gds / mean_gnm) if mean_gnm > 0 else None,
    'n_diff_splice_genes': len(diff_splice_genes),
    'n_normal_genes':      len(normal_genes),
    'per_go': auprc_rows,
    'elapsed_min': elapsed / 60,
}
meta_path = f'{OUT_DIR}/v15d_splice_meta_{timestamp}.json'
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\n  Score matrix → {score_path}")
print(f"  Meta         → {meta_path}")
print(f"{'='*70}")
