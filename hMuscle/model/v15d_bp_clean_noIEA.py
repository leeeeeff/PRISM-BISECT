# -*- coding: utf-8 -*-
"""
v15d_bp_clean_noIEA.py
========================
v15d_bp_clean과 동일한 아키텍처/하이퍼파라미터.
단일 변경: human_annotations_unified_bp.txt → human_annotations_noIEA_bp.txt

목적: IEA(Inferred from Electronic Annotation) 제외 훈련 → noIEA 라벨로 평가.
      DEF-B1 full ablation — reviewer 요구 시 제출용.
      결과는 reports/v15_bp_clean_noIEA/ 에 저장.

실행:
  conda activate isoform_env
  nohup python v15d_bp_clean_noIEA.py > ../../logs_isoform/v15d_noIEA_$(date +%Y%m%d_%H%M).log 2>&1 &

비교 기준 (DEF-B1 결과):
  Full-train / Full-eval:    0.7022 (기준)
  Full-train / noIEA-eval:   0.6584 (−6.2%, 비대칭 평가)
  noIEA-train / noIEA-eval:  TBD (이 스크립트의 목표)
"""

import os, json, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
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

DATA_DIR  = '../data'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v15_bp_clean_noIEA'

# KEY DIFFERENCE: noIEA annotation file
ANNOT_FILE = f'{ANNOT_DIR}/human_annotations_noIEA_bp.txt'

N_SEEDS      = 5
REVERSAL_GAP = 0.20
TARGET_GENES = ['TPM1', 'TPM2', 'KIF1B', 'SEH1L', 'GABARAPL1', 'DMD', 'TPM3', 'OBSCN']

# ── 18 GO terms (identical to v15d_bp_clean) ─────────────────────────────────
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
GO_KEYS = list(GO_TERMS.keys())
GO_NAMES = list(GO_TERMS.values())
N_GO = len(GO_KEYS)

# ── Load features (identical to v15d_bp_clean) ────────────────────────────────
print("="*70)
print("  v15d_bp_clean_noIEA — IEA-excluded training (reviewer ablation)")
print(f"  Annotation file: {ANNOT_FILE}")
print("="*70)

def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

def load_ensembl_to_symbol():
    mapping = {}
    with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5 and parts[4]:
                mapping[parts[0]] = parts[4]
                mapping[parts[1]] = parts[4]
    return mapping

print("\nLoading IDs and features...")
te_iso  = load_ids('my_isoform_list_fixed.npy')
te_gene = load_ids('my_gene_list_fixed.npy')
tr_gene = load_ids(f'{ID_DIR}/train_gene_list.npy')

X_tr = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

tr_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_gene]
te_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_gene]
print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

N_TE = X_te.shape[0]

def build_model(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out)

def train_and_eval(y_tr, y_te, seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    m = build_model(X_tr.shape[1])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    )
    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor='val_loss'),
    ]
    m.fit(X_tr, y_tr, epochs=100, batch_size=256, validation_split=0.1,
          callbacks=cb, verbose=0)
    preds = m.predict(X_te, batch_size=1024, verbose=0).flatten()
    auprc = average_precision_score(y_te, preds) if y_te.sum() > 0 else 0.0
    return preds, auprc

def load_labels(go_term):
    pos = set()
    with open(ANNOT_FILE) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te

# ── Build full score matrix ───────────────────────────────────────────────────
print(f"\n  Building full score matrix ({N_TE} isoforms × {N_GO} GO terms) ...")
score_matrix = np.zeros((N_TE, N_GO), dtype=np.float32)
auprc_row    = []

os.makedirs(OUT_DIR, exist_ok=True)
t0 = time.time()

for j, (go, name) in enumerate(zip(GO_KEYS, GO_NAMES)):
    y_tr, y_te = load_labels(go)
    n_pos_tr = int(y_tr.sum()); n_pos_te = int(y_te.sum())
    print(f"  [{j+1:2d}/18] {go} ({name}) | tr_pos={n_pos_tr}, te_pos={n_pos_te}", flush=True)

    if n_pos_tr < 5:
        print(f"         SKIP (< 5 training positives)")
        continue

    seed_preds = []
    seed_auprcs = []
    for seed in range(N_SEEDS):
        preds, auprc = train_and_eval(y_tr, y_te, seed)
        seed_preds.append(preds)
        seed_auprcs.append(auprc)
        print(f"         seed={seed} AUPRC={auprc:.4f}", flush=True)

    ens_preds = np.mean(seed_preds, axis=0)
    ens_auprc = average_precision_score(y_te, ens_preds) if n_pos_te > 0 else 0.0
    score_matrix[:, j] = ens_preds
    auprc_row.append((go, name, ens_auprc, n_pos_tr, n_pos_te))
    print(f"         Ensemble AUPRC={ens_auprc:.4f}", flush=True)

valid_auprcs = [a for _, _, a, _, _ in auprc_row if a > 0]
macro_auprc = np.mean(valid_auprcs) if valid_auprcs else 0.0
elapsed = time.time() - t0

print(f"\n{'='*70}")
print(f"  noIEA Training Complete — Macro AUPRC: {macro_auprc:.4f}")
print(f"  Elapsed: {elapsed/60:.1f} min")
print(f"{'='*70}")

# Compare with full-label results
print("\n  === COMPARISON ===")
print(f"  Full-train / Full-eval:    0.7022 (v15d_bp_clean, reference)")
print(f"  Full-train / noIEA-eval:   0.6584 (DEF-B1 asymmetric, -6.2%)")
print(f"  noIEA-train / noIEA-eval:  {macro_auprc:.4f} (this run)")

timestamp = time.strftime('%Y%m%d_%H%M')
score_path = f'{OUT_DIR}/score_matrix_noIEA_{timestamp}.npy'
np.save(score_path, score_matrix)

meta = {
    'model': 'v15d_bp_clean_noIEA',
    'annot_file': 'human_annotations_noIEA_bp.txt',
    'macro_auprc': macro_auprc,
    'full_train_full_eval_reference': 0.7022,
    'full_train_noIEA_eval': 0.6584,
    'per_go': [
        {'go': go, 'name': nm, 'auprc': float(a), 'n_pos_tr': int(nt), 'n_pos_te': int(ne)}
        for go, nm, a, nt, ne in auprc_row
    ],
    'timestamp': timestamp,
}
with open(f'{OUT_DIR}/noIEA_meta_{timestamp}.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\n  Score matrix → {score_path}")
print(f"  Meta → {OUT_DIR}/noIEA_meta_{timestamp}.json")
