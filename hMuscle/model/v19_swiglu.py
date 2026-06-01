# -*- coding: utf-8 -*-
"""
v19_swiglu.py
==============
v15d_bp_clean 기반 아키텍처 고도화:

변경사항:
  1. SwiGLU activation: ReLU → Swish(xW_gate) ⊙ xW_val
     - LLaMA-3/PaLM 표준. feature interaction 정교화.
  2. Residual connection: 각 SwiGLU block에 skip connection
     - 640d → 512d → 256d → 128d 진행 중 원본 신호 보존
  3. Delta embedding: X_i - mean(X_gene) 추가 입력
     - gene-level dominance 구조적 차단
     - within-gene isoform-specific 신호 격리
     - XGBoost 대비 isoform 차별화 강점 극대화

Architecture:
  Input: [ESM-2 640d, delta_640d]
  → concat → 1280d
  → SwiGLU_Block(1280→512) + Residual(1280→512)
  → SwiGLU_Block(512→256) + Residual
  → Dense(128, relu) → Dense(64, relu)
  → Dense(1, sigmoid)

  SwiGLU Block:
    gate  = Dense(dim)(x)
    value = Dense(dim)(x)
    out   = swish(gate) * value   [feature interaction]
    out   = LayerNorm(out)
    out   = out + Dense(dim)(x)   [residual]
    out   = Dropout(rate)(out)
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
OUT_DIR   = '../../reports/v19_swiglu'
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS      = 5
REVERSAL_GAP = 0.20
TARGET_GENES = ['TPM1', 'TPM2', 'KIF1B', 'SEH1L', 'GABARAPL1', 'DMD', 'TPM3', 'OBSCN']

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

print("=" * 70)
print("  v19 SwiGLU + Residual + Delta Embedding")
print("=" * 70)

def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

X_tr_raw = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te_raw = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')
te_geneid = load_ids('my_gene_list_fixed.npy')
te_isoid  = load_ids('my_isoform_list_fixed.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

tr_sym       = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_geneid]
te_sym       = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_geneid]
te_ensg_base = [g.split('.')[0] for g in te_geneid]
te_enst_base = [x.split('.')[0] for x in te_isoid]
N_TE = len(te_isoid)

# ── Delta embedding 계산 ──────────────────────────────────────────────────────
print("  Computing delta embeddings (X_i - gene_mean) ...")

def compute_delta(X, syms):
    """X_i - mean(X_j for j in same gene)"""
    sym_arr = np.array(syms)
    delta = np.zeros_like(X)
    unique_genes = np.unique(sym_arr)
    for gene in unique_genes:
        idx = np.where(sym_arr == gene)[0]
        if len(idx) == 1:
            delta[idx] = 0.0   # single isoform: delta = 0
        else:
            gene_mean = X[idx].mean(axis=0)
            delta[idx] = X[idx] - gene_mean
    return delta

X_tr_delta = compute_delta(X_tr_raw, tr_sym)
X_te_delta = compute_delta(X_te_raw, te_sym)

# concat: [ESM-2 | delta] → 1280d
X_tr = np.concatenate([X_tr_raw, X_tr_delta], axis=1).astype(np.float32)
X_te = np.concatenate([X_te_raw, X_te_delta], axis=1).astype(np.float32)
print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")
print(f"  GO terms: {N_GO} (BP-only, 18 terms)")

# ── SwiGLU block ──────────────────────────────────────────────────────────────
def swiglu_residual_block(x, dim, dropout_rate=0.3):
    """SwiGLU + Residual + LayerNorm"""
    in_dim = x.shape[-1]
    gate  = layers.Dense(dim)(x)
    value = layers.Dense(dim)(x)
    out   = layers.Activation('swish')(gate) * value   # SwiGLU
    out   = layers.LayerNormalization()(out)
    # Residual: project input to same dim if needed
    if in_dim != dim:
        shortcut = layers.Dense(dim, use_bias=False)(x)
    else:
        shortcut = x
    out = out + shortcut
    out = layers.Dropout(dropout_rate)(out)
    return out

def build_v19():
    inp = layers.Input(shape=(1280,))           # ESM-2 640d + delta 640d
    x = swiglu_residual_block(inp, 512, 0.3)    # 1280 → 512
    x = swiglu_residual_block(x,   256, 0.2)    # 512  → 256
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64,  activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=inp, outputs=out)

def run_ensemble(y_tr, y_te):
    seed_preds = []
    for s in range(N_SEEDS):
        tf.random.set_seed(s * 137 + 42)
        np.random.seed(s * 137 + 42)
        idx = np.random.permutation(len(X_tr))
        m = build_v19()
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
        cb = [callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                      restore_best_weights=True, verbose=0),
              callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)]
        m.fit(X_tr[idx], y_tr[idx], epochs=80, batch_size=512,
              validation_split=0.1, callbacks=cb, verbose=0)
        seed_preds.append(m.predict(X_te, batch_size=1024, verbose=0).flatten())
    preds = np.mean(seed_preds, axis=0)
    auprc = average_precision_score(y_te, preds) if y_te.sum() > 0 else 0.0
    return preds, auprc

def load_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations_unified_bp.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\n  Score matrix: {N_TE} isoforms × {N_GO} GO terms, {N_SEEDS} seeds")
score_matrix = np.zeros((N_TE, N_GO), dtype=np.float32)
auprc_row    = []

for gi, (go_term, go_name) in enumerate(GO_TERMS.items()):
    t0 = time.time()
    y_tr, y_te = load_labels(go_term)
    preds, auprc = run_ensemble(y_tr, y_te)
    score_matrix[:, gi] = preds
    auprc_row.append(auprc)
    n_pos = int(y_te.sum())
    tag = ' [NEW]' if gi >= 13 else ''
    print(f"  [{gi+1:2d}/{N_GO}] {go_name[:22]:22s}  AUPRC={auprc:.4f}  pos={n_pos:4d}  ({time.time()-t0:.0f}s){tag}")

valid_auprcs = [a for a in auprc_row if a > 0]
macro_all  = float(np.mean(auprc_row))
macro_valid = float(np.mean(valid_auprcs))
print(f"\n  Macro AUPRC (all 18):   {macro_all:.4f}")
print(f"  Macro AUPRC (valid):    {macro_valid:.4f}")
print(f"  v15d_bp_clean baseline: (pending)")
print(f"  v10-B original:         0.6935")

# ── Gene-level index ──────────────────────────────────────────────────────────
gene_isos = defaultdict(list)
for i, sym in enumerate(te_sym):
    gene_isos[sym].append((i, te_enst_base[i], te_ensg_base[i]))

# ── Reversal detection ────────────────────────────────────────────────────────
reversals = []
for sym, iso_list in gene_isos.items():
    if len(iso_list) < 2: continue
    idxs  = [x[0] for x in iso_list]
    ensts = [x[1] for x in iso_list]
    ensg  = iso_list[0][2]
    gs    = score_matrix[idxs, :]
    for a in range(len(iso_list)):
        for b in range(a+1, len(iso_list)):
            diff   = gs[a] - gs[b]
            a_wins = [(gi, diff[gi])  for gi in range(N_GO) if diff[gi]  >= REVERSAL_GAP]
            b_wins = [(gi, -diff[gi]) for gi in range(N_GO) if diff[gi] <= -REVERSAL_GAP]
            if not a_wins or not b_wins: continue
            best_a = max(a_wins, key=lambda x: x[1])
            best_b = max(b_wins, key=lambda x: x[1])
            reversals.append({
                'gene': sym, 'ensg': ensg,
                'iso_a': ensts[a], 'iso_b': ensts[b],
                'go_A': GO_KEYS[best_a[0]], 'name_A': GO_NAMES[best_a[0]], 'gap_A': round(float(best_a[1]),4),
                'go_B': GO_KEYS[best_b[0]], 'name_B': GO_NAMES[best_b[0]], 'gap_B': round(float(best_b[1]),4),
                'reversal_strength': round(float(best_a[1]+best_b[1]),4),
            })

reversals.sort(key=lambda x: -x['reversal_strength'])
seen = set()
top_rev = [r for r in reversals if not seen.__contains__(r['gene']) and not seen.add(r['gene'])]

print(f"\n  Reversal genes: {len(top_rev)}  (v15d baseline: 123)")

# ── Save ──────────────────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M')

np.save(f'{OUT_DIR}/score_matrix_v19_{ts}.npy', score_matrix)
result = {
    'timestamp': ts, 'model': 'v19_swiglu_residual_delta',
    'arch': {'activation': 'SwiGLU', 'residual': True, 'delta_embedding': True,
             'input_dim': 1280, 'hidden': [512, 256, 128, 64]},
    'n_go_terms': N_GO, 'go_terms': GO_TERMS,
    'auprc_per_go': {GO_KEYS[i]: round(auprc_row[i], 4) for i in range(N_GO)},
    'macro_auprc_all18': round(macro_all, 4),
    'macro_auprc_valid': round(macro_valid, 4),
    'v10b_baseline': 0.6935,
    'n_reversal_genes': len(top_rev),
    'top20_reversals': top_rev[:20],
}
json.dump(result, open(f'{OUT_DIR}/v19_results_{ts}.json', 'w'), indent=2)
print(f"\n  [Saved] {OUT_DIR}/v19_results_{ts}.json")
print(f"  [Saved] {OUT_DIR}/score_matrix_v19_{ts}.npy")
print("\nALL DONE")
