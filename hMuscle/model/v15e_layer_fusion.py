# -*- coding: utf-8 -*-
"""
v15e_layer_fusion.py
=====================
v15d 기반. ESM-2 L7+L18+L27 3-layer fusion (1920-dim) 입력으로 교체.

변경 근거:
  - Layer probe 결과: L30 단일 레이어는 18개 GO term 중 0개에서 최적
  - L7(interaction surface) + L18(functional motif) + L27(multi-domain) 융합 시
    probe 기준 95.8% of peak vs L30 기준 87.2% of peak (+8.6 pp)
  - MLP 비선형 이득 (NAS > 1.5) GO term들에서 특히 효과적

입력 파일:
  (training)
  ../data/esm2_train_human_layer07_t30_150M.npy  (31668, 640)
  ../data/esm2_train_human_layer18_t30_150M.npy  (31668, 640)
  ../data/esm2_train_human_layer27_t30_150M.npy  (31668, 640)

  (test)
  ../data/esm2_layer_07_t30_150M.npy  (36748, 640)
  ../data/esm2_layer_18_t30_150M.npy  (36748, 640)
  ../data/esm2_layer_27_t30_150M.npy  (36748, 640)

  (brain - optional, if layer files available)
  ../data/brain_isoquant_esm2/full/brain_full_esm2_layer07_t30_150M.npy  (63994, 640)
  ../data/brain_isoquant_esm2/full/brain_full_esm2_layer18_t30_150M.npy  (63994, 640)
  ../data/brain_isoquant_esm2/full/brain_full_esm2_layer27_t30_150M.npy  (63994, 640)

실행:
  cd hMuscle/model/
  nohup python3 -u v15e_layer_fusion.py \
      > ../../logs_isoform/v15e_$(date +%Y%m%d_%H%M).log 2>&1 &
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
OUT_DIR   = '../../reports/v15_bp_clean'
BRAIN_DIR = '../data/brain_isoquant_esm2/full'

FUSION_LAYERS = [7, 18, 27]
INPUT_DIM     = 640 * len(FUSION_LAYERS)  # 1920

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
N_GO = len(GO_KEYS)

# ── v15d baseline AUPRC (from previous runs) — for comparison ─────────────────
V15D_AUPRC_BASELINE = {
    'GO:0007204': None, 'GO:0045214': None, 'GO:0006941': None, 'GO:0006914': None,
    'GO:0043161': None, 'GO:0007519': None, 'GO:0042692': None, 'GO:0055074': None,
    'GO:0007005': None, 'GO:0007517': None, 'GO:0032006': None, 'GO:0030048': None,
    'GO:0006096': None, 'GO:0007268': None, 'GO:0007018': None, 'GO:0031175': None,
    'GO:0030182': None, 'GO:0000226': None,
}

print("=" * 70)
print("  v15e_layer_fusion: L7+L18+L27 concat (1920-dim) input")
print("=" * 70)


def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def load_fused_embeddings(layer_files):
    """레이어별 embedding을 로드하여 axis=1 concatenation."""
    parts = []
    for f in layer_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing layer file: {f}")
        parts.append(np.load(f).astype(np.float32))
    return np.concatenate(parts, axis=1)


# ── Load test embeddings (L7+L18+L27 fused) ──────────────────────────────────
test_layer_files = [
    os.path.join(DATA_DIR, f'esm2_layer_{L:02d}_t30_150M.npy') for L in FUSION_LAYERS
]
X_te = load_fused_embeddings(test_layer_files)

# ── Load training embeddings ──────────────────────────────────────────────────
train_layer_files = [
    os.path.join(DATA_DIR, f'esm2_train_human_layer{L:02d}_t30_150M.npy') for L in FUSION_LAYERS
]
missing_train = [f for f in train_layer_files if not os.path.exists(f)]
if missing_train:
    print(f"\n  ERROR: Training layer files missing:")
    for f in missing_train:
        print(f"    {f}")
    print("  Run compute_esm2_train_layers.py first, then re-run v15e.")
    import sys; sys.exit(1)

X_tr = load_fused_embeddings(train_layer_files)
TRAIN_MODE = 'L7+L18+L27'

# ── Load brain embeddings (optional) ─────────────────────────────────────────
brain_layer_files = [
    os.path.join(BRAIN_DIR, f'brain_full_esm2_layer{L:02d}_t30_150M.npy') for L in FUSION_LAYERS
]
missing_brain = [f for f in brain_layer_files if not os.path.exists(f)]
if not missing_brain:
    X_brain = load_fused_embeddings(brain_layer_files)
    brain_labels = np.load(f'{BRAIN_DIR}/brain_full_labels.npy', allow_pickle=True)
    brain_ids    = np.load(f'{BRAIN_DIR}/brain_full_ids.npy',    allow_pickle=True)
    BRAIN_AVAILABLE = True
    print(f"  Brain embeddings loaded: {X_brain.shape}")
else:
    BRAIN_AVAILABLE = False
    print(f"  Brain embeddings not available (run compute_brain_isoquant_layers.py)")

# ── ID lists ──────────────────────────────────────────────────────────────────
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

print(f"  Train: {X_tr.shape}  Test: {X_te.shape}  Mode: {TRAIN_MODE}")
print(f"  GO terms: {N_GO}")


def build_v15e():
    inp = layers.Input(shape=(INPUT_DIM,))
    x = layers.Dense(512, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64,  activation='relu')(x)
    out = layers.Dense(1,  activation='sigmoid')(x)
    return models.Model(inputs=inp, outputs=out)


def run_ensemble(X_train, y_tr, X_test, y_te, X_b=None, y_b=None):
    seed_preds_te = []
    seed_preds_b  = [] if (X_b is not None) else None

    for s in range(N_SEEDS):
        tf.random.set_seed(s * 137 + 42)
        np.random.seed(s * 137 + 42)
        idx = np.random.permutation(len(X_train))
        m = build_v15e()
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
        cb = [callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                      restore_best_weights=True, verbose=0),
              callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)]
        m.fit(X_train[idx], y_tr[idx], epochs=80, batch_size=512,
              validation_split=0.1, callbacks=cb, verbose=0)
        seed_preds_te.append(m.predict(X_test, batch_size=1024, verbose=0).flatten())
        if X_b is not None:
            seed_preds_b.append(m.predict(X_b, batch_size=1024, verbose=0).flatten())

    preds_te = np.mean(seed_preds_te, axis=0)
    auprc_te = average_precision_score(y_te, preds_te) if y_te.sum() > 0 else 0.0

    if X_b is not None and y_b is not None and y_b.sum() > 0:
        preds_b  = np.mean(seed_preds_b, axis=0)
        auprc_b  = average_precision_score(y_b, preds_b)
    else:
        preds_b, auprc_b = None, None

    return preds_te, auprc_te, preds_b, auprc_b


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


print(f"\n  Building score matrix ({N_TE} isoforms × {N_GO} GO terms) ...")
print(f"  Training mode: {TRAIN_MODE}")
if BRAIN_AVAILABLE:
    print(f"  Brain evaluation: ENABLED ({X_brain.shape[0]} isoforms)")
print()

score_matrix = np.zeros((N_TE, N_GO), dtype=np.float32)
auprc_te_row  = []
auprc_br_row  = []

for gi, (go_term, go_name) in enumerate(GO_TERMS.items()):
    t0 = time.time()
    y_tr, y_te = load_labels(go_term)

    if BRAIN_AVAILABLE:
        y_brain = brain_labels[:, gi].astype(np.float32)
        preds_te, auprc_te, preds_b, auprc_b = run_ensemble(
            X_tr, y_tr, X_te, y_te, X_brain, y_brain)
        auprc_br_row.append(auprc_b)
        brain_str = f"  Brain={auprc_b:.4f}" if auprc_b is not None else "  Brain=N/A"
    else:
        preds_te, auprc_te, _, _ = run_ensemble(X_tr, y_tr, X_te, y_te)
        brain_str = ""

    score_matrix[:, gi] = preds_te
    auprc_te_row.append(auprc_te)
    n_pos = int(y_te.sum())
    print(f"  [{gi+1:2d}/{N_GO}] {go_name[:22]:22s}  AUPRC={auprc_te:.4f}  "
          f"pos={n_pos:4d}  ({time.time()-t0:.0f}s){brain_str}")

print(f"\n  Macro AUPRC (all 18): {np.mean(auprc_te_row):.4f}")
print(f"  Macro AUPRC (13 muscle): {np.mean(auprc_te_row[:13]):.4f}")
print(f"  Macro AUPRC (5 neuro): {np.mean(auprc_te_row[13:]):.4f}")
if BRAIN_AVAILABLE and auprc_br_row:
    valid_br = [x for x in auprc_br_row if x is not None]
    print(f"  Brain Macro AUPRC: {np.mean(valid_br):.4f}")

# ── Save results ──────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out = {
    'model': 'v15e_layer_fusion',
    'input_dim': INPUT_DIM,
    'fusion_layers': FUSION_LAYERS,
    'train_mode': TRAIN_MODE,
    'go_terms': GO_TERMS,
    'auprc_muscle_test': {go: float(auprc_te_row[i]) for i, go in enumerate(GO_KEYS)},
}
if BRAIN_AVAILABLE and auprc_br_row:
    out['auprc_brain'] = {go: float(auprc_br_row[i]) if auprc_br_row[i] is not None else None
                          for i, go in enumerate(GO_KEYS)}

out_path = os.path.join(OUT_DIR, 'v15e_layer_fusion_results.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Results saved: {out_path}")

# ── Reversal detection (same as v15d) ─────────────────────────────────────────
gene_isos = defaultdict(list)
for i, sym in enumerate(te_sym):
    gene_isos[sym].append((i, te_enst_base[i], te_ensg_base[i]))

print("\n  Detecting cross-GO reversals ...")
reversals = []
for sym, iso_list in gene_isos.items():
    if len(iso_list) < 2: continue
    idxs   = [x[0] for x in iso_list]
    ensts  = [x[1] for x in iso_list]
    ensg   = iso_list[0][2]
    gene_scores = score_matrix[idxs, :]
    for a in range(len(iso_list)):
        for b in range(a+1, len(iso_list)):
            diff   = gene_scores[a] - gene_scores[b]
            a_wins = [(gi, diff[gi])  for gi in range(N_GO) if diff[gi]  >=  REVERSAL_GAP]
            b_wins = [(gi, -diff[gi]) for gi in range(N_GO) if diff[gi]  <= -REVERSAL_GAP]
            if not a_wins or not b_wins: continue
            best_a = max(a_wins, key=lambda x: x[1])
            best_b = max(b_wins, key=lambda x: x[1])
            reversals.append({
                'gene': sym, 'ensg': ensg,
                'isoA': ensts[a], 'isoB': ensts[b],
                'go_a_wins': GO_NAMES[best_a[0]], 'score_a': float(best_a[1]),
                'go_b_wins': GO_NAMES[best_b[0]], 'score_b': float(best_b[1]),
            })

print(f"  Total reversals detected: {len(reversals)}")

# ── Target gene highlights ────────────────────────────────────────────────────
print(f"\n  Target gene reversals: {TARGET_GENES}")
target_revs = [r for r in reversals if r['gene'] in TARGET_GENES]
for r in sorted(target_revs, key=lambda x: -max(x['score_a'], x['score_b'])):
    print(f"    {r['gene']}: {r['isoA']} vs {r['isoB']}  "
          f"| A={r['go_a_wins']}(+{r['score_a']:.3f})  B={r['go_b_wins']}(+{r['score_b']:.3f})")

# Save full results
out['reversals'] = reversals[:500]
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nFull results saved: {out_path}")
print("DONE")
