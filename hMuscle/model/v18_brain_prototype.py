# -*- coding: utf-8 -*-
"""
v18_brain_prototype.py
=======================
v17 brain model에 v9-1 prototype contrastive learning 적용.

동기:
  - v17 AUPRC=0.036 (단일 결정경계)
  - Silhouette 검증: 모든 brain GO term에서 k=2 구조 확인 (0.61~0.67)
  - 전제: pre-/post-synaptic, 막단백질/비막단백질 이분 구조를 prototype이 포착

Architecture:
  Phase 1 (prototype contrastive):
    ESM-2 640d → Dense(256,relu)→BN→Drop(0.3) → Dense(128,relu)→Drop(0.2)
              → Dense(64,relu) → L2-normalize → 64d embedding
    Loss: PrototypeContrastiveLoss(k=2) + L_diversity
    Init: KMeans(k=2) on positive embeddings
    EMA: prototype 갱신

  Phase 2 (classifier):
    [frozen feature extractor] → 64d → Dense(1, sigmoid)
    Loss: BinaryFocalCrossentropy(gamma=2.0)

데이터: brain isoforms, gene-block 80/20 split (seed=42, v17과 동일)
Annotation: human_annotations_unified_bp.txt (SwissProt BP + NCBI BP union)
GO terms: 10개 (brain-relevant, glutamatergic/GABAergic 포함)
"""

import os, json, time, re
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.cluster import KMeans
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

from v7c_prototype_contrastive import PrototypeContrastiveLoss

# ── Paths ─────────────────────────────────────────────────────────────────────
BRAIN_DIR = '../data/brain_esm2'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/v18_brain_prototype'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_FRAC   = 0.80
PROTO_K      = 2          # silhouette 검증으로 k=2 확정
PHASE1_EPOCH = 20
PHASE1_BATCH = 128
PHASE1_N_BATCH = 30       # batches per epoch
PHASE2_EPOCH = 80
PHASE2_BATCH = 512
N_SEEDS      = 3          # 5→3 (phase1 추가로 시간 절약)
LR           = 1e-3
GAMMA        = 2.0
LAMBDA_DIV   = 0.1
EMA_DECAY    = 0.9
TEMP         = 0.1

# ── GO terms (brain-focused, 10개) ────────────────────────────────────────────
GO_TERMS = {
    'GO:0007268': 'Synaptic transmission',
    'GO:0007010': 'Cytoskeleton org',
    'GO:0007018': 'MT-based movement',
    'GO:0048488': 'Synaptic vesicle endocytosis',
    'GO:0035249': 'Glutamatergic',
    'GO:0051932': 'GABAergic',
    'GO:0060078': 'Postsynaptic potential',
    'GO:0007005': 'Mitochondrion org',
    'GO:0043005': 'Neuron projection',
    'GO:0030182': 'Neuron diff',
}
GO_KEYS  = list(GO_TERMS.keys())
GO_NAMES = list(GO_TERMS.values())
N_GO = len(GO_KEYS)

# ── Load brain data ───────────────────────────────────────────────────────────
print("=" * 70)
print("  v18 Brain Prototype Model")
print("=" * 70)

X_all = np.load(f'{BRAIN_DIR}/brain_only_esm2_t30_150M.npy').astype(np.float32)

# gene symbols from GTF (one per transcript = one per isoform)
all_syms = []
with open(f'{BRAIN_DIR}/brain_only.gtf') as f:
    for line in f:
        if line.startswith('#'): continue
        fields = line.split('\t')
        if len(fields) < 9 or fields[2] != 'transcript': continue
        m = re.search(r'gene_name "([^"]+)"', fields[8])
        all_syms.append(m.group(1) if m else 'NA')
all_syms = np.array(all_syms)
assert len(all_syms) == len(X_all), "GTF/embedding size mismatch"

print(f"  Brain isoforms: {len(X_all)}, unique genes: {len(set(all_syms))}")

# ── Gene-block 80/20 split (v17과 동일: seed=42) ─────────────────────────────
all_genes = np.array(sorted(set(all_syms)))
np.random.seed(42)
gene_order = np.random.permutation(len(all_genes))
n_train_g  = int(len(all_genes) * TRAIN_FRAC)
train_genes = set(all_genes[gene_order[:n_train_g]])
test_genes  = set(all_genes[gene_order[n_train_g:]])

tr_mask = np.array([s in train_genes for s in all_syms])
ev_mask = np.array([s in test_genes  for s in all_syms])

X_tr = X_all[tr_mask]
X_ev = X_all[ev_mask]
tr_sym = all_syms[tr_mask]
ev_sym = all_syms[ev_mask]

print(f"  Train: {X_tr.shape}, Eval: {X_ev.shape}")

# ── Annotation ────────────────────────────────────────────────────────────────
gene2go = {}
with open(f'{ANNOT_DIR}/human_annotations_unified_bp.txt') as f:
    for line in f:
        p = line.strip().split()
        if p: gene2go[p[0]] = set(p[1:])

def load_labels(go_term):
    y_tr = np.array([1 if go_term in gene2go.get(s, set()) else 0
                     for s in tr_sym], dtype=np.float32)
    y_ev = np.array([1 if go_term in gene2go.get(s, set()) else 0
                     for s in ev_sym], dtype=np.float32)
    return y_tr, y_ev

# ── Model builders ────────────────────────────────────────────────────────────
def build_feature_model():
    """ESM-2 640d → 64d L2-normalized embedding"""
    inp = layers.Input(shape=(640,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64,  activation='relu')(x)
    emb = layers.Lambda(lambda z: tf.math.l2_normalize(z, axis=1))(x)
    return models.Model(inputs=inp, outputs=emb, name='feature_model')

def build_classifier(feature_model):
    """feature_model → sigmoid (feature model 학습 가능)"""
    inp = layers.Input(shape=(640,))
    emb = feature_model(inp, training=True)
    out = layers.Dense(1, activation='sigmoid')(emb)
    return models.Model(inputs=inp, outputs=out, name='classifier')

# ── Phase 1: Prototype contrastive training ───────────────────────────────────
def run_phase1(feature_model, proto_fn, X_pos, X_neg, adam_p1):
    """단순 미니배치 prototype contrastive 학습"""
    pos_idx_all = np.arange(len(X_pos))
    neg_idx_all = np.arange(len(X_neg))

    n_pos_b = min(PHASE1_BATCH // 2, len(X_pos))
    n_neg_b = min(n_pos_b, len(X_neg))
    replace_pos = len(X_pos) < n_pos_b
    replace_neg = len(X_neg) < n_neg_b

    epoch_proto_losses = []
    for epoch in range(PHASE1_EPOCH):
        batch_losses = []
        all_pos_emb = []
        for _ in range(PHASE1_N_BATCH):
            pi = np.random.choice(pos_idx_all, n_pos_b, replace=replace_pos)
            ni = np.random.choice(neg_idx_all, n_neg_b, replace=replace_neg)
            with tf.GradientTape() as tape:
                pos_emb = feature_model(X_pos[pi], training=True)
                neg_emb = feature_model(X_neg[ni], training=True)
                L_p, L_d = proto_fn.compute_loss(pos_emb, neg_emb)
                loss = L_p + LAMBDA_DIV * L_d
            grads = tape.gradient(loss, feature_model.trainable_variables)
            adam_p1.apply_gradients(zip(grads, feature_model.trainable_variables))
            batch_losses.append(float(loss))
            all_pos_emb.append(pos_emb.numpy())

        # EMA prototype update
        proto_fn.update_prototypes_ema(np.vstack(all_pos_emb))
        epoch_proto_losses.append(np.mean(batch_losses))

    return np.mean(epoch_proto_losses[-5:])  # last-5 mean

# ── Per-GO ensemble ───────────────────────────────────────────────────────────
def run_ensemble(y_tr, y_ev, go_name):
    n_pos_tr = int(y_tr.sum())
    n_pos_ev = int(y_ev.sum())
    if n_pos_tr < PROTO_K * 3 or n_pos_ev == 0:
        return np.zeros(len(X_ev)), 0.0

    seed_preds = []
    for s in range(N_SEEDS):
        tf.random.set_seed(s * 137 + 42)
        np.random.seed(s * 137 + 42)

        feat_m = build_feature_model()

        # Phase 1 ── prototype contrastive
        pos_init_emb = feat_m.predict(X_tr[y_tr == 1], batch_size=512, verbose=0)
        proto_fn = PrototypeContrastiveLoss(
            n_prototypes=PROTO_K, emb_dim=64,
            temperature=TEMP, ema_decay=EMA_DECAY, lambda_div=LAMBDA_DIV)
        proto_fn.initialize_from_embeddings(pos_init_emb)

        adam_p1 = tf.keras.optimizers.Adam(LR)
        p1_loss = run_phase1(feat_m, proto_fn,
                             X_tr[y_tr == 1], X_tr[y_tr == 0], adam_p1)

        # Phase 2 ── classifier (feature model unfrozen)
        clf = build_classifier(feat_m)
        clf.compile(
            optimizer=tf.keras.optimizers.Adam(LR * 0.5),
            loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=GAMMA))
        cb = [callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                      restore_best_weights=True, verbose=0),
              callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)]
        idx = np.random.permutation(len(X_tr))
        clf.fit(X_tr[idx], y_tr[idx],
                epochs=PHASE2_EPOCH, batch_size=PHASE2_BATCH,
                validation_split=0.1, callbacks=cb, verbose=0)

        seed_preds.append(clf.predict(X_ev, batch_size=1024, verbose=0).flatten())

    preds = np.mean(seed_preds, axis=0)
    auprc = average_precision_score(y_ev, preds)
    return preds, auprc

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\n  Score matrix: {len(X_ev)} eval isoforms × {N_GO} GO terms, {N_SEEDS} seeds")
print(f"  Protocol: Phase1(proto k={PROTO_K}, {PHASE1_EPOCH}ep) + Phase2({PHASE2_EPOCH}ep)")

score_matrix = np.zeros((len(X_ev), N_GO), dtype=np.float32)
auprc_row    = []

for gi, (go_term, go_name) in enumerate(GO_TERMS.items()):
    t0 = time.time()
    y_tr, y_ev = load_labels(go_term)
    preds, auprc = run_ensemble(y_tr, y_ev, go_name)
    score_matrix[:, gi] = preds
    auprc_row.append(auprc)
    n_pos_tr, n_pos_ev = int(y_tr.sum()), int(y_ev.sum())
    print(f"  [{gi+1:2d}/{N_GO}] {go_name[:22]:22s}  AUPRC={auprc:.4f}"
          f"  pos_tr={n_pos_tr:4d}  pos_ev={n_pos_ev:4d}  ({time.time()-t0:.0f}s)")

macro = float(np.mean([a for a in auprc_row if a > 0]))
print(f"\n  Macro AUPRC (valid terms): {macro:.4f}")
print(f"  v17 baseline:              0.0355")
print(f"  Improvement:               {macro - 0.0355:+.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M')

np.save(f'{OUT_DIR}/score_matrix_v18_{ts}.npy', score_matrix)

result = {
    'timestamp': ts,
    'model': 'v18_brain_prototype_k2',
    'n_go_terms': N_GO,
    'go_terms': GO_TERMS,
    'proto_k': PROTO_K,
    'phase1_epochs': PHASE1_EPOCH,
    'phase2_epochs': PHASE2_EPOCH,
    'n_seeds': N_SEEDS,
    'annotation': 'human_annotations_unified_bp.txt',
    'auprc_per_go': {GO_KEYS[i]: round(auprc_row[i], 4) for i in range(N_GO)},
    'macro_auprc': round(macro, 4),
    'v17_baseline': 0.0355,
    'improvement': round(macro - 0.0355, 4),
}
json.dump(result, open(f'{OUT_DIR}/v18_results_{ts}.json', 'w'), indent=2)
print(f"\n  [Saved] {OUT_DIR}/v18_results_{ts}.json")
print(f"  [Saved] {OUT_DIR}/score_matrix_v18_{ts}.npy")
print("\nALL DONE")
