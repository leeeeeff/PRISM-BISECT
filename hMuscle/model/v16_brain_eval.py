# -*- coding: utf-8 -*-
"""
v16_brain_eval.py — 4-Stream Gated Residual Fusion: Brain Zero-Shot Transfer
=============================================================================

Train: 근육 데이터 (ESM-2 + Splice + RNA + LOC)
Test:  뇌 IsoQuant full set (63,994 isoforms)
       Stream B = zeros (brain splice_delta 미계산)
       Stream C = rna_features_brain_full.npy (46.5% coverage)
       Stream D = loc_features_brain_full.npy (46.5% coverage)

Baseline: v15d_brain_eval macro AUPRC = 0.5998
목표:     > 0.5998 (zero-shot transfer 개선)

use_bias=False guarantee:
  brain splice/RNA/LOC = 0 → h_aux = 0 → h_fused = h_esm (v15d fallback)
  coverage 46.5% 이소폼은 추가 정보 활용, 나머지 53.5%는 ESM-2만 사용

실행:
  cd /home/welcome1/sw1686/DIFFUSE/hMuscle/model
  PYTHONUNBUFFERED=1 ~/miniconda3/envs/isoform_env/bin/python v16_brain_eval.py \\
    > ../../logs_isoform/v16_brain_$(date +%Y%m%d_%H%M).log 2>&1 &
"""

import os, json, time
import numpy as np
import pandas as pd
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

# ── 경로 ─────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
BRAIN_DIR = '../data/brain_isoquant_esm2/full'
FEAT_DIR  = '../results_isoform/features'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v16_brain'
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS           = 5
BASELINE_AUPRC_BR = 0.5998   # v15d brain zero-shot
BASELINE_AUPRC_MU = 0.7022   # v15d muscle (reference)
FEAT_DROP_P       = 0.20

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


def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


print("=" * 70)
print("  v16_brain_eval — 4-Stream Brain Zero-Shot Transfer")
print(f"  Streams: ESM-2(640) + Splice(150,zeros) + RNA(9) + LOC(8)")
print(f"  Feature Dropout p={FEAT_DROP_P} | Brain Baseline={BASELINE_AUPRC_BR}")
print("=" * 70)

# ── 1. Feature 로드 ─────────────────────────────────────────────────────────
print("\n[1] Loading features...")

# IDs
tr_gene = load_ids(f'{ID_DIR}/train_gene_list.npy')
te_iso  = load_ids(f'{BRAIN_DIR}/brain_full_ids.npy')
te_sym  = load_ids(f'{BRAIN_DIR}/brain_full_gene_names.npy')  # already gene symbols

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
tr_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_gene]

# Stream A: ESM-2
X_tr_esm = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te_esm = np.load(f'{BRAIN_DIR}/brain_full_esm2_t30_150M.npy').astype(np.float32)
print(f"  ESM-2:  tr={X_tr_esm.shape}, te={X_te_esm.shape}")

# Stream B: Splice delta
# Train: real values (if available)
SD_PATH       = f'{FEAT_DIR}/splicing/splicing_delta_v2.npy'
TRAIN_SD_PATH = f'{FEAT_DIR}/splicing/train_splicing_delta.npy'
X_te_sd_tmp = np.load(SD_PATH).astype(np.float32)     # muscle test (shape reference)
SD_DIM = X_te_sd_tmp.shape[1]                          # 150
X_tr_sd = (np.load(TRAIN_SD_PATH).astype(np.float32)
           if os.path.exists(TRAIN_SD_PATH)
           else np.zeros((X_tr_esm.shape[0], SD_DIM), np.float32))
X_te_sd = np.zeros((X_te_esm.shape[0], SD_DIM), np.float32)   # brain: all zeros
print(f"  Splice: tr nonzero={( X_tr_sd.sum(1)!=0).mean()*100:.1f}%, "
      f"te=zeros (brain splice_delta not computed)")

# Stream C: RNA stability
RNA_TR   = f'{FEAT_DIR}/rna/rna_features_train.npy'
RNA_BR   = f'{FEAT_DIR}/rna/rna_features_brain_full.npy'
X_tr_rna = np.load(RNA_TR).astype(np.float32)
X_te_rna = np.load(RNA_BR).astype(np.float32)
print(f"  RNA:    tr={X_tr_rna.shape} (nonzero={(X_tr_rna.sum(1)!=0).mean()*100:.1f}%), "
      f"te={X_te_rna.shape} (nonzero={(X_te_rna.sum(1)!=0).mean()*100:.1f}%)")

# Stream D: Subcellular localization
LOC_TR   = f'{FEAT_DIR}/loc/loc_features_train.npy'
LOC_BR   = f'{FEAT_DIR}/loc/loc_features_brain_full.npy'
X_tr_loc = np.load(LOC_TR).astype(np.float32)
X_te_loc = np.load(LOC_BR).astype(np.float32)
print(f"  LOC:    tr={X_tr_loc.shape} (nonzero={(X_tr_loc.sum(1)!=0).mean()*100:.1f}%), "
      f"te={X_te_loc.shape} (nonzero={(X_te_loc.sum(1)!=0).mean()*100:.1f}%)")

# Novel isoform flag
is_novel = np.array(['transcript' in x.lower() for x in te_iso], dtype=bool)
print(f"\n  Brain: {X_te_esm.shape[0]:,} isoforms "
      f"(novel={is_novel.sum():,}, known={(~is_novel).sum():,})")

ESM_DIM = X_tr_esm.shape[1]   # 640
RNA_DIM = X_te_rna.shape[1]   # 9
LOC_DIM = X_te_loc.shape[1]   # 8


# ── 2. Feature Dropout 레이어 ────────────────────────────────────────────────
class StreamDropout(layers.Layer):
    def __init__(self, p=0.20, **kwargs):
        super().__init__(**kwargs); self.p = p

    def call(self, x, training=False):
        if training:
            mask = tf.cast(
                tf.random.uniform([tf.shape(x)[0], 1]) > self.p, tf.float32)
            return x * mask
        return x

    def get_config(self):
        cfg = super().get_config(); cfg['p'] = self.p; return cfg


# ── 3. 모델 정의 (v16.py와 동일) ──────────────────────────────────────────────
def build_model():
    inp_esm = layers.Input(shape=(ESM_DIM,), name='esm2')
    inp_sd  = layers.Input(shape=(SD_DIM,),  name='splice')
    inp_rna = layers.Input(shape=(RNA_DIM,), name='rna')
    inp_loc = layers.Input(shape=(LOC_DIM,), name='loc')

    # Stream A: ESM-2
    h = layers.Dense(256, activation='relu')(inp_esm)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(128, activation='relu')(h)
    h_esm = layers.Dropout(0.2)(h)

    # Stream B: Splice delta
    sd_drop  = StreamDropout(p=FEAT_DROP_P, name='drop_splice')(inp_sd)
    h_splice = layers.Dense(32, activation='relu',
                             use_bias=False, name='enc_splice')(sd_drop)
    h_splice = layers.Dropout(0.1)(h_splice)

    # Stream C: RNA stability
    rna_drop = StreamDropout(p=FEAT_DROP_P, name='drop_rna')(inp_rna)
    h_rna    = layers.Dense(32, activation='relu',
                             use_bias=False, name='enc_rna')(rna_drop)
    h_rna    = layers.Dropout(0.1)(h_rna)

    # Stream D: Subcellular localization
    loc_drop = StreamDropout(p=FEAT_DROP_P, name='drop_loc')(inp_loc)
    h_loc    = layers.Dense(32, activation='relu',
                             use_bias=False, name='enc_loc')(loc_drop)
    h_loc    = layers.Dropout(0.1)(h_loc)

    # Aux aggregation
    h_aux      = layers.Concatenate(name='aux_concat')([h_splice, h_rna, h_loc])
    h_aux_proj = layers.Dense(128, use_bias=False, name='aux_proj')(h_aux)

    # Unified gate
    gate_in = layers.Concatenate()([h_esm, h_aux])
    gate    = layers.Dense(128, activation='sigmoid', name='gate')(gate_in)

    # Gated residual fusion
    gated_aux = layers.Multiply()([gate, h_aux_proj])
    h_fused   = layers.Add(name='fused')([h_esm, gated_aux])

    h_out = layers.Dense(64, activation='relu')(h_fused)
    out   = layers.Dense(1,  activation='sigmoid')(h_out)

    return models.Model([inp_esm, inp_sd, inp_rna, inp_loc], out)


# ── 4. 라벨 로드 ─────────────────────────────────────────────────────────────
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


# ── 5. 훈련 + 평가 ──────────────────────────────────────────────────────────
def train_and_eval(y_tr, y_te, seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    m = build_model()
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
        [X_tr_esm, X_tr_sd, X_tr_rna, X_tr_loc], y_tr,
        epochs=100, batch_size=256,
        validation_split=0.1,
        callbacks=cb, verbose=0,
    )
    preds = m.predict(
        [X_te_esm, X_te_sd, X_te_rna, X_te_loc],
        batch_size=1024, verbose=0,
    ).flatten()

    auprc = average_precision_score(y_te, preds) if y_te.sum() > 0 else 0.0

    # Novel isoform AUPRC
    auprc_novel = 0.0
    if is_novel.sum() > 0 and y_te[is_novel].sum() > 0:
        auprc_novel = average_precision_score(y_te[is_novel], preds[is_novel])

    # Gate stats
    gate_model = models.Model(m.inputs, m.get_layer('gate').output)
    gate_vals  = gate_model.predict(
        [X_te_esm, X_te_sd, X_te_rna, X_te_loc],
        batch_size=1024, verbose=0,
    )
    # Aux coverage flag: isoforms with any non-zero RNA or LOC
    has_aux = ((X_te_rna.sum(1) != 0) | (X_te_loc.sum(1) != 0))
    gate_mean = gate_vals.mean(axis=1)
    gate_aux = gate_mean[has_aux].mean() if has_aux.any() else np.nan
    gate_noaux = gate_mean[~has_aux].mean() if (~has_aux).any() else np.nan

    return preds, auprc, auprc_novel, gate_aux, gate_noaux


# ── 6. 전체 GO term 루프 ─────────────────────────────────────────────────────
print(f"\n[2] Training v16 on muscle, evaluating on brain ({N_SEEDS} seeds × {N_GO} GO)...")
score_matrix = np.zeros((len(te_iso), N_GO), dtype=np.float32)
auprc_rows   = []
t0 = time.time()

for j, (go, name) in enumerate(GO_TERMS.items()):
    y_tr, y_te = load_labels(go)
    n_pos_tr   = int(y_tr.sum())
    n_pos_te   = int(y_te.sum())
    print(f"\n  [{j+1:2d}/{N_GO}] {go} ({name}) | tr_pos={n_pos_tr}, br_pos={n_pos_te}",
          flush=True)

    if n_pos_tr < 5:
        print("         SKIP (< 5 train positives)")
        continue

    seed_preds   = []
    seed_auprcs  = []
    seed_novels  = []
    seed_gaux    = []
    seed_gnoaux  = []

    for seed in range(N_SEEDS):
        preds, auprc, auprc_nov, g_aux, g_noaux = train_and_eval(y_tr, y_te, seed)
        seed_preds.append(preds)
        seed_auprcs.append(auprc)
        seed_novels.append(auprc_nov)
        seed_gaux.append(g_aux)
        seed_gnoaux.append(g_noaux)
        print(f"         seed={seed} AUPRC={auprc:.4f} novel={auprc_nov:.4f} "
              f"gate_aux={g_aux:.3f} gate_noaux={g_noaux:.3f}", flush=True)

    ens_preds   = np.mean(seed_preds, axis=0)
    ens_auprc   = average_precision_score(y_te, ens_preds) if n_pos_te > 0 else 0.0
    ens_novel   = (average_precision_score(y_te[is_novel], ens_preds[is_novel])
                   if (is_novel.sum() > 0 and y_te[is_novel].sum() > 0) else None)
    score_matrix[:, j] = ens_preds

    avg_g_aux   = float(np.nanmean(seed_gaux))
    avg_g_noaux = float(np.nanmean(seed_gnoaux))

    nov_str = f"{ens_novel:.4f}" if ens_novel is not None else "  N/A "
    print(f"         Ensemble: AUPRC={ens_auprc:.4f} novel={nov_str} "
          f"gate_aux={avg_g_aux:.3f} gate_noaux={avg_g_noaux:.3f}", flush=True)

    auprc_rows.append({
        'go': go, 'name': name,
        'auprc_all': ens_auprc, 'auprc_novel': ens_novel,
        'gate_aux': avg_g_aux, 'gate_noaux': avg_g_noaux,
        'n_pos_tr': n_pos_tr, 'n_pos_brain': n_pos_te,
    })


# ── 7. 결과 요약 ─────────────────────────────────────────────────────────────
elapsed = time.time() - t0

def safe_mean(lst): return float(np.nanmean([x for x in lst if x is not None])) if lst else 0.0

macro_all   = safe_mean([r['auprc_all']   for r in auprc_rows])
macro_novel = safe_mean([r['auprc_novel'] for r in auprc_rows])
mean_gaux   = safe_mean([r['gate_aux']    for r in auprc_rows])
mean_gnoaux = safe_mean([r['gate_noaux']  for r in auprc_rows])

print(f"\n{'='*70}")
print(f"  v16 BRAIN ZERO-SHOT  (elapsed: {elapsed/60:.1f} min)")
print(f"{'='*70}")
print(f"\n  {'Metric':<30} {'Baseline (v15d)':>16} {'v16':>10} {'Δ':>8}")
print(f"  {'-'*66}")
print(f"  {'Macro AUPRC (ALL brain)':<30} {BASELINE_AUPRC_BR:>16.4f} "
      f"{macro_all:>10.4f} {macro_all-BASELINE_AUPRC_BR:>+8.4f}")
print(f"  {'Macro AUPRC (Novel iso)':<30} {'  --  ':>16} {macro_novel:>10.4f} {'':>8}")
print(f"\n  Gate activation (brain):")
print(f"    With RNA/LOC (46.5%):    {mean_gaux:.4f}")
print(f"    Without RNA/LOC (53.5%): {mean_gnoaux:.4f}")
print(f"    Ratio (aux/noaux):       {mean_gaux/(mean_gnoaux+1e-8):.3f}x")
print(f"\n  REGRESSION CHECK: {'PASS' if macro_all >= BASELINE_AUPRC_BR - 0.005 else 'FAIL'}")
print(f"    (required ≥ {BASELINE_AUPRC_BR - 0.005:.4f}, got {macro_all:.4f})")

print(f"\n  Per-term breakdown:")
print(f"  {'GO term':<12} {'Name':<25} {'AUPRC':>8} {'Δ baseline':>10} {'Novel':>8}")
print(f"  {'-'*65}")
for r in auprc_rows:
    nov = f"{r['auprc_novel']:.4f}" if r['auprc_novel'] is not None else "  N/A"
    delta = r['auprc_all'] - BASELINE_AUPRC_BR
    print(f"  {r['go']:<12} {r['name']:<25} {r['auprc_all']:>8.4f} "
          f"  {delta:>+8.4f}  {nov:>8}")

# ── 8. 저장 ─────────────────────────────────────────────────────────────────
ts = time.strftime('%Y%m%d_%H%M')
np.save(f'{OUT_DIR}/brain_score_matrix_v16_{ts}.npy', score_matrix)

meta = {
    'model': 'v16_brain_eval',
    'timestamp': ts,
    'streams': {'A_esm2': ESM_DIM, 'B_splice': SD_DIM, 'C_rna': RNA_DIM, 'D_loc': LOC_DIM},
    'brain_aux_coverage': float((X_te_rna.sum(1) != 0).mean()),
    'feat_drop_p': FEAT_DROP_P,
    'baseline_brain_auprc': BASELINE_AUPRC_BR,
    'macro_auprc_all': macro_all,
    'macro_auprc_novel': macro_novel,
    'mean_gate_aux': mean_gaux,
    'mean_gate_noaux': mean_gnoaux,
    'gate_ratio': float(mean_gaux / (mean_gnoaux + 1e-8)),
    'n_brain_isoforms': int(X_te_esm.shape[0]),
    'n_novel': int(is_novel.sum()),
    'per_go': auprc_rows,
    'elapsed_min': elapsed / 60,
}
with open(f'{OUT_DIR}/v16_brain_meta_{ts}.json', 'w') as f:
    json.dump(meta, f, indent=2, default=str)

print(f"\n  Score matrix → {OUT_DIR}/brain_score_matrix_v16_{ts}.npy")
print(f"  Meta         → {OUT_DIR}/v16_brain_meta_{ts}.json")
print("DONE")
