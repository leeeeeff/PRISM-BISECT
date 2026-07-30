# -*- coding: utf-8 -*-
"""
v16.py — 4-Stream Gated Residual Fusion
=========================================

Architecture:
  Stream A  ESM-2 (640)    Dense(256)→BN→Drop(0.3)→Dense(128)→Drop(0.2) → h_esm(128)
  Stream B  Splice (150)   Dense(32, use_bias=False)→Drop(0.1)            → h_splice(32)
  Stream C  RNA (9)        Dense(32, use_bias=False)→Drop(0.1)            → h_rna(32)
  Stream D  LOC (8)        Dense(32, use_bias=False)→Drop(0.1)            → h_loc(32)

  h_aux      = Concat([h_splice, h_rna, h_loc])              # (96,)
  h_aux_proj = Dense(128, use_bias=False)(h_aux)             # (128,)
  gate       = sigmoid(Dense(128)(Concat([h_esm, h_aux])))   # (128,)
  h_fused    = h_esm + gate * h_aux_proj                     # residual addition

  Head: Dense(64, ReLU) → Dense(1, sigmoid)
  Loss: BinaryFocalCrossentropy(γ=2.0) [R1.1]

Key design:
  - use_bias=False in B/C/D: zero input → zero output → h_fused = h_esm (v15d fallback)
  - Feature Dropout (p=0.20 per stream): handles 21.3% RNA / 11.5% LOC train miss
  - Unified gate: ESM-2 + all auxiliary context → single gating decision

v15d_splice 실패 원인 해결:
  - Train splice_delta = zeros 문제: Feature Dropout으로 학습 중에도 비활성화 경험
  - C/D train coverage < 100%: Feature Dropout으로 graceful degradation 학습

기준선: v15d macro AUPRC = 0.7022
목표:   ALL ≥ 0.70, DIFF_SPLICE subset 개선, pos_bias 개선

실행:
  cd /home/welcome1/sw1686/DIFFUSE/hMuscle/model
  conda activate isoform_env
  nohup python v16.py > ../../logs_isoform/v16_$(date +%Y%m%d_%H%M).log 2>&1 &
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

# ── 경로 ─────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v16'
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS        = 5
BASELINE_AUPRC = 0.7022
FEAT_DROP_P    = 0.20   # Feature Dropout: 각 auxiliary stream 비활성화 확률

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
print("  v16 — 4-Stream Gated Residual Fusion")
print(f"  Streams: ESM-2(640) + Splice(150) + RNA(9) + LOC(8)")
print(f"  Feature Dropout p={FEAT_DROP_P} | Baseline AUPRC={BASELINE_AUPRC}")
print("=" * 70)

# ── 1. 모든 feature 로드 ──────────────────────────────────────────────────────
print("\n[1] Loading features...")

te_iso  = load_ids('my_isoform_list_fixed.npy')
te_gene = load_ids('my_gene_list_fixed.npy')
tr_gene = load_ids(f'{ID_DIR}/train_gene_list.npy')

# Stream A: ESM-2
X_tr_esm = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te_esm = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
print(f"  ESM-2: tr={X_tr_esm.shape}, te={X_te_esm.shape}")

# Stream B: Splice delta
SD_PATH       = f'{FEAT_DIR}/splicing/splicing_delta_v2.npy'
TRAIN_SD_PATH = f'{FEAT_DIR}/splicing/train_splicing_delta.npy'
X_te_sd = np.load(SD_PATH).astype(np.float32)
X_tr_sd = (np.load(TRAIN_SD_PATH).astype(np.float32)
           if os.path.exists(TRAIN_SD_PATH)
           else np.zeros((X_tr_esm.shape[0], X_te_sd.shape[1]), np.float32))
print(f"  Splice: tr={X_tr_sd.shape} (nonzero={( X_tr_sd.sum(1)!=0).mean()*100:.1f}%), "
      f"te={X_te_sd.shape} (nonzero={(X_te_sd.sum(1)!=0).mean()*100:.1f}%)")

# Stream C: RNA stability
RNA_TE = f'{FEAT_DIR}/rna/rna_features_test.npy'
RNA_TR = f'{FEAT_DIR}/rna/rna_features_train.npy'
X_te_rna = np.load(RNA_TE).astype(np.float32)
X_tr_rna = np.load(RNA_TR).astype(np.float32)
print(f"  RNA:    tr={X_tr_rna.shape} (nonzero={(X_tr_rna.sum(1)!=0).mean()*100:.1f}%), "
      f"te={X_te_rna.shape} (nonzero={(X_te_rna.sum(1)!=0).mean()*100:.1f}%)")

# Stream D: Subcellular localization
LOC_TE = f'{FEAT_DIR}/loc/loc_features_test.npy'
LOC_TR = f'{FEAT_DIR}/loc/loc_features_train.npy'
X_te_loc = np.load(LOC_TE).astype(np.float32)
X_tr_loc = np.load(LOC_TR).astype(np.float32)
print(f"  LOC:    tr={X_tr_loc.shape} (nonzero={(X_tr_loc.sum(1)!=0).mean()*100:.1f}%), "
      f"te={X_te_loc.shape} (nonzero={(X_te_loc.sum(1)!=0).mean()*100:.1f}%)")

ESM_DIM  = X_tr_esm.shape[1]   # 640
SD_DIM   = X_te_sd.shape[1]    # 150
RNA_DIM  = X_te_rna.shape[1]   # 9
LOC_DIM  = X_te_loc.shape[1]   # 8

# Symbol mapping
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
te_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_gene]


# ── 2. DIFF_SPLICE 서브셋 구축 ────────────────────────────────────────────────
print("\n[2] Building DIFF_SPLICE subsets...")
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
        if not np.allclose(deltas, deltas[0]):
            diff_splice_genes.append(g)
    else:
        normal_genes.append(g)

def genes_to_idx(gl):
    out = []
    for g in gl:
        out.extend(g2idx_te.get(g, []))
    return sorted(set(out))

ds_idx  = genes_to_idx(diff_splice_genes)
nm_idx  = genes_to_idx(normal_genes)

print(f"  DIFF_SPLICE: {len(diff_splice_genes)} genes ({len(ds_idx)} isoforms)")
print(f"  Normal:      {len(normal_genes)} genes ({len(nm_idx)} isoforms)")


# ── 3. Feature Dropout 레이어 ─────────────────────────────────────────────────
class StreamDropout(layers.Layer):
    """
    Feature Dropout: 훈련 중 전체 스트림을 확률 p로 0으로 마스킹.
    추론 시 no-op.
    목적: train coverage < 100%인 스트림에 대해 모델이 '없는 경우'를 학습하도록.
    """
    def __init__(self, p=0.20, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def call(self, x, training=False):
        if training:
            mask = tf.cast(
                tf.random.uniform([tf.shape(x)[0], 1]) > self.p,
                tf.float32
            )
            return x * mask
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg['p'] = self.p
        return cfg


# ── 4. 모델 정의 ──────────────────────────────────────────────────────────────
def build_model(esm_dim, sd_dim, rna_dim, loc_dim):
    """
    4-Stream Gated Residual Fusion.

    use_bias=False guarantee:
      auxiliary input = 0 → h = 0 → h_aux_proj = 0 → h_fused = h_esm
      (exact v15d behavior for missing streams)
    """
    inp_esm = layers.Input(shape=(esm_dim,), name='esm2')
    inp_sd  = layers.Input(shape=(sd_dim,),  name='splice')
    inp_rna = layers.Input(shape=(rna_dim,), name='rna')
    inp_loc = layers.Input(shape=(loc_dim,), name='loc')

    # ── Stream A: ESM-2 (v15d identical) ──────────────────────────────────
    h = layers.Dense(256, activation='relu')(inp_esm)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(128, activation='relu')(h)
    h_esm = layers.Dropout(0.2)(h)                          # (128,)

    # ── Stream B: Splice delta ─────────────────────────────────────────────
    sd_drop = StreamDropout(p=FEAT_DROP_P, name='drop_splice')(inp_sd)
    h_splice = layers.Dense(32, activation='relu',
                            use_bias=False, name='enc_splice')(sd_drop)  # (32,)
    h_splice = layers.Dropout(0.1)(h_splice)

    # ── Stream C: RNA stability ────────────────────────────────────────────
    rna_drop = StreamDropout(p=FEAT_DROP_P, name='drop_rna')(inp_rna)
    h_rna = layers.Dense(32, activation='relu',
                         use_bias=False, name='enc_rna')(rna_drop)       # (32,)
    h_rna = layers.Dropout(0.1)(h_rna)

    # ── Stream D: Subcellular localization ────────────────────────────────
    loc_drop = StreamDropout(p=FEAT_DROP_P, name='drop_loc')(inp_loc)
    h_loc = layers.Dense(32, activation='relu',
                         use_bias=False, name='enc_loc')(loc_drop)       # (32,)
    h_loc = layers.Dropout(0.1)(h_loc)

    # ── Auxiliary aggregation ─────────────────────────────────────────────
    h_aux      = layers.Concatenate(name='aux_concat')([h_splice, h_rna, h_loc])  # (96,)
    h_aux_proj = layers.Dense(128, use_bias=False,
                              name='aux_proj')(h_aux)                               # (128,)

    # ── Unified gate ──────────────────────────────────────────────────────
    gate_in = layers.Concatenate()([h_esm, h_aux])                       # (224,)
    gate    = layers.Dense(128, activation='sigmoid',
                           name='gate')(gate_in)                          # (128,)

    # ── Gated residual fusion ─────────────────────────────────────────────
    gated_aux = layers.Multiply()([gate, h_aux_proj])                    # (128,)
    h_fused   = layers.Add(name='fused')([h_esm, gated_aux])            # (128,)

    # ── Output head ───────────────────────────────────────────────────────
    h_out = layers.Dense(64, activation='relu')(h_fused)
    out   = layers.Dense(1,  activation='sigmoid')(h_out)

    return models.Model([inp_esm, inp_sd, inp_rna, inp_loc], out)


# ── 5. 라벨 로드 ──────────────────────────────────────────────────────────────
def load_labels(go_term):
    tr_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_gene]
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations_unified_bp.txt') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) > 1 and go_term in p[1:]:
                pos.add(p[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te


# ── 6. pos_bias 계산 ─────────────────────────────────────────────────────────
def compute_pos_bias(preds, labels, gene_list, subset_idx=None):
    if subset_idx is not None:
        preds  = preds[subset_idx]
        labels = labels[subset_idx]
        genes  = [gene_list[i] for i in subset_idx]
    else:
        genes = gene_list
    eps = 1e-10
    global_std = preds.std() + eps
    df = pd.DataFrame({'gene': genes, 'score': preds, 'label': labels})
    multi  = df.groupby('gene').filter(lambda g: len(g) >= 2)
    pos_g  = df[df['label'] == 1]['gene'].unique()
    pos_multi = multi[multi['gene'].isin(pos_g)]
    if len(pos_multi) == 0:
        return np.nan
    within_stds = pos_multi.groupby('gene')['score'].std().dropna()
    return float(within_stds.mean() / global_std)


# ── 7. Gate 활성화 수집 ───────────────────────────────────────────────────────
def get_gate_stats(m, inputs, ds_idx, nm_idx):
    gate_model = models.Model(m.inputs, m.get_layer('gate').output)
    gate_vals  = gate_model.predict(inputs, batch_size=1024, verbose=0)  # (N, 128)
    gate_mean  = gate_vals.mean(axis=1)
    gate_ds    = gate_mean[ds_idx].mean() if ds_idx else np.nan
    gate_nm    = gate_mean[nm_idx].mean() if nm_idx else np.nan
    return gate_ds, gate_nm


# ── 8. 훈련 + 평가 ───────────────────────────────────────────────────────────
def train_and_eval(y_tr, y_te, seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    m = build_model(ESM_DIM, SD_DIM, RNA_DIM, LOC_DIM)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    )
    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                monitor='val_loss'),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor='val_loss'),
    ]
    tr_inputs = [X_tr_esm, X_tr_sd, X_tr_rna, X_tr_loc]
    te_inputs = [X_te_esm, X_te_sd, X_te_rna, X_te_loc]

    m.fit(
        tr_inputs, y_tr,
        epochs=100, batch_size=256,
        validation_split=0.1,
        callbacks=cb, verbose=0,
    )
    preds    = m.predict(te_inputs, batch_size=1024, verbose=0).flatten()
    auprc    = average_precision_score(y_te, preds) if y_te.sum() > 0 else 0.0
    g_ds, g_nm = get_gate_stats(m, te_inputs, ds_idx, nm_idx)

    return preds, auprc, g_ds, g_nm


# ── 9. 전체 GO term 루프 ─────────────────────────────────────────────────────
print(f"\n[3] Training v16 ({N_SEEDS} seeds × {N_GO} GO terms)...")
score_matrix = np.zeros((len(te_iso), N_GO), dtype=np.float32)
auprc_rows   = []
t0 = time.time()

BASELINE_DS = 0.8078
BASELINE_NM = 0.6481

for j, (go, name) in enumerate(GO_TERMS.items()):
    y_tr, y_te = load_labels(go)
    n_pos_tr   = int(y_tr.sum())
    n_pos_te   = int(y_te.sum())
    print(f"\n  [{j+1:2d}/{N_GO}] {go} ({name}) | tr_pos={n_pos_tr}, te_pos={n_pos_te}",
          flush=True)

    if n_pos_tr < 5:
        print("         SKIP (< 5 train positives)")
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

    ens_preds = np.mean(seed_preds, axis=0)
    ens_auprc = average_precision_score(y_te, ens_preds) if n_pos_te > 0 else 0.0
    score_matrix[:, j] = ens_preds

    y_ds = y_te[ds_idx]; p_ds = ens_preds[ds_idx]
    y_nm = y_te[nm_idx]; p_nm = ens_preds[nm_idx]
    auprc_ds = average_precision_score(y_ds, p_ds) if y_ds.sum() > 0 else None
    auprc_nm = average_precision_score(y_nm, p_nm) if y_nm.sum() > 0 else None

    pb_ds = compute_pos_bias(ens_preds, y_te, te_gene, subset_idx=ds_idx)
    pb_nm = compute_pos_bias(ens_preds, y_te, te_gene, subset_idx=nm_idx)
    avg_gate_ds = float(np.nanmean(seed_gate_ds))
    avg_gate_nm = float(np.nanmean(seed_gate_nm))

    ds_str = f"{auprc_ds:.4f}" if auprc_ds is not None else "  N/A "
    nm_str = f"{auprc_nm:.4f}" if auprc_nm is not None else "  N/A "
    print(f"         Ensemble: AUPRC={ens_auprc:.4f} "
          f"DS={ds_str} NM={nm_str} "
          f"pb_DS={pb_ds:.4f} pb_NM={pb_nm:.4f} "
          f"gate DS={avg_gate_ds:.3f} NM={avg_gate_nm:.3f}")

    auprc_rows.append({
        'go': go, 'name': name,
        'auprc_all': ens_auprc, 'auprc_ds': auprc_ds, 'auprc_nm': auprc_nm,
        'pos_bias_ds': pb_ds, 'pos_bias_nm': pb_nm,
        'gate_ds': avg_gate_ds, 'gate_nm': avg_gate_nm,
        'n_pos_tr': n_pos_tr, 'n_pos_te': n_pos_te,
    })


# ── 10. 결과 요약 ─────────────────────────────────────────────────────────────
elapsed = time.time() - t0

def safe_mean(lst): return float(np.nanmean(lst)) if lst else 0.0

macro_all  = safe_mean([r['auprc_all'] for r in auprc_rows if r['auprc_all'] is not None])
macro_ds   = safe_mean([r['auprc_ds']  for r in auprc_rows if r['auprc_ds']  is not None])
macro_nm   = safe_mean([r['auprc_nm']  for r in auprc_rows if r['auprc_nm']  is not None])
mean_pb_ds = safe_mean([r['pos_bias_ds'] for r in auprc_rows if r['pos_bias_ds'] and not np.isnan(r['pos_bias_ds'])])
mean_pb_nm = safe_mean([r['pos_bias_nm'] for r in auprc_rows if r['pos_bias_nm'] and not np.isnan(r['pos_bias_nm'])])
mean_gds   = safe_mean([r['gate_ds']  for r in auprc_rows])
mean_gnm   = safe_mean([r['gate_nm']  for r in auprc_rows])

print(f"\n{'='*70}")
print(f"  v16 RESULTS  (elapsed: {elapsed/60:.1f} min)")
print(f"{'='*70}")
print(f"\n  {'Metric':<30} {'Baseline (v15d)':>16} {'v16':>10} {'Δ':>8}")
print(f"  {'-'*66}")
print(f"  {'Macro AUPRC (ALL)':<30} {BASELINE_AUPRC:>16.4f} {macro_all:>10.4f} {macro_all-BASELINE_AUPRC:>+8.4f}")
print(f"  {'Macro AUPRC (DIFF_SPLICE)':<30} {BASELINE_DS:>16.4f} {macro_ds:>10.4f} {macro_ds-BASELINE_DS:>+8.4f}")
print(f"  {'Macro AUPRC (Normal)':<30} {BASELINE_NM:>16.4f} {macro_nm:>10.4f} {macro_nm-BASELINE_NM:>+8.4f}")
print(f"\n  pos_bias DIFF_SPLICE: {mean_pb_ds:.4f}")
print(f"  pos_bias Normal:      {mean_pb_nm:.4f}")
print(f"\n  Gate activation:  DS={mean_gds:.4f}  NM={mean_gnm:.4f}  ratio={mean_gds/(mean_gnm+1e-8):.3f}x")
print(f"\n  REGRESSION CHECK: {'PASS' if macro_all >= BASELINE_AUPRC - 0.005 else 'FAIL'}")
print(f"    (required ≥ {BASELINE_AUPRC - 0.005:.4f}, got {macro_all:.4f})")

# ── 11. 저장 ─────────────────────────────────────────────────────────────────
ts = time.strftime('%Y%m%d_%H%M')
np.save(f'{OUT_DIR}/score_matrix_v16_{ts}.npy', score_matrix)

meta = {
    'model': 'v16',
    'timestamp': ts,
    'streams': {'A_esm2': ESM_DIM, 'B_splice': SD_DIM, 'C_rna': RNA_DIM, 'D_loc': LOC_DIM},
    'feat_drop_p': FEAT_DROP_P,
    'baseline_auprc': BASELINE_AUPRC,
    'macro_auprc_all': macro_all, 'macro_auprc_ds': macro_ds, 'macro_auprc_nm': macro_nm,
    'baseline_ds': BASELINE_DS, 'baseline_nm': BASELINE_NM,
    'mean_pos_bias_ds': mean_pb_ds, 'mean_pos_bias_nm': mean_pb_nm,
    'mean_gate_ds': mean_gds, 'mean_gate_nm': mean_gnm,
    'gate_ratio_ds_nm': float(mean_gds / (mean_gnm + 1e-8)),
    'n_diff_splice_genes': len(diff_splice_genes), 'n_normal_genes': len(normal_genes),
    'per_go': auprc_rows,
    'elapsed_min': elapsed / 60,
}
with open(f'{OUT_DIR}/v16_meta_{ts}.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\n  Score matrix → {OUT_DIR}/score_matrix_v16_{ts}.npy")
print(f"  Meta         → {OUT_DIR}/v16_meta_{ts}.json")
print(f"{'='*70}")
