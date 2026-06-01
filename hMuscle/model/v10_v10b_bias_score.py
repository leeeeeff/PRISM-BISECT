"""
v10_v10b_bias_score.py — v10-B bias_score 측정
================================================
목적: XGB(0.026) vs LR(0.080) 비교를 완성하기 위해 v10-B의
      bias_score = within_gene_score_std / global_score_std 측정.

예상:
  v10-B bias_score > LR bias_score (0.080) → isoform-specific 예측
  → "동일 AUPRC에서 v10-B가 isoform-level 분화를 더 잘 포착" 주장 확립

출력:
  reports/xgb_baseline/v10b_bias_score_{ts}.json
"""

import os, sys, json, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        tf.config.set_visible_devices(gpus[1] if len(gpus) > 1 else gpus[0], 'GPU')
    except RuntimeError:
        pass

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
ts = time.strftime('%Y%m%d_%H%M')
OUT_DIR   = '../../reports/xgb_baseline'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS   = [42, 123, 456, 789, 2024]
N_BOOT  = 500
SEED    = 42
np.random.seed(SEED)

GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0030017': 'Sarcomere org',
    'GO:0006941': 'Muscle contraction',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0003774': 'Motor activity',
    'GO:0006096': 'Glycolysis',
}

# Known comparison values
XGB_BIAS = {
    'GO:0007204': 0.0155, 'GO:0030017': 0.0230, 'GO:0006941': 0.0163,
    'GO:0006914': 0.0258, 'GO:0043161': 0.0452, 'GO:0007519': 0.0125,
    'GO:0042692': 0.0211, 'GO:0055074': 0.0308, 'GO:0007005': 0.0685,
    'GO:0007517': 0.0299, 'GO:0032006': 0.0182, 'GO:0003774': 0.0130,
    'GO:0006096': None,
}
LR_BIAS = {
    'GO:0007204': 0.0747, 'GO:0030017': 0.0618, 'GO:0006941': 0.0665,
    'GO:0006914': 0.0948, 'GO:0043161': 0.1098, 'GO:0007519': 0.0491,
    'GO:0042692': 0.1046, 'GO:0055074': 0.1168, 'GO:0007005': 0.1600,
    'GO:0007517': 0.0966, 'GO:0032006': 0.0542, 'GO:0003774': 0.0325,
    'GO:0006096': None,
}

print("=" * 65)
print(" v10-B bias_score measurement (5-seed ensemble)")
print("=" * 65)

# Load data
def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

X_tr_esm2 = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te_esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
te_isoid  = load_ids('my_isoform_list_fixed.npy')
te_geneid = load_ids('my_gene_list_fixed.npy')
tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

te_sym      = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_geneid]
te_genebase = [g.split('.')[0] for g in te_geneid]
tr_sym      = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_geneid]
te_genes    = np.array(te_genebase)

print(f"  Train: {len(tr_geneid)}, Test: {len(te_isoid)}")

def load_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te

def bootstrap_ci(y_true, y_score, gene_ids, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    unique_genes = np.unique(gene_ids)
    base = float(average_precision_score(y_true, y_score))
    g2i = defaultdict(list)
    for i, g in enumerate(gene_ids):
        g2i[g].append(i)
    g2i = {g: np.array(idxs) for g, idxs in g2i.items()}
    boot = []
    for _ in range(n_boot):
        gs = rng.choice(unique_genes, size=len(unique_genes), replace=True)
        idx = np.concatenate([g2i[g] for g in gs])
        if y_true[idx].sum() == 0: continue
        boot.append(average_precision_score(y_true[idx], y_score[idx]))
    boot = np.array(boot)
    return base, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def compute_bias_score(scores, gene_ids):
    global_std = scores.std()
    if global_std < 1e-10:
        return np.nan
    g2i = defaultdict(list)
    for i, g in enumerate(gene_ids):
        g2i[g].append(i)
    within_stds = [scores[idxs].std() for idxs in g2i.values() if len(idxs) >= 2]
    if not within_stds:
        return np.nan
    return float(np.mean(within_stds) / global_std)

# v10-B architecture
def build_v10B():
    inp = layers.Input(shape=(640,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out)

def get_cw(y):
    n_pos = max(int(y.sum()), 1)
    return {0: 1.0, 1: int((y == 0).sum()) / n_pos}

def train_v10b_seed(X_tr_sc, y_tr, X_te_sc, seed):
    K.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    n_val = max(int(len(y_tr) * 0.1), 100)
    vi = rng.choice(len(y_tr), n_val, replace=False)
    ti = np.setdiff1d(np.arange(len(y_tr)), vi)
    model = build_v10B()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True, verbose=0),
          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=5, verbose=0)]
    model.fit(X_tr_sc[ti], y_tr[ti],
              validation_data=(X_tr_sc[vi], y_tr[vi]),
              epochs=80, batch_size=512, class_weight=get_cw(y_tr),
              callbacks=cb, verbose=0)
    return model.predict(X_te_sc, verbose=0).ravel()

# Main loop
print(f"\n{'GO Term':<20} {'BiasV10B':>9} {'BiasXGB':>9} {'BiasLR':>9} "
      f"{'AUPRC':>8} {'Interpret':>15}")
print("-" * 76)

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr_esm2)
X_te_sc = sc.transform(X_te_esm2)

all_results = {}
v10b_bias_vals = []
xgb_bias_vals  = []
lr_bias_vals   = []

for go, go_name in GO_TERMS.items():
    y_tr, y_te = load_labels(go)
    if y_tr.sum() < 10 or y_te.sum() < 5:
        continue

    t0 = time.time()
    # 5-seed ensemble
    seed_probs = []
    for seed in SEEDS:
        probs = train_v10b_seed(X_tr_sc, y_tr, X_te_sc, seed)
        seed_probs.append(probs)
    v10b_scores = np.mean(seed_probs, axis=0)

    auprc, ci_lo, ci_hi = bootstrap_ci(y_te, v10b_scores, te_genes)
    bias_v10b = compute_bias_score(v10b_scores, te_genebase)

    xgb_b = XGB_BIAS.get(go)
    lr_b  = LR_BIAS.get(go)

    v10b_is_best = (bias_v10b > (xgb_b or 0)) and (bias_v10b > (lr_b or 0))
    interpret = "ISOFORM-SPEC" if bias_v10b > 0.15 else \
                ("MIXED" if bias_v10b > 0.08 else "GENE-LEVEL")

    xgb_s = f"{xgb_b:.4f}" if xgb_b is not None else "  n/a"
    lr_s  = f"{lr_b:.4f}"  if lr_b  is not None else "  n/a"
    print(f"{go_name:<20} {bias_v10b:>9.4f} {xgb_s:>9} {lr_s:>9} "
          f"{auprc:>8.4f} {interpret:>15}   ({time.time()-t0:.0f}s)")

    v10b_bias_vals.append(bias_v10b)
    if xgb_b: xgb_bias_vals.append(xgb_b)
    if lr_b:  lr_bias_vals.append(lr_b)

    all_results[go] = {
        'go_name': go_name,
        'bias_v10b': round(bias_v10b, 4),
        'bias_xgb': xgb_b,
        'bias_lr': lr_b,
        'v10b_auprc': round(auprc, 4),
        'v10b_ci': [round(ci_lo, 4), round(ci_hi, 4)],
        'v10b_bias_interpretation': interpret,
    }

print("-" * 76)
print(f"{'MEAN':<20} {np.mean(v10b_bias_vals):>9.4f} "
      f"{np.mean(xgb_bias_vals):>9.4f} {np.mean(lr_bias_vals):>9.4f}")

print(f"\n  v10-B bias: {np.mean(v10b_bias_vals):.4f}  "
      f"XGB bias: {np.mean(xgb_bias_vals):.4f}  "
      f"LR bias: {np.mean(lr_bias_vals):.4f}")

v10b_vs_xgb = np.mean(v10b_bias_vals) / np.mean(xgb_bias_vals)
v10b_vs_lr  = np.mean(v10b_bias_vals) / np.mean(lr_bias_vals)
print(f"  v10-B / XGB bias ratio: {v10b_vs_xgb:.2f}×")
print(f"  v10-B / LR  bias ratio: {v10b_vs_lr:.2f}×")

out_path = f'{OUT_DIR}/v10b_bias_score_{ts}.json'
with open(out_path, 'w') as f:
    json.dump({
        'results': all_results,
        'timestamp': ts,
        'summary': {
            'mean_bias_v10b': round(float(np.mean(v10b_bias_vals)), 4),
            'mean_bias_xgb':  round(float(np.mean(xgb_bias_vals)), 4),
            'mean_bias_lr':   round(float(np.mean(lr_bias_vals)), 4),
            'v10b_vs_xgb_ratio': round(float(v10b_vs_xgb), 2),
            'v10b_vs_lr_ratio':  round(float(v10b_vs_lr), 2),
        }
    }, f, indent=2)
print(f"\nFINAL: {out_path}")
