"""
v10_typeAB_classifier.py — GO term Type-A/B 자동 분류기 (M1)
=============================================================
sep_ratio = inter_dist / intra_dist (ESM-2 test embedding 기반)

Type-A (sep_ratio >= threshold): gene-level 지배 → LR 권고
Type-B (sep_ratio <  threshold): isoform-discriminative → v10-B 권고

출력:
  1. sep_ratio per GO term → Type-A/B 분류
  2. Decision threshold 최적화 (기존 5 GO term 결과 기반)
  3. 신규 GO term 입력 시 타입 예측 함수
  4. 논문용 Figure 데이터 (sep_ratio vs Performance Gap)
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/typeAB_classifier'
os.makedirs(OUT_DIR, exist_ok=True)

GO_TERMS = ['GO:0006096', 'GO:0003774', 'GO:0007204', 'GO:0030017', 'GO:0006941']

# ─── Load features ────────────────────────────────────────────────────────────
print("=" * 70)
print("  Loading features ...")
print("=" * 70)

X_tr_esm2 = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te_esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

X_tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')
X_te_geneid = load_ids('my_gene_list_fixed.npy')
X_te_isoid  = load_ids('my_isoform_list_fixed.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

X_te_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in X_te_geneid]
print(f"  Train {X_tr_esm2.shape}, Test {X_te_esm2.shape}")


# ─── Sep ratio computation ────────────────────────────────────────────────────
def compute_sep_ratio(X, y, n_sample=2000, random_state=42):
    """
    sep_ratio = inter_centroid_dist / mean_intra_centroid_dist

    inter_dist: euclidean distance between positive centroid and negative centroid
    intra_dist: mean pairwise distance within positive class (spread measure)

    > 1.15 → positive cluster tight relative to pos-neg separation → Type-A
    < 1.15 → positive cluster dispersed → Type-B
    """
    rng = np.random.RandomState(random_state)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) < 3:
        return np.nan

    # Sub-sample to speed up
    if len(neg_idx) > n_sample:
        neg_idx = rng.choice(neg_idx, n_sample, replace=False)
    if len(pos_idx) > n_sample:
        pos_idx = rng.choice(pos_idx, n_sample, replace=False)

    X_pos = X[pos_idx]
    X_neg = X[neg_idx]

    pos_centroid = X_pos.mean(axis=0)
    neg_centroid = X_neg.mean(axis=0)

    inter_dist = np.linalg.norm(pos_centroid - neg_centroid)

    # Intra: mean pairwise distance in positive cluster
    if len(X_pos) > 500:
        sub = rng.choice(len(X_pos), 500, replace=False)
        X_pos_sub = X_pos[sub]
    else:
        X_pos_sub = X_pos

    dists = pairwise_distances(X_pos_sub, metric='euclidean')
    # Upper triangle only
    mask = np.triu(np.ones(dists.shape, dtype=bool), k=1)
    intra_dist = dists[mask].mean() if mask.sum() > 0 else 1e-10

    return float(inter_dist / (intra_dist + 1e-10))


def compute_sep_ratio_cosine(X, y, n_sample=2000, random_state=42):
    """Alternative: cosine-based sep_ratio (scale-invariant)"""
    rng = np.random.RandomState(random_state)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < 3:
        return np.nan

    if len(neg_idx) > n_sample:
        neg_idx = rng.choice(neg_idx, n_sample, replace=False)
    if len(pos_idx) > n_sample:
        pos_idx = rng.choice(pos_idx, n_sample, replace=False)

    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    pos_centroid = X_norm[pos_idx].mean(axis=0)
    neg_centroid = X_norm[neg_idx].mean(axis=0)

    cos_inter = np.dot(pos_centroid, neg_centroid) / (
        np.linalg.norm(pos_centroid) * np.linalg.norm(neg_centroid) + 1e-10)
    inter_dist = 1 - cos_inter

    X_pos = X_norm[pos_idx]
    if len(X_pos) > 500:
        sub = rng.choice(len(X_pos), 500, replace=False)
        X_pos = X_pos[sub]

    cos_mat = X_pos @ X_pos.T
    mask = np.triu(np.ones(cos_mat.shape, dtype=bool), k=1)
    intra_cos = cos_mat[mask].mean() if mask.sum() > 0 else 0.0
    intra_dist = max(1 - intra_cos, 1e-10)

    return float(inter_dist / intra_dist)


# ─── Model helpers ────────────────────────────────────────────────────────────
def build_v10B():
    inp = layers.Input(shape=(640,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name='v10B')

def get_cw(y):
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    return {0: 1.0, 1: n_neg / max(n_pos, 1)}

def train_v10B(X_tr, X_te, y_tr, epochs=80, batch=512):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)
    K.clear_session(); tf.random.set_seed(SEED)
    model = build_v10B()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True, verbose=0),
          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=5, verbose=0)]
    rng = np.random.RandomState(SEED)
    n_val = max(int(len(y_tr)*0.1), 100)
    vi = rng.choice(len(y_tr), n_val, replace=False)
    ti = np.setdiff1d(np.arange(len(y_tr)), vi)
    model.fit(Xtr[ti], y_tr[ti], validation_data=(Xtr[vi], y_tr[vi]),
              epochs=epochs, batch_size=batch, class_weight=get_cw(y_tr),
              callbacks=cb, verbose=0)
    return model.predict(Xte, verbose=0).ravel()

def train_lr(X_tr, X_te, y_tr):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)
    clf = LogisticRegression(class_weight='balanced', C=1.0,
                              max_iter=2000, random_state=SEED)
    clf.fit(Xtr, y_tr)
    return clf.predict_proba(Xte)[:, 1]

def load_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) > 1 and go_term in p[1:]:
                pos.add(p[0])
    y_te = np.array([1 if s in pos else 0 for s in X_te_sym], dtype=np.float32)
    y_tr = np.array([1 if g in pos else 0 for g in X_tr_geneid], dtype=np.float32)
    return y_tr, y_te


# ─── Known results (bootstrap CI from previous experiments) ──────────────────
# From F37: v10-B vs LR bootstrap results
KNOWN_RESULTS = {
    'GO:0006096': {'v10b': 0.6712, 'lr': 0.6949, 'type': 'A'},
    'GO:0003774': {'v10b': 0.8128, 'lr': 0.8253, 'type': 'A'},
    'GO:0007204': {'v10b': 0.7653, 'lr': 0.4147, 'type': 'B'},
    'GO:0030017': {'v10b': 0.7426, 'lr': 0.5635, 'type': 'B'},
    'GO:0006941': {'v10b': 0.5968, 'lr': 0.3102, 'type': 'B'},
}


# ─── Main analysis ────────────────────────────────────────────────────────────
all_results = []

for go in GO_TERMS:
    print(f"\n{'='*70}\n  {go}\n{'='*70}")
    y_tr, y_te = load_labels(go)
    if y_tr.sum() < 2 or y_te.sum() == 0:
        print("  SKIP"); continue

    n_pos_te = int(y_te.sum())
    n_pos_tr = int(y_tr.sum())
    print(f"  n_pos train={n_pos_tr}, test={n_pos_te}")

    # Sep ratio on test embeddings (ESM-2 representation pre-model)
    sep_eucl = compute_sep_ratio(X_te_esm2, y_te)
    sep_cos  = compute_sep_ratio_cosine(X_te_esm2, y_te)
    print(f"  sep_ratio (euclidean)={sep_eucl:.4f}, (cosine)={sep_cos:.4f}")

    # Sep ratio on train embeddings (for comparison)
    sep_eucl_tr = compute_sep_ratio(X_tr_esm2, y_tr)
    sep_cos_tr  = compute_sep_ratio_cosine(X_tr_esm2, y_tr)
    print(f"  sep_ratio train (eucl)={sep_eucl_tr:.4f}, (cos)={sep_cos_tr:.4f}")

    # Use known results for AUPRC (no need to retrain)
    known = KNOWN_RESULTS.get(go, {})
    v10b_auprc = known.get('v10b', np.nan)
    lr_auprc   = known.get('lr', np.nan)
    true_type  = known.get('type', '?')
    perf_gap   = v10b_auprc - lr_auprc if not np.isnan(v10b_auprc) else np.nan

    print(f"  v10-B={v10b_auprc:.4f}, LR={lr_auprc:.4f}, gap={perf_gap:+.4f}, true_type={true_type}")

    all_results.append({
        'go': go,
        'n_pos_train': n_pos_tr,
        'n_pos_test': n_pos_te,
        'sep_euclidean_test': sep_eucl,
        'sep_cosine_test': sep_cos,
        'sep_euclidean_train': sep_eucl_tr,
        'sep_cosine_train': sep_cos_tr,
        'v10b_auprc': v10b_auprc,
        'lr_auprc': lr_auprc,
        'perf_gap': perf_gap,
        'true_type': true_type,
    })


# ─── Classifier design ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TYPE-A/B CLASSIFIER")
print("=" * 70)

df = pd.DataFrame(all_results)
print(f"\n{'GO Term':<15} {'sep(cos)':>10} {'sep(eucl)':>10} {'v10-B':>8} {'LR':>8} {'Gap':>8} {'Type':>6}")
print("-" * 70)
for _, r in df.iterrows():
    print(f"{r['go']:<15} {r['sep_cosine_test']:>10.4f} {r['sep_euclidean_test']:>10.4f} "
          f"{r['v10b_auprc']:>8.4f} {r['lr_auprc']:>8.4f} "
          f"{r['perf_gap']:>+8.4f} {r['true_type']:>6}")

# Optimal threshold search (cosine sep_ratio)
print("\n  Threshold optimization (cosine sep_ratio, classify A if >= threshold):")
best_threshold = None
best_accuracy = 0
thresholds = np.arange(1.0, 2.5, 0.05)
for thr in thresholds:
    predictions = ['A' if r >= thr else 'B' for r in df['sep_cosine_test']]
    correct = sum(1 for pred, true in zip(predictions, df['true_type']) if pred == true)
    acc = correct / len(df)
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = thr

print(f"  Best threshold: {best_threshold:.2f} (accuracy={best_accuracy:.0%})")

# Apply threshold
df['predicted_type'] = df['sep_cosine_test'].apply(
    lambda r: 'A' if r >= best_threshold else 'B')
df['correct'] = df['predicted_type'] == df['true_type']

print(f"\n  Classification results (cosine threshold={best_threshold:.2f}):")
print(f"{'GO Term':<15} {'sep_cos':>10} {'Predicted':>10} {'True':>8} {'Correct':>8}")
print("-" * 55)
for _, r in df.iterrows():
    mark = '✓' if r['correct'] else '✗'
    print(f"{r['go']:<15} {r['sep_cosine_test']:>10.4f} {r['predicted_type']:>10} "
          f"{r['true_type']:>8} {mark:>8}")

# Euclidean threshold
best_thr_eucl = None
best_acc_eucl = 0
for thr in thresholds:
    predictions = ['A' if r >= thr else 'B' for r in df['sep_euclidean_test']]
    correct = sum(1 for pred, true in zip(predictions, df['true_type']) if pred == true)
    acc = correct / len(df)
    if acc > best_acc_eucl:
        best_acc_eucl = acc
        best_thr_eucl = thr

print(f"\n  Euclidean threshold: {best_thr_eucl:.2f} (accuracy={best_acc_eucl:.0%})")

# ─── Guideline for new GO terms ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PRACTICAL GUIDELINE FOR NEW GO TERMS")
print("=" * 70)
print(f"""
  def classify_go_term(esm2_test, y_test, threshold_cos={best_threshold:.2f}):
      '''
      Classify a new GO term as Type-A (LR preferred) or Type-B (v10-B preferred).

      Args:
          esm2_test: ESM-2 embeddings (n_isoforms, 640)
          y_test: gene-level binary labels (n_isoforms,)
          threshold_cos: cosine sep_ratio threshold (default: {best_threshold:.2f})

      Returns:
          type: 'A' or 'B'
          sep_ratio: float
          recommended_model: 'LR' or 'v10-B'
      '''
      sep = compute_sep_ratio_cosine(esm2_test, y_test)
      if sep >= threshold_cos:
          return 'A', sep, 'LR'
      else:
          return 'B', sep, 'v10-B'

  Decision boundary:
    sep_cosine >= {best_threshold:.2f} → Type-A → Use LR
    sep_cosine <  {best_threshold:.2f} → Type-B → Use v10-B

  Current 5 GO terms:
    GO:0006096: sep={df[df.go=='GO:0006096']['sep_cosine_test'].values[0]:.3f} → {df[df.go=='GO:0006096']['predicted_type'].values[0]} ({'LR' if df[df.go=='GO:0006096']['predicted_type'].values[0]=='A' else 'v10-B'})
    GO:0003774: sep={df[df.go=='GO:0003774']['sep_cosine_test'].values[0]:.3f} → {df[df.go=='GO:0003774']['predicted_type'].values[0]} ({'LR' if df[df.go=='GO:0003774']['predicted_type'].values[0]=='A' else 'v10-B'})
    GO:0007204: sep={df[df.go=='GO:0007204']['sep_cosine_test'].values[0]:.3f} → {df[df.go=='GO:0007204']['predicted_type'].values[0]} ({'LR' if df[df.go=='GO:0007204']['predicted_type'].values[0]=='A' else 'v10-B'})
    GO:0030017: sep={df[df.go=='GO:0030017']['sep_cosine_test'].values[0]:.3f} → {df[df.go=='GO:0030017']['predicted_type'].values[0]} ({'LR' if df[df.go=='GO:0030017']['predicted_type'].values[0]=='A' else 'v10-B'})
    GO:0006941: sep={df[df.go=='GO:0006941']['sep_cosine_test'].values[0]:.3f} → {df[df.go=='GO:0006941']['predicted_type'].values[0]} ({'LR' if df[df.go=='GO:0006941']['predicted_type'].values[0]=='A' else 'v10-B'})
""")

# ─── Correlation analysis ─────────────────────────────────────────────────────
print("  CORRELATION: sep_ratio vs performance gap")
valid = df.dropna(subset=['perf_gap', 'sep_cosine_test'])
if len(valid) >= 3:
    r_cos = np.corrcoef(valid['sep_cosine_test'], valid['perf_gap'])[0,1]
    r_eucl = np.corrcoef(valid['sep_euclidean_test'], valid['perf_gap'])[0,1]
    print(f"  Pearson(sep_cosine, gap) = {r_cos:.4f}")
    print(f"  Pearson(sep_euclidean, gap) = {r_eucl:.4f}")
    print(f"  Interpretation: negative r means high sep → LR better (Type-A)")

# Save results
ts = time.strftime('%Y%m%d_%H%M')
out_path = f'{OUT_DIR}/typeAB_results_{ts}.json'
with open(out_path, 'w') as f:
    json.dump({
        'results': all_results,
        'best_threshold_cosine': float(best_threshold),
        'best_accuracy_cosine': float(best_accuracy),
        'best_threshold_euclidean': float(best_thr_eucl),
        'best_accuracy_euclidean': float(best_acc_eucl),
        'correlation_sep_cos_vs_gap': float(r_cos) if len(valid) >= 3 else None,
        'classification': df[['go','sep_cosine_test','sep_euclidean_test',
                               'predicted_type','true_type','correct']].to_dict('records'),
        'timestamp': ts,
    }, f, indent=2, default=str)
print(f"\n[Saved] {out_path}")
print("\nDone.")
