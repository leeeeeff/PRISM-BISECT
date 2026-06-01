"""
v11_deviation.py — Isoform-Deviation Architecture (v11-B)
==========================================================
v11-A 실패 원인 분석:
  mean context (mean of same-gene isoforms) ≈ gene-level representation
  → gene-level bias 주입 → pos_bias 붕괴 (-66~75%)
  → [R2.1] 원칙 위반 (gene context as shortcut)

v11-B 핵심 아이디어:
  delta_i = x_i - mean(x_j for j in same gene, including self)
  → gene-level 정보 제거 → isoform-specific deviation 인코딩
  → positive isoform은 negative isoform과 다른 delta를 가짐
  → within-gene contrast를 직접 학습

아키텍처:
  inp_abs (640d) = x_i (absolute ESM-2)
  inp_dev (640d) = x_i - gene_mean  [isoform-specific deviation]

  h_abs = Dense(256, relu) → BN → Dropout(0.3)   [v10-B와 동일]
  h_dev = Dense(256, relu)                         [deviation projection]

  gate = sigmoid(Dense(256)([h_abs, h_dev]))       [언제 deviation을 쓸지]
  h = h_abs + gate * h_dev                         [gated integration, [R2.1]]

  Dense(128, relu) → Dropout(0.2) → Dense(64, relu) → Dense(1, sigmoid)

단일 isoform 유전자: delta = 0 → gate ≈ 0 → v10-B와 동일
다중 isoform 유전자: delta = isoform-specific deviation → within-gene contrast 학습

실행:
  /home/welcome1/miniconda3/envs/isoform_env/bin/python -u v11_deviation.py
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_isoform'))

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

# ─── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v11_deviation'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS         = [42, 123, 456, 789, 2024]
N_BOOT        = 500
SEP_THRESHOLD = 0.060

GO_TERMS = [
    ('GO:0007204', 'Ca2+ signaling'),
    ('GO:0030017', 'Sarcomere org'),
    ('GO:0006941', 'Muscle contraction'),
    ('GO:0006914', 'Autophagy'),
    ('GO:0043161', 'Proteasome-UPS'),
    ('GO:0007519', 'Skeletal muscle dev'),
    ('GO:0042692', 'Muscle cell diff'),
    ('GO:0055074', 'Ca2+ homeostasis'),
    ('GO:0007005', 'Mitochondrion org'),
    ('GO:0007517', 'Muscle organ dev'),
    ('GO:0032006', 'TOR signaling'),
    ('GO:0003774', 'Motor activity'),
    ('GO:0006096', 'Glycolysis'),
]

V10B_REF = {
    'GO:0007204': 0.7653, 'GO:0030017': 0.7426, 'GO:0006941': 0.5968,
    'GO:0006914': 0.6397, 'GO:0043161': 0.7174, 'GO:0007519': 0.7250,
    'GO:0042692': 0.6526, 'GO:0055074': 0.7255, 'GO:0007005': 0.6624,
    'GO:0007517': 0.7017, 'GO:0032006': 0.6023, 'GO:0003774': 0.8128,
    'GO:0006096': 0.6712,
}

V10B_POS_BIAS = {
    'GO:0007204': 0.475, 'GO:0030017': 1.176, 'GO:0006941': 1.902,
    'GO:0003774': 1.435, 'GO:0006096': 0.663,
}

print("=" * 70)
print("  v11-B Isoform-Deviation Architecture")
print("=" * 70)
print("  Key: delta_i = x_i - gene_mean  (removes gene-level component)")

# ─── Load data ──────────────────────────────────────────────────────────────────
def load_ids(p):
    a = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]

X_tr = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

tr_genes = load_ids(f'{ID_DIR}/train_gene_list.npy')
te_genes = load_ids('my_gene_list_fixed.npy')
te_isos  = load_ids('my_isoform_list_fixed.npy')

ensg2sym = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ensg2sym[p[0]] = p[4]

te_syms       = [ensg2sym.get(g.split('.')[0], g.split('.')[0]) for g in te_genes]
tr_genes_base = [g.split('.')[0] for g in tr_genes]
te_genes_base = [g.split('.')[0] for g in te_genes]

print(f"  Train {X_tr.shape}, Test {X_te.shape}")


# ─── Gene deviation computation ─────────────────────────────────────────────────
def compute_gene_deviations(X, gene_base_list):
    """
    delta_i = x_i - mean(x_j for j in same gene, INCLUDING self)
    단일 isoform 유전자: delta = 0 (no within-gene contrast)
    """
    N, D = X.shape
    deltas = np.zeros((N, D), dtype=np.float32)

    gene_to_idxs = defaultdict(list)
    for i, g in enumerate(gene_base_list):
        gene_to_idxs[g].append(i)

    n_multi = 0
    for g, idxs in gene_to_idxs.items():
        if len(idxs) < 2:
            continue
        n_multi += len(idxs)
        gene_mean = X[idxs].mean(axis=0)
        for i in idxs:
            deltas[i] = X[i] - gene_mean

    n_unique_genes = len([g for g, idxs in gene_to_idxs.items() if len(idxs) >= 2])
    print(f"    Multi-isoform isoforms: {n_multi}/{N} ({n_multi/N*100:.1f}%) "
          f"from {n_unique_genes} genes")
    # delta statistics for non-zero rows
    nz = np.any(deltas != 0, axis=1)
    print(f"    Delta L2 norm (mean±std): {np.linalg.norm(deltas[nz], axis=1).mean():.3f} "
          f"± {np.linalg.norm(deltas[nz], axis=1).std():.3f}")
    return deltas


print("\n[1] Computing isoform deviations ...")
print("  Train:")
dev_tr = compute_gene_deviations(X_tr, tr_genes_base)
print("  Test:")
dev_te = compute_gene_deviations(X_te, te_genes_base)


# ─── Labels ────────────────────────────────────────────────────────────────────
go_to_genes = defaultdict(set)
with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        for go in parts[1:]:
            go_to_genes[go].add(parts[0])

def load_labels(go_term):
    pos = go_to_genes[go_term]
    y_te = np.array([1 if s in pos else 0 for s in te_syms], dtype=np.float32)
    y_tr = np.array([1 if g in pos else 0 for g in tr_genes], dtype=np.float32)
    return y_tr, y_te


# ─── sep_cosine ─────────────────────────────────────────────────────────────────
def sep_cosine(X, y, rs=42):
    rng = np.random.RandomState(rs)
    pi = np.where(y == 1)[0]; ni = np.where(y == 0)[0]
    if len(pi) < 5:
        return np.nan
    if len(ni) > 2000: ni = rng.choice(ni, 2000, replace=False)
    if len(pi) > 2000: pi = rng.choice(pi, 2000, replace=False)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    pc = Xn[pi].mean(0); nc = Xn[ni].mean(0)
    inter = 1 - np.dot(pc, nc) / (np.linalg.norm(pc) * np.linalg.norm(nc) + 1e-10)
    Xp = Xn[pi[:500]]
    m = Xp @ Xp.T; up = np.triu(np.ones(m.shape, bool), k=1)
    return float(inter / max(1 - m[up].mean(), 1e-10))


# ─── Model builder ─────────────────────────────────────────────────────────────
def build_v11B():
    """
    v11-B: absolute ESM-2 + isoform-deviation → gated integration → MLP
    Gate controls how much within-gene deviation signal to use.
    delta = 0 for single-isoform genes → gate outputs 0 → pure v10-B behavior
    """
    inp_abs = layers.Input(shape=(640,), name='isoform_emb')
    inp_dev = layers.Input(shape=(640,), name='isoform_deviation')

    # Absolute path: same as v10-B
    h_abs = layers.Dense(256, activation='relu')(inp_abs)
    h_abs = layers.BatchNormalization()(h_abs)
    h_abs = layers.Dropout(0.3)(h_abs)

    # Deviation path: encodes what's unique about this isoform
    h_dev = layers.Dense(256, activation='relu')(inp_dev)

    # Attention gate: when deviation signal is informative, integrate it
    gate_input = layers.Concatenate()([h_abs, h_dev])
    gate = layers.Dense(256, activation='sigmoid', name='dev_gate')(gate_input)

    # Gated integration [R2.1]: gene-level deviation via gating
    h = layers.Add(name='gated_dev')([h_abs, layers.Multiply()([gate, h_dev])])

    # Predictor head
    x = layers.Dense(128, activation='relu')(h)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    return models.Model([inp_abs, inp_dev], out, name='v11B')


def get_cw(y):
    n_pos = max(int(y.sum()), 1)
    return {0: 1.0, 1: int((y == 0).sum()) / n_pos}


def train_v11B(X_tr_, dev_tr_, X_te_, dev_te_, y_tr_, seed, epochs=80, batch=512):
    sc_abs = StandardScaler()
    Xtr = sc_abs.fit_transform(X_tr_)
    Xte = sc_abs.transform(X_te_)

    # Scale deviation on non-zero rows; re-zero single-isoform entries
    non_zero_tr = np.any(dev_tr_ != 0, axis=1)
    non_zero_te = np.any(dev_te_ != 0, axis=1)
    if non_zero_tr.sum() > 50:
        sc_dev = StandardScaler()
        sc_dev.fit(dev_tr_[non_zero_tr])
        dev_tr_sc = sc_dev.transform(dev_tr_)
        dev_te_sc = sc_dev.transform(dev_te_)
        dev_tr_sc[~non_zero_tr] = 0.0
        dev_te_sc[~non_zero_te] = 0.0
    else:
        dev_tr_sc = dev_tr_.copy()
        dev_te_sc = dev_te_.copy()

    K.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = build_v11B()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    )

    rng = np.random.RandomState(seed)
    n_val = max(int(len(y_tr_) * 0.1), 100)
    vi = rng.choice(len(y_tr_), n_val, replace=False)
    ti = np.setdiff1d(np.arange(len(y_tr_)), vi)

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                 restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=5, verbose=0),
    ]

    model.fit(
        [Xtr[ti], dev_tr_sc[ti]], y_tr_[ti],
        validation_data=([Xtr[vi], dev_tr_sc[vi]], y_tr_[vi]),
        epochs=epochs, batch_size=batch,
        class_weight=get_cw(y_tr_),
        callbacks=cb, verbose=0,
    )

    return model.predict([Xte, dev_te_sc], verbose=0).ravel()


def train_lr(X_tr_, X_te_, y_tr_):
    sc = StandardScaler()
    clf = LogisticRegression(class_weight='balanced', C=1.0,
                              max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(X_tr_), y_tr_)
    return clf.predict_proba(sc.transform(X_te_))[:, 1]


# ─── Bootstrap CI ───────────────────────────────────────────────────────────────
def bootstrap_ci(y_true, probs, gene_ids, n_boot=N_BOOT, alpha=0.05, rs=42):
    rng = np.random.RandomState(rs)
    unique_genes = np.unique(gene_ids)
    gene_to_idx = defaultdict(list)
    for i, g in enumerate(gene_ids):
        gene_to_idx[g].append(i)

    boot_scores = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_genes, len(unique_genes), replace=True)
        idx = np.concatenate([gene_to_idx[g] for g in sampled])
        if y_true[idx].sum() == 0:
            continue
        try:
            boot_scores.append(average_precision_score(y_true[idx], probs[idx]))
        except Exception:
            continue

    if len(boot_scores) < 10:
        return np.nan, np.nan, np.nan
    ci_lo = np.percentile(boot_scores, 100 * alpha / 2)
    ci_hi = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return float(ci_lo), float(ci_hi), float((np.array(boot_scores) <= 0.5).mean())


# ─── pos_bias ───────────────────────────────────────────────────────────────────
def compute_pos_bias(probs, y_te, sym_ids):
    df = pd.DataFrame({'Gene': sym_ids, 'Score': probs, 'Label': y_te.astype(int)})
    global_std = df['Score'].std() + 1e-10
    pos_genes = df[df['Label'] == 1]['Gene'].unique()
    multi = df.groupby('Gene').filter(lambda g: len(g) >= 2)
    pos_multi = multi[multi['Gene'].isin(pos_genes)]
    if len(pos_multi) == 0:
        return np.nan, 0
    stds = pos_multi.groupby('Gene')['Score'].std().dropna()
    return float(stds.mean() / global_std), int(len(stds))


# ─── Main evaluation loop ────────────────────────────────────────────────────────
print("\n[2] Running v11-B evaluation on 13 GO terms ...")
print("=" * 70)

all_results = []
ts_start = time.strftime('%Y%m%d_%H%M')
te_genes_arr = np.array(te_syms)

for go_id, go_name in GO_TERMS:
    print(f"\n{'='*70}")
    print(f"  {go_id}  {go_name}")
    print(f"{'='*70}")

    y_tr, y_te = load_labels(go_id)
    n_pos_tr = int(y_tr.sum()); n_pos_te = int(y_te.sum())
    if n_pos_tr < 5 or n_pos_te == 0:
        print("  SKIP"); continue
    print(f"  n_pos: train={n_pos_tr}, test={n_pos_te}")

    sep = sep_cosine(X_te, y_te)
    go_type = 'A' if sep >= SEP_THRESHOLD else 'B'
    v10b_ref = V10B_REF.get(go_id, np.nan)
    v10b_pb  = V10B_POS_BIAS.get(go_id, None)
    pb_ref_str = f"{v10b_pb:.3f}" if v10b_pb is not None else 'N/A'
    print(f"  sep_cosine={sep:.4f}  Type-{go_type}  v10-B ref={v10b_ref:.4f}  pos_bias_ref={pb_ref_str}")

    # LR baseline
    t0 = time.time()
    lr_probs = train_lr(X_tr, X_te, y_tr)
    lr_auprc = float(average_precision_score(y_te, lr_probs))
    lr_ci_lo, lr_ci_hi, lr_p = bootstrap_ci(y_te, lr_probs, te_genes_arr)
    print(f"  LR:    AUPRC={lr_auprc:.4f} [{lr_ci_lo:.4f},{lr_ci_hi:.4f}] ({time.time()-t0:.0f}s)")

    # v11-B (5 seeds)
    seed_probs  = []
    seed_auprcs = []
    for seed in SEEDS:
        t0 = time.time()
        probs = train_v11B(X_tr, dev_tr, X_te, dev_te, y_tr, seed)
        auprc = float(average_precision_score(y_te, probs))
        seed_probs.append(probs)
        seed_auprcs.append(auprc)
        print(f"  v11-B seed={seed}: AUPRC={auprc:.4f} ({time.time()-t0:.0f}s)")

    v11b_mean      = float(np.mean(seed_auprcs))
    v11b_std       = float(np.std(seed_auprcs))
    avg_probs      = np.mean(seed_probs, axis=0)
    v11b_avg_auprc = float(average_precision_score(y_te, avg_probs))
    v11b_ci_lo, v11b_ci_hi, v11b_p = bootstrap_ci(y_te, avg_probs, te_genes_arr)
    pos_bias, n_multi = compute_pos_bias(avg_probs, y_te, te_syms)

    delta_vs_lr   = v11b_mean - lr_auprc
    delta_vs_v10b = v11b_mean - v10b_ref
    sig = '***' if v11b_p < 0.001 else ('**' if v11b_p < 0.01 else ('*' if v11b_p < 0.05 else 'n.s.'))
    pb_change = f"{pos_bias:.3f}" if not np.isnan(pos_bias) else 'N/A'
    if v10b_pb and not np.isnan(pos_bias):
        pb_delta = pos_bias - v10b_pb
        pb_change = f"{pos_bias:.3f} (Δ={pb_delta:+.3f})"

    print(f"  v11-B: mean={v11b_mean:.4f}±{v11b_std:.4f} "
          f"avg={v11b_avg_auprc:.4f} [{v11b_ci_lo:.4f},{v11b_ci_hi:.4f}] {sig}")
    print(f"  Δ(v11-B - LR)={delta_vs_lr:+.4f}  "
          f"Δ(v11-B - v10-B)={delta_vs_v10b:+.4f}  "
          f"pos_bias={pb_change}")

    result = {
        'go': go_id, 'name': go_name, 'type': go_type,
        'sep_cosine':   round(sep, 4),
        'n_pos_train':  n_pos_tr, 'n_pos_test': n_pos_te,
        'lr_auprc':     round(lr_auprc, 4),
        'lr_ci_lo':     round(lr_ci_lo, 4),
        'lr_ci_hi':     round(lr_ci_hi, 4),
        'v10b_ref':     round(v10b_ref, 4) if not np.isnan(v10b_ref) else None,
        'v10b_pos_bias': v10b_pb,
        'v11b_mean':     round(v11b_mean, 4),
        'v11b_std':      round(v11b_std, 4),
        'v11b_avg':      round(v11b_avg_auprc, 4),
        'v11b_ci_lo':    round(v11b_ci_lo, 4),
        'v11b_ci_hi':    round(v11b_ci_hi, 4),
        'v11b_p':        round(v11b_p, 4),
        'seed_auprcs':   [round(x, 4) for x in seed_auprcs],
        'delta_vs_lr':   round(delta_vs_lr, 4),
        'delta_vs_v10b': round(delta_vs_v10b, 4),
        'sig':           sig,
        'pos_bias':      round(pos_bias, 4) if not np.isnan(pos_bias) else None,
        'n_multi_pos_gene': n_multi,
    }
    all_results.append(result)

    with open(f'{OUT_DIR}/v11b_partial_{ts_start}.json', 'w') as f:
        json.dump({'results': all_results, 'timestamp': ts_start}, f, indent=2)

# ─── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  v11-B SUMMARY vs v10-B (13 GO TERMS)")
print("=" * 90)
print(f"{'GO':<15} {'Type':>5} {'LR':>8} {'v10-B':>8} {'v11-B':>8} {'Δ(B-ref)':>9} {'sig':>5} {'pos_bias':>9}")
print("-" * 82)

type_b_v11b = []; type_b_v10b = []; type_b_delta = []
type_a_v11b = []; type_a_v10b = []; type_a_delta = []

for r in all_results:
    mark  = '★' if r['type'] == 'B' else ' '
    v10b  = r.get('v10b_ref', float('nan'))
    v11b  = r['v11b_mean']
    delta = r['delta_vs_v10b']
    pb    = r.get('pos_bias')
    pb_str = f"{pb:.3f}" if pb is not None else 'N/A'
    print(f"{r['go']:<15} {mark+r['type']:>5} "
          f"{r['lr_auprc']:>8.4f} {v10b:>8.4f} {v11b:>8.4f} {delta:>+9.4f} "
          f"{r['sig']:>5} {pb_str:>9}")
    if r['type'] == 'B':
        type_b_v11b.append(v11b); type_b_v10b.append(v10b); type_b_delta.append(delta)
    else:
        type_a_v11b.append(v11b); type_a_v10b.append(v10b); type_a_delta.append(delta)

print("-" * 82)
if type_b_v11b:
    n_improved = sum(1 for d in type_b_delta if d > 0.01)
    print(f"  Type-B macro: v10-B={np.mean(type_b_v10b):.4f}  "
          f"v11-B={np.mean(type_b_v11b):.4f}  "
          f"Δ={np.mean(type_b_delta):+.4f}  "
          f"improved(>0.01)={n_improved}/{len(type_b_delta)}")
if type_a_v11b:
    print(f"  Type-A macro: v10-B={np.mean(type_a_v10b):.4f}  "
          f"v11-B={np.mean(type_a_v11b):.4f}  "
          f"Δ={np.mean(type_a_delta):+.4f}")

out_path = f'{OUT_DIR}/v11b_results_{ts_start}.json'
with open(out_path, 'w') as f:
    json.dump({
        'results': all_results,
        'timestamp': ts_start,
        'type_b_macro_v11b':  round(float(np.mean(type_b_v11b)), 4) if type_b_v11b else None,
        'type_b_macro_v10b':  round(float(np.mean(type_b_v10b)), 4) if type_b_v10b else None,
        'type_b_macro_delta': round(float(np.mean(type_b_delta)), 4) if type_b_delta else None,
        'type_a_macro_v11b':  round(float(np.mean(type_a_v11b)), 4) if type_a_v11b else None,
        'type_a_macro_v10b':  round(float(np.mean(type_a_v10b)), 4) if type_a_v10b else None,
        'v11a_negative_finding': {
            'description': 'mean_context approach caused gene-level bias injection',
            'ca2_delta': -0.1139,
            'sarcomere_delta': -0.0889,
            'pos_bias_ca2': {'v10b': 0.475, 'v11a': 0.1621, 'change': -66},
            'pos_bias_sarcomere': {'v10b': 1.176, 'v11a': 0.2933, 'change': -75},
        }
    }, f, indent=2)
print(f"\nFINAL: {out_path}")
