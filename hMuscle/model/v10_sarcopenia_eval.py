"""
v10_sarcopenia_eval.py — 근감소증 관련 8개 신규 GO term 전체 평가
=================================================================
v10-B (ESM-2 MLP) vs LR baseline
+ gene-block bootstrap CI (n=500)
+ pos_bias_score
+ Type-A/B classification (sep_ratio cosine threshold=0.111)

실행: conda activate isoform_env && python v10_sarcopenia_eval.py
소요: ~1-2시간 (GPU 1장 기준)
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

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/sarcopenia_eval'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 8 신규 GO terms (sarcopenia/skeletal muscle) ─────────────────────────────
NEW_GO_TERMS = [
    ('GO:0006914', 'Autophagy'),
    ('GO:0043161', 'Proteasome-UPS catabolic'),
    ('GO:0032006', 'Regulation of TOR signaling'),
    ('GO:0007519', 'Skeletal muscle tissue dev'),
    ('GO:0042692', 'Muscle cell differentiation'),
    ('GO:0055074', 'Calcium ion homeostasis'),
    ('GO:0007005', 'Mitochondrion organization'),
    ('GO:0007517', 'Muscle organ development'),
]

SEEDS  = [42, 123, 456, 789, 2024]
N_BOOT = 500
SEP_THRESHOLD = 0.060   # Type-A/B cosine threshold (13-term LOOCV 100%, updated from 0.111)

# ─── Load features ────────────────────────────────────────────────────────────
print("=" * 70)
print("  v10 Sarcopenia GO Term Evaluation")
print("=" * 70)

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

te_syms = [ensg2sym.get(g.split('.')[0], g.split('.')[0]) for g in te_genes]
print(f"  Train {X_tr.shape}, Test {X_te.shape}")

go_to_genes = defaultdict(set)
with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2: continue
        for go in parts[1:]:
            go_to_genes[go].add(parts[0])

def load_labels(go_term):
    pos = go_to_genes[go_term]
    y_te = np.array([1 if s in pos else 0 for s in te_syms], dtype=np.float32)
    y_tr = np.array([1 if g in pos else 0 for g in tr_genes], dtype=np.float32)
    return y_tr, y_te

# ─── sep_ratio (Type-A/B) ─────────────────────────────────────────────────────
def sep_cosine(X, y, rs=42):
    rng = np.random.RandomState(rs)
    pi = np.where(y == 1)[0]; ni = np.where(y == 0)[0]
    if len(pi) < 5: return np.nan
    if len(ni) > 2000: ni = rng.choice(ni, 2000, replace=False)
    if len(pi) > 2000: pi = rng.choice(pi, 2000, replace=False)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    pc = Xn[pi].mean(0); nc = Xn[ni].mean(0)
    inter = 1 - np.dot(pc, nc) / (np.linalg.norm(pc) * np.linalg.norm(nc) + 1e-10)
    Xp = Xn[pi[:500]]
    m = Xp @ Xp.T; up = np.triu(np.ones(m.shape, bool), k=1)
    return float(inter / max(1 - m[up].mean(), 1e-10))

# ─── Model builders ───────────────────────────────────────────────────────────
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
    n_pos = max(int(y.sum()), 1)
    return {0: 1.0, 1: int((y == 0).sum()) / n_pos}

def train_v10B(X_tr_, X_te_, y_tr_, seed, epochs=80, batch=512):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr_); Xte = sc.transform(X_te_)
    K.clear_session(); tf.random.set_seed(seed); np.random.seed(seed)
    model = build_v10B()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
    rng = np.random.RandomState(seed)
    n_val = max(int(len(y_tr_) * 0.1), 100)
    vi = rng.choice(len(y_tr_), n_val, replace=False)
    ti = np.setdiff1d(np.arange(len(y_tr_)), vi)
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True, verbose=0),
          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=5, verbose=0)]
    model.fit(Xtr[ti], y_tr_[ti], validation_data=(Xtr[vi], y_tr_[vi]),
              epochs=epochs, batch_size=batch, class_weight=get_cw(y_tr_),
              callbacks=cb, verbose=0)
    return model.predict(Xte, verbose=0).ravel()

def train_lr(X_tr_, X_te_, y_tr_):
    sc = StandardScaler()
    clf = LogisticRegression(class_weight='balanced', C=1.0,
                              max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(X_tr_), y_tr_)
    return clf.predict_proba(sc.transform(X_te_))[:, 1]

# ─── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_ci(y_true, probs, gene_ids, n_boot=500, alpha=0.05, rs=42):
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
    p_val = (np.array(boot_scores) <= 0.5).mean()
    return float(ci_lo), float(ci_hi), float(p_val)

# ─── pos_bias ─────────────────────────────────────────────────────────────────
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

# ─── Main loop ────────────────────────────────────────────────────────────────
all_results = []
ts_start = time.strftime('%Y%m%d_%H%M')

for go_id, go_name in NEW_GO_TERMS:
    print(f"\n{'='*70}")
    print(f"  {go_id}  {go_name}")
    print(f"{'='*70}")

    y_tr, y_te = load_labels(go_id)
    n_pos_tr = int(y_tr.sum()); n_pos_te = int(y_te.sum())
    if n_pos_tr < 5 or n_pos_te == 0:
        print("  SKIP (insufficient positives)"); continue
    print(f"  n_pos: train={n_pos_tr}, test={n_pos_te}")

    # Type-A/B classification
    sep = sep_cosine(X_te, y_te)
    go_type = 'A' if sep >= SEP_THRESHOLD else 'B'
    print(f"  sep_cosine={sep:.4f}  Type-{go_type}")

    # LR baseline (single run, deterministic)
    t0 = time.time()
    lr_probs = train_lr(X_tr, X_te, y_tr)
    lr_auprc = float(average_precision_score(y_te, lr_probs))
    lr_ci_lo, lr_ci_hi, lr_p = bootstrap_ci(y_te, lr_probs, np.array(te_syms), N_BOOT)
    print(f"  LR:    AUPRC={lr_auprc:.4f} [{lr_ci_lo:.4f},{lr_ci_hi:.4f}] "
          f"p={lr_p:.3f} ({time.time()-t0:.0f}s)")

    # v10-B (5 seeds)
    seed_probs = []
    seed_auprcs = []
    for seed in SEEDS:
        t0 = time.time()
        probs = train_v10B(X_tr, X_te, y_tr, seed)
        auprc = float(average_precision_score(y_te, probs))
        seed_probs.append(probs)
        seed_auprcs.append(auprc)
        print(f"  v10-B seed={seed}: AUPRC={auprc:.4f} ({time.time()-t0:.0f}s)")

    v10b_mean  = float(np.mean(seed_auprcs))
    v10b_std   = float(np.std(seed_auprcs))
    avg_probs  = np.mean(seed_probs, axis=0)
    v10b_avg_auprc = float(average_precision_score(y_te, avg_probs))
    v10b_ci_lo, v10b_ci_hi, v10b_p = bootstrap_ci(
        y_te, avg_probs, np.array(te_syms), N_BOOT)
    pos_bias, n_multi = compute_pos_bias(avg_probs, y_te, te_syms)

    delta = v10b_mean - lr_auprc
    sig = '***' if v10b_p < 0.001 else ('**' if v10b_p < 0.01 else ('*' if v10b_p < 0.05 else 'n.s.'))

    print(f"  v10-B: mean={v10b_mean:.4f}±{v10b_std:.4f} "
          f"avg={v10b_avg_auprc:.4f} [{v10b_ci_lo:.4f},{v10b_ci_hi:.4f}] "
          f"p={v10b_p:.3f} {sig}")
    print(f"  Δ(v10-B - LR) = {delta:+.4f}  |  pos_bias={pos_bias:.4f} (n_multi={n_multi})")

    result = {
        'go':        go_id,
        'name':      go_name,
        'type':      go_type,
        'sep_cosine': round(sep, 4),
        'n_pos_train': n_pos_tr,
        'n_pos_test':  n_pos_te,
        # LR
        'lr_auprc':  round(lr_auprc, 4),
        'lr_ci_lo':  round(lr_ci_lo, 4),
        'lr_ci_hi':  round(lr_ci_hi, 4),
        'lr_p':      round(lr_p, 4),
        # v10-B
        'v10b_mean':     round(v10b_mean, 4),
        'v10b_std':      round(v10b_std, 4),
        'v10b_avg':      round(v10b_avg_auprc, 4),
        'v10b_ci_lo':    round(v10b_ci_lo, 4),
        'v10b_ci_hi':    round(v10b_ci_hi, 4),
        'v10b_p':        round(v10b_p, 4),
        'seed_auprcs':   [round(x, 4) for x in seed_auprcs],
        # Delta
        'delta':     round(delta, 4),
        'sig':       sig,
        # Bias
        'pos_bias':  round(pos_bias, 4) if not np.isnan(pos_bias) else None,
        'n_multi_pos_gene': n_multi,
    }
    all_results.append(result)

    # Intermediate save
    with open(f'{OUT_DIR}/sarcopenia_partial_{ts_start}.json', 'w') as f:
        json.dump({'results': all_results, 'timestamp': ts_start}, f, indent=2)

# ─── Combined summary (new + existing) ────────────────────────────────────────
EXISTING = [
    {'go':'GO:0007204','name':'Ca2+ signaling','type':'B','n_pos_test':310,
     'lr_auprc':0.4147,'v10b_mean':0.7653,'delta':+0.3506,'sig':'***','pos_bias':0.475},
    {'go':'GO:0030017','name':'Sarcomere org','type':'B','n_pos_test':452,
     'lr_auprc':0.5635,'v10b_mean':0.7426,'delta':+0.1791,'sig':'***','pos_bias':1.176},
    {'go':'GO:0006941','name':'Muscle contraction','type':'B','n_pos_test':253,
     'lr_auprc':0.3102,'v10b_mean':0.5968,'delta':+0.2866,'sig':'***','pos_bias':1.902},
    {'go':'GO:0003774','name':'Motor activity','type':'A','n_pos_test':164,
     'lr_auprc':0.8253,'v10b_mean':0.8128,'delta':-0.0125,'sig':'n.s.','pos_bias':1.435},
    {'go':'GO:0006096','name':'Glycolysis','type':'A','n_pos_test':76,
     'lr_auprc':0.6949,'v10b_mean':0.6712,'delta':-0.0237,'sig':'n.s.','pos_bias':0.663},
]

print("\n" + "=" * 90)
print("  COMPREHENSIVE SUMMARY: 13 GO TERMS (5 existing + 8 new)")
print("=" * 90)
print(f"{'GO':<15} {'Name':<32} {'Type':>5} {'n_pos':>6} "
      f"{'LR':>8} {'v10-B':>8} {'Δ':>8} {'sig':>5} {'pos_bias':>9}")
print("-" * 90)

all_b_deltas = []; all_b_v10b = []; all_b_lr = []
all_a_deltas = []; all_a_v10b = []; all_a_lr = []

def print_row(r, source='new'):
    mark = '★' if r.get('type') == 'B' else ' '
    delta = r.get('delta', np.nan)
    pb = r.get('pos_bias')
    pb_str = f"{pb:.3f}" if pb is not None else 'N/A'
    v10b = r.get('v10b_mean', r.get('v10b_avg', np.nan))
    lr   = r.get('lr_auprc', np.nan)
    print(f"{r['go']:<15} {r['name'][:31]:<32} {mark+r.get('type','?'):>5} "
          f"{r['n_pos_test']:>6} {lr:>8.4f} {v10b:>8.4f} "
          f"{delta:>+8.4f} {r.get('sig',''):>5} {pb_str:>9}")

# Print existing
for r in EXISTING:
    print_row(r, 'existing')
    if r['type'] == 'B':
        all_b_deltas.append(r['delta']); all_b_v10b.append(r['v10b_mean'])
        all_b_lr.append(r['lr_auprc'])
    else:
        all_a_deltas.append(r['delta']); all_a_v10b.append(r['v10b_mean'])
        all_a_lr.append(r['lr_auprc'])

print("  ···")
# Print new
for r in all_results:
    print_row(r, 'new')
    if r['type'] == 'B':
        all_b_deltas.append(r['delta']); all_b_v10b.append(r['v10b_mean'])
        all_b_lr.append(r['lr_auprc'])
    else:
        all_a_deltas.append(r['delta']); all_a_v10b.append(r['v10b_mean'])
        all_a_lr.append(r['lr_auprc'])

print("-" * 90)
print(f"  Type-B macro (v10-B > LR expected): "
      f"v10-B={np.mean(all_b_v10b):.4f}, LR={np.mean(all_b_lr):.4f}, "
      f"Δ={np.mean(all_b_deltas):+.4f}")
print(f"  Type-A macro (LR >= v10-B expected): "
      f"v10-B={np.mean(all_a_v10b):.4f}, LR={np.mean(all_a_lr):.4f}, "
      f"Δ={np.mean(all_a_deltas):+.4f}")

# Final save
ts = time.strftime('%Y%m%d_%H%M')
out = f'{OUT_DIR}/sarcopenia_final_{ts}.json'
with open(out, 'w') as f:
    json.dump({
        'new_results': all_results,
        'existing_results': EXISTING,
        'summary': {
            'n_typeB': len(all_b_deltas),
            'n_typeA': len(all_a_deltas),
            'typeB_macro_v10b': float(np.mean(all_b_v10b)),
            'typeB_macro_lr':   float(np.mean(all_b_lr)),
            'typeB_macro_delta': float(np.mean(all_b_deltas)),
            'typeA_macro_v10b': float(np.mean(all_a_v10b)),
            'typeA_macro_lr':   float(np.mean(all_a_lr)),
        },
        'timestamp': ts,
    }, f, indent=2)
print(f"\n[Saved] {out}")
print(f"Done at {ts}.")
