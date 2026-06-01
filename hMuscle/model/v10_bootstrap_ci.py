"""
v10_bootstrap_ci.py — Bootstrap CI for v10-B AUPRC
====================================================
목적: v10-B Macro AUPRC의 통계적 신뢰구간 확립.

방법:
1. Multi-seed: SEED = {42, 123, 456, 789, 1234} → 5개 독립 런
2. 각 런: v10-B (ESM-2 MLP) 학습 → test AUPRC per GO term
3. Gene-block bootstrap CI: test set에서 gene 단위 resampling
   (n_bootstrap=1000, 95% CI)
4. LR baseline과 비교: v10-B vs LR Δ의 통계적 유의성

비교 기준:
  LR baseline (human-only): Macro=0.561 (per GO: 확인 필요)
  v10-B: Macro=0.73-0.76 (두 런 모두 안정)

출력:
  reports/bootstrap_ci/{ts}/
    multi_seed_auprc.json    — seed별 per-GO AUPRC
    bootstrap_ci_results.json — 95% CI per GO term
    bootstrap_summary.txt    — 논문용 통계 요약

실행:
  conda activate isoform_env
  python hMuscle/model/v10_bootstrap_ci.py
"""

import os, sys, json, time
import numpy as np
from scipy.stats import percentileofscore
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        tf.config.set_visible_devices(gpus[1] if len(gpus) > 1 else gpus[0], 'GPU')
    except RuntimeError:
        pass

# ─── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
ts = time.strftime('%Y%m%d_%H%M')
OUT_DIR   = f'../../reports/bootstrap_ci/{ts}'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS        = [42, 123, 456, 789, 1234]
N_BOOTSTRAP  = 1000
GO_TERMS     = ['GO:0006096', 'GO:0003774', 'GO:0007204', 'GO:0030017', 'GO:0006941']

# ─── Data loading ──────────────────────────────────────────────────────────
print("=" * 65)
print(" Bootstrap CI for v10-B — Loading data ...")
print("=" * 65)

X_te_esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
X_tr_esm2 = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)

def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

X_te_isoid  = load_ids('my_isoform_list_fixed.npy')
X_te_geneid = load_ids('my_gene_list_fixed.npy')
X_tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

X_te_sym      = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in X_te_geneid]
X_te_genebase = [g.split('.')[0] for g in X_te_geneid]

# gene index → 해당 gene의 모든 isoform indices (bootstrap에서 gene 단위 resampling)
from collections import defaultdict
gene_to_idxs = defaultdict(list)
for i, g in enumerate(X_te_genebase):
    gene_to_idxs[g].append(i)
unique_genes = np.array(sorted(gene_to_idxs.keys()))
print(f"  Unique genes in test: {len(unique_genes)}")

N = len(X_te_isoid)


# ─── Model ─────────────────────────────────────────────────────────────────
def build_v10B(esm_dim=640):
    inp = layers.Input(shape=(esm_dim,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name='v10B')


def load_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_te = np.array([1 if s in pos else 0 for s in X_te_sym], dtype=np.float32)
    y_tr = np.array([1 if g in pos else 0 for g in X_tr_geneid], dtype=np.float32)
    return y_tr, y_te


def get_cw(y):
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    return {0: 1.0, 1: n_neg / max(n_pos, 1)}


def train_predict_v10b(go_term, seed):
    y_tr, y_te = load_labels(go_term)
    if y_tr.sum() < 2 or y_te.sum() == 0:
        return None, None

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_esm2)
    X_te_sc = sc.transform(X_te_esm2)

    rng = np.random.RandomState(seed)
    n_val = max(int(len(y_tr) * 0.1), 100)
    vi = rng.choice(len(y_tr), size=n_val, replace=False)
    ti = np.setdiff1d(np.arange(len(y_tr)), vi)

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    model = build_v10B()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    )
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=5, verbose=0),
    ]
    model.fit(X_tr_sc[ti], y_tr[ti],
              validation_data=(X_tr_sc[vi], y_tr[vi]),
              epochs=80, batch_size=512,
              class_weight=get_cw(y_tr),
              callbacks=cb_list, verbose=0)

    probs = model.predict(X_te_sc, verbose=0).ravel()
    return probs, y_te


# ─── Gene-block bootstrap CI ───────────────────────────────────────────────
def bootstrap_auprc(probs, y_te, gene_to_idxs, unique_genes,
                    n_bootstrap=N_BOOTSTRAP, random_state=42):
    """
    Gene 단위로 resampling → bootstrap AUPRC 분포.
    gene-level clustering 반영 (같은 gene isoforms 함께 이동).
    """
    rng = np.random.RandomState(random_state)
    boot_auprcs = []

    for _ in range(n_bootstrap):
        sampled_genes = rng.choice(unique_genes, size=len(unique_genes), replace=True)
        idxs = []
        for g in sampled_genes:
            idxs.extend(gene_to_idxs[g])
        idxs = np.array(idxs)

        y_boot = y_te[idxs]
        p_boot = probs[idxs]

        if y_boot.sum() == 0:
            continue
        boot_auprcs.append(float(average_precision_score(y_boot, p_boot)))

    return np.array(boot_auprcs)


# ─── LR baseline (단일 seed) ───────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression

def lr_auprc(go_term):
    y_tr, y_te = load_labels(go_term)
    if y_tr.sum() < 2:
        return None
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_esm2)
    X_te_sc = sc.transform(X_te_esm2)
    lr = LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000)
    lr.fit(X_tr_sc, y_tr)
    probs = lr.predict_proba(X_te_sc)[:, 1]
    return float(average_precision_score(y_te, probs))


# ─── Main ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" Step 1: LR Baseline AUPRC (single seed)")
print("=" * 65)

lr_results = {}
for go in GO_TERMS:
    a = lr_auprc(go)
    lr_results[go] = a
    print(f"  {go}: LR AUPRC = {a:.4f}")
lr_macro = float(np.mean([v for v in lr_results.values() if v is not None]))
print(f"  LR Macro = {lr_macro:.4f}")


print("\n" + "=" * 65)
print(f" Step 2: v10-B Multi-seed ({len(SEEDS)} seeds × {len(GO_TERMS)} GO terms)")
print("=" * 65)

# seed × go → probs, y_te 저장
multi_seed_probs = {go: {} for go in GO_TERMS}
multi_seed_labels = {}
multi_seed_auprc = {go: [] for go in GO_TERMS}

for seed in SEEDS:
    print(f"\n  --- SEED={seed} ---")
    for go in GO_TERMS:
        probs, y_te = train_predict_v10b(go, seed)
        if probs is None:
            continue
        auprc = float(average_precision_score(y_te, probs))
        multi_seed_probs[go][seed] = probs
        multi_seed_labels[go] = y_te
        multi_seed_auprc[go].append(auprc)
        print(f"    {go}: AUPRC={auprc:.4f}")

# per-GO 통계
print("\n" + "=" * 65)
print(" Step 3: Multi-seed Statistics")
print("=" * 65)
seed_stats = {}
for go in GO_TERMS:
    auprcs = multi_seed_auprc[go]
    if not auprcs:
        continue
    stats = {
        'mean': float(np.mean(auprcs)),
        'std':  float(np.std(auprcs)),
        'min':  float(np.min(auprcs)),
        'max':  float(np.max(auprcs)),
        'all':  auprcs,
        'lr':   lr_results.get(go),
    }
    seed_stats[go] = stats
    print(f"  {go}: mean={stats['mean']:.4f} ±{stats['std']:.4f} "
          f"[{stats['min']:.4f}-{stats['max']:.4f}] | LR={stats['lr']:.4f}")

macro_means = [v['mean'] for v in seed_stats.values()]
print(f"\n  v10-B Macro (mean of means): {np.mean(macro_means):.4f}")
print(f"  LR Macro: {lr_macro:.4f}")
print(f"  Δ Macro: +{np.mean(macro_means) - lr_macro:.4f}")


print("\n" + "=" * 65)
print(f" Step 4: Gene-block Bootstrap CI (n={N_BOOTSTRAP})")
print("=" * 65)

# best seed (seed=42)의 probs로 bootstrap
ci_results = {}
for go in GO_TERMS:
    if 42 not in multi_seed_probs[go]:
        continue
    probs = multi_seed_probs[go][42]
    y_te  = multi_seed_labels[go]

    print(f"  {go}: bootstrapping...", end='', flush=True)
    boot = bootstrap_auprc(probs, y_te, gene_to_idxs, unique_genes)
    obs_auprc = float(average_precision_score(y_te, probs))

    # LR bootstrap도 계산
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_esm2)
    X_te_sc = sc.transform(X_te_esm2)
    y_tr, _ = load_labels(go)
    lr_model = LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000)
    lr_model.fit(X_tr_sc, y_tr)
    lr_probs = lr_model.predict_proba(X_te_sc)[:, 1]

    boot_lr = bootstrap_auprc(lr_probs, y_te, gene_to_idxs, unique_genes)
    obs_lr = float(average_precision_score(y_te, lr_probs))

    # 95% CI
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    lr_ci_lo, lr_ci_hi = float(np.percentile(boot_lr, 2.5)), float(np.percentile(boot_lr, 97.5))

    # one-sided p-value: p(boot_v10b_delta > 0)
    delta_boot = boot - boot_lr
    p_val = float(np.mean(delta_boot <= 0))

    ci_results[go] = {
        'v10b_auprc':   obs_auprc,
        'v10b_ci_lo':   ci_lo,
        'v10b_ci_hi':   ci_hi,
        'lr_auprc':     obs_lr,
        'lr_ci_lo':     lr_ci_lo,
        'lr_ci_hi':     lr_ci_hi,
        'delta_auprc':  obs_auprc - obs_lr,
        'p_value':      p_val,
        'n_bootstrap':  len(boot),
    }
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))
    print(f" v10B={obs_auprc:.4f}[{ci_lo:.3f}-{ci_hi:.3f}] "
          f"LR={obs_lr:.4f}[{lr_ci_lo:.3f}-{lr_ci_hi:.3f}] "
          f"Δ={obs_auprc-obs_lr:+.4f} p={p_val:.3f}{sig}")


# ─── Save results ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" Saving results ...")
print("=" * 65)

# JSON: multi-seed AUPRC
ms_path = os.path.join(OUT_DIR, 'multi_seed_auprc.json')
with open(ms_path, 'w') as f:
    json.dump({
        'seeds': SEEDS,
        'lr_baseline': lr_results,
        'lr_macro': lr_macro,
        'seed_stats': seed_stats,
    }, f, indent=2)
print(f"  Multi-seed stats → {ms_path}")

# JSON: bootstrap CI
ci_path = os.path.join(OUT_DIR, 'bootstrap_ci_results.json')
with open(ci_path, 'w') as f:
    json.dump({
        'n_bootstrap': N_BOOTSTRAP,
        'per_go': ci_results,
        'v10b_macro': float(np.mean([v['v10b_auprc'] for v in ci_results.values()])),
        'lr_macro': float(np.mean([v['lr_auprc'] for v in ci_results.values()])),
    }, f, indent=2)
print(f"  Bootstrap CI → {ci_path}")

# TXT: 논문용 요약
summary_path = os.path.join(OUT_DIR, 'bootstrap_summary.txt')
with open(summary_path, 'w') as f:
    f.write("v10-B vs LR Bootstrap CI Results\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Method: Gene-block bootstrap (n={N_BOOTSTRAP}), "
            f"multi-seed (n={len(SEEDS)} seeds)\n\n")

    f.write("[ Multi-seed AUPRC (v10-B) ]\n")
    for go in GO_TERMS:
        if go not in seed_stats:
            continue
        s = seed_stats[go]
        f.write(f"  {go}: {s['mean']:.4f} ± {s['std']:.4f} "
                f"(range: {s['min']:.4f}–{s['max']:.4f}), "
                f"LR={s['lr']:.4f}\n")
    f.write(f"  Macro v10-B: {np.mean(macro_means):.4f}, LR: {lr_macro:.4f}, "
            f"Δ={np.mean(macro_means)-lr_macro:+.4f}\n\n")

    f.write("[ Bootstrap 95% CI (seed=42) ]\n")
    f.write(f"{'GO term':<15} {'v10-B':<22} {'LR':<22} {'Δ':<8} {'p':<8}\n")
    f.write("-" * 80 + "\n")
    for go in GO_TERMS:
        if go not in ci_results:
            continue
        r = ci_results[go]
        sig = '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01 else ('*' if r['p_value'] < 0.05 else 'n.s.'))
        f.write(f"{go:<15} "
                f"{r['v10b_auprc']:.4f}[{r['v10b_ci_lo']:.3f}-{r['v10b_ci_hi']:.3f}]  "
                f"{r['lr_auprc']:.4f}[{r['lr_ci_lo']:.3f}-{r['lr_ci_hi']:.3f}]  "
                f"{r['delta_auprc']:+.4f}  "
                f"{r['p_value']:.3f}{sig}\n")
    f.write(f"\n*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant\n")
print(f"  Summary → {summary_path}")

print(f"\n  Output: {OUT_DIR}/")
print("  Done.")
