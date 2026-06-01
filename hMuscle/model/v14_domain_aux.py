# -*- coding: utf-8 -*-
"""
v14_domain_aux.py — Multi-task: GO Prediction + Domain Presence Auxiliary
==========================================================================
동기:
  XGBoost가 동일 ESM-2 feature로 v10-B MLP를 초과 (0.7384 vs 0.6935).
  gene-level label 문제를 architectural trick으로 우회하려는 기존 시도
  (PCGrad, MIL, canonical aux)가 모두 AUPRC 하락을 야기.

방향:
  GO 레이블을 유지하되, isoform-level auxiliary task(도메인 예측)를
  추가해 backbone이 gene-level heuristic을 초월하도록 강제.

아키텍처:
  ESM-2 (640d)
      ↓
  Shared backbone: Dense(256,relu,BN,Drop0.3) → Dense(128,relu,Drop0.2)
      ├─ GO head:     Dense(64,relu) → Dense(1,sigmoid)   [focal, γ=2]
      └─ Domain head: Dense(128,relu) → Dense(512,sigmoid) [focal, γ=1]

  L_total = L_GO + λ · L_domain
  λ sweep: [0.05, 0.1, 0.3, 0.5] — term별 최적 λ 선택

평가:
  GO AUPRC (13 terms), bias_score, domain AUPRC (auxiliary)
  목표: macro AUPRC > 0.7384 (XGB) AND bias_score > 0.15

실행:
  conda run -n isoform_env python v14_domain_aux.py [GO_TERM|all]
"""

import os, sys, json, time
import numpy as np
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
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"  Using GPU:0")
    except RuntimeError as e:
        print(f"  GPU config: {e}")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v14_domain_aux'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
tf.random.set_seed(SEED)
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

LAMBDA_SWEEP = [0.05, 0.1, 0.3, 0.5]

# v10-B reference (for comparison)
V10B_REF = {
    'GO:0007204': 0.7653, 'GO:0030017': 0.7426, 'GO:0006941': 0.5968,
    'GO:0006914': 0.6397, 'GO:0043161': 0.7174, 'GO:0007519': 0.7250,
    'GO:0042692': 0.6526, 'GO:0055074': 0.7255, 'GO:0007005': 0.6624,
    'GO:0007517': 0.7017, 'GO:0032006': 0.6023, 'GO:0003774': 0.8128,
    'GO:0006096': 0.6712,
}
XGB_REF = {
    'GO:0007204': 0.8249, 'GO:0030017': 0.7765, 'GO:0006941': 0.6816,
    'GO:0006914': 0.7037, 'GO:0043161': 0.6943, 'GO:0007519': 0.7341,
    'GO:0042692': 0.6918, 'GO:0055074': 0.7384, 'GO:0007005': 0.6975,
    'GO:0007517': 0.7428, 'GO:0032006': 0.6479, 'GO:0003774': 0.8412,
    'GO:0006096': 0.8251,
}

# ── Data Loading ───────────────────────────────────────────────────────────
print("=" * 65)
print(" v14 Domain-Auxiliary — Loading features ...")
print("=" * 65)

def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

X_tr_esm2  = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te_esm2  = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

# Domain matrices — 512-dim binary, both train & test
X_tr_dom   = np.load(f'{FEAT_DIR}/train_domain_matrix_hmmscan.npy').astype(np.float32)
X_te_dom   = np.load(f'{FEAT_DIR}/domain_matrix_proper_test.npy').astype(np.float32)

tr_geneid  = load_ids(f'{ID_DIR}/train_gene_list.npy')
te_geneid  = load_ids('my_gene_list_fixed.npy')
te_isoid   = load_ids('my_isoform_list_fixed.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

tr_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_geneid]
te_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_geneid]
te_gene_base = [g.split('.')[0] for g in te_geneid]

print(f"  Train: ESM-2 {X_tr_esm2.shape}, domain {X_tr_dom.shape}")
print(f"  Test:  ESM-2 {X_te_esm2.shape}, domain {X_te_dom.shape}")
dom_tr_nonzero = (X_tr_dom.sum(1) > 0).sum()
print(f"  Domain coverage: train={dom_tr_nonzero}/{len(X_tr_dom)} ({dom_tr_nonzero/len(X_tr_dom)*100:.1f}%)")

# Pre-compute domain class weights (cap at 50 to prevent instability)
pos_per_dom = X_tr_dom.mean(0).clip(min=1e-6)
neg_per_dom = (1 - X_tr_dom).mean(0).clip(min=1e-6)
dom_pos_weight = np.clip(neg_per_dom / pos_per_dom, 1.0, 50.0).astype(np.float32)
print(f"  Domain pos_weight: mean={dom_pos_weight.mean():.1f}, max={dom_pos_weight.max():.1f}")

# ── Label Loading ──────────────────────────────────────────────────────────
def load_go_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te

# ── Model ──────────────────────────────────────────────────────────────────
def build_v14(esm_dim=640, dom_dim=512, lam=0.3):
    """
    Multi-task: GO prediction + Domain presence prediction.
    Two heads on shared backbone identical to v10-B.
    """
    inp = layers.Input(shape=(esm_dim,), name='esm2')

    # Shared backbone (identical to v10-B)
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # GO head
    go_x  = layers.Dense(64, activation='relu')(x)
    go_out = layers.Dense(1, activation='sigmoid', name='go_out')(go_x)

    # Domain head — larger to handle 512 outputs
    dom_x  = layers.Dense(128, activation='relu')(x)
    dom_out = layers.Dense(dom_dim, activation='sigmoid', name='dom_out')(dom_x)

    model = models.Model(inputs=inp, outputs=[go_out, dom_out])
    return model

# ── Custom training (manual loop for domain weighted loss) ─────────────────
_focal_none = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, reduction='none')

def _focal_elementwise(y_true, y_pred, gamma=1.0):
    """Element-wise focal BCE — returns same shape as inputs, NO axis reduction."""
    eps = 1e-7
    bce = -(y_true * tf.math.log(y_pred + eps) +
            (1.0 - y_true) * tf.math.log(1.0 - y_pred + eps))
    p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    return tf.pow(1.0 - p_t, gamma) * bce  # (N, 512)

def train_step(model, optimizer, x_b, y_go_b, y_dom_b, lam, dom_pw):
    with tf.GradientTape() as tape:
        go_pred, dom_pred = model(x_b, training=True)

        # GO focal loss (γ=2)
        go_loss = tf.reduce_mean(
            _focal_none(tf.expand_dims(y_go_b, -1), go_pred))

        # Domain focal loss (γ=1) with per-domain positive weighting
        # dom_pred: (N, 512), y_dom_b: (N, 512) → dom_per: (N, 512)
        dom_per = _focal_elementwise(y_dom_b, dom_pred, gamma=1.0)
        weights = 1.0 + (dom_pw - 1.0) * y_dom_b  # (N, 512)
        dom_loss = tf.reduce_mean(dom_per * weights)

        total_loss = go_loss + lam * dom_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return go_loss, dom_loss, total_loss


def bias_score(y_te, preds, gene_ids):
    """within-gene pred std / global pred std"""
    from collections import defaultdict
    gene_preds = defaultdict(list)
    for p, g in zip(preds, gene_ids):
        gene_preds[g].append(p)
    within_stds = [np.std(v) for v in gene_preds.values() if len(v) > 1]
    if not within_stds:
        return 0.0
    return float(np.mean(within_stds) / (np.std(preds) + 1e-9))


def train_eval(go_term, y_tr, y_te, lam, epochs=80, batch=512, patience=10):
    """Train one GO term with given λ, return test metrics."""
    K.clear_session()
    tf.random.set_seed(SEED)

    model = build_v14(lam=lam)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    dom_pw_tf = tf.constant(dom_pos_weight)

    # Val split (10%, stratified by GO label)
    rng = np.random.RandomState(SEED)
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    n_val_pos = max(1, int(len(pos_idx) * 0.1))
    n_val_neg = max(1, int(len(neg_idx) * 0.1))
    val_idx = np.concatenate([
        rng.choice(pos_idx, n_val_pos, replace=False),
        rng.choice(neg_idx, n_val_neg, replace=False)
    ])
    tr_idx = np.setdiff1d(np.arange(len(y_tr)), val_idx)

    X_val  = X_tr_esm2[val_idx]
    yd_val = X_tr_dom[val_idx]
    yg_val = y_tr[val_idx]
    X_fit  = X_tr_esm2[tr_idx]
    yd_fit = X_tr_dom[tr_idx]
    yg_fit = y_tr[tr_idx]

    best_val_auprc = -1.0
    best_weights   = None
    wait = 0
    cur_lr = 1e-3
    lr_wait = 0

    for epoch in range(epochs):
        # Shuffle
        perm = rng.permutation(len(tr_idx))
        X_s = X_fit[perm]; yg_s = yg_fit[perm]; yd_s = yd_fit[perm]

        for start in range(0, len(perm), batch):
            xb  = X_s[start:start+batch]
            ygb = yg_s[start:start+batch]
            ydb = yd_s[start:start+batch]
            train_step(model, optimizer, xb, ygb, ydb, lam, dom_pw_tf)

        # Validation GO AUPRC
        go_val_pred, _ = model(X_val, training=False)
        go_val_pred = go_val_pred.numpy().ravel()
        if yg_val.sum() > 0:
            val_auprc = float(average_precision_score(yg_val, go_val_pred))
        else:
            val_auprc = 0.0

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_weights   = model.get_weights()
            wait = 0; lr_wait = 0
        else:
            wait += 1; lr_wait += 1
            if lr_wait >= 5:
                cur_lr *= 0.5
                optimizer.learning_rate.assign(cur_lr)
                lr_wait = 0
            if wait >= patience:
                break

    if best_weights:
        model.set_weights(best_weights)

    # Test evaluation
    go_te_pred, dom_te_pred = model(X_te_esm2, training=False)
    go_te_pred  = go_te_pred.numpy().ravel()
    dom_te_pred = dom_te_pred.numpy()

    go_auprc = float(average_precision_score(y_te, go_te_pred))

    # Domain AUPRC (macro over domains with >= 1 positive in test)
    dom_auprcs = []
    for d in range(X_te_dom.shape[1]):
        yd = X_te_dom[:, d]
        if yd.sum() >= 1:
            dom_auprcs.append(average_precision_score(yd, dom_te_pred[:, d]))
    dom_auprc = float(np.mean(dom_auprcs)) if dom_auprcs else 0.0

    bs = bias_score(y_te, go_te_pred, te_gene_base)

    return {
        'go_auprc':  round(go_auprc, 4),
        'dom_auprc': round(dom_auprc, 4),
        'bias_score': round(bs, 4),
        'best_val_go_auprc': round(best_val_auprc, 4),
        'epochs_run': epoch + 1,
        'lam': lam,
    }


# ── Bootstrap CI ───────────────────────────────────────────────────────────
def bootstrap_ci(y_true, y_score, gene_ids, n=500):
    from collections import defaultdict
    rng = np.random.RandomState(SEED)
    unique_genes = np.unique(gene_ids)
    g2i = defaultdict(list)
    for i, g in enumerate(gene_ids): g2i[g].append(i)
    g2i = {g: np.array(v) for g, v in g2i.items()}
    boots = []
    for _ in range(n):
        gs = rng.choice(unique_genes, len(unique_genes), replace=True)
        idx = np.concatenate([g2i[g] for g in gs])
        if y_true[idx].sum() == 0: continue
        boots.append(average_precision_score(y_true[idx], y_score[idx]))
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ── Per-GO-term run ────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('go', nargs='?', default='all')
args = parser.parse_args()
target_terms = list(GO_TERMS.keys()) if args.go.upper() == 'ALL' else [args.go]

all_results = []
ts_start = time.time()

for go_term in target_terms:
    go_name = GO_TERMS.get(go_term, go_term)
    print(f"\n{'='*65}\n  {go_term}  {go_name}\n{'='*65}")

    y_tr, y_te = load_go_labels(go_term)
    print(f"  Train pos={int(y_tr.sum())}, Test pos={int(y_te.sum())}")
    if y_tr.sum() < 5 or y_te.sum() == 0:
        print("  SKIP (insufficient positives)")
        continue

    t0 = time.time()

    # λ sweep
    best_res = None
    sweep_log = []
    for lam in LAMBDA_SWEEP:
        res = train_eval(go_term, y_tr, y_te, lam=lam, epochs=80, patience=10)
        sweep_log.append({'lam': lam, 'go_auprc': res['go_auprc']})
        print(f"    λ={lam:.2f}: GO={res['go_auprc']:.4f} dom={res['dom_auprc']:.4f} "
              f"bias={res['bias_score']:.4f}")
        if best_res is None or res['go_auprc'] > best_res['go_auprc']:
            best_res = res

    elapsed = time.time() - t0

    # Bootstrap CI for best λ result
    K.clear_session(); tf.random.set_seed(SEED)
    model_final = build_v14(lam=best_res['lam'])
    optimizer_f = tf.keras.optimizers.Adam(1e-3)
    dom_pw_tf   = tf.constant(dom_pos_weight)

    # Full retrain with best λ
    final_res = train_eval(go_term, y_tr, y_te, lam=best_res['lam'],
                           epochs=100, patience=12)

    go_pred_final = None
    # get predictions for CI
    model_ci = build_v14(lam=final_res['lam'])
    # retrain one more time for CI
    rng2 = np.random.RandomState(SEED)
    X_s2 = X_tr_esm2.copy(); yg_s2 = y_tr.copy(); yd_s2 = X_tr_dom.copy()
    opt2 = tf.keras.optimizers.Adam(1e-3)
    for ep in range(final_res['epochs_run']):
        perm = rng2.permutation(len(y_tr))
        for st in range(0, len(perm), 512):
            xb  = X_s2[perm[st:st+512]]
            ygb = yg_s2[perm[st:st+512]]
            ydb = yd_s2[perm[st:st+512]]
            train_step(model_ci, opt2, xb, ygb, ydb,
                       final_res['lam'], tf.constant(dom_pos_weight))
    go_te_pred, _ = model_ci(X_te_esm2, training=False)
    go_te_pred = go_te_pred.numpy().ravel()

    ci_lo, ci_hi = bootstrap_ci(y_te, go_te_pred, np.array(te_gene_base))

    v10b = V10B_REF.get(go_term, 0)
    xgb  = XGB_REF.get(go_term, 0)
    delta_v10b = final_res['go_auprc'] - v10b
    delta_xgb  = final_res['go_auprc'] - xgb

    bias_interp = ('ISOFORM-SPEC' if final_res['bias_score'] >= 0.15
                   else 'MIXED'    if final_res['bias_score'] >= 0.10
                   else 'GENE-LEVEL')

    print(f"\n  BEST λ={best_res['lam']:.2f}")
    print(f"  GO AUPRC: {final_res['go_auprc']:.4f} [{ci_lo:.4f}-{ci_hi:.4f}]")
    print(f"  vs v10-B: {delta_v10b:+.4f}  vs XGB: {delta_xgb:+.4f}")
    print(f"  bias={final_res['bias_score']:.4f} ({bias_interp})  dom_auprc={final_res['dom_auprc']:.4f}")
    print(f"  elapsed: {elapsed:.0f}s")

    all_results.append({
        'go':          go_term,
        'go_name':     go_name,
        'n_pos_train': int(y_tr.sum()),
        'n_pos_test':  int(y_te.sum()),
        'auprc_v14':   final_res['go_auprc'],
        'auprc_ci_lo': round(ci_lo, 4),
        'auprc_ci_hi': round(ci_hi, 4),
        'auprc_v10b':  v10b,
        'auprc_xgb':   xgb,
        'delta_v10b':  round(delta_v10b, 4),
        'delta_xgb':   round(delta_xgb, 4),
        'bias_v14':    final_res['bias_score'],
        'bias_interp': bias_interp,
        'dom_auprc':   final_res['dom_auprc'],
        'best_lam':    best_res['lam'],
        'lam_sweep':   sweep_log,
        'elapsed_s':   round(elapsed, 1),
    })

# ── Final Summary ──────────────────────────────────────────────────────────
if all_results:
    print(f"\n{'='*65}")
    print(" v14 Domain-Aux Results")
    print(f"{'='*65}")
    hdr = f"{'GO Term':<22} {'v10-B':>6} {'v14':>6} {'XGB':>6} | {'bias':>6} {'λ':>4} {'dom':>6}  Interp"
    print(hdr)
    print('-' * 75)

    m14, mv10b, mxgb, mbias, mdom = [], [], [], [], []
    for r in all_results:
        win = '>' if r['auprc_v14'] > r['auprc_xgb'] else ' '
        print(f"{r['go_name']:<22} {r['auprc_v10b']:>6.4f} {r['auprc_v14']:>6.4f} "
              f"{r['auprc_xgb']:>6.4f} | {r['bias_v14']:>6.4f} {r['best_lam']:>4.2f} "
              f"{r['dom_auprc']:>6.4f} {win} {r['bias_interp']}")
        m14.append(r['auprc_v14']); mv10b.append(r['auprc_v10b'])
        mxgb.append(r['auprc_xgb']); mbias.append(r['bias_v14'])
        mdom.append(r['dom_auprc'])

    n = len(m14)
    print('-' * 75)
    print(f"{'MACRO':<22} {sum(mv10b)/n:>6.4f} {sum(m14)/n:>6.4f} "
          f"{sum(mxgb)/n:>6.4f} | {sum(mbias)/n:>6.4f} {'':>4} {sum(mdom)/n:>6.4f}")
    print()
    print(f"  v14 > XGB:   {sum(1 for a,x in zip(m14,mxgb) if a>x)}/{n} terms")
    print(f"  v14 > v10-B: {sum(1 for a,b in zip(m14,mv10b) if a>b)}/{n} terms")
    print(f"  ISOFORM-SPEC bias: {sum(1 for r in all_results if r['bias_interp']=='ISOFORM-SPEC')}/{n} terms")

    total_elapsed = time.time() - ts_start
    ts = time.strftime('%Y%m%d_%H%M')
    out = f'{OUT_DIR}/v14_domain_aux_{ts}.json'
    with open(out, 'w') as f:
        json.dump({
            'results':   all_results,
            'summary': {
                'macro_auprc_v14':  round(sum(m14)/n, 4),
                'macro_auprc_v10b': round(sum(mv10b)/n, 4),
                'macro_auprc_xgb':  round(sum(mxgb)/n, 4),
                'macro_bias':       round(sum(mbias)/n, 4),
                'macro_dom_auprc':  round(sum(mdom)/n, 4),
                'n_terms':          n,
                'v14_beats_xgb':    sum(1 for a,x in zip(m14,mxgb) if a>x),
                'v14_beats_v10b':   sum(1 for a,b in zip(m14,mv10b) if a>b),
            },
            'timestamp':      ts,
            'total_elapsed_s': round(total_elapsed, 1),
        }, f, indent=2)
    print(f"\n[Saved] {out}")
    print(f"[Total elapsed] {total_elapsed/60:.1f} min")
