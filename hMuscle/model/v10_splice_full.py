"""
v10_splice_full.py — splicing BiGRU 기여 full train→test 측정
=============================================================
동기:
  v10-D (test CV): emb_only=0.513 → emb+splice=0.637 (+0.124) 확인됨.
  그러나 test CV는 고정 embedding 위에서의 CV이므로, train→test와 직접 비교 불가.

전략 — Zero-Imputation:
  train set (31668): NM ID 기반, GTF ENST 매핑 불가 → splicing_delta = zeros(150)
  test  set (36748): 실제 splicing_delta_v2 값 사용

  BiGRU가 학습하는 것:
    - zeros → "canonical or unknown" → embedding만 의존
    - non-zero → isoform-specific exon usage → refinement

Ablation:
  v10-B:  ESM-2 only (이미 0.7302)                   ← reference
  v10-E0: ESM-2 + splicing_zeros (both train/test)   ← zero-imputation upper bound check
  v10-E:  ESM-2 + splicing_BiGRU (train=0, test=real) ← 핵심 실험
  v10-D:  embedding + splice CV (이미 0.6367)         ← re-run for direct comparison

실행:
  conda activate isoform_env
  python hMuscle/model/v10_splice_full.py [GO_TERM | all]
"""

import os, sys, json, time, argparse
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
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
        if len(gpus) > 1:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"  Using GPU:0")
        else:
            tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(f"  GPU config warning: {e}")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v10_splice_real'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

GO_TERMS = [
    'GO:0007204', 'GO:0030017', 'GO:0006941', 'GO:0006914', 'GO:0043161',
    'GO:0007519', 'GO:0042692', 'GO:0055074', 'GO:0007005', 'GO:0007517',
    'GO:0032006', 'GO:0003774', 'GO:0006096',
]

# 이미 확정된 baselines
BASELINES = {
    'LR':   {'GO:0006096': 0.6949, 'GO:0003774': 0.8253, 'GO:0007204': 0.4138,
              'GO:0030017': 0.5609, 'GO:0006941': 0.3124, 'Macro': 0.5615},
    'v10B': {'GO:0006096': 0.7821, 'GO:0003774': 0.7648, 'GO:0007204': 0.7562,
              'GO:0030017': 0.7264, 'GO:0006941': 0.6214, 'Macro': 0.7302},
    'v10D_emb':   {'Macro': 0.5126},
    'v10D_splice':{'Macro': 0.6367},
}

# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

print("=" * 65)
print(" v10-Splice Full — Loading features ...")
print("=" * 65)

X_tr_esm2 = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te_esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
X_te_sd   = np.load(f'{FEAT_DIR}/splicing/splicing_delta_v2.npy').astype(np.float32)

# Train splicing delta — 실제 값 사용 (build_train_splicing_delta.py 생성)
_train_sd_path = f'{FEAT_DIR}/splicing/train_splicing_delta.npy'
if os.path.exists(_train_sd_path):
    X_tr_sd = np.load(_train_sd_path).astype(np.float32)
    # 차원 불일치 시 zero-pad 또는 truncate
    if X_tr_sd.shape[1] != X_te_sd.shape[1]:
        _d = X_te_sd.shape[1]
        _tmp = np.zeros((len(X_tr_sd), _d), dtype=np.float32)
        _tmp[:, :min(X_tr_sd.shape[1], _d)] = X_tr_sd[:, :min(X_tr_sd.shape[1], _d)]
        X_tr_sd = _tmp
    print(f"  [v10-E] Train splicing REAL loaded: non-zero={( np.abs(X_tr_sd).sum(1) > 0).sum()}")
else:
    X_tr_sd = np.zeros((len(X_tr_esm2), X_te_sd.shape[1]), dtype=np.float32)
    print("  [v10-E] Train splicing ZEROS (run build_train_splicing_delta.py to generate real values)")
X_tr_sd_zero = np.zeros((len(X_tr_esm2), X_te_sd.shape[1]), dtype=np.float32)

X_tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')
X_te_geneid = load_ids('my_gene_list_fixed.npy')

# Ensembl → symbol
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

X_te_genesymbol = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in X_te_geneid]

_tr_tag = 'REAL' if (np.abs(X_tr_sd).sum() > 0) else 'ZEROS'
print(f"  Train: ESM-2 {X_tr_esm2.shape}, splicing {_tr_tag} {X_tr_sd.shape}")
print(f"  Test:  ESM-2 {X_te_esm2.shape}, splicing REAL  {X_te_sd.shape}")
print(f"  Test splicing non-zero rows: {(X_te_sd.abs() if hasattr(X_te_sd, 'abs') else np.abs(X_te_sd)).sum(1).astype(bool).sum()}")


# ─── Label Loading ────────────────────────────────────────────────────────────
def load_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_te = np.array([1 if s in pos else 0 for s in X_te_genesymbol], dtype=np.float32)
    y_tr = np.array([1 if g in pos else 0 for g in X_tr_geneid], dtype=np.float32)
    return y_tr, y_te


# ─── Models ───────────────────────────────────────────────────────────────────
def build_v10E(esm_dim=640, sd_dim=150, emb_dim=64):
    """
    v10-E: ESM-2 → MLP (same as v10-B) + splicing BiGRU branch.
    Train with zeros for splicing (train set), real values at test time.
    """
    inp_esm = layers.Input(shape=(esm_dim,), name='esm2')
    inp_sd  = layers.Input(shape=(sd_dim,), name='splicing')

    # ESM-2 branch (identical to v10-B)
    x_esm = layers.Dense(256, activation='relu')(inp_esm)
    x_esm = layers.BatchNormalization()(x_esm)
    x_esm = layers.Dropout(0.3)(x_esm)
    x_esm = layers.Dense(128, activation='relu')(x_esm)
    x_esm = layers.Dropout(0.2)(x_esm)
    x_esm = layers.Dense(emb_dim, activation='relu', name='esm_feat')(x_esm)

    # Splicing BiGRU branch
    x_sd = layers.Reshape((sd_dim, 1))(inp_sd)
    x_sd = layers.Bidirectional(layers.GRU(32, return_sequences=False))(x_sd)
    x_sd = layers.Dense(32, activation='relu', name='splice_feat')(x_sd)

    # Fusion
    x = layers.Concatenate()([x_esm, x_sd])   # 64+32=96d
    x = layers.Dense(48, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid', name='pred')(x)

    emb_out = layers.Lambda(lambda t: K.l2_normalize(t, axis=-1),
                            name='embedding')(x_esm)
    return (models.Model([inp_esm, inp_sd], out, name='v10E'),
            models.Model([inp_esm, inp_sd], emb_out, name='v10E_emb'))


def build_v10E0(esm_dim=640, sd_dim=150, emb_dim=64):
    """
    v10-E0: same arch as v10-E but test also uses zeros.
    Checks if the BiGRU itself hurts vs v10-B (parameter overhead, noise).
    """
    return build_v10E(esm_dim, sd_dim, emb_dim)


# ─── Training ─────────────────────────────────────────────────────────────────
def get_cw(y):
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    return {0: 1.0, 1: n_neg / max(n_pos, 1)}


def train_eval(model, X_tr_list, X_te_list, y_tr, y_te, tag, epochs=80, batch=512):
    if y_tr.sum() < 2 or y_te.sum() == 0:
        return 0.0, 0.0

    scaler_esm = StandardScaler()
    X_tr_esm_s = scaler_esm.fit_transform(X_tr_list[0])
    X_te_esm_s = scaler_esm.transform(X_te_list[0])

    X_tr_in = [X_tr_esm_s] + X_tr_list[1:]
    X_te_in = [X_te_esm_s] + X_te_list[1:]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
        metrics=['AUC'],
    )
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=5, verbose=0),
    ]
    rng = np.random.RandomState(SEED)
    n_val = max(int(len(y_tr) * 0.1), 100)
    val_idx = rng.choice(len(y_tr), size=n_val, replace=False)
    tr_idx  = np.setdiff1d(np.arange(len(y_tr)), val_idx)

    X_tr_fit = [x[tr_idx] for x in X_tr_in]
    X_val    = [x[val_idx] for x in X_tr_in]

    t0 = time.time()
    model.fit(
        X_tr_fit, y_tr[tr_idx],
        validation_data=(X_val, y_tr[val_idx]),
        epochs=epochs, batch_size=batch,
        class_weight=get_cw(y_tr),
        callbacks=cb_list, verbose=0,
    )
    elapsed = time.time() - t0
    probs = model.predict(X_te_in, verbose=0).ravel()
    auprc = float(average_precision_score(y_te, probs))
    auroc = float(roc_auc_score(y_te, probs))
    print(f"    [{tag}] AUPRC={auprc:.4f}  AUROC={auroc:.4f}  ({elapsed:.0f}s)")
    return auprc, auroc


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_go(go_term):
    print(f"\n{'=' * 65}\n  {go_term}\n{'=' * 65}")
    y_tr, y_te = load_labels(go_term)
    print(f"  Train pos={int(y_tr.sum())}, Test pos={int(y_te.sum())}")
    if y_tr.sum() < 2 or y_te.sum() == 0:
        print("  SKIP")
        return None

    scaler = StandardScaler()
    scaler.fit(X_tr_esm2)  # fit on train for transform

    results = {'go': go_term, 'n_pos_train': int(y_tr.sum()), 'n_pos_test': int(y_te.sum())}

    # ── v10-E: ESM-2 + splicing_BiGRU (train=REAL if available, test=real) ─────
    _tr_sd_tag = 'REAL' if (np.abs(X_tr_sd).sum() > 0) else 'ZEROS'
    print(f"\n  [v10-E] ESM-2 + splicing_BiGRU (train={_tr_sd_tag}, test=REAL)")
    K.clear_session(); tf.random.set_seed(SEED)
    model_E, _ = build_v10E()
    auprc_E, auroc_E = train_eval(
        model_E,
        [X_tr_esm2, X_tr_sd],
        [X_te_esm2, X_te_sd],
        y_tr, y_te, tag='v10-E'
    )
    results['v10E_auprc'] = auprc_E
    results['v10E_auroc'] = auroc_E

    # ── v10-E0: ESM-2 + splicing_BiGRU (ZEROS for both) ─────────────────────
    # Verifies that BiGRU overhead alone doesn't hurt
    print("\n  [v10-E0] ESM-2 + splicing_BiGRU (ZEROS for both, control)")
    K.clear_session(); tf.random.set_seed(SEED)
    X_te_sd_zero = np.zeros((len(X_te_esm2), X_te_sd.shape[1]), dtype=np.float32)
    model_E0, _ = build_v10E0()
    auprc_E0, auroc_E0 = train_eval(
        model_E0,
        [X_tr_esm2, X_tr_sd_zero],
        [X_te_esm2, X_te_sd_zero],   # zeros at test too (correct shape)
        y_tr, y_te, tag='v10-E0'
    )
    results['v10E0_auprc'] = auprc_E0
    results['v10E0_auroc'] = auroc_E0

    return results


parser = argparse.ArgumentParser()
parser.add_argument('go', nargs='?', default='all')
args = parser.parse_args()
target = GO_TERMS if args.go.upper() == 'ALL' else [args.go]

all_results = []
for go in target:
    r = run_go(go)
    if r:
        all_results.append(r)

# ─── Summary ──────────────────────────────────────────────────────────────────
if all_results:
    GO_ORDER = ['GO:0006096', 'GO:0003774', 'GO:0007204', 'GO:0030017', 'GO:0006941']
    res_map = {r['go']: r for r in all_results}

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"{'GO Term':<15} {'v10-E':>8} {'v10-E0':>8} {'v10-B':>8} {'LR':>8}")
    print(f"{'':15} {'(+splice)':>8} {'(ctrl)':>8} {'(ref)':>8} {'(ref)':>8}")
    print("-" * 80)

    e_vals, e0_vals, b_vals = [], [], []
    for go in GO_ORDER:
        if go not in res_map:
            continue
        r = res_map[go]
        b = BASELINES['v10B'][go]
        lr = BASELINES['LR'][go]
        print(f"{go:<15} {r['v10E_auprc']:>8.4f} {r['v10E0_auprc']:>8.4f} {b:>8.4f} {lr:>8.4f}")
        e_vals.append(r['v10E_auprc']); e0_vals.append(r['v10E0_auprc']); b_vals.append(b)

    if e_vals:
        print("-" * 80)
        mE = np.mean(e_vals); mE0 = np.mean(e0_vals); mB = BASELINES['v10B']['Macro']
        print(f"{'Macro':<15} {mE:>8.4f} {mE0:>8.4f} {mB:>8.4f} {BASELINES['LR']['Macro']:>8.4f}")
        print()
        print("  Interpretation:")
        print(f"  splicing BiGRU contribution (E - v10B):      {mE - mB:+.4f}")
        print(f"  BiGRU overhead alone (E0 - v10B):            {mE0 - mB:+.4f}")
        print(f"  Real splicing signal (E - E0):               {mE - mE0:+.4f}")
        print(f"  v10-D splice CV reference:                   {BASELINES['v10D_splice']['Macro']:.4f}")
        print(f"  v10-D emb CV reference:                      {BASELINES['v10D_emb']['Macro']:.4f}")

    ts = time.strftime('%Y%m%d_%H%M')
    out = f'{OUT_DIR}/v10_splice_results_{ts}.json'
    with open(out, 'w') as f:
        json.dump({'results': all_results, 'baselines': BASELINES, 'timestamp': ts}, f, indent=2)
    print(f"\n[Saved] {out}")
