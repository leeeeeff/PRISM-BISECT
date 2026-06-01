"""
v10_bias_score.py — v10-B / v10-E0 / v10-E bias_score 비교
============================================================
bias_score = mean(within_gene_score_std) / global_score_std
- 0에 가까울수록: gene-level shortcut (같은 유전자 이소폼이 동일 점수)
- 1에 가까울수록: isoform-specific 예측 (이소폼 간 점수 분산)
- 기준: < 0.10 → gene-level shortcut [R2.1]
         > 0.30 → isoform-specific 특징 활용 중

비교:
  v10-B:  ESM-2 only           → splice 없이 얼마나 isoform-level?
  v10-E0: ESM-2 + BiGRU(zeros) → 구조 overhead 효과
  v10-E:  ESM-2 + BiGRU(real)  → real splicing_delta 기여

실행:
  conda activate isoform_env
  python hMuscle/model/v10_bias_score.py [GO | all]
"""

import os, sys, json, time, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

from evaluation import compute_bias_score

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v10_mlp'

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

GO_TERMS = ['GO:0006096', 'GO:0003774', 'GO:0007204', 'GO:0030017', 'GO:0006941']

# ─── Data ─────────────────────────────────────────────────────────────────────
def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

print("=" * 65)
print(" v10 bias_score — Loading features ...")
print("=" * 65)

X_tr_esm2 = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te_esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
X_te_sd   = np.load(f'{FEAT_DIR}/splicing/splicing_delta_v2.npy').astype(np.float32)
X_tr_sd0  = np.zeros((len(X_tr_esm2), X_te_sd.shape[1]), dtype=np.float32)
X_te_sd0  = np.zeros((len(X_te_esm2), X_te_sd.shape[1]), dtype=np.float32)

X_tr_geneid  = load_ids(f'{ID_DIR}/train_gene_list.npy')
X_te_geneid  = load_ids('my_gene_list_fixed.npy')
X_te_isoid   = load_ids('my_isoform_list_fixed.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

X_te_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in X_te_geneid]
print(f"  Train {X_tr_esm2.shape}, Test {X_te_esm2.shape}, Splice {X_te_sd.shape}")


# ─── Labels ───────────────────────────────────────────────────────────────────
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


# ─── Models ───────────────────────────────────────────────────────────────────
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


def build_v10E(esm_dim=640, sd_dim=150):
    inp_esm = layers.Input(shape=(esm_dim,), name='esm2')
    inp_sd  = layers.Input(shape=(sd_dim,),  name='splicing')
    x_esm = layers.Dense(256, activation='relu')(inp_esm)
    x_esm = layers.BatchNormalization()(x_esm)
    x_esm = layers.Dropout(0.3)(x_esm)
    x_esm = layers.Dense(128, activation='relu')(x_esm)
    x_esm = layers.Dropout(0.2)(x_esm)
    x_esm = layers.Dense(64, activation='relu')(x_esm)
    x_sd  = layers.Reshape((sd_dim, 1))(inp_sd)
    x_sd  = layers.Bidirectional(layers.GRU(32))(x_sd)
    x_sd  = layers.Dense(32, activation='relu')(x_sd)
    x = layers.Concatenate()([x_esm, x_sd])
    x = layers.Dense(48, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model([inp_esm, inp_sd], out, name='v10E')


# ─── Train / Predict ──────────────────────────────────────────────────────────
def get_cw(y):
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    return {0: 1.0, 1: n_neg / max(n_pos, 1)}


def train_predict(model, X_tr, X_te, y_tr, tag, epochs=80, batch=512):
    """Train model, return test-set prediction probabilities."""
    sc = StandardScaler()
    if isinstance(X_tr, list):
        X_tr_in = [sc.fit_transform(X_tr[0])] + X_tr[1:]
        X_te_in = [sc.transform(X_te[0])] + X_te[1:]
    else:
        X_tr_in = sc.fit_transform(X_tr)
        X_te_in = sc.transform(X_te)

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
    rng = np.random.RandomState(SEED)
    n_val = max(int(len(y_tr) * 0.1), 100)
    vi = rng.choice(len(y_tr), size=n_val, replace=False)
    ti = np.setdiff1d(np.arange(len(y_tr)), vi)

    if isinstance(X_tr_in, list):
        Xf, Xv = [x[ti] for x in X_tr_in], [x[vi] for x in X_tr_in]
    else:
        Xf, Xv = X_tr_in[ti], X_tr_in[vi]

    t0 = time.time()
    model.fit(Xf, y_tr[ti], validation_data=(Xv, y_tr[vi]),
              epochs=epochs, batch_size=batch,
              class_weight=get_cw(y_tr),
              callbacks=cb_list, verbose=0)

    probs = model.predict(X_te_in, verbose=0).ravel()
    elapsed = time.time() - t0
    print(f"    [{tag}] ({elapsed:.0f}s)", end='')
    return probs


# ─── Per-GO Experiment ────────────────────────────────────────────────────────
def run_go(go_term):
    print(f"\n{'=' * 65}\n  {go_term}\n{'=' * 65}")
    y_tr, y_te = load_labels(go_term)
    if y_tr.sum() < 2 or y_te.sum() == 0:
        print("  SKIP")
        return None

    # Build DataFrame skeleton
    df_base = pd.DataFrame({
        'GeneID':   X_te_sym,
        'IsoformID': X_te_isoid,
        'Label':    y_te.astype(int),
    })

    results = {'go': go_term, 'n_pos': int(y_te.sum())}

    for tag, (build_fn, X_tr, X_te) in [
        ('v10-B',  (build_v10B,  X_tr_esm2,            X_te_esm2)),
        ('v10-E0', (build_v10E,  [X_tr_esm2, X_tr_sd0], [X_te_esm2, X_te_sd0])),
        ('v10-E',  (build_v10E,  [X_tr_esm2, X_tr_sd0], [X_te_esm2, X_te_sd])),
    ]:
        K.clear_session(); tf.random.set_seed(SEED)
        model = build_fn()
        probs = train_predict(model, X_tr, X_te, y_tr, tag)

        auprc = float(average_precision_score(y_te, probs))
        df = df_base.copy()
        df['Score'] = probs
        bs = compute_bias_score(df)

        tag_key = tag.replace('-', '').replace('v10', '').lower()  # b, e0, e
        results[f'{tag_key}_auprc']      = auprc
        results[f'{tag_key}_bias']       = bs['bias_score']
        results[f'{tag_key}_pos_bias']   = bs['pos_bias_score']
        results[f'{tag_key}_within_std'] = bs['within_gene_std']
        results[f'{tag_key}_global_std'] = bs['global_std']
        results[f'{tag_key}_n_multi']    = bs['n_multi_gene']

        print(f"  AUPRC={auprc:.4f}  bias={bs['bias_score']:.4f}  "
              f"pos_bias={bs['pos_bias_score']:.4f}  "
              f"n_multi={bs['n_multi_gene']}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('go', nargs='?', default='all')
args = parser.parse_args()
target = GO_TERMS if args.go.upper() == 'ALL' else [args.go]

all_results = []
for go in target:
    r = run_go(go)
    if r:
        all_results.append(r)

# ─── Summary Table ────────────────────────────────────────────────────────────
if all_results:
    res = {r['go']: r for r in all_results}

    print("\n" + "=" * 90)
    print("  BIAS SCORE TABLE")
    print("=" * 90)
    print(f"{'GO Term':<15} {'v10-B':>20} {'v10-E0':>20} {'v10-E':>20}")
    print(f"{'':15} {'AUPRC  bias  p_bias':>20} {'AUPRC  bias  p_bias':>20} {'AUPRC  bias  p_bias':>20}")
    print("-" * 90)

    b_bias=[]; e0_bias=[]; e_bias=[]
    b_pb=[]; e0_pb=[]; e_pb=[]
    b_auprc=[]; e0_auprc=[]; e_auprc=[]

    for go in GO_TERMS:
        if go not in res: continue
        r = res[go]
        print(f"{go:<15} "
              f"{r['b_auprc']:>6.4f} {r['b_bias']:>6.3f} {r['b_pos_bias']:>6.3f}   "
              f"{r['e0_auprc']:>6.4f} {r['e0_bias']:>6.3f} {r['e0_pos_bias']:>6.3f}   "
              f"{r['e_auprc']:>6.4f} {r['e_bias']:>6.3f} {r['e_pos_bias']:>6.3f}")
        b_bias.append(r['b_bias']); e0_bias.append(r['e0_bias']); e_bias.append(r['e_bias'])
        b_pb.append(r['b_pos_bias']); e0_pb.append(r['e0_pos_bias']); e_pb.append(r['e_pos_bias'])
        b_auprc.append(r['b_auprc']); e0_auprc.append(r['e0_auprc']); e_auprc.append(r['e_auprc'])

    import numpy as np
    print("-" * 90)
    print(f"{'Macro':<15} "
          f"{np.mean(b_auprc):>6.4f} {np.mean(b_bias):>6.3f} {np.mean(b_pb):>6.3f}   "
          f"{np.mean(e0_auprc):>6.4f} {np.mean(e0_bias):>6.3f} {np.mean(e0_pb):>6.3f}   "
          f"{np.mean(e_auprc):>6.4f} {np.mean(e_bias):>6.3f} {np.mean(e_pb):>6.3f}")

    print("\n  Legend: bias = all-isoform bias_score, p_bias = positive-gene bias_score")
    print(f"  Threshold: < 0.10 gene-level shortcut | > 0.30 isoform-specific")
    print(f"\n  splice contribution to bias_score (E - B): {np.mean(e_bias)-np.mean(b_bias):+.4f}")
    print(f"  splice contribution to pos_bias    (E - B): {np.mean(e_pb)-np.mean(b_pb):+.4f}")

    ts = time.strftime('%Y%m%d_%H%M')
    out = f'{OUT_DIR}/v10_bias_results_{ts}.json'
    with open(out, 'w') as f:
        json.dump({'results': all_results, 'timestamp': ts}, f, indent=2)
    print(f"\n[Saved] {out}")
