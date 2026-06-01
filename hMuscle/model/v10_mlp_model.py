"""
v10_mlp_model.py — Gate 1 PASS 후속: PFN → MLP 교체
=======================================================
진단 결과:
  D1_tt Macro = 0.6898 (>= 0.52) → Gate 1 PASS
  → PFN+압축이 병목. 단순 MLP가 LR보다 22.8%, v8b보다 43.3% 우위

Architecture:
  Phase 1 (train→test, full):
    Branch 1: ESM-2 abs (640d) → Dense(256) → Dropout(0.3) → Dense(128) → 128d
    Branch 2: domain_delta_sign (251d) → Conv1D(32,k=5) → GlobalMaxPool → 32d
    Concat(160d) → Dense(64) → L2norm → embedding(64d)
    → Dense(1, sigmoid) → prediction

  Phase 2 (test CV, splicing ablation):
    embedding(64d) + splicing_delta_v2(150d) → BiGRU(32) → concat → Dense(32) → prediction
    Gene-stratified GroupKFold (K=5)

Ablation plan:
  v10-A: ESM-2 + domain_CNN (full train→test)       ← primary
  v10-B: ESM-2 only (no domain)                      ← domain contribution
  v10-C: ESM-2 + domain_Dense (no Conv1D)            ← Conv1D vs Dense
  v10-D: embedding + splicing_BiGRU (test CV)        ← splicing contribution

실행:
  conda activate isoform_env
  python hMuscle/model/v10_mlp_model.py [GO_TERM]
  예: python hMuscle/model/v10_mlp_model.py GO:0006096
  예: python hMuscle/model/v10_mlp_model.py all
"""

import os, sys, json, time, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
tf.get_logger().setLevel('ERROR')
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Use GPU:1 if GPU:0 is occupied; enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        # Prefer GPU:1 (GPU:0 tends to be occupied by other jobs)
        if len(gpus) > 1:
            tf.config.set_visible_devices(gpus[1], 'GPU')
            print(f"  Using GPU:1 ({gpus[1].name})")
        else:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"  Using GPU:0 ({gpus[0].name})")
    except RuntimeError as e:
        print(f"  GPU config warning: {e}")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v10_mlp'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── GO Terms ─────────────────────────────────────────────────────────────────
GO_TERMS = [
    'GO:0006096',  # Glycolysis       (Type-A)
    'GO:0003774',  # Motor Activity   (Type-A)
    'GO:0007204',  # Ca2+ Signaling   (Type-B)
    'GO:0030017',  # Sarcomere        (Type-B)
    'GO:0006941',  # Muscle Contraction (Type-B)
]

BASELINES = {
    'LR':  {'GO:0006096': 0.6949, 'GO:0003774': 0.8253, 'GO:0007204': 0.4138,
            'GO:0030017': 0.5609, 'GO:0006941': 0.3124, 'Macro': 0.5615},
    'D1_MLP': {'GO:0006096': 0.4792, 'GO:0003774': 0.7897, 'GO:0007204': 0.7664,
               'GO:0030017': 0.7612, 'GO:0006941': 0.6525, 'Macro': 0.6898},
    'v8b': {'GO:0006096': 0.7945, 'GO:0003774': 0.5686, 'GO:0007204': 0.1462,
            'GO:0030017': 0.1570, 'GO:0006941': 0.1177, 'Macro': 0.3568},
}

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


print("=" * 65)
print(" v10-MLP — Loading features ...")
print("=" * 65)

# Train features
X_tr_esm2 = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_tr_dd   = np.load(f'{FEAT_DIR}/train_domain_delta_sign.npy').astype(np.float32)

# Test features
X_te_esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
X_te_dd   = np.sign(np.load(f'{FEAT_DIR}/domain_delta_v2.npy')).astype(np.float32)
X_te_sd   = np.load(f'{FEAT_DIR}/splicing/splicing_delta_v2.npy').astype(np.float32)

# IDs
X_tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')
X_te_geneid = load_ids('my_gene_list_fixed.npy')

print(f"  Train: ESM-2 {X_tr_esm2.shape}, domain_delta {X_tr_dd.shape}")
print(f"  Test:  ESM-2 {X_te_esm2.shape}, domain_delta {X_te_dd.shape}, splicing {X_te_sd.shape}")

# Ensembl → symbol mapping
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

# Gene base IDs and symbols for test set
X_te_genebase   = [g.split('.')[0] for g in X_te_geneid]
X_te_genesymbol = [ENSG2SYM.get(g, g) for g in X_te_genebase]


# ─── Label Loading ────────────────────────────────────────────────────────────
def load_labels(go_term):
    pos_genes = set()
    for fname in ['human_annotations.txt']:
        fpath = f'{ANNOT_DIR}/{fname}'
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > 1 and go_term in parts[1:]:
                    pos_genes.add(parts[0])
    y_te = np.array([1 if s in pos_genes else 0 for s in X_te_genesymbol], dtype=np.float32)
    y_tr = np.array([1 if g in pos_genes else 0 for g in X_tr_geneid], dtype=np.float32)
    return y_tr, y_te, pos_genes


# ─── Model Builders ───────────────────────────────────────────────────────────
def build_model_A(esm_dim=640, dd_dim=251, emb_dim=64):
    """v10-A: ESM-2 + domain_delta Conv1D → embedding"""
    inp_esm = layers.Input(shape=(esm_dim,), name='esm2')
    inp_dd  = layers.Input(shape=(dd_dim,), name='domain_delta')

    # ESM-2 branch
    x_esm = layers.Dense(256, activation='relu')(inp_esm)
    x_esm = layers.BatchNormalization()(x_esm)
    x_esm = layers.Dropout(0.3)(x_esm)
    x_esm = layers.Dense(128, activation='relu')(x_esm)
    x_esm = layers.Dropout(0.2)(x_esm)

    # Domain_delta Conv1D branch (captures local domain gain/loss patterns)
    x_dd = layers.Reshape((dd_dim, 1))(inp_dd)
    x_dd = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(x_dd)
    x_dd = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu')(x_dd)
    x_dd = layers.GlobalMaxPooling1D()(x_dd)

    # Fusion
    x = layers.Concatenate()([x_esm, x_dd])  # 128+16=144d
    x = layers.Dense(emb_dim, activation='relu')(x)
    x = layers.Lambda(lambda t: K.l2_normalize(t, axis=-1), name='embedding')(x)

    # Prediction head
    out = layers.Dense(1, activation='sigmoid', name='pred')(x)

    return models.Model([inp_esm, inp_dd], out, name='v10A')


def build_model_B(esm_dim=640, emb_dim=64):
    """v10-B: ESM-2 only → embedding (ablation: no domain_delta)"""
    inp_esm = layers.Input(shape=(esm_dim,), name='esm2')

    x = layers.Dense(256, activation='relu')(inp_esm)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(emb_dim, activation='relu')(x)
    x = layers.Lambda(lambda t: K.l2_normalize(t, axis=-1), name='embedding')(x)
    out = layers.Dense(1, activation='sigmoid', name='pred')(x)

    return models.Model(inp_esm, out, name='v10B')


def build_model_C(esm_dim=640, dd_dim=251, emb_dim=64):
    """v10-C: ESM-2 + domain_delta Dense (ablation: no Conv1D, check architecture effect)"""
    inp_esm = layers.Input(shape=(esm_dim,), name='esm2')
    inp_dd  = layers.Input(shape=(dd_dim,), name='domain_delta')

    x_esm = layers.Dense(256, activation='relu')(inp_esm)
    x_esm = layers.BatchNormalization()(x_esm)
    x_esm = layers.Dropout(0.3)(x_esm)
    x_esm = layers.Dense(128, activation='relu')(x_esm)
    x_esm = layers.Dropout(0.2)(x_esm)

    # Dense instead of Conv1D
    x_dd = layers.Dense(64, activation='relu')(inp_dd)
    x_dd = layers.Dense(16, activation='relu')(x_dd)

    x = layers.Concatenate()([x_esm, x_dd])
    x = layers.Dense(emb_dim, activation='relu')(x)
    x = layers.Lambda(lambda t: K.l2_normalize(t, axis=-1), name='embedding')(x)
    out = layers.Dense(1, activation='sigmoid', name='pred')(x)

    return models.Model([inp_esm, inp_dd], out, name='v10C')


# ─── Training Utilities ────────────────────────────────────────────────────────
def get_class_weight(y):
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return {0: 1.0, 1: 1.0}
    return {0: 1.0, 1: n_neg / n_pos}


def train_eval_model(model, X_tr, X_te, y_tr, y_te, tag='', epochs=80, batch=512):
    """Train model, evaluate on test set. Returns AUPRC."""
    if y_tr.sum() < 2 or y_te.sum() == 0:
        return 0.0, 0.0, None

    cw = get_class_weight(y_tr)
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

    # Validation split from train
    n_val = max(int(len(y_tr) * 0.1), 100)
    rng = np.random.RandomState(SEED)
    val_idx = rng.choice(len(y_tr), size=n_val, replace=False)
    tr_idx  = np.setdiff1d(np.arange(len(y_tr)), val_idx)

    if isinstance(X_tr, list):
        X_tr_fit = [x[tr_idx] for x in X_tr]
        X_val    = [x[val_idx] for x in X_tr]
        X_te_in  = X_te
    else:
        X_tr_fit = X_tr[tr_idx]
        X_val    = X_tr[val_idx]
        X_te_in  = X_te

    t0 = time.time()
    model.fit(
        X_tr_fit, y_tr[tr_idx],
        validation_data=(X_val, y_tr[val_idx]),
        epochs=epochs, batch_size=batch,
        class_weight=cw,
        callbacks=cb_list,
        verbose=0,
    )
    elapsed = time.time() - t0

    probs = model.predict(X_te_in, verbose=0).ravel()
    auprc = float(average_precision_score(y_te, probs))
    auroc = float(roc_auc_score(y_te, probs))
    print(f"    [{tag}] AUPRC={auprc:.4f}  AUROC={auroc:.4f}  ({elapsed:.0f}s)")
    return auprc, auroc, probs


# ─── Phase 2: Test-set CV with Splicing (BiGRU) ───────────────────────────────
def build_splice_head(emb_dim=64, sd_dim=150):
    """Splicing BiGRU head on top of frozen embedding."""
    inp_emb = layers.Input(shape=(emb_dim,), name='embedding')
    inp_sd  = layers.Input(shape=(sd_dim,), name='splicing')

    # BiGRU on splicing_delta (150 timesteps, each = one exon cluster delta)
    x_sd = layers.Reshape((sd_dim, 1))(inp_sd)
    x_sd = layers.Bidirectional(layers.GRU(32, return_sequences=False))(x_sd)
    x_sd = layers.Dense(32, activation='relu')(x_sd)

    x = layers.Concatenate()([inp_emb, x_sd])  # 64+32=96d
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid', name='pred')(x)
    return models.Model([inp_emb, inp_sd], out, name='splice_head')


def cv_auprc_with_splicing(embeddings, X_sd, y, groups, n_splits=5):
    """Gene-stratified CV: embedding + splicing_BiGRU → AUPRC."""
    gkf = GroupKFold(n_splits=n_splits)
    auprcs = []
    for tr, val in gkf.split(embeddings, y, groups):
        if y[val].sum() == 0:
            continue

        splice_model = build_splice_head(emb_dim=embeddings.shape[1], sd_dim=X_sd.shape[1])
        splice_model.compile(
            optimizer=tf.keras.optimizers.Adam(5e-4),
            loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
        )
        cw = get_class_weight(y[tr])
        splice_model.fit(
            [embeddings[tr], X_sd[tr]], y[tr],
            epochs=40, batch_size=256, class_weight=cw,
            verbose=0,
        )
        probs = splice_model.predict([embeddings[val], X_sd[val]], verbose=0).ravel()
        auprcs.append(float(average_precision_score(y[val], probs)))
        K.clear_session()
        tf.random.set_seed(SEED)
    return float(np.mean(auprcs)) if auprcs else 0.0


def cv_auprc_emb_only(embeddings, y, groups, n_splits=5):
    """Gene-stratified CV: embedding only → AUPRC (Phase 2 baseline)."""
    from sklearn.linear_model import LogisticRegression
    gkf = GroupKFold(n_splits=n_splits)
    auprcs = []
    for tr, val in gkf.split(embeddings, y, groups):
        if y[val].sum() == 0:
            continue
        sc = StandardScaler()
        clf = LogisticRegression(class_weight='balanced', C=1.0,
                                 solver='lbfgs', max_iter=1000)
        clf.fit(sc.fit_transform(embeddings[tr]), y[tr])
        probs = clf.predict_proba(sc.transform(embeddings[val]))[:, 1]
        auprcs.append(float(average_precision_score(y[val], probs)))
    return float(np.mean(auprcs)) if auprcs else 0.0


# ─── Main Experiment Loop ─────────────────────────────────────────────────────
def run_go_term(go_term):
    print(f"\n{'=' * 65}")
    print(f"  {go_term}")
    print(f"{'=' * 65}")

    y_tr, y_te, pos_genes = load_labels(go_term)
    print(f"  Train pos={int(y_tr.sum())}, Test pos={int(y_te.sum())}")

    if y_tr.sum() < 2 or y_te.sum() == 0:
        print("  SKIP: insufficient positives")
        return None

    gene_groups = np.array(X_te_genesymbol)
    results = {'go': go_term, 'n_pos_train': int(y_tr.sum()), 'n_pos_test': int(y_te.sum())}

    # Scale domain_delta for Dense branches
    sc_esm  = StandardScaler()
    sc_dd   = StandardScaler()
    X_tr_esm_s = sc_esm.fit_transform(X_tr_esm2)
    X_tr_dd_s  = sc_dd.fit_transform(X_tr_dd)
    X_te_esm_s = sc_esm.transform(X_te_esm2)
    X_te_dd_s  = sc_dd.transform(X_te_dd)

    # ── v10-A: ESM-2 + domain_delta_CNN ──────────────────────────────────────
    print("\n  [v10-A] ESM-2 + domain_delta_Conv1D (train→test)")
    K.clear_session(); tf.random.set_seed(SEED)
    model_A = build_model_A()
    auprc_A, auroc_A, probs_A = train_eval_model(
        model_A,
        [X_tr_esm_s, X_tr_dd_s], [X_te_esm_s, X_te_dd_s],
        y_tr, y_te, tag='v10-A'
    )
    results['v10A_auprc'] = auprc_A
    results['v10A_auroc'] = auroc_A

    # Extract embeddings for Phase 2
    embed_model_A = models.Model(model_A.input,
                                  model_A.get_layer('embedding').output)
    te_emb_A = embed_model_A.predict([X_te_esm_s, X_te_dd_s], verbose=0)

    # ── v10-B: ESM-2 only (ablation) ─────────────────────────────────────────
    print("\n  [v10-B] ESM-2 only (ablation, train→test)")
    K.clear_session(); tf.random.set_seed(SEED)
    model_B = build_model_B()
    auprc_B, auroc_B, probs_B = train_eval_model(
        model_B,
        X_tr_esm_s, X_te_esm_s,
        y_tr, y_te, tag='v10-B'
    )
    results['v10B_auprc'] = auprc_B
    results['v10B_auroc'] = auroc_B

    embed_model_B = models.Model(model_B.input,
                                  model_B.get_layer('embedding').output)
    te_emb_B = embed_model_B.predict(X_te_esm_s, verbose=0)

    # ── v10-C: ESM-2 + domain_delta Dense (ablation) ─────────────────────────
    print("\n  [v10-C] ESM-2 + domain_delta_Dense (ablation, train→test)")
    K.clear_session(); tf.random.set_seed(SEED)
    model_C = build_model_C()
    auprc_C, auroc_C, probs_C = train_eval_model(
        model_C,
        [X_tr_esm_s, X_tr_dd_s], [X_te_esm_s, X_te_dd_s],
        y_tr, y_te, tag='v10-C'
    )
    results['v10C_auprc'] = auprc_C
    results['v10C_auroc'] = auroc_C

    # ── v10-D: embedding + splicing_delta_BiGRU (test CV) ────────────────────
    print("\n  [v10-D] splicing_BiGRU (test CV on top of v10-A embedding)")
    K.clear_session(); tf.random.set_seed(SEED)
    # Baseline: embedding only CV
    cv_emb = cv_auprc_emb_only(te_emb_A, y_te, gene_groups)
    print(f"    [v10-D baseline] emb-only CV AUPRC={cv_emb:.4f}")
    # With splicing
    cv_splice = cv_auprc_with_splicing(te_emb_A, X_te_sd, y_te, gene_groups)
    print(f"    [v10-D splice  ] emb+BiGRU CV AUPRC={cv_splice:.4f}")
    results['v10D_cv_emb']    = cv_emb
    results['v10D_cv_splice'] = cv_splice

    return results


# ─── Argument Parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('go', nargs='?', default='all',
                    help='GO term or "all" (default: all)')
args = parser.parse_args()

if args.go.upper() == 'ALL':
    target_terms = GO_TERMS
else:
    target_terms = [args.go]

# ─── Run ──────────────────────────────────────────────────────────────────────
all_results = []
for go in target_terms:
    res = run_go_term(go)
    if res:
        all_results.append(res)

# ─── Summary ──────────────────────────────────────────────────────────────────
if all_results:
    print("\n" + "=" * 75)
    print(" SUMMARY")
    print("=" * 75)
    print(f"{'GO Term':<15} {'v10-A':>7} {'v10-B':>7} {'v10-C':>7} {'D-emb':>7} {'D-spli':>7} {'LR_ref':>7} {'D1_MLP':>7}")
    print("-" * 75)

    macros = {k: [] for k in ['v10A', 'v10B', 'v10C', 'D_emb', 'D_spli']}
    for r in all_results:
        go = r['go']
        lr  = BASELINES['LR'].get(go, 0.0)
        d1  = BASELINES['D1_MLP'].get(go, 0.0)
        print(f"{go:<15} {r['v10A_auprc']:>7.4f} {r['v10B_auprc']:>7.4f} {r['v10C_auprc']:>7.4f} "
              f"{r['v10D_cv_emb']:>7.4f} {r['v10D_cv_splice']:>7.4f} {lr:>7.4f} {d1:>7.4f}")
        macros['v10A'].append(r['v10A_auprc'])
        macros['v10B'].append(r['v10B_auprc'])
        macros['v10C'].append(r['v10C_auprc'])
        macros['D_emb'].append(r['v10D_cv_emb'])
        macros['D_spli'].append(r['v10D_cv_splice'])

    print("-" * 75)
    print(f"{'Macro':<15} "
          f"{np.mean(macros['v10A']):>7.4f} "
          f"{np.mean(macros['v10B']):>7.4f} "
          f"{np.mean(macros['v10C']):>7.4f} "
          f"{np.mean(macros['D_emb']):>7.4f} "
          f"{np.mean(macros['D_spli']):>7.4f} "
          f"{BASELINES['LR']['Macro']:>7.4f} "
          f"{BASELINES['D1_MLP']['Macro']:>7.4f}")

    print("\n Ablation interpretation:")
    v10A_m = np.mean(macros['v10A'])
    v10B_m = np.mean(macros['v10B'])
    v10C_m = np.mean(macros['v10C'])
    D_spl  = np.mean(macros['D_spli'])
    D_emb  = np.mean(macros['D_emb'])
    print(f"  domain_CNN contribution (A-B):  {v10A_m - v10B_m:+.4f}")
    print(f"  Conv1D vs Dense (A-C):          {v10A_m - v10C_m:+.4f}")
    print(f"  splicing_BiGRU contribution (D_spli-D_emb): {D_spl - D_emb:+.4f}")
    print(f"  v10-A vs LR:    {v10A_m - BASELINES['LR']['Macro']:+.4f}")
    print(f"  v10-A vs D1_MLP:{v10A_m - BASELINES['D1_MLP']['Macro']:+.4f}")
    print(f"  v10-A vs v8b:   {v10A_m - 0.3568:+.4f}")

    # Save results
    ts = time.strftime('%Y%m%d_%H%M')
    out_path = f'{OUT_DIR}/v10_mlp_results_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'results': all_results,
            'macros': {k: float(np.mean(v)) for k, v in macros.items()},
            'baselines': BASELINES,
            'timestamp': ts,
        }, f, indent=2)
    print(f"\n[Saved] {out_path}")
