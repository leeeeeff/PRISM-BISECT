"""
v10_isoform_delta_model.py — Isoform-Specific Multimodal Model
===============================================================
핵심 설계 원칙:
  - ESM-2 절댓값 대신 canonical-differential (ESM-2 Δ) 사용 → gene-level bias 제거
  - domain_delta (proper binary, 512d): Pfam presence/absence 변화
  - splicing_delta (150d): exon inclusion 변화
  - 모든 feature가 "canonical 대비 차이" → isoform-specific 정보만 인코딩

Ablation variants:
  v10-Full : ESM2-Δ + domain_CNN + splice_Dense
  v10-NoDD : ESM2-Δ only (domain_delta 기여 분리)
  v10-NoSD : ESM2-Δ + domain_CNN (splice_delta 기여 분리)
  v10-Abs  : ESM2-abs + domain_CNN + splice_Dense (Δ vs abs 기여 분리)

실행:
  conda activate isoform_env
  python hMuscle/model/v10_isoform_delta_model.py [all|GO:XXXXXXX]
"""

import os, sys, json, time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import argparse

tf.get_logger().setLevel('ERROR')
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        target = gpus[1] if len(gpus) > 1 else gpus[0]
        tf.config.set_visible_devices(target, 'GPU')
        print(f"  Using {target.name}")
    except RuntimeError as e:
        print(f"  GPU config: {e}")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v10_isoform_delta'
os.makedirs(OUT_DIR, exist_ok=True)

GO_TERMS = [
    'GO:0006096',  # Glycolysis
    'GO:0003774',  # Motor Activity
    'GO:0007204',  # Ca2+ Signaling
    'GO:0030017',  # Sarcomere
    'GO:0006941',  # Muscle Contraction
]

BASELINES = {
    'LR':   0.5615,
    'v10B': 0.7177,  # ESM-2 abs MLP (이전 세션 확립)
}

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─── Feature Loading ──────────────────────────────────────────────────────────
def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


print("=" * 65)
print(" v10-IsoformDelta — Loading features")
print("=" * 65)

# ── Test ESM-2 (absolute, delta 계산에 사용) ──────────────────────
X_te_esm2_abs = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
X_tr_esm2_abs = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)

# ── ESM-2 delta 계산: iso - canonical ────────────────────────────
# Test: canonical_reference.tsv (MANE Select, 12,709 genes)
X_te_iso   = load_ids('my_isoform_list_fixed.npy')
X_te_genes = load_ids('my_gene_list_fixed.npy')
X_te_gbase = [g.split('.')[0] for g in X_te_genes]

canon_ref = {}
with open(f'{FEAT_DIR}/canonical_reference.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 4:
            try:
                canon_ref[p[0]] = int(p[3])
            except ValueError:
                pass

N_te = len(X_te_iso)
X_te_esm2_delta = np.zeros_like(X_te_esm2_abs)
n_canon_found = 0
for i, gb in enumerate(X_te_gbase):
    cidx = canon_ref.get(gb)
    if cidx is not None and cidx < N_te:
        X_te_esm2_delta[i] = X_te_esm2_abs[i] - X_te_esm2_abs[cidx]
        n_canon_found += 1

print(f"  Test ESM-2 delta: {n_canon_found}/{N_te} computed from MANE canonical")

# Train: gene symbol → most-domain canonical (gene_list 기반)
X_tr_genes = load_ids(f'{ID_DIR}/train_gene_list.npy')
from collections import defaultdict

# Train 내에서 gene당 canonical = 첫 번째 등장 isoform (가장 낮은 NM 번호가 canonical에 가까움)
gene_first_idx = {}
for i, gene in enumerate(X_tr_genes):
    if gene not in gene_first_idx:
        gene_first_idx[gene] = i

N_tr = len(X_tr_genes)
X_tr_esm2_delta = np.zeros_like(X_tr_esm2_abs)
for i, gene in enumerate(X_tr_genes):
    cidx = gene_first_idx.get(gene, i)
    X_tr_esm2_delta[i] = X_tr_esm2_abs[i] - X_tr_esm2_abs[cidx]

print(f"  Train ESM-2 delta: computed (first-NM canonical per gene)")

# ── Domain delta (proper binary, 512d) ───────────────────────────
# Test: domain_delta_proper_test_v2.npy (37.2% coverage, r_nc=0.128)
# Train: train_domain_delta_hmmscan.npy if available, else train_domain_delta_proper.npy
X_te_dd = np.load(f'{FEAT_DIR}/domain_delta_proper_test_v2.npy').astype(np.float32)

hmmscan_train_dd = f'{FEAT_DIR}/train_domain_delta_hmmscan.npy'
cdd_train_dd     = f'{FEAT_DIR}/train_domain_delta_proper.npy'
if os.path.exists(hmmscan_train_dd):
    X_tr_dd = np.load(hmmscan_train_dd).astype(np.float32)
    print(f"  Train domain delta: hmmscan ({X_tr_dd.shape})")
else:
    X_tr_dd = np.load(cdd_train_dd).astype(np.float32)
    print(f"  Train domain delta: CDD-based ({X_tr_dd.shape}) [hmmscan pending]")

# ── Splicing delta (150d) ─────────────────────────────────────────
# Test: real (52.1% coverage); Train: zeros (no annotation available)
X_te_sd = np.load(f'{FEAT_DIR}/splicing/splicing_delta_v2.npy').astype(np.float32)
X_tr_sd = np.zeros((N_tr, X_te_sd.shape[1]), dtype=np.float32)

print(f"  Test:  ESM2-Δ {X_te_esm2_delta.shape}, domain_Δ {X_te_dd.shape}, splice_Δ {X_te_sd.shape}")
print(f"  Train: ESM2-Δ {X_tr_esm2_delta.shape}, domain_Δ {X_tr_dd.shape}, splice_Δ zeros {X_tr_sd.shape}")
print(f"  [NOTE] Train splice_Δ = zeros (test-time-only feature; contribution test via test-CV)")

# ── Gene symbols for label loading ────────────────────────────────
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

X_te_genesymbol = [ENSG2SYM.get(g, g) for g in X_te_gbase]


# ─── Label Loading ────────────────────────────────────────────────────────────
def load_labels(go_term):
    pos_genes = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos_genes.add(parts[0])
    y_te = np.array([1 if s in pos_genes else 0 for s in X_te_genesymbol], dtype=np.float32)
    y_tr = np.array([1 if g in pos_genes else 0 for g in X_tr_genes], dtype=np.float32)
    return y_tr, y_te


# ─── Model Builders ───────────────────────────────────────────────────────────
ESM_DIM = X_te_esm2_delta.shape[1]   # 640
DD_DIM  = X_te_dd.shape[1]            # 512
SD_DIM  = X_te_sd.shape[1]            # 150
EMB_DIM = 64


def esm_branch(inp, name='esm'):
    x = layers.Dense(256, activation='relu', name=f'{name}_d256')(inp)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.Dropout(0.3, name=f'{name}_do3')(x)
    x = layers.Dense(128, activation='relu', name=f'{name}_d128')(x)
    x = layers.Dropout(0.2, name=f'{name}_do2')(x)
    return x  # 128d


def domain_cnn_branch(inp, name='dd'):
    # 1D CNN: captures local Pfam gain/loss cluster patterns
    x = layers.Reshape((DD_DIM, 1), name=f'{name}_rs')(inp)
    x = layers.Conv1D(32, kernel_size=5, padding='same',
                      activation='relu', name=f'{name}_c5')(x)
    x = layers.Conv1D(16, kernel_size=3, padding='same',
                      activation='relu', name=f'{name}_c3')(x)
    x = layers.GlobalMaxPooling1D(name=f'{name}_gmp')(x)
    return x  # 16d


def splice_dense_branch(inp, name='sd'):
    x = layers.Dense(64, activation='relu', name=f'{name}_d64')(inp)
    x = layers.Dense(32, activation='relu', name=f'{name}_d32')(x)
    return x  # 32d


def build_full(use_delta=True, use_domain=True, use_splice=True):
    """v10-Full: ESM2-Δ + domain_CNN + splice_Dense"""
    inp_esm = layers.Input(shape=(ESM_DIM,), name='esm2')
    inp_dd  = layers.Input(shape=(DD_DIM,),  name='domain_delta')
    inp_sd  = layers.Input(shape=(SD_DIM,),  name='splice_delta')

    branches = [esm_branch(inp_esm)]
    if use_domain:
        branches.append(domain_cnn_branch(inp_dd))
    if use_splice:
        branches.append(splice_dense_branch(inp_sd))

    x = layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
    x = layers.Dense(EMB_DIM, activation='relu', name='fusion')(x)
    x = layers.Lambda(lambda t: K.l2_normalize(t, axis=-1), name='embedding')(x)
    out = layers.Dense(1, activation='sigmoid', name='pred')(x)

    inputs = [inp_esm, inp_dd, inp_sd]
    return models.Model(inputs, out, name='v10_Full')


def build_no_dd():
    """v10-NoDD: ESM2-Δ + splice_Dense (domain_delta 기여 측정)"""
    inp_esm = layers.Input(shape=(ESM_DIM,), name='esm2')
    inp_sd  = layers.Input(shape=(SD_DIM,),  name='splice_delta')

    x = layers.Concatenate()([esm_branch(inp_esm), splice_dense_branch(inp_sd)])
    x = layers.Dense(EMB_DIM, activation='relu', name='fusion')(x)
    x = layers.Lambda(lambda t: K.l2_normalize(t, axis=-1), name='embedding')(x)
    out = layers.Dense(1, activation='sigmoid', name='pred')(x)
    return models.Model([inp_esm, inp_sd], out, name='v10_NoDD')


def build_no_sd():
    """v10-NoSD: ESM2-Δ + domain_CNN (splice_delta 기여 측정)"""
    inp_esm = layers.Input(shape=(ESM_DIM,), name='esm2')
    inp_dd  = layers.Input(shape=(DD_DIM,),  name='domain_delta')

    x = layers.Concatenate()([esm_branch(inp_esm), domain_cnn_branch(inp_dd)])
    x = layers.Dense(EMB_DIM, activation='relu', name='fusion')(x)
    x = layers.Lambda(lambda t: K.l2_normalize(t, axis=-1), name='embedding')(x)
    out = layers.Dense(1, activation='sigmoid', name='pred')(x)
    return models.Model([inp_esm, inp_dd], out, name='v10_NoSD')


def build_abs():
    """v10-Abs: ESM2-abs + domain_CNN + splice_Dense (Δ vs abs 기여 측정)"""
    inp_esm = layers.Input(shape=(ESM_DIM,), name='esm2')
    inp_dd  = layers.Input(shape=(DD_DIM,),  name='domain_delta')
    inp_sd  = layers.Input(shape=(SD_DIM,),  name='splice_delta')

    x = layers.Concatenate()([
        esm_branch(inp_esm, name='esm_abs'),
        domain_cnn_branch(inp_dd),
        splice_dense_branch(inp_sd),
    ])
    x = layers.Dense(EMB_DIM, activation='relu', name='fusion')(x)
    x = layers.Lambda(lambda t: K.l2_normalize(t, axis=-1), name='embedding')(x)
    out = layers.Dense(1, activation='sigmoid', name='pred')(x)
    return models.Model([inp_esm, inp_dd, inp_sd], out, name='v10_Abs')


# ─── Training ─────────────────────────────────────────────────────────────────
def get_class_weight(y):
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    return {0: 1.0, 1: n_neg / max(n_pos, 1)}


def train_eval(model, X_tr_list, X_te_list, y_tr, y_te, tag, epochs=100):
    if y_tr.sum() < 2 or y_te.sum() == 0:
        return 0.0, 0.0, None

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
        metrics=['AUC'],
    )
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=12,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=6, verbose=0),
    ]

    rng = np.random.RandomState(SEED)
    n_val = max(int(len(y_tr) * 0.1), 100)
    val_idx = rng.choice(len(y_tr), size=n_val, replace=False)
    tr_idx  = np.setdiff1d(np.arange(len(y_tr)), val_idx)

    X_tr_fit = [x[tr_idx] for x in X_tr_list]
    X_val    = [x[val_idx] for x in X_tr_list]

    t0 = time.time()
    model.fit(
        X_tr_fit, y_tr[tr_idx],
        validation_data=(X_val, y_tr[val_idx]),
        epochs=epochs, batch_size=512,
        class_weight=get_class_weight(y_tr),
        callbacks=cb, verbose=0,
    )
    elapsed = time.time() - t0

    probs = model.predict(X_te_list, verbose=0).ravel()
    auprc = float(average_precision_score(y_te, probs))
    auroc = float(roc_auc_score(y_te, probs))
    print(f"    [{tag:12s}] AUPRC={auprc:.4f}  AUROC={auroc:.4f}  ({elapsed:.0f}s)")
    return auprc, auroc, probs


# ─── Scaler utilities ─────────────────────────────────────────────────────────
def fit_scalers():
    """StandardScaler는 ESM-2에만 적용. domain/splice delta는 이미 [-1,+1] 범위."""
    sc_esm = StandardScaler()
    sc_esm.fit(X_tr_esm2_delta)
    return sc_esm


sc_esm = fit_scalers()

X_tr_esm_s = sc_esm.transform(X_tr_esm2_delta)
X_te_esm_s = sc_esm.transform(X_te_esm2_delta)

# abs variant scaler (v10-Abs용)
sc_esm_abs = StandardScaler()
sc_esm_abs.fit(X_tr_esm2_abs)
X_tr_esm_abs_s = sc_esm_abs.transform(X_tr_esm2_abs)
X_te_esm_abs_s = sc_esm_abs.transform(X_te_esm2_abs)


# ─── Per-GO experiment ────────────────────────────────────────────────────────
def run_go(go_term):
    print(f"\n{'=' * 65}")
    print(f"  {go_term}")
    print(f"{'=' * 65}")

    y_tr, y_te = load_labels(go_term)
    print(f"  Train pos={int(y_tr.sum())}, Test pos={int(y_te.sum())}")
    if y_tr.sum() < 2 or y_te.sum() == 0:
        print("  SKIP")
        return None

    res = {'go': go_term, 'n_pos_train': int(y_tr.sum()), 'n_pos_test': int(y_te.sum())}

    # ── v10-Full: ESM2-Δ + domain_CNN + splice_Dense ─────────────
    print("\n  [v10-Full] ESM2-Δ + domain_CNN + splice_Dense")
    K.clear_session(); tf.random.set_seed(SEED)
    m = build_full()
    auprc, auroc, probs = train_eval(
        m,
        [X_tr_esm_s, X_tr_dd, X_tr_sd],
        [X_te_esm_s, X_te_dd, X_te_sd],
        y_tr, y_te, 'v10-Full'
    )
    res['full_auprc'] = auprc; res['full_auroc'] = auroc

    # ── v10-NoDD: ESM2-Δ + splice only (domain 기여 측정) ────────
    print("\n  [v10-NoDD] ESM2-Δ + splice_Dense (no domain)")
    K.clear_session(); tf.random.set_seed(SEED)
    m = build_no_dd()
    auprc, auroc, _ = train_eval(
        m,
        [X_tr_esm_s, X_tr_sd],
        [X_te_esm_s, X_te_sd],
        y_tr, y_te, 'v10-NoDD'
    )
    res['nodd_auprc'] = auprc

    # ── v10-NoSD: ESM2-Δ + domain only (splice 기여 측정) ────────
    print("\n  [v10-NoSD] ESM2-Δ + domain_CNN (no splice)")
    K.clear_session(); tf.random.set_seed(SEED)
    m = build_no_sd()
    auprc, auroc, _ = train_eval(
        m,
        [X_tr_esm_s, X_tr_dd],
        [X_te_esm_s, X_te_dd],
        y_tr, y_te, 'v10-NoSD'
    )
    res['nosd_auprc'] = auprc

    # ── v10-Abs: ESM2-abs + domain_CNN + splice (Δ 기여 측정) ────
    print("\n  [v10-Abs]  ESM2-abs + domain_CNN + splice_Dense")
    K.clear_session(); tf.random.set_seed(SEED)
    m = build_abs()
    auprc, auroc, _ = train_eval(
        m,
        [X_tr_esm_abs_s, X_tr_dd, X_tr_sd],
        [X_te_esm_abs_s, X_te_dd, X_te_sd],
        y_tr, y_te, 'v10-Abs'
    )
    res['abs_auprc'] = auprc

    return res


# ─── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('go', nargs='?', default='all')
args = parser.parse_args()
target = GO_TERMS if args.go.upper() == 'ALL' else [args.go]

# ─── Run ──────────────────────────────────────────────────────────────────────
all_results = []
for go in target:
    r = run_go(go)
    if r:
        all_results.append(r)

# ─── Summary ──────────────────────────────────────────────────────────────────
if all_results:
    print("\n" + "=" * 75)
    print(" SUMMARY")
    print("=" * 75)
    header = f"{'GO':<15} {'Full':>7} {'NoDD':>7} {'NoSD':>7} {'Abs':>7} {'v10B':>7} {'LR':>7}"
    print(header)
    print("-" * 75)

    cols = ['full', 'nodd', 'nosd', 'abs']
    macros = {c: [] for c in cols}
    for r in all_results:
        print(f"{r['go']:<15} "
              f"{r['full_auprc']:>7.4f} "
              f"{r['nodd_auprc']:>7.4f} "
              f"{r['nosd_auprc']:>7.4f} "
              f"{r['abs_auprc']:>7.4f} "
              f"{BASELINES['v10B']:>7.4f} "
              f"{BASELINES['LR']:>7.4f}")
        for c in cols:
            macros[c].append(r[f'{c}_auprc'])

    print("-" * 75)
    m_full = np.mean(macros['full'])
    m_nodd = np.mean(macros['nodd'])
    m_nosd = np.mean(macros['nosd'])
    m_abs  = np.mean(macros['abs'])
    print(f"{'Macro':<15} {m_full:>7.4f} {m_nodd:>7.4f} {m_nosd:>7.4f} "
          f"{m_abs:>7.4f} {BASELINES['v10B']:>7.4f} {BASELINES['LR']:>7.4f}")

    print(f"\n Ablation (Macro AUPRC):")
    print(f"  ESM2-Δ contribution   (Full - Abs):   {m_full - m_abs:+.4f}  [Δ vs abs]")
    print(f"  domain_delta contrib  (Full - NoDD):  {m_full - m_nodd:+.4f}  [domain CNN]")
    print(f"  splicing_delta contrib(Full - NoSD):  {m_full - m_nosd:+.4f}  [splice Dense]")
    print(f"  vs v10-B (ESM2-abs):                  {m_full - BASELINES['v10B']:+.4f}")
    print(f"  vs LR baseline:                        {m_full - BASELINES['LR']:+.4f}")
    print(f"\n  [NOTE] splice contribution은 train=zeros이므로 과소평가됨.")
    print(f"         hmmscan 완료 후 train_domain_delta_hmmscan.npy 사용 시 domain 기여 재측정 필요.")

    ts = time.strftime('%Y%m%d_%H%M')
    out_path = f'{OUT_DIR}/results_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'results': all_results,
            'macros': {c: float(np.mean(macros[c])) for c in cols},
            'baselines': BASELINES,
            'domain_source': 'hmmscan' if os.path.exists(f'{FEAT_DIR}/train_domain_delta_hmmscan.npy') else 'CDD',
            'timestamp': ts,
        }, f, indent=2)
    print(f"\n[Saved] {out_path}")
