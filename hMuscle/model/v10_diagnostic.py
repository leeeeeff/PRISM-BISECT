"""
v10_diagnostic.py — Phase 1 진단 실험
======================================
세 가지 실험을 통해 병목을 확정한다.

  D1: ESM-2 640d → 2-layer MLP  (train→test, PFN/압축 없이)
  D2: ESM-2 Δ = ESM2(iso) − ESM2(canonical) → LR  (CV on test set)
  D3: ESM-2 Δ + domain_delta_sign + splicing_delta_v2 → LR  (CV on test set)

Gate 기준:
  D1_tt Macro ≥ 0.52 → PFN+압축이 병목 → v10-MLP 방향
  D2_cv > D1_cv + 0.02 → canonical-differential 유효
  D3_cv > D2_cv + 0.01 → delta 공간에서 D/S 기여 확인

실행:
  conda activate isoform_env
  python hMuscle/model/v10_diagnostic.py
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = '../data'
ANNOT_DIR  = '../data/raw_data/data/annotations/'
ID_DIR     = '../data/raw_data/data/id_lists/'
FEAT_DIR   = '../results_isoform/features'
OUT_DIR    = '../../reports/diagnostics'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── GO terms ─────────────────────────────────────────────────────────────────
GO_TERMS = [
    'GO:0006096',  # Glycolysis       (Type-A)
    'GO:0003774',  # Motor Activity   (Type-A)
    'GO:0007204',  # Ca2+ Signaling   (Type-B)
    'GO:0030017',  # Sarcomere        (Type-B)
    'GO:0006941',  # Muscle Contraction (Type-B)
]

LR_BASELINE = {
    'GO:0006096': 0.6949,
    'GO:0003774': 0.8253,
    'GO:0007204': 0.4138,
    'GO:0030017': 0.5609,
    'GO:0006941': 0.3124,
    'Macro':      0.5615,
}

V8B_BASELINE = {
    'GO:0006096': 0.7945,
    'GO:0003774': 0.5686,
    'GO:0007204': 0.1462,
    'GO:0030017': 0.1570,
    'GO:0006941': 0.1177,
    'Macro':      0.3568,
}

# ─── Load fixed data ──────────────────────────────────────────────────────────
print("=" * 65)
print(" v10 Diagnostic — Loading data ...")
print("=" * 65)

def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

X_test_esm2    = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
X_train_esm2   = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)

X_test_dd      = np.sign(np.load(f'{FEAT_DIR}/domain_delta_v2.npy')).astype(np.float32)
X_train_dd     = np.load(f'{FEAT_DIR}/train_domain_delta_sign.npy').astype(np.float32)
X_test_sd      = np.load(f'{FEAT_DIR}/splicing/splicing_delta_v2.npy').astype(np.float32)

X_test_geneid  = load_ids('my_gene_list_fixed.npy')
X_train_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')

# Ensembl → gene symbol mapping (needed for test set label lookup)
ENSG2SYM = {}
sym_map_path = f'{ID_DIR}/ensembl_to_symbol.txt'
with open(sym_map_path) as _f:
    next(_f)
    for _line in _f:
        _parts = _line.strip().split()
        if len(_parts) >= 5:
            ENSG2SYM[_parts[0]] = _parts[4]
print(f"  Ensembl→Symbol mapping: {len(ENSG2SYM)} entries")

# Canonical reference → gene_base → canonical_idx in test set
_ref = pd.read_csv(f'{FEAT_DIR}/canonical_reference.tsv', sep='\t')
canonical_map  = dict(zip(_ref['gene_base'].astype(str), _ref['canonical_iso_idx'].astype(int)))

# Test gene base IDs (strip version)
X_test_genebase  = [g.split('.')[0] for g in X_test_geneid]
# Test gene symbols (for annotation lookup)
X_test_genesymbol = [ENSG2SYM.get(g, g) for g in X_test_genebase]

# Build ESM-2 delta for test set: iso − canonical
print("[D2/D3] Computing ESM-2 delta (test set) ...")
esm2_delta_test = np.zeros_like(X_test_esm2)
n_mapped = 0
for i, gbase in enumerate(X_test_genebase):
    if gbase in canonical_map:
        cidx = canonical_map[gbase]
        esm2_delta_test[i] = X_test_esm2[i] - X_test_esm2[cidx]
        n_mapped += 1
    else:
        esm2_delta_test[i] = X_test_esm2[i]  # fallback: use absolute
print(f"  Mapped {n_mapped}/{len(X_test_geneid)} isoforms ({n_mapped/len(X_test_geneid)*100:.1f}%)")

# D3 full feature (delta + D + S)
X_test_D3 = np.hstack([esm2_delta_test, X_test_dd, X_test_sd])
print(f"  D3 feature dim: {X_test_D3.shape[1]} (640+251+150)")

print()

# ─── Annotation loading ───────────────────────────────────────────────────────
def load_annotations(go_term):
    """Build y_test and y_train from human_annotations.txt.
    Test set: Ensembl IDs → gene symbols via ENSG2SYM → compare with pos_genes (symbols).
    Train set: already gene symbols → compare directly.
    """
    pos_genes = set()
    ann_path = f'{ANNOT_DIR}/human_annotations.txt'
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos_genes.add(parts[0])  # gene symbol (no version stripping needed)
    # Test: use symbol mapping (X_test_genesymbol built from ENSG2SYM)
    y_test  = np.array([1 if sym in pos_genes else 0 for sym in X_test_genesymbol])
    # Train: symbols directly
    y_train = np.array([
        1 if g in pos_genes else 0
        for g in X_train_geneid
    ])
    return y_test, y_train, pos_genes

# ─── CV utility ───────────────────────────────────────────────────────────────
def cv_auprc(X, y, groups, n_splits=5):
    """Gene-stratified GroupKFold CV → mean AUPRC (replicates esm2_640dim_ablation logic)."""
    gkf = GroupKFold(n_splits=n_splits)
    auprcs = []
    for tr, val in gkf.split(X, y, groups):
        if y[val].sum() == 0:
            continue
        clf = LogisticRegression(class_weight='balanced', C=1.0,
                                 solver='lbfgs', max_iter=1000)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_val = scaler.transform(X[val])
        clf.fit(X_tr, y[tr])
        scores = clf.predict_proba(X_val)[:, 1]
        auprcs.append(average_precision_score(y[val], scores))
    return float(np.mean(auprcs)) if auprcs else 0.0

def train_test_lr(X_tr, y_tr, X_te, y_te, tag='LR'):
    """Standard train→test LR evaluation."""
    scaler = StandardScaler()
    clf = LogisticRegression(class_weight='balanced', C=1.0,
                             solver='lbfgs', max_iter=1000)
    clf.fit(scaler.fit_transform(X_tr), y_tr)
    scores = clf.predict_proba(scaler.transform(X_te))[:, 1]
    return float(average_precision_score(y_te, scores))

# ─── D1: MLP (train→test) ─────────────────────────────────────────────────────
def run_D1_mlp(y_test, y_train, go_tag):
    """
    Train a 2-layer MLP on human train set (31668×640) → evaluate on test set.
    Uses Keras with class_weight and early stopping.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    tf.random.set_seed(42)

    if y_train.sum() < 2:
        return 0.0

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_esm2)
    X_te = scaler.transform(X_test_esm2)

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    cw = {0: 1.0, 1: n_neg / max(n_pos, 1)}

    inp = layers.Input(shape=(640,))
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
        metrics=['AUC'],
    )

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=4, verbose=0),
    ]
    # Small validation split from train
    val_frac = 0.1
    n_val = max(int(len(X_tr) * val_frac), 100)
    perm = np.random.RandomState(42).permutation(len(X_tr))
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]

    model.fit(
        X_tr[tr_idx], y_train[tr_idx],
        validation_data=(X_tr[val_idx], y_train[val_idx]),
        epochs=60, batch_size=256,
        class_weight=cw,
        callbacks=cb,
        verbose=0,
    )

    scores = model.predict(X_te, batch_size=512, verbose=0).ravel()
    tf.keras.backend.clear_session()
    return float(average_precision_score(y_test, scores))

# ─── D1 CV (for apples-to-apples with D2/D3) ─────────────────────────────────
def run_D1_cv(y_test, groups):
    return cv_auprc(X_test_esm2, y_test, groups)

# ─── Main loop ────────────────────────────────────────────────────────────────
results = {}
groups = np.array(X_test_genebase)

print(f"{'GO Term':<15} {'D1_tt':>7} {'D1_cv':>7} {'D2_cv':>7} {'D3_cv':>7} "
      f"{'LR_ref':>7} {'v8b_ref':>7}")
print("-" * 65)

for go in GO_TERMS:
    t0 = time.time()
    y_test, y_train, pos_genes = load_annotations(go)
    n_pos = int(y_test.sum())

    if n_pos == 0:
        print(f"{go:<15}  NO POSITIVES — skipping")
        continue

    # D1: MLP train→test
    d1_tt = run_D1_mlp(y_test, y_train, go)

    # D1: LR CV on test (baseline for delta comparison)
    d1_cv = run_D1_cv(y_test, groups)

    # D2: ESM-2 delta → LR CV
    d2_cv = cv_auprc(esm2_delta_test, y_test, groups)

    # D3: ESM-2 delta + D + S → LR CV
    d3_cv = cv_auprc(X_test_D3, y_test, groups)

    elapsed = time.time() - t0
    results[go] = {
        'n_pos': n_pos,
        'D1_tt': d1_tt,
        'D1_cv': d1_cv,
        'D2_cv': d2_cv,
        'D3_cv': d3_cv,
    }
    print(f"{go:<15} {d1_tt:7.4f} {d1_cv:7.4f} {d2_cv:7.4f} {d3_cv:7.4f} "
          f"{LR_BASELINE.get(go, 0):7.4f} {V8B_BASELINE.get(go, 0):7.4f}  "
          f"({n_pos} pos, {elapsed:.0f}s)")

# ─── Summary & Gate judgement ─────────────────────────────────────────────────
print()
print("=" * 65)
print(" SUMMARY")
print("=" * 65)

d1_tt_vals = [results[g]['D1_tt'] for g in GO_TERMS if g in results]
d1_cv_vals = [results[g]['D1_cv'] for g in GO_TERMS if g in results]
d2_cv_vals = [results[g]['D2_cv'] for g in GO_TERMS if g in results]
d3_cv_vals = [results[g]['D3_cv'] for g in GO_TERMS if g in results]

macro_d1_tt = float(np.mean(d1_tt_vals)) if d1_tt_vals else 0.0
macro_d1_cv = float(np.mean(d1_cv_vals)) if d1_cv_vals else 0.0
macro_d2_cv = float(np.mean(d2_cv_vals)) if d2_cv_vals else 0.0
macro_d3_cv = float(np.mean(d3_cv_vals)) if d3_cv_vals else 0.0

results['Macro'] = {
    'D1_tt': macro_d1_tt,
    'D1_cv': macro_d1_cv,
    'D2_cv': macro_d2_cv,
    'D3_cv': macro_d3_cv,
}

print(f"{'Metric':<20} {'D1_tt':>7} {'D1_cv':>7} {'D2_cv':>7} {'D3_cv':>7} {'LR_ref':>7}")
print("-" * 60)
print(f"{'Macro-AUPRC':<20} {macro_d1_tt:7.4f} {macro_d1_cv:7.4f} "
      f"{macro_d2_cv:7.4f} {macro_d3_cv:7.4f} {LR_BASELINE['Macro']:7.4f}")
print(f"{'vs LR (%)':<20} {macro_d1_tt/LR_BASELINE['Macro']*100:6.1f}% "
      f"{macro_d1_cv/LR_BASELINE['Macro']*100:6.1f}%  "
      f"{macro_d2_cv/LR_BASELINE['Macro']*100:6.1f}%  "
      f"{macro_d3_cv/LR_BASELINE['Macro']*100:6.1f}%")

print()
print("=" * 65)
print(" GATE JUDGEMENT")
print("=" * 65)

# Gate 1: PFN+압축 병목 여부
gate1 = macro_d1_tt >= 0.52
print(f"[Gate 1] D1_tt Macro = {macro_d1_tt:.4f} (threshold ≥ 0.52)")
if gate1:
    print("  → PASS: PFN+압축이 병목. v10-MLP (PFN 교체) 방향 채택")
else:
    print("  → FAIL: 표현 자체 문제 지속. v10-PFN (입력만 교체) 방향 검토")

# Gate 2: canonical-differential 유효성
d2_gain = macro_d2_cv - macro_d1_cv
gate2 = d2_gain > 0.02
print(f"\n[Gate 2] D2_cv − D1_cv = {d2_gain:+.4f} (threshold > +0.02)")
if gate2:
    print("  → PASS: ESM-2 Δ 표현이 gene-bias 제거에 유효. ESM-2 delta 입력 전환 확정")
else:
    print(f"  → FAIL: delta 표현 이득 미미 (Δ={d2_gain:+.4f}). 다른 gene-bias 해소 방법 검토 필요")

# Gate 3: D/S feature delta-space 기여
d3_gain = macro_d3_cv - macro_d2_cv
gate3 = d3_gain > 0.01
print(f"\n[Gate 3] D3_cv − D2_cv = {d3_gain:+.4f} (threshold > +0.01)")
if gate3:
    print("  → PASS: delta 공간에서 D/S 기여 확인. BiGRU/Conv1D 통합 Phase 2 진행")
else:
    print(f"  → FAIL: D/S 기여 없음 (Δ={d3_gain:+.4f}). pLDDT_delta 등 새 feature 설계 필요")

print()
print("=" * 65)
print(" RECOMMENDED NEXT STEP")
print("=" * 65)
if gate1 and gate2 and gate3:
    print("→ v10-MLP 전체 통합 (PFN 교체 + ESM-2 Δ + BiGRU/Conv1D D/S)")
elif gate1 and gate2:
    print("→ v10-MLP (PFN 교체 + ESM-2 Δ) + D/S feature 재설계 병렬 진행")
elif gate1:
    print("→ v10-MLP (PFN 교체) + canonical-differential 재검토")
elif gate2 and gate3:
    print("→ v10-PFN (PFN 유지 + ESM-2 Δ + BiGRU/Conv1D)")
else:
    print("→ Phase 2 feature 보강 우선 (pLDDT_delta, SpliceAI 등)")

# ─── Save results ─────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'v10_diagnostic_results.json')
with open(out_path, 'w') as f:
    json.dump({
        'results': results,
        'gates': {
            'gate1_PFN_bottleneck': gate1,
            'gate2_delta_valid': gate2,
            'gate3_DS_contributes': gate3,
        },
        'meta': {
            'D1_method': 'ESM-2 640d → 2-layer MLP, train→test (human-only)',
            'D2_method': 'ESM-2 delta → LR, 5-fold gene-stratified CV',
            'D3_method': 'ESM-2 delta + domain_delta_sign + splicing_delta_v2 → LR, 5-fold CV',
        }
    }, f, indent=2)
print(f"\n[Saved] {out_path}")
