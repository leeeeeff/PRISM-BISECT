"""
v12_sequential.py — Sequential Pre-training (Gradient Conflict 원천 차단)
=========================================================================
근거: v11-Slim(λ=0.1) AUPRC=0.42 → canonical aux가 GO gradient와 inherent conflict.
동시 학습으로는 해결 불가 → 시간적 분리로 conflict 원천 차단.

Phase A: Backbone + Canonical head 학습 (GO loss=0)
         → backbone이 isoform-distinct embedding 습득
Phase B: backbone freeze → GO head만 학습
         → canonical gradient가 backbone에 도달하지 않음

기대:
  PASS: AUPRC ≥ 0.67 (v10-B ±0.05) + bias > 0.10
  FAIL: bias ≈ v10-B → gene-level annotation으로 isoform resolution 불가 확정

빠른 진단: Ca2+ signaling / Motor activity / Glycolysis (3 terms, 5 seeds)
"""

import os, sys, json, time, gzip
from collections import defaultdict
import numpy as np
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
tf.get_logger().setLevel('ERROR')
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        target = gpus[1] if len(gpus) > 1 else gpus[0]
        tf.config.set_visible_devices(target, 'GPU')
        print(f"  GPU: {target.name}")
    except RuntimeError as e:
        print(f"  GPU config: {e}")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v12_sequential'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
# 3-term 빠른 진단: 대표 패턴 각 1개
DIAG_TERMS = {
    'GO:0007204': 'Ca2+ signaling',    # 전형적 AUPRC 붕괴 케이스
    'GO:0003774': 'Motor activity',    # 소규모 하락 케이스
    'GO:0006096': 'Glycolysis',        # AUPRC 개선 케이스
}
V10B_AUPRC = {
    'GO:0007204': 0.7761,
    'GO:0003774': 0.8198,
    'GO:0006096': 0.7938,
}
V10B_BIAS = {
    'GO:0007204': 0.0856,
    'GO:0003774': 0.0897,
    'GO:0006096': 0.0965,
}
V11SLIM_AUPRC = {
    'GO:0007204': 0.5092,
    'GO:0003774': 0.7389,
    'GO:0006096': 0.8134,
}

SEEDS          = [42, 123, 456, 789, 1024]
N_BOOT         = 500
EPOCHS_A       = 80    # Phase A: canonical pre-training
EPOCHS_B       = 120   # Phase B: GO head fine-tuning
PATIENCE_A     = 15
PATIENCE_B     = 10

# ─── Utilities ────────────────────────────────────────────────────────────────
def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

def compute_bias(scores, gene_ids):
    g2i = defaultdict(list)
    for i, g in enumerate(gene_ids):
        g2i[g].append(i)
    gstd = scores.std()
    if gstd < 1e-10:
        return np.nan
    within = [scores[v].std() for v in g2i.values() if len(v) >= 2]
    return float(np.mean(within) / gstd) if within else np.nan

def bootstrap_ci(y_true, y_score, gene_ids, n_boot=N_BOOT, seed=42):
    rng = np.random.RandomState(seed)
    genes = np.unique(gene_ids)
    g2i = {g: np.array([i for i,gg in enumerate(gene_ids) if gg==g]) for g in genes}
    base = float(average_precision_score(y_true, y_score))
    boot = []
    for _ in range(n_boot):
        gs = rng.choice(genes, size=len(genes), replace=True)
        idx = np.concatenate([g2i[g] for g in gs])
        if y_true[idx].sum() == 0:
            continue
        boot.append(average_precision_score(y_true[idx], y_score[idx]))
    boot = np.array(boot)
    return base, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def interpret_bias(b):
    if np.isnan(b): return 'N/A'
    if b >= 0.15:   return 'ISOFORM-SPEC'
    if b >= 0.10:   return 'MIXED'
    return 'GENE-LEVEL'

def bce_focal(gamma=2.0):
    return tf.keras.losses.BinaryFocalCrossentropy(gamma=gamma, from_logits=False)

def get_cw(y):
    n = max(int(y.sum()), 1)
    return {0: 1.0, 1: int((y == 0).sum()) / n}

# ─── Data ────────────────────────────────────────────────────────────────────
print("=" * 65)
print(" v12-Sequential — Loading features")
print("=" * 65)

X_tr_esm = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te_esm = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

te_isoid  = load_ids('my_isoform_list_fixed.npy')
te_geneid = load_ids('my_gene_list_fixed.npy')
tr_isoid  = load_ids(f'{ID_DIR}/train_isoform_list.npy')
tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

te_sym  = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_geneid]
tr_sym  = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_geneid]
te_genes = np.array([g.split('.')[0] for g in te_geneid])
te_gbase = [g.split('.')[0] for g in te_geneid]

print(f"  Train: ESM-2 {X_tr_esm.shape}")
print(f"  Test:  ESM-2 {X_te_esm.shape}")

# ─── Canonical Labels (MANE Select 우선순위) ──────────────────────────────────
print("\n[Canonical Labels]")

canon_ref = {}
with open(f'{FEAT_DIR}/canonical_reference.tsv') as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 4:
            try: canon_ref[p[0].split('.')[0]] = int(p[3])
            except: pass

y_te_canon   = np.full(len(te_isoid), -1.0, dtype=np.float32)
y_te_canon_w = np.zeros(len(te_isoid), dtype=np.float32)
te_g2i = defaultdict(list)
for i, g in enumerate(te_gbase): te_g2i[g].append(i)
for gb, ci in canon_ref.items():
    idxs = te_g2i.get(gb, [])
    if len(idxs) < 2 or ci not in range(len(te_isoid)): continue
    for i in idxs:
        y_te_canon[i] = 1.0 if i == ci else 0.0
        y_te_canon_w[i] = 1.0

mane_sym_to_nm = {}
mane_path = f'{DATA_DIR}/MANE_summary.txt.gz'
if os.path.exists(mane_path):
    with gzip.open(mane_path, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            p = line.strip().split('\t')
            if len(p) >= 10 and p[9] == 'MANE Select':
                mane_sym_to_nm[p[3]] = p[5].split('.')[0]
    print(f"  MANE Select: {len(mane_sym_to_nm)} entries")

X_tr_dm = None
dm_path = f'{FEAT_DIR}/domain_matrix_proper_train.npy'
if os.path.exists(dm_path):
    X_tr_dm = np.load(dm_path).astype(np.float32)

tr_iso_base = [nm.split('.')[0] for nm in tr_isoid]
gene_to_nms = defaultdict(list)
for i, (iso_b, sym) in enumerate(zip(tr_iso_base, tr_sym)):
    n_dom = int((X_tr_dm[i] > 0).sum()) if X_tr_dm is not None else 0
    gene_to_nms[sym].append((iso_b, i, n_dom))

y_tr_canon   = np.full(len(tr_isoid), -1.0, dtype=np.float32)
y_tr_canon_w = np.zeros(len(tr_isoid), dtype=np.float32)
tr_src = {'MANE': 0, 'domain': 0, 'skip': 0}
for sym, iso_list in gene_to_nms.items():
    if len(iso_list) < 2: continue
    nm_bases = [x[0] for x in iso_list]
    mane_nm = mane_sym_to_nm.get(sym)
    if mane_nm and mane_nm in nm_bases:
        can_i = iso_list[nm_bases.index(mane_nm)][1]; src = 'MANE'
    else:
        mx = max(x[2] for x in iso_list)
        if mx > 0: can_i = max(iso_list, key=lambda x: x[2])[1]; src = 'domain'
        else: tr_src['skip'] += 1; continue
    tr_src[src] += 1
    for _, i, _ in iso_list:
        y_tr_canon[i] = 1.0 if i == can_i else 0.0
        y_tr_canon_w[i] = 1.0

print(f"  Train: {int((y_tr_canon_w>0).sum())} labeled  MANE={tr_src['MANE']}")

# ─── Scaling ──────────────────────────────────────────────────────────────────
sc = StandardScaler()
sc.fit(X_tr_esm)
X_tr_sc = sc.transform(X_tr_esm).astype(np.float32)
X_te_sc = sc.transform(X_te_esm).astype(np.float32)

# canonical class weight
can_mask = y_tr_canon_w > 0
n_pos_c  = int((y_tr_canon[can_mask] == 1).sum())
n_neg_c  = int((y_tr_canon[can_mask] == 0).sum())
can_ratio = n_neg_c / max(n_pos_c, 1)
sw_can    = y_tr_canon_w.copy()
sw_can    = np.where(y_tr_canon == 1.0, sw_can * can_ratio, sw_can)

# ─── Model Builders ───────────────────────────────────────────────────────────
ESM_DIM = X_tr_esm.shape[1]

def build_backbone():
    """Shared backbone — v10-B와 동일 구조."""
    inp = layers.Input(shape=(ESM_DIM,), name='esm2')
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(128, activation='relu', name='backbone_out')(x)
    return models.Model(inp, out, name='backbone')

def build_canonical_head(backbone):
    """Phase A: backbone → canonical prediction."""
    inp = backbone.input
    feat = backbone.output
    x = layers.Dense(32, activation='relu')(feat)
    out = layers.Dense(1, activation='sigmoid', name='canonical_pred')(x)
    return models.Model(inp, out, name='phase_a_model')

def build_go_head(backbone):
    """Phase B: frozen backbone → GO prediction."""
    inp = backbone.input
    feat = backbone.output  # backbone is frozen at this point
    x = layers.Dropout(0.2)(feat)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid', name='go_pred')(x)
    return models.Model(inp, out, name='phase_b_model')

def load_go_labels(go_id):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) > 1 and go_id in p[1:]: pos.add(p[0])
    return (np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32),
            np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32))

# ─── Main Loop ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f" Sequential Pre-training: {len(DIAG_TERMS)} terms × {len(SEEDS)} seeds")
print(f" Phase A: canonical pre-training ({EPOCHS_A} ep, patience={PATIENCE_A})")
print(f" Phase B: GO head fine-tuning   ({EPOCHS_B} ep, patience={PATIENCE_B})")
print("=" * 65)

header = (f"{'GO Term':<22} {'v10-B':>6} {'vSlim':>6} {'v12':>6} {'Δ(v12-v10B)':>12}"
          f" | {'bias_v10B':>9} {'bias_v12':>8} {'Δbias':>6}  Interp")
print(f"\n{header}")
print("-" * 90)

results = {}
ts = time.strftime('%Y%m%d_%H%M')

for go_id, go_name in DIAG_TERMS.items():
    t0 = time.time()
    y_tr, y_te = load_go_labels(go_id)
    if y_tr.sum() == 0 or y_te.sum() == 0:
        print(f"  {go_name}: skipped"); continue

    cw = get_cw(y_tr)
    sw_go = np.where(y_tr == 1, cw[1], cw[0]).astype(np.float32)

    all_preds = []

    for seed in SEEDS:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        K.clear_session()

        # ── Phase A: Canonical Pre-training ─────────────────────────────────
        backbone = build_backbone()
        model_a  = build_canonical_head(backbone)
        model_a.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=bce_focal(gamma=2.0),
        )
        cb_a = [
            callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE_A,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=7, min_lr=1e-5, verbose=0),
        ]
        model_a.fit(
            X_tr_sc, np.clip(y_tr_canon, 0, 1),
            sample_weight=sw_can,
            epochs=EPOCHS_A, batch_size=256,
            validation_split=0.1, callbacks=cb_a, verbose=0,
        )
        t_a = time.time() - t0

        # ── Phase B: Freeze backbone, train GO head ──────────────────────────
        # backbone 가중치 고정
        backbone.trainable = False

        model_b = build_go_head(backbone)
        model_b.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=bce_focal(gamma=2.0),
        )
        cb_b = [
            callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE_B,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=5, min_lr=1e-5, verbose=0),
        ]
        model_b.fit(
            X_tr_sc, y_tr,
            sample_weight=sw_go,
            epochs=EPOCHS_B, batch_size=256,
            validation_split=0.1, callbacks=cb_b, verbose=0,
        )

        preds = model_b.predict(X_te_sc, verbose=0, batch_size=512).flatten()
        all_preds.append(preds)

    scores   = np.mean(all_preds, axis=0)
    auprc    = float(average_precision_score(y_te, scores))
    ci       = bootstrap_ci(y_te, scores, te_genes)
    bias     = compute_bias(scores, te_gbase)
    elapsed  = time.time() - t0
    interp   = interpret_bias(bias)
    v10b     = V10B_AUPRC[go_id]
    slim     = V11SLIM_AUPRC[go_id]
    d_auprc  = auprc - v10b
    d_bias   = bias - V10B_BIAS[go_id]

    print(f"{go_name:<22} {v10b:>6.4f} {slim:>6.4f} {auprc:>6.4f} {d_auprc:>+12.4f}"
          f" | {V10B_BIAS[go_id]:>9.4f} {bias:>8.4f} {d_bias:>+6.4f}  {interp:<13}"
          f" ({elapsed:.0f}s)")
    sys.stdout.flush()

    results[go_id] = {
        'go_name': go_name, 'auprc_v12': round(auprc, 4),
        'auprc_ci': [round(ci[1],4), round(ci[2],4)],
        'auprc_v10b': v10b, 'auprc_slim': slim,
        'delta_auprc': round(d_auprc, 4),
        'bias_v12': round(bias, 4), 'bias_v10b': V10B_BIAS[go_id],
        'delta_bias': round(d_bias, 4), 'bias_interp': interp,
        'elapsed_s': round(elapsed, 1),
    }

# ─── Summary ─────────────────────────────────────────────────────────────────
auprcs = [r['auprc_v12'] for r in results.values()]
biases = [r['bias_v12']  for r in results.values()]
v10b_m = np.mean([r['auprc_v10b'] for r in results.values()])
bias_m = np.mean([r['bias_v10b'] for r in results.values()])

print("\n" + "=" * 65)
print(f"  v10-B mean AUPRC={v10b_m:.4f}  v12-Seq={np.mean(auprcs):.4f}"
      f"  Δ={np.mean(auprcs)-v10b_m:+.4f}")
print(f"  v10-B mean bias={bias_m:.4f}   v12-Seq={np.mean(biases):.4f}"
      f"  Δ={np.mean(biases)-bias_m:+.4f}")

# 판정
auprc_ok = np.mean(auprcs) >= v10b_m - 0.05
bias_ok  = np.mean(biases) > bias_m + 0.01
if auprc_ok and bias_ok:
    verdict = "PASS: AUPRC 유지 + bias 개선 → sequential pre-training 유효"
elif auprc_ok and not bias_ok:
    verdict = "AUPRC 유지 but bias 미개선 → frozen backbone이 isoform 구분 못함 → annotation paradigm limitation 확정"
elif not auprc_ok and bias_ok:
    verdict = "TRADE-OFF (v11-Slim과 동일 패턴) → sequential도 동일 근본 문제"
else:
    verdict = "FAIL → backbone freeze가 GO 성능까지 저하 (예상치 못한 결과)"
print(f"  판정: {verdict}")
print("=" * 65)

out = {'results': results, 'timestamp': ts,
       'strategy': 'v12-Sequential: Phase-A canonical pretrain + Phase-B frozen GO'}
with open(f'{OUT_DIR}/v12_seq_{ts}.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved → {OUT_DIR}/v12_seq_{ts}.json")
