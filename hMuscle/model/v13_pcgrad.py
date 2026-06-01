"""
v13_pcgrad.py — PCGrad (Projecting Conflicting Gradients)
==========================================================
목적: gradient conflict 가설 직접 검증
  v11-Slim TRADE-OFF 원인: GO(gene-level) ↔ Canonical(isoform-level) gradient 충돌
  PCGrad: 충돌 성분만 수학적으로 제거 (양방향 대칭)
    g_go_proj  = g_go  - min(g_go·g_can / ||g_can||², 0) × g_can
    g_can_proj = g_can - min(g_can·g_go / ||g_go||²,  0) × g_go
    backbone ← (g_go_proj + g_can_proj) / 2

아키텍처: v11-Slim과 완전 동일 (ESM-2 only)
  backbone: Dense(256→BN→Drop0.3→128)
  go_head:  Drop0.2 → Dense(64→1, sigmoid)
  can_head: Dense(32→1, sigmoid)

변경점:
  .fit() → custom GradientTape loop
  conflict_ratio (backbone 변수 중 dot < 0 비율) 매 epoch 추적

판정 기준:
  AUPRC > 0.65 → gradient conflict 주원인 확정 → Phase 2A (splice auxiliary)
  AUPRC < 0.50 → label space 문제 주원인     → Phase 2B (MIL)
  conflict_ratio > 0.5 → gradient 충돌이 실제로 심각함 확인
"""

import os, sys, json, time, gzip
from collections import defaultdict
import numpy as np
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
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
OUT_DIR   = '../../reports/v13_pcgrad'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
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

V10B_AUPRC = {
    'GO:0007204': 0.7761, 'GO:0030017': 0.7802, 'GO:0006941': 0.6407,
    'GO:0006914': 0.6964, 'GO:0043161': 0.7405, 'GO:0007519': 0.7035,
    'GO:0042692': 0.6733, 'GO:0055074': 0.7436, 'GO:0007005': 0.6989,
    'GO:0007517': 0.7299, 'GO:0032006': 0.6115, 'GO:0003774': 0.8198,
    'GO:0006096': 0.7938,
}
V10B_BIAS = {
    'GO:0007204': 0.0856, 'GO:0030017': 0.0573, 'GO:0006941': 0.0868,
    'GO:0006914': 0.1096, 'GO:0043161': 0.1042, 'GO:0007519': 0.1239,
    'GO:0042692': 0.1041, 'GO:0055074': 0.0717, 'GO:0007005': 0.1190,
    'GO:0007517': 0.0797, 'GO:0032006': 0.0912, 'GO:0003774': 0.0897,
    'GO:0006096': 0.0965,
}

MANE_FILE  = '../data/MANE_summary.txt.gz'
SEEDS      = [42, 123, 456, 789, 1024]
N_BOOT     = 500
PATIENCE   = 10
MAX_EPOCHS = 150
BATCH_SIZE = 256

# ─── Utilities (v11-Slim과 동일) ──────────────────────────────────────────────
def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

def compute_bias_score(scores, gene_ids):
    global_std = scores.std()
    if global_std < 1e-10:
        return np.nan
    g2i = defaultdict(list)
    for i, g in enumerate(gene_ids):
        g2i[g].append(i)
    within_stds = [scores[idxs].std() for idxs in g2i.values() if len(idxs) >= 2]
    return float(np.mean(within_stds) / global_std) if within_stds else np.nan

def bootstrap_ci(y_true, y_score, gene_ids, n_boot=N_BOOT, seed=42):
    rng = np.random.RandomState(seed)
    unique_genes = np.unique(gene_ids)
    base = float(average_precision_score(y_true, y_score))
    g2i = defaultdict(list)
    for i, g in enumerate(gene_ids):
        g2i[g].append(i)
    g2i = {g: np.array(idxs) for g, idxs in g2i.items()}
    boot = []
    for _ in range(n_boot):
        gs = rng.choice(unique_genes, size=len(unique_genes), replace=True)
        idx = np.concatenate([g2i[g] for g in gs])
        if y_true[idx].sum() == 0:
            continue
        boot.append(average_precision_score(y_true[idx], y_score[idx]))
    return base, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def interpret_bias(b):
    if np.isnan(b): return 'N/A'
    if b >= 0.15:   return 'ISOFORM-SPEC'
    if b >= 0.10:   return 'MIXED'
    return 'GENE-LEVEL'

def get_cw(y):
    n_pos = max(int(y.sum()), 1)
    return {0: 1.0, 1: int((y == 0).sum()) / n_pos}

# ─── Data Loading (v11-Slim과 동일) ──────────────────────────────────────────
print("=" * 65)
print(" v13-PCGrad — Loading features (ESM-2 only)")
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
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

te_sym      = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_geneid]
te_genebase = [g.split('.')[0] for g in te_geneid]
tr_sym      = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_geneid]
te_genes    = np.array(te_genebase)

print(f"  Train: ESM-2 {X_tr_esm.shape}")
print(f"  Test:  ESM-2 {X_te_esm.shape}")

# ─── Canonical Labels (v11-Slim과 동일) ──────────────────────────────────────
print("\n[Canonical Labels]")

canon_ref = {}
with open(f'{FEAT_DIR}/canonical_reference.tsv') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            try:
                canon_ref[parts[0].split('.')[0]] = int(parts[3])
            except (ValueError, IndexError):
                pass

y_te_canon   = np.full(len(te_isoid), -1.0, dtype=np.float32)
y_te_canon_w = np.zeros(len(te_isoid), dtype=np.float32)
te_gene2idxs = defaultdict(list)
for i, g in enumerate(te_genebase):
    te_gene2idxs[g].append(i)
for gene_b, can_idx in canon_ref.items():
    idxs = te_gene2idxs.get(gene_b, [])
    if len(idxs) < 2 or can_idx not in range(len(te_isoid)):
        continue
    for i in idxs:
        y_te_canon[i] = 1.0 if i == can_idx else 0.0
        y_te_canon_w[i] = 1.0

te_n_labeled = int((y_te_canon_w > 0).sum())
te_n_pos = int((y_te_canon == 1.0).sum())
print(f"  Test:  {te_n_labeled} labeled ({te_n_pos} canonical, {te_n_labeled-te_n_pos} alternative)")

mane_sym_to_nm = {}
if os.path.exists(MANE_FILE):
    with gzip.open(MANE_FILE, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.strip().split('\t')
            if len(p) < 10:
                continue
            sym, nm_ver, status = p[3], p[5], p[9]
            if status == 'MANE Select':
                mane_sym_to_nm[sym] = nm_ver.split('.')[0]
    print(f"  MANE Select entries loaded: {len(mane_sym_to_nm)}")

X_tr_dm_proxy = None
dm_path = f'{FEAT_DIR}/domain_matrix_proper_train.npy'
if os.path.exists(dm_path):
    X_tr_dm_proxy = np.load(dm_path).astype(np.float32)

tr_iso_base = [nm.split('.')[0] for nm in tr_isoid]
gene_to_nms = defaultdict(list)
for i, (iso_b, sym) in enumerate(zip(tr_iso_base, tr_sym)):
    n_dom = int((X_tr_dm_proxy[i] > 0).sum()) if X_tr_dm_proxy is not None else 0
    gene_to_nms[sym].append((iso_b, i, n_dom))

y_tr_canon   = np.full(len(tr_isoid), -1.0, dtype=np.float32)
y_tr_canon_w = np.zeros(len(tr_isoid), dtype=np.float32)
tr_src = {'MANE': 0, 'domain': 0, 'skip': 0}

for sym, iso_list in gene_to_nms.items():
    if len(iso_list) < 2:
        continue
    nm_bases = [iso_b for iso_b, _, _ in iso_list]
    mane_nm = mane_sym_to_nm.get(sym)
    if mane_nm and mane_nm in nm_bases:
        can_i = iso_list[nm_bases.index(mane_nm)][1]
        src = 'MANE'
    else:
        max_dom = max(n for _, _, n in iso_list)
        if max_dom > 0:
            can_i = max(iso_list, key=lambda x: x[2])[1]
            src = 'domain'
        else:
            tr_src['skip'] += 1
            continue
    tr_src[src] += 1
    for _, i, _ in iso_list:
        y_tr_canon[i] = 1.0 if i == can_i else 0.0
        y_tr_canon_w[i] = 1.0

tr_n_labeled = int((y_tr_canon_w > 0).sum())
tr_n_pos = int((y_tr_canon == 1.0).sum())
print(f"  Train: {tr_n_labeled} labeled ({tr_n_pos} canonical, {tr_n_labeled-tr_n_pos} alternative)")
print(f"    source: MANE={tr_src['MANE']}, domain-proxy={tr_src['domain']}, skip={tr_src['skip']}")

# ─── Label Loading ────────────────────────────────────────────────────────────
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

# ─── Model: Subclassing API (backbone/go_head/can_head 분리) ──────────────────
ESM_DIM = X_tr_esm.shape[1]

class DiffuseV13(tf.keras.Model):
    """
    v11-Slim과 동일한 아키텍처를 subclassing API로 재구현.
    backbone, go_head, can_head를 별도 submodel로 분리 →
    PCGrad에서 model.backbone.trainable_variables 참조 가능.
    """
    def __init__(self, esm_dim=640):
        super().__init__()
        # v11-Slim backbone과 동일
        self.backbone = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(esm_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
        ], name='backbone')
        # v11-Slim go_head와 동일
        self.go_head = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ], name='go_head')
        # v11-Slim can_head와 동일
        self.can_head = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ], name='can_head')

    def call(self, x, training=False):
        h = self.backbone(x, training=training)
        return self.go_head(h, training=training), self.can_head(h, training=training)

# ─── Loss Helper ──────────────────────────────────────────────────────────────
# BinaryFocalCrossentropy + sample_weight를 custom loop에서 사용할 때
# reduction='none' 후 수동 가중 평균이 shape 충돌 없이 안전함
_focal_none = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, reduction='none')

def weighted_focal(y_true, y_pred_flat, sample_weight):
    """
    y_pred_flat: (batch,) — 반드시 flatten된 상태
    y_true:      (batch,)
    sample_weight: (batch,)
    """
    per_sample = _focal_none(
        tf.expand_dims(y_true, -1),
        tf.expand_dims(y_pred_flat, -1)
    )  # → (batch,) 또는 (batch,1)
    per_sample = tf.reshape(per_sample, [-1])
    return tf.reduce_mean(per_sample * sample_weight)

# ─── PCGrad Core ──────────────────────────────────────────────────────────────
def project_gradient(g_src, g_ref):
    """
    g_src에서 g_ref와 충돌하는 성분을 제거.
    수식: g_src_proj = g_src - min(dot(g_src,g_ref)/||g_ref||², 0) × g_ref
    tf.minimum(x, 0): x < 0이면 x, x >= 0이면 0 → 충돌 시에만 제거
    """
    projected = []
    for g1, g2 in zip(g_src, g_ref):
        if g1 is None or g2 is None:
            projected.append(g1)
            continue
        g1_flat = tf.reshape(g1, [-1])
        g2_flat = tf.reshape(g2, [-1])
        dot    = tf.reduce_sum(g1_flat * g2_flat)
        g2_sq  = tf.reduce_sum(g2_flat * g2_flat) + 1e-12
        scale  = tf.minimum(dot / g2_sq, 0.0)  # 0이면 변화 없음
        projected.append(g1 - tf.reshape(scale * g2_flat, tf.shape(g1)))
    return projected

def conflict_ratio(g_go_bb, g_can_bb):
    """backbone 변수 중 gradient가 충돌(dot < 0)하는 비율."""
    conflicts, total = 0, 0
    for g1, g2 in zip(g_go_bb, g_can_bb):
        if g1 is not None and g2 is not None:
            dot = float(tf.reduce_sum(
                tf.reshape(g1, [-1]) * tf.reshape(g2, [-1])
            ).numpy())
            conflicts += int(dot < 0)
            total += 1
    return conflicts / total if total > 0 else 0.0

def pcgrad_step(model, optimizer, x_b, y_go_b, y_can_b,
                sw_go_b, sw_can_b):
    """
    단일 batch PCGrad update.
    1) persistent tape로 두 Loss의 gradient 분리 추출
    2) backbone gradient: 양방향 projection → 평균
    3) head gradient: 원본 유지
    4) apply_gradients
    """
    with tf.GradientTape(persistent=True) as tape:
        go_pred, can_pred = model(x_b, training=True)
        go_pred_flat  = tf.reshape(go_pred,  [-1])
        can_pred_flat = tf.reshape(can_pred, [-1])
        L_go  = weighted_focal(y_go_b,  go_pred_flat,  sw_go_b)
        L_can = weighted_focal(y_can_b, can_pred_flat, sw_can_b)

    bb_vars  = model.backbone.trainable_variables
    go_vars  = model.go_head.trainable_variables
    can_vars = model.can_head.trainable_variables

    # backbone gradient: 두 Loss에서 각각
    g_go_bb  = tape.gradient(L_go,  bb_vars)
    g_can_bb = tape.gradient(L_can, bb_vars)

    # head gradient: 각자 자신의 Loss만
    g_go_hd  = tape.gradient(L_go,  go_vars)
    g_can_hd = tape.gradient(L_can, can_vars)
    del tape  # persistent tape 수동 해제 필수

    # ── PCGrad projection (양방향 대칭) ──────────────────────────────────
    g_go_proj  = project_gradient(g_go_bb,  g_can_bb)
    g_can_proj = project_gradient(g_can_bb, g_go_bb)
    g_bb_final = [(a + b) / 2.0 for a, b in zip(g_go_proj, g_can_proj)]
    # ─────────────────────────────────────────────────────────────────────

    all_grads = g_bb_final + list(g_go_hd) + list(g_can_hd)
    all_vars  = list(bb_vars) + list(go_vars) + list(can_vars)
    optimizer.apply_gradients(zip(all_grads, all_vars))

    return float(L_go.numpy()), float(L_can.numpy()), g_go_bb, g_can_bb

# ─── Custom Training Loop ─────────────────────────────────────────────────────
def train_pcgrad(model, X_tr, y_go_tr, y_can_tr, sw_go, sw_can,
                 max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 val_frac=0.1, batch_size=BATCH_SIZE, go_id=''):
    """
    Returns: (epoch_stopped, mean_conflict_ratio_over_training)
    모델 weights는 best val GO loss로 restore됨.
    """
    n = len(X_tr)
    n_val = int(n * val_frac)
    n_tr  = n - n_val

    X_tr_, X_val_ = X_tr[:n_tr],       X_tr[n_tr:]
    y_go_, y_go_v = y_go_tr[:n_tr],    y_go_tr[n_tr:]
    y_can_, y_can_v = y_can_tr[:n_tr], y_can_tr[n_tr:]
    sw_go_, sw_go_v   = sw_go[:n_tr],  sw_go[n_tr:]
    sw_can_, sw_can_v = sw_can[:n_tr], sw_can[n_tr:]

    X_val_t   = tf.constant(X_val_,                              dtype=tf.float32)
    y_go_vt   = tf.constant(y_go_v,                              dtype=tf.float32)
    y_can_vt  = tf.constant(np.clip(y_can_v, 0, 1),             dtype=tf.float32)
    sw_go_vt  = tf.constant(sw_go_v,                             dtype=tf.float32)
    sw_can_vt = tf.constant(sw_can_v,                            dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    dataset = tf.data.Dataset.from_tensor_slices((
        X_tr_,
        y_go_,
        np.clip(y_can_, 0, 1).astype(np.float32),
        sw_go_.astype(np.float32),
        sw_can_.astype(np.float32),
    )).shuffle(n_tr, reshuffle_each_iteration=True).batch(batch_size)

    best_val  = np.inf
    best_w    = None
    no_improve = 0
    no_improve_lr = 0
    all_conflicts = []

    for ep in range(1, max_epochs + 1):
        epoch_conflicts = []

        for x_b, y_go_b, y_can_b, sw_go_b, sw_can_b in dataset:
            l_go, l_can, g_go_bb, g_can_bb = pcgrad_step(
                model, optimizer, x_b, y_go_b, y_can_b,
                sw_go_b, sw_can_b
            )
            epoch_conflicts.append(conflict_ratio(g_go_bb, g_can_bb))

        # Validation: GO loss만 monitor (v11-Slim과 동일)
        go_pred_v, _ = model(X_val_t, training=False)
        val_go = float(weighted_focal(
            y_go_vt, tf.reshape(go_pred_v, [-1]), sw_go_vt
        ).numpy())

        mean_cr = float(np.mean(epoch_conflicts))
        all_conflicts.append(mean_cr)

        # EarlyStopping
        if val_go < best_val - 1e-5:
            best_val   = val_go
            best_w     = model.get_weights()
            no_improve = 0
            no_improve_lr = 0
        else:
            no_improve    += 1
            no_improve_lr += 1

        # ReduceLROnPlateau (patience=5, factor=0.5)
        if no_improve_lr >= 5:
            cur_lr = float(optimizer.learning_rate.numpy())
            if cur_lr > 1e-5:
                optimizer.learning_rate.assign(cur_lr * 0.5)
            no_improve_lr = 0

        if no_improve >= patience:
            break

    if best_w is not None:
        model.set_weights(best_w)

    return ep, float(np.mean(all_conflicts))

# ─── Scaler ───────────────────────────────────────────────────────────────────
sc_esm = StandardScaler()
sc_esm.fit(X_tr_esm)
X_tr_sc = sc_esm.transform(X_tr_esm).astype(np.float32)
X_te_sc = sc_esm.transform(X_te_esm).astype(np.float32)

can_labeled_mask = (y_tr_canon_w > 0)
n_can_pos = int((y_tr_canon[can_labeled_mask] == 1).sum())
n_can_neg = int((y_tr_canon[can_labeled_mask] == 0).sum())
can_ratio_w = n_can_neg / max(n_can_pos, 1)

# ─── Training ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f" Training: {len(GO_TERMS)} GO terms × {len(SEEDS)} seeds")
print(f" Method:   PCGrad (GradientTape, symmetric projection)")
print(f" Arch:     ESM-2 only — identical to v11-Slim")
print("=" * 65)

ts = time.strftime('%Y%m%d_%H%M')
results = {}
auprc_list, bias_list = [], []

header = (f"{'GO Term':<22} {'v10B':>7} {'v13PCG':>7} {'Δ':>7} | "
          f"{'bias_v10B':>9} {'bias_PCG':>9} {'Δ':>6}  "
          f"{'conflict':>8}  Interp")
print(f"\n{header}")
print("-" * 88)

for go_id, go_name in GO_TERMS.items():
    t0 = time.time()
    y_tr, y_te = load_go_labels(go_id)

    if y_tr.sum() == 0 or y_te.sum() == 0:
        print(f"  {go_name}: skipped (no positives)")
        continue

    cw = get_cw(y_tr)
    sw_go  = np.where(y_tr == 1, cw[1], cw[0]).astype(np.float32)
    sw_can = y_tr_canon_w.copy()
    sw_can = np.where(y_tr_canon == 1.0, sw_can * can_ratio_w, sw_can).astype(np.float32)

    all_preds, all_can_preds, all_conflicts = [], [], []

    for seed in SEEDS:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        tf.keras.backend.clear_session()

        model = DiffuseV13(esm_dim=ESM_DIM)
        _ = model(tf.zeros((1, ESM_DIM), dtype=tf.float32), training=False)

        ep_stopped, mean_cr = train_pcgrad(
            model, X_tr_sc, y_tr, y_tr_canon, sw_go, sw_can,
            go_id=go_id,
        )
        all_conflicts.append(mean_cr)

        go_te, can_te = model(tf.constant(X_te_sc), training=False)
        all_preds.append(go_te.numpy().flatten())
        all_can_preds.append(can_te.numpy().flatten())

    pcg_scores = np.mean(all_preds,     axis=0)
    can_scores = np.mean(all_can_preds, axis=0)

    auprc    = float(average_precision_score(y_te, pcg_scores))
    auprc_ci = bootstrap_ci(y_te, pcg_scores, te_genes)
    bias     = compute_bias_score(pcg_scores, te_genebase)

    can_mask = y_te_canon_w > 0
    can_auc  = float(roc_auc_score(y_te_canon[can_mask], can_scores[can_mask])) \
               if can_mask.sum() > 10 else float('nan')

    mean_cr  = float(np.mean(all_conflicts))
    elapsed  = time.time() - t0
    d_auprc  = auprc - V10B_AUPRC.get(go_id, 0)
    d_bias   = bias  - V10B_BIAS.get(go_id, 0)
    interp   = interpret_bias(bias)

    print(f"{go_name:<22} {V10B_AUPRC.get(go_id,0):>7.4f} {auprc:>7.4f} {d_auprc:>+7.4f} | "
          f"{V10B_BIAS.get(go_id,0):>9.4f} {bias:>9.4f} {d_bias:>+6.4f}  "
          f"{mean_cr:>8.3f}  {interp:<14}  ({elapsed:.0f}s)")
    sys.stdout.flush()

    results[go_id] = {
        'go_name':          go_name,
        'n_pos_train':      int(y_tr.sum()),
        'n_pos_test':       int(y_te.sum()),
        'auprc_pcgrad':     round(auprc, 4),
        'auprc_ci_lo':      round(auprc_ci[1], 4),
        'auprc_ci_hi':      round(auprc_ci[2], 4),
        'auprc_v10b':       V10B_AUPRC.get(go_id),
        'delta_auprc':      round(d_auprc, 4),
        'bias_pcgrad':      round(bias, 4),
        'bias_v10b':        V10B_BIAS.get(go_id),
        'delta_bias':       round(d_bias, 4),
        'bias_interp':      interp,
        'canonical_auc':    round(can_auc, 4),
        'mean_conflict_ratio': round(mean_cr, 3),
        'elapsed_s':        round(elapsed, 1),
    }
    auprc_list.append(auprc)
    bias_list.append(bias)

# ─── Summary ──────────────────────────────────────────────────────────────────
macro_pcg      = float(np.mean(auprc_list))
macro_v10b     = float(np.mean(list(V10B_AUPRC.values())))
mean_bias_pcg  = float(np.nanmean(bias_list))
mean_bias_v10b = float(np.mean(list(V10B_BIAS.values())))
n_isospec = sum(1 for b in bias_list if b >= 0.15)
n_mixed   = sum(1 for b in bias_list if 0.10 <= b < 0.15)
n_genelev = sum(1 for b in bias_list if b < 0.10)
mean_cr_all = float(np.mean([v['mean_conflict_ratio'] for v in results.values()]))

print("\n" + "=" * 88)
print(f"  Macro AUPRC:      v10-B={macro_v10b:.4f}  v13-PCGrad={macro_pcg:.4f}  "
      f"Δ={macro_pcg-macro_v10b:+.4f}")
print(f"  Mean bias:        v10-B={mean_bias_v10b:.4f}  v13-PCGrad={mean_bias_pcg:.4f}  "
      f"Δ={mean_bias_pcg-mean_bias_v10b:+.4f}")
print(f"  Bias interp:      ISOFORM-SPEC={n_isospec}  MIXED={n_mixed}  GENE-LEVEL={n_genelev}")
print(f"  Conflict ratio:   {mean_cr_all:.3f}  "
      f"({'충돌 심각 — PCGrad 투영 효과 있었음' if mean_cr_all > 0.5 else 'label space 문제가 주원인'})")
print("=" * 88)

# ─── Save ─────────────────────────────────────────────────────────────────────
out = {
    'results': results,
    'summary': {
        'macro_auprc_pcgrad': round(macro_pcg, 4),
        'macro_auprc_v10b':   round(macro_v10b, 4),
        'delta_macro_auprc':  round(macro_pcg - macro_v10b, 4),
        'mean_bias_pcgrad':   round(mean_bias_pcg, 4),
        'mean_bias_v10b':     round(mean_bias_v10b, 4),
        'delta_mean_bias':    round(mean_bias_pcg - mean_bias_v10b, 4),
        'n_isospec':          n_isospec,
        'n_mixed':            n_mixed,
        'n_genelevel':        n_genelev,
        'mean_conflict_ratio_all': round(mean_cr_all, 3),
        'seeds':              SEEDS,
        'n_go_terms':         len(results),
        'architecture':       'ESM-2 only + PCGrad symmetric projection',
    },
    'timestamp': ts,
    'strategy':  'v13-PCGrad: GradientTape, symmetric projection, backbone average',
}

out_path = f'{OUT_DIR}/v13_pcgrad_{ts}.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved → {out_path}")

# ─── 판정 ─────────────────────────────────────────────────────────────────────
print("\n[v13-PCGrad 효과 판정]")
if macro_pcg >= 0.65:
    print(f"  PASS: AUPRC={macro_pcg:.4f} (≥ 0.65 기준)")
    print("  → gradient conflict가 주원인 확정. Phase 2A (splice-aware auxiliary) 진입")
elif macro_pcg >= 0.50:
    print(f"  PARTIAL: AUPRC={macro_pcg:.4f} (0.50~0.65)")
    print("  → 부분 효과. MIL + PCGrad 결합 검토")
else:
    print(f"  FAIL: AUPRC={macro_pcg:.4f} (< 0.50)")
    print("  → label space 문제 주원인 확정. Phase 2B (MIL) 전환")

bias_ok = mean_bias_pcg > mean_bias_v10b + 0.01
print(f"  Bias:     {'개선' if bias_ok else '미개선'} "
      f"({mean_bias_pcg:.4f} vs v10-B {mean_bias_v10b:.4f})")
print(f"  Conflict: {mean_cr_all:.3f} "
      f"({'> 0.5 — PCGrad 투영 효과 있었음' if mean_cr_all > 0.5 else '≤ 0.5 — gradient가 애초에 많이 충돌하지 않음'})")
