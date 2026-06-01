"""
v13_mil.py — Multiple Instance Learning GO Loss
================================================
근본 원인 해결:
  현재 (v10-B, v11-Slim): L_GO = focal(y_gene, f(x_i))  for ALL isoforms i of gene G
    → 동일 유전자 아이소폼 전체에 동일 target → backbone이 within-gene 차이를 학습할 동기 없음
    → Canonical aux와 방향 충돌 (Canonical: 아이소폼 분리, GO: 수렴)

  MIL: L_GO = focal(y_gene, max_{i ∈ G} f(x_i))
    → Gene G에서 "가장 기능적인" 아이소폼 1개만 GO gradient 수신
    → 나머지 아이소폼: GO gradient 없음 → Canonical aux가 자유롭게 분리 학습
    → 충돌 구조 해소: Canonical head가 "max isoform = canonical"로 수렴하도록 MIL이 돕게 됨

구현:
  tf.math.unsorted_segment_max(preds, gene_seg_ids, n_genes) — 미분 가능 max-pooling
  gradient: max 아이소폼에만 1, 나머지 0 (straight-through)

아키텍처: v11-Slim과 동일
  ESM-2(640) → backbone(256→BN→Drop0.3→128) → go_head(Drop0.2→64→1) + can_head(32→1)

진단 지표 (PCGrad 비교용):
  conflict_ratio: MIL에서 GO↔Canonical gradient 충돌이 감소하는지 확인
  → 충돌 감소 확인 시 MIL이 label space 충돌을 실제로 해소한다는 증거
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
        target = gpus[0]  # PCGrad가 GPU:1 사용 중 → GPU:0으로 분리 실행
        tf.config.set_visible_devices(target, 'GPU')
        print(f"  GPU: {target.name}")
    except RuntimeError as e:
        print(f"  GPU config: {e}")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v13_mil'
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
LAMBDA_AUX = 0.1   # v11-Slim과 동일
PATIENCE   = 10
MAX_EPOCHS = 150
BATCH_SIZE = 256

# ─── Utilities ────────────────────────────────────────────────────────────────
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

# ─── Data Loading ─────────────────────────────────────────────────────────────
print("=" * 65)
print(" v13-MIL — Loading features (ESM-2 only)")
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

# train gene ID: MIL pooling에 사용 (string → int segment ID로 변환)
tr_genebase = [g.split('.')[0] for g in tr_geneid]

print(f"  Train: ESM-2 {X_tr_esm.shape}")
print(f"  Test:  ESM-2 {X_te_esm.shape}")

# train gene → integer mapping (전체 dataset 기준, batch 내에서 재매핑)
tr_unique_genes = list(dict.fromkeys(tr_genebase))
print(f"  Train genes: {len(tr_unique_genes)}, isoforms: {len(tr_genebase)}")
print(f"  Avg isoforms/gene: {len(tr_genebase)/len(tr_unique_genes):.1f}")

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
    mane_nm  = mane_sym_to_nm.get(sym)
    if mane_nm and mane_nm in nm_bases:
        can_i = iso_list[nm_bases.index(mane_nm)][1]
        src   = 'MANE'
    else:
        max_dom = max(n for _, _, n in iso_list)
        if max_dom > 0:
            can_i = max(iso_list, key=lambda x: x[2])[1]
            src   = 'domain'
        else:
            tr_src['skip'] += 1
            continue
    tr_src[src] += 1
    for _, i, _ in iso_list:
        y_tr_canon[i] = 1.0 if i == can_i else 0.0
        y_tr_canon_w[i] = 1.0

print(f"  Test:  {int((y_te_canon_w>0).sum())} labeled ({int((y_te_canon==1.0).sum())} canonical)")
print(f"  Train: {int((y_tr_canon_w>0).sum())} labeled ({int((y_tr_canon==1.0).sum())} canonical)")
print(f"    MANE={tr_src['MANE']}, domain-proxy={tr_src['domain']}, skip={tr_src['skip']}")

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

# ─── Model (v11-Slim과 동일 아키텍처, subclassing for variable separation) ────
ESM_DIM = X_tr_esm.shape[1]

class DiffuseMIL(tf.keras.Model):
    """v11-Slim 아키텍처와 동일. backbone/go_head/can_head 분리 (conflict_ratio 측정용)."""
    def __init__(self, esm_dim=640):
        super().__init__()
        self.backbone = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(esm_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
        ], name='backbone')
        self.go_head = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ], name='go_head')
        self.can_head = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ], name='can_head')

    def call(self, x, training=False):
        h = self.backbone(x, training=training)
        return self.go_head(h, training=training), self.can_head(h, training=training)

# ─── Loss Helpers ─────────────────────────────────────────────────────────────
_focal_none = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, reduction='none')

def weighted_focal(y_true, y_pred_flat, sample_weight):
    """isoform-level weighted focal loss (canonical aux용)."""
    per_sample = _focal_none(
        tf.expand_dims(y_true, -1),
        tf.expand_dims(y_pred_flat, -1)
    )
    return tf.reduce_mean(tf.reshape(per_sample, [-1]) * sample_weight)

def mil_focal_loss(go_preds_flat, y_go, sw_go, gene_ids_batch):
    """
    MIL GO loss: gene 단위 max-pooling 후 focal loss.

    핵심 수식: L_GO = mean_G[ focal(y_G, max_{i∈G} f(x_i)) ]

    구현: tf.math.unsorted_segment_max — 미분 가능
      gradient: max를 취한 아이소폼에만 흘러감 (나머지 아이소폼 gradient=0)
      → GO loss가 동일 유전자 아이소폼 전체를 수렴시키는 압력 제거
    """
    # gene ID → integer segment ID (batch 내 unique gene 기준)
    if hasattr(gene_ids_batch, 'numpy'):
        gids = gene_ids_batch.numpy()
    else:
        gids = np.asarray(gene_ids_batch)

    if gids.dtype.kind in ('S', 'O') or (len(gids) > 0 and isinstance(gids[0], bytes)):
        gids = [g.decode() if isinstance(g, bytes) else g for g in gids]

    unique_g  = list(dict.fromkeys(gids))   # insertion-order unique
    g2int     = {g: i for i, g in enumerate(unique_g)}
    seg_ids   = tf.constant([g2int[g] for g in gids], dtype=tf.int32)
    n_g       = len(unique_g)

    # Max-pool prediction per gene (미분 가능)
    gene_preds = tf.math.unsorted_segment_max(go_preds_flat, seg_ids, n_g)  # (n_g,)

    # Gene-level label: 동일 유전자는 모두 같은 label이므로 max = 해당 유전자 label
    gene_y = tf.math.unsorted_segment_max(y_go, seg_ids, n_g)   # (n_g,)

    # Class weight: gene 내 평균 (사실 모두 동일, mean으로 안전하게)
    gene_w = tf.math.unsorted_segment_mean(sw_go, seg_ids, n_g)  # (n_g,)

    per_gene = _focal_none(
        tf.expand_dims(gene_y, -1),
        tf.expand_dims(gene_preds, -1)
    )  # (n_g, 1)
    per_gene = tf.reshape(per_gene, [-1])  # (n_g,)
    return tf.reduce_mean(per_gene * gene_w)

# ─── Conflict Ratio (진단용 — PCGrad 결과와 비교) ────────────────────────────
def measure_conflict_ratio(model, x_b, y_go_b, y_can_b, sw_go_b, sw_can_b, gene_ids_b):
    """MIL GO loss ↔ Canonical loss의 backbone gradient 충돌 비율 측정."""
    with tf.GradientTape(persistent=True) as tape:
        go_pred, can_pred = model(x_b, training=True)
        go_flat  = tf.reshape(go_pred,  [-1])
        can_flat = tf.reshape(can_pred, [-1])
        L_mil = mil_focal_loss(go_flat, y_go_b, sw_go_b, gene_ids_b)
        L_can = weighted_focal(y_can_b, can_flat, sw_can_b)

    bb_vars = model.backbone.trainable_variables
    g_mil   = tape.gradient(L_mil, bb_vars)
    g_can   = tape.gradient(L_can, bb_vars)
    del tape

    conflicts, total = 0, 0
    for g1, g2 in zip(g_mil, g_can):
        if g1 is not None and g2 is not None:
            dot = float(tf.reduce_sum(
                tf.reshape(g1, [-1]) * tf.reshape(g2, [-1])
            ).numpy())
            conflicts += int(dot < 0)
            total     += 1
    return conflicts / total if total > 0 else 0.0

# ─── Training Step ────────────────────────────────────────────────────────────
def mil_train_step(model, optimizer, x_b, y_go_b, y_can_b,
                   sw_go_b, sw_can_b, gene_ids_b, lambda_aux):
    with tf.GradientTape() as tape:
        go_pred, can_pred = model(x_b, training=True)
        go_flat  = tf.reshape(go_pred,  [-1])
        can_flat = tf.reshape(can_pred, [-1])

        # ── MIL: max-pool per gene → focal loss ──────────────────────────────
        L_mil = mil_focal_loss(go_flat, y_go_b, sw_go_b, gene_ids_b)

        # ── Canonical auxiliary (isoform-level, 변경 없음) ────────────────────
        L_can = weighted_focal(y_can_b, can_flat, sw_can_b)

        total_loss = L_mil + lambda_aux * L_can

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return float(L_mil.numpy()), float(L_can.numpy())

# ─── Custom Training Loop ─────────────────────────────────────────────────────
def train_mil(model, X_tr, y_go_tr, y_can_tr, sw_go, sw_can,
              tr_gene_ids,           # gene ID per training sample (for MIL grouping)
              max_epochs=MAX_EPOCHS, patience=PATIENCE,
              val_frac=0.1, batch_size=BATCH_SIZE,
              lambda_aux=LAMBDA_AUX, measure_conflict_every=20):
    """
    Returns: (epoch_stopped, mean_conflict_ratio)
    conflict_ratio: v13-PCGrad (~0.49)와 비교하는 핵심 진단 지표
    """
    n     = len(X_tr)
    n_val = int(n * val_frac)
    n_tr  = n - n_val

    X_tr_   = X_tr[:n_tr];     X_val_  = X_tr[n_tr:]
    y_go_   = y_go_tr[:n_tr];  y_go_v  = y_go_tr[n_tr:]
    y_can_  = y_can_tr[:n_tr]; y_can_v = y_can_tr[n_tr:]
    sw_go_  = sw_go[:n_tr];    sw_go_v = sw_go[n_tr:]
    sw_can_ = sw_can[:n_tr];   sw_can_v= sw_can[n_tr:]
    gids_   = tr_gene_ids[:n_tr]

    X_val_t   = tf.constant(X_val_,                  dtype=tf.float32)
    y_go_vt   = tf.constant(y_go_v,                  dtype=tf.float32)
    y_can_vt  = tf.constant(np.clip(y_can_v, 0, 1),  dtype=tf.float32)
    sw_go_vt  = tf.constant(sw_go_v,                 dtype=tf.float32)
    sw_can_vt = tf.constant(sw_can_v,                dtype=tf.float32)
    gids_v    = gids_[n_tr - n_tr:][:0]              # val에서는 MIL 불필요 (예측만)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # gene IDs를 bytes로 통일
    gids_bytes = np.array([g.encode() if isinstance(g, str) else g for g in gids_])

    dataset = tf.data.Dataset.from_tensor_slices((
        X_tr_,
        y_go_,
        np.clip(y_can_, 0, 1).astype(np.float32),
        sw_go_.astype(np.float32),
        sw_can_.astype(np.float32),
        gids_bytes,
    )).shuffle(n_tr, reshuffle_each_iteration=True).batch(batch_size)

    best_val   = np.inf
    best_w     = None
    no_improve = 0
    no_improve_lr = 0
    all_conflicts = []

    for ep in range(1, max_epochs + 1):

        # conflict 측정: 매 measure_conflict_every epoch마다 첫 배치에서
        ep_conflict = None

        for x_b, y_go_b, y_can_b, sw_go_b, sw_can_b, gids_b in dataset:
            if ep_conflict is None and ep % measure_conflict_every == 0:
                ep_conflict = measure_conflict_ratio(
                    model, x_b, y_go_b, y_can_b, sw_go_b, sw_can_b, gids_b
                )
            mil_train_step(
                model, optimizer, x_b, y_go_b, y_can_b,
                sw_go_b, sw_can_b, gids_b, lambda_aux
            )

        if ep_conflict is not None:
            all_conflicts.append(ep_conflict)

        # Validation GO loss
        go_pred_v, _ = model(X_val_t, training=False)
        val_go = float(weighted_focal(
            y_go_vt, tf.reshape(go_pred_v, [-1]), sw_go_vt
        ).numpy())

        if val_go < best_val - 1e-5:
            best_val      = val_go
            best_w        = model.get_weights()
            no_improve    = 0
            no_improve_lr = 0
        else:
            no_improve    += 1
            no_improve_lr += 1

        if no_improve_lr >= 5:
            cur_lr = float(optimizer.learning_rate.numpy())
            if cur_lr > 1e-5:
                optimizer.learning_rate.assign(cur_lr * 0.5)
            no_improve_lr = 0

        if no_improve >= patience:
            break

    if best_w is not None:
        model.set_weights(best_w)

    mean_cr = float(np.mean(all_conflicts)) if all_conflicts else float('nan')
    return ep, mean_cr

# ─── Scaler & shared weights ──────────────────────────────────────────────────
sc_esm = StandardScaler()
sc_esm.fit(X_tr_esm)
X_tr_sc = sc_esm.transform(X_tr_esm).astype(np.float32)
X_te_sc = sc_esm.transform(X_te_esm).astype(np.float32)

can_labeled_mask = (y_tr_canon_w > 0)
n_can_pos  = int((y_tr_canon[can_labeled_mask] == 1).sum())
n_can_neg  = int((y_tr_canon[can_labeled_mask] == 0).sum())
can_ratio_w = n_can_neg / max(n_can_pos, 1)

# gene ID array for training (MIL grouping)
tr_gene_ids_arr = np.array(tr_genebase)

# ─── Training ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f" Training: {len(GO_TERMS)} GO terms × {len(SEEDS)} seeds")
print(f" Loss: MIL (max-pool GO) + λ={LAMBDA_AUX} Canonical")
print(f" MIL: tf.math.unsorted_segment_max per gene in batch")
print(f" Conflict ratio: 측정 every 20 epochs (PCGrad ~0.49와 비교)")
print("=" * 70)

ts = time.strftime('%Y%m%d_%H%M')
results = {}
auprc_list, bias_list = [], []

header = (f"{'GO Term':<22} {'v10B':>7} {'v13MIL':>7} {'Δ':>7} | "
          f"{'bias_v10B':>9} {'bias_MIL':>9} {'Δ':>6}  "
          f"{'conflict':>8}  Interp")
print(f"\n{header}")
print("-" * 90)

for go_id, go_name in GO_TERMS.items():
    t0 = time.time()
    y_tr, y_te = load_go_labels(go_id)

    if y_tr.sum() == 0 or y_te.sum() == 0:
        print(f"  {go_name}: skipped (no positives)")
        continue

    cw     = get_cw(y_tr)
    sw_go  = np.where(y_tr == 1, cw[1], cw[0]).astype(np.float32)
    sw_can = y_tr_canon_w.copy()
    sw_can = np.where(y_tr_canon == 1.0, sw_can * can_ratio_w, sw_can).astype(np.float32)

    all_preds, all_can_preds, all_conflicts = [], [], []

    for seed in SEEDS:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        tf.keras.backend.clear_session()

        model = DiffuseMIL(esm_dim=ESM_DIM)
        _ = model(tf.zeros((1, ESM_DIM), dtype=tf.float32), training=False)

        ep_stopped, mean_cr = train_mil(
            model, X_tr_sc, y_tr, y_tr_canon, sw_go, sw_can,
            tr_gene_ids_arr,
            lambda_aux=LAMBDA_AUX,
        )
        all_conflicts.append(mean_cr)

        go_te, can_te = model(tf.constant(X_te_sc), training=False)
        all_preds.append(go_te.numpy().flatten())
        all_can_preds.append(can_te.numpy().flatten())

    mil_scores = np.mean(all_preds,     axis=0)
    can_scores = np.mean(all_can_preds, axis=0)

    auprc    = float(average_precision_score(y_te, mil_scores))
    auprc_ci = bootstrap_ci(y_te, mil_scores, te_genes)
    bias     = compute_bias_score(mil_scores, te_genebase)

    can_mask = y_te_canon_w > 0
    can_auc  = float(roc_auc_score(y_te_canon[can_mask], can_scores[can_mask])) \
               if can_mask.sum() > 10 else float('nan')

    mean_cr  = float(np.nanmean(all_conflicts))
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
        'auprc_mil':        round(auprc, 4),
        'auprc_ci_lo':      round(auprc_ci[1], 4),
        'auprc_ci_hi':      round(auprc_ci[2], 4),
        'auprc_v10b':       V10B_AUPRC.get(go_id),
        'delta_auprc':      round(d_auprc, 4),
        'bias_mil':         round(bias, 4),
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
macro_mil      = float(np.mean(auprc_list))
macro_v10b     = float(np.mean(list(V10B_AUPRC.values())))
mean_bias_mil  = float(np.nanmean(bias_list))
mean_bias_v10b = float(np.mean(list(V10B_BIAS.values())))
n_isospec = sum(1 for b in bias_list if b >= 0.15)
n_mixed   = sum(1 for b in bias_list if 0.10 <= b < 0.15)
n_genelev = sum(1 for b in bias_list if b < 0.10)
mean_cr_all = float(np.nanmean([v.get('mean_conflict_ratio', float('nan'))
                                for v in results.values()]))

print("\n" + "=" * 90)
print(f"  Macro AUPRC:    v10-B={macro_v10b:.4f}  v13-MIL={macro_mil:.4f}  "
      f"Δ={macro_mil-macro_v10b:+.4f}")
print(f"  Mean bias:      v10-B={mean_bias_v10b:.4f}  v13-MIL={mean_bias_mil:.4f}  "
      f"Δ={mean_bias_mil-mean_bias_v10b:+.4f}")
print(f"  Bias interp:    ISOFORM-SPEC={n_isospec}  MIXED={n_mixed}  GENE-LEVEL={n_genelev}")
print(f"  Conflict ratio: {mean_cr_all:.3f}  "
      f"(PCGrad 기준 ~0.49 — 감소 시 MIL이 충돌 구조 해소 확인)")
print("=" * 90)

out = {
    'results': results,
    'summary': {
        'macro_auprc_mil':   round(macro_mil, 4),
        'macro_auprc_v10b':  round(macro_v10b, 4),
        'delta_macro_auprc': round(macro_mil - macro_v10b, 4),
        'mean_bias_mil':     round(mean_bias_mil, 4),
        'mean_bias_v10b':    round(mean_bias_v10b, 4),
        'delta_mean_bias':   round(mean_bias_mil - mean_bias_v10b, 4),
        'n_isospec':         n_isospec,
        'n_mixed':           n_mixed,
        'n_genelevel':       n_genelev,
        'mean_conflict_ratio_all': round(mean_cr_all, 3),
        'lambda_aux':        LAMBDA_AUX,
        'seeds':             SEEDS,
        'n_go_terms':        len(results),
        'architecture':      'ESM-2 only + MIL GO loss + Canonical aux',
    },
    'timestamp': ts,
    'strategy':  'v13-MIL: unsorted_segment_max per-gene + λ=0.1 canonical',
}

out_path = f'{OUT_DIR}/v13_mil_{ts}.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved → {out_path}")

# ─── 판정 ─────────────────────────────────────────────────────────────────────
print("\n[v13-MIL 효과 판정]")
auprc_ok = macro_mil >= macro_v10b - 0.02
bias_ok  = mean_bias_mil > mean_bias_v10b + 0.01

if auprc_ok and bias_ok:
    print(f"  PASS: AUPRC={macro_mil:.4f} (≥ v10-B-0.02={macro_v10b-0.02:.4f}) "
          f"+ bias 개선 ({mean_bias_mil:.4f})")
    print("  → MIL 성공: label space 문제 해결 확정")
    print("  → 다음: splice-aware auxiliary 추가 (Phase 2A)")
elif bias_ok and not auprc_ok:
    print(f"  PARTIAL: bias 개선 ({mean_bias_mil:.4f}) but AUPRC Δ={macro_mil-macro_v10b:+.4f}")
    print("  → MIL GO loss 자체는 방향 맞음, AUPRC 회복 위해 추가 조정 필요")
    print("  → attention-pooling MIL (max 대신 learned pool) 검토")
elif auprc_ok and not bias_ok:
    print(f"  AUPRC 유지 ({macro_mil:.4f}) but bias 개선 미미 ({mean_bias_mil:.4f})")
    print("  → canonical λ 증가 (0.1→0.2) 또는 harder canonical mining 검토")
else:
    print(f"  FAIL: AUPRC Δ={macro_mil-macro_v10b:+.4f}, bias Δ={mean_bias_mil-mean_bias_v10b:+.4f}")
    print("  → attention MIL 또는 gene-level 재집계 방식 재설계 필요")

cr_note = ('↓ 감소 — MIL이 label space 충돌 해소 확인'
           if mean_cr_all < 0.35 else
           '→ 유사 — label space 이외 다른 충돌 원인 존재'
           if mean_cr_all < 0.45 else
           '↑ 비슷 — 충돌 구조 여전히 존재 (canonical이 강한 반대 방향)')
print(f"  Conflict: {mean_cr_all:.3f} vs PCGrad {cr_note}")
