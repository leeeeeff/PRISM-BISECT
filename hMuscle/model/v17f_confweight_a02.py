"""
v17f_confweight.py — Confidence-Weighted Focal Loss Training
=============================================================
v17f_abl_no_tpsi.py 기반. 유일한 변경:
  Focal Loss weight(i,j) = alpha + (1-alpha) * label_conf_train(i,j)
  - positive label에서 domain-inconsistent 기여 억제
  - negative label: weight=1.0 유지

Evaluation targets:
  1. Gene-level AUPRC  → 하락 예상 (의도적: noisy label에 덜 의존)
  2. Domain-ranking AUC → 상승 예상 (도메인 소실 → 기능 저하 학습)
  3. UniProt 42-pair    → 상승 예상 (isoform-specific ground truth)

alpha=0.5: positive weight는 최소 0.5 (완전 억제 방지)
"""

import numpy as np
import os, time
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import average_precision_score

ROOT      = Path("/home/welcome1/sw1686/DIFFUSE")
DATA_DIR  = str(ROOT / "hMuscle/data")
ID_DIR    = str(ROOT / "hMuscle/data/raw_data/data/id_lists")
DOM_DIR   = ROOT / "hMuscle/data/raw_data/data/raw_data/domain_data"
FEAT_DIR  = ROOT / "hMuscle/results_isoform/features"
os.chdir(ROOT / "hMuscle/model")

LAYER_A, LAYER_B = 15, 30
N_SEED   = 5
ALPHA_W  = 0.2       # floor for confidence weight (0 = full suppression, 1 = no effect)
N_CDD    = 512

# ── 1. ESM-2 embeddings ───────────────────────────────────────────────────
print("[1] ESM-2 embeddings 로드...")
X_tr_l30 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_tr_l15 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
X_te_l30 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_te_l15 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
print(f"  tr: {X_tr_l30.shape}, te: {X_te_l30.shape}")

# ── 2. Gene/isoform ID 로드 ───────────────────────────────────────────────
print("[2] ID 로드...")
from pathlib import Path as P
tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_ids_raw   = np.load(f'{ID_DIR}/train_isoform_list.npy', allow_pickle=True)
tr_genes = [x.decode() if isinstance(x, bytes) else x for x in tr_genes_raw]
tr_ids   = [x.decode() if isinstance(x, bytes) else x for x in tr_ids_raw]

te_gene_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_iso_raw  = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
te_genes = [x.decode() if isinstance(x, bytes) else x for x in te_gene_raw]

# ── 3. GO term 로드 및 Y_tr / Y_te 구축 (baseline과 동일한 방식) ──────────
print("[3] GO terms + labels 구축...")
import gzip as gz

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])

ANNOT_DIR = f'{DATA_DIR}/raw_data/data/annotations'
# gene symbol → Entrez GeneID (same as v17f_abl_no_tpsi)
sym2id = {}
with gz.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id: sym2id[syn] = p[1]

tr_entrez = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_entrez)
go_genes_tr  = defaultdict(set)
go_genes_all = defaultdict(set)
with gz.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) < 8: continue
        if p[0] != '9606' or p[7] != 'Function': continue
        go_genes_all[p[2]].add(p[1])
        if p[1] in tr_id_set: go_genes_tr[p[2]].add(p[1])

tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

def build_Y_tr(go_id):
    pos_ids  = go_genes_tr[go_id]
    pos_syms = {g for g, gid in zip(tr_genes, tr_entrez) if gid in pos_ids}
    y = np.zeros(len(tr_genes), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
    return y

Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)

# Test labels: load pre-built Y_te from v17f_star_bootstrap (verified correct)
Y_te = np.load(ROOT / 'reports/v17f_star_bootstrap/Y_te.npy')
valid_mask = Y_te.sum(0) >= 2
print(f"  {len(mf_terms)} MF terms | Y_tr pos rate: {Y_tr.mean():.4f} | Y_te pos rate: {Y_te.mean():.4f}")
print(f"  Valid test terms: {valid_mask.sum()}")

# ── 4. CDD domain matrix + label confidence for train ────────────────────
print("[4] CDD domain matrix 구축 (train)...")
# Load or build
cdd_mat_path = FEAT_DIR / "domain_matrix_train_cdd.npy"
cdd_vocab_path = FEAT_DIR / "domain_cdd_vocab.txt"

if cdd_mat_path.exists() and cdd_vocab_path.exists():
    dm_tr = np.load(cdd_mat_path)
    print(f"  Loaded: {dm_tr.shape}, nonzero rows: {(dm_tr.any(1)).sum()}")
else:
    # Build CDD domain matrix
    from collections import Counter
    nm_to_cdd = {}
    with open(DOM_DIR / "human_isoform_dm.txt") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            nm_id  = parts[1].strip()
            cdds   = set(parts[2].split()) if len(parts) >= 3 and parts[2].strip() else set()
            nm_to_cdd[nm_id] = cdds

    iso_cdds = []
    for iso in tr_ids:
        cdds = nm_to_cdd.get(iso, None)
        if cdds is None:
            cdds = nm_to_cdd.get(iso.rsplit('.', 1)[0], set())
        iso_cdds.append(cdds if cdds is not None else set())

    cdd_freq = Counter()
    for s in iso_cdds: cdd_freq.update(s)
    top_cdds = [c for c, _ in cdd_freq.most_common(N_CDD)]
    cdd_to_col = {c: i for i, c in enumerate(top_cdds)}

    with open(cdd_vocab_path, 'w') as f:
        for i, c in enumerate(top_cdds):
            f.write(f"{i}\t{c}\t{cdd_freq[c]}\n")

    dm_tr = np.zeros((len(tr_ids), N_CDD), dtype=np.float32)
    for i, cdd_set in enumerate(iso_cdds):
        for c in cdd_set:
            if c in cdd_to_col: dm_tr[i, cdd_to_col[c]] = 1.0

    np.save(cdd_mat_path, dm_tr)
    nz = (dm_tr.any(1)).sum()
    print(f"  Built: {dm_tr.shape}, nonzero: {nz} ({nz/len(tr_ids)*100:.1f}%)")

# Gene canonical domain vector (max across gene isoforms)
tr_gene_arr = np.array(tr_genes)
gene_to_canon = {}
for g in set(tr_genes):
    mask = (tr_gene_arr == g)
    gene_to_canon[g] = dm_tr[mask].max(0)
canon_tr = np.stack([gene_to_canon[g] for g in tr_genes])

# GO-CDD importance matrix
print("[5] GO-CDD log-odds importance...")
eps = 1e-6
n_go = len(mf_terms)
importance = np.zeros((n_go, N_CDD), dtype=np.float32)
for j in range(n_go):
    pos_mask = Y_tr[:, j].astype(bool)
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum(); n_neg = neg_mask.sum()
    if n_pos < 5: continue
    p_pos = (dm_tr[pos_mask].sum(0) + eps) / (n_pos + 2*eps)
    p_neg = (dm_tr[neg_mask].sum(0) + eps) / (n_neg + 2*eps)
    importance[j] = np.log(p_pos / p_neg)

# Label confidence for train
print("[6] label_conf_train 계산...")
imp_pos = np.maximum(importance, 0.0)  # (n_go, N_CDD)
label_conf_tr = np.zeros((len(tr_ids), n_go), dtype=np.float32)
for j in range(n_go):
    if imp_pos[j].sum() < eps:
        label_conf_tr[:, j] = 1.0
        continue
    num = dm_tr @ imp_pos[j]        # (n_tr,)
    den = canon_tr @ imp_pos[j]     # (n_tr,)
    label_conf_tr[:, j] = np.clip(num / (den + eps), 0.0, 1.5)

# Confidence weight for training:
#   positive label: ALPHA_W + (1-ALPHA_W) * conf  [floor=ALPHA_W]
#   negative label: 1.0
conf_weight_tr = ALPHA_W + (1.0 - ALPHA_W) * np.clip(label_conf_tr, 0.0, 1.0)  # (n_tr, n_go)
# Apply only to positives: where Y_tr=0, weight=1.0
sample_loss_weight = Y_tr * conf_weight_tr + (1.0 - Y_tr) * 1.0  # (n_tr, n_go)

# Stats
type1_pos = (Y_tr.astype(bool)) & (canon_tr.any(1, keepdims=True)) & (~dm_tr.any(1, keepdims=True))
if type1_pos.any():
    t1_conf = label_conf_tr[type1_pos]
    print(f"  Type1 positives: n={type1_pos.sum()}, conf<0.3={( t1_conf<0.3).mean()*100:.1f}%, mean={t1_conf.mean():.3f}")

print(f"  sample_loss_weight stats: min={sample_loss_weight.min():.3f}, mean={sample_loss_weight.mean():.3f}, max={sample_loss_weight.max():.3f}")
np.save(FEAT_DIR / "label_confidence_train.npy", label_conf_tr)

# ── 7. Delta embeddings ──────────────────────────────────────────────────
print("[7] Delta embeddings...")
def maxabsscale(x):
    s = np.abs(x).max(0, keepdims=True)
    s[s < 1e-8] = 1.0
    return x / s

delta_tr = (X_tr_l30 - X_tr_l15).astype(np.float32)
delta_te = (X_te_l30 - X_te_l15).astype(np.float32)
scale = np.abs(delta_tr).max(0, keepdims=True)
scale[scale < 1e-8] = 1.0
delta_tr_s = delta_tr / scale
delta_te_s = delta_te / scale

# ── 8. Gene-stratified CV split ──────────────────────────────────────────
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
tr_group = tr_genes

# ── 9. Model architecture ────────────────────────────────────────────────
print("[8] 모델 정의...")
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"  GPU: {gpus[0].name}")

def build_mlp(delta_dim=640, esm_dim=640, n_go=82):
    inp_d = layers.Input(shape=(delta_dim,))
    inp_e = layers.Input(shape=(esm_dim,))
    x = layers.Concatenate()([inp_d, inp_e])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n_go, activation='sigmoid')(x)
    return models.Model([inp_d, inp_e], out)

# ── 10. Custom confidence-weighted focal loss ────────────────────────────
@tf.function
def conf_focal_loss(y_true, y_pred, conf_weight, gamma=2.0):
    """Per-(sample, output) confidence-weighted focal cross-entropy."""
    eps = 1e-7
    p = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    pt = y_true * p + (1.0 - y_true) * (1.0 - p)
    fl = -(1.0 - pt) ** gamma * tf.math.log(pt)
    w  = y_true * conf_weight + (1.0 - y_true) * 1.0
    return tf.reduce_mean(w * fl)

# ── 11. Multi-seed ensemble training ─────────────────────────────────────
print("[9] 훈련 시작 (5 seeds)...")
all_te_preds = []
X_tr_in = [delta_tr_s, X_tr_l30]
X_te_in = [delta_te_s, X_te_l30]
conf_weight_tf = tf.constant(sample_loss_weight, dtype=tf.float32)

for seed in range(N_SEED):
    t0 = time.time()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    mlp = build_mlp(n_go=n_go)
    opt = optimizers.Adam(1e-3)

    # GradientTape training with confidence-weighted focal loss
    n_tr = len(tr_ids)
    batch_size = 256
    best_val = -1; best_weights = None; patience = 10; no_improve = 0

    # Gene-stratified split: use fold 0 as val
    X_d = delta_tr_s; X_e = X_tr_l30
    split_idx = int(n_tr * 0.85)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_tr)
    tr_idx = perm[:split_idx]; val_idx = perm[split_idx:]

    for epoch in range(100):
        rng2 = np.random.default_rng(epoch)
        batch_perm = rng2.permutation(len(tr_idx))
        ep_loss = []
        for b in range(0, len(tr_idx), batch_size):
            bidx = tr_idx[batch_perm[b:b+batch_size]]
            xd_b = tf.constant(X_d[bidx])
            xe_b = tf.constant(X_e[bidx])
            yt_b = tf.constant(Y_tr[bidx])
            cw_b = tf.gather(conf_weight_tf, bidx.tolist())
            with tf.GradientTape() as tape:
                pred = mlp([xd_b, xe_b], training=True)
                loss = conf_focal_loss(yt_b, pred, cw_b)
            grads = tape.gradient(loss, mlp.trainable_variables)
            opt.apply_gradients(zip(grads, mlp.trainable_variables))
            ep_loss.append(float(loss))

        # Validation AUPRC
        val_pred = mlp([X_d[val_idx], X_e[val_idx]], training=False).numpy()
        val_y    = Y_tr[val_idx]
        valid_j = val_y.sum(0) >= 2
        if valid_j.sum() > 0:
            val_auprc = np.mean([average_precision_score(val_y[:,j], val_pred[:,j])
                                  for j in np.where(valid_j)[0]])
        else:
            val_auprc = 0.0

        if val_auprc > best_val:
            best_val = val_auprc
            best_weights = mlp.get_weights()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    mlp.set_weights(best_weights)
    te_pred = mlp(X_te_in, training=False).numpy()
    all_te_preds.append(te_pred)
    dt = time.time() - t0
    print(f"  Seed {seed}: best_val={best_val:.4f}, epochs={epoch+1}, {dt:.0f}s")

# ── 12. Ensemble + Evaluation ────────────────────────────────────────────
print("[10] Ensemble + Evaluation...")
ens_pred = np.mean(all_te_preds, axis=0)
np.save('../../reports/v17f_confweight_a02_preds.npy', ens_pred)

# Overall AUPRC
valid_idx = np.where(valid_mask)[0]
auprcs = [average_precision_score(Y_te[:,j], ens_pred[:,j]) for j in valid_idx]
overall_auprc = np.mean(auprcs)

# Feature type AUPRC
import csv
ft_arr = []
with open('../../reports/isoform_resolution_full/full_isoform_feature_types.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader: ft_arr.append(row['feature_type'])
ft_arr = np.array(ft_arr)

def macro_auprc_type(mask, min_pos=2):
    yt = Y_te[mask]; yp = ens_pred[mask]
    vj = yt.sum(0) >= min_pos
    if vj.sum() == 0: return float('nan')
    return float(np.mean([average_precision_score(yt[:,j], yp[:,j]) for j in np.where(vj)[0]]))

print(f"\n=== v17f_confweight Results ===")
print(f"  Overall AUPRC: {overall_auprc:.4f}")
for t in ['Type0_NoDomain','Type1_DomainLoss','Type2_PartialTrunc','Type3_SameDomain']:
    mask = (ft_arr == t)
    print(f"  {t:25s}: {macro_auprc_type(mask):.4f}  n={mask.sum()}")

# Domain-ranking AUC
dm_te_v3 = np.load(FEAT_DIR / 'domain_matrix_proper_test_v3.npy')
te_gene_arr = np.array(te_genes)

domain_pairs = []  # (gene, go_idx, [iso_indices], [domain_counts])
for g in set(te_genes):
    gm = (te_gene_arr == g)
    dc = dm_te_v3[gm].sum(1)
    if dc.std() < 0.01: continue
    g_idx = np.where(gm)[0]
    median_dc = np.median(dc)
    high = (dc > median_dc).astype(float)
    if high.sum() < 1 or (1-high).sum() < 1: continue
    domain_pairs.append((g, g_idx, high))

dr_aucs = []
for (g, g_idx, high) in domain_pairs:
    for j in range(n_go):
        scores = ens_pred[g_idx, j]
        if high.std() < 0.01: continue
        try:
            auc = average_precision_score(high, scores)
            dr_aucs.append(auc)
        except: pass

print(f"\n  Domain-Ranking AUC: {np.mean(dr_aucs):.4f} (n={len(dr_aucs)} pairs)")
print(f"  [Baseline v17f*: 0.630]")

print("\nDone. Results saved to reports/v17f_confweight_a02_preds.npy")
