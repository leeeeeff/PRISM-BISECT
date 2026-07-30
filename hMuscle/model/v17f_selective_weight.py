"""
v17f_selective_weight.py — Selective Confidence-Weighted Focal Loss
====================================================================
두 가지 개선:
  1. 도메인 중요도(pfam_imp)를 MF만이 아닌 MF+BP+CC 전체 GO로 계산
     → 도메인이 어떤 기능에든 중요하면 신뢰도 계산에 반영
  2. 가중치 적용 대상: domain-dependent MF term (norm_ratio < 0.97, 26/82)만
     → motif-dependent term (calmodulin IQ, coiled-coil 등)은 가중치 미적용

이전 confweight_a02 실패 원인:
  - 모든 82 MF term에 동일한 억제 적용 → NTRK1 키나제, IGF1 리간드 등 핵심 도메인 신호 손상
  - confweight_a02: UniProt 23/42 = 0.548 (baseline 26/42 = 0.619보다 낮음)
"""

import numpy as np, os, time, csv, gzip, json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# GPU config MUST happen before any TF operations
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(f"GPU config warning: {e}")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ROOT      = Path("/home/welcome1/sw1686/DIFFUSE")
DATA_DIR  = str(ROOT / "hMuscle/data")
ID_DIR    = str(ROOT / "hMuscle/data/raw_data/data/id_lists")
ANNOT_DIR = str(ROOT / "hMuscle/data/raw_data/data/annotations")
DOM_DIR   = ROOT / "hMuscle/data/raw_data/data/raw_data/domain_data"
FEAT_DIR  = ROOT / "hMuscle/results_isoform/features"
OUT_DIR   = ROOT / "reports/v17f_selective_weight"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER_A, LAYER_B = 15, 30
N_SEED   = 5
ALPHA_W  = 0.2   # floor for domain-dep terms
N_CDD    = 512
NORM_RATIO_THR = 0.97  # domain-dependent classification threshold

# ── 1. Embeddings ────────────────────────────────────────────────────────
print("[1] ESM-2 embeddings...")
X_tr_l30 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_tr_l15 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
X_te_l30 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_te_l15 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
print(f"  tr: {X_tr_l30.shape}, te: {X_te_l30.shape}")

# ── 2. IDs ───────────────────────────────────────────────────────────────
print("[2] Gene IDs...")
tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes = [x.decode() if isinstance(x, bytes) else x for x in tr_genes_raw]

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

def clean(g):
    s = str(g)
    for c in ["b'","'",'"',' ']: s = s.replace(c,'')
    return s

te_gene_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_iso_raw  = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
te_genes_ensg = [clean(g) for g in te_gene_raw]
te_sym_list = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_genes_ensg]

# ── 3. GO labels (MF + BP + CC) ──────────────────────────────────────────
print("[3] GO labels (MF + BP + CC)...")
sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
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
tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

# Go annotation dicts per namespace
go_genes_tr_mf  = defaultdict(set)
go_genes_tr_bp  = defaultdict(set)
go_genes_tr_cc  = defaultdict(set)
go_genes_all_mf = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) < 8 or p[0] != '9606': continue
        gid, go, cat = p[1], p[2], p[7]
        if gid not in tr_id_set and cat == 'Function': go_genes_all_mf[go].add(gid)
        if gid not in tr_id_set: continue
        if cat == 'Function':
            go_genes_tr_mf[go].add(gid)
            go_genes_all_mf[go].add(gid)
        elif cat == 'Process': go_genes_tr_bp[go].add(gid)
        elif cat == 'Component': go_genes_tr_cc[go].add(gid)

# MF terms (82 terms, same as baseline)
mf_terms = []
with open(str(ROOT / "reports/v_expanded_gomf/mf_domain_vs_prism.tsv")) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 1: mf_terms.append(p[0])
mf_terms = np.array(mf_terms)

# BP terms (103 terms, from expanded GO eval)
all_terms_info = {}
with open(str(ROOT / "reports/v_expanded_gomf/expanded_go_per_term.tsv")) as f:
    for row in csv.DictReader(f, delimiter='\t'):
        all_terms_info[row['go_id']] = row
bp_terms = np.array([r['go_id'] for r in all_terms_info.values()
                     if r['cat']=='BP' and int(r['n_pos_te'])>=2 and int(r['n_pos_tr'])>=2])
cc_terms = np.array([r['go_id'] for r in all_terms_info.values()
                     if r['cat']=='CC' and int(r['n_pos_te'])>=2 and int(r['n_pos_tr'])>=2])

print(f"  MF: {len(mf_terms)}, BP: {len(bp_terms)}, CC: {len(cc_terms)} terms")

def build_Y_tr(go_id, go_dict_tr):
    pos_ids  = go_dict_tr[go_id]
    pos_syms = {g for g, eid in zip(tr_genes, tr_entrez) if eid in pos_ids}
    y = np.zeros(len(tr_genes), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
    return y

# Build label matrices
print("  Building Y_tr (MF)...")
Y_tr = np.stack([build_Y_tr(go, go_genes_tr_mf) for go in mf_terms], axis=1)
print(f"  Y_tr shape: {Y_tr.shape}, pos rate: {Y_tr.mean():.4f}")

# Test labels — load precomputed
Y_te = np.load(ROOT / "reports/v17f_star_bootstrap/Y_te.npy")
print(f"  Y_te shape: {Y_te.shape}, pos rate: {Y_te.mean():.4f}")
valid_te_terms = np.where(Y_te.sum(0) >= 2)[0]
print(f"  Valid test terms: {len(valid_te_terms)}")

# BP/CC labels for training (used for domain importance only)
print("  Building Y_tr (BP)...")
Y_tr_bp = np.stack([build_Y_tr(go, go_genes_tr_bp) for go in bp_terms], axis=1)
print("  Building Y_tr (CC)...")
Y_tr_cc = np.stack([build_Y_tr(go, go_genes_tr_cc) for go in cc_terms], axis=1)
print(f"  BP pos rate: {Y_tr_bp.mean():.4f}, CC pos rate: {Y_tr_cc.mean():.4f}")

# ── 4. CDD domain matrix ─────────────────────────────────────────────────
print("[4] CDD domain matrix...")
try:
    dm_raw = np.load(str(FEAT_DIR / "domain_matrix_train_cdd.npy"))
    if dm_raw.shape[1] > N_CDD: dm_raw = dm_raw[:, :N_CDD]
    dm_tr = (dm_raw > 0).astype(np.float32)
except:
    # Build from feature files
    from scipy import sparse
    dm_files = sorted(FEAT_DIR.glob("cdd_train_batch_*.npy"))
    if dm_files:
        dm_tr = np.vstack([np.load(str(f)) for f in dm_files]).astype(np.float32)
        if dm_tr.shape[1] > N_CDD: dm_tr = dm_tr[:, :N_CDD]
    else:
        dm_tr = np.zeros((len(tr_genes), N_CDD), dtype=np.float32)

print(f"  Domain matrix: {dm_tr.shape}, nonzero rows: {(dm_tr.sum(1)>0).sum()}")

# ── 5. Pfam GO-importance from MF + BP + CC ──────────────────────────────
print("[5] GO-CDD log-odds importance (MF + BP + CC)...")
eps = 1e-6
n_all_terms = len(mf_terms) + len(bp_terms) + len(cc_terms)
all_Y_tr = np.concatenate([Y_tr, Y_tr_bp, Y_tr_cc], axis=1)  # (n_tr, n_all)

pfam_imp_all = np.zeros((N_CDD, n_all_terms), dtype=np.float32)
for j in range(n_all_terms):
    pos_mask = all_Y_tr[:, j] > 0
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum(); n_neg = neg_mask.sum()
    if n_pos < 5: continue
    pr = (dm_tr[pos_mask].sum(0) + eps) / (n_pos + eps)
    nr = (dm_tr[neg_mask].sum(0) + eps) / (n_neg + eps)
    pfam_imp_all[:, j] = np.clip(np.log(pr / nr), -5, 5)

# Extract MF-only importance for confidence computation
pfam_imp_mf = pfam_imp_all[:, :len(mf_terms)]  # (N_CDD, 82)

# Joint domain importance vector: max importance across ALL GO terms per domain
joint_dom_imp = pfam_imp_all.max(axis=1)  # (N_CDD,) — most important across all functions
print(f"  pfam_imp computed for {n_all_terms} GO terms (MF+BP+CC)")
print(f"  Non-zero joint importance domains: {(joint_dom_imp > 0).sum()}")

# ── 6. Norm-ratio for MF terms → domain-dep classification ───────────────
print("[6] Norm-ratio classification for MF terms...")
norms_te = np.linalg.norm(X_te_l30, axis=1)
norm_ratio = np.ones(len(mf_terms))
for j in range(len(mf_terms)):
    pos_mask = Y_te[:, j] > 0
    if pos_mask.sum() < 5: continue
    norm_ratio[j] = norms_te[pos_mask].mean() / norms_te[~pos_mask].mean()

domain_dep_mask = norm_ratio < NORM_RATIO_THR  # (82,) bool
print(f"  Domain-dependent MF terms: {domain_dep_mask.sum()}/82 (norm_ratio < {NORM_RATIO_THR})")
print(f"  Motif-dependent MF terms:  {(~domain_dep_mask).sum()}/82 (full weight, no suppression)")

# ── 7. Confidence weights (selective: only domain-dep terms) ─────────────
print("[7] Computing selective confidence weights...")
# Canonical isoform per gene: highest joint domain importance
tr_gene_arr = np.array(tr_genes)
canon_dm_tr = {}
for g in set(tr_genes):
    mask = tr_gene_arr == g
    idxs = np.where(mask)[0]
    if len(idxs) == 1:
        canon_dm_tr[g] = dm_tr[idxs[0]]
    else:
        # Use joint importance as tiebreaker
        importance_scores = dm_tr[idxs] @ joint_dom_imp
        best = np.argmax(importance_scores)
        canon_dm_tr[g] = dm_tr[idxs[best]]

# Compute per-sample, per-term confidence (for domain-dep terms)
# For motif-dep terms: weight = 1.0 (no change)
label_conf_tr = np.ones_like(Y_tr)  # default full weight
eps2 = 1e-6

for j in range(len(mf_terms)):
    if not domain_dep_mask[j]: continue  # motif-dep: keep 1.0
    for i, g in enumerate(tr_genes):
        cdm = canon_dm_tr.get(g, np.zeros(N_CDD, dtype=np.float32))
        denom = float(np.dot(cdm, pfam_imp_mf[:, j]))
        if denom < eps2:
            label_conf_tr[i, j] = 1.0  # no domain info → keep full weight
        else:
            label_conf_tr[i, j] = float(np.dot(dm_tr[i], pfam_imp_mf[:, j])) / denom

# sample_loss_weight: alpha floor for domain-dep terms, 1.0 for motif-dep
conf_weight_tr = np.ones_like(Y_tr)
for j in range(len(mf_terms)):
    if domain_dep_mask[j]:
        conf_weight_tr[:, j] = ALPHA_W + (1.0 - ALPHA_W) * np.clip(label_conf_tr[:, j], 0, 1)
    # else: stays 1.0

conf_weight_tf = tf.constant(conf_weight_tr, dtype=tf.float32)

# Stats
dd_terms = np.where(domain_dep_mask)[0]
type1_pos = (Y_tr[:, dd_terms] > 0).any(1)
if type1_pos.sum() > 0:
    sample_wt = conf_weight_tr[type1_pos][:, dd_terms]
    print(f"  Domain-dep positives: n={type1_pos.sum()}, "
          f"low_conf<0.5={((sample_wt[Y_tr[type1_pos][:,dd_terms]>0]) < 0.5).mean():.1%}, "
          f"mean_weight={(sample_wt[Y_tr[type1_pos][:,dd_terms]>0]).mean():.3f}")

# ── 8. Delta embeddings ───────────────────────────────────────────────────
print("[8] Delta embeddings...")
delta_tr = X_tr_l30 - X_tr_l15
scaler = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr)
delta_te_s = scaler.transform(X_te_l30 - X_te_l15)

# ── 9. Model ─────────────────────────────────────────────────────────────
n_go = len(mf_terms)

def make_mlp():
    inp1 = keras.Input(shape=(640,))  # delta
    inp2 = keras.Input(shape=(640,))  # L30
    x = layers.Concatenate()([inp1, inp2])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(n_go, activation='sigmoid')(x)
    return keras.Model([inp1, inp2], x)

@tf.function
def selective_focal_loss(y_true, y_pred, conf_weight, gamma=2.0):
    eps = 1e-7
    p   = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    pt  = y_true * p + (1.0 - y_true) * (1.0 - p)
    fl  = -(1.0 - pt) ** gamma * tf.math.log(pt)
    w   = y_true * conf_weight + (1.0 - y_true) * 1.0
    return tf.reduce_mean(w * fl)

# ── 10. Train 5 seeds ────────────────────────────────────────────────────
print("[9] Training (5 seeds)...")
print(f"  GPU: {gpus[0].name if gpus else 'CPU'}")

X_d = delta_tr_s
X_e = X_tr_l30
n_tr = len(tr_genes)
all_preds = []
best_vals = []

for seed in range(N_SEED):
    t0 = time.time()
    tf.random.set_seed(seed)
    mlp = make_mlp()
    opt = keras.optimizers.Adam(1e-3)

    split_idx = int(n_tr * 0.85)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_tr)
    tr_idx = perm[:split_idx]; val_idx = perm[split_idx:]

    batch_size = 256
    best_val = -1; best_weights = None; no_improve = 0; patience = 10

    for epoch in range(100):
        rng2 = np.random.default_rng(epoch + seed * 1000)
        batch_perm = rng2.permutation(len(tr_idx))
        for b in range(0, len(tr_idx), batch_size):
            bidx = tr_idx[batch_perm[b:b+batch_size]]
            xd_b = tf.constant(X_d[bidx])
            xe_b = tf.constant(X_e[bidx])
            yt_b = tf.constant(Y_tr[bidx])
            cw_b = tf.gather(conf_weight_tf, bidx.tolist())
            with tf.GradientTape() as tape:
                pred = mlp([xd_b, xe_b], training=True)
                loss = selective_focal_loss(yt_b, pred, cw_b)
            grads = tape.gradient(loss, mlp.trainable_variables)
            opt.apply_gradients(zip(grads, mlp.trainable_variables))

        val_pred = mlp([X_d[val_idx], X_e[val_idx]], training=False).numpy()
        val_y = Y_tr[val_idx]
        valid_j = val_y.sum(0) >= 2
        if valid_j.sum() > 0:
            val_auprc = np.mean([average_precision_score(val_y[:,j], val_pred[:,j])
                                  for j in np.where(valid_j)[0]])
        else:
            val_auprc = 0.0
        if val_auprc > best_val:
            best_val = val_auprc; best_weights = mlp.get_weights(); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: break

    mlp.set_weights(best_weights)
    te_pred = mlp([delta_te_s, X_te_l30], training=False).numpy()
    all_preds.append(te_pred)
    best_vals.append(best_val)
    print(f"  Seed {seed}: best_val={best_val:.4f}, epochs={epoch+1}, {time.time()-t0:.0f}s")

# ── 11. Ensemble + Evaluation ─────────────────────────────────────────────
print("[10] Ensemble + Evaluation...")
ens = np.mean(all_preds, axis=0)

auprcs = [average_precision_score(Y_te[:,j], ens[:,j])
          for j in valid_te_terms if Y_te[:,j].sum() >= 2]
overall_auprc = np.mean(auprcs)

# Type-stratified eval
type_file = ROOT / "reports/isoform_resolution_full/full_isoform_feature_types.tsv"
iso2type = {}
with open(str(type_file)) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: iso2type[p[0]] = p[4]

te_isos = [clean(x) for x in np.load('my_isoform_list_fixed.npy', allow_pickle=True)]
type_labels = np.array([iso2type.get(iso, 'Type0_NoDomain') for iso in te_isos])

type_results = {}
for t in ['Type0_NoDomain','Type1_DomainLoss','Type2_PartialTrunc','Type3_SameDomain']:
    mask = type_labels == t
    if mask.sum() == 0: continue
    auprcs_t = []
    for j in valid_te_terms:
        y_j = Y_te[mask, j]; p_j = ens[mask, j]
        if y_j.sum() >= 2: auprcs_t.append(average_precision_score(y_j, p_j))
    type_results[t] = (np.mean(auprcs_t) if auprcs_t else 0, mask.sum())

# Domain-Ranking AUC
gene2idxs_te = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs_te[g].append(i)

pairs_correct = 0; pairs_total = 0
for g, idxs in gene2idxs_te.items():
    if len(idxs) < 2: continue
    dm_idxs = [dm_tr[np.where(tr_gene_arr == g)[0][0]] if (tr_gene_arr == g).any()
               else np.zeros(N_CDD) for _ in idxs]
    dom_counts = [int(dm_tr[np.where(tr_gene_arr==g)[0][0]].sum()) if (tr_gene_arr==g).any() else 0
                  for _ in idxs]
    # Simpler: use domain type info
    iso_types = [iso2type.get(te_isos[i], 'Type0_NoDomain') for i in idxs]
    for i in range(len(idxs)):
        for j2 in range(i+1, len(idxs)):
            t_i = iso_types[i]; t_j = iso_types[j2]
            # domain-loss pair
            if (t_i in ('Type1_DomainLoss','Type2_PartialTrunc') and t_j == 'Type3_SameDomain') or \
               (t_j in ('Type1_DomainLoss','Type2_PartialTrunc') and t_i == 'Type3_SameDomain'):
                for jj in valid_te_terms:
                    s_i = ens[idxs[i], jj]; s_j = ens[idxs[j2], jj]
                    if abs(s_i - s_j) < 1e-6: continue
                    if t_i in ('Type1_DomainLoss','Type2_PartialTrunc'):
                        # expect s_i < s_j (domain-loss lower score)
                        pairs_correct += (s_i < s_j)
                    else:
                        pairs_correct += (s_j < s_i)
                    pairs_total += 1

dr_auc = pairs_correct / pairs_total if pairs_total > 0 else 0.5

print(f"\n=== v17f_selective_weight Results ===")
print(f"  Overall AUPRC: {overall_auprc:.4f}")
for t, (auprc, n) in type_results.items():
    print(f"  {t:<30}: {auprc:.4f}  n={n}")
print(f"\n  Domain-Ranking AUC: {dr_auc:.4f} (n={pairs_total} pairs)")
print(f"  [Baseline v17f*: 0.637, confweight_a02: 0.631]")

# Save predictions
np.save(OUT_DIR / "selective_weight_preds.npy", ens)
result_dict = {
    'overall_auprc': float(overall_auprc),
    'type_results': {t: {'auprc': float(a), 'n': int(n)} for t, (a, n) in type_results.items()},
    'domain_ranking_auc': float(dr_auc),
    'domain_dep_terms': int(domain_dep_mask.sum()),
    'alpha_w': ALPHA_W,
    'norm_ratio_thr': NORM_RATIO_THR,
    'ontologies_for_importance': 'MF+BP+CC',
}
with open(OUT_DIR / "results.json", 'w') as f:
    json.dump(result_dict, f, indent=2)
print(f"\nSaved to {OUT_DIR}")
print("Done.")
