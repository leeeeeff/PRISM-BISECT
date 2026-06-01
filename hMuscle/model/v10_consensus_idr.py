"""
v10_consensus_idr.py — Gene Consensus Ablation + IDR/Domain Delta Analysis
===========================================================================
두 가지 실험:

[A] Gene Consensus Ablation (C1-b 검증)
  목적: pos_bias=1.2가 isoform-specific ESM-2 정보에서 오는가,
        아니면 gene-level ESM-2에서도 동일하게 나오는가?
  방법:
    - 동일 v10-B 모델(학습 그대로) 사용
    - Test 시 각 isoform의 ESM-2 embedding을 해당 gene의 canonical(longest) isoform 임베딩으로 교체
    - pos_bias(consensus) << pos_bias(v10-B) → v10-B이 실제로 isoform-specific 정보 사용
    - pos_bias(consensus) ≈ pos_bias(v10-B) → gene-level 정보가 pos_bias 지배 (C1-b 확인)

[B] ESM-2 vs domain_delta 정보 중복성 (IDR 가설 검증)
  목적: ESM-2 embedding delta가 domain_delta와 얼마나 독립적인가?
  방법:
    1. esm2_delta[i] = ESM-2(iso_i) - ESM-2(canonical_i)  (640d)
    2. Pearson(||esm2_delta||₂,  ||domain_delta||₁) per gene group
    3. Pearson(||esm2_delta||₂,  Δaa_length) — 서열 길이 변화와의 상관
    4. 결론: r 낮음 → domain_delta/splicing_delta가 ESM-2와 독립적 IDR 정보 제공 가능성

판정:
  C1-b 기각(OK): pos_bias_consensus << pos_bias_v10b → v10-B genuine isoform learning
  C1-b 확인(FAIL): pos_bias_consensus ≈ pos_bias_v10b → gene-level shortcut
  IDR orthogonal: Pearson(esm2_delta_norm, domain_delta_norm) < 0.3 → 독립 정보

실행:
  conda activate isoform_env
  python hMuscle/model/v10_consensus_idr.py
"""

import os, sys, json, time
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        tf.config.set_visible_devices(gpus[1] if len(gpus) > 1 else gpus[0], 'GPU')
    except RuntimeError:
        pass

# ─── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
FEAT_DIR  = '../results_isoform/features'
ts = time.strftime('%Y%m%d_%H%M')
OUT_DIR   = f'../../reports/consensus_idr/{ts}'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
GO_TERMS = ['GO:0006096', 'GO:0003774', 'GO:0007204', 'GO:0030017', 'GO:0006941']

# ─── Load data ─────────────────────────────────────────────────────────────
print("=" * 65)
print(" Gene Consensus Ablation + IDR Analysis — Loading ...")
print("=" * 65)

X_te_esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
X_tr_esm2 = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
domain_delta = np.load(f'{FEAT_DIR}/domain_delta_v2.npy').astype(np.float32)
seq_mat = np.load('my_sequence_matrix_fixed.npy')
aa_lengths = (seq_mat != 0).sum(axis=1).astype(np.float32)

def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

X_te_isoid  = load_ids('my_isoform_list_fixed.npy')
X_te_geneid = load_ids('my_gene_list_fixed.npy')
X_tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

X_te_sym      = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in X_te_geneid]
X_te_genebase = [g.split('.')[0] for g in X_te_geneid]

N = len(X_te_isoid)

# ─── Canonical isoform per gene (MANE Select → Ensembl → APPRIS → longest_CDS) ──
# canonical_reference.tsv: 87% MANE Select, built from GENCODE v44
CANONICAL_REF = f'{FEAT_DIR}/canonical_reference.tsv'
gene_base_to_canonical_idx = {}  # gene_base → index in 36748 array
with open(CANONICAL_REF) as f:
    next(f)  # skip header: gene_base | gene_versioned | canonical_enst_base | canonical_iso_idx | canonical_source | canonical_iso_id
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            gene_base = parts[0]
            try:
                canonical_idx = int(parts[3])
                gene_base_to_canonical_idx[gene_base] = canonical_idx
            except ValueError:
                pass

gene_to_idxs = defaultdict(list)
for i, g in enumerate(X_te_geneid):
    gene_to_idxs[g].append(i)

gene_to_canonical = {}  # gene_versioned → canonical isoform index in 36748 array
n_ref, n_fallback = 0, 0
for g, idxs in gene_to_idxs.items():
    g_base = g.split('.')[0]
    if g_base in gene_base_to_canonical_idx:
        cidx = gene_base_to_canonical_idx[g_base]
        if cidx < len(X_te_esm2):
            gene_to_canonical[g] = cidx
            n_ref += 1
            continue
    # fallback: longest in test set (for BambuGene / unmatched)
    lengths = [aa_lengths[i] for i in idxs]
    gene_to_canonical[g] = idxs[int(np.argmax(lengths))]
    n_fallback += 1

print(f"  Canonical source: {n_ref} from reference (MANE/Ensembl/APPRIS), {n_fallback} longest fallback")

# gene-consensus ESM-2 matrix: 각 isoform을 gene canonical 임베딩으로 교체
X_te_esm2_consensus = np.zeros_like(X_te_esm2)
for i, g in enumerate(X_te_geneid):
    canonical_idx = gene_to_canonical[g]
    X_te_esm2_consensus[i] = X_te_esm2[canonical_idx]

# canonical ESM-2 delta (isoform - canonical per gene)
esm2_delta = X_te_esm2 - X_te_esm2_consensus   # (N, 640)
esm2_delta_norm = np.linalg.norm(esm2_delta, axis=1)  # (N,)

# domain delta norm (L1)
domain_delta_sign = np.sign(domain_delta)
domain_delta_norm = np.abs(domain_delta_sign).sum(axis=1)  # (N,) — n_domains changed

print(f"  Test isoforms: {N}, unique genes: {len(gene_to_canonical)}")
print(f"  ESM-2 delta norm: mean={esm2_delta_norm.mean():.3f}, std={esm2_delta_norm.std():.3f}")
print(f"  Domain delta norm: mean={domain_delta_norm.mean():.2f} (n_domains changed)")

# ─── Label loading ──────────────────────────────────────────────────────────
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

def get_cw(y):
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    return {0: 1.0, 1: n_neg / max(n_pos, 1)}

# ─── v10-B ──────────────────────────────────────────────────────────────────
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


def train_v10b(go_term):
    y_tr, y_te = load_labels(go_term)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_esm2)
    X_te_iso_sc  = sc.transform(X_te_esm2)           # isoform-specific
    X_te_cons_sc = sc.transform(X_te_esm2_consensus)  # gene-consensus

    rng = np.random.RandomState(SEED)
    n_val = max(int(len(y_tr) * 0.1), 100)
    vi = rng.choice(len(y_tr), size=n_val, replace=False)
    ti = np.setdiff1d(np.arange(len(y_tr)), vi)

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    model = build_v10B()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=5, verbose=0),
    ]
    model.fit(X_tr_sc[ti], y_tr[ti],
              validation_data=(X_tr_sc[vi], y_tr[vi]),
              epochs=80, batch_size=512,
              class_weight=get_cw(y_tr),
              callbacks=cb_list, verbose=0)

    probs_iso  = model.predict(X_te_iso_sc,  verbose=0).ravel()
    probs_cons = model.predict(X_te_cons_sc, verbose=0).ravel()

    auprc_iso  = float(average_precision_score(y_te, probs_iso))
    auprc_cons = float(average_precision_score(y_te, probs_cons))
    return probs_iso, probs_cons, y_te, auprc_iso, auprc_cons


# ─── pos_bias ────────────────────────────────────────────────────────────────
def compute_pos_bias(scores, y_te):
    gene_idx = defaultdict(list)
    for i, g in enumerate(X_te_genebase):
        gene_idx[g].append(i)
    within_stds = []
    for g, idxs in gene_idx.items():
        if y_te[idxs[0]] == 0:
            continue
        if len(idxs) < 2:
            continue
        within_stds.append(float(np.std([scores[i] for i in idxs])))
    if not within_stds:
        return float('nan')
    return float(np.mean(within_stds) / max(float(np.std(scores)), 1e-9))


# ─── Experiment A: Gene Consensus Ablation ──────────────────────────────────
print("\n" + "=" * 65)
print(" [A] Gene Consensus Ablation")
print("=" * 65)

results_A = {}
for go in GO_TERMS:
    print(f"\n  [{go}]")
    probs_iso, probs_cons, y_te, auprc_iso, auprc_cons = train_v10b(go)

    pb_iso  = compute_pos_bias(probs_iso,  y_te)
    pb_cons = compute_pos_bias(probs_cons, y_te)

    delta_auprc = auprc_iso - auprc_cons
    delta_pb    = pb_iso - pb_cons
    drop_pct    = (pb_iso - pb_cons) / max(pb_iso, 1e-9) * 100

    print(f"    AUPRC:    iso={auprc_iso:.4f}, cons={auprc_cons:.4f}, Δ={delta_auprc:+.4f}")
    print(f"    pos_bias: iso={pb_iso:.4f},  cons={pb_cons:.4f}, Δ={delta_pb:+.4f} ({drop_pct:+.1f}%)")

    results_A[go] = {
        'auprc_iso': auprc_iso, 'auprc_cons': auprc_cons,
        'pb_iso': pb_iso, 'pb_cons': pb_cons,
        'delta_pb': delta_pb, 'drop_pct': drop_pct,
    }

print("\n" + "-" * 65)
print(f"  {'GO term':<15} {'pb_iso':>8} {'pb_cons':>9} {'drop%':>8} {'AUPRC_iso':>10} {'AUPRC_cons':>11}")
print("-" * 65)
for go in GO_TERMS:
    r = results_A[go]
    print(f"  {go:<15} {r['pb_iso']:>8.4f} {r['pb_cons']:>9.4f} "
          f"{r['drop_pct']:>+8.1f}% {r['auprc_iso']:>10.4f} {r['auprc_cons']:>11.4f}")

pb_iso_mean  = np.mean([results_A[g]['pb_iso']  for g in GO_TERMS])
pb_cons_mean = np.mean([results_A[g]['pb_cons'] for g in GO_TERMS])
print(f"\n  Macro pos_bias: iso={pb_iso_mean:.4f}, cons={pb_cons_mean:.4f}, "
      f"Δ={pb_iso_mean-pb_cons_mean:+.4f}")


# ─── Experiment B: ESM-2 delta vs domain_delta independence (IDR) ──────────
print("\n" + "=" * 65)
print(" [B] ESM-2 Delta vs Domain Delta — Information Independence")
print("=" * 65)

results_B = {}
for go in GO_TERMS:
    _, y_te = load_labels(go)
    pos_mask = (y_te == 1)
    neg_mask = (y_te == 0)

    # 전체 isoforms
    r_all_esm_dd,  _ = pearsonr(esm2_delta_norm, domain_delta_norm)
    r_all_esm_len, _ = pearsonr(esm2_delta_norm, aa_lengths)

    # 양성 유전자 내 isoforms
    pos_idxs = np.where(pos_mask)[0]
    if len(pos_idxs) > 10:
        r_pos_esm_dd,  _ = pearsonr(esm2_delta_norm[pos_idxs], domain_delta_norm[pos_idxs])
        r_pos_esm_len, _ = pearsonr(esm2_delta_norm[pos_idxs], aa_lengths[pos_idxs])
    else:
        r_pos_esm_dd = r_pos_esm_len = float('nan')

    # canonical isoforms 제외 (canonical은 delta=0 → 필터링)
    non_canonical = np.where(esm2_delta_norm > 0.01)[0]
    r_nc_esm_dd, _ = pearsonr(esm2_delta_norm[non_canonical],
                               domain_delta_norm[non_canonical]) if len(non_canonical) > 100 else (float('nan'), float('nan'))

    results_B[go] = {
        'r_all_esm_dd':  float(r_all_esm_dd),
        'r_all_esm_len': float(r_all_esm_len),
        'r_pos_esm_dd':  float(r_pos_esm_dd),
        'r_pos_esm_len': float(r_pos_esm_len),
        'r_nc_esm_dd':   float(r_nc_esm_dd),
        'n_pos': int(pos_mask.sum()),
        'n_nc': int(len(non_canonical)),
    }
    print(f"  [{go}] n_pos={int(pos_mask.sum())}")
    print(f"    All: Pearson(ESM2_delta_norm, DD_norm)={r_all_esm_dd:+.4f}, "
          f"Pearson(ESM2_delta_norm, length)={r_all_esm_len:+.4f}")
    print(f"    Pos: Pearson(ESM2_delta_norm, DD_norm)={r_pos_esm_dd:+.4f}, "
          f"Pearson(ESM2_delta_norm, length)={r_pos_esm_len:+.4f}")
    print(f"    Non-canonical only (n={len(non_canonical)}): "
          f"ESM2↔DD r={r_nc_esm_dd:+.4f}")

# 전체 상관관계 요약
print("\n  Overall ESM-2 delta vs domain delta (all isoforms):")
r_global, p_global = pearsonr(esm2_delta_norm, domain_delta_norm)
r_len,    p_len    = pearsonr(esm2_delta_norm, aa_lengths)
print(f"    Pearson(ESM2_delta_norm, domain_delta_norm) = {r_global:+.4f} (p={p_global:.2e})")
print(f"    Pearson(ESM2_delta_norm, aa_length)         = {r_len:+.4f}     (p={p_len:.2e})")
r_dd_len, _ = pearsonr(domain_delta_norm, aa_lengths)
print(f"    Pearson(domain_delta_norm, aa_length)       = {r_dd_len:+.4f}")


# ─── VERDICT ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" VERDICT")
print("=" * 65)

print("\n[A] Gene Consensus Ablation (C1-b):")
for go in GO_TERMS:
    r = results_A[go]
    if r['drop_pct'] > 30:
        v = f"ISOFORM_SPECIFIC ✅ (pos_bias drops {r['drop_pct']:+.1f}% → ESM-2 isoform 차이 사용 중)"
    elif r['drop_pct'] > 10:
        v = f"PARTIAL (drop {r['drop_pct']:+.1f}%)"
    else:
        v = f"GENE_LEVEL ❌ (drop only {r['drop_pct']:+.1f}% → gene-level shortcut)"
    print(f"  {go}: {v}")

print("\n[B] ESM-2 delta vs domain_delta independence:")
if abs(r_global) < 0.3:
    print(f"  r={r_global:.4f} < 0.3 → ESM-2 delta와 domain_delta는 독립적 정보")
    print("  → domain_delta가 ESM-2에서 포착하지 못한 이소폼 변화 정보 제공 가능 (IDR 가설 지지)")
elif abs(r_global) < 0.5:
    print(f"  r={r_global:.4f} (중간 상관) → 부분적 중복, domain_delta 독립 기여 있음")
else:
    print(f"  r={r_global:.4f} ≥ 0.5 → 강한 중복 → domain_delta는 ESM-2의 대리 변수")

# ─── Save ────────────────────────────────────────────────────────────────────
out = {'experiment_A': results_A, 'experiment_B': results_B,
       'global_r_esm_dd': float(r_global), 'global_r_esm_len': float(r_len),
       'global_r_dd_len': float(r_dd_len)}
with open(os.path.join(OUT_DIR, 'consensus_idr_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Results → {OUT_DIR}/consensus_idr_results.json")
print("  Done.")
