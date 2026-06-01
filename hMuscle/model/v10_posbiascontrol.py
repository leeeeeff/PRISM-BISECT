"""
v10_posbiascontrol.py — pos_bias Sequence-Length Ablation
==========================================================
목적: pos_bias=1.196이 sequence length confound인지 검증.

실험 3종:
  E1. Spearman(aa_length, v10B_score) within positive genes per GO term
      → length-score 직접 상관관계 측정
  E2. Length-only model pos_bias
      → log(aa_length) 단독으로 pos_bias > 1 달성 가능한지
  E3. Residualized pos_bias
      → v10B score에서 length 효과(linear regression) 제거 후 pos_bias
  E4. Length-matched pos_bias (추가 검증)
      → 같은 유전자 내 이소폼을 길이 유사 쌍으로 제한 후 pos_bias

판정 기준:
  - E2 pos_bias_length ≈ 1.196 → length가 주원인 (pos_bias 주장 기각)
  - E3 pos_bias_residual ≈ 1.0  → length 제거 시 isoform discrimination 사라짐
  - E1 mean Spearman r > 0.5   → length가 강한 confounding factor

실행:
  conda activate isoform_env
  python hMuscle/model/v10_posbiascontrol.py
"""

import os, sys, json, time
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
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
ts = time.strftime('%Y%m%d_%H%M')
OUT_DIR   = f'../../reports/posbiascontrol/{ts}'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

GO_TERMS = ['GO:0006096', 'GO:0003774', 'GO:0007204', 'GO:0030017', 'GO:0006941']

# ─── Load data ─────────────────────────────────────────────────────────────
print("=" * 65)
print(" pos_bias Length Ablation — Loading data ...")
print("=" * 65)

X_te_esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
X_tr_esm2 = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
seq_mat   = np.load('my_sequence_matrix_fixed.npy')

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

# aa_length 계산 (non-zero positions = actual amino acids)
aa_lengths = (seq_mat != 0).sum(axis=1).astype(np.float32)
log_lengths = np.log1p(aa_lengths)  # log transform for linearity
print(f"  aa_length: mean={aa_lengths.mean():.1f}, std={aa_lengths.std():.1f}, "
      f"min={aa_lengths.min():.0f}, max={aa_lengths.max():.0f}")

N = len(X_te_isoid)


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


# ─── v10-B model ────────────────────────────────────────────────────────────
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
    X_te_sc = sc.transform(X_te_esm2)

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
    probs = model.predict(X_te_sc, verbose=0).ravel()
    auprc = float(average_precision_score(y_te, probs))
    return probs, y_te, auprc


# ─── pos_bias computation ────────────────────────────────────────────────────
from collections import defaultdict

def compute_pos_bias(scores, y_te, genebase_list, mode='pos'):
    """
    mode='all': 전체 isoform (원래 all-isoform bias)
    mode='pos': 양성 유전자 내 isoform만 (pos_bias)
    """
    gene_idx = defaultdict(list)
    for i, g in enumerate(genebase_list):
        gene_idx[g].append(i)

    within_stds = []
    for g, idxs in gene_idx.items():
        if mode == 'pos' and y_te[idxs[0]] == 0:
            continue
        if len(idxs) < 2:
            continue
        within_stds.append(float(np.std([scores[i] for i in idxs])))

    if not within_stds:
        return float('nan')
    global_std = float(np.std(scores))
    return float(np.mean(within_stds) / max(global_std, 1e-9))


# ─── E1: Spearman(aa_length, score) within positive genes ──────────────────
def e1_length_score_correlation(scores, y_te):
    """양성 유전자 내 isoform들에 대해 length-score Spearman 계산"""
    gene_idx = defaultdict(list)
    for i, g in enumerate(X_te_genebase):
        if y_te[i] == 1:
            gene_idx[g].append(i)

    r_vals, p_vals = [], []
    for g, idxs in gene_idx.items():
        if len(idxs) < 3:
            continue
        lens  = [aa_lengths[i] for i in idxs]
        scrs  = [scores[i] for i in idxs]
        if np.std(lens) == 0 or np.std(scrs) == 0:
            continue
        r, p = spearmanr(lens, scrs)
        r_vals.append(float(r))
        p_vals.append(float(p))

    if not r_vals:
        return {'mean_r': float('nan'), 'median_r': float('nan'),
                'n_genes': 0, 'frac_sig': 0.0}
    return {
        'mean_r':   float(np.mean(r_vals)),
        'median_r': float(np.median(r_vals)),
        'std_r':    float(np.std(r_vals)),
        'n_genes':  len(r_vals),
        'frac_sig_0.05': float(np.mean(np.array(p_vals) < 0.05)),
        'frac_sig_0.01': float(np.mean(np.array(p_vals) < 0.01)),
    }


# ─── E2: Length-only model pos_bias ────────────────────────────────────────
def e2_length_only_pos_bias(y_tr, y_te):
    """log(aa_length) 단독 LR → pos_bias 계산"""
    # 훈련: train set의 sequence length 필요 (train ESM-2에서 추정 불가 → test split 사용)
    # 단순화: test set에서 log_length → LR 학습 (cross-val이 아닌 직접 fit)
    # 목적: "length 단독 signal의 pos_bias 잠재력" 측정
    X_len = log_lengths.reshape(-1, 1)  # (36748, 1)

    # train set용 length 없음 → test-only split으로 pos_bias 측정
    # (train 따로 없이 test label 사용 — E2는 최악의 경우 측정이므로 OK)
    lr = LogisticRegression(class_weight='balanced', C=1.0, max_iter=200)
    lr.fit(X_len, y_te)
    length_scores = lr.predict_proba(X_len)[:, 1]

    pb_all = compute_pos_bias(length_scores, y_te, X_te_genebase, mode='all')
    pb_pos = compute_pos_bias(length_scores, y_te, X_te_genebase, mode='pos')
    auprc = float(average_precision_score(y_te, length_scores))
    return length_scores, pb_all, pb_pos, auprc


# ─── E3: Residualized pos_bias ─────────────────────────────────────────────
def e3_residualized_pos_bias(v10b_scores, y_te):
    """v10B score에서 log(length) 선형 효과 제거 후 pos_bias"""
    # test isoforms 전체에 대해 linear regression: score ~ log_length
    X_len = log_lengths.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X_len, v10b_scores)
    predicted_by_length = reg.predict(X_len)
    residuals = v10b_scores - predicted_by_length
    # residuals를 [0,1]로 정규화 (비율 유지)
    r_min, r_max = residuals.min(), residuals.max()
    residuals_norm = (residuals - r_min) / max(r_max - r_min, 1e-9)

    pb_all  = compute_pos_bias(residuals_norm, y_te, X_te_genebase, mode='all')
    pb_pos  = compute_pos_bias(residuals_norm, y_te, X_te_genebase, mode='pos')
    corr_r  = float(pearsonr(log_lengths, v10b_scores)[0])
    return residuals_norm, pb_all, pb_pos, corr_r


# ─── E4: ESM-2 norm pos_bias (추가: embedding 크기가 driver인지) ────────────
def e4_esm2norm_pos_bias(y_te):
    """ESM-2 embedding L2 norm → pos_bias (MLP 없이 embedding quality만)"""
    norms = np.linalg.norm(X_te_esm2, axis=1)
    # LR on norm
    X_n = norms.reshape(-1, 1)
    lr = LogisticRegression(class_weight='balanced', C=1.0, max_iter=200)
    lr.fit(X_n, y_te)
    norm_scores = lr.predict_proba(X_n)[:, 1]

    pb_all = compute_pos_bias(norm_scores, y_te, X_te_genebase, mode='all')
    pb_pos = compute_pos_bias(norm_scores, y_te, X_te_genebase, mode='pos')
    # correlation between norm and length
    corr_nl = float(pearsonr(norms, aa_lengths)[0])
    return norm_scores, pb_all, pb_pos, corr_nl


# ─── Main ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" Training v10-B per GO term ...")
print("=" * 65)

all_results = {}

for go in GO_TERMS:
    print(f"\n[{go}]")
    y_tr, y_te = load_labels(go)

    # v10-B
    v10b_scores, _, auprc = train_v10b(go)
    print(f"  v10-B AUPRC={auprc:.4f}")

    # pos_bias (original)
    pb_v10b_pos = compute_pos_bias(v10b_scores, y_te, X_te_genebase, mode='pos')
    pb_v10b_all = compute_pos_bias(v10b_scores, y_te, X_te_genebase, mode='all')
    print(f"  pos_bias (v10-B):   all={pb_v10b_all:.4f}, pos={pb_v10b_pos:.4f}")

    # E1: Spearman(length, v10B) within positive genes
    e1 = e1_length_score_correlation(v10b_scores, y_te)
    print(f"  E1 Spearman(len,score): mean_r={e1['mean_r']:.4f}, "
          f"median_r={e1['median_r']:.4f}, n_genes={e1['n_genes']}, "
          f"frac_sig_0.05={e1['frac_sig_0.05']:.3f}")

    # E2: Length-only pos_bias
    _, pb_len_all, pb_len_pos, auprc_len = e2_length_only_pos_bias(y_tr, y_te)
    print(f"  E2 Length-only:     all={pb_len_all:.4f}, pos={pb_len_pos:.4f} "
          f"(AUPRC={auprc_len:.4f})")

    # E3: Residualized pos_bias
    _, pb_res_all, pb_res_pos, corr_r = e3_residualized_pos_bias(v10b_scores, y_te)
    print(f"  E3 Residualized:    all={pb_res_all:.4f}, pos={pb_res_pos:.4f} "
          f"(Pearson(len,score)={corr_r:.4f})")

    # E4: ESM-2 norm pos_bias
    _, pb_norm_all, pb_norm_pos, corr_nl = e4_esm2norm_pos_bias(y_te)
    print(f"  E4 ESM2_norm:       all={pb_norm_all:.4f}, pos={pb_norm_pos:.4f} "
          f"(Pearson(norm,len)={corr_nl:.4f})")

    all_results[go] = {
        'auprc_v10b': auprc,
        'v10b':   {'pos': pb_v10b_pos, 'all': pb_v10b_all},
        'E1':     e1,
        'length': {'pos': pb_len_pos,  'all': pb_len_all, 'auprc': auprc_len},
        'resid':  {'pos': pb_res_pos,  'all': pb_res_all, 'pearson_len_score': corr_r},
        'esm2norm': {'pos': pb_norm_pos, 'all': pb_norm_all, 'pearson_norm_len': corr_nl},
    }


# ─── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" SUMMARY — pos_bias Length Ablation")
print("=" * 65)
print(f"\n{'GO term':<15} {'v10B_pos':>9} {'len_pos':>9} {'resid_pos':>10} {'ESM2norm':>9} | E1_meanR")
print("-" * 75)
for go in GO_TERMS:
    r = all_results[go]
    print(f"{go:<15} "
          f"{r['v10b']['pos']:>9.4f} "
          f"{r['length']['pos']:>9.4f} "
          f"{r['resid']['pos']:>10.4f} "
          f"{r['esm2norm']['pos']:>9.4f} | "
          f"{r['E1']['mean_r']:>+.4f}")

# 판정
print("\n" + "=" * 65)
print(" VERDICT")
print("=" * 65)
for go in GO_TERMS:
    r = all_results[go]
    v10b_pos = r['v10b']['pos']
    len_pos  = r['length']['pos']
    res_pos  = r['resid']['pos']
    e1_r     = r['E1']['mean_r']

    verdict = []
    if len_pos > 0.8 * v10b_pos:
        verdict.append(f"LENGTH_CONFOUND (len_pos={len_pos:.3f} = {len_pos/v10b_pos*100:.0f}% of v10B)")
    if res_pos < 0.5 * v10b_pos:
        verdict.append(f"RESID_DROPS (pos_bias residual={res_pos:.3f} = {res_pos/v10b_pos*100:.0f}%)")
    if abs(e1_r) > 0.4:
        verdict.append(f"HIGH_CORR r={e1_r:.3f}")
    if not verdict:
        verdict.append(f"OK — pos_bias_length({len_pos:.3f}) << v10B({v10b_pos:.3f})")
    print(f"  {go}: {' | '.join(verdict)}")

# ─── Save ────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'posbiascontrol_results.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Results → {out_path}")
print("  Done.")
