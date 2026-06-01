"""
v10_brain_bias_score.py — Brain tissue bias_score
==================================================
목적: brain transcripts (60,165) × AD GO terms 기준으로
      v10-B / LR / XGB의 bias_score를 측정하여
      뇌조직 독립 검증에서 isoform-specificity를 비교.

bias_score = within_gene_score_std / global_score_std
  < 0.10: gene-level shortcut
  > 0.30: isoform-specific

muscle bias_score (v10_v10b_bias_score.py)와 완전 분리된 독립 분석.

출력:
  reports/brain_fiu/brain_bias_score_{ts}.json
"""

import os, sys, json, time, re
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        tf.config.set_visible_devices(gpus[1] if len(gpus) > 1 else gpus[0], 'GPU')
    except RuntimeError:
        pass

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_DIR   = '../data'
ID_DIR     = '../data/raw_data/data/id_lists'
ANNOT_DIR  = '../data/raw_data/data/annotations'
BRAIN_EMB  = '../data/brain_esm2/brain_esm2_60165.npy'
BRAIN_H5AD = '/home/dhkim1674/Samsung_Alzheimer/03_AnnData/adata_transcript_for_UMAP.h5ad'
ts = time.strftime('%Y%m%d_%H%M')
OUT_DIR    = '../../reports/brain_fiu'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS  = [42, 123, 456, 789, 2024]
SEED   = 42
N_BOOT = 300
np.random.seed(SEED)

# Brain AD GO terms (same as FIU analysis)
GO_TERMS = [
    ('GO:0007204', 'Ca2+ signaling',          'Early'),
    ('GO:0055074', 'Ca2+ homeostasis',         'Early'),
    ('GO:0007005', 'Mitochondrion org',        'Early'),
    ('GO:0034976', 'ER stress response',       'Early'),
    ('GO:0000226', 'Microtubule cytoskeleton', 'Mid'),
    ('GO:0007268', 'Synaptic transmission',    'Mid'),
    ('GO:0006914', 'Autophagy',                'Mid'),
    ('GO:0043161', 'Proteasome-UPS',           'Mid-Late'),
    ('GO:0006979', 'Oxidative stress',         'Mid-Late'),
    ('GO:0006954', 'Inflammatory response',    'Late'),
    ('GO:0045087', 'Innate immune response',   'Late'),
    ('GO:0006909', 'Phagocytosis',             'Late'),
    ('GO:0042552', 'Myelination',              'Late'),
]

print("=" * 70)
print(" Brain Tissue bias_score (v10-B vs LR vs XGB)")
print(" Input: brain 60,165 transcripts × 13 AD GO terms")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────
def load_ids(p):
    a = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]

X_tr   = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_brain = np.load(BRAIN_EMB).astype(np.float32)
tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
tr_syms = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_geneid]

print(f"  Muscle train: {X_tr.shape}")
print(f"  Brain embed:  {X_brain.shape}")

# ── Load brain metadata ────────────────────────────────────────────────
import h5py
print("[1] Loading brain metadata ...")
with h5py.File(BRAIN_H5AD, 'r') as f_brain:
    brain_enst_arr  = [x.decode() for x in f_brain['var']['ENST_ID'][()]]
    gene_name_cats  = [x.decode() for x in f_brain['var']['gene_name']['categories'][()]]
    gene_name_codes = f_brain['var']['gene_name']['codes'][()]
brain_gene_arr = [gene_name_cats[c] for c in gene_name_codes]
brain_genebase = brain_gene_arr   # gene symbols already

# ── Protein-coding filter (same as FIU) ────────────────────────────────
print("[2] Protein-coding filter ...")
enst_to_type = {}
_ref_gtf = '/home/dhkim1674/reference/refdata-gex-GRCh38-2024-A/genes/genes.gtf'
with open(_ref_gtf) as _f:
    for _line in _f:
        if '\ttranscript\t' not in _line: continue
        _m_id   = re.search(r'transcript_id \"(ENST\d+)\"', _line)
        _m_type = re.search(r'transcript_type \"([^\"]+)\"', _line)
        if _m_id and _m_type:
            enst_to_type[_m_id.group(1)] = _m_type.group(1)

protein_coding_mask = np.array([
    enst_to_type.get(e, 'unknown') == 'protein_coding'
    for e in brain_enst_arr
], dtype=bool)
print(f"  protein_coding: {protein_coding_mask.sum()}/{len(brain_enst_arr)}")

brain_gene_arr_arr = np.array(brain_gene_arr)

# ── GO annotations ────────────────────────────────────────────────────
go_to_genes = defaultdict(set)
with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2: continue
        for go in parts[1:]:
            go_to_genes[go].add(parts[0])

# ── Helper functions ──────────────────────────────────────────────────
def compute_bias_score(scores, gene_ids, protein_coding_mask=None):
    """within_gene_std / global_std, protein_coding only"""
    if protein_coding_mask is not None:
        mask_idx = np.where(protein_coding_mask)[0]
        scores_pc = scores[mask_idx]
        genes_pc  = [gene_ids[i] for i in mask_idx]
    else:
        scores_pc = scores
        genes_pc  = gene_ids
    global_std = scores_pc.std()
    if global_std < 1e-10: return np.nan
    g2i = defaultdict(list)
    for i, g in enumerate(genes_pc):
        g2i[g].append(i)
    within_stds = [scores_pc[idxs].std() for idxs in g2i.values() if len(idxs) >= 2]
    if not within_stds: return np.nan
    return float(np.mean(within_stds) / global_std)

def bootstrap_ci_brain(y_true, y_score, gene_ids, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    unique_genes = np.unique(gene_ids)
    base = float(average_precision_score(y_true, y_score))
    g2i = defaultdict(list)
    for i, g in enumerate(gene_ids):
        g2i[g].append(i)
    g2i = {g: np.array(v) for g, v in g2i.items()}
    boot = []
    for _ in range(n_boot):
        gs  = rng.choice(unique_genes, size=len(unique_genes), replace=True)
        idx = np.concatenate([g2i[g] for g in gs])
        if y_true[idx].sum() == 0: continue
        boot.append(average_precision_score(y_true[idx], y_score[idx]))
    boot = np.array(boot)
    return base, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def build_v10B():
    inp = layers.Input(shape=(640,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out)

def get_cw(y):
    n_pos = max(int(y.sum()), 1)
    return {0: 1.0, 1: int((y == 0).sum()) / n_pos}

def train_v10b_seed(X_tr_sc, y_tr, X_brain_sc, seed):
    K.clear_session(); tf.random.set_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)
    n_val = max(int(len(y_tr) * 0.1), 100)
    vi = rng.choice(len(y_tr), n_val, replace=False)
    ti = np.setdiff1d(np.arange(len(y_tr)), vi)
    model = build_v10B()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True, verbose=0),
          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=5, verbose=0)]
    model.fit(X_tr_sc[ti], y_tr[ti],
              validation_data=(X_tr_sc[vi], y_tr[vi]),
              epochs=80, batch_size=512, class_weight=get_cw(y_tr),
              callbacks=cb, verbose=0)
    return model.predict(X_brain_sc, verbose=0).ravel()

# ── Scale ──────────────────────────────────────────────────────────────
sc = StandardScaler()
X_tr_sc     = sc.fit_transform(X_tr)
X_brain_sc  = sc.transform(X_brain)

# ── Main loop ──────────────────────────────────────────────────────────
print(f"\n{'GO Term':<26} {'Stage':<10} {'Bias_v10B':>10} {'Bias_LR':>8} {'Bias_XGB':>8}  Interpret")
print("-" * 75)

all_results = {}
v10b_bias_vals, lr_bias_vals, xgb_bias_vals = [], [], []

for go_id, go_name, stage in GO_TERMS:
    pos_syms = go_to_genes[go_id]
    y_tr = np.array([1 if s in pos_syms else 0 for s in tr_syms], dtype=np.float32)
    y_brain = np.array([1 if g in pos_syms else 0 for g in brain_gene_arr], dtype=np.float32)

    if y_tr.sum() < 10 or y_brain[protein_coding_mask].sum() < 5:
        print(f"{go_name:<26} {stage:<10} SKIP (too few positives)")
        continue

    pos_weight = (y_tr == 0).sum() / max(y_tr.sum(), 1)
    t0 = time.time()

    # v10-B: 5-seed ensemble on brain
    seed_probs = []
    for seed in SEEDS:
        probs = train_v10b_seed(X_tr_sc, y_tr, X_brain_sc, seed)
        seed_probs.append(probs)
    v10b_scores = np.mean(seed_probs, axis=0)

    # LR on brain
    clf_lr = LogisticRegression(class_weight='balanced', C=1.0,
                                 max_iter=1000, random_state=SEED)
    clf_lr.fit(X_tr_sc, y_tr)
    lr_scores = clf_lr.predict_proba(X_brain_sc)[:, 1]

    # XGB on brain
    if HAS_XGB:
        clf_xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=float(pos_weight), eval_metric='aucpr',
            use_label_encoder=False, random_state=SEED, n_jobs=4, verbosity=0)
        clf_xgb.fit(X_tr_sc, y_tr)
        xgb_scores = clf_xgb.predict_proba(X_brain_sc)[:, 1]
    else:
        xgb_scores = None

    # bias_score: protein_coding brain transcripts only
    bias_v10b = compute_bias_score(v10b_scores, brain_genebase, protein_coding_mask)
    bias_lr   = compute_bias_score(lr_scores,   brain_genebase, protein_coding_mask)
    bias_xgb  = compute_bias_score(xgb_scores,  brain_genebase, protein_coding_mask) if xgb_scores is not None else None

    # AUPRC on brain protein_coding positives
    pc_idx = np.where(protein_coding_mask)[0]
    y_br_pc = y_brain[pc_idx]
    if y_br_pc.sum() >= 5:
        v10b_auprc, ci_lo, ci_hi = bootstrap_ci_brain(
            y_br_pc, v10b_scores[pc_idx], brain_gene_arr_arr[pc_idx])
    else:
        v10b_auprc = ci_lo = ci_hi = float('nan')

    interpret = "ISOFORM-SPEC" if bias_v10b > 0.15 else \
                ("MIXED" if bias_v10b > 0.08 else "GENE-LEVEL")
    xgb_s = f"{bias_xgb:.4f}" if bias_xgb is not None else "  n/a"
    print(f"{go_name:<26} {stage:<10} {bias_v10b:>10.4f} {bias_lr:>8.4f} {xgb_s:>8}  {interpret}  ({time.time()-t0:.0f}s)")

    v10b_bias_vals.append(bias_v10b)
    lr_bias_vals.append(bias_lr)
    if bias_xgb is not None: xgb_bias_vals.append(bias_xgb)

    all_results[go_id] = {
        'go_name': go_name, 'ad_stage': stage,
        'bias_v10b': round(bias_v10b, 4),
        'bias_lr':   round(bias_lr, 4),
        'bias_xgb':  round(bias_xgb, 4) if bias_xgb else None,
        'v10b_auprc_brain': round(v10b_auprc, 4) if not np.isnan(v10b_auprc) else None,
        'v10b_ci_brain': [round(ci_lo, 4), round(ci_hi, 4)] if not np.isnan(ci_lo) else None,
        'interpret': interpret,
    }

print("-" * 75)
print(f"{'MEAN':<26} {'':10} {np.mean(v10b_bias_vals):>10.4f} {np.mean(lr_bias_vals):>8.4f} {np.mean(xgb_bias_vals):>8.4f}")
print(f"\n  v10-B brain bias: {np.mean(v10b_bias_vals):.4f}")
print(f"  LR    brain bias: {np.mean(lr_bias_vals):.4f}")
print(f"  XGB   brain bias: {np.mean(xgb_bias_vals):.4f}")
print(f"  v10-B/XGB ratio:  {np.mean(v10b_bias_vals)/np.mean(xgb_bias_vals):.2f}x")
print(f"  v10-B/LR  ratio:  {np.mean(v10b_bias_vals)/np.mean(lr_bias_vals):.2f}x")

out_path = f'{OUT_DIR}/brain_bias_score_{ts}.json'
with open(out_path, 'w') as f:
    json.dump({
        'results': all_results,
        'timestamp': ts,
        'tissue': 'brain',
        'n_transcripts': int(X_brain.shape[0]),
        'n_protein_coding': int(protein_coding_mask.sum()),
        'summary': {
            'mean_bias_v10b': round(float(np.mean(v10b_bias_vals)), 4),
            'mean_bias_lr':   round(float(np.mean(lr_bias_vals)), 4),
            'mean_bias_xgb':  round(float(np.mean(xgb_bias_vals)), 4),
            'v10b_vs_xgb_ratio': round(float(np.mean(v10b_bias_vals)/np.mean(xgb_bias_vals)), 2),
            'v10b_vs_lr_ratio':  round(float(np.mean(v10b_bias_vals)/np.mean(lr_bias_vals)), 2),
        }
    }, f, indent=2)
print(f"\nFINAL: {out_path}")
