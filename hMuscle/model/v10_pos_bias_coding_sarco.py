"""
v10_pos_bias_coding.py — coding isoform 필터링 후 pos_bias 재검증
=================================================================
paper-critic 지적 검증: pos_bias=1.196이 coding/non-coding 구별인가?

분석 레벨:
  Level 0: 전체 isoform (기존 결과 재현)
  Level 1: coding only (complete/5prime_partial ORF, >=100aa 필터)
  Level 2: coding multi-gene (coding isoform >=2개인 유전자만)

결론 기준:
  L1 ≈ L0 (Δ < 0.05) → coding/non-coding 구별이 주요 원인 아님 ✅
  L1 << L0 (Δ > 0.20) → coding/non-coding 구별이 주요 원인 → 논문 주장 수정 필요 ❌
"""

import os, sys, json, time, re
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_isoform'))

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

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/pos_bias_coding'
PEP_FILE  = '../data/top30k_isoforms.pep'

os.makedirs(OUT_DIR, exist_ok=True)

GO_TERMS = ['GO:0006914', 'GO:0043161', 'GO:0032006', 'GO:0007519', 'GO:0042692', 'GO:0055074', 'GO:0007005', 'GO:0007517']

# ─── Step 1: Build coding isoform set from TransDecoder PEP ──────────────────
print("=" * 70)
print("  Loading TransDecoder PEP → coding isoform classification")
print("=" * 70)

iso_orf_info = {}  # iso_id -> (orf_type, length)
with open(PEP_FILE) as f:
    for line in f:
        if line.startswith('>'):
            parts = line.split()
            full_id = parts[0][1:]
            iso_id = re.sub(r'\.p\d+$', '', full_id)
            orf_type = 'unknown'; length = 0
            for part in parts:
                if part.startswith('type:'):
                    orf_type = part.replace('type:', '')
                if part.startswith('len:'):
                    length = int(part.replace('len:', ''))
            if iso_id not in iso_orf_info or iso_orf_info[iso_id][1] < length:
                iso_orf_info[iso_id] = (orf_type, length)

# Coding: complete or 5prime_partial, length >= 100aa
CODING_IDS = set(
    k for k, v in iso_orf_info.items()
    if v[0] in ('complete', '5prime_partial') and v[1] >= 100
)
print(f"  Coding isoforms (complete/5prime_partial, ≥100aa): {len(CODING_IDS):,}")

# ─── Step 2: Load features ────────────────────────────────────────────────────
print("\n  Loading ESM-2 embeddings ...")
X_tr_esm2 = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te_esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

X_tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')
X_te_geneid = load_ids('my_gene_list_fixed.npy')
X_te_isoid  = load_ids('my_isoform_list_fixed.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

X_te_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in X_te_geneid]

# Per-isoform coding flag
is_coding = np.array([iso in CODING_IDS for iso in X_te_isoid])
n_coding   = is_coding.sum()
n_noncoding = (~is_coding).sum()
print(f"  Test isoforms: {len(X_te_isoid):,} total, "
      f"{n_coding:,} coding ({100*n_coding/len(X_te_isoid):.1f}%), "
      f"{n_noncoding:,} non-coding ({100*n_noncoding/len(X_te_isoid):.1f}%)")

# ─── Step 3: Bias computation helper ─────────────────────────────────────────
EPS = 1e-10

def compute_pos_bias_filtered(df, label_col='Label', gene_col='GeneID',
                               score_col='Score', coding_col='IsCoding',
                               level='all'):
    """
    level='all'         : 전체 isoform
    level='coding'      : coding isoform만
    level='coding_multi': coding isoform >=2개인 유전자만
    """
    if level == 'coding':
        sub = df[df[coding_col]].copy()
    elif level == 'coding_multi':
        sub = df[df[coding_col]].copy()
        # Keep genes with >=2 coding isoforms
        gene_coding_count = sub.groupby(gene_col).size()
        valid_genes = gene_coding_count[gene_coding_count >= 2].index
        sub = sub[sub[gene_col].isin(valid_genes)]
    else:
        sub = df.copy()

    if len(sub) == 0:
        return dict(pos_bias=np.nan, n_pos_genes=0, n_multi_genes=0, n_isos=0)

    global_std = sub[score_col].std()
    pos_genes = sub[sub[label_col] == 1][gene_col].unique()

    multi_iso = sub.groupby(gene_col).filter(lambda g: len(g) >= 2)
    pos_multi = multi_iso[multi_iso[gene_col].isin(pos_genes)]

    if len(pos_multi) == 0:
        return dict(pos_bias=np.nan, n_pos_genes=len(pos_genes),
                    n_multi_genes=0, n_isos=len(sub))

    pos_within_stds = pos_multi.groupby(gene_col)[score_col].std().dropna()
    pos_bias = pos_within_stds.mean() / (global_std + EPS)

    return dict(
        pos_bias      = float(pos_bias),
        n_pos_genes   = len(pos_genes),
        n_multi_genes = int(pos_multi[gene_col].nunique()),
        n_isos        = len(sub),
    )


# ─── Step 4: Model ────────────────────────────────────────────────────────────
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

def get_cw(y):
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    return {0: 1.0, 1: n_neg / max(n_pos, 1)}

def train_predict(X_tr, X_te, y_tr, epochs=80, batch=512):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)

    K.clear_session(); tf.random.set_seed(SEED)
    model = build_v10B()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    )
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=5, verbose=0),
    ]
    rng = np.random.RandomState(SEED)
    n_val = max(int(len(y_tr) * 0.1), 100)
    vi = rng.choice(len(y_tr), size=n_val, replace=False)
    ti = np.setdiff1d(np.arange(len(y_tr)), vi)
    model.fit(Xtr[ti], y_tr[ti], validation_data=(Xtr[vi], y_tr[vi]),
              epochs=epochs, batch_size=batch, class_weight=get_cw(y_tr),
              callbacks=cb_list, verbose=0)
    return model.predict(Xte, verbose=0).ravel()

def load_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) > 1 and go_term in p[1:]:
                pos.add(p[0])
    y_te = np.array([1 if s in pos else 0 for s in X_te_sym], dtype=np.float32)
    y_tr = np.array([1 if g in pos else 0 for g in X_tr_geneid], dtype=np.float32)
    return y_tr, y_te


# ─── Step 5: Per-GO analysis ──────────────────────────────────────────────────
all_results = []

for go in GO_TERMS:
    print(f"\n{'='*70}\n  {go}\n{'='*70}")
    y_tr, y_te = load_labels(go)
    if y_tr.sum() < 2 or y_te.sum() == 0:
        print("  SKIP"); continue

    n_pos = int(y_te.sum())
    print(f"  n_pos={n_pos}")

    t0 = time.time()
    probs = train_predict(X_tr_esm2, X_te_esm2, y_tr)
    elapsed = time.time() - t0
    auprc = float(average_precision_score(y_te, probs))
    print(f"  v10-B AUPRC={auprc:.4f} ({elapsed:.0f}s)")

    # Build full DataFrame
    df = pd.DataFrame({
        'GeneID'  : X_te_sym,
        'IsoformID': X_te_isoid,
        'Label'   : y_te.astype(int),
        'Score'   : probs,
        'IsCoding': is_coding,
    })

    # Compute pos_bias at 3 levels
    res_all  = compute_pos_bias_filtered(df, level='all')
    res_cod  = compute_pos_bias_filtered(df, level='coding')
    res_multi= compute_pos_bias_filtered(df, level='coding_multi')

    print(f"  pos_bias | All: {res_all['pos_bias']:.4f} (n_iso={res_all['n_isos']:,}, "
          f"pos_multi={res_all['n_multi_genes']})")
    print(f"           | Coding: {res_cod['pos_bias']:.4f} (n_iso={res_cod['n_isos']:,}, "
          f"pos_multi={res_cod['n_multi_genes']})")
    print(f"           | CodingMulti: {res_multi['pos_bias']:.4f} (n_iso={res_multi['n_isos']:,}, "
          f"pos_multi={res_multi['n_multi_genes']})")

    # Per-gene breakdown for positive genes — how many coding vs non-coding isoforms?
    pos_gene_set = set(df[df['Label']==1]['GeneID'].unique())
    pos_df = df[df['GeneID'].isin(pos_gene_set)]
    gene_coding_frac = pos_df.groupby('GeneID')['IsCoding'].mean()
    mixed_genes = (gene_coding_frac < 1.0).sum()
    all_noncoding_genes = (gene_coding_frac == 0.0).sum()
    print(f"  Positive genes: {len(pos_gene_set)} total, "
          f"{mixed_genes} have ≥1 non-coding isoform, "
          f"{all_noncoding_genes} are all-non-coding")

    all_results.append({
        'go': go,
        'n_pos': n_pos,
        'auprc': auprc,
        'pos_bias_all':   res_all['pos_bias'],
        'pos_bias_coding': res_cod['pos_bias'],
        'pos_bias_coding_multi': res_multi['pos_bias'],
        'n_iso_all':    res_all['n_isos'],
        'n_iso_coding': res_cod['n_isos'],
        'n_pos_genes': len(pos_gene_set),
        'n_mixed_genes': int(mixed_genes),
        'n_multi_all':   res_all['n_multi_genes'],
        'n_multi_coding': res_cod['n_multi_genes'],
        'n_multi_coding_multi': res_multi['n_multi_genes'],
        'elapsed': round(elapsed, 1),
    })


# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  POS_BIAS CODING VALIDATION SUMMARY")
print("=" * 80)
print(f"{'GO Term':<15} {'pos_bias_all':>14} {'pos_bias_coding':>16} {'coding_multi':>13} {'Δ(cod-all)':>12}")
print("-" * 80)

all_deltas = []
for r in all_results:
    delta = r['pos_bias_coding'] - r['pos_bias_all']
    all_deltas.append(delta)
    print(f"{r['go']:<15} {r['pos_bias_all']:>14.4f} {r['pos_bias_coding']:>16.4f} "
          f"{r['pos_bias_coding_multi']:>13.4f} {delta:>+12.4f}")

print("-" * 80)
macro_all    = np.mean([r['pos_bias_all'] for r in all_results])
macro_coding = np.mean([r['pos_bias_coding'] for r in all_results])
macro_multi  = np.mean([r['pos_bias_coding_multi'] for r in all_results])
macro_delta  = np.mean(all_deltas)
print(f"{'Macro':<15} {macro_all:>14.4f} {macro_coding:>16.4f} "
      f"{macro_multi:>13.4f} {macro_delta:>+12.4f}")

print("\n" + "=" * 80)
print("  INTERPRETATION")
print("=" * 80)
if abs(macro_delta) < 0.05:
    verdict = "✅ CONFIRMED: pos_bias is NOT driven by coding/non-coding distinction"
    interpretation = (
        f"Coding-only pos_bias ({macro_coding:.3f}) ≈ All-isoform pos_bias ({macro_all:.3f}).\n"
        f"Δ={macro_delta:+.4f} (< 0.05 threshold). The paper's isoform discrimination\n"
        f"claim is genuinely capturing within-coding-isoform functional differences."
    )
elif abs(macro_delta) < 0.15:
    verdict = "⚠️  PARTIAL: coding/non-coding distinction contributes but is not dominant"
    interpretation = (
        f"Coding-only pos_bias ({macro_coding:.3f}) vs All ({macro_all:.3f}).\n"
        f"Δ={macro_delta:+.4f} (0.05-0.15 range). Mention coding fraction (98%) in Methods.\n"
        f"The genuine functional discrimination claim still holds but needs nuancing."
    )
else:
    verdict = "❌ CONCERN: pos_bias substantially driven by coding/non-coding distinction"
    interpretation = (
        f"Coding-only pos_bias ({macro_coding:.3f}) vs All ({macro_all:.3f}).\n"
        f"Δ={macro_delta:+.4f} (> 0.15). Paper claim needs significant revision."
    )

print(f"\n  {verdict}")
print(f"\n  {interpretation}")
print(f"\n  Non-coding isoform fraction: {n_noncoding}/{len(X_te_isoid)} ({100*n_noncoding/len(X_te_isoid):.1f}%)")
print(f"  → Even if all pos_bias came from coding/non-coding, only {100*n_noncoding/len(X_te_isoid):.1f}%")
print(f"    of isoforms are non-coding — arithmetically cannot explain pos_bias~1.2")

# Save
ts = time.strftime('%Y%m%d_%H%M')
out_path = f'{OUT_DIR}/pos_bias_coding_{ts}.json'
with open(out_path, 'w') as f:
    json.dump({
        'results': all_results,
        'macro': {
            'pos_bias_all': float(macro_all),
            'pos_bias_coding': float(macro_coding),
            'pos_bias_coding_multi': float(macro_multi),
            'delta': float(macro_delta),
        },
        'n_coding': int(n_coding),
        'n_noncoding': int(n_noncoding),
        'coding_fraction': float(n_coding/len(X_te_isoid)),
        'verdict': verdict,
        'timestamp': ts,
    }, f, indent=2)
print(f"\n[Saved] {out_path}")
