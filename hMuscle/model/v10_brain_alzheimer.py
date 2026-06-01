"""
v10_brain_alzheimer.py — Brain Tissue Validation (Normal vs Alzheimer's)
=========================================================================
목적: v10-B의 muscle-trained 모델이 brain tissue에서도 의미있는 GO function
     예측을 하는지 검증 (조직 간 일반화 가능성)

데이터:
  - Brain: /home/dhkim1674/Samsung_Alzheimer/03_AnnData/adata_transcript_for_UMAP.h5ad
    - 63,884 cells (AD: 30,879 / Control: 23,712 / Active control: 9,293)
    - 60,165 known ENST IDs (no version numbers)
    - Cell types: Astrocyte, Excitatory/Inhibitory neuron, Microglia, OPC, Oligodendrocyte, Vascular
  - Muscle ESM-2: esm2_embeddings_t30_150M.npy (36,748 isoforms)
    - 35,198 ENST (with version) + 1,550 BambuTx
    - Overlap with brain: 20,437 transcripts

분석 전략:
  1. Brain-muscle ENST overlap → 기존 ESM-2 embedding 재사용
  2. v10-B 모델을 muscle training data로 학습 후 overlapping 뇌 이소폼에 적용
  3. Pseudo-bulk aggregation: (condition × cell_type)별 평균 발현으로 집계
  4. GO function activity 비교: AD vs Control, per cell type
  5. Isoform switch 탐지: differential expression of high-score isoforms

GO terms 선택 (AD 관련성 근거):
  - GO:0006914 Autophagy: autophagosome clearance → AD amyloid/tau pathology
  - GO:0007005 Mitochondrion org: 미토콘드리아 기능 → AD 초기 병인
  - GO:0043161 Proteasome-UPS: 단백질 분해 → AD ubiquitinated aggregates

실행:
  /home/welcome1/miniconda3/envs/isoform_env/bin/python -u v10_brain_alzheimer.py
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import h5py
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
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

# ─── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = '../data'
ANNOT_DIR  = '../data/raw_data/data/annotations'
ID_DIR     = '../data/raw_data/data/id_lists'
BRAIN_H5AD = '/home/dhkim1674/Samsung_Alzheimer/03_AnnData/adata_transcript_for_UMAP.h5ad'
OUT_DIR    = '../../reports/brain_alzheimer'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS     = [42, 123, 456, 789, 2024]
N_BOOT    = 300
SEP_THRESHOLD = 0.060

# AD-relevant GO terms for brain validation
GO_TERMS_BRAIN = [
    ('GO:0006914', 'Autophagy',         'AD amyloid/tau clearance'),
    ('GO:0007005', 'Mitochondrion org', 'AD early pathogenesis'),
    ('GO:0043161', 'Proteasome-UPS',    'AD ubiquitinated aggregates'),
]

print("=" * 70)
print("  v10-B Brain Tissue Validation (Normal vs Alzheimer's)")
print("=" * 70)

# ─── Load muscle data ────────────────────────────────────────────────────────────
def load_ids(p):
    a = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]

X_tr = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
tr_genes = load_ids(f'{ID_DIR}/train_gene_list.npy')
te_isos  = load_ids('my_isoform_list_fixed.npy')   # ENST with version + BambuTx
te_genes = load_ids('my_gene_list_fixed.npy')

ensg2sym = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ensg2sym[p[0]] = p[4]

te_syms = [ensg2sym.get(g.split('.')[0], g.split('.')[0]) for g in te_genes]
print(f"  Muscle train {X_tr.shape}, test {X_te.shape}")


# ─── Build brain-muscle mapping ──────────────────────────────────────────────────
print("\n[1] Building brain-muscle ENST overlap ...")

f_brain = h5py.File(BRAIN_H5AD, 'r')
brain_enst_arr  = [x.decode() for x in f_brain['var']['ENST_ID'][()]]
gene_name_cats  = [x.decode() for x in f_brain['var']['gene_name']['categories'][()]]
gene_name_codes = f_brain['var']['gene_name']['codes'][()]
brain_gene_arr  = [gene_name_cats[c] for c in gene_name_codes]

# Get condition and cell type per cell
cond_cats    = [c.decode() for c in f_brain['obs']['condition']['categories'][()]]
cond_codes   = f_brain['obs']['condition']['codes'][()]
ct_cats      = [c.decode() for c in f_brain['obs']['cell_type']['categories'][()]]
ct_codes     = f_brain['obs']['cell_type']['codes'][()]
cell_barcodes = [b.decode() for b in f_brain['obs']['_index'][()]]

# Map muscle ENST → index in test set (no-version key)
muscle_enst_base_to_idx = defaultdict(list)
for i, iso_id in enumerate(te_isos):
    base = iso_id.split('.')[0]
    muscle_enst_base_to_idx[base].append(i)

# Build overlap: brain_idx → muscle_idx (take first match if duplicates)
brain_to_muscle_idx = {}  # brain var index → muscle test set index
for brain_idx, enst_id in enumerate(brain_enst_arr):
    if enst_id in muscle_enst_base_to_idx:
        # If duplicates in muscle, take the one at the first index
        brain_to_muscle_idx[brain_idx] = muscle_enst_base_to_idx[enst_id][0]

overlap_brain_idxs  = np.array(sorted(brain_to_muscle_idx.keys()), dtype=np.int32)
overlap_muscle_idxs = np.array([brain_to_muscle_idx[i] for i in overlap_brain_idxs], dtype=np.int32)
print(f"  Overlap: {len(overlap_brain_idxs)} transcripts (brain idx → muscle idx)")
print(f"  Overlap %: {len(overlap_brain_idxs)/len(brain_enst_arr)*100:.1f}% of brain")

# Overlap ESM-2 embeddings
X_overlap = X_te[overlap_muscle_idxs]  # (n_overlap, 640)


# ─── Pseudo-bulk aggregation ─────────────────────────────────────────────────────
print("\n[2] Building pseudo-bulk profiles (condition × cell_type) ...")
print("  Reading expression matrix (h5py dense, row-by-row for overlapping columns)...")

# Build group masks
conditions = sorted(set(cond_cats))
cell_types  = sorted(set(ct_cats))
groups = [(c, ct) for c in conditions for ct in cell_types]

# For each group, aggregate expression of overlapping transcripts
# X is (63884, 60165) dense float32 — read in chunks to avoid OOM
# Strategy: read row by row for each group, sum only overlapping columns

group_cells = {}  # (cond, ct) → list of cell row indices
for cell_idx in range(len(cell_barcodes)):
    c  = cond_cats[cond_codes[cell_idx]]
    ct = ct_cats[ct_codes[cell_idx]]
    key = (c, ct)
    if key not in group_cells:
        group_cells[key] = []
    group_cells[key].append(cell_idx)

n_overlap = len(overlap_brain_idxs)
X_raw = f_brain['X']          # dense matrix (63884, 60165)
CHUNK = 500                    # cells per read chunk

pseudo_bulk = {}   # (cond, ct) → mean expression vector (n_overlap,)
n_cells_per_group = {}

for (cond, ct), cell_idxs in sorted(group_cells.items()):
    n = len(cell_idxs)
    n_cells_per_group[(cond, ct)] = n
    if n < 10:  # skip tiny groups
        continue

    expr_sum = np.zeros(n_overlap, dtype=np.float64)
    sorted_idxs = sorted(cell_idxs)

    for start in range(0, n, CHUNK):
        batch = sorted_idxs[start:start + CHUNK]
        # Read batch rows, extract overlap columns
        rows = X_raw[batch][:, overlap_brain_idxs]  # (batch, n_overlap)
        expr_sum += rows.sum(axis=0)

    pseudo_bulk[(cond, ct)] = (expr_sum / n).astype(np.float32)
    print(f"  {cond:15} {ct:20}: {n:5} cells aggregated")

f_brain.close()
print(f"  Pseudo-bulk groups computed: {len(pseudo_bulk)}")


# ─── Load GO labels ──────────────────────────────────────────────────────────────
go_to_genes = defaultdict(set)
with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        for go in parts[1:]:
            go_to_genes[go].add(parts[0])

def load_labels(go_term):
    pos = go_to_genes[go_term]
    y_te = np.array([1 if s in pos else 0 for s in te_syms], dtype=np.float32)
    y_tr = np.array([1 if g in pos else 0 for g in tr_genes], dtype=np.float32)
    return y_tr, y_te


# ─── sep_cosine ──────────────────────────────────────────────────────────────────
def sep_cosine(X, y, rs=42):
    rng = np.random.RandomState(rs)
    pi = np.where(y == 1)[0]; ni = np.where(y == 0)[0]
    if len(pi) < 5:
        return np.nan
    if len(ni) > 2000: ni = rng.choice(ni, 2000, replace=False)
    if len(pi) > 2000: pi = rng.choice(pi, 2000, replace=False)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    pc = Xn[pi].mean(0); nc = Xn[ni].mean(0)
    inter = 1 - np.dot(pc, nc) / (np.linalg.norm(pc) * np.linalg.norm(nc) + 1e-10)
    Xp = Xn[pi[:500]]
    m = Xp @ Xp.T; up = np.triu(np.ones(m.shape, bool), k=1)
    return float(inter / max(1 - m[up].mean(), 1e-10))


# ─── v10-B model ─────────────────────────────────────────────────────────────────
def build_v10B():
    inp = layers.Input(shape=(640,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name='v10B')

def get_cw(y):
    n_pos = max(int(y.sum()), 1)
    return {0: 1.0, 1: int((y == 0).sum()) / n_pos}

def train_v10B_and_predict(X_tr_, X_te_full, y_tr_, seed, epochs=80, batch=512):
    """Train on muscle, predict on all test isoforms (including overlap)."""
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr_)
    Xte = sc.transform(X_te_full)
    K.clear_session(); tf.random.set_seed(seed); np.random.seed(seed)
    model = build_v10B()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
    rng = np.random.RandomState(seed)
    n_val = max(int(len(y_tr_) * 0.1), 100)
    vi = rng.choice(len(y_tr_), n_val, replace=False)
    ti = np.setdiff1d(np.arange(len(y_tr_)), vi)
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True, verbose=0),
          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=5, verbose=0)]
    model.fit(Xtr[ti], y_tr_[ti], validation_data=(Xtr[vi], y_tr_[vi]),
              epochs=epochs, batch_size=batch, class_weight=get_cw(y_tr_),
              callbacks=cb, verbose=0)
    return model.predict(Xte, verbose=0).ravel()  # (n_test,)


def train_lr(X_tr_, X_te_, y_tr_):
    sc = StandardScaler()
    clf = LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000, random_state=42)
    clf.fit(sc.fit_transform(X_tr_), y_tr_)
    return clf.predict_proba(sc.transform(X_te_))[:, 1]


# ─── Main analysis ────────────────────────────────────────────────────────────────
print("\n[3] Training v10-B per GO term and computing brain GO activity ...")
ts = time.strftime('%Y%m%d_%H%M')
all_go_results = {}
overlap_syms = [te_syms[i] for i in overlap_muscle_idxs]

for go_id, go_name, ad_relevance in GO_TERMS_BRAIN:
    print(f"\n{'='*70}")
    print(f"  {go_id} {go_name}  [{ad_relevance}]")
    print(f"{'='*70}")

    y_tr, y_te = load_labels(go_id)
    n_pos_tr = int(y_tr.sum()); n_pos_te = int(y_te.sum())
    print(f"  n_pos: train={n_pos_tr}, test={n_pos_te}")

    sep = sep_cosine(X_te, y_te)
    go_type = 'A' if sep >= SEP_THRESHOLD else 'B'
    print(f"  sep_cosine={sep:.4f}  Type-{go_type}")

    # Train v10-B (5 seeds) and predict on all test isoforms
    seed_probs_all = []
    for seed in SEEDS:
        t0 = time.time()
        probs = train_v10B_and_predict(X_tr, X_te, y_tr, seed)
        auprc_muscle = float(average_precision_score(y_te, probs))
        seed_probs_all.append(probs)
        print(f"  seed={seed}: AUPRC_muscle={auprc_muscle:.4f} ({time.time()-t0:.0f}s)")

    avg_probs_all  = np.mean(seed_probs_all, axis=0)   # (n_test,)
    muscle_auprc   = float(average_precision_score(y_te, avg_probs_all))
    print(f"  Muscle AUPRC (ensemble): {muscle_auprc:.4f}")

    # Extract scores for overlap isoforms
    overlap_scores = avg_probs_all[overlap_muscle_idxs]  # (n_overlap,)
    n_pos_overlap  = int(np.array([1 if s in go_to_genes[go_id] else 0 for s in overlap_syms]).sum())
    print(f"  Overlap positive isoforms: {n_pos_overlap} / {len(overlap_scores)}")

    # Compute GO activity per pseudo-bulk group
    # activity = weighted mean score (weight = expression level)
    group_activity = {}
    for (cond, ct), expr_vec in pseudo_bulk.items():
        # expr_vec: (n_overlap,) mean expression across cells in group
        total_expr = expr_vec.sum()
        if total_expr < 1e-6:
            group_activity[(cond, ct)] = np.nan
        else:
            # Weighted mean of v10-B scores by expression
            group_activity[(cond, ct)] = float(np.dot(expr_vec, overlap_scores) / total_expr)

    # LR baseline activity
    lr_probs_all   = train_lr(X_tr, X_te, y_tr)
    lr_overlap     = lr_probs_all[overlap_muscle_idxs]
    group_activity_lr = {}
    for (cond, ct), expr_vec in pseudo_bulk.items():
        total_expr = expr_vec.sum()
        if total_expr < 1e-6:
            group_activity_lr[(cond, ct)] = np.nan
        else:
            group_activity_lr[(cond, ct)] = float(np.dot(expr_vec, lr_overlap) / total_expr)

    # Build summary table
    print(f"\n  GO Activity by Group (v10-B vs LR):")
    print(f"  {'Condition':<17} {'CellType':<22} {'n_cells':>7} {'v10-B':>7} {'LR':>7}")
    print(f"  {'-'*62}")
    rows = []
    for (cond, ct) in sorted(group_activity.keys()):
        n = n_cells_per_group.get((cond, ct), 0)
        v10b_act = group_activity[(cond, ct)]
        lr_act   = group_activity_lr.get((cond, ct), np.nan)
        if np.isnan(v10b_act): continue
        v10b_str = f"{v10b_act:.4f}"
        lr_str   = f"{lr_act:.4f}" if not np.isnan(lr_act) else 'N/A'
        print(f"  {cond:<17} {ct:<22} {n:>7} {v10b_str:>7} {lr_str:>7}")
        rows.append({'condition': cond, 'cell_type': ct, 'n_cells': n,
                     'v10b_activity': v10b_act, 'lr_activity': lr_act})

    # AD vs Control per cell type: Mann-Whitney on raw cell-level (approximated by group mean)
    # For proper stats: compare AD groups vs Control groups across cell types
    ad_acts    = [r['v10b_activity'] for r in rows if r['condition'] == 'AD']
    ctrl_acts  = [r['v10b_activity'] for r in rows if r['condition'] == 'Control']
    if len(ad_acts) >= 3 and len(ctrl_acts) >= 3:
        stat, pval = stats.mannwhitneyu(ad_acts, ctrl_acts, alternative='two-sided')
        print(f"\n  AD vs Control (across cell types): "
              f"mean_AD={np.nanmean(ad_acts):.4f}  mean_Ctrl={np.nanmean(ctrl_acts):.4f}  "
              f"MW p={pval:.4f}")

    all_go_results[go_id] = {
        'go_name': go_name, 'ad_relevance': ad_relevance,
        'sep_cosine': round(sep, 4), 'go_type': go_type,
        'muscle_auprc': round(muscle_auprc, 4),
        'n_pos_overlap': n_pos_overlap,
        'group_activity': {f"{c}|{ct}": v for (c, ct), v in group_activity.items()},
        'group_activity_lr': {f"{c}|{ct}": v for (c, ct), v in group_activity_lr.items()},
        'pseudo_bulk_rows': rows,
    }


# ─── Figure ──────────────────────────────────────────────────────────────────────
print("\n[4] Generating figure ...")

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.4)
cond_colors = {'AD': '#E63946', 'Control': '#2196F3', 'Active control': '#FF9800'}
ct_order = ['Excitatory neuron', 'Inhibitory neuron', 'Astrocyte',
            'Oligodendrocyte', 'OPC', 'Microglia', 'Vascular cell']

for gi, (go_id, go_name, _) in enumerate(GO_TERMS_BRAIN):
    res = all_go_results.get(go_id, {})
    rows = res.get('pseudo_bulk_rows', [])
    if not rows:
        continue

    ax = fig.add_subplot(gs[gi // 3, gi % 3])
    df = pd.DataFrame(rows)

    x = np.arange(len(ct_order))
    width = 0.25
    offsets = {'AD': -width, 'Control': 0, 'Active control': width}

    for cond in ['AD', 'Control', 'Active control']:
        vals = []
        for ct in ct_order:
            row = df[(df['condition'] == cond) & (df['cell_type'] == ct)]
            vals.append(row['v10b_activity'].values[0] if len(row) > 0 else np.nan)
        bars = ax.bar(x + offsets[cond], vals, width * 0.9,
                      label=cond, color=cond_colors[cond], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([ct.replace(' ', '\n') for ct in ct_order], fontsize=7)
    ax.set_ylabel('GO Activity Score', fontsize=9)
    go_type = res.get('go_type', '?')
    muscle_auprc = res.get('muscle_auprc', 0)
    ax.set_title(f"{go_name}\n(Type-{go_type}, muscle AUPRC={muscle_auprc:.3f})",
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

# Bottom row: AD vs Control difference per cell type across all GO terms
ax_diff = fig.add_subplot(gs[1, :])
all_diffs = []
for go_id, go_name, _ in GO_TERMS_BRAIN:
    res = all_go_results.get(go_id, {})
    rows = res.get('pseudo_bulk_rows', [])
    if not rows:
        continue
    df = pd.DataFrame(rows)
    for ct in ct_order:
        ad_row   = df[(df['condition'] == 'AD') & (df['cell_type'] == ct)]
        ctrl_row = df[(df['condition'] == 'Control') & (df['cell_type'] == ct)]
        if len(ad_row) > 0 and len(ctrl_row) > 0:
            diff = ad_row['v10b_activity'].values[0] - ctrl_row['v10b_activity'].values[0]
            all_diffs.append({'go': go_name[:12], 'cell_type': ct, 'diff': diff})

if all_diffs:
    diff_df = pd.DataFrame(all_diffs)
    pivot = diff_df.pivot(index='cell_type', columns='go', values='diff')
    pivot = pivot.reindex(index=[ct for ct in ct_order if ct in pivot.index])
    x_ct = np.arange(len(pivot.index))
    go_cols = pivot.columns.tolist()
    w = 0.25
    offsets_go = {g: (i - len(go_cols)/2 + 0.5) * w for i, g in enumerate(go_cols)}
    colors_go = ['#8B4513', '#228B22', '#4B0082']
    for i, go in enumerate(go_cols):
        ax_diff.bar(x_ct + offsets_go[go], pivot[go].values, w * 0.9,
                    label=go, color=colors_go[i % len(colors_go)], alpha=0.8)
    ax_diff.axhline(0, color='black', linewidth=0.5)
    ax_diff.set_xticks(x_ct)
    ax_diff.set_xticklabels([ct.replace(' ', '\n') for ct in pivot.index], fontsize=8)
    ax_diff.set_ylabel('AD - Control GO Activity\n(v10-B score)', fontsize=9)
    ax_diff.set_title('Differential GO Activity: AD vs Control by Cell Type', fontsize=10, fontweight='bold')
    ax_diff.legend(fontsize=8)
    ax_diff.grid(axis='y', alpha=0.3)

plt.suptitle("v10-B Brain Tissue Validation: Normal vs Alzheimer's\n"
             f"(v10-B trained on muscle, applied to {len(overlap_brain_idxs)} overlapping isoforms)",
             fontsize=11, fontweight='bold', y=1.01)

fig_path = f'{OUT_DIR}/brain_alzheimer_{ts}.pdf'
fig_path_png = f'{OUT_DIR}/brain_alzheimer_{ts}.png'
plt.savefig(fig_path, bbox_inches='tight', dpi=150)
plt.savefig(fig_path_png, bbox_inches='tight', dpi=150)
plt.close()
print(f"  Figure saved: {fig_path}")

# Save JSON
out_path = f'{OUT_DIR}/brain_alzheimer_{ts}.json'
with open(out_path, 'w') as f:
    json.dump({
        'timestamp': ts,
        'n_overlap': int(len(overlap_brain_idxs)),
        'n_brain_total': 60165,
        'overlap_pct': round(len(overlap_brain_idxs) / 60165 * 100, 1),
        'n_cells': {'AD': 30879, 'Control': 23712, 'Active control': 9293},
        'go_terms': all_go_results,
    }, f, indent=2)
print(f"FINAL: {out_path}")
