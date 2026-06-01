"""
v10_brain_fiu_analysis.py — Brain Tissue AD vs Control FIU Analysis
====================================================================
목적: v10-B (근육 학습) → 뇌 60,165 이소폼 기능 점수 예측
     → Functional Isoform Usage (FIU) 지표로 AD vs Control 비교

분석 전략:
  Primary   : FISI (Functional Isoform Switch Index) — PSI 기반
  Secondary : WFU  (Weighted Functional Usage)
  Validation: FDS  (Functional Dominance Score)

GO Terms (13개, AD 발전 단계별):
  Early  : Ca2+ signaling, Ca2+ homeostasis, Mitochondrion org, ER stress
  Mid    : Microtubule cytoskeleton, Synaptic transmission, Autophagy
  Mid-Late: Proteasome-UPS, Oxidative stress
  Late   : Inflammatory response, Innate immune, Phagocytosis, Myelination

세포 유형별 독립 분석 (7개 cell types)

데이터:
  brain_esm2_60165.npy      (60165, 640) — full brain ESM-2 embeddings
  layers/counts (CSC sparse) — raw integer counts
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.sparse as sp
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import h5py
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

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR   = '../data'
ANNOT_DIR  = '../data/raw_data/data/annotations'
ID_DIR     = '../data/raw_data/data/id_lists'
BRAIN_H5AD = '/home/dhkim1674/Samsung_Alzheimer/03_AnnData/adata_transcript_for_UMAP.h5ad'
BRAIN_EMB  = '../data/brain_esm2/brain_esm2_60165.npy'
OUT_DIR    = '../../reports/brain_fiu'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]
SEP_THRESHOLD = 0.060
MIN_TOTAL_COUNTS = 10   # per gene per group, minimum total counts
MIN_ISOFORMS     = 2    # minimum number of isoforms per gene for FIU

# ─── GO Terms: 13개 AD-relevant ───────────────────────────────────────────────
GO_TERMS = [
    # (go_id, name, ad_stage, primary_cell_types)
    ('GO:0007204', 'Ca2+ signaling',          'Early',    ['Excitatory neuron', 'Inhibitory neuron']),
    ('GO:0055074', 'Ca2+ homeostasis',         'Early',    ['Excitatory neuron', 'Inhibitory neuron']),
    ('GO:0007005', 'Mitochondrion org',        'Early',    ['Excitatory neuron', 'Astrocyte']),
    ('GO:0034976', 'ER stress response',       'Early',    ['Excitatory neuron', 'Astrocyte']),
    ('GO:0000226', 'Microtubule cytoskeleton', 'Mid',      ['Excitatory neuron', 'Inhibitory neuron']),
    ('GO:0007268', 'Synaptic transmission',    'Mid',      ['Excitatory neuron', 'Inhibitory neuron']),
    ('GO:0006914', 'Autophagy',                'Mid',      ['Excitatory neuron', 'Microglia']),
    ('GO:0043161', 'Proteasome-UPS',           'Mid-Late', ['Excitatory neuron', 'Microglia']),
    ('GO:0006979', 'Oxidative stress',         'Mid-Late', ['Excitatory neuron', 'Inhibitory neuron']),
    ('GO:0006954', 'Inflammatory response',    'Late',     ['Microglia', 'Astrocyte']),
    ('GO:0045087', 'Innate immune response',   'Late',     ['Microglia']),
    ('GO:0006909', 'Phagocytosis',             'Late',     ['Microglia']),
    ('GO:0042552', 'Myelination',              'Late',     ['Oligodendrocyte', 'OPC']),
]

print("=" * 70)
print("  v10-B Brain FIU Analysis: AD vs Control")
print("=" * 70)

# ─── Load muscle training data (for v10-B training) ──────────────────────────
def load_ids(p):
    a = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]

X_tr    = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
tr_genes = load_ids(f'{ID_DIR}/train_gene_list.npy')

ensg2sym = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ensg2sym[p[0]] = p[4]
tr_syms = [ensg2sym.get(g.split('.')[0], g.split('.')[0]) for g in tr_genes]
print(f"  Muscle train: {X_tr.shape}")

# ─── Load brain ESM-2 embeddings (60,165) ─────────────────────────────────────
print(f"  Loading brain embeddings: {BRAIN_EMB}")
X_brain = np.load(BRAIN_EMB).astype(np.float32)   # (60165, 640)
print(f"  Brain embeddings: {X_brain.shape}")

# ─── Load brain metadata ──────────────────────────────────────────────────────
print("\n[1] Loading brain metadata ...")
f_brain = h5py.File(BRAIN_H5AD, 'r')
brain_enst_arr  = [x.decode() for x in f_brain['var']['ENST_ID'][()]]
gene_name_cats  = [x.decode() for x in f_brain['var']['gene_name']['categories'][()]]
gene_name_codes = f_brain['var']['gene_name']['codes'][()]
brain_gene_arr  = [gene_name_cats[c] for c in gene_name_codes]   # gene symbol per transcript

cond_cats  = [c.decode() for c in f_brain['obs']['condition']['categories'][()]]
cond_codes = f_brain['obs']['condition']['codes'][()]
ct_cats    = [c.decode() for c in f_brain['obs']['cell_type']['categories'][()]]
ct_codes   = f_brain['obs']['cell_type']['codes'][()]

# ─── Load raw counts (CSC sparse) ────────────────────────────────────────────
print("[2] Loading raw counts (layers/counts, CSC) ...")
lc = f_brain['layers']['counts']
csc_data   = lc['data'][()]
csc_inds   = lc['indices'][()]
csc_indptr = lc['indptr'][()]
f_brain.close()

counts_csc = sp.csc_matrix((csc_data, csc_inds, csc_indptr), shape=(63884, 60165))
counts_csr = counts_csc.tocsr()   # convert for row-wise access
print(f"  counts shape: {counts_csr.shape}, nnz={counts_csr.nnz}")

# ─── Build per-group (condition × cell_type) count matrices ──────────────────
print("[3] Aggregating counts per (condition × cell_type) ...")
group_cells = defaultdict(list)
for cell_idx in range(counts_csr.shape[0]):
    cond = cond_cats[cond_codes[cell_idx]]
    ct   = ct_cats[ct_codes[cell_idx]]
    group_cells[(cond, ct)].append(cell_idx)

# Sum counts per group → (n_transcripts,) vector per group
group_counts = {}   # (cond, ct) → np.array (60165,)
for (cond, ct), cell_idxs in sorted(group_cells.items()):
    if len(cell_idxs) < 10:
        continue
    rows = counts_csr[cell_idxs]       # (n_cells, 60165) sparse
    summed = np.asarray(rows.sum(axis=0)).ravel()   # (60165,)
    group_counts[(cond, ct)] = summed.astype(np.float32)
    print(f"  {cond:15} {ct:22}: {len(cell_idxs):5} cells, "
          f"total_counts={summed.sum():.0f}")

print(f"  Groups: {len(group_counts)}")

# ─── Filter to protein_coding transcripts only ────────────────────────────────
# Non-coding transcripts (lncRNA, retained_intron, NMD, etc.) have spurious
# TransDecoder ORFs → ESM-2 embeddings are meaningless for GO function prediction.
# FIU analysis should only use protein_coding ENST isoforms.
import re as _re
print("[2b] Filtering to protein_coding transcripts ...")
enst_to_type = {}
_ref_gtf = '/home/dhkim1674/reference/refdata-gex-GRCh38-2024-A/genes/genes.gtf'
with open(_ref_gtf) as _f:
    for _line in _f:
        if '\ttranscript\t' not in _line:
            continue
        _m_id   = _re.search(r'transcript_id \"(ENST\d+)\"', _line)
        _m_type = _re.search(r'transcript_type \"([^\"]+)\"', _line)
        if _m_id and _m_type:
            enst_to_type[_m_id.group(1)] = _m_type.group(1)

n_total_brain = len(brain_enst_arr)
protein_coding_mask = np.array([
    enst_to_type.get(enst, 'unknown') == 'protein_coding'
    for enst in brain_enst_arr
], dtype=bool)
n_coding = int(protein_coding_mask.sum())
print(f"  protein_coding: {n_coding}/{n_total_brain} ({n_coding/n_total_brain*100:.1f}%)")

# ─── Build gene → isoform index mapping (protein_coding only) ─────────────────
gene_to_brain_idxs = defaultdict(list)
for var_idx, gene_sym in enumerate(brain_gene_arr):
    if protein_coding_mask[var_idx]:
        gene_to_brain_idxs[gene_sym].append(var_idx)

multi_isoform_genes = {g: idxs for g, idxs in gene_to_brain_idxs.items()
                       if len(idxs) >= MIN_ISOFORMS}
print(f"  Multi-isoform protein_coding genes (≥{MIN_ISOFORMS}): {len(multi_isoform_genes)}")

# ─── Load GO annotations ──────────────────────────────────────────────────────
go_to_genes = defaultdict(set)
with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2: continue
        for go in parts[1:]:
            go_to_genes[go].add(parts[0])

# ─── sep_cosine ───────────────────────────────────────────────────────────────
def sep_cosine(X, y, rs=42):
    rng = np.random.RandomState(rs)
    pi = np.where(y==1)[0]; ni = np.where(y==0)[0]
    if len(pi) < 5: return np.nan
    if len(ni) > 2000: ni = rng.choice(ni, 2000, replace=False)
    if len(pi) > 2000: pi = rng.choice(pi, 2000, replace=False)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    pc = Xn[pi].mean(0); nc = Xn[ni].mean(0)
    inter = 1 - np.dot(pc,nc)/(np.linalg.norm(pc)*np.linalg.norm(nc)+1e-10)
    Xp = Xn[pi[:500]]; m = Xp @ Xp.T
    up = np.triu(np.ones(m.shape, bool), k=1)
    return float(inter / max(1-m[up].mean(), 1e-10))

# ─── v10-B model ──────────────────────────────────────────────────────────────
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
    return {0: 1.0, 1: int((y==0).sum()) / n_pos}

def train_and_predict_brain(X_tr_, y_tr_, X_brain_, seed, epochs=80, batch=512):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr_)
    Xbr = sc.transform(X_brain_)
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
    return model.predict(Xbr, verbose=0).ravel()   # (60165,)


# ─── FIU computation functions ────────────────────────────────────────────────
def compute_fiu(gene_to_idxs, group_counts_dict, v10b_scores, go_positive_syms,
                min_counts=10):
    """
    Returns per-gene FIU metrics for each (condition, cell_type) group.

    gene_to_idxs: {gene_sym: [brain_var_idx, ...]}  (multi-isoform only)
    group_counts_dict: {(cond, ct): np.array (60165,)}
    v10b_scores: np.array (60165,) — ensemble mean

    Returns: DataFrame with columns:
      gene, group_cond, group_ct, n_isoforms,
      FISI, WFU, FDS,
      is_go_positive  (gene in go_positive_syms)
    """
    rows = []

    for gene, idxs in gene_to_idxs.items():
        scores = v10b_scores[idxs]       # (n_iso,)
        n_iso  = len(idxs)
        score_median = np.median(scores)

        for (cond, ct), cnt_vec in group_counts_dict.items():
            cnts = cnt_vec[idxs].astype(np.float64)    # (n_iso,)
            total = cnts.sum()
            if total < min_counts:
                continue

            psi = cnts / total                         # (n_iso,) sum=1

            # FISI: fraction of expression going to above-median score isoforms
            high_mask = scores > score_median
            if high_mask.sum() == 0 or high_mask.sum() == n_iso:
                continue   # degenerate case
            fisi = float(psi[high_mask].sum())

            # WFU: expression-weighted mean score
            wfu = float(np.dot(psi, scores))

            # FDS: score of the dominant isoform
            dom_idx = np.argmax(psi)
            fds = float(scores[dom_idx])

            rows.append({
                'gene': gene,
                'cond': cond, 'ct': ct,
                'n_iso': n_iso,
                'total_counts': total,
                'FISI': fisi, 'WFU': wfu, 'FDS': fds,
                'is_go_pos': gene in go_positive_syms,
            })

    return pd.DataFrame(rows)


def test_ad_vs_control(df_fiu, metric='FISI'):
    """
    For each cell type, test: does ΔFIU(AD - Control) differ from 0?
    Only include GO-positive genes. Returns stats per cell type.
    """
    results = []
    df_go = df_fiu[df_fiu['is_go_pos']]

    for ct in df_go['ct'].unique():
        df_ct = df_go[df_go['ct'] == ct]
        # Paired: same gene in AD and Control
        genes_ad   = df_ct[df_ct['cond'] == 'AD'].set_index('gene')[metric]
        genes_ctrl = df_ct[df_ct['cond'] == 'Control'].set_index('gene')[metric]
        common = genes_ad.index.intersection(genes_ctrl.index)
        if len(common) < 5:
            continue
        delta = genes_ad[common] - genes_ctrl[common]
        if (delta == 0).all() or delta.std() == 0:
            continue  # no variance to test
        try:
            stat, pval = stats.wilcoxon(delta, alternative='two-sided',
                                        zero_method='zsplit')
        except ValueError:
            continue
        results.append({
            'cell_type': ct,
            'metric': metric,
            'n_genes': len(common),
            'mean_delta': float(delta.mean()),
            'median_delta': float(delta.median()),
            'pval': float(pval),
            'direction': 'AD↓' if delta.mean() < 0 else 'AD↑',
        })

    return pd.DataFrame(results).sort_values('pval')


# ─── Main loop: per GO term ────────────────────────────────────────────────────
print("\n[4] Per-GO-term v10-B training + FIU analysis ...")
ts = time.strftime('%Y%m%d_%H%M')
all_results = {}

for go_id, go_name, ad_stage, primary_cts in GO_TERMS:
    print(f"\n{'='*70}")
    print(f"  {go_id} {go_name}  [AD {ad_stage}]")
    print(f"{'='*70}")

    pos_syms = go_to_genes[go_id]
    y_tr     = np.array([1 if s in pos_syms else 0 for s in tr_syms], dtype=np.float32)
    n_pos_tr = int(y_tr.sum())

    sep = sep_cosine(X_brain, np.array([1 if g in pos_syms else 0 for g in brain_gene_arr],
                                        dtype=np.float32))
    go_type = 'A' if (not np.isnan(sep) and sep >= SEP_THRESHOLD) else 'B'
    print(f"  n_pos_train={n_pos_tr}, sep={sep:.4f}, Type-{go_type}")

    # Multi-isoform genes in this GO term
    go_multi_genes = {g: idxs for g, idxs in multi_isoform_genes.items()
                      if g in pos_syms}
    print(f"  GO multi-isoform genes: {len(go_multi_genes)}")

    # Train v10-B (5 seeds) → predict on all 60,165 brain transcripts
    seed_probs = []
    for seed in SEEDS:
        t0 = time.time()
        probs = train_and_predict_brain(X_tr, y_tr, X_brain, seed)
        seed_probs.append(probs)
        print(f"  seed={seed}: done ({time.time()-t0:.0f}s)")

    v10b_brain = np.mean(seed_probs, axis=0)   # (60165,) ensemble

    # LR baseline on brain
    sc_lr = StandardScaler()
    clf_lr = LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000, random_state=42)
    clf_lr.fit(sc_lr.fit_transform(X_tr), y_tr)
    lr_brain = clf_lr.predict_proba(sc_lr.transform(X_brain))[:, 1]

    # Compute FIU metrics for all multi-isoform genes
    df_fiu_v10b = compute_fiu(
        multi_isoform_genes, group_counts,
        v10b_brain, pos_syms, min_counts=MIN_TOTAL_COUNTS
    )
    df_fiu_lr = compute_fiu(
        multi_isoform_genes, group_counts,
        lr_brain, pos_syms, min_counts=MIN_TOTAL_COUNTS
    )

    if len(df_fiu_v10b) == 0:
        print("  WARNING: No multi-isoform genes passed count filter")
        continue

    n_go_genes = df_fiu_v10b[df_fiu_v10b['is_go_pos']]['gene'].nunique()
    print(f"  Genes with FIU data: total={df_fiu_v10b['gene'].nunique()}, "
          f"GO-positive={n_go_genes}")

    # Statistical tests (v10-B)
    stats_fisi = test_ad_vs_control(df_fiu_v10b, 'FISI')
    stats_wfu  = test_ad_vs_control(df_fiu_v10b, 'WFU')
    stats_fds  = test_ad_vs_control(df_fiu_v10b, 'FDS')

    print(f"\n  FISI (AD vs Control, GO-positive genes):")
    print(f"  {'CellType':<22} {'n':>5} {'ΔmeanFISI':>10} {'p-value':>10} {'dir':>5}")
    print(f"  {'-'*55}")
    for _, row in stats_fisi.iterrows():
        sig = '***' if row.pval < 0.001 else ('**' if row.pval < 0.01 else
               ('*' if row.pval < 0.05 else 'n.s.'))
        print(f"  {row.cell_type:<22} {row.n_genes:>5} {row.mean_delta:>10.4f} "
              f"{row.pval:>10.4f} {row.direction:>5} {sig}")

    go_result = {
        'go_name': go_name, 'ad_stage': ad_stage,
        'sep_cosine': round(float(sep), 4) if not np.isnan(sep) else None,
        'go_type': go_type,
        'n_pos_train': n_pos_tr,
        'n_go_multi_genes': len(go_multi_genes),
        'fisi_stats': stats_fisi.to_dict('records'),
        'wfu_stats':  stats_wfu.to_dict('records'),
        'fds_stats':  stats_fds.to_dict('records'),
    }
    all_results[go_id] = go_result


# ─── Figure ───────────────────────────────────────────────────────────────────
print("\n[5] Generating figure ...")

n_go = len(all_results)
n_cols = 4
n_rows = (n_go + n_cols - 1) // n_cols + 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 4))
axes = axes.ravel()

CT_ORDER = ['Excitatory neuron', 'Inhibitory neuron', 'Astrocyte',
            'Oligodendrocyte', 'OPC', 'Microglia', 'Vascular cell']
CT_COLORS = {ct: c for ct, c in zip(CT_ORDER,
    ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#8338EC', '#3A86FF', '#AAAAAA'])}

for gi, (go_id, res) in enumerate(all_results.items()):
    ax = axes[gi]
    df_stats = pd.DataFrame(res['fisi_stats'])
    if len(df_stats) == 0:
        ax.set_title(f"{res['go_name']}\n(no data)"); ax.axis('off'); continue

    df_stats_go = df_stats[df_stats['cell_type'].isin(CT_ORDER)]
    bars = ax.barh(df_stats_go['cell_type'], df_stats_go['mean_delta'],
                   color=[CT_COLORS.get(ct, '#888') for ct in df_stats_go['cell_type']],
                   alpha=0.8, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

    for _, row in df_stats_go.iterrows():
        sig = '***' if row['pval'] < 0.001 else ('**' if row['pval'] < 0.01 else
               ('*' if row['pval'] < 0.05 else ''))
        if sig:
            x = row['mean_delta'] + (0.003 if row['mean_delta'] >= 0 else -0.003)
            ax.text(x, row['cell_type'], sig, va='center', fontsize=8)

    go_type = res.get('go_type', '?')
    ax.set_title(f"{res['go_name']}\n(Type-{go_type}, {res['ad_stage']})", fontsize=8.5)
    ax.set_xlabel('ΔFISI (AD − Control)', fontsize=8)
    ax.grid(axis='x', alpha=0.3)

for gi in range(len(all_results), len(axes)):
    axes[gi].axis('off')

plt.suptitle("FISI: AD vs Control per Cell Type per GO Term\n"
             "(v10-B trained on muscle → brain 60,165 isoforms)",
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()

fig_path = f'{OUT_DIR}/brain_fiu_{ts}.pdf'
plt.savefig(fig_path, bbox_inches='tight', dpi=150)
plt.savefig(fig_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
plt.close()
print(f"  Figure: {fig_path}")

# ─── Save JSON ────────────────────────────────────────────────────────────────
out_json = f'{OUT_DIR}/brain_fiu_{ts}.json'
with open(out_json, 'w') as f:
    json.dump({'timestamp': ts, 'go_terms': all_results}, f, indent=2)
print(f"FINAL: {out_json}")
