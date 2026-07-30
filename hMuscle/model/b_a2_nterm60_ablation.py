#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
b_a2_nterm60_ablation.py

A2 candidate (edit-core additive channel, standalone-isoform translation): concat[L30, Nterm60]
vs production L30 alone. Same 3-seed focal-loss head / 18-GO-BP-term protocol as
b_l9_anchor_ablation.py (A1, DEAD -- decodability-proxy gain did not transfer to the real task),
so this result is directly comparable.

Evaluated on FOUR metrics, not just macro-AUPRC (per user request: macro is gene-family-level,
DR-AUC is domain-structural -- diversify to isoform-level covariates specifically targeted at the
A2 mechanism):
  1. macro-AUPRC (18 GO-BP terms)                       -- gene-classification-dominated, per A1 lesson
  2. domain-ranking AUC (Pfam count, within-gene)        -- existing DR-AUC design, reused as-is
  3. nterm-deviation-ranking AUC (within-gene mode-vs-isoform N-term-60 mismatch) -- NEW, targets
     A2's specific mechanism directly
  4. disorder-nterm-ranking AUC (metapredict, mean over first 60 residues, within-gene median split) -- NEW

PREDICT-BEFORE-LOOK (S2, stated before running): given the SLiM-manifold / editcore / discarded-mode
tracks (this session + prior) have shown encoding-without-usage 7 times in a row for every external
target tried, predict metric (1) shows no significant gain (repeats A1). Metrics (3)/(4) are the
better-powered, mechanism-targeted tests -- if there is ANY real, recoverable signal from adding this
channel, it should show up there first, even if (1) stays flat. A clean negative across all 4 would be
the most decisive closure yet of the "pooling-loss recovery" line.
"""
import os, sys, time, json
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '1')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings; warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
tf.get_logger().setLevel('ERROR')
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

DATA = '../data'; ANNOT = '../data/raw_data/data/annotations'; ID = '../data/raw_data/data/id_lists'
COV_FILE = '../../reports/model_interpretability_map/a1a2_isoform_covariates.npz'
N_SEEDS = 3
GO_KEYS = ['GO:0007204','GO:0030017','GO:0006941','GO:0006914','GO:0043161','GO:0007519','GO:0042692',
 'GO:0055074','GO:0007005','GO:0007517','GO:0032006','GO:0003774','GO:0006096','GO:0007268',
 'GO:0007018','GO:0043005','GO:0030182','GO:0000226']


def load_ids(p):
    a = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


tr_gene = load_ids(f'{ID}/train_gene_list.npy'); te_gene = load_ids('my_gene_list_fixed.npy')
ENSG2SYM = {}
with open(f'{ID}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
tr_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_gene]
te_sym = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in te_gene]

gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym): gene2idxs[g].append(i)


def load_labels(go):
    pos = set()
    with open(f'{ANNOT}/human_annotations_unified_bp.txt') as f:
        for line in f:
            pt = line.strip().split('\t')
            if len(pt) > 1 and go in pt[1:]: pos.add(pt[0])
    return (np.array([1 if s in pos else 0 for s in tr_sym], np.float32),
            np.array([1 if s in pos else 0 for s in te_sym], np.float32))


L30_tr = np.load(f'{DATA}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
L30_te = np.load(f'{DATA}/esm2_layer_30_t30_150M.npy').astype(np.float32)
NT_tr = np.load(f'{DATA}/esm2_train_nterm60_layer30_t30_150M.npy').astype(np.float32)
NT_te = np.load(f'{DATA}/esm2_nterm60_layer30_t30_150M.npy').astype(np.float32)
print(f'train {L30_tr.shape} test {L30_te.shape}  nterm60 train {NT_tr.shape} test {NT_te.shape}', flush=True)

VARIANTS = {
    'L30': (L30_tr, L30_te),
    'L30+Nterm60': (np.concatenate([L30_tr, NT_tr], 1), np.concatenate([L30_te, NT_te], 1)),
}


def head(d):
    inp = layers.Input(shape=(d,)); x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x); x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x); out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out)


def ensemble(Xtr, Xte, ytr):
    ps = []
    for s in range(N_SEEDS):
        tf.random.set_seed(s * 137 + 42); np.random.seed(s * 137 + 42)
        idx = np.random.permutation(len(Xtr)); m = head(Xtr.shape[1])
        m.compile(tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
        cb = [callbacks.EarlyStopping('val_loss', patience=10, restore_best_weights=True, verbose=0),
              callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)]
        m.fit(Xtr[idx], ytr[idx], epochs=80, batch_size=512, validation_split=0.1, callbacks=cb, verbose=0)
        ps.append(m.predict(Xte, batch_size=2048, verbose=0).flatten()); tf.keras.backend.clear_session()
    return np.mean(ps, 0)


labels = [load_labels(go) for go in GO_KEYS]
Yte = np.stack([lt for _, lt in labels], 1)
scores = {}
for v, (Xtr, Xte) in VARIANTS.items():
    t0 = time.time()
    S = np.zeros_like(Yte)
    for gi, go in enumerate(GO_KEYS):
        S[:, gi] = ensemble(Xtr, Xte, labels[gi][0])
    scores[v] = S
    macro = np.mean([average_precision_score(Yte[:, gi], S[:, gi]) for gi in range(18)])
    print(f'  {v:15s} macro-AUPRC={macro:.4f}  ({time.time()-t0:.0f}s)', flush=True)

# ---- covariate-ranking AUCs (generalizes domain_ranking_validation.compute_domain_ranking_auc) ----
cov = np.load(COV_FILE)
COVARIATES = {'domain': cov['n_domains'], 'nterm_deviates': cov['nterm_deviates'],
              'disorder_nterm': cov['disorder_nterm'], 'helix_nterm': cov['helix_nterm'],
              'sheet_nterm': cov['sheet_nterm'], 'hydro_nterm': cov['hydro_nterm']}


def covariate_ranking_auc(preds_mat, gene2idxs, covariate, Yte, binary=False):
    aucs = []
    for g, idxs in gene2idxs.items():
        if len(idxs) < 2: continue
        vals = covariate[idxs]
        if binary:
            if vals.sum() == 0 or vals.sum() == len(idxs): continue
            label = vals
        else:
            if vals.std() < 1e-6: continue
            label = (vals > np.median(vals)).astype(float)
            if label.sum() == 0 or label.sum() == len(idxs): continue
        pos_terms = np.where(Yte[idxs[0]] > 0)[0]
        if len(pos_terms) == 0: continue
        p_g = preds_mat[idxs]
        for t in pos_terms:
            sc = p_g[:, t]
            if sc.std() < 1e-8:
                aucs.append(0.5); continue
            try: aucs.append(roc_auc_score(label, sc))
            except Exception: pass
    return (np.mean(aucs), len(aucs)) if aucs else (0.5, 0)


def bootstrap_covariate_auc(preds_mat, gene2idxs, covariate, Yte, binary=False, n_boot=500, seed=42):
    genes = [g for g, idxs in gene2idxs.items() if len(idxs) >= 2]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sampled = rng.choice(genes, size=len(genes), replace=True)
        auc, _ = covariate_ranking_auc(preds_mat, {g: gene2idxs[g] for g in sampled}, covariate, Yte, binary)
        boots.append(auc)
    boots = np.array(boots)
    return boots


print('\n=== Covariate-ranking AUCs (within-gene, gene-mean-immune), bootstrap CI n=500 ===')
cov_results = {}
for cname, cvals in COVARIATES.items():
    binary = (cname == 'nterm_deviates')
    print(f'\n-- {cname} --')
    for v in VARIANTS:
        auc, n = covariate_ranking_auc(scores[v], gene2idxs, cvals, Yte, binary)
        boots = bootstrap_covariate_auc(scores[v], gene2idxs, cvals, Yte, binary)
        lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        print(f'  {v:15s} AUC={auc:.4f} [{lo:.4f},{hi:.4f}]  (N={n:,} term-pairs)')
        cov_results.setdefault(cname, {})[v] = dict(auc=float(auc), lo=float(lo), hi=float(hi), n=int(n))

# ---- paired bootstrap for macro-AUPRC delta (matches b_l9_anchor_ablation.py style) ----
rng = np.random.default_rng(0); B = 1000; N = len(Yte)
def macro_ap(S, y, idx):
    return np.mean([average_precision_score(y[idx, gi], S[idx, gi]) for gi in range(18) if y[idx, gi].sum() > 0])
boot = {v: [] for v in VARIANTS}
for _ in range(B):
    idx = rng.integers(0, N, N)
    for v in VARIANTS: boot[v].append(macro_ap(scores[v], Yte, idx))
boot = {v: np.array(b) for v, b in boot.items()}
print('\n=== macro-AUPRC (95% bootstrap CI) + Delta vs L30 ===')
base = boot['L30']
for v in VARIANTS:
    b = boot[v]; d = b - base
    print(f'  {v:15s} {b.mean():.4f} [{np.percentile(b,2.5):.4f},{np.percentile(b,97.5):.4f}]'
          + ('' if v == 'L30' else f'   Delta={d.mean():+.4f} [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]'
             + ('  **CI excl 0**' if (np.percentile(d,2.5) > 0 or np.percentile(d,97.5) < 0) else '')))

json.dump({'macro': {v: float(boot[v].mean()) for v in VARIANTS}, 'covariate_auc': cov_results},
          open('../../reports/model_interpretability_map/a2_nterm60_ablation.json', 'w'), indent=2)
print('\n[done]', flush=True)
