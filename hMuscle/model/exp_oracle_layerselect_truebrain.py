#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_oracle_layerselect_truebrain.py
-----------------------------------
DECISIVE 3-arm ORACLE test: is per-GO layer selection a *lever* on brain?

Motivation
==========
exp_delta_dr_decisive showed the fixed muscle-tuned delta pair (L30,L15) earns
nothing on brain (DR-AUC 0.745 vs single-L15 0.762). The layer-shift Fisher
analysis (muscle MF peak median L15 -> brain L19, 42/81 terms shift >=5 layers)
suggests the fixed pair is *mislocated* for brain. The user's hypothesis: a model
that picks, per GO term, the layer where information actually concentrates (v20b's
premise) should be more tissue-flexible.

The catch: the peak is chosen with LABELS (argmax Fisher). At true brain zero-shot
we only have MUSCLE peaks; using BRAIN peaks is leakage (an oracle, not deployable).
So we bracket the direction:

  fixed_L15    : every GO uses L15 (reference; decisive got DR-AUC 0.762)
  fixed_L30    : every GO uses L30 (reference; decisive 0.754)
  muscle_peak  : per-GO layer = argmax(MUSCLE Fisher)  -> DEPLOYABLE zero-shot
  brain_peak   : per-GO layer = argmax(BRAIN  Fisher)  -> ORACLE ceiling (leakage)

Decisive quantities:
  oracle_gain     = brain_peak.DR - fixed_L15.DR   (is selection a lever AT ALL?)
  deployable_gain = muscle_peak.DR - fixed_L15.DR  (does the deployable form help?)
  recoverable_gap = brain_peak.DR - muscle_peak.DR (upper bound any label-free
                                                    selector could ever recover)

PREDICT-BEFORE-LOOK (HARKing guard):
  - fixed_L15 ~ 0.76, fixed_L30 ~ 0.75 (reproduce decisive within harness noise).
  - muscle_peak ~ 0.75-0.76 (median muscle peak IS ~L15; per-GO scatter may even
    hurt slightly if some muscle peaks are noisy).
  - brain_peak (ORACLE): the crux. If layer-selection is a real lever -> notably
    > 0.78 (info present, just mislocated). If it is NOT -> ~ 0.76 (perfect peaks
    give nothing => delta_layer boundary is not breakable by layer choice).
  Decision rule: oracle_gain > 0.02 -> direction ALIVE (pursue input-adaptive
  label-free selector). oracle_gain <= 0.02 -> direction DEAD (kill v20b-style
  per-GO selection for brain; the within-gene ceiling is intrinsic).

Single-layer (640-dim) is deliberately used rather than v20b's PCA-8 window,
because (a) it is directly comparable to the decisive fixed-L15/L30 references,
and (b) the v20b domain-evidence null already showed the window adds noisy layers
that HURT domain alignment on muscle. If even the best single layer per GO is not
a lever, the window (a superset with noise) cannot be either. Window is a
conditional follow-up ONLY if brain_peak single-layer wins.

Reuses label construction, DR-AUC, gene2idxs, iso_n_domains VERBATIM from
exp_delta_dr_decisive_truebrain.py.
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "8"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
try: os.nice(10)
except Exception: pass

import json, gzip, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
BRAIN_DIR = '../data/brain_isoquant_esm2/full'
DOMAIN_MAT= '../results_isoform/features/domain_matrix_brain_full.npy'
FISHER_MUS= '../../reports/exp_C1_layer_probe_279/layer_probe_279_fisher.json'
FISHER_BRN= '../../reports/exp_C1_layer_probe_279/layer_probe_279_fisher_brain.json'
OUT_DIR   = '../../reports/truebrain_rerun_20260714/exp_oracle_layerselect'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 7, 13, 21, 99]
EPOCHS_MLP = 60
BATCH_MLP  = 512
N_LAYERS   = 30

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 72)
print("  ORACLE 3-arm: per-GO layer selection lever test (TRUE BRAIN)")
print("=" * 72)

# ── 1. IDs + GO labels (verbatim from decisive) ───────────────────────
print("\n[1] IDs + GO labels...")
tr_genes = [clean(g) for g in np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)]
te_sym_list = [clean(g) for g in np.load(f'{BRAIN_DIR}/brain_full_gene_names.npy', allow_pickle=True)]
te_iso_list = [clean(x) for x in np.load(f'{BRAIN_DIR}/brain_full_ids.npy', allow_pickle=True)]
n_iso = len(te_iso_list)

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

tr_ids    = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_ids)
go_genes_tr, go_genes_all = defaultdict(set), defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606' or p[7] != 'Function': continue
        go_genes_all[p[2]].add(p[1])
        if p[1] in tr_id_set: go_genes_tr[p[2]].add(p[1])

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])

tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

def build_Y_tr(go_id):
    pos_ids  = go_genes_tr[go_id]
    pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in pos_ids}
    y = np.zeros(len(tr_genes), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
    return y

def build_Y_te(go_id):
    pos_ids = go_genes_all[go_id]
    return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0 for s in te_sym_list], dtype=np.float32)

Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)
Y_te = np.stack([build_Y_te(go) for go in mf_terms], axis=1)
valid_mask = Y_te.sum(0) >= 2
valid_idx  = [i for i in range(len(mf_terms)) if valid_mask[i]]
n_go = len(mf_terms)

L2_TERMS = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural': L2_TERMS.add(p[0])
l2_valid = [i for i in range(len(mf_terms)) if mf_terms[i] in L2_TERMS and valid_mask[i]]
print(f"  {n_go} MF | valid {int(valid_mask.sum())} | L2 {len(l2_valid)}")

# ── 2. Domain-ranking prerequisites (verbatim) ────────────────────────
gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs[g].append(i)
iso_n_domains = np.load(DOMAIN_MAT).sum(axis=1).astype(np.int32)
Y_te_v = Y_te[:, valid_mask]

def compute_domain_ranking_auc(preds_v, gene2idxs, iso_n_domains, Y_te_v):
    aucs = []
    for g, idxs in gene2idxs.items():
        if len(idxs) < 2: continue
        idxs = np.array(idxs)
        domains = iso_n_domains[idxs]
        if domains.std() < 0.1: continue
        gene_pos_terms = np.where(Y_te_v[idxs[0]] > 0)[0]
        if len(gene_pos_terms) == 0: continue
        med = np.median(domains)
        domain_binary = (domains > med).astype(float)
        if domain_binary.sum() == 0 or domain_binary.sum() == len(idxs): continue
        p_g = preds_v[idxs]
        for t in gene_pos_terms:
            scores = p_g[:, t]
            if scores.std() < 1e-8:
                aucs.append(0.5); continue
            try: aucs.append(roc_auc_score(domain_binary, scores))
            except: pass
    return (float(np.mean(aucs)) if aucs else 0.5), len(aucs)

# ── 3. Fisher peaks (0-indexed) per GO ────────────────────────────────
print("\n[2] Fisher peaks (muscle + brain)...")
fmus = json.load(open(FISHER_MUS)); fmus = fmus.get('per_go', fmus)
fbrn = json.load(open(FISHER_BRN)); fbrn = fbrn.get('per_go', fbrn)

def peak_map(fdict, default0=14):
    m = {}
    for gi, go in enumerate(mf_terms):
        if go in fdict and 'fisher_per_layer' in fdict[go]:
            m[gi] = int(np.argmax(fdict[go]['fisher_per_layer']))
        else:
            m[gi] = default0  # default L15 (0-idx 14)
    return m

ARMS = {
    'fixed_L15':   {gi: 14 for gi in range(n_go)},
    'fixed_L30':   {gi: 29 for gi in range(n_go)},
    'muscle_peak': peak_map(fmus),
    'brain_peak':  peak_map(fbrn),
}
for name, amap in ARMS.items():
    layers_used = sorted(set(amap[gi] for gi in valid_idx))
    print(f"  {name:<12} unique layers over valid GO: {[l+1 for l in layers_used]}")

# ── 4. Per-layer embedding cache (MaxAbs scaled, train-fit) ───────────
_cache = {}
def get_layer(L0):
    """L0: 0-indexed layer. Returns (Xtr_s, Xte_s) MaxAbs-scaled float32."""
    if L0 in _cache: return _cache[L0]
    L1 = L0 + 1
    Xtr = np.load(f'{DATA_DIR}/esm2_train_human_layer{L1:02d}_t30_150M.npy').astype(np.float32)
    Xte = np.load(f'{BRAIN_DIR}/brain_full_esm2_layer{L1:02d}_t30_150M.npy').astype(np.float32)
    sc = MaxAbsScaler()
    Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)
    _cache[L0] = (Xtr_s, Xte_s)
    return _cache[L0]

# ── 5. TF MLP (multi-label per layer-group) ───────────────────────────
import tensorflow as tf
from tensorflow.keras import layers as KL, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

def build_mlp(dim, nout):
    inp = KL.Input(shape=(dim,))
    x = KL.Dense(256, activation='relu')(inp)
    x = KL.BatchNormalization()(x); x = KL.Dropout(0.2)(x)
    x = KL.Dense(128, activation='relu')(x)
    out = KL.Dense(nout, activation='sigmoid')(x)
    return models.Model(inp, out)

def macro(preds, idxs):
    aps = [average_precision_score(Y_te[:, i], preds[:, i]) for i in idxs if Y_te[:, i].sum() >= 2]
    return float(np.mean(aps)) if aps else float('nan')

def run_arm(name, amap):
    """Train per layer-group multi-label MLPs; return (n_iso, n_go) preds."""
    t0 = time.time()
    preds = np.zeros((n_iso, n_go), dtype=np.float32)
    layer_to_gos = defaultdict(list)
    for gi, L0 in amap.items(): layer_to_gos[L0].append(gi)
    for L0, gos in sorted(layer_to_gos.items()):
        Xtr_s, Xte_s = get_layer(L0)
        gos = np.array(gos)
        Ytr_sub = Y_tr[:, gos]
        seed_preds = []
        for seed in SEEDS:
            np.random.seed(seed); tf.random.set_seed(seed)
            perm = np.random.permutation(len(Xtr_s))
            nv = int(len(Xtr_s) * 0.1); vi, ti = perm[:nv], perm[nv:]
            m = build_mlp(Xtr_s.shape[1], len(gos))
            m.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
            m.fit(Xtr_s[ti], Ytr_sub[ti], validation_data=(Xtr_s[vi], Ytr_sub[vi]),
                  epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                  verbose=0)
            seed_preds.append(m.predict(Xte_s, batch_size=1024, verbose=0))
            tf.keras.backend.clear_session()
        preds[:, gos] = np.mean(seed_preds, axis=0)
    auprc_all = macro(preds, valid_idx)
    auprc_l2  = macro(preds, l2_valid)
    dr_auc, n_pairs = compute_domain_ranking_auc(preds[:, valid_mask], gene2idxs, iso_n_domains, Y_te_v)
    el = round(time.time()-t0)
    print(f"  {name:<12} macro_All={auprc_all:.4f}  macro_L2={auprc_l2:.4f}  "
          f"DR-AUC={dr_auc:.4f} (n={n_pairs})  [{el}s]", flush=True)
    return {'macro_all_mf': auprc_all, 'macro_l2': auprc_l2,
            'domain_ranking_auc': dr_auc, 'dr_n_pairs': n_pairs,
            'n_layer_groups': len(layer_to_gos), 'elapsed_s': el}

# ── 6. Run all arms ───────────────────────────────────────────────────
print("\n[3] Training arms (per-GO layer selection, 5-seed multi-label ensembles)...")
results = {}
for name, amap in ARMS.items():
    results[name] = run_arm(name, amap)

# ── 7. Decisive summary ───────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"  {'Arm':<12} {'macro All':>10} {'macro L2':>9} {'DR-AUC':>8}")
print("  " + "-" * 42)
for name, r in results.items():
    print(f"  {name:<12} {r['macro_all_mf']:>10.4f} {r['macro_l2']:>9.4f} {r['domain_ranking_auc']:>8.4f}")

dr = {k: results[k]['domain_ranking_auc'] for k in results}
oracle_gain     = dr['brain_peak']  - dr['fixed_L15']
deployable_gain = dr['muscle_peak'] - dr['fixed_L15']
recoverable_gap = dr['brain_peak']  - dr['muscle_peak']
alive = oracle_gain > 0.02
verdict = ("DIRECTION ALIVE: brain-oracle per-GO layer selection is a real lever "
           "(info present but mislocated) -> pursue label-free / input-adaptive selector"
           if alive else
           "DIRECTION DEAD: even perfect per-GO brain peaks do not beat fixed L15 "
           "-> within-gene ceiling is intrinsic, not a layer-choice problem")
print(f"\n  oracle_gain     (brain_peak - fixed_L15)  = {oracle_gain:+.4f}")
print(f"  deployable_gain (muscle_peak - fixed_L15) = {deployable_gain:+.4f}")
print(f"  recoverable_gap (brain_peak - muscle_peak)= {recoverable_gap:+.4f}")
print(f"  VERDICT: {verdict}")

results['_meta'] = {
    'oracle_gain': oracle_gain, 'deployable_gain': deployable_gain,
    'recoverable_gap': recoverable_gap, 'alive': bool(alive), 'verdict': verdict,
    'decisive_ref_fixed_L15_multilabel': 0.7625, 'decisive_ref_cat_L30_delta': 0.7449,
    'note': 'single 640-dim layer per GO; fixed_L15/L30 re-run in THIS per-group harness '
            'for fair comparison. oracle arms use BRAIN Fisher = leakage (ceiling only).'
}
json.dump(results, open(f'{OUT_DIR}/results.json', 'w'), indent=2)
print(f"\n[Saved] {OUT_DIR}/results.json")
