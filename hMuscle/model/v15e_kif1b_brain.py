# -*- coding: utf-8 -*-
"""
v15e_kif1b_brain.py
====================
KIF1Bα(근육) vs KIF1Bβ(신경) cross-GO functional switch 검증

핵심:
  ENST00000377093 (KIF1Bα): SKM_frac=0.848, Brain_frac=0.061  — 근육 dominant
  ENST00000377086 (KIF1Bβ): SKM_frac=0.072, Brain_frac=0.709  — 뇌 dominant
  → 두 isoform이 18 GO term에서 교차 역전을 보이는가?

설계:
  - X_te_ext = [기존 test set (36748개)] + [brain_only KIF1B isoforms (N개)]
  - 모든 18 GO term 모델로 extended test set 스코어링
  - KIF1Bα vs KIF1Bβ 프로파일 직접 비교
"""

import os, json, re, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("  Using GPU:0")
    except: pass

DATA_DIR   = '../data'
BRAIN_DIR  = '../data/brain_esm2'
ANNOT_DIR  = '../data/raw_data/data/annotations'
ID_DIR     = '../data/raw_data/data/id_lists'
OUT_DIR    = '../../reports/v15_switch_dtu'

N_SEEDS     = 5
REVERSAL_GAP = 0.15   # v15d보다 완화 (brain isoform은 소수이므로)

GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0030017': 'Sarcomere',
    'GO:0006941': 'Muscle contraction',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0003774': 'Motor activity',
    'GO:0006096': 'Glycolysis',
    'GO:0007268': 'Synaptic transmission',
    'GO:0007018': 'MT-based movement',
    'GO:0043005': 'Neuron projection',
    'GO:0030182': 'Neuron diff',
    'GO:0000226': 'MT cytoskeleton org',
}
GO_KEYS  = list(GO_TERMS.keys())
GO_NAMES = list(GO_TERMS.values())
N_GO = len(GO_KEYS)

# ── KIF1B isoforms of interest ─────────────────────────────────────────────────
# 근육형: test set에 이미 있음
KIF1B_ALPHA = 'ENST00000377093'   # SKM=57.48, Brain=6.42
# 신경형: brain_only ESM-2에만 있음
KIF1B_BETA  = 'ENST00000377086'   # SKM=4.90,  Brain=74.32  (KIF1B-204)
# 추가 brain isoforms in test set
KIF1B_BRAIN_TE = ['ENST00000676179', 'ENST00000620295', 'ENST00000635499']

print("="*70)
print("  v15e KIF1Bα(muscle) vs KIF1Bβ(brain) Cross-GO Validation")
print("="*70)

# ── Load features ─────────────────────────────────────────────────────────────
def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

X_tr = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')
te_geneid = load_ids('my_gene_list_fixed.npy')
te_isoid  = load_ids('my_isoform_list_fixed.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

tr_sym       = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_geneid]
te_sym       = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_geneid]
te_enst_base = [x.split('.')[0] for x in te_isoid]

N_TE = len(te_isoid)

# ── Load brain-only ESM-2 and extract KIF1Bβ embedding ────────────────────────
print("  Loading brain-only ESM-2 ...")
X_brain = np.load(f'{BRAIN_DIR}/brain_only_esm2_t30_150M.npy').astype(np.float32)

# Parse brain_only GTF to get ENST ID order
brain_gtf = f'{BRAIN_DIR}/brain_only.gtf'
brain_enst_list = []
with open(brain_gtf) as f:
    for line in f:
        if '\ttranscript\t' in line:
            m = re.search(r'transcript_id "(ENST\d+)"', line)
            if m:
                brain_enst_list.append(m.group(1))

assert len(brain_enst_list) == X_brain.shape[0], \
    f"GTF ({len(brain_enst_list)}) vs ESM2 ({X_brain.shape[0]}) mismatch"

brain_enst2idx = {e: i for i, e in enumerate(brain_enst_list)}

# Collect brain-only isoforms to add to test set
# 1) KIF1Bβ (primary)  2) other brain KIF1B isoforms NOT in test set
brain_add_ids  = []  # ENST base IDs
brain_add_embs = []  # (640,) embeddings
brain_add_labels = []  # descriptive labels

te_enst_set = set(te_enst_base)

# KIF1Bβ
if KIF1B_BETA in brain_enst2idx and KIF1B_BETA not in te_enst_set:
    brain_add_ids.append(KIF1B_BETA)
    brain_add_embs.append(X_brain[brain_enst2idx[KIF1B_BETA]])
    brain_add_labels.append('KIF1Bb_brain')
    print(f"  Added KIF1Bβ ({KIF1B_BETA}) from brain ESM-2 [idx={brain_enst2idx[KIF1B_BETA]}]")

# Other KIF1B brain isoforms absent from test set
for enst in KIF1B_BRAIN_TE:
    if enst in brain_enst2idx and enst not in te_enst_set:
        brain_add_ids.append(enst)
        brain_add_embs.append(X_brain[brain_enst2idx[enst]])
        brain_add_labels.append(f'KIF1B_brain_{enst[-4:]}')
        print(f"  Added {enst} from brain ESM-2 [idx={brain_enst2idx[enst]}]")

if brain_add_embs:
    X_brain_extra = np.array(brain_add_embs, dtype=np.float32)  # (n_extra, 640)
    X_te_ext = np.vstack([X_te, X_brain_extra])
    N_EXTRA  = len(brain_add_ids)
else:
    X_te_ext = X_te
    N_EXTRA  = 0

N_TE_EXT = len(X_te_ext)
print(f"  Extended test set: {N_TE} + {N_EXTRA} brain isoforms = {N_TE_EXT}")
print(f"  Train: {X_tr.shape}  Test (ext): {X_te_ext.shape}")

# ── Model ─────────────────────────────────────────────────────────────────────
def build_v10b():
    inp = layers.Input(shape=(640,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64,  activation='relu')(x)
    out = layers.Dense(1,  activation='sigmoid')(x)
    return models.Model(inputs=inp, outputs=out)

def run_ensemble(y_tr, y_te_gt):
    seed_preds = []
    for s in range(N_SEEDS):
        tf.random.set_seed(s * 137 + 42)
        np.random.seed(s * 137 + 42)
        idx = np.random.permutation(len(X_tr))
        m = build_v10b()
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
        cb = [callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                      restore_best_weights=True, verbose=0),
              callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)]
        m.fit(X_tr[idx], y_tr[idx], epochs=80, batch_size=512,
              validation_split=0.1, callbacks=cb, verbose=0)
        # Score extended test set (includes brain isoforms at the end)
        seed_preds.append(m.predict(X_te_ext, batch_size=1024, verbose=0).flatten())
    preds = np.mean(seed_preds, axis=0)
    # AUPRC only on original test set (not brain-only additions)
    auprc = average_precision_score(y_te_gt, preds[:N_TE]) if y_te_gt.sum() > 0 else 0.0
    return preds, auprc

def load_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te

# ── Build full score matrix: (N_TE_EXT × N_GO) ────────────────────────────────
print(f"\n  Building score matrix ({N_TE_EXT} isoforms × {N_GO} GO terms) ...")
score_matrix = np.zeros((N_TE_EXT, N_GO), dtype=np.float32)
auprc_row = []

for gi, (go_term, go_name) in enumerate(GO_TERMS.items()):
    t0 = time.time()
    y_tr, y_te = load_labels(go_term)
    preds, auprc = run_ensemble(y_tr, y_te)
    score_matrix[:, gi] = preds
    auprc_row.append(auprc)
    n_pos = int(y_te.sum())
    tag = ' [NEW]' if gi >= 13 else ''
    print(f"  [{gi+1:2d}/{N_GO}] {go_name[:20]:20s}  AUPRC={auprc:.4f}  pos={n_pos:4d}  ({time.time()-t0:.0f}s){tag}")

print(f"\n  Macro AUPRC (all 18): {np.mean(auprc_row):.4f}")
print(f"  Macro AUPRC (orig 13): {np.mean(auprc_row[:13]):.4f}")
print(f"  Score matrix shape: {score_matrix.shape}")

# ── KIF1Bα vs KIF1Bβ direct comparison ────────────────────────────────────────
print("\n" + "="*70)
print("  KIF1Bα (muscle) vs KIF1Bβ (brain) — GO term score profiles")
print("="*70)

# Find KIF1Bα index in test set
alpha_idx = None
for i, enst in enumerate(te_enst_base):
    if enst == KIF1B_ALPHA:
        alpha_idx = i
        break

# Find KIF1Bβ index in extended test set
beta_idx = N_TE + brain_add_ids.index(KIF1B_BETA) if KIF1B_BETA in brain_add_ids else None

# All KIF1B isoforms in test set
te_kif1b_idx = [i for i, enst in enumerate(te_enst_base)
                if enst.startswith('ENST') and enst in {
                    'ENST00000377093','ENST00000676179','ENST00000620295',
                    'ENST00000622724','ENST00000696500','ENST00000696501'}]
te_kif1b_uniq = sorted(set(te_enst_base[i] for i in te_kif1b_idx))

# Header
hdr = "  ".join(f"{n[:7]:7s}" for n in GO_NAMES)
print(f"  {'ENST':20s}  {hdr}")
print(f"  {'Source':20s}  {'[muscle GO terms →]':50s} {'[← neuro GO terms]':30s}")
print("  " + "-" * (22 + 10*N_GO))

def print_iso_row(label, idx, hpa_info=""):
    if idx is None:
        print(f"  {label:20s}  NOT FOUND")
        return
    row = "  ".join(f"{score_matrix[idx, gi]:7.3f}" for gi in range(N_GO))
    print(f"  {label:20s}  {row}  {hpa_info}")

# Print KIF1Bα
if alpha_idx is not None:
    print_iso_row(f"{KIF1B_ALPHA[:18]} (α)", alpha_idx, "SKM=57.5 Brain=6.4")

# Print KIF1Bβ
if beta_idx is not None:
    print_iso_row(f"{KIF1B_BETA[:18]} (β)", beta_idx, "SKM=4.9  Brain=74.3")
else:
    print(f"  KIF1Bβ not found in extended set")

# Print other brain isoforms in test set
for enst in ['ENST00000676179', 'ENST00000620295', 'ENST00000622724']:
    idxs = [i for i, e in enumerate(te_enst_base) if e == enst]
    if idxs:
        print_iso_row(f"{enst[:18]}", idxs[0], "brain-biased (test)")

# Print extra brain isoforms
for j, enst in enumerate(brain_add_ids):
    if enst != KIF1B_BETA:
        print_iso_row(f"{enst[:18]} (brain)", N_TE + j)

# ── Cross-GO reversal: α vs β ─────────────────────────────────────────────────
print("\n  Cross-GO reversal analysis (α vs β):")
if alpha_idx is not None and beta_idx is not None:
    sa   = score_matrix[alpha_idx]
    sb   = score_matrix[beta_idx]
    diff = sa - sb  # positive: α > β

    a_wins = [(gi, diff[gi])  for gi in range(N_GO) if diff[gi]  >=  REVERSAL_GAP]
    b_wins = [(gi, -diff[gi]) for gi in range(N_GO) if diff[gi]  <= -REVERSAL_GAP]

    print(f"  α wins (gap≥{REVERSAL_GAP}): {[(GO_NAMES[gi], round(d,3)) for gi,d in sorted(a_wins,key=lambda x:-x[1])]}")
    print(f"  β wins (gap≥{REVERSAL_GAP}): {[(GO_NAMES[gi], round(d,3)) for gi,d in sorted(b_wins,key=lambda x:-x[1])]}")

    if a_wins and b_wins:
        best_a = max(a_wins, key=lambda x: x[1])
        best_b = max(b_wins, key=lambda x: x[1])
        print(f"\n  ✓ CROSS-GO REVERSAL CONFIRMED")
        print(f"    α high in: {GO_NAMES[best_a[0]]} (gap={best_a[1]:.3f})")
        print(f"    β high in: {GO_NAMES[best_b[0]]} (gap={best_b[1]:.3f})")
        print(f"    Total reversal strength: {best_a[1]+best_b[1]:.3f}")
    else:
        missing = []
        if not a_wins: missing.append(f"α never clearly wins (max diff={diff.max():.3f})")
        if not b_wins: missing.append(f"β never clearly wins (max diff={-diff.min():.3f})")
        print(f"  ✗ No reversal: {'; '.join(missing)}")
        print(f"  α-β diff range: [{diff.min():.3f}, {diff.max():.3f}]")
        print(f"  Top 5 diffs: {[(GO_NAMES[gi], round(float(diff[gi]),3)) for gi in np.argsort(-np.abs(diff))[:5]]}")
else:
    print(f"  Cannot compare: α_found={alpha_idx is not None}, β_found={beta_idx is not None}")

# ── Save ──────────────────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M')

# Build full profiles for KIF1B isoforms
kif1b_profiles = {}
if alpha_idx is not None:
    kif1b_profiles[KIF1B_ALPHA] = {
        'label': 'KIF1B_alpha_muscle',
        'hpa_skm': 57.48, 'hpa_brain': 6.42, 'skm_frac': 0.848,
        'scores': {GO_KEYS[gi]: round(float(score_matrix[alpha_idx, gi]), 4) for gi in range(N_GO)},
        'source': 'test_set',
    }
if beta_idx is not None:
    kif1b_profiles[KIF1B_BETA] = {
        'label': 'KIF1B_beta_brain',
        'hpa_skm': 4.90, 'hpa_brain': 74.32, 'skm_frac': 0.072,
        'scores': {GO_KEYS[gi]: round(float(score_matrix[beta_idx, gi]), 4) for gi in range(N_GO)},
        'source': 'brain_only_esm2',
    }

result = {
    'timestamp': ts,
    'n_go_terms': N_GO,
    'auprc_per_go': {GO_KEYS[i]: round(auprc_row[i], 4) for i in range(N_GO)},
    'macro_auprc_all18': round(float(np.mean(auprc_row)), 4),
    'macro_auprc_orig13': round(float(np.mean(auprc_row[:13])), 4),
    'macro_auprc_new5': round(float(np.mean(auprc_row[13:])), 4),
    'kif1b_alpha_enst': KIF1B_ALPHA,
    'kif1b_beta_enst': KIF1B_BETA,
    'kif1b_profiles': kif1b_profiles,
    'reversal_gap_threshold': REVERSAL_GAP,
    'brain_add_ids': brain_add_ids,
}

json.dump(result, open(f'{OUT_DIR}/kif1b_brain_{ts}.json', 'w'), indent=2)
np.save(f'{OUT_DIR}/score_matrix_kif1b_{ts}.npy', score_matrix)
print(f"\n  [Saved] {OUT_DIR}/kif1b_brain_{ts}.json")
print(f"  [Saved] {OUT_DIR}/score_matrix_kif1b_{ts}.npy")
print("\nALL DONE")
