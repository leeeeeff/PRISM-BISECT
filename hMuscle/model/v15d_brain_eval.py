# -*- coding: utf-8 -*-
"""
v15d_brain_eval.py
==================
v15d_bp_clean 아키텍처로 뇌 조직 IsoQuant 아이소폼 평가.

Train: 근육 데이터 (SwissProt + human annotations) — v15d_bp_clean과 동일
Test:  뇌 조직 IsoQuant full isoforms (63,994개)
       data/brain_isoquant_esm2/full/

사용 시점: brain full ESM-2 완료 후
  conda run -n isoform_env python v15d_brain_eval.py
"""

import os, json, time
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

DATA_DIR    = '../data'
BRAIN_DIR   = '../data/brain_isoquant_esm2/full'
ANNOT_DIR   = '../data/raw_data/data/annotations'
ID_DIR      = '../data/raw_data/data/id_lists'
OUT_DIR     = '../../reports/v15d_brain_eval'
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS      = 5
REVERSAL_GAP = 0.20

# AD 관련 target genes (F46에서 발굴된 sarcopenia/AD 관련 유전자 포함)
TARGET_GENES = [
    'GABARAPL1', 'PINK1', 'BNIP3',           # F46: novel isoform discovery genes
    'APP', 'MAPT', 'APOE',                    # AD hallmark genes
    'BECN1', 'ATG5', 'ATG7',                  # Autophagy
    'PINK1', 'PRKN',                          # Mitophagy
]

GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0045214': 'Sarcomere organization',
    'GO:0006941': 'Muscle contraction',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0030048': 'Actin-based movement',
    'GO:0006096': 'Glycolysis',
    'GO:0007268': 'Synaptic transmission',
    'GO:0007018': 'MT-based movement',
    'GO:0031175': 'Neuron proj development',
    'GO:0030182': 'Neuron diff',
    'GO:0000226': 'MT cytoskeleton org',
}
GO_KEYS  = list(GO_TERMS.keys())
GO_NAMES = list(GO_TERMS.values())
N_GO     = len(GO_KEYS)

print("=" * 70)
print("  v15d_brain_eval — Brain IsoQuant × 18 BP GO terms")
print("=" * 70)

# ── Load IDs ──────────────────────────────────────────────────────────────────
def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

# Train: muscle
X_tr     = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
tr_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_geneid]

# Test: brain IsoQuant full
X_te        = np.load(f'{BRAIN_DIR}/brain_full_esm2_t30_150M.npy').astype(np.float32)
te_isoid    = load_ids(f'{BRAIN_DIR}/brain_full_ids.npy')
te_sym      = load_ids(f'{BRAIN_DIR}/brain_full_gene_names.npy')  # 이미 gene symbol
te_mask     = np.load(f'{BRAIN_DIR}/brain_full_mask.npy')         # 1=coding, 0=non-coding

# novel isoform 식별: transcript*.nnic 또는 .nic
is_novel = np.array(['transcript' in x for x in te_isoid], dtype=bool)

N_TE     = len(te_isoid)
N_NOVEL  = int(is_novel.sum())
N_KNOWN  = N_TE - N_NOVEL
N_CODING = int(te_mask.sum())

print(f"  Train: {X_tr.shape}")
print(f"  Test (brain): {X_te.shape}")
print(f"    novel={N_NOVEL}, known={N_KNOWN}, coding={N_CODING}")
print(f"  GO terms: {N_GO}")

# ── Model (v10b = v15d_bp_clean 아키텍처) ─────────────────────────────────────
def build_v10b():
    inp = layers.Input(shape=(640,))
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64,  activation='relu')(x)
    out = layers.Dense(1,   activation='sigmoid')(x)
    return models.Model(inputs=inp, outputs=out)

def run_ensemble(y_tr, y_te):
    seed_preds = []
    for s in range(N_SEEDS):
        tf.random.set_seed(s * 137 + 42)
        np.random.seed(s * 137 + 42)
        idx = np.random.permutation(len(X_tr))
        m   = build_v10b()
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
        cb  = [callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                       restore_best_weights=True, verbose=0),
               callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)]
        m.fit(X_tr[idx], y_tr[idx], epochs=80, batch_size=512,
              validation_split=0.1, callbacks=cb, verbose=0)
        seed_preds.append(m.predict(X_te, batch_size=1024, verbose=0).flatten())
    preds  = np.mean(seed_preds, axis=0)
    auprc  = average_precision_score(y_te, preds) if y_te.sum() > 0 else 0.0
    # novel 전용 AUPRC
    auprc_novel = 0.0
    if is_novel.sum() > 0 and y_te[is_novel].sum() > 0:
        auprc_novel = average_precision_score(y_te[is_novel], preds[is_novel])
    return preds, auprc, auprc_novel

def load_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations_unified_bp.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te

# ── Score matrix ──────────────────────────────────────────────────────────────
print(f"\n  Building score matrix ({N_TE} isoforms × {N_GO} GO terms) ...")
score_matrix  = np.zeros((N_TE, N_GO), dtype=np.float32)
auprc_all     = []
auprc_novel_l = []
n_pos_all     = []

for gi, (go_term, go_name) in enumerate(GO_TERMS.items()):
    t0 = time.time()
    y_tr, y_te = load_labels(go_term)
    preds, auprc, auprc_novel = run_ensemble(y_tr, y_te)
    score_matrix[:, gi] = preds
    auprc_all.append(auprc)
    auprc_novel_l.append(auprc_novel)
    n_pos = int(y_te.sum())
    n_pos_nov = int(y_te[is_novel].sum())
    n_pos_all.append(n_pos)
    print(f"  [{gi+1:2d}/{N_GO}] {go_name[:20]:20s}  AUPRC={auprc:.4f}  novel_AUPRC={auprc_novel:.4f}  "
          f"pos={n_pos:4d}(n={n_pos_nov:3d})  ({time.time()-t0:.0f}s)")

print(f"\n  Macro AUPRC (all   18, full): {np.mean(auprc_all):.4f}")
print(f"  Macro AUPRC (orig  13, full): {np.mean(auprc_all[:13]):.4f}")
print(f"  Macro AUPRC (neuro  5, full): {np.mean(auprc_all[13:]):.4f}")
print(f"  Macro AUPRC (novel isos):     {np.nanmean(auprc_novel_l):.4f}")

# ── Cross-GO reversal detection (novel isoforms만) ────────────────────────────
print("\n  Detecting cross-GO reversals (novel isoforms) ...")
gene_isos_novel = defaultdict(list)
for i, (sym, tid) in enumerate(zip(te_sym, te_isoid)):
    if is_novel[i]:
        gene_isos_novel[sym].append((i, tid))

reversals = []
for sym, iso_list in gene_isos_novel.items():
    if len(iso_list) < 2: continue
    idxs = [x[0] for x in iso_list]
    tids = [x[1] for x in iso_list]
    gs   = score_matrix[idxs, :]

    for a in range(len(iso_list)):
        for b in range(a+1, len(iso_list)):
            sa, sb = gs[a], gs[b]
            diff   = sa - sb
            a_wins = [(gi, diff[gi])  for gi in range(N_GO) if  diff[gi] >= REVERSAL_GAP]
            b_wins = [(gi, -diff[gi]) for gi in range(N_GO) if -diff[gi] >= REVERSAL_GAP]
            if not a_wins or not b_wins: continue
            best_a = max(a_wins, key=lambda x: x[1])
            best_b = max(b_wins, key=lambda x: x[1])
            reversals.append({
                'gene': sym,
                'iso_a': tids[a], 'iso_b': tids[b],
                'score_a': {GO_KEYS[gi]: round(float(sa[gi]), 4) for gi in range(N_GO)},
                'score_b': {GO_KEYS[gi]: round(float(sb[gi]), 4) for gi in range(N_GO)},
                'go_A': GO_KEYS[best_a[0]], 'name_A': GO_NAMES[best_a[0]],
                'gap_A': round(float(best_a[1]), 4),
                'go_B': GO_KEYS[best_b[0]], 'name_B': GO_NAMES[best_b[0]],
                'gap_B': round(float(best_b[1]), 4),
                'reversal_strength': round(float(best_a[1] + best_b[1]), 4),
                'a_wins_terms': [(GO_KEYS[gi], GO_NAMES[gi], round(float(d), 3)) for gi, d in a_wins],
                'b_wins_terms': [(GO_KEYS[gi], GO_NAMES[gi], round(float(d), 3)) for gi, d in b_wins],
            })

reversals.sort(key=lambda x: -x['reversal_strength'])
seen = set()
top_reversals = []
for r in reversals:
    if r['gene'] not in seen:
        seen.add(r['gene'])
        top_reversals.append(r)

print(f"\n  Reversal pairs (novel): {len(reversals)}")
print(f"  Unique genes (novel):   {len(top_reversals)}")

# ── Target gene profiles ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  AD / AUTOPHAGY TARGET GENE PROFILES (novel isoforms)")
print("=" * 70)

gene_isos_all = defaultdict(list)
for i, (sym, tid) in enumerate(zip(te_sym, te_isoid)):
    gene_isos_all[sym].append((i, tid))

for gene in TARGET_GENES:
    isos = gene_isos_all.get(gene, [])
    nov  = [(i, t) for i, t in isos if is_novel[i]]
    if not isos: continue
    idxs = [x[0] for x in nov] if nov else [x[0] for x in isos]
    tids = [x[1] for x in nov] if nov else [x[1] for x in isos]
    gs   = score_matrix[idxs, :]
    label = "novel" if nov else "known"
    print(f"\n  {gene} ({label}, n={len(idxs)})")
    for oi in sorted(range(len(idxs)), key=lambda k: -gs[k].max())[:4]:
        row = " ".join(f"{gs[oi, gi]:.3f}" for gi in range(N_GO))
        print(f"    {tids[oi][:30]:30s}  {row}")

# ── Top reversals ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TOP-20 CROSS-GO FUNCTIONAL SWITCHES (novel isoforms)")
print("=" * 70)
for r in top_reversals[:20]:
    print(f"  {r['gene']:15s} {r['name_A'][:20]:20s} {r['gap_A']:5.3f}  "
          f"{r['name_B'][:20]:20s} {r['gap_B']:5.3f}  str={r['reversal_strength']:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M')

np.save(f'{OUT_DIR}/brain_full_score_matrix_{ts}.npy', score_matrix)

result = {
    'timestamp': ts,
    'n_isoforms_total': N_TE,
    'n_novel': N_NOVEL,
    'n_known': N_KNOWN,
    'n_coding': N_CODING,
    'go_terms': GO_TERMS,
    'auprc_per_go': {GO_KEYS[i]: round(auprc_all[i], 4) for i in range(N_GO)},
    'auprc_novel_per_go': {GO_KEYS[i]: round(auprc_novel_l[i], 4) for i in range(N_GO)},
    'n_pos_per_go': {GO_KEYS[i]: n_pos_all[i] for i in range(N_GO)},
    'macro_auprc_all18': round(float(np.mean(auprc_all)), 4),
    'macro_auprc_orig13': round(float(np.mean(auprc_all[:13])), 4),
    'macro_auprc_neuro5': round(float(np.mean(auprc_all[13:])), 4),
    'macro_auprc_novel': round(float(np.nanmean(auprc_novel_l)), 4),
    'n_reversal_pairs_novel': len(reversals),
    'n_genes_with_reversal_novel': len(top_reversals),
    'reversal_gap_threshold': REVERSAL_GAP,
    'top20_reversals': top_reversals[:20],
    'target_gene_profiles': {},
    'muscle_baseline': {
        'macro_auprc_all18': 0.7022,
        'macro_auprc_orig13': None,
        'model': 'v15d_bp_clean',
        'test_set': 'BambuTx muscle',
    },
}

for gene in TARGET_GENES:
    isos = gene_isos_all.get(gene, [])
    if not isos: continue
    nov  = [(i, t) for i, t in isos if is_novel[i]]
    src  = nov if nov else isos
    idxs = [x[0] for x in src]
    tids = [x[1] for x in src]
    gs   = score_matrix[idxs, :]
    result['target_gene_profiles'][gene] = {
        'n_total': len(isos),
        'n_novel': len(nov),
        'isoforms': [
            {'id': tids[i], 'novel': bool(is_novel[idxs[i]]),
             'scores': {GO_KEYS[gi]: round(float(gs[i, gi]), 4) for gi in range(N_GO)},
             'max_go': GO_KEYS[int(gs[i].argmax())]}
            for i in sorted(range(len(src)), key=lambda k: -gs[k].max())
        ],
        'reversals': [r for r in reversals if r['gene'] == gene],
    }

json.dump(result, open(f'{OUT_DIR}/brain_eval_{ts}.json', 'w'), indent=2)
print(f"\n  [Saved] {OUT_DIR}/brain_eval_{ts}.json")
print(f"  [Saved] {OUT_DIR}/brain_full_score_matrix_{ts}.npy")
print("\nALL DONE")
