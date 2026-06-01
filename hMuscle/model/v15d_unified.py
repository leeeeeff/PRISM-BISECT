# -*- coding: utf-8 -*-
"""
v15d_unified.py
================
v15d 기반. annotation을 human_annotations_unified_bp.txt (SwissProt BP + NCBI gene2go BP union)로 교체.
목적: 다조직 비교 공정성 확보 — 근육 모델 unified annotation 기준 AUPRC 측정
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

DATA_DIR  = '../data'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v15_unified'

N_SEEDS     = 5
REVERSAL_GAP = 0.20
TARGET_GENES = ['TPM1', 'TPM2', 'KIF1B', 'SEH1L', 'GABARAPL1', 'DMD', 'TPM3', 'OBSCN']

# ── 18 GO terms (13 muscle + 5 neuro/cytoskeletal) ───────────────────────────
GO_TERMS = {
    # --- 기존 13개 (근육/대사) ---
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
    # --- 신규 5개 (신경/세포골격) ---
    'GO:0007268': 'Synaptic transmission',   # KIF1Bβ 기능
    'GO:0007018': 'MT-based movement',        # kinesin 계열
    'GO:0043005': 'Neuron projection',        # 축삭 수송 맥락
    'GO:0030182': 'Neuron diff',              # 신경분화
    'GO:0000226': 'MT cytoskeleton org',      # 세포골격
}
GO_KEYS = list(GO_TERMS.keys())
GO_NAMES = list(GO_TERMS.values())
N_GO = len(GO_KEYS)

# ── Load features ─────────────────────────────────────────────────────────────
print("="*70)
print("  v15d_unified Cross-GO Functional Switch — Unified Annotation (18 terms)")
print("="*70)

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
te_ensg_base = [g.split('.')[0] for g in te_geneid]
te_enst_base = [x.split('.')[0] for x in te_isoid]

N_TE = len(te_isoid)
print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")
print(f"  GO terms: {N_GO} (13 muscle + 5 neuro/cytoskeletal)")

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

def run_ensemble(y_tr, y_te):
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
        seed_preds.append(m.predict(X_te, batch_size=1024, verbose=0).flatten())
    preds = np.mean(seed_preds, axis=0)
    auprc = average_precision_score(y_te, preds) if y_te.sum() > 0 else 0.0
    return preds, auprc

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

# ── Build full score matrix: (N_TE × N_GO) ───────────────────────────────────
print(f"\n  Building full score matrix ({N_TE} isoforms × {N_GO} GO terms) ...")
score_matrix = np.zeros((N_TE, N_GO), dtype=np.float32)
auprc_row    = []

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
print(f"  Macro AUPRC (new  5): {np.mean(auprc_row[13:]):.4f}")
print(f"  Score matrix shape: {score_matrix.shape}")

# ── Gene-level index ──────────────────────────────────────────────────────────
gene_isos = defaultdict(list)
for i, sym in enumerate(te_sym):
    gene_isos[sym].append((i, te_enst_base[i], te_ensg_base[i]))

# ── Cross-GO reversal detection ───────────────────────────────────────────────
print("\n  Detecting cross-GO reversals ...")

reversals = []

for sym, iso_list in gene_isos.items():
    if len(iso_list) < 2: continue

    idxs = [x[0] for x in iso_list]
    ensts = [x[1] for x in iso_list]
    ensg  = iso_list[0][2]
    gene_scores = score_matrix[idxs, :]  # (n_iso, N_GO)

    for a in range(len(iso_list)):
        for b in range(a+1, len(iso_list)):
            sa   = gene_scores[a]
            sb   = gene_scores[b]
            diff = sa - sb

            a_wins = [(gi, diff[gi])  for gi in range(N_GO) if diff[gi]  >=  REVERSAL_GAP]
            b_wins = [(gi, -diff[gi]) for gi in range(N_GO) if diff[gi]  <= -REVERSAL_GAP]

            if not a_wins or not b_wins: continue

            best_a = max(a_wins, key=lambda x: x[1])
            best_b = max(b_wins, key=lambda x: x[1])

            reversals.append({
                'gene': sym, 'ensg': ensg,
                'iso_a': ensts[a], 'iso_b': ensts[b],
                'score_a': {GO_KEYS[gi]: round(float(sa[gi]), 4) for gi in range(N_GO)},
                'score_b': {GO_KEYS[gi]: round(float(sb[gi]), 4) for gi in range(N_GO)},
                'go_A':   GO_KEYS[best_a[0]], 'name_A': GO_NAMES[best_a[0]],
                'gap_A':  round(float(best_a[1]), 4),
                'go_B':   GO_KEYS[best_b[0]], 'name_B': GO_NAMES[best_b[0]],
                'gap_B':  round(float(best_b[1]), 4),
                'reversal_strength': round(float(best_a[1] + best_b[1]), 4),
                'a_wins_terms': [(GO_KEYS[gi], GO_NAMES[gi], round(float(d), 3)) for gi, d in a_wins],
                'b_wins_terms': [(GO_KEYS[gi], GO_NAMES[gi], round(float(d), 3)) for gi, d in b_wins],
                'n_a_wins': len(a_wins), 'n_b_wins': len(b_wins),
            })

reversals.sort(key=lambda x: -x['reversal_strength'])

seen_genes = set()
top_reversals = []
for r in reversals:
    if r['gene'] not in seen_genes:
        seen_genes.add(r['gene'])
        top_reversals.append(r)

print(f"\n  Total isoform pairs with cross-GO reversal: {len(reversals)}")
print(f"  Unique genes with reversal: {len(top_reversals)}")

# ── Target gene deep-dive ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("  TARGET GENE CROSS-GO PROFILES")
print("="*70)

for gene in TARGET_GENES:
    isos = gene_isos.get(gene, [])
    if not isos: continue

    idxs  = [x[0] for x in isos]
    ensts = [x[1] for x in isos]
    gs    = score_matrix[idxs, :]  # (n_iso, N_GO)

    # Column headers (abbreviated)
    header = "  ".join(f"{n[:6]:6s}" for n in GO_NAMES)
    sep    = "-" * (20 + 9 * N_GO)

    print(f"\n  {gene}  (n_isos={len(isos)})")
    print(f"  {'ENST':18s}  {header}")
    print(f"  {sep}")

    order = sorted(range(len(isos)), key=lambda i: -gs[i].max())
    for oi in order[:6]:
        row = "  ".join(f"{gs[oi, gi]:6.3f}" for gi in range(N_GO))
        print(f"  {ensts[oi]:18s}  {row}")

    gene_revs = [r for r in reversals if r['gene'] == gene]
    if gene_revs:
        best = gene_revs[0]
        print(f"\n  Best reversal: {best['iso_a']} vs {best['iso_b']}")
        print(f"    A wins ({best['name_A']}): gap={best['gap_A']:.3f}")
        print(f"    B wins ({best['name_B']}): gap={best['gap_B']:.3f}")
        print(f"    A wins {best['n_a_wins']} terms, B wins {best['n_b_wins']} terms")
        print(f"    A_wins: {[(n, d) for _, n, d in best['a_wins_terms']]}")
        print(f"    B_wins: {[(n, d) for _, n, d in best['b_wins_terms']]}")
    else:
        print(f"\n  No cross-GO reversal found (gap < {REVERSAL_GAP})")

# ── Global top reversals ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("  TOP-20 CROSS-GO FUNCTIONAL SWITCHES (by reversal strength)")
print("="*70)
print(f"  {'Gene':12s} {'A wins':22s} {'gapA':5s}  {'B wins':22s} {'gapB':5s}  {'total':5s}")
print(f"  {'-'*85}")
for r in top_reversals[:20]:
    print(f"  {r['gene']:12s} {r['name_A'][:22]:22s} {r['gap_A']:5.3f}  "
          f"{r['name_B'][:22]:22s} {r['gap_B']:5.3f}  {r['reversal_strength']:5.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n  Reversal gap threshold: {REVERSAL_GAP}")
print(f"  Genes with ≥1 reversal pair: {len(top_reversals)}")
n_target_found = sum(1 for g in TARGET_GENES if any(r['gene'] == g for r in reversals))
print(f"  Target genes with reversal: {n_target_found}/{len(TARGET_GENES)}")

# KIF1B 상세 — 새 GO term 반응 확인
kif_isos = gene_isos.get('KIF1B', [])
if kif_isos:
    print("\n  KIF1B score profile (old → new GO terms):")
    idxs  = [x[0] for x in kif_isos]
    ensts = [x[1] for x in kif_isos]
    gs    = score_matrix[idxs, :]
    # Show all isoforms for new GO terms (col 13-17)
    new_names = GO_NAMES[13:]
    print(f"  {'ENST':18s}  " + "  ".join(f"{n[:14]:14s}" for n in new_names))
    for i, oi in enumerate(sorted(range(len(kif_isos)), key=lambda k: -gs[k, 13:].max())):
        row = "  ".join(f"{gs[oi, 13+j]:14.4f}" for j in range(5))
        print(f"  {ensts[oi]:18s}  {row}")

# ── Save ──────────────────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M')

np.save(f'{OUT_DIR}/score_matrix_18go_{ts}.npy', score_matrix)

result = {
    'timestamp': ts,
    'n_go_terms': N_GO,
    'go_terms': GO_TERMS,
    'auprc_per_go': {GO_KEYS[i]: round(auprc_row[i], 4) for i in range(N_GO)},
    'macro_auprc_all18': round(float(np.mean(auprc_row)), 4),
    'macro_auprc_orig13': round(float(np.mean(auprc_row[:13])), 4),
    'macro_auprc_new5': round(float(np.mean(auprc_row[13:])), 4),
    'n_reversal_pairs': len(reversals),
    'n_genes_with_reversal': len(top_reversals),
    'reversal_gap_threshold': REVERSAL_GAP,
    'top20_reversals': top_reversals[:20],
    'target_gene_profiles': {},
}

for gene in TARGET_GENES:
    isos = gene_isos.get(gene, [])
    if not isos: continue
    idxs  = [x[0] for x in isos]
    ensts = [x[1] for x in isos]
    gs    = score_matrix[idxs, :]
    result['target_gene_profiles'][gene] = {
        'isoforms': [
            {'enst': ensts[i],
             'scores': {GO_KEYS[gi]: round(float(gs[i, gi]), 4) for gi in range(N_GO)},
             'max_score': round(float(gs[i].max()), 4),
             'max_go': GO_KEYS[int(gs[i].argmax())],
             'max_go_name': GO_NAMES[int(gs[i].argmax())],
            }
            for i in sorted(range(len(isos)), key=lambda k: -gs[k].max())
        ],
        'reversals': [r for r in reversals if r['gene'] == gene],
    }

json.dump(result, open(f'{OUT_DIR}/cross_go_18go_{ts}.json', 'w'), indent=2)
print(f"\n  [Saved] {OUT_DIR}/cross_go_18go_{ts}.json")
print(f"  [Saved] {OUT_DIR}/score_matrix_18go_{ts}.npy")
print("\nALL DONE")
