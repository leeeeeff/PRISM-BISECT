# -*- coding: utf-8 -*-
"""
v15c_cross_go_switch.py
========================
Cross-GO functional switch validation.

핵심 주장:
  "Functional switching"은 단순히 GO_A에서 isoform X↑ Y↓가 아니다.
  Y가 GO_A의 기능을 안 하는 이유는 Y가 다른 GO_B의 기능을 하기 때문이어야 한다.
  즉, GO_A: X↑ Y↓  AND  GO_B: X↓ Y↑ — 교차(reversal)가 있어야 진짜 switch.

분석:
  1. 모든 13 GO term × 모든 test isoform → 점수 행렬 (n_iso × 13)
  2. 같은 유전자의 isoform 쌍에서: GO term별 score 프로파일 비교
  3. Reversal 검출: term A에서 X>Y + term B에서 Y>X (gap ≥ threshold)
  4. 4개 target gene 심층 분석 + 전체 통계

실행:
  conda run -n isoform_env python v15c_cross_go_switch.py
"""

import os, json, csv, time
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
OUT_DIR   = '../../reports/v15_switch_dtu'

N_SEEDS = 5
REVERSAL_GAP = 0.20   # 역전 인정 최소 score gap
TARGET_GENES = ['TPM1', 'TPM2', 'KIF1B', 'SEH1L', 'GABARAPL1', 'DMD', 'TPM3', 'OBSCN']

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
}
GO_KEYS = list(GO_TERMS.keys())
GO_NAMES = list(GO_TERMS.values())

# ── Load features ────────────────────────────────────────────────────────────
print("="*65)
print("  v15c Cross-GO Functional Switch Validation")
print("="*65)

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

# ── Model ────────────────────────────────────────────────────────────────────
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
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]: pos.add(parts[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te

# ── Build full score matrix: (N_TE × N_GO) ───────────────────────────────────
print("\n  Building full score matrix (all isoforms × all GO terms) ...")
score_matrix = np.zeros((N_TE, len(GO_KEYS)), dtype=np.float32)
auprc_row    = []

for gi, (go_term, go_name) in enumerate(GO_TERMS.items()):
    t0 = time.time()
    y_tr, y_te = load_labels(go_term)
    preds, auprc = run_ensemble(y_tr, y_te)
    score_matrix[:, gi] = preds
    auprc_row.append(auprc)
    n_pos = int(y_te.sum())
    print(f"  [{gi+1:2d}/13] {go_name[:18]:18s}  AUPRC={auprc:.4f}  pos={n_pos:4d}  ({time.time()-t0:.0f}s)")

print(f"\n  Macro AUPRC: {np.mean(auprc_row):.4f}")
print(f"  Score matrix shape: {score_matrix.shape}")

# ── Gene-level score profile ──────────────────────────────────────────────────
# For each gene: collect all isoform indices
gene_isos = defaultdict(list)  # sym -> [(idx, enst_base, ensg)]
for i, sym in enumerate(te_sym):
    gene_isos[sym].append((i, te_enst_base[i], te_ensg_base[i]))

# ── Cross-GO reversal detection ───────────────────────────────────────────────
print("\n  Detecting cross-GO reversals ...")

reversals = []  # list of switch records

for sym, iso_list in gene_isos.items():
    if len(iso_list) < 2: continue

    # Score matrix for this gene's isoforms: (n_iso, 13)
    idxs = [x[0] for x in iso_list]
    ensts = [x[1] for x in iso_list]
    ensg  = iso_list[0][2]
    gene_scores = score_matrix[idxs, :]  # (n_iso, 13)

    # For each pair of isoforms (i, j), find:
    #   GO terms where i >> j (A_up)
    #   GO terms where j >> i (B_up)
    for a in range(len(iso_list)):
        for b in range(a+1, len(iso_list)):
            sa = gene_scores[a]  # score vector for isoform a (13,)
            sb = gene_scores[b]  # score vector for isoform b (13,)
            diff = sa - sb       # positive: a > b

            a_wins = [(gi, diff[gi]) for gi in range(13) if diff[gi] >=  REVERSAL_GAP]
            b_wins = [(gi, -diff[gi]) for gi in range(13) if diff[gi] <= -REVERSAL_GAP]

            if not a_wins or not b_wins: continue  # no reversal

            # Find strongest reversal pair
            best_a = max(a_wins, key=lambda x: x[1])
            best_b = max(b_wins, key=lambda x: x[1])

            reversals.append({
                'gene': sym,
                'ensg': ensg,
                'iso_a': ensts[a],
                'iso_b': ensts[b],
                'score_a': {GO_KEYS[gi]: round(float(sa[gi]), 4) for gi in range(13)},
                'score_b': {GO_KEYS[gi]: round(float(sb[gi]), 4) for gi in range(13)},
                # Best reversal pair
                'go_A':   GO_KEYS[best_a[0]],
                'name_A': GO_NAMES[best_a[0]],
                'gap_A':  round(float(best_a[1]), 4),   # a wins here
                'go_B':   GO_KEYS[best_b[0]],
                'name_B': GO_NAMES[best_b[0]],
                'gap_B':  round(float(best_b[1]), 4),   # b wins here
                'reversal_strength': round(float(best_a[1] + best_b[1]), 4),
                # All terms where a wins / b wins
                'a_wins_terms': [(GO_KEYS[gi], GO_NAMES[gi], round(float(d), 3)) for gi, d in a_wins],
                'b_wins_terms': [(GO_KEYS[gi], GO_NAMES[gi], round(float(d), 3)) for gi, d in b_wins],
                'n_a_wins': len(a_wins),
                'n_b_wins': len(b_wins),
            })

reversals.sort(key=lambda x: -x['reversal_strength'])

# Deduplicate by gene (keep best per gene)
seen_genes = set()
top_reversals = []
for r in reversals:
    if r['gene'] not in seen_genes:
        seen_genes.add(r['gene'])
        top_reversals.append(r)

print(f"\n  Total isoform pairs with cross-GO reversal: {len(reversals)}")
print(f"  Unique genes with reversal: {len(top_reversals)}")

# ── Target gene deep-dive ─────────────────────────────────────────────────────
print("\n" + "="*65)
print("  TARGET GENE CROSS-GO PROFILES")
print("="*65)

for gene in TARGET_GENES:
    isos = gene_isos.get(gene, [])
    if not isos: continue

    idxs  = [x[0] for x in isos]
    ensts = [x[1] for x in isos]
    gs    = score_matrix[idxs, :]  # (n_iso, 13)

    print(f"\n  {gene}  (n_isos={len(isos)})")
    print(f"  {'ENST':18s}  " + "  ".join(f"{n[:6]:6s}" for n in GO_NAMES))
    print(f"  {'-'*(18 + 9*13)}")

    # Sort by max score desc
    order = sorted(range(len(isos)), key=lambda i: -gs[i].max())
    for oi in order[:6]:  # show top 6 isoforms
        row = "  ".join(f"{gs[oi, gi]:6.3f}" for gi in range(13))
        print(f"  {ensts[oi]:18s}  {row}")

    # Find reversals for this gene
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
print("\n" + "="*65)
print("  TOP-20 CROSS-GO FUNCTIONAL SWITCHES (by reversal strength)")
print("="*65)
print(f"  {'Gene':12s} {'A:iso_a wins':20s} {'gapA':5s}  {'B:iso_b wins':20s} {'gapB':5s}  {'total':5s}")
print(f"  {'-'*80}")
for r in top_reversals[:20]:
    print(f"  {r['gene']:12s} {r['name_A'][:20]:20s} {r['gap_A']:5.3f}  {r['name_B'][:20]:20s} {r['gap_B']:5.3f}  {r['reversal_strength']:5.3f}")

# ── Summary statistics ────────────────────────────────────────────────────────
print(f"\n  Reversal gap threshold: {REVERSAL_GAP}")
print(f"  Genes with ≥1 reversal pair: {len(top_reversals)}")
n_target_found = sum(1 for g in TARGET_GENES if any(r['gene'] == g for r in reversals))
print(f"  Target genes with reversal: {n_target_found}/{len(TARGET_GENES)}")

# ── Save ──────────────────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M')

# Save score matrix
np.save(f'{OUT_DIR}/score_matrix_{ts}.npy', score_matrix)

result = {
    'timestamp': ts,
    'auprc_per_go': {GO_KEYS[i]: round(auprc_row[i], 4) for i in range(13)},
    'macro_auprc': round(float(np.mean(auprc_row)), 4),
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
             'scores': {GO_KEYS[gi]: round(float(gs[i, gi]), 4) for gi in range(13)},
             'max_score': round(float(gs[i].max()), 4),
             'max_go': GO_KEYS[int(gs[i].argmax())],
             'max_go_name': GO_NAMES[int(gs[i].argmax())],
            }
            for i in sorted(range(len(isos)), key=lambda k: -gs[k].max())
        ],
        'reversals': [r for r in reversals if r['gene'] == gene],
    }

json.dump(result, open(f'{OUT_DIR}/cross_go_{ts}.json', 'w'), indent=2)
print(f"\n  [Saved] {OUT_DIR}/cross_go_{ts}.json")
print(f"  [Saved] {OUT_DIR}/score_matrix_{ts}.npy")
print("\nALL DONE")
