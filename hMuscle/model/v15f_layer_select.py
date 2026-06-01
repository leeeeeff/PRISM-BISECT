# -*- coding: utf-8 -*-
"""
v15f_layer_select.py
=====================
v15d 아키텍처 동일. GO term별 최적 단일 ESM-2 레이어 선택 (640-dim 유지).

핵심 가설:
  probe 분석에서 각 GO term은 특정 레이어에서 최적 표현을 가짐.
  L30(v15d)이 아닌 term별 최적 레이어를 사용하면 차원 유지 + probe 이득 전달.

비교 대상:
  v15d: 모든 term에 L30 (macro AUPRC 0.7022)
  v15e: 모든 term에 L7+L18+L27 concat (macro AUPRC 0.6409, -0.0613)
  v15f: term별 최적 레이어 (640-dim, 예상 > v15d)

레이어 매핑 출처:
  [CONFIRMED] layer_probe_results.json + layer_probe_expanded_results.json
  [PROBE] v_layer_probe_v15d_terms.py 완료 후 교체 필요

실행:
  cd hMuscle/model/
  CUDA_VISIBLE_DEVICES=0 nohup conda run -n isoform_env python3 -u v15f_layer_select.py \
      > ../../logs_isoform/v15f_$(date +%Y%m%d_%H%M).log 2>&1 &
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
OUT_DIR   = '../../reports/v15_bp_clean'

N_SEEDS      = 5
REVERSAL_GAP = 0.20
TARGET_GENES = ['TPM1', 'TPM2', 'KIF1B', 'SEH1L', 'GABARAPL1', 'DMD', 'TPM3', 'OBSCN']

# ── GO terms (v15d와 동일 순서) ────────────────────────────────────────────────
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

# ── Per-GO-term 최적 레이어 매핑 ──────────────────────────────────────────────
# [CONFIRMED]: probe 데이터로 검증된 값
# [PROBE]: v_layer_probe_v15d_terms.py 완료 후 교체 필요
#   probe 결과 경로: ../../reports/layer_probe/layer_probe_v15d_terms_results.json
GO_TERM_OPTIMAL_LAYER = {
    # ── CONFIRMED (original probe + expanded probe) ─────────────────────────
    'GO:0006914': 18,   # Autophagy — CM type (NAS=2.05), LR peaks L18        [CONFIRMED]
    'GO:0007005': 27,   # Mito org — MC type (NAS=1.75), LR peaks L27         [CONFIRMED]
    'GO:0006096': 29,   # Glycolysis — SF type (NAS=0.93), LR peaks L29       [CONFIRMED]
    'GO:0007519': 30,   # Skeletal muscle dev — WK type, weak signal → L30    [CONFIRMED]
    'GO:0006941': 11,   # Muscle contraction — ID type, LR peaks L11          [CONFIRMED]
    'GO:0000226': 26,   # MT cyto org — MC type, expanded probe: L26 best     [CONFIRMED]

    # ── PLACEHOLDER (v_layer_probe_v15d_terms.py 결과로 교체 필요) ──────────
    'GO:0007204': 9,   # Ca2+ signaling — ID/MC hybrid est.  L13  [PROBE]
    'GO:0045214': 28,   # Sarcomere org — MC type (structural) L27  [PROBE]
    'GO:0043161': 25,   # Proteasome-UPS — MC type (≈Ubiquitin) L27 [PROBE]
    'GO:0042692': 22,   # Muscle cell diff — CM/MC est.         L18  [PROBE]
    'GO:0055074': 13,   # Ca2+ homeostasis — MC est.            L18  [PROBE]
    'GO:0007517': 14,   # Muscle organ dev — WK-like (v15d 0.64) L30 [PROBE]
    'GO:0032006': 26,   # TOR signaling — ID type (kinase) est. L13  [PROBE]
    'GO:0030048': 30,   # Actin-based movement — MC (cytoskel.) L26  [PROBE]
    'GO:0007268': 28,   # Synaptic transmission — ID type est.  L13  [PROBE]
    'GO:0007018': 16,   # MT-based movement — MC type est.      L27  [PROBE]
    'GO:0031175': 9,   # Neuron proj dev — MC type est.        L26  [PROBE]
    'GO:0030182': 26,   # Neuron diff — CM/ID type est.         L18  [PROBE]
}

# 매핑 완전성 검증
assert set(GO_TERM_OPTIMAL_LAYER.keys()) == set(GO_KEYS), \
    "GO_TERM_OPTIMAL_LAYER key mismatch — update required"

confirmed_terms = {
    'GO:0006914', 'GO:0007005', 'GO:0006096',
    'GO:0007519', 'GO:0006941', 'GO:0000226',
}

# ── Load embeddings (레이어별 on-demand) ──────────────────────────────────────
_emb_cache = {}

def load_layer_emb(layer_num, split='train'):
    key = (layer_num, split)
    if key not in _emb_cache:
        if split == 'train':
            path = f'{DATA_DIR}/esm2_train_human_layer{layer_num:02d}_t30_150M.npy'
        else:
            path = f'{DATA_DIR}/esm2_layer_{layer_num:02d}_t30_150M.npy'
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing embedding: {path}")
        _emb_cache[key] = np.load(path).astype(np.float32)
    return _emb_cache[key]

# ── Load IDs & gene mappings ───────────────────────────────────────────────────
def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

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

print("="*70)
print("  v15f_layer_select: per-GO-term optimal single layer (640-dim)")
print("="*70)
print(f"  Test isoforms: {N_TE}  GO terms: {N_GO}")
print(f"  Confirmed layer assignments: {len(confirmed_terms)}/18")
print(f"  Placeholder assignments:     {N_GO - len(confirmed_terms)}/18")
print()
print(f"  {'GO term':<28} | Layer | Status")
print(f"  {'-'*50}")
for go_id, name in GO_TERMS.items():
    L = GO_TERM_OPTIMAL_LAYER[go_id]
    status = 'CONFIRMED' if go_id in confirmed_terms else 'PLACEHOLDER'
    print(f"  {name:<28} | L{L:02d}   | {status}")
print()

# ── Model (v15d와 동일) ────────────────────────────────────────────────────────
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

def run_ensemble(X_tr, X_te, y_tr, y_te):
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

# ── v15d baseline for inline delta display ────────────────────────────────────
V15D_AUPRC = {
    'GO:0007204': 0.6884, 'GO:0045214': 0.8667, 'GO:0006941': 0.7016,
    'GO:0006914': 0.6600, 'GO:0043161': 0.7772, 'GO:0007519': 0.7775,
    'GO:0042692': 0.6740, 'GO:0055074': 0.6729, 'GO:0007005': 0.6873,
    'GO:0007517': 0.6401, 'GO:0032006': 0.4959, 'GO:0030048': 0.7356,
    'GO:0006096': 0.8143, 'GO:0007268': 0.6672, 'GO:0007018': 0.7402,
    'GO:0031175': 0.6823, 'GO:0030182': 0.6466, 'GO:0000226': 0.7118,
}

# ── Build score matrix ────────────────────────────────────────────────────────
print(f"  Building score matrix ({N_TE} isoforms × {N_GO} GO terms) ...")
score_matrix = np.zeros((N_TE, N_GO), dtype=np.float32)
auprc_row    = []
layer_used   = []

_prev_layer = None
for gi, (go_term, go_name) in enumerate(GO_TERMS.items()):
    t0 = time.time()
    opt_layer = GO_TERM_OPTIMAL_LAYER[go_term]
    layer_used.append(opt_layer)

    # 레이어 변경 시에만 로드 (같은 레이어 연속이면 캐시 사용)
    X_tr = load_layer_emb(opt_layer, 'train')
    X_te = load_layer_emb(opt_layer, 'test')

    y_tr, y_te = load_labels(go_term)
    preds, auprc = run_ensemble(X_tr, X_te, y_tr, y_te)
    score_matrix[:, gi] = preds
    auprc_row.append(auprc)

    n_pos   = int(y_te.sum())
    v15d_v  = V15D_AUPRC.get(go_term, 0.0)
    delta   = auprc - v15d_v
    sign    = '+' if delta >= 0 else ''
    status  = 'OK' if go_term in confirmed_terms else 'PH'
    print(f"  [{gi+1:2d}/{N_GO}] {go_name[:20]:20s}  L{opt_layer:02d}  "
          f"AUPRC={auprc:.4f}  v15d={v15d_v:.4f}  {sign}{delta:.4f}  "
          f"pos={n_pos:4d}  ({time.time()-t0:.0f}s)  [{status}]")

macro_all  = float(np.mean(auprc_row))
macro_v15d = float(np.mean(list(V15D_AUPRC.values())))

print(f"\n  Macro AUPRC (v15f all 18): {macro_all:.4f}")
print(f"  Macro AUPRC (v15d all 18): {macro_v15d:.4f}")
print(f"  Macro delta (v15f - v15d): {macro_all - macro_v15d:+.4f}")

# confirmed terms only
conf_ids  = list(confirmed_terms)
conf_v15f = [auprc_row[GO_KEYS.index(g)] for g in conf_ids if g in GO_KEYS]
conf_v15d = [V15D_AUPRC[g] for g in conf_ids if g in GO_KEYS]
print(f"\n  Confirmed-term delta: {float(np.mean(conf_v15f)) - float(np.mean(conf_v15d)):+.4f}")
print(f"  Placeholder-term delta: "
      f"{macro_all - float(np.mean(conf_v15f + [auprc_row[GO_KEYS.index(g)] for g in GO_KEYS if g not in confirmed_terms])):.4f}")

# ── Gene index + reversal detection (v15d와 동일 로직) ────────────────────────
gene_isos = defaultdict(list)
for i, sym in enumerate(te_sym):
    gene_isos[sym].append((i, te_enst_base[i], te_ensg_base[i]))

print("\n  Detecting cross-GO reversals ...")
reversals = []
for sym, iso_list in gene_isos.items():
    if len(iso_list) < 2: continue
    idxs  = [x[0] for x in iso_list]
    ensts = [x[1] for x in iso_list]
    ensg  = iso_list[0][2]
    gs    = score_matrix[idxs, :]
    for a in range(len(iso_list)):
        for b in range(a+1, len(iso_list)):
            sa, sb = gs[a], gs[b]
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
seen_genes    = set()
top_reversals = []
for r in reversals:
    if r['gene'] not in seen_genes:
        seen_genes.add(r['gene'])
        top_reversals.append(r)

print(f"  Total reversals: {len(reversals)}  Unique genes: {len(top_reversals)}")

# ── Target gene summary ───────────────────────────────────────────────────────
print("\n  Target gene reversal summary:")
for gene in TARGET_GENES:
    gene_revs = [r for r in reversals if r['gene'] == gene]
    if gene_revs:
        best = gene_revs[0]
        print(f"  {gene}: {best['iso_a']} vs {best['iso_b']}  "
              f"A={best['name_A']}(+{best['gap_A']:.3f}) B={best['name_B']}(+{best['gap_B']:.3f})")
    else:
        print(f"  {gene}: no reversal (gap < {REVERSAL_GAP})")

# ── Save ──────────────────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M')

os.makedirs(OUT_DIR, exist_ok=True)
np.save(f'{OUT_DIR}/score_matrix_v15f_{ts}.npy', score_matrix)

result = {
    'model': 'v15f_layer_select',
    'description': 'per-GO-term optimal single ESM-2 layer (640-dim)',
    'timestamp': ts,
    'n_go_terms': N_GO,
    'go_terms': GO_TERMS,
    'layer_mapping': GO_TERM_OPTIMAL_LAYER,
    'confirmed_terms': sorted(confirmed_terms),
    'placeholder_terms': [g for g in GO_KEYS if g not in confirmed_terms],
    'auprc_per_go': {GO_KEYS[i]: round(auprc_row[i], 4) for i in range(N_GO)},
    'macro_auprc_all18': round(macro_all, 4),
    'macro_auprc_v15d': round(macro_v15d, 4),
    'macro_delta': round(macro_all - macro_v15d, 4),
    'v15d_auprc_per_go': V15D_AUPRC,
    'delta_per_go': {GO_KEYS[i]: round(auprc_row[i] - V15D_AUPRC.get(GO_KEYS[i], 0), 4)
                     for i in range(N_GO)},
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

out_json = f'{OUT_DIR}/v15f_layer_select_{ts}.json'
json.dump(result, open(out_json, 'w'), indent=2)
print(f"\n  [Saved] {out_json}")
print(f"  [Saved] {OUT_DIR}/score_matrix_v15f_{ts}.npy")
print("\nDONE")
