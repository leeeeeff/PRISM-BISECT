# -*- coding: utf-8 -*-
"""
v17_brain_model.py
==================
뇌조직 전용 isoform functional switch 분석 (v16 수정)

v16 문제 수정:
  - v16: 근육 학습셋(31668)으로 모델 학습 → 뇌조직 테스트 → AUPRC=0.029 (random)
  - v17: 뇌조직 ESM-2로 gene-block 80/20 split → 뇌 특화 모델 학습
         → AUPRC가 의미있어야 functional switch가 신뢰 가능

설계:
  - Train: brain_only ESM-2에서 gene-block 80% (~11,883 genes, ~31,780 isoforms)
  - Eval:  held-out 20% genes의 PC isoforms (AUPRC 측정)
  - Score: 학습된 모델로 전체 15,943 brain-PC isoforms 스코어링
  - Switch/Reversal: 전체 brain-PC 기반 분석 (v16과 동일 로직)
"""

import os, json, re, time, zipfile
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
HPA_PATH   = '../results_isoform/features/hpa_isoform/transcript_rna_tissue.tsv.zip'
OUT_DIR    = '../../reports/v17_brain_model'

os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_FRAC  = 0.80
N_SEEDS     = 5
SCORE_GAP   = 0.30
SCORE_RATIO = 3.0
DTU_DELTA   = 0.10
REVERSAL_GAP = 0.20

GO_TERMS = {
    'GO:0043005': 'Neuron projection',
    'GO:0030182': 'Neuron diff',
    'GO:0048666': 'Neuron dev',
    'GO:0045202': 'Synapse assembly',
    'GO:0007268': 'Synaptic transmission',
    'GO:0050804': 'Modulation of syn.trans',
    'GO:0007411': 'Axon guidance',
    'GO:0007018': 'MT-based movement',
    'GO:0000226': 'MT cytoskeleton org',
    'GO:0006836': 'Neurotrans. transport',
    'GO:0006979': 'Oxidative stress resp',
    'GO:0006954': 'Inflammatory response',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007204': 'Ca2+ signaling',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0032006': 'TOR signaling',
}
GO_KEYS  = list(GO_TERMS.keys())
GO_NAMES = list(GO_TERMS.values())
N_GO     = len(GO_KEYS)

BRAIN_TISSUES = ['cerebral cortex']
COMP_TISSUES  = ['skeletal muscle', 'adipose tissue', 'liver', 'kidney',
                 'colon', 'thyroid gland', 'lung', 'heart muscle']

TARGET_GENES = ['KIF1B', 'APP', 'MAPT', 'SNCA', 'BDNF', 'NRXN1',
                'PTPRD', 'CNTNAP2', 'SHANK3', 'DLGAP1']

print("="*70)
print("  v17 Brain-Specific Isoform Function Prediction")
print("="*70)

# ── Parse brain GTF ──────────────────────────────────────────────────────────
print("  Parsing brain GTF ...")
brain_enst, brain_ensg, brain_sym, brain_biotype = [], [], [], []
with open(f'{BRAIN_DIR}/brain_only.gtf') as f:
    for line in f:
        if '\ttranscript\t' not in line: continue
        enst  = re.search(r'transcript_id "(ENST\d+)"', line)
        ensg  = re.search(r'gene_id "(ENSG\d+)"', line)
        gname = re.search(r'gene_name "([^"]+)"', line)
        btype = re.search(r'transcript_type "([^"]+)"', line)
        if not (enst and ensg): continue
        brain_enst.append(enst.group(1))
        brain_ensg.append(ensg.group(1))
        brain_sym.append(gname.group(1) if gname else ensg.group(1))
        brain_biotype.append(btype.group(1) if btype else 'unknown')

brain_enst  = np.array(brain_enst)
brain_sym   = np.array(brain_sym)
brain_biotype = np.array(brain_biotype)
pc_mask = (brain_biotype == 'protein_coding')
pc_idx  = np.where(pc_mask)[0]

all_genes = sorted(set(brain_sym))
print(f"  Brain: {len(brain_enst)} isoforms, {len(all_genes)} genes, {pc_mask.sum()} PC isoforms")

# ── Gene-block 80/20 split ───────────────────────────────────────────────────
np.random.seed(42)
gene_order = np.random.permutation(len(all_genes))
n_train_g  = int(len(all_genes) * TRAIN_FRAC)
train_genes = set(all_genes[i] for i in gene_order[:n_train_g])
test_genes  = set(all_genes[i] for i in gene_order[n_train_g:])

tr_mask = np.array([s in train_genes for s in brain_sym])
# eval set: PC isoforms from held-out genes (AUPRC measurement)
ev_mask = np.array([s in test_genes and b == 'protein_coding'
                    for s, b in zip(brain_sym, brain_biotype)])

print(f"  Train genes: {len(train_genes)}, eval genes: {len(test_genes)}")
print(f"  Train isoforms: {tr_mask.sum()}, Eval PC isoforms: {ev_mask.sum()}")

# ── Load ESM-2 ───────────────────────────────────────────────────────────────
print("  Loading brain ESM-2 ...")
X_brain_all = np.load(f'{BRAIN_DIR}/brain_only_esm2_t30_150M.npy').astype(np.float32)
X_tr = X_brain_all[tr_mask]
X_ev = X_brain_all[ev_mask]
X_all_pc = X_brain_all[pc_idx]   # all 15943 brain-PC isoforms for switch analysis

tr_sym_arr = brain_sym[tr_mask]
ev_sym_arr = brain_sym[ev_mask]
ev_enst_arr = brain_enst[ev_mask]

# PC metadata for functional switch analysis
pc_enst_arr = brain_enst[pc_idx]
pc_sym_arr  = brain_sym[pc_idx]
pc_ensg_arr = np.array(brain_ensg)[pc_idx] if len(brain_ensg) > 0 else pc_sym_arr

N_TR, N_EV, N_PC = len(X_tr), len(X_ev), len(X_all_pc)
print(f"  X_tr: {X_tr.shape}  X_ev: {X_ev.shape}  X_all_pc: {X_all_pc.shape}")

# ── GO annotations ───────────────────────────────────────────────────────────
gene2go = {}
with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
    for line in f:
        p = line.strip().split()
        if p: gene2go[p[0]] = set(p[1:])

def load_labels(go_term):
    y_tr = np.array([1 if s in gene2go and go_term in gene2go[s] else 0
                     for s in tr_sym_arr], dtype=np.float32)
    y_ev = np.array([1 if s in gene2go and go_term in gene2go[s] else 0
                     for s in ev_sym_arr], dtype=np.float32)
    return y_tr, y_ev

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

def run_ensemble(y_tr, y_ev):
    """Train on X_tr, score X_all_pc for functional switch analysis.
    Eval AUPRC on X_ev (held-out genes) for unbiased quality estimate."""
    seed_preds_pc = []
    seed_preds_ev = []
    for s in range(N_SEEDS):
        tf.random.set_seed(s * 137 + 42)
        np.random.seed(s * 137 + 42)
        idx = np.random.permutation(N_TR)
        m = build_v10b()
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
        cb = [callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                      restore_best_weights=True, verbose=0),
              callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)]
        m.fit(X_tr[idx], y_tr[idx], epochs=80, batch_size=512,
              validation_split=0.1, callbacks=cb, verbose=0)
        seed_preds_pc.append(m.predict(X_all_pc, batch_size=1024, verbose=0).flatten())
        seed_preds_ev.append(m.predict(X_ev, batch_size=1024, verbose=0).flatten())

    preds_pc = np.mean(seed_preds_pc, axis=0)   # for switch analysis (all 15943)
    preds_ev = np.mean(seed_preds_ev, axis=0)   # for AUPRC eval (held-out genes)
    auprc = average_precision_score(y_ev, preds_ev) if y_ev.sum() > 0 else 0.0
    pos_rate = float(y_ev.mean()) if y_ev.sum() > 0 else 0.0
    return preds_pc, preds_ev, auprc, pos_rate

# ── Score matrix (all PC isoforms) ──────────────────────────────────────────
print(f"\n  Building score matrix ({N_PC} brain-PC × {N_GO} GO terms) ...")
print(f"  [AUPRC measured on {N_EV} held-out PC isoforms from {len(test_genes)} test genes]")
score_matrix = np.zeros((N_PC, N_GO), dtype=np.float32)
ev_score_mat = np.zeros((N_EV, N_GO), dtype=np.float32)
auprc_row    = []
pos_rates    = []

for gi, (go_term, go_name) in enumerate(GO_TERMS.items()):
    t0 = time.time()
    y_tr, y_ev = load_labels(go_term)
    preds_pc, preds_ev, auprc, pos_rate = run_ensemble(y_tr, y_ev)
    score_matrix[:, gi] = preds_pc
    ev_score_mat[:, gi] = preds_ev
    auprc_row.append(auprc)
    pos_rates.append(pos_rate)
    n_pos_tr = int(y_tr.sum())
    n_pos_ev = int(y_ev.sum())
    print(f"  [{gi+1:2d}/{N_GO}] {go_name[:22]:22s}  "
          f"AUPRC={auprc:.4f}  pos_rate={pos_rate:.4f}  "
          f"tr_pos={n_pos_tr}  ev_pos={n_pos_ev}  ({time.time()-t0:.0f}s)")

macro_auprc = float(np.mean(auprc_row))
print(f"\n  Macro AUPRC (eval, held-out genes): {macro_auprc:.4f}")
print(f"  Score matrix shape: {score_matrix.shape}")

# ── Save score matrix ────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M')
np.save(f'{OUT_DIR}/score_matrix_brain_v17_{ts}.npy', score_matrix)

# ── HPA Brain DTU ────────────────────────────────────────────────────────────
print("\n  Loading HPA expression data for brain DTU ...")
hpa = {}
with zipfile.ZipFile(HPA_PATH) as z:
    with z.open('transcript_rna_tissue.tsv') as f:
        header = f.readline().decode().strip().split('\t')
        tissue_cols = defaultdict(list)
        for ci, col in enumerate(header):
            if col.startswith('TPM.'):
                tname = re.sub(r'\.\d+$', '', col[4:])
                tissue_cols[tname].append(ci)
        brain_cols = []
        for t in BRAIN_TISSUES:
            brain_cols.extend(tissue_cols.get(t, []))
        comp_cols = {t: tissue_cols.get(t, []) for t in COMP_TISSUES}
        for line in f:
            row  = line.decode().strip().split('\t')
            enst = row[1].split('.')[0] if len(row) > 1 else ''
            if not enst: continue
            brain_tpm = np.mean([float(row[ci]) for ci in brain_cols
                                 if ci < len(row) and row[ci] != '']) if brain_cols else 0.0
            comp_tpm = {}
            for t, cols in comp_cols.items():
                vals = [float(row[ci]) for ci in cols if ci < len(row) and row[ci] != '']
                comp_tpm[t] = np.mean(vals) if vals else 0.0
            hpa[enst] = {'brain': brain_tpm, **comp_tpm}
print(f"  HPA loaded: {len(hpa)} transcripts")

def compute_dtu(gene_enst_list):
    total_brain = sum(hpa.get(e, {}).get('brain', 0.0) for e in gene_enst_list)
    dtu = {}
    for enst in gene_enst_list:
        tpm_b = hpa.get(enst, {}).get('brain', 0.0)
        frac_b = tpm_b / total_brain if total_brain > 0 else 0.0
        comp_fracs = []
        for t in COMP_TISSUES:
            total_t = sum(hpa.get(e, {}).get(t, 0.0) for e in gene_enst_list)
            frac_t  = hpa.get(enst, {}).get(t, 0.0) / total_t if total_t > 0 else 0.0
            comp_fracs.append(frac_t)
        dtu[enst] = {
            'frac_brain': round(frac_b, 4),
            'frac_comp_mean': round(float(np.mean(comp_fracs)), 4),
            'dtu_delta': round(frac_b - float(np.mean(comp_fracs)), 4),
            'brain_tpm': round(tpm_b, 3),
        }
    return dtu

# ── Gene-level index (all PC isoforms) ──────────────────────────────────────
gene_isos = defaultdict(list)
for i in range(N_PC):
    gene_isos[pc_sym_arr[i]].append((i, pc_enst_arr[i]))

# ── Functional switch detection ──────────────────────────────────────────────
print("\n  Detecting functional switches ...")
switches = []
for sym, iso_list in gene_isos.items():
    if len(iso_list) < 2: continue
    idxs  = [x[0] for x in iso_list]
    ensts = [x[1] for x in iso_list]
    gs    = score_matrix[idxs, :]
    dtu   = compute_dtu(ensts)
    for a_idx in range(len(iso_list)):
        for b_idx in range(a_idx+1, len(iso_list)):
            sa, sb = gs[a_idx], gs[b_idx]
            for gi in range(N_GO):
                gap   = abs(float(sa[gi] - sb[gi]))
                hi_i  = a_idx if sa[gi] >= sb[gi] else b_idx
                if gap < SCORE_GAP: continue
                hi_s  = max(float(sa[gi]), float(sb[gi]))
                lo_s  = min(float(sa[gi]), float(sb[gi]))
                ratio = hi_s / lo_s if lo_s > 1e-6 else float('inf')
                if ratio < SCORE_RATIO: continue
                hi_enst = ensts[hi_i]
                lo_enst = ensts[b_idx if hi_i == a_idx else a_idx]
                dtu_hi  = dtu[hi_enst]['dtu_delta']
                frac_hi = dtu[hi_enst]['frac_brain']
                switches.append({
                    'gene': sym,
                    'go_term': GO_KEYS[gi],
                    'go_name': GO_NAMES[gi],
                    'top_iso': hi_enst,
                    'bot_iso': lo_enst,
                    'score_top': round(float(hi_s), 4),
                    'score_bot': round(float(lo_s), 4),
                    'score_gap': round(gap, 4),
                    'score_ratio': round(min(ratio, 9999.9), 2),
                    'dtu_top': round(float(dtu_hi), 4),
                    'frac_brain_top': round(float(frac_hi), 4),
                    'brain_tpm_top': dtu[hi_enst]['brain_tpm'],
                    'dtu_concordant': bool(dtu_hi >= DTU_DELTA),
                })

switches.sort(key=lambda x: -(x['score_gap'] + x['dtu_top']))
seen_k = set()
top_switches = []
for s in switches:
    k = (s['gene'], s['go_term'])
    if k not in seen_k:
        seen_k.add(k)
        top_switches.append(s)

dtu_concordant = [s for s in top_switches if s['dtu_concordant']]
print(f"  Switches (gap≥{SCORE_GAP}, ratio≥{SCORE_RATIO}×): {len(top_switches)}")
print(f"  DTU concordant (delta≥{DTU_DELTA}): {len(dtu_concordant)}")

# ── Cross-GO reversal detection ──────────────────────────────────────────────
print("\n  Detecting cross-GO reversals ...")
reversals = []
for sym, iso_list in gene_isos.items():
    if len(iso_list) < 2: continue
    idxs  = [x[0] for x in iso_list]
    ensts = [x[1] for x in iso_list]
    gs    = score_matrix[idxs, :]
    dtu   = compute_dtu(ensts)
    for a in range(len(iso_list)):
        for b in range(a+1, len(iso_list)):
            sa, sb = gs[a], gs[b]
            diff = sa - sb
            a_wins = [(gi, float(diff[gi]))  for gi in range(N_GO) if diff[gi]  >=  REVERSAL_GAP]
            b_wins = [(gi, float(-diff[gi])) for gi in range(N_GO) if diff[gi]  <= -REVERSAL_GAP]
            if not a_wins or not b_wins: continue
            best_a = max(a_wins, key=lambda x: x[1])
            best_b = max(b_wins, key=lambda x: x[1])
            reversals.append({
                'gene': sym,
                'iso_a': ensts[a], 'iso_b': ensts[b],
                'go_A': GO_KEYS[best_a[0]], 'name_A': GO_NAMES[best_a[0]],
                'gap_A': round(best_a[1], 4),
                'go_B': GO_KEYS[best_b[0]], 'name_B': GO_NAMES[best_b[0]],
                'gap_B': round(best_b[1], 4),
                'reversal_strength': round(best_a[1]+best_b[1], 4),
                'n_a_wins': len(a_wins), 'n_b_wins': len(b_wins),
                'a_wins_terms': [(GO_NAMES[gi], round(d,3)) for gi,d in a_wins],
                'b_wins_terms': [(GO_NAMES[gi], round(d,3)) for gi,d in b_wins],
                'dtu_a': float(dtu[ensts[a]]['dtu_delta']),
                'dtu_b': float(dtu[ensts[b]]['dtu_delta']),
            })

reversals.sort(key=lambda x: -x['reversal_strength'])
seen_g = set()
top_reversals = []
for r in reversals:
    if r['gene'] not in seen_g:
        seen_g.add(r['gene'])
        top_reversals.append(r)

print(f"  Total reversal pairs: {len(reversals)}")
print(f"  Unique genes with reversal: {len(top_reversals)}")

# ── Print results ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  TOP-30 BRAIN FUNCTIONAL SWITCHES (DTU concordant first)")
print("="*70)
print(f"  {'Gene':10s} {'GO':22s} {'gap':5s} {'ratio':7s} {'dtu':6s}  {'top_iso'}")
print(f"  {'-'*75}")
all_top = sorted(top_switches,
                 key=lambda x: (not x['dtu_concordant'], -(x['score_gap']+x['dtu_top'])))
for s in all_top[:30]:
    mark = '✓' if s['dtu_concordant'] else ' '
    print(f"  {mark}{s['gene']:9s} {s['go_name'][:22]:22s} {s['score_gap']:5.3f} "
          f"{s['score_ratio']:6.1f}x {s['dtu_top']:+6.3f}  {s['top_iso']}")

print("\n" + "="*70)
print("  TOP-20 CROSS-GO FUNCTIONAL SWITCHES")
print("="*70)
print(f"  {'Gene':10s} {'A wins':24s} {'gapA':5s}  {'B wins':24s} {'gapB':5s}  total  dtu_a   dtu_b")
print(f"  {'-'*95}")
for r in top_reversals[:20]:
    print(f"  {r['gene']:10s} {r['name_A'][:24]:24s} {r['gap_A']:5.3f}  "
          f"{r['name_B'][:24]:24s} {r['gap_B']:5.3f}  "
          f"{r['reversal_strength']:5.3f}  {r['dtu_a']:+5.3f}  {r['dtu_b']:+5.3f}")

# ── Target gene profiles ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("  TARGET GENE PROFILES")
print("="*70)
for gene in TARGET_GENES:
    isos = gene_isos.get(gene, [])
    if not isos:
        print(f"\n  {gene}: not in brain PC set")
        continue
    idxs  = [x[0] for x in isos]
    ensts = [x[1] for x in isos]
    gs    = score_matrix[idxs, :]
    dtu   = compute_dtu(ensts)
    print(f"\n  {gene}  (n_isos={len(isos)})")
    hdr = "  ".join(f"{n[:7]:7s}" for n in GO_NAMES)
    print(f"  {'ENST':18s}  {hdr}  brain_frac  dtu_delta")
    print("  " + "-"*(20+9*N_GO+25))
    order = sorted(range(len(isos)), key=lambda i: -gs[i].max())
    for oi in order[:6]:
        row   = "  ".join(f"{gs[oi,gi]:7.3f}" for gi in range(N_GO))
        enst  = ensts[oi]
        bf    = dtu[enst]['frac_brain']
        delta = dtu[enst]['dtu_delta']
        print(f"  {enst:18s}  {row}  {bf:10.3f}  {delta:+9.3f}")
    gene_revs = [r for r in reversals if r['gene'] == gene]
    if gene_revs:
        best = gene_revs[0]
        print(f"\n  Best cross-GO reversal: {best['iso_a']} vs {best['iso_b']}")
        print(f"    A wins ({best['name_A']}): gap={best['gap_A']:.3f}, DTU={best['dtu_a']:+.3f}")
        print(f"    B wins ({best['name_B']}): gap={best['gap_B']:.3f}, DTU={best['dtu_b']:+.3f}")
        print(f"    A_wins: {best['a_wins_terms']}")
        print(f"    B_wins: {best['b_wins_terms']}")
    else:
        print(f"\n  No cross-GO reversal (gap < {REVERSAL_GAP})")

# ── AUPRC summary ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  GO TERM AUPRC SUMMARY (eval on held-out genes)")
print("="*70)
print(f"  {'GO term':28s}  {'AUPRC':8s}  {'pos_rate':8s}  {'vs_random':9s}")
for gi, (go_term, go_name) in enumerate(GO_TERMS.items()):
    vs_random = auprc_row[gi] / pos_rates[gi] if pos_rates[gi] > 0 else 0.0
    print(f"  {go_name[:28]:28s}  {auprc_row[gi]:.4f}    {pos_rates[gi]:.4f}    {vs_random:.2f}×")
print(f"  {'MACRO':28s}  {macro_auprc:.4f}")

# ── Save results ─────────────────────────────────────────────────────────────
result = {
    'timestamp': ts,
    'model': 'v17_brain_only_gene_block_80_20',
    'n_brain_isoforms': int(len(brain_enst)),
    'n_train_genes': len(train_genes),
    'n_test_genes': len(test_genes),
    'n_train_isoforms': int(N_TR),
    'n_eval_isoforms': int(N_EV),
    'n_score_isoforms': int(N_PC),
    'go_terms': GO_TERMS,
    'auprc_per_go': {GO_KEYS[i]: round(auprc_row[i], 4) for i in range(N_GO)},
    'pos_rate_per_go': {GO_KEYS[i]: round(pos_rates[i], 4) for i in range(N_GO)},
    'macro_auprc': round(macro_auprc, 4),
    'n_switches': len(top_switches),
    'n_dtu_concordant': len(dtu_concordant),
    'n_reversal_genes': len(top_reversals),
    'top_switches': top_switches[:100],
    'top_reversals': top_reversals[:50],
    'dtu_concordant_switches': dtu_concordant[:50],
    'target_gene_profiles': {},
}

for gene in TARGET_GENES:
    isos = gene_isos.get(gene, [])
    if not isos: continue
    idxs  = [x[0] for x in isos]
    ensts = [x[1] for x in isos]
    gs    = score_matrix[idxs, :]
    dtu   = compute_dtu(ensts)
    result['target_gene_profiles'][gene] = {
        'n_isoforms': len(isos),
        'isoforms': [
            {'enst': ensts[i],
             'scores': {GO_KEYS[gi]: round(float(gs[i,gi]), 4) for gi in range(N_GO)},
             'max_score': round(float(gs[i].max()), 4),
             'max_go_name': GO_NAMES[int(gs[i].argmax())],
             'dtu_delta': float(dtu[ensts[i]]['dtu_delta']),
             'frac_brain': float(dtu[ensts[i]]['frac_brain']),
             'brain_tpm': float(dtu[ensts[i]]['brain_tpm']),
            }
            for i in sorted(range(len(isos)), key=lambda k: -gs[k].max())
        ],
        'reversals': [r for r in reversals if r['gene'] == gene],
        'switches': [s for s in top_switches if s['gene'] == gene],
    }

tsv_path = f'{OUT_DIR}/brain_switches_v17_{ts}.tsv'
with open(tsv_path, 'w') as f:
    cols = ['gene','go_name','score_top','score_bot','score_gap','score_ratio',
            'dtu_top','frac_brain_top','brain_tpm_top','dtu_concordant','top_iso','bot_iso']
    f.write('\t'.join(cols) + '\n')
    for s in top_switches:
        f.write('\t'.join(str(s.get(c,'')) for c in cols) + '\n')

with open(f'{OUT_DIR}/brain_switch_v17_{ts}.json', 'w') as f:
    json.dump(result, f, indent=2)

np.save(f'{OUT_DIR}/score_matrix_brain_v17_{ts}.npy', score_matrix)

print(f"\n  [Saved] {OUT_DIR}/brain_switch_v17_{ts}.json")
print(f"  [Saved] {OUT_DIR}/brain_switches_v17_{ts}.tsv")
print(f"  [Saved] {OUT_DIR}/score_matrix_brain_v17_{ts}.npy")
print("\nALL DONE")
