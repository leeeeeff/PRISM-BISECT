# -*- coding: utf-8 -*-
"""
v16_brain_functional_switch.py
================================
뇌조직 데이터 기반 isoform functional switch 분석
  - v15의 골격근 분석을 뇌조직으로 확장
  - 뇌 특화 GO term 18개 사용
  - HPA cerebral cortex DTU (vs 비뇌 조직들)
  - Cross-GO reversal로 진짜 functional switch 검증
  - KIF1B α/β 포함 타겟 유전자 심층 분석

Brain GO set (18개):
  신경기능: neuron projection/diff/dev, synapse, synaptic transmission,
            modulation of synaptic transmission, axon guidance, MT movement
  신경퇴행: oxidative stress, inflammation, autophagy, proteasome-UPS
  신호전달: Ca2+ signaling/homeostasis, TOR signaling, mitochondrion org
  수송:     neurotransmitter transport, MT cytoskeleton org
"""

import os, json, re, time, zipfile, csv
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
OUT_DIR    = '../../reports/v16_brain_switch'

os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS     = 5
SCORE_GAP   = 0.30
SCORE_RATIO = 3.0
DTU_DELTA   = 0.10
REVERSAL_GAP = 0.20

# ── Brain-focused GO terms (18개) ──────────────────────────────────────────────
GO_TERMS = {
    # 신경 기능 핵심
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
    # 신경퇴행 / AD 관련
    'GO:0006979': 'Oxidative stress resp',
    'GO:0006954': 'Inflammatory response',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    # 신호전달 / 에너지
    'GO:0007204': 'Ca2+ signaling',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0032006': 'TOR signaling',
}
GO_KEYS  = list(GO_TERMS.keys())
GO_NAMES = list(GO_TERMS.values())
N_GO     = len(GO_KEYS)

# Brain DTU: cerebral cortex vs 비뇌 조직
BRAIN_TISSUES = ['cerebral cortex']
COMP_TISSUES  = ['skeletal muscle', 'adipose tissue', 'liver', 'kidney',
                 'colon', 'thyroid gland', 'lung', 'heart muscle']

TARGET_GENES = ['KIF1B', 'APP', 'MAPT', 'SNCA', 'BDNF', 'NRXN1',
                'PTPRD', 'CNTNAP2', 'SHANK3', 'DLGAP1']

print("="*70)
print("  v16 Brain Tissue Functional Switch Analysis")
print("="*70)

# ── Load IDs ──────────────────────────────────────────────────────────────────
def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

# ── Parse brain GTF → isoform/gene lists ──────────────────────────────────────
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

# protein_coding only mask (CDS 있는 isoform만 GO 예측 의미 있음)
pc_mask = np.array([b == 'protein_coding' for b in brain_biotype])
pc_idx  = np.where(pc_mask)[0]

print(f"  Brain isoforms: {len(brain_enst)} total, {pc_mask.sum()} protein_coding")

# ── Load ESM-2 embeddings ─────────────────────────────────────────────────────
print("  Loading ESM-2 ...")
X_tr     = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_brain_all = np.load(f'{BRAIN_DIR}/brain_only_esm2_t30_150M.npy').astype(np.float32)
X_te     = X_brain_all[pc_idx]   # protein_coding only

# Filtered ID lists
te_enst = [brain_enst[i] for i in pc_idx]
te_ensg = [brain_ensg[i] for i in pc_idx]
te_sym  = [brain_sym[i]  for i in pc_idx]
N_TE = len(te_enst)

tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')
tr_sym    = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_geneid]

print(f"  Train: {X_tr.shape}  Test(brain-PC): {X_te.shape}")

# ── GO label loader ───────────────────────────────────────────────────────────
gene2go = {}
with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
    for line in f:
        p = line.strip().split()
        if p: gene2go[p[0]] = set(p[1:])

def load_labels(go_term):
    y_tr = np.array([1 if s in gene2go and go_term in gene2go[s] else 0
                     for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in gene2go and go_term in gene2go[s] else 0
                     for s in te_sym], dtype=np.float32)
    return y_tr, y_te

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

# ── Score matrix ──────────────────────────────────────────────────────────────
print(f"\n  Building score matrix ({N_TE} brain-PC isoforms × {N_GO} GO terms) ...")
score_matrix = np.zeros((N_TE, N_GO), dtype=np.float32)
auprc_row    = []

for gi, (go_term, go_name) in enumerate(GO_TERMS.items()):
    t0 = time.time()
    y_tr, y_te = load_labels(go_term)
    preds, auprc = run_ensemble(y_tr, y_te)
    score_matrix[:, gi] = preds
    auprc_row.append(auprc)
    n_pos = int(y_te.sum())
    print(f"  [{gi+1:2d}/{N_GO}] {go_name[:22]:22s}  AUPRC={auprc:.4f}  pos={n_pos:4d}  ({time.time()-t0:.0f}s)")

print(f"\n  Macro AUPRC: {np.mean(auprc_row):.4f}")
print(f"  Score matrix: {score_matrix.shape}")

# ── Save score matrix ─────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M')
np.save(f'{OUT_DIR}/score_matrix_brain_{ts}.npy', score_matrix)

# ── HPA Brain DTU ─────────────────────────────────────────────────────────────
print("\n  Loading HPA expression data for brain DTU ...")

hpa = {}   # enst_base -> {tissue: mean_tpm}
with zipfile.ZipFile(HPA_PATH) as z:
    with z.open('transcript_rna_tissue.tsv') as f:
        header = f.readline().decode().strip().split('\t')
        # Identify column indices per tissue
        tissue_cols = defaultdict(list)
        for ci, col in enumerate(header):
            if col.startswith('TPM.'):
                tname = re.sub(r'\.\d+$', '', col[4:])
                tissue_cols[tname].append(ci)

        brain_cols = []
        for t in BRAIN_TISSUES:
            brain_cols.extend(tissue_cols.get(t, []))
        comp_cols  = {t: tissue_cols.get(t, []) for t in COMP_TISSUES}

        for line in f:
            row  = line.decode().strip().split('\t')
            enst = row[1].split('.')[0] if len(row) > 1 else ''
            if not enst: continue
            brain_tpm = np.mean([float(row[ci]) for ci in brain_cols
                                 if ci < len(row) and row[ci] != '']) if brain_cols else 0.0
            comp_tpm  = {}
            for t, cols in comp_cols.items():
                vals = [float(row[ci]) for ci in cols if ci < len(row) and row[ci] != '']
                comp_tpm[t] = np.mean(vals) if vals else 0.0
            hpa[enst] = {'brain': brain_tpm, **comp_tpm}

print(f"  HPA loaded: {len(hpa)} transcripts")

# Compute DTU: within each gene, fraction in brain vs fraction in each comparison tissue
def compute_dtu(gene_enst_list):
    """Returns dtu_delta: brain_frac - mean(comp_fracs) for each isoform."""
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
        mean_comp = np.mean(comp_fracs)
        dtu[enst] = {
            'frac_brain': round(frac_b, 4),
            'frac_comp_mean': round(mean_comp, 4),
            'dtu_delta': round(frac_b - mean_comp, 4),
            'brain_tpm': round(tpm_b, 3),
        }
    return dtu

# ── Gene-level index ──────────────────────────────────────────────────────────
gene_isos = defaultdict(list)   # sym -> [(idx, enst, ensg)]
for i in range(N_TE):
    gene_isos[te_sym[i]].append((i, te_enst[i], te_ensg[i]))

# ── Functional switch detection ───────────────────────────────────────────────
print("\n  Detecting functional switches ...")
switches = []

for sym, iso_list in gene_isos.items():
    if len(iso_list) < 2: continue

    idxs  = [x[0] for x in iso_list]
    ensts = [x[1] for x in iso_list]
    gs    = score_matrix[idxs, :]   # (n_iso, N_GO)

    # DTU for this gene
    dtu = compute_dtu(ensts)

    for a_idx in range(len(iso_list)):
        for b_idx in range(a_idx+1, len(iso_list)):
            sa, sb = gs[a_idx], gs[b_idx]
            for gi in range(N_GO):
                gap   = abs(float(sa[gi] - sb[gi]))
                hi_i  = a_idx if sa[gi] >= sb[gi] else b_idx
                lo_i  = b_idx if sa[gi] >= sb[gi] else a_idx
                if gap < SCORE_GAP: continue
                hi_s  = max(float(sa[gi]), float(sb[gi]))
                lo_s  = min(float(sa[gi]), float(sb[gi]))
                ratio = hi_s / lo_s if lo_s > 1e-6 else float('inf')
                if ratio < SCORE_RATIO: continue

                hi_enst = ensts[hi_i]
                lo_enst = ensts[lo_i]
                dtu_hi  = dtu[hi_enst]['dtu_delta']
                frac_hi = dtu[hi_enst]['frac_brain']
                concordant = dtu_hi >= DTU_DELTA

                switches.append({
                    'gene': sym,
                    'go_term': GO_KEYS[gi],
                    'go_name': GO_NAMES[gi],
                    'top_iso': hi_enst,
                    'bot_iso': lo_enst,
                    'score_top': round(float(hi_s), 4),
                    'score_bot': round(float(lo_s), 4),
                    'score_gap': round(gap, 4),
                    'score_ratio': round(ratio, 2),
                    'dtu_top': round(dtu_hi, 4),
                    'frac_brain_top': round(frac_hi, 4),
                    'brain_tpm_top': dtu[hi_enst]['brain_tpm'],
                    'dtu_concordant': concordant,
                })

switches.sort(key=lambda x: -(x['score_gap'] + x['dtu_top']))
seen = set()
top_switches = []
for s in switches:
    k = (s['gene'], s['go_term'])
    if k not in seen:
        seen.add(k)
        top_switches.append(s)

dtu_concordant = [s for s in top_switches if s['dtu_concordant']]
print(f"  Switches (gap≥{SCORE_GAP}, ratio≥{SCORE_RATIO}×): {len(top_switches)}")
print(f"  DTU concordant (delta≥{DTU_DELTA}): {len(dtu_concordant)}")

# ── Cross-GO reversal detection ───────────────────────────────────────────────
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
            sa   = gs[a]; sb = gs[b]
            diff = sa - sb
            a_wins = [(gi, diff[gi])  for gi in range(N_GO) if diff[gi]  >=  REVERSAL_GAP]
            b_wins = [(gi, -diff[gi]) for gi in range(N_GO) if diff[gi]  <= -REVERSAL_GAP]
            if not a_wins or not b_wins: continue

            best_a = max(a_wins, key=lambda x: x[1])
            best_b = max(b_wins, key=lambda x: x[1])

            reversals.append({
                'gene': sym,
                'iso_a': ensts[a], 'iso_b': ensts[b],
                'go_A': GO_KEYS[best_a[0]], 'name_A': GO_NAMES[best_a[0]],
                'gap_A': round(float(best_a[1]), 4),
                'go_B': GO_KEYS[best_b[0]], 'name_B': GO_NAMES[best_b[0]],
                'gap_B': round(float(best_b[1]), 4),
                'reversal_strength': round(float(best_a[1]+best_b[1]), 4),
                'n_a_wins': len(a_wins), 'n_b_wins': len(b_wins),
                'a_wins_terms': [(GO_KEYS[gi], GO_NAMES[gi], round(float(d),3)) for gi,d in a_wins],
                'b_wins_terms': [(GO_KEYS[gi], GO_NAMES[gi], round(float(d),3)) for gi,d in b_wins],
                'dtu_a': dtu[ensts[a]]['dtu_delta'],
                'dtu_b': dtu[ensts[b]]['dtu_delta'],
                'score_a': {GO_KEYS[gi]: round(float(sa[gi]),4) for gi in range(N_GO)},
                'score_b': {GO_KEYS[gi]: round(float(sb[gi]),4) for gi in range(N_GO)},
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

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  TOP-30 BRAIN FUNCTIONAL SWITCHES (DTU concordant first)")
print("="*70)
print(f"  {'Gene':10s} {'GO':22s} {'gap':5s} {'ratio':5s} {'dtu':5s} {'top_iso'}")
print(f"  {'-'*75}")

# DTU-concordant first, then sorted by gap+dtu
all_top = sorted(top_switches,
                 key=lambda x: (not x['dtu_concordant'], -(x['score_gap']+x['dtu_top'])))
for s in all_top[:30]:
    mark = '✓' if s['dtu_concordant'] else ' '
    print(f"  {mark}{s['gene']:9s} {s['go_name'][:22]:22s} {s['score_gap']:5.3f} "
          f"{s['score_ratio']:5.1f}x {s['dtu_top']:+5.3f}  {s['top_iso']}")

# ── Cross-GO top reversals ────────────────────────────────────────────────────
print("\n" + "="*70)
print("  TOP-20 CROSS-GO FUNCTIONAL SWITCHES")
print("="*70)
print(f"  {'Gene':10s} {'A wins':24s} {'gapA':5s}  {'B wins':24s} {'gapB':5s}  total  dtu_a   dtu_b")
print(f"  {'-'*90}")
for r in top_reversals[:20]:
    print(f"  {r['gene']:10s} {r['name_A'][:24]:24s} {r['gap_A']:5.3f}  "
          f"{r['name_B'][:24]:24s} {r['gap_B']:5.3f}  "
          f"{r['reversal_strength']:5.3f}  {r['dtu_a']:+5.3f}  {r['dtu_b']:+5.3f}")

# ── Target gene deep-dive ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("  TARGET GENE PROFILES")
print("="*70)

for gene in TARGET_GENES:
    isos = gene_isos.get(gene, [])
    if not isos:
        print(f"\n  {gene}: not in brain test set")
        continue

    idxs  = [x[0] for x in isos]
    ensts = [x[1] for x in isos]
    gs    = score_matrix[idxs, :]
    dtu   = compute_dtu(ensts)

    print(f"\n  {gene}  (n_isos={len(isos)})")
    hdr = "  ".join(f"{n[:7]:7s}" for n in GO_NAMES)
    print(f"  {'ENST':18s}  {hdr}  brain_frac  dtu_delta")
    print("  " + "-"*(20+10*N_GO+25))

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
        print(f"    A_wins: {[(n,d) for _,n,d in best['a_wins_terms']]}")
        print(f"    B_wins: {[(n,d) for _,n,d in best['b_wins_terms']]}")
    else:
        print(f"\n  No cross-GO reversal (gap < {REVERSAL_GAP})")

# ── AUPRC summary ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  GO TERM AUPRC SUMMARY")
print("="*70)
for gi, (go_term, go_name) in enumerate(GO_TERMS.items()):
    print(f"  {go_name[:28]:28s}  AUPRC={auprc_row[gi]:.4f}")
print(f"  {'MACRO':28s}  AUPRC={np.mean(auprc_row):.4f}")

# ── Save full results ─────────────────────────────────────────────────────────
result = {
    'timestamp': ts,
    'n_go_terms': N_GO,
    'go_terms': GO_TERMS,
    'auprc_per_go': {GO_KEYS[i]: round(auprc_row[i], 4) for i in range(N_GO)},
    'macro_auprc': round(float(np.mean(auprc_row)), 4),
    'n_test_isoforms': N_TE,
    'n_switches': len(top_switches),
    'n_dtu_concordant': len(dtu_concordant),
    'n_reversal_genes': len(top_reversals),
    'score_gap_threshold': SCORE_GAP,
    'score_ratio_threshold': SCORE_RATIO,
    'dtu_delta_threshold': DTU_DELTA,
    'reversal_gap_threshold': REVERSAL_GAP,
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
             'scores': {GO_KEYS[gi]: round(float(gs[i,gi]),4) for gi in range(N_GO)},
             'max_score': round(float(gs[i].max()),4),
             'max_go': GO_KEYS[int(gs[i].argmax())],
             'max_go_name': GO_NAMES[int(gs[i].argmax())],
             'dtu_delta': dtu[ensts[i]]['dtu_delta'],
             'frac_brain': dtu[ensts[i]]['frac_brain'],
             'brain_tpm': dtu[ensts[i]]['brain_tpm'],
            }
            for i in sorted(range(len(isos)), key=lambda k: -gs[k].max())
        ],
        'reversals': [r for r in reversals if r['gene'] == gene],
        'switches': [s for s in top_switches if s['gene'] == gene],
    }

# TSV output
tsv_path = f'{OUT_DIR}/brain_switches_{ts}.tsv'
with open(tsv_path, 'w') as f:
    cols = ['gene','go_name','score_top','score_bot','score_gap','score_ratio',
            'dtu_top','frac_brain_top','brain_tpm_top','dtu_concordant','top_iso','bot_iso']
    f.write('\t'.join(cols) + '\n')
    for s in top_switches:
        f.write('\t'.join(str(s.get(c,'')) for c in cols) + '\n')

json.dump(result, open(f'{OUT_DIR}/brain_switch_{ts}.json', 'w'), indent=2)
np.save(f'{OUT_DIR}/score_matrix_brain_{ts}.npy', score_matrix)

print(f"\n  [Saved] {OUT_DIR}/brain_switch_{ts}.json")
print(f"  [Saved] {OUT_DIR}/brain_switches_{ts}.tsv")
print(f"  [Saved] {OUT_DIR}/score_matrix_brain_{ts}.npy")
print("\nALL DONE")
