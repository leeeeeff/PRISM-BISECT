# -*- coding: utf-8 -*-
"""
v15_functional_switch_dtu.py
=============================
모델: v10-B (ESM-2 MLP, 5-seed ensemble)
  선정 근거: Macro AUPRC=0.6935 (gene-level 수용 가능) + pos_bias_macro=1.006
  (isoform-level 구별력 존재, coding-only Δ=-0.022 — artifact 아님)

분석:
  1. 13 GO term 전체 → per-isoform prediction score (5-seed mean)
  2. Functional switch 후보: 같은 유전자 내 top/bottom isoform score ratio 높은 경우
  3. DTU (Differential Transcript Usage): HPA skeletal muscle (5 samples) vs
     타 조직 평균 → 모델 top-iso가 근육에서 실제로 많이 발현되는지 검증
  4. 일치(concordance): 모델 top-iso == skeletal muscle dominant iso
  5. 문헌 검증: TPM1/DMD/ANK2/GABARAPL1/PINK1 등 known cases 확인

실행:
  conda run -n isoform_env python v15_functional_switch_dtu.py
"""

import os, sys, json, zipfile, csv, time
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
FEAT_DIR  = '../results_isoform/features'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
HPA_ZIP   = f'{FEAT_DIR}/hpa_isoform/transcript_rna_tissue.tsv.zip'
OUT_DIR   = '../../reports/v15_switch_dtu'
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = 5
SCORE_GAP_THRESH = 0.30   # top - bottom 최소 차이
SCORE_RATIO_THRESH = 3.0  # top / bottom 최소 비율
DTU_DELTA_THRESH = 0.10   # skeletal muscle fraction delta 최소값 (concordance 강도)

GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0030017': 'Sarcomere org',
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

KNOWN_SWITCHES = {
    'TPM1':      'Sarcomere-integrating vs non-integrating isoforms (Cardiovasc.Res.)',
    'DMD':       'Dp427m(muscle,high) vs Dp71(brain,low) — textbook isoform switch',
    'ANK2':      'AnkB-212 cardiac M-line specific (Camors 2015)',
    'GABARAPL1': 'ATG8/GABARAP family, mitophagy — ratio=2222x confirmed',
    'PINK1':     'Parkin-mediated mitophagy kinase — ratio=20x, cross-GO',
    'BNIP3':     'Hypoxia mitophagy receptor (muscle atrophy)',
    'SOD2':      'Mitochondrial antioxidant (sarcopenia marker) — ratio=1632x',
    'ATG12':     'Ubiquitin-like ATG conjugation — ratio=2.5x',
}

# ── Load features ───────────────────────────────────────────────────────────
print("="*65)
print("  v15 Isoform Functional Switch + DTU Analysis")
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
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

tr_sym       = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_geneid]
te_sym       = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_geneid]
te_ensg_base = [g.split('.')[0] for g in te_geneid]
te_enst_base = [x.split('.')[0] for x in te_isoid]

print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")
print(f"  Test isoforms: {len(te_isoid)}")

# ── v10-B model ─────────────────────────────────────────────────────────────
def build_v10b():
    inp = layers.Input(shape=(640,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=inp, outputs=out)

def run_v10b_ensemble(y_tr, y_te):
    seed_preds = []
    for s in range(N_SEEDS):
        tf.random.set_seed(s * 137 + 42)
        np.random.seed(s * 137 + 42)
        # Shuffle training data per seed so validation_split is effectively random
        idx = np.random.permutation(len(X_tr))
        X_tr_s, y_tr_s = X_tr[idx], y_tr[idx]
        m = build_v10b()
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
        cb = [callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                      restore_best_weights=True, verbose=0),
              callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)]
        m.fit(X_tr_s, y_tr_s, epochs=80, batch_size=512,
              validation_split=0.1, callbacks=cb, verbose=0)
        seed_preds.append(m.predict(X_te, batch_size=1024, verbose=0).flatten())
    preds = np.mean(seed_preds, axis=0)
    auprc = average_precision_score(y_te, preds) if y_te.sum() > 0 else 0.0
    return preds, auprc

# ── Load GO labels ───────────────────────────────────────────────────────────
def load_go_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te

# ── HPA DTU pre-loading ──────────────────────────────────────────────────────
print("\n  Loading HPA transcript expression data ...")
t_hpa = time.time()

# 비교 조직: skeletal muscle과 기능적으로 다른 조직 선택
COMPARISON_TISSUES = [
    'adipose tissue', 'liver', 'kidney', 'lung',
    'cerebral cortex', 'colon', 'pancreas', 'thyroid gland'
]

skm_mean  = {}  # (ensg, enst) -> mean TPM in skeletal muscle
comp_mean = {}  # (ensg, enst) -> mean TPM in comparison tissues

with zipfile.ZipFile(HPA_ZIP) as z:
    fname = z.namelist()[0]
    with z.open(fname) as f:
        header = f.readline().decode().strip().split('\t')
        skm_idx = [i for i, h in enumerate(header)
                   if 'skeletal muscle' in h.lower() and h.startswith('TPM')]
        comp_idx = []
        for t in COMPARISON_TISSUES:
            cols = [i for i, h in enumerate(header)
                    if t in h.lower() and h.startswith('TPM')]
            comp_idx.extend(cols[:3])

        for line in f:
            parts = line.decode().strip().split('\t')
            if len(parts) < 3: continue
            ensg = parts[0]  # already base
            enst = parts[1]  # already base
            key  = (ensg, enst)
            try:
                sv = np.mean([float(parts[i]) for i in skm_idx if i < len(parts)])
                cv = np.mean([float(parts[i]) for i in comp_idx if i < len(parts)])
            except: continue
            skm_mean[key]  = sv
            comp_mean[key] = cv

print(f"  HPA loaded: {len(skm_mean)} (ENSG,ENST) entries  ({time.time()-t_hpa:.0f}s)")


def compute_dtu(ensg, enst_top, enst_bot):
    """DTU concordance: does skeletal muscle upregulate the model's top isoform?"""
    # Collect all isoforms of this gene in HPA
    gene_isos_skm  = {k[1]: v for k, v in skm_mean.items()  if k[0] == ensg}
    gene_isos_comp = {k[1]: v for k, v in comp_mean.items() if k[0] == ensg}

    if not gene_isos_skm:
        return {'has_hpa': False, 'concordance': None}

    total_skm  = sum(gene_isos_skm.values())  + 1e-9
    total_comp = sum(gene_isos_comp.values()) + 1e-9

    frac_skm_top  = gene_isos_skm.get(enst_top,  0) / total_skm
    frac_skm_bot  = gene_isos_skm.get(enst_bot,  0) / total_skm
    frac_comp_top = gene_isos_comp.get(enst_top, 0) / total_comp
    frac_comp_bot = gene_isos_comp.get(enst_bot, 0) / total_comp

    dtu_top = frac_skm_top - frac_comp_top   # +: muscle upregulates top iso
    dtu_bot = frac_skm_bot - frac_comp_bot   # -: muscle downregulates bot iso

    dom_skm = max(gene_isos_skm, key=gene_isos_skm.get) if gene_isos_skm else None
    concordance = (dom_skm == enst_top)

    return {
        'has_hpa':       True,
        'n_isos_hpa':    len(gene_isos_skm),
        'frac_skm_top':  round(frac_skm_top, 4),
        'frac_comp_top': round(frac_comp_top, 4),
        'dtu_top':       round(dtu_top, 4),
        'dtu_bot':       round(dtu_bot, 4),
        'dom_skm':       dom_skm,
        'concordance':   concordance,
        'tpm_top_skm':   round(gene_isos_skm.get(enst_top, 0), 3),
        'tpm_bot_skm':   round(gene_isos_skm.get(enst_bot, 0), 3),
        'tpm_top_comp':  round(gene_isos_comp.get(enst_top, 0), 3),
    }


# ── Main loop: all 13 GO terms ───────────────────────────────────────────────
all_switches = []
go_summaries = []
T_TOTAL = time.time()

for go_term, go_name in GO_TERMS.items():
    print(f"\n{'='*65}")
    print(f"  {go_term}  {go_name}")
    print(f"{'='*65}")

    y_tr, y_te = load_go_labels(go_term)
    n_pos_tr, n_pos_te = int(y_tr.sum()), int(y_te.sum())
    print(f"  Train pos={n_pos_tr}, Test pos={n_pos_te}")

    t0 = time.time()
    preds, auprc = run_v10b_ensemble(y_tr, y_te)
    print(f"  v10-B AUPRC={auprc:.4f}  ({time.time()-t0:.0f}s)")

    # ── Per-gene isoform score collection ───────────────────────────────
    pos_syms = {te_sym[i] for i in range(len(y_te)) if y_te[i] == 1}
    gene_data = defaultdict(list)
    for i in range(len(preds)):
        if te_sym[i] not in pos_syms: continue
        gene_data[te_sym[i]].append({
            'enst': te_enst_base[i],
            'ensg': te_ensg_base[i],
            'score': float(preds[i]),
        })

    # ── Switch detection ─────────────────────────────────────────────────
    term_switches = 0
    term_concordant = 0

    for sym, isos in gene_data.items():
        if len(isos) < 2: continue
        isos_s = sorted(isos, key=lambda x: x['score'], reverse=True)
        top = isos_s[0];  bot = isos_s[-1]

        gap   = top['score'] - bot['score']
        ratio = top['score'] / (bot['score'] + 1e-7)

        if gap < SCORE_GAP_THRESH or ratio < SCORE_RATIO_THRESH: continue

        ensg = top['ensg']
        dtu  = compute_dtu(ensg, top['enst'], bot['enst'])

        rec = {
            'go_term':   go_term,
            'go_name':   go_name,
            'gene':      sym,
            'ensg':      ensg,
            'n_isos_model': len(isos),
            'top_enst':  top['enst'],
            'top_score': round(top['score'], 4),
            'bot_enst':  bot['enst'],
            'bot_score': round(bot['score'], 4),
            'score_gap': round(gap, 4),
            'score_ratio': round(ratio, 1),
            'mid_scores': [round(x['score'], 3) for x in isos_s[1:-1]] if len(isos_s) > 2 else [],
            'is_known':  sym in KNOWN_SWITCHES,
            'known_evidence': KNOWN_SWITCHES.get(sym, ''),
        }
        rec.update(dtu)
        all_switches.append(rec)
        term_switches += 1
        if dtu.get('concordance'): term_concordant += 1

    print(f"  Switches (gap≥{SCORE_GAP_THRESH}, ratio≥{SCORE_RATIO_THRESH}x): {term_switches}  "
          f"DTU concordant: {term_concordant}")
    go_summaries.append({
        'go_term': go_term, 'go_name': go_name,
        'auprc': round(auprc, 4),
        'n_switches': term_switches,
        'n_concordant': term_concordant,
    })

print(f"\n\nTotal elapsed: {(time.time()-T_TOTAL)/60:.1f} min")

# ── Combined ranking ─────────────────────────────────────────────────────────
# Ranking criterion: score_ratio * DTU_boost
# DTU_boost: concordance + muscle upregulation bonus
for s in all_switches:
    if s.get('has_hpa') and s.get('concordance') and s.get('dtu_top', 0) > 0:
        dtu_boost = 1.0 + 4.0 * min(s['dtu_top'], 0.5)
    elif s.get('has_hpa') and s.get('concordance'):
        dtu_boost = 1.5
    elif s.get('is_known'):
        dtu_boost = 1.3
    else:
        dtu_boost = 1.0
    s['rank_score'] = s['score_ratio'] * dtu_boost

all_switches.sort(key=lambda x: x['rank_score'], reverse=True)

# ── Top 30 deduplicated by gene ──────────────────────────────────────────────
seen = set()
top30 = []
for s in all_switches:
    if s['gene'] not in seen:
        seen.add(s['gene'])
        top30.append(s)
    if len(top30) >= 30: break

# ── Known switch validation check ───────────────────────────────────────────
print("\n" + "="*65)
print("  KNOWN SWITCH VALIDATION")
print("="*65)
for known_sym in KNOWN_SWITCHES:
    hits = [s for s in all_switches if s['gene'] == known_sym]
    if hits:
        h = hits[0]
        dtu_str = f"DTU={h.get('dtu_top', 0):+.3f}  conc={'✓' if h.get('concordance') else '✗'}" if h.get('has_hpa') else "no HPA"
        print(f"  ✓ {known_sym:12s}  ratio={h['score_ratio']:6.0f}x  gap={h['score_gap']:.3f}  {dtu_str}")
        print(f"    {h['known_evidence']}")
    else:
        print(f"  ✗ {known_sym:12s}  NOT found (below threshold or no multiple isoforms in test set)")

# ── Full ranked table ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  TOP-30 FUNCTIONAL SWITCH CANDIDATES (deduplicated by gene)")
print("="*65)
print(f"  {'#':2s}  {'Gene':12s} {'GO':14s} {'Top':5s} {'Bot':5s} {'Ratio':6s} "
      f"{'Gap':5s} {'DTUtop':7s} {'Conc':4s} {'nHPA':4s} {'Kn'}")
print(f"  {'-'*85}")
for i, s in enumerate(top30):
    dtu_s   = f"{s.get('dtu_top', 0):+.3f}" if s.get('has_hpa') else '  N/A'
    conc    = '✓' if s.get('concordance') else ('✗' if s.get('has_hpa') else '-')
    n_hpa   = str(s.get('n_isos_hpa', '-'))
    known   = '★' if s['is_known'] else ''
    print(f"  {i+1:2d}  {s['gene']:12s} {s['go_name'][:14]:14s} "
          f"{s['top_score']:.3f} {s['bot_score']:.3f} "
          f"{s['score_ratio']:6.0f}x {s['score_gap']:.3f} "
          f"{dtu_s:7s} {conc:4s} {n_hpa:4s} {known}")

# ── DTU concordance summary ───────────────────────────────────────────────────
n_has_hpa    = sum(1 for s in all_switches if s.get('has_hpa'))
n_concordant = sum(1 for s in all_switches if s.get('concordance'))
n_strong_dtu = sum(1 for s in all_switches
                   if s.get('concordance') and s.get('dtu_top', 0) >= DTU_DELTA_THRESH)

print(f"\n  Total switches found:        {len(all_switches)}")
print(f"  With HPA expression data:    {n_has_hpa}")
print(f"  DTU concordant:              {n_concordant}")
print(f"  Strong DTU (delta≥{DTU_DELTA_THRESH}):     {n_strong_dtu}")
print(f"  Known switches recovered:    {sum(1 for s in all_switches if s['is_known'])}")

# ── Save ──────────────────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M')

tsv_path  = f'{OUT_DIR}/switch_candidates_{ts}.tsv'
json_path = f'{OUT_DIR}/switch_dtu_{ts}.json'

TSV_KEYS = [
    'go_term','go_name','gene','ensg','n_isos_model',
    'top_enst','top_score','bot_enst','bot_score',
    'score_gap','score_ratio','rank_score',
    'has_hpa','n_isos_hpa','frac_skm_top','frac_comp_top',
    'dtu_top','dtu_bot','concordance','tpm_top_skm','tpm_top_comp',
    'is_known','known_evidence'
]

with open(tsv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=TSV_KEYS, delimiter='\t', extrasaction='ignore')
    w.writeheader()
    w.writerows(all_switches)

result = {
    'timestamp': ts,
    'model': 'v10-B (5-seed ensemble)',
    'thresholds': {'score_gap': SCORE_GAP_THRESH, 'score_ratio': SCORE_RATIO_THRESH,
                   'dtu_delta': DTU_DELTA_THRESH},
    'summary': {
        'n_total_switches': len(all_switches),
        'n_has_hpa':        n_has_hpa,
        'n_concordant':     n_concordant,
        'n_strong_dtu':     n_strong_dtu,
        'n_known_recovered': sum(1 for s in all_switches if s['is_known']),
    },
    'go_summaries': go_summaries,
    'top30': top30,
}
json.dump(result, open(json_path, 'w'), indent=2, default=str)

print(f"\n  [Saved TSV]  {tsv_path}")
print(f"  [Saved JSON] {json_path}")
print(f"\nALL DONE")
