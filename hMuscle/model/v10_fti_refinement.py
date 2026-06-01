"""
v10_fti_refinement.py — FTI 프레임워크 고도화
=============================================
원래 TCS의 결함: human positive 유전자의 조직 특이성 측정
              → 정작 오염원인 SwissProt positive들의 공간을 무시

핵심 추가 지표:
  SED (SP Embedding Divergence):
    cosine_dist(centroid_SP+(GO), centroid_human+(GO)) in ESM-2 space
    → SP positive들이 human positive들로부터 얼마나 멀리 있는가
    → GO:0007204 비근육 GPCR 오염을 직접 포착

  SP_SMSI (SwissProt Human Orthologue SMSI):
    HUMAN 엔트리 SwissProt positive들의 골격근 발현량 평균
    → SP에서 HUMAN 단백질들이 근육에서 발현되는가

분석:
  1. SED + TBS + TCS → Δ AUPRC 회귀 (n=13)
  2. FTI_5 (5 GO term FTI 있는 것) 별도 검증
  3. 어떤 지표 조합이 가장 높은 R²를 주는가

결과: reports/fti_refinement/
"""

import os, sys, json, zipfile, collections
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
HPA_ZIP   = '/tmp/hpa_tissue2.zip'
OUT_DIR   = '../../reports/fti_refinement'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 13 GO terms ──────────────────────────────────────────────────────────────
GO_TERMS = [
    'GO:0007204', 'GO:0030017', 'GO:0006941', 'GO:0006914', 'GO:0043161',
    'GO:0032006', 'GO:0007519', 'GO:0042692', 'GO:0055074', 'GO:0007005',
    'GO:0007517', 'GO:0003774', 'GO:0006096',
]
GO_LABELS = {
    'GO:0007204': 'Ca²⁺ signaling',    'GO:0030017': 'Sarcomere org',
    'GO:0006941': 'Muscle contraction','GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',    'GO:0032006': 'TOR signaling',
    'GO:0007519': 'Skeletal muscle dev','GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca²⁺ homeostasis',  'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',  'GO:0003774': 'Motor activity',
    'GO:0006096': 'Glycolysis',
}

# F45 + F37 Δ AUPRC (5-seed mean)
DELTA_AUPRC = {
    'GO:0007204': {'delta':  0.350, 'v10b': 0.765, 'lr': 0.415, 'type': 'B', 'sig': '***'},
    'GO:0030017': {'delta':  0.179, 'v10b': 0.743, 'lr': 0.564, 'type': 'B', 'sig': '***'},
    'GO:0006941': {'delta':  0.287, 'v10b': 0.597, 'lr': 0.310, 'type': 'B', 'sig': '***'},
    'GO:0006914': {'delta':  0.354, 'v10b': 0.640, 'lr': 0.285, 'type': 'B', 'sig': '***'},
    'GO:0043161': {'delta':  0.356, 'v10b': 0.717, 'lr': 0.362, 'type': 'B', 'sig': '***'},
    'GO:0032006': {'delta':  0.092, 'v10b': 0.602, 'lr': 0.510, 'type': 'B', 'sig': 'n.s.'},
    'GO:0007519': {'delta':  0.138, 'v10b': 0.725, 'lr': 0.587, 'type': 'B', 'sig': '*'},
    'GO:0042692': {'delta':  0.421, 'v10b': 0.653, 'lr': 0.232, 'type': 'B', 'sig': '***'},
    'GO:0055074': {'delta':  0.475, 'v10b': 0.726, 'lr': 0.251, 'type': 'B', 'sig': '***'},
    'GO:0007005': {'delta':  0.424, 'v10b': 0.662, 'lr': 0.238, 'type': 'B', 'sig': '***'},
    'GO:0007517': {'delta':  0.465, 'v10b': 0.702, 'lr': 0.237, 'type': 'B', 'sig': '***'},
    'GO:0003774': {'delta': -0.013, 'v10b': 0.813, 'lr': 0.825, 'type': 'A', 'sig': 'n.s.'},
    'GO:0006096': {'delta': -0.023, 'v10b': 0.671, 'lr': 0.695, 'type': 'A', 'sig': 'n.s.'},
}

# Known FTI (256d, 5 GO terms only)
FTI_256_KNOWN = {
    'GO:0007204': 0.1947 / 0.3109,  # 0.626
    'GO:0030017': 0.1366 / 0.2836,  # 0.482
    'GO:0006941': 0.1567 / 0.1540,  # 1.018
    'GO:0003774': 0.5982 / 0.5830,  # 1.026
    'GO:0006096': 0.8331 / 0.4445,  # 1.875
}

# ─── Load embeddings ──────────────────────────────────────────────────────────
print("=" * 65)
print("  Loading ESM-2 embeddings...")
print("=" * 65)

def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

# SP train embeddings + IDs
X_sp  = np.load(f'{DATA_DIR}/esm2_train_swissprot_t30_150M.npy').astype(np.float32)
sp_ids = load_ids(f'{ID_DIR}/train_swissprot_list.npy')
print(f"  SP embeddings:    {X_sp.shape}  IDs: {len(sp_ids)}")

# Human test embeddings + IDs
X_te  = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
te_gene_ids = load_ids('my_gene_list_fixed.npy')
print(f"  Test embeddings:  {X_te.shape}  IDs: {len(te_gene_ids)}")

# Human train embeddings + IDs
X_tr  = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
tr_genes = load_ids(f'{ID_DIR}/train_gene_list.npy')
print(f"  Human train emb:  {X_tr.shape}  IDs: {len(tr_genes)}")

# Gene symbol mapping
ensg2sym = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ensg2sym[p[0]] = p[4]
te_syms = [ensg2sym.get(g.split('.')[0], g.split('.')[0]) for g in te_gene_ids]

# SP ID → gene symbol (format: GENE_SPECIES, human = GENE_HUMAN)
def sp_to_gene(sp_id):
    parts = sp_id.split('_')
    if len(parts) >= 2:
        return parts[0]
    return sp_id

# ─── Parse annotations ────────────────────────────────────────────────────────
print("\n  Parsing annotations...")
sp_prot_go = {}
with open(f'{ANNOT_DIR}/swissprot_annotations.txt') as f:
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2:
            sp_prot_go[p[0]] = set(p[1:])

human_prot_go = {}
with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2:
            human_prot_go[p[0]] = set(p[1:])

print(f"  SP proteins: {len(sp_prot_go):,}, Human proteins: {len(human_prot_go):,}")

# SP ID → index mapping
sp_id_to_idx = {sid: i for i, sid in enumerate(sp_ids)}

# ─── Feature 1: SED (SP Embedding Divergence) ─────────────────────────────────
print("\n" + "=" * 65)
print("  Feature 1: SED — SP Embedding Divergence")
print("=" * 65)

def cosine_dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(1 - np.dot(a, b))

sed_results = {}
for go in GO_TERMS:
    # SP positives → embeddings
    sp_pos_ids = [sid for sid in sp_ids if sid in sp_prot_go and go in sp_prot_go[sid]]
    sp_pos_idx = [sp_id_to_idx[sid] for sid in sp_pos_ids if sid in sp_id_to_idx]

    # Human test positives → embeddings
    human_pos_mask = np.array([1 if s in human_prot_go and go in human_prot_go.get(s, set())
                                else 0 for s in te_syms], dtype=bool)
    n_sp_pos  = len(sp_pos_idx)
    n_hum_pos = human_pos_mask.sum()

    if n_sp_pos == 0 or n_hum_pos == 0:
        sed_results[go] = {'sed': np.nan, 'n_sp': n_sp_pos, 'n_human': n_hum_pos}
        print(f"  {go}: SKIP (SP+={n_sp_pos}, human+={n_hum_pos})")
        continue

    sp_centroid    = X_sp[sp_pos_idx].mean(axis=0)
    human_centroid = X_te[human_pos_mask].mean(axis=0)
    sed = cosine_dist(sp_centroid, human_centroid)

    # Also compute within-SP intra-class spread (as reference)
    if n_sp_pos > 50:
        rng = np.random.RandomState(42)
        sub = rng.choice(n_sp_pos, 50, replace=False)
        sp_sub = X_sp[[sp_pos_idx[i] for i in sub]]
    else:
        sp_sub = X_sp[sp_pos_idx]
    sp_norm = sp_sub / (np.linalg.norm(sp_sub, axis=1, keepdims=True) + 1e-10)
    intra_cos = (sp_norm @ sp_norm.T)
    mask_upper = np.triu(np.ones(intra_cos.shape, bool), k=1)
    sp_intra_dist = float(1 - intra_cos[mask_upper].mean())

    sed_results[go] = {
        'sed': sed,
        'sp_intra_dist': sp_intra_dist,
        'sed_normalized': sed / (sp_intra_dist + 1e-10),
        'n_sp': n_sp_pos,
        'n_human': n_hum_pos,
    }
    print(f"  {go} ({GO_LABELS[go]:<22}): SED={sed:.4f}  "
          f"SP_intra={sp_intra_dist:.4f}  "
          f"SED/intra={sed/(sp_intra_dist+1e-10):.3f}  "
          f"n_SP+={n_sp_pos}")

# ─── Feature 2: SP-SMSI (SwissProt HUMAN entry SMSI) ─────────────────────────
print("\n" + "=" * 65)
print("  Feature 2: SP-SMSI — HUMAN ortholog SMSI in SwissProt positives")
print("=" * 65)

# Load HPA tissue expression
gene_expr = collections.defaultdict(dict)
tissues_found = set()
with zipfile.ZipFile(HPA_ZIP) as z:
    fname = [n for n in z.namelist() if n.endswith('.tsv')][0]
    with z.open(fname) as f:
        f.readline()
        for line in f:
            p = line.decode('utf-8').strip().split('\t')
            if len(p) < 4:
                continue
            gene_name, tissue = p[1], p[2]
            try:
                ntpm = float(p[3])
            except ValueError:
                continue
            gene_expr[gene_name][tissue] = ntpm
            tissues_found.add(tissue)

tissues_list = sorted(tissues_found)
muscle_tissues = [t for t in tissues_list if 'skeletal' in t.lower()]
print(f"  Muscle tissues: {muscle_tissues}")

def smsi_score(expr_dict, muscle_ts):
    muscle_expr = np.mean([expr_dict.get(t, 0.0) for t in muscle_ts])
    all_expr = np.mean(list(expr_dict.values())) if expr_dict else 0.0
    return float(muscle_expr / (all_expr + 1e-10))

sp_smsi_results = {}
for go in GO_TERMS:
    # SP HUMAN entries only
    sp_human_pos = [sid for sid in sp_ids
                    if sid.endswith('_HUMAN') and sid in sp_prot_go
                    and go in sp_prot_go[sid]]
    sp_human_genes = [sp_to_gene(sid) for sid in sp_human_pos]

    smsi_vals = []
    for gene in sp_human_genes:
        if gene in gene_expr:
            smsi_vals.append(smsi_score(gene_expr[gene], muscle_tissues))

    if not smsi_vals:
        sp_smsi_results[go] = {'sp_smsi': np.nan, 'n_sp_human': 0, 'n_hpa_matched': 0}
        print(f"  {go}: no HUMAN SP entries or HPA match")
        continue

    sp_smsi_mean = float(np.mean(smsi_vals))
    frac_muscle  = float(np.mean([v >= 1.0 for v in smsi_vals]))

    sp_smsi_results[go] = {
        'sp_smsi': sp_smsi_mean,
        'sp_smsi_frac_muscle': frac_muscle,
        'n_sp_human': len(sp_human_pos),
        'n_hpa_matched': len(smsi_vals),
    }
    print(f"  {go} ({GO_LABELS[go]:<22}): SP-SMSI={sp_smsi_mean:.3f}  "
          f"frac_muscle={frac_muscle:.2%}  "
          f"n_HUMAN_SP={len(sp_human_pos)}  matched={len(smsi_vals)}")

# ─── Load TBS/TCS from previous run ──────────────────────────────────────────
print("\n" + "=" * 65)
print("  Loading TBS/TCS from previous analysis...")
print("=" * 65)

# Recompute TBS quickly (copy logic from v10_tbs_tcs_13terms.py)
KINGDOMS = ['Bacteria', 'Archaea', 'Fungi', 'Viridiplantae', 'Invertebrata', 'Vertebrata']
SPECIES_TO_KINGDOM = {}
for sp_code in ['MOUSE','RAT','BOVIN','PIG','SHEEP','HORSE','FELCA','CANFA','RABIT','CHICK',
                'XENLA','XENTR','DANRE','ORYLA','MACMU','PANTR','GORGO','PONAB','HUMAN']:
    SPECIES_TO_KINGDOM[sp_code] = 'Vertebrata'
for sp_code in ['DROME','CAEEL','CAEBR','SCHMA','TRICA','BOMMO','AEDAE','ANOGA','DAPPU']:
    SPECIES_TO_KINGDOM[sp_code] = 'Invertebrata'
for sp_code in ['YEAST','SCHPO','CANAL','ASPFU','ASPNI','NEUCR']:
    SPECIES_TO_KINGDOM[sp_code] = 'Fungi'
for sp_code in ['ARATH','ORYSJ','MAIZE','SOLTU','SOYBN','MEDTR']:
    SPECIES_TO_KINGDOM[sp_code] = 'Viridiplantae'
for sp_code in ['ECOLI','BACSU','STAAU','SALTY','MYCTU','PSEAE']:
    SPECIES_TO_KINGDOM[sp_code] = 'Bacteria'
for sp_code in ['METJA','PYRAE','SULSO','ARCFU']:
    SPECIES_TO_KINGDOM[sp_code] = 'Archaea'

tbs_dict = {}
for go in GO_TERMS:
    sp_pos = [sid for sid in sp_ids if sid in sp_prot_go and go in sp_prot_go[sid]]
    kingdoms = set()
    for sid in sp_pos:
        k = SPECIES_TO_KINGDOM.get(sid.split('_')[-1], 'Unknown')
        if k != 'Unknown':
            kingdoms.add(k)
    tbs_dict[go] = len(kingdoms) / len(KINGDOMS)

# Load TCS from previous json
import glob
tcs_json_files = sorted(glob.glob('../../reports/tbs_tcs_13terms/tbs_tcs_results_*.json'))
tcs_dict = {}
if tcs_json_files:
    with open(tcs_json_files[-1]) as f:
        prev = json.load(f)
    for row in prev.get('per_go', []):
        tcs_dict[row['go']] = {'tcs': row['tcs'], 'smsi': row['smsi']}
    print(f"  Loaded TCS from {tcs_json_files[-1]}")

# ─── Combined regression ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Combined Regression Analysis")
print("=" * 65)

rows = []
for go in GO_TERMS:
    d     = DELTA_AUPRC[go]
    tbs   = tbs_dict.get(go, np.nan)
    tcs   = tcs_dict.get(go, {}).get('tcs', np.nan)
    smsi  = tcs_dict.get(go, {}).get('smsi', np.nan)
    sed   = sed_results.get(go, {}).get('sed', np.nan)
    sed_n = sed_results.get(go, {}).get('sed_normalized', np.nan)
    sp_smsi = sp_smsi_results.get(go, {}).get('sp_smsi', np.nan)

    if any(np.isnan(v) for v in [tbs, tcs, sed]):
        print(f"  SKIP {go}: NaN in features")
        continue

    rows.append({
        'go': go, 'label': GO_LABELS[go], 'type': d['type'],
        'tbs': tbs, 'tcs': tcs, 'smsi': smsi,
        'sed': sed, 'sed_normalized': sed_n,
        'sp_smsi': sp_smsi,
        'delta': d['delta'], 'v10b': d['v10b'], 'lr': d['lr'],
        'sig': d['sig'],
        'fti': FTI_256_KNOWN.get(go),
    })

n = len(rows)
print(f"\n  n={n} GO terms with complete features")

arr = {k: np.array([r[k] for r in rows]) for k in
       ['tbs','tcs','smsi','sed','sed_normalized','sp_smsi','delta']}

def pearsonr_safe(x, y):
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 3:
        return np.nan, np.nan
    return stats.pearsonr(x[valid], y[valid])

def ols_r2(X_cols, y):
    valid = ~np.isnan(y)
    for c in X_cols:
        valid &= ~np.isnan(c)
    if valid.sum() < len(X_cols) + 2:
        return np.nan, None
    Xm = np.column_stack([np.ones(valid.sum())] + [c[valid] for c in X_cols])
    yv = y[valid]
    coeffs, _, _, _ = np.linalg.lstsq(Xm, yv, rcond=None)
    y_hat = Xm @ coeffs
    ss_res = np.sum((yv - y_hat) ** 2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    return (1 - ss_res/ss_tot if ss_tot > 0 else np.nan), coeffs

# Individual correlations
print(f"\n  {'Feature':<20} {'r':>7} {'p':>8} {'n':>4}")
print("  " + "-"*44)
for feat, name in [('tbs','TBS'), ('tcs','TCS (human Tau)'),
                   ('sed','SED'), ('sed_normalized','SED_normalized'),
                   ('sp_smsi','SP-SMSI'), ('smsi','SMSI')]:
    x = arr[feat]
    r, p = pearsonr_safe(x, arr['delta'])
    n_valid = np.sum(~(np.isnan(x)|np.isnan(arr['delta'])))
    if not np.isnan(r):
        sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ''))
        print(f"  {name:<20} {r:>+7.3f} {p:>8.4f} {n_valid:>4}  {sig}")
    else:
        print(f"  {name:<20} {'nan':>7} {'nan':>8} {n_valid:>4}")

# Multi-feature regressions
print(f"\n  Regression models (outcome = Δ AUPRC):")
print(f"  {'Model':<40} {'R²':>6} {'df':>4}")
print("  " + "-"*55)

model_specs = [
    ("TBS only",              [arr['tbs']]),
    ("TCS only",              [arr['tcs']]),
    ("SED only",              [arr['sed']]),
    ("SP-SMSI only",          [arr['sp_smsi']]),
    ("TBS + TCS",             [arr['tbs'], arr['tcs']]),
    ("TBS + SED",             [arr['tbs'], arr['sed']]),
    ("TCS + SED",             [arr['tcs'], arr['sed']]),
    ("SED + SP-SMSI",         [arr['sed'], arr['sp_smsi']]),
    ("TBS + TCS + SED",       [arr['tbs'], arr['tcs'], arr['sed']]),
    ("TBS + SED + SP-SMSI",   [arr['tbs'], arr['sed'], arr['sp_smsi']]),
    ("TBS+TCS+TBS×TCS (orig)",[arr['tbs'], arr['tcs'], arr['tbs']*arr['tcs']]),
    ("SED+TBS+SED×TBS",       [arr['sed'], arr['tbs'], arr['sed']*arr['tbs']]),
    ("SED+TCS+SED×TCS",       [arr['sed'], arr['tcs'], arr['sed']*arr['tcs']]),
    ("Full: TBS+TCS+SED+SP-SMSI",[arr['tbs'],arr['tcs'],arr['sed'],arr['sp_smsi']]),
]

best_r2, best_model = -999, ''
for name, cols in model_specs:
    valid = np.ones(n, dtype=bool)
    for c in cols:
        valid &= ~np.isnan(c)
    nv = valid.sum()
    r2, _ = ols_r2(cols, arr['delta'])
    df = nv - len(cols) - 1
    if not np.isnan(r2):
        marker = ' ←best' if r2 > best_r2 else ''
        print(f"  {name:<40} {r2:>6.3f} {df:>4}  {marker}")
        if r2 > best_r2:
            best_r2, best_model = r2, name

print(f"\n  Best model: {best_model}  R²={best_r2:.3f}")

# Also correlate with FTI (n=5 only)
print(f"\n  FTI validation (n=5):")
fti_rows = [(r['fti'], r['sed'], r['tbs'], r['tcs'], r['sp_smsi'])
            for r in rows if r['fti'] is not None]
if len(fti_rows) >= 3:
    fti_arr = np.array([x[0] for x in fti_rows])
    for feat_idx, feat_name in [(1,'SED'),(2,'TBS'),(3,'TCS'),(4,'SP-SMSI')]:
        feat_arr = np.array([x[feat_idx] for x in fti_rows])
        r, p = pearsonr_safe(feat_arr, fti_arr)
        print(f"    r(FTI, {feat_name}) = {r:+.3f}  p={p:.3f}  n={len(fti_arr)}")

# ─── Figures ──────────────────────────────────────────────────────────────────
print(f"\n  Generating figures...")
type_c = {'A': '#d62728', 'B': '#1f77b4'}

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('FTI Framework Refinement: SED + SP-SMSI as New Features\n'
             'n=13 GO terms | Outcome: Δ AUPRC (v10-B − LR)', fontsize=12, fontweight='bold')

panels = [
    ('tbs',          'TBS (Taxonomic Breadth)',          axes[0][0]),
    ('tcs',          'TCS (Human Tau, original)',        axes[0][1]),
    ('sed',          'SED (SP Embedding Divergence)',    axes[0][2]),
    ('sp_smsi',      'SP-SMSI (SwissProt HUMAN SMSI)',   axes[1][0]),
    ('sed_normalized','SED / SP_intra_dist',             axes[1][1]),
]

patches = [mpatches.Patch(color=c, label=f'Type-{t}') for t, c in type_c.items()]

for feat, xlabel, ax in panels:
    x = arr[feat]
    y = arr['delta']
    r, p = pearsonr_safe(x, y)
    sig_str = '***' if not np.isnan(p) and p < 0.001 else ('**' if not np.isnan(p) and p<0.01
              else ('*' if not np.isnan(p) and p<0.05 else 'n.s.'))

    valid = ~(np.isnan(x)|np.isnan(y))
    for r2_i, row in enumerate(rows):
        if np.isnan(row[feat]) if feat in row else True:
            continue
        ax.scatter(row[feat], row['delta'], s=120, c=type_c[row['type']],
                   alpha=0.85, edgecolors='black', linewidth=0.8, zorder=5)
        ax.annotate(row['go'].split(':')[1], xy=(row[feat], row['delta']),
                    xytext=(4, 3), textcoords='offset points', fontsize=7)

    if valid.sum() >= 3:
        m, b = np.polyfit(x[valid], y[valid], 1)
        xs = np.linspace(x[valid].min()-0.02, x[valid].max()+0.02, 100)
        ax.plot(xs, m*xs+b, 'k--', alpha=0.5, lw=1.2)

    ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('Δ AUPRC (v10-B − LR)', fontsize=9)
    r_str = f'{r:+.3f}' if not np.isnan(r) else 'nan'
    p_str = f'{p:.3f}' if not np.isnan(p) else 'nan'
    ax.set_title(f'{xlabel}\nr={r_str}  p={p_str}  {sig_str}', fontsize=9)
    ax.legend(handles=patches, fontsize=8, loc='best')

# Panel 6: 2D SED × TBS
ax = axes[1][2]
delta_vals = arr['delta']
vmax = max(abs(delta_vals[~np.isnan(delta_vals)].min()),
           abs(delta_vals[~np.isnan(delta_vals)].max()))
norm = plt.Normalize(vmin=-vmax, vmax=vmax)
cmap = plt.cm.RdBu_r

for row in rows:
    if np.isnan(row['sed']): continue
    color = cmap(norm(row['delta']))
    sz = 80 + abs(row['delta']) * 400
    ax.scatter(row['sed'], row['tbs'], s=sz, c=[color], alpha=0.85,
               edgecolors='red' if row['type']=='A' else 'black',
               linewidth=1.5 if row['type']=='A' else 0.8, zorder=5)
    ax.annotate(row['go'].split(':')[1], xy=(row['sed'], row['tbs']),
                xytext=(4, 3), textcoords='offset points', fontsize=7)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Δ AUPRC')
ax.set_xlabel('SED (SP Embedding Divergence)', fontsize=9)
ax.set_ylabel('TBS (Taxonomic Breadth)', fontsize=9)
r2_sed_tbs, _ = ols_r2([arr['sed'], arr['tbs']], delta_vals)
ax.set_title(f'2D: SED × TBS → Δ AUPRC\nR²(SED+TBS)={r2_sed_tbs:.3f}', fontsize=9)

plt.tight_layout()
fig_path = f'{OUT_DIR}/fti_refinement.pdf'
plt.savefig(fig_path, bbox_inches='tight')
plt.savefig(fig_path.replace('.pdf','.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")

# ─── Summary table ────────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print(f"  FINAL SUMMARY TABLE")
print(f"{'='*85}")
print(f"{'GO':<14} {'Label':<22} {'T'} {'TBS':>5} {'TCS':>6} {'SED':>6} "
      f"{'SP-SMSI':>8} {'Δ':>6} {'FTI':>5}")
print("-"*85)
for row in rows:
    fti_str = f"{row['fti']:.3f}" if row['fti'] else '  —  '
    sp_str  = f"{row['sp_smsi']:.3f}" if row['sp_smsi'] and not np.isnan(row['sp_smsi']) else '  —  '
    print(f"{row['go']:<14} {row['label']:<22} {row['type']}  "
          f"{row['tbs']:5.3f} {row['tcs']:6.4f} {row['sed']:6.4f} "
          f"{sp_str:>8} {row['delta']:+6.3f} {fti_str:>5}")

# ─── Save results ─────────────────────────────────────────────────────────────
import time
ts = time.strftime('%Y%m%d_%H%M')
out_json = f'{OUT_DIR}/fti_refinement_{ts}.json'
with open(out_json, 'w') as f:
    json.dump({
        'n': n,
        'features': ['tbs','tcs','sed','sp_smsi'],
        'pearson': {
            feat: {'r': float(pearsonr_safe(arr[feat], arr['delta'])[0]),
                   'p': float(pearsonr_safe(arr[feat], arr['delta'])[1])}
            for feat in ['tbs','tcs','sed','sp_smsi','smsi']
            if not all(np.isnan(arr[feat]))
        },
        'regression_r2': {name: float(ols_r2(cols, arr['delta'])[0])
                          for name, cols in model_specs
                          if not np.isnan(ols_r2(cols, arr['delta'])[0])},
        'best_model': best_model,
        'best_r2': float(best_r2),
        'per_go': rows,
        'timestamp': ts,
    }, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
print(f"\n  [Saved] {out_json}")
print("Done.")
