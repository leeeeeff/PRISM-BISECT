"""
v10_tbs_tcs_13terms.py — TBS/TCS 계산을 5 → 13 GO term으로 확장
================================================================
목적: SwissProt annotation 품질(TBS=종간 보존성, TCS=조직 순도)이
      v10-B vs LR 성능 차이(Δ AUPRC)를 예측하는가?

분석:
  1. TBS (Taxonomic Breadth Score): annotation 파일에서 kingdom 다양성 계산
  2. TCS (Tissue Context Specificity): HPA Tau index 기반 조직 특이성
  3. Δ AUPRC = v10-B_mean - LR_mean (F45 결과, 이미 확보)
  4. Pearson r (TBS vs Δ), (TCS vs Δ)
  5. Regression: Δ ~ TBS + TCS + TBS×TCS (n=13, df=9)
  6. Figure: 2D scatter (TBS x, TCS y, size/color = Δ AUPRC)

결과: reports/tbs_tcs_13terms/
"""

import os, sys, json, zipfile, collections
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─── Paths ────────────────────────────────────────────────────────────────────
ANNOT_SP     = '../data/raw_data/data/annotations/swissprot_annotations.txt'
ANNOT_HUMAN  = '../data/raw_data/data/annotations/human_annotations.txt'
HPA_ZIP      = '/tmp/hpa_tissue2.zip'
OUT_DIR      = '../../reports/tbs_tcs_13terms'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 13 GO terms ──────────────────────────────────────────────────────────────
GO_TERMS = [
    # Type-B (11개)
    'GO:0007204',  # Ca2+ signaling
    'GO:0030017',  # Sarcomere org
    'GO:0006941',  # Muscle contraction
    'GO:0006914',  # Autophagy
    'GO:0043161',  # Proteasome-UPS
    'GO:0032006',  # TOR signaling
    'GO:0007519',  # Skeletal muscle dev
    'GO:0042692',  # Muscle cell diff
    'GO:0055074',  # Ca2+ homeostasis
    'GO:0007005',  # Mitochondrion org
    'GO:0007517',  # Muscle organ dev
    # Type-A (2개)
    'GO:0003774',  # Motor activity
    'GO:0006096',  # Glycolysis
]

GO_LABELS = {
    'GO:0007204': 'Ca²⁺ signaling',
    'GO:0030017': 'Sarcomere org',
    'GO:0006941': 'Muscle contraction',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0032006': 'TOR signaling',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca²⁺ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0003774': 'Motor activity',
    'GO:0006096': 'Glycolysis',
}

# F45 + F37 결과 (5-seed mean AUPRC)
DELTA_AUPRC = {
    'GO:0007204': {'v10b': 0.765, 'lr': 0.415, 'delta': 0.350, 'sig': '***', 'type': 'B'},
    'GO:0030017': {'v10b': 0.743, 'lr': 0.564, 'delta': 0.179, 'sig': '***', 'type': 'B'},
    'GO:0006941': {'v10b': 0.597, 'lr': 0.310, 'delta': 0.287, 'sig': '***', 'type': 'B'},
    'GO:0006914': {'v10b': 0.640, 'lr': 0.285, 'delta': 0.354, 'sig': '***', 'type': 'B'},
    'GO:0043161': {'v10b': 0.717, 'lr': 0.362, 'delta': 0.356, 'sig': '***', 'type': 'B'},
    'GO:0032006': {'v10b': 0.602, 'lr': 0.510, 'delta': 0.092, 'sig': 'n.s.','type': 'B'},
    'GO:0007519': {'v10b': 0.725, 'lr': 0.587, 'delta': 0.138, 'sig': '*',   'type': 'B'},
    'GO:0042692': {'v10b': 0.653, 'lr': 0.232, 'delta': 0.421, 'sig': '***', 'type': 'B'},
    'GO:0055074': {'v10b': 0.726, 'lr': 0.251, 'delta': 0.475, 'sig': '***', 'type': 'B'},
    'GO:0007005': {'v10b': 0.662, 'lr': 0.238, 'delta': 0.424, 'sig': '***', 'type': 'B'},
    'GO:0007517': {'v10b': 0.702, 'lr': 0.237, 'delta': 0.465, 'sig': '***', 'type': 'B'},
    'GO:0003774': {'v10b': 0.813, 'lr': 0.825, 'delta': -0.013,'sig': 'n.s.','type': 'A'},
    'GO:0006096': {'v10b': 0.671, 'lr': 0.695, 'delta': -0.023,'sig': 'n.s.','type': 'A'},
}

# ─── Kingdom mapping (tbs_quantification.py에서 상속) ─────────────────────────
KINGDOMS = ['Bacteria', 'Archaea', 'Fungi', 'Viridiplantae', 'Invertebrata', 'Vertebrata']

SPECIES_TO_KINGDOM = {}
for sp in ['MOUSE','RAT','BOVIN','PIG','SHEEP','HORSE','FELCA','CANFA','RABIT','CHICK',
           'XENLA','XENTR','DANRE','ORYLA','TAKRU','FUGRU','MEDJA','MACMU','PANTR','GORGO',
           'PONAB','MACFA','CHLAE','PAPAN','NOMLE','AILME','LOXAF','TURTR','MYOLU','ICTTR',
           'ECHTE','SOREX','TUPGB','CAVPO','MESAU','CRIGR','SPECI','SPETR','OCHPR','DIPOR',
           'PEDPE','HETGL','NANPA','MICOC','MICMU','OTOGA','TARSY','CALJA','AOTNA','SAGOE',
           'COLMO','ATEGE','LAGGL','PROCO','PANHA','RHIFE','CERDI','MANDO','SARHA','MONDO',
           'ORNAN','CRILO','ONOVI','SALSA','ICTPU','PETMA','LAMGE','HUMAN','BUFBU','ANAPL',
           'HALAL','MELGA','TAEGU','GEFOR','ACRSC','OPHHA','PELSI','ANOCA','CROAM','ALIMA','TRICA']:
    SPECIES_TO_KINGDOM[sp] = 'Vertebrata'

for sp in ['DROME','DROMZ','DROPS','DROYA','DROVI','DROPE','DROAN','DROMI','CAEEL','CAEBR',
           'CAERE','HAECO','PRIPA','APLCA','APLKW','LOLFO','OCTVU','OCTBM','STRPU','ARCPU',
           'CIOIN','CIOSA','SCHMA','FASHE','PLAF7','PLAF4','PLACH','PLAAF','LOTGI','CRAGI',
           'AMQC7','HELRO','CAPTE','BOMMO','AEDAE','ANOGA','CULQU','PHYCI','APIS','APIME',
           'NASCO','ACYPI','IXOSC','RHIMP','PENMO','DAPPU','ARTSU','HYDVU','NEMVE','LUBLU']:
    SPECIES_TO_KINGDOM[sp] = 'Invertebrata'

for sp in ['YEAST','YEAS7','YEAS8','SCHPO','CANAL','CANPA','CANGA','NEUCR','PODE3','PODAZ',
           'ASPFU','ASPNI','ASPOR','ASPFL','ASPCL','ASPA1','ASPAC','PENCH','PENRW','PENOX',
           'PENBR','PENBI','PENDC','CRYNE','CRYNJ','CRYNH','USTMA','MYCMD','PHANO','COPCI',
           'LACBI','AGABI','MAGNP','MAGO7','GIBZE','FUSGR','FUSOX','TALATR','TALMT','COCIM',
           'HISCR','PARBA','BOTFU','SCLS1','VEDALB','RHISO','RHIA1','PHYBL','ENTHI','BATDE']:
    SPECIES_TO_KINGDOM[sp] = 'Fungi'

for sp in ['ARATH','ORYSJ','ORYSI','ORYBR','ORYOF','MAIZE','MAIZB','SOLTU','SOLLC','SOLPE',
           'TOBAC','NICTA','NICAT','NICSY','SOYBN','GLYMA','MEDTR','LOTJA','PHAVU','VICFA',
           'CICAR','CAJCA','POPTR','RICCO','JATCU','MANES','LUSAN','VITVI','GRAPE','SPIOL',
           'BETVU','CHEAM','ATHAL','WHEAT','ORYSA','HORVV','SECCE','TRIUA','BRAOL','BRACM',
           'BORPA','CAPAR','PHYPA','SELMO','CHLRE','VOLVP','OSTTA','MICPU','MARPO','CERPU',
           'PINSY','ABIES','PICAB']:
    SPECIES_TO_KINGDOM[sp] = 'Viridiplantae'

for sp in ['ECOLI','ECOLX','ECO57','BACSU','BACST','BACME','BACAN','BACCE','BACCR','BACHD',
           'STAAU','STAAN','STAEQ','STAEP','STAHY','STRCO','STRPU','STRPR','STRSV','STRMU',
           'STRPN','STRPY','MYCTU','MYCBO','MYCUL','MYCS2','SALTY','SALTI','SALPA','SALDC',
           'PSEAE','PSEPF','PSESM','HAEIN','HAES1','NEIGO','NEIMB','HELPJ','HELPY','HELPH',
           'CAMC5','CAMJE','CAMCOL','THEMA','THEM4','THETH','AQUAE','DEIRA','DEIRR','CHLTR',
           'CHLT2','BORBU','BORGA','BORHE','TREDE','TREPA','RICPR','RICAE','LACPL','LACLA',
           'LACP3','CLOPE','CLOAB','CLOBO','CLOTE','CLOD6','BIFLO','BIFAD','RHIME','RHILO',
           'AGRRT','MESSA','BRASO','SINFN','CAUVC','BDEBA','SHEON','SHEAM','SHEFN','VIBCH',
           'VIBVU','VIBPA','YERPE','YERPS','YEREN']:
    SPECIES_TO_KINGDOM[sp] = 'Bacteria'

for sp in ['METJA','METTH','METKA','METAC','METMA','METBU','METS5','PYRAE','PYRFU','PYRAB',
           'PYRHO','SULSO','SULAC','SULTO','SULSH','SULNO','SULT1','HALOAR','HALSA','HALVD',
           'HALMD','HALMS','ARCFU','THEAC','THEVO','PICTO','NAEGR','IGNH4','METAR','METHJ',
           'AERPE','CALTE','NANEQ','CORMM','NITMS']:
    SPECIES_TO_KINGDOM[sp] = 'Archaea'


def get_kingdom(sp_code):
    return SPECIES_TO_KINGDOM.get(sp_code.upper(), 'Unknown')


# ─── TBS 계산 ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Step 1: TBS — Taxonomic Breadth Score (13 GO terms)")
print("=" * 65)

print("  Parsing annotation files...")
sp_prot_go = {}
with open(ANNOT_SP) as f:
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2:
            sp_prot_go[p[0]] = set(p[1:])

human_prot_go = {}
with open(ANNOT_HUMAN) as f:
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2:
            human_prot_go[p[0]] = set(p[1:])

print(f"  SwissProt proteins: {len(sp_prot_go):,}, Human proteins: {len(human_prot_go):,}")

tbs_results = {}
for go in GO_TERMS:
    kingdoms_present = set()
    sp_pos = [p for p, gs in sp_prot_go.items() if go in gs]
    human_pos = [p for p, gs in human_prot_go.items() if go in gs]

    kingdom_counts = collections.Counter()
    for prot in sp_pos:
        sp_code = prot.split('_')[-1] if '_' in prot else ''
        k = get_kingdom(sp_code)
        if k != 'Unknown':
            kingdoms_present.add(k)
            kingdom_counts[k] += 1

    tbs = len(kingdoms_present) / len(KINGDOMS)
    sp_dep = len(sp_pos) / (len(sp_pos) + len(human_pos)) if (len(sp_pos) + len(human_pos)) > 0 else 0

    tbs_results[go] = {
        'tbs': tbs,
        'kingdoms_present': sorted(kingdoms_present),
        'n_kingdoms': len(kingdoms_present),
        'n_sp_positive': len(sp_pos),
        'n_human_positive': len(human_pos),
        'sp_dependency': sp_dep,
        'kingdom_counts': dict(kingdom_counts),
    }
    print(f"  {go} ({GO_LABELS[go]:<22}): TBS={tbs:.3f}  "
          f"SP+={len(sp_pos):4d}  Human+={len(human_pos):4d}  "
          f"kingdoms={sorted(kingdoms_present)}")

# ─── TCS 계산 ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("  Step 2: TCS — Tissue Context Specificity (Tau-based)")
print("=" * 65)

go_genes = collections.defaultdict(set)
with open(ANNOT_HUMAN) as f:
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2:
            for go in p[1:]:
                if go in GO_TERMS:
                    go_genes[go].add(p[0])

all_pos_genes = set().union(*go_genes.values())
print(f"  Total unique positive genes across 13 GO terms: {len(all_pos_genes)}")

print("  Loading HPA tissue expression data...")
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
N_tissues = len(tissues_list)
print(f"  HPA tissues: {N_tissues}, Genes: {len(gene_expr)}")


def tau_index(expr_dict, tissue_list):
    vals = np.array([expr_dict.get(t, 0.0) for t in tissue_list], dtype=float)
    mx = vals.max()
    if mx == 0:
        return np.nan
    x_hat = vals / mx
    return float(np.sum(1 - x_hat) / (len(tissue_list) - 1))


def smsi(expr_dict, muscle_tissues=None):
    if muscle_tissues is None:
        muscle_tissues = ['skeletal muscle']
    muscle_expr = np.mean([expr_dict.get(t, 0.0) for t in muscle_tissues])
    all_expr = np.mean(list(expr_dict.values())) if expr_dict else 0.0
    return float(muscle_expr / (all_expr + 1e-10))


# Detect skeletal muscle tissue name
sample_gene = next(iter(gene_expr))
sample_tissues = list(gene_expr[sample_gene].keys())
muscle_hits = [t for t in sample_tissues if 'skeletal' in t.lower() or 'muscle' in t.lower()]
print(f"  Muscle-related tissues found: {muscle_hits}")
MUSCLE_TISSUES = muscle_hits if muscle_hits else ['skeletal muscle']

tcs_results = {}
for go in GO_TERMS:
    pos_genes = go_genes[go]
    tau_vals, smsi_vals = [], []

    for gene in pos_genes:
        if gene in gene_expr:
            tau = tau_index(gene_expr[gene], tissues_list)
            sm  = smsi(gene_expr[gene], MUSCLE_TISSUES)
            if not np.isnan(tau):
                tau_vals.append(tau)
                smsi_vals.append(sm)

    tcs = float(np.mean(tau_vals)) if tau_vals else np.nan
    smsi_mean = float(np.mean(smsi_vals)) if smsi_vals else np.nan
    coverage = len(tau_vals) / len(pos_genes) if pos_genes else 0

    tcs_results[go] = {
        'tcs': tcs,
        'smsi': smsi_mean,
        'n_genes_total': len(pos_genes),
        'n_genes_matched': len(tau_vals),
        'hpa_coverage': coverage,
    }
    print(f"  {go} ({GO_LABELS[go]:<22}): TCS={tcs:.4f}  "
          f"SMSI={smsi_mean:.3f}  coverage={coverage:.0%} ({len(tau_vals)}/{len(pos_genes)})")

# ─── Combined analysis ────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("  Step 3: Correlation TBS/TCS → Δ AUPRC")
print("=" * 65)

rows = []
for go in GO_TERMS:
    d = DELTA_AUPRC[go]
    tbs = tbs_results[go]['tbs']
    tcs = tcs_results[go]['tcs']
    if np.isnan(tcs):
        print(f"  SKIP {go}: TCS=NaN")
        continue
    rows.append({
        'go': go,
        'label': GO_LABELS[go],
        'type': d['type'],
        'tbs': tbs,
        'tcs': tcs,
        'smsi': tcs_results[go]['smsi'],
        'delta': d['delta'],
        'v10b': d['v10b'],
        'lr': d['lr'],
        'sig': d['sig'],
        'n_sp': tbs_results[go]['n_sp_positive'],
        'n_human': tbs_results[go]['n_human_positive'],
        'sp_dep': tbs_results[go]['sp_dependency'],
    })

n = len(rows)
tbs_arr   = np.array([r['tbs']   for r in rows])
tcs_arr   = np.array([r['tcs']   for r in rows])
delta_arr = np.array([r['delta'] for r in rows])

# Pearson correlations
r_tbs,  p_tbs  = stats.pearsonr(tbs_arr,   delta_arr)
r_tcs,  p_tcs  = stats.pearsonr(tcs_arr,   delta_arr)
r_spdep,p_spdep= stats.pearsonr(np.array([r['sp_dep'] for r in rows]), delta_arr)

print(f"\n  n = {n}")
print(f"  Pearson r(TBS,  Δ) = {r_tbs:+.3f}  p={p_tbs:.4f}")
print(f"  Pearson r(TCS,  Δ) = {r_tcs:+.3f}  p={p_tcs:.4f}")
print(f"  Pearson r(SP_dep, Δ) = {r_spdep:+.3f}  p={p_spdep:.4f}")

# OLS regression: Δ ~ TBS + TCS + TBS×TCS
interaction = tbs_arr * tcs_arr
X = np.column_stack([np.ones(n), tbs_arr, tcs_arr, interaction])
try:
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, delta_arr, rcond=None)
    y_hat = X @ coeffs
    ss_res = np.sum((delta_arr - y_hat) ** 2)
    ss_tot = np.sum((delta_arr - delta_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    print(f"\n  OLS: Δ ~ intercept + TBS + TCS + TBS×TCS")
    print(f"    intercept = {coeffs[0]:+.4f}")
    print(f"    β_TBS     = {coeffs[1]:+.4f}")
    print(f"    β_TCS     = {coeffs[2]:+.4f}")
    print(f"    β_TBS×TCS = {coeffs[3]:+.4f}")
    print(f"    R²        = {r2:.4f}  (n={n}, df={n-4})")
except Exception as e:
    coeffs = [np.nan] * 4
    r2 = np.nan
    print(f"  OLS failed: {e}")

# Simple model: Δ ~ TBS + TCS (no interaction)
X2 = np.column_stack([np.ones(n), tbs_arr, tcs_arr])
try:
    c2, _, _, _ = np.linalg.lstsq(X2, delta_arr, rcond=None)
    y2 = X2 @ c2
    ss2 = 1 - np.sum((delta_arr - y2)**2) / np.sum((delta_arr - delta_arr.mean())**2)
    print(f"\n  OLS (no interaction): R² = {ss2:.4f}")
    print(f"    intercept={c2[0]:+.4f}, β_TBS={c2[1]:+.4f}, β_TCS={c2[2]:+.4f}")
except Exception:
    ss2 = np.nan

# ─── Figure ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("  Step 4: Generating figures")
print("=" * 65)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('SwissProt Annotation Quality → v10-B vs LR Performance Gap\n'
             '(13 GO terms, Δ AUPRC = v10-B mean − LR mean)',
             fontsize=12, fontweight='bold')

type_colors = {'A': '#d62728', 'B': '#1f77b4'}
type_labels = {'A': 'Type-A (LR ≥ v10-B)', 'B': 'Type-B (v10-B >> LR)'}

# ── Panel A: TBS vs Δ AUPRC ───────────────────────────────────────────────────
ax = axes[0]
for r in rows:
    c = type_colors[r['type']]
    ax.scatter(r['tbs'], r['delta'], s=120, c=c, alpha=0.85,
               edgecolors='black', linewidth=0.8, zorder=5)
    ax.annotate(r['go'].split(':')[1], xy=(r['tbs'], r['delta']),
                xytext=(4, 3), textcoords='offset points', fontsize=7)

# regression line
xs = np.linspace(tbs_arr.min()-0.05, tbs_arr.max()+0.05, 100)
m_tbs, b_tbs = np.polyfit(tbs_arr, delta_arr, 1)
ax.plot(xs, m_tbs*xs + b_tbs, 'k--', alpha=0.5, lw=1.2)

ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
ax.set_xlabel('TBS (Taxonomic Breadth Score)', fontsize=10)
ax.set_ylabel('Δ AUPRC (v10-B − LR)', fontsize=10)
ax.set_title(f'TBS vs Δ AUPRC\nr={r_tbs:+.3f}  p={p_tbs:.3f}', fontsize=10)

patches = [mpatches.Patch(color=c, label=type_labels[t]) for t, c in type_colors.items()]
ax.legend(handles=patches, fontsize=8, loc='upper left')

# ── Panel B: TCS vs Δ AUPRC ───────────────────────────────────────────────────
ax = axes[1]
for r in rows:
    c = type_colors[r['type']]
    ax.scatter(r['tcs'], r['delta'], s=120, c=c, alpha=0.85,
               edgecolors='black', linewidth=0.8, zorder=5)
    ax.annotate(r['go'].split(':')[1], xy=(r['tcs'], r['delta']),
                xytext=(4, 3), textcoords='offset points', fontsize=7)

xs = np.linspace(tcs_arr.min()-0.01, tcs_arr.max()+0.01, 100)
m_tcs, b_tcs = np.polyfit(tcs_arr, delta_arr, 1)
ax.plot(xs, m_tcs*xs + b_tcs, 'k--', alpha=0.5, lw=1.2)

ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
ax.set_xlabel('TCS (Tissue Context Specificity, Tau-based)', fontsize=10)
ax.set_ylabel('Δ AUPRC (v10-B − LR)', fontsize=10)
ax.set_title(f'TCS vs Δ AUPRC\nr={r_tcs:+.3f}  p={p_tcs:.3f}', fontsize=10)
ax.legend(handles=patches, fontsize=8, loc='upper right')

# ── Panel C: 2D TBS × TCS, color=Δ ───────────────────────────────────────────
ax = axes[2]
delta_vals = np.array([r['delta'] for r in rows])
vmax = max(abs(delta_vals.min()), abs(delta_vals.max()))
norm = plt.Normalize(vmin=-vmax, vmax=vmax)
cmap = plt.cm.RdBu_r

for r in rows:
    color = cmap(norm(r['delta']))
    sz = 60 + abs(r['delta']) * 600
    ax.scatter(r['tbs'], r['tcs'], s=sz, c=[color], alpha=0.85,
               edgecolors='black' if r['type'] == 'B' else 'red',
               linewidth=1.5 if r['type'] == 'A' else 0.8, zorder=5)
    ax.annotate(r['go'].split(':')[1], xy=(r['tbs'], r['tcs']),
                xytext=(4, 3), textcoords='offset points', fontsize=7)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Δ AUPRC (v10-B − LR)')

ax.set_xlabel('TBS (Taxonomic Breadth Score)', fontsize=10)
ax.set_ylabel('TCS (Tissue Context Specificity)', fontsize=10)
ax.set_title(f'2D: TBS × TCS → Δ AUPRC\n'
             f'R²(TBS+TCS+interaction)={r2:.3f}  n={n}', fontsize=10)

# Annotate quadrant
ax.axvline(0.5, color='gray', lw=0.7, alpha=0.4, linestyle=':')
ax.axhline(0.88, color='gray', lw=0.7, alpha=0.4, linestyle=':')
ax.text(0.15, 0.87, 'Low TBS\nLow TCS\n→ SP helpful?', fontsize=7,
        color='gray', ha='center')
ax.text(0.75, 0.92, 'High TBS\nHigh TCS\n→ SP harmful', fontsize=7,
        color='darkred', ha='center')

plt.tight_layout()
fig_path = f'{OUT_DIR}/tbs_tcs_13terms.pdf'
fig_path_png = f'{OUT_DIR}/tbs_tcs_13terms.png'
plt.savefig(fig_path, bbox_inches='tight')
plt.savefig(fig_path_png, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")

# ─── Summary table ────────────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  SUMMARY TABLE")
print(f"{'=' * 80}")
print(f"{'GO Term':<14} {'Label':<24} {'Type'} {'TBS':>5} {'TCS':>6} {'SMSI':>6} "
      f"{'v10-B':>6} {'LR':>6} {'Δ':>6} {'sig':>4}")
print("-" * 80)
for r in rows:
    print(f"{r['go']:<14} {r['label']:<24} {r['type']:^4}  "
          f"{r['tbs']:5.3f} {r['tcs']:6.4f} {r['smsi']:6.3f} "
          f"{r['v10b']:6.3f} {r['lr']:6.3f} {r['delta']:+6.3f} {r['sig']:>4}")

print(f"\n  Pearson r(TBS,  Δ AUPRC) = {r_tbs:+.3f}  p={p_tbs:.4f}  n={n}")
print(f"  Pearson r(TCS,  Δ AUPRC) = {r_tcs:+.3f}  p={p_tcs:.4f}  n={n}")
print(f"  R²(TBS+TCS+interaction)   = {r2:.4f}  df={n-4}")
print(f"  R²(TBS+TCS, no interact.) = {ss2:.4f}  df={n-3}")

# ─── Save results ─────────────────────────────────────────────────────────────
import time
ts = time.strftime('%Y%m%d_%H%M')
out_json = f'{OUT_DIR}/tbs_tcs_results_{ts}.json'
save_data = {
    'n_go_terms': n,
    'pearson_tbs_delta': {'r': float(r_tbs), 'p': float(p_tbs)},
    'pearson_tcs_delta': {'r': float(r_tcs), 'p': float(p_tcs)},
    'pearson_spdep_delta': {'r': float(r_spdep), 'p': float(p_spdep)},
    'ols_tbs_tcs_interaction': {
        'r2': float(r2) if not np.isnan(r2) else None,
        'df': n - 4,
        'intercept': float(coeffs[0]),
        'beta_tbs': float(coeffs[1]),
        'beta_tcs': float(coeffs[2]),
        'beta_interaction': float(coeffs[3]),
    },
    'ols_tbs_tcs_only': {
        'r2': float(ss2) if not np.isnan(ss2) else None,
        'df': n - 3,
    },
    'per_go': rows,
    'timestamp': ts,
}
with open(out_json, 'w') as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"\n  [Saved] {out_json}")
print("\nDone.")
