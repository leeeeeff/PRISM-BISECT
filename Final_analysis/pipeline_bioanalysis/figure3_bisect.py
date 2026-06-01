#!/usr/bin/env python3
"""
Figure 3: BISECT M10-M12 Multi-Evidence Structural Analysis
Three panels:
  A) 9-case evidence heatmap (M10 pLDDT / M11 STRING / M12 phyloP)
  B) KIF21B pLDDT-by-residue trace from ESMAtlas PDB (CT kinesin vs AD WD40)
  C) PTPRF domain-level pLDDT comparison (CT: PTP domains vs AD: Ig domains)
"""
import json, pathlib, struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from Bio.PDB import PDBParser

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_plddt_from_pdb(pdb_path):
    """Extract per-residue pLDDT from ESMAtlas PDB (B-factor in 0–1 scale → ×100)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('mol', str(pdb_path))
    plddt_vals = []
    seen_res = set()
    for model in structure:
        for chain in model:
            for residue in chain:
                rid = residue.get_id()
                if rid in seen_res:
                    continue
                seen_res.add(rid)
                atoms = list(residue.get_atoms())
                if atoms:
                    bfac = atoms[0].get_bfactor()
                    # ESMAtlas stores pLDDT/100 in B-factor
                    plddt = bfac * 100 if bfac <= 1.0 else bfac
                    plddt_vals.append(min(plddt, 100.0))
    return np.array(plddt_vals)


# ── data ─────────────────────────────────────────────────────────────────────

CASES = ['NDUFS4','DLG1','KIF21B','PTPRF','IFT122','FANCA','SYNE1','RGS3','ADGRB2']
CASE_DIRS = {
    'NDUFS4':'outputs/NDUFS4_Excitatory/analysis.json',
    'DLG1':  'outputs/DLG1_OPC/analysis.json',
    'KIF21B':'outputs/KIF21B_Excitatory/analysis.json',
    'PTPRF': 'outputs/PTPRF_Inhibitory/analysis.json',
    'IFT122':'outputs/IFT122_Excitatory/analysis.json',
    'FANCA': 'outputs/FANCA_Excitatory/analysis.json',
    'SYNE1': 'outputs/SYNE1_Inhibitory/analysis.json',
    'RGS3':  'outputs/RGS3_Astrocyte/analysis.json',
    'ADGRB2':'outputs/ADGRB2_Inhibitory/analysis.json',
}

# Manual ESMAtlas values (not in pipeline JSON — obtained via API segmental runs)
ESMFOLD_OVERRIDE = {
    # (CT, AD) — None where pipeline captured AlphaFold DB value
    'KIF21B': {'ct': 93.2, 'ad': 94.6},  # CT kinesin aa1-380; AD WD40 aa370-620
}

# Cell type labels
CELL_TYPES = {
    'NDUFS4':'Excitatory','DLG1':'OPC','KIF21B':'Excitatory','PTPRF':'Inhibitory',
    'IFT122':'Excitatory','FANCA':'Excitatory','SYNE1':'Inhibitory',
    'RGS3':'Astrocyte','ADGRB2':'Inhibitory',
}
CELL_COLORS = {
    'Excitatory':'#E74C3C','Inhibitory':'#3498DB','OPC':'#9B59B6',
    'Astrocyte':'#2ECC71',
}

# Tier labels
TIER = {
    'KIF21B':'A','PTPRF':'A','FANCA':'A',
    'NDUFS4':'B','DLG1':'B','IFT122':'B','SYNE1':'B','RGS3':'B',
    'ADGRB2':'C',
}

records = {}
for gene in CASES:
    d = json.loads(pathlib.Path(CASE_DIRS[gene]).read_text())
    m10 = d.get('m11_alphafold', {})
    m11 = d.get('m12_ppi', {})
    m12 = d.get('m13_conservation', {})
    ct_plddt = m10.get('ct', {}).get('plddt_mean')
    ad_plddt = m10.get('ad', {}).get('plddt_mean')
    # override with ESMAtlas manual results
    if gene in ESMFOLD_OVERRIDE:
        ov = ESMFOLD_OVERRIDE[gene]
        if ct_plddt is None: ct_plddt = ov.get('ct')
        if ad_plddt is None: ad_plddt = ov.get('ad')
    hits = m11.get('string_hits', [])
    top_score = max((h.get('combined_score', 0) for h in hits), default=0) / 10.0  # → 0–100
    ad_phyloP = m12.get('summary', {}).get('ad_specific_mean_phyloP')
    ct_phyloP = m12.get('summary', {}).get('ct_specific_mean_phyloP')
    records[gene] = {
        'ct_plddt': ct_plddt, 'ad_plddt': ad_plddt,
        'string_score': top_score,
        'ad_phyloP': ad_phyloP, 'ct_phyloP': ct_phyloP,
        'ct_domains': m10.get('ct', {}).get('domain_plddt', {}),
        'ad_domains': m10.get('ad', {}).get('domain_plddt', {}),
    }

# ── build heatmap matrix ──────────────────────────────────────────────────────
# Columns: CT pLDDT | AD pLDDT | STRING top (scaled 0-100) | AD phyloP | CT phyloP
# Normalise each column to 0-1 for colour, keep raw for annotation

hm_cols  = ['CT\npLDDT','AD\npLDDT','STRING\nscore','AD\nphyloP','CT\nphyloP']
col_max  = [100,        100,         100,             6.0,         6.0      ]
col_min  = [50,          50,           0,            -1.0,        -1.0      ]

mat_raw  = np.full((9, 5), np.nan)
for i, gene in enumerate(CASES):
    r = records[gene]
    mat_raw[i] = [
        r['ct_plddt'],
        r['ad_plddt'],
        r['string_score'],
        r['ad_phyloP'],
        r['ct_phyloP'],
    ]

mat_norm = np.full_like(mat_raw, np.nan)
for j in range(5):
    lo, hi = col_min[j], col_max[j]
    mat_norm[:, j] = np.where(
        np.isnan(mat_raw[:, j]),
        np.nan,
        np.clip((mat_raw[:, j] - lo) / (hi - lo), 0, 1)
    )

# ── KIF21B pLDDT traces ───────────────────────────────────────────────────────
pdb_ct_path = pathlib.Path('/tmp/kif21b_ct_380.pdb')
pdb_ad_path = pathlib.Path('/tmp/kif21b_ad_wd40core.pdb')
ct_plddt_trace = parse_plddt_from_pdb(pdb_ct_path) if pdb_ct_path.exists() else np.array([])
ad_plddt_trace = parse_plddt_from_pdb(pdb_ad_path) if pdb_ad_path.exists() else np.array([])

# ── PTPRF domain comparison ──────────────────────────────────────────────────
ptprf_ct_domains = records['PTPRF']['ct_domains']
ptprf_ad_domains = records['PTPRF']['ad_domains']

# Select display domains (non-redundant; pick most interpretable)
CT_DISPLAY = {
    'Y_phosphatase': 'Y_Phosphatase\n(72.3)',
    'DSPc':          'DSPc\n(78.6)',
    'fn3':           'FN3\n(85.8)',
    'Interfer-bind': 'Interfer-bind\n(85.5)',
}
AD_DISPLAY = {
    'Ig_3':    'Ig_3\n(84.8)',
    'I-set':   'I-set\n(85.0)',
    'V-set':   'V-set\n(86.7)',
    'C2-set_2':'C2-set\n(87.5)',
}

def get_domain_plddt_safe(domain_dict, key):
    if key in domain_dict and domain_dict[key].get('mean') is not None:
        return domain_dict[key]['mean']
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 13))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    height_ratios=[1.6, 1.0, 1.0],
    width_ratios=[2.0, 1.5, 1.5],
    hspace=0.52, wspace=0.38,
)

ax_hm   = fig.add_subplot(gs[0, :])          # full-width heatmap
ax_kif  = fig.add_subplot(gs[1, :2])         # KIF21B pLDDT trace (wide)
ax_ptprf = fig.add_subplot(gs[1, 2])         # PTPRF domain bar chart
ax_leg  = fig.add_subplot(gs[2, :])          # legend / notes row
ax_leg.axis('off')

# ── Panel A: heatmap ─────────────────────────────────────────────────────────
cmap_plddt = LinearSegmentedColormap.from_list(
    'plddt', ['#f0f0f0','#ffd966','#6aa84f'], N=256)
cmap_string = LinearSegmentedColormap.from_list(
    'string', ['#f0f0f0','#4a86c8','#1a3a6b'], N=256)
cmap_phyloP = LinearSegmentedColormap.from_list(
    'phyloP', ['#c0392b','#f0f0f0','#27ae60'], N=256)

col_cmaps = [cmap_plddt, cmap_plddt, cmap_string, cmap_phyloP, cmap_phyloP]

# Draw manually with per-column colormap
cell_w = 0.14
cell_h = 0.08
xs = [0.12, 0.26, 0.42, 0.58, 0.74]
for j, (x0, cmap) in enumerate(zip(xs, col_cmaps)):
    for i, gene in enumerate(CASES):
        val_norm = mat_norm[i, j]
        val_raw  = mat_raw[i, j]
        y0 = 0.92 - i * 0.085
        if np.isnan(val_norm):
            color = '#e8e8e8'
            txt   = 'N/A'
        else:
            color = cmap(val_norm)
            if j <= 1:   txt = f'{val_raw:.1f}'
            elif j == 2: txt = f'{val_raw:.0f}'
            else:        txt = f'{val_raw:.2f}' if not np.isnan(val_raw) else 'N/A'
        rect = mpatches.FancyBboxPatch(
            (x0, y0 - cell_h), cell_w, cell_h,
            boxstyle='round,pad=0.005', linewidth=0.5,
            edgecolor='#cccccc', facecolor=color,
            transform=ax_hm.transAxes, clip_on=False,
        )
        ax_hm.add_patch(rect)
        ax_hm.text(
            x0 + cell_w / 2, y0 - cell_h / 2, txt,
            ha='center', va='center', fontsize=7.5, fontweight='bold',
            color='black' if val_norm is not None and (np.isnan(val_norm) or val_norm < 0.75) else 'white',
            transform=ax_hm.transAxes,
        )

# Gene labels (left)
for i, gene in enumerate(CASES):
    y0 = 0.92 - i * 0.085
    tier = TIER[gene]
    tier_color = {'A':'#E74C3C','B':'#E67E22','C':'#7F8C8D'}[tier]
    ax_hm.text(
        0.01, y0 - cell_h / 2 + 0.01,
        f'[{tier}] {gene}',
        ha='left', va='center', fontsize=9, fontweight='bold',
        color='#333333', transform=ax_hm.transAxes,
    )
    # cell type chip
    ct_label = CELL_TYPES[gene]
    ct_col = CELL_COLORS[ct_label]
    ax_hm.text(
        0.095, y0 - cell_h / 2 + 0.01,
        ct_label[:3],
        ha='center', va='center', fontsize=6.5, color='white', fontweight='bold',
        transform=ax_hm.transAxes,
        bbox=dict(boxstyle='round,pad=0.15', facecolor=ct_col, edgecolor='none'),
    )

# Column headers
for j, (x0, label) in enumerate(zip(xs, hm_cols)):
    ax_hm.text(
        x0 + cell_w / 2, 0.96, label,
        ha='center', va='bottom', fontsize=8.5, fontweight='bold',
        color='#444444', transform=ax_hm.transAxes,
    )

ax_hm.set_xlim(0, 1); ax_hm.set_ylim(0, 1)
ax_hm.axis('off')
ax_hm.set_title('A  BISECT Multi-Evidence Panel: 9 AD Isoform Switches',
                 loc='left', fontsize=11, fontweight='bold', pad=6, color='#222222')

# ── Panel B: KIF21B pLDDT trace ───────────────────────────────────────────────
if len(ct_plddt_trace) > 0 and len(ad_plddt_trace) > 0:
    x_ct = np.arange(1, len(ct_plddt_trace) + 1)
    x_ad = np.arange(371, 371 + len(ad_plddt_trace))   # WD40 starts ~aa370
    ax_kif.fill_between(x_ct, ct_plddt_trace, alpha=0.25, color='#E74C3C')
    ax_kif.plot(x_ct, ct_plddt_trace, color='#E74C3C', lw=1.5, label='CT kinesin motor (aa1–380)')
    ax_kif.fill_between(x_ad, ad_plddt_trace, alpha=0.25, color='#3498DB')
    ax_kif.plot(x_ad, ad_plddt_trace, color='#3498DB', lw=1.5, label='AD WD40 core (aa370–620)')
    # mean pLDDT lines
    ax_kif.axhline(np.mean(ct_plddt_trace), color='#E74C3C', ls='--', lw=0.9, alpha=0.7)
    ax_kif.axhline(np.mean(ad_plddt_trace), color='#3498DB', ls='--', lw=0.9, alpha=0.7)
    # annotate means
    ax_kif.text(50,  np.mean(ct_plddt_trace)+2, f'μ={np.mean(ct_plddt_trace):.1f}',
                color='#E74C3C', fontsize=8)
    ax_kif.text(580, np.mean(ad_plddt_trace)+2, f'μ={np.mean(ad_plddt_trace):.1f}',
                color='#3498DB', fontsize=8)
    # domain annotations
    ax_kif.axvspan(1, 380,   alpha=0.07, color='#E74C3C')
    ax_kif.axvspan(370, 620, alpha=0.07, color='#3498DB')
    ax_kif.text(190, 14, 'Kinesin motor', ha='center', fontsize=8, color='#E74C3C', style='italic')
    ax_kif.text(495, 14, 'WD40 propeller', ha='center', fontsize=8, color='#3498DB', style='italic')
    ax_kif.axhline(70, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax_kif.text(5, 71, 'pLDDT=70', fontsize=7, color='gray')
else:
    ax_kif.text(0.5, 0.5, 'PDB files not found\n(/tmp/kif21b_*.pdb)',
                ha='center', va='center', fontsize=11, transform=ax_kif.transAxes, color='gray')

ax_kif.set_xlim(0, 640)
ax_kif.set_ylim(0, 105)
ax_kif.set_xlabel('Residue position (canonical coordinates)', fontsize=9)
ax_kif.set_ylabel('ESMAtlas pLDDT', fontsize=9)
ax_kif.set_title('B  KIF21B: GOF switch — kinesin motor → WD40 scaffold',
                 loc='left', fontsize=10, fontweight='bold')
ax_kif.legend(fontsize=8, loc='lower right', framealpha=0.85)
ax_kif.spines[['top','right']].set_visible(False)
ax_kif.tick_params(labelsize=8)

# ── Panel C: PTPRF domain pLDDT ──────────────────────────────────────────────
ct_dom_labels = []
ct_dom_vals   = []
ad_dom_labels = []
ad_dom_vals   = []

for k, label in CT_DISPLAY.items():
    v = get_domain_plddt_safe(ptprf_ct_domains, k)
    if v is not None:
        ct_dom_labels.append(label)
        ct_dom_vals.append(v)

for k, label in AD_DISPLAY.items():
    v = get_domain_plddt_safe(ptprf_ad_domains, k)
    if v is not None:
        ad_dom_labels.append(label)
        ad_dom_vals.append(v)

all_labels = [f'CT: {l}' for l in ct_dom_labels] + [f'AD: {l}' for l in ad_dom_labels]
all_vals   = ct_dom_vals + ad_dom_vals
colors_bar = ['#E74C3C'] * len(ct_dom_vals) + ['#3498DB'] * len(ad_dom_vals)

bars = ax_ptprf.barh(range(len(all_labels)), all_vals, color=colors_bar,
                     edgecolor='white', height=0.65)
ax_ptprf.set_yticks(range(len(all_labels)))
ax_ptprf.set_yticklabels(all_labels, fontsize=7.5)
ax_ptprf.set_xlim(50, 100)
ax_ptprf.axvline(70, color='gray', ls=':', lw=0.8)
ax_ptprf.text(70.5, len(all_labels) - 0.4, 'pLDDT=70', fontsize=7, color='gray')
ax_ptprf.set_xlabel('Mean domain pLDDT', fontsize=9)
ax_ptprf.set_title('C  PTPRF: domain-level\nstructural confidence',
                   loc='left', fontsize=9.5, fontweight='bold')
ax_ptprf.spines[['top','right']].set_visible(False)
ax_ptprf.tick_params(labelsize=8)

# Add separator line between CT and AD groups
sep_y = len(ct_dom_vals) - 0.5
ax_ptprf.axhline(sep_y, color='#cccccc', lw=1.0, ls='-')
ax_ptprf.text(99, sep_y + 0.15, 'CT isoform\n(PTP)', ha='right', va='bottom',
              fontsize=7.5, color='#E74C3C', fontweight='bold')
ax_ptprf.text(99, sep_y - 1.5, 'AD isoform\n(Ig only)', ha='right', va='top',
              fontsize=7.5, color='#3498DB', fontweight='bold')

# ── Legend row ───────────────────────────────────────────────────────────────
legend_text = (
    "Tier A (high-confidence): KIF21B (GOF motor→scaffold), PTPRF (dominant-negative PTP), FANCA (phyloP=−0.493, accelerated evolution)\n"
    "Tier B (moderate): NDUFS4, DLG1, IFT122, SYNE1, RGS3 | Tier C (marginal): ADGRB2 (M11/M12 divergence, calibration)\n"
    "pLDDT: AlphaFold DB (canonical) or ESMAtlas API (novel isoforms, segmental, ≤400aa). "
    "STRING: combined score ÷ 10 shown. phyloP 100-way: positive = conserved, negative = accelerated evolution."
)
ax_leg.text(0.01, 0.90, legend_text, ha='left', va='top', fontsize=8,
            transform=ax_leg.transAxes, color='#444444',
            wrap=True, linespacing=1.6)

# Tier legend chips
for tier, col, xoff in [('Tier A','#E74C3C',0.0),('Tier B','#E67E22',0.12),('Tier C','#7F8C8D',0.24)]:
    ax_leg.text(xoff + 0.0, 0.22, f'■ {tier}', ha='left', va='bottom',
                fontsize=8.5, color=col, fontweight='bold',
                transform=ax_leg.transAxes)

plt.suptitle(
    'Figure 3 | Structural and evolutionary evidence for AD-associated isoform functional reprogramming',
    fontsize=11.5, fontweight='bold', y=1.002, color='#111111',
)

out = pathlib.Path('outputs/figure3_bisect_M10_M12.pdf')
fig.savefig(str(out), dpi=300, bbox_inches='tight', facecolor='white')
out_png = pathlib.Path('outputs/figure3_bisect_M10_M12.png')
fig.savefig(str(out_png), dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
print(f'Saved: {out_png}')
plt.close(fig)
