"""
IsoformSwitchAnalyzeR-style feature-based consequence classification
for BISECT 138 cases.

Consequence types (per field standard):
  1. domain_loss     — n_lost > 0
  2. domain_gain     — n_gained > 0
  3. nmd_gain        — AD isoform is NMD-sensitive
  4. intron_retention — AD structural category = retained_intron
  5. coding_loss      — CT is protein-coding, AD is not
  6. loc_change       — MTS signal category changed (proxy for signal peptide)
  7. alt_promoter     — TSS class is alt_promoter_candidate or tss_shift
  8. apa_change       — APA class is major_apa or moderate_apa
  9. ppi_disruption   — PPI_verdict = SUPPORTED (loss of known interaction)

These are STRUCTURAL/FEATURE-LEVEL labels (Methods-level).
No mechanism (compensatory/pathological) assignment.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# ── Load data ────────────────────────────────────────────────────────────────
BASE = '/home/welcome1/sw1686/DIFFUSE/Final_analysis/pipeline_bioanalysis/outputs'
df138 = pd.read_csv(f'{BASE}/cases_summary_20260625_1321.tsv', sep='\t')
df121 = pd.read_csv(f'{BASE}/supplementary_table_S_bisect_121cases.tsv', sep='\t')

# Normalize column names in 121 file
df121.columns = [c.lower() for c in df121.columns]
df121 = df121.rename(columns={
    'prism_delta': 'diffuse_delta',
    'ct_transcript': 'ct_transcript_id',
    'ad_transcript': 'ad_transcript_id',
    'n_lost': 'n_lost_121',
    'n_gained': 'n_gained_121',
    'domains_lost': 'domains_lost_121',
    'domains_gained': 'domains_gained_121',
})

# ── Merge: enrich 138 with 121 annotation columns ────────────────────────────
merge_cols = ['gene', 'cell_type', 'nmd_gate', 'ad_nmd', 'mechanism',
              'tss_class', 'apa_class', 'ppi_verdict', 'n_ppi_hits',
              'top_ppi_partner', 'ad_phylop_mean']
available_cols = [c for c in merge_cols if c in df121.columns]
df_rich = df121[available_cols].copy()

df = df138.merge(df_rich, on=['gene', 'cell_type'], how='left')

# Fill NMD from structural_category_ad for new cases not in 121
# structural_category_ad == 'nonsense_mediated_decay' → NMD
def infer_nmd(row):
    if pd.notna(row.get('nmd_gate')) and row['nmd_gate'] != '':
        return row['nmd_gate']
    cat = str(row.get('structural_category_ad', ''))
    if 'nonsense_mediated_decay' in cat:
        return 'YES'
    return 'NO'

df['nmd_gate_final'] = df.apply(infer_nmd, axis=1)

# ── Feature-based consequence classification ─────────────────────────────────
def classify_consequences(row):
    consequences = []

    # 1. Protein domain loss
    if row.get('n_lost', 0) > 0:
        consequences.append('domain_loss')

    # 2. Protein domain gain
    if row.get('n_gained', 0) > 0:
        consequences.append('domain_gain')

    # 3. NMD sensitivity gain (AD isoform degraded)
    if row.get('nmd_gate_final') == 'YES':
        consequences.append('nmd_gain')

    # 4. Intron retention
    cat_ad = str(row.get('structural_category_ad', ''))
    if 'retained_intron' in cat_ad:
        consequences.append('intron_retention')

    # 5. Coding potential loss (protein_coding CT → non-coding AD)
    cat_ct = str(row.get('structural_category_ct', ''))
    non_coding_cats = {'retained_intron', 'nonsense_mediated_decay',
                       'incomplete-splice_match', 'TEC'}
    if 'protein_coding' in cat_ct:
        if any(nc in cat_ad for nc in non_coding_cats):
            consequences.append('coding_loss')

    # 6. Localization signal change (MTS proxy for signal peptide)
    mts_ct = row.get('mts_ct', np.nan)
    mts_ad = row.get('mts_ad', np.nan)
    if pd.notna(mts_ct) and pd.notna(mts_ad):
        # MTS scores: higher = stronger signal; >1 level change = meaningful
        if abs(float(mts_ct) - float(mts_ad)) >= 1.5:
            consequences.append('loc_change')

    # 7. Alternative promoter (TSS change)
    tss = str(row.get('tss_class', ''))
    if 'alt_promoter' in tss or 'tss_shift' in tss:
        consequences.append('alt_promoter')

    # 8. APA change (3' end change)
    apa = str(row.get('apa_class', ''))
    if 'major_apa' in apa or 'moderate_apa' in apa:
        consequences.append('apa_change')

    # 9. PPI disruption
    if str(row.get('ppi_verdict', '')) == 'SUPPORTED':
        consequences.append('ppi_disruption')

    if not consequences:
        consequences.append('no_consequence_detected')

    return consequences

df['consequences'] = df.apply(classify_consequences, axis=1)
df['n_consequences'] = df['consequences'].apply(len)
df['consequence_string'] = df['consequences'].apply(lambda x: ';'.join(x))

# ── Primary consequence (highest priority) ───────────────────────────────────
PRIORITY = ['coding_loss', 'nmd_gain', 'domain_loss', 'domain_gain',
            'loc_change', 'intron_retention', 'alt_promoter',
            'apa_change', 'ppi_disruption', 'no_consequence_detected']

def primary_consequence(cons_list):
    for p in PRIORITY:
        if p in cons_list:
            return p
    return 'no_consequence_detected'

df['primary_consequence'] = df['consequences'].apply(primary_consequence)

# ── Statistics ────────────────────────────────────────────────────────────────
PASS = df[df['stage2_pass'] == 'YES'].copy()
print(f"\n{'='*65}")
print(f"  BISECT Feature-Based Consequence Classification")
print(f"  IsoformSwitchAnalyzeR-style | n={len(df)} total, n={len(PASS)} PASS")
print(f"{'='*65}\n")

# 1. Consequence type frequency (all PASS cases)
print("── 1. Consequence type frequency (PASS cases, n=101) ─────────────")
from collections import Counter
all_cons = []
for c in PASS['consequences']:
    all_cons.extend(c)
cnt = Counter(all_cons)
total_pass = len(PASS)
for con, n in sorted(cnt.items(), key=lambda x: -x[1]):
    pct = 100 * n / total_pass
    bar = '█' * int(pct / 3)
    print(f"  {con:<28} {n:3d} / {total_pass}  ({pct:5.1f}%)  {bar}")

# 2. Primary consequence distribution
print("\n── 2. Primary consequence distribution (PASS cases) ──────────────")
pc = PASS['primary_consequence'].value_counts()
for con, n in pc.items():
    pct = 100 * n / total_pass
    print(f"  {con:<28} {n:3d}  ({pct:5.1f}%)")

# 3. Consequence by cell type
print("\n── 3. Consequence by cell type (PASS, any consequence) ───────────")
ct_cons = defaultdict(Counter)
for _, row in PASS.iterrows():
    for c in row['consequences']:
        ct_cons[row['cell_type']][c] += 1
ct_order = ['Excitatory', 'Inhibitory', 'Oligodendrocyte', 'Astrocyte', 'Microglia', 'OPC']
for ct in ct_order:
    if ct not in ct_cons:
        continue
    n_ct = len(PASS[PASS['cell_type'] == ct])
    top = ct_cons[ct].most_common(3)
    top_str = ', '.join([f"{c}({n})" for c, n in top])
    print(f"  {ct:<18} n={n_ct:3d}  top: {top_str}")

# 4. Multi-consequence cases (≥2 consequence types)
print("\n── 4. Multi-consequence cases (PASS, ≥2 types) ───────────────────")
multi = PASS[PASS['n_consequences'] >= 2].sort_values('n_consequences', ascending=False)
print(f"  Multi-consequence cases: {len(multi)} / {total_pass} ({100*len(multi)/total_pass:.1f}%)")
for _, row in multi.head(15).iterrows():
    print(f"  {row['gene']:<12} {row['cell_type']:<18} [{row['consequence_string']}]")

# 5. Domain loss details
print("\n── 5. High-impact domain loss cases (PASS, n_lost ≥ 2) ───────────")
domain_loss = PASS[(PASS['n_lost'] >= 2)].sort_values('n_lost', ascending=False)
for _, row in domain_loss.iterrows():
    print(f"  {row['gene']:<12} {row['cell_type']:<18} lost={row['n_lost']:.0f}  domains: {row['domains_lost']}")

# 6. Coding potential loss cases
print("\n── 6. Coding potential loss cases (PASS) ─────────────────────────")
coding_loss = PASS[PASS['primary_consequence'] == 'coding_loss']
for _, row in coding_loss.iterrows():
    print(f"  {row['gene']:<12} {row['cell_type']:<18} CT:{row['structural_category_ct']} → AD:{row['structural_category_ad']}")

# 7. NMD cases
print("\n── 7. NMD-sensitive AD isoform cases (PASS) ──────────────────────")
nmd_cases = PASS[PASS['nmd_gate_final'] == 'YES']
if len(nmd_cases) == 0:
    print("  [None detected in PASS set]")
for _, row in nmd_cases.iterrows():
    print(f"  {row['gene']:<12} {row['cell_type']:<18} struct_ad: {row['structural_category_ad']}")

# 8. No consequence detected — worth examining
print("\n── 8. No consequence detected cases (PASS) ───────────────────────")
no_cons = PASS[PASS['primary_consequence'] == 'no_consequence_detected']
print(f"  Count: {len(no_cons)}")
for _, row in no_cons.iterrows():
    print(f"  {row['gene']:<12} {row['cell_type']:<18} struct_ct:{row['structural_category_ct']} struct_ad:{row['structural_category_ad']}")

# 9. Direction × Consequence cross-tab
print("\n── 9. AD_high vs CT_high by primary consequence ──────────────────")
ct_tab = pd.crosstab(PASS['direction'], PASS['primary_consequence'])
print(ct_tab.to_string())

# 10. Summary: feature coverage
print("\n── 10. Feature annotation coverage ──────────────────────────────")
features = {
    'domain_loss/gain': (df['domain_change'] == 'YES').sum(),
    'NMD (final)': (df['nmd_gate_final'] == 'YES').sum(),
    'intron_retention': (df['structural_category_ad'] == 'retained_intron').sum(),
    'loc_change (MTS)': (df['consequences'].apply(lambda x: 'loc_change' in x)).sum(),
    'alt_promoter (121 subset)': (df['tss_class'].notna() & df['tss_class'].str.contains('alt_promoter|tss_shift', na=False)).sum(),
    'apa_change (121 subset)': (df['apa_class'].notna() & df['apa_class'].str.contains('major|moderate', na=False)).sum(),
    'ppi_disruption (121 subset)': (df['ppi_verdict'] == 'SUPPORTED').sum(),
    'signal_peptide': 0,  # not in current data
    'IDR': 0,             # not in current data
}
for f, n in features.items():
    flag = ' ← MISSING' if n == 0 else ''
    print(f"  {f:<35} {n:3d} cases{flag}")

# ── Save output ───────────────────────────────────────────────────────────────
out_cols = ['gene', 'cell_type', 'stage2_pass', 'diffuse_delta', 'dtu_pvalue',
            'direction', 'n_lost', 'n_gained', 'domains_lost', 'domains_gained',
            'nmd_gate_final', 'structural_category_ct', 'structural_category_ad',
            'mts_ct', 'mts_ad', 'tss_class', 'apa_class', 'ppi_verdict',
            'primary_consequence', 'consequence_string', 'n_consequences']
out_cols = [c for c in out_cols if c in df.columns]
df[out_cols].to_csv(f'{BASE}/bisect_138_feature_classified.tsv', sep='\t', index=False)
print(f"\n  Saved: {BASE}/bisect_138_feature_classified.tsv")
print(f"{'='*65}\n")
