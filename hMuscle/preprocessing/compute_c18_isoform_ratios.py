"""
C18 L4 IT atypical cluster — isoform ratio analysis
Computes per-donor CT/AD isoform ratios for Complex I + A-DR target genes
within C18 (AD-enriched) vs C10/C11 (canonical L4) vs C19 (L5 ET) clusters.
"""

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import json

# Paths
H5AD = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/03_AnnData/adata_transcript_loose_filtering_for_bulk_analysis.h5ad')
BARCODE_DIR = Path('reports/c18_barcodes')
OUT_DIR = Path('reports/c18_deep_dive')
OUT_DIR.mkdir(exist_ok=True)

# Target genes and their key CT/AD transcript pairs (from BISECT + DRIMSeq)
TARGET_PAIRS = {
    'NDUFS4':  ('NDUFS4-201',                   'transcript73243.chr5.nnic'),  # A-BP Excitatory
    'NDUFS7':  ('NDUFS7-202',                    'NDUFS7-210'),                 # A-BP Excitatory
    'NDUFS8':  ('transcript100761.chr11.nic',    'transcript100759.chr11.nic'), # A-BP Inhibitory→uses Excitatory proxy
    'NDUFAF5': ('NDUFAF5-202',                   'NDUFAF5-201'),                # A-DR Excitatory
    'ZNF736':  ('ZNF736-202',                    'transcript76787.chr7.nnic'),  # A-DR Excitatory
    'ZNF582':  ('transcript145765.chr19.nnic',   'ZNF582-202'),                 # A-DR Excitatory
    'DCAF5':   ('DCAF5-201',                     'DCAF5-209'),                  # A-DR Excitatory
    'RPS3':    ('RPS3-217',                      'RPS3-209'),                   # A-DR Excitatory
}

CLUSTERS = {'C18': 'C18', 'C10': 'C10', 'C11': 'C11', 'C19': 'C19'}

print('Loading h5ad metadata...')
f = h5py.File(H5AD, 'r')

# Read var (transcript) info
tx_names = np.array([x.decode() if isinstance(x, bytes) else x for x in f['var']['transcript_name'][:]])
gn_grp = f['var']['gene_name']
if hasattr(gn_grp, 'keys'):
    cats = [x.decode() if isinstance(x, bytes) else x for x in gn_grp['categories'][:]]
    codes = np.array(gn_grp['codes'])
    gene_names = np.array([cats[c] for c in codes])
else:
    gene_names = np.array([x.decode() if isinstance(x, bytes) else x for x in gn_grp[:]])

# Read obs (cell) info
obs_barcodes = np.array([x.decode() if isinstance(x, bytes) else x for x in f['obs']['_index'][:]])
# condition
cond_grp = f['obs']['condition']
if hasattr(cond_grp, 'keys'):
    cond_cats = [x.decode() if isinstance(x, bytes) else x for x in cond_grp['categories'][:]]
    cond_codes = np.array(cond_grp['codes'])
    conditions = np.array([cond_cats[c] for c in cond_codes])
else:
    conditions = np.array([x.decode() if isinstance(x, bytes) else x for x in cond_grp[:]])
# donor
donor_grp = f['obs']['donor']
if hasattr(donor_grp, 'keys'):
    donor_cats = [x.decode() if isinstance(x, bytes) else x for x in donor_grp['categories'][:]]
    donor_codes = np.array(donor_grp['codes'])
    donors = np.array([donor_cats[c] for c in donor_codes])
else:
    donors = np.array([x.decode() if isinstance(x, bytes) else x for x in donor_grp[:]])

print(f'  {len(obs_barcodes)} cells, {len(tx_names)} transcripts')

# Read sparse matrix X (CSR)
print('Reading sparse count matrix...')
data = f['X']['data'][:]
indices = f['X']['indices'][:]
indptr = f['X']['indptr'][:]
X = sp.csr_matrix((data, indices, indptr), shape=(len(obs_barcodes), len(tx_names)))
f.close()
print('  Done. nnz:', X.nnz)

# Build barcode → obs_idx map
bc2idx = {bc: i for i, bc in enumerate(obs_barcodes)}

# Load cluster barcodes
cluster_cells = {}
for cluster_name, tsv_stem in CLUSTERS.items():
    tsv = BARCODE_DIR / f'{tsv_stem}_barcodes.tsv'
    if not tsv.exists():
        print(f'  WARNING: {tsv} not found')
        continue
    df_bc = pd.read_csv(tsv, sep='\t')
    bc_col = df_bc.columns[0]  # barcode column
    barcodes = df_bc[bc_col].tolist()
    idxs = [bc2idx[bc] for bc in barcodes if bc in bc2idx]
    missing = len(barcodes) - len(idxs)
    if missing > 0:
        print(f'  {cluster_name}: {len(idxs)} matched, {missing} missing from h5ad')
    cluster_cells[cluster_name] = np.array(idxs)
    print(f'  {cluster_name}: {len(idxs)} cells')

# For each target gene, extract isoform ratios per cluster per condition per donor
print('\nComputing isoform ratios...')
rows = []

for gene, (ct_tx, ad_tx) in TARGET_PAIRS.items():
    ct_col = np.where(tx_names == ct_tx)[0]
    ad_col = np.where(tx_names == ad_tx)[0]
    if len(ct_col) == 0 or len(ad_col) == 0:
        print(f'  {gene}: transcript not found (ct={ct_tx}: {len(ct_col)}, ad={ad_tx}: {len(ad_col)})')
        continue
    ct_col, ad_col = ct_col[0], ad_col[0]

    for cluster_name, cell_idxs in cluster_cells.items():
        if len(cell_idxs) == 0:
            continue
        # Per-cell counts
        ct_counts = np.array(X[cell_idxs, ct_col].todense()).flatten()
        ad_counts = np.array(X[cell_idxs, ad_col].todense()).flatten()
        cell_conds = conditions[cell_idxs]
        cell_donors = donors[cell_idxs]

        # Aggregate per donor
        for donor in np.unique(cell_donors):
            donor_mask = cell_donors == donor
            ct_sum = ct_counts[donor_mask].sum()
            ad_sum = ad_counts[donor_mask].sum()
            total = ct_sum + ad_sum
            if total == 0:
                continue
            ct_ratio = ct_sum / total
            ad_ratio = ad_sum / total
            cond = cell_conds[donor_mask][0]
            n_cells = donor_mask.sum()
            rows.append({
                'gene': gene,
                'cluster': cluster_name,
                'donor': donor,
                'condition': cond,
                'n_cells': n_cells,
                'ct_transcript': ct_tx,
                'ad_transcript': ad_tx,
                'ct_count': ct_sum,
                'ad_count': ad_sum,
                'total_count': total,
                'ct_ratio': ct_ratio,
                'ad_ratio': ad_ratio,
                'ad_minus_ct_ratio': ad_ratio - ct_ratio,
            })

df = pd.DataFrame(rows)
print(f'  {len(df)} rows (donor × gene × cluster)')

out_tsv = OUT_DIR / 'c18_isoform_ratios.tsv'
df.to_csv(out_tsv, sep='\t', index=False)
print(f'Saved: {out_tsv}')

# Summary: AD vs CT mean ratio per cluster per gene
print('\n=== AD vs CT mean isoform ratio (AD-enriched transcript) ===')
summary_rows = []
for gene in TARGET_PAIRS:
    gdf = df[df['gene'] == gene]
    if gdf.empty:
        continue
    for cluster in CLUSTERS:
        cdf = gdf[gdf['cluster'] == cluster]
        if cdf.empty:
            continue
        ad_donors = cdf[cdf['condition'] == 'AD']['ad_ratio']
        ct_donors = cdf[cdf['condition'] == 'Control']['ad_ratio']
        delta = ad_donors.mean() - ct_donors.mean() if len(ct_donors) > 0 and len(ad_donors) > 0 else np.nan
        summary_rows.append({
            'gene': gene,
            'cluster': cluster,
            'AD_mean_ad_ratio': round(ad_donors.mean(), 4) if len(ad_donors) > 0 else np.nan,
            'CT_mean_ad_ratio': round(ct_donors.mean(), 4) if len(ct_donors) > 0 else np.nan,
            'AD_minus_CT': round(delta, 4) if not np.isnan(delta) else np.nan,
            'AD_n_donors': len(ad_donors),
            'CT_n_donors': len(ct_donors),
        })

df_sum = pd.DataFrame(summary_rows)
out_sum = OUT_DIR / 'c18_isoform_ratio_summary.tsv'
df_sum.to_csv(out_sum, sep='\t', index=False)
print(df_sum.to_string(index=False))
print(f'\nSaved: {out_sum}')

# Highlight C18-specific effects
print('\n=== C18-specific enrichment (C18 vs C10+C11 average) ===')
for gene in TARGET_PAIRS:
    gdf = df_sum[df_sum['gene'] == gene]
    c18_delta = gdf[gdf['cluster'] == 'C18']['AD_minus_CT'].values
    c10_delta = gdf[gdf['cluster'] == 'C10']['AD_minus_CT'].values
    c11_delta = gdf[gdf['cluster'] == 'C11']['AD_minus_CT'].values
    c19_delta = gdf[gdf['cluster'] == 'C19']['AD_minus_CT'].values
    canonical_delta = np.nanmean(list(c10_delta) + list(c11_delta))
    c18_val = c18_delta[0] if len(c18_delta) > 0 else np.nan
    c19_val = c19_delta[0] if len(c19_delta) > 0 and not np.isnan(c19_delta[0]) else float('nan')
    print(f'  {gene}: C18={c18_val:.4f}, C10/C11 avg={canonical_delta:.4f}, C19={c19_val:.4f}, C18-specific={c18_val - canonical_delta:.4f}')
