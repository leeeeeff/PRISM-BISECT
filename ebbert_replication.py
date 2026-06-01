#!/usr/bin/env python3
"""
ebbert_replication.py
Check whether key AD isoform switches from Samsung scRNA-seq
replicate in Ebbert Lab bulk frontal cortex long-read dataset.

ENSG IDs verified from IsoQuant GTF (ENSG00000116852/164258) and
current Ensembl (ENSG00000142949). Proxy ENST IDs chosen by
structural similarity to Samsung IsoQuant novel transcripts:
  - KIF21B AD proxy ENST00000422435: IsoQuant similar_reference_id of tr292978
  - NDUFS4 AD proxy ENST00000507026: NDUFS4-205, closest 6-exon form to tr73243
  - PTPRF CT proxy ENST00000436724: 8-exon short form (PTPRF-207, no TM domain)
"""
import pandas as pd
import numpy as np
from scipy import stats
import json, os

os.chdir('/home/welcome1/sw1686/DIFFUSE')

df = pd.read_csv('hMuscle/data/brain_ebbert/counts_transcript_ebbert.txt', sep='\t')
sample_cols = [c for c in df.columns if c not in ['TXNAME', 'GENEID']]

# From GitHub notebook: create_deseq2_annotations.ipynb
conditions = ['AD','CT','AD','CT','CT','CT','CT','AD','AD','CT','AD','AD']
ad_cols = [sample_cols[i] for i,c in enumerate(conditions) if c=='AD']
ct_cols = [sample_cols[i] for i,c in enumerate(conditions) if c=='CT']

meta = pd.DataFrame({'sample': sample_cols, 'condition': conditions})
meta.to_csv('hMuscle/data/brain_ebbert/sample_metadata.csv', index=False)
print("Metadata saved. AD=%d, CT=%d" % (len(ad_cols), len(ct_cols)))

# Cases: (gene, ENSG, CT_transcript, AD_transcript, Samsung_note)
# CT_transcript = form dominant in CT, expected to DECREASE in AD
# AD_transcript = form dominant in AD, expected to INCREASE in AD
#
# ENSG IDs: current Ensembl/GENCODE (matching IsoQuant GTF gene_id field)
# ENST proxies: structurally closest annotated transcripts in Ebbert GENCODE 38 data
cases = [
    ('KIF21B', 'ENSG00000116852',
     'ENST00000332129',  # KIF21B-201, 34 exons, full canonical with motor domain (CT-high)
     'ENST00000422435',  # KIF21B alt, 35 exons, same 3p end as tr292978 (IsoQuant similar_ref)
     'Motor-domain switch: CT canonical (34ex) -> AD WD40 alt (same TSS as tr292978)'),

    ('NDUFS4', 'ENSG00000164258',
     'ENST00000296684',  # NDUFS4-201, 5 exons, canonical (CT-high, 91-93% of gene in both)
     'ENST00000506974',  # NDUFS4-204, 6 exons, increases in AD p=0.041 in Ebbert (AD-high proxy)
     'NDUFS4 isoform shift: CT canonical (5ex) -> AD alt NDUFS4-204 (6ex, p=0.041 in Ebbert)'),

    ('DLG1', 'ENSG00000075711',
     'ENST00000667157',  # DLG1-261, 25 exons, dominant canonical form (CT=41%, decreases in AD)
     'ENST00000392381',  # DLG1-204, 19 exons, closest to tr319500 (IsoQuant similar_ref; AD proxy)
     'OPC dedifferentiation: CT canonical (25ex, 41%) -> AD novel DLG1-204 (19ex, OPC-specific tr319500 proxy)'),
]

print()
print("%-8s %-22s %-22s %10s %10s %10s %10s %10s  Replicate?" %
      ("Gene","CT_tx","AD_tx","p(MWU)","pCT_ct","pAD_ct","pCT_ad","pAD_ad"))
print("-" * 110)

results = []
for gene, ensg, ct_tx, ad_tx, note in cases:
    sub = df[df['GENEID'].str.startswith(ensg)].copy()
    row_ct = sub[sub['TXNAME'] == ct_tx]
    row_ad = sub[sub['TXNAME'] == ad_tx]

    if len(row_ct) == 0 or len(row_ad) == 0:
        print("%s: transcript missing (ct=%d, ad=%d)" % (gene, len(row_ct), len(row_ad)))
        continue

    gene_totals = sub[sample_cols].sum(axis=0).values + 1e-9
    ct_raw = row_ct[sample_cols].values[0]
    ad_raw = row_ad[sample_cols].values[0]
    ct_props = ct_raw / gene_totals
    ad_props = ad_raw / gene_totals

    ct_idx = [sample_cols.index(c) for c in ct_cols]
    ad_idx = [sample_cols.index(c) for c in ad_cols]

    pCT_ct = ct_props[ct_idx].mean()   # CT transcript in CT samples
    pAD_ct = ct_props[ad_idx].mean()   # CT transcript in AD samples (should decrease)
    pCT_ad = ad_props[ct_idx].mean()   # AD transcript in CT samples
    pAD_ad = ad_props[ad_idx].mean()   # AD transcript in AD samples (should increase)

    delta = pAD_ct - pCT_ct  # negative = CT tx decreases in AD (replication)

    # Mann-Whitney U
    try:
        _, pval = stats.mannwhitneyu(ct_props[ad_idx], ct_props[ct_idx], alternative='two-sided')
    except Exception:
        pval = float('nan')

    replicate = "YES" if delta < 0 else "NO"
    print("%-8s %-22s %-22s %10.4f %10.4f %10.4f %10.4f %10.4f  %s" %
          (gene, ct_tx, ad_tx, pval, pCT_ct, pAD_ct, pCT_ad, pAD_ad, replicate))

    results.append({
        'gene': gene, 'note': note,
        'ct_transcript': ct_tx, 'ad_transcript': ad_tx,
        'prop_CT_in_CT': float(pCT_ct), 'prop_CT_in_AD': float(pAD_ct),
        'prop_AD_in_CT': float(pCT_ad), 'prop_AD_in_AD': float(pAD_ad),
        'delta_ct_tx': float(delta),
        'mwu_pval': float(pval),
        'direction_replicated': bool(delta < 0),
    })

os.makedirs('reports/ebbert_replication', exist_ok=True)
with open('reports/ebbert_replication/replication_result.json', 'w') as f:
    json.dump({'cohort': 'Ebbert Lab (Nature Biotech 2024)',
               'n_AD': len(ad_cols), 'n_CT': len(ct_cols),
               'tissue': 'frontal cortex (BA9/46)', 'tech': 'Oxford Nanopore PromethION',
               'switches': results}, f, indent=2)
print("\nSaved: reports/ebbert_replication/replication_result.json")
