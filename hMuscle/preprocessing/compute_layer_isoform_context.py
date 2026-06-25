"""
Layer × Isoform Context Expansion
Extends layer_isoform_context.tsv to all A-DR / A-BP genes by cross-linking:
  - DRIMSeq/permutation results (cell-type-level isoform switches)
  - cluster → layer mapping (cluster_layer_mapping.tsv)
  - cluster AD enrichment (layer_composition_mwu.tsv)

Output: reports/bisect_celltype/layer_isoform_context_full.tsv
        reports/bisect_celltype/layer_axis_summary.tsv
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parents[2]

# ── Input paths ──────────────────────────────────────────────────────────────
BISECT_JSON  = ROOT / 'prism_app/data/demo/bisect_cases.json'
DRIMSEQ      = ROOT / 'prism_app/data/cell_atlas/drimseq_primary_cases.tsv'
DONOR_PERM   = ROOT / 'prism_app/data/cell_atlas/donor_level_isoform_switches.tsv'
CLUSTER_MAP  = ROOT / 'reports/cluster_layer_mapping.tsv'
LAYER_MWU    = ROOT / 'reports/layer_annotation/layer_composition_mwu.tsv'
FULL_DONOR   = ROOT / 'reports/bisect_celltype/full_donor_level_analysis.tsv'

# ── Output paths ─────────────────────────────────────────────────────────────
OUT_FULL     = ROOT / 'reports/bisect_celltype/layer_isoform_context_full.tsv'
OUT_SUMMARY  = ROOT / 'reports/bisect_celltype/layer_axis_summary.tsv'


# ── Cell type canonical names mapping ────────────────────────────────────────
CT_NORMALIZE = {
    'Excitatory_neuron':    'Excitatory',
    'Excitatory':           'Excitatory',
    'Inhibitory_neuron':    'Inhibitory',
    'Inhibitory':           'Inhibitory',
    'Oligodendrocyte':      'Oligodendrocyte',
    'OPC':                  'OPC',
    'Astrocyte':            'Astrocyte',
    'Microglia':            'Microglia',
    'Vascular':             'Vascular',
}

# Layer depth order (superficial → deep)
LAYER_ORDER = ['L1', 'L1-2', 'L1-3', 'L2/3', 'L3-5', 'L4', 'L5', 'L6', 'WM/L6', 'Mixed', 'Perivasc']


def load_data():
    cluster_df = pd.read_csv(CLUSTER_MAP, sep='\t')
    layer_mwu  = pd.read_csv(LAYER_MWU, sep='\t')

    # Normalize cluster_layer column
    layer_mwu['cluster_id'] = layer_mwu['cluster_layer'].str.extract(r'C(\d+)_')[0].astype(str)
    layer_mwu['layer_from_mwu'] = layer_mwu['cluster_layer'].str.extract(r'C\d+_(.+)')[0]

    # Build cluster → AD delta map
    cluster_ad_delta = {}
    cluster_mwu_p    = {}
    for _, row in layer_mwu.iterrows():
        cid = str(row['cluster_id'])
        cluster_ad_delta[cid] = row['delta_pct']
        cluster_mwu_p[cid]    = row['MWU_p']

    # Build cell type → clusters → layers map
    ct_to_clusters: dict[str, list[dict]] = defaultdict(list)
    for _, row in cluster_df.iterrows():
        ct = CT_NORMALIZE.get(str(row.get('cell_type', '')), str(row.get('cell_type', '')))
        cid = str(int(row['leiden']))
        layer = str(row.get('layer_label', ''))
        ct_to_clusters[ct].append({
            'cluster_id': cid,
            'layer':      layer,
            'subtype':    str(row.get('subtype', '')),
            'ad_delta':   cluster_ad_delta.get(cid, 0.0),
            'mwu_p':      cluster_mwu_p.get(cid, 1.0),
        })

    return cluster_df, cluster_ad_delta, cluster_mwu_p, ct_to_clusters


def load_gene_switches():
    """Load all A-DR + A-BP isoform switches with cell type and statistics."""
    rows = []

    # A-DR from DRIMSeq
    if DRIMSEQ.exists():
        dr = pd.read_csv(DRIMSEQ, sep='\t')
        for _, r in dr.iterrows():
            ct = CT_NORMALIZE.get(str(r.get('cell_type', '')), str(r.get('cell_type', '')))
            rows.append({
                'gene':         str(r.get('gene', '')),
                'cell_type':    ct,
                'tier':         'A-DR',
                'delta':        None,
                'stager_p':     r.get('stager_p'),
                'perm_p':       None,
                'mechanism':    str(r.get('mechanism_class', '')),
                'ct_iso':       str(r.get('ct_isoform', '')),
                'ad_iso':       str(r.get('ad_isoform', '')),
            })

    # A-BP from donor permutation
    if DONOR_PERM.exists():
        dp = pd.read_csv(DONOR_PERM, sep='\t')
        a_bp = dp[dp.get('prism_tier', pd.Series(dtype=str)).str.startswith('tier') == False] \
            if 'prism_tier' not in dp.columns else dp
        # Just take all perm_p < 0.05 as A-BP
        if 'perm_p' in dp.columns:
            a_bp = dp[pd.to_numeric(dp['perm_p'], errors='coerce') <= 0.05]
        else:
            a_bp = dp
        for _, r in a_bp.iterrows():
            ct = CT_NORMALIZE.get(str(r.get('cell_type', '')), str(r.get('cell_type', '')))
            rows.append({
                'gene':      str(r.get('gene', '')),
                'cell_type': ct,
                'tier':      'A-BP',
                'delta':     r.get('delta'),
                'stager_p':  None,
                'perm_p':    r.get('perm_p'),
                'mechanism': 'Complex I / GEF',
                'ct_iso':    str(r.get('ct_iso_used', '')),
                'ad_iso':    '',
            })

    # Also load full_donor_level for additional genes
    if FULL_DONOR.exists():
        fd = pd.read_csv(FULL_DONOR, sep='\t')
        for _, r in fd.iterrows():
            ct = CT_NORMALIZE.get(str(r.get('cell_type', '')), str(r.get('cell_type', '')))
            gene = str(r.get('gene', ''))
            # Avoid duplicates with A-DR/A-BP
            if any(row['gene'] == gene and row['cell_type'] == ct for row in rows):
                continue
            rows.append({
                'gene':      gene,
                'cell_type': ct,
                'tier':      'C',
                'delta':     r.get('delta'),
                'stager_p':  None,
                'perm_p':    r.get('perm_p'),
                'mechanism': '',
                'ct_iso':    str(r.get('ct_iso_used', '')),
                'ad_iso':    '',
            })

    return pd.DataFrame(rows)


def compute_layer_context(switch_df: pd.DataFrame, ct_to_clusters: dict) -> pd.DataFrame:
    """For each gene×cell_type, expand to cluster×layer rows."""
    out_rows = []
    for _, sw in switch_df.iterrows():
        gene    = sw['gene']
        ct      = sw['cell_type']
        tier    = sw['tier']
        delta   = sw.get('delta')
        stager  = sw.get('stager_p')
        perm_p  = sw.get('perm_p')
        mech    = sw.get('mechanism', '')

        clusters = ct_to_clusters.get(ct, [])
        if not clusters:
            # Try without normalization
            for k, v in ct_to_clusters.items():
                if ct.lower() in k.lower():
                    clusters = v
                    break

        if not clusters:
            out_rows.append({
                'gene': gene, 'cell_type': ct, 'tier': tier,
                'cluster_id': 'N/A', 'subtype': '', 'layer': 'unknown',
                'delta': delta, 'stager_p': stager, 'perm_p': perm_p,
                'cluster_AD_delta': None, 'cluster_mwu_p': None,
                'mechanism': mech,
                'co_enrichment_score': None,
            })
            continue

        for cl in clusters:
            cluster_ad = cl['ad_delta']
            # Co-enrichment: isoform shift same direction as cluster AD enrichment?
            try:
                d = float(delta) if delta is not None else 0.0
                co_score = cluster_ad * (-d)  # AD-enriched cluster + AD-decreased isoform = positive
            except Exception:
                co_score = None

            out_rows.append({
                'gene':              gene,
                'cell_type':         ct,
                'tier':              tier,
                'cluster_id':        cl['cluster_id'],
                'subtype':           cl['subtype'],
                'layer':             cl['layer'],
                'delta':             delta,
                'stager_p':          stager,
                'perm_p':            perm_p,
                'cluster_AD_delta':  cluster_ad,
                'cluster_mwu_p':     cl['mwu_p'],
                'mechanism':         mech,
                'co_enrichment_score': co_score,
            })

    return pd.DataFrame(out_rows)


def compute_layer_axis_summary(ctx_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate: for each gene, which layers show strongest co-enrichment?"""
    rows = []
    for (gene, tier, ct), grp in ctx_df.groupby(['gene', 'tier', 'cell_type']):
        layer_grp = grp.groupby('layer').agg(
            n_clusters=('cluster_id', 'count'),
            mean_cluster_AD_delta=('cluster_AD_delta', 'mean'),
            mean_co_enrichment=('co_enrichment_score', 'mean'),
        ).reset_index()

        # Find peak layer
        valid = layer_grp.dropna(subset=['mean_co_enrichment'])
        if len(valid) == 0:
            peak_layer = 'N/A'
            peak_score = None
        else:
            peak_idx   = valid['mean_co_enrichment'].abs().idxmax()
            peak_layer = valid.loc[peak_idx, 'layer']
            peak_score = valid.loc[peak_idx, 'mean_co_enrichment']

        delta = grp['delta'].iloc[0] if len(grp) > 0 else None
        stager = grp['stager_p'].iloc[0] if 'stager_p' in grp else None
        perm_p = grp['perm_p'].iloc[0] if 'perm_p' in grp else None
        mech   = grp['mechanism'].iloc[0] if 'mechanism' in grp else ''

        rows.append({
            'gene':             gene,
            'cell_type':        ct,
            'tier':             tier,
            'mechanism':        mech,
            'delta':            delta,
            'stager_p':         stager,
            'perm_p':           perm_p,
            'peak_layer':       peak_layer,
            'peak_co_score':    peak_score,
            'n_layers_covered': layer_grp['layer'].nunique(),
            'layers':           '; '.join(sorted(layer_grp['layer'].dropna().tolist())),
        })

    return pd.DataFrame(rows).sort_values(
        ['tier', 'peak_co_score'], ascending=[True, False], key=lambda x: x if x.name != 'peak_co_score' else -x.abs()
    )


def main():
    print("Loading data...")
    cluster_df, cluster_ad_delta, cluster_mwu_p, ct_to_clusters = load_data()

    print("Loading gene switches...")
    switch_df = load_gene_switches()
    print(f"  {len(switch_df)} gene×celltype rows loaded")
    print(f"  Tiers: {switch_df['tier'].value_counts().to_dict()}")

    print("Computing layer context...")
    ctx_df = compute_layer_context(switch_df, ct_to_clusters)

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    ctx_df.to_csv(OUT_FULL, sep='\t', index=False)
    print(f"Written: {OUT_FULL} ({len(ctx_df)} rows)")

    print("Computing layer axis summary...")
    summary_df = compute_layer_axis_summary(ctx_df)
    summary_df.to_csv(OUT_SUMMARY, sep='\t', index=False)
    print(f"Written: {OUT_SUMMARY} ({len(summary_df)} rows)")

    # Print key findings
    print("\n=== Key Layer-Isoform Co-enrichment Findings ===")
    tier_a = summary_df[summary_df['tier'].isin(['A-DR', 'A-BP'])].copy()
    tier_a_sorted = tier_a.sort_values('peak_co_score', key=lambda x: -x.abs().fillna(0))
    for _, row in tier_a_sorted.head(15).iterrows():
        print(f"  {row['gene']} ({row['cell_type']}, {row['tier']}): "
              f"peak layer={row['peak_layer']}, co_score={row['peak_co_score']}, "
              f"delta={row['delta']}, mechanism={row['mechanism']}")


if __name__ == '__main__':
    main()
