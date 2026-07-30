#!/usr/bin/env python3
"""
tissue_compare_and_enrichment.py
===================================
(1) Confirm the tissue-asymmetry framing directly: at matched large size, is
    muscle's fragmentation excess really muscle-specific, or does brain show
    an analogous top-gene concentration pattern of its own (for comparison)?
(2) Formal enrichment test: are genes with high n_intervals in the large-edit
    subset enriched for cytoskeleton-related GO annotation (GO:0005856
    cytoskeleton, GO:0008092 cytoskeletal protein binding, GO:0015629 actin
    cytoskeleton), against the background of all genes contributing to that
    subset? Fisher's exact test, both tissues.
"""
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')

CYTO_GO = {'GO:0005856', 'GO:0008092', 'GO:0015629', 'GO:0030036', 'GO:0007010'}
# cytoskeleton / cytoskeletal protein binding / actin cytoskeleton / actin cytoskeleton organization / cytoskeleton organization


def load_gene2go_symbol_map():
    """NCBI gene2go gzip -> {GeneID: set(GO_ID)}, human only (tax_id 9606), plus
    a GeneID->Symbol map from the same file (Symbol column)."""
    go_by_geneid = {}
    symbol_by_geneid = {}
    with gzip.open(ROOT / 'hMuscle/data/raw_data/data/annotations/gene2go.gz', 'rt') as f:
        header = f.readline().rstrip('\n').split('\t')
        idx = {c: i for i, c in enumerate(header)}
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts[idx['#tax_id']] != '9606':
                continue
            gid = parts[idx['GeneID']]
            go_id = parts[idx['GO_ID']]
            go_by_geneid.setdefault(gid, set()).add(go_id)
    return go_by_geneid


def load_ensg_to_geneid():
    """Use genes.gtf.gz to map ENSG (gene_id, versionless) -> gene_name (symbol),
    then rely on a symbol->NCBI GeneID crosswalk if available; simpler: use
    NCBI gene_info to map Symbol<->GeneID directly, and genes.gtf.gz for ENSG<->Symbol."""
    ensg_to_symbol = {}
    with gzip.open(ROOT / 'hMuscle/data/refdata-gex-GRCh38-2024-A/genes/genes.gtf.gz', 'rt') as f:
        for line in f:
            if '\tgene\t' not in line:
                continue
            import re
            g = re.search(r'gene_id "([^"."]+)', line)
            n = re.search(r'gene_name "([^"]+)"', line)
            if g and n:
                ensg_to_symbol[g.group(1)] = n.group(1)
    return ensg_to_symbol


def load_gene_info_symbol_to_geneid():
    path = ROOT / 'hMuscle/data/raw_data/data/annotations/Homo_sapiens.gene_info.gz'
    if not path.exists():
        # fall back: search for any gene_info file
        cands = list((ROOT / 'hMuscle/data/raw_data/data/annotations').glob('*gene_info*'))
        if not cands:
            return {}
        path = cands[0]
    sym2id = {}
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        header = f.readline().rstrip('\n').split('\t')
        idx = {c: i for i, c in enumerate(header)}
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts[idx['#tax_id']] != '9606':
                continue
            sym2id[parts[idx['Symbol']]] = parts[idx['GeneID']]
    return sym2id


def analyze_tissue(tissue, go_by_geneid, ensg_to_symbol, sym2id):
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_with_dispersion.tsv', sep='\t')
    large = df[df['size'] >= df['size'].quantile(0.7)].copy()
    gene_stats = large.groupby('gene').agg(n_pairs=('n_intervals', 'size'),
                                            sum_intervals=('n_intervals', 'sum'),
                                            mean_intervals=('n_intervals', 'mean')).sort_values('sum_intervals', ascending=False)
    total = gene_stats['sum_intervals'].sum()
    top10 = gene_stats.head(10)
    print(f"\n=== {tissue}: top-10 genes by summed n_intervals (large-size subset, n_genes={len(gene_stats)}) ===")

    # muscle 'gene' column is ENSG-versioned; brain 'gene' column is already a gene symbol
    # (build_severity_pairs.py: brain gene_ids come from brain_full_gene_names.npy)
    def to_symbol(gene_val):
        if tissue == 'muscle':
            return ensg_to_symbol.get(gene_val.split('.')[0], gene_val)
        return gene_val

    top10 = top10.copy()
    top10['symbol'] = [to_symbol(g) for g in top10.index]
    print(top10[['symbol', 'n_pairs', 'sum_intervals', 'mean_intervals']].round(2).to_string())
    print(f"top-10 share: {top10['sum_intervals'].sum()/total*100:.1f}%  (n_genes contributing={len(gene_stats)})")

    # ---- Fisher enrichment: high-mean_intervals genes vs cytoskeleton GO annotation ----
    median_mi = gene_stats['mean_intervals'].median()
    gene_stats = gene_stats.copy()
    gene_stats['high_frag'] = gene_stats['mean_intervals'] > median_mi

    is_cyto = []
    n_mapped = 0
    for gene_val in gene_stats.index:
        sym = to_symbol(gene_val)
        gid = sym2id.get(sym) if sym else None
        go_set = go_by_geneid.get(gid, set()) if gid else set()
        if gid is not None:
            n_mapped += 1
        is_cyto.append(bool(go_set & CYTO_GO))
    gene_stats['is_cyto'] = is_cyto
    print(f"GeneID-mapped: {n_mapped}/{len(gene_stats)}; cytoskeleton-annotated: {sum(is_cyto)}")

    ct = pd.crosstab(gene_stats['high_frag'], gene_stats['is_cyto'])
    print(ct)
    if ct.shape == (2, 2):
        odds, p = stats.fisher_exact(ct)
        print(f"Fisher's exact: OR={odds:.3f}, p={p:.4f}")
    return gene_stats


def main():
    print("Loading GO annotation (gene2go) and gene_info...")
    go_by_geneid = load_gene2go_symbol_map()
    ensg_to_symbol = load_ensg_to_geneid()
    sym2id = load_gene_info_symbol_to_geneid()
    print(f"gene2go: {len(go_by_geneid)} genes; ENSG->symbol: {len(ensg_to_symbol)}; symbol->GeneID: {len(sym2id)}")

    for tissue in ['muscle', 'brain']:
        analyze_tissue(tissue, go_by_geneid, ensg_to_symbol, sym2id)

    # ---- direct tissue comparison: fragmentation magnitude at large size ----
    print("\n=== Direct comparison: mean n_intervals in large-size (top-30%) subset ===")
    for tissue in ['muscle', 'brain']:
        df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_with_dispersion.tsv', sep='\t')
        large = df[df['size'] >= df['size'].quantile(0.7)]
        print(f"{tissue}: mean n_intervals={large['n_intervals'].mean():.2f}, n_pairs={len(large)}, n_genes={large['gene'].nunique()}")


if __name__ == '__main__':
    main()
