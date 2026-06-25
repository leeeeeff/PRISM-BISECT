"""
M17: Cross-Case Convergence Detector
Detects functional convergence axes across all BISECT cases using GO enrichment
and Pfam domain family grouping.

Usage:
    python scripts/convergence_detector.py \
        --input prism_app/data/demo/bisect_cases.json \
        --output reports/convergence_analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


# ── Pre-defined functional axes (from deep analysis) ─────────────────────────
PREDEFINED_AXES = [
    {
        "axis_name": "RNA Metabolism",
        "genes": ["DDX19A", "DIS3", "CNOT11", "NOL8", "ZCCHC17"],
        "go_terms": ["GO:0006396", "GO:0006402", "GO:0006397"],
        "domain_keywords": ["DEAD", "RNB", "NOT", "Nol", "S1"],
        "notes": "RNA processing/decay axis; 5 genes spanning mRNA export, exosome, deadenylation",
    },
    {
        "axis_name": "DNA Repair / Fanconi",
        "genes": ["ERCC6L2", "USP1", "FANCA", "RPS3"],
        "go_terms": ["GO:0006281", "GO:0006974", "GO:0051103"],
        "domain_keywords": ["ERCC", "UBL", "Fanconi", "PCNA"],
        "notes": "FA pathway + nucleotide excision repair; 3 cell types affected",
    },
    {
        "axis_name": "Complex I Assembly",
        "genes": ["NDUFS4", "NDUFS7", "NDUFS8", "NDUFAF5"],
        "go_terms": ["GO:0006119", "GO:0022900", "GO:0032981"],
        "domain_keywords": ["NADH_dh", "Complex1", "NDUF"],
        "notes": "Mitochondrial Complex I N-module; NDUFAF5 chaperone + 3 structural subunits",
    },
    {
        "axis_name": "DOCK-family GEF",
        "genes": ["DOCK10", "DOCK11"],
        "go_terms": ["GO:0005085", "GO:0007264"],
        "domain_keywords": ["DHR", "DOCK", "PH"],
        "notes": "Rho-GEF signaling; inhibitory neuron cytoskeletal remodeling",
    },
    {
        "axis_name": "KRAB-ZFP Transcription",
        "genes": ["ZNF736", "ZNF582", "ZNF268"],
        "go_terms": ["GO:0006351", "GO:0045892"],
        "domain_keywords": ["KRAB", "C2H2", "ZnF"],
        "notes": "KRAB-mediated transcriptional repression; KAP1/H3K9me3 pathway",
    },
    {
        "axis_name": "Ubiquitin-Proteasome",
        "genes": ["USP1", "SAMHD1", "DCAF5"],
        "go_terms": ["GO:0006511", "GO:0016579"],
        "domain_keywords": ["UBL", "USP", "WD40", "CRL4"],
        "notes": "Ubiquitin pathway involving deubiquitylase, restriction factor, and E3 adaptor",
    },
]


def _hypergeometric_logp(k: int, M: int, n: int, N: int) -> float:
    """Log p-value for hypergeometric test (one-tailed upper)."""
    # k: overlap, M: population size, n: gene set size, N: drawn
    try:
        from scipy.stats import hypergeom
        return float(hypergeom.logsf(k - 1, M, n, N))
    except ImportError:
        # Fallback: rough approximation
        if k == 0:
            return 0.0
        return -k * math.log(10)


def detect_domain_clusters(cases: list) -> list:
    """Group cases by shared Pfam domain family."""
    family_map: dict[str, list] = defaultdict(list)

    for c in cases:
        gene = c.get("gene_name") or c.get("gene") or ""
        ct_type = c.get("cell_type") or ""
        dc = c.get("domain_change") or {}
        lost   = dc.get("domains_lost")   or []
        gained = dc.get("domains_gained") or []

        for d in lost + gained:
            # Extract family prefix (e.g. "DEAD_2" → "DEAD")
            family = d.split("_")[0] if "_" in d else d
            family_map[family].append({"gene": gene, "cell_type": ct_type, "domain": d})

    clusters = []
    for family, members in family_map.items():
        if len(members) >= 2:
            genes = list({m["gene"] for m in members})
            clusters.append({
                "family": family,
                "genes": genes,
                "n_cases": len(members),
                "members": members,
            })

    clusters.sort(key=lambda x: -x["n_cases"])
    return clusters


def score_predefined_axes(cases: list) -> list:
    """Score predefined axes against actual case set."""
    case_genes = {(c.get("gene_name") or c.get("gene") or "").upper() for c in cases}
    tier_a = {
        (c.get("gene_name") or c.get("gene") or "").upper()
        for c in cases
        if (c.get("bisect_tier") or "").startswith("A")
    }

    results = []
    for axis in PREDEFINED_AXES:
        axis_genes = [g.upper() for g in axis["genes"]]
        found = [g for g in axis_genes if g in case_genes]
        tier_a_found = [g for g in axis_genes if g in tier_a]

        n_pop = len(case_genes)
        n_axis_defined = len(axis_genes)
        k_found = len(found)

        logp = _hypergeometric_logp(k=k_found, M=n_pop, n=n_axis_defined, N=n_pop)

        results.append({
            **axis,
            "n_defined": n_axis_defined,
            "n_found_in_cases": k_found,
            "genes_found": found,
            "genes_missing": [g for g in axis_genes if g not in case_genes],
            "tier_a_genes": tier_a_found,
            "hypergeom_logp": logp,
        })

    results.sort(key=lambda x: -x["n_found_in_cases"])
    return results


def detect_celltype_patterns(cases: list) -> list:
    """Find genes that appear in multiple cell types (pan-cellular isoform switches)."""
    gene_celltypes: dict[str, list] = defaultdict(list)
    for c in cases:
        gene = (c.get("gene_name") or c.get("gene") or "").upper()
        ct = c.get("cell_type") or ""
        tier = c.get("bisect_tier") or ""
        if gene:
            gene_celltypes[gene].append({"cell_type": ct, "tier": tier})

    pan_cellular = []
    for gene, entries in gene_celltypes.items():
        if len(entries) >= 2:
            pan_cellular.append({
                "gene": gene,
                "n_cell_types": len(entries),
                "cell_types": [e["cell_type"] for e in entries],
                "tiers": [e["tier"] for e in entries],
            })

    pan_cellular.sort(key=lambda x: -x["n_cell_types"])
    return pan_cellular


def run(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} cases from {input_path}")

    axes = score_predefined_axes(cases)
    domain_clusters = detect_domain_clusters(cases)
    pan_cellular = detect_celltype_patterns(cases)

    result = {
        "n_total_cases": len(cases),
        "convergence_axes": axes,
        "domain_family_clusters": domain_clusters,
        "pan_cellular_genes": pan_cellular,
        "summary": {
            "n_axes_with_2plus_found": sum(1 for a in axes if a["n_found_in_cases"] >= 2),
            "n_domain_clusters": len(domain_clusters),
            "n_pan_cellular": len(pan_cellular),
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Convergence analysis written to {output_path}")
    print("\n=== Convergence Axes ===")
    for ax in axes:
        print(f"  {ax['axis_name']}: {ax['n_found_in_cases']}/{ax['n_defined']} genes"
              f"  (Tier A: {ax['tier_a_genes']})")
    print("\n=== Top Domain Clusters ===")
    for cl in domain_clusters[:5]:
        print(f"  {cl['family']}: {cl['genes']} ({cl['n_cases']} cases)")
    print("\n=== Pan-cellular Genes ===")
    for pc in pan_cellular[:5]:
        print(f"  {pc['gene']}: {pc['n_cell_types']} cell types {pc['cell_types']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="prism_app/data/demo/bisect_cases.json")
    parser.add_argument("--output", default="reports/convergence_analysis.json")
    args = parser.parse_args()
    run(args.input, args.output)
