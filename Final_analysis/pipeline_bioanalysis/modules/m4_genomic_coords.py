"""
M4: Genomic Coordinate & Strand Analysis
Extracts exon structure, CDS coordinates, strand, and detects antisense (NAT) relationship.
"""
import csv
import os
import re


def get_transcript_info(transcript_id: str, config: dict) -> dict:
    """
    Extract genomic info from SQANTI3 classification + GTF.
    Returns full coordinate and structural info.
    """
    clf_path = config["paths"]["brain_classification"]
    gtf_path = config["paths"]["brain_gtf"]

    clf = _parse_classification(transcript_id, clf_path)
    exons = _parse_gtf_exons(transcript_id, gtf_path)
    cds_info = _parse_cds_from_pep_header(transcript_id, config)

    result = {
        "transcript_id": transcript_id,
        "chrom": clf.get("chrom"),
        "strand": clf.get("strand"),
        "associated_gene": clf.get("associated_gene"),
        "structural_category": clf.get("structural_category"),
        "transcript_length": clf.get("length"),
        "exon_count": len(exons),
        "exons": exons,
        "cds_start_genomic": cds_info.get("cds_genomic_start"),
        "cds_end_genomic": cds_info.get("cds_genomic_end"),
        "protein_length": cds_info.get("protein_length"),
        "is_nat": False,
        "nat_relationship": None,
    }

    # NAT detection: compare transcript strand with gene strand from Ensembl annotation
    if clf.get("strand") and clf.get("associated_gene"):
        gene_strand = _lookup_gene_strand(clf["associated_gene"], gtf_path)
        if gene_strand and gene_strand != clf.get("strand"):
            result["is_nat"] = True
            result["nat_relationship"] = {
                "transcript_strand": clf.get("strand"),
                "gene_strand": gene_strand,
                "description": f"NAT: transcript on {clf['strand']} strand, gene on {gene_strand} strand"
            }

    # Compute genomic span
    if exons:
        result["genomic_span_start"] = min(e[0] for e in exons)
        result["genomic_span_end"] = max(e[1] for e in exons)
        result["genomic_span_kb"] = (result["genomic_span_end"] - result["genomic_span_start"]) / 1000

    return result


def _parse_classification(transcript_id: str, clf_path: str) -> dict:
    """Parse SQANTI3 classification table for one transcript."""
    if not os.path.exists(clf_path):
        return {}
    with open(clf_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if transcript_id in row.get("isoform", ""):
                return {
                    "chrom": row.get("chrom"),
                    "strand": row.get("strand"),
                    "length": int(row.get("length", 0)) if row.get("length", "").isdigit() else None,
                    "exons": int(row.get("exons", 0)) if row.get("exons", "").isdigit() else None,
                    "structural_category": row.get("structural_category"),
                    "associated_gene": row.get("associated_gene"),
                    "associated_transcript": row.get("associated_transcript"),
                    "diff_to_TSS": row.get("diff_to_TSS"),
                    "diff_to_TTS": row.get("diff_to_TTS"),
                }
    return {}


def _parse_gtf_exons(transcript_id: str, gtf_path: str) -> list[tuple[int, int]]:
    """Extract exon coordinates from GTF for a transcript."""
    if not os.path.exists(gtf_path):
        return []
    exons = []
    with open(gtf_path) as f:
        for line in f:
            if transcript_id not in line:
                continue
            cols = line.strip().split("\t")
            if len(cols) < 9:
                continue
            if cols[2] == "exon":
                exons.append((int(cols[3]), int(cols[4])))
    return sorted(exons)


def _parse_cds_from_pep_header(transcript_id: str, config: dict) -> dict:
    """
    Parse CDS coordinates from TransDecoder PEP header.
    Header format: >transcript_id.pN|len_aa|strand|cds_start|cds_end
    """
    pep_path = config["paths"]["brain_pep"]
    if not os.path.exists(pep_path):
        return {}
    with open(pep_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            if transcript_id not in line:
                continue
            # Parse header
            m = re.search(r'\|(\d+)_aa\|[+-]\|(\d+)\|(\d+)', line)
            if m:
                return {
                    "protein_length": int(m.group(1)),
                    "cds_tx_start": int(m.group(2)),
                    "cds_tx_end": int(m.group(3)),
                    "cds_genomic_start": None,  # Will compute below if exons available
                    "cds_genomic_end": None,
                }
    return {}


def compute_genomic_cds(exons: list, cds_tx_start: int, cds_tx_end: int,
                         strand: str) -> tuple[int, int]:
    """Map transcript-level CDS coordinates to genomic coordinates."""
    if strand == "+":
        cumlen = 0
        g_start = g_end = None
        for es, ee in sorted(exons):
            length = ee - es + 1
            if g_start is None and cumlen + length >= cds_tx_start:
                offset = cds_tx_start - cumlen - 1
                g_start = es + offset
            if g_end is None and cumlen + length >= cds_tx_end:
                offset = cds_tx_end - cumlen - 1
                g_end = es + offset
            cumlen += length
        return g_start, g_end
    else:
        # Minus strand: exons sorted descending, CDS coordinates in transcript 5'→3'
        cumlen = 0
        g_start = g_end = None
        for es, ee in sorted(exons, reverse=True):
            length = ee - es + 1
            if g_end is None and cumlen + length >= cds_tx_start:
                offset = cds_tx_start - cumlen - 1
                g_end = ee - offset  # minus strand: subtract from right
            if g_start is None and cumlen + length >= cds_tx_end:
                offset = cds_tx_end - cumlen - 1
                g_start = ee - offset
            cumlen += length
        return g_start, g_end


def _lookup_gene_strand(gene_name: str, gtf_path: str) -> str:
    """Quick lookup of gene strand from GTF. Returns '+' or '-' or None."""
    if not os.path.exists(gtf_path):
        return None
    with open(gtf_path) as f:
        for line in f:
            if gene_name not in line:
                continue
            cols = line.strip().split("\t")
            if len(cols) >= 7 and cols[2] in ("gene", "transcript"):
                # Check if this is the reference gene annotation
                if f'gene_id "{gene_name}"' in line or f'gene_name "{gene_name}"' in line:
                    return cols[6]
    return None
