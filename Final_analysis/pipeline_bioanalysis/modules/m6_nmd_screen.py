"""
M6: NMD (Nonsense-Mediated mRNA Decay) Susceptibility Screening (BISECT v1.1)

For each isoform (CT and AD), determines whether the transcript is likely
subject to NMD using the canonical "50-nucleotide rule":

    PTC is > 50 nt upstream of the last exon-exon junction → NMD susceptible.

Two-tier logic:
  1. SQANTI3 classification file: reads is_NMD / NMD column directly.
  2. Fallback (CDS + exon coords from M4): calculates PTC position relative
     to last exon-exon junction using genomic exon boundaries.

Entry point: run(case_result, config) -> dict
"""

import os
import csv
from typing import Optional


# ── SQANTI3 classification reader ────────────────────────────────────────────

def _parse_sqanti_classification(sqanti_path: str, transcript_ids: list) -> dict:
    """
    Parse SQANTI3 classification file. Returns {transcript_id: row_dict}.
    Handles both tab and comma delimiters; handles BOM.
    """
    if not sqanti_path or not os.path.exists(sqanti_path):
        return {}
    id_set = set(transcript_ids)
    results = {}
    try:
        with open(sqanti_path, newline="", encoding="utf-8-sig") as f:
            sample = f.read(4096)
            f.seek(0)
            delim = "\t" if "\t" in sample else ","
            reader = csv.DictReader(f, delimiter=delim)
            id_col = None
            for row in reader:
                if id_col is None:
                    # Auto-detect transcript ID column
                    for cname in ("isoform", "transcript_id", "pb", "id"):
                        if cname in row:
                            id_col = cname
                            break
                    if id_col is None:
                        id_col = reader.fieldnames[0]
                tid = row.get(id_col, "").strip()
                if tid in id_set:
                    results[tid] = dict(row)
    except Exception:
        pass
    return results


def _sqanti_nmd_status(sqanti_row: dict) -> Optional[bool]:
    """Extract NMD flag from a SQANTI3 row. Returns True/False/None."""
    for col in ("is_NMD", "NMD", "nmd", "is_nmd"):
        val = sqanti_row.get(col, "").strip().upper()
        if val in ("TRUE", "YES", "1", "T"):
            return True
        if val in ("FALSE", "NO", "0", "F"):
            return False
    return None


def _sqanti_structural_category(sqanti_row: dict) -> str:
    for col in ("structural_category", "SQANTI_cat", "category"):
        val = sqanti_row.get(col, "")
        if val:
            return val.strip()
    return ""


# ── 50-nt rule calculation from coordinates ───────────────────────────────────

def _calc_nmd_from_coords(exons: list, cds_start: Optional[int],
                           cds_end: Optional[int], strand: str) -> dict:
    """
    Apply the 50-nt rule using genomic exon boundaries.

    Args:
        exons: list of {start: int, end: int} (0-based half-open, genomic order)
        cds_start: genomic coordinate of CDS start (5' end of ORF)
        cds_end: genomic coordinate of CDS end (stop codon last nt)
        strand: '+' or '-'

    Returns dict with keys: nmd_susceptible, ptc_to_last_junction_nt, details
    """
    if not exons or cds_start is None or cds_end is None:
        return {"nmd_susceptible": None, "ptc_to_last_junction_nt": None,
                "details": "insufficient coordinate data"}

    # Sort exons by genomic position
    sorted_exons = sorted(exons, key=lambda e: e["start"])

    # Build exon-exon junction list (positions in mRNA space)
    # Junction i is between exon[i] and exon[i+1]; position = cumulative end of exon[i]
    if len(sorted_exons) < 2:
        return {"nmd_susceptible": False, "ptc_to_last_junction_nt": None,
                "details": "single-exon transcript — NMD exempt"}

    # Compute mRNA coordinate for each genomic position (approximate via exon cumsum)
    def genomic_to_mrna(gpos: int) -> Optional[int]:
        """Map genomic position to 0-based mRNA offset (strand-aware)."""
        cumulative = 0
        if strand == "+":
            for ex in sorted_exons:
                if ex["start"] <= gpos < ex["end"]:
                    return cumulative + (gpos - ex["start"])
                cumulative += ex["end"] - ex["start"]
        else:
            for ex in reversed(sorted_exons):
                if ex["start"] <= gpos < ex["end"]:
                    return cumulative + (ex["end"] - 1 - gpos)
                cumulative += ex["end"] - ex["start"]
        return None

    # PTC = stop codon position in mRNA
    # On + strand: cds_end is last nt of stop codon → PTC mRNA pos = genomic_to_mrna(cds_end - 1)
    # On - strand: cds_end is genomic lower bound → PTC = genomic_to_mrna(cds_end)
    if strand == "+":
        ptc_mrna = genomic_to_mrna(cds_end - 1)
    else:
        ptc_mrna = genomic_to_mrna(cds_end)

    if ptc_mrna is None:
        return {"nmd_susceptible": None, "ptc_to_last_junction_nt": None,
                "details": "CDS end not within annotated exons"}

    # Last exon-exon junction mRNA position
    # Junction between exon[N-2] and exon[N-1]:
    # position = sum of lengths of exons 0..N-2
    if strand == "+":
        junction_exons = sorted_exons
    else:
        junction_exons = list(reversed(sorted_exons))

    last_junction_mrna = sum(
        ex["end"] - ex["start"] for ex in junction_exons[:-1]
    )

    # Distance from PTC to last junction (positive = PTC is upstream = NMD-eligible)
    dist = last_junction_mrna - ptc_mrna

    return {
        "nmd_susceptible": dist > 50,
        "ptc_to_last_junction_nt": dist,
        "details": (
            f"PTC at mRNA pos {ptc_mrna}; last EEJ at mRNA pos {last_junction_mrna}; "
            f"distance = {dist} nt ({'NMD' if dist > 50 else 'escape'})"
        ),
    }


# ── Per-isoform NMD assessment ────────────────────────────────────────────────

def _assess_isoform(transcript_id: str, case_result: dict,
                    sqanti_rows: dict, info_key: str) -> dict:
    """Assess NMD susceptibility for one isoform."""
    record = {
        "transcript_id": transcript_id,
        "source": None,
        "nmd_susceptible": None,
        "ptc_to_last_junction_nt": None,
        "structural_category": "",
        "sqanti_nmd": None,
        "coord_calc": {},
        "conclusion": "",
    }

    # Tier 1: SQANTI3 classification
    sqanti_row = sqanti_rows.get(transcript_id)
    if sqanti_row is not None:
        record["source"] = "sqanti3"
        sqanti_nmd = _sqanti_nmd_status(sqanti_row)
        record["sqanti_nmd"] = sqanti_nmd
        record["structural_category"] = _sqanti_structural_category(sqanti_row)
        if sqanti_nmd is not None:
            record["nmd_susceptible"] = sqanti_nmd
            record["conclusion"] = (
                f"{'NMD_SUSCEPTIBLE' if sqanti_nmd else 'NMD_ESCAPE'} "
                f"[SQANTI3; category={record['structural_category']}]"
            )
            return record

    # Tier 2: coordinate-based 50-nt rule
    info = case_result.get(info_key, {})
    exons = info.get("exons", [])
    cds_start = info.get("cds_start_genomic")
    cds_end = info.get("cds_end_genomic")
    strand = info.get("strand", "+")

    coord_result = _calc_nmd_from_coords(exons, cds_start, cds_end, strand)
    record["coord_calc"] = coord_result
    record["ptc_to_last_junction_nt"] = coord_result.get("ptc_to_last_junction_nt")

    if coord_result.get("nmd_susceptible") is not None:
        record["source"] = "50nt_rule"
        record["nmd_susceptible"] = coord_result["nmd_susceptible"]
        record["conclusion"] = (
            f"{'NMD_SUSCEPTIBLE' if record['nmd_susceptible'] else 'NMD_ESCAPE'} "
            f"[50-nt rule; {coord_result['details']}]"
        )
    else:
        record["source"] = "unavailable"
        record["conclusion"] = f"UNKNOWN [{coord_result.get('details', 'no data')}]"

    return record


# ── Entry point ───────────────────────────────────────────────────────────────

def run(case_result: dict, config: dict) -> dict:
    """
    M9 entry point. Screens CT and AD isoforms for NMD susceptibility.

    Returns:
        dict with keys:
          ct    : per-isoform NMD assessment dict
          ad    : per-isoform NMD assessment dict
          summary: str — one-line interpretation
          nmd_relevant: bool — True if AD isoform is NMD susceptible
                               (reduces effective protein output → loss-of-function)
    """
    ct_id = case_result.get("ct_transcript_id", "")
    ad_id = case_result.get("ad_transcript_id", "")

    sqanti_path = config.get("paths", {}).get("sqanti_classification", "")
    sqanti_rows = _parse_sqanti_classification(sqanti_path, [ct_id, ad_id])

    ct_result = _assess_isoform(ct_id, case_result, sqanti_rows, "ct_info")
    ad_result = _assess_isoform(ad_id, case_result, sqanti_rows, "ad_info")

    # Interpret combined result
    ct_nmd = ct_result["nmd_susceptible"]
    ad_nmd = ad_result["nmd_susceptible"]

    if ad_nmd is True and (ct_nmd is False or ct_nmd is None):
        summary = (
            "AD isoform is NMD susceptible while CT isoform escapes NMD. "
            "Isoform switch may reduce functional protein via NMD-mediated degradation "
            "(loss-of-function mechanism)."
        )
        nmd_relevant = True
    elif ad_nmd is False and ct_nmd is True:
        summary = (
            "CT isoform is NMD susceptible; AD isoform escapes NMD. "
            "AD switch may stabilize an otherwise degraded transcript "
            "(gain-of-expression mechanism)."
        )
        nmd_relevant = True
    elif ad_nmd is True and ct_nmd is True:
        summary = "Both isoforms are NMD susceptible; switch does not alter NMD status."
        nmd_relevant = False
    elif ad_nmd is False and ct_nmd is False:
        summary = "Both isoforms escape NMD; NMD not the primary mechanism."
        nmd_relevant = False
    else:
        summary = "NMD status could not be determined for one or both isoforms."
        nmd_relevant = False

    return {
        "ct": ct_result,
        "ad": ad_result,
        "summary": summary,
        "nmd_relevant": nmd_relevant,
    }
