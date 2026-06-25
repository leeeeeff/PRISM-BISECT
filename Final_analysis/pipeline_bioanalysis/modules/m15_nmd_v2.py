"""
M15: NMD Calculator v2
Fixes M6's high None-return rate by computing PTC/EJC distance from exon coordinates
and protein sequence length instead of relying on transcript annotation.
"""
from __future__ import annotations
from typing import Optional


NMD_THRESHOLD_NT = 50  # PTC upstream of last EJC by >50 nt → NMD predicted


def run_m15(case: dict, config: dict) -> dict:
    """Calculate NMD susceptibility for both CT and AD isoforms."""
    ct_result = _calc_nmd(case, isoform="ct")
    ad_result = _calc_nmd(case, isoform="ad")

    nmd_switch = (
        not ct_result.get("nmd_predicted", False)
        and ad_result.get("nmd_predicted", False)
    )

    return {
        "m15_nmd_v2": {
            "ct": ct_result,
            "ad": ad_result,
            "nmd_switch": nmd_switch,
        }
    }


def _calc_nmd(case: dict, isoform: str) -> dict:
    info_key = f"{isoform}_info"
    seq_key  = f"{isoform}_seq"

    info = case.get(info_key) or {}
    exons_raw = info.get("exons") or []

    seq_val = case.get(seq_key)
    if isinstance(seq_val, dict):
        aa_seq = seq_val.get("seq") or ""
    elif isinstance(seq_val, str):
        aa_seq = seq_val
    else:
        aa_seq = ""

    aa_len = len(aa_seq)

    if not exons_raw or aa_len == 0:
        return {
            "status": "insufficient_data",
            "nmd_predicted": None,
            "nmd_confidence": "none",
            "notes": f"Missing {'exons' if not exons_raw else 'protein sequence'}",
        }

    exons = sorted([tuple(e) for e in exons_raw if len(e) >= 2], key=lambda x: x[0])
    strand = info.get("strand", "+")

    if strand == "-":
        exons = list(reversed(exons))

    # Build spliced mRNA coordinate system
    mrna_positions = []  # (genomic_start, genomic_end, mrna_start, mrna_end)
    mrna_offset = 0
    for ex in exons:
        ex_len = ex[1] - ex[0]
        mrna_positions.append((ex[0], ex[1], mrna_offset, mrna_offset + ex_len))
        mrna_offset += ex_len

    total_mrna_len = mrna_offset

    # EJC positions: ~20 nt downstream (on mRNA) of each exon-exon junction
    n_junctions = len(exons) - 1
    ejc_positions_mrna = []
    cumlen = 0
    for i, ex in enumerate(exons[:-1]):
        cumlen += ex[1] - ex[0]
        ejc_pos = cumlen - 20  # EJC deposited 20-24 nt upstream of junction on mRNA
        if ejc_pos > 0:
            ejc_positions_mrna.append(ejc_pos)

    if not ejc_positions_mrna:
        return {
            "status": "single_exon",
            "n_exon_junctions": 0,
            "nmd_predicted": False,
            "nmd_confidence": "high",
            "notes": "Single exon transcript; no EJC → NMD impossible",
        }

    # CDS end position on mRNA (estimated from aa length)
    cds_nt = aa_len * 3 + 3  # coding + stop codon
    # Assume CDS starts at mRNA position 0 (conservative) or detect 5' UTR
    # For now assume minimal 5'UTR (conservative for NMD prediction)
    ptc_mrna_pos = cds_nt  # stop codon position

    if ptc_mrna_pos > total_mrna_len:
        # CDS extends beyond known exons → annotation incomplete
        confidence = "low"
        ptc_mrna_pos = total_mrna_len  # clamp
    else:
        confidence = "medium"

    last_ejc = max(ejc_positions_mrna) if ejc_positions_mrna else 0

    # NMD rule: PTC >50 nt upstream of last EJC
    if ptc_mrna_pos < last_ejc:
        ptc_to_last_ejc = last_ejc - ptc_mrna_pos
        nmd_predicted = ptc_to_last_ejc > NMD_THRESHOLD_NT
    else:
        ptc_to_last_ejc = 0
        nmd_predicted = False

    return {
        "status": "computed",
        "n_exon_junctions": n_junctions,
        "n_ejc": len(ejc_positions_mrna),
        "ejc_positions_mrna": ejc_positions_mrna,
        "cds_nt_estimated": cds_nt,
        "ptc_mrna_pos": ptc_mrna_pos,
        "last_ejc_mrna_pos": last_ejc,
        "ptc_to_last_ejc_nt": ptc_to_last_ejc,
        "nmd_predicted": nmd_predicted,
        "nmd_confidence": confidence,
        "notes": (
            "PTC upstream of last EJC by " + str(ptc_to_last_ejc) + " nt → NMD"
            if nmd_predicted
            else "PTC downstream of or at last EJC → NMD not predicted"
        ),
    }
