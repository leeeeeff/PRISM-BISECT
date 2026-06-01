"""
M7: Genomic Sequence Validation (BISECT v1.1)

For cases with young LINE-1 elements overlapping exons, fetches the hg38
genomic sequence of each LINE-1 element via UCSC REST API, performs 6-frame
translation, and runs Smith-Waterman alignment against the AD isoform's
Pfam-annotated domain region.

Confirms that the protein domain originates from the retroelement without
RNA editing or sequence divergence.

Entry point: run(case_result, config) -> dict
"""

import time
import urllib.request
import json
from typing import Optional

try:
    from Bio.Align import PairwiseAligner, substitution_matrices
    from Bio.Seq import Seq
    _BIOPYTHON = True
except ImportError:
    _BIOPYTHON = False


def _ucsc_seq(chrom: str, start: int, end: int,
              genome: str = "hg38", retries: int = 3,
              delay: float = 1.0) -> Optional[str]:
    """Fetch genomic sequence from UCSC REST API (0-based half-open)."""
    url = (f"https://api.genome.ucsc.edu/getData/sequence?"
           f"genome={genome};chrom={chrom};start={start};end={end}")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read().decode())
            return data.get("dna", "").upper()
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
    return None


def _six_frame_translate(dna: str) -> dict:
    """Translate DNA in all 6 frames. Returns {'+1':str, '+2':str, '+3':str, '-1':str, ...}"""
    if not _BIOPYTHON:
        return {}
    frames = {}
    seq = Seq(dna)
    rev = seq.reverse_complement()
    for i in range(3):
        fwd = str(seq[i:].translate())
        rvs = str(rev[i:].translate())
        frames[f"+{i+1}"] = fwd
        frames[f"-{i+1}"] = rvs
    return frames


def _sw_align(query: str, target: str) -> dict:
    """Smith-Waterman local alignment with BLOSUM62. Returns alignment stats."""
    if not _BIOPYTHON or not query or not target:
        return {"score": 0, "pct_identity": 0.0, "coverage": 0.0,
                "n_identical": 0, "n_aligned": 0, "q_aligned": "",
                "t_aligned": ""}
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.mode = "local"

    try:
        alignments = aligner.align(query, target)
        best = next(iter(alignments))
    except StopIteration:
        return {"score": 0, "pct_identity": 0.0, "coverage": 0.0,
                "n_identical": 0, "n_aligned": 0, "q_aligned": "",
                "t_aligned": ""}

    q_coords = best.aligned[0]
    t_coords = best.aligned[1]

    if hasattr(q_coords, "tolist"):
        q_coords = q_coords.tolist()
        t_coords = t_coords.tolist()

    q_str = "".join(query[qs:qe] for qs, qe in q_coords)
    t_str = "".join(target[ts:te] for ts, te in t_coords)
    n_id = sum(a == b for a, b in zip(q_str, t_str))
    n_al = max(len(q_str), 1)
    q_cov = sum(qe - qs for qs, qe in q_coords)

    return {
        "score": float(best.score),
        "pct_identity": round(100 * n_id / n_al, 2),
        "coverage": round(100 * q_cov / max(len(query), 1), 2),
        "n_identical": n_id,
        "n_aligned": n_al,
        "q_aligned": q_str[:120],
        "t_aligned": t_str[:120],
    }


def run(case_result: dict, config: dict) -> dict:
    """
    M8 entry point. Only runs if ad_repeats contains young LINE-1 exon hits.

    Returns:
        dict with keys:
          skipped        : bool
          skip_reason    : str (if skipped)
          elements       : list of per-element validation results
          best_identity  : float (max pct_identity across all elements)
          conclusion     : str
    """
    result = {
        "skipped": False,
        "skip_reason": "",
        "elements": [],
        "best_identity": 0.0,
        "conclusion": "",
    }

    if not _BIOPYTHON:
        result["skipped"] = True
        result["skip_reason"] = "biopython not installed"
        return result

    # Check: does this case have young LINE-1 in exons?
    ad_repeats = case_result.get("ad_repeats", {})
    exon_hits = ad_repeats.get("exon_hits", {})
    young_hits = []
    for exon_id, hits in exon_hits.items():
        for h in hits:
            if h.get("is_young_l1", False):
                young_hits.append((exon_id, h))

    if not young_hits:
        result["skipped"] = True
        result["skip_reason"] = "no young LINE-1 elements in exons"
        return result

    # Get AD protein sequence and Pfam domains for query region
    ad_seq = case_result.get("ad_seq", {}).get("seq", "")
    ad_domains = case_result.get("ad_domains", [])
    ad_info = case_result.get("ad_info", {})
    cds_start = ad_info.get("cds_start_genomic")
    chrom = ad_info.get("chrom", "")
    strand = ad_info.get("strand", "+")

    if not ad_seq or not chrom:
        result["skipped"] = True
        result["skip_reason"] = "missing AD sequence or chromosome info"
        return result

    delay = config.get("repeatmasker_api", {}).get("rate_limit_delay", 1.0)

    # For each young L1 element, validate by 6-frame translation + alignment
    for exon_id, hit in young_hits:
        elem_name = hit["name"]
        elem_start = hit["start"]    # UCSC 0-based
        elem_end = hit["end"]
        elem_strand = hit["strand"]
        pct_div = hit["pct_divergence"]

        # Determine query: use the RVT_1 (or strongest Pfam) domain if available
        # else use full AD protein
        query_seq = ad_seq
        query_label = "full AD protein"
        for dom in ad_domains:
            if dom["domain"] in ("RVT_1", "RVT_2", "Kinesin", "WD40"):
                a_from = dom["ali_from"] - 1   # 0-indexed
                a_to   = dom["ali_to"]
                if a_to > a_from + 20:
                    query_seq = ad_seq[a_from:a_to]
                    query_label = f"{dom['domain']} aa {dom['ali_from']}–{dom['ali_to']}"
                    break

        # Fetch element genomic sequence
        dna = _ucsc_seq(chrom, elem_start, elem_end, delay=delay)
        time.sleep(delay)

        if not dna:
            result["elements"].append({
                "element": elem_name,
                "exon": exon_id,
                "pct_divergence": pct_div,
                "error": "UCSC fetch failed",
            })
            continue

        # Determine reading frame from CDS start if available
        frame_info = {}
        if cds_start is not None and strand == "+":
            offset_from_cds = elem_start - cds_start
            if offset_from_cds >= 0:
                phase = offset_from_cds % 3
                # phase=0 → start at elem[0], phase=1 → skip 2, phase=2 → skip 1
                trans_skip = (3 - phase) % 3
                frame_info = {
                    "offset_from_cds_bp": offset_from_cds,
                    "phase": phase,
                    "trans_skip_bp": trans_skip,
                    "frame_id": f"+{trans_skip + 1}",
                }
        elif cds_start is not None and strand == "-":
            # On minus strand: CDS start (higher coord) relative to element end
            offset_from_cds = cds_start - elem_end
            if offset_from_cds >= 0:
                phase = offset_from_cds % 3
                trans_skip = (3 - phase) % 3
                frame_info = {
                    "offset_from_cds_bp": offset_from_cds,
                    "phase": phase,
                    "trans_skip_bp": trans_skip,
                    "frame_id": f"-{trans_skip + 1}",
                }

        # 6-frame translation + alignment scoring
        frames = _six_frame_translate(dna)
        best_frame_id = None
        best_frame_score = -1
        best_aln = {}

        for fid, prot in frames.items():
            prot_clean = prot.split("*")[0]
            if len(prot_clean) < 10:
                continue
            aln = _sw_align(query_seq, prot_clean)
            if aln["score"] > best_frame_score:
                best_frame_score = aln["score"]
                best_frame_id = fid
                best_aln = aln
                best_prot = prot_clean

        # Also run alignment with CDS-derived frame if available
        cds_frame_aln = {}
        if frame_info:
            skip = frame_info.get("trans_skip_bp", 0)
            fid = frame_info["frame_id"]
            if fid.startswith("+"):
                prot_cds = str(Seq(dna[skip:]).translate()).split("*")[0]
            else:
                rev_dna = str(Seq(dna).reverse_complement())
                prot_cds = str(Seq(rev_dna[skip:]).translate()).split("*")[0]
            if prot_cds:
                cds_frame_aln = _sw_align(query_seq, prot_cds)
                cds_frame_aln["frame_id"] = fid
                cds_frame_aln["protein_length"] = len(prot_cds)

        elem_result = {
            "element": elem_name,
            "exon": exon_id,
            "coordinates": f"{chrom}:{elem_start}-{elem_end}",
            "strand": elem_strand,
            "pct_divergence": pct_div,
            "query_region": query_label,
            "query_length": len(query_seq),
            "dna_fetched_bp": len(dna),
            "best_6frame": {
                "frame_id": best_frame_id,
                "score": best_aln.get("score", 0),
                "pct_identity": best_aln.get("pct_identity", 0.0),
                "coverage": best_aln.get("coverage", 0.0),
                "n_identical": best_aln.get("n_identical", 0),
                "n_aligned": best_aln.get("n_aligned", 0),
                "q_aligned_preview": best_aln.get("q_aligned", "")[:60],
                "t_aligned_preview": best_aln.get("t_aligned", "")[:60],
            },
            "cds_frame_alignment": cds_frame_aln if cds_frame_aln else None,
        }
        result["elements"].append(elem_result)

        pct_id = best_aln.get("pct_identity", 0.0)
        if pct_id > result["best_identity"]:
            result["best_identity"] = pct_id

    # Build conclusion
    bi = result["best_identity"]
    if bi >= 99.0:
        result["conclusion"] = (
            f"CONFIRMED: {bi:.1f}% identity — domain sequence is directly encoded "
            f"by LINE-1 ORF2p without modification."
        )
    elif bi >= 80.0:
        result["conclusion"] = (
            f"LIKELY: {bi:.1f}% identity — high similarity to LINE-1 ORF2p; "
            f"minor divergence may reflect RNA editing or LINE-1 copy variation."
        )
    elif bi >= 50.0:
        result["conclusion"] = (
            f"PARTIAL: {bi:.1f}% identity — moderate similarity. "
            f"LINE-1 origin plausible but not directly confirmable from sequence alone."
        )
    elif bi > 0:
        result["conclusion"] = (
            f"WEAK: {bi:.1f}% identity — low protein-level similarity. "
            f"Genomic overlap suggests structural context; protein may be diverged or chimeric."
        )
    else:
        result["conclusion"] = "NO ALIGNMENT: unable to confirm LINE-1 protein origin."

    return result
