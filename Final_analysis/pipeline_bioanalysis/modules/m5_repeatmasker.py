"""
M5: RepeatMasker / Retroelement Annotation
Queries UCSC REST API for repeat elements in genomic regions of interest.
"""
import json
import time
import urllib.request
import urllib.error
from typing import Optional


def query_repeatmasker(chrom: str, start: int, end: int, config: dict,
                        retries: int = 3) -> list[dict]:
    """
    Query UCSC RepeatMasker track via REST API.
    Returns list of repeat hits in the region.
    """
    cfg = config.get("repeatmasker", {})
    genome = cfg.get("genome", "hg38")
    track = cfg.get("track", "rmsk")
    base_url = cfg.get("ucsc_api", "https://api.genome.ucsc.edu/getData/track")

    url = f"{base_url}?genome={genome};track={track};chrom={chrom};start={start};end={end}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DIFFUSE-pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
            return data.get(track, [])
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [M5] RepeatMasker API failed for {chrom}:{start}-{end}: {e}")
                return []


def annotate_exons(exons: list[tuple[int, int]], chrom: str, cds_start: Optional[int],
                   cds_end: Optional[int], config: dict) -> dict:
    """
    For each exon and CDS, query RepeatMasker and return annotated hits.
    Returns structured repeat annotation.
    """
    if not exons:
        return {"error": "no exons provided"}

    cfg = config.get("repeatmasker", {})
    target_classes = set(cfg.get("target_classes", ["LINE", "SINE", "LTR", "DNA"]))
    young_thresh = cfg.get("young_l1_max_div", 15.0)
    extend = cfg.get("window_extend", 5000)

    # Query full locus (all exons + extend)
    locus_start = min(e[0] for e in exons) - extend
    locus_end = max(e[1] for e in exons) + extend

    all_hits = query_repeatmasker(chrom, locus_start, locus_end, config)

    # Filter to target classes
    filtered = [h for h in all_hits if h.get("repClass") in target_classes]

    # Annotate each hit with exon overlap
    cds_hits = []
    exon_hits = {f"E{i+1}": [] for i in range(len(exons))}

    for h in filtered:
        hs = h.get("genoStart", 0)
        he = h.get("genoEnd", 0)
        pct_div = h.get("milliDiv", 0) / 10
        is_young = (h.get("repClass") == "LINE" and pct_div < young_thresh)

        hit_info = {
            "name": h.get("repName", "?"),
            "class": h.get("repClass", "?"),
            "family": h.get("repFamily", "?"),
            "strand": h.get("strand", "?"),
            "start": hs,
            "end": he,
            "pct_divergence": round(pct_div, 2),
            "sw_score": h.get("swScore", 0),
            "is_young_l1": is_young,
        }

        # CDS overlap
        if cds_start and cds_end and hs < cds_end and he > cds_start:
            ov = min(he, cds_end) - max(hs, cds_start)
            hit_info["cds_overlap_bp"] = ov
            cds_hits.append(hit_info)

        # Exon overlaps
        for i, (es, ee) in enumerate(exons):
            if hs < ee and he > es:
                ov = min(he, ee) - max(hs, es)
                ei = hit_info.copy()
                ei["exon_overlap_bp"] = ov
                exon_hits[f"E{i+1}"].append(ei)

    # Young L1 summary for CDS
    young_l1_cds = [h for h in cds_hits if h.get("is_young_l1")]

    return {
        "locus_queried": f"{chrom}:{locus_start}-{locus_end}",
        "total_repeat_hits": len(filtered),
        "cds_overlap_hits": cds_hits,
        "cds_young_l1_hits": young_l1_cds,
        "exon_hits": {k: v for k, v in exon_hits.items() if v},
        "has_l1_in_cds": len(young_l1_cds) > 0,
        "summary": {
            "LINE": len([h for h in filtered if h.get("repClass") == "LINE"]),
            "SINE": len([h for h in filtered if h.get("repClass") == "SINE"]),
            "LTR": len([h for h in filtered if h.get("repClass") == "LTR"]),
            "DNA": len([h for h in filtered if h.get("repClass") == "DNA"]),
        }
    }
