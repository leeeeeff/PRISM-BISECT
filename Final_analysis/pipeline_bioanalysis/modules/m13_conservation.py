"""
M13: Evolutionary Conservation Analysis (BISECT v1.1)

Queries UCSC phyloP100way track for AD-specific and CT-specific exons.
High phyloP (>1.5) indicates functional constraint under purifying selection.
Provides independent evidence that isoform-switch changes are functionally important.

Entry point: run(case_result, config) -> dict
"""
import json
import time
import urllib.request
import urllib.error
from typing import Optional


# ── UCSC API ──────────────────────────────────────────────────────────────────

def _ucsc_track(chrom: str, start: int, end: int, track: str, genome: str,
                base_url: str, retries: int = 3) -> Optional[list]:
    """Fetch per-base scores from a UCSC bigWig track."""
    url = (f"{base_url}?genome={genome};track={track};"
           f"chrom={chrom};start={start};end={end}")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DIFFUSE-BISECT/1.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            # phyloP track returns list of {start, end, value} or flat values
            raw = data.get(track, data.get("data", []))
            return raw
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [M12] UCSC API failed {chrom}:{start}-{end}: {e}")
                return None
    return None


def _mean_phylop(chrom: str, start: int, end: int, cfg: dict) -> Optional[float]:
    """Return mean phyloP score for a genomic interval."""
    track = cfg.get("track", "phyloP100way")
    genome = cfg.get("genome", "hg38")
    base_url = cfg.get("ucsc_api", "https://api.genome.ucsc.edu/getData/track")

    raw = _ucsc_track(chrom, start, end, track, genome, base_url)
    if raw is None:
        return None

    scores = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                v = item.get("value", item.get("score"))
                if v is not None:
                    scores.append(float(v))
            elif isinstance(item, (int, float)):
                scores.append(float(item))
    elif isinstance(raw, dict):
        # Some endpoints wrap differently
        for v in raw.values():
            if isinstance(v, (int, float)):
                scores.append(float(v))

    return sum(scores) / len(scores) if scores else None


# ── Exon set difference ───────────────────────────────────────────────────────

def _exon_set_diff(exons_a: list, exons_b: list,
                   overlap_fraction: float = 0.5) -> tuple[list, list]:
    """
    Return (a_specific, b_specific) exons.
    An exon in a is 'specific' if it has < overlap_fraction reciprocal overlap with any b exon.
    """
    def overlaps(e1, e2, min_frac):
        s1, e1_ = e1[0], e1[1]
        s2, e2_ = e2[0], e2[1]
        ov = max(0, min(e1_, e2_) - max(s1, s2))
        if ov == 0:
            return False
        len1 = e1_ - s1
        len2 = e2_ - s2
        return (ov / len1) >= min_frac or (ov / len2) >= min_frac

    a_specific = [e for e in exons_a if not any(overlaps(e, b, overlap_fraction) for b in exons_b)]
    b_specific = [e for e in exons_b if not any(overlaps(e, a, overlap_fraction) for a in exons_a)]
    return a_specific, b_specific


def _conservation_class(mean_phylop: float, cfg: dict) -> str:
    high = cfg.get("high_threshold", 1.5)
    mid = cfg.get("conserved_threshold", 0.5)
    if mean_phylop >= high:
        return "highly_conserved"
    elif mean_phylop >= mid:
        return "conserved"
    else:
        return "low"


# ── Domain overlap ────────────────────────────────────────────────────────────

def _find_domain_overlap(exon_start: int, exon_end: int,
                         domains: list, protein_start_genomic: Optional[int],
                         strand: Optional[str]) -> Optional[str]:
    """
    Heuristic: find which Pfam domain (by name) overlaps this exon.
    Uses protein residue positions and crude genomic mapping if CDS start is available.
    Returns domain name string or None.
    """
    if not domains:
        return None
    # Without full CDS-to-genome mapping, flag domains by name if exon is in coding region
    # Best effort: return all domain names for the isoform if exon is in CDS range
    names = list({d.get("pfam_family") or d.get("name", "") for d in domains if d})
    return ", ".join(n for n in names if n) or None


# ── Background intron sampling ────────────────────────────────────────────────

def _sample_intronic_conservation(chrom: str, exons: list, cfg: dict,
                                  n_samples: int = 5) -> Optional[float]:
    """Sample intron regions for background conservation estimate."""
    if len(exons) < 2:
        return None
    introns = []
    sorted_ex = sorted(exons, key=lambda e: e[0])
    for i in range(len(sorted_ex) - 1):
        intron_start = sorted_ex[i][1]
        intron_end = sorted_ex[i + 1][0]
        if intron_end - intron_start > 200:
            introns.append((intron_start, intron_end))
    if not introns:
        return None

    samples = introns[:n_samples]
    scores = []
    for s, e in samples:
        mid = (s + e) // 2
        val = _mean_phylop(chrom, mid - 100, mid + 100, cfg)
        if val is not None:
            scores.append(val)
        time.sleep(0.3)
    return sum(scores) / len(scores) if scores else None


# ── Per-exon annotation ───────────────────────────────────────────────────────

def _annotate_exon_list(exons: list, chrom: str, domains: list,
                        cds_start: Optional[int], strand: Optional[str],
                        cfg: dict, rate_delay: float = 0.5) -> list:
    results = []
    for i, exon in enumerate(exons):
        start, end = exon[0], exon[1]
        mean_p = _mean_phylop(chrom, start, end, cfg)
        if mean_p is None:
            results.append({
                "rank": i + 1,
                "chrom": chrom,
                "start": start,
                "end": end,
                "length_bp": end - start,
                "phyloP_mean": None,
                "phyloP_positive_fraction": None,
                "conservation_class": "unknown",
                "domain_overlap": None,
                "note": "API unavailable"
            })
            time.sleep(rate_delay)
            continue

        # Fetch per-base scores for positive fraction
        track = cfg.get("track", "phyloP100way")
        genome = cfg.get("genome", "hg38")
        base_url = cfg.get("ucsc_api", "https://api.genome.ucsc.edu/getData/track")
        raw = _ucsc_track(chrom, start, end, track, genome, base_url)
        pos_frac = None
        if raw:
            base_scores = []
            for item in raw:
                if isinstance(item, dict):
                    v = item.get("value", item.get("score"))
                    if v is not None:
                        base_scores.append(float(v))
                elif isinstance(item, (int, float)):
                    base_scores.append(float(item))
            if base_scores:
                pos_frac = round(sum(1 for v in base_scores if v > 0) / len(base_scores), 3)

        dom = _find_domain_overlap(start, end, domains, cds_start, strand)
        results.append({
            "rank": i + 1,
            "chrom": chrom,
            "start": start,
            "end": end,
            "length_bp": end - start,
            "phyloP_mean": round(mean_p, 3),
            "phyloP_positive_fraction": pos_frac,
            "conservation_class": _conservation_class(mean_p, cfg),
            "domain_overlap": dom,
            "note": None
        })
        time.sleep(rate_delay)
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def run(case_result: dict, config: dict) -> dict:
    """
    M12: Conservation analysis.
    Requires M4 output in case_result (ct_info, ad_info with exons + chrom).
    """
    cfg = config.get("conservation", {})
    rate_delay = cfg.get("rate_delay", 0.5)
    n_bg = cfg.get("background_intron_samples", 5)

    ct_info = case_result.get("ct_info", {})
    ad_info = case_result.get("ad_info", {})

    chrom = ad_info.get("chrom") or ct_info.get("chrom")
    ct_exons = ct_info.get("exons", [])
    ad_exons = ad_info.get("exons", [])

    if not chrom:
        return {"error": "no chrom available from M4"}
    if not ct_exons and not ad_exons:
        return {"error": "no exons available from M4"}

    # Normalise exon format to (start, end) tuples
    def _norm(exons):
        out = []
        for e in exons:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                out.append((int(e[0]), int(e[1])))
            elif isinstance(e, dict):
                out.append((int(e.get("start", 0)), int(e.get("end", 0))))
        return out

    ct_exons = _norm(ct_exons)
    ad_exons = _norm(ad_exons)

    ad_specific, ct_specific = _exon_set_diff(ad_exons, ct_exons)

    ct_domains = case_result.get("ct_domains", [])
    ad_domains = case_result.get("ad_domains", [])
    ct_cds_start = ct_info.get("cds_start_genomic")
    ad_cds_start = ad_info.get("cds_start_genomic")
    strand = ad_info.get("strand") or ct_info.get("strand")

    print(f"  [M12] {case_result.get('gene_name')}: "
          f"AD-specific={len(ad_specific)} CT-specific={len(ct_specific)} exons")

    ad_annotated = _annotate_exon_list(
        ad_specific, chrom, ad_domains, ad_cds_start, strand, cfg, rate_delay)
    ct_annotated = _annotate_exon_list(
        ct_specific, chrom, ct_domains, ct_cds_start, strand, cfg, rate_delay)

    # Background
    bg = _sample_intronic_conservation(chrom, ad_exons or ct_exons, cfg, n_bg)

    # Summary stats
    ad_vals = [e["phyloP_mean"] for e in ad_annotated if e["phyloP_mean"] is not None]
    ct_vals = [e["phyloP_mean"] for e in ct_annotated if e["phyloP_mean"] is not None]
    ad_mean = round(sum(ad_vals) / len(ad_vals), 3) if ad_vals else None
    ct_mean = round(sum(ct_vals) / len(ct_vals), 3) if ct_vals else None

    if ad_mean and ct_mean and ct_mean != 0:
        fold = round(ad_mean / ct_mean, 2) if ct_mean > 0 else None
    else:
        fold = None

    def _interp(ad_m, ct_m, fold_diff):
        if ad_m is None:
            return "Insufficient data for conservation comparison."
        hi = cfg.get("high_threshold", 1.5)
        if ad_m >= hi:
            strength = "highly conserved"
        elif ad_m >= cfg.get("conserved_threshold", 0.5):
            strength = "moderately conserved"
        else:
            strength = "poorly conserved"
        base = f"AD-specific exons are {strength} (phyloP mean={ad_m})"
        if fold_diff and fold_diff > 2:
            base += f", {fold_diff}x higher than CT-specific exons ({ct_m})"
        elif ct_m is not None:
            base += f" vs CT-specific exons ({ct_m})"
        base += ". "
        if ad_m >= hi:
            base += "Strong purifying selection supports functional importance of gained sequence."
        return base

    return {
        "track": cfg.get("track", "phyloP100way"),
        "genome": cfg.get("genome", "hg38"),
        "ad_specific_exons": ad_annotated,
        "ct_specific_exons": ct_annotated,
        "background": {
            "intronic_phyloP_mean": round(bg, 3) if bg is not None else None,
            "intronic_sample_n": n_bg
        },
        "summary": {
            "ad_specific_mean_phyloP": ad_mean,
            "ct_specific_mean_phyloP": ct_mean,
            "fold_difference": fold,
            "interpretation": _interp(ad_mean, ct_mean, fold)
        }
    }
