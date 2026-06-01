"""
M9: Alternative Promoter Usage Detection (BISECT v2.0)

Identifies whether CT and AD isoforms originate from different promoters by:
1. Computing 5' TSS distance between isoform pairs
2. Querying ENCODE SCREEN cCRE database for regulatory element classification
3. Reclassifying M14 "transcriptional" mechanism to "alternative_promoter" when warranted

Three TSS classes:
  same_promoter     : TSS_diff < 100 bp (within capture noise)
  tss_shift         : 100 ≤ TSS_diff < 500 bp (same promoter, microexon/TSS heterogeneity)
  alt_promoter      : TSS_diff ≥ 500 bp (distinct promoter elements)

Evidence levels:
  strong     — both TSS in ENCODE SCREEN PLS (promoter-like sequence)
  moderate   — one TSS in PLS, other in pELS or no annotation
  correlative— large TSS_diff with no SCREEN support (novel/unannotated promoter)
  none       — same_promoter or tss_shift

Entry point: run(case_result, config) -> dict
"""
import json
import math
import os
import time
import urllib.request
import urllib.error
from typing import Optional


# ── TSS extraction ────────────────────────────────────────────────────────────

def _extract_tss_tts(exons: list, strand: str):
    """
    Extract TSS (5' end) and TTS (3' end) from exon coordinate list.
    Exons stored as [[start, end], ...] in genomic order (ascending).
    For minus strand, TSS = max(exon ends), TTS = min(exon starts).
    Returns (tss, tts) or (None, None).
    """
    if not exons:
        return None, None
    try:
        exons_int = [[int(e[0]), int(e[1])] for e in exons]
    except (ValueError, IndexError):
        return None, None

    if strand == '+':
        tss = exons_int[0][0]
        tts = max(e[1] for e in exons_int)
    else:
        tss = max(e[1] for e in exons_int)
        tts = exons_int[0][0]
    return tss, tts


def _tss_class(diff: int) -> str:
    if diff < 100:
        return "same_promoter"
    elif diff < 500:
        return "tss_shift"
    else:
        return "alt_promoter_candidate"


# ── ENCODE SCREEN cCRE API ────────────────────────────────────────────────────

_SCREEN_CCRE_URL = "https://api.screen.encodeproject.org/cgi-bin/bedmask/"

# cCRE type priority: higher index = stronger promoter evidence
_CCRE_PRIORITY = {
    "PLS":  4,  # Promoter-Like Sequence (strongest)
    "pELS": 3,  # proximal Enhancer-Like Sequence
    "dELS": 2,  # distal Enhancer-Like Sequence
    "CTCF": 1,  # CTCF-only (insulator)
    "DNase": 0,
}


def _query_screen(chrom: str, pos: int, window: int = 500,
                  genome: str = "GRCh38",
                  rate_delay: float = 1.0) -> list:
    """
    Query ENCODE SCREEN for cCREs overlapping [pos-window, pos+window].
    Returns list of {accession, ccre_class, start, end, zscore_atac} dicts.
    Falls back to empty list on any error.
    """
    start = max(0, pos - window)
    end   = pos + window
    # SCREEN REST API: POST JSON body
    payload = json.dumps({
        "assembly": genome,
        "coord_chrom": chrom,
        "coord_start": start,
        "coord_end": end,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            _SCREEN_CCRE_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        time.sleep(rate_delay)
        # Normalise response: SCREEN returns list of cCRE records
        if isinstance(data, list):
            results = []
            for item in data:
                ccre_class = item.get("pct") or item.get("ccRE_group") or item.get("group") or "unknown"
                results.append({
                    "accession": item.get("accession", ""),
                    "ccre_class": str(ccre_class).upper().strip(),
                    "start": item.get("start", start),
                    "end":   item.get("end", end),
                })
            return results
        return []
    except Exception:
        return []


def _best_ccre(hits: list) -> Optional[str]:
    """Return highest-priority cCRE class from a list of SCREEN hits."""
    if not hits:
        return None
    best = max(hits, key=lambda h: _CCRE_PRIORITY.get(h.get("ccre_class", ""), -1))
    return best.get("ccre_class")


# ── APA (poly-A) analysis ─────────────────────────────────────────────────────

def _apa_class(tts_diff: int) -> str:
    if tts_diff < 100:
        return "same_apa"
    elif tts_diff < 500:
        return "minor_apa"
    elif tts_diff < 5000:
        return "moderate_apa"
    else:
        return "major_apa"


def _utr_regulatory_motifs(seq: str) -> dict:
    """
    Scan 3'UTR sequence for key regulatory elements:
    - PAS (poly-A signal): AATAAA, ATTAAA and variants
    - AU-rich elements (ARE): AUUUA pentamer → ATTTA
    - miR-132 seed: AACAGT (key AD-relevant miRNA)
    - miR-21 seed:  TCAACA
    """
    import re
    if not seq:
        return {}
    seq_u = seq.upper()
    pas_variants = ["AATAAA", "ATTAAA", "AGTAAA", "TATAAA",
                    "CATAAA", "GATAAA", "ACTAAA", "AATATA"]
    pas_count = sum(seq_u.count(v) for v in pas_variants)
    are_count = seq_u.count("ATTTA")
    mir132_count = len(re.findall("AACAGT", seq_u))
    mir21_count  = len(re.findall("TCAACA", seq_u))
    return {
        "pas_count":    pas_count,
        "are_count":    are_count,
        "mir132_seeds": mir132_count,
        "mir21_seeds":  mir21_count,
    }


# ── Evidence scoring ──────────────────────────────────────────────────────────

def _promoter_evidence(tss_diff: int, ct_ccre: Optional[str],
                       ad_ccre: Optional[str]) -> str:
    """Assign evidence strength for alternative promoter conclusion."""
    if tss_diff < 500:
        return "none"
    ct_p = _CCRE_PRIORITY.get(ct_ccre or "", -1)
    ad_p = _CCRE_PRIORITY.get(ad_ccre or "", -1)
    if ct_p >= 4 and ad_p >= 4:
        return "strong"       # both PLS confirmed
    elif ct_p >= 3 or ad_p >= 3:
        return "moderate"     # at least one PLS
    else:
        return "correlative"  # large diff, no SCREEN support


# ── Main entry point ──────────────────────────────────────────────────────────

def run(case_result: dict, config: dict) -> dict:
    """
    M15: Alternative Promoter Usage

    Returns dict with:
      tss_diff_bp, tts_diff_bp, tss_class, apa_class
      ct_ccre, ad_ccre, promoter_evidence
      mechanism_reclassify (bool): True if M14 "transcriptional" should → "alternative_promoter"
      summary (str)
    """
    cfg = config.get("m15", {})
    use_screen  = cfg.get("use_screen_api", True)
    rate_delay  = cfg.get("rate_delay", 1.0)
    window      = cfg.get("ccre_window", 500)
    genome      = cfg.get("genome", "GRCh38")

    gene      = case_result.get("gene_name", "UNKNOWN")
    cell_type = case_result.get("cell_type", "")

    print(f"  [M9] {gene} ({cell_type}): Alternative promoter analysis")

    ct_info  = case_result.get("ct_info")  or {}
    ad_info  = case_result.get("ad_info")  or {}
    ct_exons = ct_info.get("exons", [])
    ad_exons = ad_info.get("exons", [])
    strand   = ct_info.get("strand") or ad_info.get("strand") or "+"
    chrom    = ct_info.get("chrom")  or ad_info.get("chrom",  "chr1")

    if not ct_exons or not ad_exons:
        return {"skipped": "missing_exon_coordinates", "mechanism_reclassify": False}

    # ── Step 1: TSS/TTS extraction ─────────────────────────────────────────
    ct_tss, ct_tts = _extract_tss_tts(ct_exons, strand)
    ad_tss, ad_tts = _extract_tss_tts(ad_exons, strand)

    if ct_tss is None or ad_tss is None:
        return {"skipped": "tss_extraction_failed", "mechanism_reclassify": False}

    tss_diff = abs(ct_tss - ad_tss)
    tts_diff = abs(ct_tts - ad_tts) if ct_tts and ad_tts else 0
    tss_cls  = _tss_class(tss_diff)
    apa_cls  = _apa_class(tts_diff)

    print(f"  [M9] TSS_diff={tss_diff:,} bp ({tss_cls}), TTS_diff={tts_diff:,} bp ({apa_cls})")

    # ── Step 2: ENCODE SCREEN query ────────────────────────────────────────
    ct_ccre = ad_ccre = None
    ct_screen_hits = ad_screen_hits = []

    if use_screen and tss_diff >= 500:
        print(f"  [M9] Querying ENCODE SCREEN for CT TSS ({chrom}:{ct_tss})")
        ct_screen_hits = _query_screen(chrom, ct_tss, window, genome, rate_delay)
        ct_ccre = _best_ccre(ct_screen_hits)

        print(f"  [M9] Querying ENCODE SCREEN for AD TSS ({chrom}:{ad_tss})")
        ad_screen_hits = _query_screen(chrom, ad_tss, window, genome, rate_delay)
        ad_ccre = _best_ccre(ad_screen_hits)

        print(f"  [M9] CT cCRE: {ct_ccre or 'none'}, AD cCRE: {ad_ccre or 'none'}")
    elif tss_diff >= 500:
        print("  [M9] SCREEN API disabled — skipping cCRE lookup")

    # ── Step 3: Evidence and reclassification ──────────────────────────────
    prom_evidence = _promoter_evidence(tss_diff, ct_ccre, ad_ccre)
    reclassify = (tss_diff >= 500)

    # ── Step 4: APA motif scan (if 3' end differs) ────────────────────────
    apa_motifs = {}
    if tts_diff >= 500:
        ct_seq = (case_result.get("ct_seq") or {}).get("sequence", "")
        ad_seq = (case_result.get("ad_seq") or {}).get("sequence", "")
        if ct_seq or ad_seq:
            apa_motifs = {
                "ct_3utr": _utr_regulatory_motifs(ct_seq[-300:] if ct_seq else ""),
                "ad_3utr": _utr_regulatory_motifs(ad_seq[-300:] if ad_seq else ""),
            }

    # ── Step 5: Summary ────────────────────────────────────────────────────
    summary = _build_summary(gene, tss_diff, tss_cls, apa_cls, ct_ccre, ad_ccre,
                              prom_evidence, reclassify)
    print(f"  [M9] Conclusion: {tss_cls}, evidence={prom_evidence}")

    return {
        "tss_diff_bp":        tss_diff,
        "tts_diff_bp":        tts_diff,
        "ct_tss":             ct_tss,
        "ad_tss":             ad_tss,
        "tss_class":          tss_cls,
        "apa_class":          apa_cls,
        "ct_ccre":            ct_ccre,
        "ad_ccre":            ad_ccre,
        "ct_screen_hits":     ct_screen_hits[:5],
        "ad_screen_hits":     ad_screen_hits[:5],
        "promoter_evidence":  prom_evidence,
        "mechanism_reclassify": reclassify,
        "apa_motifs":         apa_motifs,
        "summary":            summary,
    }


def _build_summary(gene: str, tss_diff: int, tss_cls: str, apa_cls: str,
                   ct_ccre: Optional[str], ad_ccre: Optional[str],
                   evidence: str, reclassify: bool) -> str:
    if not reclassify:
        return (f"{gene}: CT and AD isoforms share the same promoter region "
                f"(TSS_diff={tss_diff:,} bp, {tss_cls}). "
                f"Mechanism classification unchanged from M14.")

    ccre_str = ""
    if ct_ccre or ad_ccre:
        ccre_str = (f" ENCODE SCREEN annotation: CT TSS = {ct_ccre or 'unannotated'}, "
                    f"AD TSS = {ad_ccre or 'unannotated'}.")

    apa_note = ""
    if apa_cls in ("moderate_apa", "major_apa"):
        apa_note = (f" The 3' end also differs substantially (TTS class={apa_cls}), "
                    f"suggesting concurrent APA or distinct terminal exon usage.")

    return (
        f"{gene}: CT and AD isoforms likely originate from distinct promoters "
        f"(TSS_diff={tss_diff:,} bp).{ccre_str}{apa_note} "
        f"Evidence strength: {evidence}. "
        f"M14 mechanism_type reclassified to 'alternative_promoter'. "
        f"Caution: promoter identity requires CAGE-seq or scATAC-seq validation."
    )
