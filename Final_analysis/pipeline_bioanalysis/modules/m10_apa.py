"""
M10: Alternative Polyadenylation (APA) Analysis (BISECT v2.0)

Detects 3'-end differences between CT and AD isoforms and evaluates
their regulatory consequences:
  1. 3'UTR length change (from TTS coordinate diff)
  2. PolyASite 2.0 annotation of poly-A cluster identity
  3. cis-regulatory motif scan in the differential 3'UTR region:
       - PAS variants (AATAAA and 7 alternatives)
       - AU-rich elements (ARE): ATTTA pentamer, TATTTAT heptamer
       - miR-132 seed (AACAGT): major AD-dysregulated miRNA
       - miR-21 seed  (TCAACA): neuroinflammation
       - miR-9 seed   (CTTTGG): AD enriched
  4. NMD 3'UTR flag: stop codon to last EEJ distance change

APA classes (by TTS_diff):
  same_apa      < 100 bp   (technical noise)
  minor_apa     100–500 bp
  moderate_apa  500–5 kb   (3'UTR trimming/extension)
  major_apa     > 5 kb     (alternative terminal exon / distant poly-A)

Entry point: run(case_result, config) -> dict
"""
import json
import math
import os
import re
import time
import urllib.request
import urllib.error
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

# Poly-A Signal variants (Beaudoing et al. 2000; Gruber et al. 2016)
_PAS_VARIANTS = [
    "AATAAA",  # canonical (~60% of all poly-A sites)
    "ATTAAA",  # 2nd most common (~15%)
    "AGTAAA", "TATAAA", "CATAAA", "GATAAA",
    "AATATA", "AATACA",
]

# AU-rich element core motifs (Shaw & Kamen 1986; Bakheet et al. 2006)
_ARE_MOTIFS = [
    ("ATTTA",    "class_I_pentamer"),
    ("TATTTAT",  "class_II_heptamer"),
    ("TTATTTATT","class_III_nonanmer"),
]

# miRNA seeds relevant to AD (6-mer seed match in 3'UTR)
# Seed = positions 2–7 of mature miRNA (reverse complement of target)
_MIRNA_SEEDS = {
    "miR-132": "AACAGT",   # Hebert et al. 2008 Nat Neurosci; repressed in AD
    "miR-21":  "TCAACA",   # elevated in AD, targets SPRY1/PDCD4
    "miR-9":   "CTTTGG",   # elevated in AD, targets REST/BACE1
    "miR-107": "GCAGCAG",  # CDK6 regulation, reduced in AD
    "miR-34a": "ACTGCC",   # p53 pathway, elevated in AD
}

# PolyASite 2.0 REST API (EMBL-EBI hosted — stable)
_POLYASITE_URL = "https://polyasite.unibas.ch/api/v2/clusters/search/"


# ── TTS extraction (reuse from M15 logic) ────────────────────────────────────

def _extract_tts(exons: list, strand: str) -> Optional[int]:
    if not exons:
        return None
    try:
        exons_int = [[int(e[0]), int(e[1])] for e in exons]
    except (ValueError, IndexError):
        return None
    if strand == '+':
        return max(e[1] for e in exons_int)
    else:
        return min(e[0] for e in exons_int)


def _extract_tss(exons: list, strand: str) -> Optional[int]:
    if not exons:
        return None
    try:
        exons_int = [[int(e[0]), int(e[1])] for e in exons]
    except (ValueError, IndexError):
        return None
    if strand == '+':
        return exons_int[0][0]
    else:
        return max(e[1] for e in exons_int)


def _apa_class(diff: int) -> str:
    if diff < 100:
        return "same_apa"
    elif diff < 500:
        return "minor_apa"
    elif diff < 5000:
        return "moderate_apa"
    else:
        return "major_apa"


# ── PolyASite 2.0 API ─────────────────────────────────────────────────────────

def _query_polyasite(chrom: str, pos: int, strand: str,
                     window: int = 500,
                     rate_delay: float = 1.0) -> list:
    """
    Query PolyASite 2.0 for annotated poly-A clusters near genomic position.
    Returns list of {site_id, coord, score, tpm} dicts.
    Docs: https://polyasite.unibas.ch/api/
    """
    start = max(0, pos - window)
    end   = pos + window
    strand_enc = "%2B" if strand == "+" else "-"
    chrom_clean = chrom.replace("chr", "")  # PolyASite uses "1", not "chr1"

    # GET request: /api/v2/clusters/search/?chromosome=1&start=X&end=Y&strand=+
    url = (f"{_POLYASITE_URL}"
           f"?chromosome={chrom_clean}&start={start}&end={end}"
           f"&strand={strand_enc}&species=9606")

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        time.sleep(rate_delay)

        # PolyASite returns {"count": N, "results": [...]}
        results = data.get("results", []) if isinstance(data, dict) else []
        sites = []
        for r in results:
            sites.append({
                "site_id":    r.get("id", ""),
                "chrom":      r.get("chromosome", chrom),
                "coord":      r.get("representative_coordinate", pos),
                "strand":     r.get("strand", strand),
                "score":      r.get("pas_score", 0),
                "tpm":        r.get("tpm", 0),
                "pas_signal": r.get("pas_signal", ""),
            })
        return sites
    except Exception:
        return []


# ── 3'UTR sequence motif scanning ────────────────────────────────────────────

def _scan_3utr(seq: str, label: str = "") -> dict:
    """
    Scan C-terminal region of protein sequence's mRNA proxy (protein 3' end ≈
    mRNA CDS end; we scan the last N aa of the protein sequence as a proxy
    since we don't have the UTR sequence directly).

    NOTE: For more accurate results, the actual 3'UTR genomic sequence should be
    used. Here we use the available protein sequence tail as a partial proxy.
    When genome_fa is available, the actual 3'UTR sequence should be fetched.
    """
    if not seq:
        return {"available": False}

    seq_u = seq.upper()
    result = {"available": True, "length": len(seq_u)}

    # PAS variants
    pas_hits = {}
    total_pas = 0
    for v in _PAS_VARIANTS:
        cnt = seq_u.count(v)
        if cnt:
            pas_hits[v] = cnt
            total_pas += cnt
    result["pas_count"] = total_pas
    result["pas_hits"]  = pas_hits

    # ARE motifs
    are_hits = {}
    for motif, cls in _ARE_MOTIFS:
        cnt = len(re.findall(motif, seq_u))
        if cnt:
            are_hits[cls] = cnt
    result["are_count"] = sum(are_hits.values())
    result["are_hits"]  = are_hits

    # miRNA seeds
    mir_hits = {}
    for mir, seed in _MIRNA_SEEDS.items():
        cnt = len(re.findall(seed, seq_u))
        if cnt:
            mir_hits[mir] = cnt
    result["mirna_seeds"] = mir_hits

    return result


def _scan_genomic_3utr(genome_fa: str, fai_idx: dict,
                        chrom: str, strand: str,
                        tts: int, window: int = 2000) -> dict:
    """
    Fetch actual 3'UTR genomic sequence (TTS ± window) and scan for motifs.
    Returns scan dict with genomic=True flag.
    """
    if not genome_fa or not fai_idx or tts is None:
        return {"available": False, "reason": "no_genome_or_tts"}

    # 3'UTR: downstream of TTS
    if strand == '+':
        utr_start, utr_end = tts, tts + window
    else:
        utr_start, utr_end = max(0, tts - window), tts

    # Import fetch function from m14 if available
    try:
        from modules.m8_regulatory_context import _fetch_seq, _revcomp
        seq = _fetch_seq(genome_fa, fai_idx, chrom, utr_start, utr_end)
        if seq and strand == '-':
            seq = _revcomp(seq)
    except Exception:
        return {"available": False, "reason": "fetch_failed"}

    if not seq:
        return {"available": False, "reason": "no_sequence_returned"}

    result = _scan_3utr(seq, label="genomic_3utr")
    result["genomic"] = True
    result["coords"]  = f"{chrom}:{utr_start}-{utr_end}"
    result["seq_len"] = len(seq)
    return result


# ── 3'UTR differential analysis ───────────────────────────────────────────────

def _compare_utr_scans(ct_scan: dict, ad_scan: dict) -> dict:
    """
    Compare CT and AD 3'UTR motif scans and flag regulatory changes.
    Returns dict of gained/lost regulatory elements.
    """
    if not ct_scan.get("available") or not ad_scan.get("available"):
        return {"comparable": False}

    changes = {"comparable": True, "gains": {}, "losses": {}}

    # PAS: more sites = more stable canonical termination
    ct_pas = ct_scan.get("pas_count", 0)
    ad_pas = ad_scan.get("pas_count", 0)
    if ad_pas > ct_pas:
        changes["gains"]["PAS_sites"] = ad_pas - ct_pas
    elif ct_pas > ad_pas:
        changes["losses"]["PAS_sites"] = ct_pas - ad_pas

    # ARE: more = less stable mRNA
    ct_are = ct_scan.get("are_count", 0)
    ad_are = ad_scan.get("are_count", 0)
    if ad_are > ct_are:
        changes["gains"]["ARE_destabilizing"] = ad_are - ct_are
    elif ct_are > ad_are:
        changes["losses"]["ARE_destabilizing"] = ct_are - ad_are

    # miRNA seeds: gained sites = more silencing
    ct_mir = ct_scan.get("mirna_seeds", {})
    ad_mir = ad_scan.get("mirna_seeds", {})
    all_mir = set(list(ct_mir.keys()) + list(ad_mir.keys()))
    for mir in all_mir:
        diff = ad_mir.get(mir, 0) - ct_mir.get(mir, 0)
        if diff > 0:
            changes["gains"][f"{mir}_seed"] = diff
        elif diff < 0:
            changes["losses"][f"{mir}_seed"] = abs(diff)

    # Stability prediction
    stability_change = "unknown"
    if changes["gains"].get("ARE_destabilizing", 0) > 2:
        stability_change = "AD_less_stable"
    elif changes["losses"].get("ARE_destabilizing", 0) > 2:
        stability_change = "AD_more_stable"
    elif changes["gains"].get("miR-132_seed", 0) > 0:
        stability_change = "AD_suppressed_in_CT_neurons"
    elif changes["losses"].get("miR-132_seed", 0) > 0:
        stability_change = "AD_escapes_miR-132_repression"
    changes["predicted_stability"] = stability_change

    return changes


# ── Evidence strength ─────────────────────────────────────────────────────────

def _apa_evidence(apa_cls: str, ct_sites: list, ad_sites: list,
                  utr_changes: dict) -> str:
    """
    strong     — confirmed PolyASite cluster at both TTS positions
    moderate   — confirmed at one TTS, or major_apa + PAS motif change
    correlative— large TTS diff without PolyASite annotation
    none       — same_apa or minor_apa
    """
    if apa_cls in ("same_apa", "minor_apa"):
        return "none"

    has_ct_polyasite = bool(ct_sites)
    has_ad_polyasite = bool(ad_sites)

    if has_ct_polyasite and has_ad_polyasite:
        return "strong"
    elif has_ct_polyasite or has_ad_polyasite:
        return "moderate"
    elif apa_cls == "major_apa" and utr_changes.get("comparable"):
        pas_changed = ("PAS_sites" in utr_changes.get("gains", {}) or
                       "PAS_sites" in utr_changes.get("losses", {}))
        return "moderate" if pas_changed else "correlative"
    else:
        return "correlative"


# ── Main entry point ──────────────────────────────────────────────────────────

def run(case_result: dict, config: dict) -> dict:
    """
    M10: Alternative Polyadenylation

    Returns dict with:
      tts_diff_bp, apa_class, ct_polyasite, ad_polyasite
      ct_utr_scan, ad_utr_scan, utr_changes
      apa_evidence, summary
    """
    cfg = config.get("m16", {})
    use_polyasite = cfg.get("use_polyasite_api", True)
    rate_delay    = cfg.get("rate_delay", 1.0)
    window        = cfg.get("polyasite_window", 300)
    utr_window    = cfg.get("utr_scan_window", 2000)
    genome_fa     = cfg.get("genome_fa") or config.get("m14", {}).get("genome_fa", "")

    gene      = case_result.get("gene_name", "UNKNOWN")
    cell_type = case_result.get("cell_type", "")

    print(f"  [M10] {gene} ({cell_type}): APA analysis")

    ct_info  = case_result.get("ct_info")  or {}
    ad_info  = case_result.get("ad_info")  or {}
    ct_exons = ct_info.get("exons", [])
    ad_exons = ad_info.get("exons", [])
    strand   = ct_info.get("strand") or ad_info.get("strand") or "+"
    chrom    = ct_info.get("chrom")  or ad_info.get("chrom", "chr1")

    if not ct_exons or not ad_exons:
        return {"skipped": "missing_exon_coordinates"}

    # ── Step 1: TTS extraction ─────────────────────────────────────────────
    ct_tts = _extract_tts(ct_exons, strand)
    ad_tts = _extract_tts(ad_exons, strand)

    if ct_tts is None or ad_tts is None:
        return {"skipped": "tts_extraction_failed"}

    tts_diff = abs(ct_tts - ad_tts)
    apa_cls  = _apa_class(tts_diff)

    print(f"  [M10] TTS_diff={tts_diff:,} bp ({apa_cls})")

    # ── Step 2: PolyASite 2.0 lookup ──────────────────────────────────────
    ct_polyasite = ad_polyasite = []

    if use_polyasite and apa_cls not in ("same_apa",):
        print(f"  [M10] Querying PolyASite 2.0 for CT TTS ({chrom}:{ct_tts})")
        ct_polyasite = _query_polyasite(chrom, ct_tts, strand, window, rate_delay)

        if apa_cls not in ("minor_apa",):
            print(f"  [M10] Querying PolyASite 2.0 for AD TTS ({chrom}:{ad_tts})")
            ad_polyasite = _query_polyasite(chrom, ad_tts, strand, window, rate_delay)

        print(f"  [M10] CT poly-A sites: {len(ct_polyasite)}, "
              f"AD poly-A sites: {len(ad_polyasite)}")
    elif apa_cls not in ("same_apa",):
        print("  [M10] PolyASite API disabled — skipping lookup")

    # ── Step 3: 3'UTR motif scan ───────────────────────────────────────────
    # Try genomic sequence first; fall back to protein sequence tail
    ct_utr_scan = ad_utr_scan = {}

    fai_path = genome_fa + ".fai" if genome_fa else ""
    fai_idx  = {}
    if genome_fa and os.path.exists(fai_path):
        try:
            from modules.m8_regulatory_context import _load_fai
            fai_idx = _load_fai(fai_path)
        except Exception:
            pass

    if apa_cls not in ("same_apa", "minor_apa"):
        if fai_idx:
            ct_utr_scan = _scan_genomic_3utr(genome_fa, fai_idx, chrom, strand,
                                              ct_tts, utr_window)
            ad_utr_scan = _scan_genomic_3utr(genome_fa, fai_idx, chrom, strand,
                                              ad_tts, utr_window)
            print(f"  [M10] Genomic 3'UTR scan: "
                  f"CT_PAS={ct_utr_scan.get('pas_count','?')}, "
                  f"AD_PAS={ad_utr_scan.get('pas_count','?')}")
        else:
            # Fallback: last 300 nt of protein sequence as rough proxy
            ct_seq = (case_result.get("ct_seq") or {}).get("sequence", "")
            ad_seq = (case_result.get("ad_seq") or {}).get("sequence", "")
            ct_utr_scan = _scan_3utr(ct_seq[-300:] if ct_seq else "")
            ad_utr_scan = _scan_3utr(ad_seq[-300:] if ad_seq else "")

    # ── Step 4: Differential analysis ─────────────────────────────────────
    utr_changes = _compare_utr_scans(ct_utr_scan, ad_utr_scan)

    # ── Step 5: Evidence strength ──────────────────────────────────────────
    evidence = _apa_evidence(apa_cls, ct_polyasite, ad_polyasite, utr_changes)
    print(f"  [M10] APA evidence: {evidence}")

    # ── Step 6: Summary ────────────────────────────────────────────────────
    summary = _build_summary(gene, tts_diff, apa_cls, ct_polyasite, ad_polyasite,
                              utr_changes, evidence)

    return {
        "tts_diff_bp":    tts_diff,
        "ct_tts":         ct_tts,
        "ad_tts":         ad_tts,
        "apa_class":      apa_cls,
        "ct_polyasite":   ct_polyasite[:5],
        "ad_polyasite":   ad_polyasite[:5],
        "ct_utr_scan":    ct_utr_scan,
        "ad_utr_scan":    ad_utr_scan,
        "utr_changes":    utr_changes,
        "apa_evidence":   evidence,
        "summary":        summary,
    }


def _build_summary(gene: str, tts_diff: int, apa_cls: str,
                   ct_polyasite: list, ad_polyasite: list,
                   utr_changes: dict, evidence: str) -> str:
    if apa_cls in ("same_apa", "minor_apa"):
        return (f"{gene}: CT and AD isoforms share essentially the same 3' end "
                f"(TTS_diff={tts_diff:,} bp). No significant APA detected.")

    polyasite_note = ""
    if ct_polyasite and ad_polyasite:
        polyasite_note = (f" Both 3' ends correspond to annotated PolyASite 2.0 "
                          f"poly-A clusters (CT: {len(ct_polyasite)} sites, "
                          f"AD: {len(ad_polyasite)} sites).")
    elif ct_polyasite:
        polyasite_note = (f" CT 3' end is annotated in PolyASite 2.0 "
                          f"({len(ct_polyasite)} poly-A cluster(s)); "
                          f"AD 3' end is unannotated.")
    elif ad_polyasite:
        polyasite_note = (f" AD 3' end is annotated in PolyASite 2.0 "
                          f"({len(ad_polyasite)} poly-A cluster(s)); "
                          f"CT 3' end is unannotated.")

    change_note = ""
    if utr_changes.get("comparable"):
        stab = utr_changes.get("predicted_stability", "unknown")
        gains  = list(utr_changes.get("gains", {}).keys())
        losses = list(utr_changes.get("losses", {}).keys())
        if gains:
            change_note += f" AD isoform gains: {', '.join(gains)}."
        if losses:
            change_note += f" AD isoform loses: {', '.join(losses)}."
        if stab != "unknown":
            change_note += f" Predicted mRNA stability effect: {stab}."

    return (
        f"{gene}: CT and AD isoforms differ in 3' end by {tts_diff:,} bp ({apa_cls}).{polyasite_note}"
        f"{change_note} APA evidence strength: {evidence}. "
        f"Caution: 3'UTR changes require RIP-seq or poly-A-seq validation for functional confirmation."
    )
