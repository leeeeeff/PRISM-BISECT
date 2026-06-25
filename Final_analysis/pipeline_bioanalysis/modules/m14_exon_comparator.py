"""
M14: Exon Structure Comparator
Quantitatively compares CT vs AD isoform exon structures to classify splicing event types.
"""
from __future__ import annotations


def run_m14(case: dict, config: dict) -> dict:
    ct_info = case.get("ct_info") or {}
    ad_info = case.get("ad_info") or {}

    ct_exons_raw = ct_info.get("exons") or []
    ad_exons_raw = ad_info.get("exons") or []

    ct_exons = [tuple(e) for e in ct_exons_raw if len(e) >= 2]
    ad_exons = [tuple(e) for e in ad_exons_raw if len(e) >= 2]

    if not ct_exons and not ad_exons:
        return {
            "m14_exon_comparator": {
                "status": "no_exon_data",
                "event_type": "unknown",
            }
        }

    ct_set = set(ct_exons)
    ad_set = set(ad_exons)

    shared  = sorted(ct_set & ad_set)
    ct_only = sorted(ct_set - ad_set)
    ad_only = sorted(ad_set - ct_set)

    ct_only_bp = sum(e[1] - e[0] for e in ct_only)
    ad_only_bp = sum(e[1] - e[0] for e in ad_only)
    net_bp = ad_only_bp - ct_only_bp

    # Super-exon: a single AD exon spans 3+ CT-only exons
    super_exon = False
    super_exon_details = []
    for ae in ad_only:
        contained = [ce for ce in ct_only if ce[0] >= ae[0] and ce[1] <= ae[1]]
        if len(contained) >= 3:
            super_exon = True
            super_exon_details.append({
                "ad_exon": list(ae),
                "contained_ct_exons": [list(ce) for ce in contained],
            })

    # ALE: no shared exons
    ale = len(shared) == 0

    # Protein lengths
    def _get_aa(key: str) -> int:
        val = case.get(key)
        if isinstance(val, dict):
            return len(val.get("seq") or "")
        if isinstance(val, str):
            return len(val)
        return 0

    ct_aa = _get_aa("ct_seq")
    ad_aa = _get_aa("ad_seq")
    aa_diff = ad_aa - ct_aa

    # Event classification
    event_type = _classify_event(
        ale=ale,
        super_exon=super_exon,
        shared=shared,
        ct_only=ct_only,
        ad_only=ad_only,
        ct_aa=ct_aa,
        ad_aa=ad_aa,
    )

    # TSS/TTS shift context (from m9/m10 if available)
    tss_shift = None
    tts_shift = None
    if "m9_promoter_usage" in case:
        tss_shift = case["m9_promoter_usage"].get("tss_diff_bp")
    if "m10_apa" in case:
        tts_shift = case["m10_apa"].get("tts_diff_bp")

    return {
        "m14_exon_comparator": {
            "n_ct_exons": len(ct_exons),
            "n_ad_exons": len(ad_exons),
            "n_shared": len(shared),
            "n_ct_only": len(ct_only),
            "n_ad_only": len(ad_only),
            "ct_only_bp": ct_only_bp,
            "ad_only_bp": ad_only_bp,
            "net_bp_change": net_bp,
            "aa_diff": aa_diff,
            "super_exon_detected": super_exon,
            "super_exon_details": super_exon_details,
            "ale_detected": ale,
            "event_type": event_type,
            "ct_only_exons": [list(e) for e in ct_only],
            "ad_only_exons": [list(e) for e in ad_only],
            "shared_exons": [list(e) for e in shared[:20]],  # cap for JSON size
            "tss_shift_bp": tss_shift,
            "tts_shift_bp": tts_shift,
        }
    }


def _classify_event(
    ale: bool,
    super_exon: bool,
    shared: list,
    ct_only: list,
    ad_only: list,
    ct_aa: int,
    ad_aa: int,
) -> str:
    if ale:
        return "ale"
    if super_exon:
        return "retained_intron"
    if ad_aa > ct_aa and len(ad_only) >= 3:
        return "cassette_exon_cluster"
    if ct_aa == ad_aa and len(ct_only) + len(ad_only) <= 4:
        return "alt_splice_site"
    if len(ct_only) > 5:
        return "major_exon_loss"
    if len(ct_only) > 0 and len(ad_only) > 0:
        return "exon_exchange"
    if len(ct_only) > 0 and len(ad_only) == 0:
        return "exon_skipping"
    if len(ct_only) == 0 and len(ad_only) > 0:
        return "exon_inclusion"
    return "complex"
