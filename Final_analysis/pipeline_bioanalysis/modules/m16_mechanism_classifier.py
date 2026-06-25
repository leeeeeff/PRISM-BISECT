"""
M16: Mechanism Classifier
Integrates M2 (domains), M6/M15 (NMD), M9 (TSS), M10 (APA), M14 (exon structure)
to automatically classify the primary biological mechanism of each isoform switch.
"""
from __future__ import annotations


_MECHANISM_PRIORITY = [
    "nmd_switch",
    "major_truncation",
    "domain_loss",
    "domain_gain",
    "ale",
    "retained_intron",
    "cassette_exon_cluster",
    "exon_exchange",
    "alt_tss",
    "alt_apa",
    "alt_splice_site",
    "minor_exon",
    "complex",
]


def run_m16(case: dict, config: dict) -> dict:
    ev = _collect_evidence(case)
    primary, secondary = _classify(ev)
    narrative = _make_narrative(primary, secondary, ev, case)

    return {
        "m16_mechanism_classifier": {
            "primary_mechanism": primary,
            "secondary_mechanism": secondary,
            "mechanism_evidence": ev,
            "narrative": narrative,
        }
    }


def _collect_evidence(case: dict) -> dict:
    # --- NMD (prefer M15, fallback M6) ---
    nmd_switch = False
    nmd_source = "none"
    m15 = case.get("m15_nmd_v2") or {}
    if m15:
        nmd_switch = bool(m15.get("nmd_switch", False))
        nmd_source = "m15"
    else:
        m6 = case.get("m6_nmd_screen") or {}
        ct_nmd = bool(m6.get("ct_nmd_predicted", False))
        ad_nmd = bool(m6.get("ad_nmd_predicted", False))
        nmd_switch = (not ct_nmd) and ad_nmd
        nmd_source = "m6"

    # --- Domain change (M2) ---
    dc = case.get("domain_change") or {}
    domains_lost   = dc.get("domains_lost")   or []
    domains_gained = dc.get("domains_gained") or []
    domain_change  = bool(domains_lost or domains_gained)

    # --- Protein lengths ---
    def _aa(key):
        val = case.get(key)
        if isinstance(val, dict):
            return len(val.get("seq") or "")
        if isinstance(val, str):
            return len(val)
        return 0

    ct_aa = _aa("ct_seq")
    ad_aa = _aa("ad_seq")
    aa_diff = ad_aa - ct_aa
    major_truncation = (aa_diff < -200) and bool(domains_lost)

    # --- Exon structure (M14) ---
    m14 = case.get("m14_exon_comparator") or {}
    event_type    = m14.get("event_type", "unknown")
    ale           = bool(m14.get("ale_detected", False))
    super_exon    = bool(m14.get("super_exon_detected", False))
    n_ct_only     = m14.get("n_ct_only", 0) or 0
    n_ad_only     = m14.get("n_ad_only", 0) or 0
    exon_count_change = n_ad_only - n_ct_only

    # --- TSS / APA (M9, M10) ---
    m9 = case.get("m9_promoter_usage") or {}
    tss_shift = abs(m9.get("tss_diff_bp") or 0)
    m10 = case.get("m10_apa") or {}
    tts_shift = abs(m10.get("tts_diff_bp") or 0)

    return {
        "nmd_switch":             nmd_switch,
        "nmd_source":             nmd_source,
        "domain_change":          domain_change,
        "domains_lost":           domains_lost,
        "domains_gained":         domains_gained,
        "major_truncation":       major_truncation,
        "protein_length_change_aa": aa_diff,
        "ct_aa":                  ct_aa,
        "ad_aa":                  ad_aa,
        "ale":                    ale,
        "super_exon":             super_exon,
        "event_type_m14":         event_type,
        "exon_count_change":      exon_count_change,
        "tss_shift_bp":           tss_shift,
        "tts_shift_bp":           tts_shift,
    }


def _classify(ev: dict):
    primary = "complex"
    secondary = []

    # Priority order: structural mechanism first, then functional consequence
    if ev["nmd_switch"]:
        primary = "nmd_switch"
        if ev["domain_change"]:
            secondary.append("domain_loss" if ev["domains_lost"] else "domain_gain")
    elif ev["ale"]:
        # ALE is the splicing mechanism; consequences (truncation, domain loss) are secondary
        primary = "ale"
        if ev["protein_length_change_aa"] < -200:
            secondary.append("major_truncation")
        if ev["domains_lost"]:
            secondary.append("domain_loss")
    elif ev["super_exon"]:
        primary = "retained_intron"
        if ev["nmd_switch"] is False and ev["protein_length_change_aa"] < -100:
            secondary.append("major_truncation")
    elif ev["major_truncation"]:
        primary = "major_truncation"
        if ev["domains_lost"]:
            secondary.append("domain_loss")
    elif ev["domains_lost"]:
        primary = "domain_loss"
    elif ev["domains_gained"]:
        primary = "domain_gain"
    elif ev["event_type_m14"] == "cassette_exon_cluster":
        primary = "cassette_exon_cluster"
    elif ev["event_type_m14"] == "exon_exchange":
        primary = "exon_exchange"
    elif ev["tss_shift_bp"] > 500 and ev["protein_length_change_aa"] == 0:
        primary = "alt_tss"
    elif ev["tts_shift_bp"] > 500 and ev["protein_length_change_aa"] == 0:
        primary = "alt_apa"
    elif ev["event_type_m14"] in ("alt_splice_site",):
        primary = "alt_splice_site"
    elif abs(ev["exon_count_change"]) <= 1 and not ev["domain_change"]:
        primary = "minor_exon"

    # Always add tss if large and not already primary
    if ev["tss_shift_bp"] > 500 and primary not in ("alt_tss",):
        if "alt_tss" not in secondary:
            secondary.append("alt_tss")

    return primary, secondary


def _make_narrative(primary: str, secondary: list, ev: dict, case: dict) -> str:
    gene = case.get("gene_name") or case.get("gene") or "This isoform"
    ct_aa = ev["ct_aa"]
    ad_aa = ev["ad_aa"]
    aa_diff = ev["protein_length_change_aa"]

    texts = {
        "nmd_switch": (
            f"{gene} undergoes an NMD-linked isoform switch: "
            f"the AD-associated isoform ({ad_aa} aa) contains a premature termination codon "
            f"upstream of the last exon junction complex, predicting nonsense-mediated mRNA decay. "
            + (f"This co-occurs with loss of {', '.join(ev['domains_lost'][:2])}." if ev["domains_lost"] else "")
        ),
        "major_truncation": (
            f"{gene} shifts from a {ct_aa} aa to a {ad_aa} aa isoform (Δ{aa_diff} aa), "
            f"with major domain loss ({', '.join(ev['domains_lost'][:3])})."
        ),
        "domain_loss": (
            f"The AD-associated {gene} isoform ({ad_aa} aa) loses "
            f"{', '.join(ev['domains_lost'][:3])} domain(s) present in the CT isoform ({ct_aa} aa), "
            f"likely impairing canonical function."
        ),
        "domain_gain": (
            f"The AD-associated {gene} isoform ({ad_aa} aa) gains "
            f"{', '.join(ev['domains_gained'][:3])} domain(s) absent from the CT isoform ({ct_aa} aa), "
            f"potentially conferring neomorphic function."
        ),
        "ale": (
            f"{gene} switches to an Alternative Last Exon isoform in AD: "
            f"CT and AD isoforms share no exons, with the AD isoform using a completely distinct "
            f"exon set (CT: {ct_aa} aa → AD: {ad_aa} aa)."
        ),
        "retained_intron": (
            f"The AD {gene} isoform incorporates a retained intron forming a 'super-exon', "
            f"replacing multiple CT-specific exons with a single large exon "
            f"and truncating the protein from {ct_aa} to {ad_aa} aa."
        ),
        "cassette_exon_cluster": (
            f"In AD, {gene} includes a cassette exon cluster absent from the CT isoform, "
            f"extending the protein from {ct_aa} to {ad_aa} aa (+{aa_diff} aa C-terminal addition)."
        ),
        "exon_exchange": (
            f"{gene} undergoes exon exchange between CT and AD isoforms "
            f"(CT: {ct_aa} aa, AD: {ad_aa} aa), altering the internal domain architecture."
        ),
        "alt_tss": (
            f"{gene} uses an alternative transcription start site in AD "
            f"(TSS shift: {ev['tss_shift_bp']:,} bp), producing an isoform with altered N-terminus."
        ),
        "alt_splice_site": (
            f"{gene} exhibits alternative splice site usage between CT and AD isoforms "
            f"(CT: {ct_aa} aa, AD: {ad_aa} aa), with minor structural changes."
        ),
        "minor_exon": (
            f"{gene} shows a minor exon variant between CT ({ct_aa} aa) and AD ({ad_aa} aa) isoforms "
            f"without domain change — potentially a UTR or regulatory element difference."
        ),
        "complex": (
            f"{gene} shows a complex isoform switch (CT: {ct_aa} aa → AD: {ad_aa} aa) "
            f"that does not fit a single canonical mechanism."
        ),
    }

    base = texts.get(primary, texts["complex"])
    if secondary:
        base += f" Secondary: {', '.join(secondary)}."
    return base
