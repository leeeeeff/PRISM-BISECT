"""
Merge 17 new BISECT analysis.json results into bisect_cases.json.
Converts orchestrate.py output format → bisect_cases.json field schema.
"""
import json
import csv
from pathlib import Path

BASE       = Path(__file__).parent.parent
OUTPUTS    = BASE / "Final_analysis/pipeline_bioanalysis/outputs"
CASES_CSV  = BASE / "Final_analysis/pipeline_bioanalysis/cases_input_new.csv"
JSON_PATH  = BASE / "prism_app/data/demo/bisect_cases.json"


def _join_list(lst: list) -> str:
    if not lst:
        return ""
    return ";".join(str(x) for x in lst)


def _top_ppi(hyp_support: dict) -> tuple:
    """Return (top_partner, top_score) from hypothesis_support dict."""
    best_gene, best_score = "", 0
    for gene, info in hyp_support.items():
        sc = info.get("combined_score", 0) or 0
        if sc > best_score:
            best_score = sc
            best_gene = gene
    return best_gene, best_score


def _reg_str(regs: list) -> str:
    """Convert regulators list → semicolon-separated repr strings."""
    if not regs:
        return ""
    return ";".join(repr(r) for r in regs)


def analysis_to_case(d: dict, meta: dict) -> dict:
    """Convert analysis.json dict → bisect_cases.json row."""
    dc   = d.get("domain_change", {})
    m6   = d.get("m6_nmd_screen", {})
    m7   = d.get("m7_seq_validation", {})
    m8   = d.get("m8_regulatory_context", {})
    m9   = d.get("m9_promoter_usage", {})
    m10  = d.get("m10_apa", {})
    m11  = d.get("m11_alphafold", {})
    m12  = d.get("m12_ppi", {})
    m13  = d.get("m13_conservation", {})
    ct11 = m11.get("ct", {}) or {}
    ad11 = m11.get("ad", {}) or {}
    cmp11 = m11.get("comparison", {}) or {}
    ad_exons = m13.get("ad_specific_exons", [])

    top_partner, top_score = _top_ppi(m12.get("hypothesis_support", {}))

    # PPI verdict
    ppi_v = m12.get("summary_verdict", "")
    if not ppi_v and m12.get("string_hits"):
        ppi_v = "SUPPORTED"

    # Conservation
    cons_phylop = None
    cons_class  = ""
    if ad_exons:
        cons_phylop = ad_exons[0].get("phyloP_mean")
        cons_class  = ad_exons[0].get("conservation_class", "")

    # Seq validation
    seq_id  = None
    seq_con = ""
    if not m7.get("skipped"):
        seq_id  = m7.get("identity")
        seq_con = m7.get("conclusion", "")

    # DTU method
    bisect_tier = meta.get("bisect_tier", "C")
    if bisect_tier == "A-DR":
        dtu_method = "drimseq_stager"
    elif bisect_tier == "A-BP":
        dtu_method = "bisect_permutation"
    else:
        dtu_method = "pooled_chisq_exploratory"

    # stager_p / perm_p
    stager_p = None
    perm_p   = None
    sp = meta.get("stager_p", "")
    pp = meta.get("perm_p", "")
    try:
        stager_p = float(sp) if sp and sp.strip() else None
    except (ValueError, AttributeError):
        pass
    try:
        perm_p = float(pp) if pp and pp.strip() else None
    except (ValueError, AttributeError):
        pass

    return {
        "gene":              d.get("gene_name", ""),
        "cell_type":         d.get("cell_type", ""),
        "delta":             d.get("diffuse_delta"),
        "dtu_p":             d.get("dtu_pvalue"),
        "stage2_pass":       d.get("stage2_pass", False),
        "domains_lost":      _join_list(dc.get("domains_lost", [])),
        "domains_gained":    _join_list(dc.get("domains_gained", [])),
        "nat":               False,
        "young_l1_cds":      bool(d.get("ad_repeats", {}).get("has_l1_in_cds", False)),
        "seq_val_identity":  seq_id,
        "seq_val_conclusion": seq_con,
        "ad_nmd":            m6.get("ad", {}).get("nmd_susceptible"),
        "nmd_relevant":      bool(m6.get("nmd_relevant", False)),
        "ct_transcript_id":  d.get("ct_transcript_id", ""),
        "ad_transcript_id":  d.get("ad_transcript_id", ""),
        "tss_class":         m9.get("tss_class", ""),
        "tss_diff_bp":       m9.get("tss_diff_bp"),
        "apa_class":         m10.get("apa_class", ""),
        "tts_diff_bp":       m10.get("tts_diff_bp"),
        "af_ct_plddt_mean":  ct11.get("plddt_mean"),
        "af_ad_plddt_mean":  ad11.get("plddt_mean"),
        "af_ct_plddt_high_frac": ct11.get("plddt_high_fraction"),
        "af_ad_plddt_high_frac": ad11.get("plddt_high_fraction"),
        "af_gained_confident": _join_list(cmp11.get("gained_domain_confident", [])),
        "af_lost_confident":   _join_list(cmp11.get("lost_domain_confident", [])),
        "af_delta_plddt":      cmp11.get("delta_plddt"),
        "ppi_verdict":         ppi_v,
        "ppi_n_string_hits":   len(m12.get("string_hits", [])),
        "ppi_top_partner":     top_partner,
        "ppi_top_score":       top_score,
        "cons_ad_phylop":      cons_phylop,
        "cons_ad_class":       cons_class,
        "cons_background_phylop": m13.get("background", {}).get("intronic_phyloP_mean"),
        "mechanism_type":      m8.get("mechanism_type", ""),
        "top_regulators":      _reg_str(m8.get("top_regulators", [])),
        "ct_nmd":              m6.get("ct", {}).get("nmd_susceptible"),
        # New fields
        "bisect_tier":         bisect_tier,
        "dtu_method":          dtu_method,
        "stager_p":            stager_p,
        "perm_p":              perm_p,
    }


def main():
    # Load existing cases
    with open(JSON_PATH) as f:
        cases = json.load(f)
    existing_keys = {(c["gene"], c["cell_type"]) for c in cases}
    print(f"Existing cases: {len(cases)}")

    # Load meta from CSV
    meta_map = {}
    with open(CASES_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["gene_name"], row["cell_type"])
            meta_map[key] = row

    # Process each output directory
    added = 0
    errors = []
    for csv_row in meta_map.values():
        gene = csv_row["gene_name"]
        ct   = csv_row["cell_type"]
        key  = (gene, ct)

        if key in existing_keys:
            print(f"  SKIP {gene}/{ct}: already in bisect_cases.json")
            continue

        # Find output directory
        outdir = OUTPUTS / f"{gene}_{ct}"
        aj_path = outdir / "analysis.json"
        if not aj_path.exists():
            errors.append(f"{gene}/{ct}: analysis.json not found at {aj_path}")
            continue

        with open(aj_path) as f:
            d = json.load(f)

        case = analysis_to_case(d, meta_map[key])
        cases.append(case)
        existing_keys.add(key)
        added += 1
        print(f"  ADDED {gene}/{ct} — tier={case['bisect_tier']}, "
              f"stage2={case['stage2_pass']}, "
              f"domains_lost={case['domains_lost'][:40] or '(none)'}")

    with open(JSON_PATH, "w") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    print(f"\nTotal cases: {len(cases)} (added {added})")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  {e}")

    # Verify
    assert len(cases) == 84 + added, f"Expected {84+added}, got {len(cases)}"
    print(f"Validation passed: {len(cases)} total cases")


if __name__ == "__main__":
    main()
