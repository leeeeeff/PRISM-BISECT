"""
M15: Cross-Case Comparison Table
Reads all analysis.json files from output directories and produces
cases_summary.tsv suitable for paper Table 1.
"""
import csv
import json
import os
from pathlib import Path
from datetime import datetime


SUMMARY_COLS = [
    "gene", "cell_type", "diffuse_delta", "dtu_pvalue", "direction", "priority",
    "ct_transcript_id", "ad_transcript_id",
    "ct_length_aa", "ad_length_aa", "length_diff_aa",
    "ct_domains", "ad_domains",
    "domains_lost", "domains_gained", "n_lost", "n_gained",
    "domain_change",
    "mts_ct", "mts_ad", "lyr_ct", "lyr_ad", "hhh_ct", "hhh_ad",
    "is_nat", "nat_description",
    "has_young_l1_cds", "l1_elements",
    "structural_category_ct", "structural_category_ad",
    "exon_count_ct", "exon_count_ad", "span_kb_ct", "span_kb_ad",
    "stage2_pass", "pipeline_ts",
]


def _fmt_list(lst: list) -> str:
    return ";".join(str(x) for x in lst) if lst else ""


def _extract(j: dict) -> dict:
    """Flatten analysis.json into one summary row."""
    dc = j.get("domain_change", {})
    ct_m = j.get("ct_motifs", {}).get("mts", {})
    ad_m = j.get("ad_motifs", {}).get("mts", {})
    ct_info = j.get("ct_info", {})
    ad_info = j.get("ad_info", {})
    ad_rep = j.get("ad_repeats", {})

    ct_domains_list = [d.get("pfam_family", d.get("domain", "?")) for d in j.get("ct_domains", [])]
    ad_domains_list = [d.get("pfam_family", d.get("domain", "?")) for d in j.get("ad_domains", [])]
    lost = dc.get("domains_lost", [])
    gained = dc.get("domains_gained", [])

    ct_len = j.get("ct_seq", {}).get("length") or ct_info.get("protein_length") or ""
    ad_len = j.get("ad_seq", {}).get("length") or ad_info.get("protein_length") or ""
    len_diff = (ad_len - ct_len) if (isinstance(ct_len, int) and isinstance(ad_len, int)) else ""

    # Repeat element names
    l1_hits = ad_rep.get("cds_young_l1_hits", [])
    l1_names = ";".join(h.get("name", "?") for h in l1_hits) if l1_hits else ""

    return {
        "gene": j.get("gene_name", ""),
        "cell_type": j.get("cell_type", ""),
        "diffuse_delta": j.get("diffuse_delta", ""),
        "dtu_pvalue": j.get("dtu_pvalue", ""),
        "direction": j.get("direction", ""),
        "priority": j.get("priority", ""),
        "ct_transcript_id": j.get("ct_transcript_id", ""),
        "ad_transcript_id": j.get("ad_transcript_id", ""),
        "ct_length_aa": ct_len,
        "ad_length_aa": ad_len,
        "length_diff_aa": len_diff,
        "ct_domains": _fmt_list(ct_domains_list),
        "ad_domains": _fmt_list(ad_domains_list),
        "domains_lost": _fmt_list(lost),
        "domains_gained": _fmt_list(gained),
        "n_lost": len(lost),
        "n_gained": len(gained),
        "domain_change": "YES" if dc.get("has_domain_change") else "NO",
        "mts_ct": ct_m.get("composite_score", ""),
        "mts_ad": ad_m.get("composite_score", ""),
        "lyr_ct": "YES" if ct_m.get("lyr_motif") else "no",
        "lyr_ad": "YES" if ad_m.get("lyr_motif") else "no",
        "hhh_ct": "YES" if ct_m.get("hhh_motif") else "no",
        "hhh_ad": "YES" if ad_m.get("hhh_motif") else "no",
        "is_nat": "YES" if ad_info.get("is_nat") else "no",
        "nat_description": (ad_info.get("nat_relationship") or {}).get("description", ""),
        "has_young_l1_cds": "YES" if ad_rep.get("has_l1_in_cds") else "no",
        "l1_elements": l1_names,
        "structural_category_ct": ct_info.get("structural_category", ""),
        "structural_category_ad": ad_info.get("structural_category", ""),
        "exon_count_ct": ct_info.get("exon_count", ""),
        "exon_count_ad": ad_info.get("exon_count", ""),
        "span_kb_ct": round(ct_info.get("genomic_span_kb", 0), 2) if ct_info.get("genomic_span_kb") else "",
        "span_kb_ad": round(ad_info.get("genomic_span_kb", 0), 2) if ad_info.get("genomic_span_kb") else "",
        "stage2_pass": "YES" if j.get("stage2_pass") else "NO",
        "pipeline_ts": j.get("pipeline_ts", ""),
    }


def _priority_sort_key(row: dict) -> tuple:
    """Sort: HIGH→LOW priority, then |delta| descending."""
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    prio = priority_order.get(str(row.get("priority", "")).upper(), 9)
    delta = abs(float(row["diffuse_delta"])) if row.get("diffuse_delta") != "" else 0
    return (prio, -delta)


def build_summary(output_root: str) -> str:
    """
    Scan output_root for analysis.json files, build cases_summary.tsv.
    Returns path to written TSV.
    """
    rows = []
    for case_dir in sorted(Path(output_root).iterdir()):
        json_path = case_dir / "analysis.json"
        if not json_path.exists():
            continue
        try:
            with open(json_path) as f:
                j = json.load(f)
            rows.append(_extract(j))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [M7] Skipping {json_path}: {e}")

    if not rows:
        print("  [M7] No analysis.json files found.")
        return ""

    rows.sort(key=_priority_sort_key)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    tsv_path = os.path.join(output_root, f"cases_summary_{ts}.tsv")
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [M7] cases_summary → {tsv_path}  ({len(rows)} cases)")
    return tsv_path
