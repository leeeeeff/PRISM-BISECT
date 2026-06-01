#!/usr/bin/env python3
"""
summarize_batch.py
==================
Reads all outputs/*/analysis.json files and produces a TSV summary
with M10 (AlphaFold pLDDT), M11 (PPI verdict), and M12 (phyloP conservation)
metrics for each case.

Output : outputs/batch_summary_all53.tsv
Flags  :
  - flag_accel_ad   : AD phyloP < 0  (accelerated evolution)
  - flag_unsupported: M11 verdict == UNSUPPORTED
  - flag_disordered : pLDDT < 50 (either CT or AD isoform)

Tier A candidates (printed at end):
  M12 AD phyloP > 3 AND M11 == SUPPORTED

Usage:
  conda run -n isoform_env python summarize_batch.py
  python summarize_batch.py [--output-dir path/to/outputs]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def parse_args():
    p = argparse.ArgumentParser(description="Summarize BISECT batch results")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Path to outputs/ directory (default: <script_dir>/outputs)",
    )
    p.add_argument(
        "--tsv-out",
        default=None,
        help="Output TSV path (default: <output-dir>/batch_summary_all53.tsv)",
    )
    return p.parse_args()


def _safe(value, fmt=None):
    """Return formatted value or 'NA' for None."""
    if value is None:
        return "NA"
    if fmt:
        try:
            return fmt.format(value)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _flag(condition):
    return "YES" if condition else ""


def load_case(json_path: Path) -> Optional[dict]:
    """Load and parse a single analysis.json into a summary row dict."""
    try:
        with open(json_path) as f:
            d = json.load(f)
    except Exception as e:
        print(f"  [WARN] Could not read {json_path}: {e}", file=sys.stderr)
        return None

    gene = d.get("gene_name", "?")
    cell_type = d.get("cell_type", "?")
    diffuse_delta = d.get("diffuse_delta")
    direction = d.get("direction", "")
    stage2_pass = d.get("stage2_pass", False)

    # ── M10: AlphaFold structural confidence ─────────────────────────────────
    m10 = d.get("m11_alphafold") or {}
    m10_ct = m10.get("ct") or {}
    m10_ad = m10.get("ad") or {}
    ct_plddt = m10_ct.get("plddt_mean")
    ad_plddt = m10_ad.get("plddt_mean")

    # ── M11: PPI network validation ───────────────────────────────────────────
    m11 = d.get("m12_ppi") or {}
    m11_verdict = m11.get("summary_verdict")
    string_hits = m11.get("string_hits") or []
    top_string_score = None
    top_string_partner = None
    if string_hits:
        top_hit = max(string_hits, key=lambda h: h.get("combined_score", 0), default=None)
        if top_hit:
            top_string_score = top_hit.get("combined_score")
            top_string_partner = top_hit.get("partner")

    # ── M12: Evolutionary conservation ───────────────────────────────────────
    m12 = d.get("m13_conservation") or {}
    m12_summary = m12.get("summary") or {}
    ad_phylop = m12_summary.get("ad_specific_mean_phyloP")
    ct_phylop = m12_summary.get("ct_specific_mean_phyloP")
    fold_diff = m12_summary.get("fold_difference")

    # ── Domain change ─────────────────────────────────────────────────────────
    dc = d.get("domain_change") or {}
    domains_lost = ";".join(dc.get("domains_lost") or []) or "—"
    domains_gained = ";".join(dc.get("domains_gained") or []) or "—"

    # ── Flags ─────────────────────────────────────────────────────────────────
    flag_accel_ad = (ad_phylop is not None) and (ad_phylop < 0)
    flag_unsupported = (m11_verdict is not None) and (m11_verdict.upper() == "UNSUPPORTED")
    min_plddt = min(
        [x for x in [ct_plddt, ad_plddt] if x is not None],
        default=None,
    )
    flag_disordered = (min_plddt is not None) and (min_plddt < 50)

    # ── M10/M11/M12 presence ──────────────────────────────────────────────────
    has_m10 = bool(m10_ct or m10_ad)
    has_m11 = bool(m11_verdict)
    has_m12 = bool(m12_summary)

    return {
        "gene": gene,
        "cell_type": cell_type,
        "diffuse_delta": diffuse_delta,
        "direction": direction,
        "stage2_pass": stage2_pass,
        "domains_lost": domains_lost,
        "domains_gained": domains_gained,
        # M10
        "m10_ct_plddt": ct_plddt,
        "m10_ad_plddt": ad_plddt,
        "has_m10": has_m10,
        # M11
        "m11_verdict": m11_verdict,
        "top_string_score": top_string_score,
        "top_string_partner": top_string_partner,
        "has_m11": has_m11,
        # M12
        "m12_ad_phylop": ad_phylop,
        "m12_ct_phylop": ct_phylop,
        "m12_fold_diff": fold_diff,
        "has_m12": has_m12,
        # Flags
        "flag_accel_ad": flag_accel_ad,
        "flag_unsupported": flag_unsupported,
        "flag_disordered": flag_disordered,
    }


def main():
    args = parse_args()

    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "outputs"

    if not output_dir.exists():
        print(f"ERROR: output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    tsv_out = Path(args.tsv_out) if args.tsv_out else output_dir / "batch_summary_all53.tsv"

    # ── Collect all analysis.json files ──────────────────────────────────────
    json_files = sorted(output_dir.glob("*/analysis.json"))
    print(f"Found {len(json_files)} analysis.json files in {output_dir}")

    rows = []
    for jf in json_files:
        row = load_case(jf)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No valid analysis.json files found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ── Sort: by gene name ────────────────────────────────────────────────────
    rows.sort(key=lambda r: r["gene"])

    # ── Write TSV ─────────────────────────────────────────────────────────────
    COLUMNS = [
        "gene",
        "cell_type",
        "diffuse_delta",
        "direction",
        "stage2_pass",
        "domains_lost",
        "domains_gained",
        "m10_ct_plddt",
        "m10_ad_plddt",
        "m11_verdict",
        "top_string_score",
        "top_string_partner",
        "m12_ad_phylop",
        "m12_ct_phylop",
        "m12_fold_diff",
        "flag_accel_ad",
        "flag_unsupported",
        "flag_disordered",
        "has_m10",
        "has_m11",
        "has_m12",
    ]

    def fmt_row(r: dict) -> list[str]:
        return [
            r["gene"],
            r["cell_type"],
            _safe(r["diffuse_delta"], "{:.4f}"),
            r["direction"],
            "TRUE" if r["stage2_pass"] else "FALSE",
            r["domains_lost"],
            r["domains_gained"],
            _safe(r["m10_ct_plddt"], "{:.2f}"),
            _safe(r["m10_ad_plddt"], "{:.2f}"),
            _safe(r["m11_verdict"]),
            _safe(r["top_string_score"]),
            _safe(r["top_string_partner"]),
            _safe(r["m12_ad_phylop"], "{:.3f}"),
            _safe(r["m12_ct_phylop"], "{:.3f}"),
            _safe(r["m12_fold_diff"], "{:.3f}"),
            _flag(r["flag_accel_ad"]),
            _flag(r["flag_unsupported"]),
            _flag(r["flag_disordered"]),
            "TRUE" if r["has_m10"] else "FALSE",
            "TRUE" if r["has_m11"] else "FALSE",
            "TRUE" if r["has_m12"] else "FALSE",
        ]

    with open(tsv_out, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(fmt_row(r)) + "\n")

    print(f"\nTSV written to: {tsv_out}")
    print(f"Total rows     : {len(rows)}")

    # ── Statistics ────────────────────────────────────────────────────────────
    n_with_m10 = sum(1 for r in rows if r["has_m10"])
    n_with_m11 = sum(1 for r in rows if r["has_m11"])
    n_with_m12 = sum(1 for r in rows if r["has_m12"])

    verdicts = [r["m11_verdict"] for r in rows if r["m11_verdict"] is not None]
    n_supported = sum(1 for v in verdicts if v.upper() == "SUPPORTED")
    n_unsupported = sum(1 for v in verdicts if v.upper() == "UNSUPPORTED")
    n_verdict_other = len(verdicts) - n_supported - n_unsupported

    flagged_accel = [r for r in rows if r["flag_accel_ad"]]
    flagged_unsupported = [r for r in rows if r["flag_unsupported"]]
    flagged_disordered = [r for r in rows if r["flag_disordered"]]

    # Tier A: AD phyloP > 3 AND M11 SUPPORTED
    tier_a = [
        r for r in rows
        if (r["m12_ad_phylop"] is not None and r["m12_ad_phylop"] > 3.0)
        and (r["m11_verdict"] is not None and r["m11_verdict"].upper() == "SUPPORTED")
    ]

    # Stage 2 pass cases
    stage2_pass_cases = [r for r in rows if r["stage2_pass"]]

    print("\n" + "=" * 60)
    print("  BISECT Batch Summary Statistics")
    print("=" * 60)
    print(f"  Total cases parsed        : {len(rows)}")
    print(f"  Stage 2 PASS              : {len(stage2_pass_cases)}")
    print(f"  Cases with M10 data       : {n_with_m10}")
    print(f"  Cases with M11 data       : {n_with_m11}")
    print(f"  Cases with M12 data       : {n_with_m12}")
    print()
    print(f"  M11 SUPPORTED             : {n_supported}")
    print(f"  M11 UNSUPPORTED           : {n_unsupported}")
    if n_verdict_other:
        print(f"  M11 other verdict         : {n_verdict_other}")
    print()
    print(f"  [FLAG] AD phyloP < 0      : {len(flagged_accel)}")
    if flagged_accel:
        for r in flagged_accel:
            print(f"    {r['gene']} [{r['cell_type']}] AD_phyloP={_safe(r['m12_ad_phylop'], '{:.3f}')}")
    print()
    print(f"  [FLAG] M11 UNSUPPORTED    : {len(flagged_unsupported)}")
    if flagged_unsupported:
        for r in flagged_unsupported:
            print(f"    {r['gene']} [{r['cell_type']}]")
    print()
    print(f"  [FLAG] pLDDT < 50         : {len(flagged_disordered)}")
    if flagged_disordered:
        for r in flagged_disordered:
            ct = _safe(r["m10_ct_plddt"], "{:.1f}")
            ad = _safe(r["m10_ad_plddt"], "{:.1f}")
            print(f"    {r['gene']} [{r['cell_type']}] CT_pLDDT={ct} AD_pLDDT={ad}")
    print()
    print(f"  Tier A candidates         : {len(tier_a)}")
    print(f"  (AD phyloP > 3.0 AND M11 SUPPORTED)")
    if tier_a:
        for r in tier_a:
            print(
                f"    {r['gene']} [{r['cell_type']}] "
                f"AD_phyloP={_safe(r['m12_ad_phylop'], '{:.3f}')} "
                f"STRING={_safe(r['top_string_score'])} "
                f"CT_pLDDT={_safe(r['m10_ct_plddt'], '{:.1f}')} "
                f"AD_pLDDT={_safe(r['m10_ad_plddt'], '{:.1f}')}"
            )
    print("=" * 60)


if __name__ == "__main__":
    main()
