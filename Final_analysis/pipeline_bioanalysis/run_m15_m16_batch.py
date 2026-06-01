"""
LEGACY batch runner for M9 (Alternative Promoter) + M10 (APA).
[Formerly M15 + M16 — renumbered in BISECT v2.0]

These modules are now integrated into orchestrate.py (Stage 4).
This script remains for re-running on existing analysis.json files only.

Usage:
    conda activate isoform_env
    python run_m15_m16_batch.py [--regen-report] [--case GENE_CELLTYPE]

Output:
    outputs/<GENE_CELLTYPE>/analysis.json  (updated with m9/m10 keys)
    outputs/<GENE_CELLTYPE>/report.md      (if --regen-report)
    outputs/m9_m10_batch_summary.tsv
    outputs/m9_m10_batch_summary.json
"""
import json, os, sys, argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_DIR))

from modules import m9_promoter_usage, m10_apa, m14_report

OUTPUTS_DIR = PIPELINE_DIR / "outputs"
CONFIG_PATH = PIPELINE_DIR / "config.yaml"


def load_config():
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    except ImportError:
        import re
        config = {}
        with open(CONFIG_PATH) as f:
            text = f.read()
        # minimal parse — just return empty sub-configs if yaml unavailable
        config.setdefault("m15", {"use_screen_api": False, "tss_alt_threshold": 500})
        config.setdefault("m16", {"use_polyasite_api": False})
        return config


def get_case_dirs():
    return sorted([d for d in OUTPUTS_DIR.iterdir()
                   if d.is_dir() and (d / "analysis.json").exists()])


def run_case(case_dir: Path, config: dict, regen_report: bool) -> dict:
    with open(case_dir / "analysis.json") as f:
        case_result = json.load(f)

    m9_result  = m9_promoter_usage.run(case_result, config)
    m10_result = m10_apa.run(case_result, config)

    case_result["m9_promoter_usage"] = m9_result
    case_result["m10_apa"] = m10_result

    # M9 reclassification: update M8 mechanism if alt_promoter confirmed
    if m9_result.get("mechanism_reclassify"):
        reg = case_result.get("m8_regulatory_context") or {}
        if reg.get("mechanism_type") == "transcriptional":
            reg["mechanism_type"] = "alternative_promoter"
            reg["mechanism_reclassified_by"] = "M9"
            reg["tss_diff_bp"] = m9_result.get("tss_diff_bp")
            case_result["m8_regulatory_context"] = reg

    with open(case_dir / "analysis.json", "w") as f:
        json.dump(case_result, f, indent=2, default=str)

    if regen_report:
        m14_report.generate_markdown(case_result, str(case_dir))

    return {
        "case":           case_dir.name,
        "tss_class":      m15_result.get("tss_class", "?"),
        "tss_diff_bp":    m15_result.get("tss_diff_bp", 0),
        "prom_evidence":  m15_result.get("promoter_evidence", "?"),
        "reclassify":     m15_result.get("mechanism_reclassify", False),
        "apa_class":      m16_result.get("apa_class", "?"),
        "tts_diff_bp":    m16_result.get("tts_diff_bp", 0),
        "apa_evidence":   m16_result.get("apa_evidence", "?"),
        "stability":      (m16_result.get("utr_changes") or {}).get("predicted_stability", "unknown"),
        "m9_error":       m9_result.get("skipped", ""),
        "m10_error":      m10_result.get("skipped", ""),
    }


def print_stats(rows):
    ok = [r for r in rows if not r["m9_error"] and not r["m10_error"]]
    print(f"\n{'='*65}")
    print("M9 + M10 BATCH STATISTICS")
    print(f"{'='*65}")
    print(f"Total: {len(rows)} | OK: {len(ok)}")

    # M15
    print("\n── M9 TSS Class distribution ──────────────────────────────")
    for cls, cnt in sorted(Counter(r["tss_class"] for r in ok).items(),
                           key=lambda x: -x[1]):
        pct = cnt / len(ok) * 100
        reclass = sum(1 for r in ok if r["tss_class"] == cls and r["reclassify"])
        print(f"  {cls:<25} {cnt:3d} ({pct:.0f}%)  reclassified={reclass}")

    print("\n  Cases with alt_promoter_candidate (TSS_diff > 500 bp):")
    alt_cases = sorted([r for r in ok if r["reclassify"]], key=lambda x: -x["tss_diff_bp"])
    for r in alt_cases:
        print(f"  {r['case']:<35} TSS={r['tss_diff_bp']:>8,} bp  "
              f"evidence={r['prom_evidence']}")

    # M16
    print("\n── M10 APA Class distribution ──────────────────────────────")
    for cls, cnt in sorted(Counter(r["apa_class"] for r in ok).items(),
                           key=lambda x: -x[1]):
        pct = cnt / len(ok) * 100
        print(f"  {cls:<20} {cnt:3d} ({pct:.0f}%)")

    print("\n  Cases with major_apa (TTS_diff > 5 kb):")
    major_apa = sorted([r for r in ok if r["apa_class"] == "major_apa"],
                       key=lambda x: -x["tts_diff_bp"])
    for r in major_apa:
        stab = r["stability"]
        print(f"  {r['case']:<35} TTS={r['tts_diff_bp']:>8,} bp  "
              f"stability={stab}")

    # Combined (both M15 + M16)
    both = [r for r in ok if r["reclassify"] and r["apa_class"] == "major_apa"]
    if both:
        print(f"\n  Cases with BOTH alt_promoter + major_apa ({len(both)}):")
        for r in both:
            print(f"  {r['case']}")

    # Stability predictions
    stab_counts = Counter(r["stability"] for r in ok if r["stability"] != "unknown")
    if stab_counts:
        print("\n── M16 Predicted mRNA stability changes ─────────────────")
        for stab, cnt in sorted(stab_counts.items(), key=lambda x: -x[1]):
            print(f"  {stab:<40} {cnt:3d}")

    print(f"{'='*65}\n")


def save_summary(rows):
    fields = ["case", "tss_class", "tss_diff_bp", "prom_evidence", "reclassify",
              "apa_class", "tts_diff_bp", "apa_evidence", "stability",
              "m9_error", "m10_error"]
    tsv_path = OUTPUTS_DIR / "m9_m10_batch_summary.tsv"
    with open(tsv_path, "w") as f:
        f.write("\t".join(fields) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in fields) + "\n")

    json_path = OUTPUTS_DIR / "m9_m10_batch_summary.json"
    with open(json_path, "w") as f:
        json.dump({"generated": datetime.now().isoformat(),
                   "n_total": len(rows), "rows": rows}, f, indent=2)
    print(f"Summary: {tsv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen-report", action="store_true")
    parser.add_argument("--case", default=None)
    args = parser.parse_args()

    config = load_config()
    case_dirs = get_case_dirs()
    if args.case:
        case_dirs = [d for d in case_dirs if d.name == args.case]
        if not case_dirs:
            print(f"ERROR: case '{args.case}' not found"); sys.exit(1)

    print(f"Running M15+M16 on {len(case_dirs)} cases...")
    rows = []
    for i, case_dir in enumerate(case_dirs, 1):
        print(f"\n[{i:2d}/{len(case_dirs)}] {case_dir.name}")
        try:
            row = run_case(case_dir, config, args.regen_report)
            print(f"  M15={row['tss_class']}({row['tss_diff_bp']:,}bp) "
                  f"reclassify={row['reclassify']} | "
                  f"M16={row['apa_class']}({row['tts_diff_bp']:,}bp) "
                  f"stability={row['stability']}")
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            row = {"case": case_dir.name, "tss_class": "exception",
                   "tss_diff_bp": 0, "prom_evidence": "error", "reclassify": False,
                   "apa_class": "exception", "tts_diff_bp": 0,
                   "apa_evidence": "error", "stability": "unknown",
                   "m15_error": str(e), "m16_error": str(e)}
        rows.append(row)

    print_stats(rows)
    save_summary(rows)


if __name__ == "__main__":
    main()
