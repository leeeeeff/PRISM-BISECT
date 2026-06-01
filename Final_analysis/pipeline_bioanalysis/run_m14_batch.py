"""
Batch M8 runner (formerly M14): Regulatory Context Evidence analysis on all cases.

Usage:
    conda activate isoform_env
    python run_m14_batch.py [--regen-report]

Reads existing analysis.json for each case, runs M8, saves results back,
and optionally regenerates report.md.

Output:
  - outputs/<GENE_CELLTYPE>/analysis.json  (updated with m8_regulatory_context)
  - outputs/<GENE_CELLTYPE>/report.md      (regenerated if --regen-report)
  - outputs/m8_batch_summary.tsv           (statistics across all cases)
  - outputs/m8_batch_summary.json          (machine-readable)
"""
import json
import os
import sys
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_DIR))

from modules import m8_regulatory_context
from modules import m14_report

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
        # minimal parse for m14 section
        m14_section = {}
        in_m14 = False
        for line in text.splitlines():
            if line.strip() == "m14:":
                in_m14 = True
                continue
            if in_m14:
                if line and not line.startswith(" "):
                    in_m14 = False
                    continue
                m = re.match(r'\s+(\w+):\s*(.+)', line)
                if m:
                    key, val = m.group(1), m.group(2).strip()
                    try:
                        val = float(val) if "." in val else int(val)
                    except ValueError:
                        pass
                    m14_section[key] = val
        config["m14"] = m14_section
        return config


def get_case_dirs():
    dirs = []
    for d in sorted(OUTPUTS_DIR.iterdir()):
        if d.is_dir() and (d / "analysis.json").exists():
            dirs.append(d)
    return dirs


def run_case(case_dir: Path, config: dict, regen_report: bool) -> dict:
    gene_cell = case_dir.name
    json_path = case_dir / "analysis.json"

    with open(json_path) as f:
        case_result = json.load(f)

    m8_result = m8_regulatory_context.run(case_result, config)
    case_result["m8_regulatory_context"] = m8_result

    with open(json_path, "w") as f:
        json.dump(case_result, f, indent=2, default=str)

    if regen_report:
        m14_report.generate_markdown(case_result, str(case_dir))

    mechanism = m8_result.get("mechanism_type", "error")
    evidence = m8_result.get("evidence_strength", "error")
    n_regs = len(m8_result.get("significant_regulators", []))
    error = m8_result.get("error", "")

    return {
        "case": gene_cell,
        "mechanism": mechanism,
        "evidence": evidence,
        "n_regulators": n_regs,
        "error": error,
    }


def print_stats(rows: list):
    print("\n" + "="*60)
    print("M14 BATCH STATISTICS")
    print("="*60)

    total = len(rows)
    errors = [r for r in rows if r["error"]]
    ok = [r for r in rows if not r["error"]]

    print(f"\nTotal cases: {total}")
    print(f"  Succeeded: {len(ok)}")
    print(f"  Errors:    {len(errors)}")

    if errors:
        print("\nFailed cases:")
        for r in errors:
            print(f"  {r['case']}: {r['error'][:80]}")

    print("\nMechanism type distribution:")
    mech_counts = Counter(r["mechanism"] for r in ok)
    for mech, cnt in sorted(mech_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(ok) * 100
        print(f"  {mech:<30} {cnt:3d} ({pct:.0f}%)")

    print("\nEvidence strength distribution:")
    ev_counts = Counter(r["evidence"] for r in ok)
    for ev, cnt in sorted(ev_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(ok) * 100
        print(f"  {ev:<15} {cnt:3d} ({pct:.0f}%)")

    # Cross-tabulate mechanism × evidence
    print("\nMechanism × Evidence cross-table:")
    mechs = sorted(mech_counts)
    evs = sorted(ev_counts)
    header = f"{'Mechanism':<30}" + "".join(f"{e:<12}" for e in evs)
    print(f"  {header}")
    for mech in mechs:
        row_counts = defaultdict(int)
        for r in ok:
            if r["mechanism"] == mech:
                row_counts[r["evidence"]] += 1
        vals = "".join(f"{row_counts[e]:<12}" for e in evs)
        print(f"  {mech:<30}{vals}")

    print("\nMean regulators per case (by mechanism):")
    mech_regs = defaultdict(list)
    for r in ok:
        mech_regs[r["mechanism"]].append(r["n_regulators"])
    for mech, regs in sorted(mech_regs.items()):
        print(f"  {mech:<30} mean={sum(regs)/len(regs):.1f}  max={max(regs)}")

    print("="*60 + "\n")


def save_summary(rows: list):
    tsv_path = OUTPUTS_DIR / "m14_batch_summary.tsv"
    fields = ["case", "mechanism", "evidence", "n_regulators", "error"]
    with open(tsv_path, "w") as f:
        f.write("\t".join(fields) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in fields) + "\n")

    json_path = OUTPUTS_DIR / "m14_batch_summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "n_total": len(rows),
            "n_ok": sum(1 for r in rows if not r["error"]),
            "rows": rows,
        }, f, indent=2)

    print(f"Summary saved: {tsv_path}")
    print(f"              {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch M14 runner")
    parser.add_argument("--regen-report", action="store_true",
                        help="Regenerate report.md for each case")
    parser.add_argument("--case", default=None,
                        help="Run only this case (e.g. KIF21B_Excitatory)")
    args = parser.parse_args()

    config = load_config()
    case_dirs = get_case_dirs()

    if args.case:
        case_dirs = [d for d in case_dirs if d.name == args.case]
        if not case_dirs:
            print(f"ERROR: case '{args.case}' not found")
            sys.exit(1)

    print(f"Running M14 on {len(case_dirs)} cases...")
    if args.regen_report:
        print("  (--regen-report: report.md will be regenerated)")

    rows = []
    for i, case_dir in enumerate(case_dirs, 1):
        print(f"  [{i:2d}/{len(case_dirs)}] {case_dir.name} ...", end=" ", flush=True)
        try:
            row = run_case(case_dir, config, args.regen_report)
            status = f"{row['mechanism']} / {row['evidence']} / {row['n_regulators']} regs"
            if row["error"]:
                status = f"ERROR: {row['error'][:50]}"
            print(status)
        except Exception as e:
            print(f"EXCEPTION: {e}")
            row = {"case": case_dir.name, "mechanism": "exception",
                   "evidence": "error", "n_regulators": 0, "error": str(e)}
        rows.append(row)

    print_stats(rows)
    save_summary(rows)


if __name__ == "__main__":
    main()
