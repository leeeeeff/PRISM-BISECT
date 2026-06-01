"""
Recover exon coordinates for NONE structural_category cases by extracting
from the SQANTI3-corrected GTF. Populates ct_info.exons and ad_info.exons
in each analysis.json so M15/M16 can run.

Usage:
    conda activate isoform_env
    python recover_none_exons.py [--dry-run] [--rerun-m15m16]
"""
import json, sys, argparse, re
from pathlib import Path
from collections import defaultdict

PIPELINE_DIR = Path(__file__).parent
OUTPUTS_DIR  = PIPELINE_DIR / "outputs"
GTF_PATH     = Path("/home/dhkim1674/Project_AD_with_refTSS_novel/02_Isoquant_Output/SQANTI3_output/isoforms_corrected.gtf")
SYS_PATH     = str(PIPELINE_DIR)
sys.path.insert(0, SYS_PATH)


# ── GTF parser ────────────────────────────────────────────────────────────────

def parse_gtf_exons(gtf_path: Path) -> dict:
    """Return {transcript_id: {"chrom": str, "strand": str, "exons": [[start, end], ...]}}."""
    tx2exons: dict = defaultdict(lambda: {"chrom": "", "strand": "", "exons": []})
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            chrom, start, end, strand = parts[0], int(parts[3]), int(parts[4]), parts[6]
            attrs = parts[8]
            m = re.search(r'transcript_id "([^"]+)"', attrs)
            if not m:
                continue
            tid = m.group(1)
            tx2exons[tid]["chrom"]  = chrom
            tx2exons[tid]["strand"] = strand
            tx2exons[tid]["exons"].append([start, end])
    # Sort exons by start
    for tid in tx2exons:
        tx2exons[tid]["exons"].sort(key=lambda e: e[0])
    return dict(tx2exons)


# ── Case update ───────────────────────────────────────────────────────────────

def needs_recovery(case_dir: Path) -> bool:
    try:
        d = json.load(open(case_dir / "analysis.json"))
        ct = d.get("ct_info") or {}
        ad = d.get("ad_info") or {}
        return not ct.get("exons") or not ad.get("exons")
    except Exception:
        return False


def update_case(case_dir: Path, tx2exons: dict, dry_run: bool) -> str:
    d = json.load(open(case_dir / "analysis.json"))

    ct_tid = d.get("ct_transcript_id", "")
    ad_tid = d.get("ad_transcript_id", "")
    ct_info = d.get("ct_info") or {}
    ad_info = d.get("ad_info") or {}

    ct_found = ct_tid in tx2exons
    ad_found = ad_tid in tx2exons

    if not ct_found and not ad_found:
        return f"SKIP — neither {ct_tid} nor {ad_tid} found in GTF"

    updated = []
    if ct_found and not ct_info.get("exons"):
        rec = tx2exons[ct_tid]
        ct_info["exons"]  = rec["exons"]
        ct_info["chrom"]  = ct_info.get("chrom") or rec["chrom"]
        ct_info["strand"] = ct_info.get("strand") or rec["strand"]
        d["ct_info"] = ct_info
        updated.append(f"ct({len(rec['exons'])} exons)")

    if ad_found and not ad_info.get("exons"):
        rec = tx2exons[ad_tid]
        ad_info["exons"]  = rec["exons"]
        ad_info["chrom"]  = ad_info.get("chrom") or rec["chrom"]
        ad_info["strand"] = ad_info.get("strand") or rec["strand"]
        d["ad_info"] = ad_info
        updated.append(f"ad({len(rec['exons'])} exons)")

    if not updated:
        return "SKIP — exons already present"

    if not dry_run:
        with open(case_dir / "analysis.json", "w") as f:
            json.dump(d, f, indent=2, default=str)

    return f"UPDATED {', '.join(updated)}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",      action="store_true",
                        help="Report what would change without writing")
    parser.add_argument("--rerun-m15m16", action="store_true",
                        help="After recovery, run M15+M16 batch (--regen-report)")
    args = parser.parse_args()

    case_dirs = sorted([d for d in OUTPUTS_DIR.iterdir()
                        if d.is_dir() and (d / "analysis.json").exists()])
    need = [d for d in case_dirs if needs_recovery(d)]
    print(f"Cases needing exon recovery: {len(need)} / {len(case_dirs)}")

    if not need:
        print("Nothing to do."); return

    print(f"\nLoading GTF: {GTF_PATH}")
    tx2exons = parse_gtf_exons(GTF_PATH)
    print(f"  Loaded {len(tx2exons):,} transcripts")

    ok = skip = fail = 0
    for case_dir in need:
        result = update_case(case_dir, tx2exons, args.dry_run)
        status = "OK" if result.startswith("UPDATED") else "SKIP" if result.startswith("SKIP") else "FAIL"
        if   status == "OK":   ok   += 1
        elif status == "SKIP": skip += 1
        else:                  fail += 1
        print(f"  {case_dir.name:<35} {result}")

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Updated={ok}  Skipped={skip}  Failed={fail}")

    if args.rerun_m15m16 and not args.dry_run and ok > 0:
        import subprocess
        cmd = [sys.executable,
               str(PIPELINE_DIR / "run_m15_m16_batch.py"),
               "--regen-report"]
        print(f"\nRe-running M15+M16 batch: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
