#!/usr/bin/env python3
"""
BISECT — Biological Isoform-Switch Evidence Characterization Tool
v1.1 Orchestrator
=================================================================
Runs M1→M2→(Stage 2 filter)→M3→M4→M5→M8→M6 for each case in cases_input.csv.
Supports parallel workers, resume, --force, tqdm, pipeline.log, and M7 summary.

Stage 1 filter : |DIFFUSE Δ| ≥ threshold AND DTU p-value ≤ threshold
Stage 2 filter : domain change detected between CT and AD isoforms
Output per case: outputs/<gene>_<cell_type>/analysis.json + domains.tsv
                 domain_map.pdf/png + report.md (Jinja2 or inline)

Usage:
    python orchestrate.py [options]
    python orchestrate.py --dry-run
    python orchestrate.py --case KIF21B --force
    python orchestrate.py --workers 4 --mode deep
"""
import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml
try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

# ── Path setup (needed for worker processes too) ───────────────────────────────
_BASE = Path(__file__).parent
sys.path.insert(0, str(_BASE))


# ── Logging ────────────────────────────────────────────────────────────────────

def _setup_logging(output_root: str) -> logging.Logger:
    log_path = os.path.join(output_root, "pipeline.log")
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                          datefmt="%H:%M:%S"))
        logger.addHandler(fh)
    return logger


def _log(msg: str, level: str = "INFO", logger: logging.Logger = None):
    ts = datetime.now().strftime("%H:%M:%S")
    sym = {"INFO": "  ", "PASS": "✓ ", "FAIL": "✗ ", "WARN": "⚠ ", "STEP": "► "}.get(level, "  ")
    line = f"[{ts}] {sym}{msg}"
    print(line, flush=True)
    if logger:
        getattr(logger, level.lower() if level in ("INFO", "WARN") else "info")(msg)


# ── Config / Cases ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_cases(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def stage1_pass(row: dict, cfg: dict) -> bool:
    """Stage 1: |DIFFUSE Δ| and DTU p-value filter."""
    t = cfg.get("stage1", {})
    min_delta = t.get("min_abs_diffuse_delta", 0.5)
    max_p = t.get("max_dtu_pvalue", 1e-5)
    try:
        return abs(float(row["diffuse_delta"])) >= min_delta and \
               float(row["dtu_pvalue"]) <= max_p
    except (KeyError, ValueError):
        return False


def case_output_dir(row: dict, output_root: str) -> str:
    gene = row.get("gene_name", "unknown")
    cell = row.get("cell_type", "unknown").replace(" ", "_")
    return os.path.join(output_root, f"{gene}_{cell}")


def is_completed(case_dir: str) -> bool:
    """Return True if analysis.json already exists for this case."""
    return os.path.exists(os.path.join(case_dir, "analysis.json"))


# ── Worker function (module-level for pickling) ────────────────────────────────

def _worker(args: tuple) -> dict:
    """Entry point for ProcessPoolExecutor workers."""
    row, config, output_root, dry_run, mode = args
    # Ensure path is set up in worker process (spawn mode safety)
    base = Path(__file__).parent
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    from modules import m1_extract_seq, m2_hmmscan, m3_motif_analysis
    from modules import m4_genomic_coords, m5_repeatmasker, m6_report
    from modules import m8_seq_validation, m9_nmd_screen
    from modules import m10_alphafold, m11_ppi, m12_conservation
    return _run_case(row, config, output_root, dry_run, mode,
                     m1_extract_seq, m2_hmmscan, m3_motif_analysis,
                     m4_genomic_coords, m5_repeatmasker, m6_report,
                     m8_seq_validation, m9_nmd_screen,
                     m10_alphafold, m11_ppi, m12_conservation)


def _run_case(row, config, output_root, dry_run, mode,
              m1, m2, m3, m4, m5, m6, m8=None, m9=None,
              m10=None, m11=None, m12=None) -> dict:
    """Execute M1→M6 for one case. Returns case_result dict."""
    gene = row["gene_name"]
    ct_id = row["ct_transcript_id"]
    ad_id = row["ad_transcript_id"]
    cell_type = row["cell_type"]
    case_dir = case_output_dir(row, output_root)
    os.makedirs(case_dir, exist_ok=True)

    case_result = {
        "gene_name": gene,
        "ct_transcript_id": ct_id,
        "ad_transcript_id": ad_id,
        "cell_type": cell_type,
        "diffuse_delta": float(row["diffuse_delta"]),
        "dtu_pvalue": float(row["dtu_pvalue"]),
        "direction": row.get("direction", ""),
        "priority": row.get("priority", ""),
        "pipeline_ts": datetime.now().isoformat(),
    }

    if dry_run:
        return case_result

    # ── M1: Extract sequences ──────────────────────────────────────────────────
    seq_dict = m1.extract_sequences([ct_id, ad_id], config)
    ct_entry = seq_dict.get(ct_id, {})
    ad_entry = seq_dict.get(ad_id, {})
    ct_seq = ct_entry.get("seq", "")
    ad_seq = ad_entry.get("seq", "")
    case_result["ct_seq"] = {"seq": ct_seq, "length": len(ct_seq), "source": ct_entry.get("source")}
    case_result["ad_seq"] = {"seq": ad_seq, "length": len(ad_seq), "source": ad_entry.get("source")}

    # ── M2: hmmscan + Stage 2 filter ──────────────────────────────────────────
    fasta_path = os.path.join(case_dir, "targets.faa")
    m1.write_fasta(seq_dict, fasta_path)
    domain_hits = m2.run_hmmscan(fasta_path, config, case_dir)
    ct_domains = domain_hits.get(ct_id, [])
    ad_domains = domain_hits.get(ad_id, [])
    case_result["ct_domains"] = ct_domains
    case_result["ad_domains"] = ad_domains

    domain_change = m2.detect_domain_changes(ct_domains, ad_domains)
    case_result["domain_change"] = domain_change

    require_change = config.get("stage2", {}).get("require_domain_change", True)
    if require_change and not domain_change["has_domain_change"]:
        case_result["stage2_pass"] = False
        m6.save_json(case_result, case_dir)
        m6.save_domains_tsv(case_result, case_dir)
        return case_result

    case_result["stage2_pass"] = True

    if mode == "screen":
        m6.save_json(case_result, case_dir)
        m6.save_domains_tsv(case_result, case_dir)
        return case_result

    # ── M3: Motif / MTS ───────────────────────────────────────────────────────
    if ct_seq:
        case_result["ct_motifs"] = m3.analyze_sequence(ct_seq, config)
    if ad_seq:
        case_result["ad_motifs"] = m3.analyze_sequence(ad_seq, config)

    # ── M4: Genomic coordinates + NAT detection ────────────────────────────────
    try:
        case_result["ct_info"] = m4.get_transcript_info(ct_id, config)
        case_result["ad_info"] = m4.get_transcript_info(ad_id, config)
    except Exception:
        case_result["ct_info"] = {}
        case_result["ad_info"] = {}

    # ── M5: RepeatMasker ───────────────────────────────────────────────────────
    rate_delay = config.get("repeatmasker_api", {}).get("rate_limit_delay", 1.0)
    for isoform_key, info_dict in [("ad_repeats", case_result.get("ad_info", {})),
                                    ("ct_repeats", case_result.get("ct_info", {}))]:
        chrom = info_dict.get("chrom")
        exons = info_dict.get("exons", [])
        if chrom and exons:
            try:
                case_result[isoform_key] = m5.annotate_exons(
                    exons, chrom,
                    info_dict.get("cds_start_genomic"),
                    info_dict.get("cds_end_genomic"),
                    config,
                )
                time.sleep(rate_delay)
            except Exception as e:
                case_result[isoform_key] = {"error": str(e)}
        else:
            case_result[isoform_key] = {"skipped": "no chrom/exons"}

    # ── M8: Genomic sequence validation (young LINE-1 elements only) ─────────
    if m8 is not None:
        try:
            case_result["seq_validation"] = m8.run(case_result, config)
        except Exception as e:
            case_result["seq_validation"] = {"skipped": True, "skip_reason": f"error: {e}"}

    # ── M9: NMD susceptibility screening ──────────────────────────────────────
    if m9 is not None:
        try:
            case_result["nmd_screen"] = m9.run(case_result, config)
        except Exception as e:
            case_result["nmd_screen"] = {"summary": f"error: {e}", "nmd_relevant": False}

    # ── M10: AlphaFold structural confidence ───────────────────────────────────
    if m10 is not None:
        try:
            case_result["m10_alphafold"] = m10.run(case_result, config)
        except Exception as e:
            case_result["m10_alphafold"] = {"error": str(e)}

    # ── M11: PPI network validation ────────────────────────────────────────────
    if m11 is not None:
        try:
            case_result["m11_ppi"] = m11.run(case_result, config)
        except Exception as e:
            case_result["m11_ppi"] = {"error": str(e)}

    # ── M12: Evolutionary conservation ────────────────────────────────────────
    if m12 is not None:
        try:
            case_result["m12_conservation"] = m12.run(case_result, config)
        except Exception as e:
            case_result["m12_conservation"] = {"error": str(e)}

    # ── M6: JSON + domains.tsv + Figure + Markdown ────────────────────────────
    m6.save_json(case_result, case_dir)
    m6.save_domains_tsv(case_result, case_dir)
    m6.plot_domain_map(case_result, case_dir, config)
    m6.generate_markdown(case_result, case_dir)

    return case_result


# ── Pipeline summary ───────────────────────────────────────────────────────────

def _write_run_summary(results: list[dict], output_root: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    records = []
    for r in results:
        dc = r.get("domain_change", {})
        sv = r.get("seq_validation", {})
        nmd = r.get("nmd_screen", {})
        af = r.get("m10_alphafold", {})
        ppi = r.get("m11_ppi", {})
        cons = r.get("m12_conservation", {})
        records.append({
            "gene": r.get("gene_name"),
            "cell_type": r.get("cell_type"),
            "delta": r.get("diffuse_delta"),
            "dtu_p": r.get("dtu_pvalue"),
            "stage2_pass": r.get("stage2_pass"),
            "domains_lost": ";".join(dc.get("domains_lost", [])),
            "domains_gained": ";".join(dc.get("domains_gained", [])),
            "nat": r.get("ad_info", {}).get("is_nat", False),
            "young_l1_cds": r.get("ad_repeats", {}).get("has_l1_in_cds", False),
            "seq_val_identity": sv.get("best_identity", None),
            "seq_val_conclusion": sv.get("conclusion", ""),
            "ad_nmd": nmd.get("ad", {}).get("nmd_susceptible", None),
            "nmd_relevant": nmd.get("nmd_relevant", False),
            "af_ad_plddt_mean": (af.get("ad") or {}).get("plddt_mean"),
            "af_gained_confident": ";".join((af.get("comparison") or {}).get("gained_domain_confident", [])),
            "ppi_verdict": ppi.get("summary_verdict"),
            "cons_ad_phylop": (cons.get("summary") or {}).get("ad_specific_mean_phyloP"),
        })

    json_out = os.path.join(output_root, f"run_summary_{ts}.json")
    with open(json_out, "w") as f:
        json.dump(records, f, indent=2, default=str)

    md_lines = [
        "# BISECT — Run Summary",
        f"**Run**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Cases: {len(results)}",
        "",
        "| Gene | Cell type | DIFFUSE Δ | DTU p | Stage 2 | Domains lost | Domains gained | NAT | Young L1 | SeqVal ID% | AD NMD | AF AD pLDDT | AF Confident Gain | PPI Verdict | PhyloP (AD) |",
        "|------|-----------|-----------|-------|---------|--------------|----------------|-----|----------|-----------|--------|-------------|-------------------|-------------|-------------|",
    ]
    for rec in records:
        delta = f"{rec['delta']:+.3f}" if rec["delta"] is not None else "?"
        p_str = f"{rec['dtu_p']:.1e}" if rec["dtu_p"] is not None else "?"
        sv_id = f"{rec['seq_val_identity']:.1f}" if rec["seq_val_identity"] is not None else "—"
        ad_nmd_str = ("YES" if rec["ad_nmd"] else "no") if rec["ad_nmd"] is not None else "?"
        af_plddt = f"{rec['af_ad_plddt_mean']:.1f}" if rec["af_ad_plddt_mean"] is not None else "—"
        af_gain = rec["af_gained_confident"] or "—"
        ppi_v = rec["ppi_verdict"] or "—"
        phylop = f"{rec['cons_ad_phylop']:.2f}" if rec["cons_ad_phylop"] is not None else "—"
        md_lines.append(
            f"| {rec['gene']} | {rec['cell_type']} | {delta} | {p_str} | "
            f"{'✓' if rec['stage2_pass'] else '✗'} | "
            f"{rec['domains_lost'] or '—'} | {rec['domains_gained'] or '—'} | "
            f"{'YES' if rec['nat'] else 'no'} | {'YES' if rec['young_l1_cds'] else 'no'} | "
            f"{sv_id} | {ad_nmd_str} | "
            f"{af_plddt} | {af_gain} | {ppi_v} | {phylop} |"
        )
    md_out = os.path.join(output_root, f"run_summary_{ts}.md")
    with open(md_out, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\n[Run summary] JSON → {json_out}")
    print(f"[Run summary] MD   → {md_out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BISECT v1.1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cases", default="cases_input.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode", choices=["screen", "deep"], default="deep",
                        help="screen=M1+M2 only; deep=M1-M6 full")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (1=sequential)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if analysis.json already exists")
    parser.add_argument("--case", default=None, help="Run only this gene name")
    args = parser.parse_args()

    base = Path(__file__).parent
    config_path = Path(args.config) if Path(args.config).is_absolute() else base / args.config
    cases_path = Path(args.cases) if Path(args.cases).is_absolute() else base / args.cases

    config = load_config(str(config_path))
    all_cases = load_cases(str(cases_path))

    output_root = args.output_dir or config.get("paths", {}).get("output_dir", str(base / "outputs"))
    os.makedirs(output_root, exist_ok=True)

    logger = _setup_logging(output_root)

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  BISECT  v1.1")
    print(f"  Config  : {config_path}")
    print(f"  Cases   : {cases_path}  ({len(all_cases)} rows)")
    print(f"  Output  : {output_root}")
    print(f"  Mode    : {args.mode.upper()}  |  Workers: {args.workers}  |  "
          f"{'DRY-RUN' if args.dry_run else 'Force' if args.force else 'Resume'}")
    print("=" * 70)

    # ── Filter cases ──────────────────────────────────────────────────────────
    work_queue = []
    skipped_s1 = skipped_done = 0

    for row in all_cases:
        gene = row.get("gene_name", "?")
        if args.case and gene.upper() != args.case.upper():
            continue
        if not stage1_pass(row, config):
            _log(f"Stage 1 FAIL: {gene} ({row.get('cell_type','?')}) "
                 f"|Δ|={abs(float(row.get('diffuse_delta',0))):.3f} "
                 f"p={row.get('dtu_pvalue','?')}", "FAIL", logger)
            skipped_s1 += 1
            continue
        cdir = case_output_dir(row, output_root)
        if not args.force and not args.dry_run and is_completed(cdir):
            _log(f"Resume: {gene} already done → skipping (use --force to rerun)", "INFO", logger)
            skipped_done += 1
            continue
        work_queue.append(row)

    print(f"\n  Queued: {len(work_queue)}  |  Stage 1 skipped: {skipped_s1}  |  "
          f"Already done (skipped): {skipped_done}")

    if not work_queue:
        print("\n  Nothing to do. Use --force to re-run completed cases.")
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    results = []
    errors = []
    t0 = time.time()

    # Import modules for sequential mode (avoids repeated imports)
    if args.workers == 1:
        from modules import m1_extract_seq, m2_hmmscan, m3_motif_analysis
        from modules import m4_genomic_coords, m5_repeatmasker, m6_report
        from modules import m8_seq_validation, m9_nmd_screen
        from modules import m10_alphafold, m11_ppi, m12_conservation

    def _run_sequential(row):
        gene = row["gene_name"]
        ct_id = row["ct_transcript_id"]
        ad_id = row["ad_transcript_id"]
        cell = row.get("cell_type", "?")
        print(f"\n{'─'*70}")
        print(f"  {gene}  [{cell}]  CT:{ct_id[:40]} → AD:{ad_id[:40]}")
        try:
            result = _run_case(row, config, output_root, args.dry_run, args.mode,
                               m1_extract_seq, m2_hmmscan, m3_motif_analysis,
                               m4_genomic_coords, m5_repeatmasker, m6_report,
                               m8_seq_validation, m9_nmd_screen,
                               m10_alphafold, m11_ppi, m12_conservation)
            passed2 = result.get("stage2_pass", False)
            dc = result.get("domain_change", {})
            _log(f"Done — Stage 2: {'PASS' if passed2 else 'FAIL'} | "
                 f"lost={dc.get('domains_lost',[])} gained={dc.get('domains_gained',[])}",
                 "PASS" if passed2 else "FAIL", logger)
            return result, None
        except Exception as e:
            tb = traceback.format_exc()
            _log(f"ERROR: {gene}: {e}", "FAIL", logger)
            logger.error(tb)
            return None, (gene, str(e))

    iterator = work_queue
    if _TQDM and len(work_queue) > 1:
        iterator = tqdm(work_queue, desc="Cases", unit="case",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")

    if args.workers == 1:
        for row in iterator:
            res, err = _run_sequential(row)
            if res is not None:
                results.append(res)
            if err:
                errors.append(err)
    else:
        # Parallel mode — workers > 1
        task_args = [(row, config, output_root, args.dry_run, args.mode)
                     for row in work_queue]
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futures = {exe.submit(_worker, a): a[0]["gene_name"] for a in task_args}
            if _TQDM:
                pbar = tqdm(as_completed(futures), total=len(futures),
                            desc="Cases", unit="case")
            else:
                pbar = as_completed(futures)
            for fut in pbar:
                gene = futures[fut]
                try:
                    res = fut.result()
                    results.append(res)
                    dc = res.get("domain_change", {})
                    _log(f"{gene}: Stage 2 {'PASS' if res.get('stage2_pass') else 'FAIL'} "
                         f"lost={dc.get('domains_lost',[])} gained={dc.get('domains_gained',[])}",
                         "PASS" if res.get("stage2_pass") else "FAIL", logger)
                except Exception as e:
                    _log(f"ERROR {gene}: {e}", "FAIL", logger)
                    errors.append((gene, str(e)))

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed:.1f}s")
    stage2_pass = sum(1 for r in results if r.get("stage2_pass"))
    print(f"  Processed: {len(results)}  |  Stage 2 pass: {stage2_pass}  |  Errors: {len(errors)}")
    if errors:
        print("  Failed cases: " + ", ".join(g for g, _ in errors))
    print(f"{'='*70}")

    if results and not args.dry_run:
        _write_run_summary(results, output_root)
        # M7: cross-case comparison table
        try:
            from modules import m7_compare
            print("\n  Building cases_summary.tsv (M7)...")
            m7_compare.build_summary(output_root)
        except Exception as e:
            _log(f"M7 failed: {e}", "WARN", logger)


if __name__ == "__main__":
    main()
