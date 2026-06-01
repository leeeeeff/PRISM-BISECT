#!/usr/bin/env python3
"""
BISECT v2.0 — Biological Isoform-Switch Evidence Characterization Tool
=======================================================================
Orchestrator — rationally ordered pipeline:

  Stage 1 · Sequence characterization
    M1  extract_seq        — protein sequences from SQANTI3 FAA
    M2  hmmscan            — Pfam domain annotation + Stage 2 filter
    M3  motif_analysis     — catalytic motifs, MTS scoring

  Stage 2 · Genomic context
    M4  genomic_coords     — exon structure, TSS/TTS, strand, NAT detection
    M5  repeatmasker       — transposable element annotation (UCSC rmsk)

  Stage 3 · Translation quality gate
    M6  nmd_screen         — NMD susceptibility (50-nt rule)
                             ⚑ NMD gate: if AD isoform NMD-susceptible,
                               M11/M12 (protein-level) are skipped
    M7  seq_validation     — LINE-1 6-frame → domain identity check
                             (only if M5 young L1 detected AND NMD-resistant)

  Stage 4 · Upstream causal mechanism
    M8  regulatory_context — WHY the switch occurs: splicing/epigenetic/
                             transcriptional mechanism classification
    M9  promoter_usage     — TSS displacement; reclassifies M8 mechanism
                             "transcriptional" → "alternative_promoter" if
                             TSS_diff ≥ threshold
    M10 apa                — 3′ APA / alternative terminal exon; miR-132
                             seed loss in 3′UTR

  Stage 5 · Functional consequence validation
    M11 alphafold          — pLDDT structural confidence of gained/lost
                             domains (skipped if NMD gate active)
    M12 ppi                — STRING network falsification; hypothesis
                             informed by M8 mechanism_type
                             (skipped if NMD gate active)
    M13 conservation       — phyloP 100-way; always runs (NMD itself
                             requires evolutionary context)

  Stage 6 · Output
    M14 report             — analysis.json + domains.tsv + domain_map.pdf
                             + report.md (integrates all stages)
    M15 compare            — cross-case summary TSV (post-pipeline)

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
    t = cfg.get("stage1", {})
    min_delta = t.get("min_abs_diffuse_delta", 0.5)
    max_p = t.get("max_dtu_pvalue", 1e-5)
    try:
        delta = abs(float(row["diffuse_delta"]))
        p = float(row["dtu_pvalue"])
        # Single-condition context (cardiac/psoas/iPSC): no comparison group,
        # p is always 1.0. Apply PRISM-only threshold.
        if p == 1.0:
            prism_min = cfg.get("stage1_prism_only", {}).get(
                "min_abs_diffuse_delta", min_delta
            )
            return delta >= prism_min
        return delta >= min_delta and p <= max_p
    except (KeyError, ValueError):
        return False


def case_output_dir(row: dict, output_root: str) -> str:
    gene = row.get("gene_name", "unknown")
    cell = row.get("cell_type", "unknown").replace(" ", "_")
    return os.path.join(output_root, f"{gene}_{cell}")


def is_completed(case_dir: str) -> bool:
    return os.path.exists(os.path.join(case_dir, "analysis.json"))


# ── Worker function (module-level for pickling) ────────────────────────────────

def _worker(args: tuple) -> dict:
    """Entry point for ProcessPoolExecutor workers."""
    row, config, output_root, dry_run, mode = args
    base = Path(__file__).parent
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    from modules import (
        m1_extract_seq, m2_hmmscan, m3_motif_analysis,
        m4_genomic_coords, m5_repeatmasker,
        m6_nmd_screen, m7_seq_validation,
        m8_regulatory_context, m9_promoter_usage, m10_apa,
        m11_alphafold, m12_ppi, m13_conservation,
        m14_report,
    )
    return _run_case(
        row, config, output_root, dry_run, mode,
        m1_extract_seq, m2_hmmscan, m3_motif_analysis,
        m4_genomic_coords, m5_repeatmasker,
        m6_nmd_screen, m7_seq_validation,
        m8_regulatory_context, m9_promoter_usage, m10_apa,
        m11_alphafold, m12_ppi, m13_conservation,
        m14_report,
    )


# ── NMD gate helper ───────────────────────────────────────────────────────────

def _is_nmd_gated(case_result: dict) -> bool:
    """Return True if AD isoform is NMD-susceptible → skip M11/M12."""
    nmd = case_result.get("m6_nmd_screen", {})
    ad_nmd = nmd.get("ad", {})
    return bool(ad_nmd.get("nmd_susceptible", False))


# ── M9 → M8 mechanism reclassification ───────────────────────────────────────

def _apply_promoter_reclassification(case_result: dict) -> None:
    """
    If M9 flags mechanism_reclassify=True AND M8 classified as 'transcriptional',
    upgrade mechanism_type to 'alternative_promoter' in-place.
    Uses M9's own reclassify flag (set when tss_class contains 'alt_promoter'
    and tss_diff_bp >= threshold) rather than string-matching tss_class directly.
    """
    m9 = case_result.get("m9_promoter_usage", {})
    m8 = case_result.get("m8_regulatory_context", {})
    if not m8 or not m9:
        return
    should_reclassify = m9.get("mechanism_reclassify", False)
    current_mech = m8.get("mechanism_type", "")
    if should_reclassify and current_mech == "transcriptional":
        m8["mechanism_type"] = "alternative_promoter"
        m8["mechanism_reclassified_by"] = "M9"
        m8["tss_diff_bp"] = m9.get("tss_diff_bp")
        case_result["m8_regulatory_context"] = m8


# ── Core per-case pipeline ────────────────────────────────────────────────────

def _run_case(
    row, config, output_root, dry_run, mode,
    m1, m2, m3, m4, m5,
    m6=None, m7=None,
    m8=None, m9=None, m10=None,
    m11=None, m12=None, m13=None,
    m14=None,
) -> dict:
    """
    Execute full BISECT pipeline for one candidate case.

    Execution order (Stage 1–6):
      M1 → M2(gate) → M3 → M4 → M5 →
      M6(NMD gate) → M7(cond.) →
      M8 → M9(reclassify M8) → M10 →
      M11(NMD-gated) → M12(NMD-gated) → M13 →
      M14(report)
    """
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
        "pipeline_version": "2.0",
        "pipeline_ts": datetime.now().isoformat(),
    }

    if dry_run:
        return case_result

    # ══ STAGE 1: Sequence characterization ════════════════════════════════════

    # ── M1: Extract protein sequences ─────────────────────────────────────────
    seq_dict = m1.extract_sequences([ct_id, ad_id], config)
    ct_entry = seq_dict.get(ct_id, {})
    ad_entry = seq_dict.get(ad_id, {})
    ct_seq = ct_entry.get("seq", "")
    ad_seq = ad_entry.get("seq", "")
    case_result["ct_seq"] = {"seq": ct_seq, "length": len(ct_seq), "source": ct_entry.get("source")}
    case_result["ad_seq"] = {"seq": ad_seq, "length": len(ad_seq), "source": ad_entry.get("source")}

    # ── M2: Pfam domain annotation + Stage 2 filter ───────────────────────────
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
        if m14:
            m14.save_json(case_result, case_dir)
            m14.save_domains_tsv(case_result, case_dir)
        return case_result

    case_result["stage2_pass"] = True

    if mode == "screen":
        if m14:
            m14.save_json(case_result, case_dir)
            m14.save_domains_tsv(case_result, case_dir)
        return case_result

    # ── M3: Protein motif analysis (catalytic, MTS, PDZ, LYR …) ──────────────
    if ct_seq:
        case_result["ct_motifs"] = m3.analyze_sequence(ct_seq, config)
    if ad_seq:
        case_result["ad_motifs"] = m3.analyze_sequence(ad_seq, config)

    # ══ STAGE 2: Genomic context ══════════════════════════════════════════════

    # ── M4: Exon structure, TSS/TTS, strand, NAT detection ────────────────────
    try:
        case_result["ct_info"] = m4.get_transcript_info(ct_id, config)
        case_result["ad_info"] = m4.get_transcript_info(ad_id, config)
    except Exception as e:
        case_result["ct_info"] = {}
        case_result["ad_info"] = {"error": str(e)}

    # ── M5: RepeatMasker — transposable element annotation ────────────────────
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

    # ══ STAGE 3: Translation quality gate ════════════════════════════════════

    # ── M6: NMD susceptibility screening ──────────────────────────────────────
    # ⚑ NMD gate: if AD isoform is NMD-susceptible, M11/M12 are skipped
    # because the protein does not reach the ribosome.
    # M13 (conservation) still runs — NMD itself can be functionally selected.
    if m6 is not None:
        try:
            case_result["m6_nmd_screen"] = m6.run(case_result, config)
        except Exception as e:
            case_result["m6_nmd_screen"] = {"summary": f"error: {e}", "nmd_relevant": False}

    nmd_gate = _is_nmd_gated(case_result)
    case_result["nmd_gate_active"] = nmd_gate
    if nmd_gate:
        case_result["nmd_gate_reason"] = (
            "AD isoform NMD-susceptible: M11 (AlphaFold) and M12 (PPI) skipped; "
            "protein-level claims not supported."
        )

    # ── M7: LINE-1 sequence validation (conditional) ─────────────────────────
    # Run only when M5 detected young LINE-1 overlapping CDS AND AD isoform
    # is not NMD-susceptible (otherwise the protein does not exist).
    has_young_l1 = case_result.get("ad_repeats", {}).get("has_l1_in_cds", False)
    if m7 is not None and has_young_l1 and not nmd_gate:
        try:
            case_result["m7_seq_validation"] = m7.run(case_result, config)
        except Exception as e:
            case_result["m7_seq_validation"] = {"skipped": True, "skip_reason": f"error: {e}"}
    elif m7 is not None:
        skip_reason = (
            "no young LINE-1 in CDS" if not has_young_l1
            else "NMD gate active — protein not translated"
        )
        case_result["m7_seq_validation"] = {"skipped": True, "skip_reason": skip_reason}

    # ══ STAGE 4: Upstream causal mechanism ═══════════════════════════════════
    # These modules answer WHY the switch occurs and classify the mechanism
    # BEFORE M11/M12 functional validation so that mechanism type can inform
    # PPI hypothesis generation in M12.

    # ── M8: Regulatory context — splicing / epigenetic / transcriptional ──────
    if m8 is not None:
        try:
            case_result["m8_regulatory_context"] = m8.run(case_result, config)
        except Exception as e:
            case_result["m8_regulatory_context"] = {"error": str(e)}

    # ── M9: Alternative promoter usage — TSS displacement ────────────────────
    # Reclassifies M8 mechanism_type to "alternative_promoter" when warranted.
    if m9 is not None:
        try:
            case_result["m9_promoter_usage"] = m9.run(case_result, config)
            _apply_promoter_reclassification(case_result)
        except Exception as e:
            case_result["m9_promoter_usage"] = {"error": str(e)}

    # ── M10: Alternative polyadenylation — 3′ end changes / miR seeds ─────────
    if m10 is not None:
        try:
            case_result["m10_apa"] = m10.run(case_result, config)
        except Exception as e:
            case_result["m10_apa"] = {"error": str(e)}

    # ══ STAGE 5: Functional consequence validation ════════════════════════════

    # ── M11: AlphaFold / ESMFold structural confidence ────────────────────────
    # Skipped when NMD gate is active (no protein to assess structure for).
    if m11 is not None and not nmd_gate:
        try:
            case_result["m11_alphafold"] = m11.run(case_result, config)
        except Exception as e:
            case_result["m11_alphafold"] = {"error": str(e)}
    elif nmd_gate:
        case_result["m11_alphafold"] = {"skipped": True, "skip_reason": "NMD gate active"}

    # ── M12: PPI network validation (STRING) ──────────────────────────────────
    # Skipped when NMD gate is active.
    # M8 mechanism_type is available here to guide hypothesis generation;
    # M12.run() receives case_result which includes m8_regulatory_context.
    if m12 is not None and not nmd_gate:
        try:
            case_result["m12_ppi"] = m12.run(case_result, config)
        except Exception as e:
            case_result["m12_ppi"] = {"error": str(e)}
    elif nmd_gate:
        case_result["m12_ppi"] = {"skipped": True, "skip_reason": "NMD gate active"}

    # ── M13: Evolutionary conservation (phyloP 100-way) ──────────────────────
    # Always runs — high conservation of an NMD-susceptible exon may indicate
    # that NMD itself is the functionally selected regulatory outcome.
    if m13 is not None:
        try:
            case_result["m13_conservation"] = m13.run(case_result, config)
        except Exception as e:
            case_result["m13_conservation"] = {"error": str(e)}

    # ══ STAGE 6: Output ═══════════════════════════════════════════════════════

    # ── M14: Report — JSON + TSV + domain map + Markdown ─────────────────────
    if m14 is not None:
        m14.save_json(case_result, case_dir)
        m14.save_domains_tsv(case_result, case_dir)
        m14.plot_domain_map(case_result, case_dir, config)
        m14.generate_markdown(case_result, case_dir)

    return case_result


# ── Pipeline run summary ───────────────────────────────────────────────────────

def _write_run_summary(results: list[dict], output_root: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    records = []
    for r in results:
        dc   = r.get("domain_change", {})
        nmd  = r.get("m6_nmd_screen", {})
        sv   = r.get("m7_seq_validation", {})
        reg  = r.get("m8_regulatory_context", {})
        prom = r.get("m9_promoter_usage", {})
        apa  = r.get("m10_apa", {})
        af   = r.get("m11_alphafold", {})
        ppi  = r.get("m12_ppi", {})
        cons = r.get("m13_conservation", {})
        records.append({
            "gene":               r.get("gene_name"),
            "cell_type":          r.get("cell_type"),
            "delta":              r.get("diffuse_delta"),
            "dtu_p":              r.get("dtu_pvalue"),
            "stage2_pass":        r.get("stage2_pass"),
            "domains_lost":       ";".join(dc.get("domains_lost", [])),
            "domains_gained":     ";".join(dc.get("domains_gained", [])),
            "nat":                r.get("ad_info", {}).get("is_nat", False),
            "young_l1_cds":       r.get("ad_repeats", {}).get("has_l1_in_cds", False),
            "nmd_gate_active":    r.get("nmd_gate_active", False),
            "ad_nmd":             nmd.get("ad", {}).get("nmd_susceptible"),
            "seq_val_identity":   sv.get("best_identity"),
            "seq_val_conclusion": sv.get("conclusion", ""),
            "mechanism_type":     reg.get("mechanism_type", ""),
            "tss_class":          prom.get("tss_class", ""),
            "tss_diff_bp":        prom.get("tss_diff_bp"),
            "apa_class":          apa.get("apa_class", ""),
            "tts_diff_bp":        apa.get("tts_diff_bp"),
            "af_ad_plddt_mean":   (af.get("ad") or {}).get("plddt_mean"),
            "af_gained_confident":";".join((af.get("comparison") or {}).get("gained_domain_confident", [])),
            "ppi_verdict":        ppi.get("summary_verdict"),
            "cons_ad_phylop":     (cons.get("summary") or {}).get("ad_specific_mean_phyloP"),
        })

    json_out = os.path.join(output_root, f"run_summary_{ts}.json")
    with open(json_out, "w") as f:
        json.dump(records, f, indent=2, default=str)

    header = (
        "| Gene | Cell | Δ | DTU p | S2 | NMD-gate | Mechanism | TSS class | APA class "
        "| SeqVal% | PPI verdict | PhyloP(AD) |"
    )
    sep = "|------|------|---|-------|----|---------|-----------|-----------|-----------"
    sep += "---------|------------|------------|"

    md_lines = [
        "# BISECT v2.0 — Run Summary",
        f"**Run**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Cases: {len(results)}",
        "",
        header, sep,
    ]
    for rec in records:
        delta  = f"{rec['delta']:+.3f}" if rec["delta"] is not None else "?"
        p_str  = f"{rec['dtu_p']:.1e}"  if rec["dtu_p"]  is not None else "?"
        sv_id  = f"{rec['seq_val_identity']:.1f}" if rec["seq_val_identity"] is not None else "—"
        ppi_v  = rec["ppi_verdict"] or "—"
        phylop = f"{rec['cons_ad_phylop']:.2f}" if rec["cons_ad_phylop"] is not None else "—"
        tss_d  = f"{rec['tss_diff_bp']:,}" if rec["tss_diff_bp"] is not None else "—"
        tts_d  = f"{rec['tts_diff_bp']:,}" if rec["tts_diff_bp"] is not None else "—"
        md_lines.append(
            f"| {rec['gene']} | {rec['cell_type']} | {delta} | {p_str} | "
            f"{'✓' if rec['stage2_pass'] else '✗'} | "
            f"{'⚑' if rec['nmd_gate_active'] else '—'} | "
            f"{rec['mechanism_type'] or '—'} | {rec['tss_class'] or '—'} | "
            f"{rec['apa_class'] or '—'} | {sv_id} | {ppi_v} | {phylop} |"
        )
    md_out = os.path.join(output_root, f"run_summary_{ts}.md")
    with open(md_out, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\n[Run summary] JSON → {json_out}")
    print(f"[Run summary] MD   → {md_out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BISECT v2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--cases",      default="cases_input.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--mode", choices=["screen", "deep"], default="deep",
        help="screen=M1+M2 only; deep=full M1–M14 pipeline",
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (1=sequential)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force",   action="store_true",
                        help="Re-run even if analysis.json exists")
    parser.add_argument("--case",    default=None,
                        help="Run only this gene name")
    args = parser.parse_args()

    base = Path(__file__).parent
    config_path = Path(args.config) if Path(args.config).is_absolute() else base / args.config
    cases_path  = Path(args.cases)  if Path(args.cases).is_absolute()  else base / args.cases

    config    = load_config(str(config_path))
    all_cases = load_cases(str(cases_path))

    output_root = (args.output_dir
                   or config.get("paths", {}).get("output_dir", str(base / "outputs")))
    os.makedirs(output_root, exist_ok=True)
    logger = _setup_logging(output_root)

    print("=" * 70)
    print("  BISECT  v2.0")
    print(f"  Config  : {config_path}")
    print(f"  Cases   : {cases_path}  ({len(all_cases)} rows)")
    print(f"  Output  : {output_root}")
    print(f"  Mode    : {args.mode.upper()}  |  Workers: {args.workers}  |  "
          f"{'DRY-RUN' if args.dry_run else 'Force' if args.force else 'Resume'}")
    print("=" * 70)

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

    results = []
    errors  = []
    t0      = time.time()

    if args.workers == 1:
        from modules import (
            m1_extract_seq, m2_hmmscan, m3_motif_analysis,
            m4_genomic_coords, m5_repeatmasker,
            m6_nmd_screen, m7_seq_validation,
            m8_regulatory_context, m9_promoter_usage, m10_apa,
            m11_alphafold, m12_ppi, m13_conservation,
            m14_report,
        )

    def _run_sequential(row):
        gene = row["gene_name"]
        ct_id = row["ct_transcript_id"]
        ad_id = row["ad_transcript_id"]
        cell  = row.get("cell_type", "?")
        print(f"\n{'─'*70}")
        print(f"  {gene}  [{cell}]  CT:{ct_id[:40]} → AD:{ad_id[:40]}")
        try:
            result = _run_case(
                row, config, output_root, args.dry_run, args.mode,
                m1_extract_seq, m2_hmmscan, m3_motif_analysis,
                m4_genomic_coords, m5_repeatmasker,
                m6_nmd_screen, m7_seq_validation,
                m8_regulatory_context, m9_promoter_usage, m10_apa,
                m11_alphafold, m12_ppi, m13_conservation,
                m14_report,
            )
            passed2  = result.get("stage2_pass", False)
            nmd_gate = result.get("nmd_gate_active", False)
            dc       = result.get("domain_change", {})
            _log(
                f"Done — Stage 2: {'PASS' if passed2 else 'FAIL'} | "
                f"NMD-gate: {'⚑ ON' if nmd_gate else 'off'} | "
                f"lost={dc.get('domains_lost',[])} gained={dc.get('domains_gained',[])}",
                "PASS" if passed2 else "FAIL", logger,
            )
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
        task_args = [(row, config, output_root, args.dry_run, args.mode)
                     for row in work_queue]
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futures = {exe.submit(_worker, a): a[0]["gene_name"] for a in task_args}
            pbar = (tqdm(as_completed(futures), total=len(futures),
                         desc="Cases", unit="case") if _TQDM else as_completed(futures))
            for fut in pbar:
                gene = futures[fut]
                try:
                    res = fut.result()
                    results.append(res)
                    dc       = res.get("domain_change", {})
                    nmd_gate = res.get("nmd_gate_active", False)
                    _log(
                        f"{gene}: Stage 2 {'PASS' if res.get('stage2_pass') else 'FAIL'} "
                        f"NMD-gate={'ON' if nmd_gate else 'off'} "
                        f"lost={dc.get('domains_lost',[])} gained={dc.get('domains_gained',[])}",
                        "PASS" if res.get("stage2_pass") else "FAIL", logger,
                    )
                except Exception as e:
                    _log(f"ERROR {gene}: {e}", "FAIL", logger)
                    errors.append((gene, str(e)))

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  BISECT v2.0 complete in {elapsed:.1f}s")
    stage2_pass = sum(1 for r in results if r.get("stage2_pass"))
    nmd_gated   = sum(1 for r in results if r.get("nmd_gate_active"))
    print(f"  Processed: {len(results)}  |  Stage 2 pass: {stage2_pass}  "
          f"|  NMD-gated: {nmd_gated}  |  Errors: {len(errors)}")
    if errors:
        print("  Failed cases: " + ", ".join(g for g, _ in errors))
    print(f"{'='*70}")

    if results and not args.dry_run:
        _write_run_summary(results, output_root)
        # M15: cross-case comparison table (post-pipeline)
        try:
            from modules import m15_compare
            print("\n  Building cases_summary.tsv (M15)...")
            m15_compare.build_summary(output_root)
        except Exception as e:
            _log(f"M15 failed: {e}", "WARN", logger)


if __name__ == "__main__":
    main()
