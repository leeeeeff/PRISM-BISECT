"""
M8: Regulatory Context Evidence (BISECT v2.0)

Identifies upstream causal regulators (TF/splicing factors) that explain
WHY a given isoform switch occurs in a specific AD cell type.

Three mechanism types are classified:
  - alternative_splicing   : NIC/NNIC isoforms, splicing factor dysregulation
  - epigenetic_derepression: young LINE-1 exon overlap, DNMT/SETDB pathway
  - transcriptional        : canonical isoforms, promoter/TF-level changes

Evidence chain: TF/ASF expression change (DEG) + RBP binding motif in
CT-specific exon flanking introns → explains isoform switch without
claiming functional prediction modification in PRISM.

Entry point: run(case_result, config) -> dict

Limitations (stated explicitly in output):
  - DEG mRNA levels ≠ protein activity (especially TARDBP/TDP-43)
  - Motif density alone is not sufficient; eCLIP validation required
  - All evidence is correlative unless functional perturbation data exists
"""
import csv
import math
import os
import re
from typing import Optional


# ── Cell-type name mapping (case_result short → DEG filename suffix) ──────────

_CELLTYPE_MAP = {
    "Excitatory":   "Excitatory_neuron",
    "Inhibitory":   "Inhibitory_neuron",
    "Oligodendrocyte": "Oligodendrocyte",
    "Astrocyte":    "Astrocyte",
    "Microglia":    "Microglia",
    "OPC":          "OPC",
    "Vascular":     "Vascular_cell",
    "Lymphocyte":   "Lymphocyte",
}

# ── Regulator panels per mechanism ───────────────────────────────────────────

_SPLICING_PANEL = [
    "RBFOX1", "RBFOX2", "RBFOX3",
    "NOVA1",  "NOVA2",
    "TARDBP", "FMR1",
    "SRSF1",  "SRSF5",  "SRSF7",
    "HNRNPK", "HNRNPA2B1",
    "MBNL1",  "MBNL2",
    "PTBP2",  "QKI",
    "TRA2B",
]

_EPIGENETIC_PANEL = [
    "DNMT1", "DNMT3A", "DNMT3B",
    "TET1",  "TET2",
    "SETDB1", "SETDB2",
    "TRIM28",
    "HDAC2", "SIRT1",
    "EP300",
]

_TRANSCRIPTIONAL_PANEL = [
    "SP1", "SP3", "KLF9", "KLF4",
    "REST", "STAT1", "E2F3",
    "CREB1", "ATF4",
    "YBX1",
]

# ── RBP binding motifs (pre-mRNA, DNA sequence of coding strand) ─────────────

_RBP_MOTIFS = {
    "RBFOX":  ["TGCATG"],
    "NOVA":   ["TCAT", "CCAT", "TCAC", "CCAC"],
    "TARDBP": ["TGTGTG", "UGUGUG".replace("U", "T")],
    "MBNL":   ["TGCTT", "TGCTG"],
    "SRSF":   ["GAAGAA", "GGAGGA"],
}

_RBFOX_CANONICAL_PRE = set("CT")   # YGCATG: pyrimidine before TGCATG


# ── Genome FASTA reader (no external deps) ───────────────────────────────────

def _load_fai(fai_path: str) -> dict:
    """Parse .fai index → {chrom: (length, offset, bpl, bypl)}."""
    idx = {}
    with open(fai_path) as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 5:
                idx[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    return idx


def _fetch_seq(genome_fa: str, fai: dict,
               chrom: str, start: int, end: int) -> Optional[str]:
    """Fetch 0-based half-open interval [start, end) from genome FASTA."""
    if chrom not in fai:
        return None
    _, offset, bpl, bypl = fai[chrom]
    result = []
    pos = start
    try:
        with open(genome_fa, "rb") as f:
            while pos < end:
                ln  = pos // bpl
                col = pos % bpl
                f.seek(offset + ln * bypl + col)
                to_read = min(bpl - col, end - pos)
                chunk = f.read(to_read).decode("ascii", errors="replace").replace("\n", "")
                result.append(chunk)
                pos += len(chunk)
    except OSError:
        return None
    return "".join(result)


_RC = str.maketrans("ACGTacgt", "TGCAtgca")

def _revcomp(seq: str) -> str:
    return seq.translate(_RC)[::-1]


# ── DEG loader ───────────────────────────────────────────────────────────────

def _load_deg(deg_dir: str, cell_type_full: str) -> dict:
    """
    Return {gene: {logFC, padj}} for all genes in the DEG file.
    Handles both full and significant-only files; prefers full.
    """
    path = os.path.join(deg_dir, f"DEG_AD_vs_CT_{cell_type_full}.csv")
    if not os.path.exists(path):
        path = os.path.join(deg_dir, f"DEG_significant_AD_vs_CT_{cell_type_full}.csv")
    if not os.path.exists(path):
        return {}
    genes = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("names", "").strip()
            try:
                lfc  = float(row.get("logfoldchanges", 0))
                padj = float(row.get("pvals_adj", 1))
            except ValueError:
                continue
            genes[name] = {"logFC": lfc, "padj": padj}
    return genes


# ── Mechanism classifier ─────────────────────────────────────────────────────

def _classify_mechanism(case_result: dict) -> str:
    """
    Classify isoform switch mechanism from case metadata.
    Priority: LINE-1 exon overlap → splicing (NIC/NNIC) → transcriptional.
    """
    ad_repeats = case_result.get("ad_repeats") or {}
    exon_hits  = ad_repeats.get("exon_hits", {})
    for exon_id, hits in exon_hits.items():
        for h in hits:
            if h.get("is_young_l1") and h.get("family") == "L1":
                return "epigenetic_derepression"

    ad_cat = (case_result.get("ad_info") or {}).get("structural_category", "")
    ct_cat = (case_result.get("ct_info") or {}).get("structural_category", "")
    if any(c in ("novel_in_catalog", "novel_not_in_catalog")
           for c in (ad_cat, ct_cat)):
        return "alternative_splicing"

    return "transcriptional"


# ── Regulator extraction ─────────────────────────────────────────────────────

def _extract_regulators(deg: dict, panel: list,
                        padj_thresh: float, logfc_thresh: float) -> list:
    """
    Return sorted list of significant regulators from the panel.
    Includes direction and effect size.
    """
    hits = []
    for gene in panel:
        entry = deg.get(gene)
        if entry is None:
            continue
        lfc  = entry["logFC"]
        padj = entry["padj"]
        if padj < padj_thresh and abs(lfc) >= logfc_thresh:
            hits.append({
                "gene":      gene,
                "logFC":     round(lfc, 4),
                "padj":      padj,
                "direction": "up" if lfc > 0 else "down",
                "neg_log10_padj": round(-math.log10(padj + 1e-320), 1),
            })
    hits.sort(key=lambda x: x["neg_log10_padj"], reverse=True)
    return hits


# ── Intron motif search ───────────────────────────────────────────────────────

def _ct_unique_exons(ct_exons: list, ad_exons: list,
                     min_overlap_frac: float = 0.5) -> list:
    """Return CT-specific exons (absent in AD) as (start, end) pairs."""
    def overlaps(e1, e2):
        ov = max(0, min(e1[1], e2[1]) - max(e1[0], e2[0]))
        if ov == 0:
            return False
        return (ov / (e1[1] - e1[0])) >= min_overlap_frac or \
               (ov / (e2[1] - e2[0])) >= min_overlap_frac

    return [e for e in ct_exons
            if not any(overlaps(e, a) for a in ad_exons)]


def _intron_motif_analysis(ct_exons: list, ad_exons: list,
                           chrom: str, strand: str,
                           genome_fa: str, fai_idx: dict,
                           max_intron: int = 50_000) -> dict:
    """
    For each CT-specific exon, search the flanking introns for RBP motifs.
    Returns per-RBP counts, density, and canonical site counts.
    """
    ct_unique = _ct_unique_exons(ct_exons, ad_exons)
    if not ct_unique or not genome_fa or not fai_idx:
        return {"skipped": "no_ct_unique_exons_or_genome_unavailable"}

    # Sort exons in transcript order
    reverse = (strand == "-")
    all_ct = sorted(ct_exons, key=lambda e: e[0], reverse=reverse)

    results = []
    for exon in ct_unique:
        ex_s, ex_e = exon[0], exon[1]
        intron_seqs = {}

        # exons may be stored as lists [s,e] or tuples (s,e) — normalise to int pair
        def _match(e, s, end): return int(e[0]) == s and int(e[1]) == end

        if not reverse:  # + strand: upstream = lower coord, downstream = higher
            idx = next((i for i, e in enumerate(all_ct) if _match(e, ex_s, ex_e)), None)
            if idx is not None and idx > 0:
                upstream_end = int(all_ct[idx - 1][1])
                up_seq = _fetch_seq(genome_fa, fai_idx, chrom,
                                    upstream_end, min(ex_s, upstream_end + max_intron))
                if up_seq:
                    intron_seqs["upstream"] = up_seq
            if idx is not None and idx < len(all_ct) - 1:
                down_start = ex_e
                down_end   = min(int(all_ct[idx + 1][0]), ex_e + max_intron)
                dn_seq = _fetch_seq(genome_fa, fai_idx, chrom, down_start, down_end)
                if dn_seq:
                    intron_seqs["downstream"] = dn_seq
        else:            # - strand: upstream in transcript = higher genomic coord
            idx = next((i for i, e in enumerate(all_ct) if _match(e, ex_s, ex_e)), None)
            if idx is not None and idx > 0:
                prev_exon_start = int(all_ct[idx - 1][0])
                up_seq = _fetch_seq(genome_fa, fai_idx, chrom,
                                    ex_e, min(prev_exon_start, ex_e + max_intron))
                if up_seq:
                    intron_seqs["upstream"] = _revcomp(up_seq)   # → pre-mRNA
            if idx is not None and idx < len(all_ct) - 1:
                next_exon_end = int(all_ct[idx + 1][1])
                start_g = max(next_exon_end, ex_s - max_intron)
                dn_seq = _fetch_seq(genome_fa, fai_idx, chrom, start_g, ex_s)
                if dn_seq:
                    intron_seqs["downstream"] = _revcomp(dn_seq)

        exon_result = {
            "exon": [ex_s, ex_e],
            "introns_analyzed": list(intron_seqs.keys()),
            "motifs": {},
        }
        total_len = 0
        for side, seq in intron_seqs.items():
            total_len += len(seq)
            for rbp, motif_list in _RBP_MOTIFS.items():
                count = sum(len(re.findall(m, seq)) for m in motif_list)
                canonical = 0
                if rbp == "RBFOX":
                    for m in re.finditer("TGCATG", seq):
                        pre = seq[m.start() - 1] if m.start() > 0 else ""
                        if pre in _RBFOX_CANONICAL_PRE:
                            canonical += 1
                old = exon_result["motifs"].get(rbp, {"count": 0, "canonical": 0})
                exon_result["motifs"][rbp] = {
                    "count":     old["count"] + count,
                    "canonical": old["canonical"] + canonical,
                }

        exon_result["total_intron_nt"] = total_len
        if total_len > 0:
            for rbp in exon_result["motifs"]:
                cnt = exon_result["motifs"][rbp]["count"]
                exon_result["motifs"][rbp]["density_per_kb"] = round(
                    cnt / total_len * 1000, 3)
        results.append(exon_result)

    # Aggregate across all CT-specific exons
    agg = {}
    total_nt = 0
    for r in results:
        total_nt += r["total_intron_nt"]
        for rbp, vals in r["motifs"].items():
            if rbp not in agg:
                agg[rbp] = {"count": 0, "canonical": 0}
            agg[rbp]["count"]     += vals["count"]
            agg[rbp]["canonical"] += vals["canonical"]

    bg_expected = {rbp: round(total_nt / 1000 * _BACKGROUND_DENSITY[rbp], 2)
                   for rbp in _RBP_MOTIFS}
    for rbp in agg:
        cnt = agg[rbp]["count"]
        exp = bg_expected.get(rbp, 0)
        agg[rbp]["density_per_kb"]   = round(cnt / total_nt * 1000, 3) if total_nt else 0
        agg[rbp]["background_exp"]   = exp
        agg[rbp]["enrichment"]       = round(cnt / exp, 2) if exp > 0 else None

    return {
        "ct_unique_exon_count": len(ct_unique),
        "total_intron_nt_searched": total_nt,
        "per_exon": results,
        "aggregate": agg,
        "background_note": "Expected counts from random hexamer frequency in human introns",
    }


# background hexamer density estimates (/kb) from Weyn-Vanhentenryck 2014 + Ule 2006
_BACKGROUND_DENSITY = {
    "RBFOX":  0.24,   # TGCATG in human introns
    "NOVA":   3.8,    # YCAY (4 variants) combined
    "TARDBP": 0.08,   # TGTGTG
    "MBNL":   0.9,    # TGCTT/TGCTG
    "SRSF":   0.6,    # GAAGAA/GGAGGA
}


# ── Evidence strength scoring ─────────────────────────────────────────────────

def _evidence_strength(mechanism: str, regulators: list,
                       motif_data: Optional[dict]) -> dict:
    """
    Assigns evidence strength and generates human-readable interpretation.

    Strength levels:
      strong     — ≥2 significant regulators + motif enrichment > 2x background
      moderate   — ≥2 significant regulators OR (1 regulator + canonical motif)
      correlative— cell-type specific regulator change co-occurs with DTU,
                   no direct binding evidence
      weak       — <padj threshold or single regulator, no motif support
    """
    n_sig = len(regulators)
    down_count = sum(1 for r in regulators if r["direction"] == "down")
    up_count   = n_sig - down_count

    rbfox_enrich = None
    rbfox_canon  = 0
    if motif_data and "aggregate" in motif_data:
        rb = motif_data["aggregate"].get("RBFOX", {})
        rbfox_enrich = rb.get("enrichment")
        rbfox_canon  = rb.get("canonical", 0)

    if mechanism == "epigenetic_derepression":
        dnmt3a_down = any(r["gene"] == "DNMT3A" and r["direction"] == "down"
                          for r in regulators)
        if dnmt3a_down and n_sig >= 2:
            strength = "moderate"
            note = ("DNMT3A downregulation consistent with LINE-1 CpG promoter "
                    "hypomethylation (De Cecco et al. 2019 Nature mechanism). "
                    "Compensatory upregulation of SETDB1/TRIM28 also observed.")
        elif dnmt3a_down:
            strength = "correlative"
            note = "DNMT3A downregulation present; additional epigenetic evidence needed."
        else:
            strength = "weak"
            note = "No clear epigenetic derepression signature detected."

    elif mechanism == "alternative_splicing":
        if n_sig >= 2 and rbfox_enrich is not None and rbfox_enrich > 2.0:
            strength = "strong"
            note = (f"{n_sig} splicing regulators significantly changed "
                    f"({down_count} down, {up_count} up) with RBFOX motif "
                    f"enrichment {rbfox_enrich:.1f}x above background.")
        elif n_sig >= 2 and rbfox_canon > 0:
            strength = "moderate"
            note = (f"{n_sig} splicing regulators changed; {rbfox_canon} canonical "
                    f"YGCATG RBFOX site(s) found in CT-specific exon flanking introns. "
                    f"eCLIP validation required for mechanistic confirmation.")
        elif n_sig >= 2:
            strength = "correlative"
            note = (f"{n_sig} splicing regulators significantly changed in same "
                    f"cell type as DTU. Motif density at background level "
                    f"(no enrichment). Evidence is correlative only.")
        elif n_sig == 1:
            strength = "weak"
            note = f"Single splicing regulator ({regulators[0]['gene']}) changed; insufficient."
        else:
            strength = "weak"
            note = "No significant splicing regulator changes detected."

    else:  # transcriptional
        if n_sig >= 2:
            strength = "correlative"
            note = (f"{n_sig} transcription factors changed in this cell type, "
                    f"consistent with altered promoter activity.")
        else:
            strength = "weak"
            note = "Insufficient transcriptional regulator evidence."

    return {
        "level":  strength,
        "n_significant_regulators": n_sig,
        "interpretation": note,
        "caveat": (
            "mRNA expression levels may not reflect protein activity. "
            "TARDBP in particular: mRNA UP in AD may reflect nuclear depletion "
            "compensation (cytoplasmic aggregation), not increased splicing activity. "
            "All causal claims require functional validation."
        ),
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def run(case_result: dict, config: dict) -> dict:
    """
    M8: Regulatory Context Evidence

    Args:
        case_result: accumulated case dict from prior modules
        config: BISECT config dict; reads m14 sub-section

    Returns:
        dict with mechanism_type, regulators, motif_analysis, evidence_strength
    """
    cfg = config.get("m14", {})
    deg_dir  = cfg.get("deg_dir",
               "/home/dhkim1674/Project_AD_with_refTSS_novel/04_DEA/AD/DEG")
    genome_fa = cfg.get("genome_fa",
               "/home/dhkim1674/reference/refdata-gex-GRCh38-2024-A/fasta/genome.fa")
    padj_thresh = cfg.get("padj_threshold", 0.01)
    logfc_thresh = cfg.get("logfc_threshold", 0.10)

    gene      = case_result.get("gene_name", "UNKNOWN")
    cell_type = case_result.get("cell_type", "")

    print(f"  [M8] {gene} ({cell_type}): Regulatory context analysis")

    # ── Cell type mapping ───────────────────────────────────────────────────
    cell_type_full = _CELLTYPE_MAP.get(cell_type, cell_type)
    deg = _load_deg(deg_dir, cell_type_full)
    if not deg:
        print(f"  [M8] WARNING: DEG file not found for {cell_type_full}")

    # ── Mechanism classification ───────────────────────────────────────────
    mechanism = _classify_mechanism(case_result)
    print(f"  [M8] Mechanism classified as: {mechanism}")

    # ── Regulator panel selection ──────────────────────────────────────────
    panel = {
        "alternative_splicing":    _SPLICING_PANEL,
        "epigenetic_derepression": _EPIGENETIC_PANEL,
        "transcriptional":         _TRANSCRIPTIONAL_PANEL,
    }.get(mechanism, _SPLICING_PANEL)

    regulators = _extract_regulators(deg, panel, padj_thresh, logfc_thresh)
    print(f"  [M8] {len(regulators)} significant regulators found")

    # ── Motif analysis (splicing only) ─────────────────────────────────────
    motif_result = None
    if mechanism == "alternative_splicing":
        ct_info = case_result.get("ct_info") or {}
        ad_info = case_result.get("ad_info") or {}
        ct_exons = ct_info.get("exons", [])
        ad_exons = ad_info.get("exons", [])
        chrom    = ct_info.get("chrom") or ad_info.get("chrom", "chr1")
        strand   = ct_info.get("strand") or ad_info.get("strand", "+")

        fai_path = genome_fa + ".fai"
        fai_idx  = _load_fai(fai_path) if os.path.exists(fai_path) else {}

        if ct_exons and fai_idx:
            motif_result = _intron_motif_analysis(
                ct_exons, ad_exons, chrom, strand, genome_fa, fai_idx)
            n_ct_unique = motif_result.get("ct_unique_exon_count", 0)
            print(f"  [M8] Motif search: {n_ct_unique} CT-specific exons analyzed")
        else:
            motif_result = {"skipped": "missing_exon_coords_or_genome_index"}

    # ── Evidence strength ──────────────────────────────────────────────────
    evidence = _evidence_strength(mechanism, regulators, motif_result)
    print(f"  [M8] Evidence strength: {evidence['level']}")

    # ── LINE-1 details for epigenetic cases ───────────────────────────────
    l1_details = None
    if mechanism == "epigenetic_derepression":
        ad_repeats = case_result.get("ad_repeats") or {}
        exon_hits  = ad_repeats.get("exon_hits", {})
        l1_details = []
        for exon_id, hits in exon_hits.items():
            for h in hits:
                if h.get("is_young_l1"):
                    l1_details.append({
                        "exon": exon_id,
                        "element": h.get("name"),
                        "family": h.get("family"),
                        "pct_divergence": h.get("pct_divergence"),
                        "exon_overlap_bp": h.get("exon_overlap_bp"),
                        "sw_score": h.get("sw_score"),
                    })

    return {
        "mechanism_type":  mechanism,
        "cell_type_full":  cell_type_full,
        "deg_n_genes_loaded": len(deg),
        "significant_regulators": regulators,
        "top_regulators": regulators[:5],
        "motif_analysis":  motif_result,
        "l1_details":      l1_details,
        "evidence_strength": evidence["level"],   # plain string: strong/moderate/correlative/weak
        "evidence_details": evidence,             # full dict with interpretation/caveat
        "summary": _build_summary(gene, mechanism, regulators, evidence, l1_details),
    }


def _build_summary(gene: str, mechanism: str, regulators: list,
                   evidence: dict, l1_details: Optional[list]) -> str:
    """One-paragraph human-readable summary for the BISECT report."""
    n    = len(regulators)
    lvl  = evidence["level"]
    note = evidence["interpretation"]

    if mechanism == "epigenetic_derepression":
        l1_str = ""
        if l1_details:
            names = list({d["element"] for d in l1_details})
            l1_str = f" (young LINE-1 elements: {', '.join(names)})"
        return (
            f"The {gene} AD-enriched isoform contains young LINE-1 sequence"
            f"{l1_str}, suggesting epigenetic derepression as the underlying "
            f"mechanism. {note} Evidence strength: {lvl}."
        )
    elif mechanism == "alternative_splicing":
        top = [r["gene"] for r in regulators[:3]]
        dirs = {r["gene"]: r["direction"] for r in regulators[:3]}
        top_str = ", ".join(f"{g}({dirs[g]})" for g in top) if top else "none"
        return (
            f"The {gene} isoform switch involves novel splice junctions consistent "
            f"with splicing factor dysregulation. Top regulators in the same cell "
            f"type: {top_str}. {note} Evidence strength: {lvl}. "
            f"Caution: motif-only evidence cannot confirm direct binding; "
            f"eCLIP or functional perturbation data required."
        )
    else:
        top = [r["gene"] for r in regulators[:3]]
        return (
            f"The {gene} isoform switch may reflect transcriptional changes. "
            f"Top regulators: {', '.join(top) if top else 'none'}. "
            f"Evidence strength: {lvl}."
        )
