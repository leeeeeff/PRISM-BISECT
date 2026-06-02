"""
M6: Report Generation
Outputs JSON (structured data), domains.tsv, Figure (domain map),
and Markdown report (Jinja2 template if available, fallback to inline).
"""
import csv
import json
import os
import math
from pathlib import Path
from datetime import datetime


# ── JSON Output ──────────────────────────────────────────────────────────────

def save_json(case_result: dict, output_dir: str) -> str:
    """Save full analysis JSON for a case."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "analysis.json")
    with open(path, "w") as f:
        json.dump(case_result, f, indent=2, default=str)
    return path


# ── Domains TSV ──────────────────────────────────────────────────────────────

def save_domains_tsv(case_result: dict, output_dir: str) -> str:
    """Save CT + AD domain hits as domains.tsv (directly usable for paper Table 1)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "domains.tsv")
    fields = ["isoform", "role", "domain", "pfam_family", "ali_from", "ali_to",
              "hmm_from", "hmm_to", "evalue", "score"]
    ct_id = case_result.get("ct_transcript_id", "CT")
    ad_id = case_result.get("ad_transcript_id", "AD")
    dc = case_result.get("domain_change", {})
    lost_set = set(dc.get("domains_lost", []))
    gained_set = set(dc.get("domains_gained", []))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields + ["change_status"], delimiter="\t")
        writer.writeheader()
        for d in case_result.get("ct_domains", []):
            fam = d.get("pfam_family", d.get("domain", "?"))
            status = "LOST" if fam in lost_set else "shared"
            writer.writerow({
                "isoform": ct_id, "role": "CT",
                "domain": d.get("domain", "?"), "pfam_family": fam,
                "ali_from": d.get("ali_from", ""), "ali_to": d.get("ali_to", ""),
                "hmm_from": d.get("hmm_from", ""), "hmm_to": d.get("hmm_to", ""),
                "evalue": d.get("evalue", ""), "score": d.get("score", ""),
                "change_status": status,
            })
        for d in case_result.get("ad_domains", []):
            fam = d.get("pfam_family", d.get("domain", "?"))
            status = "GAINED" if fam in gained_set else "shared"
            writer.writerow({
                "isoform": ad_id, "role": "AD",
                "domain": d.get("domain", "?"), "pfam_family": fam,
                "ali_from": d.get("ali_from", ""), "ali_to": d.get("ali_to", ""),
                "hmm_from": d.get("hmm_from", ""), "hmm_to": d.get("hmm_to", ""),
                "evalue": d.get("evalue", ""), "score": d.get("score", ""),
                "change_status": status,
            })
    return path


# ── Figure Generation ─────────────────────────────────────────────────────────

def plot_domain_map(case_result: dict, output_dir: str, config: dict) -> str:
    """
    Generate domain map figure for CT and AD isoforms.
    Shows: protein length bar, domain annotations, repeat elements (if any).
    Returns path to saved figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyArrowPatch
    except ImportError:
        print("  [M6] matplotlib not available — skipping figure")
        return None

    gene = case_result.get("gene_name", "Unknown")
    ct_id = case_result.get("ct_transcript_id", "CT")
    ad_id = case_result.get("ad_transcript_id", "AD")
    ct_len = case_result.get("ct_info", {}).get("protein_length") or \
             case_result.get("ct_seq", {}).get("length", 100)
    ad_len = case_result.get("ad_info", {}).get("protein_length") or \
             case_result.get("ad_seq", {}).get("length", 100)

    ct_domains = case_result.get("ct_domains", [])
    ad_domains = case_result.get("ad_domains", [])
    ct_repeats = case_result.get("ct_repeats", {}).get("cds_overlap_hits", [])
    ad_repeats = case_result.get("ad_repeats", {}).get("cds_overlap_hits", [])
    domain_change = case_result.get("domain_change", {})

    fig_cfg = config.get("figure", {})
    domain_colors = fig_cfg.get("domain_colors", {})
    repeat_colors = fig_cfg.get("repeat_colors", {})

    # Layout
    max_len = max(ct_len, ad_len, 1)
    fig_w = min(max(fig_cfg.get("min_width", 8), max_len / 100 * fig_cfg.get("width_per_100aa", 1.0)),
                fig_cfg.get("max_width", 20))
    n_rows = 2  # CT row + AD row
    has_repeats = bool(ct_repeats or ad_repeats)
    fig_h = 5.5 + (1.5 if has_repeats else 0)   # taller to accommodate stacked labels

    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h),
                              gridspec_kw={"hspace": 1.0})

    # Three y-levels for narrow-domain labels; greedy assignment avoids horizontal overlap
    _LABEL_LEVELS = [2.1, 2.55, 3.0]
    _MIN_LABEL_SEP = max_len * 0.09   # min x-distance between labels on the same level

    def draw_isoform_row(ax, label, length, domains, repeats, is_ct):
        ax.set_xlim(0, max_len)
        ax.set_ylim(-0.5, 3.4)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Protein backbone
        color = "#2196F3" if is_ct else "#F44336"
        ax.barh(1.0, length, height=0.3, left=0, color=color, alpha=0.25, zorder=1)
        ax.text(-max_len * 0.01, 1.0, label, ha="right", va="center", fontsize=9, fontweight="bold")
        ax.text(length + max_len * 0.01, 1.0, f"{length} aa", ha="left", va="center", fontsize=7, color="#555")

        # ── Domain rectangles + two-pass label placement ────────────────────
        # Pass 1: draw all boxes, collect wide vs narrow domains
        wide_labels  = []   # (cx, col, name) — wide enough for inside label
        narrow_queue = []   # (cx, col, name) — need leader-line label

        used_colors = {}
        for dom in domains:
            d_name = dom.get("pfam_family", dom.get("domain", "?"))
            d_name_short = d_name.split("_")[0]
            dfrom = dom.get("ali_from", 0)
            dto   = dom.get("ali_to", 0)
            dlen  = dto - dfrom
            col   = domain_colors.get(d_name, domain_colors.get(
                        d_name_short, domain_colors.get("default", "#BDC3C7")))
            rect  = mpatches.FancyBboxPatch(
                (dfrom, 0.75), dlen, 0.5,
                boxstyle="round,pad=0.01", facecolor=col,
                edgecolor="white", linewidth=0.8, zorder=3,
            )
            ax.add_patch(rect)
            cx = dfrom + dlen / 2
            if dlen > max_len * 0.04:
                wide_labels.append((cx, col, d_name_short))
            else:
                narrow_queue.append((cx, col, d_name_short))
            used_colors[d_name_short] = col

        # Pass 2a: wide domain labels (inside box, white)
        for cx, col, name in wide_labels:
            ax.text(cx, 1.0, name, ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold", zorder=4)

        # Pass 2b: narrow domain labels — greedy 3-level stacking + dashed leader line
        narrow_queue.sort(key=lambda x: x[0])   # sort by x position
        level_last_x = {lvl: -_MIN_LABEL_SEP * 2 for lvl in _LABEL_LEVELS}

        for cx, col, name in narrow_queue:
            # Choose the lowest level where there is enough horizontal clearance
            chosen = _LABEL_LEVELS[0]
            for lvl in _LABEL_LEVELS:
                if cx - level_last_x[lvl] >= _MIN_LABEL_SEP:
                    chosen = lvl
                    break
            level_last_x[chosen] = cx

            # Dashed leader line from domain-top centre to just below the label
            ax.plot([cx, cx], [1.16, chosen - 0.08],
                    color=col, linestyle="--", linewidth=0.75,
                    dash_capstyle="round", zorder=4)
            # Small dot at domain end
            ax.plot(cx, 1.16, "o", color=col, markersize=2.5, zorder=5)
            # Label text
            ax.text(cx, chosen, name, ha="center", va="bottom",
                    fontsize=6, color=col, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor=col, linewidth=0.5, alpha=0.92))

        # Repeat element overlays (below protein bar)
        for rep in repeats:
            rs = rep.get("start", 0)
            re_ = rep.get("end", 0)
            rc = rep.get("class", "")
            rstrand = rep.get("strand", "+")
            col = repeat_colors.get(rc, "#95A5A6")
            rlen = re_ - rs
            # Map genomic to protein coords if possible (approximate for display)
            cds_start = case_result.get("ad_info" if not is_ct else "ct_info", {}).get("cds_start_genomic") or 0
            if cds_start:
                prot_s = (rs - cds_start) // 3
                prot_e = (re_ - cds_start) // 3
                if 0 <= prot_s < length and prot_e > 0:
                    rect = mpatches.FancyBboxPatch(
                        (max(0, prot_s), 0.1), min(prot_e, length) - max(0, prot_s), 0.25,
                        boxstyle="round,pad=0.01", facecolor=col, edgecolor="none", alpha=0.5, zorder=2
                    )
                    ax.add_patch(rect)
                    ax.text((max(0, prot_s) + min(prot_e, length)) / 2, 0.0,
                            f"{rep['name']}({rstrand})", ha="center", va="top",
                            fontsize=5, color=col, zorder=5)

        ax.set_xlabel(f"Amino acid position (0–{max_len})", fontsize=7)

    draw_isoform_row(axes[0], f"CT: {ct_id[:30]}", ct_len, ct_domains, ct_repeats, is_ct=True)
    draw_isoform_row(axes[1], f"AD: {ad_id[:30]}", ad_len, ad_domains, ad_repeats, is_ct=False)

    # Title with domain change summary
    lost = domain_change.get("domains_lost", [])
    gained = domain_change.get("domains_gained", [])
    change_str = ""
    if lost:
        change_str += f"Lost: {', '.join(lost)}  "
    if gained:
        change_str += f"Gained: {', '.join(gained)}"
    title = f"{gene} — AD Isoform Switch\n{change_str or 'No domain change detected'}"
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)

    # Legend
    legend_patches = []
    all_domains_in_fig = {d.get("pfam_family", "?") for d in ct_domains + ad_domains}
    for dname in all_domains_in_fig:
        short = dname.split("_")[0]
        col = domain_colors.get(dname, domain_colors.get(short, domain_colors.get("default", "#BDC3C7")))
        legend_patches.append(mpatches.Patch(color=col, label=short))
    if legend_patches:
        fig.legend(handles=legend_patches, loc="upper right", fontsize=7,
                   ncol=min(len(legend_patches), 6), bbox_to_anchor=(1.0, 1.0))

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "domain_map.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    fig.savefig(fig_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fig_path


# ── Markdown Report ───────────────────────────────────────────────────────────

def _build_jinja_context(case_result: dict) -> dict:
    """Build Jinja2 template context from case_result."""
    ct_seq_len = case_result.get("ct_seq", {}).get("length", 0) or 0
    ad_seq_len = case_result.get("ad_seq", {}).get("length", 0) or 0
    return {
        "meta": {
            "gene": case_result.get("gene_name", "Unknown"),
            "cell_type": case_result.get("cell_type", "?"),
            "ct_transcript_id": case_result.get("ct_transcript_id", "?"),
            "ad_transcript_id": case_result.get("ad_transcript_id", "?"),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "pipeline_version": "1.0",
        },
        "stats": {
            "diffuse_delta": case_result.get("diffuse_delta", 0),
            "dtu_p": case_result.get("dtu_pvalue", 1),
            "direction": case_result.get("direction", ""),
            "priority": case_result.get("priority", ""),
        },
        "sequences": {
            "ct_length": ct_seq_len,
            "ad_length": ad_seq_len,
            "length_diff": ad_seq_len - ct_seq_len,
        },
        "domain_analysis": {
            "ct_domains": case_result.get("ct_domains", []),
            "ad_domains": case_result.get("ad_domains", []),
            "domain_change": case_result.get("domain_change", {}),
        },
        "motifs": {
            "ct_mts": case_result.get("ct_motifs", {}).get("mts"),
            "ad_mts": case_result.get("ad_motifs", {}).get("mts"),
            "ct_functional": case_result.get("ct_motifs", {}).get("functional_motifs", {}),
            "ad_functional": case_result.get("ad_motifs", {}).get("functional_motifs", {}),
        },
        "genomic": {
            "ct": case_result.get("ct_info", {}),
            "ad": case_result.get("ad_info", {}),
        },
        "repeat_elements": {
            "ct": case_result.get("ct_repeats", {}),
            "ad": case_result.get("ad_repeats", {}),
        },
        "seq_validation": case_result.get("seq_validation", {}),
        "nmd_screen": case_result.get("nmd_screen", {}),
        "alphafold": case_result.get("m11_alphafold", {}),
        "ppi": case_result.get("m12_ppi", {}),
        "conservation": case_result.get("m13_conservation", {}),
        "regulatory": case_result.get("m8_regulatory_context", {}),
        "promoter": case_result.get("m9_promoter_usage", {}),
        "apa": case_result.get("m10_apa", {}),
    }


def generate_markdown(case_result: dict, output_dir: str,
                      template_dir: str = None) -> str:
    """
    Generate Markdown report. Uses Jinja2 template if available,
    falls back to inline generation.
    """
    md_path = os.path.join(output_dir, "report.md")
    os.makedirs(output_dir, exist_ok=True)

    # Try Jinja2 template
    if template_dir is None:
        template_dir = str(Path(__file__).parent.parent / "templates")
    template_file = os.path.join(template_dir, "case_report.md.j2")

    if os.path.exists(template_file):
        try:
            from jinja2 import Environment, FileSystemLoader
            env = Environment(loader=FileSystemLoader(template_dir),
                              trim_blocks=True, lstrip_blocks=True)
            tmpl = env.get_template("case_report.md.j2")
            ctx = _build_jinja_context(case_result)
            content = tmpl.render(**ctx)
            with open(md_path, "w") as f:
                f.write(content)
            return md_path
        except Exception as e:
            print(f"  [M6] Jinja2 render failed ({e}), falling back to inline")

    # ── Inline fallback ────────────────────────────────────────────────────────
    return _generate_markdown_inline(case_result, output_dir, md_path)


def _generate_markdown_inline(case_result: dict, output_dir: str,
                               md_path: str) -> str:
    """Inline Markdown generation (no Jinja2 dependency)."""
    gene = case_result.get("gene_name", "Unknown")
    ct_id = case_result.get("ct_transcript_id", "?")
    ad_id = case_result.get("ad_transcript_id", "?")
    delta = case_result.get("diffuse_delta", 0)
    dtu_p = case_result.get("dtu_pvalue", 1)
    cell_type = case_result.get("cell_type", "?")
    domain_change = case_result.get("domain_change", {})

    lines = [
        f"# {gene} — Biological Analysis Report",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')} | BISECT v1.1",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Gene | {gene} |",
        f"| Cell type | {cell_type} |",
        f"| CT isoform | {ct_id} |",
        f"| AD isoform | {ad_id} |",
        f"| DIFFUSE Δ | {delta:+.3f} |",
        f"| DTU p-value | {dtu_p:.2e} |",
        f"| Domain change | {'YES' if domain_change.get('has_domain_change') else 'NO'} |",
        "",
    ]

    # Domain change
    if domain_change.get("has_domain_change"):
        lines += [
            "## Domain Architecture Change",
            "",
            f"**Domains lost in AD**: {', '.join(domain_change.get('domains_lost', [])) or 'None'}",
            f"**Domains gained in AD**: {', '.join(domain_change.get('domains_gained', [])) or 'None'}",
            f"**Domains shared**: {', '.join(domain_change.get('domains_shared', [])) or 'None'}",
            "",
        ]

    # CT isoform domains
    ct_domains = case_result.get("ct_domains", [])
    if ct_domains:
        lines += ["## CT Isoform Pfam Domains", "", "| Domain | Position | E-value | Score |",
                  "|--------|----------|---------|-------|"]
        for d in ct_domains:
            lines.append(f"| {d['domain']} | aa {d['ali_from']}–{d['ali_to']} | {d['evalue']:.1e} | {d['score']:.1f} |")
        lines.append("")

    # AD isoform domains
    ad_domains = case_result.get("ad_domains", [])
    if ad_domains:
        lines += ["## AD Isoform Pfam Domains", "", "| Domain | Position | E-value | Score |",
                  "|--------|----------|---------|-------|"]
        for d in ad_domains:
            lines.append(f"| {d['domain']} | aa {d['ali_from']}–{d['ali_to']} | {d['evalue']:.1e} | {d['score']:.1f} |")
        lines.append("")

    # MTS analysis
    ad_mts = case_result.get("ad_motifs", {}).get("mts", {})
    ct_mts = case_result.get("ct_motifs", {}).get("mts", {})
    if ad_mts or ct_mts:
        lines += ["## MTS Feature Analysis", "", "| Feature | CT isoform | AD isoform |",
                  "|---------|------------|------------|"]
        for feat in ["net_charge_30aa", "de_count_40aa", "hydrophobic_moment", "composite_score"]:
            ct_v = ct_mts.get(feat, "N/A")
            ad_v = ad_mts.get(feat, "N/A")
            lines.append(f"| {feat} | {ct_v} | {ad_v} |")
        lines.append(f"| HHH motif | {ct_mts.get('hhh_motif') or 'ABSENT'} | {ad_mts.get('hhh_motif') or 'ABSENT'} |")
        lines.append(f"| Prediction | {ct_mts.get('prediction', 'N/A')} | {ad_mts.get('prediction', 'N/A')} |")
        lines.append("")

    # Genomic info
    ad_info = case_result.get("ad_info", {})
    if ad_info.get("is_nat"):
        lines += [
            "## Natural Antisense Transcript (NAT) Detection",
            "",
            f"**AD isoform is a NAT**: {ad_info['nat_relationship']['description']}",
            f"**Genomic span**: {ad_info.get('chrom')}:{ad_info.get('genomic_span_start'):,}–{ad_info.get('genomic_span_end'):,} ({ad_info.get('genomic_span_kb', 0):.1f} kb)",
            f"**CDS start**: {ad_info.get('chrom')}:{ad_info.get('cds_start_genomic', '?'):,}",
            "",
        ]

    # RepeatMasker
    ad_repeats = case_result.get("ad_repeats", {})
    if ad_repeats.get("has_l1_in_cds"):
        lines += ["## LINE-1 Element in CDS", ""]
        for hit in ad_repeats.get("cds_young_l1_hits", []):
            lines.append(f"- **{hit['name']}** ({hit['strand']} strand, {hit['pct_divergence']}% div, SW={hit['sw_score']}): "
                        f"overlaps CDS by {hit.get('cds_overlap_bp', '?')} bp")
        lines.append("")

    # NMD screening (M9)
    nmd = case_result.get("nmd_screen", {})
    if nmd and nmd.get("summary"):
        lines += ["## NMD Susceptibility (M9)", ""]
        lines.append(f"**{nmd['summary']}**")
        lines.append("")
        ct_nmd = nmd.get("ct", {})
        ad_nmd = nmd.get("ad", {})
        if ct_nmd or ad_nmd:
            lines += ["| Isoform | NMD susceptible | Dist to last EEJ | Source |",
                      "|---------|----------------|-----------------|--------|"]
            if ct_nmd:
                lines.append(f"| CT | {ct_nmd.get('nmd_susceptible','?')} | {ct_nmd.get('ptc_to_last_junction_nt','—')} nt | {ct_nmd.get('source','—')} |")
            if ad_nmd:
                lines.append(f"| AD | {ad_nmd.get('nmd_susceptible','?')} | {ad_nmd.get('ptc_to_last_junction_nt','—')} nt | {ad_nmd.get('source','—')} |")
        lines.append("")

    # Genomic sequence validation (M8)
    sv = case_result.get("seq_validation", {})
    if sv and not sv.get("skipped"):
        lines += ["## Genomic Sequence Validation (M8)", ""]
        lines.append(f"**Conclusion**: {sv.get('conclusion', '—')}")
        lines.append("")
        for elem in sv.get("elements", []):
            if elem.get("error"):
                lines.append(f"- **{elem['element']}**: {elem['error']}")
            else:
                bf = elem.get("best_6frame", {})
                lines.append(
                    f"- **{elem['element']}** exon {elem['exon']} | frame {bf.get('frame_id','?')} | "
                    f"SW={bf.get('score',0):.0f} | identity={bf.get('pct_identity',0):.1f}% | "
                    f"coverage={bf.get('coverage',0):.1f}%"
                )
        lines.append("")

    # ── M10: AlphaFold structural confidence ─────────────────────────────────
    af = case_result.get("m11_alphafold", {})
    if af and not af.get("error"):
        ct_af = af.get("ct", {})
        ad_af = af.get("ad", {})
        comp = af.get("comparison", {})
        lines += ["## AlphaFold Structural Confidence (M10)", ""]
        lines.append(f"| | CT isoform | AD isoform |")
        lines.append(f"|--|-----------|-----------|")
        lines.append(f"| Source | {ct_af.get('source', '—')} | {ad_af.get('source', '—')} |")
        lines.append(f"| UniProt | {ct_af.get('uniprot_id', '—')} | {ad_af.get('uniprot_id', '—')} |")
        lines.append(f"| Mean pLDDT | {ct_af.get('plddt_mean', '—')} | {ad_af.get('plddt_mean', '—')} |")
        pct_ct = f"{ct_af.get('plddt_high_fraction', 0)*100:.0f}%" if ct_af.get('plddt_high_fraction') is not None else "—"
        pct_ad = f"{ad_af.get('plddt_high_fraction', 0)*100:.0f}%" if ad_af.get('plddt_high_fraction') is not None else "—"
        lines.append(f"| Fraction >70 pLDDT | {pct_ct} | {pct_ad} |")
        lines.append("")

        ct_dp = ct_af.get("domain_plddt", {})
        ad_dp = ad_af.get("domain_plddt", {})
        all_doms = sorted(set(list(ct_dp.keys()) + list(ad_dp.keys())))
        if all_doms:
            lines += ["**Per-domain pLDDT:**", "",
                      "| Domain | CT pLDDT | CT conf | AD pLDDT | AD conf |",
                      "|--------|----------|---------|----------|---------|"]
            for dom in all_doms:
                ct_d = ct_dp.get(dom, {})
                ad_d = ad_dp.get(dom, {})
                lines.append(
                    f"| {dom} | {ct_d.get('mean', '—')} | {ct_d.get('confidence', '—')} "
                    f"| {ad_d.get('mean', '—')} | {ad_d.get('confidence', '—')} |"
                )
            lines.append("")

        interp = comp.get("interpretation", "")
        if interp:
            lines += [f"> {interp}", ""]

    # ── M11: PPI network validation ───────────────────────────────────────────
    ppi = case_result.get("m12_ppi", {})
    if ppi and not ppi.get("error"):
        verdict = ppi.get("summary_verdict", "—")
        verdict_icon = {"SUPPORTED": "✅", "PARTIAL": "⚠️", "UNSUPPORTED": "❌"}.get(verdict, "")
        lines += [f"## PPI Network Validation (M11) {verdict_icon}", ""]
        lines.append(f"**Verdict**: {verdict}")
        lines.append(f"**Interpretation**: {ppi.get('interpretation', '—')}")
        lines.append("")

        hyp = ppi.get("hypothesis_support", {})
        if hyp:
            lines += ["**Hypothesis partner support:**", "",
                      "| Partner | Score | Confidence | Evidence |",
                      "|---------|-------|-----------|---------|"]
            for partner, info in hyp.items():
                ev = ", ".join(info.get("evidence_types", [])) or "none"
                lines.append(
                    f"| {partner} | {info.get('combined_score', 0)} "
                    f"| {info.get('confidence', '—')} | {ev} |"
                )
            lines.append("")

        top = ppi.get("string_hits", [])[:5]
        if top:
            lines += ["**Top STRING partners (all):**", "",
                      "| Partner | Score | Exp score | Evidence |",
                      "|---------|-------|-----------|---------|"]
            for h in top:
                ev = ", ".join(h.get("evidence_types", []))
                lines.append(
                    f"| {h['partner']} | {h.get('combined_score', 0)} "
                    f"| {h.get('experimental_score', 0)} | {ev} |"
                )
            lines.append("")

        dl = ppi.get("domain_interaction_link", {}).get("domain_links", [])
        if dl:
            lines.append("**Domain–interaction links:**")
            for link in dl:
                lines.append(
                    f"- **{link['domain']}** ({link['change']}): {link['interaction_role']}"
                )
            lines.append("")

    # ── M12: Evolutionary conservation ────────────────────────────────────────
    cons = case_result.get("m13_conservation", {})
    if cons and not cons.get("error"):
        summ = cons.get("summary", {})
        bg = cons.get("background", {})
        lines += ["## Evolutionary Conservation (M12)", ""]
        lines.append(f"**Track**: {cons.get('track', 'phyloP100way')} | "
                     f"**Genome**: {cons.get('genome', 'hg38')}")
        lines.append("")

        lines += ["| Region | N exons | Mean phyloP | Conservation class |",
                  "|--------|---------|-------------|-------------------|"]
        ad_exons = cons.get("ad_specific_exons", [])
        ct_exons_c = cons.get("ct_specific_exons", [])
        ad_mean = summ.get("ad_specific_mean_phyloP")
        ct_mean = summ.get("ct_specific_mean_phyloP")
        ad_cls = "highly_conserved" if ad_mean and ad_mean >= 1.5 else ("conserved" if ad_mean and ad_mean >= 0.5 else "low")
        ct_cls = "highly_conserved" if ct_mean and ct_mean >= 1.5 else ("conserved" if ct_mean and ct_mean >= 0.5 else "low")
        bg_mean = bg.get("intronic_phyloP_mean")
        lines.append(f"| AD-specific exons | {len(ad_exons)} | {ad_mean or '—'} | {ad_cls} |")
        lines.append(f"| CT-specific exons | {len(ct_exons_c)} | {ct_mean or '—'} | {ct_cls} |")
        lines.append(f"| Intronic background | {bg.get('intronic_sample_n', '—')} windows | {bg_mean or '—'} | — |")
        lines.append("")

        fold = summ.get("fold_difference")
        if fold:
            lines.append(f"**AD/CT fold difference**: {fold}×")
        lines.append(f"**Interpretation**: {summ.get('interpretation', '—')}")
        lines.append("")

        top_ad = [e for e in ad_exons if e.get("phyloP_mean") is not None][:3]
        if top_ad:
            lines += ["**Top conserved AD-specific exons:**", "",
                      "| Exon | Coords | phyloP mean | Conservation | Domain |",
                      "|------|--------|-------------|-------------|--------|"]
            for e in top_ad:
                lines.append(
                    f"| {e['rank']} | {e['chrom']}:{e['start']:,}–{e['end']:,} "
                    f"| {e.get('phyloP_mean', '—')} | {e.get('conservation_class', '—')} "
                    f"| {e.get('domain_overlap') or '—'} |"
                )
            lines.append("")

    # ── M14: Regulatory Context Evidence ─────────────────────────────────────
    reg = case_result.get("m8_regulatory_context", {})
    if reg and not reg.get("error"):
        mechanism = reg.get("mechanism_type", "—")
        evidence = reg.get("evidence_strength", "—")
        # evidence_strength is a plain string; evidence_details has interpretation
        if isinstance(evidence, dict):
            evidence = evidence.get("level", "—")
        evidence_icon = {"strong": "🔴", "moderate": "🟠", "correlative": "🟡", "weak": "⚪"}.get(evidence, "")
        lines += [f"## Regulatory Context Evidence (M14) {evidence_icon}", ""]
        lines.append(f"**Mechanism type**: {mechanism}")
        lines.append(f"**Evidence strength**: {evidence}")
        ev_details = reg.get("evidence_details", {})
        if ev_details and ev_details.get("interpretation"):
            lines.append(f"**Interpretation**: {ev_details['interpretation']}")
        lines.append("")

        # Significant regulators
        regs = reg.get("significant_regulators", [])
        if regs:
            lines += ["**Significant regulators (padj < 0.01):**", "",
                      "| Gene | logFC | padj | Direction |",
                      "|------|-------|------|-----------|"]
            for r in regs[:10]:
                lines.append(
                    f"| {r['gene']} | {r['logFC']:+.3f} | {r['padj']:.1e} "
                    f"| {r['direction']} |"
                )
            if len(regs) > 10:
                lines.append(f"| *...and {len(regs)-10} more* | | | |")
            lines.append("")

        # Motif analysis (aggregate)
        motif = reg.get("motif_analysis", {})
        if motif and not motif.get("skipped"):
            agg = motif.get("aggregate", {})
            if agg:
                lines += ["**RBP binding motif enrichment (CT-specific exon flanking introns):**", "",
                          "| RBP | Sites | Density/kb | Enrichment vs background |",
                          "|-----|-------|-----------|--------------------------|"]
                for rbp, info in agg.items():
                    if isinstance(info, dict):
                        fold = info.get("enrichment")
                        fold_str = f"{fold:.2f}×" if fold is not None else "—"
                        dens = info.get("density_per_kb", "—")
                        lines.append(f"| {rbp} | {info['count']} | {dens} | {fold_str} |")
                lines.append(f"*Total intron sequence searched: "
                             f"{motif.get('total_intron_nt_searched', 0):,} nt "
                             f"across {motif.get('ct_unique_exon_count', 0)} CT-specific exons*")
                lines.append("")

        # L1 details for epigenetic cases
        l1 = reg.get("l1_details")
        if l1:
            lines += ["**Young LINE-1 elements in AD isoform:**", ""]
            for h in l1:
                lines.append(
                    f"- **{h['element']}** ({h['family']}, {h['pct_divergence']}% div, "
                    f"SW={h['sw_score']}): {h['exon_overlap_bp']} bp exon overlap"
                )
            lines.append("")

        # Summary paragraph
        summary = reg.get("summary", "")
        if summary:
            lines += ["**Summary:**", ""]
            lines.append(f"> {summary}")
            lines.append("")

    # Figure reference
    lines += [
        "## Figure",
        "",
        "![Domain Map](domain_map.png)",
        "",
        "---",
        "",
        "*Generated by BISECT v1.1*",
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    return md_path
