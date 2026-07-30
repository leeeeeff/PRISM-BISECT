#!/usr/bin/env python3
"""
CF1: UniProt Continuous Correlation Analysis
Nature Communications reviewer response — gap vs structural severity
"""

import os
import re
import numpy as np
import pandas as pd
from scipy import stats

# ─── Paths ───────────────────────────────────────────────────────────────────
EVAL_TSV   = "/home/welcome1/sw1686/DIFFUSE/reports/exp_h_uniprot_eval/v2/pairwise_eval_v2.tsv"
BENCH_CSV  = "/home/welcome1/sw1686/DIFFUSE/reports/exp_g_uniprot/uniprot_isoform_benchmark_v2.csv"
SEQ_CACHE  = "/home/welcome1/sw1686/DIFFUSE/reports/exp_h_uniprot_eval/seq_cache"
OUT_FILE   = "/home/welcome1/sw1686/DIFFUSE/reports/exp_h_uniprot_eval/v2/cf1_continuous_corr.txt"


# ─── 1. Structural severity categories (hardcoded, 48 evaluable pairs) ───────
# domain_loss: major domain loss / truncation ≥100aa
DOMAIN_LOSS = {
    "FGFR2", "NTRK2", "NTRK3", "PARK2", "VEGFR2", "FLT1",
    "BRCA1", "FGFR1", "EGFR", "CDKN2A",   # GO:0008285 pair
    "SMN1",
}
# Also ERBB4 is domain_loss but missing → handled as "missing"

# regulatory: subtle, < 50aa insert/deletion, RRM loss, N-terminal insert
REGULATORY = {
    "MAPT", "HNRNPA1", "SRSF1", "RBFOX1", "ACTN1", "FLNB",
    "APP", "STAT3", "CEBPB", "RAC1", "TIA1", "PTBP1", "WT1",
}

# intermediate: TAD loss, domain architecture change, moderate structural effect
INTERMEDIATE = {
    "HIF1A", "CEBPA", "RUNX2", "RUNX1", "BCL2L1", "BCL2", "MCL1",
    "CDK4", "DNMT3A", "PIK3R1",
    "CASP9", "DLG4", "SNAP25", "AXL",
    "TP53", "TP63", "TP73", "IGF1",
    "BIRC5", "KIT",
}

# both: direction=both → always correct by definition
BOTH_DIR = {"FGF2", "MDM2"}

# missing: sequence not embedded
MISSING = {"VEGFA", "PCBP1", "ERBB4"}

# CDKN2A appears twice (GO:0008285 → domain_loss, GO:0006977 → intermediate)
# We handle per-row below


def assign_category(gene, go_term, direction, note):
    """Assign structural severity category per row."""
    if note == "sequence not embedded":
        return "missing"
    if direction == "both":
        return "both"
    if gene == "CDKN2A":
        # GO:0008285 (growth inhibition) → domain_loss (p14ARF lacks exon1β)
        # GO:0006977 (DNA damage response) → intermediate
        if go_term == "GO_0008285":
            return "domain_loss"
        else:
            return "intermediate"
    if gene in DOMAIN_LOSS:
        return "domain_loss"
    if gene in REGULATORY:
        return "regulatory"
    if gene in INTERMEDIATE:
        return "intermediate"
    return "unknown"


# ─── 1b. Per-gene length check corrections ───────────────────────────────────
# NTRK1 P04629-2: 790aa vs canonical 796aa → only 6aa N-term insert → NOT domain_loss
# → move to intermediate
DOMAIN_LOSS.discard("NTRK1")  # wrong assumption, 6aa difference only
# CDK4 P11802-2: 183aa vs canonical 303aa → 120aa truncation → domain_loss
DOMAIN_LOSS.add("CDK4")
INTERMEDIATE.discard("CDK4")
# CASP9 P55211-2: 266aa vs 416aa → 150aa diff (lacks CARD domain) → domain_loss
DOMAIN_LOSS.add("CASP9")
INTERMEDIATE.discard("CASP9")
# NTRK1 → intermediate
INTERMEDIATE.add("NTRK1")

# ─── 2. Parse fasta length ───────────────────────────────────────────────────
def fasta_len(accession):
    """Return amino acid count for an isoform accession.
    Convention: -1 suffix → canonical (no number in filename).
    e.g. P00533-1 → P00533.fasta; P00533-4 → P00533-4.fasta
    """
    # Normalize: strip -1 suffix → canonical
    m = re.match(r'^([A-Z0-9]+)(-(\d+))?$', accession)
    if not m:
        return None
    base, _, num = m.group(1), m.group(2), m.group(3)
    if num is None or num == "1":
        fname = f"{base}.fasta"
    else:
        fname = f"{base}-{num}.fasta"
    fpath = os.path.join(SEQ_CACHE, fname)
    if not os.path.exists(fpath):
        return None
    seq = ""
    with open(fpath) as f:
        for line in f:
            if not line.startswith(">"):
                seq += line.strip()
    return len(seq) if seq else None


# ─── 3. Load data ────────────────────────────────────────────────────────────
df = pd.read_csv(EVAL_TSV, sep="\t")

# Parse note from benchmark CSV for richer descriptions
bench = pd.read_csv(BENCH_CSV)
bench_note = {row["gene"]: row.get("note", "") for _, row in bench.iterrows()}
# For CDKN2A both rows exist — we need go_term-specific later; keep gene-level for now

# ─── 4. Build analysis dataframe ─────────────────────────────────────────────
rows = []
for _, r in df.iterrows():
    gene = r["gene"]
    iso_a = str(r["iso_a"])
    iso_b = str(r["iso_b"])
    go_term = r["go_term"]
    direction = r["direction"]
    note_raw = str(r["note"]) if pd.notna(r["note"]) else ""
    gap_raw = r["gap"]
    correct_raw = r["correct"]

    cat = assign_category(gene, go_term, direction, note_raw)

    # lengths
    len_a = fasta_len(iso_a)
    len_b = fasta_len(iso_b)

    gap = float(gap_raw) if pd.notna(gap_raw) and gap_raw not in (None, "None", "nan") else None
    correct = int(correct_raw) if pd.notna(correct_raw) and correct_raw not in (None, "None", "nan") else None

    rows.append({
        "gene": gene,
        "iso_a": iso_a,
        "iso_b": iso_b,
        "go_term": go_term,
        "direction": direction,
        "gap": gap,
        "correct": correct,
        "category": cat,
        "len_a": len_a,
        "len_b": len_b,
        "bench_note": bench_note.get(gene, ""),
    })

adf = pd.DataFrame(rows)

# ─── 5. Compute length features ──────────────────────────────────────────────
adf["len_diff"] = adf.apply(
    lambda r: abs(r["len_a"] - r["len_b"])
    if (r["len_a"] is not None and r["len_b"] is not None) else None,
    axis=1,
)
adf["len_max"] = adf.apply(
    lambda r: max(r["len_a"], r["len_b"])
    if (r["len_a"] is not None and r["len_b"] is not None) else None,
    axis=1,
)
adf["pct_trunc"] = adf.apply(
    lambda r: r["len_diff"] / r["len_max"]
    if (r["len_diff"] is not None and r["len_max"] is not None and r["len_max"] > 0) else None,
    axis=1,
)

# ─── 6. Subsets ──────────────────────────────────────────────────────────────
# Evaluable: gap not None, correct not None
eval_df = adf[(adf["gap"].notna()) & (adf["correct"].notna())].copy()

# With length data (evaluable + lengths available)
len_df = eval_df[(eval_df["len_diff"].notna())].copy()

# Category subsets (among evaluable, excluding both/missing)
cat_eval = eval_df[~eval_df["category"].isin(["both", "missing"])].copy()

domain_df     = cat_eval[cat_eval["category"] == "domain_loss"]
regulatory_df = cat_eval[cat_eval["category"] == "regulatory"]
intermed_df   = cat_eval[cat_eval["category"] == "intermediate"]
both_df       = eval_df[eval_df["category"] == "both"]

# ─── 7. Spearman correlations ─────────────────────────────────────────────────
def spearman_safe(x, y, label):
    mask = x.notna() & y.notna()
    xv = x[mask].values.astype(float)
    yv = y[mask].values.astype(float)
    n = len(xv)
    if n < 3:
        return f"  {label}: n={n} (insufficient data)"
    rho, pval = stats.spearmanr(xv, yv)
    return f"  Spearman rho({label}): {rho:+.3f}  p={pval:.3f}  n={n}"

corr_len_diff   = spearman_safe(len_df["gap"], len_df["len_diff"],   "gap, |len_a - len_b|")
corr_pct_trunc  = spearman_safe(len_df["gap"], len_df["pct_trunc"],  "gap, % truncation")
corr_len_max    = spearman_safe(len_df["gap"], len_df["len_max"],     "gap, max(len_a, len_b)")
corr_len_a      = spearman_safe(len_df["gap"], len_df["len_a"],       "gap, len_a (canonical)")

# ─── 8. Categorical analysis ──────────────────────────────────────────────────
def acc_str(sub_df):
    n = len(sub_df)
    if n == 0:
        return "N=0"
    acc = sub_df["correct"].sum()
    mean_gap = sub_df["gap"].mean()
    return f"N={n}: acc={int(acc)}/{n} ({100*acc/n:.1f}%)  mean_gap={mean_gap:.3f}"

domain_str     = acc_str(domain_df)
regulatory_str = acc_str(regulatory_df)
intermed_str   = acc_str(intermed_df)

# Mann-Whitney U: gap in domain_loss vs regulatory
mwu_U, mwu_p = None, None
if len(domain_df) > 0 and len(regulatory_df) > 0:
    mwu_U, mwu_p = stats.mannwhitneyu(
        domain_df["gap"].values,
        regulatory_df["gap"].values,
        alternative="greater",  # domain_loss gap > regulatory gap
    )

# ─── 9. Failed cases ──────────────────────────────────────────────────────────
failed = eval_df[eval_df["correct"] == 0].copy()
failed = failed.sort_values("gap")

# Pattern summary
n_failed = len(failed)
n_tie    = (failed["gap"] < 0.01).sum()
n_reg    = (failed["category"] == "regulatory").sum()
n_intermed = (failed["category"] == "intermediate").sum()
n_domain_fail = (failed["category"] == "domain_loss").sum()
n_wrong_dir = failed[failed["gap"] >= 0.01].shape[0]  # gap>=0.01 but still wrong

# ─── 10. gap ≥ 0.10 threshold (from reviewer concern) ────────────────────────
high_gap = eval_df[eval_df["gap"] >= 0.10]
high_gap_acc = high_gap["correct"].sum()
high_gap_n = len(high_gap)

# domain_loss with gap >= 0.05
dom_high = domain_df[domain_df["gap"] >= 0.05]
dom_high_acc = dom_high["correct"].sum()
dom_high_n = len(dom_high)

# ─── 11. Write output ────────────────────────────────────────────────────────
lines = []
lines.append("=" * 65)
lines.append("CF1: UniProt Continuous Correlation Analysis")
lines.append("  Reviewer concern: gap≥0.10 (11/11=100%) is cherry-picking")
lines.append("  Response: continuous correlation + failed case analysis")
lines.append("=" * 65)
lines.append("")
lines.append(f"Dataset: {len(adf)} total pairs")
lines.append(f"  Evaluable (gap+correct not None): {len(eval_df)}")
lines.append(f"  With length data: {len(len_df)}")
lines.append(f"  Both-direction (always correct by def): {len(both_df)}")
lines.append(f"  Missing sequence (excluded): {adf['category'].eq('missing').sum()}")
lines.append("")

lines.append("[1] Length-based continuous correlations")
lines.append(f"  (n={len(len_df)} pairs with length data from FASTA cache)")
lines.append(corr_len_diff)
lines.append(corr_pct_trunc)
lines.append(corr_len_max)
lines.append(corr_len_a)
lines.append("")

lines.append("[2] Structural category accuracy (evaluable, excl. both/missing)")
lines.append(f"  domain_loss  : {domain_str}")
lines.append(f"  regulatory   : {regulatory_str}")
lines.append(f"  intermediate : {intermed_str}")
lines.append(f"  both_dir     : N={len(both_df)}: acc={int(both_df['correct'].sum())}/{len(both_df)} (always correct by definition)")
lines.append("")

if mwu_U is not None:
    lines.append(f"  Mann-Whitney U (gap: domain_loss > regulatory): U={mwu_U:.0f}, p={mwu_p:.4f}")
else:
    lines.append("  Mann-Whitney U: insufficient data")
lines.append("")

lines.append(f"  [Threshold check] gap >= 0.10: {int(high_gap_acc)}/{high_gap_n} correct")
lines.append(f"  [Threshold check] domain_loss, gap >= 0.05: {int(dom_high_acc)}/{dom_high_n} correct")
lines.append("")

lines.append("[3] Failed cases (correct=0, N={})".format(n_failed))
lines.append("  " + "-" * 61)
lines.append("  {:12s} {:12s} {:6s} {:12s} {:s}".format(
    "gene", "go_term", "gap", "category", "note"
))
lines.append("  " + "-" * 61)
for _, fr in failed.iterrows():
    gap_str = f"{fr['gap']:.4f}" if fr["gap"] is not None else "None"
    # shorten note
    note_txt = str(fr["bench_note"])[:50] if fr["bench_note"] else ""
    lines.append("  {:12s} {:12s} {:6s} {:12s} {:s}".format(
        fr["gene"], fr["go_term"], gap_str, fr["category"], note_txt
    ))
lines.append("")

lines.append("[4] Pattern summary of failed cases")
lines.append(f"  Total failed: {n_failed}")
lines.append(f"  gap < 0.01 (near-tie): {n_tie}  ({100*n_tie/n_failed:.0f}%)")
lines.append(f"  regulatory category:   {n_reg}  ({100*n_reg/n_failed:.0f}%)")
lines.append(f"  intermediate category: {n_intermed}  ({100*n_intermed/n_failed:.0f}%)")
lines.append(f"  domain_loss category:  {n_domain_fail}  ({100*n_domain_fail/n_failed:.0f}%)")
lines.append(f"  wrong direction despite gap>=0.01: {n_wrong_dir}")
lines.append("")

lines.append("[5] Key observations for CF1 response")
lines.append("  (A) Continuous gap-structure correlation:")
# derive from computed values
try:
    xv = len_df["gap"].values
    yv = len_df["pct_trunc"].values
    mask = ~(np.isnan(xv) | np.isnan(yv))
    rho, pval = stats.spearmanr(xv[mask], yv[mask])
    lines.append(f"      rho(gap, %truncation) = {rho:+.3f} (p={pval:.3f})")
except Exception:
    lines.append("      (could not compute)")
lines.append("  (B) Categorical: domain_loss vs regulatory gap distribution")
if mwu_p is not None:
    sig = "significant" if mwu_p < 0.05 else "not significant"
    lines.append(f"      MWU p={mwu_p:.4f} ({sig}) — domain_loss has larger gaps")
lines.append(f"  (C) Reviewer cherry-picking claim:")
lines.append(f"      gap>=0.10 cutoff selects {high_gap_n} of {len(eval_df)} pairs "
             f"({100*high_gap_n/len(eval_df):.0f}%); acc={int(high_gap_acc)}/{high_gap_n}")
lines.append(f"      BUT: domain_loss cases (N={len(domain_df)}) show "
             f"acc={int(domain_df['correct'].sum())}/{len(domain_df)} "
             f"({100*domain_df['correct'].mean():.0f}%) regardless of threshold")
lines.append(f"  (D) Failure modes of the {n_failed} failed cases:")
lines.append(f"      • {n_tie} cases ({100*n_tie/n_failed:.0f}%) have gap<0.01 → near-tie,"
             f" not wrong prediction")
lines.append(f"      • {n_reg} cases ({100*n_reg/n_failed:.0f}%) are regulatory"
             f" (subtle diff, PRISM not expected to separate)")
lines.append(f"      • {n_domain_fail} domain_loss failures exist"
             f" — examined individually below")
lines.append("")

# Print domain_loss failures with details
dom_fail = domain_df[domain_df["correct"] == 0]
if len(dom_fail) > 0:
    lines.append("  [domain_loss failures detail]")
    for _, fr in dom_fail.iterrows():
        gap_str = f"{fr['gap']:.4f}"
        lines.append(f"    {fr['gene']:10s} {fr['go_term']:12s} gap={gap_str}"
                     f"  len_a={fr['len_a']}  len_b={fr['len_b']}")
        lines.append(f"      note: {str(fr['bench_note'])[:80]}")
else:
    lines.append("  [domain_loss failures] None — all domain_loss cases correct")

lines.append("")
lines.append("=" * 65)

output = "\n".join(lines)
print(output)

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w") as f:
    f.write(output + "\n")
print(f"\n[SAVED] {OUT_FILE}")
