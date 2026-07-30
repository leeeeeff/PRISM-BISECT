"""
compare_muscle_brain_fisher.py
==============================
Head-to-head comparison of per-layer Fisher discriminant peaks between muscle
and brain tissue on the same 279 GO seed set.

Outputs:
  reports/exp_C1_layer_probe_279/muscle_vs_brain_fisher.tsv
  reports/exp_C1_layer_probe_279/muscle_vs_brain_summary.md
"""
from __future__ import annotations
import json, math
from pathlib import Path

OUT_DIR = Path("/home/welcome1/sw1686/DIFFUSE/reports/exp_C1_layer_probe_279")
MUS = OUT_DIR / "layer_probe_279_fisher.json"
BRN = OUT_DIR / "layer_probe_279_fisher_brain.json"


def bucket(pl: int) -> str:
    if pl <= 10: return "Early"
    if pl <= 20: return "Mid"
    return "Late"


def main():
    m = json.load(open(MUS))
    b = json.load(open(BRN))

    common = sorted(set(m["per_go"].keys()) & set(b["per_go"].keys()))
    rows = []
    for go in common:
        mi, bi = m["per_go"][go], b["per_go"][go]
        rows.append({
            "go": go,
            "name": mi["name"],
            "cat": mi["category"],
            "m_peak": mi["peak_layer"],
            "m_fisher": mi["peak_fisher"],
            "m_npos": mi["n_pos_te"],
            "b_peak": bi["peak_layer"],
            "b_fisher": bi["peak_fisher"],
            "b_npos": bi.get("n_pos_br", bi.get("n_pos_te", 0)),
            "shift": bi["peak_layer"] - mi["peak_layer"],
            "m_bucket": bucket(mi["peak_layer"]),
            "b_bucket": bucket(bi["peak_layer"]),
        })

    # TSV
    tsv = OUT_DIR / "muscle_vs_brain_fisher.tsv"
    with open(tsv, "w") as f:
        f.write("go\tname\tcat\tm_peak\tm_fisher\tm_npos\tb_peak\tb_fisher\tb_npos\tshift\tm_bucket\tb_bucket\n")
        for r in rows:
            f.write(f"{r['go']}\t{r['name']}\t{r['cat']}\t{r['m_peak']}\t{r['m_fisher']:.4f}\t{r['m_npos']}\t"
                    f"{r['b_peak']}\t{r['b_fisher']:.4f}\t{r['b_npos']}\t{r['shift']}\t{r['m_bucket']}\t{r['b_bucket']}\n")
    print(f"[saved] {tsv}")

    # Summary stats
    n = len(rows)
    same_bucket = sum(1 for r in rows if r["m_bucket"] == r["b_bucket"])
    early2mid = sum(1 for r in rows if r["m_bucket"] == "Early" and r["b_bucket"] == "Mid")
    early2late = sum(1 for r in rows if r["m_bucket"] == "Early" and r["b_bucket"] == "Late")
    mid2early = sum(1 for r in rows if r["m_bucket"] == "Mid" and r["b_bucket"] == "Early")
    mid2late = sum(1 for r in rows if r["m_bucket"] == "Mid" and r["b_bucket"] == "Late")
    late2early = sum(1 for r in rows if r["m_bucket"] == "Late" and r["b_bucket"] == "Early")
    late2mid = sum(1 for r in rows if r["m_bucket"] == "Late" and r["b_bucket"] == "Mid")

    mean_shift = sum(r["shift"] for r in rows) / n
    var_shift = sum((r["shift"] - mean_shift)**2 for r in rows) / n
    sd_shift = math.sqrt(var_shift)
    up = sum(1 for r in rows if r["shift"] > 0)
    down = sum(1 for r in rows if r["shift"] < 0)
    same = sum(1 for r in rows if r["shift"] == 0)

    # Category-stratified
    by_cat = {}
    for cat in ["BP", "MF", "CC"]:
        cat_rows = [r for r in rows if r["cat"] == cat]
        if not cat_rows: continue
        cs = sum(r["shift"] for r in cat_rows) / len(cat_rows)
        by_cat[cat] = {"n": len(cat_rows), "mean_shift": cs}

    # Fisher magnitude
    mean_m_f = sum(r["m_fisher"] for r in rows) / n
    mean_b_f = sum(r["b_fisher"] for r in rows) / n

    md = OUT_DIR / "muscle_vs_brain_summary.md"
    with open(md, "w") as f:
        f.write("# Muscle vs Brain Fisher L1~L30 — 279 GO Comparison\n\n")
        f.write(f"- Common GOs (both eval): **{n}**\n")
        f.write(f"- Muscle Fisher method: {m.get('method')}\n")
        f.write(f"- Brain Fisher method: {b.get('method')}\n\n")

        f.write("## Peak-layer shift (brain − muscle)\n\n")
        f.write(f"- Mean shift: **{mean_shift:+.2f}** layers (SD {sd_shift:.2f})\n")
        f.write(f"- Shifted deeper: {up}/{n} ({100*up/n:.1f}%)\n")
        f.write(f"- Shifted shallower: {down}/{n} ({100*down/n:.1f}%)\n")
        f.write(f"- Same layer: {same}/{n} ({100*same/n:.1f}%)\n\n")

        f.write("## Bucket concordance\n\n")
        f.write(f"- Same bucket (Early/Mid/Late): **{same_bucket}/{n} ({100*same_bucket/n:.1f}%)**\n")
        f.write("- Bucket transitions:\n")
        f.write(f"  - Early → Mid: {early2mid}\n")
        f.write(f"  - Early → Late: {early2late}\n")
        f.write(f"  - Mid → Early: {mid2early}\n")
        f.write(f"  - Mid → Late: {mid2late}\n")
        f.write(f"  - Late → Early: {late2early}\n")
        f.write(f"  - Late → Mid: {late2mid}\n\n")

        f.write("## Bucket distribution\n\n")
        f.write("| Bucket | Muscle BP/MF/CC | Brain BP/MF/CC |\n|---|---|---|\n")
        m_b = m["bucket_summary"]; b_b = b["bucket_summary"]
        for bk in ["Early (L1-10)", "Mid (L11-20)", "Late (L21-30)"]:
            f.write(f"| {bk} | {m_b[bk]['BP']}/{m_b[bk]['MF']}/{m_b[bk]['CC']} | "
                    f"{b_b[bk]['BP']}/{b_b[bk]['MF']}/{b_b[bk]['CC']} |\n")
        f.write(f"\n## Category-stratified mean shift\n\n")
        for cat, s in by_cat.items():
            f.write(f"- {cat}: mean shift = {s['mean_shift']:+.2f} (n={s['n']})\n")

        f.write(f"\n## Fisher magnitude (mean peak signal)\n")
        f.write(f"- Muscle: {mean_m_f:.3f}\n")
        f.write(f"- Brain:  {mean_b_f:.3f}\n\n")

        f.write("## Top 10 largest deeper shifts (brain > muscle)\n\n")
        f.write("| GO | Name | Cat | m_peak | b_peak | shift |\n|---|---|---|---|---|---|\n")
        for r in sorted(rows, key=lambda x: -x["shift"])[:10]:
            f.write(f"| {r['go']} | {r['name'][:40]} | {r['cat']} | L{r['m_peak']} | L{r['b_peak']} | {r['shift']:+d} |\n")

        f.write("\n## Top 10 largest shallower shifts (brain < muscle)\n\n")
        f.write("| GO | Name | Cat | m_peak | b_peak | shift |\n|---|---|---|---|---|---|\n")
        for r in sorted(rows, key=lambda x: x["shift"])[:10]:
            f.write(f"| {r['go']} | {r['name'][:40]} | {r['cat']} | L{r['m_peak']} | L{r['b_peak']} | {r['shift']:+d} |\n")

    print(f"[saved] {md}")


if __name__ == "__main__":
    main()
