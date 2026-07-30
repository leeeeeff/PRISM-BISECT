"""
exp_fluid_stage2d_fisher_expanded.py
=====================================
Expanded bio-validation for fluid-trajectory side-branch candidates.

Stage 2b had Fisher p=0.13 (n=44 curated references, n=135 unique genes).
This analysis swaps in:
  - Tier A: APPRIS ALTERNATIVE (n~2178 in pilot) — curated functional-alternative label
  - Tier B: non-MANE + gene-dominant expression (n~5682)
  - Tier C: Tier A ∩ Tier B (n~794) — highest confidence
  - Tier U: Tier A ∪ Tier B (n~7066) — broadest

Test (Fisher one-sided):
  H1: side-branch candidates are enriched in functionally-alternative
  isoforms compared to the background of all non-side-branch pilot isoforms.

Stratifications reported:
  1. All candidates (GO+ and GO−)
  2. GO− only  (encoded-but-unexpressed hypothesis — the key claim)
  3. By GO type (early / mid / late / flat)
  4. By peak_layer band (L1-10 / L11-20 / L21-30)
  5. By sb_score quantile (top 25% vs bottom 75%)

Outputs:
  reports/fluid_stage2/fisher_expanded_20260706.json   — all results
  reports/fluid_stage2/fisher_expanded_20260706.txt    — readable report
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL = ROOT / "model"
OUT_DIR = ROOT.parent / "reports" / "fluid_stage2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAMP = datetime.now().strftime("%Y%m%d_%H%M")

SB_CSV = OUT_DIR / "all_side_branches_20260706_1957.csv"
REF_TSV = DATA / "reference_labels_v1.tsv"
ISO_NPY = MODEL / "my_isoform_list_fixed.npy"


# ─────────────────────────────────────────────────────────
def load_pilot_isoforms() -> set[str]:
    raw = np.load(ISO_NPY, allow_pickle=True)
    ids = [x.decode() if isinstance(x, bytes) else x for x in raw]
    return set(ids)


def load_ref() -> pd.DataFrame:
    ref = pd.read_csv(REF_TSV, sep="\t", dtype=str, na_filter=False)
    ref["is_gene_dominant"] = ref["is_gene_dominant"].map({"True": True, "False": False})
    ref["is_alt_functional"] = ref["is_alt_functional"].map({"True": True, "False": False})
    # Tier flags
    ref["tier_A"] = ref["appris_label"].str.startswith("ALTERNATIVE")
    ref["tier_B"] = (
        ~ref["mane_status"].isin(["MANE Select", "MANE Plus Clinical"])
        & ref["is_gene_dominant"]
        & ref["tx_id_versioned"].str.startswith("ENST")
    )
    ref["tier_C"] = ref["tier_A"] & ref["tier_B"]
    ref["tier_U"] = ref["tier_A"] | ref["tier_B"]
    return ref


def enrich_with_tier(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Merge side-branch candidates (or pilot background) with tier labels."""
    ref_lookup = ref.set_index("tx_id_versioned")[
        ["tier_A", "tier_B", "tier_C", "tier_U", "appris_label", "mane_status"]
    ]
    merged = df.join(ref_lookup, on="iso", how="left")
    for col in ["tier_A", "tier_B", "tier_C", "tier_U"]:
        merged[col] = merged[col].fillna(False)
    return merged


def fisher_test(
    n_cand: int,
    n_cand_pos: int,
    n_bg: int,
    n_bg_pos: int,
    label: str,
) -> dict:
    """Fisher exact (greater: candidate enriched)."""
    # contingency: [[cand_pos, cand_neg], [bg_pos, bg_neg]]
    cand_neg = n_cand - n_cand_pos
    bg_neg = n_bg - n_bg_pos
    table = [[n_cand_pos, cand_neg], [n_bg_pos, bg_neg]]
    or_, p = fisher_exact(table, alternative="greater")
    return {
        "label": label,
        "n_cand": n_cand,
        "n_cand_pos": n_cand_pos,
        "n_bg": n_bg,
        "n_bg_pos": n_bg_pos,
        "OR": round(float(or_), 3),
        "p_one_sided": round(float(p), 4),
    }


def run_stratified(
    sb_unique: pd.DataFrame,
    bg_unique: pd.DataFrame,
    tier: str,
    label: str,
) -> list[dict]:
    """Run Fisher for one tier across all stratifications."""
    results = []

    def _go(cand: pd.DataFrame, bg: pd.DataFrame, strat: str) -> dict:
        return fisher_test(
            len(cand),
            int(cand[tier].sum()),
            len(bg),
            int(bg[tier].sum()),
            f"{label} | {strat}",
        )

    # 1. All candidates
    results.append(_go(sb_unique, bg_unique, "all"))

    # 2. GO− only
    sb_neg = sb_unique[sb_unique["any_go_pos"] == False]
    results.append(_go(sb_neg, bg_unique, "GO-neg only"))

    # 3. GO type
    for gtype in ["early", "mid", "late", "flat"]:
        sub = sb_unique[sb_unique["dominant_type"] == gtype]
        if len(sub) >= 5:
            results.append(_go(sub, bg_unique, f"type={gtype}"))

    # 4. Peak layer band
    for band, lo, hi in [("L1-10", 1, 10), ("L11-20", 11, 20), ("L21-30", 21, 30)]:
        sub = sb_unique[sb_unique["peak_layer"].between(lo, hi)]
        if len(sub) >= 5:
            results.append(_go(sub, bg_unique, f"peak={band}"))

    # 5. Top-25% sb_score
    q75 = sb_unique["max_sb_score"].quantile(0.75)
    sub = sb_unique[sb_unique["max_sb_score"] >= q75]
    if len(sub) >= 5:
        results.append(_go(sub, bg_unique, "top25%_sb"))

    return results


def main():
    print("[load] side-branch candidates...")
    sb = pd.read_csv(SB_CSV)
    print(f"  rows={len(sb)}, unique iso={sb['iso'].nunique()}")

    print("[load] reference labels...")
    ref = load_ref()

    print("[load] pilot isoform list...")
    pilot = load_pilot_isoforms()
    pilot_enst = {s for s in pilot if s.startswith("ENST")}

    # ── Build unique-iso side-branch table ─────────────────────────
    # An isoform appears in multiple GO bundles; collapse to per-iso stats.
    # dominant_type = mode over its bundles; max_sb_score; any_go_pos
    sb["any_go_pos"] = sb["is_go_pos"].astype(bool)
    agg = sb.groupby("iso").agg(
        max_sb_score=("sb_score", "max"),
        peak_layer=("peak_layer", "median"),
        dominant_type=("type", lambda x: x.mode().iloc[0]),
        any_go_pos=("any_go_pos", "any"),
        n_bundles=("go", "nunique"),
    ).reset_index()
    agg["peak_layer"] = agg["peak_layer"].round().astype(int)

    # Add tier labels via ref
    sb_unique = enrich_with_tier(agg.set_index("iso"), ref).reset_index()
    sb_unique.rename(columns={"index": "iso"}, inplace=True)
    print(f"  unique candidates: {len(sb_unique)}")

    # ── Build background ────────────────────────────────────────────
    # Background = pilot ENST not in side-branch set
    sb_ids = set(sb_unique["iso"].tolist())
    bg_ids = pilot_enst - sb_ids
    bg_ref = ref[ref["tx_id_versioned"].isin(bg_ids)].drop_duplicates("tx_id_versioned")
    bg_ref = bg_ref.rename(columns={"tx_id_versioned": "iso"}).set_index("iso")
    bg_unique = pd.DataFrame(index=list(bg_ids)).join(
        bg_ref[["tier_A", "tier_B", "tier_C", "tier_U"]], how="left"
    )
    for col in ["tier_A", "tier_B", "tier_C", "tier_U"]:
        bg_unique[col] = bg_unique[col].fillna(False)
    bg_unique["any_go_pos"] = False
    bg_unique["dominant_type"] = "unknown"
    bg_unique["max_sb_score"] = 0.0
    bg_unique["peak_layer"] = 15
    print(f"  background size: {len(bg_unique)}")

    # ── Print tier counts in each group ────────────────────────────
    print("\n--- Tier membership in side-branch candidates ---")
    for t in ["tier_A", "tier_B", "tier_C", "tier_U"]:
        print(f"  {t}: {sb_unique[t].sum()} / {len(sb_unique)} candidates")
    print("--- Tier membership in background ---")
    for t in ["tier_A", "tier_B", "tier_C", "tier_U"]:
        print(f"  {t}: {bg_unique[t].sum()} / {len(bg_unique)} background")

    # ── Run Fisher for each tier ───────────────────────────────────
    all_results = []
    for tier, label in [
        ("tier_A", "APPRIS-ALTERNATIVE"),
        ("tier_B", "nonMANE-dominant"),
        ("tier_C", "TierA∩B"),
        ("tier_U", "TierA∪B"),
    ]:
        r = run_stratified(sb_unique, bg_unique, tier, label)
        all_results.extend(r)

    # ── Print report ───────────────────────────────────────────────
    report_lines = [
        "=" * 70,
        "FLUID TRAJECTORY SIDE-BRANCH BIO-VALIDATION (EXPANDED REFERENCE)",
        f"Timestamp: {STAMP}",
        f"Candidates (unique iso): {len(sb_unique)}",
        f"Background (unique pilot ENST not in candidates): {len(bg_unique)}",
        "=" * 70,
        "",
    ]

    sig_count = 0
    for r in all_results:
        sig = "*** " if r["p_one_sided"] < 0.05 else ("*   " if r["p_one_sided"] < 0.10 else "    ")
        if r["p_one_sided"] < 0.05:
            sig_count += 1
        line = (
            f"{sig}{r['label']:<45} "
            f"OR={r['OR']:5.2f}  "
            f"p={r['p_one_sided']:.4f}  "
            f"cand={r['n_cand_pos']}/{r['n_cand']}  "
            f"bg={r['n_bg_pos']}/{r['n_bg']}"
        )
        report_lines.append(line)

    report_lines += [
        "",
        f"Significant (p<0.05, one-sided Fisher): {sig_count} / {len(all_results)}",
        "",
        "Tier definitions:",
        "  Tier A: APPRIS ALTERNATIVE:1/2 (curated functional-alternative)",
        "  Tier B: non-MANE Select + gene-dominant in >=1 muscle sample",
        "  Tier C: Tier A ∩ Tier B (highest confidence)",
        "  Tier U: Tier A ∪ Tier B (broadest)",
    ]

    report_txt = "\n".join(report_lines)
    print("\n" + report_txt)

    # ── Save ───────────────────────────────────────────────────────
    json_path = OUT_DIR / f"fisher_expanded_{STAMP}.json"
    txt_path = OUT_DIR / f"fisher_expanded_{STAMP}.txt"

    with open(json_path, "w") as f:
        json.dump({"results": all_results, "n_cand": len(sb_unique), "n_bg": len(bg_unique)}, f, indent=2)
    with open(txt_path, "w") as f:
        f.write(report_txt)

    print(f"\n[write] {json_path}")
    print(f"[write] {txt_path}")


if __name__ == "__main__":
    main()
