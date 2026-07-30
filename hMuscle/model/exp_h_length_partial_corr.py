"""
UniProt benchmark: partial correlation between prediction gap and direction accuracy,
controlling for sequence length difference.

Addresses paper-critic MR2: length bias concern (ρ=+0.617 between gap and len_diff).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

SEQ_CACHE = "/home/welcome1/sw1686/DIFFUSE/reports/exp_h_uniprot_eval/seq_cache"
EVAL_TSV   = "/home/welcome1/sw1686/DIFFUSE/reports/exp_h_uniprot_eval/pairwise_eval_v3_remapped.tsv"
BENCH_CSV  = "/home/welcome1/sw1686/DIFFUSE/reports/exp_g_uniprot/uniprot_isoform_benchmark.csv"


def fasta_len(accession):
    # canonical isoform stored without suffix (e.g. P00533.fasta not P00533-1.fasta)
    if accession.endswith("-1"):
        path = os.path.join(SEQ_CACHE, f"{accession[:-2]}.fasta")
    else:
        path = os.path.join(SEQ_CACHE, f"{accession}.fasta")
    if not os.path.exists(path):
        # fallback: try with the accession as-is
        path2 = os.path.join(SEQ_CACHE, f"{accession}.fasta")
        if not os.path.exists(path2):
            return None
        path = path2
    seq = ""
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq += line.strip()
    return len(seq)


def partial_corr_manual(x, y, z):
    """Partial correlation of x and y controlling for z (all arrays)."""
    def residuals(a, b):
        slope, intercept, *_ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    rx = residuals(x, z)
    ry = residuals(y, z)
    r, p = stats.pearsonr(rx, ry)
    return r, p


def main():
    df = pd.read_csv(EVAL_TSV, sep="\t")
    bench = pd.read_csv(BENCH_CSV)

    print("="*60)
    print("  UniProt Benchmark — Length Partial Correlation Analysis")
    print("="*60)

    # Get sequence lengths for each pair
    lens_a, lens_b, len_diffs = [], [], []
    valid_rows = []

    for _, row in df.iterrows():
        if pd.isna(row.get("score_a")) or row.get("note", "").startswith("sequence not embedded"):
            continue

        acc_a = row["iso_a"]
        acc_b = row["iso_b"]

        la = fasta_len(acc_a)
        lb = fasta_len(acc_b)

        if la is None or lb is None:
            print(f"  [WARN] No fasta for {acc_a} or {acc_b}")
            continue

        lens_a.append(la)
        lens_b.append(lb)
        len_diffs.append(abs(la - lb))
        valid_rows.append(row)

    df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
    df_valid["len_a"] = lens_a
    df_valid["len_b"] = lens_b
    df_valid["len_diff"] = len_diffs
    # "correct" column may be string "True"/"False" from TSV
    df_valid["correct"] = df_valid["correct"].map({"True": True, "False": False, True: True, False: False})
    df_valid["correct_int"] = df_valid["correct"].astype(int)

    n = len(df_valid)
    print(f"\nEvaluable pairs with length data: {n}")

    gap = df_valid["gap"].values.astype(float)
    correct = df_valid["correct_int"].values.astype(float)
    len_diff = df_valid["len_diff"].values.astype(float)

    # --- Raw Spearman: gap vs len_diff ---
    r_gap_len, p_gap_len = stats.spearmanr(gap, len_diff)
    print(f"\n1. Spearman ρ(gap, len_diff)       = {r_gap_len:+.3f}  p = {p_gap_len:.4f}")

    # --- Raw Spearman: gap vs correct ---
    r_gap_corr, p_gap_corr = stats.spearmanr(gap, correct)
    print(f"2. Spearman ρ(gap, correct)         = {r_gap_corr:+.3f}  p = {p_gap_corr:.4f}")

    # --- Raw Spearman: len_diff vs correct ---
    r_len_corr, p_len_corr = stats.spearmanr(len_diff, correct)
    print(f"3. Spearman ρ(len_diff, correct)    = {r_len_corr:+.3f}  p = {p_len_corr:.4f}")

    # --- Partial correlation: gap vs correct | len_diff ---
    r_partial, p_partial = partial_corr_manual(gap, correct, len_diff)
    print(f"\n4. Partial r(gap, correct | len_diff) = {r_partial:+.3f}  p = {p_partial:.4f}")
    print(f"   → After controlling for length difference, gap {'IS' if p_partial < 0.05 else 'IS NOT'} "
          f"significantly associated with direction accuracy")

    # --- Point-biserial: gap vs correct ---
    r_pb, p_pb = stats.pointbiserialr(correct, gap)
    print(f"\n5. Point-biserial r(correct, gap)   = {r_pb:+.3f}  p = {p_pb:.4f}")

    # --- Stratified analysis: gap >= 0.10 vs < 0.05 ---
    high_gap = df_valid[df_valid["gap"] >= 0.10]
    low_gap  = df_valid[df_valid["gap"] < 0.05]

    print(f"\n--- Stratified by gap ---")
    print(f"  gap ≥ 0.10 (n={len(high_gap)}): accuracy = {high_gap['correct'].sum()}/{len(high_gap)}")
    print(f"    mean len_diff = {high_gap['len_diff'].mean():.0f} aa")
    print(f"  gap < 0.05  (n={len(low_gap)}):  accuracy = {low_gap['correct'].sum()}/{len(low_gap)}")
    print(f"    mean len_diff = {low_gap['len_diff'].mean():.0f} aa")

    # --- Logistic regression: correct ~ gap + len_diff ---
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = np.column_stack([gap, len_diff])
        y = correct.astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr = LogisticRegression(random_state=42)
        lr.fit(X_scaled, y)

        coefs = lr.coef_[0]
        print(f"\n--- Logistic regression: correct ~ gap + len_diff ---")
        print(f"  coef(gap)      = {coefs[0]:+.3f}")
        print(f"  coef(len_diff) = {coefs[1]:+.3f}")
        print(f"  → Gap coefficient is {'LARGER' if abs(coefs[0]) > abs(coefs[1]) else 'SMALLER'} "
              f"than len_diff coefficient")
    except ImportError:
        print("\n  [SKIP] sklearn not available for logistic regression")

    # --- Per-pair breakdown for high-gap cases ---
    print(f"\n--- High-gap cases (gap ≥ 0.10) breakdown ---")
    print(f"{'Gene':<10} {'Correct':<8} {'Gap':>8} {'len_a':>7} {'len_b':>7} {'len_diff':>9} {'GO':<15} {'Note'}")
    print("-"*95)
    for _, row in high_gap.sort_values("gap", ascending=False).iterrows():
        c = "✓" if row["correct"] else "✗"
        print(f"{row['gene']:<10} {c:<8} {row['gap']:>8.3f} {row['len_a']:>7.0f} "
              f"{row['len_b']:>7.0f} {row['len_diff']:>9.0f} {row['go_term']:<15} {str(row.get('note',''))[:40]}")

    print("\n" + "="*60)
    print("  INTERPRETATION")
    print("="*60)

    if p_partial < 0.05:
        print(f"\n  ✓ Partial correlation is SIGNIFICANT (r={r_partial:+.3f}, p={p_partial:.4f})")
        print(f"    → Gap predicts direction accuracy INDEPENDENTLY of length difference.")
        print(f"    → Length bias concern is NOT SUPPORTED.")
        print(f"\n  Manuscript defense: 'After controlling for sequence length difference,")
        print(f"  the prediction gap remains significantly associated with direction accuracy")
        print(f"  (partial r={r_partial:.3f}, p={p_partial:.4f}), confirming that the gap ≥ 0.10")
        print(f"  threshold reflects domain-completeness differences beyond simple length effects.'")
    else:
        print(f"\n  ✗ Partial correlation is NOT significant (r={r_partial:+.3f}, p={p_partial:.4f})")
        print(f"    → Cannot rule out length bias as a confound.")
        print(f"    → Revise claim: 'gap-based discrimination may partially reflect length differences'.")

    # Save results
    out_path = "/home/welcome1/sw1686/DIFFUSE/reports/exp_h_uniprot_eval/length_partial_corr.tsv"
    df_valid[["gene","iso_a","iso_b","gap","correct","len_a","len_b","len_diff","go_term","note"]].to_csv(
        out_path, sep="\t", index=False
    )
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
