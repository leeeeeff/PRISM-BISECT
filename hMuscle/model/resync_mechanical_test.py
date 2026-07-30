#!/usr/bin/env python3
"""
resync_mechanical_test.py
===========================
Confound-vs-mediator discrimination for resync_failure_binary's coupling with
size. Definition recap (build_severity_pairs.py): from the first changed
position in the alignment, resync_failure=1 iff the longest single contiguous
"equal" block downstream covers <50% of the remaining sequence.

TEST 1 (mechanical predictability): fit P(resync_failure=1 | size) via
logistic regression, per tissue. Report AUC (how much of resync_failure is
"explained" by size alone) and whether the fitted probability CURVE overlaps
between muscle and brain at matched size deciles.
  - If curves coincide (no tissue gap in P(resync|size) at matched size):
    resync's size-coupling is tissue-INDEPENDENT -- consistent with a
    largely mechanical/definitional relationship (more separate/large edits
    mechanically reduce the chance of one long re-sync block, regardless of
    tissue biology).
  - If curves diverge at matched size (e.g. muscle has higher resync-failure
    rate than brain even for isoforms with identical edit size): there is a
    genuine tissue-specific effect beyond pure size mechanics -- supports
    treating size as a partial mediator of a real muscle-specific signal,
    not a pure confound to be scrubbed.

TEST 2 (decile table): empirical P(resync=1) per size decile, both tissues
side by side, for direct visual comparison (no model-form assumption).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')


def load(tissue):
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    log_size = np.log1p(df['size'])
    df['size_z'] = (log_size - log_size.mean()) / log_size.std()
    df['log_size'] = log_size
    return df


def test1_logistic(dfs):
    print("=== TEST 1: logistic P(resync_failure=1 | size), per tissue ===")
    for tissue, df in dfs.items():
        X = df[['size_z']].to_numpy()
        y = df['resync_failure_binary'].to_numpy()
        clf = LogisticRegression().fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        print(f"{tissue}: coef(size_z)={clf.coef_[0][0]:+.4f}  intercept={clf.intercept_[0]:+.4f}  "
              f"AUC(size alone predicting resync)={auc:.3f}  base_rate={y.mean():.3f}")
    return


def test2_decile_table(dfs):
    print("\n=== TEST 2: empirical P(resync=1) by RAW SIZE decile (matched bin edges across tissues) ===")
    pooled_size = pd.concat([dfs['muscle']['size'], dfs['brain']['size']])
    edges = np.quantile(pooled_size, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    for tissue, df in dfs.items():
        d = df.copy()
        d['decile'] = pd.cut(d['size'], edges, labels=False, include_lowest=True)
        g = d.groupby('decile').agg(n=('resync_failure_binary', 'size'),
                                     size_median=('size', 'median'),
                                     resync_rate=('resync_failure_binary', 'mean'))
        g['tissue'] = tissue
        rows.append(g)
    tab = pd.concat(rows).reset_index()
    piv = tab.pivot(index='decile', columns='tissue', values=['size_median', 'n', 'resync_rate'])
    print(piv.round(3).to_string())
    # gap per decile
    gap = tab.pivot(index='decile', columns='tissue', values='resync_rate')
    gap['muscle_minus_brain'] = gap['muscle'] - gap['brain']
    print("\nPer-decile muscle-minus-brain resync rate gap (matched size):")
    print(gap.round(3).to_string())


def main():
    dfs = {t: load(t) for t in ['muscle', 'brain']}
    test1_logistic(dfs)
    test2_decile_table(dfs)


if __name__ == '__main__':
    main()
