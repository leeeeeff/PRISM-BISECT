#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
axis_functional_matrix_reliance.py
====================================
Test-time (reliance) variant of axis_functional_matrix.py.

axis_functional_matrix.py used occlusion-RETRAIN and found universal ~0 drops
INCLUDING the positive controls (size->axis5, domain_binary->axis3): the 1280-d
representation encodes every covariate redundantly, so refitting after removing
a single low-variance 8-PCA axis recovers it fully (uniqueness lens is null for
all -> uninformative for localising WHICH axis carries the function).

This is the DR-AUC-occlusion analog the reference note actually used (reference-
esm2-pca-axes-final.md reported "DR test z" = test-time): train the covariate
probe once on FULL 1280-d Dphi, then at test time remove axis_k from the (held-
out) features WITHOUT refitting, and measure how much the frozen probe's
prediction of C degrades. This measures RELIANCE (does the best C-predictor
lean on axis_k), recovers the positive controls, and localises function.

Reporting keeps BOTH z-vs-random-null AND relative effect size
(delta / (baseline - chance)); a verdict requires both
(approach-axis-encoding-vs-usage-occlusion.md self-correction).

PRE-REGISTERED (S2): size->axis5 and domain_binary->axis3 MUST be the per-
covariate standouts or the method is broken. disorder->axis0 expected.
nterm_overlap / resync_failure_binary genuinely open.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_FOLDS = 5
N_RANDOM = 20
N_AXES = 8
SEED = 42

BINARY_COVS = ['domain_binary', 'nterm_overlap', 'resync_failure_binary']
AXIS_LABEL = {
    0: 'axis0(betaSheet/TM)', 1: 'axis1(LRR/Ig)', 2: 'axis2(Pro-turn)',
    3: 'axis3(domain)', 4: 'axis4(helix-charge)', 5: 'axis5(LENGTH)',
    6: 'axis6(KRAB-ZNF)', 7: 'axis7(acidic-hel)',
}


def load_layer_stats():
    W = np.load(ROOT / 'reports/v20b_pca_interp/W_axes_8x640.npy')
    mu = np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_mu.npy')
    sd = np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_sd.npy')
    return W, mu, sd


def load_raw_layers(tissue):
    if tissue == 'muscle':
        L15 = np.load(ROOT / 'hMuscle/data/esm2_layer_15_t30_150M.npy').astype(np.float32)
        L30 = np.load(ROOT / 'hMuscle/data/esm2_layer_30_t30_150M.npy').astype(np.float32)
    else:
        L15 = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer15_t30_150M.npy').astype(np.float32)
        L30 = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
    return L15, L30


def occlude(raw, mu_l, sd_l, direction):
    z = (raw - mu_l) / sd_l
    z_occ = z - np.outer(z @ direction, direction)
    return z_occ * sd_l + mu_l


def gene_disjoint_folds(genes, n_folds=N_FOLDS, seed=SEED):
    uniq = np.array(sorted(set(genes)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of_gene = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of_gene[g] for g in genes])


def build_dphi(L15, L30, long_idx, short_idx):
    return np.concatenate([L15[long_idx] - L15[short_idx],
                           L30[long_idx] - L30[short_idx]], axis=1)


def oof_testtime(Dphi_base, Dphi_occ, y, fold):
    """Train linear probe on FULL train Dphi_base; predict held-out test using
    Dphi_occ (occluded) standardised with train stats. Returns OOF predictions.
    When Dphi_occ is Dphi_base this yields the unoccluded baseline OOF."""
    pred = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr = fold != k
        te = fold == k
        mu = Dphi_base[tr].mean(0)
        sd = Dphi_base[tr].std(0) + 1e-8
        Xtr = np.column_stack([np.ones(tr.sum()), (Dphi_base[tr] - mu) / sd])
        beta, *_ = np.linalg.lstsq(Xtr, y[tr] - y[tr].mean(), rcond=None)
        Xte = np.column_stack([np.ones(te.sum()), (Dphi_occ[te] - mu) / sd])
        pred[te] = Xte @ beta
    return pred


def metric(y, pred, ctype):
    return roc_auc_score(y, pred) if ctype == 'binary' else stats.spearmanr(y, pred).correlation


def run_tissue(tissue, W, mu, sd):
    print(f"\n{'='*72}\n{tissue}\n{'='*72}", flush=True)
    L15, L30 = load_raw_layers(tissue)
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    long_idx = df['long_idx'].to_numpy()
    short_idx = df['short_idx'].to_numpy()
    fold = gene_disjoint_folds(df['gene'].to_numpy())

    targets = {c: ('binary', df[c].to_numpy(float)) for c in BINARY_COVS}
    targets['size'] = ('cont', np.log1p(df['size'].to_numpy(float)))
    targets['disorder_frac'] = ('cont', df['disorder_frac'].to_numpy(float))

    rng = np.random.default_rng(SEED)
    random_dirs = rng.normal(size=(N_RANDOM, 640))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
    axis_dirs = [W[k] / np.linalg.norm(W[k]) for k in range(N_AXES)]

    Dphi_base = build_dphi(L15, L30, long_idx, short_idx)

    # cache occluded Dphi per axis + random dir
    occ_dphi = {}
    for k in range(N_AXES):
        occ_dphi[f'axis{k}'] = build_dphi(occlude(L15, mu[14], sd[14], axis_dirs[k]),
                                          occlude(L30, mu[29], sd[29], axis_dirs[k]),
                                          long_idx, short_idx)
    for i in range(N_RANDOM):
        occ_dphi[f'rand{i}'] = build_dphi(occlude(L15, mu[14], sd[14], random_dirs[i]),
                                          occlude(L30, mu[29], sd[29], random_dirs[i]),
                                          long_idx, short_idx)

    rows = []
    for cname, (ctype, y) in targets.items():
        base = metric(y, oof_testtime(Dphi_base, Dphi_base, y, fold), ctype)
        chance = 0.5 if ctype == 'binary' else 0.0
        denom = (base - chance) if ctype == 'binary' else abs(base)

        null_deltas = np.array([base - metric(y, oof_testtime(Dphi_base, occ_dphi[f'rand{i}'], y, fold), ctype)
                                for i in range(N_RANDOM)])
        n_lo, n_hi = np.percentile(null_deltas, [2.5, 97.5])
        n_mean, n_sd = null_deltas.mean(), null_deltas.std()

        print(f"  [{cname}] baseline={base:+.4f}  random-null delta CI=[{n_lo:+.4f},{n_hi:+.4f}]", flush=True)
        for k in range(N_AXES):
            occ_metric = metric(y, oof_testtime(Dphi_base, occ_dphi[f'axis{k}'], y, fold), ctype)
            delta = base - occ_metric
            z = (delta - n_mean) / n_sd if n_sd > 0 else np.nan
            rel = delta / denom if abs(denom) > 1e-9 else np.nan
            used = (not (n_lo <= delta <= n_hi)) and (abs(rel) >= 0.10)
            rows.append({'tissue': tissue, 'covariate': cname, 'ctype': ctype,
                         'axis': k, 'axis_label': AXIS_LABEL[k], 'baseline': base,
                         'delta': delta, 'rel_effect': rel, 'z_vs_null': z, 'used': used})
            flag = '  <== USED' if used else ''
            print(f"      {AXIS_LABEL[k]:<20} delta={delta:+.4f} rel={rel:+.1%} z={z:+.1f}{flag}", flush=True)
    return rows


def main():
    W, mu, sd = load_layer_stats()
    all_rows = []
    for tissue in ['muscle', 'brain']:
        all_rows += run_tissue(tissue, W, mu, sd)
    out = pd.DataFrame(all_rows)
    out_path = ROOT / 'reports/severity_pairs/axis_functional_matrix_reliance.tsv'
    out.to_csv(out_path, sep='\t', index=False)
    print(f"\n[Save] {out_path}")

    print("\n=== per-covariate top axis (max |rel_effect|) ===")
    for (tis, cov), g in out.groupby(['tissue', 'covariate']):
        top = g.loc[g['rel_effect'].abs().idxmax()]
        print(f"  {tis:<6} {cov:<22} -> {top['axis_label']:<20} rel={top['rel_effect']:+.1%} z={top['z_vs_null']:+.1f}")

    print("\n=== FUNCTIONAL-USED (outside random-null AND |rel_effect|>=10%) ===")
    surv = out[out['used']]
    if len(surv) == 0:
        print("  NONE.")
    else:
        for _, r in surv.sort_values('rel_effect', key=np.abs, ascending=False).iterrows():
            print(f"  {r['tissue']:<6} {r['covariate']:<22} {r['axis_label']:<20} "
                  f"rel={r['rel_effect']:+.1%} z={r['z_vs_null']:+.1f} (baseline={r['baseline']:+.3f})")


if __name__ == '__main__':
    main()
