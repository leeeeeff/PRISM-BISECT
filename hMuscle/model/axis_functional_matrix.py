#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
axis_functional_matrix.py
===========================
Broadens the axis FUNCTIONAL-usage evaluation beyond domain-completeness.

Motivation (user, 2026-07-20): the established "functional axis" characterisation
(reference-esm2-pca-axes-final.md occlusion table) is anchored ONLY to DR-AUC
(ground truth = Pfam domain count) + macro-AUPRC (GO). The isoform-level
functional target is therefore domain-completeness alone. This script treats
EACH of the 5 severity covariates as its own downstream functional target
(the DR-AUC generalisation) and measures every joint-PCA axis's causal
contribution to predicting it -> an 8-axis x 5-covariate functional-usage
matrix, the functional counterpart to axis_covariate_partial_corr.py's
encoding matrix.

Design:
  target: predict covariate C from Dphi = phi(long)-phi(short), L15||L30 (1280d),
          gene-disjoint 5-fold, out-of-fold. binary C -> AUROC; continuous C
          (size=log1p, disorder) -> Spearman.
  causal contribution: occlusion-RETRAIN. Occlude axis_k from phi (per-layer
          z-space projection), rebuild Dphi, refit probe 5-fold, out-of-fold
          metric. delta = baseline - occluded (positive => axis non-redundantly
          needed to predict C).
  null: N_RANDOM random unit directions, same occlude+retrain, delta distribution.
  reporting: z vs random-null AND relative effect = delta/(baseline-chance)
          (chance=0.5 AUROC, |baseline| for Spearman). NEITHER alone decides
          (cf. approach-axis-encoding-vs-usage-occlusion.md self-correction).
  NOTE: size is deliberately NOT residualised, so a covariate whose
          decodability rides on the length axis will surface as axis5-dependence
          -- an internal cross-check against the size-matched discrimination
          finding (muscle domain-tracking was size-artefact).

PRE-REGISTERED (S2): positive controls size->axis5, domain_binary->axis3 must be
standouts or the method is broken (novel rows untrusted). disorder->axis0 expected.
nterm_overlap / resync_failure_binary: genuinely open (may be all encoded-only).
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
CONT_COVS = ['size', 'disorder_frac']  # size -> log1p, positive control (length)

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


def oof_linear_predict(X, y, fold):
    """Out-of-fold linear (ridge-ish via lstsq) predictions, features standardised
    per fold on train stats."""
    pred = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr = fold != k
        te = fold == k
        mu = X[tr].mean(0)
        sd = X[tr].std(0) + 1e-8
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        Xtr1 = np.column_stack([np.ones(Xtr.shape[0]), Xtr])
        Xte1 = np.column_stack([np.ones(Xte.shape[0]), Xte])
        beta, *_ = np.linalg.lstsq(Xtr1, y[tr] - y[tr].mean(), rcond=None)
        pred[te] = Xte1 @ beta
    return pred


def metric_binary(y, pred):
    return roc_auc_score(y, pred)


def metric_cont(y, pred):
    return stats.spearmanr(y, pred).correlation


def build_dphi(L15, L30, long_idx, short_idx):
    return np.concatenate([L15[long_idx] - L15[short_idx],
                           L30[long_idx] - L30[short_idx]], axis=1)


def run_tissue(tissue, W, mu, sd):
    print(f"\n{'='*72}\n{tissue}\n{'='*72}", flush=True)
    L15, L30 = load_raw_layers(tissue)
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    long_idx = df['long_idx'].to_numpy()
    short_idx = df['short_idx'].to_numpy()
    fold = gene_disjoint_folds(df['gene'].to_numpy())

    targets = {}
    for c in BINARY_COVS:
        targets[c] = ('binary', df[c].to_numpy(dtype=np.float64))
    targets['size'] = ('cont', np.log1p(df['size'].to_numpy(dtype=np.float64)))
    targets['disorder_frac'] = ('cont', df['disorder_frac'].to_numpy(dtype=np.float64))

    # precompute occluded L15/L30 for each axis + random dirs (reused across covariates)
    rng = np.random.default_rng(SEED)
    random_dirs = rng.normal(size=(N_RANDOM, 640))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)

    axis_dirs = [W[k] / np.linalg.norm(W[k]) for k in range(N_AXES)]

    # cache occluded embeddings once (axis + random)
    occ_cache = {}  # key -> (L15_occ, L30_occ)
    for k in range(N_AXES):
        occ_cache[f'axis{k}'] = (occlude(L15, mu[14], sd[14], axis_dirs[k]),
                                 occlude(L30, mu[29], sd[29], axis_dirs[k]))
    for i in range(N_RANDOM):
        occ_cache[f'rand{i}'] = (occlude(L15, mu[14], sd[14], random_dirs[i]),
                                 occlude(L30, mu[29], sd[29], random_dirs[i]))

    Dphi_base = build_dphi(L15, L30, long_idx, short_idx)

    rows = []
    for cname, (ctype, y) in targets.items():
        met = metric_binary if ctype == 'binary' else metric_cont
        pred_base = oof_linear_predict(Dphi_base, y, fold)
        base = met(y, pred_base)
        chance = 0.5 if ctype == 'binary' else 0.0
        denom = (base - chance) if ctype == 'binary' else abs(base)

        # random-null deltas
        null_deltas = np.empty(N_RANDOM)
        for i in range(N_RANDOM):
            l15o, l30o = occ_cache[f'rand{i}']
            D = build_dphi(l15o, l30o, long_idx, short_idx)
            null_deltas[i] = base - met(y, oof_linear_predict(D, y, fold))
        n_lo, n_hi = np.percentile(null_deltas, [2.5, 97.5])
        n_mean, n_sd = null_deltas.mean(), null_deltas.std()

        for k in range(N_AXES):
            l15o, l30o = occ_cache[f'axis{k}']
            D = build_dphi(l15o, l30o, long_idx, short_idx)
            occ_metric = met(y, oof_linear_predict(D, y, fold))
            delta = base - occ_metric
            z = (delta - n_mean) / n_sd if n_sd > 0 else np.nan
            rel = delta / denom if abs(denom) > 1e-9 else np.nan
            outside = not (n_lo <= delta <= n_hi)
            used = outside and (abs(rel) >= 0.10)
            rows.append({
                'tissue': tissue, 'covariate': cname, 'ctype': ctype,
                'axis': k, 'axis_label': AXIS_LABEL[k],
                'baseline': base, 'occluded': occ_metric, 'delta': delta,
                'rel_effect': rel, 'z_vs_null': z, 'used': used,
            })
        print(f"  [{cname}] baseline {met.__name__.replace('metric_','')}={base:+.4f} "
              f"(random-null delta CI=[{n_lo:+.4f},{n_hi:+.4f}])", flush=True)
        for k in range(N_AXES):
            r = rows[-N_AXES + k]
            flag = '  <== USED' if r['used'] else ''
            print(f"      {r['axis_label']:<20} delta={r['delta']:+.4f} rel={r['rel_effect']:+.1%} "
                  f"z={r['z_vs_null']:+.1f}{flag}", flush=True)
    return rows


def main():
    W, mu, sd = load_layer_stats()
    all_rows = []
    for tissue in ['muscle', 'brain']:
        all_rows += run_tissue(tissue, W, mu, sd)
    out = pd.DataFrame(all_rows)
    out_path = ROOT / 'reports/severity_pairs/axis_functional_matrix.tsv'
    out.to_csv(out_path, sep='\t', index=False)
    print(f"\n[Save] {out_path}")

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
