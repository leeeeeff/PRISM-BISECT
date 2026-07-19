#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
devils_c2_decisive.py
======================
Decisive test of C2 (devils-advocate): is axis0's causal effect on disorder_frac
tracking (axis_occlusion_usage.py, 40-54% drop) real usage, or just an artifact
of axis0 being the highest-variance axis (evr .156) so its removal perturbs the
re-derived coherence direction more than a low-variance random direction?

DECISIVE DESIGN: occlude ALL 8 PCA axes against disorder_frac (same coherence-
retrain occlusion), then ask whether the per-axis rel_effect ranks by
  (H_variance) each axis's variance-explained, OR
  (H_content)  each axis's disorder ENCODING strength (|partial rho| from
               axis_covariate_partial_corr.tsv).
KEY DISCRIMINATOR: axis5 is the LENGTH axis -- LOW variance (evr .014, ~1/11 of
axis0) but HIGH disorder encoding (partial rho ~ -0.16). If axis5 occlusion
causes a large disorder-corr drop DESPITE its low variance, the effect is
content-driven (C2 defeated). If axis5's drop is small (tracking its low
variance), the effect is variance-driven (C2 confirmed).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_FOLDS = 5
SEED = 42


def load_layer_stats():
    W = np.load(ROOT / 'reports/v20b_pca_interp/W_axes_8x640.npy')
    mu = np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_mu.npy')
    sd = np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_sd.npy')
    return W, mu, sd


def load_raw(tissue, layer):
    if tissue == 'muscle':
        return np.load(ROOT / f'hMuscle/data/esm2_layer_{layer:02d}_t30_150M.npy').astype(np.float32)
    return np.load(ROOT / f'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer{layer:02d}_t30_150M.npy').astype(np.float32)


def occlude(raw, mu_l, sd_l, d):
    z = (raw - mu_l) / sd_l
    return (z - np.outer(z @ d, d)) * sd_l + mu_l


def var_explained_by(raw, mu_l, sd_l, d):
    """Fraction of total z-scored variance along unit direction d."""
    z = (raw - mu_l) / sd_l
    proj = z @ d
    return proj.var() / (z.var(axis=0).sum())


def folds(genes):
    uniq = np.array(sorted(set(genes)))
    rng = np.random.default_rng(SEED); rng.shuffle(uniq)
    fmap = {g: i % N_FOLDS for i, g in enumerate(uniq)}
    return np.array([fmap[g] for g in genes])


def coherence(emb, li, si, fold):
    D = emb[li] - emb[si]
    s = np.zeros(len(D))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        dirn = D[tr].mean(0); n = np.linalg.norm(dirn)
        dirn = dirn / n if n > 0 else dirn
        s[te] = D[te] @ dirn
    return s


def resid(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def pcorr(s, c, sz):
    return np.corrcoef(resid(s, sz), resid(c, sz))[0, 1]


def run(tissue, W, mu, sd):
    L15, L30 = load_raw(tissue, 15), load_raw(tissue, 30)
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    li, si = df['long_idx'].to_numpy(), df['short_idx'].to_numpy()
    fold = folds(df['gene'].to_numpy())
    sz = ((np.log1p(df['size']) - np.log1p(df['size']).mean()) / np.log1p(df['size']).std()).to_numpy()
    cov = df['disorder_frac'].to_numpy(float)

    base = pcorr(coherence(np.concatenate([L15, L30], 1), li, si, fold), cov, sz)

    enc = pd.read_csv(ROOT / 'reports/severity_pairs/axis_covariate_partial_corr.tsv', sep='\t')
    enc = enc[(enc.tissue == tissue) & (enc.covariate == 'disorder_frac')].set_index('axis')['partial_rho']

    print(f"\n=== {tissue} (baseline corr(severity,disorder|size)={base:+.4f}) ===")
    print(f"{'axis':<6}{'rel_effect':>12}{'var_expl':>12}{'disorder_enc|rho|':>18}")
    rows = []
    for k in range(8):
        d = W[k] / np.linalg.norm(W[k])
        ve = 0.5 * (var_explained_by(L15, mu[14], sd[14], d) + var_explained_by(L30, mu[29], sd[29], d))
        occ = np.concatenate([occlude(L15, mu[14], sd[14], d), occlude(L30, mu[29], sd[29], d)], 1)
        c_occ = pcorr(coherence(occ, li, si, fold), cov, sz)
        rel = (c_occ - base) / abs(base) if abs(base) > 1e-9 else np.nan  # signed shrinkage
        rel_shrink = abs(base - c_occ) / abs(base)
        rows.append((k, rel_shrink, ve, abs(enc.get(k, np.nan))))
        print(f"{k:<6}{rel_shrink:>11.1%}{ve:>12.4f}{abs(enc.get(k, np.nan)):>18.3f}")

    r = pd.DataFrame(rows, columns=['axis', 'rel_shrink', 'var_expl', 'enc'])
    rho_var = stats.spearmanr(r.rel_shrink, r.var_expl).correlation
    rho_enc = stats.spearmanr(r.rel_shrink, r.enc).correlation
    print(f"  Spearman(rel_shrink, variance)  = {rho_var:+.3f}")
    print(f"  Spearman(rel_shrink, encoding)  = {rho_enc:+.3f}")
    ax5 = r[r.axis == 5].iloc[0]
    ax0 = r[r.axis == 0].iloc[0]
    print(f"  DISCRIMINATOR axis5(LENGTH): var_expl={ax5.var_expl:.4f}(low) enc={ax5.enc:.3f}(high) "
          f"rel_shrink={ax5.rel_shrink:.1%}")
    print(f"                axis0        : var_expl={ax0.var_expl:.4f}(high) enc={ax0.enc:.3f}(high) "
          f"rel_shrink={ax0.rel_shrink:.1%}")
    if rho_enc > rho_var and ax5.rel_shrink > 0.15:
        print("  => H_content favored (rel tracks encoding, low-var axis5 still large) -> C2 DEFEATED")
    elif rho_var > rho_enc and ax5.rel_shrink < 0.10:
        print("  => H_variance favored (rel tracks variance, low-var axis5 small) -> C2 CONFIRMED")
    else:
        print("  => mixed/ambiguous")


def main():
    W, mu, sd = load_layer_stats()
    for t in ['muscle', 'brain']:
        run(t, W, mu, sd)


if __name__ == '__main__':
    main()
