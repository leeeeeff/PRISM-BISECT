#!/usr/bin/env python3
"""
c3_gate_variance_vs_reliance.py

C3 GATE (devils-advocate): before using ridge test-time reliance as the B4-usage
instrument, prove it is NOT variance-confounded — the exact failure that killed
coherence-retrain occlusion (axis5 counterexample: low-variance high-encoding axis
got ~1% occlusion effect because occlusion tracked axis variance, not usage).

Reuses the IDENTICAL occlude()/ridge_oof_testtime() reliance definition from
devils_c4_ridge_reliance.py (reliance_k = base_metric - occluded_test_metric).

Two tests, PRE-REGISTERED predictions (HARKing guard):
  T1 Spearman(axis_scalar_variance, reliance) across all 8 real axes, per target.
     PREDICT PASS if rho < 0.5: axis3 (evr .025, "used" for domain) should carry the
     largest domain_binary reliance while axis0/axis5 (evr .156/.014) do NOT track it.
     If reliance were variance-monotonic, axis0 (max evr) would dominate every target.
  T2 variance-matched null: across N random unit directions, measure (captured_var,
     reliance) jointly, fit the reliance-vs-variance trend, and test whether the real
     flagship axis (domain->axis3) sits ABOVE that trend (signal beyond variance).
     PREDICT PASS if real axis3 reliance exceeds the variance-matched random band.

Targets: domain_binary (binary AUC; flagship B4 claim) and disorder_frac (cont; the
axis0 'encoded-not-used' case) for symmetry.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_FOLDS = 5
N_RANDOM = 40
N_AXES = 8
SEED = 42
ALPHA = 0.1

AXIS_LABEL = {0: 'axis0(disorder)', 1: 'axis1', 2: 'axis2', 3: 'axis3(domain)',
              4: 'axis4', 5: 'axis5(LENGTH)', 6: 'axis6', 7: 'axis7'}


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


def captured_var(raw, mu_l, sd_l, direction):
    """Variance removed by occluding `direction` = Var of the axis scalar (z-scored)."""
    z = (raw - mu_l) / sd_l
    return float(np.var(z @ direction))


def gene_disjoint_folds(genes, n_folds=N_FOLDS, seed=SEED):
    uniq = np.array(sorted(set(genes)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of_gene = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of_gene[g] for g in genes])


def build_dphi(L15, L30, long_idx, short_idx):
    return np.concatenate([L15[long_idx] - L15[short_idx],
                           L30[long_idx] - L30[short_idx]], axis=1)


def ridge_oof_testtime(Dphi_base, Dphi_occ, y, fold, alpha):
    pred = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr = fold != k
        te = fold == k
        mu = Dphi_base[tr].mean(0)
        sd = Dphi_base[tr].std(0) + 1e-8
        Xtr = (Dphi_base[tr] - mu) / sd
        ytr = y[tr] - y[tr].mean()
        XtX = Xtr.T @ Xtr
        XtX += alpha * np.eye(XtX.shape[0])
        beta = np.linalg.solve(XtX, Xtr.T @ ytr)
        Xte = (Dphi_occ[te] - mu) / sd
        pred[te] = Xte @ beta
    return pred


def metric(y, pred, ctype):
    return roc_auc_score(y, pred) if ctype == 'binary' else stats.spearmanr(y, pred).correlation


def run(tissue, W, mu, sd):
    print(f"\n{'='*74}\n{tissue}  (alpha={ALPHA})\n{'='*74}", flush=True)
    L15, L30 = load_raw_layers(tissue)
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    long_idx = df['long_idx'].to_numpy()
    short_idx = df['short_idx'].to_numpy()
    fold = gene_disjoint_folds(df['gene'].to_numpy())
    Dphi_base = build_dphi(L15, L30, long_idx, short_idx)

    targets = {
        'domain_binary': ('binary', df['domain_binary'].to_numpy(float)),
        'disorder_frac': ('cont', df['disorder_frac'].to_numpy(float)),
    }

    rng = np.random.default_rng(SEED)
    axis_dirs = [W[k] / np.linalg.norm(W[k]) for k in range(N_AXES)]
    rand_dirs = rng.normal(size=(N_RANDOM, 640))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)

    # captured variance (L30 z-scored) for real axes and random dirs
    axis_var = np.array([captured_var(L30, mu[29], sd[29], axis_dirs[k]) for k in range(N_AXES)])
    rand_var = np.array([captured_var(L30, mu[29], sd[29], rand_dirs[i]) for i in range(N_RANDOM)])

    # precompute occluded Dphi
    occ_axis = {k: build_dphi(occlude(L15, mu[14], sd[14], axis_dirs[k]),
                              occlude(L30, mu[29], sd[29], axis_dirs[k]), long_idx, short_idx)
                for k in range(N_AXES)}
    occ_rand = {i: build_dphi(occlude(L15, mu[14], sd[14], rand_dirs[i]),
                              occlude(L30, mu[29], sd[29], rand_dirs[i]), long_idx, short_idx)
                for i in range(N_RANDOM)}

    print(f"  axis captured-var (L30 z): " + " ".join(f"a{k}={axis_var[k]:.3f}" for k in range(N_AXES)))
    print(f"  random captured-var: mean={rand_var.mean():.3f} range=[{rand_var.min():.3f},{rand_var.max():.3f}]")

    for cname, (ctype, y) in targets.items():
        base = metric(y, ridge_oof_testtime(Dphi_base, Dphi_base, y, fold, ALPHA), ctype)
        chance = 0.5 if ctype == 'binary' else 0.0
        denom = (base - chance) if ctype == 'binary' else abs(base)

        rel_axis = np.array([base - metric(y, ridge_oof_testtime(Dphi_base, occ_axis[k], y, fold, ALPHA), ctype)
                             for k in range(N_AXES)])
        rel_rand = np.array([base - metric(y, ridge_oof_testtime(Dphi_base, occ_rand[i], y, fold, ALPHA), ctype)
                             for i in range(N_RANDOM)])

        print(f"\n  --- target={cname} (base={base:+.4f}) ---")
        for k in range(N_AXES):
            print(f"      {AXIS_LABEL[k]:<20} var={axis_var[k]:6.3f}  reliance={rel_axis[k]:+.4f}  rel%={rel_axis[k]/denom:+.1%}")

        # T1: Spearman(variance, reliance) across 8 real axes
        rho_axes, p_axes = stats.spearmanr(axis_var, rel_axis)
        # also across random directions (is the METRIC variance-confounded?)
        rho_rand, p_rand = stats.spearmanr(rand_var, rel_rand)
        print(f"  [T1] Spearman(var,reliance)  8 real axes: rho={rho_axes:+.3f} p={p_axes:.3f}"
              f"   |  {N_RANDOM} random dirs: rho={rho_rand:+.3f} p={p_rand:.3f}")

        # T2: is the flagship axis above the variance-matched random band?
        flag = 3 if cname == 'domain_binary' else 0
        v = axis_var[flag]
        band = np.abs(rand_var - v) <= 0.25 * v  # random dirs within +-25% of flagship variance
        if band.sum() >= 3:
            matched = rel_rand[band]
            print(f"  [T2] {AXIS_LABEL[flag]} reliance={rel_axis[flag]:+.4f} vs variance-matched "
                  f"random (n={band.sum()}, var~{v:.3f}): mean={matched.mean():+.4f} "
                  f"max={matched.max():+.4f}  ->  {'ABOVE band (signal>variance)' if rel_axis[flag] > matched.max() else 'within band (CONFOUND risk)'}")
        else:
            # fallback: compare to all random dirs with var >= flagship var
            hi = rand_var >= v
            matched = rel_rand[hi] if hi.sum() > 0 else rel_rand
            print(f"  [T2 fallback] {AXIS_LABEL[flag]} reliance={rel_axis[flag]:+.4f} vs random "
                  f"var>={v:.3f} (n={hi.sum()}): mean={matched.mean():+.4f} max={matched.max():+.4f}  ->  "
                  f"{'ABOVE (signal>variance)' if rel_axis[flag] > matched.max() else 'within (CONFOUND risk)'}")


def main():
    W, mu, sd = load_layer_stats()
    for tissue in ['muscle', 'brain']:
        run(tissue, W, mu, sd)
    print(f"\n{'='*74}\nGATE VERDICT\n{'='*74}")
    print("PASS if: T1 rho(real axes) < 0.5 AND flagship axis3 is T2-ABOVE the")
    print("variance-matched random band in both tissues (reliance reflects encoding,")
    print("not axis variance). Then B4 (compositional usage) may proceed.")


if __name__ == '__main__':
    main()
