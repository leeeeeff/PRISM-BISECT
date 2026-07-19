#!/usr/bin/env python3
"""
devils_c4_ridge_reliance.py
C4 CRITICAL 재시도: axis_functional_matrix_reliance.py가 lstsq(no regularization)로
test-time reliance를 degenerate(random-null과 real 불구분)로 판정했으나,
1280-dim design matrix의 ill-conditioning이 문제였을 수 있다.

Ridge-regularized probe(alpha=0.01, 0.1, 1.0)로 재시도 → random-null이 좁아지고
real 축(특히 positive controls: size->axis5, domain_binary->axis3)이 분리되는지 확인.
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
ALPHAS = [0.01, 0.1, 1.0]  # ridge 정규화 강도

AXIS_LABEL = {
    0: 'axis0(disorder)', 1: 'axis1', 2: 'axis2',
    3: 'axis3(domain)', 4: 'axis4', 5: 'axis5(LENGTH)',
    6: 'axis6', 7: 'axis7',
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


def ridge_oof_testtime(Dphi_base, Dphi_occ, y, fold, alpha):
    """Ridge-regularized probe: train on FULL Dphi_base, predict held-out test
    using Dphi_occ. beta = (X'X + alpha*I)^-1 X'y."""
    pred = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr = fold != k
        te = fold == k
        mu = Dphi_base[tr].mean(0)
        sd = Dphi_base[tr].std(0) + 1e-8
        Xtr = (Dphi_base[tr] - mu) / sd
        ytr = y[tr] - y[tr].mean()
        # ridge: beta = (X'X + alpha*I)^-1 X'y
        XtX = Xtr.T @ Xtr
        XtX += alpha * np.eye(XtX.shape[0])
        Xty = Xtr.T @ ytr
        beta = np.linalg.solve(XtX, Xty)
        Xte = (Dphi_occ[te] - mu) / sd
        pred[te] = Xte @ beta
    return pred


def metric(y, pred, ctype):
    return roc_auc_score(y, pred) if ctype == 'binary' else stats.spearmanr(y, pred).correlation


def run_one_alpha(tissue, W, mu, sd, alpha):
    print(f"\n{'='*72}\n{tissue}  alpha={alpha}\n{'='*72}", flush=True)
    L15, L30 = load_raw_layers(tissue)
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    long_idx = df['long_idx'].to_numpy()
    short_idx = df['short_idx'].to_numpy()
    fold = gene_disjoint_folds(df['gene'].to_numpy())

    # 두 positive control만 테스트(시간 절약)
    targets = {
        'size': ('cont', np.log1p(df['size'].to_numpy(float))),
        'domain_binary': ('binary', df['domain_binary'].to_numpy(float)),
        'disorder_frac': ('cont', df['disorder_frac'].to_numpy(float)),
    }

    rng = np.random.default_rng(SEED)
    random_dirs = rng.normal(size=(N_RANDOM, 640))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
    axis_dirs = [W[k] / np.linalg.norm(W[k]) for k in range(N_AXES)]

    Dphi_base = build_dphi(L15, L30, long_idx, short_idx)

    occ_dphi = {}
    for k in range(N_AXES):
        occ_dphi[f'axis{k}'] = build_dphi(occlude(L15, mu[14], sd[14], axis_dirs[k]),
                                          occlude(L30, mu[29], sd[29], axis_dirs[k]),
                                          long_idx, short_idx)
    for i in range(N_RANDOM):
        occ_dphi[f'rand{i}'] = build_dphi(occlude(L15, mu[14], sd[14], random_dirs[i]),
                                          occlude(L30, mu[29], sd[29], random_dirs[i]),
                                          long_idx, short_idx)

    for cname, (ctype, y) in targets.items():
        base = metric(y, ridge_oof_testtime(Dphi_base, Dphi_base, y, fold, alpha), ctype)
        chance = 0.5 if ctype == 'binary' else 0.0
        denom = (base - chance) if ctype == 'binary' else abs(base)

        null_deltas = np.array([base - metric(y, ridge_oof_testtime(Dphi_base, occ_dphi[f'rand{i}'], y, fold, alpha), ctype)
                                for i in range(N_RANDOM)])
        n_lo, n_hi = np.percentile(null_deltas, [2.5, 97.5])
        n_mean, n_sd = null_deltas.mean(), null_deltas.std()

        print(f"  [{cname}] baseline={base:+.4f}  random-null delta: mean={n_mean:+.4f} CI=[{n_lo:+.4f},{n_hi:+.4f}] sd={n_sd:.4f}", flush=True)

        # positive controls: size->axis5, domain_binary->axis3
        if cname == 'size':
            test_axis = 5
        elif cname == 'domain_binary':
            test_axis = 3
        else:
            test_axis = 0

        occ_metric = metric(y, ridge_oof_testtime(Dphi_base, occ_dphi[f'axis{test_axis}'], y, fold, alpha), ctype)
        delta = base - occ_metric
        z = (delta - n_mean) / n_sd if n_sd > 0 else np.nan
        rel = delta / denom if abs(denom) > 1e-9 else np.nan
        outside = not (n_lo <= delta <= n_hi)

        print(f"      {AXIS_LABEL[test_axis]:<20} delta={delta:+.4f} rel={rel:+.1%} z={z:+.2f}  "
              f"{'OUTSIDE null' if outside else 'within null'}", flush=True)


def main():
    W, mu, sd = load_layer_stats()
    for alpha in ALPHAS:
        for tissue in ['muscle', 'brain']:
            run_one_alpha(tissue, W, mu, sd, alpha)


if __name__ == '__main__':
    main()
