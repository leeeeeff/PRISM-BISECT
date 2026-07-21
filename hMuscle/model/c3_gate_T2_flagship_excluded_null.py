#!/usr/bin/env python3
"""
c3_gate_T2_flagship_excluded_null.py

Definitive close of T2. The span-combo null (c3_gate_T2_variance_matched_null.py)
included axis3 in its basis, so the null contained scaled copies of the flagship,
inflating resid_sd and preventing axis3 from standing out in muscle (studentized
+1.24, on-trend). This cannot distinguish (a) null contamination from (b) a genuine
muscle variance->reliance confound.

Fix: build the high-variance null from random combinations of the SEVEN axes EXCLUDING
axis3 (the domain flagship). These are high-variance AND ~orthogonal to axis3 (PCA axes
are approximately orthonormal), so their domain reliance = "domain reliance obtainable
from high-variance directions that are NOT the domain axis." Then:
  - If axis3 reliance >> this clean high-variance null band -> axis3's domain usage is
    variance-independent (muscle span-combo on-trend was null contamination). GATE PASS.
  - If the axis3-excluded high-variance null ALSO yields high domain reliance -> muscle
    genuinely has a variance->domain-reliance confound beyond axis3. Muscle B4 suspect.

Target domain_binary, both tissues. Same occlude()/ridge reliance definition.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_FOLDS, N_COMBO, N_AXES, SEED, ALPHA = 5, 80, 8, 42, 0.1
FLAGSHIP = 3  # axis3 = domain


def load_layer_stats():
    return (np.load(ROOT / 'reports/v20b_pca_interp/W_axes_8x640.npy'),
            np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_mu.npy'),
            np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_sd.npy'))


def load_raw_layers(tissue):
    if tissue == 'muscle':
        return (np.load(ROOT / 'hMuscle/data/esm2_layer_15_t30_150M.npy').astype(np.float32),
                np.load(ROOT / 'hMuscle/data/esm2_layer_30_t30_150M.npy').astype(np.float32))
    return (np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer15_t30_150M.npy').astype(np.float32),
            np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer30_t30_150M.npy').astype(np.float32))


def occlude(raw, mu_l, sd_l, d):
    z = (raw - mu_l) / sd_l
    return (z - np.outer(z @ d, d)) * sd_l + mu_l


def captured_var(raw, mu_l, sd_l, d):
    return float(np.var(((raw - mu_l) / sd_l) @ d))


def folds(genes):
    uniq = np.array(sorted(set(genes)))
    np.random.default_rng(SEED).shuffle(uniq)
    fog = {g: i % N_FOLDS for i, g in enumerate(uniq)}
    return np.array([fog[g] for g in genes])


def dphi(L15, L30, li, si):
    return np.concatenate([L15[li] - L15[si], L30[li] - L30[si]], axis=1)


def ridge(Xb, Xo, y, fold):
    pred = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        mu, sd = Xb[tr].mean(0), Xb[tr].std(0) + 1e-8
        Xtr = (Xb[tr] - mu) / sd
        beta = np.linalg.solve(Xtr.T @ Xtr + ALPHA * np.eye(Xb.shape[1]), Xtr.T @ (y[tr] - y[tr].mean()))
        pred[te] = ((Xo[te] - mu) / sd) @ beta
    return pred


def reliance(Xb, L15, L30, mu, sd, d, li, si, y, fold, base):
    occ = dphi(occlude(L15, mu[14], sd[14], d), occlude(L30, mu[29], sd[29], d), li, si)
    return base - roc_auc_score(y, ridge(Xb, occ, y, fold))


def run(tissue, W, mu, sd):
    print(f"\n{'='*74}\n{tissue}  domain_binary  (axis{FLAGSHIP}-EXCLUDED high-variance null)\n{'='*74}", flush=True)
    L15, L30 = load_raw_layers(tissue)
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    li, si = df['long_idx'].to_numpy(), df['short_idx'].to_numpy()
    y = df['domain_binary'].to_numpy(float)
    fold = folds(df['gene'].to_numpy())
    Xb = dphi(L15, L30, li, si)
    base = roc_auc_score(y, ridge(Xb, Xb, y, fold))

    axis_dirs = [W[k] / np.linalg.norm(W[k]) for k in range(N_AXES)]
    keep = [k for k in range(N_AXES) if k != FLAGSHIP]

    rng = np.random.default_rng(SEED)
    C = rng.normal(size=(N_COMBO, len(keep)))
    null_dirs = C @ np.array([axis_dirs[k] for k in keep])   # span of the 7 non-flagship axes
    null_dirs /= np.linalg.norm(null_dirs, axis=1, keepdims=True)

    null_var = np.array([captured_var(L30, mu[29], sd[29], null_dirs[i]) for i in range(N_COMBO)])
    null_rel = np.array([reliance(Xb, L15, L30, mu, sd, null_dirs[i], li, si, y, fold, base) for i in range(N_COMBO)])

    a3_var = captured_var(L30, mu[29], sd[29], axis_dirs[FLAGSHIP])
    a3_rel = reliance(Xb, L15, L30, mu, sd, axis_dirs[FLAGSHIP], li, si, y, fold, base)

    print(f"  base AUC={base:+.4f}", flush=True)
    print(f"  axis{FLAGSHIP}-excluded null: captured-var mean={null_var.mean():.2f} range=[{null_var.min():.2f},{null_var.max():.2f}] "
          f"(matches axis3 var={a3_var:.2f})", flush=True)
    print(f"  null domain-reliance: mean={null_rel.mean():+.4f} sd={null_rel.std():.4f} "
          f"p95={np.percentile(null_rel,95):+.4f} max={null_rel.max():+.4f}", flush=True)

    # variance-conditioned comparison: null dirs within +-30% of axis3 variance
    band = np.abs(null_var - a3_var) <= 0.30 * a3_var
    emp_p = float((null_rel >= a3_rel).mean())
    print(f"\n  axis{FLAGSHIP}(domain) reliance={a3_rel:+.4f}  (var={a3_var:.2f})")
    print(f"    vs full null: empirical p(null>=axis3) = {emp_p:.3f}  "
          f"({'ABOVE null (variance-independent usage)' if emp_p < 0.05 else 'WITHIN null (variance-confounded)'})")
    if band.sum() >= 3:
        m = null_rel[band]
        z = (a3_rel - m.mean()) / (m.std() + 1e-9)
        print(f"    vs variance-matched null (n={band.sum()}, var~{a3_var:.1f}): mean={m.mean():+.4f} max={m.max():+.4f} "
              f"z={z:+.2f}  {'ABOVE' if a3_rel > m.max() else 'WITHIN'}")
    return emp_p, a3_rel, null_rel


def main():
    W, mu, sd = load_layer_stats()
    res = {}
    for t in ['muscle', 'brain']:
        res[t] = run(t, W, mu, sd)
    print(f"\n{'='*74}\nDEFINITIVE T2 VERDICT (axis3-excluded null)\n{'='*74}")
    for t in ['muscle', 'brain']:
        p, a3, _ = res[t]
        print(f"  {t}: axis3 reliance={a3:+.4f}, empirical p vs clean high-var null={p:.3f} -> "
              f"{'variance-INDEPENDENT (PASS)' if p < 0.05 else 'variance-confounded (FAIL)'}")
    print("\n  GATE FULLY PASSES iff axis3 is variance-independent in BOTH tissues against the")
    print("  clean (flagship-excluded) high-variance null. Muscle result here settles whether the")
    print("  span-combo on-trend was null contamination (-> PASS) or a real muscle confound (-> FAIL).")


if __name__ == '__main__':
    main()
