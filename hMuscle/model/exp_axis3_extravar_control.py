#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_axis3_extravar_control.py  (Option A: resolve the pooling-noise caveat of the extra axis3 variance)

exp_axis3_extravar_identity.py found region-pool nearly DOUBLES the within-pair axis3 variance
(var 7.18 mean -> 13.42 region, 1.87x) but that extra variance correlates with NO tested covariate
(function, geometry, biophysics). CAVEAT left unresolved: region-pool averages FEWER residues (the
short edited interval) than mean-pool (whole protein), so part of the variance rise could be pure
central-limit POOLING NOISE, not edited-region signal.

DECISIVE CONTROL: a length-matched SCRAMBLED-region pooling (random contiguous window of the SAME
length as the edited interval) has the IDENTICAL pooling-noise level but NO edited-region content.
  var(y_scram) = pure pooling-noise floor at the edit's window length.
  var(y_region) - var(y_scram) = the edited-region-SPECIFIC excess variance (signal beyond noise).
Gene-cluster bootstrap CI on that difference.
PRE-REGISTERED (S2):
  H_signal: var(y_region) - var(y_scram) > 0, 95% CI excludes 0 -> the extra axis3 variance is
            edited-region-specific SIGNAL, not merely short-window pooling noise; the "dark within-
            gene variance" finding stands and becomes manuscript-integrable.
  H_noise : CI includes 0 (var_region ~= var_scram) -> the variance doubling is (largely) pooling
            noise; the extra-variance finding must be reported as an artifact, NOT integrated.
Pooling-noise signature check: corr(|y_scram|, 1/edit_len) should be POSITIVE if noise dominates
(shorter window -> fewer residues averaged -> larger excursion magnitude).
"""
import os, importlib.util
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
import json, torch

ROOT = Path('/home/welcome1/sw1686/DIFFUSE'); MODEL = ROOT / 'hMuscle/model'
PCA_DIR = ROOT / 'reports/v20b_pca_interp'
OUT = ROOT / 'reports/muscle_labelgap/axis3_extravar_control.json'
N_LAYERS, EMB_DIM, K, MAXLEN, AX = 30, 640, 8, 1022, 3
rng = np.random.default_rng(1111)

# reuse the exact 3-pool trajectory computation (mean/region/scram) + projection from the prior script
spec = importlib.util.spec_from_file_location('trp', MODEL / 'exp_traj_region_projection.py')
trp = importlib.util.module_from_spec(spec); spec.loader.exec_module(trp)
ddr = trp.ddr


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    pairs = ddr.build_domdiff_pairs()
    print(f"domain-differing 2-iso pairs: {len(pairs)}", flush=True)
    mu = np.load(PCA_DIR / 'layer_stats_mu.npy'); sd = np.load(PCA_DIR / 'layer_stats_sd.npy')
    pmean = np.load(PCA_DIR / 'pca_mean_640.npy'); W = np.load(PCA_DIR / 'W_axes_8x640.npy')
    traj = trp.compute_traj(pairs, int(os.environ.get('EMB_GPU', '0')))

    hi_ids = [p['hi_id'] for p in pairs]; lo_ids = [p['lo_id'] for p in pairs]
    gid = np.array([p['g'] for p in pairs])
    z = {}
    for pool in ['mean', 'region', 'scram']:
        Thi = np.stack([traj[i][pool] for i in hi_ids]); Tlo = np.stack([traj[i][pool] for i in lo_ids])
        Zhi = trp.project(Thi, mu, sd, pmean, W).mean(1); Zlo = trp.project(Tlo, mu, sd, pmean, W).mean(1)
        z[pool] = (Zhi[:, AX], Zlo[:, AX])
    flip = np.sign((z['mean'][0] - z['mean'][1]).mean()); flip = flip if flip != 0 else 1.0
    y = {pool: (z[pool][0] - z[pool][1]) * flip for pool in z}

    # edited-region length (hi side; the window length region/scram both pool over)
    edit_len = np.array([max(len([k for a, b in p['hi_ivs'] for k in range(a, b) if k < min(len(p['hi_seq']), MAXLEN)]), 1)
                         for p in pairs], float)

    vmean, vregion, vscram = y['mean'].var(), y['region'].var(), y['scram'].var()
    excess = vregion - vscram
    print(f"\n=== raw within-pair axis3 variance by pooling (n={len(pairs)}) ===", flush=True)
    print(f"  var(y_mean)  = {vmean:.4f}   (whole protein; low pooling noise)", flush=True)
    print(f"  var(y_scram) = {vscram:.4f}   (length-matched random window; pooling-noise FLOOR)", flush=True)
    print(f"  var(y_region)= {vregion:.4f}   (edited region)", flush=True)
    print(f"  region-specific excess = var_region - var_scram = {excess:+.4f}", flush=True)

    # gene-cluster bootstrap on var(y_region)-var(y_scram)
    ug = np.unique(gid); gi_map = {g: np.where(gid == g)[0] for g in ug}
    boot = []
    for _ in range(1000):
        gs = rng.choice(ug, len(ug))
        sel = np.concatenate([gi_map[g] for g in gs])
        boot.append(y['region'][sel].var() - y['scram'][sel].var())
    boot = np.array(boot); ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    # also scram vs mean (is the noise floor itself above whole-protein?)
    boot_sm = []
    for _ in range(1000):
        gs = rng.choice(ug, len(ug)); sel = np.concatenate([gi_map[g] for g in gs])
        boot_sm.append(y['scram'][sel].var() - y['mean'][sel].var())
    ci_sm = [float(np.percentile(boot_sm, 2.5)), float(np.percentile(boot_sm, 97.5))]

    # pooling-noise signature: |y_scram| vs 1/edit_len  (positive => noise scales with 1/window)
    sig_scram = spearman(np.abs(y['scram']), 1.0 / edit_len)
    sig_region = spearman(np.abs(y['region']), 1.0 / edit_len)
    print(f"\n  gene-cluster bootstrap  var_region - var_scram  95%CI [{ci[0]:+.4f},{ci[1]:+.4f}]", flush=True)
    print(f"  gene-cluster bootstrap  var_scram  - var_mean   95%CI [{ci_sm[0]:+.4f},{ci_sm[1]:+.4f}]", flush=True)
    print(f"  pooling-noise signature  rho(|y_scram|, 1/edit_len) = {sig_scram:+.3f}", flush=True)
    print(f"                           rho(|y_region|,1/edit_len) = {sig_region:+.3f}", flush=True)

    verdict = ('H_signal (region excess variance > pooling-noise floor; edited-region-specific)'
               if ci[0] > 0 else
               'H_noise (region variance not distinguishable from length-matched pooling-noise floor)')
    noise_share = float(max(0.0, (vscram - vmean)) / max(vregion - vmean, 1e-9)) if vregion > vmean else float('nan')
    print(f"\n  -> {verdict}", flush=True)
    print(f"  pooling-noise share of the excess = (var_scram-var_mean)/(var_region-var_mean) = {noise_share:.1%}", flush=True)

    res = {'n_pairs': len(pairs), 'var_mean': float(vmean), 'var_scram': float(vscram),
           'var_region': float(vregion), 'region_minus_scram': float(excess),
           'ci_region_minus_scram': ci, 'ci_scram_minus_mean': ci_sm,
           'noise_signature_scram': sig_scram, 'noise_signature_region': sig_region,
           'pooling_noise_share_of_excess': noise_share, 'verdict': verdict}
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[saved] {OUT}", flush=True)


if __name__ == '__main__':
    main()
