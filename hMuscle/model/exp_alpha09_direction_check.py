#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_alpha09_direction_check.py  (devils-advocate MODERATE finding, decisive check)

The globallocal alpha-sweep (exp_crux_globallocal_dr.py) found DR-AUC-optimal alpha=0.9
(mean-pool weight 0.9, region-pool weight 0.1), statistically indistinguishable from pure
mean-pool (alpha=1.0) on the per-GO centroid DR-AUC metric. The manuscript sentence inserted
into natcomm_v0.md §4d claims "there is no intermediate recipe that recovers both the region-
pool's direction coherence and the mean-pool's labelling signal at once" -- but this was never
directly verified: does alpha=0.9 still carry the WITHIN-PAIR domain-loss direction-accuracy gain
that pure region-pool (alpha=0.0) has (0.844, vs mean-pool's 0.708; see exp_domdiff_region_pool.py)?

If alpha=0.9 direction-accuracy has collapsed back near mean-pool's 0.708, the manuscript sentence
is SAFE as written (no recipe gets both properties). If alpha=0.9 retains direction-accuracy well
above 0.708 (e.g. >0.78), the sentence is WRONG: alpha=0.9 already partially recovers both axes,
and needs to be revised to something like "the DR-AUC-optimal blend does not recover [X]% of
region-pool's direction coherence" rather than an absolute "no recipe" claim.

PRE-REGISTERED (S2):
  H_safe   : direction-acc(alpha=0.9) < 0.73 (i.e. closer to mean-pool 0.708 than to region-pool
             0.844; recovers <25% of the 0.136 gap) -> manuscript sentence stands as written.
  H_partial: direction-acc(alpha=0.9) >= 0.73 -> sentence overclaims; must be revised to note
             partial simultaneous recovery.
Reuses exp_domdiff_region_pool.py's pair construction + embedding computation + CV-fold direction-
accuracy estimator (30-seed stable), applied to E_alpha = normalize(alpha*mean + (1-alpha)*region)
for alpha in {0.0, 0.5, 0.7, 0.8, 0.9, 1.0} on the SAME 178 domain-differing pairs used both in
exp_domdiff_region_pool.py (0.708/0.844 positive control) and exp_crux_globallocal_dr.py (DR-AUC
sweep), so this bridges the two metrics on identical data.
"""
import os, importlib.util
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from pathlib import Path
import json

ROOT = Path('/home/welcome1/sw1686/DIFFUSE'); MODEL = ROOT / 'hMuscle/model'
OUT = ROOT / 'reports/muscle_labelgap/alpha09_direction_check.json'
ALPHAS = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]

spec = importlib.util.spec_from_file_location('ddr', MODEL / 'exp_domdiff_region_pool.py')
ddr = importlib.util.module_from_spec(spec); spec.loader.exec_module(ddr)


def main():
    pairs = ddr.build_domdiff_pairs()
    print(f"domain-differing pairs: {len(pairs)}", flush=True)
    pools = ddr.compute(pairs, int(os.environ.get('EMB_GPU', '0')))
    gid = np.array([p['g'] for p in pairs])

    def emb(key, pid):
        v = np.concatenate([pools[pid][key + '15'], pools[pid][key + '30']])
        return v / (np.linalg.norm(v) + 1e-9)

    Mhi = np.stack([emb('m', p['hi_id']) for p in pairs]); Mlo = np.stack([emb('m', p['lo_id']) for p in pairs])
    Chi = np.stack([emb('c', p['hi_id']) for p in pairs]); Clo = np.stack([emb('c', p['lo_id']) for p in pairs])

    print("\n=== direction-accuracy (domain-differing, toward domain-complete) across alpha ===", flush=True)
    res = {'n_pairs': len(pairs), 'reference': {'meanpool_acc': None, 'regionpool_acc': None}, 'sweep': {}}
    for a in ALPHAS:
        Ehi = a * Mhi + (1 - a) * Chi; Ehi /= (np.linalg.norm(Ehi, axis=1, keepdims=True) + 1e-9)
        Elo = a * Mlo + (1 - a) * Clo; Elo /= (np.linalg.norm(Elo, axis=1, keepdims=True) + 1e-9)
        D = Ehi - Elo
        mu, sd = ddr.acc_stab(D, gid)
        res['sweep'][a] = {'acc': mu, 'sd': sd}
        print(f"  alpha={a:.1f}  dir-acc={mu:.4f} ± {sd:.4f}", flush=True)

    m_acc = res['sweep'][1.0]['acc']; r_acc = res['sweep'][0.0]['acc']; a09 = res['sweep'][0.9]['acc']
    gap = r_acc - m_acc
    recovered_frac = (a09 - m_acc) / gap if gap != 0 else float('nan')
    res['reference'] = {'meanpool_acc': m_acc, 'regionpool_acc': r_acc, 'gap': gap}
    res['alpha09_recovered_fraction_of_gap'] = recovered_frac
    verdict = 'H_partial (alpha=0.9 recovers >=25% of the direction-coherence gap)' if recovered_frac >= 0.25 else \
              'H_safe (alpha=0.9 recovers <25% of the gap, close to pure mean-pool)'
    print(f"\n  meanpool(a=1.0)={m_acc:.4f}  regionpool(a=0.0)={r_acc:.4f}  gap={gap:+.4f}", flush=True)
    print(f"  alpha=0.9 dir-acc={a09:.4f}  recovered_fraction_of_gap={recovered_frac:.2%}", flush=True)
    print(f"  -> {verdict}", flush=True)
    res['verdict'] = verdict
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[saved] {OUT}", flush=True)


if __name__ == '__main__':
    main()
