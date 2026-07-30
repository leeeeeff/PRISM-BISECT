#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_internal_describability.py  (Option C: does the recovered INTERNAL direction align with any
ESM-2-independent functional axis? — trying to close the describability gap for internal edits)

Option A showed the N-terminal recovered direction aligns with a mito-presequence (targeting)
proxy. Internal edits recovered coherence under changed-region pool (0.739 clean) but their
functional meaning is untested. Here we test whether the internal changed-region anchor projection
aligns (partialling out edited-region length) with sequence-computable proxies of what INTERNAL
alternative splicing typically toggles: intrinsic disorder (regulatory/PTM-bearing linkers),
net charge, and hydrophobicity of the EDITED region.
  H_gap (null): no proxy aligns (internal residual stays label/description-undescribable).
  H_describe: internal direction aligns with disorder/charge (partial rho>0, p<0.05) => internal
    residual gains a partial functional handle (regulatory-linker axis), analogous to N-term MTS.
Uses the region_pool_cache (changed-region pooled embeddings) + muscle_2iso sequences.
Restricted to the CLEAN internal subset (both sides' edited region non-empty; the honest 0.739 set).
"""
import os, re
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from pathlib import Path
import json, importlib.util
from scipy.stats import spearmanr, rankdata, t as tdist

ROOT = Path('/home/welcome1/sw1686/DIFFUSE'); MODEL = ROOT / 'hMuscle/model'
CACHE = ROOT / 'reports/muscle_labelgap/region_pool_cache.npz'
OUT = ROOT / 'reports/muscle_labelgap/internal_describability.json'
spec = importlib.util.spec_from_file_location('rp', MODEL / 'exp_nondomain_region_pool.py')
rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)
rng = np.random.default_rng(77)

# TOP-IDP disorder propensity (Campen 2008); disorder-promoting positive
TOPIDP = {'A':0.06,'R':0.180,'N':0.007,'D':0.192,'C':0.02,'Q':0.318,'E':0.736,'G':0.166,
          'H':0.303,'I':-0.486,'L':-0.326,'K':0.586,'M':-0.397,'F':-0.697,'P':0.987,
          'S':0.341,'T':0.059,'W':-0.884,'Y':-0.510,'V':-0.121}
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
      'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}


def region_seq(seq, ivs, maxlen=1022):
    idx = [k for a, b in ivs for k in range(a, b) if k < min(len(seq), maxlen)]
    return ''.join(seq[k] for k in idx)


def disorder(s): return float(np.mean([TOPIDP.get(a, 0) for a in s])) if s else 0.0
def netchg(s):   return (s.count('R') + s.count('K') - s.count('D') - s.count('E')) / len(s) if s else 0.0
def hydro(s):    return float(np.mean([KD.get(a, 0) for a in s])) if s else 0.0


def partial_spearman(x, y, covars):
    def resid(v):
        r = rankdata(v).astype(float)
        C = np.column_stack([rankdata(c).astype(float) for c in covars] + [np.ones_like(r)])
        beta, *_ = np.linalg.lstsq(C, r, rcond=None); return r - C @ beta
    rx, ry = resid(x), resid(y); rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x); dof = max(n - len(covars) - 2, 1)
    ts = rho * np.sqrt(dof / max(1 - rho * rho, 1e-12))
    return rho, float(2 * tdist.sf(abs(ts), dof))


def cv_proj(D, gid):
    ug = np.unique(gid); rng.shuffle(ug); f = {g: i % 5 for i, g in enumerate(ug)}
    fid = np.array([f[g] for g in gid]); proj = np.zeros(len(D))
    for k in range(5):
        te = fid == k; tr = ~te
        if tr.sum() < 5 or te.sum() == 0: continue
        a = D[tr].mean(0); a /= np.linalg.norm(a) + 1e-9; proj[te] = D[te] @ a
    return proj


def main():
    pairs = rp.build_pairs()
    pools = np.load(CACHE, allow_pickle=True)['pools'].item()

    def empty(ivs): return len([i for a, b in ivs for i in range(a, b)]) == 0
    rows = []
    for p in pairs:
        if not (p['first'] < rp.NTERM_WIN):     # INTERNAL only
            if empty(p['lo_ivs']) or empty(p['sh_ivs']):  # CLEAN only
                continue
            lo, sh = pools[p['lo_id']], pools[p['sh_id']]
            D = np.concatenate([lo['c15'] - sh['c15'], lo['c30'] - sh['c30']])  # changed-region delta (long-short)
            lr = region_seq(p['lo_seq'], p['lo_ivs']); sr = region_seq(p['sh_seq'], p['sh_ivs'])
            rows.append({'g': p['g'], 'D': D,
                         'dDis': disorder(lr) - disorder(sr),
                         'dChg': netchg(lr) - netchg(sr),
                         'dHyd': hydro(lr) - hydro(sr),
                         'dLen': len(lr) - len(sr)})
    D = np.array([r['D'] for r in rows]); gid = np.array([r['g'] for r in rows])
    dDis = np.array([r['dDis'] for r in rows]); dChg = np.array([r['dChg'] for r in rows])
    dHyd = np.array([r['dHyd'] for r in rows]); dLen = np.array([r['dLen'] for r in rows])
    proj = cv_proj(D, gid)
    print(f"clean internal pairs: {len(rows)}", flush=True)
    res = {'n': len(rows), 'subset': 'clean internal (both edited regions non-empty)',
           'prediction': 'H_gap: no proxy aligns (internal stays undescribable)'}
    for name, arr in [('disorder', dDis), ('net_charge', dChg), ('hydrophobicity', dHyd)]:
        raw_r, raw_p = spearmanr(proj, arr)
        par_r, par_p = partial_spearman(proj, arr, [dLen])
        res[name] = {'raw': {'rho': float(raw_r), 'p': float(raw_p)},
                     'partial_len': {'rho': float(par_r), 'p': float(par_p)}}
        print(f"  proj~Δ{name:14} raw ρ={raw_r:+.3f} p={raw_p:.2e} | partial|len ρ={par_r:+.3f} p={par_p:.2e}", flush=True)
    # placebo: proj vs edited-region length
    lr, lp = spearmanr(proj, dLen)
    res['proj_vs_dLen'] = {'rho': float(lr), 'p': float(lp)}
    print(f"  proj~ΔLen(edited)  ρ={lr:+.3f} p={lp:.2e}  (generic axis)", flush=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[saved] {OUT}", flush=True)


if __name__ == '__main__':
    main()
