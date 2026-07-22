#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""domain_size_r2_artifact_check.py  (Option C artifact-first gate, devils-advocate PROCEED-MINIMAL)

Is the low domain size->|dPRISM| R^2 (0.042 vs non-domain 0.160) a genuine 'size is a poor proxy'
result, or an artifact? Two artifact hypotheses:
  (i)  RANGE RESTRICTION: domain edits cluster at large sizes -> compressed size variance -> low R^2.
  (ii) SATURATION: domain |dPRISM| piles up near its high end -> compressed target variance -> low R^2.
Also test whether a LOG-size fit recovers R^2 (the Panel D relation is log-linear; a linear-on-raw fit
would understate a wide but log-distributed size range). Decision rules per devils-advocate. Read-only.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
BRAIN = ROOT / 'hMuscle/data/brain_isoquant_esm2/full'
MAXLEN = 1022


def cvr2(X, y, grp):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, oof)


def main():
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bsp)
    df = pd.read_csv(SEV / 'brain_severity_pairs_scored.tsv', sep='\t')
    df = df[df['tissue'] == 'brain'].reset_index(drop=True)
    iso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    prism = np.load(ROOT / 'reports/brain_full_672_scores.npy').astype(np.float32)

    G = {'domain': {'s': [], 'm': [], 'g': []}, 'nondomain': {'s': [], 'm': [], 'g': []}}
    for _, r in df.iterrows():
        li, si = int(r['long_idx']), int(r['short_idx'])
        lid, sid = iso[li], iso[si]
        if lid not in seqs or sid not in seqs or seqs[lid][:MAXLEN] == seqs[sid][:MAXLEN]:
            continue
        k = 'domain' if int(r['domain_binary']) == 1 else 'nondomain'
        G[k]['s'].append(float(r['size']))
        G[k]['m'].append(float(np.abs(prism[li] - prism[si]).sum()))
        G[k]['g'].append(str(r['gene']))

    print("=" * 88)
    print("DOMAIN size->output R^2: ARTIFACT CHECK (range restriction / saturation / log-fit)")
    print("=" * 88)
    stats = {}
    for k in ['nondomain', 'domain']:
        s = np.array(G[k]['s']); m = np.array(G[k]['m']); g = np.array(G[k]['g'])
        ls = np.log10(s + 1.0)
        cv_s = s.std() / s.mean(); cv_m = m.std() / m.mean()
        cv_ls = ls.std() / ls.mean()
        # saturation: fraction of pairs within top 10% band of the group's own output range
        top_band = m > (m.max() - 0.1 * (m.max() - m.min()))
        rho = spearmanr(s, m).correlation
        r2_raw = cvr2(s[:, None], m, g)
        r2_log = cvr2(ls[:, None], m, g)
        r2_logboth = cvr2(np.column_stack([s, ls]), m, g)
        stats[k] = dict(cv_s=cv_s, cv_m=cv_m, cv_ls=cv_ls)
        print(f"\n[{k.upper()}]  n={len(s)}")
        print(f"  edit size : mean={s.mean():7.1f}  std={s.std():7.1f}  CV={cv_s:.3f}   "
              f"median={np.median(s):.0f}  p10={np.percentile(s,10):.0f}  p90={np.percentile(s,90):.0f}")
        print(f"  log10size : mean={ls.mean():7.3f}  std={ls.std():7.3f}  CV(log)={cv_ls:.3f}")
        print(f"  |dPRISM|1 : mean={m.mean():7.2f}  std={m.std():7.2f}  CV={cv_m:.3f}   "
              f"max={m.max():.1f}  frac in top-10%-band={top_band.mean()*100:.1f}%")
        print(f"  Spearman(size,|dPRISM|)={rho:.3f}")
        print(f"  R^2  raw-size={r2_raw:+.3f}   log-size={r2_log:+.3f}   raw+log={r2_logboth:+.3f}")
    print("\n" + "-" * 88)
    rs = stats['domain']['cv_s'] / stats['nondomain']['cv_s']
    rm = stats['domain']['cv_m'] / stats['nondomain']['cv_m']
    rls = stats['domain']['cv_ls'] / stats['nondomain']['cv_ls']
    print(f"CV ratio domain/nondomain:  size={rs:.2f}  log-size={rls:.2f}  |dPRISM|={rm:.2f}")
    print("DECISION (devils-advocate rules):")
    print(f"  range restriction if size or log-size CV ratio < 0.5   -> got {rs:.2f} / {rls:.2f}")
    print(f"  saturation        if |dPRISM| CV ratio < 0.5           -> got {rm:.2f}")
    print("  if log-size R^2 recovers toward non-domain -> low raw R^2 was linearity, not 'poor proxy'.")


if __name__ == '__main__':
    main()
