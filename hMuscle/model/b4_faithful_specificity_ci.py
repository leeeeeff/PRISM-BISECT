#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""b4_faithful_specificity_ci.py  — Option A tighten: gene-bootstrap CI on the SPECIFICITY metric.

b4_faithful_nterm_usage.py used the non-specific metric (incr axis3|size, large for everyone incl
non-domain) for its USED/NOT-used labels — a flaw. The true domain-usage discriminator is the
SPECIFICITY metric:  incr = R^2(size + all 8 axes) - R^2(size + other-7 axes)  = does axis3 add output
signal BEYOND size and the other 7 axes. Point estimates were domain~0.025-0.03 vs true-non-domain
~0.001. This adds gene-bootstrap 95% CI (B=500) on that metric, and a paired between-group bootstrap
so we can state whether faithful∩N-terminal (the AD-switch class) is significantly separated from true
non-domain (i.e. resolvable tier) and indistinguishable from faithful∩internal.

Decisive subsets only: faithful=0 (true non-domain), faithful∩internal (nterm=0),
faithful∩N-terminal (nterm=1). Gene-disjoint ridge CV. Brain. Read-only.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
BRAIN = ROOT / 'hMuscle/data/brain_isoquant_esm2/full'
INTERP = ROOT / 'reports/v20b_pca_interp'
MAXLEN = 1022
SEED = 42
B = 500


def cvr2(X, y, grp):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, oof)


def specificity(size, ax, y, grp):
    """incr axis3 beyond size+other-7 = R^2(size+all8) - R^2(size+other7)."""
    other7 = ax[:, [0, 1, 2, 4, 5, 6, 7]]
    return cvr2(np.hstack([size, ax]), y, grp) - cvr2(np.hstack([size, other7]), y, grp)


def build():
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bsp)
    df = pd.read_csv(SEV / 'brain_severity_pairs_scored.tsv', sep='\t')
    df = df[df['tissue'] == 'brain'].reset_index(drop=True)
    iso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    prism = np.load(ROOT / 'reports/brain_full_672_scores.npy').astype(np.float32)
    Z = np.load(INTERP / 'Z_brain_Nx30x8.npy')
    recs = []
    for _, r in df.iterrows():
        li, si = int(r['long_idx']), int(r['short_idx'])
        lid, sid = iso[li], iso[si]
        if lid not in seqs or sid not in seqs or seqs[lid][:MAXLEN] == seqs[sid][:MAXLEN]:
            continue
        disp = np.linalg.norm(Z[li] - Z[si], axis=0)
        recs.append({'y': float(np.abs(prism[li] - prism[si]).sum()), 'size': float(r['size']),
                     'ax': disp, 'g': str(r['gene']),
                     'faithful': int(r['domain_binary_faithful']), 'nterm': int(r['nterm_overlap'])})
    R = pd.DataFrame(recs)
    return R[R['faithful'] >= 0].reset_index(drop=True)


def arrays(s):
    return (s['size'].to_numpy()[:, None], np.stack(s['ax'].to_numpy()),
            s['y'].to_numpy(), s['g'].to_numpy())


def boot_spec(size, ax, y, grp, seedshift=0):
    genes = np.unique(grp)
    g2i = {g: np.where(grp == g)[0] for g in genes}
    rng = np.random.default_rng(SEED + seedshift)
    vals = []
    for _ in range(B):
        samp = rng.choice(genes, size=len(genes), replace=True)
        idx = np.concatenate([g2i[g] for g in samp])
        gg = np.concatenate([np.full(len(g2i[g]), k) for k, g in enumerate(samp)])
        try:
            vals.append(specificity(size[idx], ax[idx], y[idx], gg))
        except Exception:
            continue
    return np.array(vals)


def main():
    R = build()
    groups = {
        'TRUE non-domain (faithful=0)': R['faithful'] == 0,
        'domain ∩ INTERNAL (nterm=0)': (R['faithful'] == 1) & (R['nterm'] == 0),
        'domain ∩ N-TERMINAL (nterm=1) [AD class]': (R['faithful'] == 1) & (R['nterm'] == 1),
    }
    print("=" * 90)
    print(f"SPECIFICITY metric (axis3 beyond size+other-7) with gene-bootstrap 95pct CI (B={B})")
    print("resolvable/domain-used <=> specificity CI lower bound > 0 AND separated from non-domain")
    print("=" * 90)
    dist = {}
    for label, mask in groups.items():
        size, ax, y, grp = arrays(R[mask])
        pt = specificity(size, ax, y, grp)
        bs = boot_spec(size, ax, y, grp)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        dist[label] = bs
        print(f"\n[{label}]  n={mask.sum()}, genes={len(np.unique(grp))}")
        print(f"    specificity = {pt:+.4f}   95%CI[{lo:+.4f}, {hi:+.4f}]   "
              f"{'>0 (axis3-USED)' if lo > 0 else 'CI incl 0 (not sig)'}")

    # between-group separation (paired on bootstrap index): AD-class vs true non-domain, and vs internal
    def sep(a, b):
        n = min(len(dist[a]), len(dist[b]))
        d = dist[a][:n] - dist[b][:n]
        lo, hi = np.percentile(d, [2.5, 97.5])
        return d.mean(), lo, hi
    print("\n" + "-" * 90)
    ad = 'domain ∩ N-TERMINAL (nterm=1) [AD class]'
    nd = 'TRUE non-domain (faithful=0)'
    it = 'domain ∩ INTERNAL (nterm=0)'
    for a, b, tag in [(ad, nd, 'AD-class − non-domain (want >0: AD is domain-like)'),
                      (ad, it, 'AD-class − internal   (want ~0: AD ≈ internal domain)')]:
        m, lo, hi = sep(a, b)
        verdict = ('SEPARATED (>0)' if lo > 0 else ('INDISTINGUISHABLE (CI incl 0)' if hi > 0 > lo else 'BELOW (<0)'))
        print(f"  Δ {tag}\n      = {m:+.4f}  95%CI[{lo:+.4f}, {hi:+.4f}]  -> {verdict}")


if __name__ == '__main__':
    main()
