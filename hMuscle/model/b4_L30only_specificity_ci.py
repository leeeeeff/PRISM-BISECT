#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""b4_L30only_specificity_ci.py — Option A DECISIVE test: L30-only axis3 usage (devils Attack 1).

Attack 1 (FATAL): b4_faithful_specificity_ci.py measured axis3 usage as 30-LAYER trajectory
displacement ‖ΔZ₃‖ over L1–L30. But PRISM ingests ONLY φ_L30 (single pooled vector); it never sees
layers 1–29. So trajectory-based "usage" is circular re-description, not what PRISM's output uses.

This re-runs the IDENTICAL specificity metric but with L30-ONLY axis features:
  feature_k = |Z[li,29,k] - Z[si,29,k]|   (axis-k component of the L30 vector PRISM actually ingests;
                                            layer index 29 = L30, confirmed via L=arange(1,31)).
specificity = R^2(size + all-8-axes@L30) - R^2(size + other-7-axes@L30) = does axis3@L30 add output
signal BEYOND size and the other 7 axes AT THE LAYER PRISM READS.

PRE-REGISTERED decision rule (set before looking):
  * L30-only axis3 specificity CI includes 0, OR indistinguishable from true non-domain
      => PRISM's actual input carries NO domain-specific axis3 signal => NOT resolvable via map
      => drop unified narrative, keep parallel findings.
  * L30-only CI > 0 AND separated from non-domain
      => survives Attack 1 => proceed to external validation (Attack 2).

Gene-disjoint ridge CV(5), gene-bootstrap B=500. Brain. Read-only.
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
L30_IDX = 29  # Z is (N,30,8) for L1..L30 => index 29 = L30 (the layer PRISM ingests)


def cvr2(X, y, grp):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, oof)


def specificity(size, ax, y, grp):
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
        disp = np.abs(Z[li, L30_IDX, :] - Z[si, L30_IDX, :])      # L30-ONLY axis component diff (8,)
        recs.append({'y': float(np.abs(prism[li] - prism[si]).sum()), 'size': float(r['size']),
                     'ax': disp, 'g': str(r['gene']),
                     'faithful': int(r['domain_binary_faithful']), 'nterm': int(r['nterm_overlap'])})
    R = pd.DataFrame(recs)
    return R[R['faithful'] >= 0].reset_index(drop=True)


def arrays(s):
    return (s['size'].to_numpy()[:, None], np.stack(s['ax'].to_numpy()),
            s['y'].to_numpy(), s['g'].to_numpy())


def boot_spec(size, ax, y, grp):
    genes = np.unique(grp)
    g2i = {g: np.where(grp == g)[0] for g in genes}
    rng = np.random.default_rng(SEED)
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
        'domain INTERNAL (nterm=0)': (R['faithful'] == 1) & (R['nterm'] == 0),
        'domain N-TERMINAL (nterm=1) [AD class]': (R['faithful'] == 1) & (R['nterm'] == 1),
    }
    print("=" * 92)
    print(f"L30-ONLY axis3 usage (devils Attack 1) — specificity + gene-bootstrap 95pct CI (B={B})")
    print("feature = |Z[li,L30,k]-Z[si,L30,k]| (the vector PRISM ingests), NOT 30-layer trajectory")
    print("=" * 92)
    dist = {}
    for label, mask in groups.items():
        size, ax, y, grp = arrays(R[mask])
        pt = specificity(size, ax, y, grp)
        bs = boot_spec(size, ax, y, grp)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        dist[label] = bs
        tag = '>0 (axis3@L30 USED)' if lo > 0 else 'CI incl 0 (NOT sig — Attack1 CONFIRMED)'
        print(f"\n[{label}]  n={mask.sum()}, genes={len(np.unique(grp))}")
        print(f"    specificity@L30 = {pt:+.4f}   95%CI[{lo:+.4f}, {hi:+.4f}]   {tag}")

    def sep(a, b):
        n = min(len(dist[a]), len(dist[b])); d = dist[a][:n] - dist[b][:n]
        return d.mean(), *np.percentile(d, [2.5, 97.5])
    ad = 'domain N-TERMINAL (nterm=1) [AD class]'; nd = 'TRUE non-domain (faithful=0)'
    it = 'domain INTERNAL (nterm=0)'
    print("\n" + "-" * 92)
    for a, b, t in [(ad, nd, 'AD-class − non-domain (want >0)'),
                    (ad, it, 'AD-class − internal (want ~0)')]:
        m, lo, hi = sep(a, b)
        v = 'SEPARATED (>0)' if lo > 0 else ('INDISTINGUISHABLE' if hi > 0 > lo else 'BELOW (<0)')
        print(f"  Δ {t} = {m:+.4f}  95%CI[{lo:+.4f}, {hi:+.4f}]  -> {v}")
    print("\nDECISION (pre-registered): AD-class specificity@L30 CI incl 0 OR not separated from")
    print("non-domain => NOT resolvable via map => parallel findings. Else => survives Attack 1.")


if __name__ == '__main__':
    main()
