#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""b4_deltaL30L15_specificity_ci.py — Option B: does v17f*'s δ=L30−L15 recover the domain-axis3 signal?

This session found: domain-axis3 discrimination peaks mid-layer (L17) and DECAYS by L30, so the
deployed L30-only PRISM (v15d, BP/CC) cannot output-use it (AD-class specificity@L30 indistinguishable
from non-domain). Independently, memory (finding-delta-layer-domain-scope) shows v17f*'s δ_layer=L30−L15
HELPS MF (+0.039 macro) but LOSES BP. Convergence hypothesis: v17f*'s δ recovers the SAME mid-layer
domain-completeness signal (axis3) that L30-only loses -> explains the MF-specific δ benefit.

Test: recompute the axis3-usage SPECIFICITY metric under the v17f* input basis [L30 ∥ δ], i.e. add the
δ-axis components δ_k = Z[·,L30,k] − Z[·,L15,k] (linear proxy of proj(L30−L15); L15=idx14, L30=idx29)
alongside the L30 axis components. "axis3 channel" = {axis3@L30, axis3@δ} (2 cols); "other-7" = the
other axes at BOTH L30 and δ (14 cols). specificity = R^2(size+all@{L30,δ}) − R^2(size+other7@{L30,δ}).

PRE-REGISTERED decision (set before looking):
  * [L30∥δ] AD-class specificity SEPARATES from non-domain (Δ CI>0) AND exceeds L30-only gap (−0.0038)
      => δ RECOVERS domain-axis3 signal => convergence CONFIRMED (δ's MF benefit plausibly via mid-layer domain).
  * [L30∥δ] AD-class still INDISTINGUISHABLE from non-domain
      => δ does NOT recover domain-axis3 => convergence FALSE.
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
L15_IDX, L30_IDX = 14, 29  # Z is (N,30,8) for L1..L30 => L15=idx14, L30=idx29

# axis columns within the 16-dim [8 @L30 ∥ 8 @δ] feature block
AX3_COLS = [3, 3 + 8]                                   # axis3 at L30 and at δ
OTHER7_COLS = [c for c in range(16) if c not in AX3_COLS]


def cvr2(X, y, grp):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, oof)


def specificity(size, feat16, y, grp):
    """feat16 = [ |Δaxis@L30|(8) , |Δaxis@δ|(8) ].  axis3 channel = cols [3, 11]."""
    full = np.hstack([size, feat16])
    no3 = np.hstack([size, feat16[:, OTHER7_COLS]])
    return cvr2(full, y, grp) - cvr2(no3, y, grp)


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
        a30_l, a30_s = Z[li, L30_IDX, :], Z[si, L30_IDX, :]                 # axis@L30
        ad_l = Z[li, L30_IDX, :] - Z[li, L15_IDX, :]                        # δ-axis = L30−L15
        ad_s = Z[si, L30_IDX, :] - Z[si, L15_IDX, :]
        feat = np.concatenate([np.abs(a30_l - a30_s), np.abs(ad_l - ad_s)])  # (16,)
        recs.append({'y': float(np.abs(prism[li] - prism[si]).sum()), 'size': float(r['size']),
                     'feat': feat, 'g': str(r['gene']),
                     'faithful': int(r['domain_binary_faithful']), 'nterm': int(r['nterm_overlap'])})
    R = pd.DataFrame(recs)
    return R[R['faithful'] >= 0].reset_index(drop=True)


def arrays(s):
    return (s['size'].to_numpy()[:, None], np.stack(s['feat'].to_numpy()),
            s['y'].to_numpy(), s['g'].to_numpy())


def boot_spec(size, feat, y, grp):
    genes = np.unique(grp)
    g2i = {g: np.where(grp == g)[0] for g in genes}
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(B):
        samp = rng.choice(genes, size=len(genes), replace=True)
        idx = np.concatenate([g2i[g] for g in samp])
        gg = np.concatenate([np.full(len(g2i[g]), k) for k, g in enumerate(samp)])
        try:
            vals.append(specificity(size[idx], feat[idx], y[idx], gg))
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
    print("=" * 94)
    print(f"v17f* basis [L30 ∥ δ=L30−L15] axis3 usage — specificity + gene-bootstrap 95pct CI (B={B})")
    print("compare vs L30-only: does adding δ recover domain-axis3 that L30-alone lost?")
    print("L30-only ref: non-domain +0.0175 / internal +0.0223 / AD-class +0.0114 ; Δ(AD−nd)=-0.0038 (indist)")
    print("=" * 94)
    dist = {}
    for label, mask in groups.items():
        size, feat, y, grp = arrays(R[mask])
        pt = specificity(size, feat, y, grp)
        bs = boot_spec(size, feat, y, grp)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        dist[label] = bs
        print(f"\n[{label}]  n={mask.sum()}, genes={len(np.unique(grp))}")
        print(f"    specificity@[L30∥δ] = {pt:+.4f}   95%CI[{lo:+.4f}, {hi:+.4f}]   "
              f"{'>0' if lo > 0 else 'CI incl 0'}")

    def sep(a, b):
        n = min(len(dist[a]), len(dist[b])); d = dist[a][:n] - dist[b][:n]
        return d.mean(), *np.percentile(d, [2.5, 97.5])
    ad = 'domain N-TERMINAL (nterm=1) [AD class]'; nd = 'TRUE non-domain (faithful=0)'
    it = 'domain INTERNAL (nterm=0)'
    print("\n" + "-" * 94)
    for a, b, t in [(ad, nd, 'AD-class − non-domain (want >0 & > L30-only -0.0038: δ RECOVERS)'),
                    (it, nd, 'internal − non-domain (context)')]:
        m, lo, hi = sep(a, b)
        v = 'SEPARATED (>0)' if lo > 0 else ('INDISTINGUISHABLE' if hi > 0 > lo else 'BELOW (<0)')
        print(f"  Δ {t}\n      = {m:+.4f}  95%CI[{lo:+.4f}, {hi:+.4f}]  -> {v}")
    print("\nDECISION (pre-registered): AD-class SEPARATES from non-domain & exceeds L30-only gap")
    print("=> δ recovers domain-axis3 => convergence CONFIRMED. Else => convergence FALSE.")


if __name__ == '__main__':
    main()
