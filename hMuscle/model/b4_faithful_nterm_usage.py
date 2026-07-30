#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""b4_faithful_nterm_usage.py  — Option A re-audit of B4 domain-usage under FAITHFUL annotation.

Context: the map's B4 usage instrument (b4_domain_magnitude_usage.py) split domain vs non-domain by
the PLAIN domain_binary = truncated top-512 Pfam vocab, which is a false-negative for rare domains
(16,218 / 33,802 brain pairs are faithful-domain but plain=0, incl. all Complex I / AD switches).
This re-runs the SAME instrument (incremental held-out R^2 of axis3 trajectory displacement BEYOND
size = "is the domain channel output-USED") but splits by:
  (1) domain_binary_faithful  (full hmmscan envelope overlap, E<=0.01, the manuscript's §6 definition)
  (2) within faithful-domain, stratified by nterm_overlap  (N-terminal edit = the AD-switch geometry).

Question the AD switches pose: faithful=1 (real Complex I domain-loss) but nterm=1 (N-terminal edit,
which this session showed is pooling-inaccessible). Is such domain-loss B4-USED (resolvable tier) or
NOT (residual tier)?

Predict-before-you-look:
  faithful-domain ∩ internal (nterm=0):  incremental axis3 | size  > 0.01   (USED)
  faithful-domain ∩ N-terminal (nterm=1): incremental axis3 | size  ~ 0      (NOT used; AD-switch class)
  => AD switches: domain-loss real but PRISM output does not use the domain channel (size-only) => residual.
Gene-disjoint ridge CV R^2 + gene-bootstrap 95% CI on the incremental metric. Brain. Read-only.
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
from scipy.stats import spearmanr

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
BRAIN = ROOT / 'hMuscle/data/brain_isoquant_esm2/full'
INTERP = ROOT / 'reports/v20b_pca_interp'
MAXLEN = 1022
SEED = 42


def cvr2(X, y, grp):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, oof)


def incr_axis3(size, ax, y, grp):
    """incremental held-out R^2 of axis3 displacement beyond size, and beyond size+other-7."""
    ax3 = ax[:, [3]]
    ax_no3 = ax[:, [0, 1, 2, 4, 5, 6, 7]]
    r2_size = cvr2(size, y, grp)
    r2_ax3 = cvr2(np.hstack([size, ax3]), y, grp)
    r2_all = cvr2(np.hstack([size, ax]), y, grp)
    r2_no3 = cvr2(np.hstack([size, ax_no3]), y, grp)
    return {
        'r2_size': r2_size,
        'incr_ax3_over_size': r2_ax3 - r2_size,
        'incr_ax3_over_size_and_other7': r2_all - r2_no3,
        'rho_size': spearmanr(size[:, 0], y).correlation,
        'n': len(y), 'genes': len(np.unique(grp)),
    }


def boot_ci(size, ax, y, grp, B=300, seed=SEED):
    """gene-bootstrap 95% CI on incremental axis3 | size (the primary 'used' metric)."""
    genes = np.unique(grp)
    g2i = {g: np.where(grp == g)[0] for g in genes}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        samp = rng.choice(genes, size=len(genes), replace=True)
        idx = np.concatenate([g2i[g] for g in samp])
        gg = np.concatenate([np.full(len(g2i[g]), k) for k, g in enumerate(samp)])  # relabel groups
        try:
            r2s = cvr2(size[idx], y[idx], gg)
            r2a = cvr2(np.hstack([size[idx], ax[idx][:, [3]]]), y[idx], gg)
            vals.append(r2a - r2s)
        except Exception:
            continue
    vals = np.array(vals)
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
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
        recs.append({
            'y': float(np.abs(prism[li] - prism[si]).sum()),
            'size': float(r['size']),
            'ax': disp,
            'g': str(r['gene']),
            'plain': int(r['domain_binary']),
            'faithful': int(r['domain_binary_faithful']),
            'nterm': int(r['nterm_overlap']),
        })
    R = pd.DataFrame(recs)
    R = R[R['faithful'] >= 0].reset_index(drop=True)

    def subset(mask, label):
        s = R[mask]
        size = s['size'].to_numpy()[:, None]
        ax = np.stack(s['ax'].to_numpy())
        y = s['y'].to_numpy(); grp = s['g'].to_numpy()
        m = incr_axis3(size, ax, y, grp)
        lo, hi = boot_ci(size, ax, y, grp)
        used = 'USED' if lo > 0.01 else ('weak' if lo > 0 else 'NOT-used')
        print(f"\n[{label}]  n={m['n']}, genes={m['genes']}, Spearman(size,|dPRISM|)={m['rho_size']:.3f}")
        print(f"    size-alone R^2                    = {m['r2_size']:+.3f}")
        print(f"    incremental axis3 | size          = {m['incr_ax3_over_size']:+.4f}  "
              f"95%CI[{lo:+.4f},{hi:+.4f}]   -> {used}")
        print(f"    incremental axis3 | size+other-7  = {m['incr_ax3_over_size_and_other7']:+.4f}  "
              f"(specificity: ~0 => generic magnitude, not axis3)")
        return m

    print("=" * 95)
    print("B4 DOMAIN-USAGE RE-AUDIT under FAITHFUL annotation (Option A)")
    print("used-criterion: incremental axis3|size 95%CI lower bound > 0.01")
    print("=" * 95)
    print("\n--- reference: PLAIN (truncated-512) split, as the map originally computed ---")
    subset(R['plain'] == 1, 'PLAIN domain=1 (map original)')
    subset(R['plain'] == 0, 'PLAIN domain=0 (map original, incl 16218 false-neg)')

    print("\n--- FAITHFUL (full hmmscan) split ---")
    subset(R['faithful'] == 1, 'FAITHFUL domain=1 (all)')
    subset(R['faithful'] == 0, 'FAITHFUL domain=0 (true non-domain)')

    print("\n--- FAITHFUL domain, stratified by N-terminal geometry (the AD-switch tension) ---")
    subset((R['faithful'] == 1) & (R['nterm'] == 0), 'FAITHFUL domain ∩ INTERNAL (nterm=0)')
    subset((R['faithful'] == 1) & (R['nterm'] == 1), 'FAITHFUL domain ∩ N-TERMINAL (nterm=1) [AD-switch class]')

    print("\n" + "-" * 95)
    print("Read: if faithful∩internal is USED but faithful∩N-terminal is NOT-used, then the AD switches")
    print("      (nterm=1 domain-loss) are RESIDUAL tier: real domain content PRISM cannot output-use")
    print("      because it is realized N-terminally (pooling-inaccessible). If both USED, resolvable tier.")


if __name__ == '__main__':
    main()
