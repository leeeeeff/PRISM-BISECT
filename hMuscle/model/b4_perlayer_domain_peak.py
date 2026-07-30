#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""b4_perlayer_domain_peak.py — Option A: WHERE (which layer) does axis3 domain-specificity peak?

This session showed domain-axis3 output-specificity is INDISTINGUISHABLE from non-domain at L30 and
at [L30∥δ]. viz scripts claim a "mid-layer L17 divergence peak" but that is unmeasured folklore here.
This measures the per-layer curve directly (folklore -> measurement, CLAUDE.md folklore-금지):
  for each layer l in {L1..L30}: specificity_gap(l) = spec_domain(l) − spec_nondomain(l),
  where spec_X(l) = R^2(size + axis@l[8]) − R^2(size + other7@l[7])  on subset X (faithful domain split).
Peak layer = argmax gap. Confirms whether the domain signal peaks mid-layer and decays by L30 (idx29),
grounding the manuscript limitation nuance in this analysis. Bootstrap CI only at peak & at L30.
Gene-disjoint ridge CV(5). Brain. Read-only.
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
B = 400
OTHER7 = [0, 1, 2, 4, 5, 6, 7]


def cvr2(X, y, grp):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, oof)


def spec_at(size, axl, y, grp):
    return cvr2(np.hstack([size, axl]), y, grp) - cvr2(np.hstack([size, axl[:, OTHER7]]), y, grp)


def build():
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bsp)
    df = pd.read_csv(SEV / 'brain_severity_pairs_scored.tsv', sep='\t')
    df = df[df['tissue'] == 'brain'].reset_index(drop=True)
    iso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    prism = np.load(ROOT / 'reports/brain_full_672_scores.npy').astype(np.float32)
    Z = np.load(INTERP / 'Z_brain_Nx30x8.npy')                             # (N,30,8)
    li_l, si_l, ys, sizes, gs, fs = [], [], [], [], [], []
    for _, r in df.iterrows():
        li, si = int(r['long_idx']), int(r['short_idx'])
        lid, sid = iso[li], iso[si]
        if lid not in seqs or sid not in seqs or seqs[lid][:MAXLEN] == seqs[sid][:MAXLEN]:
            continue
        li_l.append(li); si_l.append(si)
        ys.append(float(np.abs(prism[li] - prism[si]).sum())); sizes.append(float(r['size']))
        gs.append(str(r['gene'])); fs.append(int(r['domain_binary_faithful']))
    li_l, si_l = np.array(li_l), np.array(si_l)
    # per-layer pair feature |ΔZ@layer| : (npairs, 30, 8)
    F = np.abs(Z[li_l] - Z[si_l])                                          # (npairs,30,8)
    return (F, np.array(ys), np.array(sizes)[:, None], np.array(gs), np.array(fs))


def boot_gap(size, dl, ds, yd, ys_, gd, gs_):
    """CI on spec_domain(layer) − spec_nondomain(layer) via gene-bootstrap (paired index)."""
    def one(size_, axl, y, grp, rng):
        genes = np.unique(grp); g2i = {g: np.where(grp == g)[0] for g in genes}
        samp = rng.choice(genes, size=len(genes), replace=True)
        idx = np.concatenate([g2i[g] for g in samp])
        gg = np.concatenate([np.full(len(g2i[g]), k) for k, g in enumerate(samp)])
        return spec_at(size_[idx], axl[idx], y[idx], gg)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(B):
        try:
            vals.append(one(size[gd], dl, yd, None, gd, rng) if False else None)
        except Exception:
            pass
    return None  # (unused; kept simple — CI computed inline in main)


def main():
    F, y, size, g, faith = build()
    dom = faith == 1; nd = faith == 0
    L = np.arange(1, 31)
    print("=" * 88)
    print("Per-layer axis3 domain-specificity: gap(l) = spec_domain(l) − spec_nondomain(l)")
    print(f"(faithful domain split; point estimates all layers, bootstrap CI at peak & L30, B={B})")
    print("=" * 88)
    gaps, sd_, sn_ = [], [], []
    for l in range(30):
        sd = spec_at(size[dom], F[dom, l, :], y[dom], g[dom])
        sn = spec_at(size[nd], F[nd, l, :], y[nd], g[nd])
        gaps.append(sd - sn); sd_.append(sd); sn_.append(sn)
    gaps = np.array(gaps)
    for l in range(30):
        bar = '#' * max(0, int(gaps[l] / 0.001)) if gaps[l] > 0 else ''
        star = '  <-- L30 (PRISM input)' if l == 29 else ('  <-- PEAK' if l == int(np.argmax(gaps)) else '')
        print(f"  L{L[l]:>2}  domain={sd_[l]:+.4f}  nondom={sn_[l]:+.4f}  gap={gaps[l]:+.4f} {bar}{star}")
    peak = int(np.argmax(gaps))
    print(f"\n  PEAK layer = L{L[peak]}  gap={gaps[peak]:+.4f}   |   L30 gap={gaps[29]:+.4f}")

    # bootstrap CI on gap at peak and at L30
    def boot(l):
        rng = np.random.default_rng(SEED)
        gd, gn = g[dom], g[nd]
        gdu, gnu = np.unique(gd), np.unique(gn)
        d2i = {gg: np.where(gd == gg)[0] for gg in gdu}; n2i = {gg: np.where(gn == gg)[0] for gg in gnu}
        vals = []
        for _ in range(B):
            sd_s = rng.choice(gdu, len(gdu), replace=True); idxd = np.concatenate([d2i[x] for x in sd_s])
            ggd = np.concatenate([np.full(len(d2i[x]), k) for k, x in enumerate(sd_s)])
            sn_s = rng.choice(gnu, len(gnu), replace=True); idxn = np.concatenate([n2i[x] for x in sn_s])
            ggn = np.concatenate([np.full(len(n2i[x]), k) for k, x in enumerate(sn_s)])
            try:
                sd = spec_at(size[dom][idxd], F[dom, l, :][idxd], y[dom][idxd], ggd)
                sn = spec_at(size[nd][idxn], F[nd, l, :][idxn], y[nd][idxn], ggn)
                vals.append(sd - sn)
            except Exception:
                pass
        return np.percentile(vals, [2.5, 97.5])
    for l, tag in [(peak, f'PEAK L{L[peak]}'), (29, 'L30')]:
        lo, hi = boot(l)
        print(f"  gap@{tag} 95%CI[{lo:+.4f},{hi:+.4f}]  {'>0 (domain separates)' if lo>0 else 'CI incl 0'}")


if __name__ == '__main__':
    main()
