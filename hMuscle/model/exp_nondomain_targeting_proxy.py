#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_nondomain_targeting_proxy.py  (Option A: is the N-terminal anchor a FUNCTIONAL targeting axis?)

exp_nondomain_anchor_decomp.py found a strong cross-gene common anchor for N-terminal-edit
non-domain pairs (CV-dir-acc 0.805 vs internal chance 0.488). But COHERENCE != CORRECT FUNCTIONAL
DIRECTION: a coherent axis could be a generic "more N-terminal sequence / longer" geometric axis
rather than a targeting axis. The describability gap is precisely that isoform-resolved
localization labels do not exist, so we validate against an ESM-2-INDEPENDENT biophysical proxy of
mitochondrial/secretory targeting (MitoFates/TargetP-style N-terminal presequence features).

DISCRIMINATOR (S2, pre-registered):
  Let anchor = CV-consistent mean(long-short) direction (from exp_nondomain_anchor_decomp).
  proj_i = dot(meanpool_delta_i, anchor_hat).  dMTS_i = MTS(long) - MTS(short).
  H1 (null / describability gap stands): the coherent axis is generic length. proj correlates with
     dLength but NOT specifically with dMTS once dLength is partialled out (partial rho ~ 0).
  H2 (targeting axis): proj aligns with dMTS SPECIFICALLY: partial Spearman(proj, dMTS | dLength,
     dHydrophobicity) is significantly > 0 for N-terminal pairs, and this specificity is ABSENT for
     internal-edit pairs (placebo).
INDEPENDENCE: MTS score is computed from raw amino-acid biophysics only; it never sees ESM-2.
  Caveat (stated, not hidden): ESM-2 can in principle encode N-terminal composition, so the guard is
  SPECIFICITY (dMTS beyond dLength/dHydrophobicity), not raw independence.
"""
import os, re
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from difflib import SequenceMatcher
from pathlib import Path
import json
from scipy.stats import spearmanr, rankdata

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
DATA = ROOT / 'hMuscle/data'
MODEL = ROOT / 'hMuscle/model'
FAA = ROOT / 'reports/muscle_labelgap/muscle_2iso.fa'
DOMAIN_MAT = ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy'
OUT = ROOT / 'reports/muscle_labelgap/nondomain_targeting_proxy.json'
NTERM_WIN = 60
PRESEQ = 30      # mito presequence typical window (MitoFates ~ first 30 aa)
rng = np.random.default_rng(42)

# Kyte-Doolittle hydropathy
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
      'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,
      'Y':-1.3,'V':4.2}


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(' ', "")


def strip_orf(n):
    return re.sub(r'\.p\d+$', '', n)


def parse_faa():
    best, cur, buf = {}, None, []

    def flush():
        if cur is None:
            return
        s = ''.join(buf); b = strip_orf(cur)
        if b not in best or len(s) > len(best[b]):
            best[b] = s
    for line in open(FAA):
        if line.startswith('>'):
            flush(); cur = line[1:].split()[0]; buf = []
        else:
            buf.append(line.strip())
    flush()
    return best


def changed_intervals(long_s, short_s):
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    ivs, changed = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        changed += max(i2 - i1, j2 - j1)
        if i2 > i1:
            ivs.append((i1, i2))
    return ivs, changed


def hydrophobic_moment(seq, angle=100.0):
    """Eisenberg mean hydrophobic moment over the window (max over helical frame)."""
    if len(seq) < 5:
        return 0.0
    ang = np.deg2rad(angle)
    hs = np.array([KD.get(a, 0.0) for a in seq])
    n = len(hs)
    sx = np.sum(hs * np.cos(np.arange(n) * ang))
    sy = np.sum(hs * np.sin(np.arange(n) * ang))
    return float(np.sqrt(sx * sx + sy * sy) / n)


def mts_score(seq, win=PRESEQ):
    """MitoFates/TargetP-lite mitochondrial presequence propensity from raw sequence.
    Mito presequences: N-terminal amphipathic helix, Arg-rich, net positive, acidic-depleted."""
    w = seq[:win]
    if len(w) < 5:
        return 0.0, {}
    R = w.count('R'); K = w.count('K'); D = w.count('D'); E = w.count('E')
    net_charge = (R + K) - (D + E)
    arg_frac = R / len(w)
    acidic_frac = (D + E) / len(w)
    muH = hydrophobic_moment(w)
    # composite (MitoFates spirit): positive presequence signal
    comp = net_charge * 0.5 + arg_frac * 10.0 + muH * 3.0 - acidic_frac * 10.0
    return comp, {'net_charge': net_charge, 'arg_frac': arg_frac,
                  'acidic_frac': acidic_frac, 'muH': muH}


def whole_hydrophobicity(seq):
    if not seq:
        return 0.0
    return float(np.mean([KD.get(a, 0.0) for a in seq]))


def partial_spearman(x, y, covars):
    """Spearman partial correlation of x,y controlling for covars (list of arrays), via
    rank-residualization + linear regression on ranks."""
    def resid(v):
        r = rankdata(v).astype(float)
        C = np.column_stack([rankdata(c).astype(float) for c in covars] + [np.ones_like(r)])
        beta, *_ = np.linalg.lstsq(C, r, rcond=None)
        return r - C @ beta
    rx, ry = resid(x), resid(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x); k = len(covars)
    # t-approx p-value
    from scipy.stats import t as tdist
    dof = max(n - k - 2, 1)
    tstat = rho * np.sqrt(dof / max(1 - rho * rho, 1e-12))
    p = float(2 * tdist.sf(abs(tstat), dof))
    return rho, p


def cv_anchor_proj(D, gene_id):
    """gene-disjoint 5-fold: proj_i = dot(delta_i, anchor_hat(train without i's gene)).
    orient = long-short already baked into D. Returns proj array aligned to D rows."""
    n = len(D)
    ug = np.unique(gene_id); rng.shuffle(ug)
    folds = {g: i % 5 for i, g in enumerate(ug)}
    fid = np.array([folds[g] for g in gene_id])
    proj = np.zeros(n)
    for k in range(5):
        te = fid == k; tr = ~te
        if tr.sum() < 5 or te.sum() == 0:
            continue
        a = D[tr].mean(0); a /= (np.linalg.norm(a) + 1e-9)
        proj[te] = D[te] @ a
    return proj


def main():
    iso = np.array([clean(x) for x in np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)])
    gen = np.array([clean(x) for x in np.load(MODEL / 'my_gene_list_fixed.npy', allow_pickle=True)])
    dom = np.load(DOMAIN_MAT).sum(1).astype(int)
    L15 = np.load(DATA / 'esm2_layer_15_t30_150M.npy').astype(np.float32)
    L30 = np.load(DATA / 'esm2_layer_30_t30_150M.npy').astype(np.float32)
    faa = parse_faa()
    gl, gi = np.unique(gen, return_inverse=True)
    cnt = np.bincount(gi, minlength=len(gl))

    def emb(i):
        return np.concatenate([L15[i], L30[i]])

    D, first_pos, size, gene_id = [], [], [], []
    dMTS, dLen, dHyd = [], [], []
    dfeat = {'net_charge': [], 'arg_frac': [], 'acidic_frac': [], 'muH': []}
    for g in np.where(cnt == 2)[0]:
        a, b = np.where(gi == g)[0]
        if dom[a] != dom[b]:
            continue
        if iso[a] not in faa or iso[b] not in faa:
            continue
        sa, sb = faa[iso[a]], faa[iso[b]]
        if sa == sb:
            continue
        lo, sh = (a, b) if len(sa) >= len(sb) else (b, a)
        ls, ss = faa[iso[lo]], faa[iso[sh]]
        ivs, changed = changed_intervals(ls, ss)
        if changed == 0 or not ivs:
            continue
        D.append(emb(lo) - emb(sh))
        first_pos.append(ivs[0][0]); size.append(changed); gene_id.append(g)
        mL, fL = mts_score(ls); mS, fS = mts_score(ss)
        dMTS.append(mL - mS)
        dLen.append(len(ls) - len(ss))
        dHyd.append(whole_hydrophobicity(ls) - whole_hydrophobicity(ss))
        for kf in dfeat:
            dfeat[kf].append(fL[kf] - fS[kf])
    D = np.array(D); first_pos = np.array(first_pos); size = np.array(size)
    gene_id = np.array(gene_id); dMTS = np.array(dMTS); dLen = np.array(dLen); dHyd = np.array(dHyd)
    for kf in dfeat:
        dfeat[kf] = np.array(dfeat[kf])
    nterm = first_pos < NTERM_WIN
    proj = cv_anchor_proj(D, gene_id)
    print(f"pairs={len(D)}  N-term={nterm.sum()}  internal={(~nterm).sum()}", flush=True)

    res = {'n': int(len(D)), 'n_nterm': int(nterm.sum()), 'n_internal': int((~nterm).sum()),
           'proxy': 'MitoFates/TargetP-lite N-terminal presequence (ESM-2-independent)',
           'prediction': 'H1 null: proj~dLength only, no dMTS specificity (gap stands)'}

    for lab, m in [('N-terminal', nterm), ('internal', ~nterm)]:
        pj, dm, dl, dh = proj[m], dMTS[m], dLen[m], dHyd[m]
        raw_rho, raw_p = spearmanr(pj, dm)
        # specificity: partial out length + whole-seq hydrophobicity
        par_rho, par_p = partial_spearman(pj, dm, [dl, dh])
        # placebo: proj vs dLength (should be strong regardless)
        len_rho, len_p = spearmanr(pj, dl)
        cell = {'n': int(m.sum()),
                'proj_vs_dMTS_raw': {'rho': float(raw_rho), 'p': float(raw_p)},
                'proj_vs_dMTS_partial_len_hyd': {'rho': float(par_rho), 'p': float(par_p)},
                'proj_vs_dLength': {'rho': float(len_rho), 'p': float(len_p)}}
        # per-feature partial
        cell['per_feature_partial'] = {}
        for kf in dfeat:
            fr, fp = partial_spearman(pj, dfeat[kf][m], [dl, dh])
            cell['per_feature_partial'][kf] = {'rho': float(fr), 'p': float(fp)}
        res[lab] = cell
        print(f"\n[{lab}] n={m.sum()}", flush=True)
        print(f"  proj~dMTS raw     rho={raw_rho:+.3f} p={raw_p:.2e}", flush=True)
        print(f"  proj~dMTS |len,hyd rho={par_rho:+.3f} p={par_p:.2e}  <-- specificity test", flush=True)
        print(f"  proj~dLength      rho={len_rho:+.3f} p={len_p:.2e}  (placebo/generic axis)", flush=True)
        for kf in dfeat:
            c = res[lab]['per_feature_partial'][kf]
            print(f"    partial {kf:12} rho={c['rho']:+.3f} p={c['p']:.2e}", flush=True)

    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[saved] {OUT}", flush=True)


if __name__ == '__main__':
    main()
