#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_nondomain_anchor_decomp.py  (Option B: is the non-domain class anchor-DECOMPOSABLE?)

§6 frames the non-domain within-gene minority as "encoded but mechanism-heterogeneous with no
common reference direction" (so region-pool/anchor cannot recover functional direction the way it
can for domain-completeness). This experiment tests whether that "no common anchor" claim is too
strong for a MECHANISM-HOMOGENEOUS sub-class: N-terminal targeting edits (MTS/signal window),
exactly the class of NDUFAF5 (NMD) / NDUFS8 (MTS localization) switches.

Reuses the SAME 1505 non-domain (equal-domain-count) 2-iso muscle pairs as
exp_nondomain_disorder_saturation.py, adds splice POSITION, and asks:

  Q  Does the inter-isoform mean-pooled delta carry a CROSS-GENE-CONSISTENT direction (a common
     anchor) that is POSITION-SPECIFIC (N-terminal) rather than merely length-driven?

Discriminator (controls length): compare, AT MATCHED SPLICE SIZE, the direction coherence of
N-terminal-edit pairs vs internal-edit pairs. If both are only length-driven, coherence is equal;
if N-terminal carries extra targeting-specific structure, N-term >> internal at matched size.

Metrics per (position x size-bin) cell:
  (a) coherence R = ||mean(unit delta)||  vs sign-shuffle null (random orientation -> R ~ 1/sqrt(n))
  (b) gene-disjoint 5-fold CV direction accuracy: anchor = mean(train delta); predict sign(dot(test,anchor))
      against the consistently-defined long-minus-short orientation.

PRE-REGISTERED PREDICTION (S2, before running):
  H1 (null, manuscript's current claim): at mean-pool level, N-term ~= internal at matched size
     (coherence difference CI includes 0); any coherence is the shared length axis. => region-pool
     (per-residue) is REQUIRED and this mean-pool test cannot recover a targeting anchor.
  H2 (sub-class anchors exist): N-term coherence/CV-acc significantly exceeds internal at matched
     size => a deployable targeting anchor exists even at mean-pool.
POSITIVE CONTROL: domain-DIFFERING pairs (dom differs), oriented by domain count, must give strong
  CV accuracy (reproduces the known ~0.63 domain-ranking), validating the anchor machinery.
"""
import os, re
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from difflib import SequenceMatcher
from pathlib import Path
import json

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
DATA = ROOT / 'hMuscle/data'
MODEL = ROOT / 'hMuscle/model'
FAA = ROOT / 'reports/muscle_labelgap/muscle_2iso.fa'
DOMAIN_MAT = ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy'
OUT = ROOT / 'reports/muscle_labelgap/nondomain_anchor_decomp.json'
NTERM_WIN = 60  # MTS / signal-peptide / mito-targeting N-terminal window
rng = np.random.default_rng(42)


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(' ', '')


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


def coherence(D):
    """mean resultant length of unit vectors; null via sign shuffle."""
    U = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    R = float(np.linalg.norm(U.mean(0)))
    nulls = []
    for _ in range(1000):
        s = rng.choice([-1.0, 1.0], size=len(U))[:, None]
        nulls.append(np.linalg.norm((U * s).mean(0)))
    nulls = np.array(nulls)
    return R, float(nulls.mean()), float(np.percentile(nulls, 97.5)), float((nulls >= R).mean())


def cv_dir_acc(D, gene_id, orient):
    """gene-disjoint 5-fold: anchor=mean(train oriented delta); acc = sign(dot(test,anchor))==orient."""
    n = len(D)
    Do = D * orient[:, None]  # orient so 'positive class' direction is consistent
    ug = np.unique(gene_id)
    rng.shuffle(ug)
    folds = {g: i % 5 for i, g in enumerate(ug)}
    fid = np.array([folds[g] for g in gene_id])
    correct = 0
    for k in range(5):
        te = fid == k; tr = ~te
        if tr.sum() < 5 or te.sum() == 0:
            continue
        a = Do[tr].mean(0); a /= (np.linalg.norm(a) + 1e-9)
        pred = np.dot(Do[te], a) > 0  # oriented deltas should align with anchor
        correct += int(pred.sum())
    acc = correct / n
    # binomial CI
    se = np.sqrt(acc * (1 - acc) / n)
    return acc, acc - 1.96 * se, acc + 1.96 * se


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

    # ---- non-domain (equal-domain) 2-iso pairs, with splice position ----
    D, size, first_pos, gene_id = [], [], [], []
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
        D.append(emb(lo) - emb(sh))           # oriented long - short
        size.append(changed)
        first_pos.append(ivs[0][0])           # first changed residue on longer seq
        gene_id.append(g)
    D = np.array(D); size = np.array(size); first_pos = np.array(first_pos); gene_id = np.array(gene_id)
    n = len(D)
    nterm = first_pos < NTERM_WIN
    print(f"non-domain pairs: {n} | N-terminal(<{NTERM_WIN}aa): {nterm.sum()} | internal: {(~nterm).sum()}", flush=True)
    print(f"splice size median  N-term={np.median(size[nterm]):.0f}  internal={np.median(size[~nterm]):.0f}", flush=True)

    res = {'n_pairs': int(n), 'n_nterm': int(nterm.sum()), 'n_internal': int((~nterm).sum()),
           'NTERM_WIN': NTERM_WIN, 'prediction': 'H1 null: N-term~=internal at matched size (mean-pool anchor-less)'}

    # orientation for CV: for non-domain we have no functional label, so we test COHERENCE of the
    # (long-short) direction and its CV self-consistency; orient = +1 (already long-short).
    orient = np.ones(n)

    # ---- (1) overall coherence N-term vs internal ----
    print("\n=== coherence R (common-direction strength) vs sign-shuffle null ===")
    for lab, m in [('N-terminal', nterm), ('internal', ~nterm)]:
        R, nmean, nhi, p = coherence(D[m])
        acc, lo, hi = cv_dir_acc(D[m], gene_id[m], orient[m])
        res[f'{lab}_overall'] = {'n': int(m.sum()), 'R': R, 'R_null_mean': nmean, 'R_null_p975': nhi,
                                 'p_R': p, 'cv_dir_acc': acc, 'cv_ci': [lo, hi]}
        print(f"  {lab:11} n={m.sum():4} R={R:.3f} (null {nmean:.3f}, p={p:.3f}) | CV-dir-acc={acc:.3f} [{lo:.3f},{hi:.3f}]")

    # ---- (2) MATCHED-SIZE discriminator: N-term vs internal within common size bins ----
    print("\n=== matched-size: N-term vs internal coherence within splice-size bins ===")
    edges = np.percentile(size, [0, 33.33, 66.67, 100])
    res['matched_size'] = {}
    for bi in range(3):
        lo_e, hi_e = edges[bi], edges[bi + 1]
        inbin = (size >= lo_e) & (size <= hi_e)
        cell = {}
        for lab, m in [('N-terminal', nterm & inbin), ('internal', (~nterm) & inbin)]:
            if m.sum() < 20:
                cell[lab] = {'n': int(m.sum()), 'R': None}
                continue
            R, nmean, nhi, p = coherence(D[m])
            acc, clo, chi = cv_dir_acc(D[m], gene_id[m], orient[m])
            cell[lab] = {'n': int(m.sum()), 'R': R, 'R_null_p975': nhi, 'p_R': p,
                         'cv_dir_acc': acc, 'cv_ci': [clo, chi]}
        res['matched_size'][f'bin{bi}_{lo_e:.0f}-{hi_e:.0f}aa'] = cell
        nt, it = cell.get('N-terminal', {}), cell.get('internal', {})
        if nt.get('R') is not None and it.get('R') is not None:
            print(f"  size {lo_e:.0f}-{hi_e:.0f}aa | N-term(n={nt['n']}) R={nt['R']:.3f} acc={nt['cv_dir_acc']:.3f}"
                  f"  vs  internal(n={it['n']}) R={it['R']:.3f} acc={it['cv_dir_acc']:.3f}"
                  f"  | ΔR={nt['R']-it['R']:+.3f}")

    # ---- (3) POSITIVE CONTROL: domain-differing pairs oriented by domain count ----
    print("\n=== positive control: domain-DIFFERING pairs, oriented by domain completeness ===")
    Dd, gd, orient_d = [], [], []
    for g in np.where(cnt == 2)[0]:
        a, b = np.where(gi == g)[0]
        if dom[a] == dom[b]:
            continue
        if iso[a] not in faa or iso[b] not in faa:
            continue
        hi_i, lo_i = (a, b) if dom[a] > dom[b] else (b, a)
        Dd.append(emb(a) - emb(b))
        orient_d.append(1.0 if dom[a] > dom[b] else -1.0)  # orient toward domain-complete
        gd.append(g)
    Dd = np.array(Dd); gd = np.array(gd); orient_d = np.array(orient_d)
    Rc, nmc, nhc, pc = coherence(Dd * orient_d[:, None])
    accc, cloc, chic = cv_dir_acc(Dd, gd, orient_d)
    res['positive_control_domain'] = {'n': int(len(Dd)), 'R': Rc, 'R_null_p975': nhc, 'p_R': pc,
                                      'cv_dir_acc': accc, 'cv_ci': [cloc, chic]}
    print(f"  domain-complete anchor n={len(Dd)} R={Rc:.3f} (null p975 {nhc:.3f}, p={pc:.3f})"
          f" | CV-dir-acc={accc:.3f} [{cloc:.3f},{chic:.3f}]")

    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[saved] {OUT}")


if __name__ == '__main__':
    main()
