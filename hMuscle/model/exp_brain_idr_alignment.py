#!/usr/bin/env python3
"""
exp_brain_idr_alignment.py  (Option A: IDR alignment test)
==========================================================
Hypothesis: brain's within-gene delta "dark variance" (>=74% unexplained by
domain/celltype/length) originates from alternative splicing of INTRINSICALLY
DISORDERED REGIONS (IDRs) -- which are NOT Pfam domains, so they are dark to our
functional labels while ESM-2 delta sees them.

IDR proxy: TOP-IDP composition scale (Campen et al., Protein Pept Lett 2008),
window-smoothed per-residue disorder propensity. No external tool. Per isoform:
  idr_mean = mean smoothed TOP-IDP ;  idr_frac = fraction residues (smoothed>0).

For distinct 2-iso brain pairs (brain-known coverage via GTF name->ENST + .pep):
  - rho(delta_spread, pair_mean_IDR)     : do high-disorder pairs diverge more?
  - rho(delta_spread, |Delta IDR_frac|)  : do isoforms differing in disorder diverge?
  - partial(delta_spread, IDR | length, domain) : IDR independent of length & domain?
FALSIFIABLE PREDICTION (within-brain, no muscle seqs available): if IDR is the origin,
delta_spread must RISE across IDR tertiles (low->high). If flat -> IDR refuted.
Bootstrap CI n=1000 + shuffle null.
"""
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr, rankdata
import json

DATA = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/data')
BRAIN = DATA / 'brain_isoquant_esm2/full'
FEAT = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/features')
GTF = DATA / 'brain_esm2/brain_only.gtf'
PEP = DATA / 'brain_esm2/brain_only_transcripts.fa.transdecoder.pep'
OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/exp_variance_structure')
RNG = np.random.default_rng(42)
WIN = 15

# TOP-IDP scale (Campen et al. 2008): higher = more disorder-promoting
TOPIDP = {'A': 0.06, 'R': 0.180, 'N': 0.007, 'D': 0.192, 'C': 0.02, 'Q': 0.318,
          'E': 0.736, 'G': 0.166, 'H': 0.303, 'I': -0.486, 'L': -0.326, 'K': 0.586,
          'M': -0.397, 'F': -0.697, 'P': 0.987, 'S': 0.341, 'T': 0.059, 'W': -0.884,
          'Y': -0.510, 'V': -0.121}


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")


def idr_scores(seq):
    v = np.array([TOPIDP.get(a, 0.0) for a in seq], dtype=float)
    if len(v) == 0:
        return 0.0, 0.0, 0
    if len(v) >= WIN:
        k = np.ones(WIN) / WIN
        sm = np.convolve(v, k, mode='same')
    else:
        sm = np.full_like(v, v.mean())
    return float(sm.mean()), float((sm > 0).mean()), len(v)


def parse_pep():
    """base id -> (best_len, idr_mean, idr_frac) using the LONGEST ORF per base."""
    best = {}
    cur_id, cur_len, cur_seq = None, 0, []

    def flush(bid, ln, seq):
        if bid is None:
            return
        s = ''.join(seq)
        if bid not in best or ln > best[bid][0]:
            im, ifrac, _ = idr_scores(s)
            best[bid] = (ln, im, ifrac)
    for line in open(PEP):
        if line.startswith('>'):
            flush(cur_id, cur_len, cur_seq)
            pid = line[1:].split()[0]
            cur_id = pid.split('.p')[0].split('.')[0]
            m = re.search(r'len:(\d+)', line)
            cur_len = int(m.group(1)) if m else 0
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    flush(cur_id, cur_len, cur_seq)
    return best


def name2enst_map():
    d = {}
    for line in open(GTF):
        if "\ttranscript\t" not in line:
            continue
        em = re.search(r'transcript_id "([^"]+)"', line)
        nm = re.search(r'transcript_name "([^"]+)"', line)
        if em and nm:
            d[nm.group(1)] = em.group(1).split(".")[0]
    return d


def base_of(bid, n2e):
    if bid.startswith('ENST'):
        return bid.split('.')[0]
    if bid.startswith('transcript'):
        return bid
    return n2e.get(bid, '')


def boot_ci(x, y, n=1000):
    x = np.asarray(x); y = np.asarray(y); N = len(x)
    rs = np.array([spearmanr(x[s], y[s]).correlation
                   for s in (RNG.integers(0, N, N) for _ in range(n))])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def resid_rank(y, *ctrls):
    ry = rankdata(y); X = np.ones((len(ry), 1))
    for c in ctrls:
        X = np.c_[X, rankdata(c)]
    beta = np.linalg.lstsq(X, ry, rcond=None)[0]
    return ry - X @ beta


def partial(a, b, *ctrls):
    return float(np.corrcoef(resid_rank(a, *ctrls), resid_rank(b, *ctrls))[0, 1])


def main():
    n2e = name2enst_map()
    pep = parse_pep()
    print(f"GTF name->enst {len(n2e)}   pep bases {len(pep)}")

    delta = (np.load(BRAIN / 'brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
             - np.load(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy').astype(np.float32))
    genes = np.array([clean(x) for x in np.load(BRAIN / 'brain_full_gene_names.npy', allow_pickle=True)])
    ids = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    dommat = np.load(FEAT / 'domain_matrix_brain_full.npy')

    L = np.full(len(ids), np.nan); IM = np.full(len(ids), np.nan); IF = np.full(len(ids), np.nan)
    for k, bid in enumerate(ids):
        b = base_of(bid, n2e)
        if b and b in pep:
            L[k], IM[k], IF[k] = pep[b]
    cov = np.isfinite(IM).mean()
    print(f"IDR coverage {cov*100:.1f}% ({int(np.isfinite(IM).sum())}/{len(ids)})")

    Z = (delta - delta.mean(0)) / (delta.std(0) + 1e-8)
    gl, ginv = np.unique(genes, return_inverse=True)
    cnt = np.bincount(ginv, None, len(gl))
    spread, pair_idr, didr, dlen, dcontent = [], [], [], [], []
    for gi in np.where(cnt == 2)[0]:
        idx = np.where(ginv == gi)[0]; i, j = idx
        if np.array_equal(delta[i], delta[j]):
            continue
        if not (np.isfinite(IM[i]) and np.isfinite(IM[j])):
            continue
        spread.append(float(np.linalg.norm(Z[i] - Z[j])))
        pair_idr.append(0.5 * (IM[i] + IM[j]))            # pair mean disorder
        didr.append(abs(IF[i] - IF[j]))                    # disorder-fraction difference
        dlen.append(abs(L[i] - L[j]))
        dcontent.append(float(np.abs(dommat[i] - dommat[j]).sum()))
    spread = np.array(spread); pair_idr = np.array(pair_idr); didr = np.array(didr)
    dlen = np.array(dlen); dcontent = np.array(dcontent)
    print(f"pairs used = {len(spread)}")

    res = {'n_pairs': int(len(spread)), 'idr_coverage': float(cov)}
    r_pi = spearmanr(spread, pair_idr).correlation
    r_di = spearmanr(spread, didr).correlation
    res['rho_spread_vs_pair_mean_IDR'] = {'rho': float(r_pi), 'ci': boot_ci(spread, pair_idr),
                                          'null': float(spearmanr(spread, RNG.permutation(pair_idr)).correlation)}
    res['rho_spread_vs_deltaIDRfrac'] = {'rho': float(r_di), 'ci': boot_ci(spread, didr)}
    res['partial_spread_IDR_given_length'] = partial(spread, pair_idr, dlen)
    res['partial_spread_IDR_given_length_domain'] = partial(spread, pair_idr, dlen, dcontent)
    res['partial_spread_deltaIDRfrac_given_len_dom'] = partial(spread, didr, dlen, dcontent)
    # falsifiable: delta_spread across IDR tertiles
    q = np.quantile(pair_idr, [1/3, 2/3])
    tert = np.digitize(pair_idr, q)
    res['spread_by_IDR_tertile'] = {int(t): {'mean_spread': float(spread[tert == t].mean()),
                                             'n': int((tert == t).sum())} for t in [0, 1, 2]}
    res['falsifiable_IDR_rises'] = bool(
        res['spread_by_IDR_tertile'][2]['mean_spread'] > res['spread_by_IDR_tertile'][0]['mean_spread'])
    (OUT / 'idr_alignment.json').write_text(json.dumps(res, indent=2))

    print("\n=== BRAIN IDR alignment ===")
    print(f" spread~pair_mean_IDR   rho={r_pi:.3f} CI{[round(x,3) for x in res['rho_spread_vs_pair_mean_IDR']['ci']]} (null {res['rho_spread_vs_pair_mean_IDR']['null']:.3f})")
    print(f" spread~|dIDRfrac|      rho={r_di:.3f} CI{[round(x,3) for x in res['rho_spread_vs_deltaIDRfrac']['ci']]}")
    print(f" partial spread~IDR | length          = {res['partial_spread_IDR_given_length']:.4f}")
    print(f" partial spread~IDR | length,domain    = {res['partial_spread_IDR_given_length_domain']:.4f}")
    print(f" partial spread~|dIDRfrac| | len,domain = {res['partial_spread_deltaIDRfrac_given_len_dom']:.4f}")
    print(" spread by IDR tertile (low->high):",
          [round(res['spread_by_IDR_tertile'][t]['mean_spread'], 3) for t in [0, 1, 2]])
    print(f" FALSIFIABLE prediction (spread rises with IDR): {res['falsifiable_IDR_rises']}")
    print(f"Saved -> {OUT / 'idr_alignment.json'}")


if __name__ == '__main__':
    main()
