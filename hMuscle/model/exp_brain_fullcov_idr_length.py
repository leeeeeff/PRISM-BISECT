#!/usr/bin/env python3
"""
exp_brain_fullcov_idr_length.py  (Option B: full-coverage re-run)
=================================================================
The length/IDR tests were capped at 46.5% coverage (brain-known .pep only), excluding
novel isoforms where microexons/IDR events concentrate. Using the SQANTI3 protein FASTA
(covers novel 'transcriptNNN.chrX.nic' ids directly) + GTF name->ENST for known, extend
coverage to ~90% and re-validate the KEY finding: does delta_spread ~ |Delta IDR frac|
(partial | length, domain) survive at full coverage? And does mean-IDR stay null?

Non-negotiable check: the strongest dark-variance-origin signal (|Delta IDR frac| partial
0.43 at 46% cov) must replicate at full coverage or it was a known-isoform artifact.
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
FAA = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/02_Isoquant_Output/SQANTI3_output/isoforms_corrected.faa')
OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/exp_variance_structure')
RNG = np.random.default_rng(42)
WIN = 15
TOPIDP = {'A': 0.06, 'R': 0.180, 'N': 0.007, 'D': 0.192, 'C': 0.02, 'Q': 0.318,
          'E': 0.736, 'G': 0.166, 'H': 0.303, 'I': -0.486, 'L': -0.326, 'K': 0.586,
          'M': -0.397, 'F': -0.697, 'P': 0.987, 'S': 0.341, 'T': 0.059, 'W': -0.884,
          'Y': -0.510, 'V': -0.121}


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")


def idr(seq):
    v = np.array([TOPIDP.get(a, 0.0) for a in seq], float)
    if len(v) == 0:
        return 0.0, 0.0, 0
    sm = np.convolve(v, np.ones(WIN) / WIN, mode='same') if len(v) >= WIN else np.full_like(v, v.mean())
    return float(sm.mean()), float((sm > 0).mean()), len(v)


def parse_faa():
    """base transcript id -> (len, idr_mean, idr_frac) keeping longest ORF."""
    best = {}
    cur, seq = None, []

    def flush(bid, s):
        if bid is None:
            return
        seqs = ''.join(s)
        if bid not in best or len(seqs) > best[bid][0]:
            im, ifrac, ln = idr(seqs)
            best[bid] = (ln, im, ifrac)
    for line in open(FAA):
        if line.startswith('>'):
            flush(cur, seq)
            cur = line[1:].split()[0].split('.p')[0]     # base id (ENST or transcriptNNN.chrX.xxic)
            seq = []
        else:
            seq.append(line.strip())
    flush(cur, seq)
    return best


def name2enst():
    d = {}
    for line in open(GTF):
        if "\ttranscript\t" not in line:
            continue
        em = re.search(r'transcript_id "([^"]+)"', line); nm = re.search(r'transcript_name "([^"]+)"', line)
        if em and nm:
            d[nm.group(1)] = em.group(1).split('.')[0]
    return d


def base_of(bid, n2e):
    if bid.startswith('ENST'):
        return bid.split('.')[0]
    if bid.startswith('transcript'):
        return bid                                       # novel: faa key is full 'transcriptNNN.chrX.xxic'
    return n2e.get(bid, '')


def boot_ci(x, y, n=1000):
    x = np.asarray(x); y = np.asarray(y); N = len(x)
    rs = np.array([spearmanr(x[s], y[s]).correlation for s in (RNG.integers(0, N, N) for _ in range(n))])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def resid(y, *c):
    ry = rankdata(y); X = np.ones((len(ry), 1))
    for cc in c:
        X = np.c_[X, rankdata(cc)]
    return ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]


def partial(a, b, *c):
    return float(np.corrcoef(resid(a, *c), resid(b, *c))[0, 1])


def main():
    faa = parse_faa(); n2e = name2enst()
    print(f"faa bases {len(faa)}  gtf name->enst {len(n2e)}")
    delta = (np.load(BRAIN / 'brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
             - np.load(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy').astype(np.float32))
    genes = np.array([clean(x) for x in np.load(BRAIN / 'brain_full_gene_names.npy', allow_pickle=True)])
    ids = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    dommat = np.load(FEAT / 'domain_matrix_brain_full.npy')

    L = np.full(len(ids), np.nan); IM = np.full(len(ids), np.nan); IF = np.full(len(ids), np.nan)
    n_known, n_novel = 0, 0
    for k, bid in enumerate(ids):
        b = base_of(bid, n2e)
        if b and b in faa:
            L[k], IM[k], IF[k] = faa[b]
            if bid.startswith('transcript'):
                n_novel += 1
            else:
                n_known += 1
    cov = np.isfinite(IM).mean()
    print(f"coverage {cov*100:.1f}% ({int(np.isfinite(IM).sum())}/{len(ids)})  known={n_known} novel={n_novel}")

    Z = (delta - delta.mean(0)) / (delta.std(0) + 1e-8)
    gl, ginv = np.unique(genes, return_inverse=True); cnt = np.bincount(ginv, None, len(gl))
    spread, pmid, difr, dlen, dcon = [], [], [], [], []
    n_novel_pair = 0
    for gi in np.where(cnt == 2)[0]:
        a, b = np.where(ginv == gi)[0]
        if np.array_equal(delta[a], delta[b]):
            continue
        if not (np.isfinite(IM[a]) and np.isfinite(IM[b])):
            continue
        spread.append(float(np.linalg.norm(Z[a] - Z[b])))
        pmid.append(0.5 * (IM[a] + IM[b]))
        difr.append(abs(IF[a] - IF[b]))
        dlen.append(abs(L[a] - L[b]))
        dcon.append(float(np.abs(dommat[a] - dommat[b]).sum()))
        if ids[a].startswith('transcript') or ids[b].startswith('transcript'):
            n_novel_pair += 1
    spread = np.array(spread); pmid = np.array(pmid); difr = np.array(difr)
    dlen = np.array(dlen); dcon = np.array(dcon)
    print(f"pairs used = {len(spread)} (novel-involving = {n_novel_pair})  vs 46%-cov was 428")

    res = {'coverage': float(cov), 'n_pairs': int(len(spread)), 'n_novel_pairs': int(n_novel_pair),
           'rho_spread_meanIDR': {'rho': float(spearmanr(spread, pmid).correlation), 'ci': boot_ci(spread, pmid)},
           'rho_spread_dIDRfrac': {'rho': float(spearmanr(spread, difr).correlation), 'ci': boot_ci(spread, difr)},
           'rho_spread_length': {'rho': float(spearmanr(spread, dlen).correlation), 'ci': boot_ci(spread, dlen)},
           'rho_spread_domain': {'rho': float(spearmanr(spread, dcon).correlation), 'ci': boot_ci(spread, dcon)},
           'partial_dIDRfrac_given_len_dom': partial(spread, difr, dlen, dcon),
           'partial_domain_given_len': partial(spread, dcon, dlen),
           'partial_length_given_dom': partial(spread, dlen, dcon)}
    (OUT / 'fullcov_idr_length.json').write_text(json.dumps(res, indent=2))
    print("\n=== FULL COVERAGE re-run ===")
    print(f" spread~meanIDR   rho={res['rho_spread_meanIDR']['rho']:.3f} CI{[round(x,3) for x in res['rho_spread_meanIDR']['ci']]}")
    print(f" spread~|dIDRfrac| rho={res['rho_spread_dIDRfrac']['rho']:.3f} CI{[round(x,3) for x in res['rho_spread_dIDRfrac']['ci']]}")
    print(f" spread~length   rho={res['rho_spread_length']['rho']:.3f}")
    print(f" spread~domain   rho={res['rho_spread_domain']['rho']:.3f}")
    print(f" partial |dIDRfrac| | len,dom = {res['partial_dIDRfrac_given_len_dom']:.4f}   (was 0.43 @46%)")
    print(f" partial domain | len         = {res['partial_domain_given_len']:.4f}   (was 0.14 @46%)")
    print(f" partial length | dom         = {res['partial_length_given_dom']:.4f}")
    print(f"Saved -> {OUT / 'fullcov_idr_length.json'}")


if __name__ == '__main__':
    main()
