#!/usr/bin/env python3
"""
exp_brain_length_control.py  (Option A, length axis)
====================================================
Brain within-gene delta_spread aligns with domain content at only rho~0.29 (rho^2<9%);
the majority is "dark variance". Prime suspect for the unaligned magnitude: sequence
LENGTH change (bigger sequence change -> bigger delta, functionally agnostic).

Build per-isoform protein length for brain_full (via GTF transcript_name->ENST + TransDecoder
.pep 'len:' field, covering GENCODE + IsoQuant novel), then for distinct 2-iso pairs:
  - rho(spread, |Delta length|)
  - partial rho(spread, domain | length)   -> is domain signal length-independent?
  - partial rho(spread, length | domain)   -> does length add beyond domain?
  - fraction of spread rank-variance explained by length alone vs domain alone vs both
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


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")


def build_length_map():
    # transcript_name -> transcript_id (base, no version)
    name2enst = {}
    for line in open(GTF):
        if "\ttranscript\t" not in line:
            continue
        em = re.search(r'transcript_id "([^"]+)"', line)
        nm = re.search(r'transcript_name "([^"]+)"', line)
        if em and nm:
            name2enst[nm.group(1)] = em.group(1).split(".")[0]
    # pep: base id -> max ORF protein length
    base2len = defaultdict(int)
    for line in open(PEP):
        if not line.startswith('>'):
            continue
        pid = line[1:].split()[0]                     # e.g. ENST00000003084.p1
        base = pid.split('.p')[0].split('.')[0]       # -> ENST00000003084 / transcriptNNN
        mlen = re.search(r'len:(\d+)', line)
        if mlen:
            base2len[base] = max(base2len[base], int(mlen.group(1)))
    return name2enst, base2len


def length_of(bid, name2enst, base2len):
    if bid.startswith('ENST'):
        return base2len.get(bid.split('.')[0], np.nan)
    if bid.startswith('transcript'):
        return base2len.get(bid, np.nan)
    enst = name2enst.get(bid, '')
    return base2len.get(enst, np.nan) if enst else np.nan


def boot_ci(x, y, n=1000):
    x = np.asarray(x); y = np.asarray(y); N = len(x)
    rs = [spearmanr(x[s], y[s]).correlation for s in (RNG.integers(0, N, N) for _ in range(n))]
    rs = np.array(rs)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def resid_rank(y, *ctrls):
    ry = rankdata(y)
    X = np.c_[np.ones_like(ry)] + 0.0
    for c in ctrls:
        X = np.c_[X, rankdata(c)]
    beta = np.linalg.lstsq(X, ry, rcond=None)[0]
    return ry - X @ beta


def partial(a, b, *ctrls):
    ea, eb = resid_rank(a, *ctrls), resid_rank(b, *ctrls)
    return float(np.corrcoef(ea, eb)[0, 1])


def main():
    name2enst, base2len = build_length_map()
    print(f"GTF name->enst: {len(name2enst)}   pep base->len: {len(base2len)}")

    delta = (np.load(BRAIN / 'brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
             - np.load(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy').astype(np.float32))
    genes = np.array([clean(x) for x in np.load(BRAIN / 'brain_full_gene_names.npy', allow_pickle=True)])
    ids = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    dommat = np.load(FEAT / 'domain_matrix_brain_full.npy')

    lengths = np.array([length_of(b, name2enst, base2len) for b in ids], dtype=float)
    cov = np.isfinite(lengths).mean()
    print(f"length coverage: {cov*100:.1f}% ({int(np.isfinite(lengths).sum())}/{len(ids)})")

    Z = (delta - delta.mean(0)) / (delta.std(0) + 1e-8)
    domcount = dommat.sum(1)
    gl, ginv = np.unique(genes, return_inverse=True)
    cnt = np.bincount(ginv, None, len(gl))

    spread, dcontent, dlen = [], [], []
    n_drop_ident, n_drop_len = 0, 0
    for gi in np.where(cnt == 2)[0]:
        idx = np.where(ginv == gi)[0]
        i, j = idx[0], idx[1]
        if np.array_equal(delta[i], delta[j]):
            n_drop_ident += 1; continue
        if not (np.isfinite(lengths[i]) and np.isfinite(lengths[j])):
            n_drop_len += 1; continue
        spread.append(float(np.linalg.norm(Z[i] - Z[j])))
        dcontent.append(float(np.abs(dommat[i] - dommat[j]).sum()))
        dlen.append(abs(lengths[i] - lengths[j]))
    spread = np.array(spread); dcontent = np.array(dcontent); dlen = np.array(dlen)
    print(f"pairs used={len(spread)}  (dropped identical={n_drop_ident}, no-length={n_drop_len})")

    r_len = spearmanr(spread, dlen).correlation
    r_dom = spearmanr(spread, dcontent).correlation
    r_len_dom = spearmanr(dlen, dcontent).correlation
    res = {
        'n_pairs': len(spread), 'length_coverage': float(cov),
        'rho_spread_vs_length': {'rho': float(r_len), 'ci': boot_ci(spread, dlen),
                                 'shuffle_null': float(spearmanr(spread, RNG.permutation(dlen)).correlation)},
        'rho_spread_vs_domain': {'rho': float(r_dom), 'ci': boot_ci(spread, dcontent)},
        'rho_length_vs_domain': float(r_len_dom),
        'partial_spread_domain_given_length': partial(spread, dcontent, dlen),
        'partial_spread_length_given_domain': partial(spread, dlen, dcontent),
    }
    # variance explained (rank R^2) length alone / domain alone / both
    ry = rankdata(spread)
    def r2(*ctrls):
        e = resid_rank(spread, *ctrls)
        return 1 - np.var(e) / np.var(ry)
    res['rankR2_length'] = float(r2(dlen))
    res['rankR2_domain'] = float(r2(dcontent))
    res['rankR2_both'] = float(r2(dlen, dcontent))
    (OUT / 'length_control.json').write_text(json.dumps(res, indent=2))

    print("\n=== BRAIN length control ===")
    print(f" spread~length   rho={r_len:.3f} CI{[round(x,3) for x in res['rho_spread_vs_length']['ci']]} (null {res['rho_spread_vs_length']['shuffle_null']:.3f})")
    print(f" spread~domain   rho={r_dom:.3f} CI{[round(x,3) for x in res['rho_spread_vs_domain']['ci']]}")
    print(f" length~domain   rho={r_len_dom:.3f}  (are they entangled?)")
    print(f" partial spread~domain | length = {res['partial_spread_domain_given_length']:.4f}  (domain length-independent?)")
    print(f" partial spread~length | domain = {res['partial_spread_length_given_domain']:.4f}  (length beyond domain?)")
    print(f" rank-R2: length={res['rankR2_length']:.4f}  domain={res['rankR2_domain']:.4f}  both={res['rankR2_both']:.4f}")
    print(f" -> dark fraction (1-both) = {1-res['rankR2_both']:.4f}")
    print(f"Saved -> {OUT / 'length_control.json'}")


if __name__ == '__main__':
    main()
