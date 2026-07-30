#!/usr/bin/env python3
"""
exp_brain_alignment_axes.py  (Option A)
=======================================
Brain has ~2.9x more within-gene delta variance than muscle
(finding-brain-within-gene-variance), yet the within-gene DR ceiling is unbeatable
(3-fold bracket). Question: what does brain's EXCESS within-gene delta variance align
with? Is it the functional domain axis (the current criterion, which saturates), a
brain-specific cell-type axis (a NEW resolution axis), or neither (spread != function)?

For each distinct 2-isoform gene, measure per-gene within-gene delta_spread and its
Spearman correlation with candidate alignment axes:
  - domain_count_div = |Delta n_domains|             (the DR anchor; current criterion)
  - domain_content_div = L1(domain_vec_i, domain_vec_j)  (which domains changed)
  - celltype_div = Jensen-Shannon div of cell-type usage (brain-specific axis)
partial Spearman(delta_spread, celltype | domain) tells if celltype is a NEW axis
independent of domain. Bootstrap CI n=1000 + shuffle null. Muscle = domain only (no
celltype), as a cross-tissue contrast.

Works in ORIGINAL index space (domain matrix & celltype vectors are 63994 original rows);
identical-embedding sibling pairs (23% in brain) are dropped at the PAIR level.
"""
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, rankdata
import json

DATA = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/data')
BRAIN = DATA / 'brain_isoquant_esm2/full'
ID_DIR = DATA / 'raw_data/data/id_lists'
FEAT = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/features')
OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/exp_variance_structure')
RNG = np.random.default_rng(42)


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")


def jsd(p, q):
    p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / (b[mask] + 1e-12))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pair_indices(genes, delta):
    """distinct 2-iso genes -> list of (i,j) original indices, dropping identical-delta pairs."""
    gl, ginv = np.unique(genes, return_inverse=True)
    cnt = np.bincount(ginv, None, len(gl))
    pairs = []
    for gi in np.where(cnt == 2)[0]:
        idx = np.where(ginv == gi)[0]
        if np.array_equal(delta[idx[0]], delta[idx[1]]):
            continue                        # identical sequence -> not a distinct pair
        pairs.append((idx[0], idx[1]))
    return pairs


def boot_ci(x, y, n=1000):
    x = np.asarray(x); y = np.asarray(y); N = len(x)
    rs = []
    for _ in range(n):
        s = RNG.integers(0, N, N)
        rs.append(spearmanr(x[s], y[s]).correlation)
    rs = np.array(rs)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def partial_spearman(a, b, c):
    """Spearman partial correlation of a,b controlling for c (rank-residual method)."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]
        beta = np.linalg.lstsq(x1, y, rcond=None)[0]
        return y - x1 @ beta
    ea, eb = resid(ra, rc), resid(rb, rc)
    return float(np.corrcoef(ea, eb)[0, 1])


def analyze(tissue, delta, genes, dommat, celltype=None):
    # z-score delta per dim on full tissue
    Z = (delta - delta.mean(0)) / (delta.std(0) + 1e-8)
    pairs = pair_indices(genes, delta)
    domcount = dommat.sum(1)
    spread, dcount, dcontent, cdiv = [], [], [], []
    for i, j in pairs:
        spread.append(float(np.linalg.norm(Z[i] - Z[j])))
        dcount.append(abs(float(domcount[i] - domcount[j])))
        dcontent.append(float(np.abs(dommat[i] - dommat[j]).sum()))
        if celltype is not None:
            cdiv.append(jsd(celltype[i].astype(np.float64), celltype[j].astype(np.float64)))
    spread = np.array(spread); dcount = np.array(dcount)
    dcontent = np.array(dcontent)
    res = {'tissue': tissue, 'n_distinct_pairs': len(pairs)}
    r_dc = spearmanr(spread, dcount).correlation
    r_dct = spearmanr(spread, dcontent).correlation
    res['rho_spread_vs_domaincount'] = {'rho': float(r_dc), 'ci': boot_ci(spread, dcount)}
    res['rho_spread_vs_domaincontent'] = {'rho': float(r_dct), 'ci': boot_ci(spread, dcontent)}
    # shuffle null
    r_null = spearmanr(spread, RNG.permutation(dcontent)).correlation
    res['domaincontent_shuffle_null_rho'] = float(r_null)
    if celltype is not None:
        cdiv = np.array(cdiv)
        r_ct = spearmanr(spread, cdiv).correlation
        res['rho_spread_vs_celltype'] = {'rho': float(r_ct), 'ci': boot_ci(spread, cdiv)}
        res['rho_domaincontent_vs_celltype'] = float(spearmanr(dcontent, cdiv).correlation)
        res['partial_rho_spread_celltype_given_domain'] = partial_spearman(spread, cdiv, dcontent)
        res['celltype_shuffle_null_rho'] = float(spearmanr(spread, RNG.permutation(cdiv)).correlation)
    return res


def main():
    out = {}
    # BRAIN
    b_d = (np.load(BRAIN / 'brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
           - np.load(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy').astype(np.float32))
    b_g = np.array([clean(x) for x in np.load(BRAIN / 'brain_full_gene_names.npy', allow_pickle=True)])
    b_dom = np.load(FEAT / 'domain_matrix_brain_full.npy')
    b_ct = np.load(FEAT / 'cell_type_expression_vectors.npy')
    assert b_d.shape[0] == b_dom.shape[0] == b_ct.shape[0] == len(b_g)
    out['brain'] = analyze('brain', b_d, b_g, b_dom, celltype=b_ct)
    print("BRAIN:", json.dumps(out['brain'], indent=2))

    # MUSCLE (domain only) -- try to match a domain matrix to 31668 train rows
    m_d = (np.load(DATA / 'esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
           - np.load(DATA / 'esm2_train_human_layer15_t30_150M.npy').astype(np.float32))
    m_g = np.array([clean(x) for x in np.load(ID_DIR / 'train_gene_list.npy', allow_pickle=True)])
    m_dom = None
    for cand in ['train_domain_matrix_hmmscan.npy', 'domain_matrix_proper_train.npy',
                 'domain_matrix_proper_train_v3.npy']:
        try:
            mm = np.load(FEAT / cand)
            if mm.shape[0] == m_d.shape[0]:
                m_dom = mm; print(f"muscle domain matrix: {cand} {mm.shape}"); break
        except Exception:
            pass
    if m_dom is not None:
        out['muscle'] = analyze('muscle', m_d, m_g, m_dom, celltype=None)
        print("MUSCLE:", json.dumps(out['muscle'], indent=2))
    else:
        out['muscle'] = {'note': 'no muscle domain matrix matched 31668 rows'}
        print("MUSCLE: no matching domain matrix")

    (OUT / 'alignment_axes.json').write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {OUT / 'alignment_axes.json'}")


if __name__ == '__main__':
    main()
