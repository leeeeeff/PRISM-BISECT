#!/usr/bin/env python3
"""
exp_brain_idr_region_confirm.py  (Option B: direct IDR-region confirmation)
===========================================================================
Prior evidence used WHOLE-PROTEIN disorder fraction difference (|Delta IDR frac| partial
0.50). This directly measures the disorder of the ACTUAL SPLICED REGION and tests whether
LARGE disordered-region splices carry the large, domain-dark variance.

For each within-gene 2-iso pair (faa seqs, full coverage incl novel):
  align (difflib) -> changed residues (insert/delete/replace segments)
  region_disorder = mean TOP-IDP over changed residues
  changed_aa      = number of changed residues
  delta_spread, domain_diff (Pfam label, NON-circular for the contrast)

Tests:
 (1) partial(delta_spread, region_disorder | changed_aa, domain) : does spliced-region
     disorder predict delta BEYOND its size and domain change?
 (2) LARGE splices (changed_aa>27, beyond microexon) split by region_disorder (>median):
     prediction -> disordered-region splices have LARGER delta and LOWER domain change
     (domain-dark) vs ordered-region splices (domain-aligned). The domain contrast is
     NOT sequence-circular (domain = Pfam).
"""
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from scipy.stats import spearmanr, rankdata, mannwhitneyu
import re, json

DATA = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/data')
BRAIN = DATA / 'brain_isoquant_esm2/full'
FEAT = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/features')
GTF_REF = DATA / 'brain_esm2/brain_only.gtf'
FAA = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/02_Isoquant_Output/SQANTI3_output/isoforms_corrected.faa')
OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/exp_variance_structure')
TOPIDP = {'A': 0.06, 'R': 0.180, 'N': 0.007, 'D': 0.192, 'C': 0.02, 'Q': 0.318,
          'E': 0.736, 'G': 0.166, 'H': 0.303, 'I': -0.486, 'L': -0.326, 'K': 0.586,
          'M': -0.397, 'F': -0.697, 'P': 0.987, 'S': 0.341, 'T': 0.059, 'W': -0.884,
          'Y': -0.510, 'V': -0.121}


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")


def parse_faa_seq():
    best = {}; cur, seq = None, []
    def flush(bid, s):
        if bid is None: return
        ss = ''.join(s)
        if bid not in best or len(ss) > len(best[bid]): best[bid] = ss
    for line in open(FAA):
        if line.startswith('>'):
            flush(cur, seq); cur = line[1:].split()[0].split('.p')[0]; seq = []
        else:
            seq.append(line.strip())
    flush(cur, seq); return best


def name2enst():
    d = {}
    for line in open(GTF_REF):
        if "\ttranscript\t" not in line: continue
        em = re.search(r'transcript_id "([^"]+)"', line); nm = re.search(r'transcript_name "([^"]+)"', line)
        if em and nm: d[nm.group(1)] = em.group(1).split('.')[0]
    return d


def base_of(bid, n2e):
    if bid.startswith('ENST'): return bid.split('.')[0]
    if bid.startswith('transcript'): return bid
    return n2e.get(bid, '')


def changed_residues(a, b):
    sm = SequenceMatcher(None, a, b, autojunk=False)
    res = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'delete' or tag == 'replace':
            res.append(a[i1:i2])
        if tag == 'insert' or tag == 'replace':
            res.append(b[j1:j2])
    return ''.join(res)


def disorder(seq):
    if not seq: return 0.0
    return float(np.mean([TOPIDP.get(c, 0.0) for c in seq]))


def resid(y, *c):
    ry = rankdata(y); X = np.ones((len(ry), 1))
    for cc in c: X = np.c_[X, rankdata(cc)]
    return ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]


def partial(a, b, *c):
    return float(np.corrcoef(resid(a, *c), resid(b, *c))[0, 1])


def main():
    faa = parse_faa_seq(); n2e = name2enst()
    delta = (np.load(BRAIN / 'brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
             - np.load(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy').astype(np.float32))
    genes = np.array([clean(x) for x in np.load(BRAIN / 'brain_full_gene_names.npy', allow_pickle=True)])
    ids = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    dommat = np.load(FEAT / 'domain_matrix_brain_full.npy')
    Z = (delta - delta.mean(0)) / (delta.std(0) + 1e-8)
    seqmap = {k: faa[base_of(b, n2e)] for k, b in enumerate(ids) if base_of(b, n2e) in faa}

    gl, ginv = np.unique(genes, return_inverse=True); cnt = np.bincount(ginv, None, len(gl))
    spread, rdis, caa, domd = [], [], [], []
    for gi in np.where(cnt == 2)[0]:
        a, b = np.where(ginv == gi)[0]
        if a not in seqmap or b not in seqmap: continue
        sa, sb = seqmap[a], seqmap[b]
        if sa == sb: continue
        ch = changed_residues(sa, sb)
        if len(ch) == 0: continue
        spread.append(float(np.linalg.norm(Z[a] - Z[b])))
        rdis.append(disorder(ch)); caa.append(len(ch))
        domd.append(float(np.abs(dommat[a] - dommat[b]).sum()))
    spread = np.array(spread); rdis = np.array(rdis); caa = np.array(caa); domd = np.array(domd)
    print(f"pairs = {len(spread)}")

    res = {'n_pairs': int(len(spread))}
    res['rho_spread_region_disorder'] = float(spearmanr(spread, rdis).correlation)
    res['rho_spread_changed_aa'] = float(spearmanr(spread, caa).correlation)
    res['partial_spread_regiondisorder_given_size'] = partial(spread, rdis, caa)
    res['partial_spread_regiondisorder_given_size_domain'] = partial(spread, rdis, caa, domd)

    # LARGE splices (beyond microexon) split by region disorder
    large = caa > 27
    med = np.median(rdis[large])
    dis_region = large & (rdis > med)
    ord_region = large & (rdis <= med)
    def summ(m):
        return dict(n=int(m.sum()), mean_spread=float(spread[m].mean()),
                    mean_changed_aa=float(caa[m].mean()),
                    frac_domain_changed=float((domd[m] > 0).mean()),
                    mean_domain_diff=float(domd[m].mean()),
                    mean_region_disorder=float(rdis[m].mean()))
    res['large_disordered_region'] = summ(dis_region)
    res['large_ordered_region'] = summ(ord_region)
    res['LARGE_disordered_spread_gt_p'] = float(mannwhitneyu(spread[dis_region], spread[ord_region], alternative='greater')[1])
    res['LARGE_disordered_domain_lt_p'] = float(mannwhitneyu(domd[dis_region], domd[ord_region], alternative='less')[1])
    (OUT / 'idr_region_confirm.json').write_text(json.dumps(res, indent=2))

    print("\n=== IDR-region direct confirmation ===")
    print(f" rho(spread, region_disorder)             = {res['rho_spread_region_disorder']:.3f}")
    print(f" rho(spread, changed_aa)                  = {res['rho_spread_changed_aa']:.3f}")
    print(f" partial(spread, region_disorder | size)  = {res['partial_spread_regiondisorder_given_size']:.4f}")
    print(f" partial(spread, region_disorder | size,dom)= {res['partial_spread_regiondisorder_given_size_domain']:.4f}")
    print(f"\n LARGE splices (changed_aa>27):")
    for k in ['large_disordered_region', 'large_ordered_region']:
        d = res[k]
        print(f"  {k:26s} n={d['n']:4d} spread={d['mean_spread']:.2f} chAA={d['mean_changed_aa']:.0f}"
              f" domΔ={d['frac_domain_changed']*100:.0f}% disorder={d['mean_region_disorder']:+.3f}")
    print(f" disordered>ordered spread p={res['LARGE_disordered_spread_gt_p']:.2e}")
    print(f" disordered<ordered domain p={res['LARGE_disordered_domain_lt_p']:.2e}  (dark = domain-external)")
    print(f"Saved -> {OUT / 'idr_region_confirm.json'}")


if __name__ == '__main__':
    main()
