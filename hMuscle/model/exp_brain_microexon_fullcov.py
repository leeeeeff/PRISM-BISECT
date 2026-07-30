#!/usr/bin/env python3
"""
exp_brain_microexon_fullcov.py  (Option B: microexon test at full coverage)
===========================================================================
The reference-GTF microexon test was capped at 266 CDS pairs (known-only), giving only
n=9 isolated microexon events -> directionally supportive but underpowered. Here we use
the EXTENDED IsoQuant GTF (236k transcripts incl. novel; CDS keyed by transcript_id =
ENST for known, 'transcriptNNN.chrX.xxic' for novel = brain_full novel id directly) to
recover novel isoforms, where microexons concentrate (short-read misses >40%; this is
long-read).

Re-tests (non-circular, exon structure independent of delta):
  microexon-differential = >=1 differing CDS exon 3-27nt.
  ISOLATED stratum (total aa changed <=15) : pure small events -> microexon vs non.
  prediction: isolated microexon = domain-external (low domain change) + high delta density.
"""
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import mannwhitneyu
import re, json

DATA = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/data')
BRAIN = DATA / 'brain_isoquant_esm2/full'
FEAT = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/features')
GTF_REF = DATA / 'brain_esm2/brain_only.gtf'
GTF_EXT = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/02_Isoquant_Output/extended_annotation_including_refTSS_umi10_donor3_supported_novel_tx.gtf')
OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/exp_variance_structure')


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")


def name2enst():
    d = {}
    for line in open(GTF_REF):
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
        return bid
    return n2e.get(bid, '')


def main():
    n2e = name2enst()
    genes = np.array([clean(x) for x in np.load(BRAIN / 'brain_full_gene_names.npy', allow_pickle=True)])
    ids = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    bases = [base_of(b, n2e) for b in ids]
    needed = set(b for b in bases if b)
    print(f"needed transcript bases: {len(needed)}")

    # stream extended GTF, keep CDS (start,end) for needed transcript_ids only
    cds = defaultdict(set)
    nline = 0
    for line in open(GTF_EXT):
        if "\tCDS\t" not in line:
            continue
        nline += 1
        i = line.find('transcript_id "')
        if i < 0:
            continue
        tid = line[i + 15:line.find('"', i + 15)].split('.p')[0]
        # known -> strip version ; novel keeps full 'transcriptNNN.chrX.xxic'
        tid_base = tid.split('.')[0] if tid.startswith('ENST') else tid
        if tid_base in needed:
            p = line.split('\t')
            cds[tid_base].add((int(p[3]), int(p[4])))
    print(f"scanned CDS lines={nline}, transcripts with CDS matched={len(cds)}")

    delta = (np.load(BRAIN / 'brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
             - np.load(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy').astype(np.float32))
    dommat = np.load(FEAT / 'domain_matrix_brain_full.npy')
    Z = (delta - delta.mean(0)) / (delta.std(0) + 1e-8)
    gl, ginv = np.unique(genes, return_inverse=True); cnt = np.bincount(ginv, None, len(gl))

    spread, micro, aa, domd, novelp = [], [], [], [], []
    for gi in np.where(cnt == 2)[0]:
        a, b = np.where(ginv == gi)[0]
        if np.array_equal(delta[a], delta[b]):
            continue
        ba, bb = bases[a], bases[b]
        if ba not in cds or bb not in cds:
            continue
        diff = cds[ba].symmetric_difference(cds[bb])
        if not diff:
            continue
        sizes = sorted(e - s + 1 for s, e in diff)
        spread.append(float(np.linalg.norm(Z[a] - Z[b])))
        micro.append(any(3 <= sz <= 27 for sz in sizes))
        aa.append(sum(sizes) / 3.0)
        domd.append(float(np.abs(dommat[a] - dommat[b]).sum()))
        novelp.append(ids[a].startswith('transcript') or ids[b].startswith('transcript'))
    spread = np.array(spread); micro = np.array(micro, bool); aa = np.array(aa, float)
    domd = np.array(domd); density = spread / np.maximum(aa, 1.0)
    novelp = np.array(novelp, bool)
    print(f"pairs with CDS diff = {len(spread)} (novel-involving {int(novelp.sum())})  vs ref-GTF was 266")

    def summ(m):
        if m.sum() == 0:
            return {'n': 0}
        return dict(n=int(m.sum()), mean_spread=float(spread[m].mean()),
                    median_aa=float(np.median(aa[m])), mean_density=float(density[m].mean()),
                    frac_domain_changed=float((domd[m] > 0).mean()))
    small = aa <= 15
    res = {
        'n_pairs': int(len(spread)), 'n_novel_pairs': int(novelp.sum()),
        'all_microexon': summ(micro), 'all_nonmicro': summ(~micro),
        'isolated_microexon': summ(small & micro), 'isolated_nonmicro': summ(small & (~micro)),
    }
    im, inm = small & micro, small & (~micro)
    if im.sum() >= 3 and inm.sum() >= 3:
        res['isolated_density_micro_gt_p'] = float(mannwhitneyu(density[im], density[inm], alternative='greater')[1])
        res['isolated_spread_micro_gt_p'] = float(mannwhitneyu(spread[im], spread[inm], alternative='greater')[1])
    (OUT / 'microexon_fullcov.json').write_text(json.dumps(res, indent=2))

    print("\n=== microexon full-coverage ===")
    for k in ['all_microexon', 'all_nonmicro', 'isolated_microexon', 'isolated_nonmicro']:
        d = res[k]
        if d.get('n', 0):
            print(f" {k:20s} n={d['n']:5d} spread={d['mean_spread']:.3f} aa(med)={d['median_aa']:.0f} "
                  f"density={d['mean_density']:.4f} domΔ={d['frac_domain_changed']*100:.0f}%")
    if 'isolated_density_micro_gt_p' in res:
        print(f" isolated: density micro>non p={res['isolated_density_micro_gt_p']:.2e}  spread micro>non p={res['isolated_spread_micro_gt_p']:.2e}")
    print(f"Saved -> {OUT / 'microexon_fullcov.json'}")


if __name__ == '__main__':
    main()
