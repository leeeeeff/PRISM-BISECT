#!/usr/bin/env python3
"""
exp_brain_microexon_protein.py  (Option B: protein-alignment microexon, full coverage)
======================================================================================
The extended GTF has NO CDS features for novel isoforms (novel ORFs live only in the
SQANTI faa), so coordinate-based microexon detection cannot reach novel. Solution:
detect microexon-scale events at the PROTEIN level by aligning the two isoform sequences
(faa, full coverage incl novel) and finding short indel blocks (1-9 aa = 3-27 nt).
This is protein-level -> directly relevant to delta, and non-circular (indel structure
is independent of the ESM-2 delta magnitude).

Within-gene 2-iso pairs (both have faa seq):
  difflib opcodes -> insert/delete blocks. microexon-diff = any indel block 1-9 aa.
  ISOLATED (total changed aa <=15): pure small events, microexon vs non.
  prediction: isolated microexon = domain-external (low domain change) + high delta density.
"""
import numpy as np
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from scipy.stats import mannwhitneyu
import re, json

DATA = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/data')
BRAIN = DATA / 'brain_isoquant_esm2/full'
FEAT = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/features')
GTF_REF = DATA / 'brain_esm2/brain_only.gtf'
FAA = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/02_Isoquant_Output/SQANTI3_output/isoforms_corrected.faa')
OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/exp_variance_structure')


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")


def parse_faa_seq():
    best = {}; cur, seq = None, []
    def flush(bid, s):
        if bid is None: return
        ss = ''.join(s)
        if bid not in best or len(ss) > len(best[bid]):
            best[bid] = ss
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


def indel_profile(a, b):
    """difflib opcodes -> (has_microexon_indel, total_changed_aa)."""
    sm = SequenceMatcher(None, a, b, autojunk=False)
    micro = False; changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal': continue
        la, lb = i2 - i1, j2 - j1
        changed += max(la, lb)
        if tag in ('insert', 'delete'):
            sz = lb if tag == 'insert' else la
            if 1 <= sz <= 9: micro = True
        # 'replace' blocks: a short balanced replace can also be microexon-like; count if small
        elif tag == 'replace' and 1 <= max(la, lb) <= 9 and abs(la - lb) <= 9:
            micro = True
    return micro, changed


def main():
    faa = parse_faa_seq(); n2e = name2enst()
    print(f"faa seqs {len(faa)}  name2enst {len(n2e)}")
    delta = (np.load(BRAIN / 'brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
             - np.load(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy').astype(np.float32))
    genes = np.array([clean(x) for x in np.load(BRAIN / 'brain_full_gene_names.npy', allow_pickle=True)])
    ids = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    dommat = np.load(FEAT / 'domain_matrix_brain_full.npy')
    Z = (delta - delta.mean(0)) / (delta.std(0) + 1e-8)
    seqmap = {}
    for k, bid in enumerate(ids):
        base = base_of(bid, n2e)
        if base in faa: seqmap[k] = faa[base]

    gl, ginv = np.unique(genes, return_inverse=True); cnt = np.bincount(ginv, None, len(gl))
    spread, micro, aa, domd, novelp = [], [], [], [], []
    for gi in np.where(cnt == 2)[0]:
        a, b = np.where(ginv == gi)[0]
        if a not in seqmap or b not in seqmap: continue
        sa, sb = seqmap[a], seqmap[b]
        if sa == sb: continue
        m, changed = indel_profile(sa, sb)
        if changed == 0: continue
        spread.append(float(np.linalg.norm(Z[a] - Z[b]))); micro.append(m)
        aa.append(float(changed)); domd.append(float(np.abs(dommat[a] - dommat[b]).sum()))
        novelp.append(ids[a].startswith('transcript') or ids[b].startswith('transcript'))
    spread = np.array(spread); micro = np.array(micro, bool); aa = np.array(aa, float)
    domd = np.array(domd); density = spread / np.maximum(aa, 1.0); novelp = np.array(novelp, bool)
    print(f"pairs = {len(spread)} (novel-involving {int(novelp.sum())})")

    def summ(m):
        if m.sum() == 0: return {'n': 0}
        return dict(n=int(m.sum()), mean_spread=float(spread[m].mean()),
                    median_aa=float(np.median(aa[m])), mean_density=float(density[m].mean()),
                    frac_domain_changed=float((domd[m] > 0).mean()))
    small = aa <= 15
    res = {'n_pairs': int(len(spread)), 'n_novel_pairs': int(novelp.sum()),
           'all_microexon': summ(micro), 'all_nonmicro': summ(~micro),
           'isolated_microexon': summ(small & micro), 'isolated_nonmicro': summ(small & (~micro))}
    im, inm = small & micro, small & (~micro)
    if im.sum() >= 3 and inm.sum() >= 3:
        res['isolated_density_micro_gt_p'] = float(mannwhitneyu(density[im], density[inm], alternative='greater')[1])
        res['isolated_domain_micro_lt_p'] = float(mannwhitneyu(domd[im], domd[inm], alternative='less')[1])
    (OUT / 'microexon_protein.json').write_text(json.dumps(res, indent=2))
    print("\n=== protein-alignment microexon (full coverage) ===")
    for k in ['all_microexon', 'all_nonmicro', 'isolated_microexon', 'isolated_nonmicro']:
        d = res[k]
        if d.get('n', 0):
            print(f" {k:20s} n={d['n']:5d} spread={d['mean_spread']:.3f} aa(med)={d['median_aa']:.0f} "
                  f"density={d['mean_density']:.4f} domΔ={d['frac_domain_changed']*100:.0f}%")
    if 'isolated_density_micro_gt_p' in res:
        print(f" isolated: density micro>non p={res['isolated_density_micro_gt_p']:.2e}  domain micro<non p={res['isolated_domain_micro_lt_p']:.2e}")
    print(f"Saved -> {OUT / 'microexon_protein.json'}")


if __name__ == '__main__':
    main()
