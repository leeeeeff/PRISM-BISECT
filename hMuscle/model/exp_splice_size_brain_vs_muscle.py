#!/usr/bin/env python3
"""
exp_splice_size_brain_vs_muscle.py  (Option A: parsimony test)
==============================================================
delta_spread is driven by splice SIZE (changed_aa, rho=0.70; idr_region_confirm).
Parsimony hypothesis for finding (2) (brain within-gene variance 2.9x muscle): brain
simply has LARGER within-gene splices than muscle. Test directly by comparing the
within-gene splice FRACTION (changed residues / protein length) distributions.

Splice fraction is unit-invariant, so it is comparable across:
  - muscle: my_sequence_matrix_fixed.npy = AA 3-mer token seqs (vocab 8000=20^3);
            align token int-lists (no decode needed), changed_tokens/len.
  - brain : SQANTI faa AA seqs; align AA, changed_aa/len.
Both restricted to exactly-2-iso genes (matched cardinality). MWU + medians.
PREDICTION: if parsimony holds, brain splice fraction > muscle.
"""
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from scipy.stats import mannwhitneyu
import re, json

DATA = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/data')
BRAIN = DATA / 'brain_isoquant_esm2/full'
MODEL = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/model')
GTF_REF = DATA / 'brain_esm2/brain_only.gtf'
FAA = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/02_Isoquant_Output/SQANTI3_output/isoforms_corrected.faa')
OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/exp_variance_structure')


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")


def changed_frac(a, b):
    """fraction of the (longer) sequence that is in non-equal alignment blocks."""
    sm = SequenceMatcher(None, a, b, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'equal':
            changed += max(i2 - i1, j2 - j1)
    return changed / max(len(a), len(b))


def muscle_fracs():
    sm = np.load(MODEL / 'my_sequence_matrix_fixed.npy')
    g = np.array([clean(x) for x in np.load(MODEL / 'my_gene_list_fixed.npy', allow_pickle=True)])
    gl, gi = np.unique(g, return_inverse=True); cnt = np.bincount(gi, None, len(gl))
    fr = []
    for k in np.where(cnt == 2)[0]:
        a, b = np.where(gi == k)[0]
        ta = sm[a][sm[a] != 0].tolist(); tb = sm[b][sm[b] != 0].tolist()
        if not ta or not tb or ta == tb:
            continue
        # token seqs must be lists of hashable ints -> map to chars via tuple not needed; difflib on ints
        fr.append(changed_frac(ta, tb))
    return np.array(fr)


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


def brain_fracs():
    faa = parse_faa_seq(); n2e = name2enst()
    genes = np.array([clean(x) for x in np.load(BRAIN / 'brain_full_gene_names.npy', allow_pickle=True)])
    ids = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    seqmap = {k: faa[base_of(b, n2e)] for k, b in enumerate(ids) if base_of(b, n2e) in faa}
    gl, gi = np.unique(genes, return_inverse=True); cnt = np.bincount(gi, None, len(gl))
    fr = []
    for k in np.where(cnt == 2)[0]:
        a, b = np.where(gi == k)[0]
        if a not in seqmap or b not in seqmap: continue
        sa, sb = seqmap[a], seqmap[b]
        if sa == sb: continue
        fr.append(changed_frac(sa, sb))
    return np.array(fr)


def main():
    mf = muscle_fracs(); bf = brain_fracs()
    def desc(x):
        return dict(n=int(len(x)), median=float(np.median(x)), mean=float(x.mean()),
                    q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)))
    res = {'muscle_splice_fraction': desc(mf), 'brain_splice_fraction': desc(bf),
           'MWU_brain_gt_muscle_p': float(mannwhitneyu(bf, mf, alternative='greater')[1]),
           'brain_over_muscle_median': float(np.median(bf) / np.median(mf))}
    (OUT / 'splice_size_brain_vs_muscle.json').write_text(json.dumps(res, indent=2))
    print("=== within-gene splice FRACTION (changed residues / length), exactly-2-iso ===")
    for t, d in [('muscle', res['muscle_splice_fraction']), ('brain', res['brain_splice_fraction'])]:
        print(f" {t:7s} n={d['n']:5d}  median={d['median']:.3f}  mean={d['mean']:.3f}  IQR[{d['q25']:.3f},{d['q75']:.3f}]")
    print(f" brain/muscle median ratio = {res['brain_over_muscle_median']:.2f}")
    print(f" MWU brain>muscle p = {res['MWU_brain_gt_muscle_p']:.2e}")
    print(f"Saved -> {OUT / 'splice_size_brain_vs_muscle.json'}")


if __name__ == '__main__':
    main()
