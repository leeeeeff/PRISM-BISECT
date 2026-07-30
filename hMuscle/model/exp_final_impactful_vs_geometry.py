#!/usr/bin/env python3
"""
exp_final_impactful_vs_geometry.py  (Option A: FINAL discriminator)
==================================================================
brain within-gene |D15| (L15 pooled isoform difference) is 2.83x muscle. Splice-size
(parsimony) was refuted only with a token-based muscle measure. Now with RELIABLE muscle
AA sequences (isoform_cds_sequences.txt, protein, keyed by RefSeq NM matching
train_isoform_list) we settle it: for BOTH tissues compute (changed_aa, |D15|) per
within-gene 2-iso pair in identical AA units and same ESM-2 space.

DISCRIMINATOR:
  if at MATCHED changed_aa, brain |D15| >> muscle |D15|  -> REPRESENTATION-GEOMETRY
     (per-residue embedding impact is tissue-different; brain isoform changes are more
      'impactful' per aa in ESM-2 L15 space).
  if matched changed_aa gives EQUAL |D15|                -> SPLICE-SIZE (brain just splices
     more; the token muscle measure was wrong).
"""
import re, json
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from scipy.stats import mannwhitneyu

DATA = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/data')
BRAIN = DATA / 'brain_isoquant_esm2/full'
ID = DATA / 'raw_data/data/id_lists'
MSEQ = DATA / 'raw_data/data/raw_data/sequence_data/isoform_cds_sequences.txt'
GTF_REF = DATA / 'brain_esm2/brain_only.gtf'
FAA = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/02_Isoquant_Output/SQANTI3_output/isoforms_corrected.faa')
OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/exp_variance_structure')
_AA = set('ACDEFGHIKLMNPQRSTVWY')


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")


def sani(s):
    s = s.replace('*', '')
    return ''.join(c if c in _AA else 'X' for c in s)


def changed_aa(a, b):
    sm = SequenceMatcher(None, a, b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != 'equal')


def load_muscle_seqs():
    """RefSeq NM id -> protein seq (key line '>GENE|NM_xxxx')."""
    seqs = {}; cur, buf = None, []
    def flush():
        if cur and buf: seqs[cur] = sani(''.join(buf))
    for line in open(MSEQ):
        line = line.rstrip()
        if line.startswith('>'):
            flush(); buf = []
            k = line[1:].split('|')
            cur = k[1] if len(k) > 1 else k[0]
        else:
            buf.append(line)
    flush(); return seqs


def muscle_pairs():
    h15 = np.load(DATA / 'esm2_train_human_layer15_t30_150M.npy').astype(np.float32)
    g = np.array([clean(x) for x in np.load(ID / 'train_gene_list.npy', allow_pickle=True)])
    iso = np.array([clean(x) for x in np.load(ID / 'train_isoform_list.npy', allow_pickle=True)])
    seqs = load_muscle_seqs()
    print(f"muscle protein seqs {len(seqs)}")
    gl, gi = np.unique(g, return_inverse=True); cnt = np.bincount(gi, None, len(gl))
    caa, d15 = [], []
    for k in np.where(cnt == 2)[0]:
        a, b = np.where(gi == k)[0]
        sa, sb = seqs.get(iso[a]), seqs.get(iso[b])
        if not sa or not sb or sa == sb: continue
        caa.append(changed_aa(sa, sb)); d15.append(float(np.linalg.norm(h15[a] - h15[b])))
    return np.array(caa, float), np.array(d15, float)


def brain_pairs():
    h15 = np.load(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy').astype(np.float32)
    genes = np.array([clean(x) for x in np.load(BRAIN / 'brain_full_gene_names.npy', allow_pickle=True)])
    ids = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    # faa
    best = {}; cur, s = None, []
    def flush(b, seq):
        if b is None: return
        ss = sani(''.join(seq))
        if b not in best or len(ss) > len(best[b]): best[b] = ss
    for line in open(FAA):
        if line.startswith('>'):
            flush(cur, s); cur = line[1:].split()[0].split('.p')[0]; s = []
        else: s.append(line.strip())
    flush(cur, s)
    n2e = {}
    for line in open(GTF_REF):
        if "\ttranscript\t" in line:
            em = re.search(r'transcript_id "([^"]+)"', line); nm = re.search(r'transcript_name "([^"]+)"', line)
            if em and nm: n2e[nm.group(1)] = em.group(1).split('.')[0]
    def bof(bid):
        if bid.startswith('ENST'): return bid.split('.')[0]
        if bid.startswith('transcript'): return bid
        return n2e.get(bid, '')
    seqmap = {k: best[bof(b)] for k, b in enumerate(ids) if bof(b) in best}
    gl, gi = np.unique(genes, return_inverse=True); cnt = np.bincount(gi, None, len(gl))
    caa, d15 = [], []
    for k in np.where(cnt == 2)[0]:
        a, b = np.where(gi == k)[0]
        if a not in seqmap or b not in seqmap: continue
        sa, sb = seqmap[a], seqmap[b]
        if sa == sb: continue
        caa.append(changed_aa(sa, sb)); d15.append(float(np.linalg.norm(h15[a] - h15[b])))
    return np.array(caa, float), np.array(d15, float)


def main():
    mc, md = muscle_pairs(); bc, bd = brain_pairs()
    print(f"muscle pairs {len(mc)}  brain pairs {len(bc)}")
    res = {
        'muscle': {'n': len(mc), 'median_changed_aa': float(np.median(mc)), 'median_D15': float(np.median(md)),
                   'median_D15_per_aa': float(np.median(md / np.maximum(mc, 1)))},
        'brain': {'n': len(bc), 'median_changed_aa': float(np.median(bc)), 'median_D15': float(np.median(bd)),
                  'median_D15_per_aa': float(np.median(bd / np.maximum(bc, 1)))},
    }
    # matched changed_aa bins
    bins = [(10, 30), (30, 60), (60, 120), (120, 250), (250, 600)]
    res['binned_D15_median'] = {}
    for lo, hi in bins:
        mm = (mc >= lo) & (mc < hi); bb = (bc >= lo) & (bc < hi)
        entry = {'muscle_n': int(mm.sum()), 'brain_n': int(bb.sum()),
                 'muscle_D15': float(np.median(md[mm])) if mm.sum() else None,
                 'brain_D15': float(np.median(bd[bb])) if bb.sum() else None}
        if mm.sum() > 5 and bb.sum() > 5:
            entry['brain_over_muscle'] = entry['brain_D15'] / entry['muscle_D15']
            entry['MWU_brain_gt_p'] = float(mannwhitneyu(bd[bb], md[mm], alternative='greater')[1])
        res['binned_D15_median'][f'{lo}-{hi}aa'] = entry
    res['verdict'] = ('REPRESENTATION-GEOMETRY: brain |D15| higher at matched splice size'
                      if res['brain']['median_D15_per_aa'] > 1.5 * res['muscle']['median_D15_per_aa']
                      else 'SPLICE-SIZE or mixed: see binned table')
    (OUT / 'final_impactful_vs_geometry.json').write_text(json.dumps(res, indent=2))
    print("\n=== FINAL: |D15| vs splice size (both reliable AA, same ESM-2) ===")
    for t in ['muscle', 'brain']:
        d = res[t]
        print(f" {t:7s} n={d['n']:5d}  med_changed_aa={d['median_changed_aa']:.0f}  med_|D15|={d['median_D15']:.2f}  |D15|/aa={d['median_D15_per_aa']:.4f}")
    print(" matched changed_aa bins (|D15| median):")
    for k, e in res['binned_D15_median'].items():
        if e.get('brain_over_muscle'):
            print(f"  {k:10s} muscle={e['muscle_D15']:.2f}(n{e['muscle_n']}) brain={e['brain_D15']:.2f}(n{e['brain_n']})"
                  f"  b/m={e['brain_over_muscle']:.2f}  p={e['MWU_brain_gt_p']:.1e}")
    print(f" VERDICT: {res['verdict']}")
    print(f"Saved -> {OUT / 'final_impactful_vs_geometry.json'}")


if __name__ == '__main__':
    main()
