#!/usr/bin/env python3
"""
exp_brain_perresidue_sensitivity.py  (Option A: per-residue delta-sensitivity)
==============================================================================
Brain within-gene delta is 2.82x muscle but splices are NOT larger -> higher delta per
residue. Hypothesis: brain isoforms splice at positions of high per-residue L15->L30
movement m(r)=||h_L30(r)-h_L15(r)||. Test: within each protein, is m(r) in the SPLICED
region higher than in the SHARED backbone? If yes -> the 2.82x is LOCATION-driven
(splices land on high-delta-sensitivity residues), not size-driven.

Also: what makes m(r) high? correlate per-residue m with TOP-IDP disorder + terminal
position -> characterize the sensitivity structure.

Focused ESM-2 t30_150M run (GPU1), per-residue reps at layers 15 & 30, store only the
scalar m(r) profile. Pairs: brain 2-iso, both faa seq, both len<=1022, sample<=700.
"""
import os, re, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
import torch, esm

DATA = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/data')
BRAIN = DATA / 'brain_isoquant_esm2/full'
GTF_REF = DATA / 'brain_esm2/brain_only.gtf'
FAA = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/02_Isoquant_Output/SQANTI3_output/isoforms_corrected.faa')
OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/exp_variance_structure')
MAXLEN = 1022; BATCH = 16; N_PAIRS = 700
TOPIDP = {'A': 0.06, 'R': 0.180, 'N': 0.007, 'D': 0.192, 'C': 0.02, 'Q': 0.318, 'E': 0.736,
          'G': 0.166, 'H': 0.303, 'I': -0.486, 'L': -0.326, 'K': 0.586, 'M': -0.397,
          'F': -0.697, 'P': 0.987, 'S': 0.341, 'T': 0.059, 'W': -0.884, 'Y': -0.510, 'V': -0.121}


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")


_AA = set('ACDEFGHIKLMNPQRSTVWY')
def sanitize(ss):
    ss = ss.replace('*', '')
    return ''.join(c if c in _AA else 'X' for c in ss)

def parse_faa():
    best = {}; cur, s = None, []
    def flush(b, seq):
        if b is None: return
        ss = sanitize(''.join(seq))
        if b not in best or len(ss) > len(best[b]): best[b] = ss
    for line in open(FAA):
        if line.startswith('>'):
            flush(cur, s); cur = line[1:].split()[0].split('.p')[0]; s = []
        else: s.append(line.strip())
    flush(cur, s); return best


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


def main():
    faa = parse_faa(); n2e = name2enst()
    genes = np.array([clean(x) for x in np.load(BRAIN / 'brain_full_gene_names.npy', allow_pickle=True)])
    ids = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    seqmap = {}
    for k, bid in enumerate(ids):
        b = base_of(bid, n2e)
        if b in faa and len(faa[b]) <= MAXLEN and len(faa[b]) >= 20:
            seqmap[k] = faa[b]
    gl, gi = np.unique(genes, return_inverse=True); cnt = np.bincount(gi, None, len(gl))
    pairs = []
    rng = np.random.default_rng(42)
    cand = np.where(cnt == 2)[0]; rng.shuffle(cand)
    for k in cand:
        a, b = np.where(gi == k)[0]
        if a in seqmap and b in seqmap and seqmap[a] != seqmap[b]:
            pairs.append((a, b))
        if len(pairs) >= N_PAIRS: break
    uniq = sorted(set([x for p in pairs for x in p]))
    print(f"pairs={len(pairs)} unique isoforms={len(uniq)}", flush=True)

    dev = 'cuda'
    model, alph = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.to(dev).eval(); bc = alph.get_batch_converter()
    mprof = {}  # isoform idx -> per-residue m(r)
    t0 = time.time()
    uniq_sorted = sorted(uniq, key=lambda k: len(seqmap[k]))
    with torch.no_grad():
        for s in range(0, len(uniq_sorted), BATCH):
            chunk = uniq_sorted[s:s + BATCH]
            data = [(str(k), seqmap[k]) for k in chunk]
            _, _, toks = bc(data); toks = toks.to(dev)
            out = model(toks, repr_layers=[15, 30], return_contacts=False)
            r15 = out['representations'][15]; r30 = out['representations'][30]
            m = torch.norm(r30 - r15, dim=-1)  # (B, L+2)
            for j, k in enumerate(chunk):
                L = len(seqmap[k])
                mprof[k] = m[j, 1:L + 1].float().cpu().numpy()
            if s % (BATCH * 20) == 0:
                print(f"  {s}/{len(uniq_sorted)}  {time.time()-t0:.0f}s", flush=True)
    print(f"ESM-2 done {time.time()-t0:.0f}s", flush=True)

    # per-pair: spliced vs shared m(r)
    spliced_m, shared_m, spliced_dis, shared_dis = [], [], [], []
    all_m, all_dis = [], []
    for a, b in pairs:
        sa, sb = seqmap[a], seqmap[b]; ma, mb = mprof[a], mprof[b]
        sm = SequenceMatcher(None, sa, sb, autojunk=False)
        spa = np.zeros(len(sa), bool); spb = np.zeros(len(sb), bool)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ('delete', 'replace'): spa[i1:i2] = True
            if tag in ('insert', 'replace'): spb[j1:j2] = True
        for seq, mm, sp in [(sa, ma, spa), (sb, mb, spb)]:
            dis = np.array([TOPIDP.get(c, 0.0) for c in seq])
            if sp.any():
                spliced_m.append(float(mm[sp].mean())); spliced_dis.append(float(dis[sp].mean()))
            if (~sp).any():
                shared_m.append(float(mm[~sp].mean())); shared_dis.append(float(dis[~sp].mean()))
            all_m.append(mm); all_dis.append(dis)
    spliced_m = np.array(spliced_m); shared_m = np.array(shared_m)
    from scipy.stats import wilcoxon, spearmanr
    allm = np.concatenate(all_m); alld = np.concatenate(all_dis)
    n = min(len(spliced_m), len(shared_m))
    res = {
        'n_pairs': len(pairs), 'n_isoforms': len(uniq),
        'mean_m_spliced': float(spliced_m.mean()), 'mean_m_shared': float(shared_m.mean()),
        'median_m_spliced': float(np.median(spliced_m)), 'median_m_shared': float(np.median(shared_m)),
        'spliced_over_shared_ratio': float(spliced_m.mean() / shared_m.mean()),
        'wilcoxon_spliced_gt_shared_p': float(wilcoxon(spliced_m[:n], shared_m[:n], alternative='greater')[1]) if n > 10 else None,
        'rho_m_vs_disorder_perres': float(spearmanr(allm, alld).correlation),
        'mean_disorder_spliced': float(np.mean(spliced_dis)), 'mean_disorder_shared': float(np.mean(shared_dis)),
    }
    (OUT / 'perresidue_sensitivity.json').write_text(json.dumps(res, indent=2))
    print("\n=== per-residue delta-sensitivity ===")
    print(f" m(r) spliced={res['mean_m_spliced']:.3f}  shared={res['mean_m_shared']:.3f}  ratio={res['spliced_over_shared_ratio']:.3f}")
    print(f" wilcoxon spliced>shared p={res['wilcoxon_spliced_gt_shared_p']}")
    print(f" rho(m, disorder) per-residue = {res['rho_m_vs_disorder_perres']:.3f}")
    print(f" disorder spliced={res['mean_disorder_spliced']:+.3f} shared={res['mean_disorder_shared']:+.3f}")
    print(f"Saved -> {OUT / 'perresidue_sensitivity.json'}")


if __name__ == '__main__':
    main()
