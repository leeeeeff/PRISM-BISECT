#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_brain_labelgap_rate.py  (Option B: genome-wide label-describability RATE)
=============================================================================
Converts the §6 "label-describability limitation" from an existence proof (3 BISECT
cases) into a genome-wide rate, on an UNBIASED instrument (all brain within-gene 2-iso
pairs), not the domain-loss-selected BISECT set.

For each brain within-gene 2-isoform pair:
  - changed region(s) via difflib on faa protein sequences (AA coords)
  - Pfam domain ENVELOPES from hmmscan domtbl (env coords = cols 20,21 on the query)
  - DOMAIN-AFFECTING if any changed interval overlaps any domain envelope (either isoform);
    else DOMAIN-PRESERVING (non-domain splice) = a change domain-centric labels cannot describe.
  - null: random-placement expectation = domain-covered fraction of the protein (if a splice
    of the observed size were dropped at random, P(hit domain) ≈ domain coverage). Compare
    observed domain-affecting rate to this null.
  - non-domain sub-classification: N-terminal (start ≤ 60 aa, targeting/MTS region),
    C-terminal (end ≥ len−60, truncation/PTC-adjacent), internal; disorder (TOP-IDP) of
    the changed residues.

PREDICTION (pre-registered): domain-preserving (non-domain) is a MAJORITY of splicing pairs,
and observed domain-affecting rate <= random-placement null (splices do not target domains).
Refutation: domain-affecting >> null.
"""
import os, re, json, glob
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

DATA = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/data')
BRAIN = DATA / 'brain_isoquant_esm2/full'
GTF_REF = DATA / 'brain_esm2/brain_only.gtf'
FAA = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
DOMTBL = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/data/hmmscan_out')
OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/exp_variance_structure')
_AA = set('ACDEFGHIKLMNPQRSTVWY')
TOPIDP = {'A': 0.06, 'R': 0.180, 'N': 0.007, 'D': 0.192, 'C': 0.02, 'Q': 0.318, 'E': 0.736,
          'G': 0.166, 'H': 0.303, 'I': -0.486, 'L': -0.326, 'K': 0.586, 'M': -0.397,
          'F': -0.697, 'P': 0.987, 'S': 0.341, 'T': 0.059, 'W': -0.884, 'Y': -0.510, 'V': -0.121}
IEVAL_MAX = 0.01
NTERM = 60          # targeting/MTS region window
CTERM = 60          # C-terminal window


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(' ', '')


def sani(s):
    s = s.replace('*', '')
    return ''.join(c if c in _AA else 'X' for c in s)


def strip_orf(name):
    return re.sub(r'\.p\d+$', '', name)


def parse_faa():
    """base -> (best_fullname, seq) keeping the longest ORF per transcript base."""
    best = {}
    cur_full, buf = None, []

    def flush():
        if cur_full is None:
            return
        seq = sani(''.join(buf))
        base = strip_orf(cur_full)
        if base not in best or len(seq) > len(best[base][1]):
            best[base] = (cur_full, seq)
    for line in open(FAA):
        if line.startswith('>'):
            flush()
            cur_full = line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    flush()
    return best


def parse_domtbl():
    """fullname -> list of (env_from, env_to) confident domain envelopes."""
    dom = defaultdict(list)
    for f in sorted(glob.glob(str(DOMTBL / '*.domtbl'))):
        for line in open(f):
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) < 23:
                continue
            try:
                ievalue = float(p[12])
                q = p[3]
                ef, et = int(p[19]), int(p[20])   # env from/to (cols 20,21, 0-based idx 19,20)
            except (ValueError, IndexError):
                continue
            if ievalue <= IEVAL_MAX and et > ef:
                dom[q].append((ef, et))
    return dom


def name2enst():
    d = {}
    for line in open(GTF_REF):
        if "\ttranscript\t" not in line:
            continue
        em = re.search(r'transcript_id "([^"]+)"', line)
        nm = re.search(r'transcript_name "([^"]+)"', line)
        if em and nm:
            d[nm.group(1)] = em.group(1).split('.')[0]
    return d


def base_of(bid, n2e):
    # brain_full_proteins.fa is keyed by the SAME transcript-name base as brain_full_ids;
    # strip any ORF suffix and match directly (no ENST/GTF remap needed).
    return strip_orf(clean(bid))


def overlaps(iv, doms):
    for (df, dt) in doms:
        if iv[0] < dt and df < iv[1]:
            return True
    return False


def changed_intervals(a, b):
    """intervals on a (0-based) that differ from b, and total changed_aa."""
    sm = SequenceMatcher(None, a, b, autojunk=False)
    iva, ivb, changed = [], [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        changed += max(i2 - i1, j2 - j1)
        if i2 > i1:
            iva.append((i1, i2))
        if j2 > j1:
            ivb.append((j1, j2))
    return iva, ivb, changed


def dom_cov_frac(L, doms):
    if L == 0:
        return 0.0
    cov = np.zeros(L, bool)
    for (df, dt) in doms:
        cov[max(0, df - 1):min(L, dt)] = True
    return cov.mean()


def main():
    faa = parse_faa()
    dom = parse_domtbl()
    n2e = name2enst()
    print(f"faa bases {len(faa)}  domtbl ORFs {len(dom)}", flush=True)

    genes = np.array([clean(x) for x in np.load(BRAIN / 'brain_full_gene_names.npy', allow_pickle=True)])
    ids = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    # map brain isoform idx -> (fullname, seq, domains)
    rec = {}
    for k, bid in enumerate(ids):
        base = base_of(bid, n2e)
        if base in faa:
            full, seq = faa[base]
            if 20 <= len(seq) <= 5000:
                rec[k] = (full, seq, dom.get(full, []))
    gl, gi = np.unique(genes, return_inverse=True)
    cnt = np.bincount(gi, minlength=len(gl))

    n_pairs = n_splice = n_domaff = 0
    null_hits = 0.0
    nterm = cterm = internal = 0
    dis_nondom, dis_domaff = [], []
    size_nondom, size_domaff = [], []
    examples = []
    for g in np.where(cnt == 2)[0]:
        a, b = np.where(gi == g)[0]
        if a not in rec or b not in rec:
            continue
        (_, sa, da), (_, sb, db) = rec[a], rec[b]
        if sa == sb:
            continue
        n_pairs += 1
        iva, ivb, changed = changed_intervals(sa, sb)
        if changed == 0:
            continue
        n_splice += 1
        hit = any(overlaps(iv, da) for iv in iva) or any(overlaps(iv, db) for iv in ivb)
        # random-placement null: prob a change of this size hits a domain ~ domain coverage
        cov = 0.5 * (dom_cov_frac(len(sa), da) + dom_cov_frac(len(sb), db))
        null_hits += cov
        # disorder + size of the changed region (use isoform a's intervals if any, else b)
        ivs, seq = (iva, sa) if iva else (ivb, sb)
        residues = [i for (x, y) in ivs for i in range(x, y)]
        dis = float(np.mean([TOPIDP.get(seq[i], 0.0) for i in residues])) if residues else 0.0
        if hit:
            n_domaff += 1
            dis_domaff.append(dis)
            size_domaff.append(changed)
        else:
            dis_nondom.append(dis)
            size_nondom.append(changed)
            # position class (on the changed intervals of the longer-changed isoform)
            starts = min(x for (x, y) in ivs)
            ends = max(y for (x, y) in ivs)
            L = len(seq)
            if starts <= NTERM:
                nterm += 1
            elif ends >= L - CTERM:
                cterm += 1
            else:
                internal += 1
            if len(examples) < 15:
                examples.append({'gene': gl[g], 'changed_aa': changed,
                                 'start': int(starts), 'end': int(ends), 'len': L, 'disorder': round(dis, 3)})

    nd = n_splice - n_domaff
    res = {
        'n_pairs_2iso_mapped': n_pairs,
        'n_splicing_pairs': n_splice,
        'domain_affecting': n_domaff,
        'domain_preserving_nondomain': nd,
        'nondomain_rate': nd / n_splice if n_splice else None,
        'observed_domain_affecting_rate': n_domaff / n_splice if n_splice else None,
        'random_placement_null_rate': null_hits / n_splice if n_splice else None,
        'nondomain_position': {'N_terminal_<=60aa': nterm, 'C_terminal_last60': cterm, 'internal': internal},
        'disorder_nondomain_median': float(np.median(dis_nondom)) if dis_nondom else None,
        'disorder_domainaffecting_median': float(np.median(dis_domaff)) if dis_domaff else None,
        'splice_size_nondomain_median': float(np.median(size_nondom)) if size_nondom else None,
        'splice_size_domainaffecting_median': float(np.median(size_domaff)) if size_domaff else None,
        'ieval_max': IEVAL_MAX,
        'examples_nondomain': examples,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'brain_labelgap_rate.json').write_text(json.dumps(res, indent=2, default=str))
    print("\n=== brain within-gene label-describability RATE (unbiased, all 2-iso pairs) ===")
    print(f" mapped 2-iso pairs: {n_pairs}   splicing pairs: {n_splice}")
    print(f" DOMAIN-PRESERVING (non-domain): {nd}/{n_splice} = {res['nondomain_rate']:.3f}")
    print(f" domain-affecting (observed):    {n_domaff}/{n_splice} = {res['observed_domain_affecting_rate']:.3f}")
    print(f" random-placement NULL rate:     {res['random_placement_null_rate']:.3f}  "
          f"(observed {res['observed_domain_affecting_rate']:.3f} {'<=' if res['observed_domain_affecting_rate']<=res['random_placement_null_rate'] else '>'} null)")
    print(f" non-domain position: N-term {nterm}  C-term {cterm}  internal {internal}")
    print(f" disorder(TOP-IDP) non-domain {res['disorder_nondomain_median']} vs domain-affecting {res['disorder_domainaffecting_median']}")
    print(f" splice size non-domain {res['splice_size_nondomain_median']} vs domain-affecting {res['splice_size_domainaffecting_median']}")
    print(f"[saved] {OUT / 'brain_labelgap_rate.json'}")


if __name__ == '__main__':
    main()
