#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""panelD_size_output_data.py  (paper-critic MINOR#3: make 'output = size reaction' visually checkable)

Precompute the per-pair (edit size, PRISM |ΔScore|₁, domain_binary) scatter for S_map Panel D, over
the BRAIN within-gene pair population (domain + non-domain), so the figure can show output magnitude
rising with edit size in BOTH classes → the 'output is a size reaction, not a domain-status reaction'
claim (b4_magnitude_usage.py) is verifiable by eye. Also extracts the 3 diagnostic case-study genes
(NDUFS4/MAPT/LRPPRC) in the same coordinates. Saves an npz for the figure script. Read-only inputs.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
BRAIN = ROOT / 'hMuscle/data/brain_isoquant_esm2/full'
OUT = ROOT / 'reports/model_interpretability_map'
MAXLEN = 1022
CASES = {'NDUFS4', 'MAPT', 'LRPPRC'}


def main():
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bsp)
    df = pd.read_csv(SEV / 'brain_severity_pairs_scored.tsv', sep='\t')
    df = df[df['tissue'] == 'brain'].reset_index(drop=True)
    iso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    prism = np.load(ROOT / 'reports/brain_full_672_scores.npy').astype(np.float32)

    size, mag, dom, is_case, case_gene = [], [], [], [], []
    for _, r in df.iterrows():
        li, si = int(r['long_idx']), int(r['short_idx'])
        lid, sid = iso[li], iso[si]
        if lid not in seqs or sid not in seqs:
            continue
        if seqs[lid][:MAXLEN] == seqs[sid][:MAXLEN]:
            continue
        size.append(float(r['size']))
        mag.append(float(np.abs(prism[li] - prism[si]).sum()))
        dom.append(int(r['domain_binary']))
        g = str(r['gene'])
        is_case.append(g in CASES); case_gene.append(g if g in CASES else '')
    size = np.array(size); mag = np.array(mag); dom = np.array(dom)
    is_case = np.array(is_case); case_gene = np.array(case_gene)

    # spearman size vs mag, within each class (for the annotation)
    from scipy.stats import spearmanr
    rho_all = spearmanr(size, mag).correlation
    rho_dom = spearmanr(size[dom == 1], mag[dom == 1]).correlation
    rho_nd = spearmanr(size[dom == 0], mag[dom == 0]).correlation
    print(f"n={len(size)}  domain={int((dom==1).sum())}  nondomain={int((dom==0).sum())}")
    print(f"spearman(size, |dPRISM|): all={rho_all:.3f}  domain={rho_dom:.3f}  nondomain={rho_nd:.3f}")
    print(f"case-study pairs found: {int(is_case.sum())}  genes={sorted(set(case_gene[is_case]))}")

    np.savez(OUT / 'panelD_size_output.npz', size=size, mag=mag, dom=dom,
             is_case=is_case, case_gene=case_gene,
             rho_all=rho_all, rho_dom=rho_dom, rho_nd=rho_nd)
    print(f"saved: {OUT/'panelD_size_output.npz'}")


if __name__ == '__main__':
    main()
