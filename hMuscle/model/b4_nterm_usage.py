#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""b4_nterm_usage.py

Strengthen the weakest map cell: does the N-TERMINAL targeting sub-stream (51.5% of non-domain
edits, currently "partial B4" on the strength of a weak MTS-alignment ρ=0.126) actually reach
PRISM's OUTPUT? Same instrument as b4_compositional_usage.py (directly comparable), but on the
non-domain N-TERMINAL subset (domain_binary==0 AND nterm_overlap==1), oriented by MTS/targeting
proxies of the changed N-terminal residues (net charge — mitochondrial presequences are positively
charged & amphipathic; also helix propensity and hydrophobicity):

  B3 = cv_dir_acc(D_embed , targeting-median-orient)   [embedding difference]
  B4 = cv_dir_acc(D_prism , targeting-median-orient)   [PRISM score-vector difference]

  B4 exceeds its gene-permutation null -> targeting direction USED at output (upgrade the cell)
  B4 at gene-perm null                 -> encoded/surfaced but NOT used (downgrade to "not used")

Brain primary. Read-only.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
from difflib import SequenceMatcher
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
DATA = ROOT / 'hMuscle/data'
BRAIN = DATA / 'brain_isoquant_esm2/full'
MAXLEN = 1022

HELIX = {'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,'H':1.00,
         'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,'T':0.83,'W':1.08,'Y':0.69,'V':1.06}
HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
         'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHARGE = {'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.1}


def changed_intervals(long_s, short_s):
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    ivs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'equal' and i2 > i1:
            ivs.append((i1, i2))
    return ivs


def cv_dir_acc(D, gene_id, orient):
    n = len(D); Do = D * orient[:, None]
    ug = np.unique(gene_id); lr = np.random.default_rng(42); ugc = ug.copy(); lr.shuffle(ugc)
    folds = {g: i % 5 for i, g in enumerate(ugc)}
    fid = np.array([folds[g] for g in gene_id]); correct = 0
    for k in range(5):
        te = fid == k; tr = ~te
        if tr.sum() < 5 or te.sum() == 0:
            continue
        a = Do[tr].mean(0); a /= (np.linalg.norm(a) + 1e-9)
        correct += int((np.dot(Do[te], a) > 0).sum())
    return correct / n


def gene_perm_null(D, gene_id, orient, n=10):
    accs = []
    for i in range(n):
        pr = np.random.default_rng(100 + i); g = gene_id.copy(); pr.shuffle(g)
        accs.append(cv_dir_acc(D, g, orient))
    return np.mean(accs), np.std(accs)


def main():
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bsp)
    df = pd.read_csv(SEV / 'brain_severity_pairs_scored.tsv', sep='\t')
    df = df[(df['tissue'] == 'brain') & (df['domain_binary'] == 0) & (df['nterm_overlap'] == 1)]
    iso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    L15 = np.load(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy').astype(np.float32)
    L30 = np.load(BRAIN / 'brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
    emb = np.concatenate([L15, L30], axis=1)
    prism = np.load(ROOT / 'reports/brain_full_672_scores.npy').astype(np.float32)

    Demb, Dprism, gene = [], [], []
    cov = {'charge': [], 'helix': [], 'hydro': []}
    for _, r in df.iterrows():
        li, si = int(r['long_idx']), int(r['short_idx'])
        lid, sid = iso[li], iso[si]
        if lid not in seqs or sid not in seqs:
            continue
        ls, ss = seqs[lid][:MAXLEN], seqs[sid][:MAXLEN]
        if ls == ss:
            continue
        ivs = changed_intervals(ls, ss)
        cri = [i for (u, v) in ivs for i in range(u, v) if i < len(ls)]
        if not cri:
            continue
        res = [ls[i] for i in cri]
        Demb.append(emb[li] - emb[si]); Dprism.append(prism[li] - prism[si]); gene.append(str(r['gene']))
        cov['charge'].append(sum(CHARGE.get(a, 0.0) for a in res) / len(res))
        cov['helix'].append(np.mean([HELIX.get(a, 1.0) for a in res]))
        cov['hydro'].append(np.mean([HYDRO.get(a, 1.0) for a in res]))
    Demb = np.array(Demb); Dprism = np.array(Dprism); gene = np.array(gene)
    cov = {k: np.array(v) for k, v in cov.items()}

    print("=" * 84)
    print("B4 N-TERMINAL USAGE — does the targeting sub-stream reach PRISM's output? (brain)")
    print("=" * 84)
    print(f"non-domain N-terminal pairs: n={len(Demb)}  genes={len(np.unique(gene))}")
    print(f"|PRISM score-diff| per pair: mean={np.abs(Dprism).sum(1).mean():.3f}")
    print("-" * 84)
    for name in ['charge', 'helix', 'hydro']:
        orient = np.where(cov[name] > np.median(cov[name]), 1.0, -1.0)
        b3 = cv_dir_acc(Demb, gene, orient); b3n, _ = gene_perm_null(Demb, gene, orient)
        b4 = cv_dir_acc(Dprism, gene, orient); b4n, b4sd = gene_perm_null(Dprism, gene, orient)
        used = (b4 > b4n + 2 * b4sd) and (b4 - 0.5 > 0.02)
        print(f"  [{name:6}] B3_embed={b3:.3f}(null {b3n:.3f}) | B4_output={b4:.3f}"
              f"(null {b4n:.3f}±{b4sd:.3f}) | {'USED at output' if used else 'NOT used (at gene-perm null)'}")
    print("-" * 84)
    print("verdict: if all NOT used -> N-terminal cell downgrades to 'encoded/partial-surfaced, not")
    print("used at output' (same status as compositional); MTS ρ=0.126 is a B1/B3 alignment, not B4.")


if __name__ == '__main__':
    main()
