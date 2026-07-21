#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""b4_compositional_usage.py

B4 (usage) test for the compositional covariates (helix/sheet/hydro/charge), method (a):
does the compositional within-gene direction that is present in the EMBEDDING (B3:
CV-dir-acc 0.59-0.65 on the non-domain internal-edit subset) propagate to PRISM's OUTPUT?

Same instrument for B3 and B4 (parallel, directly comparable):
  B3 = cv_dir_acc( D_embed , compositional-median-orient )   [embedding difference; reproduces prior]
  B4 = cv_dir_acc( D_prism , compositional-median-orient )   [PRISM score-vector difference; NEW]

Interpretation:
  B4 ~= B3  -> compositional signal reaches PRISM output = USED (B4 pass)
  B4 <<  B3 -> present in embedding, absent from output = encoded/surfaced but NOT used
               (the describability-gap thesis, made quantitative)

Discipline (from C3 gate): BRAIN is primary/clean, muscle is a noisier upper bound; require
cross-tissue consistency; significance vs gene-permutation null.

PRISM scores: muscle v17f* 82-MF ensemble; brain full 672. Indices match severity_pairs.
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
rng = np.random.default_rng(42)

HELIX = {'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,'H':1.00,
         'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,'T':0.83,'W':1.08,'Y':0.69,'V':1.06}
SHEET = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,'H':0.87,
         'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,'T':1.19,'W':1.37,'Y':1.47,'V':1.70}
HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
         'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHARGE = {'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.1}


def changed_intervals(long_s, short_s):
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    ivs, changed = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        changed += max(i2 - i1, j2 - j1)
        if i2 > i1:
            ivs.append((i1, i2))
    return ivs, changed


def cv_dir_acc(D, gene_id, orient):
    n = len(D)
    Do = D * orient[:, None]
    ug = np.unique(gene_id)
    lr = np.random.default_rng(42); ugc = ug.copy(); lr.shuffle(ugc)
    folds = {g: i % 5 for i, g in enumerate(ugc)}
    fid = np.array([folds[g] for g in gene_id])
    correct = 0
    for k in range(5):
        te = fid == k; tr = ~te
        if tr.sum() < 5 or te.sum() == 0:
            continue
        a = Do[tr].mean(0); a /= (np.linalg.norm(a) + 1e-9)
        correct += int((np.dot(Do[te], a) > 0).sum())
    acc = correct / n
    return acc, acc - 1.96*np.sqrt(acc*(1-acc)/n), acc + 1.96*np.sqrt(acc*(1-acc)/n)


def gene_perm_null(D, gene_id, orient, n=10):
    accs = []
    for i in range(n):
        pr = np.random.default_rng(100+i); g = gene_id.copy(); pr.shuffle(g)
        accs.append(cv_dir_acc(D, g, orient)[0])
    return np.mean(accs), np.std(accs)


def load(tissue):
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bsp)
    df = pd.read_csv(SEV / f'{tissue}_severity_pairs_scored.tsv', sep='\t')
    df = df[df['tissue'] == tissue].reset_index(drop=True)
    if tissue == 'muscle':
        iso = np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)
        iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
        seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
        L15 = np.load(DATA / 'esm2_layer_15_t30_150M.npy').astype(np.float32)
        L30 = np.load(DATA / 'esm2_layer_30_t30_150M.npy').astype(np.float32)
        prism = np.load(ROOT / 'reports/v17f_bootstrap/v17f_preds_ensemble.npy').astype(np.float32)
    else:
        iso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
        seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
        L15 = np.load(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy').astype(np.float32)
        L30 = np.load(BRAIN / 'brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
        prism = np.load(ROOT / 'reports/brain_full_672_scores.npy').astype(np.float32)
    emb = np.concatenate([L15, L30], axis=1)
    return df, iso, seqs, emb, prism


def build(df, iso, seqs, emb, prism):
    sub = df[(df['domain_binary'] == 0) & (df['nterm_overlap'] == 0)]
    Demb, Dprism, gene, cov = [], [], [], {'helix': [], 'sheet': [], 'hydro': [], 'charge': []}
    for _, r in sub.iterrows():
        li, si = int(r['long_idx']), int(r['short_idx'])
        lid, sid = iso[li], iso[si]
        if lid not in seqs or sid not in seqs:
            continue
        ls, ss = seqs[lid][:MAXLEN], seqs[sid][:MAXLEN]
        if ls == ss:
            continue
        ivs, ch = changed_intervals(ls, ss)
        if ch == 0 or not ivs:
            continue
        cri = [i for (u, v) in ivs for i in range(u, v) if i < len(ls)]
        if not cri:
            continue
        Demb.append(emb[li] - emb[si])
        Dprism.append(prism[li] - prism[si])
        gene.append(str(r['gene']))
        cov['helix'].append(np.mean([HELIX.get(ls[i], 1.0) for i in cri]))
        cov['sheet'].append(np.mean([SHEET.get(ls[i], 1.0) for i in cri]))
        cov['hydro'].append(np.mean([HYDRO.get(ls[i], 1.0) for i in cri]))
        cov['charge'].append(sum(CHARGE.get(ls[i], 0.0) for i in cri) / len(cri))
    return (np.array(Demb), np.array(Dprism), np.array(gene),
            {k: np.array(v) for k, v in cov.items()})


print("="*82)
print("B4 COMPOSITIONAL USAGE: does the B3 embedding direction reach PRISM's output?")
print("="*82)

results = {}
for tissue in ['muscle', 'brain']:
    print(f"\n[{tissue.upper()}]  (PRISM {'82-MF v17f*' if tissue=='muscle' else '672'})")
    print("-"*82)
    df, iso, seqs, emb, prism = load(tissue)
    Demb, Dprism, gene, cov = build(df, iso, seqs, emb, prism)
    print(f"  non-domain internal-edit pairs: n={len(Demb)}")
    # PRISM output difference magnitude on this subset (describability-gap check)
    print(f"  |PRISM score-diff| per pair: mean={np.abs(Dprism).sum(1).mean():.4f}  "
          f"(mean per-term |diff|={np.abs(Dprism).mean():.5f})")
    results[tissue] = {}
    for name in ['helix', 'sheet', 'hydro', 'charge']:
        orient = np.where(cov[name] > np.median(cov[name]), 1.0, -1.0)
        b3, b3lo, _ = cv_dir_acc(Demb, gene, orient)
        b3n, _ = gene_perm_null(Demb, gene, orient)
        b4, b4lo, b4hi = cv_dir_acc(Dprism, gene, orient)
        b4n, b4nsd = gene_perm_null(Dprism, gene, orient)
        # fraction of B3-above-null signal that survives to B4
        frac = (b4 - 0.5) / (b3 - 0.5) if b3 > 0.5 else np.nan
        used = (b4lo > b4n + 2*b4nsd) and (b4 - 0.5 > 0.02)
        print(f"  [{name:6}] B3_embed={b3:.3f}(null {b3n:.3f}) | B4_output={b4:.3f}"
              f"(null {b4n:.3f}±{b4nsd:.3f}) | output/embed signal={frac:+.0%} | "
              f"{'USED' if used else 'not used at output'}")
        results[tissue][name] = (b3, b4, b4n, b4nsd, frac, used)

print("\n" + "="*82)
print("VERDICT (brain primary; cross-tissue consistency required)")
print("="*82)
for name in ['helix', 'sheet', 'hydro', 'charge']:
    mb3, mb4, mn, msd, mf, mu = results['muscle'][name]
    bb3, bb4, bn, bsd, bf, bu = results['brain'][name]
    cross = mu and bu
    print(f"  {name:6}: brain output/embed={bf:+.0%} ({'USED' if bu else 'not'}), "
          f"muscle={mf:+.0%} ({'USED' if mu else 'not'}) -> "
          f"{'CROSS-TISSUE USED' if cross else 'not cross-tissue used (encoded/surfaced, not used at output)'}")
