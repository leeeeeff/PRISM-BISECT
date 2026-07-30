#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""explore_internal_edit_full_population.py

Breaks the 2-isoform-gene limitation of exp_nondomain_anchor_decomp.py /
explore_internal_edit_covariates.py (which used only muscle_2iso.fa, n=1505
non-domain pairs, n=627 internal-edit) by rerunning the SAME analysis
(N-terminal positive control, internal-edit chance baseline, hydrophobic-
composition subgroup clustering, size-matched confound check) on the FULL
canonical-anchored multi-isoform population, BOTH tissues:
  reports/severity_pairs/{muscle,brain}_severity_pairs_scored.tsv
  (muscle 15,885 pairs / 5,941+ genes; brain 33,802 pairs / ~9,958 genes)

Population definitions (adapted from the 2-iso version):
  - "non-domain" = domain_binary==0 (long/short have IDENTICAL Pfam domain
    SETS) -- note: STRICTER than the 2-iso version's domain-COUNT equality
    (dom[a]==dom[b], which allows same count/different domains). Flagged as
    a definitional refinement, not an oversight.
  - N-terminal edit = nterm_overlap==1 (already computed in the pair table).
  - internal edit = nterm_overlap==0.
  - helix/sheet/hydro/charge deltas: recomputed via alignment (opcode not
    persisted in the original table) over the LONG sequence's changed
    residues, IDENTICAL convention to disorder_frac's own computation in
    build_severity_pairs.py.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
from difflib import SequenceMatcher
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
DATA = ROOT / 'hMuscle/data'
BRAIN = DATA / 'brain_isoquant_esm2/full'
MAXLEN = 1022
rng = np.random.default_rng(42)

HELIX = {'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,'H':1.00,
         'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,'T':0.83,'W':1.08,
         'Y':0.69,'V':1.06}
SHEET = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,'H':0.87,
         'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,'T':1.19,'W':1.37,
         'Y':1.47,'V':1.70}
HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
         'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,
         'Y':-1.3,'V':4.2}
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


def coherence(D):
    U = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    R = float(np.linalg.norm(U.mean(0)))
    nulls = []
    for _ in range(1000):
        s = rng.choice([-1.0, 1.0], size=len(U))[:, None]
        nulls.append(np.linalg.norm((U * s).mean(0)))
    nulls = np.array(nulls)
    return R, float(nulls.mean()), float(np.percentile(nulls, 97.5)), float((nulls >= R).mean())


def cv_dir_acc(D, gene_id, orient):
    n = len(D)
    Do = D * orient[:, None]
    ug = np.unique(gene_id)
    rng.shuffle(ug)
    folds = {g: i % 5 for i, g in enumerate(ug)}
    fid = np.array([folds[g] for g in gene_id])
    correct = 0
    for k in range(5):
        te = fid == k; tr = ~te
        if tr.sum() < 5 or te.sum() == 0:
            continue
        a = Do[tr].mean(0); a /= (np.linalg.norm(a) + 1e-9)
        pred = np.dot(Do[te], a) > 0
        correct += int(pred.sum())
    acc = correct / n
    se = np.sqrt(acc * (1 - acc) / n)
    return acc, acc - 1.96 * se, acc + 1.96 * se


def test_candidate(name, D_int, gene_int, cov_int):
    orient = np.where(cov_int > np.median(cov_int), 1.0, -1.0)
    R, nmean, nhi, p_R = coherence(D_int * orient[:, None])
    acc, lo, hi = cv_dir_acc(D_int, gene_int, orient)
    verdict = 'H_explain (>0.5 CI)' if lo > 0.5 else ('H_null (chance)' if lo <= 0.5 <= hi else 'below chance(?)')
    print(f"  [{name:<20}] R={R:.3f} (null {nmean:.3f}, p={p_R:.3f}) | "
          f"CV-dir-acc={acc:.3f} [{lo:.3f},{hi:.3f}]  => {verdict}")


def load_pairs_and_seqs(tissue):
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsp)

    df = pd.read_csv(SEV / f'{tissue}_severity_pairs_scored.tsv', sep='\t')
    df = df[df['tissue'] == tissue].reset_index(drop=True)
    if tissue == 'muscle':
        iso = np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)
        iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
        seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
        L15 = np.load(DATA / 'esm2_layer_15_t30_150M.npy').astype(np.float32)
        L30 = np.load(DATA / 'esm2_layer_30_t30_150M.npy').astype(np.float32)
    else:
        iso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
        seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
        L15 = np.load(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy').astype(np.float32)
        L30 = np.load(BRAIN / 'brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
    emb = np.concatenate([L15, L30], axis=1)
    return df, iso, seqs, emb


def build_group(df, iso, seqs, emb, domain0, nterm_val):
    sub = df[(df['domain_binary'] == domain0) & (df['nterm_overlap'] == nterm_val)]
    D, gene_id, size_arr, helix_d, sheet_d, hydro_d, charge_d = [], [], [], [], [], [], []
    n_missing = 0
    for _, r in sub.iterrows():
        long_id, short_id = iso[int(r['long_idx'])], iso[int(r['short_idx'])]
        if long_id not in seqs or short_id not in seqs:
            n_missing += 1; continue
        long_s, short_s = seqs[long_id][:MAXLEN], seqs[short_id][:MAXLEN]
        if long_s == short_s:
            n_missing += 1; continue
        ivs, changed = changed_intervals(long_s, short_s)
        if changed == 0 or not ivs:
            n_missing += 1; continue
        changed_res_idx = [i for (u, v) in ivs for i in range(u, v) if i < len(long_s)]
        if not changed_res_idx:
            n_missing += 1; continue
        D.append(emb[int(r['long_idx'])] - emb[int(r['short_idx'])])
        gene_id.append(r['gene']); size_arr.append(changed)
        helix_d.append(np.mean([HELIX.get(long_s[i], 1.0) for i in changed_res_idx]))
        sheet_d.append(np.mean([SHEET.get(long_s[i], 1.0) for i in changed_res_idx]))
        hydro_d.append(np.mean([HYDRO.get(long_s[i], 0.0) for i in changed_res_idx]))
        charge_d.append(np.mean([CHARGE.get(long_s[i], 0.0) for i in changed_res_idx]))
    return (np.array(D), np.array(gene_id), np.array(size_arr), np.array(helix_d),
            np.array(sheet_d), np.array(hydro_d), np.array(charge_d), n_missing, len(sub))


def size_matched_check(D_int, gene_int, size_int, cluster_label, name_pos, name_neg):
    print(f"\n  --- size-matched: {name_pos} vs {name_neg} ---")
    edges = np.percentile(size_int, [0, 33.33, 66.67, 100])
    for bi in range(3):
        lo_e, hi_e = edges[bi], edges[bi + 1]
        inbin = (size_int >= lo_e) & (size_int <= hi_e)
        cell = {}
        for lab, m in [(name_pos, cluster_label & inbin), (name_neg, (~cluster_label) & inbin)]:
            if m.sum() < 15:
                cell[lab] = None; continue
            R, nmean, nhi, p_R = coherence(D_int[m])
            acc, lo, hi = cv_dir_acc(D_int[m], gene_int[m], np.ones(m.sum()))
            cell[lab] = (m.sum(), acc, lo, hi)
        pos, neg = cell.get(name_pos), cell.get(name_neg)
        if pos and neg:
            print(f"    size {lo_e:.0f}-{hi_e:.0f}aa | {name_pos}(n={pos[0]}) acc={pos[1]:.3f}[{pos[2]:.3f},{pos[3]:.3f}]"
                  f"  vs  {name_neg}(n={neg[0]}) acc={neg[1]:.3f}[{neg[2]:.3f},{neg[3]:.3f}]")
        else:
            print(f"    size {lo_e:.0f}-{hi_e:.0f}aa | insufficient n")


def analyze(tissue):
    print(f"\n{'='*70}\n[{tissue}] full canonical-anchored population (breaking 2-iso limit)\n{'='*70}")
    df, iso, seqs, emb = load_pairs_and_seqs(tissue)

    print(f"[population] domain_binary==0 & nterm_overlap==1: "
          f"{len(df[(df.domain_binary==0)&(df.nterm_overlap==1)])} candidate pairs")
    print(f"[population] domain_binary==0 & nterm_overlap==0: "
          f"{len(df[(df.domain_binary==0)&(df.nterm_overlap==0)])} candidate pairs")

    # ---- N-terminal positive control (sanity: does 0.805 replicate at scale?) ----
    D_nt, gene_nt, size_nt, *_ , n_missing_nt, n_total_nt = build_group(df, iso, seqs, emb, 0, 1)
    print(f"\n[N-terminal, domain_binary==0] usable pairs: {len(D_nt)} / {n_total_nt} "
          f"(missing-seq/degenerate: {n_missing_nt})")
    if len(D_nt) > 30:
        R, nmean, nhi, p_R = coherence(D_nt)
        acc, lo, hi = cv_dir_acc(D_nt, gene_nt, np.ones(len(D_nt)))
        print(f"  N-terminal overall: R={R:.3f} (null {nmean:.3f}, p={p_R:.3f}) | "
              f"CV-dir-acc={acc:.3f} [{lo:.3f},{hi:.3f}]  (2-iso reference was 0.805)")

    # ---- internal edit ----
    (D_int, gene_int, size_int, helix_d, sheet_d, hydro_d, charge_d,
     n_missing_int, n_total_int) = build_group(df, iso, seqs, emb, 0, 0)
    print(f"\n[internal edit, domain_binary==0] usable pairs: {len(D_int)} / {n_total_int} "
          f"(missing-seq/degenerate: {n_missing_int})")
    if len(D_int) > 30:
        R, nmean, nhi, p_R = coherence(D_int)
        acc, lo, hi = cv_dir_acc(D_int, gene_int, np.ones(len(D_int)))
        print(f"  internal overall (unoriented, self-consistency): R={R:.3f} (null {nmean:.3f}, p={p_R:.3f}) | "
              f"CV-dir-acc={acc:.3f} [{lo:.3f},{hi:.3f}]  (2-iso reference was 0.488=chance)")

        print("\n  candidates 1-4 (median-split orientation):")
        test_candidate('helix_delta', D_int, gene_int, helix_d)
        test_candidate('sheet_delta', D_int, gene_int, sheet_d)
        test_candidate('hydrophobicity_delta', D_int, gene_int, hydro_d)
        test_candidate('charge_delta', D_int, gene_int, charge_d)

        comp = np.column_stack([helix_d, sheet_d, hydro_d, charge_d])
        Xc = (comp - comp.mean(0)) / (comp.std(0) + 1e-9)
        print("\n  candidate 5: sub-clustering on independent composition features (k=2)")
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(Xc)
        for c in range(2):
            m = km.labels_ == c
            if m.sum() < 20:
                continue
            R, nmean, nhi, p_R = coherence(D_int[m])
            acc, lo, hi = cv_dir_acc(D_int[m], gene_int[m], np.ones(m.sum()))
            means = comp[m].mean(0)
            print(f"    cluster {c} (n={m.sum()}): R={R:.3f} (null {nmean:.3f}) | "
                  f"CV-dir-acc={acc:.3f} [{lo:.3f},{hi:.3f}] | "
                  f"helix={means[0]:+.3f} sheet={means[1]:+.3f} hydro={means[2]:+.3f} charge={means[3]:+.3f}")

        c0h, c1h = hydro_d[km.labels_ == 0].mean(), hydro_d[km.labels_ == 1].mean()
        hydrophilic = 0 if c0h < c1h else 1
        cluster_bool = km.labels_ == hydrophilic
        size_matched_check(D_int, gene_int, size_int, cluster_bool, 'hydrophilic-cluster', 'other-cluster')


def main():
    for tissue in ['muscle', 'brain']:
        analyze(tissue)


if __name__ == '__main__':
    main()
