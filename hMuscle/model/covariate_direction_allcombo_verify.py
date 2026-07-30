#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""covariate_direction_allcombo_verify.py

Option A: does covariate_direction_axis_overlap.py's "all 5 covariates near-
orthogonal to the 8 PCA axes" result survive when built on the SAME
all-combinations within-gene-pair population and length-decile-stratified
fitting protocol as the established ceiling_640dim_domain.py (domain_binary
AUROC 0.838 brain) -- rather than the canonical-anchored severity-pairs
population used everywhere else this session? Runs TWO checks per covariate:
  (A1) exact established protocol reproduction: length-decile-stratified
       half-split LR AUROC (sanity check against the known 0.838 brain number
       for domain_binary).
  (A2) SAME gene-disjoint-CV direction-extraction method as
       covariate_direction_axis_overlap.py, but on this all-combinations
       population -- isolates whether POPULATION (not fitting protocol)
       explains the earlier near-zero axis-overlap result.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
PCA_DIR = ROOT / 'reports/v20b_pca_interp'
DATA = ROOT / 'hMuscle/data'
BRAIN = DATA / 'brain_isoquant_esm2/full'
GTF = DATA / 'brain_esm2/brain_only.gtf'
HMM = ROOT / 'hMuscle/results_isoform/features/hmmscan_brain.domtblout'
MAX_PAIRS = 60000
N_FOLDS = 5
SEED = 42
NTERM_WINDOW = 60
rng = np.random.default_rng(42)


def opcode_intervals_full(long_s, short_s):
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    return sm.get_opcodes()


def nterm_and_resync(long_s, short_s):
    ops = opcode_intervals_full(long_s, short_s)
    ivs = [(i1, i2) for tag, i1, i2, j1, j2 in ops if tag != 'equal' and i2 > i1]
    nterm = int(any(i1 < NTERM_WINDOW for i1, i2 in ivs)) if ivs else 0
    first_change_start = None
    for tag, i1, i2, j1, j2 in ops:
        if tag != 'equal':
            first_change_start = i1
            break
    if first_change_start is None:
        return None, None
    downstream_len = len(long_s) - first_change_start
    if downstream_len <= 0:
        return None, None
    max_equal = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == 'equal' and i1 >= first_change_start:
            max_equal = max(max_equal, i2 - i1)
    resync_fail = int((max_equal / downstream_len) < 0.5)
    return nterm, resync_fail


def build_muscle_population():
    import importlib.util
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsp)

    iso = np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)
    iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
    gene = np.load(MODEL / 'my_gene_list_fixed.npy', allow_pickle=True)
    gene = [g.decode() if isinstance(g, bytes) else str(g) for g in gene]
    dom = np.load(ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy')
    seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')

    g2i = defaultdict(list)
    for i, g in enumerate(gene):
        g2i[g].append(i)
    pairs = []
    for g, idx in g2i.items():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                pairs.append((idx[a], idx[b], g))
    pairs = np.array([(a, b) for a, b, g in pairs]), [g for a, b, g in pairs]
    idx_pairs, genes = pairs
    if len(idx_pairs) > MAX_PAIRS:
        sel = rng.choice(len(idx_pairs), MAX_PAIRS, replace=False)
        idx_pairs = idx_pairs[sel]
        genes = [genes[s] for s in sel]

    domdiff, nterm, resync, length_diff, keep = [], [], [], [], []
    for k, (a, b) in enumerate(idx_pairs):
        ida, idb = iso[a], iso[b]
        if ida not in seqs or idb not in seqs:
            keep.append(False); continue
        sa, sb = seqs[ida][:1022], seqs[idb][:1022]
        if sa == sb:
            keep.append(False); continue
        long_s, short_s = (sa, sb) if len(sa) >= len(sb) else (sb, sa)
        long_idx, short_idx = (a, b) if len(sa) >= len(sb) else (b, a)
        n, r = nterm_and_resync(long_s, short_s)
        if n is None:
            keep.append(False); continue
        dd = int(np.any(dom[a] != dom[b]))
        domdiff.append(dd); nterm.append(n); resync.append(r)
        length_diff.append(abs(len(sa) - len(sb)))
        keep.append(True)
    keep = np.array(keep)
    idx_pairs = idx_pairs[keep]
    genes = np.array(genes)[keep]
    print(f"[muscle] all-combinations pairs kept: {len(idx_pairs)} / {keep.sum()+((~keep).sum())}", flush=True)
    return idx_pairs, genes, np.array(domdiff), np.array(nterm), np.array(resync), np.array(length_diff)


def build_brain_population():
    import interp_within_family_pca as M
    import importlib.util
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsp)

    sym, _ = M.gene_to_family()
    bids = np.array([str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)])
    N = len(bids)
    seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')

    name2enst = {}
    for line in open(GTF):
        if "\ttranscript\t" not in line:
            continue
        em = re.search(r'transcript_id "([^"]+)"', line)
        nm = re.search(r'transcript_name "([^"]+)"', line)
        if em and nm:
            name2enst[nm.group(1)] = em.group(1).split(".")[0]
    enst2dom = defaultdict(set)
    for line in open(HMM):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if float(p[12]) > 1e-5:
            continue
        enst2dom[p[3].split(".p")[0].split(".")[0]].add(p[1].split(".")[0])

    def dom_of(i):
        b = bids[i]
        e = b.split(".")[0] if b.startswith("ENST") else (b if b.startswith("transcript") else name2enst.get(b, ""))
        return enst2dom.get(e, frozenset())
    domset = [dom_of(i) for i in range(N)]

    g2i = defaultdict(list)
    for i in range(N):
        g2i[sym[i]].append(i)
    pairs = []
    for g, idx in g2i.items():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                pairs.append((idx[a], idx[b], g))
    if len(pairs) > MAX_PAIRS:
        sel = rng.choice(len(pairs), MAX_PAIRS, replace=False)
        pairs = [pairs[s] for s in sel]

    domdiff, nterm, resync, length_diff, idx_pairs, genes = [], [], [], [], [], []
    for a, b, g in pairs:
        ida, idb = bids[a], bids[b]
        if ida not in seqs or idb not in seqs:
            continue
        sa, sb = seqs[ida][:1022], seqs[idb][:1022]
        if sa == sb:
            continue
        long_s, short_s = (sa, sb) if len(sa) >= len(sb) else (sb, sa)
        n, r = nterm_and_resync(long_s, short_s)
        if n is None:
            continue
        dd = int(len(domset[a] ^ domset[b]) > 0)
        domdiff.append(dd); nterm.append(n); resync.append(r)
        length_diff.append(abs(len(sa) - len(sb)))
        idx_pairs.append((a, b)); genes.append(g)
    print(f"[brain] all-combinations pairs kept: {len(idx_pairs)}", flush=True)
    return np.array(idx_pairs), np.array(genes), np.array(domdiff), np.array(nterm), np.array(resync), np.array(length_diff)


def build_640_layermean(tissue, N):
    mu = np.load(PCA_DIR / 'layer_stats_mu.npy').astype(np.float64)
    sd = np.load(PCA_DIR / 'layer_stats_sd.npy').astype(np.float64)
    if tissue == 'muscle':
        path_fmt = str(ROOT / 'hMuscle/data/esm2_layer_{:02d}_t30_150M.npy')
    else:
        path_fmt = str(BRAIN / 'brain_full_esm2_layer{:02d}_t30_150M.npy')
    X640 = np.zeros((N, 640), dtype=np.float64)
    for L in range(30):
        arr = np.load(path_fmt.format(L + 1), mmap_mode='r')
        X640 += (arr.astype(np.float64) - mu[L]) / sd[L]
        del arr
    X640 /= 30.0
    return X640


def lenmatched_auroc(X, y, ln, seed=0):
    r = np.random.default_rng(seed)
    dec = np.digitize(ln, np.quantile(ln, np.linspace(0.1, 0.9, 9)))
    aucs = []
    for d in range(10):
        m = dec == d
        if m.sum() < 100 or y[m].sum() < 20 or y[m].sum() == m.sum():
            continue
        Xd, yd = X[m], y[m]
        mu_ = Xd.mean(0); sdv = Xd.std(0) + 1e-8
        Xd = (Xd - mu_) / sdv
        n = len(Xd); perm = r.permutation(n); cut = n // 2
        tr, te = perm[:cut], perm[cut:]
        C = 0.05 if Xd.shape[1] > 50 else 1.0
        lr = LogisticRegression(max_iter=1000, C=C).fit(Xd[tr], yd[tr])
        aucs.append(roc_auc_score(yd[te], lr.predict_proba(Xd[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs)


def gene_disjoint_folds(genes, n_folds=N_FOLDS, seed=SEED):
    uniq = np.array(sorted(set(genes)))
    r = np.random.default_rng(seed)
    r.shuffle(uniq)
    fold_of_gene = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of_gene[g] for g in genes])


def cv_direction_and_auroc(absD, y, genes):
    fold = gene_disjoint_folds(genes)
    oof_pred = np.zeros(len(y))
    directions = []
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        if y[tr].sum() < 5 or (len(y[tr]) - y[tr].sum()) < 5:
            continue
        mu_, sd_ = absD[tr].mean(0), absD[tr].std(0) + 1e-8
        Xtr = (absD[tr] - mu_) / sd_
        Xte = (absD[te] - mu_) / sd_
        clf = LogisticRegression(max_iter=2000, C=0.05).fit(Xtr, y[tr])
        oof_pred[te] = clf.predict_proba(Xte)[:, 1]
        directions.append(clf.coef_[0] / sd_)
    auroc = roc_auc_score(y, oof_pred)
    direction = np.mean(directions, axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return auroc, direction


def analyze(tissue):
    print(f"\n{'='*70}\n[{tissue}] all-combinations population, established-protocol verification\n{'='*70}")
    if tissue == 'muscle':
        idx_pairs, genes, domdiff, nterm, resync, ldiff = build_muscle_population()
    else:
        idx_pairs, genes, domdiff, nterm, resync, ldiff = build_brain_population()

    N = max(idx_pairs.max() + 1, 1)
    X640 = build_640_layermean(tissue, N if tissue == 'brain' else
                                np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True).shape[0])
    absD = np.abs(X640[idx_pairs[:, 0]] - X640[idx_pairs[:, 1]])
    W = np.load(PCA_DIR / 'W_axes_8x640.npy')
    W_unit = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

    for name, y in [('domain_binary', domdiff), ('nterm_overlap', nterm), ('resync_failure_binary', resync)]:
        vl = ~pd.isna(y) if y.dtype == object else np.ones(len(y), dtype=bool)
        yv = y[vl].astype(int)
        if yv.sum() < 20 or (len(yv) - yv.sum()) < 20:
            print(f"  [{name}] insufficient class balance, skipped"); continue

        a_est, s_est, n_bins = lenmatched_auroc(absD[vl], yv, ldiff[vl], seed=1)
        print(f"\n  [{name}] (A1) established length-decile-stratified AUROC = {a_est:.3f} +/- {s_est:.3f} "
              f"(n_bins={n_bins})")

        auroc_cv, direction = cv_direction_and_auroc(absD[vl], yv, genes[vl])
        cos_to_axes = W_unit @ direction
        top_axis = np.argmax(np.abs(cos_to_axes))
        print(f"  [{name}] (A2) gene-disjoint-CV AUROC = {auroc_cv:.3f}")
        print(f"       cosine to 8 axes: " + " ".join(f"ax{k}={c:+.3f}" for k, c in enumerate(cos_to_axes)))
        print(f"       top axis: axis{top_axis} ({cos_to_axes[top_axis]:+.3f}); "
              f"sum|cos|^2 = {np.sum(cos_to_axes**2):.3f}")


def main():
    for tissue in ['brain', 'muscle']:
        analyze(tissue)


if __name__ == '__main__':
    main()
