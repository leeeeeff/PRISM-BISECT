#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""covariate_direction_allcombo_v2.py

Fixes both bugs found in covariate_direction_allcombo_verify.py:
  BUG 1 (population mismatch, brain domain_binary 0.630 vs established 0.838):
    the original ceiling_640dim_domain.py needs NO sequences for domain_binary
    (domain sets + length come from precomputed feat_matrix_brain.npy), so its
    60k-pair sample is drawn from the FULL population. My v1 script required
    FASTA-resolvable sequences for ALL THREE covariates jointly (needed for
    nterm/resync alignment), which silently shrank/biased the domain_binary
    population too. FIX: build domain_binary's population WITHOUT any sequence
    requirement (matches original exactly for brain); nterm/resync keep their
    own, necessarily sequence-filtered, population -- reported separately, not
    forced onto identical rows.
  BUG 2 (fitting-method gap, muscle domain_binary A1=0.857 vs A2=0.616 on
    IDENTICAL population/features): a single global gene-disjoint-CV logistic
    fit with flat L2 (C=0.05) underperforms the established length-decile-
    stratified protocol by a wide margin -- meaning the "direction" it finds is
    not a good discriminator, so its cosine-overlap with the 8 PCA axes was not
    a trustworthy answer to "does the best available direction align with any
    axis." FIX: extract each length-decile bin's OWN logistic direction (same
    fitting recipe that achieves the established AUROC), rescale each back to
    raw-feature space, and combine via a SAMPLE-SIZE-WEIGHTED average into one
    comparable direction -- report both the per-bin (already-trustworthy) AUROC
    and the pooled direction's cosine-overlap with the 8 axes.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
PCA_DIR = ROOT / 'reports/v20b_pca_interp'
DATA = ROOT / 'hMuscle/data'
BRAIN = DATA / 'brain_isoquant_esm2/full'
GTF = DATA / 'brain_esm2/brain_only.gtf'
HMM = ROOT / 'hMuscle/results_isoform/features/hmmscan_brain.domtblout'
MAX_PAIRS = 60000
NTERM_WINDOW = 60
rng = np.random.default_rng(42)

W = np.load(PCA_DIR / 'W_axes_8x640.npy')
W_unit = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)


def nterm_and_resync(long_s, short_s):
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    ops = sm.get_opcodes()
    ivs = [(i1, i2) for tag, i1, i2, j1, j2 in ops if tag != 'equal' and i2 > i1]
    nterm = int(any(i1 < NTERM_WINDOW for i1, i2 in ivs)) if ivs else 0
    first_change_start = None
    for tag, i1, i2, j1, j2 in ops:
        if tag != 'equal':
            first_change_start = i1; break
    if first_change_start is None:
        return None, None
    downstream_len = len(long_s) - first_change_start
    if downstream_len <= 0:
        return None, None
    max_equal = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == 'equal' and i1 >= first_change_start:
            max_equal = max(max_equal, i2 - i1)
    return nterm, int((max_equal / downstream_len) < 0.5)


def all_combo_pairs(gene_ids):
    g2i = defaultdict(list)
    for i, g in enumerate(gene_ids):
        g2i[g].append(i)
    pairs = []
    for g, idx in g2i.items():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                pairs.append((idx[a], idx[b], g))
    return pairs


def cap_sample(pairs, max_pairs=MAX_PAIRS, seed=42):
    if len(pairs) <= max_pairs:
        return pairs
    r = np.random.default_rng(seed)
    sel = r.choice(len(pairs), max_pairs, replace=False)
    return [pairs[s] for s in sel]


def build_640_layermean(tissue, N):
    mu = np.load(PCA_DIR / 'layer_stats_mu.npy').astype(np.float64)
    sd = np.load(PCA_DIR / 'layer_stats_sd.npy').astype(np.float64)
    path_fmt = (str(ROOT / 'hMuscle/data/esm2_layer_{:02d}_t30_150M.npy') if tissue == 'muscle'
                else str(BRAIN / 'brain_full_esm2_layer{:02d}_t30_150M.npy'))
    X640 = np.zeros((N, 640), dtype=np.float64)
    for L in range(30):
        arr = np.load(path_fmt.format(L + 1), mmap_mode='r')
        X640 += (arr.astype(np.float64) - mu[L]) / sd[L]
        del arr
    X640 /= 30.0
    return X640


def lenmatched_direction(X, y, ln, seed=0):
    """Established per-length-decile-bin fitting (reproduces ceiling AUROC),
    PLUS extraction of each bin's direction, combined via sample-weighted avg."""
    r = np.random.default_rng(seed)
    dec = np.digitize(ln, np.quantile(ln, np.linspace(0.1, 0.9, 9)))
    aucs, bin_dirs, bin_ns = [], [], []
    for d in range(10):
        m = dec == d
        if m.sum() < 100 or y[m].sum() < 20 or y[m].sum() == m.sum():
            continue
        Xd, yd = X[m], y[m]
        mu_ = Xd.mean(0); sdv = Xd.std(0) + 1e-8
        Xdn = (Xd - mu_) / sdv
        n = len(Xdn); perm = r.permutation(n); cut = n // 2
        tr, te = perm[:cut], perm[cut:]
        C = 0.05 if Xdn.shape[1] > 50 else 1.0
        lr = LogisticRegression(max_iter=1000, C=C).fit(Xdn[tr], yd[tr])
        aucs.append(roc_auc_score(yd[te], lr.predict_proba(Xdn[te])[:, 1]))
        bin_dirs.append(lr.coef_[0] / sdv)
        bin_ns.append(m.sum())
    bin_dirs = np.array(bin_dirs); bin_ns = np.array(bin_ns, dtype=np.float64)
    pooled = (bin_dirs * bin_ns[:, None]).sum(0) / bin_ns.sum()
    pooled = pooled / (np.linalg.norm(pooled) + 1e-12)
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs), pooled


def report_direction(name, auroc, std, n_bins, direction):
    cos = W_unit @ direction
    top = np.argmax(np.abs(cos))
    print(f"\n  [{name}] established length-decile AUROC = {auroc:.3f} +/- {std:.3f} (n_bins={n_bins})")
    print(f"       POOLED direction cosine to 8 axes: " + " ".join(f"ax{k}={c:+.3f}" for k, c in enumerate(cos)))
    print(f"       top axis: axis{top} ({cos[top]:+.3f}); sum|cos|^2 = {np.sum(cos**2):.3f} "
          f"(1.0=fully inside 8-axis subspace, 0=fully orthogonal)")


def analyze_brain():
    print(f"\n{'='*70}\n[brain] domain_binary: NO sequence filter (matches established protocol exactly)\n{'='*70}")
    import interp_within_family_pca as M
    sym, _ = M.gene_to_family()
    feat = np.load(PCA_DIR / 'feat_matrix_brain.npy')
    length = feat[:, M.FEAT_NAMES.index('length')]
    bids = np.array([str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)])
    N = len(bids)

    name2enst = {}
    for line in open(GTF):
        if "\ttranscript\t" not in line:
            continue
        em = re.search(r'transcript_id "([^"]+)"', line); nm = re.search(r'transcript_name "([^"]+)"', line)
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

    pairs = cap_sample(all_combo_pairs(sym))
    i1 = np.array([p[0] for p in pairs]); i2 = np.array([p[1] for p in pairs])
    dham = np.array([len(domset[a] ^ domset[b]) for a, b in zip(i1, i2)])
    dlen = np.abs(length[i1] - length[i2])
    domdiff = (dham > 0).astype(int)
    vl = ~np.isnan(dlen)
    print(f"  pairs n={len(pairs):,} (NO sequence requirement, matches original protocol)  "
          f"domain-different={int(domdiff[vl].sum()):,}")

    X640 = build_640_layermean('brain', N)
    absD = np.abs(X640[i1] - X640[i2])[vl]
    auroc, std, nb, direction = lenmatched_direction(absD, domdiff[vl], dlen[vl], seed=1)
    report_direction('domain_binary (no-seq-filter population)', auroc, std, nb, direction)


def analyze_alignment_covariates(tissue):
    print(f"\n{'='*70}\n[{tissue}] nterm_overlap / resync_failure_binary "
          f"(sequence-filtered population, unavoidable for alignment) + domain_binary same-pop sanity\n{'='*70}")
    import importlib.util
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsp)

    if tissue == 'muscle':
        iso = np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)
        iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
        gene = np.load(MODEL / 'my_gene_list_fixed.npy', allow_pickle=True)
        gene = [g.decode() if isinstance(g, bytes) else str(g) for g in gene]
        dom = np.load(ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy')
        seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
        N_total = len(iso)
    else:
        import interp_within_family_pca as M
        sym, _ = M.gene_to_family()
        iso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
        gene = list(sym)
        seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
        N_total = len(iso)
        dom = None  # domain handled separately for brain (analyze_brain)

    pairs = cap_sample(all_combo_pairs(gene))
    domdiff, nterm, resync, ldiff, i1, i2 = [], [], [], [], [], []
    for a, b, g in pairs:
        ida, idb = iso[a], iso[b]
        if ida not in seqs or idb not in seqs:
            continue
        sa, sb = seqs[ida][:1022], seqs[idb][:1022]
        if sa == sb:
            continue
        long_s, short_s = (sa, sb) if len(sa) >= len(sb) else (sb, sa)
        n, r = nterm_and_resync(long_s, short_s)
        if n is None:
            continue
        nterm.append(n); resync.append(r); ldiff.append(abs(len(sa) - len(sb)))
        i1.append(a); i2.append(b)
        if dom is not None:
            domdiff.append(int(np.any(dom[a] != dom[b])))
    i1 = np.array(i1); i2 = np.array(i2)
    print(f"  sequence-resolvable pairs kept: {len(i1)} / {len(pairs)}")

    X640 = build_640_layermean(tissue, N_total)
    absD = np.abs(X640[i1] - X640[i2])
    ldiff = np.array(ldiff, dtype=np.float64)

    if dom is not None:
        auroc, std, nb, direction = lenmatched_direction(absD, np.array(domdiff), ldiff, seed=1)
        report_direction('domain_binary (seq-filtered, same pop as nterm/resync below)', auroc, std, nb, direction)

    auroc, std, nb, direction = lenmatched_direction(absD, np.array(nterm), ldiff, seed=1)
    report_direction('nterm_overlap', auroc, std, nb, direction)

    auroc, std, nb, direction = lenmatched_direction(absD, np.array(resync), ldiff, seed=1)
    report_direction('resync_failure_binary', auroc, std, nb, direction)


def main():
    analyze_brain()
    analyze_alignment_covariates('brain')
    analyze_alignment_covariates('muscle')


if __name__ == '__main__':
    main()
