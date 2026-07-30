#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""explore_internal_edit_covariates.py

S1 (multiple hypotheses, parallel test): what explains the "internal edit"
residual that finding-nondomain-anchor-decomposable found at chance level
(CV-dir-acc=0.488 [0.449,0.527], n=627, same 1505 non-domain 2-iso muscle
pairs as exp_nondomain_anchor_decomp.py)? Reuses the IDENTICAL population,
sequences, and coherence/cv_dir_acc protocol for direct comparability with
that established 0.488 chance baseline and the 0.805 N-terminal positive
control.

Candidates tested (SLiM excluded per user: already indirectly shown
"unencoded" via approach-within-gene-layer-divergence's Bucket3 same-length/
same-domain zero-gap inference -- NOTE this is an INDIRECT inference, not a
direct SLiM probe, flagged honestly rather than treated as fully closed.
PTM deferred: no phosphorylation/PTM database available locally, would need
external data acquisition, a separate decision):
  1. Helix propensity delta (Chou-Fasman Pα, mean over changed residues)
  2. Sheet propensity delta (Chou-Fasman Pβ, mean over changed residues)
  3. Hydrophobicity delta (Kyte-Doolittle, mean over changed residues)
  4. Net charge delta (simple physiological-pH scale, mean over changed residues)
  5. Unsupervised sub-clustering (k=2,3 on D) -- tests whether "internal" is a
     MIXTURE of coherent sub-mechanisms that cancel out when pooled (chance
     result at the whole-population level would then be an averaging artifact,
     not evidence of "no anchor exists").

For candidates 1-4: orient = sign(covariate - median(covariate)) [a median-
split binary indicator, the natural generalization of domain_binary's
"long has more domains" orientation to a continuous edit-composition
descriptor], then coherence(D*orient) + gene-disjoint CV-dir-acc(D, gene, orient).
PRE-REGISTERED (S2): H_null = CV-dir-acc ~= 0.5 (CI includes 0.5) for all four,
matching the internal-edit chance baseline (no orientable structure found);
H_explain = any candidate's CV-dir-acc significantly exceeds 0.5 -> a genuine
new covariate for the residual.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
DATA = ROOT / 'hMuscle/data'
MODEL = ROOT / 'hMuscle/model'
FAA = ROOT / 'reports/muscle_labelgap/muscle_2iso.fa'
DOMAIN_MAT = ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy'
NTERM_WIN = 60
rng = np.random.default_rng(42)

# Chou-Fasman helix (Pa) / sheet (Pb) propensities; Kyte-Doolittle hydrophobicity;
# simple physiological net charge -- all standard, widely-published scales.
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


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(' ', '')


def strip_orf(n):
    return re.sub(r'\.p\d+$', '', n)


def parse_faa():
    best, cur, buf = {}, None, []
    def flush():
        if cur is None:
            return
        s = ''.join(buf); b = strip_orf(cur)
        if b not in best or len(s) > len(best[b]):
            best[b] = s
    for line in open(FAA):
        if line.startswith('>'):
            flush(); cur = line[1:].split()[0]; buf = []
        else:
            buf.append(line.strip())
    flush()
    return best


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


def build_population():
    iso = np.array([clean(x) for x in np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)])
    gen = np.array([clean(x) for x in np.load(MODEL / 'my_gene_list_fixed.npy', allow_pickle=True)])
    dom = np.load(DOMAIN_MAT).sum(1).astype(int)
    L15 = np.load(DATA / 'esm2_layer_15_t30_150M.npy').astype(np.float32)
    L30 = np.load(DATA / 'esm2_layer_30_t30_150M.npy').astype(np.float32)
    faa = parse_faa()
    gl, gi = np.unique(gen, return_inverse=True)
    cnt = np.bincount(gi, minlength=len(gl))

    def emb(i):
        return np.concatenate([L15[i], L30[i]])

    D, size, first_pos, gene_id = [], [], [], []
    helix_d, sheet_d, hydro_d, charge_d = [], [], [], []
    for g in np.where(cnt == 2)[0]:
        a, b = np.where(gi == g)[0]
        if dom[a] != dom[b]:
            continue
        if iso[a] not in faa or iso[b] not in faa:
            continue
        sa, sb = faa[iso[a]], faa[iso[b]]
        if sa == sb:
            continue
        lo, sh = (a, b) if len(sa) >= len(sb) else (b, a)
        ls, ss = faa[iso[lo]], faa[iso[sh]]
        ivs, changed = changed_intervals(ls, ss)
        if changed == 0 or not ivs:
            continue
        changed_res_idx = [i for (u, v) in ivs for i in range(u, v)]
        D.append(emb(lo) - emb(sh))
        size.append(changed)
        first_pos.append(ivs[0][0])
        gene_id.append(g)
        helix_d.append(np.mean([HELIX.get(ls[i], 1.0) for i in changed_res_idx]))
        sheet_d.append(np.mean([SHEET.get(ls[i], 1.0) for i in changed_res_idx]))
        hydro_d.append(np.mean([HYDRO.get(ls[i], 0.0) for i in changed_res_idx]))
        charge_d.append(np.mean([CHARGE.get(ls[i], 0.0) for i in changed_res_idx]))

    return (np.array(D), np.array(size), np.array(first_pos), np.array(gene_id),
            np.array(helix_d), np.array(sheet_d), np.array(hydro_d), np.array(charge_d))


def test_candidate(name, D_int, gene_int, cov_int):
    orient = np.where(cov_int > np.median(cov_int), 1.0, -1.0)
    R, nmean, nhi, p_R = coherence(D_int * orient[:, None])
    acc, lo, hi = cv_dir_acc(D_int, gene_int, orient)
    verdict = 'H_explain (>0.5 CI)' if lo > 0.5 else ('H_null (chance)' if lo <= 0.5 <= hi else 'below chance(?)')
    print(f"  [{name:<20}] R={R:.3f} (null {nmean:.3f}, p={p_R:.3f}) | "
          f"CV-dir-acc={acc:.3f} [{lo:.3f},{hi:.3f}]  => {verdict}")
    return acc, lo, hi


def unsupervised_subcluster(D_int, gene_int, comp_features, feature_names):
    """FIXED: cluster on the composition-feature space (helix/sheet/hydro/charge --
    independent of D), THEN test D's coherence within each cluster. Clustering
    directly on D (the v1 bug) is circular: k-means groups points by directional
    similarity in the SAME space subsequently tested for directional coherence,
    guaranteeing near-1.0 CV-dir-acc regardless of any real structure."""
    print("\n=== candidate 5 (FIXED): sub-clustering on INDEPENDENT composition features ===")
    print(f"    clustering features: {feature_names}")
    Xc = (comp_features - comp_features.mean(0)) / (comp_features.std(0) + 1e-9)
    k2_labels = None
    for k in [2, 3]:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xc)
        print(f"\n  -- k={k} --")
        for c in range(k):
            m = km.labels_ == c
            if m.sum() < 20:
                print(f"    cluster {c}: n={m.sum()} too small, skipped"); continue
            R, nmean, nhi, p_R = coherence(D_int[m])
            acc, lo, hi = cv_dir_acc(D_int[m], gene_int[m], np.ones(m.sum()))
            means = comp_features[m].mean(0)
            mean_str = ", ".join(f"{fn}={v:+.3f}" for fn, v in zip(feature_names, means))
            print(f"    cluster {c} (n={m.sum()}): R={R:.3f} (null {nmean:.3f}, p={p_R:.3f}) | "
                  f"CV-dir-acc={acc:.3f} [{lo:.3f},{hi:.3f}]")
            print(f"        composition means: {mean_str}")
        if k == 2:
            k2_labels = km.labels_
    return k2_labels


def size_matched_check(D_int, gene_int, size_int, cluster_label, cluster_name_pos, cluster_name_neg):
    """Established N-term-vs-internal discriminator protocol, reapplied: within
    matched SIZE terciles, does the hydrophilic-coherent cluster still beat the
    chance-level cluster? If the gap disappears once size is controlled, the
    hydrophobic-subgroup finding was actually a size confound."""
    print(f"\n=== size-matched check: {cluster_name_pos} vs {cluster_name_neg} within size terciles ===")
    edges = np.percentile(size_int, [0, 33.33, 66.67, 100])
    for bi in range(3):
        lo_e, hi_e = edges[bi], edges[bi + 1]
        inbin = (size_int >= lo_e) & (size_int <= hi_e)
        cell = {}
        for lab, m in [(cluster_name_pos, cluster_label & inbin), (cluster_name_neg, (~cluster_label) & inbin)]:
            if m.sum() < 15:
                cell[lab] = None; continue
            R, nmean, nhi, p_R = coherence(D_int[m])
            acc, lo, hi = cv_dir_acc(D_int[m], gene_int[m], np.ones(m.sum()))
            cell[lab] = (m.sum(), R, acc, lo, hi)
        pos, neg = cell.get(cluster_name_pos), cell.get(cluster_name_neg)
        if pos and neg:
            print(f"  size {lo_e:.0f}-{hi_e:.0f}aa | {cluster_name_pos}(n={pos[0]}) R={pos[1]:.3f} "
                  f"acc={pos[2]:.3f}[{pos[3]:.3f},{pos[4]:.3f}]  vs  "
                  f"{cluster_name_neg}(n={neg[0]}) R={neg[1]:.3f} acc={neg[2]:.3f}[{neg[3]:.3f},{neg[4]:.3f}]")
        else:
            print(f"  size {lo_e:.0f}-{hi_e:.0f}aa | insufficient n in one or both groups")

    print(f"\n  also: median edit size, {cluster_name_pos}={np.median(size_int[cluster_label]):.0f}aa "
          f"vs {cluster_name_neg}={np.median(size_int[~cluster_label]):.0f}aa")


def main():
    D, size, first_pos, gene_id, helix_d, sheet_d, hydro_d, charge_d = build_population()
    nterm = first_pos < NTERM_WIN
    internal = ~nterm
    print(f"non-domain pairs: {len(D)} | N-terminal: {nterm.sum()} | internal: {internal.sum()}")
    print(f"(reference: N-term CV-dir-acc established=0.805, internal established=0.488=chance)\n")

    D_int, gene_int = D[internal], gene_id[internal]
    print("=== candidates 1-4: median-split orientation, coherence + CV-dir-acc ===")
    test_candidate('helix_delta', D_int, gene_int, helix_d[internal])
    test_candidate('sheet_delta', D_int, gene_int, sheet_d[internal])
    test_candidate('hydrophobicity_delta', D_int, gene_int, hydro_d[internal])
    test_candidate('charge_delta', D_int, gene_int, charge_d[internal])

    comp_features = np.column_stack([helix_d[internal], sheet_d[internal],
                                      hydro_d[internal], charge_d[internal]])
    k2_labels = unsupervised_subcluster(D_int, gene_int, comp_features,
                                         ['helix_delta', 'sheet_delta', 'hydro_delta', 'charge_delta'])

    # identify which k=2 cluster is the hydrophilic/coherent one (lower mean hydro_delta)
    hydro_int = hydro_d[internal]
    c0_hydro = hydro_int[k2_labels == 0].mean(); c1_hydro = hydro_int[k2_labels == 1].mean()
    hydrophilic_cluster = 0 if c0_hydro < c1_hydro else 1
    cluster_bool = (k2_labels == hydrophilic_cluster)
    size_int = size[internal]
    size_matched_check(D_int, gene_int, size_int, cluster_bool,
                        'hydrophilic-cluster', 'other-cluster')


if __name__ == '__main__':
    main()
