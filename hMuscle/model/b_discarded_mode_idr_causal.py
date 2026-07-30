#!/usr/bin/env python
"""
b_discarded_mode_idr_causal.py

Usage test for the disorder-axis lead (finding-discarded-manifold-disorder-axis-lead,
b_manifold_biology_match.py T2: the pooling-DISCARDED non-DC manifold's reproducible subspace
overlaps axis0(disorder) far more than domain/length axes, +0.250 excess over random-K null).
That result is a directional-ALIGNMENT fact (encoding), not a functional/predictive one.

This extends b_idr_supervision_causal.py's EXACT protocol (same manifest population n~1200 slim+
domain, same boundary_cross label from metapredict order<->disorder transition, same confound
oracle, same gene-permutation null) by adding ONE new channel: DISCARD (the top discarded mode,
same object as b_slim_dispersion_structure.py / b_manifold_biology_match.py) alongside the already-
tested COMP / POOL(=DC) / EC(editcore) / ORACLE channels.

DECISIVE CHECK (predict-before-look, stated before running): given axis0 is already known
"encoded but NOT used by B4" (ridge-reliance negative, per FRAMEWORK.md), and T1's per-pair
composition correlations were weak (r~0.14-0.16, ~2% variance), predict comp+discard will NOT
decisively beat the oracle (0.636 in the prior editcore run) -- i.e. expect encoding-without-usage
to replicate here too. A surprise positive (discard clears the oracle) would be the first real
"disorder-manifold is functionally recoverable" result in this track.
"""
import os
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '4'
import sys
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/model_interpretability_map'
PERRES = OUT / 'b_perres'
L = 9
sys.path.insert(0, str(ROOT / 'hMuscle/model'))
import build_severity_pairs as bsp  # noqa: E402
import metapredict as meta  # noqa: E402

AA = 'ACDEFGHIKLMNPQRSTVWY'
aaidx = {c: i for i, c in enumerate(AA)}
TERM_FRAC = 0.20
N_PERM = 20
SEED = 42
MINRES = 24


def comp(s):
    v = np.zeros(20)
    for c in s:
        if c in aaidx:
            v[aaidx[c]] += 1
    return v / max(len(s), 1)


def load_sequences_and_perres():
    iso_ids = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)
    iso_ids = [str(x) for x in iso_ids]
    seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    return iso_ids, seqs


def main():
    manifest = pd.read_csv(OUT / 'b_manifest_pairs.tsv', sep='\t')
    iso_ids, seqs = load_sequences_and_perres()

    uniq_ids = set()
    for _, r in manifest.iterrows():
        uniq_ids.add(iso_ids[int(r['long_idx'])])
        uniq_ids.add(iso_ids[int(r['short_idx'])])
    disorder_cache = {sid: meta.predict_disorder(seqs[sid]) for sid in uniq_ids}
    print(f'disorder-scored {len(uniq_ids)} sequences')

    COMP, POOL, EC, DISCARD, ORACLE, Y, G = [], [], [], [], [], [], []
    n_no_perres, n_no_both_disorder, n_short_residual = 0, 0, 0

    for _, r in manifest.iterrows():
        lid, sid = iso_ids[int(r['long_idx'])], iso_ids[int(r['short_idx'])]
        long_s, short_s = seqs[lid], seqs[sid]
        pl, ps = PERRES / f'{r.long_idx}.npz', PERRES / f'{r.short_idx}.npz'
        if not pl.exists() or not ps.exists():
            n_no_perres += 1
            continue
        HL = np.load(pl)[f'L{L}'].astype(np.float32)
        HS = np.load(ps)[f'L{L}'].astype(np.float32)
        dl, ds = disorder_cache[lid], disorder_cache[sid]

        ops = SequenceMatcher(None, long_s, short_s, autojunk=False).get_opcodes()
        epos_long, epos_short = [], []
        long_dis, short_dis = [], []
        eq_D = []
        for tag, i1, i2, j1, j2 in ops:
            if tag == 'equal':
                for o in range(i2 - i1):
                    if i1 + o < HL.shape[0] and j1 + o < HS.shape[0]:
                        eq_D.append(HL[i1 + o] - HS[j1 + o])
                continue
            if i2 > i1:
                epos_long.extend(range(i1, min(i2, HL.shape[0])))
                long_dis.extend(dl[i1:i2])
            if j2 > j1:
                epos_short.extend(range(j1, j2))
                short_dis.extend(ds[j1:j2])

        if not long_dis or not short_dis:
            n_no_both_disorder += 1
            continue
        if not epos_long:
            continue
        if len(eq_D) < MINRES:
            n_short_residual += 1
            continue

        long_mean_dis = float(np.mean(long_dis))
        short_mean_dis = float(np.mean(short_dis))
        boundary_cross = int((long_mean_dis > 0.5) != (short_mean_dis > 0.5))

        editseq = ''.join(long_s[i1:i2] for tag, i1, i2, j1, j2 in ops if tag != 'equal')
        pooled = HL.mean(0) - HS.mean(0)
        ec = HL[epos_long].mean(0)

        D = np.stack(eq_D); R = D - D.mean(0)
        _, S, Vt = np.linalg.svd(R, full_matrices=False)
        discard_mode = S[0] * Vt[0]
        discard_mode = discard_mode * np.sign(discard_mode[np.argmax(np.abs(discard_mode))])

        first_edit = min(epos_long)
        len_long = len(long_s)
        near_term = int(first_edit < TERM_FRAC * len_long or first_edit > (1 - TERM_FRAC) * len_long)
        edit_len = len(epos_long)
        whole_protein_disorder = float(np.mean(dl))
        oracle_feats = np.array([near_term, edit_len, whole_protein_disorder, len_long], dtype=np.float32)

        COMP.append(comp(editseq)); POOL.append(pooled); EC.append(ec); DISCARD.append(discard_mode)
        ORACLE.append(oracle_feats); Y.append(boundary_cross); G.append(r.gene)

    COMP, POOL, EC, DISCARD, ORACLE = map(np.stack, (COMP, POOL, EC, DISCARD, ORACLE))
    Y, G = np.array(Y), np.array(G)
    print(f'n={len(Y)} genes={len(set(G))}  (dropped: no_perres={n_no_perres}, '
          f'no_both_side_disorder={n_no_both_disorder}, short_residual={n_short_residual})')
    print(f'label balance: pos={Y.sum()} neg={len(Y) - Y.sum()}')

    def cv(X, y, g, seed=0):
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            c = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.06,
                                                l2_regularization=1.0, random_state=seed)
            c.fit(X[tr], y[tr])
            oof[te] = c.predict_proba(X[te])[:, 1]
        return roc_auc_score(y, oof)

    a_c = cv(COMP, Y, G)
    a_p = cv(np.concatenate([COMP, POOL], 1), Y, G)
    a_e = cv(np.concatenate([COMP, EC], 1), Y, G)
    a_d = cv(np.concatenate([COMP, DISCARD], 1), Y, G)
    a_pd = cv(np.concatenate([COMP, POOL, DISCARD], 1), Y, G)
    a_o = cv(ORACLE, Y, G)
    a_oc = cv(np.concatenate([COMP, ORACLE], 1), Y, G)

    print('\n=== Main comparison (gene-disjoint GroupKFold(5), boundary_cross label) ===')
    print(f'{"comp only":20s} {a_c:.3f}')
    print(f'{"comp+pooled(DC)":20s} {a_p:.3f}   (pool-Δ = {a_p - a_c:+.3f})')
    print(f'{"comp+editcore":20s} {a_e:.3f}   (ec-Δ   = {a_e - a_c:+.3f})   [prior run: 0.556]')
    print(f'{"comp+discard(non-DC)":20s} {a_d:.3f}   (discard-Δ = {a_d - a_c:+.3f})   <-- NEW')
    print(f'{"comp+pool+discard":20s} {a_pd:.3f}')
    print(f'{"oracle only":20s} {a_o:.3f}   (positional+gene-gross, no embeddings)   [prior run: 0.636]')
    print(f'{"comp+oracle":20s} {a_oc:.3f}')
    print(f'\nDECISIVE CHECK 1 (discard must beat oracle to count as real, non-confound signal): '
          f'discard={a_d:.3f} vs oracle={a_o:.3f}  (discard-oracle = {a_d - a_o:+.3f})')

    rng = np.random.RandomState(SEED)
    null_aucs = []
    for i in range(N_PERM):
        g_perm = rng.permutation(G)
        auc = cv(np.concatenate([COMP, DISCARD], 1), Y, g_perm, seed=i)
        null_aucs.append(auc)
    null_aucs = np.array(null_aucs)
    print(f'\n=== Gene-permutation null (comp+discard, N={N_PERM} shuffles) ===')
    print(f'true-gene AUROC = {a_d:.3f}')
    print(f'null mean={null_aucs.mean():.3f}  sd={null_aucs.std():.3f}  '
          f'range=[{null_aucs.min():.3f}, {null_aucs.max():.3f}]')
    z = (a_d - null_aucs.mean()) / (null_aucs.std() + 1e-9)
    print(f'z(true vs null) = {z:+.2f}')
    print(f'\nDECISIVE CHECK 2 (true-gene should NOT be suspiciously above null): '
          f'{"PASS (within/below null band)" if a_d <= null_aucs.mean() + 2*null_aucs.std() else "FAIL (leakage-suspicious)"}')

    out = OUT / 'assets' / 'discarded_mode_idr_causal_results.txt'
    with open(out, 'w') as f:
        f.write(f'n={len(Y)} genes={len(set(G))} pos={Y.sum()} neg={len(Y)-Y.sum()}\n')
        f.write(f'comp={a_c:.4f} pooled={a_p:.4f} editcore={a_e:.4f} discard={a_d:.4f} '
                f'pool+discard={a_pd:.4f} oracle={a_o:.4f} comp+oracle={a_oc:.4f}\n')
        f.write(f'discard-oracle={a_d-a_o:+.4f}\n')
        f.write(f'null_mean={null_aucs.mean():.4f} null_sd={null_aucs.std():.4f} z={z:+.2f}\n')
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
