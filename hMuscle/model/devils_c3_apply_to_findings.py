#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""devils_c3_apply_to_findings.py

Applies the gene-permutation null test (devils_c3_scale_bias_check.py's
decisive method) DIRECTLY to the three headline numbers from today's session
that all used the vulnerable "orient=+1, self-consistency" CV-dir-acc test:
  1. N-terminal edit group (domain_binary==0 & nterm_overlap==1), full muscle
     population -- real acc was 0.568.
  2. Hydrophilic cluster (from k=2 clustering on independent composition
     features within internal-edit pairs) -- real acc was 0.777.
  3. "Other" (near-neutral) cluster -- real acc was 0.557.

For each: compute the REAL (unpermuted) CV-dir-acc, then a 300-replicate
gene-id-permuted null (destroys gene-family structure, keeps D vectors and
subgroup membership fixed) -- report whether the real value sits outside the
null band (real signal beyond the generic long-minus-short population bias)
or inside it (the observed "coherence" is indistinguishable from the artifact
devils_c3_scale_bias_check.py discovered).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util
from difflib import SequenceMatcher
from sklearn.cluster import KMeans

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
DATA = ROOT / 'hMuscle/data'
MAXLEN = 1022
N_PERM = 300
rng = np.random.default_rng(123)

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


def cv_dir_acc(D, gene_id, seed=None):
    r = np.random.default_rng(seed) if seed is not None else rng
    n = len(D)
    ug = np.unique(gene_id)
    ug_perm = ug.copy(); r.shuffle(ug_perm)
    folds = {g: i % 5 for i, g in enumerate(ug_perm)}
    fid = np.array([folds[g] for g in gene_id])
    correct = 0
    for k in range(5):
        te = fid == k; tr = ~te
        if tr.sum() < 5 or te.sum() == 0:
            continue
        a = D[tr].mean(0); a /= (np.linalg.norm(a) + 1e-9)
        pred = np.dot(D[te], a) > 0
        correct += int(pred.sum())
    return correct / n


def null_band(D, gene_id, n_perm=N_PERM):
    vals = np.empty(n_perm)
    for p in range(n_perm):
        perm_gene = rng.permutation(gene_id)
        vals[p] = cv_dir_acc(D, perm_gene, seed=p)
    return vals


def report(name, D, gene_id):
    real = cv_dir_acc(D, gene_id, seed=999999)  # real (unpermuted) gene assignment, own fold split
    null = null_band(D, gene_id)
    lo, hi = np.percentile(null, [2.5, 97.5])
    verdict = 'REAL signal beyond artifact' if real > hi else (
        'INDISTINGUISHABLE from artifact' if lo <= real <= hi else 'BELOW null(?)')
    print(f"\n[{name}] n={len(D)}, genes={len(np.unique(gene_id))}")
    print(f"  real (unpermuted) CV-dir-acc = {real:.4f}")
    print(f"  gene-permuted null: mean={null.mean():.4f} CI=[{lo:.4f},{hi:.4f}]")
    print(f"  => {verdict}  (real - null_mean = {real - null.mean():+.4f})")
    return real, null


def build_group(df, iso, seqs, emb, domain0, nterm_val):
    sub = df[(df['domain_binary'] == domain0) & (df['nterm_overlap'] == nterm_val)]
    D, gene_id, helix_d, sheet_d, hydro_d, charge_d = [], [], [], [], [], []
    for _, r in sub.iterrows():
        long_id, short_id = iso[int(r['long_idx'])], iso[int(r['short_idx'])]
        if long_id not in seqs or short_id not in seqs:
            continue
        ls, ss = seqs[long_id][:MAXLEN], seqs[short_id][:MAXLEN]
        if ls == ss:
            continue
        ivs, changed = changed_intervals(ls, ss)
        if changed == 0 or not ivs:
            continue
        changed_res_idx = [i for (u, v) in ivs for i in range(u, v) if i < len(ls)]
        if not changed_res_idx:
            continue
        D.append(emb[int(r['long_idx'])] - emb[int(r['short_idx'])])
        gene_id.append(r['gene'])
        helix_d.append(np.mean([HELIX.get(ls[i], 1.0) for i in changed_res_idx]))
        sheet_d.append(np.mean([SHEET.get(ls[i], 1.0) for i in changed_res_idx]))
        hydro_d.append(np.mean([HYDRO.get(ls[i], 0.0) for i in changed_res_idx]))
        charge_d.append(np.mean([CHARGE.get(ls[i], 0.0) for i in changed_res_idx]))
    return (np.array(D), np.array(gene_id), np.array(helix_d), np.array(sheet_d),
            np.array(hydro_d), np.array(charge_d))


def main():
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsp)

    df = pd.read_csv(SEV / 'muscle_severity_pairs_scored.tsv', sep='\t')
    df = df[df['tissue'] == 'muscle'].reset_index(drop=True)
    iso = np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)
    iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
    seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
    L15 = np.load(DATA / 'esm2_layer_15_t30_150M.npy').astype(np.float32)
    L30 = np.load(DATA / 'esm2_layer_30_t30_150M.npy').astype(np.float32)
    emb = np.concatenate([L15, L30], axis=1)

    print("[build] N-terminal group (domain_binary==0 & nterm_overlap==1)...")
    D_nt, gene_nt, *_ = build_group(df, iso, seqs, emb, 0, 1)
    report('N-terminal edit (full muscle)', D_nt, gene_nt)

    print("\n[build] internal-edit group + hydrophilic/other clustering...")
    D_int, gene_int, helix_d, sheet_d, hydro_d, charge_d = build_group(df, iso, seqs, emb, 0, 0)
    comp = np.column_stack([helix_d, sheet_d, hydro_d, charge_d])
    Xc = (comp - comp.mean(0)) / (comp.std(0) + 1e-9)
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(Xc)
    c0h, c1h = hydro_d[km.labels_ == 0].mean(), hydro_d[km.labels_ == 1].mean()
    hydrophilic = 0 if c0h < c1h else 1
    m_hydro = km.labels_ == hydrophilic
    m_other = ~m_hydro

    report('hydrophilic cluster', D_int[m_hydro], gene_int[m_hydro])
    report('other cluster', D_int[m_other], gene_int[m_other])
    report('internal-edit overall (unclustered)', D_int, gene_int)


if __name__ == '__main__':
    main()
