#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_f_plm_delta_geometry.py
----------------------------
Corrected version of the naively-proposed "Panel B: Normalized Delta Space Shift".

The naive version ("does concat[phi||delta] increase isoform pairwise distance vs
plain phi?") is a MATHEMATICAL TAUTOLOGY: by L2 block-additivity,
  ||cat_a - cat_b||^2 = ||phi_a - phi_b||^2 + ||delta_a - delta_b||^2  >=  ||phi_a - phi_b||^2
always holds regardless of whether delta carries real signal or is pure noise.
"Distance increased" is guaranteed by construction and has zero evidentiary value.

The only non-trivial version: does REAL delta separate within-gene isoform pairs
MORE than a norm-matched RANDOM delta would (same control used in the AUPRC
random-delta ablation, v17f_layer_breakdown.py)? This is tested here, across
ESM-2 150M / ProtT5-XL / Ankh-base, with:
  - within-gene vs between-gene split (within-gene is the scientifically relevant one;
    between-gene is dominated by gene-family separation per prior PCA hierarchy findings)
  - length-matched subset (length is the strongest PCA-encoded axis but weakest
    predictive one per prior findings; must rule out length driving any excess separation)
  - gene-level bootstrap CI (n=1000) to avoid pair-level pseudo-replication

Per [[approach-coherence-vs-labeling-stopping-rule]]: even a positive result here
(real delta > random delta in geometric separation) is representation-level
coherence, NOT evidence of label-alignment — must be read against DR-AUC
(reports/exp_f_plm_scale/dr_auc.json), which was already flat (~0.63) across
all three architectures.
"""
import os, json, time
import numpy as np
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from exp_f_plm_scale_embed import load_brain_seqs, clean_id, BRAIN_PEP, BRAIN_IDS

DATA_DIR = '../data'
OUT_DIR  = '../../reports/exp_f_plm_scale'
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    ('t30_150M',   30, 't30_150M'),
    ('prot_t5_xl', 24, 'prot_t5_xl'),
    ('ankh_base',  48, 'ankh_base'),
]
N_BOOT = 1000
RNG_SEED = 2026


def load_test_pair(tag, n_layers):
    L_f, L_m = n_layers, n_layers // 2
    if tag == 't30_150M':
        pf = f'{DATA_DIR}/esm2_layer_{L_f}_t30_150M.npy'
        pm = f'{DATA_DIR}/esm2_layer_{L_m}_t30_150M.npy'
    else:
        pf = f'{DATA_DIR}/esm2_layer_{L_f:02d}_{tag}.npy'
        pm = f'{DATA_DIR}/esm2_layer_{L_m:02d}_{tag}.npy'
    return np.load(pf).astype(np.float32), np.load(pm).astype(np.float32)


def gene_bootstrap_ci(pair_vals, pair_gene_idx, n_genes, n_boot=N_BOOT, seed=RNG_SEED):
    """Resample GENES with replacement (not pairs) to avoid pseudo-replication
    from isoforms appearing in multiple within-gene pairs."""
    rng = np.random.default_rng(seed)
    gene_to_pairs = defaultdict(list)
    for i, g in enumerate(pair_gene_idx):
        gene_to_pairs[g].append(i)
    genes = list(gene_to_pairs.keys())
    boots = []
    for _ in range(n_boot):
        sampled = rng.choice(genes, size=len(genes), replace=True)
        idxs = np.concatenate([gene_to_pairs[g] for g in sampled])
        boots.append(pair_vals[idxs].mean())
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    t0 = time.time()
    print("=" * 70)
    print("  Delta-Geometry: real-delta vs norm-matched random-delta")
    print("  within-gene isoform pairwise separation, 3 PLM architectures")
    print("=" * 70)

    gene_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
    iso_raw  = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
    genes    = np.array([clean_id(g) for g in gene_raw])
    isos     = [clean_id(x) for x in iso_raw]
    N = len(genes)

    print(f"\n[1] Loading brain peptide sequences for length + valid-seq mask...")
    seq_dict = load_brain_seqs(BRAIN_PEP, isos)
    lengths  = np.array([len(seq_dict.get(iid, '')) for iid in isos])
    valid_seq = lengths > 0
    print(f"  {valid_seq.sum()}/{N} isoforms with sequence (length available)")

    gene2idxs = defaultdict(list)
    for i, g in enumerate(genes):
        if valid_seq[i]:
            gene2idxs[g].append(i)
    gene2idxs = {g: np.array(ix) for g, ix in gene2idxs.items() if len(ix) >= 2}
    print(f"  {len(gene2idxs)} genes with >=2 sequence-valid isoforms (within-gene pairs)")

    # length-difference median for the length-matched subset
    all_within_len_diffs = []
    for g, ix in gene2idxs.items():
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                all_within_len_diffs.append(abs(lengths[ix[a]] - lengths[ix[b]]))
    len_diff_median = float(np.median(all_within_len_diffs))
    print(f"  within-gene pair |length diff| median = {len_diff_median:.0f} aa "
          f"(length-matched subset = pairs below this)")

    results = {}
    for tag, n_layers, label in MODELS:
        print(f"\n{'-'*60}\n  {label}")
        phi_f, phi_m = load_test_pair(tag, n_layers)
        delta_real = phi_f - phi_m

        rng = np.random.default_rng(hash(tag) % (2**31))
        delta_rand = rng.standard_normal(delta_real.shape).astype(np.float32)
        real_norms = np.linalg.norm(delta_real, axis=1, keepdims=True)
        delta_rand = delta_rand / (np.linalg.norm(delta_rand, axis=1, keepdims=True) + 1e-8) * real_norms

        # within-gene pairs
        pair_d2_real, pair_d2_rand, pair_gene_idx, pair_len_diff = [], [], [], []
        gene_list = list(gene2idxs.items())
        for gi, (g, ix) in enumerate(gene_list):
            for a in range(len(ix)):
                for b in range(a + 1, len(ix)):
                    i, j = ix[a], ix[b]
                    pair_d2_real.append(float(np.sum((delta_real[i] - delta_real[j]) ** 2)))
                    pair_d2_rand.append(float(np.sum((delta_rand[i] - delta_rand[j]) ** 2)))
                    pair_gene_idx.append(gi)
                    pair_len_diff.append(abs(lengths[i] - lengths[j]))
        pair_d2_real = np.array(pair_d2_real)
        pair_d2_rand = np.array(pair_d2_rand)
        pair_gene_idx = np.array(pair_gene_idx)
        pair_len_diff = np.array(pair_len_diff)
        n_genes = len(gene_list)

        def summarize(mask, name):
            r, n = pair_d2_real[mask], pair_d2_rand[mask]
            excess_pct = (r.mean() / n.mean() - 1) * 100
            diff = r - n
            lo, hi = gene_bootstrap_ci(diff, pair_gene_idx[mask], n_genes)
            sig = not (lo <= 0 <= hi)
            print(f"  [{name:20s}] N_pairs={mask.sum():6d}  "
                  f"E[d_real^2]={r.mean():10.2f}  E[d_rand^2]={n.mean():10.2f}  "
                  f"excess={excess_pct:+6.1f}%  diff_CI95=[{lo:.2f},{hi:.2f}]  "
                  f"sig(0 excluded)={sig}")
            return {'n_pairs': int(mask.sum()), 'mean_d2_real': float(r.mean()),
                    'mean_d2_rand': float(n.mean()), 'excess_pct': float(excess_pct),
                    'diff_ci95': [lo, hi], 'significant': bool(sig)}

        all_mask = np.ones(len(pair_d2_real), dtype=bool)
        matched_mask = pair_len_diff <= len_diff_median

        res_all = summarize(all_mask, 'within-gene ALL')
        res_matched = summarize(matched_mask, 'within-gene length-matched')

        # ── Between-gene control: is real-delta tightness gene-SPECIFIC or global collapse? ──
        rng_bg = np.random.default_rng(hash(tag + '_bg') % (2**31))
        valid_idx = np.array([i for i in range(N) if valid_seq[i]])
        n_bg = len(pair_d2_real)  # match within-gene pair count
        gi = rng_bg.choice(valid_idx, size=n_bg * 3)
        gj = rng_bg.choice(valid_idx, size=n_bg * 3)
        keep = genes[gi] != genes[gj]  # ensure actually cross-gene
        gi, gj = gi[keep][:n_bg], gj[keep][:n_bg]
        bg_d2_real = np.sum((delta_real[gi] - delta_real[gj]) ** 2, axis=1)
        bg_d2_rand = np.sum((delta_rand[gi] - delta_rand[gj]) ** 2, axis=1)
        bg_excess_pct = (bg_d2_real.mean() / bg_d2_rand.mean() - 1) * 100
        # gene-bootstrap not meaningful here (pairs span many unrelated genes);
        # use simple pair-level bootstrap for the point estimate CI instead
        rng_boot = np.random.default_rng(RNG_SEED)
        boot_diffs = []
        diff_bg = bg_d2_real - bg_d2_rand
        for _ in range(N_BOOT):
            samp = rng_boot.choice(len(diff_bg), size=len(diff_bg), replace=True)
            boot_diffs.append(diff_bg[samp].mean())
        lo_bg, hi_bg = np.percentile(boot_diffs, [2.5, 97.5])
        print(f"  [{'between-gene (control)':20s}] N_pairs={n_bg:6d}  "
              f"E[d_real^2]={bg_d2_real.mean():10.2f}  E[d_rand^2]={bg_d2_rand.mean():10.2f}  "
              f"excess={bg_excess_pct:+6.1f}%  diff_CI95=[{lo_bg:.2f},{hi_bg:.2f}]")

        within_over_between = res_all['mean_d2_real'] / bg_d2_real.mean()
        print(f"  → within-gene real-d2 / between-gene real-d2 = {within_over_between:.3f}  "
              f"({'gene-SPECIFIC tightness' if within_over_between < 0.9 else 'no gene-specificity (global collapse)'})")

        results[tag] = {'label': label, 'n_layers': n_layers,
                         'all_pairs': res_all, 'length_matched_pairs': res_matched,
                         'between_gene_control': {
                             'n_pairs': int(n_bg), 'mean_d2_real': float(bg_d2_real.mean()),
                             'mean_d2_rand': float(bg_d2_rand.mean()), 'excess_pct': float(bg_excess_pct),
                             'diff_ci95': [float(lo_bg), float(hi_bg)]},
                         'within_over_between_real_d2_ratio': float(within_over_between)}

    outpath = f'{OUT_DIR}/delta_geometry.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {outpath}")
    print(f"[elapsed] {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
