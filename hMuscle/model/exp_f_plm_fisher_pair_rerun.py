#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_f_plm_fisher_pair_rerun.py
--------------------------------
Re-run concat[phi_final || delta] AUPRC using Fisher-sweep-identified mid layers
instead of the naive 50%-depth assumption, for ProtT5-XL and Ankh-base.

Mid layer chosen from TRAIN-split median per-GO Fisher peak (leakage-safe — the
test-split Fisher curve is not used for this design choice, only for the earlier
diagnostic characterization).

  ProtT5-XL: L_mid 12 (50% depth, baseline) -> 15 (train median Fisher peak, 62% depth)
  Ankh-base: L_mid 24 (50% depth, baseline) -> 20 (train median Fisher peak = test median, 42% depth)

Compares against the existing 50%-depth results already in results.tsv.
"""
import os, json
import numpy as np
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from exp_f_plm_scale_scan import load_labels, run_mlp, SEEDS

DATA_DIR = '../data'
OUT_DIR  = '../../reports/exp_f_plm_scale'

CONFIGS = [
    ('prot_t5_xl', 24, 12, 15, 'ProtT5-XL'),
    ('ankh_base',  48, 24, 20, 'Ankh-base'),
]


def load_pair(tag, L_final, L_mid):
    ftr = f'{DATA_DIR}/esm2_train_human_layer{L_final:02d}_{tag}.npy'
    mtr = f'{DATA_DIR}/esm2_train_human_layer{L_mid:02d}_{tag}.npy'
    fte = f'{DATA_DIR}/esm2_layer_{L_final:02d}_{tag}.npy'
    mte = f'{DATA_DIR}/esm2_layer_{L_mid:02d}_{tag}.npy'
    return (np.load(ftr).astype(np.float32), np.load(mtr).astype(np.float32),
            np.load(fte).astype(np.float32), np.load(mte).astype(np.float32))


def main():
    print("=" * 70)
    print("  Fisher-identified layer pair rerun (vs 50%-depth baseline)")
    print("=" * 70)
    Y_tr, Y_te, valid_mask = load_labels()
    print(f"  Valid GO: {valid_mask.sum()}")

    results = {}
    for tag, n_layers, L_mid_base, L_mid_fisher, label in CONFIGS:
        print(f"\n{'-'*60}\n  {label}  (L_final={n_layers}, "
              f"baseline L_mid={L_mid_base} 50%, fisher L_mid={L_mid_fisher})")

        phi_f_tr, phi_m_tr, phi_f_te, phi_m_te = load_pair(tag, n_layers, L_mid_fisher)
        delta_tr = phi_f_tr - phi_m_tr
        delta_te = phi_f_te - phi_m_te
        cat_tr   = np.concatenate([phi_f_tr, delta_tr], axis=1)
        cat_te   = np.concatenate([phi_f_te, delta_te], axis=1)

        auprc_delta, std_delta = run_mlp(delta_tr, Y_tr, delta_te, Y_te, valid_mask,
                                          SEEDS, f'delta(L{n_layers}-L{L_mid_fisher}) [fisher]')
        auprc_cat, std_cat = run_mlp(cat_tr, Y_tr, cat_te, Y_te, valid_mask,
                                      SEEDS, f'concat [fisher pair]')

        results[tag] = {
            'label': label, 'L_final': n_layers,
            'L_mid_baseline_50pct': L_mid_base, 'L_mid_fisher': L_mid_fisher,
            'delta_only_fisher': round(auprc_delta, 4), 'delta_only_fisher_std': round(std_delta, 4),
            'concat_fisher': round(auprc_cat, 4), 'concat_fisher_std': round(std_cat, 4),
        }

    # ── Load baseline (50%-depth) numbers already computed ──────────────────
    baseline = {}
    with open(f'{OUT_DIR}/results.tsv') as f:
        hdr = f.readline().strip().split('\t')
        for line in f:
            row = dict(zip(hdr, line.strip().split('\t')))
            if row['model'] in ('ProtT5-XL', 'Ankh-base'):
                baseline[row['model']] = row

    print(f"\n{'='*70}\n  COMPARISON: 50%-depth baseline vs Fisher-identified pair\n{'='*70}")
    for tag, n_layers, L_mid_base, L_mid_fisher, label in CONFIGS:
        b = baseline.get(label, {})
        r = results[tag]
        print(f"\n  {label}:")
        print(f"    delta_only : baseline(L{L_mid_base})={b.get('delta_only','?')}  "
              f"-> fisher(L{L_mid_fisher})={r['delta_only_fisher']:.4f}  "
              f"Δ={r['delta_only_fisher'] - float(b.get('delta_only', 0)):+.4f}")
        print(f"    concat     : baseline(L{L_mid_base})={b.get('concat_delta','?')}  "
              f"-> fisher(L{L_mid_fisher})={r['concat_fisher']:.4f}  "
              f"Δ={r['concat_fisher'] - float(b.get('concat_delta', 0)):+.4f}")
        results[tag]['baseline_delta_only'] = b.get('delta_only')
        results[tag]['baseline_concat'] = b.get('concat_delta')

    with open(f'{OUT_DIR}/fisher_pair_rerun.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {OUT_DIR}/fisher_pair_rerun.json")


if __name__ == '__main__':
    main()
