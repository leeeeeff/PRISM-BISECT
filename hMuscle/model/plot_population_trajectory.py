"""
plot_population_trajectory.py
==============================
Fig 3 for natcomm_Flow.md — population-scale trajectory statistics.

Panels:
  (a) L1 vs L30 pair-distance scatter (subsample of 5000 pairs),
      overlaid with CONV & DIV filter boxes.
  (b) Histogram of L30/L1 ratio (population), with case-set ratios overlaid.
  (c) Layer-wise pairwise distance trajectory: mean ± IQR across population
      + individual conv/div case trajectories.
  (d) Convergence/divergence layer distribution (histogram over case set).
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family'      : ['DejaVu Sans'],
    'font.size'        : 11,
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 11,
    'axes.linewidth'   : 1.0,
    'axes.unicode_minus': False,
})

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
CACHE = ROOT / "reports/v20_cache"
OUT = ROOT / "reports/curve_sweep"
CDJSON = ROOT / "reports/curve_sweep/convergence_divergence.json"


def main():
    print("[1] Loading Z cache …")
    Z = np.load(CACHE / "Z_te.npy")  # (36748, 30, 8)
    N = Z.shape[0]

    print("[2] Loading conv/div cases …")
    d = json.load(open(CDJSON))
    conv = d["convergent_top"]
    div = d["divergent_top"]

    # Subsample pairs
    n_iso = 5000
    rng = np.random.default_rng(42)
    idx = rng.choice(N, n_iso, replace=False)
    Zs = Z[idx]

    def pd(L):
        X = Zs[:, L, :]
        return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))

    print("[3] Computing L1, L30 distance matrices …")
    D_L1 = pd(0)
    D_L30 = pd(29)
    iu = np.triu_indices(n_iso, 1)
    d1_pop = D_L1[iu]
    d30_pop = D_L30[iu]
    del D_L1, D_L30
    print(f"    n_pairs = {len(d1_pop):,}")

    # For panel (c), compute mean ± IQR distance per layer over the population
    print("[4] Computing per-layer distance profile …")
    n_traj = 400  # limit to reduce compute
    tr_idx = rng.choice(n_iso, n_traj, replace=False)
    Ztraj = Zs[tr_idx]
    per_layer_stats = np.zeros((30, 4))  # p25, median, p75, mean
    for L in range(30):
        DL = np.sqrt(((Ztraj[:, L, :][:, None, :] -
                        Ztraj[:, L, :][None, :, :]) ** 2).sum(-1))
        vals = DL[np.triu_indices(n_traj, 1)]
        per_layer_stats[L] = [np.percentile(vals, 25),
                              np.median(vals),
                              np.percentile(vals, 75),
                              vals.mean()]

    # ── Figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13))
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.22,
                       left=0.06, right=0.97, top=0.94, bottom=0.06)
    fig.suptitle(
        "Population-scale trajectory statistics (test set, n=36,748 isoforms)",
        fontsize=15, fontweight='bold', y=0.985)

    # (a) L1 vs L30 scatter with filter boxes
    ax_a = fig.add_subplot(gs[0, 0])
    # subsample for scatter
    n_show = 20000
    sh_idx = rng.choice(len(d1_pop), min(n_show, len(d1_pop)), replace=False)
    ax_a.scatter(d1_pop[sh_idx], d30_pop[sh_idx], s=1.5,
                 c='#B0BEC5', alpha=0.35, edgecolors='none')

    # Overlay CONV cases
    d1_conv = [c["d_L1"] for c in conv]
    d30_conv = [c["d_L30"] for c in conv]
    ax_a.scatter(d1_conv, d30_conv, s=180, c='#1E88E5', marker='o',
                 edgecolors='navy', lw=1.6, zorder=5,
                 label=f'CONV cases (n={len(conv)})')

    # Overlay DIV cases
    d1_div = [c["d_L1"] for c in div]
    d30_div = [c["d_L30"] for c in div]
    ax_a.scatter(d1_div, d30_div, s=180, c='#43A047', marker='D',
                 edgecolors='darkgreen', lw=1.6, zorder=5,
                 label=f'DIV cases (n={len(div)})')

    # Filter boxes
    ax_a.add_patch(plt.Rectangle((8, 0), 20, 6, fill=False,
                                  edgecolor='#1E88E5', lw=2.0, ls='--',
                                  label='CONV filter'))
    ax_a.add_patch(plt.Rectangle((0, 15), 5, 25, fill=False,
                                  edgecolor='#43A047', lw=2.0, ls='--',
                                  label='DIV filter'))

    # Diagonal reference lines
    ax_a.plot([0, 40], [0, 40], color='black', lw=0.8, alpha=0.4,
              label='d(L1) = d(L30)')

    ax_a.set_xlabel('Pairwise distance at layer 1  |  d(L1)')
    ax_a.set_ylabel('Pairwise distance at layer 30  |  d(L30)')
    ax_a.set_title('(a) L1 vs L30 pair-distance scatter\n'
                   f'population subsample: {len(sh_idx):,} pairs')
    ax_a.set_xlim(0, 25)
    ax_a.set_ylim(0, 40)
    ax_a.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax_a.grid(True, alpha=0.3, linestyle=':')

    # (b) L30/L1 ratio histogram
    ax_b = fig.add_subplot(gs[0, 1])
    valid = d1_pop > 0.5
    ratio_pop = d30_pop[valid] / d1_pop[valid]
    ax_b.hist(ratio_pop, bins=100, range=(0, 12), color='#B0BEC5',
              edgecolor='none', alpha=0.9)

    # Case ratios
    r_conv = [c["d_L30"] / c["d_L1"] for c in conv]
    r_div = [c["d_L30"] / c["d_L1"] for c in div]

    for r in r_conv:
        ax_b.axvline(r, color='#1E88E5', lw=1.8, alpha=0.8)
    for r in r_div:
        ax_b.axvline(r, color='#43A047', lw=1.8, alpha=0.8)

    # Stats annotations
    med = np.median(ratio_pop)
    p1 = np.percentile(ratio_pop, 1)
    p99 = np.percentile(ratio_pop, 99)
    ax_b.axvline(med, color='red', lw=2.4, ls='--',
                 label=f'population median = {med:.2f}')
    ax_b.axvline(p1, color='#455A64', lw=1.2, ls=':',
                 label=f'p1 = {p1:.2f}')
    ax_b.axvline(p99, color='#455A64', lw=1.2, ls=':',
                 label=f'p99 = {p99:.2f}')

    ax_b.set_xlabel('L30 / L1 pair-distance ratio')
    ax_b.set_ylabel('Number of pairs')
    ax_b.set_title(f'(b) L30/L1 ratio histogram (n = {valid.sum():,} pairs)\n'
                   'Blue lines = CONV cases (ratio ≪ 1)  |  '
                   'Green lines = DIV cases (ratio ≫ 1)')
    ax_b.set_yscale('log')
    ax_b.legend(fontsize=9, loc='upper right')
    ax_b.grid(True, alpha=0.3, linestyle=':')

    # (c) Per-layer distance profile
    ax_c = fig.add_subplot(gs[1, 0])
    layers = np.arange(1, 31)
    ax_c.plot(layers, per_layer_stats[:, 3], color='#455A64', lw=2.5,
              label=f'population mean (subsample n={n_traj})')
    ax_c.fill_between(layers,
                       per_layer_stats[:, 0], per_layer_stats[:, 2],
                       color='#455A64', alpha=0.20,
                       label='population IQR')

    # Overlay each conv/div case's per-layer distance
    for c in conv:
        d_seq = c.get("d_seq", None)
        if d_seq is None:
            i_a, i_b = c["i_a"], c["i_b"]
            d_seq = np.sqrt(((Z[i_a] - Z[i_b]) ** 2).sum(-1))
        ax_c.plot(layers, d_seq, color='#1E88E5', lw=1.4, alpha=0.7)
    ax_c.plot([], [], color='#1E88E5', lw=1.4, label='CONV cases')

    for c in div:
        d_seq = c.get("d_seq", None)
        if d_seq is None:
            i_a, i_b = c["i_a"], c["i_b"]
            d_seq = np.sqrt(((Z[i_a] - Z[i_b]) ** 2).sum(-1))
        ax_c.plot(layers, d_seq, color='#43A047', lw=1.4, alpha=0.7)
    ax_c.plot([], [], color='#43A047', lw=1.4, label='DIV cases')

    ax_c.set_xlabel('ESM-2 layer L')
    ax_c.set_ylabel('Pair-wise distance d(A, B) at layer L')
    ax_c.set_title('(c) Layer-wise pair-distance trajectory\n'
                   'population IQR envelope + individual case trajectories')
    ax_c.set_xlim(1, 30)
    ax_c.grid(True, alpha=0.3, linestyle=':')
    ax_c.legend(fontsize=9, loc='upper left')

    # (d) Conv/div layer distribution
    ax_d = fig.add_subplot(gs[1, 1])
    conv_L = [c["conv_layer"] for c in conv]
    div_L = [c["div_layer"] for c in div]
    bins = np.arange(1, 32) - 0.5
    ax_d.hist([conv_L, div_L], bins=bins,
              color=['#1E88E5', '#43A047'],
              label=[f'CONV convergence layer (median L{int(np.median(conv_L))})',
                     f'DIV divergence layer (median L{int(np.median(div_L))})'],
              edgecolor='white', linewidth=0.8)
    ax_d.set_xlabel('Transition layer (convergence for CONV, divergence for DIV)')
    ax_d.set_ylabel('Number of cases')
    ax_d.set_title('(d) Transition-layer distribution across case sets')
    ax_d.set_xlim(0.5, 30.5)
    ax_d.legend(fontsize=9, loc='upper right')
    ax_d.grid(True, alpha=0.3, linestyle=':', axis='y')

    fig.savefig(OUT / "fig_population_trajectory.png", dpi=140,
                bbox_inches='tight')
    fig.savefig(OUT / "fig_population_trajectory.pdf",
                bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {OUT}/fig_population_trajectory.png")

    # Print summary stats for manuscript
    print("\nPopulation statistics for manuscript:")
    print(f"  n pairs subsampled: {len(d1_pop):,}")
    print(f"  L1  distance: mean={d1_pop.mean():.2f} median={np.median(d1_pop):.2f}")
    print(f"  L30 distance: mean={d30_pop.mean():.2f} median={np.median(d30_pop):.2f}")
    print(f"  L30/L1 ratio: median={med:.3f} p1={p1:.3f} p99={p99:.3f}")
    print(f"  CONV cases: mean L30/L1 = {np.mean(r_conv):.3f}")
    print(f"  DIV  cases: mean L30/L1 = {np.mean(r_div):.3f}")
    print(f"  CONV convergence layer median = L{int(np.median(conv_L))}")
    print(f"  DIV  divergence layer median  = L{int(np.median(div_L))}")


if __name__ == "__main__":
    main()
