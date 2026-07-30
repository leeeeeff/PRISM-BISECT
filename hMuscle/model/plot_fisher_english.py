"""
plot_fisher_english.py
======================
English version of Fig 1 for natcomm_Flow.md:
Per-GO Fisher signal (LR AUPRC) across ESM-2 layers, with peak marked.
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family'    : ['DejaVu Sans'],
    'font.size'      : 11,
    'axes.linewidth' : 1.0,
    'axes.unicode_minus': False,
})

ROOT     = Path("/home/welcome1/sw1686/DIFFUSE")
PROBE    = ROOT / "reports/layer_probe"
OUT      = ROOT / "reports/curve_sweep"

GO_18 = {
    "GO:0007204": "Ca2+ signaling (MID)",
    "GO:0045214": "Sarcomere organization",
    "GO:0006941": "Muscle contraction",
    "GO:0006914": "Autophagy",
    "GO:0043161": "Proteasome / UPS",
    "GO:0007519": "Skeletal muscle development",
    "GO:0042692": "Muscle cell differentiation",
    "GO:0055074": "Ca2+ homeostasis",
    "GO:0007005": "Mitochondrion organization",
    "GO:0007517": "Muscle organ development",
    "GO:0032006": "TOR signaling",
    "GO:0030048": "Actin-based movement",
    "GO:0006096": "Glycolysis",
    "GO:0007268": "Synaptic transmission",
    "GO:0007018": "Microtubule movement (MID)",
    "GO:0031175": "Neuron projection development",
    "GO:0030182": "Neuron differentiation",
    "GO:0000226": "MT cytoskeleton org. (MID)",
}
MID_GOs = {"GO:0007204", "GO:0007018", "GO:0000226"}
W_HALF  = 5


def load_fisher():
    all_lr = {}
    for fname in ["layer_probe_v15d_terms_results.json",
                  "layer_probe_expanded_results.json",
                  "layer_probe_results.json"]:
        p = PROBE / fname
        if p.exists():
            d = json.load(open(p))
            all_lr.update(d["lr_auprc"])
    return {go: np.array(all_lr[go], dtype=np.float32) for go in GO_18}


def main():
    fisher = load_fisher()
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    fig.suptitle(
        "Per-GO Fisher discriminative signal across ESM-2 transformer layers\n"
        "(shaded blue = ±5-layer window around peak used by v20b Flow model)",
        fontsize=14, fontweight='bold', y=1.005
    )

    peaks_list = []

    for idx, go in enumerate(GO_18):
        ax = axes[idx // 6][idx % 6]
        fs = fisher[go]
        layers = np.arange(1, 31)
        peak = int(np.argmax(fs))       # 0-indexed
        peaks_list.append(peak + 1)     # store 1-indexed

        # Window layers
        lo = max(0, peak - W_HALF)
        hi = min(29, peak + W_HALF)
        win_layers_1idx = np.arange(lo + 1, hi + 2)  # 1-indexed
        win_mask = (layers >= (lo + 1)) & (layers <= (hi + 1))

        # Grey bars (all layers)
        ax.bar(layers[~win_mask], fs[~win_mask],
               color='#B0BEC5', width=0.85, edgecolor='none')
        # Blue bars (window)
        ax.bar(layers[win_mask], fs[win_mask],
               color='#1E88E5', width=0.85, edgecolor='none',
               alpha=0.9)
        # Peak marker
        ax.axvline(peak + 1, color='#D81B60', ls='--', lw=1.2, alpha=0.9)
        ax.text(peak + 1, fs.max() * 1.02, f'L{peak+1}',
                fontsize=8, color='#D81B60', ha='center', fontweight='bold')

        is_mid = go in MID_GOs
        title_col = '#B71C1C' if is_mid else '#1A237E'
        ax.set_title(GO_18[go], fontsize=9.5, color=title_col,
                     fontweight='bold' if is_mid else 'normal', pad=3)
        ax.set_xlabel('ESM-2 layer', fontsize=8)
        ax.set_ylabel('LR-probe AUPRC', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0.5, 30.5)
        ax.grid(True, axis='y', alpha=0.25, linestyle=':')

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(OUT / "fig_fisher_profiles_en.png", dpi=140, bbox_inches='tight')
    fig.savefig(OUT / "fig_fisher_profiles_en.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {OUT}/fig_fisher_profiles_en.png")

    # Summary of peaks
    print("\nFisher peak layers across 18 BP GO:")
    print(f"  median = L{int(np.median(peaks_list))}")
    print(f"  IQR    = [L{int(np.percentile(peaks_list, 25))}, "
          f"L{int(np.percentile(peaks_list, 75))}]")
    print(f"  range  = [L{min(peaks_list)}, L{max(peaks_list)}]")
    n_L30 = sum(1 for p in peaks_list if p == 30)
    print(f"  # GO with peak at L30: {n_L30}/{len(peaks_list)}")


if __name__ == "__main__":
    main()
