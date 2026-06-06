"""41GO AUPRC comparison figure — Figure for §3.10 (brain zero-shot evaluation).

Generates:
  reports/figures/fig_41go_auprc_comparison.png
  reports/figures/fig_bisect_tier_distribution.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path
from collections import Counter

OUT_DIR = Path(__file__).parents[1] / "reports/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Fig A: Panel comparison bar chart ─────────────────────────────────────────
def fig_panel_comparison():
    fig, ax = plt.subplots(figsize=(7, 4))

    models   = ["18-term\n(muscle)", "41-term\n(expanded)", "672-term\n(exploratory)"]
    aurpcs   = [0.5998, 0.6724, 0.357]
    colors   = ["#94a3b8", "#2563eb", "#f87171"]
    hatches  = ["", "", "//"]

    bars = ax.bar(models, aurpcs, color=colors, hatch=hatches,
                  edgecolor='white', linewidth=1.2, width=0.55)

    # Value labels
    for bar, val in zip(bars, aurpcs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Improvement annotation
    ax.annotate("", xy=(1, aurpcs[1]), xytext=(0, aurpcs[0]),
                arrowprops=dict(arrowstyle='->', color='#1d4ed8', lw=1.5))
    ax.text(0.5, (aurpcs[0]+aurpcs[1])/2 + 0.012, "+12.1%",
            ha='center', color='#1d4ed8', fontsize=9, fontweight='bold')

    ax.axhline(aurpcs[0], color='#94a3b8', linestyle='--', linewidth=0.8, alpha=0.6)

    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Brain Zero-shot Macro AUPRC", fontsize=11)
    ax.set_title("PRISM GO Panel Size vs Brain Prediction Performance\n"
                 "(Samsung AD cohort, 63,994 isoforms, no brain supervision)",
                 fontsize=10, pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    out = OUT_DIR / "fig_41go_auprc_comparison.png"
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── Fig B: BISECT Tier distribution ───────────────────────────────────────────
def fig_bisect_tiers():
    cases_path = Path(__file__).parents[1] / "prism_app/data/demo/bisect_cases.json"
    with open(cases_path) as f:
        cases = json.load(f)

    tier_labels = {
        'tier1_functional_switch':  'Tier 1\nFunctional Switch',
        'tier2_complex_loss':       'Tier 2\nComplex I Collapse',
        'tier2_functional_loss':    'Tier 2\nFunctional Loss',
        'tier2_partial_change':     'Tier 2\nPartial Change',
        'tier3_gene_median':        'Tier 3\nGene Median',
        'tier3_structural_only':    'Tier 3\nStructural Only',
    }
    tier_colors = {
        'tier1_functional_switch':  '#dc2626',
        'tier2_complex_loss':       '#7f1d1d',
        'tier2_functional_loss':    '#ea580c',
        'tier2_partial_change':     '#d97706',
        'tier3_gene_median':        '#6b7280',
        'tier3_structural_only':    '#9ca3af',
    }

    counts = Counter(c.get('prism_tier') for c in cases)
    order  = list(tier_labels.keys())
    vals   = [counts.get(t, 0) for t in order]
    labels = [tier_labels[t] for t in order]
    colors = [tier_colors[t] for t in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Bar chart
    bars = ax1.barh(labels, vals, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 str(val), va='center', fontsize=10, fontweight='bold')
    ax1.set_xlim(0, max(vals) * 1.2)
    ax1.set_xlabel("Number of BISECT Cases", fontsize=10)
    ax1.set_title("BISECT Functional Tier Distribution\n(n=83 cases)", fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(labelsize=9)

    # Stacked breakdown: brain DTU vs single-condition
    dtu_counts = {'brain_dtu': Counter(), 'single_cond': Counter()}
    for c in cases:
        key = 'brain_dtu' if c.get('dtu_p') is not None else 'single_cond'
        dtu_counts[key][c.get('prism_tier')] += 1

    brain   = [dtu_counts['brain_dtu'].get(t, 0) for t in order]
    single  = [dtu_counts['single_cond'].get(t, 0) for t in order]
    x = np.arange(len(order))
    w = 0.55
    ax2.bar(x, brain,  w, label='Brain (DTU tested)',    color='#3b82f6', alpha=0.85)
    ax2.bar(x, single, w, bottom=brain, label='Muscle/Cardiac\n(single-condition)', color='#fbbf24', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace('\n', ' ') for l in labels], rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel("Cases", fontsize=10)
    ax2.set_title("Tier × Data Source", fontsize=10)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "fig_bisect_tier_distribution.png"
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── Fig C: 3대 AD 수렴 테마 ──────────────────────────────────────────────────
def fig_ad_themes():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

    theme_data = [
        # (x_center, y_center, title, color, genes, n_cases)
        (2.0, 3.5, "(i) Synaptic Scaffold\nRemodeling", "#1d4ed8",
         "DLG1 · PHB2 · KIF21B", "3 cases (Tier 1+2)"),
        (5.0, 3.5, "(ii) Cytoskeletal\nTransport Disruption", "#dc2626",
         "KIF21B · DYNLT1 · SPTBN1\nRHOT2 · MAPRE2 · TRAF3IP1", "12 cases (Tier 2)"),
        (8.0, 3.5, "(iii) Mitochondrial\nEnergy Collapse", "#7c3aed",
         "NDUFS4 · NDUFS7 · NDUFS8\nLETMD1 · SAMM50 · COA1", "10 cases (Tier 2)"),
    ]

    for x, y, title, color, genes, ncases in theme_data:
        # Box
        rect = mpatches.FancyBboxPatch((x-1.7, y-1.4), 3.4, 2.8,
            boxstyle="round,pad=0.15", facecolor=color+'18',
            edgecolor=color, linewidth=2.0)
        ax.add_patch(rect)
        ax.text(x, y+0.9, title, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=color)
        ax.text(x, y+0.05, genes, ha='center', va='center',
                fontsize=7.5, color='#374151', style='italic')
        ax.text(x, y-0.8, ncases, ha='center', va='center',
                fontsize=8, color=color, fontweight='bold')

    # Coupling arrows
    arrow_kw = dict(arrowstyle='<->', color='#374151', lw=1.2,
                    connectionstyle='arc3,rad=-0.15')
    ax.annotate("", xy=(3.35, 3.5), xytext=(6.65, 3.5),
                arrowprops=dict(**arrow_kw))
    ax.text(5.0, 3.1, "MT transport\nimpairs mito position", ha='center',
            fontsize=7, color='#6b7280', style='italic')

    ax.set_title("Three Convergent Functional Themes in AD Isoform Switches (BISECT 83 cases)",
                 fontsize=10.5, pad=12, fontweight='bold')

    plt.tight_layout()
    out = OUT_DIR / "fig_ad_three_themes.png"
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    fig_panel_comparison()
    fig_bisect_tiers()
    fig_ad_themes()
    print("\nAll figures saved to reports/figures/")
