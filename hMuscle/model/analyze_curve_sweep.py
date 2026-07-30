"""
analyze_curve_sweep.py
======================
curve window sweep 결과 분석 + 시각화.
sweep_results.json 로드 후 즉시 실행.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

REPORTS  = Path("/home/welcome1/sw1686/DIFFUSE/reports")
OUT_DIR  = REPORTS / "curve_sweep"
PROBE_DIR= REPORTS / "layer_probe"
FIG_DIR  = OUT_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── References ─────────────────────────────────────────────────────
V15D_AUPRC  = 0.7022;  V15D_RATIO  = 0.3377
V19_AUPRC   = 0.6721;  V19_RATIO   = 0.3843
MID_GOs     = {"GO:0007204", "GO:0007018", "GO:0000226"}
GO_18 = ["GO:0007204","GO:0045214","GO:0006941","GO:0006914","GO:0043161",
         "GO:0007519","GO:0042692","GO:0055074","GO:0007005","GO:0007517",
         "GO:0032006","GO:0030048","GO:0006096","GO:0007268","GO:0007018",
         "GO:0031175","GO:0030182","GO:0000226"]

def load_fisher():
    all_lr = {}
    for fname in ["layer_probe_v15d_terms_results.json",
                  "layer_probe_expanded_results.json",
                  "layer_probe_results.json"]:
        d = json.load(open(PROBE_DIR / fname))
        all_lr.update(d["lr_auprc"])
    return {go: np.array(all_lr[go]) for go in GO_18}


# ── Load results ───────────────────────────────────────────────────
d = json.load(open(OUT_DIR / "sweep_results.json"))
base_auprc = d["baseline_L30_only"]["macro_auprc"]
base_ratio = d["baseline_L30_only"]["t3_t12_ratio"]
base_pos   = d["baseline_L30_only"]["pos_mean"]

tk  = d["strategy_A_topk"]
win = d["strategy_B_window"]

K_vals   = [r["K"]     for r in tk]
K_auprc  = [r["macro_auprc"]  for r in tk]
K_ratio  = [r["t3_t12_ratio"] for r in tk]
K_pos    = [r["pos_mean"]     for r in tk]
K_dim    = [r["dim"]          for r in tk]

W_vals   = [r["w"]     for r in win]
W_auprc  = [r["macro_auprc"]  for r in win]
W_ratio  = [r["t3_t12_ratio"] for r in win]
W_dim    = [r["dim_approx"]   for r in win]

print("="*70)
print("  Curve Layer Selection Sweep — 분석 결과")
print("="*70)

# ── Numeric table ──────────────────────────────────────────────────
print("\n[Strategy A: Top-K layers (GO-specific, non-contiguous)]")
print(f"  {'K':>3} {'dim':>5} {'AUPRC':>7} {'Δv15d':>7} {'T3/T12':>7} "
      f"{'ΔBase':>7} {'pos_mean':>8}  score")
print(f"  {'0(LR)':>5} {640:>5} {base_auprc:>7.4f} {base_auprc-V15D_AUPRC:>+7.4f} "
      f"{base_ratio:>7.4f} {'---':>7} {base_pos:>8.4f}  [LR baseline]")

# composite score: T3/T12 gain vs v15d, penalize LR-AUPRC drop vs LR-baseline
best_score_k, best_k = -999, 1
for r in tk:
    auprc_loss = base_auprc - r["macro_auprc"]   # LR proxy drop vs LR-baseline (want low)
    ratio_gain = r["t3_t12_ratio"] - V15D_RATIO  # T3/T12 gain vs v15d MLP (want high)
    score = ratio_gain - 2.0 * max(0, auprc_loss - 0.005)
    if score > best_score_k:
        best_score_k = score; best_k = r["K"]
    print(f"  {r['K']:>3} {r['dim']:>5} {r['macro_auprc']:>7.4f} "
          f"{r['macro_auprc']-V15D_AUPRC:>+7.4f} {r['t3_t12_ratio']:>7.4f} "
          f"{r['t3_t12_ratio']-base_ratio:>+7.4f} {r['pos_mean']:>8.4f}  "
          f"composite={score:+.4f}")

print(f"\n  → Best K = {best_k} (composite score)")

print(f"\n[Strategy B: Window ±w (contiguous around peak)]")
print(f"  {'±w':>3} {'n_L':>4} {'dim':>5} {'AUPRC':>7} {'Δv15d':>7} {'T3/T12':>7} {'ΔBase':>7}")
best_score_w, best_w = -999, 0
for r in win:
    auprc_loss = base_auprc - r["macro_auprc"]
    ratio_gain = r["t3_t12_ratio"] - V15D_RATIO
    score = ratio_gain - 2.0 * max(0, auprc_loss - 0.005)
    if score > best_score_w:
        best_score_w = score; best_w = r["w"]
    print(f"  {r['w']:>3} {2*r['w']+1:>4} {r['dim_approx']:>5} "
          f"{r['macro_auprc']:>7.4f} {r['macro_auprc']-V15D_AUPRC:>+7.4f} "
          f"{r['t3_t12_ratio']:>7.4f} {r['t3_t12_ratio']-base_ratio:>+7.4f}  "
          f"composite={score:+.4f}")

print(f"\n  → Best w = {best_w} (composite score)")

# ── FIGURE 1: Main sweep curves ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Curve Layer Selection Sweep\n"
             "Goal: maximize T3/T12 ratio while keeping AUPRC ≥ v15d − 0.01",
             fontsize=13, fontweight='bold')

# Color scheme
C_TOPK  = '#2196F3'
C_WIN   = '#FF5722'
C_V15D  = '#4CAF50'
C_V19   = '#9E9E9E'
C_IDEAL = '#FF9800'

# ── Panel (0,0): AUPRC vs K ────────────────────────────────────────
ax = axes[0, 0]
ax.axhline(V15D_AUPRC, color=C_V15D, ls='--', lw=1.5, label='v15d MLP (0.7022)')
ax.axhline(V19_AUPRC,  color=C_V19,  ls='--', lw=1.2, label='v19 curve240 (0.6721)')
ax.axhline(V15D_AUPRC - 0.01, color='red', ls=':', lw=1, alpha=0.5, label='허용 한계 (v15d−0.01)')
ax.plot(K_vals, K_auprc, 'o-', color=C_TOPK, lw=2, ms=7, label='Top-K (GO-specific)')
ax.plot([2*w+1 for w in W_vals], W_auprc, 's--', color=C_WIN, lw=2, ms=6, label='Window ±w')
ax.axhline(base_auprc, color='purple', ls=':', lw=1.2, alpha=0.7, label=f'LR baseline ({base_auprc:.3f})')
# Mark best K
bk_auprc = next(r['macro_auprc'] for r in tk if r['K'] == best_k)
ax.axvline(best_k, color=C_IDEAL, ls='--', lw=1.5, alpha=0.7)
ax.scatter([best_k], [bk_auprc], color=C_IDEAL, s=180, zorder=5, marker='*',
           label=f'Best K={best_k}')
ax.set_xlabel('선택 Layer 수 (K 또는 2w+1)')
ax.set_ylabel('Macro AUPRC (LR proxy)')
ax.set_title('① AUPRC vs Layer 수')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 32)

# ── Panel (0,1): T3/T12 ratio vs K ────────────────────────────────
ax = axes[0, 1]
ax.axhline(V15D_RATIO, color=C_V15D, ls='--', lw=1.5, label=f'v15d ({V15D_RATIO:.3f})')
ax.axhline(V19_RATIO,  color=C_V19,  ls='--', lw=1.2, label=f'v19 ({V19_RATIO:.3f})')
ax.plot(K_vals, K_ratio, 'o-', color=C_TOPK, lw=2, ms=7, label='Top-K')
ax.plot([2*w+1 for w in W_vals], W_ratio, 's--', color=C_WIN, lw=2, ms=6, label='Window ±w')
ax.axhline(base_ratio, color='purple', ls=':', lw=1.2, alpha=0.7, label=f'LR base ({base_ratio:.3f})')
bk_ratio = next(r['t3_t12_ratio'] for r in tk if r['K'] == best_k)
ax.axvline(best_k, color=C_IDEAL, ls='--', lw=1.5, alpha=0.7)
ax.scatter([best_k], [bk_ratio], color=C_IDEAL, s=180, zorder=5, marker='*',
           label=f'Best K={best_k}')
ax.set_xlabel('선택 Layer 수')
ax.set_ylabel('T3/T12 spread ratio')
ax.set_title('② Type-3 분리력 vs Layer 수')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 32)

# ── Panel (1,0): AUPRC-ratio trade-off scatter ─────────────────────
ax = axes[1, 0]
sc = ax.scatter(K_ratio, K_auprc, c=K_vals, cmap='Blues', s=100,
                zorder=3, edgecolors='navy', lw=0.5, label='Top-K')
for i, (r, a, k) in enumerate(zip(K_ratio, K_auprc, K_vals)):
    ax.annotate(f'K={k}', (r, a), textcoords='offset points',
                xytext=(4, 3), fontsize=7, color='navy')
sc2 = ax.scatter(W_ratio, W_auprc, c=[2*w+1 for w in W_vals],
                 cmap='Oranges', s=80, marker='s', zorder=3,
                 edgecolors='darkred', lw=0.5, label='Window')
for i, (r, a, w) in enumerate(zip(W_ratio, W_auprc, W_vals)):
    ax.annotate(f'±{w}', (r, a), textcoords='offset points',
                xytext=(4, -8), fontsize=7, color='darkred')

ax.scatter([V15D_RATIO], [V15D_AUPRC], color=C_V15D, s=200, marker='*',
           zorder=5, label='v15d MLP', edgecolors='darkgreen', lw=1)
ax.scatter([V19_RATIO], [V19_AUPRC], color=C_V19, s=150, marker='D',
           zorder=5, label='v19 curve240', edgecolors='gray', lw=1)
ax.scatter([base_ratio], [base_auprc], color='purple', s=120, marker='P',
           zorder=5, label='LR baseline', edgecolors='purple', lw=1)

# Ideal zone: AUPRC > v15d-0.01 AND ratio > v15d+0.01
ax.axhline(V15D_AUPRC-0.01, color='red', ls=':', lw=1, alpha=0.5)
ax.axvline(V15D_RATIO+0.01, color='green', ls=':', lw=1, alpha=0.5)
ax.fill_betweenx([V15D_AUPRC-0.01, V15D_AUPRC+0.05],
                  V15D_RATIO+0.01, 0.55, alpha=0.08, color='gold',
                  label='이상 영역')
ax.set_xlabel('T3/T12 spread ratio (↑ Type-3 분리 좋음)')
ax.set_ylabel('Macro AUPRC (↑ 좋음)')
ax.set_title('③ AUPRC-T3분리 trade-off\n(우상단 = 이상)')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

# ── Panel (1,1): Composite score ──────────────────────────────────
ax = axes[1, 1]
composite_k = [r["t3_t12_ratio"]-V15D_RATIO
               - 2.0*max(0, base_auprc - r["macro_auprc"] - 0.005)
               for r in tk]
composite_w = [r["t3_t12_ratio"]-V15D_RATIO
               - 2.0*max(0, base_auprc - r["macro_auprc"] - 0.005)
               for r in win]

bars_k = ax.bar([k-0.3 for k in K_vals], composite_k, 0.5,
                color=[C_TOPK if c >= 0 else '#EF9A9A' for c in composite_k],
                alpha=0.8, label='Top-K', edgecolor='navy', lw=0.5)
bars_w = ax.bar([2*w+1+0.3 for w in W_vals], composite_w, 0.5,
                color=[C_WIN if c >= 0 else '#EF9A9A' for c in composite_w],
                alpha=0.8, label='Window', edgecolor='darkred', lw=0.5)

ax.axhline(0, color='black', lw=1.2)
ax.axvline(best_k, color=C_IDEAL, ls='--', lw=1.5, alpha=0.8,
           label=f'Best K={best_k}')
ax.set_xlabel('Layer 수')
ax.set_ylabel('Composite score\n= ΔT3ratio(vs v15d) − 2×max(0, LR↓>0.5%)')
ax.set_title('④ 균형점 composite score\n(양수 = T3분리 순이익 / 음수 = LR AUPRC↓ 패널티 초과)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(0, 32)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_sweep_main.png", dpi=150, bbox_inches='tight')
plt.savefig(FIG_DIR / "fig_sweep_main.pdf", bbox_inches='tight')
plt.close()
print(f"\n[Figure 1 saved] fig_sweep_main.png")

# ── FIGURE 2: Per-GO Fisher profile ───────────────────────────────
fisher = load_fisher()
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
fig.suptitle("GO별 per-layer Fisher Signal (LR AUPRC)\n"
             "색 영역 = Top-K 선택 범위 (K=best_k)",
             fontsize=12, fontweight='bold')

go_names = {
    "GO:0007204": "Ca2+ signal (MID)",   "GO:0045214": "Sarcomere org",
    "GO:0006941": "Muscle contract",     "GO:0006914": "Autophagy",
    "GO:0043161": "Proteasome-UPS",      "GO:0007519": "Skeletal musc dev",
    "GO:0042692": "Muscle cell diff",    "GO:0055074": "Ca2+ homeos",
    "GO:0007005": "Mitochondrion org",   "GO:0007517": "Muscle organ dev",
    "GO:0032006": "TOR signaling",       "GO:0030048": "Actin-based mov",
    "GO:0006096": "Glycolysis",          "GO:0007268": "Synaptic transm",
    "GO:0007018": "MT-based mov (MID)",  "GO:0031175": "Neuron proj dev",
    "GO:0030182": "Neuron diff",         "GO:0000226": "MT cytosk org (MID)",
}

for idx, go in enumerate(GO_18):
    ax = axes[idx // 6][idx % 6]
    fs = fisher[go]
    layers = np.arange(1, 31)
    peak = int(np.argmax(fs))

    # Top-K layers for best_k
    top_k_idx = np.argsort(fs)[::-1][:best_k]

    ax.bar(layers, fs, color='#B0BEC5', width=0.8, label='all layers')
    ax.bar(layers[top_k_idx], fs[top_k_idx], color=C_TOPK, width=0.8,
           alpha=0.9, label=f'Top-{best_k}')
    ax.axvline(peak+1, color='red', ls='--', lw=1.2, alpha=0.7)

    is_mid = go in MID_GOs
    color = '#E53935' if is_mid else '#1565C0'
    ax.set_title(go_names.get(go, go), fontsize=7, color=color, fontweight='bold' if is_mid else 'normal')
    ax.set_xlim(0, 31)
    ax.tick_params(labelsize=6)
    ax.set_xlabel('Layer', fontsize=6)
    if idx % 6 == 0:
        ax.set_ylabel('LR AUPRC', fontsize=6)

    # Annotate peak
    ax.text(peak+1, fs[peak]*1.02, f'L{peak+1}', ha='center', fontsize=6, color='red')

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_fisher_profiles.png", dpi=150, bbox_inches='tight')
plt.savefig(FIG_DIR / "fig_fisher_profiles.pdf", bbox_inches='tight')
plt.close()
print(f"[Figure 2 saved] fig_fisher_profiles.png")

# ── FIGURE 3: Optimal K decision ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"균형점 결정 — Best K={best_k} (Top-K), Best w={best_w} (Window)",
             fontsize=12, fontweight='bold')

# Panel A: AUPRC + ratio dual axis
ax1 = axes[0]
ax2 = ax1.twinx()
l1, = ax1.plot(K_vals, K_auprc, 'o-', color='#1565C0', lw=2.5, ms=8, label='AUPRC (Left)')
l2, = ax2.plot(K_vals, K_ratio, 's--', color='#E53935', lw=2.5, ms=8, label='T3/T12 ratio (Right)')
ax1.axhline(V15D_AUPRC, color='#1565C0', ls=':', lw=1, alpha=0.5)
ax1.axhline(V15D_AUPRC-0.01, color='red', ls=':', lw=1, alpha=0.4)
ax2.axhline(V15D_RATIO, color='#E53935', ls=':', lw=1, alpha=0.5)
ax1.axvline(best_k, color='orange', ls='--', lw=2, label=f'Best K={best_k}')
ax1.fill_between([best_k-0.5, best_k+0.5], [V15D_AUPRC-0.02]*2, [V15D_AUPRC+0.02]*2,
                  alpha=0.15, color='orange')
ax1.set_xlabel('K (Top-K layers)'); ax1.set_ylabel('Macro AUPRC', color='#1565C0')
ax2.set_ylabel('T3/T12 ratio', color='#E53935')
ax1.set_title('Top-K: AUPRC vs T3분리 dual')
lines = [l1, l2] + [plt.Line2D([0],[0], color='orange', ls='--', lw=2, label=f'Best K={best_k}')]
ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='center right')
ax1.grid(True, alpha=0.3)

# Panel B: Same for Window
ax1 = axes[1]
ax2 = ax1.twinx()
n_layers_w = [2*w+1 for w in W_vals]
l1, = ax1.plot(n_layers_w, W_auprc, 'o-', color='#1565C0', lw=2.5, ms=8)
l2, = ax2.plot(n_layers_w, W_ratio, 's--', color='#E53935', lw=2.5, ms=8)
ax1.axhline(V15D_AUPRC, color='#1565C0', ls=':', lw=1, alpha=0.5)
ax1.axhline(V15D_AUPRC-0.01, color='red', ls=':', lw=1, alpha=0.4)
ax2.axhline(V15D_RATIO, color='#E53935', ls=':', lw=1, alpha=0.5)
best_w_nlayers = 2*best_w+1
ax1.axvline(best_w_nlayers, color='orange', ls='--', lw=2)
ax1.set_xlabel('Layer 수 (2w+1)'); ax1.set_ylabel('Macro AUPRC', color='#1565C0')
ax2.set_ylabel('T3/T12 ratio', color='#E53935')
ax1.set_title(f'Window ±w: AUPRC vs T3분리 dual\n(Best w={best_w}, {best_w_nlayers}개 layer)')
ax1.grid(True, alpha=0.3)
for wi, n in enumerate(n_layers_w):
    ax1.annotate(f'±{W_vals[wi]}', (n, W_auprc[wi]),
                 textcoords='offset points', xytext=(3, 5), fontsize=7)

# Panel C: Composite score comparison + recommendation
ax = axes[2]
composite_k_arr = np.array(composite_k)
composite_w_arr = np.array(composite_w)
x_k = np.array(K_vals)
x_w = np.array([2*w+1 for w in W_vals])

ax.fill_between([0, 33], [0, 0], alpha=0.05, color='red')
ax.fill_between([0, 33], [0, 0.1], alpha=0.05, color='green')
ax.plot(x_k, composite_k_arr, 'o-', color=C_TOPK, lw=2.5, ms=8, label='Top-K')
ax.plot(x_w, composite_w_arr, 's--', color=C_WIN, lw=2, ms=6, label='Window')
ax.axhline(0, color='black', lw=1.5, label='Break-even (v15d 기준)')
ax.axvline(best_k, color='orange', ls='--', lw=2, label=f'Recommended K={best_k}')
ax.scatter([best_k], [composite_k_arr[K_vals.index(best_k)]],
           s=280, color='orange', zorder=5, marker='*')

# Label each point
for k, c in zip(K_vals, composite_k_arr):
    ax.annotate(f'K={k}\n{c:+.3f}', (k, c),
                textcoords='offset points', xytext=(5, 0), fontsize=6.5,
                color=C_TOPK)

ax.set_xlabel('Layer 수')
ax.set_ylabel('Composite score (ΔT3ratio − LR penalty)')
ax.set_title('균형점 결정\n= ΔT3ratio(vs v15d) − 2×max(0, LR↓>0.5%)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 33)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig_optimal_k.png", dpi=150, bbox_inches='tight')
plt.savefig(FIG_DIR / "fig_optimal_k.pdf", bbox_inches='tight')
plt.close()
print(f"[Figure 3 saved] fig_optimal_k.png")

# ── Summary printout ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  최종 권고")
print(f"{'='*60}")
bk_r  = next(r for r in tk if r['K'] == best_k)
bw_r  = next(r for r in win if r['w'] == best_w)
print(f"  Top-K best:   K={best_k}  dim={bk_r['dim']}")
print(f"    AUPRC={bk_r['macro_auprc']:.4f}  (Δv15d={bk_r['macro_auprc']-V15D_AUPRC:+.4f})")
print(f"    T3/T12={bk_r['t3_t12_ratio']:.4f}  (Δv15d={bk_r['t3_t12_ratio']-V15D_RATIO:+.4f})")
print(f"\n  Window best:  ±w={best_w}  n_layers={2*best_w+1}  dim={bw_r['dim_approx']}")
print(f"    AUPRC={bw_r['macro_auprc']:.4f}  (Δv15d={bw_r['macro_auprc']-V15D_AUPRC:+.4f})")
print(f"    T3/T12={bw_r['t3_t12_ratio']:.4f}  (Δv15d={bw_r['t3_t12_ratio']-V15D_RATIO:+.4f})")
print(f"\n  v20b 권고: {'Top-K K=' + str(best_k) if best_score_k >= best_score_w else 'Window ±' + str(best_w)}")
print(f"  MLP 전체 훈련 후 v15d 대비 AUPRC+T3ratio 동시 개선 확인 필요")
print(f"{'='*60}")
