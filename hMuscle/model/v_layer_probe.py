#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v_layer_probe.py
----------------
ESM-2 Layer-wise Probing Analysis (Nature Methods figure)

Evaluates how biological information in each ESM-2 layer affects:
  A. Isoform separation (intra_sim, sep_cosine per GO term)
  B. Function prediction (linear probe AUPRC, gene-stratified 5-fold CV)
  C. Case study distances (DMD, PINK1, PTPRF isoform pairs)

실행:
  cd hMuscle/model/
  conda run -n isoform_env python v_layer_probe.py

출력:
  ../../reports/layer_probe/layer_probe_results.json
  ../../reports/layer_probe/fig_layer_probe.pdf
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = '../data'
ANNOT_FILE = '../data/raw_data/data/annotations/human_annotations_unified_bp.txt'
ID_DIR     = '../data/raw_data/data/id_lists'
OUT_DIR    = '../../reports/layer_probe'
N_LAYERS   = 30

# ─── GO terms for Eval B (5 representative terms) ─────────────────────────────
PROBE_GO = {
    'GO:0006941': 'Muscle contraction',
    'GO:0006096': 'Glycolysis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0006914': 'Autophagy',
    'GO:0007519': 'Skeletal muscle dev',
}

# ─── Case study isoform pairs (indices in my_isoform_list_fixed.npy) ──────────
CASE_PAIRS = {
    'DMD\n(Dp427m vs minor)': (5942, 10833),    # Dp427m vs alt
    'PINK1\n(canonical vs alt)': (8732, 18787),  # canonical vs alt
    'PTPRF\n(CT-form vs alt)': (12589, 14653),   # canonical vs shorter
}


# ─── Load metadata ─────────────────────────────────────────────────────────────
def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def load_symbol_map():
    m = {}
    with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                m[p[0]] = p[4]
    return m


def load_positive_set(go_term):
    pos = set()
    with open(ANNOT_FILE) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    return pos


# ─── Eval A: Isoform Separation ───────────────────────────────────────────────
def cosine_sim_matrix(X):
    """Pairwise cosine similarity (N×N)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    return Xn @ Xn.T


def eval_separation(emb, gene_list):
    """
    Returns:
      intra_sim  : mean within-gene pairwise cosine similarity (multi-iso genes only)
      inter_sim  : mean across-gene cosine similarity (random 5k pairs)
    """
    gene2idxs = defaultdict(list)
    for i, g in enumerate(gene_list):
        gene2idxs[g].append(i)

    # intra-gene
    intra_sims = []
    for g, idxs in gene2idxs.items():
        if len(idxs) < 2:
            continue
        X_g = emb[idxs]
        norms = np.linalg.norm(X_g, axis=1, keepdims=True) + 1e-12
        Xn = X_g / norms
        S = Xn @ Xn.T
        # upper triangle only
        n = len(idxs)
        vals = [S[i, j] for i in range(n) for j in range(i+1, n)]
        intra_sims.extend(vals)

    intra_sim = float(np.mean(intra_sims)) if intra_sims else 0.0

    # inter-gene (random 5000 pairs)
    rng = np.random.default_rng(42)
    n_total = len(gene_list)
    idxA = rng.integers(0, n_total, 5000)
    idxB = rng.integers(0, n_total, 5000)
    Xn_all = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    inter_sim = float(np.mean(np.sum(Xn_all[idxA] * Xn_all[idxB], axis=1)))

    return intra_sim, inter_sim


def eval_sep_cosine(emb, gene_list, sym_list, pos_set):
    """
    sep_cosine for one GO term at one layer:
    = cosine_dist(pos_centroid, neg_centroid) / mean_intra_pos_cosine_dist

    Returns float or NaN if no positive genes.
    """
    pos_idxs = [i for i, s in enumerate(sym_list) if s in pos_set]
    neg_idxs = [i for i, s in enumerate(sym_list) if s not in pos_set]

    if len(pos_idxs) < 2 or len(neg_idxs) < 2:
        return float('nan')

    Xn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    pos_centroid = Xn[pos_idxs].mean(axis=0)
    neg_centroid = Xn[neg_idxs].mean(axis=0)

    # normalise centroids
    pc = pos_centroid / (np.linalg.norm(pos_centroid) + 1e-12)
    nc = neg_centroid / (np.linalg.norm(neg_centroid) + 1e-12)

    inter_cos_dist = 1.0 - float(pc @ nc)  # 0..2

    # mean pairwise cosine dist within positives (subsample if large)
    rng = np.random.default_rng(0)
    if len(pos_idxs) > 500:
        pos_idxs_sub = rng.choice(pos_idxs, 500, replace=False).tolist()
    else:
        pos_idxs_sub = pos_idxs
    Xp = Xn[pos_idxs_sub]
    S = Xp @ Xp.T
    n = len(pos_idxs_sub)
    intra_vals = [1.0 - S[i, j] for i in range(n) for j in range(i+1, n)]
    intra_cos_dist = float(np.mean(intra_vals)) if intra_vals else 1e-6

    return inter_cos_dist / (intra_cos_dist + 1e-8)


# ─── Eval B: Linear Probe AUPRC (gene-stratified CV) ─────────────────────────
def eval_lr_auprc(emb, sym_list, gene_list, pos_set, n_folds=5, seed=42):
    """
    Gene-stratified K-fold LR probe.
    Returns AUPRC averaged across folds.
    """
    y = np.array([1 if s in pos_set else 0 for s in sym_list], dtype=np.float32)

    if y.sum() < 5:
        return float('nan')

    # Unique gene groups for stratification
    unique_genes = list(dict.fromkeys(gene_list))  # preserve order
    gene2group = {g: i for i, g in enumerate(unique_genes)}
    groups = np.array([gene2group[g] for g in gene_list])

    # y_gene for stratification (gene-level label, use first isoform)
    y_gene = np.array([y[np.where(groups == gi)[0][0]] for gi in range(len(unique_genes))])

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_auprcs = []
    scaler = StandardScaler()

    for train_idx, val_idx in skf.split(emb, y, groups):
        X_tr = scaler.fit_transform(emb[train_idx])
        X_va = scaler.transform(emb[val_idx])
        y_tr = y[train_idx]
        y_va = y[val_idx]

        if y_va.sum() == 0 or y_tr.sum() == 0:
            continue

        lr = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs',
                                class_weight='balanced', random_state=seed)
        try:
            lr.fit(X_tr, y_tr)
            probs = lr.predict_proba(X_va)[:, 1]
            fold_auprcs.append(average_precision_score(y_va, probs))
        except Exception:
            pass

    return float(np.mean(fold_auprcs)) if fold_auprcs else float('nan')


# ─── Eval C: Case Study Distances ─────────────────────────────────────────────
def cosine_distance(a, b):
    na = a / (np.linalg.norm(a) + 1e-12)
    nb = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - na @ nb)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_total = time.time()

    print("=" * 70)
    print("  ESM-2 Layer-wise Probing Analysis")
    print("=" * 70)

    # Load metadata
    iso_list  = load_ids('my_isoform_list_fixed.npy')
    gene_list = load_ids('my_gene_list_fixed.npy')
    sym_map   = load_symbol_map()
    sym_list  = [sym_map.get(g.split('.')[0], g.split('.')[0]) for g in gene_list]

    N = len(iso_list)
    print(f"  Isoforms: {N}  Genes (unique): {len(set(gene_list))}")

    # Load positive sets for all probe GO terms
    pos_sets = {}
    for go_term, go_name in PROBE_GO.items():
        pos_sets[go_term] = load_positive_set(go_term)
        n_pos = sum(1 for s in sym_list if s in pos_sets[go_term])
        print(f"  {go_name:25s} {go_term}: {n_pos} positive isoforms")

    # ── Per-layer loop ──────────────────────────────────────────────────────────
    results = {
        'layers': list(range(1, N_LAYERS + 1)),
        'intra_sim': [],
        'inter_sim': [],
        'sep_cosine': {k: [] for k in PROBE_GO},
        'lr_auprc':   {k: [] for k in PROBE_GO},
        'case_dist':  {name: [] for name in CASE_PAIRS},
    }

    for layer in range(1, N_LAYERS + 1):
        layer_path = os.path.join(DATA_DIR, f'esm2_layer_{layer:02d}_t30_150M.npy')
        if not os.path.exists(layer_path):
            print(f"  [SKIP] Layer {layer:02d}: file not found ({layer_path})")
            for key in results:
                if isinstance(results[key], list):
                    results[key].append(None)
                elif isinstance(results[key], dict):
                    for k in results[key]:
                        results[key][k].append(None)
            continue

        t0 = time.time()
        emb = np.load(layer_path)  # (36748, 640)

        # Eval A: separation
        intra_sim, inter_sim = eval_separation(emb, gene_list)
        results['intra_sim'].append(intra_sim)
        results['inter_sim'].append(inter_sim)

        # sep_cosine per GO term
        sep_str = []
        for go_term in PROBE_GO:
            sc = eval_sep_cosine(emb, gene_list, sym_list, pos_sets[go_term])
            results['sep_cosine'][go_term].append(sc)
            sep_str.append(f'{sc:.3f}' if not np.isnan(sc) else 'NaN')

        # Eval B: LR probe AUPRC
        lr_str = []
        for go_term in PROBE_GO:
            auprc = eval_lr_auprc(emb, sym_list, gene_list, pos_sets[go_term])
            results['lr_auprc'][go_term].append(auprc)
            lr_str.append(f'{auprc:.3f}' if not np.isnan(auprc) else 'NaN')

        # Eval C: case study distances
        for pair_name, (idx_a, idx_b) in CASE_PAIRS.items():
            d = cosine_distance(emb[idx_a], emb[idx_b])
            results['case_dist'][pair_name].append(d)

        elapsed = time.time() - t0
        print(f"  Layer {layer:02d}  intra={intra_sim:.4f}  inter={inter_sim:.4f}  "
              f"sep=[{','.join(sep_str)}]  auprc=[{','.join(lr_str)}]  ({elapsed:.1f}s)")

    # Save JSON
    json_path = os.path.join(OUT_DIR, 'layer_probe_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # ── Generate Figure ─────────────────────────────────────────────────────────
    layers = list(range(1, N_LAYERS + 1))
    valid  = [l for l in layers if results['intra_sim'][l-1] is not None]

    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    cmap_go = plt.cm.tab10

    # Panel A: Intra-gene vs Inter-gene cosine similarity
    ax_a = fig.add_subplot(gs[0, 0])
    intra_v = [results['intra_sim'][l-1] for l in valid]
    inter_v = [results['inter_sim'][l-1] for l in valid]
    ax_a.plot(valid, intra_v, 'o-', color='#D62728', linewidth=2, markersize=5,
              label='Within-gene (intra)')
    ax_a.plot(valid, inter_v, 's--', color='#1F77B4', linewidth=2, markersize=5,
              label='Across-gene (inter)')
    ax_a.set_xlabel('ESM-2 layer', fontsize=11)
    ax_a.set_ylabel('Mean cosine similarity', fontsize=11)
    ax_a.set_title('A  Isoform geometric separation', fontsize=12, fontweight='bold')
    ax_a.legend(fontsize=9)
    ax_a.set_xlim(0.5, N_LAYERS + 0.5)
    ax_a.grid(True, alpha=0.3)
    ax_a.axvline(30, color='gray', linestyle=':', alpha=0.6, label='Current (L30)')

    # Panel B: sep_cosine per GO term
    ax_b = fig.add_subplot(gs[0, 1])
    for ci, (go_term, go_name) in enumerate(PROBE_GO.items()):
        vals = [results['sep_cosine'][go_term][l-1] for l in valid]
        ax_b.plot(valid, vals, 'o-', color=cmap_go(ci), linewidth=1.8,
                  markersize=4, label=go_name[:22])
    ax_b.set_xlabel('ESM-2 layer', fontsize=11)
    ax_b.set_ylabel('sep_cosine', fontsize=11)
    ax_b.set_title('B  GO-term separability (sep_cosine)', fontsize=12, fontweight='bold')
    ax_b.legend(fontsize=8, loc='upper left')
    ax_b.set_xlim(0.5, N_LAYERS + 0.5)
    ax_b.grid(True, alpha=0.3)

    # Panel C: LR probe AUPRC heatmap (layer × GO)
    ax_c = fig.add_subplot(gs[1, 0])
    go_keys  = list(PROBE_GO.keys())
    go_names = [PROBE_GO[k] for k in go_keys]
    mat = np.full((len(go_keys), N_LAYERS), np.nan)
    for gi, go_term in enumerate(go_keys):
        for li, l in enumerate(range(1, N_LAYERS + 1)):
            v = results['lr_auprc'][go_term][li]
            if v is not None and not np.isnan(v):
                mat[gi, li] = v
    im = ax_c.imshow(mat, aspect='auto', cmap='YlOrRd',
                     extent=[0.5, N_LAYERS + 0.5, -0.5, len(go_keys) - 0.5],
                     origin='upper', vmin=np.nanmin(mat) - 0.01,
                     vmax=np.nanmax(mat) + 0.01)
    ax_c.set_yticks(range(len(go_keys)))
    ax_c.set_yticklabels(go_names, fontsize=9)
    ax_c.set_xlabel('ESM-2 layer', fontsize=11)
    ax_c.set_title('C  Linear probe AUPRC (gene-stratified 5-fold CV)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax_c, fraction=0.03, pad=0.02, label='AUPRC')
    ax_c.axvline(30, color='white', linestyle=':', linewidth=1.5, alpha=0.8)

    # Panel D: Case study cosine distances
    ax_d = fig.add_subplot(gs[1, 1])
    pair_colors = ['#9467BD', '#E377C2', '#8C564B']
    for ci, (pair_name, (idx_a, idx_b)) in enumerate(CASE_PAIRS.items()):
        vals = results['case_dist'][pair_name]
        ax_d.plot(valid, [vals[l-1] for l in valid],
                  'o-', color=pair_colors[ci], linewidth=2, markersize=5,
                  label=pair_name.replace('\n', ' '))
    ax_d.set_xlabel('ESM-2 layer', fontsize=11)
    ax_d.set_ylabel('Cosine distance', fontsize=11)
    ax_d.set_title('D  Within-gene isoform pair distance', fontsize=12, fontweight='bold')
    ax_d.legend(fontsize=9)
    ax_d.set_xlim(0.5, N_LAYERS + 0.5)
    ax_d.grid(True, alpha=0.3)
    ax_d.axvline(30, color='gray', linestyle=':', alpha=0.6)

    # LR AUPRC curves overlay on Panel D (secondary y-axis style as inset)
    # (optional — skip for clarity)

    fig.suptitle('ESM-2 layer-wise probing: isoform separation and function prediction\n'
                 '(ESM-2 t30_150M_UR50D, 36,748 skeletal muscle isoforms)',
                 fontsize=11, y=0.98)

    pdf_path = os.path.join(OUT_DIR, 'fig_layer_probe.pdf')
    png_path = os.path.join(OUT_DIR, 'fig_layer_probe.png')
    fig.savefig(pdf_path, bbox_inches='tight', dpi=150)
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")

    # ── Summary table ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  SUMMARY — Layer 30 vs best layer")
    print("="*70)
    print(f"  {'Metric':35s}  L30    Best_L   Best_val")
    print(f"  {'-'*60}")

    def best_nonnan(vals):
        valid_pairs = [(l, v) for l, v in zip(layers, vals)
                       if v is not None and not np.isnan(v)]
        if not valid_pairs:
            return None, None
        return max(valid_pairs, key=lambda x: x[1])

    l30 = 29  # 0-indexed
    print(f"  {'intra_sim':35s}  {results['intra_sim'][l30]:.4f}  "
          f"L{best_nonnan(results['intra_sim'])[0]}    "
          f"{best_nonnan(results['intra_sim'])[1]:.4f}")

    for go_term, go_name in PROBE_GO.items():
        v30 = results['lr_auprc'][go_term][l30]
        bl, bv = best_nonnan(results['lr_auprc'][go_term])
        print(f"  {('LR AUPRC ' + go_name)[:35]:35s}  {v30:.4f}  L{bl:2d}    {bv:.4f}")

    total = time.time() - t_total
    print(f"\n  Total time: {total:.0f}s ({total/60:.1f} min)")
    print(f"  Results: {OUT_DIR}/")


if __name__ == '__main__':
    main()
