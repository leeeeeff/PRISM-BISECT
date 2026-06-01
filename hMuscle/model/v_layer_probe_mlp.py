#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v_layer_probe_mlp.py
--------------------
LR probe 결과(layer_probe_results.json)를 재사용하고,
동일한 gene-stratified 5-fold CV 세팅에서 MLP probe를 추가 실행.

가설: LR은 중간 레이어에서 피크 (선형 신호 최대).
     MLP는 L30에 가까울수록 높아야 함 (비선형 재인코딩된 신호 복원).

실행:
  cd hMuscle/model/
  conda run -n isoform_env python v_layer_probe_mlp.py [--gpu 1]

출력:
  ../../reports/layer_probe/layer_probe_mlp_results.json
  ../../reports/layer_probe/fig_layer_probe_lr_vs_mlp.pdf
"""

import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

# ── 경로 ─────────────────────────────────────────────────────────────────────
DATA_DIR   = '../data'
ANNOT_FILE = '../data/raw_data/data/annotations/human_annotations_unified_bp.txt'
ID_DIR     = '../data/raw_data/data/id_lists'
OUT_DIR    = '../../reports/layer_probe'
PREV_JSON  = os.path.join(OUT_DIR, 'layer_probe_results.json')
N_LAYERS   = 30

PROBE_GO = {
    'GO:0006941': 'Muscle contraction',
    'GO:0006096': 'Glycolysis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0006914': 'Autophagy',
    'GO:0007519': 'Skeletal muscle dev',
}

# ── MLP 구조 ──────────────────────────────────────────────────────────────────
class MLPProbe(nn.Module):
    def __init__(self, input_dim=640, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ── 메타데이터 로딩 ───────────────────────────────────────────────────────────
def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

def load_symbol_map():
    m = {}
    with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5: m[p[0]] = p[4]
    return m

def load_positive_set(go_term):
    pos = set()
    with open(ANNOT_FILE) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    return pos

# ── GPU MLP probe ─────────────────────────────────────────────────────────────
def eval_mlp_auprc(emb_np, sym_list, gene_list, pos_set,
                   device, n_folds=5, n_epochs=100, batch_size=4096,
                   lr=3e-3, hidden=64, patience=12, seed=42):
    """
    Returns fold-mean AUPRC (float).
    """
    y = np.array([1.0 if s in pos_set else 0.0 for s in sym_list], dtype=np.float32)
    if y.sum() < 5:
        return float('nan')

    gene2group = {g: i for i, g in enumerate(dict.fromkeys(gene_list))}
    groups = np.array([gene2group[g] for g in gene_list])

    scaler = StandardScaler()
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_auprcs = []
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(emb_np, y, groups)):
        X_tr = scaler.fit_transform(emb_np[train_idx])
        X_va = scaler.transform(emb_np[val_idx])
        y_tr = y[train_idx]
        y_va = y[val_idx]

        if y_va.sum() == 0 or y_tr.sum() == 0:
            continue

        # pos_weight for class imbalance
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        pos_w = torch.tensor([n_neg / (n_pos + 1e-8)], device=device)

        X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

        model = MLPProbe(input_dim=X_tr.shape[1], hidden=hidden).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        n_tr = len(y_tr)
        best_loss = float('inf')
        no_improve = 0

        torch.manual_seed(seed + fold_i)
        for epoch in range(n_epochs):
            model.train()
            perm = torch.randperm(n_tr, device=device)
            epoch_loss = 0.0
            for i in range(0, n_tr, batch_size):
                bi = perm[i:i + batch_size]
                logits = model(X_t[bi])
                loss = criterion(logits, y_t[bi])
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()
            # early stopping on train loss
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        model.eval()
        with torch.no_grad():
            X_v = torch.tensor(X_va, dtype=torch.float32, device=device)
            logits_v = model(X_v).cpu().numpy()
            probs_v = 1 / (1 + np.exp(-logits_v))  # sigmoid

        try:
            fold_auprcs.append(average_precision_score(y_va, probs_v))
        except Exception:
            pass

    return float(np.mean(fold_auprcs)) if fold_auprcs else float('nan')

# ── 메인 ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=1)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 이전 LR 결과 로드
    with open(PREV_JSON) as f:
        lr_data = json.load(f)
    print(f"Loaded LR results from {PREV_JSON}")

    # 메타데이터
    iso_list  = load_ids('my_isoform_list_fixed.npy')
    gene_list = load_ids('my_gene_list_fixed.npy')
    sym_map   = load_symbol_map()
    sym_list  = [sym_map.get(g.split('.')[0], g.split('.')[0]) for g in gene_list]

    pos_sets = {go: load_positive_set(go) for go in PROBE_GO}
    for go, name in PROBE_GO.items():
        n_pos = sum(1 for s in sym_list if s in pos_sets[go])
        print(f"  {name}: {n_pos} positive isoforms")

    print(f"\nRunning MLP probe: 30 layers × {len(PROBE_GO)} GO terms × 5 folds")
    print("="*70)

    mlp_auprc = {go: [] for go in PROBE_GO}
    t_total = time.time()

    for layer in range(1, N_LAYERS + 1):
        layer_path = os.path.join(DATA_DIR, f'esm2_layer_{layer:02d}_t30_150M.npy')
        if not os.path.exists(layer_path):
            for go in PROBE_GO:
                mlp_auprc[go].append(None)
            print(f"  Layer {layer:02d}: SKIP (file not found)")
            continue

        emb = np.load(layer_path)
        t0 = time.time()

        go_strs = []
        for go in PROBE_GO:
            a = eval_mlp_auprc(emb, sym_list, gene_list, pos_sets[go], device)
            mlp_auprc[go].append(a)
            go_strs.append(f'{a:.4f}' if a == a else 'NaN')

        elapsed = time.time() - t0
        lr_strs = [f'{lr_data["lr_auprc"][go][layer-1]:.4f}' for go in PROBE_GO]
        print(f"  Layer {layer:02d} ({elapsed:.1f}s)  "
              f"LR=[{','.join(lr_strs)}]  MLP=[{','.join(go_strs)}]")

    # 저장
    out = {
        'layers': list(range(1, N_LAYERS + 1)),
        'lr_auprc': lr_data['lr_auprc'],
        'mlp_auprc': mlp_auprc,
        'probe_go': PROBE_GO,
    }
    json_path = os.path.join(OUT_DIR, 'layer_probe_mlp_results.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── 비교 Figure ─────────────────────────────────────────────────────────
    go_keys  = list(PROBE_GO.keys())
    go_names = list(PROBE_GO.values())
    layers   = list(range(1, N_LAYERS + 1))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    cmap = plt.cm.tab10

    for pi, (go, go_name) in enumerate(PROBE_GO.items()):
        ax = axes[pi]
        lr_v  = lr_data['lr_auprc'][go]
        mlp_v = mlp_auprc[go]

        valid = [l for l in layers if lr_v[l-1] is not None and mlp_v[l-1] is not None]
        lr_vals  = [lr_v[l-1]  for l in valid]
        mlp_vals = [mlp_v[l-1] for l in valid]

        ax.plot(valid, lr_vals,  's--', color='#1F77B4', linewidth=2, markersize=5,
                label='LR probe (linear)', alpha=0.9)
        ax.plot(valid, mlp_vals, 'o-',  color='#D62728', linewidth=2, markersize=5,
                label='MLP probe (non-linear)', alpha=0.9)

        # 피크 표시
        lr_best  = valid[int(np.argmax(lr_vals))]
        mlp_best = valid[int(np.argmax(mlp_vals))]
        ax.axvline(lr_best,  color='#1F77B4', linestyle=':', alpha=0.6,
                   label=f'LR peak L{lr_best}')
        ax.axvline(mlp_best, color='#D62728', linestyle=':', alpha=0.6,
                   label=f'MLP peak L{mlp_best}')
        ax.axvline(30, color='gray', linestyle='-', alpha=0.3, linewidth=1)

        ax.fill_between(valid, lr_vals, mlp_vals,
                        where=[m > l for m, l in zip(mlp_vals, lr_vals)],
                        alpha=0.12, color='#D62728', label='MLP > LR')
        ax.fill_between(valid, lr_vals, mlp_vals,
                        where=[l > m for m, l in zip(mlp_vals, lr_vals)],
                        alpha=0.12, color='#1F77B4', label='LR > MLP')

        ax.set_title(f'{go_name}\n({go})', fontsize=11, fontweight='bold')
        ax.set_xlabel('ESM-2 layer', fontsize=10)
        ax.set_ylabel('AUPRC (5-fold CV)', fontsize=10)
        ax.legend(fontsize=7.5, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, N_LAYERS + 0.5)

        lr30  = lr_v[29]
        mlp30 = mlp_v[29]
        ax.text(0.02, 0.97,
                f'L30: LR={lr30:.3f}  MLP={mlp30:.3f}  Δ={mlp30-lr30:+.3f}',
                transform=ax.transAxes, fontsize=8.5, va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 요약 패널
    ax_sum = axes[5]
    go_labels = [n[:18] for n in go_names]
    lr30_vals  = [lr_data['lr_auprc'][go][29] for go in go_keys]
    mlp30_vals = [mlp_auprc[go][29] for go in go_keys]
    lr_best_vals  = [max(v for v in lr_data['lr_auprc'][go] if v is not None) for go in go_keys]
    mlp_best_vals = [max(v for v in mlp_auprc[go] if v is not None) for go in go_keys]

    x = np.arange(len(go_keys))
    w = 0.18
    ax_sum.bar(x - 1.5*w, lr_best_vals,  w, label='LR best layer',  color='#AEC7E8', edgecolor='black', linewidth=0.5)
    ax_sum.bar(x - 0.5*w, lr30_vals,     w, label='LR at L30',      color='#1F77B4', edgecolor='black', linewidth=0.5)
    ax_sum.bar(x + 0.5*w, mlp30_vals,    w, label='MLP at L30',     color='#D62728', edgecolor='black', linewidth=0.5)
    ax_sum.bar(x + 1.5*w, mlp_best_vals, w, label='MLP best layer', color='#FFAAAA', edgecolor='black', linewidth=0.5)
    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels(go_labels, rotation=25, ha='right', fontsize=9)
    ax_sum.set_ylabel('AUPRC', fontsize=10)
    ax_sum.set_title('Summary: LR vs MLP at L30 and best layer', fontsize=11, fontweight='bold')
    ax_sum.legend(fontsize=8)
    ax_sum.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Non-linear re-encoding hypothesis: LR probe vs MLP probe across ESM-2 layers\n'
                 '(gene-stratified 5-fold CV, 36,748 skeletal muscle isoforms)',
                 fontsize=11, y=0.99)

    plt.tight_layout()
    pdf_path = os.path.join(OUT_DIR, 'fig_layer_probe_lr_vs_mlp.pdf')
    png_path = os.path.join(OUT_DIR, 'fig_layer_probe_lr_vs_mlp.png')
    fig.savefig(pdf_path, bbox_inches='tight', dpi=150)
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {pdf_path}")

    # ── 요약 출력 ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"  {'GO term':25s}  LR_best(L)  LR_L30  MLP_L30  Δ(MLP-LR)  MLP_best(L)")
    print("  " + "-"*65)
    for go, go_name in PROBE_GO.items():
        lv  = lr_data['lr_auprc'][go]
        mv  = mlp_auprc[go]
        lr_best_l  = int(np.argmax([v for v in lv if v is not None])) + 1
        mlp_best_l = int(np.argmax([v for v in mv if v is not None])) + 1
        lr_best_v  = max(v for v in lv if v is not None)
        mlp_best_v = max(v for v in mv if v is not None)
        lr30  = lv[29]
        mlp30 = mv[29]
        print(f"  {go_name:25s}  {lr_best_v:.4f}(L{lr_best_l:2d})  "
              f"{lr30:.4f}   {mlp30:.4f}    {mlp30-lr30:+.4f}     "
              f"{mlp_best_v:.4f}(L{mlp_best_l:2d})")

    total = time.time() - t_total
    print(f"\n  Total: {total:.0f}s ({total/60:.1f} min)")

if __name__ == '__main__':
    main()
