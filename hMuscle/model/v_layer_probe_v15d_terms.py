#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v_layer_probe_v15d_terms.py
----------------------------
v15d 평가에 사용되는 18개 GO term 중 아직 probe하지 않은 12개에 대해
LR + MLP probe를 수행하여 NAS / LDS / Shift 분류 체계를 검증.

이미 probe된 6개 (별도 실험에서):
  GO:0006941 Muscle contraction     (NAS=1.47, Type ID)
  GO:0006914 Autophagy              (NAS=2.05, Type CM)
  GO:0007519 Skeletal muscle dev    (NAS=1.00, Type WK)
  GO:0007005 Mitochondrion org      (NAS=1.75, Type MC)
  GO:0006096 Glycolysis             (NAS=0.93, Type SF)
  GO:0000226 MT cytoskeleton org    (NAS=1.93, Type MC)

본 스크립트에서 probe할 12개:
  GO:0007204 Ca2+ signaling
  GO:0045214 Sarcomere organization
  GO:0043161 Proteasome-UPS
  GO:0042692 Muscle cell diff
  GO:0055074 Ca2+ homeostasis
  GO:0007517 Muscle organ dev
  GO:0032006 TOR signaling
  GO:0030048 Actin-based movement
  GO:0007268 Synaptic transmission
  GO:0007018 MT-based movement
  GO:0031175 Neuron proj development
  GO:0030182 Neuron diff

실행:
  cd hMuscle/model/
  conda run -n isoform_env python3 -u v_layer_probe_v15d_terms.py [--gpu 1]

출력:
  ../../reports/layer_probe/layer_probe_v15d_terms_results.json
"""

import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR   = '../data'
ANNOT_FILE = '../data/raw_data/data/annotations/human_annotations_unified_bp.txt'
ID_DIR     = '../data/raw_data/data/id_lists'
OUT_DIR    = '../../reports/layer_probe'
N_LAYERS   = 30

# ── v15d 미검증 12개 GO term ──────────────────────────────────────────────────
PROBE_GO_V15D = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0045214': 'Sarcomere organization',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0030048': 'Actin-based movement',
    'GO:0007268': 'Synaptic transmission',
    'GO:0007018': 'MT-based movement',
    'GO:0031175': 'Neuron proj development',
    'GO:0030182': 'Neuron diff',
}

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


def eval_lr_auprc(emb, sym_list, gene_list, pos_set, n_folds=5, seed=42):
    y = np.array([1 if s in pos_set else 0 for s in sym_list], dtype=np.float32)
    if y.sum() < 5:
        return float('nan')
    gene2group = {g: i for i, g in enumerate(dict.fromkeys(gene_list))}
    groups = np.array([gene2group[g] for g in gene_list])
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scaler = StandardScaler()
    fold_auprcs = []
    import warnings
    for train_idx, val_idx in skf.split(emb, y, groups):
        X_tr = scaler.fit_transform(emb[train_idx])
        X_va = scaler.transform(emb[val_idx])
        y_tr, y_va = y[train_idx], y[val_idx]
        if y_va.sum() == 0 or y_tr.sum() == 0:
            continue
        lr = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs',
                                class_weight='balanced', random_state=seed)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lr.fit(X_tr, y_tr)
            probs = lr.predict_proba(X_va)[:, 1]
            fold_auprcs.append(average_precision_score(y_va, probs))
        except Exception:
            pass
    return float(np.mean(fold_auprcs)) if fold_auprcs else float('nan')


def eval_mlp_auprc(emb_np, sym_list, gene_list, pos_set,
                   device, n_folds=5, n_epochs=100, batch_size=4096,
                   lr_rate=3e-3, hidden=64, patience=12, seed=42):
    y = np.array([1.0 if s in pos_set else 0.0 for s in sym_list], dtype=np.float32)
    if y.sum() < 5:
        return float('nan')
    gene2group = {g: i for i, g in enumerate(dict.fromkeys(gene_list))}
    groups = np.array([gene2group[g] for g in gene_list])
    scaler = StandardScaler()
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_auprcs = []
    import warnings
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(emb_np, y, groups)):
        X_tr = scaler.fit_transform(emb_np[train_idx])
        X_va = scaler.transform(emb_np[val_idx])
        y_tr, y_va = y[train_idx], y[val_idx]
        if y_va.sum() == 0 or y_tr.sum() == 0:
            continue
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        pos_w = torch.tensor([n_neg / (n_pos + 1e-8)], device=device)
        X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
        model = MLPProbe(input_dim=X_tr.shape[1], hidden=hidden).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-4)
        n_tr = len(y_tr)
        best_loss = float('inf')
        no_improve = 0
        torch.manual_seed(seed + fold_i)
        for epoch in range(n_epochs):
            model.train()
            perm = torch.randperm(n_tr, device=device)
            epoch_loss = 0.0
            for i in range(0, n_tr, batch_size):
                bi = perm[i:i+batch_size]
                logits = model(X_t[bi])
                loss = criterion(logits, y_t[bi])
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss; no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
        model.eval()
        with torch.no_grad():
            X_v = torch.tensor(X_va, dtype=torch.float32, device=device)
            logits_v = model(X_v).cpu().numpy()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                probs_v = 1 / (1 + np.exp(-np.clip(logits_v, -500, 500)))
        try:
            fold_auprcs.append(average_precision_score(y_va, probs_v))
        except Exception:
            pass
    return float(np.mean(fold_auprcs)) if fold_auprcs else float('nan')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    iso_list  = load_ids('my_isoform_list_fixed.npy')
    gene_list = load_ids('my_gene_list_fixed.npy')
    sym_map   = load_symbol_map()
    sym_list  = [sym_map.get(g.split('.')[0], g.split('.')[0]) for g in gene_list]
    print(f"Isoforms: {len(iso_list)}, Genes: {len(set(gene_list))}", flush=True)

    pos_sets = {}
    print("\nGO term coverage:", flush=True)
    for go, name in PROBE_GO_V15D.items():
        pos_sets[go] = load_positive_set(go)
        n_pos = sum(1 for s in sym_list if s in pos_sets[go])
        pct = n_pos / len(sym_list) * 100
        print(f"  {go}: {name:35s} n_pos={n_pos:5d} ({pct:.2f}%)", flush=True)

    print(f"\nRunning LR+MLP probe: {N_LAYERS} layers × {len(PROBE_GO_V15D)} GO terms × 5 folds", flush=True)
    print("="*78, flush=True)

    lr_auprc  = {go: [] for go in PROBE_GO_V15D}
    mlp_auprc = {go: [] for go in PROBE_GO_V15D}
    t_total = time.time()

    for layer in range(1, N_LAYERS + 1):
        layer_path = os.path.join(DATA_DIR, f'esm2_layer_{layer:02d}_t30_150M.npy')
        if not os.path.exists(layer_path):
            for go in PROBE_GO_V15D:
                lr_auprc[go].append(None)
                mlp_auprc[go].append(None)
            print(f"  Layer {layer:02d}: SKIP (missing)", flush=True)
            continue

        emb = np.load(layer_path)
        t0 = time.time()
        lr_strs, mlp_strs = [], []

        for go in PROBE_GO_V15D:
            lr_a  = eval_lr_auprc(emb, sym_list, gene_list, pos_sets[go])
            mlp_a = eval_mlp_auprc(emb, sym_list, gene_list, pos_sets[go], device)
            lr_auprc[go].append(lr_a)
            mlp_auprc[go].append(mlp_a)
            lr_strs.append(f'{lr_a:.4f}' if lr_a == lr_a else 'nan')
            mlp_strs.append(f'{mlp_a:.4f}' if mlp_a == mlp_a else 'nan')

        elapsed = time.time() - t0
        print(f"  Layer {layer:02d} ({elapsed:.1f}s)  "
              f"LR=[{','.join(lr_strs)}]  MLP=[{','.join(mlp_strs)}]", flush=True)
        print(f"Layers completed: {layer}/{N_LAYERS}", flush=True)

    out = {
        'go_terms':  PROBE_GO_V15D,
        'layers':    list(range(1, N_LAYERS + 1)),
        'lr_auprc':  lr_auprc,
        'mlp_auprc': mlp_auprc,
    }
    out_path = os.path.join(OUT_DIR, 'layer_probe_v15d_terms_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)

    # ── 요약 테이블 ──────────────────────────────────────────────────────────────
    print("\n" + "="*90, flush=True)
    print("  GO term                          NAS       LDS    Shift  Type(pred)  LR_best  MLP_best", flush=True)
    print("  " + "-"*86, flush=True)

    # NAS 기준 타입 분류:
    # SF:  NAS < 1.1                   (비선형 이득 없음, 이미 L30에서 포화)
    # ID:  1.1 <= NAS < 1.5            (약한 비선형 이득, 상호작용 도메인)
    # MC:  1.5 <= NAS < 2.0            (중간 비선형, 다중 조건)
    # CM:  NAS >= 2.0                  (강한 비선형, 컨텍스트 의존적)
    # WK:  MLP_best < 0.10             (신호 약함)
    NAS_SF, NAS_ID, NAS_MC = 1.1, 1.5, 2.0

    def classify(nas, mlp_best):
        if mlp_best < 0.10: return 'WK'
        if nas < NAS_SF: return 'SF'
        if nas < NAS_ID: return 'ID'
        if nas < NAS_MC: return 'MC'
        return 'CM'

    summary_rows = []
    for go, name in PROBE_GO_V15D.items():
        lr_arr  = [v for v in lr_auprc[go]  if v is not None and v == v]
        mlp_arr = [v for v in mlp_auprc[go] if v is not None and v == v]
        if not lr_arr or not mlp_arr:
            continue
        lr_best_val    = max(lr_arr)
        lr_best_layer  = lr_auprc[go].index(lr_best_val) + 1
        mlp_best_val   = max(mlp_arr)
        mlp_best_layer = mlp_auprc[go].index(mlp_best_val) + 1
        lr_l30  = lr_auprc[go][29]  if lr_auprc[go][29]  == lr_auprc[go][29]  else float('nan')
        mlp_l30 = mlp_auprc[go][29] if mlp_auprc[go][29] == mlp_auprc[go][29] else float('nan')

        nas    = mlp_best_val / lr_best_val if lr_best_val > 0 else float('nan')
        lds    = lr_best_layer / N_LAYERS
        shift  = mlp_best_layer - lr_best_layer
        t_type = classify(nas, mlp_best_val)
        summary_rows.append((go, name, nas, lds, shift, t_type, lr_best_val, lr_best_layer,
                              mlp_best_val, mlp_best_layer))

        print(f"  {name:32s} NAS={nas:.3f}  LDS={lds:.3f}  Shift={shift:+3d}  "
              f"[{t_type}]  LR={lr_best_val:.4f}(L{lr_best_layer:02d})  "
              f"MLP={mlp_best_val:.4f}(L{mlp_best_layer:02d})", flush=True)

    # 레이어 선택 권고 (NAS/LDS 기반)
    print("\n" + "="*90, flush=True)
    print("  Layer selection recommendation (based on NAS/LDS/Shift):", flush=True)
    print("  SF (NAS<1.1)  → L27-L30 only", flush=True)
    print("  ID (1.1-1.5)  → L7 OR L18 dominant, fusion helps mildly", flush=True)
    print("  MC (1.5-2.0)  → L7+L18+L27 fusion: captures multi-scale", flush=True)
    print("  CM (NAS>=2.0) → L18+L27 dominant (context-dependent)", flush=True)
    print("  WK            → any layer; signal too weak to distinguish", flush=True)

    total_time = time.time() - t_total
    print(f"\n  Total: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
