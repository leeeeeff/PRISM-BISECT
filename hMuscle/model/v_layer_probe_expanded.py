#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v_layer_probe_expanded.py
--------------------------
GO term 유형 분류 체계 검증을 위한 확장 probe 실험.
10개 새 GO term에 대해 LR + MLP probe를 동시에 실행.

선택 기준: 단순(Translation, Protein folding) → 중간(Phosphorylation, Ubiquitin) →
          복잡(GPCR, MAPK, RNA splicing) → 조절(Transcription, Cell adhesion)
          → 구조적(Microtubule)로 점진적 복잡도 커버.

실행:
  cd hMuscle/model/
  conda run -n isoform_env python v_layer_probe_expanded.py [--gpu 1]

출력:
  ../../reports/layer_probe/layer_probe_expanded_results.json
"""

import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

# ── 경로 ──────────────────────────────────────────────────────────────────────
DATA_DIR   = '../data'
ANNOT_FILE = '../data/raw_data/data/annotations/human_annotations_unified_bp.txt'
ID_DIR     = '../data/raw_data/data/id_lists'
OUT_DIR    = '../../reports/layer_probe'
N_LAYERS   = 30

# ── 확장 GO term 세트 (10개 신규) ──────────────────────────────────────────────
# 생물학적 복잡도 및 메커니즘 다양성 기준 선정:
#
# SF 예상 (Structural Fold, NAS<1.2, depth 상관 양):
#   GO:0006412  Translation            - 리보솜 단백질: 매우 보존된 fold
#   GO:0006457  Protein folding        - chaperone: 잘 정의된 AAA/DnaJ 도메인
#
# ID 예상 (Interaction Domain, NAS 1.2-1.7, Shift<0):
#   GO:0006468  Protein phosphorylation - 키나아제: substrate diversity, kinase fold 보존
#   GO:0000226  Microtubule cyto org   - 튜불린+조절단백질: structural + dynamic
#
# CM 예상 (Context Motif, NAS>2.0, Shift≈0):
#   GO:0006511  Ubiq-dep proteasomal   - UBL domain + substrate 다양성
#   GO:0000165  MAPK cascade           - 키나아제 cascade, 컨텍스트 의존적
#
# MC 예상 (Multi-Condition, NAS>1.7, LDS>0.75):
#   GO:0007186  GPCR signaling         - 7TM 구조 + 다양한 C-terminus
#   GO:0000375  RNA splicing           - 스플라이소좀: RRM/RS 도메인 조합
#
# WK 예상 (Weak Signal, NAS≈1.0):
#   GO:0006355  Regulation of transc   - 전사인자: 극도로 다양한 DBD
#   GO:0007155  Cell adhesion          - cadherin/integrin: 세포 맥락 의존
#
PROBE_GO_NEW = {
    'GO:0006412': 'Translation',
    'GO:0006457': 'Protein folding',
    'GO:0006468': 'Protein phosphorylation',
    'GO:0000226': 'Microtubule cytoskeleton org',
    'GO:0006511': 'Ubiq-dep proteasomal degr',
    'GO:0000165': 'MAPK cascade',
    'GO:0007186': 'GPCR signaling',
    'GO:0000375': 'RNA splicing',
    'GO:0006355': 'Regulation of transcription',
    'GO:0007155': 'Cell adhesion',
}

# ── MLP 구조 (기존과 동일) ────────────────────────────────────────────────────
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


# ── 메타데이터 로딩 ────────────────────────────────────────────────────────────
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


# ── LR probe ──────────────────────────────────────────────────────────────────
def eval_lr_auprc(emb, sym_list, gene_list, pos_set, n_folds=5, seed=42):
    y = np.array([1 if s in pos_set else 0 for s in sym_list], dtype=np.float32)
    if y.sum() < 5:
        return float('nan')

    unique_genes = list(dict.fromkeys(gene_list))
    gene2group = {g: i for i, g in enumerate(unique_genes)}
    groups = np.array([gene2group[g] for g in gene_list])

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scaler = StandardScaler()
    fold_auprcs = []

    for train_idx, val_idx in skf.split(emb, y, groups):
        X_tr = scaler.fit_transform(emb[train_idx])
        X_va = scaler.transform(emb[val_idx])
        y_tr, y_va = y[train_idx], y[val_idx]
        if y_va.sum() == 0 or y_tr.sum() == 0:
            continue
        lr = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs',
                                class_weight='balanced', random_state=seed)
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lr.fit(X_tr, y_tr)
            probs = lr.predict_proba(X_va)[:, 1]
            fold_auprcs.append(average_precision_score(y_va, probs))
        except Exception:
            pass

    return float(np.mean(fold_auprcs)) if fold_auprcs else float('nan')


# ── MLP probe ─────────────────────────────────────────────────────────────────
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
                bi = perm[i:i + batch_size]
                logits = model(X_t[bi])
                loss = criterion(logits, y_t[bi])
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()
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
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                probs_v = 1 / (1 + np.exp(-np.clip(logits_v, -500, 500)))

        try:
            fold_auprcs.append(average_precision_score(y_va, probs_v))
        except Exception:
            pass

    return float(np.mean(fold_auprcs)) if fold_auprcs else float('nan')


# ── 메인 ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 메타데이터 로드
    iso_list  = load_ids('my_isoform_list_fixed.npy')
    gene_list = load_ids('my_gene_list_fixed.npy')
    sym_map   = load_symbol_map()
    sym_list  = [sym_map.get(g.split('.')[0], g.split('.')[0]) for g in gene_list]

    print(f"Isoforms: {len(iso_list)}, Genes: {len(set(gene_list))}", flush=True)

    # Positive sets + 실제 커버리지 확인
    pos_sets = {}
    print("\nGO term coverage in hMuscle dataset:", flush=True)
    for go, name in PROBE_GO_NEW.items():
        pos_sets[go] = load_positive_set(go)
        n_pos = sum(1 for s in sym_list if s in pos_sets[go])
        pct = n_pos / len(sym_list) * 100
        print(f"  {go}: {name:35s} n_pos={n_pos:5d} ({pct:.2f}%)")

    print(f"\nRunning LR+MLP probe: {N_LAYERS} layers × {len(PROBE_GO_NEW)} GO terms × 5 folds", flush=True)
    print("="*78, flush=True)

    lr_auprc  = {go: [] for go in PROBE_GO_NEW}
    mlp_auprc = {go: [] for go in PROBE_GO_NEW}
    t_total = time.time()

    for layer in range(1, N_LAYERS + 1):
        layer_path = os.path.join(DATA_DIR, f'esm2_layer_{layer:02d}_t30_150M.npy')
        if not os.path.exists(layer_path):
            for go in PROBE_GO_NEW:
                lr_auprc[go].append(None)
                mlp_auprc[go].append(None)
            print(f"  Layer {layer:02d}: SKIP")
            continue

        emb = np.load(layer_path)
        t0 = time.time()

        lr_strs, mlp_strs = [], []
        for go in PROBE_GO_NEW:
            lr_a  = eval_lr_auprc(emb, sym_list, gene_list, pos_sets[go])
            mlp_a = eval_mlp_auprc(emb, sym_list, gene_list, pos_sets[go], device)
            lr_auprc[go].append(lr_a)
            mlp_auprc[go].append(mlp_a)
            lr_strs.append(f'{lr_a:.4f}' if lr_a == lr_a else 'nan')
            mlp_strs.append(f'{mlp_a:.4f}' if mlp_a == mlp_a else 'nan')

        elapsed = time.time() - t0
        print(f"  Layer {layer:02d} ({elapsed:.1f}s)  "
              f"LR=[{','.join(lr_strs)}]  MLP=[{','.join(mlp_strs)}]", flush=True)

    # 저장
    out = {
        'go_terms':  PROBE_GO_NEW,
        'layers':    list(range(1, N_LAYERS + 1)),
        'lr_auprc':  lr_auprc,
        'mlp_auprc': mlp_auprc,
    }
    out_path = os.path.join(OUT_DIR, 'layer_probe_expanded_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── 요약 테이블 ────────────────────────────────────────────────────────────
    print("\n" + "="*78)
    print("  GO term                          LR_best(L)  LR_L30  MLP_L30  Δ(MLP-LR)  MLP_best(L)")
    print("  " + "-"*74)
    go_names_list = list(PROBE_GO_NEW.values())
    for go, name in PROBE_GO_NEW.items():
        lr_arr  = [v for v in lr_auprc[go]  if v is not None and v == v]
        mlp_arr = [v for v in mlp_auprc[go] if v is not None and v == v]
        if not lr_arr or not mlp_arr:
            continue
        lr_best_val   = max(lr_arr)
        lr_best_layer = lr_auprc[go].index(lr_best_val) + 1
        mlp_best_val   = max(mlp_arr)
        mlp_best_layer = mlp_auprc[go].index(mlp_best_val) + 1
        lr_l30  = lr_auprc[go][29]  if lr_auprc[go][29]  == lr_auprc[go][29]  else float('nan')
        mlp_l30 = mlp_auprc[go][29] if mlp_auprc[go][29] == mlp_auprc[go][29] else float('nan')
        delta = mlp_l30 - lr_l30
        print(f"  {name:32s} {lr_best_val:.4f}(L{lr_best_layer:02d}) "
              f" {lr_l30:.4f}   {mlp_l30:.4f}   {delta:+.4f}    {mlp_best_val:.4f}(L{mlp_best_layer:02d})")

    total_time = time.time() - t_total
    print(f"\n  Total: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()
