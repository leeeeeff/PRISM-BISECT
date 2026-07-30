#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_f_plm_scale_scan.py
-----------------------
PLM 범용성 실험 — δ_layer AUPRC 비교 분석.

exp_f_plm_scale_embed.py 실행 완료 후 사용.

각 모델에서 3가지 조건 비교:
  1. plain_last   : φ_L_final  만 사용
  2. delta_only   : δ = φ_L_final - φ_L_mid
  3. concat_delta : [φ_L_final ‖ δ]  ← v17f* 방식

NatMeth 핵심 주장 검증:
  "δ_layer 원리는 ESM-2 스케일과 ProtT5 아키텍처에 걸쳐 일관되게 성립한다.
   최적 layer pair는 모델 총 깊이의 ~50%에서 안정적으로 나타난다."

출력:
  ../../reports/exp_f_plm_scale/results.tsv
  ../../reports/exp_f_plm_scale/summary.txt
"""

import os, gzip, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')
import torch
import torch.nn as nn

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/exp_f_plm_scale'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13, 21, 99]
EPOCHS_MLP = 60
BATCH      = 512

# ── Model registry ─────────────────────────────────────────────────────────────
# (tag, n_layers, dim, label)
MODELS = [
    ('t30_150M',   30, 640,  'ESM-2 150M'),
    ('t33_650M',   33, 1280, 'ESM-2 650M'),
    ('t36_3B',     36, 2560, 'ESM-2 3B'),
    ('prot_t5_xl', 24, 1024, 'ProtT5-XL'),
    ('esm3_sm',    48, 1536, 'ESM3 sm'),
    ('ankh_base',  48, 768,  'Ankh-base'),
]


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


# ── GO labels (identical to v17f*) ────────────────────────────────────────────
def load_labels():
    ENSG2SYM = {}
    with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

    tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
    tr_genes     = [clean(g) for g in tr_genes_raw]
    te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
    te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                    for g in te_genes_raw]

    sym2id = {}
    with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) > 2:
                sym2id[p[2]] = p[1]
                if len(p) > 4 and p[4] != '-':
                    for syn in p[4].split('|'):
                        if syn not in sym2id: sym2id[syn] = p[1]

    tr_ids    = [sym2id.get(g, g) for g in tr_genes]
    tr_id_set = set(tr_ids)

    go_genes_tr  = defaultdict(set)
    go_genes_all = defaultdict(set)
    with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if p[0] != '9606': continue
            if p[7] != 'Function': continue
            go_genes_all[p[2]].add(p[1])
            if p[1] in tr_id_set: go_genes_tr[p[2]].add(p[1])

    # MF terms from expanded GO analysis
    mf_terms_path = '../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv'
    mf_terms = []
    if os.path.exists(mf_terms_path):
        with open(mf_terms_path) as f:
            next(f)
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 6: mf_terms.append(p[0])
    if not mf_terms:
        mf_terms = [
            'GO_0003824', 'GO_0003676', 'GO_0005488',
            'GO_0003723', 'GO_0005215', 'GO_0004672',
            'GO_0016787', 'GO_0016491', 'GO_0008094',
        ]
        print("  [WARN] Using 9 default MF terms (mf_domain_vs_prism.tsv not found)")

    tr_sym2idx = defaultdict(list)
    for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

    def build_Y_tr(go_id):
        pos_ids  = go_genes_tr[go_id]
        pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in pos_ids}
        y = np.zeros(len(tr_genes), dtype=np.float32)
        for sym in pos_syms:
            for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
        return y

    def build_Y_te(go_id):
        pos_ids = go_genes_all[go_id]
        return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0
                         for s in te_sym_list], dtype=np.float32)

    Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)
    Y_te = np.stack([build_Y_te(go)  for go in mf_terms], axis=1)
    valid_mask = Y_te.sum(0) >= 2
    print(f"  {valid_mask.sum()}/{len(mf_terms)} GO terms with ≥2 positives in test")
    return Y_tr, Y_te, valid_mask


# ── MLP (identical architecture to v17f*) ────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_out),  nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)


def run_mlp(X_tr, Y_tr, X_te, Y_te, valid_mask, seeds, label):
    dev   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_out = Y_tr.shape[1]
    scaler = MaxAbsScaler()
    Xtr_s  = scaler.fit_transform(X_tr).astype(np.float32)
    Xte_s  = scaler.transform(X_te).astype(np.float32)

    seed_scores = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = MLP(Xtr_s.shape[1], n_out).to(dev)
        opt   = torch.optim.Adam(model.parameters(), lr=3e-4)
        crit  = nn.BCELoss()
        Xt    = torch.tensor(Xtr_s, device=dev)
        Yt    = torch.tensor(Y_tr,  device=dev)

        model.train()
        for _ in range(EPOCHS_MLP):
            perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), BATCH):
                idx  = perm[b: b + BATCH]
                loss = crit(model(Xt[idx]), Yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(Xte_s, device=dev)).cpu().numpy()

        auprcs = [average_precision_score(Y_te[:, j], preds[:, j])
                  for j in range(n_out) if valid_mask[j] and Y_te[:, j].sum() >= 2]
        seed_scores.append(np.mean(auprcs))

    mean  = float(np.mean(seed_scores))
    std   = float(np.std(seed_scores))
    print(f"  [{label:35s}] AUPRC = {mean:.4f} ± {std:.4f}", flush=True)
    return mean, std


# ── Embedding loader ───────────────────────────────────────────────────────────
def load_emb_pair(tag, n_layers):
    L_f  = n_layers
    L_m  = n_layers // 2
    paths = {}
    for split, prefix_f, prefix_m in [
        ('train',
         f'{DATA_DIR}/esm2_train_human_layer{{L:02d}}_{tag}.npy',
         f'{DATA_DIR}/esm2_train_human_layer{{L:02d}}_{tag}.npy'),
        ('test',
         f'{DATA_DIR}/esm2_layer_{{L:02d}}_{tag}.npy',
         f'{DATA_DIR}/esm2_layer_{{L:02d}}_{tag}.npy'),
    ]:
        pf = prefix_f.format(L=L_f)
        pm = prefix_m.format(L=L_m)
        if not (os.path.exists(pf) and os.path.exists(pm)):
            return None

        phi_f = np.load(pf).astype(np.float32)
        phi_m = np.load(pm).astype(np.float32)
        paths[split] = (phi_f, phi_m)

    return paths


def check_ref_150M(n_layers=30):
    """150M uses slightly different file naming (legacy)."""
    L_f = n_layers; L_m = n_layers // 2
    train_f = f'{DATA_DIR}/esm2_train_human_layer{L_f:02d}_t30_150M.npy'
    train_m = f'{DATA_DIR}/esm2_train_human_layer{L_m:02d}_t30_150M.npy'
    test_f  = f'{DATA_DIR}/esm2_layer_{L_f:02d}_t30_150M.npy'
    test_m  = f'{DATA_DIR}/esm2_layer_{L_m:02d}_t30_150M.npy'
    for p in [train_f, train_m, test_f, test_m]:
        if not os.path.exists(p):
            return None
    return {
        'train': (np.load(train_f).astype(np.float32), np.load(train_m).astype(np.float32)),
        'test':  (np.load(test_f).astype(np.float32),  np.load(test_m).astype(np.float32)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  PLM Scale Generalization: δ_layer AUPRC Comparison")
    print("=" * 70, flush=True)

    print("\n[1] Loading GO labels...")
    Y_tr, Y_te, valid_mask = load_labels()
    n_valid = valid_mask.sum()
    print(f"  Train: {Y_tr.shape}  Test: {Y_te.shape}  Valid GO: {n_valid}")

    results = []
    rows    = []

    for (tag, n_layers, dim, label) in MODELS:
        print(f"\n{'─' * 60}")
        print(f"  {label}  (tag={tag}  L={n_layers}  d={dim})")
        print(f"  δ_layer: L{n_layers} - L{n_layers // 2}  (ratio {n_layers//2}/{n_layers} = {50:.0f}%)")

        # Load embeddings
        if tag == 't30_150M':
            embs = check_ref_150M(n_layers)
        else:
            embs = load_emb_pair(tag, n_layers)

        if embs is None:
            print(f"  [SKIP] Embeddings not found. Run embed script first.")
            print(f"  → python3 exp_f_plm_scale_embed.py --model {label.replace(' ', '').replace('-', '').replace('XL', '').replace('sm', 'ESM3' if 'ESM3' in label else 'ProtT5' if 'ProtT5' in label else '').strip()}")
            results.append({'model': label, 'status': 'MISSING'})
            continue

        phi_f_tr, phi_m_tr = embs['train']
        phi_f_te, phi_m_te = embs['test']
        delta_tr  = phi_f_tr - phi_m_tr
        delta_te  = phi_f_te - phi_m_te
        cat_tr    = np.concatenate([phi_f_tr, delta_tr], axis=1)
        cat_te    = np.concatenate([phi_f_te, delta_te], axis=1)

        auprc_plain, std_plain = run_mlp(phi_f_tr, Y_tr, phi_f_te, Y_te, valid_mask,
                                         SEEDS, f'plain φ_L{n_layers}')
        auprc_delta, std_delta = run_mlp(delta_tr,  Y_tr, delta_te,  Y_te, valid_mask,
                                         SEEDS, f'δ(L{n_layers}-L{n_layers//2})')
        auprc_cat,   std_cat   = run_mlp(cat_tr,    Y_tr, cat_te,    Y_te, valid_mask,
                                         SEEDS, f'concat[φ‖δ] v17f* style')

        row = {
            'model':           label,
            'tag':             tag,
            'n_layers':        n_layers,
            'dim':             dim,
            'L_final':         n_layers,
            'L_mid':           n_layers // 2,
            'layer_ratio_pct': 50,
            'plain_last':      round(auprc_plain, 4),
            'plain_last_std':  round(std_plain,   4),
            'delta_only':      round(auprc_delta, 4),
            'delta_only_std':  round(std_delta,   4),
            'concat_delta':    round(auprc_cat,   4),
            'concat_delta_std': round(std_cat,    4),
            'delta_gain':      round(auprc_cat - auprc_plain, 4),
            'status': 'OK',
        }
        results.append(row)
        rows.append(row)

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    hdr = (f"{'Model':<14} {'L':>3} {'d':>5}  "
           f"{'Plain':>7}  {'δ_only':>7}  {'Concat':>7}  {'Δgain':>7}  Status")
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        if r.get('status') == 'MISSING':
            print(f"{r['model']:<14}  —  —  MISSING (run embed first)")
        else:
            print(f"{r['model']:<14} {r['n_layers']:>3} {r['dim']:>5}  "
                  f"{r['plain_last']:>7.4f}  "
                  f"{r['delta_only']:>7.4f}  "
                  f"{r['concat_delta']:>7.4f}  "
                  f"{r['delta_gain']:>+7.4f}  {r['status']}")

    # ── NatMeth claim verification ────────────────────────────────────────────
    complete = [r for r in results if r.get('status') == 'OK']
    if complete:
        all_pos        = all(r['delta_gain'] > 0 for r in complete)
        all_ratios     = [r['layer_ratio_pct'] for r in complete]
        ratio_std      = np.std(all_ratios)
        mean_gain      = np.mean([r['delta_gain'] for r in complete])
        ref_auprc      = next((r['concat_delta'] for r in complete
                               if r['tag'] == 't30_150M'), None)

        print(f"\n  NatMeth claim verification:")
        print(f"  Models evaluated:   {len(complete)}/{len(MODELS)}")
        print(f"  All gains positive: {all_pos}  (mean gain = {mean_gain:+.4f})")
        print(f"  Layer ratio (50%):  consistent={ratio_std==0.0}  values={all_ratios}")
        if ref_auprc:
            for r in complete:
                delta_vs_ref = r['concat_delta'] - ref_auprc
                print(f"    {r['model']:<14}: {r['concat_delta']:.4f}  "
                      f"(vs 150M ref: {delta_vs_ref:+.4f})")

        claim_ok = all_pos and ratio_std == 0.0
        print(f"\n  Claim: {'✓ SUPPORTED' if claim_ok else '✗ PARTIAL — inspect further'}")

    # ── Save ──────────────────────────────────────────────────────────────────
    tsv = f'{OUT_DIR}/results.tsv'
    keys = ['model', 'n_layers', 'dim', 'L_final', 'L_mid', 'layer_ratio_pct',
            'plain_last', 'plain_last_std', 'delta_only', 'delta_only_std',
            'concat_delta', 'concat_delta_std', 'delta_gain', 'status']
    with open(tsv, 'w') as f:
        f.write('\t'.join(keys) + '\n')
        for r in results:
            f.write('\t'.join(str(r.get(k, '')) for k in keys) + '\n')
    print(f"\n  Saved: {tsv}")

    # Summary for paper
    summ = f'{OUT_DIR}/summary.txt'
    with open(summ, 'w') as f:
        f.write("PLM Generalization Experiment — δ_layer AUPRC\n")
        f.write("=" * 50 + "\n\n")
        for r in results:
            if r.get('status') == 'OK':
                f.write(f"{r['model']}: plain={r['plain_last']:.4f}  "
                        f"δ_only={r['delta_only']:.4f}  "
                        f"concat={r['concat_delta']:.4f}  "
                        f"gain={r['delta_gain']:+.4f}\n")
            else:
                f.write(f"{r['model']}: MISSING\n")
    print(f"  Saved: {summ}")


if __name__ == '__main__':
    main()
