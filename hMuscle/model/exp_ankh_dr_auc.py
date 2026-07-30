#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_ankh_dr_auc.py
==================
Cross-PLM within-gene Domain-Ranking AUC (novelty metric generality).

Trains the v17f*-style MLP head on each PLM's concat[φ_final ‖ δ_layer] embeddings
(muscle held-out), produces per-isoform predictions, and computes the within-gene
Domain-Ranking AUC (whether isoforms with MORE Pfam domains are ranked higher within
their gene). ESM-2 150M is the sanity anchor (manuscript DR-AUC ≈ 0.630); gene-mean
oracle must score 0.500 by construction.

Reuses: exp_f_plm_scale_scan.load_labels / MLP  +  domain_ranking_validation logic.
"""
import os, json
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')
from exp_f_plm_scale_scan import load_labels, MLP, SEEDS, EPOCHS_MLP, BATCH, clean

DATA_DIR = '../data'
DOMAIN_MAT = '../results_isoform/features/domain_matrix_proper_test.npy'
OUT = '../../reports/exp_f_plm_scale/dr_auc.json'

# PLM concat embeddings: (tag, n_layers)
PLMS = [('t30_150M', 30), ('prot_t5_xl', 24), ('ankh_base', 48)]


def load_concat(tag, n_layers):
    Lf, Lm = n_layers, n_layers // 2
    if tag == 't30_150M':
        ftr = f'{DATA_DIR}/esm2_train_human_layer{Lf}_t30_150M.npy'
        mtr = f'{DATA_DIR}/esm2_train_human_layer{Lm}_t30_150M.npy'
        fte = f'{DATA_DIR}/esm2_layer_{Lf}_t30_150M.npy'
        mte = f'{DATA_DIR}/esm2_layer_{Lm}_t30_150M.npy'
    else:
        ftr = f'{DATA_DIR}/esm2_train_human_layer{Lf:02d}_{tag}.npy'
        mtr = f'{DATA_DIR}/esm2_train_human_layer{Lm:02d}_{tag}.npy'
        fte = f'{DATA_DIR}/esm2_layer_{Lf:02d}_{tag}.npy'
        mte = f'{DATA_DIR}/esm2_layer_{Lm:02d}_{tag}.npy'
    phi_f_tr, phi_m_tr = np.load(ftr).astype(np.float32), np.load(mtr).astype(np.float32)
    phi_f_te, phi_m_te = np.load(fte).astype(np.float32), np.load(mte).astype(np.float32)
    cat_tr = np.concatenate([phi_f_tr, phi_f_tr - phi_m_tr], axis=1)
    cat_te = np.concatenate([phi_f_te, phi_f_te - phi_m_te], axis=1)
    return cat_tr, cat_te


def train_predict(X_tr, Y_tr, X_te, seeds):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sc = MaxAbsScaler()
    Xtr = sc.fit_transform(X_tr).astype(np.float32)
    Xte = sc.transform(X_te).astype(np.float32)
    preds_seeds = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = MLP(Xtr.shape[1], Y_tr.shape[1]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4); crit = nn.BCELoss()
        Xt = torch.tensor(Xtr, device=dev); Yt = torch.tensor(Y_tr, device=dev)
        model.train()
        for _ in range(EPOCHS_MLP):
            perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), BATCH):
                idx = perm[b:b + BATCH]
                loss = crit(model(Xt[idx]), Yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds_seeds.append(model(torch.tensor(Xte, device=dev)).cpu().numpy())
    return np.mean(preds_seeds, axis=0)


def domain_ranking_auc(preds, gene2idxs, iso_n_domains, Y_te):
    aucs = []
    for g, idxs in gene2idxs.items():
        if len(idxs) < 2: continue
        dom = iso_n_domains[idxs]
        if dom.std() < 0.1: continue
        pos_terms = np.where(Y_te[idxs[0]] > 0)[0]
        if len(pos_terms) == 0: continue
        med = np.median(dom)
        binlabel = (dom > med).astype(float)
        if binlabel.sum() == 0 or binlabel.sum() == len(idxs): continue
        pg = preds[idxs]
        for t in pos_terms:
            s = pg[:, t]
            if s.std() < 1e-8:
                aucs.append(0.5); continue
            try: aucs.append(roc_auc_score(binlabel, s))
            except: pass
    return (np.mean(aucs) if aucs else 0.5), len(aucs)


def main():
    print("=" * 66); print("  Cross-PLM within-gene Domain-Ranking AUC"); print("=" * 66)
    Y_tr, Y_te, valid_mask = load_labels()
    print(f"  labels: train {Y_tr.shape}  test {Y_te.shape}")

    iso_n_domains = np.load(DOMAIN_MAT).sum(axis=1).astype(np.int32)
    te_genes = [clean(g).split('.')[0] for g in np.load('my_gene_list_fixed.npy', allow_pickle=True)]
    assert len(te_genes) == Y_te.shape[0] == len(iso_n_domains), \
        f"align: genes {len(te_genes)} Y_te {Y_te.shape[0]} dom {len(iso_n_domains)}"
    from collections import defaultdict
    gene2idxs = defaultdict(list)
    for i, g in enumerate(te_genes): gene2idxs[g].append(i)
    gene2idxs = {g: np.array(ix) for g, ix in gene2idxs.items()}

    out = {}
    for tag, nL in PLMS:
        cat_tr, cat_te = load_concat(tag, nL)
        preds = train_predict(cat_tr, Y_tr, cat_te, SEEDS)
        dr, n = domain_ranking_auc(preds, gene2idxs, iso_n_domains, Y_te)
        # gene-mean null on same preds
        gm = preds.copy()
        for g, ix in gene2idxs.items():
            gm[ix] = preds[ix].mean(axis=0, keepdims=True)
        dr_gm, _ = domain_ranking_auc(gm, gene2idxs, iso_n_domains, Y_te)
        out[tag] = {'dr_auc': float(dr), 'n_term_pairs': int(n), 'dr_gene_mean_null': float(dr_gm)}
        print(f"  {tag:12s}  DR-AUC = {dr:.4f}  (N={n:,})   gene-mean null = {dr_gm:.4f}")

    json.dump(out, open(OUT, 'w'), indent=2)
    print(f"\n[Saved] {OUT}")


if __name__ == '__main__':
    main()
