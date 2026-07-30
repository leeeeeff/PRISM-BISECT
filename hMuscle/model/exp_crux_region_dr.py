#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_crux_region_dr.py  (CRUX: does region-pool lift the manuscript's Domain-Ranking AUC?)

Attack-4 crux. My direction-accuracy estimator showed region-pool >> mean-pool (0.708->0.844) on
domain-differing pairs. But that estimator uses the PAIRWISE orientation (a domain-count-trained
anchor). The manuscript's DR-AUC is a PER-ISOFORM, PER-GO score (training-free centroid = 0.638;
trained PRISM = 0.630). This script rebuilds the manuscript's centroid DR-AUC protocol EXACTLY on
the same population and same MF labels, changing ONLY the per-isoform embedding pooling:
  score(isoform, term) = cosine(embedding_isoform, term_centroid_LOGO)
  term_centroid_LOGO   = mean embedding of positive isoforms for the term, excluding current gene
  DR-AUC per gene x positive-term: AUC of (within-gene domain-count binary) vs score; aggregate.

Compare mean-pool vs changed-region-pool embeddings.
PRE-REGISTERED (S2):
  H_lift : region-pool centroid DR-AUC > mean-pool centroid DR-AUC by >0.02 -> the manuscript's
           per-GO within-gene ceiling is liftable by region-pooling (central-claim revision needed).
  H_stand: region-pool centroid DR-AUC ~= mean-pool (CI overlaps) -> the 0.844 direction gain does
           NOT transfer to the per-GO scorable metric; it was pairwise-orientation information the
           per-isoform predictor cannot use -> the manuscript's ceiling stands, region-pool only
           improves an alignment-informed pairwise direction task (distinct from the DR metric).
Region-pool embedding for an isoform = pooled over residues differing from its within-gene sibling
(alignment-based, no domain-annotation leak). Uses non-domain region_pool_cache + recomputes the
domain-differing isoforms.
"""
import os, re, gzip, time
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from pathlib import Path
from collections import defaultdict
import json, torch, importlib.util
from difflib import SequenceMatcher
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/welcome1/sw1686/DIFFUSE'); DATA = ROOT / 'hMuscle/data'; MODEL = ROOT / 'hMuscle/model'
ID_DIR = DATA / 'raw_data/data/id_lists'; ANNOT = DATA / 'raw_data/data/annotations'
DOMAIN_MAT = ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy'
MF_TSV = ROOT / 'reports/v_expanded_gomf/mf_domain_vs_prism.tsv'
NONDOM_CACHE = ROOT / 'reports/muscle_labelgap/region_pool_cache.npz'
OUT = ROOT / 'reports/muscle_labelgap/crux_region_dr.json'
NTERM_WIN = 60; MAXLEN = 1022
rng = np.random.default_rng(404)

spec = importlib.util.spec_from_file_location('rp', MODEL / 'exp_nondomain_region_pool.py')
rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)


def clean(g): return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(' ', "")


def opcode_ivs(a, b):
    ivs = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != 'equal' and i2 > i1:
            ivs.append((i1, i2))
    return ivs


def main():
    iso = np.array([clean(x) for x in np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)])
    gen = np.array([clean(x) for x in np.load(MODEL / 'my_gene_list_fixed.npy', allow_pickle=True)])
    dom = np.load(DOMAIN_MAT).sum(1).astype(int)
    L15 = np.load(DATA / 'esm2_layer_15_t30_150M.npy').astype(np.float32)
    L30 = np.load(DATA / 'esm2_layer_30_t30_150M.npy').astype(np.float32)
    faa = rp.parse_faa()
    idx_of = {i: k for k, i in enumerate(iso)}
    gl, gi = np.unique(gen, return_inverse=True); cnt = np.bincount(gi, minlength=len(gl))

    # ---- MF GO labels (manuscript machinery) ----
    ENSG2SYM = {}
    with open(ID_DIR / 'ensembl_to_symbol.txt') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
    sym2id = {}
    with gzip.open(ANNOT / 'Homo_sapiens.gene_info.gz', 'rt') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) > 2:
                sym2id[p[2]] = p[1]
                if len(p) > 4 and p[4] != '-':
                    for s in p[4].split('|'):
                        sym2id.setdefault(s, p[1])
    go_mf = defaultdict(set)
    with gzip.open(ANNOT / 'gene2go.gz', 'rt') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] != '9606' or p[7] != 'Function': continue
            go_mf[p[2]].add(p[1])
    mf_terms = []
    with open(MF_TSV) as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 6: mf_terms.append(p[0])
    sym = np.array([ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in gen])
    Y = np.stack([np.array([1.0 if sym2id.get(s, '__') in go_mf[go] else 0.0 for s in sym], np.float32)
                  for go in mf_terms], axis=1)
    print(f"MF terms: {len(mf_terms)}  Y shape {Y.shape}", flush=True)

    # ---- collect ALL 2-iso pairs (both domain-differing and non-domain) for region-pool ----
    pairs = []
    for g in np.where(cnt == 2)[0]:
        a, b = np.where(gi == g)[0]
        if iso[a] not in faa or iso[b] not in faa: continue
        sa, sb = faa[iso[a]], faa[iso[b]]
        if sa == sb: continue
        pairs.append((g, a, b))
    # sibling changed region per isoform id
    sib_ivs = {}
    for g, a, b in pairs:
        sa, sb = faa[iso[a]][:MAXLEN], faa[iso[b]][:MAXLEN]
        sib_ivs[iso[a]] = opcode_ivs(sa, sb)
        sib_ivs[iso[b]] = opcode_ivs(sb, sa)

    # ---- region-pool embeddings: load non-domain cache, recompute the rest ----
    nd = np.load(NONDOM_CACHE, allow_pickle=True)['pools'].item()
    region = {}  # iso_id -> 1280 (c15|c30)
    mean = {}    # iso_id -> 1280 (m15|m30) from global npy by index
    need = []
    for g, a, b in pairs:
        for x in (a, b):
            iid = iso[x]
            mean[iid] = np.concatenate([L15[x], L30[x]])
            if iid in nd:
                region[iid] = np.concatenate([nd[iid]['c15'], nd[iid]['c30']])
            else:
                need.append(iid)
    print(f"pairs={len(pairs)}  region cached={len(region)}  to-recompute={len(set(need))}", flush=True)

    if need:
        import esm
        seqs = {iid: faa[iid][:MAXLEN] for iid in set(need)}
        ids = sorted(seqs, key=lambda i: len(seqs[i]))
        dev = 'cuda:0'
        model, alph = esm.pretrained.esm2_t30_150M_UR50D(); bc = alph.get_batch_converter()
        model.eval().to(dev); B = 16; t0 = time.time()
        with torch.no_grad():
            for s in range(0, len(ids), B):
                ch = ids[s:s + B]
                _, _, toks = bc([(i, seqs[i]) for i in ch]); toks = toks.to(dev)
                out = model(toks, repr_layers=[15, 30])
                r15 = out['representations'][15].cpu().numpy(); r30 = out['representations'][30].cpu().numpy()
                for bi, i in enumerate(ch):
                    Ln = len(seqs[i]); a15 = r15[bi, 1:Ln + 1]; a30 = r30[bi, 1:Ln + 1]
                    cidx = [k for u, v in sib_ivs[i] for k in range(u, v) if k < Ln]
                    if not cidx: cidx = list(range(min(NTERM_WIN, Ln)))
                    region[i] = np.concatenate([a15[cidx].mean(0), a30[cidx].mean(0)])
                if s % (B * 20) == 0: print(f"  recompute {s+len(ch)}/{len(ids)} {time.time()-t0:.0f}s", flush=True)

    # embedding matrices over pair-isoform universe
    universe = sorted(mean.keys())
    uidx = {i: k for k, i in enumerate(universe)}
    Emean = np.stack([mean[i] for i in universe]); Ereg = np.stack([region[i] for i in universe])
    Emean /= (np.linalg.norm(Emean, axis=1, keepdims=True) + 1e-9)
    Ereg /= (np.linalg.norm(Ereg, axis=1, keepdims=True) + 1e-9)
    # term membership over universe
    univ_globalidx = np.array([idx_of[i] for i in universe])
    Yu = Y[univ_globalidx]                      # (U, T)
    univ_gene = gi[univ_globalidx]

    # ---- DR-AUC via LOGO centroid, on domain-DIFFERING 2-iso genes ----
    def dr_auc(E):
        aucs = []
        for g, a, b in pairs:
            if dom[a] == dom[b]:
                continue                        # domain-differing only (manuscript DR population)
            ia, ib = uidx[iso[a]], uidx[iso[b]]
            da, db = dom[a], dom[b]
            dbin = np.array([1.0 if da > db else 0.0, 1.0 if db > da else 0.0])
            pos_terms = np.where(Yu[ia] > 0)[0]
            for t in pos_terms:
                # LOGO centroid: positive isoforms for term t, excluding this gene
                mask = (Yu[:, t] > 0) & (univ_gene != g)
                if mask.sum() < 3:
                    continue
                cen = E[mask].mean(0); cen /= (np.linalg.norm(cen) + 1e-9)
                sc = np.array([E[ia] @ cen, E[ib] @ cen])
                if len(np.unique(dbin)) < 2:
                    continue
                aucs.append(roc_auc_score(dbin, sc))
        return float(np.mean(aucs)), len(aucs)

    m_auc, m_n = dr_auc(Emean)
    r_auc, r_n = dr_auc(Ereg)
    print(f"\n=== CRUX: centroid Domain-Ranking AUC (domain-differing 2-iso genes) ===", flush=True)
    print(f"  mean-pool     DR-AUC = {m_auc:.4f}  (n={m_n} gene×term)", flush=True)
    print(f"  region-pool   DR-AUC = {r_auc:.4f}  (n={r_n})", flush=True)
    print(f"  Δ(region-mean) = {r_auc-m_auc:+.4f}", flush=True)

    # gene-cluster bootstrap on the paired per-(gene,term) AUC difference
    def dr_pairs(E):
        recs = []
        for g, a, b in pairs:
            if dom[a] == dom[b]: continue
            ia, ib = uidx[iso[a]], uidx[iso[b]]
            da, db = dom[a], dom[b]
            dbin = np.array([1.0 if da > db else 0.0, 1.0 if db > da else 0.0])
            if len(np.unique(dbin)) < 2: continue
            for t in np.where(Yu[ia] > 0)[0]:
                mask = (Yu[:, t] > 0) & (univ_gene != g)
                if mask.sum() < 3: continue
                cen = E[mask].mean(0); cen /= (np.linalg.norm(cen) + 1e-9)
                recs.append((g, roc_auc_score(dbin, np.array([E[ia] @ cen, E[ib] @ cen]))))
        return recs
    rm = dr_pairs(Emean); rr = dr_pairs(Ereg)
    # align by (index) — same order/filters
    genes_arr = np.array([x[0] for x in rm]); dm = np.array([x[1] for x in rm]); dr = np.array([x[1] for x in rr])
    ug = np.unique(genes_arr)
    boot = []
    for _ in range(1000):
        gs = rng.choice(ug, len(ug))
        sel = np.concatenate([np.where(genes_arr == g)[0] for g in gs])
        boot.append(dr[sel].mean() - dm[sel].mean())
    boot = np.array(boot); ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    verdict = 'H_lift (CI excludes 0, region lifts per-GO DR)' if ci[0] > 0.02 else \
              ('H_partial (CI>0 but <0.02)' if ci[0] > 0 else 'H_stand (CI includes 0)')
    print(f"  gene-cluster bootstrap Δ 95%CI [{ci[0]:+.4f},{ci[1]:+.4f}] -> {verdict}", flush=True)

    res = {'mean_pool_dr_auc': m_auc, 'region_pool_dr_auc': r_auc, 'delta': r_auc - m_auc,
           'n_gene_term': m_n, 'bootstrap_ci': ci, 'verdict': verdict,
           'prediction': 'H_lift if >0.02 else H_stand'}
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[saved] {OUT}", flush=True)


if __name__ == '__main__':
    main()
