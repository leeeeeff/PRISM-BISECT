#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_crux_globallocal_dr.py  (Option B follow-up: does GLOBAL+LOCAL fusion rescue the per-GO DR-AUC?)

exp_crux_region_dr.py showed pure region-pool LOWERS the manuscript's centroid DR-AUC on
domain-differing 2-iso genes (mean-pool 0.8152 vs region-pool 0.7101, CI[-0.183,-0.032]).
That result only tests the two pooling strategies in isolation. This script tests whether a
CONVEX COMBINATION of the two (global context + local edit-region emphasis) can recover some of
the region-pool coherence gain (0.708->0.844 pairwise direction-acc) WITHOUT losing the mean-pool's
per-GO labeling signal -- i.e. whether region-pool is a pure net loss for DR-AUC or only a loss in
its extreme (alpha=0) form and a net gain is hiding at some interior alpha.

PRE-REGISTERED (S2, predict-before-look):
  H_conditional: some alpha in (0,1) gives DR-AUC(alpha) - DR-AUC(mean) > +0.02 (95% CI excludes 0)
                 -> region information is a *conditional* gain when blended with global context;
                    manuscript recipe claim should be REVISED to "blend", not retracted.
  H_pure_loss  : DR-AUC(alpha) is monotonic non-decreasing in alpha (best at alpha=1, i.e. pure
                 mean-pool) across the whole sweep -> region-pool contributes nothing but dilution
                 to the per-GO metric at any mixing ratio; crux verdict (region-pool = pure loss)
                 stands and the manuscript's fifth-line paragraph needs no further hedging.

Same population (domain-differing 2-iso genes), same MF labels, same LOGO-centroid DR-AUC protocol
as exp_crux_region_dr.py. E_alpha = normalize(alpha*E_mean + (1-alpha)*E_region), alpha in
{0.0,0.1,...,1.0}. Reuses region_pool_cache.npz (non-domain) + esm2_layer_*.npy (global mean-pool);
no new ESM-2 forward pass needed since exp_crux_region_dr.py already recomputed and would have
cached the domain-differing region-pool vectors -- so this script re-derives them the same way
(recompute-if-missing) rather than assuming a cache file exists.
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
OUT = ROOT / 'reports/muscle_labelgap/crux_globallocal_dr.json'
NTERM_WIN = 60; MAXLEN = 1022
ALPHAS = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]
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

    # ---- MF GO labels (manuscript machinery, identical to crux) ----
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

    # ---- ALL 2-iso pairs ----
    pairs = []
    for g in np.where(cnt == 2)[0]:
        a, b = np.where(gi == g)[0]
        if iso[a] not in faa or iso[b] not in faa: continue
        sa, sb = faa[iso[a]], faa[iso[b]]
        if sa == sb: continue
        pairs.append((g, a, b))
    sib_ivs = {}
    for g, a, b in pairs:
        sa, sb = faa[iso[a]][:MAXLEN], faa[iso[b]][:MAXLEN]
        sib_ivs[iso[a]] = opcode_ivs(sa, sb)
        sib_ivs[iso[b]] = opcode_ivs(sb, sa)

    # ---- mean + region embeddings ----
    nd = np.load(NONDOM_CACHE, allow_pickle=True)['pools'].item()
    region = {}; mean = {}; need = []
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

    universe = sorted(mean.keys())
    uidx = {i: k for k, i in enumerate(universe)}
    Emean = np.stack([mean[i] for i in universe]); Ereg = np.stack([region[i] for i in universe])
    Emean /= (np.linalg.norm(Emean, axis=1, keepdims=True) + 1e-9)
    Ereg /= (np.linalg.norm(Ereg, axis=1, keepdims=True) + 1e-9)
    univ_globalidx = np.array([idx_of[i] for i in universe])
    Yu = Y[univ_globalidx]
    univ_gene = gi[univ_globalidx]
    domdiff_pairs = [(g, a, b) for g, a, b in pairs if dom[a] != dom[b]]
    print(f"domain-differing pairs = {len(domdiff_pairs)}", flush=True)

    def dr_pairs(E):
        recs = []
        for g, a, b in domdiff_pairs:
            ia, ib = uidx[iso[a]], uidx[iso[b]]
            da, db = dom[a], dom[b]
            dbin = np.array([1.0 if da > db else 0.0, 1.0 if db > da else 0.0])
            for t in np.where(Yu[ia] > 0)[0]:
                mask = (Yu[:, t] > 0) & (univ_gene != g)
                if mask.sum() < 3: continue
                cen = E[mask].mean(0); cen /= (np.linalg.norm(cen) + 1e-9)
                recs.append((g, roc_auc_score(dbin, np.array([E[ia] @ cen, E[ib] @ cen]))))
        return recs

    print("\n=== alpha sweep: E_alpha = normalize(alpha*mean + (1-alpha)*region) ===", flush=True)
    sweep = {}
    per_alpha_recs = {}
    for a in ALPHAS:
        Ea = a * Emean + (1 - a) * Ereg
        Ea /= (np.linalg.norm(Ea, axis=1, keepdims=True) + 1e-9)
        recs = dr_pairs(Ea)
        aucs = np.array([r[1] for r in recs])
        sweep[a] = float(aucs.mean())
        per_alpha_recs[a] = recs
        print(f"  alpha={a:.1f}  DR-AUC={aucs.mean():.4f}  (n={len(aucs)})", flush=True)

    mean_auc = sweep[1.0]
    best_alpha = max((a for a in ALPHAS if a < 1.0), key=lambda a: sweep[a])
    best_auc = sweep[best_alpha]
    print(f"\n  pure mean-pool (alpha=1.0)   DR-AUC = {mean_auc:.4f}", flush=True)
    print(f"  best interior alpha={best_alpha:.1f}         DR-AUC = {best_auc:.4f}  Δ={best_auc-mean_auc:+.4f}", flush=True)

    # gene-cluster bootstrap: best interior alpha vs pure mean-pool
    rm = per_alpha_recs[1.0]; rb = per_alpha_recs[best_alpha]
    genes_arr = np.array([x[0] for x in rm]); dm = np.array([x[1] for x in rm]); db_ = np.array([x[1] for x in rb])
    assert len(rm) == len(rb), "record count mismatch between alpha=1.0 and best_alpha (filter drift)"
    ug = np.unique(genes_arr)
    boot = []
    for _ in range(1000):
        gs = rng.choice(ug, len(ug))
        sel = np.concatenate([np.where(genes_arr == g)[0] for g in gs])
        boot.append(db_[sel].mean() - dm[sel].mean())
    boot = np.array(boot); ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    monotonic = all(sweep[ALPHAS[k]] <= sweep[ALPHAS[k + 1]] + 1e-9 for k in range(len(ALPHAS) - 1))
    if ci[0] > 0.02:
        verdict = 'H_conditional (CI excludes 0, best interior alpha beats pure mean-pool by >0.02)'
    elif monotonic:
        verdict = 'H_pure_loss (DR-AUC monotonic non-decreasing in alpha; best point IS pure mean-pool)'
    else:
        verdict = 'H_flat (non-monotonic but interior gain not CI-significant at >0.02)'
    print(f"\n  bootstrap Δ(best_alpha={best_alpha:.1f} - mean) 95%CI [{ci[0]:+.4f},{ci[1]:+.4f}]", flush=True)
    print(f"  sweep monotonic-in-alpha: {monotonic}", flush=True)
    print(f"  -> {verdict}", flush=True)

    res = {'alpha_sweep': sweep, 'mean_pool_auc': mean_auc, 'best_alpha': best_alpha,
           'best_alpha_auc': best_auc, 'delta_best_vs_mean': best_auc - mean_auc,
           'bootstrap_ci': ci, 'monotonic_in_alpha': monotonic, 'verdict': verdict,
           'n_gene_term': len(rm),
           'prediction': 'H_conditional if some interior alpha CI>0.02 else H_pure_loss'}
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[saved] {OUT}", flush=True)


if __name__ == '__main__':
    main()
