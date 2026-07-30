#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_traj_region_projection.py  (decisive-subset: does region-pool reshape trajectory-PCA axis biology toward LABELS?)

devils-advocate verdict = SCOPE-DECISIVE-SUBSET-FIRST on the multi-isoform region-pool trajectory
proposal. This is that decisive subset, incorporating the three mandatory design fixes:
  C2 (PCA comparability): DO NOT refit PCA on region-pool. Project region/mean/scrambled poolings
     onto the ORIGINAL mean-pool axes W(8x640) + the original per-layer z-score stats + PCA mean,
     so all three live in the identical coordinate system and their variance components are commensurable.
  C1 (tautology guard): include a length-matched SCRAMBLED-region pooling null. A within-gene
     variance-fraction rise is only meaningful if it exceeds what an arbitrary same-length window gives.
  H2 (parsimony): run only the 2-iso domain-DIFFERING pairs (changed interval unambiguous, domain-
     completeness label available), all 30 layers, ~356 isoforms -> <1 GPU-hour, before any 64k run.

The LABELING gate (the load-bearing, falsifiable test):
  axis3 is the established DOMAIN axis (carries within-gene Domain-Ranking, dDR z=+8.39, manuscript S4c).
  Orient axis k ONCE by mean-pool (flip sign so mean-pool within-pair direction-accuracy >= 0.5;
  domain-complete isoform = higher score). Then, under that FIXED orientation, direction-accuracy of
  a pooling = fraction of pairs where score[domain-complete] > score[domain-truncated]. This is an
  EXTERNAL-label test (domain completeness is known a priori), distinct from self-consistent coherence.
PRE-REGISTERED (S2):
  H_reproduce: region-pool axis3 direction-accuracy <= mean-pool (CI includes 0 or negative) ->
     coherence!=labeling reproduced at the PCA-interpretation level; region-pool concentrates within-
     gene variance (descriptive) but does NOT make the domain axis more label-aligned. The full 64k
     recompute would then add sample size, not a new falsifiable claim.
  H_falsify: region-pool axis3 direction-accuracy > mean-pool by a gene-cluster-bootstrap CI excluding
     0 -> region-pool REORGANISES the trajectory so the domain axis becomes MORE label-aligned; this
     would overturn coherence!=labeling at the representation level (surprising; justifies the 64k run).
Tautology gate: within-gene variance fraction(region) vs (scrambled). region>>scram => changed region
  specifically concentrates within-gene variance; region~=scram => the rise is a generic-window artifact.
"""
import os, time, importlib.util
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from pathlib import Path
import json, torch

ROOT = Path('/home/welcome1/sw1686/DIFFUSE'); MODEL = ROOT / 'hMuscle/model'
PCA_DIR = ROOT / 'reports/v20b_pca_interp'
OUT = ROOT / 'reports/muscle_labelgap/traj_region_projection.json'
N_LAYERS, EMB_DIM, K = 30, 640, 8
MAXLEN = 1022
rng = np.random.default_rng(707)

spec = importlib.util.spec_from_file_location('ddr', MODEL / 'exp_domdiff_region_pool.py')
ddr = importlib.util.module_from_spec(spec); spec.loader.exec_module(ddr)


def project(T, mu, sd, pmean, W):
    """T:(n,30,640) any-pool per-layer -> Z:(n,30,8) in ORIGINAL mean-pool axis system."""
    n = T.shape[0]
    Tn = (T - mu[None]) / sd[None]
    flat = Tn.reshape(n * N_LAYERS, EMB_DIM) - pmean[None]
    return (flat @ W.T).reshape(n, N_LAYERS, K).astype(np.float32)


def compute_traj(pairs, gpu):
    """Return dict iso_id -> {'mean':(30,640),'region':(30,640),'scram':(30,640)} across all 30 layers."""
    import esm
    seqs, ivs_map = {}, {}
    for p in pairs:
        seqs[p['hi_id']] = p['hi_seq']; seqs[p['lo_id']] = p['lo_seq']
        ivs_map[p['hi_id']] = p['hi_ivs']; ivs_map[p['lo_id']] = p['lo_ivs']
    ids = sorted(seqs, key=lambda i: len(seqs[i]))
    dev = f'cuda:{gpu}'
    model, alph = esm.pretrained.esm2_t30_150M_UR50D(); bc = alph.get_batch_converter()
    model.eval().to(dev); B = 8; t0 = time.time()
    layers = list(range(1, N_LAYERS + 1))
    out = {}
    with torch.no_grad():
        for s in range(0, len(ids), B):
            ch = ids[s:s + B]
            _, _, toks = bc([(i, seqs[i]) for i in ch]); toks = toks.to(dev)
            rep = model(toks, repr_layers=layers)['representations']
            R = np.stack([rep[L].cpu().numpy() for L in layers], axis=1)  # (b,30,T,640)
            for bi, i in enumerate(ch):
                Ln = len(seqs[i]); a = R[bi, :, 1:Ln + 1, :]              # (30,Ln,640)
                cidx = [k for u, v in ivs_map[i] for k in range(u, v) if k < Ln]
                if not cidx: cidx = list(range(min(60, Ln)))
                L0 = len(cidx)
                # length-matched scrambled contiguous window (same length as changed region)
                if L0 >= Ln: sidx = list(range(Ln))
                else:
                    start = int(rng.integers(0, Ln - L0 + 1)); sidx = list(range(start, start + L0))
                out[i] = {'mean': a.mean(1), 'region': a[:, cidx, :].mean(1), 'scram': a[:, sidx, :].mean(1)}
            if s % (B * 20) == 0: print(f"  {s+len(ch)}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
    return out


def within_gene_frac(Zlm, gid):
    """Zlm:(n,8) layer-mean axis scores. one-way ANOVA within-gene fraction per axis (2-iso genes)."""
    out = []
    for k in range(K):
        x = Zlm[:, k]; grand = x.mean(); sstot = ((x - grand) ** 2).sum()
        ssw = 0.0
        for g in np.unique(gid):
            xi = x[gid == g]; ssw += ((xi - xi.mean()) ** 2).sum()
        out.append(float(ssw / (sstot + 1e-12)))
    return out


def dir_acc(Zlm_hi, Zlm_lo, flip):
    """direction accuracy per axis under fixed orientation `flip` (+1/-1). domain-complete=hi."""
    d = (Zlm_hi - Zlm_lo) * flip[None]           # (n,8)
    return (d > 0).mean(0)                        # fraction hi>lo per axis


def main():
    pairs = ddr.build_domdiff_pairs()
    print(f"domain-differing 2-iso pairs: {len(pairs)}", flush=True)
    mu = np.load(PCA_DIR / 'layer_stats_mu.npy'); sd = np.load(PCA_DIR / 'layer_stats_sd.npy')
    pmean = np.load(PCA_DIR / 'pca_mean_640.npy'); W = np.load(PCA_DIR / 'W_axes_8x640.npy')

    traj = compute_traj(pairs, int(os.environ.get('EMB_GPU', '0')))
    hi_ids = [p['hi_id'] for p in pairs]; lo_ids = [p['lo_id'] for p in pairs]
    gid = np.array([p['g'] for p in pairs])

    res = {'n_pairs': len(pairs), 'design': 'project region/mean/scram onto ORIGINAL mean-pool axes (C2)',
           'axes': {}, 'within_gene_frac': {}, 'dir_acc_axis3': {}}
    Zproj = {}
    for pool in ['mean', 'region', 'scram']:
        Thi = np.stack([traj[i][pool] for i in hi_ids]); Tlo = np.stack([traj[i][pool] for i in lo_ids])
        Zhi = project(Thi, mu, sd, pmean, W); Zlo = project(Tlo, mu, sd, pmean, W)
        Zproj[pool] = (Zhi.mean(1), Zlo.mean(1))   # layer-mean (n,8)

    # orientation fixed by MEAN-POOL (domain-complete = hi under this sign)
    dhi_m, dlo_m = Zproj['mean']
    flip = np.sign((dhi_m - dlo_m).mean(0)); flip[flip == 0] = 1.0

    print("\n=== within-gene variance fraction (layer-mean, projected onto ORIGINAL axes) ===", flush=True)
    for pool in ['mean', 'region', 'scram']:
        hi, lo = Zproj[pool]
        allz = np.concatenate([hi, lo]); allg = np.concatenate([gid, gid])
        wf = within_gene_frac(allz, allg)
        res['within_gene_frac'][pool] = wf
        print(f"  {pool:7} axis-wise within-frac: " + " ".join(f"{v:.3f}" for v in wf), flush=True)
        print(f"          axis3={wf[3]:.3f}  mean-over-8={np.mean(wf):.3f}", flush=True)

    print("\n=== direction accuracy vs DOMAIN-COMPLETENESS label (fixed mean-pool orientation) ===", flush=True)
    da = {}
    for pool in ['mean', 'region', 'scram']:
        hi, lo = Zproj[pool]; acc = dir_acc(hi, lo, flip)
        da[pool] = acc; res['axes'][pool] = {'dir_acc_per_axis': acc.tolist()}
        print(f"  {pool:7} axis3 dir-acc={acc[3]:.4f}   (all axes: " + " ".join(f"{v:.2f}" for v in acc) + ")", flush=True)

    # gene-cluster bootstrap on axis3 direction-accuracy difference region - mean
    dhi_r, dlo_r = Zproj['region']; dhi_m2, dlo_m2 = Zproj['mean']
    corr_r = ((dhi_r[:, 3] - dlo_r[:, 3]) * flip[3] > 0).astype(float)
    corr_m = ((dhi_m2[:, 3] - dlo_m2[:, 3]) * flip[3] > 0).astype(float)
    ug = np.unique(gid)
    gr = np.array([corr_r[gid == g].mean() for g in ug]); gm = np.array([corr_m[gid == g].mean() for g in ug])
    boot = np.array([(gr[idx] - gm[idx]).mean() for idx in (rng.integers(0, len(ug), len(ug)) for _ in range(1000))])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    res['dir_acc_axis3'] = {'mean': float(da['mean'][3]), 'region': float(da['region'][3]),
                            'scram': float(da['scram'][3]), 'delta_region_minus_mean': float(da['region'][3] - da['mean'][3]),
                            'bootstrap_ci': ci}
    label_gate = 'H_falsify (region axis3 MORE label-aligned; CI>0)' if ci[0] > 0 else \
                 'H_reproduce (region axis3 NOT more label-aligned; coherence!=labeling holds at PCA level)'
    taut = res['within_gene_frac']
    taut_gate = ('changed-region concentrates within-gene variance beyond generic window'
                 if taut['region'][3] > taut['scram'][3] + 0.02 else
                 'within-frac rise is generic-window artifact (tautology)')
    print(f"\n  axis3 dir-acc  mean={da['mean'][3]:.4f}  region={da['region'][3]:.4f}  scram={da['scram'][3]:.4f}", flush=True)
    print(f"  Δ(region-mean) axis3 dir-acc 95%CI [{ci[0]:+.4f},{ci[1]:+.4f}] -> {label_gate}", flush=True)
    print(f"  within-frac axis3: region={taut['region'][3]:.3f} scram={taut['scram'][3]:.3f} -> {taut_gate}", flush=True)
    res['label_gate'] = label_gate; res['tautology_gate'] = taut_gate
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[saved] {OUT}", flush=True)


if __name__ == '__main__':
    main()
