#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_axis3_extravar_identity.py  (Option B follow-up: what is the EXTRA within-gene variance region-pool loads onto axis3?)

exp_traj_region_projection.py established: region-pool concentrates within-gene variance on the
domain axis (axis3 within-frac 0.341 mean -> 0.556 region, exceeding scrambled 0.399) WITHOUT
improving domain-completeness labeling (dir-acc 0.826 -> 0.843, CI [-0.028,+0.062] includes 0).
So region-pool loads MORE within-gene variance onto axis3 than mean-pool, but that extra variance
does NOT align with the domain-completeness label. What IS it?

Discriminative test: for each domain-differing pair, the within-pair axis3 excursion
  y = z3[hi] - z3[lo]  (hi = domain-complete; oriented so domain-complete is positive)
carries the axis3 signal. Its MEAN sign gives the labeling (0.843, unchanged). Its residual
MAGNITUDE (partialling out the domain-count difference, the label) is the "extra encoded" variance.
We ask which sequence-computable covariate of the SPLICE EDIT explains y beyond domain count, and
crucially which one region-pool AMPLIFIES relative to mean-pool (amplification = rho_region - rho_mean).
The amplified covariate = the identity of the extra within-gene variance region-pool concentrates.

Covariates (per pair, hi-minus-lo edited-region properties + edit geometry), all ESM-2-independent:
  ddom     = dom[hi]-dom[lo]              (the LABEL; control/partial-out)
  dlen_edit= |hi edited residues| - |lo edited residues|   (edit SIZE, signed)
  pos_edit = mean fractional position of hi edited residues in hi  (edit LOCATION, 0=N,1=C)
  ddis     = disorder(hi_edit) - disorder(lo_edit)   (TOP-IDP)
  dhyd     = hydropathy(hi_edit) - hydropathy(lo_edit) (Kyte-Doolittle)
  dchg     = netcharge(hi_edit) - netcharge(lo_edit)
  dlen_prot= len(hi) - len(lo)            (whole-protein length diff)

PRE-REGISTERED (S2): given axis5 was the encoded-only LENGTH axis (strong encoding, minimal
prediction usage; reference-esm2-pca-axes-final), I predict the EDIT-GEOMETRY covariates
(dlen_edit and/or pos_edit) amplify most under region-pool, identifying the extra within-gene
variance as edit-size/location geometry -- encoded-only isoform identity orthogonal to domain
function. If instead a functional covariate (ddis/dhyd/dchg) amplifies, the extra variance is
functionally describable and the encoding-only reading is wrong.
"""
import os, importlib.util
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from pathlib import Path
from scipy.stats import rankdata, t as tdist
import json, time, torch

ROOT = Path('/home/welcome1/sw1686/DIFFUSE'); MODEL = ROOT / 'hMuscle/model'
PCA_DIR = ROOT / 'reports/v20b_pca_interp'
OUT = ROOT / 'reports/muscle_labelgap/axis3_extravar_identity.json'
N_LAYERS, EMB_DIM, K, MAXLEN = 30, 640, 8, 1022
rng = np.random.default_rng(909)

spec = importlib.util.spec_from_file_location('ddr', MODEL / 'exp_domdiff_region_pool.py')
ddr = importlib.util.module_from_spec(spec); spec.loader.exec_module(ddr)

KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
      'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
TOPIDP = {'A':0.06,'R':0.180,'N':0.007,'D':0.192,'C':0.02,'Q':0.318,'E':0.736,'G':0.166,'H':0.303,
          'I':-0.486,'L':-0.326,'K':0.586,'M':-0.397,'F':-0.697,'P':0.987,'S':0.341,'T':0.059,
          'W':-0.884,'Y':-0.510,'V':-0.121}
def disorder(s): return float(np.mean([TOPIDP.get(a, 0) for a in s])) if s else 0.0
def hydro(s): return float(np.mean([KD.get(a, 0) for a in s])) if s else 0.0
def netchg(s): return float(sum((a in 'RK') - (a in 'DE') for a in s))


def partial_spearman(x, y, covars):
    def resid(v):
        r = rankdata(v).astype(float)
        C = np.column_stack([rankdata(c).astype(float) for c in covars] + [np.ones_like(r)])
        beta, *_ = np.linalg.lstsq(C, r, rcond=None); return r - C @ beta
    rx, ry = resid(x), resid(y); rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x); dof = max(n - len(covars) - 2, 1)
    ts = rho * np.sqrt(dof / max(1 - rho * rho, 1e-12))
    return rho, float(2 * tdist.sf(abs(ts), dof))


def project(T, mu, sd, pmean, W):
    n = T.shape[0]; Tn = (T - mu[None]) / sd[None]
    flat = Tn.reshape(n * N_LAYERS, EMB_DIM) - pmean[None]
    return (flat @ W.T).reshape(n, N_LAYERS, K).astype(np.float32)


def compute_traj(pairs, gpu):
    import esm
    seqs, ivs_map = {}, {}
    for p in pairs:
        seqs[p['hi_id']] = p['hi_seq']; seqs[p['lo_id']] = p['lo_seq']
        ivs_map[p['hi_id']] = p['hi_ivs']; ivs_map[p['lo_id']] = p['lo_ivs']
    ids = sorted(seqs, key=lambda i: len(seqs[i])); dev = f'cuda:{gpu}'
    model, alph = esm.pretrained.esm2_t30_150M_UR50D(); bc = alph.get_batch_converter()
    model.eval().to(dev); B = 8; t0 = time.time(); layers = list(range(1, N_LAYERS + 1)); out = {}
    with torch.no_grad():
        for s in range(0, len(ids), B):
            ch = ids[s:s + B]; _, _, toks = bc([(i, seqs[i]) for i in ch]); toks = toks.to(dev)
            rep = model(toks, repr_layers=layers)['representations']
            R = np.stack([rep[L].cpu().numpy() for L in layers], axis=1)
            for bi, i in enumerate(ch):
                Ln = len(seqs[i]); a = R[bi, :, 1:Ln + 1, :]
                cidx = [k for u, v in ivs_map[i] for k in range(u, v) if k < Ln]
                if not cidx: cidx = list(range(min(60, Ln)))
                out[i] = {'mean': a.mean(1), 'region': a[:, cidx, :].mean(1)}
            if s % (B * 20) == 0: print(f"  {s+len(ch)}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
    return out


def region_seq(seq, ivs):
    s = seq[:MAXLEN]; idx = [k for a, b in ivs for k in range(a, b) if k < len(s)]
    return ''.join(s[k] for k in idx)


def main():
    pairs = ddr.build_domdiff_pairs()
    print(f"domain-differing 2-iso pairs: {len(pairs)}", flush=True)
    mu = np.load(PCA_DIR / 'layer_stats_mu.npy'); sd = np.load(PCA_DIR / 'layer_stats_sd.npy')
    pmean = np.load(PCA_DIR / 'pca_mean_640.npy'); W = np.load(PCA_DIR / 'W_axes_8x640.npy')
    traj = compute_traj(pairs, int(os.environ.get('EMB_GPU', '0')))

    hi_ids = [p['hi_id'] for p in pairs]; lo_ids = [p['lo_id'] for p in pairs]
    zr = {}; zm = {}
    for pool, store in [('mean', zm), ('region', zr)]:
        Thi = np.stack([traj[i][pool] for i in hi_ids]); Tlo = np.stack([traj[i][pool] for i in lo_ids])
        Zhi = project(Thi, mu, sd, pmean, W).mean(1); Zlo = project(Tlo, mu, sd, pmean, W).mean(1)
        store['hi'] = Zhi; store['lo'] = Zlo
    AX = 3
    # orient axis3 by mean-pool so domain-complete (hi) is positive
    flip = np.sign((zm['hi'][:, AX] - zm['lo'][:, AX]).mean()); flip = flip if flip != 0 else 1.0
    y_mean = (zm['hi'][:, AX] - zm['lo'][:, AX]) * flip
    y_region = (zr['hi'][:, AX] - zr['lo'][:, AX]) * flip

    # covariates (hi-minus-lo edit properties; edit geometry)
    import numpy as _np
    rows = []
    for p in pairs:
        hi_r = region_seq(p['hi_seq'], p['hi_ivs']); lo_r = region_seq(p['lo_seq'], p['lo_ivs'])
        hidx = [k for a, b in p['hi_ivs'] for k in range(a, b) if k < min(len(p['hi_seq']), MAXLEN)]
        pos = float(_np.mean(hidx) / max(len(p['hi_seq'][:MAXLEN]), 1)) if hidx else 0.0
        rows.append(dict(
            ddom=ddr_domdiff(p),
            dlen_edit=len(hi_r) - len(lo_r),
            pos_edit=pos,
            ddis=disorder(hi_r) - disorder(lo_r),
            dhyd=hydro(hi_r) - hydro(lo_r),
            dchg=netchg(hi_r) - netchg(lo_r),
            dlen_prot=len(p['hi_seq']) - len(p['lo_seq'])))
    cov = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    ddom = cov['ddom']

    print(f"\n=== axis3 within-pair excursion decomposition (n={len(pairs)}, partial out ddom) ===", flush=True)
    print(f"  labeling check: mean(y_region>0)={(y_region>0).mean():.3f}  mean(y_mean>0)={(y_mean>0).mean():.3f}", flush=True)
    print(f"  var(y_region)={y_region.var():.4f}  var(y_mean)={y_mean.var():.4f}  ratio={y_region.var()/max(y_mean.var(),1e-9):.2f}", flush=True)
    print(f"\n  {'covariate':11} {'rho_region':>11} {'p_reg':>9} {'rho_mean':>10} {'amplification':>13}", flush=True)
    res = {'n_pairs': len(pairs), 'labeling': {'region_pos_frac': float((y_region > 0).mean()),
           'mean_pos_frac': float((y_mean > 0).mean())},
           'var_ratio_region_over_mean': float(y_region.var() / max(y_mean.var(), 1e-9)), 'covariates': {}}
    amps = []
    for c in ['dlen_edit', 'pos_edit', 'ddis', 'dhyd', 'dchg', 'dlen_prot']:
        rr, pr = partial_spearman(y_region, cov[c], [ddom])
        rm, pm = partial_spearman(y_mean, cov[c], [ddom])
        amp = abs(rr) - abs(rm)
        res['covariates'][c] = {'rho_region': rr, 'p_region': pr, 'rho_mean': rm, 'p_mean': pm, 'amplification': amp}
        amps.append((c, amp, rr, pr))
        print(f"  {c:11} {rr:>+11.3f} {pr:>9.2e} {rm:>+10.3f} {amp:>+13.3f}", flush=True)
    # also correlation of domain label itself (no partial) for reference
    rr_d, pr_d = partial_spearman(y_region, ddom, [np.zeros_like(ddom)])
    res['ddom_direct'] = {'rho_region': rr_d, 'p': pr_d}
    print(f"  {'ddom(ref)':11} {rr_d:>+11.3f} {pr_d:>9.2e}  (direct, the label)", flush=True)

    amps.sort(key=lambda t: -t[1])
    top = amps[0]
    bonf = 0.05 / 6
    verdict = (f"EDIT-GEOMETRY encoded-only: '{top[0]}' amplifies most (Δ|rho|={top[1]:+.3f}, region rho={top[2]:+.3f} p={top[3]:.1e})"
               if top[0] in ('dlen_edit', 'pos_edit') else
               f"FUNCTIONAL covariate '{top[0]}' amplifies most -> extra variance is functionally describable (rho={top[2]:+.3f})")
    res['top_amplified'] = top[0]; res['verdict'] = verdict; res['bonferroni_alpha'] = bonf
    print(f"\n  -> {verdict}", flush=True)
    print(f"  (Bonferroni α=0.05/6={bonf:.4f})", flush=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[saved] {OUT}", flush=True)


def ddr_domdiff(p):
    """dom[hi]-dom[lo] from the domain matrix (hi is domain-complete by construction)."""
    return _DOMDIFF[(p['hi_id'], p['lo_id'])]


_DOMDIFF = {}
def _build_domdiff():
    iso = np.array([ddr.clean(x) for x in np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)])
    dom = np.load(ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy').sum(1).astype(int)
    idx = {i: k for k, i in enumerate(iso)}
    for p in ddr.build_domdiff_pairs():
        _DOMDIFF[(p['hi_id'], p['lo_id'])] = int(dom[idx[p['hi_id']]] - dom[idx[p['lo_id']]])


if __name__ == '__main__':
    _build_domdiff()
    main()
