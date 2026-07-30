#!/usr/bin/env python
"""
S1 multi-hypothesis triangulation for the pooling-DISCARDED SLiM manifold's biological identity.

Background: b_slim_dispersion_structure.py found the mean-pool-discarded non-DC residual is
STRUCTURED (gene-disjoint reproducible, ~48-55D). b_option_B_slim_target.py tested exactly ONE
candidate identity (project's own SLiM regex classes) and found it null (generic, "position/
length/context"). That was a single-hypothesis test treated as if exhaustive -- this script fills
the gap: cross-correlate the SAME discarded top-mode against candidate biology NEVER checked
against this specific object: (a) composition covariates (helix/sheet/hydro/charge delta, Chou-
Fasman/Kyte-Doolittle scales -- reused verbatim from explore_internal_edit_covariates.py) and
(b) the established 8-axis joint-PCA framework (W_axes_8x640.npy, axis0=disorder/axis3=domain/
axis5=length etc, already characterised in FRAMEWORK.md) via direction-alignment.

Label-free, uses the FULL slim manifest population (n~hundreds) -- NOT the n=20 ELM-instance set,
so this is NOT subject to the CV-fold power wall devils-advocate flagged (FATAL, Attack 1) for the
B5 anchor-calibration proposal. This is purely descriptive/correlational, not a supervision label.

T1: per-pair top discarded mode v (sign-fixed SVD direction of non-DC residual, L9) -> z-scored via
    layer_stats_sd (matching how W_axes/Z were built) -> unit-normalize -> cosine with each of the
    8 axis directions. Correlate |cosine|_axis_k against |composition_covariate| across pairs
    (Pearson r + permutation null, since sign of v and of W rows is an arbitrary SVD/PCA convention
    -- only magnitude-vs-magnitude is directly interpretable without a sign-alignment step).
T2: subspace-level check -- does the gene-disjoint reproducible top-K subspace of these modes
    overlap the 8-dim W_axes subspace more than a random-K subspace would? (principal-angle energy)
"""
import os
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[v]='4'
import numpy as np, pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from scipy import stats

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/model_interpretability_map'
PERRES = OUT / 'b_perres'
L = 9
MINRES = 24
rng = np.random.default_rng(0)
N_PERM = 2000

HELIX = {'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,'H':1.00,
         'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,'T':0.83,'W':1.08,
         'Y':0.69,'V':1.06}
SHEET = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,'H':0.87,
         'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,'T':1.19,'W':1.37,
         'Y':1.47,'V':1.70}
HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
         'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,
         'Y':-1.3,'V':4.2}
CHARGE = {'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.1}
AXIS_LABEL = {0:'axis0(disorder)',1:'axis1(LRR/Ig)',2:'axis2(Pro-turn)',3:'axis3(domain)',
              4:'axis4(helix-charge)',5:'axis5(length)',6:'axis6(KRAB-ZNF)',7:'axis7(acidic-helical)'}

seq = {}
with open(OUT / 'b_extract_sequences.fasta') as f:
    cur = None
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('>'): cur = line[1:]
        elif line: seq[cur] = line
man = pd.read_csv(OUT / 'b_manifest_pairs.tsv', sep='\t')
man = man[man.cls == 'slim'].reset_index(drop=True)

W = np.load(ROOT / 'reports/v20b_pca_interp/W_axes_8x640.npy')          # (8,640) unit rows, z-scored space
sd_l9 = np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_sd.npy')[L-1]  # (640,) layer-9 per-dim sd

def load(idx):
    p = PERRES / f'{idx}.npz'
    return np.load(p)[f'L{L}'].astype(np.float32) if p.exists() else None

def pair_data(r):
    ls, ss = seq.get(str(r.long_idx)), seq.get(str(r.short_idx))
    HLm, HSm = load(r.long_idx), load(r.short_idx)
    if ls is None or ss is None or HLm is None or HSm is None: return None
    ops = SequenceMatcher(None, ls, ss, autojunk=False).get_opcodes()
    D = [HLm[i1+o]-HSm[j1+o] for tag,i1,i2,j1,j2 in ops if tag=='equal'
         for o in range(i2-i1) if i1+o<HLm.shape[0] and j1+o<HSm.shape[0]]
    if len(D) < MINRES: return None
    D = np.stack(D); R = D - D.mean(0)
    _,S,Vt = np.linalg.svd(R, full_matrices=False)
    v = S[0]*Vt[0]; v = v*np.sign(v[np.argmax(np.abs(v))])
    changed = [ls[i1+o] for tag,i1,i2,j1,j2 in ops if tag!='equal' for o in range(i2-i1) if i1+o<len(ls)]
    if not changed: changed = list(ls)  # fallback: whole long seq if no edited residues resolvable
    helix = np.mean([HELIX.get(a,1.0) for a in changed])
    sheet = np.mean([SHEET.get(a,1.0) for a in changed])
    hydro = np.mean([HYDRO.get(a,0.0) for a in changed])
    charge = np.mean([CHARGE.get(a,0.0) for a in changed])
    return v, helix, sheet, hydro, charge

modes, helix_l, sheet_l, hydro_l, charge_l, genes = [], [], [], [], [], []
for _, r in man.iterrows():
    out = pair_data(r)
    if out is None: continue
    v, h, s, hy, c = out
    modes.append(v); helix_l.append(h); sheet_l.append(s); hydro_l.append(hy); charge_l.append(c)
    genes.append(r.gene)
modes = np.stack(modes)
cov = pd.DataFrame({'helix':helix_l,'sheet':sheet_l,'hydro':hydro_l,'charge':charge_l})
genes = np.array(genes)
n = len(modes)
print(f'n pairs (slim, L9, resolvable)={n}  genes={len(set(genes))}')

# ---- T1: per-pair axis alignment ----
modes_z = modes / sd_l9[None,:]
norms = np.linalg.norm(modes_z, axis=1)
keep = norms > 1e-12
if (~keep).sum():
    print(f'  [drop] {(~keep).sum()} degenerate (zero-norm) mode vectors dropped before T1/T2')
modes_z, cov, genes = modes_z[keep], cov[keep].reset_index(drop=True), genes[keep]
n = keep.sum()
modes_z = modes_z / norms[keep, None]
cosines = modes_z @ W.T     # (n,8)

print('\n=== T1: |cosine(discarded-mode, axis_k)| vs |composition covariate| (Pearson r, perm p) ===')
rows=[]
for k in range(8):
    absc = np.abs(cosines[:,k])
    for cname in ['helix','sheet','hydro','charge']:
        x = np.abs(cov[cname].to_numpy() - np.median(cov[cname].to_numpy()))
        r_obs,_ = stats.pearsonr(absc, x)
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            null[i] = stats.pearsonr(absc, rng.permutation(x))[0]
        p = (np.sum(np.abs(null) >= np.abs(r_obs)) + 1) / (N_PERM + 1)
        rows.append(dict(axis=AXIS_LABEL[k], covariate=cname, r=r_obs, p=p))
resdf = pd.DataFrame(rows)
print(resdf.pivot(index='axis', columns='covariate', values='r').round(3).to_string())
print('\n-- p-values --')
print(resdf.pivot(index='axis', columns='covariate', values='p').round(4).to_string())
sig = resdf[resdf.p < 0.05/32]
print(f'\nBonferroni(32 tests) survivors: {len(sig)}')
if len(sig): print(sig.to_string(index=False))

# ---- T2: subspace-level overlap with W_axes (8D), gene-disjoint, vs random-K null ----
print('\n=== T2: gene-disjoint reproducible subspace overlap with W_axes(8D) ===')
ug = np.unique(genes); rng.shuffle(ug)
tr_mask = np.isin(genes, ug[:len(ug)//2])
Xtr = modes_z[tr_mask]
K = 50
_,_,Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False)
P = Vt[:K]                                   # (K,640) subspace basis from discarded modes
# energy of W_axes rows captured by P (both already ~unit-ish in z-space)
cap_axes = np.sum((W @ P.T)**2, axis=1)      # per-axis captured energy (0..1 ideally, P not orthonormal to W but rows of P ARE orthonormal)
nulls = np.zeros((20,8))
for i in range(20):
    Q = np.linalg.qr(rng.standard_normal((640, K)))[0].T
    nulls[i] = np.sum((W @ Q.T)**2, axis=1)
for k in range(8):
    print(f'  {AXIS_LABEL[k]:22s} captured={cap_axes[k]:.3f}  random-K null mean={nulls[:,k].mean():.3f}  '
          f'excess={cap_axes[k]-nulls[:,k].mean():+.3f}')
print('\n[done]')
