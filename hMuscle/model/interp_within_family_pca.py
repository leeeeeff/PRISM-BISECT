#!/usr/bin/env python3
"""
interp_within_family_pca.py  —  Option A: pooled within-family single-basis PCA
==============================================================================
사용자 통찰 승격 (S0 재프레임):
  원래 within-gene 추출(between-gene 상관 축과 독립인 축만 채택)은 *projection*이라
  "신호를 발견한 게 아니라 정의로 만들어냈을" 순환 위험이 있다. 대안은 *conditioning*:
  gene-family를 상수로 고정(변인통제)한 조건부 분포 안에서 축을 다시 뽑는다.

Option A = pooled within-family PCA (단일 basis):
  각 isoform의 (per-layer z-scored) 임베딩에서 그 isoform이 속한 gene-family의 평균을 빼고
  (= between-family variance 구조적 제거), 전부 pool 하여 PCA(8) 한 번.
  → 남는 축 = between-paralog(family 내부) + within-gene. 클러스터별 basis 매칭 불필요.

핵심 질문 (사용자: "주성분축이 보존되는가?"):
  gene-family confound(현 8축 전부를 지배)를 제거하면
    (i) 원본 joint 축과 같은 구조/조성 축(axis0=β-sheet, axis3=size...)이 그대로 재현되는가?
        → 축이 보존 = within 신호는 background 제거의 artifact가 아니라 실재.
    (ii) 아니면 isoform 변별(within-gene) 축이 새로 표면화하는가?
  비교 도구: |cosine| best-match + principal angles(부분공간) + Procrustes 잔차.
  Null: family label을 크기 보존하며 셔플(random grouping) → 보존/within_frac 상승이
        family-specific인지, 아니면 "아무 그룹 평균이나 빼도 나오는" trivial 효과인지 가른다.

설계: control(brain-fit joint PCA, 무-centering) vs treatment(family-centered PCA),
      둘 다 동일 per-layer z-score(muscle-train stats) 위에서 → 차이 = 순수 family-centering 효과.
      원본 muscle-fit W(reports/v20b_pca_interp/W_axes_8x640.npy)와도 대조.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import re, json, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
DATA = ROOT / "hMuscle/data"
BRAIN_DIR = DATA / "brain_isoquant_esm2/full"
GTF = DATA / "brain_esm2/brain_only.gtf"
HMM = ROOT / "hMuscle/results_isoform/features/hmmscan_brain.domtblout"
OUT = ROOT / "reports" / "v20b_pca_interp"
INTERP_OUT = OUT / "within_family"
INTERP_OUT.mkdir(parents=True, exist_ok=True)

N_LAYERS, EMB_DIM, K, SEED, N_PERM = 30, 640, 8, 42, 10
rng = np.random.default_rng(SEED)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

FEAT_NAMES = ["length","gravy","aromatic","instab","pI","ndom","helix","turn","sheet",
              "net_charge","pos_frac","neg_frac","charged_frac","cys","pro","gly","his",
              "trp","aliphatic","aromatic2","max_aa","entropy"]


# ---------------------------------------------------------------- data loading
def load_brain_traj_normalized():
    """brain trajectory (N,30,640) with muscle-train per-layer z-score (원본과 동일 정규화)."""
    mu = np.load(OUT / "layer_stats_mu.npy")   # (30,640)
    sd = np.load(OUT / "layer_stats_sd.npy")
    sym = np.array([str(s).replace("b'","").replace("'","").replace('"',"").replace(" ","")
                    for s in np.load(BRAIN_DIR / "brain_full_gene_names.npy", allow_pickle=True)])
    N = len(sym)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(N_LAYERS):
        arr = np.load(BRAIN_DIR / f"brain_full_esm2_layer{L+1:02d}_t30_150M.npy", mmap_mode="r")
        traj[:, L, :] = (arr.astype(np.float32) - mu[L]) / sd[L]
        del arr
    return traj, sym


def gene_to_family():
    """gene(symbol) -> family label. family = gene isoform들에서 가장 흔한 dominant Pfam accession.
    도메인 없는 gene -> 'NODOM::<gene>' (자기 자신 = singleton family)."""
    bids = np.array([str(x) for x in np.load(BRAIN_DIR / "brain_full_ids.npy", allow_pickle=True)])
    sym  = np.array([str(s).replace("b'","").replace("'","").replace('"',"").replace(" ","")
                     for s in np.load(BRAIN_DIR / "brain_full_gene_names.npy", allow_pickle=True)])
    name2enst = {}
    for line in open(GTF):
        if "\ttranscript\t" not in line: continue
        em = re.search(r'transcript_id "([^"]+)"', line); nm = re.search(r'transcript_name "([^"]+)"', line)
        if em and nm: name2enst[nm.group(1)] = em.group(1).split(".")[0]
    enst2dom = defaultdict(list)
    for line in open(HMM):
        if line.startswith("#") or not line.strip(): continue
        p = line.split()
        if float(p[12]) > 1e-5: continue
        enst2dom[p[3].split(".p")[0].split(".")[0]].append((p[1].split(".")[0], float(p[12])))
    def enst_of(i, bid):
        if bid.startswith("ENST"): return bid.split(".")[0]
        if bid.startswith("transcript"): return bid
        return name2enst.get(bid, "")
    # per-gene domain vote (best-evalue domain of each isoform)
    gene_domvote = defaultdict(Counter)
    for i, bid in enumerate(bids):
        doms = enst2dom.get(enst_of(i, bid), [])
        if doms:
            best = min(doms, key=lambda x: x[1])[0]
            gene_domvote[sym[i]][best] += 1
    fam_of_gene = {}
    for g in np.unique(sym):
        vote = gene_domvote.get(g)
        fam_of_gene[g] = vote.most_common(1)[0][0] if vote else f"NODOM::{g}"
    return sym, np.array([fam_of_gene[g] for g in sym])


# ------------------------------------------------------------- family centering
def family_center_inplace(traj, fam_idx, F):
    """각 layer에서 family 평균을 빼 within-family residual로 (in place). 반환: 삭제할 zero-residual 행 mask 없음(전부 유지)."""
    for L in range(N_LAYERS):
        sums = np.zeros((F, EMB_DIM), dtype=np.float64)
        np.add.at(sums, fam_idx, traj[:, L, :].astype(np.float64))
        counts = np.bincount(fam_idx, minlength=F).astype(np.float64)[:, None]
        means = (sums / np.maximum(counts, 1)).astype(np.float32)
        traj[:, L, :] -= means[fam_idx]


def fit_pca(flat):
    pca = PCA(n_components=K, random_state=SEED, svd_solver="randomized")
    Zf = pca.fit_transform(flat)
    return pca.components_.astype(np.float32), pca.explained_variance_ratio_.astype(np.float32), \
           Zf.reshape(-1, N_LAYERS, K).astype(np.float32)


# ----------------------------------------------------------------- comparisons
def best_match_cos(Wa, Wb):
    """greedy |cos| best match: returns list of (a_idx, b_idx, |cos|)."""
    C = np.abs(Wa @ Wb.T)  # (K,K)
    used_b = set(); pairs = []
    order = np.dstack(np.unravel_index(np.argsort(-C, axis=None), C.shape))[0]
    used_a = set()
    for a, b in order:
        if a in used_a or b in used_b: continue
        used_a.add(a); used_b.add(b); pairs.append((int(a), int(b), float(C[a, b])))
        if len(pairs) == K: break
    return sorted(pairs)


def principal_angles_cos(Wa, Wb):
    """PCA components are orthonormal rows -> singular values of Wa@Wb.T = cos(principal angles)."""
    sv = np.linalg.svd(Wa @ Wb.T, compute_uv=False)
    return np.clip(sv, 0, 1)


def within_frac(Zk, gene, multi_mask):
    x = Zk[multi_mask]; g = gene[multi_mask]
    grand = x.mean(); ss_tot = float(((x - grand) ** 2).sum())
    idx = defaultdict(list)
    for i, gg in enumerate(g): idx[gg].append(i)
    ss_w = sum(float(((x[ii] - x[ii].mean()) ** 2).sum()) for ii in idx.values())
    return ss_w / ss_tot if ss_tot > 0 else np.nan


def biology_id(Z, feat, gene, gene2idx):
    """각 축 gene-mean 점수 vs 22 feature Spearman; iso-level(gene-mean 잔차)도. 상위 3개 반환."""
    axis = Z.mean(1)  # (N,8) layer-mean
    out = {}
    for k in range(K):
        ag = np.full(len(gene), np.nan)
        for gg, ix in gene2idx.items():
            ag[np.array(ix)] = np.nanmean(axis[ix, k])
        cors = []
        for j, fn in enumerate(FEAT_NAMES):
            m = ~np.isnan(ag) & ~np.isnan(feat[:, j])
            if m.sum() < 50: continue
            r = stats.spearmanr(ag[m], feat[m, j])[0]
            cors.append((fn, float(r)))
        cors.sort(key=lambda x: -abs(x[1]))
        out[k] = cors[:3]
    return out


# ---------------------------------------------------------------------- main
def main():
    log("[1] brain trajectory + muscle-stat z-score")
    traj, sym = load_brain_traj_normalized()
    N = len(sym)
    _, fam = gene_to_family()
    fams, fam_idx = np.unique(fam, return_inverse=True); F = len(fams)
    g2c = Counter(sym); multi_mask = np.array([g2c[g] >= 2 for g in sym])
    gene2idx = defaultdict(list)
    for i, g in enumerate(sym): gene2idx[g].append(i)
    # family size in GENES
    fam_genes = defaultdict(set)
    for i in range(N): fam_genes[fam[i]].add(sym[i])
    multi_gene_fams = sum(1 for v in fam_genes.values() if len(v) >= 2)
    n_iso_multifam = sum(1 for i in range(N) if len(fam_genes[fam[i]]) >= 2)
    log(f"    N={N}, genes={len(g2c)}, families={F}, multi-gene families={multi_gene_fams}, "
        f"isoforms in multi-gene families={n_iso_multifam} ({100*n_iso_multifam/N:.1f}%), "
        f"multi-iso isoforms={int(multi_mask.sum())}")

    feat = np.load(OUT / "feat_matrix_brain.npy")
    W_orig = np.load(OUT / "W_axes_8x640.npy")  # muscle-fit joint (원본)

    # ---- CONTROL: brain-fit joint PCA (no centering) ----
    log("[2] CONTROL: brain-fit joint PCA(8)")
    W_joint, evr_joint, Z_joint = fit_pca(traj.reshape(N * N_LAYERS, EMB_DIM))
    # within_frac per axis: use layer-mean axis score
    wf_joint = [within_frac(Z_joint[:, :, k].mean(1), sym, multi_mask) for k in range(K)]
    log(f"    joint evr={[round(float(v),4) for v in evr_joint]} sum={float(evr_joint.sum()):.4f}")
    log(f"    joint within_frac={[round(v,3) for v in wf_joint]}")

    # ---- TREATMENT: family-centered PCA ----
    log("[3] TREATMENT: family-center in place -> within-family PCA(8)")
    family_center_inplace(traj, fam_idx, F)
    W_within, evr_within, Z_within = fit_pca(traj.reshape(N * N_LAYERS, EMB_DIM))
    del traj
    wf_within = [within_frac(Z_within[:, :, k].mean(1), sym, multi_mask) for k in range(K)]
    log(f"    within-family evr={[round(float(v),4) for v in evr_within]} sum={float(evr_within.sum()):.4f}")
    log(f"    within-family within_frac={[round(v,3) for v in wf_within]}")

    # ---- axis preservation ----
    log("[4] axis preservation (within-family vs joint vs original)")
    pa_wj = principal_angles_cos(W_within, W_joint)
    pa_wo = principal_angles_cos(W_within, W_orig)
    pa_jo = principal_angles_cos(W_joint, W_orig)
    log(f"    principal-angle cos  within~joint : {[round(float(v),3) for v in pa_wj]}  mean={pa_wj.mean():.3f}")
    log(f"    principal-angle cos  within~orig  : {[round(float(v),3) for v in pa_wo]}  mean={pa_wo.mean():.3f}")
    log(f"    principal-angle cos  joint ~orig  : {[round(float(v),3) for v in pa_jo]}  mean={pa_jo.mean():.3f}")
    bm = best_match_cos(W_within, W_joint)
    log("    within-axis -> best joint-axis |cos|:")
    for a, b, c in bm: log(f"       within{a} ~ joint{b}: |cos|={c:.3f}")

    # ---- biology re-identification of within-family axes ----
    log("[5] biology identity of within-family axes (gene-mean vs 22 feat)")
    bio_within = biology_id(Z_within, feat, sym, gene2idx)
    for k in range(K):
        log(f"    within-axis {k} (evr={evr_within[k]:.4f}, wf={wf_within[k]:.3f}): "
            + ", ".join(f"{fn}({r:+.2f})" for fn, r in bio_within[k]))

    # ---- NULL: random grouping (family-size-preserving permutation) ----
    log(f"[6] NULL: random grouping x{N_PERM} (size-preserving) — reload traj")
    traj2, _ = load_brain_traj_normalized()
    # permute gene->family among genes keeping family gene-composition sizes:
    # shuffle the gene labels' family assignment by permuting families' member-gene sets
    genes = np.array(sorted(g2c.keys()))
    gfam = {g: fam[np.where(sym == g)[0][0]] for g in genes}  # gene's family
    null_wf = {k: [] for k in range(K)}; null_pa = []
    base_traj = traj2.copy()
    for p in range(N_PERM):
        perm_g = rng.permutation(genes)
        remap = {perm_g[i]: gfam[genes[i]] for i in range(len(genes))}  # gene -> shuffled family
        fam_p = np.array([remap[g] for g in sym])
        _, fam_idx_p = np.unique(fam_p, return_inverse=True); Fp = len(np.unique(fam_p))
        traj2[...] = base_traj
        family_center_inplace(traj2, fam_idx_p, Fp)
        Wp, evrp, Zp = fit_pca(traj2.reshape(N * N_LAYERS, EMB_DIM))
        for k in range(K): null_wf[k].append(within_frac(Zp[:, :, k].mean(1), sym, multi_mask))
        null_pa.append(float(principal_angles_cos(Wp, W_joint).mean()))
        log(f"    perm {p+1}/{N_PERM}: within_frac(top axes)="
            f"{[round(null_wf[k][-1],3) for k in range(3)]} pa~joint={null_pa[-1]:.3f}")
    del traj2, base_traj

    log("[7] REAL vs NULL summary")
    for k in range(K):
        nm, ns = np.mean(null_wf[k]), np.std(null_wf[k]) + 1e-9
        z = (wf_within[k] - nm) / ns
        log(f"    axis {k}: within_frac real={wf_within[k]:.3f}  null={nm:.3f}±{ns:.3f}  z={z:+.2f}")
    nm_pa, ns_pa = np.mean(null_pa), np.std(null_pa) + 1e-9
    log(f"    principal-angle(within~joint) real mean={pa_wj.mean():.3f}  "
        f"null={nm_pa:.3f}±{ns_pa:.3f}  z={(pa_wj.mean()-nm_pa)/ns_pa:+.2f}")

    # ---- save ----
    result = {
        "N": int(N), "n_families": int(F), "multi_gene_families": int(multi_gene_fams),
        "frac_iso_in_multigene_fam": float(n_iso_multifam / N),
        "evr_joint": [float(v) for v in evr_joint], "evr_within": [float(v) for v in evr_within],
        "within_frac_joint": [float(v) for v in wf_joint],
        "within_frac_within": [float(v) for v in wf_within],
        "within_frac_null_mean": {k: float(np.mean(null_wf[k])) for k in range(K)},
        "within_frac_null_std": {k: float(np.std(null_wf[k])) for k in range(K)},
        "principal_angle_cos": {"within_joint": [float(v) for v in pa_wj],
                                 "within_orig": [float(v) for v in pa_wo],
                                 "joint_orig": [float(v) for v in pa_jo],
                                 "null_within_joint_mean": float(nm_pa),
                                 "null_within_joint_std": float(ns_pa)},
        "best_match_within_to_joint": bm,
        "biology_within": {k: bio_within[k] for k in range(K)},
    }
    np.save(INTERP_OUT / "W_within_8x640.npy", W_within)
    np.save(INTERP_OUT / "Z_within_Nx30x8.npy", Z_within)
    with open(INTERP_OUT / "within_family_pca.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"[done] saved -> {INTERP_OUT}")


if __name__ == "__main__":
    main()
