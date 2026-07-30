# -*- coding: utf-8 -*-
"""
extract_loc_features.py
========================
Stream D 특징 추출: 세포 내 위치 신호 (8-dim per isoform)

Features (인덱스 순서 고정):
  0  mts_score          : MTS score — N-term 양전하 + 양친매성 helix
  1  nterm_hydrophob    : N-term 20aa 평균 Kyte-Doolittle 소수성  [-4.5, 4.5] → norm
  2  nterm_pro_density  : N-term 30aa 중 Pro 비율  [0, 1]
  3  sp_score           : 신호 펩타이드 점수 — N-term 30aa 소수성 코어
  4  nls_score          : NLS 점수 — K/R 군집 밀도
  5  tm_count_norm      : 추정 TM 도메인 수 (norm by 5)
  6  gpi_score          : GPI 앵커 신호 — C-term 25aa 소수성
  7  amphipathic_score  : N-term 30aa 양친매성 helix score (수소결합 모멘트)

생물학적 연결:
  - NDUFS8 NIC switch: mts_score (CT-NIC TLLWTELFR → MTS), nterm_pro_density (AD-NIC PVLPTG)
  - NDUFS4/7 Complex I: mts_score (미토콘드리아 타겟팅)
  - ZNF736/ZNF582: nls_score (핵 전사인자)
  - DOCK11/DOCK10: tm_count_norm (세포질 GEF)

데이터 소스:
  - hMuscle/data/transcripts.fasta.transdecoder.pep    (test isoforms)
  - hMuscle/results_isoform/features/train_proteins.fasta  (train isoforms)
  - NM→ENST 매핑 불필요 (NM_ ID가 train_proteins.fasta에 직접 존재)

출력:
  hMuscle/results_isoform/features/loc/
    loc_features_test.npy    (36748 × 8, float32)
    loc_features_train.npy   (31668 × 8, float32)
    loc_feature_stats.json

실행:
  cd /home/welcome1/sw1686/DIFFUSE
  conda activate isoform_env
  python hMuscle/preprocessing/extract_loc_features.py
"""

import os, json, math
import numpy as np
from collections import defaultdict

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PEP     = os.path.join(BASE_DIR, 'data/transcripts.fasta.transdecoder.pep')
TRAIN_PEP    = os.path.join(BASE_DIR, 'results_isoform/features/train_proteins.fasta')
MODEL_DIR    = os.path.join(BASE_DIR, 'model')
ID_DIR       = os.path.join(BASE_DIR, 'data/raw_data/data/id_lists')
OUT_DIR      = os.path.join(BASE_DIR, 'results_isoform/features/loc')
os.makedirs(OUT_DIR, exist_ok=True)

N_FEAT     = 8
FEAT_NAMES = [
    'mts_score', 'nterm_hydrophob', 'nterm_pro_density', 'sp_score',
    'nls_score', 'tm_count_norm', 'gpi_score', 'amphipathic_score',
]

# ── Kyte-Doolittle 소수성 스케일 ──────────────────────────────────────────────
KD = {
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8,  'K': -3.9, 'M': 1.9,  'F': 2.8,  'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}
KD_MAX = max(KD.values())   # 4.5
KD_MIN = min(KD.values())   # -4.5

# Eisenberg 양친매성 helix 계산용 — 나선형 각도 (100°/잔기)
HELIX_ANGLE = math.radians(100.0)


def mean_kd(seq, start=0, end=20):
    """mean Kyte-Doolittle for seq[start:end]"""
    sub = seq[start:end]
    if not sub:
        return 0.0
    vals = [KD.get(aa, 0.0) for aa in sub]
    return sum(vals) / len(vals)


def hydrophobic_moment(seq, start=0, length=18):
    """
    Eisenberg hydrophobic moment for alpha-helix (μH).
    μH = sqrt( (Σ KD_i * sin(i*θ))² + (Σ KD_i * cos(i*θ))² ) / N
    θ = 100° for alpha-helix
    """
    sub = seq[start:start + length]
    if len(sub) < 3:
        return 0.0
    sin_sum, cos_sum = 0.0, 0.0
    for i, aa in enumerate(sub):
        h = KD.get(aa, 0.0)
        angle = i * HELIX_ANGLE
        sin_sum += h * math.sin(angle)
        cos_sum += h * math.cos(angle)
    mu = math.sqrt(sin_sum**2 + cos_sum**2) / len(sub)
    return mu / KD_MAX  # normalize by max KD


def max_amphipathic(seq, search_len=60, helix_len=18):
    """
    슬라이딩 윈도우 최대 양친매성 helix 점수.
    첫 search_len aa 내에서 helix_len=18aa 윈도우를 이동하며 최대 μH 반환.

    고정 N-term 윈도우 대비 장점:
      - MTS 양친매성 helix가 N-term 어느 위치에 있든 포착
      - NDUFS8 CT-NIC: TLLWTELFR (pos 18-26) → 위치 14 시작 윈도우에서 최대값
      - AD-NIC: PVLPTG proline-break → 어느 윈도우도 높은 μH 형성 불가
    """
    end = min(len(seq), search_len)
    best = 0.0
    for start in range(0, end - helix_len + 1):
        mu = hydrophobic_moment(seq, start=start, length=helix_len)
        if mu > best:
            best = mu
    return best


def mts_score(seq):
    """
    MTS (mitochondrial targeting sequence) score:
    N-term 30aa 기준
      - 양전하: R/K 수 (> 2 → MTS 특징)
      - 양친매성 helix: hydrophobic_moment
      - Pro 부재: Pro 비율 낮을수록 MTS 가능
    Score = (pos_charge_score * 0.4 + amphipathic * 0.4 + no_pro * 0.2) ∈ [0, 1]

    생물학 근거: NDUFS8 CT-NIC TLLWTELFR = amphipathic, R/K 풍부, Pro 없음
                 AD-NIC PVLPTG = Pro-break → MTS 불가
    """
    nterm = seq[:30]
    if not nterm:
        return 0.0

    pos_charge = sum(1 for aa in nterm if aa in ('R', 'K'))
    pos_score  = min(pos_charge / 5.0, 1.0)  # ≥5개이면 1.0

    # 슬라이딩 윈도우 최대 μH — 고정 윈도우 대비 MTS helix 포착 정확도 향상
    amphi = max_amphipathic(seq, search_len=40, helix_len=min(18, len(nterm)))

    pro_count = nterm.count('P')
    no_pro    = max(0.0, 1.0 - pro_count / 3.0)  # Pro 3개 이상 → 0.0

    return float(pos_score * 0.4 + amphi * 0.4 + no_pro * 0.2)


def nterm_hydrophob(seq, length=20):
    """N-term 20aa 평균 KD, 정규화 → [0, 1]"""
    val = mean_kd(seq, 0, length)
    return float((val - KD_MIN) / (KD_MAX - KD_MIN + 1e-8))


def nterm_pro_density(seq, length=30):
    """N-term 30aa 중 Pro 비율"""
    sub = seq[:length]
    if not sub:
        return 0.0
    return float(sub.count('P') / len(sub))


def sp_score(seq):
    """
    Signal peptide proxy:
    - 첫 5aa: n-region (+ charge)
    - 10-25aa: hydrophobic core (h-region)
    - 25-35aa: cleavage site (polar)
    Score = (hydrophobic core mean KD / KD_MAX + cleavage polarity)
    """
    if len(seq) < 20:
        return 0.0
    h_region = seq[5:min(25, len(seq))]
    c_region = seq[min(25, len(seq)):min(35, len(seq))]
    h_kd  = mean_kd(h_region)
    c_pol = sum(1 for aa in c_region if aa in ('A', 'G', 'S', 'T')) / (len(c_region) + 1)
    return float(max(0.0, h_kd / KD_MAX) * 0.7 + c_pol * 0.3)


def nls_score(seq):
    """
    NLS (nuclear localization signal) score:
    Classical NLS: K/R-rich monopartite (KKKRKV) or bipartite
    Score = max K/R density in any 8aa window
    """
    if not seq:
        return 0.0
    best = 0.0
    for i in range(len(seq) - 7):
        window = seq[i:i+8]
        kr_frac = sum(1 for aa in window if aa in ('K', 'R')) / 8.0
        if kr_frac > best:
            best = kr_frac
    return float(best)


def tm_count(seq, min_len=17):
    """
    TM domain count: 연속 소수성 stretch ≥ min_len aa
    (KD > 1.6 for all residues)
    """
    count = 0
    run = 0
    for aa in seq:
        if KD.get(aa, 0.0) > 1.6:
            run += 1
            if run == min_len:
                count += 1
        else:
            run = 0
    return count


def gpi_score(seq, tail_len=25):
    """
    GPI anchor signal: C-terminal 소수성 tail (omega site).
    High hydrophobicity in last 25aa → GPI candidate
    """
    tail = seq[-tail_len:] if len(seq) >= tail_len else seq
    if not tail:
        return 0.0
    kd_val = mean_kd(tail)
    return float(max(0.0, kd_val / KD_MAX))


def compute_features(aa_seq):
    """
    aa_seq: 단백질 서열 (1-letter AA code, 대문자)
    Returns: np.array (8,) float32
    """
    if not aa_seq or len(aa_seq) < 5:
        return None

    # Stop codon 제거 ('*')
    aa_seq = aa_seq.replace('*', '').replace('X', '').upper()

    n_tm = tm_count(aa_seq)

    feat = np.array([
        mts_score(aa_seq),
        nterm_hydrophob(aa_seq),
        nterm_pro_density(aa_seq),
        sp_score(aa_seq),
        nls_score(aa_seq),
        min(n_tm / 5.0, 1.0),           # tm_count_norm (≥5 → 1.0)
        gpi_score(aa_seq),
        max_amphipathic(aa_seq),         # 슬라이딩 윈도우 최대 μH (전체 서열)
    ], dtype=np.float32)

    return feat


# ── 1. 단백질 FASTA 파싱 함수 ─────────────────────────────────────────────────
def parse_pep_fasta(path, id_transform=None):
    """
    FASTA 파싱 → {base_id: protein_seq}
    id_transform: ID 변환 함수 (없으면 기본 처리)
    multi-isoform (*.p1, *.p2 등) 중 첫 번째 (최고 점수) 사용
    """
    seqs = {}
    current_id = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if current_id is not None and buf:
                    if current_id not in seqs:  # 첫 번째 isoform 우선
                        seqs[current_id] = ''.join(buf)
                raw_id = line[1:].strip().split()[0]
                if id_transform:
                    current_id = id_transform(raw_id)
                else:
                    current_id = raw_id
                buf = []
            else:
                buf.append(line.upper())
    if current_id is not None and buf and current_id not in seqs:
        seqs[current_id] = ''.join(buf)
    return seqs


def test_id_transform(raw_id):
    """TransDecoder pep ID: BambuTx10.p1 → BambuTx10 / ENST00000286479.4.p1 → ENST00000286479"""
    base = raw_id.split('.p')[0]   # remove .p1, .p2
    base = base.split('.')[0]      # remove version (ENST.X)
    return base


# ── 2. Test set 단백질 로드 ───────────────────────────────────────────────────
print("[1] Loading test protein sequences ...")
test_prots = parse_pep_fasta(TEST_PEP, id_transform=test_id_transform)
print(f"   Loaded: {len(test_prots):,} unique isoforms")


# ── 3. Train set 단백질 로드 ──────────────────────────────────────────────────
print("[2] Loading train protein sequences ...")
train_prots = parse_pep_fasta(TRAIN_PEP)
print(f"   Loaded: {len(train_prots):,} unique isoforms")


# ── 4. ID 로더 ────────────────────────────────────────────────────────────────
def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


# ── 5. Test feature 계산 ─────────────────────────────────────────────────────
print("[3] Computing test localization features ...")
te_iso  = load_ids(os.path.join(MODEL_DIR, 'my_isoform_list_fixed.npy'))
N_TE    = len(te_iso)
X_te    = np.zeros((N_TE, N_FEAT), dtype=np.float32)
te_miss = 0

for i, iso in enumerate(te_iso):
    base = iso.split('.')[0]
    prot = test_prots.get(base, '')
    feat = compute_features(prot) if prot else None
    if feat is not None:
        X_te[i] = feat
    else:
        te_miss += 1

te_cov = (N_TE - te_miss) / N_TE
print(f"   Test coverage: {N_TE - te_miss}/{N_TE} ({te_cov*100:.1f}%)")


# ── 6. Train feature 계산 ─────────────────────────────────────────────────────
print("[4] Computing train localization features ...")
tr_iso  = load_ids(os.path.join(ID_DIR, 'train_isoform_list.npy'))
N_TR    = len(tr_iso)
X_tr    = np.zeros((N_TR, N_FEAT), dtype=np.float32)
tr_miss = 0

for i, iso in enumerate(tr_iso):
    prot = train_prots.get(iso, '')
    feat = compute_features(prot) if prot else None
    if feat is not None:
        X_tr[i] = feat
    else:
        tr_miss += 1

tr_cov = (N_TR - tr_miss) / N_TR
print(f"   Train coverage: {N_TR - tr_miss}/{N_TR} ({tr_cov*100:.1f}%)")


# ── 7. Feature 통계 ───────────────────────────────────────────────────────────
covered = X_te[X_te.sum(axis=1) != 0]
feat_mean = covered.mean(axis=0)
feat_std  = covered.std(axis=0) + 1e-8

print("\n[5] Feature statistics (test covered):")
for j, name in enumerate(FEAT_NAMES):
    print(f"   {name:25s}  mean={feat_mean[j]:.4f}  std={feat_std[j]:.4f}")

# ── Sanity check: BISECT 핵심 유전자 확인 ────────────────────────────────────
# NDUFS8 isoform이 MTS score 기준으로 구분되는지 확인
print("\n[6] Sanity check — NDUFS8 isoform MTS scores:")
import subprocess

te_gene = load_ids(os.path.join(MODEL_DIR, 'my_gene_list_fixed.npy'))
ensg2sym = {}
with open(os.path.join(ID_DIR, 'ensembl_to_symbol.txt')) as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ensg2sym[p[0]] = p[4]

te_sym = [ensg2sym.get(g.split('.')[0], g.split('.')[0]) for g in te_gene]
ndufs8_idx = [i for i, s in enumerate(te_sym) if s == 'NDUFS8']

if ndufs8_idx:
    for idx in ndufs8_idx[:6]:
        iso  = te_iso[idx]
        feat = X_te[idx]
        print(f"   {iso:20s}  mts={feat[0]:.3f}  hydrophob={feat[1]:.3f}  pro_dens={feat[2]:.3f}  amphipathic={feat[7]:.3f}")
else:
    print("   NDUFS8 not in test set (expected in brain data)")


# ── 8. 저장 ───────────────────────────────────────────────────────────────────
test_out  = os.path.join(OUT_DIR, 'loc_features_test.npy')
train_out = os.path.join(OUT_DIR, 'loc_features_train.npy')
np.save(test_out,  X_te)
np.save(train_out, X_tr)

stats = {
    'n_features':    N_FEAT,
    'feature_names': FEAT_NAMES,
    'test_coverage':  round(te_cov, 4),
    'train_coverage': round(tr_cov, 4),
    'feat_mean':  feat_mean.tolist(),
    'feat_std':   feat_std.tolist(),
    'test_shape':  list(X_te.shape),
    'train_shape': list(X_tr.shape),
}
with open(os.path.join(OUT_DIR, 'loc_feature_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n  Saved: {test_out}  {X_te.shape}")
print(f"  Saved: {train_out}  {X_tr.shape}")
print("DONE")
