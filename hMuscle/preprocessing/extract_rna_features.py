# -*- coding: utf-8 -*-
"""
extract_rna_features.py
========================
Stream C 특징 추출: RNA 안정성 + 번역 효율 (9-dim per isoform)

Features (인덱스 순서 고정):
  0  log1p_utr5_len    : log1p(5'UTR length in nt)
  1  log1p_utr3_len    : log1p(3'UTR length in nt)
  2  log1p_cds_len     : log1p(CDS length in nt)
  3  cds_frac          : CDS length / transcript length
  4  uorf_count_norm   : ATG count in 5'UTR (excl. start) / (utr5_len/3 + 1)
  5  kozak_score       : Kozak consensus score around start ATG  [-1, 1]
  6  are_density       : AUUUA motif count / (utr3_len + 1)
  7  rare_codon_frac   : fraction of rare codons in CDS  (CAI proxy)
  8  nmd_proxy         : sigmoid((utr3_len - 400) / 200)  (long 3'UTR → NMD risk)

데이터 소스:
  - hMuscle/data/transcripts.fasta.transdecoder.gff3  (UTR/CDS positions)
  - hMuscle/data/transcripts.fasta                    (RNA sequences)
  - hMuscle/results_isoform/features/splicing/nm_to_enst_mapping.tsv

출력:
  hMuscle/results_isoform/features/rna/
    rna_features_test.npy    (36748 × 9, float32)
    rna_features_train.npy   (31668 × 9, float32)
    rna_feature_stats.json   (coverage + normalization stats)

실행:
  cd /home/welcome1/sw1686/DIFFUSE
  conda activate isoform_env
  python hMuscle/preprocessing/extract_rna_features.py
"""

import os, re, json, math
import numpy as np
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GFF3_PATH  = os.path.join(BASE_DIR, 'data/transcripts.fasta.transdecoder.gff3')
FASTA_PATH = os.path.join(BASE_DIR, 'data/transcripts.fasta')
NM2ENST_PATH = os.path.join(BASE_DIR, 'results_isoform/features/splicing/nm_to_enst_mapping.tsv')
MODEL_DIR    = os.path.join(BASE_DIR, 'model')
ID_DIR       = os.path.join(BASE_DIR, 'data/raw_data/data/id_lists')
OUT_DIR      = os.path.join(BASE_DIR, 'results_isoform/features/rna')
os.makedirs(OUT_DIR, exist_ok=True)

N_FEAT = 9
FEAT_NAMES = [
    'log1p_utr5_len', 'log1p_utr3_len', 'log1p_cds_len',
    'cds_frac', 'uorf_count_norm', 'kozak_score',
    'are_density', 'rare_codon_frac', 'nmd_proxy',
]

# ── 코돈 빈도 테이블 (인간 희귀 코돈: RSCU < 0.5) ──────────────────────────
RARE_CODONS = {
    'TCG', 'CCG', 'ACG', 'GCG',  # CpG-suppressed codons
    'CGT', 'CGC', 'CGA', 'CGG',  # rare Arg
    'ATA',                         # rare Ile
    'CTA', 'CTT', 'CTC',          # less common Leu
    'GTA', 'GTT',                  # rare Val
}

# Kozak consensus: [A/G]CCACC AUG G (RNA format — sequences stored as U)
# positions relative to window[0] = -3 of AUG
KOZAK_POS   = {0: {'A': 1, 'G': 0.5},   # -3 : A/G preferred
               3: {'A': 0.5, 'C': 1},    #  0 : A of AUG
               4: {'U': 1},              # +1 : U of AUG (RNA!)
               5: {'G': 1},              # +2 : G of AUG
               6: {'G': 0.5}}            # +3 : G preferred
KOZAK_MAX   = 1 + 0.5 + 0.5 + 1 + 1 + 1 + 0.5   # = 5.5

ARE_MOTIF = 'AUUUA'


def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


# ── 1. TransDecoder GFF3 파싱 → isoform당 UTR/CDS 정보 ──────────────────────
print("[1] Parsing TransDecoder GFF3 ...")

StructInfo = {}  # base_id → {utr5, cds, utr3, tx_len}

with open(GFF3_PATH) as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 9:
            continue
        tid  = parts[0]
        feat = parts[2]
        s, e = int(parts[3]), int(parts[4])
        length = e - s + 1
        base = tid.split('.')[0]  # ENST00000286479.4 → ENST00000286479

        if base not in StructInfo:
            StructInfo[base] = {'utr5': 0, 'utr3': 0, 'cds': 0, 'tx_len': 0}

        if feat == 'mRNA':
            StructInfo[base]['tx_len'] = length
        elif feat == 'five_prime_UTR':
            StructInfo[base]['utr5'] += length
        elif feat == 'three_prime_UTR':
            StructInfo[base]['utr3'] += length
        elif feat == 'CDS':
            StructInfo[base]['cds'] += length

print(f"   GFF3 entries: {len(StructInfo):,}")


# ── 2. transcripts.fasta 파싱 → base_id → RNA sequence ──────────────────────
print("[2] Parsing transcripts.fasta ...")

fasta_seqs = {}
current_id = None
buf = []
with open(FASTA_PATH) as f:
    for line in f:
        line = line.rstrip()
        if line.startswith('>'):
            if current_id is not None:
                fasta_seqs[current_id] = ''.join(buf)
            current_id = line[1:].split()[0].split('.')[0]
            buf = []
        else:
            buf.append(line.upper().replace('T', 'U'))
    if current_id is not None:
        fasta_seqs[current_id] = ''.join(buf)

print(f"   FASTA sequences: {len(fasta_seqs):,}")


# ── 3. NM→ENST 매핑 ──────────────────────────────────────────────────────────
print("[3] Loading NM→ENST mapping ...")
nm2enst = {}
with open(NM2ENST_PATH) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2 and p[1]:
            nm2enst[p[0]] = p[1]


# ── 4. 핵심 feature 계산 함수들 ──────────────────────────────────────────────
def count_uorfs(seq_utr5):
    """5'UTR에서 ATG 수 (첫 ATG 제외) = uORF 후보 수"""
    if not seq_utr5:
        return 0
    count = 0
    pos = 0
    first = True
    while pos < len(seq_utr5) - 2:
        if seq_utr5[pos:pos+3] == 'AUG':
            if first:
                first = False
            else:
                count += 1
        pos += 1
    return count


def kozak_score(seq_tx, cds_start):
    """
    CDS start (0-based on transcript) 주변의 Kozak score.
    Returns: [-1, 1] (음수 = anti-Kozak, 양수 = strong Kozak)
    """
    if cds_start is None or cds_start < 3:
        return 0.0
    window_start = cds_start - 3
    window_end   = cds_start + 4
    if window_end > len(seq_tx):
        return 0.0
    window = seq_tx[window_start:window_end]
    if len(window) < 7:
        return 0.0

    score = 0.0
    score += KOZAK_POS[0].get(window[0], 0)   # pos -3
    score += KOZAK_POS[3].get(window[3], 0)   # pos 0 (A of AUG)
    score += KOZAK_POS[4].get(window[4], 0)   # pos +1
    score += KOZAK_POS[5].get(window[5], 0)   # pos +2
    score += KOZAK_POS[6].get(window[6], 0)   # pos +3
    return (2.0 * score / KOZAK_MAX) - 1.0    # normalize to [-1, 1]


def are_density(seq_utr3):
    """3'UTR에서 AUUUA 밀도 (per 100 nt)"""
    if not seq_utr3:
        return 0.0
    count = seq_utr3.count(ARE_MOTIF)
    return count / (len(seq_utr3) / 100.0 + 1e-6)


def rare_codon_frac(seq_cds):
    """CDS에서 희귀 코돈 비율 (CAI proxy)"""
    if not seq_cds or len(seq_cds) < 3:
        return 0.0
    total = 0
    rare  = 0
    seq_cds = seq_cds.replace('U', 'T')
    for i in range(0, len(seq_cds) - 2, 3):
        codon = seq_cds[i:i+3]
        if len(codon) == 3:
            total += 1
            if codon in RARE_CODONS:
                rare += 1
    return rare / total if total > 0 else 0.0


def nmd_proxy(utr3_len):
    """
    NMD proxy: sigmoid((utr3_len - 400) / 200)
    3'UTR > 400nt → NMD risk 증가 (long 3'UTR rule)
    """
    return 1.0 / (1.0 + math.exp(-(utr3_len - 400) / 200))


# ── 5. 단일 isoform에 대한 9-dim feature 계산 ────────────────────────────────
def compute_features(base_id):
    """
    base_id: ENST00000xxxx 또는 BambuTxNNN (버전 없이)
    Returns: np.array shape (9,) float32, or zeros if missing
    """
    info = StructInfo.get(base_id)
    seq  = fasta_seqs.get(base_id, '')

    if info is None or info['tx_len'] == 0:
        return None  # 미커버 → 0-fill 처리

    utr5_len = info['utr5']
    utr3_len = info['utr3']
    cds_len  = info['cds']
    tx_len   = info['tx_len']

    # Subsequence extraction (transcript-level, 1-based GFF3 → 0-based Python)
    seq_utr5 = seq[:utr5_len]                           if seq else ''
    seq_cds  = seq[utr5_len:utr5_len + cds_len]        if seq else ''
    seq_utr3 = seq[utr5_len + cds_len:]                if seq else ''

    cds_start = utr5_len  # 0-based

    cds_frac_val = min(cds_len / tx_len, 1.0) if tx_len > 0 else 0.0

    feat = np.array([
        math.log1p(utr5_len),
        math.log1p(utr3_len),
        math.log1p(cds_len),
        cds_frac_val,
        count_uorfs(seq_utr5) / (utr5_len / 3 + 1),
        kozak_score(seq, cds_start),
        are_density(seq_utr3),
        rare_codon_frac(seq_cds),
        nmd_proxy(utr3_len),
    ], dtype=np.float32)

    return feat


# ── 6. Test set 계산 ──────────────────────────────────────────────────────────
print("[4] Computing test RNA features ...")
te_iso  = load_ids(os.path.join(MODEL_DIR, 'my_isoform_list_fixed.npy'))
N_TE    = len(te_iso)
X_te    = np.zeros((N_TE, N_FEAT), dtype=np.float32)
te_miss = 0

for i, iso in enumerate(te_iso):
    base = iso.split('.')[0]
    feat = compute_features(base)
    if feat is not None:
        X_te[i] = feat
    else:
        te_miss += 1

te_cov = (N_TE - te_miss) / N_TE
print(f"   Test coverage: {N_TE - te_miss}/{N_TE} ({te_cov*100:.1f}%)")


# ── 7. Train set 계산 ─────────────────────────────────────────────────────────
print("[5] Computing train RNA features ...")
tr_iso  = load_ids(os.path.join(ID_DIR, 'train_isoform_list.npy'))
N_TR    = len(tr_iso)
X_tr    = np.zeros((N_TR, N_FEAT), dtype=np.float32)
tr_miss = 0

for i, iso in enumerate(tr_iso):
    base_id = nm2enst.get(iso, '')   # NM_ → ENST base
    if not base_id:
        tr_miss += 1
        continue
    feat = compute_features(base_id)
    if feat is not None:
        X_tr[i] = feat
    else:
        tr_miss += 1

tr_cov = (N_TR - tr_miss) / N_TR
print(f"   Train coverage: {N_TR - tr_miss}/{N_TR} ({tr_cov*100:.1f}%)")


# ── 8. Global normalization stats (mean/std on covered rows only) ─────────────
covered_rows = X_te[X_te.sum(axis=1) != 0]
feat_mean = covered_rows.mean(axis=0)
feat_std  = covered_rows.std(axis=0) + 1e-8

print("\n[6] Feature statistics (test covered rows):")
for j, name in enumerate(FEAT_NAMES):
    print(f"   {name:25s}  mean={feat_mean[j]:.4f}  std={feat_std[j]:.4f}")


# ── 9. 저장 ───────────────────────────────────────────────────────────────────
test_out  = os.path.join(OUT_DIR, 'rna_features_test.npy')
train_out = os.path.join(OUT_DIR, 'rna_features_train.npy')
np.save(test_out,  X_te)
np.save(train_out, X_tr)

stats = {
    'n_features': N_FEAT,
    'feature_names': FEAT_NAMES,
    'test_coverage':  round(te_cov, 4),
    'train_coverage': round(tr_cov, 4),
    'feat_mean':  feat_mean.tolist(),
    'feat_std':   feat_std.tolist(),
    'test_shape':  list(X_te.shape),
    'train_shape': list(X_tr.shape),
}
with open(os.path.join(OUT_DIR, 'rna_feature_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n  Saved: {test_out}  {X_te.shape}")
print(f"  Saved: {train_out}  {X_tr.shape}")
print("DONE")
