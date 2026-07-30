# -*- coding: utf-8 -*-
"""
extract_brain_aux_features.py
==============================
Brain zero-shot용 보조 feature 추출 (RNA + LOC, Stream C & D)

brain_full evaluation set (63,994 이소폼)에 대해:
  - RNA features (9-dim): brain_only TransDecoder GFF3 + brain_only FASTA
  - LOC features (8-dim): brain_only TransDecoder pep
  - ID 매핑: GTF transcript_name → ENST → TransDecoder

커버리지 (~46%) 미커버 이소폼 → 0-fill (use_bias=False로 ESM-2 fallback)

출력:
  features/rna/rna_features_brain_full.npy  (63994, 9)
  features/loc/loc_features_brain_full.npy  (63994, 8)
"""

import os, re, json, math
import numpy as np

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAIN_DIR = os.path.join(BASE_DIR, 'data/brain_esm2')
BRAIN_GTF = os.path.join(BRAIN_DIR, 'brain_only.gtf')
BRAIN_FA  = os.path.join(BRAIN_DIR, 'brain_only_transcripts.fa')
BRAIN_GFF3 = os.path.join(BRAIN_DIR, 'brain_only_transcripts.fa.transdecoder.gff3')
BRAIN_PEP  = os.path.join(BRAIN_DIR, 'brain_only_transcripts.fa.transdecoder.pep')
BRAIN_IDS  = os.path.join(BASE_DIR, 'data/brain_isoquant_esm2/full/brain_full_ids.npy')
RNA_OUT    = os.path.join(BASE_DIR, 'results_isoform/features/rna/rna_features_brain_full.npy')
LOC_OUT    = os.path.join(BASE_DIR, 'results_isoform/features/loc/loc_features_brain_full.npy')

# ── 공통 파라미터 (muscle 스크립트와 동일) ────────────────────────────────────
RARE_CODONS = {'TCG','CCG','ACG','GCG','CGT','CGC','CGA','CGG','ATA','CTA','CTT','CTC','GTA','GTT'}
KOZAK_POS   = {0:{'A':1,'G':0.5}, 3:{'A':0.5,'C':1}, 4:{'U':1}, 5:{'G':1}, 6:{'G':0.5}}
KOZAK_MAX   = 5.5
ARE_MOTIF   = 'AUUUA'

KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
      'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
      'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
KD_MAX = 4.5; KD_MIN = -4.5
HELIX_ANGLE = math.radians(100.0)


# ── RNA feature 함수들 ────────────────────────────────────────────────────────
def count_uorfs(seq):
    count = 0; first = True
    for i in range(len(seq)-2):
        if seq[i:i+3] == 'AUG':
            if first: first = False
            else: count += 1
    return count

def kozak_score(seq, cds_start):
    if cds_start < 3: return 0.0
    w = seq[cds_start-3:cds_start+4]
    if len(w) < 7: return 0.0
    s = sum(KOZAK_POS.get(i, {}).get(w[i], 0) for i in [0,3,4,5,6])
    return (2.0 * s / KOZAK_MAX) - 1.0

def are_density(seq):
    if not seq: return 0.0
    return seq.count(ARE_MOTIF) / (len(seq) / 100.0 + 1e-6)

def rare_codon_frac(seq):
    seq = seq.replace('U','T')
    total = rare = 0
    for i in range(0, len(seq)-2, 3):
        c = seq[i:i+3]
        if len(c)==3: total+=1; rare += (c in RARE_CODONS)
    return rare/total if total>0 else 0.0

def nmd_proxy(utr3): return 1.0/(1.0+math.exp(-(utr3-400)/200))

def compute_rna_feat(info, seq):
    u5, u3, cds, tx = info['utr5'], info['utr3'], info['cds'], info['tx_len']
    if tx == 0: return None
    s5 = seq[:u5]; sc = seq[u5:u5+cds]; s3 = seq[u5+cds:]
    return np.array([
        math.log1p(u5), math.log1p(u3), math.log1p(cds),
        min(cds/tx, 1.0) if tx else 0.0,
        count_uorfs(s5)/(u5/3+1),
        kozak_score(seq, u5),
        are_density(s3),
        rare_codon_frac(sc),
        nmd_proxy(u3),
    ], dtype=np.float32)


# ── LOC feature 함수들 ────────────────────────────────────────────────────────
def hydrophobic_moment(seq, start=0, length=18):
    sub = seq[start:start+length]
    if len(sub)<3: return 0.0
    ss, cs = 0.0, 0.0
    for i, aa in enumerate(sub):
        h=KD.get(aa,0.0); a=i*HELIX_ANGLE; ss+=h*math.sin(a); cs+=h*math.cos(a)
    return math.sqrt(ss**2+cs**2)/len(sub)/KD_MAX

def max_amphipathic(seq, search_len=60, helix_len=18):
    end = min(len(seq), search_len); best = 0.0
    for s in range(0, end-helix_len+1):
        mu = hydrophobic_moment(seq, s, helix_len)
        if mu>best: best=mu
    return best

def mean_kd(seq, s=0, e=20):
    sub=seq[s:e]; return sum(KD.get(aa,0) for aa in sub)/len(sub) if sub else 0.0

def mts_score(seq):
    n=seq[:30]
    if not n: return 0.0
    pos=sum(1 for aa in n if aa in ('R','K')); pos_s=min(pos/5.0,1.0)
    amphi=max_amphipathic(seq,40,min(18,len(n)))
    no_pro=max(0.0,1.0-n.count('P')/3.0)
    return pos_s*0.4+amphi*0.4+no_pro*0.2

def nls_score(seq):
    best=0.0
    for i in range(len(seq)-7):
        kr=sum(1 for aa in seq[i:i+8] if aa in ('K','R'))/8.0
        if kr>best: best=kr
    return best

def tm_count(seq, min_len=17):
    cnt=run=0
    for aa in seq:
        if KD.get(aa,0.0)>1.6: run+=1; cnt+=(run==min_len)
        else: run=0
    return cnt

def compute_loc_feat(seq):
    seq=seq.replace('*','').replace('X','').upper()
    if len(seq)<5: return None
    return np.array([
        mts_score(seq), (mean_kd(seq,0,20)-KD_MIN)/(KD_MAX-KD_MIN+1e-8),
        seq[:30].count('P')/min(30,len(seq)),
        max(0.0, mean_kd(seq[5:min(25,len(seq))])/KD_MAX)*0.7 + \
            sum(1 for aa in seq[min(25,len(seq)):min(35,len(seq))] if aa in 'AGST')/(len(seq[min(25,len(seq)):min(35,len(seq))])+1)*0.3,
        nls_score(seq), min(tm_count(seq)/5.0,1.0),
        max(0.0, mean_kd(seq[-25:])/KD_MAX), max_amphipathic(seq),
    ], dtype=np.float32)


# ── 1. GTF: transcript_name → ENST 매핑 ──────────────────────────────────────
print("[1] Parsing brain GTF (transcript_name → ENST) ...")
name2enst = {}
with open(BRAIN_GTF) as f:
    for line in f:
        if '\ttranscript\t' not in line: continue
        em = re.search(r'transcript_id "([^"]+)"', line)
        nm = re.search(r'transcript_name "([^"]+)"', line)
        if em and nm: name2enst[nm.group(1)] = em.group(1).split('.')[0]
print(f"   Mappings: {len(name2enst):,}")


# ── 2. GFF3: ENST → RNA structure ────────────────────────────────────────────
print("[2] Parsing brain TransDecoder GFF3 ...")
StructInfo = {}
with open(BRAIN_GFF3) as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        parts = line.split('\t')
        if len(parts) < 9: continue
        tid  = parts[0].split('.')[0]
        feat = parts[2]
        s, e = int(parts[3]), int(parts[4])
        length = e - s + 1
        if tid not in StructInfo:
            StructInfo[tid] = {'utr5':0,'utr3':0,'cds':0,'tx_len':0}
        if feat == 'mRNA':           StructInfo[tid]['tx_len'] = length
        elif feat == 'five_prime_UTR': StructInfo[tid]['utr5'] += length
        elif feat == 'three_prime_UTR': StructInfo[tid]['utr3'] += length
        elif feat == 'CDS':           StructInfo[tid]['cds']  += length
print(f"   GFF3 entries: {len(StructInfo):,}")


# ── 3. FASTA: ENST → RNA sequence ────────────────────────────────────────────
print("[3] Parsing brain FASTA ...")
fasta_seqs = {}
current_id = None; buf = []
with open(BRAIN_FA) as f:
    for line in f:
        line = line.rstrip()
        if line.startswith('>'):
            if current_id: fasta_seqs[current_id] = ''.join(buf)
            current_id = line[1:].split()[0].split('.')[0]; buf = []
        else: buf.append(line.upper().replace('T','U'))
    if current_id: fasta_seqs[current_id] = ''.join(buf)
print(f"   FASTA sequences: {len(fasta_seqs):,}")


# ── 4. PEP: ENST → protein sequence ─────────────────────────────────────────
print("[4] Parsing brain TransDecoder pep ...")
pep_seqs = {}
current_id = None; buf = []
with open(BRAIN_PEP) as f:
    for line in f:
        line = line.rstrip()
        if line.startswith('>'):
            if current_id and buf and current_id not in pep_seqs:
                pep_seqs[current_id] = ''.join(buf)
            current_id = line[1:].split()[0].split('.p')[0].split('.')[0]; buf = []
        else: buf.append(line.upper())
    if current_id and buf and current_id not in pep_seqs:
        pep_seqs[current_id] = ''.join(buf)
print(f"   PEP sequences: {len(pep_seqs):,}")


# ── 5. Brain IDs 로드 ─────────────────────────────────────────────────────────
brain_ids = np.load(BRAIN_IDS, allow_pickle=True)
brain_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in brain_ids]
N = len(brain_ids)
print(f"\n[5] Brain eval set: {N:,} isoforms")


# ── 6. Feature 계산 ───────────────────────────────────────────────────────────
X_rna = np.zeros((N, 9), dtype=np.float32)
X_loc = np.zeros((N, 8), dtype=np.float32)
rna_hit = loc_hit = 0

for i, bid in enumerate(brain_ids):
    # ID → ENST base
    if bid.startswith('ENST'):
        enst = bid.split('.')[0]
    elif bid.startswith('transcript'):
        enst = bid            # NIC/NNIC transcript IDs from IsoquAnt
    else:
        enst = name2enst.get(bid, '')

    if not enst:
        continue

    # RNA
    info = StructInfo.get(enst)
    seq  = fasta_seqs.get(enst, '')
    if info and seq:
        feat = compute_rna_feat(info, seq)
        if feat is not None:
            X_rna[i] = feat; rna_hit += 1

    # LOC
    prot = pep_seqs.get(enst, '')
    if prot:
        feat = compute_loc_feat(prot)
        if feat is not None:
            X_loc[i] = feat; loc_hit += 1

print(f"\n   RNA coverage: {rna_hit}/{N} ({rna_hit/N*100:.1f}%)")
print(f"   LOC coverage: {loc_hit}/{N} ({loc_hit/N*100:.1f}%)")

# ── 7. 저장 ───────────────────────────────────────────────────────────────────
np.save(RNA_OUT, X_rna)
np.save(LOC_OUT, X_loc)

stats = {'n_isoforms': N, 'rna_coverage': round(rna_hit/N,4),
         'loc_coverage': round(loc_hit/N,4),
         'rna_shape': list(X_rna.shape), 'loc_shape': list(X_loc.shape)}
with open(os.path.join(BASE_DIR,'results_isoform/features/rna/brain_feature_stats.json'),'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n  Saved: {RNA_OUT}")
print(f"  Saved: {LOC_OUT}")
print("DONE")
