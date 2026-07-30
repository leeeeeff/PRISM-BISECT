#!/usr/bin/env python3
"""
exp_contextual_spread.py — pooling 경계 vs 인코딩 경계 판별 (옵션 A, 순환성 없음)
================================================================================
세션 발견: within-gene DR 0.630 = δ_layer 표현 경계(capacity·supervision 둘 다 실패, [[approach-bracketing-nulls]]).
표현 = ESM-2 per-residue 인코딩 ∘ mean-pooling. 어느 연산이 경계인가?
논문 주장 "mean-pooling이 국소 splice 신호를 희석"을 직접 검증(head·재훈련·function-direction 불필요).

측정: 두 isoform 정렬 → **공유 backbone** residue의 per-residue L30 임베딩 변화
      δ_p = ||e_A(aligned p) − e_B(aligned p)|| 를 splice junction 거리별로 프로파일.
      (differential region은 서열로만 정의 = GO 무관 → 라벨 누출 0.)

사전등록 예측 (predict-before-look):
  H_pooling(경계=pooling): δ_p가 junction 근처만 크고 ~W 내 감쇠 → 국소신호를 mean-pool이 1/L 희석 → region-pool로 복원가능.
  H_encoding(경계=인코딩): δ_p가 어디서나 작음(junction 포함) → ESM-2가 motif 변화 미인코딩 → tier(iii) 근본, 복원 불가.
  H_spread(반증): δ_p가 전역 확산 → mean-pool이 오히려 신호 보존 → "pooling이 희석" 서사 반증(DR 낮음과 모순).
  positive control(domain-loss/truncation 쌍): δ_p가 광범위하게 큼 → 측정이 실제 구조변화 감지 확인.
  motif/regulatory 쌍(저 length-diff)이 본 실험 대상.
"""
import os, csv, glob, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3"); os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from Bio import Align
import warnings; warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
CACHE = '../../reports/exp_h_uniprot_eval/seq_cache'
BENCH = '../../reports/exp_g_uniprot/uniprot_isoform_benchmark_v2.csv'
OUT = '../../reports/v20b_pca_interp/within_family'
SCRATCH = '/tmp/claude-1811/-home-welcome1-sw1686-DIFFUSE/010f9706-8801-4761-a27e-2255c2663dd1/scratchpad'
MAXLEN = 1022
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

def readfa(iso):
    """cache stores canonical '-1' under base accession."""
    for cand in ([iso, iso[:-2]] if iso.endswith('-1') else [iso]):
        p = f'{CACHE}/{cand}.fasta'
        if os.path.exists(p) and os.path.getsize(p) > 50:
            return ''.join(l.strip() for l in open(p) if not l.startswith('>'))
    return None

rows = list(csv.DictReader(open(BENCH)))
pairs = []
for r in rows:
    sa, sb = readfa(r['iso_a']), readfa(r['iso_b'])
    if sa and sb and len(sa) <= MAXLEN and len(sb) <= MAXLEN:
        pairs.append((r['gene'], r['iso_a'], r['iso_b'], sa, sb, r['go_term'], r['direction']))
log(f"usable pairs: {len(pairs)}/{len(rows)}")

# ---- per-residue ESM-2 L30 embeddings (cache to scratch) ----
seqs = {}
for _,a,b,sa,sb,_,_ in pairs: seqs[a]=sa; seqs[b]=sb
emb_cache = f'{SCRATCH}/perres_L30.npz'
if os.path.exists(emb_cache):
    log("load cached per-residue embeddings")
    Z = np.load(emb_cache, allow_pickle=True); PR = {k: Z[k] for k in Z.files}
else:
    import esm
    log("load ESM-2 t30_150M")
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(dev).eval(); bc = alphabet.get_batch_converter()
    PR = {}
    ids = list(seqs)
    for i in range(0, len(ids), 8):
        bt = ids[i:i+8]
        data = [(k, seqs[k]) for k in bt]
        _,_,toks = bc(data); toks = toks.to(dev)
        with torch.no_grad():
            out = model(toks, repr_layers=[30], return_contacts=False)
        rep = out['representations'][30]
        for k, key in enumerate(bt):
            L = len(seqs[key]); PR[key] = rep[k,1:L+1].cpu().float().numpy()  # (L,640)
        log(f"  embedded {min(i+8,len(ids))}/{len(ids)}")
    np.savez(emb_cache, **PR); log(f"cached {emb_cache}")

# ---- align + per-shared-residue delta vs junction distance ----
aligner = Align.PairwiseAligner()
aligner.mode = 'global'; aligner.open_gap_score = -10; aligner.extend_gap_score = -0.5
aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")

BINS = [0,5,10,20,40,80,160,100000]  # junction-distance bins (residues)
def analyze(sa, sb, ea, eb):
    aln = aligner.align(sa, sb)[0]
    # aligned index arrays
    ia = aln.indices[0]; ib = aln.indices[1]  # -1 for gap
    # junction positions = alignment columns where a gap starts/ends or mismatch
    gap_cols = [c for c in range(len(ia)) if ia[c]<0 or ib[c]<0]
    mism_cols = [c for c in range(len(ia)) if ia[c]>=0 and ib[c]>=0 and sa[ia[c]]!=sb[ib[c]]]
    diff_cols = sorted(set(gap_cols)|set(mism_cols))
    if not diff_cols: return None
    # shared (aligned, matched) columns
    shared = [c for c in range(len(ia)) if ia[c]>=0 and ib[c]>=0]
    if len(shared) < 5: return None
    # per shared residue: delta embedding norm + distance to nearest diff column
    dc = np.array(diff_cols)
    prof = {}  # bin -> list of deltas
    e_shared_a=[]; e_shared_b=[]
    for c in shared:
        d = ea[ia[c]] - eb[ib[c]]
        dn = float(np.linalg.norm(d))
        dist = int(np.min(np.abs(dc - c)))
        for j in range(len(BINS)-1):
            if BINS[j] <= dist < BINS[j+1]:
                prof.setdefault(j, []).append(dn); break
        e_shared_a.append(ea[ia[c]]); e_shared_b.append(eb[ib[c]])
    # global shared-backbone perturbation (cosine dist of mean-pooled SHARED residues)
    ma = np.mean(e_shared_a,0); mb = np.mean(e_shared_b,0)
    global_shared = float(1 - np.dot(ma,mb)/(np.linalg.norm(ma)*np.linalg.norm(mb)+1e-9))
    # global full-protein cosine dist (what PRISM sees)
    fa = ea.mean(0); fb = eb.mean(0)
    global_full = float(1 - np.dot(fa,fb)/(np.linalg.norm(fa)*np.linalg.norm(fb)+1e-9))
    return prof, global_shared, global_full, len(shared), len(diff_cols)

recs = []
for gene,a,b,sa,sb,go,dr in pairs:
    if a not in PR or b not in PR: continue
    res = analyze(sa, sb, PR[a], PR[b])
    if res is None: continue
    prof, gsh, gfull, nsh, ndiff = res
    lendiff = abs(len(sa)-len(sb))
    recs.append({'gene':gene,'a':a,'b':b,'go':go,'lendiff':lendiff,'nshared':nsh,'ndiff':ndiff,
                 'global_shared':gsh,'global_full':gfull,
                 'binmeans':{str(BINS[j])+'-'+str(BINS[j+1]): (float(np.mean(prof[j])) if j in prof else None) for j in range(len(BINS)-1)}})
log(f"analyzed {len(recs)} pairs")

# ---- aggregate: motif (small lendiff) vs domain-loss (large lendiff) ----
def agg(sub, label):
    print(f"\n=== {label} (n={len(sub)}) : mean δ_p by junction distance ===")
    for j in range(len(BINS)-1):
        key=str(BINS[j])+'-'+str(BINS[j+1])
        vals=[r['binmeans'][key] for r in sub if r['binmeans'][key] is not None]
        print(f"  dist [{BINS[j]:>4},{BINS[j+1] if BINS[j+1]<1000 else 'inf':>4}) : δ_p={np.mean(vals):.4f}  (n_res pairs {len(vals)})" if vals else f"  dist [{BINS[j]},..): (none)")
    print(f"  global_shared(backbone meanpool cos-dist)={np.mean([r['global_shared'] for r in sub]):.4f} | global_full={np.mean([r['global_full'] for r in sub]):.4f}")

motif = [r for r in recs if r['lendiff'] <= 30]
domloss = [r for r in recs if r['lendiff'] >= 100]
mid = [r for r in recs if 30 < r['lendiff'] < 100]
agg(domloss, "POSITIVE CONTROL domain-loss/truncation (|Δlen|≥100)")
agg(motif, "TEST motif/regulatory (|Δlen|≤30)")
agg(mid, "mid (30<|Δlen|<100)")

# verdict on motif pairs: near-junction δ vs far δ
def near_far(sub):
    near=[r['binmeans']['0-5'] for r in sub if r['binmeans']['0-5'] is not None]
    far =[r['binmeans']['160-100000'] for r in sub if r['binmeans']['160-100000'] is not None]
    return (np.mean(near) if near else None, np.mean(far) if far else None)
mn,mf = near_far(motif)
print(f"\n=> MOTIF pairs: near-junction δ_p={mn} vs far δ_p={mf}")
if mn and mf:
    ratio = mn/(mf+1e-9)
    verdict = ("H_pooling: 국소 신호 존재+감쇠 → mean-pool 희석(region-pool 복원가능)" if ratio>1.5 and mn>0.5 else
               "H_encoding: junction조차 δ 작음 → ESM-2 motif 미인코딩(tier iii 근본)" if mn<0.3 else
               "H_spread/mixed: 전역 확산 or 약한 국소 → 재검")
    print(f"   near/far ratio={ratio:.2f} → {verdict}")
json.dump({'n_pairs':len(recs),'bins':BINS,'records':recs}, open(f'{OUT}/exp_contextual_spread.json','w'), indent=2)
log(f"[saved] {OUT}/exp_contextual_spread.json")
