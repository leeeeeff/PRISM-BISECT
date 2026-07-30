#!/usr/bin/env python3
"""
exp_within_pair_anchor.py — 앵커 병목 가설 검정 (옵션 A)
=======================================================
선행 진단([[finding-pooling-not-encoding-boundary]]): region-pool F−N이 전역 GO-centroid를 cos 0.08로 겨우 향함
→ 병목은 pooling 아닌 ANCHOR(전역 centroid=gene-family 방향, within-pair 함수대비 아님).
검정: within-pair 적합 앵커 = **domain-completeness 방향**(학습 within-gene 도메인쌍의 meanpool(complete)−meanpool(truncated) 평균).
이 앵커로 UniProt pair direction을 global-pool vs region-pool로 재검.
예측:
  H_anchor(앵커가 병목이었다): domain-completeness 앵커가 GO-centroid(cos 0.08)보다 domain-loss 쌍 direction 크게↑ +
    region-pool이 그 위에서 global 능가 → "올바른 앵커면 region-pool 유효".
  귀무: 앵커 바꿔도 region≈global → pooling이 진짜 병목 아님/더 깊은 문제.
  motif 쌍: domain 앵커는 기전 불일치라 chance 예상(정직 대조).
"""
import os, csv, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
import numpy as np
from collections import defaultdict
from Bio import Align
import warnings; warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR,ID_DIR='../data','../data/raw_data/data/id_lists'
CACHE='../../reports/exp_h_uniprot_eval/seq_cache'; BENCH='../../reports/exp_g_uniprot/uniprot_isoform_benchmark_v2.csv'
SCRATCH='/tmp/claude-1811/-home-welcome1-sw1686-DIFFUSE/010f9706-8801-4761-a27e-2255c2663dd1/scratchpad'
clean=lambda r:str(r).replace("b'","").replace("'","").replace('"','').replace(" ","")
t0=time.time(); log=lambda m:print(f"[{time.time()-t0:6.0f}s] {m}",flush=True)

# ---- build domain-completeness anchor from training within-gene domain pairs ----
log("build domain-completeness anchor")
Xtr30=np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
tr_genes=[clean(g) for g in np.load(f'{ID_DIR}/train_gene_list.npy',allow_pickle=True)]
tr_ndom=np.load('../results_isoform/features/domain_matrix_proper_train.npy').sum(1)
g2i=defaultdict(list)
for i,g in enumerate(tr_genes): g2i[g].append(i)
diffs=[]
for g,idx in g2i.items():
    if len(idx)<2: continue
    idx=np.array(idx); d=tr_ndom[idx]
    if d.std()<0.1: continue
    for a in idx:
        for b in idx:
            if tr_ndom[a]>tr_ndom[b]:
                diffs.append(Xtr30[a]-Xtr30[b])
diffs=np.array(diffs); anchor=diffs.mean(0); anchor/= (np.linalg.norm(anchor)+1e-9)
log(f"anchor from {len(diffs)} domain-diff train pairs")

# ---- UniProt pairs + per-residue L30 cache ----
Z=np.load(f'{SCRATCH}/perres_L1530.npz',allow_pickle=True); PR={k:Z[k] for k in Z.files}
def readfa(iso):
    for c in ([iso,iso[:-2]] if iso.endswith('-1') else [iso]):
        p=f'{CACHE}/{c}.fasta'
        if os.path.exists(p) and os.path.getsize(p)>50: return ''.join(l.strip() for l in open(p) if not l.startswith('>'))
    return None
aligner=Align.PairwiseAligner(); aligner.mode='global'; aligner.open_gap_score=-10; aligner.extend_gap_score=-0.5
aligner.substitution_matrix=Align.substitution_matrices.load("BLOSUM62")
def diffres(sa,sb):
    aln=aligner.align(sa,sb)[0]; ia,ib=aln.indices[0],aln.indices[1]
    dc=[c for c in range(len(ia)) if ia[c]<0 or ib[c]<0 or (ia[c]>=0 and ib[c]>=0 and sa[ia[c]]!=sb[ib[c]])]
    return np.array(sorted({ia[c] for c in dc if ia[c]>=0})), np.array(sorted({ib[c] for c in dc if ib[c]>=0}))
def rpool(E,d,W):
    if len(d)==0: return E.mean(0)
    L=E.shape[0]; m=np.zeros(L,bool)
    for x in d: m[max(0,x-W):min(L,x+W+1)]=True
    return E[m].mean(0) if m.sum() else E.mean(0)
cosv=lambda u,v: float(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-9))

rows=list(csv.DictReader(open(BENCH)))
W=20
recs=[]
for r in rows:
    sa,sb=readfa(r['iso_a']),readfa(r['iso_b'])
    if not(sa and sb) or r['iso_a']+'|30' not in PR or r['direction'] not in ('A_only','B_only'): continue
    Ea,Eb=PR[r['iso_a']+'|30'],PR[r['iso_b']+'|30']; da,db=diffres(sa,sb)
    gA,gB=Ea.mean(0),Eb.mean(0); rA,rB=rpool(Ea,da,W),rpool(Eb,db,W)
    if r['direction']=='A_only': gF,gN,rF,rN=gA,gB,rA,rB
    else: gF,gN,rF,rN=gB,gA,rB,rA
    recs.append({'gene':r['gene'],'lendiff':abs(len(sa)-len(sb)),
                 'g_cos':cosv(gF-gN,anchor),'r_cos':cosv(rF-rN,anchor),
                 'g_ok':int(np.dot(gF-gN,anchor)>0),'r_ok':int(np.dot(rF-rN,anchor)>0)})

def rep(sub,label):
    if not sub: print(f"  {label}: (none)"); return
    g=sum(r['g_ok'] for r in sub); rr=sum(r['r_ok'] for r in sub); n=len(sub)
    gc=np.mean([r['g_cos'] for r in sub]); rc=np.mean([r['r_cos'] for r in sub])
    print(f"  {label:28} n={n:2} | GLOBAL {g}/{n}={g/n:.3f} (cos{gc:+.3f}) | REGION {rr}/{n}={rr/n:.3f} (cos{rc:+.3f})")

print(f"\n=== domain-completeness anchor, direction accuracy (W={W}) ===")
rep([r for r in recs if r['lendiff']>=100],"domain-loss (|Δlen|≥100)")
rep([r for r in recs if 30<r['lendiff']<100],"mid (30-100)")
rep([r for r in recs if r['lendiff']<=30],"motif (|Δlen|≤30)")
rep(recs,"ALL")
dl=[r for r in recs if r['lendiff']>=100]
if dl:
    g=sum(r['g_ok'] for r in dl)/len(dl); rr=sum(r['r_ok'] for r in dl)/len(dl)
    print(f"\n=> domain-loss: GO-centroid 앵커(선행)=0.43 → domain 앵커 global={g:.3f} region={rr:.3f}")
    print("   "+("H_anchor 지지: domain 앵커가 direction 크게↑" if g>0.6 else "앵커 바꿔도 개선 미미")
          +f" | region−global(cos)={np.mean([r['r_cos'] for r in dl])-np.mean([r['g_cos'] for r in dl]):+.3f}")
