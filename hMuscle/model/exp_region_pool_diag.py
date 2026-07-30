#!/usr/bin/env python3
"""
exp_region_pool_diag.py — region-pool null 진단: 방법 탓인가 근본(magnitude≠alignment)인가
==========================================================================================
선행: exp_region_pool_direction.py — region≈global sign accuracy(motif 9/13). "centroid mismatch 탓?"
장애물: region-pool centroid는 학습 isoform에 pair/region 없어 정의 불가.
대안 진단: sign만 보던 걸 (a) **cosine 정렬**(F−N이 function centroid를 얼마나 향하나) + (b) **크기** ||F−N||로 분해.
  H_method(방법 탓): region이 global보다 cosine 정렬↑인데 n=13 threshold가 가렸다 → 정렬은 개선.
  H_fundamental(근본): region 크기↑(contextual spread)이나 cosine 정렬은 global과 동등/낮음 → magnitude≠alignment 확정.
        = [[reference-esm2-pca-axes-final]] axis-0 leakage와 동형(큰 차이≠함수방향).
"""
import os, csv, gzip, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3"); os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from collections import defaultdict
from Bio import Align
import warnings; warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR,ID_DIR,ANNOT_DIR='../data','../data/raw_data/data/id_lists','../data/raw_data/data/annotations'
CACHE='../../reports/exp_h_uniprot_eval/seq_cache'; BENCH='../../reports/exp_g_uniprot/uniprot_isoform_benchmark_v2.csv'
SCRATCH='/tmp/claude-1811/-home-welcome1-sw1686-DIFFUSE/010f9706-8801-4761-a27e-2255c2663dd1/scratchpad'
clean=lambda r:str(r).replace("b'","").replace("'","").replace('"','').replace(" ","")
t0=time.time(); log=lambda m:print(f"[{time.time()-t0:6.0f}s] {m}",flush=True)

# centroids (mean-pool L30 of train positives)
Xtr30=np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
tr_genes=[clean(g) for g in np.load(f'{ID_DIR}/train_gene_list.npy',allow_pickle=True)]
sym2id={}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz','rt') as f:
    next(f)
    for line in f:
        p=line.strip().split('\t')
        if len(p)>2:
            sym2id[p[2]]=p[1]
            if len(p)>4 and p[4]!='-':
                for syn in p[4].split('|'): sym2id.setdefault(syn,p[1])
tr_ids=[sym2id.get(g,g) for g in tr_genes]; tr_id_set=set(tr_ids)
go_tr=defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz','rt') as f:
    next(f)
    for line in f:
        p=line.strip().split('\t')
        if p[0]=='9606' and p[7]=='Function' and p[1] in tr_id_set: go_tr[p[2]].add(p[1])
mf_terms=[]
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p=line.strip().split('\t')
        if len(p)>=6: mf_terms.append(p[0])
go_to_idx={g:i for i,g in enumerate(mf_terms)}
tr_sym2idx=defaultdict(list)
for i,g in enumerate(tr_genes): tr_sym2idx[g].append(i)
CENT={}
for go in mf_terms:
    pos={g for g,gid in zip(tr_genes,tr_ids) if gid in go_tr[go]}
    idx=[i for s in pos for i in tr_sym2idx.get(s,[])]
    if len(idx)>=2: CENT[go]=Xtr30[idx].mean(0)
log(f"centroids {len(CENT)}")

Z=np.load(f'{SCRATCH}/perres_L1530.npz',allow_pickle=True); PR={k:Z[k] for k in Z.files}
def readfa(iso):
    for c in ([iso,iso[:-2]] if iso.endswith('-1') else [iso]):
        p=f'{CACHE}/{c}.fasta'
        if os.path.exists(p) and os.path.getsize(p)>50: return ''.join(l.strip() for l in open(p) if not l.startswith('>'))
    return None
rows=list(csv.DictReader(open(BENCH)))
aligner=Align.PairwiseAligner(); aligner.mode='global'; aligner.open_gap_score=-10; aligner.extend_gap_score=-0.5
aligner.substitution_matrix=Align.substitution_matrices.load("BLOSUM62")
remap={'GO:0004714':'GO:0004713','GO:0005007':'GO:0004713','GO:0004693':'GO:0004674','GO:0097553':'GO:0046982',
 'GO:0004197':'GO:0003824','GO:0006281':'GO:0003677','GO:0006977':'GO:0003700','GO:0008285':'GO:0019901',
 'GO:0005178':'GO:0048018','GO:0005158':'GO:0048018','GO:0000398':'GO:0003723','GO:0006357':'GO:0003700',
 'GO:0005200':'GO:0003779','GO:0007399':'GO:0005515','GO:0016079':'GO:0046982'}
def gidx(go):
    g=go.replace('GO_','GO:'); g=g if g in go_to_idx else remap.get(g,g)
    return g if g in CENT else None
def diffres(sa,sb):
    aln=aligner.align(sa,sb)[0]; ia,ib=aln.indices[0],aln.indices[1]
    dc=[c for c in range(len(ia)) if ia[c]<0 or ib[c]<0 or (ia[c]>=0 and ib[c]>=0 and sa[ia[c]]!=sb[ib[c]])]
    ra=sorted({ia[c] for c in dc if ia[c]>=0}); rb=sorted({ib[c] for c in dc if ib[c]>=0})
    return np.array(ra),np.array(rb)
def rpool(E,d,W):
    if len(d)==0: return E.mean(0)
    L=E.shape[0]; m=np.zeros(L,bool)
    for x in d: m[max(0,x-W):min(L,x+W+1)]=True
    return E[m].mean(0) if m.sum() else E.mean(0)
cosv=lambda u,v: float(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-9))

W=20
motif_g=[]; motif_r=[]  # per-pair (signed cos to centroid, magnitude ||F-N||)
for r in rows:
    sa,sb=readfa(r['iso_a']),readfa(r['iso_b'])
    if not(sa and sb) or r['iso_a']+'|30' not in PR: continue
    if abs(len(sa)-len(sb))>30 or r['direction'] not in ('A_only','B_only'): continue
    g=gidx(r['go_term'])
    if g is None: continue
    cent=CENT[g]; da,db=diffres(sa,sb); Ea,Eb=PR[r['iso_a']+'|30'],PR[r['iso_b']+'|30']
    gA,gB=Ea.mean(0),Eb.mean(0); rA,rB=rpool(Ea,da,W),rpool(Eb,db,W)
    if r['direction']=='A_only': gF,gN,rF,rN=gA,gB,rA,rB
    else: gF,gN,rF,rN=gB,gA,rB,rA
    motif_g.append((cosv(gF-gN,cent), float(np.linalg.norm(gF-gN))))
    motif_r.append((cosv(rF-rN,cent), float(np.linalg.norm(rF-rN))))
mg=np.array(motif_g); mr=np.array(motif_r)
print(f"\nmotif pairs n={len(mg)} (W={W})")
print(f"  GLOBAL : mean signed-cos(F-N, centroid)={mg[:,0].mean():+.4f}  | mean ||F-N||={mg[:,1].mean():.3f}  | sign+ {int((mg[:,0]>0).sum())}/{len(mg)}")
print(f"  REGION : mean signed-cos(F-N, centroid)={mr[:,0].mean():+.4f}  | mean ||F-N||={mr[:,1].mean():.3f}  | sign+ {int((mr[:,0]>0).sum())}/{len(mr)}")
print(f"\n  Δcos(region-global) = {mr[:,0].mean()-mg[:,0].mean():+.4f}   Δmag = {mr[:,1].mean()-mg[:,1].mean():+.3f}")
verdict=("H_method: region 정렬↑(방법/threshold이 가림) → region-pool 유망, 더 큰 n 필요" if mr[:,0].mean()-mg[:,0].mean()>0.05 else
         "H_fundamental: region 크기↑이나 정렬 개선 없음 → magnitude≠alignment 확정(axis-0 leakage 동형)" if mr[:,1].mean()>mg[:,1].mean() and mr[:,0].mean()<=mg[:,0].mean()+0.02 else
         "혼재/불명확")
print(f"\n=> {verdict}")
