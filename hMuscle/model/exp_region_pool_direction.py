#!/usr/bin/env python3
"""
exp_region_pool_direction.py — payoff: region-pooling이 functional direction을 복원하나 (옵션 A 2단계)
====================================================================================================
선행: [[finding-pooling-not-encoding-boundary]] — motif 신호는 per-residue에 있으나 junction 국소(17.6× 감쇠),
전체 mean-pool이 희석. 그것은 필요조건(신호 존재). 충분조건 검증: region-pool이 실제 direction을 살리나?

방법(순환·OOD-head·재훈련 無 — paper의 centroid-similarity 앵커 사용, DR-AUC 0.638 baseline):
  GO term t의 function direction = 학습 positive isoform의 mean-pool φ_L30 centroid_t.
  pair(F,N): direction 정답 = (repr(F) − repr(N)) · centroid_t > 0.
  repr ∈ {global mean-pool, region-pool(junction±W)}. differential region은 서열로만 정의(라벨 무관).
  UniProt 48쌍(ground-truth direction), 특히 저-gap motif 쌍(현재 near-chance)이 대상.
사전등록 예측:
  H_pooling 참이면: region-pool direction accuracy >> global on motif pairs(신호 국소→region이 살림).
  귀무: region ≈ global(신호가 direction으로 안 감/전역이 이미 충분).
  W 스윕(10/20/40)으로 decay length와 최적 window 확인.
"""
import os, csv, gzip, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3"); os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np, torch
from collections import defaultdict
from Bio import Align
import warnings; warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, ID_DIR, ANNOT_DIR = '../data','../data/raw_data/data/id_lists','../data/raw_data/data/annotations'
CACHE='../../reports/exp_h_uniprot_eval/seq_cache'; BENCH='../../reports/exp_g_uniprot/uniprot_isoform_benchmark_v2.csv'
OUT='../../reports/v20b_pca_interp/within_family'
SCRATCH='/tmp/claude-1811/-home-welcome1-sw1686-DIFFUSE/010f9706-8801-4761-a27e-2255c2663dd1/scratchpad'
MAXLEN=1022; WSWEEP=[10,20,40]
clean=lambda r:str(r).replace("b'","").replace("'","").replace('"','').replace(" ","")
t0=time.time(); log=lambda m:print(f"[{time.time()-t0:6.0f}s] {m}",flush=True)

# ---- GO labels + training centroids (mean-pool L30) ----
log("[1] GO labels + train centroids")
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
        if p[0]!='9606' or p[7]!='Function': continue
        if p[1] in tr_id_set: go_tr[p[2]].add(p[1])
mf_terms=[]
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p=line.strip().split('\t')
        if len(p)>=6: mf_terms.append(p[0])
go_to_idx={g:i for i,g in enumerate(mf_terms)}
tr_sym2idx=defaultdict(list)
for i,g in enumerate(tr_genes): tr_sym2idx[g].append(i)
def Ytr(go):
    pos={g for g,gid in zip(tr_genes,tr_ids) if gid in go_tr[go]}
    y=np.zeros(len(tr_genes),bool)
    for s in pos:
        for idx in tr_sym2idx.get(s,[]): y[idx]=True
    return y
CENT={}
for go in mf_terms:
    y=Ytr(go)
    if y.sum()>=2: CENT[go]=Xtr30[y].mean(0)
log(f"   centroids for {len(CENT)}/{len(mf_terms)} MF terms")

# ---- per-residue L15+L30 for UniProt isoforms ----
def readfa(iso):
    for c in ([iso,iso[:-2]] if iso.endswith('-1') else [iso]):
        p=f'{CACHE}/{c}.fasta'
        if os.path.exists(p) and os.path.getsize(p)>50:
            return ''.join(l.strip() for l in open(p) if not l.startswith('>'))
    return None
rows=list(csv.DictReader(open(BENCH)))
pairs=[]
for r in rows:
    sa,sb=readfa(r['iso_a']),readfa(r['iso_b'])
    if sa and sb and len(sa)<=MAXLEN and len(sb)<=MAXLEN:
        pairs.append((r['gene'],r['iso_a'],r['iso_b'],sa,sb,r['go_term'],r['direction']))
seqs={}
for _,a,b,sa,sb,_,_ in pairs: seqs[a]=sa; seqs[b]=sb
cache=f'{SCRATCH}/perres_L1530.npz'
if os.path.exists(cache):
    log("load cached per-residue L15+L30"); Z=np.load(cache,allow_pickle=True); PR={k:Z[k] for k in Z.files}
else:
    import esm
    log("load ESM-2 t30_150M"); model,alphabet=esm.pretrained.esm2_t30_150M_UR50D()
    dev='cuda' if torch.cuda.is_available() else 'cpu'; model=model.to(dev).eval(); bc=alphabet.get_batch_converter()
    PR={}; ids=list(seqs)
    for i in range(0,len(ids),8):
        bt=ids[i:i+8]; _,_,toks=bc([(k,seqs[k]) for k in bt]); toks=toks.to(dev)
        with torch.no_grad(): out=model(toks,repr_layers=[15,30],return_contacts=False)
        r15,r30=out['representations'][15],out['representations'][30]
        for k,key in enumerate(bt):
            L=len(seqs[key])
            PR[key+'|15']=r15[k,1:L+1].cpu().float().numpy(); PR[key+'|30']=r30[k,1:L+1].cpu().float().numpy()
        log(f"  embedded {min(i+8,len(ids))}/{len(ids)}")
    np.savez(cache,**PR); log("cached")

# ---- align + differential window + direction scoring ----
aligner=Align.PairwiseAligner(); aligner.mode='global'; aligner.open_gap_score=-10; aligner.extend_gap_score=-0.5
aligner.substitution_matrix=Align.substitution_matrices.load("BLOSUM62")
remap={'GO:0004714':'GO:0004713','GO:0005007':'GO:0004713','GO:0004693':'GO:0004674','GO:0097553':'GO:0046982',
 'GO:0004197':'GO:0003824','GO:0006281':'GO:0003677','GO:0006977':'GO:0003700','GO:0008285':'GO:0019901',
 'GO:0005178':'GO:0048018','GO:0005158':'GO:0048018','GO:0000398':'GO:0003723','GO:0006357':'GO:0003700',
 'GO:0005200':'GO:0003779','GO:0007399':'GO:0005515','GO:0016079':'GO:0046982'}
def diff_positions(sa,sb):
    aln=aligner.align(sa,sb)[0]; ia,ib=aln.indices[0],aln.indices[1]
    dcols=[c for c in range(len(ia)) if ia[c]<0 or ib[c]<0 or (ia[c]>=0 and ib[c]>=0 and sa[ia[c]]!=sb[ib[c]])]
    # map diff alignment-cols to residue indices in each isoform (nearest present)
    ra=set(); rb=set()
    for c in dcols:
        if ia[c]>=0: ra.add(ia[c])
        if ib[c]>=0: rb.add(ib[c])
    return np.array(sorted(ra)), np.array(sorted(rb))
def regionpool(E, diffres, W):
    if len(diffres)==0: return E.mean(0)
    L=E.shape[0]; mask=np.zeros(L,bool)
    for d in diffres: mask[max(0,d-W):min(L,d+W+1)]=True
    return E[mask].mean(0) if mask.sum()>0 else E.mean(0)

def go_idx(go):
    g=go.replace('GO_','GO:'); g=g if g in go_to_idx else remap.get(g,g)
    return go_to_idx.get(g) if g in CENT else None

results={}
for W in WSWEEP+['global']:
    recs=[]
    for gene,a,b,sa,sb,go,dr in pairs:
        gi=go_idx(go)
        if gi is None or a+'|30' not in PR: continue
        cent=CENT[mf_terms[gi]]
        da,db=diff_positions(sa,sb)
        if W=='global':
            ra=PR[a+'|30'].mean(0); rb=PR[b+'|30'].mean(0)
        else:
            ra=regionpool(PR[a+'|30'],da,W); rb=regionpool(PR[b+'|30'],db,W)
        # F=functional isoform: A_only→a, B_only→b, both→skip direction
        if dr=='A_only': F,N=ra,rb
        elif dr=='B_only': F,N=rb,ra
        else: continue
        score=float(np.dot(F-N,cent))  # >0 = F more aligned to function = correct
        recs.append({'gene':gene,'go':go,'lendiff':abs(len(sa)-len(sb)),'correct':int(score>0),'score':score})
    motif=[r for r in recs if r['lendiff']<=30]; dom=[r for r in recs if r['lendiff']>=100]
    acc=lambda s:(sum(r['correct'] for r in s),len(s))
    results[str(W)]={'all':acc(recs),'motif':acc(motif),'domloss':acc(dom)}
    ca,na=acc(recs); cm,nm=acc(motif); cd,nd=acc(dom)
    print(f"W={str(W):>6} | all {ca}/{na}={ca/max(na,1):.3f} | motif {cm}/{nm}={cm/max(nm,1):.3f} | domloss {cd}/{nd}={cd/max(nd,1):.3f}")

g_m=results['global']['motif']; best=max(WSWEEP,key=lambda W:results[str(W)]['motif'][0]/max(results[str(W)]['motif'][1],1))
bm=results[str(best)]['motif']
print(f"\n=> MOTIF direction: global {g_m[0]}/{g_m[1]}={g_m[0]/max(g_m[1],1):.3f} vs region(W={best}) {bm[0]}/{bm[1]}={bm[0]/max(bm[1],1):.3f}")
print("   "+("PAYOFF: region-pool이 motif direction 복원(H_pooling 충분조건 성립)" if bm[0]/max(bm[1],1)-g_m[0]/max(g_m[1],1)>=0.15 else
             "귀무: region≈global — 신호 존재하나 direction으론 안 감(또는 n 부족)"))
json.dump(results,open(f'{OUT}/exp_region_pool_direction.json','w'),indent=2); log("[saved]")
