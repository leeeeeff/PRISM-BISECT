# -*- coding: utf-8 -*-
"""
L9-anchor ablation on the REAL PRISM task (v15d_unified protocol). The L9-anchor benefit was only a
domain-DECODABILITY proxy (+0.028 AUROC). Here: does an L9-containing input beat production L30 on
18-GO macro-AUPRC, with bootstrap CI? Predict-before-look: NO significant gain (consistent with the
manuscript's "per-term layer selection does not exceed the ceiling").

Variants (same head, same 3-seed protocol, focal loss):
  L30            baseline = production input (esm2_embeddings_t30_150M == layer30)
  L9             single mid-network peak layer
  L9+L30         concat (1280-dim)      -- the domain-proxy winner
  L30-L9         depth contrast (640)
  L15+L30        PRISM delta_layer-style reference (1280)
Bootstrap over test isoforms (paired resample) -> 95% CI on macro-AUPRC and on Δ vs L30.
"""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES']=os.environ.get('CUDA_VISIBLE_DEVICES','1')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, json
from sklearn.metrics import average_precision_score
import warnings; warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
tf.get_logger().setLevel('ERROR')
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

DATA='../data'; ANNOT='../data/raw_data/data/annotations'; ID='../data/raw_data/data/id_lists'
N_SEEDS=3
GO_KEYS=['GO:0007204','GO:0030017','GO:0006941','GO:0006914','GO:0043161','GO:0007519','GO:0042692',
 'GO:0055074','GO:0007005','GO:0007517','GO:0032006','GO:0003774','GO:0006096','GO:0007268',
 'GO:0007018','GO:0043005','GO:0030182','GO:0000226']

def load_ids(p):
    a=np.load(p,allow_pickle=True); return [x.decode() if isinstance(x,bytes) else str(x) for x in a]
def lyr(split,l): return np.load(f'{DATA}/esm2_train_human_layer{l:02d}_t30_150M.npy' if split=='tr'
                                 else f'{DATA}/esm2_layer_{l:02d}_t30_150M.npy').astype(np.float32)
tr_gene=load_ids(f'{ID}/train_gene_list.npy'); te_gene=load_ids('my_gene_list_fixed.npy')
ENSG2SYM={}
with open(f'{ID}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p=line.strip().split()
        if len(p)>=5: ENSG2SYM[p[0]]=p[4]
tr_sym=[ENSG2SYM.get(g.split('.')[0],g.split('.')[0]) for g in tr_gene]
te_sym=[ENSG2SYM.get(g.split('.')[0],g.split('.')[0]) for g in te_gene]

def load_labels(go):
    pos=set()
    with open(f'{ANNOT}/human_annotations_unified_bp.txt') as f:
        for line in f:
            pt=line.strip().split('\t')
            if len(pt)>1 and go in pt[1:]: pos.add(pt[0])
    return (np.array([1 if s in pos else 0 for s in tr_sym],np.float32),
            np.array([1 if s in pos else 0 for s in te_sym],np.float32))

L={l:{'tr':lyr('tr',l),'te':lyr('te',l)} for l in [9,15,30]}
VARIANTS={'L30':lambda s:L[30][s],'L9':lambda s:L[9][s],
 'L9+L30':lambda s:np.concatenate([L[9][s],L[30][s]],1),
 'L30-L9':lambda s:L[30][s]-L[9][s],'L15+L30':lambda s:np.concatenate([L[15][s],L[30][s]],1)}
print(f'train {L[30]["tr"].shape} test {L[30]["te"].shape}', flush=True)

def head(d):
    inp=layers.Input(shape=(d,)); x=layers.Dense(256,activation='relu')(inp); x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.3)(x); x=layers.Dense(128,activation='relu')(x); x=layers.Dropout(0.2)(x)
    x=layers.Dense(64,activation='relu')(x); out=layers.Dense(1,activation='sigmoid')(x)
    return models.Model(inp,out)
def ensemble(Xtr,Xte,ytr):
    ps=[]
    for s in range(N_SEEDS):
        tf.random.set_seed(s*137+42); np.random.seed(s*137+42)
        idx=np.random.permutation(len(Xtr)); m=head(Xtr.shape[1])
        m.compile(tf.keras.optimizers.Adam(1e-3),loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
        cb=[callbacks.EarlyStopping('val_loss',patience=10,restore_best_weights=True,verbose=0),
            callbacks.ReduceLROnPlateau(patience=5,factor=0.5,verbose=0)]
        m.fit(Xtr[idx],ytr[idx],epochs=80,batch_size=512,validation_split=0.1,callbacks=cb,verbose=0)
        ps.append(m.predict(Xte,batch_size=2048,verbose=0).flatten()); tf.keras.backend.clear_session()
    return np.mean(ps,0)

labels=[load_labels(go) for go in GO_KEYS]
Yte=np.stack([lt for _,lt in labels],1)   # (N_TE, 18)
scores={v:np.zeros_like(Yte,np.float32) for v in VARIANTS}
for v,fn in VARIANTS.items():
    Xtr,Xte=fn('tr'),fn('te'); t0=time.time()
    for gi,go in enumerate(GO_KEYS):
        scores[v][:,gi]=ensemble(Xtr,Xte,labels[gi][0])
    macro=np.mean([average_precision_score(Yte[:,gi],scores[v][:,gi]) for gi in range(18)])
    print(f'  {v:9s} macro-AUPRC={macro:.4f}  ({time.time()-t0:.0f}s)', flush=True)

# paired bootstrap over test isoforms
rng=np.random.default_rng(0); B=1000; N=len(Yte)
def macro_ap(S,y,idx): return np.mean([average_precision_score(y[idx,gi],S[idx,gi])
                                        for gi in range(18) if y[idx,gi].sum()>0])
boot={v:[] for v in VARIANTS}
for _ in range(B):
    idx=rng.integers(0,N,N)
    for v in VARIANTS: boot[v].append(macro_ap(scores[v],Yte,idx))
boot={v:np.array(b) for v,b in boot.items()}
print('\n=== macro-AUPRC (95% bootstrap CI) + Δ vs L30 ===', flush=True)
base=boot['L30']
for v in VARIANTS:
    b=boot[v]; d=b-base
    print(f'  {v:9s} {b.mean():.4f} [{np.percentile(b,2.5):.4f},{np.percentile(b,97.5):.4f}]'
          +('' if v=='L30' else f'   Δ={d.mean():+.4f} [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]'
            +('  **CI excl 0**' if (np.percentile(d,2.5)>0 or np.percentile(d,97.5)<0) else '')), flush=True)
json.dump({v:{'macro':float(boot[v].mean())} for v in VARIANTS},
          open('../../reports/model_interpretability_map/l9_anchor_ablation.json','w'),indent=2)
print('\n[done]', flush=True)
