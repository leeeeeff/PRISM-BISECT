import numpy as np, re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings; warnings.filterwarnings('ignore')

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

BRAIN_DIR = '../data/brain_esm2'
ANNOT_DIR = '../data/raw_data/data/annotations'

X_all = np.load(f'{BRAIN_DIR}/brain_only_esm2_t30_150M.npy').astype(np.float32)

brain_syms = []
with open(f'{BRAIN_DIR}/brain_only.gtf') as f:
    for line in f:
        if line.startswith('#'): continue
        fields = line.split('\t')
        if len(fields) < 9 or fields[2] != 'transcript': continue
        m = re.search(r'gene_name "([^"]+)"', fields[8])
        brain_syms.append(m.group(1) if m else 'NA')
brain_syms = np.array(brain_syms)

gene2go = {}
with open(f'{ANNOT_DIR}/human_annotations_unified_bp.txt') as f:
    for line in f:
        p = line.strip().split()
        if p: gene2go[p[0]] = set(p[1:])

GO_TERMS = {
    'GO:0007268': 'synaptic transmission',
    'GO:0007010': 'cytoskeleton org',
    'GO:0007018': 'MT-based movement',
    'GO:0048488': 'synaptic vesicle endocytosis',
    'GO:0035249': 'glutamatergic',
    'GO:0051932': 'GABAergic',
}

header = '  {:<33s}  {:>5s}  {:>7s}  {:>7s}  {:>7s}  best_k  verdict'
print(header.format('GO term', 'n_pos', 'k=2 sil', 'k=3 sil', 'k=4 sil'))
print('-' * 90)

results = {}
for gid, gname in GO_TERMS.items():
    pos_mask = np.array([gid in gene2go.get(s, set()) for s in brain_syms])
    X_pos = X_all[pos_mask]
    n_pos = len(X_pos)

    if n_pos < 20:
        print('  {:<33s}  {:>5d}  (insufficient)'.format(gname, n_pos))
        continue

    norms = np.linalg.norm(X_pos, axis=1, keepdims=True) + 1e-10
    X_norm = X_pos / norms
    pca = PCA(n_components=min(50, X_norm.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_norm)

    scores = {}
    for k in [2, 3, 4]:
        if n_pos < k * 5:
            scores[k] = float('nan')
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_pca)
        sc = silhouette_score(X_pca, labels, sample_size=min(500, n_pos))
        scores[k] = sc

    valid = {k: v for k, v in scores.items() if not np.isnan(v)}
    best_k = max(valid, key=valid.get) if valid else 1
    best_sc = valid.get(best_k, 0)
    if best_sc > 0.10:
        verdict = 'cluster 구조 있음'
    elif best_sc > 0.05:
        verdict = '약한 구조'
    else:
        verdict = '단일 군집'

    def fmt(v):
        return '{:.3f}'.format(v) if not np.isnan(v) else '  -  '

    print('  {:<33s}  {:>5d}  {:>7s}  {:>7s}  {:>7s}  k={}  {}'.format(
        gname, n_pos, fmt(scores.get(2, float('nan'))),
        fmt(scores.get(3, float('nan'))), fmt(scores.get(4, float('nan'))),
        best_k, verdict))

    results[gname] = {'n_pos': n_pos, 'scores': scores, 'best_k': best_k, 'best_sc': best_sc}

print('\n=== 해석 기준 ===')
print('silhouette > 0.10 : prototype k 사용 가치 있음')
print('silhouette 0.05~0.10 : 약한 근거, 주의 필요')
print('silhouette < 0.05 : 단일 prototype과 동등')
