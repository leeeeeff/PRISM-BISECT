"""
v10_rf_baseline.py — Non-linear Baseline: LR vs Random Forest (DA Rebuttal)
===========================================================================
GBT 제거 (느림). LR vs RF 비교만.
RF: n_estimators=100, max_depth=6, n_jobs=8 (빠른 버전)

실행: conda activate isoform_env && python hMuscle/model/v10_rf_baseline.py
"""

import os, sys, json, time
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
ts = time.strftime('%Y%m%d_%H%M')
OUT_DIR   = '../../reports/xgb_baseline'
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOT = 500
SEED   = 42
np.random.seed(SEED)

GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0030017': 'Sarcomere org',
    'GO:0006941': 'Muscle contraction',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0003774': 'Motor activity',
    'GO:0006096': 'Glycolysis',
}

V10B_KNOWN = {
    'GO:0007204': 0.7653, 'GO:0030017': 0.7426, 'GO:0006941': 0.5968,
    'GO:0006914': 0.6397, 'GO:0043161': 0.7174, 'GO:0007519': 0.7250,
    'GO:0042692': 0.6526, 'GO:0055074': 0.7255, 'GO:0007005': 0.6624,
    'GO:0007517': 0.7017, 'GO:0032006': 0.6023, 'GO:0003774': 0.8128,
    'GO:0006096': 0.6712,
}

print("=" * 60)
print(" RF Baseline — Loading data ...")
print("=" * 60, flush=True)

X_esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

X_isoid  = load_ids('my_isoform_list_fixed.npy')
X_geneid = load_ids('my_gene_list_fixed.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

X_sym      = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in X_geneid]
X_genebase = [g.split('.')[0] for g in X_geneid]
print(f"  Isoforms: {len(X_isoid)}", flush=True)


def load_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y = np.array([1 if s in pos else 0 for s in X_sym], dtype=np.float32)
    return y


def bootstrap_auprc(y_true, y_score, gene_ids, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    unique_genes = np.unique(gene_ids)
    # Precompute gene→isoform index map once (O(n) vs O(n_boot * n_genes * n_isoforms))
    gene_to_idx = {g: np.where(gene_ids == g)[0] for g in unique_genes}
    base = float(average_precision_score(y_true, y_score))
    boots = []
    for _ in range(n_boot):
        g_sample = rng.choice(unique_genes, size=len(unique_genes), replace=True)
        idx = np.concatenate([gene_to_idx[g] for g in g_sample])
        if y_true[idx].sum() > 0:
            boots.append(average_precision_score(y_true[idx], y_score[idx]))
    boots = np.array(boots)
    return base, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_go(go_term):
    y = load_labels(go_term)
    if y.sum() < 10:
        return None

    sc = StandardScaler()
    X_sc = sc.fit_transform(X_esm2)
    genes = np.array(X_genebase)
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_sc, y, groups=genes))

    models = {
        'LR': LogisticRegression(class_weight='balanced', C=1.0, max_iter=500, random_state=SEED),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced',
                                      random_state=SEED, n_jobs=8),
    }

    results = {}
    for mname, model in models.items():
        ys_all, ps_all, gs_all = [], [], []
        for tr_idx, te_idx in splits:
            m = model.__class__(**model.get_params())
            m.fit(X_sc[tr_idx], y[tr_idx])
            probs = m.predict_proba(X_sc[te_idx])[:, 1]
            ys_all.extend(y[te_idx].tolist())
            ps_all.extend(probs.tolist())
            gs_all.extend(genes[te_idx].tolist())

        ys = np.array(ys_all); ps = np.array(ps_all); gs = np.array(gs_all)
        if ys.sum() == 0:
            continue
        auprc, ci_lo, ci_hi = bootstrap_auprc(ys, ps, gs)
        results[mname] = {'auprc': round(auprc, 4), 'ci_lo': round(ci_lo, 4),
                           'ci_hi': round(ci_hi, 4)}
        print(f"    {mname}: {auprc:.4f} [{ci_lo:.4f}-{ci_hi:.4f}]", flush=True)

    results['v10B'] = {'auprc': V10B_KNOWN[go_term]}
    return results


print("\n" + "=" * 60)
print(" Running LR vs RF per GO term ...")
print("=" * 60, flush=True)

all_results = {}
for go, go_name in GO_TERMS.items():
    print(f"\n[{go}] {go_name}", flush=True)
    t0 = time.time()
    res = run_go(go)
    if res:
        all_results[go] = {'go_name': go_name, 'models': res}
    print(f"  elapsed: {time.time()-t0:.0f}s", flush=True)

print("\n" + "=" * 60)
print(" Summary Table")
print("=" * 60, flush=True)
print('{:<15} {:>8} {:>8} {:>8}'.format('GO', 'LR', 'RF', 'v10-B'))
print('-'*42)

macro = {'LR': [], 'RF': [], 'v10B': []}
for go, data in all_results.items():
    row = '{:<15}'.format(go)
    for m in ['LR', 'RF', 'v10B']:
        v = data['models'].get(m, {}).get('auprc', float('nan'))
        row += ' {:>8.4f}'.format(v) if not (isinstance(v, float) and v != v) else ' {:>8}'.format('n/a')
        if m in macro and not (isinstance(v, float) and v != v):
            macro[m].append(v)
    print(row, flush=True)

print('-'*42)
print('{:<15} {:>8.4f} {:>8.4f} {:>8.4f}'.format(
    'MACRO', np.mean(macro['LR']), np.mean(macro['RF']), np.mean(macro['v10B'])))

out_path = f'{OUT_DIR}/rf_results_{ts}.json'
with open(out_path, 'w') as f:
    json.dump({'results': all_results, 'timestamp': ts,
               'macros': {m: round(float(np.mean(v)), 4) for m, v in macro.items() if v}}, f, indent=2)
print(f"\nFINAL: {out_path}", flush=True)
