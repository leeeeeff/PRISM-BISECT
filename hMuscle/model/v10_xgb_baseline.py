"""
v10_xgb_baseline.py — Non-linear Baseline Comparison (DA Rebuttal)
===================================================================
목적: DA 비판 #3 대응 — "MLP > LR은 capacity 차이이지 방법론 기여 아님"
      XGBoost / Random Forest를 동일 ESM-2 640d 입력으로 비교.

만약 v10-B MLP >> XGB >> RF >> LR이면:
  → non-linearity 자체가 핵심이고, MLP가 최선임을 입증.
만약 XGB ≈ v10-B MLP이면:
  → "MLP 특이적 기여" 주장 약화, XGB도 동일 효과. 방법론 강화 필요.

평가:
  - 5-fold GroupKFold (gene-stratified)
  - AUPRC primary (bootstrap CI n=500)
  - 13 GO terms 전체

결과:
  reports/xgb_baseline/xgb_results_{ts}.json

실행:
  conda activate isoform_env
  python hMuscle/model/v10_xgb_baseline.py
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Using GradientBoostingClassifier instead.")

# ─── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
ts = time.strftime('%Y%m%d_%H%M')
OUT_DIR   = f'../../reports/xgb_baseline'
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOT  = 500
SEED    = 42
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

# v10-B multi-seed results from F37 + F45 (for comparison)
V10B_KNOWN = {
    'GO:0007204': 0.7653, 'GO:0030017': 0.7426, 'GO:0006941': 0.5968,
    'GO:0006914': 0.6397, 'GO:0043161': 0.7174, 'GO:0007519': 0.7250,
    'GO:0042692': 0.6526, 'GO:0055074': 0.7255, 'GO:0007005': 0.6624,
    'GO:0007517': 0.7017, 'GO:0032006': 0.6023, 'GO:0003774': 0.8128,
    'GO:0006096': 0.6712,
}

# ─── Load data ──────────────────────────────────────────────────────────────
print("=" * 65)
print(" XGBoost/RF Baseline — Loading test data ...")
print("=" * 65)

# ── Train set (same as v10-B) ──
X_tr_esm2  = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
tr_geneid  = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)

# ── Test set ──
X_te_esm2  = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

te_isoid   = load_ids('my_isoform_list_fixed.npy')
te_geneid  = load_ids('my_gene_list_fixed.npy')
tr_geneid  = [x.decode() if isinstance(x, bytes) else str(x) for x in tr_geneid]

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

te_sym      = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_geneid]
te_genebase = [g.split('.')[0] for g in te_geneid]
tr_sym      = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in tr_geneid]
print(f"  Train isoforms: {len(tr_geneid)}, Test isoforms: {len(te_isoid)}")

# backward compat alias
X_esm2    = X_te_esm2
X_sym     = te_sym
X_genebase = te_genebase
N = len(te_isoid)


def load_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te, pos


def bootstrap_ci(y_true, y_score, gene_ids, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    unique_genes = np.unique(gene_ids)
    base_auprc = float(average_precision_score(y_true, y_score))

    # Pre-index gene → isoform positions (O(N) once vs O(N²) per bootstrap)
    from collections import defaultdict
    gene_to_idxs = defaultdict(list)
    for i, g in enumerate(gene_ids):
        gene_to_idxs[g].append(i)
    gene_to_idxs = {g: np.array(idxs) for g, idxs in gene_to_idxs.items()}

    boot_auprcs = []
    for _ in range(n_boot):
        g_sample = rng.choice(unique_genes, size=len(unique_genes), replace=True)
        idx = np.concatenate([gene_to_idxs[g] for g in g_sample])
        if y_true[idx].sum() == 0:
            continue
        boot_auprcs.append(average_precision_score(y_true[idx], y_score[idx]))
    boot_auprcs = np.array(boot_auprcs)
    ci_lo = float(np.percentile(boot_auprcs, 2.5))
    ci_hi = float(np.percentile(boot_auprcs, 97.5))
    return base_auprc, ci_lo, ci_hi


def run_models(go_term):
    # v10-B 동일 프로토콜: train set으로 학습 → test set으로 평가
    y_tr, y_te, pos_set = load_labels(go_term)
    n_pos_tr = int(y_tr.sum()); n_pos_te = int(y_te.sum())
    if n_pos_tr < 10 or n_pos_te < 5:
        return None

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_esm2)
    X_te_sc = sc.transform(X_te_esm2)
    te_genes = np.array(te_genebase)

    if HAS_XGB:
        pos_weight = (y_tr == 0).sum() / max(y_tr.sum(), 1)
    models = {
        'LR': LogisticRegression(class_weight='balanced', C=1.0,
                                  max_iter=1000, random_state=SEED),
        'RF': RandomForestClassifier(n_estimators=200, max_depth=8,
                                      class_weight='balanced', random_state=SEED, n_jobs=4),
    }
    if HAS_XGB:
        models['XGB'] = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=float(pos_weight), eval_metric='aucpr',
            use_label_encoder=False, random_state=SEED, n_jobs=4,
            verbosity=0,
        )
    else:
        models['XGB(GB)'] = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            random_state=SEED,
        )

    results = {}
    for mname, model in models.items():
        model.fit(X_tr_sc, y_tr)
        ps = model.predict_proba(X_te_sc)[:, 1]
        ys = y_te
        gs = te_genes
        if ys.sum() == 0:
            continue
        auprc = float(average_precision_score(ys, ps))
        auprc, ci_lo, ci_hi = bootstrap_ci(ys, ps, gs)
        results[mname] = {'auprc': round(auprc, 4), 'ci_lo': round(ci_lo, 4),
                           'ci_hi': round(ci_hi, 4)}
        print(f"    {mname}: AUPRC={auprc:.4f} [{ci_lo:.4f}-{ci_hi:.4f}]")

    return results


# ─── Main ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" Running models per GO term ...")
print("=" * 65)

all_results = {}
for go, go_name in GO_TERMS.items():
    print(f"\n[{go}] {go_name}")
    res = run_models(go)
    if res is None:
        print("  SKIP")
        continue
    res['v10B'] = {'auprc': V10B_KNOWN[go], 'ci_lo': None, 'ci_hi': None}
    all_results[go] = {'go_name': go_name, 'models': res}

# ─── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" Summary Table")
print("=" * 65)

print('{:<15} {:>6} {:>6} {:>6} {:>6}'.format('GO Term', 'LR', 'RF', 'XGB', 'v10-B'))
print('-' * 45)

model_names = ['LR', 'RF', 'XGB' if HAS_XGB else 'XGB(GB)']
macro = {m: [] for m in model_names + ['v10B']}

for go, data in all_results.items():
    row = '{:<15}'.format(go)
    for m in model_names + ['v10B']:
        v = data['models'].get(m, {}).get('auprc', float('nan'))
        row += ' {:>6.4f}'.format(v) if not np.isnan(v) else ' {:>6}'.format('n/a')
        if not np.isnan(v):
            macro[m].append(v)
    print(row)

print('-' * 45)
macro_row = '{:<15}'.format('MACRO')
for m in model_names + ['v10B']:
    vals = macro[m]
    macro_row += ' {:>6.4f}'.format(np.mean(vals)) if vals else ' {:>6}'.format('n/a')
print(macro_row)

# Save
out_path = f'{OUT_DIR}/xgb_results_{ts}.json'
with open(out_path, 'w') as f:
    json.dump({'results': all_results, 'timestamp': ts,
               'macros': {m: round(float(np.mean(v)), 4) for m, v in macro.items() if v}}, f, indent=2)
print(f"\nFINAL: {out_path}")
