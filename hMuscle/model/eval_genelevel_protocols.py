"""
Phase B 세 가지 평가 프로토콜로 재계산 (sklearn MLP, CPU):
  P1: isoform-level, macro-mean  AUPRC/AUROC
  P2: gene-level max, median     AUPRC/AUROC  (CrossIsoFun 프로토콜)
  P3: gene-level max, normalized baseline=0.1 (DIFFUSE 원논문 프로토콜)
  + Fmax, within-gene CV, pos_bias
"""
import os, json, warnings, time
import numpy as np
from collections import defaultdict
warnings.filterwarnings('ignore')

BENCH = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/benchmark_diffuse'
D2    = '/home/welcome1/sw1686/DIFFUSE/hMuscle/data/diffuse_benchmark/datasets2&3/processed_data/dataset2/data'
SEEDS = [42, 123, 456, 789, 2024]

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
t0 = time.time()
print("Loading data...", flush=True)

test_gene  = [x.decode() if isinstance(x,bytes) else x
              for x in np.load(f'{D2}/id_lists/test_gene_list.npy', allow_pickle=True)]
train_gene = [x.decode() if isinstance(x,bytes) else x
              for x in np.load(f'{D2}/id_lists/train_gene_list.npy', allow_pickle=True)]

go_slim = []
with open(f'{D2}/go_terms/go_slim.txt') as f:
    for line in f:
        p = line.strip().split('\t')
        if p: go_slim.append(p[0])

gene2go = defaultdict(set)
with open(f'{D2}/annotations/human_annotations.txt') as f:
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2:
            gene2go[p[0]] = set(p[1:])

X_test  = np.load(f'{BENCH}/esm2_dataset2_test.npy').astype(np.float32)
X_train = np.load(f'{BENCH}/esm2_dataset2_train.npy').astype(np.float32)

test_genes_arr  = np.array(test_gene)
train_genes_arr = np.array(train_gene)
n_unique_genes  = len(set(test_gene))

print(f"  X_test={X_test.shape}, X_train={X_train.shape}", flush=True)
print(f"  Test unique genes: {n_unique_genes}", flush=True)
print(f"  GO terms: {len(go_slim)}, Seeds: {len(SEEDS)}", flush=True)

# ── sklearn MLP (PRISM 아키텍처 근사) ───────────────────────────────────────
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def build_prism_sklearn(seed):
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=1e-3,
        max_iter=80,
        batch_size=512,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=seed,
        verbose=False
    )

# ── 평가 함수 ────────────────────────────────────────────────────────────────
def safe_auprc(y, s):
    if y.sum() == 0 or y.sum() == len(y): return float('nan')
    return float(average_precision_score(y, s))

def safe_auroc(y, s):
    if y.sum() == 0 or y.sum() == len(y): return float('nan')
    return float(roc_auc_score(y, s))

def safe_fmax(y, s):
    if y.sum() == 0: return float('nan')
    prec, rec, _ = precision_recall_curve(y, s)
    f1 = 2*prec*rec / np.maximum(prec+rec, 1e-9)
    return float(np.nanmax(f1))

def auprc_normalized(y_gene, s_gene, target=0.1, rng=None):
    """DIFFUSE: positive 복제로 baseline=0.1 맞춤."""
    pos_idx = np.where(y_gene == 1)[0]
    if len(pos_idx) == 0: return float('nan')
    n_neg = int((y_gene == 0).sum())
    target_pos = int(np.ceil(n_neg * target / (1 - target)))
    n_dup = max(0, target_pos - len(pos_idx))
    if n_dup > 0:
        dup = (rng or np.random).choice(pos_idx, size=n_dup, replace=True)
        y_aug = np.concatenate([y_gene, np.ones(len(dup))])
        s_aug = np.concatenate([s_gene, s_gene[dup]])
    else:
        y_aug, s_aug = y_gene, s_gene
    return float(average_precision_score(y_aug, s_aug))

def gene_level_max(preds, y_iso, genes):
    g_scores, g_labels = {}, {}
    for p, y, g in zip(preds, y_iso, genes):
        if g not in g_scores:
            g_scores[g] = p; g_labels[g] = y
        else:
            g_scores[g] = max(g_scores[g], p)
    ug = sorted(g_scores)
    return np.array([g_labels[g] for g in ug]), np.array([g_scores[g] for g in ug])

def within_gene_cv(preds, genes):
    gp = defaultdict(list)
    for p, g in zip(preds, genes): gp[g].append(p)
    cvs = []
    for ps in gp.values():
        if len(ps) > 1:
            a = np.array(ps); m = a.mean()
            if m > 1e-9: cvs.append(a.std() / m)
    return float(np.mean(cvs)) if cvs else 0.0

def pos_bias_metric(preds, y_iso, genes):
    gd = defaultdict(lambda: {'pos':[], 'all':[]})
    for p, y, g in zip(preds, y_iso, genes):
        gd[g]['all'].append(p)
        if y == 1: gd[g]['pos'].append(p)
    biases = []
    for d in gd.values():
        if d['pos'] and d['all']:
            m = np.mean(d['all'])
            if m > 1e-9: biases.append(np.mean(d['pos']) / m)
    return float(np.mean(biases)) if biases else float('nan')

# ── 단일 GO term 처리 함수 (joblib용) ────────────────────────────────────────
def process_go(go, y_tr, y_te, X_tr, X_te, test_genes):
    if y_tr.sum() < 5 or y_te.sum() < 2:
        return go, {'skip': True}

    seed_preds = []
    for seed in SEEDS:
        clf = build_prism_sklearn(seed)
        clf.fit(X_tr, y_tr)
        seed_preds.append(clf.predict_proba(X_te)[:, 1])
    preds = np.mean(seed_preds, axis=0)

    p1_auprc = safe_auprc(y_te, preds)
    p1_auroc = safe_auroc(y_te, preds)
    p1_fmax  = safe_fmax(y_te, preds)
    p1_cv    = within_gene_cv(preds, test_genes)
    p1_pb    = pos_bias_metric(preds, y_te, test_genes)

    y_g, s_g = gene_level_max(preds, y_te, test_genes)
    p2_auprc = safe_auprc(y_g, s_g)
    p2_auroc = safe_auroc(y_g, s_g)
    p2_fmax  = safe_fmax(y_g, s_g)
    rng_local = np.random.RandomState(0)
    p3_auprc = auprc_normalized(y_g, s_g, rng=rng_local)

    return go, {
        'P1_AUPRC': p1_auprc, 'P1_AUROC': p1_auroc, 'P1_Fmax': p1_fmax,
        'P1_within_CV': p1_cv, 'P1_pos_bias': p1_pb,
        'P2_AUPRC': p2_auprc, 'P2_AUROC': p2_auroc, 'P2_Fmax': p2_fmax,
        'P3_AUPRC_norm': p3_auprc,
        'n_pos_iso': int(y_te.sum()), 'n_pos_gene': int(y_g.sum()),
    }

# ── 메인 루프 (joblib 병렬) ───────────────────────────────────────────────────
from joblib import Parallel, delayed

N_JOBS = 20  # 64코어 중 20개 사용 (GO term별 5 seeds × 병렬)
print(f"\nRetraining PRISM (sklearn MLP) on {len(go_slim)} GO × {len(SEEDS)} seeds")
print(f"Parallel: {N_JOBS} jobs (64 cores available)", flush=True)

# 레이블 미리 계산
go_labels = [
    (go,
     np.array([1 if go in gene2go.get(g,set()) else 0 for g in train_genes_arr]),
     np.array([1 if go in gene2go.get(g,set()) else 0 for g in test_genes_arr]))
    for go in go_slim
]

job_results = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(process_go)(go, y_tr, y_te, X_train, X_test, test_genes_arr)
    for go, y_tr, y_te in go_labels
)

results = dict(job_results)

# 완료 후 요약 출력
print("\nPer-term results:")
print(f"{'GO':14s}  P1_AUPRC  P2_AUPRC  P3_norm   AUROC(P1)  n_pos  CV")
for go in go_slim:
    v = results.get(go, {})
    if v.get('skip'):
        print(f"{go}  SKIP")
        continue
    def fmt(x): return f"{x:.4f}" if x and not np.isnan(x) else "  nan "
    print(f"{go}  {fmt(v['P1_AUPRC'])}    {fmt(v['P2_AUPRC'])}    {fmt(v['P3_AUPRC_norm'])}    "
          f"{fmt(v['P1_AUROC'])}   {v['n_pos_iso']:4d}  {v['P1_within_CV']:.3f}")

# ── 요약 ────────────────────────────────────────────────────────────────────
def nanstats(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    if not v: return {'mean': float('nan'), 'median': float('nan'), 'n': 0}
    return {'mean': float(np.mean(v)), 'median': float(np.median(v)), 'n': len(v)}

valid = {k: v for k, v in results.items() if not v.get('skip')}

summary = {
    'n_terms': len(valid),
    'runtime_sec': round(time.time() - t0),
    # P1: isoform-level, mean (우리 기존 프로토콜)
    'P1_AUPRC':      nanstats([v['P1_AUPRC']      for v in valid.values()]),
    'P1_AUROC':      nanstats([v['P1_AUROC']       for v in valid.values()]),
    'P1_Fmax':       nanstats([v['P1_Fmax']        for v in valid.values()]),
    'P1_within_CV':  nanstats([v['P1_within_CV']   for v in valid.values()]),
    'P1_pos_bias':   nanstats([v['P1_pos_bias']    for v in valid.values()]),
    # P2: gene-level, standard, median (CrossIsoFun 프로토콜)
    'P2_AUPRC':      nanstats([v['P2_AUPRC']       for v in valid.values()]),
    'P2_AUROC':      nanstats([v['P2_AUROC']       for v in valid.values()]),
    'P2_Fmax':       nanstats([v['P2_Fmax']        for v in valid.values()]),
    # P3: gene-level, normalized baseline=0.1 (DIFFUSE 원논문 프로토콜)
    'P3_AUPRC_norm': nanstats([v['P3_AUPRC_norm']  for v in valid.values()]),
}

print("\n" + "="*72, flush=True)
print("SUMMARY", flush=True)
print("="*72, flush=True)
print(f"P1 isoform-level (mean AUPRC): {summary['P1_AUPRC']['mean']:.4f}")
print(f"P1 isoform-level (mean AUROC): {summary['P1_AUROC']['mean']:.4f}")
print(f"P1 Fmax (mean):                {summary['P1_Fmax']['mean']:.4f}")
print(f"P1 within-gene CV (mean):      {summary['P1_within_CV']['mean']:.4f}")
print(f"P1 pos_bias (mean):            {summary['P1_pos_bias']['mean']:.4f}")
print()
print(f"P2 gene-level std  (median AUPRC): {summary['P2_AUPRC']['median']:.4f}  ← CrossIsoFun 비교 기준")
print(f"P2 gene-level std  (mean   AUPRC): {summary['P2_AUPRC']['mean']:.4f}")
print(f"P2 gene-level std  (median AUROC): {summary['P2_AUROC']['median']:.4f}")
print(f"P2 Fmax (median):                  {summary['P2_Fmax']['median']:.4f}")
print()
print(f"P3 gene-level norm (mean   AUPRC): {summary['P3_AUPRC_norm']['mean']:.4f}  ← DIFFUSE 원논문 비교 기준")
print()
print(f"CrossIsoFun DIFFUSE re-eval (P2 기준): 0.404")
print(f"DIFFUSE self-reported       (P3 기준): 0.537")
print(f"Runtime: {summary['runtime_sec']//60}m {summary['runtime_sec']%60}s", flush=True)

out_path = f'{BENCH}/eval_protocols_results.json'
with open(out_path, 'w') as f:
    json.dump({'per_term': results, 'summary': summary}, f, indent=2)
print(f"Saved → {out_path}")
