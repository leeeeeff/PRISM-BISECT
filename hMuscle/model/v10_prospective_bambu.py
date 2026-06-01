#!/usr/bin/env python3
"""
v10_prospective_bambu.py
=========================
Novel isoform expression ranking — prospective validation.

설계:
  Train:  known (non-BambuTx) isoform pairwise D/S LR
  Test:   BambuTx vs known isoform 쌍 — model이 본 적 없는 isoform
  검증:   실제 N-sample 발현량 기반 ranking과 비교 → pairwise AUROC

추가 분석:
  [A] BambuTx expression distribution (dominant vs minor)
  [B] N vs D 발현 비율 비교 → isoform switch 후보 발굴
  [C] D/S feature가 실제로 BambuTx ranking을 맞추는가?
      (ESM-2 vs D/S vs Full)

논문 주장:
  "Pairwise ranking model trained on known isoforms generalizes to
  novel long-read isoforms (BambuTx), achieving AUROC X on prospective
  expression rank prediction."
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import os, json, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FEAT_DIR = '../results_isoform/features'
DATA_DIR = '../data'
COUNTS   = '../data/counts_transcript.txt'
OUT_DIR  = '../../reports/within_gene'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print(" Prospective BambuTx Expression Ranking Validation")
print("=" * 65)

# ── 1. 기본 데이터 로드 ───────────────────────────────────────────
print("\n[1] Loading data ...")
ratio_arr = np.load(f'{FEAT_DIR}/within_gene_ratio.npy')  # N-sample 기반
iso_arr   = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
gene_arr  = np.load('my_gene_list_fixed.npy', allow_pickle=True)
iso_list  = [s.decode() if isinstance(s, bytes) else s for s in iso_arr]
gene_list = [s.decode() if isinstance(s, bytes) else s for s in gene_arr]
gene_base_list = [g.split('.')[0] for g in gene_list]

valid_mask  = ~np.isnan(ratio_arr)
valid_idxs  = np.where(valid_mask)[0]
valid_genes = np.array([gene_base_list[i] for i in valid_idxs])
valid_ratios = ratio_arr[valid_idxs]
valid_isos  = np.array([iso_list[i] for i in valid_idxs])

esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)[valid_idxs]
dd   = np.load(f'{FEAT_DIR}/domain_delta_proper_test_v2.npy').astype(np.float32)[valid_idxs]
sd   = np.load(f'{FEAT_DIR}/splicing/splicing_delta_v2.npy').astype(np.float32)[valid_idxs]

is_bambu = np.array(['BambuTx' in iso for iso in valid_isos])
is_known = ~is_bambu

print(f"  Valid isoforms: {len(valid_idxs)}")
print(f"  BambuTx expressed: {is_bambu.sum()}")
print(f"  Known isoforms: {is_known.sum()}")

# ── 2. N-sample vs D-sample 발현 로드 ─────────────────────────────
print("\n[2] Loading N/D sample expression ...")
df = pd.read_csv(COUNTS, sep='\t')
n_cols = [c for c in df.columns if c.startswith('N')]
d_cols = [c for c in df.columns if c.startswith('D')]
df['gene_base'] = df['GENEID'].str.split('.').str[0]
df['mean_N'] = df[n_cols].mean(axis=1)
df['mean_D'] = df[d_cols].mean(axis=1)
iso_to_meanN = dict(zip(df['TXNAME'], df['mean_N']))
iso_to_meanD = dict(zip(df['TXNAME'], df['mean_D']))
print(f"  N samples: {len(n_cols)}, D samples: {len(d_cols)}")

# ── 3. BambuTx 발현 분포 분석 ─────────────────────────────────────
print("\n[3] BambuTx expression analysis ...")
bambu_ratios = valid_ratios[is_bambu]
bambu_isos   = valid_isos[is_bambu]
bambu_genes  = valid_genes[is_bambu]

# BambuTx 중 dominant (ratio > 0.3)인 비율
dominant_bambu = (bambu_ratios > 0.3).sum()
print(f"  BambuTx with ratio > 0.3 (co-dominant): {dominant_bambu}/{len(bambu_ratios)} ({dominant_bambu/len(bambu_ratios)*100:.1f}%)")
print(f"  BambuTx median ratio: {np.median(bambu_ratios):.4f}")
print(f"  BambuTx max ratio: {np.max(bambu_ratios):.4f}")

# N vs D ratio 비교 (isoform switch 후보)
gene_to_local = {}
for local_i, g in enumerate(valid_genes):
    gene_to_local.setdefault(g, []).append(local_i)

# BambuTx isoforms that have both N-sample and D-sample expression
switch_candidates = []
for local_i in np.where(is_bambu)[0]:
    iso = valid_isos[local_i]
    g = valid_genes[local_i]
    mean_n = iso_to_meanN.get(iso, np.nan)
    mean_d = iso_to_meanD.get(iso, np.nan)
    if np.isnan(mean_n) or np.isnan(mean_d): continue
    # gene-level total
    gene_total_n = sum(iso_to_meanN.get(valid_isos[j], 0) for j in gene_to_local.get(g, []))
    gene_total_d = sum(iso_to_meanD.get(valid_isos[j], 0) for j in gene_to_local.get(g, []))
    ratio_n = mean_n / (gene_total_n + 1e-8)
    ratio_d = mean_d / (gene_total_d + 1e-8)
    switch_score = abs(ratio_n - ratio_d)
    switch_candidates.append({
        'iso': iso, 'gene': g,
        'ratio_N': ratio_n, 'ratio_D': ratio_d,
        'switch_score': switch_score,
        'direction': 'UP' if ratio_d > ratio_n else 'DOWN'
    })

switch_df = pd.DataFrame(switch_candidates).sort_values('switch_score', ascending=False)
top_switches = switch_df[switch_df['switch_score'] > 0.1].head(10)
print(f"\n  BambuTx with N/D ratio shift > 0.1: {len(top_switches)}")
if len(top_switches) > 0:
    print("  Top isoform switches (BambuTx):")
    for _, row in top_switches.head(5).iterrows():
        print(f"    {row['iso']:20s} gene={row['gene']:12s} N={row['ratio_N']:.3f} D={row['ratio_D']:.3f} ({row['direction']})")

# ── 4. Pairwise 학습 (non-BambuTx만) ─────────────────────────────
print("\n[4] Training pairwise model on known isoforms ...")

# Gene 그룹 구성 (known only, ≥2 isoform)
known_gene_to_local = {}
for local_i in np.where(is_known)[0]:
    g = valid_genes[local_i]
    known_gene_to_local.setdefault(g, []).append(local_i)
known_multi_genes = {g: v for g, v in known_gene_to_local.items() if len(v) >= 2}
print(f"  Training genes (known, ≥2 iso): {len(known_multi_genes)}")

# 학습 쌍 생성
def make_train_pairs(gene_dict, ratios, isos, max_per_gene=10):
    pairs_esm, pairs_ds, labels = [], [], []
    rng = np.random.RandomState(42)
    for g, idxs in gene_dict.items():
        from itertools import combinations
        pair_list = list(combinations(range(len(idxs)), 2))
        if len(pair_list) > max_per_gene:
            sel = rng.choice(len(pair_list), max_per_gene, replace=False)
            pair_list = [pair_list[k] for k in sel]
        for (a, b) in pair_list:
            ia, ib = idxs[a], idxs[b]
            diff = ratios[ia] - ratios[ib]
            if abs(diff) < 0.02: continue
            esm_d = esm2[ia] - esm2[ib]
            ds_d  = np.concatenate([sd[ia] - sd[ib], dd[ia] - dd[ib]])
            label = 1 if diff > 0 else 0
            pairs_esm.append(esm_d)
            pairs_ds.append(ds_d)
            labels.append(label)
    return np.array(pairs_esm), np.array(pairs_ds), np.array(labels)

X_tr_esm, X_tr_ds, y_tr = make_train_pairs(known_multi_genes, valid_ratios, valid_isos)
print(f"  Training pairs: {len(y_tr)}")

# 모델 학습
pipe_esm  = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=1.0, max_iter=500))])
pipe_ds   = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=1.0, max_iter=500))])
pipe_full = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=1.0, max_iter=500))])

X_tr_full = np.hstack([X_tr_esm, X_tr_ds])
pipe_esm.fit(X_tr_esm,  y_tr)
pipe_ds.fit(X_tr_ds,    y_tr)
pipe_full.fit(X_tr_full, y_tr)
print("  Models trained.")

# ── 5. BambuTx Prospective Test ───────────────────────────────────
print("\n[5] Prospective test: BambuTx vs known pairs ...")

# 같은 gene 내 BambuTx vs known 쌍
gene_to_bambu_local = {}
for local_i in np.where(is_bambu)[0]:
    g = valid_genes[local_i]
    if g in known_gene_to_local:  # known iso가 있는 gene만
        gene_to_bambu_local.setdefault(g, []).append(local_i)

test_esm, test_ds, test_labels = [], [], []
test_meta = []

for g, bambu_locals in gene_to_bambu_local.items():
    known_locals = known_gene_to_local.get(g, [])
    if not known_locals: continue

    for bi in bambu_locals:
        for ki in known_locals:
            diff = valid_ratios[bi] - valid_ratios[ki]
            if abs(diff) < 0.01: continue  # tie 제외 (BambuTx ratio 낮아서 작은 threshold)

            esm_d = esm2[bi] - esm2[ki]
            ds_d  = np.concatenate([sd[bi] - sd[ki], dd[bi] - dd[ki]])
            label = 1 if diff > 0 else 0

            test_esm.append(esm_d)
            test_ds.append(ds_d)
            test_labels.append(label)
            test_meta.append({
                'gene': g,
                'bambu': valid_isos[bi],
                'known': valid_isos[ki],
                'ratio_bambu': valid_ratios[bi],
                'ratio_known': valid_ratios[ki],
            })

test_esm  = np.array(test_esm)
test_ds   = np.array(test_ds)
test_full = np.hstack([test_esm, test_ds])
test_labels = np.array(test_labels)
print(f"  Test pairs (BambuTx vs known): {len(test_labels)}")
print(f"  BambuTx more expressed in: {test_labels.sum()}/{len(test_labels)} pairs ({test_labels.mean():.1%})")

if len(test_labels) > 20 and test_labels.std() > 0:
    proba_esm  = pipe_esm.predict_proba(test_esm)[:, 1]
    proba_ds   = pipe_ds.predict_proba(test_ds)[:, 1]
    proba_full = pipe_full.predict_proba(test_full)[:, 1]

    auc_esm  = roc_auc_score(test_labels, proba_esm)
    auc_ds   = roc_auc_score(test_labels, proba_ds)
    auc_full = roc_auc_score(test_labels, proba_full)
    print(f"\n  [Prospective AUROC]")
    print(f"    ESM-2:    {auc_esm:.4f}")
    print(f"    D/S only: {auc_ds:.4f}")
    print(f"    Full:     {auc_full:.4f}")
    print(f"    Random:   0.5000")

    # Bootstrap CI
    meta_df  = pd.DataFrame(test_meta)
    all_test_genes = meta_df['gene'].unique()
    rng = np.random.RandomState(42)
    boot_ds, boot_esm, boot_full = [], [], []
    for _ in range(500):
        bg = set(rng.choice(all_test_genes, len(all_test_genes), replace=True))
        mask = meta_df['gene'].isin(bg).values
        if mask.sum() < 10 or test_labels[mask].std() < 1e-8: continue
        boot_ds.append(roc_auc_score(test_labels[mask], proba_ds[mask]))
        boot_esm.append(roc_auc_score(test_labels[mask], proba_esm[mask]))
        boot_full.append(roc_auc_score(test_labels[mask], proba_full[mask]))

    def ci(arr):
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    m_ds,  lo_ds,  hi_ds  = ci(boot_ds)
    m_esm, lo_esm, hi_esm = ci(boot_esm)
    m_full,lo_full,hi_full= ci(boot_full)
    p_ds  = (np.array(boot_ds) <= 0.5).mean()
    p_esm = (np.array(boot_esm) <= 0.5).mean()

    print(f"\n  [Bootstrap CI (n=500, gene-block)]")
    print(f"    ESM-2:    {m_esm:.4f} 95%CI[{lo_esm:.4f},{hi_esm:.4f}] p={p_esm:.4f}")
    print(f"    D/S only: {m_ds:.4f}  95%CI[{lo_ds:.4f},{hi_ds:.4f}]  p={p_ds:.4f}")
    print(f"    Full:     {m_full:.4f} 95%CI[{lo_full:.4f},{hi_full:.4f}]")
else:
    auc_esm = auc_ds = auc_full = None
    m_ds = lo_ds = hi_ds = p_ds = None
    print("  [SKIP] Too few test pairs")

# ── 6. 상위 BambuTx 예측 (dominant으로 예측된 novel iso) ──────────
print("\n[6] Top BambuTx predicted as expression-dominant ...")
if len(test_labels) > 0 and auc_ds is not None:
    meta_df['p_bambu_dominant_ds']   = proba_ds
    meta_df['p_bambu_dominant_full'] = proba_full
    meta_df['label'] = test_labels

    # 각 BambuTx의 평균 dominant probability
    bambu_scores = meta_df.groupby('bambu').agg(
        mean_p_ds=('p_bambu_dominant_ds', 'mean'),
        mean_p_full=('p_bambu_dominant_full', 'mean'),
        mean_ratio_bambu=('ratio_bambu', 'mean'),
        gene=('gene', 'first'),
        n_pairs=('gene', 'count'),
    ).reset_index().sort_values('mean_p_full', ascending=False)

    print("  Top BambuTx predicted dominant (Full model):")
    for _, row in bambu_scores.head(8).iterrows():
        print(f"    {row['bambu']:20s} gene={row['gene']:12s} "
              f"p_dom={row['mean_p_full']:.3f}  actual_ratio={row['mean_ratio_bambu']:.3f}")

    # Spearman (predicted score vs actual ratio, per BambuTx)
    merged = bambu_scores[bambu_scores['mean_ratio_bambu'] > 0]
    if len(merged) > 10:
        r_sp, p_sp = spearmanr(merged['mean_p_full'], merged['mean_ratio_bambu'])
        print(f"\n  Spearman(predicted_dominance, actual_ratio): r={r_sp:.4f} p={p_sp:.4f}")

# ── 7. BambuTx가 dominant인 경우 vs N-only 신호 ───────────────────
print("\n[7] Novel isoforms with high expression ratio (potential discoveries) ...")
bambu_high = [(valid_isos[i], valid_genes[i], valid_ratios[i])
              for i in np.where(is_bambu)[0] if valid_ratios[i] > 0.4]
bambu_high.sort(key=lambda x: -x[2])
print(f"  BambuTx with ratio > 0.4: {len(bambu_high)}")
for iso, gene, rat in bambu_high[:8]:
    print(f"    {iso:20s}  gene={gene:12s}  ratio={rat:.3f}")

# ── 8. 저장 ──────────────────────────────────────────────────────
results = {
    'n_bambu_expressed': int(is_bambu.sum()),
    'n_bambu_dominant': int((bambu_ratios > 0.3).sum()),
    'n_test_pairs': int(len(test_labels)),
    'n_top_switches': int(len(top_switches)),
    'prospective_auroc': {
        'esm2': float(auc_esm) if auc_esm else None,
        'ds':   float(auc_ds)  if auc_ds else None,
        'full': float(auc_full) if auc_full else None,
    },
    'bootstrap': {
        'ds':   {'mean': m_ds,  'ci95': [lo_ds,  hi_ds],  'p': p_ds}  if m_ds  else None,
        'esm2': {'mean': m_esm, 'ci95': [lo_esm, hi_esm], 'p': p_esm} if 'p_esm' in dir() else None,
        'full': {'mean': m_full,'ci95': [lo_full,hi_full]}              if 'lo_full' in dir() else None,
    },
    'top_switches': switch_df.head(10).to_dict('records') if len(switch_df) > 0 else [],
    'bambu_high_expression': [{'iso': iso, 'gene': gene, 'ratio': float(rat)} for iso, gene, rat in bambu_high[:10]],
    'timestamp': time.strftime('%Y%m%d_%H%M'),
}

fname = f'{OUT_DIR}/prospective_bambu_{time.strftime("%Y%m%d_%H%M")}.json'
with open(fname, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[Saved] {fname}")
print("\n" + "=" * 65)
print(" PROSPECTIVE VALIDATION SUMMARY")
print("=" * 65)
if auc_ds is not None:
    print(f"  ESM-2:    Prospective AUROC = {auc_esm:.4f}")
    print(f"  D/S only: Prospective AUROC = {auc_ds:.4f}  95%CI[{lo_ds:.4f},{hi_ds:.4f}]  p={p_ds:.4f}")
    print(f"  Full:     Prospective AUROC = {auc_full:.4f}")
    print(f"  BambuTx dominant (ratio>0.3): {int((bambu_ratios > 0.3).sum())} isoforms")
    print(f"  Top isoform switches: {len(top_switches)}")
print("Done.")
