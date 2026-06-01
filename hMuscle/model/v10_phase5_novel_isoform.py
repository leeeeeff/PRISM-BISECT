"""
v10_phase5_novel_isoform.py — Phase 5: Novel Isoform Discovery
===============================================================
목적: v10-B 고점수 isoform 중 학습 라벨에 없는 후보 발굴.

두 가지 candidate type:
  A. Novel gene:     label=0 gene의 isoform 중 score > THRESH
                     → 현재 GO 미주석 유전자에 대한 새 기능 예측
  B. Isoform-switch: label=1 gene 내 isoform 간 score range > 0.3
                     → 특정 isoform만이 기능적임을 예측
                     (현재 라벨링은 gene 전체에 label=1 부여 — 비현실적 가정)

출력:
  reports/phase5_novel/{ts}/
    novel_candidates.tsv      — 전체 후보 목록
    isoform_switch.tsv        — isoform-switch 목록
    phase5_summary.json       — GO term별 통계
    top_candidates.txt        — 문헌 검색용 상위 후보 요약

실행:
  conda activate isoform_env
  python hMuscle/model/v10_phase5_novel_isoform.py
"""

import os, sys, json, time, requests
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        tf.config.set_visible_devices(gpus[1] if len(gpus) > 1 else gpus[0], 'GPU')
    except RuntimeError:
        pass

# ─── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
ts = time.strftime('%Y%m%d_%H%M')
OUT_DIR   = f'../../reports/phase5_novel/{ts}'
os.makedirs(OUT_DIR, exist_ok=True)

SEED  = 42
NOVEL_THRESH  = 0.5   # novel gene candidate threshold (낮추면 더 많은 후보)
SWITCH_THRESH = 0.3   # within-gene score range threshold for isoform-switch
TOP_N = 20            # per GO term top candidates

tf.random.set_seed(SEED)
np.random.seed(SEED)

GO_TERMS = ['GO:0006096', 'GO:0003774', 'GO:0007204', 'GO:0030017', 'GO:0006941']

# ─── Load data ─────────────────────────────────────────────────────────────
print("=" * 65)
print(" Phase 5: Novel Isoform Discovery — Loading data ...")
print("=" * 65)

X_te_esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
X_tr_esm2 = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)

def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

X_te_isoid  = load_ids('my_isoform_list_fixed.npy')
X_te_geneid = load_ids('my_gene_list_fixed.npy')
X_tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')

# ENSG → gene symbol 매핑
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

X_te_sym      = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in X_te_geneid]
X_te_genebase = [g.split('.')[0] for g in X_te_geneid]

# 전체 GO 주석 사전 구축 (gene_symbol → set of GO terms)
gene_go_map = {}
with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > 1:
            gene_go_map[parts[0]] = set(parts[1:])

N = len(X_te_isoid)
print(f"  Test isoforms: {N},  Train genes: {len(X_tr_geneid)}")
print(f"  Annotated genes: {len(gene_go_map)}")


# ─── Model: v10-B ──────────────────────────────────────────────────────────
def build_v10B(esm_dim=640):
    inp = layers.Input(shape=(esm_dim,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name='v10B')


def load_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_te = np.array([1 if s in pos else 0 for s in X_te_sym], dtype=np.float32)
    y_tr = np.array([1 if g in pos else 0 for g in X_tr_geneid], dtype=np.float32)
    return y_tr, y_te, pos


def get_cw(y):
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    return {0: 1.0, 1: n_neg / max(n_pos, 1)}


def train_v10b(go_term):
    y_tr, y_te, pos_set = load_labels(go_term)
    if y_tr.sum() < 2 or y_te.sum() == 0:
        return None, None, None

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_esm2)
    X_te_sc = sc.transform(X_te_esm2)

    rng = np.random.RandomState(SEED)
    n_val = max(int(len(y_tr) * 0.1), 100)
    vi = rng.choice(len(y_tr), size=n_val, replace=False)
    ti = np.setdiff1d(np.arange(len(y_tr)), vi)

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    model = build_v10B()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    )
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=5, verbose=0),
    ]
    model.fit(X_tr_sc[ti], y_tr[ti],
              validation_data=(X_tr_sc[vi], y_tr[vi]),
              epochs=80, batch_size=512,
              class_weight=get_cw(y_tr),
              callbacks=cb_list, verbose=0)

    probs = model.predict(X_te_sc, verbose=0).ravel()
    auprc = float(average_precision_score(y_te, probs))
    print(f"    [{go_term}] AUPRC={auprc:.4f}, n_pos_te={int(y_te.sum())}")
    return probs, y_te, pos_set


# ─── Ensembl biotype lookup (optional, cached) ─────────────────────────────
_biotype_cache = {}
def get_biotype(enst_id):
    clean = enst_id.split('.')[0]
    if clean in _biotype_cache:
        return _biotype_cache[clean]
    try:
        r = requests.get(
            f'https://rest.ensembl.org/lookup/id/{clean}',
            headers={'Content-Type': 'application/json'},
            params={'expand': '0'},
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            bt = data.get('biotype', 'unknown')
            _biotype_cache[clean] = bt
            return bt
    except Exception:
        pass
    _biotype_cache[clean] = 'unknown'
    return 'unknown'


# ─── Novel gene candidate extraction ───────────────────────────────────────
def find_novel_gene_candidates(go_term, scores, y_te, pos_set, top_n=TOP_N):
    """
    label=0 isoform 중 score > NOVEL_THRESH → novel gene 후보
    gene당 최대 1개 (최고 점수 isoform 선택)
    """
    candidates = []
    for i in range(N):
        if y_te[i] == 0 and scores[i] >= NOVEL_THRESH:
            sym = X_te_sym[i]
            # 해당 gene의 다른 GO 주석 확인
            other_go = gene_go_map.get(sym, set()) - {go_term}
            candidates.append({
                'iso_id': X_te_isoid[i],
                'gene_symbol': sym,
                'ensg': X_te_genebase[i],
                'go_term': go_term,
                'v10b_score': float(scores[i]),
                'label': 0,
                'candidate_type': 'novel_gene',
                'gene_annotated_for_go': sym in pos_set,
                'other_go_count': len(other_go),
                'other_go_sample': ','.join(sorted(other_go)[:3]),
            })

    if not candidates:
        return []

    # gene당 최고 점수 isoform만 유지
    df = pd.DataFrame(candidates)
    df = df.sort_values('v10b_score', ascending=False)
    df = df.drop_duplicates(subset='gene_symbol', keep='first')
    df = df.head(top_n)
    return df.to_dict('records')


# ─── Isoform-switch candidate extraction ───────────────────────────────────
def find_isoform_switch_candidates(go_term, scores, y_te, pos_set, top_n=TOP_N):
    """
    label=1 gene에서 within-gene score range > SWITCH_THRESH → isoform-switch 후보
    가장 큰 range를 가진 유전자 선택
    """
    # 양성 유전자별 isoform index 그룹화
    gene_idx = {}
    for i in range(N):
        if y_te[i] == 1:
            sym = X_te_sym[i]
            gene_idx.setdefault(sym, []).append(i)

    switch_candidates = []
    for sym, idxs in gene_idx.items():
        if len(idxs) < 2:
            continue
        gene_scores = [scores[i] for i in idxs]
        score_range = max(gene_scores) - min(gene_scores)
        if score_range < SWITCH_THRESH:
            continue

        # 최고/최저 isoform 모두 기록
        sorted_idxs = sorted(idxs, key=lambda i: -scores[i])
        top_iso   = X_te_isoid[sorted_idxs[0]]
        bot_iso   = X_te_isoid[sorted_idxs[-1]]
        top_score = float(scores[sorted_idxs[0]])
        bot_score = float(scores[sorted_idxs[-1]])

        switch_candidates.append({
            'gene_symbol': sym,
            'ensg': X_te_genebase[sorted_idxs[0]],
            'go_term': go_term,
            'n_isoforms': len(idxs),
            'score_range': float(score_range),
            'top_iso': top_iso,
            'top_score': top_score,
            'bot_iso': bot_iso,
            'bot_score': bot_score,
            'ratio': float(top_score / max(bot_score, 1e-6)),
            'all_scores': sorted([float(scores[i]) for i in idxs], reverse=True),
            'candidate_type': 'isoform_switch',
        })

    if not switch_candidates:
        return []

    df = pd.DataFrame(switch_candidates)
    df = df.sort_values('score_range', ascending=False)
    df = df.head(top_n)
    return df.to_dict('records')


# ─── Main ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" Training v10-B per GO term ...")
print("=" * 65)

all_novel   = []
all_switch  = []
summary     = {}

for go in GO_TERMS:
    print(f"\n[{go}]")
    scores, y_te, pos_set = train_v10b(go)
    if scores is None:
        continue

    # A: Novel gene candidates
    novel = find_novel_gene_candidates(go, scores, y_te, pos_set)
    all_novel.extend(novel)
    print(f"  Novel gene candidates (score>{NOVEL_THRESH}): {len(novel)}")
    for c in novel[:3]:
        print(f"    {c['gene_symbol']} {c['iso_id']}: {c['v10b_score']:.4f} "
              f"[other_GO:{c['other_go_count']}]")

    # B: Isoform-switch candidates
    switch = find_isoform_switch_candidates(go, scores, y_te, pos_set)
    all_switch.extend(switch)
    print(f"  Isoform-switch (range>{SWITCH_THRESH}): {len(switch)}")
    for c in switch[:3]:
        print(f"    {c['gene_symbol']}: top={c['top_score']:.3f}({c['top_iso']}) "
              f"bot={c['bot_score']:.3f}({c['bot_iso']}) ratio={c['ratio']:.1f}x")

    summary[go] = {
        'n_novel': len(novel),
        'n_switch': len(switch),
        'novel_thresh': NOVEL_THRESH,
        'switch_thresh': SWITCH_THRESH,
    }

# ─── Biotype enrichment for top novel candidates (API 호출) ──────────────
print("\n" + "=" * 65)
print(" Fetching biotype for top novel candidates ...")
print("=" * 65)

# Score 상위 30개에만 biotype 조회 (API 비용 최소화)
top_novel_df = pd.DataFrame(all_novel).sort_values('v10b_score', ascending=False).head(30) if all_novel else pd.DataFrame()
if not top_novel_df.empty:
    for idx, row in top_novel_df.iterrows():
        bt = get_biotype(row['iso_id'])
        # all_novel 리스트 내 동일 iso_id 업데이트
        for c in all_novel:
            if c['iso_id'] == row['iso_id']:
                c['biotype'] = bt
        time.sleep(0.15)  # Ensembl rate limit

# biotype 없는 항목 채우기
for c in all_novel:
    c.setdefault('biotype', 'not_fetched')

# ─── Save results ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" Saving results ...")
print("=" * 65)

# TSV: novel gene candidates
novel_path = os.path.join(OUT_DIR, 'novel_candidates.tsv')
if all_novel:
    df_novel = pd.DataFrame(all_novel)
    df_novel = df_novel.sort_values('v10b_score', ascending=False)
    df_novel.to_csv(novel_path, sep='\t', index=False)
    print(f"  Novel candidates: {len(all_novel)} → {novel_path}")
else:
    print("  No novel candidates found.")

# TSV: isoform-switch candidates
switch_path = os.path.join(OUT_DIR, 'isoform_switch.tsv')
if all_switch:
    df_switch = pd.DataFrame(all_switch)
    df_switch = df_switch.drop(columns=['all_scores'])  # JSON 컬럼 제거
    df_switch = df_switch.sort_values('score_range', ascending=False)
    df_switch.to_csv(switch_path, sep='\t', index=False)
    print(f"  Isoform-switch: {len(all_switch)} → {switch_path}")
else:
    print("  No isoform-switch candidates found.")

# JSON summary
summary_path = os.path.join(OUT_DIR, 'phase5_summary.json')
with open(summary_path, 'w') as f:
    json.dump({
        'timestamp': ts,
        'novel_thresh': NOVEL_THRESH,
        'switch_thresh': SWITCH_THRESH,
        'total_novel': len(all_novel),
        'total_switch': len(all_switch),
        'per_go': summary,
    }, f, indent=2)
print(f"  Summary → {summary_path}")

# 문헌 검색용 텍스트 요약
report_path = os.path.join(OUT_DIR, 'top_candidates.txt')
with open(report_path, 'w') as f:
    f.write("Phase 5 Novel Isoform Discovery — Top Candidates for Literature Search\n")
    f.write("=" * 70 + "\n\n")

    f.write("[ A. Novel Gene Candidates (score > {:.1f}) ]\n\n".format(NOVEL_THRESH))
    if all_novel:
        df_n = pd.DataFrame(all_novel).sort_values('v10b_score', ascending=False)
        # protein-coding만 우선 출력 (biotype 확인된 경우)
        coding_first = pd.concat([
            df_n[df_n['biotype'] == 'protein_coding'],
            df_n[df_n['biotype'] != 'protein_coding'],
        ]).head(30)
        for _, row in coding_first.iterrows():
            f.write(f"  [{row['go_term']}] {row['gene_symbol']} ({row['iso_id']})\n")
            f.write(f"    score={row['v10b_score']:.4f}, biotype={row['biotype']}, "
                    f"other_GO={row['other_go_count']}\n")
            f.write(f"    → Search: \"{row['gene_symbol']} isoform {row['go_term']} function\"\n\n")
    else:
        f.write("  (없음)\n\n")

    f.write("\n[ B. Isoform-Switch Candidates (score range > {:.1f}) ]\n\n".format(SWITCH_THRESH))
    if all_switch:
        df_s = pd.DataFrame(all_switch).sort_values('ratio', ascending=False).head(30)
        for _, row in df_s.iterrows():
            f.write(f"  [{row['go_term']}] {row['gene_symbol']} ({row['n_isoforms']} isoforms)\n")
            f.write(f"    top: {row['top_iso']} score={row['top_score']:.4f}\n")
            f.write(f"    bot: {row['bot_iso']} score={row['bot_score']:.4f}\n")
            f.write(f"    range={row['score_range']:.3f}, ratio={row['ratio']:.1f}x\n")
            f.write(f"    → Search: \"{row['gene_symbol']} dominant-negative isoform OR "
                    f"tissue-specific isoform {row['go_term']}\"\n\n")
    else:
        f.write("  (없음)\n\n")

print(f"  Literature search report → {report_path}")

# ─── Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" Phase 5 Complete")
print("=" * 65)
print(f"  Novel gene candidates:   {len(all_novel)}")
print(f"  Isoform-switch cases:    {len(all_switch)}")
if all_novel:
    top5 = sorted(all_novel, key=lambda x: -x['v10b_score'])[:5]
    print("\n  Top 5 novel gene candidates:")
    for c in top5:
        print(f"    [{c['go_term']}] {c['gene_symbol']} "
              f"score={c['v10b_score']:.4f} biotype={c.get('biotype','?')}")
if all_switch:
    top5s = sorted(all_switch, key=lambda x: -x['score_range'])[:5]
    print("\n  Top 5 isoform-switch cases:")
    for c in top5s:
        print(f"    [{c['go_term']}] {c['gene_symbol']} "
              f"range={c['score_range']:.3f} ratio={c['ratio']:.1f}x")
print(f"\n  Output: {OUT_DIR}/")
