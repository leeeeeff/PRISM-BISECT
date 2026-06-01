"""
v10_sarcopenia_novel.py — Phase 2: Sarcopenia Novel Isoform Discovery
======================================================================
목적: 근감소증 핵심 GO term 3개에 대해 v10-B 고점수 isoform 후보 발굴.

대상 GO terms:
  GO:0006914  Autophagy          (5-seed ensemble, Δ=+0.354 ***)
  GO:0032006  TOR signaling      (5-seed ensemble, Δ=+0.092 n.s.)
  GO:0007005  Mitochondrion org  (5-seed ensemble, Δ=+0.424 ***)

후보 유형:
  A. Isoform-switch: 양성 유전자 내 score_range > 0.30 → 특정 isoform만 기능적
  B. Novel gene:     음성 유전자 isoform score > 0.60 → 미주석 기능 예측

근감소증 핵심 후보 유전자 하이라이트:
  Autophagy  : ATG7, ULK1, BECN1, ATG5, ATG12, SQSTM1, ATG14, RB1CC1
  TOR        : RPTOR, RICTOR, DEPTOR, TSC1, TSC2, AKT1S1, MLST8, RHEBL1
  Mitochondria: FUNDC1, PINK1, PRKN, BNIP3, BNIP3L, TOMM20, MFN1, MFN2, OPA1

출력:
  reports/sarcopenia_novel/{ts}/
    isoform_switch.tsv   — isoform-switch 후보 (score_range 내림차순)
    novel_gene.tsv       — novel gene 후보 (score 내림차순)
    highlight_report.txt — 근감소증 핵심 후보 요약
    summary.json         — GO term별 통계

실행:
  conda activate isoform_env
  python hMuscle/model/v10_sarcopenia_novel.py
"""

import os, sys, json, time
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

# ─── Configuration ─────────────────────────────────────────────────────────
DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
ts = time.strftime('%Y%m%d_%H%M')
OUT_DIR   = f'../../reports/sarcopenia_novel/{ts}'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]
SWITCH_THRESH = 0.30   # within-gene score range
NOVEL_THRESH  = 0.60   # novel gene score cutoff
TOP_N         = 25

GO_TERMS = {
    'GO:0006914': 'Autophagy',
    'GO:0032006': 'TOR signaling',
    'GO:0007005': 'Mitochondrion org',
}

SARCOPENIA_CANDIDATES = {
    'GO:0006914': {'ATG7', 'ULK1', 'BECN1', 'ATG5', 'ATG12', 'SQSTM1',
                   'ATG14', 'RB1CC1', 'ATG101', 'PIK3C3', 'AMBRA1',
                   'ATG3', 'ATG10', 'ATG4A', 'ATG4B', 'MAP1LC3A', 'GABARAPL1'},
    'GO:0032006': {'RPTOR', 'RICTOR', 'DEPTOR', 'TSC1', 'TSC2', 'AKT1S1',
                   'MLST8', 'RHEBL1', 'PRAS40', 'MTOR', 'LAMTOR1', 'LAMTOR2',
                   'RPS6KB1', 'EIF4EBP1', 'DDIT4', 'RRAGC', 'RRAGD'},
    'GO:0007005': {'FUNDC1', 'PINK1', 'PRKN', 'BNIP3', 'BNIP3L',
                   'TOMM20', 'MFN1', 'MFN2', 'OPA1', 'DRP1', 'FIS1',
                   'MFN2', 'NIPSNAP1', 'CALCOCO2', 'OPTN', 'TAX1BP1'},
}

# ─── Load data ──────────────────────────────────────────────────────────────
print("=" * 65)
print(" Sarcopenia Novel Isoform Discovery — Loading data ...")
print("=" * 65)

X_te_esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
X_tr_esm2 = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)

def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

X_te_isoid  = load_ids('my_isoform_list_fixed.npy')
X_te_geneid = load_ids('my_gene_list_fixed.npy')
X_tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

X_te_sym      = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in X_te_geneid]
X_te_genebase = [g.split('.')[0] for g in X_te_geneid]
N = len(X_te_isoid)

# 전체 GO 주석 사전 (gene_symbol → GO terms)
gene_go_map = {}
with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > 1:
            gene_go_map[parts[0]] = set(parts[1:])

print(f"  Test isoforms: {N},  Train genes: {len(X_tr_geneid)}")


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


def train_seed(seed, go_term, y_tr, y_te):
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_esm2)
    X_te_sc = sc.transform(X_te_esm2)

    rng = np.random.RandomState(seed)
    n_val = max(int(len(y_tr) * 0.1), 100)
    vi = rng.choice(len(y_tr), size=n_val, replace=False)
    ti = np.setdiff1d(np.arange(len(y_tr)), vi)

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

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
              epochs=100, batch_size=512,
              class_weight=get_cw(y_tr),
              callbacks=cb_list, verbose=0)

    probs = model.predict(X_te_sc, verbose=0).ravel()
    auprc = float(average_precision_score(y_te, probs))
    return probs, auprc


def get_ensemble_scores(go_term):
    y_tr, y_te, pos_set = load_labels(go_term)
    if y_tr.sum() < 2 or y_te.sum() == 0:
        return None, None, None

    seed_scores, seed_auprcs = [], []
    for seed in SEEDS:
        probs, auprc = train_seed(seed, go_term, y_tr, y_te)
        seed_scores.append(probs)
        seed_auprcs.append(auprc)
        print(f"    seed={seed}: AUPRC={auprc:.4f}")

    ensemble = np.mean(seed_scores, axis=0)
    ens_auprc = float(average_precision_score(y_te, ensemble))
    print(f"    Ensemble AUPRC={ens_auprc:.4f}  mean±std={np.mean(seed_auprcs):.4f}±{np.std(seed_auprcs):.4f}")
    return ensemble, y_te, pos_set


# ─── Candidate extraction ───────────────────────────────────────────────────
def find_isoform_switch(go_term, scores, y_te, pos_set, highlight_genes):
    gene_idx = {}
    for i in range(N):
        if y_te[i] == 1:
            sym = X_te_sym[i]
            gene_idx.setdefault(sym, []).append(i)

    results = []
    for sym, idxs in gene_idx.items():
        if len(idxs) < 2:
            continue
        gene_scores = [scores[i] for i in idxs]
        score_range = max(gene_scores) - min(gene_scores)
        if score_range < SWITCH_THRESH:
            continue

        sorted_idxs = sorted(idxs, key=lambda i: -scores[i])
        top_i = sorted_idxs[0]
        bot_i = sorted_idxs[-1]

        results.append({
            'go_term': go_term,
            'go_name': GO_TERMS[go_term],
            'gene_symbol': sym,
            'n_isoforms': len(idxs),
            'score_range': float(score_range),
            'top_iso': X_te_isoid[top_i],
            'top_score': float(scores[top_i]),
            'bot_iso': X_te_isoid[bot_i],
            'bot_score': float(scores[bot_i]),
            'ratio': float(scores[top_i] / max(scores[bot_i], 1e-6)),
            'is_sarcopenia_candidate': sym in highlight_genes,
            'all_isoform_scores': sorted([float(scores[i]) for i in idxs], reverse=True),
        })

    df = pd.DataFrame(results) if results else pd.DataFrame()
    if not df.empty:
        df = df.sort_values('score_range', ascending=False)
    return df


def find_novel_genes(go_term, scores, y_te, pos_set, highlight_genes):
    results = []
    seen_genes = set()
    for i in range(N):
        if y_te[i] == 1:
            continue
        sym = X_te_sym[i]
        if sym in seen_genes:
            continue
        if float(scores[i]) < NOVEL_THRESH:
            continue
        # take max-scoring isoform per gene
        results.append({
            'go_term': go_term,
            'go_name': GO_TERMS[go_term],
            'gene_symbol': sym,
            'iso_id': X_te_isoid[i],
            'score': float(scores[i]),
            'is_sarcopenia_candidate': sym in highlight_genes,
            'other_go_count': len(gene_go_map.get(sym, set())),
            'other_go_sample': ','.join(sorted(gene_go_map.get(sym, set()))[:3]),
        })
        seen_genes.add(sym)

    df = pd.DataFrame(results) if results else pd.DataFrame()
    if not df.empty:
        # per gene: keep highest scoring isoform
        df = df.sort_values('score', ascending=False).head(TOP_N)
    return df


# ─── Main ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" Training v10-B (5 seeds) per GO term ...")
print("=" * 65)

all_switch = []
all_novel  = []
summary    = {}

for go, go_name in GO_TERMS.items():
    print(f"\n[{go}] {go_name}")
    highlight = SARCOPENIA_CANDIDATES.get(go, set())

    ensemble, y_te, pos_set = get_ensemble_scores(go)
    if ensemble is None:
        print("  SKIP: insufficient labels")
        continue

    # A: Isoform-switch
    df_switch = find_isoform_switch(go, ensemble, y_te, pos_set, highlight)
    all_switch.append(df_switch)
    n_switch = len(df_switch)
    n_switch_highlight = len(df_switch[df_switch['is_sarcopenia_candidate']]) if not df_switch.empty else 0
    print(f"  Isoform-switch (range>{SWITCH_THRESH}): {n_switch}, sarcopenia_candidate: {n_switch_highlight}")
    if not df_switch.empty:
        for _, row in df_switch.head(5).iterrows():
            flag = ' <<SARCOPENIA>>' if row['is_sarcopenia_candidate'] else ''
            print(f"    {row['gene_symbol']}: range={row['score_range']:.3f} "
                  f"top={row['top_score']:.3f} bot={row['bot_score']:.3f} "
                  f"ratio={row['ratio']:.1f}x{flag}")

    # B: Novel gene
    df_novel = find_novel_genes(go, ensemble, y_te, pos_set, highlight)
    all_novel.append(df_novel)
    n_novel = len(df_novel)
    n_novel_highlight = len(df_novel[df_novel['is_sarcopenia_candidate']]) if not df_novel.empty else 0
    print(f"  Novel gene (score>{NOVEL_THRESH}): {n_novel}, sarcopenia_candidate: {n_novel_highlight}")
    if not df_novel.empty:
        for _, row in df_novel.head(5).iterrows():
            flag = ' <<SARCOPENIA>>' if row['is_sarcopenia_candidate'] else ''
            print(f"    {row['gene_symbol']} {row['iso_id']}: {row['score']:.4f}{flag}")

    summary[go] = {
        'go_name': go_name,
        'n_switch': n_switch,
        'n_switch_sarco': n_switch_highlight,
        'n_novel': n_novel,
        'n_novel_sarco': n_novel_highlight,
    }

# ─── Save results ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f" Saving results to {OUT_DIR} ...")
print("=" * 65)

if all_switch:
    df_sw_all = pd.concat([df for df in all_switch if not df.empty], ignore_index=True)
    df_sw_all.to_csv(f'{OUT_DIR}/isoform_switch.tsv', sep='\t', index=False)
    print(f"  isoform_switch.tsv: {len(df_sw_all)} rows")

if all_novel:
    df_nv_all = pd.concat([df for df in all_novel if not df.empty], ignore_index=True)
    df_nv_all.to_csv(f'{OUT_DIR}/novel_gene.tsv', sep='\t', index=False)
    print(f"  novel_gene.tsv: {len(df_nv_all)} rows")

# Highlight report
with open(f'{OUT_DIR}/highlight_report.txt', 'w') as fh:
    fh.write("SARCOPENIA NOVEL ISOFORM DISCOVERY — Highlight Report\n")
    fh.write(f"Generated: {ts}\n\n")

    for go, go_name in GO_TERMS.items():
        fh.write(f"\n{'='*60}\n")
        fh.write(f"{go} — {go_name}\n")
        fh.write(f"{'='*60}\n")

        if all_switch:
            sw_df = pd.concat([df for df in all_switch if not df.empty], ignore_index=True)
            sw_go = sw_df[sw_df['go_term'] == go] if not sw_df.empty else pd.DataFrame()
            if not sw_go.empty:
                fh.write("\nISOFORM-SWITCH Candidates (top 10):\n")
                for _, row in sw_go.head(10).iterrows():
                    flag = ' [SARCOPENIA_CANDIDATE]' if row['is_sarcopenia_candidate'] else ''
                    fh.write(f"  {row['gene_symbol']}: range={row['score_range']:.3f} "
                             f"  top={row['top_iso']} ({row['top_score']:.3f})"
                             f"  bot={row['bot_iso']} ({row['bot_score']:.3f})"
                             f"  ratio={row['ratio']:.1f}x{flag}\n")

        if all_novel:
            nv_df = pd.concat([df for df in all_novel if not df.empty], ignore_index=True)
            nv_go = nv_df[nv_df['go_term'] == go] if not nv_df.empty else pd.DataFrame()
            if not nv_go.empty:
                fh.write("\nNOVEL GENE Candidates (top 10):\n")
                for _, row in nv_go.head(10).iterrows():
                    flag = ' [SARCOPENIA_CANDIDATE]' if row['is_sarcopenia_candidate'] else ''
                    fh.write(f"  {row['gene_symbol']} {row['iso_id']}: "
                             f"score={row['score']:.4f}{flag}\n")

        # Sarcopenia candidates in switch
        if all_switch:
            sw_df = pd.concat([df for df in all_switch if not df.empty], ignore_index=True)
            if not sw_df.empty:
                sw_sarco = sw_df[(sw_df['go_term'] == go) & (sw_df['is_sarcopenia_candidate'])]
                if not sw_sarco.empty:
                    fh.write(f"\n[PRIORITY] Sarcopenia-candidate isoform switches ({len(sw_sarco)}):\n")
                    for _, row in sw_sarco.iterrows():
                        fh.write(f"  ** {row['gene_symbol']} (range={row['score_range']:.3f}, "
                                 f"ratio={row['ratio']:.1f}x) **\n")
                        fh.write(f"     HIGH: {row['top_iso']} score={row['top_score']:.3f}\n")
                        fh.write(f"     LOW:  {row['bot_iso']} score={row['bot_score']:.3f}\n")

with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump({'summary': summary, 'timestamp': ts, 'thresholds': {
        'switch': SWITCH_THRESH, 'novel': NOVEL_THRESH, 'seeds': SEEDS
    }}, f, indent=2)

print(f"\nFINAL: {OUT_DIR}")
print("=" * 65)
print(" Summary:")
for go, stats in summary.items():
    print(f"  [{go}] {stats['go_name']}")
    print(f"    Isoform-switch: {stats['n_switch']} (sarcopenia: {stats['n_switch_sarco']})")
    print(f"    Novel gene:     {stats['n_novel']} (sarcopenia: {stats['n_novel_sarco']})")
