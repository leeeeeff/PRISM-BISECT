"""
v11_slim_canonical.py — ESM-2 Only Backbone + Canonical Auxiliary
==================================================================
진단 결과 (2026-05-18):
  v11(λ=0.2) AUPRC 붕괴 원인 = domain matrix(512d sparse) 추가
  λ=0 진단: Ca2+ signaling λ=0=0.2959 ≈ λ=0.2=0.2848  → canonical aux 자체는 영향 미미
  → domain/splice 제거, ESM-2 only backbone 복원

Architecture:
  Input:    ESM-2 (640d) only
  Backbone: Dense(256→BN→Drop0.3→128) [v10-B와 동일]
  GO head:        Dropout(0.2) → Dense(64→1, sigmoid)
  Canonical head: Dense(32→1, sigmoid)
  Loss = focal(GO,γ=2) + λ × focal(canonical,γ=2),  λ=0.1

Phase 1 확장 설계:
  canonical head는 Phase 1에서 splice_delta(150d→32d)를 추가 입력으로
  받도록 설계됨. GO head 경로는 변경 없음.
  [현재] backbone(128d) → Canonical Dense(32→1)
  [Phase1] backbone(128d) + splice(→32d) → Concat → Dense(32→1)

비교 기준:
  v10-B: Macro AUPRC=0.6935, mean bias=0.0938
  v11(λ=0.2, multi-modal): Macro AUPRC=0.3536, mean bias=0.1660
"""

import os, sys, json, time, gzip
from collections import defaultdict
import numpy as np
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
tf.get_logger().setLevel('ERROR')
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        target = gpus[1] if len(gpus) > 1 else gpus[0]
        tf.config.set_visible_devices(target, 'GPU')
        print(f"  GPU: {target.name}")
    except RuntimeError as e:
        print(f"  GPU config: {e}")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
OUT_DIR   = '../../reports/v11_slim'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
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

V10B_AUPRC = {
    'GO:0007204': 0.7761, 'GO:0030017': 0.7802, 'GO:0006941': 0.6407,
    'GO:0006914': 0.6964, 'GO:0043161': 0.7405, 'GO:0007519': 0.7035,
    'GO:0042692': 0.6733, 'GO:0055074': 0.7436, 'GO:0007005': 0.6989,
    'GO:0007517': 0.7299, 'GO:0032006': 0.6115, 'GO:0003774': 0.8198,
    'GO:0006096': 0.7938,
}
V10B_BIAS = {
    'GO:0007204': 0.0856, 'GO:0030017': 0.0573, 'GO:0006941': 0.0868,
    'GO:0006914': 0.1096, 'GO:0043161': 0.1042, 'GO:0007519': 0.1239,
    'GO:0042692': 0.1041, 'GO:0055074': 0.0717, 'GO:0007005': 0.1190,
    'GO:0007517': 0.0797, 'GO:0032006': 0.0912, 'GO:0003774': 0.0897,
    'GO:0006096': 0.0965,
}

MANE_FILE  = '../data/MANE_summary.txt.gz'
SEEDS      = [42, 123, 456, 789, 1024]
N_BOOT     = 500
LAMBDA_AUX = 0.1      # v11=0.2에서 줄임 (domain 제거로 canonical gradient 비중 조정)
PATIENCE   = 10

# ─── Utilities ────────────────────────────────────────────────────────────────
def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

def compute_bias_score(scores, gene_ids):
    global_std = scores.std()
    if global_std < 1e-10:
        return np.nan
    g2i = defaultdict(list)
    for i, g in enumerate(gene_ids):
        g2i[g].append(i)
    within_stds = [scores[idxs].std() for idxs in g2i.values() if len(idxs) >= 2]
    return float(np.mean(within_stds) / global_std) if within_stds else np.nan

def bootstrap_ci(y_true, y_score, gene_ids, n_boot=N_BOOT, seed=42):
    rng = np.random.RandomState(seed)
    unique_genes = np.unique(gene_ids)
    base = float(average_precision_score(y_true, y_score))
    g2i = defaultdict(list)
    for i, g in enumerate(gene_ids):
        g2i[g].append(i)
    g2i = {g: np.array(idxs) for g, idxs in g2i.items()}
    boot = []
    for _ in range(n_boot):
        gs = rng.choice(unique_genes, size=len(unique_genes), replace=True)
        idx = np.concatenate([g2i[g] for g in gs])
        if y_true[idx].sum() == 0:
            continue
        boot.append(average_precision_score(y_true[idx], y_score[idx]))
    boot = np.array(boot)
    return base, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def interpret_bias(b):
    if np.isnan(b): return 'N/A'
    if b >= 0.15:   return 'ISOFORM-SPEC'
    if b >= 0.10:   return 'MIXED'
    return 'GENE-LEVEL'

# ─── Data Loading ─────────────────────────────────────────────────────────────
print("=" * 65)
print(" v11-Slim — Loading features (ESM-2 only)")
print("=" * 65)

X_tr_esm = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)
X_te_esm = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)

te_isoid  = load_ids('my_isoform_list_fixed.npy')
te_geneid = load_ids('my_gene_list_fixed.npy')
tr_isoid  = load_ids(f'{ID_DIR}/train_isoform_list.npy')
tr_geneid = load_ids(f'{ID_DIR}/train_gene_list.npy')

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
te_genes    = np.array(te_genebase)

print(f"  Train: ESM-2 {X_tr_esm.shape}")
print(f"  Test:  ESM-2 {X_te_esm.shape}")

# ─── Canonical Labels (동일 우선순위: MANE Select → max domain → skip) ────────
print("\n[Canonical Labels]")

# Test: canonical_reference.tsv (MANE Select 87%)
canon_ref = {}
with open(f'{FEAT_DIR}/canonical_reference.tsv') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            try:
                canon_ref[parts[0].split('.')[0]] = int(parts[3])
            except (ValueError, IndexError):
                pass

y_te_canon   = np.full(len(te_isoid), -1.0, dtype=np.float32)
y_te_canon_w = np.zeros(len(te_isoid), dtype=np.float32)
te_gene2idxs = defaultdict(list)
for i, g in enumerate(te_genebase):
    te_gene2idxs[g].append(i)
for gene_b, can_idx in canon_ref.items():
    idxs = te_gene2idxs.get(gene_b, [])
    if len(idxs) < 2 or can_idx not in range(len(te_isoid)):
        continue
    for i in idxs:
        y_te_canon[i] = 1.0 if i == can_idx else 0.0
        y_te_canon_w[i] = 1.0

te_n_labeled = int((y_te_canon_w > 0).sum())
te_n_pos = int((y_te_canon == 1.0).sum())
print(f"  Test:  {te_n_labeled} labeled ({te_n_pos} canonical, {te_n_labeled-te_n_pos} alternative)")

# Train: MANE Select NM_ (P1) → max domain count (P2) → skip
mane_sym_to_nm = {}
if os.path.exists(MANE_FILE):
    with gzip.open(MANE_FILE, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.strip().split('\t')
            if len(p) < 10:
                continue
            sym, nm_ver, status = p[3], p[5], p[9]
            if status == 'MANE Select':
                mane_sym_to_nm[sym] = nm_ver.split('.')[0]
    print(f"  MANE Select entries loaded: {len(mane_sym_to_nm)}")

# domain count를 proxy로 사용 (P2 fallback — 파일 없으면 skip)
X_tr_dm_proxy = None
dm_path = f'{FEAT_DIR}/domain_matrix_proper_train.npy'
if os.path.exists(dm_path):
    X_tr_dm_proxy = np.load(dm_path).astype(np.float32)

tr_iso_base = [nm.split('.')[0] for nm in tr_isoid]
gene_to_nms = defaultdict(list)
for i, (iso_b, sym) in enumerate(zip(tr_iso_base, tr_sym)):
    n_dom = int((X_tr_dm_proxy[i] > 0).sum()) if X_tr_dm_proxy is not None else 0
    gene_to_nms[sym].append((iso_b, i, n_dom))

y_tr_canon   = np.full(len(tr_isoid), -1.0, dtype=np.float32)
y_tr_canon_w = np.zeros(len(tr_isoid), dtype=np.float32)
tr_src = {'MANE': 0, 'domain': 0, 'skip': 0}

for sym, iso_list in gene_to_nms.items():
    if len(iso_list) < 2:
        continue
    nm_bases = [iso_b for iso_b, _, _ in iso_list]
    mane_nm = mane_sym_to_nm.get(sym)
    if mane_nm and mane_nm in nm_bases:
        can_i = iso_list[nm_bases.index(mane_nm)][1]
        src = 'MANE'
    else:
        max_dom = max(n for _, _, n in iso_list)
        if max_dom > 0:
            can_i = max(iso_list, key=lambda x: x[2])[1]
            src = 'domain'
        else:
            tr_src['skip'] += 1
            continue
    tr_src[src] += 1
    for _, i, _ in iso_list:
        y_tr_canon[i] = 1.0 if i == can_i else 0.0
        y_tr_canon_w[i] = 1.0

tr_n_labeled = int((y_tr_canon_w > 0).sum())
tr_n_pos = int((y_tr_canon == 1.0).sum())
print(f"  Train: {tr_n_labeled} labeled ({tr_n_pos} canonical, {tr_n_labeled-tr_n_pos} alternative)")
print(f"    source: MANE={tr_src['MANE']}, domain-proxy={tr_src['domain']}, skip={tr_src['skip']}")

# ─── Label Loading ────────────────────────────────────────────────────────────
def load_go_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te

# ─── Model ────────────────────────────────────────────────────────────────────
ESM_DIM = X_tr_esm.shape[1]  # 640

def build_v11_slim():
    """
    ESM-2 only backbone + canonical auxiliary head.
    GO 경로: ESM-2 → backbone → GO head (v10-B와 동일)
    Canonical 경로: backbone → canonical head
    [Phase 1] canonical head에 splice_delta(150→32)를 Concat으로 추가 예정.
              GO head 경로는 변경 없음.
    """
    inp_esm = layers.Input(shape=(ESM_DIM,), name='esm2')

    # Backbone (v10-B와 동일)
    x = layers.Dense(256, activation='relu')(inp_esm)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    backbone = layers.Dense(128, activation='relu')(x)

    # GO head
    x_go = layers.Dropout(0.2)(backbone)
    x_go = layers.Dense(64, activation='relu')(x_go)
    out_go = layers.Dense(1, activation='sigmoid', name='go_pred')(x_go)

    # Canonical auxiliary head
    # [Phase 1] 여기에 splice_delta branch Concat 추가:
    #   x_splice = Dense(32, relu)(inp_sd) * Dense(32, sigmoid)(inp_hs)  # gated
    #   x_can_in = Concatenate()([backbone, x_splice])  # 128+32=160d
    x_can = layers.Dense(32, activation='relu')(backbone)
    out_can = layers.Dense(1, activation='sigmoid', name='canonical_pred')(x_can)

    return models.Model(inp_esm, [out_go, out_can], name='v11_slim')

def get_cw(y):
    n_pos = max(int(y.sum()), 1)
    return {0: 1.0, 1: int((y == 0).sum()) / n_pos}

def bce_focal(gamma=2.0):
    return tf.keras.losses.BinaryFocalCrossentropy(gamma=gamma, from_logits=False)

# ─── Training ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f" Training: {len(GO_TERMS)} GO terms × {len(SEEDS)} seeds")
print(f" λ_canonical = {LAMBDA_AUX},  patience = {PATIENCE}")
print(f" Architecture: ESM-2 only (domain/splice REMOVED)")
print("=" * 65)

sc_esm = StandardScaler()
sc_esm.fit(X_tr_esm)
X_tr_esm_sc = sc_esm.transform(X_tr_esm).astype(np.float32)
X_te_esm_sc = sc_esm.transform(X_te_esm).astype(np.float32)

ts = time.strftime('%Y%m%d_%H%M')
results = {}
auprc_list, bias_list = [], []

header = (f"{'GO Term':<22} {'v10B':>7} {'vSlim':>7} {'Δ':>6} | "
          f"{'bias_v10B':>9} {'bias_Slim':>9} {'Δ':>6}  Interp")
print(f"\n{header}")
print("-" * 78)

for go_id, go_name in GO_TERMS.items():
    t0 = time.time()
    y_tr, y_te = load_go_labels(go_id)

    if y_tr.sum() == 0 or y_te.sum() == 0:
        print(f"  {go_name}: skipped (no positives)")
        continue

    cw_go = get_cw(y_tr)

    can_labeled_mask = (y_tr_canon_w > 0)
    n_can_pos = int((y_tr_canon[can_labeled_mask] == 1).sum())
    n_can_neg = int((y_tr_canon[can_labeled_mask] == 0).sum())
    can_ratio = n_can_neg / max(n_can_pos, 1)

    all_preds, all_can_preds = [], []

    for seed in SEEDS:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        K.clear_session()

        model = build_v11_slim()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss={'go_pred': bce_focal(gamma=2.0),
                  'canonical_pred': bce_focal(gamma=2.0)},
            loss_weights={'go_pred': 1.0, 'canonical_pred': LAMBDA_AUX},
        )

        cb_list = [
            callbacks.EarlyStopping(monitor='val_go_pred_loss',
                                    patience=PATIENCE, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_go_pred_loss',
                                        factor=0.5, patience=5, min_lr=1e-5, verbose=0),
        ]

        sw_go  = np.where(y_tr == 1, cw_go[1], cw_go[0]).astype(np.float32)
        sw_can = y_tr_canon_w.copy()
        sw_can = np.where(y_tr_canon == 1.0, sw_can * can_ratio, sw_can)

        model.fit(
            x=X_tr_esm_sc,
            y={'go_pred': y_tr, 'canonical_pred': np.clip(y_tr_canon, 0, 1)},
            sample_weight={'go_pred': sw_go, 'canonical_pred': sw_can},
            epochs=150,
            batch_size=256,
            validation_split=0.1,
            callbacks=cb_list,
            verbose=0,
        )

        preds = model.predict(X_te_esm_sc, verbose=0, batch_size=512)
        all_preds.append(preds[0].flatten())
        all_can_preds.append(preds[1].flatten())

    slim_scores = np.mean(all_preds, axis=0)
    can_scores  = np.mean(all_can_preds, axis=0)

    auprc    = float(average_precision_score(y_te, slim_scores))
    auprc_ci = bootstrap_ci(y_te, slim_scores, te_genes)
    bias     = compute_bias_score(slim_scores, te_genebase)

    can_mask = y_te_canon_w > 0
    can_auc  = float(roc_auc_score(y_te_canon[can_mask], can_scores[can_mask])) \
               if can_mask.sum() > 10 else float('nan')

    elapsed = time.time() - t0
    d_auprc = auprc - V10B_AUPRC.get(go_id, 0)
    d_bias  = bias  - V10B_BIAS.get(go_id, 0)
    interp  = interpret_bias(bias)

    print(f"{go_name:<22} {V10B_AUPRC.get(go_id,0):>7.4f} {auprc:>7.4f} {d_auprc:>+6.4f}"
          f" | {V10B_BIAS.get(go_id,0):>9.4f} {bias:>9.4f} {d_bias:>+6.4f}  {interp:<14}"
          f" (can_AUC={can_auc:.3f}, {elapsed:.0f}s)")
    sys.stdout.flush()

    results[go_id] = {
        'go_name':       go_name,
        'n_pos_train':   int(y_tr.sum()),
        'n_pos_test':    int(y_te.sum()),
        'auprc_slim':    round(auprc, 4),
        'auprc_ci_lo':   round(auprc_ci[1], 4),
        'auprc_ci_hi':   round(auprc_ci[2], 4),
        'auprc_v10b':    V10B_AUPRC.get(go_id),
        'delta_auprc':   round(d_auprc, 4),
        'bias_slim':     round(bias, 4),
        'bias_v10b':     V10B_BIAS.get(go_id),
        'delta_bias':    round(d_bias, 4),
        'bias_interp':   interp,
        'canonical_auc': round(can_auc, 4),
        'elapsed_s':     round(elapsed, 1),
    }
    auprc_list.append(auprc)
    bias_list.append(bias)

# ─── Summary ──────────────────────────────────────────────────────────────────
macro_slim  = float(np.mean(auprc_list))
macro_v10b  = float(np.mean(list(V10B_AUPRC.values())))
mean_bias_slim  = float(np.nanmean(bias_list))
mean_bias_v10b  = float(np.mean(list(V10B_BIAS.values())))
n_isospec = sum(1 for b in bias_list if b >= 0.15)
n_mixed   = sum(1 for b in bias_list if 0.10 <= b < 0.15)
n_genelev = sum(1 for b in bias_list if b < 0.10)

print("\n" + "=" * 78)
print(f"  Macro AUPRC:  v10-B={macro_v10b:.4f}  v11-Slim={macro_slim:.4f}  "
      f"Δ={macro_slim-macro_v10b:+.4f}")
print(f"  Mean bias:    v10-B={mean_bias_v10b:.4f}  v11-Slim={mean_bias_slim:.4f}  "
      f"Δ={mean_bias_slim-mean_bias_v10b:+.4f}")
print(f"  Bias interp:  ISOFORM-SPEC={n_isospec}  MIXED={n_mixed}  GENE-LEVEL={n_genelev}")
print("=" * 78)

# ─── Save ─────────────────────────────────────────────────────────────────────
out = {
    'results': results,
    'summary': {
        'macro_auprc_slim':   round(macro_slim, 4),
        'macro_auprc_v10b':   round(macro_v10b, 4),
        'delta_macro_auprc':  round(macro_slim - macro_v10b, 4),
        'mean_bias_slim':     round(mean_bias_slim, 4),
        'mean_bias_v10b':     round(mean_bias_v10b, 4),
        'delta_mean_bias':    round(mean_bias_slim - mean_bias_v10b, 4),
        'n_isospec':          n_isospec,
        'n_mixed':            n_mixed,
        'n_genelevel':        n_genelev,
        'lambda_aux':         LAMBDA_AUX,
        'seeds':              SEEDS,
        'n_go_terms':         len(results),
        'architecture':       'ESM-2 only, no domain, no splice',
    },
    'timestamp': ts,
    'strategy':  'v11-Slim: ESM-2-only + Canonical-Auxiliary',
}

out_path = f'{OUT_DIR}/v11_slim_{ts}.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved → {out_path}")

# ─── 판정 ─────────────────────────────────────────────────────────────────────
print("\n[v11-Slim 효과 판정]")
auprc_ok = macro_slim >= macro_v10b - 0.02
bias_ok  = mean_bias_slim > mean_bias_v10b + 0.01
if auprc_ok and bias_ok:
    print(f"  PASS: AUPRC 유지 ({macro_slim:.4f} vs {macro_v10b:.4f}) + bias 개선 ({mean_bias_slim:.4f})")
    print("  → Phase 1 (real splice delta → canonical head 추가) 진입 권장")
elif bias_ok and not auprc_ok:
    print(f"  TRADE-OFF: bias 개선 ({mean_bias_slim:.4f}) but AUPRC Δ={macro_slim-macro_v10b:+.4f}")
    print("  → λ 감소 또는 backbone 분리 검토")
elif auprc_ok and not bias_ok:
    print(f"  AUPRC 유지 but bias 개선 미미 ({mean_bias_slim:.4f})")
    print("  → λ 증가 (0.1→0.2) 재시도")
else:
    print(f"  FAIL: AUPRC Δ={macro_slim-macro_v10b:+.4f}, bias Δ={mean_bias_slim-mean_bias_v10b:+.4f}")
    print("  → 아키텍처 재검토 필요")
