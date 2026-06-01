"""
v10_phase4_alphafold.py — Phase 4: AlphaFold Structure Validation
=================================================================
목적: v10-B 예측 점수와 AlphaFold pLDDT의 상관관계로 예측의 구조적 타당성 검증.

전략 (ESMFold 미사용, 순수 AlphaFold DB 기반):
1. v10-B를 5개 GO term별 학습 → per-isoform 예측 점수 저장
2. AlphaFold DB API → canonical protein per-residue pLDDT 수집
3. Ensembl REST API → 각 isoform의 단백질 서열 수집 (ENSP 경유)
4. difflib.SequenceMatcher → isoform이 보존하는 canonical 잔기의 projected pLDDT 계산
5. within-gene Spearman(v10-B_score, projected_pLDDT) 검증

target genes:
  GO:0030017 sarcomere → PPP1R12B (6 isoforms, 186~1043 aa)
  GO:0006096 glycolysis → PFKP (4 isoforms), PKM (5 isoforms)
  GO:0003774 motor      → MYH7 (3 isoforms), MYH2 (2 isoforms)

가설:
  H1: 높은 v10-B score isoform → 보존된 잔기의 pLDDT 높음 (기능적 구조 유지)
  H2: 짧은/truncated isoform   → projected pLDDT 낮음 (도메인 상실)
  검증: Spearman r > 0.3, p < 0.05 per gene

실행:
  conda activate isoform_env
  python hMuscle/model/v10_phase4_alphafold.py
"""

import os, sys, json, time, requests
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
STRUCT_DIR = '../results_isoform/structures'
OUT_DIR   = '../../reports/alphafold_validation'

os.makedirs(STRUCT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─── Target Genes (gene_symbol → UniProt canonical, GO_terms) ──────────────
TARGET_GENES = {
    'PPP1R12B': {'uniprot': 'O60237', 'go': 'GO:0030017', 'ensg': 'ENSG00000077157'},
    'PFKP':     {'uniprot': 'Q01813', 'go': 'GO:0006096', 'ensg': 'ENSG00000067057'},
    'PKM':      {'uniprot': 'P14618', 'go': 'GO:0006096', 'ensg': 'ENSG00000067225'},
    'MYH7':     {'uniprot': 'P12883', 'go': 'GO:0003774', 'ensg': 'ENSG00000092054'},
    'MYH2':     {'uniprot': 'P13535', 'go': 'GO:0003774', 'ensg': 'ENSG00000125414'},
    'ACTN2':    {'uniprot': 'P35609', 'go': 'GO:0030017', 'ensg': 'ENSG00000077522'},
}

GO_TERMS = ['GO:0006096', 'GO:0003774', 'GO:0007204', 'GO:0030017', 'GO:0006941']

# ─── Load core data ────────────────────────────────────────────────────────
print("=" * 65)
print(" Phase 4: AlphaFold Validation — Loading data ...")
print("=" * 65)

X_te_esm2  = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
X_tr_esm2  = np.load(f'{DATA_DIR}/esm2_train_human_t30_150M.npy').astype(np.float32)

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

X_te_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in X_te_geneid]
X_te_genebase = [g.split('.')[0] for g in X_te_geneid]
print(f"  Test set: {len(X_te_isoid)} isoforms, {X_te_esm2.shape}")


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
    return y_tr, y_te


def get_cw(y):
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    return {0: 1.0, 1: n_neg / max(n_pos, 1)}


def train_v10b(go_term):
    """v10-B 학습 → 전체 test set 예측 점수 반환"""
    y_tr, y_te = load_labels(go_term)
    if y_tr.sum() < 2 or y_te.sum() == 0:
        return None

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
    print(f"    [{go_term}] AUPRC={auprc:.4f}, n_pos={int(y_te.sum())}")
    return probs


# ─── AlphaFold DB ──────────────────────────────────────────────────────────
AF_BASE = "https://alphafold.ebi.ac.uk"

def fetch_alphafold_plddt(uniprot_id, cache_dir=STRUCT_DIR):
    """AlphaFold DB API → per-residue pLDDT 반환"""
    cache_path = os.path.join(cache_dir, f'{uniprot_id}_plddt.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    r = requests.get(f'{AF_BASE}/api/prediction/{uniprot_id}', timeout=30)
    if r.status_code != 200:
        print(f"    [AF] {uniprot_id}: API error {r.status_code}")
        return None

    meta = r.json()
    if not meta:
        return None
    meta = meta[0]

    pdb_url = meta.get('pdbUrl')
    if not pdb_url:
        return None

    r2 = requests.get(pdb_url, timeout=60)
    if r2.status_code != 200:
        print(f"    [AF] {uniprot_id}: PDB download error {r2.status_code}")
        return None

    plddt_list = []
    for line in r2.text.split('\n'):
        if line.startswith('ATOM') and line[13:15].strip() == 'CA':
            try:
                bfactor = float(line[60:66].strip())
                plddt_list.append(bfactor)
            except ValueError:
                continue

    result = {
        'uniprot': uniprot_id,
        'n_residues': len(plddt_list),
        'plddt': plddt_list,
        'mean_plddt': float(np.mean(plddt_list)) if plddt_list else 0,
    }
    with open(cache_path, 'w') as f:
        json.dump(result, f)
    print(f"    [AF] {uniprot_id}: {len(plddt_list)} residues, "
          f"mean_pLDDT={result['mean_plddt']:.1f}")
    return result


# ─── Ensembl API: ENST → protein sequence ──────────────────────────────────
def get_protein_seq(enst_id, cache={}):
    """ENST ID → (protein_seq, aa_length) via Ensembl REST"""
    if enst_id in cache:
        return cache[enst_id]

    clean = enst_id.split('.')[0]
    try:
        r = requests.get(f'https://rest.ensembl.org/lookup/id/{clean}',
                         headers={'Content-Type': 'application/json'},
                         params={'expand': 1}, timeout=15)
        if r.status_code != 200:
            cache[enst_id] = (None, None); return None, None

        info = r.json()
        ensp_info = info.get('Translation')
        if not ensp_info:
            cache[enst_id] = (None, None); return None, None

        ensp = ensp_info.get('id')
        r2 = requests.get(f'https://rest.ensembl.org/sequence/id/{ensp}',
                          headers={'Content-Type': 'application/json'},
                          params={'type': 'protein'}, timeout=15)
        if r2.status_code != 200:
            cache[enst_id] = (None, None); return None, None

        seq = r2.json().get('seq', '')
        cache[enst_id] = (seq, len(seq))
        return seq, len(seq)
    except Exception as e:
        cache[enst_id] = (None, None); return None, None


def get_canonical_seq(uniprot_id, cache={}):
    """UniProt FASTA → canonical protein sequence"""
    if uniprot_id in cache:
        return cache[uniprot_id]
    try:
        r = requests.get(f'https://www.uniprot.org/uniprot/{uniprot_id}.fasta',
                         timeout=15)
        if r.status_code != 200:
            cache[uniprot_id] = None; return None
        lines = r.text.strip().split('\n')
        seq = ''.join(lines[1:])
        cache[uniprot_id] = seq; return seq
    except Exception:
        cache[uniprot_id] = None; return None


# ─── Projected pLDDT ───────────────────────────────────────────────────────
def compute_projected_plddt(iso_seq, canonical_seq, plddt_values):
    """
    isoform 서열을 canonical에 매핑하여 보존된 잔기의 pLDDT 평균 계산.
    SequenceMatcher로 matching blocks 추출 → canonical 위치의 pLDDT 사용.
    """
    if not iso_seq or not canonical_seq or not plddt_values:
        return np.nan, 0

    sm = SequenceMatcher(None, iso_seq, canonical_seq, autojunk=False)
    blocks = sm.get_matching_blocks()

    retained_plddt = []
    for block in blocks:
        iso_start, can_start, length = block.a, block.b, block.size
        if length == 0:
            continue
        for offset in range(length):
            can_pos = can_start + offset
            if can_pos < len(plddt_values):
                retained_plddt.append(plddt_values[can_pos])

    if not retained_plddt:
        return np.nan, 0

    return float(np.mean(retained_plddt)), len(retained_plddt)


# ─── Per-gene AlphaFold analysis ───────────────────────────────────────────
def analyze_gene(sym, gene_info, all_scores):
    """
    단일 유전자에 대해 isoform별 (v10-B score, projected pLDDT) 쌍 계산.
    """
    uniprot = gene_info['uniprot']
    go_term = gene_info['go']
    ensg    = gene_info['ensg']

    print(f"\n  [{sym}] ({uniprot}) GO={go_term}")

    # AlphaFold pLDDT
    af_data = fetch_alphafold_plddt(uniprot)
    if not af_data:
        print(f"    [SKIP] AlphaFold data unavailable")
        return None

    plddt_values = af_data['plddt']

    # canonical sequence
    can_seq = get_canonical_seq(uniprot)
    if not can_seq:
        print(f"    [SKIP] canonical seq unavailable")
        return None

    # test set에서 해당 gene의 isoform 인덱스
    gene_idxs = [i for i, g in enumerate(X_te_genebase) if ensg in X_te_geneid[i]]
    if not gene_idxs:
        print(f"    [SKIP] no isoforms in test set")
        return None

    unique_isos = {}
    for i in gene_idxs:
        iso_id = X_te_isoid[i]
        if iso_id not in unique_isos:
            unique_isos[iso_id] = i  # first occurrence

    print(f"    {len(unique_isos)} unique isoforms in test set")

    # v10-B scores for this GO term
    if go_term not in all_scores:
        print(f"    [SKIP] {go_term} scores not available")
        return None
    scores = all_scores[go_term]  # array of length len(X_te_isoid)

    # per-isoform analysis
    rows = []
    for iso_id, idx in unique_isos.items():
        v10b_score = float(scores[idx])

        # protein sequence
        if iso_id.startswith('ENST'):
            iso_seq, aa_len = get_protein_seq(iso_id)
        else:
            iso_seq, aa_len = None, None

        if not iso_seq:
            # non-coding or BambuTx: use canonical as proxy (len from ESM-2)
            proj_plddt = np.nan
            n_retained = 0
        else:
            proj_plddt, n_retained = compute_projected_plddt(
                iso_seq, can_seq, plddt_values)

        rows.append({
            'iso_id':       iso_id,
            'gene':         sym,
            'go':           go_term,
            'v10b_score':   v10b_score,
            'aa_len':       aa_len,
            'proj_plddt':   proj_plddt,
            'n_retained':   n_retained,
            'can_len':      len(can_seq),
            'coverage':     n_retained / max(len(can_seq), 1),
        })
        plddt_str = f'{proj_plddt:.1f}' if not (isinstance(proj_plddt, float) and np.isnan(proj_plddt)) else 'N/A'
        print(f"    {iso_id}: score={v10b_score:.4f}, aa={aa_len}, "
              f"proj_pLDDT={plddt_str}, coverage={n_retained}/{len(can_seq)}")

    df = pd.DataFrame(rows)

    # filter: only rows with valid pLDDT and at least 20% coverage
    valid = df.dropna(subset=['proj_plddt'])
    valid = valid[valid['coverage'] >= 0.2]

    if len(valid) < 2:
        print(f"    [Spearman] insufficient data (n={len(valid)})")
        return df

    r, p = spearmanr(valid['v10b_score'], valid['proj_plddt'])
    print(f"    [Spearman] n={len(valid)}, r={r:.3f}, p={p:.4f}")

    # also check score vs coverage
    r_cov, p_cov = spearmanr(valid['v10b_score'], valid['coverage'])
    print(f"    [Spearman vs coverage] r={r_cov:.3f}, p={p_cov:.4f}")

    df['spearman_r'] = r
    df['spearman_p'] = p
    return df


# ─── Main ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" Step 1: Train v10-B per GO term, save per-isoform scores")
print("=" * 65)

all_scores = {}
for go in GO_TERMS:
    print(f"\n  Training v10-B for {go} ...")
    probs = train_v10b(go)
    if probs is not None:
        all_scores[go] = probs

print(f"\n  Saved scores for {len(all_scores)} GO terms")

print("\n" + "=" * 65)
print(" Step 2: AlphaFold pLDDT + sequence alignment per gene")
print("=" * 65)

all_gene_dfs = []
for sym, gene_info in TARGET_GENES.items():
    df = analyze_gene(sym, gene_info, all_scores)
    if df is not None:
        all_gene_dfs.append(df)

if all_gene_dfs:
    full_df = pd.concat(all_gene_dfs, ignore_index=True)

    # ─── Summary Table ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  ALPHAFOLD VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Gene':<12} {'GO Term':<14} {'n_iso':<6} {'Spearman r':<12} {'p-value':<10} {'verdict'}")
    print("-" * 80)

    summary = []
    for sym in TARGET_GENES:
        sub = full_df[full_df['gene'] == sym].copy()
        valid = sub.dropna(subset=['proj_plddt'])
        valid = valid[valid['coverage'] >= 0.2]
        if len(valid) >= 2:
            r, p = spearmanr(valid['v10b_score'], valid['proj_plddt'])
            verdict = ("✅ PASS" if (r > 0.3 and p < 0.05)
                       else "⚠️  marginal" if (r > 0.1)
                       else "❌ FAIL")
            print(f"{sym:<12} {TARGET_GENES[sym]['go']:<14} {len(valid):<6} "
                  f"{r:>10.3f}   {p:>8.4f}   {verdict}")
            summary.append({'gene': sym, 'go': TARGET_GENES[sym]['go'],
                             'n_iso': len(valid), 'spearman_r': float(r),
                             'spearman_p': float(p)})

    # pLDDT thresholds
    print("\n  pLDDT reference: >90=very high, 70-90=confident, 50-70=low, <50=very low")
    print("  Hypothesis: high v10-B score ↔ high projected pLDDT (structural retention)")

    # per-gene isoform table
    print("\n" + "-" * 80)
    print("  Per-isoform details (sorted by gene + score):")
    display_cols = ['gene', 'iso_id', 'aa_len', 'coverage', 'v10b_score', 'proj_plddt']
    print(full_df[display_cols].sort_values(['gene', 'v10b_score'], ascending=[True, False])
          .to_string(index=False, float_format='{:.3f}'.format))

    # ─── Save ────────────────────────────────────────────────────────────
    ts = time.strftime('%Y%m%d_%H%M')
    out_csv = f'{OUT_DIR}/phase4_isoform_plddt_{ts}.csv'
    out_json = f'{OUT_DIR}/phase4_summary_{ts}.json'

    full_df.to_csv(out_csv, index=False)
    with open(out_json, 'w') as f:
        json.dump({'summary': summary, 'timestamp': ts,
                   'n_genes': len(summary),
                   'n_pass': sum(1 for s in summary if s.get('spearman_r', 0) > 0.3)},
                  f, indent=2)
    print(f"\n[Saved] {out_csv}")
    print(f"[Saved] {out_json}")
else:
    print("[WARNING] No gene data collected — check target gene definitions")
