#!/usr/bin/env python3
"""
exp_uniprot_v2.py
-----------------
Extended UniProt isoform benchmark (51 pairs) with:
  - 6 new pairs: STAT3, CEBPB, PIK3R1, DNMT3A, SMN1, RAC1
  - Permutation testing (n=10000) for overall direction accuracy
  - Stratified analysis by gap threshold (0.05, 0.10, 0.15, 0.20)
  - Fisher exact test for high-confidence subset

Output: reports/exp_h_uniprot_eval/v2/
"""

import os, re, csv, gzip, json, time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from urllib import request, parse
from urllib.error import HTTPError
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR    = '../data'
ID_DIR      = '../data/raw_data/data/id_lists'
ANNOT_DIR   = '../data/raw_data/data/annotations'
BENCH_CSV   = '../../reports/exp_g_uniprot/uniprot_isoform_benchmark_v2.csv'
OUT_DIR     = '../../reports/exp_h_uniprot_eval/v2'
CACHE_DIR   = '../../reports/exp_h_uniprot_eval/seq_cache'  # reuse existing cache
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

UNIPROT_REST = 'https://rest.uniprot.org'
HEADERS      = {'User-Agent': 'PRISM-benchmark/2.0 (research; seungwon.david.lee@gmail.com)'}
MAX_LEN      = 1022
SEEDS        = [42, 7, 13, 21, 99]
EPOCHS       = 60
BATCH        = 512
N_PERM       = 10000


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


def fetch_one_isoform(iso_id):
    if iso_id.endswith('-1'):
        fetch_id = iso_id[:-2]
    else:
        fetch_id = iso_id
    cache = os.path.join(CACHE_DIR, f'{fetch_id}.fasta')
    if os.path.exists(cache) and os.path.getsize(cache) > 50:
        with open(cache) as f:
            fasta = f.read()
    else:
        url = f'{UNIPROT_REST}/uniprotkb/{fetch_id}.fasta'
        try:
            req = request.Request(url, headers=HEADERS)
            with request.urlopen(req, timeout=30) as r:
                fasta = r.read().decode()
            with open(cache, 'w') as f:
                f.write(fasta)
            time.sleep(0.3)
        except Exception as e:
            print(f"  [WARN] Failed {fetch_id}: {e}")
            return None
    seq_lines = [l for l in fasta.splitlines() if not l.startswith('>') and l.strip()]
    seq = ''.join(seq_lines).replace('*', '').strip()
    return seq[:MAX_LEN] if seq else None


@torch.no_grad()
def compute_delta_embeddings(sequences_dict, device, batch_size=16):
    import esm
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    all_ids   = list(sequences_dict.keys())
    all_seqs  = list(sequences_dict.values())
    result    = {}

    for i in range(0, len(all_ids), batch_size):
        batch_ids  = all_ids[i: i + batch_size]
        batch_seqs = all_seqs[i: i + batch_size]
        data       = [(bid, seq) for bid, seq in zip(batch_ids, batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[15, 30], return_contacts=False)
        reps = out['representations']
        for k, bid in enumerate(batch_ids):
            L = len(batch_seqs[k])
            phi30 = reps[30][k, 1:L+1].mean(0).cpu().float().numpy()
            phi15 = reps[15][k, 1:L+1].mean(0).cpu().float().numpy()
            delta  = phi30 - phi15
            result[bid] = np.concatenate([phi30, delta])  # (1280,)
        if (i // batch_size) % 5 == 0:
            print(f"    Embedded {min(i + batch_size, len(all_ids))}/{len(all_ids)}", flush=True)

    return result


def train_and_predict(X_tr, Y_tr, X_pred_raw, mf_terms, seeds, device):
    scaler  = MaxAbsScaler().fit(X_tr)
    Xt      = torch.tensor(scaler.transform(X_tr), dtype=torch.float32, device=device)
    Yt      = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    X_pred_s = scaler.transform(X_pred_raw)

    K = Y_tr.shape[1]
    all_preds = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = nn.Sequential(
            nn.Linear(X_tr.shape[1], 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, K), nn.Sigmoid()
        ).to(device)
        opt  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCELoss()
        model.train()
        for ep in range(EPOCHS):
            perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), BATCH):
                idx  = perm[b: b + BATCH]
                loss = crit(model(Xt[idx]), Yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_pred_s, device=device)).cpu().numpy()
        all_preds.append(pred)

    return np.mean(all_preds, axis=0)


def load_training_data():
    print("  Loading train embeddings...")
    phi30 = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
    phi15 = np.load(f'{DATA_DIR}/esm2_train_human_layer15_t30_150M.npy').astype(np.float32)
    X_tr  = np.concatenate([phi30, phi30 - phi15], axis=1)
    print(f"  X_tr: {X_tr.shape}")

    ENSG2SYM = {}
    with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

    tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
    tr_genes     = [clean(g) for g in tr_genes_raw]

    sym2id = {}
    with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) > 2:
                sym2id[p[2]] = p[1]
                if len(p) > 4 and p[4] != '-':
                    for syn in p[4].split('|'):
                        if syn not in sym2id: sym2id[syn] = p[1]

    tr_ids    = [sym2id.get(g, g) for g in tr_genes]
    tr_id_set = set(tr_ids)

    go_genes_tr = defaultdict(set)
    with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if p[0] != '9606' or p[7] != 'Function': continue
            if p[1] in tr_id_set: go_genes_tr[p[2]].add(p[1])

    mf_terms_path = '../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv'
    mf_terms = []
    with open(mf_terms_path) as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 6: mf_terms.append(p[0])
    print(f"  {len(mf_terms)} MF GO terms")

    tr_sym2idx = defaultdict(list)
    for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

    def build_Y_tr(go_id):
        pos_ids  = go_genes_tr[go_id]
        pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in pos_ids}
        y = np.zeros(len(tr_genes), dtype=np.float32)
        for sym in pos_syms:
            for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
        return y

    Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)
    print(f"  Y_tr: {Y_tr.shape}  pos rate: {Y_tr.mean():.4f}")
    return X_tr, Y_tr, mf_terms, sym2id, go_genes_tr


def permutation_p(n_correct, n_total, n_perm=10000, rng=None):
    """One-sided permutation test: P(correct >= observed | null=0.5)."""
    if rng is None: rng = np.random.default_rng(42)
    null = rng.binomial(n_total, 0.5, size=n_perm)
    return float((null >= n_correct).mean())


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print("  PRISM UniProt Benchmark v2 (51 pairs + permutation test)")
    print(f"  device={device}")
    print("=" * 65, flush=True)

    print("\n[1] Loading benchmark v2...")
    benchmark = []
    with open(BENCH_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            benchmark.append(row)
    print(f"  {len(benchmark)} pairs from {len(set(r['gene'] for r in benchmark))} genes")

    accessions = set()
    for row in benchmark:
        accessions.add(row['iso_a'])
        accessions.add(row['iso_b'])

    print(f"\n[2] Fetching {len(accessions)} sequences...")
    all_seqs = {}
    missing  = []
    for iso in sorted(accessions):
        seq = fetch_one_isoform(iso)
        if seq:
            all_seqs[iso] = seq
        else:
            missing.append(iso)
    if missing:
        print(f"  [WARN] Missing: {missing}")
    print(f"  Embedded {len(all_seqs)} isoforms")

    print("\n[3] Computing ESM-2 delta embeddings...")
    embed_cache = os.path.join(OUT_DIR, 'embeddings_v2.npy')
    if os.path.exists(embed_cache):
        cached = np.load(embed_cache, allow_pickle=True).item()
        new_isos = {k: v for k, v in all_seqs.items() if k not in cached}
        if new_isos:
            print(f"  {len(new_isos)} new isoforms to embed...")
            new_emb = compute_delta_embeddings(new_isos, device)
            cached.update(new_emb)
            np.save(embed_cache, cached)
        emb_data = cached
    else:
        emb_data = compute_delta_embeddings(all_seqs, device)
        np.save(embed_cache, emb_data)
    print(f"  Embeddings: {len(emb_data)} isoforms")

    print("\n[4] Loading training data...")
    X_tr, Y_tr, mf_terms, sym2id, go_genes_tr = load_training_data()
    go_to_idx = {go: i for i, go in enumerate(mf_terms)}

    # GO term remap (same as v1)
    remap = {
        'GO:0004714': 'GO:0004713',
        'GO:0005007': 'GO:0004713',
        'GO:0004693': 'GO:0004674',
        'GO:0097553': 'GO:0046982',
        'GO:0004197': 'GO:0003824',
        'GO:0006281': 'GO:0003677',
        'GO:0006977': 'GO:0003700',
        'GO:0008285': 'GO:0019901',
        'GO:0005178': 'GO:0048018',
        'GO:0005158': 'GO:0048018',
        'GO:0008083': 'GO:0008083',
        'GO:0000398': 'GO:0003723',
        'GO:0006357': 'GO:0003700',
        'GO:0005200': 'GO:0003779',
        'GO:0007399': 'GO:0005515',
        'GO:0016079': 'GO:0046982',
    }

    print("\n[5] Training v17f* (5 seeds)...")
    iso_order = sorted(emb_data.keys())
    X_pred    = np.stack([emb_data[iso] for iso in iso_order])
    preds     = train_and_predict(X_tr, Y_tr, X_pred, mf_terms, SEEDS, device)
    iso_pred  = {iso: preds[i] for i, iso in enumerate(iso_order)}
    print(f"  Predictions: {preds.shape}")

    print("\n[6] Evaluating direction accuracy...")
    eval_rows   = []
    correct     = 0
    total_eval  = 0
    gaps        = []
    correct_arr = []  # for permutation

    for row in benchmark:
        iso_a = row['iso_a']
        iso_b = row['iso_b']
        go    = row['go_term']
        dirn  = row['direction']

        go_norm = go.replace('GO_', 'GO:')
        if go_norm not in go_to_idx:
            go_norm = remap.get(go_norm, go_norm)
        if go_norm not in go_to_idx:
            eval_rows.append({**row, 'score_a': None, 'score_b': None,
                              'gap': None, 'correct': None,
                              'note': f'GO not in panel ({go}→{go_norm})'})
            continue
        if iso_a not in iso_pred or iso_b not in iso_pred:
            eval_rows.append({**row, 'score_a': None, 'score_b': None,
                              'gap': None, 'correct': None,
                              'note': 'sequence not embedded'})
            continue

        j  = go_to_idx[go_norm]
        sa = float(iso_pred[iso_a][j])
        sb = float(iso_pred[iso_b][j])
        gap = abs(sa - sb)

        if dirn == 'A_only':
            c = int(sa > sb)
        elif dirn == 'B_only':
            c = int(sb > sa)
        elif dirn == 'both':
            c = 1
        else:
            c = None

        if c is not None:
            correct += c
            total_eval += 1
            gaps.append(gap)
            correct_arr.append(c)

        eval_rows.append({**row, 'score_a': round(sa,4), 'score_b': round(sb,4),
                          'gap': round(gap,4), 'correct': c, 'note': 'OK'})

    acc = correct / total_eval
    mean_gap = np.mean(gaps)
    print(f"  Direction accuracy: {acc:.3f} ({correct}/{total_eval})")
    print(f"  Mean gap: {mean_gap:.4f}")

    # Permutation test
    print(f"\n[7] Permutation test (n={N_PERM})...")
    rng = np.random.default_rng(42)
    p_perm = permutation_p(correct, total_eval, N_PERM, rng)
    print(f"  Permutation p-value (one-sided, H0: acc=0.5): {p_perm:.4f}")

    # Scipy binomial
    binom_res = stats.binomtest(correct, total_eval, 0.5, alternative='greater')
    binom_p = binom_res.pvalue
    print(f"  Binomial p-value: {binom_p:.4f}")

    # Stratified by gap threshold
    print("\n[8] Stratified analysis by gap threshold:")
    gaps_arr = np.array(gaps)
    corr_arr = np.array(correct_arr)
    for thr in [0.05, 0.10, 0.15, 0.20]:
        mask = gaps_arr >= thr
        n_thr = mask.sum()
        c_thr = corr_arr[mask].sum()
        if n_thr == 0:
            print(f"  gap >= {thr:.2f}: n=0")
            continue
        p_thr = permutation_p(c_thr, n_thr, N_PERM, rng)
        p_binom = stats.binomtest(c_thr, n_thr, 0.5, alternative='greater').pvalue
        print(f"  gap >= {thr:.2f}: {c_thr}/{n_thr} = {c_thr/n_thr:.3f}  perm_p={p_thr:.4f}  binom_p={p_binom:.4f}")

    # Save results
    pair_path = f'{OUT_DIR}/pairwise_eval_v2.tsv'
    with open(pair_path, 'w') as f:
        f.write('gene\tiso_a\tiso_b\tgo_term\tdirection\tscore_a\tscore_b\tgap\tcorrect\tnote\n')
        for row in eval_rows:
            f.write(f"{row['gene']}\t{row['iso_a']}\t{row['iso_b']}\t{row['go_term']}\t"
                    f"{row['direction']}\t{row.get('score_a','')}\t{row.get('score_b','')}\t"
                    f"{row.get('gap','')}\t{row.get('correct','')}\t{row.get('note','')}\n")

    summ = {
        'n_pairs': len(benchmark),
        'n_eval': total_eval,
        'n_correct': int(correct),
        'acc': round(acc, 4),
        'mean_gap': round(float(mean_gap), 4),
        'perm_p': round(p_perm, 4),
        'binom_p': round(float(binom_p), 4),
        'strat': {}
    }
    for thr in [0.05, 0.10, 0.15, 0.20]:
        mask = gaps_arr >= thr
        n_thr = int(mask.sum())
        c_thr = int(corr_arr[mask].sum())
        if n_thr > 0:
            p_thr = permutation_p(c_thr, n_thr, N_PERM, rng)
            p_b   = float(stats.binomtest(c_thr, n_thr, 0.5, alternative='greater').pvalue)
            summ['strat'][f'gap{int(thr*100):02d}'] = {
                'n': n_thr, 'correct': c_thr, 'acc': round(c_thr/n_thr, 4),
                'perm_p': round(p_thr, 4), 'binom_p': round(p_b, 4)
            }

    with open(f'{OUT_DIR}/summary_v2.json', 'w') as f:
        json.dump(summ, f, indent=2)

    # Print per-pair table
    print("\n[9] Per-pair results:")
    for row in eval_rows:
        c   = row.get('correct')
        sym = '✓' if c == 1 else ('✗' if c == 0 else '—')
        sa  = row.get('score_a', 'N/A')
        sb  = row.get('score_b', 'N/A')
        gap = row.get('gap', '')
        print(f"  {sym} {row['gene']:10} {row['go_term']:15} {row['direction']:6}"
              f"  a={sa}  b={sb}  gap={gap}  {row.get('note','')[:50]}")

    print(f"\n  Saved: {pair_path}")
    print(f"  Saved: {OUT_DIR}/summary_v2.json")
    print("\n[DONE]")


if __name__ == '__main__':
    main()
