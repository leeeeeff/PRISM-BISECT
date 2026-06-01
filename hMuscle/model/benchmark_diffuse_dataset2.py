"""
DIFFUSE Dataset#2 Benchmark: PRISM vs DIFFUSE
==============================================
Phase A (Zero-shot):  기존 PRISM 모델 → Dataset#2 test isoforms 직접 추론
Phase B (Retrained):  Dataset#2 train set으로 PRISM 재학습 → fair comparison
Phase C (Metrics):    within-gene CV, pos_bias → isoform-level resolution 증명

Run:
    conda activate isoform_env
    python benchmark_diffuse_dataset2.py --phase A   # zero-shot only
    python benchmark_diffuse_dataset2.py --phase AB  # zero-shot + retrain
"""

import os, sys, json, zipfile, argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BENCH_DIR = "/home/welcome1/sw1686/DIFFUSE/hMuscle/data/diffuse_benchmark"
MAIN_ZIP  = f"{BENCH_DIR}/diffuse_main_data.zip"
D2_DIR    = f"{BENCH_DIR}/datasets2&3/processed_data/dataset2/data"
OUT_DIR   = "/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/benchmark_diffuse"
os.makedirs(OUT_DIR, exist_ok=True)

# PRISM production model (v15d_bp_clean) saved weights directory
PRISM_WEIGHTS = "/home/welcome1/sw1686/DIFFUSE/hMuscle/saved_models"

# PRISM 18 GO terms
PRISM_18_GO = [
    "GO:0006096","GO:0006412","GO:0006936","GO:0022900",
    "GO:0006119","GO:0007005","GO:0006941","GO:0048813",
    "GO:0045214","GO:0014706","GO:0014813","GO:0014819",
    "GO:0060048","GO:0043534","GO:0006979","GO:0043066",
    "GO:0035966","GO:0000165",
]

# DIFFUSE 공개 수치 (Huang et al., Bioinformatics 2019, Dataset#2 human)
DIFFUSE_NUMBERS = {
    "macro_AUPRC": 0.581, "macro_AUROC": 0.840,
    "source": "Huang et al., Bioinformatics 2019"
}
DMIL_NUMBERS = {
    "macro_AUPRC": None, "Fmax_vs_DIFFUSE": "+40.8%",
    "AUROC_vs_DIFFUSE": "+63.3%",
    "source": "Guan et al., Bioinformatics 2021"
}

def build_prism():
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=(640,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=inp, outputs=out)

# ── Load Dataset#2 ─────────────────────────────────────────────────────────
def load_dataset2():
    print("Loading Dataset#2 isoform IDs...")
    test_iso  = [x.decode() if isinstance(x,bytes) else x
                 for x in np.load(f"{D2_DIR}/id_lists/test_isoform_list.npy",  allow_pickle=True)]
    test_gene = [x.decode() if isinstance(x,bytes) else x
                 for x in np.load(f"{D2_DIR}/id_lists/test_gene_list.npy",     allow_pickle=True)]
    train_iso = [x.decode() if isinstance(x,bytes) else x
                 for x in np.load(f"{D2_DIR}/id_lists/train_isoform_list.npy", allow_pickle=True)]
    train_gene= [x.decode() if isinstance(x,bytes) else x
                 for x in np.load(f"{D2_DIR}/id_lists/train_gene_list.npy",    allow_pickle=True)]

    # Protein sequences
    seq_dict = {}
    with zipfile.ZipFile(MAIN_ZIP) as z:
        with z.open("data/raw_data/sequence_data/isoform_cds_sequences.txt") as f:
            header, seq = None, []
            for line in f:
                line = line.decode().strip()
                if line.startswith(">"):
                    if header and seq:
                        nm = header.split("|")[1] if "|" in header else header.lstrip(">")
                        seq_dict[nm] = "".join(seq)
                    header, seq = line.lstrip(">"), []
                else:
                    seq.append(line)
            if header and seq:
                nm = header.split("|")[1] if "|" in header else header.lstrip(">")
                seq_dict[nm] = "".join(seq)

    # Match sequences
    test_data  = [(iso, g, seq_dict[iso]) for iso, g in zip(test_iso,  test_gene)  if iso in seq_dict]
    train_data = [(iso, g, seq_dict[iso]) for iso, g in zip(train_iso, train_gene) if iso in seq_dict]
    print(f"Test  matched: {len(test_data)}/{len(test_iso)}")
    print(f"Train matched: {len(train_data)}/{len(train_iso)}")

    # GO labels (gene-level)
    gene2go = defaultdict(set)
    with open(f"{D2_DIR}/annotations/human_annotations.txt") as f:
        for line in f:
            p = line.strip().split("\t")
            gene2go[p[0]] = set(p[1:])

    # GO slim terms
    go_slim = [l.strip().split()[0] for l in open(f"{D2_DIR}/go_terms/go_slim.txt") if l.strip()]

    return test_data, train_data, gene2go, go_slim

# ── ESM-2 embeddings ───────────────────────────────────────────────────────
def compute_esm2(data, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached ESM-2: {cache_path}")
        return np.load(cache_path)

    import torch, esm as esm_lib
    model, alphabet = esm_lib.pretrained.esm2_t30_150M_UR50D()
    model = model.cuda().eval()
    bc = alphabet.get_batch_converter()

    BATCH, MAX_LEN = 32, 1022
    all_emb = []
    seqs = [d[2] for d in data]
    ids  = [d[0] for d in data]

    for start in range(0, len(seqs), BATCH):
        batch = [(ids[i], seqs[i][:MAX_LEN]) for i in range(start, min(start+BATCH, len(seqs)))]
        _, _, tokens = bc(batch)
        tokens = tokens.cuda()
        with torch.no_grad():
            out = model(tokens, repr_layers=[30], return_contacts=False)
        for j, (_, seq) in enumerate(batch):
            L = min(len(seq), MAX_LEN)
            all_emb.append(out["representations"][30][j, 1:L+1].mean(0).cpu().numpy())
        if (start // BATCH) % 20 == 0:
            print(f"  ESM-2: {start+len(batch)}/{len(seqs)}")

    emb = np.array(all_emb, dtype=np.float32)
    np.save(cache_path, emb)
    print(f"Saved: {cache_path}, shape: {emb.shape}")
    return emb

# ── Isoform-level discrimination metrics ──────────────────────────────────
def compute_discrimination_metrics(preds, labels, genes):
    gene_groups = defaultdict(list)
    for idx, g in enumerate(genes):
        gene_groups[g].append(idx)

    cv_list, pos_bias_list = [], []
    for g, idxs in gene_groups.items():
        if len(idxs) < 2: continue
        s = preds[idxs]; l = labels[idxs]
        cv_list.append(s.std() / (s.mean() + 1e-8))
        if l.sum() > 0:
            pos_bias_list.append(s[l==1].mean() / (s.mean() + 1e-8))

    return {
        "within_gene_CV":  float(np.mean(cv_list))      if cv_list      else 0,
        "pos_bias":        float(np.mean(pos_bias_list)) if pos_bias_list else 0,
        "n_multigene":     len(cv_list),
    }

# ── Phase A: Zero-shot PRISM (기존 muscle 학습 모델 직접 사용) ───────────
def phase_A_zerosshot(test_data, gene2go, go_slim, X_test):
    print("\n" + "="*60)
    print("Phase A: Zero-shot PRISM on Dataset#2 (muscle-trained model)")
    print("="*60)
    overlap = [g for g in PRISM_18_GO if g in go_slim]
    print(f"Overlapping GO terms: {overlap}")

    genes  = [d[1] for d in test_data]
    results = {}

    for go in overlap:
        labels = np.array([1 if go in gene2go.get(g, set()) else 0 for g in genes])
        if labels.sum() < 10:
            print(f"{go}: skip ({labels.sum()} positives)")
            continue

        # v15d_bp_clean 프로덕션 모델만 사용 (focal_fitted_DNN.h5)
        safe = go.replace(":","_")
        model_pattern = [f for f in os.listdir(PRISM_WEIGHTS)
                         if safe in f and f.endswith('focal_fitted_DNN.h5')]
        if not model_pattern:
            print(f"{go}: no focal_fitted_DNN.h5 found, skipping")
            continue

        seed_preds = []
        for mf in sorted(model_pattern)[:5]:
            m = build_prism()
            m.load_weights(f"{PRISM_WEIGHTS}/{mf}")
            seed_preds.append(m.predict(X_test, batch_size=1024, verbose=0).flatten())

        preds  = np.mean(seed_preds, axis=0)
        auprc  = average_precision_score(labels, preds)
        auroc  = roc_auc_score(labels, preds)
        disc   = compute_discrimination_metrics(preds, labels, genes)

        results[go] = {"AUPRC": auprc, "AUROC": auroc, "n_pos": int(labels.sum()), **disc}
        print(f"  {go}: AUPRC={auprc:.4f} AUROC={auroc:.4f} | "
              f"pos_bias={disc['pos_bias']:.3f} within_CV={disc['within_gene_CV']:.4f}")

    return results

# ── Phase B: Retrain PRISM on Dataset#2 train set ─────────────────────────
def phase_B_retrain(test_data, train_data, gene2go, go_slim, X_test, X_train):
    print("\n" + "="*60)
    print("Phase B: PRISM retrained on Dataset#2 (all 96 GO terms)")
    print("="*60)
    from tensorflow.keras import callbacks as K_cb

    test_genes  = [d[1] for d in test_data]
    train_genes = [d[1] for d in train_data]
    N_SEEDS     = 5
    results     = {}

    for go in go_slim:
        y_tr = np.array([1 if go in gene2go.get(g, set()) else 0 for g in train_genes])
        y_te = np.array([1 if go in gene2go.get(g, set()) else 0 for g in test_genes])
        if y_tr.sum() < 10 or y_te.sum() < 5: continue

        seed_preds = []
        for seed in range(N_SEEDS):
            tf.random.set_seed(seed * 137 + 42); np.random.seed(seed * 137 + 42)
            m = build_prism()
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
            cb = [K_cb.EarlyStopping(monitor='val_loss', patience=10,
                                     restore_best_weights=True, verbose=0),
                  K_cb.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)]
            m.fit(X_train, y_tr, epochs=80, batch_size=512,
                  validation_split=0.1, callbacks=cb, verbose=0)
            seed_preds.append(m.predict(X_test, batch_size=1024, verbose=0).flatten())

        preds = np.mean(seed_preds, axis=0)
        if y_te.sum() == 0: continue
        auprc = average_precision_score(y_te, preds)
        auroc = roc_auc_score(y_te, preds)
        disc  = compute_discrimination_metrics(preds, y_te, test_genes)
        results[go] = {"AUPRC": auprc, "AUROC": auroc, "n_pos": int(y_te.sum()), **disc}

    macro_auprc = np.mean([r["AUPRC"] for r in results.values()])
    macro_auroc = np.mean([r["AUROC"] for r in results.values()])
    macro_cv    = np.mean([r["within_gene_CV"] for r in results.values()])
    macro_pb    = np.mean([r["pos_bias"] for r in results.values()])

    print(f"\nPRISM retrained ({len(results)} GO terms):")
    print(f"  Macro AUPRC = {macro_auprc:.4f}  [DIFFUSE: {DIFFUSE_NUMBERS['macro_AUPRC']}]")
    print(f"  Macro AUROC = {macro_auroc:.4f}  [DIFFUSE: {DIFFUSE_NUMBERS['macro_AUROC']}]")
    print(f"  Mean within-gene CV = {macro_cv:.4f}  [DIFFUSE expected: ~0.000]")
    print(f"  Mean pos_bias       = {macro_pb:.4f}  [DIFFUSE expected: ~1.000]")

    return results, {"macro_AUPRC": macro_auprc, "macro_AUROC": macro_auroc,
                     "macro_within_CV": macro_cv, "macro_pos_bias": macro_pb}

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="A", choices=["A","B","AB"],
                        help="A=zero-shot, B=retrain, AB=both")
    args = parser.parse_args()

    test_data, train_data, gene2go, go_slim = load_dataset2()

    # ESM-2 test embeddings
    X_test = compute_esm2(test_data, f"{OUT_DIR}/esm2_dataset2_test.npy")

    all_results = {}

    if "A" in args.phase:
        print("[Phase A skipped: saved models are legacy CNN+LSTM format, not PRISM ESM-2 MLP]")
        # Zero-shot Phase A requires PRISM ESM-2 MLP weights; saved_models/ has old DIFFUSE weights

    if "B" in args.phase:
        X_train = compute_esm2(train_data, f"{OUT_DIR}/esm2_dataset2_train.npy")
        res_B, summary_B = phase_B_retrain(test_data, train_data, gene2go, go_slim, X_test, X_train)
        all_results["phase_B_retrained"] = res_B
        all_results["phase_B_summary"]   = summary_B

    all_results["diffuse_reported"] = DIFFUSE_NUMBERS
    all_results["dmil_reported"]    = DMIL_NUMBERS

    out_path = f"{OUT_DIR}/benchmark_dataset2_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")
