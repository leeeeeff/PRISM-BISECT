"""
build_domain_matrix_train_cdd.py — 학습 세트 CDD domain matrix 구축
=======================================================================
train_isoform_list.npy는 NM_ RefSeq ID를 사용하므로
human_isoform_dm.txt (gene, NM_ID, CDD integer IDs) 매핑으로 구축.

출력:
  results_isoform/features/domain_matrix_train_cdd.npy  (31668, 512)
  results_isoform/features/domain_cdd_vocab.txt         (상위 512 CDD)
  results_isoform/features/label_confidence_train.npy   (31668, 82)
"""

import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.metrics import average_precision_score

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
ID_DIR   = ROOT / "hMuscle/data/raw_data/data/id_lists"
DOM_DIR  = ROOT / "hMuscle/data/raw_data/data/raw_data/domain_data"
OUT_DIR  = ROOT / "hMuscle/results_isoform/features"
LOG_DIR  = ROOT / "reports/isoform_resolution_full"
N_CDD    = 512

# ── 1. NM_ → CDD domain set 매핑 ─────────────────────────────────────────
print("[1] human_isoform_dm.txt 파싱...")
nm_to_cdd = {}
with open(DOM_DIR / "human_isoform_dm.txt") as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        nm_id  = parts[1].strip()
        cdds   = set(parts[2].split()) if len(parts) >= 3 and parts[2].strip() else set()
        nm_to_cdd[nm_id] = cdds
print(f"  NM→CDD dict size: {len(nm_to_cdd)}")

# ── 2. train isoform list 로드 ─────────────────────────────────────────────
print("[2] train isoform list 로드...")
iso_raw = np.load(ID_DIR / "train_isoform_list.npy", allow_pickle=True)
iso_list = [x.decode() if isinstance(x, bytes) else x for x in iso_raw]
n = len(iso_list)
print(f"  n = {n}")

# ── 3. 매칭 ───────────────────────────────────────────────────────────────
print("[3] 매칭...")
iso_cdds = []
stats = {'hit': 0, 'miss': 0}
for iso in iso_list:
    cdds = nm_to_cdd.get(iso, None)
    if cdds is not None:
        stats['hit'] += 1
        iso_cdds.append(cdds)
    else:
        # strip version suffix  NM_000015.7 → NM_000015
        base = iso.rsplit('.', 1)[0]
        cdds = nm_to_cdd.get(base, set())
        if cdds:
            stats['hit'] += 1
        else:
            stats['miss'] += 1
        iso_cdds.append(cdds)

n_with = sum(1 for s in iso_cdds if s)
print(f"  hit={stats['hit']:,}, miss={stats['miss']:,}")
print(f"  With domain: {n_with:,}/{n:,} ({n_with/n*100:.1f}%)")

# ── 4. CDD vocabulary (top-512) ───────────────────────────────────────────
print("[4] CDD vocabulary 구축...")
cdd_freq = Counter()
for s in iso_cdds:
    cdd_freq.update(s)
print(f"  Unique CDD IDs: {len(cdd_freq)}")
top_cdds = [c for c, _ in cdd_freq.most_common(N_CDD)]
cdd_to_col = {c: i for i, c in enumerate(top_cdds)}
print(f"  Top-{N_CDD}, min freq: {cdd_freq[top_cdds[-1]]}")

vocab_path = OUT_DIR / "domain_cdd_vocab.txt"
with open(vocab_path, 'w') as f:
    for i, c in enumerate(top_cdds):
        f.write(f"{i}\t{c}\t{cdd_freq[c]}\n")
print(f"  Vocab saved: {vocab_path}")

# ── 5. Binary presence matrix ─────────────────────────────────────────────
print("[5] Binary presence matrix 구축...")
dm_tr = np.zeros((n, N_CDD), dtype=np.float32)
for i, cdd_set in enumerate(iso_cdds):
    for c in cdd_set:
        if c in cdd_to_col:
            dm_tr[i, cdd_to_col[c]] = 1.0

nz = (dm_tr != 0).any(axis=1).sum()
print(f"  Nonzero rows: {nz:,} ({nz/n*100:.1f}%)")
out_path = OUT_DIR / "domain_matrix_train_cdd.npy"
np.save(out_path, dm_tr)
print(f"  Saved: {out_path}")

# ── 6. GO-CDD importance + label confidence for train set ─────────────────
print("[6] GO-CDD importance matrix 계산...")
# Load training labels
import subprocess, tempfile, sys

# Load Y_tr using the same GO term list as v17f*
go_list_path = ROOT / "hMuscle/data/raw_data/data/raw_data/go_data/go_mf_terms.txt"
if not go_list_path.exists():
    # fallback: use the MF terms list used in v17f* (82 terms)
    go_list_path = ROOT / "hMuscle/data/MF_go_terms.txt"

# Read GO list
mf_terms = []
if go_list_path.exists():
    with open(go_list_path) as f:
        for line in f:
            term = line.strip()
            if term:
                mf_terms.append(term)
print(f"  GO terms loaded: {len(mf_terms)}")

# Build Y_tr from iso_id_mapping.txt + go annotations
# iso_id_mapping: RefSeqID → isoformID (numeric)
iso_id_map = {}
with open(ROOT / "hMuscle/data/raw_data/data/raw_data/expression_data/iso_id_mapping.txt") as f:
    next(f)  # header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            iso_id_map[parts[0].split('.')[0]] = int(parts[2])  # RefSeqID (no version) → isoformID

print(f"  iso_id_map size: {len(iso_id_map)}")

# Load GO annotations per isoform
go_ann_path = ROOT / "hMuscle/data/raw_data/data/raw_data/go_data"
if (go_ann_path / "iso_go_mf_annotations.txt").exists():
    iso_go_file = go_ann_path / "iso_go_mf_annotations.txt"
elif (go_ann_path / "go_mf_annotations.txt").exists():
    iso_go_file = go_ann_path / "go_mf_annotations.txt"
else:
    # Search for annotation file
    candidates = list(go_ann_path.glob("*.txt"))
    print(f"  GO annotation files: {[c.name for c in candidates]}")
    iso_go_file = candidates[0] if candidates else None

if iso_go_file and iso_go_file.exists():
    print(f"  Reading GO annotations from: {iso_go_file.name}")
    # Build isoformID → set of GO terms
    iso_num_to_go = {}
    with open(iso_go_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    iso_num = int(parts[0])
                    go_set  = set(parts[1].split() if len(parts) > 1 else [])
                    iso_num_to_go[iso_num] = go_set
                except:
                    pass
    print(f"  Isoforms with GO annotations: {len(iso_num_to_go)}")

    if mf_terms and iso_num_to_go:
        go_to_col = {g: i for i, g in enumerate(mf_terms)}
        n_go = len(mf_terms)
        Y_tr = np.zeros((n, n_go), dtype=np.float32)
        for k, iso in enumerate(iso_list):
            iso_base = iso.split('.')[0]
            iso_num = iso_id_map.get(iso_base, None)
            if iso_num and iso_num in iso_num_to_go:
                for go in iso_num_to_go[iso_num]:
                    if go in go_to_col:
                        Y_tr[k, go_to_col[go]] = 1.0
        pos_rate = Y_tr.mean()
        print(f"  Y_tr shape: {Y_tr.shape}, pos_rate: {pos_rate:.4f}")
        np.save(OUT_DIR / "Y_tr_mf.npy", Y_tr)

        # GO-CDD log-odds importance
        eps = 1e-6
        importance = np.zeros((n_go, N_CDD), dtype=np.float32)
        for j in range(n_go):
            pos_mask = Y_tr[:, j].astype(bool)
            neg_mask = ~pos_mask
            n_pos = pos_mask.sum()
            n_neg = neg_mask.sum()
            if n_pos < 5:
                continue
            p_pos = (dm_tr[pos_mask].sum(0) + eps) / (n_pos + 2 * eps)
            p_neg = (dm_tr[neg_mask].sum(0) + eps) / (n_neg + 2 * eps)
            importance[j] = np.log(p_pos / p_neg)

        # Gene-canonical domain vector
        gene_raw = np.load(ID_DIR / "train_gene_list.npy", allow_pickle=True)
        gene_list = [x.decode() if isinstance(x, bytes) else x for x in gene_raw]
        gene_arr = np.array(gene_list)
        gene_to_canon = {}
        for g in set(gene_list):
            mask = (gene_arr == g)
            gene_to_canon[g] = dm_tr[mask].max(0)
        canon_tr = np.stack([gene_to_canon[g] for g in gene_arr])

        # Label confidence
        eps2 = 1e-6
        imp_pos = np.maximum(importance, 0.0)  # (n_go, N_CDD)
        label_conf_tr = np.zeros((n, n_go), dtype=np.float32)
        for j in range(n_go):
            if imp_pos[j].sum() < eps2:
                label_conf_tr[:, j] = 1.0  # no info → neutral
                continue
            numerator   = dm_tr @ imp_pos[j]         # (n,)
            denominator = canon_tr @ imp_pos[j]      # (n,)
            label_conf_tr[:, j] = np.clip(
                numerator / (denominator + eps2), 0.0, 1.5)

        np.save(OUT_DIR / "label_confidence_train.npy", label_conf_tr)
        print(f"  label_confidence_train saved: {label_conf_tr.shape}")
        print(f"  Conf < 0.3 among positives: "
              f"{(label_conf_tr[Y_tr.astype(bool)] < 0.3).mean()*100:.1f}%")
    else:
        print("  WARNING: GO terms or annotations not found — skipping label confidence")
else:
    print(f"  WARNING: GO annotation file not found in {go_ann_path}")
    print(f"  Files: {list(go_ann_path.iterdir()) if go_ann_path.exists() else 'dir missing'}")
