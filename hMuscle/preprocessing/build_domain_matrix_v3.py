"""
build_domain_matrix_v3.py — 버전 번호 무시 매칭 + BambuTx 도메인 포함
=============================================================================
수정 사항:
  1. ENST 버전 번호 무시 매칭 (ENST00000394825.6 → ENST00000394825)
  2. BambuTx 도메인 정보 포함 (기존: 무조건 empty)
  3. 결과: domain_matrix_proper_test_v3.npy (36748, 512)

출력:
  results_isoform/features/domain_matrix_proper_test_v3.npy
  results_isoform/features/domain_matrix_proper_test_v3_vocab.txt
  reports/isoform_resolution_full/domain_v3_build_log.txt
"""

import numpy as np
from collections import Counter
from pathlib import Path

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")

DOMAIN_LIST = ROOT / "hMuscle/data/domain/domain_list.txt"
ISO_LIST    = ROOT / "hMuscle/model/my_isoform_list_fixed.npy"
OUT_DIR     = ROOT / "hMuscle/results_isoform/features"
LOG_DIR     = ROOT / "reports/isoform_resolution_full"
LOG_DIR.mkdir(parents=True, exist_ok=True)

N_PFAM = 512

print("[1] domain_list.txt 파싱 (버전 무시 + BambuTx 포함)...")
# key: base_id (버전/p suffix 제거) → pfam set
base_to_pfams = {}        # 버전 없는 base: ENST00000394825
versioned_to_pfams = {}   # 버전 있는: ENST00000394825.6

with open(DOMAIN_LIST) as f:
    for line in f:
        parts = line.strip().split('\t')
        raw_id = parts[0]
        pfams = set(parts[1].split()) if len(parts) > 1 and parts[1].strip() else set()

        # .pN suffix 제거
        no_p = raw_id.split('.p')[0] if '.p' in raw_id else raw_id
        versioned_to_pfams[no_p] = pfams

        # 버전 번호도 제거: ENST00000394825.6 → ENST00000394825
        base = no_p.split('.')[0] if '.' in no_p else no_p
        if base not in base_to_pfams:
            base_to_pfams[base] = pfams
        else:
            base_to_pfams[base] = base_to_pfams[base] | pfams  # union

print(f"  Versioned entries: {len(versioned_to_pfams)}, Base entries: {len(base_to_pfams)}")

print("[2] isoform list 로드 및 매칭...")
iso_list = np.load(ISO_LIST, allow_pickle=True)
iso_list = [x.decode() if isinstance(x, bytes) else x for x in iso_list]

iso_pfams = []
stats = {"versioned_hit": 0, "base_hit": 0, "miss": 0, "bambu_hit": 0, "bambu_miss": 0}

for iso in iso_list:
    if 'BambuTx' in iso:
        pfams = versioned_to_pfams.get(iso, set()) or base_to_pfams.get(iso, set())
        if pfams:
            stats["bambu_hit"] += 1
        else:
            stats["bambu_miss"] += 1
        iso_pfams.append(pfams)
    else:
        # 1) exact versioned match
        pfams = versioned_to_pfams.get(iso, None)
        if pfams is not None:
            stats["versioned_hit"] += 1
            iso_pfams.append(pfams)
            continue
        # 2) base match (버전 무시)
        base = iso.split('.')[0]
        pfams = base_to_pfams.get(base, None)
        if pfams is not None:
            stats["base_hit"] += 1
            iso_pfams.append(pfams)
        else:
            stats["miss"] += 1
            iso_pfams.append(set())

n_with = sum(1 for s in iso_pfams if s)
n = len(iso_pfams)
print(f"  versioned_hit={stats['versioned_hit']:,}, base_hit={stats['base_hit']:,}, "
      f"miss={stats['miss']:,}")
print(f"  BambuTx: hit={stats['bambu_hit']:,}, miss={stats['bambu_miss']:,}")
print(f"  With domain: {n_with:,}/{n:,} ({n_with/n*100:.1f}%)")

print("[3] Pfam vocabulary (top-512 by frequency)...")
pfam_freq = Counter()
for s in iso_pfams:
    pfam_freq.update(s)
print(f"  Unique Pfam IDs: {len(pfam_freq)}")
top_pfams = [p for p, _ in pfam_freq.most_common(N_PFAM)]
pfam_to_col = {p: c for c, p in enumerate(top_pfams)}
print(f"  Top-{N_PFAM}, min freq: {pfam_freq[top_pfams[-1]]}")

vocab_path = OUT_DIR / "domain_pfam_vocab_v3.txt"
with open(vocab_path, 'w') as f:
    for c, p in enumerate(top_pfams):
        f.write(f"{c}\t{p}\t{pfam_freq[p]}\n")
print(f"  Vocab saved: {vocab_path}")

print("[4] Binary presence matrix 구축...")
dm = np.zeros((n, N_PFAM), dtype=np.float32)
for i, pfam_set in enumerate(iso_pfams):
    for p in pfam_set:
        if p in pfam_to_col:
            dm[i, pfam_to_col[p]] = 1.0

nz = (dm != 0).any(axis=1).sum()
print(f"  Nonzero rows: {nz:,} ({nz/n*100:.1f}%)")
print(f"  Domain count range: {dm.sum(1).min():.0f}–{dm.sum(1).max():.0f}, "
      f"mean={dm.sum(1).mean():.3f}")

out_path = OUT_DIR / "domain_matrix_proper_test_v3.npy"
np.save(out_path, dm)
print(f"  Saved: {out_path}")

# delta matrix (vs gene canonical) — gene_list 필요
print("[5] delta matrix (per-isoform vs gene canonical)...")
gene_list_raw = np.load(ROOT / "hMuscle/data/test_set/gene_list.npy", allow_pickle=True)
iso_list_raw  = np.load(ROOT / "hMuscle/data/test_set/isoform_list.npy", allow_pickle=True)
ts_iso  = [x.decode() if isinstance(x,bytes) else x for x in iso_list_raw]
ts_gene = [x.decode() if isinstance(x,bytes) else x for x in gene_list_raw]
iso2gene = dict(zip(ts_iso, ts_gene))
gene_arr = np.array([iso2gene.get(i, "UNKNOWN") for i in iso_list])

# per-gene canonical domain vector = union of all domains in gene (max presence)
gene_to_canonical_vec = {}
for g in set(gene_arr):
    mask = (gene_arr == g)
    # canonical = isoform with max domain count (or union)
    gene_to_canonical_vec[g] = dm[mask].max(axis=0)  # (512,) max

canonical_vec = np.stack([gene_to_canonical_vec.get(g, np.zeros(N_PFAM, dtype=np.float32))
                           for g in gene_arr])
delta = dm - canonical_vec  # negative = domain lost vs canonical

np.save(OUT_DIR / "domain_delta_proper_test_v3.npy", delta)
print(f"  Delta saved: {OUT_DIR}/domain_delta_proper_test_v3.npy")

# 로그
log = f"""domain_matrix v3 build log
===========================
source: {DOMAIN_LIST}
isoform list: {ISO_LIST} ({n} entries)

Matching stats:
  versioned_hit: {stats['versioned_hit']:,}
  base_hit (version-agnostic): {stats['base_hit']:,}
  miss: {stats['miss']:,}
  BambuTx hit: {stats['bambu_hit']:,}
  BambuTx miss: {stats['bambu_miss']:,}

Result:
  With domain: {n_with:,}/{n:,} ({n_with/n*100:.1f}%)  ← v2 was 37.2%
  Unique Pfam: {len(pfam_freq)}
  Top-{N_PFAM} selected

Files:
  {out_path}
  {OUT_DIR}/domain_pfam_vocab_v3.txt
  {OUT_DIR}/domain_delta_proper_test_v3.npy
"""
with open(LOG_DIR / "domain_v3_build_log.txt", "w") as f:
    f.write(log)
print(log)
