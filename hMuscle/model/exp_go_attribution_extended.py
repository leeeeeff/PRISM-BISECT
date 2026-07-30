"""
exp_go_attribution_extended.py
==============================
Extended GO attribution analysis beyond S1/S2 dichotomy:

1. S2 mode-top analysis (excluding GO:0005515):
   - Which specific functions do mode-top isoforms specialize in?
   - Domain composition, score gaps, domain-dep vs motif-dep classification

2. MIXED gene per-isoform analysis:
   - Does mode-top / 5515-winner framework hold for MIXED genes?
   - Specialist isoforms in MIXED genes vs S2 genes

3. Extended filter (ALL multi-isoform, >=1 GO):
   - Relax from (>=3 isoform, >=2 GO) to (>=2 isoform, >=1 GO)
   - S1/S2/MIXED rates at relaxed filter
   - mode-top framework validity
"""

import numpy as np, os, json, csv
from pathlib import Path
from collections import defaultdict
from scipy import stats
import gzip

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
ID_DIR   = ROOT / "hMuscle/data/raw_data/data/id_lists"
DATA_DIR = ROOT / "hMuscle/data"
FEAT_DIR = ROOT / "hMuscle/results_isoform/features"
OUT_DIR  = ROOT / "reports/isoform_resolution_full"
ANNOT_DIR = ROOT / "hMuscle/data/raw_data/data/annotations"

# ── 1. Load v17f* predictions and labels ──────────────────────────────────
print("[1] Loading predictions...")
preds = np.load(ROOT / "reports/v17f_star_bootstrap/v17f_star_preds.npy")  # (36748, 82)
Y_te  = np.load(ROOT / "reports/v17f_star_bootstrap/Y_te.npy")              # (36748, 82)
print(f"  preds={preds.shape}, Y_te={Y_te.shape}")

# MF terms (82 terms)
mf_terms = []
with open(str(ROOT / "reports/v_expanded_gomf/mf_domain_vs_prism.tsv")) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 1: mf_terms.append(p[0])
mf_terms = np.array(mf_terms)
print(f"  MF terms: {len(mf_terms)}")

go5515_idx = np.where(mf_terms == 'GO:0005515')[0]
print(f"  GO:0005515 index: {go5515_idx}")

# ── 2. Load test isoform/gene lists ───────────────────────────────────────
print("[2] Loading test IDs...")
te_gene_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_iso_raw  = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
te_genes = [x.decode() if isinstance(x, bytes) else x for x in te_gene_raw]
te_isos  = [x.decode() if isinstance(x, bytes) else x for x in te_iso_raw]

# ENSG → gene symbol
ENSG2SYM = {}
with open(str(ID_DIR / "ensembl_to_symbol.txt")) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

def clean_gene(g):
    s = str(g)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

te_genes_sym = [ENSG2SYM.get(clean_gene(g).split('.')[0], clean_gene(g).split('.')[0]) for g in te_genes]

# Group isoforms by gene symbol
gene2idxs = defaultdict(list)
for i, g in enumerate(te_genes_sym):
    gene2idxs[g].append(i)
print(f"  {len(te_isos)} isoforms, {len(gene2idxs)} unique genes")

# ── 3. Load domain type information ───────────────────────────────────────
print("[3] Loading domain type info...")
iso2type = {}
iso2domcount = {}
iso2canon_domcount = {}
iso2gap = {}
with open(str(OUT_DIR / "full_isoform_feature_types.tsv")) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6:
            iso_id = p[0]
            iso2domcount[iso_id] = int(p[2])
            iso2canon_domcount[iso_id] = int(p[3])
            iso2type[iso_id] = p[4]
            try: iso2gap[iso_id] = float(p[5])
            except: iso2gap[iso_id] = 0.0

# ── 4. Load norm_ratio for MF domain-dependent classification ─────────────
print("[4] Computing norm_ratio for MF terms...")
X_te_l30 = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)
norms = np.linalg.norm(X_te_l30, axis=1)  # (36748,)

norm_ratio = {}
for j, go_id in enumerate(mf_terms):
    pos_mask = Y_te[:, j] > 0
    neg_mask = Y_te[:, j] == 0
    n_pos = pos_mask.sum()
    if n_pos < 5:
        norm_ratio[go_id] = 1.0
        continue
    norm_pos = norms[pos_mask].mean()
    norm_neg = norms[neg_mask].mean()
    norm_ratio[go_id] = float(norm_pos / norm_neg) if norm_neg > 0 else 1.0

domain_dep_terms = set(go for go, r in norm_ratio.items() if r < 0.97)
print(f"  Domain-dependent MF terms (norm_ratio < 0.97): {len(domain_dep_terms)}/82")

# ── 5. Load existing S1/S2/MIXED gene classification ─────────────────────
print("[5] Loading S1/S2/MIXED gene classification...")
gene_scenario = {}
gene_info = {}
with open(str(OUT_DIR / "go_attribution_per_gene.tsv")) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 9:
            gene_scenario[p[0]] = p[8]  # S1/S2/MIXED
            gene_info[p[0]] = {
                'n_iso': int(p[1]), 'n_go': int(p[2]),
                'corr': float(p[3]), 'fsi': float(p[4]),
                'top_cons': float(p[5]),
            }

print(f"  Total genes classified: {len(gene_scenario)}")
print(f"  S1={sum(1 for s in gene_scenario.values() if s=='S1')}, "
      f"S2={sum(1 for s in gene_scenario.values() if s=='S2')}, "
      f"Mixed={sum(1 for s in gene_scenario.values() if s=='Mixed')}")

# ── 6. Mode-top analysis per gene ─────────────────────────────────────────
# For each gene: for each GO term, which isoform tops the score?
print("[6] Computing mode-top isoforms per gene...")

# gene ENSG → symbol map for gene_scenario keys
# gene_scenario uses ENSG IDs
ENSG2SYM_gene = {}
for ensg, sym in ENSG2SYM.items():
    ENSG2SYM_gene[ensg] = sym

def ensg_to_sym(ensg_ver):
    base = str(ensg_ver).replace("b'", "").replace("'", "").split('.')[0]
    return ENSG2SYM.get(base, base)

# Build gene_to_sym mapping from go_attribution_per_gene.tsv gene column
# gene column has ENSG IDs like ENSG00000162664.17

gene_mode_top_results = {}  # gene_id → {go_j: top_iso_idx, mode_top_iso: ..., ...}

for gene_id, scenario in gene_scenario.items():
    gene_sym = ensg_to_sym(gene_id)
    idxs = gene2idxs.get(gene_sym, [])
    if len(idxs) < 2: continue

    gene_preds = preds[idxs]  # (n_iso, 82)
    gene_y     = Y_te[idxs]   # (n_iso, 82)

    # For each GO term: which isoform tops the score?
    top_iso_per_go = np.argmax(gene_preds, axis=0)  # (82,)

    # Which GOs are positive for this gene (gene-level annotation)?
    pos_go_mask = gene_y.max(0) > 0  # (82,) — any isoform annotated
    pos_go_idxs = np.where(pos_go_mask)[0]
    if len(pos_go_idxs) < 1: continue

    # mode-top isoform: the isoform that tops the most GO terms (among positive GOs)
    top_votes = np.bincount(top_iso_per_go[pos_go_idxs], minlength=len(idxs))
    mode_top_local = np.argmax(top_votes)
    mode_top_idx = idxs[mode_top_local]
    mode_top_iso = te_isos[mode_top_idx]

    # 5515-winner if GO:0005515 is present
    g5515_idx = go5515_idx[0] if len(go5515_idx) > 0 else -1
    winner5515_idx = idxs[int(top_iso_per_go[g5515_idx])] if g5515_idx >= 0 else None
    winner5515_iso = te_isos[winner5515_idx] if winner5515_idx is not None else None

    # GOs where mode-top wins (excl. GO:0005515)
    mode_top_wins = [j for j in pos_go_idxs if top_iso_per_go[j] == mode_top_local and j != g5515_idx]

    gene_mode_top_results[gene_id] = {
        'gene_sym': gene_sym,
        'scenario': scenario,
        'n_iso': len(idxs),
        'n_pos_go': len(pos_go_idxs),
        'mode_top_iso': mode_top_iso,
        'mode_top_votes': int(top_votes[mode_top_local]),
        'mode_top_wins_go': mode_top_wins,
        'winner5515_iso': winner5515_iso,
        'is_same_iso': (mode_top_iso == winner5515_iso) if winner5515_iso else None,
        'top_iso_per_go': top_iso_per_go.tolist(),
        'pos_go_idxs': pos_go_idxs.tolist(),
    }

print(f"  Analyzed {len(gene_mode_top_results)} genes")

# ── 7. S2 mode-top detailed analysis (excl. GO:0005515) ───────────────────
print("\n[7] S2 mode-top analysis (excl. GO:0005515)...")
s2_genes = {g: v for g, v in gene_mode_top_results.items() if v['scenario'] == 'S2'}
print(f"  S2 genes with predictions: {len(s2_genes)}")

# For S2 genes: tabulate which GO terms are "specialized" (mode-top wins exclusively)
go_specialization_count = defaultdict(int)   # go_j → count of S2 genes where mode-top specializes
go_specialization_domtype = defaultdict(list) # go_j → list of domain types of mode-top isoforms

for gene_id, v in s2_genes.items():
    gene_sym = v['gene_sym']
    idxs = gene2idxs.get(gene_sym, [])
    if len(idxs) < 2: continue

    gene_preds = preds[idxs]
    top_iso_per_go = np.array(v['top_iso_per_go'])
    pos_go_idxs = v['pos_go_idxs']
    g5515 = go5515_idx[0] if len(go5515_idx) > 0 else -1

    for j in v['mode_top_wins_go']:
        if j == g5515: continue
        go_id = mf_terms[j]
        go_specialization_count[go_id] += 1
        mode_top_local = np.argmax(np.bincount(top_iso_per_go[[p for p in pos_go_idxs if p != g5515]], minlength=len(idxs)))
        mode_top_iso_id = te_isos[idxs[mode_top_local]]
        dom_type = iso2type.get(mode_top_iso_id, 'Unknown')
        go_specialization_domtype[go_id].append(dom_type)

print("\n  S2 specialized GO terms (excl. GO:0005515), sorted by count:")
sorted_terms = sorted(go_specialization_count.items(), key=lambda x: -x[1])
for go_id, cnt in sorted_terms[:20]:
    dom_types = go_specialization_domtype[go_id]
    type_counts = {}
    for t in dom_types: type_counts[t] = type_counts.get(t, 0) + 1
    dep = "domain-dep" if go_id in domain_dep_terms else "motif-dep"
    nr = norm_ratio.get(go_id, 1.0)
    print(f"  {go_id}: n={cnt}, {dep}(ratio={nr:.3f}), types={dict(sorted(type_counts.items(), key=lambda x:-x[1]))}")

# Score gap distribution for mode-top S2 specialized GO terms
print("\n  Score gaps in S2 mode-top GO wins (excl. GO:0005515):")
gaps_domain_dep = []
gaps_motif_dep  = []
for gene_id, v in s2_genes.items():
    gene_sym = v['gene_sym']
    idxs = gene2idxs.get(gene_sym, [])
    if len(idxs) < 2: continue
    gene_preds = preds[idxs]
    top_iso_per_go = np.array(v['top_iso_per_go'])
    g5515 = go5515_idx[0] if len(go5515_idx) > 0 else -1

    for j in v['mode_top_wins_go']:
        if j == g5515: continue
        go_id = mf_terms[j]
        scores_j = gene_preds[:, j]
        gap = scores_j.max() - np.sort(scores_j)[-2] if len(idxs) >= 2 else 0
        if go_id in domain_dep_terms: gaps_domain_dep.append(gap)
        else: gaps_motif_dep.append(gap)

if gaps_domain_dep and gaps_motif_dep:
    print(f"  Domain-dep GO wins: n={len(gaps_domain_dep)}, mean_gap={np.mean(gaps_domain_dep):.4f}")
    print(f"  Motif-dep GO wins:  n={len(gaps_motif_dep)}, mean_gap={np.mean(gaps_motif_dep):.4f}")
    _, p = stats.mannwhitneyu(gaps_domain_dep, gaps_motif_dep, alternative='two-sided')
    print(f"  MWU p-value: {p:.4e}")

# ── 8. MIXED gene analysis ─────────────────────────────────────────────────
print("\n[8] MIXED gene analysis...")
mixed_genes = {g: v for g, v in gene_mode_top_results.items() if v['scenario'] == 'Mixed'}
print(f"  MIXED genes with predictions: {len(mixed_genes)}")

# For each MIXED gene: classify each isoform as "generalist" vs "specialist"
# Generalist: tops the most GO terms (highest mode vote count)
# Specialist: tops fewer GO terms, often specific ones
mixed_specialist_analysis = []
for gene_id, v in mixed_genes.items():
    gene_sym = v['gene_sym']
    idxs = gene2idxs.get(gene_sym, [])
    if len(idxs) < 2: continue
    gene_preds = preds[idxs]
    gene_y = Y_te[idxs]
    top_iso_per_go = np.array(v['top_iso_per_go'])
    pos_go_idxs = v['pos_go_idxs']
    g5515 = go5515_idx[0] if len(go5515_idx) > 0 else -1

    if len(pos_go_idxs) < 2: continue

    # Votes per isoform
    votes = np.bincount(top_iso_per_go[pos_go_idxs], minlength=len(idxs))
    sorted_local = np.argsort(-votes)
    generalist_local = sorted_local[0]
    specialist_locals = sorted_local[1:]

    # Does 5515-winner differ from generalist?
    winner5515_local = int(top_iso_per_go[g5515]) if g5515 >= 0 else -1
    five515_same_as_generalist = (winner5515_local == generalist_local) if g5515 >= 0 else None

    # GOs where specialist tops (excl. 5515)
    for sp_local in specialist_locals:
        sp_go_wins = [j for j in pos_go_idxs if top_iso_per_go[j] == sp_local and j != g5515]
        if len(sp_go_wins) == 0: continue
        sp_idx = idxs[sp_local]
        sp_iso = te_isos[sp_idx]
        sp_type = iso2type.get(sp_iso, 'Unknown')
        gen_iso = te_isos[idxs[generalist_local]]
        gen_type = iso2type.get(gen_iso, 'Unknown')

        # Score gap for specialist wins
        gaps = []
        for j in sp_go_wins:
            scores_j = gene_preds[:, j]
            gap = scores_j.max() - np.sort(scores_j)[-2] if len(idxs) >= 2 else 0
            gaps.append(gap)

        mixed_specialist_analysis.append({
            'gene_id': gene_id,
            'gene_sym': gene_sym,
            'generalist_type': gen_type,
            'specialist_type': sp_type,
            'specialist_wins': [mf_terms[j] for j in sp_go_wins],
            'mean_gap': float(np.mean(gaps)) if gaps else 0,
            'five515_same_as_generalist': five515_same_as_generalist,
        })

if mixed_specialist_analysis:
    n_valid = len(mixed_specialist_analysis)
    n_five515_diff = sum(1 for x in mixed_specialist_analysis if x['five515_same_as_generalist'] == False)
    print(f"  MIXED specialist pairs: {n_valid}")
    print(f"  5515-winner differs from generalist: {n_five515_diff}/{n_valid} = {n_five515_diff/n_valid:.2%}")
    mean_gap = np.mean([x['mean_gap'] for x in mixed_specialist_analysis])
    print(f"  Mean score gap for specialist wins: {mean_gap:.4f}")

    # Type distribution of specialists in MIXED genes
    type_cnt = defaultdict(int)
    for x in mixed_specialist_analysis:
        type_cnt[x['specialist_type']] += 1
    print(f"  Specialist type distribution: {dict(sorted(type_cnt.items(), key=lambda x: -x[1]))}")

    # GO term distribution for specialist wins
    spec_go_cnt = defaultdict(int)
    for x in mixed_specialist_analysis:
        for go in x['specialist_wins']: spec_go_cnt[go] += 1
    print(f"\n  Top specialist GOs in MIXED genes:")
    for go_id, cnt in sorted(spec_go_cnt.items(), key=lambda x: -x[1])[:10]:
        dep = "domain-dep" if go_id in domain_dep_terms else "motif-dep"
        print(f"    {go_id}: {cnt} ({dep})")

# ── 9. Extended filter: ALL multi-isoform genes (>=2 iso, >=1 GO) ─────────
print("\n[9] Extended filter analysis (>=2 isoforms, >=1 GO term)...")

# Load gene-level GO annotations from gene2go
sym2id = {}
with gzip.open(str(ANNOT_DIR / "Homo_sapiens.gene_info.gz"), 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id: sym2id[syn] = p[1]

go_genes_all = defaultdict(set)
with gzip.open(str(ANNOT_DIR / "gene2go.gz"), 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) < 8 or p[0] != '9606' or p[7] != 'Function': continue
        go_genes_all[p[2]].add(p[1])

# For all genes in test set with >=2 isoforms
extended_results = []
already_in_original = set(gene_mode_top_results.keys())

for gene_sym, idxs in gene2idxs.items():
    if len(idxs) < 2: continue
    gene_preds = preds[idxs]  # (n_iso, 82)
    gene_y = Y_te[idxs]

    # Find positive GO terms for this gene
    pos_go_mask = gene_y.max(0) > 0
    pos_go_idxs = np.where(pos_go_mask)[0]
    if len(pos_go_idxs) < 1: continue

    top_iso_per_go = np.argmax(gene_preds, axis=0)
    votes = np.bincount(top_iso_per_go[pos_go_idxs], minlength=len(idxs))

    # FSI = Functional Specialization Index
    if len(pos_go_idxs) >= 2:
        scores = gene_preds[:, pos_go_idxs]  # (n_iso, n_go)
        corr_vals = []
        for i in range(len(pos_go_idxs)):
            for j2 in range(i+1, len(pos_go_idxs)):
                if scores[:, i].std() > 1e-8 and scores[:, j2].std() > 1e-8:
                    r, _ = stats.spearmanr(scores[:, i], scores[:, j2])
                    corr_vals.append(r)
        mean_corr = np.mean(corr_vals) if corr_vals else 1.0
        fsi = 1.0 - mean_corr
    else:
        mean_corr = 1.0
        fsi = 0.0

    # Classify (relaxed S1/S2/Mixed with 1 GO: all S1 by def)
    top_cons = (np.bincount(top_iso_per_go[pos_go_idxs], minlength=len(idxs)).max() / len(pos_go_idxs))
    if len(pos_go_idxs) >= 2:
        scenario = 'S1' if mean_corr >= 0.7 and top_cons >= 0.8 else \
                   'S2' if mean_corr < 0.3 else 'Mixed'
    else:
        scenario = 'S1_single'

    extended_results.append({
        'gene_sym': gene_sym,
        'n_iso': len(idxs),
        'n_pos_go': len(pos_go_idxs),
        'mean_corr': float(mean_corr),
        'fsi': float(fsi),
        'top_cons': float(top_cons),
        'scenario': scenario,
        'in_original': gene_sym in {v['gene_sym'] for v in gene_mode_top_results.values()},
    })

print(f"  All genes with >=2 iso, >=1 GO: {len(extended_results)}")
scen_counts = defaultdict(int)
for r in extended_results: scen_counts[r['scenario']] += 1
print(f"  Scenario distribution:")
for s, c in sorted(scen_counts.items(), key=lambda x: -x[1]):
    print(f"    {s}: {c} ({c/len(extended_results):.1%})")

# Subset: genes NOT in original analysis (1-2 isoforms or 1 GO)
new_genes = [r for r in extended_results if not r['in_original']]
if new_genes:
    new_scen = defaultdict(int)
    for r in new_genes: new_scen[r['scenario']] += 1
    print(f"\n  Genes NEW to extended analysis (outside original >=3 iso, >=2 GO filter): {len(new_genes)}")
    for s, c in sorted(new_scen.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c} ({c/len(new_genes):.1%})")

    # mode-top validity for new genes
    valid_splits = sum(1 for r in new_genes if r['scenario'] in ('S2', 'Mixed'))
    print(f"\n  S2+Mixed (non-trivial splits) in new genes: {valid_splits}/{len(new_genes)} = {valid_splits/len(new_genes):.1%}")

# ── 10. Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\n[S2 mode-top, excl. GO:0005515]")
print(f"  Total S2 genes: {len(s2_genes)}")
dom_dep_wins = sum(1 for g, v in s2_genes.items()
                   for j in v['mode_top_wins_go']
                   if j != (go5515_idx[0] if len(go5515_idx)>0 else -1)
                   and mf_terms[j] in domain_dep_terms)
motif_dep_wins = sum(1 for g, v in s2_genes.items()
                     for j in v['mode_top_wins_go']
                     if j != (go5515_idx[0] if len(go5515_idx)>0 else -1)
                     and mf_terms[j] not in domain_dep_terms)
total_wins = dom_dep_wins + motif_dep_wins
if total_wins > 0:
    print(f"  Mode-top specializes in domain-dep GO: {dom_dep_wins}/{total_wins} = {dom_dep_wins/total_wins:.1%}")
    print(f"  Mode-top specializes in motif-dep GO:  {motif_dep_wins}/{total_wins} = {motif_dep_wins/total_wins:.1%}")

print("\n[MIXED gene specialist isoforms]")
if mixed_specialist_analysis:
    print(f"  Specialist pairs: {len(mixed_specialist_analysis)}")
    five515_diff = sum(1 for x in mixed_specialist_analysis if x['five515_same_as_generalist'] == False)
    print(f"  5515-winner ≠ generalist: {five515_diff}/{len(mixed_specialist_analysis)} = {five515_diff/len(mixed_specialist_analysis):.1%}")

print("\n[Extended filter: ALL multi-isoform genes]")
n_multi_go = len([r for r in extended_results if r['n_pos_go'] >= 2])
n_s2_ext = scen_counts.get('S2', 0)
n_mixed_ext = scen_counts.get('Mixed', 0)
n_total = len(extended_results)
print(f"  Total: {n_total} genes (vs original 2738)")
print(f"  S2 rate: {n_s2_ext}/{n_total} = {n_s2_ext/n_total:.1%} (original: 42.4%)")
print(f"  Mixed rate: {n_mixed_ext}/{n_total} = {n_mixed_ext/n_total:.1%} (original: 23.8%)")

# Save summary
summary = {
    's2_mode_top_n_genes': len(s2_genes),
    's2_dom_dep_wins': int(dom_dep_wins),
    's2_motif_dep_wins': int(motif_dep_wins),
    'top_s2_specialized_gos': [(go, cnt) for go, cnt in sorted_terms[:10]],
    'mixed_specialist_pairs': len(mixed_specialist_analysis),
    'mixed_5515_diff_from_generalist': int(n_five515_diff) if mixed_specialist_analysis else 0,
    'extended_total_genes': n_total,
    'extended_s2_rate': n_s2_ext / n_total if n_total else 0,
    'extended_mixed_rate': n_mixed_ext / n_total if n_total else 0,
    'domain_dep_terms_count': len(domain_dep_terms),
}

out_path = OUT_DIR / "go_attribution_extended_summary.json"
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {out_path}")
