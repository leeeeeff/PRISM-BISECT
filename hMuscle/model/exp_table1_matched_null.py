#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_table1_matched_null.py
==========================
[A] Table 1 matched-null: is the 22.1% (23/104) literature-support rate among
PRISM-selected beyond-gene-label pairs ENRICHED relative to random (gene, GO-term)
pairs drawn from the SAME population strata and screened by the SAME criterion?

Answers the devils-advocate "cherry-pick" critique with data (not speculation).

Design (pre-registered; see session notes 2026-07-17):
  Observed 104 pairs = 40 highest-scoring Case A instances + all 64 Case B instances
      (reconstructed identically to analyze_novel_go_acquisition.py, TRUE brain).
  Screening criterion (applied IDENTICALLY to observed & null, offline gene2go):
      supported_primary(gene, target) <=>
          gene has >=1 BP GO annotation with a DIRECT EXPERIMENTAL code
          (IDA/IMP/IPI/IGI/IEP) whose term lies in the GO-DAG closure of target
          (target U ancestors U descendants ; is_a + part_of).
      supported_loose(gene, target) <=>  gene has ANY experimental BP code
          (pure study-bias floor; if observed fails this, dead).
  Matched null (removes PRISM selection, preserves population + term distribution):
      Case A null: gene ~ Uniform(fully-negative brain genes), term ~ observed CaseA term freq.
      Case B null: gene ~ Uniform(partially-annotated brain genes),
                   term ~ observed CaseB term freq, resampled to be a FOREIGN term (not annotated).
      B=2000 resamples -> null distribution of support rate over 104 matched pairs.
  Report: observed_auto rate (test statistic) vs null mean±SD, empirical p, z, CI.
      Manual 22.1% (23/104) is the human-curated headline, NOT the test statistic
      (automated closeness != manual "closely-matching"); reported alongside for context.

Data sources (all offline, read-only):
  brain score matrix / ids / gene names : same as analyze_novel_go_acquisition.py
  gene2go.gz + Homo_sapiens.gene_info.gz : evidence codes, symbol<->GeneID
  go.obo (2019-01)                       : GO DAG closure
"""
import os, gzip, json
import numpy as np
from collections import defaultdict
import networkx as nx

os.chdir(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(20260717)
B_RESAMPLE = 2000

BRAIN_DIR = '../data/brain_isoquant_esm2/full'
ANNOT_DIR = '../data/raw_data/data/annotations'
SCORE_MAT = '../../reports/v15d_brain_eval/brain_full_score_matrix_20260519_2125.npy'
OBO       = '../../reports/benchmark_external/deepgoplus/data/deepgoplus_data/go.obo'
OUT_DIR   = '../../reports/novel_go_matched_null'
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD = 0.5
EXP_CODES = {'IDA', 'IMP', 'IPI', 'IGI', 'IEP'}

# ---- the 18 BP terms (identical to analyze_novel_go_acquisition.py) ---------
GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling', 'GO:0045214': 'Sarcomere organization',
    'GO:0006941': 'Muscle contraction', 'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS', 'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff', 'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org', 'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling', 'GO:0030048': 'Actin-based movement',
    'GO:0006096': 'Glycolysis', 'GO:0007268': 'Synaptic transmission',
    'GO:0007018': 'MT-based movement', 'GO:0031175': 'Neuron proj development',
    'GO:0030182': 'Neuron diff', 'GO:0000226': 'MT cytoskeleton org',
}
GO_KEYS = list(GO_TERMS.keys()); N_GO = len(GO_KEYS)


def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


# ============================================================================
# 1. Reconstruct observed Case A/B (identical logic to analyze_novel_go_acquisition.py)
# ============================================================================
print("=" * 72)
print("  [1] Reconstructing observed 104 pairs (40 top Case A + 64 Case B)")
print("=" * 72)
te_isoid = load_ids(f'{BRAIN_DIR}/brain_full_ids.npy')
te_sym   = load_ids(f'{BRAIN_DIR}/brain_full_gene_names.npy')
N_TE = len(te_isoid)
score_matrix = np.load(SCORE_MAT).astype(np.float32)
assert score_matrix.shape == (N_TE, N_GO), score_matrix.shape

gene_pos = defaultdict(set)   # gene_sym -> set of GO indices positive (18-term vector)
for k_idx, go_term in enumerate(GO_KEYS):
    with open(f'{ANNOT_DIR}/human_annotations_unified_bp.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                gene_pos[parts[0]].add(k_idx)

gene_to_idx = defaultdict(list)
for i, sym in enumerate(te_sym):
    gene_to_idx[sym].append(i)

caseA, caseB = [], []
for g, idxs in gene_to_idx.items():
    pos_k = gene_pos.get(g, set())
    if len(pos_k) == 0:
        for i in idxs:
            row = score_matrix[i]
            for k in np.where(row > THRESHOLD)[0]:
                caseA.append((g, te_isoid[i], GO_KEYS[k], float(row[k])))
    else:
        neg_k = [k for k in range(N_GO) if k not in pos_k]
        for i in idxs:
            row = score_matrix[i]
            own_best = row[list(pos_k)].max()
            for k in neg_k:
                if row[k] > THRESHOLD and row[k] > own_best:
                    caseB.append((g, te_isoid[i], GO_KEYS[k], float(row[k]), float(own_best)))

fully_neg_genes = [g for g in gene_to_idx if len(gene_pos.get(g, set())) == 0]
partial_genes   = [g for g in gene_to_idx if len(gene_pos.get(g, set())) > 0]

# ---- per-GENE reduction (Table S_novelGO construction: one representative
#      pair per affected gene) --------------------------------------------
# Case A: 378 affected genes ranked by best score -> top 40, each -> (gene, best-score term)
caseA_best = {}   # gene -> (term, score)  keep max-score instance per gene
for (g, iso, k, s) in caseA:
    if g not in caseA_best or s > caseA_best[g][1]:
        caseA_best[g] = (k, s)
caseA_genes_ranked = sorted(caseA_best.items(), key=lambda kv: -kv[1][1])
caseA_top40 = caseA_genes_ranked[:40]
# Case B: all 64 affected genes ranked by gap, each -> (gene, top-gap foreign term)
caseB_best = {}   # gene -> (term, gap)
for (g, iso, k, s, ob) in caseB:
    gap = s - ob
    if g not in caseB_best or gap > caseB_best[g][1]:
        caseB_best[g] = (k, gap)
print(f"  Case A affected genes={len(caseA_best)} (manuscript: 378), take top-40 by score")
print(f"  Case B affected genes={len(caseB_best)}  (manuscript: 64)")
print(f"  fully-negative genes={len(fully_neg_genes)}  partial genes={len(partial_genes)}")

# observed pairs: (gene, target_go, case) -- one per gene
obs_pairs  = [(g, kv[0], 'A') for (g, kv) in caseA_top40] \
           + [(g, kv[0], 'B') for (g, kv) in caseB_best.items()]
print(f"  OBSERVED pairs = {len(obs_pairs)}  (target: 104)")

# observed term-frequency (for matched null term sampling)
obsA_terms = [k for (g, k, c) in obs_pairs if c == 'A']
obsB_terms = [k for (g, k, c) in obs_pairs if c == 'B']


# ============================================================================
# 2. gene2go offline: symbol -> experimental BP terms
# ============================================================================
print("\n" + "=" * 72)
print("  [2] Building offline gene2go BP evidence map (human)")
print("=" * 72)
# symbol -> GeneID (primary Symbol + synonyms as fallback)
sym2gid, syn2gid = {}, {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) < 5 or p[0] != '9606':
            continue
        gid, symbol, syns = p[1], p[2], p[4]
        sym2gid[symbol] = gid
        for s in syns.split('|'):
            if s and s != '-':
                syn2gid.setdefault(s, gid)

def resolve_gid(sym):
    return sym2gid.get(sym) or syn2gid.get(sym)

# GeneID -> list of (GO_ID, evidence) for BP (Category == Process)
gid_bp = defaultdict(list)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) < 8 or p[0] != '9606' or p[7] != 'Process':
            continue
        gid_bp[p[1]].append((p[2], p[3]))

def gene_exp_bp(sym):
    """Return set of experimentally-supported BP GO terms for a gene symbol."""
    gid = resolve_gid(sym)
    if gid is None:
        return set()
    return {go for (go, ev) in gid_bp.get(gid, []) if ev in EXP_CODES}

n_map = sum(1 for g in set(g for g, k, c in obs_pairs) if resolve_gid(g))
print(f"  gene_info symbols={len(sym2gid)}  synonyms={len(syn2gid)}")
print(f"  gene2go BP genes={len(gid_bp)}")
print(f"  observed distinct genes mappable to GeneID: {n_map}/"
      f"{len(set(g for g,k,c in obs_pairs))}")


# ============================================================================
# 3. GO DAG closure (is_a + part_of) for each target term
# ============================================================================
print("\n" + "=" * 72)
print("  [3] Building GO DAG + target-term closures")
print("=" * 72)
G = nx.DiGraph()      # is_a + part_of only (strict structural closure)
G_reg = nx.DiGraph()  # + regulates/pos/neg (approximates manual "closely-matching")
REG_PREFIXES = ('relationship: regulates ', 'relationship: positively_regulates ',
                'relationship: negatively_regulates ')
cur, obsolete = None, False
with open(OBO) as f:
    for line in f:
        line = line.rstrip('\n')
        if line == '[Term]':
            cur, obsolete = None, False
        elif line.startswith('id: GO:'):
            cur = line[4:]
        elif line == 'is_obsolete: true':
            obsolete = True
        elif line.startswith('is_a: ') and cur and not obsolete:
            tgt = line[6:].split(' ! ')[0]
            G.add_edge(cur, tgt); G_reg.add_edge(cur, tgt)
        elif line.startswith('relationship: part_of ') and cur and not obsolete:
            tgt = line[len('relationship: part_of '):].split(' ! ')[0]
            G.add_edge(cur, tgt); G_reg.add_edge(cur, tgt)
        elif cur and not obsolete and line.startswith(REG_PREFIXES):
            # "regulation of X" --regulates--> X ; treat regulator as related-to X
            tgt = line.split(' ', 2)[2].split(' ! ')[0]
            G_reg.add_edge(cur, tgt)
print(f"  GO DAG strict : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"  GO DAG +reg   : {G_reg.number_of_nodes()} nodes, {G_reg.number_of_edges()} edges")

def build_closure(graph):
    cl = {}
    for k in GO_KEYS:
        c = {k}
        if k in graph:
            c |= nx.descendants(graph, k)  # more general (ancestors)
            c |= nx.ancestors(graph, k)    # more specific + regulators
        cl[k] = c
    return cl

closure = build_closure(G)
closure_reg = build_closure(G_reg)
for k in GO_KEYS:
    print(f"    {k} {GO_TERMS[k]:26s} closure={len(closure[k]):4d}  +reg={len(closure_reg[k]):4d}")


# ============================================================================
# 4. Screening functions (identical for observed & null)
# ============================================================================
def supported_primary(sym, target):
    return len(gene_exp_bp(sym) & closure[target]) > 0

def supported_reg(sym, target):
    return len(gene_exp_bp(sym) & closure_reg[target]) > 0

def supported_loose(sym, target):
    return len(gene_exp_bp(sym)) > 0


# ============================================================================
# 5. Observed automated support
# ============================================================================
print("\n" + "=" * 72)
print("  [5] Observed automated support (test statistic)")
print("=" * 72)
obs_prim = [(g, k, c, supported_primary(g, k)) for (g, k, c) in obs_pairs]
obs_reg  = [(g, k, c, supported_reg(g, k)) for (g, k, c) in obs_pairs]
obs_loose = [(g, k, c, supported_loose(g, k)) for (g, k, c) in obs_pairs]
n_prim = sum(1 for *_, s in obs_prim if s)
n_reg  = sum(1 for *_, s in obs_reg if s)
n_loose = sum(1 for *_, s in obs_loose if s)
obs_prim_rate = n_prim / len(obs_pairs)
obs_reg_rate  = n_reg / len(obs_pairs)
obs_loose_rate = n_loose / len(obs_pairs)
print(f"  observed PRIMARY (is_a/part_of closure) : {n_prim}/{len(obs_pairs)} = {obs_prim_rate:.3f}")
print(f"  observed REG (+regulates closure)       : {n_reg}/{len(obs_pairs)} = {obs_reg_rate:.3f}")
print(f"  observed LOOSE (any-exp floor)          : {n_loose}/{len(obs_pairs)} = {obs_loose_rate:.3f}")
print(f"  [context] manual headline = 23/104 = {23/104:.3f}")
# by case
for cs in ['A', 'B']:
    tot = sum(1 for g, k, c in obs_pairs if c == cs)
    sp  = sum(1 for g, k, c, s in obs_prim if c == cs and s)
    print(f"    Case {cs}: primary {sp}/{tot}")
print("  supported (primary) observed pairs:")
for g, k, c, s in obs_prim:
    if s:
        print(f"    [{c}] {g:14s} {k} {GO_TERMS[k]}")


# ============================================================================
# 6. Matched null
# ============================================================================
print("\n" + "=" * 72)
print(f"  [6] Matched null  (B={B_RESAMPLE} resamples)")
print("=" * 72)
partial_pos = {g: gene_pos[g] for g in partial_genes}
fully_neg_arr = np.array(fully_neg_genes)
partial_arr   = np.array(partial_genes)
nA, nB = len(obsA_terms), len(obsB_terms)

null_prim = np.zeros(B_RESAMPLE)
null_reg  = np.zeros(B_RESAMPLE)
null_loose = np.zeros(B_RESAMPLE)
for b in range(B_RESAMPLE):
    n_p = n_r = n_l = 0
    # Case A matched: random fully-neg gene x observed-freq term
    gA = fully_neg_arr[RNG.integers(0, len(fully_neg_arr), nA)]
    tA = [obsA_terms[i] for i in RNG.integers(0, nA, nA)]
    for g, k in zip(gA, tA):
        if supported_primary(g, k): n_p += 1
        if supported_reg(g, k):     n_r += 1
        if supported_loose(g, k):   n_l += 1
    # Case B matched: random partial gene x observed-freq FOREIGN term
    gB = partial_arr[RNG.integers(0, len(partial_arr), nB)]
    for g in gB:
        # sample a term from observed CaseB freq that this gene is NOT annotated to
        posk = partial_pos[g]
        for _ in range(20):
            k = obsB_terms[RNG.integers(0, nB)]
            if GO_KEYS.index(k) not in posk:
                break
        if supported_primary(g, k): n_p += 1
        if supported_reg(g, k):     n_r += 1
        if supported_loose(g, k):   n_l += 1
    null_prim[b] = n_p / len(obs_pairs)
    null_reg[b]  = n_r / len(obs_pairs)
    null_loose[b] = n_l / len(obs_pairs)
    if (b + 1) % 500 == 0:
        print(f"    ...{b+1}/{B_RESAMPLE}")

def summarize(name, obs_rate, null_arr):
    mean, sd = null_arr.mean(), null_arr.std(ddof=1)
    lo, hi = np.percentile(null_arr, [2.5, 97.5])
    p_emp = (np.sum(null_arr >= obs_rate) + 1) / (len(null_arr) + 1)
    z = (obs_rate - mean) / sd if sd > 0 else float('nan')
    lift = obs_rate / mean if mean > 0 else float('inf')
    print(f"\n  --- {name} ---")
    print(f"    observed          = {obs_rate:.3f}")
    print(f"    null mean +- SD   = {mean:.3f} +- {sd:.3f}")
    print(f"    null 95% CI       = [{lo:.3f}, {hi:.3f}]")
    print(f"    lift (obs/null)   = {lift:.2f}x")
    print(f"    z-score           = {z:.2f}")
    print(f"    empirical p       = {p_emp:.4g}")
    return dict(observed=obs_rate, null_mean=mean, null_sd=sd,
                null_ci=[lo, hi], lift=lift, z=z, p_emp=p_emp)

res_prim = summarize("PRIMARY (is_a/part_of closure, experimental)", obs_prim_rate, null_prim)
res_reg  = summarize("REG (+regulates closure; approximates manual)", obs_reg_rate, null_reg)
res_loose = summarize("LOOSE (any experimental BP; study-bias floor)", obs_loose_rate, null_loose)


# ============================================================================
# 7. Verdict + save
# ============================================================================
print("\n" + "=" * 72)
print("  [7] PRE-REGISTERED VERDICT")
print("=" * 72)
verdict = ("ENRICHED -> Table 1 stays MAIN"
           if res_prim['p_emp'] < 0.05 and obs_prim_rate > res_prim['null_ci'][1]
           else "NOT SIGNIFICANT -> move Table 1 + Case A/B to Supplementary")
print(f"  PRIMARY p_emp={res_prim['p_emp']:.4g}, obs {obs_prim_rate:.3f} vs "
      f"null CI hi {res_prim['null_ci'][1]:.3f}")
print(f"  --> {verdict}")

out = {
    'n_observed_pairs': len(obs_pairs),
    'n_caseA': sum(1 for g, k, c in obs_pairs if c == 'A'),
    'n_caseB': sum(1 for g, k, c in obs_pairs if c == 'B'),
    'manual_headline_rate': 23 / 104,
    'observed_primary_rate': obs_prim_rate,
    'observed_primary_count': n_prim,
    'observed_reg_rate': obs_reg_rate,
    'observed_reg_count': n_reg,
    'observed_loose_rate': obs_loose_rate,
    'observed_loose_count': n_loose,
    'B_resample': B_RESAMPLE,
    'result_primary': res_prim,
    'result_reg': res_reg,
    'result_loose': res_loose,
    'verdict': verdict,
    'observed_supported_primary': [
        {'case': c, 'gene': g, 'go': k, 'name': GO_TERMS[k]}
        for g, k, c, s in obs_prim if s],
}
with open(f'{OUT_DIR}/results.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n  [Saved] {OUT_DIR}/results.json")
