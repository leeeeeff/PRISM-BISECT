# -*- coding: utf-8 -*-
"""
v17_diagnosis.py
================
v17(뇌) vs v10-B(근육) 성능 차이의 근본 원인 분석

가설 3가지:
  H1. Architecture 문제: 동일 아키텍처로 근육은 0.693, 뇌는 0.036 → Architecture는 무죄
  H2. ESM-2 공간에서 GO term 분리 가능성 (sep_cosine): 근육 GO term이 뇌보다 분리 잘 됨?
  H3. Within-gene isoform variation: 뇌에서 같은 유전자 이소폼들이 너무 비슷함?

추가 분석:
  - Score distribution: 뇌 모델이 mode collapse(0.1~0.35 균일 출력)?
  - pos_bias: 뇌 모델에서 within-gene variation 측정
  - GO term 난이도: cross-species conservation proxy
    → 근육 term(Sarcomere/Motor activity) vs 뇌 term(Synaptic transmission)
    → 얼마나 다른 단백질 family가 같은 GO term을 공유하는가
"""

import re, json
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

DATA_DIR  = '../data'
BRAIN_DIR = '../data/brain_esm2'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
REP_DIR_MUSCLE = '../../reports/v15_switch_dtu'
REP_DIR_BRAIN  = '../../reports/v17_brain_model'

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("  v17 Diagnosis: Brain vs Muscle Performance Gap Analysis")
print("="*70)

# ── 1. Load embeddings and labels ────────────────────────────────────────────
print("\n[1] Loading embeddings ...")

# Muscle test isoforms
muscle_emb   = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)
muscle_mask  = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M_mask.npy')
muscle_isos  = [x.decode() if isinstance(x, bytes) else str(x)
                for x in np.load(f'{ID_DIR}/test_isoform_list.npy', allow_pickle=True)]
muscle_genes = [x.decode() if isinstance(x, bytes) else str(x)
                for x in np.load(f'{ID_DIR}/test_gene_list.npy', allow_pickle=True)]

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
muscle_syms = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in muscle_genes]

# Brain PC isoforms
brain_emb_all = np.load(f'{BRAIN_DIR}/brain_only_esm2_t30_150M.npy').astype(np.float32)
brain_enst, brain_sym, brain_biotype = [], [], []
with open(f'{BRAIN_DIR}/brain_only.gtf') as f:
    for line in f:
        if '\ttranscript\t' not in line: continue
        enst  = re.search(r'transcript_id "(ENST\d+)"', line)
        gname = re.search(r'gene_name "([^"]+)"', line)
        btype = re.search(r'transcript_type "([^"]+)"', line)
        if not enst: continue
        brain_enst.append(enst.group(1))
        brain_sym.append(gname.group(1) if gname else '')
        brain_biotype.append(btype.group(1) if btype else 'unknown')
pc_mask = np.array([b == 'protein_coding' for b in brain_biotype])
pc_idx  = np.where(pc_mask)[0]
brain_emb   = brain_emb_all[pc_idx]
brain_syms  = [brain_sym[i] for i in pc_idx]

print(f"  Muscle: {muscle_emb.shape}, Brain-PC: {brain_emb.shape}")

# GO annotations
gene2go = {}
with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
    for line in f:
        p = line.strip().split()
        if p: gene2go[p[0]] = set(p[1:])

# ── GO terms ─────────────────────────────────────────────────────────────────
MUSCLE_GO = {
    'GO:0030017': 'Sarcomere',
    'GO:0006936': 'Muscle contraction',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0055002': 'Muscle cell diff',
    'GO:0003774': 'Motor activity',
    'GO:0006094': 'Gluconeogenesis',
    'GO:0006941': 'Striated muscle contraction',
    'GO:0006631': 'Fatty acid metab',
    'GO:0007507': 'Muscle organ dev',
    'GO:0045214': 'Sarcomere org',
    'GO:0048659': 'Smooth muscle dev',
    'GO:0046034': 'ATP metabolic proc',
    'GO:0022900': 'Electron transport',
}
BRAIN_GO = {
    'GO:0043005': 'Neuron projection',
    'GO:0030182': 'Neuron diff',
    'GO:0048666': 'Neuron dev',
    'GO:0045202': 'Synapse assembly',
    'GO:0007268': 'Synaptic transmission',
    'GO:0050804': 'Mod syn.trans',
    'GO:0007411': 'Axon guidance',
    'GO:0007018': 'MT-based movement',
    'GO:0000226': 'MT cytoskeleton org',
    'GO:0006836': 'Neurotrans. transport',
}

# ── 2. ESM-2 space separability (sep_cosine) per GO term ────────────────────
print("\n[2] ESM-2 space separability (sep_cosine) ...")
print(f"  {'GO term':30s}  {'n_pos':5s}  {'n_neg':5s}  {'sep_cosine':10s}  type")
print("  " + "-"*65)

def compute_sep_cosine(emb, syms, go_term, n_sample=500):
    """
    sep_cosine = mean_sim(pos,pos) - mean_sim(pos,neg)
    Higher = positive genes more clustered in ESM-2 space.
    """
    pos_idx = [i for i, s in enumerate(syms) if s in gene2go and go_term in gene2go[s]]
    neg_idx = [i for i, s in enumerate(syms) if s not in gene2go or go_term not in gene2go[s]]
    if len(pos_idx) < 10 or len(neg_idx) < 10:
        return 0.0, len(pos_idx), len(neg_idx)
    # Sample
    rng = np.random.default_rng(42)
    pos_s = rng.choice(pos_idx, min(n_sample, len(pos_idx)), replace=False)
    neg_s = rng.choice(neg_idx, min(n_sample, len(neg_idx)), replace=False)
    E_pos = normalize(emb[pos_s])
    E_neg = normalize(emb[neg_s])
    sim_pp = float(np.mean(E_pos @ E_pos.T) - np.mean(np.eye(len(E_pos))))
    sim_pn = float(np.mean(E_pos @ E_neg.T))
    return round(sim_pp - sim_pn, 4), len(pos_idx), len(neg_idx)

muscle_sep, brain_sep = {}, {}
for go_id, name in MUSCLE_GO.items():
    sep, n_pos, n_neg = compute_sep_cosine(muscle_emb, muscle_syms, go_id)
    muscle_sep[name] = sep
    print(f"  {name:30s}  {n_pos:5d}  {n_neg:5d}  {sep:10.4f}  [MUSCLE]")

for go_id, name in BRAIN_GO.items():
    sep_m, n_pos_m, _ = compute_sep_cosine(muscle_emb, muscle_syms, go_id)
    sep_b, n_pos_b, n_neg_b = compute_sep_cosine(brain_emb, brain_syms, go_id)
    brain_sep[name] = sep_b
    print(f"  {name:30s}  {n_pos_b:5d}  {n_neg_b:5d}  {sep_b:10.4f}  [BRAIN-ESM2]  (muscle-ESM2: {sep_m:.4f})")

print(f"\n  Muscle GO avg sep:  {np.mean(list(muscle_sep.values())):.4f}")
print(f"  Brain GO avg sep:   {np.mean(list(brain_sep.values())):.4f}")

# ── 3. Within-gene isoform variation: muscle vs brain ───────────────────────
print("\n[3] Within-gene isoform variation (EMB space)")

def within_gene_emb_stats(emb, syms):
    gene_groups = defaultdict(list)
    for i, s in enumerate(syms):
        gene_groups[s].append(i)
    multi = {g: idxs for g, idxs in gene_groups.items() if len(idxs) >= 2}

    within_sim = []
    cross_sim_samples = []
    rng = np.random.default_rng(42)

    for g, idxs in list(multi.items())[:2000]:
        E = normalize(emb[idxs])
        sims = E @ E.T
        n = len(idxs)
        if n >= 2:
            upper = [sims[i,j] for i in range(n) for j in range(i+1,n)]
            within_sim.extend(upper)

    # Cross-gene (random pairs)
    all_idxs = list(range(len(syms)))
    for _ in range(5000):
        i, j = rng.choice(all_idxs, 2, replace=False)
        if syms[i] != syms[j]:
            ei = normalize(emb[i:i+1])
            ej = normalize(emb[j:j+1])
            cross_sim_samples.append(float(ei @ ej.T))

    return {
        'within_gene_sim_mean':  round(float(np.mean(within_sim)), 4),
        'within_gene_sim_std':   round(float(np.std(within_sim)), 4),
        'within_gene_sim_min':   round(float(np.min(within_sim)), 4),
        'cross_gene_sim_mean':   round(float(np.mean(cross_sim_samples)), 4),
        'cross_gene_sim_std':    round(float(np.std(cross_sim_samples)), 4),
        'n_multi_gene_isoforms': sum(len(v) for v in multi.values()),
        'n_multi_genes':         len(multi),
    }

print("  Computing muscle within-gene isoform similarity ...")
mstats = within_gene_emb_stats(muscle_emb, muscle_syms)
print(f"  Muscle: within-gene sim={mstats['within_gene_sim_mean']:.4f}±{mstats['within_gene_sim_std']:.4f}"
      f"  (min={mstats['within_gene_sim_min']:.4f})"
      f"  cross-gene sim={mstats['cross_gene_sim_mean']:.4f}±{mstats['cross_gene_sim_std']:.4f}"
      f"  n_genes={mstats['n_multi_genes']}")

print("  Computing brain within-gene isoform similarity ...")
bstats = within_gene_emb_stats(brain_emb, brain_syms)
print(f"  Brain:  within-gene sim={bstats['within_gene_sim_mean']:.4f}±{bstats['within_gene_sim_std']:.4f}"
      f"  (min={bstats['within_gene_sim_min']:.4f})"
      f"  cross-gene sim={bstats['cross_gene_sim_mean']:.4f}±{bstats['cross_gene_sim_std']:.4f}"
      f"  n_genes={bstats['n_multi_genes']}")

delta_within_m = mstats['within_gene_sim_mean'] - mstats['cross_gene_sim_mean']
delta_within_b = bstats['within_gene_sim_mean'] - bstats['cross_gene_sim_mean']
print(f"\n  Within-cross gap (higher = isoforms more similar to each other than to other genes):")
print(f"  Muscle: {delta_within_m:+.4f}  Brain: {delta_within_b:+.4f}")

# ── 4. Score distribution analysis (mode collapse?) ─────────────────────────
print("\n[4] Score distribution analysis")

score_muscle = np.load(f'{REP_DIR_MUSCLE}/score_matrix_18go_20260519_1403.npy')
score_brain  = np.load(f'{REP_DIR_BRAIN}/score_matrix_brain_v17_20260519_1458.npy')

print(f"  Muscle score matrix: {score_muscle.shape}")
print(f"  Brain score matrix:  {score_brain.shape}")
print(f"\n  {'GO term':28s}  {'M_std':6s}  {'M_mean':6s}  {'B_std':6s}  {'B_mean':6s}  {'M_max':6s}  {'B_max':6s}")
print("  " + "-"*70)

SHARED_NAMES = {
    'MT-based movement': (7, 7),   # (muscle_col, brain_col)
    'MT cytoskeleton org': (8, 8),
    'Synaptic transmission': (13, 4),
    'Neuron projection': (15, 0),
    'Autophagy': (12, 12),
}

# Use all 18 columns with their names
MUSCLE_18_NAMES = ['Ca2+sig','Sarcomere','Muscle_cont','Autophagy','Proteasome',
                   'SKM_dev','Muscle_diff','Ca2+homeo','Mitochon','Muscle_org',
                   'TOR','Motor','Glycolysis','Synaptic','MT_move','Neuron_proj','Neuron_diff','MT_cyto']
BRAIN_18_NAMES  = ['Neuron_proj','Neuron_diff','Neuron_dev','Synapse','Synaptic',
                   'Mod_syn','Axon_guid','MT_move','MT_cyto','Neurotrans',
                   'Ox_stress','Inflam','Autophagy','Proteasome','Ca2+sig',
                   'Ca2+homeo','Mitochon','TOR']

print(f"  Muscle scores across all 18 terms:")
print(f"    std_mean={score_muscle.std(axis=0).mean():.4f}  "
      f"mean_mean={score_muscle.mean(axis=0).mean():.4f}  "
      f"max_mean={score_muscle.max(axis=0).mean():.4f}")
print(f"  Brain scores across all 18 terms:")
print(f"    std_mean={score_brain.std(axis=0).mean():.4f}  "
      f"mean_mean={score_brain.mean(axis=0).mean():.4f}  "
      f"max_mean={score_brain.max(axis=0).mean():.4f}")

# Per-term std/mean/max
for gi in range(18):
    ms = score_muscle[:, gi]
    bs = score_brain[:, gi]
    print(f"  {MUSCLE_18_NAMES[gi]:12s}/{BRAIN_18_NAMES[gi]:12s}  "
          f"{ms.std():.4f}  {ms.mean():.4f}  {bs.std():.4f}  {bs.mean():.4f}  "
          f"{ms.max():.4f}  {bs.max():.4f}")

# ── 5. Pos_bias for brain model ───────────────────────────────────────────────
print("\n[5] Pos_bias (within-gene score variation) in brain vs muscle")

def compute_pos_bias(score_mat, syms, go_terms_ids, go_terms_names):
    results = {}
    for gi, (go_id, go_name) in enumerate(zip(go_terms_ids, go_terms_names)):
        scores_col = score_mat[:, gi]
        global_std = scores_col.std()
        if global_std < 1e-6:
            results[go_name] = 0.0
            continue
        gene_groups = defaultdict(list)
        for i, s in enumerate(syms):
            if s in gene2go and go_id in gene2go[s]:
                gene_groups[s].append(i)
        within_stds = []
        for g, idxs in gene_groups.items():
            if len(idxs) >= 2:
                within_stds.append(scores_col[idxs].std())
        if not within_stds:
            results[go_name] = 0.0
            continue
        pos_bias = float(np.mean(within_stds)) / float(global_std)
        results[go_name] = round(pos_bias, 4)
    return results

MUSCLE_GO_IDS   = list(MUSCLE_GO.keys())
MUSCLE_GO_NAMES = list(MUSCLE_GO.values())
BRAIN_GO_IDS    = list(BRAIN_GO.keys())
BRAIN_GO_NAMES  = list(BRAIN_GO.values())

# For muscle score matrix, use first 13 terms (original GO terms from v15d)
MUSCLE_13_IDS = [
    'GO:0007519','GO:0030017','GO:0006936','GO:0055002','GO:0003774',
    'GO:0006094','GO:0006941','GO:0045214','GO:0006631','GO:0007507',
    'GO:0048659','GO:0046034','GO:0022900',
]
MUSCLE_13_NAMES = [
    'Skeletal_dev','Sarcomere','Muscle_cont','Muscle_diff','Motor_act',
    'Gluconeo','Striated_cont','Sarcomere_org','FA_metab','Muscle_org',
    'Smooth_dev','ATP_metab','Electron_trans',
]
# Use columns 0-12 from muscle score matrix for pos_bias
m_pb = compute_pos_bias(score_muscle[:, :13], muscle_syms, MUSCLE_13_IDS, MUSCLE_13_NAMES)
b_pb = compute_pos_bias(score_brain[:, :10], brain_syms, BRAIN_GO_IDS, BRAIN_GO_NAMES)

print(f"  {'GO term':28s}  {'pos_bias':8s}  tissue")
for name, pb in sorted(m_pb.items(), key=lambda x: -x[1]):
    print(f"  {name:28s}  {pb:8.4f}  [MUSCLE]")
for name, pb in sorted(b_pb.items(), key=lambda x: -x[1]):
    print(f"  {name:28s}  {pb:8.4f}  [BRAIN]")

print(f"\n  Muscle mean pos_bias: {np.mean(list(m_pb.values())):.4f}")
print(f"  Brain  mean pos_bias: {np.mean(list(b_pb.values())):.4f}")

# ── 6. GO term protein family diversity (conservation proxy) ─────────────────
print("\n[6] GO term diversity: how many distinct protein families share the term?")
print("  (Higher diversity = harder to predict from sequence alone)")

def compute_go_diversity(go_id, emb, syms, n_sample=300):
    """Mean pairwise cosine distance between positive genes = diversity score"""
    pos_idx = [i for i, s in enumerate(syms) if s in gene2go and go_id in gene2go[s]]
    if len(pos_idx) < 10:
        return 0.0, len(pos_idx)
    rng = np.random.default_rng(42)
    samp = rng.choice(pos_idx, min(n_sample, len(pos_idx)), replace=False)
    E = normalize(emb[samp])
    sim = E @ E.T
    # Mean pairwise cosine similarity (higher = more homogeneous)
    n = len(samp)
    mean_sim = (sim.sum() - n) / (n * (n-1))
    # Diversity = 1 - similarity (lower similarity = more diverse families)
    diversity = 1.0 - float(mean_sim)
    return round(diversity, 4), len(pos_idx)

print(f"  {'GO term':30s}  {'diversity':10s}  {'n_pos':5s}  type")
print("  " + "-"*60)
for go_id, name in {**MUSCLE_GO, **dict(list(BRAIN_GO.items()))}.items():
    tissue = 'MUSCLE' if go_id in MUSCLE_GO else 'BRAIN'
    emb_use = muscle_emb if tissue == 'MUSCLE' else brain_emb
    sym_use = muscle_syms if tissue == 'MUSCLE' else brain_syms
    div, n_pos = compute_go_diversity(go_id, emb_use, sym_use)
    print(f"  {name:30s}  {div:10.4f}  {n_pos:5d}  [{tissue}]")

# ── 7. Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SUMMARY: Root Cause Analysis")
print("="*70)
print(f"""
  Muscle model (AUPRC=0.693):
    - sep_cosine (avg):    {np.mean(list(muscle_sep.values())):.4f}
    - within-gene gap:     {delta_within_m:+.4f}
    - score std (avg):     {score_muscle.std(axis=0).mean():.4f}
    - pos_bias (avg):      {np.mean(list(m_pb.values())):.4f}

  Brain model (AUPRC=0.036):
    - sep_cosine (avg):    {np.mean(list(brain_sep.values())):.4f}
    - within-gene gap:     {delta_within_b:+.4f}
    - score std (avg):     {score_brain.std(axis=0).mean():.4f}
    - pos_bias (avg):      {np.mean(list(b_pb.values())):.4f}
""")

print("ALL DONE")
