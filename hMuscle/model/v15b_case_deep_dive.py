# -*- coding: utf-8 -*-
"""
v15b_case_deep_dive.py
=======================
TPM1 / TPM2 / KIF1B / SEH1L — isoform switch 심층 분석
  1. 모든 isoform score 분포 (v15 기존 예측 재사용)
  2. HPA skeletal muscle vs 타 조직 발현 패턴 (전체 조직)
  3. 각 isoform별 단백질 길이 (서열 길이 proxy)
  4. 생물학적 해석용 요약 테이블

실행:
  conda run -n isoform_env python v15b_case_deep_dive.py
"""

import os, json, zipfile, csv
import numpy as np
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
FEAT_DIR  = '../results_isoform/features'
ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'
HPA_ZIP   = f'{FEAT_DIR}/hpa_isoform/transcript_rna_tissue.tsv.zip'
PREV_JSON = '../../reports/v15_switch_dtu/switch_dtu_20260519_1104.json'
OUT_DIR   = '../../reports/v15_switch_dtu'

TARGET_GENES = ['TPM1', 'TPM2', 'KIF1B', 'SEH1L']

# Known isoform biology
ISOFORM_NOTES = {
    'TPM1': {
        'biology': 'Tropomyosin-1 (alpha-TM). Regulates actin filament dynamics in sarcomere.',
        'known_switch': 'TPM1α (striated muscle, exon 2a/9a) vs non-muscle isoforms (exon 2b/9b/9c). '
                        'Only α-isoform integrates into sarcomere Z-disc (Pittenger 1994, J Cell Biol).',
        'disease': 'Hypertrophic cardiomyopathy (HCM), nemaline myopathy (NEM).',
    },
    'TPM2': {
        'biology': 'Tropomyosin-2 (beta-TM). Co-assembles with TPM1 in slow-twitch skeletal muscle.',
        'known_switch': 'Slow-twitch skeletal & cardiac isoform (exon 6b) vs smooth/non-muscle (exon 6a). '
                        'beta-TM preferentially expressed in type-I (slow) fibers (Bhatt 2021).',
        'disease': 'Nemaline myopathy type 4 (NEM4). Cap disease.',
    },
    'KIF1B': {
        'biology': 'Kinesin-3 motor. Two major isoforms: KIF1Bα (mitochondria transport) '
                   'and KIF1Bβ (synaptic vesicle transport, longer).',
        'known_switch': 'KIF1Bβ contains PH domain for vesicle binding; '
                        'KIF1Bα lacks this domain and transports mitochondria. '
                        'Functionally distinct despite shared N-terminal motor (Zhao 2001, Cell).',
        'disease': 'Charcot-Marie-Tooth type 2A (CMT2A) — KIF1Bβ haploinsufficiency.',
    },
    'SEH1L': {
        'biology': 'SEH1-like nucleoporin. Component of GATOR2 complex (mTORC1 activation).',
        'known_switch': 'SEH1L is part of both the nuclear pore complex AND the GATOR2 complex. '
                        'Different isoforms may preferentially associate with one complex. '
                        'GATOR2-bound SEH1L is required for amino acid sensing by mTORC1 (Bar-Peled 2013, Science).',
        'disease': 'mTOR pathway dysregulation in sarcopenia; GATOR2 mutations in epileptic encephalopathy.',
    },
}

# ── Load IDs ──────────────────────────────────────────────────────────────────
def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

te_geneid = load_ids('my_gene_list_fixed.npy')
te_isoid  = load_ids('my_isoform_list_fixed.npy')

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

te_sym       = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_geneid]
te_ensg_base = [g.split('.')[0] for g in te_geneid]
te_enst_base = [x.split('.')[0] for x in te_isoid]
te_enst_full = list(te_isoid)  # with version

# Build index: gene_sym → list of (enst_full, enst_base, ensg, idx)
gene_iso_idx = defaultdict(list)
for i, sym in enumerate(te_sym):
    gene_iso_idx[sym].append({
        'enst_full': te_enst_full[i],
        'enst_base': te_enst_base[i],
        'ensg': te_ensg_base[i],
        'idx': i,
    })

# ── Load ESM-2 sequence lengths as proxy for isoform length ──────────────────
# Use protein sequence length from npy shape or compute from ESM-2 norms
# (Proxy: ESM-2 embedding L2 norm correlates with protein length for large proteins)
# Better: load from actual sequence data if available
SEQ_LEN = {}  # enst_base -> aa_length (if available)

# Try to load from sequences directory
seq_fasta = f'{DATA_DIR}/raw_data/data/sequences'
if os.path.isdir(seq_fasta):
    for fname in os.listdir(seq_fasta):
        if fname.endswith('.fa') or fname.endswith('.fasta'):
            try:
                with open(f'{seq_fasta}/{fname}') as f:
                    cur_id = None
                    cur_len = 0
                    for line in f:
                        line = line.strip()
                        if line.startswith('>'):
                            if cur_id:
                                SEQ_LEN[cur_id] = cur_len
                            # parse ENST from header
                            parts = line[1:].split()
                            for p in parts:
                                if p.startswith('ENST'):
                                    cur_id = p.split('.')[0]
                                    break
                            cur_len = 0
                        else:
                            cur_len += len(line)
                    if cur_id:
                        SEQ_LEN[cur_id] = cur_len
            except: pass

# ── Load previous v15 predictions ─────────────────────────────────────────────
print("Loading previous v15 predictions ...")
prev = json.load(open(PREV_JSON))
# Extract per-gene per-iso scores from top30
prev_scores = {}  # gene -> {'top_enst', 'bot_enst', 'top_score', 'bot_score', 'score_ratio', 'dtu_top', ...}
for s in prev['top30']:
    prev_scores[s['gene']] = s

# ── HPA: full tissue breakdown for target genes ────────────────────────────────
print("Loading HPA full tissue expression ...")

TARGET_ENSG = {}
for gene in TARGET_GENES:
    isos = gene_iso_idx.get(gene, [])
    if isos:
        TARGET_ENSG[gene] = isos[0]['ensg']

# Parse HPA with per-tissue breakdown
hpa_by_gene = {}  # ensg -> {enst -> {tissue -> mean_tpm}}

with zipfile.ZipFile(HPA_ZIP) as z:
    fname = z.namelist()[0]
    with z.open(fname) as f:
        header = f.readline().decode().strip().split('\t')
        # Build tissue -> [col_indices] map
        tissue_cols = defaultdict(list)
        for i, h in enumerate(header):
            if h.startswith('TPM.'):
                parts = h.split('.')
                tissue = '.'.join(parts[1:-1])  # e.g. "skeletal muscle"
                tissue_cols[tissue].append(i)

        target_ensg_set = set(TARGET_ENSG.values())

        for line in f:
            parts = line.decode().strip().split('\t')
            if len(parts) < 3: continue
            ensg = parts[0]
            if ensg not in target_ensg_set: continue
            enst = parts[1]

            if ensg not in hpa_by_gene:
                hpa_by_gene[ensg] = {}
            hpa_by_gene[ensg][enst] = {}

            for tissue, cols in tissue_cols.items():
                vals = [float(parts[c]) for c in cols if c < len(parts)]
                if vals:
                    hpa_by_gene[ensg][enst][tissue] = round(np.mean(vals), 4)

print(f"  Loaded HPA for {len(hpa_by_gene)} target genes")

# ── Analysis per gene ──────────────────────────────────────────────────────────
# We need actual per-isoform scores — rerun v10-B for the 4 target genes
# But to avoid full 13-term rerun, use prev_scores for score info
# and supplement with HPA expression

TISSUES_OF_INTEREST = [
    'skeletal muscle', 'heart muscle', 'smooth muscle',
    'liver', 'kidney', 'cerebral cortex', 'lung',
    'adipose tissue', 'colon', 'testis', 'placenta',
]

results = {}

for gene in TARGET_GENES:
    print(f"\n{'='*65}")
    print(f"  {gene}")
    print(f"{'='*65}")

    isos = gene_iso_idx.get(gene, [])
    if not isos:
        print(f"  NOT FOUND in test set")
        continue

    ensg = isos[0]['ensg']
    n_isos = len(isos)
    print(f"  ENSG: {ensg}  |  isoforms in test set: {n_isos}")

    # Biology summary
    notes = ISOFORM_NOTES.get(gene, {})
    print(f"  Biology: {notes.get('biology','')}")
    print(f"  Known switch: {notes.get('known_switch','')}")
    print(f"  Disease: {notes.get('disease','')}")

    # Previous v15 score info
    pv = prev_scores.get(gene)
    if pv:
        print(f"\n  [v15 Prediction]")
        print(f"    Top iso: {pv['top_enst']}  score={pv['top_score']}")
        print(f"    Bot iso: {pv['bot_enst']}  score={pv['bot_score']}")
        print(f"    ratio={pv['score_ratio']}x  gap={pv['score_gap']}")
        if pv.get('has_hpa'):
            print(f"    DTU_top={pv.get('dtu_top',0):+.3f}  concordance={'✓' if pv.get('concordance') else '✗'}")
            print(f"    top_iso skm_frac={pv.get('frac_skm_top',0):.3f}  comp_frac={pv.get('frac_comp_top',0):.3f}")

    # HPA expression breakdown
    gene_hpa = hpa_by_gene.get(ensg, {})
    if not gene_hpa:
        print(f"\n  [HPA] No expression data found")
        continue

    print(f"\n  [HPA Expression]  n_isos_in_HPA={len(gene_hpa)}")

    # For each tissue of interest: dominant isoform + TPM
    print(f"\n  {'Tissue':20s}  {'Dom ENST':18s}  {'Dom TPM':8s}  {'Top ENST (model) TPM':22s}  {'Frac_dom':8s}")
    print(f"  {'-'*90}")

    top_enst_base = pv['top_enst'] if pv else None

    tissue_rows = []
    for tissue in TISSUES_OF_INTEREST:
        if tissue not in next(iter(gene_hpa.values()), {}):
            continue
        iso_tpm = {enst: data.get(tissue, 0) for enst, data in gene_hpa.items()}
        total = sum(iso_tpm.values()) + 1e-9
        if total < 0.1: continue
        dom_enst = max(iso_tpm, key=iso_tpm.get)
        dom_tpm  = iso_tpm[dom_enst]
        dom_frac = dom_tpm / total
        top_tpm  = iso_tpm.get(top_enst_base, 0) if top_enst_base else 0

        marker = '<-- model top' if dom_enst == top_enst_base else ''
        print(f"  {tissue:20s}  {dom_enst:18s}  {dom_tpm:8.2f}  {top_tpm:22.2f}  {dom_frac:.3f} {marker}")
        tissue_rows.append({
            'tissue': tissue, 'dom_enst': dom_enst, 'dom_tpm': dom_tpm,
            'dom_frac': dom_frac, 'top_enst_tpm': top_tpm,
            'model_concordant': dom_enst == top_enst_base
        })

    # All isoforms in skeletal muscle
    skm_tpm = {enst: data.get('skeletal muscle', 0) for enst, data in gene_hpa.items()}
    total_skm = sum(skm_tpm.values()) + 1e-9
    print(f"\n  [Skeletal Muscle: all isoforms]")
    print(f"  {'ENST':18s}  {'TPM':8s}  {'Frac':6s}  {'Note'}")
    print(f"  {'-'*60}")
    for enst, tpm in sorted(skm_tpm.items(), key=lambda x: -x[1]):
        frac = tpm / total_skm
        note = ''
        if pv:
            if enst == pv['top_enst']: note = '<-- model TOP'
            elif enst == pv['bot_enst']: note = '<-- model BOT'
        aa_len = SEQ_LEN.get(enst, '?')
        print(f"  {enst:18s}  {tpm:8.2f}  {frac:.3f}  {note}  (aa={aa_len})")

    # DTU summary: skeletal muscle fraction vs mean of comparison tissues
    comp_tissues = ['liver', 'kidney', 'cerebral cortex', 'lung', 'adipose tissue', 'colon']
    for enst in sorted(gene_hpa.keys()):
        skm_t = gene_hpa[enst].get('skeletal muscle', 0)
        comp_vals = [gene_hpa[enst].get(t, 0) for t in comp_tissues if t in gene_hpa[enst]]
        # store for later

    # Per-isoform DTU table
    print(f"\n  [Per-isoform DTU: skeletal muscle fraction vs comparison mean]")
    print(f"  {'ENST':18s}  {'skm_frac':9s}  {'comp_frac':9s}  {'DTU':7s}  {'Note'}")
    print(f"  {'-'*65}")
    for enst, data in sorted(gene_hpa.items()):
        skm_v  = data.get('skeletal muscle', 0)
        comp_v = np.mean([data.get(t, 0) for t in comp_tissues if t in data] or [0])
        total_s = total_skm
        total_c = sum(gene_hpa[e].get(t, 0) for e in gene_hpa for t in comp_tissues if t in gene_hpa[e]) / (len(gene_hpa) * len(comp_tissues) + 1e-9) * len(gene_hpa) + 1e-9

        # recompute per-gene fractions properly
        comp_totals = {}
        for t in comp_tissues:
            ct = sum(d.get(t, 0) for d in gene_hpa.values()) + 1e-9
            comp_totals[t] = ct

        skm_frac  = skm_v / total_skm
        comp_fracs = [gene_hpa[enst].get(t, 0) / comp_totals[t]
                     for t in comp_tissues if t in comp_totals]
        comp_frac = np.mean(comp_fracs) if comp_fracs else 0
        dtu = skm_frac - comp_frac

        note = ''
        if pv:
            if enst == pv['top_enst']: note = '<-- model TOP'
            elif enst == pv['bot_enst']: note = '<-- model BOT'
        print(f"  {enst:18s}  {skm_frac:9.4f}  {comp_frac:9.4f}  {dtu:+7.4f}  {note}")

    results[gene] = {
        'ensg': ensg,
        'n_isos_testset': n_isos,
        'n_isos_hpa': len(gene_hpa),
        'v15_score_ratio': pv['score_ratio'] if pv else None,
        'v15_dtu_top': pv.get('dtu_top') if pv else None,
        'v15_concordance': pv.get('concordance') if pv else None,
        'biology': notes.get('biology', ''),
        'known_switch': notes.get('known_switch', ''),
        'tissue_breakdown': tissue_rows,
    }

# ── Save ──────────────────────────────────────────────────────────────────────
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M')
out_path = f'{OUT_DIR}/deep_dive_{ts}.json'
json.dump(results, open(out_path, 'w'), indent=2, default=str)
print(f"\n  [Saved] {out_path}")
print("\nALL DONE")
