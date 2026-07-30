#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_g_uniprot_benchmark.py
--------------------------
UniProt 기반 isoform-specific 기능 차이 ground-truth 벤치마크 구축.

NatMeth 요건:
  "PRISM이 실험적으로 검증된 isoform-specific GO 기능 차이를 예측할 수 있는가?"

방법:
  1. UniProt SwissProt (reviewed) 인간 단백질 중 대체 isoform이 존재하는 항목 수집
  2. 각 isoform 쌍에 대해 pfam2go로 도메인→GO 차이 계산
  3. 실험 증거 기반 이형체 특이적 기능 차이를 benchmark CSV로 저장
  4. (선택) PRISM 예측값과 비교, within-gene AUROC 계산

핵심 벤치마크 설계:
  각 행: (gene_symbol, iso_A_id, iso_B_id, go_term, direction, evidence_type)
  direction: A_only / B_only / both / neither
  evidence_type: pfam2go_domain_loss / uniprot_isoform_note / experimental

출력:
  ../../reports/exp_g_uniprot/uniprot_isoform_benchmark.csv
  ../../reports/exp_g_uniprot/benchmark_stats.txt
  ../../reports/exp_g_uniprot/prism_eval.tsv  (PRISM 점수와 비교, 선택)

실행:
  cd hMuscle/model/
  conda activate isoform_env
  python3 exp_g_uniprot_benchmark.py

  # PRISM 예측 포함 평가:
  python3 exp_g_uniprot_benchmark.py --eval-prism
"""

import os, re, json, time, argparse, csv
from pathlib import Path
from urllib import request, parse
from urllib.error import HTTPError
import numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR   = '../../reports/exp_g_uniprot'
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

UNIPROT_REST = 'https://rest.uniprot.org'
MAX_RETRIES  = 3
SLEEP_SEC    = 1.0

PFAM2GO_PATH = '../data/pfam2go'  # Pfam→GO mapping (pfam2go r2023_06)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--eval-prism', action='store_true',
                   help='After building benchmark, evaluate PRISM predictions')
    p.add_argument('--max-proteins', type=int, default=5000,
                   help='Max UniProt entries to process (default 5000)')
    p.add_argument('--evidence', choices=['all', 'experimental', 'pfam2go'],
                   default='all', help='Evidence type filter')
    return p.parse_args()


# ── UniProt REST helpers ───────────────────────────────────────────────────────
HEADERS = {
    'Accept': 'application/json',
    'User-Agent': 'Mozilla/5.0 (PRISM-benchmark/1.0; research)',
}


def uniprot_query(url, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            req = request.Request(url, headers=HEADERS)
            with request.urlopen(req, timeout=30) as r:
                body = r.read().decode()
                # Link header for pagination (RFC 5988)
                link_header = r.headers.get('Link', '')
                return json.loads(body), link_header
        except HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
            elif e.code in (400, 404):
                return None, ''
            else:
                time.sleep(SLEEP_SEC)
        except Exception:
            time.sleep(SLEEP_SEC)
    return None, ''


def parse_next_link(link_header):
    """Extract 'next' URL from Link header: <url>; rel="next"."""
    if not link_header:
        return None
    for part in link_header.split(','):
        part = part.strip()
        m_url = re.search(r'<([^>]+)>', part)
        m_rel = re.search(r'rel="([^"]+)"', part)
        if m_url and m_rel and m_rel.group(1) == 'next':
            return m_url.group(1)
    return None


def fetch_human_multiisoform_entries(max_n=5000):
    """
    Retrieve SwissProt human proteins with ≥2 annotated isoforms.
    Uses UniProt REST v2 with Link-header pagination.
    """
    cache_path = os.path.join(CACHE_DIR, f'human_multiisoform_{max_n}.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached:
            print(f"  Loaded {len(cached)} entries from cache.")
            return cached

    print("  Querying UniProt for human proteins with multiple isoforms...")
    # Use simple space-separated query (no URL encoding of colons in field names)
    query  = 'reviewed:true AND organism_id:9606'
    fields = 'accession,gene_names,cc_alternative_products'
    url    = (f'{UNIPROT_REST}/uniprotkb/search'
              f'?query={parse.quote(query)}'
              f'&fields={parse.quote(fields)}'
              f'&format=json&size=500')

    accessions = []
    page_count = 0
    next_url   = url

    while next_url and len(accessions) < max_n:
        data, link_hdr = uniprot_query(next_url)
        if data is None:
            break

        for entry in data.get('results', []):
            n_isoforms = 0
            for c in entry.get('comments', []):
                if c.get('commentType') == 'ALTERNATIVE PRODUCTS':
                    n_isoforms = len(c.get('isoforms', []))
                    break
            if n_isoforms >= 2:
                accessions.append({
                    'accession': entry['primaryAccession'],
                    'gene':      _extract_gene(entry),
                    'n_isoforms': n_isoforms,
                })

        page_count += 1
        if page_count % 5 == 0:
            print(f"    fetched {len(accessions)} multi-isoform entries "
                  f"(page {page_count})...", flush=True)

        next_url = parse_next_link(link_hdr)
        time.sleep(SLEEP_SEC)

    accessions = accessions[:max_n]
    with open(cache_path, 'w') as f:
        json.dump(accessions, f, indent=2)
    print(f"  Saved {len(accessions)} entries to cache.")
    return accessions


def _extract_gene(entry):
    genes = entry.get('genes', [])
    if genes:
        gn = genes[0].get('geneName', {})
        return gn.get('value', '')
    return ''


def fetch_entry_detail(accession):
    """Fetch full entry with isoform sequences and GO annotations."""
    cache_path = os.path.join(CACHE_DIR, f'{accession}.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    url   = (f'{UNIPROT_REST}/uniprotkb/{accession}'
             f'?fields=accession,sequence,ft_var_seq,xref_pfam,go,'
             f'cc_alternative_products&format=json')
    data, _ = uniprot_query(url)
    if data:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    time.sleep(SLEEP_SEC * 0.5)
    return data


# ── Pfam → GO mapping ─────────────────────────────────────────────────────────
def load_pfam2go(pfam2go_path):
    """
    Load pfam2go mapping file.
    Format: Pfam:PF00001 4-helical cytokines > GO:type I cytokine receptor activity ; GO:0004912
    Returns dict: pfam_id → set of GO IDs
    """
    mapping = defaultdict(set)
    paths_to_try = [
        pfam2go_path,
        os.path.join(pfam2go_path, 'pfam2go'),
        '../data/domain/pfam2go',
        '../../data/pfam2go',
        os.path.join(CACHE_DIR, 'pfam2go'),
    ]
    found = None
    for p in paths_to_try:
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            found = p; break

    if found is None:
        print("  pfam2go file not found. Attempting download...")
        url  = 'https://current.geneontology.org/ontology/external2go/pfam2go'
        dest = os.path.join(CACHE_DIR, 'pfam2go')
        try:
            req = request.Request(url, headers=HEADERS)
            with request.urlopen(req, timeout=30) as r:
                with open(dest, 'wb') as f:
                    f.write(r.read())
            found = dest
            print(f"  Downloaded pfam2go to {dest}")
        except Exception as e:
            print(f"  [WARN] Could not download pfam2go: {e}")
            return mapping

    with open(found) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!'):
                continue
            m = re.match(r'Pfam:(PF\d+)\s.+?;\s*(GO:\d+)', line)
            if m:
                mapping[m.group(1)].add(m.group(2).replace(':', '_'))
    print(f"  pfam2go: {len(mapping)} Pfam domains mapped")
    return mapping


# ── Isoform domain difference → GO difference ────────────────────────────────
def build_isoform_go_differences(entry_data, pfam2go_map):
    """
    For a UniProt entry, derive expected GO term differences between isoforms
    based on domain gain/loss (pfam2go).

    Returns list of dicts:
      {gene, iso_a, iso_b, go_term, direction, domain, evidence}
    """
    records = []
    if not entry_data:
        return records

    accession = entry_data.get('primaryAccession', '')
    genes     = entry_data.get('genes', [])
    gene_sym  = genes[0].get('geneName', {}).get('value', '') if genes else ''

    # Canonical Pfam domains
    xrefs    = entry_data.get('uniProtKBCrossReferences', [])
    pfam_ids = set()
    for xr in xrefs:
        if xr.get('database') == 'Pfam':
            pfam_ids.add(xr['id'])

    # GO IDs from canonical (all evidence levels)
    canonical_go_mf = set()
    for go_entry in entry_data.get('uniProtKBCrossReferences', []):
        if go_entry.get('database') == 'GO':
            go_id = go_entry['id'].replace(':', '_')
            props = {p['key']: p['value'] for p in go_entry.get('properties', [])}
            if props.get('GoTerm', '').startswith('F:'):  # MF only
                canonical_go_mf.add(go_id)

    # Get alternative sequences (var_seq features)
    var_seqs = []
    for feat in entry_data.get('features', []):
        if feat.get('type') == 'Alternative sequence':
            isoforms = feat.get('featureCrossReferences', [])
            iso_ids  = [r['id'] for r in isoforms if r.get('database') == 'UniProtKB isoform']
            note     = feat.get('description', '')
            var_seqs.append({'iso_ids': iso_ids, 'note': note, 'feature': feat})

    # For isoforms with variant sequences that remove Pfam-annotated regions,
    # infer GO term loss
    for var in var_seqs:
        for iso_id in var['iso_ids']:
            note = var['note'].lower()
            # Heuristic: if the note mentions domain loss
            domain_lost = None
            for pfam_id in pfam_ids:
                # We can't run hmmscan here, but we can check if the note
                # mentions a known domain name
                pass

            # Use note-based evidence for known functional isoforms
            if any(kw in note for kw in ['missing', 'truncat', 'absent', 'deleted',
                                          'lacks', 'no ', 'without']):
                # This isoform likely loses some function
                for pfam_id in pfam_ids:
                    for go_id in pfam2go_map.get(pfam_id, set()):
                        if go_id in canonical_go_mf:
                            records.append({
                                'gene':       gene_sym,
                                'accession':  accession,
                                'iso_a':      f'{accession}-1',  # canonical
                                'iso_b':      iso_id,
                                'go_term':    go_id,
                                'direction':  'A_only',  # canonical has it, isoform doesn't
                                'domain':     pfam_id,
                                'evidence':   'pfam2go_var_seq_note',
                                'note':       note[:200],
                            })

    return records


# ── High-confidence curated pairs ─────────────────────────────────────────────
CURATED_PAIRS = [
    # fmt: (gene, uniprot_acc, iso_a, iso_b, go_term, direction, evidence)
    # direction: iso_a has GO term but iso_b does NOT (A_only) or vice versa (B_only)

    # ── Kinases / signalling ──────────────────────────────────────────────────
    ('EGFR',   'P00533', 'P00533-1', 'P00533-4',
     'GO_0004714', 'A_only', 'experimental_lit'),   # kinase domain lost in secreted iso
    ('FGFR1',  'P11362', 'P11362-1', 'P11362-11',
     'GO_0005007', 'A_only', 'experimental_lit'),   # FGF binding
    ('NTRK1',  'P04629', 'P04629-1', 'P04629-2',
     'GO_0004714', 'A_only', 'experimental_lit'),   # neurotrophin receptor kinase
    ('VEGFR2', 'P35968', 'P35968-1', 'P35968-2',
     'GO_0004714', 'A_only', 'experimental_lit'),   # kinase-dead soluble VEGFR2 (sVEGFR2)
    ('CDK4',   'P11802', 'P11802-1', 'P11802-2',
     'GO_0004693', 'A_only', 'experimental_lit'),   # cyclin-dependent kinase activity

    # ── Apoptosis / Bcl-2 family ──────────────────────────────────────────────
    ('BCL2',   'P10415', 'P10415-1', 'P10415-2',
     'GO_0097553', 'A_only', 'experimental_lit'),   # calcium channel inhibition (anti-apop)
    ('MCL1',   'Q07820', 'Q07820-1', 'Q07820-2',
     'GO_0097553', 'B_only', 'experimental_lit'),   # MCL1-S pro-apoptotic (no BH3-only)
    ('BCL2L1', 'Q07817', 'Q07817-1', 'Q07817-2',
     'GO_0005244', 'B_only', 'experimental_lit'),   # Bcl-xS promotes apoptosis
    ('CASP9',  'P55211', 'P55211-1', 'P55211-2',
     'GO_0004197', 'A_only', 'experimental_lit'),   # caspase-9b dominant-negative

    # ── DNA damage / repair ───────────────────────────────────────────────────
    ('BRCA1',  'P38398', 'P38398-1', 'P38398-2',
     'GO_0006281', 'A_only', 'experimental_lit'),   # BRCA1-Δ11 lacks BRCT-dependent repair
    ('TP53',   'P04637', 'P04637-1', 'P04637-9',
     'GO_0006977', 'A_only', 'experimental_lit'),   # Δ40p53 lacks transactivation domain
    ('CDKN2A', 'P42771', 'P42771-1', 'P42771-2',
     'GO_0008285', 'A_only', 'experimental_lit'),   # p16INK4a vs p14ARF (unrelated seqs)
    ('CDKN2A', 'P42771', 'P42771-2', 'P42771-1',
     'GO_0006977', 'B_only', 'experimental_lit'),   # p14ARF has MDM2 binding, p16 doesn't

    # ── Growth factors / ligands ───────────────────────────────────────────────
    ('VEGFA',  'P15692', 'P15692-1', 'P15692-7',
     'GO_0005178', 'A_only', 'experimental_lit'),   # heparin binding lost in VEGF121
    ('IGF1',   'P05019', 'P05019-1', 'P05019-2',
     'GO_0005158', 'A_only', 'experimental_lit'),   # IGF1 receptor binding
    ('FGF2',   'P09038', 'P09038-1', 'P09038-4',
     'GO_0008083', 'both',   'experimental_lit'),   # both bind FGFR but nuclear vs secreted

    # ── RNA binding / splicing ────────────────────────────────────────────────
    ('PCBP1',  'Q15365', 'Q15365-1', 'Q15365-2',
     'GO_0003723', 'A_only', 'experimental_lit'),   # KH domains → RNA binding
    ('HNRNPA1', 'P09651', 'P09651-1', 'P09651-2',
     'GO_0003729', 'A_only', 'experimental_lit'),   # mRNA binding, A1b lacks RGG box
    ('SRSF1',  'Q07955', 'Q07955-1', 'Q07955-3',
     'GO_0000398', 'A_only', 'experimental_lit'),   # mRNA splicing; SRSF1-3 loses RRM2
    ('RBFOX1', 'O43251', 'O43251-1', 'O43251-3',
     'GO_0000398', 'A_only', 'experimental_lit'),   # neuronal splicing regulation

    # ── Transcription factors ─────────────────────────────────────────────────
    ('RUNX1',  'Q01196', 'Q01196-1', 'Q01196-2',
     'GO_0006357', 'A_only', 'experimental_lit'),   # full-length vs RUNX1a (hematopoiesis)
    ('TP63',   'Q9H3D4', 'Q9H3D4-1', 'Q9H3D4-5',
     'GO_0006357', 'A_only', 'experimental_lit'),   # TAp63 vs ΔNp63 (opposite targets)
    ('TP73',   'O15350', 'O15350-1', 'O15350-3',
     'GO_0006977', 'A_only', 'experimental_lit'),   # TAp73 vs ΔNp73 (dominant-negative)

    # ── Cytoskeletal / structural ─────────────────────────────────────────────
    ('ACTN1',  'P12814', 'P12814-1', 'P12814-2',
     'GO_0005200', 'A_only', 'experimental_lit'),   # muscle vs non-muscle actin binding
    ('FLNB',   'O75369', 'O75369-1', 'O75369-5',
     'GO_0005200', 'A_only', 'experimental_lit'),   # filamin-B actin binding, exon 30B

    # ── Neuronal / AD-relevant ────────────────────────────────────────────────
    ('APP',    'P05067', 'P05067-1', 'P05067-7',
     'GO_0005178', 'A_only', 'experimental_lit'),   # APP695 (neuronal) lacks BPTI domain
    ('MAPT',   'P10636', 'P10636-1', 'P10636-5',
     'GO_0008017', 'A_only', 'experimental_lit'),   # 4R vs 3R tau, MT binding difference
    ('PARK2',  'O60260', 'O60260-1', 'O60260-3',
     'GO_0004842', 'A_only', 'experimental_lit'),   # parkin E3 ligase; isoform 3 lacks UBL
    ('DLG4',   'P78352', 'P78352-1', 'P78352-2',
     'GO_0007399', 'A_only', 'experimental_lit'),   # PSD-95 vs PSD-95β neuronal scaffold
    ('SNAP25', 'P60880', 'P60880-1', 'P60880-2',
     'GO_0016079', 'A_only', 'experimental_lit'),   # SNARE complex / secretory vesicle
]


def build_curated_benchmark():
    """Return curated pairs as structured records."""
    records = []
    for gene, acc, iso_a, iso_b, go_term, direction, ev in CURATED_PAIRS:
        records.append({
            'gene':      gene,
            'accession': acc,
            'iso_a':     iso_a,
            'iso_b':     iso_b,
            'go_term':   go_term,
            'direction': direction,
            'evidence':  ev,
            'domain':    'curated',
            'note':      '',
        })
    return records


# ── PRISM evaluation (if requested) ───────────────────────────────────────────
def evaluate_prism_on_benchmark(benchmark_records):
    """
    Compare PRISM v17f* predictions against benchmark.
    For each (iso_a, iso_b, go_term) pair:
      score_a = PRISM prediction for iso_a on go_term
      score_b = PRISM prediction for iso_b on go_term
      Expected: direction='A_only' → score_a > score_b
    """
    from sklearn.metrics import roc_auc_score

    # Load PRISM predictions (if available)
    prism_path = '../../reports/exp_g_uniprot/prism_isoform_scores.npy'
    if not os.path.exists(prism_path):
        print("  [INFO] PRISM scores not found at", prism_path)
        print("  To generate: run v17f* model on UniProt isoform sequences")
        print("  Skipping PRISM evaluation.")
        return None

    print("  Loading PRISM predictions...")
    prism_data = np.load(prism_path, allow_pickle=True).item()
    # Expected: {'iso_id': {'go_term': score, ...}, ...}

    n_eval = 0
    correct = 0
    scores  = []
    labels  = []

    for rec in benchmark_records:
        iso_a = rec['iso_a']
        iso_b = rec['iso_b']
        go    = rec['go_term']

        if iso_a not in prism_data or iso_b not in prism_data:
            continue
        sa = prism_data[iso_a].get(go, None)
        sb = prism_data[iso_b].get(go, None)
        if sa is None or sb is None:
            continue

        n_eval += 1
        if rec['direction'] == 'A_only':
            correct  += int(sa > sb)
            scores.append(sa - sb)
            labels.append(1)
        elif rec['direction'] == 'B_only':
            correct  += int(sb > sa)
            scores.append(sb - sa)
            labels.append(1)

    if n_eval == 0:
        print("  No evaluable pairs found.")
        return None

    acc = correct / n_eval
    try:
        auroc = roc_auc_score(labels, scores)
    except Exception:
        auroc = float('nan')

    print(f"\n  PRISM UniProt Benchmark Evaluation:")
    print(f"  Evaluable pairs: {n_eval}")
    print(f"  Direction accuracy: {acc:.3f} ({correct}/{n_eval})")
    print(f"  AUROC: {auroc:.3f}")
    return {'n_eval': n_eval, 'accuracy': acc, 'auroc': auroc}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 70)
    print("  UniProt Isoform Benchmark Builder")
    print(f"  Evidence filter: {args.evidence}  Max proteins: {args.max_proteins}")
    print("=" * 70)

    all_records = []

    # ── 1. Curated high-confidence pairs ──────────────────────────────────────
    print("\n[1] Loading curated benchmark pairs...")
    curated = build_curated_benchmark()
    print(f"  {len(curated)} curated pairs from literature")
    all_records.extend(curated)

    # ── 2. pfam2go-based inference from UniProt variant sequences ─────────────
    if args.evidence in ('all', 'pfam2go'):
        print("\n[2] Loading pfam2go mapping...")
        pfam2go_map = load_pfam2go(PFAM2GO_PATH)

        print("\n[3] Fetching multi-isoform UniProt entries...")
        entries = fetch_human_multiisoform_entries(args.max_proteins)
        print(f"  {len(entries)} proteins with ≥2 isoforms")

        print("\n[4] Deriving GO differences per protein...")
        n_proc = 0
        for entry_meta in entries:
            acc  = entry_meta['accession']
            data = fetch_entry_detail(acc)
            recs = build_isoform_go_differences(data, pfam2go_map)
            all_records.extend(recs)
            n_proc += 1
            if n_proc % 100 == 0:
                print(f"  Processed {n_proc}/{len(entries)}  "
                      f"total records: {len(all_records)}", flush=True)

    # ── 3. Save benchmark ──────────────────────────────────────────────────────
    print(f"\n[5] Writing benchmark...")
    out_csv = f'{OUT_DIR}/uniprot_isoform_benchmark.csv'
    fieldnames = ['gene', 'accession', 'iso_a', 'iso_b', 'go_term',
                  'direction', 'evidence', 'domain', 'note']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_records:
            writer.writerow({k: rec.get(k, '') for k in fieldnames})

    # ── 4. Stats ──────────────────────────────────────────────────────────────
    stats_path = f'{OUT_DIR}/benchmark_stats.txt'
    ev_counts  = defaultdict(int)
    dir_counts = defaultdict(int)
    go_counts  = defaultdict(int)
    gene_set   = set()
    for r in all_records:
        ev_counts[r['evidence']]  += 1
        dir_counts[r['direction']] += 1
        go_counts[r['go_term']]    += 1
        gene_set.add(r['gene'])

    with open(stats_path, 'w') as f:
        f.write("=== UniProt Isoform Benchmark Stats ===\n\n")
        f.write(f"Total records:   {len(all_records)}\n")
        f.write(f"Unique genes:    {len(gene_set)}\n")
        f.write(f"Unique GO terms: {len(go_counts)}\n\n")
        f.write("By evidence:\n")
        for ev, cnt in sorted(ev_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {ev:<40} {cnt}\n")
        f.write("\nBy direction:\n")
        for d, cnt in sorted(dir_counts.items()):
            f.write(f"  {d:<20} {cnt}\n")
        f.write("\nTop GO terms:\n")
        for go, cnt in sorted(go_counts.items(), key=lambda x: -x[1])[:20]:
            f.write(f"  {go:<20} {cnt}\n")

    print(f"  Total records:   {len(all_records)}")
    print(f"  Unique genes:    {len(gene_set)}")
    print(f"  Unique GO terms: {len(go_counts)}")
    print(f"  Saved: {out_csv}")
    print(f"  Saved: {stats_path}")

    # ── 5. PRISM evaluation (optional) ────────────────────────────────────────
    if args.eval_prism:
        print("\n[6] Evaluating PRISM predictions...")
        result = evaluate_prism_on_benchmark(all_records)
        if result:
            eval_path = f'{OUT_DIR}/prism_eval.tsv'
            with open(eval_path, 'w') as f:
                f.write('n_eval\taccuracy\tauroc\n')
                f.write(f"{result['n_eval']}\t{result['accuracy']:.4f}\t{result['auroc']:.4f}\n")
            print(f"  Saved: {eval_path}")

    print("\n[DONE]")


if __name__ == '__main__':
    main()
