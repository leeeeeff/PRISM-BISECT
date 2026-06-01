"""
nextprot_coverage_check.py — NeXtProt isoform-level annotation 커버리지 확인
=============================================================================
목적: 우리 13 GO term의 단백질들에 대해 NeXtProt가
      isoform-specific 기능 annotation을 얼마나 보유하는지 확인

결과 해석:
  > 30%  → NeXtProt를 대안 label source로 사용 가능 검토
  < 10%  → MIL이 유일한 현실적 해결책 확정
"""

import os, sys, json, time
from collections import defaultdict
import urllib.request, urllib.parse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ANNOT_DIR = '../data/raw_data/data/annotations'
ID_DIR    = '../data/raw_data/data/id_lists'

GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0030017': 'Sarcomere org',
    'GO:0006941': 'Muscle contraction',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0003774': 'Motor activity',
    'GO:0006096': 'Glycolysis',
}

# ─── 1. 우리 13 GO term 관련 유전자 수집 ──────────────────────────────────────
print("=" * 65)
print(" NeXtProt Isoform Annotation Coverage Check")
print("=" * 65)

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]

# test set gene symbols
import numpy as np
def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

te_geneid = load_ids('my_gene_list_fixed.npy')
te_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_geneid]

go_genes = defaultdict(set)
all_target_genes = set()
with open(f'{ANNOT_DIR}/human_annotations.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        sym = parts[0]
        for go_id in GO_TERMS:
            if go_id in parts[1:]:
                go_genes[go_id].add(sym)
                all_target_genes.add(sym)

print(f"\n[우리 데이터]")
print(f"  13 GO term 관련 유전자: {len(all_target_genes)}개")
for go_id, name in GO_TERMS.items():
    print(f"  {name:<22} {len(go_genes[go_id]):>3}개 유전자")

# ─── 2. NeXtProt REST API로 isoform-specific annotation 확인 ─────────────────
print(f"\n[NeXtProt API 조회] — 샘플 {min(30, len(all_target_genes))}개 유전자 대상")
print("  isoform-specific functional annotation 보유 여부 확인\n")

NEXTPROT_BASE = "https://api.nextprot.org"
SAMPLE_GENES  = sorted(all_target_genes)[:30]  # 알파벳순 30개 샘플

def query_nextprot_entry(gene_symbol, timeout=10):
    """
    NeXtProt API:
      1) gene symbol → NX_ accession 검색
      2) isoform-specific annotation count 조회
    Returns dict with keys: found, nx_id, n_isoform_annots, categories
    """
    result = {'gene': gene_symbol, 'found': False, 'nx_id': None,
              'n_isoform_annots': 0, 'has_isoform_function': False,
              'categories': [], 'error': None}
    try:
        # Step 1: gene name → NX_ ID
        search_url = (f"{NEXTPROT_BASE}/entry/search?"
                      f"query={urllib.parse.quote(gene_symbol)}"
                      f"&filter=human&rows=5")
        req = urllib.request.Request(search_url,
                                     headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())

        entries = data.get('entry', [])
        if not entries:
            result['error'] = 'not_found'
            return result

        # gene symbol 정확히 매칭되는 entry 우선
        nx_id = None
        for e in entries:
            if isinstance(e, dict):
                gene_names = e.get('geneName', [])
                if isinstance(gene_names, list):
                    if gene_symbol.upper() in [g.upper() for g in gene_names]:
                        nx_id = e.get('accession') or e.get('uniqueName')
                        break
                elif isinstance(gene_names, str):
                    if gene_names.upper() == gene_symbol.upper():
                        nx_id = e.get('accession') or e.get('uniqueName')
                        break
        if nx_id is None:
            # 첫 번째 결과 사용
            first = entries[0] if isinstance(entries[0], dict) else {}
            nx_id = first.get('accession') or first.get('uniqueName')

        if not nx_id:
            result['error'] = 'no_accession'
            return result

        result['found'] = True
        result['nx_id'] = nx_id

        # Step 2: isoform-specific annotation 수 조회
        annot_url = f"{NEXTPROT_BASE}/entry/{nx_id}/isoform"
        req2 = urllib.request.Request(annot_url,
                                      headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req2, timeout=timeout) as resp2:
            annot_data = json.loads(resp2.read().decode())

        # isoform-specific annotations 카운트
        isoforms = annot_data.get('isoformList', []) or []
        n_isoform_specific = 0
        seen_categories = set()
        for iso in isoforms:
            annotations = iso.get('annotations', []) or []
            for ann in annotations:
                cat = ann.get('category', '')
                if cat:
                    seen_categories.add(cat)
                    n_isoform_specific += 1

        result['n_isoform_annots'] = n_isoform_specific
        result['categories'] = list(seen_categories)
        result['has_isoform_function'] = n_isoform_specific > 0

    except urllib.error.URLError as e:
        result['error'] = f'url_error: {e}'
    except json.JSONDecodeError as e:
        result['error'] = f'json_error: {e}'
    except Exception as e:
        result['error'] = f'error: {e}'

    return result

# ─── 3. SPARQL로 isoform-specific GO annotation 직접 조회 ─────────────────────
def query_nextprot_sparql_sample(gene_symbols, timeout=30):
    """
    NeXtProt SPARQL endpoint로 isoform-specific functional annotation 조회.
    gene_symbols: 최대 20개
    """
    gene_list_str = " ".join(f'"{g}"' for g in gene_symbols[:20])

    sparql_query = f"""
    PREFIX : <http://nextprot.org/rdf#>
    PREFIX cv: <http://nextprot.org/rdf/terminology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?geneName ?isoformAcc ?category ?term
    WHERE {{
      ?entry :gene ?gene .
      ?gene rdfs:label ?geneName .
      FILTER(?geneName IN ({gene_list_str}))
      ?entry :isoform ?isoform .
      ?isoform :swissprotDisplayed ?isoformAcc .
      ?isoform :annotation ?annotation .
      ?annotation :in-class/:label ?category .
      OPTIONAL {{ ?annotation :term/:label ?term }}
      FILTER(?category IN (
        "go molecular function",
        "go biological process",
        "go cellular component"
      ))
    }}
    LIMIT 500
    """

    endpoint = "https://sparql.nextprot.org"
    data = urllib.parse.urlencode({
        'query': sparql_query,
        'format': 'json'
    }).encode()

    try:
        req = urllib.request.Request(endpoint, data=data,
                                     headers={'Accept': 'application/sparql-results+json'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {'error': str(e)}

# ─── 4. 실행 ──────────────────────────────────────────────────────────────────
print("[방법 A: SPARQL로 isoform-specific GO annotation 직접 조회]")
print(f"  대상: {list(SAMPLE_GENES[:10])} ...")

sparql_result = query_nextprot_sparql_sample(list(SAMPLE_GENES[:20]))

if 'error' in sparql_result:
    print(f"  SPARQL 실패: {sparql_result['error']}")
    print("  → REST API 방식으로 전환\n")
    use_sparql = False
else:
    bindings = sparql_result.get('results', {}).get('bindings', [])
    print(f"  결과 행수: {len(bindings)}")
    if bindings:
        genes_with_iso_go = set(b['geneName']['value'] for b in bindings if 'geneName' in b)
        categories_found  = set(b['category']['value'] for b in bindings if 'category' in b)
        print(f"  isoform-level GO annotation 보유 유전자: {len(genes_with_iso_go)}개")
        print(f"  카테고리: {categories_found}")
        print(f"  예시 결과:")
        for b in bindings[:5]:
            print(f"    {b.get('geneName',{}).get('value','')} "
                  f"/ {b.get('isoformAcc',{}).get('value','')} "
                  f"/ {b.get('category',{}).get('value','')} "
                  f"/ {b.get('term',{}).get('value','')}")
    use_sparql = True

# ─── 5. REST API 방식 (SPARQL 실패 시) ────────────────────────────────────────
if not use_sparql:
    print("[방법 B: REST API per-protein 조회]")
    results = []
    found_count = 0
    isoform_annot_count = 0

    for i, gene in enumerate(SAMPLE_GENES[:20]):
        r = query_nextprot_entry(gene)
        results.append(r)
        status = ''
        if r['error']:
            status = f"ERROR({r['error'][:30]})"
        elif r['found']:
            found_count += 1
            if r['has_isoform_function']:
                isoform_annot_count += 1
                status = f"✓ NX={r['nx_id']} iso_annots={r['n_isoform_annots']} {r['categories'][:2]}"
            else:
                status = f"  NX={r['nx_id']} (no isoform-specific annots)"
        else:
            status = "not found"
        print(f"  {gene:<15} {status}")
        time.sleep(0.3)  # rate limit

    coverage = isoform_annot_count / max(found_count, 1)
    print(f"\n  Found: {found_count}/20, isoform-specific: {isoform_annot_count}")
    print(f"  Coverage (among found): {coverage:.1%}")

# ─── 6. 전체 요약 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("[NeXtProt Coverage 판정]")

if use_sparql and 'results' in sparql_result:
    bindings = sparql_result.get('results', {}).get('bindings', [])
    genes_with = set(b.get('geneName', {}).get('value', '') for b in bindings)
    genes_with.discard('')
    n_queried = len(SAMPLE_GENES[:20])
    coverage  = len(genes_with) / n_queried if n_queried > 0 else 0

    print(f"  조회 유전자: {n_queried}개 (샘플)")
    print(f"  isoform-level GO annotation 보유: {len(genes_with)}개 ({coverage:.1%})")
    if coverage > 0.30:
        print("  → GOOD: 대안 label source 검토 가능")
        print("    NeXtProt isoform-level annotations을 canonical auxiliary 대신")
        print("    직접 GO label로 사용하는 방향 고려")
    elif coverage > 0.10:
        print("  → PARTIAL: 일부 term에서 보완 가능, 전체 커버리지 부족")
    else:
        print("  → LOW: isoform-level GO annotation 부족")
        print("    → MIL (gene-level label의 label space 문제 해결)이 현실적 방향")
elif not use_sparql and 'results' in dir():
    pass  # already printed above
else:
    print("  → API 접근 실패. 수동 확인 필요.")
    print("  참고: https://www.nextprot.org/proteins/search?query=TNNT2&filter=human")

print("=" * 65)
