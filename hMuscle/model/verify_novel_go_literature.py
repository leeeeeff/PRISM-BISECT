#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_novel_go_literature.py
==============================
Case A(완전 무주석 gene)/Case B(타 term 획득) 후보 유전자들에 대해
UniProt REST API에서 실제 GO Biological Process annotation + evidence code를 조회,
target GO term과의 직접/간접(관련어) 일치 여부를 자동 스크리닝한다.

직접 실험 evidence code(IDA/IMP/IPI/IGI/IEP)가 target term 또는 그 밀접 관련어에
붙어 있으면 'experimental_support'로 표시. 없으면 'no_direct_evidence'.
"""
import json, time, urllib.request, urllib.parse, sys

CANDIDATES_TSV = sys.argv[1] if len(sys.argv) > 1 else None

EXPERIMENTAL_CODES = {'IDA', 'IMP', 'IPI', 'IGI', 'IEP'}

def fetch_go_bp(gene):
    q = f'gene:{gene} AND organism_id:9606 AND reviewed:true'
    url = 'https://rest.uniprot.org/uniprotkb/search?' + urllib.parse.urlencode({
        'query': q, 'format': 'json', 'fields': 'accession,gene_names,go_p'
    })
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.load(resp)
    except Exception as e:
        return None, f'ERROR:{e}'
    if not data.get('results'):
        return None, 'NOT_FOUND'
    entry = data['results'][0]
    acc = entry.get('primaryAccession')
    bp_terms = []
    for xref in entry.get('uniProtKBCrossReferences', []):
        if xref.get('database') != 'GO':
            continue
        props = {p['key']: p['value'] for p in xref.get('properties', [])}
        term = props.get('GoTerm', '')
        if not term.startswith('P:'):
            continue
        ev = props.get('GoEvidenceType', '')
        code = ev.split(':')[0] if ev else ''
        bp_terms.append((xref['id'], term[2:], code, ev))
    return acc, bp_terms

def main():
    rows = []
    if CANDIDATES_TSV:
        with open(CANDIDATES_TSV) as f:
            for line in f:
                p = line.rstrip('\n').split('\t')
                rows.append(p)  # gene, isoform, go_id, go_name, score[, own, gap]

    seen_genes = {}
    out = []
    for r in rows:
        gene = r[0]
        if gene.startswith('ENSG'):
            continue  # skip unnamed/ambiguous gene IDs
        target_go = r[2]
        target_name = r[3]
        score = r[4]
        if gene not in seen_genes:
            acc, bp = fetch_go_bp(gene)
            seen_genes[gene] = (acc, bp)
            time.sleep(0.15)
        acc, bp = seen_genes[gene]
        if isinstance(bp, str):
            out.append({'gene': gene, 'target_go': target_go, 'target_name': target_name,
                        'score': score, 'status': bp, 'acc': acc})
            continue
        exact = [t for t in bp if t[0] == target_go]
        exp_any = [t for t in bp if t[2] in EXPERIMENTAL_CODES]
        out.append({
            'gene': gene, 'acc': acc, 'target_go': target_go, 'target_name': target_name,
            'score': score,
            'exact_match': bool(exact),
            'exact_match_evidence': exact,
            'n_bp_terms': len(bp),
            'experimental_terms': [(t[0], t[1], t[2]) for t in exp_any],
        })
        print(f"{gene:12s} target={target_go} {target_name:28s} score={score}  "
              f"exact={'YES' if exact else 'no':4s}  n_BP={len(bp):3d}  "
              f"exp_terms={len(exp_any)}")
    with open(CANDIDATES_TSV + '.verified.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[Saved] {CANDIDATES_TSV}.verified.json")

if __name__ == '__main__':
    main()
