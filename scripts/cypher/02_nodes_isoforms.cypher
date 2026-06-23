// Isoform 노드 + Gene-HAS_ISOFORM
// 생성: bisect_to_neo4j.py

// ─── Isoform 노드 ────────────────────────────────────────────
// 출처: braak_correlation_results.tsv (ct_isoform 컬럼)
// transcript_id: ENST... 또는 novel ID (IsoQuant 출력)

MERGE (i:Isoform {transcript_id: 'transcript293004.chr1.nic'}) ON CREATE SET i.gene_symbol = 'KIF21B', i.source = 'novel_IsoQuant', i.description = 'Novel CT isoform (motor domain) vs KIF21B-203 (AD WD40)';
MERGE (g:Gene {symbol: 'KIF21B'}) MERGE (i:Isoform {transcript_id: 'transcript293004.chr1.nic'}) MERGE (g)-[:HAS_ISOFORM]->(i);

MERGE (i:Isoform {transcript_id: 'NDUFS4-201'}) ON CREATE SET i.gene_symbol = 'NDUFS4', i.source = 'novel_IsoQuant', i.description = 'NDUFS4-201 (canonical MTS) vs novel AD isoform';
MERGE (g:Gene {symbol: 'NDUFS4'}) MERGE (i:Isoform {transcript_id: 'NDUFS4-201'}) MERGE (g)-[:HAS_ISOFORM]->(i);

MERGE (i:Isoform {transcript_id: 'transcript100761.chr11.nic'}) ON CREATE SET i.gene_symbol = 'NDUFS8', i.source = 'novel_IsoQuant', i.description = 'Novel CT isoform vs NDUFS8-201 (AD-enriched)';
MERGE (g:Gene {symbol: 'NDUFS8'}) MERGE (i:Isoform {transcript_id: 'transcript100761.chr11.nic'}) MERGE (g)-[:HAS_ISOFORM]->(i);

