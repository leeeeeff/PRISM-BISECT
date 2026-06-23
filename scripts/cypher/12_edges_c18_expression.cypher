// Gene → OVEREXPRESSED_IN/UNDEREXPRESSED_IN → CellCluster(C18)
// 생성: bisect_to_neo4j.py

// ─── OVEREXPRESSED_IN / UNDEREXPRESSED_IN (Gene → CellCluster C18) ─
// C18 (leiden_id=18) vs L4 IT 평균 (C10+C11) 비교
// 기준: |log2FC| > 1 (2배 이상 차이)

MATCH (g:Gene {symbol: 'ETV1'}) MATCH (c:CellCluster {leiden_id: 18}) MERGE (g)-[r:UNDEREXPRESSED_IN]->(c) ON CREATE SET r.log2FC = -2.706, r.C18_mean_expr = 0.0235, r.C10C11_mean_expr = 0.1591, r.comparison = 'C18_vs_L4IT_C10C11_mean', r.method = 'log2FC_snRNA-seq';

MATCH (g:Gene {symbol: 'BCL11A'}) MATCH (c:CellCluster {leiden_id: 18}) MERGE (g)-[r:UNDEREXPRESSED_IN]->(c) ON CREATE SET r.log2FC = -2.492, r.C18_mean_expr = 0.1844, r.C10C11_mean_expr = 1.0422, r.comparison = 'C18_vs_L4IT_C10C11_mean', r.method = 'log2FC_snRNA-seq';

MATCH (g:Gene {symbol: 'MBP'}) MATCH (c:CellCluster {leiden_id: 18}) MERGE (g)-[r:UNDEREXPRESSED_IN]->(c) ON CREATE SET r.log2FC = -1.872, r.C18_mean_expr = 0.1951, r.C10C11_mean_expr = 0.7167, r.comparison = 'C18_vs_L4IT_C10C11_mean', r.method = 'log2FC_snRNA-seq';

MATCH (g:Gene {symbol: 'BCL11B'}) MATCH (c:CellCluster {leiden_id: 18}) MERGE (g)-[r:UNDEREXPRESSED_IN]->(c) ON CREATE SET r.log2FC = -1.449, r.C18_mean_expr = 0.0078, r.C10C11_mean_expr = 0.0231, r.comparison = 'C18_vs_L4IT_C10C11_mean', r.method = 'log2FC_snRNA-seq';

MATCH (g:Gene {symbol: 'PRSS12'}) MATCH (c:CellCluster {leiden_id: 18}) MERGE (g)-[r:OVEREXPRESSED_IN]->(c) ON CREATE SET r.log2FC = 2.018, r.C18_mean_expr = 0.3184, r.C10C11_mean_expr = 0.0779, r.comparison = 'C18_vs_L4IT_C10C11_mean', r.method = 'log2FC_snRNA-seq';

