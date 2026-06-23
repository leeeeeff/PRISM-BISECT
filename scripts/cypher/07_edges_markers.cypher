// CellCluster → DEFINED_BY_MARKER → Gene
// 생성: bisect_to_neo4j.py

// ─── DEFINED_BY_MARKER 관계 (CellCluster → Gene) ────────────
MATCH (c:CellCluster {leiden_id: 1}), (g:Gene {symbol: 'CUX1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.91, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 1}), (g:Gene {symbol: 'SATB2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.55, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 2}), (g:Gene {symbol: 'CUX1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.06, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 2}), (g:Gene {symbol: 'RORB'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.05, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 27}), (g:Gene {symbol: 'CUX1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.05, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 27}), (g:Gene {symbol: 'SATB2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.13, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 10}), (g:Gene {symbol: 'RORB'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.45, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 10}), (g:Gene {symbol: 'ETV1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.02, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 11}), (g:Gene {symbol: 'RORB'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.15, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 11}), (g:Gene {symbol: 'ETV1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.3, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 18}), (g:Gene {symbol: 'RORB'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.15, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 18}), (g:Gene {symbol: 'PRSS12'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.32, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 18}), (g:Gene {symbol: 'ETV1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.02, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 18}), (g:Gene {symbol: 'RBFOX1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.94, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 19}), (g:Gene {symbol: 'FEZF2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.19, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 19}), (g:Gene {symbol: 'BCL11B'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.72, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 19}), (g:Gene {symbol: 'TLE4'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.7, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 26}), (g:Gene {symbol: 'FEZF2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.29, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 26}), (g:Gene {symbol: 'BCL11B'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.31, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 25}), (g:Gene {symbol: 'FOXP2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.05, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 25}), (g:Gene {symbol: 'SYT6'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.1, r.evidence = 'snRNA-seq_long-read';


MATCH (c:CellCluster {leiden_id: 7}), (g:Gene {symbol: 'PVALB'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.34, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 7}), (g:Gene {symbol: 'LHX6'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.51, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 7}), (g:Gene {symbol: 'GAD2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.59, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 21}), (g:Gene {symbol: 'PVALB'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.61, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 21}), (g:Gene {symbol: 'LHX6'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.68, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 9}), (g:Gene {symbol: 'SST'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.8, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 9}), (g:Gene {symbol: 'LHX6'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.58, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 9}), (g:Gene {symbol: 'GAD2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.0, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 8}), (g:Gene {symbol: 'VIP'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.21, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 8}), (g:Gene {symbol: 'ADARB2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 3.71, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 8}), (g:Gene {symbol: 'PROX1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.3, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 16}), (g:Gene {symbol: 'VIP'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.76, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 16}), (g:Gene {symbol: 'ADARB2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 4.08, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 16}), (g:Gene {symbol: 'RBFOX1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.85, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 13}), (g:Gene {symbol: 'LAMP5'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.21, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 13}), (g:Gene {symbol: 'NDNF'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.5, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 13}), (g:Gene {symbol: 'ADARB2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 4.03, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 15}), (g:Gene {symbol: 'LAMP5'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.58, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 15}), (g:Gene {symbol: 'KIT'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.08, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 15}), (g:Gene {symbol: 'ADARB2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 4.09, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 22}), (g:Gene {symbol: 'LAMP5'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.91, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 22}), (g:Gene {symbol: 'LHX6'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.5, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 22}), (g:Gene {symbol: 'ADARB2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 3.4, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 24}), (g:Gene {symbol: 'ADARB2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 4.19, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 24}), (g:Gene {symbol: 'SP8'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.39, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 24}), (g:Gene {symbol: 'NDNF'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.29, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 0}), (g:Gene {symbol: 'MBP'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 3.55, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 0}), (g:Gene {symbol: 'OPALIN'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.66, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 0}), (g:Gene {symbol: 'ENPP2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.08, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 3}), (g:Gene {symbol: 'MBP'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 3.43, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 3}), (g:Gene {symbol: 'ENPP2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.14, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 28}), (g:Gene {symbol: 'MBP'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 3.74, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 28}), (g:Gene {symbol: 'KLK6'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.71, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 28}), (g:Gene {symbol: 'RBFOX1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.51, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 32}), (g:Gene {symbol: 'MBP'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.82, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 32}), (g:Gene {symbol: 'KLK6'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.6, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 6}), (g:Gene {symbol: 'PDGFRA'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.85, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 6}), (g:Gene {symbol: 'CSPG4'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.64, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 6}), (g:Gene {symbol: 'OLIG2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.64, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 31}), (g:Gene {symbol: 'PDGFRA'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.3, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 31}), (g:Gene {symbol: 'OLIG1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.21, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 4}), (g:Gene {symbol: 'SLC1A2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 4.05, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 4}), (g:Gene {symbol: 'AQP4'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.14, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 4}), (g:Gene {symbol: 'GJA1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.65, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 4}), (g:Gene {symbol: 'C3'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.05, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 5}), (g:Gene {symbol: 'SPP1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.18, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 5}), (g:Gene {symbol: 'P2RY12'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.58, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 5}), (g:Gene {symbol: 'TMEM119'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.22, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 5}), (g:Gene {symbol: 'TREM2'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.25, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 17}), (g:Gene {symbol: 'PDGFRB'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.33, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 17}), (g:Gene {symbol: 'NOTCH3'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.7, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 17}), (g:Gene {symbol: 'RGS5'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.86, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 23}), (g:Gene {symbol: 'FLT1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 3.0, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 23}), (g:Gene {symbol: 'VWF'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 2.48, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 23}), (g:Gene {symbol: 'CLDN5'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 1.86, r.evidence = 'snRNA-seq_long-read';

MATCH (c:CellCluster {leiden_id: 29}), (g:Gene {symbol: 'NKG7'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.88, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 29}), (g:Gene {symbol: 'GNLY'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.66, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 29}), (g:Gene {symbol: 'CD3E'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.32, r.evidence = 'snRNA-seq_long-read';
MATCH (c:CellCluster {leiden_id: 29}), (g:Gene {symbol: 'MS4A1'}) MERGE (c)-[r:DEFINED_BY_MARKER]->(g) ON CREATE SET r.mean_expression = 0.33, r.evidence = 'snRNA-seq_long-read';

