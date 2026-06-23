// CellCluster → LOCATED_IN → CorticalLayer
// 생성: bisect_to_neo4j.py

// ─── LOCATED_IN 관계 (CellCluster → CorticalLayer) ──────────
MATCH (c:CellCluster {leiden_id: 1}), (l:CorticalLayer {label: 'L2/3'}) MERGE (c)-[r:LOCATED_IN]->(l) ON CREATE SET r.layer_score = 1.0933, r.ambiguity = 1.4646, r.method = 'Allen_Human_Brain_Atlas_markers';

MATCH (c:CellCluster {leiden_id: 2}), (l:CorticalLayer {label: 'L2/3'}) MERGE (c)-[r:LOCATED_IN]->(l) ON CREATE SET r.layer_score = 1.4777, r.ambiguity = 1.6502, r.method = 'Allen_Human_Brain_Atlas_markers';

MATCH (c:CellCluster {leiden_id: 27}), (l:CorticalLayer {label: 'L2/3'}) MERGE (c)-[r:LOCATED_IN]->(l) ON CREATE SET r.layer_score = 1.4158, r.ambiguity = 1.5641, r.method = 'Allen_Human_Brain_Atlas_markers';

MATCH (c:CellCluster {leiden_id: 10}), (l:CorticalLayer {label: 'L4'}) MERGE (c)-[r:LOCATED_IN]->(l) ON CREATE SET r.layer_score = 1.2349, r.ambiguity = 0.3566, r.method = 'Allen_Human_Brain_Atlas_markers';

MATCH (c:CellCluster {leiden_id: 11}), (l:CorticalLayer {label: 'L4'}) MERGE (c)-[r:LOCATED_IN]->(l) ON CREATE SET r.layer_score = 1.2248, r.ambiguity = 0.6824, r.method = 'Allen_Human_Brain_Atlas_markers';

MATCH (c:CellCluster {leiden_id: 18}), (l:CorticalLayer {label: 'L4'}) MERGE (c)-[r:LOCATED_IN]->(l) ON CREATE SET r.layer_score = 0.9456, r.ambiguity = 0.5918, r.method = 'Allen_Human_Brain_Atlas_markers';

MATCH (c:CellCluster {leiden_id: 19}), (l:CorticalLayer {label: 'L5'}) MERGE (c)-[r:LOCATED_IN]->(l) ON CREATE SET r.layer_score = 1.0527, r.ambiguity = 0.2565, r.method = 'Allen_Human_Brain_Atlas_markers';

MATCH (c:CellCluster {leiden_id: 26}), (l:CorticalLayer {label: 'L5'}) MERGE (c)-[r:LOCATED_IN]->(l) ON CREATE SET r.layer_score = 2.4321, r.ambiguity = 2.0792, r.method = 'Allen_Human_Brain_Atlas_markers';

MATCH (c:CellCluster {leiden_id: 25}), (l:CorticalLayer {label: 'L6'}) MERGE (c)-[r:LOCATED_IN]->(l) ON CREATE SET r.layer_score = 1.3323, r.ambiguity = 0.376, r.method = 'Allen_Human_Brain_Atlas_markers';

MATCH (c:CellCluster {leiden_id: 20}), (l:CorticalLayer {label: 'L6'}) MERGE (c)-[r:LOCATED_IN]->(l) ON CREATE SET r.layer_score = -0.3444, r.ambiguity = 0.1611, r.method = 'Allen_Human_Brain_Atlas_markers';

