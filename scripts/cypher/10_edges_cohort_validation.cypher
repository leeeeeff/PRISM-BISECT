// CellCluster → VALIDATED_IN → Cohort
// 생성: bisect_to_neo4j.py

// ─── VALIDATED_IN 관계 (CellCluster → Cohort) ───────────────
// 코호트별 독립 검증: PO(3'v4) + SMC(3'v3) 방향 일치 = 배치효과 아님

MATCH (c:CellCluster {leiden_id: 18}), (h:Cohort {name: 'PO'}) MERGE (c)-[r:VALIDATED_IN]->(h) ON CREATE SET r.delta_pct = 0.971, r.p_value = 0.1905, r.sig = 'ns', r.n_AD = 5, r.n_CT = 4, r.direction = '↑AD';

MATCH (c:CellCluster {leiden_id: 18}), (h:Cohort {name: 'SMC'}) MERGE (c)-[r:VALIDATED_IN]->(h) ON CREATE SET r.delta_pct = 1.343, r.p_value = 0.3677, r.sig = 'ns', r.n_AD = 8, r.n_CT = 4, r.direction = '↑AD';

MATCH (c:CellCluster {leiden_id: 19}), (h:Cohort {name: 'PO'}) MERGE (c)-[r:VALIDATED_IN]->(h) ON CREATE SET r.delta_pct = -1.86, r.p_value = 0.2857, r.sig = 'ns', r.n_AD = 5, r.n_CT = 4, r.direction = '↓AD';

MATCH (c:CellCluster {leiden_id: 19}), (h:Cohort {name: 'SMC'}) MERGE (c)-[r:VALIDATED_IN]->(h) ON CREATE SET r.delta_pct = -1.118, r.p_value = 0.1091, r.sig = 'ns', r.n_AD = 8, r.n_CT = 4, r.direction = '↓AD';

MATCH (c:CellCluster {leiden_id: 9}), (h:Cohort {name: 'PO'}) MERGE (c)-[r:VALIDATED_IN]->(h) ON CREATE SET r.delta_pct = -0.364, r.p_value = 0.7302, r.sig = 'ns', r.n_AD = 5, r.n_CT = 4, r.direction = '↓AD';

MATCH (c:CellCluster {leiden_id: 9}), (h:Cohort {name: 'SMC'}) MERGE (c)-[r:VALIDATED_IN]->(h) ON CREATE SET r.delta_pct = -2.456, r.p_value = 0.1535, r.sig = 'ns', r.n_AD = 8, r.n_CT = 4, r.direction = '↓AD';

MATCH (c:CellCluster {leiden_id: 15}), (h:Cohort {name: 'PO'}) MERGE (c)-[r:VALIDATED_IN]->(h) ON CREATE SET r.delta_pct = 0.739, r.p_value = 0.1111, r.sig = 'ns', r.n_AD = 5, r.n_CT = 4, r.direction = '↑AD';

MATCH (c:CellCluster {leiden_id: 15}), (h:Cohort {name: 'SMC'}) MERGE (c)-[r:VALIDATED_IN]->(h) ON CREATE SET r.delta_pct = 1.142, r.p_value = 0.2141, r.sig = 'ns', r.n_AD = 8, r.n_CT = 4, r.direction = '↑AD';

MATCH (c:CellCluster {leiden_id: 11}), (h:Cohort {name: 'PO'}) MERGE (c)-[r:VALIDATED_IN]->(h) ON CREATE SET r.delta_pct = 0.504, r.p_value = 0.9048, r.sig = 'ns', r.n_AD = 5, r.n_CT = 4, r.direction = '↑AD';

MATCH (c:CellCluster {leiden_id: 11}), (h:Cohort {name: 'SMC'}) MERGE (c)-[r:VALIDATED_IN]->(h) ON CREATE SET r.delta_pct = 1.512, r.p_value = 0.4606, r.sig = 'ns', r.n_AD = 8, r.n_CT = 4, r.direction = '↑AD';

