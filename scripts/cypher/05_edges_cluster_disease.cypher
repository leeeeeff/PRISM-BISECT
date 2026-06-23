// CellCluster → AD_ENRICHED/AD_DEPLETED → Disease
// 생성: bisect_to_neo4j.py

// ─── AD_ENRICHED / AD_DEPLETED 관계 ─────────────────────────
// 조건: sig in [***, **, *, †]
// 속성: delta(%), p_value, sig, AD_pct, CT_pct, n_cells, cohort

MATCH (c:CellCluster {leiden_id: 11}), (d:Disease {mondo_id: 'MONDO:0004975'}) MERGE (c)-[r:AD_ENRICHED]->(d) ON CREATE SET r.delta_pct = 1.12, r.p_value = 0.089, r.sig = '†', r.AD_pct = 4.08, r.CT_pct = 2.96, r.n_cells = 4290, r.cohort = 'PO+SMC', r.n_donors = 21, r.tissue = 'UBERON:0001851', r.species = 'NCBITaxon:9606';

MATCH (c:CellCluster {leiden_id: 18}), (d:Disease {mondo_id: 'MONDO:0004975'}) MERGE (c)-[r:AD_ENRICHED]->(d) ON CREATE SET r.delta_pct = 1.46, r.p_value = 0.0099, r.sig = '**', r.AD_pct = 2.27, r.CT_pct = 0.81, r.n_cells = 1603, r.cohort = 'PO+SMC', r.n_donors = 21, r.tissue = 'UBERON:0001851', r.species = 'NCBITaxon:9606';

MATCH (c:CellCluster {leiden_id: 19}), (d:Disease {mondo_id: 'MONDO:0004975'}) MERGE (c)-[r:AD_DEPLETED]->(d) ON CREATE SET r.delta_pct = -1.6, r.p_value = 0.0368, r.sig = '*', r.AD_pct = 0.77, r.CT_pct = 2.37, r.n_cells = 1410, r.cohort = 'PO+SMC', r.n_donors = 21, r.tissue = 'UBERON:0001851', r.species = 'NCBITaxon:9606';

MATCH (c:CellCluster {leiden_id: 9}), (d:Disease {mondo_id: 'MONDO:0004975'}) MERGE (c)-[r:AD_DEPLETED]->(d) ON CREATE SET r.delta_pct = -1.28, r.p_value = 0.03, r.sig = '*', r.AD_pct = 1.98, r.CT_pct = 3.27, r.n_cells = 2864, r.cohort = 'PO+SMC', r.n_donors = 21, r.tissue = 'UBERON:0001851', r.species = 'NCBITaxon:9606';

MATCH (c:CellCluster {leiden_id: 15}), (d:Disease {mondo_id: 'MONDO:0004975'}) MERGE (c)-[r:AD_ENRICHED]->(d) ON CREATE SET r.delta_pct = 1.04, r.p_value = 0.034, r.sig = '*', r.AD_pct = 2.33, r.CT_pct = 1.29, r.n_cells = 2097, r.cohort = 'PO+SMC', r.n_donors = 21, r.tissue = 'UBERON:0001851', r.species = 'NCBITaxon:9606';

