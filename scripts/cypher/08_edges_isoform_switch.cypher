// CellType → SHOWS_ISOFORM_SWITCH → Gene
// 생성: bisect_to_neo4j.py

// ─── SHOWS_ISOFORM_SWITCH 관계 (CellType → Gene) ───────────
// 조건: sig in [***, **, *, †]
// 속성: AD_ratio, CT_ratio, delta, p_value, sig, n_AD, n_CT

MATCH (t:CellType {name: 'Excitatory_neuron'}) MATCH (g:Gene {symbol: 'NDUFS7'}) MERGE (t)-[r:SHOWS_ISOFORM_SWITCH]->(g) ON CREATE SET r.AD_ratio = 0.9486, r.CT_ratio = 1.0, r.delta = -0.0514, r.p_value = 0.045071, r.sig = '*', r.n_AD_donors = 9, r.n_CT_donors = 8, r.disease = 'MONDO:0004975', r.tissue = 'UBERON:0001851', r.method = 'DIU_permutation_n10000';

MATCH (t:CellType {name: 'Inhibitory_neuron'}) MATCH (g:Gene {symbol: 'NDUFS4'}) MERGE (t)-[r:SHOWS_ISOFORM_SWITCH]->(g) ON CREATE SET r.AD_ratio = 0.0833, r.CT_ratio = 0.4037, r.delta = -0.3204, r.p_value = 0.098729, r.sig = '†', r.n_AD_donors = 4, r.n_CT_donors = 3, r.disease = 'MONDO:0004975', r.tissue = 'UBERON:0001851', r.method = 'DIU_permutation_n10000';

MATCH (t:CellType {name: 'Inhibitory_neuron'}) MATCH (g:Gene {symbol: 'NDUFS8'}) MERGE (t)-[r:SHOWS_ISOFORM_SWITCH]->(g) ON CREATE SET r.AD_ratio = 0.62, r.CT_ratio = 0.9895, r.delta = -0.3694, r.p_value = 0.056536, r.sig = '†', r.n_AD_donors = 7, r.n_CT_donors = 5, r.disease = 'MONDO:0004975', r.tissue = 'UBERON:0001851', r.method = 'DIU_permutation_n10000';

