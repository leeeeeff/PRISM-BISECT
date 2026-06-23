// Isoform → CORRELATES_WITH_BRAAK → Disease
// 생성: bisect_to_neo4j.py

// ─── CORRELATES_WITH_BRAAK 관계 (Isoform → Disease) ─────────
// 속성: spearman_r, p_value, p_bonferroni, ci_95_lo, ci_95_hi, n_donors

MATCH (i:Isoform {transcript_id: 'transcript293004.chr1.nic'}) MATCH (d:Disease {mondo_id: 'MONDO:0004975'}) MERGE (i)-[r:CORRELATES_WITH_BRAAK]->(d) ON CREATE SET r.spearman_r = -0.527, r.p_value = 0.117495, r.p_bonferroni = 0.352484, r.n_donors = 10, r.braak_scale = 'B_0to3', r.method = 'Spearman_bootstrap_n1000', r.description = 'Novel CT isoform (motor domain) vs KIF21B-203 (AD WD40)';

MATCH (i:Isoform {transcript_id: 'NDUFS4-201'}) MATCH (d:Disease {mondo_id: 'MONDO:0004975'}) MERGE (i)-[r:CORRELATES_WITH_BRAAK]->(d) ON CREATE SET r.spearman_r = -0.3075, r.p_value = 0.306732, r.p_bonferroni = 0.920196, r.n_donors = 13, r.braak_scale = 'B_0to3', r.method = 'Spearman_bootstrap_n1000', r.description = 'NDUFS4-201 (canonical MTS) vs novel AD isoform', r.ci_95_lo = -0.7682, r.ci_95_hi = 0.2644;

MATCH (i:Isoform {transcript_id: 'transcript100761.chr11.nic'}) MATCH (d:Disease {mondo_id: 'MONDO:0004975'}) MERGE (i)-[r:CORRELATES_WITH_BRAAK]->(d) ON CREATE SET r.spearman_r = -0.3747, r.p_value = 0.114003, r.p_bonferroni = 0.342008, r.n_donors = 19, r.braak_scale = 'B_0to3', r.method = 'Spearman_bootstrap_n1000', r.description = 'Novel CT isoform vs NDUFS8-201 (AD-enriched)', r.ci_95_lo = -0.7199, r.ci_95_hi = 0.0894;

