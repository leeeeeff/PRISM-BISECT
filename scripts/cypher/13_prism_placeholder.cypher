// GOTerm 노드 + PREDICTS_GO placeholder
// 생성: bisect_to_neo4j.py

// ─── PREDICTS_GO 관계 placeholder (PRISM 출력 연결용) ────────
// 현재 PRISM AUPRC 결과를 Isoform-GOTerm 엣지로 변환하는 구조
// 실제 적재: generate_brain_full_extended_scores.py 출력 연동 필요

// 예시 GOTerm 노드 (18 BP terms):
MERGE (g:GOTerm {go_id: 'GO:0022900'}) ON CREATE SET g.name = 'electron transport chain', g.namespace = 'biological_process';
MERGE (g:GOTerm {go_id: 'GO:0006412'}) ON CREATE SET g.name = 'translation', g.namespace = 'biological_process';
MERGE (g:GOTerm {go_id: 'GO:0006936'}) ON CREATE SET g.name = 'muscle contraction', g.namespace = 'biological_process';
MERGE (g:GOTerm {go_id: 'GO:0006096'}) ON CREATE SET g.name = 'glycolysis', g.namespace = 'biological_process';
MERGE (g:GOTerm {go_id: 'GO:0007018'}) ON CREATE SET g.name = 'microtubule-based movement', g.namespace = 'biological_process';
MERGE (g:GOTerm {go_id: 'GO:0006814'}) ON CREATE SET g.name = 'sodium ion transport', g.namespace = 'biological_process';
MERGE (g:GOTerm {go_id: 'GO:0007010'}) ON CREATE SET g.name = 'cytoskeleton organization', g.namespace = 'biological_process';
MERGE (g:GOTerm {go_id: 'GO:0006811'}) ON CREATE SET g.name = 'ion transport', g.namespace = 'biological_process';

// PREDICTS_GO 엣지 적재 예시 (PRISM 점수 연동 후 실행):
// MATCH (i:Isoform {transcript_id: $transcript_id})
// MATCH (go:GOTerm {go_id: $go_id})
// MERGE (i)-[r:PREDICTS_GO]->(go)
// ON CREATE SET r.auprc = $auprc,
//               r.tissue = $tissue,
//               r.model = 'PRISM_v15d_ESM2',
//               r.prism_version = 'v15d_bp_clean';

