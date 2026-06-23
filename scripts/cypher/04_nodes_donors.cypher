// Donor 노드
// 생성: bisect_to_neo4j.py

// ─── Donor 노드 + FROM_COHORT ───────────────────────────────
MERGE (n:Donor {donor_id: 'PO05'}) ON CREATE SET n.cohort = 'PO', n.condition = 'AD';
MERGE (n:Donor {donor_id: 'PO05'}) MERGE (h:Cohort {name: 'PO'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'PO11'}) ON CREATE SET n.cohort = 'PO', n.condition = 'AD';
MERGE (n:Donor {donor_id: 'PO11'}) MERGE (h:Cohort {name: 'PO'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'PO13'}) ON CREATE SET n.cohort = 'PO', n.condition = 'Control';
MERGE (n:Donor {donor_id: 'PO13'}) MERGE (h:Cohort {name: 'PO'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'PO15'}) ON CREATE SET n.cohort = 'PO', n.condition = 'Control';
MERGE (n:Donor {donor_id: 'PO15'}) MERGE (h:Cohort {name: 'PO'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'PO20'}) ON CREATE SET n.cohort = 'PO', n.condition = 'Control';
MERGE (n:Donor {donor_id: 'PO20'}) MERGE (h:Cohort {name: 'PO'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'PO23'}) ON CREATE SET n.cohort = 'PO', n.condition = 'Control';
MERGE (n:Donor {donor_id: 'PO23'}) MERGE (h:Cohort {name: 'PO'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'PO28'}) ON CREATE SET n.cohort = 'PO', n.condition = 'AD';
MERGE (n:Donor {donor_id: 'PO28'}) MERGE (h:Cohort {name: 'PO'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'PO41'}) ON CREATE SET n.cohort = 'PO', n.condition = 'AD';
MERGE (n:Donor {donor_id: 'PO41'}) MERGE (h:Cohort {name: 'PO'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'PO42'}) ON CREATE SET n.cohort = 'PO', n.condition = 'AD';
MERGE (n:Donor {donor_id: 'PO42'}) MERGE (h:Cohort {name: 'PO'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC027'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'Control';
MERGE (n:Donor {donor_id: 'SMC027'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC029'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'AD';
MERGE (n:Donor {donor_id: 'SMC029'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC030'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'Control';
MERGE (n:Donor {donor_id: 'SMC030'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC031'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'Active control';
MERGE (n:Donor {donor_id: 'SMC031'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC032'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'AD';
MERGE (n:Donor {donor_id: 'SMC032'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC033'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'Control';
MERGE (n:Donor {donor_id: 'SMC033'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC035'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'Active control';
MERGE (n:Donor {donor_id: 'SMC035'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC036'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'AD';
MERGE (n:Donor {donor_id: 'SMC036'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC037'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'Active control';
MERGE (n:Donor {donor_id: 'SMC037'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC038'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'AD';
MERGE (n:Donor {donor_id: 'SMC038'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC039'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'AD';
MERGE (n:Donor {donor_id: 'SMC039'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC041'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'AD';
MERGE (n:Donor {donor_id: 'SMC041'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC043'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'AD';
MERGE (n:Donor {donor_id: 'SMC043'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC049'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'AD';
MERGE (n:Donor {donor_id: 'SMC049'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC052'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'Control';
MERGE (n:Donor {donor_id: 'SMC052'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

MERGE (n:Donor {donor_id: 'SMC053'}) ON CREATE SET n.cohort = 'SMC', n.condition = 'Active control';
MERGE (n:Donor {donor_id: 'SMC053'}) MERGE (h:Cohort {name: 'SMC'}) MERGE (n)-[:FROM_COHORT]->(h);

