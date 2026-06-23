// 진단 및 탐색 쿼리 — Neo4j Browser에서 실행
// 생성: bisect_to_neo4j.py

// ─── 진단 및 탐색 쿼리 모음 ─────────────────────────────────
// Neo4j Browser에서 개별 실행

// [1] 전체 노드/엣지 통계
MATCH (n) RETURN labels(n) AS label, count(*) AS cnt ORDER BY cnt DESC;

// [2] AD-enriched 클러스터 목록 (p<0.05)
MATCH (c:CellCluster)-[r:AD_ENRICHED]->(d:Disease)
WHERE r.p_value < 0.05
RETURN c.leiden_id, c.subtype, c.cell_type, r.delta_pct, r.p_value, r.sig
ORDER BY r.p_value;

// [3] KIF21B isoform switch 세포 타입별 조회
MATCH (t:CellType)-[r:SHOWS_ISOFORM_SWITCH]->(g:Gene {symbol: 'KIF21B'})
RETURN t.name, r.AD_ratio, r.CT_ratio, r.delta, r.p_value, r.sig
ORDER BY r.p_value;

// [4] C18 클러스터의 모든 관계 탐색
MATCH (c:CellCluster {leiden_id: 18})-[r]-(x)
RETURN type(r), labels(x), x.symbol, x.name, x.subtype LIMIT 30;

// [5] Braak 상관 이소폼 (r < -0.3)
MATCH (i:Isoform)-[r:CORRELATES_WITH_BRAAK]->(d:Disease)
WHERE r.spearman_r < -0.3
RETURN i.transcript_id, i.gene_symbol, r.spearman_r, r.p_value,
       r.p_bonferroni, r.description
ORDER BY r.spearman_r;

// [6] 코호트 방향 일치 검증 (배치효과 확인)
MATCH (c:CellCluster)-[po:VALIDATED_IN]->(h1:Cohort {name: "PO"})
MATCH (c)-[smc:VALIDATED_IN]->(h2:Cohort {name: "SMC"})
RETURN c.leiden_id, c.subtype,
       po.direction AS PO_dir, po.delta_pct AS PO_delta,
       smc.direction AS SMC_dir, smc.delta_pct AS SMC_delta,
       po.direction = smc.direction AS batch_ok
ORDER BY c.leiden_id;

// [7] 3-hop 경로: LAMP5/KIT → isoform switch → AD
MATCH p = (g:Gene {symbol: 'KIT'})<-[:DEFINED_BY_MARKER]-(c:CellCluster)
         -[:AD_ENRICHED]->(d:Disease)
RETURN p;

// [8] Complex I 수렴 확인 (NDUFS4/7/8 동시 switch)
MATCH (t:CellType)-[r:SHOWS_ISOFORM_SWITCH]->(g:Gene)
WHERE g.symbol IN ['NDUFS4', 'NDUFS7', 'NDUFS8']
RETURN t.name, g.symbol, r.sig, r.p_value
ORDER BY g.symbol, r.p_value;
