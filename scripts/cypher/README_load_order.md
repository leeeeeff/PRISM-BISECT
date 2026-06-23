# Neo4j 적재 순서

cypher-shell 또는 Neo4j Browser에서 파일 번호 순서대로 실행.

```bash
# 환경 변수 설정
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASS=<password>

for f in $(ls cypher/*.cypher | sort); do
    echo "=== $f ==="
    cypher-shell -a $NEO4J_URL -u $NEO4J_USER -p $NEO4J_PASS --file "$f"
done
```

## 파일 목록 및 역할

| 파일 | 내용 |
|------|------|
| 00_constraints.cypher | 고유성 제약조건 + 인덱스 |
| 01_nodes_genes.cypher | Gene 노드 (HGNC 식별자) |
| 02_nodes_isoforms.cypher | Isoform 노드 + Gene-HAS_ISOFORM |
| 03_nodes_structural.cypher | CellCluster / Layer / CellType / Disease / Cohort |
| 04_nodes_donors.cypher | Donor 노드 + FROM_COHORT |
| 05_edges_cluster_disease.cypher | AD_ENRICHED / AD_DEPLETED |
| 06_edges_layer.cypher | LOCATED_IN (층 할당) |
| 07_edges_markers.cypher | DEFINED_BY_MARKER |
| 08_edges_isoform_switch.cypher | SHOWS_ISOFORM_SWITCH |
| 09_edges_braak.cypher | CORRELATES_WITH_BRAAK |
| 10_edges_cohort_validation.cypher | VALIDATED_IN (배치효과 검증) |
| 11_edges_donor_cluster.cypher | HAS_CLUSTER_PCT |
| 12_edges_c18_expression.cypher | OVEREXPRESSED_IN / UNDEREXPRESSED_IN |
| 13_prism_placeholder.cypher | GOTerm 노드 + PREDICTS_GO 구조 |
| 99_diagnostic_queries.cypher | 탐색 쿼리 모음 |

## 노드 통계 예상값

| 레이블 | 예상 수 |
|--------|---------|
| Gene | ~60 |
| Isoform | ~3 (braak 데이터 기준, PRISM 연동 후 수천 개로 확장) |
| CellCluster | 30 |
| CorticalLayer | ~10 |
| CellType | 8 |
| Disease | 2 (AD, Control) |
| Cohort | 3 (PO, SMC, ALL) |
| Donor | ~25 |

## PRISM 연동 (향후)

`13_prism_placeholder.cypher`의 예시 쿼리를 참고하여,
`generate_brain_full_extended_scores.py` 출력과 연동하면
Isoform → PREDICTS_GO → GOTerm 엣지 수천 개 추가 가능.
