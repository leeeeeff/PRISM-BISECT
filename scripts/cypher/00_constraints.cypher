// 제약조건 및 인덱스
// 생성: bisect_to_neo4j.py

// ─── 고유성 제약조건 ───────────────────────────────────────────
CREATE CONSTRAINT gene_symbol IF NOT EXISTS
  FOR (g:Gene) REQUIRE g.symbol IS UNIQUE;

CREATE CONSTRAINT isoform_transcript_id IF NOT EXISTS
  FOR (i:Isoform) REQUIRE i.transcript_id IS UNIQUE;

CREATE CONSTRAINT cluster_leiden_id IF NOT EXISTS
  FOR (c:CellCluster) REQUIRE c.leiden_id IS UNIQUE;

CREATE CONSTRAINT layer_label IF NOT EXISTS
  FOR (l:CorticalLayer) REQUIRE l.label IS UNIQUE;

CREATE CONSTRAINT celltype_name IF NOT EXISTS
  FOR (t:CellType) REQUIRE t.name IS UNIQUE;

CREATE CONSTRAINT disease_mondo IF NOT EXISTS
  FOR (d:Disease) REQUIRE d.mondo_id IS UNIQUE;

CREATE CONSTRAINT cohort_name IF NOT EXISTS
  FOR (h:Cohort) REQUIRE h.name IS UNIQUE;

CREATE CONSTRAINT donor_id IF NOT EXISTS
  FOR (n:Donor) REQUIRE n.donor_id IS UNIQUE;

CREATE CONSTRAINT goterm_id IF NOT EXISTS
  FOR (g:GOTerm) REQUIRE g.go_id IS UNIQUE;

// ─── 조회 성능 인덱스 ──────────────────────────────────────────
CREATE INDEX cluster_sig IF NOT EXISTS
  FOR (c:CellCluster) ON (c.sig);

CREATE INDEX isoform_gene IF NOT EXISTS
  FOR (i:Isoform) ON (i.gene_symbol);
