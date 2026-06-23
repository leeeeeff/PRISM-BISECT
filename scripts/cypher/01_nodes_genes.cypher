// Gene 노드
// 생성: bisect_to_neo4j.py

// ─── Gene 노드 ───────────────────────────────────────────────
MERGE (g:Gene {symbol: 'ADARB2'})
  ON CREATE SET g.hgnc_id = 'HGNC:226', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'APP'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'AQP4'})
  ON CREATE SET g.hgnc_id = 'HGNC:634', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'BCL11A'})
  ON CREATE SET g.hgnc_id = 'HGNC:13221', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'BCL11B'})
  ON CREATE SET g.hgnc_id = 'HGNC:13222', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'BIN1'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'C3'})
  ON CREATE SET g.hgnc_id = 'HGNC:1318', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'CALB2'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'CAMK2A'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'CD3E'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'CLDN5'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'CLU'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'CSPG4'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'CUX1'})
  ON CREATE SET g.hgnc_id = 'HGNC:2654', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'DLG1'})
  ON CREATE SET g.hgnc_id = 'HGNC:2900', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'ENPP2'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'ETV1'})
  ON CREATE SET g.hgnc_id = 'HGNC:3490', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'FEZF2'})
  ON CREATE SET g.hgnc_id = 'HGNC:26246', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'FLT1'})
  ON CREATE SET g.hgnc_id = 'HGNC:3763', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'FOXP2'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'GAD2'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'GJA1'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'GNLY'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'KIF21B'})
  ON CREATE SET g.hgnc_id = 'HGNC:18349', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'KIF5A'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'KIF5C'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'KIT'})
  ON CREATE SET g.hgnc_id = 'HGNC:6342', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'KLK6'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'LAMP5'})
  ON CREATE SET g.hgnc_id = 'HGNC:18994', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'LHX2'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'LHX6'})
  ON CREATE SET g.hgnc_id = 'HGNC:6591', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'LHX9'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'MAG'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'MAPT'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'MBP'})
  ON CREATE SET g.hgnc_id = 'HGNC:6925', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'MOG'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'MS4A1'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'MT-CO1'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'NDNF'})
  ON CREATE SET g.hgnc_id = 'HGNC:25696', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'NDUFS4'})
  ON CREATE SET g.hgnc_id = 'HGNC:7714', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'NDUFS7'})
  ON CREATE SET g.hgnc_id = 'HGNC:7717', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'NDUFS8'})
  ON CREATE SET g.hgnc_id = 'HGNC:7718', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'NKG7'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'NOTCH3'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'NOVA1'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'NOVA2'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'NRGN'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'OLIG1'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'OLIG2'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'OPALIN'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'P2RY12'})
  ON CREATE SET g.hgnc_id = 'HGNC:18124', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'PDGFRA'})
  ON CREATE SET g.hgnc_id = 'HGNC:8803', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'PDGFRB'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'PROX1'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'PRSS12'})
  ON CREATE SET g.hgnc_id = 'HGNC:9477', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'PVALB'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'RBFOX1'})
  ON CREATE SET g.hgnc_id = 'HGNC:23674', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'RBFOX2'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'RGS5'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'RORB'})
  ON CREATE SET g.hgnc_id = 'HGNC:10260', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'RSPO1'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'SATB2'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'SLC17A7'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'SLC1A2'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'SP8'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'SPP1'})
  ON CREATE SET g.hgnc_id = 'HGNC:11255', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'SRSF1'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'SRSF5'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'SST'})
  ON CREATE SET g.hgnc_id = 'HGNC:11329', g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'SULF2'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'SYT6'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'TLE4'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'TMEM119'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'TREM2'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'VIP'})
  ON CREATE SET g.source = 'BISECT/PRISM';

MERGE (g:Gene {symbol: 'VWF'})
  ON CREATE SET g.source = 'BISECT/PRISM';

