// CellCluster / CorticalLayer / CellType / Disease / Cohort 노드
// 생성: bisect_to_neo4j.py

// ─── CellCluster 노드 ────────────────────────────────────────
MERGE (c:CellCluster {leiden_id: 1}) ON CREATE SET c.subtype = 'L2/3 IT', c.cell_type = 'Excitatory', c.n_cells = 10258, c.confidence = 'High', c.layer = 'L2/3', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';
MERGE (t:CellType {name: 'Excitatory'}) ON CREATE SET t.cl_id = 'CL:0008030', t.ontology = 'Cell Ontology';
MERGE (l:CorticalLayer {label: 'L2/3'}) ON CREATE SET l.allen_id = 'MBA:304', l.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 2}) ON CREATE SET c.subtype = 'L2/3 IT', c.cell_type = 'Excitatory', c.n_cells = 9711, c.confidence = 'High', c.layer = 'L2/3', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 27}) ON CREATE SET c.subtype = 'L2/3 IT', c.cell_type = 'Excitatory', c.n_cells = 666, c.confidence = 'High', c.layer = 'L2/3', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 10}) ON CREATE SET c.subtype = 'L4 IT', c.cell_type = 'Excitatory', c.n_cells = 4156, c.confidence = 'Medium', c.layer = 'L4', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';
MERGE (l:CorticalLayer {label: 'L4'}) ON CREATE SET l.allen_id = 'MBA:657', l.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 11}) ON CREATE SET c.subtype = 'L4 IT', c.cell_type = 'Excitatory', c.n_cells = 4290, c.confidence = 'Medium', c.layer = 'L4', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 18}) ON CREATE SET c.subtype = 'L4 IT atypical', c.cell_type = 'Excitatory', c.n_cells = 1603, c.confidence = 'Medium', c.layer = 'L4', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 19}) ON CREATE SET c.subtype = 'L5 ET', c.cell_type = 'Excitatory', c.n_cells = 1410, c.confidence = 'High', c.layer = 'L5', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';
MERGE (l:CorticalLayer {label: 'L5'}) ON CREATE SET l.allen_id = 'MBA:747', l.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 26}) ON CREATE SET c.subtype = 'L5 ET', c.cell_type = 'Excitatory', c.n_cells = 744, c.confidence = 'High', c.layer = 'L5', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 25}) ON CREATE SET c.subtype = 'L6 CT', c.cell_type = 'Excitatory', c.n_cells = 757, c.confidence = 'High', c.layer = 'L6', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';
MERGE (l:CorticalLayer {label: 'L6'}) ON CREATE SET l.allen_id = 'MBA:800', l.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 20}) ON CREATE SET c.subtype = 'Unclassified', c.cell_type = 'Excitatory', c.n_cells = 1356, c.confidence = 'Low', c.layer = 'L6', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 7}) ON CREATE SET c.subtype = 'PV', c.cell_type = 'Inhibitory', c.n_cells = 5537, c.confidence = 'High', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';
MERGE (t:CellType {name: 'Inhibitory'}) ON CREATE SET t.cl_id = 'CL:0008031', t.ontology = 'Cell Ontology';

MERGE (c:CellCluster {leiden_id: 21}) ON CREATE SET c.subtype = 'PV', c.cell_type = 'Inhibitory', c.n_cells = 1209, c.confidence = 'High', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 9}) ON CREATE SET c.subtype = 'SST', c.cell_type = 'Inhibitory', c.n_cells = 2864, c.confidence = 'High', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 8}) ON CREATE SET c.subtype = 'VIP', c.cell_type = 'Inhibitory', c.n_cells = 5088, c.confidence = 'High', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 16}) ON CREATE SET c.subtype = 'VIP', c.cell_type = 'Inhibitory', c.n_cells = 2321, c.confidence = 'High', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 13}) ON CREATE SET c.subtype = 'LAMP5/NDNF', c.cell_type = 'Inhibitory', c.n_cells = 2652, c.confidence = 'Medium', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 15}) ON CREATE SET c.subtype = 'LAMP5/KIT', c.cell_type = 'Inhibitory', c.n_cells = 2097, c.confidence = 'Medium', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 22}) ON CREATE SET c.subtype = 'LAMP5/LHX6', c.cell_type = 'Inhibitory', c.n_cells = 979, c.confidence = 'Medium', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 24}) ON CREATE SET c.subtype = 'CGE-derived misc', c.cell_type = 'Inhibitory', c.n_cells = 681, c.confidence = 'Low', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 0}) ON CREATE SET c.subtype = 'MOL1/2', c.cell_type = 'Oligodendrocyte', c.n_cells = 14993, c.confidence = 'High', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';
MERGE (t:CellType {name: 'Oligodendrocyte'}) ON CREATE SET t.cl_id = 'CL:0000128', t.ontology = 'Cell Ontology';

MERGE (c:CellCluster {leiden_id: 3}) ON CREATE SET c.subtype = 'MOL2', c.cell_type = 'Oligodendrocyte', c.n_cells = 3433, c.confidence = 'High', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 28}) ON CREATE SET c.subtype = 'MOL5/6', c.cell_type = 'Oligodendrocyte', c.n_cells = 502, c.confidence = 'Medium', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 32}) ON CREATE SET c.subtype = 'MOL5/6', c.cell_type = 'Oligodendrocyte', c.n_cells = 8, c.confidence = 'Low', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 6}) ON CREATE SET c.subtype = 'OPC', c.cell_type = 'OPC', c.n_cells = 6784, c.confidence = 'High', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';
MERGE (t:CellType {name: 'OPC'}) ON CREATE SET t.cl_id = 'CL:0002453', t.ontology = 'Cell Ontology';

MERGE (c:CellCluster {leiden_id: 31}) ON CREATE SET c.subtype = 'Immature OL', c.cell_type = 'OPC', c.n_cells = 86, c.confidence = 'Low', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 4}) ON CREATE SET c.subtype = 'Protoplasmic', c.cell_type = 'Astrocyte', c.n_cells = 6047, c.confidence = 'High', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';
MERGE (t:CellType {name: 'Astrocyte'}) ON CREATE SET t.cl_id = 'CL:0000127', t.ontology = 'Cell Ontology';

MERGE (c:CellCluster {leiden_id: 5}) ON CREATE SET c.subtype = 'SPP1⁺ activated', c.cell_type = 'Microglia', c.n_cells = 3577, c.confidence = 'Medium', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';
MERGE (t:CellType {name: 'Microglia'}) ON CREATE SET t.cl_id = 'CL:0000129', t.ontology = 'Cell Ontology';

MERGE (c:CellCluster {leiden_id: 17}) ON CREATE SET c.subtype = 'Pericyte', c.cell_type = 'Vascular', c.n_cells = 944, c.confidence = 'High', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';
MERGE (t:CellType {name: 'Vascular'}) ON CREATE SET t.cl_id = '', t.ontology = 'Cell Ontology';

MERGE (c:CellCluster {leiden_id: 23}) ON CREATE SET c.subtype = 'Endothelial', c.cell_type = 'Vascular', c.n_cells = 563, c.confidence = 'High', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';

MERGE (c:CellCluster {leiden_id: 29}) ON CREATE SET c.subtype = 'NK/T cell', c.cell_type = 'Lymphocyte', c.n_cells = 171, c.confidence = 'Low', c.layer = '', c.species = 'NCBITaxon:9606', c.tissue = 'UBERON:0001851';
MERGE (t:CellType {name: 'Lymphocyte'}) ON CREATE SET t.cl_id = 'CL:0000542', t.ontology = 'Cell Ontology';

// ─── Disease 노드 ────────────────────────────────────────
MERGE (d:Disease {mondo_id: 'MONDO:0004975'}) ON CREATE SET d.name = 'AD', d.ontology = 'MONDO';
MERGE (d:Disease {mondo_id: 'MONDO:0000001'}) ON CREATE SET d.name = 'Control', d.ontology = 'MONDO';

// ─── Cohort 노드 ─────────────────────────────────────────
MERGE (h:Cohort {name: 'PO'}) ON CREATE SET h.library_version = '10x Chromium 3'v4', h.species = 'NCBITaxon:9606';
MERGE (h:Cohort {name: 'SMC'}) ON CREATE SET h.library_version = '10x Chromium 3'v3', h.species = 'NCBITaxon:9606';
MERGE (h:Cohort {name: 'ALL'}) ON CREATE SET h.library_version = 'combined', h.species = 'NCBITaxon:9606';

