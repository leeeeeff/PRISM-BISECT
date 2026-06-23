"""
BISECT → Neo4j Cypher 변환 파이프라인
======================================
입력  : prism_app/data/cell_atlas/*.tsv  (BISECT 결과 전체)
출력  : scripts/cypher/  디렉토리 *.cypher 파일
실행  : conda activate isoform_env && python scripts/bisect_to_neo4j.py
적재  : cypher-shell -u neo4j -p <pw> --file cypher/00_constraints.cypher
        또는 Neo4j Browser에서 파일별 순서대로 실행

Neo4j 스키마 (노드 레이블 / 관계 유형)
---------------------------------------
노드:
  Gene          {symbol, hgnc_id}
  Isoform       {transcript_id, gene_symbol, source, description}
  CellCluster   {leiden_id, subtype, cell_type, n_cells, confidence}
  CorticalLayer {label, allen_id}
  CellType      {name, cl_id}
  Disease       {name, mondo_id}
  Cohort        {name, library_version}
  Donor         {donor_id, cohort, condition}
  GOTerm        {go_id, name, namespace}          ← PRISM 연결용 (미래)

관계:
  (CellCluster)-[:AD_ENRICHED]→(Disease)         delta>0, sig
  (CellCluster)-[:AD_DEPLETED]→(Disease)         delta<0, sig
  (CellCluster)-[:LOCATED_IN]→(CorticalLayer)    layer_score, ambiguity
  (CellCluster)-[:DEFINED_BY_MARKER]→(Gene)      expression
  (CellType)-[:SHOWS_ISOFORM_SWITCH]→(Gene)      AD/CT ratio, p, sig
  (Isoform)-[:CORRELATES_WITH_BRAAK]→(Disease)   spearman_r, CI
  (CellCluster)-[:VALIDATED_IN]→(Cohort)         delta, p, cohort-specific
  (Donor)-[:HAS_CLUSTER_PCT]→(CellCluster)       pct, n_cells, condition
  (Gene)-[:OVEREXPRESSED_IN]→(CellCluster)       log2FC (C18 분석)
  (Gene)-[:HAS_ISOFORM]→(Isoform)               GENCODE/novel
  (Isoform)-[:PREDICTS_GO]→(GOTerm)             PRISM AUPRC (placeholder)
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pandas as pd

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parents[1]
DATA   = ROOT / 'prism_app' / 'data' / 'cell_atlas'
OUT    = Path(__file__).parent / 'cypher'
OUT.mkdir(exist_ok=True)

# ── 표준 식별자 매핑 테이블 ────────────────────────────────────────────────────
# 협업자 시스템에서 사용할 표준 온톨로지 ID (초기값; 추후 확장)
GENE_IDS: dict[str, str] = {
    'KIF21B':  'HGNC:18349',
    'NDUFS4':  'HGNC:7714',
    'NDUFS7':  'HGNC:7717',
    'NDUFS8':  'HGNC:7718',
    'DLG1':    'HGNC:2900',
    'PRSS12':  'HGNC:9477',
    'LAMP5':   'HGNC:18994',
    'KIT':     'HGNC:6342',
    'SST':     'HGNC:11329',
    'RBFOX1':  'HGNC:23674',
    'CUX1':    'HGNC:2654',
    'RORB':    'HGNC:10260',
    'FEZF2':   'HGNC:26246',
    'BCL11B':  'HGNC:13222',
    'ADARB2':  'HGNC:226',
    'NDNF':    'HGNC:25696',
    'LHX6':    'HGNC:6591',
    'MBP':     'HGNC:6925',
    'PDGFRA':  'HGNC:8803',
    'AQP4':    'HGNC:634',
    'P2RY12':  'HGNC:18124',
    'SPP1':    'HGNC:11255',
    'C3':      'HGNC:1318',
    'FLT1':    'HGNC:3763',
    'ETV1':    'HGNC:3490',
    'BCL11A':  'HGNC:13221',
}

CELL_TYPE_CL: dict[str, str] = {
    'Excitatory neuron':  'CL:0008030',
    'Excitatory':         'CL:0008030',
    'Inhibitory neuron':  'CL:0008031',
    'Inhibitory':         'CL:0008031',
    'Oligodendrocyte':    'CL:0000128',
    'OPC':                'CL:0002453',
    'Astrocyte':          'CL:0000127',
    'Microglia':          'CL:0000129',
    'Vascular cell':      'CL:0000058',
    'Lymphocyte':         'CL:0000542',
    'Excitatory_neuron':  'CL:0008030',
    'Inhibitory_neuron':  'CL:0008031',
}

LAYER_ALLEN: dict[str, str] = {
    'L1': 'MBA:195', 'L2/3': 'MBA:304', 'L2-3': 'MBA:304',
    'L4': 'MBA:657', 'L5': 'MBA:747', 'L5/6': 'MBA:747',
    'L6': 'MBA:800', 'WM': 'MBA:901', 'WM/L6': 'MBA:901',
    'Mixed': 'UBERON:0008933', 'Perivasc': 'UBERON:0001977',
}

DISEASE_MONDO = {
    'AD': 'MONDO:0004975',
    'Control': 'MONDO:0000001',
}

COHORT_LIB = {
    'PO':  "10x Chromium 3'v4",
    'SMC': "10x Chromium 3'v3",
    'ALL': 'combined',
}

# ── 헬퍼 ──────────────────────────────────────────────────────────────────────
def esc(v: object) -> str:
    """Neo4j 문자열 값 이스케이프."""
    return str(v).replace("'", "\\'").replace('"', '\\"')


def props(**kw) -> str:
    """dict → Neo4j 인라인 프로퍼티 문자열."""
    parts = []
    for k, v in kw.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if isinstance(v, (int, float)):
            parts.append(f'{k}: {v}')
        else:
            parts.append(f"{k}: '{esc(v)}'")
    return '{' + ', '.join(parts) + '}'


def write_cypher(filename: str, lines: list[str], header: str = '') -> None:
    path = OUT / filename
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(f'// {header}\n')
            f.write(f'// 생성: bisect_to_neo4j.py\n\n')
        f.write('\n'.join(lines))
        f.write('\n')
    print(f'  ✓ {path.name}  ({len(lines)} 문장)')


def parse_markers(marker_str: str) -> list[tuple[str, float]]:
    """'CUX1=0.91, SATB2=1.55' → [('CUX1', 0.91), ('SATB2', 1.55)]"""
    result = []
    for m in re.findall(r'([A-Z][A-Z0-9]+)=([\d.]+)', str(marker_str)):
        try:
            result.append((m[0], float(m[1])))
        except ValueError:
            pass
    return result


# ── 00. 제약조건 & 인덱스 ─────────────────────────────────────────────────────
def build_constraints() -> None:
    lines = [
        '// ─── 고유성 제약조건 ───────────────────────────────────────────',
        'CREATE CONSTRAINT gene_symbol IF NOT EXISTS',
        '  FOR (g:Gene) REQUIRE g.symbol IS UNIQUE;',
        '',
        'CREATE CONSTRAINT isoform_transcript_id IF NOT EXISTS',
        '  FOR (i:Isoform) REQUIRE i.transcript_id IS UNIQUE;',
        '',
        'CREATE CONSTRAINT cluster_leiden_id IF NOT EXISTS',
        '  FOR (c:CellCluster) REQUIRE c.leiden_id IS UNIQUE;',
        '',
        'CREATE CONSTRAINT layer_label IF NOT EXISTS',
        '  FOR (l:CorticalLayer) REQUIRE l.label IS UNIQUE;',
        '',
        'CREATE CONSTRAINT celltype_name IF NOT EXISTS',
        '  FOR (t:CellType) REQUIRE t.name IS UNIQUE;',
        '',
        'CREATE CONSTRAINT disease_mondo IF NOT EXISTS',
        '  FOR (d:Disease) REQUIRE d.mondo_id IS UNIQUE;',
        '',
        'CREATE CONSTRAINT cohort_name IF NOT EXISTS',
        '  FOR (h:Cohort) REQUIRE h.name IS UNIQUE;',
        '',
        'CREATE CONSTRAINT donor_id IF NOT EXISTS',
        '  FOR (n:Donor) REQUIRE n.donor_id IS UNIQUE;',
        '',
        'CREATE CONSTRAINT goterm_id IF NOT EXISTS',
        '  FOR (g:GOTerm) REQUIRE g.go_id IS UNIQUE;',
        '',
        '// ─── 조회 성능 인덱스 ──────────────────────────────────────────',
        'CREATE INDEX cluster_sig IF NOT EXISTS',
        '  FOR (c:CellCluster) ON (c.sig);',
        '',
        'CREATE INDEX isoform_gene IF NOT EXISTS',
        '  FOR (i:Isoform) ON (i.gene_symbol);',
    ]
    write_cypher('00_constraints.cypher', lines, '제약조건 및 인덱스')


# ── 01. Gene 노드 ─────────────────────────────────────────────────────────────
def build_gene_nodes(sub_df: pd.DataFrame, ct_df: pd.DataFrame,
                     c18_df: pd.DataFrame) -> set[str]:
    """모든 소스에서 유전자 심볼 수집 → Gene MERGE 문."""
    genes: set[str] = set()

    # final_subtype_classification markers
    for _, row in sub_df.iterrows():
        for g, _ in parse_markers(row.get('markers', '')):
            genes.add(g)

    # celltype_isoform_switch
    if not ct_df.empty:
        genes.update(ct_df['gene'].dropna().unique())

    # c18_vs_c10c11_diff
    if not c18_df.empty:
        genes.update(c18_df['gene'].dropna().unique())

    # GENE_IDS 테이블에 등록된 유전자도 포함
    genes.update(GENE_IDS.keys())

    lines = ['// ─── Gene 노드 ───────────────────────────────────────────────']
    for sym in sorted(genes):
        hgnc = GENE_IDS.get(sym, '')
        merge_clause = f"MERGE (g:Gene {{symbol: '{sym}'}})"
        lines.append(merge_clause)
        if hgnc:
            lines.append(f"  ON CREATE SET g.hgnc_id = '{hgnc}', g.source = 'BISECT/PRISM';")
        else:
            lines.append(f"  ON CREATE SET g.source = 'BISECT/PRISM';")
        lines.append('')

    write_cypher('01_nodes_genes.cypher', lines, 'Gene 노드')
    return genes


# ── 02. Isoform 노드 ──────────────────────────────────────────────────────────
def build_isoform_nodes(braak_df: pd.DataFrame) -> None:
    """braak_correlation_results.tsv의 ct_isoform 컬럼에서 이소폼 노드 생성."""
    if braak_df.empty:
        return
    lines = ['// ─── Isoform 노드 ────────────────────────────────────────────']
    lines.append('// 출처: braak_correlation_results.tsv (ct_isoform 컬럼)')
    lines.append('// transcript_id: ENST... 또는 novel ID (IsoQuant 출력)')
    lines.append('')

    for _, row in braak_df.iterrows():
        tid = str(row.get('ct_isoform', '')).strip()
        gene = str(row.get('gene', '')).strip()
        desc = str(row.get('description', '')).strip()
        if not tid or not gene:
            continue
        source = 'GENCODE' if tid.startswith('ENST') else 'novel_IsoQuant'
        lines.append(f"MERGE (i:Isoform {{transcript_id: '{esc(tid)}'}}) "
                     f"ON CREATE SET i.gene_symbol = '{esc(gene)}', "
                     f"i.source = '{source}', "
                     f"i.description = '{esc(desc)}';")
        lines.append(f"MERGE (g:Gene {{symbol: '{esc(gene)}'}}) "
                     f"MERGE (i:Isoform {{transcript_id: '{esc(tid)}'}}) "
                     f"MERGE (g)-[:HAS_ISOFORM]->(i);")
        lines.append('')

    write_cypher('02_nodes_isoforms.cypher', lines, 'Isoform 노드 + Gene-HAS_ISOFORM')


# ── 03. CellCluster / CorticalLayer / CellType / Disease / Cohort 노드 ────────
def build_structural_nodes(sub_df: pd.DataFrame, layer_df: pd.DataFrame) -> None:
    lines = ['// ─── CellCluster 노드 ────────────────────────────────────────']

    layer_map: dict[str, str] = {}
    if not layer_df.empty:
        layer_map = dict(zip(layer_df['cluster'].astype(str),
                             layer_df['layer_label']))

    seen_ct: set[str] = set()
    seen_layer: set[str] = set()

    for _, row in sub_df.iterrows():
        lid  = int(row['cluster'])
        ct   = str(row.get('cell_type', '')).strip()
        st   = str(row.get('subtype', '')).strip()
        n    = int(row.get('n', 0))
        conf = str(row.get('confidence', '')).strip()
        lay  = layer_map.get(str(lid), '')

        lines.append(
            f"MERGE (c:CellCluster {{leiden_id: {lid}}}) "
            f"ON CREATE SET c.subtype = '{esc(st)}', "
            f"c.cell_type = '{esc(ct)}', "
            f"c.n_cells = {n}, "
            f"c.confidence = '{esc(conf)}', "
            f"c.layer = '{esc(lay)}', "
            f"c.species = 'NCBITaxon:9606', "
            f"c.tissue = 'UBERON:0001851';"   # prefrontal cortex
        )

        # CellType 노드
        if ct and ct not in seen_ct:
            cl_id = CELL_TYPE_CL.get(ct, '')
            lines.append(
                f"MERGE (t:CellType {{name: '{esc(ct)}'}}) "
                f"ON CREATE SET t.cl_id = '{cl_id}', t.ontology = 'Cell Ontology';"
            )
            seen_ct.add(ct)

        # CorticalLayer 노드
        if lay and lay not in seen_layer:
            allen = LAYER_ALLEN.get(lay, '')
            lines.append(
                f"MERGE (l:CorticalLayer {{label: '{esc(lay)}'}}) "
                f"ON CREATE SET l.allen_id = '{allen}', "
                f"l.tissue = 'UBERON:0001851';"
            )
            seen_layer.add(lay)

        lines.append('')

    # Disease 노드
    lines.append('// ─── Disease 노드 ────────────────────────────────────────')
    for name, mondo in DISEASE_MONDO.items():
        lines.append(
            f"MERGE (d:Disease {{mondo_id: '{mondo}'}}) "
            f"ON CREATE SET d.name = '{name}', d.ontology = 'MONDO';"
        )
    lines.append('')

    # Cohort 노드
    lines.append('// ─── Cohort 노드 ─────────────────────────────────────────')
    for name, lib in COHORT_LIB.items():
        lines.append(
            f"MERGE (h:Cohort {{name: '{name}'}}) "
            f"ON CREATE SET h.library_version = '{lib}', "
            f"h.species = 'NCBITaxon:9606';"
        )
    lines.append('')

    write_cypher('03_nodes_structural.cypher', lines,
                 'CellCluster / CorticalLayer / CellType / Disease / Cohort 노드')


# ── 04. Donor 노드 ────────────────────────────────────────────────────────────
def build_donor_nodes(donor_df: pd.DataFrame) -> None:
    if donor_df.empty:
        return
    lines = ['// ─── Donor 노드 + FROM_COHORT ───────────────────────────────']
    seen: set[str] = set()
    for _, row in donor_df.iterrows():
        donor = str(row.get('donor', '')).strip()
        cohort = str(row.get('cohort', '')).strip()
        cond   = str(row.get('condition', '')).strip()
        if not donor or donor in seen:
            continue
        seen.add(donor)
        lines.append(
            f"MERGE (n:Donor {{donor_id: '{esc(donor)}'}}) "
            f"ON CREATE SET n.cohort = '{esc(cohort)}', "
            f"n.condition = '{esc(cond)}';"
        )
        if cohort in ('PO', 'SMC'):
            lines.append(
                f"MERGE (n:Donor {{donor_id: '{esc(donor)}'}}) "
                f"MERGE (h:Cohort {{name: '{esc(cohort)}'}}) "
                f"MERGE (n)-[:FROM_COHORT]->(h);"
            )
        lines.append('')
    write_cypher('04_nodes_donors.cypher', lines, 'Donor 노드')


# ── 05. CellCluster → Disease 관계 (enriched / depleted) ─────────────────────
def build_cluster_disease_edges(sub_df: pd.DataFrame) -> None:
    lines = [
        '// ─── AD_ENRICHED / AD_DEPLETED 관계 ─────────────────────────',
        '// 조건: sig in [***, **, *, †]',
        '// 속성: delta(%), p_value, sig, AD_pct, CT_pct, n_cells, cohort',
        '',
    ]
    for _, row in sub_df.iterrows():
        sig = str(row.get('sig', 'ns')).strip()
        if sig == 'ns':
            continue
        lid   = int(row['cluster'])
        delta = float(row.get('delta', 0))
        p     = float(row.get('p', 1))
        ad_p  = float(row.get('AD_pct', 0))
        ct_p  = float(row.get('CT_pct', 0))
        n     = int(row.get('n', 0))
        rel   = 'AD_ENRICHED' if delta > 0 else 'AD_DEPLETED'
        mondo = DISEASE_MONDO['AD']

        lines.append(
            f"MATCH (c:CellCluster {{leiden_id: {lid}}}), "
            f"(d:Disease {{mondo_id: '{mondo}'}}) "
            f"MERGE (c)-[r:{rel}]->(d) "
            f"ON CREATE SET r.delta_pct = {round(delta, 4)}, "
            f"r.p_value = {round(p, 6)}, "
            f"r.sig = '{sig}', "
            f"r.AD_pct = {round(ad_p, 4)}, "
            f"r.CT_pct = {round(ct_p, 4)}, "
            f"r.n_cells = {n}, "
            f"r.cohort = 'PO+SMC', "
            f"r.n_donors = 21, "
            f"r.tissue = 'UBERON:0001851', "
            f"r.species = 'NCBITaxon:9606';"
        )
        lines.append('')

    write_cypher('05_edges_cluster_disease.cypher', lines,
                 'CellCluster → AD_ENRICHED/AD_DEPLETED → Disease')


# ── 06. CellCluster → CorticalLayer 관계 ─────────────────────────────────────
def build_layer_edges(sub_df: pd.DataFrame, layer_df: pd.DataFrame) -> None:
    if layer_df.empty:
        return
    lines = ['// ─── LOCATED_IN 관계 (CellCluster → CorticalLayer) ──────────']
    layer_score_map: dict[str, float] = {}
    layer_ambig_map: dict[str, float] = {}
    if not layer_df.empty:
        for _, r in layer_df.iterrows():
            cl = str(r['cluster'])
            layer_score_map[cl] = float(r.get('layer_score_max', 0))
            layer_ambig_map[cl] = float(r.get('layer_ambiguity', 0))

    layer_map = dict(zip(layer_df['cluster'].astype(str), layer_df['layer_label']))

    for _, row in sub_df.iterrows():
        lid = int(row['cluster'])
        cl_str = str(lid)
        lay = layer_map.get(cl_str, '')
        if not lay:
            continue
        score = layer_score_map.get(cl_str, 0.0)
        ambig = layer_ambig_map.get(cl_str, 0.0)
        lines.append(
            f"MATCH (c:CellCluster {{leiden_id: {lid}}}), "
            f"(l:CorticalLayer {{label: '{esc(lay)}'}}) "
            f"MERGE (c)-[r:LOCATED_IN]->(l) "
            f"ON CREATE SET r.layer_score = {round(score, 4)}, "
            f"r.ambiguity = {round(ambig, 4)}, "
            f"r.method = 'Allen_Human_Brain_Atlas_markers';"
        )
        lines.append('')
    write_cypher('06_edges_layer.cypher', lines,
                 'CellCluster → LOCATED_IN → CorticalLayer')


# ── 07. CellCluster → Gene (DEFINED_BY_MARKER) ───────────────────────────────
def build_marker_edges(sub_df: pd.DataFrame) -> None:
    lines = ['// ─── DEFINED_BY_MARKER 관계 (CellCluster → Gene) ────────────']
    for _, row in sub_df.iterrows():
        lid = int(row['cluster'])
        for gene, expr in parse_markers(row.get('markers', '')):
            lines.append(
                f"MATCH (c:CellCluster {{leiden_id: {lid}}}), "
                f"(g:Gene {{symbol: '{esc(gene)}'}}) "
                f"MERGE (c)-[r:DEFINED_BY_MARKER]->(g) "
                f"ON CREATE SET r.mean_expression = {round(expr, 4)}, "
                f"r.evidence = 'snRNA-seq_long-read';"
            )
        lines.append('')
    write_cypher('07_edges_markers.cypher', lines,
                 'CellCluster → DEFINED_BY_MARKER → Gene')


# ── 08. CellType → Gene (SHOWS_ISOFORM_SWITCH) ───────────────────────────────
def build_isoform_switch_edges(ct_df: pd.DataFrame) -> None:
    if ct_df.empty:
        return
    lines = [
        '// ─── SHOWS_ISOFORM_SWITCH 관계 (CellType → Gene) ───────────',
        '// 조건: sig in [***, **, *, †]',
        '// 속성: AD_ratio, CT_ratio, delta, p_value, sig, n_AD, n_CT',
        '',
    ]
    for _, row in ct_df.iterrows():
        sig = str(row.get('sig', 'ns')).strip()
        if sig == 'ns':
            continue
        ct    = str(row.get('cell_type', '')).strip()
        gene  = str(row.get('gene', '')).strip()
        ad_r  = float(row.get('AD_ratio', 0))
        ct_r  = float(row.get('CT_ratio', 0))
        delta = float(row.get('delta', 0))
        p     = float(row.get('p', 1))
        n_ad  = int(row.get('n_AD', 0))
        n_ct  = int(row.get('n_CT', 0))
        lines.append(
            f"MATCH (t:CellType {{name: '{esc(ct)}'}}) "
            f"MATCH (g:Gene {{symbol: '{esc(gene)}'}}) "
            f"MERGE (t)-[r:SHOWS_ISOFORM_SWITCH]->(g) "
            f"ON CREATE SET r.AD_ratio = {round(ad_r, 4)}, "
            f"r.CT_ratio = {round(ct_r, 4)}, "
            f"r.delta = {round(delta, 4)}, "
            f"r.p_value = {round(p, 6)}, "
            f"r.sig = '{sig}', "
            f"r.n_AD_donors = {n_ad}, "
            f"r.n_CT_donors = {n_ct}, "
            f"r.disease = 'MONDO:0004975', "
            f"r.tissue = 'UBERON:0001851', "
            f"r.method = 'DIU_permutation_n10000';"
        )
        lines.append('')
    write_cypher('08_edges_isoform_switch.cypher', lines,
                 'CellType → SHOWS_ISOFORM_SWITCH → Gene')


# ── 09. Isoform → Disease (CORRELATES_WITH_BRAAK) ────────────────────────────
def build_braak_edges(braak_df: pd.DataFrame) -> None:
    if braak_df.empty:
        return
    lines = [
        '// ─── CORRELATES_WITH_BRAAK 관계 (Isoform → Disease) ─────────',
        '// 속성: spearman_r, p_value, p_bonferroni, ci_95_lo, ci_95_hi, n_donors',
        '',
    ]
    mondo = DISEASE_MONDO['AD']
    for _, row in braak_df.iterrows():
        tid   = str(row.get('ct_isoform', '')).strip()
        if not tid:
            continue
        r_val = float(row.get('spearman_r', 0))
        p     = float(row.get('p_value', 1))
        p_bon = float(row.get('p_bonferroni', 1))
        n     = int(row.get('n_donors', 0))
        ci_lo = row.get('ci_95_lo')
        ci_hi = row.get('ci_95_hi')
        desc  = str(row.get('description', '')).strip()

        set_parts = (
            f"r.spearman_r = {round(r_val, 4)}, "
            f"r.p_value = {round(p, 6)}, "
            f"r.p_bonferroni = {round(p_bon, 6)}, "
            f"r.n_donors = {n}, "
            f"r.braak_scale = 'B_0to3', "
            f"r.method = 'Spearman_bootstrap_n1000', "
            f"r.description = '{esc(desc)}'"
        )
        if ci_lo and not pd.isna(ci_lo):
            set_parts += f", r.ci_95_lo = {round(float(ci_lo), 4)}"
        if ci_hi and not pd.isna(ci_hi):
            set_parts += f", r.ci_95_hi = {round(float(ci_hi), 4)}"

        lines.append(
            f"MATCH (i:Isoform {{transcript_id: '{esc(tid)}'}}) "
            f"MATCH (d:Disease {{mondo_id: '{mondo}'}}) "
            f"MERGE (i)-[r:CORRELATES_WITH_BRAAK]->(d) "
            f"ON CREATE SET {set_parts};"
        )
        lines.append('')
    write_cypher('09_edges_braak.cypher', lines,
                 'Isoform → CORRELATES_WITH_BRAAK → Disease')


# ── 10. CellCluster → Cohort (VALIDATED_IN) ──────────────────────────────────
def build_cohort_validation_edges(cohort_df: pd.DataFrame) -> None:
    if cohort_df.empty:
        return
    lines = [
        '// ─── VALIDATED_IN 관계 (CellCluster → Cohort) ───────────────',
        '// 코호트별 독립 검증: PO(3\'v4) + SMC(3\'v3) 방향 일치 = 배치효과 아님',
        '',
    ]
    for _, row in cohort_df.iterrows():
        lid    = int(row['cluster'])
        cohort = str(row.get('cohort', '')).strip()
        if cohort == 'ALL':
            continue
        delta  = float(row.get('delta', 0))
        p      = float(row.get('p', 1))
        sig    = str(row.get('sig', 'ns')).strip()
        n_ad   = int(row.get('n_AD', 0))
        n_ct   = int(row.get('n_CT', 0))
        dirn   = str(row.get('direction', '')).strip()
        lines.append(
            f"MATCH (c:CellCluster {{leiden_id: {lid}}}), "
            f"(h:Cohort {{name: '{esc(cohort)}'}}) "
            f"MERGE (c)-[r:VALIDATED_IN]->(h) "
            f"ON CREATE SET r.delta_pct = {round(delta, 4)}, "
            f"r.p_value = {round(p, 6)}, "
            f"r.sig = '{sig}', "
            f"r.n_AD = {n_ad}, "
            f"r.n_CT = {n_ct}, "
            f"r.direction = '{esc(dirn)}';"
        )
        lines.append('')
    write_cypher('10_edges_cohort_validation.cypher', lines,
                 'CellCluster → VALIDATED_IN → Cohort')


# ── 11. Donor → CellCluster (HAS_CLUSTER_PCT) ────────────────────────────────
def build_donor_cluster_edges(donor_df: pd.DataFrame) -> None:
    if donor_df.empty:
        return
    lines = [
        '// ─── HAS_CLUSTER_PCT 관계 (Donor → CellCluster) ────────────',
        '// 속성: pct(%), n_cells, condition',
        '',
    ]
    for _, row in donor_df.iterrows():
        donor  = str(row.get('donor', '')).strip()
        lid    = int(row['cluster'])
        pct    = float(row.get('pct', 0))
        n      = int(row.get('n_cells', 0)) if 'n_cells' in row else 0
        cond   = str(row.get('condition', '')).strip()
        lines.append(
            f"MATCH (n:Donor {{donor_id: '{esc(donor)}'}}) "
            f"MATCH (c:CellCluster {{leiden_id: {lid}}}) "
            f"MERGE (n)-[r:HAS_CLUSTER_PCT]->(c) "
            f"ON CREATE SET r.pct = {round(pct, 6)}, "
            f"r.n_cells = {n}, "
            f"r.condition = '{esc(cond)}';"
        )
    lines.append('')
    write_cypher('11_edges_donor_cluster.cypher', lines,
                 'Donor → HAS_CLUSTER_PCT → CellCluster')


# ── 12. Gene → CellCluster (OVEREXPRESSED_IN / UNDEREXPRESSED_IN) — C18 ───────
def build_c18_expression_edges(c18_df: pd.DataFrame) -> None:
    if c18_df.empty:
        return
    lines = [
        '// ─── OVEREXPRESSED_IN / UNDEREXPRESSED_IN (Gene → CellCluster C18) ─',
        '// C18 (leiden_id=18) vs L4 IT 평균 (C10+C11) 비교',
        '// 기준: |log2FC| > 1 (2배 이상 차이)',
        '',
    ]
    for _, row in c18_df.iterrows():
        gene   = str(row.get('gene', '')).strip()
        lfc    = float(row.get('log2FC_C18_vs_C10C11', 0))
        c18    = float(row.get('C18', 0))
        c10c11 = float(row.get('C10_C11_mean', 0))
        if abs(lfc) < 1.0:
            continue
        rel = 'OVEREXPRESSED_IN' if lfc > 0 else 'UNDEREXPRESSED_IN'
        lines.append(
            f"MATCH (g:Gene {{symbol: '{esc(gene)}'}}) "
            f"MATCH (c:CellCluster {{leiden_id: 18}}) "
            f"MERGE (g)-[r:{rel}]->(c) "
            f"ON CREATE SET r.log2FC = {round(lfc, 4)}, "
            f"r.C18_mean_expr = {round(c18, 4)}, "
            f"r.C10C11_mean_expr = {round(c10c11, 4)}, "
            f"r.comparison = 'C18_vs_L4IT_C10C11_mean', "
            f"r.method = 'log2FC_snRNA-seq';"
        )
        lines.append('')
    write_cypher('12_edges_c18_expression.cypher', lines,
                 'Gene → OVEREXPRESSED_IN/UNDEREXPRESSED_IN → CellCluster(C18)')


# ── 13. PRISM 기능 예측 placeholder ──────────────────────────────────────────
def build_prism_placeholder() -> None:
    lines = [
        '// ─── PREDICTS_GO 관계 placeholder (PRISM 출력 연결용) ────────',
        '// 현재 PRISM AUPRC 결과를 Isoform-GOTerm 엣지로 변환하는 구조',
        '// 실제 적재: generate_brain_full_extended_scores.py 출력 연동 필요',
        '',
        '// 예시 GOTerm 노드 (18 BP terms):',
    ]
    go_terms = [
        ('GO:0022900', 'electron transport chain', 'biological_process'),
        ('GO:0006412', 'translation', 'biological_process'),
        ('GO:0006936', 'muscle contraction', 'biological_process'),
        ('GO:0006096', 'glycolysis', 'biological_process'),
        ('GO:0007018', 'microtubule-based movement', 'biological_process'),
        ('GO:0006814', 'sodium ion transport', 'biological_process'),
        ('GO:0007010', 'cytoskeleton organization', 'biological_process'),
        ('GO:0006811', 'ion transport', 'biological_process'),
    ]
    for go_id, name, ns in go_terms:
        lines.append(
            f"MERGE (g:GOTerm {{go_id: '{go_id}'}}) "
            f"ON CREATE SET g.name = '{name}', g.namespace = '{ns}';"
        )
    lines.append('')
    lines.append(
        '// PREDICTS_GO 엣지 적재 예시 (PRISM 점수 연동 후 실행):',
    )
    lines.append(textwrap.dedent("""\
        // MATCH (i:Isoform {transcript_id: $transcript_id})
        // MATCH (go:GOTerm {go_id: $go_id})
        // MERGE (i)-[r:PREDICTS_GO]->(go)
        // ON CREATE SET r.auprc = $auprc,
        //               r.tissue = $tissue,
        //               r.model = 'PRISM_v15d_ESM2',
        //               r.prism_version = 'v15d_bp_clean';
    """))
    write_cypher('13_prism_placeholder.cypher', lines,
                 'GOTerm 노드 + PREDICTS_GO placeholder')


# ── 99. 진단 쿼리 모음 ────────────────────────────────────────────────────────
def build_diagnostic_queries() -> None:
    lines = [
        '// ─── 진단 및 탐색 쿼리 모음 ─────────────────────────────────',
        '// Neo4j Browser에서 개별 실행',
        '',
        '// [1] 전체 노드/엣지 통계',
        'MATCH (n) RETURN labels(n) AS label, count(*) AS cnt ORDER BY cnt DESC;',
        '',
        '// [2] AD-enriched 클러스터 목록 (p<0.05)',
        'MATCH (c:CellCluster)-[r:AD_ENRICHED]->(d:Disease)',
        'WHERE r.p_value < 0.05',
        'RETURN c.leiden_id, c.subtype, c.cell_type, r.delta_pct, r.p_value, r.sig',
        'ORDER BY r.p_value;',
        '',
        '// [3] KIF21B isoform switch 세포 타입별 조회',
        "MATCH (t:CellType)-[r:SHOWS_ISOFORM_SWITCH]->(g:Gene {symbol: 'KIF21B'})",
        'RETURN t.name, r.AD_ratio, r.CT_ratio, r.delta, r.p_value, r.sig',
        'ORDER BY r.p_value;',
        '',
        '// [4] C18 클러스터의 모든 관계 탐색',
        'MATCH (c:CellCluster {leiden_id: 18})-[r]-(x)',
        'RETURN type(r), labels(x), x.symbol, x.name, x.subtype LIMIT 30;',
        '',
        '// [5] Braak 상관 이소폼 (r < -0.3)',
        'MATCH (i:Isoform)-[r:CORRELATES_WITH_BRAAK]->(d:Disease)',
        'WHERE r.spearman_r < -0.3',
        'RETURN i.transcript_id, i.gene_symbol, r.spearman_r, r.p_value,',
        '       r.p_bonferroni, r.description',
        'ORDER BY r.spearman_r;',
        '',
        '// [6] 코호트 방향 일치 검증 (배치효과 확인)',
        'MATCH (c:CellCluster)-[po:VALIDATED_IN]->(h1:Cohort {name: "PO"})',
        'MATCH (c)-[smc:VALIDATED_IN]->(h2:Cohort {name: "SMC"})',
        'RETURN c.leiden_id, c.subtype,',
        '       po.direction AS PO_dir, po.delta_pct AS PO_delta,',
        '       smc.direction AS SMC_dir, smc.delta_pct AS SMC_delta,',
        '       po.direction = smc.direction AS batch_ok',
        'ORDER BY c.leiden_id;',
        '',
        '// [7] 3-hop 경로: LAMP5/KIT → isoform switch → AD',
        "MATCH p = (g:Gene {symbol: 'KIT'})<-[:DEFINED_BY_MARKER]-(c:CellCluster)",
        '         -[:AD_ENRICHED]->(d:Disease)',
        'RETURN p;',
        '',
        '// [8] Complex I 수렴 확인 (NDUFS4/7/8 동시 switch)',
        "MATCH (t:CellType)-[r:SHOWS_ISOFORM_SWITCH]->(g:Gene)",
        "WHERE g.symbol IN ['NDUFS4', 'NDUFS7', 'NDUFS8']",
        'RETURN t.name, g.symbol, r.sig, r.p_value',
        'ORDER BY g.symbol, r.p_value;',
    ]
    write_cypher('99_diagnostic_queries.cypher', lines,
                 '진단 및 탐색 쿼리 — Neo4j Browser에서 실행')


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print('BISECT → Neo4j Cypher 변환 시작')
    print(f'  입력: {DATA}')
    print(f'  출력: {OUT}')
    print()

    # TSV 로드
    def load(name: str) -> pd.DataFrame:
        p = DATA / name
        if p.exists():
            df = pd.read_csv(p, sep='\t')
            print(f'  로드: {name}  ({len(df)} rows)')
            return df
        print(f'  없음: {name}')
        return pd.DataFrame()

    sub_df    = load('final_subtype_classification.tsv')
    ct_df     = load('celltype_isoform_switch.tsv')
    braak_df  = load('braak_correlation_results.tsv')
    cohort_df = load('cohort_batch_results.tsv')
    donor_df  = load('per_donor_pct.tsv')
    layer_df  = load('layer_annotation_results.tsv')
    c18_df    = load('c18_vs_c10c11_diff.tsv')
    print()

    print('Cypher 파일 생성 중...')
    build_constraints()
    build_gene_nodes(sub_df, ct_df, c18_df)
    build_isoform_nodes(braak_df)
    build_structural_nodes(sub_df, layer_df)
    build_donor_nodes(donor_df)
    build_cluster_disease_edges(sub_df)
    build_layer_edges(sub_df, layer_df)
    build_marker_edges(sub_df)
    build_isoform_switch_edges(ct_df)
    build_braak_edges(braak_df)
    build_cohort_validation_edges(cohort_df)
    build_donor_cluster_edges(donor_df)
    build_c18_expression_edges(c18_df)
    build_prism_placeholder()
    build_diagnostic_queries()

    # 적재 순서 안내 README
    readme = OUT / 'README_load_order.md'
    readme.write_text(textwrap.dedent("""\
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
    """), encoding='utf-8')

    print()
    print(f'완료: {OUT} 에 {len(list(OUT.glob("*.cypher")))} 개 Cypher 파일 생성')
    print(f'적재 안내: {readme}')


if __name__ == '__main__':
    main()
