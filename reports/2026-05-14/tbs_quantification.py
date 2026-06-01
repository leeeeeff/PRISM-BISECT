#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tbs_quantification.py
Taxonomic Breadth Score (TBS) 정량화

정의:
    TBS(GO) = |{k ∈ K : ∃x ∈ SP+(GO) s.t. taxon(x) ∈ k}| / |K|
    K = {Bacteria, Archaea, Fungi, Viridiplantae, Invertebrata, Vertebrata}

입력: swissprot_annotations.txt (GENE_SPECIES\tGO:...\tGO:... 형식)
출력: tbs_results.json, tbs_fti_scatter.png
"""

import os, json, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 경로 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ANNOT_SP     = os.path.join(SCRIPT_DIR,
    '../../hMuscle/data/raw_data/data/annotations/swissprot_annotations.txt')
ANNOT_HUMAN  = os.path.join(SCRIPT_DIR,
    '../../hMuscle/data/raw_data/data/annotations/human_annotations.txt')
OUT_DIR      = SCRIPT_DIR

# ── 분석 대상 GO terms ────────────────────────────────────────────────────────
GO_TERMS = ['GO:0007204', 'GO:0030017', 'GO:0006941', 'GO:0003774', 'GO:0006096']
GO_LABELS = {
    'GO:0007204': 'Ca²⁺ signaling\n(GO:0007204)',
    'GO:0030017': 'Sarcomere org.\n(GO:0030017)',
    'GO:0006941': 'Striated contraction\n(GO:0006941)',
    'GO:0003774': 'Myosin motor\n(GO:0003774)',
    'GO:0006096': 'Glycolysis\n(GO:0006096)',
}

# 실험으로 측정된 FTI (256d 기준: D256 / P3-256)
FTI_256 = {
    'GO:0007204': 0.1947 / 0.3109,   # 0.626
    'GO:0030017': 0.1366 / 0.2836,   # 0.482
    'GO:0006941': 0.1567 / 0.1540,   # 1.018
    'GO:0003774': 0.5982 / 0.5830,   # 1.026
    'GO:0006096': 0.8331 / 0.4445,   # 1.875
}
# FTI (64d 기준: v8b / P3)
FTI_64 = {
    'GO:0007204': 0.1462 / 0.1624,   # 0.900
    'GO:0030017': 0.1570 / 0.1918,   # 0.819
    'GO:0006941': 0.1177 / 0.1292,   # 0.911
    'GO:0003774': 0.5686 / 0.5405,   # 1.052
    'GO:0006096': 0.7945 / 0.4640,   # 1.712
}

# ── Kingdom 분류표 ──────────────────────────────────────────────────────────
# UniProt entry name 형식: GENE_SPECIES (species = 5자리 코드)
KINGDOMS = ['Bacteria', 'Archaea', 'Fungi', 'Viridiplantae', 'Invertebrata', 'Vertebrata']

SPECIES_TO_KINGDOM = {}

# Vertebrata (비인간 척추동물)
for sp in [
    'MOUSE', 'RAT', 'BOVIN', 'PIG', 'SHEEP', 'HORSE', 'FELCA', 'CANFA', 'RABIT',
    'CHICK', 'XENLA', 'XENTR', 'DANRE', 'ORYLA', 'TAKRU', 'FUGRU', 'MEDJA',
    'MACMU', 'PANTR', 'GORGO', 'PONAB', 'MACFA', 'CHLAE', 'PAPAN', 'NOMLE',
    'AILME', 'LOXAF', 'TURTR', 'MYOLU', 'ICTTR', 'ECHTE', 'SOREX', 'TUPGB',
    'CAVPO', 'MESAU', 'CRIGR', 'SPECI', 'SPETR', 'OCHPR', 'DIPOR', 'PEDPE',
    'HETGL', 'NANPA', 'MICOC', 'MICMU', 'OTOGA', 'TARSY', 'CALJA', 'AOTNA',
    'SAGOE', 'COLMO', 'ATEGE', 'LAGGL', 'PROCO', 'PANHA', 'RHIFE', 'CERDI',
    'MANDO', 'SARHA', 'MONDO', 'ORNAN', 'CRILO', 'ONOVI', 'SALSA', 'ICTPU',
    'PETMA', 'LAMGE', 'HUMAN',  # HUMAN은 여기도 포함 (swissprot에 있을 경우)
    'SHEEP', 'BUFBU', 'ANAPL', 'HALAL', 'MELGA', 'TAEGU', 'GEFOR', 'ACRSC',
    'OPHHA', 'PELSI', 'ANOCA', 'CROAM', 'ALIMA', 'TRICA',  # 일부 파충류/양서류
]:
    SPECIES_TO_KINGDOM[sp] = 'Vertebrata'

# Invertebrata
for sp in [
    'DROME', 'DROMZ', 'DROPS', 'DROYA', 'DROVI', 'DROPE', 'DROAN', 'DROMI',
    'CAEEL', 'CAEBR', 'CAERE', 'HAECO', 'PRIPA',
    'APLCA', 'APLKW', 'LOLFO',
    'OCTVU', 'OCTBM',
    'STRPU', 'ARCPU',
    'CIOIN', 'CIOSA',
    'SCHMA', 'FASHE', 'PLAF7', 'PLAF4', 'PLACH', 'PLAAF',
    'LOTGI', 'CRAGI', 'AMQC7',
    'HELRO', 'CAPTE',
    'TRICA', 'BOMMO', 'MANDO', 'AEDAE', 'ANOGA', 'CULQU', 'PHYCI',
    'APIS', 'APIME', 'NASCO', 'ACYPI', 'IXOSC', 'RHIMP',
    'PENMO', 'DAPPU', 'ARTSU',
    'HYDVU', 'NEMVE',
    'LUBLU',
]:
    SPECIES_TO_KINGDOM[sp] = 'Invertebrata'

# Fungi
for sp in [
    'YEAST', 'YEAS7', 'YEAS8',
    'SCHPO',
    'CANAL', 'CANPA', 'CANGA',
    'NEUCR', 'PODE3', 'PODAZ',
    'ASPFU', 'ASPNI', 'ASPOR', 'ASPFL', 'ASPCL', 'ASPA1', 'ASPAC',
    'PENCH', 'PENRW', 'PENOX', 'PENBR', 'PENBI', 'PENDC',
    'CRYNE', 'CRYNJ', 'CRYNH',
    'USTMA', 'MYCMD', 'PHANO',
    'COPCI', 'LACBI', 'AGABI',
    'MAGNP', 'MAGO7', 'GIBZE', 'FUSGR', 'FUSOX',
    'TALATR', 'TALMT',
    'COCIM', 'HISCR', 'PARBA',
    'BOTFU', 'SCLS1', 'VEDALB',
    'RHISO', 'RHIA1', 'PHYBL', 'ENTHI', 'BATDE',
]:
    SPECIES_TO_KINGDOM[sp] = 'Fungi'

# Viridiplantae (Plants)
for sp in [
    'ARATH', 'ARATH',
    'ORYSJ', 'ORYSI', 'ORYBR', 'ORYOF',
    'MAIZE', 'MAIZB',
    'SOLTU', 'SOLLC', 'SOLPE',
    'TOBAC', 'NICTA', 'NICAT', 'NICSY',
    'SOYBN', 'GLYMA',
    'MEDTR', 'LOTJA', 'PHAVU', 'VICFA', 'CICAR', 'CAJCA',
    'POPTR', 'RICCO', 'JATCU', 'MANES', 'LUSAN',
    'VITVI', 'GRAPE',
    'SPIOL', 'BETVU', 'CHEAM', 'ATHAL',
    'WHEAT', 'ORYSA', 'HORVV', 'SECCE', 'TRIUA',
    'BRAOL', 'BRACM', 'BRAOL', 'BORPA', 'CAPAR',
    'PHYPA', 'SELMO', 'CHLRE', 'VOLVP', 'OSTTA', 'MICPU',
    'MARPO', 'CERPU',
    'PINSY', 'ABIES', 'PICAB',
]:
    SPECIES_TO_KINGDOM[sp] = 'Viridiplantae'

# Bacteria
for sp in [
    'ECOLI', 'ECOLX', 'ECO57',
    'BACSU', 'BACST', 'BACME', 'BACAN', 'BACCE', 'BACCR', 'BACHD',
    'STAAU', 'STAAN', 'STAEQ', 'STAEP', 'STAHY',
    'STRCO', 'STRPU', 'STRPR', 'STRSV', 'STRMU', 'STRPN', 'STRPY',
    'MYCTU', 'MYCBO', 'MYCUL', 'MYCS2',
    'SALTY', 'SALTI', 'SALPA', 'SALDC',
    'PSEAE', 'PSEPF', 'PSESM',
    'HAEIN', 'HAES1',
    'NEIGO', 'NEIMB',
    'HELPJ', 'HELPY', 'HELPH',
    'CAMC5', 'CAMJE', 'CAMCOL',
    'THEMA', 'THEM4', 'THETH',
    'AQUAE',
    'DEIRA', 'DEIRR',
    'CHLTR', 'CHLT2',
    'BORBU', 'BORGA', 'BORHE',
    'TREDE', 'TREPA',
    'RICPR', 'RICAE',
    'LACPL', 'LACLA', 'LACP3',
    'CLOPE', 'CLOAB', 'CLOBO', 'CLOTE', 'CLOD6',
    'BIFLO', 'BIFAD',
    'RHIME', 'RHILO', 'AGRRT', 'MESSA', 'BRASO', 'SINFN',
    'CAUVC', 'BDEBA',
    'SHEON', 'SHEAM', 'SHEFN',
    'VIBCH', 'VIBVU', 'VIBPA',
    'YERPE', 'YERPS', 'YEREN',
]:
    SPECIES_TO_KINGDOM[sp] = 'Bacteria'

# Archaea
for sp in [
    'METJA', 'METTH', 'METKA', 'METAC', 'METMA', 'METBU', 'METS5',
    'PYRAE', 'PYRFU', 'PYRAB', 'PYRHO',
    'SULSO', 'SULAC', 'SULTO', 'SULSH', 'SULNO', 'SULT1',
    'HALOAR', 'HALSA', 'HALVD', 'HALMD', 'HALMS',
    'ARCFU',
    'THEAC', 'THEVO', 'PICTO',
    'NAEGR', 'IGNH4',
    'METAR', 'METHJ',
    'AERPE', 'CALTE',
    'NANEQ',
    'CORMM', 'NITMS',
]:
    SPECIES_TO_KINGDOM[sp] = 'Archaea'


def parse_annotations(filepath):
    """annotation 파일을 {protein: set(GO terms)} 딕셔너리로 파싱."""
    prot_to_go = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            prot = parts[0]
            go_terms = set(parts[1:])
            prot_to_go[prot] = go_terms
    return prot_to_go


def get_kingdom(species_code):
    """5자리 species 코드를 kingdom으로 변환. 미등록시 'Unknown'."""
    return SPECIES_TO_KINGDOM.get(species_code.upper(), 'Unknown')


def compute_tbs(go_term, sp_prot_to_go):
    """GO term에 대한 TBS를 계산."""
    kingdoms_present = set()
    sp_positives = []

    for prot, go_set in sp_prot_to_go.items():
        if go_term in go_set:
            sp_positives.append(prot)
            parts = prot.split('_')
            if len(parts) >= 2:
                species = parts[-1]
                kingdom = get_kingdom(species)
                if kingdom != 'Unknown':
                    kingdoms_present.add(kingdom)

    tbs = len(kingdoms_present) / len(KINGDOMS)
    return tbs, kingdoms_present, sp_positives


def compute_human_positives(go_term, human_prot_to_go):
    """Human positive 단백질 목록."""
    return [p for p, go_set in human_prot_to_go.items() if go_term in go_set]


def main():
    print("=" * 60)
    print(" TBS Quantification")
    print("=" * 60)

    print("\n[1] Parsing annotation files...")
    sp_prot_to_go    = parse_annotations(ANNOT_SP)
    human_prot_to_go = parse_annotations(ANNOT_HUMAN)
    print(f"  SwissProt proteins: {len(sp_prot_to_go):,}")
    print(f"  Human proteins:     {len(human_prot_to_go):,}")

    print("\n[2] Computing TBS per GO term...")
    results = {}
    for go in GO_TERMS:
        tbs, kingdoms, sp_pos = compute_tbs(go, sp_prot_to_go)
        human_pos = compute_human_positives(go, human_prot_to_go)

        # kingdom별 단백질 수 집계
        kingdom_counts = collections.Counter()
        for prot in sp_pos:
            parts = prot.split('_')
            if len(parts) >= 2:
                k = get_kingdom(parts[-1])
                if k != 'Unknown':
                    kingdom_counts[k] += 1

        # Unknown 비율 확인
        unknown_prots = [p for p in sp_pos
                         if get_kingdom(p.split('_')[-1]) == 'Unknown']

        results[go] = {
            'tbs': tbs,
            'kingdoms_present': sorted(kingdoms),
            'n_sp_positive': len(sp_pos),
            'n_human_positive': len(human_pos),
            'sp_dependency': len(sp_pos) / (len(sp_pos) + len(human_pos)) if (len(sp_pos) + len(human_pos)) > 0 else 0,
            'kingdom_counts': dict(kingdom_counts),
            'n_unknown_species': len(unknown_prots),
            'fti_64':  FTI_64.get(go),
            'fti_256': FTI_256.get(go),
        }

        print(f"\n  {go}:")
        print(f"    TBS = {tbs:.3f}  ({len(kingdoms)}/{len(KINGDOMS)} kingdoms)")
        print(f"    Kingdoms: {sorted(kingdoms)}")
        print(f"    SP+: {len(sp_pos):3d}  Human+: {len(human_pos):3d}  SP-dep: {results[go]['sp_dependency']:.1%}")
        print(f"    Kingdom breakdown: {dict(sorted(kingdom_counts.items()))}")
        print(f"    Unknown species (unmapped): {len(unknown_prots)} proteins")
        print(f"    FTI_64={FTI_64.get(go, '?'):.3f}  FTI_256={FTI_256.get(go, '?'):.3f}")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, 'tbs_results.json')
    # JSON serializable 변환
    results_json = {k: {kk: (list(vv) if isinstance(vv, set) else vv)
                        for kk, vv in v.items()}
                    for k, v in results.items()}
    with open(out_json, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[3] Saved: {out_json}")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    print("\n[4] Generating figures...")

    tbs_vals   = [results[g]['tbs'] for g in GO_TERMS]
    fti64_vals = [results[g]['fti_64'] for g in GO_TERMS]
    fti256_vals= [results[g]['fti_256'] for g in GO_TERMS]
    n_human    = [results[g]['n_human_positive'] for g in GO_TERMS]
    labels     = [GO_LABELS[g] for g in GO_TERMS]

    # 색상: FTI 기반 (FTI>1 파랑, FTI<1 빨강)
    colors = ['#2166ac' if f > 1 else '#d6604d' for f in fti256_vals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Taxonomic Breadth Score (TBS) vs Functional Transferability Index (FTI)',
                 fontsize=13, fontweight='bold')

    # ── Plot 1: TBS vs FTI_256 ─────────────────────────────────────────────
    ax = axes[0]
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='FTI=1 (neutral)')

    scatter = ax.scatter(tbs_vals, fti256_vals,
                         s=[n*1.5 for n in n_human],
                         c=colors, alpha=0.85, edgecolors='black', linewidth=1.2,
                         zorder=5)

    for i, go in enumerate(GO_TERMS):
        ax.annotate(labels[i],
                    xy=(tbs_vals[i], fti256_vals[i]),
                    xytext=(8, 4), textcoords='offset points',
                    fontsize=8, ha='left')

    # Tier 영역 배경
    ax.axhspan(1.0, 2.1, alpha=0.06, color='blue', label='Tier 1 (SP beneficial)')
    ax.axhspan(0.0, 1.0, alpha=0.06, color='red', label='Tier 3 (SP harmful)')

    ax.set_xlabel('Taxonomic Breadth Score (TBS)\n0 = narrow (vertebrate only)  →  1 = pan-kingdom', fontsize=10)
    ax.set_ylabel('Functional Transferability Index (FTI)\nAUPRC(SP✓) / AUPRC(SP✗)  at 256d', fontsize=10)
    ax.set_title('TBS vs FTI (256d)', fontsize=11)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(0.3, 2.1)

    # 크기 범례
    for n_ex, label_ex in [(50, 'n=50'), (150, 'n=150'), (250, 'n=250')]:
        ax.scatter([], [], s=n_ex*1.5, c='gray', alpha=0.5,
                   label=f'Human+ {label_ex}')
    ax.legend(fontsize=8, loc='upper left')

    # ── Plot 2: TBS vs FTI (64d vs 256d 비교) ─────────────────────────────
    ax2 = axes[1]
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    for i, go in enumerate(GO_TERMS):
        ax2.annotate('', xy=(tbs_vals[i], fti256_vals[i]),
                     xytext=(tbs_vals[i], fti64_vals[i]),
                     arrowprops=dict(arrowstyle='->', color=colors[i],
                                     lw=1.5, alpha=0.7))
        ax2.scatter(tbs_vals[i], fti64_vals[i], s=80,
                    c='white', edgecolors=colors[i], linewidth=2, zorder=5)
        ax2.scatter(tbs_vals[i], fti256_vals[i], s=120,
                    c=colors[i], edgecolors='black', linewidth=1, zorder=6)
        ax2.annotate(go.split(':')[1], xy=(tbs_vals[i], fti256_vals[i]),
                     xytext=(5, 3), textcoords='offset points', fontsize=7)

    ax2.set_xlabel('Taxonomic Breadth Score (TBS)', fontsize=10)
    ax2.set_ylabel('FTI', fontsize=10)
    ax2.set_title('TBS vs FTI: dim 64d→256d amplification\n(open=64d, filled=256d, arrow=direction)', fontsize=10)
    ax2.set_xlim(-0.05, 1.15)

    # 범례
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='gray', markersize=9, label='FTI at 64d'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=11, label='FTI at 256d'),
        mpatches.Patch(facecolor='#2166ac', alpha=0.7, label='SP beneficial (FTI>1)'),
        mpatches.Patch(facecolor='#d6604d', alpha=0.7, label='SP harmful (FTI<1)'),
    ]
    ax2.legend(handles=legend_elements, fontsize=8, loc='upper left')

    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, 'tbs_fti_scatter.png')
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_fig}")

    # ── Kingdom Heatmap ────────────────────────────────────────────────────
    fig2, ax3 = plt.subplots(figsize=(10, 4))
    kingdom_matrix = np.zeros((len(GO_TERMS), len(KINGDOMS)))
    for i, go in enumerate(GO_TERMS):
        for j, k in enumerate(KINGDOMS):
            kingdom_matrix[i, j] = results[go]['kingdom_counts'].get(k, 0)

    # 로그 스케일 표시
    import matplotlib.colors as mcolors
    norm = mcolors.LogNorm(vmin=1, vmax=kingdom_matrix.max() + 1)
    im = ax3.imshow(kingdom_matrix, cmap='Blues', aspect='auto',
                    norm=norm)

    ax3.set_xticks(range(len(KINGDOMS)))
    ax3.set_xticklabels(KINGDOMS, rotation=30, ha='right', fontsize=9)
    ax3.set_yticks(range(len(GO_TERMS)))
    ax3.set_yticklabels([f"{g}\n(TBS={results[g]['tbs']:.2f}, FTI={results[g]['fti_256']:.2f})"
                         for g in GO_TERMS], fontsize=8)
    ax3.set_title('SwissProt Positive Count by Kingdom × GO term\n(log scale)', fontsize=11)
    plt.colorbar(im, ax=ax3, label='# proteins (log scale)')

    for i in range(len(GO_TERMS)):
        for j in range(len(KINGDOMS)):
            val = int(kingdom_matrix[i, j])
            if val > 0:
                ax3.text(j, i, str(val), ha='center', va='center',
                         fontsize=8, color='white' if val > 50 else 'black')
            else:
                ax3.text(j, i, '0', ha='center', va='center',
                         fontsize=8, color='lightgray')

    plt.tight_layout()
    out_fig2 = os.path.join(OUT_DIR, 'tbs_kingdom_heatmap.png')
    plt.savefig(out_fig2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_fig2}")

    # ── 요약 출력 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" TBS 결과 요약")
    print("=" * 60)
    print(f"{'GO term':<15} {'TBS':>6} {'Kingdoms':>8} {'SP+':>6} {'Human+':>7} {'FTI_64':>7} {'FTI_256':>8}")
    print("-" * 60)
    for go in GO_TERMS:
        r = results[go]
        print(f"{go:<15} {r['tbs']:>6.3f} {len(r['kingdoms_present']):>8} "
              f"{r['n_sp_positive']:>6} {r['n_human_positive']:>7} "
              f"{r['fti_64']:>7.3f} {r['fti_256']:>8.3f}")

    # TBS vs FTI 상관 확인
    tbs_arr = np.array(tbs_vals)
    fti_arr = np.array(fti256_vals)
    corr = np.corrcoef(tbs_arr, fti_arr)[0, 1]
    print(f"\n  TBS vs FTI_256 Pearson r = {corr:.3f}")
    print(f"  (주목: GO:0007204가 high TBS에도 low FTI → 1D 모델 반례)")
    print(f"\n  → TCS 추가 시 GO:0007204 위치가 TCS 높음으로 설명됨.")


if __name__ == '__main__':
    main()
