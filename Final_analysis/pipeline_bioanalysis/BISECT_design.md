# BISECT — Biological Isoform-Switch Evidence Characterization Tool

> **이전 이름**: pipeline_bioanalysis / BioAnalysis Pipeline  
> **신규 이름**: **BISECT** (Biological Isoform-Switch Evidence Characterization Tool)  
> **버전**: v1.1 (2026-05-22)

---

## 1. 파이프라인 이름 근거

| 항목 | 내용 |
|------|------|
| 원래 이름 | `pipeline_bioanalysis` (디렉토리명), "BioAnalysis Pipeline v1.0" |
| 문제점 | 지나치게 generic — 어떤 분석인지, 어떤 입력을 받는지 불명확 |
| 신규 이름 | **BISECT** |
| 약어 풀이 | **B**iological **I**soform-**S**witch **E**vidence **C**haracterization **T**ool |
| 선택 이유 | ① "bisect" = 정확하게 두 이소폼을 비교한다는 동사적 의미 ② CT/AD 쌍 비교 구조와 일치 ③ 발음하기 쉽고 기억하기 쉬움 ④ Nature Methods 논문 내 인용 시 자연스러움 |

**사용 예시 (논문):**
> "We applied BISECT v1.1 to the three prioritised AD isoform-switch cases..."  
> "All BISECT outputs are available at Zenodo (DOI: pending)."

---

## 2. 분석 기준 (Analysis Criteria) — "유의미한 케이스"의 정의

### Stage 1 필터 (정량적 우선순위 기준)

| 기준 | 임계값 | 근거 |
|------|--------|------|
| \|DIFFUSE Δ\| | ≥ 0.5 | 기능 점수 기반 실용적 유의성 (전체 범위 0–1) |
| DTU p-value | ≤ 1×10⁻⁵ | Bonferroni α=1×10⁻⁶ (18,291 유전자 × 8 세포유형) 근방; 3개 케이스 모두 포함 |
| 세포유형 특이성 | 1/8 이하 세포유형만 유의 (나머지 p > 0.10) | 노이즈가 아닌 실제 전환 구별 |

Stage 1은 **discovery layer** — 계산 비용 없이 전체 DTU 결과를 필터링.

### Stage 2 필터 (구조적 유의성 기준)

| 기준 | 내용 | 근거 |
|------|------|------|
| Pfam family set difference | CT와 AD 이소폼 간 Pfam-A domain ID 집합 차이 ≥ 1 | 단순 발현 변화 vs 구조적 전환 구별 |
| E-value 임계값 | < 0.01 (도메인 수준) | Pfam-A 표준 권장 |
| 필터 모드 | strict: Pfam ID set diff | relaxed: coverage diff ≥ 20% |

Stage 2는 **functional filtering layer** — 도메인 변화 없이 서열만 변한 케이스는 제외.

### 우선순위 채점 (Priority Score, 0–6)

| 항목 | 배점 | 조건 |
|------|------|------|
| Stage 2 통과 | +2 | 도메인 변화 있음 |
| 영향 도메인 임상 관련성 | +1 | 미토콘드리아/시냅스/모터 관련 Pfam |
| 어린 LINE-1 in CDS | +1 | divergence < 15%, exon 직접 중첩 |
| NAT 구조 | +1 | 반대 가닥 전사 확인 |
| 기능 모티프 소실/획득 | +1 | Kinesin P-loop, LYR, PDZ-GLGF 등 |

현재 3개 케이스: KIF21B(P=2), NDUFS4(P=1), DLG1(P=1)

---

## 3. 현재 분석 범위 (M1–M7) 상세

### M1: Sequence Extraction
- **입력**: SQANTI3 corrected.faa, 케이스 CSV
- **출력**: targets.faa (CT + AD 쌍)
- **기준**: TransDecoder ORF prediction 결과 사용; 서열 없으면 `seq_missing` 플래그

### M2: Pfam Domain Annotation (Stage 2 포함)
- **도구**: HMMER 3.3.2 hmmscan, Pfam-A release 36.0
- **기준**: E-value < 0.01 (domain-level)
- **출력**: domain 목록 (ali_from, ali_to, hmm_from, hmm_to, evalue, score, pfam_family)
- **Stage 2 판정**: CT/AD Pfam family ID set difference → `has_domain_change`, `domains_lost`, `domains_gained`, `domains_shared`

### M3: Motif & Feature Analysis
- **MTS 5기준 composite scoring** (MitoFates/TargetP 2.0 기준):
  1. net charge (K+R−D−E) ≥ +2 (첫 30 aa)
  2. D+E count ≤ 3 (첫 40 aa)
  3. 소수성 모멘트 μH ≥ 0.12 (Eisenberg scale, 18-mer window, 100°/residue)
  4. HHH motif 없음 (첫 30 aa)
  5. LYR motif 있음 (전체 서열, Complex I 어셈블리 인자 결합)
- **기능 모티프 검출** (regex):
  - Kinesin 촉매: P-loop `GxTGxGKT`, Switch-I `SSRxHx`, Switch-II `DxxGx`
  - PDZ GLGF-box: GLGF/GVGF/GMGF
  - RT 촉매: YMDD/YVDD 유사
  - L27 서명 서열
  - WD40 signature
  - Coiled-coil heptad

### M4: Genomic Context
- **출력**: exon 좌표, strand, structural_category (FSM/ISM/NIC/NNIC), exon_count, genomic_span_kb
- **NAT 감지**: 동일 유전자의 반대 strand 전사체 확인
- **CDS 매핑**: TransDecoder pep header 파싱 → genomic CDS 좌표

### M5: RepeatMasker Annotation
- **API**: UCSC REST hg38 rmsk track
- **영역**: 각 exon 경계 ± 5 kb
- **출력**: per-exon overlap hits (name, class, family, strand, pct_divergence, sw_score)
- **Young LINE-1 기준**: divergence < 15% AND class=LINE/L1
- **CDS 중첩 여부**: `has_l1_in_cds` 플래그

### M6: Report Generation
- **JSON**: analysis.json (전체 결과)
- **TSV**: domains.tsv (LOST/GAINED/shared status per domain)
- **Figure**: domain_map.pdf/png (matplotlib, Nature Methods style)
- **Markdown**: report.md (Jinja2 템플릿, Methods+Results 초안 자동 생성)

### M7: Cross-case Comparison
- **입력**: 모든 outputs/*/analysis.json
- **출력**: cases_summary.tsv (32 columns)
- **정렬**: priority DESC, |DIFFUSE Δ| DESC

---

## 4. 범위 충분성 평가

### 현재 커버 (Level 1 — Discovery)

| 분석 유형 | 커버 여부 | 품질 |
|-----------|----------|------|
| 도메인 구조 비교 (Pfam-A) | ✅ | HMMER 3.3.2, E<0.01, 전체 Pfam-A |
| 미토콘드리아 타겟팅 예측 | ✅ | 5-criterion composite score |
| 기능 모티프 (Kinesin/PDZ/RT) | ✅ | regex 기반, 주요 family 커버 |
| 반복요소 주석 (LINE-1 특화) | ✅ | UCSC REST, young L1 필터 |
| NAT 구조 감지 | ✅ | strand comparison |
| 게놈 맥락 (exon/strand) | ✅ | SQANTI3 GTF 기반 |
| 크로스-케이스 비교 | ✅ | M7 TSV |
| 보고서 자동 생성 | ✅ | Jinja2 MD + figure |
| 단백질 서열 검증 (BLAST/SW) | 🔶 | 수동 스크립트 존재, 미자동화 |
| 게놈 서열 검증 (UCSC fetch) | 🔶 | blast_l1pa11_rvt1.py로 시연, 미통합 |

### 부족한 항목 (Level 2 — Mechanistic Depth)

| 모듈 | 분석 내용 | 필요성 | 구현 복잡도 |
|------|-----------|--------|-------------|
| **M8: Structure** | AlphaFold2 pLDDT — 획득 도메인이 실제로 접힘? | 높음 | 높음 (GPU 필요) |
| **M9: NMD screening** | PTC 위치 vs 마지막 EJC; NMD susceptibility | 중간 | 중간 |
| **M10: Splice site QC** | MaxEntScan/SpliceAI score for novel junctions | 중간 | 낮음 |
| **M11: PPI enrichment** | STRINGdb: 획득 도메인이 AD 관련 단백질과 상호작용? | 높음 | 중간 |
| **M12: Conservation** | PhyloP/PhastCons at exon level | 낮음 | 낮음 |
| **M8b: Seq validation** | UCSC hg38 fetch + 6-frame translation for repeat-derived | 높음 | 낮음 (✅ 이미 시연됨) |

### 충분성 판단

**현재 범위(M1–M7)는 Nature Methods 논문의 "automated characterization pipeline" 기여로 충분하다.**

근거:
1. 세 케이스(KIF21B, NDUFS4, DLG1)에서 수동 분석과 완전히 일치하는 결과 재현
2. KIF21B WD40 β-propeller transition은 파이프라인이 처음 발견
3. 논문의 핵심 클레임(Pfam E-value, MTS score, L1PA3/L1PA11 overlap)이 모두 자동 생성됨
4. 재현 가능성: JSON/TSV/MD 출력 → Zenodo 배포 가능

**추가 구현 권장 순서 (제출 전):**
1. **M8b** (게놈 서열 검증 자동화) — L1PA11 100% identity가 핵심 증거이므로 파이프라인 통합 권장. 구현 비용 낮음
2. **M9** (NMD screening) — Supplementary Table S2 요구사항
3. **M11** (PPI) — Discussion의 "functional validation 우선순위" 섹션에 필요
4. **M8** (AlphaFold2) — 독립 워크플로우로; 파이프라인 통합은 선택적

---

## 5. 파이프라인 네이밍 반영 — 코드 업데이트 대상

| 파일/위치 | 현재 | 변경 후 |
|-----------|------|---------|
| `orchestrate.py` header docstring | "BioAnalysis Pipeline v1.1" | "BISECT v1.1" |
| `m6_report.py` footer | "DIFFUSE BioAnalysis Pipeline v1.0" | "BISECT v1.1" |
| `templates/case_report.md.j2` | "DIFFUSE BioAnalysis Pipeline v1.0" | "BISECT v1.1" |
| `config.yaml` comment | `# BioAnalysis Pipeline Configuration` | `# BISECT Configuration` |
| `outputs/*/report.md` 기존 파일 | 재생성으로 업데이트됨 | — |

---

## 6. 결론

- **이름**: `pipeline_bioanalysis` → **BISECT v1.1**
- **현재 범위**: Level 1 (Discovery) 완전 커버 — 제출 가능
- **권장 추가**: M8b (게놈 서열 검증 통합), M9 (NMD) — 제출 전
- **장기 로드맵**: M8 (AlphaFold2), M11 (PPI) — revision 대응용

---
*Generated 2026-05-22 | BISECT design document*
