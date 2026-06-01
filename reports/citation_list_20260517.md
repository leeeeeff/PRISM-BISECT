# Citation List — DIFFUSE Paper
**Draft 2026-05-17 (updated 2026-05-17 evening)**

---

## Main Text Citations (확인 필요 순서)

### Biological Context
1. **Cruz-Jentoft et al. (2019)** — Sarcopenia definition (EWGSOP2 European consensus)
   - Cruz-Jentoft AJ, et al. Sarcopenia: revised European consensus on definition and diagnosis.
     *Age Ageing* 2019;48(1):16-31. PMID: 30312372 ✓
   - 사용처: Introduction Para 1 (sarcopenia burden, 임상적 정의)
   - **수정 완료**: Introduction에서 잘못된 *Lancet* PMID:31282166 → *Age Ageing* PMID:30312372로 수정

1b. **Beaudart et al. (2017)** — Sarcopenia mortality/falls risk
    - Beaudart C, et al. Health Outcomes of Sarcopenia: A Systematic Review and Meta-Analysis.
      *J Bone Miner Res* 2017 [JBMR equivalent]; PMID: 27377766
    - ⚠️ 저널명 확인 필요: Introduction에서 "*JBMR*"로 기재 — PubMed에서 실제 저널 확인 필요
    - 사용처: Introduction Para 1 (falls risk 2×, nursing home 3×, mortality 2.3-fold)

2. **Koenig & Kunkel (1990)** — DMD isoform biology
   - Koenig M, Kunkel LM. Detailed analysis of the repeat domain of dystrophin reveals
     four potential hinge segments that may confer flexibility.
     *J Biol Chem* 1990;265(8):4560-6. PMID: 2037986
   - 사용처: Results 3.4 (DMD Panel A; Dp427m DYS repeat structure)

3. **Aerts et al. (2015)** — PINK1 isoform
   - Aerts L, et al. PINK1 kinase catalytic activity is regulated by phosphorylation on
     serines 228 and 402.
     *J Biol Chem* 2015;290(5):2798-811. PMID: 25505270
   - 사용처: Results 3.4 (PINK1 Panel B; 66-kDa vs 55-kDa isoform)

4. **Saada et al. (2012)** — NDUFAF6 function
   - Saada A, et al. Mutations in NDUFAF3 (C3orf60), encoding an orphan mitochondrial complex
     I assembly protein, cause fatal neonatal mitochondrial disease.
     *Am J Hum Genet* 2009;84(6):718-27. PMID: 19463983
   - **주의**: NDUFAF6 (not NDUFAF3) — need to find correct NDUFAF6 reference
   - Better: Guerrero-Castillo S, et al. The Assembly Pathway of Mitochondrial Respiratory
     Chain Complex I. *Cell Metab* 2017;25(1):128-139. PMID: 28094012
   - 사용처: Results 3.4 (NDUFAF6 Panel C, Complex I assembly)

### Machine Learning / Methods
5. **Lin et al. (2023)** — ESM-2 protein language model
   - Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a
     language model. *Science* 2023;379(6637):1123-1130. PMID: 36927031
   - 사용처: Methods 4.2 (ESM-2 embeddings; 실제 사용 모델: esm2_t30_150M_UR50D, 640d, 150M params)
   - **주의**: Methods에서 t33_650M으로 잘못 기재되어 있었음 → t30_150M으로 수정 완료

6. **Haas et al. (2013)** — TransDecoder
   - Haas BJ, et al. De novo transcript sequence reconstruction from RNA-seq using the
     Trinity platform for reference generation and analysis.
     *Nat Protoc* 2013;8(8):1494-512. PMID: 23845962
   - 사용처: Methods 4.2, 4.9 (ORF prediction, NMD screening)

7. **Rives et al. (2021)** — ESM-1 (ESM family origin; may use Lin 2023 for ESM-2 directly)
   - Rives A, et al. Biological structure and function emerge from scaling unsupervised
     learning to 250 million protein sequences.
     *PNAS* 2021;118(15):e2016239118. PMID: 33876751
   - 사용처: Methods 4.2 (ESM model family background)

8. **Gal & Ghahramani (2016)** — Dropout as Bayesian approximation (optional)
   - Gal Y, Ghahramani Z. Dropout as a Bayesian Approximation.
     *ICML* 2016. arxiv:1506.02142
   - 사용처: Methods 4.3 (Dropout 사용 근거; optional)

9. **Lin et al. (2017)** — Focal Loss
   - Lin TY, et al. Focal Loss for Dense Object Detection.
     *ICCV* 2017. PMID: N/A (arxiv:1708.02002)
   - 사용처: Methods 4.4 (BinaryFocalCrossentropy 근거)

9b. **Hermans et al. (2017)** — Triplet Loss (In Defense)
    - Hermans A, Beyer L, Leibe B. In Defense of the Triplet Loss for Person Re-Identification.
      *arXiv* 2017. arXiv:1703.07737 [cs.CV]
    - 사용처: Methods 4.4 (Triplet loss; L(a,p,n) = max(d(a,p)-d(a,n)+margin, 0))
    - **주의**: 저널 미발표 (arXiv only). Nature Methods 스타일 인용: "Hermans et al., 2017 (arXiv:1703.07737)"

### Biological Context — Additional Introduction Para 2-3 References
(These appear in Introduction but were not in original citation list)

A1. **Lexell et al. (1988)** — Type II fibre atrophy in ageing
    - Lexell J, et al. What is the cause of the ageing atrophy? Total number, size and
      proportion of different fiber types studied in whole vastus lateralis muscle from
      15- to 83-year-old men.
      *J Neurol Sci* 1988;84(2-3):275-94. PMID: 3258390
    - 사용처: Introduction Para 2 (type II fast-twitch atrophy)

A2. **Hepple & Rice (2016)** — Mitochondrial biogenesis in sarcopenia
    - Hepple RT, Rice CL. Innervation and neuromuscular control in ageing skeletal muscle.
      *J Physiol* 2016;594(8):1965-78. PMID: 26040455
    - ⚠️ 제목 확인 필요: Introduction에서 "mitochondrial biogenesis and oxidative stress"로 인용 —
      실제 논문이 innervation 관련이면 다른 ref 필요. PubMed 확인 필수.
    - 사용처: Introduction Para 2 (impaired mitochondrial biogenesis)

A3. **Masiero et al. (2009)** — UPS/autophagy in sarcopenia
    - Masiero E, et al. Autophagy is required to maintain muscle mass.
      *Cell Metab* 2009;10(6):507-15. PMID: 19818709
    - 사용처: Introduction Para 2 (ubiquitin-proteasome + autophagy-lysosome pathways)

A4. **Sousa-Victor et al. (2014)** — Satellite cell regenerative capacity
    - Sousa-Victor P, et al. Geriatric muscle stem cells switch reversible quiescence into
      senescence. *Nature* 2014;506(7488):316-21. PMID: 24590016
    - 사용처: Introduction Para 2 (satellite cell regenerative capacity)

A5. **Sandri (2013)** — mTOR/autophagy signalling
    - Sandri M. Protein breakdown in muscle wasting: role of autophagy-lysosome and ubiquitin-
      proteasome. *Int J Biochem Cell Biol* 2013;45(10):2121-9. PMID: 23599891
    - ⚠️ Introduction 기재: *Physiol Rev*, 2013; PMID: 23899566 — 실제 저널 확인 필요
    - 사용처: Introduction Para 2 (mTOR balance anabolic/catabolic)

A6. **Pan et al. (2008)** — 95% of multi-exon genes spliced
    - Pan Q, et al. Deep surveying of alternative splicing complexity in the human transcriptome
      by high-throughput sequencing. *Nat Genet* 2008;40(12):1413-5. PMID: 18978772
    - 사용처: Introduction Para 3 (>95% multi-exon genes produce ≥2 isoforms)

A7. **Baralle & Giudice (2017)** — Tissue-specific alternative splicing
    - Baralle FE, Giudice J. Alternative splicing as a regulator of development and tissue
      identity. *Nat Rev Mol Cell Biol* 2017;18(7):437-451. PMID: 28792009
    - 사용처: Introduction Para 3 (tissue-specific isoform usage)

A8. **Guo et al. (2021)** — Muscle channelopathies / isoform switches
    - Guo W, et al. ... *Nat Commun* 2021. PMID: 34429420
    - ⚠️ 실제 제목 확인 필요 (Introduction "Guo et al., Nat Commun, 2021; PMID: 34429420")
    - 사용처: Introduction Para 3 (DMD/titin/TPM1 isoform switches → channelopathies)

A9. **Hao et al. (2024)** — Long-read single-cell RNA-seq
    - Hao Y, et al. *Nature* 2024. PMID: 38114474
    - ⚠️ 실제 제목 확인 필요
    - 사용처: Introduction Para 3 (long-read scRNA-seq enabling isoform cataloguing)

---

### Isoform-Aware Prior Methods (Introduction)
10. **Vitting-Seerup & Sandelin (2019)** — IsoformSwitchAnalyzeR
    - Vitting-Seerup K, Sandelin A. IsoformSwitchAnalyzeR: analysis of changes in
      genome-wide patterns of alternative splicing and its functional consequences.
      *Bioinformatics* 2019;35(21):4469-4471. **PMID: 30989184** ✓ (확정)
    - PMID 30988129는 다른 논문 (우연히 인접한 PMID). 30989184가 올바름.
    - 원조 논문: Vitting-Seerup K, Sandelin A. The landscape of isoform switches in human
      cancers. *Mol Cancer Res* 2017;15(9):1206-1220. PMID: 28584021 (별도 선택 가능)
    - 사용처: Introduction Para 4 (prior isoform-aware methods)

11. **Tardaguila et al. (2018)** — SQANTI
    - Tardaguila M, et al. SQANTI: extensive characterization of long-read transcript
      sequences for quality control in full-length transcriptome identification and
      quantification. *Genome Res* 2018;28(3):396-411. PMID: 29440212
    - 사용처: Introduction Para 4 (prior isoform-aware methods)

12. **Rodriguez et al. (2022)** — APPRIS
    - Rodriguez JM, et al. APPRIS: selecting functionally important isoforms.
      *Nucleic Acids Res* 2022;50(D1):D54-D59. PMID: 34755864 ✓
    - **수정 완료**: Introduction에서 구버전 PMID:30239896 (2018) → PMID:34755864 (2022)로 수정
    - 사용처: Introduction Para 4 (prior isoform-aware methods)

### Function Prediction Background
13. **Gligorijević et al. (2021)** — DeepFRI
    - Gligorijević V, et al. Structure-based protein function prediction using graph
      convolutional networks. *Nat Commun* 2021;12(1):3168. PMID: 34039969
    - ⚠️ PMID 불일치: citation list=34039969 vs Introduction 초기 기재=34210978
      → citation list의 34039969 사용 (Nat Commun 2021 12(1):3168 에 해당); 제출 전 PubMed 확인 필요
    - 사용처: Introduction (DL-based function prediction background)

13b. **Guo et al. (2023)** — CLEAN function prediction
     - Guo Z, et al. Protein function annotation with knowledge-enriched contrastive learning.
       *Nat Comput Sci* 2023;3(9):789-800. PMID: 37217634
     - ⚠️ PMID 및 저자 확인 필요 (Introduction 기재 값 — PubMed 확인)
     - 사용처: Introduction Para 4 (CLEAN method)

13c. **Jaganathan et al. (2019)** — SpliceAI
     - Jaganathan K, et al. Predicting Splicing from Primary Sequence with Deep Learning.
       *Cell* 2019;176(3):535-548. PMID: 30661751
     - 사용처: Introduction Para 4 (SpliceAI — splicing consequence prediction)

~~13d.~~ **[삭제됨 2026-05-28 — 허구 인용 확인]**
     - PMID 35637417 = Dan Li & Cong Liu, *Nat Rev Neurosci* 2022, "Conformational strains of
       pathogenic amyloid proteins in neurodegenerative diseases" — DIFFUSE/이소폼과 무관
     - "Yao et al., Nat Methods, 2022 DIFFUSE novel transcript quantification" 논문은 존재하지 않음
     - introduction_draft, manuscript_full_english 참고문헌에서 모두 제거 완료

14. **Ashburner et al. (2000)** — Gene Ontology
    - Ashburner M, et al. Gene ontology: tool for the unification of biology.
      *Nat Genet* 2000;25(1):25-29. PMID: 10802651
    - 사용처: Methods 4.1 (GO database)

15. **Frankish et al. (2023)** — GENCODE v43
    - Frankish A, et al. GENCODE: reference annotation for the human and mouse genomes
      in 2023. *Nucleic Acids Res* 2023;51(D1):D942-D949. PMID: 36420895
    - 사용처: Methods 4.1, 4.10 (annotation source)

### Discussion — Additional Citations

D1. **Doorenweerd et al. (2017)** — DMD tissue expression / isoform context
    - Doorenweerd N, et al. Timing and localization of human dystrophin isoform expression
      provide insights into the cognitive phenotype of Duchenne muscular dystrophy.
      *NPJ Genom Med* 2017;2:12. PMID: 28808589
    - ⚠️ 제목 확인 필요 (Discussion 기재 값 — PubMed 확인)
    - 사용처: Discussion 5.3 (DMD — experimental confirmation context)

D2. **Formosa et al. (2020)** — NDUFAF6/LYRM domain, Complex I assembly
    - Formosa LE, et al. Building a complex complex: Assembly of mitochondrial respiratory
      chain complex I. *EMBO J* 2020;39(e102817). PMID: 32432371
    - ⚠️ 실제 PMID 및 제목 확인 필요 (Discussion 기재 값)
    - 사용처: Discussion 5.3 (NDUFAF6 — LYRM domain required for complex I integration)

D3. **Saada et al. (2012)** — NDUFAF6/Leigh syndrome
    - Saada A, et al. NDUFAF6 mutations cause a complex I deficiency associated with
      Leigh syndrome and infantile-onset epileptic encephalopathy.
      *Am J Hum Genet* 2012. PMID: 22405087
    - ⚠️ 실제 제목 및 저자 확인 필요 (Discussion 기재 값)
    - **주의**: citation list #4는 Saada 2009 NDUFAF3 (PMID:19463983) — 다른 논문.
      Discussion은 NDUFAF6 전용 Saada 2012 (PMID:22405087)를 사용.
      방법론: NDUFAF6 결과에는 Guerrero-Castillo 2017 (PMID:28094012) + Saada 2012 (22405087) 둘 다 인용
    - 사용처: Discussion 5.3 (NDUFAF6 — Leigh syndrome loss-of-function)

D4. **Wu et al. (2022)** — 5′-partial transcripts / NMD context
    - Wu X, et al. Incomplete annotation of long-read transcriptomes...
      *PLOS Comput Biol* 2022. PMID: 35802768
    - ⚠️ 제목 확인 필요
    - 사용처: Discussion 5.4 (NMD — 5′ truncation common in incomplete transcript assemblies)

D5. **Julien et al. (2016)** — Massively parallel splicing reporters
    - Julien P, et al. Activation of a cryptic splice site in the human ATP7B gene.
      *Nat Biotechnol* 2016. PMID: 27111722
    - ⚠️ 실제 제목/내용 확인 필요 (Discussion 기재 "massively parallel splicing reporters"와
      이 PMID 일치하는지 PubMed 확인 필수 — 제목이 다를 수 있음)
    - 사용처: Discussion 5.5 (Limitations — massively parallel splicing reporter datasets)

### NMD References
16. **Maquat (2004)** — NMD PTC threshold (55 nt rule)
    - Maquat LE. Nonsense-mediated mRNA decay: splicing, translation and mRNP dynamics.
      *Nat Rev Mol Cell Biol* 2004;5(2):89-99. PMID: 15040442
    - 사용처: Methods 4.10, Supp Table S2B

17. **Le Hir et al. (2001)** — EJC position (22 nt upstream of junction)
    - Le Hir H, et al. The exon-exon junction complex provides a binding platform for factors
      involved in mRNA export and nonsense-mediated mRNA decay.
      *EMBO J* 2001;20(17):4987-97. PMID: 11532962
    - 사용처: Methods 4.10, Supp Table S2B

18. **Bambu long-read assembler** — transcript discovery
    - Dong X, et al. Accurate identification of transcript structures using Bambu.
      *bioRxiv* 2022. doi:10.1101/2022.11.14.516358
    - **주의**: 이 preprint가 Nature Methods 2023 게재 확인 필요
      (Dong X, et al. Accurate identification of transcript structures with multiple
      sequencing technologies using Bambu. *Nat Methods* 2023;20(8):1187–1196.
      PMID: 37349533 — if confirmed, use this)
    - 사용처: Methods 4.1 (Bambu-detected novel transcripts)
    - 버전: 제출 전 확인 필요 (conda/R env에서 `packageVersion("bambu")`)

---

## Citations Still Needed / Verification

| Item | Status | Action |
|------|--------|--------|
| NDUFAF6 reference | Guerrero-Castillo 2017 (PMID 28094012) — 기재됨 | ✓ |
| NIPSNAP1 | Abudu et al. Dev Cell 2019 PMID 31063758 | ✓ |
| TAFAZZIN | Schlame 2009 (PMID 19540201) + Barth 2004 (PMID 15303003) | ✓ |
| IsoformSwitchAnalyzeR PMID | PMID 30989184 ✓ Introduction 수정 완료 | ✓ |
| Bambu assembler | Nat Methods PMID 37349533 — isoform_env에 미설치 | ⚠️ 데이터 생성 환경에서 확인 |
| Hermans 2017 triplet loss | arXiv:1703.07737 — #9b | ✓ |
| ESM-2 model name | t30_150M_UR50D ✓ Introduction/Methods 수정 완료 | ✓ |
| Cruz-Jentoft PMID | *Age Ageing* PMID:30312372 ✓ Introduction 수정 완료 | ✓ |
| APPRIS PMID | 2022 업데이트 PMID:34755864 ✓ Introduction 수정 완료 | ✓ |
| **DeepFRI PMID** | **citation list=34039969 vs 초기 Introduction=34210978 — 불일치** | ⚠️ PubMed 확인 필요 |
| **Beaudart et al. 2017** | PMID:27377766 — 저널명 (*JBMR*?) 확인 필요 | ⚠️ PubMed 확인 필요 |
| **Hepple & Rice 2016** | PMID:26040455 — 실제 논문이 mito biogenesis인지 확인 필요 | ⚠️ 내용 확인 필요 |
| **Sandri 2013** | *Physiol Rev* PMID:23899566 — 저널/제목 확인 필요 | ⚠️ PubMed 확인 필요 |
| **Guo et al. 2021 (channelopathies)** | PMID:34429420 — 제목 확인 필요 | ⚠️ PubMed 확인 필요 |
| **Hao et al. 2024** | *Nature* PMID:38114474 — 제목 확인 필요 | ⚠️ PubMed 확인 필요 |
| **CLEAN (Guo 2023)** | *Nat Comput Sci* PMID:37217634 — 저자/제목 확인 필요 | ⚠️ PubMed 확인 필요 |
| ~~**Yao et al. 2022 (DIFFUSE tool)**~~ | **[확인완료 2026-05-28] 허구 인용 — PMID 35637417 = 아밀로이드 논문 (Dan Li & Cong Liu, Nat Rev Neurosci)** | ❌ 논문에서 삭제 완료 |
| PDE4B inhibitor sarcopenia | Background for BambuTx85 finding | LOW priority |

---

## Format Notes
- All PMIDs confirmed from PubMed or noted as "needs verification"
- Journal abbreviations: follow Nature Methods style (Nat Methods, Nat Commun, etc.)
- For preprints: use bioRxiv DOI if published version not available

---

## NIPSNAP1 / TAFAZZIN Key References (needed for Results 3.4 novel candidates)

**NIPSNAP1:**
- Nagaraj R, et al. Nuclear localization of mitochondrial TCA cycle enzymes as a critical
  step in mammalian zygotic genome activation. *Cell* 2017;168(1-2):210-223.
  → may not be the right ref
- Farré D, et al. Proteomic analysis of autophagy deficiency in heart identifies
  NIPSNAP1 as a target of SQSTM1/p62. *Autophagy* 2019 → search needed
- **More likely**: Abudu YP, et al. NIPSNAP1 and NIPSNAP2 act as "eat-me" signals for
  mitophagy. *Dev Cell* 2019;49(4):509-525. PMID: 31063758

**TAFAZZIN (Barth syndrome):**
- Schlame M, Ren M. The role of cardiolipin in the structural organization of
  mitochondrial membranes. *Biochim Biophys Acta* 2009;1788(10):2080-3. PMID: 19540201
- Barth PG, et al. X-linked cardioskeletal myopathy and neutropenia (Barth syndrome):
  clinical manifestations in 16 male patients.
  *J Inherit Metab Dis* 2004;27(4):555-67. PMID: 15303003
