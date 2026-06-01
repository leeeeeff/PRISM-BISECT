# Supplementary Table S2 — NMD Screening Results
**Draft 2026-05-17 | DIFFUSE: Isoform-Level Function Prediction**

---

## Overview

All 126 isoform-switch candidate pairs (top-scoring and low-scoring isoform per pair)
were screened symmetrically for nonsense-mediated decay (NMD) risk using TransDecoder
ORF classification and GTF-based exon junction complex (EJC) distance calculation.
Both the high-scoring (top) and low-scoring (bot) isoforms in each pair were evaluated.
Pairs where either isoform was flagged were excluded from the main analysis.

**Screening criteria:**
- **NMD_RISK**: isoform has a 5′-partial ORF (no detected start codon) AND the
  premature termination codon (PTC) is > 55 nt upstream of the nearest downstream EJC
  (EJC defined as 22 nt upstream of each exon-exon junction; Maquat, 2004)
- **SAFE**: isoform has a complete ORF (TransDecoder); OR 3′-partial ORF; OR
  5′-partial with PTC-to-EJC ≤ 55 nt
- **UNKNOWN**: isoform not found in TransDecoder output or outside transcript boundaries

**Summary:**
| Verdict | Count | % |
|---------|-------|---|
| SAFE (complete ORF or PTC ≤ 55 nt in both isoforms) | 102 | 81.0% |
| EXCLUDED (NMD_RISK in top and/or bot isoform) | 23 | 18.3% |
| UNKNOWN (at least one isoform unresolved) | 1 | 0.8% |
| **Total** | **126** | **100%** |

Cases in the **EXCLUDED** category are removed from the primary isoform-switch analysis
in the main text (Results, Section 3.4). All cases from the SAFE category with complete
ORFs in both isoforms are included in Figure 2C and Supplementary Figure 3.

---

## Table S2A — Excluded Pairs (n = 23, sorted by ratio descending)

### Part 1: Top isoform NMD_RISK (new catches from symmetric screen; n = 9 unique pairs, 11 entries)

| Gene | GO Term | Function | Top Isoform | Bot Isoform | Ratio | Top PTC→EJC (nt) | Note |
|------|---------|----------|-------------|-------------|-------|-----------------|------|
| UQCRB | GO:0007005 | Mito. org | NMD_RISK | SAFE | 2,939× | 64 | Complex III subunit; top isoform would have been highest-ratio novel prediction |
| SOD2 | GO:0007005 | Mito. org | NMD_RISK | SAFE | 1,632× | 142 | Mitochondrial superoxide dismutase |
| CLU | GO:0007005 | Mito. org | NMD_RISK | SAFE | 467× | 102 | Clusterin; mitochondria-associated |
| CLU | GO:0006914 | Autophagy | NMD_RISK | SAFE | 61× | 102 | Same top isoform, independent GO term |
| COX7A2L | GO:0007005 | Mito. org | NMD_RISK | SAFE | 25× | 1,599 | COX assembly factor |
| FHL3 | GO:0030017 | Sarcomere org | NMD_RISK | SAFE | 9× | 104 | LIM domain muscle protein |
| UQCC2 | GO:0007005 | Mito. org | NMD_RISK | SAFE | 3× | 570 | Complex III assembly factor |
| NDUFAF5 | GO:0007005 | Mito. org | NMD_RISK | SAFE | 3× | 198 | Complex I assembly factor |
| ATG12 | GO:0006914 | Autophagy | NMD_RISK | SAFE | 3× | 3,206 | ATG12 conjugation pathway |
| BNIP3 | GO:0007005 | Mito. org | NMD_RISK | NMD_RISK | 3× | 162† | Both isoforms 5′-partial |
| BNIP3 | GO:0006914 | Autophagy | NMD_RISK | NMD_RISK | 2× | 162† | Same pair, independent GO term |

**† BNIP3**: Both top and bot isoforms are 5′-partial; the entire gene entry is flagged for
both GO terms. These were already detected in the original bot-only screen; the symmetric
screen additionally confirms the top isoform is also NMD_RISK.

### Part 2: Bot isoform NMD_RISK (original screen, retained unchanged; n = 12 entries)

| Gene | GO Term | Function | Bot Isoform | Top ORF | Bot aa | Ratio | Bot PTC→EJC (nt) | Note |
|------|---------|----------|-------------|---------|--------|-------|-----------------|------|
| GABARAPL1 | GO:0006914 | Autophagy | ENST00000541960.5 | complete | 111 aa | 2,222× | 1,316 | ATG8 family |
| SORBS2 | GO:0030017 | Sarcomere org | ENST00000698537.1 | complete | 295 aa | 2,105× | 1,333 | Actin-binding sorbin domain |
| UQCC1 | GO:0007005 | Mito. org | ENST00000397554.5 | internal‡ | 224 aa | 2,047× | 2,464 | Complex III assembly factor |
| WDR45B | GO:0007005 | Mito. org | ENST00000572583.5 | complete | 200 aa | 134× | 823 | WD-repeat autophagy protein |
| GABARAPL1 | GO:0007005 | Mito. org | ENST00000541960.5 | complete | 111 aa | 52× | 1,316 | Same bot isoform, independent GO term |
| WDR45B | GO:0006914 | Autophagy | ENST00000572583.5 | complete | 200 aa | 22× | 823 | Same bot isoform, independent GO term |
| TPM2 | GO:0030017 | Sarcomere org | ENST00000644325.1 | complete | 132 aa | 7× | 490 | β-tropomyosin |
| MGME1 | GO:0007005 | Mito. org | ENST00000467391.1 | complete | 109 aa | 6× | 94 | Mitochondrial genome maintenance |
| LAMP2 | GO:0006914 | Autophagy | ENST00000706600.1 | complete | 496 aa | 5× | 157 | Lysosomal membrane protein |
| NDUFA5 | GO:0007005 | Mito. org | ENST00000470123.2 | complete | 100 aa | 3× | 170 | Complex I accessory subunit |
| CAMK2A | GO:0007005 | Mito. org | ENST00000351010.6 | complete | 206 aa | 3× | 297 | Ca²⁺/calmodulin-dependent kinase |
| BID | GO:0007005 | Mito. org | ENST00000399765.5 | complete | 157 aa | 2× | 921 | BH3-only apoptosis protein |

**‡ UQCC1 top isoform**: TransDecoder classified as 'internal' (no start or stop codon within
ORF boundaries) — treated as unreliable; pair excluded regardless of bot verdict.

---

## Table S2B — Screening Method Details

| Parameter | Value | Reference |
|-----------|-------|-----------|
| TransDecoder version | 5.7.1 | Haas et al. (2013) *Genome Biol* |
| Minimum ORF length | 30 aa (for NMD classification) | — |
| EJC offset upstream of junction | 22 nt | Le Hir et al. (2001) *EMBO J* |
| NMD PTC threshold | > 55 nt upstream of EJC | Maquat (2004) *Nat Rev Mol Cell Biol* |
| GTF source | GENCODE v43 (cleaned_annotations.gtf) | Frankish et al. (2023) *NAR* |
| Screening scope | Both top AND bot isoforms per pair | This study |
| Input pairs | 126 candidate pairs (2 TSV files) | reports/nmd_screening_symmetric_20260516.json |
| Script | /tmp/nmd_screening_symmetric.py | This study |

---

## Notes on Interpretation

**Symmetric screening rationale:** The original (bot-only) screen excluded 14/126 pairs (11.1%).
When extended to also screen the top (high-scoring) isoform, 9 additional pairs were excluded
(7.1%), bringing the total to 23/126 (18.3%). The most consequential new catches are
UQCRB (2,939×) and SOD2 (1,632×), which would have been the two highest-ratio novel
predictions had only the bot isoform been screened.

**Cross-GO redundancy:** Several genes appear twice (once per GO term). These represent the
same isoform pair flagged independently in two GO term analyses, not two distinct transcript
assemblies. Unique flagged gene-isoform pairs: 14 (not 23 entries).

**BNIP3 special case:** Both top and bot isoforms of BNIP3 are 5′-partial (detected in both
the original and symmetric screens). The model's score ratio (2–3×) reflects a comparison
between two potentially translation-incompetent isoforms and is therefore uninformative about
functional difference.

**Impact on main analysis:** After symmetric NMD exclusion, 102/126 candidate pairs (81.0%)
remain for the primary isoform-switch analysis. The reported candidates in Figure 2 and
Supplementary Figure 3 all have complete ORFs verified in both the top and bot isoforms.

**Comparison with original screen:**

| Screen | Excluded | Safe | Detection scope |
|--------|----------|------|-----------------|
| Original (bot-only) | 14 (11.1%) | 111 (88.1%) | Low-scoring isoform only |
| Symmetric (this study) | 23 (18.3%) | 102 (81.0%) | Both isoforms per pair |
| Additional catches | +9 (7.1%) | — | High-scoring isoform with NMD_RISK |
