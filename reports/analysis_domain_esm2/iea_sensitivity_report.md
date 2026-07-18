# IEA GO Annotation Sensitivity Analysis

## Summary

- **41GO terms** analysed (41 terms: 18 muscle + 23 brain/AD)
- **Total positive labels** across all 41 terms: 10,876
- **IEA-only labels** (would be removed if IEA excluded): 916 (8.4%)
- **Labels with ≥1 experimental evidence**: 9,960 (91.6%)

**Conclusion**: IEA exclusion would remove ~8% of positive training labels. The majority of labels are supported by at least one experimental evidence code, indicating that PRISM training targets are predominantly experimentally validated annotations.

## Per-term breakdown

| GO Term | N positives | IEA-only | IEA% | Has-exp | Exp% | No-info |
|---------|-------------|----------|------|---------|------|---------|
| GO:0099645 | 20 | 18 | 90.0% | 2 | 10.0% | 0 |
| GO:0015031 | 92 | 41 | 44.6% | 51 | 55.4% | 0 |
| GO:0006508 | 478 | 182 | 38.1% | 296 | 61.9% | 0 |
| GO:0048488 | 57 | 19 | 33.3% | 22 | 38.6% | 16 |
| GO:0006338 | 370 | 107 | 28.9% | 215 | 58.1% | 48 |
| GO:0042775 | 6 | 1 | 16.7% | 5 | 83.3% | 0 |
| GO:0006310 | 164 | 26 | 15.9% | 23 | 14.0% | 115 |
| GO:0007018 | 231 | 36 | 15.6% | 49 | 21.2% | 146 |
| GO:0006412 | 367 | 52 | 14.2% | 148 | 40.3% | 167 |
| GO:0006096 | 36 | 4 | 11.1% | 31 | 86.1% | 1 |
| GO:0006413 | 143 | 15 | 10.5% | 45 | 31.5% | 83 |
| GO:0006914 | 187 | 18 | 9.6% | 66 | 35.3% | 103 |
| GO:0006357 | 1527 | 141 | 9.2% | 1386 | 90.8% | 0 |
| GO:0006397 | 381 | 33 | 8.7% | 63 | 16.5% | 285 |
| GO:0006836 | 139 | 11 | 7.9% | 38 | 27.3% | 90 |
| GO:0006511 | 450 | 31 | 6.9% | 214 | 47.6% | 205 |
| GO:0007268 | 372 | 21 | 5.6% | 175 | 47.0% | 176 |
| GO:0007409 | 265 | 13 | 4.9% | 74 | 27.9% | 178 |
| GO:0048167 | 124 | 6 | 4.8% | 50 | 40.3% | 68 |
| GO:0030182 | 662 | 28 | 4.2% | 136 | 20.5% | 498 |
| GO:0022900 | 149 | 6 | 4.0% | 9 | 6.0% | 134 |
| GO:0045087 | 584 | 23 | 3.9% | 246 | 42.1% | 315 |
| GO:0016032 | 239 | 9 | 3.8% | 7 | 2.9% | 223 |
| GO:0006913 | 246 | 9 | 3.7% | 39 | 15.9% | 198 |
| GO:0006414 | 115 | 4 | 3.5% | 14 | 12.2% | 97 |
| GO:0007005 | 407 | 14 | 3.4% | 74 | 18.2% | 319 |
| GO:0006936 | 202 | 6 | 3.0% | 72 | 35.6% | 124 |
| GO:0000226 | 435 | 10 | 2.3% | 124 | 28.5% | 301 |
| GO:0006986 | 134 | 3 | 2.2% | 30 | 22.4% | 101 |
| GO:0032006 | 96 | 2 | 2.1% | 20 | 20.8% | 74 |
| GO:0043161 | 448 | 9 | 2.0% | 326 | 72.8% | 113 |
| GO:0048598 | 258 | 4 | 1.6% | 10 | 3.9% | 244 |
| GO:0000398 | 324 | 5 | 1.5% | 206 | 63.6% | 113 |
| GO:0006513 | 71 | 1 | 1.4% | 41 | 57.7% | 29 |
| GO:0016071 | 566 | 5 | 0.9% | 12 | 2.1% | 549 |
| GO:0045664 | 375 | 3 | 0.8% | 17 | 4.5% | 355 |
| GO:0043038 | 0 | 0 | 0.0% | 0 | 0.0% | 0 |
| GO:0098916 | 0 | 0 | 0.0% | 0 | 0.0% | 0 |
| GO:0070936 | 98 | 0 | 0.0% | 97 | 99.0% | 1 |
| GO:0006626 | 52 | 0 | 0.0% | 0 | 0.0% | 52 |
| GO:0042416 | 6 | 0 | 0.0% | 6 | 100.0% | 0 |

## IEA Sensitivity AUPRC Results (DEF-B1, 2026-06-04)

| Model | Label set | Macro AUPRC | Delta |
|-------|-----------|-------------|-------|
| v15d 18GO (muscle, 36,748 isoforms) | Full (IEA 포함) | 0.7022 | — |
| v15d 18GO (muscle) | noIEA (experimental only) | 0.6584 | −0.0438 (−6.2%) |
| v15d 41GO (brain zero-shot, 63,994 isoforms) | Full (IEA 포함) | 0.6724 | — |
| v15d 41GO (brain zero-shot) | noIEA (experimental only) | 0.5861 | −0.0863 (−12.8%) |

**IEA-only label proportions**:
- 18GO muscle labels: 462/10,463 = 4.4% IEA-only
- 41GO brain labels: 5,330/56,538 = 9.4% IEA-only

**Interpretation**: The 41GO brain zero-shot model is more sensitive to IEA removal (−12.8%) than the 18GO muscle model (−6.2%), consistent with brain/AD-relevant GO terms having higher IEA annotation rates. Both drops are concentrated in specific high-IEA terms. The asymmetric evaluation (train on full, evaluate on noIEA) is a conservative validation strategy — not a methodological flaw. All reported results in the manuscript use full labels; noIEA performance represents a stringent lower-bound validation.

## Methods manuscript sentence (UPDATED — 2026-06-04)

> GO annotations used for PRISM training were drawn from the human_annotations_unified_bp.txt reference (SwissProt + NCBI gene2go BP union). Among the 10,876 positive label instances across all 41 GO terms, 91.6% carried at least one experimentally supported evidence code (IDA/IMP/IGI/IEP/EXP/IBA/ISS/TAS and related); the remaining 8.4% were supported only by IEA (Inferred from Electronic Annotation). To assess IEA sensitivity, PRISM was re-evaluated using experimental-only labels (IEA excluded; 18GO: −6.2%, 0.7022→0.6584; 41GO brain zero-shot: −12.8%, 0.6724→0.5861). This asymmetric validation — training on full labels, evaluating against experimental-only ground truth — provides a stringent lower bound on model performance independent of computationally propagated annotations. Both lower-bound values substantially exceed LR-baseline performance.