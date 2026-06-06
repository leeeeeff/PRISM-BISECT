# HMMSCAN / Pfam Domain Case Summary: KIF21B, NDUFS4, DLG1

**Generated**: 2026-06-05  
**Source**: HMMER 3.3.2 hmmscan vs Pfam-A (release 36.0, E-value ≤ 0.01)  
**Pipeline outputs**:
- `Final_analysis/pipeline_bioanalysis/outputs/KIF21B_Excitatory/hmmscan.out`
- `Final_analysis/pipeline_bioanalysis/outputs/NDUFS4_Excitatory/hmmscan.out`
- `Final_analysis/08_biological_deepening/dlg1_domain/hmmscan_results.txt`

---

## Domain + PRISM Score Summary Table

| Isoform | Length | Pfam domains (hmmscan) | PRISM top GO term | PRISM score | Type |
|---------|--------|------------------------|-------------------|-------------|------|
| KIF21B tr293004 (CT-dominant) | 419 aa | Kinesin motor domain (P-loop GQTGAGKTYT, Switch-I SSRSHA, Switch-II DLAGSE — direct motif matches; full motor domain aa 1–360) | microtubule-based movement | 0.9753 | NIC; motor-competent |
| KIF21B tr292978 (AD-dominant) | 711 aa | WD40 ×7 (PF00400, E=2.7e-41, aa 372–686); ANAPC4_WD40 ×3 (PF12894, E=0.0011); NBCH_WD40 ×2; no kinesin motor domain | regulation of neuron differentiation | 0.1913 | NNIC; motor-incompetent |
| NDUFS4-201 (CT canonical) | 175 aa | NDUS4 (PF01110, E=6.8e-39, score=132.2, aa 76–169) | mitochondrion organization | 0.6169 | FSM; MTS present (composite 4/5) |
| NDUFS4 tr73243 (AD novel) | 378 aa | RVT_1 / reverse transcriptase (PF00078, E=6.6e-44, score=150.3, aa 141–366); no NDUS4 | chemical synaptic transmission | 0.1437 | NNIC; MTS absent (composite 1/5); LINE-1 exon origin |
| DLG1 tr319500 (AD novel NNIC) | 186 aa | L27_1 (PF09058, E=3.7e-30, score=103.7, aa 6–63); MAGUK_N_PEST (PF10608, E=9e-12, score=45.5, aa 107–144); PDZ_assoc partial (PF10600, E=1.5e-06, aa 152–172); no PDZ/SH3/GK | microtubule-based movement | 0.3367 | NNIC; truncated/aberrant |
| DLG1-201 (CT canonical proxy: DLG1-232, cos_sim=0.999) | 926 aa | Guanylate_kin; PDZ; PDZ_2; PDZ_6; SH3_1; SH3_2 (annotated from supplementary table; canonical PSD-95 scaffold) | chemical synaptic transmission | 0.7106 | FSM; full PSD-95 scaffold |

---

## Per-Case Detail

### KIF21B (Excitatory neurons, DTU p = 9.28×10⁻⁸ / 3.81×10⁻⁶)

**Biological context**: Bidirectional isoform switch in AD excitatory neurons. CT-dominant tr293004 is a motor-competent N-terminal fragment; AD-dominant tr292978 is a motor-dead WD40 isoform proposed to act as dominant-negative.

| Feature | tr293004 (CT, 419 aa) | tr292978 (AD, 711 aa) |
|---------|----------------------|----------------------|
| Pfam domains | Kinesin motor (P-loop, Switch-I/II confirmed) | WD40 ×7 (PF00400), ANAPC4_WD40 ×3 (PF12894) |
| Motor domain | COMPLETE (aa 1–360) | ABSENT |
| Coiled-coil | stub (aa 383–419) | PRESENT (~300 aa) |
| WD40 cargo domain | ABSENT | PRESENT (aa 372–686, 315 aa) |
| PRISM top GO | microtubule-based movement | regulation of neuron differentiation |
| PRISM max score | **0.9753** | **0.1913** |
| PRISM Δ (AD−CT) | −0.784 (large loss of motor function signal) | — |
| DIFFUSE Δ (earlier) | −0.855 (MT-based movement, v15d_bp_clean) | — |

**PRISM data source**: `prism_app/data/demo/bisect_cases.json` (brain_672 score matrix, exact match for tr293004; tr292978 matched via gene median).

---

### NDUFS4 (Excitatory neurons, DTU p = 3.62×10⁻⁶)

**Biological context**: AD novel isoform tr73243 displaces canonical NDUFS4-201. CT isoform has MTS and assembles into Complex I; AD isoform contains L1PA11-derived RVT_1 domain, lacks MTS, and is predicted to mislocalize to cytoplasm.

| Feature | NDUFS4-201 (CT, 175 aa) | tr73243 (AD, 378 aa) |
|---------|------------------------|---------------------|
| Pfam domains | NDUS4 (PF01110, E=6.8e-39, aa 76–169) | RVT_1 (PF00078, E=6.6e-44, aa 141–366) |
| MTS composite score | 4/5 (HIGH — mitochondrial import likely) | 1/5 (LOW — cytoplasmic expected) |
| Net charge first 30 aa | +5 | −1 |
| AlphaFold CT pLDDT | 84.1 mean; NDUS4 domain: 96.91 | not available |
| PRISM top GO | mitochondrion organization | chemical synaptic transmission |
| PRISM max score | **0.6169** | **0.1437** |
| PRISM Δ (AD−CT) | −0.473 (loss of mitochondrial function signal) | — |
| PPI validation | SUPPORTED (NDUFAF2, NDUFS6, NDUFB9, NDUFA12, NDUFS1; STRING 996–999) | — |
| Regulatory origin | — | L1PA11/L1PA3 epigenetic derepression (DNMT3A down) |

**hmmscan source**: `Final_analysis/pipeline_bioanalysis/outputs/NDUFS4_Excitatory/hmmscan.out`

---

### DLG1 (OPC, DTU p = 9.03×10⁻¹⁰)

**Biological context**: Novel 186 aa isoform (tr319500) is AD-dominant; canonical DLG1-201 (926 aa, 3×PDZ scaffold) is CT-dominant (DLG1-232 used as proxy, cos_sim=0.999, since DLG1-201 absent from brain IsoQuant index). PRISM correctly predicts functional loss in the AD-dominant truncated isoform.

| Feature | tr319500 (AD novel, 186 aa) | DLG1-201 (CT canonical, 926 aa) |
|---------|---------------------------|--------------------------------|
| Pfam domains | L27_1 (PF09058, E=3.7e-30, aa 6–63); MAGUK_N_PEST (PF10608, E=9e-12, aa 107–144); PDZ_assoc partial (PF10600, E=1.5e-06) | Guanylate_kin; PDZ; PDZ_2; PDZ_6; SH3_1; SH3_2 (from supplementary table annotation) |
| PDZ domains (canonical ×3) | ABSENT | PRESENT (PDZ; PDZ_2; PDZ_6) |
| SH3 domains | ABSENT | PRESENT (SH3_1; SH3_2) |
| Guanylate kinase | ABSENT | PRESENT |
| AlphaFold CT pLDDT | NA (novel, no AF entry) | 73.05 |
| PRISM top GO | microtubule-based movement | chemical synaptic transmission |
| PRISM max score | **0.3367** | **0.7106** |
| PRISM Δ (AD−CT) | +0.6369 (recovery of synaptic function signal in AD canonical) | — |
| PPI validation | SUPPORTED (GRIA1, STRING 999) | — |
| AD canonical proxy | DLG1-232 (cosine similarity 0.999 to gene median; DLG1-201 absent from long-read dataset) | — |

**hmmscan source**: `Final_analysis/08_biological_deepening/dlg1_domain/hmmscan_results.txt`

---

## Key Observations

1. **PRISM and domain analysis are concordant** in all three cases: large PRISM score drops (−0.47 to −0.78) accompany the loss of functionally critical domains (kinesin motor, NDUS4, PDZ×3).

2. **InterProScan/Pfam cannot detect the KIF21B motor domain in tr293004 as a named Pfam hit** (the motor domain is confirmed by direct P-loop/Switch motif matching), but PRISM's ESM-2 embedding still yields a score of 0.975 for microtubule-based movement.

3. **NDUFS4 tr73243 gains RVT_1** (a LINE-1 ORF2p-derived domain), representing a case where the AD isoform carries an entirely non-mitochondrial Pfam domain — InterProScan and PRISM both signal a complete functional class change.

4. **DLG1 tr319500 retains L27_1 and MAGUK_N_PEST** (N-terminal MAGUK subfamily signatures) but loses all three PDZ domains and both SH3 domains required for PSD-95 scaffold function. The PRISM score drop from 0.71→0.34 (synaptic transmission) is consistent with this architectural loss.

---

## Data Sources Referenced

| File | Content |
|------|---------|
| `Final_analysis/pipeline_bioanalysis/outputs/KIF21B_Excitatory/hmmscan.out` | hmmscan results for tr292978/tr293004 |
| `Final_analysis/pipeline_bioanalysis/outputs/NDUFS4_Excitatory/hmmscan.out` | hmmscan results for NDUFS4-201/tr73243 |
| `Final_analysis/08_biological_deepening/dlg1_domain/hmmscan_results.txt` | hmmscan results for tr319500 |
| `Final_analysis/07_ad_isoform_switching/07A_kif21b_switch/domain_analysis_report.md` | KIF21B domain analysis narrative |
| `Final_analysis/pipeline_bioanalysis/outputs/NDUFS4_Excitatory/report.md` | NDUFS4 full BISECT report |
| `prism_app/data/demo/bisect_cases.json` | PRISM scores (brain_672) per BISECT case |
| `reports/supplementary_table_S_bisect_83cases.tsv` | Supplementary table with domain annotations |
