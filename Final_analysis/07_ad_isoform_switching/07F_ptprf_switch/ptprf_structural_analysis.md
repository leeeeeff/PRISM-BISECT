# PTPRF — Inhibitory Neuron Isoform Switch: Structural Analysis
**Generated**: 2026-05-22 | BISECT pipeline output + manual curation

---

## 1. Summary

PTPRF encodes the Leukocyte Antigen Related (LAR) receptor-type protein tyrosine phosphatase,
a Type IIa LAR-RPTP family member central to synapse formation and maintenance. In AD inhibitory
neurons, PTPRF undergoes the most extensive domain reorganization among all 23 batch cases:
the CT-dominant isoform (ENST00000414879, 1266 aa) loses 10 Pfam domain families and the AD-
dominant isoform (ENST00000617451, 262 aa) gains 8 distinct Ig-like domain families. Critically,
the two isoforms share **zero sequence identity** — they arise from completely non-overlapping
exon sets, indicating distinct transcription start sites (TSS) and/or mutually exclusive first
exon usage. This is not a simple domain-truncation event but a complete architectural replacement.

| Parameter | CT isoform | AD isoform |
|-----------|-----------|-----------|
| Transcript | ENST00000414879 | ENST00000617451 |
| Length | 1266 aa | 262 aa |
| SQANTI3 class | full-splice_match | full-splice_match |
| Shared sequence | 0/20 aa blocks | (same — no overlap) |
| TM domain | YES (aa615-632; KD=3.9-4.3) | NO |
| Signal peptide | NO (cytoplasmic N-term) | YES (aa1-26; KD=2.77) |
| Phosphatase | 2× Y_phosphatase | NONE |
| Extracellular type | 5× fn3 (HSPG/CSPG binding) | 2× Ig (NGL-3/Slitrk binding) |
| Predicted localization | Membrane-bound RPTP | Secreted 2-Ig fragment |
| DIFFUSE Δ | — | +0.729 (AD_high; Inhibitory) |
| DTU p | — | 3.89×10⁻¹⁷ |

---

## 2. CT Isoform Architecture (ENST00000414879, 1266 aa)

### 2.1 Domain map

```
aa1                    614    615-632         633        734            965    1023           1256 1266
|——— Extracellular ———|——TM——|——————————————— Intracellular ——————————————————|
    5× fn3 repeats        |||   PTP1 (catalytic)         PTP2 (regulatory)
    (HSPG/CSPG binding)   |||   Y_phosphatase E=7×10⁻⁹²  Y_phosphatase E=2×10⁻⁸⁶
```

**Extracellular domain (aa1-614):**
| fn3 copy | Position | E-value | Score |
|----------|----------|---------|-------|
| fn3 #1 | aa2-50 | 1.4×10⁻¹⁰ | 30.9 |
| fn3 #2 | aa68-157 | 5.1×10⁻¹⁷ | 51.5 |
| fn3 #3 | aa180-251 | 9.6×10⁻¹⁶ | 47.4 |
| fn3 #4 | aa267-348 | 1.1×10⁻¹⁸ | 56.9 |
| fn3 #5 | aa364-424 | 7.9×10⁻⁵ | 12.4 |

Additional extracellular hits: NDNF (aa15-43), Interfer-bind (×3), Pur_ac_phosph_N (×4),
Arylsulfotran_N (×2), fn3_2 (aa269-344).

**Transmembrane domain (aa615-632):**
Sequence: PVLAVILIILIVIAILLF
Kyte-Doolittle max: 4.28 (window aa620-628: VILIILIVI) — classic hydrophobic single-pass TM.

**Intracellular phosphatase domains:**
- PTP1 (catalytic): aa734-965 (E=7.1×10⁻⁹², score=296.6) — primary catalytic domain.
  Contains Y_phosphatase3 (aa888-920) and PTPlike_phytase (aa873-924) active-site loops.
  DSPc motif aa900-947. The catalytic Cys (HC(x)5R motif) expected at ~aa900.
- PTP2 (regulatory/substrate-binding): aa1023-1256 (E=2.0×10⁻⁸⁶, score=278.8).
  Contains DSPc (aa1176-1243) and Y_phosphatase3 (aa1178-1233). PTP2 typically acts as
  a regulatory phosphatase or stabilises substrate interaction without primary catalysis.

### 2.2 Functional logic of CT isoform

This CT isoform is a membrane-bound receptor phosphatase with:
- **Extracellular fn3 repeats**: bind heparan sulfate proteoglycans (HSPGs: Agrin, Glypican-4)
  and chondroitin sulfate proteoglycans (CSPGs: Aggrecan). fn3-HSPG interaction regulates
  growth cone guidance, axonal pathfinding, and the perisynaptic extracellular matrix.
- **Dual PTP**: PTP1 active site dephosphorylates β-catenin, Akt (pSer473), TrkB, EGFR, ErbB2,
  and the insulin receptor. PTP2 provides regulatory substrate binding. Net effect: LAR-PTP
  maintains synaptic signalling homeostasis by controlling receptor tyrosine kinase phosphorylation
  at inhibitory synaptic membranes.

---

## 3. AD Isoform Architecture (ENST00000617451, 262 aa)

### 3.1 Domain map

```
aa1   26  27          126 127     135            225 226              262
|—SP—|——|—— Ig domain 1 ——|——linker——|—— Ig domain 2 ——|—— C-tail ——|
      ↓          (I-set/Ig_3)              (I-set/Ig_3)    4× Cys
  cleavage
  (signal
  peptide)
```

**Signal peptide (aa1-26):**
Sequence: MAPEPAPGRTMVPLVPALVMLGLVAG
- Hydrophobic core: aa10-25 (TMVPLVPALVMLGLVAG; KD max = 2.77 at aa17-25)
- Charged N-term: aa1-9 (MAPEPAPGR; 1 Arg, net charge +1)
- Cleavage site: AGA at aa25-27 (AXA motif; SignalP canonical pattern)
- Predicted: cleaved after Ala26 → 236-aa mature secreted protein

**Ig domain 1 (aa32-126 of preprotein; aa6-100 of mature protein):**
| Pfam hit | Position | E-value | Score |
|----------|----------|---------|-------|
| I-set | aa33-124 | 1.9×10⁻²¹ | 65.1 |
| Ig_3 | aa32-111 | 3.6×10⁻¹⁷ | 51.9 |
| Ig_2 | aa47-121 | 6.2×10⁻⁷ | 18.8 |
| ig | aa39-121 | 3.2×10⁻⁹ | 26.2 |
| C2-set_2 | aa49-77 | 3.5×10⁻⁶ | 16.2 |
Best E-value: I-set E=1.9×10⁻²¹ → intermediate Ig fold (I-set = intermediate subset,
between V-set and C-set; characteristic of cell adhesion molecules).

**Ig domain 2 (aa135-225 of preprotein; aa109-199 of mature protein):**
| Pfam hit | Position | E-value | Score |
|----------|----------|---------|-------|
| I-set | aa136-216 | 2.1×10⁻¹⁷ | 52.2 |
| Ig_3 | aa135-211 | 7.4×10⁻¹⁷ | 50.9 |
| Ig_2 | aa144-223 | 4.8×10⁻¹¹ | 32.0 |
| ig | aa145-214 | 1.5×10⁻⁶ | 17.6 |
| Ig_5 | aa150-214 | 9.4×10⁻⁵ | 12.0 |
Best E-value: I-set E=2.1×10⁻¹⁷. Structurally equivalent to Ig domain 1.

**C-terminal tail (aa226-262; aa200-236 of mature):**
Sequence: RGLAAWARSPMGIWSHPIRLLGVCACVCAHTGTLICV
Cys residues: aa249, aa251, aa253, aa261 (4 Cys in 37aa)
- VCACVC pattern (aa248-253): alternating Val-Cys → may form 2 disulfide bridges
- No hydrophobic TM domain (KD max ~2.0 for aa245-253, borderline)
- GPI anchor signal check: potential omega site at aa240 (Ser) or aa247 (Gly), but
  downstream sequences are polar/charged — GPI anchor unlikely
- Most likely: a Cys-rich extracellular tail that forms intramolecular disulfide bonds
  stabilising the C-terminus. Not a membrane anchor.

### 3.2 Sequence non-overlap with CT isoform

Direct comparison of CT and AD sequences at all 20-aa windows: **zero shared blocks**.
The isoforms diverge at amino acid 2 (CT=Tyr, AD=Ala). This proves that the two isoforms
use completely different exons — almost certainly alternative first exons with distinct TSS.
This is mechanistically different from the KIF21B switch (which involved alternative splicing
of a continuous pre-mRNA) — PTPRF may undergo alternative promoter activation in AD inhibitory
neurons.

---

## 4. Mechanistic Model

### 4.1 The "Secreted Antagonist" Model

```
                    CT (Control) inhibitory neuron
                    
  Perisynaptic ECM           Synapse            Intracellular
  ──────────────────    ───────────────────   ─────────────────
  HSPG/CSPG     fn3──TM──PTP1──PTP2──→ β-catenin/Akt dephosphorylation
  (Agrin, etc)                             ↓
                                         Synaptic stability
                                         
                    AD inhibitory neuron
                    
  Perisynaptic ECM           Synapse            Intracellular
  ──────────────────    ───────────────────   ─────────────────
  NGL-3/Slitrk1   Ig1──Ig2──[SECRETED]──×──  NO signal
                   ↑
                   Competes with canonical LAR-RPTP for NGL-3/Slitrk binding
                   → canonical LAR-RPTP displaced from synapse
                   → PTP-dependent β-catenin dephosphorylation LOST
                   → β-catenin signaling upregulated (Wnt pathway activation?)
```

**Step 1: CT isoform lost.** PTPRF fn3-TM-PTP form disappears from inhibitory neuron membranes.
Loss of fn3 removes HSPG-mediated perisynaptic matrix contact. Loss of PTP1 removes
β-catenin/Akt dephosphorylation.

**Step 2: AD isoform secreted.** Signal peptide-cleaved 2-Ig fragment enters the secretory
pathway and is released into the synaptic cleft. The 2 I-set Ig domains are structurally
homologous to Ig1-Ig2 of canonical PTPRF, which bind synaptic organizers NGL-3 (LRRTM4
homologue), Slitrk1-6, and IL1RAPL1.

**Step 3: Competitive antagonism.** The secreted AD Ig fragment binds the same synaptic
organizer epitopes as full-length PTPRF Ig1-3, without membrane anchoring. This:
(a) Blocks incoming NGL-3/Slitrk from binding any residual full-length PTPRF
(b) Cannot signal (no PTP, not membrane-anchored)
(c) Effectively acts as a dominant-negative decoy at inhibitory synapses

**Step 4: Downstream consequences.** Loss of LAR-PTP-mediated dephosphorylation:
- β-catenin hyperphosphorylation → altered Wnt/β-catenin target gene expression
- TrkB hypophosphorylation (LAR-PTP normally activates TrkB by removing inhibitory pTyr) →
  reduced BDNF signalling in inhibitory neurons
- Akt pSer473 retained → altered mTOR regulation (convergence with DMD/SOGA finding)

---

## 5. Comparison with KIF21B Switch

| Feature | KIF21B | PTPRF |
|---------|--------|-------|
| Isoform type | Alternative splicing (shared genomic region) | Alternative first exon (no shared sequence) |
| Domain change | Motor lost, WD40 gained (same gene product, different modules) | fn3-TM-PTP replaced by 2-Ig-secreted (entirely different protein) |
| Cell type | Excitatory | Inhibitory |
| Mechanism | Dominant-negative motor (intracellular, dimerization) | Dominant-negative receptor (extracellular, secreted decoy) |
| DIFFUSE convergence | tr292978 scores 0.111 vs tr293004 0.966 | AD 262aa scores higher (Ig drives Synaptic transmission score) |
| Cross-case link | IFT122 mirror (WD40 redistribution) | PTPRS also switches in Astrocytes (LAR-RPTP family) |

---

## 6. LAR-RPTP Family Convergence: PTPRF + PTPRS

Both PTPRF (LAR) and PTPRS (PTP-σ) appear in our 23-case analysis:
- PTPRF: Inhibitory neurons, Δ=+0.729, DTU p=3.9×10⁻¹⁷
- PTPRS: Astrocytes, Δ=+0.788, DTU p=1.4×10⁻²⁹, lost=SusE, gained=Ig_C17orf99

PTPRS in astrocytes: the CT isoform loses a SusE domain (bacterial-type carbohydrate-binding)
while the AD isoform gains an Ig_C17orf99 domain. This is a smaller domain swap, but the
direction is again from HSPG/CSPG-binding (SusE analog) toward Ig-type interaction — similar
to the PTPRF transition. Two independent LAR-RPTP family members simultaneously switching
their extracellular domain type (fn3/HSPG-binding → Ig) in two different cell types
(Inhibitory neurons + Astrocytes) suggests a coordinated AD-associated shift in
RPTP-ligand landscape across the synaptic microenvironment.

---

## 7. Evidence Levels

| Claim | Evidence level | Basis |
|-------|---------------|-------|
| AD PTPRF switch exists in inhibitory neurons | Established (this study) | DTU p=3.9×10⁻¹⁷, DIFFUSE Δ=+0.729 |
| CT isoform is membrane-bound fn3-TM-PTP | Established | KD=4.28 TM; Y_phosphatase E=7×10⁻⁹² |
| AD isoform has signal peptide | Strong | KD 2.77 hydrophobic core; AXA cleavage |
| AD isoform is secreted | Strong (predicted) | No TM detected; SP present |
| 2-Ig domains are I-set type | Established | I-set E=1.9×10⁻²¹ and 2.1×10⁻¹⁷ |
| No shared sequence between isoforms | Established | 0/20-aa window search |
| Alternative first exon mechanism | Strong inference | Zero sequence overlap + both FSM |
| AD Ig domains bind NGL-3/Slitrk1-6 | Hypothesis | Structural analogy to canonical PTPRF Ig1-3 |
| Secreted decoy competitive antagonism | Hypothesis | Mechanism analogy to receptor ectodomain shedding |
| β-catenin/TrkB/Akt signalling altered | Hypothesis | Known PTPRF substrates; no direct evidence |
| PTPRF+PTPRS coordinated switch | Observation | Two cases in same dataset; no causal link |

---

## 8. Experimental Validation Plan

**Priority 1 (1-2 months): Confirm secreted protein**
- Conditioned medium from AD donor iPSC-derived inhibitory neurons → anti-PTPRF-Ig1 antibody
  western blot and ELISA for ~27 kDa band (mature 236 aa = ~27 kDa).
- Alternatively: recombinant expression of aa1-262 in HEK293 → confirm media secretion by
  N-terminal sequencing (confirm signal peptide cleavage at Ala26).

**Priority 2 (2-4 months): Confirm ligand competition**
- Surface plasmon resonance (SPR): recombinant AD Ig1-2 (aa27-225) vs. NGL-3 ectodomain.
  Compare Kd with canonical PTPRF Ig1-3 ectodomain. If Kd is within 10×, competition is plausible.
- Dot blot / ELISA competition assay: titrate AD fragment against NGL-3 binding to full-length
  PTPRF. IC50 should be in nM range if decoy model holds.

**Priority 3 (4-6 months): Confirm phosphatase downstream effects**
- AD iPSC-derived inhibitory neurons (or patient-derived organoids): anti-pβ-catenin (Y142)
  and anti-pTrkB immunostaining. Compare AD vs. CT neurons.
- Ectopic expression of ENST00000617451 in inhibitory neuronal culture → measure β-catenin
  phosphorylation and Akt pSer473 changes relative to empty vector and ENST00000414879.

---

## 9. Implications for AD Drug Development

The secreted AD PTPRF Ig fragment constitutes a potential **biomarker** and **therapeutic target**:
1. **CSF biomarker**: If secreted, the 27 kDa fragment may be detectable in CSF of AD patients.
   Mass-spec proteomics of AD CSF with PTPRF peptide panel (Ig1-2-specific peptides) could
   confirm isoform-level expression in vivo.
2. **Therapeutic neutralization**: A monoclonal antibody targeting the Ig1/Ig2 ligand-binding
   surface of the AD fragment could block its competitive antagonism — restoring normal NGL-3/
   Slitrk interaction with endogenous full-length PTPRF.
3. **Alternative promoter targeting**: If the AD isoform arises from an alternative promoter,
   identifying the responsible transcription factor or epigenetic change could provide an
   upstream target to suppress the switch.
