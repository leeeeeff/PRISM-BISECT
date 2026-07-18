# KIF21B Isoform Switch Validation — Experimental Proposal

**Target**: Validate PRISM+BISECT prediction that KIF21B undergoes AD-specific motor domain loss  
**Priority**: Track B (Nature Neuro) critical path  
**Estimated timeline**: 6 months (iPSC differentiation: 6 weeks; functional assays: 4 weeks × 2)

---

## Biological Background

PRISM predicts KIF21B-exon17-containing isoform (motor domain intact, CT max = 0.975) is replaced
in AD excitatory neurons by a WD40-repeat isoform (AD max = 0.105) — a 9.3× functional loss
in microtubule-based movement (DTU p = 9.3×10⁻⁸, padj = 1.54×10⁻¹⁰ in discovery cohort).

SRSF5 (logFC = −0.279, padj = 1.6×10⁻⁷) and RBFOX1 (logFC = +0.180, padj = 1.2×10⁻⁴) are
identified as likely trans-regulators of this switch (M8 regulatory context module).

**Why KIF21B specifically**: (1) highest CT max score in entire dataset (0.975, 100th percentile);
(2) independent DTU replication in Ebbert long-read cohort (MWU p=0.026, delta=−0.30);
(3) RBFOX1-mediated alternative splicing of neuronal kinesins is a known mechanism.

---

## Experiment 1: iPSC → Neuron — KIF21B Isoform Switch Replication

**Goal**: Confirm CT→AD isoform ratio change in a cellular AD model

### Protocol
1. iPSC lines: 3 AD patients (APOE4/4 or APP mutation), 3 age-matched controls
2. Differentiation: Dual SMAD inhibition → cortical neuron (day 30, ~70% MAP2+)
3. Long-read RNA-seq: Oxford Nanopore PromethION or PacBio Revio, per-cell or bulk
4. Analysis: IsoQuant → KIF21B isoform quantification → chi-sq DTU

**Expected result**: CT-canonical isoform (motor domain) depleted in AD-iPSC neurons  
**Success criterion**: delta_IF ≥ 0.20, p < 0.05

**Validation by RT-PCR (faster, cheaper)**:
- Primers: exon17-junction (canonical) vs exon17-skip junction (WD40 alt)
- Quantify: exon17+/(exon17+ + exon17-) ratio in AD vs CT neurons
- Expected: ratio decreases in AD (CT ~0.73, AD ~0.43 from Ebbert)

---

## Experiment 2: ASO Knockdown of Canonical KIF21B → MT Transport Assay

**Goal**: Demonstrate functional consequence of isoform switch (motor domain loss)

### Protocol
1. Antisense oligonucleotide (ASO) targeting exon17 inclusion:
   - ASO-17: block RBFOX1 binding site upstream of exon17 → force exon17 skipping
   - Control: scramble ASO
2. iPSC-derived neurons, day 30 post-differentiation
3. Mitochondrial transport assay:
   - MitoTracker Red + live imaging (30 min, 1 frame/2 sec)
   - Kymograph analysis: anterograde/retrograde velocity, run length, pause frequency
4. MT dynamics:
   - EB3-GFP comets: growth rate, catastrophe frequency

**Expected result**: ASO-17 (forced exon17 skipping) → reduced anterograde MT transport,
consistent with loss of kinesin-2 processivity

**Success criterion**: ≥25% reduction in anterograde velocity or run length, p < 0.05

---

## Experiment 3: AD Patient Brain Tissue Validation (Computational)

**Goal**: Confirm KIF21B isoform switch in independent AD brain dataset

### Existing data options
1. **Ebbert cohort** (already completed): MWU p=0.026, direction replicated ✅
2. **Allen Brain Cell Atlas** (snRNA-seq): check KIF21B exon usage if long-read available
3. **ROSMAP** (Religious Orders Study/Memory and Aging Project): bulk RNA-seq
   - Gene-level: check if KIF21B total expression differs AD vs CT
   - If long-read subset available: isoform-level validation

### Analysis plan
- Cross-cohort meta-analysis: combine Samsung (p=9.3×10⁻⁸) + Ebbert (p=0.026)
- Fisher's combined p: ~10⁻⁸ (two independent long-read cohorts)

---

## Timeline

```
Month 1–2:  iPSC maintenance + AD model characterisation (APOE4 confirmation)
Month 2–4:  iPSC → neuron differentiation (6 weeks per batch)
Month 4:    RT-PCR: KIF21B isoform ratio (quick validation, 2 weeks)
Month 4–5:  Long-read RNA-seq (if RT-PCR confirms)
Month 5–6:  ASO knockdown + MT transport assay (4 weeks)
Month 6:    Data analysis + manuscript integration
```

---

## Collaboration Pitch (Email draft)

```
Subject: Collaboration on KIF21B isoform switch validation in AD — computational prediction seeking experimental confirmation

Dear [Collaborator],

We have identified a compelling AD-associated isoform switch in KIF21B (kinesin-2 motor)
using PRISM, a deep learning isoform function prediction framework trained on long-read
single-cell data. The switch — from a motor-domain-intact CT isoform (PRISM microtubule
movement score: 0.975) to a WD40-repeat isoform (0.105) — was independently replicated
in the Ebbert long-read cohort (MWU p=0.026) and shows the highest CT functional score
in our entire 83-case BISECT dataset.

We believe this represents an experimentally tractable and biologically significant finding
with direct relevance to axonal transport defects in AD. We are looking for a collaborator
with iPSC → neuron differentiation capacity and MT transport assay infrastructure to
validate this prediction experimentally.

Would you be open to a 30-minute call to discuss the data and potential collaboration scope?

Best regards,
Seungwon Lee
[Institution]
```

---

## Publication Impact

With KIF21B experimental validation:
- Current: NAR ~75% (computational methods paper)
- With Experiment 1 alone: Nature Methods ~50%, Nature Neuro possible
- With Experiments 1+2: Nature Neuro ~55%, Molecular Cell ~45%

The single most impactful experiment for publication tier upgrade.
