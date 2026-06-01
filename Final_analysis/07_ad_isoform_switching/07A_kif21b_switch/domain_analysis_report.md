# KIF21B Isoform Domain Analysis Report
**2026-05-20 | DIFFUSE Project**

---

## Summary

A bidirectional isoform switch at the *KIF21B* locus in Alzheimer's disease excitatory neurons
involves two novel isoforms with opposite functional profiles — one motor-competent and one
motor-dead — identified by both sequence domain analysis and DIFFUSE functional score prediction.

---

## Isoforms

| Isoform | Category | Exons | ORF (aa) | CDS genomic span | DTU direction |
|---------|----------|-------|----------|------------------|---------------|
| KIF21B-201 (canonical) | FSM | 34 | 1625 | chr1:201,023,383–200,974,095 | CT reference |
| transcript293004.chr1.nic | NIC | 8 | 419 | chr1:201,023,383–201,003,541 | CT-dominant |
| transcript292978.chr1.nnic | NNIC | 19 | 711 | chr1:200,990,629–200,974,095 | AD-dominant |

---

## Domain Analysis

### tr293004 (CT-dominant, 419 aa)

**CDS start**: Chr1:201,023,383 — **identical to canonical KIF21B CDS start**. The protein
is an N-terminal fragment covering the first 25.8% of the canonical KIF21B sequence.

**Structural classification**: NIC (intron_retention subcategory) — the transcript uses
known splice sites but retains an intron, causing premature termination at aa 419.

**Confirmed functional motifs** (by direct sequence match):
- P-loop (GXGXXGKT): **GQTGAGKTYT** at aa 86 — ATP-binding site of kinesin motor
- Switch-I: **SSRSHA** — nucleotide sensing upon ATP hydrolysis
- Switch-II: **DLAGSE** at aa 272 — DxxG motif that triggers powerstroke

**Domain coverage** (~canonical coordinates):
- Kinesin motor domain (aa 1–360): COMPLETE
- Neck linker (aa 361–382): COMPLETE
- Coiled-coil stalk (aa 383–419): 37/808 aa (stub only)
- WD40 cargo-binding domain: ABSENT

**Conclusion**: tr293004 is a **motor-competent isoform**. It retains all three catalytic
motifs required for microtubule binding, ATPase activity, and mechanical force generation.

---

### tr292978 (AD-dominant, 711 aa)

**CDS start**: Chr1:200,990,629 — **32,754 bp downstream** of canonical CDS start (minus strand).
This places the translation initiation in the middle of the canonical coiled-coil stalk region.

**Structural classification**: NNIC (at_least_one_novel_splicesite) — entirely novel
splicing with ≥1 splice site not in any reference transcript.

**Confirmed functional motifs** (by direct sequence search):
- P-loop: **ABSENT** — no GXGXXGKT motif in entire 711 aa sequence
- Switch-I: ABSENT
- Switch-II: ABSENT
- WD40 repeats: **WDIRDS** at aa 446+ — cargo-binding beta-propeller signature
- Coiled-coil: **LLQEAL** heptad repeat pattern — confirmed dimerization domain

**Domain coverage** (~canonical coordinates aa 1040–1625):
- Kinesin motor domain: ABSENT (genomic region not included)
- Neck linker: ABSENT
- Coiled-coil stalk (C-terminal half, ~aa 383–1190 canonical): PARTIAL (300 aa)
- WD40 cargo-binding domain (~aa 1191–1560): PRESENT (365 aa)

**Conclusion**: tr292978 is a **motor-incompetent isoform**. It lacks all kinesin motor
domain motifs, retains the coiled-coil dimerization domain and WD40 cargo-binding domain.

---

## DIFFUSE Functional Scores (v15d_bp_clean)

| Isoform | MT-based mvt (GO:0007018) | Neuron proj dev (GO:0031175) | Synaptic trans |
|---------|--------------------------|------------------------------|----------------|
| KIF21B-201 (canonical) | 0.9497 | 0.2627 | 0.2351 |
| tr293004 (CT-dominant) | **0.9664** | 0.1098 | 0.0427 |
| tr292978 (AD-dominant) | **0.1112** | 0.2003 | 0.1731 |

The DIFFUSE model independently predicts the motor domain presence/absence without access
to the sequence domain analysis. Δ MT-based movement score: 0.9664 − 0.1112 = **−0.855**.
This is one of the largest within-gene functional score divergences in the brain dataset.

---

## DTU Statistics (Excitatory neurons, Samsung AD IsoQuant)

| Isoform | CT usage | AD usage | chi-sq p | Cell types affected |
|---------|----------|----------|----------|---------------------|
| tr293004 | 35.1% | 0.0% | **9.28×10⁻⁸** | Excitatory only (1/8) |
| tr292978 | 0.0% | 42.9% | **3.81×10⁻⁶** | Excitatory only (1/8) |
| Canonical (combined) | 44.1% | 7.1% | — | — |

---

## Mechanistic Hypothesis

**AD-specific locus switch**: In control excitatory neurons, tr293004 (motor-competent)
co-exists with full-length canonical KIF21B, together constituting ~79% of KIF21B usage.
In AD excitatory neurons, tr293004 is completely absent and tr292978 (motor-incompetent)
accounts for 43% of usage — larger than canonical.

**Dominant-negative model**: tr292978 retains the WD40 cargo-binding domain and
coiled-coil dimerization domain. Given that KIF21B functions as a dimer (via its coiled-coil),
tr292978 could:
1. Form heterodimers with full-length KIF21B, tethering the functional motor partner
2. Compete with full-length KIF21B for dendritic cargo binding via WD40
3. Neither mechanism requires motor activity — the net effect is cargo immobilization

**Connection to AD pathology**: KIF21B participates in anterograde dendritic transport of
synaptic cargo (AMPA receptors, mRNPs). Loss of motor-competent isoform and gain of
motor-dead cargo-binding isoform predicts reduced dendritic transport efficiency —
consistent with known axonal/dendritic transport defects in early-stage AD excitatory
neurons (Stokin et al., *Science* 2005; PMID: 15731448).

---

## Key Validation Needed

1. **Proteomics**: Confirm tr292978 protein presence (711 aa) in AD prefrontal cortex
2. **Co-IP**: Test whether tr292978 co-immunoprecipitates with full-length KIF21B-201 (heterodimer formation)
3. **Transport assay**: Express tr292978 in neurons, measure dendritic AMPA receptor transport
4. **Independent cohort**: Replicate tr293004/tr292978 switch in second AD long-read dataset

---

## Files

- Figure: `fig_07A_kif21b_domain_analysis.pdf/png`
- Protein sequences: extracted from SQANTI3 `isoforms_corrected.faa`
- Classification: SQANTI3 `isoforms_classification.txt` rows 20981, 20982
