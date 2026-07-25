# PRISM Encoding–Representation–Label Map: A Rigorous Framework

> Working design document (not manuscript). Goal: a defensible, measurable map of what
> ESM-2/PRISM encodes, surfaces, uses, and cannot describe at isoform resolution — organised
> so that every category has (a) an established metric, (b) a null/oracle, and (c) a link to
> the biological prior AND the architectural mechanism simultaneously.
>
> Numbers cited from prior sessions are flagged [prior] and must be re-verified before any
> manuscript use.

---

## 1. Why a 2-D framework, not a 1-D tier list

The user's instinct — split into **encoding axis / representation axis / label-contribution** —
is correct but under-resolved. Two refinements make it rigorous:

**Refinement 1 — "representation" hides three distinct bottlenecks.** A per-residue signal can be
(i) destroyed by the *pooling operator*, (ii) preserved as a *magnitude* but lack a *reference
direction (anchor)* to score against, or (iii) present in the input vector but *not used* by the
readout head. These are mechanistically different failure modes with different fixes, and the
project has already dissociated them empirically (region-pool raises coherence but lowers DR-AUC;
axis3-domain is *used* while axis0-disorder is *encoded but not used*). Collapsing them into one
"representation" layer reproduces exactly the magnitude/direction conflation found in §6b above.

**Refinement 2 — interpretability is orthogonal to information flow.** A feature can be
encoded + surfaced + used + labelled and *still* not be describable by our 8 interpretable axes,
because the 8 axes are a **linear lower bound**: for domain-change decoding, 8 axes reach
AUROC 0.715, the full 640-dim linear probe 0.838, and a non-linear probe 0.890 [prior]. So
"what the model uses" strictly exceeds "what we can currently narrate." This gap is itself a
mapped territory (a future-work bucket), not a blank.

Hence the map is a **2-D grid**:

- **Vertical — information-flow cascade (B0→B5):** where, along the sequence→score pipeline, does
  the isoform difference get lost? Information about a *fixed* isoform difference is monotone
  non-increasing along a *single* extraction path (B0→B1→…→B5): each stage is a deterministic
  function of the previous, so it cannot manufacture information the previous stage discarded.
  (Devils-advocate attack 7: this is per-path monotonicity, not a claim that a later stage always
  carries less *total* signal than an earlier one across different edits or that pooling/anchoring
  are information-lossless — B2 and B3 are precisely the lossy stages the map localises.)
- **Horizontal — description resolution:** given that something survives to a given bottleneck,
  how finely can we characterise it — by 8 axes / by full-640 linear / by non-linear only /
  undescribable?

Domain and non-domain edits are two **traces** through this grid.

---

## 2. The information-flow cascade (vertical axis)

Architectural pipeline:
```
sequence  →  ESM-2 per-residue {h_i ∈ ℝ⁶⁴⁰}  →  mean-pool → (φ_L30, δ_layer=φ_L30−φ_L15)  →  MLP head g → GO scores
             └── B1 encoding ──┘               └── B2 pooling ──┘  └─ B3 anchor ─┘ └─ B4 usage ─┘        └─ B5 label ─┘
```

> **§14 (2026-07-23) empirically refines B1 and B2b below.** B1 is NOT a single lossless box:
> layer-resolved probes show feature-specific peak depths (disorder ~L2, domain ~L9) plus a shared
> late-layer erosion. B2b is NOT an O(k/L) low-pass: per-residue re-extraction shows it is a
> **spatial-coherence filter** (coherent shifts survive, incoherent spread is averaged out). See §14.

| Bottleneck | Question | Metric (established) | Null / oracle |
|---|---|---|---|
| **B0 Physical** | Does the splice change the sequence? | changed-residue count, Pfam-envelope overlap | — (ground truth) |
| **B1 Encoding** | Is the difference in ESM-2's per-residue tensor *at all*? | junction δ_p magnitude & decay; per-residue probe AUROC | length-matched scrambled residues |
| **B2 Pooling** | Does mean-pool preserve it into the fixed vector? | region-pool vs mean-pool recovery of direction/decodability | mean-pool = chance floor for the class |
| **B3 Anchor** | Is there a learnable common *direction* to score against? | cross-gene direction-consistency (CV-dir-acc) | **gene-permutation null** (this session's tool) |
| **B4 Usage** | Does the MLP head actually rely on it? | ridge test-time reliance; occlusion-retrain (variance-corrected) | shuffled-axis / redundant-axis control |
| **B5 Label** | Does an isoform-level label exist to reward it? | label-confidence (GO-domain log-odds); describable fraction | gene-mean oracle / label-identity oracle |

Two hard-won discipline rules attach here (from the project's negative results):

- **B3 needs the gene-permutation null, not sign-shuffle.** Cross-gene "common anchor" claims are
  only valid against gene-ID permutation; the orient=+1 self-consistency artifact and the
  sheet_delta brain fold-universality both masqueraded as signal under weaker nulls.
- **B4 must use ridge reliance, not coherence-retrain occlusion.** Coherence-retrain occlusion is
  variance-confounded (it tracks axis variance, not usage); the decisive counter-example was
  axis5 (low variance, high disorder-encoding, ~1% occlusion effect).

---

## 3. Description resolution (horizontal axis)

For whatever survives to B4 (used by the model), how well can we *narrate* it?

| Resolution tier | Operational definition | Domain-change value [prior] |
|---|---|---|
| **R1 — 8-axis interpretable** | linearly decodable from the 8 joint-PCA axes | AUROC 0.715 |
| **R2 — 640-linear (not 8-axis)** | decodable by full-640 linear probe beyond the 8 axes | 0.715 → 0.838 |
| **R3 — non-linear only** | recovered only by a non-linear probe | 0.838 → 0.890 |
| **R4 — used but undescribable** | contributes to prediction yet escapes all probes above | 0.890 → (representational ceiling) |

Converting to "share of the decodable signal" needs an *additive* measure — AUROC differences are
not (AUROC is a rank statistic; C4). Under a common logistic decoder for domain-architecture change
(`case_study_interpretability.py`, brain canonical-anchored pairs, n=33,802), the deviance-based
**McFadden pseudo-R²** is 0.101 for the 8 axes and 0.155 for the full 640-dim linear probe, so the
8 axes capture **65.5% of the linearly-decodable domain deviance**. (This additive re-derivation
lands almost exactly on the earlier, indefensible AUROC-difference arithmetic ≈64% — reassuring that
the *qualitative* "~2/3 narratable" holds, but only the deviance ratio is a defensible number.) The
non-linear probe adds a further slice (R3). So even the best-understood class (domain) is only ~2/3
narratable by our current interpretable coordinates — worth stating plainly rather than implying the
axes are the whole story. *(AUROC on this set: 8-axis 0.750, 640-dim 0.779; the manuscript's
length-matched decile classifier gives 0.715/0.838 on the full within-gene pair enumeration — a
different pair set, reported there, not replaced here.)*

---

## 4. The two traces (the tree the user asked for)

### 4a. DOMAIN-change edits  (brain 69.8% / muscle 53.5% of within-gene splices [prior])

```
B0 present   ── Pfam count / envelope changes
B1 ENCODED   ── strong & global; L30 probe AUROC 0.860, per-residue ~0.99 [prior]
B2 SURVIVES  ── large contiguous coherent shift; mean-pool retains it
                (centroid-similarity on mean-pool alone gives DR-AUC 0.638 [prior])
B3 ANCHOR ✓  ── domain-completeness is a single common learnable direction
                (axis3, length-independent, within-gene ρ≈0.27; BISECT concordance 0.898 [prior])
B4 USED ✓    ── axis3 ridge reliance positive; DR-AUC 0.630 (muscle) / 0.775 (brain) [prior]
B5 LABEL ~   ── PARTIAL: Type-1 domain-loss labels are 89.9% low-confidence (gene-level
                inheritance onto domain-lost isoforms) [prior]
CEILING      ── set by LABEL NOISE + representational saturation, NOT by encoding.
                Capacity / within-gene supervision / layer-selection all fail to raise 0.630.
Description  ── R1 0.715 → R2 0.838 → R3 0.890 (8 axes = 65.5% of linear domain DEVIANCE,
                McFadden R² 0.101/0.155; the additive C4 measure, ≈ the old AUROC-diff 64%)
```
→ Domain is **"solved to the representation, capped by labels."** The residual here is an
*annotation* limit, provable because ESM-2 places Type-1 isoforms 2.89× farther from canonical
than Type-3 [prior] — it encodes what the labels do not express.

### 4b. NON-DOMAIN edits  (brain 30.2% / muscle 46.5%)

This class does **not** fail at one bottleneck — it fractures into sub-mechanisms that fail at
*different* bottlenecks. That heterogeneity is the whole point.

> **Read the rows below as overlapping features, not an exhaustive/exclusive partition (C6).**
> Targeting, disorder, and compositional descriptors are independent regressors, so a single
> edit can be simultaneously N-terminal, disorder-region, and helix-rich; the per-bucket shares
> in §5 are therefore *nested* (domain → non-domain → within-non-domain descriptors), not summed
> side by side. And B2 (pooling) → B3 (anchor) are **sequential-conditional stages** (B3 acts only
> on what B2 preserves), dissociated by the domain × pooling interaction — not two independent
> bottlenecks (C1).

```
                                    B1 enc.  B2 pool  B3 anchor        B4 use  B5 label
N-terminal targeting (MTS/signal)   yes      partial  partial(MTS ρ.126, weak, gene-perm caveat)  ~       almost none
Disorder-region edits               yes(.79) diffuse  none(diffuse,low-amp; §6b β<0)              ~       none
Compositional (helix/sheet/…)       yes      partial  median-split usable (CV .59–.65, THIS SESSION) ?    none
Short linear motifs (SLiM)          barely   LOST     none (mechanism-heterogeneous)              no      none
PTM-state–dependent                 site-yes n/a      n/a (state is extrinsic to sequence)        no      none
```

Key transitions and where each dies:
- **SLiM / motif**: lost at **B1→B2**. Encoded locally (junction shift 3.26, 17.6× decay [prior])
  but a 3–10 aa perturbation in an IDR is averaged out by 1/L mean-pooling. Region-pool recovers
  *coherence* (0.481→0.739 [prior]) but this is embedding self-consistency, **anti-correlated**
  with labelling (region-pool *lowers* DR-AUC) — dies again at **B3** (no common anchor).
- **Disorder-region**: survives B1 (decodes at AUROC 0.79) but as a **diffuse low-amplitude**
  shift (§6b: disorder β<0 after size control) — effectively no scorable direction (B3).
- **Compositional (this session; RESOLVED in §9)**: helix/sheet/hydro/charge give a *median-split*
  direction that is **real** (label-permutation drops it to 0.5, e.g. brain sheet 0.653→0.500) but
  **gene-INDEPENDENT** — real CV-dir-acc sits at/below its gene-permutation null in all cases, so it
  is a dataset-wide long/short regularity, **not** a cross-gene anchor (the earlier "survives the
  gene-perm null" reading was a verdict-logic bug; see §9). Its **B4 usage** was then computed and is
  **negative** — the direction attenuates to the gene-perm null at PRISM output (`b4_compositional_
  usage.py`), and its **B5 label** is absent. So this bullet is no longer a "live frontier": it is
  **encoded / weakly-surfaced-as-dataset-wide, un-anchored, un-used, un-labelled**.
  *(C5: computed on the domain_binary==0 subset so domain change cannot proxy it, and sheet_delta ⊥
  axis0 (r=−0.089); a within-subset size_z partial was the planned confirmation but is now moot given
  the §9 B4-negative placement.)*
- **PTM-state**: the fundamental floor — sequence encodes the site, not the modification state;
  no sequence model can cross B3/B5 here.

→ Non-domain is **"encoded but not surfaced, and where surfaced, un-anchored and un-labelled."**
This is the describability gap, now resolved into *which* bottleneck each sub-mechanism dies at.

---

## 5. Quantification with provenance (the "몇 %" the user wants)

Denominator = within-gene splice events (isoform-resolution population). Numbers [prior]; confidence flagged.

```
Brain within-gene splices  (100%)
├─ 69.8%  DOMAIN-affecting              [FIRM: genome-wide rate, §6]
│         ├─ used by model: DR-AUC 0.775 vs 0.500 null
│         ├─ 8-axis-narratable: 65.5% of linear domain deviance (McFadden R², additive; C4)
│         └─ ceiling cause: label noise (89.9% Type-1 low-conf), not representation
│
└─ 30.2%  NON-DOMAIN                    [FIRM: genome-wide rate]
          measured composition (brain, n=25,262; `nondomain_mechanism_fractions.py`):
          ├─ 51.5%  N-terminal targeting (muscle 63.0%) -> encoded, weak MTS alignment ρ=0.126 at
          │         EMBEDDING (B1/B3), but B4 NOT used (`b4_nterm_usage.py`: charge/helix/hydro
          │         targeting-orient B4 sits at gene-perm null, same as compositional) [FIRM]
          ├─ 48.1%  structured-internal -> of which specific-SLiM-bearing ~54.6% (pooling-lost, B2);
          │         compositional signal real but GENE-INDEPENDENT & B4 NOT used (§9) [FIRM]
          └─  0.4%  disorder-DOMINANT (>0.5) — negligible as a stream; disorder is an overlapping
                    PROPERTY (any-overlap encoded AUROC 0.79, diffuse/no-anchor, §6b β<0), not a bucket
```
> **Rows overlap (C6):** the exclusive 3-way partition above (N-terminal / structured / disorder-dom)
> is for the flow widths; the underlying descriptors are non-exclusive (an N-terminal edit can also be
> disorder-rich), so shares are nested, never summed. **Correction from the measurement:** the first
> draft treated disorder as a co-equal sub-mechanism — it is not; disorder-DOMINANT edits are <1%, and
> N-terminal targeting is by far the largest single non-domain class (~half of all non-domain edits).

**Resolved since first draft (§9):** the compositional question — the one flagged here as "not yet a
number" — has been computed. The direction is real but gene-independent (not a cross-gene anchor) and
is **not propagated to PRISM's output** (B4 negative in both tissues). So this slice of the 30.2%
non-domain residual is **encoded, weakly surfaced as a dataset-wide orientation, un-anchored, un-used,
and un-labelled** — it *reinforces* the describability gap rather than opening recoverable signal.

**The R2/R3 tail number, now computed (`nondomain_residual_decomposition.py`, Option A, brain
n=11,666 non-domain pairs).** Decomposing the non-domain embedding-difference (1280-dim, centered)
variance by gene-disjoint reproducibility (held-out variance captured by the train top-K PC subspace,
in excess of a random-K-subspace null, K=50):

```
Non-domain embedding-difference variance (100%, centered)
├─ 45.3%  per-pair NOISE FLOOR      — not gene-reproducible = the true describability floor
└─ 54.7%  gene-REPRODUCIBLE structure (future-recoverable in principle)
          ├─ 18.4%  named by the 4 compositional descriptors (helix/sheet/hydro/charge)
          └─ 36.3%  reproducible-but-NON-compositional  ← the R2/R3 tail (future work)
                    (includes size/position geometry partly nameable by other covariates,
                     plus genuinely un-named structure)
```
Domain-change pairs are the positive reference: 66.1% reproducible / 33.9% floor (more structured,
less noise, as expected). **Key reading:** the non-domain describability gap is *not* dominated by an
information floor — roughly **half** the signal is gene-reproducible structure, of which our current
descriptors name only **~34%** (0.184/0.547). So most of the gap is *structure we can see is
reproducibly there but cannot yet name*, i.e. a missing-descriptor / missing-label problem, not
irreducible noise. Caveats: the reproducible fraction grows with K (0.29 at K=4 → 0.55 at K=50) so the
split is reported at a K=50 structured-dimensionality cutoff; and the compositional subspace here
excludes size/disorder/nterm, so part of the 36.3% tail is nameable by those covariates rather than
truly novel. The dataset-wide mean(long−short) orientation is reproducible (cv-dir-acc 0.664) but
carries only 1.5% of raw energy — consistent with the compositional "weakly-surfaced dataset-wide"
placement (§9).

**Tail narrowed + floor probed (`nondomain_tail_and_floor.py`, brain non-domain).**
- *Option A — how much of the tail is truly un-named?* Removing the **full 7-covariate subspace**
  (compositional + size + disorder + resync — every covariate that varies inside the non-domain
  subset) leaves a covariate-orthogonal residual whose reproducible structure is still large (K=50
  excess **+0.427**, versus the original +0.547; the 7 covariates jointly capture only +0.252). So
  the reproducible non-domain structure is **high-dimensional and mostly outside our hand-crafted
  descriptor basis** — the R2/R3 tail is genuinely large, not an artefact of leaving size/disorder
  out. Caveat: "un-named" = outside these 7 descriptors, which could still include other nameable
  properties (full AA composition, secondary-structure content), so this is an invitation to richer
  descriptors, not proof of deep un-nameability.
- *Option A refinement — how much of the tail is beyond amino-acid COMPOSITION?* (`nondomain_tail_
  rich.py`) The full 20-dim amino-acid composition of the changed residues (which linearly dominates
  helix/sheet/hydro/charge) names **+0.335** of the centered variance — ~61% of the reproducible
  0.547, far more than the coarse 4 (0.185) or the 7 covariates (0.252). But a real reproducible
  residual (**+0.387**) survives full composition, so the deepest tail is **structural / positional /
  contextual, genuinely beyond amino-acid composition** — this is the true future-work core, smaller
  than the raw tail but not an artefact of coarse descriptors.
- *Option B — is the 45% floor a pooling artefact or genuine noise?* If mean-pooling merely diluted a
  fixed local signal, the reproducible fraction would **rise** with edit changed-fraction (larger
  edits diluted less). It does not: Spearman(changed-fraction, reproducible-excess) = **−0.40 (n.s.,
  5 strata)**, and the *smallest* edits (Q1, 2.3% changed) are the *most* reproducible (floor 33%)
  while larger edits sit at floor 55–61%. The floor therefore is **not** simple pooling dilution —
  larger, more heterogeneous edits are more per-pair idiosyncratic (genuine biological diversity),
  so the floor will not be recovered by swapping the pooling operator alone. Caveat: the two effects
  (dilution↑, heterogeneity↓) could partly cancel; the firm conclusion is only that there is **no net
  positive size trend**, ruling out the simple recoverable-by-pooling story.

**Four terminal buckets** (the user's requested end-categories), now precisely defined:

| Bucket | Definition (bottleneck signature) | Domain | Non-domain |
|---|---|---|---|
| **Explained & used** | survives B1–B4, has B5 label | DR 0.630/0.775, label-capped | **~none** (N-terminal B4 downgraded — b4_nterm_usage.py: not used at output) |
| **Encoded, not surfaced** | dies B2 (pooling) | ~none | SLiM, part of disorder |
| **Surfaced, un-anchored/un-labelled** | survives B2, dies B3 or B5 | (label-noise tail) | compositional, disorder, internal |
| **True black box (not encoded)** | dies B1 | ~none | PTM-state, motif-*function* |
| **Below ceiling, unexplained (FUTURE)** | used (B4) but description-tier R4 | R2+R3 gap (non-linear domain tail) | decodable-but-non-compositional non-domain residual (compositional now resolved as B4-negative, §9) |

---

## 6. Concrete instrument: the isoform case study

To make the grid legible, decompose *individual* pairs onto the two coordinate systems
simultaneously:
- **8 axes** (unsupervised encoding geometry): project δ = φ(long)−φ(short) onto W(8×640).
- **7 covariates** (supervised edit descriptors): domain, size, disorder, helix, sheet, hydro, charge.

Then the "explainer" asks, per pair: *which axes carry this pair's shift, and do the covariates
predict that axis profile?* The gap between covariate-predicted and actual axis-profile = the
pair's R2+R3+R4 residual (represented but not covariate-describable).

**Diagnostic triple** (spans the grid), *a-priori expectation*:
- **Type A — NDUFS4** (MTS-exon / domain loss): expected B1–B4 all pass, R1 high. The "everything works" reference.
- **Type B — MAPT 3R/4R** (N-terminal insert count, no domain change): expected die at B2/B3, near-zero score gap. The "encoded-but-lost" reference.
- **Type C — LRPPRC** (same-domain, CT=AD score): B1 may encode, but B4/B5 flat. The negative control.

### 6a. Case-study RESULT (`case_study_interpretability.py`, brain; 2026-07-22)

Per-pair projection onto the 8 axes (Δ in population-SD units) + the edit covariates + PRISM's
output shift. The a-priori triple is **partly refined (not refuted) — and the refinement is the finding.**
(Devils-advocate attack 6: "refuted" overstates it. NDUFS4's re-classification and MAPT's output
movement do not break the cascade; they *sharpen* which stage each pair terminates at. Reframed below.)

| gene | edit (aa) | domain_binary | nterm | top axes (SD) | |ΔA|Σ | PRISM |ΔScore|₁ / #terms>0.05 |
|---|---|---|---|---|---|---|
| NDUFS4 | 219–464 | **0** | 1 | ax6 −4.1, ax2, ax3 | 6.7–12.3 | 11–54 / 51–435 |
| MAPT | 384–620 | **0** | 0–1 | ax3 +2.7, ax2, ax1 | 7.4–11.8 | 17–43 / 83–330 |
| LRPPRC | 801–1166 | **0** | 0–1 | ax6, ax3, ax2 | **4.9–5.7** | 18–23 / **95–152** |

Three corrections to the framework, forced by the data:
1. **NDUFS4 is not a Pfam-domain case.** Its MTS-exon loss scores `domain_binary=0, nterm=1` — a
   **N-terminal targeting** edit, not a domain-architecture change. So it belongs to the §4b
   *targeting* row, not §4a. The "domain loss" label was a biological gloss; at the Pfam/covariate
   resolution the map uses, NDUFS4 is non-domain. (This is itself a describability lesson: the
   biologically meaningful loss — mitochondrial import — is invisible to the domain covariate.)
2. **MAPT is not B2/B3-silent.** Its large edits move PRISM's output substantially (up to 330/672
   terms), revising the "encoded-but-lost, near-zero score gap" expectation. But this *confirms*
   B1/B2 pass for these edits (the signal is encoded and survives pooling into an output reaction) —
   it does not contradict the cascade; it relocates MAPT from "silent at B2/B3" to "reacts at B1/B2,
   un-anchored at B3". Output *magnitude* tracks edit *size*, not domain status (confirmed
   quantitatively: `b4_magnitude_usage.py`, incremental compositional R² beyond size = +0.003) — the
   model reacts by size; whether it reacts *directionally/correctly* (B3/B5) is the separate question,
   and the B4 magnitude test shows it does not use composition beyond size. **Tissue-general**
   (`b4_magnitude_usage_muscle.py`): muscle replicates brain — size R²=+0.188, incremental
   composition beyond size = +0.001 (brain +0.003). "Output = pure edit-size reaction" holds in both.
3. **LRPPRC partially validates as the low-perturbation control** — despite the *largest* edits
   (>1000 aa), it has the *smallest* total axis displacement (4.9–5.7 SD) and fewest PRISM terms
   moved. Same-family PPR-repeat edits move the representation least per residue: consistent with a
   weak-perturbation control, though not literally flat.

**Map lesson from the triple:** at the individual-pair level the model's *output magnitude* is
driven by edit size and is a poor read-out of the domain/non-domain distinction; the encoding axes
(|ΔA| profile) separate the cases better (LRPPRC lowest), but the covariates do **not** cleanly
predict which axes carry each pair — the per-pair axis profile is heterogeneous (ax6/ax2/ax3 rotate
across pairs of the same gene), i.e. most of each pair's shift sits in the R2+R3 residual the 8
axes + 7 covariates do not narrate. This is the describability gap made concrete on single isoforms.

This makes the abstract cascade a concrete, per-isoform readout — and is directly extensible to
any BISECT case.

---

## 7. Immediate manuscript consequence

The §6b conflation (magnitude OLS vs direction CV-dir-acc) is an instance of collapsing B2/B3
into one "representation" layer. The framework dictates the fix: **report the two analyses as two
bottlenecks, not one regression** —
- keep the **5-covariate OLS** (size, domain, nterm, disorder, resync) as the *magnitude* result
  (B2-level: how large a representational shift), matching S_severity;
- add the **4 compositional covariates** as a *separate* non-domain-residual *direction* result
  (B3-level), with its own subset, CV-dir-acc, and the gene-/label-permutation nulls.

This preserves S_severity (no rewrite of the OLS) and prevents the cross-file contradiction.

---

## 8. Devils-advocate revision (2026-07-22)

Framework referred to devils-advocate before any B4 computation (anti-local-minima: new
framework proposal). Six attacks C1–C6 returned; adjudicated on scientific-depth grounds (the
researcher decides direction, the critique flags logical holes only). Net: framework survives
with **three revisions, one refutation of the critique, and one mandatory pre-computation gate.**

**C1 — B2/B3 tautology? PARTIALLY ACCEPTED (critique's logic reversed).** The critique argued
region-pool moving coherence↑/DR↓ in *opposite* directions means one phenomenon, two faces. This
is backwards: a single knob driving one underlying quantity moves both readouts the *same* way;
*dissociation* (opposite directions under one manipulation) is the strongest evidence of
separateness. Valid residual point: B2 and B3 are **sequential-conditional** (B3 acts only on what
B2 preserves), not orthogonal. → Revise language: "sequential-conditional stages, dissociated by
the domain × pooling interaction," not "independent bottlenecks." Clean single-manipulation
separation (a pooling op that moves coherence but not label-alignment) noted as future test.

**C2 — δ_layer violates monotonicity? REFUTED; framework strengthened.** δ_layer = φ_L30 − φ_L15
is a contrast of two *pooled* vectors at different depths; it does not undo residue-averaging, it
accesses the **depth axis**, a separate information dimension. Pooling occurs per-layer; δ_layer
bypasses final-layer integration, not the 1/L residue average. → **Split B2 into B2a (depth/
layer-collapse, δ_layer-recoverable = manuscript tier ii) and B2b (residue-averaging, motif-
destroying = tier iii, unrecoverable).** Monotonicity holds *per fixed extraction path*; δ_layer
is a different path, not a counterexample. This aligns the framework with the manuscript's
existing tier ii/iii distinction.

**C3 — ridge reliance also variance-confounded? ACCEPTED as a mandatory gate.** Indirect evidence
favours ridge (coherence-occlusion and ridge *disagreed* on axis0 — if both tracked variance they
would agree), but no *direct* control exists. → **GATE before any B4 computation:** compute
Spearman(axis variance, ridge reliance) across the 8 axes and a variance-matched synthetic-axis
null. If reliance tracks variance (ρ ≳ 0.5 or synthetic ≈ real), B4 has no valid instrument and
the map truncates at B3. This must pass before the compositional-B4-usage computation.

**C4 — 64% = AUROC linearisation? ACCEPTED.** AUROC is a rank statistic; its differences are not
additive information quantities, so (0.715−0.5)/(0.838−0.5) is indefensible as a "fraction of
signal." → Recompute with explained-deviance / log-loss ratio under a common linear decoder, or
retreat to qualitative ("8 axes capture most but not all of the linearly-decodable domain signal").
Drop the "64%" figure until recomputed on an additive measure.

**C5 — compositional a domain_binary proxy? LARGELY REFUTED (critique missed the design).** The
compositional CV-dir-acc was computed on the domain_binary==0 subset — domain_binary is *constant*
in the tested population and cannot be proxied. sheet_delta⊥axis0 already shown (r=−0.089). Valid
residual: within-subset size_z partial not yet run; and "edit inside vs outside a retained domain"
is a distinct covariate worth checking (not the domain_binary the critique named). → Add size_z
partial as confirmation; note it is a within-non-domain composition signal by construction.

**C6 — non-domain sub-mechanisms exhaustive/exclusive? ACCEPTED.** targeting/disorder/compositional
are overlapping features (independent regressors), not a partition; an MTS loss can be simultaneously
helix-rich. → Present §4b as a **hierarchical Venn decomposition** with explicit "not an exhaustive
partition" caveat; the "% per bucket" must be nested (domain 69.8% → non-domain 30.2% → N-term X% →
of which disorder Y% → compositional-residual Z% → pure-SLiM <1%), not summed side-by-side.

**Revised cascade (post-critique):**
```
B0 physical → B1 encoding → B2a depth-collapse → B2b residue-averaging → B3 anchor → B4 usage → B5 label
                            (δ_layer-recoverable)  (motif-destroying)    (sequential-conditional on B2b)
```
**Gate before B4 work:** C3 variance-vs-reliance control. **Writing fixes:** C4 (drop 64%), C6
(Venn), C1 (language). **Cheap add-on:** C5 size_z partial.

### 8b. C3 gate result (2026-07-22) — CONDITIONAL PASS

`c3_gate_variance_vs_reliance.py` (reuses devils_c4_ridge_reliance occlusion; 8 real axes + 40
random dirs; domain_binary & disorder_frac targets; muscle + brain).

**Passes (robust):**
- Metric is not mechanically variance-driven: across 40 random directions Spearman(captured-var,
  reliance) ≈ 0 (muscle −0.19/+0.08, brain −0.06/−0.02, all n.s.).
- Within-data variance control via the used/not-used dissociation running *against* variance:
  axis3→domain has BELOW-median variance (11.0/9.5) yet the highest cross-tissue reliance
  (+50.4%/+11.0%), while axis0→disorder has HIGHER variance (13.4/11.1) yet negative reliance
  (−3.4%/−0.8%). Higher-variance axes 0/1/2 (var 13–16) carry near-zero domain reliance. Variance
  rank ≠ reliance rank ⇒ the instrument measures usage, not variance. Replicates in both tissues.

**Blemishes (block a clean pass):**
- **axis6 muscle anomaly:** highest variance (26.5) AND highest muscle domain reliance (66.6%),
  but does NOT replicate in brain (3.1%); axis3 replicates (50→11%). axis6 is a domain-family axis
  (KRAB-ZNF/spectrin) so its muscle signal may be genuine, but non-replication + max variance =
  can't exclude partial variance inflation. This single axis raises T1 rho(8 real axes) to 0.43–0.48
  (below the 0.5 threshold but close).
- **T2 void:** random unit directions capture ~0.9 variance vs real axes' 5–26 — no overlap, so the
  variance-matched random null could not be constructed (fell back to n=0). T2 provided no
  information; the real variance control is the cross-axis comparison above.

**Verdict & consequence:**
- Ridge reliance is usable as the B4 instrument, but usage claims require a **strengthened
  criterion: exceed the random band AND replicate across tissues** (guards the axis6 failure mode).
- To fully close T2, a proper variance-matched null must draw random directions from the 8-axis
  span / top-PC subspace (high captured variance, target-unaligned); this directly adjudicates the
  axis6 anomaly. Recommended before per-axis high-variance usage claims; the flagship axis3 claim
  already survives via the cross-axis + cross-tissue evidence.
- For the compositional-covariate B4 test: run in BOTH tissues; count only cross-tissue-consistent
  usage.

### 8c. T2 proper variance-matched null (2026-07-22) — verdict revised, MUSCLE confound found

`c3_gate_T2_variance_matched_null.py`: high-variance null built from random unit combinations of
the 8-axis span (captured var 4–29, now matched to real axes' 5–26). Fit reliance ~ captured_var
across combos; place each real axis as a studentized residual. Target domain_binary, both tissues.

**This proper null revised the §8b "conditional pass" — partially vindicating devils-advocate C3:**
- **axis6 anomaly RESOLVED as variance inflation:** studentized muscle +1.13 / brain +0.07 = on-trend
  both tissues. Its high muscle domain-reliance (0.083) is exactly what its variance (26.5) predicts.
  Not a genuine domain signal — the cross-tissue guard correctly flagged it.
- **MUSCLE has a genuine secondary variance→reliance trend** (r=+0.247, **p=0.027**), and axis3 sits
  only +1.24 SD above it (within the resid_sd=0.032 scatter) = **on-trend, not clearly above.** The
  §8b "rho≈0 ⇒ pass" was based on LOW-variance random dirs and missed this high-variance-regime
  confound. The proper null changed the muscle verdict.
- **BRAIN is clean:** axis3 studentized +3.06 (above trend), no significant variance trend (r=0.118 n.s.).
- **Caveat on this null:** the span-combos include axis3 in the basis, so the null contains scaled
  copies of the flagship — this inflates resid_sd and makes the muscle +1.24 a LOWER bound. An
  axis3-EXCLUDED null (random combos of the other 7 axes) is needed to separate null-contamination
  from a genuine muscle confound.
- **Decisive evidence the instrument is NOT pure variance:** axis0→disorder reliance is NEGATIVE in
  both tissues despite axis0's high variance (13.4/11.1). Pure variance would force positive. So the
  metric measures target-specific usage; the confound is a secondary additive trend, not domination.

**Revised gate verdict:** ridge reliance measures usage (not just variance), but carries a
**tissue-dependent secondary variance inflation — significant in muscle, absent in brain.**
Consequences for B4:
1. **Brain is the primary tissue for usage claims** (clean); muscle per-axis usage magnitudes are
   upper bounds and require cross-tissue replication.
2. Before finalising any per-axis usage magnitude, close the muscle ambiguity with the axis3-EXCLUDED
   high-variance null.
3. The compositional-B4 test inherits this: report brain as primary, require cross-tissue consistency,
   and treat muscle magnitudes as upper bounds.

### 8d. Definitive T2 (flagship-excluded null, 2026-07-22) — GATE PASSES

`c3_gate_T2_flagship_excluded_null.py`: high-variance null from random combinations of the 7 axes
EXCLUDING axis3, so the null is high-variance yet ~orthogonal to the domain axis (no self-contamination).

- **muscle:** axis3 reliance 0.0626 vs clean null (mean 0.019, p95 0.057); empirical p=0.025,
  variance-matched z=+2.24 → **ABOVE (variance-independent), marginal.**
- **brain:** axis3 reliance 0.0345 vs clean null (mean 0.003); empirical p=0.000, z=+15.95 →
  **ABOVE, emphatic.**

**Resolution:** the muscle span-combo "on-trend" (§8c) was NULL CONTAMINATION — the span-combos
included axis3, so its scaled copies inflated the null band. Removing axis3 from the null basis, the
flagship domain-usage signal is variance-independent in BOTH tissues. **C3 GATE PASSES.**

**Tissue asymmetry is real (carried forward as B4 discipline):** brain axis3 towers over its null
(z=+16); muscle only marginally clears it (p=0.025; some high-variance non-domain directions reach
0.083 > axis3's 0.063). So muscle has genuine secondary variance-noise. Combined with axis0→disorder
negative reliance (high variance, negative effect — impossible under pure variance), the instrument
is validated as measuring usage, with **brain as the primary/clean tissue and muscle magnitudes as
noisier upper bounds.**

**B4 protocol (finalised for the compositional-usage computation):**
- significance = exceed the flagship-EXCLUDED high-variance null (not raw random dirs, not span-combos);
- report brain as primary; require cross-tissue consistency; muscle = upper bound;
- axis6-type high-variance single-tissue effects do not count as usage.


---

## 9. B4 compositional usage result (2026-07-22) — NOT used, and a B3 self-correction

`b4_compositional_usage.py`, method (a): same instrument (cv_dir_acc under compositional
median-split orientation) applied to the EMBEDDING difference (B3) and the PRISM score-vector
difference (B4), non-domain internal-edit subset, both tissues, gene-permutation null.

**B4 = not used at output.** Direction consistency attenuates from embedding (0.59–0.65) to PRISM
output (0.53–0.58), and the output value sits AT its gene-permutation null in all 8 cases
(4 covariates × 2 tissues). The compositional signal does not propagate to PRISM's functional
scores. This reinforces the describability gap: encoded, weakly surfaced, but not used at output,
and no label (B5).

**B3 self-correction (surfaced by the B4 reproduction of B3).** The B3 embedding CV-dir-acc sits AT
its gene-permutation null too (helix 0.616 vs 0.617; sheet 0.653 vs 0.653; all four Δ = −0.002 to
−0.009, i.e. real ≤ null). The earlier session verdict "✅ CROSS-GENE SIGNAL" (test_covariate_
gene_permutation_null.py) was a verdict-logic bug: it labelled "signal" whenever |real−null| ≥
null_std with real_lo>0.5, even when real was BELOW null — mislabelling a gene-INDEPENDENT high
baseline (the median-split orientation bias) as cross-gene signal. Correct reading: the compositional
direction is REAL (label-permutation drops it to 0.5) but GENE-INDEPENDENT (real ≈ gene-perm null) =
a dataset-wide long/short regularity, NOT a cross-gene anchor. It joins severity_score and the
N-terminal "anchor" in the same category (this session's central orient-bias theme).

**Map placement of compositional (helix/sheet/hydro/charge):**
- B1 encoded: yes (computable; embedding difference has a label-driven median-split direction).
- B2 surfaced: partially, as a gene-INDEPENDENT dataset-wide direction (not a per-gene-shared anchor).
- B3 anchor: NO cross-gene anchor (real ≈ gene-perm null); only a dataset-wide orientation.
- B4 used: NO (attenuated to the gene-perm null at PRISM output).
- B5 label: absent.
⇒ bucket = "encoded / weakly-surfaced-as-dataset-wide-direction, NOT used, NOT labelled" —
reinforces the non-domain describability gap rather than opening a new usable signal.

**Consequences:**
- Manuscript §6b compositional paragraph (commit 69d67c8) states "All four exceed their gene-permuted
  nulls in muscle" — FALSE; they sit at/below the null. Requires correction (see §10 pending).
- Memory finding-6covariate-survivors ("cross-gene signal") and finding-sheet_delta_brain_artifact
  ("muscle signal vs brain artifact") need the same correction: all four are gene-independent
  dataset-wide, real only via label-permutation; the muscle/brain asymmetry was n-driven cleanliness,
  not a signal/artifact split.
- A fully parallel B4 could add a label-permutation null on the OUTPUT (does PRISM output carry ANY
  real long/short structure under compositional orient?); the output's attenuation toward 0.5 already
  indicates this is weak.

---

## 10. Devils-advocate writing fixes applied to the body (2026-07-22)

All §8 verdicts are now folded into the body (previously they lived only in §8 as a record):

- **C1 (sequential-conditional):** §4b caveat box — B2→B3 are sequential-conditional stages
  dissociated by the domain×pooling interaction, not independent bottlenecks.
- **C4 (drop AUROC-diff 64%):** §3, §4a, §5 now cite the deviance-based **McFadden R² ratio = 65.5%**
  (0.101/0.155, `case_study_interpretability.py` Part 2). The additive number coincides with the old
  ≈64%, so the qualitative "~2/3 narratable" survives; only the deviance ratio is defensible.
- **C5 (size_z partial):** §4b compositional bullet — noted as moot given the §9 B4-negative placement
  (domain_binary constant in-subset, sheet_delta ⊥ axis0 r=−0.089 already shown).
- **C6 (Venn, not partition):** §4b + §5 caveats — non-domain sub-mechanisms are overlapping
  descriptors, shares are nested not summed.

**Case study (§6a) executed (`case_study_interpretability.py`, Option A).** The a-priori diagnostic
triple was *partly refuted by the data* (NDUFS4's MTS loss is domain_binary=0 = a targeting edit, not
a Pfam-domain case; MAPT's large edits move PRISM output, not silent; LRPPRC validated as the lowest
per-residue perturbation). Net map lesson: per-pair PRISM output magnitude tracks edit **size**, not
domain status; the 8 axes + 7 covariates leave most of each individual pair's shift in the R2+R3
residual — the describability gap made concrete on single isoforms.

**Status:** C1/C4/C5/C6 writing fixes DONE; C3 gate PASSED (§8d); B4 computed (§9); case study DONE
(§6a). The framework body is now internally consistent with its own §8/§9 findings.

## 11. Second devils-advocate pass on the finished map (2026-07-22, Option C)

After the map + figure + manuscript integration were complete, the whole map was referred to
devils-advocate a second time (anti-local-minima: "this direction is right"-type consolidation).
Seven attacks returned; adjudicated on scientific-depth grounds, not paper-polish.

**Tested-and-rejected (the strongest objection failed a decisive test):**
- **Attacks 2 & 5 — "PRISM output moves hundreds of GO terms per non-domain edit; calling it 'not
  used' is indefensible."** This conflates *directional* usage (does the edit's compositional
  orientation reach output? — already negative via cv_dir_acc) with *magnitude* usage (does output
  SIZE depend on composition beyond edit size?). New decisive test `b4_magnitude_usage.py`
  (brain non-domain, n=24,677, 8,590 genes, gene-disjoint ridge on |ΔPRISM|₁): size alone R²=+0.160;
  4 compositional alone R²=+0.004; size+composition R²=+0.162 → **incremental composition beyond
  size = +0.003**. PRISM's output reaction is a pure edit-SIZE reaction; composition adds ~nothing
  in magnitude *or* direction. **"Not used" is precise in both senses.** Objection FAILS.

**Accepted-and-softened (valid holes in the *language*, not the result):**
- **Attack 1 — "reproducible ≠ functional."** The 54.7% gene-reproducible non-domain fraction was
  written as if optimistic (future-recoverable). Correct: it is *geometric* reproducibility (top-PC
  subspace generalises to held-out genes), and most of it is composition, which is gene-independent
  and unused. Discussion softened to "geometric rather than an established functional property."
- **Attack 3 — "'not a pooling artefact' overstated."** The floor's Spearman −0.40-with-edit-size
  (n.s.) rules out a *simple monotone* dilution account but not dilution offset by heterogeneity.
  Discussion softened to "does not support a simple mean-pooling-dilution account, although dilution
  and biological heterogeneity could offset."
- **Attack 4 — "McFadden 65.5% is pair-set-specific."** True; it is computed on the canonical-
  anchored severity-pair set. S_map Panel D legend now states this explicitly.

**Framing-only (reclassified, no new computation):**
- **Attack 6 — "refuted" overstates the case study.** Reframed §6a: the triple is *refined*, not
  refuted; MAPT's output movement *confirms* B1/B2 pass and relocates it within the cascade.
- **Attack 7 — monotonicity claim.** Clarified §2: per-extraction-path monotonicity (each stage a
  deterministic function of the previous), not a cross-edit total-signal ordering; B2/B3 are exactly
  the lossy stages the map localises.

**Net:** no finding overturned. The one attack that could have overturned "not used" was tested and
failed. The rest tightened language that had drifted ahead of the evidence. The map's load-bearing
claims — domain is the only demonstrably output-used class; non-domain fractures across B2/B3/B4;
the non-domain remainder is two open problems not one recoverable gap — all stand.

## 12. Terminal decision: axis2 / axis7 NOT characterized (DROP, 2026-07-22)

The last two unidentified joint-PCA axes (axis2, evr 0.022; axis7, evr 0.013 — the smallest) were
referred to devils-advocate *before* any characterization effort (predict-before-you-look). Verdict:
**DROP.** Recorded here as a first-class negative decision, with the mechanism.

**Why naming them changes no load-bearing claim (predicted, not measured):**
- axis2 is composition-weak (global-feature max |r| = 0.26: proline/turn/instability negative) →
  it lands *in* the 36.3% reproducible-but-non-compositional undescribed tail; naming it "proline/turn
  at r=0.26" does not shrink that tail, it *is* the tail.
- axis7 has a moderate "acidic α-helix" signature (helix +0.36, acidic/charged +0.33), but helix and
  charge are exactly the compositional descriptors already shown **B4-negative** (§6a, §9;
  `b4_magnitude_usage.py`, incremental composition beyond size = +0.003 brain / +0.001 muscle). So
  axis7 is pre-judged encoded-but-not-used, joining axis0/1/4/5/6 in the "encoded-only" bucket — the
  "axis3 is the only output-used axis" claim is unchanged.
- Combined they are evr 0.022 + 0.013 = 3.5% of total variance → at most ~10% of the 36.3% tail even
  if wholly non-compositional (axis7 isn't). Rounding error, not tail-shrinkage.

**Robustness sub-check — SELF-CORRECTION (Option B, `axis_rotation_stability.py`, 2026-07-23).**
My first-pass argument here (and the devils-advocate's "rotation-mixture" mechanism) claimed axis2/7
are *rotation-unstable* because they sit in a near-degenerate low-variance block (axis3–axis7,
eigenvalue ratios ≈ 1.0–1.2). **The data refutes this.** First, W_axes is *exactly* orthonormal (max
off-diagonal 4×10⁻⁸), so axis2/7 are not linear mixtures of the identified axes — that literal confound
is impossible. Second, a split-half PCA re-extraction (fit PCA(8) on two *disjoint* halves of the 31,668
muscle-train isoforms × 30 layers, per-layer z-scored; measure per-axis max|cos| between halves) shows
**every axis direction is highly reproducible**: s_k = 0.999 (axis0), 0.998 (axis1), **0.996 (axis2)**,
0.960 (axis3), 0.960 (axis4), 0.990 (axis5), 0.989 (axis6), **0.978 (axis7)**; all subspace-overlaps
u_k ≥ 0.98. With ~950k samples the directions are pinned tight despite the small eigenvalue gaps, so
the near-degeneracy → instability inference was theoretically plausible but empirically wrong.

Two corrected conclusions, both of which *still* support the DROP but for the right reason:
1. **Direction stability is NOT the identified/grey discriminator.** The "grey" axes 2/7 (s = 0.996,
   0.978) are *as reproducible or more* than the flagship domain axis. So axis2/7 stay grey not because
   their direction is unstable but because their compositional signature is weak (axis2) or redundant
   with already-B4-negative descriptors (axis7 = helix/charge), and neither is output-used.
2. **The one genuinely rotation-sensitive axis is axis3 itself** (s = 0.960 ± 0.031, the lowest —
   because axis3/axis4 is the tightest eigenvalue pair, ratio 1.02; it rotates ~16° with axis4 across
   halves). Yet axis3's *identity* (domain/used) is the most robust of all, because it is pinned by an
   **external functional anchor** (ridge usage + BISECT concordance), not by its PCA geometry. This is
   the clean lesson: *geometric direction stability* and *identity robustness* are different properties;
   only external anchoring guarantees the latter, and eigenvalue separation guarantees neither here.

So the honest justification for leaving axis2/7 grey is composition-weakness + B4-negativity (above),
**not** rotation-instability. (This correction is working-doc-only: axis2/7 were never named in the
manuscript, so S_map/Discussion are unaffected.)

**Decision rule if ever revisited:** run ridge test-time reliance (flagship-excluded high-variance
null, §8d protocol) for axis2/axis7 → domain_binary/disorder targets, brain+muscle. Only outcome
"axis2 or axis7 B4-positive cross-tissue" (predicted < 5%, contradicts §6a) would justify biological
characterization. Expected outcome (both B4-negative) requires no run: the existing evidence
(compositional B4-negative; low-evr encoded-only precedent = axis5) pre-determines it. **Do not run
Pfam/UniProt/structural characterization on un-used, low-variance axes** — axis0/1/4/5/6 were never
characterized beyond composition either; axis2/7 get the same treatment. Map is complete.

## 13. Domain output-magnitude vs edit size — [LEARN] follow-up (Option B, 2026-07-23)

Panel D showed Spearman(edit size, |ΔPRISM|₁) = 0.41 domain vs 0.68 non-domain. Hypothesis tested
(`b4_domain_magnitude_usage.py`, brain): the domain gap is domain-structural *usage* in magnitude,
carried by the output-used domain axis (axis3). **Prediction largely FAILED — self-caught confound.**

Naive result looked confirmatory: axis3 trajectory displacement ||ΔZ₃||₂ adds +0.191 R² (domain) /
+0.254 (non-domain) beyond size. But this is a **magnitude–magnitude confound**: ||ΔZ₃|| is a
displacement *magnitude* and |ΔPRISM| is a *magnitude*, so both scale with total perturbation — the
exact magnitude/direction conflation the project already flagged ([[lesson-magnitude-direction-conflation-severity]]),
here applied to myself. Two tells: (i) axis3 *alone* predicts non-domain output (0.376) BETTER than
domain (0.206) — backwards if axis3 were the domain channel; (ii) the discriminating test (incremental
axis3 BEYOND size + the *other 7* axes) collapses to **+0.032 domain / +0.014 non-domain** — i.e. the
other 7 axes (a total-perturbation proxy) already capture ~85% of the naive axis3 signal.

**Honest conclusions:**
1. The naive "axis3 magnitude usage" is ~83% generic perturbation magnitude, not axis3-specific. The
   map's "domain is output-used" claim continues to rest on the *directional* ridge-reliance test
   (§8d), which is untouched; this magnitude test is not a valid usage instrument.
2. A *small* axis3-specific residual survives and is ~2× larger for domain (+0.032) than non-domain
   (+0.014) — weakly in the predicted direction, but too small to carry a claim.
3. The robust content of the ρ gap is different: **size alone explains domain output magnitude poorly
   (R²=0.042) vs non-domain (0.160).** For domain edits, residue count is a weak proxy for
   representational consequence (a whole compact domain vs part can be removed at similar residue
   counts); non-domain output scales more linearly with size. Embedding *displacement* (any axis)
   predicts domain output far better than residue count does.

**Manuscript impact: none** — Panel D already reports the 0.41/0.68 split without a mechanistic gloss;
this refinement stays in the working doc. Method lesson reinforced: to test usage of a component,
use its *signed directional* reliance, never its displacement *magnitude* against an output magnitude.

### 13a. Artifact check on the domain size-R² gap (Option C, PROCEED-MINIMAL, 2026-07-23)

Before chasing "which Pfam family determines domain output" (Option C), devils-advocate forced an
artifact-first gate (`domain_size_r2_artifact_check.py`). It **deflates §13 point 3** — a third
self-caught slip in this sub-track:

- The raw R²=0.042 (domain) vs 0.160 (non-domain) headline was **mostly a LINEARITY artifact.** I fit
  ridge *linearly on raw* edit size, but the size→output relation is log-linear (Panel D is on log
  axes). With **log-size**: domain R² recovers to **0.134** (×3.2), non-domain to 0.300. So "size
  explains domain output ~4× worse" was an artifact of the linear-on-raw fit.
- A **real but modest residual** remains after the log-fix: domain 0.134 vs non-domain 0.300 (ratio
  0.45), partly attributable to mild **range compression** — domain edits cluster large (mean 815 vs
  287 residues; median 508 vs 139; log-size CV ratio 0.56, i.e. domain spans a narrower log-range).
- **No saturation**: only 0.1% of domain pairs sit in the top-10% output band; max |ΔPRISM| 138
  (domain) ≈ 144 (non-domain). CV ratios size 0.76 / log-size 0.56 / output 0.72 — none below the 0.5
  clean-artifact threshold, but linearity + partial range-compression jointly explain most of the gap.

**Verdict: DROP the Pfam-family investigation.** The residual heterogeneity is too small, and (per the
pre-validation) Pfam identity is gene-family-confounded and underpowered (long tail). The Spearman ρ
the *manuscript* reports (0.41 / 0.68) is rank-based and log-invariant, so the manuscript is unaffected;
only this working-doc §13 overstated the effect via a linear fit. **Method lesson (third in this
sub-track): match the fit to the known functional form — a low *linear* R² on a heavy-tailed predictor
is a fit-specification artifact, not evidence the predictor is uninformative.** Interpretability-map
track closed.

## 14. Operator re-framing → layer-resolved (A) + pooling-kernel (B) tests (2026-07-23)

The map was re-expressed as an explicit operator cascade ("kernel of each stage's operator = the
information it destroys"), referred to devils-advocate (7 attacks), and the two load-bearing
assumptions were tested with new computation. **Both attacks improved the framing rather than
breaking it: two literal math claims were dropped, and the B2b mechanism was upgraded.**

**Self-corrections to the operator language (accepted before testing):**
- *mean-pool is NOT an "orthogonal projection onto the DC mode."* Type mismatch: `P: ℝ^{L×640}→ℝ^640`,
  `m=(1/L)1ᵀH`, is a lossy linear *surjection*; the genuine projection `H↦(1/L)11ᵀH` outputs a rank-1
  matrix `1mᵀ`. The kernel `{H:1ᵀH=0}` is real but the "orthogonal projection" label is wrong.
- *"kernel = information destroyed" holds only for the LINEAR stages* (B2b `P`; B3 fixed-`w` projection).
  For nonlinear B1 (LayerNorm/attention/softmax) and B4 (ReLU/sigmoid), `ker(Jacobian)` is a local
  tangent object ≠ MI loss. Decisive counter: LayerNorm's Jacobian is generically full-rank
  (`ker={0}`) yet it globally removes per-token mean+scale — "kernel=destroyed" would falsely call it
  lossless.

### 14a. Option A — layer-resolved depth trajectory (cached pooled, all 30 layers, gene-disjoint CV)

`layer_resolved_depth_trajectory.py`, `layer_depth_close_A.py`. Canonical-anchored
`δφ^(ℓ)=φ^(ℓ)[other]−φ^(ℓ)[canonical]`; logistic AUROC per layer. Brain n=33,802 / muscle n=15,885.

- **B1 is not monotone-lossless — feature-specific peak depth + shared late erosion (tissue-general).**
  Domain decodability peaks MID-network (brain L9 0.815, muscle L11–12 0.638), erodes to a trough
  (L19–24, brain ~0.745 / muscle ~0.58), partially recovers by L30 (0.787/0.609). Disorder (within
  non-domain) peaks at the EARLIEST layers (L2, brain 0.871 / muscle 0.735) and erodes monotonically.
  Reading: **local composition properties (disorder) are readable shallow; global/structural
  properties (domain) require mid-network context; both are then eroded as late layers specialise for
  the MLM objective.** So "B1 encoding" is a depth-ordered emergence, not one box. Both shapes
  replicate across tissues (muscle weaker throughout, consistent with DR-AUC 0.630 vs 0.775).
- **Final-layer LN removes SCALE but preserves domain DIRECTION.** `‖δφ^(ℓ)‖` grows monotonically
  L1→L29 (7.1→54.0) then collapses at L30 (1.8, the final emb-layer-norm), yet domain AUROC *recovers*
  L20→L30 — a concrete instance that LN destroys magnitude DOF while the domain-discriminative
  direction survives (domain is not scale-encoded). Because AUROC is on per-dim-standardised features,
  the curve reflects *directional* discriminability, not the norm growth.
- **δ_layer (B2a) partly recovers the missed mid-peak, and the L9 anchor beats PRISM's L15/L30.**
  Brain: `φ9` alone 0.815 > `φ15` 0.798 > `φ30` 0.787; `φ30−φ15` (what PRISM uses) 0.796; **concat[φ9,φ30]
  0.824 (+0.028 over φ30−φ15)**. This is *why* δ_layer helps — L15 sits nearer the L9 peak than L30.
  An architecture lead (not acted on without a full ablation; muscle shows the same ordering but weaker
  and no concat gain, its domain signal being intrinsically weak).
- Caveat: the late-layer trough could be partly a massive-activation/rogue-dimension × per-dim-standardisation
  artifact; and this is all POOLED, so it cannot separate "B1 eroded it" from "B2b pooled it away at each
  depth." Both are resolved by B (per-residue).

### 14b. Option B — the pooling kernel is a COHERENCE filter, not O(k/L) (per-residue, L9/L15/L30)

`b_extract_perresidue.py` re-ran ESM-2 keeping per-residue tensors at L9/L15/L30 for 2,262 isoforms
(brain: 800 SLiM-candidate pairs = non-domain 3–40 aa localized edits; 400 domain controls ≥80 aa).
`b_analyze_pooling_kernel.py` aligns each long/short pair (SequenceMatcher); at 'equal'-aligned
positions `δh_p=h_long[p]−h_short[aligned]` is a **pure context effect** (same residue).

- **T1 contextual-spread (‖δh_p‖ / edit-core magnitude, by distance from the edit):** the edit is NOT
  locally supported — attention spreads it (1–2 aa neighbours carry 34% [SLiM] / 50% [domain] of the
  edit-core magnitude), so the O(k/L) *local-support* premise is false (devils Attack 2 correct). BUT
  the spatial EXTENT differs sharply: SLiM decays 0.341→0.031 (>100 aa) = ~11× (quasi-local); domain
  decays 0.496→0.152 = ~3×, still 15% at >100 aa (quasi-global).
- **T2 pooling survival (DC coherence `frac_kept=‖mean_p δh_p‖²/mean_p‖δh_p‖²`):** SLiM 0.082/0.050/0.071
  (L9/L15/L30) vs domain 0.312/0.216/0.160 — **mean-pool discards ~92% of SLiM's per-residue δ-energy
  vs ~69% of domain's (≈3.8× survival gap at L9).** Raw edit pooling weight `edit_len/L`: SLiM 3.8%,
  domain 67%.
- **Mechanism (upgrade of B2b):** mean-pool keeps only the DC (mean) component. SLiM's spread is
  spatially INCOHERENT (positions point different directions → mean cancels → ~5–8% survives); domain
  shifts are COHERENT (aligned directions → mean survives → 16–31%). So **B2b is a spatial-COHERENCE
  filter, not a k/L low-pass — it preserves coherent shifts (domain) and averages away incoherent
  perturbations (SLiM), independent of raw edit size.** This is the same maths as coherent-integration
  gain in signal processing (only phase-aligned components survive averaging). T1 (distance-normalised,
  confound-free) and T2 corroborate via independent routes.
- Caveat: `frac_kept` depends on the equal-flank fraction (domain edits are 67% of their protein, so
  fewer/nearer equal positions) — a partial confound on T2; T1 is independent of it and gives the same
  verdict.

### 14c. Verdicts on the two attacks

- **Attack 1 SEVERE ("B1 destroys SLiM") — REFUTED for SLiM.** SLiM is strongly encoded per-residue at
  the edit core at ALL layers (L9/L15/L30); it dies at POOLING (T2, ~92% discarded), not inside B1.
  This confirms the original "B1 encodes, B2b destroys SLiM" placement. (B1 *is* mildly lossy in the
  §14a sense — a shared late-layer erosion of pooled decodability — but it does not destroy the local
  SLiM signal.)
- **Attack 2 ("O(k/L) low-pass") — conclusion upheld, mechanism replaced.** SLiMs do die at mean-pool,
  but by coherence filtering, not k/L dilution. The literal O(k/L) and "orthogonal-projection" claims
  are dropped; the coherence-filter statement replaces them and is more defensible.

**Net:** the operator re-framing survives as a *verified mechanistic model* (not a pedagogical
restatement): B1 = depth-ordered feature-specific emergence + late erosion; **B2b = spatial-coherence
filter (the geometric root of the domain vs non-domain trace split)**; B2a = depth-contrast recovering
the mid-peak (L9 anchor > L15/L30). Scope: brain-primary, 1,200 pairs, SLiM operationally 3–40 aa.
Reusable asset: `reports/model_interpretability_map/b_perres{,_muscle}/` (per-residue npz L9/L15/L30).

### 14d. Coherence causal test + muscle replication (2026-07-23)

**Muscle replication of the pooling kernel (`b_prep_subset_muscle.py`→`b_extract_perresidue.py`→
`b_analyze_pooling_kernel.py _muscle`; 800 SLiM / 400 domain).** The coherence filter is
tissue-general. T2 `frac_kept`: SLiM 0.087/0.051/0.077 (L9/L15/L30) vs domain 0.260/0.204/0.167 —
~3× survival gap (brain 3.8×); edit weight SLiM 0.039 / domain 0.60. T1 spread (L9): SLiM 0.334→0.031
(~11×, quasi-local), domain 0.503→0.117 (~4×, quasi-global). Near-identical to brain §14b. *(Gotcha
recorded: muscle sequences must be parsed with `build_severity_pairs.parse_pep_sequences`, not
`compute_esm2_all_layers.parse_pep_file` — the latter mis-returns ENST-id records; a first muscle run
with the wrong parser gave zero equal-blocks. The muscle pair indices are into `my_isoform_list_fixed`.)*

**Causal test — is the pooling-DISCARDED (incoherent, non-DC) energy INFORMATIVE, or noise?**
(`b_causal_coherence.py`, brain, L9, gene-disjoint.) Per pair: DC = mean of equal-aligned δh (survives
mean-pool); mode = S₁·Vt₁ of centered δh (dominant DISCARDED direction × magnitude); rand =
magnitude-matched random direction. Decode within-class targets from DC vs DC+mode vs DC+rand.
- **SLiM:** disorder DC 0.732 → +mode 0.675 / +rand 0.667 (**mode-beyond-null +0.009 ≈ 0**);
  nterm DC 0.849 → +mode 0.812 / +rand 0.765 (**mode-beyond-null +0.047**, small).
- **domain:** disorder DC 0.748 → +mode 0.665 / +rand 0.738 (**mode-beyond-null −0.073**, i.e. the
  discarded mode is *worse than random* — pure noise).
- **Verdict (matches predict-before-you-look from region-pool raising coherence but LOWERING DR-AUC):**
  the discarded incoherent energy is **not usefully label-informative** — mean-pool is not throwing away
  recoverable signal, it is correctly discarding incoherent noise. So the coherence filter is a real
  bottleneck for *survival* but **NOT a recoverable architecture opportunity**; SLiM labelability is
  limited by the genuine spatial diffuseness of the signal, not by the pooling operator. (Minor
  exception: N-terminal targeting carries a little recoverable structure in the discarded mode,
  +0.047 — consistent with its terminus-localized coherence.) This independently reconfirms §5's
  "the floor is not recovered by swapping the pooling operator." **Interpretability-map operator track
  (C→A→B) closed.**

### 14e. SLiM-dispersion follow-up — "noise" corrected to structured-but-not-SLiM-functional (2026-07-23)

Prompted by the biological point that SLiM multi-directional dispersion is *the mechanism* of
per-isoform identity (Davey/Gibson; Buljan et al. 2012), not noise. Two label-free/native tests on
the pooling-discarded (non-DC) equal-aligned residual of brain SLiM pairs (L9/L15).

- **§14d's "noise" was too strong — CORRECTED (`b_slim_dispersion_structure.py`).** The discarded
  energy is **not random**: gene-disjoint top-K subspace reproducibility excess **+0.31** over a random
  subspace (SLiM, replicated L9 & L15), i.e. it occupies a reproducible **~48–55-dim** manifold
  (participation ratio 54.5/48.2), *higher-dimensional* than domain's (37.5/28.9). Within a pair there
  is **no single dominant discarded direction** (split-half top-PC |cos| real 0.55–0.64 < marginal-
  variance null 0.76–0.84, whereas domain real > null) — genuinely multi-directional. So the precise
  statement is **"structured but high-dimensional / un-poolable by a 1-D DC and un-anchorable by a 1-D
  B3,"** not "noise." (Caveat: this reproducible subspace can include generic position/length/context
  geometry, not necessarily SLiM-identity.)
- **But it is NOT linearly SLiM-functional (`b_option_B_slim_target.py`).** Target = change in the
  project's own SLiM regex classes (exp_true_motif_level.py A3: NLS/NES/PXXP/CK2/PKA/…) in the edit.
  Decode from DC vs DC+discarded-mode vs DC+magnitude-matched-rand, gene-disjoint. The discarded mode
  adds **~0 beyond null** for all classes (NLS +0.047, PXXP −0.028, CK2 −0.028, PKA −0.004), and DC
  itself only weakly carries SLiM-class (AUROC 0.54–0.62). So the reproducible ~50-dim discarded
  structure is **generic (position/length/context), not linearly SLiM-class-specific.**
- **Net:** the discarded SLiM energy is real reproducible structure (correcting "noise") yet neither
  poolable, cross-gene-anchorable, GO/disorder-informative (§14d), nor linearly SLiM-class-informative
  (§14e). Three routes remained; **two are now tested and closed:**
  - **Edit-core route (`b_editcore_slim.py`) — SLiM survives the core but it is COMPOSITION.** The
    edit-core representation predicts SLiM-class *better than the pooled δφ* (NLS 0.764 vs 0.658, PXXP
    0.671 vs 0.575, CK2/PKA +0.04–0.07) — so identity survives in the residues and mean-pool dilutes it
    (vindicating a core-focused/motif-centric pooling). BUT the 20-dim edit AA-**composition** matches or
    beats the ESM edit-core (0.65–0.78 ≥ editcore), so the recoverable signal is *compositional* — and
    composition is already gene-independent and B4-negative (§6a/§9, unused by PRISM). "Core-focused
    pooling" would thus recover the edit's composition, not context-dependent switching.
  - **Non-linear route (`b_nonlinear_beyond_comp.py`, HistGradientBoosting) — does NOT rescue it.** Across
    all four classes ESM edit-core adds ~0 beyond composition (editcore-beyond-comp +0.019/+0.012/−0.028/
    −0.065), and the discarded mode is at chance (0.497–0.574). Non-linearity does not surface SLiM signal
    the linear probes missed, in either the core-beyond-composition or the discarded context.
  - **ELM high-specificity-label route (`b_elm_beyond_comp.py`, 353 ELM class regexes) — TESTED,
    partial positive with a clean mechanism.** For the 25 most-specific testable ELM classes, the median
    editcore-beyond-composition is −0.017 (composition is still the ceiling for most), BUT **6/25 exceed
    it, and the strongest are all POSITIONALLY-defined SLiMs**: DEG_Nend_Nbox_1 +0.239, DEG_Nend_UBRbox_2
    +0.236, LIG_BIR_II_1 +0.197, DEG_Nend_UBRbox_1 +0.093 (N-end-rule degrons + N-terminal IAP ligand).
    Interpretation: the ESM edit-core beats composition **exactly for the SLiMs whose identity is
    positional (N-terminal)** — ESM encodes N-terminal position, which 20-dim composition cannot;
    internal composition-defined SLiMs (SH2/MAPK-docking/PKA/caspase/LIR) still add ~0. So the model DOES
    carry SLiM-functional information beyond composition, but it is dominated by POSITIONAL (N-terminal)
    encoding, not context-dependent binding-interface switching.
  - **Supervision-causal test (`b_supervision_causal.py`) — SELF-CORRECTS the "supervision-only" reading.**
    The +0.24 positional-SLiM encoding was measured on the EDIT-CORE, but PRISM ingests the MEAN-POOLED δφ.
    Decoding the N-end-degron change from comp vs comp+pooled-δφ (PRISM's actual input) vs comp+editcore:
    **pooled-beyond-comp ≈ 0 (−0.029 to +0.035) while editcore-beyond-comp = +0.09 to +0.24.** So the
    encoded positional signal is **NOT accessible from PRISM's pooled input** even for the SLiMs where the
    edit-core carries it — mean-pool destroys the pooled-accessibility. Controls (SH2/endocytic) flat for
    both. **Therefore the earlier "bottleneck = supervision, not geometry" was overstated.** The correct
    reading: it is a **JOINT B2b (pooling) + B4/B5 (supervision) bottleneck, with pooling PRIMARY for
    output-accessibility** — even a perfect SLiM label cannot be used by PRISM's pooled architecture because
    the pooled input no longer carries the signal beyond composition.
  - **Terminal reading (3-layer, corrected):** (i) *Structural*: discarded dispersion is real ~50-dim
    reproducible structure. (ii) *Encoding*: positionally-defined SLiMs (N-end degrons) are encoded in the
    per-residue edit-core beyond composition (+0.24, via ESM's N-terminal positional encoding); composition-
    defined SLiMs are not. (iii) *Output-accessibility & usage*: mean-pool (B2b) removes the pooled-
    accessibility of even the encoded positional signal (pooled-beyond-comp ≈ 0), and separately no isoform-
    level label rewards it (B4/B5, `b4_nterm_usage.py`). **The SLiM bottleneck is B2b-pooling AND B4/B5-
    supervision jointly — pooling first for accessibility. This VINDICATES a motif/edit-core-centric pooling
    architecture (bypass B2b) as NECESSARY, to be paired with SLiM-specific supervision** — the "geometry-
    bypass is necessary-not-sufficient, supervision is the binding constraint" phrasing of §14e is corrected:
    both bind, and pooling is the primary gate to output-accessibility. **SLiM sub-track closed (with two
    self-corrections: "noise"→structured; "supervision-only"→joint-pooling-primary).**
  Assets: `b_slim_dispersion_structure.py`, `b_option_B_slim_target.py`, `b_editcore_slim.py`,
  `b_nonlinear_beyond_comp.py`, `b_{optionB_slim_target,editcore_slim}.tsv`. **SLiM sub-track closed.**
