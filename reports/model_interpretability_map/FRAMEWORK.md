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
  the isoform difference get lost? Information is monotone non-increasing down this axis.
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

Converting to "share of the above-chance decodable signal": the 8 axes capture
(0.715−0.5)/(0.838−0.5) ≈ **64%** of the *linearly*-decodable domain signal; the non-linear
probe adds a further slice (R3). So even the best-understood class (domain) is only ~2/3
narratable by our current interpretable coordinates — a number worth stating plainly rather
than implying the axes are the whole story.

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
Description  ── R1 0.715 → R2 0.838 → R3 0.890 (≈64% of linear signal is 8-axis-narratable)
```
→ Domain is **"solved to the representation, capped by labels."** The residual here is an
*annotation* limit, provable because ESM-2 places Type-1 isoforms 2.89× farther from canonical
than Type-3 [prior] — it encodes what the labels do not express.

### 4b. NON-DOMAIN edits  (brain 30.2% / muscle 46.5%)

This class does **not** fail at one bottleneck — it fractures into sub-mechanisms that fail at
*different* bottlenecks. That heterogeneity is the whole point.

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
- **Compositional (this session)**: helix/sheet/hydro/charge give a *median-split* usable direction
  that survives the gene-permutation null (and, for brain sheet_delta, the label-permutation null
  where gene-perm was fold-saturated). This is a **genuine B3-level signal on the non-domain
  residual** — but its **B4 usage** (does the current MLP head exploit it?) and **B5 label** (what
  GO term rewards it?) are both **untested/absent**. This is the live frontier.
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
│         ├─ 8-axis-narratable: ~64% of the linear signal
│         └─ ceiling cause: label noise (89.9% Type-1 low-conf), not representation
│
└─ 30.2%  NON-DOMAIN                    [FIRM: genome-wide rate]
          ├─ disorder-region overlap: encoded AUROC 0.79 [FIRM], diffuse/no-anchor  [MED]
          ├─ N-terminal targeting: partial MTS alignment ρ=0.126 [SOFT — gene-perm caveat]
          ├─ compositional signal (helix/sheet/hydro/charge): B3 real [FIRM this session],
          │  B4/B5 UNTESTED  ← quantifiable share of the 30.2% not yet measured
          └─ pure SLiM: median score gap 0.0, 1.06% exceed 0.05 [FIRM] = the floor
```

**What is honestly NOT yet a number:** the *fraction of the 30.2% non-domain residual* that the
compositional covariates actually explain, and whether that fraction is *used* by PRISM (B4) or
merely *encodable* (B1–B3). Producing that number is the natural next computation — and it is the
number that would tell us how much of the describability gap is "future-recoverable" vs "true floor."

**Four terminal buckets** (the user's requested end-categories), now precisely defined:

| Bucket | Definition (bottleneck signature) | Domain | Non-domain |
|---|---|---|---|
| **Explained & used** | survives B1–B4, has B5 label | DR 0.630/0.775, label-capped | small (targeting, partial) |
| **Encoded, not surfaced** | dies B2 (pooling) | ~none | SLiM, part of disorder |
| **Surfaced, un-anchored/un-labelled** | survives B2, dies B3 or B5 | (label-noise tail) | compositional, disorder, internal |
| **True black box (not encoded)** | dies B1 | ~none | PTM-state, motif-*function* |
| **Below ceiling, unexplained (FUTURE)** | used (B4) but description-tier R4 | R2+R3 gap ≈ (0.838→0.890+) | compositional B4 test |

---

## 6. Concrete instrument: the isoform case study

To make the grid legible, decompose *individual* pairs onto the two coordinate systems
simultaneously:
- **8 axes** (unsupervised encoding geometry): project δ = φ(long)−φ(short) onto W(8×640).
- **7 covariates** (supervised edit descriptors): domain, size, disorder, helix, sheet, hydro, charge.

Then the "explainer" asks, per pair: *which axes carry this pair's shift, and do the covariates
predict that axis profile?* The gap between covariate-predicted and actual axis-profile = the
pair's R2+R3+R4 residual (represented but not covariate-describable).

**Diagnostic triple** (spans the grid):
- **Type A — NDUFS4** (MTS-exon / domain loss): expected B1–B4 all pass, R1 high. The "everything works" reference.
- **Type B — MAPT 3R/4R** (N-terminal insert count, no domain change): expected die at B2/B3, near-zero score gap. The "encoded-but-lost" reference.
- **Type C — LRPPRC** (same-domain, CT=AD score): B1 may encode, but B4/B5 flat. The negative control.

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

