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
