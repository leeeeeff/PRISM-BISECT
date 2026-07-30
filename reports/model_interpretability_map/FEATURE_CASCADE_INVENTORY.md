# Feature × Cascade Inventory — what drops where, what's known, what's unknown

> Working ledger (not manuscript). Consolidates FRAMEWORK.md §1–§14e + memory
> `finding-pooling-coherence-filter`, `finding-6covariate-survivors`, `future-editcore-supervision-prototype`.
> Purpose: a single reference for **architecture-revision candidates** — which stage each feature dies at,
> and the mechanism-appropriate intervention to recover it. Numbers flagged [prior] must be re-verified
> before manuscript use. Created 2026-07-24.

---

## 0. The two axes (recap)

**Vertical — information-flow cascade** (a *fixed* isoform difference is monotone non-increasing along one path):
```
B0 physical edit → B1 ESM per-residue encoding → B2a depth-collapse (δ_layer=φ_L30−φ_L15)
   → B2b residue-avg pooling (COHERENCE filter) → B3 anchor (cross-gene reference direction)
   → B4 usage (does the readout head act on it?) → B5 label (is there supervision?)
```
**Horizontal — description resolution:** R1 (8 interpretable axes, linear lower bound) → R2 (full-640 linear)
→ R3 (non-linear only) → R4 (undescribable / not-yet-named).

Two traces run through the grid: **DOMAIN** edits (coherent, survive) and **NON-DOMAIN** edits (mixed fate).

Population denominator = within-gene splice events. Brain: 69.8% domain-affecting / 30.2% non-domain [FIRM, genome-wide]. Of non-domain: **51.5% N-terminal targeting**, **48.1% structured-internal** (of which ~54.6% specific-SLiM-bearing), **0.4% disorder-dominant** (overlapping property, not a bucket). [prior]

---

## 1. Feature ledger — fate of each known signal

Legend: ✓ pass · ✗ die/absent · ~ partial. Stage where the feature is LOST is **bold**.

| # | Feature / signal | B1 encode | B2 pool | B3 anchor | B4 used | B5 label | Verdict | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | **Domain-architecture change** (axis3) | ✓ peak L9 | ✓ coherent (frac_kept 16–31%) | ✓ cross-gene | ✓ ridge rel 11–51%, DR-AUC 0.630/0.775 | ~ label-capped (89.9% Type-1 low-conf) | **WORKING — the one usable stream.** Ceiling = label noise, not representation | `b_analyze_pooling_kernel`, ridge reliance, `case_study_interpretability` |
| 2 | **N-terminal targeting** (51.5% non-dom, largest single class) | ✓ (weak MTS align ρ=0.126) | ~ position survives beyond comp (ELM DEG_Nend +0.239) | ✗ no cross-gene anchor | **✗ NOT used** (b4_nterm at gene-perm null) | ✗ | Strongest editcore-beyond-comp signal but **downstream-dead** at current output | `b4_nterm_usage`, `b_elm_beyond_comp` |
| 3 | **SLiM / specific short motifs** (~54.6% of structured-internal) | ✓ per-residue at edit-core (+0.24 beyond comp, positional) | **✗ DIES** (incoherent, frac_kept 5–8%, pooled-Δ≈0) | — | ✗ (absent from pooled input) | ✗ (regex/ELM comp-dominated) | **Encoded, pooling-lost.** ~50-dim discarded manifold is *structural* (+0.31 reproducible) but *generic* (position/length/context), NOT linear/nonlinear SLiM-function | `b_supervision_causal`, `b_editcore_slim`, `b_option_B_slim_target`, `b_nonlinear_beyond_comp` |
| 4 | **Compositional** (helix/sheet/hydro/charge) | ✓ | ~ weakly-surfaced *dataset-wide* direction (gene-INDEPENDENT) | ✗ ≈ gene-perm null | **✗ NOT used** (0.59–0.65 embed → 0.53–0.58 output = null) | ✗ | **Encoded / gene-independent / unused.** B4-negative both tissues (robust) | `b4_compositional_usage` (§9) |
| 5 | **Disorder** (axis0, overlapping property) | ✓ shallow peak L2, any-overlap AUROC 0.79 | ~ diffuse, no anchor (β<0) | ✗ | **✗ usage RETRACTED** (ridge rel −3%, redundant) | ✗ | Encoded property, <1% as dominant stream; earlier "causally used" was variance-confound | `finding-pooling-coherence-filter`, ridge reliance retraction |
| 6 | **Edit SIZE** | — | — | — | tracks output magnitude (tissue-general) | — | **Confound, not a functional feature** — per-pair PRISM |ΔScore| follows edit size, not domain status. Must be controlled in every claim | `case_study_interpretability` |
| 7 | **severity covariates** (severity_score / resync / n_intervals) | — | — | ✗ gene-perm invariant | (regression only) | — | Dataset-wide statistical regularity, NOT cross-gene mechanism. n_intervals (edit fragmentation) = most fundamental (89.2% mediation) | `finding-6covariate-survivors`, severity mediation |

**Reading:** exactly **one** stream (domain, row 1) makes it all the way to *used*. Every non-domain stream dies at B2 (SLiM), B3 (compositional/N-term anchor), B4 (N-term/compositional usage), or B5 (labels) — the describability gap is real and stage-localized.

---

## 2. Four terminal buckets (where signals end up)

| Bucket | Bottleneck signature | Domain | Non-domain |
|---|---|---|---|
| **Explained & used** | survives B1–B4 + has B5 label | ✓ DR 0.630/0.775 (label-capped) | ~none |
| **Encoded, not surfaced** | dies B2 pooling | ~none | SLiM, part of disorder |
| **Surfaced, un-anchored / un-labelled** | survives B2, dies B3 or B5 | (label-noise tail) | compositional, N-terminal, internal |
| **Not encoded (true black box)** | dies B1 | ~none | PTM-state, motif-*function* |
| **Used but undescribable (FUTURE)** | passes B4, description-tier R3/R4 | non-linear domain tail (0.838→0.890) | decodable-but-non-compositional non-domain residual |

---

## 3. UNKNOWN / undescribed territory (the future-work core)

Quantified from `nondomain_residual_decomposition.py` / `nondomain_tail_rich.py` (brain non-domain, gene-disjoint):

```
Non-domain embedding-difference variance (100%, centered)
├─ 45.3%  per-pair NOISE FLOOR  — NOT gene-reproducible, NOT pooling-dilution
│         (Spearman(changed-frac, reproducible-excess) = −0.40 n.s.; smallest edits are MOST
│          reproducible) → genuine biological per-pair idiosyncrasy, NOT recoverable by swapping pooling
└─ 54.7%  gene-REPRODUCIBLE structure (recoverable in principle)
          ├─ 33.5%  named by full 20-dim AA composition (+0.335 of variance)
          └─ ~19–36% reproducible-but-BEYOND-composition  ← R2/R3 tail, survives full comp (+0.387 residual)
                     high-dim (K=50 excess +0.427 after removing 7 covariates) = structural/positional/contextual
```

Open unknowns (ranked by tractability):
1. **R2/R3 non-domain tail** — reproducible, high-dim, genuinely beyond amino-acid composition. Needs *positional/contextual* descriptors, not more composition. **[DESCRIPTOR CLASS NAMED — 2026-07-25, §B1/§B1b]**: the beyond-comp signal is *within-class positional residue identity* (which residue at which position), established by two independent lines — dipeptide-class Occam test NEGATIVE (does not collapse to cheap local-order descriptor, §B1) + synthetic within-class-permutation pilot POSITIVE (ESM editcore AUC 0.759 with mono- and dipeptide-class held identical by construction, §B1b). So "can't name it" → "named in *kind* (positional identity), function still open." Function is confirmed only for the targeting/degron subset; the *bulk* tail's function remains label-blocked (B5, 6/6 dead). Written into manuscript §Discussion map paragraph + §4c synthesis (2026-07-25, commit after 7c87c2b).
2. **Non-linear domain tail** — 0.838 (640-linear) → 0.890 (non-linear) = ~5% domain signal only non-linearly decodable. R3 territory.
3. **axis2 / axis7** — DROPPED as robust (magnitude test), identity still uncharacterized.
4. **Does the GO-function target reward ANY pooling-lost signal?** — untested (this is Option A). If GO doesn't reward positional/SLiM signal, no pooling fix helps the *current* output.
5. **The 45% floor** — likely irreducible with current data/labels (not pooling-recoverable).

---

## 4. FORWARD WORKLIST — architecture-revision candidates to "recover / surface"

Ranked by **readiness × mechanism-fit × risk**. Every item is ablation-gated (architecture.md) and needs null/oracle-first (research-method S2).

| Pri | Candidate | Stage fixed | Mechanism / rationale | Evidence status | Risk |
|---|---|---|---|---|---|
| **A1** | ~~**L9 anchor** — concat[φ9, φ30] replacing/augmenting δ_layer=φ30−φ15~~ | B1/B2a | domain peaks at **L9**, current anchor (L15/L30) samples past the peak; concat[φ9,φ30]=0.824 vs 0.796 (+0.028 brain) was a domain-**decodability proxy only** | **DEAD (2026-07-25)** — `b_l9_anchor_ablation.py` on the real 18-GO macro-AUPRC task, bootstrap CI n=1000: L30(production)=0.6899 [0.6757,0.7037]; every L9-containing variant is significantly **worse**, CI excludes 0 for all: L9+L30 Δ=−0.0089 [−0.0165,−0.0018], L9 Δ=−0.0106, L30−L9 Δ=−0.0210, L15+L30 Δ=−0.0703. Predict-before-look ("NO gain, consistent with per-term layer-selection-does-not-exceed-ceiling") confirmed, and stronger than predicted (reliable loss, not just no-gain). Proxy→real-task transfer failed. | — closed |
| **A2** | ~~**Edit-core additive channel** — concat N-terminal-window pool (standalone-isoform analog) to existing mean-pool δφ~~ | B2b bypass | surface pooling-lost positional/SLiM (offline editcore-Δ +0.09–0.24) | **DEAD (2026-07-25)** — 4-tier ablation (macro-AUPRC, DR-AUC, 6 within-gene covariate-AUCs incl. the mechanism-targeted nterm_deviates): macro-AUPRC CI-excl-0 worse (Δ=-0.0367), nterm_deviates worse (both <0.5 chance), all other covariates CI-overlap (no change). See §5i. | — closed |
| **A3** | **Non-linear readout head** (domain tail) | B4/R3 | capture 0.838→0.890 domain-decoding gap the linear head misses | testable; bio-risk low | MED (capacity in sparse regime) |
| **B1** | **Richer positional/contextual descriptors** (R2/R3 tail) | horizontal (R2→R3) | name the +0.387 beyond-composition reproducible structure; feature-engineering, not architecture | **CHARACTERIZED (2026-07-25)** — descriptor class = within-class positional residue identity (dipeptide-Occam NEG §B1 + synthetic-permutation POS §B1b); named in *kind*, function still label-blocked (B5 6/6 dead). Manuscript-integrated (§Discussion map para, §4c). Remaining: attach function to the *bulk* tail → requires a B5 label (out of cheap scope) | — description closed |
| **B2** | **New B5 labels** — isoform-level functional (localization / half-life / conservation-filtered ELM) | B5 supervision | the binding half of the joint bottleneck for every non-domain stream | **6/6 candidates tried have failed** (GO, is_alt_functional, ELM×2, UniProt/HPA, IDR-boundary — §5a–5e), each via a different mechanism | provisionally exhausted for cheaply-reachable labels; needs either wet-lab-adjacent curation or a synthetic/constructed-ground-truth design (out of current scope) |
| **C** | **Bilinear / 2nd-order incoherent pooling** (Route 3) | B2b | rescue ~50-dim incoherent manifold | **DEPRIORITIZED** — function-null already FAILED (option_B + nonlinear ≈ chance); permutation-invariance destroys the N-term position that IS recoverable | HIGH — build only after a label makes the manifold function-predictive |

**Guardrails carried forward:** (1) additive-not-replace for any pooling change (preserve domain trace / DR-AUC); (2) frozen-ESM small readout BEFORE full retrain; (3) predict-before-look null for every gain claim; (4) joint macro AND DR-AUC monitoring (reject macro↑/DR↓); (5) composition-residual control on every editcore claim (edit AA-comp ≥ ESM editcore for SLiM-class).

---

## 5. The single decision gate that unblocks A2 (= Option A)

Reuse `b_supervision_causal.py` machinery, **swap the label from SLiM-class → GO-function** (isoform-resolution).
- comp vs comp+pooled δφ vs comp+editcore, gene-disjoint, composition-residual, magnitude-matched null, bootstrap CI.
- **Predict-before-look:** editcore−pooled Δ < +0.03 (CI incl. 0) on GO-BP, because GO-BP is process-level and driven by the coherent domain signal (already pooling-accessible), while editcore's unique contribution (positional-SLiM) is not what these terms reward. If this holds → pooling-fix is moot on the *current* target (problem is B5 labels). If refuted (editcore ≫ pooled) → A2 green-lit.

### 5a. RESOLVED (2026-07-24) — no current label tests the pooling hypothesis; B5 is binding

Option A ran and resolved **decisively against a pooling-first bet on currently-available labels**, more sharply than a noisy AUROC would have:

1. **GO-BP labels are GENE-INHERITED** (`load_labels`, v15d_bp_clean.py:135–144, keyed by gene symbol; `human_annotations_unified_bp.txt` is gene-symbol→GO). All 1,200 manifest pairs are within-gene ⇒ **differential-GO ≡ 0 by construction**. The literal Option A target does not exist. This *is* the answer: the deployed target has zero isoform-resolution variance for ANY representation (pooled or editcore) to differentiate on ⇒ **B5 (labels) confirmed as the binding constraint.**
2. **Salvage candidate found — `is_alt_functional`** (`reference_labels_v1.tsv`) is a genuine isoform-resolution label (varies within-gene for 31.6% of genes; 27,327 manifest isoforms mappable via tx_id_versioned). BUT its definition (`fetch_appris_reference.py:196`) is `APPRIS==ALTERNATIVE OR (non-MANE & gene-dominant & ENST)` ⇒ **APPRIS/domain-integrity-driven = the coherent domain trace pooling already keeps.** It plays to pooled's strength, not editcore's (local/SLiM). Predict-before-look: editcore ≈ pooled here too. Plus confounds: per-isoform (not per-edit) label, and principal-vs-alternative correlates with size/length (ledger row 6).
3. ⟹ **No currently-available label rewards the LOCAL/positional signal that editcore uniquely surfaces.** A clean test of the pooling hypothesis requires a **local-feature isoform label** (localization, half-life, conservation-filtered ELM instance) — exactly `future-editcore-supervision-prototype`'s prescription. **B2 (curate such a label) is a hard precondition for A2.**

**Re-ranked consequence:** A1 (L9 anchor) is already-measured (§14a probe, not a fresh win — see correction below). A2 (edit-core pooling) is **blocked on B5 label curation**, not on architecture.

### 5b. B5 curation pilot (2026-07-24) — ELM conservation-filtered instances also fail, on POWER not degeneracy

Pursuing B5 curation (user-selected direction: ELM conservation-filtered instance), applying the same
cheap-check-before-curation discipline that resolved §5a:

1. **Retroactive finding**: the 4 positional ELM classes that beat composition in `b_supervision_causal`
   (DEG_Nend_UBRbox_1/2, DEG_Nend_Nbox_1, LIG_BIR_II_1) have **0 curated instances** in the ELM DB (classes
   index `#Instances` col, verified via API sanity-check on a renamed class). Those +0.09–0.24 gains were
   measured on unvalidated regex, not confirmed degron biology — open question, unresolved this session.
2. Real instance-bearing classes (TRG_NLS/TRG_ER/DEG_SCF/DEG_APCC, 25 classes, maps to the N-terminal
   targeting bucket) fetched from elm.eu.org: 203 human instances / 175 UniProt accessions.
3. Cross-referenced against manifest genes (UniProt→symbol): **18/1,062 genes (1.7%) overlap**, **20 manifest
   pairs total, spread across 18 ELM classes** (≈1–2 per class).
4. This track's viability floor (n≥40 pos & neg, `b_supervision_causal`) is missed by an order of magnitude.
   **Different failure mode than §5a (power, not zero-variance) but same practical verdict: unusable as scoped.**

**Cause**: the manifest (1,200 pairs) is a curated SUBSET for the pooling-kernel study (SLiM 3–40aa 800 +
domain 400), not the genome-wide population (brain non-domain n=25,262, §5). ELM curated instances are also
globally sparse (10–30/class).

**Open forward paths** (not started): (a) re-extract per-residue on the full genome-wide non-domain
population and re-cross-reference — multi-session cost; (b) pivot to UniProt/HPA localization (broader
per-protein coverage, but must run the SAME cheap within-gene-variance check first — isoform-inherited
annotation is a live risk, same shape as the GO failure); (c) treat n=20 as exploratory/qualitative only
(breaks this project's bootstrap-CI discipline, not recommended).

Assets persisted: `reports/model_interpretability_map/assets/` (elms_classes_index.tsv, elm_instances_real/,
elm_instances_human.tsv, uniprot_gene_map.tsv).

### 5c. Both §5b forward paths checked cheaply (2026-07-24, no re-extraction run) — (a) UniProt/HPA dead, (b) genome-wide reframed to a pooled label

Both open paths from §5b resolved via existing assets, without launching the multi-session per-residue
re-extraction implied by path (a) there.

**Path (b) [UniProt/HPA localization pivot] — DEAD, same failure shape as §5a (GO).**
- 40-gene random sample from manifest, UniProtKB REST (`cc_subcellular_location`, `cc_alternative_products`):
  31/40 genes have ≥2 UniProt isoforms recorded, but only **1/31 (3.2%, HM13)** carries an isoform-resolved
  (`molecule`-tagged) subcellular-location annotation — the rest is one comment on the gene entry, inherited
  by every isoform. HPA API keys its response by `Gene`, not transcript — confirms the same gene-level
  structure. **Predicted risk in §5b's own text ("동일한 저비용 degeneracy check... GO에서 죽은 것과 같은
  패턴일 수 있어") materialized exactly.** Third candidate label (GO, ELM-per-class, now UniProt/HPA) to fail
  the cheap check before curation — the pattern itself is now the finding: isoform-level function/location
  annotation is structurally scarce in public DBs, not just missing for our two tries.

**Path (a) [genome-wide ELM re-extraction] — reframed, cheaper test run instead of the heavy one.**
Rather than re-running per-residue extraction on the full n=25,262 population, cross-referenced the ELM
instance gene set (451 symbols) against the pipeline's FULL gene universe (`my_gene_list_fixed.npy`, 12,709
ENSG → 11,841 symbol-mapped via `reference_labels_v1.tsv`, ~11× the manifest's 1,062 genes):
- Overlap = **135/11,841 genes (1.14%)** — essentially the same rate as the manifest's 1.7%. **The bottleneck
  is ELM's own global sparsity for these 25 classes (~450 human genes total), not our subsetting.**
- Projected via manifest's yield (20 pairs/18 genes ≈ 1.1 pairs/gene): genome-wide ≈ **150 pairs**, still
  spread over 25 classes (~6/class) ⇒ **per-class classification still fails n≥40 even at full scale.**
- **New option surfaced**: pool all 25 classes into a single binary label ("edit region overlaps ANY curated
  ELM instance") ⇒ n≈150 clears the viability floor. Requires a coordinate-level check (edit-region vs
  instance position, not just same-gene) that has NOT been run yet — pure sequence-diff, no ESM re-embedding
  needed, and scoped to only the 135-gene subset (cheap, not the original multi-session estimate).

**Status: decision point, not yet executed.** Next action if pursued: build edit-region coordinates for the
135-gene subset (extend `b_manifest_pairs.tsv`-style extraction, sequence-alignment only) and check position
overlap against `elm_instances_human.tsv`. If overlap count stays well above 40 after this stricter filter,
this becomes the first B5 label to survive both the variance check (§5a lesson) and the power check (§5b/§5c
lesson).

### 5d. Devils-advocate on §5c's pooled-ELM label → FATAL circularity → pivot to IDR boundary-shift (2026-07-24)

Before executing the §5c coordinate check, ran devils-advocate on the "pooled binary ELM-overlap" label design.
**Verdict: RECONSIDER.**

- **FATAL (self-re-derived, not just accepted)**: the label ("edit region overlaps a curated instance") and
  the representation being tested (editcore = edit-region-only embedding) are both defined by edit-region
  membership. Since pooling-dilution is an *already-established* mechanism (`finding-pooling-coherence-filter`:
  editcore preserves local signal, mean-pool dilutes it, independent of what that signal *means*), editcore
  beating pooled on this label is guaranteed by known dilution mechanics regardless of whether ELM's curation
  reflects real motif biology. The experiment would reconfirm a known fact, not test a new one — Occam
  violation (S3).
- **SEVERE**: ELM curation = publication-bias proxy (negative class likely contaminated with true positives,
  same failure shape as `domain_binary` Type-1 noise); 25 heterogeneous classes pooled risks reducing to a
  trivial N/C-terminal-position signal (already-flagged guard in `future-editcore-supervision-prototype`); no
  gene-permutation null was planned (mandatory per the severity-regression precedent, `finding-6covariate-survivors`).

**Pivot — IDR boundary-shift** (devils-advocate's suggested alternative, structurally free of both the
circularity and the selection-bias risk: disorder state is computed from sequence *context* by an
independently-trained predictor, not defined in terms of edit-region membership or curated external
annotation).

**Pilot run** (`hMuscle/model/b_idr_boundary_pilot.py`, full manifest, metapredict v3.0.2 — NOT the existing
TOP-IDP-based `disorder_frac`, which is a composition lookup table and would be redundant with the
composition-residual control already baked into every editcore claim):
- 2,262 unique sequences scored in **59s on CPU** (cheap — genome-wide extrapolation ≈ 16 min, unlike ELM/UniProt).
- **512/1,200 manifest pairs** have residues on both sides of the edit (substitution-type; pure-indel pairs
  excluded since one side has no aligned region to compare).
- Binary "order↔disorder boundary crossed" label: **183 positive / 329 negative** — clears n≥40 on both
  sides by a wide margin (first B5 candidate to do so cleanly).
- **Novelty check**: correlation(metapredict Δ, TOP-IDP-composition Δ) = **0.292** (R²≈8.5%) — genuinely
  mostly-orthogonal to AA composition, unlike a label that would just re-encode `disorder_frac`.

**Status: pilot passed, not yet the full supervision test.** Before running editcore-vs-pooled on this label,
pre-register (per the §5d checklist, to not repeat the ELM mistake): (1) gene-permutation null, (2) positional
oracle (N/C-terminal binary + edit length, since disorder is generically enriched at termini — check this
doesn't already explain the label), (3) predict-before-look threshold for editcore−pooled Δ.

Assets: `hMuscle/model/b_idr_boundary_pilot.py`, `reports/model_interpretability_map/assets/idr_boundary_pilot_{raw,both}.tsv`.

### 5e. Full supervision-causal test (2026-07-24) — DECISIVE FAIL, 6th consecutive B5 label death

Ran `hMuscle/model/b_idr_supervision_causal.py` with all 3 pre-registrations from §5d (confound oracle,
gene-permutation null, predict-before-look). n=499 pairs (465 genes), label balance 180/319.

```
comp only       0.571
comp+pooled     0.597   (pool-Δ = +0.026)
comp+editcore   0.556   (ec-Δ   = -0.015)
oracle only     0.636   (positional N/C-term + edit length + whole-protein mean-disorder + length; NO embeddings)
comp+oracle     0.598
```

- **Decisive check 1 FAILED**: editcore (0.556) loses to the confound oracle (0.636) by 8pt. The
  boundary-crossing signal is almost entirely explained by gross positional/gene-level features, not
  by any local edit-region embedding content — the exact failure mode devils-advocate warned about
  for the ELM label, recurring here via a structurally different, non-circular route.
- **Predict-before-look partially REFUTED**: predicted editcore-beyond-comp > pooled-beyond-comp
  (mirroring the established pooling-dilution mechanism). Observed the reverse — pooled (0.597) beats
  editcore (0.556), which even underperforms comp-only (curse-of-dimensionality on n=499 with 640-dim
  editcore, most likely). A refuted prediction here is a *good* sign methodologically: it proves the
  test wasn't circular/pre-determined (unlike the FATAL §5d design), even though the substantive
  answer is negative.
- **Decisive check 2 PASSED**: true-gene CV AUROC (0.556) falls inside/below the 20-shuffle
  gene-permutation null (mean 0.603, sd 0.015, z=−3.17) — no leakage-driven inflation, an honest
  no-signal result.

**Verdict: 6th consecutive B5 label failure** (GO gene-inherited / `is_alt_functional` APPRIS-biased /
ELM per-class underpowered / ELM pooled-binary FATAL-circular / UniProt-HPA gene-level / IDR-boundary
confound-dominated) — each via a *different* mechanism, which strengthens rather than weakens the
pattern: no currently-reachable external label isolates editcore-specific local signal from gross
gene/positional confounds. Re-ranks B2 (new B5 labels) from "scope decision" to **provisionally
exhausted for cheaply-reachable candidates** — see memory `finding-idr-supervision-decisive-fail` for
the forward-options discussion.

Assets: `hMuscle/model/b_idr_supervision_causal.py`, `reports/model_interpretability_map/assets/idr_supervision_causal_results.txt`.

### 5f. Discarded-manifold biological identity — S1 multi-hypothesis triangulation (2026-07-25)

User challenge (correctly identified, S0 self-correction): `b_option_B_slim_target.py` (§ SLiM
sub-track, see `finding-pooling-coherence-filter`) tested the pooling-discarded ~48-55D SLiM manifold
against exactly ONE candidate identity (project's own regex SLiM classes), found it null, and jumped
straight to "generic (position/length/context)" without testing alternative biological hypotheses —
a single-hypothesis-then-abandon pattern that violates S1 (multiple working hypotheses). Composition
covariates (helix/sheet/hydro/charge, already computed elsewhere for a *different* object — pooled φ
vs the 8-axis framework) and the 8-axis joint-PCA framework (axis0=disorder ... axis7=acidic-helical,
`W_axes_8x640.npy`) were never cross-correlated against this specific discarded-mode object.

Ran `hMuscle/model/b_manifold_biology_match.py`: per-SLiM-pair top discarded mode (L9, sign-fixed SVD
direction of the non-DC residual, same object as `b_slim_dispersion_structure.py`) vs (a) composition
covariates and (b) axis0-7 directions. Full manifest population (n=776 resolvable pairs, 735 genes) —
NOT the n=20 ELM-instance set, so not subject to the CV-fold power wall (§5b, devils-advocate Attack 1
on the anchor-first B5 proposal). Purely descriptive/correlational, no supervision label involved, no
circularity risk (axis0-7 built from isoform-level pooled-embedding joint-PCA, wholly independent
procedure from the per-residue delta-SVD used here).

```
T1 (per-pair |cosine(mode,axis_k)| vs |composition covariate|, Bonferroni-32, perm-null p):
  axis0(disorder) x sheet_delta   r=+0.144  p=0.0005  SURVIVES
  axis7(acidic-helical) x helix_delta  r=+0.164  p=0.0005  SURVIVES
  (all other 30 combinations: not significant after correction)

T2 (gene-disjoint reproducible top-50D subspace vs W_axes 8 directions, random-K null, energy captured):
  axis0 (disorder)        0.326 vs null 0.076  excess +0.250   <- strongest by far
  axis2 (Pro-turn)        0.281 vs null 0.074  excess +0.207
  axis7 (acidic-helical)  0.222 vs null 0.078  excess +0.144
  axis1 (LRR/Ig)          0.169 vs null 0.082  excess +0.087
  axis6 (KRAB-ZNF)        0.166 vs null 0.078  excess +0.087
  axis4 (helix-charge)    0.150 vs null 0.079  excess +0.071
  axis3 (domain)          0.134 vs null 0.080  excess +0.054   <- weak
  axis5 (length)          0.109 vs null 0.074  excess +0.035   <- weakest
```

**Reading**: axis3(domain) and axis5(length) — the two axes a "generic/position-length" verdict would
predict should dominate — are the WEAKEST overlaps. axis0(disorder) dominates by a wide margin, with
axis2/axis7 secondary. The T1 per-pair correlations are small (~2% variance) but Bonferroni-survive and
point the same direction (axis0 relates to a compositional covariate, sheet propensity). **The
"generic, not SLiM-identity" verdict from `b_option_B_slim_target.py` stands (still not SLiM-class-
predictive), but "generic = position/length" was an unverified extra claim that this data does not
support — disorder-related geometry is the better-supported candidate identity.**

**Caveats (explicit, not yet closed)**: (1) T2's "captured energy" measures directional alignment
between subspaces, not functional/predictive power — axis0 itself is already `encoded but NOT used by
B4` per the map's core finding, so even if this discarded manifold IS substantially disorder-geometry,
that doesn't yet mean it's usable downstream (same encoding≠usage separation the map enforces
everywhere else). (2) No non-linear / functional test run yet against disorder-specific targets (only
the regex-SLiM target was tested previously). (3) T1 effect sizes are small; this is a *lead*, not a
closed finding.

Assets: `hMuscle/model/b_manifold_biology_match.py`. No label risk, cheap to re-run/extend (e.g. axis0-
specific non-linear disorder-boundary target, reusing the already-built `b_idr_boundary_pilot.py`
disorder machinery, would be the natural next test — separate from and NOT a revival of the B5
anchor-calibration line killed in this session's devils-advocate review).

### 5g. A1 (L9 anchor) formally closed — DEAD, confirmed with bootstrap CI (2026-07-25)

The task ledger had marked A1 "READY" (measured lead, execute when convenient). On attempting to
execute it this session, found `b_l9_anchor_ablation.py` had already been run once (2026-07-23,
untracked/uncommitted, no bootstrap CI saved, never written up) — orphaned result. Re-ran cleanly with
full stdout log:

```
L30 (production)  0.6899 [0.6757, 0.7037]
L9                 0.6794   Delta=-0.0106 [-0.0177,-0.0034]  CI excl 0
L9+L30 (the "domain-decodability-proxy winner")  0.6810   Delta=-0.0089 [-0.0165,-0.0018]  CI excl 0
L30-L9             0.6689   Delta=-0.0210 [-0.0279,-0.0141]  CI excl 0
L15+L30            0.6196   Delta=-0.0703 [-0.0790,-0.0622]  CI excl 0
```

Every L9-containing variant is significantly WORSE than production L30 on the real 18-GO
macro-AUPRC task, all CI excluding 0. Pre-registered prediction ("NO significant gain, consistent with
per-term layer-selection-not-exceeding-ceiling") is confirmed and exceeded — not merely no-gain, but
reliable loss. **The domain-decodability proxy gain (+0.028 AUROC, concat[phi9,phi30]=0.824 vs 0.796)
does not transfer to the deployed task.** A1 is dead; no further action. Lesson generalizes the
manuscript's existing "per-term layer selection does not exceed the ceiling" finding to this specific
probe. Table entry in §4 updated accordingly.

Assets: `hMuscle/model/b_l9_anchor_ablation.py`, `reports/model_interpretability_map/l9_anchor_ablation.json` (orphaned point-estimate run), full log `hMuscle/logs_isoform/b_l9_anchor_ablation_rerun_20260725_0007.log`.

### 5h. §5f disorder-axis lead — usage test, same fate as editcore (2026-07-25)

§5f found the discarded (non-DC) manifold's reproducible subspace overlaps axis0(disorder) far more
than domain/length (encoding fact, direction-alignment only). Tested whether this translates into
functional usage on the SAME target `b_idr_supervision_causal.py` already used (boundary_cross,
metapredict order<->disorder transition at the edit region) — extended that exact protocol
(`b_discarded_mode_idr_causal.py`) with one new channel: the top discarded mode (non-DC), alongside
the already-tested comp/pooled(DC)/editcore/confound-oracle.

```
comp only              0.608
comp+pooled(DC)        0.590
comp+editcore          0.575   (prior run, different n/seed: 0.556 — consistent direction)
comp+discard(non-DC)   0.628   <- first embedding channel to beat comp-only
comp+pool+discard      0.608   (combined with pooled, the edge vanishes)
oracle only            0.626   (positional + gene-gross, NO embeddings)
comp+oracle            0.640
```

**Predict-before-look (stated before running): "comp+discard will NOT decisively beat the oracle,
encoding-without-usage will replicate." CONFIRMED**: discard−oracle = +0.002 (0.628 vs 0.626,
statistically a tie). Gene-permutation null PASSED (z=+0.87, true AUROC within null band — no leakage,
honest null result).

**Reading**: the discarded/non-DC channel is the first embedding representation in this entire track
to beat pooled AND editcore internally for a disorder-adjacent target — a small positive signal that
the axis0 alignment from §5f isn't wholly illusory. But it still cannot separate itself from a
trivial no-embedding confound oracle (position + edit length + whole-protein disorder + protein
length), the exact same failure mode that killed editcore in §5e. **Encoding (axis0-alignment, §5f) ≠
usage (this test) — confirmed once more, on a new representation channel.** The disorder-axis lead
from §5f is not falsified as an encoding fact, but is now closed as a *functional* dead end for this
specific target; B5/B2 remains provisionally exhausted for cheaply-reachable labels (§5e verdict
stands, extended to this new candidate channel too).

Assets: `hMuscle/model/b_discarded_mode_idr_causal.py`, `reports/model_interpretability_map/assets/discarded_mode_idr_causal_results.txt`.

### 5i. A2 (edit-core/N-terminal additive channel) formally executed — DEAD across all 4 evaluation tiers (2026-07-25)

A2 was "GATED by Option A" pending resolution of two blockers found on attempting to execute it:
(1) "edit-core" as offline-defined (paired long/short edit-region embedding) has no direct analog in
PRISM's standalone per-isoform scoring pipeline; resolved by translating it to the most faithful
standalone-isoform feature -- N-terminal-60aa-window mean-pool at layer 30 (`NTERM_WIN=60`
convention already used elsewhere), motivated by the N-terminal degron classes that showed the
strongest beyond-composition offline signal (`b_elm_beyond_comp.py`, DEG_Nend_Nbox +0.239 etc).
(2) this requires per-residue N-terminal embeddings for the FULL 68K-isoform corpus (train 31,668 +
test 36,748), which didn't exist (only a small pilot subset did) -- ran a dedicated extraction
(`hMuscle/preprocessing/compute_esm2_nterm60_layer30.py`, layer-30-only + N-term-window-only to keep
it a modest incremental job, not a full per-residue re-extraction).

**Evaluation diversified beyond macro-AUPRC/DR-AUC per user request** (motivated by the standing
critique that GO-BP labels are gene-inherited corpus-wide, 11,594 unique terms in
`human_annotations_unified_bp.txt` all keyed by gene symbol -- confirmed empirically, so expanding
GO-term COUNT does not add isoform resolution; the correct existing tool is DR-AUC, generalized here
to 6 within-gene, gene-mean-immune covariate-ranking AUCs, `compute_isoform_level_covariates_a1a2pop.py`
+ `b_a2_nterm60_ablation.py`). Severity_pairs_scored.tsv's covariates could NOT be reused directly --
that population (brain/muscle isoquant assembly, e.g. `A1BG-204`) has ZERO id overlap with the A1/A2
population (Bambu-assembled ids, e.g. `BambuTx10`) -- so covariates were redefined as true per-isoform
scalars (within-gene mode-deviation / window-mean), not pairwise deltas, to slot into
`compute_domain_ranking_auc`'s exact design unchanged.

```
Tier 1 macro-AUPRC (18 GO-BP):      L30=0.6941[.6796,.7080]  L30+Nterm60=0.6574[.6411,.6730]
                                     Delta=-0.0367 [-0.0438,-0.0302]  CI excl 0 (WORSE)
Tier 2 domain-ranking AUC:          L30=0.5851[.5409,.6315]  L30+Nterm60=0.5966[.5571,.6420]  (CI overlap, no change)
Tier 3 nterm_deviates AUC:          L30=0.4090[.3863,.4322]  L30+Nterm60=0.3789[.3559,.4018]  (both <0.5, WORSE)
       disorder_nterm AUC:          L30=0.4605[.4386,.4801]  L30+Nterm60=0.4768[.4559,.4988]  (CI overlap)
       helix_nterm AUC:             L30=0.5141[.4880,.5393]  L30+Nterm60=0.5144[.4871,.5389]  (no change)
       sheet_nterm AUC:             L30=0.4313[.4054,.4540]  L30+Nterm60=0.4277[.3999,.4516]  (CI overlap)
       hydro_nterm AUC:             L30=0.4445[.4184,.4708]  L30+Nterm60=0.4274[.4009,.4530]  (CI overlap)
```

**Verdict: DEAD, and cleaner than A1.** A1 had exactly one decisive metric (macro-AUPRC, worse).
A2 has TWO decisive-direction results: macro-AUPRC significantly worse (CI excl 0), AND the ONE
covariate directly targeting A2's own hypothesized mechanism (nterm_deviates) also moved in the WORSE
direction (both variants below 0.5 chance, gap widened). None of the 6 covariates or domain-AUC showed
a CI-excluding improvement for L30+Nterm60 over L30. A predict-before-look registered before running
(informed by this session's 7 consecutive encoding-without-usage failures, §5f-§5h) correctly
anticipated no decisive gain anywhere; a competing external prediction (+0.10 AUC gain, p<0.01,
proposed mid-session) was decisively refuted in the opposite direction.

**Side finding (not directly supporting A2, flagged for future reference)**: `nterm_deviates` AUC is
significantly BELOW 0.5 even at baseline (L30 alone, CI excludes 0.5) -- isoforms whose N-terminal-60
sequence deviates from their gene's modal N-terminus score systematically LOWER for that gene's
typical GO-BP function, already in production PRISM. Adding the Nterm60 channel amplifies this
(anti-)correlation rather than correcting it. Plausible reading: N-term-deviant transcripts more often
represent truncated/atypical isoforms lacking canonical function, and PRISM already (partially)
tracks this via the standard L30 pooled representation -- not yet investigated further.

**This closes the "pooling-loss recovery" worklist (A1 dead, A2 dead, B5/SLiM-manifold dead across all
targets tried) completely.** No further items on this specific track without a genuinely new label or
architecture idea that clears the same 4-tier bar. Assets: `hMuscle/preprocessing/compute_esm2_nterm60_layer30.py`,
`hMuscle/model/compute_isoform_level_covariates_a1a2pop.py`, `hMuscle/model/extend_isoform_covariates_secstruct_hydro.py`,
`hMuscle/model/b_a2_nterm60_ablation.py`, `reports/model_interpretability_map/a2_nterm60_ablation.json`,
full log `hMuscle/logs_isoform/b_a2_nterm60_ablation_20260725_0320.log`.

### B1. Naming the beyond-composition signal — dipeptide-class Occam test (2026-07-25)

B1 ("richer positional/contextual descriptors, name the reproducible beyond-comp structure") tested:
does a cheap local-order descriptor (dipeptide-CLASS composition, 36-dim physicochemical bigrams --
NOT full 400-dim AA dipeptides, too sparse for ~20aa edit regions) already explain what ESM editcore
captures beyond mono-AA composition for the classes where that gain was real (`b_elm_beyond_comp.py`'s
top 5: DEG_Nend_Nbox_1 +0.239, DEG_Nend_UBRbox_2 +0.236, LIG_BIR_II_1 +0.197, DOC_MAPK_gen_1 +0.106,
DEG_Nend_UBRbox_1 +0.093)?

```
ELM class            comp   comp+dipep  dipep-beyond-comp   (ESM ec-beyond-comp, prior)
DEG_Nend_Nbox_1      0.608    0.637         +0.029                  +0.239
DEG_Nend_UBRbox_2    0.577    0.586         +0.009                  +0.236
LIG_BIR_II_1         0.653    0.595         -0.058                  +0.197
DOC_MAPK_gen_1       0.539    0.608         +0.070                  +0.106
DEG_Nend_UBRbox_1    0.677    0.663         -0.014                  +0.093
```

Predict-before-look ("N-end rule biology is about specific residue identity at specific positions,
not local class-pattern, so dipeptide-class should explain little of ESM's gain") CONFIRMED: dipeptide
explains 4-12% of ESM's gain in 3/5 classes, is negative in 2/5, and never comes close to closing the
gap (best case DOC_MAPK_gen_1 at 66%, still short). **Occam gate result: does NOT collapse to a cheap
local-order descriptor.** Positively narrows the open R2/R3 "what is this signal" question: the
N-terminal degron beyond-comp signal requires genuine positional/identity information (which exact
residue sits at which position), consistent with the N-end-rule/UBR-box recognition mechanism and
with why A2 (§5i, pooling destroys positional info) failed -- same underlying signal, two
independent lines of evidence now agree on its nature (positional identity, not composition or local
pattern).

Assets: `hMuscle/model/b1_dipeptide_vs_editcore.py`, `reports/model_interpretability_map/b1_dipeptide_vs_editcore.tsv`
(superseded/scoped-out: `b1_dipeptide_beyond_comp.py`, wrong target population -- project's own 9-class
SLIMS regex set is mono-comp-saturated, 0.65-0.80 AUC from mono alone, no beyond-comp question to ask there).

### B1b. Synthetic within-class-permutation pilot — first CLEAN positive beyond-comp result this session (2026-07-25)

Direct, scarcity-free, non-circular test of B1's "positional identity" hypothesis: insert a
textbook-validated SLiM (SV40 NLS "PKKKRKV", not a regex guess -- avoids the risk flagged in
`finding-elm-instance-power-degenerate`) at a random internal position in 150 real backbone sequences,
vs a control that permutes ONLY the within-class (basic-residue) positions (K,K,K,R,K reordered among
themselves), keeping the class-level sequence (special-basic-basic-basic-basic-basic-hydrophobic)
IDENTICAL between conditions -- forces mono-comp AND dipeptide-class-comp to be uninformative by
construction, isolating within-class residue-identity-order as the only remaining signal.

(First attempt used a full random permutation of the 7 residues as control -- confounded: letting P/V
land away from their true extremal positions let dipeptide-class alone solve it at 0.994 AUC, not
testing what was intended. Corrected design below.)

```
mono-comp only         AUC=0.500  (sanity check: identical multiset by construction)
mono+dipeptide-class   AUC=0.500  (class-level sequence identical by construction -- clean this time)
ESM editcore only      AUC=0.759  <- beyond both mono and dipeptide-class
mono+ESM editcore      AUC=0.759  (editcore-beyond-comp = +0.259)
```

**First clean, decisively POSITIVE beyond-composition result in this session's entire characterization
line** (contrast with A2 §5i, B1 dipeptide test above -- both negative/null for their respective
questions). ESM's local (editcore-style) representation genuinely encodes within-class residue-
identity placement (which exact position holds K vs R among 5 interchangeable-by-class basic
residues) -- information that is by construction invisible to both mono-AA and dipeptide-class
composition. Confirms B1's "positional identity" hypothesis via a synthetic, scarcity-free, fully
non-circular design (label defined by intentional construction, not mined from any external DB or
project regex, backbone-disjoint CV).

**Honest caveat**: the basic-residue control permutation has a 1/5=20% chance of landing back on the
true ordering by chance (K,K,K,R,K has only 5 distinguishable permutations), meaning some "control"
examples are indistinguishable from "motif" by construction -- 0.759 likely UNDERSTATES the true
achievable signal (optimistic-direction caveat, not a concern).

**Open question (not resolved here, next natural step)**: does this order-sensitive editcore signal
survive PRODUCTION mean-pooling, or does it meet the same fate as A2 (§5i, decisively dead once pooled
and evaluated on the real task)? This pilot only establishes that editcore/local representation CAN
carry it -- necessary but per [[approach-proxy-metric-vs-deployed-task-ablation]] (this session's
now-3x-confirmed pattern) NOT sufficient for a production gain claim.

Assets: `hMuscle/model/b1b_synthetic_motif_insertion_pilot.py`,
`reports/model_interpretability_map/assets/b1b_synthetic_motif_pilot_results.txt`.

### B1c. Multi-motif robustness + pooling-survival test — refines (does not repeat) the encoding-without-usage pattern (2026-07-25)

Extended B1b two ways: (C) 3 validated NLS motifs instead of 1 (SV40 "PKKKRKV" 7aa, c-Myc "PAAKRVKLD"
9aa, Nucleoplasmin bipartite "KRPAATKKAGQAKKKK" 16aa), 200 backbones each (n=1152 total after NaN
drop), same within-class-permutation control design. (A) added a POOLED feature (mean over the WHOLE
sequence, exactly matching production PRISM's real input construction) alongside editcore, to test
pooling survival directly.

```
motif              n     mono   mono+dipep  editcore  pooled   (pool-editcore gap)
SV40_NLS          384    0.500    0.500       0.746    0.550        -0.196
cMyc_NLS          384    0.500    0.500       0.905    0.663        -0.242
Nucleoplasmin_NLS 384    0.500    0.500       0.907    0.632        -0.274
combined         1152    0.500    0.500       0.745    0.573

editcore-beyond-comp = +0.245   pooled-beyond-comp = +0.073   pooling-survival-gap = +0.172
```

**(C) Robustness: CONFIRMED and strengthened.** All 3 independent, textbook-validated motifs show
mono/dipeptide exactly 0.5 (design sanity check passes every time) and editcore clearly beyond both
(0.746-0.907) -- not a fluke of one sequence choice (B1b's original SV40-only pilot).

**(A) Pooling survival: PARTIAL, not the clean total death seen in A2 (§5i).** Unlike A2's real-task
4-tier ablation (decisive death, adverse direction on the one mechanism-targeted covariate), here
pooled representation retains SOME signal above chance (0.550-0.663) rather than collapsing fully to
0.5. **The degree of survival scales with motif length relative to backbone length**: SV40 (7aa,
shortest) shows the steepest collapse (editcore 0.746 -> pooled 0.550, near-total loss), while
Nucleoplasmin (16aa, longest) retains the most (0.907 -> 0.632). This is a quantitative confirmation
of the already-established coherence-filter/dilution mechanism
(`frac_kept ~= 1/L + (1-1/L)*mean_coherence`, [[finding-pooling-coherence-filter]]) -- longer inserts
are a larger fraction of the pooled average, so less diluted, exactly as that formula predicts. This
also retroactively contextualizes A2's total death (§5i): production A2 used a 60-residue N-terminal
WINDOW (much larger than these 7-16aa motifs but still small relative to typical full-length proteins
of hundreds of residues), so A2's failure sits on the same dilution curve, at a point where survival
is apparently near-zero for this project's real backbone-length distribution and real (noisier, not
synthetic) supervision target.

**Reading**: this REFINES rather than repeats [[approach-proxy-metric-vs-deployed-task-ablation]] --
pooling loss is not a binary switch but a continuous dilution function of insert-length/total-length,
matching prior theory exactly. It does NOT establish that this partial synthetic survival (0.073
beyond-comp at the pooled level) would translate into a decisive REAL macro-AUPRC/covariate-AUC gain
under A2's rigorous 4-tier bar -- that remains untested and the magnitude here is modest, likely
still swamped by the same real-world noise/confounds that killed A2's real-population test.

Assets: `hMuscle/model/b1c_synthetic_multimotif_pooling.py`, `reports/model_interpretability_map/b1c_synthetic_multimotif_results.tsv`,
`reports/model_interpretability_map/assets/b1c_synthetic_multimotif_combined.txt`.

### B1d/e. Quantitative dilution law + real-SLiM validation — resolves whether a redesigned A2 (edit-size-matched window) would help, WITHOUT a new full-corpus ablation (2026-07-25)

**B1d (synthetic, quantitative)**: for each of the 515 valid (backbone,motif) pairs from B1c, computed
`ratio = ||pool_delta|| / ||editcore_delta||` and `insert_fraction = insert_length/full_length`.
Pearson r(insert_fraction, ratio)=0.482, p=2.2e-31. Log-log fit: `ratio ~= exp(-0.292) * insert_frac^0.480`
(R²=0.263, sublinear/~sqrt scaling, slower decay than naive 1/L dilution). Real SLiM manifest
insert_fraction (size/len_long): median=0.0408, IQR=[0.0198,0.0731] -- this predicts ratio~0.16 at the
real median if the synthetic law applied directly.

**B1e (real-data validation, cached data only, NO new GPU extraction)**: computed the SAME ratio
(`||HL.mean(0)-HS.mean(0)|| / ||HL[edit_region].mean(0)||`) directly on the 773 real SLiM manifest
pairs (reusing `b_perres/` cache from earlier this session). Result: **r=0.762, p=9.4e-148** (an even
cleaner power law than the synthetic data, R²=0.650) but **real ratio at the real median insert_fraction
(0.043) is only 0.024 -- roughly 6-7x MORE diluted than the synthetic law predicted (0.16).**

**Mechanistic resolution**: real SLiM edits suffer a COMPOUND loss the synthetic motifs did not --
(1) pure length-dilution (present in both), PLUS (2) intrinsic spatial INCOHERENCE, already established
this project (`finding-pooling-coherence-filter`: SLiM frac_kept 0.05-0.08 vs domain 0.16-0.31, "SLiM
spread is spatially incoherent"). The synthetic motifs (evolutionarily validated, functionally coherent
NLS sequences) only suffer effect (1); real alternative-splicing-induced edits suffer both (1) AND (2)
compounded, explaining the ~6-7x extra dilution gap.

**Decision (Occam, resource discipline)**: this ALREADY answers the question a full A2-redesign-and-
reablate cycle (edit-size-matched window instead of the fixed 60aa window) would have tested, using
only cached data (no new 68K-corpus GPU extraction). **A redesigned A2 with a better-matched window
size is unlikely to recover the signal** -- the incoherence gap is a separate, compounding barrier
that window-size alone does not address. Recommend NOT investing in a full re-ablation; this closes
the "was A2's window size the problem" question with a clear NO (it's a smaller contributor than
intrinsic edit incoherence).

Assets: `hMuscle/model/b1d_dilution_curve_vs_real_slim.py`, `hMuscle/model/b1e_dilution_law_real_slim_validation.py`,
`reports/model_interpretability_map/b1d_dilution_curve.tsv`, `reports/model_interpretability_map/b1e_real_slim_dilution.tsv`.
