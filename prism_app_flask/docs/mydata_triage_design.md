# MY DATA — Isoform Function Triage: design spec (v0 draft)

Status: **DRAFT under active design discussion (2026-07-29).** The upload contract and
ladder skeleton below are stable; the exact Tier-1 gate (IDR/compositional boundary) is
**pending a devils-advocate review** and may change. Do not implement the Tier-1 gate as
final until that section loses its ⏳ marker.

This supersedes the DTU × novel-GO "4-scenario (S1–S4)" donut on `/mydata`, which was found
to (a) force an *optional* axis (DTU needs a condition comparison) as a primary axis, and
(b) center the *weakest, unvalidated* axis (novel-GO: literature audit this session found
0/81 isoform-specific evidence and a flat 14–25% contradiction rate across score tiers).

---

## 1. Purpose & scope

`/mydata` is a **user-upload triage tool**: a researcher runs long-read sequencing, detects
isoforms (known + novel), and wants to know **which isoforms are worth following up**, at
**isoform resolution**, with **honest confidence**. Condition comparisons (CT vs X) are
**optional** — most users will not have them — so nothing in the primary classification may
require them.

Design principle: the primary axis must be simultaneously **(a) always computable without
conditions, (b) function-centric, (c) aligned with a validated signal** (not with the
model's unvalidated raw GO output).

## 2. Universe / provenance (read this before touching data)

Three DISTINCT universes exist in this project — do not conflate them (a prior design pass
wrongly proposed re-basing the brain app onto the training universe):

| universe | long-read caller | role | canonical source |
|---|---|---|---|
| 36,748 **BambuTx** (`my_isoform_list_fixed.npy`) | **Bambu** | model **training** (muscle-lineage) | `canonical_reference.tsv`, `domain_delta_v2.npy` |
| 63,994 / 39,728-detected (**brain**) | **IsoQuant** | **this app (`/mydata`)** | `canonical_reference_brain.tsv` |
| GENCODE v44 | — | reference coordinates | — |

The brain app is already on the correct (IsoQuant) universe. Its `/mydata` id-list is the
full GENCODE 672-gene reference (63,994) super-set; only 39,728 were IsoQuant-detected in
brain and carry ESM/TransDecoder/domain annotation. For a **real user upload** there is no
such mismatch: every uploaded isoform (known + novel) is the user's own detection and gets
domains/canonical/delta computed on it directly by the upload pipeline (§4).

## 3. The triage view: single ranked list, no hard gates

**Decided 2026-07-29 (devils-advocate RECONSIDER → UX fork resolved to "ranked list, no gates").**
The earlier draft used a hard Tier-1(domain/ORF) vs Tier-2(IDR/compositional) wall. It was
**retracted** because four attacks converged: a domain/ORF gate **buries** a large share of brain
alternative isoforms (only 41.7% show any domain-family change vs canonical on current app data —
the remainder is partly crosswalk-gap artifact, so the *true* IDR/compositional fraction needs the
clean upload pipeline to measure, but the direction is firm); the N-terminal/targeting carve-out is
**self-contradictory** (NLS/degron motifs live *in* IDRs); and the B5 IDR-supervision failure is
consistent with mean-pool **destroying** the signal (dilution-law), not the biology being absent —
so a "no consequence" tier label launders an architecture limitation as an isoform property.

### 3.1 The model
**One ranked list of every changed isoform vs its gene's canonical.** No tier walls — nothing is
hidden or demoted into a "not worth looking" bucket. Each row carries **per-feature honest
confidence badges**; the user reads consequence *and* its epistemic status in one glance, and
filters/sorts as they wish.

```
CHANGED ISOFORMS (ranked, 512)   [filter: type ▾] [sort ▾]
──────────────────────────────────────────────────────────
NDUFS2-215  ✓ domain-loss (Complex1_49kDa)  ✓ lit-matched   ▸
GRIN1-210   ✓ ORF-truncation               ⚠ unchecked     ▸
SRRM4-203   ⚠ IDR-insert — model can't validate
            └ lit: brain IDRs may be functional             ▸
MAPT-208    ⚠ compositional — not used at output            ▸
ABCA5-213   ✓ domain-gain (ABC_tran)        ✓ lit-matched   ▸
```

### 3.2 Default sort (avoids the edit-size confound)
Rank by **validated-consequence evidence first**, NOT by edit magnitude:
1. named-domain change (identity) or ORF change present  → top
2. then targeting-signal change
3. then IDR / compositional change
4. then known-isoform-with-novel-GO (annotation-gap)

Edit magnitude is shown as a **separate, transparent column**, never multiplied into the sort
(multiplying by `size_z` would re-create "big edits first = domain-change first" — Attack 5). Sort
is user-switchable (by score, by magnitude, by confidence).

### 3.3 Evidence badges — by KIND, not a confidence ladder
**Decided 2026-07-30.** An earlier draft badged IDR/compositional as "⚠ model can't validate,"
which conflated three distinct epistemic levels for a feature change and laundered a real,
measured descriptor into a distrust warning. The correct framing separates them: any change on an
isoform can be supported by different *kinds* of evidence, and a row simply shows which kinds it
carries. "Only structural-descriptor evidence" means "we can describe *how* it differs, function
open" — NOT "low confidence." (This also aligns triage with the individual-analysis page, which
already plots axis0-disorder / axis3-domain as legitimate coordinates.)

Three epistemic levels behind the kinds:
1. **descriptor (FACT)** — directly measured, model-independent (disorder_frac, composition;
   computed from sequence, *bypassing* the mean-pool that Attack 6 said dilutes prediction — so the
   descriptor survives even where the prediction can't).
2. **representation anchor** — position on the app's 8-axis joint-PCA coordinates (axis0≈disorder,
   axis3≈domain). Real geometry of how the model *places* the isoform; not an output-prediction
   driver (the axis0-output-usage claim was retracted as variance-confound), but a valid descriptor.
3. **functional-consequence prediction** — "this change drives GO function X." Only domain/ORF reach
   this level; compositional is "encoded but not used at output," IDR-supervision failed (B5).

**Row layout decided 2026-07-30 (2nd devils-advocate → reconcile with the user's "descriptor is
real evidence" point).** The descriptor IS legitimate evidence and is NOT maligned as "can't
validate" (the user's correction stands). BUT it is NOT a top-line badge next to the prediction,
because a 2nd review showed that (1) *proximity-fusion* — a descriptor badge adjacent to a novel-GO
badge gets mentally fused into "this disordered isoform does nuclear-import," lending false weight to
the weak prediction; (2) axis0/axis3 displacement badges are **redundant** with the raw descriptor
(measured |disorder Δ| vs |axis0 displacement| spearman ρ=0.625, and axis0 tracks disorder Δ no more
than axis3 does → a blurry composite, less interpretable than the raw value) — **cut them**; (3)
~8 flat badges/row re-creates the "too much info" this redesign exists to remove. So descriptor is
**relocated to an expandable per-row section / the individual-analysis page** (which already plots
axis0/axis3/disorder as legitimate coordinates), not demoted or maligned.

**Top-line badges (the row itself — these are the SORT keys):**
- `✓ domain change (Complex1_49kDa)` — named Pfam family gain/loss/truncation vs canonical. On
  **family identity**, not magnitude (identity = isoform-specific signal, DR-AUC 0.630 > gene-mean
  0.500; magnitude alone ≈ edit size, visible without a model). *Validated discriminator.*
- `✓ ORF change` — coding→non-coding / frameshift / large truncation. Direct sequence fact.
- `Novel-GO: <term> (0.72)` + `✓ lit-matched` / `⚠ lit-contradicted` / `⚠ unchecked` — the model's
  GO guess and its literature status. Clearly the *prediction*, spatially separated from descriptor.

**Expandable "structural characterization" (per-row ▾, or on the linked individual page) — real
measured differences that describe the isoform; NOT a function claim, NOT maligned:**
- `disorder Δ +0.30 vs canonical` (magnitude + direction).
- `composition shift (N-term helix / charge / …)`.
- `targeting motif (N-terminal MTS/NLS)` — **REMOVED (2026-07-30), audited and found insufficient**,
  not merely "unaudited" anymore. Null-first FPR audit (composition-matched shuffle-null, paired
  bootstrap CI n=1000, on the same N-term-60 sequence set as `hydro_nterm`/`charge_nterm`) of both
  regex candidates: monopartite `[KR]{3,}|[KR].{1,2}[KR]{2,}` — real hit 25.0% vs null 20.8%, gap
  significant (CI excludes 0) but **83% of hits are composition-noise-explainable**; bipartite
  Robbins-1991-consensus `[KR]{2}.{10,12}[KR]{3,}` — specificity improves (59% noise) but coverage
  collapses to 0.92%, too rare to be a useful per-isoform badge. A second, independent angle
  (biological ground-truth: GO:0005634 nucleus-annotated genes vs other-CC-annotated genes,
  `gene2go.gz`) confirms this from the recall side rather than the precision side: nuclear genes
  are only weakly enriched for either pattern (risk ratio 1.19× monopartite, 2.40× bipartite — but
  bipartite recall among true nuclear genes is just 1.4%). Both angles agree: the regex is weak on
  precision (round 1) AND on recall/discrimination (round 2) — this isn't a sparsity artifact, the
  detector itself is weak. Neither passes the "domain mechanism + math both hold" bar. A
  trustworthy detector needs a PSSM/HMM-based tool validated on a curated positive/negative set
  (e.g. NLStradamus/cNLS-Mapper-style) — out of scope for this pass. Full numbers: memory
  `finding-nls-regex-fpr-audit-dead.md`.
- (axis0/axis3 displacement **removed** — redundant per above.)
- Guardrails: noise threshold on |Δ|; ORF-confidence gate (a wrong TransDecoder ORF fabricates
  spurious composition); disorder_frac is a ratio (partly size-normalized) but show raw magnitude.

**Sort** by validated-discriminator strength (domain + ORF composite), then DTU-significance if
present — **never by descriptor magnitude** (would re-amplify the edit-size confound).

The boundary kept crisp everywhere: descriptor/anchor evidence explains **what** differs and how the
model represents it; it does **not** by itself establish **which function** changes. Both are shown,
each in its place, each labeled for what it is.

### 3.4 Filters / facets (replace the hard buckets)
Non-destructive views on the same one list — the user opts in, nothing is pre-hidden:
- consequence type (domain / ORF / targeting / IDR / compositional)
- structural novelty (FSM / NIC / NNIC)
- confidence (lit-matched only, domain-backed only, …)
- **no-protein-reference** facet: genes with no CDS canonical (`no_CDS`, ~4.7% of brain genes) and
  NNIC in novel loci have no canonical anchor → domain/ORF undefined; shown as "novel, no protein
  reference," not silently dropped.

### 3.5 DTU overlay (optional)
If the upload has condition groups, significant-DTU isoforms get a `DTU` badge and an optional
"DTU first" sort. Absent condition data the list is fully functional — DTU never gates anything.

## 4. Minimal upload data contract (what the tutorial pipeline must produce)

Backward-designed from §3 so the ladder is computable. Per uploaded isoform:

| field | needed for | how (tutorial step) |
|---|---|---|
| isoform id, gene assignment | everything | IsoQuant/long-read output |
| protein ORF sequence | ESM → PRISM GO scores; domains | TransDecoder |
| **structural category** (FSM/ISM/NIC/NNIC) | novelty axis | SQANTI3 |
| **canonical flag per gene** (MANE→Ensembl→APPRIS→longest) | vs-canonical anchor | `build_canonical_reference*` logic |
| **Pfam domains** (hmmscan, i-Eval<1e-3) | domain_delta | hmmscan on ORFs — **run on ALL isoforms incl. novel**, not only GENCODE-name-mappable ones (the current app's crosswalk drops 44.5%; the upload pipeline must not) |
| disorder_frac + N-term composition | Tier-2 / targeting | metapredict + simple composition |
| PRISM GO scores | novel-GO, predictions | model forward pass |
| *(optional)* per-condition expression | DTU overlay | DRIMSeq/condition counts |
| *(optional)* abundance / read count | filtering | quantifier |

Derived server-side (not uploaded): **domain_delta** = per-isoform domain matrix − its gene's
canonical row (1-step derivation from `domain_matrix_brain_full.npy` + `canonical_reference_brain.tsv`;
for uploads, from the uploaded domains + canonical flag). **This is what the current app is
missing** — it falls back to a "max-domain-span sibling" reference, which is confounded
(structurally loss-biased) and should be replaced by the real canonical anchor.

## 5. What this fixes vs S1–S4
- No optional axis forced as primary (DTU → optional overlay).
- Function stays the front door, but Tier gating uses **validated** signal (domain/ORF), so
  the weak novel-GO axis is isolated to predictions + confidence badges, not tiering.
- Ladder (priority order) instead of a 2×2 grid → answers "what first," lower cognitive load.
- Every isoform tiered → no empty-row / Solo degeneracy.

## 6. Open decisions (tracked)
1. ✅ RESOLVED (2026-07-29): hard Tier wall retracted → **single ranked list, no gates, per-feature
   honest badges** (§3). Rejected: honest-tiers (still separates IDR into a tier → readable as
   "domain-only is real"), tissue-conditional (assumes brain-IDR-function before measuring it).
2. Verify on real upload data that domain_delta *identity* has signal beyond edit size (null:
   size-matched). Validated in research via DR-AUC; re-confirm on upload artifacts.
3. Whether N-terminal/targeting gets a lower-confidence badge inside Tier 1 (and whether that
   re-introduces the "too much info" gradient the redesign is trying to remove).
4. Demo data: keep the 63,994 GENCODE-superset as the shipped demo, or ship a smaller
   IsoQuant-only detected set closer to real user data.
5. **Canonical-quality caveat (found during 2026-07-30 impl verification):** where
   `canonical_reference_brain.tsv` fell back to `longest_CDS` (1,435 genes; MANE 72% / Ensembl 14%
   are fine), the chosen canonical may not be domain-representative — e.g. NDUFS2's longest_CDS
   canonical lacks Complex1_49kDa, so its alt isoforms read as domain-**gained** (direction flipped).
   The badge honestly reports "differs from canonical in domain X," but gain/loss *direction* is
   unreliable for longest_CDS-anchored genes. Options: prefer a domain-richest tie-breaker within the
   MANE/Ensembl/APPRIS tiers, or flag longest_CDS-anchored rows.

## 7. Implementation progress (2026-07-30)
- ✅ `precompute/build_canonical_map.py` → `data/isoform_index/brain/canonical.json` (17,635 genes,
   canonical present in index; no_CDS 879 skipped). Replaces the confounded "max domain-span sibling"
   reference.
- ✅ Per-isoform evidence computation logic verified on real data (domain-change vs canonical,
   disorder-Δ descriptor, structural type, no-ref / is-canonical / not-computed branches all correct).
- ✅ **(a) Backend `triage_ranked` endpoint** — `data_layer/dataset_summary.py` (`_domain_changes`,
   `_novel_go_owner_map`, `_triage_rows_raw`, `triage_ranked`) + `loaders.canonical_map()` +
   `blueprints/mydata.py::api_triage_ranked` (`GET /api/summary/<tissue>/triage_ranked`). 45,764 alt
   rows on brain_672 (2+-isoform genes only, canonical row excluded); `_TRIAGE_KIND_RANK` sorts
   domain(4) > orf(3) > idr(2, disorder-Δ noise floor 0.1) > novel_go(1) > none(0), never by
   magnitude — matches §3.2/3.3. `muscle` degrades gracefully (no isoform_index built for it — all
   rows fall back to `kind='none'`/novel_go-only, verified). Raw per-(tissue,threshold,hiconf)
   evidence list is `lru_cache`d (~4s cold / genes-loop cost, same order as `_novel_go_venn`);
   filter/sort/cap apply per-request on the cached list.
- ✅ **(b) Frontend ranked-list panel** — `templates/mydata.html` (`#triagePanel` replaces the old
   `#scenPie` "4-scenario classification" panel) + `static/js/mydata.js` (`loadTriageRanked`,
   `renderTriageRow`, `triageEvidenceBadges`, `toggleTriageDetail`). Top-line badges = domain/ORF
   (✓, green `badge match`) + novel-GO (plain, green only if hiconf) per §3.3; descriptor
   (disorder Δ) and no-canonical-ref note are expandable-only (▸/▾ toggle row), never top-line.
   Consequence/structural-type/sort dropdowns re-fetch on change.
- ✅ **(c) Live Playwright verification** (2026-07-30, against the running dev server on :8600) —
   panel renders 500/45,764 rows with correct badges; `sort=score` vs `sort=consequence` produces
   genuinely different top-6 ordering (confirmed both via curl and in-browser); `consequence=domain`
   filter zeroes the other kind_counts correctly; expand-row click reveals the disorder-Δ descriptor
   text; zero browser console errors. Screenshot confirms CSS renders cleanly (reuses existing
   `badge`/`bi-tablewrap`/`seg-sel` classes, no new CSS needed).
- ✅ **(d) Scenario reference cleanup** — removed the `scenario` option from the UMAP "Colour by"
   dropdown (`mydata.html`) and its render branch (`mydata.js` `drawUmap`), and removed the
   `scenario` column from the UMAP-selection isoform table (`#umapIsoTbody`). Scope note: the
   backend `_classify_cached`/`SCENARIO_META`/`scenario` UMAP color-by case in
   `dataset_summary.py` and the separate "Functional-switch candidates (S1)" table
   (`switch-tbody`) were deliberately left untouched — out of scope for this pass, not mentioned in
   the (a)-(d) plan; harmless dead paths if never selected from the UI.
- ✅ **(e) Structural-type filter chips** (2026-07-30, same session) — replaced the
  `#triageStructType` `<select>` with a single-select `.sib-chip` toggle group (all/FSM/NIC/NNIC),
  matching the existing sibling-chip pattern in `individual.js`. Playwright-verified: clicking a
  chip re-fetches and correctly narrows `total` (all 45,764 → FSM 38,361 → NNIC 4,426), active-chip
  state toggles correctly, zero console errors.
- ❌ **Targeting-motif (N-term MTS/NLS) badge — investigated this session, DEAD, not shipped.**
  §3.3's placeholder is now updated in place (see above) with the actual audit numbers rather than
  "unaudited": both a monopartite and a bipartite classical-NLS regex were run through a null-first
  FPR audit (composition-matched shuffle-null + paired bootstrap CI) on the app's real N-term-60
  sequence set. Monopartite: 83% of hits are noise-explainable. Bipartite: noise share improves to
  59% but coverage collapses to 0.92% — neither is trustworthy enough for a per-isoform badge. Full
  numbers and reasoning: memory `finding-nls-regex-fpr-audit-dead.md`. A real detector needs a
  PSSM/HMM-based tool validated on a curated positive/negative set — a separate research task.
- `no-protein-reference` and `structural novelty` facets (§3.4): `structural_type` is now a chip
  group (item e above); `no-protein-reference` is still a row-level badge/expand note, not a
  separate filter chip — minor completeness gap, not a blocker.
