"""Tutorial — academic-blog-style knowledge base with a sidebar TOC.
Content is written per-topic in SECTIONS; sections may use 'code' (short snippet) or 'body'
(long-form HTML: subsections, code cells, figures) — see tutorial.html for the render rule."""
from flask import Blueprint, render_template

from prism_app_flask import config

bp = Blueprint('tutorial', __name__)

ARCHITECTURE_BODY = r"""
<div class="tut-quicknav">
  <span class="tut-quicknav-label">On this page</span>
  <a href="#arch-backbone">1. Backbone choice: why a per-residue protein LM</a>
  <a href="#arch-training">2. Training configuration &amp; entrypoint</a>
  <a href="#arch-ablation">3. Ablation discipline: single-component removal</a>
  <a href="#arch-repro">4. Reproducing a checkpoint end to end</a>
</div>

<h3 id="arch-backbone">1. Backbone choice: why a per-residue protein LM</h3>
<p>The backbone requirement follows directly from the two DIFFUSE failure modes in Topic 02 §1: the
representation has to expose <i>isoform-intrinsic</i> sequence signal, and it has to do so at
per-residue resolution so that domain-level and motif-level edits are geometrically distinguishable
(Topic 02 §5). A protein language model pretrained on masked-residue prediction across evolutionary
sequence diversity satisfies both — it never sees gene identity, and it exposes a full per-layer,
per-residue hidden-state tensor rather than a single pooled gene-level vector. ESM-2 is available at
several parameter scales (t6/t12/t30/t33/t36); production settles on <b>t30 (150M params)</b> as the
practical fixed point on a shared GPU server — see the resource-discipline note in Topic 01 — trading
some capacity headroom against throughput and memory footprint that would otherwise force smaller
batch sizes and slower iteration during the ablation cycles in §3.</p>
<p>The figures below are the standard sparse-class benchmark pair (AUPRC primary, AUROC secondary —
Topic 02 §8) run through <code>hMuscle/results_isoform/evaluation.py</code> against the comparison
baselines in the same evaluation harness. Read the AUPRC panel first: on the sparse positive classes
this project cares about, AUROC can look deceptively similar across models while AUPRC — which is
sensitive to exactly the class-imbalance regime here — separates them.</p>
<figure class="tut-figure tut-figure-pair">
  <img src="/static/figures/comparison_auprc.png" alt="Macro-AUPRC comparison across baselines" loading="lazy">
  <img src="/static/figures/comparison_auroc.png" alt="Macro-AUROC comparison across baselines" loading="lazy">
  <figcaption>Macro-AUPRC (primary) and macro-AUROC (secondary) comparison, regenerated directly from
  <code>hMuscle/results_isoform/evaluation.py</code> — see §4 to reproduce.</figcaption>
</figure>

<h3 id="arch-training">2. Training configuration &amp; entrypoint</h3>
<p>Per-GO-term training uses the configuration already derived in Topic 02 §2 (focal loss γ=2, Adam
lr=1e-3, batch 512, early stopping patience 10 with LR-on-plateau, 5-seed ensemble). The production
entrypoint wraps this over the full GO-term set on GPU:</p>
<pre><code># environment
conda activate isoform_env

# full production run (backgrounded, logged — long-running)
nohup python run_GPU_Full.py &gt; ../logs_isoform/run_$(date +%Y%m%d_%H%M).log 2&gt;&amp;1 &amp;
tail -f ../logs_isoform/$(ls -t ../logs_isoform/ | head -1)

# production model definition / training loop
hMuscle/model/v15d_bp_clean.py</code></pre>
<p>Resource discipline on a shared server applies here directly (see project rules): thread counts
capped via <code>OMP_NUM_THREADS</code>/<code>MKL_NUM_THREADS</code>, a single GPU pinned per run, and
<code>nice</code>d if launched alongside other jobs.</p>

<h3 id="arch-ablation">3. Ablation discipline: single-component removal</h3>
<p>Component contribution is never asserted from an architecture description alone — it is shown as a
number from a <b>no_X</b> ablation: the full pipeline re-run with exactly one component removed or
swapped, evaluated with the same bootstrap-CI + null/oracle discipline as any other improvement claim
(Topic 02 §8). This project does not use a single flag-driven training script for ablations — each
ablation is its own standalone, independently runnable script that reproduces the relevant slice of
the pipeline with one change, so the diff between "baseline" and "ablated" is explicit and inspectable
rather than hidden behind a conditional. Representative examples actually in the codebase:</p>
<pre><code># backbone dimensionality ablation
python hMuscle/model/esm2_640dim_ablation.py

# pooling-window ablation
python hMuscle/model/exp_B2_window_ablation.py

# anchor-layer ablation
python hMuscle/model/b_l9_anchor_ablation.py</code></pre>
<p>The standing rule (project experiment discipline): a component is only credited with a contribution
once its removal is shown to hurt a bootstrap-CI-backed metric against an explicit null/oracle — "we
included it because it seemed like it should help" is not a result.</p>

<h3 id="arch-repro">4. Reproducing a checkpoint end to end</h3>
<p>Combining Topic 01 (data) with this topic's training entrypoint gives the full chain from raw reads
to a scored isoform × GO matrix:</p>
<pre><code># 1) isoform quantification + ORF extraction — Topic 01
bambu -r reads.bam -a annotation.gtf -g genome.fa
TransDecoder.LongOrfs -t transcripts.fa

# 2) per-layer ESM-2 embeddings (preprocessing)
python hMuscle/preprocessing/compute_esm2_all_layers.py

# 3) train / score
conda activate isoform_env
python run_GPU_Full.py            # → *_scores.npy

# 4) evaluate (produces the comparison figures above)
python hMuscle/results_isoform/evaluation.py</code></pre>
<div class="tut-callout">
<b>Why the ablations are separate scripts, not flags.</b> A shared flag-driven trainer invites a
subtle failure mode: a flag that's supposed to be a no-op when unset can silently change unrelated
defaults elsewhere in the same code path. Separate, independently runnable ablation scripts make the
exact diff between baseline and ablated configurations something you can literally read, not something
you have to trust.
</div>
"""

BISECT_GUIDE_BODY = r"""
<div class="tut-quicknav">
  <span class="tut-quicknav-label">On this page</span>
  <a href="#bg-tiers">1. Tier levels — what PRISM Tier 1/2/3 mean</a>
  <a href="#bg-confidence">2. The confidence score</a>
  <a href="#bg-pathway">3. The causal pathway strip</a>
  <a href="#bg-domains">4. Domain gain/loss chips</a>
  <a href="#bg-notes">5. Interpretive notes</a>
  <a href="#bg-worked">6. Worked example — NDUFS8</a>
</div>

<h3 id="bg-tiers">1. Tier levels — what PRISM Tier 1/2/3 mean</h3>
<p>Every BISECT case is assigned exactly one tier, in priority order (checked top to bottom against
the case's structural evidence — the first rule that matches wins):</p>
<pre><code>tier1_functional_switch    confident AlphaFold-supported domain GAIN present
tier2_functional_loss      confident AlphaFold-supported domain LOSS present
tier2_complex_loss         gene is a Complex I subunit (NDUFS4 / NDUFS7 / NDUFS8) — special-cased
tier2_partial_change       domain gain/loss present but not AlphaFold-confident
tier2_gain_no_direction    partial change, net direction toward gain
tier3_gene_median          representative-sequence estimate (no direct isoform-level match)
tier3_structural_only      structural evidence only, no PRISM GO match
tier3_no_match             no supporting evidence found</code></pre>
<p>Tier is a <b>structural</b> summary — it answers "how much domain-level evidence is there," not
"how functionally important is this switch." That second question is what the confidence score and
narrative notes below are for. A Tier&nbsp;1 case with low confidence is common: domain evidence can
be clean while the surrounding regulatory/conservation/PPI evidence is thin.</p>

<h3 id="bg-confidence">2. The confidence score</h3>
<p>Confidence is <i>not</i> a model output — it is an evidence-module count. Eleven independent
checks are evaluated per case (DTU significance, AlphaFold Δ-confidence, domain gain/loss, STRING
PPI support, cross-species conservation, TF/ASF regulator hits, non-transcriptional mechanism,
TSS shift, APA shift, NMD relevance, and an exact PRISM GO match), and the count is bucketed:</p>
<pre><code>0-1 evidence modules  → Low
2-3 evidence modules  → Moderate
4-5 evidence modules  → High
6+ evidence modules   → Very High</code></pre>
<p>Reading it this way matters: two cases at "High" confidence can be high for entirely different
reasons — one from strong structural + conservation evidence, another from strong regulatory +
mechanism evidence. Always open the evidence detail rather than trusting the label alone.</p>

<h3 id="bg-pathway">3. The causal pathway strip</h3>
<p>The pathway strip renders the case's inferred causal chain as an ordered list of steps — e.g.
<i>Alternative Splicing → TF/ASF activity change → Isoform-ratio switch (DTU) → Domain composition
change → GO functional-space reshaping</i>. Steps are only included when their supporting evidence
exists for that case (a case with no detected TSS shift simply omits that step) — the strip is a
<i>subset</i> of the full possible chain, not a fixed template with blanks.</p>

<h3 id="bg-domains">4. Domain gain/loss chips</h3>
<p>Gained and lost Pfam domains are listed with a short function gloss (e.g. <code>Kinesin</code> →
"microtubule-based motor activity"). A domain with no entry in the internal function map is labeled
<code>function uncharacterised</code> rather than guessed — this is deliberate: an unlabeled domain
is a prompt to look it up externally (Pfam/InterPro), not a claim that it does nothing.</p>

<h3 id="bg-notes">5. Interpretive notes</h3>
<p>Below the headline, each case shows a short list of labeled interpretive notes — DTU (usage-share
direction), Structure (AlphaFold stability comparison), PPI (a new STRING-predicted interaction),
Conservation (cross-species phyloP support), and NMD risk (nonsense-mediated-decay-sensitive
structure requiring translation verification). These are deliberately kept separate from the numeric
evidence grid above them: the grid carries facts, the notes carry the one-sentence <i>reading</i> of
each fact — no number is repeated in both places.</p>

<h3 id="bg-worked">6. Worked example — NDUFS8</h3>
<p>NDUFS8 is one of three Complex&nbsp;I subunit genes (with NDUFS4, NDUFS7) that are special-cased
into <code>tier2_complex_loss</code> regardless of raw domain evidence, because a partial-loss switch
in any Complex&nbsp;I subunit has an outsized functional consequence — assembly failure of the whole
complex — that a generic domain-gain/loss rule would understate. The figure below is the underlying
per-layer subflow for this gene (independent of the BISECT case UI, generated by the same brain-672
reference space the app uses) — it is a useful companion view when reading the BISECT report: the
report tells you <i>what</i> changed functionally, this shows <i>where in the representation</i> the
CT→AD trajectories diverge.</p>
<figure class="tut-figure">
  <img src="/static/figures/subflow_NDUFS8.png" alt="NDUFS8 per-layer trajectory subflow, CT vs AD isoform" loading="lazy">
  <figcaption>NDUFS8 CT↔AD isoform-switch trajectory through the 30-layer representation — the same
  gene BISECT flags as <code>tier2_complex_loss</code> from structural evidence alone.</figcaption>
</figure>
<div class="tut-callout">
<b>How to read a case end to end.</b> Tier tells you the structural-evidence class; confidence tells
you how many independent evidence modules corroborate it; the pathway strip tells you the inferred
mechanism chain; the domain chips and notes fill in the specific biology. Read them in that order —
tier and confidence first (is this case worth attention at all), then the mechanism and biology detail.
</div>
"""

DATA_BODY = r"""
<div class="tut-quicknav">
  <span class="tut-quicknav-label">On this page</span>
  <a href="#data-env">1. Environment &amp; package installation</a>
  <a href="#data-raw">2. Raw data this pipeline expects</a>
  <a href="#data-preprocess">3. Preprocessing: reads → per-residue ESM-2 embeddings</a>
  <a href="#data-train">4. Training PRISM</a>
  <a href="#data-bisect">5. Running BISECT on your own switch calls</a>
  <a href="#data-app">6. Populating this app with your own data</a>
</div>

<h3 id="data-env">1. Environment &amp; package installation</h3>
<p>Everything below assumes a single conda environment (<code>isoform_env</code>) shared by
preprocessing, training, BISECT, and this Flask app. Three separate requirement files exist because
the three stages have genuinely different dependency footprints (deep learning vs. bioinformatics
utilities vs. the web app itself) — installing all three into one env is intentional, not an
accident of repo history.</p>
<pre><code># clone
git clone https://github.com/leeeeeff/PRISM-BISECT.git
cd PRISM-BISECT

# one shared conda env for the whole pipeline
conda create -n isoform_env python=3.9
conda activate isoform_env

# PRISM: deep learning + ESM-2 (torch, fair-esm, tensorflow for the Keras training head)
pip install -r requirements_training.txt
pip install tensorflow==2.12

# BISECT: evidence-pipeline utilities (requests/Jinja2 for case reports, biopython, ...)
pip install -r requirements_bisect.txt

# this Flask app (adds only flask + gunicorn on top of the above — numpy/pandas already present)
pip install -r prism_app_flask/requirements.txt</code></pre>
<p>BISECT's domain-annotation module (M2) additionally needs <b>HMMER</b> installed as a system
binary (not a pip package) plus the Pfam-A HMM database:</p>
<pre><code># HMMER (hmmscan) — via conda-forge, or your package manager
conda install -c bioconda hmmer

# Pfam-A.hmm — https://www.ebi.ac.uk/interpro/download/pfam/
hmmpress Pfam-A.hmm   # build the binary index hmmscan needs</code></pre>
<div class="tut-callout">
<b>GPU is required for §3–4, not for the app itself.</b> ESM-2 extraction and PRISM training both
need CUDA (tested on CUDA 12.x). This Flask app only reads precomputed <code>.npy</code>/<code>.json</code>
indices at runtime (§6) and runs fine on CPU — you do not need a GPU merely to browse or deploy it.
</div>

<h3 id="data-raw">2. Raw data this pipeline expects</h3>
<p>Nothing in this repository ships the raw sequencing data itself — it is available on request (see
the Data Availability table below), reflecting normal practice for in-house patient/tissue material.
What the pipeline needs, concretely:</p>
<pre><code>long-read scRNA-seq BAM  (ONT, aligned)          — one per tissue/condition
reference genome FASTA + annotation GTF          — species-matched to the BAM
Pfam-A.hmm                                       — BISECT M2 domain annotation (§1)
ESM-2 checkpoint (esm2_t30_150M_UR50D)           — auto-downloaded by fair-esm on first call, no manual step</code></pre>
<p>The muscle and brain datasets this project's own numbers come from are in-house (ONT long-read for
muscle; IsoQuant-called GTF with 10,817 novel isoforms for brain AD/CT) — available upon request, not
bundled. A public 42-sample SRA validation cohort is listed in
<code>Final_analysis/pipeline_bioanalysis/cases_input_sra.csv</code> and is the easiest way to exercise
the full pipeline end to end without requesting the in-house data first.</p>

<h3 id="data-preprocess">3. Preprocessing: reads → per-residue ESM-2 embeddings</h3>
<p>Three stages, each a deterministic transformation of the last (this is stage B0→B1 of the
representation cascade in Topic 02 §4 — the biology happens before any of it):</p>
<pre><code># 1) long-read isoform quantification → GTF + per-isoform counts
bambu -r reads.bam -a annotation.gtf -g genome.fa
# (brain uses IsoQuant instead of Bambu — same role, different caller)

# 2) ORF calling → translated protein sequence per isoform
TransDecoder.LongOrfs -t transcripts.fa

# 3) per-residue, all-30-layer ESM-2 embeddings (single forward pass, all layers at once)
cd hMuscle/preprocessing/
CUDA_VISIBLE_DEVICES=0 python compute_esm2_all_layers.py --batch_size 32
# → hMuscle/data/esm2_layer_NN_t30_150M.npy  (NN=01..30), shape (n_isoform, 640), float32</code></pre>
<p>Domain-level covariates (Pfam gain/loss, used throughout BISECT and the app's domain chips) are
built from the same protein sequences via a separate <code>hmmscan</code> pass, then assembled into a
matrix with one of the <code>build_*domain*.py</code> scripts in <code>hMuscle/preprocessing/</code>
(e.g. <code>build_train_domain_from_hmmscan.py</code> for the training population,
<code>build_domain_matrix_v3.py</code> for later dataset versions) — these are dataset-shape-specific
rather than one universal entrypoint, because the crosswalk from transcript ID to HMMER's output
differs slightly per annotation source (Bambu vs. IsoQuant naming).</p>

<h3 id="data-train">4. Training PRISM</h3>
<p>Once §3's per-layer embeddings exist, training and evaluation are two commands (full architecture,
loss, and hyperparameters are derived from first principles in Topic 02 §2–3; ablation discipline in
Topic 05 §3):</p>
<pre><code>conda activate isoform_env
cd hMuscle

# train production model (v15d_bp_clean) directly
python model/v15d_bp_clean.py

# or the backgrounded, logged production entrypoint (long-running — resource discipline: pin one
# GPU, cap thread counts, nice if sharing the server)
OMP_NUM_THREADS=4 nice -n 10 nohup python run_GPU_Full.py \
  &gt; logs_isoform/run_$(date +%Y%m%d_%H%M).log 2&gt;&amp;1 &amp;
tail -f logs_isoform/$(ls -t logs_isoform | head -1)

# evaluate → macro-AUPRC/AUROC vs. baselines (Topic 03)
python results_isoform/evaluation.py</code></pre>

<h3 id="data-bisect">5. Running BISECT on your own switch calls</h3>
<pre><code>cd Final_analysis/pipeline_bioanalysis
cp config.yaml.example config.yaml   # fill in local paths (genome, Pfam-A.hmm, ESM-2 cache, ...)

# full case list (15-module pipeline, M1 sequence extraction → M15 cross-case comparison)
python orchestrate.py --cases cases_input_sra.csv --output outputs/

# a single gene/cell-type case
python orchestrate.py --gene KIF21B --cell_type Excitatory</code></pre>
<p>BISECT needs a differential-transcript-usage (DTU) call as its starting point — a gene, a
cell type, and which two isoforms swap dominance between conditions. That upstream DTU step
(DRIMSeq or equivalent on your own condition-labeled long-read data) is outside BISECT's own scope;
<code>cases_input_sra.csv</code> is a pre-computed example case list so you can run the 15 downstream
modules without first standing up your own DTU pipeline.</p>

<h3 id="data-app">6. Populating this app with your own data</h3>
<p>This app never reads the ~600-dimensional score matrix or embeddings directly at request time —
it reads small precomputed indices built once by the scripts in <code>prism_app_flask/precompute/</code>,
then serves them from a fast on-disk/mmap format. To point the app at a new dataset (a new tissue, or
a rerun of an existing one), regenerate the index for that dataset:</p>
<pre><code>conda activate isoform_env

# core per-isoform index: 8-axis position + percentile, per-layer trajectory magnitude,
# peak-information layer, gene→isoform / id→row lookup tables
python prism_app_flask/precompute/build_isoform_index.py

# domain architecture (Pfam gain/loss chips) for the same isoform population
python prism_app_flask/precompute/build_brain_domains.py

# sequence covariates (disorder fraction, N-terminal helix propensity, charge/aromatic content, ...)
python prism_app_flask/precompute/build_brain_covariates.py

# gene → canonical-isoform anchor (MANE &gt; Ensembl_canonical &gt; APPRIS &gt; longest-CDS), used by
# the /mydata triage ranked list to compare every isoform against its own gene's reference
python prism_app_flask/precompute/build_canonical_map.py

# per-GO × per-layer discriminability (Fisher LDA AUROC) — descriptive only, not a deployed-performance claim
python prism_app_flask/precompute/build_go_layer_fisher.py

# BISECT-specific: per-case CT/AD usage fractions + disorder/domain tracks for the report UI
python prism_app_flask/precompute/build_bisect_dtu.py
python prism_app_flask/precompute/build_bisect_tracks.py</code></pre>
<p>Each script's own docstring documents its exact input paths and output shape — read it before
running, since a couple (notably <code>build_bisect_dtu.py</code>) read from an external
collaborator's DTU output directory that is not part of this repository and will need to be pointed at
your own DTU results.</p>
<p>Once the index exists, launch (or restart) the app:</p>
<pre><code>./prism_app_flask/run.sh dev     # Flask dev server, http://0.0.0.0:8600
./prism_app_flask/run.sh prod    # gunicorn, 2 workers, --preload (shared-memory index across workers)
./prism_app_flask/run.sh stop</code></pre>
<div class="tut-callout">
<b>Why precompute instead of computing on request.</b> The per-isoform index (axis position, layer
trajectory, canonical anchor, ...) is the same fixed lookup for every request that touches a given
isoform — recomputing an 8-axis projection or a Fisher AUROC per page load would turn an O(1) lookup
into unnecessary repeated work. <code>run.sh prod</code>'s <code>--preload</code> flag exists for the
same reason one level up: the index is loaded once by the gunicorn master and shared copy-on-write
across workers, rather than re-read per worker.
</div>
"""

GLOSSARY_BODY = r"""
<dl class="tut-glossary">
  <dt>editcore</dt>
  <dd>The set of residues actually altered by an isoform edit (insertion/deletion/substitution
  relative to a reference isoform), as opposed to the full sequence context around it. Whether a
  signal survives mean-pooling depends on the <i>directional coherence</i> of the editcore's
  per-residue effect — see Topic 02 §5.</dd>

  <dt>axis (8-axis map)</dt>
  <dd>One of 8 interpretable directions in the 640-dim ESM-2 embedding space, obtained by a
  per-layer-standardized, jointly-fit PCA over all 30 layers (Topic 02 §6). Axes are named only
  after passing a within-gene variance-decomposition screen and a usage check (Topic 02 §7) —
  encoded-but-unused axes are reported as such, not renamed away.</dd>

  <dt>trajectory</dt>
  <dd>An isoform's sequence of 8-axis coordinates across all 30 ESM-2 layers, <span
  class="tut-math-inline">\(Z_{i,1:30,:} \in \mathbb{R}^{30\times 8}\)</span> — the "path" a
  representation takes from early to late layers. Visualized directly in the 3D PCA trajectory panel
  (Individual Analysis, panel 3D) and in the per-gene subflow figures used throughout this app.</dd>

  <dt>coherence-filter / frac_kept</dt>
  <dd>The fraction of an edit's per-residue perturbation energy that survives mean-pooling,
  <span class="tut-math-inline">\(\text{frac\_kept} \approx 1/L + (1-1/L)\bar c\)</span> where
  <span class="tut-math-inline">\(\bar c\)</span> is the mean pairwise cosine similarity of the
  per-residue edit vectors. Domain edits are spatially coherent and survive; short linear motifs are
  not and are mostly averaged away. Full derivation: Topic 02 §5.</dd>

  <dt>tier (BISECT)</dt>
  <dd>A structural-evidence classification for a BISECT case (Tier 1 functional switch, Tier 2
  functional loss / Complex I collapse / partial change, Tier 3 weaker-evidence variants). Not a
  measure of biological importance by itself — see Topic 06 §1–2.</dd>

  <dt>DTU</dt>
  <dd>Differential Transcript Usage — a shift in the relative usage share of a gene's isoforms
  between two conditions (e.g. CT vs AD), tested with DRIMSeq. A significant DTU p-value means the
  isoform <i>ratio</i> changed, independent of whether total gene expression changed.</dd>

  <dt>B0 → B5 cascade</dt>
  <dd>The explicit chain used to diagnose where isoform-specific signal is lost:
  B0 sequence → B1 ESM-2 encoding → B2 pooling → B3 cross-gene anchor → B4 readout usage → B5 external
  label. Full definition and the gene-permutation-null discipline for B3: Topic 02 §4.</dd>

  <dt>ESM-2</dt>
  <dd>A protein language model (Meta AI) pretrained by masked-residue prediction on hundreds of
  millions of natural protein sequences. PRISM uses the 30-layer, 150M-parameter checkpoint (t30) and
  reads per-residue hidden states at every layer, not just the final one.</dd>

  <dt>APPRIS</dt>
  <dd>An external isoform-annotation resource (principal-isoform calls) used as one of the "does an
  isoform-resolution label even exist" checks referenced in the B5 discussion (Topic 02 §4, §7).</dd>

  <dt>IDR</dt>
  <dd>Intrinsically Disordered Region — a stretch of a protein without a fixed 3D fold. Disorder
  fraction is one of the sequence covariates shown in the Individual Analysis panel B and one of the
  8 interpretability axes.</dd>

  <dt>SLiM</dt>
  <dd>Short Linear Motif — a compact (typically 3–40 aa) functional sequence element (e.g. a
  degron, localization signal, or binding motif). SLiM edits are the case where mean-pooling loses
  the most signal (Topic 02 §5) — the geometric reason PRISM is more reliable at domain-level than
  motif-level resolution.</dd>

  <dt>PPI verdict</dt>
  <dd>Whether a BISECT case's isoform switch is predicted (via STRING) to gain or lose a
  protein–protein interaction partner. <code>SUPPORTED</code> means a new interaction partner is
  predicted for the AD-associated isoform.</dd>

  <dt>NMD</dt>
  <dd>Nonsense-Mediated Decay — an mRNA quality-control pathway that degrades transcripts with a
  premature stop codon positioned to trigger it. A BISECT case flagged NMD-sensitive should have
  protein-level translation confirmed experimentally (Ribo-seq / mass spec) rather than assumed from
  the transcript alone.</dd>

  <dt>gene-permutation null</dt>
  <dd>The null distribution used to test whether an apparent cross-gene direction (stage B3) is real:
  gene identity is randomly reshuffled and the statistic is recomputed. Stricter than a sign-shuffle
  null because it specifically tests for — and can detect — a spurious per-gene batch effect, which a
  sign-shuffle null cannot. See Topic 02 §4 and §8.</dd>

  <dt>macro-AUPRC</dt>
  <dd>Area under the precision–recall curve, averaged per-class then across classes — the primary
  sparse-class metric used throughout this project (AUROC is reported secondary; see Topic 02 §8 and
  Topic 03). Always reported against a gene-level null / oracle baseline, never as a bare number.</dd>
</dl>
"""

SECTIONS = [
    {'id': 'data', 'k': 'TOPIC 01', 'icon': 'download', 'title': 'Data Generation',
     'desc': 'Environment and package installation, the raw data this pipeline expects, and the full '
             'chain from long-read reads through preprocessing, PRISM training, BISECT, and populating '
             'this app with your own data — everything needed to reproduce this repository end to end.',
     'body': DATA_BODY},
    {'id': 'principles', 'k': 'TOPIC 02', 'icon': 'sigma', 'title': 'Core Principles & Mathematical Derivation',
     'desc': 'Why PRISM is built the way it is, worked out from first principles: the two failure modes of '
             'gene-level annotation transfer, the forward pass and loss, the B0→B5 representation cascade, '
             'the coherence-filter derivation for mean-pooling, the 8-axis interpretability map, and the '
             'encoding-vs-usage measurement-validity pitfall that governs how any of this is allowed to be read.',
     'body': r"""
<div class="tut-quicknav">
  <span class="tut-quicknav-label">On this page</span>
  <a href="#principles-problem">1. Two failure modes of gene-level transfer</a>
  <a href="#principles-forward">2. Forward pass: sequence → score</a>
  <a href="#principles-loss">3. Loss design: focal loss for sparse switches</a>
  <a href="#principles-cascade">4. The representation cascade, B0→B5</a>
  <a href="#principles-pooling">5. Mean-pooling as a spatial-coherence filter</a>
  <a href="#principles-axes">6. The 8-axis interpretability map</a>
  <a href="#principles-usage">7. Encoding ≠ usage: a measurement-validity pitfall</a>
  <a href="#principles-validation">8. Statistical validation discipline</a>
</div>

<h3 id="principles-problem">1. Two failure modes of gene-level transfer</h3>
<p>The DIFFUSE baseline (Chen <i>et&nbsp;al.</i>, <i>Bioinformatics</i> 2019) predicts isoform function by
transferring a gene-level GO annotation onto every isoform of that gene. This is a reasonable prior — most
isoforms of a gene do share most of the gene's function — but it fails in exactly the cases that matter
for isoform-resolved biology, and it fails in two structurally different ways.</p>
<p><b>Data sparsity / mode collapse.</b> The isoform-level events a model actually needs to learn — a
splice event that <i>gains</i> or <i>loses</i> function relative to the gene's consensus annotation — are
rare positives sitting inside a much larger population of isoforms that trivially agree with their gene
label. Under plain cross-entropy, the gradient is dominated by the easy, already-confident majority; the
predictor collapses toward the marginal (gene-level) rate and stops discriminating within a gene at all.
This is diagnosable directly: <code>collapse_score = std(pred) / mean(pred)</code> falls toward 0 as a
model degenerates into a near-constant gene-level predictor (see Topic 03).</p>
<p><b>Gene-level reference dominance.</b> Independently of the loss, any architecture that lets the model
see gene identity — or pools information at the gene rather than the isoform level — can shortcut to
reproducing the gene-level label without ever reading isoform-intrinsic sequence. A model can look highly
accurate in exactly the regime where it has learned nothing isoform-specific (macro-AUPRC computed without
a gene-level null is the classic way this hides).</p>
<p>PRISM's two design choices map directly onto these two failure modes: <b>focal loss</b> (§3) targets the
sparsity/collapse problem at the objective level, and <b>isoform-intrinsic ESM-2 features</b> — the model
receives only the translated amino-acid sequence of one isoform's own ORF, never a gene identifier — remove
the shortcut at the representation level. Any predictive signal PRISM produces must, by construction,
originate in that isoform's own sequence.</p>

<h3 id="principles-forward">2. Forward pass: sequence → score</h3>
<p>Each isoform's ORF (extracted with TransDecoder, Topic 01) is translated to an amino-acid sequence and
passed through ESM-2 (t30, 150M parameters), a protein language model pretrained by masked-residue
prediction on hundreds of millions of natural sequences. ESM-2 exposes a per-residue hidden state at each
of 30 transformer layers, <span class="tut-math-inline">\(h_i \in \mathbb{R}^{640}\)</span> for residue
<span class="tut-math-inline">\(i\)</span>. Production PRISM (<code>v15d_bp_clean</code>) reads the final
layer (L30), mean-pooled over the sequence:</p>
<div class="tut-math">\[ \varphi = \frac{1}{L}\sum_{i=1}^{L} h_i^{(30)} \in \mathbb{R}^{640} \]</div>
<p>A small MLP head maps the pooled embedding to a probability per GO term:</p>
<pre><code>Dense(256, relu) → BatchNorm → Dropout(0.3)
→ Dense(128, relu) → Dropout(0.2)
→ Dense(64,  relu)
→ Dense(1,   sigmoid)          # one independent head per GO term</code></pre>
<p>One such head is trained per GO term (production: 18 BP-only terms, see Topic 01/03), Adam
(<span class="tut-math-inline">\(\text{lr}=10^{-3}\)</span>), batch size 512, early stopping on validation
loss (patience 10) with LR reduction on plateau, and a 5-seed ensemble to average out initialization
variance before the final AUPRC is reported.</p>

<h3 id="principles-loss">3. Loss design: focal loss for sparse switches</h3>
<p>Standard binary cross-entropy weights every example equally regardless of how confidently the model
already classifies it — which is exactly wrong when the informative examples (isoform-level functional
switches) are a small minority inside a sea of easy, already-correct negatives. PRISM trains with the
binary focal loss (<code>BinaryFocalCrossentropy(gamma=2.0)</code>):</p>
<div class="tut-math">\[ FL(p_t) = -\,\alpha_t\,(1-p_t)^{\gamma}\,\log(p_t) \]</div>
<p>where <span class="tut-math-inline">\(p_t\)</span> is the model's predicted probability of the true
class. The modulating factor <span class="tut-math-inline">\((1-p_t)^{\gamma}\)</span> is the entire idea:
for an easy example the model already gets right confidently
(<span class="tut-math-inline">\(p_t \to 1\)</span>), the factor vanishes and that example's gradient
contribution is suppressed by a power of <span class="tut-math-inline">\(\gamma\)</span>; for a hard or
still-misclassified example (<span class="tut-math-inline">\(p_t\)</span> small) the factor is
<span class="tut-math-inline">\(\approx 1\)</span> and the gradient is essentially the full
cross-entropy gradient. At <span class="tut-math-inline">\(\gamma=0\)</span> this degenerates exactly to
CE, which is why <span class="tut-math-inline">\(\gamma < 1\)</span> is treated as a regression to CE
rather than a meaningful choice; production uses <span class="tut-math-inline">\(\gamma=2\)</span>.
Net effect: gradient mass is reallocated away from the majority "trivially-agrees-with-the-gene" isoforms
and toward the rare switch-relevant ones — directly counteracting the mode-collapse mechanism in §1.</p>

<h3 id="principles-cascade">4. The representation cascade, B0→B5</h3>
<p>It is useful to formalize the forward pass as an explicit chain of deterministic transformations, because
this licenses a specific kind of diagnostic question: <i>at which stage does isoform-specific signal
actually disappear?</i></p>
<div class="tut-math">\[
x_{\text{seq}} \;\xrightarrow{\;B1:\ \text{ESM-2}\;}\; H \in \mathbb{R}^{L\times 640}
\;\xrightarrow{\;B2:\ \text{pool}\;}\; \varphi \in \mathbb{R}^{640}
\;\xrightarrow{\;B3:\ \text{anchor}\;}\; \cdot
\;\xrightarrow{\;B4:\ \text{readout}\;}\; \hat y
\;\xrightarrow{\;B5:\ \text{label}\;}\; \{0,1\}
\]</div>
<p>B0 is the physical amino-acid sequence itself; B1 is what ESM-2 encodes per residue; B2 is what survives
pooling (§5); B3 is whether a shared, cross-gene direction in the pooled space carries the signal (tested
against a <b>gene-permutation</b> null, not a sign-shuffle null — shuffling gene identity is the correct
way to ask "is this a real cross-gene direction," because a sign-shuffle null leaves gene structure intact
and cannot detect a spurious per-gene batch effect); B4 is whether the trained readout actually relies on
that direction to make its decision (§7); B5 is whether an external annotation exists at isoform resolution
to check the prediction against at all (Topic 03 covers this). Because each arrow is a deterministic
function of its input, information about the original sequence can only be lost or preserved at each step,
never recovered downstream of where it was destroyed — so if a signal is not recoverable at stage
<span class="tut-math-inline">\(B_k\)</span>, no amount of readout capacity at
<span class="tut-math-inline">\(B_{k+1}\)</span> can bring it back. That is what makes "diagnose the stage
of failure" a well-posed question rather than a post-hoc narrative.</p>

<h3 id="principles-pooling">5. Mean-pooling as a spatial-coherence filter</h3>
<p>Stage B2 turns out to be the single most consequential step for short-motif biology, and it is worth
deriving why. Write the per-residue effect of an isoform edit (an insertion, deletion, or substitution
relative to a reference isoform) as
<span class="tut-math-inline">\(\delta h_i = h_i^{\text{edit}} - h_i^{\text{ref}}\)</span> for every
residue <span class="tut-math-inline">\(i\)</span> in the aligned sequence. The pooled difference is the
residue-average of these per-residue vectors:</p>
<div class="tut-math">\[ \delta\varphi = \frac{1}{L}\sum_{i=1}^{L}\delta h_i \]</div>
<p>By the triangle/Cauchy–Schwarz inequality, <span class="tut-math-inline">\(\lVert\delta\varphi\rVert^2
\le \frac{1}{L}\sum_i \lVert\delta h_i\rVert^2\)</span>, with equality only when every
<span class="tut-math-inline">\(\delta h_i\)</span> points in the same direction. Defining
<span class="tut-math-inline">\(\bar c\)</span> as the mean pairwise cosine similarity among the
<span class="tut-math-inline">\(\delta h_i\)</span> (their <i>directional coherence</i>), the fraction of
per-residue perturbation energy that survives pooling follows:</p>
<div class="tut-math">\[
\text{frac\_kept} \;=\; \frac{\lVert\delta\varphi\rVert^2}{\overline{\lVert\delta h_i\rVert^2}}
\;\approx\; \frac{1}{L} + \left(1-\frac{1}{L}\right)\bar c
\]</div>
<p>Two limits make the mechanism concrete. If the per-residue response is spatially <i>incoherent</i>
(<span class="tut-math-inline">\(\bar c \to 0\)</span> — the edit's effect is scattered across many
directions, as when attention spreads a short inserted motif's signature to its neighbours without a
consistent shared direction), survival collapses to <span class="tut-math-inline">\(1/L\)</span>: the
longer the sequence, the more completely the edit is averaged away. If the response is <i>coherent</i>
(<span class="tut-math-inline">\(\bar c \to 1\)</span> — a large domain insertion or deletion shifts the
whole residual stream in a consistent direction), survival approaches 1 essentially independent of
<span class="tut-math-inline">\(L\)</span>. This is the same mathematics as coherent averaging /
signal-integration gain: only the phase-aligned component of a sum of vectors survives an average, the rest
cancels.</p>
<p>This was checked directly, not just assumed: per-residue re-extraction on domain-changing vs.
short-linear-motif (SLiM, 3–40 aa) edit pairs shows domain edits are spatially coherent
(<span class="tut-math-inline">\(\text{frac\_kept}\approx 0.16\text{–}0.31\)</span>) while SLiM edits are
incoherent (<span class="tut-math-inline">\(\approx 0.05\text{–}0.08\)</span>) — roughly a 3–4× survival
gap that holds independent of raw edit size and replicates across brain and muscle. It also overturned a
simpler first guess: the edited signal is <i>not</i> locally confined before pooling (attention already
spreads a motif's effect to its 1–2 nearest neighbours, carrying 34–50% of the edit-core magnitude), so a
naive "O(k/L), edit width over sequence length" low-pass picture is the wrong mechanism — what actually
gates survival is the <i>directional coherence</i> of that spread, not its residue footprint. This is the
geometric reason PRISM's domain-level predictions are systematically more reliable than its short-motif-level
ones: it is not a training or capacity deficit, it is what a linear pooling operator can pass through at all.</p>

<h3 id="principles-axes">6. The 8-axis interpretability map</h3>
<p>To make the 640-dimensional ESM-2 representation legible, PRISM projects it onto 8 interpretable axes
with a per-layer-standardized, jointly-fit PCA. First, every one of the 640 dimensions is z-scored at each
layer <span class="tut-math-inline">\(L\)</span> over the isoform population:</p>
<div class="tut-math">\[ z_{i,L} = \frac{h_{i,L} - \mu_L}{\sigma_L} \]</div>
<p>Every isoform's 30-layer trajectory is then flattened into one
<span class="tut-math-inline">\((N\!\cdot\!30,\ 640)\)</span> matrix and a single PCA with
<span class="tut-math-inline">\(K=8\)</span> components is fit <i>jointly</i> across all layers — so one
shared basis <span class="tut-math-inline">\(W \in \mathbb{R}^{8\times 640}\)</span> is reused at every
depth, which is what makes it meaningful to say "axis 3 at layer 9" and "axis 3 at layer 30" are the same
direction. Any embedding then has interpretable coordinates:</p>
<div class="tut-math">\[ Z_{i,L,:} = z_{i,L} \, W^{\top} \in \mathbb{R}^{8} \]</div>
<p><span class="tut-math-inline">\(W\)</span> is fit once on a held-out training population and only ever
<i>applied</i> — never refit — to the evaluation population, so an axis coordinate is always an
out-of-sample projection onto a fixed direction, not a population-specific re-derivation.</p>
<p>Not every high-variance axis is biologically usable, and this is checked before any axis is given a
name. Most sequence variance in a gene family separates <i>gene identity</i> (paralogs), not <i>isoform
choice</i> within one gene — an axis that only tracks the former cannot support an isoform-function claim
no matter how clean it looks. Each axis is screened with a one-way ANOVA variance decomposition restricted
to multi-isoform genes:</p>
<div class="tut-math">\[
SS_{\text{total}} = SS_{\text{between-gene}} + SS_{\text{within-gene}},\qquad
\text{within\_frac} = \frac{SS_{\text{within-gene}}}{SS_{\text{total}}}
\]</div>
<p>Only axes with non-trivial <span class="tut-math-inline">\(\text{within\_frac}\)</span> are eligible for
an isoform-level biological interpretation at all.</p>

<h3 id="principles-usage">7. Encoding ≠ usage: a measurement-validity pitfall</h3>
<p>A recurring trap in this whole framework: a direction can be <i>encoded</i> in the representation
(present, linearly decodable) without being <i>used</i> by the trained readout (actually moving its
predictions). Two instruments were tried for stage B4 in the cascade.</p>
<p><b>Rejected: coherence-retrain occlusion.</b> Project out a direction, retrain a probe from scratch on
what remains, and measure the metric drop. The confound: retraining re-optimizes over whatever variance is
left, so the measured drop tracks how much raw variance the removed direction carried, not whether the
original trained model relied on it — a high-variance but functionally irrelevant direction can "look used"
purely because removing it perturbs the effective scale of everything else.</p>
<p><b>Adopted: ridge test-time reliance.</b> Fit the readout once. At test time only, project out axis
<span class="tut-math-inline">\(k\)</span> from the input
(<span class="tut-math-inline">\(z \mapsto z - (z\!\cdot\! d)\,d\)</span>) <i>without retraining</i>, and
measure:</p>
<div class="tut-math">\[ \text{reliance}_k = \text{metric}_{\text{base}} - \text{metric}_{\text{occluded}} \]</div>
<p>Because the readout's weights are frozen, this isolates how much the specific decision boundary that was
actually learned depends on axis <span class="tut-math-inline">\(k\)</span>, decoupled from how much
variance that axis happens to carry.</p>
<p>Even this instrument needs its own confound check, because a large reliance number could still just mean
"axis <span class="tut-math-inline">\(k\)</span> is high-variance." The valid null is a
<b>flagship-excluded, variance-matched random-direction null</b>: random combinations drawn from the span
of the <i>other</i> axes only (excluding the one under test), rescaled to match its variance. Two simpler
nulls were tried first and rejected — a low-variance random unit vector doesn't overlap the real axis's
variance regime at all, and a random combination of <i>all</i> axes contaminates the null with a diluted
copy of the axis being tested, inflating its apparent significance. A worked example: on brain data,
testing whether axis 3 is used for a domain-level label gives reliance 0.0345 against a variance-matched
null with mean 0.003 (z ≈ +16 — not close) — genuinely used. Axis 0, despite being one of the
<i>highest</i>-variance axes in the whole basis, shows <i>negative</i> reliance for that same label: a
variance-driven artifact cannot produce a negative number, which is itself the evidence that the instrument
is measuring target-specific usage rather than being fooled by scale.</p>

<h3 id="principles-validation">8. Statistical validation discipline</h3>
<p>Every improvement number anywhere in this app or the accompanying manuscript is required to clear the
same bar before being reported as real: a <b>bootstrap confidence interval</b> (n=1000 resamples) against
an explicit <b>null or oracle baseline</b>, and — specifically for any claim of a shared, cross-gene
direction (stage B3 above) — a <b>gene-permutation null</b>, which is a different and stricter test than
the more familiar label/sign-shuffle null: it asks whether the apparent direction survives when gene
identity itself is scrambled, which is exactly the failure mode §1 warns about. Sparse-class evaluation
uses AUPRC as the primary metric (AUROC is reported secondary, since it is insensitive to class imbalance
in exactly the regime this project cares about). See <a href="#stats">Topic 03 — Key Statistical
Analyses</a> for the concrete code.</p>

<div class="tut-callout">
<b>Why this much ceremony.</b> Every one of the three "adopted" instruments above (focal loss, the
gene-permutation null, ridge test-time reliance) replaced a simpler first attempt that looked reasonable
and turned out to be measuring something else — easy-example-dominated CE looked fine until checked against
a collapse-score diagnostic; occlusion-retrain looked like a usage test until checked against a
variance-matched null; a naive edit-magnitude/length pooling picture looked right until checked against
directly re-extracted per-residue vectors. The pattern, not any one instrument, is the actual methodological
commitment of this project: an improvement or an interpretation is not accepted until it has survived an
explicit attempt to explain it away as an artifact of the measurement tool itself.
</div>
"""},
    {'id': 'stats', 'k': 'TOPIC 03', 'icon': 'trending-up', 'title': 'Key Statistical Analyses',
     'desc': 'DTU via DRIMSeq/permutation; improvement claims require bootstrap CI (n=1000) against '
             'null/oracle. For sparse classes, AUPRC is primary.',
     'code': '# DTU (per-cell-type isoform usage change)\n'
             'DRIMSeq::dmTest(d, coef="condition")\n'
             '# bootstrap CI (validate an improvement number)\n'
             'auprc_ci = bootstrap(y_true, y_score, n=1000, metric=average_precision)\n'
             '# gene-permutation null (validate cross-gene claims)'},
    {'id': 'figures', 'k': 'TOPIC 04', 'icon': 'image', 'title': 'Key Figure-Generation Code',
     'desc': 'The representation map (8 axes), layer trajectories, and covariates come from '
             '30-layer ESM-2 mean-pooled embeddings. The 8-axis projection is reproduced with saved '
             'W_axes / layer_stats (exact match verified).',
     'code': '# 30-layer mean-pooled embeddings (already generated)\n'
             'python hMuscle/preprocessing/compute_esm2_all_layers.py\n'
             '# 8-axis projection:  Z[:,L,:] = ((emb_L - mu_L)/sd_L) @ W_axes.T\n'
             '# build the app index (covariate · axis · trajectory · peak layer)\n'
             'python prism_app_flask/precompute/build_isoform_index.py'},
    {'id': 'architecture', 'k': 'TOPIC 05', 'icon': 'layers', 'title': 'Model Architecture & Training',
     'desc': 'What PRISM is under the hood: backbone choice, loss design (focal loss for class sparsity), '
             'and the ablation discipline used to justify each component.',
     'body': ARCHITECTURE_BODY},
    {'id': 'bisect-guide', 'k': 'TOPIC 06', 'icon': 'flask-conical', 'title': 'Reading a BISECT Report',
     'desc': 'How to interpret a BISECT case: tier levels, domain gain/loss chips, the switch-pair layout, '
             'and what confidence means in this context.',
     'body': BISECT_GUIDE_BODY},
    {'id': 'glossary', 'k': 'TOPIC 07', 'icon': 'book', 'title': 'Glossary',
     'desc': 'Short definitions for terms used throughout the app and this tutorial — editcore, axis, '
             'trajectory, coherence-filter, tier, DTU, and others.',
     'body': GLOSSARY_BODY},
]


@bp.route('/tutorial')
def home():
    return render_template('tutorial.html', sections=SECTIONS, github=config.GITHUB_URL)
