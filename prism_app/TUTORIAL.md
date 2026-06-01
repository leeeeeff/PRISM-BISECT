# PRISM+BISECT Interactive Analysis Tool — User Guide

**PRISM** (Protein-isoform Resolution via Intrinsic Sequence Modeling) predicts GO-term function for isoforms from ESM-2 embeddings. **BISECT** (Biological Isoform-Switch Evidence Characterization Tool) classifies each isoform by its DTU status and functional novelty.

---

## Quick Start

```bash
# Local: launch the app
streamlit run prism_app/app/main.py

# Or, via CLI without the web UI
prism-app analyze --scores scores.npy --ids ids.npy --tissue muscle --output results/
```

---

## Two Modes

### Demo Mode (no upload required)
Loads the pre-computed muscle and brain results from the paper (Lee et al. 2026).
Use this to reproduce paper figures or explore the analysis interactively.

Select **"Demo (Muscle — 36,748 isoforms)"** or **"Demo (Brain — 63,994 isoforms)"** from the sidebar.

### Upload Mode (your own data)
Upload your own score matrix and isoform IDs from any long-read sequencing experiment.

---

## Input File Format

All files are standard NumPy `.npy` or plain text.

| File | Required | Format | Description |
|------|----------|--------|-------------|
| Score matrix | **Yes** | `.npy` — shape `(N_isoforms, N_GO)` float32 | PRISM output scores, range 0–1 |
| Isoform IDs | **Yes** | `.npy` or `.txt` — 1D array of strings | One transcript ID per row, same order as score matrix |
| Gene IDs | Recommended | `.npy` or `.txt` — 1D array of strings | One gene symbol or ENSG ID per isoform |
| Isoform types | Optional | `.npy` or `.txt` — values: `known`, `nic`, `nnic` | Enables novel isoform section |
| DTU results | Optional | `.tsv` or `.csv` | Enables condition analysis (see DTU format below) |

### Score matrix columns
The column order must match the GO term preset you select in the sidebar:
- **Muscle (18 GO terms)**: the 18 BP terms from PRISM training
- **Brain (18 GO terms)**: same 18 terms, zero-shot transfer
- **Brain extended (73 GO terms)**: 18 + 55 brain-specific terms

### DTU file format
Any of the following column names are auto-detected:

```
isoform_id    gene_id    condition    delta_IF    pvalue
ENST00000...  NDUFS4     disease      0.32        0.0001
ENST00000...  NDUFS4     disease     -0.28        0.0003
```

Accepted column aliases:
- `isoform_id`: `transcriptID`, `transcript_id`, `isoform`, `featureID`
- `delta_IF`: `dIF`, `deltaPSI`, `logFC`, `log2FC`, `effect_size`
- `pvalue`: `padj`, `FDR`, `adj.p.value`, `q_value`, `pval`
- `condition`: `comparison`, `contrast`, `group`

Supported tools: **satuRn**, **DEXSeq**, **IsoformSwitchAnalyzeR**, **rMATS**.

---

## Pages

### Main page
Data source selection. Choose Demo or Upload. Sets up `st.session_state['cfg']` for all other pages.

### 1 · Overview
- **A1 Coverage Summary**: total isoforms, breakdown by type (Known/NIC/NNIC), isoforms with score > threshold
- **D1 4-Scenario Classification**: pie/bar of Scenario 1–4 distribution
- **A3 Novel Isoform Functions**: top GO terms predicted for NIC/NNIC isoforms
- **A2 Known Annotation Validation**: per-GO AUPRC vs. UniProt annotations; 95% bootstrap CI

### 2 · Functional Map
- **B1 UMAP**: isoforms embedded in GO score space; color by isoform type, scenario, or top GO term
- **A4 Heatmap**: isoform type × GO term mean score grid
- **B4 Within-gene comparison**: scatter/radar for all isoforms of a selected gene

### 3 · Condition Analysis
Requires DTU file upload. Tabs:
- **GAIN/LOSS/NEUTRAL**: per-GO functional consequence of DTU switching
- **GO Enrichment**: hypergeometric test on significantly switched isoforms (BH-FDR)
- **Sankey**: condition-cluster membership changes
- **Gene detail**: per-gene GAIN/LOSS bar chart
- **Full results**: downloadable tables

### 4 · Individual Analysis
- Search by gene or isoform ID
- Scenario card: scenario label, max GO score, DTU status
- Within-gene comparison chart
- Download case report (Markdown)

### 5 · Advanced
- **Cross-tissue**: compare muscle vs. brain GO score profiles
- **Expression filter**: joint score × CPM threshold filter (upload count matrix)
- **NMD screening**: flag isoforms at risk of nonsense-mediated decay

---

## 4-Scenario Classification

Each isoform is assigned one scenario based on DTU status and functional novelty:

| Scenario | DTU | Novel GO (score > thr) | Interpretation |
|----------|-----|----------------------|----------------|
| **S1** Functional Switch | Yes | Yes | Isoform switch with new function |
| **S2** Expression Switch | Yes | No | Isoform switch, function unchanged |
| **S3** Constitutive Novel Function | No | Yes | Novel function regardless of condition |
| **S4** Background | No | No | No evidence of functional novelty |

*"Novel GO"* means any GO term where the isoform lacks existing annotation and scores above threshold.
*DTU* requires: p-value < 0.05 (adjustable in sidebar) and |delta_IF| > 0.1.

---

## CLI Usage

### Batch analysis
```bash
# Full analysis: scenarios + coverage + validation
prism-app analyze \
    --scores scores.npy \
    --ids isoform_ids.npy \
    --genes gene_ids.npy \
    --types isoform_types.npy \
    --dtu dtu_results.tsv \
    --tissue muscle \
    --threshold 0.5 \
    --output results/

# Output:
#   results/scenarios.tsv        — per-isoform scenario labels
#   results/coverage.json        — coverage statistics
#   results/novel_summary.tsv    — novel isoform GO table
#   results/validation.json      — AUPRC metrics
```

### Per-isoform case report
```bash
# Report for all isoforms of a gene
prism-app-report \
    --scores scores.npy \
    --ids isoform_ids.npy \
    --gene NDUFS4 \
    --tissue brain \
    --format markdown \
    --output ndufs4_report.md

# Report for a specific isoform ID
prism-app-report \
    --scores scores.npy \
    --ids isoform_ids.npy \
    --isoform NDUFS4-201 \
    --tissue brain
```

---

## Reproducing Paper Results (Demo Mode)

| Figure / Stat | Page | What to do |
|---------------|------|------------|
| Macro AUPRC 0.7022 (muscle) | Overview → A2 | Load Demo Muscle; check AUPRC metric |
| 541 Scenario 3 novel isoforms (brain) | Overview → D1 | Load Demo Brain Extended; check S3 count |
| NDUFS4 Scenario 3 case | Individual | Search gene "NDUFS4"; check scenario card |
| KIF21B Scenario 1 (with DTU) | Individual | Upload brain DTU + Search "KIF21B" |
| Complex I cluster in UMAP | Functional Map → UMAP | Color by top GO term; find Mito_org cluster |
| GAIN/LOSS in disease DTU | Condition | Upload brain DTU; see GAIN/LOSS bar chart |

---

## Streamlit Community Cloud Deployment

1. Fork the repository and push to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Set **Main file path**: `prism_app/app/main.py`
4. Set **Requirements file**: `prism_app/requirements.txt`
5. Optional: add secrets in the Streamlit Cloud dashboard (see `.streamlit/secrets.toml.example`).
6. Deploy — demo data is bundled in `prism_app/data/demo/`.

Demo mode works without any configuration. Upload mode works for any user who uploads their own NPY files.

---

## Frequently Asked Questions

**Q: My isoform IDs don't match any annotations — AUPRC shows no data.**  
A: Gene IDs must be either gene symbols (NDUFS4) or Ensembl IDs (ENSG00000143183). Version suffixes (.13) are stripped automatically. If uploading ENSG IDs, ensure they match the bundled mapping (muscle+brain demo data). For custom datasets, the annotation lookup may not find matches — this is expected and validation is skipped gracefully.

**Q: The DTU analysis shows mostly UNKNOWN consequences.**  
A: Check that your DTU file has multiple isoforms per gene *within the same condition*. GAIN/LOSS is computed by pairing up- and down-regulated isoforms of the same gene in the same contrast. If each isoform appears only once per condition, UNKNOWN is correct.

**Q: UMAP takes a long time.**  
A: UMAP is computed on the first page visit and cached for the session. For 60,000+ isoforms, expect ~30 seconds. The demo datasets use pre-computed coordinates for instant display.

**Q: Can I use this with IsoformSwitchAnalyzeR output?**  
A: Yes. Export the `isoformFeatures` table as TSV. The column names `isoformUpregulated`, `dIF`, `isoform_id`, `gene_id`, and `padj` are auto-mapped.

**Q: Does this run PRISM (the neural network) inference?**  
A: No — the web app works with pre-computed score matrices. Run PRISM inference locally:
```bash
conda activate isoform_env
python hMuscle/model/v15d_bp_clean.py --predict --embeddings your_esm2.npy
```
Then upload the resulting `scores.npy` to the web app.
