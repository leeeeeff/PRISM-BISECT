# Section 06A: Brain Tissue Data Assembly

## Data Source: ● Samsung Alzheimer's Disease Long-read scRNA-seq (IsoQuant)

**Raw data location**: `/home/dhkim1674/Project_AD_with_refTSS_novel/`

## Pipeline

```
Raw long-read scRNA-seq FASTQ
    ↓ Alignment (minimap2 → GRCh38 ◇)
    ↓ IsoQuant transcript detection
    ↓ SQANTI3 structural classification ◇
    ↓ ORF prediction (SQANTI3 output)
    ↓ ESM-2 t30_150M embedding (●→◇)
    ↓ GO label assembly (UniProt/SwissProt ◇)
    ↓
Brain IsoQuant Test Set
```

## Output Data Files

| File | Shape | Description |
|------|-------|-------------|
| `brain_full_esm2_t30_150M.npy` | (63994, 640) | ESM-2 embeddings |
| `brain_full_ids.npy` | (63994,) | Transcript IDs |
| `brain_full_gene_names.npy` | (63994,) | Associated gene names |
| `brain_full_mask.npy` | (63994,) | Coding=1, non-coding=0 |
| `brain_full_labels.npy` | (63994, 18) | GO BP binary labels |

**Location**: `/home/welcome1/sw1686/DIFFUSE/hMuscle/data/brain_isoquant_esm2/full/`

## Key Statistics

### Isoform Composition
| Category | Count | % |
|----------|-------|---|
| Total isoforms | 63,994 | 100% |
| Novel (NNIC/NIC, "transcript*") | 7,899 | 12.3% |
| Known (FSM/ISM) | 56,095 | 87.7% |
| Coding (ORF available) | 53,826 | 84.1% |
| Non-coding | 10,168 | 15.9% |

### SQANTI3 Structural Categories
| Category | Abbreviation | Description |
|----------|-------------|-------------|
| full-splice_match | FSM | Exact match to reference transcript |
| incomplete-splice_match | ISM | Subset of reference exons |
| novel_in_catalog | NIC | New exon combination, known splice sites |
| novel_not_in_catalog | NNIC | At least one novel splice site |
| genic | — | Genic region, no assigned transcript |

### GO Label Distribution (18 BP terms)
| GO Term | Name | n_positive |
|---------|------|-----------|
| GO:0007204 | Ca2+ signaling | ~300 |
| GO:0045214 | Sarcomere organization | ~108 |
| GO:0006941 | Muscle contraction | ~200 |
| GO:0006914 | Autophagy | ~500 |
| GO:0043161 | Proteasome-UPS | ~600 |
| GO:0007519 | Skeletal muscle dev | ~150 |
| GO:0042692 | Muscle cell diff | ~300 |
| GO:0055074 | Ca2+ homeostasis | ~400 |
| GO:0007005 | Mitochondrion org | ~700 |
| GO:0007517 | Muscle organ dev | ~350 |
| GO:0032006 | TOR signaling | ~250 |
| GO:0030048 | Actin-based movement | ~200 |
| GO:0006096 | Glycolysis | ~100 |
| GO:0007268 | Synaptic transmission | ~800 |
| GO:0007018 | MT-based movement | ~500 |
| GO:0031175 | Neuron proj development | ~1200 |
| GO:0030182 | Neuron differentiation | ~2740 |
| GO:0000226 | MT cytoskeleton org | ~900 |

## DTU Analysis
- 8 cell types × AD/CT conditions
- Dirichlet-multinomial test for differential transcript usage
- Location: `/home/dhkim1674/Project_AD_with_refTSS_novel/06_DIU/`

### Sample Sizes (approximate)
| Cell type | AD n | CT n |
|-----------|------|------|
| Excitatory neuron | ~21 | ~37 |
| Inhibitory neuron | ~15 | ~25 |
| Astrocyte | ~12 | ~20 |
| Oligodendrocyte | ~10 | ~18 |
| OPC | ~8 | ~15 |
| Microglia | ~6 | ~12 |
| Vascular cell | ~5 | ~10 |
| Lymphocyte | ~4 | ~8 |

## Comparison with Muscle Data

| | Muscle (hMuscle ●) | Brain (Samsung AD ●) |
|--|---------------------|----------------------|
| Total isoforms | ~10,000 | 63,994 |
| Novel isoforms | ~0 (Bambu) | 7,899 (12.3%) |
| Tissue | Skeletal muscle | Prefrontal cortex |
| Disease context | None | Alzheimer's vs Control |
| Cell resolution | Bulk | Single-cell |
| Model training | ✓ (train set) | ✗ (test set only) |
