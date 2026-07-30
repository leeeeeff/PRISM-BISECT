# Muscle vs Brain Fisher L1~L30 — 279 GO Comparison

- Common GOs (both eval): **277**
- Muscle Fisher method: Fisher discriminant per layer (test-set only)
- Brain Fisher method: Fisher discriminant per layer (brain isoforms, muscle-defined 279-GO seed)

## Peak-layer shift (brain − muscle)

- Mean shift: **-0.52** layers (SD 12.55)
- Shifted deeper: 96/277 (34.7%)
- Shifted shallower: 105/277 (37.9%)
- Same layer: 76/277 (27.4%)

## Bucket concordance

- Same bucket (Early/Mid/Late): **145/277 (52.3%)**
- Bucket transitions:
  - Early → Mid: 19
  - Early → Late: 30
  - Mid → Early: 17
  - Mid → Late: 12
  - Late → Early: 25
  - Late → Mid: 29

## Bucket distribution

| Bucket | Muscle BP/MF/CC | Brain BP/MF/CC |
|---|---|---|
| Early (L1-10) | 42/32/40 | 43/25/39 |
| Mid (L11-20) | 30/18/25 | 36/19/37 |
| Late (L21-30) | 31/31/28 | 25/38/17 |

## Category-stratified mean shift

- BP: mean shift = -0.81 (n=103)
- MF: mean shift = +1.38 (n=81)
- CC: mean shift = -1.85 (n=93)

## Fisher magnitude (mean peak signal)
- Muscle: 51.209
- Brain:  43.094

## Top 10 largest deeper shifts (brain > muscle)

| GO | Name | Cat | m_peak | b_peak | shift |
|---|---|---|---|---|---|
| GO:0005730 | nucleolus | CC | L1 | L30 | +29 |
| GO:0007283 | spermatogenesis | BP | L1 | L30 | +29 |
| GO:0030317 | flagellated sperm motility | BP | L1 | L30 | +29 |
| GO:0031514 | motile cilium | CC | L1 | L30 | +29 |
| GO:0036126 | sperm flagellum | CC | L1 | L30 | +29 |
| GO:0060271 | cilium assembly | BP | L1 | L30 | +29 |
| GO:1902600 | proton transmembrane transport | BP | L1 | L30 | +29 |
| GO:0003779 | actin binding | MF | L2 | L30 | +28 |
| GO:0008017 | microtubule binding | MF | L2 | L30 | +28 |
| GO:0005813 | centrosome | CC | L3 | L30 | +27 |

## Top 10 largest shallower shifts (brain < muscle)

| GO | Name | Cat | m_peak | b_peak | shift |
|---|---|---|---|---|---|
| GO:0031625 | ubiquitin protein ligase binding | MF | L30 | L1 | -29 |
| GO:0032991 | protein-containing complex | CC | L30 | L1 | -29 |
| GO:0005769 | early endosome | CC | L29 | L1 | -28 |
| GO:0005829 | cytosol | CC | L29 | L1 | -28 |
| GO:0048471 | perinuclear region of cytoplasm | CC | L29 | L1 | -28 |
| GO:0050821 | protein stabilization | BP | L30 | L2 | -28 |
| GO:0055037 | recycling endosome | CC | L29 | L1 | -28 |
| GO:0098794 | postsynapse | CC | L29 | L1 | -28 |
| GO:0005515 | protein binding | MF | L29 | L2 | -27 |
| GO:0005737 | cytoplasm | CC | L29 | L2 | -27 |
