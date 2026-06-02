"""GO term utilities: presets, name mapping, annotation loading."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional

# ── PRISM 18 muscle training GO terms ──────────────────────────────────────
MUSCLE_18 = {
    'GO:0007204': 'Ca2+-mediated signaling',
    'GO:0045214': 'Sarcomere organization',
    'GO:0006941': 'Striated muscle contraction',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-mediated UPS',
    'GO:0007519': 'Skeletal muscle tissue development',
    'GO:0042692': 'Muscle cell differentiation',
    'GO:0055074': 'Calcium ion homeostasis',
    'GO:0007005': 'Mitochondrial organization',
    'GO:0007517': 'Muscle organ development',
    'GO:0032006': 'Regulation of TOR signaling',
    'GO:0030048': 'Actin filament-based movement',
    'GO:0006096': 'Glycolytic process',
    'GO:0007268': 'Chemical synaptic transmission',
    'GO:0007018': 'Microtubule-based movement',
    'GO:0031175': 'Neuron projection development',
    'GO:0030182': 'Neuron differentiation',
    'GO:0000226': 'Microtubule cytoskeleton organization',
}

# ── Brain extended GO terms (73 terms, from extension experiment) ───────────
BRAIN_EXTENDED = {
    'GO:0007186': 'G protein-coupled receptor signaling pathway',
    'GO:0030182': 'Neuron differentiation',
    'GO:0007167': 'Enzyme-linked receptor protein signaling pathway',
    'GO:0048666': 'Neuron development',
    'GO:0050767': 'Regulation of neurogenesis',
    'GO:0007169': 'Cell surface receptor protein tyrosine kinase signaling',
    'GO:0010469': 'Regulation of signaling receptor activity',
    'GO:0031175': 'Neuron projection development',
    'GO:0002768': 'Immune response-regulating cell surface receptor signaling',
    'GO:0007420': 'Brain development',
    'GO:0002429': 'Immune response-activating cell surface receptor signaling',
    'GO:0045664': 'Regulation of neuron differentiation',
    'GO:0007268': 'Chemical synaptic transmission',
    'GO:0007166': 'Cell surface receptor signaling pathway',
    'GO:0055074': 'Calcium ion homeostasis',
    'GO:0006874': 'Intracellular calcium ion homeostasis',
    'GO:0048812': 'Neuron projection morphogenesis',
    'GO:0042391': 'Regulation of membrane potential',
    'GO:0010975': 'Regulation of neuron projection development',
    'GO:0050804': 'Modulation of chemical synaptic transmission',
    'GO:0061564': 'Axon development',
    'GO:0007409': 'Axonogenesis',
    'GO:0050769': 'Positive regulation of neurogenesis',
    'GO:0051480': 'Regulation of cytosolic calcium ion concentration',
    'GO:0007204': 'Positive regulation of cytosolic calcium ion concentration',
    'GO:0006898': 'Receptor-mediated endocytosis',
    'GO:0006816': 'Calcium ion transport',
    'GO:0007411': 'Axon guidance',
    'GO:0050851': 'Antigen receptor-mediated signaling pathway',
    'GO:0045666': 'Positive regulation of neuron differentiation',
    'GO:0007187': 'GPCR signaling, coupled to cyclic nucleotide 2nd messenger',
    'GO:0010976': 'Positive regulation of neuron projection development',
    'GO:0007189': 'Adenylate cyclase-activating GPCR signaling',
    'GO:0007188': 'Adenylate cyclase-modulating GPCR signaling',
    'GO:0070588': 'Calcium ion transmembrane transport',
    'GO:0008037': 'Cell recognition',
    'GO:0038093': 'Fc receptor signaling pathway',
    'GO:0019722': 'Calcium-mediated signaling',
    'GO:0051924': 'Regulation of calcium ion transport',
    'GO:0050808': 'Synapse organization',
    'GO:0071805': 'Potassium ion transmembrane transport',
    'GO:0097485': 'Neuron projection guidance',
    'GO:0006813': 'Potassium ion transport',
    'GO:0007178': 'Cell surface receptor protein serine/threonine kinase signaling',
    'GO:0030522': 'Intracellular receptor signaling pathway',
    'GO:0050768': 'Negative regulation of neurogenesis',
    'GO:0007200': 'Phospholipase C-activating GPCR signaling',
    'GO:0050890': 'Cognition',
    'GO:0038094': 'Fc-gamma receptor signaling pathway',
    'GO:0006836': 'Neurotransmitter transport',
    'GO:0038096': 'Fc-gamma receptor signaling in phagocytosis',
    'GO:0030512': 'Negative regulation of TGF-beta receptor signaling',
    'GO:0002431': 'Fc receptor mediated stimulatory signaling',
    'GO:0050852': 'T cell receptor signaling pathway',
    'GO:0045665': 'Negative regulation of neuron differentiation',
    'GO:0008277': 'Regulation of GPCR signaling',
    'GO:0051592': 'Response to calcium ion',
    'GO:0050853': 'B cell receptor signaling pathway',
    'GO:0048167': 'Regulation of synaptic plasticity',
    'GO:0002221': 'Pattern recognition receptor signaling',
    'GO:0007611': 'Learning or memory',
    'GO:0038095': 'Fc-epsilon receptor signaling pathway',
    'GO:0050807': 'Regulation of synapse organization',
    'GO:0043524': 'Negative regulation of neuron apoptotic process',
    'GO:0010977': 'Negative regulation of neuron projection development',
    'GO:0046425': 'Regulation of receptor signaling via JAK-STAT',
    'GO:0050770': 'Regulation of axonogenesis',
    'GO:0007179': 'TGF-beta receptor signaling pathway',
    'GO:0007218': 'Neuropeptide signaling pathway',
    'GO:0007193': 'Adenylate cyclase-inhibiting GPCR signaling',
    'GO:1903169': 'Regulation of calcium ion transmembrane transport',
    'GO:0043523': 'Regulation of neuron apoptotic process',
    'GO:0001508': 'Action potential',
}

# ── Brain 672-term preset (loaded dynamically from JSON) ────────────────────
def _load_brain672() -> Dict[str, str]:
    import json
    candidates = [
        Path(__file__).parents[2] / 'hMuscle/data/brain672_go_terms.json',
        Path(__file__).parents[3] / 'hMuscle/data/brain672_go_terms.json',
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            return d.get('go_names', {})
    return {}

BRAIN_672: Dict[str, str] = _load_brain672()

# ── Tissue presets ──────────────────────────────────────────────────────────
# brain = 18-term PRISM model applied zero-shot to brain data
# brain_extended = 73-term extended panel for novel brain isoforms
# brain_672 = 672-term full BP panel (brain>=100, train>=50)
TISSUE_PRESETS: Dict[str, Dict[str, str]] = {
    'muscle':         MUSCLE_18,
    'brain':          MUSCLE_18,
    'muscle_only':    MUSCLE_18,
    'brain_extended': BRAIN_EXTENDED,
    'brain_672':      BRAIN_672,
}

# ── Full name mapping (union of all known terms) ────────────────────────────
GO_FULL_NAMES: Dict[str, str] = {**MUSCLE_18, **BRAIN_EXTENDED, **BRAIN_672}


def load_go_names(extra_csv: Optional[str] = None) -> Dict[str, str]:
    """Return GO ID → full name dict, optionally extended from a CSV."""
    names = dict(GO_FULL_NAMES)
    if extra_csv and os.path.exists(extra_csv):
        import csv
        with open(extra_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                go_id = row.get('go') or row.get('GO') or ''
                name  = row.get('name') or row.get('Name') or ''
                if go_id and name:
                    names[go_id] = name
    return names


def load_gene_annotations(annot_path: str) -> Dict[str, List[str]]:
    """Load gene-level GO annotations from human_annotations_unified_bp.txt.

    Format: gene_symbol<TAB>GO:xxx<TAB>GO:yyy ...
    Returns: {gene_symbol: [GO:xxx, GO:yyy, ...]}
    """
    gene_go: Dict[str, List[str]] = {}
    with open(annot_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            gene = parts[0]
            terms = [p for p in parts[1:] if p.startswith('GO:')]
            if terms:
                gene_go[gene] = terms
    return gene_go


def get_preset_terms(tissue: str) -> Dict[str, str]:
    """Return GO term dict for a given tissue preset."""
    if tissue not in TISSUE_PRESETS:
        available = list(TISSUE_PRESETS.keys())
        raise ValueError(f"Unknown tissue '{tissue}'. Available: {available}")
    return TISSUE_PRESETS[tissue]
