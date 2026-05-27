"""
M3: Motif & Feature Analysis
MTS scoring, functional motif detection, structural feature extraction.
"""
import math
import re
from typing import Optional

# Eisenberg (1984) hydrophobicity scale
HYDROPHOBICITY = {
    'A': 0.25, 'R': -1.76, 'N': -0.64, 'D': -0.72, 'C': 0.04,
    'Q': -0.69, 'E': -0.62, 'G': 0.16, 'H': -0.40, 'I': 0.73,
    'L': 0.53, 'K': -1.10, 'M': 0.26, 'F': 0.61, 'P': -0.07,
    'S': -0.26, 'T': -0.18, 'W': 0.37, 'Y': 0.02, 'V': 0.54
}

# Kyte-Doolittle for general hydrophobicity windows
KD_HYDROPHOBICITY = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'W': -0.9, 'T': -0.7, 'S': -0.8,
    'Y': -1.3, 'P': -1.6, 'H': -3.2, 'D': -3.5, 'N': -3.5,
    'Q': -3.5, 'E': -3.5, 'K': -3.9, 'R': -4.5, 'G': -0.4
}


def net_charge(seq: str, window: int = 30) -> tuple[int, int, int]:
    s = seq[:window]
    pos = s.count('K') + s.count('R')
    neg = s.count('D') + s.count('E')
    return pos - neg, pos, neg


def de_count(seq: str, window: int = 40) -> tuple[int, int, int]:
    s = seq[:window]
    return s.count('D') + s.count('E'), s.count('D'), s.count('E')


def hydrophobic_moment(seq: str, window: int = 18, angle_deg: float = 100.0,
                        search_window: int = 80) -> float:
    """Eisenberg hydrophobic moment for alpha-helix detection."""
    best_mu = 0.0
    angle = math.radians(angle_deg)
    for start in range(min(search_window - window, len(seq) - window)):
        s = seq[start:start + window]
        sin_sum = cos_sum = 0.0
        for i, aa in enumerate(s):
            h = HYDROPHOBICITY.get(aa, 0.0)
            theta = angle * i
            sin_sum += h * math.sin(theta)
            cos_sum += h * math.cos(theta)
        mu = math.sqrt(sin_sum**2 + cos_sum**2) / window
        if mu > best_mu:
            best_mu = mu
    return best_mu


def find_motif(seq: str, pattern: str, window: Optional[int] = None,
               degenerate: bool = False) -> list[tuple[int, str]]:
    """
    Find motif in sequence. Returns list of (1-indexed pos, matched_seq).
    If degenerate=True, use regex. Otherwise exact match.
    Pattern like 'GxGxxGKT' where x=any, use degenerate=True.
    """
    if window is not None:
        seq = seq[:window]
    if degenerate:
        regex = re.escape(pattern).replace(r'x', r'.')
        hits = [(m.start() + 1, m.group()) for m in re.finditer(regex, seq, re.IGNORECASE)]
    else:
        hits = [(m.start() + 1, m.group()) for m in re.finditer(re.escape(pattern), seq, re.IGNORECASE)]
    return hits


def find_pattern(seq: str, regex: str, window: Optional[int] = None) -> list[tuple[int, str]]:
    """Find regex pattern. Returns list of (1-indexed pos, matched)."""
    if window is not None:
        seq = seq[:window]
    return [(m.start() + 1, m.group()) for m in re.finditer(regex, seq)]


def mts_score(seq: str, config: dict) -> dict:
    """
    Compute MTS composite score (0-5) and individual features.
    Based on MitoFates / TargetP 2.0 criteria.
    """
    cfg = config.get("motifs", {}).get("mts", {})
    min_charge = cfg.get("min_net_charge_30aa", 2)
    max_de = cfg.get("max_de_40aa", 3)
    min_mu = cfg.get("min_hydrophobic_moment", 0.12)
    hhh_window = cfg.get("hhh_window", 30)

    charge, pos_aa, neg_aa = net_charge(seq, 30)
    de, d_cnt, e_cnt = de_count(seq, 40)
    mu = hydrophobic_moment(seq)

    hhh_hits = find_pattern(seq, r'HHH', window=hhh_window)
    has_hhh = len(hhh_hits) > 0

    lyr_hits = find_pattern(seq, r'LYR')
    has_lyr = len(lyr_hits) > 0

    criteria = [
        charge >= min_charge,
        de <= max_de,
        mu >= min_mu,
        not has_hhh,
        has_lyr,
    ]
    composite = sum(criteria)

    if composite >= 4:
        prediction = "HIGH (mitochondrial import likely)"
    elif composite >= 2:
        prediction = "INTERMEDIATE (uncertain)"
    else:
        prediction = "LOW (cytoplasmic expected)"

    return {
        "net_charge_30aa": charge,
        "pos_aa_30": pos_aa,
        "neg_aa_30": neg_aa,
        "de_count_40aa": de,
        "hydrophobic_moment": round(mu, 4),
        "hhh_motif": hhh_hits[0] if has_hhh else None,
        "lyr_motif": lyr_hits[0] if has_lyr else None,
        "composite_score": composite,
        "prediction": prediction,
        "criteria_passed": {
            "net_charge_ge_2": criteria[0],
            "de_le_3": criteria[1],
            "mu_ge_0.12": criteria[2],
            "no_hhh": criteria[3],
            "has_lyr": criteria[4],
        }
    }


def functional_motifs(seq: str, config: dict) -> dict:
    """
    Detect functional motifs relevant to known protein families.
    Returns dict of motif_name → list of hits.
    """
    results = {}

    # Kinesin motor domain
    results["kinesin_ploop"] = find_pattern(seq, r'G[QKRE]TG[AST]GK[TS]')  # P-loop
    results["kinesin_switch1"] = find_pattern(seq, r'SSR[SNAHQ][HAN][ASD]')  # Switch-I
    results["kinesin_switch2"] = find_pattern(seq, r'D[LIVMA][AGSQ]G[ST][AEQK]')  # Switch-II DxxG

    # PDZ domain
    pdz_hits = []
    for glgf in ["GLGF", "GVGF", "GMGF", "GQGF"]:
        pdz_hits.extend(find_motif(seq, glgf))
    results["pdz_glgf"] = sorted(set(pdz_hits))

    # WD40 repeat
    results["wd40"] = find_pattern(seq, r'W[DN][LIVMFA]{2,4}[LIVMFA]{1,3}[DN]')

    # LYR motif (Complex I)
    results["lyr"] = find_pattern(seq, r'LYR')

    # YMDD/YVDD (RT catalytic)
    results["rt_ymdd"] = find_pattern(seq, r'[YF][LVMI][DD]')

    # L27 domain signature (rough: helical bundle charged pattern)
    results["l27_signature"] = find_pattern(seq, r'[LI][LI][QEK][EQ][AL]')

    # Coiled-coil heptad (simplified)
    results["coiled_coil_heptad"] = find_pattern(seq, r'[LIVMA][LIVMA]{2}[LIVMA]{3}[EKR]')

    # Signal peptide: hydrophobic core in first 30aa (KD scale > 1.5 average over 7 aa)
    results["hydrophobic_core_n30"] = _find_hydrophobic_core(seq[:30], threshold=1.5, window=7)

    # Remove empty
    return {k: v for k, v in results.items() if v}


def _find_hydrophobic_core(seq: str, threshold: float = 1.5, window: int = 7) -> list:
    hits = []
    for i in range(len(seq) - window + 1):
        w = seq[i:i+window]
        avg = sum(KD_HYDROPHOBICITY.get(aa, 0) for aa in w) / window
        if avg >= threshold:
            hits.append((i + 1, w))
    return hits


def analyze_sequence(seq: str, config: dict) -> dict:
    """Run all M3 analyses on a single sequence."""
    return {
        "length": len(seq),
        "mts": mts_score(seq, config),
        "functional_motifs": functional_motifs(seq, config),
        "composition": {
            "K": seq.count("K"),
            "R": seq.count("R"),
            "D": seq.count("D"),
            "E": seq.count("E"),
            "H": seq.count("H"),
            "pct_charged": (seq.count("K") + seq.count("R") + seq.count("D") + seq.count("E")) / len(seq) * 100 if seq else 0,
        }
    }
