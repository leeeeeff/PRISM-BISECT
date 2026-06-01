"""
MTS (Mitochondrial Targeting Sequence) feature analysis
NDUFS4-201 canonical vs tr73243 (AD locus-hijacking isoform)

MitoFates-style sequence features:
- Net charge in N-terminal 30aa (positive = MTS-favorable)
- D+E count in first 40aa (high = MTS-unfavorable)
- Hydrophobic moment (amphipathic alpha-helix indicator)
- HHH motif check (unusual for MTS)
"""

import math
import re

# === Sequences ===
# tr73243: NDUFS4 locus hijacking (AD isoform, 379aa) — from SQANTI3 FAA
TR73243 = "MASQFSHHHLLNRDRFPICFCGAITTDPTEIQTTIREYYKHLYANELENLGEMGKFLDTYTLPRLNQEEVESLNRPVTGSEIEAIINSLLAKKSPGPDGFTAEFYQRYKEELVPFLLKLFQSIEKVGILPNSFYEASIILIPKPGRDTTKKDNFRPISLMNIDVKILNKILANRIQQHIKKLIHHHQVGFIPGMQGWFNICKSINVIQHVNRANDKNHMIISIDAEKAFDKIQQHFMLKTLSKLGIDGTYLKIIRAICDKPTANIILNGQKLEAFPLKTGTRQGYPLSPLLFNIVLEVLVRAIRQEKEIKGIQLGKEEVKLSLFADDMIVYLENPIISAQNLLKLIGNFSKVSGYKINVQKSQSFLYTNNRQTEPNHE"

# NDUFS4-201 canonical (UniProt Q16718, 175aa)
# Well-established MTS at N-terminal ~42aa (cleaved after import)
NDUFS4_201 = "MASRLLSVRGAIAGTAAMRNLFQSSAEDKQAQEKEAKQEAEAARQKQESEKPAVPLLQKSVPEVLKQTLSVPPQAEIPQAEQALEDVDAVDDLDAFEQELRQVDQSAADLAKRRRELERLEEARKRAEAEDVDKRRRELE"
# Note: Q16718 is 175aa; using first 140aa available from published sequence

def net_charge(seq, window=30):
    """Net charge in first N residues (K,R positive; D,E negative)"""
    s = seq[:window]
    pos = s.count('K') + s.count('R')
    neg = s.count('D') + s.count('E')
    return pos - neg, pos, neg

def de_count(seq, window=40):
    """Count D+E in first N residues (MTS-unfavorable if high)"""
    s = seq[:window]
    return s.count('D') + s.count('E'), s.count('D'), s.count('E')

def hydrophobic_moment(seq, window=18, angle=100):
    """
    Eisenberg hydrophobic moment for alpha-helix (100° rotation per residue).
    Higher μH = stronger amphipathic helix = more MTS-like.
    """
    # Eisenberg hydrophobicity scale
    hydrophobicity = {
        'A': 0.25, 'R': -1.76, 'N': -0.64, 'D': -0.72, 'C': 0.04,
        'Q': -0.69, 'E': -0.62, 'G': 0.16, 'H': -0.40, 'I': 0.73,
        'L': 0.53, 'K': -1.10, 'M': 0.26, 'F': 0.61, 'P': -0.07,
        'S': -0.26, 'T': -0.18, 'W': 0.37, 'Y': 0.02, 'V': 0.54
    }
    best_mu = 0
    # Slide window across first 80aa
    for start in range(min(80 - window, len(seq) - window)):
        s = seq[start:start + window]
        sin_sum = cos_sum = 0
        for i, aa in enumerate(s):
            h = hydrophobicity.get(aa, 0)
            theta = math.radians(angle * i)
            sin_sum += h * math.sin(theta)
            cos_sum += h * math.cos(theta)
        mu = math.sqrt(sin_sum**2 + cos_sum**2) / window
        if mu > best_mu:
            best_mu = mu
    return best_mu

def check_hhh_motif(seq):
    """Check for trihistidine motif (unusual for canonical MTS)"""
    for i in range(min(30, len(seq)-2)):
        if seq[i] == 'H' and seq[i+1] == 'H' and seq[i+2] == 'H':
            return True, i+1  # 1-indexed position
    return False, None

def mts_lyr_check(seq):
    """Check for LYR motif (required for Complex I N-module integration)"""
    # LYR motif: Leu-Tyr-Arg in specific context
    matches = [(m.start()+1, m.group()) for m in re.finditer(r'LYR', seq)]
    return len(matches) > 0, matches

def analyze_sequence(name, seq, label):
    print(f"\n{'='*60}")
    print(f"[{name}] {label}")
    print(f"Length: {len(seq)} aa")
    print(f"N-terminal 40aa: {seq[:40]}")
    print()
    
    # Net charge
    net, pos, neg = net_charge(seq, window=30)
    print(f"Net charge (first 30aa): {net:+d}  (K+R={pos}, D+E={neg})")
    print(f"  {'✅ MTS-favorable (net charge ≥ +2)' if net >= 2 else '❌ MTS-unfavorable (net charge < +2)'}")
    
    # D+E count
    de, d, e = de_count(seq, window=40)
    print(f"D+E count (first 40aa): {de}  (D={d}, E={e})")
    print(f"  {'✅ MTS-favorable (D+E ≤ 3)' if de <= 3 else '❌ MTS-unfavorable (D+E > 3, acidic residues impair import)'}")
    
    # Hydrophobic moment
    mu = hydrophobic_moment(seq)
    print(f"Hydrophobic moment (μH): {mu:.4f}")
    print(f"  {'✅ Amphipathic helix likely (μH ≥ 0.12)' if mu >= 0.12 else '❌ Weak amphipathicity (μH < 0.12)'}")
    
    # HHH motif
    has_hhh, pos_hhh = check_hhh_motif(seq)
    if has_hhh:
        print(f"HHH motif: PRESENT at position {pos_hhh} → ❌ Breaks amphipathic helix, not seen in canonical MTS")
    else:
        print(f"HHH motif: ABSENT → ✅ Normal for canonical MTS")
    
    # LYR motif
    has_lyr, lyr_matches = mts_lyr_check(seq)
    if has_lyr:
        print(f"LYR motif: PRESENT at {lyr_matches} → ✅ Complex I N-module assembly compatible")
    else:
        print(f"LYR motif: ABSENT → ❌ Cannot interact with NDUFAF4 assembly factor")
    
    # MTS favorable/unfavorable count
    flags = [net >= 2, de <= 3, mu >= 0.12, not has_hhh, has_lyr]
    score = sum(flags)
    print(f"\nMTS Composite Score: {score}/5")
    if score >= 4:
        print("  → HIGH MTS probability (mitochondrial import expected)")
    elif score >= 2:
        print("  → INTERMEDIATE (uncertain localization)")
    else:
        print("  → LOW MTS probability (cytoplasmic localization expected)")

# === Run Analysis ===
print("NDUFS4 Isoform MTS Feature Analysis")
print("="*60)
print("Reference: MitoFates (Fukasawa et al. 2015) feature categories")
print("           TargetP 2.0 (Almagro Armenteros et al. 2019)")
print()

analyze_sequence("NDUFS4-201", NDUFS4_201, "Canonical — known mitochondrial (UniProt Q16718)")
analyze_sequence("tr73243", TR73243, "AD locus-hijacking isoform (NNIC, 379aa)")

# === Summary Table ===
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"{'Feature':<30} {'NDUFS4-201':>15} {'tr73243':>15}")
print("-"*60)

n_net, n_pos, n_neg = net_charge(NDUFS4_201)
t_net, t_pos, t_neg = net_charge(TR73243)
print(f"{'Net charge (30aa)':<30} {n_net:>+15} {t_net:>+15}")

n_de, *_ = de_count(NDUFS4_201)
t_de, *_ = de_count(TR73243)
print(f"{'D+E count (40aa)':<30} {n_de:>15} {t_de:>15}")

n_mu = hydrophobic_moment(NDUFS4_201)
t_mu = hydrophobic_moment(TR73243)
print(f"{'Hydrophobic moment (μH)':<30} {n_mu:>15.4f} {t_mu:>15.4f}")

n_hhh, _ = check_hhh_motif(NDUFS4_201)
t_hhh, t_hpos = check_hhh_motif(TR73243)
print(f"{'HHH motif (aa 1-30)':<30} {'YES' if n_hhh else 'NO':>15} {'YES @'+str(t_hpos) if t_hhh else 'NO':>15}")

n_lyr, _ = mts_lyr_check(NDUFS4_201)
t_lyr, _ = mts_lyr_check(TR73243)
print(f"{'LYR motif':<30} {'YES' if n_lyr else 'NO':>15} {'YES' if t_lyr else 'NO':>15}")
