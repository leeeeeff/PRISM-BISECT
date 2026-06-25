"""
Update existing bisect_cases.json (84 cases) with bisect_tier, dtu_method, perm_p fields.

5-Tier system:
  A-DR : DRIMSeq+stageR primary discovery (14 genes, this script handles existing cases only)
  A-BP : BISECT-permutation validated (donor-label permutation n=10,000)
  B    : Independent cohort replication
  C    : BISECT multi-module only (pooled chi-sq exploratory or not applicable)
  D    : No statistical support
"""
import json
from pathlib import Path

JSON_PATH = Path(__file__).parent.parent / "prism_app/data/demo/bisect_cases.json"

BRAIN_TYPES = {"Excitatory", "Inhibitory", "Astrocyte", "Microglia", "OPC", "Oligodendrocyte"}
MUSCLE_TYPES = {"Skeletal_muscle", "Cardiomyocyte"}

# (gene, cell_type) → bisect_tier override
TIER_MAP = {
    ("DOCK11", "Inhibitory"):  "A-BP",
    ("NDUFS4", "Excitatory"): "A-BP",
    ("KIF21B", "Excitatory"): "B",
    ("DLG1",   "OPC"):        "D",
}

# (gene, cell_type) → perm_p
PERM_P_MAP = {
    ("DOCK11", "Inhibitory"):  0.0008,
    ("NDUFS4", "Excitatory"): 0.041,
}

def get_tier(gene: str, cell_type: str) -> str:
    key = (gene, cell_type)
    if key in TIER_MAP:
        return TIER_MAP[key]
    return "C"

def get_dtu_method(cell_type: str) -> str:
    if cell_type in BRAIN_TYPES:
        return "pooled_chisq_exploratory"
    return "not_applicable"

def main():
    with open(JSON_PATH) as f:
        cases = json.load(f)

    updated = 0
    for case in cases:
        gene = case.get("gene", "")
        ct = case.get("cell_type", "")
        key = (gene, ct)

        if "bisect_tier" not in case:
            case["bisect_tier"] = get_tier(gene, ct)
            updated += 1

        if "dtu_method" not in case:
            case["dtu_method"] = get_dtu_method(ct)

        if "perm_p" not in case and key in PERM_P_MAP:
            case["perm_p"] = PERM_P_MAP[key]

    with open(JSON_PATH, "w") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    print(f"Updated {updated}/{len(cases)} cases with bisect_tier.")
    # Summary
    from collections import Counter
    tier_counts = Counter(c.get("bisect_tier", "?") for c in cases)
    for tier, cnt in sorted(tier_counts.items()):
        print(f"  Tier {tier}: {cnt}")

if __name__ == "__main__":
    main()
