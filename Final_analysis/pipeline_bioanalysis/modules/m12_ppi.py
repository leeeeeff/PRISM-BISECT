"""
M12: Protein-Protein Interaction Network Validation (BISECT v1.1)

Queries STRING DB for experimental interaction evidence between the gene of interest
and hypothesized interaction partners. Validates domain-mediated interaction claims
made in M2/M3 (e.g., coiled-coil heterodimerization, decoy receptor binding).

Entry point: run(case_result, config) -> dict
"""
import json
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional


# ── STRING DB API ─────────────────────────────────────────────────────────────

_EVIDENCE_PRIORITY = [
    "escore",          # experimental
    "dscore",          # database (curated)
    "ascore",          # coexpression
    "tscore",          # textmining
    "nscore",          # neighborhood
    "fscore",          # fusion
    "pscore",          # phylogenetic cooccurrence
]

_EVIDENCE_LABELS = {
    "escore": "experimental",
    "dscore": "database",
    "ascore": "coexpression",
    "tscore": "textmining",
    "nscore": "neighborhood",
    "fscore": "fusion",
    "pscore": "phylogeny",
}


def _string_partners(gene: str, species: int, limit: int,
                     base_url: str, retries: int = 3) -> list[dict]:
    """Query STRING interaction_partners endpoint. Returns raw JSON list."""
    params = urllib.parse.urlencode({
        "identifiers": gene,
        "species": species,
        "limit": limit,
        "caller_identity": "DIFFUSE-BISECT",
    })
    url = f"{base_url}?{params}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DIFFUSE-BISECT/1.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [M11] STRING API failed for {gene}: {e}")
                return []
    return []


def _string_pairwise(gene_a: str, gene_b: str, species: int,
                     base_url: str, retries: int = 3) -> Optional[dict]:
    """Get STRING scores for a specific gene pair."""
    pair_url = base_url.replace("interaction_partners", "network")
    params = urllib.parse.urlencode({
        "identifiers": f"{gene_a}\r{gene_b}",
        "species": species,
        "caller_identity": "DIFFUSE-BISECT",
    })
    url = f"{pair_url}?{params}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DIFFUSE-BISECT/1.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            if isinstance(data, list) and data:
                return data[0]
            return None
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


# ── Scoring logic ─────────────────────────────────────────────────────────────

def _confidence_tier(score: float, cfg: dict) -> str:
    high = cfg.get("high_score", 700)
    mid = cfg.get("medium_score", 400)
    if score >= high:
        return "high"
    elif score >= mid:
        return "medium"
    elif score > 0:
        return "low"
    return "none"


def _top_evidence_types(hit: dict) -> list[str]:
    """Return evidence channel names where score > 0, ordered by reliability."""
    present = []
    for key in _EVIDENCE_PRIORITY:
        val = hit.get(key, 0)
        if isinstance(val, str):
            try:
                val = float(val)
            except ValueError:
                val = 0
        if val and float(val) > 0:
            present.append(_EVIDENCE_LABELS[key])
    return present


def _build_hit(raw: dict) -> dict:
    gene_b = raw.get("preferredName_B", raw.get("stringId_B", ""))
    score = float(raw.get("score", 0))
    escore = float(raw.get("escore", 0))
    return {
        "partner": gene_b,
        "combined_score": round(score * 1000),   # STRING returns 0-1; scale to 0-1000
        "experimental_score": round(escore * 1000),
        "evidence_types": _top_evidence_types(raw),
    }


# ── Domain–interaction mapping ────────────────────────────────────────────────

# Known domain families that mediate specific interaction types
_DOMAIN_INTERACTION_ROLES = {
    "Coiled-coil": "coiled-coil-mediated dimerization or cargo-adaptor binding",
    "PDZ": "PDZ-ligand scaffold interaction",
    "PH": "membrane lipid or protein binding",
    "WD40": "protein complex assembly (β-propeller scaffold)",
    "Kinesin": "microtubule motor activity",
    "SH3": "proline-rich motif binding",
    "SH2": "phosphotyrosine binding",
    "RVT_1": "reverse transcriptase catalysis",
    "L27_1": "L27 domain-mediated trimerization",
    "Ig": "ligand or receptor binding (immunoglobulin-like)",
}

def _domain_interaction_link(case_result: dict, hit_partners: set) -> dict:
    """Link gained/lost domains to relevant interaction partners found in STRING."""
    domain_change = case_result.get("domain_change", {})
    gained = domain_change.get("domains_gained", [])
    lost = domain_change.get("domains_lost", [])

    links = []
    for d in gained + lost:
        pfam_id = d.get("pfam_family", "") if isinstance(d, dict) else str(d)
        # Match against domain role table by substring
        role = next((v for k, v in _DOMAIN_INTERACTION_ROLES.items()
                     if k.lower() in pfam_id.lower()), None)
        if role:
            links.append({
                "domain": pfam_id,
                "change": "gained" if d in gained else "lost",
                "interaction_role": role,
                "relevant_partners_found": list(hit_partners),
            })
    return {"domain_links": links}


# ── Entry point ───────────────────────────────────────────────────────────────

def run(case_result: dict, config: dict) -> dict:
    """
    M11: PPI validation via STRING DB.
    Uses per-gene partner hypotheses from config.string_db.partner_hypotheses.
    """
    cfg = config.get("string_db", {})
    base_url = cfg.get("url", "https://string-db.org/api/json/interaction_partners")
    species = cfg.get("species", 9606)
    limit = cfg.get("limit", 50)
    min_score = cfg.get("min_score", 150)
    rate_delay = cfg.get("rate_delay", 1.0)

    gene = case_result.get("gene_name", "")
    hypotheses = cfg.get("partner_hypotheses", {}).get(gene, [])

    print(f"  [M11] {gene}: querying STRING (partners hypothesis: {hypotheses})")

    # Fetch top-N partners for this gene
    raw_partners = _string_partners(gene, species, limit, base_url)
    time.sleep(rate_delay)

    # Build indexed dict by partner gene name
    partner_index: dict[str, dict] = {}
    for raw in raw_partners:
        h = _build_hit(raw)
        # STRING returns 0-1 scores; combined_score already scaled ×1000
        if h["combined_score"] >= min_score:
            partner_index[h["partner"].upper()] = h

    # Score hypothesis partners
    hypothesis_support = {}
    for hypo_partner in hypotheses:
        key = hypo_partner.upper()
        if key in partner_index:
            h = partner_index[key]
            hypothesis_support[hypo_partner] = {
                "combined_score": h["combined_score"],
                "experimental_score": h["experimental_score"],
                "confidence": _confidence_tier(h["combined_score"], cfg),
                "evidence_types": h["evidence_types"],
            }
        else:
            hypothesis_support[hypo_partner] = {
                "combined_score": 0,
                "experimental_score": 0,
                "confidence": "none",
                "evidence_types": [],
            }

    # Top STRING hits (any partner, not just hypotheses)
    top_hits = sorted(partner_index.values(),
                      key=lambda x: x["combined_score"], reverse=True)[:10]

    # Domain link analysis
    hit_partners = {p["partner"] for p in top_hits}
    domain_link = _domain_interaction_link(case_result, hit_partners)

    # Summary verdict
    supported = [p for p, v in hypothesis_support.items() if v["confidence"] in ("high", "medium")]
    exp_supported = [p for p, v in hypothesis_support.items() if "experimental" in v.get("evidence_types", [])]

    if exp_supported:
        verdict = "SUPPORTED"
        interp = (f"Hypothesis-partner(s) {exp_supported} show experimental PPI evidence in STRING "
                  f"(scores: {[hypothesis_support[p]['combined_score'] for p in exp_supported]}).")
    elif supported:
        verdict = "PARTIAL"
        interp = (f"Hypothesis-partner(s) {supported} found in STRING but without direct experimental evidence "
                  f"(scores: {[hypothesis_support[p]['combined_score'] for p in supported]}).")
    else:
        verdict = "UNSUPPORTED"
        interp = (f"None of the hypothesized partners ({hypotheses}) found with sufficient STRING evidence. "
                  "PPI claim requires experimental validation (co-IP / proximity-ligation).")

    return {
        "gene": gene,
        "hypothesized_partners": hypotheses,
        "string_hits": top_hits,
        "hypothesis_support": hypothesis_support,
        "domain_interaction_link": domain_link,
        "summary_verdict": verdict,
        "interpretation": interp,
    }
