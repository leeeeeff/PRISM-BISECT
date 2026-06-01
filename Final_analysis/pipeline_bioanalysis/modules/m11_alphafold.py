"""
M11: AlphaFold Structural Confidence Analysis (BISECT v1.1)

For each CT/AD isoform pair, retrieves per-residue pLDDT scores:
  - Canonical isoforms: AlphaFold DB REST API → pLDDT from PDB B-factor
  - Novel isoforms (SQANTI3 NIC/NNIC): ESMFold API (POST sequence)

Then maps pLDDT onto Pfam domains (from M2) to assess structural confidence
of gained/lost domains, strengthening mechanistic claims.

pLDDT interpretation:
  >90  : very high confidence (crystal-quality)
  70-90: high confidence, structural claim supported
  50-70: low confidence, likely disordered
  <50  : unreliable, no structural claim

Entry point: run(case_result, config) -> dict
"""
import json
import time
import io
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional


# ── UniProt gene→accession lookup ─────────────────────────────────────────────

def _uniprot_accession(gene_symbol: str, cfg: dict, retries: int = 3) -> Optional[str]:
    """Fetch the reviewed (Swiss-Prot) UniProt accession for a human gene symbol."""
    base = cfg.get("uniprot_lookup_url",
                   "https://rest.uniprot.org/uniprotkb/search")
    params = urllib.parse.urlencode({
        "query": f"gene:{gene_symbol} AND organism_id:9606 AND reviewed:true",
        "format": "json",
        "fields": "accession,gene_names",
        "size": 1,
    })
    url = f"{base}?{params}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DIFFUSE-BISECT/1.1"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
            results = data.get("results", [])
            if results:
                return results[0].get("primaryAccession")
            return None
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [M10] UniProt lookup failed for {gene_symbol}: {e}")
                return None
    return None


# ── AlphaFold DB ──────────────────────────────────────────────────────────────

def _alphafold_plddt(uniprot_id: str, cfg: dict, retries: int = 3) -> Optional[list[float]]:
    """
    Fetch per-residue pLDDT from AlphaFold DB.
    Returns list of floats (one per residue) or None on failure.
    """
    db_url = cfg.get("db_url", "https://alphafold.ebi.ac.uk/api/prediction")
    url = f"{db_url}/{uniprot_id}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DIFFUSE-BISECT/1.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            if isinstance(data, list) and data:
                entry = data[0]
                # v4: plddt list in JSON
                if "plddt" in entry:
                    return [float(v) for v in entry["plddt"]]
                # Fallback: fetch PDB and parse B-factors
                pdb_url = entry.get("pdbUrl")
                if pdb_url:
                    return _plddt_from_pdb_url(pdb_url, retries=retries)
            return None
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [M10] AlphaFold DB failed for {uniprot_id}: {e}")
                return None
    return None


def _plddt_from_pdb_url(pdb_url: str, retries: int = 3) -> Optional[list[float]]:
    """Download PDB file and extract per-CA pLDDT (B-factor of CA atoms)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(pdb_url, headers={"User-Agent": "DIFFUSE-BISECT/1.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                pdb_text = resp.read().decode("utf-8", errors="ignore")
            return _parse_ca_bfactors(pdb_text)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [M10] PDB download failed: {e}")
                return None
    return None


def _parse_ca_bfactors(pdb_text: str) -> list[float]:
    """Extract B-factor (pLDDT) for CA atoms, one per residue."""
    scores = []
    seen_res = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        chain = line[21]
        res_seq = line[22:26].strip()
        key = (chain, res_seq)
        if key in seen_res:
            continue
        seen_res.add(key)
        try:
            bfac = float(line[60:66])
            scores.append(bfac)
        except (ValueError, IndexError):
            pass
    return scores if scores else []


# ── ESMFold (local via fair-esm) ──────────────────────────────────────────────

_esmfold_model = None   # module-level singleton; loaded once per process

def _load_esmfold_model(device: str = "cuda"):
    """
    Load ESMFold v1 via fair-esm (requires omegaconf + openfold).
    If openfold is unavailable, falls back silently so the API path is tried.
    """
    global _esmfold_model
    if _esmfold_model is not None:
        return _esmfold_model
    try:
        import esm as esm_pkg
        import torch
        # openfold is a hard dependency of esmfold_v1; catch ImportError cleanly
        try:
            import openfold  # noqa: F401
        except ImportError:
            print("  [M10] openfold not installed — ESMFold local unavailable. "
                  "Install with: pip install git+https://github.com/aqlaboratory/openfold.git")
            return None
        print("  [M10] Loading ESMFold v1 (first call — may download ~700 MB)...")
        model = esm_pkg.pretrained.esmfold_v1()
        model = model.eval()
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda().half()
        _esmfold_model = model
        print(f"  [M10] ESMFold v1 loaded on {device}.")
        return model
    except Exception as e:
        print(f"  [M10] ESMFold local load failed: {e}")
        return None


def _esmfold_plddt(sequence: str, cfg: dict, retries: int = 1) -> Optional[list[float]]:
    """
    Fold sequence with local ESMFold v1 (fair-esm).
    Falls back to public ESMAtlas API if local model unavailable.
    Returns per-residue pLDDT list or None.
    """
    if not sequence:
        return None
    # Truncate to ESMFold practical limit (~600aa for 24GB VRAM with fp16)
    seq_trimmed = sequence[:600]

    # ── Try local ESMFold first ──────────────────────────────────────────────
    device = cfg.get("esmfold_device", "cuda")
    model = _load_esmfold_model(device)
    if model is not None:
        try:
            import torch
            with torch.no_grad():
                output = model.infer_pdb(seq_trimmed)
            scores = _parse_ca_bfactors(output)
            if scores:
                print(f"  [M10] ESMFold local: {len(scores)} residues folded")
                return scores
        except Exception as e:
            print(f"  [M10] ESMFold local inference failed ({e}), trying API fallback")

    # ── Fallback: public ESMAtlas API ────────────────────────────────────────
    esmfold_url = cfg.get("esmfold_url", "https://esmatlas.com/api/fold")
    timeout = cfg.get("timeout", 120)
    data = seq_trimmed.encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                esmfold_url, data=data, method="POST",
                headers={"Content-Type": "application/x-www-form-urlencoded",
                         "User-Agent": "DIFFUSE-BISECT/1.1"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                pdb_text = resp.read().decode("utf-8", errors="ignore")
            scores = _parse_ca_bfactors(pdb_text)
            return scores if scores else None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"  [M10] ESMFold API also failed: {e}")
                return None
    return None


# ── pLDDT → domain mapping ────────────────────────────────────────────────────

def _confidence_class(mean_plddt: float, cfg: dict) -> str:
    high = cfg.get("plddt_high_threshold", 70.0)
    very_high = cfg.get("plddt_very_high_threshold", 90.0)
    if mean_plddt >= very_high:
        return "very_high"
    elif mean_plddt >= high:
        return "high"
    elif mean_plddt >= 50.0:
        return "low"
    return "very_low"


def _map_domains_to_plddt(domains: list[dict], plddt: list[float],
                           cfg: dict) -> dict[str, dict]:
    """
    Map Pfam domain residue ranges to pLDDT slices.
    Domains store ali_from/ali_to (1-based protein residue coords).
    """
    result = {}
    for d in domains:
        pfam_id = d.get("pfam_family") or d.get("name", "unknown")
        try:
            res_start = int(d.get("ali_from", 1)) - 1   # 0-based
            res_end = int(d.get("ali_to", len(plddt)))
        except (TypeError, ValueError):
            continue
        if res_start >= len(plddt):
            continue
        res_end = min(res_end, len(plddt))
        slice_plddt = plddt[res_start:res_end]
        if not slice_plddt:
            continue
        mean_p = sum(slice_plddt) / len(slice_plddt)
        result[pfam_id] = {
            "pfam_id": d.get("pfam_id", pfam_id),
            "residues": f"{res_start + 1}-{res_end}",
            "mean": round(mean_p, 2),
            "min": round(min(slice_plddt), 2),
            "max": round(max(slice_plddt), 2),
            "confidence": _confidence_class(mean_p, cfg),
        }
    return result


# ── Per-isoform analysis ──────────────────────────────────────────────────────

def _analyze_isoform(transcript_id: str, seq: str, domains: list,
                     structural_category: Optional[str],
                     gene: str, cfg: dict) -> dict:
    """
    Fetch pLDDT for one isoform and map to domains.
    Chooses AlphaFold DB (canonical) or ESMFold (novel).
    """
    result: dict = {
        "transcript_id": transcript_id,
        "uniprot_id": None,
        "source": None,
        "plddt_mean": None,
        "plddt_high_fraction": None,
        "domain_plddt": {},
        "error": None,
    }

    # SQANTI3 uses full strings: "novel_in_catalog", "novel_not_in_catalog", etc.
    _cat = (structural_category or "").lower()
    is_novel = ("novel" in _cat or _cat in
                {"intergenic", "genic_intron", "genic intron", "antisense",
                 "nic", "nnic", "fusion"})
    plddt = None

    if not is_novel:
        # Try AlphaFold DB via UniProt gene lookup
        uniprot = _uniprot_accession(gene, cfg)
        result["uniprot_id"] = uniprot
        if uniprot:
            plddt = _alphafold_plddt(uniprot, cfg)
            if plddt:
                result["source"] = "alphafold_db"
                print(f"  [M10] {transcript_id}: AlphaFold DB OK (UniProt={uniprot}, {len(plddt)} residues)")

    if plddt is None:
        # Fallback: ESMFold for novel or when AFDB unavailable
        if seq:
            print(f"  [M10] {transcript_id}: using ESMFold (category={structural_category})")
            plddt = _esmfold_plddt(seq, cfg)
            if plddt:
                result["source"] = "esmfold"
            else:
                result["error"] = "ESMFold API unavailable"
                return result
        else:
            result["error"] = "no sequence available"
            return result

    high_thresh = cfg.get("plddt_high_threshold", 70.0)
    result["plddt_mean"] = round(sum(plddt) / len(plddt), 2)
    result["plddt_high_fraction"] = round(
        sum(1 for v in plddt if v >= high_thresh) / len(plddt), 3)
    result["domain_plddt"] = _map_domains_to_plddt(domains, plddt, cfg)
    return result


# ── Domain comparison ─────────────────────────────────────────────────────────

def _compare_domain_plddt(ct_result: dict, ad_result: dict,
                           domain_change: dict) -> dict:
    """Summarise structural confidence changes for gained/lost domains."""
    gained = domain_change.get("domains_gained", [])
    lost = domain_change.get("domains_lost", [])

    gained_confident, lost_confident, degraded = [], [], []

    # Gained domains: check AD confidence
    for d in gained:
        name = d.get("pfam_family") or d.get("name", "") if isinstance(d, dict) else str(d)
        ad_dom = ad_result.get("domain_plddt", {}).get(name)
        if ad_dom and ad_dom["confidence"] in ("high", "very_high"):
            gained_confident.append(f"{name} (pLDDT={ad_dom['mean']})")

    # Lost domains: check CT confidence to show what was present
    for d in lost:
        name = d.get("pfam_family") or d.get("name", "") if isinstance(d, dict) else str(d)
        ct_dom = ct_result.get("domain_plddt", {}).get(name)
        if ct_dom and ct_dom["confidence"] in ("high", "very_high"):
            lost_confident.append(f"{name} (pLDDT={ct_dom['mean']})")

    # Shared domains: check if confidence degraded in AD
    shared = domain_change.get("domains_shared", [])
    for d in shared:
        name = d.get("pfam_family") or d.get("name", "") if isinstance(d, dict) else str(d)
        ct_dom = ct_result.get("domain_plddt", {}).get(name)
        ad_dom = ad_result.get("domain_plddt", {}).get(name)
        if ct_dom and ad_dom:
            drop = ct_dom["mean"] - ad_dom["mean"]
            if drop > 10:
                degraded.append(f"{name} ({ct_dom['mean']}→{ad_dom['mean']}, Δ={drop:.1f})")

    parts = []
    if gained_confident:
        parts.append(f"AD gains confident domain(s): {'; '.join(gained_confident)}.")
    if lost_confident:
        parts.append(f"AD loses confident domain(s): {'; '.join(lost_confident)}.")
    if degraded:
        parts.append(f"Shared domain(s) show reduced confidence in AD: {'; '.join(degraded)}.")
    if not parts:
        parts.append("No significant structural confidence difference detected between CT and AD isoforms.")

    return {
        "gained_domain_confident": gained_confident,
        "lost_domain_confident": lost_confident,
        "degraded_domains": degraded,
        "interpretation": " ".join(parts),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def run(case_result: dict, config: dict) -> dict:
    """
    M11: AlphaFold structural confidence analysis.
    Requires M1 (sequences) and M2 (domains) in case_result.
    """
    cfg = config.get("alphafold", {})
    rate_delay = cfg.get("rate_delay", 2.0)

    gene = case_result.get("gene_name", "")
    ct_id = case_result.get("ct_transcript_id", "")
    ad_id = case_result.get("ad_transcript_id", "")
    ct_seq = (case_result.get("ct_seq") or {}).get("seq", "")
    ad_seq = (case_result.get("ad_seq") or {}).get("seq", "")
    ct_domains = case_result.get("ct_domains", [])
    ad_domains = case_result.get("ad_domains", [])
    ct_category = (case_result.get("ct_info") or {}).get("structural_category")
    ad_category = (case_result.get("ad_info") or {}).get("structural_category")
    domain_change = case_result.get("domain_change", {})

    print(f"  [M10] {gene}: CT={ct_id} ({ct_category}), AD={ad_id} ({ad_category})")

    ct_result = _analyze_isoform(ct_id, ct_seq, ct_domains, ct_category, gene, cfg)
    time.sleep(rate_delay)
    ad_result = _analyze_isoform(ad_id, ad_seq, ad_domains, ad_category, gene, cfg)

    comparison = _compare_domain_plddt(ct_result, ad_result, domain_change)

    return {
        "ct": ct_result,
        "ad": ad_result,
        "comparison": comparison,
    }
