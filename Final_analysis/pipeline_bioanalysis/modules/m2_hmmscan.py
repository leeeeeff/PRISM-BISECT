"""
M2: Pfam-A hmmscan domain annotation
Runs hmmscan and parses domtblout. Also implements Stage 2 domain-change filter.
"""
import os
import subprocess
import tempfile
from pathlib import Path


def run_hmmscan(fasta_path: str, config: dict, out_dir: str) -> dict[str, list]:
    """
    Run hmmscan and return {transcript_id: [domain_hits]}.
    Each hit: {domain, evalue, score, ali_from, ali_to, hmm_from, hmm_to, rep_family}
    """
    pfam_db = config["paths"]["pfam_db"]
    hmmscan_bin = config["paths"]["hmmscan_bin"]
    max_e = config["stage2"]["min_domain_evalue"]

    tblout = os.path.join(out_dir, "hmmscan_domains.tblout")
    hmm_out = os.path.join(out_dir, "hmmscan.out")

    cmd = [
        hmmscan_bin,
        "--domtblout", tblout,
        "-E", str(max_e),
        "--domE", str(max_e),
        "--cpu", "4",
        pfam_db, fasta_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [M2] hmmscan error: {result.stderr[:200]}")
        return {}
    with open(hmm_out, "w") as f:
        f.write(result.stdout)

    return parse_domtblout(tblout, max_e)


def parse_domtblout(tblout_path: str, max_e: float = 0.01) -> dict[str, list]:
    """Parse hmmscan --domtblout output."""
    results = {}
    if not os.path.exists(tblout_path):
        return results

    with open(tblout_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split()
            if len(cols) < 22:
                continue
            domain = cols[0]
            query = cols[3]
            seq_evalue = float(cols[6])
            dom_evalue = float(cols[11])
            dom_score = float(cols[13])
            hmm_from = int(cols[15])
            hmm_to = int(cols[16])
            ali_from = int(cols[17])
            ali_to = int(cols[18])

            if dom_evalue > max_e:
                continue

            if query not in results:
                results[query] = []
            results[query].append({
                "domain": domain,
                "evalue": dom_evalue,
                "score": dom_score,
                "ali_from": ali_from,
                "ali_to": ali_to,
                "hmm_from": hmm_from,
                "hmm_to": hmm_to,
                "pfam_family": domain.split(".")[0],
            })

    # Sort each list by ali_from
    for tid in results:
        results[tid].sort(key=lambda x: x["ali_from"])

    return results


def detect_domain_changes(ct_domains: list, ad_domains: list) -> dict:
    """
    Stage 2 filter: compare domain sets between CT and AD isoforms.
    Returns domain change summary.
    """
    ct_set = {h["pfam_family"] for h in ct_domains}
    ad_set = {h["pfam_family"] for h in ad_domains}

    lost = ct_set - ad_set      # Domains in CT but not AD
    gained = ad_set - ct_set    # Domains in AD but not CT
    shared = ct_set & ad_set

    return {
        "has_domain_change": bool(lost or gained),
        "domains_lost": sorted(lost),
        "domains_gained": sorted(gained),
        "domains_shared": sorted(shared),
        "ct_domain_count": len(ct_set),
        "ad_domain_count": len(ad_set),
    }


def get_domain_summary(domains: list) -> list[str]:
    """Return list of domain names for display."""
    return [f"{h['domain']}(aa{h['ali_from']}-{h['ali_to']},E={h['evalue']:.1e})" for h in domains]
