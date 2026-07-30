"""
fetch_appris_reference.py
=========================
Build an expanded reference-label table for isoform-level "functional
alternativeness" annotations, used by the fluid-trajectory Stage 2
side-branch bio-validation (see reports/fluid_stage2/session_report_20260706.md
Priority 1).

Sources
-------
1. MANE_summary.txt.gz (local, already present) — one MANE Select
   transcript per gene. Non-MANE = alternative by definition.
2. APPRIS principal isoforms (download on demand from
   apprisws.bioinfo.cnio.es) — PRINCIPAL:1..5 / ALTERNATIVE:1..2
   functional-importance gradient.
3. Bambu counts_transcript.txt (local) — per-transcript expression
   across 24 muscle samples. Dominant non-MANE isoform in any sample
   = expression-switched proxy (avoids GTEx tarball download).

Output
------
hMuscle/data/reference_labels_v1.tsv with columns:
    tx_id_versioned    (ENST00000123456.7)
    tx_id_stripped     (ENST00000123456)
    gene_id_versioned  (ENSG00000123456.7)
    mane_status        (MANE Select | MANE Plus Clinical | "")
    appris_label       (PRINCIPAL:1..5 | ALTERNATIVE:1..2 | "")
    max_muscle_count   (float)
    is_gene_dominant   (bool, dominant in >= 1 sample)
    is_alt_functional  (bool; final flag = APPRIS ALTERNATIVE or
                        (non-MANE AND gene-dominant))

Rerun on demand; cheap (~1 min).
"""
from __future__ import annotations

import gzip
import os
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = ROOT / "model"
OUT_TSV = DATA / "reference_labels_v1.tsv"

MANE_GZ = DATA / "MANE_summary.txt.gz"
BAMBU_COUNTS = DATA / "bambu_data" / "counts_transcript.txt"
APPRIS_CACHE = DATA / "appris_data.principal.txt"
APPRIS_URL = (
    "https://apprisws.bioinfo.cnio.es/pub/current_release/"
    "datafiles/homo_sapiens/GRCh38/appris_data.principal.txt"
)

ISO_LIST_NPY = MODEL_DIR / "my_isoform_list_fixed.npy"
GENE_LIST_NPY = MODEL_DIR / "my_gene_list_fixed.npy"


def download_appris(force: bool = False) -> None:
    if APPRIS_CACHE.exists() and not force:
        print(f"[appris] cached: {APPRIS_CACHE} ({APPRIS_CACHE.stat().st_size} bytes)")
        return
    print(f"[appris] downloading {APPRIS_URL}")
    # APPRIS server has SSL cert chain issue on this host; use curl -k
    # rather than urllib which does not support cert bypass cleanly.
    res = subprocess.run(
        ["curl", "-k", "-sSL", "-o", str(APPRIS_CACHE), APPRIS_URL],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"curl failed: {res.stderr}")
    print(f"[appris] wrote {APPRIS_CACHE} ({APPRIS_CACHE.stat().st_size} bytes)")


def load_mane() -> pd.DataFrame:
    with gzip.open(MANE_GZ, "rt") as fh:
        df = pd.read_csv(fh, sep="\t")
    # keep ENST + status + gene (versioned)
    keep = df[["Ensembl_nuc", "Ensembl_Gene", "symbol", "MANE_status"]].copy()
    keep.columns = ["tx_id_versioned", "gene_id_versioned", "symbol", "mane_status"]
    keep["tx_id_stripped"] = keep["tx_id_versioned"].str.split(".").str[0]
    print(f"[mane] loaded {len(keep)} entries; statuses: "
          f"{keep['mane_status'].value_counts().to_dict()}")
    return keep


def load_appris() -> pd.DataFrame:
    # APPRIS principal file (6 tab columns, has header row):
    #   Gene name  Gene ID  Transcript ID  CCDS ID  APPRIS Annotation  MANE
    # Transcript ID is UNVERSIONED (ENST00000123456).
    df = pd.read_csv(APPRIS_CACHE, sep="\t", dtype=str, na_filter=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Gene name": "symbol",
        "Gene ID": "gene_id_stripped",
        "Transcript ID": "tx_id_stripped",
        "APPRIS Annotation": "appris_label",
    })
    print(f"[appris] loaded {len(df)} entries; label counts:")
    print(df["appris_label"].value_counts().to_string())
    return df[["tx_id_stripped", "appris_label"]]


def load_bambu_dominance() -> pd.DataFrame:
    print(f"[bambu] reading {BAMBU_COUNTS}")
    counts = pd.read_csv(BAMBU_COUNTS, sep="\t")
    sample_cols = counts.columns[2:].tolist()
    print(f"[bambu] {len(counts)} rows, {len(sample_cols)} samples")

    # per-gene per-sample max: which tx is dominant in each sample?
    # is_gene_dominant = 1 if this tx has the top count for its gene in any sample
    counts["max_count"] = counts[sample_cols].max(axis=1)
    dominant = (
        counts.groupby("GENEID", group_keys=False)
        .apply(lambda g: g.assign(
            is_gene_dominant=(g[sample_cols].values.argmax(axis=0) ==
                              np.arange(len(g))[:, None]).any(axis=1)
        ))
    )
    # simpler: for each sample col, find idxmax per gene; the union of those
    # across samples is the dominant set
    dom_ids = set()
    for col in sample_cols:
        tops = counts.loc[counts.groupby("GENEID")[col].idxmax(), "TXNAME"]
        dom_ids.update(tops.tolist())

    result = counts[["TXNAME", "GENEID", "max_count"]].copy()
    result["is_gene_dominant"] = result["TXNAME"].isin(dom_ids)
    result.columns = ["tx_id_versioned", "gene_id_bambu", "max_muscle_count", "is_gene_dominant"]
    result["tx_id_stripped"] = result["tx_id_versioned"].str.split(".").str[0]
    print(f"[bambu] {result['is_gene_dominant'].sum()} tx are gene-dominant in >=1 sample")
    return result


def coverage_report(ref: pd.DataFrame) -> None:
    print("\n=== coverage vs pilot isoform list ===")
    iso_raw = np.load(ISO_LIST_NPY, allow_pickle=True)
    iso_str = np.array([x.decode() if isinstance(x, bytes) else x for x in iso_raw])
    uniq = pd.unique(iso_str)
    print(f"unique isoforms in fixed list: {len(uniq)}")
    enst = pd.Series([s for s in uniq if s.startswith("ENST")], name="tx_id_versioned")
    bambu = pd.Series([s for s in uniq if s.startswith("BambuTx")])
    print(f"  ENST: {len(enst)}  BambuTx: {len(bambu)}")
    enst_str_df = pd.DataFrame({"tx_id_versioned": enst})
    enst_str_df["tx_id_stripped"] = enst_str_df["tx_id_versioned"].str.split(".").str[0]
    merged = enst_str_df.merge(ref, on="tx_id_stripped", how="left")
    n_mane = merged["mane_status"].fillna("").ne("").sum()
    n_appris = merged["appris_label"].fillna("").ne("").sum()
    n_dominant = merged["is_gene_dominant"].fillna(False).sum()
    n_alt_functional = merged["is_alt_functional"].fillna(False).sum()
    print(f"  covered by MANE: {n_mane} ({n_mane/len(enst):.1%})")
    print(f"  covered by APPRIS: {n_appris} ({n_appris/len(enst):.1%})")
    print(f"  gene-dominant (bambu): {n_dominant}")
    print(f"  is_alt_functional (final flag): {n_alt_functional}")
    print(f"  cf. previous UniProt+BISECT+TARGET total: 51+13+8 = 72 references")

    # Stringency tiers for downstream Fisher exact
    apr_alt = merged["appris_label"].fillna("").str.startswith("ALTERNATIVE")
    non_mane = ~merged["mane_status"].fillna("").isin(["MANE Select", "MANE Plus Clinical"])
    dom = merged["is_gene_dominant"].fillna(False)
    print("\n  --- stringency tiers within pilot (unique ENST) ---")
    print(f"  Tier A  APPRIS ALTERNATIVE (curated):        {apr_alt.sum()}")
    print(f"  Tier B  non-MANE + gene-dominant (expr):     {(non_mane & dom).sum()}")
    print(f"  Tier C  intersection A ∩ B (highest conf.):  {(apr_alt & non_mane & dom).sum()}")
    print(f"  Tier U  union A ∪ B (is_alt_functional):     {(apr_alt | (non_mane & dom)).sum()}")


def main():
    if not MANE_GZ.exists():
        raise SystemExit(f"missing MANE file: {MANE_GZ}")
    if not BAMBU_COUNTS.exists():
        raise SystemExit(f"missing bambu counts: {BAMBU_COUNTS}")

    download_appris(force="--force" in sys.argv)

    mane = load_mane()
    appris = load_appris()
    bambu = load_bambu_dominance()

    # merge on stripped tx id (MANE + APPRIS are ENST-based; Bambu is
    # already versioned ENST or BambuTx*)
    # Only ENST transcripts can carry MANE/APPRIS labels; keep those.
    ref = bambu.merge(mane[["tx_id_stripped", "mane_status", "symbol"]],
                      on="tx_id_stripped", how="left")
    ref = ref.merge(appris, on="tx_id_stripped", how="left")
    ref["mane_status"] = ref["mane_status"].fillna("")
    ref["appris_label"] = ref["appris_label"].fillna("")

    ref["is_alt_functional"] = (
        ref["appris_label"].str.startswith("ALTERNATIVE") |
        (~ref["mane_status"].isin(["MANE Select", "MANE Plus Clinical"])
         & ref["is_gene_dominant"]
         & ref["tx_id_stripped"].str.startswith("ENST"))
    )

    ref = ref[[
        "tx_id_versioned", "tx_id_stripped", "gene_id_bambu",
        "symbol", "mane_status", "appris_label",
        "max_muscle_count", "is_gene_dominant", "is_alt_functional",
    ]]
    ref.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\n[write] {OUT_TSV}  rows={len(ref)}")

    coverage_report(ref)


if __name__ == "__main__":
    main()
