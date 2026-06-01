"""NMD Risk Screening integration (Module E3).

Loads pre-computed NMD screening results (from SQANTI3 or custom PTC detection)
and adds NMD risk flags to classified isoform DataFrames.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd


def load_nmd_screening(json_path: str) -> Dict[str, dict]:
    """Load NMD screening results from JSON.

    Expected format:
    {
        "ENST000xxx": {"nmd_susceptible": true, "ptc_position": 123, ...},
        ...
    }
    Also accepts list format: [{"isoform_id": "...", "nmd_susceptible": true}, ...]
    """
    with open(json_path) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return {r['isoform_id']: r for r in raw if 'isoform_id' in r}
    if isinstance(raw, dict):
        # May be nested under a key
        for key in ['results', 'isoforms', 'data']:
            if key in raw and isinstance(raw[key], list):
                return {r['isoform_id']: r for r in raw[key] if 'isoform_id' in r}
        return raw
    return {}


def add_nmd_flags(
    classified_df: pd.DataFrame,
    nmd_data: Dict[str, dict],
    nmd_key: str = 'nmd_susceptible',
) -> pd.DataFrame:
    """Add nmd_risk boolean column to classified isoform DataFrame.

    Parameters
    ----------
    classified_df : output of classify_isoforms()
    nmd_data      : dict from load_nmd_screening()
    nmd_key       : field name within each isoform dict that holds the flag
    """
    df = classified_df.copy()
    df['nmd_risk'] = df['isoform_id'].map(
        lambda iso: bool(nmd_data.get(str(iso), {}).get(nmd_key, False))
    )
    return df


def filter_nmd_safe(
    classified_df: pd.DataFrame,
    nmd_data: Dict[str, dict],
    nmd_key: str = 'nmd_susceptible',
) -> pd.DataFrame:
    """Return only isoforms that are NOT NMD-susceptible."""
    flagged = add_nmd_flags(classified_df, nmd_data, nmd_key)
    return flagged[~flagged['nmd_risk']].drop(columns=['nmd_risk'])
