"""
M1: Sequence Extraction
Extract protein sequences for target transcript IDs from SQANTI3 FAA files.
"""
import os
import re
from pathlib import Path


def extract_sequences(transcript_ids: list[str], config: dict) -> dict[str, dict]:
    """
    Returns: {transcript_id: {"seq": str, "length": int, "source": str, "header": str}}
    Searches brain FAA first, then muscle FAA.
    """
    faa_sources = [
        ("brain_faa", config["paths"]["brain_faa"]),
        ("brain_pep", config["paths"]["brain_pep"]),
        ("muscle_faa", config["paths"]["muscle_faa"]),
    ]

    results = {}
    targets = {tid.split(".")[0]: tid for tid in transcript_ids}  # prefix → full ID

    for source_name, faa_path in faa_sources:
        if not os.path.exists(faa_path):
            continue
        if len(results) == len(transcript_ids):
            break
        current_id = None
        current_header = ""
        buffer = []
        with open(faa_path) as f:
            for line in f:
                if line.startswith(">"):
                    # Flush previous
                    if current_id and current_id not in results:
                        seq = "".join(buffer).rstrip("*")
                        results[current_id] = {
                            "seq": seq,
                            "length": len(seq),
                            "source": source_name,
                            "header": current_header,
                        }
                    buffer = []
                    raw_id = line[1:].split()[0]
                    current_header = line.strip()
                    # Match against targets
                    current_id = None
                    for full_tid in transcript_ids:
                        if full_tid in raw_id or raw_id in full_tid:
                            current_id = full_tid
                            break
                elif current_id:
                    buffer.append(line.strip())
            # Flush last
            if current_id and current_id not in results and buffer:
                seq = "".join(buffer).rstrip("*")
                results[current_id] = {
                    "seq": seq,
                    "length": len(seq),
                    "source": source_name,
                    "header": current_header,
                }

    missing = [tid for tid in transcript_ids if tid not in results]
    if missing:
        print(f"  [M1] WARNING: sequences not found for {missing}")

    return results


def write_fasta(seq_dict: dict, output_path: str) -> str:
    """Write sequences to FASTA file. Returns path."""
    with open(output_path, "w") as f:
        for tid, info in seq_dict.items():
            f.write(f">{tid}\n")
            seq = info["seq"]
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")
    return output_path
