"""Pytest configuration for PRISM test suite."""
import sys
from pathlib import Path

# Ensure prism_app is importable from the repo root
sys.path.insert(0, str(Path(__file__).parents[2]))
