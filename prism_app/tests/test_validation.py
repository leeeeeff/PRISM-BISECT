"""Tests for Known Annotation Validation (A2).

Key check: Macro AUPRC must reproduce 0.7022 (±0.005) for muscle data.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from prism_app.reports.validation import generate_validation_report
from prism_app.core.go_utils import TISSUE_PRESETS

DEMO_DIR = Path(__file__).parents[1] / 'data' / 'demo'
ANNOT_PATH = str(
    Path(__file__).parents[2]
    / 'hMuscle/data/raw_data/data/annotations/human_annotations_unified_bp.txt'
)


@pytest.fixture(scope='module')
def muscle_validation():
    sm    = np.load(DEMO_DIR / 'muscle_scores.npy')
    ids   = np.load(DEMO_DIR / 'muscle_ids.npy',   allow_pickle=True)
    genes = np.load(DEMO_DIR / 'muscle_gene_ids.npy', allow_pickle=True)
    go    = list(TISSUE_PRESETS['muscle'].keys())
    gnames = TISSUE_PRESETS['muscle']
    return generate_validation_report(
        sm, ids, go, gnames,
        annotation_path=ANNOT_PATH,
        gene_ids=genes,
        n_bootstrap=50,   # fewer reps for speed in CI
    )


class TestValidationReport:
    def test_report_is_not_none(self, muscle_validation):
        assert muscle_validation is not None, (
            "Validation report is None — annotation file may be missing or "
            "no ENSG→symbol mapping found."
        )

    def test_macro_auprc_matches_paper(self, muscle_validation):
        """Core reproducibility check: paper reports 0.7022."""
        if muscle_validation is None:
            pytest.skip("No annotation overlap")
        auprc = muscle_validation.macro_auprc
        assert abs(auprc - 0.7022) < 0.005, (
            f"Macro AUPRC {auprc:.4f} deviates from paper value 0.7022 by "
            f"{abs(auprc - 0.7022):.4f} (threshold: 0.005)"
        )

    def test_bootstrap_ci_contains_paper_value(self, muscle_validation):
        if muscle_validation is None:
            pytest.skip("No annotation overlap")
        lo, hi = muscle_validation.macro_auprc_ci
        assert lo <= 0.7022 <= hi, (
            f"Paper value 0.7022 not within 95% CI ({lo:.4f}, {hi:.4f})"
        )

    def test_per_go_has_entries(self, muscle_validation):
        if muscle_validation is None:
            pytest.skip("No annotation overlap")
        assert len(muscle_validation.per_go) >= 10

    def test_per_go_auprc_in_range(self, muscle_validation):
        if muscle_validation is None:
            pytest.skip("No annotation overlap")
        for entry in muscle_validation.per_go:
            assert 0 <= entry['auprc'] <= 1, f"AUPRC out of range for {entry['go']}"

    def test_isoforms_with_annotation(self, muscle_validation):
        if muscle_validation is None:
            pytest.skip("No annotation overlap")
        # Should find thousands with ENSG→symbol mapping
        assert muscle_validation.n_isoforms_with_annotation > 1000
