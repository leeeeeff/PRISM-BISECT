"""Tests for Coverage Report (A1) and Novel Summary (A3)."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from prism_app.reports.coverage import generate_coverage_report, CoverageReport
from prism_app.reports.novel_summary import generate_novel_summary, NovelSummaryReport
from prism_app.core.go_utils import TISSUE_PRESETS

DEMO_DIR = Path(__file__).parents[1] / 'data' / 'demo'
MUSCLE_GO = list(TISSUE_PRESETS['muscle'].keys())


@pytest.fixture(scope='module')
def muscle_data():
    return {
        'sm':    np.load(DEMO_DIR / 'muscle_scores.npy'),
        'ids':   np.load(DEMO_DIR / 'muscle_ids.npy',   allow_pickle=True),
        'types': np.load(DEMO_DIR / 'muscle_types.npy', allow_pickle=True),
    }


@pytest.fixture(scope='module')
def muscle_coverage(muscle_data):
    return generate_coverage_report(
        muscle_data['sm'], muscle_data['ids'], muscle_data['types'],
        MUSCLE_GO, TISSUE_PRESETS['muscle'], score_threshold=0.5,
    )


class TestCoverageReport:
    def test_returns_coverage_report(self, muscle_coverage):
        assert isinstance(muscle_coverage, CoverageReport)

    def test_total_isoforms_matches_data(self, muscle_data, muscle_coverage):
        assert muscle_coverage.total_isoforms == len(muscle_data['ids'])

    def test_type_counts_sum_to_total(self, muscle_coverage):
        assert (muscle_coverage.n_known + muscle_coverage.n_nic + muscle_coverage.n_nnic
                == muscle_coverage.total_isoforms)

    def test_muscle_known_count(self, muscle_coverage):
        # ~35198 known ENST isoforms
        assert muscle_coverage.n_known == 35198

    def test_muscle_nnic_count(self, muscle_coverage):
        # 1550 BambuTx novel
        assert muscle_coverage.n_nnic == 1550

    def test_pct_with_any_high(self, muscle_coverage):
        # Should be around 11-12%
        assert 8.0 <= muscle_coverage.pct_with_any_high <= 15.0

    def test_per_go_has_18_entries(self, muscle_coverage):
        assert len(muscle_coverage.per_go) == 18

    def test_per_go_contains_required_fields(self, muscle_coverage):
        for entry in muscle_coverage.per_go:
            for field in ['go', 'name', 'n_high', 'mean_score']:
                assert field in entry, f"Missing field {field}"

    def test_score_stats_in_range(self, muscle_coverage):
        assert 0 <= muscle_coverage.score_mean <= 1
        assert 0 <= muscle_coverage.score_max  <= 1


class TestNovelSummary:
    def test_returns_novel_summary_report(self, muscle_data):
        rep = generate_novel_summary(
            muscle_data['sm'], muscle_data['ids'], muscle_data['types'],
            MUSCLE_GO, TISSUE_PRESETS['muscle'], score_threshold=0.5,
        )
        assert isinstance(rep, NovelSummaryReport)

    def test_novel_isoforms_count(self, muscle_data):
        rep = generate_novel_summary(
            muscle_data['sm'], muscle_data['ids'], muscle_data['types'],
            MUSCLE_GO, TISSUE_PRESETS['muscle'], score_threshold=0.5,
        )
        # 1550 BambuTx classified as nnic
        assert rep.total_novel == 1550

    def test_to_dataframe(self, muscle_data):
        rep = generate_novel_summary(
            muscle_data['sm'], muscle_data['ids'], muscle_data['types'],
            MUSCLE_GO, TISSUE_PRESETS['muscle'], score_threshold=0.5,
        )
        df = rep.to_dataframe()
        assert len(df) > 0
        for col in ['GO Term', 'Function', 'N novel (>thr)', 'Mean score']:
            assert col in df.columns
