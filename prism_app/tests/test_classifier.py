"""Tests for 4-Scenario Classifier (D1).

Validates BISECT Tier A cases and core classification logic.
Run with: pytest prism_app/tests/test_classifier.py -v
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from prism_app.core.classifier import (
    classify_isoforms, scenario_summary, get_scenario_candidates, IsoformScenario,
)
from prism_app.core.go_utils import TISSUE_PRESETS


# ── Fixtures ──────────────────────────────────────────────────────────────────

DEMO_DIR = Path(__file__).parents[1] / 'data' / 'demo'
MUSCLE_GO = list(TISSUE_PRESETS['muscle'].keys())
BRAIN_EXT_GO = list(TISSUE_PRESETS['brain_extended'].keys())


@pytest.fixture(scope='module')
def muscle_data():
    sm    = np.load(DEMO_DIR / 'muscle_scores.npy')
    ids   = np.load(DEMO_DIR / 'muscle_ids.npy', allow_pickle=True)
    types = np.load(DEMO_DIR / 'muscle_types.npy', allow_pickle=True)
    genes = np.load(DEMO_DIR / 'muscle_gene_ids.npy', allow_pickle=True)
    return {'sm': sm, 'ids': ids, 'types': types, 'genes': genes}


@pytest.fixture(scope='module')
def brain_novel_data():
    sm  = np.load(DEMO_DIR / 'brain_novel_scores.npy')
    ids = np.load(DEMO_DIR / 'brain_novel_ids.npy', allow_pickle=True)
    n   = min(len(ids), sm.shape[0])
    return {'sm': sm[:n], 'ids': ids[:n]}


@pytest.fixture(scope='module')
def muscle_classified(muscle_data):
    return classify_isoforms(
        muscle_data['sm'], muscle_data['ids'], muscle_data['genes'],
        MUSCLE_GO, score_threshold=0.5,
    )


@pytest.fixture(scope='module')
def brain_novel_classified(brain_novel_data):
    go = BRAIN_EXT_GO[:brain_novel_data['sm'].shape[1]]
    return classify_isoforms(
        brain_novel_data['sm'], brain_novel_data['ids'], None,
        go, score_threshold=0.5,
    )


# ── Core logic tests ─────────────────────────────────────────────────────────

class TestScenarioEnum:
    def test_scenarios_have_correct_values(self):
        assert IsoformScenario.FUNCTIONAL_SWITCH  == 1
        assert IsoformScenario.EXPRESSION_SWITCH  == 2
        assert IsoformScenario.CONSTITUTIVE_NOVEL == 3
        assert IsoformScenario.BACKGROUND         == 4


class TestClassifyIsoforms:
    def test_returns_dataframe(self, muscle_classified):
        assert isinstance(muscle_classified, pd.DataFrame)

    def test_has_required_columns(self, muscle_classified):
        required = ['isoform_id', 'scenario', 'scenario_label',
                    'max_score', 'max_go', 'dtu_flag', 'novel_go_flag']
        for col in required:
            assert col in muscle_classified.columns, f"Missing column: {col}"

    def test_row_count_matches_input(self, muscle_data, muscle_classified):
        assert len(muscle_classified) == len(muscle_data['ids'])

    def test_scenarios_are_valid_integers(self, muscle_classified):
        assert set(muscle_classified['scenario'].unique()).issubset({1, 2, 3, 4})

    def test_max_score_in_range(self, muscle_classified):
        assert muscle_classified['max_score'].between(0, 1).all()

    def test_no_dtu_means_no_scenario_1_or_2(self, muscle_classified):
        # No DTU data provided → cannot have S1 or S2
        assert 1 not in muscle_classified['scenario'].values
        assert 2 not in muscle_classified['scenario'].values


class TestScenario3BrainNovel:
    """Key paper result: 541 novel brain isoforms → Scenario 3 (6.8%)."""

    def test_scenario3_count_is_541(self, brain_novel_classified):
        s3 = brain_novel_classified[brain_novel_classified['scenario'] == 3]
        assert len(s3) == 541, (
            f"Expected 541 Scenario 3 isoforms, got {len(s3)}. "
            "This number must match the manuscript (§3.12)."
        )

    def test_scenario3_percentage_is_6_8(self, brain_novel_classified):
        n_total = len(brain_novel_classified)
        n_s3    = (brain_novel_classified['scenario'] == 3).sum()
        pct     = round(100 * n_s3 / n_total, 1)
        assert pct == 6.8, f"Expected 6.8%, got {pct}%"

    def test_scenario3_isoforms_are_novel_types(self, brain_novel_classified):
        s3 = brain_novel_classified[brain_novel_classified['scenario'] == 3]
        # All IDs should match transcript... pattern (IsoQuant novel format)
        nic_nnic = s3['isoform_id'].str.contains('transcript', case=False, na=False)
        assert nic_nnic.mean() > 0.8, "Most S3 isoforms should be IsoQuant novel transcripts"


class TestScenario4Muscle:
    def test_muscle_background_is_majority(self, muscle_classified):
        n_s4  = (muscle_classified['scenario'] == 4).sum()
        total = len(muscle_classified)
        assert n_s4 / total > 0.8, "Background (S4) should be >80% without DTU"

    def test_muscle_has_some_scenario3(self, muscle_classified):
        n_s3 = (muscle_classified['scenario'] == 3).sum()
        assert n_s3 > 100, f"Expected >100 Constitutive Novel isoforms, got {n_s3}"


class TestDTUScenarios:
    """Test Scenario 1/2 classification when DTU data is provided."""

    def test_with_dtu_creates_scenario1(self, muscle_data):
        dtu = pd.DataFrame({
            'isoform_id': [str(muscle_data['ids'][0])],
            'pvalue':     [1e-6],
            'delta_IF':   [0.4],
        })
        clf = classify_isoforms(
            muscle_data['sm'], muscle_data['ids'], muscle_data['genes'],
            MUSCLE_GO, score_threshold=0.5, dtu_df=dtu,
        )
        dtu_iso = clf[clf['isoform_id'] == str(muscle_data['ids'][0])]
        assert not dtu_iso.empty
        # Should be S1 or S2 depending on whether novel GO predicted
        assert dtu_iso['scenario'].values[0] in (1, 2)

    def test_dtu_flag_set_for_significant_isoforms(self, muscle_data):
        dtu = pd.DataFrame({
            'isoform_id': [str(muscle_data['ids'][5])],
            'pvalue':     [1e-4],
        })
        clf = classify_isoforms(
            muscle_data['sm'], muscle_data['ids'], muscle_data['genes'],
            MUSCLE_GO, score_threshold=0.5, dtu_df=dtu,
            dtu_pval_threshold=0.05,
        )
        flagged = clf[clf['isoform_id'] == str(muscle_data['ids'][5])]
        assert bool(flagged['dtu_flag'].values[0]) is True


# ── Scenario summary tests ────────────────────────────────────────────────────

class TestScenarioSummary:
    def test_returns_dataframe_with_4_rows(self, muscle_classified):
        summ = scenario_summary(muscle_classified)
        assert len(summ) <= 4

    def test_pct_sums_to_100(self, muscle_classified):
        summ = scenario_summary(muscle_classified)
        assert abs(summ['pct'].sum() - 100.0) < 0.5

    def test_count_sums_to_total(self, muscle_classified):
        summ = scenario_summary(muscle_classified)
        assert summ['count'].sum() == len(muscle_classified)


class TestGetScenarioCandidates:
    def test_returns_only_requested_scenario(self, brain_novel_classified):
        cands = get_scenario_candidates(brain_novel_classified, scenario=3, min_score=0.5)
        assert (cands['scenario'] == 3).all()

    def test_respects_min_score_filter(self, brain_novel_classified):
        cands = get_scenario_candidates(brain_novel_classified, scenario=3, min_score=0.7)
        assert (cands['max_score'] >= 0.7).all()

    def test_top_n_limit(self, brain_novel_classified):
        cands = get_scenario_candidates(brain_novel_classified, scenario=3, top_n=10)
        assert len(cands) <= 10
