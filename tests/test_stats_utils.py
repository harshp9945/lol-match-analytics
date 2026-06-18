"""
test_stats_utils.py
-------------------
Unit tests for the statistical utility functions.
Run with: python -m pytest tests/ -v

Having tests in an analyst portfolio is rare and signals
engineering maturity beyond just knowing statistics.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stats_utils import (
    cohens_h, interpret_cohens_h, cohens_d,
    test_win_rate, test_normality,
    bonferroni_correction, benjamini_hochberg,
    spearman_correlation, odds_ratio_ci
)
from config import ALPHA


# ── cohens_h ─────────────────────────────────────────────────────────────────

class TestCohensH:
    def test_identical_proportions(self):
        """Zero effect when both proportions are equal."""
        assert cohens_h(0.5, 0.5) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        """p1=0.8, p2=0.5 should produce a positive effect."""
        h = cohens_h(0.8, 0.5)
        assert h > 0

    def test_reversed_proportions(self):
        """Effect size should be equal magnitude but opposite sign when reversed."""
        h1 = cohens_h(0.7, 0.5)
        h2 = cohens_h(0.5, 0.7)
        assert abs(h1) == pytest.approx(abs(h2), rel=1e-6)

    def test_extreme_proportions(self):
        """Handles 0 and 1 proportions without error."""
        # arcsin(sqrt(0)) = 0, arcsin(sqrt(1)) = pi/2
        h = cohens_h(1.0, 0.0)
        assert np.isfinite(h)


class TestInterpretCohensH:
    def test_small_effect(self):
        assert interpret_cohens_h(0.1) == "small"

    def test_medium_effect(self):
        assert interpret_cohens_h(0.35) == "medium"

    def test_large_effect(self):
        assert interpret_cohens_h(0.6) == "large"

    def test_negative_value_treated_as_absolute(self):
        """Negative effect sizes should be interpreted on absolute value."""
        assert interpret_cohens_h(-0.6) == "large"


# ── cohens_d ─────────────────────────────────────────────────────────────────

class TestCohensD:
    def test_identical_groups(self):
        """Zero effect when groups are identical."""
        g1 = np.array([1.0, 2.0, 3.0, 4.0])
        g2 = np.array([1.0, 2.0, 3.0, 4.0])
        assert cohens_d(g1, g2) == pytest.approx(0.0, abs=1e-10)

    def test_direction(self):
        """Group 1 > Group 2 should produce positive d."""
        g1 = np.array([10.0, 11.0, 9.0, 10.5, 9.5])
        g2 = np.array([5.0, 6.0, 4.0, 5.5, 4.5])
        assert cohens_d(g1, g2) > 0

    def test_symmetry(self):
        """Reversing groups should negate d."""
        g1 = np.array([10.0, 11.0, 9.0])
        g2 = np.array([5.0, 6.0, 4.0])
        assert cohens_d(g1, g2) == pytest.approx(-cohens_d(g2, g1), rel=1e-6)


# ── test_win_rate ─────────────────────────────────────────────────────────────

class TestWinRate:
    def test_exactly_50_pct(self):
        """Exactly 50% win rate should NOT be significant."""
        result = test_win_rate(wins=500, total=1000, h0_rate=0.5)
        assert not result["significant"]
        assert result["observed_pct"] == pytest.approx(50.0)

    def test_highly_significant(self):
        """Very high win rate should be significant."""
        result = test_win_rate(wins=810, total=1000, h0_rate=0.5, label="Baron")
        assert result["significant"]
        assert result["observed_pct"] == pytest.approx(81.0)
        assert result["p_value"] < ALPHA

    def test_confidence_interval_contains_observed(self):
        """95% CI should always contain the observed rate."""
        result = test_win_rate(wins=650, total=1000, h0_rate=0.5)
        assert result["ci_low_pct"] <= result["observed_pct"] <= result["ci_high_pct"]

    def test_label_preserved(self):
        """Label should be passed through to result dict."""
        result = test_win_rate(wins=500, total=1000, h0_rate=0.5, label="Test Label")
        assert result["label"] == "Test Label"

    def test_result_has_required_keys(self):
        """Result must contain all expected keys."""
        result = test_win_rate(wins=500, total=1000, h0_rate=0.5)
        required = {"label", "wins", "total", "observed_pct", "p_value",
                    "significant", "effect_size", "effect_label", "ci_low_pct", "ci_high_pct"}
        assert required.issubset(set(result.keys()))

    def test_small_sample(self):
        """Should handle small samples without error."""
        result = test_win_rate(wins=8, total=10, h0_rate=0.5)
        assert 0 <= result["p_value"] <= 1

    def test_all_wins(self):
        """100% win rate should be significant."""
        result = test_win_rate(wins=100, total=100, h0_rate=0.5)
        assert result["significant"]
        assert result["observed_pct"] == 100.0


# ── bonferroni_correction ─────────────────────────────────────────────────────

class TestBonferroniCorrection:
    def test_single_significant(self):
        """p-value of 0.001 should remain significant with Bonferroni."""
        result = bonferroni_correction([0.001, 0.5, 0.5], alpha=0.05)
        # Corrected alpha = 0.05/3 = 0.0167
        assert result[0] is True
        assert result[1] is False

    def test_none_significant_after_correction(self):
        """Borderline p-values should lose significance after correction."""
        p_vals = [0.04] * 10  # Each barely significant at 0.05, but 10 tests
        result = bonferroni_correction(p_vals, alpha=0.05)
        # Corrected alpha = 0.05/10 = 0.005, so 0.04 > 0.005 = not significant
        assert all(r is False for r in result)

    def test_empty_list(self):
        """Empty input should return empty output."""
        assert bonferroni_correction([]) == []

    def test_length_preserved(self):
        """Output length must match input length."""
        p_vals = [0.01, 0.05, 0.1, 0.001]
        result = bonferroni_correction(p_vals)
        assert len(result) == len(p_vals)


# ── benjamini_hochberg ────────────────────────────────────────────────────────

class TestBenjaminiHochberg:
    def test_less_conservative_than_bonferroni(self):
        """BH should find at least as many significant results as Bonferroni."""
        p_vals = [0.001, 0.01, 0.02, 0.04, 0.5] * 10
        bh = benjamini_hochberg(p_vals, alpha=0.05)
        bf = bonferroni_correction(p_vals, alpha=0.05)
        bh_count = sum(bh)
        bf_count = sum(bf)
        assert bh_count >= bf_count

    def test_length_preserved(self):
        p_vals = [0.001, 0.01, 0.05, 0.1, 0.5]
        result = benjamini_hochberg(p_vals)
        assert len(result) == len(p_vals)

    def test_all_significant(self):
        """All very small p-values should all be significant."""
        p_vals = [1e-10] * 20
        result = benjamini_hochberg(p_vals)
        assert all(result)


# ── spearman_correlation ──────────────────────────────────────────────────────

class TestSpearmanCorrelation:
    def test_perfect_positive_correlation(self):
        """Perfectly correlated arrays should give rho=1."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = spearman_correlation(x, y)
        assert result["correlation"] == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        """Perfectly inverse arrays should give rho=-1."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 8, 6, 4, 2])
        result = spearman_correlation(x, y)
        assert result["correlation"] == pytest.approx(-1.0)

    def test_result_keys(self):
        """Result must contain all expected keys."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        result = spearman_correlation(x, y)
        required = {"label", "correlation", "p_value", "significant", "strength", "direction"}
        assert required.issubset(set(result.keys()))


# ── odds_ratio_ci ─────────────────────────────────────────────────────────────

class TestOddsRatioCi:
    def test_known_odds_ratio(self):
        """2x2 table with known OR."""
        # a=10, b=5, c=5, d=10 -> OR = (10*10)/(5*5) = 4.0
        result = odds_ratio_ci(10, 5, 5, 10)
        assert result["odds_ratio"] == pytest.approx(4.0, rel=1e-3)

    def test_ci_contains_or(self):
        """Confidence interval should contain the point estimate."""
        result = odds_ratio_ci(100, 50, 60, 90)
        assert result["ci_low"] <= result["odds_ratio"] <= result["ci_high"]

    def test_or_of_one_for_equal_groups(self):
        """Equal exposed and unexposed groups should give OR~1."""
        result = odds_ratio_ci(50, 50, 50, 50)
        assert result["odds_ratio"] == pytest.approx(1.0, rel=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
