"""Tests for ConfidenceScorer."""

from __future__ import annotations

import pytest

from src.engine.confidence_scorer import ConfidenceScorer, _clamp
from src.models.signal import TrendDirection, TrendResult


def _make_trend(
    direction: TrendDirection = TrendDirection.UP,
    strength: float = 0.8,
) -> TrendResult:
    return TrendResult(
        direction=direction,
        strength=strength,
        momentum=0.5,
        green_count=4,
        red_count=1,
        cumulative_move_pct=0.5,
    )


class TestConfidenceScorer:
    """Tests for ConfidenceScorer.score()."""

    def test_perfect_scores(self, config_loader) -> None:
        scorer = ConfidenceScorer(config_loader)
        trend = _make_trend(strength=1.0)
        result = scorer.score(trend, 1.0, 1.0, 1.0)
        assert result.overall == pytest.approx(1.0)

    def test_zero_scores(self, config_loader) -> None:
        scorer = ConfidenceScorer(config_loader)
        trend = _make_trend(strength=0.0)
        result = scorer.score(trend, 0.0, 0.0, 0.0)
        assert result.overall == pytest.approx(0.0)

    def test_partial_scores(self, config_loader) -> None:
        scorer = ConfidenceScorer(config_loader)
        trend = _make_trend(strength=0.8)
        result = scorer.score(trend, 0.9, 0.7, 0.5)
        # 0.35*0.8 + 0.25*0.9 + 0.20*0.7 + 0.20*0.5
        expected = 0.35 * 0.8 + 0.25 * 0.9 + 0.20 * 0.7 + 0.20 * 0.5
        assert result.overall == pytest.approx(expected)

    def test_overall_bounded_zero_to_one(self, config_loader) -> None:
        scorer = ConfidenceScorer(config_loader)
        trend = _make_trend(strength=1.0)
        result = scorer.score(trend, 1.0, 1.0, 5.0)  # exceedance > 1
        assert 0.0 <= result.overall <= 1.0

    def test_individual_scores_stored(self, config_loader) -> None:
        scorer = ConfidenceScorer(config_loader)
        trend = _make_trend(strength=0.7)
        result = scorer.score(trend, 0.8, 0.6, 0.4)
        assert result.trend_strength == pytest.approx(0.7)
        assert result.book_normality == pytest.approx(0.8)
        assert result.liquidity_quality == pytest.approx(0.6)
        assert result.threshold_exceedance == pytest.approx(0.4)

    def test_negative_inputs_clamped(self, config_loader) -> None:
        scorer = ConfidenceScorer(config_loader)
        trend = _make_trend(strength=0.5)
        result = scorer.score(trend, -0.5, -1.0, -2.0)
        assert result.book_normality == 0.0
        assert result.liquidity_quality == 0.0
        assert result.threshold_exceedance == 0.0
        assert result.overall >= 0.0

    def test_meets_minimum(self, config_loader) -> None:
        scorer = ConfidenceScorer(config_loader)
        trend = _make_trend(strength=1.0)
        result = scorer.score(trend, 1.0, 1.0, 1.0)
        assert result.meets_minimum(0.6) is True
        assert result.meets_minimum(1.1) is False


class TestClamp:
    """Tests for _clamp helper."""

    def test_clamp_within_range(self) -> None:
        assert _clamp(0.5) == 0.5

    def test_clamp_below_range(self) -> None:
        assert _clamp(-1.0) == 0.0

    def test_clamp_above_range(self) -> None:
        assert _clamp(2.0) == 1.0

    def test_clamp_at_boundaries(self) -> None:
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0
