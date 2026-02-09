"""Tests for TickGranularityTracker."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.engine.tick_granularity import EarlyMoveResult, TickGranularityTracker


class TestTickGranularityTracker:
    """Tests for TickGranularityTracker early-move detection."""

    def test_no_ticks_returns_neutral(self, config_loader) -> None:
        tracker = TickGranularityTracker(config_loader)
        result = tracker.get_early_move()
        assert result.direction == "neutral"
        assert result.confidence == 0.0
        assert result.tick_count == 0

    def test_below_min_ticks_returns_neutral(self, config_loader) -> None:
        tracker = TickGranularityTracker(config_loader)
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(5):
            tracker.on_tick(Decimal("100.00"), t0 + timedelta(seconds=i))
        result = tracker.get_early_move()
        assert result.direction == "neutral"
        assert result.confidence == 0.0
        assert result.tick_count == 5

    def test_strong_upward_early_move(self, config_loader) -> None:
        tracker = TickGranularityTracker(config_loader)
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        # 20 ticks, each going up by 0.10 (strong sustained upward move)
        for i in range(20):
            price = Decimal("100.00") + Decimal(str(i * 0.10))
            tracker.on_tick(price, t0 + timedelta(seconds=i))
        result = tracker.get_early_move()
        assert result.direction == "green"
        assert result.confidence > 0.5
        assert result.sustain_ratio > 0.8
        assert result.early_half_return_pct > 0.0
        assert result.tick_count == 20

    def test_strong_downward_early_move(self, config_loader) -> None:
        tracker = TickGranularityTracker(config_loader)
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        # 20 ticks, each going down
        for i in range(20):
            price = Decimal("100.00") - Decimal(str(i * 0.10))
            tracker.on_tick(price, t0 + timedelta(seconds=i))
        result = tracker.get_early_move()
        assert result.direction == "red"
        assert result.confidence > 0.5
        assert result.sustain_ratio > 0.8
        assert result.early_half_return_pct < 0.0

    def test_mixed_ticks_low_sustain(self, config_loader) -> None:
        tracker = TickGranularityTracker(config_loader)
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        # Alternating up and down: low sustain ratio
        for i in range(20):
            if i % 2 == 0:
                price = Decimal("100.00") + Decimal("0.05")
            else:
                price = Decimal("100.00") - Decimal("0.05")
            tracker.on_tick(price, t0 + timedelta(seconds=i))
        result = tracker.get_early_move()
        # Sustain ratio should be low due to alternating direction
        assert result.sustain_ratio < 0.6
        assert result.confidence < 0.6

    def test_reset_clears_state(self, config_loader) -> None:
        tracker = TickGranularityTracker(config_loader)
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(20):
            tracker.on_tick(Decimal("100.00") + Decimal(str(i)), t0 + timedelta(seconds=i))
        result_before = tracker.get_early_move()
        assert result_before.tick_count == 20

        tracker.reset_minute()
        result_after = tracker.get_early_move()
        assert result_after.tick_count == 0
        assert result_after.direction == "neutral"
        assert result_after.confidence == 0.0

    def test_below_threshold_limits_confidence(self, config_loader) -> None:
        tracker = TickGranularityTracker(config_loader)
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        # Very small moves: below default threshold of 0.08%
        for i in range(20):
            # Total move of ~0.019% (well below 0.08% threshold)
            price = Decimal("100.000") + Decimal(str(i * 0.001))
            tracker.on_tick(price, t0 + timedelta(seconds=i))
        result = tracker.get_early_move()
        # Confidence should be limited by magnitude factor
        assert result.confidence < 0.5
        assert result.direction == "green"
        assert result.sustain_ratio > 0.8

    def test_ticks_outside_early_window_excluded(self, config_loader) -> None:
        tracker = TickGranularityTracker(config_loader)
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        # Only 5 ticks in the first 30s, rest after window
        for i in range(5):
            tracker.on_tick(Decimal("100.00") + Decimal(str(i)), t0 + timedelta(seconds=i))
        for i in range(20):
            tracker.on_tick(
                Decimal("200.00") + Decimal(str(i)),
                t0 + timedelta(seconds=35 + i),
            )
        # Early window only has 5 ticks, below min_early_ticks (15)
        result = tracker.get_early_move()
        assert result.direction == "neutral"
        assert result.confidence == 0.0


class TestEarlyMoveResult:
    """Tests for EarlyMoveResult model."""

    def test_frozen_model(self) -> None:
        result = EarlyMoveResult(
            early_half_return_pct=0.1,
            direction="green",
            sustain_ratio=0.8,
            confidence=0.7,
            tick_count=20,
        )
        with pytest.raises(Exception):  # noqa: B017
            result.direction = "red"  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        result = EarlyMoveResult(
            early_half_return_pct=0.5,
            direction="red",
            sustain_ratio=0.9,
            confidence=0.85,
            tick_count=30,
        )
        assert result.early_half_return_pct == pytest.approx(0.5)
        assert result.direction == "red"
        assert result.sustain_ratio == pytest.approx(0.9)
        assert result.confidence == pytest.approx(0.85)
        assert result.tick_count == 30
