"""Tests for MarketSpeedTracker."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.engine.market_speed import MarketSpeedTracker


class TestMarketSpeedTracker:
    """Tests for MarketSpeedTracker."""

    def test_no_data_returns_zero(self) -> None:
        tracker = MarketSpeedTracker()
        assert tracker.correction_speed() == 0.0
        assert tracker.is_fast_market() is False

    def test_single_price_returns_zero(self) -> None:
        tracker = MarketSpeedTracker()
        tracker.add_price(Decimal("0.50"), datetime(2024, 1, 1, 12, 0, 0))
        assert tracker.correction_speed() == 0.0

    def test_fast_price_changes(self) -> None:
        """0.10 change in 1 second = 0.10 cents/sec."""
        tracker = MarketSpeedTracker()
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        tracker.add_price(Decimal("0.50"), t0)
        tracker.add_price(Decimal("0.60"), t0 + timedelta(seconds=1))
        speed = tracker.correction_speed()
        assert speed == pytest.approx(0.10, abs=1e-6)
        assert tracker.is_fast_market(threshold=0.05) is True

    def test_slow_price_changes(self) -> None:
        """0.01 change in 10 seconds = 0.001 cents/sec."""
        tracker = MarketSpeedTracker()
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        tracker.add_price(Decimal("0.50"), t0)
        tracker.add_price(Decimal("0.51"), t0 + timedelta(seconds=10))
        speed = tracker.correction_speed()
        assert speed == pytest.approx(0.001, abs=1e-6)
        assert tracker.is_fast_market(threshold=0.01) is False

    def test_multiple_price_points(self) -> None:
        tracker = MarketSpeedTracker()
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        tracker.add_price(Decimal("0.50"), t0)
        tracker.add_price(Decimal("0.55"), t0 + timedelta(seconds=5))
        tracker.add_price(Decimal("0.52"), t0 + timedelta(seconds=10))
        # total change = |0.05| + |0.03| = 0.08, total time = 10s
        speed = tracker.correction_speed()
        assert speed == pytest.approx(0.008, abs=1e-6)

    def test_max_history_respected(self) -> None:
        tracker = MarketSpeedTracker(max_history=3)
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(5):
            tracker.add_price(Decimal(str(0.50 + i * 0.01)), t0 + timedelta(seconds=i))
        # Only last 3 prices kept
        speed = tracker.correction_speed()
        assert speed > 0.0

    def test_is_fast_market_default_threshold(self) -> None:
        tracker = MarketSpeedTracker()
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        # Very fast: 1.0 change in 1 second
        tracker.add_price(Decimal("0.50"), t0)
        tracker.add_price(Decimal("1.50"), t0 + timedelta(seconds=1))
        assert tracker.is_fast_market() is True

    def test_is_not_fast_market(self) -> None:
        tracker = MarketSpeedTracker()
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        tracker.add_price(Decimal("0.50"), t0)
        tracker.add_price(Decimal("0.50001"), t0 + timedelta(seconds=100))
        assert tracker.is_fast_market() is False

    def test_same_timestamp_no_crash(self) -> None:
        """Two prices at the same time should not divide by zero."""
        tracker = MarketSpeedTracker()
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        tracker.add_price(Decimal("0.50"), t0)
        tracker.add_price(Decimal("0.60"), t0)
        speed = tracker.correction_speed()
        assert speed == 0.0
