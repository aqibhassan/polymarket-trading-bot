"""Tests for OI Delta Tracker."""

from __future__ import annotations

import time

import pytest

from src.engine.oi_delta_tracker import OIDeltaTracker


class TestOIDeltaTracker:
    def test_oi_delta_empty(self) -> None:
        """No data → delta = 0.0."""
        tracker = OIDeltaTracker()
        assert tracker.get_delta() == 0.0

    def test_oi_delta_rising(self) -> None:
        """OI increases → positive delta."""
        tracker = OIDeltaTracker()
        now = time.time()
        tracker._history.append((now - 60, 1000.0))
        tracker._history.append((now, 1010.0))
        delta = tracker.get_delta()
        assert delta > 0, f"Rising OI should give positive delta, got {delta}"
        assert abs(delta - 0.01) < 0.001  # 1% increase

    def test_oi_delta_falling(self) -> None:
        """OI decreases → negative delta."""
        tracker = OIDeltaTracker()
        now = time.time()
        tracker._history.append((now - 60, 1000.0))
        tracker._history.append((now, 990.0))
        delta = tracker.get_delta()
        assert delta < 0, f"Falling OI should give negative delta, got {delta}"

    def test_oi_direction_rising_with_up_price(self) -> None:
        """Rising OI + rising price → confirms YES."""
        tracker = OIDeltaTracker()
        now = time.time()
        tracker._history.append((now - 60, 1000.0))
        tracker._history.append((now, 1020.0))  # 2% rise
        direction = tracker.get_direction_with_price("YES")
        assert direction == "YES"

    def test_oi_direction_rising_with_down_price(self) -> None:
        """Rising OI + falling price → confirms NO (new shorts)."""
        tracker = OIDeltaTracker()
        now = time.time()
        tracker._history.append((now - 60, 1000.0))
        tracker._history.append((now, 1020.0))  # Rising OI
        direction = tracker.get_direction_with_price("NO")
        assert direction == "NO"

    def test_oi_direction_falling_neutral(self) -> None:
        """Falling OI → neutral (closing positions, not new conviction)."""
        tracker = OIDeltaTracker()
        now = time.time()
        tracker._history.append((now - 60, 1000.0))
        tracker._history.append((now, 990.0))  # Falling OI
        direction = tracker.get_direction_with_price("YES")
        assert direction == "neutral"

    def test_oi_direction_small_change_neutral(self) -> None:
        """OI change < 0.1% → neutral."""
        tracker = OIDeltaTracker()
        now = time.time()
        tracker._history.append((now - 60, 1000.0))
        tracker._history.append((now, 1000.5))  # 0.05% change
        direction = tracker.get_direction_with_price("YES")
        assert direction == "neutral"
