"""Tests for DynamicThreshold."""

from __future__ import annotations

from decimal import Decimal

from src.engine.dynamic_threshold import DynamicThreshold


class TestDynamicThreshold:
    """Tests for DynamicThreshold.calculate() and should_enter()."""

    def test_threshold_at_minute_zero(self, config_loader) -> None:
        dt = DynamicThreshold(config_loader)
        result = dt.calculate(0.0)
        assert result == Decimal("0.59")

    def test_threshold_at_minute_7_5(self, config_loader) -> None:
        dt = DynamicThreshold(config_loader)
        result = dt.calculate(7.5)
        expected = Decimal("0.59") + Decimal("0.15") * Decimal("7.5") / Decimal("15.0")
        assert result == expected
        assert result == Decimal("0.665")

    def test_threshold_at_minute_15(self, config_loader) -> None:
        dt = DynamicThreshold(config_loader)
        result = dt.calculate(15.0)
        expected = Decimal("0.59") + Decimal("0.15") * Decimal("15.0") / Decimal("15.0")
        assert result == expected
        assert result == Decimal("0.74")

    def test_threshold_increases_with_time(self, config_loader) -> None:
        dt = DynamicThreshold(config_loader)
        t0 = dt.calculate(0.0)
        t5 = dt.calculate(5.0)
        t10 = dt.calculate(10.0)
        assert t0 < t5 < t10

    def test_should_enter_above_threshold(self, config_loader) -> None:
        dt = DynamicThreshold(config_loader)
        assert dt.should_enter(Decimal("0.60"), 0.0) is True

    def test_should_enter_at_threshold(self, config_loader) -> None:
        dt = DynamicThreshold(config_loader)
        assert dt.should_enter(Decimal("0.59"), 0.0) is True

    def test_should_not_enter_below_threshold(self, config_loader) -> None:
        dt = DynamicThreshold(config_loader)
        assert dt.should_enter(Decimal("0.58"), 0.0) is False

    def test_should_enter_respects_time(self, config_loader) -> None:
        dt = DynamicThreshold(config_loader)
        # At minute 10, threshold = 0.59 + 0.15 * 10/15 = 0.69
        assert dt.should_enter(Decimal("0.68"), 10.0) is False
        assert dt.should_enter(Decimal("0.70"), 10.0) is True
