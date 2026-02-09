"""Tests for LiquidityFilter."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from src.engine.liquidity_filter import LiquidityFilter
from src.models.market import OrderBookLevel, OrderBookSnapshot


class TestLiquidityFilter:
    """Tests for LiquidityFilter.check()."""

    def test_passes_with_good_liquidity(self, config_loader, sample_orderbook) -> None:
        lf = LiquidityFilter(config_loader)
        result = lf.check(sample_orderbook, hourly_volume=Decimal("500"))
        assert result.passed is True
        assert result.quality > 0.0
        assert result.reason == ""

    def test_fails_low_volume(self, config_loader, sample_orderbook) -> None:
        lf = LiquidityFilter(config_loader)
        result = lf.check(sample_orderbook, hourly_volume=Decimal("50"))
        assert result.passed is False
        assert "volume" in result.reason

    def test_fails_wide_spread(self, config_loader) -> None:
        """Spread of 0.10 > max 0.05."""
        orderbook = OrderBookSnapshot(
            bids=[OrderBookLevel(price=Decimal("0.50"), size=Decimal("100"))],
            asks=[OrderBookLevel(price=Decimal("0.60"), size=Decimal("100"))],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            market_id="test",
        )
        lf = LiquidityFilter(config_loader)
        result = lf.check(orderbook, hourly_volume=Decimal("500"))
        assert result.passed is False
        assert "spread" in result.reason

    def test_fails_no_spread(self, config_loader) -> None:
        """Empty book has no spread."""
        orderbook = OrderBookSnapshot(
            bids=[],
            asks=[],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            market_id="test",
        )
        lf = LiquidityFilter(config_loader)
        result = lf.check(orderbook, hourly_volume=Decimal("500"))
        assert result.passed is False
        assert "no spread" in result.reason

    def test_quality_increases_with_volume(self, config_loader, sample_orderbook) -> None:
        lf = LiquidityFilter(config_loader)
        r1 = lf.check(sample_orderbook, hourly_volume=Decimal("100"))
        r2 = lf.check(sample_orderbook, hourly_volume=Decimal("200"))
        assert r1.quality <= r2.quality

    def test_quality_bounded(self, config_loader, sample_orderbook) -> None:
        lf = LiquidityFilter(config_loader)
        result = lf.check(sample_orderbook, hourly_volume=Decimal("10000"))
        assert 0.0 <= result.quality <= 1.0

    def test_exact_minimum_volume_passes(self, config_loader, sample_orderbook) -> None:
        lf = LiquidityFilter(config_loader)
        result = lf.check(sample_orderbook, hourly_volume=Decimal("100"))
        assert result.passed is True

    def test_just_below_minimum_volume_fails(self, config_loader, sample_orderbook) -> None:
        lf = LiquidityFilter(config_loader)
        result = lf.check(sample_orderbook, hourly_volume=Decimal("99"))
        assert result.passed is False
