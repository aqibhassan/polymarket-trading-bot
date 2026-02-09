"""Tests for BookAnalyzer."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from src.engine.book_analyzer import BookAnalyzer
from src.models.market import OrderBookLevel, OrderBookSnapshot


class TestBookAnalyzer:
    """Tests for BookAnalyzer.analyze()."""

    def test_balanced_book(self, config_loader, sample_orderbook) -> None:
        analyzer = BookAnalyzer(config_loader)
        result = analyzer.analyze(sample_orderbook)
        assert result.is_spoofed is False
        # Both sides have equal depth (100+200+150 = 450 each)
        assert result.bid_depth == Decimal("450")
        assert result.ask_depth == Decimal("450")
        assert result.imbalance_ratio == pytest.approx(0.0)
        assert result.normality_score == pytest.approx(1.0)

    def test_heavily_bid_sided_spoofing(self, config_loader) -> None:
        """Bid side has 4x more depth than ask => spoofed."""
        orderbook = OrderBookSnapshot(
            bids=[
                OrderBookLevel(price=Decimal("0.55"), size=Decimal("400")),
                OrderBookLevel(price=Decimal("0.54"), size=Decimal("400")),
            ],
            asks=[
                OrderBookLevel(price=Decimal("0.57"), size=Decimal("100")),
                OrderBookLevel(price=Decimal("0.58"), size=Decimal("100")),
            ],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            market_id="test",
        )
        analyzer = BookAnalyzer(config_loader)
        result = analyzer.analyze(orderbook)
        assert result.is_spoofed is True
        assert result.bid_depth == Decimal("800")
        assert result.ask_depth == Decimal("200")

    def test_heavily_ask_sided_spoofing(self, config_loader) -> None:
        """Ask side has 4x more depth than bid => spoofed."""
        orderbook = OrderBookSnapshot(
            bids=[
                OrderBookLevel(price=Decimal("0.55"), size=Decimal("50")),
            ],
            asks=[
                OrderBookLevel(price=Decimal("0.57"), size=Decimal("100")),
                OrderBookLevel(price=Decimal("0.58"), size=Decimal("100")),
            ],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            market_id="test",
        )
        analyzer = BookAnalyzer(config_loader)
        result = analyzer.analyze(orderbook)
        # 200/50 = 4.0 >= 3.0 multiplier
        assert result.is_spoofed is True

    def test_empty_book(self, config_loader) -> None:
        orderbook = OrderBookSnapshot(
            bids=[],
            asks=[],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            market_id="test",
        )
        analyzer = BookAnalyzer(config_loader)
        result = analyzer.analyze(orderbook)
        assert result.is_spoofed is False
        assert result.normality_score == 0.0
        assert result.imbalance_ratio == 0.0
        assert result.bid_depth == Decimal("0")
        assert result.ask_depth == Decimal("0")

    def test_just_below_spoofing_threshold(self, config_loader) -> None:
        """2.9x ratio should NOT trigger spoofing (threshold is 3.0x)."""
        orderbook = OrderBookSnapshot(
            bids=[
                OrderBookLevel(price=Decimal("0.55"), size=Decimal("290")),
            ],
            asks=[
                OrderBookLevel(price=Decimal("0.57"), size=Decimal("100")),
            ],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            market_id="test",
        )
        analyzer = BookAnalyzer(config_loader)
        result = analyzer.analyze(orderbook)
        assert result.is_spoofed is False

    def test_normality_score_bounded(self, config_loader, sample_orderbook) -> None:
        analyzer = BookAnalyzer(config_loader)
        result = analyzer.analyze(sample_orderbook)
        assert 0.0 <= result.normality_score <= 1.0

    def test_imbalance_positive_when_bid_heavy(self, config_loader) -> None:
        orderbook = OrderBookSnapshot(
            bids=[OrderBookLevel(price=Decimal("0.55"), size=Decimal("200"))],
            asks=[OrderBookLevel(price=Decimal("0.57"), size=Decimal("100"))],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            market_id="test",
        )
        analyzer = BookAnalyzer(config_loader)
        result = analyzer.analyze(orderbook)
        assert result.imbalance_ratio > 0

    def test_imbalance_negative_when_ask_heavy(self, config_loader) -> None:
        orderbook = OrderBookSnapshot(
            bids=[OrderBookLevel(price=Decimal("0.55"), size=Decimal("100"))],
            asks=[OrderBookLevel(price=Decimal("0.57"), size=Decimal("200"))],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            market_id="test",
        )
        analyzer = BookAnalyzer(config_loader)
        result = analyzer.analyze(orderbook)
        assert result.imbalance_ratio < 0
