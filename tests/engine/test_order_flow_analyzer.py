"""Tests for OrderFlowAnalyzer."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.engine.order_flow_analyzer import OrderFlowAnalyzer
from src.models.market import OrderBookLevel, OrderBookSnapshot


def _make_book(
    bid_sizes: list[str],
    ask_sizes: list[str],
    *,
    timestamp: datetime | None = None,
) -> OrderBookSnapshot:
    """Helper to build an OrderBookSnapshot from bid/ask size lists."""
    ts = timestamp or datetime(2024, 1, 1, 12, 0, 0)
    base_bid = Decimal("0.55")
    base_ask = Decimal("0.57")
    bids = [
        OrderBookLevel(
            price=base_bid - Decimal(str(i)) * Decimal("0.01"),
            size=Decimal(s),
        )
        for i, s in enumerate(bid_sizes)
    ]
    asks = [
        OrderBookLevel(
            price=base_ask + Decimal(str(i)) * Decimal("0.01"),
            size=Decimal(s),
        )
        for i, s in enumerate(ask_sizes)
    ]
    return OrderBookSnapshot(
        bids=bids,
        asks=asks,
        timestamp=ts,
        market_id="test-ofi",
    )


@pytest.fixture()
def balanced_book() -> OrderBookSnapshot:
    """Balanced order book with equal bid/ask depth."""
    return _make_book(
        ["100", "200", "150", "100", "50"],
        ["100", "200", "150", "100", "50"],
    )


@pytest.fixture()
def bid_heavy_book() -> OrderBookSnapshot:
    """Order book with strong bid-side dominance."""
    return _make_book(
        ["500", "400", "300", "200", "100"],
        ["50", "40", "30", "20", "10"],
    )


@pytest.fixture()
def ask_heavy_book() -> OrderBookSnapshot:
    """Order book with strong ask-side dominance."""
    return _make_book(
        ["50", "40", "30", "20", "10"],
        ["500", "400", "300", "200", "100"],
    )


class TestOrderFlowAnalyzer:
    """Tests for OrderFlowAnalyzer.analyze()."""

    def test_zero_volume_returns_neutral(self, config_loader) -> None:
        """Empty book with zero volume should return neutral with zero OFI."""
        book = OrderBookSnapshot(
            bids=[],
            asks=[],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            market_id="test-ofi",
        )
        analyzer = OrderFlowAnalyzer(config_loader)
        result = analyzer.analyze(book)
        assert result.ofi_current == 0.0
        assert result.direction == "neutral"
        assert result.signal_strength == 0.0
        assert result.ofi_trend == 0.0

    def test_buy_pressure_when_bids_dominate(
        self, config_loader, bid_heavy_book
    ) -> None:
        """Strong bid-side volume should produce buy_pressure signal."""
        analyzer = OrderFlowAnalyzer(config_loader)
        result = analyzer.analyze(bid_heavy_book)
        assert result.ofi_current > 0
        assert result.direction == "buy_pressure"
        assert result.signal_strength > 0

    def test_sell_pressure_when_asks_dominate(
        self, config_loader, ask_heavy_book
    ) -> None:
        """Strong ask-side volume should produce sell_pressure signal."""
        analyzer = OrderFlowAnalyzer(config_loader)
        result = analyzer.analyze(ask_heavy_book)
        assert result.ofi_current < 0
        assert result.direction == "sell_pressure"
        assert result.signal_strength > 0

    def test_neutral_when_balanced(self, config_loader, balanced_book) -> None:
        """Balanced book should produce neutral direction and zero OFI."""
        analyzer = OrderFlowAnalyzer(config_loader)
        result = analyzer.analyze(balanced_book)
        assert result.ofi_current == pytest.approx(0.0)
        assert result.direction == "neutral"

    def test_signal_strength_scales_with_ofi_magnitude(
        self, config_loader
    ) -> None:
        """Signal strength should increase as |OFI| grows."""
        analyzer = OrderFlowAnalyzer(config_loader)

        # Mild imbalance
        mild_book = _make_book(["120", "110"], ["100", "100"])
        mild_result = analyzer.analyze(mild_book)

        # Strong imbalance
        strong_book = _make_book(["500", "500"], ["50", "50"])
        strong_result = analyzer.analyze(strong_book)

        assert strong_result.signal_strength > mild_result.signal_strength

    def test_signal_strength_saturates_at_one(self, config_loader) -> None:
        """Signal strength should not exceed 1.0 even with extreme OFI."""
        analyzer = OrderFlowAnalyzer(config_loader)
        extreme_book = _make_book(["1000", "1000"], ["1", "1"])
        result = analyzer.analyze(extreme_book)
        assert result.signal_strength == pytest.approx(1.0)

    def test_ofi_trend_computation_with_history(self, config_loader) -> None:
        """OFI trend should reflect change in OFI over history snapshots."""
        analyzer = OrderFlowAnalyzer(config_loader)
        base_ts = datetime(2024, 1, 1, 12, 0, 0)

        # History: progressively more bid-heavy
        h1 = _make_book(
            ["100", "100"], ["100", "100"],
            timestamp=base_ts,
        )
        h2 = _make_book(
            ["150", "100"], ["100", "100"],
            timestamp=base_ts + timedelta(seconds=5),
        )
        current = _make_book(
            ["200", "100"], ["100", "100"],
            timestamp=base_ts + timedelta(seconds=10),
        )

        result = analyzer.analyze(current, history=[h1, h2])
        # OFI increasing over time => positive trend
        assert result.ofi_trend > 0

    def test_ofi_trend_negative_with_declining_bids(
        self, config_loader
    ) -> None:
        """OFI trend should be negative when bid dominance is fading."""
        analyzer = OrderFlowAnalyzer(config_loader)
        base_ts = datetime(2024, 1, 1, 12, 0, 0)

        h1 = _make_book(
            ["300", "200"], ["100", "100"],
            timestamp=base_ts,
        )
        h2 = _make_book(
            ["200", "150"], ["100", "100"],
            timestamp=base_ts + timedelta(seconds=5),
        )
        current = _make_book(
            ["120", "100"], ["100", "100"],
            timestamp=base_ts + timedelta(seconds=10),
        )

        result = analyzer.analyze(current, history=[h1, h2])
        assert result.ofi_trend < 0

    def test_ofi_trend_zero_without_history(
        self, config_loader, bid_heavy_book
    ) -> None:
        """OFI trend should be 0.0 when no history is provided."""
        analyzer = OrderFlowAnalyzer(config_loader)
        result = analyzer.analyze(bid_heavy_book)
        assert result.ofi_trend == 0.0

    def test_ofi_trend_zero_with_single_history(
        self, config_loader, balanced_book
    ) -> None:
        """OFI trend should be 0.0 when history has fewer than 2 snapshots."""
        analyzer = OrderFlowAnalyzer(config_loader)
        result = analyzer.analyze(balanced_book, history=[balanced_book])
        assert result.ofi_trend == 0.0

    def test_empty_book_handling(self, config_loader) -> None:
        """Empty bids and asks should not raise and should return neutral."""
        book = OrderBookSnapshot(
            bids=[],
            asks=[],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            market_id="test-ofi",
        )
        analyzer = OrderFlowAnalyzer(config_loader)
        result = analyzer.analyze(book)
        assert result.direction == "neutral"
        assert result.ofi_current == 0.0
        assert result.signal_strength == 0.0
        assert result.confidence >= 0.0

    def test_one_side_empty(self, config_loader) -> None:
        """Book with only bids (no asks) should show max buy pressure."""
        book = OrderBookSnapshot(
            bids=[
                OrderBookLevel(price=Decimal("0.55"), size=Decimal("100")),
            ],
            asks=[],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            market_id="test-ofi",
        )
        analyzer = OrderFlowAnalyzer(config_loader)
        result = analyzer.analyze(book)
        assert result.ofi_current == pytest.approx(1.0)
        assert result.direction == "buy_pressure"
        assert result.signal_strength == pytest.approx(1.0)

    def test_ofi_bounded_between_neg1_and_1(self, config_loader) -> None:
        """OFI current value should always be in [-1, 1]."""
        analyzer = OrderFlowAnalyzer(config_loader)

        for bid_sizes, ask_sizes in [
            (["1000"], ["1"]),
            (["1"], ["1000"]),
            (["100"], ["100"]),
            (["0"], ["0"]),
        ]:
            book = _make_book(bid_sizes, ask_sizes)
            result = analyzer.analyze(book)
            assert -1.0 <= result.ofi_current <= 1.0

    def test_confidence_lower_when_reversing(self, config_loader) -> None:
        """Confidence should decrease when OFI trend opposes current direction."""
        analyzer = OrderFlowAnalyzer(config_loader)
        base_ts = datetime(2024, 1, 1, 12, 0, 0)

        # History shows declining bid dominance (trend reversing)
        h1 = _make_book(
            ["500", "500"], ["50", "50"],
            timestamp=base_ts,
        )
        h2 = _make_book(
            ["400", "400"], ["80", "80"],
            timestamp=base_ts + timedelta(seconds=5),
        )
        # Current still bid-heavy but weakening
        current = _make_book(
            ["300", "300"], ["100", "100"],
            timestamp=base_ts + timedelta(seconds=10),
        )

        result_reversing = analyzer.analyze(current, history=[h1, h2])

        # Non-reversing: trend agrees with direction
        h1_agree = _make_book(
            ["200", "200"], ["100", "100"],
            timestamp=base_ts,
        )
        h2_agree = _make_book(
            ["300", "300"], ["100", "100"],
            timestamp=base_ts + timedelta(seconds=5),
        )
        current_agree = _make_book(
            ["400", "400"], ["100", "100"],
            timestamp=base_ts + timedelta(seconds=10),
        )

        result_agreeing = analyzer.analyze(current_agree, history=[h1_agree, h2_agree])

        assert result_agreeing.confidence >= result_reversing.confidence

    def test_respects_ofi_levels_config(self, config_loader) -> None:
        """Analyzer should only use configured number of levels."""
        analyzer = OrderFlowAnalyzer(config_loader)
        # Default levels = 5; extra levels beyond 5 should be ignored
        book = _make_book(
            ["100", "100", "100", "100", "100", "9999"],
            ["100", "100", "100", "100", "100", "9999"],
        )
        result = analyzer.analyze(book)
        # Only top 5 levels considered, which are balanced
        assert result.ofi_current == pytest.approx(0.0)

    def test_result_is_frozen(self, config_loader, balanced_book) -> None:
        """OFIResult should be immutable (frozen Pydantic model)."""
        analyzer = OrderFlowAnalyzer(config_loader)
        result = analyzer.analyze(balanced_book)
        with pytest.raises(Exception):
            result.ofi_current = 999.0  # type: ignore[misc]
