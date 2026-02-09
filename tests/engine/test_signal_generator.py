"""Tests for FalseSentimentSignalGenerator."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from src.engine.signal_generator import FalseSentimentSignalGenerator
from src.models.market import (
    Candle,
    MarketState,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
)
from src.models.signal import SignalType


def _make_green_candles(n: int = 5) -> list[Candle]:
    candles = []
    base = Decimal("50000")
    start = datetime(2024, 1, 1, 12, 0, 0)
    for i in range(n):
        o = base + Decimal(str(i * 100))
        c = o + Decimal("80")
        candles.append(
            Candle(
                exchange="binance",
                symbol="BTCUSDT",
                open=o,
                high=c + Decimal("20"),
                low=o - Decimal("10"),
                close=c,
                volume=Decimal("100"),
                timestamp=start + timedelta(minutes=i * 15),
            )
        )
    return candles


def _make_orderbook(balanced: bool = True) -> OrderBookSnapshot:
    if balanced:
        return OrderBookSnapshot(
            bids=[
                OrderBookLevel(price=Decimal("0.55"), size=Decimal("100")),
                OrderBookLevel(price=Decimal("0.54"), size=Decimal("200")),
            ],
            asks=[
                OrderBookLevel(price=Decimal("0.57"), size=Decimal("100")),
                OrderBookLevel(price=Decimal("0.58"), size=Decimal("200")),
            ],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            market_id="test-market",
        )
    # Unbalanced (spoofed) book
    return OrderBookSnapshot(
        bids=[
            OrderBookLevel(price=Decimal("0.55"), size=Decimal("1000")),
        ],
        asks=[
            OrderBookLevel(price=Decimal("0.57"), size=Decimal("100")),
        ],
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        market_id="test-market",
    )


def _make_market_state(
    yes_price: str = "0.65",
    time_remaining: int = 600,
) -> MarketState:
    return MarketState(
        market_id="test-market",
        yes_price=Decimal(yes_price),
        no_price=Decimal("1") - Decimal(yes_price),
        yes_bid=Decimal(yes_price) - Decimal("0.01"),
        yes_ask=Decimal(yes_price) + Decimal("0.01"),
        time_remaining_seconds=time_remaining,
        question="Test question",
    )


class TestFalseSentimentSignalGenerator:
    """Tests for signal generation pipeline."""

    def test_generates_entry_signal(self, config_loader) -> None:
        gen = FalseSentimentSignalGenerator(config_loader)
        signal = gen.generate(
            candles=_make_green_candles(),
            market_state=_make_market_state(yes_price="0.65"),
            orderbook=_make_orderbook(balanced=True),
            hourly_volume=Decimal("500"),
        )
        assert signal.signal_type == SignalType.ENTRY
        assert signal.entry_price is not None
        assert signal.stop_loss is not None
        assert signal.take_profit is not None
        assert signal.strategy_id == "false_sentiment"
        assert signal.direction == Side.YES

    def test_skip_when_below_threshold(self, config_loader) -> None:
        """Price below threshold should produce SKIP."""
        gen = FalseSentimentSignalGenerator(config_loader)
        signal = gen.generate(
            candles=_make_green_candles(),
            market_state=_make_market_state(yes_price="0.50"),
            orderbook=_make_orderbook(balanced=True),
            hourly_volume=Decimal("500"),
        )
        assert signal.signal_type == SignalType.SKIP
        assert "threshold" in signal.metadata.get("skip_reason", "")

    def test_skip_after_no_entry_minute(self, config_loader) -> None:
        """After no_entry_after_minute (8), should produce SKIP."""
        gen = FalseSentimentSignalGenerator(config_loader)
        # time_remaining = 60 seconds => minutes_elapsed = 14.0 > 8
        signal = gen.generate(
            candles=_make_green_candles(),
            market_state=_make_market_state(yes_price="0.75", time_remaining=60),
            orderbook=_make_orderbook(balanced=True),
            hourly_volume=Decimal("500"),
        )
        assert signal.signal_type == SignalType.SKIP
        assert "time gate" in signal.metadata.get("skip_reason", "")

    def test_skip_low_liquidity(self, config_loader) -> None:
        """Low volume should produce SKIP."""
        gen = FalseSentimentSignalGenerator(config_loader)
        signal = gen.generate(
            candles=_make_green_candles(),
            market_state=_make_market_state(yes_price="0.65"),
            orderbook=_make_orderbook(balanced=True),
            hourly_volume=Decimal("10"),  # below min 100
        )
        assert signal.signal_type == SignalType.SKIP
        assert "liquidity" in signal.metadata.get("skip_reason", "")

    def test_entry_has_correct_stop_and_target(self, config_loader) -> None:
        gen = FalseSentimentSignalGenerator(config_loader)
        signal = gen.generate(
            candles=_make_green_candles(),
            market_state=_make_market_state(yes_price="0.65"),
            orderbook=_make_orderbook(balanced=True),
            hourly_volume=Decimal("500"),
        )
        if signal.signal_type == SignalType.ENTRY:
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            assert signal.stop_loss < signal.entry_price
            assert signal.take_profit > signal.entry_price

    def test_entry_metadata_populated(self, config_loader) -> None:
        gen = FalseSentimentSignalGenerator(config_loader)
        signal = gen.generate(
            candles=_make_green_candles(),
            market_state=_make_market_state(yes_price="0.65"),
            orderbook=_make_orderbook(balanced=True),
            hourly_volume=Decimal("500"),
        )
        if signal.signal_type == SignalType.ENTRY:
            assert "trend_direction" in signal.metadata
            assert "trend_strength" in signal.metadata
            assert "book_normality" in signal.metadata

    def test_confidence_above_minimum_for_entry(self, config_loader) -> None:
        gen = FalseSentimentSignalGenerator(config_loader)
        signal = gen.generate(
            candles=_make_green_candles(),
            market_state=_make_market_state(yes_price="0.65"),
            orderbook=_make_orderbook(balanced=True),
            hourly_volume=Decimal("500"),
        )
        if signal.signal_type == SignalType.ENTRY:
            assert signal.confidence.overall >= 0.6

    def test_skip_signal_has_no_prices(self, config_loader) -> None:
        gen = FalseSentimentSignalGenerator(config_loader)
        signal = gen.generate(
            candles=_make_green_candles(),
            market_state=_make_market_state(yes_price="0.50"),
            orderbook=_make_orderbook(balanced=True),
            hourly_volume=Decimal("500"),
        )
        assert signal.signal_type == SignalType.SKIP
        assert signal.entry_price is None
        assert signal.stop_loss is None
        assert signal.take_profit is None
