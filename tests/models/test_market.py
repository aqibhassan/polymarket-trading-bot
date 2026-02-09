"""Tests for market data models."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.models.market import (
    Candle,
    CandleDirection,
    MarketState,
    OrderBookSnapshot,
    Position,
    Side,
)


class TestCandle:
    def test_green_candle(self, sample_candle: Candle) -> None:
        assert sample_candle.direction == CandleDirection.GREEN
        assert sample_candle.close > sample_candle.open

    def test_red_candle(self) -> None:
        candle = Candle(
            exchange="binance",
            symbol="BTCUSDT",
            open=Decimal("50300"),
            high=Decimal("50500"),
            low=Decimal("49800"),
            close=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime(2024, 1, 1),
        )
        assert candle.direction == CandleDirection.RED

    def test_neutral_candle(self) -> None:
        candle = Candle(
            exchange="binance",
            symbol="BTCUSDT",
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime(2024, 1, 1),
        )
        assert candle.direction == CandleDirection.NEUTRAL

    def test_body_size(self, sample_candle: Candle) -> None:
        expected = abs(sample_candle.close - sample_candle.open)
        assert sample_candle.body_size == expected

    def test_range_size(self, sample_candle: Candle) -> None:
        expected = sample_candle.high - sample_candle.low
        assert sample_candle.range_size == expected

    def test_high_less_than_low_raises(self) -> None:
        with pytest.raises(ValueError, match="high must be >= low"):
            Candle(
                exchange="binance",
                symbol="BTCUSDT",
                open=Decimal("50000"),
                high=Decimal("49000"),
                low=Decimal("50000"),
                close=Decimal("50000"),
                volume=Decimal("100"),
                timestamp=datetime(2024, 1, 1),
            )

    def test_candle_is_frozen(self, sample_candle: Candle) -> None:
        with pytest.raises(ValidationError):
            sample_candle.close = Decimal("99999")  # type: ignore[misc]


class TestOrderBookSnapshot:
    def test_total_depths(self, sample_orderbook: OrderBookSnapshot) -> None:
        assert sample_orderbook.total_bid_depth == Decimal("450")
        assert sample_orderbook.total_ask_depth == Decimal("450")

    def test_best_bid_ask(self, sample_orderbook: OrderBookSnapshot) -> None:
        assert sample_orderbook.best_bid == Decimal("0.55")
        assert sample_orderbook.best_ask == Decimal("0.57")

    def test_spread(self, sample_orderbook: OrderBookSnapshot) -> None:
        assert sample_orderbook.spread == Decimal("0.02")

    def test_empty_book(self) -> None:
        book = OrderBookSnapshot(bids=[], asks=[], timestamp=datetime(2024, 1, 1))
        assert book.total_bid_depth == Decimal("0")
        assert book.total_ask_depth == Decimal("0")
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.spread is None


class TestMarketState:
    def test_dominant_side_yes(self, sample_market_state: MarketState) -> None:
        assert sample_market_state.dominant_side == Side.YES
        assert sample_market_state.dominant_price == Decimal("0.62")

    def test_dominant_side_no(self) -> None:
        state = MarketState(
            market_id="test",
            yes_price=Decimal("0.35"),
            no_price=Decimal("0.65"),
            time_remaining_seconds=600,
        )
        assert state.dominant_side == Side.NO
        assert state.dominant_price == Decimal("0.65")

    def test_minutes_elapsed(self) -> None:
        state = MarketState(
            market_id="test",
            yes_price=Decimal("0.50"),
            no_price=Decimal("0.50"),
            time_remaining_seconds=600,  # 10 min remaining → 5 min elapsed
        )
        assert state.minutes_elapsed == pytest.approx(5.0)

    def test_full_time_remaining(self) -> None:
        state = MarketState(
            market_id="test",
            yes_price=Decimal("0.50"),
            no_price=Decimal("0.50"),
            time_remaining_seconds=900,  # full 15 min
        )
        assert state.minutes_elapsed == pytest.approx(0.0)


class TestPosition:
    def test_open_position(self, sample_position: Position) -> None:
        assert sample_position.is_open is True
        assert sample_position.realized_pnl() is None

    def test_closed_position(self, sample_position: Position) -> None:
        closed = sample_position.model_copy(
            update={
                "exit_price": Decimal("0.50"),
                "exit_time": datetime(2024, 1, 1, 12, 5, 0),
                "exit_reason": "PROFIT_TARGET",
            }
        )
        assert closed.is_open is False
        pnl = closed.realized_pnl()
        assert pnl is not None
        assert pnl == Decimal("10.00")  # (0.50 - 0.40) * 100

    def test_pnl_pct(self, sample_position: Position) -> None:
        closed = sample_position.model_copy(
            update={"exit_price": Decimal("0.50")}
        )
        pct = closed.pnl_pct()
        assert pct is not None
        assert pct == Decimal("0.25")  # 25% gain

    def test_loss_position(self, sample_position: Position) -> None:
        closed = sample_position.model_copy(
            update={"exit_price": Decimal("0.36")}
        )
        pnl = closed.realized_pnl()
        assert pnl is not None
        assert pnl == Decimal("-4.00")  # (0.36 - 0.40) * 100
