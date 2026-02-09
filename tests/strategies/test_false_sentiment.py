"""Tests for FalseSentimentStrategy."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from src.config.loader import ConfigLoader
from src.models.market import (
    Candle,
    MarketState,
    OrderBookLevel,
    OrderBookSnapshot,
    Position,
    Side,
)
from src.models.order import Fill
from src.models.signal import SignalType
from src.strategies.false_sentiment import FalseSentimentStrategy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config(tmp_path):
    """Create a minimal config for strategy tests."""
    default = tmp_path / "default.toml"
    default.write_text(
        """\
[strategy.false_sentiment]
entry_threshold_base = 0.59
threshold_time_scaling = 0.15
lookback_candles = 5
min_confidence = 0.6
max_hold_minutes = 7
force_exit_minute = 11
no_entry_after_minute = 8

[exit]
profit_target_pct = 0.05
trailing_stop_pct = 0.03
hard_stop_loss_pct = 0.04
max_hold_seconds = 420

[risk]
max_position_pct = 0.02
max_daily_drawdown_pct = 0.05
max_order_book_pct = 0.10

[orderbook]
heavy_book_multiplier = 3.0

[liquidity]
min_hourly_volume = 100
max_spread_cents = 0.05
"""
    )
    loader = ConfigLoader(config_dir=str(tmp_path), env="development")
    loader.load()
    return loader


@pytest.fixture()
def strategy(config):
    return FalseSentimentStrategy(config=config)


@pytest.fixture()
def uptrend_candles():
    """5 green candles => UP trend."""
    candles = []
    base = 60000
    for i in range(5):
        candles.append(
            Candle(
                exchange="binance",
                symbol="btcusdt",
                open=Decimal(str(base + i * 100)),
                high=Decimal(str(base + i * 100 + 150)),
                low=Decimal(str(base + i * 100 - 10)),
                close=Decimal(str(base + i * 100 + 100)),
                volume=Decimal("500"),
                timestamp=datetime(2025, 1, 1, i, 0),
            )
        )
    return candles


@pytest.fixture()
def downtrend_candles():
    """5 red candles => DOWN trend."""
    candles = []
    base = 60000
    for i in range(5):
        candles.append(
            Candle(
                exchange="binance",
                symbol="btcusdt",
                open=Decimal(str(base - i * 100 + 100)),
                high=Decimal(str(base - i * 100 + 150)),
                low=Decimal(str(base - i * 100 - 10)),
                close=Decimal(str(base - i * 100)),
                volume=Decimal("500"),
                timestamp=datetime(2025, 1, 1, i, 0),
            )
        )
    return candles


@pytest.fixture()
def market_state_early():
    """Market state at minute 3 (within entry window, price above threshold)."""
    return MarketState(
        market_id="btc-15m-test",
        yes_price=Decimal("0.65"),
        no_price=Decimal("0.35"),
        time_remaining_seconds=12 * 60,  # 12 min remaining => 3 min elapsed
    )


@pytest.fixture()
def market_state_late():
    """Market state at minute 10 (past no_entry_after_minute=8)."""
    return MarketState(
        market_id="btc-15m-test",
        yes_price=Decimal("0.65"),
        no_price=Decimal("0.35"),
        time_remaining_seconds=5 * 60,  # 5 min remaining => 10 min elapsed
    )


@pytest.fixture()
def orderbook():
    """Balanced order book with tight spread."""
    return OrderBookSnapshot(
        bids=[
            OrderBookLevel(price=Decimal("0.64"), size=Decimal("500")),
            OrderBookLevel(price=Decimal("0.63"), size=Decimal("300")),
        ],
        asks=[
            OrderBookLevel(price=Decimal("0.66"), size=Decimal("500")),
            OrderBookLevel(price=Decimal("0.67"), size=Decimal("300")),
        ],
        timestamp=datetime(2025, 1, 1, 0, 3),
        market_id="btc-15m-test",
    )


@pytest.fixture()
def context_up(uptrend_candles):
    return {"candles": uptrend_candles, "hourly_volume": 200}


@pytest.fixture()
def context_down(downtrend_candles):
    return {"candles": downtrend_candles, "hourly_volume": 200}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFalseSentimentSignals:
    def test_uptrend_generates_entry(self, strategy, market_state_early, orderbook, context_up):
        """Uptrend with good conditions should produce an ENTRY signal."""
        signals = strategy.generate_signals(market_state_early, orderbook, context_up)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 1
        sig = entry_signals[0]
        assert sig.strategy_id == "false_sentiment"
        assert sig.market_id == "btc-15m-test"
        assert sig.entry_price is not None
        assert sig.stop_loss is not None
        assert sig.take_profit is not None

    def test_downtrend_generates_entry(self, strategy, market_state_early, orderbook, context_down):
        """Downtrend still generates signal — strategy uses dominant price."""
        signals = strategy.generate_signals(market_state_early, orderbook, context_down)
        # May produce ENTRY or SKIP depending on confidence threshold
        assert len(signals) >= 1
        assert all(s.signal_type in (SignalType.ENTRY, SignalType.SKIP) for s in signals)

    def test_skip_low_confidence(self, strategy, market_state_early, orderbook):
        """No candles -> low confidence -> SKIP."""
        context = {"candles": [], "hourly_volume": 200}
        signals = strategy.generate_signals(market_state_early, orderbook, context)
        skip_signals = [s for s in signals if s.signal_type == SignalType.SKIP]
        assert len(skip_signals) >= 1

    def test_skip_past_entry_window(self, strategy, market_state_late, orderbook, context_up):
        """Past no_entry_after_minute should produce SKIP."""
        signals = strategy.generate_signals(market_state_late, orderbook, context_up)
        skip_signals = [s for s in signals if s.signal_type == SignalType.SKIP]
        assert len(skip_signals) >= 1
        assert any("no_entry_after" in s.metadata.get("skip_reason", "") for s in skip_signals)

    def test_exit_signal_for_open_position(self, strategy, config):
        """Open position past stop-loss should trigger EXIT."""
        position = Position(
            market_id="btc-15m-test",
            side=Side.YES,
            token_id="yes-token",
            entry_price=Decimal("0.65"),
            quantity=Decimal("100"),
            entry_time=datetime(2025, 1, 1, 0, 0),
            stop_loss=Decimal("0.61"),
            take_profit=Decimal("0.70"),
        )
        strategy.add_position(position)

        # Market price below stop-loss
        market_state = MarketState(
            market_id="btc-15m-test",
            yes_price=Decimal("0.60"),
            no_price=Decimal("0.40"),
            time_remaining_seconds=12 * 60,
        )
        orderbook = OrderBookSnapshot(
            bids=[OrderBookLevel(price=Decimal("0.59"), size=Decimal("500"))],
            asks=[OrderBookLevel(price=Decimal("0.61"), size=Decimal("500"))],
            timestamp=datetime(2025, 1, 1, 0, 3),
            market_id="btc-15m-test",
        )
        context = {"candles": [], "hourly_volume": 200}

        signals = strategy.generate_signals(market_state, orderbook, context)
        exit_signals = [s for s in signals if s.signal_type == SignalType.EXIT]
        assert len(exit_signals) == 1

    def test_on_fill_logs(self, strategy):
        """on_fill should not raise."""
        from uuid import uuid4

        fill = Fill(
            order_id=uuid4(),
            price=Decimal("0.65"),
            size=Decimal("100"),
        )
        position = Position(
            market_id="btc-15m-test",
            side=Side.YES,
            token_id="yes-token",
            entry_price=Decimal("0.65"),
            quantity=Decimal("100"),
            entry_time=datetime(2025, 1, 1, 0, 0),
            stop_loss=Decimal("0.61"),
            take_profit=Decimal("0.70"),
        )
        strategy.on_fill(fill, position)  # Should not raise

    def test_on_cancel_logs(self, strategy):
        """on_cancel should not raise."""
        strategy.on_cancel("order-123", "test_reason")  # Should not raise
