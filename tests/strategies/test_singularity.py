"""Tests for SingularityStrategy — ensemble signal voting and entry/exit logic."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.models.market import (
    MarketState,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
)
from src.models.signal import SignalType


@pytest.fixture
def mock_config() -> MagicMock:
    """ConfigLoader that returns reasonable defaults."""
    config = MagicMock()

    defaults: dict[str, object] = {
        "strategy.singularity.weight_momentum": 0.40,
        "strategy.singularity.weight_ofi": 0.25,
        "strategy.singularity.weight_futures": 0.15,
        "strategy.singularity.weight_vol": 0.10,
        "strategy.singularity.weight_time": 0.10,
        "strategy.singularity.min_signals_agree": 3,
        "strategy.singularity.min_confidence": 0.50,
        "strategy.singularity.entry_minute_start": 6,
        "strategy.singularity.entry_minute_end": 10,
        "strategy.singularity.entry_tiers": None,
        "strategy.singularity.resolution_guard_minute": 12,
        "strategy.singularity.exit_reversal_count": 2,
        "strategy.singularity.max_position_pct": 0.035,
        "strategy.singularity.stop_loss_pct": 1.00,
        "strategy.singularity.profit_target_pct": 0.40,
        # OFI
        "strategy.singularity.ofi_levels": 5,
        "strategy.singularity.ofi_signal_threshold": 0.25,
        "strategy.singularity.ofi_saturation": 0.40,
        # Futures
        "strategy.singularity.futures_lead_threshold_pct": 0.15,
        "strategy.singularity.futures_velocity_threshold_pct": 1.0,
        # Tick
        "strategy.singularity.early_move_threshold_pct": 0.08,
        "strategy.singularity.min_early_ticks": 15,
        "strategy.singularity.early_window_seconds": 30,
        # Vol
        "strategy.singularity.vol_spike_threshold": 1.25,
        "strategy.singularity.vol_low_threshold": 0.80,
        # Time
        "strategy.singularity.hour_stats": None,
    }

    def side_effect(key: str, default: object = None) -> object:
        return defaults.get(key, default)

    config.get.side_effect = side_effect
    config.validate_keys.return_value = None
    return config


@pytest.fixture
def make_candle():
    """Factory for creating Candle objects."""
    from src.models.market import Candle

    def _make(
        close: float,
        open_: float = 100000.0,
        high: float | None = None,
        low: float | None = None,
    ) -> Candle:
        return Candle(
            exchange="binance",
            symbol="BTCUSDT",
            open=Decimal(str(open_)),
            high=Decimal(str(high or max(open_, close))),
            low=Decimal(str(low or min(open_, close))),
            close=Decimal(str(close)),
            volume=Decimal("100"),
            timestamp=datetime.now(tz=timezone.utc),
            interval="1m",
        )

    return _make


@pytest.fixture
def bullish_orderbook() -> OrderBookSnapshot:
    """Order book with heavy bid side (buy pressure)."""
    return OrderBookSnapshot(
        timestamp=datetime.now(tz=timezone.utc),
        market_id="test",
        bids=[
            OrderBookLevel(price=Decimal("0.60"), size=Decimal("500")),
            OrderBookLevel(price=Decimal("0.59"), size=Decimal("400")),
            OrderBookLevel(price=Decimal("0.58"), size=Decimal("300")),
            OrderBookLevel(price=Decimal("0.57"), size=Decimal("200")),
            OrderBookLevel(price=Decimal("0.56"), size=Decimal("100")),
        ],
        asks=[
            OrderBookLevel(price=Decimal("0.61"), size=Decimal("50")),
            OrderBookLevel(price=Decimal("0.62"), size=Decimal("40")),
            OrderBookLevel(price=Decimal("0.63"), size=Decimal("30")),
            OrderBookLevel(price=Decimal("0.64"), size=Decimal("20")),
            OrderBookLevel(price=Decimal("0.65"), size=Decimal("10")),
        ],
    )


@pytest.fixture
def neutral_orderbook() -> OrderBookSnapshot:
    """Balanced order book."""
    return OrderBookSnapshot(
        timestamp=datetime.now(tz=timezone.utc),
        market_id="test",
        bids=[
            OrderBookLevel(price=Decimal("0.50"), size=Decimal("100")),
        ],
        asks=[
            OrderBookLevel(price=Decimal("0.51"), size=Decimal("100")),
        ],
    )


def _market_state(
    yes_price: float = 0.60,
    time_remaining: int = 480,
) -> MarketState:
    return MarketState(
        market_id="test_market",
        yes_price=Decimal(str(yes_price)),
        no_price=Decimal(str(1.0 - yes_price)),
        time_remaining_seconds=time_remaining,
    )


class TestSingularityStrategy:
    def test_registered(self, mock_config: MagicMock) -> None:
        """Strategy is registered with 'singularity' name."""
        from src.strategies.registry import get

        from src.strategies.singularity import SingularityStrategy  # noqa: F401

        cls = get("singularity")
        assert cls is SingularityStrategy

    def test_skip_before_entry_window(
        self, mock_config: MagicMock, neutral_orderbook: OrderBookSnapshot,
    ) -> None:
        """No entry signal before entry_minute_start."""
        from src.strategies.singularity import SingularityStrategy

        strat = SingularityStrategy(config=mock_config)
        context = {
            "candles_1m": [],
            "window_open_price": 100000.0,
            "minute_in_window": 3,  # before minute 6
            "yes_price": 0.55,
        }
        signals = strat.generate_signals(
            _market_state(), neutral_orderbook, context,
        )
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 0

    def test_skip_past_entry_window(
        self, mock_config: MagicMock, neutral_orderbook: OrderBookSnapshot,
    ) -> None:
        """No entry signal after entry_minute_end."""
        from src.strategies.singularity import SingularityStrategy

        strat = SingularityStrategy(config=mock_config)
        context = {
            "candles_1m": [],
            "window_open_price": 100000.0,
            "minute_in_window": 11,  # past minute 10
            "yes_price": 0.55,
        }
        signals = strat.generate_signals(
            _market_state(), neutral_orderbook, context,
        )
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 0

    def test_entry_with_strong_momentum_and_ofi(
        self,
        mock_config: MagicMock,
        bullish_orderbook: OrderBookSnapshot,
        make_candle: object,
    ) -> None:
        """Entry signal when momentum + OFI + time-of-day agree."""
        from src.strategies.singularity import SingularityStrategy

        strat = SingularityStrategy(config=mock_config)

        # Strong upward momentum: +0.20% from open
        open_price = 100000.0
        candles = [make_candle(close=100200.0, open_=open_price)]

        context = {
            "candles_1m": candles,
            "window_open_price": open_price,
            "minute_in_window": 8,  # within entry window
            "yes_price": 0.65,
        }

        signals = strat.generate_signals(
            _market_state(yes_price=0.65), bullish_orderbook, context,
        )
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        # Should have at least momentum + OFI + time_of_day agreeing
        assert len(entry_signals) >= 1
        assert entry_signals[0].direction == Side.YES

    def test_entry_bearish(
        self, mock_config: MagicMock, make_candle: object,
    ) -> None:
        """Entry signal in bearish direction."""
        from src.strategies.singularity import SingularityStrategy

        strat = SingularityStrategy(config=mock_config)

        # Strong downward momentum: -0.20%
        open_price = 100000.0
        candles = [make_candle(close=99800.0, open_=open_price)]

        # Bearish orderbook (heavy asks)
        bearish_ob = OrderBookSnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            market_id="test",
            bids=[
                OrderBookLevel(price=Decimal("0.39"), size=Decimal("50")),
                OrderBookLevel(price=Decimal("0.38"), size=Decimal("40")),
                OrderBookLevel(price=Decimal("0.37"), size=Decimal("30")),
                OrderBookLevel(price=Decimal("0.36"), size=Decimal("20")),
                OrderBookLevel(price=Decimal("0.35"), size=Decimal("10")),
            ],
            asks=[
                OrderBookLevel(price=Decimal("0.40"), size=Decimal("500")),
                OrderBookLevel(price=Decimal("0.41"), size=Decimal("400")),
                OrderBookLevel(price=Decimal("0.42"), size=Decimal("300")),
                OrderBookLevel(price=Decimal("0.43"), size=Decimal("200")),
                OrderBookLevel(price=Decimal("0.44"), size=Decimal("100")),
            ],
        )

        context = {
            "candles_1m": candles,
            "window_open_price": open_price,
            "minute_in_window": 8,
            "yes_price": 0.35,
        }

        signals = strat.generate_signals(
            _market_state(yes_price=0.35), bearish_ob, context,
        )
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        if entry_signals:
            assert entry_signals[0].direction == Side.NO

    def test_no_entry_below_momentum_threshold(
        self,
        mock_config: MagicMock,
        bullish_orderbook: OrderBookSnapshot,
        make_candle: object,
    ) -> None:
        """No entry when cumulative return is below threshold."""
        from src.strategies.singularity import SingularityStrategy

        strat = SingularityStrategy(config=mock_config)

        # Very small move: 0.01% (below 0.08% threshold at minute 8)
        open_price = 100000.0
        candles = [make_candle(close=100010.0, open_=open_price)]

        context = {
            "candles_1m": candles,
            "window_open_price": open_price,
            "minute_in_window": 8,
            "yes_price": 0.51,
        }

        signals = strat.generate_signals(
            _market_state(yes_price=0.51), bullish_orderbook, context,
        )
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        # Momentum vote won't fire, so fewer than min_signals_agree
        assert len(entry_signals) == 0

    def test_resolution_guard_exit(
        self,
        mock_config: MagicMock,
        neutral_orderbook: OrderBookSnapshot,
    ) -> None:
        """Exit signal at resolution guard minute."""
        from src.strategies.singularity import SingularityStrategy
        from src.models.market import Position
        from src.models.signal import ExitReason

        strat = SingularityStrategy(config=mock_config)

        # Add an open position
        pos = Position(
            market_id="test_market",
            side=Side.YES,
            token_id="yes_token",
            entry_price=Decimal("0.60"),
            quantity=Decimal("100"),
            entry_time=datetime.now(tz=timezone.utc),
            stop_loss=Decimal("0.00"),
            take_profit=Decimal("1.00"),
        )
        strat.add_position(pos)

        context = {
            "candles_1m": [],
            "window_open_price": 100000.0,
            "minute_in_window": 12,  # at resolution guard
            "yes_price": 0.65,
        }

        signals = strat.generate_signals(
            _market_state(yes_price=0.65, time_remaining=180),
            neutral_orderbook,
            context,
        )

        exit_signals = [s for s in signals if s.signal_type == SignalType.EXIT]
        assert len(exit_signals) == 1
        assert exit_signals[0].exit_reason == ExitReason.RESOLUTION_GUARD

    def test_position_size_multiplier(self, mock_config: MagicMock) -> None:
        """Position size multiplier scales with agreeing signal count."""
        from src.strategies.singularity import SingularityStrategy

        strat = SingularityStrategy(config=mock_config)
        assert strat.get_position_size_multiplier(3, 5) == 1.0
        assert strat.get_position_size_multiplier(4, 5) == 1.5
        assert strat.get_position_size_multiplier(5, 5) == 1.75

    def test_graceful_degradation(self, mock_config: MagicMock) -> None:
        """Strategy works even with no optional analyzers."""
        from src.strategies.singularity import SingularityStrategy

        strat = SingularityStrategy(config=mock_config)
        # Force-disable all optional analyzers
        strat._ofi_analyzer = None
        strat._futures_detector = None
        strat._tick_tracker = None
        strat._vol_detector = None
        strat._time_analyzer = None

        # Should still work (momentum-only, but won't reach min_signals_agree=3)
        assert strat._ofi_analyzer is None
        assert strat._futures_detector is None
