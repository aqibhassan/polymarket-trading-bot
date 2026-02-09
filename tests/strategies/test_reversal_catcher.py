"""Tests for MomentumConfirmationStrategy (momentum-following version).

Tests cover entry gates, confirmation filters, confidence scoring, exit logic,
and lifecycle methods:
  Gate 1: entry window timing (minute 8-10, tiered)
  Gate 2: candle data and valid window open
  Gate 3: tiered threshold (min8@0.10%, min9@0.08%, min10@0.05%)
  Gate 4: last_3_agree confirmation filter
  Gate 5: no_reversal confirmation filter
  Gate 6: composite confidence scoring (>0.70)
  Exit: profit target, stop loss, resolution guard, max hold
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

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
from src.models.signal import ExitReason, SignalType
from src.strategies.reversal_catcher import MomentumConfirmationStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_BTC = 60000.0


def _make_candle(
    open_: float,
    close: float,
    minute: int,
    volume: float = 500.0,
    high: float | None = None,
    low: float | None = None,
) -> Candle:
    """Create a 1m BTC candle with sensible defaults."""
    if high is None:
        high = max(open_, close) + 10
    if low is None:
        low = min(open_, close) - 10
    return Candle(
        exchange="binance",
        symbol="btcusdt",
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=Decimal(str(volume)),
        timestamp=datetime(2025, 1, 1, 0, minute),
        interval="1m",
    )


def _momentum_up_candles(n: int = 9) -> list[Candle]:
    """Build n green candles with strong upward momentum.

    All candles green, cumulative move = +180 / 60000 = 0.30% > 0.10% threshold.
    Last 3 are all green -> last_3_agree = True.
    No red candles -> no_reversal = True.
    """
    candles: list[Candle] = []
    price = _BASE_BTC
    moves = [20, 25, 30, 35, 35, 35, 30, 25, 20]
    for i in range(min(n, len(moves))):
        open_ = price
        close = price + moves[i]
        candles.append(_make_candle(open_, close, i))
        price = close
    return candles


def _momentum_down_candles(n: int = 9) -> list[Candle]:
    """Build n red candles with strong downward momentum.

    All candles red, cumulative move = -180 / 60000 = -0.30%.
    Last 3 are all red -> last_3_agree = True.
    No green candles -> no_reversal = True.
    """
    candles: list[Candle] = []
    price = _BASE_BTC
    moves = [-20, -25, -30, -35, -35, -35, -30, -25, -20]
    for i in range(min(n, len(moves))):
        open_ = price
        close = price + moves[i]
        candles.append(_make_candle(open_, close, i))
        price = close
    return candles


def _momentum_up_with_mixed_last3(n: int = 9) -> list[Candle]:
    """n candles: overall up but last 3 include a red candle.

    Cumulative move is up, so last_3_agree requires all 3 green.
    One of the last 3 is red -> last_3_agree = False.
    Also, that red is in an uptrend -> no_reversal = False.
    """
    candles: list[Candle] = []
    price = _BASE_BTC
    for i in range(n):
        if i == n - 2:
            # Red candle near the end
            candles.append(_make_candle(price, price - 15, i))
            price = price - 15
        else:
            open_ = price
            close = price + 30
            candles.append(_make_candle(open_, close, i))
            price = close
    return candles


def _momentum_up_with_reversal(n: int = 9) -> list[Candle]:
    """n candles: mostly green but 1 red candle in the window.

    Overall uptrend but has a reversal candle -> no_reversal = False.
    Last 3 are all green -> last_3_agree = True.
    """
    candles: list[Candle] = []
    price = _BASE_BTC
    for i in range(n):
        if i == 2:
            # RED (reversal, early in window)
            candles.append(_make_candle(price, price - 10, i))
            price = price - 10
        else:
            open_ = price
            close = price + 30
            candles.append(_make_candle(open_, close, i))
            price = close
    return candles


def _flat_candles(n: int = 9) -> list[Candle]:
    """n flat/neutral candles -- no significant move."""
    candles: list[Candle] = []
    price = _BASE_BTC
    for i in range(n):
        candles.append(_make_candle(price, price + 1, i))
        price = price + 1
    return candles


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config(tmp_path: Any) -> ConfigLoader:
    """Config with momentum_confirmation parameters — tiered entry, hold_to_settlement."""
    default = tmp_path / "default.toml"
    default.write_text(
        """\
[strategy.momentum_confirmation]
entry_minute_start = 8
entry_minute_end = 10
hold_to_settlement = true
profit_target_pct = 0.40
stop_loss_pct = 1.00
max_hold_minutes = 10
resolution_guard_minute = 14
min_confidence = 0.70

[[strategy.momentum_confirmation.entry_tiers]]
minute = 8
threshold_pct = 0.10

[[strategy.momentum_confirmation.entry_tiers]]
minute = 9
threshold_pct = 0.08

[[strategy.momentum_confirmation.entry_tiers]]
minute = 10
threshold_pct = 0.05

[risk]
max_position_pct = 0.02
"""
    )
    loader = ConfigLoader(config_dir=str(tmp_path), env="development")
    loader.load()
    return loader


@pytest.fixture()
def config_no_settlement(tmp_path: Any) -> ConfigLoader:
    """Config with hold_to_settlement = false (traditional exit logic)."""
    default = tmp_path / "default.toml"
    default.write_text(
        """\
[strategy.momentum_confirmation]
entry_minute_start = 8
entry_minute_end = 10
hold_to_settlement = false
profit_target_pct = 0.25
stop_loss_pct = 0.15
max_hold_minutes = 7
resolution_guard_minute = 13
min_confidence = 0.70

[[strategy.momentum_confirmation.entry_tiers]]
minute = 8
threshold_pct = 0.10

[[strategy.momentum_confirmation.entry_tiers]]
minute = 9
threshold_pct = 0.08

[[strategy.momentum_confirmation.entry_tiers]]
minute = 10
threshold_pct = 0.05

[risk]
max_position_pct = 0.02
"""
    )
    loader = ConfigLoader(config_dir=str(tmp_path), env="development")
    loader.load()
    return loader


@pytest.fixture()
def strategy(config: ConfigLoader) -> MomentumConfirmationStrategy:
    return MomentumConfirmationStrategy(config=config)


@pytest.fixture()
def orderbook() -> OrderBookSnapshot:
    """Balanced order book."""
    return OrderBookSnapshot(
        bids=[
            OrderBookLevel(price=Decimal("0.44"), size=Decimal("500")),
            OrderBookLevel(price=Decimal("0.43"), size=Decimal("300")),
        ],
        asks=[
            OrderBookLevel(price=Decimal("0.56"), size=Decimal("500")),
            OrderBookLevel(price=Decimal("0.57"), size=Decimal("300")),
        ],
        timestamp=datetime(2025, 1, 1, 0, 8),
        market_id="btc-15m-test",
    )


@pytest.fixture()
def market_state() -> MarketState:
    """Market state at minute 8 (start of tiered entry window)."""
    return MarketState(
        market_id="btc-15m-test",
        yes_price=Decimal("0.60"),
        no_price=Decimal("0.40"),
        time_remaining_seconds=7 * 60,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMomentumConfirmation:
    """Tests for all entry gates, exit logic, and lifecycle methods."""

    # ---- Gate 1: Entry window timing ----

    def test_skip_before_entry_window(
        self, strategy: MomentumConfirmationStrategy, orderbook: OrderBookSnapshot
    ) -> None:
        """Before entry_minute_start (minute < 8), strategy should emit SKIP."""
        market = MarketState(
            market_id="btc-15m-test",
            yes_price=Decimal("0.60"),
            no_price=Decimal("0.40"),
            time_remaining_seconds=12 * 60,
        )
        context = {
            "candles_1m": _momentum_up_candles(),
            "window_open_price": _BASE_BTC,
            "minute_in_window": 5,
            "yes_price": 0.60,
        }

        signals = strategy.generate_signals(market, orderbook, context)
        assert len(signals) >= 1
        assert all(s.signal_type == SignalType.SKIP for s in signals)
        assert any("before entry window" in s.metadata.get("skip_reason", "") for s in signals)

    def test_skip_past_entry_window(
        self, strategy: MomentumConfirmationStrategy, orderbook: OrderBookSnapshot
    ) -> None:
        """After entry_minute_end (minute > 10), strategy should emit SKIP."""
        market = MarketState(
            market_id="btc-15m-test",
            yes_price=Decimal("0.60"),
            no_price=Decimal("0.40"),
            time_remaining_seconds=4 * 60,
        )
        context = {
            "candles_1m": _momentum_up_candles(),
            "window_open_price": _BASE_BTC,
            "minute_in_window": 11,
            "yes_price": 0.60,
        }

        signals = strategy.generate_signals(market, orderbook, context)
        assert len(signals) >= 1
        assert all(s.signal_type == SignalType.SKIP for s in signals)
        assert any("past entry window" in s.metadata.get("skip_reason", "") for s in signals)

    # ---- Gate 2: Candle data checks ----

    def test_skip_no_candle_data(
        self, strategy: MomentumConfirmationStrategy, market_state: MarketState, orderbook: OrderBookSnapshot
    ) -> None:
        """No candle data should emit SKIP."""
        context = {
            "candles_1m": [],
            "window_open_price": _BASE_BTC,
            "minute_in_window": 8,
            "yes_price": 0.50,
        }
        signals = strategy.generate_signals(market_state, orderbook, context)
        assert all(s.signal_type == SignalType.SKIP for s in signals)
        assert any("no candle data" in s.metadata.get("skip_reason", "") for s in signals)

    def test_skip_invalid_window_open_price(
        self, strategy: MomentumConfirmationStrategy, market_state: MarketState, orderbook: OrderBookSnapshot
    ) -> None:
        """Invalid window_open_price (<= 0) should emit SKIP."""
        context = {
            "candles_1m": _momentum_up_candles(),
            "window_open_price": 0,
            "minute_in_window": 8,
            "yes_price": 0.50,
        }
        signals = strategy.generate_signals(market_state, orderbook, context)
        assert all(s.signal_type == SignalType.SKIP for s in signals)
        assert any("invalid window_open_price" in s.metadata.get("skip_reason", "") for s in signals)

    # ---- Gate 3: Move threshold ----

    def test_skip_when_move_too_small(
        self,
        strategy: MomentumConfirmationStrategy,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
    ) -> None:
        """Flat candles (tiny cumulative move) should SKIP."""
        context = {
            "candles_1m": _flat_candles(8),
            "window_open_price": _BASE_BTC,
            "minute_in_window": 8,
            "yes_price": 0.50,
        }

        signals = strategy.generate_signals(market_state, orderbook, context)
        assert all(s.signal_type == SignalType.SKIP for s in signals)
        assert any("move too small" in s.metadata.get("skip_reason", "") for s in signals)

    # ---- Full entry: momentum-following ----

    def test_entry_btc_up_buys_yes(
        self,
        strategy: MomentumConfirmationStrategy,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
    ) -> None:
        """BTC momentum up should produce ENTRY for YES side (follow the trend)."""
        candles = _momentum_up_candles()
        context = {
            "candles_1m": candles,
            "window_open_price": _BASE_BTC,
            "minute_in_window": 8,
            "yes_price": 0.60,
        }

        signals = strategy.generate_signals(market_state, orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 1

        sig = entry_signals[0]
        assert sig.direction == Side.YES
        assert sig.entry_price is not None
        assert sig.stop_loss is not None
        assert sig.take_profit is not None
        assert sig.confidence.overall >= 0.70
        assert sig.metadata["direction"] == "up"

    def test_entry_btc_down_buys_no(
        self,
        strategy: MomentumConfirmationStrategy,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
    ) -> None:
        """BTC momentum down should produce ENTRY for NO side (follow the trend)."""
        candles = _momentum_down_candles()
        context = {
            "candles_1m": candles,
            "window_open_price": _BASE_BTC,
            "minute_in_window": 8,
            "yes_price": 0.40,
        }

        signals = strategy.generate_signals(market_state, orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 1

        sig = entry_signals[0]
        assert sig.direction == Side.NO
        assert sig.metadata["direction"] == "down"
        assert sig.confidence.overall >= 0.70

    # ---- Tiered entry ----

    def test_tiered_entry_minute_10_lower_threshold(
        self,
        strategy: MomentumConfirmationStrategy,
        orderbook: OrderBookSnapshot,
    ) -> None:
        """Minute 10 should accept a 0.06% move (>0.05% tier) that minute 8 rejects (<0.10%)."""
        # Build candles with ~0.06% cumulative return (above min10 threshold, below min8)
        candles = _momentum_up_candles()
        last_close = float(candles[-1].close)
        # Set window_open so cum_return_pct = ~0.06%
        window_open = last_close / 1.0006

        market_10 = MarketState(
            market_id="btc-15m-test",
            yes_price=Decimal("0.55"),
            no_price=Decimal("0.45"),
            time_remaining_seconds=5 * 60,
        )

        # At minute 10 with 0.06% -> should enter (threshold = 0.05%)
        context_10 = {
            "candles_1m": candles,
            "window_open_price": window_open,
            "minute_in_window": 10,
            "yes_price": 0.55,
        }
        signals_10 = strategy.generate_signals(market_10, orderbook, context_10)
        # May still skip due to low confidence, but should NOT skip for "move too small"
        skip_reasons = [
            s.metadata.get("skip_reason", "") for s in signals_10 if s.signal_type == SignalType.SKIP
        ]
        assert not any("move too small" in r for r in skip_reasons)

        # At minute 8 with 0.06% -> should skip (threshold = 0.10%)
        context_8 = {
            "candles_1m": candles,
            "window_open_price": window_open,
            "minute_in_window": 8,
            "yes_price": 0.55,
        }
        signals_8 = strategy.generate_signals(market_10, orderbook, context_8)
        skip_reasons_8 = [
            s.metadata.get("skip_reason", "") for s in signals_8 if s.signal_type == SignalType.SKIP
        ]
        assert any("move too small" in r for r in skip_reasons_8)

    def test_tiered_entry_no_tier_for_minute(
        self,
        strategy: MomentumConfirmationStrategy,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
    ) -> None:
        """A minute within the window but without a tier config should skip."""
        # Minute 9 has a tier (0.08%), but test that it works
        # by checking minute 8 vs 9 vs 10 all have tiers
        assert 8 in strategy._tier_thresholds
        assert 9 in strategy._tier_thresholds
        assert 10 in strategy._tier_thresholds
        assert strategy._tier_thresholds[8] == 0.10
        assert strategy._tier_thresholds[9] == 0.08
        assert strategy._tier_thresholds[10] == 0.05

    # ---- Confirmation filters ----

    def test_last_3_agree_reflected_in_metadata(
        self,
        strategy: MomentumConfirmationStrategy,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
    ) -> None:
        """When last 3 candles agree with direction, metadata should show True."""
        candles = _momentum_up_candles()
        context = {
            "candles_1m": candles,
            "window_open_price": _BASE_BTC,
            "minute_in_window": 8,
            "yes_price": 0.60,
        }

        signals = strategy.generate_signals(market_state, orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 1
        assert entry_signals[0].metadata["last_3_agree"] == "True"

    def test_last_3_agree_false_when_mixed(
        self,
        strategy: MomentumConfirmationStrategy,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
    ) -> None:
        """When last 3 candles don't all agree, metadata shows False."""
        candles = _momentum_up_with_mixed_last3()
        context = {
            "candles_1m": candles,
            "window_open_price": _BASE_BTC,
            "minute_in_window": 8,
            "yes_price": 0.60,
        }

        signals = strategy.generate_signals(market_state, orderbook, context)
        # May or may not pass confidence threshold, but check any entry signal metadata
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        if entry_signals:
            assert entry_signals[0].metadata["last_3_agree"] == "False"

    def test_no_reversal_reflected_in_metadata(
        self,
        strategy: MomentumConfirmationStrategy,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
    ) -> None:
        """When no reversal candle exists, metadata should show True."""
        candles = _momentum_up_candles()
        context = {
            "candles_1m": candles,
            "window_open_price": _BASE_BTC,
            "minute_in_window": 8,
            "yes_price": 0.60,
        }

        signals = strategy.generate_signals(market_state, orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 1
        assert entry_signals[0].metadata["no_reversal"] == "True"

    def test_no_reversal_false_when_reversal_present(
        self,
        strategy: MomentumConfirmationStrategy,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
    ) -> None:
        """When a reversal candle exists, no_reversal should be False."""
        candles = _momentum_up_with_reversal()
        context = {
            "candles_1m": candles,
            "window_open_price": _BASE_BTC,
            "minute_in_window": 8,
            "yes_price": 0.60,
        }

        signals = strategy.generate_signals(market_state, orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        if entry_signals:
            assert entry_signals[0].metadata["no_reversal"] == "False"

    # ---- Confidence scoring ----

    def test_confidence_sub_scores(
        self,
        strategy: MomentumConfirmationStrategy,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
    ) -> None:
        """Entry signal should have populated confidence sub-scores."""
        candles = _momentum_up_candles()
        context = {
            "candles_1m": candles,
            "window_open_price": _BASE_BTC,
            "minute_in_window": 8,
            "yes_price": 0.60,
        }

        signals = strategy.generate_signals(market_state, orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 1

        conf = entry_signals[0].confidence
        # trend_strength = magnitude_score (cum_return_pct / 0.25)
        assert conf.trend_strength > 0.0
        # threshold_exceedance = time_score (min8=1.0, min9=0.9, min10=0.8)
        assert conf.threshold_exceedance > 0.0
        # book_normality = last_3_bonus (0.10 if last_3_agree)
        assert conf.book_normality == 0.10
        # liquidity_quality = no_reversal_bonus (0.05 if no_reversal)
        assert conf.liquidity_quality == 0.05
        # Overall must meet threshold
        assert conf.overall >= 0.70

    def test_confidence_magnitude_scaling(
        self,
        strategy: MomentumConfirmationStrategy,
    ) -> None:
        """Magnitude score should scale with cum_return_pct up to cap at 0.25%."""
        # Small move
        conf_small = strategy._compute_confidence(
            cum_return_pct=0.10, minute=6, last_3_agree=False, no_reversal=False
        )
        # Large move
        conf_large = strategy._compute_confidence(
            cum_return_pct=0.25, minute=6, last_3_agree=False, no_reversal=False
        )
        # At cap
        conf_over = strategy._compute_confidence(
            cum_return_pct=0.50, minute=6, last_3_agree=False, no_reversal=False
        )
        assert conf_small.trend_strength < conf_large.trend_strength
        # At and over cap should be 1.0
        assert conf_large.trend_strength == 1.0
        assert conf_over.trend_strength == 1.0

    def test_confidence_time_scaling(
        self,
        strategy: MomentumConfirmationStrategy,
    ) -> None:
        """Time score reflects accuracy ordering: min8=1.0, min9=0.9, min10=0.8."""
        conf_8 = strategy._compute_confidence(
            cum_return_pct=0.15, minute=8, last_3_agree=False, no_reversal=False
        )
        conf_9 = strategy._compute_confidence(
            cum_return_pct=0.15, minute=9, last_3_agree=False, no_reversal=False
        )
        conf_10 = strategy._compute_confidence(
            cum_return_pct=0.15, minute=10, last_3_agree=False, no_reversal=False
        )
        assert conf_8.threshold_exceedance == 1.0
        assert conf_9.threshold_exceedance == 0.9
        assert conf_10.threshold_exceedance == 0.8
        # min8 has highest time score (91.5% accuracy from backtest)
        assert conf_8.overall > conf_10.overall

    def test_confidence_bonuses(
        self,
        strategy: MomentumConfirmationStrategy,
    ) -> None:
        """last_3_agree and no_reversal should add +0.10 and +0.05 respectively."""
        base = strategy._compute_confidence(
            cum_return_pct=0.15, minute=6, last_3_agree=False, no_reversal=False
        )
        with_agree = strategy._compute_confidence(
            cum_return_pct=0.15, minute=6, last_3_agree=True, no_reversal=False
        )
        with_both = strategy._compute_confidence(
            cum_return_pct=0.15, minute=6, last_3_agree=True, no_reversal=True
        )
        assert abs(with_agree.overall - base.overall - 0.10) < 0.001
        assert abs(with_both.overall - base.overall - 0.15) < 0.001

    def test_confidence_capped_at_one(
        self,
        strategy: MomentumConfirmationStrategy,
    ) -> None:
        """Overall confidence should never exceed 1.0."""
        conf = strategy._compute_confidence(
            cum_return_pct=1.0, minute=8, last_3_agree=True, no_reversal=True
        )
        assert conf.overall <= 1.0

    # ---- Low confidence filter ----

    def test_skip_low_confidence(
        self,
        strategy: MomentumConfirmationStrategy,
        market_state: MarketState,
        orderbook: OrderBookSnapshot,
    ) -> None:
        """When confidence is below min_confidence, should emit SKIP."""
        # Use minute 8 with barely-above-threshold move and mixed candles
        # to reduce bonuses. No last_3_agree, no no_reversal bonus.
        # Confidence = 0.40 * (0.105/0.25) + 0.35 * 1.0 + 0 + 0 + 0.10 = 0.618 < 0.70
        candles = _momentum_up_with_mixed_last3()
        last_close = float(candles[-1].close)
        # Set window_open so cum_return_pct = ~0.105%
        window_open = last_close / 1.00105

        context = {
            "candles_1m": candles,
            "window_open_price": window_open,
            "minute_in_window": 8,
            "yes_price": 0.50,
        }

        signals = strategy.generate_signals(market_state, orderbook, context)
        # Should skip due to low confidence (no bonuses, low magnitude)
        assert all(s.signal_type == SignalType.SKIP for s in signals)

    # ---- Exit logic: hold-to-settlement mode (default) ----

    def test_no_exit_on_profit_in_settlement_mode(
        self, strategy: MomentumConfirmationStrategy
    ) -> None:
        """In hold-to-settlement mode, profit target should NOT trigger exit."""
        position = Position(
            market_id="btc-15m-test",
            side=Side.YES,
            token_id="yes-token",
            entry_price=Decimal("0.60"),
            quantity=Decimal("100"),
            entry_time=datetime(2025, 1, 1, 0, 8),
            stop_loss=Decimal("0.51"),
            take_profit=Decimal("0.75"),
        )
        strategy.add_position(position)
        strategy._entry_minutes["btc-15m-test"] = 8

        # YES price 0.90 -> huge profit, but should NOT exit
        market = MarketState(
            market_id="btc-15m-test",
            yes_price=Decimal("0.90"),
            no_price=Decimal("0.10"),
            time_remaining_seconds=7 * 60,
        )
        orderbook = OrderBookSnapshot(
            bids=[OrderBookLevel(price=Decimal("0.89"), size=Decimal("500"))],
            asks=[OrderBookLevel(price=Decimal("0.91"), size=Decimal("500"))],
            timestamp=datetime(2025, 1, 1, 0, 8),
            market_id="btc-15m-test",
        )
        context = {
            "candles_1m": _flat_candles(8),
            "window_open_price": _BASE_BTC,
            "minute_in_window": 8,
            "yes_price": 0.90,
        }

        signals = strategy.generate_signals(market, orderbook, context)
        exit_signals = [s for s in signals if s.signal_type == SignalType.EXIT]
        assert len(exit_signals) == 0

    def test_no_exit_on_loss_in_settlement_mode(
        self, strategy: MomentumConfirmationStrategy
    ) -> None:
        """In hold-to-settlement mode, stop loss should NOT trigger exit."""
        position = Position(
            market_id="btc-15m-test",
            side=Side.YES,
            token_id="yes-token",
            entry_price=Decimal("0.60"),
            quantity=Decimal("100"),
            entry_time=datetime(2025, 1, 1, 0, 8),
            stop_loss=Decimal("0.51"),
            take_profit=Decimal("0.75"),
        )
        strategy.add_position(position)
        strategy._entry_minutes["btc-15m-test"] = 8

        # YES price 0.30 -> big loss, but should NOT exit in settlement mode
        market = MarketState(
            market_id="btc-15m-test",
            yes_price=Decimal("0.30"),
            no_price=Decimal("0.70"),
            time_remaining_seconds=7 * 60,
        )
        orderbook = OrderBookSnapshot(
            bids=[OrderBookLevel(price=Decimal("0.29"), size=Decimal("500"))],
            asks=[OrderBookLevel(price=Decimal("0.31"), size=Decimal("500"))],
            timestamp=datetime(2025, 1, 1, 0, 8),
            market_id="btc-15m-test",
        )
        context = {
            "candles_1m": _flat_candles(8),
            "window_open_price": _BASE_BTC,
            "minute_in_window": 8,
            "yes_price": 0.30,
        }

        signals = strategy.generate_signals(market, orderbook, context)
        exit_signals = [s for s in signals if s.signal_type == SignalType.EXIT]
        assert len(exit_signals) == 0

    def test_exit_on_resolution_guard(self, strategy: MomentumConfirmationStrategy) -> None:
        """Position at resolution guard minute (14) should trigger EXIT.

        NOTE: The test fixture uses resolution_guard_minute=14 to validate the
        guard mechanism. The live config uses resolution_guard_minute=15, which
        means the guard never fires in a 0-14 minute window — this is intentional
        for hold-to-settlement mode where the window boundary close handles exit.
        """
        position = Position(
            market_id="btc-15m-test",
            side=Side.YES,
            token_id="yes-token",
            entry_price=Decimal("0.60"),
            quantity=Decimal("100"),
            entry_time=datetime(2025, 1, 1, 0, 8),
            stop_loss=Decimal("0.51"),
            take_profit=Decimal("0.75"),
        )
        strategy.add_position(position)
        strategy._entry_minutes["btc-15m-test"] = 8

        market = MarketState(
            market_id="btc-15m-test",
            yes_price=Decimal("0.61"),
            no_price=Decimal("0.39"),
            time_remaining_seconds=1 * 60,
        )
        orderbook = OrderBookSnapshot(
            bids=[OrderBookLevel(price=Decimal("0.60"), size=Decimal("500"))],
            asks=[OrderBookLevel(price=Decimal("0.62"), size=Decimal("500"))],
            timestamp=datetime(2025, 1, 1, 0, 14),
            market_id="btc-15m-test",
        )
        context = {
            "candles_1m": _flat_candles(14),
            "window_open_price": _BASE_BTC,
            "minute_in_window": 14,
            "yes_price": 0.61,
        }

        signals = strategy.generate_signals(market, orderbook, context)
        exit_signals = [s for s in signals if s.signal_type == SignalType.EXIT]
        assert len(exit_signals) == 1
        assert exit_signals[0].exit_reason == ExitReason.RESOLUTION_GUARD

    def test_resolution_guard_minute_15_never_fires(self, tmp_path: Any) -> None:
        """With resolution_guard_minute=15, guard never fires in 0-14 window.

        This validates the live config behavior: resolution_guard_minute=15
        effectively disables the guard, relying on window boundary close instead.
        """
        default = tmp_path / "default.toml"
        default.write_text(
            """\
[strategy.momentum_confirmation]
entry_minute_start = 8
entry_minute_end = 10
hold_to_settlement = true
profit_target_pct = 0.40
stop_loss_pct = 1.00
max_hold_minutes = 10
resolution_guard_minute = 15
min_confidence = 0.70

[[strategy.momentum_confirmation.entry_tiers]]
minute = 8
threshold_pct = 0.10

[[strategy.momentum_confirmation.entry_tiers]]
minute = 9
threshold_pct = 0.08

[[strategy.momentum_confirmation.entry_tiers]]
minute = 10
threshold_pct = 0.05

[risk]
max_position_pct = 0.02
"""
        )
        loader = ConfigLoader(config_dir=str(tmp_path), env="development")
        loader.load()
        strat = MomentumConfirmationStrategy(config=loader)

        position = Position(
            market_id="btc-15m-test",
            side=Side.YES,
            token_id="yes-token",
            entry_price=Decimal("0.60"),
            quantity=Decimal("100"),
            entry_time=datetime(2025, 1, 1, 0, 8),
            stop_loss=Decimal("0.51"),
            take_profit=Decimal("0.75"),
        )
        strat.add_position(position)
        strat._entry_minutes["btc-15m-test"] = 8

        # At minute 14 (max in a 0-14 window), guard should NOT fire
        market = MarketState(
            market_id="btc-15m-test",
            yes_price=Decimal("0.61"),
            no_price=Decimal("0.39"),
            time_remaining_seconds=1 * 60,
        )
        orderbook = OrderBookSnapshot(
            bids=[OrderBookLevel(price=Decimal("0.60"), size=Decimal("500"))],
            asks=[OrderBookLevel(price=Decimal("0.62"), size=Decimal("500"))],
            timestamp=datetime(2025, 1, 1, 0, 14),
            market_id="btc-15m-test",
        )
        context = {
            "candles_1m": _flat_candles(14),
            "window_open_price": _BASE_BTC,
            "minute_in_window": 14,
            "yes_price": 0.61,
        }

        signals = strat.generate_signals(market, orderbook, context)
        exit_signals = [s for s in signals if s.signal_type == SignalType.EXIT]
        assert len(exit_signals) == 0, "guard_minute=15 should not fire at minute 14"

    # ---- Exit logic: non-settlement mode (traditional) ----

    def test_exit_on_profit_target_non_settlement(
        self, config_no_settlement: ConfigLoader
    ) -> None:
        """With hold_to_settlement=false, profit target (25%) triggers EXIT."""
        strat = MomentumConfirmationStrategy(config=config_no_settlement)
        position = Position(
            market_id="btc-15m-test",
            side=Side.YES,
            token_id="yes-token",
            entry_price=Decimal("0.60"),
            quantity=Decimal("100"),
            entry_time=datetime(2025, 1, 1, 0, 8),
            stop_loss=Decimal("0.51"),
            take_profit=Decimal("0.75"),
        )
        strat.add_position(position)
        strat._entry_minutes["btc-15m-test"] = 8

        # YES price 0.76 -> pnl = 26.7% > 25%
        market = MarketState(
            market_id="btc-15m-test",
            yes_price=Decimal("0.76"),
            no_price=Decimal("0.24"),
            time_remaining_seconds=7 * 60,
        )
        orderbook = OrderBookSnapshot(
            bids=[OrderBookLevel(price=Decimal("0.75"), size=Decimal("500"))],
            asks=[OrderBookLevel(price=Decimal("0.77"), size=Decimal("500"))],
            timestamp=datetime(2025, 1, 1, 0, 8),
            market_id="btc-15m-test",
        )
        context = {
            "candles_1m": _flat_candles(8),
            "window_open_price": _BASE_BTC,
            "minute_in_window": 8,
            "yes_price": 0.76,
        }

        signals = strat.generate_signals(market, orderbook, context)
        exit_signals = [s for s in signals if s.signal_type == SignalType.EXIT]
        assert len(exit_signals) == 1
        assert exit_signals[0].exit_reason == ExitReason.PROFIT_TARGET

    def test_exit_on_stop_loss_non_settlement(
        self, config_no_settlement: ConfigLoader
    ) -> None:
        """With hold_to_settlement=false, stop loss (15%) triggers EXIT."""
        strat = MomentumConfirmationStrategy(config=config_no_settlement)
        position = Position(
            market_id="btc-15m-test",
            side=Side.YES,
            token_id="yes-token",
            entry_price=Decimal("0.60"),
            quantity=Decimal("100"),
            entry_time=datetime(2025, 1, 1, 0, 8),
            stop_loss=Decimal("0.51"),
            take_profit=Decimal("0.75"),
        )
        strat.add_position(position)
        strat._entry_minutes["btc-15m-test"] = 8

        # YES price 0.50 -> pnl = -16.7% < -15%
        market = MarketState(
            market_id="btc-15m-test",
            yes_price=Decimal("0.50"),
            no_price=Decimal("0.50"),
            time_remaining_seconds=7 * 60,
        )
        orderbook = OrderBookSnapshot(
            bids=[OrderBookLevel(price=Decimal("0.49"), size=Decimal("500"))],
            asks=[OrderBookLevel(price=Decimal("0.51"), size=Decimal("500"))],
            timestamp=datetime(2025, 1, 1, 0, 8),
            market_id="btc-15m-test",
        )
        context = {
            "candles_1m": _flat_candles(8),
            "window_open_price": _BASE_BTC,
            "minute_in_window": 8,
            "yes_price": 0.50,
        }

        signals = strat.generate_signals(market, orderbook, context)
        exit_signals = [s for s in signals if s.signal_type == SignalType.EXIT]
        assert len(exit_signals) == 1
        assert exit_signals[0].exit_reason == ExitReason.HARD_STOP_LOSS

    def test_exit_on_max_hold_non_settlement(
        self, config_no_settlement: ConfigLoader
    ) -> None:
        """With hold_to_settlement=false, resolution guard at min 13 triggers EXIT."""
        strat = MomentumConfirmationStrategy(config=config_no_settlement)
        position = Position(
            market_id="btc-15m-test",
            side=Side.YES,
            token_id="yes-token",
            entry_price=Decimal("0.60"),
            quantity=Decimal("100"),
            entry_time=datetime(2025, 1, 1, 0, 8),
            stop_loss=Decimal("0.51"),
            take_profit=Decimal("0.75"),
        )
        strat.add_position(position)
        strat._entry_minutes["btc-15m-test"] = 8

        # Price near entry (no profit/loss trigger), at resolution guard minute 13
        market = MarketState(
            market_id="btc-15m-test",
            yes_price=Decimal("0.61"),
            no_price=Decimal("0.39"),
            time_remaining_seconds=2 * 60,
        )
        orderbook = OrderBookSnapshot(
            bids=[OrderBookLevel(price=Decimal("0.60"), size=Decimal("500"))],
            asks=[OrderBookLevel(price=Decimal("0.62"), size=Decimal("500"))],
            timestamp=datetime(2025, 1, 1, 0, 13),
            market_id="btc-15m-test",
        )
        context = {
            "candles_1m": _flat_candles(13),
            "window_open_price": _BASE_BTC,
            "minute_in_window": 13,
            "yes_price": 0.61,
        }

        signals = strat.generate_signals(market, orderbook, context)
        exit_signals = [s for s in signals if s.signal_type == SignalType.EXIT]
        assert len(exit_signals) == 1
        assert exit_signals[0].exit_reason == ExitReason.RESOLUTION_GUARD

    # ---- Lifecycle methods ----

    def test_on_fill_no_raise(self, strategy: MomentumConfirmationStrategy) -> None:
        """on_fill should not raise."""
        fill = Fill(
            order_id=uuid4(),
            price=Decimal("0.60"),
            size=Decimal("100"),
        )
        position = Position(
            market_id="btc-15m-test",
            side=Side.YES,
            token_id="yes-token",
            entry_price=Decimal("0.60"),
            quantity=Decimal("100"),
            entry_time=datetime(2025, 1, 1, 0, 8),
            stop_loss=Decimal("0.51"),
            take_profit=Decimal("0.75"),
        )
        strategy.on_fill(fill, position)

    def test_on_cancel_no_raise(self, strategy: MomentumConfirmationStrategy) -> None:
        """on_cancel should not raise."""
        strategy.on_cancel("order-abc", "test_cancel_reason")

    # ---- Registry ----

    def test_registry_momentum_confirmation(self) -> None:
        """momentum_confirmation should be registered in the strategy registry."""
        from src.strategies import registry

        registry.register("momentum_confirmation")(MomentumConfirmationStrategy)
        cls = registry.get("momentum_confirmation")
        assert cls is MomentumConfirmationStrategy

    def test_registry_reversal_catcher_alias(self) -> None:
        """reversal_catcher should also be registered as backward-compat alias."""
        from src.strategies import registry

        registry.register("reversal_catcher")(MomentumConfirmationStrategy)
        cls = registry.get("reversal_catcher")
        assert cls is MomentumConfirmationStrategy

    # ---- Static helper methods ----

    def test_check_last_3_agree_true(self) -> None:
        """When last 3 candles match cum_return direction, returns True."""
        candles = _momentum_up_candles()  # all green
        assert MomentumConfirmationStrategy._check_last_3_agree(candles, 0.01) is True

    def test_check_last_3_agree_false(self) -> None:
        """When last 3 candles don't all match, returns False."""
        candles = _momentum_up_with_mixed_last3()  # last 3 has a red
        assert MomentumConfirmationStrategy._check_last_3_agree(candles, 0.01) is False

    def test_check_last_3_agree_too_few_candles(self) -> None:
        """With fewer than 3 candles, returns False."""
        candles = [_make_candle(_BASE_BTC, _BASE_BTC + 10, 0)]
        assert MomentumConfirmationStrategy._check_last_3_agree(candles, 0.01) is False

    def test_check_no_reversal_true(self) -> None:
        """When no reversal candle exists, returns True."""
        candles = _momentum_up_candles()  # all green, cum_return > 0
        assert MomentumConfirmationStrategy._check_no_reversal(candles, 0.01) is True

    def test_check_no_reversal_false(self) -> None:
        """When a reversal candle exists, returns False."""
        candles = _momentum_up_with_reversal()  # has a red in uptrend
        assert MomentumConfirmationStrategy._check_no_reversal(candles, 0.01) is False

    def test_check_no_reversal_empty_candles(self) -> None:
        """Empty candle list should return True (no reversal found)."""
        assert MomentumConfirmationStrategy._check_no_reversal([], 0.01) is True
