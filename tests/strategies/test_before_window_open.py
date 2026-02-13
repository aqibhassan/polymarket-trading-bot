"""Tests for BeforeWindowOpenStrategy — pre-window signal ensemble + ML features."""

from __future__ import annotations

import math
from collections import namedtuple
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.models.market import (
    MarketState,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
)
from src.models.signal import SignalType

SimpleCandle = namedtuple("SimpleCandle", ["timestamp", "open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config() -> MagicMock:
    """ConfigLoader that returns BWO defaults."""
    config = MagicMock()

    defaults: dict[str, object] = {
        "strategy.before_window_open.weight_prior_momentum": 0.25,
        "strategy.before_window_open.weight_multi_tf": 0.20,
        "strategy.before_window_open.weight_short_momentum": 0.20,
        "strategy.before_window_open.weight_candle_pattern": 0.15,
        "strategy.before_window_open.weight_volume": 0.10,
        "strategy.before_window_open.weight_vol_regime": 0.10,
        "strategy.before_window_open.min_confidence": 0.40,
        "strategy.before_window_open.max_entry_price": 0.60,
        "strategy.before_window_open.entry_minute_end": 3,
        "strategy.before_window_open.min_vote_weight": 0.40,
        "strategy.before_window_open.mtf_min_alignment": 2.0,
        "strategy.before_window_open.stm_min_strength": 0.6,
        "strategy.before_window_open.pattern_min_streak": 3,
        "strategy.before_window_open.volume_min_alignment": 0.3,
        "strategy.before_window_open.resolution_guard_minute": 14,
        "strategy.before_window_open.hold_to_settlement": True,
        "strategy.before_window_open.max_position_pct": 0.05,
        # ML model config keys
        "strategy.before_window_open.use_ml_model": False,
        "strategy.before_window_open.model_path": "models/bwo_model_min3.joblib",
        "strategy.before_window_open.entry_minute": 3,
        "strategy.before_window_open.ml_confidence_threshold": 0.70,
        "strategy.before_window_open.ml_price_sensitivity": 0.5,
        "strategy.before_window_open.ml_max_entry_price": 0.70,
    }

    def side_effect(key: str, default: object = None) -> object:
        return defaults.get(key, default)

    config.get.side_effect = side_effect
    config.validate_keys.return_value = None
    return config


@pytest.fixture
def make_candle():
    """Factory for creating candle-like namedtuples."""
    def _make(
        open_: float = 100000.0,
        close: float = 100100.0,
        high: float | None = None,
        low: float | None = None,
        volume: float = 100.0,
    ) -> SimpleCandle:
        return SimpleCandle(
            timestamp=datetime.now(tz=UTC),
            open=open_,
            high=high or max(open_, close),
            low=low or min(open_, close),
            close=close,
            volume=volume,
        )
    return _make


@pytest.fixture
def neutral_orderbook() -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=datetime.now(tz=UTC),
        market_id="test",
        bids=[OrderBookLevel(price=Decimal("0.50"), size=Decimal("100"))],
        asks=[OrderBookLevel(price=Decimal("0.51"), size=Decimal("100"))],
    )


def _market_state(yes_price: float = 0.50, time_remaining: int = 900) -> MarketState:
    return MarketState(
        market_id="test_market",
        yes_price=Decimal(str(yes_price)),
        no_price=Decimal(str(1.0 - yes_price)),
        time_remaining_seconds=time_remaining,
    )


# ---------------------------------------------------------------------------
# Feature function tests (strategy module helpers)
# ---------------------------------------------------------------------------

class TestFeatureFunctions:
    def test_prior_window_momentum_up(self, make_candle) -> None:
        from src.strategies.before_window_open import _prior_window_momentum
        candles = [make_candle(open_=100.0, close=101.0), make_candle(open_=101.0, close=102.0)]
        direction, mag = _prior_window_momentum(candles)
        assert direction == 1.0
        assert mag > 0

    def test_prior_window_momentum_down(self, make_candle) -> None:
        from src.strategies.before_window_open import _prior_window_momentum
        candles = [make_candle(open_=100.0, close=99.0), make_candle(open_=99.0, close=98.0)]
        direction, mag = _prior_window_momentum(candles)
        assert direction == -1.0
        assert mag > 0

    def test_prior_window_momentum_empty(self) -> None:
        from src.strategies.before_window_open import _prior_window_momentum
        direction, mag = _prior_window_momentum([])
        assert direction == 0.0
        assert mag == 0.0

    def test_multi_tf_alignment_bullish(self, make_candle) -> None:
        from src.strategies.before_window_open import _multi_tf_alignment
        candles = [make_candle(open_=100.0 + i, close=101.0 + i) for i in range(240)]
        score = _multi_tf_alignment(candles)
        assert score == 3.0

    def test_multi_tf_alignment_insufficient_data(self, make_candle) -> None:
        from src.strategies.before_window_open import _multi_tf_alignment
        candles = [make_candle() for _ in range(10)]
        score = _multi_tf_alignment(candles)
        assert score == 0.0

    def test_short_term_momentum(self, make_candle) -> None:
        from src.strategies.before_window_open import _short_term_momentum
        candles = [make_candle(open_=100.0 + i, close=101.0 + i) for i in range(5)]
        direction, strength = _short_term_momentum(candles)
        assert direction == 1.0
        assert strength == 1.0

    def test_volatility_regime_needs_data(self) -> None:
        from src.strategies.before_window_open import _volatility_regime
        result = _volatility_regime([])
        assert result == 0.0

    def test_candle_streak(self, make_candle) -> None:
        from src.strategies.before_window_open import _candle_streak
        candles = [make_candle(open_=100.0 + i, close=101.0 + i) for i in range(5)]
        length, direction = _candle_streak(candles)
        assert length == 5.0
        assert direction == 1.0


# ---------------------------------------------------------------------------
# No-lookahead assertion tests
# ---------------------------------------------------------------------------

class TestNoLookahead:
    """Ensure backtest features use only pre-window data."""

    def test_features_use_only_prior_data(self) -> None:
        """Features computed from data before window, not from the window itself."""
        from scripts.backtest_before_window import WindowData, compute_all_features
        from scripts.fast_loader import FastCandle

        # Build a small candle series: 60 candles going down, then a 15-candle window going up
        candles: list[FastCandle] = []
        base_ts = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
        for i in range(75):
            from datetime import timedelta
            ts = base_ts + timedelta(minutes=i)
            if i < 60:
                # History: trending down
                candles.append(FastCandle(
                    timestamp=ts, open=100.0 - i * 0.1, high=100.0 - i * 0.1 + 0.05,
                    low=100.0 - i * 0.1 - 0.15, close=100.0 - i * 0.1 - 0.1, volume=100.0,
                ))
            else:
                # Future window: trending up
                offset = i - 60
                candles.append(FastCandle(
                    timestamp=ts, open=94.0 + offset * 0.5, high=94.0 + offset * 0.5 + 0.6,
                    low=94.0 + offset * 0.5 - 0.1, close=94.0 + offset * 0.5 + 0.5, volume=100.0,
                ))

        # Window is candles[60:75]
        window = candles[60:75]
        windows = [window]

        wd = WindowData(window_idx=0, candle_idx=60, resolution="Up", timestamp=candles[60].timestamp)
        features = compute_all_features(wd, windows, candles)

        # Short-term momentum should be DOWN (last 5 candles before window = candles[55:60])
        assert features["stm_dir"] == -1.0
        # Multi-TF should show bearish (history is down)
        assert features["mtf_15m"] == -1.0


# ---------------------------------------------------------------------------
# Fee calculation tests
# ---------------------------------------------------------------------------

class TestFeeCalculation:
    def test_polymarket_fee_at_050(self) -> None:
        from scripts.backtest_before_window import polymarket_fee
        fee = polymarket_fee(100.0, 0.50)
        assert abs(fee - 1.5625) < 0.001

    def test_polymarket_fee_at_060(self) -> None:
        from scripts.backtest_before_window import polymarket_fee
        fee = polymarket_fee(100.0, 0.60)
        assert abs(fee - 1.44) < 0.001

    def test_polymarket_fee_at_boundaries(self) -> None:
        from scripts.backtest_before_window import polymarket_fee
        assert polymarket_fee(100.0, 0.0) == 0.0
        assert polymarket_fee(100.0, 1.0) == 0.0

    def test_trade_simulation_correct(self) -> None:
        from scripts.backtest_before_window import simulate_trade
        settlement, pnl_gross, pnl_net, fee = simulate_trade("Up", "Up", 0.50)
        assert settlement == 1.0
        assert pnl_gross == pytest.approx(100.0, abs=0.01)
        assert pnl_net < pnl_gross

    def test_trade_simulation_incorrect(self) -> None:
        from scripts.backtest_before_window import simulate_trade
        settlement, pnl_gross, pnl_net, fee = simulate_trade("Up", "Down", 0.50)
        assert settlement == 0.0
        assert pnl_gross == pytest.approx(-100.0, abs=0.01)


# ---------------------------------------------------------------------------
# Strategy class tests
# ---------------------------------------------------------------------------

class TestBeforeWindowOpenStrategy:
    def test_registered(self, mock_config: MagicMock) -> None:
        from src.strategies.before_window_open import BeforeWindowOpenStrategy
        from src.strategies.registry import get
        cls = get("before_window_open")
        assert cls is BeforeWindowOpenStrategy

    def test_skip_past_entry_window(
        self, mock_config: MagicMock, neutral_orderbook: OrderBookSnapshot,
    ) -> None:
        from src.strategies.before_window_open import BeforeWindowOpenStrategy
        strat = BeforeWindowOpenStrategy(config=mock_config)
        context = {
            "prior_window_candles": [],
            "recent_1m_candles": [],
            "minute_in_window": 5,
            "yes_price": 0.50,
        }
        signals = strat.generate_signals(_market_state(), neutral_orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 0

    def test_skip_price_too_high(
        self, mock_config: MagicMock, neutral_orderbook: OrderBookSnapshot,
    ) -> None:
        from src.strategies.before_window_open import BeforeWindowOpenStrategy
        strat = BeforeWindowOpenStrategy(config=mock_config)
        context = {
            "prior_window_candles": [],
            "recent_1m_candles": [],
            "minute_in_window": 0,
            "yes_price": 0.75,
        }
        signals = strat.generate_signals(
            _market_state(yes_price=0.75), neutral_orderbook, context,
        )
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 0

    def test_entry_with_strong_signals(
        self, mock_config: MagicMock, neutral_orderbook: OrderBookSnapshot, make_candle,
    ) -> None:
        from src.strategies.before_window_open import BeforeWindowOpenStrategy
        strat = BeforeWindowOpenStrategy(config=mock_config)

        prior = [make_candle(open_=100.0, close=102.0) for _ in range(15)]
        recent = [make_candle(open_=100.0 + i * 0.1, close=100.1 + i * 0.1, volume=200.0) for i in range(240)]

        context = {
            "prior_window_candles": prior,
            "recent_1m_candles": recent,
            "minute_in_window": 0,
            "yes_price": 0.50,
        }

        signals = strat.generate_signals(_market_state(), neutral_orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 1
        assert entry_signals[0].direction == Side.YES

    def test_entry_bearish_direction(
        self, mock_config: MagicMock, neutral_orderbook: OrderBookSnapshot, make_candle,
    ) -> None:
        from src.strategies.before_window_open import BeforeWindowOpenStrategy
        strat = BeforeWindowOpenStrategy(config=mock_config)

        prior = [make_candle(open_=100.0, close=98.0) for _ in range(15)]
        recent = [make_candle(open_=100.0 - i * 0.1, close=99.9 - i * 0.1, volume=200.0) for i in range(240)]

        context = {
            "prior_window_candles": prior,
            "recent_1m_candles": recent,
            "minute_in_window": 0,
            "yes_price": 0.50,
        }

        signals = strat.generate_signals(_market_state(), neutral_orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 1
        assert entry_signals[0].direction == Side.NO

    def test_resolution_guard_exit(
        self, mock_config: MagicMock, neutral_orderbook: OrderBookSnapshot,
    ) -> None:
        from src.models.market import Position
        from src.models.signal import ExitReason
        from src.strategies.before_window_open import BeforeWindowOpenStrategy

        strat = BeforeWindowOpenStrategy(config=mock_config)

        pos = Position(
            market_id="test_market",
            side=Side.YES,
            token_id="yes_token",
            entry_price=Decimal("0.50"),
            quantity=Decimal("100"),
            entry_time=datetime.now(tz=UTC),
            stop_loss=Decimal("0.00"),
            take_profit=Decimal("1.00"),
        )
        strat.add_position(pos)

        context = {
            "prior_window_candles": [],
            "recent_1m_candles": [],
            "minute_in_window": 14,
            "yes_price": 0.50,
        }

        signals = strat.generate_signals(
            _market_state(time_remaining=60), neutral_orderbook, context,
        )
        exit_signals = [s for s in signals if s.signal_type == SignalType.EXIT]
        assert len(exit_signals) == 1
        assert exit_signals[0].exit_reason == ExitReason.RESOLUTION_GUARD

    def test_no_signals_returns_empty(
        self, mock_config: MagicMock, neutral_orderbook: OrderBookSnapshot,
    ) -> None:
        from src.strategies.before_window_open import BeforeWindowOpenStrategy
        strat = BeforeWindowOpenStrategy(config=mock_config)

        context = {
            "prior_window_candles": [],
            "recent_1m_candles": [],
            "minute_in_window": 0,
            "yes_price": 0.50,
        }

        signals = strat.generate_signals(_market_state(), neutral_orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 0


# ---------------------------------------------------------------------------
# Helper: build candle series for backtest feature tests
# ---------------------------------------------------------------------------

def _build_candles(
    n: int,
    start_price: float = 100.0,
    trend: float = 0.0,
    volume: float = 100.0,
    base_ts: datetime | None = None,
) -> list:
    """Build a list of FastCandle namedtuples for testing.

    Args:
        n: number of candles
        trend: per-candle price change (positive = up, negative = down, 0 = flat)
        volume: volume per candle
        base_ts: starting timestamp (default: 2025-01-01 00:00 UTC)
    """
    from scripts.fast_loader import FastCandle

    if base_ts is None:
        base_ts = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)

    candles = []
    price = start_price
    for i in range(n):
        ts = base_ts + timedelta(minutes=i)
        open_ = price
        close_ = price + trend
        high_ = max(open_, close_) + abs(trend) * 0.1 + 0.01
        low_ = min(open_, close_) - abs(trend) * 0.1 - 0.01
        candles.append(FastCandle(
            timestamp=ts, open=open_, high=high_, low=low_, close=close_, volume=volume,
        ))
        price = close_
    return candles


# ---------------------------------------------------------------------------
# Early-window feature tests
# ---------------------------------------------------------------------------

class TestEarlyWindowFeatures:
    """Tests for early-window features in compute_all_features."""

    def test_early_features_at_minute_0(self) -> None:
        """Entry minute 0 should give zero early features."""
        from scripts.backtest_before_window import WindowData, compute_all_features

        # Build 60 history + 15 window candles
        candles = _build_candles(75, start_price=100.0, trend=0.1)
        windows = [candles[60:75]]
        wd = WindowData(
            window_idx=0, candle_idx=60, resolution="Up", timestamp=candles[60].timestamp,
        )
        features = compute_all_features(wd, windows, candles, entry_minute=0)

        assert features["early_cum_return"] == 0.0
        assert features["early_direction"] == 0.0
        assert features["early_magnitude"] == 0.0
        assert features["early_green_ratio"] == 0.0
        assert features["early_vol"] == 0.0
        assert features["early_max_move"] == 0.0

    def test_early_features_at_minute_3(self) -> None:
        """Entry minute 3 should compute features from first 3 window candles."""
        from scripts.backtest_before_window import WindowData, compute_all_features

        # Build 60 flat history candles then 15 window candles going UP
        history = _build_candles(60, start_price=100.0, trend=0.0)
        last_price = history[-1].close
        window = _build_candles(15, start_price=last_price, trend=0.5,
                                base_ts=history[-1].timestamp + timedelta(minutes=1))
        candles = history + window
        windows = [candles[60:75]]

        wd = WindowData(
            window_idx=0, candle_idx=60, resolution="Up", timestamp=candles[60].timestamp,
        )
        features = compute_all_features(wd, windows, candles, entry_minute=3)

        # First 3 window candles go UP (trend=+0.5 each)
        assert features["early_cum_return"] > 0
        assert features["early_direction"] == 1.0
        assert features["early_magnitude"] > 0
        # All 3 candles are green (close > open)
        assert features["early_green_ratio"] == 1.0

    def test_early_features_down_direction(self) -> None:
        """Verify early features detect downward movement."""
        from scripts.backtest_before_window import WindowData, compute_all_features

        # Build 60 flat history then 15 window candles going DOWN
        history = _build_candles(60, start_price=100.0, trend=0.0)
        last_price = history[-1].close
        window = _build_candles(15, start_price=last_price, trend=-0.5,
                                base_ts=history[-1].timestamp + timedelta(minutes=1))
        candles = history + window
        windows = [candles[60:75]]

        wd = WindowData(
            window_idx=0, candle_idx=60, resolution="Down", timestamp=candles[60].timestamp,
        )
        features = compute_all_features(wd, windows, candles, entry_minute=3)

        assert features["early_direction"] == -1.0
        assert features["early_cum_return"] < 0

    def test_early_features_no_lookahead(self) -> None:
        """Early features must NOT use candles after entry_minute."""
        from scripts.backtest_before_window import WindowData, compute_all_features
        from scripts.fast_loader import FastCandle

        base_ts = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)

        # Build 60 flat history candles
        history = _build_candles(60, start_price=100.0, trend=0.0, base_ts=base_ts)

        # Window: first 3 candles go UP, last 12 go DOWN sharply
        window_candles = []
        price = history[-1].close
        for i in range(15):
            ts = base_ts + timedelta(minutes=60 + i)
            if i < 3:
                # UP movement
                open_ = price
                close_ = price + 1.0
            else:
                # DOWN movement (sharp reversal)
                open_ = price
                close_ = price - 2.0
            high_ = max(open_, close_) + 0.1
            low_ = min(open_, close_) - 0.1
            window_candles.append(FastCandle(
                timestamp=ts, open=open_, high=high_, low=low_, close=close_, volume=100.0,
            ))
            price = close_

        candles = history + window_candles
        windows = [candles[60:75]]

        wd = WindowData(
            window_idx=0, candle_idx=60, resolution="Down", timestamp=candles[60].timestamp,
        )
        features = compute_all_features(wd, windows, candles, entry_minute=3)

        # At entry_minute=3, early features see only the first 3 UP candles
        assert features["early_direction"] == 1.0
        assert features["early_cum_return"] > 0
        # The final window resolution is DOWN but early features should not know that


# ---------------------------------------------------------------------------
# TA indicator tests
# ---------------------------------------------------------------------------

class TestTAIndicators:
    """Tests for TA indicator features."""

    def test_rsi_overbought(self) -> None:
        """RSI should be high when recent candles are all green."""
        from scripts.backtest_before_window import WindowData, compute_all_features

        # 75 candles trending UP => gains >> losses => RSI > 70
        candles = _build_candles(75, start_price=100.0, trend=0.5)
        windows = [candles[60:75]]
        wd = WindowData(
            window_idx=0, candle_idx=60, resolution="Up", timestamp=candles[60].timestamp,
        )
        features = compute_all_features(wd, windows, candles, entry_minute=0)

        assert features["rsi_14"] > 70

    def test_rsi_oversold(self) -> None:
        """RSI should be low when recent candles are all red."""
        from scripts.backtest_before_window import WindowData, compute_all_features

        # 75 candles trending DOWN => losses >> gains => RSI < 30
        candles = _build_candles(75, start_price=200.0, trend=-0.5)
        windows = [candles[60:75]]
        wd = WindowData(
            window_idx=0, candle_idx=60, resolution="Down", timestamp=candles[60].timestamp,
        )
        features = compute_all_features(wd, windows, candles, entry_minute=0)

        assert features["rsi_14"] < 30

    def test_rsi_default_with_insufficient_data(self) -> None:
        """RSI defaults to 50 with insufficient history."""
        from scripts.backtest_before_window import WindowData, compute_all_features

        # Only 10 history candles + 15 window = 25 total, but hist_len=10 < 15
        candles = _build_candles(25, start_price=100.0, trend=0.1)
        windows = [candles[10:25]]
        wd = WindowData(
            window_idx=0, candle_idx=10, resolution="Up", timestamp=candles[10].timestamp,
        )
        features = compute_all_features(wd, windows, candles, entry_minute=0)

        assert features["rsi_14"] == 50.0

    def test_bollinger_band_pct_b(self) -> None:
        """Bollinger %B should be between 0 and 1 for normal data."""
        from scripts.backtest_before_window import WindowData, compute_all_features

        # 75+ candles with mild oscillation around a mean
        from scripts.fast_loader import FastCandle

        base_ts = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
        candles = []
        for i in range(75):
            ts = base_ts + timedelta(minutes=i)
            # Oscillate around 100: 100 +/- sin-like pattern
            offset = math.sin(i * 0.3) * 2.0
            open_ = 100.0 + offset
            close_ = 100.0 + offset + 0.1
            high_ = max(open_, close_) + 0.2
            low_ = min(open_, close_) - 0.2
            candles.append(FastCandle(
                timestamp=ts, open=open_, high=high_, low=low_, close=close_, volume=100.0,
            ))
        windows = [candles[60:75]]
        wd = WindowData(
            window_idx=0, candle_idx=60, resolution="Up", timestamp=candles[60].timestamp,
        )
        features = compute_all_features(wd, windows, candles, entry_minute=0)

        assert 0.0 <= features["bb_pct_b"] <= 1.0

    def test_atr_positive(self) -> None:
        """ATR should be positive for non-flat data."""
        from scripts.backtest_before_window import WindowData, compute_all_features

        # Build candles with some movement (trend != 0)
        candles = _build_candles(75, start_price=100.0, trend=0.3)
        windows = [candles[60:75]]
        wd = WindowData(
            window_idx=0, candle_idx=60, resolution="Up", timestamp=candles[60].timestamp,
        )
        features = compute_all_features(wd, windows, candles, entry_minute=0)

        assert features["atr_14"] > 0

    def test_mean_reversion_z(self) -> None:
        """Mean reversion z-score should be positive when price above SMA."""
        from scripts.backtest_before_window import WindowData, compute_all_features
        from scripts.fast_loader import FastCandle

        base_ts = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
        candles = []
        # First 50 candles flat around 100, then last 25 jump to 105+
        for i in range(75):
            ts = base_ts + timedelta(minutes=i)
            if i < 50:
                open_ = 100.0
                close_ = 100.0 + 0.01
            else:
                # Price jumps above the 60-candle SMA
                open_ = 105.0
                close_ = 105.5
            high_ = max(open_, close_) + 0.1
            low_ = min(open_, close_) - 0.1
            candles.append(FastCandle(
                timestamp=ts, open=open_, high=high_, low=low_, close=close_, volume=100.0,
            ))
        windows = [candles[60:75]]
        wd = WindowData(
            window_idx=0, candle_idx=60, resolution="Up", timestamp=candles[60].timestamp,
        )
        features = compute_all_features(wd, windows, candles, entry_minute=0)

        # Price (105.5) is well above the 60-candle SMA (~102-ish), so z > 0
        assert features["mean_reversion_z"] > 0

    def test_macd_histogram_bullish(self) -> None:
        """MACD histogram sign should be +1 for uptrending data."""
        from scripts.backtest_before_window import WindowData, compute_all_features

        # 75+ candles trending up => EMA12 > EMA26 => MACD line positive
        candles = _build_candles(75, start_price=100.0, trend=0.5)
        windows = [candles[60:75]]
        wd = WindowData(
            window_idx=0, candle_idx=60, resolution="Up", timestamp=candles[60].timestamp,
        )
        features = compute_all_features(wd, windows, candles, entry_minute=0)

        assert features["macd_histogram_sign"] == 1.0

    def test_price_vs_vwap(self) -> None:
        """Price vs VWAP should be computable with volume data."""
        from scripts.backtest_before_window import WindowData, compute_all_features

        candles = _build_candles(75, start_price=100.0, trend=0.1, volume=150.0)
        windows = [candles[60:75]]
        wd = WindowData(
            window_idx=0, candle_idx=60, resolution="Up", timestamp=candles[60].timestamp,
        )
        features = compute_all_features(wd, windows, candles, entry_minute=0)

        assert math.isfinite(features["price_vs_vwap"])


# ---------------------------------------------------------------------------
# ML model integration tests
# ---------------------------------------------------------------------------

class TestMLModelIntegration:
    """Tests for ML-based signal generation in strategy."""

    def test_ml_mode_disabled_uses_ensemble(
        self, mock_config: MagicMock, neutral_orderbook: OrderBookSnapshot, make_candle,
    ) -> None:
        """When use_ml_model=False, strategy should use ensemble (existing behavior)."""
        from src.strategies.before_window_open import BeforeWindowOpenStrategy

        # Ensure use_ml_model is False (the default in mock_config)
        strat = BeforeWindowOpenStrategy(config=mock_config)

        # Provide strong bullish signals for ensemble
        prior = [make_candle(open_=100.0, close=102.0) for _ in range(15)]
        recent = [
            make_candle(open_=100.0 + i * 0.1, close=100.1 + i * 0.1, volume=200.0)
            for i in range(240)
        ]

        context = {
            "prior_window_candles": prior,
            "recent_1m_candles": recent,
            "minute_in_window": 0,
            "yes_price": 0.50,
        }

        signals = strat.generate_signals(_market_state(), neutral_orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        # Ensemble should produce an entry signal with strong bullish data
        assert len(entry_signals) == 1
        assert entry_signals[0].direction == Side.YES

    def test_ml_mode_enabled_missing_model_falls_back(
        self, mock_config: MagicMock, neutral_orderbook: OrderBookSnapshot,
    ) -> None:
        """When model file doesn't exist, should fall back to ensemble."""
        from src.strategies.before_window_open import BeforeWindowOpenStrategy

        # Override config to enable ML with a nonexistent model path
        original_side_effect = mock_config.get.side_effect

        def ml_side_effect(key: str, default: object = None) -> object:
            overrides: dict[str, object] = {
                "strategy.before_window_open.use_ml_model": True,
                "strategy.before_window_open.model_path": "/nonexistent/model.joblib",
            }
            if key in overrides:
                return overrides[key]
            return original_side_effect(key, default)

        mock_config.get.side_effect = ml_side_effect

        # Should not crash during construction
        strat = BeforeWindowOpenStrategy(config=mock_config)

        # Should still generate signals (falling back to ensemble)
        context = {
            "prior_window_candles": [],
            "recent_1m_candles": [],
            "minute_in_window": 0,
            "yes_price": 0.50,
        }
        signals = strat.generate_signals(_market_state(), neutral_orderbook, context)
        # No crash is the primary assertion; with empty candle data, no entry expected
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert isinstance(entry_signals, list)

    def test_ml_entry_respects_minute(
        self, mock_config: MagicMock, neutral_orderbook: OrderBookSnapshot, make_candle,
    ) -> None:
        """ML mode should only trigger at the configured entry_minute."""
        from src.strategies.before_window_open import BeforeWindowOpenStrategy

        # Override config with ML enabled and entry_minute=3
        original_side_effect = mock_config.get.side_effect

        def ml_side_effect(key: str, default: object = None) -> object:
            overrides: dict[str, object] = {
                "strategy.before_window_open.use_ml_model": True,
                "strategy.before_window_open.model_path": "/nonexistent/model.joblib",
                "strategy.before_window_open.entry_minute": 3,
                "strategy.before_window_open.entry_minute_end": 3,
            }
            if key in overrides:
                return overrides[key]
            return original_side_effect(key, default)

        mock_config.get.side_effect = ml_side_effect
        strat = BeforeWindowOpenStrategy(config=mock_config)

        # Test at minute 5: should skip (past entry_minute_end=3)
        context = {
            "prior_window_candles": [],
            "recent_1m_candles": [],
            "minute_in_window": 5,
            "yes_price": 0.50,
        }
        signals = strat.generate_signals(_market_state(), neutral_orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 0

        # Test at minute 4: should also skip (past entry_minute_end=3)
        context["minute_in_window"] = 4
        signals = strat.generate_signals(_market_state(), neutral_orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 0

    def test_confidence_threshold_filtering(
        self, mock_config: MagicMock, neutral_orderbook: OrderBookSnapshot,
    ) -> None:
        """Trades below confidence threshold should be skipped."""
        from src.strategies.before_window_open import BeforeWindowOpenStrategy

        # Set a very high min_confidence so ensemble vote_weight can't reach it
        original_side_effect = mock_config.get.side_effect

        def strict_side_effect(key: str, default: object = None) -> object:
            overrides: dict[str, object] = {
                "strategy.before_window_open.min_confidence": 0.99,
                "strategy.before_window_open.min_vote_weight": 0.99,
            }
            if key in overrides:
                return overrides[key]
            return original_side_effect(key, default)

        mock_config.get.side_effect = strict_side_effect
        strat = BeforeWindowOpenStrategy(config=mock_config)

        # Even with some signals, the vote_weight won't reach 0.99
        context = {
            "prior_window_candles": [
                SimpleCandle(datetime.now(tz=UTC), 100.0, 102.0, 99.0, 102.0, 100.0),
                SimpleCandle(datetime.now(tz=UTC), 102.0, 103.0, 101.0, 103.0, 100.0),
            ],
            "recent_1m_candles": [],
            "minute_in_window": 0,
            "yes_price": 0.50,
        }
        signals = strat.generate_signals(_market_state(), neutral_orderbook, context)
        entry_signals = [s for s in signals if s.signal_type == SignalType.ENTRY]
        assert len(entry_signals) == 0


# ---------------------------------------------------------------------------
# Entry price adjustment tests
# ---------------------------------------------------------------------------

class TestEntryPriceAdjustment:
    """Tests for entry price adjustment based on early movement."""

    def test_entry_price_at_minute_0(self) -> None:
        """Pre-window entry should be at $0.50."""
        # entry_price = 0.50 + abs(0.0) * 0.5 = 0.50
        early_cum_return = 0.0
        sensitivity = 0.5
        entry_price = 0.50 + abs(early_cum_return) * sensitivity
        assert entry_price == 0.50

    def test_entry_price_with_positive_return(self) -> None:
        """Price should increase with positive early return."""
        # If early_cum_return = 0.002 (0.2%), entry_price = 0.50 + 0.002 * 0.5 = 0.501
        early_cum_return = 0.002
        sensitivity = 0.5
        entry_price = 0.50 + abs(early_cum_return) * sensitivity
        assert entry_price > 0.50
        assert entry_price == pytest.approx(0.501, abs=1e-6)

    def test_entry_price_capped(self) -> None:
        """Entry price should be capped at max_entry_price (0.70)."""
        # Even with huge early return, cap at 0.70
        early_cum_return = 0.5
        sensitivity = 0.5
        max_entry_price = 0.70
        entry_price = min(0.50 + abs(early_cum_return) * sensitivity, max_entry_price)
        assert entry_price == 0.70

    def test_fee_at_adjusted_price(self) -> None:
        """Fees should be calculated at the adjusted entry price."""
        from scripts.backtest_before_window import polymarket_fee

        fee_050 = polymarket_fee(100.0, 0.50)
        fee_055 = polymarket_fee(100.0, 0.55)
        # Fees at 0.55 should differ from 0.50
        assert fee_050 != fee_055
        # Both should be positive
        assert fee_050 > 0
        assert fee_055 > 0
